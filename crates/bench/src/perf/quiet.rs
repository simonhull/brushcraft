//! The quiet guard: before each measured part, wait (for a bounded time) until the machine's
//! other work settles, and log the load all through the run.
//!
//! The measure is the 1-minute load average from `/proc/loadavg`, less the suite's own share of
//! it. The suite's parts load the machine themselves (a page-in keeps 16 workers busy), and the
//! load average remembers that for minutes: on a desktop whose floor is ~1.7, waiting for the raw
//! figure to fall below 2 after a part would take two minutes or never happen. So each part's
//! CPU time is recorded ([`Busy`]), and its share of the load average is worked out the way the
//! kernel averages ([`own_load`]): what is left is the other processes' load, which is what the
//! guard waits on.

use std::fmt;
use std::io::Write as _;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use thiserror::Error;

/// A load average: the mean number of tasks running or waiting to run (or in uninterruptible
/// sleep), finite and at least 0.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct Load(f64);

/// Why a value isn't a [`Load`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum BadLoad {
    /// Not a number, or not finite, or negative.
    #[error("{0:?} isn't a load average")]
    Value(String),
    /// `/proc/loadavg` without a first field.
    #[error("an empty /proc/loadavg")]
    Empty,
}

impl Load {
    /// `value` as a load.
    ///
    /// # Errors
    ///
    /// If it isn't finite or is negative.
    pub(crate) fn new(value: f64) -> Result<Self, BadLoad> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(BadLoad::Value(value.to_string()))
        }
    }

    /// The load as a number.
    pub(crate) const fn get(self) -> f64 {
        self.0
    }

    /// The 1-minute load average in `text`, the contents of `/proc/loadavg` (`1.52 1.40 1.31
    /// 2/1234 5678`: its first field).
    ///
    /// # Errors
    ///
    /// If the first field is missing or isn't a load.
    pub(crate) fn from_loadavg(text: &str) -> Result<Self, BadLoad> {
        text.split_whitespace()
            .next()
            .ok_or(BadLoad::Empty)?
            .parse()
    }

    /// The machine's 1-minute load average now.
    ///
    /// # Errors
    ///
    /// If `/proc/loadavg` can't be read or parsed.
    pub(crate) fn now() -> Result<Self> {
        let text = std::fs::read_to_string("/proc/loadavg").context("reading /proc/loadavg")?;
        Ok(Self::from_loadavg(&text)?)
    }

    /// This load less `other`, at least 0.
    #[must_use]
    pub(crate) fn less(self, other: Self) -> Self {
        Self((self.0 - other.0).max(0.0))
    }
}

impl FromStr for Load {
    type Err = BadLoad;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let value: f64 = text
            .parse()
            .map_err(|_not_a_number| BadLoad::Value(text.to_owned()))?;
        Self::new(value).map_err(|_out_of_range| BadLoad::Value(text.to_owned()))
    }
}

impl fmt::Display for Load {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

/// The kernel's averaging time for the 1-minute load: an exponential with this time constant.
const MINUTE: f64 = 60.0;

/// A stretch of the suite's own work: a part's child processes, from start to finish, and the
/// CPU time they took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Busy {
    /// When it started.
    pub(crate) start: Instant,
    /// When it ended.
    pub(crate) end: Instant,
    /// CPU time it took, every thread's.
    pub(crate) cpu: Duration,
}

/// The suite's own share of the 1-minute load average at `now`, from the stretches of work it
/// did (`busy`), each taken to keep `cpu / (end − start)` CPUs busy throughout.
///
/// The kernel's average is an exponential moving average with a one-minute time constant of the
/// tasks runnable, sampled every 5 s; taken as continuous, a load `u` held over `[s, e]` adds
/// `u · (e^−(now−e)/60 − e^−(now−s)/60)` to it at `now`. A CPU-bound thread is runnable while it
/// runs, so CPU time stands for runnable time; threads waiting for a CPU, or for the disk, count
/// in the kernel's figure and not here, so this is a slight underestimate.
pub(crate) fn own_load(busy: &[Busy], now: Instant) -> Load {
    let share: f64 = busy
        .iter()
        .filter_map(|b| {
            let length = b.end.checked_duration_since(b.start)?.as_secs_f64();
            if length <= 0.0 {
                return None;
            }
            let since_end = now.checked_duration_since(b.end)?.as_secs_f64();
            let since_start = since_end + length;
            let cpus = b.cpu.as_secs_f64() / length;
            Some(cpus * ((-since_end / MINUTE).exp() - (-since_start / MINUTE).exp()))
        })
        .sum();
    Load(share.max(0.0))
}

/// What the guard does after one look at the load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The others' load is below the threshold: measure.
    Measure,
    /// It isn't yet, and there is time left: look again after [`Quiet::every`].
    Wait,
    /// It isn't, and the wait is used up: measure anyway, and label the run loaded.
    GiveUp,
}

/// How the guard waits: for the others' load to fall below `below`, looking every `every`, for
/// at most `within`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Quiet {
    /// The load to fall below.
    pub(crate) below: Load,
    /// The longest wait before a part.
    pub(crate) within: Duration,
    /// How often to look.
    pub(crate) every: Duration,
}

/// How a wait ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Settled {
    /// The others' load fell below the threshold, after `waited`.
    Quiet {
        /// How long it took.
        waited: Duration,
        /// The others' load when it did.
        others: Load,
    },
    /// It never did within the bound.
    Loaded {
        /// The others' load at the last look.
        others: Load,
    },
}

impl Quiet {
    /// The guard the suite uses: below a load of 2, looking every 5 s (as often as the kernel
    /// updates the average), for at most `within`.
    pub(crate) const fn new(within: Duration) -> Self {
        Self {
            below: Load(2.0),
            within,
            every: Duration::from_secs(5),
        }
    }

    /// What to do after a look found the others' load at `others`, having waited `waited`.
    pub(crate) fn decide(&self, others: Load, waited: Duration) -> Decision {
        if others < self.below {
            Decision::Measure
        } else if waited + self.every <= self.within {
            Decision::Wait
        } else {
            Decision::GiveUp
        }
    }

    /// Waits until the others' load, as `look` reads it, falls below the threshold, sleeping
    /// with `sleep` between looks and saying what it does with `say`; gives up after
    /// [`Quiet::within`].
    ///
    /// # Errors
    ///
    /// If a look or `say` fails.
    pub(crate) fn settle(
        &self,
        mut look: impl FnMut() -> Result<Load>,
        mut sleep: impl FnMut(Duration),
        mut say: impl FnMut(&str) -> Result<()>,
    ) -> Result<Settled> {
        let mut waited = Duration::ZERO;
        loop {
            let others = look()?;
            match self.decide(others, waited) {
                Decision::Measure => {
                    if !waited.is_zero() {
                        say(&format!(
                            "quiet: other load {others} after {} s",
                            waited.as_secs()
                        ))?;
                    }
                    return Ok(Settled::Quiet { waited, others });
                }
                Decision::Wait => {
                    if waited.is_zero() {
                        say(&format!(
                            "quiet: other load {others}, waiting (up to {} s) for it to fall \
                             below {}",
                            self.within.as_secs(),
                            self.below
                        ))?;
                    }
                    sleep(self.every);
                    waited += self.every;
                }
                Decision::GiveUp => {
                    say(&format!(
                        "quiet: other load still {others} after {} s: measuring anyway, and the \
                         run is labelled loaded",
                        waited.as_secs()
                    ))?;
                    return Ok(Settled::Loaded { others });
                }
            }
        }
    }
}

/// Samples the load average on a thread of its own every [`LoadLog::EVERY`], printing each
/// sample to stderr, until stopped.
#[derive(Debug)]
pub(crate) struct LoadLog {
    /// Tells the thread to stop.
    stop: Arc<AtomicBool>,
    /// The samples so far: when, and the 1-minute load average.
    samples: Arc<Mutex<Vec<(Instant, Load)>>>,
    /// The thread; it ends with the first error it meets, if any.
    thread: JoinHandle<Result<()>>,
}

impl LoadLog {
    /// How often the log samples: as often as the kernel updates the average.
    const EVERY: Duration = Duration::from_secs(5);

    /// Starts logging.
    ///
    /// # Errors
    ///
    /// If the thread can't be started.
    pub(crate) fn start() -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let (flag, shared) = (Arc::clone(&stop), Arc::clone(&samples));
        let thread = std::thread::Builder::new()
            .name("load-log".to_owned())
            .spawn(move || {
                let start = Instant::now();
                let mut next = start;
                while !flag.load(Ordering::Relaxed) {
                    let now = Instant::now();
                    if now >= next {
                        let load = Load::now()?;
                        shared
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push((now, load));
                        writeln!(
                            std::io::stderr(),
                            "perf: load {load} at {:.0} s",
                            now.duration_since(start).as_secs_f64()
                        )?;
                        next += Self::EVERY;
                    }
                    // Short naps, so stopping is prompt.
                    std::thread::sleep(Duration::from_millis(100));
                }
                Ok(())
            })
            .context("starting the load log")?;
        Ok(Self {
            stop,
            samples,
            thread,
        })
    }

    /// The highest load sampled from `since` on (`None` if no sample was taken since).
    pub(crate) fn max_since(&self, since: Instant) -> Option<Load> {
        self.samples
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|&&(at, _)| at >= since)
            .map(|&(_, load)| load)
            .reduce(|a, b| if b > a { b } else { a })
    }

    /// Stops logging.
    ///
    /// # Errors
    ///
    /// If a sample failed, or the thread panicked.
    pub(crate) fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread
            .join()
            .map_err(|panic| anyhow!("the load log panicked: {panic:?}"))?
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A load of `x`.
    fn load(x: f64) -> Load {
        Load::new(x).unwrap()
    }

    #[test]
    fn loads_parse_from_proc_loadavg() {
        assert_eq!(
            Load::from_loadavg("1.52 1.40 1.31 2/1234 5678\n"),
            Ok(load(1.52)),
            "the first field"
        );
        assert_eq!(Load::from_loadavg(""), Err(BadLoad::Empty), "empty");
        for bad in ["x 1 1", "-1 1 1", "inf 1 1", "NaN 1 1"] {
            assert!(Load::from_loadavg(bad).is_err(), "{bad:?}");
        }
        assert!(Load::now().is_ok(), "this machine's");
        assert_eq!(load(1.0).less(load(3.0)), load(0.0), "never below 0");
        assert_eq!(load(2.5).to_string(), "2.50", "shown");
    }

    /// The guard measures below the threshold, waits above it while time is left, and gives up
    /// when a further wait would pass the bound.
    #[test]
    fn the_guard_measures_waits_or_gives_up() {
        let quiet = Quiet::new(Duration::from_secs(20));
        let s = Duration::from_secs;
        assert_eq!(quiet.decide(load(1.99), s(0)), Decision::Measure, "below");
        assert_eq!(
            quiet.decide(load(1.0), s(20)),
            Decision::Measure,
            "below, late"
        );
        assert_eq!(
            quiet.decide(load(2.0), s(0)),
            Decision::Wait,
            "at the threshold"
        );
        assert_eq!(
            quiet.decide(load(5.0), s(15)),
            Decision::Wait,
            "a last wait fits"
        );
        assert_eq!(
            quiet.decide(load(5.0), s(16)),
            Decision::GiveUp,
            "no wait fits"
        );
        assert_eq!(
            Quiet::new(Duration::ZERO).decide(load(5.0), s(0)),
            Decision::GiveUp,
            "no waiting allowed"
        );
    }

    /// Settling looks until the load falls, sleeping between looks, or gives up at the bound:
    /// driven by a scripted load and a fake sleep, so no time passes.
    #[test]
    fn settling_follows_the_decisions() {
        let quiet = Quiet::new(Duration::from_secs(20));
        let run = |loads: &[f64]| {
            let mut script = loads.iter().copied();
            let mut slept = Duration::ZERO;
            let mut said = Vec::new();
            let settled = quiet
                .settle(
                    || Ok(load(script.next().unwrap())),
                    |d| slept += d,
                    |line| {
                        said.push(line.to_owned());
                        Ok(())
                    },
                )
                .unwrap();
            (settled, slept, said)
        };
        let (settled, slept, said) = run(&[1.0]);
        assert_eq!(
            settled,
            Settled::Quiet {
                waited: Duration::ZERO,
                others: load(1.0)
            },
            "quiet at once"
        );
        assert_eq!((slept, said.len()), (Duration::ZERO, 0), "silently");
        let (later, slept_later, said_later) = run(&[3.0, 2.5, 1.5]);
        assert_eq!(
            later,
            Settled::Quiet {
                waited: Duration::from_secs(10),
                others: load(1.5)
            },
            "after two waits"
        );
        assert_eq!(slept_later, Duration::from_secs(10), "slept between looks");
        assert_eq!(
            said_later.len(),
            2,
            "said it waits, then that it settled: {said_later:?}"
        );
        let (never, slept_long, said_long) = run(&[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
        assert_eq!(never, Settled::Loaded { others: load(3.0) }, "never quiet");
        assert_eq!(
            slept_long,
            Duration::from_secs(20),
            "waited the bound, no more"
        );
        assert!(
            said_long.last().unwrap().contains("labelled loaded"),
            "{said_long:?}"
        );
    }

    /// The suite's own share decays as the kernel's average does: a minute of 16 busy CPUs
    /// leaves `16 (1 − 1/e)` just after, a minute later `1/e` of that, and nothing before it
    /// started.
    #[test]
    fn the_suites_own_load_decays_as_the_kernels_average() {
        let start = Instant::now();
        let busy = Busy {
            start,
            end: start + Duration::from_mins(1),
            cpu: Duration::from_mins(16),
        };
        let at = |secs| own_load(&[busy], start + Duration::from_secs(secs)).get();
        let just_after = 16.0 * (1.0 - (-1.0_f64).exp());
        assert!((at(60) - just_after).abs() < 1e-9, "{}", at(60));
        assert!(
            (at(120) - just_after / 1.0_f64.exp()).abs() < 1e-9,
            "{}",
            at(120)
        );
        assert!(at(600) < 0.01, "gone in ten minutes: {}", at(600));
        assert_eq!(own_load(&[busy], start), load(0.0), "not before it ended");
        let instant = Busy {
            start,
            end: start,
            cpu: Duration::ZERO,
        };
        assert_eq!(
            own_load(&[instant], start + Duration::from_secs(1)),
            load(0.0),
            "an empty stretch adds nothing"
        );
        let two = own_load(&[busy, busy], start + Duration::from_secs(90)).get();
        assert!((two - 2.0 * at(90)).abs() < 1e-9, "shares add up");
    }

    #[test]
    fn the_log_samples_and_stops() {
        let start = Instant::now();
        let log = LoadLog::start().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(log.max_since(start).is_some(), "sampled at once");
        assert!(
            log.max_since(start + Duration::from_secs(3600)).is_none(),
            "nothing from the future"
        );
        log.stop().unwrap();
    }
}
