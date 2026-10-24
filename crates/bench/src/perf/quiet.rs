//! The quiet guard: before each measured part, wait (for a bounded time) until the machine's
//! other work settles, and log the load all through the run.
//!
//! The measure is the other processes' load: their threads running, waiting to run, or in
//! uninterruptible sleep, counted every 2 s with the suite's own process tree left out
//! ([`census`]), and averaged as the Linux 1-minute load average is, from the load average as the
//! run starts ([`Average`]). The suite's parts load the machine themselves (a page-in keeps 16
//! workers busy), and the load average remembers that for minutes; it also carries noise from
//! them, as it samples once every 5 s a count that swings between 0 and 16 within milliseconds.
//! Taking a model of the suite's share off the load average left that noise in (a quiet desktop
//! read as 3.5 one page-in and 1.6 the next), so the suite is left out of the count instead.
//! A part is measured once the others' load is below 1.5 ([`SETTLE_BELOW`]), and the run is
//! labelled loaded if it reaches 2 during a part ([`LOADED_AT`]): the gap keeps a release just
//! under the mark from being undone by the average's noise.

use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, ensure};
use thiserror::Error;

use crate::perf::census::{self, Pid};

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

/// The Linux load average's time constant for its 1-minute figure: an exponential moving
/// average with this time constant, in seconds.
const MINUTE: f64 = 60.0;

/// The other processes' load: their active threads ([`census::others_active`]), averaged as the
/// Linux 1-minute load average is, from a starting figure.
///
/// It is what the load average would read without the suite: the same average of the same
/// count, with the suite's own bursts (whose sampling noise the load average carries, and no
/// model of the suite's share can take off) left out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Average {
    /// The average.
    value: f64,
    /// When it was last brought up to date.
    at: Instant,
}

impl Average {
    /// An average reading `start` at `at`. The suite seeds it with the load average as the run
    /// starts, before any part: all of it is the other processes' then.
    pub(crate) const fn seeded(start: Load, at: Instant) -> Self {
        Self { value: start.0, at }
    }

    /// Folds in `active` threads counted at `at`, taken as held since the last count: the
    /// average moves towards `active` by `1 − e^−Δt/60`, as the load average does over `Δt`.
    /// A count from before the last one changes nothing.
    pub(crate) fn fold(&mut self, active: u32, at: Instant) {
        let Some(since) = at.checked_duration_since(self.at) else {
            return;
        };
        let keep = (-since.as_secs_f64() / MINUTE).exp();
        self.value = f64::from(active) + (self.value - f64::from(active)) * keep;
        self.at = at;
    }

    /// The average as a load.
    pub(crate) const fn load(&self) -> Load {
        Load(self.value)
    }
}

/// One look of the load log: when, the 1-minute load average, and the other processes' load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sample {
    /// When it was taken.
    pub(crate) at: Instant,
    /// The 1-minute load average then, the suite's own work included.
    pub(crate) load: Load,
    /// The other processes' load then ([`Average`]).
    pub(crate) others: Load,
}

/// The other processes' highest load among `samples`; `None` without samples.
///
/// Taken once a part has ended, this sees load that arrived while the part ran, which the guard
/// (looking only before it) can't.
pub(crate) fn others_peak(samples: &[Sample]) -> Option<Load> {
    samples
        .iter()
        .map(|sample| sample.others)
        .reduce(|a, b| if b > a { b } else { a })
}

/// The others' load the guard waits to fall below before a part.
///
/// Lower than [`LOADED_AT`] by 0.5, so a part the guard released isn't labelled loaded by noise
/// alone. The others' average varies by about ±0.15 (one standard deviation) from sampling at
/// 2 s, and on a quiet desktop it wandered by up to ~0.35 within a minute as other processes
/// woke and slept; 0.5 is over three standard deviations of the first and covers the second.
/// A run that released at 2.00 as an average seeded from a busy minute decayed, then read 2.03
/// and 2.10 in its next parts with the other processes at ~0.7, is what this prevents. It stays
/// above a quiet desktop's floor (~0.7 to 1.4 here), so the wait still ends.
pub(crate) const SETTLE_BELOW: Load = Load(1.5);

/// The others' load that labels the run loaded when reached during a part: the documented
/// meaning of a loaded run, the other processes' load at 2 or more.
pub(crate) const LOADED_AT: Load = Load(2.0);

/// What the guard does after one look at the load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// The others' load is below [`Quiet::settle_below`]: measure.
    Measure,
    /// It isn't yet, and there is time left: look again after [`Quiet::every`].
    Wait,
    /// It isn't, and the wait is used up: measure anyway, and label the run loaded.
    GiveUp,
}

/// How the guard waits and judges: it waits for the others' load to fall below `settle_below`,
/// looking every `every`, for at most `within`; a part during which it reaches `loaded_at`
/// labels the run loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Quiet {
    /// The load to fall below before a part ([`SETTLE_BELOW`]).
    pub(crate) settle_below: Load,
    /// The load that labels the run loaded when reached during a part ([`LOADED_AT`]).
    pub(crate) loaded_at: Load,
    /// The longest wait before a part.
    pub(crate) within: Duration,
    /// How often to look.
    pub(crate) every: Duration,
}

/// How a wait ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Settled {
    /// The others' load was below the threshold, after `waited`.
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
    /// The guard the suite uses: it waits for [`SETTLE_BELOW`], looking every 5 s (as often as
    /// the Linux load average updates), for at most `within`, and labels a run loaded at
    /// [`LOADED_AT`].
    pub(crate) const fn new(within: Duration) -> Self {
        Self {
            settle_below: SETTLE_BELOW,
            loaded_at: LOADED_AT,
            within,
            every: Duration::from_secs(5),
        }
    }

    /// How a part went, judged after it from the others' highest load during it (`peak`, see
    /// [`others_peak`]): loaded if it reached [`Quiet::loaded_at`], as if the guard had never
    /// settled.
    pub(crate) fn held(&self, peak: Load) -> Settled {
        if peak < self.loaded_at {
            Settled::Quiet {
                waited: Duration::ZERO,
                others: peak,
            }
        } else {
            Settled::Loaded { others: peak }
        }
    }

    /// What to do after a look found the others' load at `others`, having waited `waited`.
    pub(crate) fn decide(&self, others: Load, waited: Duration) -> Decision {
        if others < self.settle_below {
            Decision::Measure
        } else if waited + self.every <= self.within {
            Decision::Wait
        } else {
            Decision::GiveUp
        }
    }

    /// Waits until the others' load, as `look` reads it, falls below [`Quiet::settle_below`],
    /// sleeping with `sleep` between looks and saying what it does with `say`; gives up after
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
                            self.settle_below
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

/// Samples the load on a thread of its own every [`LoadLog::EVERY`] until stopped.
///
/// Each sample holds the load average and the other processes' load ([`Average`]), from a
/// census of their threads ([`census`]); one in [`LoadLog::SHOWN`] is printed to stderr.
#[derive(Debug)]
pub(crate) struct LoadLog {
    /// Tells the thread to stop.
    stop: Arc<AtomicBool>,
    /// The samples so far, oldest first; never empty.
    samples: Arc<Mutex<Vec<Sample>>>,
    /// The thread; it ends with the first error it meets, if any.
    thread: JoinHandle<Result<()>>,
}

/// Takes one sample: the load average now, and `others` brought up to date with a census of
/// the other processes than this one and its descendants.
///
/// # Errors
///
/// If `/proc` can't be read.
fn sample(others: &mut Average) -> Result<Sample> {
    let load = Load::now()?;
    let active = census::others_active(&census::scan(Path::new("/proc"))?, Pid::this());
    let at = Instant::now();
    others.fold(active, at);
    Ok(Sample {
        at,
        load,
        others: others.load(),
    })
}

impl LoadLog {
    /// How often the log samples.
    ///
    /// A census costs ~17 ms of one CPU, nearly all of it the Linux kernel producing ~2,500
    /// threads' `stat` lines (measured on a 32-thread desktop with ~680 processes), so every
    /// 2 s it takes under 1 % of one CPU from the parts it watches. That is often enough: the
    /// other processes' count varies by about 1 from one look to the next, which averaged over a
    /// minute leaves noise of about ±0.15, against ±1.3 in the load average from the suite's own
    /// bursts.
    const EVERY: Duration = Duration::from_secs(2);
    /// One sample in this many is printed: every 10 s.
    const SHOWN: u32 = 5;

    /// Starts logging: takes the first sample, which seeds the other processes' load with the
    /// load average (nothing of the suite runs yet), then samples on a thread.
    ///
    /// # Errors
    ///
    /// If the first sample fails, or the thread can't be started.
    pub(crate) fn start() -> Result<Self> {
        let start = Instant::now();
        let mut others = Average::seeded(Load::now()?, start);
        let first = sample(&mut others)?;
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(vec![first]));
        let (flag, shared) = (Arc::clone(&stop), Arc::clone(&samples));
        let show = move |taken: &Sample| {
            writeln!(
                std::io::stderr(),
                "perf: load {}, others {} at {:.0} s",
                taken.load,
                taken.others,
                taken.at.duration_since(start).as_secs_f64()
            )
        };
        show(&first)?;
        let thread = std::thread::Builder::new()
            .name("load-log".to_owned())
            .spawn(move || {
                let mut next = start + Self::EVERY;
                let mut count = 1_u32;
                while !flag.load(Ordering::Relaxed) {
                    if Instant::now() >= next {
                        let taken = sample(&mut others)?;
                        shared
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push(taken);
                        if count.is_multiple_of(Self::SHOWN) {
                            show(&taken)?;
                        }
                        count = count.wrapping_add(1);
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

    /// The latest sample: at most [`LoadLog::EVERY`] old while the log runs.
    ///
    /// # Errors
    ///
    /// If the log's thread has stopped (a sample failed), so the latest sample would be stale.
    pub(crate) fn latest(&self) -> Result<Sample> {
        ensure!(
            !self.thread.is_finished(),
            "the load log stopped early (its error comes when the run ends)"
        );
        self.samples
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .last()
            .copied()
            .ok_or_else(|| anyhow!("the load log has no sample"))
    }

    /// The samples taken from `since` on.
    pub(crate) fn since(&self, since: Instant) -> Vec<Sample> {
        self.samples
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|sample| sample.at >= since)
            .copied()
            .collect()
    }

    /// The highest load average sampled from `since` on (`None` if no sample was taken since).
    pub(crate) fn max_since(&self, since: Instant) -> Option<Load> {
        self.since(since)
            .into_iter()
            .map(|sample| sample.load)
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
    use std::cell::Cell;

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
        assert_eq!(load(2.5).to_string(), "2.50", "shown");
    }

    /// The guard measures below the threshold, waits above it while time is left, and gives up
    /// when a further wait would pass the bound.
    #[test]
    fn the_guard_measures_waits_or_gives_up() {
        let quiet = Quiet::new(Duration::from_secs(20));
        let s = Duration::from_secs;
        assert_eq!(quiet.decide(load(1.49), s(0)), Decision::Measure, "below");
        assert_eq!(
            quiet.decide(load(1.0), s(20)),
            Decision::Measure,
            "below, late"
        );
        assert_eq!(
            quiet.decide(load(1.5), s(0)),
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
        let (later, slept_later, said_later) = run(&[3.0, 2.5, 1.4]);
        assert_eq!(
            later,
            Settled::Quiet {
                waited: Duration::from_secs(10),
                others: load(1.4)
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

    /// The guard releases below the lower threshold and labels loaded at the higher one, and
    /// between them a part is neither held back nor labelled loaded.
    #[test]
    fn the_two_thresholds_differ() {
        let quiet = Quiet::new(Duration::from_secs(20));
        assert!(SETTLE_BELOW < LOADED_AT, "a gap between them");
        let between = load(1.8);
        assert_eq!(
            quiet.decide(between, Duration::ZERO),
            Decision::Wait,
            "waits"
        );
        assert!(
            matches!(quiet.held(between), Settled::Quiet { .. }),
            "but a part reading it stays quiet"
        );
        assert_eq!(
            quiet.held(LOADED_AT),
            Settled::Loaded { others: LOADED_AT },
            "loaded at 2"
        );
    }

    /// An average seeded from a busy minute decays, and the guard releases it only below 1.5,
    /// out of reach of the next part's noise.
    ///
    /// The seed is 11.87, as a run started straight after a test suite saw, and the other
    /// processes' real load is 0.6. The next part's noise (up to three standard deviations,
    /// +0.45) doesn't reach 2; with one threshold at 2 the same release was undone by +0.1.
    #[test]
    fn a_decaying_seed_releases_below_the_lower_threshold() {
        let quiet = Quiet::new(Duration::from_mins(15));
        let start = Instant::now();
        let (clock, tick) = (Cell::new(start), Cell::new(start + Duration::from_secs(2)));
        let average = Cell::new(Average::seeded(load(11.87), start));
        // The other processes' counts every 2 s, as the log takes them: 0.6 on average.
        let mut counts = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0].into_iter().cycle();
        let settled = quiet
            .settle(
                || Ok(average.get().load()),
                |wait| {
                    clock.set(clock.get() + wait);
                    while tick.get() <= clock.get() {
                        let mut next = average.get();
                        next.fold(counts.next().unwrap(), tick.get());
                        average.set(next);
                        tick.set(tick.get() + Duration::from_secs(2));
                    }
                },
                |_| Ok(()),
            )
            .unwrap();
        let Settled::Quiet { waited, others } = settled else {
            panic!("never released: {settled:?}");
        };
        assert!(others < SETTLE_BELOW, "released below 1.5: {others}");
        assert!(others.get() > 1.3, "as soon as it fell below: {others}");
        // 60 ln((11.87 − 0.6) / (1.5 − 0.6)) ≈ 152 s.
        assert!(
            (145..=165).contains(&waited.as_secs()),
            "after {} s",
            waited.as_secs()
        );
        let noisy = load(others.get() + 0.45);
        assert!(
            matches!(quiet.held(noisy), Settled::Quiet { .. }),
            "noise after the release: {noisy}"
        );
        let one_threshold = load(1.99 + 0.1);
        assert_eq!(
            quiet.held(one_threshold),
            Settled::Loaded {
                others: one_threshold
            },
            "the old edge"
        );
    }

    /// The others' load starts at its seed and moves towards each count as the load average
    /// would.
    ///
    /// A minute at 0 leaves `1/e` of the seed, a count held for long is reached, and a count
    /// from the past changes nothing.
    #[test]
    fn the_others_average_is_seeded_and_decays() {
        let start = Instant::now();
        let at = |secs| start + Duration::from_secs(secs);
        let mut average = Average::seeded(load(3.0), start);
        assert_eq!(average.load(), load(3.0), "the seed at once");
        average.fold(0, at(60));
        let a_minute = 3.0 / 1.0_f64.exp();
        assert!(
            (average.load().get() - a_minute).abs() < 1e-9,
            "{}",
            average.load()
        );
        let in_steps = (1..=60).fold(Average::seeded(load(3.0), start), |mut avg, s| {
            avg.fold(0, at(s));
            avg
        });
        assert!(
            (in_steps.load().get() - a_minute).abs() < 1e-9,
            "the same in steps of a second: {}",
            in_steps.load()
        );
        average.fold(5, at(60 + 600));
        assert!(
            (average.load().get() - 5.0).abs() < 1e-3,
            "ten minutes at 5: {}",
            average.load()
        );
        let before = average;
        average.fold(40, at(30));
        assert_eq!(average, before, "a count from the past");
        average.fold(40, at(660));
        assert_eq!(
            average.load(),
            before.load(),
            "no time passed: nothing moves"
        );
    }

    /// A sample at `secs` after `start`, with the load average at `load_avg` and the others'
    /// load at `others`.
    fn sample_at(start: Instant, secs: u64, load_avg: f64, others: f64) -> Sample {
        Sample {
            at: start + Duration::from_secs(secs),
            load: load(load_avg),
            others: load(others),
        }
    }

    /// A part is judged on the others' load alone: a load average swollen by the suite's own
    /// work doesn't label it loaded, and the others' load reaching 2 at any sample does.
    #[test]
    fn a_part_is_judged_on_the_others_load() {
        let quiet = Quiet::new(Duration::from_secs(300));
        let start = Instant::now();
        let own_only = [
            sample_at(start, 0, 1.2, 1.2),
            sample_at(start, 30, 7.6, 1.3),
            sample_at(start, 60, 8.2, 1.1),
        ];
        let peak = others_peak(&own_only).unwrap();
        assert_eq!(peak, load(1.3), "the others' highest");
        assert!(matches!(quiet.held(peak), Settled::Quiet { .. }), "quiet");
        let mut busier = own_only;
        if let Some(middle) = busier.get_mut(1) {
            middle.others = load(2.0);
        }
        let busy_peak = others_peak(&busier).unwrap();
        assert_eq!(
            quiet.held(busy_peak),
            Settled::Loaded { others: load(2.0) },
            "loaded at 2"
        );
        assert_eq!(others_peak(&[]), None, "no samples");
    }

    #[test]
    fn the_log_samples_and_stops() {
        let start = Instant::now();
        let log = LoadLog::start().unwrap();
        let first = log.latest().unwrap();
        assert!(first.at >= start, "sampled at once: {first:?}");
        assert_eq!(log.max_since(start), Some(first.load), "the load average");
        assert!(
            log.max_since(start + Duration::from_secs(3600)).is_none(),
            "nothing from the future"
        );
        std::thread::sleep(Duration::from_millis(2500));
        let later = log.latest().unwrap();
        assert!(later.at > first.at, "sampled again: {later:?}");
        assert!(log.since(start).len() >= 2, "two samples or more");
        log.stop().unwrap();
    }
}
