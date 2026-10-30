//! The GPU's clocks beside a scripted run, so its frames can be judged at full clocks.
//!
//! A frame that leaves the GPU mostly idle lets the driver drop to a low clock state (on an NVIDIA
//! desktop GPU, P3 or P5 at about a third of the boost clock), and every pass then runs two or
//! three times slower. So the spec's GPU budget is judged on frames at full clocks, and every
//! frame, at whatever clock the driver picked, is held to the display's refresh apart.
//!
//! `wgpu` doesn't report clocks. The suite samples them from outside the app instead, through
//! `nvidia-smi` in a child process ([`ClockSampler`]): every [`SAMPLE_MS`] its graphics and
//! memory clocks and power state, each line stamped by the system clock as it is read (the
//! tool writes a line a sample, unbuffered). The app's record places its frame loop's clock on
//! the same system clock (`clock_unix_us`), so each frame is joined to the sample nearest its
//! start ([`nearest`]). This keeps vendor code out of the app and out of the suite's build: on a
//! machine without the tool or an NVIDIA GPU the sampler doesn't start or reads nothing, and the
//! run's clocks are [`NoClocks`]; so are a run whose record names a GPU other than NVIDIA's
//! (lavapipe beside an NVIDIA card) and samples of several GPUs (which ran isn't known).
//!
//! **Full clocks** ([`FullClocks`]): a sample in power state P0 (where the GPU reports one) whose
//! graphics clock is at least [`FULL_SHARE`] of the highest sampled in the run. The clock is
//! what the passes' times scale with, and the run's own highest stands for the boost clock (the
//! open segment loads the GPU fully); P0 keeps a run that never left a low state from calling
//! that state full.

use std::fmt;
use std::io::{BufRead as _, BufReader};
use std::process::{Child, Command, Stdio};
use std::str::FromStr;
use std::thread::JoinHandle;
use std::time::SystemTime;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// How often the sampler reads the clocks, in ms.
pub(crate) const SAMPLE_MS: u32 = 50;

/// How far from a frame's start the sample joined to it may be, in ms: two samples' spacing
/// ([`SAMPLE_MS`]), so a late line still joins and a gap in the sampling leaves the frame's clock
/// unknown.
pub(crate) const JOIN_MS: f64 = 100.0;

/// The share of the run's highest graphics clock a sample needs to be at full clocks.
pub(crate) const FULL_SHARE: f64 = 0.9;

/// A power state: `P0`, the most performance, to `P15`, the least.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PState(u8);

impl PState {
    /// The deepest power state there is.
    const LAST: u8 = 15;
    /// P0, the most performance.
    pub(crate) const P0: Self = Self(0);
}

impl FromStr for PState {
    type Err = BadPState;

    /// `P0` to `P15`.
    fn from_str(s: &str) -> Result<Self, BadPState> {
        let number = s.strip_prefix('P').ok_or(BadPState)?;
        let number: Option<u8> = number.parse().ok();
        number
            .filter(|&n| n <= Self::LAST)
            .map(Self)
            .ok_or(BadPState)
    }
}

impl fmt::Display for PState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// A power state that isn't `P0` to `P15`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not a power state P0 to P15")]
pub(crate) struct BadPState;

/// The GPU's clocks at one moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ClockSample {
    /// When it was read, in ms since the Unix epoch by the system clock.
    pub(crate) at_unix_ms: f64,
    /// Which GPU: `nvidia-smi`'s index.
    pub(crate) gpu: u32,
    /// The graphics clock, in MHz.
    pub(crate) graphics_mhz: f64,
    /// The memory clock, in MHz, if reported.
    pub(crate) memory_mhz: Option<f64>,
    /// The power state, if reported.
    pub(crate) pstate: Option<PState>,
}

impl ClockSample {
    /// The sample in `line`, one of `nvidia-smi`'s with the fields [`ClockSampler`] asks for
    /// (`index, clocks.gr, clocks.mem, pstate`, no units), read at `at_unix_ms`; `None` for a
    /// line that isn't one, or has no graphics clock. A field the GPU doesn't report (`[N/A]`)
    /// is `None`.
    pub(crate) fn parse(line: &str, at_unix_ms: f64) -> Option<Self> {
        let mut fields = line.split(',').map(str::trim);
        let gpu: u32 = fields.next()?.parse().ok()?;
        let graphics_mhz: f64 = fields.next()?.parse().ok()?;
        let memory_mhz: Option<f64> = fields.next()?.parse().ok();
        let pstate: Option<PState> = fields.next()?.parse().ok();
        if fields.next().is_some() || !graphics_mhz.is_finite() {
            return None;
        }
        Some(Self {
            at_unix_ms,
            gpu,
            graphics_mhz,
            memory_mhz,
            pstate,
        })
    }
}

/// Why a run's frames have no clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum NoClocks {
    /// None were sampled: the record alone.
    #[default]
    NotSampled,
    /// `nvidia-smi` couldn't be started (no NVIDIA driver), and how it failed.
    NoSampler(std::io::ErrorKind),
    /// `nvidia-smi` gave no samples: no NVIDIA GPU, or none it can read.
    NoSamples,
    /// The samples are of several GPUs, and which ran the app isn't known.
    SeveralGpus,
    /// The app ran on a GPU other than NVIDIA's (lavapipe, say), whose clocks these aren't.
    OtherAdapter,
    /// The record doesn't place its clock on the system clock (no `clock_unix_us`).
    NoClockOrigin,
}

impl fmt::Display for NoClocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSampled => f.write_str("not sampled"),
            Self::NoSampler(kind) => write!(f, "nvidia-smi didn't start: {kind}"),
            Self::NoSamples => f.write_str("nvidia-smi gave no samples"),
            Self::SeveralGpus => f.write_str("several NVIDIA GPUs, and which ran isn't known"),
            Self::OtherAdapter => f.write_str("the app's GPU isn't NVIDIA's"),
            Self::NoClockOrigin => {
                f.write_str("the record doesn't place its clock (no clock_unix_us)")
            }
        }
    }
}

/// What counts as full clocks in one run (the module notes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FullClocks {
    /// The lowest graphics clock at full clocks, in MHz: [`FULL_SHARE`] of the run's highest.
    pub(crate) floor_mhz: f64,
}

impl FullClocks {
    /// Full clocks in the run `samples` were taken over.
    ///
    /// # Errors
    ///
    /// [`NoClocks::NoSamples`] for none, [`NoClocks::SeveralGpus`] if they are of more than one
    /// GPU.
    pub(crate) fn of(samples: &[ClockSample]) -> Result<Self, NoClocks> {
        let first = samples.first().ok_or(NoClocks::NoSamples)?;
        if samples.iter().any(|sample| sample.gpu != first.gpu) {
            return Err(NoClocks::SeveralGpus);
        }
        let highest = samples
            .iter()
            .map(|sample| sample.graphics_mhz)
            .fold(first.graphics_mhz, f64::max);
        Ok(Self {
            floor_mhz: FULL_SHARE * highest,
        })
    }

    /// Whether `sample` is at full clocks.
    pub(crate) fn holds(self, sample: &ClockSample) -> bool {
        sample.pstate.is_none_or(|state| state == PState::P0)
            && sample.graphics_mhz >= self.floor_mhz
    }
}

/// A run's clocks, as far as they were joined to its frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Clocks {
    /// Joined: what counts as full clocks in the run.
    Joined(FullClocks),
    /// Not known, and why.
    Unknown(NoClocks),
}

impl Default for Clocks {
    fn default() -> Self {
        Self::Unknown(NoClocks::default())
    }
}

/// Of `samples`, in time order, the one nearest `at_unix_ms`, if one is within [`JOIN_MS`].
pub(crate) fn nearest(samples: &[ClockSample], at_unix_ms: f64) -> Option<&ClockSample> {
    let after = samples.partition_point(|sample| sample.at_unix_ms < at_unix_ms);
    let before = after.checked_sub(1).and_then(|index| samples.get(index));
    let candidates = before.into_iter().chain(samples.get(after));
    let distance = |sample: &&ClockSample| (sample.at_unix_ms - at_unix_ms).abs();
    candidates
        .filter(|sample| distance(sample) <= JOIN_MS)
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
}

/// Now, in ms since the Unix epoch by the system clock (0 if the clock is set before it).
fn unix_ms_now() -> f64 {
    let since = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
    since.map_or(0.0, |d| d.as_secs_f64() * 1e3)
}

/// `nvidia-smi` sampling the GPU's clocks every [`SAMPLE_MS`] in a child process, until stopped.
/// Dropped without [`ClockSampler::stop`], it stops the child all the same.
#[derive(Debug)]
pub(crate) struct ClockSampler {
    /// The `nvidia-smi` process.
    child: Child,
    /// The thread reading its lines, which returns their samples at its output's end.
    reader: Option<JoinHandle<Vec<ClockSample>>>,
}

impl ClockSampler {
    /// Starts sampling.
    ///
    /// # Errors
    ///
    /// [`NoClocks::NoSampler`] if `nvidia-smi` can't be started.
    pub(crate) fn start() -> Result<Self, NoClocks> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,clocks.gr,clocks.mem,pstate",
                "--format=csv,noheader,nounits",
                "-lms",
                &SAMPLE_MS.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| NoClocks::NoSampler(error.kind()))?;
        let Some(stdout) = child.stdout.take() else {
            // Piped above, so never: but a child left running would sample forever.
            drop(child.kill());
            drop(child.wait());
            return Err(NoClocks::NoSampler(std::io::ErrorKind::BrokenPipe));
        };
        let reader = std::thread::spawn(move || {
            let lines = BufReader::new(stdout).lines().map_while(Result::ok);
            let samples = lines.filter_map(|line| ClockSample::parse(&line, unix_ms_now()));
            samples.collect()
        });
        Ok(Self {
            child,
            reader: Some(reader),
        })
    }

    /// Stops sampling; the samples taken, in time order.
    pub(crate) fn stop(mut self) -> Vec<ClockSample> {
        self.end();
        let reader = self.reader.take();
        reader
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default()
    }

    /// Ends the child, if it still runs, and reaps it: its output then ends, and the reader with
    /// it.
    fn end(&mut self) {
        // A child that has already exited can't be killed, and that's fine: it is reaped below.
        drop(self.child.kill());
        drop(self.child.wait());
    }
}

impl Drop for ClockSampler {
    fn drop(&mut self) {
        self.end();
    }
}
