//! The renderer's metrics: the app's scripted run (`brushcraft --script`, its record written
//! with `--record`) read back and made into the suite's metrics.
//!
//! The app is a binary of its own above `render`, so the suite runs it as it runs every part: a
//! child process (`brushcraft`, built beside this binary), here offscreen. This crate stays free
//! of the GPU: it links neither `app` nor `render`, and reads the run's record, JSON lines in
//! its own records' style. A run on a copy of the project whose display layer was saved first
//! (`display-save`), since the spec's open-to-correct is "from the stored display layer and
//! bounds".
//!
//! From each run's record ([`Recorded`], [`derive()`]):
//!
//! - **orbit frame time**, p50 and p99, on the GPU (the whole frame, timestamp to timestamp) and on
//!   the CPU (the frame's latency path: the owner thread's input, brush, feed and uploads, then the
//!   render thread's encode and submit), over the orbit segment's frames;
//! - **tiles over budget** during the orbit: the most tiles in view any orbit frame left over 1 px
//!   (the level choice's count; the spec's target is 0 on desktop);
//! - **spin catch-up**: from the flick's end until the first frame shown with nothing wanted
//!   (residency) and nothing requested (its level choice) (spec: ≤ 100 ms);
//! - **open to correct**: from the project beginning to open until the first frame of the open
//!   segment shown with no tile over 1 px and nothing requested (spec: ≤ 1 s on desktop).
//!
//! A run whose record lacks one (no timestamp queries, no frame correct within its segment)
//! gives no sample of it, and the part says so. With no GPU (the app exits 69 when no adapter
//! matches), the part doesn't run: "not run: no adapter".

use std::path::Path;

use anyhow::{Context as _, Result, anyhow, ensure};

use crate::perf::json::{Json, Object};
use crate::perf::metric::{Metric, MetricName, Unit};
use crate::perf::project::{Scale, Source, WorkDir};
use crate::perf::suite::{AppExit, Runner, Samples};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The record's segment the orbit metrics come from.
const ORBIT: &str = "orbit";
/// The record's segment after whose end the catch-up is timed.
const FLICK: &str = "flick";
/// The record's segment open-to-correct must be reached in.
const OPEN: &str = "open";

/// The spec's targets, judged on the benchmark project only: tiles over budget during the
/// orbit, spin catch-up and open to correct, in the metrics' units.
const TARGETS: [(&str, f64); 3] = [
    ("render.orbit_over_budget", 0.0),
    ("render.spin_catch_up_ms", 100.0),
    ("render.open_to_correct_ms", 1000.0),
];

/// How a scale's render part runs the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Setting {
    /// Runs, each one sample of every metric.
    pub(crate) runs: usize,
    /// `--adapter`: a hardware GPU for the measured scales, lavapipe for the tiny check.
    pub(crate) adapter: &'static str,
    /// `--size`: 4K for the measured scales, the size the spec's GPU budget is set at.
    pub(crate) size: &'static str,
    /// `--script-speed`: real time for the measured scales.
    pub(crate) speed: &'static str,
}

impl Setting {
    /// The render part's setting at `scale`.
    pub(crate) const fn of(scale: Scale) -> Self {
        match scale {
            Scale::Tiny => Self {
                runs: 1,
                adapter: "software",
                size: "320x200",
                speed: "4",
            },
            Scale::Medium | Scale::Full => Self {
                runs: 3,
                adapter: "hardware",
                size: "3840x2160",
                speed: "1",
            },
        }
    }
}

/// One frame of a recorded run, as much of it as the metrics read.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Frame {
    /// Its start on the frame loop's clock, in ms.
    pub(crate) time_ms: f64,
    /// When it was shown on that clock, in ms, if it was.
    pub(crate) presented_ms: Option<f64>,
    /// The script's segment it fell in.
    pub(crate) segment: Option<String>,
    /// Its GPU time, in ms, if timed.
    pub(crate) gpu_ms: Option<f64>,
    /// Its CPU time on the latency path, in ms, if known.
    pub(crate) cpu_ms: Option<f64>,
    /// Whether its cut came back: the counts below are its own.
    pub(crate) drawn: bool,
    /// Tiles in view its level choice left over 1 px.
    pub(crate) over_budget: Option<f64>,
    /// Tiles its level choice requested.
    pub(crate) requested: Option<f64>,
    /// Tiles residency still wanted as it was staged.
    pub(crate) wanted: Option<f64>,
}

impl Frame {
    /// When it reached the screen, else when it started, in ms.
    fn shown_ms(&self) -> f64 {
        self.presented_ms.unwrap_or(self.time_ms)
    }

    /// Whether it drew nothing short of its chosen levels: its cut came back with no tile over
    /// 1 px and nothing requested.
    fn correct(&self) -> bool {
        self.drawn && self.over_budget == Some(0.0) && self.requested == Some(0.0)
    }

    /// Whether nothing was left to arrive: its cut came back requesting nothing, and residency
    /// wanted nothing as it was staged.
    fn caught_up(&self) -> bool {
        self.drawn && self.requested == Some(0.0) && self.wanted == Some(0.0)
    }
}

/// A recorded run, as much of it as the metrics read.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Recorded {
    /// From the project beginning to open to the frame loop's clock starting, in ms.
    pub(crate) open_ms: f64,
    /// Each segment's name, start and end on the frame loop's clock, in ms.
    pub(crate) segments: Vec<(String, f64, f64)>,
    /// Every frame, in order.
    pub(crate) frames: Vec<Frame>,
}

/// `object`'s number `name`; `None` for `null` or a field that isn't there.
fn number(object: &Object, name: &str) -> Option<f64> {
    object.get(name).and_then(Json::as_number)
}

/// `object`'s number `name`, which must be there.
///
/// # Errors
///
/// If it isn't a number.
fn required(object: &Object, name: &str) -> Result<f64> {
    number(object, name).ok_or_else(|| anyhow!("no number {name:?}"))
}

impl Recorded {
    /// The run recorded in `text`, the app's `--record` lines.
    ///
    /// # Errors
    ///
    /// If a line isn't a JSON object, the `run` line is missing, or a line lacks a field every
    /// line of its kind has.
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let mut recorded = Self::default();
        let mut run = false;
        let lines = text.lines().filter(|line| !line.trim().is_empty());
        for (index, line) in lines.enumerate() {
            let object =
                Object::parse(line).with_context(|| format!("record line {}", index + 1))?;
            run |= recorded
                .take(&object)
                .with_context(|| format!("record line {}: {line}", index + 1))?;
        }
        ensure!(run, "the record has no run line");
        Ok(recorded)
    }

    /// Takes in one line of the record, `object`; whether it was the `run` line.
    ///
    /// # Errors
    ///
    /// If it lacks a field every line of its kind has.
    fn take(&mut self, object: &Object) -> Result<bool> {
        match object.get("result").and_then(Json::as_str) {
            Some("run") => {
                self.open_ms = required(object, "open_ms")?;
                return Ok(true);
            }
            Some("segment") => {
                let name = object.get("segment").and_then(Json::as_str);
                let name = name.ok_or_else(|| anyhow!("a segment with no name"))?;
                let start = required(object, "start_ms")?;
                let end = required(object, "end_ms")?;
                self.segments.push((name.to_owned(), start, end));
            }
            Some("frame") => self.frames.push(Frame {
                time_ms: required(object, "time_ms")?,
                presented_ms: number(object, "presented_ms"),
                segment: object
                    .get("segment")
                    .and_then(Json::as_str)
                    .map(str::to_owned),
                gpu_ms: number(object, "gpu_ms"),
                cpu_ms: number(object, "cpu_ms"),
                drawn: object.get("drawn").and_then(Json::as_bool) == Some(true),
                over_budget: number(object, "over_budget"),
                requested: number(object, "requested"),
                wanted: number(object, "wanted"),
            }),
            _ => {}
        }
        Ok(false)
    }

    /// When segment `name` ends, in ms, if the run had it.
    fn end_of(&self, name: &str) -> Option<f64> {
        self.segments
            .iter()
            .find(|(segment, _, _)| segment == name)
            .map(|&(_, _, end)| end)
    }

    /// The frames in segment `name`.
    fn in_segment<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Frame> {
        self.frames
            .iter()
            .filter(move |frame| frame.segment.as_deref() == Some(name))
    }
}

/// The `q`th percentile (0 to 100) of `values`, by the nearest rank above (as the app's phase
/// summary takes it); `None` for none.
fn percentile(values: &[f64], q: usize) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len().checked_sub(1)? * q).div_ceil(100);
    sorted.get(index).copied()
}

/// One run's render metrics; `None` for one its record can't give.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct RenderMetrics {
    /// The orbit's GPU frame time, p50 and p99, in ms.
    pub(crate) orbit_gpu: Option<(f64, f64)>,
    /// The orbit's CPU frame time, p50 and p99, in ms.
    pub(crate) orbit_cpu: Option<(f64, f64)>,
    /// The most tiles over budget in any orbit frame whose cut came back.
    pub(crate) orbit_over_budget: Option<f64>,
    /// From the flick's end to the first frame caught up, in ms.
    pub(crate) spin_catch_up_ms: Option<f64>,
    /// From the project beginning to open to the first correct frame of the open segment, in ms.
    pub(crate) open_to_correct_ms: Option<f64>,
}

/// The render metrics of `recorded` (the module notes say how each is taken).
pub(crate) fn derive(recorded: &Recorded) -> RenderMetrics {
    let spread = |values: Vec<f64>| Some((percentile(&values, 50)?, percentile(&values, 99)?));
    let gpu: Vec<f64> = recorded
        .in_segment(ORBIT)
        .filter_map(|f| f.gpu_ms)
        .collect();
    let cpu: Vec<f64> = recorded
        .in_segment(ORBIT)
        .filter_map(|f| f.cpu_ms)
        .collect();
    let over = recorded
        .in_segment(ORBIT)
        .filter(|frame| frame.drawn)
        .filter_map(|frame| frame.over_budget);
    let orbit_over_budget = over.reduce(f64::max);
    let spin_catch_up_ms = recorded.end_of(FLICK).and_then(|flick_end| {
        let still = recorded.frames.iter().filter(|f| f.time_ms >= flick_end);
        let first = still.into_iter().find(|frame| frame.caught_up())?;
        Some((first.shown_ms() - flick_end).max(0.0))
    });
    let open_to_correct_ms = recorded
        .in_segment(OPEN)
        .find(|frame| frame.correct())
        .map(|frame| recorded.open_ms + frame.shown_ms());
    RenderMetrics {
        orbit_gpu: spread(gpu),
        orbit_cpu: spread(cpu),
        orbit_over_budget,
        spin_catch_up_ms,
        open_to_correct_ms,
    }
}

impl RenderMetrics {
    /// Each metric as `(name after the scale, unit, value)`, `None` where the run gave none.
    fn named(&self) -> [(&'static str, Unit, Option<f64>); 7] {
        [
            (
                "render.orbit_gpu_p50_ms",
                Unit::Ms,
                self.orbit_gpu.map(|g| g.0),
            ),
            (
                "render.orbit_gpu_p99_ms",
                Unit::Ms,
                self.orbit_gpu.map(|g| g.1),
            ),
            (
                "render.orbit_cpu_p50_ms",
                Unit::Ms,
                self.orbit_cpu.map(|c| c.0),
            ),
            (
                "render.orbit_cpu_p99_ms",
                Unit::Ms,
                self.orbit_cpu.map(|c| c.1),
            ),
            (
                "render.orbit_over_budget",
                Unit::Tiles,
                self.orbit_over_budget,
            ),
            ("render.spin_catch_up_ms", Unit::Ms, self.spin_catch_up_ms),
            (
                "render.open_to_correct_ms",
                Unit::Ms,
                self.open_to_correct_ms,
            ),
        ]
    }

    /// Adds this run's metrics to `samples` under `scale`'s names; returns the names of those it
    /// had none of.
    ///
    /// # Errors
    ///
    /// If a name isn't a metric name, or a unit clashes.
    fn sample(&self, scale: Scale, samples: &mut Samples) -> Result<Vec<&'static str>> {
        let mut missing = Vec::new();
        for (name, unit, value) in self.named() {
            match value {
                Some(value) => samples.push(MetricName::join(scale.name(), name)?, unit, value)?,
                None => missing.push(name),
            }
        }
        Ok(missing)
    }
}

/// What the render part came to.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RenderPart {
    /// Measured: the metrics, and a note for each run that lacked some.
    Measured {
        /// The metrics.
        metrics: Vec<Metric>,
        /// What runs lacked, one line each.
        notes: Vec<String>,
    },
    /// Not run, and why: "no adapter".
    NotRun(String),
}

/// The render part at `scale` on a copy of `project` made in `work`.
///
/// The copy's display layer is saved by `display-save`, then the app's script runs
/// [`Setting::runs`] times on it, each run's record made into one sample of each metric. The
/// copy is removed after.
///
/// # Errors
///
/// If the app binary is missing, the copy, the save or a run fails (but for finding no adapter),
/// or a record doesn't read.
pub(crate) fn render_part(
    runner: &Runner,
    work: &WorkDir,
    project: &Path,
    scale: Scale,
) -> Result<RenderPart> {
    let setting = Setting::of(scale);
    let (copy, _) = work.copy(&Source::open(project)?, "render")?;
    let dir = copy.parent().unwrap_or(&copy).to_owned();
    let measured = measure(runner, &copy, &dir, scale, setting);
    std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
    measured
}

/// [`render_part`]'s work on `copy`, its records and the app's state in `dir`.
///
/// # Errors
///
/// As [`render_part`].
fn measure(
    runner: &Runner,
    copy: &Path,
    dir: &Path,
    scale: Scale,
    setting: Setting,
) -> Result<RenderPart> {
    let level = scale.level().to_string();
    let copy_arg = copy.as_os_str().to_owned();
    runner.bench(&[
        "display-save".into(),
        "--project".into(),
        copy_arg.clone(),
        "--level".into(),
        level.clone().into(),
    ])?;
    let mut samples = Samples::default();
    let mut notes = Vec::new();
    for run in 0..setting.runs {
        let record = dir.join(format!("render-{run}.jsonl"));
        let args = [
            copy_arg.clone(),
            "--offscreen".into(),
            "--script".into(),
            "--script-speed".into(),
            setting.speed.into(),
            "--record".into(),
            record.as_os_str().to_owned(),
            "--adapter".into(),
            setting.adapter.into(),
            "--level".into(),
            level.clone().into(),
            "--size".into(),
            setting.size.into(),
        ];
        match runner.app(&args, &dir.join("state"))? {
            AppExit::NoAdapter => return Ok(RenderPart::NotRun("no adapter".to_owned())),
            AppExit::Done => {}
        }
        let text = std::fs::read_to_string(&record)
            .with_context(|| format!("reading {}", record.display()))?;
        let recorded = Recorded::parse(&text).with_context(|| format!("{}", record.display()))?;
        let missing = derive(&recorded).sample(scale, &mut samples)?;
        if !missing.is_empty() {
            notes.push(format!("run {}: no {}", run + 1, missing.join(", ")));
        }
    }
    let metrics = with_targets(samples.measure()?, scale);
    Ok(RenderPart::Measured { metrics, notes })
}

/// `metrics` of `scale`, those the spec sets a target for given it as their ceiling on the
/// benchmark project ([`TARGETS`]); the other scales' are only compared with their history.
fn with_targets(metrics: Vec<Metric>, scale: Scale) -> Vec<Metric> {
    let ceiling = |metric: &Metric| {
        let target = TARGETS
            .iter()
            .find(|(name, _)| metric.name.as_str().ends_with(name));
        target
            .filter(|_| scale == Scale::Full)
            .map(|&(_, ceiling)| ceiling)
    };
    metrics
        .into_iter()
        .map(|metric| Metric {
            ceiling: ceiling(&metric),
            ..metric
        })
        .collect()
}
