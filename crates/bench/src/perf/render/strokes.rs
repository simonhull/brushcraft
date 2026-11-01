//! The pen's metrics, from the app's strokes script read back.
//!
//! The script (`brushcraft --script --script-path strokes`, its record written with `--record`)
//! draws four strokes (`stroke_near`, `stroke_far`, `stroke_long`, `stroke_heavy`, the record's
//! segments); each
//! gives, over its segment's frames:
//!
//! - **sample to present**, p99: from the oldest pen sample whose dabs a frame applied to its
//!   present being submitted, over the frames that applied dabs (the M1 spec's pen latency: ≤ 8 ms
//!   at p99 below the footprint limit F, the ceiling on the benchmark project);
//! - **frame start to present submitted**, p99, over the same frames: the frame's own share;
//! - **brush**, p99, over the same frames: the frame's brush phase (the work between frames, whose
//!   edits it shows, is the record's `between_ms`, not counted here);
//! - **most dabs carried**: the most any frame left queued for the next. A brush that keeps up
//!   carries at most a few samples' worth for a frame or two; a queue that grows over the stroke
//!   passes the ceiling ([`MOST_CARRIED`]) within its first tens of milliseconds;
//! - **drain frames**: from pen-up (the frame after the last whose stroke was still open) to the
//!   first frame carrying none, counted inclusively (ceiling 2, the brush-engine plan's "drains
//!   within two frames of pen-up"); never drained within the segment is not reached;
//! - **frames above F**: frames with a dab above the footprint limit, where the latency target
//!   doesn't hold;
//! - **prepare**, p99, over every frame of the segment: the brush's work after the hand-off
//!   (warming the surface under a hovering pen), off the latency path.
//!
//! A run whose record lacks a segment (an older app) gives no sample of its metrics.

use crate::perf::metric::Unit;
use crate::perf::render::{Frame, Recorded, Value, percentile};

/// The strokes script's stroke segments, in order.
pub(crate) const SEGMENTS: [&str; 4] = ["stroke_near", "stroke_far", "stroke_long", "stroke_heavy"];

/// The ceiling on the most dabs carried, on the benchmark project: a few samples' worth.
///
/// A fast stroke places two or three dabs a sample, and a brush keeping up shows one or two
/// frames carrying a few as a dab runs long; a growing queue passes it within the stroke's first
/// tens of milliseconds.
pub(crate) const MOST_CARRIED: f64 = 8.0;

/// Each metric's name after the scale, by segment (as [`SEGMENTS`]) and then in
/// [`StrokeMetrics::named`]'s order.
const NAMES: [[&str; 7]; 4] = [
    [
        "render.stroke_near.sample_to_present_p99_ms",
        "render.stroke_near.start_to_submitted_p99_ms",
        "render.stroke_near.brush_p99_ms",
        "render.stroke_near.dabs_carried_max",
        "render.stroke_near.drain_frames",
        "render.stroke_near.frames_above_f",
        "render.stroke_near.prepare_p99_ms",
    ],
    [
        "render.stroke_far.sample_to_present_p99_ms",
        "render.stroke_far.start_to_submitted_p99_ms",
        "render.stroke_far.brush_p99_ms",
        "render.stroke_far.dabs_carried_max",
        "render.stroke_far.drain_frames",
        "render.stroke_far.frames_above_f",
        "render.stroke_far.prepare_p99_ms",
    ],
    [
        "render.stroke_long.sample_to_present_p99_ms",
        "render.stroke_long.start_to_submitted_p99_ms",
        "render.stroke_long.brush_p99_ms",
        "render.stroke_long.dabs_carried_max",
        "render.stroke_long.drain_frames",
        "render.stroke_long.frames_above_f",
        "render.stroke_long.prepare_p99_ms",
    ],
    [
        "render.stroke_heavy.sample_to_present_p99_ms",
        "render.stroke_heavy.start_to_submitted_p99_ms",
        "render.stroke_heavy.brush_p99_ms",
        "render.stroke_heavy.dabs_carried_max",
        "render.stroke_heavy.drain_frames",
        "render.stroke_heavy.frames_above_f",
        "render.stroke_heavy.prepare_p99_ms",
    ],
];

/// The spec's targets on the pen's metrics, by the end of their names, in their units: judged on
/// the benchmark project only, as the renderer's ([`super::TARGETS`]).
pub(crate) const TARGETS: [(&str, f64); 3] = [
    ("sample_to_present_p99_ms", 8.0),
    ("dabs_carried_max", MOST_CARRIED),
    ("drain_frames", 2.0),
];

/// Whether the queue drained after pen-up, and in how many frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Drain {
    /// In this many frames, counted from the first after pen-up to the first carrying none.
    Frames(f64),
    /// Not within the segment.
    Not,
}

/// One stroke's metrics from one run; `None` for one the record can't give.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct StrokeMetrics {
    /// Sample to present submitted, p99, in ms.
    pub(crate) sample_to_present_p99: Option<f64>,
    /// Frame start to present submitted, p99, in ms.
    pub(crate) start_to_submitted_p99: Option<f64>,
    /// The frame's brush phase, p99, in ms.
    pub(crate) brush_p99: Option<f64>,
    /// The most dabs any frame carried.
    pub(crate) most_carried: Option<f64>,
    /// Frames from pen-up until none is carried.
    pub(crate) drain: Option<Drain>,
    /// Frames with a dab above F.
    pub(crate) above_f: Option<f64>,
    /// The brush's preparation after the hand-off, p99, in ms.
    pub(crate) prepare_p99: Option<f64>,
}

/// Each stroke's metrics from `recorded`, in [`SEGMENTS`]' order.
pub(crate) fn derive(recorded: &Recorded) -> [StrokeMetrics; 4] {
    SEGMENTS.map(|segment| {
        let frames: Vec<&Frame> = recorded.in_segment(segment).collect();
        stroke(&frames)
    })
}

/// One stroke's metrics from its segment's `frames`, in order.
fn stroke(frames: &[&Frame]) -> StrokeMetrics {
    if frames.is_empty() {
        return StrokeMetrics::default();
    }
    let brushed = frames
        .iter()
        .filter(|frame| frame.dabs_applied.is_some_and(|dabs| dabs > 0.0));
    let brushed: Vec<&&Frame> = brushed.collect();
    let latency: Vec<f64> = brushed
        .iter()
        .filter_map(|f| f.sample_to_present_ms)
        .collect();
    let start_to_submitted: Vec<f64> = brushed
        .iter()
        .filter_map(|f| Some(f.submitted_ms? - f.time_ms))
        .collect();
    let brush: Vec<f64> = brushed.iter().filter_map(|f| f.brush_ms).collect();
    let prepare: Vec<f64> = frames.iter().filter_map(|f| f.prepare_ms).collect();
    let carried = frames.iter().filter_map(|f| f.dabs_carried);
    let above = frames.iter().filter_map(|f| f.above_f);
    let above: Vec<bool> = above.collect();
    StrokeMetrics {
        sample_to_present_p99: percentile(&latency, 99),
        start_to_submitted_p99: percentile(&start_to_submitted, 99),
        brush_p99: percentile(&brush, 99),
        most_carried: carried.reduce(f64::max),
        drain: drain(frames),
        above_f: (!above.is_empty()).then(|| count(above.iter().filter(|&&a| a).count())),
        prepare_p99: percentile(&prepare, 99),
    }
}

/// How the queue drained after pen-up in `frames`; `None` if no frame says whether its stroke
/// was open, or the pen never touched.
fn drain(frames: &[&Frame]) -> Option<Drain> {
    let last_open = frames
        .iter()
        .rposition(|frame| frame.stroke_open == Some(true))?;
    let after = frames.get(last_open + 1..).unwrap_or_default();
    let drained = after
        .iter()
        .position(|frame| frame.dabs_carried == Some(0.0));
    Some(drained.map_or(Drain::Not, |at| Drain::Frames(count(at + 1))))
}

/// `n` as a metric's number (a count of frames: far below where `f64` loses integers).
fn count(n: usize) -> f64 {
    u32::try_from(n).map_or_else(|_| f64::from(u32::MAX), f64::from)
}

impl StrokeMetrics {
    /// Each metric as `(name after the scale, unit, value)`, for the stroke `index` into
    /// [`SEGMENTS`].
    pub(crate) fn named(&self, index: usize) -> Vec<(&'static str, Unit, Value)> {
        let Some(names) = NAMES.get(index) else {
            return Vec::new();
        };
        let drain = match self.drain {
            Some(Drain::Frames(frames)) => Value::Of(frames),
            Some(Drain::Not) => Value::NotReached,
            None => Value::Lacking,
        };
        let [latency, start, brush, carried, drained, above, prepare] = *names;
        vec![
            (latency, Unit::Ms, self.sample_to_present_p99.into()),
            (start, Unit::Ms, self.start_to_submitted_p99.into()),
            (brush, Unit::Ms, self.brush_p99.into()),
            (carried, Unit::Dabs, self.most_carried.into()),
            (drained, Unit::Frames, drain),
            (above, Unit::Frames, self.above_f.into()),
            (prepare, Unit::Ms, self.prepare_p99.into()),
        ]
    }
}
