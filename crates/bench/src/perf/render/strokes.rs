//! The pen's metrics, from the app's strokes script read back.
//!
//! The script (`brushcraft --script --script-path strokes`, its record written with `--record`)
//! draws five strokes (`stroke_near`, `stroke_far`, `stroke_long`, `stroke_heavy` and
//! `stroke_huge`, the record's segments; the last a brush of a million vertices, far above F, the
//! M1 spec's big brush); each gives, over its segment's frames:
//!
//! - **sample to present**, p50 and p99: from the oldest pen sample whose dabs a frame showed first
//!   (applied, or previewed ahead of their finest work) to its present being submitted, over the
//!   frames that showed a pen sample's dabs first, the **pen's frames** (the M1 spec's pen latency:
//!   ≤ 8 ms at p99 below the footprint limit F, and the same through the preview for the big
//!   brush). A frame applying only a catch-up of dabs shown before shows no sample first: it isn't
//!   one of them;
//! - **frame start to present submitted**, p99, over the pen's frames: the frame's own share;
//! - **brush**, p99, over the pen's frames: the frame's brush phase (the work between frames, whose
//!   edits it shows, is the record's `between_ms`, not counted here);
//! - **most dabs carried**: the most any frame left neither applied nor shown for the next: the
//!   stroke's lag behind the pen. A brush that keeps up carries at most a few samples' worth for a
//!   frame or two; a queue that grows over the stroke passes the ceiling ([`MOST_CARRIED`]) within
//!   its first tens of milliseconds;
//! - **drain frames**: from pen-up (the frame after the last whose stroke was still open) to the
//!   first frame carrying none, counted inclusively (ceiling 2, the brush-engine plan's "drains
//!   within two frames of pen-up"); never drained within the segment is not reached;
//! - **frames above F**: frames with a dab above the footprint limit, where the latency target
//!   doesn't hold;
//! - **catch-up**, p99, over every frame of the segment: the brush's work after the hand-off, off
//!   the latency path (the app's `prepare_ms`): the finest work of dabs shown ahead of it while any
//!   are, else warming the surface under a hovering pen. Named `prepare_p99_ms` before 2026-11-03;
//!   history reads the old name as this one (`history::RENAMED`);
//! - **most dabs catching up**: the most dabs any frame left shown ahead of their finest work (the
//!   big-brush preview, M1 spec "Big brushes");
//! - **exact after pen-up**, ms: from the first frame after pen-up to the first whose brush has no
//!   dab left to apply, shown or not (the finest level exact; the M1 spec's "time until the finest
//!   level is exact after pen-up"), within the segment;
//! - **settle frames**: from the first frame after pen-up, counted inclusively, to the first that
//!   is exact and has no display tile standing in and no epoch pending: every tile the stroke
//!   touched shows its final data, the refresh's included. Looked for up to the next stroke;
//! - **most level jump**, in levels, over the frames whose stroke was open: the largest change of
//!   the level drawing a tile the stroke edited from one cut to the next (the record's
//!   `level_jump`; the M1 spec's "stroke seen from a distance": none bigger than one, the far
//!   stroke's ceiling). A frame whose cut never came back, or drew none of the stroke's tiles,
//!   gives none.
//!
//! **Ceilings** ([`TARGETS`]): the pen's targets hold on any project, so the medium project is
//! held to them as the benchmark project is (the renderer's orbit targets are the benchmark's
//! alone, [`super::TARGETS`]): sample to present ≤ 8 ms p99 on every stroke, the big brush's
//! through its preview; the queue's most carried and its drain; the far stroke's level jumps
//! ≤ 1; and the big brush exact ≤ 2 s after its 2 s stroke (the spec's provisional figure).
//!
//! A run whose record lacks a segment (an older app) gives no sample of its metrics.

use crate::perf::metric::Unit;
use crate::perf::render::{Frame, Recorded, Value, percentile};

/// The strokes script's stroke segments, in order.
pub(crate) const SEGMENTS: [&str; 5] = [
    "stroke_near",
    "stroke_far",
    "stroke_long",
    "stroke_heavy",
    "stroke_huge",
];

/// The ceiling on the most dabs carried, on the benchmark project: a few samples' worth.
///
/// A fast stroke places two or three dabs a sample, and a brush keeping up shows one or two
/// frames carrying a few as a dab runs long; a growing queue passes it within the stroke's first
/// tens of milliseconds.
pub(crate) const MOST_CARRIED: f64 = 8.0;

/// One segment's metric names after the scale, in [`StrokeMetrics::named`]'s order.
macro_rules! names {
    ($segment:literal) => {
        [
            concat!("render.", $segment, ".sample_to_present_p50_ms"),
            concat!("render.", $segment, ".sample_to_present_p99_ms"),
            concat!("render.", $segment, ".start_to_submitted_p99_ms"),
            concat!("render.", $segment, ".brush_p99_ms"),
            concat!("render.", $segment, ".dabs_carried_max"),
            concat!("render.", $segment, ".drain_frames"),
            concat!("render.", $segment, ".frames_above_f"),
            concat!("render.", $segment, ".catch_up_p99_ms"),
            concat!("render.", $segment, ".catching_up_max"),
            concat!("render.", $segment, ".exact_after_up_ms"),
            concat!("render.", $segment, ".settle_frames"),
            concat!("render.", $segment, ".level_jump_max"),
        ]
    };
}

/// Each metric's name after the scale, by segment (as [`SEGMENTS`]) and then in
/// [`StrokeMetrics::named`]'s order.
const NAMES: [[&str; 12]; 5] = [
    names!("stroke_near"),
    names!("stroke_far"),
    names!("stroke_long"),
    names!("stroke_heavy"),
    names!("stroke_huge"),
];

/// The big brush's time until exact after its pen-up, at most, in ms: the M1 spec's provisional
/// "≤ 2 s after a 2 s stroke" (the strokes script's huge stroke lasts 2 s).
pub(crate) const HUGE_EXACT_MS: f64 = 2000.0;

/// The spec's targets on the pen's metrics, by the end of their names, in their units: judged on
/// the medium and the benchmark projects (see the module notes).
pub(crate) const TARGETS: [(&str, f64); 5] = [
    ("sample_to_present_p99_ms", 8.0),
    ("dabs_carried_max", MOST_CARRIED),
    ("drain_frames", 2.0),
    ("stroke_far.level_jump_max", 1.0),
    ("stroke_huge.exact_after_up_ms", HUGE_EXACT_MS),
];

/// Whether something after pen-up came to an end, and how long it took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Drain {
    /// In this many frames (or milliseconds, for the time to exact), counted from the first
    /// frame after pen-up.
    Frames(f64),
    /// Not within the frames looked at.
    Not,
}

/// One stroke's metrics from one run; `None` for one the record can't give.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct StrokeMetrics {
    /// Sample to present submitted, p50, in ms.
    pub(crate) sample_to_present_p50: Option<f64>,
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
    /// The brush's catch-up after the hand-off, p99, in ms.
    pub(crate) catch_up_p99: Option<f64>,
    /// The most dabs any frame left catching up.
    pub(crate) most_catching_up: Option<f64>,
    /// From pen-up until exact, in ms.
    pub(crate) exact_after_up: Option<Drain>,
    /// Frames from pen-up until settled.
    pub(crate) settle: Option<Drain>,
    /// The largest level jump of the stroke's tiles while it was open, in levels.
    pub(crate) most_level_jump: Option<f64>,
}

/// Each stroke's metrics from `recorded`, in [`SEGMENTS`]' order.
pub(crate) fn derive(recorded: &Recorded) -> [StrokeMetrics; 5] {
    SEGMENTS.map(|segment| {
        let frames: Vec<&Frame> = recorded.in_segment(segment).collect();
        let mut metrics = stroke(&frames);
        metrics.settle = settle(recorded, segment);
        metrics
    })
}

/// One stroke's metrics from its segment's `frames`, in order (the settle frames, which look
/// past the segment, aside).
pub(crate) fn stroke(frames: &[&Frame]) -> StrokeMetrics {
    if frames.is_empty() {
        return StrokeMetrics::default();
    }
    let pens = frames
        .iter()
        .filter(|frame| frame.sample_to_present_ms.is_some());
    let brushed: Vec<&&Frame> = pens.collect();
    let latency: Vec<f64> = brushed
        .iter()
        .filter_map(|f| f.sample_to_present_ms)
        .collect();
    let start_to_submitted: Vec<f64> = brushed
        .iter()
        .filter_map(|f| Some(f.submitted_ms? - f.time_ms))
        .collect();
    let brush: Vec<f64> = brushed.iter().filter_map(|f| f.brush_ms).collect();
    let catch_up: Vec<f64> = frames.iter().filter_map(|f| f.prepare_ms).collect();
    let carried = frames.iter().filter_map(|f| f.dabs_carried);
    let above = frames.iter().filter_map(|f| f.above_f);
    let above: Vec<bool> = above.collect();
    let catching_up = frames.iter().filter_map(|f| f.dabs_catching_up);
    let open = frames.iter().filter(|f| f.stroke_open == Some(true));
    let jumps = open.filter_map(|f| f.level_jump);
    StrokeMetrics {
        sample_to_present_p50: percentile(&latency, 50),
        sample_to_present_p99: percentile(&latency, 99),
        start_to_submitted_p99: percentile(&start_to_submitted, 99),
        brush_p99: percentile(&brush, 99),
        most_carried: carried.reduce(f64::max),
        drain: drain(frames),
        above_f: (!above.is_empty()).then(|| count(above.iter().filter(|&&a| a).count())),
        catch_up_p99: percentile(&catch_up, 99),
        most_catching_up: catching_up.reduce(f64::max),
        exact_after_up: exact_after_up(frames),
        settle: None,
        most_level_jump: jumps.reduce(f64::max),
    }
}

/// Whether a frame's brush has no dab left to apply, shown or not; `None` if the record doesn't
/// say (an older app: no `dabs_catching_up`).
fn exact(frame: &Frame) -> Option<bool> {
    Some(frame.dabs_carried? == 0.0 && frame.dabs_catching_up? == 0.0)
}

/// The time from the first frame after pen-up in `frames` to the first that is
/// [exact], in ms; `None` if the record can't say.
fn exact_after_up(frames: &[&Frame]) -> Option<Drain> {
    let last_open = frames
        .iter()
        .rposition(|frame| frame.stroke_open == Some(true))?;
    let after = frames.get(last_open + 1..).unwrap_or_default();
    let up = after.first()?.time_ms;
    let mut exact_at = None;
    for frame in after {
        if exact(frame)? {
            exact_at = Some(frame.time_ms);
            break;
        }
    }
    Some(exact_at.map_or(Drain::Not, |at| Drain::Frames(at - up)))
}

/// Frames from the first after `segment`'s pen-up to the first settled one, counted inclusively.
///
/// Settled: [exact], with no display tile standing in and no epoch pending. Looked for
/// over the record's frames up to the next stroke; `None` if the record can't say.
fn settle(recorded: &Recorded, segment: &str) -> Option<Drain> {
    let last_open = recorded.frames.iter().rposition(|frame| {
        frame.segment.as_deref() == Some(segment) && frame.stroke_open == Some(true)
    })?;
    let after = recorded.frames.get(last_open + 1..).unwrap_or_default();
    for (n, frame) in after.iter().enumerate() {
        if frame.stroke_open == Some(true) {
            return Some(Drain::Not);
        }
        let quiet = frame.standing_in? == 0.0 && frame.pending_epochs? == 0.0;
        if exact(frame)? && quiet {
            return Some(Drain::Frames(count(n + 1)));
        }
    }
    Some(Drain::Not)
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
        let value = |drain: Option<Drain>| match drain {
            Some(Drain::Frames(frames)) => Value::Of(frames),
            Some(Drain::Not) => Value::NotReached,
            None => Value::Lacking,
        };
        let [
            median,
            latency,
            start,
            brush,
            carried,
            drained,
            above,
            catch_up,
            catching_up,
            exact,
            settle,
            jump,
        ] = *names;
        vec![
            (median, Unit::Ms, self.sample_to_present_p50.into()),
            (latency, Unit::Ms, self.sample_to_present_p99.into()),
            (start, Unit::Ms, self.start_to_submitted_p99.into()),
            (brush, Unit::Ms, self.brush_p99.into()),
            (carried, Unit::Dabs, self.most_carried.into()),
            (drained, Unit::Frames, value(self.drain)),
            (above, Unit::Frames, self.above_f.into()),
            (catch_up, Unit::Ms, self.catch_up_p99.into()),
            (catching_up, Unit::Dabs, self.most_catching_up.into()),
            (exact, Unit::Ms, value(self.exact_after_up)),
            (settle, Unit::Frames, value(self.settle)),
            (jump, Unit::Levels, self.most_level_jump.into()),
        ]
    }
}
