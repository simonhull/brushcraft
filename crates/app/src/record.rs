//! A recorded run's per-frame record (`--record`): what each frame cost and what it drew, as JSON
//! lines in the style of `brushcraft-bench`'s records, which `brushcraft-bench perf` reads.
//!
//! A scripted run is recorded, and so is a live session in a window (the pen's latency of strokes
//! drawn by hand, read by `brushcraft-bench strokes-summary`): its record has no segments and its
//! frames none, and it keeps every frame's record in memory (about a kilobyte each, some 400 MB
//! an hour at 120 frames a second) until the window closes and the lines are written.
//!
//! A frame's facts arrive at different times: its owner thread phases as it is made (its upkeep
//! and jobs once they have run), its render thread phases when it is shown, its GPU times two or
//! three frames later (read back without waiting, [`render::timing`]; the last frames' as the
//! render thread stops), and its cut (what the level
//! choice drew, with the level choice's own stage times) when residency reads that back. The
//! [`Recorder`] keeps each frame's [`FrameRecord`] by number and fills it in as they come; the
//! lines are written once the run is over ([`Recorder::lines`]). A fact that never came (the last
//! frames' readbacks, GPU times on a device without timestamp queries) is `null`.
//!
//! Every line is a flat JSON object of named numbers, strings, booleans and number arrays, with
//! `"bench":"render"` and its `"result"`: one `run` line (the setting), one `segment` line per
//! segment of the script, then one `frame` line per frame, in order. Times are in milliseconds;
//! `time_ms` and `presented_ms` are on the frame loop's clock (from its first frame), and
//! `open_ms` is how long before that the project began to open. The run line also gives
//! `clock_unix_us`, the frame loop's clock's start by the system clock (microseconds since the
//! Unix epoch), so samples taken beside the run by another process (the performance suite's GPU
//! clocks) can be joined to frames; and `refresh_ms`, the interval frames are paced to as the run
//! starts (the display's refresh, 1/60 s offscreen). The frame line's fields:
//!
//! - `frame`, `time_ms` (its start), `presented_ms` (when the owner heard it was shown), `segment`,
//!   `moving` (whether the view changed since the last frame);
//! - `cpu_ms`: the frame's CPU time on its latency path: the owner thread's input, brush, feed and
//!   uploads, then the render thread's encode and submit; and those phases, with the owner's upkeep
//!   and jobs (after the hand-off) and the render thread's acquire and present, each `<phase>_ms`;
//! - `gpu_wait_ms`: how long the frame waited, once due, for the GPU to finish an earlier one (the
//!   frame loop keeps at most [`render::FRAMES_IN_FLIGHT`] unfinished): 0 while the GPU keeps up;
//! - `cut_readback_dropped`, `request_readback_dropped`, `timing_readback_dropped`: whether the
//!   frame made none of that readback for want of a free buffer (`null` before it is known). Each
//!   ring holds more buffers than the frames in flight, so these stay false; the run line totals
//!   them as `readbacks_dropped`, so a regression shows;
//! - `gpu_ms`: the whole frame on the GPU, and each of its phases ([`GpuPhase`]) as
//!   `gpu_<phase>_ms`, the sum of its passes (`null` for a phase the frame didn't run, or whose
//!   timestamps the GPU didn't write);
//! - `lod_<stage>_ms`: the level choice's own stages, timed by its queries (bounds, choose, relax,
//!   sweep, emit, post; 0 for a stage that didn't run: the bounds, with no tile changed);
//! - from the frame's cut: `drawn` (whether it came back), `triangles`, `cut`, `tiles_per_level`
//!   (the cut's tiles at each level from 0), `over_budget` (tiles in view left over 1 px, the
//!   spec's count), `waiting`, `budget_limited` and `held_by_neighbours` (why), `shortfall`,
//!   `cone_culled` (tiles culled by their normal cone) and `cone_exempt` (tiles their cone would
//!   cull, drawn because their morph isn't settled), `requested`, `relax_passes`, the budget's
//!   `budget_binds` (the budget bound, and its choice made the cut: by need, worst first over the
//!   whole view), `charged_triangles` (what the choice was charged, by what it may draw: at least
//!   `triangles`, at most the budget), `budget_error_px` (where the budget bound and stopped short,
//!   about the screen error it stopped at; `null` otherwise), `budget_slack` (triangles the budget
//!   may have left unspent: the fine bin it stopped at and the held tiles), `relax_refused` (raises
//!   the relaxation couldn't fit, which make it bind), `admitted` and `held` (refinements the
//!   budget's choice made, and those its seams refused), and `level_jump`: the largest change of
//!   the level drawing a finest tile the current stroke edited (the last one's, until the next
//!   begins) since the cut heard before (the M1 spec's "stroke seen from a distance": no jump
//!   bigger than one), `null` with no stroked tile drawn in both;
//! - residency as the frame was staged: `resident`, `wanted`, `evicted` (this frame),
//!   `uploaded_bytes` and `uploaded_tiles`, `standing_in` (display tiles standing in for a stroke's
//!   edits, until the refresh replaces them) and `pending_epochs` (epochs staged over frames, not
//!   switched in yet): with no dab left to apply, a stroke is settled, every tile it touched
//!   showing its final data, once both are 0;
//! - the brush ([`crate::frame::BrushReport`]): `submitted_ms` (when the render thread's present
//!   call returned, on the frame loop's clock), `sample_to_present_ms` (the pen's latency: from the
//!   oldest pen sample whose dabs the frame showed first, applied or ahead of their finest work, by
//!   the pen's de-jittered clock, to the frame's present submitted; `null` for a frame that showed
//!   none) and `received_to_present_ms` (the same from when that sample reached the app),
//!   `dabs_applied` (finest work), `dabs_shown` (shown ahead of their finest work, which didn't fit
//!   the frame), `dabs_carried` (placed but neither applied nor shown, left for the next frames:
//!   the stroke's lag), `dabs_catching_up` (shown and not yet applied: the surface is exact once
//!   none are), `tiles_edited`, `above_f` (a dab of the frame above the footprint limit F, which no
//!   frame's time fits), and `stroke_open` (a stroke still open as the brush step ended: the pen
//!   down; the frames after the last open one until none carries dabs are the queue draining after
//!   pen-up), and `kept_tiles`, `kept_bytes` and `kept_beyond_budget` (what the stroke keeps of the
//!   surface as it began, counted in the store's budget, and the bytes the budget had no room for).
//!   What the brush did between the last frame and this one (whose edits this frame shows) counts
//!   as this frame's, its time `between_ms`. The owner's `prepare_ms` is the brush's work after the
//!   hand-off, off the latency path: a big brush's catch-up (the finest work of dabs shown ahead of
//!   it) while there is one, else warming the surface under a hovering pen.

use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime};

use render::lod::{LodStats, LodTimings};
use render::residency::ResidencyStats;
use render::timing::{GpuPhase, GpuTimes};
use render::{DrawTimes, DroppedReadbacks};

use crate::frame::{BrushReport, CutHeard, OwnerPhases};
use crate::script::{Script, Segment};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// A value in a record line.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    /// A string.
    Str(String),
    /// An unsigned integer.
    Int(u64),
    /// A number; `null` if it isn't finite.
    Float(f64),
    /// `true` or `false`.
    Bool(bool),
    /// An array of unsigned integers.
    Ints(Vec<u64>),
    /// `null`: a fact that never came.
    Null,
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::Str(s.to_owned())
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Self::Int(n)
    }
}

impl From<u32> for Value {
    fn from(n: u32) -> Self {
        Self::Int(n.into())
    }
}

impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Self::Int(u64::try_from(n).unwrap_or(u64::MAX))
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Self::Float(x)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<Duration> for Value {
    /// In milliseconds.
    fn from(d: Duration) -> Self {
        Self::Float(d.as_secs_f64() * 1e3)
    }
}

impl<T: Into<Self>> From<Option<T>> for Value {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Null, Into::into)
    }
}

/// One line: a flat JSON object, its fields in the order added.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Line {
    /// The fields.
    fields: Vec<(String, Value)>,
}

impl Line {
    /// A line of the render record's result `result`.
    fn new(result: &str) -> Self {
        Self {
            fields: vec![
                ("bench".to_owned(), "render".into()),
                ("result".to_owned(), result.into()),
            ],
        }
    }

    /// The line with `name` set to `value` too.
    #[must_use]
    fn with(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.push((name.into(), value.into()));
        self
    }

    /// The line as JSON, without the newline.
    pub(crate) fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (index, (name, value)) in self.fields.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            quote(&mut out, name);
            out.push(':');
            match value {
                Value::Str(s) => quote(&mut out, s),
                Value::Int(n) => out.push_str(&n.to_string()),
                Value::Float(x) if x.is_finite() => out.push_str(&x.to_string()),
                Value::Float(_) | Value::Null => out.push_str("null"),
                Value::Bool(b) => out.push_str(&b.to_string()),
                Value::Ints(ns) => {
                    let ns: Vec<String> = ns.iter().map(u64::to_string).collect();
                    out.push('[');
                    out.push_str(&ns.join(","));
                    out.push(']');
                }
            }
        }
        out.push('}');
        out
    }

    /// The line with the brush's fields: when the present was `submitted`, the pen's
    /// `latency`, and what the brush step did (`brush`).
    fn brush_fields(
        self,
        submitted: Option<Duration>,
        latency: Option<Latency>,
        brush: Option<&BrushReport>,
    ) -> Self {
        self.with("submitted_ms", submitted)
            .with("sample_to_present_ms", latency.map(|l| l.taken))
            .with("received_to_present_ms", latency.map(|l| l.received))
            .with("dabs_applied", brush.map(|b| b.dabs_applied))
            .with("dabs_shown", brush.map(|b| b.dabs_shown))
            .with("dabs_carried", brush.map(|b| b.dabs_carried))
            .with("dabs_catching_up", brush.map(|b| b.dabs_catching_up))
            .with("tiles_edited", brush.map(|b| b.tiles_edited))
            .with("above_f", brush.map(|b| b.above_limit))
            .with("stroke_open", brush.map(|b| b.open))
            .with("kept_tiles", brush.map(|b| b.kept.tiles))
            .with("kept_bytes", brush.map(|b| b.kept.bytes))
            .with("kept_beyond_budget", brush.map(|b| b.kept.beyond_budget))
    }

    /// The line with residency's fields from `counts`.
    fn residency_fields(self, counts: Option<ResidencyCounts>) -> Self {
        self.with("resident", counts.map(|c| c.resident))
            .with("wanted", counts.map(|c| c.wanted))
            .with("evicted", counts.map(|c| c.evicted))
            .with("uploaded_bytes", counts.map(|c| c.uploaded_bytes))
            .with("uploaded_tiles", counts.map(|c| c.uploaded_tiles))
            .with("standing_in", counts.map(|c| c.standing_in))
            .with("pending_epochs", counts.map(|c| c.pending_epochs))
    }
}

/// Appends `s` to `out` as a JSON string literal, quotes included.
fn quote(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if u32::from(c) < 0x20 => {
                let code = u32::from(c);
                out.push_str("\\u00");
                out.extend(
                    [code >> 4, code & 0xF]
                        .into_iter()
                        .filter_map(|digit| char::from_digit(digit, 16)),
                );
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Residency's counts as a frame was staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ResidencyCounts {
    /// Tiles resident.
    pub(crate) resident: usize,
    /// Tiles requested and not resident yet.
    pub(crate) wanted: usize,
    /// Tiles evicted this frame.
    pub(crate) evicted: u64,
    /// Bytes the frame staged.
    pub(crate) uploaded_bytes: u64,
    /// Tiles the frame staged.
    pub(crate) uploaded_tiles: u64,
    /// Whether the frame made no request readback for want of a free buffer.
    pub(crate) request_readback_dropped: bool,
    /// Display tiles standing in for a stroke's edits, until the refresh replaces them.
    pub(crate) standing_in: usize,
    /// Epochs residency has pending: staged over frames, not switched in yet.
    pub(crate) pending_epochs: usize,
}

/// What a frame's cut readback said.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct SceneRecord {
    /// Its counts.
    pub(crate) stats: LodStats,
    /// The cut's tiles at each level, from 0.
    pub(crate) levels: Vec<u64>,
    /// The level choice's stages on the GPU, if timed.
    pub(crate) stages: Option<LodTimings>,
    /// The largest level change of a tile the current (or last) stroke edited, from the cut
    /// heard before this one: `None` with no stroke drawn in both.
    pub(crate) level_jump: Option<u8>,
}

/// Everything recorded of one frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct FrameRecord {
    /// Its number, from 1.
    pub(crate) frame: u64,
    /// Its start on the frame loop's clock.
    pub(crate) time: Duration,
    /// When the owner heard it was shown, on the same clock.
    pub(crate) presented: Option<Duration>,
    /// The script's segment it fell in.
    pub(crate) segment: Option<Segment>,
    /// Whether the view changed since the last frame.
    pub(crate) moving: bool,
    /// The owner thread's phases, once the frame's jobs have run.
    pub(crate) owner: Option<OwnerPhases>,
    /// The render thread's phases, once shown.
    pub(crate) render: Option<DrawTimes>,
    /// Its GPU times, once read back.
    pub(crate) gpu: Option<GpuTimes>,
    /// The readbacks the render thread made none of for it, once shown.
    pub(crate) dropped: Option<DroppedReadbacks>,
    /// Residency as it was staged.
    pub(crate) residency: Option<ResidencyCounts>,
    /// Its cut, once read back.
    pub(crate) scene: Option<SceneRecord>,
    /// What its brush step did.
    pub(crate) brush: Option<BrushReport>,
    /// When its present was submitted, on the frame loop's clock, once shown.
    pub(crate) submitted: Option<Duration>,
    /// The pen's latency, once shown, if it applied dabs.
    pub(crate) latency: Option<Latency>,
}

/// A frame's pen latency: from the oldest pen sample whose dabs it applied to its present
/// submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Latency {
    /// From when the pen took the sample, by its de-jittered clock: the M1 spec's measure.
    pub(crate) taken: Duration,
    /// From when the sample reached the app.
    pub(crate) received: Duration,
}

impl FrameRecord {
    /// The frame's CPU time on its latency path: the owner's input, brush, feed and uploads, and
    /// the render thread's encode and submit; `None` until both are known.
    pub(crate) fn cpu(&self) -> Option<Duration> {
        let owner = self.owner?;
        let render = self.render?;
        Some(owner.input + owner.brush + owner.feed + owner.uploads + render.encode)
    }

    /// The frame's line.
    fn line(&self) -> Line {
        let owner = self.owner.unwrap_or_default();
        let known = self.owner.is_some();
        let owner_ms = |d: Duration| known.then_some(d);
        let render = self.render;
        let times = Line::new("frame")
            .with("frame", self.frame)
            .with("time_ms", self.time)
            .with("presented_ms", self.presented)
            .with("segment", self.segment.map(Segment::name))
            .with("moving", self.moving)
            .with("cpu_ms", self.cpu())
            .with("gpu_wait_ms", owner_ms(owner.gpu_wait))
            .with("input_ms", owner_ms(owner.input))
            .with("brush_ms", owner_ms(owner.brush))
            .with("between_ms", owner_ms(owner.between))
            .with("feed_ms", owner_ms(owner.feed))
            .with("uploads_ms", owner_ms(owner.uploads))
            .with("upkeep_ms", owner_ms(owner.upkeep))
            .with("prepare_ms", owner_ms(owner.prepare))
            .with("jobs_ms", owner_ms(owner.jobs))
            .with("acquire_ms", render.map(|r| r.acquire))
            .with("encode_ms", render.map(|r| r.encode))
            .with("present_ms", render.map(|r| r.present))
            .with("gpu_ms", self.gpu.map(|gpu| gpu.total()));
        let phases = GpuPhase::ALL.into_iter().fold(times, |line, phase| {
            let took = self.gpu.and_then(|gpu| gpu.phase(phase));
            line.with(format!("gpu_{}_ms", phase.name()), took)
        });
        let stages = self.scene.as_ref().and_then(|scene| scene.stages);
        let stage = |pick: fn(&LodTimings) -> Duration| stages.as_ref().map(pick);
        let stages_line = phases
            .with("lod_bounds_ms", stage(|s| s.bounds))
            .with("lod_choose_ms", stage(|s| s.choose))
            .with("lod_relax_ms", stage(|s| s.relax))
            .with("lod_sweep_ms", stage(|s| s.sweep))
            .with("lod_emit_ms", stage(|s| s.emit))
            .with("lod_post_ms", stage(|s| s.post));
        let dropped = self.dropped;
        self.scene_fields(stages_line)
            .residency_fields(self.residency)
            .with("cut_readback_dropped", dropped.map(|d| d.cut > 0))
            .with(
                "request_readback_dropped",
                self.residency.map(|c| c.request_readback_dropped),
            )
            .with("timing_readback_dropped", dropped.map(|d| d.timing > 0))
            .brush_fields(self.submitted, self.latency, self.brush.as_ref())
    }

    /// How many readbacks the frame made none of: cut, request and timing.
    fn readbacks_dropped(&self) -> u64 {
        let render = self.dropped.map_or(0, |d| d.cut + d.timing);
        let requests = self.residency.is_some_and(|c| c.request_readback_dropped);
        render + u64::from(requests)
    }

    /// `line` with the cut's fields.
    fn scene_fields(&self, line: Line) -> Line {
        let scene = self.scene.as_ref();
        let stat = |pick: fn(&LodStats) -> u32| scene.map(|s| pick(&s.stats));
        let line = line
            .with("drawn", scene.is_some())
            .with("triangles", stat(|s| s.drawn_triangles))
            .with("cut", stat(|s| s.cut));
        let line = match scene {
            Some(scene) => line.with("tiles_per_level", Value::Ints(scene.levels.clone())),
            None => line.with("tiles_per_level", Value::Null),
        };
        line.with("over_budget", stat(|s| s.over_pixel))
            .with("waiting", stat(|s| s.waiting))
            .with("budget_limited", stat(|s| s.budget_limited))
            .with("held_by_neighbours", stat(|s| s.held_by_neighbours))
            .with("shortfall", stat(|s| s.shortfall))
            .with("cone_culled", stat(|s| s.culled.1))
            .with("cone_exempt", stat(|s| s.cone_exempt))
            .with("requested", stat(|s| s.requested))
            .with("relax_passes", scene.map(|s| s.stats.passes_used()))
            .with("budget_binds", scene.map(|s| s.stats.budget_binds))
            .with("charged_triangles", stat(|s| s.refined_triangles))
            .with(
                "budget_error_px",
                scene.and_then(|s| s.stats.budget_error()).map(f64::from),
            )
            .with("budget_slack", stat(|s| s.budget_slack))
            .with("relax_refused", stat(|s| s.relax_refused))
            .with("admitted", stat(|s| s.admitted))
            .with("held", stat(|s| s.held))
            .with(
                "level_jump",
                scene.and_then(|s| s.level_jump).map(u32::from),
            )
    }
}

/// The run's setting, for its first line.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunInfo {
    /// The project file.
    pub(crate) project: String,
    /// The GPU's name and backend.
    pub(crate) adapter: String,
    /// The frame's size.
    pub(crate) size: (u32, u32),
    /// Whether the GPU's phases were timed.
    pub(crate) gpu_timing: bool,
    /// Whether the level choice's stages were timed.
    pub(crate) lod_timing: bool,
    /// From the project beginning to open to the frame loop's clock starting.
    pub(crate) open: Duration,
    /// The frame loop's clock's start, as time since the Unix epoch by the system clock; `None`
    /// if the system clock is set before the epoch.
    pub(crate) clock_unix: Option<Duration>,
    /// The interval frames are paced to as the run starts: the display's refresh, or 1/60 s
    /// offscreen.
    pub(crate) refresh: Duration,
}

/// The moment `at` (now or earlier) as time since the Unix epoch by the system clock; `None` if
/// the system clock is set before the epoch.
pub(crate) fn since_epoch(at: Instant) -> Option<Duration> {
    let ago = Instant::now().saturating_duration_since(at);
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?;
    now.checked_sub(ago)
}

/// `d` in whole microseconds, for a field a float of milliseconds would round (a time since the
/// epoch).
fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// Collects a scripted run's frames as their facts arrive (the module notes).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Recorder {
    /// Each frame's record, by number.
    frames: BTreeMap<u64, FrameRecord>,
    /// Which frame each residency frame number was staged in: cut readbacks name residency's.
    residency_frames: BTreeMap<u64, u64>,
    /// Residency's evictions ever, as the last frame was staged.
    evictions: u64,
    /// Residency's request readbacks dropped ever, as the last frame was staged.
    requests_dropped: u64,
}

impl Recorder {
    /// Frame `frame` started at `time` in `segment`, the view moving or not.
    pub(crate) fn started(
        &mut self,
        frame: u64,
        time: Duration,
        segment: Option<Segment>,
        moving: bool,
    ) {
        self.frames.insert(
            frame,
            FrameRecord {
                frame,
                time,
                segment,
                moving,
                ..FrameRecord::default()
            },
        );
    }

    /// Frame `frame` was staged by residency as its frame `staged` (if it has residency), which
    /// counted `stats` then.
    pub(crate) fn staged(
        &mut self,
        frame: u64,
        staged: Option<u64>,
        stats: Option<&ResidencyStats>,
    ) {
        if let Some(staged) = staged {
            self.residency_frames.insert(staged, frame);
        }
        let Some(stats) = stats else {
            return;
        };
        let evicted = stats.evictions.saturating_sub(self.evictions);
        self.evictions = stats.evictions;
        let requests_dropped = stats.request_readbacks_dropped > self.requests_dropped;
        self.requests_dropped = stats.request_readbacks_dropped;
        if let Some(record) = self.frames.get_mut(&frame) {
            record.residency = Some(ResidencyCounts {
                resident: stats.resident,
                wanted: stats.wanted,
                evicted,
                uploaded_bytes: stats.staged_bytes,
                uploaded_tiles: stats.staged_tiles,
                request_readback_dropped: requests_dropped,
                standing_in: stats.standing_in,
                pending_epochs: stats.pending_epochs,
            });
        }
    }

    /// Frame `frame`'s owner thread phases, its upkeep and jobs included.
    pub(crate) fn owner(&mut self, frame: u64, phases: OwnerPhases) {
        if let Some(record) = self.frames.get_mut(&frame) {
            record.owner = Some(phases);
        }
    }

    /// Frame `frame` was shown at `at`, its render thread phases `times`.
    pub(crate) fn shown(&mut self, frame: u64, times: DrawTimes, at: Duration) {
        if let Some(record) = self.frames.get_mut(&frame) {
            record.render = Some(times);
            record.presented = Some(at);
        }
    }

    /// Frame `frame`'s brush step did `report`.
    pub(crate) fn brush(&mut self, frame: u64, report: &BrushReport) {
        if let Some(record) = self.frames.get_mut(&frame) {
            record.brush = Some(*report);
        }
    }

    /// Frame `frame`'s present was submitted at `submitted` on the frame loop's clock, with the
    /// pen's `latency` if it applied dabs.
    pub(crate) fn latency(&mut self, frame: u64, submitted: Duration, latency: Option<Latency>) {
        if let Some(record) = self.frames.get_mut(&frame) {
            record.submitted = Some(submitted);
            record.latency = latency;
        }
    }

    /// Frame `frame`'s render thread made none of `dropped`'s readbacks.
    pub(crate) fn dropped(&mut self, frame: u64, dropped: DroppedReadbacks) {
        if let Some(record) = self.frames.get_mut(&frame) {
            record.dropped = Some(dropped);
        }
    }

    /// A frame's GPU times came back.
    pub(crate) fn gpu(&mut self, times: GpuTimes) {
        if let Some(record) = self.frames.get_mut(&times.frame()) {
            record.gpu = Some(times);
        }
    }

    /// A frame's cut came back.
    pub(crate) fn cut(&mut self, cut: CutHeard) {
        let frame = self.residency_frames.get(&cut.frame).copied();
        if let Some(record) = frame.and_then(|frame| self.frames.get_mut(&frame)) {
            record.scene = Some(cut.scene);
        }
    }

    /// The record's lines: the run's (`info`), each of `script`'s segments' (none for a live
    /// session, which has no script: its speed is 1), then each frame's.
    pub(crate) fn lines(&self, info: &RunInfo, script: Option<&Script>) -> Vec<String> {
        let dropped: u64 = self
            .frames
            .values()
            .map(FrameRecord::readbacks_dropped)
            .sum();
        let run = Line::new("run")
            .with("project", info.project.as_str())
            .with("adapter", info.adapter.as_str())
            .with("width", info.size.0)
            .with("height", info.size.1)
            .with("speed", script.map_or(1.0, |script| script.speed().get()))
            .with("gpu_timing", info.gpu_timing)
            .with("lod_timing", info.lod_timing)
            .with("open_ms", info.open)
            .with("clock_unix_us", info.clock_unix.map(micros))
            .with("refresh_ms", info.refresh)
            .with("frames", self.frames.len())
            .with("readbacks_dropped", dropped);
        let spans = script.map(Script::segments).unwrap_or_default();
        let segments = spans.into_iter().map(|(segment, start, end)| {
            Line::new("segment")
                .with("segment", segment.name())
                .with("start_ms", start)
                .with("end_ms", end)
        });
        let frames = self.frames.values().map(FrameRecord::line);
        std::iter::once(run)
            .chain(segments)
            .chain(frames)
            .map(|line| line.to_json())
            .collect()
    }
}
