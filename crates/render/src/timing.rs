//! How long a frame's phases take on the GPU, by timestamp queries, read back without waiting.
//!
//! A frame is timed by its **passes**: each compute or render pass of a phase carries a pair of
//! timestamps, written at its beginning and its end through the pass's own `timestamp_writes`
//! (`Stamps::compute`, `Stamps::render`), and a phase's time is the sum of its passes'. Work
//! that is no pass (the copies of the uploads and of the cut's readback) is timed by the gap it
//! sits in: from the end of the pass before it to the beginning of the pass after
//! (`Stamps::between`). A frame starts with a pass of its own (`Stamps::start`), a single
//! one-thread dispatch, so the uploads' copies have a pass before them. A phase's time thus
//! leaves out what lies between passes and is no phase's (barriers, the level choice's small
//! copy), so the phases sum to a little less than the frame's total: on lavapipe, which sets
//! each pass up on the CPU between them, about 8 % less.
//!
//! The pass-boundary timestamps need only [`wgpu::Features::TIMESTAMP_QUERY`], never
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS`, and they are the only kind every backend writes:
//!
//! - **Vulkan** (wgpu-hal 30, `vulkan/command.rs`) writes both with `vkCmdWriteTimestamp` at
//!   `BOTTOM_OF_PIPE`: the beginning once all earlier work is done, the end once the pass's is.
//!   Passes in one queue don't overlap, so the phases and gaps tile the frame.
//! - **Metal** (`metal/command.rs`) turns them into the pass descriptor's sample buffer
//!   attachments, sampled at stage boundaries (`MTLCounterSamplingPoint::AtStageBoundary`, the only
//!   point Apple GPUs offer): a compute pass's start and end of encoder, a render pass's start of
//!   vertex and end of fragment. An encoder with no work writes nothing: wgpu's own notes on blit
//!   encoders say so, and on an Apple M4 the earlier design, an empty compute pass whose beginning
//!   marked the boundary between phases, read zero or the same tick in every frame. So every
//!   stamped pass here does work, the start pass included. An unwritten sample resolves to
//!   `MTLCounterErrorValue` (all ones); wgpu takes a tick on Apple GPUs as 1 ns. Passes there may
//!   overlap (the tile-based GPU runs one's vertices beside another's fragments), so a phase's sum
//!   may exceed its wall time and a gap reads zero; the total is from the earliest stamp to the
//!   latest.
//! - **wgpu-core** resets a pass's two queries as the pass begins, and resolves a query not written
//!   in the same command encoder from what it last held (zero if never written). A frame here
//!   resolves only queries it wrote, its own in a later encoding, where they hold what it wrote: no
//!   other frame writes its slot's queries meanwhile.
//!
//! **A frame's start is no earlier than the frame before it ended.** The start pass touches no
//! resource, so nothing orders it after the earlier frame's work: on Metal, with the GPU behind,
//! an Apple M4 ran it beside the earlier frame's shading, accumulation and tone passes, and the
//! gap from its stamp to the next pass (the uploads) read 15–22 ms: the rest of that frame. With
//! the GPU keeping up, or waited for after each frame, the gap was 0.00–0.35 ms. So the start's
//! stamps are taken as no earlier than the last tick of the frame timed before
//! (`GpuTimes::from_ticks`'s `earlier_end`): the uploads are the gap from then to the next pass,
//! and the total runs from then. On Vulkan the start's stamp is at the bottom of the pipe, after
//! the earlier frame's work, so this changes nothing there.
//!
//! A tick of zero or all ones is no time: a pass with one is unwritten, and its phase (and any
//! gap beside it) is absent rather than wrong.
//!
//! The level choice's stage timer ([`crate::lod::Lod::enable_timing`]) stamps the level
//! choice's passes with its own queries, and a pass takes one pair of timestamps: when it is on,
//! the frame borrows those stamps (`Stamps::lod`) and resolves the level choice's queries
//! alongside its own.
//!
//! **A frame's own queries are resolved once the GPU has finished it**, by a later encoding.
//! Resolving them at the end of the frame's own command buffer read the last pass (the tone map
//! and overlay) as unwritten in every frame on an Apple M4. On Metal a render pass's end is
//! sampled as its fragment stage finishes, and the resolve (a blit) depends on no resource that
//! pass writes, so nothing orders it after that sample; every other pass is followed by one that
//! reads what it wrote, which is why only the last read as unwritten. So each frame takes a slot
//! of a small ring (its own query set and readback buffer); once the queue says the GPU has
//! finished the frame (`Queue::on_submitted_work_done`, heard at a device poll), the next frame's
//! encoding (or [`crate::Renderer::flush_timings`]) resolves the slot's queries and copies them
//! to its buffer, mapped once that submission is in. The level choice's queries, reused every
//! frame, are still resolved with the frame that borrowed them: its passes are followed by
//! passes that read what they wrote.
//!
//! Whichever thread polls the device next (the app's owner thread) completes the mapping, and
//! [`crate::Renderer::take_timings`] reads every timed frame that has come back since, tagged
//! with the frame it is of: two or three frames after it, a frame or so later than a resolve in
//! its own submission. The ring holds more slots than the frames the GPU may have at once
//! ([`crate::FRAMES_IN_FLIGHT`]), so a frame finds no free slot only if its user lets the GPU
//! fall further behind; it is then not timed (its times are absent, never wrong) and counted
//! ([`crate::DroppedReadbacks::timing`]).
//!
//! Without timestamp queries there is no timer (`GpuTimer`): frames are drawn the same, and
//! their times are absent.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites, ComputePipeline,
    MapMode, PollType, QUERY_SIZE, QuerySet, QuerySetDescriptor, QueryType, Queue,
    RenderPassTimestampWrites, ShaderSource,
};

use crate::gpu::{Gpu, GpuError};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// A phase of a frame on the GPU, in the order a frame runs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuPhase {
    /// The uploads' copies into tile slots and the tile tree, and the normal pass over the finest
    /// tiles they bring.
    Uploads,
    /// The bounds of the tiles whose data changed.
    Bounds,
    /// The level choice's first phase: choose, relax, sweep, emit (and its culling).
    LevelChoice,
    /// The raster: the stamps, both draws into the visibility buffer, the depth pyramids, and
    /// the level choice's second phase between the draws.
    Raster,
    /// Copying the cut out for its readback.
    Readback,
    /// The shading pass into the linear HDR image.
    Shade,
    /// The accumulation of the frame's sample into each pixel's history, while still.
    Accumulate,
    /// The last pass: the tone map (with the edge filter while moving), or the debug view, and
    /// the overlay.
    Tone,
}

impl GpuPhase {
    /// Every phase, in order.
    pub const ALL: [Self; 8] = [
        Self::Uploads,
        Self::Bounds,
        Self::LevelChoice,
        Self::Raster,
        Self::Readback,
        Self::Shade,
        Self::Accumulate,
        Self::Tone,
    ];

    /// The phase's name in records: lowercase, one word.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uploads => "uploads",
            Self::Bounds => "bounds",
            Self::LevelChoice => "lod",
            Self::Raster => "raster",
            Self::Readback => "readback",
            Self::Shade => "shade",
            Self::Accumulate => "accumulate",
            Self::Tone => "tone",
        }
    }

    /// Its place in [`GpuPhase::ALL`].
    const fn index(self) -> usize {
        self as usize
    }
}

/// Where a pass's pair of timestamps is written: its beginning at the index given, its end at
/// the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pair {
    /// The frame's own queries.
    Own(u32),
    /// The level choice's stage timer's queries ([`Stamps::lod`]).
    Lod(u32),
}

/// A stretch of a frame on the GPU, in the order encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Span {
    /// The frame's start pass: no phase's, the beginning of the frame's total.
    Start(Pair),
    /// A pass of a phase, from its beginning to its end.
    Pass(GpuPhase, Pair),
    /// Work of a phase that is no pass: from the end of the pass before to the beginning of the
    /// pass after.
    Between(GpuPhase),
}

impl Span {
    /// Its pass's timestamps; `None` for work between passes.
    const fn pair(self) -> Option<Pair> {
        match self {
            Self::Start(pair) | Self::Pass(_, pair) => Some(pair),
            Self::Between(_) => None,
        }
    }
}

/// A frame's resolved ticks: its own queries' and, when it borrowed the level choice's stamps,
/// the level choice's.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ticks<'a> {
    /// The frame's own, by query index.
    pub(crate) own: &'a [u64],
    /// The level choice's, by query index.
    pub(crate) lod: &'a [u64],
}

impl Ticks<'_> {
    /// The beginning and end of the pass stamped at `pair`; `None` if either is no time (zero,
    /// or Metal's all-ones error value: never written) or the end is before the beginning.
    fn of(self, pair: Pair) -> Option<(u64, u64)> {
        let (ticks, at) = match pair {
            Pair::Own(at) => (self.own, at),
            Pair::Lod(at) => (self.lod, at),
        };
        let at = usize::try_from(at).ok()?;
        let tick = |index: usize| {
            ticks
                .get(index)
                .copied()
                .filter(|&tick| tick != 0 && tick != u64::MAX)
        };
        let (begin, end) = (tick(at)?, tick(at.checked_add(1)?)?);
        (end >= begin).then_some((begin, end))
    }
}

/// One frame's times on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuTimes {
    /// The frame they are of.
    frame: u64,
    /// From the frame's first timestamp to its last.
    total: Duration,
    /// Each phase's time, by [`GpuPhase::ALL`]'s order; `None` for a phase the frame didn't run.
    phases: [Option<Duration>; GpuPhase::ALL.len()],
}

impl GpuTimes {
    /// The frame they are of: the number it was drawn as ([`crate::FrameInputs::frame`]).
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// From the frame's first timestamp to its last: the whole frame's GPU time.
    #[must_use]
    pub const fn total(&self) -> Duration {
        self.total
    }

    /// How long `phase` took: the sum of its passes and the gaps its work between passes sits
    /// in; `None` if the frame didn't run it, or a timestamp of it wasn't written.
    #[must_use]
    pub fn phase(&self, phase: GpuPhase) -> Option<Duration> {
        self.phases.get(phase.index()).copied().flatten()
    }

    /// The times of frame `frame` from its `spans`, in the order encoded, and their `ticks`, a
    /// tick `period` nanoseconds. `None` if no pass of it has its timestamps.
    ///
    /// `earlier_end` is the last tick of the frame timed before it, if one was: the frame's start
    /// is taken as no earlier (see the module notes: a start pass run beside the earlier
    /// frame's last passes would put the rest of that frame into this one's uploads and total).
    pub(crate) fn from_ticks(
        frame: u64,
        spans: &[Span],
        ticks: Ticks<'_>,
        period: f32,
        earlier_end: Option<u64>,
    ) -> Option<Self> {
        let floor = earlier_end.unwrap_or(0);
        // Per span, `None` for a gap, else its pass's stamps if written; the start's no earlier
        // than the frame before ended.
        let stamped: Vec<Option<Option<(u64, u64)>>> = spans
            .iter()
            .map(|span| {
                let written = span.pair().map(|pair| ticks.of(pair));
                match span {
                    Span::Start(_) => written
                        .map(|pair| pair.map(|(begin, end)| (begin.max(floor), end.max(floor)))),
                    Span::Pass(..) | Span::Between(_) => written,
                }
            })
            .collect();
        let mut times = Self {
            frame,
            ..Self::default()
        };
        let mut unwritten = [false; GpuPhase::ALL.len()];
        let mut add = |phase: GpuPhase, took: Option<u64>| {
            let index = phase.index();
            match took {
                Some(took) => {
                    if let Some(slot) = times.phases.get_mut(index) {
                        *slot = Some(slot.unwrap_or_default() + ticks_to_duration(took, period));
                    }
                }
                None => {
                    if let Some(flag) = unwritten.get_mut(index) {
                        *flag = true;
                    }
                }
            }
        };
        for (index, span) in spans.iter().enumerate() {
            match *span {
                Span::Start(_) => {}
                Span::Pass(phase, _) => {
                    let took = stamped.get(index).copied().flatten().flatten();
                    add(phase, took.map(|(begin, end)| end - begin));
                }
                Span::Between(phase) => {
                    let (before, after) = stamped.split_at(index.min(stamped.len()));
                    let before = nearest_pass(before.iter().rev());
                    let after = nearest_pass(after.iter().skip(1));
                    let gap = before
                        .zip(after)
                        .map(|((_, end), (begin, _))| begin.saturating_sub(end));
                    add(phase, gap);
                }
            }
        }
        let flags = unwritten.iter().zip(&mut times.phases);
        for (&missing, slot) in flags {
            if missing {
                *slot = None;
            }
        }
        let written = stamped.iter().flatten().flatten();
        let first = written.clone().map(|&(begin, _)| begin).min()?;
        let last = written.map(|&(_, end)| end).max()?;
        times.total = ticks_to_duration(last - first, period);
        Some(times)
    }
}

/// The latest tick written of the passes in `spans`, read from `ticks`; `None` if none was.
///
/// That is where the frame they time ended, as far as its stamps say: a pass whose end wasn't
/// written still says it began.
pub(crate) fn last_tick(spans: &[Span], ticks: Ticks<'_>) -> Option<u64> {
    let written = |tick: Option<&u64>| tick.copied().filter(|&t| t != 0 && t != u64::MAX);
    spans
        .iter()
        .filter_map(|span| span.pair())
        .flat_map(|pair| {
            let (list, at) = match pair {
                Pair::Own(at) => (ticks.own, at),
                Pair::Lod(at) => (ticks.lod, at),
            };
            let at = usize::try_from(at).unwrap_or(usize::MAX);
            [
                written(list.get(at)),
                written(list.get(at.saturating_add(1))),
            ]
        })
        .flatten()
        .max()
}

/// The first pass's stamps among `stamped` (per span, `None` for a gap), walked outward from a
/// gap; `None` if there is no pass, or its stamps weren't written.
fn nearest_pass<'a>(
    mut stamped: impl Iterator<Item = &'a Option<Option<(u64, u64)>>>,
) -> Option<(u64, u64)> {
    stamped.find_map(|span| *span).flatten()
}

/// `ticks` timestamp ticks of `period` nanoseconds each, as a duration.
pub(crate) fn ticks_to_duration(ticks: u64, period: f32) -> Duration {
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a frame's ticks are far below 2^52, and the product is whole nanoseconds"
    )]
    let nanos = (ticks as f64 * f64::from(period)) as u64;
    Duration::from_nanos(nanos)
}

/// The most of its own queries a frame writes: two a pass, with room to spare (a frame drawing
/// its scene stamps 16 passes).
const CAPACITY: u32 = 64;

/// The most of the level choice's queries a frame resolves.
const LOD_CAPACITY: u32 = 16;

/// Where the level choice's ticks start in the resolved and readback buffers: after the
/// frame's own, at a query resolve's alignment.
const LOD_AT: BufferAddress = {
    let own = CAPACITY as BufferAddress * QUERY_SIZE as BufferAddress;
    own.next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
};

/// Bytes of the resolved and readback buffers.
const BYTES: BufferAddress = LOD_AT + LOD_CAPACITY as BufferAddress * QUERY_SIZE as BufferAddress;

/// Slots in the ring: a slot is a frame's own queries and the buffer they are read back through.
///
/// A slot is busy from its frame's encoding until the render thread takes its readback back:
/// while the GPU runs the frame, until the first frame encoded after the GPU finished it resolves
/// its queries (or [`GpuTimer::flush`]), and until that frame is finished too and a poll maps the
/// buffer. Under the app's bound ([`crate::FRAMES_IN_FLIGHT`] frames unfinished) a frame being
/// encoded finds at most `FRAMES_IN_FLIGHT + 1` slots busy: the bound's frames, and one
/// resolved by a frame the GPU hasn't finished. With its own, it needs `FRAMES_IN_FLIGHT + 2`;
/// the ring has one to spare (the model in the tests,
/// `a_timing_slot_is_resolved_by_the_first_frame_after_its_own_and_freed_once_that_one_finishes`).
///
/// That relies on the order of the queue's callbacks: the timer asks for its frame's
/// (`TimedFrame::submitted`, setting the frame's done flag) as the frame is drawn, before the app
/// asks for its own (the render thread's, after the draw), and the queue runs a submission's
/// callbacks in the order asked. So by the time the app hears a frame finished and starts the
/// next, that frame's flag is set, and the next frame's encoding resolves it.
const RING: usize = crate::FRAMES_IN_FLIGHT + 3;

/// Bytes between slots' regions of the resolve buffer: a slot's own ticks, then the level
/// choice's at [`LOD_AT`], at a query resolve's alignment.
const STRIDE: BufferAddress = BYTES.next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);

/// The start pass's shader: one thread that does nothing, so the pass has work and every
/// backend writes its timestamps.
const START_WGSL: &str = "@compute @workgroup_size(1) fn start() {}";

/// A frame's own queries and the buffer they are read back through.
#[derive(Debug)]
struct Slot {
    /// The queries the frame's passes write.
    queries: QuerySet,
    /// Where the resolved ticks are copied, to map.
    readback: Buffer,
}

/// A timed frame's slot on its way back to the timer.
#[derive(Debug)]
enum Returned {
    /// Mapped: the ticks of frame `frame`'s `spans`, in slot `slot`'s buffer.
    Mapped {
        /// The slot.
        slot: usize,
        /// The frame it is of.
        frame: u64,
        /// The spans it holds the ticks of, in order.
        spans: Vec<Span>,
    },
    /// Never written (a frame that stamped no pass, or ran out of queries): free again, nothing
    /// to read, nothing lost.
    Unread(usize),
    /// Resolved for a frame, but that submission never went: free again, the frame's times
    /// lost.
    Lost(usize),
    /// Frame `0` was never submitted: its slot waits for nothing, and its times are lost.
    Abandoned(u64),
    /// Its mapping failed: the slot is gone, and the frame's times with it.
    Failed,
}

/// A frame stamped and submitted whose own queries aren't resolved yet: they are, by a later
/// encoding, once the GPU has finished it ([`GpuTimer::resolve_finished`]).
#[derive(Debug)]
struct Pending {
    /// Its slot.
    slot: usize,
    /// The frame.
    frame: u64,
    /// Its spans, in order.
    spans: Vec<Span>,
    /// Its own queries written.
    used: u32,
    /// Set once the GPU has finished the frame (`Queue::on_submitted_work_done`).
    finished: Arc<AtomicBool>,
}

/// A slot whose queries a submission resolves: mapped once that submission is.
#[derive(Debug)]
struct Resolving {
    /// The slot.
    slot: usize,
    /// The frame it is of.
    frame: u64,
    /// Its spans, in order.
    spans: Vec<Span>,
    /// The slot's readback buffer, to map.
    readback: Buffer,
}

/// Times frames on the GPU by timestamp queries: a ring of slots (each a frame's queries and its
/// readback buffer), the start pass, and the buffer queries are resolved into.
#[derive(Debug)]
pub(crate) struct GpuTimer {
    /// The slots made, by index; a slot whose mapping failed is never used again.
    slots: Vec<Slot>,
    /// Slots free, by index.
    free: Vec<usize>,
    /// Frames stamped and submitted, oldest first, their own queries not resolved yet.
    pending: VecDeque<Pending>,
    /// The start pass's pipeline.
    start: ComputePipeline,
    /// Where queries are resolved: a region of [`STRIDE`] bytes a slot, its own ticks first,
    /// then the level choice's at [`LOD_AT`].
    resolved: Buffer,
    /// Frames whose times were lost, ever: no slot free, a slot not made, a readback failed or
    /// never submitted.
    dropped: u64,
    /// Nanoseconds a tick.
    period: f32,
    /// Given to each frame's readback, to send it back once mapped.
    done: Sender<Returned>,
    /// Where readbacks come back.
    returned: Receiver<Returned>,
    /// Times read while taking buffers back, not yet collected.
    ready: Vec<GpuTimes>,
    /// The last tick of the newest frame read, which the next frame's start is no earlier than.
    last_tick: Option<u64>,
}

impl GpuTimer {
    /// A timer on `gpu`; `None` if the device has no timestamp queries.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the resolve buffer can't be made; as [`Gpu::compute_pipeline`] for
    /// the start pass.
    pub(crate) fn new(gpu: &Gpu) -> Result<Option<Self>, GpuError> {
        let Some(period) = gpu.timestamp_period() else {
            return Ok(None);
        };
        let resolved = gpu.scoped("frame timer", || {
            gpu.device().create_buffer(&BufferDescriptor {
                label: Some("frame timestamps resolved"),
                size: STRIDE * RING as BufferAddress,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        let start = gpu.compute_pipeline(
            "frame timer start",
            ShaderSource::Wgsl(START_WGSL.into()),
            "start",
        )?;
        let (done, returned) = mpsc::channel();
        Ok(Some(Self {
            slots: Vec::new(),
            free: Vec::new(),
            pending: VecDeque::new(),
            start,
            resolved,
            dropped: 0,
            period,
            done,
            returned,
            ready: Vec::new(),
            last_tick: None,
        }))
    }

    /// The stamps of frame `frame`, to give its passes as it is encoded: off if no slot is free
    /// (counted, [`GpuTimer::dropped_readbacks`]) or a new one can't be made.
    pub(crate) fn begin(&mut self, gpu: &Gpu, frame: u64) -> Stamps {
        self.take_back();
        let Some(slot) = self.free.pop().or_else(|| self.make_slot(gpu)) else {
            return Stamps::off();
        };
        let Some(queries) = self.slots.get(slot).map(|slot| slot.queries.clone()) else {
            return Stamps::off();
        };
        Stamps {
            on: Some(On {
                slot,
                queries,
                start: self.start.clone(),
                used: 0,
                lod: None,
                overflowed: false,
            }),
            spans: Vec::new(),
            frame,
        }
    }

    /// A new slot, if the ring has room for one and the GPU makes it; else the frame is counted
    /// dropped.
    fn make_slot(&mut self, gpu: &Gpu) -> Option<usize> {
        if self.slots.len() >= RING {
            self.dropped += 1;
            return None;
        }
        let made = self.try_make_slot(gpu);
        if made.is_none() {
            self.dropped += 1;
        }
        made
    }

    /// A new slot, if the GPU makes it.
    fn try_make_slot(&mut self, gpu: &Gpu) -> Option<usize> {
        let device = gpu.device();
        let slot = gpu
            .scoped("frame timer slot", || Slot {
                queries: device.create_query_set(&QuerySetDescriptor {
                    label: Some("frame timestamps"),
                    ty: QueryType::Timestamp,
                    count: CAPACITY,
                }),
                readback: device.create_buffer(&BufferDescriptor {
                    label: Some("frame timestamps readback"),
                    size: BYTES,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
            })
            .ok()?;
        self.slots.push(slot);
        Some(self.slots.len() - 1)
    }

    /// Where slot `slot`'s region of the resolve buffer starts.
    fn region(slot: usize) -> BufferAddress {
        STRIDE * slot as BufferAddress
    }

    /// Encodes resolving the own queries of every pending frame the GPU has finished (the device
    /// polled since), oldest first, and copying them to their slots' buffers; returns them, to
    /// map once `encoder` is submitted.
    fn resolve_finished(&mut self, encoder: &mut CommandEncoder) -> Vec<Resolving> {
        let mut resolving = Vec::new();
        while let Some(oldest) = self.pending.front()
            && oldest.finished.load(Ordering::Acquire)
        {
            let Some(pending) = self.pending.pop_front() else {
                break;
            };
            let Some(slot) = self.slots.get(pending.slot) else {
                continue;
            };
            let at = Self::region(pending.slot);
            encoder.resolve_query_set(&slot.queries, 0..pending.used, &self.resolved, at);
            let own = u64::from(pending.used) * u64::from(QUERY_SIZE);
            encoder.copy_buffer_to_buffer(&self.resolved, at, &slot.readback, 0, own);
            resolving.push(Resolving {
                slot: pending.slot,
                frame: pending.frame,
                spans: pending.spans,
                readback: slot.readback.clone(),
            });
        }
        resolving
    }

    /// Encodes, after the last of `stamps`' passes, resolving the level choice's queries it
    /// borrowed (reused by the next frame) and the own queries of earlier frames the GPU has
    /// finished; this frame's own wait until it has finished too. Call
    /// [`TimedFrame::submitted`] once the frame is submitted.
    ///
    /// A frame's own queries are resolved only once the GPU is done with it: on Metal a resolve
    /// in the same command buffer can run before the last pass's end is sampled (the module
    /// notes).
    pub(crate) fn finish(&mut self, encoder: &mut CommandEncoder, stamps: Stamps) -> TimedFrame {
        self.take_back();
        let resolving = self.resolve_finished(encoder);
        let Stamps { on, spans, frame } = stamps;
        let mut timed = TimedFrame {
            resolving,
            this: None,
            done: self.done.clone(),
        };
        let Some(on) = on else {
            return timed;
        };
        if on.overflowed || on.used == 0 {
            // Back unread: nothing worth resolving was written. Not counted as a drop: a frame
            // that stamped no pass has no times to lose, and one that ran out of queries
            // (`CAPACITY`) is a bug in the stamps, not a readback the ring couldn't make.
            drop(self.done.send(Returned::Unread(on.slot)));
            return timed;
        }
        if let Some((lod, count)) = &on.lod
            && let Some(slot) = self.slots.get(on.slot)
        {
            let at = Self::region(on.slot) + LOD_AT;
            encoder.resolve_query_set(lod, 0..*count, &self.resolved, at);
            let bytes = u64::from(*count) * u64::from(QUERY_SIZE);
            encoder.copy_buffer_to_buffer(&self.resolved, at, &slot.readback, LOD_AT, bytes);
        }
        let finished = Arc::new(AtomicBool::new(false));
        self.pending.push_back(Pending {
            slot: on.slot,
            frame,
            spans,
            used: on.used,
            finished: Arc::clone(&finished),
        });
        timed.this = Some((frame, finished));
        timed
    }

    /// Resolves, in a submission of its own, the own queries of every frame the GPU has finished
    /// (polling the device first, without waiting), so their times come back without waiting for
    /// a later frame: for the end of a run, and for tests.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if polling or the submission fails.
    pub(crate) fn flush(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        gpu.device().poll(PollType::Poll)?;
        self.take_back();
        if !self
            .pending
            .front()
            .is_some_and(|pending| pending.finished.load(Ordering::Acquire))
        {
            return Ok(());
        }
        let timed = gpu.scoped("flushing the frame timer", || {
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame timer flush"),
                });
            let resolving = self.resolve_finished(&mut encoder);
            gpu.queue().submit([encoder.finish()]);
            TimedFrame {
                resolving,
                this: None,
                done: self.done.clone(),
            }
        })?;
        timed.submitted(gpu.queue());
        Ok(())
    }

    /// Frames whose times were lost, ever: [`GpuTimer::begin`] found no slot free or couldn't
    /// make one, or a readback failed or was never submitted. Zero while the GPU is kept within
    /// [`crate::FRAMES_IN_FLIGHT`] frames and the device works.
    pub(crate) const fn dropped_readbacks(&self) -> u64 {
        self.dropped
    }

    /// Every timed frame read back since the last call, in the order they came back: those whose
    /// mapping a device poll has completed.
    pub(crate) fn collect(&mut self) -> Vec<GpuTimes> {
        self.take_back();
        std::mem::take(&mut self.ready)
    }

    /// Takes back every slot returned, reading the ticks of those mapped into
    /// [`GpuTimer::ready`].
    fn take_back(&mut self) {
        while let Ok(returned) = self.returned.try_recv() {
            match returned {
                Returned::Mapped { slot, frame, spans } => {
                    let Some(buffer) = self.slots.get(slot).map(|slot| &slot.readback) else {
                        continue;
                    };
                    let ticks: Vec<u64> = buffer
                        .get_mapped_range(..)
                        .ok()
                        .and_then(|view| view.get(..).map(bytemuck::pod_collect_to_vec))
                        .unwrap_or_default();
                    buffer.unmap();
                    self.free.push(slot);
                    let lod_from =
                        usize::try_from(LOD_AT / u64::from(QUERY_SIZE)).unwrap_or(usize::MAX);
                    let ticks = Ticks {
                        own: ticks.get(..lod_from).unwrap_or_default(),
                        lod: ticks.get(lod_from..).unwrap_or_default(),
                    };
                    let read =
                        GpuTimes::from_ticks(frame, &spans, ticks, self.period, self.last_tick);
                    self.last_tick = last_tick(&spans, ticks).max(self.last_tick);
                    self.ready.extend(read);
                }
                Returned::Unread(slot) => self.free.push(slot),
                Returned::Lost(slot) => {
                    self.dropped += 1;
                    self.free.push(slot);
                }
                Returned::Abandoned(frame) => {
                    self.dropped += 1;
                    let (abandoned, kept): (VecDeque<Pending>, VecDeque<Pending>) =
                        std::mem::take(&mut self.pending)
                            .into_iter()
                            .partition(|pending| pending.frame == frame);
                    self.pending = kept;
                    self.free
                        .extend(abandoned.iter().map(|pending| pending.slot));
                }
                Returned::Failed => self.dropped += 1,
            }
        }
    }
}

/// A timed frame's slot and start pass, while it is encoded.
#[derive(Debug)]
struct On {
    /// Its slot.
    slot: usize,
    /// The slot's queries.
    queries: QuerySet,
    /// The start pass's pipeline.
    start: ComputePipeline,
    /// Queries given out, two a pass.
    used: u32,
    /// The level choice's queries and how many to resolve, once a pass borrowed its stamps.
    lod: Option<(QuerySet, u32)>,
    /// Whether a pass found no queries left: the frame is then not timed.
    overflowed: bool,
}

/// A frame's timestamps as it is encoded: a pair for each pass that asks, in order. Stamps with
/// no timer ([`Stamps::off`]) give none and note nothing.
#[derive(Debug)]
pub(crate) struct Stamps {
    /// The timer's queries; `None` when off.
    on: Option<On>,
    /// The frame's spans, in the order encoded.
    spans: Vec<Span>,
    /// The frame they are of.
    frame: u64,
}

impl Stamps {
    /// The stamps of frame `frame` with `timer` on `gpu` ([`GpuTimer::begin`]), or off without
    /// one.
    pub(crate) fn of(timer: Option<&mut GpuTimer>, gpu: &Gpu, frame: u64) -> Self {
        timer.map_or_else(Self::off, |timer| timer.begin(gpu, frame))
    }

    /// Stamps that write nothing: a frame not timed.
    pub(crate) const fn off() -> Self {
        Self {
            on: None,
            spans: Vec::new(),
            frame: 0,
        }
    }

    /// Takes a pair of the frame's own queries for a pass, noted as `span` of it; `None` (and
    /// the frame not timed) once they run out, or with no timer.
    fn claim(&mut self, span: impl FnOnce(Pair) -> Span) -> Option<(&QuerySet, u32)> {
        let on = self.on.as_mut()?;
        let at = on.used;
        if on.overflowed || at + 2 > CAPACITY {
            on.overflowed = true;
            return None;
        }
        on.used += 2;
        self.spans.push(span(Pair::Own(at)));
        Some((&on.queries, at))
    }

    /// Encodes the frame's start: a pass of one thread, stamped, before anything else of it.
    /// The uploads' copies and normal pass come next, before the scene's passes: they are timed
    /// by the gap after it ([`GpuPhase::Uploads`]).
    pub(crate) fn start(&mut self, encoder: &mut CommandEncoder) {
        let Some(pipeline) = self.on.as_ref().map(|on| on.start.clone()) else {
            return;
        };
        let Some((queries, at)) = self.claim(Span::Start) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("frame timer start"),
            timestamp_writes: Some(ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(at),
                end_of_pass_write_index: Some(at + 1),
            }),
        });
        pass.set_pipeline(&pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        self.between(GpuPhase::Uploads);
    }

    /// The timestamp writes of a compute pass of `phase`, to encode at once; `None` when off.
    pub(crate) fn compute(&mut self, phase: GpuPhase) -> Option<ComputePassTimestampWrites<'_>> {
        let (query_set, at) = self.claim(|pair| Span::Pass(phase, pair))?;
        Some(ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(at),
            end_of_pass_write_index: Some(at + 1),
        })
    }

    /// The timestamp writes of a render pass of `phase`, to encode at once; `None` when off.
    pub(crate) fn render(&mut self, phase: GpuPhase) -> Option<RenderPassTimestampWrites<'_>> {
        let (query_set, at) = self.claim(|pair| Span::Pass(phase, pair))?;
        Some(RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(at),
            end_of_pass_write_index: Some(at + 1),
        })
    }

    /// Notes work of `phase` that is no pass (copies), encoded next: timed by the gap between
    /// the passes around it.
    pub(crate) fn between(&mut self, phase: GpuPhase) {
        if self.on.is_some() {
            self.spans.push(Span::Between(phase));
        }
    }

    /// Notes a pass of `phase` that the level choice's stage timer stamps: its beginning at
    /// query `at` of `queries`, its end at the next, of the `count` the frame resolves.
    pub(crate) fn lod(&mut self, phase: GpuPhase, queries: &QuerySet, count: u32, at: u32) {
        let Some(on) = self.on.as_mut() else {
            return;
        };
        if count > LOD_CAPACITY || at + 2 > count {
            on.overflowed = true;
            return;
        }
        on.lod.get_or_insert_with(|| (queries.clone(), count));
        self.spans.push(Span::Pass(phase, Pair::Lod(at)));
    }

    /// Gives up timing the frame: its encoding failed part way, so a pass may have claimed
    /// queries it never wrote.
    pub(crate) fn abandon(&mut self) {
        if let Some(on) = self.on.as_mut() {
            on.overflowed = true;
        }
    }
}

/// What a frame's encoding resolved and stamped, for the timer: mark it submitted once the
/// frame is, or drop it to give its slots back unread.
#[derive(Debug)]
#[must_use = "a timed frame not marked submitted is never read"]
pub(crate) struct TimedFrame {
    /// Earlier frames whose queries this submission resolves.
    resolving: Vec<Resolving>,
    /// This frame, if it was stamped: its number, and the flag the GPU's finishing it sets.
    this: Option<(u64, Arc<AtomicBool>)>,
    /// Where slots go back.
    done: Sender<Returned>,
}

impl TimedFrame {
    /// Now that its submission is in `queue`: maps the slots it resolved
    /// ([`GpuTimer::collect`] reads them once the device is polled after the GPU finishes it),
    /// and asks the queue to say when the GPU has finished this frame, so a later encoding
    /// resolves its queries.
    pub(crate) fn submitted(mut self, queue: &Queue) {
        let resolving = std::mem::take(&mut self.resolving);
        for Resolving {
            slot,
            frame,
            spans,
            readback,
        } in resolving
        {
            let done = self.done.clone();
            readback.map_async(MapMode::Read, .., move |result| {
                let returned = if result.is_ok() {
                    Returned::Mapped { slot, frame, spans }
                } else {
                    Returned::Failed
                };
                // A timer that's gone has nothing to read.
                drop(done.send(returned));
            });
        }
        if let Some((_, finished)) = self.this.take() {
            queue.on_submitted_work_done(move || finished.store(true, Ordering::Release));
        }
    }
}

impl Drop for TimedFrame {
    fn drop(&mut self) {
        // Unsubmitted: what it resolved goes back, those frames' times lost, and its own frame
        // waits for nothing. A timer that's gone has nothing to take back.
        for resolving in self.resolving.drain(..) {
            drop(self.done.send(Returned::Lost(resolving.slot)));
        }
        if let Some((frame, _)) = self.this.take() {
            drop(self.done.send(Returned::Abandoned(frame)));
        }
    }
}
