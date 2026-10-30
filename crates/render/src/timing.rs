//! How long a frame's phases take on the GPU, by timestamp queries, read back without waiting.
//!
//! A frame is timed by **marks**: a timestamp written between two of its phases
//! (`Marks::mark`), so a phase's time is the time between the mark before it and its own. A
//! mark is an empty compute pass whose only work is its beginning-of-pass timestamp, which needs
//! only [`wgpu::Features::TIMESTAMP_QUERY`] (timestamps at pass boundaries; Apple GPUs have no
//! others), never `TIMESTAMP_QUERY_INSIDE_ENCODERS`, and leaves every pass's own code alone. On
//! Vulkan the timestamp is written at the bottom of the pipe, once all the work before it is
//! done, so the time between two marks is the work between them, gaps included.
//!
//! Each timed frame resolves its marks into a readback buffer of a small ring, mapped once the
//! frame is submitted; whichever thread polls the device next (the app's owner thread, at the
//! start of its frame) completes the mapping, and [`crate::Renderer::take_timings`] reads every
//! timed frame that has come back since, tagged with the frame it is of. A frame finds no free
//! buffer only if readbacks fall behind; it is then not timed (its times are absent, never
//! wrong).
//!
//! Without timestamp queries there is no timer (`GpuTimer`): frames are drawn the same, and
//! their times are absent. The level choice times its own stages with its own queries
//! ([`crate::lod::Lod::enable_timing`]), read back with its cut.

use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePassTimestampWrites, MapMode, QUERY_SIZE, QuerySet, QuerySetDescriptor, QueryType,
};

use crate::gpu::{Gpu, GpuError};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// A phase of a frame on the GPU, in the order a frame runs them. Each ends at its mark.
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

/// A mark: the frame's start, or the end of a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mark {
    /// Before anything of the frame.
    Start,
    /// After the phase.
    End(GpuPhase),
}

/// One frame's times on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuTimes {
    /// The frame they are of.
    frame: u64,
    /// From the frame's first mark to its last.
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

    /// From the frame's first mark to its last: the whole frame's GPU time.
    #[must_use]
    pub const fn total(&self) -> Duration {
        self.total
    }

    /// How long `phase` took; `None` if the frame didn't run it.
    #[must_use]
    pub fn phase(&self, phase: GpuPhase) -> Option<Duration> {
        self.phases.get(phase.index()).copied().flatten()
    }

    /// The times of frame `frame` from its `marks`, in the order written, and their `ticks`, one
    /// each, a tick `period` nanoseconds: each phase runs from the mark before its own. `None`
    /// with fewer than two marks (nothing between them), or with a tick count that isn't the
    /// marks'.
    pub(crate) fn from_ticks(
        frame: u64,
        marks: &[Mark],
        ticks: &[u64],
        period: f32,
    ) -> Option<Self> {
        if marks.len() < 2 || ticks.len() != marks.len() {
            return None;
        }
        let mut times = Self {
            frame,
            ..Self::default()
        };
        let steps = marks.iter().zip(ticks).skip(1).zip(ticks);
        for ((mark, &end), &begin) in steps {
            if let Mark::End(phase) = mark
                && let Some(slot) = times.phases.get_mut(phase.index())
            {
                let took = ticks_to_duration(end.saturating_sub(begin), period);
                *slot = Some(slot.unwrap_or_default() + took);
            }
        }
        let (first, last) = (ticks.first()?, ticks.last()?);
        times.total = ticks_to_duration(last.saturating_sub(*first), period);
        Some(times)
    }
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

/// The most marks a frame writes: the start and every phase's end, with room to spare.
const CAPACITY: u32 = 16;

/// Readback buffers in the ring: one frame is drawn at a time, read back a frame or two later.
const RING: usize = 4;

/// A timed frame's readback buffer, on its way back to the timer.
#[derive(Debug)]
enum Returned {
    /// Mapped: the ticks of frame `frame`'s `marks`, in order.
    Mapped {
        /// The buffer, mapped.
        buffer: Buffer,
        /// The frame it is of.
        frame: u64,
        /// The marks it holds the ticks of, in order.
        marks: Vec<Mark>,
    },
    /// Never submitted: free again, nothing to read.
    Unread(Buffer),
    /// Its mapping failed: the buffer is gone.
    Failed,
}

/// Times frames on the GPU by timestamp queries: the queries every frame writes its marks into,
/// and the ring of buffers they are read back through.
#[derive(Debug)]
pub(crate) struct GpuTimer {
    /// The marks' queries, reused by every frame.
    queries: QuerySet,
    /// Where a frame's queries are resolved.
    resolved: Buffer,
    /// Readback buffers free.
    free: Vec<Buffer>,
    /// Readback buffers made, at most [`RING`]; one whose mapping failed is no longer counted.
    made: usize,
    /// Nanoseconds a tick.
    period: f32,
    /// Given to each frame's readback, to send it back once mapped.
    done: Sender<Returned>,
    /// Where readbacks come back.
    returned: Receiver<Returned>,
    /// Times read while taking buffers back, not yet collected.
    ready: Vec<GpuTimes>,
}

impl GpuTimer {
    /// A timer on `gpu`; `None` if the device has no timestamp queries.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the queries or their buffer can't be made.
    pub(crate) fn new(gpu: &Gpu) -> Result<Option<Self>, GpuError> {
        let Some(period) = gpu.timestamp_period() else {
            return Ok(None);
        };
        let device = gpu.device();
        let (queries, resolved) = gpu.scoped("frame timer", || {
            let queries = device.create_query_set(&QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: QueryType::Timestamp,
                count: CAPACITY,
            });
            let resolved = device.create_buffer(&BufferDescriptor {
                label: Some("frame timestamps resolved"),
                size: u64::from(CAPACITY * QUERY_SIZE),
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (queries, resolved)
        })?;
        let (done, returned) = mpsc::channel();
        Ok(Some(Self {
            queries,
            resolved,
            free: Vec::new(),
            made: 0,
            period,
            done,
            returned,
            ready: Vec::new(),
        }))
    }

    /// The marks of frame `frame`, to write as it is encoded.
    pub(crate) fn begin(&self, frame: u64) -> Marks {
        Marks {
            queries: Some(self.queries.clone()),
            written: Vec::new(),
            frame,
        }
    }

    /// Encodes reading `marks` back, after the last of them: `None` if they are too few to time
    /// anything, or every readback buffer is in flight. Call [`TimedFrame::submitted`] once the
    /// frame is submitted.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if a readback buffer can't be made.
    pub(crate) fn finish(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        marks: Marks,
    ) -> Result<Option<TimedFrame>, GpuError> {
        self.take_back();
        let count = u32::try_from(marks.written.len()).unwrap_or(u32::MAX);
        if count < 2 || marks.queries.is_none() {
            return Ok(None);
        }
        let buffer = match self.free.pop() {
            Some(buffer) => buffer,
            None if self.made < RING => {
                let buffer = gpu.scoped("frame timestamps readback", || {
                    gpu.device().create_buffer(&BufferDescriptor {
                        label: Some("frame timestamps readback"),
                        size: u64::from(CAPACITY * QUERY_SIZE),
                        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                })?;
                self.made += 1;
                buffer
            }
            None => return Ok(None),
        };
        let bytes = u64::from(count * QUERY_SIZE);
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &buffer, 0, bytes);
        Ok(Some(TimedFrame {
            buffer: Some(buffer),
            frame: marks.frame,
            marks: marks.written,
            done: self.done.clone(),
        }))
    }

    /// Every timed frame read back since the last call, in the order they came back: those whose
    /// mapping a device poll has completed.
    pub(crate) fn collect(&mut self) -> Vec<GpuTimes> {
        self.take_back();
        std::mem::take(&mut self.ready)
    }

    /// Takes back every readback buffer returned, reading the ticks of those mapped into
    /// [`GpuTimer::ready`].
    fn take_back(&mut self) {
        while let Ok(returned) = self.returned.try_recv() {
            match returned {
                Returned::Mapped {
                    buffer,
                    frame,
                    marks,
                } => {
                    let ticks: Option<Vec<u64>> = buffer.get_mapped_range(..).ok().map(|view| {
                        let bytes = marks.len() * size_of::<u64>();
                        view.get(..bytes)
                            .map(bytemuck::pod_collect_to_vec)
                            .unwrap_or_default()
                    });
                    buffer.unmap();
                    self.free.push(buffer);
                    let read = ticks
                        .and_then(|ticks| GpuTimes::from_ticks(frame, &marks, &ticks, self.period));
                    self.ready.extend(read);
                }
                Returned::Unread(buffer) => self.free.push(buffer),
                Returned::Failed => self.made = self.made.saturating_sub(1),
            }
        }
    }
}

/// A frame's marks as it is encoded: written into the timer's queries, in order. Marks with no
/// timer ([`Marks::off`]) write nothing.
#[derive(Debug)]
pub(crate) struct Marks {
    /// The timer's queries; `None` when off.
    queries: Option<QuerySet>,
    /// The marks written, in order.
    written: Vec<Mark>,
    /// The frame they are of.
    frame: u64,
}

impl Marks {
    /// The marks of frame `frame` with `timer` ([`GpuTimer::begin`]), or off without one.
    pub(crate) fn of(timer: Option<&GpuTimer>, frame: u64) -> Self {
        timer.map_or_else(Self::off, |timer| timer.begin(frame))
    }

    /// Marks that write nothing: a frame not timed.
    pub(crate) const fn off() -> Self {
        Self {
            queries: None,
            written: Vec::new(),
            frame: 0,
        }
    }

    /// Writes `mark` into `encoder`, between passes: an empty compute pass whose beginning
    /// writes the timestamp. Nothing once the queries are all written, or with no timer.
    pub(crate) fn mark(&mut self, encoder: &mut CommandEncoder, mark: Mark) {
        let Some(queries) = &self.queries else {
            return;
        };
        let Ok(index) = u32::try_from(self.written.len()) else {
            return;
        };
        if index >= CAPACITY {
            return;
        }
        drop(encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("timestamp mark"),
            timestamp_writes: Some(ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        }));
        self.written.push(mark);
    }

    /// The end of `phase`: [`Marks::mark`] with [`Mark::End`].
    pub(crate) fn end(&mut self, encoder: &mut CommandEncoder, phase: GpuPhase) {
        self.mark(encoder, Mark::End(phase));
    }
}

/// A timed frame's readback, encoded: mark it submitted once its frame is, or drop it to give
/// the buffer back unread.
#[derive(Debug)]
#[must_use = "a timed frame not marked submitted is never read"]
pub(crate) struct TimedFrame {
    /// The buffer its ticks are copied to.
    buffer: Option<Buffer>,
    /// The frame.
    frame: u64,
    /// Its marks, in order.
    marks: Vec<Mark>,
    /// Where it goes once mapped.
    done: Sender<Returned>,
}

impl TimedFrame {
    /// Maps the buffer, now that its frame is submitted: [`GpuTimer::collect`] reads it once
    /// the device is polled after the GPU finishes the frame.
    pub(crate) fn submitted(mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let back = buffer.clone();
        let done = self.done.clone();
        let frame = self.frame;
        let marks = std::mem::take(&mut self.marks);
        buffer.map_async(MapMode::Read, .., move |result| {
            let returned = if result.is_ok() {
                Returned::Mapped {
                    buffer: back,
                    frame,
                    marks,
                }
            } else {
                Returned::Failed
            };
            // A timer that's gone has nothing to read.
            drop(done.send(returned));
        });
    }
}

impl Drop for TimedFrame {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // Unsubmitted: back to the ring unread. A timer that's gone has nothing to take back.
            drop(self.done.send(Returned::Unread(buffer)));
        }
    }
}
