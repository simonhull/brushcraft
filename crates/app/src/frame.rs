//! The frame loop, on the store's owner thread (the M1 spec's "Threads": the owner thread is
//! the frame loop).
//!
//! [`Owner`] holds the project, so it lives on the one thread that may touch the store, and
//! makes each frame in a fixed order:
//!
//! 1. **input**: the pointer input queued since the last frame moves the camera; the pen samples
//!    queued since the last frame are routed ([`crate::pen::PenRouter`]: a stroke to navigation or
//!    the brush, the rest to both), navigation's moving the camera after the mouse's input, and
//!    become the frame's [`PenFrame`]; then the surface under the pointer is picked for the overlay
//!    if the pointer or the view changed ([`crate::pick`]), except while the brush is stroking (the
//!    pick is on the pen's latency path, and the stroke moves the surface under it anyway): it is
//!    made once the stroke is over, in a frame wanted for it;
//! 2. **brush**: the brush's step ([`Brush`]; the app's is [`crate::sculpting::Sculpting`]), which
//!    sees the camera as the input left it and the frame's pen samples, applies the frame's dabs
//!    and closes their epoch, so they are drained just after and drawn in this frame; what it
//!    reports ([`BrushReport`]) goes to a scripted run's record, and the oldest sample it applied
//!    is kept until the frame is shown, for the pen's latency (sample to present submitted,
//!    [`Presented::submitted`]);
//! 3. **feed**: the stale-tile feed is drained: the epochs closed since the last frame;
//! 4. **uploads**: GPU residency ([`Uploads`], [`crate::uploads::GpuResidency`]) takes the drained
//!    epochs and stages the frame's tiles from the store, until half the way to the frame's
//!    deadline;
//! 5. **hand-off**: the frame goes to the render thread, as a [`FrameRequest`] carrying the staged
//!    uploads and the scene (residency's buffers, the camera's view, the frame's time and whether
//!    the camera moves), which it starts encoding at once (the uploads first, then the passes);
//! 6. **upkeep**: meanwhile the project's recovery writer is ticked and polled;
//! 7. **prepare**: the brush's work off the latency path ([`Brush::prepare`]: warming the surface
//!    under a hovering pen), until the frame's deadline;
//! 8. **jobs**: and the store's resumable jobs (display-layer refresh and rebuild) run until the
//!    frame's deadline, at least one slice of them. This frame draws nothing they produce: the
//!    epochs they close are drained by the next frame, which they want.
//!
//! Handing off before the jobs keeps them out of the input's latency (the plan's order,
//! amended by the Task 1 review).
//!
//! **How the scene is shown** ([`Look`]: the MatCap's style, the wireframe, or the debug view of
//! clusters) changes from the keyboard ([`LookChange`]). A frame wanted for that alone, with the
//! view still and nothing staged, waiting or morphing, **shades the last scene again**: the
//! shading pass only, no level choice or raster (the spec's target for a light or material
//! change: one shading pass, independent of the mesh). Any other frame draws the scene whole, in
//! the new look.
//!
//! **Never outrunning the GPU:** a frame starts only while the GPU has fewer than
//! [`render::FRAMES_IN_FLIGHT`] frames unfinished ([`schedule`]). The owner never waits on the
//! GPU: when a frame is due and the GPU is that far behind ([`Next::Gpu`]), the loop polls the
//! device for finished frames and meanwhile runs the store's jobs and the upkeep
//! ([`Owner::wait_for_gpu`]), sleeping only when there are none ([`GPU_POLL`] says why it polls).
//! The time a frame waited so is its `gpu_wait` ([`OwnerPhases::gpu_wait`]).
//!
//! Frames are made on demand: [`schedule::Scheduler`] decides when, from why they are wanted.
//! A frame that leaves store jobs queued, uploads in flight, requests pending, or epochs in the
//! feed it hasn't drained (closed by its jobs or upkeep) wants the next one, as does upkeep
//! between frames that closes epochs; a cut change the uploads hear of keeps frames coming until
//! the morph it started is over; input that moves the camera wants one; nothing else does.
//!
//! **The morph's clock** is the frame loop's own: each frame is stamped with its start's time
//! since the loop began, on this thread, and the stamp travels with the frame, so the render
//! thread never reads a clock and a frame is the same given its inputs.
//!
//! **A scripted run** (`--script`, [`crate::script`]) puts the camera where its path has it at
//! each frame's stamp, before the input phase (pointer input is dropped), wants every frame until
//! the path is over, and records every frame ([`crate::record`]).

pub(crate) mod schedule;

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use render::lod::LodStats;
use render::raster::FramePasses;
use render::residency::ResidencyStats;
use render::surface::{Presentation, Size};
use render::upload::FrameUploads;
use render::{DrawTimes, Look, SceneView, aa};
use store::{Drained, JobsReport, Project, SpatialIndex, StaleCursor, TileStore};

use self::schedule::{Next, Reason, Reasons, Scheduler};
use crate::camera::{Arcball, Camera, Pixel};
use crate::navigate::{Input, Navigator};
use crate::overlay::{self, Snapshot};
use crate::pen::{PenFrame, PenNavigation, PenRouter, PenSample, PenStatus, Route};
use crate::pick::Picker;
use crate::record::{Latency, Recorder, RunInfo, SceneRecord};
use crate::render_thread::{FrameRequest, Presented, ResidentScene, SceneRequest};
use crate::script::{Script, ScriptPen, Segment};

/// What the brush is handed each time the frame loop calls it.
///
/// The project, its spatial index over the finest level (picking's, [`crate::pick`]; `None` if
/// the project keeps no display layer, and then nothing can be brushed), and the view as the
/// input phase left it.
#[derive(Debug)]
pub(crate) struct BrushContext<'a> {
    /// The open project.
    pub(crate) project: &'a mut Project,
    /// Its spatial index over the finest level, shared with picking; a brush that edits brings
    /// it up to date.
    pub(crate) index: Option<&'a mut SpatialIndex>,
    /// The camera, as this frame's input left it.
    pub(crate) camera: &'a Camera,
    /// The frame's size.
    pub(crate) size: Size,
}

/// A change to the brush's settings, from the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BrushChange {
    /// A smaller brush.
    Smaller,
    /// A larger brush.
    Larger,
    /// A weaker brush.
    Weaker,
    /// A stronger brush.
    Stronger,
}

/// What a frame's brush step did, for the frame loop and a scripted run's record.
///
/// Nothing here is particular to sculpting: a painting brush places and applies dabs the same
/// way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct BrushReport {
    /// The oldest pen sample whose dabs this frame applied: the frame's latency runs from it to
    /// its present being submitted. `None` if the frame applied nothing.
    pub(crate) oldest: Option<SampleTimes>,
    /// Dabs applied this frame.
    pub(crate) dabs_applied: usize,
    /// Dabs placed but left for the next frames, past the frame's budget.
    pub(crate) dabs_carried: usize,
    /// Tiles whose data the frame's dabs edited.
    pub(crate) tiles_edited: usize,
    /// Whether one of the frame's dabs was above the footprint limit F (the M1 spec's "Big
    /// brushes"): the pen's latency target holds below it.
    pub(crate) above_limit: bool,
    /// Whether the brush has work left for the next frame (dabs carried, a stroke ending): the
    /// next frame is wanted.
    pub(crate) busy: bool,
}

/// When a pen sample was taken and when it reached the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleTimes {
    /// When the pen took it, on its de-jittered clock.
    pub(crate) taken: Instant,
    /// When it reached the app.
    pub(crate) received: Instant,
}

impl SampleTimes {
    /// The earlier of two samples' times, field by field.
    #[must_use]
    pub(crate) fn earliest(self, other: Self) -> Self {
        Self {
            taken: self.taken.min(other.taken),
            received: self.received.min(other.received),
        }
    }
}

/// The brush's place in the frame: run on the owner thread after input and before the feed is
/// drained, so the epochs its edits close reach the GPU in the same frame.
pub(crate) trait Brush {
    /// One frame's brush work: the pen samples queued since the last frame (`pen`, oldest
    /// first, with the pen's hover), applied, and their epoch closed. The context's camera is
    /// the view as this frame's input left it, which the samples are projected through.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from; a store error in a stroke ends the stroke
    /// and shows in the overlay ([`Brush::status`]).
    fn step(&mut self, context: BrushContext<'_>, pen: &PenFrame) -> Result<BrushReport>;

    /// Whether the brush takes the stroke `down` starts, in the input phase, with the context's
    /// camera as the input so far left it: the one place the brush claims strokes. A stroke
    /// started with a barrel button held navigates whatever this says ([`crate::pen::route`]).
    /// By default every stroke navigates.
    fn route(&mut self, _context: BrushContext<'_>, _down: &PenSample) -> Route {
        Route::Navigate
    }

    /// Work kept off the input's latency path, run after the frame has been handed off and
    /// before the store's jobs, until `deadline`: warming the surface under a hovering pen.
    /// Returns whether some is left, which wants the next frame.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from.
    fn prepare(&mut self, _context: BrushContext<'_>, _deadline: Instant) -> Result<bool> {
        Ok(false)
    }

    /// Applies what the brush still holds and ends its stroke: the project is about to close.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from.
    fn flush(&mut self, _context: BrushContext<'_>) -> Result<()> {
        Ok(())
    }

    /// Changes the brush's settings; returns whether that changed anything, so the overlay
    /// wants a frame.
    fn change(&mut self, _change: BrushChange) -> bool {
        false
    }

    /// Sets the brush's radius to `pixels` on the screen: a scripted stroke's size.
    fn set_radius(&mut self, _pixels: f32) {}

    /// The overlay's lines about the brush: its settings and keys, its last stroke, its last
    /// error. None by default. Read on the latency path, so made beforehand.
    fn status(&self) -> &[String] {
        &[]
    }
}

/// No brush, for the frame loop's tests: the step does nothing, and every stroke navigates.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoBrush;

#[cfg(test)]
impl Brush for NoBrush {
    fn step(&mut self, _context: BrushContext<'_>, _pen: &PenFrame) -> Result<BrushReport> {
        Ok(BrushReport::default())
    }
}

/// A change to how the scene is shown, from the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LookChange {
    /// The wireframe over the shading, on or off.
    Wireframe,
    /// The next MatCap style.
    MatCap,
    /// The debug view of clusters in place of the shading, or back.
    Clusters,
}

impl LookChange {
    /// `look` changed by this.
    #[must_use]
    pub(crate) fn apply(self, mut look: Look) -> Look {
        match self {
            Self::Wireframe => look.shading.wireframe = !look.shading.wireframe,
            Self::MatCap => look.material = look.material.next(),
            Self::Clusters => {
                look.view = match look.view {
                    SceneView::Shaded => SceneView::Clusters,
                    SceneView::Clusters => SceneView::Shaded,
                };
            }
        }
        look
    }
}

/// What the uploads did with a frame's epochs, and what they staged for it.
#[derive(Debug, Default)]
pub(crate) struct UploadStatus {
    /// Whether uploads are in flight (the floor not all up, an epoch going up over several
    /// frames): the next frame is wanted.
    pub(crate) uploading: bool,
    /// Whether requests are pending (requested tiles still to come, a request readback still to
    /// read): the next frame is wanted.
    pub(crate) requesting: bool,
    /// The time (on the frame loop's clock, [`FrameUpdate::time`]) of the newest frame heard of
    /// this frame whose cut changed: frames are wanted until the morph it started is over.
    pub(crate) cut_changed: Option<Duration>,
    /// What the GPU holds, for the overlay's tiles line; `None` while it holds nothing.
    pub(crate) summary: Option<String>,
    /// The frame's staged uploads, which travel to the render thread in its
    /// [`FrameRequest`]; `None` if there is no GPU residency.
    pub(crate) frame: Option<FrameUploads>,
    /// What residency gives the frame's scene; `None` if there is no GPU residency.
    pub(crate) scene: Option<ResidentScene>,
    /// The scene's passes, from the first frame only, for the render thread to keep.
    pub(crate) passes: Option<FramePasses>,
    /// The number residency staged the frame as, which its cut readback comes back under;
    /// `None` if there is no GPU residency.
    pub(crate) staged_frame: Option<u64>,
    /// Residency's counts as the frame was staged; `None` if there is no GPU residency.
    pub(crate) stats: Option<ResidencyStats>,
    /// The cut readbacks heard this frame (of earlier ones), oldest first, for a scripted run's
    /// record.
    pub(crate) heard: Vec<CutHeard>,
}

/// A frame's cut, read back: what a scripted run's record keeps of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CutHeard {
    /// The number residency staged its frame as ([`UploadStatus::staged_frame`]).
    pub(crate) frame: u64,
    /// What it drew.
    pub(crate) scene: SceneRecord,
}

/// The uploads' place in the frame: GPU residency ([`crate::uploads::GpuResidency`]).
///
/// It takes each frame's drained epochs and stages the tiles the GPU should get into its
/// staging ring, on the owner thread (the store is read here, and only here), for the render
/// thread to encode ahead of the frame's passes.
pub(crate) trait Uploads {
    /// Applies `update.drained`, the epochs closed since the last frame (or a resync), and
    /// stages the frame's uploads, reading tiles from `store` until `update.deadline` (the
    /// floor goes up whole, whatever the deadline).
    ///
    /// # Errors
    ///
    /// If the store can't give a tile, or the GPU refuses an upload.
    fn apply(&mut self, update: FrameUpdate, store: &mut TileStore) -> Result<UploadStatus>;

    /// The render thread finished the last frame applied: `drawn` if its scene was drawn, so
    /// what the scene's passes carried (the bounds of the tiles that changed) is on the GPU.
    fn shown(&mut self, _drawn: bool) {}

    /// How long a level change morphs: frames keep coming that long after a cut change. Zero
    /// with no scene.
    fn morph(&self) -> Duration {
        Duration::ZERO
    }

    /// The latest cut readback's counts, for the overlay; `None` with no scene, or before the
    /// first.
    fn scene(&self) -> Option<&LodStats> {
        None
    }
}

/// What a frame gives the uploads.
#[derive(Debug)]
pub(crate) struct FrameUpdate {
    /// The epochs closed since the last frame, or a resync.
    pub(crate) drained: Drained,
    /// When staging should stop.
    pub(crate) deadline: Instant,
    /// Whether the view (the camera or the frame size) changed since the last frame: what the
    /// level choice asks for may change with it.
    pub(crate) moved: bool,
    /// The frame's time: its start, since the frame loop's clock began.
    pub(crate) time: Duration,
}

/// No GPU residency, for the frame loop's tests without a GPU: nothing is uploaded, and the
/// epochs are dropped.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoUploads;

#[cfg(test)]
impl Uploads for NoUploads {
    fn apply(&mut self, _update: FrameUpdate, _store: &mut TileStore) -> Result<UploadStatus> {
        Ok(UploadStatus::default())
    }
}

/// The uploads' deadline in a frame started at `now` whose work runs until `deadline`.
///
/// Half the way there: the uploads run before the hand-off, so their time is input latency;
/// the jobs after it get the rest (and at least a slice).
fn uploads_deadline(now: Instant, deadline: Instant) -> Instant {
    now + deadline.saturating_duration_since(now) / 2
}

/// How long the loop waits between polls of the device while the GPU is busy and there are no
/// jobs to run ([`Next::Gpu`]): a frame starts at most about this late after the GPU finishes
/// the one it waited for.
///
/// **Why poll, not wait:** a waiter thread blocked in a device poll would wake the loop the
/// moment the GPU finished, with no wake-ups meanwhile. But a poll runs the readback maps'
/// callbacks, and the frame loop's account of them (`crate::uploads`) needs no poll to fall
/// between two maps of one frame: the render thread asks for them one after the other as it
/// submits, so only a poll from the owner while the render thread holds no frame (or the
/// render thread's own submission) is safe. A third thread polling at any time would break that.
/// The cost of polling instead is a wake-up of the loop each millisecond while the GPU is behind
/// and there are no store jobs to run: none while the GPU keeps up, or while no frame is wanted.
pub(crate) const GPU_POLL: Duration = Duration::from_millis(1);

/// What the loop did while a frame waited for the GPU ([`Owner::wait_for_gpu`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuWait {
    /// The GPU has finished enough: ask [`Owner::next`] again, which may say to start the frame.
    Ready,
    /// Still busy; a slice of the store's jobs ran meanwhile. Look at the window's events, then
    /// wait again.
    Worked,
    /// Still busy, and nothing to do: wait [`GPU_POLL`], then poll again.
    Idle,
}

/// How long one frame's phases took on the owner thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct OwnerPhases {
    /// Waiting for the GPU to finish a frame before this one could start ([`Next::Gpu`]): from
    /// when it was due to when it started, jobs run meanwhile included. Zero when the GPU kept
    /// up.
    pub(crate) gpu_wait: Duration,
    /// Applying the queued input.
    pub(crate) input: Duration,
    /// The brush step.
    pub(crate) brush: Duration,
    /// Draining the stale-tile feed.
    pub(crate) feed: Duration,
    /// The uploads.
    pub(crate) uploads: Duration,
    /// Ticking and polling the project, after the last frame's hand-off.
    pub(crate) upkeep: Duration,
    /// The brush's work off the latency path (warming the surface under a hovering pen), after
    /// the last frame's hand-off.
    pub(crate) prepare: Duration,
    /// The store's jobs, after the last frame's hand-off.
    pub(crate) jobs: Duration,
}

/// What the stale-tile feed has delivered since the project opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FeedStats {
    /// Epochs drained.
    pub(crate) epochs: u64,
    /// Tiles they named, counted once per epoch that names them.
    pub(crate) keys: u64,
    /// Resyncs: drains that lost their epochs and must ask for everything again.
    pub(crate) resyncs: u64,
}

impl FeedStats {
    /// Counts what `drained` delivered.
    fn count(&mut self, drained: &Drained) {
        match drained {
            Drained::Epochs(epochs) => {
                let keys: u64 = epochs.iter().map(|epoch| epoch.keys.len() as u64).sum();
                self.epochs += epochs.len() as u64;
                self.keys += keys;
            }
            Drained::Resync(_) => self.resyncs += 1,
        }
    }
}

/// How the owner is set up: the window (or offscreen target) it draws for, and the run.
#[derive(Debug, Clone)]
pub(crate) struct Setup {
    /// The frame size; `None` while the window has none (minimized).
    pub(crate) size: Option<Size>,
    /// The display's refresh interval, which frames are paced to.
    pub(crate) interval: Duration,
    /// How frames reach the screen, for the overlay.
    pub(crate) presentation: Presentation,
    /// The GPU's name and backend, for the overlay.
    pub(crate) adapter: String,
    /// A scripted run's frame count (`--frames`): every frame is wanted until that many have
    /// been made, and the run is then [`Owner::done`]. `None` renders on demand for as long as
    /// the window is open.
    pub(crate) frames: Option<u64>,
    /// A camera path (`--script`): the camera follows it by the frame loop's clock, every frame
    /// is wanted until it is over, and the run is then [`Owner::done`]. Its frames are recorded.
    pub(crate) script: Option<Script>,
    /// What a scripted run's record says of its setting; `None` for no record.
    pub(crate) record: Option<RecordSetup>,
}

/// What a scripted run's record says of its setting, beside what the [`Setup`] says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordSetup {
    /// The project file, as given.
    pub(crate) project: String,
    /// Whether the render thread times the GPU's phases.
    pub(crate) gpu_timing: bool,
    /// Whether the level choice times its stages.
    pub(crate) lod_timing: bool,
    /// When the project began to open: the record's `open_ms` runs from it to the frame loop's
    /// clock.
    pub(crate) opening: Instant,
}

/// The owner thread's frame loop state: the project, the view, and what the overlay reports.
#[derive(Debug)]
pub(crate) struct Owner<B: Brush, U: Uploads> {
    /// The open project: its store is touched only on this thread.
    project: Project,
    /// The renderer's cursor in the stale-tile feed.
    cursor: StaleCursor,
    /// The brush seam.
    brush: B,
    /// The uploads seam.
    uploads: U,
    /// The view.
    camera: Camera,
    /// Turns pointer input into camera motion.
    navigator: Navigator,
    /// Pointer input since the last frame, applied in its input phase.
    input: Vec<Input>,
    /// Pen samples since the last frame, routed into `pen_frame` in its input phase.
    pen_queue: Vec<PenSample>,
    /// Which stroke is in progress, and where it goes.
    pen_router: PenRouter,
    /// Follows the pen stream into navigation's input as samples are queued.
    pen_navigation: PenNavigation,
    /// The pen samples the last input phase took, for the brush step, and the pen's hover.
    pen_frame: PenFrame,
    /// Where the pen stream comes from, for the overlay.
    pen_status: PenStatus,
    /// Decides when frames are made.
    scheduler: Scheduler,
    /// The frame size; `None` while there is none.
    size: Option<Size>,
    /// Frame pixels per overlay texel.
    overlay_scale: u32,
    /// How frames reach the screen.
    presentation: Presentation,
    /// The GPU's name and backend.
    adapter: String,
    /// A scripted run's frame count.
    limit: Option<u64>,
    /// Frames started.
    started: u64,
    /// Frames the render thread has finished.
    finished: u64,
    /// What the feed has delivered.
    feed: FeedStats,
    /// The render thread's phases for the last frame it finished.
    render: Option<DrawTimes>,
    /// The last frame's upkeep and jobs, which run after its hand-off: the next frame's
    /// overlay reports them.
    after_hand_off: AfterHandOff,
    /// Every frame's phases, in a scripted run only (an open-ended one would grow it forever).
    log: Option<PhaseLog>,
    /// The view (camera and frame size) the last frame was made with; `None` before the first.
    last_view: Option<(Camera, Size)>,
    /// Since when the next frame has waited for the GPU ([`Next::Gpu`]); `None` while it hasn't.
    gpu_wait_from: Option<Instant>,
    /// The frame loop's clock: frames are stamped with their time since it.
    clock: Instant,
    /// How the scene is shown.
    look: Look,
    /// Whether the last frame finished showed a scene (drawn or shaded again), which a frame
    /// shading it again needs.
    scene_shown: bool,
    /// Picks the surface under the pointer in the input phase; `None` if the loop doesn't pick.
    picker: Option<Picker>,
    /// Whether the image can accumulate: false once the renderer said a still frame was
    /// filtered instead (its histories don't fit the device), until it restarts them.
    accumulates: bool,
    /// The camera path the camera follows, in a scripted run.
    script: Option<Script>,
    /// Whether the script is over: a frame has been started at or past its end.
    script_over: bool,
    /// Every frame's record, in a scripted run.
    recorder: Option<Recorder>,
    /// The record's setting, if the run is recorded.
    run_info: Option<RunInfo>,
    /// The frame with the render thread and the oldest pen sample whose dabs it applied, until
    /// it is shown: its latency is measured then.
    latency_from: Option<(u64, SampleTimes)>,
    /// The last frame's pen latency, sample to present submitted, for the overlay; `None` until
    /// a frame applies dabs.
    latency: Option<Duration>,
    /// How far into a scripted run its pen samples have been queued: those up to here are.
    script_pen: Duration,
    /// Whether the brush is stroking, for the overlay's pick.
    stroking: Stroking,
}

/// Whether the brush is stroking, as the overlay's pick needs to know.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Stroking {
    /// Whether the last brush step left work for the next frame (dabs carried, a stroke
    /// ending): the brush is still stroking.
    busy: bool,
    /// Whether a frame skipped the overlay's pick because the brush was stroking: the pick is
    /// made in a frame wanted once the stroke is over.
    pick_deferred: bool,
}

/// Every frame's phase times in a scripted run, for the summary printed at its end.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PhaseLog {
    /// The owner thread's phases, frame by frame: each frame's upkeep and jobs are its own,
    /// recorded once they have run.
    pub(crate) owner: Vec<OwnerPhases>,
    /// The render thread's, for every frame it finished, in order.
    pub(crate) render: Vec<DrawTimes>,
}

impl PhaseLog {
    /// One line per phase: its median, 99th percentile and longest, in milliseconds, over
    /// every frame but the first (which uploads the floor, shown on its own).
    pub(crate) fn lines(&self) -> Vec<String> {
        let rest = self.owner.get(1..).unwrap_or_default();
        let render = self.render.get(1..).unwrap_or_default();
        let owner_phase = |name: &str, phase: fn(&OwnerPhases) -> Duration| {
            let times: Vec<Duration> = rest.iter().map(phase).collect();
            spread(name, times)
        };
        let render_phase = |name: &str, phase: fn(&DrawTimes) -> Duration| {
            let times: Vec<Duration> = render.iter().map(phase).collect();
            spread(name, times)
        };
        let mut lines = Vec::new();
        if let Some(first) = self.owner.first() {
            let encode = self.render.first().map_or(Duration::ZERO, |r| r.encode);
            lines.push(format!(
                "first frame: uploads {:.2} ms (owner), encode and submit {:.2} ms (render)",
                first.uploads.as_secs_f64() * 1e3,
                encode.as_secs_f64() * 1e3
            ));
        }
        lines.push(format!("later frames ({} of them), ms:", rest.len()));
        lines.extend([
            owner_phase("input", |p| p.input),
            owner_phase("brush", |p| p.brush),
            owner_phase("feed", |p| p.feed),
            owner_phase("uploads", |p| p.uploads),
            owner_phase("upkeep", |p| p.upkeep),
            owner_phase("prepare", |p| p.prepare),
            owner_phase("jobs", |p| p.jobs),
            render_phase("acquire", |r| r.acquire),
            render_phase("encode", |r| r.encode),
            render_phase("present", |r| r.present),
        ]);
        lines
    }
}

/// `name` and the median, 99th percentile and longest of `times`, in milliseconds.
fn spread(name: &str, mut times: Vec<Duration>) -> String {
    times.sort_unstable();
    let at = |q: usize| {
        let index = (times.len().saturating_sub(1) * q).div_ceil(100);
        times
            .get(index)
            .map_or(0.0, |time| time.as_secs_f64() * 1e3)
    };
    format!(
        "  {name:<8} p50 {:.3}  p99 {:.3}  max {:.3}",
        at(50),
        at(99),
        at(100)
    )
}

/// What a frame did after handing itself off, reported by the next frame's overlay.
#[derive(Debug, Clone, Copy, Default)]
struct AfterHandOff {
    /// How long the upkeep took.
    upkeep: Duration,
    /// How long the brush's preparation took.
    prepare: Duration,
    /// How long the jobs took.
    jobs: Duration,
    /// What the jobs did.
    report: JobsReport,
}

impl<B: Brush, U: Uploads> Owner<B, U> {
    /// The frame loop over `project`, with the brush and uploads given, subscribed to its
    /// stale-tile feed, wanting its first frame at `now`.
    pub(crate) fn new(
        mut project: Project,
        brush: B,
        uploads: U,
        setup: Setup,
        now: Instant,
    ) -> Self {
        let cursor = project.store().subscribe_stale();
        let mut scheduler = Scheduler::new(setup.interval, now).with_morph(uploads.morph());
        scheduler.want(Reason::Exposed);
        let scripted = setup.frames.is_some() || setup.script.is_some();
        if scripted {
            scheduler.want(Reason::Scripted);
        }
        let run_info = setup.record.map(|record| RunInfo {
            project: record.project,
            adapter: setup.adapter.clone(),
            size: setup
                .size
                .map_or((0, 0), |size| (size.width(), size.height())),
            gpu_timing: record.gpu_timing,
            lod_timing: record.lod_timing,
            open: now.saturating_duration_since(record.opening),
            clock_unix: crate::record::since_epoch(now),
            refresh: setup.interval,
        });
        Self {
            project,
            cursor,
            brush,
            uploads,
            camera: Camera::default(),
            navigator: Navigator::default(),
            input: Vec::new(),
            pen_queue: Vec::new(),
            pen_router: PenRouter::default(),
            pen_navigation: PenNavigation::default(),
            pen_frame: PenFrame::default(),
            pen_status: PenStatus::default(),
            scheduler,
            size: setup.size,
            overlay_scale: 1,
            presentation: setup.presentation,
            adapter: setup.adapter,
            limit: setup.frames,
            started: 0,
            finished: 0,
            feed: FeedStats::default(),
            render: None,
            after_hand_off: AfterHandOff::default(),
            log: scripted.then(PhaseLog::default),
            last_view: None,
            gpu_wait_from: None,
            clock: now,
            look: Look::default(),
            scene_shown: false,
            picker: None,
            accumulates: true,
            recorder: setup.script.as_ref().map(|_| Recorder::default()),
            script: setup.script,
            script_over: false,
            run_info,
            latency_from: None,
            latency: None,
            script_pen: Duration::ZERO,
            stroking: Stroking::default(),
        }
    }

    /// A recorded scripted run's record lines ([`crate::record`]); empty for any other run.
    pub(crate) fn record_lines(&self) -> Vec<String> {
        match (&self.recorder, &self.run_info, &self.script) {
            (Some(recorder), Some(info), Some(script)) => recorder.lines(info, script),
            _ => Vec::new(),
        }
    }

    /// Changes how the scene is shown, and wants a frame to show it.
    pub(crate) fn change_look(&mut self, change: LookChange) {
        self.look = change.apply(self.look);
        self.scheduler.want(Reason::Look);
    }

    /// Changes the brush's settings, and wants a frame for the overlay if that changed them.
    pub(crate) fn change_brush(&mut self, change: BrushChange) {
        if self.brush.change(change) {
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// How the scene is shown.
    #[cfg(test)]
    pub(crate) const fn look(&self) -> Look {
        self.look
    }

    /// Queues pointer input for the next frame's input phase, and wants that frame if the
    /// input may move the camera.
    pub(crate) fn input(&mut self, input: Input) {
        if input == Input::Cancel {
            // Navigation forgets the pen's buttons too: the next sample presses them again.
            self.pen_navigation = PenNavigation::default();
        }
        if self.navigator.may_move(&self.input, &input) {
            self.scheduler.want(Reason::Input);
        }
        if let Input::Move { at } | Input::Press { at, .. } = input {
            self.point(Some(at));
        }
        self.input.push(input);
    }

    /// Queues a pen sample for the next frame's input phase, where it is routed.
    ///
    /// **The contract:** who a sample belongs to is decided in the input phase, not here: a
    /// stroke (`Down` through `Up`) goes whole to navigation or to the brush, as decided at its
    /// `Down` ([`Brush::route`], [`crate::pen::route`]), and a sample out of a stroke to both.
    /// Here the sample only moves the pointer picking follows (away at a `Leave`), wants a frame
    /// if it belongs to a stroke, and is queued; consecutive samples out of a stroke are
    /// coalesced, the latest replacing the one before (only the pen's latest state out of a
    /// stroke counts, for the pointer, the barrel buttons and the hover), so the queue stays
    /// short while no frame comes.
    pub(crate) fn pen(&mut self, sample: PenSample) {
        self.point((sample.phase != input::Phase::Leave).then_some(sample.at));
        if sample.phase.in_stroke() {
            self.scheduler.want(Reason::Input);
        }
        match self.pen_queue.last_mut() {
            Some(last) if !last.phase.in_stroke() && !sample.phase.in_stroke() => *last = sample,
            _ => self.pen_queue.push(sample),
        }
    }

    /// The pen samples the last input phase took, and the pen's hover: what the brush step is
    /// handed.
    #[cfg(test)]
    pub(crate) const fn pen_frame(&self) -> &PenFrame {
        &self.pen_frame
    }

    /// Where the pen stream comes from, as the overlay shows it.
    pub(crate) const fn pen_status(&self) -> &PenStatus {
        &self.pen_status
    }

    /// Where the pen stream comes from now, for the overlay.
    pub(crate) fn set_pen_status(&mut self, status: PenStatus) {
        self.pen_status = status;
    }

    /// The pointer left the window: nothing is under it to pick.
    pub(crate) fn pointer_left(&mut self) {
        self.point(None);
    }

    /// The pointer is at `at` (`None`: outside the window): the picker hears of it, and a frame
    /// is wanted for the overlay's pick line if that moved it.
    fn point(&mut self, at: Option<Pixel>) {
        if let Some(picker) = &mut self.picker
            && picker.point(at)
        {
            self.scheduler.want(Reason::Pick);
        }
    }

    /// This frame loop picking the surface under the pointer with `picker` ([`crate::pick`]).
    #[must_use]
    pub(crate) fn with_picker(mut self, picker: Picker) -> Self {
        self.picker = Some(picker);
        self
    }

    /// The picker, if the loop picks.
    #[cfg(test)]
    pub(crate) const fn picker(&self) -> Option<&Picker> {
        self.picker.as_ref()
    }

    /// The frame size changed to `size` (`None`: the window has none, so no frames).
    pub(crate) fn resize(&mut self, size: Option<Size>) {
        if size != self.size {
            self.size = size;
            self.expose();
        }
    }

    /// The window's scale factor is `scale`: the overlay is drawn at it, rounded, at least 1.
    pub(crate) fn set_scale(&mut self, scale: f64) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a scale factor is a small positive number, clamped to 1..=8"
        )]
        let overlay_scale = scale.round().clamp(1.0, 8.0) as u32;
        if overlay_scale != self.overlay_scale {
            self.overlay_scale = overlay_scale;
            self.expose();
        }
    }

    /// The display's refresh interval changed (the window moved to another display).
    pub(crate) fn set_interval(&mut self, interval: Duration) {
        self.scheduler.set_interval(interval);
    }

    /// The window needs drawing again (uncovered, or asked to redraw).
    pub(crate) fn expose(&mut self) {
        self.scheduler.want(Reason::Exposed);
    }

    /// What to do at `now`: as the scheduler says, but no frame while there is no frame size
    /// (a minimised window) or once a scripted run has made its frames (or its script is over);
    /// upkeep stays due on its own clock meanwhile.
    ///
    /// A wait for the GPU ends when the answer is neither [`Next::Gpu`] nor [`Next::Frame`]
    /// (a minimised window, nothing wanted any more): the next frame's
    /// [`OwnerPhases::gpu_wait`] counts only a wait that led to it.
    pub(crate) fn next(&mut self, now: Instant) -> Next {
        let next = self.scheduler.next(now);
        let no_frames = self.size.is_none()
            || self.limit.is_some_and(|limit| self.started >= limit)
            || self.script_over;
        let next = match next {
            Next::Frame { .. } | Next::Wait { .. } | Next::Gpu if no_frames => Next::Idle {
                upkeep: self.scheduler.upkeep_due(),
            },
            Next::Frame { .. }
            | Next::Wait { .. }
            | Next::Gpu
            | Next::Render
            | Next::Idle { .. } => next,
        };
        if !matches!(next, Next::Gpu | Next::Frame { .. }) {
            self.gpu_wait_from = None;
        }
        next
    }

    /// How long the frame starting at `now` waited for the GPU ([`OwnerPhases::gpu_wait`]); the
    /// wait is over.
    fn gpu_waited(&mut self, now: Instant) -> Duration {
        self.gpu_wait_from
            .take()
            .map_or(Duration::ZERO, |from| now.saturating_duration_since(from))
    }

    /// One step of waiting for the GPU, at `now`, when [`Owner::next`] said [`Next::Gpu`]:
    /// hears from `finished` the newest frame the GPU has finished
    /// ([`crate::render_thread::RenderThread::gpu_finished`], which polls the device without
    /// waiting); if it is still too far behind, runs the upkeep if it is due, then a slice of the
    /// store's jobs, if there are any, until [`GPU_POLL`] from now. Both loops (offscreen and
    /// the window's) call it, and map what it returns to a sleep or the event loop's control
    /// flow.
    ///
    /// The wait counts toward the next frame's [`OwnerPhases::gpu_wait`], the jobs run meanwhile
    /// included; they are not added to the overlay's jobs line, which reports the jobs after the
    /// last hand-off.
    ///
    /// # Errors
    ///
    /// As `finished`, [`Project::tick`], [`Project::poll`] and the store's jobs.
    pub(crate) fn wait_for_gpu(
        &mut self,
        now: Instant,
        finished: impl FnOnce() -> Result<u64>,
    ) -> Result<GpuWait> {
        self.gpu_wait_from.get_or_insert(now);
        if !self.scheduler.polled(finished()?) {
            return Ok(GpuWait::Ready);
        }
        if now >= self.scheduler.upkeep_due() {
            self.upkeep(now)?;
        }
        if !self.project.store().has_jobs() {
            return Ok(GpuWait::Idle);
        }
        self.project
            .store()
            .run_jobs((now + GPU_POLL).max(Instant::now() + JobsReport::SLICE_TARGET))
            .context("running the store's jobs")?;
        // What the jobs closed is the next frame's: it is wanted already, waiting for the GPU.
        if self.project.store().has_jobs() || self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        Ok(GpuWait::Worked)
    }

    /// Whether a scripted run has made all its frames (or run its script through) and the
    /// render thread has finished them.
    pub(crate) fn done(&self) -> bool {
        self.limit.is_some_and(|limit| self.finished >= limit)
            || (self.script_over && self.finished >= self.started)
    }

    /// Frames the render thread has finished.
    pub(crate) const fn finished(&self) -> u64 {
        self.finished
    }

    /// Makes a frame started at `now`, hands what the render thread is to draw to `hand_off`,
    /// then runs the upkeep and the store's jobs until `deadline` (at least one slice of jobs,
    /// however late it is). Call it when [`Owner::next`] says [`Next::Frame`], and report back
    /// with [`Owner::shown`] when the render thread has drawn it.
    ///
    /// # Errors
    ///
    /// If the brush, the uploads, the hand-off, the project's upkeep or the store's jobs fail.
    pub(crate) fn frame(
        &mut self,
        now: Instant,
        deadline: Instant,
        hand_off: impl FnOnce(FrameRequest) -> Result<()>,
    ) -> Result<()> {
        let size = self
            .size
            .context("a frame was started with no frame size")?;
        let reasons = self.scheduler.started(now);
        self.started += 1;
        let mut phases = OwnerPhases {
            gpu_wait: self.gpu_waited(now),
            upkeep: self.after_hand_off.upkeep,
            prepare: self.after_hand_off.prepare,
            jobs: self.after_hand_off.jobs,
            ..OwnerPhases::default()
        };
        let mut lap = Lap::new();
        let time = now.saturating_duration_since(self.clock);

        let segment = self.follow_script(time, size);
        self.take_input(size)?;
        phases.input = lap.next();

        let brushed = self.brush_step(size)?;
        self.stroking.busy = brushed.busy;
        phases.brush = lap.next();

        let drained = self.project.store().drain_stale(&self.cursor);
        self.feed.count(&drained);
        phases.feed = lap.next();

        let this_view = (self.camera, size);
        let moved = self.last_view.replace(this_view) != Some(this_view);
        let update = FrameUpdate {
            drained,
            deadline: uploads_deadline(now, deadline),
            moved,
            time,
        };
        let mut uploads = self
            .uploads
            .apply(update, self.project.store())
            .context("uploading tiles")?;
        phases.uploads = lap.next();
        self.record_staged((time, segment, moved), &mut uploads);
        if let Some(recorder) = &mut self.recorder {
            recorder.brush(self.started, &brushed);
        }
        let shading_only = Reasons::default()
            .with(Reason::Look)
            .with(Reason::Pick)
            .with(Reason::Overlay);
        let reshade = self.scene_shown
            && !reasons.is_empty()
            && reasons.within(shading_only)
            && !moved
            && quiet(&uploads);
        let resident = uploads.scene.take().filter(|_| !reshade);
        let scene = resident.and_then(|resident| {
            Some(SceneRequest {
                resident,
                view: self.camera.view(size)?,
                time,
                moving: moved,
            })
        });
        self.want_accumulation(reasons, &uploads, (scene.is_some() || reshade, moved));

        let overlay = self.overlay(size, phases, uploads.summary.as_deref());
        hand_off(FrameRequest {
            number: self.started,
            size,
            overlay_scale: self.overlay_scale,
            overlay,
            uploads: uploads.frame.take(),
            scene,
            reshade,
            look: self.look,
            passes: uploads.passes.take(),
        })?;
        lap.next();

        let preparing = self.after_hand_off(now, (size, deadline), &mut lap)?;
        self.log_phases(OwnerPhases {
            upkeep: self.after_hand_off.upkeep,
            prepare: self.after_hand_off.prepare,
            jobs: self.after_hand_off.jobs,
            ..phases
        });

        self.want_next(&uploads);
        if brushed.busy {
            self.scheduler.want(Reason::Input);
        }
        if self.stroking.pick_deferred && !self.brushing() {
            // The stroke is over: the next frame makes the pick this one skipped.
            self.scheduler.want(Reason::Pick);
        }
        if preparing {
            self.scheduler.want(Reason::Converging);
        }
        Ok(())
    }

    /// The overlay's lines for the frame being made, of `size`, its owner thread's `phases` so
    /// far, with the uploads' `tiles` line.
    fn overlay(&self, size: Size, phases: OwnerPhases, tiles: Option<&str>) -> Vec<String> {
        let brush = self.brush.status();
        let snapshot = Snapshot {
            frame: self.started,
            refresh_hz: self.scheduler.interval().as_secs_f64().recip(),
            presentation: self.presentation,
            adapter: &self.adapter,
            size,
            owner: phases,
            render: self.render,
            feed: self.feed,
            jobs: self.after_hand_off.report,
            tiles,
            scene: self.uploads.scene(),
            camera: self.camera,
            look: self.look,
            pick: self.picker.as_ref().map(Picker::report),
            pen: &self.pen_status,
            filter_only: !self.accumulates,
            brush,
            latency: self.latency,
        };
        overlay::lines(&snapshot)
    }

    /// A frame of `size` started at `now`, once handed off: the upkeep, the brush's preparation,
    /// then the store's jobs until `deadline` (at least a slice of them), each timed by `lap`
    /// into [`Owner::after_hand_off`]. Returns whether the brush has preparation left.
    ///
    /// # Errors
    ///
    /// As [`Project::tick`], [`Project::poll`], [`Brush::prepare`] and the store's jobs.
    fn after_hand_off(
        &mut self,
        now: Instant,
        (size, deadline): (Size, Instant),
        lap: &mut Lap,
    ) -> Result<bool> {
        self.project.tick(now).context("ticking the project")?;
        self.project.poll().context("polling the project")?;
        self.after_hand_off.upkeep = lap.next();

        let preparing = self
            .with_brush(size, |brush, context| brush.prepare(context, deadline))
            .context("preparing the brush")?;
        self.after_hand_off.prepare = lap.next();

        let deadline = deadline.max(Instant::now() + JobsReport::SLICE_TARGET);
        self.after_hand_off.report = self
            .project
            .store()
            .run_jobs(deadline)
            .context("running the store's jobs")?;
        self.after_hand_off.jobs = lap.next();
        Ok(preparing)
    }

    /// Calls the brush with its context for a frame of `size`: the project, picking's spatial
    /// index and the camera.
    fn with_brush<R>(&mut self, size: Size, call: impl FnOnce(&mut B, BrushContext<'_>) -> R) -> R {
        let context = BrushContext {
            project: &mut self.project,
            index: self.picker.as_mut().map(Picker::index_mut),
            camera: &self.camera,
            size,
        };
        call(&mut self.brush, context)
    }

    /// The brush phase of a frame of `size`: the brush steps over the project with the camera
    /// and the pen samples the input phase left. Its edits move the surface, so the next pick
    /// is made afresh; the oldest sample it applied is kept until the frame is shown, for its
    /// latency.
    ///
    /// # Errors
    ///
    /// If the brush fails in a way the app can't go on from.
    fn brush_step(&mut self, size: Size) -> Result<BrushReport> {
        let pen = std::mem::take(&mut self.pen_frame);
        let stepped = self.with_brush(size, |brush, context| brush.step(context, &pen));
        self.pen_frame = pen;
        let report = stepped.context("the brush step")?;
        if report.dabs_applied > 0
            && let Some(picker) = &mut self.picker
        {
            picker.surface_changed();
        }
        self.latency_from = report.oldest.map(|oldest| (self.started, oldest));
        Ok(report)
    }

    /// In a recorded run, files the frame just started (at `time`, in `segment`, its view
    /// `moved` or not) and what its uploads said: residency's counts as it was staged, and the
    /// cut heard during it, which is an earlier frame's.
    fn record_staged(
        &mut self,
        (time, segment, moved): (Duration, Option<Segment>, bool),
        uploads: &mut UploadStatus,
    ) {
        let Some(recorder) = &mut self.recorder else {
            return;
        };
        recorder.started(self.started, time, segment, moved);
        recorder.staged(self.started, uploads.staged_frame, uploads.stats.as_ref());
        for cut in std::mem::take(&mut uploads.heard) {
            recorder.cut(cut);
        }
    }

    /// Logs the frame just made's owner thread `phases`, its upkeep and jobs included, in a
    /// scripted run.
    fn log_phases(&mut self, phases: OwnerPhases) {
        if let Some(log) = &mut self.log {
            log.owner.push(phases);
        }
        if let Some(recorder) = &mut self.recorder {
            recorder.owner(self.started, phases);
        }
    }

    /// In a scripted run, puts the camera where the script has it `time` into the run (the
    /// queued pointer input is dropped: the script drives the view), queues the script's pen
    /// samples taken since the last frame, each where its point projects in a view of `size` at
    /// its own time, and marks the script over once `time` reaches its end; returns the segment
    /// running. `None` with no script.
    fn follow_script(&mut self, time: Duration, size: Size) -> Option<Segment> {
        let script = self.script.as_ref()?;
        let moment = script.at(time);
        let due = script.pen(self.script_pen, time);
        self.script_pen = time;
        self.camera = moment.pose.camera();
        self.input.clear();
        self.script_over |= moment.over;
        let projected = due.into_iter().filter_map(|sample| {
            let view = sample.camera.view(size)?;
            let (x, y, _) = view.pixel(sample.point)?;
            Some((sample, Pixel { x, y }))
        });
        let samples: Vec<(ScriptPen, Pixel)> = projected.collect();
        for (sample, at) in samples {
            if let Some(radius) = sample.radius {
                self.brush.set_radius(radius);
            }
            let taken = self.clock + sample.time;
            self.pen(PenSample {
                phase: sample.phase,
                tool: input::Tool::Pen,
                at,
                pressure: sample.pressure,
                distance: None,
                tilt: None,
                roll: None,
                buttons: input::Buttons::default(),
                timestamp: taken,
                received: taken,
            });
        }
        Some(moment.segment)
    }

    /// The input phase of a frame of `size`: the input queued since the last frame moves the
    /// camera, the pen samples queued become the frame's [`PenFrame`], then the surface under the
    /// pointer is picked if the pointer or the view changed, unless the brush is stroking
    /// ([`Owner::brushing`]; see the module notes).
    ///
    /// # Errors
    ///
    /// If the pick can't evaluate a tile.
    fn take_input(&mut self, size: Size) -> Result<()> {
        let arcball = Arcball::new(size);
        let height = f64::from(size.height());
        let input = std::mem::take(&mut self.input);
        for event in input {
            self.navigator
                .apply(event, &mut self.camera, &arcball, height);
        }
        self.route_pen(&arcball, size);
        self.stroking.pick_deferred = self.brushing();
        if let Some(picker) = &mut self.picker
            && !self.stroking.pick_deferred
        {
            let view = self.camera.view(size);
            picker
                .update(self.project.store(), view.as_ref())
                .context("picking")?;
        }
        Ok(())
    }

    /// Whether the brush is stroking: this frame's pen samples hold a stroke of the brush's, a
    /// stroke of its goes on past them, or the last brush step left work (dabs carried, a stroke
    /// ending) for the next.
    fn brushing(&self) -> bool {
        let routed = self
            .pen_frame
            .strokes()
            .iter()
            .any(|(_, route)| *route == Route::Brush);
        routed || self.pen_router.stroke() == Some(Route::Brush) || self.stroking.busy
    }

    /// Routes the pen samples queued since the last frame into the frame's [`PenFrame`], in
    /// order: navigation's (out of a stroke, or in a stroke routed to it) move the camera, whose
    /// viewport `arcball` covers and is `size`, as they come, so a later `Down` is routed with
    /// the view the earlier samples left.
    fn route_pen(&mut self, arcball: &Arcball, size: Size) {
        let height = f64::from(size.height());
        self.pen_frame.clear();
        let queued = std::mem::take(&mut self.pen_queue);
        let samples = queued.into_iter();
        for sample in samples {
            let brush = &mut self.brush;
            let context = BrushContext {
                project: &mut self.project,
                index: self.picker.as_mut().map(Picker::index_mut),
                camera: &self.camera,
                size,
            };
            let route = self
                .pen_router
                .route(&sample, |down| brush.route(context, down));
            if route != Some(Route::Brush) {
                let mut navigation = self.pen_navigation;
                navigation.inputs(&sample, |input| {
                    self.navigator
                        .apply(input, &mut self.camera, arcball, height);
                });
                self.pen_navigation = navigation;
            }
            self.pen_frame.push(sample, route);
        }
    }

    /// Wants the frames the image needs to accumulate its samples after a frame wanted for
    /// `reasons`, whose uploads said `uploads`, which shows a scene or not and moved the view or
    /// not (`(showing, moved)`; the renderer's accumulation, `render::aa`).
    ///
    /// - A frame that moved the view: the next one, still, which restarts the image; the restart
    ///   then asks for the rest (`Owner::shown`).
    /// - A frame that may change the scene (store work converging, uploads, requests, a morph, a
    ///   script, anything staged) can reset pixels the renderer only notices on the frame that
    ///   samples them again: `2 · SAMPLES` frames, one cycle of jitters to notice, one to converge.
    /// - Any other frame: none. One that restarts the image (a look change, a lost frame, a new
    ///   size) asks for `SAMPLES − 1` more as it is shown, from the renderer's report, so input
    ///   that didn't move the view, or the window exposed, costs nothing while the image goes on.
    ///   The pick line alone shades nothing when the image holds the look already
    ///   (`render::Scene::Reshade` shows the accumulated image as it is); while accumulating, a
    ///   pick comes with `Accumulating`, so it draws the next sample and continues it.
    ///
    /// Neither the debug view of clusters nor an image too large to accumulate asks for any.
    fn want_accumulation(
        &mut self,
        reasons: Reasons,
        uploads: &UploadStatus,
        (showing, moved): (bool, bool),
    ) {
        if !showing || self.look.view != SceneView::Shaded || !self.accumulates {
            return;
        }
        if moved {
            self.scheduler.accumulate(1);
            return;
        }
        let changing = [
            Reason::Converging,
            Reason::Uploads,
            Reason::Requests,
            Reason::Morphing,
            Reason::Scripted,
        ]
        .into_iter()
        .any(|reason| reasons.contains(reason))
            || !quiet(uploads);
        if changing {
            self.scheduler.accumulate(2 * aa::SAMPLES);
        }
    }

    /// Tells the scheduler what, after a frame whose uploads said `uploads`, wants the next:
    /// store jobs or undrained epochs, uploads in flight, requests pending, a cut change (until
    /// its morph is over), a script's frames still to make.
    fn want_next(&mut self, uploads: &UploadStatus) {
        if self.project.store().has_jobs() || self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        if uploads.uploading {
            self.scheduler.want(Reason::Uploads);
        }
        if uploads.requesting {
            self.scheduler.want(Reason::Requests);
        }
        if let Some(changed) = uploads.cut_changed {
            self.scheduler.cut_changed(self.clock + changed);
        }
        let scripted = self.limit.is_some_and(|limit| self.started < limit)
            || (self.script.is_some() && !self.script_over);
        if scripted {
            self.scheduler.want(Reason::Scripted);
        }
    }

    /// The render thread finished a frame.
    pub(crate) fn shown(&mut self, mut presented: Presented) {
        let gpu = std::mem::take(&mut presented.gpu);
        let latency = self
            .latency_from
            .take_if(|(frame, _)| *frame == presented.number)
            .map(|(_, oldest)| Latency {
                taken: presented.submitted.saturating_duration_since(oldest.taken),
                received: presented
                    .submitted
                    .saturating_duration_since(oldest.received),
            });
        if let Some(latency) = latency {
            self.latency = Some(latency.taken);
        }
        if let Some(recorder) = &mut self.recorder {
            let at = Instant::now().saturating_duration_since(self.clock);
            let submitted = presented.submitted.saturating_duration_since(self.clock);
            recorder.shown(presented.number, presented.times, at);
            recorder.latency(presented.number, submitted, latency);
            recorder.dropped(presented.number, presented.dropped);
            for times in gpu {
                recorder.gpu(times);
            }
        }
        self.scheduler.gpu_finished(presented.gpu_finished);
        self.uploads.shown(presented.scene);
        if presented.aa.filtered_instead {
            self.accumulates = false;
        } else if presented.aa.restarted {
            // A restart (the camera stopped, a look change, a lost frame, a new size) wants the
            // rest of its samples; whatever the countdown said, it starts again here.
            self.accumulates = true;
            self.scheduler.accumulate(aa::SAMPLES - 1);
        }
        self.finished += 1;
        self.scene_shown = presented.shown;
        self.render = Some(presented.times);
        if let Some(log) = &mut self.log {
            log.render.push(presented.times);
        }
        self.scheduler.shown(presented.outcome);
    }

    /// GPU times read back as the render thread stopped (`render_thread::Stopped::gpu`): the
    /// last frames', for a recorded run.
    pub(crate) fn last_gpu_times(&mut self, last: Vec<render::timing::GpuTimes>) {
        if let Some(recorder) = &mut self.recorder {
            for times in last {
                recorder.gpu(times);
            }
        }
    }

    /// Every frame's phases so far, in a scripted run; `None` in an open-ended one.
    pub(crate) const fn phase_log(&self) -> Option<&PhaseLog> {
        self.log.as_ref()
    }

    /// Upkeep with no frame, at `now`: the project's recovery writer ticked and polled.
    ///
    /// # Errors
    ///
    /// As [`Project::tick`] and [`Project::poll`].
    pub(crate) fn upkeep(&mut self, now: Instant) -> Result<()> {
        self.project.tick(now).context("ticking the project")?;
        self.project.poll().context("polling the project")?;
        self.scheduler.upkept(now);
        // A poll can install prefetched tiles, closing epochs: the next frame takes them.
        if self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        Ok(())
    }

    /// Closes the project: lets the brush apply what it holds and end its stroke, leaves the
    /// feed, then closes it, keeping its recovery file only if it has edits no save holds (the
    /// app can't ask yet).
    ///
    /// # Errors
    ///
    /// As [`Brush::flush`] and [`Project::close`].
    pub(crate) fn close(mut self) -> Result<()> {
        let flushed = match self.size.or_else(|| self.last_view.map(|(_, size)| size)) {
            Some(size) => {
                let flushed = self.with_brush(size, B::flush);
                flushed.context("ending the brush's stroke")
            }
            None => Ok(()),
        };
        let Self {
            mut project,
            cursor,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        let keep = project.has_unsaved_edits();
        let closed = project.close(keep).context("closing the project");
        flushed.and(closed)
    }

    /// Whether the stale-tile feed holds epochs this loop hasn't drained (or a resync): work
    /// after the drain (the jobs, a poll) closed some, and a frame must take them.
    pub(crate) fn has_stale(&mut self) -> bool {
        self.project.store().has_stale(&self.cursor)
    }

    /// The camera.
    #[cfg(test)]
    pub(crate) const fn camera(&self) -> &Camera {
        &self.camera
    }

    /// The brush.
    #[cfg(test)]
    pub(crate) const fn brush(&self) -> &B {
        &self.brush
    }

    /// The uploads seam.
    pub(crate) const fn uploads(&self) -> &U {
        &self.uploads
    }

    /// The project.
    #[cfg(test)]
    pub(crate) const fn project(&mut self) -> &mut Project {
        &mut self.project
    }
}

/// Whether the uploads left a frame nothing new to draw.
///
/// That is nothing staged in it, in flight or requested, and no cut change heard (so no morph
/// started): shading the last scene again then shows what a whole frame would.
fn quiet(uploads: &UploadStatus) -> bool {
    uploads.frame.as_ref().is_none_or(FrameUploads::is_empty)
        && !uploads.uploading
        && !uploads.requesting
        && uploads.cut_changed.is_none()
        && uploads.passes.is_none()
}

/// Times consecutive phases: each [`Lap::next`] is the time since the last.
#[derive(Debug)]
pub(crate) struct Lap {
    /// When the current phase began.
    since: Instant,
}

impl Lap {
    /// A lap starting now.
    pub(crate) fn new() -> Self {
        Self {
            since: Instant::now(),
        }
    }

    /// The time since the last call (or the start), and starts the next phase.
    pub(crate) fn next(&mut self) -> Duration {
        let now = Instant::now();
        let took = now - self.since;
        self.since = now;
        took
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
