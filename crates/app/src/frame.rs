//! The frame loop, on the store's owner thread (the M1 spec's "Threads": the owner thread is
//! the frame loop).
//!
//! [`Owner`] holds the project, so it lives on the one thread that may touch the store, and
//! makes each frame in a fixed order:
//!
//! 1. **input**: the pointer input queued since the last frame moves the camera (not while a brush
//!    stroke is open: a stroke is drawn through one camera, which its record keeps), about its
//!    pivot ([`crate::camera`]): a navigation drag that starts over the model makes the surface
//!    under the pointer the pivot first, each brush step makes the start of a stroke it began the
//!    pivot, and F (and opening the project) frames the model, its centre the pivot; the camera's
//!    boundary is grown around what the brush edited once it is idle; the pen samples queued since
//!    the last frame are routed ([`crate::pen::PenRouter`]: a stroke to navigation or the brush,
//!    the rest to both), navigation's moving the camera after the mouse's input, and become the
//!    frame's [`PenFrame`]; then the surface under the pointer is picked for the overlay if the
//!    pointer or the view changed ([`crate::pick`]), except while the brush is stroking (the pick
//!    is on the pen's latency path, and the stroke moves the surface under it anyway): it is made
//!    once the stroke is over, in a frame wanted for it;
//! 2. **brush**: the brush's step ([`Brush`]; the app's is [`crate::sculpting::Sculpting`]), which
//!    sees the camera as the input left it and the frame's pen samples (those that arrived during
//!    the input phase too), applies as many of its queued dabs as the frame's time left allows
//!    ([`BrushBudget`]) and closes their epoch, so they are drained just after and drawn in this
//!    frame; what it reports ([`BrushReport`], with what it did between the last frame and this
//!    one) goes to a scripted run's record, and the oldest sample it applied is kept until the
//!    frame is shown, for the pen's latency (sample to present submitted,
//!    [`Presented::submitted`]);
//! 3. **feed**: the stale-tile feed is drained: the epochs closed since the last frame;
//! 4. **uploads**: GPU residency ([`Uploads`], [`crate::uploads::GpuResidency`]) takes the drained
//!    epochs and stages the frame's tiles from the store, until half the way to the frame's
//!    deadline;
//! 5. **hand-off**: the frame goes to the render thread, as a [`FrameRequest`] carrying the staged
//!    uploads and the scene (residency's buffers, the camera's view, the frame's time and whether
//!    the camera moves), which it starts encoding at once (the uploads first, then the passes);
//! 6. **upkeep**: meanwhile the project's recovery writer is ticked and polled;
//! 7. **prepare**: the brush's work off the latency path ([`Brush::prepare`]: a big brush's
//!    catch-up, the finest work of the dabs shown ahead of it, while there is one, else warming the
//!    surface under a hovering pen), until the frame's deadline;
//! 8. **jobs**: and the store's resumable jobs (display-layer refresh and rebuild) run until the
//!    frame's deadline, at least one slice of them (but none while a brush stroke is under way, and
//!    no more once a stroke's samples wait). This frame draws nothing they produce: the epochs they
//!    close are drained by the next frame, which they want.
//!
//! Handing off before the jobs keeps them out of the input's latency (the plan's order,
//! amended by the Task 1 review).
//!
//! **The pen's latency** (the M1 spec's target: sample to present submitted ≤ 8 ms at p99 below
//! F, "with mailbox presentation and late input"):
//!
//! - **frames on input:** a brush stroke's samples want their frame at once, unpaced, where
//!   presentation allows it (mailbox, or offscreen; [`schedule::StrokePacing`]), as soon as the
//!   render thread and the GPU have room; with FIFO (mailbox not offered) they keep the refresh's
//!   pace, and the overlay says so;
//! - **input read late:** the pen stream ([`PenFeed`]) is drained as a frame starts and again just
//!   before its brush phase, and the brush takes every sample up to then;
//! - **the brush's budget is the time left** ([`BrushBudget`]): the latency aim from the frame's
//!   start less its own work after the brush (estimated from the last frames), turned into dabs by
//!   the brush's cost model; more once the brush is behind, so it catches up;
//! - **the brush works between frames** ([`Owner::between_frames`]): while a stroke is under way,
//!   wherever the loop would wait (for the render thread, the GPU, or the refresh slot), it steps
//!   the brush over the samples that arrived, in short slices, and the next frame shows the edits,
//!   its latency measured from the oldest of them;
//! - **the work after the hand-off yields:** the upkeep, the brush's preparation and each step of
//!   the store's jobs stop once a stroke's samples wait (the next frame is wanted on input), and
//!   the jobs don't run at all while a brush stroke is under way (a slice can run far past its
//!   step, and their workers are the brush's): the display layer's refresh waits for the stroke to
//!   land, the stand-ins showing it meanwhile.
//!
//! None of this is particular to sculpting: a painting brush's strokes would go the same way.
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
//! the path is over, and records every frame ([`crate::record`]). Its poses are set directly: it
//! isn't framed at open, and no boundary is fitted around the model for it.

pub(crate) mod schedule;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use geom::TileKey;
use render::camera::View;
use render::lod::LodStats;
use render::raster::FramePasses;
use render::residency::ResidencyStats;
use render::surface::{Presentation, Size};
use render::upload::FrameUploads;
use render::{DrawTimes, Look, Outcome, SceneView, aa};
use store::{Drained, JobsReport, Project, SpatialIndex, StaleCursor, TileStore};

use self::schedule::{Next, Reason, Reasons, Scheduler, StrokePacing, UPKEEP_INTERVAL};
use crate::camera::{Arcball, Boundary, Camera, Pixel};
use crate::navigate::{Button, Input, Navigator};
use crate::overlay::{self, Snapshot};
use crate::pen::{PenFrame, PenNavigation, PenRouter, PenSample, PenStatus, Route};
use crate::pick::Picker;
use crate::record::{Latency, Recorder, RunInfo, SceneRecord};
use crate::render_thread::{FrameRequest, Presented, ResidentScene, SceneRequest};
use crate::script::{BrushSize, Script, ScriptPen, Segment};

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
    /// Whether the renderer holds or wants a tile: what a brush's previews are made of.
    pub(crate) drawn: Drawing<'a>,
    /// The tiles the renderer waits for a brush's previews of (held, none made yet).
    pub(crate) waiting: &'a [TileKey],
}

/// Whether the renderer holds or wants a tile ([`Uploads::draws`]): what a brush's previews are
/// made of.
#[derive(Clone, Copy)]
pub(crate) struct Drawing<'a>(pub(crate) &'a dyn Fn(&TileKey) -> bool);

impl Drawing<'_> {
    /// Every tile drawn: for a brush with no renderer to ask (the tests', and routing, which
    /// makes no preview).
    pub(crate) const ALL: Drawing<'static> = Drawing(&|_| true);

    /// Whether the renderer holds or wants `key`.
    pub(crate) fn draws(&self, key: &TileKey) -> bool {
        (self.0)(key)
    }
}

impl std::fmt::Debug for Drawing<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Drawing")
    }
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
    /// The oldest pen sample whose dabs this frame shows first (applied, or shown ahead of their
    /// finest work): the frame's latency runs from it to its present being submitted. `None` if
    /// the frame shows no dab for the first time.
    pub(crate) oldest: Option<SampleTimes>,
    /// Dabs applied this frame: finest work, the dabs' own or a catch-up of dabs shown before.
    pub(crate) dabs_applied: usize,
    /// Dabs shown this frame ahead of their finest work, which didn't fit its time: a preview
    /// the catch-up makes exact later.
    pub(crate) dabs_shown: usize,
    /// Dabs placed but neither applied nor shown, left for the next frames: the stroke lags the
    /// pen by them.
    pub(crate) dabs_carried: usize,
    /// Dabs shown and not yet applied as the step ends: the catch-up still to do before the
    /// surface is exact.
    pub(crate) dabs_catching_up: usize,
    /// Tiles whose data the frame's dabs edited, summed over its brush steps (its own and those
    /// between the last frame and it): a tile two steps edited counts twice.
    pub(crate) tiles_edited: usize,
    /// Whether one of the frame's dabs was above the footprint limit F (the M1 spec's "Big
    /// brushes"): the pen's latency target holds below it.
    pub(crate) above_limit: bool,
    /// Whether the brush has work left for the next frame (dabs carried, a stroke ending): the
    /// next frame is wanted.
    pub(crate) busy: bool,
    /// Whether a stroke is open as the step ends: the pen is still down. A scripted run's record
    /// keeps it, so the frames from pen-up until the queue drains can be counted.
    pub(crate) open: bool,
    /// What the brush's stroke keeps of the surface as it began, as the step ends: its tiles and
    /// bytes, and the bytes of them the budget had no room for.
    pub(crate) kept: Kept,
}

/// What a brush's stroke keeps of the surface as it began (a sculpting stroke's pen-down data),
/// as [`BrushReport::kept`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Kept {
    /// Tiles kept.
    pub(crate) tiles: usize,
    /// Bytes kept, counted in the store's budget.
    pub(crate) bytes: usize,
    /// Bytes of them held beyond the budget: the room the pools' minimum and the frame's pins
    /// left had none for them.
    pub(crate) beyond_budget: usize,
}

impl BrushReport {
    /// This report followed by `later`, a later step's: what both applied and showed (the tiles
    /// edited summed, not a union), the earlier sample of the two, and the later step's state
    /// (carried, catching up, busy, open, kept).
    #[must_use]
    pub(crate) fn then(self, later: Self) -> Self {
        let oldest = match (self.oldest, later.oldest) {
            (Some(a), Some(b)) => Some(a.earliest(b)),
            (a, b) => a.or(b),
        };
        Self {
            oldest,
            dabs_applied: self.dabs_applied + later.dabs_applied,
            dabs_shown: self.dabs_shown + later.dabs_shown,
            dabs_carried: later.dabs_carried,
            dabs_catching_up: later.dabs_catching_up,
            tiles_edited: self.tiles_edited + later.tiles_edited,
            above_limit: self.above_limit || later.above_limit,
            busy: later.busy,
            open: later.open,
            kept: later.kept,
        }
    }
}

/// The share of a frame the store's jobs take while a brush stroke is under way: 1.5 ms.
///
/// The display layer's refresh, ahead of the brush's catch-up, while the frame is on time (see
/// [`Owner::run_jobs`]): a few refresh slices.
pub(crate) const STROKE_REFRESH: Duration = Duration::from_micros(1_500);

/// How near the pen's next sample no slice of the store's jobs starts during a stroke: 2 ms, about
/// what a slice takes when it overruns its millisecond.
const SAMPLE_CLEAR: Duration = Duration::from_millis(2);

/// The latency aim for a stroke's input: from a sample being taken to the present of the frame
/// that shows its effect being submitted.
///
/// Below the M1 spec's target (8 ms at p99) by a margin for what the frame loop's estimates of
/// its own work miss at the tail.
pub(crate) const LATENCY_AIM: Duration = Duration::from_millis(6);

/// How long after a stroke's last sample the stroke counts as moving.
///
/// The store's jobs wait while it does ([`Owner::stroke_under_way`]), and run again once the pen
/// holds still, lifts, or its `Up` is lost. The pen reports at 173 Hz while it moves, so a few
/// samples' gap.
pub(crate) const STROKE_QUIET: Duration = Duration::from_millis(50);

/// The least time a frame's brush step gets, whatever the estimate of the frame's other work.
///
/// A frame whose other work fills the aim still applies a useful amount, so a brush behind catches
/// up rather than carrying all but one dab a frame.
const LEAST_BRUSH: Duration = Duration::from_millis(1);

/// What the frame's work after the brush is taken to cost before any frame has measured it:
/// the feed, the uploads, the hand-off, the render thread's encode and submit.
const FIRST_AFTER_BRUSH: Duration = Duration::from_millis(3);

/// How fast the estimate of the frame's work after the brush forgets a slow frame.
///
/// It rises to a longer one at once and falls this share of the way to a shorter one each frame
/// (a decaying maximum, so a slow frame among the last few is planned for, but not for long).
const AFTER_BRUSH_FALL: u32 = 8;

/// How long one step of brush work between frames may run ([`Owner::between_frames`]): short, so
/// the loop hears soon of the render thread finishing and of new samples.
pub(crate) const BETWEEN_SLICE: Duration = Duration::from_micros(1500);

/// How long the store's jobs run between looks at the pen: they yield to a stroke's samples
/// within about this (and a slice's overrun).
const JOB_STEP: Duration = Duration::from_millis(1);

/// When a brush step must have applied its dabs by: the time its frame has left.
///
/// A stroke's frame starts as its sample arrives (where strokes go on input), so to meet the
/// latency aim it must submit its present by its start plus [`LATENCY_AIM`]. The frame loop
/// knows how long its own work after the brush takes (the feed, the uploads, the hand-off and the
/// render thread's encode and submit: [`Owner`] keeps an estimate, the longest of the last
/// frames'), so the brush has until the aim less that work ([`BrushBudget::frame`]), at least
/// [`LEAST_BRUSH`]. The brush turns the time into dabs with its own cost model; the dabs that
/// don't fit are carried to the next step.
///
/// The time isn't cut further for a sample that waited before its frame started: the dabs cut
/// would only move to the next frame, a whole frame later, and the frame's latency is its oldest
/// sample's either way.
///
/// **A brush behind catches up:** once the oldest dab it holds is older than the aim, the aim is
/// lost for it whatever the frame does, and what matters is draining the queue. Then the step may
/// take up to a refresh interval ([`BrushBudget::until`]), so the frames' fixed work is spread over
/// more dabs while the screen still updates at about the display's rate. Between frames the budget
/// is a plain slice ([`BrushBudget::slice`]).
///
/// **Progress:** a step between frames (or after a frame's hand-off) is there for finest work, so
/// it applies at least one dab ([`BrushBudget::forces_progress`]); a frame's step needn't, where
/// the brush can show what doesn't fit its time ahead of its finest work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BrushBudget {
    /// When the step began.
    start: Instant,
    /// When the step must be done by, keeping to the aim.
    until: Instant,
    /// When it must be done by once behind.
    behind: Instant,
    /// Whether the step must apply at least one dab if any is queued.
    progress: bool,
}

impl BrushBudget {
    /// A frame's brush budget for a step starting at `start`, the frame's work after the brush
    /// estimated at `after`, frames paced to `interval` (the most a step behind takes).
    pub(crate) fn frame(start: Instant, after: Duration, interval: Duration) -> Self {
        let until = start + LATENCY_AIM.saturating_sub(after).max(LEAST_BRUSH);
        Self {
            start,
            until,
            behind: until.max(start + interval),
            progress: false,
        }
    }

    /// An hour from now: every queued dab, for tests that step a brush directly.
    #[cfg(test)]
    pub(crate) fn unlimited() -> Self {
        Self::slice(Instant::now(), Duration::from_secs(3600))
    }

    /// A budget of `length` from `start`: work between frames, behind or not.
    pub(crate) fn slice(start: Instant, length: Duration) -> Self {
        Self {
            start,
            until: start + length,
            behind: start + length,
            progress: true,
        }
    }

    /// When the step must be done by keeping to the aim: what a step whose dabs are all shown
    /// already (or caught up) has, however old they are.
    pub(crate) const fn aim(&self) -> Instant {
        self.until
    }

    /// Whether the step must apply at least one dab if any is queued: work between frames, not a
    /// frame's (see "Progress" above).
    pub(crate) const fn forces_progress(&self) -> bool {
        self.progress
    }

    /// When the step must be done by, the oldest dab it holds placed from a sample taken at
    /// `oldest`: the aim's deadline, or the catch-up's if that sample is older than the aim.
    pub(crate) fn until(&self, oldest: Instant) -> Instant {
        if self.start.saturating_duration_since(oldest) > LATENCY_AIM {
            self.behind
        } else {
            self.until
        }
    }
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
    /// One step of brush work: the pen samples queued since the last step (`pen`, oldest first,
    /// with the pen's hover), placed, and as many of the queued dabs applied as `budget` leaves
    /// time for (at least one, if any is queued and the budget [forces
    /// progress](BrushBudget::forces_progress) or the brush can't show dabs otherwise), the rest
    /// shown ahead of their finest work where the brush can, their epochs closed. The context's
    /// camera is the view as the input left it, which the samples are projected through.
    ///
    /// A frame's step ([`Owner::frame_fed`]), or one between frames ([`Owner::between_frames`]),
    /// whose edits the next frame shows.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from; a store error in a stroke ends the stroke
    /// and shows in the overlay ([`Brush::status`]).
    fn step(
        &mut self,
        context: BrushContext<'_>,
        pen: &PenFrame,
        budget: &BrushBudget,
    ) -> Result<BrushReport>;

    /// Whether the brush takes the stroke `down` starts, in the input phase, with the context's
    /// camera as the input so far left it: the one place the brush claims strokes. A stroke
    /// started with a barrel button held navigates whatever this says ([`crate::pen::route`]).
    /// By default every stroke navigates.
    fn route(&mut self, _context: BrushContext<'_>, _down: &PenSample) -> Route {
        Route::Navigate
    }

    /// Work kept off the input's latency path, run after the frame has been handed off and
    /// before the store's jobs, until `deadline`: the finest work of dabs shown ahead of it (the
    /// catch-up), and warming the surface under a hovering pen. Returns whether some is left,
    /// which wants the next frame.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from.
    fn prepare(&mut self, _context: BrushContext<'_>, _deadline: Instant) -> Result<bool> {
        Ok(false)
    }

    /// When the pen's next sample is due, while it draws: the store's jobs start no slice just
    /// before it. `None` with no pen drawing.
    fn next_sample_due(&self) -> Option<Instant> {
        None
    }

    /// The window was asked to close: lifts the pen of any open stroke, and has the brush land
    /// what it holds through its steps, frame by frame, with the next strokes that wait
    /// ([`Brush::landing`] counts down). The window answers meanwhile.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from.
    fn begin_closing(&mut self, _context: BrushContext<'_>) -> Result<()> {
        Ok(())
    }

    /// How many dabs the brush still holds to land, those of strokes that wait among them: what
    /// a close waits for.
    fn landing(&self) -> usize {
        0
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

    /// Where the last stroke the brush began starts (its first dab's centre), with when its
    /// `Down` was received, once: the next call gives `None` until another begins. The frame
    /// loop makes it the camera's pivot after each step (the navigation spec: orbit and zoom
    /// about the last edit), unless an event since set another ([`Owner::pivot_to`]). None by
    /// default.
    fn take_stroke_start(&mut self) -> Option<(crate::camera::Vec3, Instant)> {
        None
    }

    /// Whether the last [`Brush::route`] navigated because it found no surface under its `Down`
    /// (it looked, and missed): the frame loop then doesn't pick under it again for the pivot.
    /// False by default, as the default route looks at nothing.
    fn routed_off_surface(&self) -> bool {
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
    fn step(
        &mut self,
        _context: BrushContext<'_>,
        _pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        Ok(BrushReport::default())
    }
}

/// Where pen samples come from, read as late as the frame loop can.
///
/// The window's pen stream ([`crate::pen::AppPen`]) is pulled at a frame's start, just before its
/// brush phase, and between the work after its hand-off, so a stroke's samples are never left
/// waiting behind the store's jobs.
pub(crate) trait PenFeed {
    /// Appends every sample that arrived since the last call to `out`, in order.
    fn pull(&mut self, out: &mut Vec<PenSample>);
}

/// No pen stream: offscreen, or a test; a scripted run's samples come from its script.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoPen;

impl PenFeed for NoPen {
    fn pull(&mut self, _out: &mut Vec<PenSample>) {}
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

    /// Whether the renderer holds or wants tile `key`, which a brush's previews are made of.
    /// Every tile with no renderer.
    fn draws(&self, _key: &TileKey) -> bool {
        true
    }

    /// The tiles the last staging left waiting for a brush's previews.
    fn waiting_for_previews(&self) -> Vec<TileKey> {
        Vec::new()
    }

    /// The display tiles the renderer draws whose data is behind the store's finest level (a
    /// stand-in, a preview, or a record a stroke has changed under): the refresh builds their
    /// subtrees first. None with no renderer.
    fn drawn_stale(&self, _store: &TileStore) -> Vec<TileKey> {
        Vec::new()
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
    /// Whether a brush stroke is open as the frame's brush step ended: a stroke beginning
    /// restarts the record's level jumps of the stroke's tiles ([`crate::uploads`]).
    pub(crate) stroke_open: bool,
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
    /// The brush's work between the last frame and this one ([`Owner::between_frames`]), whose
    /// edits this frame shows: off this frame's own path, but on its samples' latency.
    pub(crate) between: Duration,
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
    /// The display tiles they built and recorded then ([`JobsReport::built`]).
    pub(crate) jobs_built: usize,
    /// The display tiles they had left to build then ([`JobsReport::left`]).
    pub(crate) jobs_left: usize,
    /// Their longest slice then ([`JobsReport::longest_slice`]): what no yield cuts short.
    pub(crate) jobs_longest: Duration,
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
    /// When each of `input` was queued, in order: a drag's press dates the pivot it sets.
    input_at: Vec<Instant>,
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
    /// The view the last frame was made with (the camera's; `None` if it gave none) and the
    /// frame size; `None` before the first. The view, not the camera: a change that moves no
    /// pixel (a new pivot, a boundary grown with the eye outside it) is no move.
    last_view: Option<(Option<View>, Size)>,
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
    /// The frames not shown yet that applied dabs, by number, each with the oldest pen sample
    /// whose dabs it applied: a frame's latency is measured as it is shown. Frames shown (or
    /// lost) leave it, so it holds the few in flight.
    latency_from: BTreeMap<u64, SampleTimes>,
    /// When the frames of `latency_from` started: how long their oldest samples waited for one.
    began_at: BTreeMap<u64, Instant>,
    /// The last frame's pen latency, sample to present submitted, for the overlay; `None` until
    /// a frame applies dabs.
    latency: Option<Duration>,
    /// How far into a scripted run its pen samples have been queued: those up to here are.
    script_pen: Duration,
    /// Why a scripted pen's brush couldn't be sized as its stroke asked
    /// ([`Owner::script_radius`]), found as its samples were queued (between frames, where
    /// nothing returns an error): the next frame fails with it, so a measured run never draws a
    /// stroke at a size it didn't ask for.
    script_failure: Option<anyhow::Error>,
    /// Whether the brush is stroking, for the overlay's pick.
    stroking: Stroking,
    /// What the brush did between the last frame and the next ([`Owner::between_frames`]), which
    /// the next frame reports as its own, its latency measured from the oldest sample.
    between: Option<BrushReport>,
    /// How long the brush worked between the last frame and the next.
    between_time: Duration,
    /// When each frame not shown yet that applied dabs ended its brush phase, by number: its work
    /// after the brush is measured as it is presented.
    brushed_at: BTreeMap<u64, Instant>,
    /// The frame's work after the brush (feed, uploads, hand-off, encode and submit), as a
    /// decaying maximum over the frames presented since the stroke's `Down`
    /// ([`AFTER_BRUSH_FALL`]); `None` before any.
    after_brush: Option<Duration>,
    /// When a brush stroke's last sample was queued: the stroke counts as moving for
    /// [`STROKE_QUIET`] after it.
    stroke_sample_at: Option<Instant>,
    /// When the project's upkeep (the recovery writer's tick and poll) last ran: it is never
    /// skipped once [`UPKEEP_INTERVAL`] has passed, pen or not.
    upkept_at: Instant,
    /// What the camera is to be fitted to the model by, at the next input phase.
    fitting: Fitting,
    /// When the event that set the camera's pivot last happened (a drag's press, a stroke's
    /// `Down`, F); `None` before any. A source later than another wins, whichever is heard first
    /// ([`Owner::pivot_to`]).
    pivot_at: Option<Instant>,
    /// Whether the window was asked to close ([`Owner::request_close`]).
    closing: bool,
}

/// What the camera is still to be fitted to the model by: framing it, and growing its boundary
/// around the brush's edits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Fitting {
    /// Whether the camera is to frame the model at the next input phase with a frame size (on
    /// open, and on F): framing needs the viewport's shape.
    frame: bool,
    /// Whether the brush has edited the surface since the camera's boundary was last fitted
    /// around it: it is grown, once the brush is idle ([`Owner::fit_boundary`]).
    bounds: bool,
}

/// Whether the brush is stroking, as the overlay's pick and the latency's rules need to know.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Stroking {
    /// Whether a sample queued since the last routing belongs to a brush stroke, or may (a `Down`
    /// not yet routed): it wants a frame on input, and the work after a hand-off yields to it.
    queued: bool,
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

impl AfterHandOff {
    /// `phases` with what this says of the work after the hand-off.
    const fn into(self, phases: OwnerPhases) -> OwnerPhases {
        OwnerPhases {
            upkeep: self.upkeep,
            prepare: self.prepare,
            jobs: self.jobs,
            jobs_built: self.report.built,
            jobs_left: self.report.left,
            jobs_longest: self.report.longest_slice,
            ..phases
        }
    }
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
        let mut scheduler = Scheduler::new(setup.interval, now)
            .with_morph(uploads.morph())
            .with_stroke_pacing(StrokePacing::of(setup.presentation));
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
            input_at: Vec::new(),
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
            recorder: (setup.script.is_some() || run_info.is_some()).then(Recorder::default),
            script: setup.script,
            script_over: false,
            run_info,
            latency_from: BTreeMap::new(),
            began_at: BTreeMap::new(),
            latency: None,
            script_pen: Duration::ZERO,
            script_failure: None,
            stroking: Stroking::default(),
            between: None,
            between_time: Duration::ZERO,
            brushed_at: BTreeMap::new(),
            after_brush: None,
            stroke_sample_at: None,
            upkept_at: now,
            fitting: Fitting::default(),
            pivot_at: None,
            closing: false,
        }
    }

    /// A recorded run's record lines ([`crate::record`]), scripted or live; empty for a run not
    /// recorded.
    pub(crate) fn record_lines(&self) -> Vec<String> {
        match (&self.recorder, &self.run_info) {
            (Some(recorder), Some(info)) => recorder.lines(info, self.script.as_ref()),
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
    /// input may move the camera, or starts a drag (whose press may move the pivot, which the
    /// overlay shows).
    pub(crate) fn input(&mut self, input: Input) {
        if input == Input::Cancel {
            // Navigation forgets the pen's buttons too: the next sample presses them again.
            self.pen_navigation = PenNavigation::default();
        }
        if self.navigator.may_move(&self.input, &input) {
            self.scheduler.want(Reason::Input);
        } else if self.picker.is_some() && self.navigator.drag_start(&input).is_some() {
            self.scheduler.want(Reason::Overlay);
        }
        if let Input::Move { at } | Input::Press { at, .. } = input {
            self.point(Some(at));
        }
        self.input.push(input);
        self.input_at.push(Instant::now());
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
    ///
    /// A sample of a brush stroke (or a `Down`, or a sample behind a `Down` not routed yet: either
    /// may be the brush's) wants a frame for the stroke ([`Reason::Stroke`], started on input where
    /// presentation allows); any other in a stroke wants one paced as ever ([`Reason::Input`]).
    pub(crate) fn pen(&mut self, sample: PenSample) {
        self.queue_pen(sample, true);
    }

    /// [`Owner::pen`]; wanting a frame for it only if `want` (a sample the frame being made takes
    /// itself wants none).
    fn queue_pen(&mut self, sample: PenSample, want: bool) {
        if self.closing {
            return;
        }
        self.point((sample.phase != input::Phase::Leave).then_some(sample.at));
        if sample.phase.in_stroke() {
            // A `Down` with a barrel button held navigates whatever the brush says (`pen::route`).
            let maybe_brush = |down: &PenSample| {
                down.phase == input::Phase::Down && down.buttons == input::Buttons::default()
            };
            let brush = maybe_brush(&sample)
                || self.stroke_open()
                || self.pen_queue.iter().any(maybe_brush);
            self.stroking.queued |= brush;
            if brush {
                self.stroke_sample_at = Some(Instant::now());
                if sample.phase == input::Phase::Down {
                    // A new stroke plans from its own frames.
                    self.after_brush = None;
                }
            }
            if want {
                self.scheduler
                    .want(if brush { Reason::Stroke } else { Reason::Input });
            }
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

    /// This frame loop picking the surface under the pointer with `picker` ([`crate::pick`]),
    /// the camera's boundary fitted around the model its index bounds.
    #[must_use]
    pub(crate) fn with_picker(mut self, picker: Picker) -> Self {
        self.picker = Some(picker);
        self.fitting.bounds = true;
        if let Some(size) = self.size {
            self.fit_boundary(size);
        }
        self
    }

    /// This frame loop starting framed: its first input phase with a frame size frames the
    /// model ([`Owner::frame_model`]), as opening a project does.
    #[must_use]
    pub(crate) fn framed(mut self) -> Self {
        self.frame_model();
        self
    }

    /// Frames the model (F): its first input phase with a frame size fits the model's bounds in
    /// view, makes its centre the pivot and unmagnifies the image ([`Camera::frame`]). Nothing
    /// in a scripted run, whose path sets the view, or with no spatial index to bound the model.
    pub(crate) fn frame_model(&mut self) {
        if self.script.is_none() && self.picker.is_some() {
            self.fitting.frame = true;
            self.scheduler.want(Reason::Input);
        }
    }

    /// Grows the camera's boundary around the model as the spatial index now bounds it, if the
    /// brush edited the surface since it was fitted; the camera keeps the eye out of it, and a
    /// frame is wanted if that moved the view in a viewport of `size`. Never in a scripted run:
    /// its path sets its poses directly, boundary or not.
    ///
    /// A pass over the index's tiles ([`Picker::model_sphere`]: 0.4 ms on the benchmark
    /// project's 153,600, 25 µs on 9,600, `store`'s `index` bench), so it runs off the pen's
    /// path: after a frame's hand-off once the brush is idle and no sample waits
    /// ([`Owner::after_hand_off`]), and before a navigation drag's pick ([`Owner::grab`]).
    fn fit_boundary(&mut self, size: Size) {
        if !std::mem::take(&mut self.fitting.bounds) || self.script.is_some() {
            return;
        }
        let sphere = self.picker.as_ref().and_then(Picker::model_sphere);
        if let Some(boundary) = sphere.and_then(|(centre, radius)| Boundary::around(centre, radius))
        {
            let before = self.camera.view(size);
            self.camera.grow_boundary(boundary);
            if self.camera.view(size) != before {
                self.scheduler.want(Reason::Input);
            }
        }
    }

    /// Makes `pivot` the camera's, set by an event at `at`, unless a later event set it
    /// already: a stroke's start noted at its tail, after a drag that began later, loses to the
    /// drag's grab. Wants a frame for the overlay's camera line if it moved the pivot; the view
    /// doesn't move ([`Camera::set_pivot`]), so the frame shades the scene again and the image
    /// keeps accumulating.
    fn pivot_to(&mut self, pivot: crate::camera::Vec3, at: Instant) {
        if self.pivot_at.is_some_and(|set| at < set) {
            return;
        }
        self.pivot_at = Some(at);
        if pivot != self.camera.pivot() {
            self.camera.set_pivot(pivot);
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// Frames the model in a viewport of `size`, if framing was asked for
    /// ([`Owner::frame_model`]).
    fn frame_if_wanted(&mut self, size: Size) {
        if !std::mem::take(&mut self.fitting.frame) {
            return;
        }
        if let Some((centre, radius)) = self.picker.as_ref().and_then(Picker::model_sphere) {
            self.camera.frame(centre, radius, size);
            self.fitting.bounds = false;
            self.pivot_at = Some(Instant::now());
        }
    }

    /// A navigation drag starts at `at`, in a viewport of `size`, by an event at `when`: if the
    /// surface is under it, that point becomes the camera's pivot (the navigation spec), before
    /// the drag moves the camera. A drag started over empty space keeps the pivot. Moves no
    /// pixel, the boundary holding the pivot ([`Camera::set_pivot`]): a boundary the brush's
    /// edits left stale is fitted first ([`Owner::fit_boundary`]), so a pick on surface a stroke
    /// raised past its margin grows it around the model, not the pivot alone.
    ///
    /// # Errors
    ///
    /// If the pick can't evaluate a tile.
    fn grab(&mut self, at: Pixel, size: Size, when: Instant) -> Result<()> {
        self.fit_boundary(size);
        let (Some(picker), Some(view)) = (&self.picker, self.camera.view(size)) else {
            return Ok(());
        };
        let hit = picker
            .pick(self.project.store(), &view, at)
            .context("picking under a navigation drag")?;
        if let Some(hit) = hit {
            let p = hit.position;
            self.pivot_to(crate::camera::Vec3::new(p.x, p.y, p.z), when);
        }
        Ok(())
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
    ///
    /// In a scripted run the script's pen samples taken by `now` are queued first, as a pen's
    /// arrive between frames, and a wait ends early for its next sample ([`Owner::pen_due`]).
    pub(crate) fn next(&mut self, now: Instant) -> Next {
        self.queue_script_pen(now.saturating_duration_since(self.clock), true);
        let next = match self.scheduler.next(now) {
            Next::Wait { until } => Next::Wait {
                until: self.pen_due().map_or(until, |due| due.min(until)),
            },
            next @ (Next::Idle { .. } | Next::Frame { .. } | Next::Render | Next::Gpu) => next,
        };
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
    /// waiting); if it is still too far behind, runs the upkeep if it is due, then the brush's
    /// work between frames if a stroke has some (the pen's samples from `pen`,
    /// [`Owner::between_frames`]), else a slice of the store's jobs, if there are any, until
    /// [`GPU_POLL`] from now. Both loops (offscreen and the window's) call it, and map what it
    /// returns to a sleep or the event loop's control flow.
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
        pen: &mut impl PenFeed,
        finished: impl FnOnce() -> Result<u64>,
    ) -> Result<GpuWait> {
        self.gpu_wait_from.get_or_insert(now);
        if !self.scheduler.polled(finished()?) {
            return Ok(GpuWait::Ready);
        }
        if now >= self.scheduler.upkeep_due() {
            self.upkeep(now)?;
        }
        if self.between_frames(now, pen)? {
            return Ok(GpuWait::Worked);
        }
        if self.stroking.queued || self.stroke_under_way() || !self.project.store().has_jobs() {
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

    /// [`Owner::frame_fed`] with no pen stream: offscreen, and the tests.
    ///
    /// # Errors
    ///
    /// As [`Owner::frame_fed`].
    pub(crate) fn frame(
        &mut self,
        now: Instant,
        deadline: Instant,
        hand_off: impl FnOnce(FrameRequest) -> Result<()>,
    ) -> Result<()> {
        self.frame_fed(now, deadline, &mut NoPen, hand_off)
    }

    /// Makes a frame started at `now`, hands what the render thread is to draw to `hand_off`,
    /// then runs the upkeep and the store's jobs until `deadline` (at least one slice of jobs,
    /// however late it is, unless a stroke's samples are waiting). Call it when [`Owner::next`]
    /// says [`Next::Frame`], and report back with [`Owner::shown`] when the render thread has
    /// drawn it.
    ///
    /// **Input read late:** `pen` is drained as the frame starts and again just before its brush
    /// phase, so the brush takes every sample up to the moment it starts; and between the upkeep,
    /// the brush's preparation and each step of the store's jobs, which stop as soon as a stroke's
    /// samples wait (the next frame is then wanted on input).
    ///
    /// # Errors
    ///
    /// If the brush, the uploads, the hand-off, the project's upkeep or the store's jobs fail,
    /// or a scripted pen's brush couldn't be sized ([`Owner::script_failure`]).
    pub(crate) fn frame_fed(
        &mut self,
        now: Instant,
        deadline: Instant,
        pen: &mut impl PenFeed,
        hand_off: impl FnOnce(FrameRequest) -> Result<()>,
    ) -> Result<()> {
        let size = self
            .size
            .context("a frame was started with no frame size")?;
        self.pull_pen(pen, now, false);
        if let Some(failure) = self.script_failure.take() {
            return Err(failure);
        }
        let reasons = self.scheduler.started(now);
        self.started += 1;
        let mut phases = self.after_hand_off.into(OwnerPhases {
            gpu_wait: self.gpu_waited(now),
            between: std::mem::take(&mut self.between_time),
            ..OwnerPhases::default()
        });
        let mut lap = Lap::new();
        let time = now.saturating_duration_since(self.clock);

        let segment = self.follow_script(time, size);
        self.take_input(size)?;
        phases.input = lap.next();

        let brushed = self.brush_phase(size, now, pen)?;
        phases.brush = lap.next();

        let drained = self.project.store().drain_stale(&self.cursor);
        self.feed.count(&drained);
        phases.feed = lap.next();

        let this_view = (self.camera.view(size), size);
        let moved = self.last_view.replace(this_view) != Some(this_view);
        let update = FrameUpdate {
            drained,
            deadline: uploads_deadline(now, deadline),
            moved,
            time,
            stroke_open: brushed.open,
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

        let preparing = self.after_hand_off(now, (size, deadline), &mut lap, pen)?;
        self.log_phases(self.after_hand_off.into(phases));

        self.want_next(&uploads);
        if brushed.busy {
            // Dabs carried: the next frame applies them, on input where strokes go on input, but
            // paced if this one made no progress (a brush that can't step, with no scene).
            self.scheduler.want(if brushed.dabs_applied > 0 {
                Reason::Stroke
            } else {
                Reason::Input
            });
        }
        let stroke_over = !self.stroke_open() && !self.stroking.busy;
        if self.stroking.pick_deferred && stroke_over && self.picker.is_some() {
            // The stroke is over (this frame may have taken its `Up`): the next frame makes the
            // pick this one skipped.
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
            pacing: self.scheduler.stroke_pacing(),
        };
        overlay::lines(&snapshot)
    }

    /// A frame of `size` started at `now`, once handed off: the upkeep, the brush's preparation,
    /// then the store's jobs until `deadline` (at least a slice of them), each timed by `lap`
    /// into [`Owner::after_hand_off`]. Returns whether the brush has preparation left.
    ///
    /// **Each yields to the pen:** `pen` is drained before each, and between steps of the
    /// preparation and of the jobs ([`JOB_STEP`]); once a stroke's samples wait, the rest is left
    /// for later (the next frame, wanted on input, runs it again), so a sample never waits behind
    /// them. The jobs' first slice, however late the frame, is skipped then too.
    ///
    /// # Errors
    ///
    /// As [`Project::tick`], [`Project::poll`], [`Brush::prepare`] and the store's jobs.
    fn after_hand_off(
        &mut self,
        now: Instant,
        (size, deadline): (Size, Instant),
        lap: &mut Lap,
        pen: &mut impl PenFeed,
    ) -> Result<bool> {
        // The upkeep yields to a stroke's samples, but never once it is due: the recovery writer
        // must keep a long stroke's edits.
        let waiting = self.pull_pen(pen, Instant::now(), true);
        if !waiting || self.upkeep_due(Instant::now()) {
            self.project.tick(now).context("ticking the project")?;
            self.project.poll().context("polling the project")?;
            self.upkept_at = Instant::now();
        }
        self.after_hand_off.upkeep = lap.next();

        // While a stroke is under way, the jobs keep a share of the frame ahead of the catch-up,
        // none once the frame runs late, when they start no slice ([`Owner::run_jobs`]).
        let stroking = self.stroke_under_way();
        let on_time_until = now + LATENCY_AIM;
        let share = if stroking && Instant::now() < on_time_until {
            STROKE_REFRESH
        } else {
            Duration::ZERO
        };
        let prepare_by = if stroking {
            deadline.checked_sub(share).unwrap_or(deadline)
        } else {
            deadline
        };
        let mut preparing = false;
        while !self.pull_pen(pen, Instant::now(), true) {
            let step = (Instant::now() + JOB_STEP).min(prepare_by);
            preparing = self
                .with_brush(size, |brush, context| brush.prepare(context, step))
                .context("preparing the brush")?;
            if !preparing || Instant::now() >= prepare_by {
                break;
            }
        }
        if !self.brushing() && !self.pull_pen(pen, Instant::now(), true) {
            // Off the pen's path: the frame is handed off and no sample waits.
            self.fit_boundary(size);
        }
        self.after_hand_off.prepare = lap.next();

        let jobs_by = if stroking {
            deadline.min(Instant::now() + share)
        } else {
            deadline
        };
        if self.project.store().has_jobs() {
            let roots = self.uploads.drawn_stale(self.project.store());
            self.project.store().refresh_first(&roots);
        }
        let boxed = stroking.then_some(on_time_until);
        self.after_hand_off.report = self.run_jobs(jobs_by, boxed, pen)?;
        self.after_hand_off.jobs = lap.next();
        Ok(preparing)
    }

    /// The store's jobs until `deadline`, at least a slice of them, in steps of [`JOB_STEP`],
    /// stopping as soon as a stroke's samples wait in `pen` (before the first slice too).
    ///
    /// **They run during a stroke too** (user decision 2026-11-03: before, the refresh waited for
    /// a stroke and its catch-up, so stand-ins piled up and the view sharpened seconds after the
    /// pen lifted), time-boxed by the caller ([`STROKE_REFRESH`] while the frame is on time), the
    /// subtrees of the display tiles the renderer draws stale first
    /// ([`TileStore::refresh_first`]). The pen keeps priority. With `on_time_until` (a stroke
    /// under way, the end of the frame's latency aim) no slice is forced, and none starts once
    /// that is past, without [`JobsReport::SLICE_TARGET`] left of `deadline`, within
    /// [`SAMPLE_CLEAR`] of the pen's next sample, or while a sample waits: a slice started can't
    /// be cut short. The record's `jobs_ms`, `jobs_longest_ms` and `sample_wait_ms` show what
    /// they cost a sample.
    ///
    /// # Errors
    ///
    /// As the store's jobs.
    fn run_jobs(
        &mut self,
        deadline: Instant,
        on_time_until: Option<Instant>,
        pen: &mut impl PenFeed,
    ) -> Result<JobsReport> {
        let mut report = JobsReport::default();
        // A slice however late, unless a stroke is under way.
        let mut first = on_time_until.is_none();
        while !self.pull_pen(pen, Instant::now(), true) {
            let at = Instant::now();
            if !first && (at >= deadline || !self.project.store().has_jobs()) {
                break;
            }
            if let Some(until) = on_time_until {
                let due = self.brush.next_sample_due();
                let too_near = due.is_some_and(|due| at + SAMPLE_CLEAR >= due);
                let no_room = at + JobsReport::SLICE_TARGET > deadline;
                if at >= until || no_room || too_near {
                    break;
                }
            }
            let least = if first {
                at + JobsReport::SLICE_TARGET
            } else {
                at
            };
            let step = (at + JOB_STEP).min(deadline).max(least);
            let ran = self
                .project
                .store()
                .run_jobs(step)
                .context("running the store's jobs")?;
            add_jobs(&mut report, &ran);
            first = false;
        }
        Ok(report)
    }

    /// Calls the brush with its context for a frame of `size`: the project, picking's spatial
    /// index and the camera.
    fn with_brush<R>(&mut self, size: Size, call: impl FnOnce(&mut B, BrushContext<'_>) -> R) -> R {
        let waiting = self.uploads.waiting_for_previews();
        let uploads = &self.uploads;
        let drawn = |key: &TileKey| uploads.draws(key);
        let context = BrushContext {
            project: &mut self.project,
            index: self.picker.as_mut().map(Picker::index_mut),
            camera: &self.camera,
            size,
            drawn: Drawing(&drawn),
            waiting: &waiting,
        };
        call(&mut self.brush, context)
    }

    /// The brush phase of a frame of `size`: the samples that arrived from `pen` during the input
    /// phase are routed too (input read late), then the brush steps within the time the frame has
    /// left ([`BrushBudget::frame`]). What it did between the last frame and this one counts as
    /// this frame's; the oldest sample applied is kept until the frame is shown, for its latency.
    ///
    /// # Errors
    ///
    /// If the brush fails in a way the app can't go on from.
    fn brush_phase(
        &mut self,
        size: Size,
        began: Instant,
        pen: &mut impl PenFeed,
    ) -> Result<BrushReport> {
        self.pull_pen(pen, Instant::now(), false);
        if !self.pen_queue.is_empty() {
            self.route_pen(&Arcball::new(size), size, false)?;
        }
        let budget = BrushBudget::frame(
            Instant::now(),
            self.after_brush_estimate(),
            self.scheduler.interval(),
        );
        let stepped = self.brush_step(size, &budget)?;
        let brushed = match self.between.take() {
            Some(between) => between.then(stepped),
            None => stepped,
        };
        self.stroking.busy = brushed.busy;
        if let Some(oldest) = brushed.oldest {
            self.latency_from.insert(self.started, oldest);
            self.began_at.insert(self.started, began);
            self.brushed_at.insert(self.started, Instant::now());
        }
        Ok(brushed)
    }

    /// A brush step for a frame of `size` (a frame's brush phase, or work between frames): the
    /// brush steps over the project with the camera and the pen samples the last routing left,
    /// within `budget`. Its edits move the surface, so the next pick is made afresh.
    ///
    /// # Errors
    ///
    /// If the brush fails in a way the app can't go on from.
    fn brush_step(&mut self, size: Size, budget: &BrushBudget) -> Result<BrushReport> {
        let pen = std::mem::take(&mut self.pen_frame);
        let stepped = self.with_brush(size, |brush, context| brush.step(context, &pen, budget));
        self.pen_frame = pen;
        let report = stepped.context("the brush step")?;
        if report.dabs_applied > 0
            && let Some(picker) = &mut self.picker
        {
            picker.surface_changed();
        }
        if report.dabs_applied > 0 || report.dabs_shown > 0 {
            self.fitting.bounds = true;
        }
        if let Some((start, at)) = self.brush.take_stroke_start() {
            // Orbit and zoom about where the stroke began; no pixel moves.
            self.pivot_to(start, at);
        }
        Ok(report)
    }

    /// Brush work between frames, at `now`: while a brush stroke is open (or its dabs are still
    /// queued), the pen's samples from `pen` (and a script's taken by now) are routed and the
    /// brush steps over them for [`BETWEEN_SLICE`], so the time a frame waits for the render
    /// thread or the GPU (or, paced to the refresh, for its slot) isn't lost to the stroke. The
    /// next frame shows the edits, and counts the step as its own: its latency runs from the
    /// oldest sample applied. Returns whether it made progress (applied dabs); the loops call it
    /// while they would otherwise wait, again until it says it made none, so a brush that can't
    /// step (no scene to project through) never spins them.
    ///
    /// Nothing while no brush stroke is under way. The mouse's input waits for the frame (it
    /// moves no view during a brush stroke anyway), but the pen's samples routed here that
    /// navigate (a barrel stroke started after the brush's lifted) move the camera here, before
    /// the frame. A due upkeep runs first, pen or not ([`Owner::upkeep`]). The time spent here
    /// isn't counted in the next frame's wait for the GPU ([`OwnerPhases::gpu_wait`]) but in its
    /// `between`.
    ///
    /// # Errors
    ///
    /// If the upkeep fails, or the brush fails in a way the app can't go on from.
    pub(crate) fn between_frames(&mut self, now: Instant, pen: &mut impl PenFeed) -> Result<bool> {
        if self.upkeep_due(now) {
            self.upkeep(now)?;
        }
        self.pull_pen(pen, now, true);
        let Some(size) = self.size else {
            return Ok(false);
        };
        let stroke = self.stroke_open() || self.stroking.queued;
        let work = (stroke && !self.pen_queue.is_empty()) || self.stroking.busy;
        if !work {
            return Ok(false);
        }
        let began = Instant::now();
        self.route_pen(&Arcball::new(size), size, true)?;
        let report = self.brush_step(size, &BrushBudget::slice(began, BETWEEN_SLICE))?;
        self.stroking.busy = report.busy;
        let progress = report.dabs_applied > 0;
        if progress {
            self.scheduler.want(Reason::Stroke);
        } else if report.busy {
            self.scheduler.want(Reason::Input);
        }
        self.between = Some(match self.between.take() {
            Some(before) => before.then(report),
            None => report,
        });
        let took = began.elapsed();
        self.between_time += took;
        if let Some(from) = &mut self.gpu_wait_from {
            *from += took;
        }
        Ok(progress)
    }

    /// Whether the project's upkeep is due at `now`: [`UPKEEP_INTERVAL`] since it last ran.
    fn upkeep_due(&self, now: Instant) -> bool {
        now >= self.upkept_at + UPKEEP_INTERVAL
    }

    /// Drains `pen` into the pen queue, and queues a scripted run's pen samples taken by `now`,
    /// each wanting a frame if `want` (see [`Owner::pen`]); returns whether a brush stroke's
    /// samples wait to be routed.
    fn pull_pen(&mut self, pen: &mut impl PenFeed, now: Instant, want: bool) -> bool {
        let mut pulled = Vec::new();
        pen.pull(&mut pulled);
        for sample in pulled {
            self.queue_pen(sample, want);
        }
        self.queue_script_pen(now.saturating_duration_since(self.clock), want);
        self.stroking.queued
    }

    /// Whether a brush stroke is open: the pen is down on a stroke routed to the brush.
    fn stroke_open(&self) -> bool {
        self.pen_router.stroke() == Some(Route::Brush)
    }

    /// Whether a brush stroke is under way: its samples still coming (one within the last
    /// [`STROKE_QUIET`]), or its dabs still queued. The store's jobs wait meanwhile; a stroke held
    /// still, or whose `Up` was lost, lets them run again.
    fn stroke_under_way(&self) -> bool {
        let moving = self
            .stroke_sample_at
            .is_some_and(|at| at.elapsed() < STROKE_QUIET);
        moving || self.stroking.busy
    }

    /// When a scripted run's next pen sample is taken, if it has one to come and a stroke's
    /// frames start on input (paced to the refresh, the frames come at their slots anyway): a
    /// loop waiting for a frame wakes for it, as a pen's arrival wakes a window's.
    pub(crate) fn pen_due(&self) -> Option<Instant> {
        if self.scheduler.stroke_pacing() != StrokePacing::OnInput || self.script_over {
            return None;
        }
        let next = self.script.as_ref()?.next_pen_after(self.script_pen)?;
        Some(self.clock + next)
    }

    /// The frame's work after the brush, estimated: the decaying maximum over the frames that
    /// applied dabs and were presented since the stroke's `Down` ([`AFTER_BRUSH_FALL`]), or
    /// [`FIRST_AFTER_BRUSH`] before any.
    fn after_brush_estimate(&self) -> Duration {
        self.after_brush.unwrap_or(FIRST_AFTER_BRUSH)
    }

    /// Learns the frame's work after the brush from one presented frame's, `took`.
    fn after_brush_took(&mut self, took: Duration) {
        self.after_brush = Some(match self.after_brush {
            Some(known) if known > took => {
                known.saturating_sub(known.saturating_sub(took) / AFTER_BRUSH_FALL)
            }
            Some(_) | None => took,
        });
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
        let moment = self.script.as_ref()?.at(time);
        self.camera = moment.pose.camera();
        self.input.clear();
        self.input_at.clear();
        self.script_over |= moment.over;
        self.queue_script_pen_at(time, size, false);
        Some(moment.segment)
    }

    /// In a scripted run, queues the script's pen samples taken since the last queued and up to
    /// `time` into the run, as [`Owner::follow_script`] does, each wanting a frame if `want`:
    /// between frames, the script's pen arrives as a pen's would. Nothing with no frame size.
    fn queue_script_pen(&mut self, time: Duration, want: bool) {
        if let Some(size) = self.size {
            self.queue_script_pen_at(time, size, want);
        }
    }

    /// [`Owner::queue_script_pen`] in a view of `size`: each sample where its point projects at
    /// its own time, taken (and received) at its time on the frame loop's clock; a sample's
    /// brush radius, on the first of a stroke, is set as it is queued.
    fn queue_script_pen_at(&mut self, time: Duration, size: Size, want: bool) {
        let Some(script) = &self.script else {
            return;
        };
        if time <= self.script_pen {
            return;
        }
        let due = script.pen(self.script_pen, time);
        self.script_pen = time;
        let projected = due.iter().copied().filter_map(|sample| {
            let view = sample.camera.view(size)?;
            let (x, y, _) = view.pixel(sample.point)?;
            Some((sample, Pixel { x, y }))
        });
        let samples: Vec<(ScriptPen, Pixel)> = projected.collect();
        for (sample, at) in samples {
            if let Some(given) = sample.size {
                match self.script_radius(given, &sample, size) {
                    Ok(pixels) => self.brush.set_radius(pixels),
                    Err(failure) => {
                        self.script_failure.get_or_insert(failure);
                    }
                }
            }
            let taken = self.clock + sample.time;
            let pen = PenSample {
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
            };
            self.queue_pen(pen, want);
        }
    }

    /// The brush radius, in physical pixels, that a scripted pen's `sample` (entering range) sets
    /// as `given` says, in a view of `size`: a radius in pixels as it is, or the radius whose ball
    /// about the sample's point holds that many finest vertices ([`sculpt::radius_covering`]) as
    /// the pixels it spans there, by the brush's own measure (a pixel's world size at the point's
    /// view depth, [`View::pixels_per_unit`]), so the stroke freezes it back to that world radius
    /// at the touch, from the same view.
    ///
    /// Sizing by vertices evaluates the tiles about the point (tens of milliseconds for a million
    /// vertices once they are in the pool, their first evaluation more): in a hover frame, 0.3 s
    /// before the touch, never in a stroke's.
    ///
    /// # Errors
    ///
    /// If sizing by vertices finds no picker's index (no display layer), the point isn't in view,
    /// or a tile can't be evaluated.
    fn script_radius(&mut self, given: BrushSize, sample: &ScriptPen, size: Size) -> Result<f32> {
        let vertices = match given {
            BrushSize::Pixels(pixels) => return Ok(pixels),
            BrushSize::Vertices(vertices) => vertices,
        };
        let picker = self
            .picker
            .as_mut()
            .context("sizing a scripted brush by its vertices needs the spatial index")?;
        let point = sample.point;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a point on the unit sphere: f32 holds it to the index's precision"
        )]
        let center = geom::Vec3::new(point.x as f32, point.y as f32, point.z as f32);
        let count = usize::try_from(vertices).unwrap_or(usize::MAX);
        let world =
            sculpt::radius_covering(picker.index_mut(), self.project.store(), center, count)
                .with_context(|| format!("finding the radius that covers {vertices} vertices"))?;
        let view = sample
            .camera
            .view(size)
            .context("a scripted brush sized with no view")?;
        let (x, y, _) = view
            .pixel(point)
            .context("a scripted brush sized at a point behind the eye")?;
        let through = view.through(x, y);
        let depth = (point - view.eye()).length() / through.length();
        let pixels = f64::from(world) * view.pixels_per_unit() / depth;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a brush radius in pixels: f32 holds it"
        )]
        let pixels = pixels as f32;
        Ok(pixels)
    }

    /// The input phase of a frame of `size`: the camera frames the model if that was asked for,
    /// and its boundary is grown around the brush's edits once the brush is idle; the input
    /// queued since the last frame moves the camera (unless a brush stroke is open: the view
    /// holds still while the brush draws), a navigation drag started over the model moving the
    /// pivot there first ([`Owner::grab`]); the pen samples queued become the frame's
    /// [`PenFrame`], then the surface under the pointer is picked if the pointer or the view
    /// changed, unless the brush is stroking ([`Owner::brushing`]; see the module notes).
    ///
    /// # Errors
    ///
    /// If a pick (under the pointer, or under a navigation drag's start) can't evaluate a tile.
    fn take_input(&mut self, size: Size) -> Result<()> {
        let arcball = Arcball::new(size);
        let height = f64::from(size.height());
        let input = std::mem::take(&mut self.input);
        let input_at = std::mem::take(&mut self.input_at);
        // While a brush stroke is open the view holds still: the stroke's record keeps one camera.
        // Navigation still follows the input (buttons held, the pointer), so nothing jumps after.
        let frozen = self.stroke_open();
        if !frozen {
            self.frame_if_wanted(size);
        }
        let timed = input.into_iter().zip(input_at);
        for (event, when) in timed {
            if !frozen && let Some(at) = self.navigator.drag_start(&event) {
                self.grab(at, size, when)?;
            }
            let mut held = self.camera;
            let camera = if frozen { &mut held } else { &mut self.camera };
            self.navigator.apply(event, camera, &arcball, height);
        }
        self.route_pen(&arcball, size, true)?;
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
        routed || self.stroke_open() || self.stroking.busy
    }

    /// Routes the pen samples queued since the last routing into the [`PenFrame`] the next brush
    /// step takes (a fresh one if `fresh`, else after what it holds: the samples that arrived
    /// during a frame's input phase), in order: navigation's (out of a stroke, or in a stroke
    /// routed to it) move the camera, whose viewport `arcball` covers and is `size`, as they come,
    /// so a later `Down` is routed with the view the earlier samples left. A navigation drag
    /// started over the model moves the pivot there first ([`Owner::grab`]).
    ///
    /// # Errors
    ///
    /// If the pick under a navigation drag's start can't evaluate a tile.
    fn route_pen(&mut self, arcball: &Arcball, size: Size, fresh: bool) -> Result<()> {
        let height = f64::from(size.height());
        if fresh {
            self.pen_frame.clear();
        }
        self.stroking.queued = false;
        let queued = std::mem::take(&mut self.pen_queue);
        let samples = queued.into_iter();
        for sample in samples {
            let brush = &mut self.brush;
            let context = BrushContext {
                project: &mut self.project,
                index: self.picker.as_mut().map(Picker::index_mut),
                camera: &self.camera,
                size,
                drawn: Drawing::ALL,
                waiting: &[],
            };
            let mut asked = false;
            let route = self.pen_router.route(&sample, |down| {
                asked = true;
                brush.route(context, down)
            });
            // A `Down` the brush raycast and found nothing under navigates: the surface isn't
            // under it, so its drag keeps the pivot without picking the same ray again.
            let missed = asked && route == Some(Route::Navigate) && self.brush.routed_off_surface();
            if route != Some(Route::Brush) {
                let mut navigation = self.pen_navigation;
                let mut failed = None;
                navigation.inputs(&sample, |input| {
                    if failed.is_some() {
                        return;
                    }
                    let tip = matches!(
                        input,
                        Input::Press {
                            button: Button::PenTip,
                            ..
                        }
                    );
                    if let Some(at) = self.navigator.drag_start(&input)
                        && !(missed && tip)
                        && let Err(error) = self.grab(at, size, sample.received)
                    {
                        failed = Some(error);
                        return;
                    }
                    self.navigator
                        .apply(input, &mut self.camera, arcball, height);
                });
                self.pen_navigation = navigation;
                if let Some(error) = failed {
                    return Err(error);
                }
            }
            self.pen_frame.push(sample, route);
        }
        Ok(())
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
        // Jobs run in every frame, a stroke's too (time-boxed, [`Owner::run_jobs`]).
        let jobs = self.project.store().has_jobs();
        if jobs || self.has_stale() {
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
        // Frames are reported in order: any before this one still kept were lost.
        self.latency_from = self.latency_from.split_off(&presented.number);
        self.began_at = self.began_at.split_off(&presented.number);
        let began = self.began_at.remove(&presented.number);
        self.brushed_at = self.brushed_at.split_off(&presented.number);
        if let Some(brushed) = self.brushed_at.remove(&presented.number)
            && presented.outcome == Outcome::Presented
        {
            self.after_brush_took(presented.submitted.saturating_duration_since(brushed));
        }
        let latency = self
            .latency_from
            .remove(&presented.number)
            .map(|oldest| Latency {
                taken: presented.submitted.saturating_duration_since(oldest.taken),
                received: presented
                    .submitted
                    .saturating_duration_since(oldest.received),
                waited: began.map_or(Duration::ZERO, |began| {
                    began.saturating_duration_since(oldest.received)
                }),
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
        self.upkept_at = now;
        // A poll can install prefetched tiles, closing epochs: the next frame takes them.
        if self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        Ok(())
    }

    /// The window was asked to close: the pen's samples are no longer taken, and the brush lands
    /// what it holds over the frames that follow, however long that takes, with nothing here
    /// waiting for it. [`Owner::ready_to_close`] says when it has.
    ///
    /// # Errors
    ///
    /// As [`Brush::begin_closing`].
    pub(crate) fn request_close(&mut self) -> Result<()> {
        if self.closing {
            return Ok(());
        }
        self.closing = true;
        self.pen_queue.clear();
        if let Some(size) = self.size.or_else(|| self.last_view.map(|(_, size)| size)) {
            self.with_brush(size, B::begin_closing)
                .context("lifting the pen to close")?;
        }
        self.scheduler.want(Reason::Stroke);
        Ok(())
    }

    /// How many dabs a close waits for ([`Brush::landing`]).
    pub(crate) fn landing(&self) -> usize {
        self.brush.landing()
    }

    /// Whether a close was asked for and nothing is left to land.
    pub(crate) fn ready_to_close(&self) -> bool {
        self.closing && self.landing() == 0
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

    /// The brush, to change: a test's stand-in for time gone by.
    #[cfg(test)]
    pub(crate) const fn brush_mut(&mut self) -> &mut B {
        &mut self.brush
    }

    /// What the brush did between the last frame and the next, not yet counted by a frame.
    #[cfg(test)]
    pub(crate) const fn between_report(&self) -> Option<BrushReport> {
        self.between
    }

    /// What the stale-tile feed has delivered so far.
    #[cfg(test)]
    pub(crate) const fn feed_stats(&self) -> FeedStats {
        self.feed
    }

    /// The oldest sample frame `frame` applied dabs from, while it isn't shown yet.
    #[cfg(test)]
    pub(crate) fn latency_from_of(&self, frame: u64) -> Option<SampleTimes> {
        self.latency_from.get(&frame).copied()
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

/// Adds `ran`, a later run of the store's jobs, to `report`: the counts summed, the longest
/// slice the longer, what is left as `ran` left it.
fn add_jobs(report: &mut JobsReport, ran: &JobsReport) {
    report.slices += ran.slices;
    report.built += ran.built;
    report.evaluated += ran.evaluated;
    report.finished += ran.finished;
    report.left = ran.left;
    report.longest_slice = report.longest_slice.max(ran.longest_slice);
    report.long_slices += ran.long_slices;
    report.failed += ran.failed;
    report.queue_finished += ran.queue_finished;
    report.queue_left = ran.queue_left;
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
