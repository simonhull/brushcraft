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
//!    uploads, the scene (residency's buffers, the camera's view, the frame's time and whether the
//!    camera moves) and what the UI shows, worked out here, after the pen's work; it starts
//!    encoding at once (the uploads first, then the passes);
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
//! **The UI comes before all of it:** with a bar up, input reaches [`crate::dialogs`] as it
//! arrives, before it is queued, and a stroke or press that lands on the bar stops there whole.
//! With none that is one branch ([`Dialogs::routes`]).
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
//!
//! **Reopening** swaps the project under a running window (Use last save, Open as a copy). It runs
//! only while nothing has been drawn since opening, so nothing of the project is at stake, and
//! [`Owner::take_request`] looks again as the window takes it, as a stroke or a save may have
//! come since it was asked for. One loop ends, [`Owner::close_for_reopen`], and another takes over
//! ([`Owner::go_on_from`]): what the user has set up and what the run has counted go across
//! ([`Carried`]), and what belongs to the project stays behind.
//!
//! **Save on the close bar while dabs are still landing** (the dab journal spec, "Closing") doesn't
//! wait for the landing. The save begins as any, but its landing holds back while the project
//! flushes ([`Project::request_flush`]); once every dab placed is in a durable save point of the
//! recovery file ([`Project::flushed`], milliseconds), the window is asked to go
//! ([`WindowRequest::Finish`]), and the process **finishes unseen** after it
//! ([`Owner::finish_unseen_unless`]): it lands in full batches with no frames, ticking the project
//! between them so a kill loses about a second, writes the save, and removes the recovery file. A
//! failure keeps the recovery file and is said in a desktop notification. That needs a brush that
//! journals its dabs ([`Brush::journals`]) and a recovery file that guards the work, the open
//! stroke's journal among it ([`Project::open_stroke_restorable`]); otherwise, or after Cancel, the
//! window waits for the save as for any.
//!
//! **A finisher can be asked to stop** ([`Owner::finish_unseen_unless`]): reopening the project
//! takes over from it ([`crate::takeover`]). Before each batch, its recovery writer just tended, it
//! looks whether it has been asked; once it has, it lands nothing more, flushes, waits up to
//! [`STOP_FLUSH_LIMIT`] for that save point to be durable, and closes the project keeping the
//! recovery file, which releases both locks. The journal holds every dab placed and the count
//! landed at that cut, so the opener restores the stroke and lands the rest. Nothing is saved and
//! nobody is notified ([`Finished::Stopped`]). A flush that can't be made durable (crash
//! protection paused, the writer failed, the disk stalled past the limit), or a close that fails,
//! leaves the recovery file at its last durable save point, as a crash would; the stop says so,
//! and the caller says it on stderr. The finisher looks only between batches: a stop asked during
//! a batch, or during the save that follows the landing, waits for it.

pub(crate) mod schedule;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, ensure};
use geom::TileKey;
use render::camera::View;
use render::lod::LodStats;
use render::raster::FramePasses;
use render::residency::ResidencyStats;
use render::surface::{Presentation, Size};
use render::upload::FrameUploads;
use render::{DrawTimes, Look, Outcome, SceneView, aa};
use store::{
    CacheError, Drained, JobsReport, OpenStroke, Project, ProjectError, Protection,
    RecoveryCandidate, RecoveryError, SaveAhead, SaveOutcome, SpatialIndex, StaleCursor, TileStore,
};
use ui::{Blend, UiFrame};

use self::schedule::{Next, Reason, Reasons, Scheduler, StrokePacing, UPKEEP_INTERVAL};
use crate::camera::{Arcball, Boundary, Camera, Pixel};
use crate::dialogs::{Action, Claim, Dialogs, Facts, LastStroke};
use crate::navigate::{Button, Input, Navigator};
use crate::overlay::{self, Snapshot};
use crate::pen::{PenFrame, PenNavigation, PenRouter, PenSample, PenStatus, Route};
use crate::pick::Picker;
use crate::record::{Latency, RecordStream, Recorder, RunInfo, SceneRecord};
use crate::render_thread::{FrameRequest, Presented, ResidentScene, SceneRequest};
use crate::saving::{
    self, DabsDropped, Protected, SAVE_SLICE, SaveFailure, SaveStep, SaveTrouble, Saving, Stage,
    TilesLost, Work,
};
use crate::script::{BrushSize, Script, ScriptPen, Segment};
use crate::{OpenNote, said_too};

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
    /// No tile drawn: for a brush landing with no renderer at all, which shows nothing ahead
    /// ([`crate::sculpting::finish_restored`]).
    pub(crate) const NONE: Drawing<'static> = Drawing(&|_| false);

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

    /// Takes on `stroke`, the stroke a restore found still landing ([`Project::take_open_stroke`]),
    /// before any step: resumed in the project ([`Project::resume_stroke`]), its dabs shown at
    /// once and landing as a stroke whose pen has lifted, a new stroke waiting behind it. One it
    /// doesn't take on is closed by the project's next stroke, its dabs not landed lost. The
    /// context's size may be a stand-in's, before a window has one: the dabs are placed already.
    ///
    /// # Errors
    ///
    /// Why it isn't taken on. By default none is.
    fn resume(&mut self, _context: BrushContext<'_>, _stroke: OpenStroke) -> Result<()> {
        Err(anyhow!("this brush takes no stroke on"))
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

    /// Whether every dab the brush places is journalled in the recovery file
    /// ([`Project::journal`]), so a crash keeps what hasn't landed, and the close bar's Save can
    /// let the window go as soon as the journal is durable, the landing finishing unseen
    /// ([`Owner::finish_unseen_unless`]). A brush that doesn't keeps the window until the save is
    /// written. None does by default.
    fn journals(&self) -> bool {
        false
    }

    /// How many of the pen events a save holds the brush hasn't placed as dabs yet: a stroke
    /// waiting behind the one landing, held as raw samples. They aren't in the journal until they
    /// are placed, so the window can't go before the save is written while any is left
    /// ([`Owner::finish_unseen_unless`]). None by default.
    fn unplaced_for_save(&self) -> usize {
        0
    }

    /// A save was asked for at `asked`: from now on a stroke whose `Down` came later waits, its
    /// samples kept, until [`Brush::end_saving`], and isn't in the save.
    fn begin_saving(&mut self, _asked: Instant) {}

    /// Whether the pen is down on a stroke a save holds: it can't begin landing yet.
    fn pen_down(&self) -> bool {
        false
    }

    /// The pen has lifted: what a save holds lands from now on only through
    /// [`Brush::save_step`], so the window's events come between batches however things stand.
    fn begin_landing(&mut self) {}

    /// How many dabs a save still waits for: [`Brush::landing`] without the strokes begun after it
    /// was asked for.
    fn landing_for_save(&self) -> usize {
        0
    }

    /// Lands a batch of what a save holds, as [`Brush::close_step`] does for a close. Returns
    /// whether any landed.
    ///
    /// # Errors
    ///
    /// If dabs were dropped: the save can't hold everything drawn.
    fn save_step(&mut self, _context: BrushContext<'_>) -> Result<bool> {
        Ok(false)
    }

    /// The save is over, however it went: strokes go on, those that waited for it first.
    fn end_saving(&mut self) {}

    /// How many times the brush has dropped dabs, a store error ending a stroke early.
    fn drops(&self) -> u64 {
        0
    }

    /// The latest error that dropped dabs, if any did since there were `before` drops.
    ///
    /// # Errors
    ///
    /// That error: a save since then can't hold everything drawn.
    fn dropped_since(&self, _before: u64) -> Result<(), DabsDropped> {
        Ok(())
    }

    /// What ending strokes reported of the recovery file since this was last asked: the frame
    /// loop's to hear ([`recovery_report`]), and no failure of the brush's.
    fn take_recovery_reports(&mut self) -> Vec<ProjectError> {
        Vec::new()
    }

    /// Forgets everything held and not landed, and lands none of it: the project is closing
    /// without saving ([`Owner::discard`]). No more steps follow.
    fn abandon(&mut self) {}

    /// Lands a batch of what the brush holds, whether or not a frame can be drawn. Returns whether
    /// any landed.
    ///
    /// # Errors
    ///
    /// Only for a failure the app can't go on from.
    fn close_step(&mut self, _context: BrushContext<'_>) -> Result<bool> {
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

    /// Takes on what the user set on `from`, the brush of the project this one's replaces.
    fn take_settings(&mut self, _from: &Self) {}

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
    /// Bringing the UI up to date for the frame: off the record's `cpu_ms`, which means what it
    /// meant before this was timed.
    pub(crate) ui: Duration,
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
    /// How the target blends the UI, which the UI is made for.
    pub(crate) blend: Blend,
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
    /// Whether a bar stays up, to measure the UI's cost ([`Dialogs::show_test_bar`]).
    pub(crate) show_bar: bool,
}

/// Whether the window was asked to close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Close {
    /// It wasn't.
    NotAsked,
    /// It was: no pen is taken any more, and what the brush holds lands.
    Asked,
}

/// What the window does once the save under way has succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterSave {
    /// It stays.
    Stay,
    /// It exits: Save on the close bar, or a close during a save. Cancelling takes that back, and
    /// the save carries on, harmless.
    Exit,
    /// It exits before the save is written, which finishes unseen
    /// ([`Owner::finish_unseen_unless`]).
    ///
    /// The close bar's Save while dabs still land: the window goes once every dab placed is in a
    /// durable save point of the recovery file (the dab journal spec, "Closing"). Cancelling takes
    /// that back as for [`AfterSave::Exit`], and the save goes on in the window.
    Unseen(Unseen),
}

/// How far a save the window goes before has got ([`AfterSave::Unseen`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unseen {
    /// Its landing waits while the recovery file takes the journal: a flush was asked for, and the
    /// window goes as soon as [`Project::flushed`] says it is durable.
    Flushing,
    /// It was durable: the window was asked to go ([`WindowRequest::Finish`]).
    Asked,
}

/// How the recovery file came to fall short of the edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Shortfall {
    /// Crash protection is paused: it says so itself, and may come back by itself.
    Paused,
    /// A flush failed, or left tiles out. Nothing says when that is over, so a save has to be.
    Lacking,
}

/// How far the bar has got with saying tiles are gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LostSaid {
    /// It hasn't begun.
    NotYet,
    /// It is saying so, and they are kept until that has been read.
    OnTheBar,
}

/// What the frame loop knows of a recovery file not guarding the edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Exposed {
    /// How it came about.
    shortfall: Shortfall,
    /// Why, as the user hears it.
    trouble: SaveTrouble,
    /// Whether the bar has said so: once, however long it lasts.
    told: bool,
}

/// What a request to close the window does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CloseAnswer {
    /// The window can go: nothing is unsaved, nobody can be asked, or the user has heard a save
    /// failed and the recovery file keeps the work.
    Now,
    /// A bar is showing: it stays until that is answered.
    Asked,
}

/// What the window must do for the owner, which can't reach it ([`Owner::take_request`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WindowRequest {
    /// Close: everything is saved.
    Exit,
    /// Close at once, the save under way finishing with no window
    /// ([`Owner::finish_unseen_unless`]): every dab placed is in a durable save point of the
    /// recovery file.
    Finish,
    /// Close, dropping what isn't saved: the user said so on the close bar.
    ExitDiscarding,
    /// Swap this project in for the one open.
    Reopen(Reopen),
}

/// A project to swap in for the one open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Reopen {
    /// The project from its last save, the recovery file `kept` left until a stroke or a save, so
    /// opening the project again undoes the choice.
    LastSave {
        /// The recovery file of the work being set aside.
        kept: PathBuf,
    },
    /// A recovery file onto an earlier save, saved beside the project.
    Copy {
        /// What was found.
        candidate: RecoveryCandidate,
    },
}

/// What has been done to the project since it opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinceOpening {
    /// Nothing: it can still be swapped for another.
    Untouched,
    /// A stroke went to the brush, which reopening would lose.
    Stroked,
}

/// A copy of found work the user asked for, waiting for a frame to say so before it holds the
/// thread.
#[derive(Debug)]
struct Copying {
    /// What to copy.
    candidate: RecoveryCandidate,
    /// The frame that says so, once one has been handed off.
    said_in: Option<u64>,
}

/// What goes from one frame loop to the next as a project is swapped ([`Owner::close_for_reopen`],
/// [`Owner::go_on_from`]).
///
/// That's what the user has set up, what their hands are doing, and what the run has counted. The
/// frames go on counting, as the render thread's finished frame and a record both run across the
/// swap, and a live record is carried whole, stream and all: a new stream would empty the file.
#[derive(Debug)]
pub(crate) struct Carried<B> {
    /// The old brush, for its settings.
    brush: B,
    /// The view.
    camera: Camera,
    /// When the pivot was last set.
    pivot_at: Option<Instant>,
    /// How the scene is shown.
    look: Look,
    /// Buttons and modifiers held.
    navigator: Navigator,
    /// Pointer input no frame has taken.
    input: Vec<Input>,
    /// When each of `input` was queued.
    input_at: Vec<Instant>,
    /// Pen samples no frame has taken: none can be a brush's, as [`Owner::may_reopen`] says.
    pen_queue: Vec<PenSample>,
    /// The stroke in progress, which can only be navigation's.
    pen_router: PenRouter,
    /// The pen's buttons as navigation has heard them.
    pen_navigation: PenNavigation,
    /// Whether the pen is down.
    pen_down: bool,
    /// The frames counted and the display's pace.
    scheduler: Scheduler,
    /// Frames started.
    started: u64,
    /// Frames the render thread has finished.
    finished: u64,
    /// The frame loop's clock, which the record's times run on.
    clock: Instant,
    /// The record so far.
    recorder: Option<Recorder>,
    /// The record on its way to its file.
    record_stream: Option<RecordStream>,
    /// The record's setting.
    run_info: Option<RunInfo>,
}

/// A key the UI may want, before anything else hears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UiKey {
    /// Enter, alone.
    Enter,
    /// Escape, alone.
    Escape,
}

/// How the brush's dabs land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Landing {
    /// In its frames and between them, as they fit.
    #[default]
    InFrames,
    /// For a close or a save: only through steps of their own ([`Owner::close_step`],
    /// [`Owner::save_step`]), so the window's events come between batches however things stand.
    InSteps,
    /// One of those steps is running ([`Brush::close_step`], [`Brush::save_step`]).
    Stepping,
}

/// What a scripted run's record says of its setting, beside what the [`Setup`] says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordSetup {
    /// Where the record goes as the run goes on ([`RecordStream`]); `None` to keep it till the
    /// end.
    pub(crate) path: Option<PathBuf>,
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
    /// The record on its way to its file, in a run recorded to one.
    record_stream: Option<RecordStream>,
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
    /// When the last sample of any stroke was received: what a bar the pen holds back goes by
    /// ([`Owner::pen_quiet_at`]).
    pen_sample_at: Option<Instant>,
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
    close: Close,
    /// The project's saves ([`Owner::request_save`]).
    saving: Saving,
    /// What the window does once the save under way has succeeded.
    after_save: AfterSave,
    /// What the last failed save meant to do after it: Try again carries on with that, so a retry
    /// of the close bar's Save that works closes the window, as the user asked.
    after_retry: AfterSave,
    /// How many of the brush's drops of dabs the user has been told of: a save counts only those
    /// since it began, so dabs dropped before it, under the close bar say, would go unsaid.
    told_drops: u64,
    /// The recovery file not guarding the edits, if so ([`Owner::heard`]).
    exposed: Option<Exposed>,
    /// Tiles the store says are gone, until a bar has been read saying so, or a save has.
    lost: BTreeSet<TileKey>,
    /// How far the bar has got with saying so.
    lost_said: LostSaid,
    /// In a scripted run, what fell short of the recovery file, which nobody can be told: its
    /// close fails on it, so a measured run can't pass.
    fell_short: Option<String>,
    /// What the window must do next ([`Owner::take_request`]).
    request: Option<WindowRequest>,
    /// A copy of found work on its way to [`Owner::request`].
    copying: Option<Copying>,
    /// What has been done to the project since it opened.
    since_opening: SinceOpening,
    /// A recovery file set aside by Use last save: removed by a stroke or a save, which end the
    /// way back to it, and only then.
    kept_recovery: Option<PathBuf>,
    /// What became of the stroke a restore found still landing ([`Owner::resume_stroke`]): the
    /// restore note says if it can't be finished.
    last_stroke: LastStroke,
    /// What the UI shows.
    dialogs: Dialogs,
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
        let saving = Saving::new(project.path(), now);
        let mut dialogs = Dialogs::new(setup.size, 1.0, setup.blend, now);
        if setup.show_bar {
            dialogs.show_test_bar(now);
        }
        let mut scheduler = Scheduler::new(setup.interval, now)
            .with_morph(uploads.morph())
            .with_stroke_pacing(StrokePacing::of(setup.presentation));
        scheduler.want(Reason::Exposed);
        let scripted = setup.frames.is_some() || setup.script.is_some();
        if scripted {
            scheduler.want(Reason::Scripted);
        }
        let record_stream = setup
            .record
            .as_ref()
            .and_then(|record| record.path.clone())
            .map(RecordStream::to);
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
            record_stream,
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
            pen_sample_at: None,
            upkept_at: now,
            fitting: Fitting::default(),
            pivot_at: None,
            close: Close::NotAsked,
            saving,
            after_save: AfterSave::Stay,
            after_retry: AfterSave::Stay,
            told_drops: 0,
            exposed: None,
            lost: BTreeSet::new(),
            lost_said: LostSaid::NotYet,
            fell_short: None,
            request: None,
            copying: None,
            since_opening: SinceOpening::Untouched,
            kept_recovery: None,
            last_stroke: LastStroke::Whole,
            dialogs,
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
        if self.dialogs.routes() && self.ui_took_input(input) {
            return;
        }
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

    /// Whether `input` was the UI's alone: a press on a bar, or the release of one. The rest goes
    /// on to navigation, the UI hearing where the pointer went.
    fn ui_took_input(&mut self, input: Input) -> bool {
        let now = Instant::now();
        let took = match input {
            Input::Press { button, at } => {
                let claim = self.dialogs.claim_mouse(button, at, now);
                claim == Claim::Ui
            }
            Input::Release { button } if self.dialogs.mouse_holds(button) => {
                let action = self.dialogs.release_mouse(now);
                self.act_on(action, now);
                true
            }
            Input::Move { at } => {
                self.dialogs.mouse_at(Some(at));
                self.ui_hover(Some(at));
                false
            }
            Input::Cancel => {
                self.dialogs.drop_mouse_press();
                false
            }
            Input::Release { .. } | Input::Wheel { .. } | Input::Modifiers(_) => false,
        };
        if took && let Input::Press { at, .. } = input {
            self.point(Some(at));
        }
        took
    }

    /// Whether `sample` was the UI's alone: a `Down` on a bar, or the rest of its stroke. A stroke
    /// the UI holds ends as any does: at its `Up`, or at a sample out of a stroke with its `Up`
    /// lost, which drops the action and goes on as it would.
    fn ui_took_pen(&mut self, sample: &PenSample) -> bool {
        use input::Phase;
        let now = Instant::now();
        if self.dialogs.pen_pressed() {
            match sample.phase {
                Phase::Drag => {
                    self.ui_hover(Some(sample.at));
                    return true;
                }
                Phase::Up => {
                    let action = self.dialogs.release(sample.at, now);
                    self.act_on(action, now);
                    return true;
                }
                Phase::Down | Phase::Enter | Phase::Hover | Phase::Leave => {
                    self.dialogs.drop_pen_press();
                }
            }
        }
        match sample.phase {
            Phase::Down => self.dialogs.claim(sample.at, now) == Claim::Ui,
            Phase::Enter | Phase::Hover => {
                self.ui_hover(Some(sample.at));
                self.new_button_on_the_bar(sample, now)
            }
            Phase::Leave => {
                self.ui_hover(None);
                false
            }
            Phase::Drag => {
                // A pen taken as lifted ([`Owner::lift_quiet_pen`]) was only resting.
                if !self.dialogs.pen_is_down() {
                    self.dialogs.pen_down(now);
                }
                false
            }
            Phase::Up => false,
        }
    }

    /// Whether `sample` presses a barrel button navigation hasn't heard, over the bar: it goes no
    /// further, as a press on the bar never reaches navigation. Off the bar the same button goes on
    /// as ever, and it stays new until it does.
    fn new_button_on_the_bar(&self, sample: &PenSample, now: Instant) -> bool {
        let last = self.pen_queue.last();
        let heard = last.map_or_else(|| self.pen_navigation.buttons(), |queued| queued.buttons);
        let new = sample.buttons.with(heard, false) != input::Buttons::default();
        new && self.dialogs.on_bar(sample.at, now)
    }

    /// When a pen holding a prompt back counts as lifted: [`STROKE_QUIET`] after its last sample.
    /// `None` with no prompt held back.
    fn pen_quiet_at(&self) -> Option<Instant> {
        let last = self
            .pen_sample_at
            .filter(|_| self.dialogs.prompt_held_back());
        last.map(|last| last + STROKE_QUIET)
    }

    /// Takes the pen as lifted if it has gone quiet with a prompt held back, by `now`. An `Up` can
    /// be lost, with no sample to end the stroke, and the bar is the only way out of a close: the
    /// user's waiting for it must end. A pen that was only resting puts it away again at its next
    /// sample ([`Owner::ui_took_pen`]).
    fn lift_quiet_pen(&mut self, now: Instant) {
        if self.pen_quiet_at().is_some_and(|quiet| quiet <= now) {
            self.dialogs.pen_up(now);
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// A stroke went to the brush: going back to work answers the bar. On the close bar that's
    /// Keep sculpting, while finishing, Cancel, the stroke waiting for the save as any does, and a
    /// note goes, what it found staying where it is, and so does a failure, or word of crash
    /// protection stopping, the user could see: one waiting for this stroke's `Up` is yet to be
    /// read. Navigating is no answer: the user may be looking at the model before deciding.
    fn back_to_work(&mut self) {
        let now = Instant::now();
        if self.dialogs.asking_close() {
            self.act(Action::KeepSculpting, now);
        } else if self.dialogs.finishing() {
            self.act(Action::Cancel, now);
        } else if self.dialogs.noting() {
            self.copying = None;
            self.dialogs.clear_prompt(now);
            self.scheduler.want(Reason::Overlay);
        } else if self.dialogs.failure_seen() || self.dialogs.stopping_seen() {
            self.dialogs.clear_prompt(now);
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// The pointer is at `at` as far as the UI goes, and a frame is wanted if that changes what
    /// draws.
    fn ui_hover(&mut self, at: Option<Pixel>) {
        if self.dialogs.hover(at) {
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// Runs `action`, if a press ended with one.
    fn act_on(&mut self, action: Option<Action>, now: Instant) {
        if let Some(action) = action {
            self.act(action, now);
        }
    }

    /// The window was asked to close at `now`. It goes at once with nothing unsaved. With a save
    /// under way that save decides, and otherwise the close bar asks, a close while it shows
    /// changing nothing; either way nothing lands or lifts for it. After a failed save the user has
    /// been told of it goes at once too, its recovery file kept and the failure an error of the run
    /// ([`Owner::leaves_on_failure`]). A scripted run never asks, so
    /// `close_held` and the save's failure hold it instead, and nor does a window with no size,
    /// where no bar could show: what isn't saved stays in the recovery file.
    pub(crate) fn ask_close(&mut self, now: Instant) -> CloseAnswer {
        if self.log.is_some() || self.size.is_none() {
            return CloseAnswer::Now;
        }
        let asked = if self.saving.stage().is_some() {
            if self.after_save == AfterSave::Stay {
                self.after_save = AfterSave::Exit;
            }
            if !self.dialogs.finishing() {
                self.dialogs.finish(self.save_counts(), now);
            }
            true
        } else if self.dialogs.asking_close() || self.dialogs.finishing() {
            true
        } else if self.leaves_on_failure(now) {
            self.saving.leave_on_failure();
            self.dialogs.clear_prompt(now);
            false
        } else if self.unsaved() || self.pen_queued(now) {
            self.dialogs.ask_close(self.saving.name(), now);
            true
        } else {
            false
        };
        if !asked {
            return CloseAnswer::Now;
        }
        self.lift_quiet_pen(now);
        self.scheduler.want(Reason::Overlay);
        CloseAnswer::Asked
    }

    /// Whether a close at `now` leaves on a failed save, asking nothing: the user has had the
    /// chance to read it, no stroke has overtaken it, and the recovery file guards the work, as far
    /// as the failure, the project and what upkeep heard say. Without that guard leaving could lose
    /// it, so a close asks, however often, unless nothing is left unsaved to lose.
    fn leaves_on_failure(&self, now: Instant) -> bool {
        let shown = self.dialogs.failure_shown(now);
        let Some(told) = self.saving.told().filter(|_| shown) else {
            return false;
        };
        let guarded =
            told.covered() && self.project.protection().is_active() && self.exposed.is_none();
        !self.pen_queued(now) && (guarded || !self.unsaved())
    }

    /// Says at `now` what opening found, and of a restore, whether its last stroke can be
    /// finished ([`Owner::resume_stroke`]).
    pub(crate) fn note(&mut self, note: &OpenNote, now: Instant) {
        self.dialogs.note(note, self.last_stroke, now);
        self.scheduler.want(Reason::Overlay);
    }

    /// Hands the brush the stroke a restore found still landing, if there was one
    /// ([`Project::take_open_stroke`], [`Brush::resume`]): before any frame, so it is on screen
    /// in the first, and no stroke or edit can come before it. A journal the project can't use,
    /// or a stroke the brush doesn't take on, leaves the restored tiles standing, as without a
    /// journal: it is said on stderr, and the restore note says the last stroke couldn't be
    /// finished ([`LastStroke::Unfinished`]). The project's next stroke closes it in the file.
    ///
    /// Every project the app starts on goes through here ([`crate::start`]); one not restored has
    /// no stroke to take.
    pub(crate) fn resume_stroke(&mut self) {
        let Some(found) = self.project.take_open_stroke() else {
            return;
        };
        let resumed = match found {
            Ok(stroke) => {
                // A window minimised, or not yet configured, has no size: a stand-in's does, as
                // nothing of the stroke is placed through it.
                let size = self.size.or_else(|| self.last_view.map(|(_, size)| size));
                let size = size.map_or_else(crate::sculpting::unseen_view, Ok);
                let taken = size.and_then(|size| {
                    self.with_brush(size, |brush, context| brush.resume(context, stroke))
                });
                if let Err(why) = &taken {
                    saving::log(&format!(
                        "brushcraft: the last stroke can't be finished: {why:#}"
                    ));
                }
                taken.is_ok()
            }
            Err(why) => {
                let why = anyhow::Error::new(why);
                saving::log(&format!("brushcraft: the last stroke's journal: {why:#}"));
                false
            }
        };
        if !resumed {
            self.last_stroke = LastStroke::Unfinished;
        }
        self.scheduler.want(Reason::Stroke);
    }

    /// What the window must do next, if anything. A reopen asked for is looked at again: a stroke
    /// or a save may have come since, which it would lose, and they stand.
    pub(crate) fn take_request(&mut self) -> Option<WindowRequest> {
        let request = self.request.take();
        if matches!(request, Some(WindowRequest::Reopen(_))) && !self.may_reopen() {
            if self.dialogs.copying() {
                self.dialogs.clear_prompt(Instant::now());
                self.scheduler.want(Reason::Overlay);
            }
            return None;
        }
        request
    }

    /// Whether the project can be swapped for another: nothing has been drawn since opening, or
    /// is about to be, and no save or close is under way. A note is up only then, as a stroke
    /// puts it away, but a sample may be waiting for its frame.
    ///
    /// A stroke a restore found still landing doesn't hold it back: the project closing keeps its
    /// journal at the count landed ([`Project::close`]), so reopening the recovery file restores
    /// it again, and the user's choice can still be undone. Nothing else lands with no stroke
    /// since opening.
    fn may_reopen(&self) -> bool {
        let drawing = self.since_opening == SinceOpening::Stroked
            || self.stroking.queued
            || self.stroke_open()
            || self.brush.pen_down();
        !drawing && self.saving.stage().is_none() && !self.closing()
    }

    /// `key` was pressed at `now`: whether the UI took it, or it goes on as before. A bar the pen
    /// holds back can't be seen, so a key can't answer it.
    pub(crate) fn key(&mut self, key: UiKey, now: Instant) -> bool {
        if self.dialogs.pen_is_down() {
            return false;
        }
        let action = match key {
            UiKey::Enter if self.dialogs.asking_close() => Action::Save,
            UiKey::Escape if self.dialogs.asking_close() => Action::KeepSculpting,
            UiKey::Escape if self.dialogs.finishing() => Action::Cancel,
            UiKey::Enter | UiKey::Escape => return false,
        };
        self.act(action, now);
        true
    }

    /// The save key, at `now`: a save, which answers the close bar if it can be seen, as the key
    /// says what the bar's Save does.
    pub(crate) fn save_key(&mut self, now: Instant) {
        if self.dialogs.asking_close() && !self.dialogs.pen_is_down() {
            self.act(Action::Save, now);
        } else {
            self.request_save(now);
        }
    }

    /// Does what a control on a bar says, and wants the frame that shows it.
    fn act(&mut self, action: Action, now: Instant) {
        match action {
            Action::Save => {
                let unseen = self.may_finish_unseen(now);
                self.request_save(now);
                self.after_save = AfterSave::Exit;
                self.dialogs.finish(self.save_counts(), now);
                if unseen {
                    self.flush_to_finish(now);
                }
            }
            // The one way to losing work on purpose: only the close bar leads here.
            Action::DontSave if self.dialogs.asking_close() => {
                self.request = Some(WindowRequest::ExitDiscarding);
            }
            // Whatever an earlier answer in the same pass asked of the window, this one stands. A
            // save still waiting for the pen goes too: it has done nothing, and its pen may never
            // lift, which would hold every later close at this bar, far from Don't save.
            Action::Cancel => {
                self.after_save = AfterSave::Stay;
                self.request = None;
                self.dialogs.clear_prompt(now);
                if self.saving.stage() == Some(Stage::PenDown) {
                    self.saving.let_go();
                    self.save_over();
                }
            }
            Action::KeepSculpting => {
                self.request = None;
                self.dialogs.dismiss(now);
            }
            Action::Dismiss => self.dialogs.dismiss(now),
            Action::SaveWork => self.request_save(now),
            Action::Ignore => self.dialogs.clear_prompt(now),
            // What failed is what goes again: a copy of found work, which the note it took the
            // place of offers, or a save. The bar goes either way, out of date.
            Action::TryAgain => {
                let failure = self.dialogs.answer_failure(now);
                if failure.is_some_and(SaveFailure::of_found_work) {
                    self.act(Action::OpenCopy, now);
                } else {
                    let exit = self.after_retry == AfterSave::Exit;
                    let unseen = exit && self.may_finish_unseen(now);
                    self.request_save(now);
                    self.after_save = self.after_retry;
                    if exit {
                        self.dialogs.finish(self.save_counts(), now);
                    }
                    if unseen {
                        self.flush_to_finish(now);
                    }
                }
            }
            Action::UseLastSave if self.dialogs.says_restored() && self.may_reopen() => {
                let kept = self.project.recovery_path().map(PathBuf::from);
                self.request = kept.map(|kept| WindowRequest::Reopen(Reopen::LastSave { kept }));
            }
            Action::OpenCopy if self.may_reopen() => {
                let candidate = self.dialogs.begin_copy(now);
                self.copying = candidate.map(|candidate| Copying {
                    candidate,
                    said_in: None,
                });
            }
            Action::DontSave | Action::UseLastSave | Action::OpenCopy => {}
        }
        self.scheduler.want(Reason::Overlay);
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
    ///
    /// **Before all that, with a bar up,** the UI hears the sample ([`Owner::ui_took_pen`]): a
    /// stroke that lands on the bar goes no further, nor does a sample of it. With no bar this is
    /// one branch.
    pub(crate) fn pen(&mut self, sample: PenSample) {
        self.queue_pen(sample, true);
    }

    /// [`Owner::pen`]; wanting a frame for it only if `want` (a sample the frame being made takes
    /// itself wants none).
    fn queue_pen(&mut self, sample: PenSample, want: bool) {
        if self.closing() {
            return;
        }
        if self.dialogs.routes() && self.ui_took_pen(&sample) {
            return;
        }
        if sample.phase == input::Phase::Down {
            self.dialogs.pen_down(Instant::now());
        } else if sample.phase != input::Phase::Drag && self.dialogs.pen_is_down() {
            // As `PenRouter::route`: any sample out of a stroke ends it, an `Up` or not, and one
            // out of a stroke wants no frame of its own.
            let now = Instant::now();
            self.dialogs.pen_up(now);
            if want && self.dialogs.needs_frame(now) {
                self.scheduler.want(Reason::Overlay);
            }
        }
        self.point((sample.phase != input::Phase::Leave).then_some(sample.at));
        if sample.phase.in_stroke() {
            self.pen_sample_at = Some(sample.received);
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
        if self.dialogs.routes() {
            self.dialogs.mouse_at(None);
            self.ui_hover(None);
        }
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
            self.dialogs.resize(size);
            self.expose();
        }
    }

    /// The window's scale factor is `scale`: the overlay is drawn at it, rounded, at least 1, and
    /// the UI at it as it is.
    pub(crate) fn set_scale(&mut self, scale: f64) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a scale factor is a small positive number, clamped to 1..=8"
        )]
        let overlay_scale = scale.round().clamp(1.0, 8.0) as u32;
        let rescaled = self.dialogs.set_scale(scale);
        if rescaled || overlay_scale != self.overlay_scale {
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
    ///
    /// An idle wait ends early too for what the UI changes by itself, which then wants a frame, and
    /// for a pen going quiet with a prompt held back ([`Owner::lift_quiet_pen`]).
    pub(crate) fn next(&mut self, now: Instant) -> Next {
        self.queue_script_pen(now.saturating_duration_since(self.clock), true);
        self.lift_quiet_pen(now);
        let change = self.dialogs.next_change();
        let change = match (change, self.pen_quiet_at()) {
            (Some(one), Some(other)) => Some(one.min(other)),
            (one, other) => one.or(other),
        };
        if change.is_some_and(|change| change <= now) {
            self.scheduler.want(Reason::Overlay);
        }
        let next = match self.scheduler.next(now) {
            Next::Wait { until } => Next::Wait {
                until: self.pen_due().map_or(until, |due| due.min(until)),
            },
            Next::Idle { upkeep } => Next::Idle {
                upkeep: change.map_or(upkeep, |change| change.min(upkeep)),
            },
            next @ (Next::Frame { .. } | Next::Render | Next::Gpu) => next,
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

        // After the pen's work, and asked once: the chip and the overlay say the same.
        let unsaved = self.unsaved();
        lap.next();
        let ui = self.ui_frame(unsaved, now);
        phases.ui = lap.next();
        let overlay = self.overlay(size, phases, uploads.summary.as_deref(), unsaved);
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
            ui,
        })?;
        lap.next();
        self.handed_off();

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

    /// The frame is handed off: a copy waiting for it has its frame, and the pen's path is clear
    /// to remove a recovery file a first stroke has ended the way back to.
    fn handed_off(&mut self) {
        if let Some(copying) = &mut self.copying {
            copying.said_in.get_or_insert(self.started);
        }
        if self.since_opening == SinceOpening::Stroked {
            self.forget_kept_recovery();
        }
    }

    /// What the UI shows in the frame started at `now`, with the project `unsaved` or not.
    fn ui_frame(&mut self, unsaved: bool, now: Instant) -> Option<UiFrame> {
        self.dialogs.update(self.facts(unsaved), now);
        self.dialogs.frame(now)
    }

    /// What decides what the UI shows, with the project `unsaved` or not.
    fn facts(&self, unsaved: bool) -> Facts {
        Facts {
            unsaved,
            saving: self.saving.stage().is_some(),
            landing: self.save_counts(),
        }
    }

    /// The overlay's lines for the frame being made, of `size`, its owner thread's `phases` so
    /// far, with the uploads' `tiles` line, the project `unsaved` or not.
    fn overlay(
        &self,
        size: Size,
        phases: OwnerPhases,
        tiles: Option<&str>,
        unsaved: bool,
    ) -> Vec<String> {
        let brush = self.brush.status();
        let save = self
            .saving
            .line(unsaved, self.save_counts(), self.closing());
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
            save: &save,
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
            self.tend_recovery(now)?;
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
            if let (Some(stream), Some(info)) = (&mut self.record_stream, &self.run_info) {
                let script = self.script.as_ref();
                stream.keep(recorder, info, script, self.started, Instant::now());
            }
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
            if asked && route == Some(Route::Brush) {
                self.since_opening = SinceOpening::Stroked;
                self.saving.told_is_stale();
                if self.dialogs.routes() {
                    self.back_to_work();
                }
            }
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
        if self.dialogs.needs_frame(Instant::now()) {
            self.scheduler.want(Reason::Overlay);
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
        self.copy_if_said(presented.number);
    }

    /// Frame `shown` is on screen: a copy waiting for the frame that says "Saving" can go.
    fn copy_if_said(&mut self, shown: u64) {
        let said = self.copying.as_ref().and_then(|copying| copying.said_in);
        if said.is_some_and(|said| said <= shown)
            && let Some(Copying { candidate, .. }) = self.copying.take()
        {
            self.request = Some(WindowRequest::Reopen(Reopen::Copy { candidate }));
        }
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
    /// As [`Owner::tend_recovery`].
    pub(crate) fn upkeep(&mut self, now: Instant) -> Result<()> {
        self.tend_recovery(now)?;
        self.scheduler.upkept(now);
        self.upkept_at = now;
        // A poll can install prefetched tiles, closing epochs: the next frame takes them.
        if self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        Ok(())
    }

    /// Ticks and polls the project at `now`, and hears what that says of crash protection.
    ///
    /// # Errors
    ///
    /// As [`Project::tick`] and [`Project::poll`], but for what a run can go on after
    /// ([`Owner::heard`]).
    fn tend_recovery(&mut self, now: Instant) -> Result<()> {
        let ticked = self.project.tick(now);
        self.heard(ticked, "ticking the project")?;
        let polled = self.project.poll();
        self.heard(polled, "polling the project")?;
        self.hear_brush()?;
        self.watch_protection(now);
        Ok(())
    }

    /// Hears what stroke ends told the brush of the recovery file.
    ///
    /// # Errors
    ///
    /// As [`Owner::heard`].
    fn hear_brush(&mut self) -> Result<()> {
        let reports = self.brush.take_recovery_reports();
        for report in reports {
            self.heard(Err(report), "ending a stroke")?;
        }
        Ok(())
    }

    /// `result` came of `doing`. A recovery file that can't keep up is no reason to stop
    /// sculpting: it is the user's to hear of, and nothing it says stops the work going on. So
    /// those errors end here ([`recovery_trouble`]), and an old file left undeleted, which
    /// guards as ever, is said in the log alone.
    ///
    /// # Errors
    ///
    /// Any other, the tile store's above all: without it nothing can be drawn or saved.
    fn heard(&mut self, result: Result<(), ProjectError>, doing: &str) -> Result<()> {
        let Err(error) = result else {
            return Ok(());
        };
        if matches!(error, ProjectError::OldRecoveryKept { .. }) {
            let why = anyhow::Error::new(error);
            saving::log(&format!("brushcraft: {doing}: {why:#}"));
            return Ok(());
        }
        if !recovery_trouble(&error) {
            return Err(anyhow::Error::new(error).context(doing.to_owned()));
        }
        if let ProjectError::Lost(tiles) = error {
            let why = format!(
                "{doing}: the latest edits of {} tiles are gone",
                tiles.len()
            );
            saving::log(&format!("brushcraft: {why}"));
            self.note_fell_short(why);
            self.lost.extend(tiles);
        } else if !matches!(error, ProjectError::ProtectionPaused(_)) {
            // A pause is the project's to say, and how it ends: watching takes it from there.
            let trouble = SaveTrouble::of_source(&error);
            self.falls_short(Shortfall::Lacking, trouble, &error, doing);
        }
        Ok(())
    }

    /// In a scripted run, keeps `why` for its close to fail on.
    fn note_fell_short(&mut self, why: String) {
        if self.log.is_some() {
            self.fell_short.get_or_insert(why);
        }
    }

    /// The recovery file isn't guarding the edits, for `shortfall`: said in the log the first time,
    /// as `error` of `doing`. A pause giving way to a shortfall of its own lasts until a save.
    fn falls_short(
        &mut self,
        shortfall: Shortfall,
        trouble: SaveTrouble,
        error: &(dyn std::error::Error + 'static),
        doing: &str,
    ) {
        if let Some(exposed) = &mut self.exposed {
            exposed.shortfall = exposed.shortfall.max(shortfall);
            return;
        }
        self.exposed = Some(Exposed {
            shortfall,
            trouble,
            told: false,
        });
        let mut why = error.to_string();
        let mut link = error.source();
        while let Some(cause) = link {
            why = format!("{why}: {cause}");
            link = cause.source();
        }
        saving::log(&format!("brushcraft: {doing}: {why}"));
        self.note_fell_short(format!("{doing}: {why}"));
    }

    /// Brings what is known of crash protection up to date with the project at `now`: however it
    /// paused, a stroke's end or a save included, and it comes back by itself, which ends all this.
    fn watch_protection(&mut self, now: Instant) {
        if let Protection::Paused(cause) = self.project.protection() {
            let trouble = SaveTrouble::of_source(&*cause);
            self.falls_short(
                Shortfall::Paused,
                trouble,
                &*cause,
                "crash protection paused",
            );
        } else if self
            .exposed
            .is_some_and(|known| known.shortfall == Shortfall::Paused)
        {
            self.exposed = None;
            saving::log("brushcraft: crash protection is back");
            if self.dialogs.protection_back(now) {
                self.scheduler.want(Reason::Overlay);
            }
        }
        self.tell_of_stopping(now);
    }

    /// Tells the user of the recovery file falling short, if they haven't been and the bar is free;
    /// and what the bar says already stops saying the work is safe.
    fn tell_of_stopping(&mut self, now: Instant) {
        if self.exposed.is_some() {
            self.saving.told_exposed();
            if self.dialogs.failure_exposed(now) {
                self.scheduler.want(Reason::Overlay);
            }
        }
        if self.lost_said == LostSaid::OnTheBar {
            // Kept until it has been read: a save before that still says so.
            if self.dialogs.failure_shown(now) {
                self.lost.clear();
                self.lost_said = LostSaid::NotYet;
            }
        } else if !self.lost.is_empty()
            && self.dialogs.can_tell_of_stopping()
            && !self.dialogs.pen_is_down()
        {
            self.lost_said = LostSaid::OnTheBar;
            let protection = if self.exposed.is_some() {
                Protected::No
            } else {
                Protected::Yes
            };
            self.dialogs
                .failed(SaveFailure::edits_gone(protection), now);
            self.scheduler.want(Reason::Overlay);
            return;
        }
        if let Some(exposed) = &mut self.exposed
            && !exposed.told
            && self.dialogs.can_tell_of_stopping()
        {
            exposed.told = true;
            self.dialogs.protection_stopped(exposed.trouble, now);
            self.scheduler.want(Reason::Overlay);
        }
    }

    /// The window was asked to close: the pen's samples are no longer taken, and the brush lands
    /// what it holds over the frames that follow, however long that takes, with nothing here
    /// waiting for it. [`Owner::ready_to_close`] says when it has.
    ///
    /// # Errors
    ///
    /// As [`Brush::begin_closing`].
    pub(crate) fn request_close(&mut self) -> Result<()> {
        if self.closing() {
            return Ok(());
        }
        self.close = Close::Asked;
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

    /// A step of a close's landing, with or without frames: a window hidden, or never sized,
    /// lands all the same. Returns whether any landed; the caller's events come between steps.
    ///
    /// # Errors
    ///
    /// As [`Brush::close_step`].
    pub(crate) fn close_step(&mut self) -> Result<bool> {
        let size = self.size.or_else(|| self.last_view.map(|(_, size)| size));
        let Some(size) = size.filter(|_| self.closing()) else {
            return Ok(false);
        };
        let landed = self.with_brush(size, B::close_step)?;
        if landed {
            self.scheduler.want(Reason::Stroke);
        }
        Ok(landed)
    }

    /// Whether the window was asked to close.
    fn closing(&self) -> bool {
        self.close == Close::Asked
    }

    /// Whether a close was asked for, nothing is left to land (nothing was ever drawn, or the
    /// brush has landed what it held), and no save is under way.
    pub(crate) fn ready_to_close(&self) -> bool {
        let drawn = self.size.is_some() || self.last_view.is_some();
        let saved = self.saving.stage().is_none() && !self.close_held();
        self.closing() && saved && (self.landing() == 0 || !drawn)
    }

    /// Whether the window stays, a close waiting for a further close: its save failed.
    pub(crate) fn close_held(&self) -> bool {
        self.saving.failed_closing().is_some()
    }

    /// Ctrl+S, at `now`: a save of everything drawn so far starts, answered at once, and goes on
    /// through [`Owner::save_step`] (see [`crate::saving`]). With a save under way this is the
    /// same save; with no edits there is nothing to do, and the overlay says so.
    pub(crate) fn request_save(&mut self, now: Instant) {
        if self.saving.stage().is_some() {
            return;
        }
        self.scheduler.want(Reason::Input);
        // Out of date from here on, whatever comes of this.
        self.dialogs.save_asked(now);
        if !self.unsaved() && !self.pen_queued(now) {
            self.saving.nothing_to_save();
            return;
        }
        self.saving.start(now, self.brush.drops());
        // A close takes every stroke there is, so none has to wait.
        if !self.closing() {
            self.brush.begin_saving(now);
        }
    }

    /// A step of the save under way, if one is, whether or not a frame can be drawn; the caller's
    /// events come between steps. A failure doesn't reach the caller: the app goes on, and the
    /// failure bar, the title and the overlay say it ([`crate::saving`]).
    ///
    /// A save the window exits after ends in the window's going, if it succeeded
    /// ([`Owner::exit_step`]).
    pub(crate) fn save_step(&mut self) -> SaveStep {
        let step = match self.saving.stage() {
            None => SaveStep::Idle,
            Some(Stage::PenDown) if self.pen_held() && !self.closing() => SaveStep::Idle,
            Some(Stage::PenDown) => {
                self.brush.begin_landing();
                self.saving.reach(Stage::Landing);
                SaveStep::Worked
            }
            Some(Stage::Landing) => self.land_for_save(),
            Some(Stage::Writing) => self.write_save(),
        };
        if self.after_save != AfterSave::Stay {
            self.exit_step(Instant::now());
        }
        step
    }

    /// With the window to exit after the save, at `now`: the bar's progress while it goes on, with
    /// or without frames, and the outcome once it ends. Only a save that succeeded lets the window
    /// go, and only if it holds all that was drawn: one that failed, or that dabs were dropped
    /// from before it began, leaves the window and the user's work. The failure bar says so, and
    /// then the user has been told, so a further close can leave, the recovery file kept
    /// ([`Owner::leaves_on_failure`]).
    ///
    /// The save fixed what it holds as it began, so a stroke that waited for it isn't in it. Save
    /// on the bar means all of it: another save follows, under the same bar.
    fn exit_step(&mut self, now: Instant) {
        if self.saving.stage().is_some() {
            self.dialogs.update(self.facts(self.unsaved()), now);
            if self.dialogs.needs_frame(now) {
                self.scheduler.want(Reason::Overlay);
            }
            return;
        }
        let saved = self.saving.last_succeeded() == Some(true);
        let lost = self.brush.dropped_since(self.told_drops);
        if saved && lost.is_ok() {
            if self.unsaved() || self.pen_queued(now) {
                self.request_save(now);
                self.dialogs.finish(self.save_counts(), now);
            } else {
                self.after_save = AfterSave::Stay;
                self.request = Some(WindowRequest::Exit);
            }
            return;
        }
        if let (true, Err(dropped)) = (saved, lost) {
            self.stroke_short_of_the_save(dropped, now);
        }
        self.after_save = AfterSave::Stay;
        if self.dialogs.finishing() {
            self.dialogs.clear_prompt(now);
        }
        self.scheduler.want(Reason::Overlay);
    }

    /// A save failed at `now` for `error`: the log, the title and the failure bar say so. Whatever
    /// the error, the bar owns up to any dabs the user hasn't heard of, and then they have been
    /// told.
    fn save_failed(&mut self, error: &anyhow::Error, now: Instant) {
        self.saving.failed(error, self.closing());
        // A retry decides afresh whether the window can go before its save is written.
        self.after_retry = match self.after_save {
            AfterSave::Unseen(_) => AfterSave::Exit,
            after @ (AfterSave::Stay | AfterSave::Exit) => after,
        };
        let failure = self.failure_of(error);
        self.told_drops = self.brush.drops();
        self.tell_failure(failure, now);
    }

    /// The failed save `error` as the user hears of it: whatever the error, it owns up to any dabs
    /// dropped that the user hasn't heard of, and says the work isn't safe if the recovery file
    /// may lack it.
    fn failure_of(&self, error: &anyhow::Error) -> SaveFailure {
        let work = match self.brush.dropped_since(self.told_drops) {
            Ok(()) => Work::Whole,
            Err(_) => Work::LessAStroke,
        };
        let failure = SaveFailure::of(error, work, &self.project.protection());
        if self.exposed.is_some() {
            failure.exposed()
        } else {
            failure
        }
    }

    /// The bar says `failure` at `now`, which a close can leave on.
    fn tell_failure(&mut self, failure: SaveFailure, now: Instant) {
        self.saving.tell(failure);
        self.dialogs.failed(failure, now);
        self.scheduler.want(Reason::Overlay);
    }

    /// A save worked, at `now`, but dabs were `dropped` before it that nobody has heard of: it
    /// mustn't say only Saved, and it mustn't say it couldn't.
    fn stroke_short_of_the_save(&mut self, dropped: DabsDropped, now: Instant) {
        self.saving.failed(&dabs_dropped(dropped), self.closing());
        self.told_drops = self.brush.drops();
        self.tell_failure(SaveFailure::stroke_lost_but_saved(), now);
    }

    /// Whether a save asked for can't begin landing: the pen is down on a stroke it holds, or a
    /// stroke's samples from before it wait to be routed.
    fn pen_held(&self) -> bool {
        self.brush.pen_down() || self.pen_queued(self.saving.asked())
    }

    /// Whether a stroke's samples from before `asked` wait to be routed, as they will be with
    /// frames to route them.
    fn pen_queued(&self, asked: Instant) -> bool {
        let mut queued = self.pen_queue.iter();
        let before = |sample: &PenSample| sample.phase.in_stroke() && sample.received <= asked;
        self.size.is_some() && queued.any(before)
    }

    /// A step of a save's landing: on to the writing once it holds nothing more. A close lands
    /// the same dabs through its own steps, which leave this only to wait for them.
    fn land_for_save(&mut self) -> SaveStep {
        let size = self.size.or_else(|| self.last_view.map(|(_, size)| size));
        let left = if self.closing() {
            self.landing()
        } else {
            self.brush.landing_for_save()
        };
        let Some(size) = size.filter(|_| left > 0) else {
            self.saving.reach(Stage::Writing);
            return SaveStep::Worked;
        };
        if self.closing() {
            return SaveStep::Worked;
        }
        if let AfterSave::Unseen(unseen) = self.after_save {
            return self.journal_step(unseen);
        }
        match self.with_brush(size, B::save_step) {
            Ok(landed) => {
                if landed {
                    self.surface_landed();
                }
            }
            Err(error) => self.end_save(Err(error.context("landing the dabs to save"))),
        }
        SaveStep::Worked
    }

    /// Whether the close bar's Save at `now` can let the window go before the save is written, the
    /// landing and the save finishing unseen: dabs are still landing, the brush journals them, the
    /// pen holds none back, the project has a file to save to, and the journal holds everything
    /// the save does ([`Owner::journal_holds_it`]). Otherwise the window waits for the save, as for
    /// any: a pen still down or a stroke not yet routed needs frames.
    fn may_finish_unseen(&self, now: Instant) -> bool {
        let landing = self.brush.journals() && self.brush.landing_for_save() > 0;
        let pen = self.brush.pen_down() || self.pen_queued(now);
        let saved_to = self.project.path().is_some();
        landing && !pen && saved_to && self.saving.stage().is_none() && self.journal_holds_it()
    }

    /// Whether a durable journal would keep everything the save holds, as far as the frame loop
    /// knows: crash protection is running, no flush has failed or fallen short since the last
    /// save, the open stroke's pen-down surface is (or will be) in a save point
    /// ([`Project::open_stroke_restorable`]), and the brush has placed every pen event the save
    /// holds as dabs. A stroke waiting behind the one landing is raw pen samples
    /// ([`Brush::unplaced_for_save`]), which no journal holds until they are placed; with crash
    /// protection paused, or a flush failed or short, the journal can't be trusted to keep what a
    /// failure would leave; and a journal whose pen-down is unknown restores nothing, which not
    /// every way there reports (a cut failing as the stroke began, a roll-over that couldn't
    /// carry it), so [`Project::flushed`] alone can't say it.
    fn journal_holds_it(&self) -> bool {
        let guarded = self.project.protection().is_active() && self.exposed.is_none();
        guarded && self.project.open_stroke_restorable() && self.brush.unplaced_for_save() == 0
    }

    /// Asks the project at `now` for a flush of every edit and every dab journalled so far, so the
    /// window can go once it is durable ([`Owner::journal_step`]). A project that can't take it
    /// leaves the save to finish in the window, as any.
    fn flush_to_finish(&mut self, now: Instant) {
        let asked = self.project.request_flush(now);
        self.after_save = match self.heard(asked, "asking for a flush to close") {
            Ok(()) if self.journal_holds_it() => AfterSave::Unseen(Unseen::Flushing),
            Ok(()) => AfterSave::Exit,
            Err(error) => {
                saving::log(&format!("brushcraft: {error:#}"));
                AfterSave::Exit
            }
        };
    }

    /// A step of the wait for the journal, in place of the save's landing while the window is to
    /// go unseen: the recovery writer polled (the save's landing holds the frames' landing back,
    /// so nothing new lands meanwhile), and once every edit and dab placed is in a durable save
    /// point, the window asked to go ([`WindowRequest::Finish`]). Expected: milliseconds, one
    /// flush's syncs. Until then each pass asks for a flush again ([`Project::request_flush`]):
    /// that starts nothing with nothing waiting, and queues at most one behind the flush in
    /// flight, so whatever else edits the project meanwhile (a store job, say) is taken too.
    ///
    /// If the journal stops holding everything the save does ([`Owner::journal_holds_it`]:
    /// protection paused, a flush failed or short, the pen-down unknown, a pen event not placed),
    /// the save goes on in the window instead, landing and then writing as any, and the window
    /// waits for it.
    fn journal_step(&mut self, unseen: Unseen) -> SaveStep {
        if unseen == Unseen::Asked {
            return SaveStep::Idle;
        }
        let polled = self.project.poll();
        if let Err(error) = self.heard(polled, "polling the project") {
            saving::log(&format!("brushcraft: {error:#}"));
            self.after_save = AfterSave::Exit;
            return SaveStep::Worked;
        }
        if !self.journal_holds_it() {
            self.after_save = AfterSave::Exit;
            return SaveStep::Worked;
        }
        if self.project.flushed() {
            self.after_save = AfterSave::Unseen(Unseen::Asked);
            self.request = Some(WindowRequest::Finish);
            return SaveStep::Worked;
        }
        self.flush_to_finish(Instant::now());
        SaveStep::Waiting
    }

    /// Dabs landed outside the frames' brush steps: the next frame shows them, the next pick is
    /// made afresh, and the camera's boundary grows around them.
    fn surface_landed(&mut self) {
        self.scheduler.want(Reason::Stroke);
        self.fitting.bounds = true;
        if let Some(picker) = &mut self.picker {
            picker.surface_changed();
        }
    }

    /// A step of a save's writing: a slice of its work done ahead, then, with only the fixed part
    /// left, the save. A close may have let strokes go that waited for the save: it waits for them
    /// to land, as a stroke's op must end before a save holds it. With dabs dropped the save
    /// fails, and with no edits, after a stroke of no dabs, there is nothing to save.
    fn write_save(&mut self) -> SaveStep {
        let drawn = self.size.is_some() || self.last_view.is_some();
        if self.closing() && drawn && self.landing() > 0 {
            return SaveStep::Worked;
        }
        if let Err(error) = self.brush.dropped_since(self.saving.drops_at()) {
            self.end_save(Err(dabs_dropped(error)));
            return SaveStep::Worked;
        }
        if !self.project.has_unsaved_edits() {
            self.saving.nothing_to_save();
            self.save_over();
            return SaveStep::Worked;
        }
        match self.project.save_ahead(Instant::now() + SAVE_SLICE) {
            Ok(SaveAhead::Flushing) => SaveStep::Waiting,
            Ok(SaveAhead::Copying(left)) => {
                self.saving.copying(left);
                SaveStep::Worked
            }
            Ok(SaveAhead::Ready) => {
                let saved = self.project.save().context("saving the project");
                self.end_save(saved);
                SaveStep::Worked
            }
            Err(error) => {
                self.end_save(Err(anyhow::Error::new(error).context("saving the project")));
                SaveStep::Worked
            }
        }
    }

    /// The save under way ended in `saved`: said, and strokes go on.
    fn end_save(&mut self, saved: Result<SaveOutcome>) {
        match saved {
            Ok(outcome) => {
                let now = Instant::now();
                // Saved, the work is the last save: there's no other to go back to.
                if self.dialogs.says_restored() {
                    self.dialogs.clear_prompt(now);
                }
                self.saved(&outcome, now);
            }
            Err(error) => self.save_failed(&error, Instant::now()),
        }
        self.save_over();
    }

    /// A save went through at `now`, with `outcome`: said, and a recovery file set aside goes. One
    /// without some tiles' latest edits is a failed save for the user ([`crate::saving`]).
    fn saved(&mut self, outcome: &SaveOutcome, now: Instant) {
        if let Err(error) = self.hear_brush() {
            saving::log(&format!("brushcraft: {error:#}"));
        }
        self.saving.saved(outcome, now);
        self.forget_kept_recovery();
        self.lost.extend(outcome.unprotected.iter().copied());
        if !self.lost.is_empty() {
            self.lost_said = LostSaid::NotYet;
            let lost = TilesLost(std::mem::take(&mut self.lost).len());
            self.save_failed(&anyhow::Error::new(lost), now);
        } else if let Err(dropped) = self.brush.dropped_since(self.told_drops) {
            self.stroke_short_of_the_save(dropped, now);
        } else {
            self.dialogs.saved(now);
            let lacking = |exposed: Exposed| exposed.shortfall == Shortfall::Lacking;
            if self.exposed.is_some_and(lacking) {
                self.exposed = None;
            }
        }
    }

    /// Removes the recovery file set aside by Use last save: there is no going back to it now. A
    /// failure is said and forgotten: the file comes back as found work at worst.
    fn forget_kept_recovery(&mut self) {
        if let Some(path) = self.kept_recovery.take()
            && let Err(error) = std::fs::remove_file(&path)
        {
            let path = path.display();
            saving::log(&format!("brushcraft: removing {path}: {error}"));
        }
    }

    /// The save under way is over: strokes go on.
    fn save_over(&mut self) {
        if !self.closing() {
            self.brush.end_saving();
        }
        // The overlay's line, and what the rebase closed.
        self.scheduler.want(Reason::Converging);
    }

    /// Whether anything drawn isn't in a save: edits of the project, or dabs the brush holds.
    fn unsaved(&self) -> bool {
        self.project.has_unsaved_edits() || self.landing() > 0
    }

    /// The dabs the title and the overlay count down: a save's, or a close's.
    fn save_counts(&self) -> usize {
        if self.closing() {
            self.landing()
        } else {
            self.brush.landing_for_save()
        }
    }

    /// The window's title, if it no longer says what it should ([`crate::saving`]).
    pub(crate) fn title_change(&mut self) -> Option<String> {
        let (unsaved, left) = (self.unsaved(), self.save_counts());
        self.saving.title_change(unsaved, left, self.closing())
    }

    /// Whether the last save went through.
    #[cfg(test)]
    pub(crate) const fn saving_succeeded(&self) -> Option<bool> {
        self.saving.last_succeeded()
    }

    /// The stage of the save under way, if one is.
    #[cfg(test)]
    pub(crate) const fn save_stage(&self) -> Option<Stage> {
        self.saving.stage()
    }

    /// The overlay's line about the project.
    #[cfg(test)]
    pub(crate) fn save_line(&self) -> String {
        self.saving
            .line(self.unsaved(), self.save_counts(), self.closing())
    }

    /// Closes the project: lets the brush apply what it holds and end its stroke, finishes a save
    /// under way (the window has gone, and the user asked for it) unless that went wrong, leaves
    /// the feed, then closes the project, keeping its recovery file only if it has edits no save
    /// holds.
    ///
    /// # Errors
    ///
    /// As [`Brush::flush`], [`Project::save`] and [`Project::close`]; a save that failed with the
    /// window closing, or can't be made as dabs were dropped.
    pub(crate) fn close(mut self) -> Result<()> {
        let flushed = self.flush_brush();
        let saved = self.last_save(flushed.is_ok());
        let closed = self.close_project();
        said_too(said_too(flushed, saved), closed)
    }

    /// Finishes unseen what the close bar's Save left when the window went at once
    /// ([`WindowRequest::Finish`]), with no window and no frames (the dab journal spec, "Finishing
    /// unseen"): the brush lands everything it holds in full batches ([`Brush::close_step`], up
    /// to [`crate::sculpting::WAITED_BATCH`] of finest work each), the project ticked and polled
    /// between them, so the journal and the landed tiles keep reaching the recovery file at least
    /// once a second and a kill at any moment loses about a second; then the save under way is
    /// written, and the project closes, its recovery file removed once the save holds everything.
    ///
    /// **A failure** (a landing that fails, dabs dropped, a save that fails, a full disk) keeps
    /// the recovery file, which holds the work: opening the project brings it back. Nobody is at a
    /// window to read it, so it is said through `notify`, as a desktop notification's summary and
    /// body ([`SaveFailure::notice`]), once the project has closed, so it says what the recovery
    /// file kept; and it is the run's error, as a close's is.
    ///
    /// **Asked to stop** (a takeover, [`crate::takeover`]): before each batch it looks whether
    /// `asked` says so. Once it does, the finisher stops where it is ([`Owner::stop_unseen`]): it
    /// flushes, waits for the save point to be durable, and closes the project keeping the
    /// recovery file, which holds every dab placed and the count of those landed, for the opener to
    /// restore. Nothing is saved and nobody is notified: [`Finished::Stopped`].
    ///
    /// # Errors
    ///
    /// As [`Owner::close`], and any error of the landing's steps or the upkeep between them; a
    /// stop has none of its own ([`Owner::stop_unseen`]).
    pub(crate) fn finish_unseen_unless(
        mut self,
        notify: impl FnOnce(&str, &str),
        mut asked: impl FnMut() -> bool,
    ) -> Result<Finished> {
        let landing = self.land_unseen(&mut asked);
        if matches!(landing, Ok(LandedUnseen::Stopped)) {
            return Ok(self.stop_unseen());
        }
        let landed = said_too(landing.map(drop), self.flush_brush());
        let left = self.landing();
        let all = if left == 0 {
            Ok(())
        } else {
            Err(anyhow!("not saved: {left} dabs the brush held didn't land"))
        };
        let landed = said_too(landed, all);
        let whole = landed.is_ok();
        let saved = said_too(landed, self.last_save(whole));
        let (saved, failure) = match saved {
            Err(error) => {
                let failure = self.failure_of(&error);
                (Err(error), Some(failure))
            }
            // Saved, but with dabs dropped that nobody was told of: said as the window's save says
            // it, and the run's error.
            Ok(()) => match self.brush.dropped_since(self.told_drops) {
                Ok(()) => (Ok(()), None),
                Err(dropped) => {
                    let failure = SaveFailure::stroke_lost_but_saved();
                    (Err(dabs_dropped(dropped)), Some(failure))
                }
            },
        };
        let name = self.saving.name().to_owned();
        let closed = self.close_project();
        if let Some(failure) = failure {
            // The close decides what the recovery file kept.
            let uncovered = closed.as_ref().err().is_some_and(saving::says_uncovered);
            let failure = if uncovered {
                failure.exposed()
            } else {
                failure
            };
            let (summary, body) = failure.notice(&name);
            notify(&summary, &body);
        }
        said_too(saved, closed).map(|()| Finished::Done)
    }

    /// [`Owner::finish_unseen_unless`], never asked to stop.
    ///
    /// # Errors
    ///
    /// As [`Owner::finish_unseen_unless`].
    #[cfg(test)]
    pub(crate) fn finish_unseen(self, notify: impl FnOnce(&str, &str)) -> Result<()> {
        self.finish_unseen_unless(notify, || false).map(drop)
    }

    /// The finisher stops for a takeover: flushes every edit and dab journalled so far, waits for
    /// that save point to be durable ([`Owner::flush_to_stop`]), and closes the project keeping
    /// its recovery file, which releases both files' locks. The dabs the brush still holds are
    /// left to the journal: the opener restores them, and lands them.
    ///
    /// Nothing here fails the run: a takeover is no failure. A flush that couldn't be made
    /// durable (crash protection paused, the writer failed), or a close that went wrong, leaves the
    /// recovery file at its last durable save point, as a crash would: said on stderr, and in
    /// [`Finished::Stopped`]'s `flushed`.
    fn stop_unseen(mut self) -> Finished {
        let mut flushed = self.flush_to_stop();
        let Self {
            mut project,
            cursor,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        if let Err(error) = project.close(true) {
            let why = anyhow::Error::new(error);
            saving::log(&format!(
                "brushcraft: closing the project to hand it over: {why:#}"
            ));
            flushed = false;
        }
        Finished::Stopped { flushed }
    }

    /// Flushes every edit and dab journalled so far, and waits until that is in a durable save
    /// point ([`Project::flushed`]), for [`STOP_FLUSH_LIMIT`] at most: whether it got there.
    /// Crash protection paused or failing ends the wait at once: nothing more can be made durable.
    fn flush_to_stop(&mut self) -> bool {
        let began = Instant::now();
        loop {
            if self.project.flushed() {
                return true;
            }
            let guarded = self.project.protection().is_active() && self.exposed.is_none();
            if !guarded || began.elapsed() >= STOP_FLUSH_LIMIT {
                return false;
            }
            let asked = self.project.request_flush(Instant::now());
            let polled = self
                .heard(asked, "asking for a flush to stop")
                .and_then(|()| {
                    let polled = self.project.poll();
                    self.heard(polled, "polling the project")
                });
            if let Err(error) = polled {
                saving::log(&format!("brushcraft: {error:#}"));
                return false;
            }
            std::thread::sleep(STOP_FLUSH_POLL);
        }
    }

    /// The finisher's landing ([`Owner::finish_unseen_unless`]): the pen's samples no longer taken,
    /// then batch after batch of what the brush holds until it holds nothing, the project's
    /// recovery writer tended before each and once after.
    ///
    /// A landing error doesn't stop it: what is left lands all the same, ticked as ever, so the
    /// recovery file keeps up with it whatever the save then makes of the error. An error tending
    /// the recovery writer does (the store is broken): the recovery file stays at its last durable
    /// save point, as after a crash. It also stops once [`FINISH_STALLS`] batches in a row have
    /// changed nothing, which leaves what is left to [`Owner::flush_brush`].
    ///
    /// # Errors
    ///
    /// The first of [`Owner::request_close`], [`Owner::close_step`] and [`Owner::tend_recovery`]
    /// that failed, those after it as its context.
    ///
    /// **Asked to stop:** `asked` is looked at before each batch, the recovery writer just
    /// tended; once it says so the landing ends there, [`LandedUnseen::Stopped`], and an error met
    /// before it is said on stderr alone: the recovery file holds the work, for the opener.
    fn land_unseen(&mut self, asked: &mut impl FnMut() -> bool) -> Result<LandedUnseen> {
        let mut outcome = self.request_close();
        let mut stalls = 0;
        loop {
            if let Err(error) = self.tend_recovery(Instant::now()) {
                return said_too(outcome, Err(error)).map(|()| LandedUnseen::Landed);
            }
            if asked() {
                if let Err(error) = outcome {
                    saving::log(&format!(
                        "brushcraft: stopping for a takeover after: {error:#}"
                    ));
                }
                return Ok(LandedUnseen::Stopped);
            }
            let before = self.landing();
            if before == 0 || stalls == FINISH_STALLS {
                break;
            }
            let landed = match self.close_step() {
                Ok(landed) => landed,
                Err(error) => {
                    outcome = said_too(outcome, Err(error));
                    false
                }
            };
            let moved = landed || self.landing() != before;
            stalls = if moved { 0 } else { stalls + 1 };
        }
        said_too(outcome, self.tend_recovery(Instant::now())).map(|()| LandedUnseen::Landed)
    }

    /// Has the brush apply what it still holds and end its stroke, as the project is about to
    /// close; nothing if it never had a view to place dabs through.
    ///
    /// # Errors
    ///
    /// As [`Brush::flush`].
    fn flush_brush(&mut self) -> Result<()> {
        match self.size.or_else(|| self.last_view.map(|(_, size)| size)) {
            Some(size) => {
                let flushed = self.with_brush(size, B::flush);
                flushed.context("ending the brush's stroke")
            }
            None => Ok(()),
        }
    }

    /// Leaves the feed and closes the project, keeping its recovery file only if it has edits no
    /// save holds. A scripted run fails on what fell short of the recovery file.
    ///
    /// # Errors
    ///
    /// As [`Project::close`], or what fell short of the recovery file in a scripted run.
    fn close_project(self) -> Result<()> {
        let Self {
            mut project,
            cursor,
            fell_short,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        let keep = project.has_unsaved_edits();
        let closed = project.close(keep).context("closing the project");
        let protected = fell_short.map_or(Ok(()), |why| Err(anyhow!("recovery fell short: {why}")));
        said_too(closed, protected)
    }

    /// Closes the project without saving: Don't save. What the brush holds is dropped, not landed,
    /// a save under way goes with it, what the project file's readers see stays as it was, and
    /// the recovery file is removed, even on a disk too full to finish a flush. Nothing to wait
    /// for. A recorded run's stream ends as the owner goes, as in [`Owner::close`], and the run's
    /// end writes the record over it as ever.
    ///
    /// # Errors
    ///
    /// As [`Project::discard`].
    pub(crate) fn discard(mut self) -> Result<()> {
        self.brush.abandon();
        let Self {
            mut project,
            cursor,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        project.discard().context("discarding the project")
    }

    /// What a closing project does about its save: says that one failed and the window held for it
    /// or left on it, finishes one under way
    /// if everything `landed` and no dabs were dropped since it was asked for.
    ///
    /// # Errors
    ///
    /// The failure, or why the save wasn't made: its edits stay unsaved, in the recovery file.
    fn last_save(&mut self, landed: bool) -> Result<()> {
        if let Some(why) = self.saving.run_failed_on() {
            return Err(anyhow!("saving the project: {why}"));
        }
        self.hear_brush()?;
        if self.saving.stage().is_none() {
            return self.nothing_lost_untold();
        }
        let dropped = self.brush.dropped_since(self.saving.drops_at());
        dropped.context("not saved: dabs were dropped")?;
        ensure!(landed, "not saved: what the brush held didn't all land");
        let outcome = self.project.save().context("saving the project")?;
        self.saving.saved(&outcome, Instant::now());
        self.forget_kept_recovery();
        self.lost.extend(outcome.unprotected.iter().copied());
        self.nothing_lost_untold()
    }

    /// Whether the run ends with nobody told of tiles gone.
    ///
    /// # Errors
    ///
    /// [`TilesLost`], if it does.
    fn nothing_lost_untold(&self) -> Result<()> {
        ensure!(self.lost.is_empty(), TilesLost(self.lost.len()));
        Ok(())
    }

    /// Closes the project so another can take over, once the render thread has settled: the file
    /// lock goes with it. Nothing is flushed or landed, as nothing was drawn
    /// ([`Owner::may_reopen`]), and the recovery file stays if it has edits no save holds, or the
    /// journal of a stroke a restore found still landing (none of its dabs may have landed yet).
    ///
    /// # Errors
    ///
    /// As [`Project::close`]: the recovery file is kept then.
    pub(crate) fn close_for_reopen(self) -> Result<Carried<B>> {
        let pen_down = self.dialogs.pen_is_down();
        let landing = self.landing() > 0;
        let Self {
            mut project,
            cursor,
            brush,
            camera,
            pivot_at,
            look,
            navigator,
            input,
            input_at,
            pen_queue,
            pen_router,
            pen_navigation,
            scheduler,
            started,
            finished,
            clock,
            recorder,
            record_stream,
            run_info,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        let keep = landing || project.has_unsaved_edits();
        project.close(keep).context("closing the project")?;
        Ok(Carried {
            brush,
            camera,
            pivot_at,
            look,
            navigator,
            input,
            input_at,
            pen_queue,
            pen_router,
            pen_navigation,
            pen_down,
            scheduler,
            started,
            finished,
            clock,
            recorder,
            record_stream,
            run_info,
        })
    }

    /// This frame loop taking over from the one that left `carried`, on the same render thread:
    /// the view stays where the user had it, rather than framing the model again.
    #[must_use]
    pub(crate) fn go_on_from(mut self, carried: Carried<B>) -> Self {
        self.brush.take_settings(&carried.brush);
        self.camera = carried.camera;
        self.pivot_at = carried.pivot_at;
        self.fitting = Fitting {
            frame: false,
            bounds: true,
        };
        self.look = carried.look;
        self.navigator = carried.navigator;
        self.input = carried.input;
        self.input_at = carried.input_at;
        self.pen_queue = carried.pen_queue;
        self.pen_router = carried.pen_router;
        self.pen_navigation = carried.pen_navigation;
        if carried.pen_down {
            self.dialogs.pen_down(Instant::now());
        }
        self.scheduler.go_on_from(&carried.scheduler);
        self.started = carried.started;
        self.finished = carried.finished;
        self.clock = carried.clock;
        self.recorder = carried.recorder;
        self.record_stream = carried.record_stream;
        self.run_info = carried.run_info;
        self
    }

    /// This frame loop leaving the recovery file at `kept` until a stroke or a save.
    #[must_use]
    pub(crate) fn keeping_recovery(mut self, kept: PathBuf) -> Self {
        self.kept_recovery = Some(kept);
        self
    }

    /// A copy of found work was saved with `outcome`: said as any save is.
    pub(crate) fn copied(&mut self, outcome: &SaveOutcome) {
        self.saved(outcome, Instant::now());
        self.scheduler.want(Reason::Overlay);
    }

    /// A copy of found work failed for `error`: the log, the title and the failure bar say so, as
    /// for a save. A note that follows asks again once the bar is answered.
    pub(crate) fn copy_failed(&mut self, error: &anyhow::Error) {
        self.saving.failed(error, false);
        self.dialogs
            .failed(SaveFailure::of_copy(error), Instant::now());
        self.scheduler.want(Reason::Overlay);
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

    /// The bar the UI shows, if any.
    #[cfg(test)]
    pub(crate) fn bar(&self) -> Option<ui::Bar<Action>> {
        self.dialogs.bar()
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

    /// The project file saves go to, if there is one: the file a finisher holds, which names its
    /// takeover socket ([`crate::takeover`]).
    pub(crate) fn project_path(&self) -> Option<&std::path::Path> {
        self.project.path()
    }

    /// The project.
    #[cfg(test)]
    pub(crate) const fn project(&mut self) -> &mut Project {
        &mut self.project
    }
}

/// How long a finisher stopping for a takeover waits for its flush ([`Owner::flush_to_stop`]).
///
/// A flush's syncs take milliseconds, so this bounds only a disk that stalls, the opener waiting
/// meanwhile.
const STOP_FLUSH_LIMIT: Duration = Duration::from_secs(3);

/// How often a finisher stopping looks at its flush.
const STOP_FLUSH_POLL: Duration = Duration::from_millis(1);

/// How the finisher's landing ended ([`Owner::land_unseen`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LandedUnseen {
    /// It landed what the brush held, or as much as it could.
    Landed,
    /// It was asked to stop, and stopped before its next batch.
    Stopped,
}

/// How a finisher ended ([`Owner::finish_unseen_unless`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finished {
    /// It landed and saved: the recovery file is gone.
    Done,
    /// It was asked to stop and did, its project closed, the recovery file kept for the opener.
    Stopped {
        /// Whether everything placed was in a durable save point as it closed; if not, the
        /// recovery file holds its last durable save point, as after a crash.
        flushed: bool,
    },
}

/// How many of the finisher's batches in a row may change nothing before it stops trying.
///
/// A batch of [`Owner::land_unseen`] that can land nothing shouldn't happen (close steps wait for a
/// stroke's cut): this guards against it, so the finisher never spins.
const FINISH_STALLS: u32 = 3;

/// Whether `error`, of a tick or a poll, is the recovery file's alone, so a run can go on after it.
///
/// A store that has found itself broken can't: a flush failing on its invariants, or the writer's
/// replies out of step, are bugs, and what they have written is in doubt.
fn recovery_trouble(error: &ProjectError) -> bool {
    let bug = matches!(
        error,
        ProjectError::Recovery(
            RecoveryError::OutOfStep | RecoveryError::Store(CacheError::Internal(_))
        )
    );
    let recovery = matches!(
        error,
        ProjectError::ProtectionPaused(_)
            | ProjectError::Lost(_)
            | ProjectError::Unprotected(_)
            | ProjectError::Recovery(_)
    );
    recovery && !bug
}

/// Whether `error`, of a stroke's end, is the frame loop's to hear ([`Owner::heard`]): the stroke
/// ended all the same, and nothing was dropped.
pub(crate) fn recovery_report(error: &ProjectError) -> bool {
    recovery_trouble(error) || matches!(error, ProjectError::OldRecoveryKept { .. })
}

/// `dropped` as the error of a save it failed.
fn dabs_dropped(dropped: DabsDropped) -> anyhow::Error {
    anyhow::Error::new(dropped).context("dabs were dropped")
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
