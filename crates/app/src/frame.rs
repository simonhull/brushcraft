//! The frame loop, on the store's owner thread (the M1 spec's "Threads": the owner thread is
//! the frame loop).
//!
//! [`Owner`] holds the project, so it lives on the one thread that may touch the store, and
//! makes each frame in a fixed order:
//!
//! 1. **input**: the pointer input queued since the last frame moves the camera;
//! 2. **brush**: a seam for M1a4's brush step ([`Brush`]; [`NoBrush`] until then), which sees the
//!    camera as the input left it;
//! 3. **feed**: the stale-tile feed is drained: the epochs closed since the last frame;
//! 4. **uploads**: GPU residency ([`Uploads`], [`crate::uploads::GpuResidency`]) takes the drained
//!    epochs and stages the frame's tiles from the store, until half the way to the frame's
//!    deadline;
//! 5. **hand-off**: the frame goes to the render thread, as a [`FrameRequest`] carrying the staged
//!    uploads, which it starts encoding at once (the uploads first, then the passes);
//! 6. **upkeep**: meanwhile the project's recovery writer is ticked and polled;
//! 7. **jobs**: and the store's resumable jobs (display-layer refresh and rebuild) run until the
//!    frame's deadline, at least one slice of them. This frame draws nothing they produce: the
//!    epochs they close are drained by the next frame, which they want.
//!
//! Handing off before the jobs keeps them out of the input's latency (the plan's order,
//! amended by the Task 1 review).
//!
//! Frames are made on demand: [`schedule::Scheduler`] decides when, from why they are wanted.
//! A frame that leaves store jobs queued, uploads pending, or epochs in the feed it hasn't
//! drained (closed by its jobs or upkeep) wants the next one, as does upkeep between frames
//! that closes epochs; input that moves the camera wants one; nothing else does.

pub(crate) mod schedule;

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use render::DrawTimes;
use render::surface::{Presentation, Size};
use render::upload::FrameUploads;
use store::{Drained, JobsReport, Project, StaleCursor, TileStore};

use self::schedule::{Next, Reason, Scheduler};
use crate::camera::{Arcball, Camera};
use crate::navigate::{Input, Navigator};
use crate::overlay::{self, Snapshot};
use crate::render_thread::{FrameRequest, Presented};

/// The brush's place in the frame: M1a4's brush step, run on the owner thread after input and
/// before the feed is drained, so the epochs its edits close reach the GPU in the same frame.
pub(crate) trait Brush {
    /// One frame's brush work on `project`: the stroke samples queued since the last frame,
    /// applied, and their epoch closed. `camera` is the view as this frame's input left it,
    /// which the samples are projected through.
    ///
    /// # Errors
    ///
    /// If the store fails the edit.
    fn step(&mut self, project: &mut Project, camera: &Camera) -> Result<()>;
}

/// No brush yet (M1a4): the step does nothing.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoBrush;

impl Brush for NoBrush {
    fn step(&mut self, _project: &mut Project, _camera: &Camera) -> Result<()> {
        Ok(())
    }
}

/// What the uploads did with a frame's epochs, and what they staged for it.
#[derive(Debug, Default)]
pub(crate) struct UploadStatus {
    /// Whether uploads are still pending (the floor not all up, an epoch going up over several
    /// frames, requested tiles still to come): the next frame is wanted.
    pub(crate) converging: bool,
    /// What the GPU holds, for the overlay's tiles line; `None` while it holds nothing.
    pub(crate) summary: Option<String>,
    /// The frame's staged uploads, which travel to the render thread in its
    /// [`FrameRequest`]; `None` if there is no GPU residency.
    pub(crate) frame: Option<FrameUploads>,
}

/// The uploads' place in the frame: GPU residency ([`crate::uploads::GpuResidency`]).
///
/// It takes each frame's drained epochs and stages the tiles the GPU should get into its
/// staging ring, on the owner thread (the store is read here, and only here), for the render
/// thread to encode ahead of the frame's passes.
pub(crate) trait Uploads {
    /// Applies `drained`, the epochs closed since the last frame (or a resync), and stages the
    /// frame's uploads, reading tiles from `store` until `deadline` (the floor goes up whole,
    /// whatever the deadline).
    ///
    /// # Errors
    ///
    /// If the store can't give a tile, or the GPU refuses an upload.
    fn apply(
        &mut self,
        drained: Drained,
        store: &mut TileStore,
        deadline: Instant,
    ) -> Result<UploadStatus>;
}

/// No GPU residency, for the frame loop's tests without a GPU: nothing is uploaded, and the
/// epochs are dropped.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoUploads;

#[cfg(test)]
impl Uploads for NoUploads {
    fn apply(
        &mut self,
        _drained: Drained,
        _store: &mut TileStore,
        _deadline: Instant,
    ) -> Result<UploadStatus> {
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

/// How long one frame's phases took on the owner thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct OwnerPhases {
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
            Drained::Resync => self.resyncs += 1,
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
        let mut scheduler = Scheduler::new(setup.interval, now);
        scheduler.want(Reason::Exposed);
        if setup.frames.is_some() {
            scheduler.want(Reason::Scripted);
        }
        Self {
            project,
            cursor,
            brush,
            uploads,
            camera: Camera::default(),
            navigator: Navigator::default(),
            input: Vec::new(),
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
            log: setup.frames.map(|_| PhaseLog::default()),
        }
    }

    /// Queues pointer input for the next frame's input phase, and wants that frame if the
    /// input may move the camera.
    pub(crate) fn input(&mut self, input: Input) {
        if self.navigator.may_move(&self.input, &input) {
            self.scheduler.want(Reason::Input);
        }
        self.input.push(input);
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
    /// (a minimised window) or once a scripted run has made its frames; upkeep stays due on
    /// its own clock meanwhile.
    pub(crate) fn next(&self, now: Instant) -> Next {
        let next = self.scheduler.next(now);
        let no_frames =
            self.size.is_none() || self.limit.is_some_and(|limit| self.started >= limit);
        match next {
            Next::Frame { .. } | Next::Wait { .. } if no_frames => Next::Idle {
                upkeep: self.scheduler.upkeep_due(),
            },
            Next::Frame { .. } | Next::Wait { .. } | Next::Render | Next::Idle { .. } => next,
        }
    }

    /// Whether a scripted run has made all its frames and the render thread has finished them.
    pub(crate) fn done(&self) -> bool {
        self.limit.is_some_and(|limit| self.finished >= limit)
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
        self.scheduler.started(now);
        self.started += 1;
        let mut phases = OwnerPhases {
            upkeep: self.after_hand_off.upkeep,
            jobs: self.after_hand_off.jobs,
            ..OwnerPhases::default()
        };
        let mut lap = Lap::new();

        let arcball = Arcball::new(size);
        let height = f64::from(size.height());
        let input = std::mem::take(&mut self.input);
        for event in input {
            self.navigator
                .apply(event, &mut self.camera, &arcball, height);
        }
        phases.input = lap.next();

        self.brush
            .step(&mut self.project, &self.camera)
            .context("the brush step")?;
        phases.brush = lap.next();

        let drained = self.project.store().drain_stale(&self.cursor);
        self.feed.count(&drained);
        phases.feed = lap.next();

        let mut uploads = self
            .uploads
            .apply(
                drained,
                self.project.store(),
                uploads_deadline(now, deadline),
            )
            .context("uploading tiles")?;
        phases.uploads = lap.next();

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
            tiles: uploads.summary.as_deref(),
            camera: self.camera,
        };
        let overlay = overlay::lines(&snapshot);
        hand_off(FrameRequest {
            number: self.started,
            size,
            overlay_scale: self.overlay_scale,
            overlay,
            uploads: uploads.frame.take(),
        })?;
        lap.next();

        self.project.tick(now).context("ticking the project")?;
        self.project.poll().context("polling the project")?;
        self.after_hand_off.upkeep = lap.next();

        let deadline = deadline.max(Instant::now() + JobsReport::SLICE_TARGET);
        self.after_hand_off.report = self
            .project
            .store()
            .run_jobs(deadline)
            .context("running the store's jobs")?;
        self.after_hand_off.jobs = lap.next();
        if let Some(log) = &mut self.log {
            log.owner.push(OwnerPhases {
                upkeep: self.after_hand_off.upkeep,
                jobs: self.after_hand_off.jobs,
                ..phases
            });
        }

        if uploads.converging || self.project.store().has_jobs() || self.has_stale() {
            self.scheduler.want(Reason::Converging);
        }
        if self.limit.is_some_and(|limit| self.started < limit) {
            self.scheduler.want(Reason::Scripted);
        }
        Ok(())
    }

    /// The render thread finished a frame.
    pub(crate) fn shown(&mut self, presented: Presented) {
        self.finished += 1;
        self.render = Some(presented.times);
        if let Some(log) = &mut self.log {
            log.render.push(presented.times);
        }
        self.scheduler.shown(presented.outcome);
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

    /// Closes the project: leaves the feed, then closes it, keeping its recovery file only if
    /// it has edits no save holds (the app can't ask yet).
    ///
    /// # Errors
    ///
    /// As [`Project::close`].
    pub(crate) fn close(self) -> Result<()> {
        let Self {
            mut project,
            cursor,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        let keep = project.has_unsaved_edits();
        project.close(keep).context("closing the project")
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
