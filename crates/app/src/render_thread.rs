//! The render thread: draws each frame the owner thread hands it (the M1 spec's "Threads": it
//! only encodes and submits).
//!
//! It owns the [`Target`] and a [`Renderer`], which holds the frame's passes in order. Frames
//! arrive one at a time over a channel ([`FrameRequest`]); each has its overlay text laid out,
//! is drawn ([`draw_frame`]: the uploads, the scene, the overlay) and presented, and what
//! happened goes back ([`Presented`]), with a wake-up for the owner's event loop. The scene's
//! passes come with the first frame ([`FrameRequest::passes`]) and stay here. Dropping the
//! sender ends the thread, which hands the target back. The thread wakes the owner as it ends,
//! however it ends (a panic included), so the owner always learns of it.
//!
//! **When the GPU finishes a frame:** after drawing each frame the thread asks the queue to say
//! when the work submitted so far is done (`Queue::on_submitted_work_done`), and the callback
//! records the frame's number as finished. Callbacks (these, and the readbacks' maps) run in
//! whichever thread polls the device next. Two threads do:
//!
//! - the owner: through [`RenderThread::gpu_finished`] while it waits for the GPU, and at the start
//!   of each frame (residency's `begin_frame`), both only while this thread holds no frame;
//! - this thread: `Queue::submit` runs the device's upkeep (`maintain(Poll)`) as it submits a
//!   frame, before that frame's maps are requested, so it can complete only earlier frames'.
//!
//! So no poll falls between two maps one frame requests. Nothing waits on the GPU for it, so the
//! owner can keep at most [`render::FRAMES_IN_FLIGHT`] frames unfinished on the GPU without ever
//! blocking.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use geom::Level;
use render::aa::AaReport;
use render::camera::View;
use render::cull::{BoundsUpdate, SlotBuffers};
use render::gpu::Gpu;
use render::raster::{FramePasses, SceneInputs};
use render::surface::{Presentation, Size, Target};
use render::timing::GpuTimes;
use render::upload::FrameUploads;
use render::{DrawTimes, DroppedReadbacks, FrameInputs, Look, Outcome, Renderer, Scene};
use ui::{Blend, UiFrame};
use wgpu::{Buffer, Color, PollType};

use crate::overlay::TextLayout;

/// The colour frames are cleared to, linear: a dark grey with a touch of blue.
pub(crate) const CLEAR: Color = Color {
    r: 0.018,
    g: 0.018,
    b: 0.022,
    a: 1.0,
};

/// A frame for the render thread to draw: everything it needs, owned.
#[derive(Debug)]
pub(crate) struct FrameRequest {
    /// The frame's number, from 1.
    pub(crate) number: u64,
    /// The size to draw at: the window's now.
    pub(crate) size: Size,
    /// Frame pixels per overlay texel: the window's scale factor, rounded, at least 1.
    pub(crate) overlay_scale: u32,
    /// The overlay's lines.
    pub(crate) overlay: Vec<String>,
    /// The uploads the owner thread staged for this frame, encoded ahead of its passes and
    /// submitted with them, even when the frame gets no image to draw in; `None` with no GPU
    /// residency. They are dropped unsubmitted only if the frame fails to draw (which ends the
    /// run) or the request is never drawn: that frame makes no request readback, so residency
    /// keeps wanting frames, and the next one hears of the loss and stages the tiles again.
    pub(crate) uploads: Option<FrameUploads>,
    /// The scene to draw; `None` with no GPU residency (or no view to draw it from), or when
    /// the frame only shades the last scene again ([`FrameRequest::reshade`]).
    pub(crate) scene: Option<SceneRequest>,
    /// With no scene: shade the last scene drawn again, with this frame's look (a change of look
    /// alone, nothing else changed; `render::Scene::Reshade`).
    pub(crate) reshade: bool,
    /// How the scene is shown.
    pub(crate) look: Look,
    /// The scene's passes, with the first frame only, whether or not it has a scene: the render
    /// thread keeps them.
    pub(crate) passes: Option<FramePasses>,
    /// The UI to draw over everything, its fades worked out from the time as it's drawn; `None`
    /// with nothing showing.
    pub(crate) ui: Option<UiFrame>,
}

/// What residency gives a frame's scene: the buffers the scene's passes read, owned (a
/// [`Buffer`] is a shared handle), made on the owner thread.
#[derive(Debug)]
pub(crate) struct ResidentScene {
    /// The tile tree's buffer.
    pub(crate) tree: Buffer,
    /// Residency's request buffer, which the level choice appends to.
    pub(crate) requests: Buffer,
    /// The slot buffers the tree's slots name.
    pub(crate) slots: SlotBuffers,
    /// The tiles whose bounds the frame recomputes: those whose data changed since the bounds the
    /// GPU has.
    pub(crate) bounds: BoundsUpdate,
    /// The project's finest level.
    pub(crate) finest: Level,
    /// Residency's frame number, which the cut's readback carries back.
    pub(crate) frame: u64,
}

/// A frame's scene: residency's part and the frame loop's.
#[derive(Debug)]
pub(crate) struct SceneRequest {
    /// What residency gives it.
    pub(crate) resident: ResidentScene,
    /// The view, of the frame's size.
    pub(crate) view: View,
    /// The frame's time on the frame loop's clock: the morph's time, stamped on the owner
    /// thread so the render thread never reads a clock and a frame is the same given its inputs.
    pub(crate) time: Duration,
    /// Whether the camera moves this frame (the view changed since the last): level changes
    /// then don't morph by time.
    pub(crate) moving: bool,
}

impl SceneRequest {
    /// The scene's passes' inputs.
    fn inputs(&self) -> SceneInputs<'_> {
        let resident = &self.resident;
        SceneInputs {
            tree: &resident.tree,
            requests: &resident.requests,
            slots: &resident.slots,
            bounds: &resident.bounds,
            view: &self.view,
            time: self.time,
            moving: self.moving,
            finest: resident.finest,
            frame: resident.frame,
        }
    }
}

/// What the render thread did with a frame.
///
/// The frame's request readback (tiles wanted but missing) doesn't come back here: residency
/// hears of it itself, through the readback buffer's mapping
/// ([`render::residency::Residency::take_requests`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Presented {
    /// The frame's number.
    pub(crate) number: u64,
    /// Whether it reached the screen.
    pub(crate) outcome: Outcome,
    /// How long its phases took.
    pub(crate) times: DrawTimes,
    /// Whether its scene was drawn (`render::Drawn::scene`).
    pub(crate) scene: bool,
    /// Whether a scene was shown, drawn or shaded again (`render::Drawn::shown`).
    pub(crate) shown: bool,
    /// What its anti-aliasing did (`render::Drawn::aa`).
    pub(crate) aa: AaReport,
    /// The GPU times of earlier frames read back by now, each naming its frame
    /// ([`Renderer::take_timings`]); empty unless the thread times the GPU.
    pub(crate) gpu: Vec<GpuTimes>,
    /// The readbacks this frame made none of for want of a free buffer: none while the GPU is
    /// kept within [`render::FRAMES_IN_FLIGHT`] frames.
    pub(crate) dropped: DroppedReadbacks,
    /// The newest frame the GPU had finished as this one was reported (every one before it
    /// too); 0 for none.
    pub(crate) gpu_finished: u64,
    /// When the render thread's present call for it returned: its present submitted, the end of
    /// the pen's latency (the M1 spec's "input sample → present submitted"). For a frame that got
    /// no image, when its uploads were submitted.
    pub(crate) submitted: Instant,
}

/// A hook the render thread calls.
type Hook = Box<dyn Fn() + Send>;

/// What the render thread calls on the owner's behalf.
pub(crate) struct Hooks {
    /// Called after each frame's [`Presented`] is sent, and as the thread ends, however it
    /// ends: wakes the owner's event loop. Never blocks.
    wake: Hook,
    /// Called just before a frame is presented: winit's `pre_present_notify`, which lets the
    /// compositor pace the window.
    pre_present: Hook,
    /// Test hook: called with each frame before it is drawn.
    #[cfg(test)]
    before_draw: Box<dyn Fn(&FrameRequest) + Send>,
}

impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks").finish_non_exhaustive()
    }
}

impl Hooks {
    /// Hooks that wake the owner with `wake` and call `pre_present` before each present.
    pub(crate) fn new(
        wake: impl Fn() + Send + 'static,
        pre_present: impl Fn() + Send + 'static,
    ) -> Self {
        Self {
            wake: Box::new(wake),
            pre_present: Box::new(pre_present),
            #[cfg(test)]
            before_draw: Box::new(|_| {}),
        }
    }

    /// Hooks that do nothing: for a target with no event loop to wake.
    pub(crate) fn none() -> Self {
        Self::new(|| {}, || {})
    }
}

/// Calls the wake hook when dropped: when the render thread ends, by returning or by
/// unwinding from a panic, so the owner always hears of it.
struct WakeOnExit(Hook);

impl Drop for WakeOnExit {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// What a render thread hands back as it stops.
#[derive(Debug)]
pub(crate) struct Stopped {
    /// Its target: an offscreen run's screenshot reads its pixels.
    pub(crate) target: Target,
    /// The GPU times of its last frames, which no later frame resolved
    /// ([`Renderer::flush_timings`]): read back once the GPU has finished every frame. Empty
    /// unless the thread times the GPU.
    pub(crate) gpu: Vec<GpuTimes>,
}

/// The render thread, from the owner's side.
#[derive(Debug)]
pub(crate) struct RenderThread {
    /// Where frames go; `None` once stopping.
    frames: Option<SyncSender<FrameRequest>>,
    /// What happened to them.
    presented: Receiver<Presented>,
    /// The thread, which hands the target back when it ends; `None` once joined.
    handle: Option<JoinHandle<Result<Stopped>>>,
    /// How the target presents.
    presentation: Presentation,
    /// How the target blends the UI.
    blend: Blend,
    /// Whether it times each frame's phases on the GPU.
    timing: bool,
    /// The GPU, polled for finished frames.
    gpu: Arc<Gpu>,
    /// The newest frame the GPU has finished, as its completion callbacks say.
    finished: Arc<AtomicU64>,
}

impl RenderThread {
    /// Starts the render thread, drawing into `target` on `gpu`, timing each frame's phases on
    /// the GPU if `timing` asks and the device can ([`RenderThread::timing`]).
    ///
    /// # Errors
    ///
    /// If the renderer's passes (or its GPU timer) can't be built, or the thread can't be
    /// started.
    pub(crate) fn spawn(gpu: Arc<Gpu>, target: Target, hooks: Hooks, timing: bool) -> Result<Self> {
        let presentation = target.presentation();
        let blend = Renderer::ui_blend(&target);
        let mut renderer =
            Renderer::new(&gpu, target.format()).context("building the renderer's passes")?;
        let timing = timing
            && renderer
                .enable_timing(&gpu)
                .context("building the GPU's frame timer")?;
        // One frame at a time: the scheduler never sends another before this one is presented.
        let (frames, incoming) = mpsc::sync_channel(1);
        let (outgoing, presented) = mpsc::channel();
        let Hooks {
            wake,
            pre_present,
            #[cfg(test)]
            before_draw,
        } = hooks;
        let finished = Arc::new(AtomicU64::new(0));
        let polled = Arc::clone(&gpu);
        let mut thread = Drawing {
            gpu,
            target,
            renderer,
            finished: Arc::clone(&finished),
            #[cfg(test)]
            before_draw,
        };
        let handle = thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                // Locals drop in reverse: `outgoing` first, so the owner, once woken, finds the
                // channel closed; then `wake`, after the result is made or while unwinding from
                // a panic.
                let wake = WakeOnExit(wake);
                let outgoing = outgoing;
                thread.run(&incoming, &outgoing, &*wake.0, &*pre_present)?;
                let last = thread.last_timings()?;
                Ok(Stopped {
                    target: thread.target,
                    gpu: last,
                })
            })
            .context("starting the render thread")?;
        Ok(Self {
            frames: Some(frames),
            presented,
            handle: Some(handle),
            presentation,
            blend,
            timing,
            gpu: polled,
            finished,
        })
    }

    /// The newest frame the GPU has finished (every one before it too; 0 for none), polling
    /// the device first without waiting, so the callbacks of frames finished since run now.
    ///
    /// Call it only while the thread holds no frame (between a frame's [`Presented`] and the
    /// next [`RenderThread::send`]): every readback map of the frames it has drawn is requested
    /// by then, so a poll never falls between two map requests of one frame and hears of one
    /// alone. The only other thread that polls is this one, as it submits (`Queue::submit` runs
    /// `maintain(Poll)`), before the frame it submits requests its maps (the module notes).
    ///
    /// # Errors
    ///
    /// If polling the device fails (the device was lost).
    pub(crate) fn gpu_finished(&self) -> Result<u64> {
        self.gpu
            .device()
            .poll(PollType::Poll)
            .context("polling the GPU for finished frames")?;
        Ok(self.finished.load(Ordering::Acquire))
    }

    /// How the target presents.
    pub(crate) const fn presentation(&self) -> Presentation {
        self.presentation
    }

    /// How the target blends the UI, which the UI is made for.
    pub(crate) const fn blend(&self) -> Blend {
        self.blend
    }

    /// Whether it times each frame's phases on the GPU: asked to, on a device with timestamp
    /// queries.
    pub(crate) const fn timing(&self) -> bool {
        self.timing
    }

    /// Hands `frame` to the thread.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread, if it has stopped.
    pub(crate) fn send(&mut self, frame: FrameRequest) -> Result<()> {
        let sent = self
            .frames
            .as_ref()
            .ok_or_else(|| anyhow!("the render thread is stopping"))?
            .send(frame);
        match sent {
            Ok(()) => Ok(()),
            Err(_) => Err(self.failure()),
        }
    }

    /// Every frame the thread has finished since the last call, without waiting.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread, once it has stopped and everything it finished
    /// before has been taken.
    pub(crate) fn finished(&mut self) -> Result<Vec<Presented>> {
        let mut done = Vec::new();
        loop {
            match self.presented.try_recv() {
                Ok(presented) => done.push(presented),
                // A thread that stopped is reported once what it finished has been taken.
                Err(TryRecvError::Disconnected) if done.is_empty() => return Err(self.failure()),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(done),
            }
        }
    }

    /// Waits for the thread to finish the frame it has.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread, if it stops instead.
    pub(crate) fn wait(&mut self) -> Result<Presented> {
        match self.presented.recv() {
            Ok(presented) => Ok(presented),
            Err(_) => Err(self.failure()),
        }
    }

    /// Waits for the thread to finish the frame it has, but no later than `deadline`: `None` if
    /// it hasn't by then (a scripted pen's next sample is due, say).
    ///
    /// # Errors
    ///
    /// The error that stopped the thread, if it stops instead.
    pub(crate) fn wait_until(&mut self, deadline: Instant) -> Result<Option<Presented>> {
        let timeout = deadline.saturating_duration_since(Instant::now());
        match self.presented.recv_timeout(timeout) {
            Ok(presented) => Ok(Some(presented)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(self.failure()),
        }
    }

    /// Stops the thread once it has finished its frame, and hands the target back with the
    /// GPU times of the last frames. The thread waits for the GPU to finish them first.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread early, that it panicked, or that the GPU failed as it
    /// finished.
    pub(crate) fn stop(mut self) -> Result<Stopped> {
        self.frames = None;
        self.join()
    }

    /// Joins the thread, for its result.
    ///
    /// # Errors
    ///
    /// Its error, or that it panicked or was joined already.
    fn join(&mut self) -> Result<Stopped> {
        let handle = self
            .handle
            .take()
            .ok_or_else(|| anyhow!("the render thread has already stopped"))?;
        handle
            .join()
            .map_err(|_panic| anyhow!("the render thread panicked"))?
    }

    /// Why the thread stopped, once it has: joins it for its error.
    fn failure(&mut self) -> anyhow::Error {
        self.frames = None;
        match self.join() {
            Ok(_) => anyhow!("the render thread stopped"),
            Err(error) => error.context("the render thread stopped"),
        }
    }
}

impl Drop for RenderThread {
    /// Stops the thread and waits for it, so the target (and the window it holds) is released
    /// before whatever owns the window goes.
    fn drop(&mut self) {
        self.frames = None;
        if let Some(handle) = self.handle.take() {
            // Its result was not asked for; the owner has an error of its own to report.
            drop(handle.join());
        }
    }
}

/// The render thread's state: the GPU, where frames go, and the passes.
struct Drawing {
    /// The GPU, shared with the owner thread.
    gpu: Arc<Gpu>,
    /// Where frames are drawn.
    target: Target,
    /// The frame's passes.
    renderer: Renderer,
    /// The newest frame the GPU has finished: each frame drawn asks the queue to set it.
    finished: Arc<AtomicU64>,
    /// Test hook: called with each frame before it is drawn.
    #[cfg(test)]
    before_draw: Box<dyn Fn(&FrameRequest) + Send>,
}

impl std::fmt::Debug for Drawing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drawing")
            .field("target", &self.target)
            .field("renderer", &self.renderer)
            .finish_non_exhaustive()
    }
}

impl Drawing {
    /// Draws every frame that arrives on `frames` until the owner hangs up, reporting each on
    /// `presented`, waking the owner after each and calling `pre_present` before each present.
    ///
    /// # Errors
    ///
    /// The first frame that fails to draw.
    fn run(
        &mut self,
        frames: &Receiver<FrameRequest>,
        presented: &mpsc::Sender<Presented>,
        wake: &dyn Fn(),
        pre_present: &dyn Fn(),
    ) -> Result<()> {
        while let Ok(frame) = frames.recv() {
            #[cfg(test)]
            (self.before_draw)(&frame);
            let mut report = draw_frame(
                &mut self.renderer,
                &self.gpu,
                &mut self.target,
                frame,
                pre_present,
            )?;
            // Once the GPU has done what this frame submitted (and all before it; a frame that
            // submitted nothing is done with the one before), it is finished.
            let finished = Arc::clone(&self.finished);
            let number = report.number;
            self.gpu.queue().on_submitted_work_done(move || {
                finished.fetch_max(number, Ordering::AcqRel);
            });
            report.gpu_finished = self.finished.load(Ordering::Acquire);
            if presented.send(report).is_err() {
                // The owner has stopped listening; it will hang up next.
                break;
            }
            wake();
        }
        Ok(())
    }

    /// The GPU times of the last frames drawn, which no later frame's encoding resolved, once the
    /// GPU has finished them: waits for the GPU (the thread is stopping; nothing waits on it).
    ///
    /// # Errors
    ///
    /// If waiting for the GPU or resolving the times fails.
    fn last_timings(&mut self) -> Result<Vec<GpuTimes>> {
        let device = self.gpu.device();
        device
            .poll(PollType::wait_indefinitely())
            .context("waiting for the last frames")?;
        self.renderer
            .flush_timings(&self.gpu)
            .context("resolving the last frames' times")?;
        device
            .poll(PollType::wait_indefinitely())
            .context("reading the last frames' times back")?;
        Ok(self.renderer.take_timings())
    }
}

/// Draws `frame` into `target` with `renderer`, calling `pre_present` before presenting.
///
/// Lays out its overlay, takes the scene's passes if it brings them, and draws its uploads, its
/// scene and the overlay ([`Renderer::draw`]).
///
/// # Errors
///
/// If the frame can't be drawn.
pub(crate) fn draw_frame(
    renderer: &mut Renderer,
    gpu: &Gpu,
    target: &mut Target,
    mut frame: FrameRequest,
    pre_present: impl FnOnce(),
) -> Result<Presented> {
    let image = TextLayout::new(&frame.overlay).image();
    if let Some(passes) = frame.passes.take() {
        renderer.set_passes(passes);
    }
    let scene_inputs = frame.scene.as_ref().map(SceneRequest::inputs);
    let scene = match (&scene_inputs, frame.reshade) {
        (Some(inputs), _) => Scene::Draw(inputs),
        (None, true) => Scene::Reshade,
        (None, false) => Scene::None,
    };
    let inputs = FrameInputs {
        size: frame.size,
        clear: CLEAR,
        overlay: image.as_ref(),
        overlay_scale: frame.overlay_scale,
        ui: frame.ui.as_ref(),
        scene,
        look: frame.look,
        frame: frame.number,
    };
    let uploads = frame.uploads.take();
    let dropped_before = renderer.dropped_readbacks();
    let drawn = renderer
        .draw(gpu, target, &inputs, uploads, pre_present)
        .with_context(|| format!("drawing frame {}", frame.number))?;
    let submitted = Instant::now();
    Ok(Presented {
        number: frame.number,
        outcome: drawn.outcome,
        times: drawn.times,
        scene: drawn.scene,
        shown: drawn.shown,
        aa: drawn.aa,
        gpu: renderer.take_timings(),
        dropped: renderer.dropped_readbacks().since(dropped_before),
        gpu_finished: 0,
        submitted,
    })
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::time::Duration;

    use render::gpu::AdapterRequest;
    use render::overlay::MARGIN;
    use render::surface::Offscreen;

    use super::*;

    /// A software GPU and an offscreen target of `size` on it.
    fn offscreen(size: Size) -> (Arc<Gpu>, Target) {
        let gpu = Arc::new(Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe"));
        let target = Target::Offscreen(Offscreen::new(&gpu, size).expect("target"));
        (gpu, target)
    }

    /// Frame 1 of `size`, its overlay `lines`.
    fn request(size: Size, lines: &[&str]) -> FrameRequest {
        FrameRequest {
            number: 1,
            size,
            overlay_scale: 1,
            overlay: lines.iter().map(|&line| line.to_owned()).collect(),
            uploads: None,
            scene: None,
            reshade: false,
            look: Look::default(),
            passes: None,
            ui: None,
        }
    }

    #[test]
    fn the_thread_lays_out_the_overlay_text_and_draws_it() {
        let size = Size::new(160, 64).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let mut thread = RenderThread::spawn(Arc::clone(&gpu), target, Hooks::none(), false)
            .expect("render thread");
        assert_eq!(thread.presentation(), Presentation::Offscreen);
        thread.send(request(size, &["#"])).expect("sent");
        let presented = thread.wait().expect("drawn");
        assert_eq!(presented.number, 1);
        assert_eq!(presented.outcome, Outcome::Presented);
        let Target::Offscreen(offscreen) = thread.stop().expect("stopped cleanly").target else {
            unreachable!("made offscreen");
        };
        let pixels = offscreen.read_pixels(&gpu).expect("read back");
        // '#''s first row has ink at its second column, 3 texels of padding in.
        let corner = MARGIN as usize + 3;
        let ink = pixels[corner * 160 + corner + 1];
        assert!(ink[0] > 200, "ink is bright: {ink:?}");
        let blank = pixels[corner * 160 + corner];
        assert!(blank[0] < 100, "its first column is backing: {blank:?}");
    }

    #[test]
    fn a_stopped_thread_hands_its_target_back_and_reports_nothing_more() {
        let size = Size::new(8, 8).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let mut thread =
            RenderThread::spawn(gpu, target, Hooks::none(), false).expect("render thread");
        assert_eq!(thread.finished().expect("running"), Vec::new());
        let returned = thread.stop().expect("stopped cleanly");
        assert_eq!(returned.target.size(), size);
    }

    #[test]
    fn a_frame_with_no_scene_is_heard_finished_by_polling() {
        let size = Size::new(8, 8).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let mut thread =
            RenderThread::spawn(gpu, target, Hooks::none(), true).expect("render thread");
        assert_eq!(thread.gpu_finished().expect("polled"), 0, "nothing drawn");
        thread.send(request(size, &[])).expect("sent");
        let presented = thread.wait().expect("drawn");
        assert!(!presented.scene, "no scene");
        // The owner's poll: never a wait on the GPU, so a few, a millisecond apart.
        let heard = (0..10_000).find_map(|_| {
            let finished = thread.gpu_finished().expect("polled");
            if finished == presented.number {
                return Some(finished);
            }
            thread::sleep(Duration::from_millis(1));
            None
        });
        assert_eq!(heard, Some(1), "frame 1 heard finished within ten seconds");
        thread.stop().expect("stopped cleanly");
    }

    #[test]
    fn a_panic_on_the_render_thread_wakes_the_owner_and_is_reported() {
        let size = Size::new(8, 8).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let (woken, wakes) = mpsc::channel();
        let mut hooks = Hooks::new(move || woken.send(()).unwrap_or(()), || {});
        hooks.before_draw = Box::new(|_| panic!("a bug in a pass"));
        let mut thread = RenderThread::spawn(gpu, target, hooks, false).expect("render thread");
        thread.send(request(size, &[])).expect("sent");
        wakes
            .recv_timeout(Duration::from_secs(30))
            .expect("the owner is woken as the thread unwinds");
        let error = thread.finished().expect_err("the thread is gone");
        assert!(
            format!("{error:#}").contains("the render thread panicked"),
            "{error:#}"
        );
    }
}
