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

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use geom::Level;
use render::camera::View;
use render::cull::{BoundsUpdate, SlotBuffers};
use render::gpu::Gpu;
use render::raster::{FramePasses, SceneInputs};
use render::surface::{Presentation, Size, Target};
use render::upload::FrameUploads;
use render::{DrawTimes, FrameInputs, Look, Outcome, Renderer, Scene};
use wgpu::{Buffer, Color};

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
    /// The scene to draw; `None` with no GPU residency (or no view to draw it from).
    pub(crate) scene: Option<SceneRequest>,
    /// The scene's passes, with the first frame only, whether or not it has a scene: the render
    /// thread keeps them.
    pub(crate) passes: Option<FramePasses>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Presented {
    /// The frame's number.
    pub(crate) number: u64,
    /// Whether it reached the screen.
    pub(crate) outcome: Outcome,
    /// How long its phases took.
    pub(crate) times: DrawTimes,
    /// Whether its scene was drawn (`render::Drawn::scene`).
    pub(crate) scene: bool,
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

/// The render thread, from the owner's side.
#[derive(Debug)]
pub(crate) struct RenderThread {
    /// Where frames go; `None` once stopping.
    frames: Option<SyncSender<FrameRequest>>,
    /// What happened to them.
    presented: Receiver<Presented>,
    /// The thread, which hands the target back when it ends; `None` once joined.
    handle: Option<JoinHandle<Result<Target>>>,
    /// How the target presents.
    presentation: Presentation,
}

impl RenderThread {
    /// Starts the render thread, drawing into `target` on `gpu`.
    ///
    /// # Errors
    ///
    /// If the renderer's passes can't be built, or the thread can't be started.
    pub(crate) fn spawn(gpu: Arc<Gpu>, target: Target, hooks: Hooks) -> Result<Self> {
        let presentation = target.presentation();
        let renderer =
            Renderer::new(&gpu, target.format()).context("building the renderer's passes")?;
        // One frame at a time: the scheduler never sends another before this one is presented.
        let (frames, incoming) = mpsc::sync_channel(1);
        let (outgoing, presented) = mpsc::channel();
        let Hooks {
            wake,
            pre_present,
            #[cfg(test)]
            before_draw,
        } = hooks;
        let mut thread = Drawing {
            gpu,
            target,
            renderer,
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
                thread
                    .run(&incoming, &outgoing, &*wake.0, &*pre_present)
                    .map(|()| thread.target)
            })
            .context("starting the render thread")?;
        Ok(Self {
            frames: Some(frames),
            presented,
            handle: Some(handle),
            presentation,
        })
    }

    /// How the target presents.
    pub(crate) const fn presentation(&self) -> Presentation {
        self.presentation
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

    /// Stops the thread once it has finished its frame, and hands the target back.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread early, or that it panicked.
    pub(crate) fn stop(mut self) -> Result<Target> {
        self.frames = None;
        self.join()
    }

    /// Joins the thread, for its result.
    ///
    /// # Errors
    ///
    /// Its error, or that it panicked or was joined already.
    fn join(&mut self) -> Result<Target> {
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
            let report = draw_frame(
                &mut self.renderer,
                &self.gpu,
                &mut self.target,
                frame,
                pre_present,
            )?;
            if presented.send(report).is_err() {
                // The owner has stopped listening; it will hang up next.
                break;
            }
            wake();
        }
        Ok(())
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
    let scene = frame.scene.as_ref().map(SceneRequest::inputs);
    let inputs = FrameInputs {
        size: frame.size,
        clear: CLEAR,
        overlay: image.as_ref(),
        overlay_scale: frame.overlay_scale,
        scene: scene.as_ref().map_or(Scene::None, Scene::Draw),
        look: Look::default(),
    };
    let uploads = frame.uploads.take();
    let drawn = renderer
        .draw(gpu, target, &inputs, uploads, pre_present)
        .with_context(|| format!("drawing frame {}", frame.number))?;
    Ok(Presented {
        number: frame.number,
        outcome: drawn.outcome,
        times: drawn.times,
        scene: drawn.scene,
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
            passes: None,
        }
    }

    #[test]
    fn the_thread_lays_out_the_overlay_text_and_draws_it() {
        let size = Size::new(160, 64).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let mut thread =
            RenderThread::spawn(Arc::clone(&gpu), target, Hooks::none()).expect("render thread");
        assert_eq!(thread.presentation(), Presentation::Offscreen);
        thread.send(request(size, &["#"])).expect("sent");
        let presented = thread.wait().expect("drawn");
        assert_eq!(presented.number, 1);
        assert_eq!(presented.outcome, Outcome::Presented);
        let Target::Offscreen(offscreen) = thread.stop().expect("stopped cleanly") else {
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
        let mut thread = RenderThread::spawn(gpu, target, Hooks::none()).expect("render thread");
        assert_eq!(thread.finished().expect("running"), Vec::new());
        let returned = thread.stop().expect("stopped cleanly");
        assert_eq!(returned.size(), size);
    }

    #[test]
    fn a_panic_on_the_render_thread_wakes_the_owner_and_is_reported() {
        let size = Size::new(8, 8).expect("non-zero");
        let (gpu, target) = offscreen(size);
        let (woken, wakes) = mpsc::channel();
        let mut hooks = Hooks::new(move || woken.send(()).unwrap_or(()), || {});
        hooks.before_draw = Box::new(|_| panic!("a bug in a pass"));
        let mut thread = RenderThread::spawn(gpu, target, hooks).expect("render thread");
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
