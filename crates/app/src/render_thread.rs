//! The render thread: encodes and submits each frame the owner thread hands it, and presents
//! it (the M1 spec's "Threads": it only encodes and submits).
//!
//! It owns the [`Target`] and the overlay's GPU side. Frames arrive one at a time over a
//! channel ([`FrameRequest`]); each is drawn (for now, a clear colour and the overlay) and
//! presented, and what happened goes back ([`Presented`]), with a wake-up for the owner's event
//! loop. Dropping the sender ends the thread, which hands the target back.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use render::gpu::Gpu;
use render::surface::{Acquired, Presentation, Size, Target};
use wgpu::{
    Color, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp,
};

use crate::frame::Lap;
use crate::frame::schedule::Shown;
use crate::overlay::{OverlayPass, TextImage};

/// The colour frames are cleared to, linear: a dark grey with a touch of blue.
pub(crate) const CLEAR: Color = Color {
    r: 0.018,
    g: 0.018,
    b: 0.022,
    a: 1.0,
};

/// A frame for the render thread to draw: everything it needs, owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrameRequest {
    /// The frame's number, from 1.
    pub(crate) number: u64,
    /// The size to draw at: the window's now.
    pub(crate) size: Size,
    /// Frame pixels per overlay texel: the window's scale factor, rounded, at least 1.
    pub(crate) overlay_scale: u32,
    /// The overlay's lines.
    pub(crate) overlay: Vec<String>,
}

/// How long the render thread's phases took for one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RenderPhases {
    /// Waiting for the target's next image (FIFO waits here for a free one).
    pub(crate) acquire: Duration,
    /// Uploading the overlay, encoding the frame's passes and submitting them.
    pub(crate) encode: Duration,
    /// Presenting the image.
    pub(crate) present: Duration,
}

/// What the render thread did with a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Presented {
    /// The frame's number.
    pub(crate) number: u64,
    /// Whether it reached the screen.
    pub(crate) shown: Shown,
    /// How long its phases took.
    pub(crate) phases: RenderPhases,
}

/// What the render thread calls on the owner's behalf.
pub(crate) struct Hooks {
    /// Called after each frame's [`Presented`] is sent, and as the thread exits: wakes the
    /// owner's event loop. Never blocks.
    pub(crate) wake: Box<dyn Fn() + Send>,
    /// Called just before a frame is presented: winit's `pre_present_notify`, which lets the
    /// compositor pace the window.
    pub(crate) pre_present: Box<dyn Fn() + Send>,
}

impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks").finish_non_exhaustive()
    }
}

impl Hooks {
    /// Hooks that do nothing: for a target with no event loop to wake.
    pub(crate) fn none() -> Self {
        Self {
            wake: Box::new(|| {}),
            pre_present: Box::new(|| {}),
        }
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
    /// If the overlay's pipeline can't be built, or the thread can't be started.
    pub(crate) fn spawn(gpu: Arc<Gpu>, target: Target, hooks: Hooks) -> Result<Self> {
        let presentation = target.presentation();
        let overlay =
            OverlayPass::new(&gpu, target.format()).context("building the overlay's pipeline")?;
        // One frame at a time: the scheduler never sends another before this one is presented.
        let (frames, incoming) = mpsc::sync_channel(1);
        let (outgoing, presented) = mpsc::channel();
        let mut renderer = Renderer {
            gpu,
            target,
            overlay,
        };
        let handle = thread::Builder::new()
            .name("render".into())
            .spawn(move || {
                let result = renderer.run(&incoming, &outgoing, &hooks);
                (hooks.wake)();
                result.map(|()| renderer.target)
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

/// The render thread's state: the GPU, where frames go, and the overlay.
#[derive(Debug)]
struct Renderer {
    /// The GPU, shared with the owner thread.
    gpu: Arc<Gpu>,
    /// Where frames are drawn.
    target: Target,
    /// The overlay's GPU side.
    overlay: OverlayPass,
}

impl Renderer {
    /// Draws every frame that arrives on `frames` until the owner hangs up, reporting each on
    /// `presented` and calling the hooks.
    ///
    /// # Errors
    ///
    /// The first frame that fails to draw.
    fn run(
        &mut self,
        frames: &Receiver<FrameRequest>,
        presented: &mpsc::Sender<Presented>,
        hooks: &Hooks,
    ) -> Result<()> {
        while let Ok(frame) = frames.recv() {
            let (shown, phases) = self
                .draw(&frame, hooks)
                .with_context(|| format!("drawing frame {}", frame.number))?;
            let report = Presented {
                number: frame.number,
                shown,
                phases,
            };
            if presented.send(report).is_err() {
                // The owner has stopped listening; it will hang up next.
                break;
            }
            (hooks.wake)();
        }
        Ok(())
    }

    /// Draws `frame`: the clear colour and the overlay, then presents it.
    ///
    /// # Errors
    ///
    /// If the target can't be resized or acquired, or the GPU reports an error.
    fn draw(&mut self, frame: &FrameRequest, hooks: &Hooks) -> Result<(Shown, RenderPhases)> {
        let mut phases = RenderPhases::default();
        self.target.resize(&self.gpu, frame.size)?;
        let mut lap = Lap::new();
        let acquired = self.target.acquire(&self.gpu)?;
        phases.acquire = lap.next();
        let image = match acquired {
            Acquired::Frame(image) => image,
            Acquired::Retry => return Ok((Shown::Retry, phases)),
            Acquired::Hidden => return Ok((Shown::Hidden, phases)),
        };
        let text = TextImage::new(&frame.overlay);
        self.overlay
            .prepare(&self.gpu, &text, frame.size, frame.overlay_scale)?;
        self.gpu.scoped("encoding a frame", || {
            let mut encoder = self
                .gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("clear and overlay"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: image.view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(CLEAR),
                            store: StoreOp::Store,
                        },
                    })],
                    ..RenderPassDescriptor::default()
                });
                self.overlay.draw(&mut pass);
            }
            self.gpu.queue().submit([encoder.finish()]);
        })?;
        phases.encode = lap.next();
        (hooks.pre_present)();
        image.present(&self.gpu);
        self.gpu.check()?;
        phases.present = lap.next();
        Ok((Shown::Presented, phases))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use render::gpu::AdapterRequest;
    use render::surface::Offscreen;

    use super::*;

    /// `value` (linear, 0 to 1) as an 8-bit sRGB code, as the target stores it.
    fn srgb(value: f64) -> u8 {
        let encoded = if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded from 0..=255"
        )]
        let code = (encoded * 255.0).round() as u8;
        code
    }

    #[test]
    fn a_frame_is_the_clear_colour_with_the_overlay_drawn_over_its_corner() {
        let gpu = Arc::new(Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe"));
        let size = Size::new(160, 64).expect("non-zero");
        let target = Target::Offscreen(Offscreen::new(&gpu, size).expect("target"));
        let mut thread =
            RenderThread::spawn(Arc::clone(&gpu), target, Hooks::none()).expect("render thread");
        assert_eq!(thread.presentation(), Presentation::Offscreen);
        thread
            .send(FrameRequest {
                number: 1,
                size,
                overlay_scale: 1,
                overlay: vec!["#".to_owned()],
            })
            .expect("sent");
        let presented = thread.wait().expect("drawn");
        assert_eq!(presented.number, 1);
        assert_eq!(presented.shown, Shown::Presented);
        let Target::Offscreen(offscreen) = thread.stop().expect("stopped cleanly") else {
            unreachable!("made offscreen");
        };
        let pixels = offscreen.read_pixels(&gpu).expect("read back");
        let at = |x: usize, y: usize| pixels[y * 160 + x];
        let clear = [srgb(CLEAR.r), srgb(CLEAR.g), srgb(CLEAR.b), 255];
        let tolerance = |a: [u8; 4], b: [u8; 4]| a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 1);
        assert!(
            tolerance(at(150, 60), clear),
            "outside the overlay: {:?}",
            at(150, 60)
        );
        // The overlay starts 8 pixels in; its backing is darker than the clear colour, and the
        // '#' glyph's first row has ink at its second column, 3 texels of padding further in.
        let backing = at(9, 9);
        assert!(
            backing[0] < clear[0],
            "backing {backing:?} darkens {clear:?}"
        );
        let ink = at(8 + 3 + 1, 8 + 3);
        assert!(ink[0] > 200, "ink is bright: {ink:?}");
    }

    #[test]
    fn a_stopped_thread_hands_its_target_back_and_reports_nothing_more() {
        let gpu = Arc::new(Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe"));
        let size = Size::new(8, 8).expect("non-zero");
        let target = Target::Offscreen(Offscreen::new(&gpu, size).expect("target"));
        let mut thread = RenderThread::spawn(gpu, target, Hooks::none()).expect("render thread");
        assert_eq!(thread.finished().expect("running"), Vec::new());
        let returned = thread.stop().expect("stopped cleanly");
        assert_eq!(returned.size(), size);
    }
}
