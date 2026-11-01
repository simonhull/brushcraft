//! One pen stream for the brush and navigation: our Wayland backend where it runs, winit's
//! tablet events as the fallback.
//!
//! **Where samples come from.** On a Wayland session the stream is `input`'s own backend
//! ([`input::wayland::PenBackend`]), read on a thread of its own and stamped through `input`'s
//! [`Dejitter`](input::Dejitter) there; the research note `pen-input-wayland.md` chose it over
//! winit (both barrel buttons, no strokes broken by a barrel release, a lower p99 under load).
//! Where it can't start (X11, a compositor without the tablet protocol, another platform), the
//! stream falls back to winit's tablet events, stamped through a [`Dejitter`](input::Dejitter)
//! of its own on arrival ([`fallback`]), and the overlay says so ([`PenStatus`]). Either way the
//! rest of the app sees one kind of sample, [`PenSample`], in physical pixels.
//!
//! **Into the frame loop.** The window drains the stream ([`PenStream::drain`]) on the owner
//! thread: once after each batch of events, so a sample that may move the camera wants a frame,
//! and again just before each frame, so the frame takes what arrived meanwhile. The owner queues
//! the samples ([`crate::frame::Owner::pen`]), turns them into navigation's input at once
//! ([`navigation`]: the tip drags, the barrel buttons are modifiers, hovering moves the pick), and
//! hands them, in order, to the frame's [`PenFrame`] in the input phase, where the brush step
//! reads them with the hover ([`Hover`]: where the pen is in range, and how far above).
//!
//! **Kept free of sculpting:** a sample is the pen's state and nothing more (no rays, no
//! surfaces, no brush settings), so a painting app could take the same stream.
//!
//! **The backend's one hazard.** `PenBackend::spawn` is `unsafe`: no `wl_surface` on the
//! connection may be destroyed while the backend runs (its `# Safety` section). The stream rules
//! that out by construction, as the `SAFETY` proof in `wayland.rs` sets out: the stream creates the
//! window itself ([`AppPen::create_window`]), undecorated, and spawns the backend once winit has
//! returned it (so after its first configure); it holds the window, dropping (and so stopping)
//! the backend first, and the window's exit path stops it explicitly too ([`PenStream::stop`]).

mod fallback;
mod navigation;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
#[cfg(target_os = "linux")]
mod wayland;

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use input::{Buttons, Phase, Tilt, Tool, Unit};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub(crate) use self::fallback::{ToolData, WinitEvent, WinitPen};
pub(crate) use self::navigation::PenNavigation;
use crate::camera::Pixel;

/// One pen sample, as the app takes it: the pen's state at one moment, in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PenSample {
    /// Where the sample sits in the tool's lifecycle over the window.
    pub(crate) phase: Phase,
    /// Which tool produced it.
    pub(crate) tool: Tool,
    /// Where the pen is, in physical pixels from the window's top-left corner, as the camera
    /// and picking count them.
    pub(crate) at: Pixel,
    /// Tip pressure; zero when not touching.
    pub(crate) pressure: Unit,
    /// How far above the tablet the pen hovers, if the tablet reports it.
    pub(crate) distance: Option<Unit>,
    /// The pen's tilt, if the tablet reports it.
    pub(crate) tilt: Option<Tilt>,
    /// Barrel rotation in degrees, if the pen reports it.
    pub(crate) roll: Option<f64>,
    /// Barrel buttons held.
    pub(crate) buttons: Buttons,
    /// When the pen took the sample, as best we can tell: on the pen's de-jittered clock for a
    /// stroke's phases (`Down` through `Up`), the receipt time otherwise.
    pub(crate) timestamp: Instant,
    /// When it reached us: decoded on the backend's thread, or delivered by winit.
    pub(crate) received: Instant,
}

impl PenSample {
    /// The pen's hover, if this sample has it in range and not touching (`Enter`, `Hover`, or
    /// the `Up` that lifts it); `None` while touching or once it has left.
    pub(crate) const fn hover(&self) -> Option<Hover> {
        match self.phase {
            Phase::Enter | Phase::Hover | Phase::Up => Some(Hover {
                at: self.at,
                distance: self.distance,
                timestamp: self.timestamp,
            }),
            Phase::Down | Phase::Drag | Phase::Leave => None,
        }
    }
}

/// The pen in range of the tablet and not touching it: where it points, and how far above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hover {
    /// Where it points, in physical pixels.
    pub(crate) at: Pixel,
    /// How far above the tablet, if the tablet reports it.
    pub(crate) distance: Option<Unit>,
    /// When it was there.
    pub(crate) timestamp: Instant,
}

/// What a frame's input phase took from the pen stream: the samples queued since the last
/// frame, in order, and the pen's hover as they leave it.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PenFrame {
    /// The frame's samples, oldest first.
    samples: Vec<PenSample>,
    /// The pen's hover after the last sample of this frame or an earlier one; `None` while it
    /// touches or is out of range.
    hover: Option<Hover>,
}

impl PenFrame {
    /// The samples this frame took, oldest first: what the brush step turns into a stroke.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the brush step reads it from M1a4's edit loop on")
    )]
    pub(crate) fn samples(&self) -> &[PenSample] {
        &self.samples
    }

    /// Where the pen hovers, if it does: kept across frames while it hovers still, so a brush
    /// can prepare the surface under it before a stroke starts.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the brush step reads it from M1a4's edit loop on")
    )]
    pub(crate) const fn hover(&self) -> Option<Hover> {
        self.hover
    }

    /// Starts the next frame with the samples of `queued`, in order, emptying it: the last
    /// frame's samples go, the hover carries over until a sample changes it.
    pub(crate) fn take(&mut self, queued: &mut Vec<PenSample>) {
        self.samples.clear();
        if let Some(last) = queued.last() {
            self.hover = last.hover();
        }
        self.samples.append(queued);
    }
}

/// Where the pen stream comes from, for the overlay.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum PenSource {
    /// No window, so no pen (an offscreen run).
    #[default]
    None,
    /// Our Wayland backend.
    Ours,
    /// winit's tablet events, because our backend couldn't run.
    Fallback {
        /// Why our backend isn't running.
        why: Arc<str>,
    },
}

/// The pen stream's state, for the overlay.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct PenStatus {
    /// Where the samples come from.
    pub(crate) source: PenSource,
    /// Samples our backend's queue dropped (motion only) because the app fell behind.
    pub(crate) dropped: u64,
    /// Phase changes our backend's queue lost because even its reserve was full.
    pub(crate) lost: u64,
}

impl fmt::Display for PenStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.source {
            PenSource::None => write!(f, "none (no window)"),
            PenSource::Ours => write!(
                f,
                "our Wayland backend  {} dropped  {} lost",
                self.dropped, self.lost
            ),
            PenSource::Fallback { why } => write!(f, "winit's tablet events (fallback: {why})"),
        }
    }
}

/// A pen backend the stream can run: our Wayland one, or a test's stand-in.
pub(crate) trait Backend {
    /// Appends the samples queued over the window since the last drain to `out`, in order, at
    /// `scale` physical pixels per logical one.
    fn drain(&mut self, scale: f64, out: &mut Vec<PenSample>);

    /// Whether the backend's thread is still reading.
    fn running(&self) -> bool;

    /// Samples dropped and phase changes lost by the backend's queue so far.
    fn losses(&self) -> (u64, u64);

    /// Stops the backend, joining its thread, and appends what it queued since the last drain
    /// to `out`.
    ///
    /// # Errors
    ///
    /// The error that stopped the backend's thread early, if one did.
    fn stop(self, scale: f64, out: &mut Vec<PenSample>) -> Result<()>;
}

/// Where the stream's samples come from now.
#[derive(Debug)]
enum Source<B> {
    /// Our backend, running.
    Ours(B),
    /// winit's tablet events.
    Fallback {
        /// winit's tool, followed and de-jittered; boxed, as it is large beside a backend.
        winit: Box<WinitPen>,
        /// Why our backend isn't running.
        why: Arc<str>,
    },
}

impl<B> Source<B> {
    /// The fallback, because of `why`.
    fn fallback(why: &str) -> Self {
        Self::Fallback {
            winit: Box::default(),
            why: Arc::from(why),
        }
    }
}

/// The app's one pen stream: our backend where it runs, winit's tablet events otherwise.
///
/// It holds the window (`W`, an `Arc` of it in the app) so the window outlives the backend:
/// `source` is declared before `window`, so dropping the stream drops the backend (joining its
/// thread) before this hold on the window goes. The order is load-bearing.
#[derive(Debug)]
pub(crate) struct PenStream<B, W> {
    /// Where samples come from. Declared before `window`: see the type's notes.
    source: Source<B>,
    /// Physical pixels per logical pixel: our backend reports logical ones.
    scale: f64,
    /// Samples from winit's events not yet drained.
    pending: Vec<PenSample>,
    /// Samples dropped and lost by a backend that has since stopped.
    losses: (u64, u64),
    /// The window. Declared after `source`: see the type's notes.
    window: W,
}

impl<B: Backend, W> PenStream<B, W> {
    /// The stream for `window`, at `scale` physical pixels per logical one: our backend if
    /// `spawn` starts it, winit's tablet events with the reason if it fails.
    pub(crate) fn start(window: W, scale: f64, spawn: impl FnOnce(&W) -> Result<B>) -> Self {
        let source = match spawn(&window) {
            Ok(backend) => Source::Ours(backend),
            Err(e) => Source::fallback(&format!("{e:#}")),
        };
        Self {
            source,
            scale,
            pending: Vec::new(),
            losses: (0, 0),
            window,
        }
    }

    /// The window the stream holds.
    pub(crate) const fn window(&self) -> &W {
        &self.window
    }

    /// The window's scale factor changed to `scale`.
    pub(crate) const fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
    }

    /// Where the samples come from, and what our backend's queue lost.
    pub(crate) fn status(&self) -> PenStatus {
        let (source, (dropped, lost)) = match &self.source {
            Source::Ours(backend) => {
                let (dropped, lost) = backend.losses();
                (PenSource::Ours, (dropped, lost))
            }
            Source::Fallback { why, .. } => (
                PenSource::Fallback {
                    why: Arc::clone(why),
                },
                self.losses,
            ),
        };
        PenStatus {
            source,
            dropped,
            lost,
        }
    }

    /// Appends every sample that arrived since the last drain to `out`, in order.
    ///
    /// A backend found stopped is joined and replaced by the fallback (its error is the
    /// fallback's reason, shown in the overlay), keeping what it queued before it stopped.
    pub(crate) fn drain(&mut self, out: &mut Vec<PenSample>) {
        match &mut self.source {
            Source::Ours(backend) if backend.running() => backend.drain(self.scale, out),
            Source::Ours(_) => self.fall_back(out),
            Source::Fallback { .. } => out.append(&mut self.pending),
        }
    }

    /// Joins a backend that stopped on its own, appending what it queued to `out`, and falls
    /// back to winit's events, with the backend's error as the reason.
    fn fall_back(&mut self, out: &mut Vec<PenSample>) {
        let placeholder = Source::fallback("");
        if let Source::Ours(backend) = std::mem::replace(&mut self.source, placeholder) {
            self.losses = backend.losses();
            let why = match backend.stop(self.scale, out) {
                Ok(()) => "our backend stopped".to_owned(),
                Err(e) => format!("our backend stopped: {e:#}"),
            };
            self.source = Source::fallback(&why);
        }
    }

    /// Takes one of winit's tablet events, `event` at `at` with the tool's `data` where it has
    /// it, received at `received` in event-loop wakeup `wakeup`. Followed only on the fallback:
    /// while our backend runs, its samples are the pen's (and the compositor sends winit none).
    pub(crate) fn winit(
        &mut self,
        event: WinitEvent,
        at: Option<Pixel>,
        data: Option<ToolData>,
        (received, wakeup): (Instant, u64),
    ) {
        if let Source::Fallback { winit, .. } = &mut self.source {
            self.pending
                .push(winit.apply(event, at, data, (received, wakeup)));
        }
    }

    /// Stops our backend, if it runs, before handing the window back: the exit path's explicit
    /// stop, so the backend never outlives the window whatever the caller's fields do. What
    /// arrived since the last drain is appended to `out`.
    ///
    /// # Errors
    ///
    /// The error that stopped the backend's thread early, if one did; the window comes back
    /// either way.
    pub(crate) fn stop(self, out: &mut Vec<PenSample>) -> (W, Result<()>) {
        let Self {
            source,
            scale,
            mut pending,
            window,
            ..
        } = self;
        let stopped = match source {
            Source::Ours(backend) => backend.stop(scale, out),
            Source::Fallback { .. } => {
                out.append(&mut pending);
                Ok(())
            }
        };
        (window, stopped)
    }
}

#[cfg(target_os = "linux")]
pub(crate) use self::wayland::Ours;

/// No pen backend of ours on this platform: winit's events are the fallback.
#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub(crate) enum Ours {}

#[cfg(not(target_os = "linux"))]
impl Backend for Ours {
    fn drain(&mut self, _: f64, _: &mut Vec<PenSample>) {
        match *self {}
    }

    fn running(&self) -> bool {
        match *self {}
    }

    fn losses(&self) -> (u64, u64) {
        match *self {}
    }

    fn stop(self, _: f64, _: &mut Vec<PenSample>) -> Result<()> {
        match self {}
    }
}

/// The window's pen stream in the app.
pub(crate) type AppPen = PenStream<Ours, Arc<dyn Window>>;

impl AppPen {
    /// Creates the app's window from `attributes`, with decorations off whatever they say, and
    /// its pen stream: our backend on Wayland, winit's tablet events elsewhere or if it fails to
    /// start. [`window`](Self::window) is the window.
    ///
    /// The backend spawns once winit has returned the window, which it does only after the
    /// window's first configure.
    ///
    /// # Errors
    ///
    /// If the window can't be created; our backend failing to start is not an error.
    pub(crate) fn create_window(
        event_loop: &dyn ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Result<Self> {
        let window: Arc<dyn Window> = Arc::from(
            event_loop
                .create_window(attributes.with_decorations(false))
                .context("creating the window")?,
        );
        let scale = window.scale_factor();
        Ok(Self::start(window, scale, |window| {
            spawn_ours(event_loop, window.as_ref())
        }))
    }
}

/// Starts our backend for `window`: only for [`AppPen::create_window`], which made `window`
/// undecorated and holds it for as long as the backend runs.
///
/// # Errors
///
/// If our backend can't start: not Wayland, no tablet support, another platform.
fn spawn_ours(event_loop: &dyn ActiveEventLoop, window: &dyn Window) -> Result<Ours> {
    #[cfg(target_os = "linux")]
    {
        Ours::spawn(event_loop, window)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (event_loop, window);
        Err(anyhow::anyhow!(
            "there is no pen backend of ours on this platform"
        ))
    }
}
