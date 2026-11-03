//! One pen stream for the brush and navigation: our Wayland backend where it runs, winit's
//! tablet events as the fallback.
//!
//! **Where samples come from.** On a Wayland session the stream is `input`'s own backend
//! ([`input::wayland::PenBackend`]), read on a thread of its own and stamped through `input`'s
//! [`Dejitter`](input::Dejitter) there; the research note `pen-input-wayland.md` chose it over
//! winit (both barrel buttons, no strokes broken by a barrel release, a lower p99 under load).
//! Where it can't start (a compositor without Wayland tablet support, another platform; winit is
//! built for Wayland alone, so there is no X11 session to fall back from), the stream falls back
//! to winit's tablet events, stamped through a [`Dejitter`](input::Dejitter) of its own on
//! arrival ([`fallback`]), and the overlay says so ([`PenStatus`]). Either way the
//! rest of the app sees one kind of sample, [`PenSample`], in physical pixels.
//!
//! **Into the frame loop.** The window drains the stream ([`PenStream::drain`]) on the owner
//! thread: once after each batch of events, so a stroke's samples want a frame, and again just
//! before each frame, so the frame takes what arrived meanwhile. The owner queues the samples
//! ([`crate::frame::Owner::pen`]), coalescing the pen's motion out of a stroke (only the latest
//! counts), and routes them in the frame's input phase ([`PenRouter`]): each stroke, `Down`
//! through `Up`, goes whole to navigation or to the brush ([`Route`], decided at its `Down` by
//! [`route`]), while the samples out of a stroke go to both: navigation reads them for the
//! pointer and the barrel buttons ([`navigation`]: the tip drags, the barrels are modifiers,
//! hovering moves the pick), the brush for the hover ([`Hover`]: where the pen is in range, and
//! how far above). The frame's [`PenFrame`] holds them, in order, with their routes.
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
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use input::{Buttons, Phase, Tilt, Tool, Unit};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub(crate) use self::fallback::{ToolData, WinitEvent, WinitPen};
pub(crate) use self::navigation::PenNavigation;
use crate::camera::Pixel;
use crate::frame::PenFeed;

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

/// Who a stroke belongs to: decided once, at its `Down`, for the whole stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    /// The stroke orbits, pans or zooms the view.
    Navigate,
    /// The stroke is the brush's.
    Brush,
}

/// The route of the stroke `down` starts: navigation whenever a barrel button is held (the
/// barrels are navigation's modifiers), what `brush` decides otherwise.
///
/// `brush` is the one place the brush claims strokes ([`crate::frame::Brush::route`]); with no
/// brush, every stroke navigates.
pub(crate) fn route(down: &PenSample, brush: impl FnOnce(&PenSample) -> Route) -> Route {
    if down.buttons == Buttons::default() {
        brush(down)
    } else {
        Route::Navigate
    }
}

/// Follows the pen stream's strokes, routing each sample: a stroke's samples go where its `Down`
/// sent them, the rest to both navigation and the brush.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PenRouter {
    /// The route of the stroke in progress; `None` between strokes.
    stroke: Option<Route>,
}

impl PenRouter {
    /// The route of the stroke in progress; `None` between strokes.
    pub(crate) const fn stroke(self) -> Option<Route> {
        self.stroke
    }

    /// Where `sample` goes: its stroke's route (at a `Down`, the one `brush` and [`route`]
    /// decide), or `None` for a sample out of a stroke, which goes to both.
    ///
    /// A stroke ends at its `Up`, or at a sample out of a stroke if the `Up` was lost. A `Drag`
    /// or `Up` with no `Down` before it (lost) navigates, where it does nothing without a drag.
    pub(crate) fn route(
        &mut self,
        sample: &PenSample,
        brush: impl FnOnce(&PenSample) -> Route,
    ) -> Option<Route> {
        match sample.phase {
            Phase::Down => {
                let stroke = route(sample, brush);
                self.stroke = Some(stroke);
                Some(stroke)
            }
            Phase::Drag => Some(*self.stroke.get_or_insert(Route::Navigate)),
            Phase::Up => Some(self.stroke.take().unwrap_or(Route::Navigate)),
            Phase::Enter | Phase::Hover | Phase::Leave => {
                self.stroke = None;
                None
            }
        }
    }
}

/// What a frame's input phase took from the pen stream: the samples queued since the last frame,
/// in order, which strokes (or parts of strokes) among them went where, and the pen's hover as
/// they leave it.
///
/// **The contract:** navigation has already had the samples outside the brush's strokes; the
/// brush reads [`for_brush`](Self::for_brush), its strokes' samples and those out of any stroke
/// (the hover, for warming the surface under the pen), never a navigation stroke's. A stroke can
/// span frames: its first frame's range starts at its `Down`, its last ends at its `Up` (or,
/// with the `Up` lost, before the next sample out of a stroke).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PenFrame {
    /// The frame's samples, oldest first.
    samples: Vec<PenSample>,
    /// The runs of `samples` that belong to a stroke, with the stroke's route, in order.
    strokes: Vec<(Range<usize>, Route)>,
    /// The pen's hover after the last sample of this frame or an earlier one; `None` while it
    /// touches or is out of range.
    hover: Option<Hover>,
}

impl PenFrame {
    /// Every sample this frame took, oldest first, whatever its route.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the brush reads for_brush and hover; the tests check the frame whole"
        )
    )]
    pub(crate) fn samples(&self) -> &[PenSample] {
        &self.samples
    }

    /// The runs of [`samples`](Self::samples) in a stroke, each with its stroke's route.
    pub(crate) fn strokes(&self) -> &[(Range<usize>, Route)] {
        &self.strokes
    }

    /// The samples the brush reads, oldest first: its strokes' and those out of any stroke.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the brush reads brush_events, which marks strokes elsewhere too"
        )
    )]
    pub(crate) fn for_brush(&self) -> impl Iterator<Item = &PenSample> {
        let navigated = |index: usize| {
            self.strokes
                .iter()
                .any(|(range, route)| *route == Route::Navigate && range.contains(&index))
        };
        self.samples
            .iter()
            .enumerate()
            .filter(move |&(index, _)| !navigated(index))
            .map(|(_, sample)| sample)
    }

    /// What the brush takes from this frame, in order: its strokes' samples and those out of
    /// any stroke, as [`for_brush`](Self::for_brush) gives them, and where a stroke routed
    /// elsewhere starts, a [`ForBrush::StrokeElsewhere`] in its place (its samples are
    /// navigation's), so the brush can end a stroke whose `Up` was lost.
    pub(crate) fn brush_events(&self) -> impl Iterator<Item = ForBrush<'_>> {
        let elsewhere = |index: usize| {
            self.strokes
                .iter()
                .find(|(range, _)| range.contains(&index))
                .is_some_and(|(_, route)| *route != Route::Brush)
        };
        self.samples
            .iter()
            .enumerate()
            .filter_map(move |(index, sample)| {
                if !elsewhere(index) {
                    Some(ForBrush::Sample(sample))
                } else if sample.phase == Phase::Down {
                    Some(ForBrush::StrokeElsewhere)
                } else {
                    None
                }
            })
    }

    /// Where the pen hovers, if it does: kept across frames while it hovers still, so a brush
    /// can prepare the surface under it before a stroke starts.
    pub(crate) const fn hover(&self) -> Option<Hover> {
        self.hover
    }

    /// Starts the next frame: the last frame's samples go, the hover carries over until a
    /// sample changes it.
    pub(crate) fn clear(&mut self) {
        self.samples.clear();
        self.strokes.clear();
    }

    /// Adds `sample`, routed `route` (`None`: out of a stroke).
    pub(crate) fn push(&mut self, sample: PenSample, route: Option<Route>) {
        let index = self.samples.len();
        self.samples.push(sample);
        self.hover = sample.hover();
        let Some(route) = route else {
            return;
        };
        match self.strokes.last_mut() {
            // The same stroke goes on: a new stroke always starts at a `Down`.
            Some((range, last))
                if range.end == index && *last == route && sample.phase != Phase::Down =>
            {
                range.end = index + 1;
            }
            _ => self.strokes.push((index..index + 1, route)),
        }
    }
}

/// What the brush is given of a frame's pen samples ([`PenFrame::brush_events`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ForBrush<'a> {
    /// A sample of one of its strokes, or out of any stroke.
    Sample(&'a PenSample),
    /// A stroke routed elsewhere (to navigation) starts here: a stroke of the brush's still open
    /// lost its `Up`.
    StrokeElsewhere,
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

/// What a backend's queue lost: motion dropped, and phase changes lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Losses {
    /// Samples dropped (motion only) because the app fell behind.
    pub(crate) dropped: u64,
    /// Phase changes lost because even the queue's reserve was full.
    pub(crate) lost: u64,
}

/// A pen backend the stream can run: our Wayland one, or a test's stand-in.
pub(crate) trait Backend {
    /// Appends the samples queued over the window since the last drain to `out`, in order, at
    /// `scale` physical pixels per logical one.
    fn drain(&mut self, scale: f64, out: &mut Vec<PenSample>);

    /// Whether the backend's thread is still reading.
    fn running(&self) -> bool;

    /// What the backend's queue has lost so far.
    fn losses(&self) -> Losses;

    /// Stops the backend, joining its thread, and appends what it queued since the last drain
    /// to `out`; returns what its queue lost over the whole run, counted once the thread has
    /// been joined.
    ///
    /// # Errors
    ///
    /// The error that stopped the backend's thread early, if one did, beside the losses.
    fn stop(self, scale: f64, out: &mut Vec<PenSample>) -> (Losses, Result<()>);
}

/// winit tablet events in a row, with nothing from our backend meanwhile, that hand the pen to
/// winit.
///
/// After so many the compositor is taken to send the pen to winit rather than to us: about a
/// third of a second of pen motion, far more than a compositor that sends to both would let pile
/// up between drains.
const STARVED: usize = 64;

/// Where the stream's samples come from now.
#[derive(Debug)]
enum Source<B> {
    /// Our backend, running.
    Ours(B),
    /// winit's tablet events.
    Fallback {
        /// Why our backend isn't running.
        why: Arc<str>,
    },
}

/// The app's one pen stream: our backend where it runs, winit's tablet events otherwise.
///
/// It holds the window (`W`, an `Arc` of it in the app) so the window outlives the backend:
/// `source` is declared before `window`, so dropping the stream drops the backend (joining its
/// thread) before this hold on the window goes. The order is load-bearing (our backend also
/// holds the window itself, after its thread).
///
/// **Which source wins.** While our backend runs, winit's tablet events are still followed (so
/// the fallback can take over mid-session with the tool's state right) but held back: the
/// compositors measured send the pen to one tablet binding per client, ours, so winit gets none
/// (`pen-input-wayland.md`). If winit gets [`STARVED`] events with nothing from ours meanwhile,
/// the compositor sends the pen to winit instead: ours is stopped and the held samples are the
/// stream from then on.
///
/// **After our backend stops** on its own, the fallback takes over, but whether the compositor
/// then sends the pen to winit's binding is unverified on Hyprland (one binding per client: it
/// may keep none for winit once ours is gone); the overlay says the stream fell back either way.
/// A stroke our backend left half done (its last sample a `Down` or `Drag`) is ended with a
/// synthetic `Leave` at that sample's position, so neither navigation nor the brush is left
/// waiting for an `Up`.
#[derive(Debug)]
pub(crate) struct PenStream<B, W> {
    /// Where samples come from. Declared before `window`: see the type's notes.
    source: Source<B>,
    /// winit's tool, followed and de-jittered whatever the source; boxed, as it is large.
    winit: Box<WinitPen>,
    /// Physical pixels per logical pixel: our backend reports logical ones.
    scale: f64,
    /// Samples not yet drained: winit's on the fallback, or held back while ours runs; first, a
    /// hand-over's last samples from ours.
    pending: Vec<PenSample>,
    /// What a backend that has since stopped lost.
    losses: Losses,
    /// The last sample drained, to end a stroke the backend leaves half done.
    last: Option<PenSample>,
    /// The window. Declared after `source`: see the type's notes.
    window: W,
}

impl<B: Backend, W> PenStream<B, W> {
    /// The stream for `window`, at `scale` physical pixels per logical one: our backend if
    /// `spawn` starts it, winit's tablet events with the reason if it fails.
    pub(crate) fn start(window: W, scale: f64, spawn: impl FnOnce(&W) -> Result<B>) -> Self {
        let source = match spawn(&window) {
            Ok(backend) => Source::Ours(backend),
            Err(e) => Source::Fallback {
                why: Arc::from(format!("{e:#}")),
            },
        };
        Self {
            source,
            winit: Box::default(),
            scale,
            pending: Vec::new(),
            losses: Losses::default(),
            last: None,
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
        let (source, losses) = match &self.source {
            Source::Ours(backend) => (PenSource::Ours, backend.losses()),
            Source::Fallback { why } => (
                PenSource::Fallback {
                    why: Arc::clone(why),
                },
                self.losses,
            ),
        };
        PenStatus {
            source,
            dropped: losses.dropped,
            lost: losses.lost,
        }
    }

    /// Appends every sample that arrived since the last drain to `out`, in order.
    ///
    /// A backend found stopped is joined and replaced by the fallback (its error is the
    /// fallback's reason, shown in the overlay), keeping what it queued before it stopped.
    pub(crate) fn drain(&mut self, out: &mut Vec<PenSample>) {
        let start = out.len();
        match &mut self.source {
            Source::Ours(backend) if backend.running() => {
                backend.drain(self.scale, out);
                if out.len() > start {
                    // Ours delivers: what winit got meanwhile is a duplicate, not the stream.
                    self.pending.clear();
                }
            }
            Source::Ours(_) => {
                self.fall_back(None);
                out.append(&mut self.pending);
            }
            Source::Fallback { .. } => out.append(&mut self.pending),
        }
        if let Some(last) = out.get(start..).and_then(<[PenSample]>::last) {
            self.last = Some(*last);
        }
    }

    /// Stops our backend (it stopped on its own, or `starved` says why winit has the pen),
    /// joining it, and falls back to winit's events. Its last samples go first into the
    /// pending ones, then a `Leave` if it left a stroke half done, then winit's held samples.
    fn fall_back(&mut self, starved: Option<&str>) {
        let placeholder = Source::Fallback { why: Arc::from("") };
        let Source::Ours(backend) = std::mem::replace(&mut self.source, placeholder) else {
            return;
        };
        let mut handed = Vec::new();
        let (losses, stopped) = backend.stop(self.scale, &mut handed);
        self.losses = losses;
        if let Some(last) = handed.last() {
            self.last = Some(*last);
        }
        if let Some(leave) = self.last.and_then(|last| ended(&last, Instant::now())) {
            handed.push(leave);
            self.last = Some(leave);
        }
        let why = match (starved, stopped) {
            (Some(why), _) => why.to_owned(),
            (None, Ok(())) => "our backend stopped".to_owned(),
            (None, Err(e)) => format!("our backend stopped: {e:#}"),
        };
        self.source = Source::Fallback {
            why: Arc::from(why),
        };
        handed.append(&mut self.pending);
        self.pending = handed;
    }

    /// Takes one of winit's tablet events, `event` at `at` with the tool's `data` where it has
    /// it, received at `received` in event-loop wakeup `wakeup`: the stream on the fallback,
    /// held back while our backend runs (see the type's notes).
    pub(crate) fn winit(
        &mut self,
        event: WinitEvent,
        at: Option<Pixel>,
        data: Option<ToolData>,
        (received, wakeup): (Instant, u64),
    ) {
        self.pending
            .push(self.winit.apply(event, at, data, (received, wakeup)));
        if matches!(self.source, Source::Ours(_)) && self.pending.len() >= STARVED {
            self.fall_back(Some(
                "the compositor sends the pen to winit, not to our backend",
            ));
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
            Source::Ours(backend) => backend.stop(scale, out).1,
            Source::Fallback { .. } => {
                out.append(&mut pending);
                Ok(())
            }
        };
        (window, stopped)
    }
}

impl<B: Backend, W> PenFeed for PenStream<B, W> {
    /// [`PenStream::drain`]: the frame loop reads the stream as late as it can.
    fn pull(&mut self, out: &mut Vec<PenSample>) {
        self.drain(out);
    }
}

/// The `Leave` that ends a stroke `last` left half done (a `Down` or `Drag` with no `Up`), at
/// its position, received `now`; `None` if `last` ended one already.
fn ended(last: &PenSample, now: Instant) -> Option<PenSample> {
    matches!(last.phase, Phase::Down | Phase::Drag).then(|| PenSample {
        phase: Phase::Leave,
        pressure: Unit::ZERO,
        timestamp: now,
        received: now,
        ..*last
    })
}

#[cfg(target_os = "linux")]
pub(crate) use self::wayland::Ours;

/// No pen backend of ours on this platform: winit's events are the fallback.
#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub(crate) enum Ours {}

#[cfg(not(target_os = "linux"))]
#[expect(
    clippy::uninhabited_references,
    reason = "Ours has no value here, so no reference to one can exist and no method be called: \
              the empty matches say so to the compiler"
)]
impl Backend for Ours {
    fn drain(&mut self, _: f64, _: &mut Vec<PenSample>) {
        match *self {}
    }

    fn running(&self) -> bool {
        match *self {}
    }

    fn losses(&self) -> Losses {
        match *self {}
    }

    fn stop(self, _: f64, _: &mut Vec<PenSample>) -> (Losses, Result<()>) {
        match self {}
    }
}

/// The window's pen stream in the app.
pub(crate) type AppPen = PenStream<Ours, Arc<dyn Window>>;

impl AppPen {
    /// Creates the app's window from `attributes`, with decorations off whatever they say, and
    /// its pen stream: our backend on Wayland, winit's tablet events on another platform or if
    /// it fails to start. [`window`](Self::window) is the window.
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
        let (window, undecorated) = undecorated::create(event_loop, attributes)?;
        let scale = window.scale_factor();
        Ok(Self::start(window, scale, |window| {
            spawn_ours(event_loop, Arc::clone(window), undecorated)
        }))
    }
}

/// The witness that a window has no decorations, which our backend's spawn asks for.
///
/// winit's `is_decorated` can't serve: on Hyprland it reports server-side decorations, which
/// leave the client no surfaces, as decorated. So the only way to a witness is to create the
/// window undecorated here.
mod undecorated {
    use std::sync::Arc;

    use anyhow::{Context as _, Result};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::{Window, WindowAttributes};

    /// A window was created with decorations off; only [`create`] makes one.
    #[derive(Debug)]
    pub(crate) struct Undecorated(());

    /// Creates a window from `attributes` with decorations off whatever they say, and the
    /// witness that it has none.
    ///
    /// # Errors
    ///
    /// If winit can't create the window.
    pub(super) fn create(
        event_loop: &dyn ActiveEventLoop,
        attributes: WindowAttributes,
    ) -> Result<(Arc<dyn Window>, Undecorated)> {
        let window = event_loop
            .create_window(attributes.with_decorations(false))
            .context("creating the window")?;
        Ok((Arc::from(window), Undecorated(())))
    }
}

/// Starts our backend for `window`, which `undecorated` proves has no decorations: only for
/// [`AppPen::create_window`], which holds the window for as long as the backend runs (as the
/// backend does itself).
///
/// # Errors
///
/// If our backend can't start: no Wayland tablet support, another platform.
fn spawn_ours(
    event_loop: &dyn ActiveEventLoop,
    window: Arc<dyn Window>,
    undecorated: undecorated::Undecorated,
) -> Result<Ours> {
    #[cfg(target_os = "linux")]
    {
        Ours::spawn(event_loop, window, undecorated)
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Nothing starts: what it was given goes back unused.
        drop((event_loop, window, undecorated));
        Err(anyhow::anyhow!(
            "there is no pen backend of ours on this platform"
        ))
    }
}
