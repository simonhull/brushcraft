//! Our Wayland pen backend in the app: `input`'s [`PenBackend`], spawned on winit's connection.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use input::wayland::{self, PenBackend};
use input::{Event, Position, StrokeSample, SurfaceId};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::undecorated::Undecorated;
use super::{Backend, Losses, PenSample};
use crate::camera::Pixel;

/// `input`'s Wayland backend, the window's surface on it, and the window itself.
///
/// Only [`super::spawn_ours`] makes one, and only [`super::AppPen::create_window`] calls that,
/// which puts it straight into a [`super::PenStream`] holding the window too: the proof in
/// [`Ours::spawn`] rests on that, and on the field order here.
#[derive(Debug)]
pub(crate) struct Ours {
    /// The backend, reading on its own thread. Declared before `_window`, so it stops (its
    /// thread joined) before this hold on the window goes.
    backend: PenBackend,
    /// The window's surface: samples over anything else aren't ours.
    surface: SurfaceId,
    /// The window, held for as long as the backend runs. Declared after `backend`.
    _window: Arc<dyn Window>,
}

impl Ours {
    /// Starts the backend on `event_loop`'s connection for `window`, which `_undecorated`
    /// proves has no decorations, waking the loop after each event.
    ///
    /// # Errors
    ///
    /// If `window` has no Wayland surface, or the backend can't start (no tablet support, a
    /// Wayland or system error).
    pub(super) fn spawn(
        event_loop: &dyn ActiveEventLoop,
        window: Arc<dyn Window>,
        _undecorated: Undecorated,
    ) -> Result<Self> {
        let surface =
            wayland::surface_id(&window.as_ref()).context("finding the window's surface")?;
        let display = event_loop.owned_display_handle();
        let proxy = event_loop.create_proxy();
        let wake = move || proxy.wake_up();
        #[expect(
            unsafe_code,
            reason = "PenBackend::spawn attaches to winit's wl_display; the window is held past it"
        )]
        // SAFETY: `spawn` needs no `wl_surface` on winit's connection destroyed while the backend
        // runs; the hazard is an event on our queue naming a surface that winit destroys as our
        // thread dispatches it (wayland-backend reads the named object's user data). The events
        // that name a surface are a tool's `proximity_in` and a pad's `enter` and `leave`, and
        // they name only a mapped surface that takes input: this client has one, the window's
        // own toplevel surface.
        // - The window's surface outlives the backend. Dropping the last `Arc<dyn Window>` only
        //   marks the window closed (winit-wayland 0.31.0-beta.3's `impl Drop for Window`); the
        //   event loop destroys the surface later, on this thread, or with itself after `run_app`
        //   has dropped the app (`drop(app)`, `event_loop/mod.rs`). This backend holds an `Arc`
        //   (`_window`, declared after `backend`, so dropping `Ours` joins the thread first), and
        //   so does the `PenStream` it goes into straight away (`AppPen::create_window`, through
        //   `spawn_ours`: nothing else makes an `Ours`), which declares the backend before its own
        //   `Arc` too; its `stop`, which the exit path calls first (`WindowApp::finish`), joins the
        //   thread before handing its `Arc` back.
        // - winit's own surfaces never reach such an event. `_undecorated` proves the window was
        //   created with decorations off (only `undecorated::create` makes the witness, as winit's
        //   `is_decorated` reports Hyprland's server-side decorations as decorated), and winit has
        //   returned it, so its first configure is done: `Window::new` waits for it ("Wait for the
        //   initial configure", `window/mod.rs`). In server-side decoration mode winit has no frame
        //   at all. In client-side mode the configure made a frame and hid it
        //   (`frame.set_hidden(true)`, `window/state.rs`), and smithay-client-toolkit 0.21.1's
        //   `FallbackFrame::set_hidden` destroyed its subsurfaces before they ever had a buffer; a
        //   later frame (after a switch of mode) is hidden the same way, never mapped, so no event
        //   can name it. Cursor surfaces take no input. Recheck all of this on a winit or
        //   smithay-client-toolkit bump.
        // - `display` is winit's `OwnedDisplayHandle`, which `spawn` keeps alive for as long as its
        //   thread uses the connection, as raw-window-handle's contract asks.
        // - `wake` only wakes the event loop; it never blocks on the thread that stops the backend.
        let backend = unsafe { PenBackend::spawn(display, wake) }
            .context("starting our Wayland pen backend")?;
        Ok(Self {
            backend,
            surface,
            _window: window,
        })
    }
}

impl Backend for Ours {
    fn drain(&mut self, scale: f64, out: &mut Vec<PenSample>) {
        keep(self.surface, self.backend.events().try_iter(), scale, out);
    }

    fn running(&self) -> bool {
        self.backend.is_running()
    }

    fn losses(&self) -> Losses {
        let events = self.backend.events();
        Losses {
            dropped: events.dropped(),
            lost: events.lost_essential(),
        }
    }

    fn stop(self, scale: f64, out: &mut Vec<PenSample>) -> (Losses, Result<()>) {
        let Self {
            backend, surface, ..
        } = self;
        let (events, result) = backend.stop();
        keep(surface, events.try_iter(), scale, out);
        let losses = Losses {
            dropped: events.dropped(),
            lost: events.lost_essential(),
        };
        (losses, result.context("our Wayland pen backend"))
    }
}

/// Appends the samples among `events` that are over `surface` to `out`, in physical pixels at
/// `scale`; pad events aren't read yet.
fn keep(
    surface: SurfaceId,
    events: impl Iterator<Item = Event>,
    scale: f64,
    out: &mut Vec<PenSample>,
) {
    let ours = events.filter_map(|event| match event {
        Event::Sample(sample) if sample.surface == surface => {
            Some(from_stroke_sample(&sample, scale))
        }
        Event::Sample(_) | Event::Pad(_) => None,
    });
    out.extend(ours);
}

/// `sample` in the app's terms, its position in physical pixels at `scale`.
fn from_stroke_sample(sample: &StrokeSample, scale: f64) -> PenSample {
    PenSample {
        phase: sample.phase,
        tool: sample.tool,
        at: physical(sample.pos, scale),
        pressure: sample.pressure,
        distance: sample.distance,
        tilt: sample.tilt,
        roll: sample.roll,
        buttons: sample.buttons,
        timestamp: sample.timestamp,
        received: sample.received,
    }
}

/// `pos`, in logical pixels, in physical ones at `scale`.
fn physical(pos: Position, scale: f64) -> Pixel {
    Pixel {
        x: pos.x * scale,
        y: pos.y * scale,
    }
}
