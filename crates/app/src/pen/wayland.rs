//! Our Wayland pen backend in the app: `input`'s [`PenBackend`], spawned on winit's connection.

use anyhow::{Context as _, Result};
use input::wayland::{self, PenBackend};
use input::{Event, Position, StrokeSample, SurfaceId};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use super::{Backend, PenSample};
use crate::camera::Pixel;

/// `input`'s Wayland backend, and the window's surface on it.
///
/// Only [`super::spawn_ours`] makes one, and only [`super::AppPen::create_window`] calls that,
/// which puts it straight into a [`super::PenStream`] holding the window: the proof in
/// [`Ours::spawn`] rests on that.
#[derive(Debug)]
pub(crate) struct Ours {
    /// The backend, reading on its own thread.
    backend: PenBackend,
    /// The window's surface: samples over anything else aren't ours.
    surface: SurfaceId,
}

impl Ours {
    /// Starts the backend on `event_loop`'s connection for `window`, waking the loop after each
    /// event.
    ///
    /// The caller ([`super::AppPen::create_window`]) made `window` undecorated.
    ///
    /// # Errors
    ///
    /// If `window` has no Wayland surface (X11), or the backend can't start (no tablet support,
    /// a Wayland or system error).
    pub(super) fn spawn(event_loop: &dyn ActiveEventLoop, window: &dyn Window) -> Result<Self> {
        let surface = wayland::surface_id(&window).context("finding the window's surface")?;
        let proxy = event_loop.create_proxy();
        #[expect(
            unsafe_code,
            reason = "PenBackend::spawn attaches to winit's wl_display; the stream keeps the window alive past it"
        )]
        // SAFETY: `spawn` needs no `wl_surface` on winit's connection destroyed while the backend
        // runs; the hazard is a tool event naming a surface that winit destroys as our thread
        // dispatches it. A tool event (`proximity_in`) names only a mapped surface that takes
        // input, and this client has one: the window's own.
        // - The window outlives the backend. Only `AppPen::create_window` reaches this (through
        //   `spawn_ours`), and it puts the backend straight into a `PenStream` holding an `Arc` of
        //   the window (the last `Arc`'s drop is what destroys the surface). The stream declares
        //   the backend before the window, so dropping it joins the backend's thread before its
        //   `Arc` goes; its `stop`, which the window's exit path calls first (`WindowApp::finish`),
        //   stops the backend before handing the `Arc` back.
        // - winit's own surfaces never reach a tool event. `create_window` made the window with
        //   decorations off, and winit returned it, so its first configure is done: winit-wayland
        //   0.31.0-beta.3's `Window::new` waits for it ("Wait for the initial configure",
        //   `window/mod.rs`). In server-side decoration mode winit has no frame at all. In
        //   client-side mode the configure made a frame and hid it (`frame.set_hidden(true)`,
        //   `window/state.rs`), and smithay-client-toolkit 0.21.1's `FallbackFrame::set_hidden`
        //   destroyed its subsurfaces before they ever had a buffer; a later frame (after a switch
        //   of mode) is hidden the same way, never mapped, so no tool event can name it. Cursor
        //   surfaces take no input. Recheck both on a winit or smithay-client-toolkit bump.
        // - The display handle is winit's `OwnedDisplayHandle`, which `spawn` keeps alive for as
        //   long as its thread uses the connection, as raw-window-handle's contract asks.
        // - `wake` only wakes the event loop; it never blocks on the thread that stops the backend.
        let backend = unsafe {
            PenBackend::spawn(event_loop.owned_display_handle(), move || proxy.wake_up())
        }
        .context("starting our Wayland pen backend")?;
        Ok(Self { backend, surface })
    }
}

impl Backend for Ours {
    fn drain(&mut self, scale: f64, out: &mut Vec<PenSample>) {
        keep(self.surface, self.backend.events().try_iter(), scale, out);
    }

    fn running(&self) -> bool {
        self.backend.is_running()
    }

    fn losses(&self) -> (u64, u64) {
        let events = self.backend.events();
        (events.dropped(), events.lost_essential())
    }

    fn stop(self, scale: f64, out: &mut Vec<PenSample>) -> Result<()> {
        let Self { backend, surface } = self;
        let (events, result) = backend.stop();
        keep(surface, events.try_iter(), scale, out);
        result.context("our Wayland pen backend")
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
