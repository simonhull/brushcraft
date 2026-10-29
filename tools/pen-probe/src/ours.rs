//! Our pen backend, where there is one: `input`'s Wayland backend on Linux, nothing elsewhere.

#[cfg(target_os = "linux")]
pub(crate) use linux::Ours;
#[cfg(not(target_os = "linux"))]
pub(crate) use other::Ours;

/// What a stopped backend leaves behind.
#[derive(Debug)]
pub(crate) struct Stopped {
    /// Events queued after the last drain, read once the input thread had been joined.
    pub(crate) events: Vec<input::Event>,
    /// Events the queue dropped over the whole run.
    pub(crate) dropped: u64,
    /// How the input thread ended.
    pub(crate) result: anyhow::Result<()>,
}

/// The Wayland backend.
#[cfg(target_os = "linux")]
mod linux {
    use anyhow::{Context as _, Result};
    use input::wayland::{self, PenBackend};
    use input::{Event, SurfaceId};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::Window;

    use super::Stopped;

    /// `input`'s Wayland backend, and the probe window's surface on it.
    #[derive(Debug)]
    pub(crate) struct Ours {
        /// The backend, reading on its own thread.
        backend: PenBackend,
        /// The probe window's surface, which the samples recorded must be over.
        surface: SurfaceId,
    }

    impl Ours {
        /// Starts the backend on `event_loop`'s connection, waking the loop after each event.
        ///
        /// # Errors
        ///
        /// If `window` has no Wayland surface, or the backend can't start.
        #[expect(
            unsafe_code,
            reason = "PenBackend::spawn requires stopping the backend before destroying windows"
        )]
        pub(crate) fn start(event_loop: &dyn ActiveEventLoop, window: &dyn Window) -> Result<Self> {
            let surface = wayland::surface_id(&window).context("finding the window's surface")?;
            let proxy = event_loop.create_proxy();
            // SAFETY: `spawn` needs every window on this connection to outlive the backend. The
            // probe's one window lives in `App::surface`, and the backend (this `Ours`) in
            // `App::backend`: `App::finish` (main.rs) stops the backend before the loop exits, and
            // should it not run, `App` declares `backend` before `surface`, so the backend drops
            // (joining its thread) before the window. What remains is winit destroying surfaces of
            // its own, such as fallback decoration subsurfaces, as a pen enters them: the residual
            // race `spawn`'s `# Safety` section describes, accepted for a diagnostic tool. The
            // window asks for no decorations (main.rs), which leaves winit fewer such surfaces,
            // but nothing here relies on it.
            let backend = unsafe {
                PenBackend::spawn(event_loop.owned_display_handle(), move || proxy.wake_up())
            }?;
            Ok(Self { backend, surface })
        }

        /// The probe window's surface, which our samples over it carry.
        pub(crate) const fn surface(&self) -> SurfaceId {
            self.surface
        }

        /// Every event queued so far.
        pub(crate) fn drain(&self) -> Vec<Event> {
            self.backend.events().try_iter().collect()
        }

        /// Stops the input thread, then takes what it queued since the last drain and the final
        /// drop count.
        pub(crate) fn stop(self) -> Stopped {
            let (events, result) = self.backend.stop();
            Stopped {
                events: events.try_iter().collect(),
                dropped: events.dropped(),
                result: result.map_err(anyhow::Error::from),
            }
        }
    }
}

/// No backend: every platform but Linux.
#[cfg(not(target_os = "linux"))]
mod other {
    use anyhow::{Result, bail};
    use input::{Event, SurfaceId};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::Window;

    use super::Stopped;

    /// No backend on this platform: this type has no values.
    #[derive(Debug)]
    pub(crate) enum Ours {}

    impl Ours {
        /// Always fails: `input` has no backend here.
        ///
        /// # Errors
        ///
        /// Always.
        pub(crate) fn start(_: &dyn ActiveEventLoop, _: &dyn Window) -> Result<Self> {
            bail!("input has no backend on this platform")
        }

        /// Never called: there are no values.
        pub(crate) const fn surface(&self) -> SurfaceId {
            match *self {}
        }

        /// Never called: there are no values.
        pub(crate) const fn drain(&self) -> Vec<Event> {
            match *self {}
        }

        /// Never called: there are no values.
        pub(crate) const fn stop(self) -> Stopped {
            match self {}
        }
    }
}
