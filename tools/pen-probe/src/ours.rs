//! Our pen backend, where there is one: `input`'s Wayland backend on Linux, nothing elsewhere.

#[cfg(target_os = "linux")]
pub(crate) use linux::Ours;
#[cfg(not(target_os = "linux"))]
pub(crate) use other::Ours;

#[cfg(target_os = "linux")]
mod linux {
    use input::wayland::{self, PenBackend};
    use input::{Event, SurfaceId};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::Window;

    /// `input`'s Wayland backend, and the probe window's surface on it.
    #[derive(Debug)]
    pub(crate) struct Ours {
        backend: PenBackend,
        surface: SurfaceId,
    }

    impl Ours {
        /// Starts the backend on `event_loop`'s connection, waking the loop after each event.
        ///
        /// # Errors
        ///
        /// Why the backend couldn't start, as a sentence for the summary.
        pub(crate) fn start(
            event_loop: &dyn ActiveEventLoop,
            window: &dyn Window,
        ) -> Result<Self, String> {
            let surface = wayland::surface_id(&window).map_err(|e| e.to_string())?;
            let proxy = event_loop.create_proxy();
            let backend =
                PenBackend::spawn(event_loop.owned_display_handle(), move || proxy.wake_up())
                    .map_err(|e| e.to_string())?;
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

        /// Events the queue dropped because the probe fell behind.
        pub(crate) fn dropped(&self) -> u64 {
            self.backend.events().dropped()
        }

        /// Stops the input thread.
        ///
        /// # Errors
        ///
        /// What stopped the thread early, as a sentence for the summary.
        pub(crate) fn stop(self) -> Result<(), String> {
            self.backend.stop().map_err(|e| e.to_string())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod other {
    use input::{Event, SurfaceId};
    use winit::event_loop::ActiveEventLoop;
    use winit::window::Window;

    /// No backend on this platform: this type has no values.
    #[derive(Debug)]
    pub(crate) enum Ours {}

    impl Ours {
        /// Always fails: `input` has no backend here.
        ///
        /// # Errors
        ///
        /// Always.
        pub(crate) fn start(_: &dyn ActiveEventLoop, _: &dyn Window) -> Result<Self, String> {
            Err("input has no backend on this platform".to_owned())
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
        pub(crate) const fn dropped(&self) -> u64 {
            match *self {}
        }

        /// Never called: there are no values.
        ///
        /// # Errors
        ///
        /// Never.
        pub(crate) const fn stop(self) -> Result<(), String> {
            match self {}
        }
    }
}
