//! Pen input from the Wayland tablet protocol (`zwp_tablet_manager_v2`), on a thread of its own.
//!
//! The backend attaches to the application's existing Wayland connection (pen events name the
//! surface they're over, and that surface only exists on that connection), with its own event
//! queue, and reads it on a dedicated thread so samples are timestamped as soon as they arrive.

mod pad;
mod tool;

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::thread::{self, JoinHandle};

use raw_window_handle::{
    HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use rustix::event::{PollFd, PollFlags, poll};
use rustix::io::Errno;
use wayland_backend::client::{Backend, ObjectId, WaylandError};
use wayland_client::globals::{
    BindError, GlobalError, GlobalList, GlobalListContents, registry_queue_init,
};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_seat::WlSeat;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, DispatchError, EventQueue, Proxy, QueueHandle};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_manager_v2::ZwpTabletManagerV2;
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_v2::ZwpTabletPadV2;
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_seat_v2::{self, ZwpTabletSeatV2};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_tool_v2::ZwpTabletToolV2;
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_v2::ZwpTabletV2;

use crate::queue::{self, Events, Sender};
use crate::{Event, SurfaceId};

/// Events buffered before the backend starts dropping: about 6 s of pen samples.
const QUEUE_CAPACITY: usize = 1024;

/// Why the backend couldn't start or stopped.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The display or window isn't a Wayland one (X11, or not Linux).
    #[error("not a Wayland display or window")]
    NotWayland,
    /// The windowing layer couldn't provide a handle.
    #[error("no display or window handle: {0}")]
    Handle(#[from] HandleError),
    /// The compositor doesn't support graphics tablets.
    #[error("the compositor doesn't offer zwp_tablet_manager_v2")]
    NoTabletManager,
    /// Listing the compositor's globals failed.
    #[error("listing Wayland globals: {0}")]
    Globals(#[from] GlobalError),
    /// Binding a global failed.
    #[error("binding a Wayland global: {0}")]
    Bind(BindError),
    /// Dispatching events failed.
    #[error("dispatching Wayland events: {0}")]
    Dispatch(#[from] DispatchError),
    /// The connection failed.
    #[error("Wayland connection: {0}")]
    Wayland(#[from] WaylandError),
    /// A system call failed.
    #[error("input thread: {0}")]
    Io(#[from] std::io::Error),
    /// The input thread panicked.
    #[error("the input thread panicked")]
    Panicked,
}

impl From<Errno> for BackendError {
    fn from(errno: Errno) -> Self {
        Self::Io(errno.into())
    }
}

/// The [`SurfaceId`] that samples over `window` will carry.
///
/// # Errors
///
/// [`BackendError::NotWayland`] for a non-Wayland window; [`BackendError::Handle`] if the window
/// has no handle right now.
pub fn surface_id(window: &impl HasWindowHandle) -> Result<SurfaceId, BackendError> {
    let RawWindowHandle::Wayland(handle) = window.window_handle()?.as_raw() else {
        return Err(BackendError::NotWayland);
    };
    Ok(SurfaceId::from_raw(handle.surface.as_ptr().addr()))
}

/// Pen and pad input from the compositor, read on a thread of its own.
///
/// Dropping it stops the thread.
#[derive(Debug)]
pub struct PenBackend {
    /// Write end of the stop pipe: dropping it wakes the thread, which then exits.
    stop: Option<OwnedFd>,
    thread: Option<JoinHandle<Result<(), BackendError>>>,
    events: Events<Event>,
}

impl PenBackend {
    /// Starts reading pen input from the Wayland connection behind `display`.
    ///
    /// `display` must own its connection (winit's `OwnedDisplayHandle` does): the backend keeps it
    /// until the thread ends, so the connection outlives every use. `wake` runs on the input thread
    /// after each event is queued; use it to wake the application's event loop.
    ///
    /// # Errors
    ///
    /// [`BackendError::NotWayland`] if `display` isn't Wayland; [`BackendError::NoTabletManager`]
    /// if the compositor has no tablet support; otherwise a Wayland or system error.
    #[expect(
        unsafe_code,
        reason = "attaching to the application's wl_display is input's one FFI boundary"
    )]
    pub fn spawn<D, W>(display: D, wake: W) -> Result<Self, BackendError>
    where
        D: HasDisplayHandle + Send + Sync + 'static,
        W: Fn() + Send + 'static,
    {
        let RawDisplayHandle::Wayland(raw) = display.display_handle()?.as_raw() else {
            return Err(BackendError::NotWayland);
        };
        // SAFETY: `from_foreign_display` needs the `wl_display` to stay live while the backend (and
        // its clones, held by `conn` and by the event queue) exist. `display` owns the connection
        // that `raw` points into. Until the `Worker` is built, `display` is a parameter and so is
        // dropped after every local; `connect` drops its other temporaries before returning. From
        // then on `display` lives in the `Worker` next to `conn` and the queue, and its fields drop
        // in declaration order, `_display` last, whether the thread finishes or never starts. If
        // `PenBackend` is leaked, the thread keeps the `Worker` alive, so leaking can't break this
        // either.
        let backend = unsafe { Backend::from_foreign_display(raw.display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);
        let (queue, state, events) = connect(&conn, Box::new(wake))?;

        let (stop_read, stop_write) = rustix::pipe::pipe()?;
        let worker = Worker {
            queue,
            state,
            conn,
            stop: stop_read,
            _display: display,
        };
        let thread = thread::Builder::new()
            .name("pen-input".into())
            .spawn(move || worker.run())?;
        Ok(Self {
            stop: Some(stop_write),
            thread: Some(thread),
            events,
        })
    }

    /// Events received so far.
    #[must_use]
    pub const fn events(&self) -> &Events<Event> {
        &self.events
    }

    /// Stops the input thread and reports how it ended.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread early, or [`BackendError::Panicked`].
    pub fn stop(mut self) -> Result<(), BackendError> {
        self.shutdown()
    }

    /// Closes the stop pipe and joins the thread, once.
    ///
    /// # Errors
    ///
    /// As [`PenBackend::stop`].
    fn shutdown(&mut self) -> Result<(), BackendError> {
        drop(self.stop.take());
        self.thread.take().map_or(Ok(()), |thread| {
            thread.join().map_err(|_panic| BackendError::Panicked)?
        })
    }
}

impl Drop for PenBackend {
    fn drop(&mut self) {
        // A failure was already visible as missing events; `stop` is there for callers who care.
        drop(self.shutdown());
    }
}

/// Creates our event queue, binds the tablet manager and the seats, and sets up the hand-off.
///
/// Everything else it creates (the global list, the queue handle) is dropped before it returns,
/// so the only holders of `conn`'s backend are `conn` itself and the returned queue.
///
/// # Errors
///
/// [`BackendError::NoTabletManager`] if the compositor has no tablet support; otherwise a Wayland
/// error.
fn connect(
    conn: &Connection,
    wake: Box<dyn Fn() + Send>,
) -> Result<(EventQueue<State>, State, Events<Event>), BackendError> {
    let (globals, queue) = registry_queue_init::<State>(conn)?;
    let qh = queue.handle();
    let manager: ZwpTabletManagerV2 = globals.bind(&qh, 1..=2, ()).map_err(|e| match e {
        BindError::NotPresent => BackendError::NoTabletManager,
        other @ BindError::UnsupportedVersion => BackendError::Bind(other),
    })?;
    let (sender, events) = queue::channel(QUEUE_CAPACITY);
    let state = State {
        manager,
        sender,
        wake,
        tools: HashMap::new(),
    };
    state.bind_existing_seats(&globals, &qh);
    Ok((queue, state, events))
}

/// What the input thread owns.
///
/// Fields drop in declaration order, so the queue and `conn` (the backend's only holders) go
/// before `_display`, which keeps the `wl_display` they point into alive.
struct Worker<D> {
    queue: EventQueue<State>,
    state: State,
    conn: Connection,
    /// Read end of the stop pipe: readable (hung up) once [`PenBackend`] closes the write end.
    stop: OwnedFd,
    /// Held only so it drops last: it keeps the `wl_display` alive.
    _display: D,
}

impl<D> Worker<D> {
    /// Reads and dispatches our queue until the stop pipe closes.
    ///
    /// # Errors
    ///
    /// A Wayland or system error that ends the thread.
    fn run(mut self) -> Result<(), BackendError> {
        loop {
            self.queue.dispatch_pending(&mut self.state)?;
            self.conn.flush()?;
            let Some(guard) = self.queue.prepare_read() else {
                continue;
            };
            let wayland_fd = guard.connection_fd();
            let mut fds = [
                PollFd::new(&wayland_fd, PollFlags::IN),
                PollFd::new(&self.stop, PollFlags::IN),
            ];
            match poll(&mut fds, None) {
                Err(Errno::INTR) => continue,
                result => result?,
            };
            let [wayland_poll, stop_poll] = &fds;
            if !stop_poll.revents().is_empty() {
                return Ok(());
            }
            if wayland_poll.revents().contains(PollFlags::IN) {
                match guard.read() {
                    Ok(_) => {}
                    Err(WaylandError::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
    }
}

impl<D> std::fmt::Debug for Worker<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

/// Everything the input thread tracks.
struct State {
    manager: ZwpTabletManagerV2,
    sender: Sender<Event>,
    /// Boxed so the many `Dispatch` impls below stay non-generic; called once per event.
    wake: Box<dyn Fn() + Send>,
    tools: HashMap<ObjectId, tool::ToolState>,
}

impl State {
    fn emit(&self, event: Event) {
        // A disconnected consumer just means nobody is listening any more.
        if self.sender.send(event).is_ok() {
            (self.wake)();
        }
    }

    fn bind_existing_seats(&self, globals: &GlobalList, qh: &QueueHandle<Self>) {
        let seats: Vec<(u32, u32)> = globals.contents().with_list(|list| {
            list.iter()
                .filter(|g| g.interface == "wl_seat")
                .map(|g| (g.name, g.version))
                .collect()
        });
        for (name, version) in seats {
            self.add_seat(globals.registry(), name, version, qh);
        }
    }

    fn add_seat(&self, registry: &WlRegistry, name: u32, version: u32, qh: &QueueHandle<Self>) {
        let seat: WlSeat = registry.bind(name, version.min(1), qh, ());
        self.manager.get_tablet_seat(&seat, qh, ());
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("tools", &self.tools.len())
            .finish_non_exhaustive()
    }
}

fn surface_of(surface: &WlSurface) -> SurfaceId {
    SurfaceId::from_raw(surface.id().as_ptr().addr())
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(
        state: &mut Self,
        proxy: &WlRegistry,
        event: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        qhandle: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
            && interface == "wl_seat"
        {
            state.add_seat(proxy, name, version, qhandle);
        }
    }
}

/// Seats are bound only to ask for their tablet seat; their own events don't matter here.
impl Dispatch<WlSeat, ()> for State {
    fn event(
        _: &mut Self,
        _: &WlSeat,
        _: <WlSeat as Proxy>::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// The manager has no events.
impl Dispatch<ZwpTabletManagerV2, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZwpTabletManagerV2,
        _: <ZwpTabletManagerV2 as Proxy>::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// Announces tablets, tools and pads; each then gets events through its own `Dispatch`.
impl Dispatch<ZwpTabletSeatV2, ()> for State {
    wayland_client::event_created_child!(Self, ZwpTabletSeatV2, [
        zwp_tablet_seat_v2::EVT_TABLET_ADDED_OPCODE => (ZwpTabletV2, ()),
        zwp_tablet_seat_v2::EVT_TOOL_ADDED_OPCODE => (ZwpTabletToolV2, ()),
        zwp_tablet_seat_v2::EVT_PAD_ADDED_OPCODE => (ZwpTabletPadV2, ()),
    ]);

    fn event(
        _: &mut Self,
        _: &ZwpTabletSeatV2,
        _: <ZwpTabletSeatV2 as Proxy>::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

/// Tablet descriptions (name, USB ids) aren't needed yet.
impl Dispatch<ZwpTabletV2, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZwpTabletV2,
        _: <ZwpTabletV2 as Proxy>::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
