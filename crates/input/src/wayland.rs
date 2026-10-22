//! Pen input from the Wayland tablet protocol (`zwp_tablet_manager_v2`), on a thread of its own.
//!
//! The backend attaches to the application's existing Wayland connection (pen events name the
//! surface they're over, and that surface only exists on that connection), with its own event
//! queue, and reads it on a dedicated thread so samples are timestamped as soon as they arrive.

mod pad;
mod tool;

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use raw_window_handle::{
    HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use rustix::event::{EventfdFlags, PollFd, PollFlags, eventfd, poll};
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
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_v2::{self, ZwpTabletV2};

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
    #[error("no display or window handle")]
    Handle(#[from] HandleError),
    /// The compositor doesn't support graphics tablets.
    #[error("the compositor doesn't offer zwp_tablet_manager_v2")]
    NoTabletManager,
    /// Listing the compositor's globals failed.
    #[error("listing Wayland globals")]
    Globals(#[from] GlobalError),
    /// Binding a global failed.
    #[error("binding a Wayland global")]
    Bind(#[source] BindError),
    /// Dispatching events failed.
    #[error("dispatching Wayland events")]
    Dispatch(#[from] DispatchError),
    /// The connection failed.
    #[error("the Wayland connection failed")]
    Wayland(#[from] WaylandError),
    /// A system call failed.
    #[error("a system call on the input thread failed")]
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
    /// The input thread, stopped and joined when the backend is.
    thread: InputThread,
    /// Where the thread's events arrive.
    events: Events<Event>,
}

/// The input thread, joined when this is stopped or dropped.
#[derive(Debug)]
struct InputThread {
    /// Signalled (made readable) to tell the thread to exit.
    stop: Arc<OwnedFd>,
    /// Set by the thread as it exits, before its last `wake`.
    exited: Arc<AtomicBool>,
    /// The thread, with what ended it; `None` once joined.
    handle: Option<JoinHandle<Result<(), BackendError>>>,
}

impl PenBackend {
    /// Starts reading pen input from the Wayland connection behind `display`.
    ///
    /// The backend holds `display` until its thread ends and uses the `wl_display` only while it
    /// does, which is what raw-window-handle's contract asks for. With winit, pass
    /// `event_loop.owned_display_handle()`.
    ///
    /// `wake` runs on the input thread after each event is queued, and once more if the thread
    /// stops on an error; use it to wake the application's event loop, and check
    /// [`is_running`](Self::is_running) there. It must not block waiting on the thread that drops
    /// the `PenBackend`: dropping joins the input thread, so that would deadlock.
    ///
    /// The backend binds its own tablet seats on its own event queue, so a second `spawn` on the
    /// same connection gets its own seats and events; each backend destroys its objects when it
    /// stops.
    ///
    /// # Safety
    ///
    /// No `wl_surface` on `display`'s connection may be destroyed while the backend runs: stop
    /// (or drop) the backend before destroying any window.
    ///
    /// The backend shares the connection with the windowing library as a "guest" on its
    /// `wl_display`. In that mode, when wayland-backend dispatches an event whose argument is an
    /// object the windowing library owns (the `wl_surface` in a tool's `proximity_in`), it reads
    /// that object's user data on our thread. If the windowing library destroys the surface at the
    /// same moment, that read is a use-after-free.
    ///
    /// Stopping first covers every window the application destroys, but the windowing library may
    /// destroy surfaces of its own while the backend runs (winit tears down the subsurfaces of its
    /// fallback client-side decorations, for example). A caller that can't rule that out, which
    /// with winit means every caller, accepts that residual race; it needs a surface destroyed
    /// in the instant a pen enters it.
    ///
    /// # Errors
    ///
    /// [`BackendError::NotWayland`] if `display` isn't Wayland; [`BackendError::NoTabletManager`]
    /// if the compositor has no tablet support; otherwise a Wayland or system error.
    #[expect(
        unsafe_code,
        reason = "attaching to the application's wl_display is input's one FFI boundary"
    )]
    pub unsafe fn spawn<D, W>(display: D, wake: W) -> Result<Self, BackendError>
    where
        D: HasDisplayHandle + Send + 'static,
        W: Fn() + Send + 'static,
    {
        let RawDisplayHandle::Wayland(raw) = display.display_handle()?.as_raw() else {
            return Err(BackendError::NotWayland);
        };
        // SAFETY: `from_foreign_display` needs the `wl_display` to stay live while the backend and
        // its clones (held by `conn` and by the event queue) exist. raw-window-handle 0.6 makes the
        // handle valid for as long as its source can be borrowed, so we hold `display`, never
        // dropping it, for every use of the pointer: the approach wgpu takes for surfaces. Until
        // the `Worker` is built, `display` is a parameter and so is dropped after every local, and
        // `connect` drops its other temporaries before returning. From then on `display` lives in
        // the `Worker` beside `conn` and the queue, whose fields drop in declaration order,
        // `_display` last, after `Worker::drop` has used the connection for the last time, whether
        // the thread finishes, panics or never starts. If `PenBackend` is leaked, the thread
        // keeps the `Worker` alive, so leaking can't break this either. `display` only has to be
        // `Send`: it moves to the input thread and is never shared. The other hazard of guest mode
        // (reading the user data of the windowing library's objects as they are destroyed) is the
        // caller's to rule out, per this function's `# Safety` section.
        let backend = unsafe { Backend::from_foreign_display(raw.display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);
        let (queue, state, events) = connect(&conn, Box::new(wake))?;

        let stop = Arc::new(eventfd(0, EventfdFlags::CLOEXEC)?);
        let exited = Arc::new(AtomicBool::new(false));
        let worker = Worker {
            queue,
            state,
            conn,
            stop: Arc::clone(&stop),
            exited: Arc::clone(&exited),
            _display: display,
        };
        let thread = thread::Builder::new()
            .name("pen-input".into())
            .spawn(move || worker.run())?;
        Ok(Self {
            thread: InputThread {
                stop,
                exited,
                handle: Some(thread),
            },
            events,
        })
    }

    /// Events received so far.
    #[must_use]
    pub const fn events(&self) -> &Events<Event> {
        &self.events
    }

    /// Whether the input thread is still reading events.
    ///
    /// It turns false when the thread stops on its own, from a Wayland or system error or a panic
    /// in `wake`; [`stop`](Self::stop) then reports why. On an error the thread calls `wake` once
    /// more after this has turned false, so checking it on each wake catches a dead thread. A
    /// panic gives no such wake: this turns false as the thread unwinds, seen on the next check.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.thread.is_running()
    }

    /// Stops the input thread and reports how it ended, handing back the queue.
    ///
    /// The thread has been joined when this returns, so the queue holds every event it delivered:
    /// drain it to keep what arrived since the last drain.
    ///
    /// # Errors
    ///
    /// The error that stopped the thread early, or [`BackendError::Panicked`], beside the queue.
    pub fn stop(self) -> (Events<Event>, Result<(), BackendError>) {
        let Self { mut thread, events } = self;
        let result = thread.shutdown();
        (events, result)
    }
}

impl InputThread {
    /// Whether the thread is neither exiting nor finished.
    fn is_running(&self) -> bool {
        // `exited` is set before the error-exit wake, which `is_finished` alone could race.
        !self.exited.load(Ordering::Acquire)
            && self
                .handle
                .as_ref()
                .is_some_and(|handle| !handle.is_finished())
    }

    /// Signals the thread to stop and joins it, once.
    ///
    /// # Errors
    ///
    /// As [`PenBackend::stop`], or [`BackendError::Io`] if the stop signal can't be sent (the
    /// thread is then left running, detached, rather than joined forever).
    fn shutdown(&mut self) -> Result<(), BackendError> {
        let Some(thread) = self.handle.take() else {
            return Ok(());
        };
        // An eventfd write adds to its counter, making it readable; it can't hit a closed reader.
        rustix::io::write(&*self.stop, &1_u64.to_ne_bytes())?;
        thread.join().map_err(|_panic| BackendError::Panicked)?
    }
}

impl Drop for InputThread {
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
    let mut state = State {
        manager,
        sender,
        wake,
        seats: HashMap::new(),
        tablets: HashMap::new(),
        tools: HashMap::new(),
        pads: HashMap::new(),
    };
    state.bind_existing_seats(&globals, &qh);
    Ok((queue, state, events))
}

/// Ignores [`WaylandError::Io`] with `WouldBlock`: the socket is shared and momentarily full or
/// empty, and the next pass of the loop retries.
///
/// # Errors
///
/// Any other error.
fn retry_later<T>(result: Result<T, WaylandError>) -> Result<(), WaylandError> {
    match result {
        Ok(_) => Ok(()),
        Err(WaylandError::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Err(e) => Err(e),
    }
}

/// What the input thread owns.
///
/// Fields drop in declaration order, so the queue and `conn` (the backend's only holders) go
/// before `_display`, which keeps the `wl_display` they point into alive. `Drop::drop` runs before
/// any field is dropped, so its use of the connection is covered too.
struct Worker<D> {
    /// The thread's own event queue on the connection, which dispatches into `state`.
    queue: EventQueue<State>,
    /// Everything the dispatch tracks.
    state: State,
    /// The connection to the compositor, wrapping the application's `wl_display`.
    conn: Connection,
    /// An eventfd that [`PenBackend`] signals to stop the thread.
    stop: Arc<OwnedFd>,
    /// Set as the thread exits, for [`PenBackend::is_running`].
    exited: Arc<AtomicBool>,
    /// Held only so it drops last: it keeps the `wl_display` alive.
    _display: D,
}

impl<D> Worker<D> {
    /// Dispatches until stopped; dropping `self` then destroys our protocol objects.
    ///
    /// # Errors
    ///
    /// A Wayland or system error that ended the thread early.
    fn run(mut self) -> Result<(), BackendError> {
        let result = self.pump();
        if result.is_err() {
            // Nobody asked us to stop: wake the application so it notices, after `is_running` has
            // turned false.
            self.exited.store(true, Ordering::Release);
            (self.state.wake)();
        }
        result
    }

    /// Reads and dispatches our queue until the stop eventfd is signalled.
    ///
    /// # Errors
    ///
    /// A Wayland or system error that ends the thread.
    fn pump(&mut self) -> Result<(), BackendError> {
        loop {
            self.queue.dispatch_pending(&mut self.state)?;
            retry_later(self.conn.flush())?;
            let Some(guard) = self.queue.prepare_read() else {
                continue;
            };
            let wayland_fd = guard.connection_fd();
            let mut fds = [
                PollFd::new(&wayland_fd, PollFlags::IN),
                PollFd::new(&*self.stop, PollFlags::IN),
            ];
            match poll(&mut fds, None) {
                Err(Errno::INTR) => continue,
                result => result?,
            };
            let [wayland_poll, stop_poll] = &fds;
            if !stop_poll.revents().is_empty() {
                return Ok(());
            }
            // On an error or hang-up without input, `read` reports it rather than spinning here.
            if !wayland_poll.revents().is_empty() {
                retry_later(guard.read())?;
            }
        }
    }
}

/// Runs however the worker ends: a normal stop, an error, a panic in `wake`, or a thread that
/// never started.
impl<D> Drop for Worker<D> {
    fn drop(&mut self) {
        self.exited.store(true, Ordering::Release);
        // Guest-mode backends must clean up their objects before going away; best effort, since
        // the connection may be what failed.
        self.state.destroy_all();
        drop(self.conn.flush());
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
    /// The tablet manager global, which gives each seat its tablet seat.
    manager: ZwpTabletManagerV2,
    /// Where events go, for [`PenBackend`]'s [`Events`].
    sender: Sender<Event>,
    /// Boxed so the many `Dispatch` impls below stay non-generic; called once per event.
    wake: Box<dyn Fn() + Send>,
    /// Seats by their `wl_seat` global's name, so a removed seat can be found.
    seats: HashMap<u32, Seat>,
    /// Tablets announced and not yet removed, by object id.
    tablets: HashMap<ObjectId, ZwpTabletV2>,
    /// Tools announced and not yet removed, by object id, with what each is in the middle of.
    tools: HashMap<ObjectId, (ZwpTabletToolV2, tool::ToolState)>,
    /// Pads announced and not yet removed, by object id, with their controls.
    pads: HashMap<ObjectId, pad::Pad>,
}

impl State {
    /// Queues `event` for the consumer and wakes it; nothing happens once the consumer is gone.
    fn emit(&self, event: Event) {
        // A disconnected consumer just means nobody is listening any more.
        if self.sender.send(event).is_ok() {
            (self.wake)();
        }
    }

    /// Binds every `wl_seat` global the registry listed at startup (later ones arrive as
    /// registry events).
    fn bind_existing_seats(&mut self, globals: &GlobalList, qh: &QueueHandle<Self>) {
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

    /// Binds the `wl_seat` global `name` (at most at the version with a destructor) and asks for
    /// its tablet seat, whose tablets, tools and pads then arrive as events.
    fn add_seat(&mut self, registry: &WlRegistry, name: u32, version: u32, qh: &QueueHandle<Self>) {
        let seat: WlSeat = registry.bind(name, version.min(SEAT_RELEASE_VERSION), qh, ());
        let tablet = self.manager.get_tablet_seat(&seat, qh, ());
        self.seats.insert(name, Seat { seat, tablet });
    }

    /// Destroys the seat bound from global `name`, if we bound it: the global was removed.
    fn remove_seat(&mut self, name: u32) {
        if let Some(seat) = self.seats.remove(&name) {
            seat.destroy();
        }
    }

    /// Destroys every protocol object we created, children first.
    fn destroy_all(&mut self) {
        // Order among siblings doesn't matter; children go before their parents.
        self.pads.drain().for_each(|(_, pad)| pad.destroy());
        self.tools.drain().for_each(|(_, (tool, _))| tool.destroy());
        self.tablets
            .drain()
            .for_each(|(_, tablet)| tablet.destroy());
        self.seats.drain().for_each(|(_, seat)| seat.destroy());
        self.manager.destroy();
    }
}

/// The first `wl_seat` version with a destructor (`release`); older seats can only be abandoned.
const SEAT_RELEASE_VERSION: u32 = 5;

/// A `wl_seat` we bound and the tablet seat we got for it.
struct Seat {
    /// The bound seat.
    seat: WlSeat,
    /// Its tablet seat.
    tablet: ZwpTabletSeatV2,
}

impl Seat {
    /// Destroys the tablet seat, then releases the `wl_seat` if its version allows.
    fn destroy(self) {
        self.tablet.destroy();
        if self.seat.version() >= SEAT_RELEASE_VERSION {
            self.seat.release();
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("seats", &self.seats.len())
            .field("tablets", &self.tablets.len())
            .field("tools", &self.tools.len())
            .field("pads", &self.pads.len())
            .finish_non_exhaustive()
    }
}

/// The id events on `surface` carry: the address of its `wl_proxy`, the same address
/// [`surface_id`] reads from a window handle.
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
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } if interface == "wl_seat" => state.add_seat(proxy, name, version, qhandle),
            // Names are unique among live globals, so a name we don't hold isn't a seat of ours.
            wl_registry::Event::GlobalRemove { name } => state.remove_seat(name),
            _ => {}
        }
    }
}

/// Seats are bound only to ask for their tablet seat; their own events (`capabilities`, and `name`
/// from version 2; versions 3 to 5 add none) don't matter here.
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

/// Announces tablets, tools and pads, which are tracked so they can be destroyed later; each then
/// gets events through its own `Dispatch`.
impl Dispatch<ZwpTabletSeatV2, ()> for State {
    wayland_client::event_created_child!(Self, ZwpTabletSeatV2, [
        zwp_tablet_seat_v2::EVT_TABLET_ADDED_OPCODE => (ZwpTabletV2, ()),
        zwp_tablet_seat_v2::EVT_TOOL_ADDED_OPCODE => (ZwpTabletToolV2, ()),
        zwp_tablet_seat_v2::EVT_PAD_ADDED_OPCODE => (ZwpTabletPadV2, ()),
    ]);

    fn event(
        state: &mut Self,
        _: &ZwpTabletSeatV2,
        event: zwp_tablet_seat_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_tablet_seat_v2::Event::TabletAdded { id } => {
                state.tablets.insert(id.id(), id);
            }
            zwp_tablet_seat_v2::Event::ToolAdded { id } => {
                state
                    .tools
                    .insert(id.id(), (id, tool::ToolState::default()));
            }
            zwp_tablet_seat_v2::Event::PadAdded { id } => {
                state.pads.insert(id.id(), pad::Pad::new(id));
            }
            // The protocol enum is non-exhaustive.
            _ => {}
        }
    }
}

/// Tablet descriptions (name, USB ids) aren't needed yet; only removal matters.
impl Dispatch<ZwpTabletV2, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwpTabletV2,
        event: <ZwpTabletV2 as Proxy>::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if matches!(event, zwp_tablet_v2::Event::Removed) {
            state.tablets.remove(&proxy.id());
            proxy.destroy();
        }
    }
}
