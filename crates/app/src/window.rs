//! The windowed app: winit's event loop on the main thread, which is the store's owner thread
//! and so the frame loop's.
//!
//! The loop's events become the frame loop's input (pointer motion, buttons, the wheel and
//! modifiers, as [`Input`]) and its window state (size, scale, exposure). After each batch of
//! events, [`WindowApp::drive`] asks the [`Owner`](crate::frame::Owner) what to do next: make a
//! frame and hand it to the render thread, wait until the next frame may start, wait for the render
//! thread (which wakes the loop when it has presented), wait for the GPU to finish a frame
//! (polling it every [`GPU_POLL`], running the store's jobs meanwhile), or sleep until an event or
//! the next upkeep.
//!
//! **The pen** comes through one stream ([`crate::pen`]): our Wayland backend, spawned once the
//! window exists (undecorated, and configured: winit returns a window only after its first
//! configure), or winit's tablet events where it can't run. The stream is drained into the frame
//! loop after each batch of events and again just before each frame. It holds the window and is
//! stopped first on the way out ([`WindowApp::finish`]), so our backend never outlives the
//! window's surface. The mouse's events stay winit's.
//!
//! **Order between the two:** within the pen stream, and within the mouse's input, everything
//! keeps the order it happened in. Across them it doesn't: a frame's input phase applies all the
//! mouse input queued since the last frame first, then routes the pen samples drained after each
//! batch of events (and just before the frame), so a mouse event and a pen sample of the same
//! batch reach navigation mouse first, whichever came first. With one hand on the pen that only
//! matters if both drag at once, which navigation doesn't support anyway (a drag keeps the button
//! that started it).

use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use render::surface::{Size, Target, WindowSurface};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{
    ButtonSource, ElementState, MouseButton, MouseScrollDelta, PointerKind, PointerSource,
    StartCause, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Pixel;
use crate::cli::Options;
use crate::frame::schedule::Next;
use crate::frame::{BrushChange, GPU_POLL, GpuWait, LookChange, Setup};
use crate::navigate::{Button, Input, Modifiers};
use crate::pace::{Pacer, Step, SystemTimer};
use crate::pen::{AppPen, PenSample, ToolData, WinitEvent};
use crate::render_thread::{Hooks, RenderThread};
use crate::saving::SaveStep;
use crate::{AppOwner, Ended, ProjectToOpen, Summary, describe_adapter, said_too, start};

/// The refresh interval assumed when the display doesn't report its rate: 60 Hz.
pub(crate) const DEFAULT_INTERVAL: Duration = Duration::from_micros(16_667);

/// Pixels of touchpad scrolling taken as one wheel notch.
const PIXELS_PER_NOTCH: f64 = 40.0;

/// The window, the render thread and the frame loop, once the window exists.
#[derive(Debug)]
struct Running {
    /// The pen stream. Declared first, so on any path that drops `Running` whole the backend
    /// stops before anything else goes; it also holds the window itself, and
    /// [`WindowApp::finish`] stops it before anything else.
    pen: AppPen,
    /// The frame loop, holding the project.
    owner: AppOwner,
    /// The render thread. Declared before `window`, so it stops (releasing the surface) before
    /// the window is dropped.
    render: RenderThread,
    /// The window.
    window: Arc<dyn Window>,
    /// The GPU's description, for the summary.
    adapter: String,
    /// Waits for the next frame precisely: the event loop's timer wakes the loop just before
    /// it, and the rest is spun ([`crate::pace`]).
    pacer: Pacer,
    /// The event loop's timer last asked for, to learn how long it ran: from when, for how long.
    asked: Option<(Instant, Duration)>,
    /// Whether the window was asked to close, the brush landing what it holds.
    closing: bool,
    /// The keyboard's modifiers as winit last reported them: what Ctrl+S reads.
    modifiers: ModifiersState,
}

/// winit's application: opens the window when it can, runs the frame loop, and reports how
/// the run ended through `done`.
#[derive(Debug)]
pub(crate) struct WindowApp {
    /// The command line.
    options: Options,
    /// The project to open, until the window, its surface and the render thread are up and
    /// the frame loop opens it.
    project: Option<ProjectToOpen>,
    /// The running app, once the window exists.
    running: Option<Running>,
    /// Where the run's outcome goes when the loop exits.
    /// Where the run's outcome goes when the loop exits; `None` once it has been sent, so a
    /// run finishes once, whatever events follow.
    done: Option<Sender<Ended>>,
    /// Event-loop wakeups so far: winit's fallback pen events of one wakeup may be one pen frame.
    wakeup: u64,
    /// Pen samples drained from the stream, handed on to the frame loop at once; kept for its
    /// allocation.
    pen_samples: Vec<PenSample>,
}

impl WindowApp {
    /// An app that opens `project` in a window as `options` say, reporting to `done`.
    pub(crate) const fn new(options: Options, project: ProjectToOpen, done: Sender<Ended>) -> Self {
        Self {
            options,
            project: Some(project),
            running: None,
            done: Some(done),
            wakeup: 0,
            pen_samples: Vec::new(),
        }
    }

    /// Opens the window, its surface and the render thread, then the project, and starts the
    /// frame loop. The project comes last, so a launch that fails before it leaves no recovery
    /// file.
    ///
    /// # Errors
    ///
    /// If the window, the surface, the GPU or the render thread can't be made, or the project
    /// doesn't open.
    fn open(&mut self, event_loop: &dyn ActiveEventLoop) -> Result<Running> {
        let project = self
            .project
            .take()
            .ok_or_else(|| anyhow!("the project was already taken"))?;
        let requested = PhysicalSize::new(self.options.size.width(), self.options.size.height());
        let attributes = WindowAttributes::default()
            .with_title("BrushCraft")
            .with_surface_size(requested);
        // The pen stream creates the window (undecorated) and starts our backend on it, so it
        // can hold the window for as long as the backend runs ([`crate::pen`]).
        let pen = AppPen::create_window(event_loop, attributes)?;
        let window = Arc::clone(pen.window());
        let size = frame_size(window.surface_size()).unwrap_or(self.options.size);
        let (gpu, surface) = WindowSurface::open(Arc::clone(&window), size, &self.options.adapter)
            .context("opening the GPU for the window")?;
        let adapter = describe_adapter(gpu.adapter_info());
        let proxy = event_loop.create_proxy();
        let presenting = Arc::clone(&window);
        let hooks = Hooks::new(
            move || proxy.wake_up(),
            move || presenting.pre_present_notify(),
        );
        let timing = self.options.times_gpu();
        let render = RenderThread::spawn(Arc::clone(&gpu), Target::Window(surface), hooks, timing)?;
        let opening = Instant::now();
        let project = project.open()?;
        let setup = Setup {
            size: frame_size(window.surface_size()),
            interval: refresh_interval(window.as_ref()),
            presentation: render.presentation(),
            blend: render.blend(),
            adapter: adapter.clone(),
            frames: self.options.frames(),
            script: self.options.script(),
            record: self.options.record_setup(render.timing(), opening),
        };
        let mut owner = start(project, gpu, setup)?;
        owner.set_scale(window.scale_factor());
        owner.set_pen_status(pen.status());
        Ok(Running {
            pen,
            owner,
            render,
            window,
            adapter,
            pacer: Pacer::default(),
            asked: None,
            closing: false,
            modifiers: ModifiersState::default(),
        })
    }

    /// Runs the frame loop until it has to wait, and tells winit how long.
    ///
    /// The display's refresh rate is read again first: the window may have moved to another
    /// display, which winit doesn't report on Wayland (no `Moved` event there).
    ///
    /// The wait for the next frame is precise ([`crate::pace`]): winit's timer is asked to wake
    /// the loop as the pacer plans (short of the frame by the spin margin, and by how long its
    /// waits have run), and the last of it is spun here. With FIFO presentation the render
    /// thread's present paces the frames anyway; with mailbox, this wait does, but for a brush
    /// stroke's frames, which start on the pen's input (`frame::schedule::StrokePacing`).
    ///
    /// **The pen** is read late: the frame drains the stream itself as it starts, before its
    /// brush phase, and between its work after the hand-off ([`crate::frame::Owner::frame_fed`]).
    /// Wherever the loop would wait for the render thread, the GPU or the refresh slot, a brush
    /// stroke's work between frames comes first ([`crate::frame::Owner::between_frames`]): one
    /// slice, then back to the event loop (so the render thread's wake-up is heard), or, waiting
    /// for the slot, until the slot.
    ///
    /// # Errors
    ///
    /// If a frame, the brush, the upkeep or the render thread fails.
    fn drive(
        running: &mut Running,
        pen_samples: &mut Vec<PenSample>,
        event_loop: &dyn ActiveEventLoop,
    ) -> Result<()> {
        running
            .owner
            .set_interval(refresh_interval(running.window.as_ref()));
        Self::feed_pen(running, pen_samples);
        if let Some((from, asked)) = running.asked.take() {
            // Woken by the timer, or by an event after it would have fired: how long it ran.
            // An event before it says nothing.
            let took = Instant::now().saturating_duration_since(from);
            if took >= asked {
                running.pacer.slept(asked, took);
            }
        }
        loop {
            let now = Instant::now();
            match running.owner.next(now) {
                Next::Frame { deadline } => {
                    // The frame drains what the pen sent since the last batch of events itself.
                    let Running {
                        pen, owner, render, ..
                    } = &mut *running;
                    owner.frame_fed(now, deadline, pen, |frame| render.send(frame))?;
                    owner.set_pen_status(pen.status());
                }
                Next::Wait { .. } if running.owner.between_frames(now, &mut running.pen)? => {}
                Next::Wait { until } => match running.pacer.plan(now, until) {
                    Step::Sleep(asked) => {
                        running.asked = Some((now, asked));
                        event_loop.set_control_flow(ControlFlow::WaitUntil(now + asked));
                        return Ok(());
                    }
                    Step::Spin | Step::Done => running.pacer.spin_until(&mut SystemTimer, until),
                },
                Next::Render => {
                    // A slice of the stroke's work, then the event loop's events (the render
                    // thread's wake-up among them) before the next; with none, wait for them.
                    let worked = running.owner.between_frames(now, &mut running.pen)?;
                    event_loop.set_control_flow(if worked {
                        ControlFlow::Poll
                    } else {
                        ControlFlow::Wait
                    });
                    return Ok(());
                }
                Next::Gpu => {
                    let Running {
                        pen, owner, render, ..
                    } = &mut *running;
                    // Still busy after a slice of the stroke's work or the jobs: back to the
                    // event loop for its events; with none, poll again shortly (`GPU_POLL` says
                    // why it polls).
                    let flow = match owner.wait_for_gpu(now, pen, || render.gpu_finished())? {
                        GpuWait::Ready => continue,
                        GpuWait::Worked => ControlFlow::Poll,
                        GpuWait::Idle => ControlFlow::WaitUntil(now + GPU_POLL),
                    };
                    event_loop.set_control_flow(flow);
                    return Ok(());
                }
                Next::Idle { upkeep } if now < upkeep => {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(upkeep));
                    return Ok(());
                }
                Next::Idle { .. } => running.owner.upkeep(now)?,
            }
        }
    }

    /// Drains the pen stream into the frame loop, through `samples` (left empty), and tells the
    /// overlay where the stream comes from (it falls back if our backend stopped).
    fn feed_pen(running: &mut Running, samples: &mut Vec<PenSample>) {
        running.pen.drain(samples);
        let drained = samples.drain(..);
        for sample in drained {
            running.owner.pen(sample);
        }
        running.owner.set_pen_status(running.pen.status());
    }

    /// Exits the loop, stops the pen stream (first: our backend must not outlive the window's
    /// surface), then the render thread, drops the window, and sends how the run went, with the
    /// frame loop for `run_window` to close; `result` is an error that ended the run early, if
    /// one did. Only the first call does more than exit.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        event_loop.exit();
        self.wind_up(result);
    }

    /// [`WindowApp::finish`] but for ending the loop: for a loop already over too
    /// ([`WindowApp`]'s `drop`), so a run that ended in an error still hands its frame loop back,
    /// to land what it holds and close the project.
    fn wind_up(&mut self, result: Result<()>) {
        let Some(done) = self.done.take() else {
            // Finished already (an error, then the close the exit brings): nothing more to say.
            return;
        };
        let outcome = match self.running.take() {
            Some(running) => {
                let Running {
                    pen,
                    mut owner,
                    render,
                    window,
                    adapter,
                    ..
                } = running;
                // The pen first: our backend must stop before the window's surface can go. What
                // it sent since the last frame is dropped with the run.
                let (pen_window, pen_stopped) = pen.stop(&mut Vec::new());
                drop(pen_window);
                let presentation = render.presentation();
                let stopped = render
                    .stop()
                    .map(|stopped| owner.last_gpu_times(stopped.gpu));
                let summary = Summary::of(&owner, adapter, presentation);
                // The window goes first: the project closes once the event loop has ended
                // (`run_window`), however long that takes, with no window left unanswered.
                drop(window);
                Ended {
                    run: said_too(said_too(result, stopped), pen_stopped).map(|()| summary),
                    owner: Some(owner),
                }
            }
            None => Ended {
                run: result.and_then(|()| Err(anyhow!("the window never opened"))),
                owner: None,
            },
        };
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(done.send(outcome));
    }

    /// The window was asked to close. The first time, the brush starts landing what it holds:
    /// the loop runs on, frames and events, the title counting the dabs down, and ends as they
    /// run out. Asked again, the window goes at once and the landing ends after it, unseen.
    fn close_requested(&mut self, event_loop: &dyn ActiveEventLoop) {
        let Some(running) = &mut self.running else {
            self.finish(event_loop, Ok(()));
            return;
        };
        if running.closing {
            self.finish(event_loop, Ok(()));
            return;
        }
        running.closing = true;
        match running.owner.request_close() {
            Ok(()) if running.owner.ready_to_close() => self.finish(event_loop, Ok(())),
            Ok(()) => {}
            Err(error) => self.finish(event_loop, Err(error)),
        }
    }

    /// Turns a window event into frame-loop input or window state.
    fn handle(&mut self, event_loop: &dyn ActiveEventLoop, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            self.close_requested(event_loop);
            return;
        }
        let Some(running) = &mut self.running else {
            return;
        };
        if let Some((tablet, at, data)) = tablet_event(&event) {
            // The pen's events go to its stream, which follows them on the fallback.
            let stamp = (Instant::now(), self.wakeup);
            running.pen.winit(tablet, at, data, stamp);
            return;
        }
        let owner = &mut running.owner;
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "WindowEvent is non-exhaustive, and the app reads only these"
        )]
        match event {
            WindowEvent::SurfaceResized(size) => owner.resize(frame_size(size)),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                owner.set_scale(scale_factor);
                running.pen.set_scale(scale_factor);
            }
            WindowEvent::RedrawRequested | WindowEvent::Occluded(false) => owner.expose(),
            // A button held as focus goes may never report its release.
            WindowEvent::Focused(false) => owner.input(Input::Cancel),
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                running.modifiers = state;
                owner.input(Input::Modifiers(Modifiers {
                    shift: state.shift_key(),
                    ctrl: state.control_key(),
                }));
            }
            WindowEvent::PointerLeft { .. } => owner.pointer_left(),
            // Physical pixels, as the frame size.
            WindowEvent::PointerMoved { position, .. }
            | WindowEvent::PointerEntered { position, .. } => owner.input(Input::Move {
                at: pixel(position),
            }),
            WindowEvent::PointerButton {
                state,
                position,
                button,
                ..
            } => {
                if let Some(button) = navigation_button(&button) {
                    owner.input(match state {
                        ElementState::Pressed => Input::Press {
                            button,
                            at: pixel(position),
                        },
                        ElementState::Released => Input::Release { button },
                    });
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(notches) = notches(&delta) {
                    owner.input(Input::Wheel { notches });
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return;
                }
                if saves(event.physical_key, running.modifiers, event.repeat) {
                    owner.request_save(Instant::now());
                } else if let Some(change) = brush_change(event.physical_key) {
                    // Held, a brush key repeats: the size or strength keeps changing.
                    owner.change_brush(change);
                } else if !event.repeat {
                    if let Some(change) = look_change(event.physical_key) {
                        owner.change_look(change);
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::KeyF) {
                        // F frames the model, its centre the pivot (by the key's place, as the
                        // look's keys are).
                        owner.frame_model();
                    }
                }
            }
            _ => {}
        }
    }
}

/// Whether `key` with `modifiers` held is Ctrl+S: Ctrl alone, S by its place on the keyboard, as
/// the other keys are, and not a held key's `repeat`. With Shift it stays free for Save As.
fn saves(key: PhysicalKey, modifiers: ModifiersState, repeat: bool) -> bool {
    !repeat && key == PhysicalKey::Code(KeyCode::KeyS) && modifiers == ModifiersState::CONTROL
}

/// The change of look a key makes, if it makes one: by the key's place on the keyboard, so
/// every layout has them where a QWERTY one does. M the next MatCap, W the wireframe, C the
/// debug view of clusters.
fn look_change(key: PhysicalKey) -> Option<LookChange> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "KeyCode is non-exhaustive, and only these keys change the look"
    )]
    match key {
        PhysicalKey::Code(KeyCode::KeyM) => Some(LookChange::MatCap),
        PhysicalKey::Code(KeyCode::KeyW) => Some(LookChange::Wireframe),
        PhysicalKey::Code(KeyCode::KeyC) => Some(LookChange::Clusters),
        _ => None,
    }
}

/// The change to the brush a key makes, if it makes one.
///
/// By the key's place on the keyboard, as [`look_change`]. The bracket keys make the brush smaller
/// (`[`) and larger (`]`); the two keys left of Backspace make it weaker (`-`) and stronger (`=`).
fn brush_change(key: PhysicalKey) -> Option<BrushChange> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "KeyCode is non-exhaustive, and only these keys change the brush"
    )]
    match key {
        PhysicalKey::Code(KeyCode::BracketLeft) => Some(BrushChange::Smaller),
        PhysicalKey::Code(KeyCode::BracketRight) => Some(BrushChange::Larger),
        PhysicalKey::Code(KeyCode::Minus) => Some(BrushChange::Weaker),
        PhysicalKey::Code(KeyCode::Equal) => Some(BrushChange::Stronger),
        _ => None,
    }
}

/// The button navigation reads for a winit button, if it reads that one: the mouse's (the
/// pen's come through its stream, [`crate::pen`]).
fn navigation_button(button: &ButtonSource) -> Option<Button> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "ButtonSource and MouseButton are non-exhaustive; other buttons don't navigate"
    )]
    match button {
        ButtonSource::Mouse(MouseButton::Left) => Some(Button::MouseLeft),
        ButtonSource::Mouse(MouseButton::Middle) => Some(Button::MouseMiddle),
        ButtonSource::Mouse(MouseButton::Right) => Some(Button::MouseRight),
        _ => None,
    }
}

/// One of winit's tablet tool events, in the pen stream's terms: the event, where (if winit
/// says), and the tool's data (if it comes with it).
type TabletEvent = (WinitEvent, Option<Pixel>, Option<ToolData>);

/// `event` as a tablet tool's event, if it is one.
fn tablet_event(event: &WindowEvent) -> Option<TabletEvent> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "WindowEvent is non-exhaustive, and only these are a tablet tool's"
    )]
    match event {
        WindowEvent::PointerEntered {
            position,
            kind: PointerKind::TabletTool(_),
            ..
        } => Some((WinitEvent::Entered, Some(pixel(*position)), None)),
        WindowEvent::PointerLeft {
            position,
            kind: PointerKind::TabletTool(_),
            ..
        } => Some((WinitEvent::Left, position.map(pixel), None)),
        WindowEvent::PointerMoved {
            position,
            source: PointerSource::TabletTool { kind, data },
            ..
        } => Some((
            WinitEvent::Moved,
            Some(pixel(*position)),
            Some(ToolData::from_winit(*kind, data)),
        )),
        WindowEvent::PointerButton {
            state,
            position,
            button: ButtonSource::TabletTool { kind, button, data },
            ..
        } => Some((
            WinitEvent::Button {
                button: (*button).into(),
                pressed: *state == ElementState::Pressed,
            },
            Some(pixel(*position)),
            Some(ToolData::from_winit(*kind, data)),
        )),
        _ => None,
    }
}

/// winit's pointer position as a viewport pixel: both physical.
const fn pixel(position: PhysicalPosition<f64>) -> Pixel {
    Pixel {
        x: position.x,
        y: position.y,
    }
}

/// How many wheel notches a scroll is (positive away from the user), if navigation reads
/// it: lines are notches, and [`PIXELS_PER_NOTCH`] of a touchpad's pixels make one.
fn notches(delta: &MouseScrollDelta) -> Option<f64> {
    match delta {
        MouseScrollDelta::LineDelta(_, lines) => Some(f64::from(*lines)),
        MouseScrollDelta::PixelDelta(pixels) => Some(pixels.y / PIXELS_PER_NOTCH),
        // Non-exhaustive: a kind added later isn't read.
        _ => None,
    }
}

/// A window surface's size as a frame size; `None` if it has none (minimized).
fn frame_size(size: PhysicalSize<u32>) -> Option<Size> {
    Size::new(size.width, size.height)
}

/// The refresh interval of the display `window` is on, or [`DEFAULT_INTERVAL`] if it doesn't
/// say.
fn refresh_interval(window: &dyn Window) -> Duration {
    window
        .current_monitor()
        .and_then(|monitor| monitor.current_video_mode())
        .and_then(|mode| mode.refresh_rate_millihertz())
        .map_or(DEFAULT_INTERVAL, |millihertz| {
            Duration::from_secs_f64(1000.0 / f64::from(millihertz.get()))
        })
}

impl ApplicationHandler for WindowApp {
    fn new_events(&mut self, _: &dyn ActiveEventLoop, _: StartCause) {
        self.wakeup += 1;
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.running.is_some() || self.done.is_none() {
            return;
        }
        match self.open(event_loop) {
            Ok(running) => self.running = Some(running),
            Err(e) => self.finish(event_loop, Err(e)),
        }
    }

    fn proxy_wake_up(&mut self, event_loop: &dyn ActiveEventLoop) {
        let Some(running) = &mut self.running else {
            return;
        };
        match running.render.finished() {
            Ok(presented) => presented
                .into_iter()
                .for_each(|frame| running.owner.shown(frame)),
            Err(e) => self.finish(event_loop, Err(e)),
        }
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, _: WindowId, event: WindowEvent) {
        self.handle(event_loop, event);
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        let Some(running) = &mut self.running else {
            return;
        };
        let stepped = if running.closing {
            running.owner.close_step().map(|_| ())
        } else {
            Ok(())
        };
        // Samples the pen sent before Ctrl+S are in the save: they are looked at first.
        Self::feed_pen(running, &mut self.pen_samples);
        let saving = running.owner.save_step();
        let driven = stepped.and_then(|()| Self::drive(running, &mut self.pen_samples, event_loop));
        // A close held by a failed save has nothing to do but wait for the user.
        let closing = running.closing && !running.owner.close_held();
        match saving {
            // Nothing to do but look again shortly: the recovery writer wakes nothing.
            SaveStep::Waiting if !closing => {
                event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + GPU_POLL));
            }
            SaveStep::Idle if !closing => {}
            // Back at once, frames or none, the window's events heard between batches.
            SaveStep::Waiting | SaveStep::Idle | SaveStep::Worked => {
                event_loop.set_control_flow(ControlFlow::Poll);
            }
        }
        if let Some(title) = running.owner.title_change() {
            running.window.set_title(&title);
        }
        if driven.is_err() || running.owner.done() || running.owner.ready_to_close() {
            self.finish(event_loop, driven);
        }
    }
}

impl Drop for WindowApp {
    /// A loop that ended on its own failure never finished: its frame loop still goes back.
    fn drop(&mut self) {
        // Unwinding, a second panic here would abort, and `Project`'s drop flushes recovery.
        if !std::thread::panicking() {
            self.wind_up(Ok(()));
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// The S key.
    const S: PhysicalKey = PhysicalKey::Code(KeyCode::KeyS);

    #[test]
    fn ctrl_s_saves_and_nothing_else_does() {
        assert!(saves(S, ModifiersState::CONTROL, false));
        assert!(!saves(S, ModifiersState::CONTROL, true), "held down");
        assert!(!saves(S, ModifiersState::default(), false), "S alone");
        let other = PhysicalKey::Code(KeyCode::KeyA);
        assert!(!saves(other, ModifiersState::CONTROL, false), "Ctrl+A");
        let held = ModifiersState::CONTROL | ModifiersState::SHIFT;
        assert!(!saves(S, held, false), "Ctrl+Shift+S stays free");
    }
}
