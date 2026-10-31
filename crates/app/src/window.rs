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
//! The pen navigates through winit's tablet events; `input`'s own pen backend is the brush's, in
//! M1a4.

use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use render::surface::{Size, Target, WindowSurface};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{
    ButtonSource, ElementState, MouseButton, MouseScrollDelta, TabletToolButton, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Pixel;
use crate::cli::Options;
use crate::frame::schedule::Next;
use crate::frame::{GPU_POLL, LookChange, Setup};
use crate::navigate::{Button, Input, Modifiers};
use crate::pace::{Pacer, Step, SystemTimer};
use crate::render_thread::{Hooks, RenderThread};
use crate::{AppOwner, ProjectToOpen, Summary, describe_adapter, start};

/// The refresh interval assumed when the display doesn't report its rate: 60 Hz.
pub(crate) const DEFAULT_INTERVAL: Duration = Duration::from_micros(16_667);

/// Pixels of touchpad scrolling taken as one wheel notch.
const PIXELS_PER_NOTCH: f64 = 40.0;

/// The window, the render thread and the frame loop, once the window exists.
#[derive(Debug)]
struct Running {
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
    done: Option<Sender<Result<Summary>>>,
}

impl WindowApp {
    /// An app that opens `project` in a window as `options` say, reporting to `done`.
    pub(crate) const fn new(
        options: Options,
        project: ProjectToOpen,
        done: Sender<Result<Summary>>,
    ) -> Self {
        Self {
            options,
            project: Some(project),
            running: None,
            done: Some(done),
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
        let window: Arc<dyn Window> = Arc::from(
            event_loop
                .create_window(attributes)
                .context("creating the window")?,
        );
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
            adapter: adapter.clone(),
            frames: self.options.frames(),
            script: self.options.script(),
            record: self.options.record_setup(render.timing(), opening),
        };
        let mut owner = start(project, gpu, setup)?;
        owner.set_scale(window.scale_factor());
        Ok(Running {
            owner,
            render,
            window,
            adapter,
            pacer: Pacer::default(),
            asked: None,
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
    /// thread's present paces the frames anyway; with mailbox, this wait does.
    ///
    /// # Errors
    ///
    /// If a frame, the upkeep or the render thread fails.
    fn drive(running: &mut Running, event_loop: &dyn ActiveEventLoop) -> Result<()> {
        running
            .owner
            .set_interval(refresh_interval(running.window.as_ref()));
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
                    let render = &mut running.render;
                    running
                        .owner
                        .frame(now, deadline, |frame| render.send(frame))?;
                }
                Next::Wait { until } => match running.pacer.plan(now, until) {
                    Step::Sleep(asked) => {
                        running.asked = Some((now, asked));
                        event_loop.set_control_flow(ControlFlow::WaitUntil(now + asked));
                        return Ok(());
                    }
                    Step::Spin | Step::Done => running.pacer.spin_until(&mut SystemTimer, until),
                },
                Next::Render => {
                    event_loop.set_control_flow(ControlFlow::Wait);
                    return Ok(());
                }
                Next::Gpu => {
                    running.owner.gpu_finished(running.render.gpu_finished()?);
                    if !matches!(running.owner.next(now), Next::Gpu) {
                        continue;
                    }
                    // Still busy: a slice of jobs, then back to the event loop for its events;
                    // with none, poll again shortly.
                    let worked = running.owner.while_gpu_busy(now, now + GPU_POLL)?;
                    let flow = if worked {
                        ControlFlow::Poll
                    } else {
                        ControlFlow::WaitUntil(now + GPU_POLL)
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

    /// Exits the loop, stops the render thread and closes the project, and sends how the run
    /// went; `result` is an error that ended the run early, if one did. Only the first call does
    /// more than exit.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        event_loop.exit();
        let Some(done) = self.done.take() else {
            // Finished already (an error, then the close the exit brings): nothing more to say.
            return;
        };
        let outcome = match self.running.take() {
            Some(running) => {
                let Running {
                    owner,
                    render,
                    window,
                    adapter,
                    ..
                } = running;
                let presentation = render.presentation();
                let stopped = render.stop().map(drop);
                let summary = Summary::of(&owner, adapter, presentation);
                let closed = owner.close();
                drop(window);
                result.and(stopped).and(closed).map(|()| summary)
            }
            None => result.and_then(|()| Err(anyhow!("the window never opened"))),
        };
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(done.send(outcome));
    }

    /// Turns a window event into frame-loop input or window state.
    fn handle(&mut self, event_loop: &dyn ActiveEventLoop, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            self.finish(event_loop, Ok(()));
            return;
        }
        let Some(running) = &mut self.running else {
            return;
        };
        let owner = &mut running.owner;
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "WindowEvent is non-exhaustive, and the app reads only these"
        )]
        match event {
            WindowEvent::SurfaceResized(size) => owner.resize(frame_size(size)),
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => owner.set_scale(scale_factor),
            WindowEvent::RedrawRequested | WindowEvent::Occluded(false) => owner.expose(),
            // A button held as focus goes may never report its release.
            WindowEvent::Focused(false) => owner.input(Input::Cancel),
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                owner.input(Input::Modifiers(Modifiers {
                    shift: state.shift_key(),
                    ctrl: state.control_key(),
                }));
            }
            WindowEvent::PointerLeft { .. } => owner.pointer_left(),
            // Physical pixels, as the frame size: the mouse's, and the pen's while it hovers.
            WindowEvent::PointerMoved { position, .. }
            | WindowEvent::PointerEntered { position, .. } => owner.input(Input::Move {
                at: Pixel {
                    x: position.x,
                    y: position.y,
                },
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
                            at: Pixel {
                                x: position.x,
                                y: position.y,
                            },
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
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && let Some(change) = look_change(event.physical_key)
                {
                    owner.change_look(change);
                }
            }
            _ => {}
        }
    }
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

/// The button navigation reads for a winit button, if it reads that one.
fn navigation_button(button: &ButtonSource) -> Option<Button> {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "ButtonSource and MouseButton are non-exhaustive; other buttons don't navigate"
    )]
    match button {
        ButtonSource::Mouse(MouseButton::Left) => Some(Button::MouseLeft),
        ButtonSource::Mouse(MouseButton::Middle) => Some(Button::MouseMiddle),
        ButtonSource::Mouse(MouseButton::Right) => Some(Button::MouseRight),
        ButtonSource::TabletTool {
            button: TabletToolButton::Contact,
            ..
        } => Some(Button::PenTip),
        ButtonSource::TabletTool {
            button: TabletToolButton::Barrel,
            ..
        } => Some(Button::PenBarrel),
        _ => None,
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
        let driven = Self::drive(running, event_loop);
        if driven.is_err() || running.owner.done() {
            self.finish(event_loop, driven);
        }
    }
}
