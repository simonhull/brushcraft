//! The windowed app: winit's event loop on the main thread, which is the store's owner thread
//! and so the frame loop's.
//!
//! The loop's events become the frame loop's input (pointer motion, buttons, the wheel and
//! modifiers, as [`Input`]) and its window state (size, scale, exposure). After each batch of
//! events, [`WindowApp::drive`] asks the [`Owner`] what to do next: make a frame and hand it to
//! the render thread, wait until the next frame may start, wait for the render thread (which
//! wakes the loop when it has presented), or sleep until an event or the next upkeep. The pen
//! navigates through winit's tablet events; `input`'s own pen backend is the brush's, in M1a4.

use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use render::surface::{Size, Target, WindowSurface};
use store::Project;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{
    ButtonSource, ElementState, MouseButton, MouseScrollDelta, TabletToolButton, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Pixel;
use crate::cli::Options;
use crate::frame::schedule::Next;
use crate::frame::{NoBrush, NoUploads, Owner, Setup};
use crate::navigate::{Button, Input, Modifiers};
use crate::render_thread::{Hooks, RenderThread};
use crate::{Summary, describe_adapter};

/// The refresh interval assumed when the display doesn't report its rate: 60 Hz.
pub(crate) const DEFAULT_INTERVAL: Duration = Duration::from_micros(16_667);

/// Pixels of touchpad scrolling taken as one wheel notch.
const PIXELS_PER_NOTCH: f64 = 40.0;

/// The window, the render thread and the frame loop, once the window exists.
#[derive(Debug)]
struct Running {
    /// The frame loop, holding the project.
    owner: Owner<NoBrush, NoUploads>,
    /// The render thread. Declared before `window`, so it stops (releasing the surface) before
    /// the window is dropped.
    render: RenderThread,
    /// The window.
    window: Arc<dyn Window>,
    /// The GPU's description, for the summary.
    adapter: String,
}

/// winit's application: opens the window when it can, runs the frame loop, and reports how
/// the run ended through `done`.
#[derive(Debug)]
pub(crate) struct WindowApp {
    /// The command line.
    options: Options,
    /// The project, until the window opens and the frame loop takes it.
    project: Option<Project>,
    /// The running app, once the window exists.
    running: Option<Running>,
    /// Where the run's outcome goes when the loop exits.
    done: Sender<Result<Summary>>,
}

impl WindowApp {
    /// An app that opens `project` in a window as `options` say, reporting to `done`.
    pub(crate) const fn new(
        options: Options,
        project: Project,
        done: Sender<Result<Summary>>,
    ) -> Self {
        Self {
            options,
            project: Some(project),
            running: None,
            done,
        }
    }

    /// Opens the window, its surface and the render thread, and starts the frame loop.
    ///
    /// # Errors
    ///
    /// If the window, the surface, the GPU or the render thread can't be made.
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
        let render = RenderThread::spawn(gpu, Target::Window(surface), hooks)?;
        let frames = match self.options.mode {
            crate::cli::Mode::Window { frames } => frames,
            crate::cli::Mode::Offscreen { .. } => None,
        };
        let setup = Setup {
            size: frame_size(window.surface_size()),
            interval: refresh_interval(window.as_ref()),
            presentation: render.presentation(),
            adapter: adapter.clone(),
            frames,
        };
        let mut owner = Owner::new(project, NoBrush, NoUploads, setup, Instant::now());
        owner.set_scale(window.scale_factor());
        Ok(Running {
            owner,
            render,
            window,
            adapter,
        })
    }

    /// Runs the frame loop until it has to wait, and tells winit how long.
    ///
    /// # Errors
    ///
    /// If a frame, the upkeep or the render thread fails.
    fn drive(running: &mut Running, event_loop: &dyn ActiveEventLoop) -> Result<()> {
        loop {
            let now = Instant::now();
            match running.owner.next(now) {
                Next::Frame { deadline } => {
                    let frame = running.owner.frame(now, deadline)?;
                    running.render.send(frame)?;
                }
                Next::Wait { until } => {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(until));
                    return Ok(());
                }
                Next::Render => {
                    event_loop.set_control_flow(ControlFlow::Wait);
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

    /// Stops the render thread and closes the project, sends how the run went, and exits the
    /// loop. `result` is an error that ended the run early, if one did.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        let outcome = match self.running.take() {
            Some(running) => {
                let Running {
                    owner,
                    render,
                    window,
                    adapter,
                } = running;
                let presentation = render.presentation();
                let stopped = render.stop().map(drop);
                let frames = owner.finished();
                let closed = owner.close();
                drop(window);
                result.and(stopped).and(closed).map(|()| Summary {
                    frames,
                    adapter,
                    presentation,
                })
            }
            None => result.and_then(|()| Err(anyhow!("the window never opened"))),
        };
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(self.done.send(outcome));
        event_loop.exit();
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
            WindowEvent::Moved(_) => owner.set_interval(refresh_interval(running.window.as_ref())),
            WindowEvent::RedrawRequested | WindowEvent::Occluded(false) => owner.expose(),
            WindowEvent::ModifiersChanged(modifiers) => {
                let state = modifiers.state();
                owner.input(Input::Modifiers(Modifiers {
                    shift: state.shift_key(),
                    ctrl: state.control_key(),
                }));
            }
            WindowEvent::PointerMoved { position, .. } => owner.input(Input::Move {
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
            _ => {}
        }
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
        if self.running.is_some() {
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
