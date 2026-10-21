//! Pen-input probe: records what winit delivers from a pen tablet and summarises it.
//!
//! Draw on the window with the pen, then close it. Samples go to the CSV path given as the first
//! argument (default `pen-samples.csv`), and a summary is printed to stdout. Method and results:
//! `research/platform/pen-input-wayland.md` in the docs repo.

mod report;

use std::io::Write as _;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{
    ButtonSource, ElementState, PointerSource, StartCause, TabletToolButton, TabletToolData,
    WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowAttributes, WindowId};

const BACKGROUND: u32 = 0x0018_1818;
const INK: u32 = 0x00E8_E0D0;
const MAX_RADIUS: f64 = 12.0;

/// One pen sample as the application received it.
#[derive(Debug, Clone)]
struct Sample {
    /// When the event reached the application, since the probe started.
    received: Duration,
    /// Index of the event-loop wakeup that delivered this sample.
    wakeup: u64,
    x: f64,
    y: f64,
    contact: bool,
    data: TabletToolData,
}

/// What the probe draws: stamped discs whose radius follows pressure.
#[derive(Debug)]
struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![BACKGROUND; width * height],
        }
    }

    fn stamp(&mut self, x: f64, y: f64, radius: f64) {
        let r2 = radius * radius;
        let (x0, x1) = span(x, radius, self.width);
        let (y0, y1) = span(y, radius, self.height);
        for py in y0..y1 {
            let row = py * self.width;
            for px in x0..x1 {
                let (dx, dy) = (report::to_f64(px) - x, report::to_f64(py) - y);
                if dx * dx + dy * dy <= r2
                    && let Some(pixel) = self.pixels.get_mut(row + px)
                {
                    *pixel = INK;
                }
            }
        }
    }
}

/// The pixel range `[lo, hi)` covered by `centre ± radius`, clamped to `0..len`.
fn span(centre: f64, radius: f64, len: usize) -> (usize, usize) {
    let clamp = |v: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=len before the cast"
        )]
        let px = v.clamp(0.0, report::to_f64(len)) as usize;
        px
    };
    (
        clamp((centre - radius).floor()),
        clamp((centre + radius).ceil() + 1.0),
    )
}

#[derive(Debug)]
struct App {
    start: Instant,
    done: mpsc::Sender<Result<Vec<Sample>>>,
    surface: Option<Surface<OwnedDisplayHandle, Box<dyn Window>>>,
    canvas: Canvas,
    contact: bool,
    wakeup: u64,
    samples: Vec<Sample>,
}

impl App {
    fn new(done: mpsc::Sender<Result<Vec<Sample>>>) -> Self {
        Self {
            start: Instant::now(),
            done,
            surface: None,
            canvas: Canvas::new(0, 0),
            contact: false,
            wakeup: 0,
            samples: Vec::new(),
        }
    }

    /// Hands the recording (or the error that ended it) back to `main` and stops the loop.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        let outcome = result.map(|()| std::mem::take(&mut self.samples));
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(self.done.send(outcome));
        event_loop.exit();
    }

    fn record(&mut self, x: f64, y: f64, data: TabletToolData) {
        if self.contact {
            let pressure = data.force.map_or(1.0, |f| f.normalized(None));
            self.canvas.stamp(x, y, pressure.mul_add(MAX_RADIUS, 1.0));
        }
        self.samples.push(Sample {
            received: self.start.elapsed(),
            wakeup: self.wakeup,
            x,
            y,
            contact: self.contact,
            data,
        });
        self.request_redraw();
    }

    /// Copies the canvas to the window.
    ///
    /// # Errors
    ///
    /// If the window surface can't be resized, mapped or presented.
    fn redraw(&mut self) -> Result<()> {
        let Some(surface) = self.surface.as_mut() else {
            return Ok(());
        };
        let size = surface.window().surface_size();
        let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else {
            return Ok(());
        };
        let (width, height) = (usize::try_from(size.width)?, usize::try_from(size.height)?);
        if (self.canvas.width, self.canvas.height) != (width, height) {
            self.canvas = Canvas::new(width, height);
            surface
                .resize(w, h)
                .map_err(|e| anyhow!("resizing the surface: {e}"))?;
        }
        surface.window().pre_present_notify();
        let mut buffer = surface
            .buffer_mut()
            .map_err(|e| anyhow!("mapping the surface: {e}"))?;
        buffer.copy_from_slice(&self.canvas.pixels);
        buffer.present().map_err(|e| anyhow!("presenting: {e}"))
    }

    /// Records tablet events and keeps the window drawn.
    ///
    /// # Errors
    ///
    /// If redrawing fails.
    fn handle(&mut self, event_loop: &dyn ActiveEventLoop, event: WindowEvent) -> Result<()> {
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "WindowEvent is non-exhaustive and the probe ignores everything else"
        )]
        match event {
            WindowEvent::CloseRequested => self.finish(event_loop, Ok(())),
            WindowEvent::RedrawRequested => self.redraw()?,
            WindowEvent::SurfaceResized(_) => self.request_redraw(),
            WindowEvent::PointerMoved {
                position,
                source: PointerSource::TabletTool { data, .. },
                ..
            } => self.record(position.x, position.y, data),
            WindowEvent::PointerButton {
                state,
                position,
                button:
                    ButtonSource::TabletTool {
                        button: TabletToolButton::Contact,
                        data,
                        ..
                    },
                ..
            } => {
                self.contact = state == ElementState::Pressed;
                self.record(position.x, position.y, data);
            }
            _ => {}
        }
        Ok(())
    }

    fn request_redraw(&self) {
        if let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _: &dyn ActiveEventLoop, _: StartCause) {
        self.wakeup += 1;
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let result = (|| -> Result<()> {
            let attributes = WindowAttributes::default().with_title("pen-probe");
            let window = event_loop.create_window(attributes)?;
            let context = Context::new(event_loop.owned_display_handle())
                .map_err(|e| anyhow!("creating the softbuffer context: {e}"))?;
            let surface =
                Surface::new(&context, window).map_err(|e| anyhow!("creating the surface: {e}"))?;
            self.surface = Some(surface);
            Ok(())
        })();
        if let Err(e) = result {
            self.finish(event_loop, Err(e));
        }
    }

    fn window_event(&mut self, event_loop: &dyn ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Err(e) = self.handle(event_loop, event) {
            self.finish(event_loop, Err(e));
        }
    }
}

/// Runs the probe window, then writes the CSV and prints the summary.
///
/// # Errors
///
/// If the window can't be created or drawn, or the CSV or summary can't be written.
fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| "pen-samples.csv".into(), PathBuf::from);
    let (done, outcome) = mpsc::channel();
    EventLoop::new()?.run_app(App::new(done))?;
    let samples = outcome
        .recv()
        .context("the window closed without a recording")??;
    report::write_csv(&samples, &output)?;
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{}", report::summarise(&samples)?)?;
    writeln!(stdout, "csv: {}", output.display())?;
    Ok(())
}
