//! Pen-input probe: records what winit and our own pen backend deliver from a pen tablet, from the
//! same strokes, and summarises both.
//!
//! Usage: `pen-probe [--source winit|ours|both] [CSV]`.
//!
//! Draw on the window with the pen, then close it. Samples go to the CSV path (default
//! `pen-samples.csv`), and a summary is printed to stdout. `--source` picks what is recorded:
//! our backend (`input::wayland`, the default), winit, or both. Some compositors deliver tablet
//! events to only one tablet binding per client, so `both` may see nothing from one of them.
//! Method and results: `research/platform/pen-input-wayland.md` in the docs repo.

mod ours;
mod report;

use std::ffi::OsString;
use std::io::Write as _;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use input::{Buttons, Phase};
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

/// Which path a sample came through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// winit's `PointerMoved` / `PointerButton`, on the event-loop thread.
    Winit,
    /// `input`'s Wayland backend, on its own thread.
    Ours,
}

impl Source {
    const ALL: [Self; 2] = [Self::Winit, Self::Ours];

    const fn name(self) -> &'static str {
        match self {
            Self::Winit => "winit",
            Self::Ours => "ours",
        }
    }
}

/// Which sources a run records, from `--source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Selection {
    Winit,
    #[default]
    Ours,
    Both,
}

impl Selection {
    /// Whether this run records `source`.
    const fn records(self, source: Source) -> bool {
        matches!(
            (self, source),
            (Self::Both, _) | (Self::Winit, Source::Winit) | (Self::Ours, Source::Ours)
        )
    }
}

/// The command line.
#[derive(Debug, PartialEq, Eq)]
struct Args {
    selection: Selection,
    output: PathBuf,
}

impl Args {
    /// Parses the arguments after the program name.
    ///
    /// # Errors
    ///
    /// On an unknown flag or source, a missing `--source` value, or more than one path.
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut selection = Selection::default();
        let mut output = None;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let source = if arg == "--source" {
                Some(
                    args.next()
                        .context("--source needs a value: winit, ours or both")?,
                )
            } else {
                arg.to_str()
                    .and_then(|a| a.strip_prefix("--source="))
                    .map(OsString::from)
            };
            if let Some(source) = source {
                selection = match source.to_str() {
                    Some("winit") => Selection::Winit,
                    Some("ours") => Selection::Ours,
                    Some("both") => Selection::Both,
                    _ => bail!(
                        "unknown source {}: expected winit, ours or both",
                        source.display()
                    ),
                };
            } else if arg.to_str().is_some_and(|a| a.starts_with('-')) {
                bail!(
                    "unknown option {}; usage: pen-probe [--source winit|ours|both] [CSV]",
                    arg.display()
                );
            } else if output.replace(PathBuf::from(arg)).is_some() {
                bail!(
                    "more than one output path; usage: pen-probe [--source winit|ours|both] [CSV]"
                );
            }
        }
        Ok(Self {
            selection,
            output: output.unwrap_or_else(|| "pen-samples.csv".into()),
        })
    }
}

/// One pen sample as the probe records it.
#[derive(Debug, Clone)]
struct Sample {
    source: Source,
    /// When it was received (winit: event-loop thread; ours: input thread), since the probe
    /// started.
    received: Duration,
    /// Our de-jittered time, since the probe started; `None` for winit samples and hover.
    dejittered: Option<Duration>,
    /// Index of the event-loop wakeup that delivered this sample.
    wakeup: u64,
    /// Position in physical pixels.
    x: f64,
    y: f64,
    contact: bool,
    pressure: Option<f64>,
    /// Tilt in degrees.
    tilt: Option<(f64, f64)>,
    distance: Option<f64>,
    roll: Option<f64>,
    /// Our phase; `None` for winit samples.
    phase: Option<Phase>,
    /// Barrel buttons held; never the tip.
    buttons: Buttons,
    /// The compositor's millisecond timestamp; `None` for winit samples.
    device_ms: Option<u32>,
}

/// Everything one run recorded.
#[derive(Debug, Default)]
struct Recording {
    /// Which sources were recorded.
    selection: Selection,
    samples: Vec<Sample>,
    /// Pad events from our backend (buttons, rings, strips, dials).
    pads: usize,
    /// Our samples over some other surface, left out.
    elsewhere: usize,
    /// Events our backend's queue dropped.
    dropped: u64,
    /// Why our backend didn't run or stopped badly, if it did.
    note: Option<String>,
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
    done: mpsc::Sender<Result<Recording>>,
    surface: Option<Surface<OwnedDisplayHandle, Box<dyn Window>>>,
    canvas: Canvas,
    /// winit's view of whether the tip is down.
    contact: bool,
    /// winit's view of which barrel buttons are held.
    buttons: Buttons,
    wakeup: u64,
    /// The window's scale factor, to put our logical positions in winit's physical pixels.
    scale: f64,
    backend: Option<ours::Ours>,
    recording: Recording,
}

impl App {
    fn new(selection: Selection, done: mpsc::Sender<Result<Recording>>) -> Self {
        Self {
            start: Instant::now(),
            done,
            surface: None,
            canvas: Canvas::new(0, 0),
            contact: false,
            buttons: Buttons::default(),
            wakeup: 0,
            scale: 1.0,
            backend: None,
            recording: Recording {
                selection,
                ..Recording::default()
            },
        }
    }

    /// Stops our backend, hands the recording (or the error that ended it) back to `main` and
    /// stops the loop.
    ///
    /// The backend stops here, before the loop exits and the window is destroyed, as its docs
    /// require.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        self.drain();
        if let Some(backend) = self.backend.take() {
            self.recording.dropped = backend.dropped();
            if let Err(e) = backend.stop() {
                self.recording.note = Some(format!("our backend stopped with an error: {e}"));
            }
        }
        let outcome = result.map(|()| std::mem::take(&mut self.recording));
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(self.done.send(outcome));
        event_loop.exit();
    }

    /// Which source the canvas draws: ours when it runs, winit's otherwise.
    const fn drawn(&self) -> Source {
        if self.backend.is_some() {
            Source::Ours
        } else {
            Source::Winit
        }
    }

    fn record(&mut self, sample: Sample) {
        if !self.recording.selection.records(sample.source) {
            return;
        }
        if sample.contact && sample.source == self.drawn() {
            let pressure = sample.pressure.unwrap_or(1.0);
            self.canvas
                .stamp(sample.x, sample.y, pressure.mul_add(MAX_RADIUS, 1.0));
            self.request_redraw();
        }
        self.recording.samples.push(sample);
    }

    fn winit_sample(&self, x: f64, y: f64, data: &TabletToolData) -> Sample {
        Sample {
            source: Source::Winit,
            received: self.start.elapsed(),
            dejittered: None,
            wakeup: self.wakeup,
            x,
            y,
            contact: self.contact,
            pressure: data.force.map(|f| f.normalized(None)),
            tilt: data.tilt.map(|t| (f64::from(t.x), f64::from(t.y))),
            distance: None,
            roll: data.twist.map(f64::from),
            phase: None,
            buttons: self.buttons,
            device_ms: None,
        }
    }

    fn our_sample(&self, s: &input::StrokeSample) -> Sample {
        let since_start = |t: Instant| t.saturating_duration_since(self.start);
        Sample {
            source: Source::Ours,
            received: since_start(s.received),
            dejittered: s.phase.in_stroke().then(|| since_start(s.timestamp)),
            wakeup: self.wakeup,
            x: s.pos.x * self.scale,
            y: s.pos.y * self.scale,
            contact: s.phase.in_stroke(),
            pressure: Some(f64::from(s.pressure.get())),
            tilt: s.tilt.map(|t| (t.x(), t.y())),
            distance: s.distance.map(|d| f64::from(d.get())),
            roll: s.roll,
            phase: Some(s.phase),
            buttons: s.buttons,
            device_ms: Some(s.device_ms),
        }
    }

    /// Records everything our backend has queued: samples over the probe window, and pad events.
    fn drain(&mut self) {
        let Some(backend) = &self.backend else {
            return;
        };
        let (events, surface) = (backend.drain(), backend.surface());
        for event in events {
            match event {
                input::Event::Sample(s) if s.surface == surface => {
                    let sample = self.our_sample(&s);
                    self.record(sample);
                }
                input::Event::Sample(_) => self.recording.elsewhere += 1,
                input::Event::Pad(_) => self.recording.pads += 1,
            }
        }
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
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => self.scale = scale_factor,
            WindowEvent::PointerMoved {
                position,
                source: PointerSource::TabletTool { data, .. },
                ..
            } => self.record(self.winit_sample(position.x, position.y, &data)),
            WindowEvent::PointerButton {
                state,
                position,
                button: ButtonSource::TabletTool { button, data, .. },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                // winit's Wayland backend maps evdev codes as SDL does: BTN_STYLUS (our primary)
                // arrives as `Contact`, like the tip, `Barrel` is BTN_STYLUS2 and `Other(1)` is
                // BTN_STYLUS3. Other codes aren't barrel buttons.
                match button {
                    TabletToolButton::Contact => self.contact = pressed,
                    TabletToolButton::Barrel => {
                        self.buttons = self.buttons.with(Buttons::SECONDARY, pressed);
                    }
                    TabletToolButton::Other(1) => {
                        self.buttons = self.buttons.with(Buttons::TERTIARY, pressed);
                    }
                    TabletToolButton::Other(_) => return Ok(()),
                }
                self.record(self.winit_sample(position.x, position.y, &data));
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
            self.scale = window.scale_factor();
            let selection = self.recording.selection;
            if selection.records(Source::Ours) {
                match ours::Ours::start(event_loop, window.as_ref()) {
                    Ok(backend) => self.backend = Some(backend),
                    Err(e) => {
                        let rest = if selection.records(Source::Winit) {
                            "winit only"
                        } else {
                            "nothing recorded"
                        };
                        self.recording.note =
                            Some(format!("our backend didn't start ({e}); {rest}"));
                    }
                }
            }
            let surface =
                Surface::new(&context, window).map_err(|e| anyhow!("creating the surface: {e}"))?;
            self.surface = Some(surface);
            Ok(())
        })();
        if let Err(e) = result {
            self.finish(event_loop, Err(e));
        }
    }

    fn proxy_wake_up(&mut self, _: &dyn ActiveEventLoop) {
        self.drain();
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
    let Args { selection, output } = Args::parse(std::env::args_os().skip(1))?;
    let (done, outcome) = mpsc::channel();
    EventLoop::new()?.run_app(App::new(selection, done))?;
    let recording = outcome
        .recv()
        .context("the window closed without a recording")??;
    report::write_csv(&recording.samples, &output)?;
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{}", report::summarise(&recording)?)?;
    writeln!(stdout, "csv: {}", output.display())?;
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use super::*;

    /// Parses `args` as the arguments after the program name.
    ///
    /// # Errors
    ///
    /// As [`Args::parse`].
    fn parse(args: &[&str]) -> Result<Args> {
        Args::parse(args.iter().map(OsString::from))
    }

    #[test]
    fn defaults_to_ours_and_the_default_path() {
        assert_eq!(
            parse(&[]).ok(),
            Some(Args {
                selection: Selection::Ours,
                output: "pen-samples.csv".into(),
            })
        );
    }

    #[test]
    fn takes_a_source_in_either_form_and_a_path_either_side() {
        let both = Args {
            selection: Selection::Both,
            output: "out.csv".into(),
        };
        assert_eq!(parse(&["--source", "both", "out.csv"]).ok(), Some(both));
        let winit = Args {
            selection: Selection::Winit,
            output: "out.csv".into(),
        };
        assert_eq!(parse(&["out.csv", "--source=winit"]).ok(), Some(winit));
        assert_eq!(
            parse(&["--source", "ours"]).ok().map(|a| a.selection),
            Some(Selection::Ours)
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        parse(&["--source"]).unwrap_err();
        parse(&["--source", "mouse"]).unwrap_err();
        parse(&["--verbose"]).unwrap_err();
        parse(&["a.csv", "b.csv"]).unwrap_err();
    }

    #[test]
    fn a_selection_records_its_sources() {
        assert!(Selection::Ours.records(Source::Ours));
        assert!(!Selection::Ours.records(Source::Winit));
        assert!(Selection::Winit.records(Source::Winit));
        assert!(!Selection::Winit.records(Source::Ours));
        assert!(Source::ALL.into_iter().all(|s| Selection::Both.records(s)));
    }
}
