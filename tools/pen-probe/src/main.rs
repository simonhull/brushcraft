//! Pen-input probe: records what our own pen backend or winit delivers from a pen tablet, and
//! summarises it.
//!
//! Usage ([`USAGE`]):
//!
//! - `pen-probe [--source winit|ours|both] [--load [--project PATH] [--level N]] [--script] [CSV]`:
//!   open a window and record. Draw on it with the pen, then close it. Samples go to the CSV path
//!   (default `pen-samples.csv`), and a summary is printed to stdout. `--source` picks what is
//!   recorded: our backend (`input::wayland`, the default), winit, or both. Some compositors
//!   deliver tablet events to only one tablet binding per client, so `both` usually sees nothing
//!   from one of them; compare the two by recording a run with each. winit's events are stamped on
//!   arrival and their contact phases go through the same de-jitter filter as ours
//!   ([`winit_tool`]). `--load` runs work like M1a4's frame loop on the main thread meanwhile
//!   ([`load`]), its frames going to `<stem>-load.csv`; `--script` guides the user through timed
//!   steps and closes the window at the end ([`script`]).
//! - `pen-probe --report CSV`: the summary of a CSV the probe wrote, with no window.
//! - `pen-probe --compare WINIT_CSV OURS_CSV`: both summaries, then W against U and the decision's
//!   checks ([`report::compare`]).
//! - `pen-probe --guided [--project PATH] [--level N] [--out DIR]`: the whole comparison in one
//!   command: both recordings, each with the load and the script, then the comparison ([`guided`]).
//! - `pen-probe --load-check [--project PATH] [--level N] [--seconds N]`: the load alone, with no
//!   window, for N seconds (10): what it costs on the main thread.
//!
//! Method and results: `research/platform/pen-input-wayland.md` in the docs repo.

mod frames;
mod guided;
mod load;
mod ours;
mod report;
mod script;
mod winit_tool;

use std::ffi::OsString;
use std::fmt;
use std::io::Write as _;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use geom::Level;
use input::{Buttons, Phase};
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{
    ButtonSource, ElementState, PointerKind, PointerSource, StartCause, TabletToolData, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::frames::FrameRecord;
use crate::load::Load;
use crate::script::{Script, Step};
use crate::winit_tool::{WinitButton, WinitEvent, WinitTool};

/// The canvas colour, `0RGB`: near black.
const BACKGROUND: u32 = 0x0018_1818;
/// The colour strokes are drawn in, `0RGB`: a warm white.
const INK: u32 = 0x00E8_E0D0;
/// A stamp's radius in pixels at full pressure, besides the one pixel every stamp has.
const MAX_RADIUS: f64 = 12.0;
/// The guided script's progress bar: its height in pixels, along the window's top.
const BAR_HEIGHT: usize = 8;
/// The progress bar's unfilled part, `0RGB`.
const BAR_BACK: u32 = 0x0030_3030;
/// How often the progress bar is redrawn.
const BAR_REDRAW: Duration = Duration::from_millis(250);
/// How long `--load-check` runs by default, in seconds.
const DEFAULT_CHECK_SECONDS: u64 = 10;

/// Prints `line` to stdout, best effort: progress and prompts.
fn say(line: impl fmt::Display) {
    drop(writeln!(std::io::stdout().lock(), "{line}"));
}

/// Which path a sample came through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// winit's `PointerMoved` / `PointerButton`, on the event-loop thread.
    Winit,
    /// `input`'s Wayland backend, on its own thread.
    Ours,
}

impl Source {
    /// Every source, in the order reports list them.
    const ALL: [Self; 2] = [Self::Winit, Self::Ours];

    /// The source's name on the command line and in the CSV.
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
    /// winit's samples only.
    Winit,
    /// Our backend's samples only.
    #[default]
    Ours,
    /// Both, side by side, to compare them.
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

/// What the probe prints when its arguments don't parse.
const USAGE: &str = "usage: pen-probe [--source winit|ours|both] [--load [--project PATH] \
                     [--level N]] [--script] [CSV]
       pen-probe --report CSV
       pen-probe --compare WINIT_CSV OURS_CSV
       pen-probe --guided [--project PATH] [--level N] [--out DIR]
       pen-probe --load-check [--project PATH] [--level N] [--seconds N]";

/// The load's options: the project it copies, and the level it brushes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LoadOptions {
    /// The project; `None` for the bench's default ([`load::default_project`]).
    project: Option<PathBuf>,
    /// The level.
    level: u8,
}

impl LoadOptions {
    /// The project's path.
    ///
    /// # Errors
    ///
    /// If there is none given and no default (see [`load::default_project`]).
    fn project(&self) -> Result<PathBuf> {
        match &self.project {
            Some(path) => Ok(path.clone()),
            None => load::default_project(),
        }
    }

    /// The level.
    ///
    /// # Errors
    ///
    /// If it is above the deepest level.
    fn level(&self) -> Result<Level> {
        Ok(Level::new(self.level)?)
    }
}

/// How a recording runs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecordOptions {
    /// The sources to record.
    selection: Selection,
    /// Where the CSV goes.
    output: PathBuf,
    /// The load to run meanwhile, if any.
    load: Option<LoadOptions>,
    /// Whether to follow the guided script.
    script: bool,
}

/// What the command line asks for.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// Open the window, record, then write the CSV and print the summary.
    Record(RecordOptions),
    /// Print the summary of a CSV the probe wrote earlier.
    Report(PathBuf),
    /// Print the summaries of a winit run and one of ours, and the comparison.
    Compare {
        /// The winit run's CSV.
        winit: PathBuf,
        /// Our run's CSV.
        ours: PathBuf,
    },
    /// Run the whole guided comparison.
    Guided {
        /// The load both runs carry.
        load: LoadOptions,
        /// Where the CSVs go; `None` for the default.
        out: Option<PathBuf>,
    },
    /// Run the load with no window.
    LoadCheck {
        /// The load.
        load: LoadOptions,
        /// How long, in seconds.
        seconds: u64,
    },
}

/// Which of the probe's modes the command line names: one at most.
#[derive(Debug, Default, PartialEq, Eq)]
enum Mode {
    /// None of the others: a recording.
    #[default]
    Record,
    /// `--report CSV`.
    Report(PathBuf),
    /// `--compare`.
    Compare,
    /// `--guided`.
    Guided,
    /// `--load-check`.
    LoadCheck,
}

/// The command line's flags, before they are checked against each other.
#[derive(Debug, Default)]
struct Flags {
    /// `--source`.
    selection: Option<Selection>,
    /// The mode: `--report`, `--compare`, `--guided` or `--load-check`, or a recording.
    mode: Mode,
    /// `--load`.
    load: bool,
    /// `--script`.
    script: bool,
    /// `--project`.
    project: Option<PathBuf>,
    /// `--level`.
    level: Option<u8>,
    /// `--out`.
    out: Option<PathBuf>,
    /// `--seconds`.
    seconds: Option<u64>,
    /// The arguments that aren't flags.
    paths: Vec<PathBuf>,
}

impl Flags {
    /// The command the flags make.
    ///
    /// # Errors
    ///
    /// If they don't go together.
    fn command(mut self) -> Result<Command> {
        let recording = self.selection.is_some() || self.load || self.script;
        let load_flags = self.project.is_some() || self.level.is_some();
        let load = LoadOptions {
            project: self.project,
            level: self.level.unwrap_or(load::DEFAULT_LEVEL),
        };
        match self.mode {
            Mode::Report(csv) => {
                if recording
                    || load_flags
                    || self.out.is_some()
                    || self.seconds.is_some()
                    || !self.paths.is_empty()
                {
                    bail!("--report takes no other arguments; {USAGE}");
                }
                return Ok(Command::Report(csv));
            }
            Mode::Compare => {
                if recording || load_flags || self.out.is_some() || self.seconds.is_some() {
                    bail!("--compare takes two CSVs and nothing else; {USAGE}");
                }
                let Ok([winit, ours]) = <[PathBuf; 2]>::try_from(self.paths) else {
                    bail!("--compare takes two CSVs, winit's then ours; {USAGE}");
                };
                return Ok(Command::Compare { winit, ours });
            }
            Mode::Guided => {
                if recording || self.seconds.is_some() || !self.paths.is_empty() {
                    bail!("--guided takes only --project, --level and --out; {USAGE}");
                }
                return Ok(Command::Guided {
                    load,
                    out: self.out,
                });
            }
            Mode::LoadCheck => {
                if recording || self.out.is_some() || !self.paths.is_empty() {
                    bail!("--load-check takes only --project, --level and --seconds; {USAGE}");
                }
                return Ok(Command::LoadCheck {
                    load,
                    seconds: self.seconds.unwrap_or(DEFAULT_CHECK_SECONDS),
                });
            }
            Mode::Record => {}
        }
        if self.out.is_some() || self.seconds.is_some() {
            bail!("--out goes with --guided, --seconds with --load-check; {USAGE}");
        }
        if load_flags && !self.load {
            bail!("--project and --level go with --load; {USAGE}");
        }
        let output = self.paths.pop();
        if !self.paths.is_empty() {
            bail!("more than one output path; {USAGE}");
        }
        Ok(Command::Record(RecordOptions {
            selection: self.selection.unwrap_or_default(),
            output: output.unwrap_or_else(|| "pen-samples.csv".into()),
            load: self.load.then_some(load),
            script: self.script,
        }))
    }

    /// Sets the mode to `mode`.
    ///
    /// # Errors
    ///
    /// If a mode was already set: `--report`, `--compare`, `--guided` and `--load-check` don't
    /// combine, nor repeat.
    fn set_mode(&mut self, mode: Mode) -> Result<()> {
        if self.mode != Mode::Record {
            bail!("one of --report, --compare, --guided and --load-check, once; {USAGE}");
        }
        self.mode = mode;
        Ok(())
    }

    /// Takes `arg`, a flag with a value (from `rest` if it isn't `flag=VALUE`) or a path.
    ///
    /// # Errors
    ///
    /// On an unknown flag or source, a flag without its value, or a bad number.
    fn take(&mut self, arg: OsString, rest: &mut impl Iterator<Item = OsString>) -> Result<()> {
        if let Some(source) = flag_value(&arg, "--source", rest)? {
            self.selection = Some(match source.to_str() {
                Some("winit") => Selection::Winit,
                Some("ours") => Selection::Ours,
                Some("both") => Selection::Both,
                _ => bail!(
                    "unknown source {}: expected winit, ours or both",
                    source.display()
                ),
            });
        } else if let Some(csv) = flag_value(&arg, "--report", rest)? {
            self.set_mode(Mode::Report(PathBuf::from(csv)))?;
        } else if let Some(project) = flag_value(&arg, "--project", rest)? {
            self.project = Some(project.into());
        } else if let Some(level) = flag_value(&arg, "--level", rest)? {
            self.level = Some(number(&level, "--level")?);
        } else if let Some(out) = flag_value(&arg, "--out", rest)? {
            self.out = Some(out.into());
        } else if let Some(seconds) = flag_value(&arg, "--seconds", rest)? {
            self.seconds = Some(number(&seconds, "--seconds")?);
        } else if arg.to_str().is_some_and(|a| a.starts_with('-')) {
            bail!("unknown option {}; {USAGE}", arg.display());
        } else {
            self.paths.push(PathBuf::from(arg));
        }
        Ok(())
    }
}

impl Command {
    /// Parses the arguments after the program name.
    ///
    /// # Errors
    ///
    /// On an unknown flag or source, a flag without its value or with a bad number, flags that
    /// don't go together, or the wrong number of paths.
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut flags = Flags::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--compare") => flags.set_mode(Mode::Compare)?,
                Some("--guided") => flags.set_mode(Mode::Guided)?,
                Some("--load-check") => flags.set_mode(Mode::LoadCheck)?,
                Some("--load") => flags.load = true,
                Some("--script") => flags.script = true,
                _ => flags.take(arg, &mut args)?,
            }
        }
        flags.command()
    }
}

/// `value`, the value of `flag`, as a number.
///
/// # Errors
///
/// If it isn't one.
fn number<T: std::str::FromStr>(value: &OsString, flag: &str) -> Result<T> {
    value
        .to_str()
        .and_then(|v| v.parse().ok())
        .with_context(|| format!("{flag} takes a number, not {}", value.display()))
}

/// The value of `flag` if `arg` is it, as `flag VALUE` (taking the next argument) or `flag=VALUE`.
///
/// # Errors
///
/// If `arg` is `flag` and no argument follows.
fn flag_value(
    arg: &OsString,
    flag: &str,
    rest: &mut impl Iterator<Item = OsString>,
) -> Result<Option<OsString>> {
    if arg == flag {
        return rest
            .next()
            .with_context(|| format!("{flag} needs a value; {USAGE}"))
            .map(Some);
    }
    Ok(arg
        .to_str()
        .and_then(|a| a.strip_prefix(flag)?.strip_prefix('='))
        .map(OsString::from))
}

/// One pen sample as the probe records it.
#[derive(Debug, Clone)]
struct Sample {
    /// The path it came through.
    source: Source,
    /// When it was received (winit: event-loop thread; ours: input thread), since the probe
    /// started.
    received: Duration,
    /// When the main thread handled it, since the probe started: for winit the receipt itself,
    /// for ours when the event loop drained it. `None` in CSVs from before it was recorded.
    handled: Option<Duration>,
    /// The de-jittered time, since the probe started, for contact phases: ours from our backend,
    /// winit's through the same filter on the event-loop thread ([`winit_tool`]); `None` for
    /// hover, and for winit rows of CSVs from before.
    dejittered: Option<Duration>,
    /// Index of the event-loop wakeup that delivered this sample.
    wakeup: u64,
    /// Horizontal position in physical pixels, from the window's left edge.
    x: f64,
    /// Vertical position in physical pixels, from the window's top edge.
    y: f64,
    /// Whether the sample belongs to a stroke, as its source sees it: from the touch through the
    /// lift.
    contact: bool,
    /// Tip pressure, `0` to `1`; `None` if the source didn't report it.
    pressure: Option<f64>,
    /// Tilt in degrees.
    tilt: Option<(f64, f64)>,
    /// Hover distance, `0` to `1`; `None` if the source didn't report it.
    distance: Option<f64>,
    /// Barrel rotation in degrees; `None` if the source didn't report it.
    roll: Option<f64>,
    /// The phase: ours from our backend, winit's as its contact implies ([`winit_tool`]); `None`
    /// for winit rows of CSVs from before.
    phase: Option<Phase>,
    /// Barrel buttons held; never the tip. For winit, never the primary either: winit reports
    /// `BTN_STYLUS` as contact.
    buttons: Buttons,
    /// The compositor's millisecond timestamp; `None` for winit samples.
    device_ms: Option<u32>,
    /// The winit event the row is; `None` for ours, and in CSVs from before it was recorded.
    winit_event: Option<WinitEvent>,
    /// Whether the tool was in proximity after the event; `None` in CSVs from before.
    proximity: Option<bool>,
    /// The guided script's step; `None` outside the script, and in CSVs from before.
    step: Option<Step>,
}

/// What our backend delivered besides samples over the probe window.
#[derive(Debug, Clone, Copy, Default)]
struct OursCounts {
    /// Pad events (buttons, rings, strips, dials).
    pads: usize,
    /// Samples over some other surface, left out.
    elsewhere: usize,
    /// Events the queue dropped.
    dropped: u64,
}

/// Everything one run recorded.
#[derive(Debug, Default)]
struct Recording {
    /// Which sources were recorded.
    selection: Selection,
    /// The samples of every recorded source, in the order they were recorded.
    samples: Vec<Sample>,
    /// What our backend delivered besides samples. `None` when the recording was read back from a
    /// CSV, which doesn't carry them.
    ours_counts: Option<OursCounts>,
    /// Why our backend didn't run or stopped badly, if it did.
    note: Option<String>,
    /// The load's frames, if the run had a load.
    frames: Option<Vec<FrameRecord>>,
}

/// What the probe draws: stamped discs whose radius follows pressure.
#[derive(Debug)]
struct Canvas {
    /// Width in pixels: the window surface's.
    width: usize,
    /// Height in pixels: the window surface's.
    height: usize,
    /// The pixels, `0RGB`, row-major.
    pixels: Vec<u32>,
}

impl Canvas {
    /// A blank canvas of `width × height` pixels.
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![BACKGROUND; width * height],
        }
    }

    /// Fills the disc of `radius` pixels around `(x, y)` with ink, clipped to the canvas.
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

/// Paints the guided script's progress bar, `progress` (`0` to `1`) filled in `colour`, over the
/// top rows of `pixels`, a frame `width` pixels wide.
fn paint_bar(pixels: &mut [u32], width: usize, progress: f64, colour: u32) {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "progress is clamped to 0..=1, so the product is in 0..=width"
    )]
    let filled = (report::to_f64(width) * progress.clamp(0.0, 1.0)) as usize;
    let rows = pixels.chunks_mut(width.max(1)).take(BAR_HEIGHT);
    for row in rows {
        for (x, pixel) in row.iter_mut().enumerate() {
            *pixel = if x < filled { colour } else { BAR_BACK };
        }
    }
}

/// The recording, and the error that ended the run early, if one did.
type Outcome = (Recording, Result<()>);

/// The probe's window and recording, driven by winit's event loop.
#[derive(Debug)]
struct App {
    /// When the probe started: sample times count from here.
    start: Instant,
    /// Where the recording goes back to `main` when the window closes.
    done: mpsc::Sender<Outcome>,
    /// Whether the recording has been sent: the run is over.
    sent: bool,
    /// Declared before `surface`, so it drops (stopping the input thread) before the window does:
    /// the backend must stop before the window is destroyed. The order is load-bearing.
    backend: Option<ours::Ours>,
    /// The window and its software surface, once the loop can create them.
    surface: Option<Surface<OwnedDisplayHandle, Box<dyn Window>>>,
    /// What the window shows.
    canvas: Canvas,
    /// winit's tablet tool, followed as a stroke engine on winit would.
    winit_tool: WinitTool,
    /// Event-loop wakeups so far, to tell which samples one wakeup delivered together.
    wakeup: u64,
    /// The window's scale factor, to put our logical positions in winit's physical pixels.
    scale: f64,
    /// What our backend delivered besides samples so far.
    ours_counts: OursCounts,
    /// The load on the main thread, if the run has one.
    load: Option<Load>,
    /// The guided script, if the run follows one.
    script: Option<Script>,
    /// When the progress bar was last redrawn.
    bar_drawn: Instant,
    /// What has been recorded so far.
    recording: Recording,
}

impl App {
    /// A probe recording the sources of `selection`, with `load` on its main thread and the
    /// guided script if `script`, which sends its recording to `done`.
    fn new(
        selection: Selection,
        load: Option<Load>,
        script: bool,
        done: mpsc::Sender<Outcome>,
    ) -> Self {
        let start = Instant::now();
        Self {
            start,
            done,
            sent: false,
            backend: None,
            surface: None,
            canvas: Canvas::new(0, 0),
            winit_tool: WinitTool::default(),
            wakeup: 0,
            scale: 1.0,
            ours_counts: OursCounts::default(),
            load,
            script: script.then(Script::default),
            bar_drawn: start,
            recording: Recording {
                selection,
                ..Recording::default()
            },
        }
    }

    /// Stops our backend and the load, hands the recording (or the error that ended it) back to
    /// `main` and stops the loop. Only the first call does more than stop the loop.
    ///
    /// The backend stops here, before the loop exits and the window is destroyed, as its docs
    /// require.
    fn finish(&mut self, event_loop: &dyn ActiveEventLoop, result: Result<()>) {
        event_loop.exit();
        if self.sent {
            return;
        }
        self.sent = true;
        self.drain();
        if let Some(backend) = self.backend.take() {
            let surface = backend.surface();
            let stopped = backend.stop();
            self.record_ours(stopped.events, surface);
            self.ours_counts.dropped = stopped.dropped;
            if let Err(e) = stopped.result {
                self.recording.note = Some(format!("our backend stopped with an error: {e:#}"));
            }
        }
        let result = match self.load.take() {
            Some(load) => {
                let (frames, closed) = load.finish();
                self.recording.frames = Some(frames);
                result.and(closed)
            }
            None => result,
        };
        self.recording.ours_counts = Some(self.ours_counts);
        let recording = std::mem::take(&mut self.recording);
        // `main` holds the receiver until the loop returns, so the send can't fail.
        drop(self.done.send((recording, result)));
    }

    /// Which source the canvas draws: ours when it runs, winit's otherwise.
    const fn drawn(&self) -> Source {
        if self.backend.is_some() {
            Source::Ours
        } else {
            Source::Winit
        }
    }

    /// The guided step the run is in, if it follows the script.
    fn step(&self) -> Option<Step> {
        self.script.as_ref().map(Script::step)
    }

    /// Records `sample` if its source is selected, and stamps it on the canvas if it touches
    /// and its source is the one drawn.
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

    /// A winit tablet event at `(x, y)` (physical pixels), received now, with its tool data if it
    /// has some: stamped, put through [`WinitTool`], and recorded.
    fn winit_event(&mut self, event: WinitEvent, x: f64, y: f64, data: Option<&TabletToolData>) {
        let received = Instant::now();
        self.pen_arrived(received);
        if !self.recording.selection.records(Source::Winit) {
            return;
        }
        let stamped = self.winit_tool.apply(event, (x, y), received, self.wakeup);
        let start = self.start;
        let since = |t: Instant| t.saturating_duration_since(start);
        let sample = Sample {
            source: Source::Winit,
            received: since(received),
            handled: Some(since(received)),
            dejittered: stamped.timestamp.map(since),
            wakeup: self.wakeup,
            x,
            y,
            contact: stamped.phase.in_stroke(),
            pressure: data.and_then(|d| d.force).map(|f| f.normalized(None)),
            tilt: data
                .and_then(|d| d.tilt)
                .map(|t| (f64::from(t.x), f64::from(t.y))),
            distance: None,
            roll: data.and_then(|d| d.twist).map(f64::from),
            phase: Some(stamped.phase),
            buttons: stamped.buttons,
            device_ms: None,
            winit_event: Some(event),
            proximity: Some(stamped.proximity),
            step: self.step(),
        };
        self.record(sample);
    }

    /// A sample of our backend's, in the probe's terms: times since the start, physical pixels;
    /// handled at `handled`.
    fn our_sample(&self, s: &input::StrokeSample, handled: Instant) -> Sample {
        let since_start = |t: Instant| t.saturating_duration_since(self.start);
        Sample {
            source: Source::Ours,
            received: since_start(s.received),
            handled: Some(since_start(handled)),
            dejittered: s.phase.in_stroke().then(|| since_start(s.timestamp)),
            wakeup: self.wakeup,
            // `scale` is the factor when the sample is drained, not when it was taken.
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
            winit_event: None,
            proximity: Some(s.phase != Phase::Leave),
            step: self.step(),
        }
    }

    /// Records everything our backend has queued.
    fn drain(&mut self) {
        let Some(backend) = &self.backend else {
            return;
        };
        let (events, surface) = (backend.drain(), backend.surface());
        self.record_ours(events, surface);
    }

    /// Records our backend's `events`, handled now: samples over the probe window's `surface`,
    /// and counts of the rest.
    fn record_ours(&mut self, events: Vec<input::Event>, surface: input::SurfaceId) {
        let handled = Instant::now();
        for event in events {
            match event {
                input::Event::Sample(s) if s.surface == surface => {
                    self.pen_arrived(handled);
                    let sample = self.our_sample(&s, handled);
                    self.record(sample);
                }
                input::Event::Sample(_) => self.ours_counts.elsewhere += 1,
                input::Event::Pad(_) => self.ours_counts.pads += 1,
            }
        }
    }

    /// The pen came over the window at `now`: the guided script's countdown starts, the first
    /// time.
    fn pen_arrived(&mut self, now: Instant) {
        if self.script.as_mut().is_some_and(|s| s.pen_arrived(now)) {
            self.step_changed();
        }
    }

    /// The guided script moved to another step: prompts for it (stdout and the window's title)
    /// and clears the canvas.
    fn step_changed(&mut self) {
        let Some(step) = self.step() else {
            return;
        };
        let number = Step::TIMED.iter().position(|&s| s == step);
        match number {
            Some(i) => say(format_args!(
                "[step {} of {}, {} s] {}",
                i + 1,
                Step::TIMED.len(),
                step.duration().as_secs(),
                step.prompt()
            )),
            None => say(step.prompt()),
        }
        if let Some(surface) = &self.surface {
            surface
                .window()
                .set_title(&format!("pen-probe: {}", step.prompt()));
        }
        self.canvas.pixels.fill(BACKGROUND);
        self.request_redraw();
    }

    /// What the loop does between batches of events: drains our backend, moves the script on
    /// (finishing the run after its last step), redraws its progress bar now and then, and runs
    /// the load's frame when it is due.
    ///
    /// # Errors
    ///
    /// If the load's frame fails.
    fn upkeep(&mut self, event_loop: &dyn ActiveEventLoop) -> Result<()> {
        self.drain();
        let now = Instant::now();
        if self.script.as_mut().is_some_and(|s| s.advance(now)) {
            self.step_changed();
        }
        if self.script.as_ref().is_some_and(Script::done) {
            self.finish(event_loop, Ok(()));
            return Ok(());
        }
        if self.script.is_some() && now >= self.bar_drawn + BAR_REDRAW {
            self.bar_drawn = now;
            self.request_redraw();
        }
        if let Some(load) = &mut self.load
            && now >= load.next_frame()
        {
            load.frame(now, self.start)?;
        }
        Ok(())
    }

    /// When the loop must next wake without an event: the load's next frame, the script's next
    /// step, or the progress bar's next redraw.
    fn wake_at(&self) -> Option<Instant> {
        let bar = self.script.map(|_| self.bar_drawn + BAR_REDRAW);
        [
            self.load.as_ref().map(Load::next_frame),
            self.script.as_ref().and_then(Script::ends),
            bar,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// Copies the canvas to the window, with the script's progress bar along the top.
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
        if let Some(script) = &self.script {
            paint_bar(
                &mut buffer,
                width,
                script.progress(Instant::now()),
                script.step().colour(),
            );
        }
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
            WindowEvent::PointerEntered {
                position,
                kind: PointerKind::TabletTool(_),
                ..
            } => self.winit_event(WinitEvent::Entered, position.x, position.y, None),
            WindowEvent::PointerLeft {
                position,
                kind: PointerKind::TabletTool(_),
                ..
            } => {
                let (x, y) = position.map_or((0.0, 0.0), |p| (p.x, p.y));
                self.winit_event(WinitEvent::Left, x, y, None);
            }
            WindowEvent::PointerMoved {
                position,
                source: PointerSource::TabletTool { data, .. },
                ..
            } => self.winit_event(WinitEvent::Moved, position.x, position.y, Some(&data)),
            WindowEvent::PointerButton {
                state,
                position,
                button: ButtonSource::TabletTool { button, data, .. },
                ..
            } => {
                // winit's Wayland backend maps evdev codes as SDL does: BTN_STYLUS (our primary)
                // arrives as `Contact`, like the tip, `Barrel` is BTN_STYLUS2 and `Other(1)` is
                // BTN_STYLUS3. Every one is recorded as it came.
                let tablet_event = WinitEvent::Button {
                    button: WinitButton::from_winit(button),
                    pressed: state == ElementState::Pressed,
                };
                self.winit_event(tablet_event, position.x, position.y, Some(&data));
            }
            _ => {}
        }
        Ok(())
    }

    /// Asks the window for a redraw, if it exists yet.
    fn request_redraw(&self) {
        if let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
    }
}

/// The refresh interval of the display `window` is on, or the load's default (60 Hz) if it
/// doesn't say: what the load's frames are paced to, as the app paces its own.
fn refresh_interval(window: &dyn Window) -> Duration {
    window
        .current_monitor()
        .and_then(|monitor| monitor.current_video_mode())
        .and_then(|mode| mode.refresh_rate_millihertz())
        .map_or(load::DEFAULT_INTERVAL, |millihertz| {
            Duration::from_secs_f64(1000.0 / f64::from(millihertz.get()))
        })
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
            if let Some(load) = &mut self.load {
                load.set_interval(refresh_interval(window.as_ref()));
                load.restart_clock(Instant::now());
            }
            // The surface first: if it fails, the window is destroyed before any backend runs.
            let surface =
                Surface::new(&context, window).map_err(|e| anyhow!("creating the surface: {e}"))?;
            let selection = self.recording.selection;
            if selection.records(Source::Ours) {
                match ours::Ours::start(event_loop, surface.window().as_ref()) {
                    Ok(backend) => self.backend = Some(backend),
                    Err(e) => {
                        let rest = if selection.records(Source::Winit) {
                            "winit only"
                        } else {
                            "nothing recorded"
                        };
                        self.recording.note =
                            Some(format!("our backend didn't start ({e:#}); {rest}"));
                    }
                }
            }
            self.surface = Some(surface);
            Ok(())
        })();
        if let Err(e) = result {
            self.finish(event_loop, Err(e));
        } else if self.script.is_some() {
            self.step_changed();
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

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.sent {
            return;
        }
        if let Err(e) = self.upkeep(event_loop) {
            self.finish(event_loop, Err(e));
        }
        if self.sent {
            return;
        }
        event_loop.set_control_flow(
            self.wake_at()
                .map_or(ControlFlow::Wait, ControlFlow::WaitUntil),
        );
    }
}

/// Opens the window and records as `options` says, then writes the CSV (and the load's CSV)
/// and prints the summary.
///
/// # Errors
///
/// If the load can't be opened, the window created or drawn (after writing what was recorded),
/// or a CSV can't be written.
fn record(options: &RecordOptions) -> Result<()> {
    let load = options
        .load
        .as_ref()
        .map(|l| -> Result<Load> { Load::open(&l.project()?, l.level()?) })
        .transpose()?;
    if options.script {
        say(format_args!(
            "pen-probe: the guided script, {} s once the pen comes over the window; follow the \
             prompts here and in the window's title. Closing the window ends it early.",
            Step::total().as_secs()
        ));
    }
    let (done, outcome) = mpsc::channel();
    let ran = EventLoop::new()?
        .run_app(App::new(options.selection, load, options.script, done))
        .context("running the event loop");
    let Ok((recording, result)) = outcome.try_recv() else {
        ran?;
        bail!("the window closed without a recording");
    };
    report::write_csv(&recording.samples, &options.output)?;
    let load_csv = frames::path_for(&options.output);
    if let Some(frames) = &recording.frames {
        frames::write(frames, &load_csv)?;
    }
    let mut stdout = std::io::stdout().lock();
    write!(stdout, "{}", report::summarise(&recording)?)?;
    writeln!(stdout, "csv: {}", options.output.display())?;
    if recording.frames.is_some() {
        writeln!(stdout, "load csv: {}", load_csv.display())?;
    }
    result.and(ran)
}

/// Runs what the command line asks for: see the crate notes.
///
/// # Errors
///
/// If the window can't be created or drawn (after writing what was recorded), a CSV can't be
/// read or written, or the load fails.
fn main() -> Result<()> {
    match Command::parse(std::env::args_os().skip(1))? {
        Command::Record(options) => record(&options),
        Command::Report(csv) => {
            let recording = report::read_csv(&csv)?;
            write!(
                std::io::stdout().lock(),
                "{}",
                report::summarise(&recording)?
            )?;
            Ok(())
        }
        Command::Compare { winit, ours } => {
            let (w, u) = (report::read_csv(&winit)?, report::read_csv(&ours)?);
            let mut stdout = std::io::stdout().lock();
            writeln!(stdout, "== {} ==", winit.display())?;
            write!(stdout, "{}", report::summarise(&w)?)?;
            writeln!(stdout, "== {} ==", ours.display())?;
            write!(stdout, "{}", report::summarise(&u)?)?;
            write!(stdout, "{}", report::compare::compare(&w, &u)?)?;
            Ok(())
        }
        Command::Guided { load, out } => guided::run(&load.project()?, load.level()?, out),
        Command::LoadCheck { load, seconds } => {
            let frames = load::check(&load.project()?, load.level()?, seconds)?;
            write!(
                std::io::stdout().lock(),
                "{}",
                report::load_summary(&frames)?
            )?;
            Ok(())
        }
    }
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
    /// As [`Command::parse`].
    fn parse(args: &[&str]) -> Result<Command> {
        Command::parse(args.iter().map(OsString::from))
    }

    fn record(selection: Selection, output: &str) -> Command {
        Command::Record(RecordOptions {
            selection,
            output: output.into(),
            load: None,
            script: false,
        })
    }

    fn default_load() -> LoadOptions {
        LoadOptions {
            project: None,
            level: load::DEFAULT_LEVEL,
        }
    }

    #[test]
    fn defaults_to_ours_and_the_default_path() {
        assert_eq!(
            parse(&[]).ok(),
            Some(record(Selection::Ours, "pen-samples.csv"))
        );
    }

    #[test]
    fn takes_a_source_in_either_form_and_a_path_either_side() {
        assert_eq!(
            parse(&["--source", "both", "out.csv"]).ok(),
            Some(record(Selection::Both, "out.csv"))
        );
        assert_eq!(
            parse(&["out.csv", "--source=winit"]).ok(),
            Some(record(Selection::Winit, "out.csv"))
        );
        assert_eq!(
            parse(&["--source", "ours"]).ok(),
            Some(record(Selection::Ours, "pen-samples.csv"))
        );
    }

    #[test]
    fn takes_a_report_path_in_either_form() {
        let report = Some(Command::Report("run.csv".into()));
        assert_eq!(parse(&["--report", "run.csv"]).ok(), report);
        assert_eq!(parse(&["--report=run.csv"]).ok(), report);
    }

    #[test]
    fn takes_a_load_and_the_script_on_a_recording() {
        assert_eq!(
            parse(&[
                "--source",
                "winit",
                "--load",
                "--project=/p.bcrf",
                "--level",
                "8",
                "--script",
                "w.csv"
            ])
            .ok(),
            Some(Command::Record(RecordOptions {
                selection: Selection::Winit,
                output: "w.csv".into(),
                load: Some(LoadOptions {
                    project: Some("/p.bcrf".into()),
                    level: 8,
                }),
                script: true,
            }))
        );
        assert_eq!(
            parse(&["--load"]).ok(),
            Some(Command::Record(RecordOptions {
                selection: Selection::Ours,
                output: "pen-samples.csv".into(),
                load: Some(default_load()),
                script: false,
            }))
        );
    }

    #[test]
    fn takes_compare_guided_and_load_check() {
        assert_eq!(
            parse(&["--compare", "w.csv", "u.csv"]).ok(),
            Some(Command::Compare {
                winit: "w.csv".into(),
                ours: "u.csv".into()
            })
        );
        assert_eq!(
            parse(&["--guided", "--out", "/scratch"]).ok(),
            Some(Command::Guided {
                load: default_load(),
                out: Some("/scratch".into())
            })
        );
        assert_eq!(
            parse(&["--load-check", "--seconds=3", "--project", "/p.bcrf"]).ok(),
            Some(Command::LoadCheck {
                load: LoadOptions {
                    project: Some("/p.bcrf".into()),
                    level: load::DEFAULT_LEVEL,
                },
                seconds: 3
            })
        );
        assert_eq!(
            parse(&["--load-check"]).ok(),
            Some(Command::LoadCheck {
                load: default_load(),
                seconds: DEFAULT_CHECK_SECONDS
            })
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        parse(&["--source"]).unwrap_err();
        parse(&["--source", "mouse"]).unwrap_err();
        parse(&["--verbose"]).unwrap_err();
        parse(&["--sourcewinit"]).unwrap_err();
        parse(&["a.csv", "b.csv"]).unwrap_err();
        parse(&["--report"]).unwrap_err();
        parse(&["--report", "a.csv", "b.csv"]).unwrap_err();
        parse(&["--report", "a.csv", "--source", "ours"]).unwrap_err();
        parse(&["--report", "a.csv", "--report", "b.csv"]).unwrap_err();
        parse(&["--compare", "a.csv"]).unwrap_err();
        parse(&["--compare", "a.csv", "b.csv", "c.csv"]).unwrap_err();
        parse(&["--compare", "a.csv", "b.csv", "--load"]).unwrap_err();
        parse(&["--guided", "--source", "ours"]).unwrap_err();
        parse(&["--guided", "out.csv"]).unwrap_err();
        parse(&["--guided", "--report", "a.csv"]).unwrap_err();
        parse(&["--load-check", "--script"]).unwrap_err();
        parse(&["--level", "9"]).unwrap_err();
        parse(&["--load", "--level", "nine"]).unwrap_err();
        parse(&["--out", "/x"]).unwrap_err();
        parse(&["--seconds", "3"]).unwrap_err();
    }

    #[test]
    fn a_selection_records_its_sources() {
        assert!(Selection::Ours.records(Source::Ours));
        assert!(!Selection::Ours.records(Source::Winit));
        assert!(Selection::Winit.records(Source::Winit));
        assert!(!Selection::Winit.records(Source::Ours));
        assert!(Source::ALL.into_iter().all(|s| Selection::Both.records(s)));
    }

    #[test]
    fn the_progress_bar_fills_its_share_of_the_top_rows() {
        let (width, height) = (10, 12);
        let mut pixels = vec![BACKGROUND; width * height];
        paint_bar(&mut pixels, width, 0.3, 0x00FF_0000);
        for (y, row) in pixels.chunks(width).enumerate() {
            for (x, &p) in row.iter().enumerate() {
                let want = match (y < BAR_HEIGHT, x < 3) {
                    (true, true) => 0x00FF_0000,
                    (true, false) => BAR_BACK,
                    (false, _) => BACKGROUND,
                };
                assert_eq!(p, want, "pixel ({x}, {y})");
            }
        }
    }
}
