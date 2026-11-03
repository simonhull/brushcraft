//! The command line: the project to open and how to run.
//!
//! Hand-parsed from `std::env::args`, as the benchmark harness does: `brushcraft PROJECT
//! [--flag VALUE | --flag=VALUE | --switch]…`. Every option has a default, and an unknown one
//! is refused, so a typo never runs with a setting silently ignored.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use geom::Level;
use render::gpu::AdapterRequest;
use render::surface::Size;

use crate::frame::RecordSetup;
use crate::script::{Script, ScriptPath, Speed};

/// What the app prints when its arguments don't parse.
pub(crate) const USAGE: &str = "\
usage: brushcraft PROJECT [OPTIONS]

Opens the project (a .bcrf file outside the source repository) in a window and draws it.
The pen sculpts where it touches the model ([ and ] its size, - and = its strength); off the
model it orbits. Drag with the left button to orbit, with Shift or the right or middle button
to pan, with Ctrl to zoom; the wheel zooms too. The pen's first barrel button zooms, its
second pans.

options:
  --level N        the finest level of the display layer the store keeps (default 9)
  --budget-mib N   the tile store's memory budget (default 2048)
  --adapter A      the GPU: default (the system's high-performance one), hardware (the
                   default unless it is a CPU rasterizer), software (a CPU rasterizer such as
                   lavapipe), or any other text, matched against the adapters' names
  --size WxH       the window's size, or the offscreen frame's (default 1280x800)
  --frames N       draw N frames, every one, then close: a scripted run
  --script         move the camera along the benchmark path (still, an orbit, a zoom in, a
                   flick, still), drawing every frame, then close: the measured run
  --script-path P  the path --script runs: benchmark (the default); strokes (still, in
                   close, a pen stroke, out, a stroke seen from a distance, round, a fast
                   long stroke across the whole model, still): the pen's latency, a stroke
                   from afar, and a brush keeping up; or sweep (still, in close looking at
                   the surface at a slant, one long stroke pressed from none to full, still):
                   how a stroke looks
  --script-speed X run the path X times as fast (default 1)
  --record FILE    with --script, write a JSON line per frame to FILE: GPU phase times, the
                   owner thread's phases, what was drawn, residency and uploads
  --no-gpu-timing  don't time the GPU, even where it can be (with --script)
  --offscreen      no window: draw into a texture (needs --frames or --script)";

/// What ends a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Run {
    /// Nothing: the window is closed.
    Open,
    /// That many frames drawn, every one.
    Frames(u64),
    /// The benchmark camera path ([`crate::script`]) run through, at this speed.
    Script(Speed),
}

/// How the app runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Mode {
    /// What ends the run.
    pub(crate) run: Run,
    /// Whether frames are drawn into a texture, with no window (never with [`Run::Open`]:
    /// nothing would end it).
    pub(crate) offscreen: bool,
}

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Options {
    /// The project file, as given.
    pub(crate) project: PathBuf,
    /// The finest level of the display layer.
    pub(crate) level: Level,
    /// The tile store's budget, in bytes.
    pub(crate) budget_bytes: usize,
    /// Which GPU to use.
    pub(crate) adapter: AdapterRequest,
    /// The window's or offscreen frame's size.
    pub(crate) size: Size,
    /// How to run.
    pub(crate) mode: Mode,
    /// Where a scripted run's per-frame records go ([`crate::record`]), if anywhere.
    pub(crate) record: Option<PathBuf>,
    /// Whether a scripted run times the GPU's phases, where the device can.
    pub(crate) gpu_timing: bool,
    /// The path a `--script` run follows.
    pub(crate) script_path: ScriptPath,
}

/// The default finest level: the benchmark project's.
const DEFAULT_LEVEL: u8 = 9;
/// The default budget, in MiB.
const DEFAULT_BUDGET_MIB: usize = 2048;
/// The default size.
const DEFAULT_SIZE: (u32, u32) = (1280, 800);

impl Options {
    /// Parses the arguments after the program name.
    ///
    /// # Errors
    ///
    /// On an unknown option, an option without its value or with one that doesn't parse, no
    /// project or more than one, `--offscreen` without `--frames` or `--script`, both of those,
    /// or `--record`, `--script-path` or `--script-speed` without `--script`.
    pub(crate) fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut project = None;
        let mut level = DEFAULT_LEVEL;
        let mut budget_mib = DEFAULT_BUDGET_MIB;
        let mut adapter = AdapterRequest::Default;
        let mut size = None;
        let mut frames = None;
        let mut script = false;
        let mut speed = None;
        let mut path = None;
        let mut record = None;
        let mut gpu_timing = true;
        let mut offscreen = false;
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            if let Some(value) = flag_value(&arg, "--level", &mut args)? {
                level = number(&value, "--level")?;
            } else if let Some(value) = flag_value(&arg, "--budget-mib", &mut args)? {
                budget_mib = number(&value, "--budget-mib")?;
            } else if let Some(value) = flag_value(&arg, "--adapter", &mut args)? {
                adapter = match text(&value, "--adapter")? {
                    "default" => AdapterRequest::Default,
                    "hardware" => AdapterRequest::Hardware,
                    "software" => AdapterRequest::Software,
                    name => AdapterRequest::Named(name.to_owned()),
                };
            } else if let Some(value) = flag_value(&arg, "--size", &mut args)? {
                size = Some(parse_size(text(&value, "--size")?)?);
            } else if let Some(value) = flag_value(&arg, "--frames", &mut args)? {
                let count: u64 = number(&value, "--frames")?;
                ensure!(count > 0, "--frames needs at least 1; {USAGE}");
                frames = Some(count);
            } else if let Some(value) = flag_value(&arg, "--script-speed", &mut args)? {
                let parsed: Speed = text(&value, "--script-speed")?.parse()?;
                speed = Some(parsed);
            } else if let Some(value) = flag_value(&arg, "--script-path", &mut args)? {
                let parsed: ScriptPath = text(&value, "--script-path")?.parse()?;
                path = Some(parsed);
            } else if let Some(value) = flag_value(&arg, "--record", &mut args)? {
                record = Some(PathBuf::from(value));
            } else if arg == "--script" {
                script = true;
            } else if arg == "--no-gpu-timing" {
                gpu_timing = false;
            } else if arg == "--offscreen" {
                offscreen = true;
            } else if arg.to_str().is_some_and(|a| a.starts_with('-')) {
                bail!("unknown option {}; {USAGE}", arg.display());
            } else if project.replace(PathBuf::from(arg)).is_some() {
                bail!("more than one project; {USAGE}");
            }
        }
        let project = project.ok_or_else(|| anyhow!("no project given; {USAGE}"))?;
        let run = match (frames, script) {
            (Some(_), true) => bail!("--frames or --script, not both: each ends the run"),
            (Some(frames), false) => Run::Frames(frames),
            (None, true) => Run::Script(speed.unwrap_or(Speed::ONE)),
            (None, false) => Run::Open,
        };
        ensure!(
            !offscreen || run != Run::Open,
            "--offscreen needs --frames or --script: with no window, nothing closes it"
        );
        let scripted = matches!(run, Run::Script(_));
        ensure!(
            scripted || (record.is_none() && speed.is_none() && path.is_none()),
            "--record, --script-path and --script-speed go with --script"
        );
        let mode = Mode { run, offscreen };
        let default_size = Size::new(DEFAULT_SIZE.0, DEFAULT_SIZE.1)
            .ok_or_else(|| anyhow!("the default size is not empty"))?;
        Ok(Self {
            project,
            level: Level::new(level).map_err(|e| anyhow!("--level {level}: {e}"))?,
            budget_bytes: budget_mib
                .checked_mul(1 << 20)
                .ok_or_else(|| anyhow!("--budget-mib {budget_mib} is too large"))?,
            adapter,
            size: size.unwrap_or(default_size),
            mode,
            record,
            gpu_timing,
            script_path: path.unwrap_or_default(),
        })
    }

    /// A `--frames` run's frame count.
    pub(crate) const fn frames(&self) -> Option<u64> {
        match self.mode.run {
            Run::Frames(frames) => Some(frames),
            Run::Open | Run::Script(_) => None,
        }
    }

    /// A `--script` run's camera path.
    pub(crate) fn script(&self) -> Option<Script> {
        match self.mode.run {
            Run::Script(speed) => Some(Script::new(self.script_path, speed)),
            Run::Open | Run::Frames(_) => None,
        }
    }

    /// Whether the run times the GPU: a script's, unless `--no-gpu-timing`.
    pub(crate) const fn times_gpu(&self) -> bool {
        self.gpu_timing && matches!(self.mode.run, Run::Script(_))
    }

    /// What a recorded run's record says of its setting, the GPU timed or not (`gpu_timing`,
    /// as the render thread found), the project beginning to open at `opening`; `None` without
    /// `--record`. Whether the level choice's stages are timed is found as the frame loop starts
    /// ([`crate::start`]).
    pub(crate) fn record_setup(&self, gpu_timing: bool, opening: Instant) -> Option<RecordSetup> {
        self.record.as_ref().map(|_| RecordSetup {
            project: self.project.display().to_string(),
            gpu_timing,
            lod_timing: false,
            opening,
        })
    }
}

/// The value of `flag` if `arg` is it, as `flag VALUE` (taking the next argument) or
/// `flag=VALUE`.
///
/// # Errors
///
/// If `arg` is `flag` and no argument follows.
fn flag_value(
    arg: &OsStr,
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

/// `value` as text.
///
/// # Errors
///
/// If it isn't UTF-8.
fn text<'a>(value: &'a OsStr, flag: &str) -> Result<&'a str> {
    value
        .to_str()
        .ok_or_else(|| anyhow!("{flag}: {} isn't text", value.display()))
}

/// `value` parsed as a number.
///
/// # Errors
///
/// If it isn't text or doesn't parse.
fn number<T: FromStr>(value: &OsStr, flag: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let text = text(value, flag)?;
    text.parse()
        .with_context(|| format!("{flag} {text} isn't a number"))
}

/// A size written `WxH`, both non-zero.
///
/// # Errors
///
/// If it isn't two numbers around an `x`, or either is zero.
fn parse_size(text: &str) -> Result<Size> {
    let (width, height) = text
        .split_once('x')
        .ok_or_else(|| anyhow!("--size {text} isn't WIDTHxHEIGHT"))?;
    let width: u32 = width
        .parse()
        .with_context(|| format!("--size {text}: the width"))?;
    let height: u32 = height
        .parse()
        .with_context(|| format!("--size {text}: the height"))?;
    Size::new(width, height).ok_or_else(|| anyhow!("--size {text} is empty"))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Parses `args` as the arguments after the program name.
    ///
    /// # Errors
    ///
    /// As [`Options::parse`].
    fn parse(args: &[&str]) -> Result<Options> {
        Options::parse(args.iter().map(OsString::from))
    }

    #[test]
    fn a_project_alone_takes_every_default() {
        let options = parse(&["p.bcrf"]).unwrap();
        assert_eq!(options.project, PathBuf::from("p.bcrf"));
        assert_eq!(options.level, Level::new(9).unwrap());
        assert_eq!(options.budget_bytes, 2048 << 20);
        assert_eq!(options.adapter, AdapterRequest::Default);
        assert_eq!(options.size, Size::new(1280, 800).unwrap());
        assert_eq!(options.mode, mode(Run::Open, false));
        assert_eq!((options.record, options.gpu_timing), (None, true));
    }

    #[test]
    fn every_option_in_either_form() {
        let options = parse(&[
            "--level=6",
            "--budget-mib",
            "512",
            "--adapter",
            "software",
            "--size=640x480",
            "--frames",
            "10",
            "--offscreen",
            "p.bcrf",
        ])
        .unwrap();
        assert_eq!(options.level, Level::new(6).unwrap());
        assert_eq!(options.budget_bytes, 512 << 20);
        assert_eq!(options.adapter, AdapterRequest::Software);
        assert_eq!(options.size, Size::new(640, 480).unwrap());
        assert_eq!(options.mode, mode(Run::Frames(10), true));
        assert_eq!(
            parse(&["p.bcrf", "--adapter=4090"]).unwrap().adapter,
            AdapterRequest::Named("4090".to_owned())
        );
        assert_eq!(
            parse(&["p.bcrf", "--frames=3"]).unwrap().mode,
            mode(Run::Frames(3), false)
        );
    }

    #[test]
    fn bad_arguments_are_refused() {
        for args in [
            &[][..],
            &["a.bcrf", "b.bcrf"],
            &["p.bcrf", "--verbose"],
            &["p.bcrf", "--level"],
            &["p.bcrf", "--level", "nine"],
            &["p.bcrf", "--level", "99"],
            &["p.bcrf", "--size", "640"],
            &["p.bcrf", "--size", "0x480"],
            &["p.bcrf", "--frames", "0"],
            &["p.bcrf", "--offscreen"],
            &["p.bcrf", "--script", "--frames", "3"],
            &["p.bcrf", "--record", "r.jsonl"],
            &["p.bcrf", "--frames", "3", "--script-speed", "2"],
            &["p.bcrf", "--script", "--script-speed", "0"],
            &["p.bcrf", "--script-path", "strokes"],
            &["p.bcrf", "--script", "--script-path", "orbit"],
        ] {
            assert!(parse(args).is_err(), "{args:?} refused");
        }
    }

    #[test]
    fn a_script_runs_offscreen_or_in_a_window_and_may_record() {
        let options = parse(&[
            "p.bcrf",
            "--script",
            "--offscreen",
            "--script-speed=2.5",
            "--record",
            "/tmp/r.jsonl",
            "--no-gpu-timing",
            "--adapter",
            "hardware",
        ])
        .unwrap();
        assert_eq!(
            options.mode,
            mode(Run::Script("2.5".parse().unwrap()), true)
        );
        assert_eq!(options.record, Some(PathBuf::from("/tmp/r.jsonl")));
        assert!(!options.gpu_timing);
        assert_eq!(options.adapter, AdapterRequest::Hardware);
        assert_eq!(
            parse(&["p.bcrf", "--script"]).unwrap().mode,
            mode(Run::Script(Speed::ONE), false),
            "in a window, at real speed"
        );
        assert_eq!(options.script_path, ScriptPath::Benchmark, "by default");
        assert_eq!(
            options.script(),
            Some(Script::benchmark("2.5".parse().unwrap()))
        );
        let strokes = parse(&["p.bcrf", "--script", "--script-path=strokes"]).unwrap();
        assert_eq!(strokes.script(), Some(Script::strokes(Speed::ONE)));
    }

    /// The mode `run`, offscreen or not.
    const fn mode(run: Run, offscreen: bool) -> Mode {
        Mode { run, offscreen }
    }
}
