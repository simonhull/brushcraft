//! The command line: the project to open and how to run.
//!
//! Hand-parsed from `std::env::args`, as the benchmark harness does: `brushcraft PROJECT
//! [--flag VALUE | --flag=VALUE | --switch]…`. Every option has a default, and an unknown one
//! is refused, so a typo never runs with a setting silently ignored.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use geom::Level;
use render::gpu::AdapterRequest;
use render::surface::Size;

/// What the app prints when its arguments don't parse.
pub(crate) const USAGE: &str = "\
usage: brushcraft PROJECT [OPTIONS]

Opens the project (a .bcrf file outside the source repository) in a window and draws it.
Drag with the left button or the pen to orbit, with Shift or the right or middle button to
pan, with Ctrl to zoom; the wheel zooms too.

options:
  --level N        the finest level of the display layer the store keeps (default 9)
  --budget-mib N   the tile store's memory budget (default 2048)
  --adapter A      the GPU: default (the system's high-performance one), software (a CPU
                   rasterizer such as lavapipe), or any other text, matched against the
                   adapters' names
  --size WxH       the window's size, or the offscreen frame's (default 1280x800)
  --frames N       draw N frames, every one, then close: a scripted run
  --offscreen      no window: draw into a texture (needs --frames)";

/// How the app runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// In a window, until it is closed or, given a count, until that many frames are drawn.
    Window {
        /// A scripted run's frame count.
        frames: Option<u64>,
    },
    /// With no window, drawing `frames` frames into a texture, then closing.
    Offscreen {
        /// How many frames to draw.
        frames: u64,
    },
}

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// project or more than one, or `--offscreen` without `--frames`.
    pub(crate) fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self> {
        let mut project = None;
        let mut level = DEFAULT_LEVEL;
        let mut budget_mib = DEFAULT_BUDGET_MIB;
        let mut adapter = AdapterRequest::Default;
        let mut size = None;
        let mut frames = None;
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
                    "software" => AdapterRequest::Software,
                    name => AdapterRequest::Named(name.to_owned()),
                };
            } else if let Some(value) = flag_value(&arg, "--size", &mut args)? {
                size = Some(parse_size(text(&value, "--size")?)?);
            } else if let Some(value) = flag_value(&arg, "--frames", &mut args)? {
                let count: u64 = number(&value, "--frames")?;
                ensure!(count > 0, "--frames needs at least 1; {USAGE}");
                frames = Some(count);
            } else if arg == "--offscreen" {
                offscreen = true;
            } else if arg.to_str().is_some_and(|a| a.starts_with('-')) {
                bail!("unknown option {}; {USAGE}", arg.display());
            } else if project.replace(PathBuf::from(arg)).is_some() {
                bail!("more than one project; {USAGE}");
            }
        }
        let project = project.ok_or_else(|| anyhow!("no project given; {USAGE}"))?;
        let mode = match (offscreen, frames) {
            (true, Some(frames)) => Mode::Offscreen { frames },
            (true, None) => bail!("--offscreen needs --frames: with no window, nothing closes it"),
            (false, frames) => Mode::Window { frames },
        };
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
        assert_eq!(options.mode, Mode::Window { frames: None });
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
        assert_eq!(options.mode, Mode::Offscreen { frames: 10 });
        assert_eq!(
            parse(&["p.bcrf", "--adapter=4090"]).unwrap().adapter,
            AdapterRequest::Named("4090".to_owned())
        );
        assert_eq!(
            parse(&["p.bcrf", "--frames=3"]).unwrap().mode,
            Mode::Window { frames: Some(3) }
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
        ] {
            assert!(parse(args).is_err(), "{args:?} refused");
        }
    }
}
