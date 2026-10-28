//! `--guided`: the whole W-against-U comparison in one command.
//!
//! The compositor delivers tablet events to one binding per client, so winit's and our
//! backend's can't be recorded in one process at once; and winit can't make a second event
//! loop in one process. So this command runs each recording as a child process of its own
//! (`pen-probe --source S --load --script …`), one after the other, asking before each:
//!
//! 1. it makes a scratch directory outside the repository for the CSVs (by default
//!    `$XDG_CACHE_HOME/brushcraft/pen-probe/run-<unix time>`);
//! 2. it prepares the load's project once: a reflinked copy of the benchmark project, beside it,
//!    with its display layer built and saved ([`load::prepare`]), so neither run rebuilds it;
//! 3. it runs winit's recording, then ours, each with the load and the guided script
//!    ([`crate::script`]): `winit.csv` and `ours.csv`, each with its `-load.csv`;
//! 4. it prints each run's summary and the comparison ([`report::compare`]), and writes them to
//!    `report.txt` beside the CSVs;
//! 5. the prepared copy goes with its work directory; the CSVs stay.

use std::fmt::Write as _;
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use geom::Level;

use crate::load::{self, WorkDir};
use crate::script::Step;
use crate::{Source, report, say};

/// The default scratch directory for a run's CSVs.
///
/// # Errors
///
/// If there is no cache directory (see [`load::cache_root`]).
fn default_out() -> Result<PathBuf> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();
    Ok(load::cache_root()?
        .join("pen-probe")
        .join(format!("run-{secs}")))
}

/// Waits for the user to press Enter.
///
/// # Errors
///
/// If standard input can't be read.
fn wait_for_enter() -> Result<()> {
    drop(std::io::stdout().lock().flush());
    std::io::stdin()
        .lock()
        .read_line(&mut String::new())
        .context("reading standard input")?;
    Ok(())
}

/// Runs the comparison: see the module notes. `project` is the benchmark project the load
/// copies, `level` the level it brushes, `out` where the CSVs go (the default if `None`).
///
/// # Errors
///
/// If the scratch directory is inside the repository or can't be made, the project can't be
/// prepared, a child can't be started, or a CSV can't be read back.
pub(crate) fn run(project: &Path, level: Level, out: Option<PathBuf>) -> Result<()> {
    let out = match out {
        Some(out) => std::path::absolute(&out)
            .with_context(|| format!("making {} absolute", out.display()))?,
        None => default_out()?,
    };
    load::ensure_outside_repo(&out)?;
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let per_run = Step::total().as_secs();
    say(format_args!(
        "pen-probe --guided: two recordings, winit's tablet events (W) and then our backend (U), \
         each under the load. Each takes {} min {} s once the pen comes over its window; the \
         prompts appear here and in the window's title. CSVs go to {}.",
        per_run / 60,
        per_run % 60,
        out.display()
    ));
    say(
        "Preparing the load's project (a reflinked copy; the first time, its display layer is built and saved).",
    );
    let work = WorkDir::beside(project)?;
    let prepared = load::prepare(project, work.path(), level)?;
    let exe = std::env::current_exe().context("finding the probe's own binary")?;
    let mut csvs = Vec::new();
    for (n, source) in Source::ALL.into_iter().enumerate() {
        let csv = out.join(format!("{}.csv", source.name()));
        say(format_args!(
            "\nRun {} of 2: {}. Press Enter to open the window, then bring the pen over it.",
            n + 1,
            match source {
                Source::Winit => "W, winit's tablet events",
                Source::Ours => "U, our Wayland backend",
            }
        ));
        wait_for_enter()?;
        let status = Command::new(&exe)
            .arg("--source")
            .arg(source.name())
            .arg("--load")
            .arg("--script")
            .arg("--project")
            .arg(&prepared)
            .arg("--level")
            .arg(level.get().to_string())
            .arg(&csv)
            .status()
            .with_context(|| format!("starting the {} recording", source.name()))?;
        if !status.success() {
            say(format_args!(
                "the {} recording ended badly ({status}); its CSV is used if it was written",
                source.name()
            ));
        }
        csvs.push(csv);
    }
    drop(work);

    let mut text = String::new();
    let mut recordings = Vec::new();
    for csv in &csvs {
        if !csv.exists() {
            say(format_args!("{} wasn't written", csv.display()));
            continue;
        }
        let recording = report::read_csv(csv)?;
        writeln!(text, "== {} ==", csv.display())?;
        text.push_str(&report::summarise(&recording)?);
        recordings.push(recording);
    }
    if let [winit, ours] = recordings.as_slice() {
        text.push_str(&report::compare::compare(winit, ours)?);
    }
    let path = out.join("report.txt");
    std::fs::write(&path, &text).with_context(|| format!("writing {}", path.display()))?;
    say(format_args!("\n{text}report: {}", path.display()));
    Ok(())
}
