//! `perf`: the tracked performance suite, so timings are watched as the code changes instead of
//! re-measured by hand.
//!
//! A run measures three things, each part in child processes of this binary, behind the quiet
//! guard ([`quiet`]):
//!
//! - **the stage benches**: `geom`'s divan benchmark of each subdivision stage on a level-9 tile,
//!   interior and face-corner, pinned to one core, five rounds ([`stages`]);
//! - **the medium project**: the benchmark sphere's shape at a sixteenth of its size, generated in
//!   seconds and cached ([`project`]): brush latency and phases, the cold first dab, index ready on
//!   open, page-in of a footprint and a region with a cold and a warm pool, the save cost model's
//!   fixed and per-vertex parts, and peak RSS under a 1 GiB budget, each from five runs
//!   ([`suite`]);
//! - with `--full`, **the full project** (the benchmark sphere, measured through a reflinked copy):
//!   the same set, for milestones.
//!
//! `--tiny` swaps the medium project for a tiny one and turns the guard off: the suite's check
//! that it runs end to end (CI, tests), whose timings are never recorded.
//!
//! Each metric prints as a JSON line (`{"bench":"perf","result":"metric",…}`) and, with
//! `--history DIR`, is appended to `DIR/<machine>.jsonl` ([`history`]). The run is compared
//! against the latest baseline of each metric on the same hardware, and a change beyond the
//! metric's noise is flagged ([`compare`]); a flagged slowdown makes the exit status 3, and
//! failing that a comparison too noisy to conclude makes it 4.
//! `--baseline` marks the run as the new baseline, which a loaded run can't be, nor one whose
//! binary wasn't built from a clean tree at the commit checked out. Records name the commit the
//! binary was built from (embedded by `build.rs`).

mod compare;
mod history;
mod json;
mod machine;
mod metric;
mod project;
mod quiet;
mod stages;
mod suite;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result, bail, ensure};

use crate::cli::{Options, Projects};
use crate::paths::{Paths, cache_root, repo_root};
use crate::perf::compare::{Target, Verdict};
use crate::perf::history::{Entry, Label, PartLoad, RunInfo};
use crate::perf::machine::{Machine, Revision, Timestamp};
use crate::perf::metric::Metric;
use crate::perf::project::{
    Copied, Scale, Source, Sweep, WorkDir, WorkRoot, cache_key, cached_file, remove_stale,
};
use crate::perf::quiet::{Load, LoadLog, Quiet, Settled, others_peak, own_load};
use crate::perf::suite::Runner;
use crate::report::Output;

/// Rounds of the stage benches: each takes a fraction of a second.
const STAGE_ROUNDS: usize = 5;

/// How a run ended, for the exit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// Nothing flagged, nothing inconclusive.
    Clean,
    /// At least one metric got worse beyond its noise.
    Flagged,
    /// None got worse, but at least one was too noisy to compare (see
    /// [`compare::MAX_THRESHOLD`]).
    Inconclusive,
}

/// What `perf` is asked to do, checked.
#[derive(Debug, Clone, PartialEq)]
struct Plan {
    /// The projects measured, in order.
    scales: Vec<Scale>,
    /// Whether the stage benches run.
    stages: bool,
    /// The history directory, if the run is recorded.
    history: Option<PathBuf>,
    /// Whether the run should become the baseline.
    baseline: bool,
    /// The quiet guard, `None` when off.
    quiet: Option<Quiet>,
    /// The full project's file.
    full_project: Option<PathBuf>,
}

impl Plan {
    /// The plan `options` ask for, with the full project at `paths`.
    ///
    /// # Errors
    ///
    /// If the options don't go together: `--tiny` with `--history` or `--baseline`;
    /// `--baseline` without `--history`.
    fn new(options: &Options, paths: &Paths) -> Result<Self> {
        let tiny = options.projects == Projects::Tiny;
        if tiny {
            ensure!(
                options.history.is_none() && !options.baseline,
                "--tiny checks the suite runs; its timings are never recorded, so it takes \
                 neither --history nor --baseline"
            );
        }
        ensure!(
            !options.baseline || options.history.is_some(),
            "--baseline needs --history: a baseline is a history record"
        );
        let scales = match options.projects {
            Projects::Tiny => vec![Scale::Tiny],
            Projects::Medium => vec![Scale::Medium],
            Projects::MediumAndFull => vec![Scale::Medium, Scale::Full],
        };
        Ok(Self {
            scales,
            stages: options.stages,
            history: options.history.clone(),
            baseline: options.baseline,
            quiet: (!tiny).then(|| Quiet::new(Duration::from_secs(options.settle_secs))),
            full_project: (options.projects == Projects::MediumAndFull)
                .then(|| paths.project.clone()),
        })
    }
}

/// A run in progress: the child processes, the load log, and how quiet it has been.
#[derive(Debug)]
struct Session {
    /// Runs the parts' child processes.
    runner: Runner,
    /// The quiet guard, `None` when off.
    quiet: Option<Quiet>,
    /// The load log.
    log: LoadLog,
    /// The run's label so far.
    label: Label,
    /// Every metric measured so far, with its project and its part's load.
    measured: Vec<(String, PartLoad, Metric)>,
}

impl Session {
    /// Runs one measured part, `measure`, after the quiet guard, and keeps its metrics under
    /// `project`.
    ///
    /// Once a part found the machine loaded, later parts don't wait: the run is labelled
    /// loaded already, and waiting again would only lengthen it.
    ///
    /// # Errors
    ///
    /// If the guard can't read the load, or `measure` fails.
    fn part<J: Write, H: Write>(
        &mut self,
        name: &str,
        project: &str,
        out: &mut Output<J, H>,
        measure: impl FnOnce(&mut Runner) -> Result<Vec<Metric>>,
    ) -> Result<()> {
        let others = |runner: &Runner| -> Result<Load> {
            Ok(Load::now()?.less(own_load(&runner.busy, Instant::now())))
        };
        if let (Some(quiet), Label::Quiet) = (self.quiet, self.label) {
            let settled = quiet.settle(
                || others(&self.runner),
                std::thread::sleep,
                |line| out.say(&format!("perf: {name}: {line}")),
            )?;
            self.label = self.label.after(settled);
        }
        let start = Instant::now();
        let (load, others_start) = (Load::now()?, others(&self.runner)?);
        out.say(&format!(
            "perf: {name} (load {load}, others {others_start})"
        ))?;
        let metrics = measure(&mut self.runner).with_context(|| format!("the {name} part"))?;
        // Load that arrived while the part ran: each sample less the suite's own share, now that
        // every stretch of the part's work is known, and the others' load as it ended.
        let at_end = others(&self.runner)?;
        let peak = others_peak(&self.log.since(start), &self.runner.busy)
            .map_or(at_end, |peak| if peak > at_end { peak } else { at_end });
        if let Some(quiet) = self.quiet {
            let held = quiet.held(peak);
            if let Settled::Loaded { .. } = held {
                out.say(&format!(
                    "perf: {name}: other load reached {peak} during the part: the run is labelled \
                     loaded"
                ))?;
            }
            self.label = self.label.after(held);
        }
        let load = PartLoad {
            start: load,
            others: others_start,
            others_peak: peak,
            max: self
                .log
                .max_since(start)
                .map_or(load, |sampled| if sampled > load { sampled } else { load }),
        };
        self.measured
            .extend(metrics.into_iter().map(|m| (project.to_owned(), load, m)));
        Ok(())
    }
}

/// Says on `out` what a clear-up removed (each as `removed`, e.g. "a dead run's"), and warns of
/// what it couldn't.
///
/// # Errors
///
/// If writing fails.
fn report_sweep<J: Write, H: Write>(
    sweep: Sweep,
    removed: &str,
    out: &mut Output<J, H>,
) -> Result<()> {
    for path in sweep.removed {
        out.say(&format!("perf: removed {removed} {}", path.display()))?;
    }
    for (path, why) in sweep.skipped {
        out.say(&format!(
            "perf: warning: left {} in place: {why:#}",
            path.display()
        ))?;
    }
    Ok(())
}

/// A fresh work directory in this user's work root under `base`, after sweeping dead runs'
/// from that root, saying on `out` what was removed and warning of what couldn't be.
///
/// # Errors
///
/// If the root can't be listed, the directory can't be made, or writing fails.
fn work_dir<J: Write, H: Write>(base: &Path, out: &mut Output<J, H>) -> Result<WorkDir> {
    let root = WorkRoot::under(base);
    report_sweep(WorkDir::sweep(&root)?, "a dead run's", out)?;
    WorkDir::new(&root)
}

/// A copy of `source` as project `name` in `work`, saying on `out` when it couldn't be a
/// reflink.
///
/// # Errors
///
/// As [`WorkDir::copy`], or if writing fails.
fn copy<J: Write, H: Write>(
    work: &WorkDir,
    source: &Source,
    name: &str,
    out: &mut Output<J, H>,
) -> Result<PathBuf> {
    let (file, copied) = work.copy(source, name)?;
    if copied == Copied::InFull {
        out.say(&format!(
            "perf: {} couldn't be reflinked: copied in full",
            source.path().display()
        ))?;
    }
    Ok(file)
}

/// The medium project, from the cache under `cache` or freshly generated into it (see
/// [`project::cache_key`]), open and locked shared so no other run removes it while it is
/// copied; returns it and its key.
///
/// A freshly generated project is opened and locked before it is moved into the cache, so it is
/// never there unlocked before this run has copied it.
///
/// # Errors
///
/// If generating it, or tidying the cache, fails.
fn medium<J: Write, H: Write>(
    runner: &mut Runner,
    cache: &Path,
    out: &mut Output<J, H>,
) -> Result<(Source, String)> {
    let shape = Scale::Medium
        .shape()
        .context("the medium project has a shape")?;
    let key = cache_key(shape);
    let file = cached_file(cache, &key);
    let source = if let Some(source) = Source::cached(&file)? {
        out.say(&format!("perf: medium project cached: {}", file.display()))?;
        source
    } else {
        out.say(&format!(
            "perf: generating the medium project: {}",
            file.display()
        ))?;
        let building = work_dir(cache, out)?;
        let fresh = building.project("medium")?;
        let start = Instant::now();
        suite::generate(runner, &fresh, Scale::Medium)?;
        let source = Source::cached(&fresh)?
            .with_context(|| format!("{} was just generated", fresh.display()))?;
        std::fs::rename(&fresh, &file)
            .with_context(|| format!("moving {} to {}", fresh.display(), file.display()))?;
        out.say(&format!(
            "perf: generated in {:.1} s",
            start.elapsed().as_secs_f64()
        ))?;
        source
    };
    report_sweep(remove_stale(cache, &file)?, "the stale", out)?;
    Ok((source, format!("medium-{key}")))
}

/// Measures every part of `scale` on the project at `file` (named `project` in the records).
///
/// The save part goes last: it adds save points to the project.
///
/// # Errors
///
/// If a part fails.
fn measure_project<J: Write, H: Write>(
    session: &mut Session,
    scale: Scale,
    file: &Path,
    project: &str,
    out: &mut Output<J, H>,
) -> Result<()> {
    let s = scale.name();
    session.part(&format!("{s} brush"), project, out, |r| {
        suite::brush_part(r, file, scale)
    })?;
    session.part(&format!("{s} page-in"), project, out, |r| {
        suite::page_in_part(r, file, scale)
    })?;
    session.part(&format!("{s} budget"), project, out, |r| {
        suite::budget_part(r, file, scale)
    })?;
    session.part(&format!("{s} save"), project, out, |r| {
        suite::save_part(r, file, scale)
    })
}

/// The comparison table's line for `entry` judged `verdict` against `baseline`.
fn table_line(entry: &Entry, baseline: Option<&Entry>, verdict: Verdict, target: Target) -> String {
    let m = &entry.metric;
    let now = format!(
        "{:<40} {:>10.3} {:<3} ±{:>4.1} %",
        m.name.as_str(),
        m.value.median(),
        m.unit.name(),
        m.value.spread() * 100.0
    );
    let was = baseline.map(|base| format!("was {:>10.3}", base.metric.value.median()));
    let change = verdict.change().map(|change| change.to_string());
    let ceiling = match (target, m.ceiling) {
        (Target::Missed { by }, Some(ceiling)) => Some(format!(
            "OVER TARGET by {by:.1} {} (ceiling {ceiling:.0})",
            m.unit.name()
        )),
        (Target::Met, Some(ceiling)) => Some(format!("within target ({ceiling:.0})")),
        (Target::Unset | Target::Met | Target::Missed { .. }, _) => None,
    };
    [
        Some(now),
        was,
        change,
        Some(verdict.word().to_owned()),
        ceiling,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("  ")
}

/// The history file of `machine` in `dir`: `<dir>/<machine name>.jsonl`.
///
/// `dir` itself is made if missing (the first run on a new docs checkout), but never its
/// parent: a mistyped `--history` or `BRUSHCRAFT_DOCS` should fail, not start a history
/// nobody reads.
///
/// # Errors
///
/// If `dir`'s parent doesn't exist, or `dir` can't be made.
fn history_file(dir: &Path, machine: &Machine) -> Result<PathBuf> {
    let parent = dir.parent().unwrap_or(dir);
    ensure!(
        parent.is_dir(),
        "{} doesn't exist: check --history (scripts/perf.sh sets it from BRUSHCRAFT_DOCS, \
         default ~/Code/brushcraft-docs)",
        parent.display()
    );
    // The parent exists, so this makes at most `dir` itself.
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir.join(format!("{}.jsonl", machine.name)))
}

/// The entries of the history file `file`, warning on `out` about a torn last line.
///
/// # Errors
///
/// As [`history::read`], or if writing fails.
fn read_history<J: Write, H: Write>(file: &Path, out: &mut Output<J, H>) -> Result<Vec<Entry>> {
    let history = history::read(file)?;
    if let Some(line) = history.torn {
        out.say(&format!(
            "perf: warning: {} line {line} has no newline, a write cut short: left out, and \
             dropped when this run appends",
            file.display()
        ))?;
    }
    Ok(history.entries)
}

/// Prints each of `entries` as a JSON line and a line of the comparison table, judged against
/// the latest baselines in `past` on the same hardware; returns how many got worse and how many
/// were inconclusive.
///
/// # Errors
///
/// If writing fails.
fn report<J: Write, H: Write>(
    entries: &[Entry],
    past: &[Entry],
    label: Label,
    out: &mut Output<J, H>,
) -> Result<(usize, usize)> {
    let baselines = entries
        .first()
        .map(|e| history::baselines(past, &e.info))
        .unwrap_or_default();
    out.say(&format!(
        "\nperf: {} metrics, run labelled {label}",
        entries.len()
    ))?;
    let (mut flagged, mut noisy) = (0, 0);
    for entry in entries {
        out.record(&entry.to_record())?;
        let base = baselines.get(&entry.metric.name).copied();
        let verdict = Verdict::judge(base.map(|b| &b.metric.value), &entry.metric.value);
        match verdict {
            Verdict::Worse(_) => flagged += 1,
            Verdict::Noisy(_) => noisy += 1,
            Verdict::New | Verdict::Same(_) | Verdict::Better(_) => {}
        }
        let target = Target::check(&entry.metric);
        if let Target::Missed { .. } = target {
            flagged += 1;
        }
        out.say(&table_line(entry, base, verdict, target))?;
    }
    Ok((flagged, noisy))
}

/// Runs `perf` (see the module notes); `paths.project` is the full project for `--full`.
///
/// # Errors
///
/// If the options don't go together, `--baseline` meets a binary that doesn't match a clean
/// tree, the history can't be read
/// or written, a part fails, or `--baseline` was asked of a run that turned out loaded (its
/// records are written, not as a baseline).
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<Status> {
    let plan = Plan::new(options, paths)?;
    let started = SystemTime::now();
    let repo = repo_root();
    let machine = Machine::this()?;
    // The records name the binary's revision; the tree's only checks it.
    let revision = Revision::built();
    if let Some(mismatch) = revision.mismatch(&Revision::of(&repo)) {
        ensure!(
            !plan.baseline,
            "--baseline needs a binary built from a clean tree at the commit checked out: \
             {mismatch}"
        );
        out.say(&format!(
            "perf: warning: the binary doesn't vouch for the code measured: {mismatch}"
        ))?;
    }
    let history_file = plan
        .history
        .as_deref()
        .map(|dir| history_file(dir, &machine))
        .transpose()?;
    let past = match &history_file {
        Some(file) => read_history(file, out)?,
        None => Vec::new(),
    };
    out.say(&format!(
        "perf: {} at {}{} on {} ({})",
        plan.scales
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(" + "),
        revision.commit,
        if revision.dirty { " (dirty)" } else { "" },
        machine.name,
        machine.fingerprint
    ))?;

    let mut session = Session {
        runner: Runner::new(options.workers, &repo)?,
        quiet: plan.quiet,
        log: LoadLog::start()?,
        label: if plan.quiet.is_some() {
            Label::Quiet
        } else {
            Label::Unchecked
        },
        measured: Vec::new(),
    };
    let measured = measure_all(&plan, &mut session, out);
    let Session {
        log,
        label,
        measured: metrics,
        ..
    } = session;
    log.stop()?;
    measured?;

    let baseline = plan.baseline && label.allows_baseline();
    let info = RunInfo {
        run: Timestamp::of(started)?,
        commit: revision.commit,
        dirty: revision.dirty,
        machine: machine.name,
        fingerprint: machine.fingerprint,
        kernel: machine.kernel,
        workers: options.workers,
        worker_threads: options.workers.count().get(),
        label,
        baseline,
    };
    let entries: Vec<Entry> = metrics
        .into_iter()
        .map(|(project, load, metric)| Entry {
            info: info.clone(),
            project,
            load,
            metric,
        })
        .collect();
    let (flagged, noisy) = report(&entries, &past, label, out)?;
    if let Some(file) = &history_file {
        let dropped = history::append(file, &entries)?;
        if dropped > 0 {
            out.say(&format!(
                "perf: dropped {dropped} bytes of a torn last line from {}",
                file.display()
            ))?;
        }
        out.say(&format!(
            "perf: appended {} records to {}{}",
            entries.len(),
            file.display(),
            if baseline { " as the new baseline" } else { "" }
        ))?;
    }
    if plan.baseline && !baseline {
        bail!(
            "the run was {label}, so it was recorded but not as a baseline: rerun on a quiet machine"
        );
    }
    status(flagged, noisy, out)
}

/// The run's status from how many metrics got worse (`flagged`) and how many were too noisy to
/// compare (`noisy`), said to `out`.
///
/// # Errors
///
/// If writing fails.
fn status<J: Write, H: Write>(
    flagged: usize,
    noisy: usize,
    out: &mut Output<J, H>,
) -> Result<Status> {
    if flagged > 0 {
        out.say(&format!(
            "perf: {flagged} metrics got worse beyond their noise"
        ))?;
        return Ok(Status::Flagged);
    }
    if noisy > 0 {
        out.say(&format!(
            "perf: {noisy} metrics too noisy to compare (a noise band over 25 %): rerun quieter"
        ))?;
        return Ok(Status::Inconclusive);
    }
    Ok(Status::Clean)
}

/// Measures every part of `plan` into `session`.
///
/// # Errors
///
/// If a project can't be prepared or a part fails.
fn measure_all<J: Write, H: Write>(
    plan: &Plan,
    session: &mut Session,
    out: &mut Output<J, H>,
) -> Result<()> {
    if plan.stages {
        session.part("stage benches", "none", out, |r| {
            suite::stage_part(r, STAGE_ROUNDS)
        })?;
    }
    for &scale in &plan.scales {
        match scale {
            Scale::Tiny => {
                let work = work_dir(&std::env::temp_dir(), out)?;
                let file = work.project("tiny")?;
                suite::generate(&mut session.runner, &file, scale)?;
                measure_project(session, scale, &file, "tiny", out)?;
            }
            Scale::Medium => {
                let cache = cache_root()?.join("perf");
                std::fs::create_dir_all(&cache)
                    .with_context(|| format!("creating {}", cache.display()))?;
                let (cached, key) = medium(&mut session.runner, &cache, out)?;
                let work = work_dir(&cache, out)?;
                let file = copy(&work, &cached, "medium", out)?;
                // The run measures its own copy: other runs may tidy the cache from here on.
                drop(cached);
                measure_project(session, scale, &file, &key, out)?;
            }
            Scale::Full => {
                let source = plan
                    .full_project
                    .as_deref()
                    .context("the full project's path")?;
                ensure!(
                    source.exists(),
                    "{} doesn't exist: run `brushcraft-bench generate` first",
                    source.display()
                );
                let bytes = std::fs::metadata(source)
                    .with_context(|| format!("reading {}'s size", source.display()))?
                    .len();
                let name = format!(
                    "{} ({bytes} bytes)",
                    source.file_name().unwrap_or_default().to_string_lossy()
                );
                let dir = source.parent().context("the full project's directory")?;
                let work = work_dir(dir, out)?;
                let file = copy(&work, &Source::open(source)?, "full", out)?;
                measure_project(session, scale, &file, &name, out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::cli::Command;
    use crate::perf::history::tests::entry;

    /// The plan the command line `args` (split on spaces) asks for.
    ///
    /// # Errors
    ///
    /// As [`Command::parse`] and [`Plan::new`].
    fn plan(args: &str) -> Result<Plan> {
        let command = Command::parse(args.split_whitespace().map(std::ffi::OsString::from))?;
        let paths = Paths {
            project: PathBuf::from("/c/sphere.bcrf"),
            recovery: PathBuf::from("/c/recovery"),
        };
        Plan::new(&command.options, &paths)
    }

    #[test]
    fn plans_follow_the_options() {
        let medium = plan("perf --history /d").unwrap();
        assert_eq!(medium.scales, [Scale::Medium], "medium by default");
        assert!(medium.stages && medium.quiet.is_some(), "stages, guarded");
        assert_eq!(medium.full_project, None, "no full project");
        let full = plan("perf --full --baseline --history /d --settle-secs 60").unwrap();
        assert_eq!(
            full.scales,
            [Scale::Medium, Scale::Full],
            "medium, then full"
        );
        assert_eq!(
            full.full_project,
            Some(PathBuf::from("/c/sphere.bcrf")),
            "the bench's project"
        );
        assert_eq!(
            full.quiet.map(|q| q.within),
            Some(Duration::from_secs(60)),
            "the bound"
        );
        let tiny = plan("perf --tiny --no-stages").unwrap();
        assert_eq!(tiny.scales, [Scale::Tiny], "tiny");
        assert!(!tiny.stages && tiny.quiet.is_none(), "no stages, no guard");
        for bad in [
            "perf --tiny --history /d",
            "perf --tiny --baseline",
            "perf --baseline",
        ] {
            assert!(plan(bad).is_err(), "{bad}");
        }
    }

    /// A slowdown outranks an inconclusive comparison, which outranks a clean run: each has its
    /// own exit status.
    #[test]
    fn statuses_rank_slowdowns_over_noise() {
        let mut out = Output::new(Vec::new(), Vec::new());
        assert_eq!(status(1, 1, &mut out).unwrap(), Status::Flagged, "worse");
        assert_eq!(
            status(0, 1, &mut out).unwrap(),
            Status::Inconclusive,
            "noisy"
        );
        assert_eq!(status(0, 0, &mut out).unwrap(), Status::Clean, "clean");
    }

    /// The history directory is made only inside an existing parent.
    #[test]
    fn history_directories_need_their_parent() {
        let machine = Machine::this().unwrap();
        let root =
            std::env::temp_dir().join(format!("brushcraft-perf-docs-{}", std::process::id()));
        let missing = root.join("typo").join("perf");
        let err = history_file(&missing, &machine).unwrap_err();
        assert!(err.to_string().contains("doesn't exist"), "{err}");
        assert!(!missing.exists(), "nothing made");
        std::fs::create_dir_all(&root).unwrap();
        let dir = root.join("perf");
        let file = history_file(&dir, &machine).unwrap();
        assert!(dir.is_dir(), "made inside an existing parent");
        assert_eq!(
            file,
            dir.join(format!("{}.jsonl", machine.name)),
            "named for the machine"
        );
        assert!(
            history_file(&dir, &machine).is_ok(),
            "again, once it exists"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn table_lines_show_value_baseline_and_verdict() {
        let now = entry(
            "medium.brush.p99_ms",
            2.2,
            "2026-10-24T00:00:02Z",
            false,
            "cpu",
        );
        let base = entry(
            "medium.brush.p99_ms",
            2.0,
            "2026-10-24T00:00:01Z",
            true,
            "cpu",
        );
        let verdict = Verdict::judge(Some(&base.metric.value), &now.metric.value);
        let line = table_line(&now, Some(&base), verdict, Target::Unset);
        assert!(line.starts_with("medium.brush.p99_ms"), "{line}");
        assert!(
            line.contains("was      2.000") && line.ends_with("WORSE"),
            "{line}"
        );
        let new = table_line(&now, None, Verdict::New, Target::Unset);
        assert!(new.ends_with("new") && !new.contains("was"), "{new}");
        let mut capped = now;
        capped.metric.ceiling = Some(2.0);
        let target = Target::check(&capped.metric);
        let over = table_line(&capped, None, Verdict::New, target);
        assert!(over.contains("OVER TARGET by 0.2 ms (ceiling 2)"), "{over}");
    }
}
