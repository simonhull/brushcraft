//! The suite's parts: each runs `brushcraft-bench` subcommands (or the stage benches) as child
//! processes and turns their JSON lines into metric samples.
//!
//! Every measurement runs in a process of its own, as the benchmark's recorded numbers were
//! taken: no part inherits another's heap, page faults or resident high-water mark.

use std::ffi::{OsStr, OsString};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, ensure};
use store::Workers;

use crate::perf::json::{Json, Object};
use crate::perf::machine::children_cpu;
use crate::perf::metric::{Measured, Metric, MetricName, Unit};
use crate::perf::project::Scale;
use crate::perf::quiet::Busy;
use crate::perf::stages;

/// Runs the suite's child processes and keeps the stretches of CPU work they did, for the
/// quiet guard to discount (see [`crate::perf::quiet::own_load`]).
#[derive(Debug)]
pub(crate) struct Runner {
    /// The `brushcraft-bench` binary: this one.
    exe: PathBuf,
    /// The evaluation workers setting every subcommand gets.
    workers: Workers,
    /// The repository, where the stage benches are built and run.
    repo: PathBuf,
    /// Every child's stretch of work so far.
    pub(crate) busy: Vec<Busy>,
}

/// The CPU the stage benches are pinned to: one core, away from CPU 0 (where interrupts tend to
/// land), and present on any machine with three or more.
const STAGE_CPU: usize = 2;

impl Runner {
    /// A runner of this binary's subcommands with `workers`, and of the stage benches in
    /// `repo`.
    ///
    /// # Errors
    ///
    /// If this binary's path can't be found.
    pub(crate) fn new(workers: Workers, repo: &Path) -> Result<Self> {
        Ok(Self {
            exe: std::env::current_exe().context("finding brushcraft-bench's own path")?,
            workers,
            repo: repo.to_owned(),
            busy: Vec::new(),
        })
    }

    /// Runs `program` with `args` (in `dir`, if given), its stderr passed through, and returns
    /// its stdout; records its stretch of work.
    ///
    /// # Errors
    ///
    /// If it can't be started, or fails.
    fn spawn(&mut self, program: &OsStr, args: &[OsString], dir: Option<&Path>) -> Result<String> {
        let shown = || {
            let words: Vec<String> = args
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            format!("{} {}", program.to_string_lossy(), words.join(" "))
        };
        let mut command = Command::new(program);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(dir) = dir {
            command.current_dir(dir);
        }
        let cpu_before = children_cpu()?;
        let start = Instant::now();
        let mut child = command
            .spawn()
            .with_context(|| format!("starting {}", shown()))?;
        let mut stdout = String::new();
        let read = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no stdout from {}", shown()))?
            .read_to_string(&mut stdout);
        let status = child
            .wait()
            .with_context(|| format!("waiting for {}", shown()))?;
        self.busy.push(Busy {
            start,
            end: Instant::now(),
            cpu: children_cpu()?.saturating_sub(cpu_before),
        });
        read.with_context(|| format!("reading {}'s output", shown()))?;
        ensure!(status.success(), "{} failed: {status}", shown());
        Ok(stdout)
    }

    /// Runs `brushcraft-bench <args> --workers <setting>` and returns its JSON lines.
    ///
    /// # Errors
    ///
    /// If it fails, or a line of its stdout isn't a JSON object.
    pub(crate) fn bench(&mut self, args: &[OsString]) -> Result<Vec<Object>> {
        let mut all = args.to_vec();
        all.extend(["--workers".into(), self.workers.to_string().into()]);
        let exe = self.exe.clone();
        let stdout = self.spawn(exe.as_os_str(), &all, None)?;
        stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Object::parse(line).with_context(|| format!("a result line: {line:?}")))
            .collect()
    }

    /// Runs the stage benches once, pinned to [`STAGE_CPU`] (or the last CPU, on a machine
    /// with fewer), through `taskset` and the `cargo` building this (`$CARGO`, else the one on
    /// the path), and returns divan's table.
    ///
    /// # Errors
    ///
    /// If `taskset` or `cargo` can't be run, or the benches fail.
    pub(crate) fn stage_benches(&mut self) -> Result<String> {
        let cpus = std::thread::available_parallelism()?.get();
        let cpu = STAGE_CPU.min(cpus - 1);
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let mut args: Vec<OsString> = vec!["-c".into(), cpu.to_string().into(), cargo];
        args.extend(
            [
                "bench",
                "-p",
                "geom",
                "--bench",
                "stages",
                "--features",
                "bench-stages",
            ]
            .map(OsString::from),
        );
        let repo = self.repo.clone();
        self.spawn(OsStr::new("taskset"), &args, Some(&repo))
    }
}

/// Samples gathered for each metric of a part, in the order the metrics first appeared.
#[derive(Debug, Default)]
pub(crate) struct Samples {
    /// Each metric's unit and samples.
    metrics: Vec<(MetricName, Unit, Vec<f64>)>,
}

impl Samples {
    /// Adds `x` to `name`'s samples.
    ///
    /// # Errors
    ///
    /// If `name` was added before with another unit.
    pub(crate) fn push(&mut self, name: MetricName, unit: Unit, x: f64) -> Result<()> {
        if let Some((_, known, xs)) = self.metrics.iter_mut().find(|(n, _, _)| *n == name) {
            ensure!(
                *known == unit,
                "{name} in {} and {}",
                known.name(),
                unit.name()
            );
            xs.push(x);
        } else {
            self.metrics.push((name, unit, vec![x]));
        }
        Ok(())
    }

    /// Each metric's median and spread.
    ///
    /// # Errors
    ///
    /// If a sample isn't finite.
    pub(crate) fn measure(self) -> Result<Vec<Metric>> {
        self.metrics
            .into_iter()
            .map(|(name, unit, samples)| {
                let value = Measured::of(samples).with_context(|| format!("measuring {name}"))?;
                Ok(Metric { name, unit, value })
            })
            .collect()
    }
}

/// The number field `field` of the one line among `lines` whose `bench` and `result` are
/// `bench` and `result`.
///
/// # Errors
///
/// If there isn't exactly one such line, or it lacks the field or holds `null` there.
fn field(lines: &[Object], bench: &str, result: &str, field: &str) -> Result<f64> {
    let is = |o: &&Object, name, want| o.get(name).and_then(Json::as_str) == Some(want);
    let mut found = lines
        .iter()
        .filter(|o| is(o, "bench", bench) && is(o, "result", result));
    let line = found
        .next()
        .ok_or_else(|| anyhow!("no {bench} {result} result"))?;
    ensure!(found.next().is_none(), "two {bench} {result} results");
    line.get(field)
        .and_then(Json::as_number)
        .ok_or_else(|| anyhow!("{bench} {result} has no number {field:?}"))
}

/// A metric taken straight from a result line: `(result, field, metric name after the
/// scale, unit)`.
type Taken = (&'static str, &'static str, &'static str, Unit);

/// What the suite takes from `brush`: latency, the phases that cost something (query and pin
/// are ~0.01 ms, timer noise), the cold first dab, and the index's load time.
const BRUSH: [Taken; 9] = [
    ("stroke", "step_median_ms", "brush.p50_ms", Unit::Ms),
    ("stroke", "step_p99_ms", "brush.p99_ms", Unit::Ms),
    ("stroke", "brush_mean_ms", "brush.brush_ms", Unit::Ms),
    ("stroke", "edit_mean_ms", "brush.edit_ms", Unit::Ms),
    (
        "stroke",
        "neighbours_mean_ms",
        "brush.neighbours_ms",
        Unit::Ms,
    ),
    ("stroke", "evaluate_mean_ms", "brush.evaluate_ms", Unit::Ms),
    ("stroke", "refit_mean_ms", "brush.refit_ms", Unit::Ms),
    (
        "stroke",
        "cold_first_dab_ms",
        "brush.cold_first_dab_ms",
        Unit::Ms,
    ),
    ("index", "ready_ms", "index.ready_ms", Unit::Ms),
];

/// What the suite takes from `page-in --pool both`: each region from a cold file, with the
/// pool cold and warm.
const PAGE_IN: [Taken; 4] = [
    (
        "footprint",
        "cold_median_ms",
        "page_in.footprint.cold_pool_ms",
        Unit::Ms,
    ),
    (
        "footprint-warm-pool",
        "cold_median_ms",
        "page_in.footprint.warm_pool_ms",
        Unit::Ms,
    ),
    (
        "region",
        "cold_median_ms",
        "page_in.region.cold_pool_ms",
        Unit::Ms,
    ),
    (
        "region-warm-pool",
        "cold_median_ms",
        "page_in.region.warm_pool_ms",
        Unit::Ms,
    ),
];

/// What the suite takes from `budget`: the walk's exact peak resident memory.
const BUDGET: [Taken; 1] = [(
    "walk",
    "max_rss_exact_mib",
    "budget.peak_rss_mib",
    Unit::Mib,
)];

/// Adds what `taken` names from one run's `lines` of subcommand `bench` to `samples`, under
/// `scale`'s names.
///
/// # Errors
///
/// If a result or field is missing.
fn take(
    samples: &mut Samples,
    scale: Scale,
    bench: &str,
    taken: &[Taken],
    lines: &[Object],
) -> Result<()> {
    for &(result, name, metric, unit) in taken {
        samples.push(
            MetricName::join(scale.name(), metric)?,
            unit,
            field(lines, bench, result, name)?,
        )?;
    }
    Ok(())
}

/// The arguments every subcommand gets at `scale`: the project and the level.
fn common(project: &Path, scale: Scale) -> Vec<OsString> {
    vec![
        "--project".into(),
        project.as_os_str().to_owned(),
        "--level".into(),
        scale.level().to_string().into(),
    ]
}

/// `args` as owned arguments.
fn words(args: &[&str]) -> Vec<OsString> {
    args.iter().map(OsString::from).collect()
}

/// The stage benches, `rounds` times: each round's divan median is a sample of its stage and
/// tile.
///
/// # Errors
///
/// If the benches fail or their table doesn't parse.
pub(crate) fn stage_part(runner: &mut Runner, rounds: usize) -> Result<Vec<Metric>> {
    stage_samples(std::iter::repeat_with(|| runner.stage_benches()), rounds)
}

/// The stage metrics from `rounds` of divan's tables taken from `outputs`, after one more taken
/// first and discarded.
///
/// The first round runs on cold caches and an unwarmed core, and measured
/// 1.6–1.9× the median in 7 of the 14 benches, which used up the spread's one trim.
///
/// # Errors
///
/// If an output fails or doesn't parse.
fn stage_samples(
    mut outputs: impl Iterator<Item = Result<String>>,
    rounds: usize,
) -> Result<Vec<Metric>> {
    let mut next = || {
        outputs
            .next()
            .ok_or_else(|| anyhow!("no stage bench output"))?
    };
    stages::parse(&next()?).context("the warm-up round")?;
    let mut samples = Samples::default();
    for _ in 0..rounds {
        for time in stages::parse(&next()?)? {
            samples.push(time.name, Unit::Us, time.median_us)?;
        }
    }
    samples.measure()
}

/// `brush` on `project`, [`Workload::runs`](crate::perf::project::Workload::runs) times.
///
/// # Errors
///
/// If a run fails or misses a result.
pub(crate) fn brush_part(runner: &mut Runner, project: &Path, scale: Scale) -> Result<Vec<Metric>> {
    let w = scale.workload();
    let mut samples = Samples::default();
    for _ in 0..w.runs {
        let mut args = words(&[
            "brush",
            "--steps",
            &w.brush_steps.to_string(),
            "--footprint",
            &w.footprint.to_string(),
        ]);
        args.extend(common(project, scale));
        let lines = runner.bench(&args)?;
        take(&mut samples, scale, "brush", &BRUSH, &lines)?;
    }
    samples.measure()
}

/// `page-in --pool both` on `project`, [`Workload::runs`](crate::perf::project::Workload::runs)
/// times.
///
/// # Errors
///
/// If a run fails or misses a result.
pub(crate) fn page_in_part(
    runner: &mut Runner,
    project: &Path,
    scale: Scale,
) -> Result<Vec<Metric>> {
    let w = scale.workload();
    let mut samples = Samples::default();
    for _ in 0..w.runs {
        let mut args = words(&[
            "page-in",
            "--pool",
            "both",
            "--samples",
            &w.page_in_samples.to_string(),
            "--footprint",
            &w.footprint.to_string(),
            "--region-quads",
            &w.region_quads.to_string(),
        ]);
        args.extend(common(project, scale));
        let lines = runner.bench(&args)?;
        take(&mut samples, scale, "page-in", &PAGE_IN, &lines)?;
    }
    samples.measure()
}

/// `budget` on `project`, [`Workload::runs`](crate::perf::project::Workload::runs) times.
///
/// # Errors
///
/// If a run fails or misses a result.
pub(crate) fn budget_part(
    runner: &mut Runner,
    project: &Path,
    scale: Scale,
) -> Result<Vec<Metric>> {
    let w = scale.workload();
    let mut samples = Samples::default();
    for _ in 0..w.runs {
        let mut args = words(&[
            "budget",
            "--moves",
            &w.moves.to_string(),
            "--steps-per-move",
            &w.steps_per_move.to_string(),
            "--view-tiles",
            &w.view_tiles.to_string(),
            "--footprint",
            &w.footprint.to_string(),
        ]);
        args.extend(common(project, scale));
        let lines = runner.bench(&args)?;
        take(&mut samples, scale, "budget", &BUDGET, &lines)?;
    }
    samples.measure()
}

/// The least-squares line `y = a + b x` through `points`: `(a, b)`; `None` for fewer than two
/// points or all at one `x`.
pub(crate) fn fit_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    #[expect(clippy::cast_precision_loss, reason = "a handful of points")]
    let n = points.len() as f64;
    if points.len() < 2 {
        return None;
    }
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    if sxx == 0.0 {
        return None;
    }
    let b = sxy / sxx;
    Some((mean_y - b * mean_x, b))
}

/// The save cost model on `project`: [`Workload::runs`](crate::perf::project::Workload::runs)
/// rounds, each saving one stroke of every
/// length in [`Workload::save_steps`](crate::perf::project::Workload::save_steps) into the same
/// project.
///
/// Each round's saves give a least-squares fit of save time against edited vertices: its
/// intercept is a sample of `save.fixed_ms`, its slope of `save.per_vertex_us`. Each round's
/// median bytes per edited vertex is a sample of `save.bytes_per_vertex`. The project gains a
/// save point per save, 25 at the medium and full scales, so one of the rounds writes a full
/// Index (every 16th save point, ~25 ms more on the full project): the median of the rounds
/// leaves it out, as the save cost model's fits did.
///
/// # Errors
///
/// If a save fails, misses a result, or a round's saves edit the same count of vertices (no
/// fit).
pub(crate) fn save_part(runner: &mut Runner, project: &Path, scale: Scale) -> Result<Vec<Metric>> {
    let w = scale.workload();
    let mut samples = Samples::default();
    for _ in 0..w.runs {
        let mut points = Vec::new();
        let mut per_vertex = Vec::new();
        for steps in w.save_steps {
            let mut args = words(&[
                "save",
                "--steps",
                &steps.to_string(),
                "--footprint",
                &w.footprint.to_string(),
            ]);
            args.extend(common(project, scale));
            let lines = runner.bench(&args)?;
            points.push((
                field(&lines, "save", "save", "edited_vertices")?,
                field(&lines, "save", "save", "save_ms")?,
            ));
            per_vertex.push(field(&lines, "save", "save", "bytes_per_edited_vertex")?);
        }
        let (fixed, slope) =
            fit_line(&points).ok_or_else(|| anyhow!("no save cost fit through {points:?}"))?;
        let name = |metric| MetricName::join(scale.name(), metric);
        samples.push(name("save.fixed_ms")?, Unit::Ms, fixed)?;
        samples.push(name("save.per_vertex_us")?, Unit::Us, slope * 1e3)?;
        samples.push(
            name("save.bytes_per_vertex")?,
            Unit::Bytes,
            Measured::of(per_vertex)?.median(),
        )?;
    }
    samples.measure()
}

/// Generates a project of `scale`'s shape at `project` with `generate`.
///
/// # Errors
///
/// If the scale has no shape (the full project isn't the suite's to make), or `generate` fails.
pub(crate) fn generate(runner: &mut Runner, project: &Path, scale: Scale) -> Result<()> {
    let shape = scale
        .shape()
        .ok_or_else(|| anyhow!("the {} project isn't generated by the suite", scale.name()))?;
    let mut args = words(&["generate", "--force"]);
    args.extend(shape.args());
    args.extend(["--project".into(), project.as_os_str().to_owned()]);
    runner.bench(&args)?;
    Ok(())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// The first round is a warm-up: its times never reach the samples.
    #[test]
    fn the_stage_warm_up_round_is_discarded() {
        let table = |median: &str| {
            format!(
                "stages   fastest │ slowest │ median │ mean │ samples │ iters\n\
                 ╰─ frames        │         │        │      │         │\n   \
                 ╰─ interior   1 µs │ 1 µs │ {median} µs │ 1 µs │ 1 │ 1\n"
            )
        };
        let outputs = ["99", "30", "31", "32"].map(|m| Ok(table(m)));
        let metrics = stage_samples(outputs.into_iter(), 3).unwrap();
        assert_eq!(metrics.len(), 1, "{metrics:?}");
        assert_eq!(metrics[0].value.samples(), [30.0, 31.0, 32.0], "no 99");
        let short = ["99", "30"].map(|m| Ok(table(m)));
        assert!(
            stage_samples(short.into_iter(), 3).is_err(),
            "too few outputs"
        );
    }

    #[test]
    fn lines_fit_and_degenerate_points_dont() {
        let (a, b) = fit_line(&[(0.0, 9.0), (100.0, 10.0), (200.0, 11.0)]).unwrap();
        assert!(
            (a - 9.0).abs() < 1e-12 && (b - 0.01).abs() < 1e-12,
            "exact: {a} {b}"
        );
        let (c, d) = fit_line(&[(1.0, 1.0), (2.0, 3.0), (3.0, 2.0)]).unwrap();
        assert!(
            (c - 1.0).abs() < 1e-12 && (d - 0.5).abs() < 1e-12,
            "least squares: {c} {d}"
        );
        assert_eq!(fit_line(&[(1.0, 1.0)]), None, "one point");
        assert_eq!(fit_line(&[(1.0, 1.0), (1.0, 2.0)]), None, "one x");
    }

    #[test]
    fn samples_gather_per_metric_in_order() {
        let mut samples = Samples::default();
        let p99: MetricName = "medium.brush.p99_ms".parse().unwrap();
        let ready: MetricName = "medium.index.ready_ms".parse().unwrap();
        for (x, y) in [(2.0, 100.0), (2.2, 110.0), (1.9, 105.0)] {
            samples.push(p99.clone(), Unit::Ms, x).unwrap();
            samples.push(ready.clone(), Unit::Ms, y).unwrap();
        }
        assert!(
            samples.push(p99.clone(), Unit::Mib, 1.0).is_err(),
            "one unit a metric"
        );
        let metrics = samples.measure().unwrap();
        assert_eq!(metrics.len(), 2, "{metrics:?}");
        assert_eq!(metrics[0].name, p99, "first seen first");
        assert!(
            (metrics[0].value.median() - 2.0).abs() < 1e-12,
            "{metrics:?}"
        );
        assert!(
            (metrics[1].value.median() - 105.0).abs() < 1e-12,
            "{metrics:?}"
        );
    }

    #[test]
    fn fields_come_from_the_one_matching_line() {
        let lines: Vec<Object> = [
            r#"{"bench":"brush","result":"index","ready_ms":0.5}"#,
            r#"{"bench":"brush","result":"stroke","step_p99_ms":2.5,"bad":null}"#,
        ]
        .iter()
        .map(|l| Object::parse(l).unwrap())
        .collect();
        assert_eq!(
            field(&lines, "brush", "stroke", "step_p99_ms").unwrap(),
            2.5,
            "found"
        );
        assert!(
            field(&lines, "brush", "stroke", "bad").is_err(),
            "null isn't a number"
        );
        assert!(
            field(&lines, "brush", "stroke", "missing").is_err(),
            "missing field"
        );
        assert!(
            field(&lines, "save", "save", "save_ms").is_err(),
            "missing result"
        );
        let twice = [lines[1].clone(), lines[1].clone()];
        assert!(
            field(&twice, "brush", "stroke", "step_p99_ms").is_err(),
            "ambiguous"
        );
        let mut samples = Samples::default();
        take(&mut samples, Scale::Medium, "brush", &BRUSH[1..2], &lines).unwrap();
        let metrics = samples.measure().unwrap();
        assert_eq!(
            metrics[0].name.as_str(),
            "medium.brush.p99_ms",
            "named for the scale"
        );
    }
}
