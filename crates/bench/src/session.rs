//! What the subcommands share: opening the project, building its spatial index, and running a
//! stroke of draw steps.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use geom::{Level, Vec3};
use sculpt::{DrawParams, StepStats, draw_step};
use store::{Project, ProjectConfig, SpatialIndex, StoreConfig};

use crate::cli::Options;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread, mib, ms, to_f64};

/// The site the harness's op ids carry: any but the reserved one.
const SITE: u16 = 1;

/// The project configuration for `options`' budget, with recovery files in `paths`' recovery
/// directory.
pub(crate) fn config(options: &Options, paths: &Paths) -> ProjectConfig {
    ProjectConfig {
        store: StoreConfig {
            budget_bytes: options.budget_bytes(),
            chunk_bytes: StoreConfig::DEFAULT_CHUNK_BYTES,
            site: SITE,
        },
        recovery_dir: paths.recovery.clone(),
    }
}

/// Opens the project at `paths` with `options`' budget.
///
/// # Errors
///
/// If it can't be opened (missing: run `generate` first), or the budget is below the store's
/// minimum.
pub(crate) fn open(options: &Options, paths: &Paths) -> Result<Project> {
    ensure!(
        paths.project.exists(),
        "{} doesn't exist: run `generate` first",
        paths.project.display()
    );
    Project::open(&paths.project, config(options, paths))
        .with_context(|| format!("opening {}", paths.project.display()))
}

/// `options`' level.
///
/// # Errors
///
/// If it is above the deepest level.
pub(crate) fn level(options: &Options) -> Result<Level> {
    Ok(Level::new(options.level)?)
}

/// Builds the spatial index over every tile of `level`, reporting progress, and records how
/// long it took (as `bench`'s `index` result). Returns the index.
///
/// Every tile of the level is evaluated once, positions and normals: on the benchmark sphere
/// that is 153,600 level-9 tiles, about two minutes on one core.
///
/// # Errors
///
/// If a tile can't be evaluated, or writing the report fails.
pub(crate) fn build_index<J: Write, H: Write>(
    project: &mut Project,
    level: Level,
    bench: &str,
    out: &mut Output<J, H>,
) -> Result<SpatialIndex> {
    out.say(&format!("building the spatial index over level {level}"))?;
    let start = Instant::now();
    let mut reported = Ok(());
    let index = SpatialIndex::build(project.store(), level, |done, total| {
        if reported.is_ok() {
            reported = out.progress("index", done, total, start);
        }
    })
    .context("building the spatial index")?;
    reported?;
    let took = start.elapsed();
    let tiles = index.bvh().len();
    out.record(
        &Record::new(bench, "index")
            .with("level", u64::from(level.get()))
            .with("tiles", tiles)
            .with("build_ms", ms(took))
            .with("heap_mib", mib(u64::try_from(index.bvh().heap_bytes())?)),
    )?;
    out.say(&format!(
        "index: {tiles} tiles in {:.1} s, {:.1} MiB",
        took.as_secs_f64(),
        mib(u64::try_from(index.bvh().heap_bytes())?)
    ))?;
    Ok(index)
}

/// One dab of a stroke as the harness measured it: the step's own statistics, and the wall
/// time of the whole call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dab {
    /// What the step did, with its phases' times.
    pub(crate) stats: StepStats,
    /// The call's wall time, start to return.
    pub(crate) wall: Duration,
}

/// Runs one stroke of draw steps with `radius` and `strength`, one at each of `centres`,
/// between [`Project::begin_stroke`] and [`Project::end_stroke`].
///
/// # Errors
///
/// If a step fails, or the stroke can't begin or end.
pub(crate) fn stroke(
    project: &mut Project,
    index: &mut SpatialIndex,
    centres: &[Vec3],
    radius: f32,
    strength: f32,
) -> Result<Vec<Dab>> {
    project.begin_stroke().context("beginning a stroke")?;
    let mut dabs = Vec::with_capacity(centres.len());
    for &center in centres {
        let params = DrawParams::new(center, radius, strength)?;
        let start = Instant::now();
        let stats = draw_step(project, index, params).context("a draw step")?;
        dabs.push(Dab {
            stats,
            wall: start.elapsed(),
        });
    }
    project.end_stroke().context("ending the stroke")?;
    Ok(dabs)
}

/// `record` with the latency of `dabs` (their wall times' spread, as `step_*`) and the mean of
/// each phase and footprint added.
pub(crate) fn add_dabs(record: Record, dabs: &[Dab]) -> Record {
    let walls: Vec<Duration> = dabs.iter().map(|d| d.wall).collect();
    let record = match Spread::of(&walls) {
        Some(spread) => spread.add_to(record, "step"),
        None => record,
    };
    let n = to_f64(dabs.len().max(1));
    let mean =
        |f: fn(&StepStats) -> Duration| ms(dabs.iter().map(|d| f(&d.stats)).sum::<Duration>()) / n;
    let count = |f: fn(&StepStats) -> usize| {
        dabs.iter().map(|d| f(&d.stats)).sum::<usize>() / dabs.len().max(1)
    };
    record
        .with("query_mean_ms", mean(|s| s.times.query))
        .with("pin_mean_ms", mean(|s| s.times.pin))
        .with("kernel_mean_ms", mean(|s| s.times.kernel))
        .with("edit_mean_ms", mean(|s| s.times.edit))
        .with("evaluate_mean_ms", mean(|s| s.times.evaluate))
        .with("refit_mean_ms", mean(|s| s.times.refit))
        .with("vertices_mean", count(|s| s.vertices))
        .with(
            "vertices_min",
            dabs.iter().map(|d| d.stats.vertices).min().unwrap_or(0),
        )
        .with(
            "vertices_max",
            dabs.iter().map(|d| d.stats.vertices).max().unwrap_or(0),
        )
        .with("candidates_mean", count(|s| s.candidates))
        .with("tiles_edited_mean", count(|s| s.tiles_edited))
        .with("tiles_updated_mean", count(|s| s.tiles_updated))
}

/// A line of human summary for a stroke's `dabs`: latency, footprint and the mean of each phase.
pub(crate) fn describe(dabs: &[Dab]) -> String {
    let walls: Vec<Duration> = dabs.iter().map(|d| d.wall).collect();
    let Some(spread) = Spread::of(&walls) else {
        return "no steps".to_owned();
    };
    let n = to_f64(dabs.len());
    let mean =
        |f: fn(&StepStats) -> Duration| ms(dabs.iter().map(|d| f(&d.stats)).sum::<Duration>()) / n;
    let vertices = dabs.iter().map(|d| d.stats.vertices).sum::<usize>() / dabs.len();
    format!(
        "{} steps of about {vertices} vertices: p50 {:.2} ms, p99 {:.2} ms (max {:.2}); mean \
         query {:.2}, pin {:.2}, kernel {:.2}, edit {:.2}, evaluate {:.2}, refit {:.2} ms",
        dabs.len(),
        spread.median,
        spread.p99,
        spread.max,
        mean(|s| s.times.query),
        mean(|s| s.times.pin),
        mean(|s| s.times.kernel),
        mean(|s| s.times.edit),
        mean(|s| s.times.evaluate),
        mean(|s| s.times.refit),
    )
}
