//! What the subcommands share: opening the project, loading its spatial index, and running a
//! stroke of draw steps.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use geom::{Level, Vec3};
use sculpt::{DrawParams, StepStats, draw_step};
use store::{Activity, Project, ProjectConfig, SpatialIndex, StoreConfig};

use crate::cli::Options;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread, mib, ms, to_f64};
use crate::sys;

/// The site the harness's op ids carry: any but the reserved one.
const SITE: u16 = 1;

/// The project configuration for `options`' budget and evaluation workers, with recovery files
/// in `paths`' recovery directory.
pub(crate) fn config(options: &Options, paths: &Paths) -> ProjectConfig {
    ProjectConfig {
        store: StoreConfig::new(options.budget_bytes(), SITE).with_workers(options.workers),
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

/// Loads the spatial index over every tile of `level` from the bounds the project keeps.
///
/// Reports progress, and records how long the index took to be ready (as `bench`'s `index`
/// result), with how many tiles' bounds were read and how many evaluated. Returns the index.
///
/// A project `generate` made at this level keeps every bound, and loads without evaluating a
/// tile. One saved without them (or at another level) has every tile evaluated, positions and
/// normals: on the benchmark sphere that is 153,600 level-9 tiles, 14.3 s on a quiet dev box.
///
/// # Errors
///
/// If the bounds can't be read or a tile can't be evaluated, or writing the report fails.
pub(crate) fn load_index<J: Write, H: Write>(
    project: &mut Project,
    level: Level,
    bench: &str,
    out: &mut Output<J, H>,
) -> Result<SpatialIndex> {
    out.say(&format!("loading the spatial index over level {level}"))?;
    let start = Instant::now();
    let mut reported = Ok(());
    let (index, stats) = project
        .load_index(level, |done, total| {
            if reported.is_ok() {
                reported = out.progress("index", done, total, start);
            }
        })
        .context("loading the spatial index")?;
    reported?;
    let took = start.elapsed();
    let tiles = index.bvh().len();
    out.record(
        &Record::new(bench, "index")
            .with("level", u64::from(level.get()))
            .with("tiles", tiles)
            .with("stored", stats.stored)
            .with("evaluated", stats.evaluated)
            .with("ready_ms", ms(took))
            .with("heap_mib", mib(u64::try_from(index.bvh().heap_bytes())?)),
    )?;
    out.say(&format!(
        "index: {tiles} tiles ready in {:.2} s ({} stored, {} evaluated), {:.1} MiB",
        took.as_secs_f64(),
        stats.stored,
        stats.evaluated,
        mib(u64::try_from(index.bvh().heap_bytes())?)
    ))?;
    Ok(index)
}

/// One dab of a stroke as the harness measured it: the step's own statistics, the wall time
/// of the whole call, and what the store and the process did during it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dab {
    /// What the step did, with its phases' times.
    pub(crate) stats: StepStats,
    /// The call's wall time, start to return.
    pub(crate) wall: Duration,
    /// What the tile store did during the call: tiles loaded, evaluated and evicted.
    pub(crate) activity: Activity,
    /// The minor page faults the process took during the call (every thread of it), if the
    /// stroke counted them ([`Detail::PerStep`]).
    pub(crate) minor_faults: Option<u64>,
}

/// How closely [`stroke`] watches each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Detail {
    /// The step's statistics, wall time and store activity: what costs nothing to take.
    Summary,
    /// Those, and the process's page faults around each step: two reads of
    /// `/proc/self/stat` a step, outside the time measured.
    PerStep,
}

/// Runs one stroke of draw steps with `radius` and `strength`, one at each of `centres`,
/// between [`Project::begin_stroke`] and [`Project::end_stroke`], watching each as `detail`
/// says.
///
/// # Errors
///
/// If a step fails, the stroke can't begin or end, or (with [`Detail::PerStep`]) the page
/// faults can't be read.
pub(crate) fn stroke(
    project: &mut Project,
    index: &mut SpatialIndex,
    centres: &[Vec3],
    radius: f32,
    strength: f32,
    detail: Detail,
) -> Result<Vec<Dab>> {
    let faults = || match detail {
        Detail::Summary => Ok(None),
        Detail::PerStep => sys::minor_faults().map(Some),
    };
    project.begin_stroke().context("beginning a stroke")?;
    let mut dabs = Vec::with_capacity(centres.len());
    for &center in centres {
        let params = DrawParams::new(center, radius, strength)?;
        let faults_before = faults()?;
        let activity_before = project.store().activity();
        let start = Instant::now();
        let stats = draw_step(project, index, params).context("a draw step")?;
        let wall = start.elapsed();
        let activity = project.store().activity().since(&activity_before);
        let minor_faults = faults()?
            .zip(faults_before)
            .map(|(after, before)| after.saturating_sub(before));
        dabs.push(Dab {
            stats,
            wall,
            activity,
            minor_faults,
        });
    }
    project.end_stroke().context("ending the stroke")?;
    Ok(dabs)
}

/// One record per dab of `dabs`, as `bench`'s `step` result, in stroke order.
///
/// Each has the dab's number (from 0), its wall time and each phase's, its footprint
/// (candidates, tiles edited, vertices moved, tiles updated), what the store did during it, and
/// the page faults if they were counted: what finds the slow steps of a stroke and tells what
/// they did differently.
pub(crate) fn step_records<'a>(
    bench: &'a str,
    dabs: &'a [Dab],
) -> impl Iterator<Item = Record> + 'a {
    dabs.iter().enumerate().map(move |(n, dab)| {
        let (stats, times, activity) = (&dab.stats, &dab.stats.times, &dab.activity);
        let record = Record::new(bench, "step")
            .with("step", n)
            .with("wall_ms", ms(dab.wall))
            .with("query_ms", ms(times.query))
            .with("pin_ms", ms(times.pin))
            .with("brush_ms", ms(times.brush))
            .with("edit_ms", ms(times.edit))
            .with("neighbours_ms", ms(times.neighbours))
            .with("evaluate_ms", ms(times.evaluate))
            .with("refit_ms", ms(times.refit))
            .with("candidates", stats.candidates)
            .with("tiles_edited", stats.tiles_edited)
            .with("vertices", stats.vertices)
            .with("tiles_updated", stats.tiles_updated)
            .with("corrections_loaded", activity.corrections_loaded)
            .with("positions_computed", activity.positions_computed)
            .with("predictions_reused", activity.predictions_reused)
            .with("normals_computed", activity.normals_computed)
            .with("corrections_evicted", activity.corrections_evicted)
            .with("evaluated_evicted", activity.evaluated_evicted);
        match dab.minor_faults {
            Some(faults) => record.with("minor_faults", faults),
            None => record,
        }
    })
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
        .with("brush_mean_ms", mean(|s| s.times.brush))
        .with("edit_mean_ms", mean(|s| s.times.edit))
        .with("neighbours_mean_ms", mean(|s| s.times.neighbours))
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
         query {:.2}, pin {:.2}, brush {:.2}, edit {:.2}, neighbours {:.2}, evaluate {:.2}, \
         refit {:.2} ms",
        dabs.len(),
        spread.median,
        spread.p99,
        spread.max,
        mean(|s| s.times.query),
        mean(|s| s.times.pin),
        mean(|s| s.times.brush),
        mean(|s| s.times.edit),
        mean(|s| s.times.neighbours),
        mean(|s| s.times.evaluate),
        mean(|s| s.times.refit),
    )
}
