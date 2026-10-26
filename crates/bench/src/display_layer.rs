//! `display`: what keeping the display layer costs on the project.
//!
//! Opens the project keeping a display layer under its finest level, and:
//!
//! 1. **rebuilds** the layer level by level, finest first (each level after the one below it, as
//!    the store's rebuild job does), through [`store::TileStore::run_jobs`]: the wall time of each
//!    level, the total, and the floor's (levels up to 5);
//! 2. **saves** it, and reports how much the project file grew;
//! 3. runs the `brush` workload's warm stroke with the display layer kept: after each step, the
//!    refresh it leaves (every display tile the step made unknown) is run to the end, timed, and
//!    the tiles it built counted.
//!
//! The save goes into the project: measure a reflinked copy.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use geom::Level;
use sculpt::{DrawParams, Stroke, draw_step};
use store::{Project, TileStore};

use crate::brush::{Plan, Size};
use crate::cli::{DisplayLayer, Options};
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread, ms, to_f64};
use crate::session;

/// The highest level of the floor (see the M1 spec's "Residency"), reported apart.
const FLOOR_TOP: u8 = 5;

/// Runs the store's jobs to the end in slices of `slice` (at least once: the first call queues
/// the refresh writes left); returns the tiles built and the time.
///
/// # Errors
///
/// If a job fails.
fn run_to_end(store: &mut TileStore, slice: Duration) -> Result<(usize, Duration)> {
    let start = Instant::now();
    let mut built = store.run_jobs(Instant::now() + slice)?.built;
    while store.has_jobs() {
        built += store.run_jobs(Instant::now() + slice)?.built;
    }
    Ok((built, start.elapsed()))
}

/// Runs `display` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, a tile built, the save or a step fail, or writing the report
/// fails.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let finest = session::level(options)?;
    let mut display_options = options.clone();
    display_options.display = DisplayLayer::Kept;
    let before = std::fs::metadata(&paths.project)?.len();
    let mut project = session::open(&display_options, paths)?;
    let mut index = session::load_index(&mut project, finest, "display", out)?;
    rebuild(project.store(), finest, out)?;
    let saving = Instant::now();
    let outcome = project.save().context("saving the display layer")?;
    let save_time = saving.elapsed();
    let after = std::fs::metadata(&paths.project)?.len();
    let grown = after.saturating_sub(before);
    out.record(
        &Record::new("display", "save")
            .with("file_bytes_before", before)
            .with("file_bytes_after", after)
            .with(
                "growth",
                to_f64(usize::try_from(grown)?) / to_f64(usize::try_from(before)?),
            )
            .with("save_ms", ms(save_time))
            .with("outcome", format!("{outcome:?}")),
    )?;
    out.say(&format!(
        "display: saved in {:.0} ms, the file grew {:.1} % ({before} to {after} bytes)",
        ms(save_time),
        100.0 * to_f64(usize::try_from(grown)?) / to_f64(usize::try_from(before)?)
    ))?;
    refresh_stroke(&mut project, &mut index, options, out)?;
    project.close(false).context("closing the project")?;
    Ok(())
}

/// Rebuilds the display layer of `store` under `finest`, level by level, reporting each.
///
/// # Errors
///
/// If a job fails, or writing the report fails.
fn rebuild<J: Write, H: Write>(
    store: &mut TileStore,
    finest: Level,
    out: &mut Output<J, H>,
) -> Result<()> {
    store.clear_jobs();
    let (mut total, mut floor, mut tiles) = (Duration::ZERO, Duration::ZERO, 0);
    for l in (0..finest.get()).rev() {
        let level = Level::new(l)?;
        store.queue_display_level(finest, level);
        let (built, time) = run_to_end(store, Duration::from_millis(50))?;
        total += time;
        tiles += built;
        if l <= FLOOR_TOP {
            floor += time;
        }
        out.record(
            &Record::new("display", "rebuild_level")
                .with("level", u64::from(l))
                .with("tiles", built)
                .with("ms", ms(time))
                .with("us_per_tile", ms(time) * 1000.0 / to_f64(built.max(1))),
        )?;
        out.say(&format!(
            "display: level {l}: {built} tiles in {:.1} s ({:.0} µs a tile)",
            time.as_secs_f64(),
            ms(time) * 1000.0 / to_f64(built.max(1))
        ))?;
    }
    out.record(
        &Record::new("display", "rebuild")
            .with("tiles", tiles)
            .with("total_s", total.as_secs_f64())
            .with("floor_s", floor.as_secs_f64()),
    )?;
    out.say(&format!(
        "display: rebuilt {tiles} tiles in {:.1} s (floor, levels ≤ {FLOOR_TOP}: {:.1} s)",
        total.as_secs_f64(),
        floor.as_secs_f64()
    ))?;
    Ok(())
}

/// The `brush` workload's warm stroke with the display layer kept: after each step, its refresh
/// run to the end and timed.
///
/// # Errors
///
/// If a step or a refresh fails, or writing the report fails.
fn refresh_stroke<J: Write, H: Write>(
    project: &mut Project,
    index: &mut store::SpatialIndex,
    options: &Options,
    out: &mut Output<J, H>,
) -> Result<()> {
    let direction = Rng::new(options.seed).direction();
    let plan = Plan::new(
        project,
        index,
        direction,
        Size::Footprint(options.footprint),
        options.steps,
    )?;
    let mut state = Stroke::new(project.store());
    project.begin_stroke()?;
    let (mut steps, mut refreshes, mut built) = (Vec::new(), Vec::new(), Vec::new());
    for &centre in &plan.centres {
        let params = DrawParams::new(centre, plan.radius, options.strength)?;
        let start = Instant::now();
        draw_step(project.store(), index, &mut state, params)?;
        steps.push(start.elapsed());
        let (tiles, time) = run_to_end(project.store(), Duration::from_millis(50))?;
        refreshes.push(time);
        built.push(tiles);
    }
    project.end_stroke()?;
    let tiles: usize = built.iter().sum();
    let total: Duration = refreshes.iter().sum();
    let mut record = Record::new("display", "refresh")
        .with("steps", steps.len())
        .with("tiles_per_step", to_f64(tiles) / to_f64(steps.len().max(1)))
        .with("us_per_tile", ms(total) * 1000.0 / to_f64(tiles.max(1)));
    if let Some(spread) = Spread::of(&refreshes) {
        record = spread.add_to(record, "refresh");
    }
    if let Some(spread) = Spread::of(&steps) {
        record = spread.add_to(record, "step");
    }
    out.record(&record)?;
    out.say(&format!(
        "display: refresh after each of {} steps: {:.1} tiles a step, {:.0} µs a tile, {:.2} ms a step",
        steps.len(),
        to_f64(tiles) / to_f64(steps.len().max(1)),
        ms(total) * 1000.0 / to_f64(tiles.max(1)),
        ms(total) / to_f64(steps.len().max(1))
    ))?;
    // The same stroke again, its refresh run once at its end: what a refresh after the stroke
    // costs, the tiles the steps share built once.
    project.begin_stroke()?;
    for &centre in &plan.centres {
        let params = DrawParams::new(centre, plan.radius, options.strength)?;
        draw_step(project.store(), index, &mut state, params)?;
    }
    project.end_stroke()?;
    let (once, time) = run_to_end(project.store(), Duration::from_millis(50))?;
    out.record(
        &Record::new("display", "refresh_after_stroke")
            .with("steps", plan.centres.len())
            .with("tiles", once)
            .with("ms", ms(time))
            .with("us_per_tile", ms(time) * 1000.0 / to_f64(once.max(1))),
    )?;
    out.say(&format!(
        "display: refresh once after a stroke of {} steps: {once} tiles in {:.0} ms",
        plan.centres.len(),
        ms(time)
    ))?;
    Ok(())
}
