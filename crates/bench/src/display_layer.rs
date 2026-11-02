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
//!    the tiles it built counted;
//! 4. **reopens** the project, alternately with the display layer kept and without, timing each
//!    open (with it, the open loads the floor), and reports the floor's size.
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

/// What running the store's jobs to the end took.
#[derive(Debug, Clone, Copy)]
struct Ran {
    /// The display tiles built.
    built: usize,
    /// The wall time.
    time: Duration,
    /// The longest slice of any call: how far past a deadline a call can run.
    longest_slice: Duration,
    /// Slices run.
    slices: usize,
    /// Slices longer than a slice is made to take (`JobsReport::SLICE_TARGET`).
    long_slices: usize,
}

/// Runs the store's jobs to the end in calls of `budget` each (at least one: the first call
/// queues the refresh writes left).
///
/// # Errors
///
/// If a job fails.
fn run_to_end(store: &mut TileStore, budget: Duration) -> Result<Ran> {
    let start = Instant::now();
    let mut ran = Ran {
        built: 0,
        time: Duration::ZERO,
        longest_slice: Duration::ZERO,
        slices: 0,
        long_slices: 0,
    };
    loop {
        let report = store.run_jobs(Instant::now() + budget)?;
        ran.built += report.built;
        ran.longest_slice = ran.longest_slice.max(report.longest_slice);
        ran.slices += report.slices;
        ran.long_slices += report.long_slices;
        if !store.has_jobs() {
            break;
        }
    }
    ran.time = start.elapsed();
    Ok(ran)
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
    // A closed quad mesh has about one vertex a quad (two more for a sphere's).
    let quads = u64::from(project.store().mesh().face_count()) << (2 * finest.get());
    out.record(
        &Record::new("display", "save")
            .with("file_bytes_before", before)
            .with("file_bytes_after", after)
            .with(
                "growth",
                to_f64(usize::try_from(grown)?) / to_f64(usize::try_from(before)?),
            )
            .with(
                "growth_bytes_per_vertex",
                to_f64(usize::try_from(grown)?) / to_f64(usize::try_from(quads)?),
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
    let refreshed = project
        .save()
        .context("saving the refreshed display layer")?;
    out.say(&format!(
        "display: the refreshed layer saved: {refreshed:?}"
    ))?;
    project.close(false).context("closing the project")?;
    reopen(options, paths, out)
}

/// `display-save`: rebuilds the project's display layer under its finest level and saves it.
///
/// The project then opens with its layer, so the app, the pen probe and render measurements on
/// the benchmark project skip a minute's rebuild.
///
/// It writes into the project, so the project must be named ([`crate::cli::Command::parse`]
/// refuses the bench's default, the benchmark sphere `perf` measures as it is): make a reflinked
/// copy and pass it.
///
/// # Errors
///
/// If the open, the rebuild, the save or the close fails.
pub(crate) fn run_save<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let finest = session::level(options)?;
    let before = std::fs::metadata(&paths.project)?.len();
    let mut project = session::open(options, paths)?;
    rebuild(project.store(), finest, out)?;
    let outcome = project.save().context("saving the display layer")?;
    project.close(false).context("closing the project")?;
    let after = std::fs::metadata(&paths.project)?.len();
    out.say(&format!(
        "display-save: saved {} ({before} to {after} bytes): {outcome:?}",
        paths.project.display()
    ))?;
    Ok(())
}

/// How many times each way [`reopen`] opens the project.
const OPENS: usize = 3;

/// Opens the project [`OPENS`] times with the display layer kept and as many without,
/// alternately, timing each, and reports the floor the store loaded.
///
/// # Errors
///
/// If an open or a close fails, or writing the report fails.
fn reopen<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let mut plain_options = options.clone();
    plain_options.display = DisplayLayer::Off;
    let mut display_options = options.clone();
    display_options.display = DisplayLayer::Kept;
    let (mut with, mut without) = (Vec::new(), Vec::new());
    let (mut floor_tiles, mut floor_bytes, mut reserved) = (0, 0, 0);
    for _ in 0..OPENS {
        let plain_start = Instant::now();
        let plain = session::open(&plain_options, paths)?;
        without.push(plain_start.elapsed());
        plain.close(false).context("closing the project")?;
        let kept_start = Instant::now();
        let mut kept = session::open(&display_options, paths)?;
        with.push(kept_start.elapsed());
        (floor_tiles, floor_bytes) = kept.store().floor_size();
        reserved = session::config(&display_options, paths)
            .store
            .floor_bytes(kept.store().mesh());
        kept.close(false).context("closing the project")?;
    }
    let mut record = Record::new("display", "open")
        .with("floor_tiles", floor_tiles)
        .with("floor_bytes", floor_bytes)
        .with("floor_reserved_bytes", reserved);
    if let Some(spread) = Spread::of(&with) {
        record = spread.add_to(record, "with_display");
    }
    if let Some(spread) = Spread::of(&without) {
        record = spread.add_to(record, "without_display");
    }
    out.record(&record)?;
    let median = |samples: &[Duration]| Spread::of(samples).map_or(0.0, |s| s.median);
    out.say(&format!(
        "display: open {:.0} ms with the display layer, {:.0} ms without (medians of {OPENS}); \
         the floor holds {floor_tiles} tiles, {:.1} MB of encodings ({:.1} MB reserved)",
        median(&with),
        median(&without),
        to_f64(floor_bytes) / 1e6,
        to_f64(reserved) / 1e6
    ))?;
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
    let (mut longest, mut slices, mut long_slices) = (Duration::ZERO, 0, 0);
    for l in (0..finest.get()).rev() {
        let level = Level::new(l)?;
        store.queue_display_level(finest, level);
        let ran = run_to_end(store, Duration::from_millis(50))?;
        let Ran {
            built,
            time,
            longest_slice,
            ..
        } = ran;
        longest = longest.max(longest_slice);
        slices += ran.slices;
        long_slices += ran.long_slices;
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
                .with("us_per_tile", ms(time) * 1000.0 / to_f64(built.max(1)))
                .with("longest_slice_ms", ms(longest_slice)),
        )?;
        out.say(&format!(
            "display: level {l}: {built} tiles in {:.1} s ({:.0} µs a tile, longest slice {:.2} ms)",
            time.as_secs_f64(),
            ms(time) * 1000.0 / to_f64(built.max(1)),
            ms(longest_slice)
        ))?;
    }
    out.record(
        &Record::new("display", "rebuild")
            .with("tiles", tiles)
            .with("total_s", total.as_secs_f64())
            .with("total_ms", ms(total))
            .with("floor_s", floor.as_secs_f64())
            .with("longest_slice_ms", ms(longest))
            .with("slices", slices)
            .with("long_slices", long_slices),
    )?;
    out.say(&format!(
        "display: rebuilt {tiles} tiles in {:.1} s (floor, levels ≤ {FLOOR_TOP}: {:.1} s); \
         longest slice {:.2} ms, {long_slices} of {slices} slices over {:.0} ms",
        total.as_secs_f64(),
        floor.as_secs_f64(),
        ms(longest),
        ms(store::JobsReport::SLICE_TARGET)
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
    let (mut longest, mut slices, mut long_slices) = (Duration::ZERO, 0, 0);
    for &centre in &plan.centres {
        let params = DrawParams::new(centre, centre, plan.radius, options.strength)?;
        let start = Instant::now();
        draw_step(project.store(), index, &mut state, params)?;
        steps.push(start.elapsed());
        let ran = run_to_end(project.store(), Duration::from_millis(50))?;
        refreshes.push(ran.time);
        built.push(ran.built);
        longest = longest.max(ran.longest_slice);
        slices += ran.slices;
        long_slices += ran.long_slices;
    }
    project.end_unrecorded_stroke()?;
    let tiles: usize = built.iter().sum();
    let total: Duration = refreshes.iter().sum();
    let mut record = Record::new("display", "refresh")
        .with("steps", steps.len())
        .with("tiles_per_step", to_f64(tiles) / to_f64(steps.len().max(1)))
        .with("us_per_tile", ms(total) * 1000.0 / to_f64(tiles.max(1)))
        .with(
            "tiles_a_second",
            to_f64(tiles) / total.as_secs_f64().max(f64::MIN_POSITIVE),
        )
        .with("longest_slice_ms", ms(longest))
        .with("slices", slices)
        .with("long_slices", long_slices);
    if let Some(spread) = Spread::of(&refreshes) {
        record = spread.add_to(record, "refresh");
    }
    if let Some(spread) = Spread::of(&steps) {
        record = spread.add_to(record, "step");
    }
    out.record(&record)?;
    out.say(&format!(
        "display: refresh after each of {} steps: {:.1} tiles a step, {:.0} µs a tile ({:.0} a \
         second), {:.2} ms a step, longest slice {:.2} ms, {long_slices} of {slices} over {:.0} ms",
        steps.len(),
        to_f64(tiles) / to_f64(steps.len().max(1)),
        ms(total) * 1000.0 / to_f64(tiles.max(1)),
        to_f64(tiles) / total.as_secs_f64().max(f64::MIN_POSITIVE),
        ms(total) / to_f64(steps.len().max(1)),
        ms(longest),
        ms(store::JobsReport::SLICE_TARGET)
    ))?;
    // The same stroke again, its refresh run once at its end: what a refresh after the stroke
    // costs, the tiles the steps share built once.
    project.begin_stroke()?;
    for &centre in &plan.centres {
        let params = DrawParams::new(centre, centre, plan.radius, options.strength)?;
        draw_step(project.store(), index, &mut state, params)?;
    }
    project.end_unrecorded_stroke()?;
    let Ran {
        built: once, time, ..
    } = run_to_end(project.store(), Duration::from_millis(50))?;
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
    paced_stroke(project, index, options, &plan, &mut state, out)
}

/// The steps a second of [`paced_stroke`]: a pen's rate.
const STEPS_A_SECOND: u32 = 120;

/// The same stroke again at [`STEPS_A_SECOND`], the jobs given the rest of each step.
///
/// After each step, the jobs run until the next step is due, as a frame loop gives them the
/// rest of each frame. Reports the display tiles
/// built during the stroke (each made unknown by several steps is built once), the tiles a
/// second that makes, the longest slice, what is left when the stroke ends and how long it takes
/// to catch up.
///
/// # Errors
///
/// If a step or a job fails, or writing the report fails.
fn paced_stroke<J: Write, H: Write>(
    project: &mut Project,
    index: &mut store::SpatialIndex,
    options: &Options,
    plan: &Plan,
    state: &mut Stroke,
    out: &mut Output<J, H>,
) -> Result<()> {
    let period = Duration::from_secs(1) / STEPS_A_SECOND;
    let began = Instant::now();
    let (mut built, mut longest, mut late) = (0_usize, Duration::ZERO, 0_usize);
    let (mut slices, mut long_slices) = (0, 0);
    project.begin_stroke()?;
    for &centre in &plan.centres {
        let due = Instant::now() + period;
        let params = DrawParams::new(centre, centre, plan.radius, options.strength)?;
        draw_step(project.store(), index, state, params)?;
        if Instant::now() > due {
            late += 1;
        }
        let report = project.store().run_jobs(due)?;
        built += report.built;
        longest = longest.max(report.longest_slice);
        slices += report.slices;
        long_slices += report.long_slices;
    }
    project.end_unrecorded_stroke()?;
    let stroke = began.elapsed();
    let left = project.store().run_jobs(Instant::now()).map(|r| r.left)?;
    let ran = run_to_end(project.store(), Duration::from_millis(50))?;
    let (after, catch_up) = (ran.built, ran.time);
    longest = longest.max(ran.longest_slice);
    slices += ran.slices;
    long_slices += ran.long_slices;
    let rate = to_f64(built) / stroke.as_secs_f64();
    out.record(
        &Record::new("display", "paced_stroke")
            .with("steps", plan.centres.len())
            .with("steps_a_second", u64::from(STEPS_A_SECOND))
            .with("steps_late", late)
            .with("tiles_during", built)
            .with("tiles_a_second", rate)
            .with("left_at_end", left)
            .with("tiles_after", after)
            .with("catch_up_ms", ms(catch_up))
            .with("longest_slice_ms", ms(longest))
            .with("slices", slices)
            .with("long_slices", long_slices),
    )?;
    out.say(&format!(
        "display: a stroke of {} steps at {STEPS_A_SECOND} a second ({late} steps late): {built} \
         tiles built during it ({rate:.0} a second), {left} left at its end, caught up with \
         {after} more in {:.0} ms; longest slice {:.2} ms, {long_slices} of {slices} over {:.0} ms",
        plan.centres.len(),
        ms(catch_up),
        ms(longest),
        ms(store::JobsReport::SLICE_TARGET)
    ))?;
    Ok(())
}
