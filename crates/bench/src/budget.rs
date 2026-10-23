//! `budget`: a random walk under a memory budget, with the process's resident memory sampled
//! every 100 ms.
//!
//! The walk pans to `moves` random places. At each it evaluates and pins the `view-tiles`
//! tiles nearest the place, as a view of that part of the model would, and brushes a short
//! stroke there (`steps-per-move` draw steps over a `footprint`-vertex brush), then lets the
//! view go and moves on. The store's budget (1 GiB unless told otherwise) is all it may use
//! for tiles; the spec's target is RSS within budget + 512 MB throughout.
//!
//! The resident memory is reported two ways: sampled every 100 ms, a series that shows how it
//! moves, and exactly, as the high-water mark (`VmHWM`) reset just before the walk and read
//! just after it, which no spike between two samples escapes.
//!
//! The baseline is the process with the project open and its spatial index built, the store
//! then emptied (`evict_all` + `shrink_to(0)`): everything but tiles. After the walk the store
//! is emptied again, and the spec's target is RSS back within baseline + 64 MB.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use store::{Project, TileStore};

use crate::brush::{Plan, Size};
use crate::cli::Options;
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, mib, ms, to_f64};
use crate::session;
use crate::surface::{nearest_tiles, tile_directions};
use crate::sys::{Rss, RssSampler, reset_peak};

/// Empties the store: dirty tiles written back, every tile evicted, every chunk given back.
///
/// # Errors
///
/// If a write-back fails.
fn empty(store: &mut TileStore) -> Result<()> {
    store.evict_all()?;
    store.shrink_to(0)?;
    Ok(())
}

/// How long the walk waits after emptying the store before reading the resident memory, so
/// the sample isn't taken mid-unmap.
const SETTLE: Duration = Duration::from_millis(200);

/// `bytes` minus `base`, in MiB, negative if below it.
fn mib_over(bytes: u64, base: u64) -> f64 {
    mib(bytes) - mib(base)
}

/// Runs `budget` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, the index built, a view pinned or a step run, the resident
/// memory read, or writing the report fails.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let level = session::level(options)?;
    let before_open = Rss::read()?;
    let mut project = session::open(options, paths)?;
    let mut index = session::build_index(&mut project, level, "budget", out)?;
    empty(project.store())?;
    std::thread::sleep(SETTLE);
    let baseline = Rss::read()?;
    let tiles = tile_directions(project.store().mesh(), level)?;
    let mut rng = Rng::new(options.seed);
    out.say(&format!(
        "budget: {} moves of {} steps under {} MiB, views of {} tiles; baseline RSS {:.1} MiB",
        options.moves,
        options.steps_per_move,
        options.budget_mib,
        options.view_tiles,
        mib(baseline.now)
    ))?;
    reset_peak()?;
    let sampler = RssSampler::start()?;
    let start = Instant::now();
    let walked = walk(
        &mut project,
        &mut index,
        &tiles,
        &mut rng,
        options,
        out,
        start,
    );
    let rss_samples = sampler.stop()?;
    let exact = Rss::read()?.peak;
    let steps = walked?;
    let walk_time = start.elapsed();
    empty(project.store())?;
    std::thread::sleep(SETTLE);
    let after = Rss::read()?;
    let most = rss_samples.iter().map(|&(_, rss)| rss).max().unwrap_or(0);
    let budget = u64::try_from(options.budget_bytes())?;
    let series: Vec<f64> = rss_samples.iter().map(|&(_, rss)| mib(rss)).collect();
    out.record(
        &Record::new("budget", "walk")
            .with("level", u64::from(level.get()))
            .with("budget_mib", options.budget_mib)
            .with("moves", options.moves)
            .with("steps", steps)
            .with("view_tiles", options.view_tiles)
            .with("walk_ms", ms(walk_time))
            .with("rss_before_open_mib", mib(before_open.now))
            .with("baseline_rss_mib", mib(baseline.now))
            .with("max_rss_mib", mib(most))
            .with("max_rss_over_budget_mib", mib_over(most, budget))
            .with("max_rss_exact_mib", mib(exact))
            .with("max_rss_exact_over_budget_mib", mib_over(exact, budget))
            .with("rss_after_empty_mib", mib(after.now))
            .with(
                "after_empty_over_baseline_mib",
                mib_over(after.now, baseline.now),
            )
            .with("sample_every_ms", ms(RssSampler::EVERY))
            .with("rss_samples_mib", series),
    )?;
    out.say(&format!(
        "budget: peak RSS {:.1} MiB ({:+.1} MiB against the {} MiB budget); {} samples over \
         {:.1} s, max {:.1} MiB; after emptying {:.1} MiB ({:+.1} MiB against the baseline)",
        mib(exact),
        mib_over(exact, budget),
        options.budget_mib,
        rss_samples.len(),
        walk_time.as_secs_f64(),
        mib(most),
        mib(after.now),
        mib_over(after.now, baseline.now)
    ))?;
    project.close(false).context("closing the project")?;
    Ok(())
}

/// The walk itself: `options.moves` places, each a pinned view and a short stroke. Returns
/// the draw steps run.
///
/// # Errors
///
/// If a view can't be pinned, a step fails, or writing progress fails.
fn walk<J: Write, H: Write>(
    project: &mut Project,
    index: &mut store::SpatialIndex,
    tiles: &[(geom::TileKey, geom::Vec3)],
    rng: &mut Rng,
    options: &Options,
    out: &mut Output<J, H>,
    start: Instant,
) -> Result<usize> {
    let mut steps = 0;
    let mut radius = None;
    for done in 1..=options.moves {
        let direction = rng.direction();
        let view = nearest_tiles(tiles, direction, options.view_tiles);
        let pins = project.store().pin(&view).context("pinning a view")?;
        // The radius is set at the first place and kept, as a user keeps a brush size.
        let size = radius.map_or(Size::Footprint(options.footprint), Size::Radius);
        let plan = Plan::new(project, index, direction, size, options.steps_per_move)?;
        radius = Some(plan.radius);
        steps +=
            session::stroke(project, index, &plan.centres, plan.radius, options.strength)?.len();
        drop(pins);
        project.poll()?;
        out.progress("moves", done, options.moves, start)?;
    }
    // A mean rate for the summary's sake; the record has the walk's time.
    let rate = to_f64(steps) / start.elapsed().as_secs_f64().max(1e-9);
    out.say(&format!("budget: {steps} steps, {rate:.1} steps/s"))?;
    Ok(steps)
}
