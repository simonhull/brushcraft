//! `page-in`: how long evaluating and pinning a region takes from an empty cache.
//!
//! It is measured with the project file's pages dropped from the page cache (cold file) and
//! still in it (warm file), and with the store's slot pool given back between samples (cold
//! pool) or kept (warm pool).
//!
//! Two regions, as the spec asks: (a) one brush footprint, the tiles holding about
//! `footprint` vertices (12 level-9 tiles for 50,000), and (b) a large region of
//! `region-quads` quads (10,254 level-9 tiles for 42M). Each sample picks a random place, takes
//! the tiles nearest it, empties the store, and times one [`store::TileStore::pin`] of them:
//! every tile evaluated coarse to fine from its corrections, and pinned with them. Cold-file
//! samples first drop the file's pages with `posix_fadvise(DONTNEED)`; each warm-file sample
//! repeats its cold-file sample's region right after.
//!
//! How the store is emptied is the pool state measured (`--pool`, see [`Pool`]):
//!
//! - **cold pool** (the default): `evict_all`, then `shrink_to(0)`. The chunks go back to the
//!   system, so every sample, warm-file ones too, also pays for mapping fresh chunks and faulting
//!   their pages in as the tiles fill them: zeroed pages from the kernel, a cost an app whose pool
//!   is already warm doesn't see. Its warm-file figure is so an upper bound on paging in from the
//!   page cache alone.
//! - **warm pool**: `evict_all` alone. Every slot is freed and goes back on its pool's free list,
//!   and the chunks stay mapped, so the next pin reuses them. Before its samples, the pool is
//!   warmed by pinning every sample's region once, untimed: the chunks the largest of them needs
//!   are mapped and their pages touched, as in an app that has been panning around. This is the
//!   state the spec's region target is judged on (cold file, warm pool).
//!
//! `--pool both` measures the cold pool, then the warm pool, over the same places. Each record
//! also gives the minor page faults per pin (median), which show whether the pool's pages were
//! already there, and, for the warm pool, how much the pool grew after its warm-up (zero when
//! every sample found its chunks mapped).

use std::fmt::Write as _;
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use geom::{TileKey, Vec3};
use store::TileStore;

use crate::cli::{Options, Pool};
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread, mib};
use crate::session;
use crate::surface::{nearest_tiles, tile_directions};
use crate::sys::{drop_page_cache, minor_faults};

/// Empties the store, every tile evicted (none is dirty here), leaving its pool in the state
/// `pool` names: chunks given back for [`Pool::Cold`], kept mapped for [`Pool::Warm`].
///
/// # Errors
///
/// If eviction fails.
fn empty(store: &mut TileStore, pool: Pool) -> Result<()> {
    store.evict_all()?;
    if pool == Pool::Cold {
        store.shrink_to(0)?;
    }
    Ok(())
}

/// One timed pin: how long it took, and the minor page faults the process took meanwhile.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// The pin's wall time.
    took: Duration,
    /// Minor page faults during it, every thread's.
    faults: u64,
}

/// One pin of `keys`, from the store as it is, timed (see [`Sample`]).
///
/// # Errors
///
/// If a tile can't be evaluated or pinned (more than the budget holds, for one), or the fault
/// count can't be read.
fn time_pin(store: &mut TileStore, keys: &[TileKey]) -> Result<Sample> {
    let faults_before = minor_faults()?;
    let start = Instant::now();
    let guard = store.pin(keys).context("evaluating and pinning")?;
    let took = start.elapsed();
    let faults = minor_faults()?.saturating_sub(faults_before);
    drop(guard);
    Ok(Sample { took, faults })
}

/// The median of `counts` (the lower middle one for an even count); `None` if there are none.
fn median(counts: impl IntoIterator<Item = u64>) -> Option<u64> {
    let mut sorted: Vec<u64> = counts.into_iter().collect();
    sorted.sort_unstable();
    sorted.get(sorted.len().checked_sub(1)? / 2).copied()
}

/// What one region's samples in one pool state measured.
#[derive(Debug, Default)]
struct Samples {
    /// Pins after the file's pages were dropped.
    cold_file: Vec<Sample>,
    /// Pins of the same regions right after, the file's pages cached.
    warm_file: Vec<Sample>,
    /// For [`Pool::Warm`]: bytes the pool held once warmed, before the first sample.
    warmed_bytes: Option<usize>,
    /// Bytes the pool held after the last sample.
    final_bytes: usize,
}

/// Measures a cold-file and a warm-file pin of each of `regions` (each a sample's tiles), with
/// the pool in state `pool` (see the module notes).
///
/// # Errors
///
/// If a pin fails, the page cache can't be dropped or the fault count read.
fn measure(
    store: &mut TileStore,
    paths: &Paths,
    pool: Pool,
    regions: &[Vec<TileKey>],
) -> Result<Samples> {
    let mut samples = Samples::default();
    if pool == Pool::Warm {
        for keys in regions {
            empty(store, pool)?;
            drop(store.pin(keys).context("warming the pool")?);
        }
        empty(store, pool)?;
        samples.warmed_bytes = Some(store.bytes());
    }
    for keys in regions {
        empty(store, pool)?;
        drop_page_cache(&paths.project)?;
        samples.cold_file.push(time_pin(store, keys)?);
        empty(store, pool)?;
        samples.warm_file.push(time_pin(store, keys)?);
    }
    samples.final_bytes = store.bytes();
    Ok(samples)
}

/// `record` and the human summary `line` with the spread of `samples` added under `prefix`
/// (`cold` or `warm`, the file's state): times and the median minor faults.
///
/// # Errors
///
/// If writing to `line` fails.
fn add_samples(
    mut record: Record,
    line: &mut String,
    prefix: &str,
    samples: &[Sample],
) -> Result<Record> {
    let times: Vec<Duration> = samples.iter().map(|s| s.took).collect();
    if let Some(spread) = Spread::of(&times) {
        record = spread.add_to(record, prefix);
        write!(
            line,
            " {prefix} file median {:.2} ms ({:.2} to {:.2});",
            spread.median, spread.min, spread.max
        )?;
    }
    if let Some(faults) = median(samples.iter().map(|s| s.faults)) {
        record = record.with(format!("{prefix}_faults_median"), faults);
    }
    Ok(record)
}

/// Runs `page-in` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, a pin fails, the page cache can't be dropped, or writing
/// the report fails.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let level = session::level(options)?;
    let mut project = session::open(options, paths)?;
    let tiles = tile_directions(project.store().mesh(), level)?;
    let quads = u64::from(level.tile_quads()).pow(2);
    let vertices_per_tile = (level.tile_quads() as usize + 1).pow(2);
    let sizes = [
        ("footprint", options.footprint.div_ceil(vertices_per_tile)),
        (
            "region",
            usize::try_from(options.region_quads.div_ceil(quads))?,
        ),
    ];
    let mut rng = Rng::new(options.seed);
    for (name, n) in sizes {
        let n = n.min(tiles.len());
        let places: Vec<Vec3> = (0..options.samples).map(|_| rng.direction()).collect();
        let regions: Vec<Vec<TileKey>> = places
            .iter()
            .map(|&direction| nearest_tiles(&tiles, direction, n))
            .collect();
        for &pool in options.pools.each() {
            out.say(&format!(
                "page-in {name}, {} pool: {n} level-{level} tiles, {} samples cold and warm",
                pool.name(),
                options.samples
            ))?;
            let samples = measure(project.store(), paths, pool, &regions)?;
            let result = match pool {
                Pool::Cold => name.to_owned(),
                Pool::Warm => format!("{name}-warm-pool"),
            };
            let mut record = Record::new("page-in", &result)
                .with("pool", pool.name())
                .with("level", u64::from(level.get()))
                .with("tiles", n)
                .with("vertices", n * vertices_per_tile)
                .with("quads", u64::try_from(n)? * quads)
                .with("budget_mib", options.budget_mib)
                .with("pool_mib", mib(u64::try_from(samples.final_bytes)?));
            let mut line = format!("page-in {name} ({n} tiles), {} pool:", pool.name());
            if let Some(warmed) = samples.warmed_bytes {
                let growth = samples.final_bytes.saturating_sub(warmed);
                record = record.with("pool_growth_mib", mib(u64::try_from(growth)?));
                write!(
                    line,
                    " grew {:.0} MiB after warming;",
                    mib(u64::try_from(growth)?)
                )?;
            }
            record = add_samples(record, &mut line, "cold", &samples.cold_file)?;
            record = add_samples(record, &mut line, "warm", &samples.warm_file)?;
            out.record(&record)?;
            out.say(&line)?;
        }
    }
    project.close(false).context("closing the project")?;
    Ok(())
}
