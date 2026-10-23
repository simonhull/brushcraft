//! `page-in`: how long evaluating and pinning a region takes from an empty cache, with the
//! project file's pages dropped from the page cache (cold) and still in it (warm).
//!
//! Two regions, as the spec asks: (a) one brush footprint, the tiles holding about
//! `footprint` vertices (12 level-9 tiles for 50,000), and (b) a large region of
//! `region-quads` quads (10,254 level-9 tiles for 42M). Each sample picks a random place, takes
//! the tiles nearest it, empties the store (`evict_all` + `shrink_to(0)`), and times one
//! [`store::TileStore::pin`] of them: every tile evaluated coarse to fine from its corrections,
//! and pinned with them. Cold samples first drop the file's pages with
//! `posix_fadvise(DONTNEED)`; each warm sample repeats its cold sample's region right after.

use std::fmt::Write as _;
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use geom::TileKey;
use store::TileStore;

use crate::cli::Options;
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread};
use crate::session;
use crate::surface::{nearest_tiles, tile_directions};
use crate::sys::drop_page_cache;

/// Empties the store: every tile evicted (none is dirty here) and every chunk given back.
///
/// # Errors
///
/// If eviction fails.
fn empty(store: &mut TileStore) -> Result<()> {
    store.evict_all()?;
    store.shrink_to(0)?;
    Ok(())
}

/// How long one pin of `keys` takes, from the store as it is.
///
/// # Errors
///
/// If a tile can't be evaluated or pinned (more than the budget holds, for one).
fn time_pin(store: &mut TileStore, keys: &[TileKey]) -> Result<Duration> {
    let start = Instant::now();
    let guard = store.pin(keys).context("evaluating and pinning")?;
    let took = start.elapsed();
    drop(guard);
    Ok(took)
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
    let regions = [
        ("footprint", options.footprint.div_ceil(vertices_per_tile)),
        (
            "region",
            usize::try_from(options.region_quads.div_ceil(quads))?,
        ),
    ];
    let mut rng = Rng::new(options.seed);
    for (name, n) in regions {
        let n = n.min(tiles.len());
        out.say(&format!(
            "page-in {name}: {n} level-{level} tiles, {} samples cold and warm",
            options.samples
        ))?;
        let (mut cold, mut warm) = (Vec::new(), Vec::new());
        for _ in 0..options.samples {
            let direction = rng.direction();
            let keys = nearest_tiles(&tiles, direction, n);
            let store = project.store();
            empty(store)?;
            drop_page_cache(&paths.project)?;
            cold.push(time_pin(store, &keys)?);
            empty(store)?;
            warm.push(time_pin(store, &keys)?);
        }
        let mut record = Record::new("page-in", name)
            .with("level", u64::from(level.get()))
            .with("tiles", n)
            .with("vertices", n * vertices_per_tile)
            .with("quads", u64::try_from(n)? * quads)
            .with("budget_mib", options.budget_mib);
        let mut line = format!("page-in {name} ({n} tiles):");
        for (mode, samples) in [("cold", &cold), ("warm", &warm)] {
            if let Some(spread) = Spread::of(samples) {
                record = spread.add_to(record, mode);
                write!(
                    line,
                    " {mode} median {:.2} ms ({:.2} to {:.2});",
                    spread.median, spread.min, spread.max
                )?;
            }
        }
        out.record(&record)?;
        out.say(&line)?;
    }
    project.close(false).context("closing the project")?;
    Ok(())
}
