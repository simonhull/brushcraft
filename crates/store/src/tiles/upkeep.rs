//! Keeping the display layer up to date: building display tiles one at a time from what the
//! store holds, as budgeted jobs (see [`crate::jobs`]).
//!
//! A display tile at level `l` is built from (see `geom::build_tile`):
//!
//! - the finest positions under it, subsampled, and the block one level finer (the finest positions
//!   there at `finest − 1`, else the decoded display tiles at `l + 1`, whose bounds chain into its
//!   own);
//! - its mean normals, the tent of the means one level finer around its vertices, read across face
//!   borders: the finest normals at `finest − 1`, else the decoded display tiles at `l + 1`, its
//!   own face's and its neighbours'.
//!
//! So building a tile can find a display tile it reads unknown, always one level finer: that one
//! is built first. Every such dependency is one level finer, and the finest level is always
//! there (it is evaluated), so the work ends. It can also find a finest tile it reads not
//! evaluated: a slice evaluates a bounded batch of those first.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use geom::{
    Channel, DisplayError, DisplayTile, FaceId, Grid, LayerId, Level, MeanNormals, PositionSource,
    Region, TileKey, Vec3, build_tile, mip_normals_tile,
};

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use crate::jobs::{Job, JobsReport, Progress, Upkeep};

/// The most finest tiles one slice evaluates for a display tile: a whole-face tile reads all of
/// its face's, which would make one slice of many milliseconds.
const EVALUATE_BATCH: usize = 4;

/// What trying to build one display tile came to.
#[derive(Debug)]
enum Attempt {
    /// The tile, built.
    Built(Box<DisplayTile>),
    /// Finest tiles it reads aren't evaluated as far as it needs: evaluate these first.
    Evaluate(Vec<TileKey>, Evaluation),
    /// Display tiles one level finer it reads are unknown: build these first.
    Needs(Vec<TileKey>),
}

/// The base displacement tile of `level` holding face-grid vertex `(i, j)` of `face`, and the
/// vertex's index in its row-major values.
fn containing(
    channel: Channel,
    face: FaceId,
    level: Level,
    i: u32,
    j: u32,
) -> Option<(TileKey, usize)> {
    let (key, li, lj) = TileKey::containing(LayerId::BASE, channel, face, level, i, j)?;
    Some((key, usize::try_from(lj * key.side() + li).ok()?))
}

/// The finest unit normals the store has evaluated, read as a [`PositionSource`] for the mip of
/// the level above the finest; every tile it lacks is noted.
struct FinestNormals<'a> {
    /// The store, read without evaluating.
    store: &'a TileStore,
    /// The tiles whose normals a read found missing.
    misses: RefCell<BTreeSet<TileKey>>,
}

impl PositionSource for FinestNormals<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, index) = containing(Channel::DISPLACEMENT, face, level, i, j)?;
        let found = self
            .store
            .cached_normals(&key)
            .and_then(|n| n.get(index).copied());
        if found.is_none() {
            self.misses.borrow_mut().insert(key);
        }
        found
    }
}

/// Decoded display tiles' means, read as a [`PositionSource`] for the mip of the level above
/// theirs; every tile it lacks is noted.
struct DisplayMeans<'a> {
    /// The means of the display tiles loaded, by key.
    tiles: &'a BTreeMap<TileKey, Vec<Vec3>>,
    /// The tiles a read found missing.
    misses: RefCell<BTreeSet<TileKey>>,
}

impl PositionSource for DisplayMeans<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, index) = containing(Channel::DISPLAY, face, level, i, j)?;
        let found = self.tiles.get(&key).and_then(|m| m.get(index).copied());
        if found.is_none() {
            self.misses.borrow_mut().insert(key);
        }
        found
    }
}

/// The display tiles of `level` under `region` (of the level above), row by row.
fn tiles_under(region: Region, level: Level) -> Vec<TileKey> {
    let tq = level.tile_quads();
    let (x0, y0) = (2 * region.x0() / tq, 2 * region.y0() / tq);
    let count = (2 * region.quads()).div_ceil(tq).max(1);
    (y0..y0 + count)
        .flat_map(|ty| (x0..x0 + count).map(move |tx| (tx, ty)))
        .filter_map(|(tx, ty)| {
            TileKey::display(
                region.face(),
                level,
                u16::try_from(tx).ok()?,
                u16::try_from(ty).ok()?,
            )
            .ok()
        })
        .collect()
}

/// The displacement tiles of `level` holding a vertex of columns `x0..=x1`, rows `y0..=y1` of
/// `face` (clamped to the face).
fn finest_tiles(
    face: FaceId,
    level: Level,
    (x0, x1): (u32, u32),
    (y0, y1): (u32, u32),
) -> Vec<TileKey> {
    let tq = level.tile_quads();
    let last = level.tiles_per_side() - 1;
    let range = |lo: u32, hi: u32| (lo.div_ceil(tq).saturating_sub(1))..=(hi / tq).min(last);
    range(y0, y1)
        .flat_map(|ty| range(x0, x1).map(move |tx| (tx, ty)))
        .filter_map(|(tx, ty)| {
            TileKey::displacement(
                face,
                level,
                u16::try_from(tx).ok()?,
                u16::try_from(ty).ok()?,
            )
            .ok()
        })
        .collect()
}

impl TileStore {
    /// Queues bringing up to date the display tiles under `finest` that writes since made
    /// unknown, if any are and no job is queued for them.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if settling the writes fails (never, for tiles of the store's
    /// mesh).
    fn queue_refresh(&mut self, finest: Level) -> Result<(), CacheError> {
        if !self.jobs.is_empty() {
            return Ok(());
        }
        self.display.settle(&self.mesh, &self.deps)?;
        let unknown = self.display.unknown_keys();
        if !unknown.is_empty() {
            self.jobs
                .push_back(Job::Refresh(Upkeep::new(finest, unknown)));
        }
        Ok(())
    }

    /// For a store keeping a display layer, queues building every display tile its files have no
    /// record of, or an unknown one ([`Job::Rebuild`]): a file without a display layer, a
    /// rebuild cut short, a save made before a refresh. Unknown records are told by their length
    /// alone ([`crate::display`] writes them with an empty body), so nothing is read.
    pub(super) fn queue_missing_display(&mut self) {
        let Some(finest) = self.display_finest else {
            return;
        };
        let unknown_len = crate::file::unknown_display_len();
        let missing: Vec<TileKey> = self
            .all_display_keys(finest)
            .into_iter()
            .filter(|key| {
                let loc = self
                    .file
                    .locate(key)
                    .or_else(|| self.base.as_ref().and_then(|b| b.locate(key)));
                match self.display.kept(key) {
                    Some(crate::bounds::Kept::Unknown) => true,
                    Some(_) => false,
                    None => loc.is_none_or(|loc| Some(loc.len()) == unknown_len),
                }
            })
            .collect();
        if !missing.is_empty() {
            self.jobs
                .push_back(Job::Rebuild(Upkeep::new(finest, missing)));
        }
    }

    /// Every display tile key of the store's mesh at every level below `finest`.
    fn all_display_keys(&self, finest: Level) -> Vec<TileKey> {
        self.mesh
            .faces()
            .flat_map(move |face| {
                (0..finest.get()).flat_map(move |l| {
                    let level = Level::new(l).unwrap_or(Level::ZERO);
                    let per_side = u16::try_from(level.tiles_per_side()).unwrap_or(0);
                    (0..per_side).flat_map(move |ty| {
                        (0..per_side)
                            .filter_map(move |tx| TileKey::display(face, level, tx, ty).ok())
                    })
                })
            })
            .collect()
    }

    /// Queues building every display tile under `finest` the store doesn't know, as a rebuild
    /// job: for a file without a display layer.
    pub fn queue_display_rebuild(&mut self, finest: Level) {
        let keys = self.all_display_keys(finest);
        self.jobs.push_back(Job::Rebuild(Upkeep::new(finest, keys)));
    }

    /// Runs the queued jobs in slices until `deadline` (and, for a store keeping a display
    /// layer, queues a refresh first when writes left display tiles unknown). Each job keeps its
    /// place; the next call goes on from there. A slice is bounded (one display tile built, or
    /// a few finest tiles evaluated for one), so the deadline is honoured to within one slice.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing what a write made stale fails (never, for tiles of the
    /// store's mesh); [`CacheError::Display`] if a display tile won't build; the errors of
    /// [`TileStore::evaluate_many`]. The job stays queued, at the tile that failed.
    pub fn run_jobs(&mut self, deadline: Instant) -> Result<JobsReport, CacheError> {
        let mut report = JobsReport::default();
        if let Some(finest) = self.display_finest {
            self.queue_refresh(finest)?;
        }
        while Instant::now() < deadline {
            let Some(mut job) = self.jobs.pop_front() else {
                break;
            };
            let began = Instant::now();
            let progress = self.slice(&mut job, &mut report);
            report.longest_slice = report.longest_slice.max(began.elapsed());
            match progress {
                Ok(Progress::Done) => report.finished += 1,
                Ok(Progress::More) => self.jobs.push_front(job),
                Err(e) => {
                    self.jobs.push_front(job);
                    return Err(e);
                }
            }
            report.slices += 1;
        }
        report.left = self.jobs.iter().map(Job::left).sum();
        Ok(report)
    }

    /// Queues building the display tiles of `level` alone (a level below `finest`) the store
    /// doesn't know: for timing a rebuild level by level, finest first.
    pub fn queue_display_level(&mut self, finest: Level, level: Level) {
        let keys: Vec<TileKey> = self
            .all_display_keys(finest)
            .into_iter()
            .filter(|key| key.level() == level)
            .collect();
        self.jobs.push_back(Job::Rebuild(Upkeep::new(finest, keys)));
    }

    /// Drops every queued job, where it is: what they had built stays.
    pub fn clear_jobs(&mut self) {
        self.jobs.clear();
    }

    /// Whether any job is queued.
    #[must_use]
    pub fn has_jobs(&self) -> bool {
        !self.jobs.is_empty()
    }

    /// One slice of `job`: builds its next display tile, or evaluates a batch of the finest
    /// tiles that one reads, or queues the finer display tiles it needs first.
    ///
    /// # Errors
    ///
    /// As [`TileStore::run_jobs`]; the tile stays to build.
    fn slice(&mut self, job: &mut Job, report: &mut JobsReport) -> Result<Progress, CacheError> {
        let upkeep = job.upkeep();
        let finest = upkeep.finest;
        let Some(key) = upkeep.next() else {
            return Ok(Progress::Done);
        };
        if self.knows_display(&key)? {
            return Ok(if upkeep.is_empty() {
                Progress::Done
            } else {
                Progress::More
            });
        }
        match self.try_build_display(key, finest) {
            Ok(Attempt::Built(tile)) => {
                self.record_display(std::slice::from_ref(&*tile))?;
                // A slice is an operation: its display tile is an epoch of its own.
                self.close_epoch()?;
                report.built += 1;
            }
            Ok(Attempt::Evaluate(keys, what)) => {
                let batch: Vec<TileKey> = keys.into_iter().take(EVALUATE_BATCH).collect();
                upkeep.push(key);
                self.evaluate_many(&batch, what)?;
                report.evaluated += batch.len();
            }
            Ok(Attempt::Needs(keys)) => {
                upkeep.push(key);
                for dep in keys {
                    upkeep.push(dep);
                }
            }
            Err(e) => {
                upkeep.push(key);
                return Err(e);
            }
        }
        Ok(if upkeep.is_empty() {
            Progress::Done
        } else {
            Progress::More
        })
    }

    /// Builds display tile `key` (a level below `finest`) from what the store holds, or says
    /// what it lacks (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Display`] if the tile won't build from its inputs; [`CacheError::Store`]
    /// if a display record can't be read.
    fn try_build_display(&mut self, key: TileKey, finest: Level) -> Result<Attempt, CacheError> {
        let level = key.level();
        let finer = level.child().ok_or(DisplayError::NoDisplayLevel(level))?;
        let shift = finest.get() - level.get();
        let region = Region::of_tile(&key);
        let face = key.face();
        // The finest tiles whose positions the tile reads: those under it, and, when its finer
        // level is the finest, its normals' apron on its own face too.
        let span = |v: u32| v << shift;
        let apron = u32::from(finer == finest);
        let n = finest.face_quads();
        let xs = (
            span(region.x0()).saturating_sub(apron),
            (span(region.x0() + region.quads()) + apron).min(n),
        );
        let ys = (
            span(region.y0()).saturating_sub(apron),
            (span(region.y0() + region.quads()) + apron).min(n),
        );
        // Normals only where its own means read the finest level.
        let what = if finer == finest {
            Evaluation::Normals
        } else {
            Evaluation::Positions
        };
        let missing: Vec<TileKey> = finest_tiles(face, finest, xs, ys)
            .into_iter()
            .filter(|t| match what {
                Evaluation::Normals => self.cached_normals(t).is_none(),
                Evaluation::Positions => self.cached_positions(t).is_none(),
            })
            .collect();
        if !missing.is_empty() {
            return Ok(Attempt::Evaluate(missing, what));
        }
        let finest_at = |store: &Self, x: u32, y: u32| -> Option<Vec3> {
            let (tile, index) = containing(Channel::DISPLACEMENT, face, finest, x, y)?;
            store.cached_positions(&tile)?.get(index).copied()
        };
        let side = key.side();
        let exact = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .map(|(i, j)| finest_at(self, span(region.x0() + i), span(region.y0() + j)))
            .collect::<Option<Vec<Vec3>>>()
            .ok_or(DisplayError::Unavailable)?;
        let exact = Grid::new(side, exact)?;
        let fine_side = 2 * region.quads() + 1;
        let (finer_block, children_bound, means) = if finer == finest {
            let block = (0..fine_side)
                .flat_map(|j| (0..fine_side).map(move |i| (i, j)))
                .map(|(i, j)| finest_at(self, 2 * region.x0() + i, 2 * region.y0() + j))
                .collect::<Option<Vec<Vec3>>>()
                .ok_or(DisplayError::Unavailable)?;
            let source = FinestNormals {
                store: self,
                misses: RefCell::new(BTreeSet::new()),
            };
            let means = mip_normals_tile(&self.mesh, &source, &key);
            let misses = source.misses.into_inner();
            if !misses.is_empty() {
                return Ok(Attempt::Evaluate(
                    misses.into_iter().collect(),
                    Evaluation::Normals,
                ));
            }
            (Grid::new(fine_side, block)?, 0.0, means?)
        } else {
            match self.finer_display(&key, finer)? {
                Ok(inputs) => inputs,
                Err(needs) => return Ok(Attempt::Needs(needs)),
            }
        };
        Ok(Attempt::Built(Box::new(build_tile(
            key,
            &exact,
            &finer_block,
            children_bound,
            &means,
        )?)))
    }

    /// The inputs of display tile `key` one level finer (`finer`, not the finest): the decoded
    /// positions of the display tiles under it, their largest bound, and its means from the
    /// display tiles around it, its neighbours' included. `Err` with the display tiles of
    /// `finer` it reads that are unknown.
    ///
    /// # Errors
    ///
    /// [`CacheError::Display`] if the tiles don't fit; [`CacheError::Store`] if a record can't
    /// be read.
    #[expect(
        clippy::type_complexity,
        reason = "one call's result: the three inputs, or the tiles still needed"
    )]
    fn finer_display(
        &mut self,
        key: &TileKey,
        finer: Level,
    ) -> Result<Result<(Grid, f32, MeanNormals), Vec<TileKey>>, CacheError> {
        let region = Region::of_tile(key);
        let children = tiles_under(region, finer);
        let mut loaded: BTreeMap<TileKey, DisplayTile> = BTreeMap::new();
        let mut needs = Vec::new();
        for child in &children {
            match self
                .stored_display(std::slice::from_ref(child))?
                .pop()
                .flatten()
            {
                Some(tile) => {
                    loaded.insert(*child, tile);
                }
                None => needs.push(*child),
            }
        }
        if !needs.is_empty() {
            return Ok(Err(needs));
        }
        let fine_side = 2 * region.quads() + 1;
        let mut block = vec![Vec3::ZERO; usize::try_from(fine_side * fine_side).unwrap_or(0)];
        let mut bound = 0.0_f32;
        for tile in loaded.values() {
            bound = bound.max(tile.error());
            let at = Region::of_tile(tile.key());
            let tile_side = tile.key().side();
            for (index, &p) in tile.positions().iter().enumerate() {
                let index = u32::try_from(index).unwrap_or(u32::MAX);
                let x = at.x0() + index % tile_side - 2 * region.x0();
                let y = at.y0() + index / tile_side - 2 * region.y0();
                if let Some(slot) = usize::try_from(y * fine_side + x)
                    .ok()
                    .and_then(|i| block.get_mut(i))
                {
                    *slot = p;
                }
            }
        }
        // The means: the tiles under it, then whatever the tent reads around it, loaded as the
        // mip asks for them.
        let mut means_of: BTreeMap<TileKey, Vec<Vec3>> =
            loaded.iter().map(|(k, t)| (*k, t.means())).collect();
        loop {
            let source = DisplayMeans {
                tiles: &means_of,
                misses: RefCell::new(BTreeSet::new()),
            };
            let means = mip_normals_tile(&self.mesh, &source, key);
            let misses = source.misses.into_inner();
            if misses.is_empty() {
                return Ok(Ok((Grid::new(fine_side, block)?, bound, means?)));
            }
            let mut unknown = Vec::new();
            for miss in misses {
                match self
                    .stored_display(std::slice::from_ref(&miss))?
                    .pop()
                    .flatten()
                {
                    Some(tile) => {
                        means_of.insert(miss, tile.means());
                    }
                    None => unknown.push(miss),
                }
            }
            if !unknown.is_empty() {
                return Ok(Err(unknown));
            }
        }
    }
}
