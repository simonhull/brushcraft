//! Keeping the display layer up to date: building display tiles on the store's workers from
//! what the store holds, as budgeted jobs (see [`crate::jobs`]).
//!
//! A display tile at level `l` is built from (see `geom::build_tile`) the block under it one
//! level finer, whose even vertices it stores (the finest positions at `finest − 1`, else the
//! decoded display tiles at `l + 1`, whose bounds and normal cones chain into its own; at
//! `finest − 1` its cone covers the finest normals of the block), and its mean normals, the
//! tent of the means one level finer around its vertices, read across face borders (the finest
//! normals at `finest − 1`, else the decoded display tiles at `l + 1`, its own face's and its
//! neighbours'). What it reads is known before it is built ([`crate::display::tent_reads`]).
//!
//! The owner thread hands a worker a tile once everything it reads is there, with those inputs,
//! owned: the finest tiles' values copied once each for the job (shared by every tile reading
//! them), or the finer display tiles decoded (by the job, or by the worker from a record, kept
//! for the job afterwards). The worker assembles the blocks, runs the normals' mip and encodes;
//! the owner records what comes back. A result built from inputs a correction change has since
//! outdated (a write, a lost tile's revert) is dropped and the tile built again. A tile whose
//! inputs aren't there waits: for the display tiles one level finer it reads (unknown, so built
//! first: every dependency is one level finer and the finest level is evaluated, so the work
//! ends), or for a slice that evaluates a few of the finest tiles it reads.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use geom::{
    BaseMesh, Channel, DisplayError, DisplayTile, FaceId, Grid, LayerId, Level, NormalCone,
    PositionSource, Region, TileKey, Vec3, build_tile, mip_normals_tile,
};

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use super::queued::Turn;
use crate::bounds::Kept;
use crate::display::{DisplayRecord, StoredDisplay, tent_reads};
use crate::file::{FileError, FileReader, Loc, display_record, unknown_display_len};
use crate::jobs::{
    Cone, ConeSteps, DecodedTile, FinestTile, Finished, Job, JobsReport, Outcome, Progress, Upkeep,
};

/// How many pending tiles a slice looks at per tile it can hand out: it skips known, failed
/// and unready tiles, and this bounds how long it looks.
const LOOK_AHEAD: usize = 4;

/// The most finest tiles a slice copies for the workers (about 100 KB each at level 9): the
/// rest wait for the next slices.
const COPY_BATCH: usize = 8;

/// The most corrections of a cone a step asks the prefetch for: a few rounds' worth, so they
/// land a few at a time.
const PREFETCH_AHEAD: usize = 2 * crate::jobs::ROUND_MAX;

/// About how long a slice spends handing tiles out (gathering and copying their inputs): it
/// stops looking once past it, so a slice stays well under a millisecond.
const HAND_OUT_TIME: std::time::Duration = std::time::Duration::from_micros(300);

/// The base tile of `channel` at `level` holding face-grid vertex `(i, j)` of `face`, and the
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

/// Tiles' values by key, read as a [`PositionSource`] for a tile's mip.
///
/// The finest normals, or the decoded means of display tiles. Remembers the last tile read (the
/// mip reads row by row, so most reads fall in it) and whether any read missed.
struct TileValues<'a> {
    /// The channel the tiles are keyed under.
    channel: Channel,
    /// The values, by tile.
    tiles: &'a BTreeMap<TileKey, &'a [Vec3]>,
    /// The tile the last read found.
    last: Cell<Option<(TileKey, &'a [Vec3])>>,
    /// Whether a read found its tile missing.
    missed: Cell<bool>,
}

impl<'a> TileValues<'a> {
    /// `tiles`, keyed under `channel`.
    const fn new(channel: Channel, tiles: &'a BTreeMap<TileKey, &'a [Vec3]>) -> Self {
        Self {
            channel,
            tiles,
            last: Cell::new(None),
            missed: Cell::new(false),
        }
    }
}

impl PositionSource for TileValues<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, index) = containing(self.channel, face, level, i, j)?;
        let values = match self.last.get() {
            Some((last, values)) if last == key => Some(values),
            _ => {
                let found = self.tiles.get(&key).copied();
                if let Some(values) = found {
                    self.last.set(Some((key, values)));
                }
                found
            }
        };
        let found = values.and_then(|v| v.get(index).copied());
        if found.is_none() {
            self.missed.set(true);
        }
        found
    }

    fn row(&self, face: FaceId, level: Level, i: u32, j: u32, out: &mut [Vec3]) -> Option<()> {
        // Runs of whole tile rows, copied.
        let tq = level.tile_quads();
        let mut x = i;
        let mut at = 0;
        while at < out.len() {
            let Some((key, index)) = containing(self.channel, face, level, x, j) else {
                self.missed.set(true);
                return None;
            };
            let tile_end = Region::of_tile(&key).x0() + tq;
            let count = usize::try_from(tile_end - x + 1).ok()?.min(out.len() - at);
            let Some(run) = self
                .tiles
                .get(&key)
                .and_then(|values| values.get(index..index + count))
            else {
                self.missed.set(true);
                return None;
            };
            out.get_mut(at..at + count)?.copy_from_slice(run);
            at += count;
            x += u32::try_from(count).ok()?;
        }
        Some(())
    }
}

/// The block of `side` × `side` vertices of `face` at `level` from `(x0, y0)`, put together row
/// by row from whole tiles of `channel` in `tiles`; `None` if one it needs is missing.
fn block_from(
    tiles: &BTreeMap<TileKey, &[Vec3]>,
    channel: Channel,
    face: FaceId,
    level: Level,
    (x0, y0): (u32, u32),
    side: u32,
) -> Option<Vec<Vec3>> {
    let tq = level.tile_quads();
    let mut out = Vec::with_capacity(usize::try_from(side * side).ok()?);
    for y in y0..y0 + side {
        let mut x = x0;
        while x < x0 + side {
            let (key, index) = containing(channel, face, level, x, y)?;
            // The run to the end of the row, or of the tile's row (a tile holds `tq + 1`
            // vertices a row, from its own `x0`).
            let tile_x0 = Region::of_tile(&key).x0();
            let end = (tile_x0 + tq).min(x0 + side - 1);
            let count = usize::try_from(end - x + 1).ok()?;
            out.extend_from_slice(tiles.get(&key)?.get(index..index + count)?);
            x = end + 1;
        }
    }
    Some(out)
}

/// Whether `inner` (a region one level finer than `outer`) lies within `outer`'s area: a tile
/// under it, not one of its neighbours.
fn lies_under(inner: Region, outer: Region) -> bool {
    inner.face() == outer.face()
        && inner.x0() >= 2 * outer.x0()
        && inner.y0() >= 2 * outer.y0()
        && inner.x0() + inner.quads() <= 2 * (outer.x0() + outer.quads())
        && inner.y0() + inner.quads() <= 2 * (outer.y0() + outer.quads())
}

/// Where a worker finds a display tile one level finer to decode.
#[derive(Debug, Clone)]
enum Finer {
    /// The store holds its record, copied for the worker to decode.
    Record(DisplayRecord),
    /// A file holds it: the worker reads its record there.
    File(FileReader, Loc),
}

/// What a worker builds or decodes a display tile from, owned by the worker.
#[derive(Debug)]
enum TaskInputs {
    /// Build it, one level above the finest: the finest tiles it reads.
    Finest(BTreeMap<TileKey, FinestTile>),
    /// Build it, higher: the display tiles one level finer it reads, decoded.
    Display(BTreeMap<TileKey, Arc<DecodedTile>>),
    /// Decode it, for the tiles of the level above: where its record is.
    Decode(Finer),
}

/// One display tile for a worker to build, with all it reads.
#[derive(Debug)]
struct Task {
    /// The tile.
    key: TileKey,
    /// The store's count of correction changes the inputs were taken at.
    as_of: u64,
    /// The generation of the tile's record when it was handed out (see
    /// [`TileStore::display_generation`]): what a decode decodes.
    record: u64,
    /// The store's mesh.
    mesh: Arc<BaseMesh>,
    /// The store's display layer's finest level: a record built from another is unknown.
    finest: Level,
    /// What it reads.
    inputs: TaskInputs,
    /// Whether it fails as one that won't build would (a test's hook).
    unbuildable: bool,
}

/// Decodes display tile `key` from where it is: `None` if its record is unknown, built from
/// another finest level than `finest`, damaged, or doesn't decode (it is then rebuilt).
///
/// # Errors
///
/// [`CacheError::Store`] if its record can't be read.
fn decode_finer(
    key: TileKey,
    place: &Finer,
    finest: Level,
) -> Result<Option<DisplayTile>, CacheError> {
    // Read from the file into this; a record held in memory is decoded where it is.
    let read: Option<DisplayRecord>;
    let record: Option<&DisplayRecord> = match place {
        Finer::Record(record) => Some(record),
        Finer::File(reader, loc) => {
            read = match reader.read_record(*loc) {
                Ok(raw) => display_record(&raw, &key).and_then(StoredDisplay::known),
                Err(FileError::Damaged { .. }) => None,
                Err(e) => return Err(CacheError::Store(e)),
            };
            read.as_ref()
        }
    };
    Ok(record
        .filter(|r| r.finest == finest)
        .and_then(|r| r.decode(&key)))
}

/// Builds or decodes `task`'s display tile (on a worker): see the module notes.
fn run_task(task: &Task) -> Outcome {
    if task.unbuildable {
        return Outcome::Failed(DisplayError::TooLarge);
    }
    let key = task.key;
    let built = match &task.inputs {
        TaskInputs::Decode(place) => {
            return match decode_finer(key, place, task.finest) {
                Ok(Some(tile)) => Outcome::Decoded(Box::new(DecodedTile::new(tile))),
                Ok(None) => Outcome::Unreadable,
                Err(e) => Outcome::Error(e),
            };
        }
        TaskInputs::Display(tiles) => build_from_finer(&task.mesh, key, tiles),
        TaskInputs::Finest(tiles) => build_from_finest(&task.mesh, key, tiles),
    };
    match built {
        Ok(tile) => Outcome::Built(Box::new(DecodedTile::new(tile))),
        Err(why) => Outcome::Failed(why),
    }
}

/// Builds display tile `key` (one level above the finest) from the finest tiles it reads.
///
/// # Errors
///
/// [`DisplayError`] if the tile won't build from them (or they don't hold what it reads).
fn build_from_finest(
    mesh: &BaseMesh,
    key: TileKey,
    tiles: &BTreeMap<TileKey, FinestTile>,
) -> Result<DisplayTile, DisplayError> {
    let region = Region::of_tile(&key);
    let finer = key
        .level()
        .child()
        .ok_or_else(|| DisplayError::NoDisplayLevel(key.level()))?;
    let fine_side = 2 * region.quads() + 1;
    let origin = (2 * region.x0(), 2 * region.y0());
    let positions: BTreeMap<TileKey, &[Vec3]> =
        tiles.iter().map(|(k, t)| (*k, &*t.positions)).collect();
    let normals: BTreeMap<TileKey, &[Vec3]> =
        tiles.iter().map(|(k, t)| (*k, &*t.normals)).collect();
    let block = block_from(
        &positions,
        Channel::DISPLACEMENT,
        key.face(),
        finer,
        origin,
        fine_side,
    )
    .ok_or(DisplayError::Unavailable)?;
    // The finest normals under the tile, row by row, as `geom::build_cones` reads them.
    let under = block_from(
        &normals,
        Channel::DISPLACEMENT,
        key.face(),
        finer,
        origin,
        fine_side,
    )
    .ok_or(DisplayError::Unavailable)?;
    let cone = NormalCone::covering(&under)?;
    let source = TileValues::new(Channel::DISPLACEMENT, &normals);
    let means = mip_normals_tile(mesh, &source, &key);
    if source.missed.get() {
        return Err(DisplayError::Unavailable);
    }
    build_tile(key, &Grid::new(fine_side, block)?, 0.0, cone, &means?)
}

/// Builds display tile `key` from the decoded display tiles one level finer it reads.
///
/// # Errors
///
/// [`DisplayError`] if the tile won't build from them (or they don't hold what it reads).
fn build_from_finer(
    mesh: &BaseMesh,
    key: TileKey,
    tiles: &BTreeMap<TileKey, Arc<DecodedTile>>,
) -> Result<DisplayTile, DisplayError> {
    let region = Region::of_tile(&key);
    let finer = key
        .level()
        .child()
        .ok_or_else(|| DisplayError::NoDisplayLevel(key.level()))?;
    let fine_side = 2 * region.quads() + 1;
    let origin = (2 * region.x0(), 2 * region.y0());
    // The bound and the cone chain from the tiles under it alone; the block reads any tile
    // holding a vertex of it (one on its edge may come from a neighbour, the same bits there).
    let under: Vec<&DisplayTile> = tiles
        .iter()
        .filter(|(tile, _)| lies_under(Region::of_tile(tile), region))
        .map(|(_, d)| &d.tile)
        .collect();
    let bound = under.iter().map(|t| t.error()).fold(0.0_f32, f32::max);
    let cones: Vec<NormalCone> = under.iter().map(|t| t.cone()).collect();
    let cone = NormalCone::holding(&cones)?;
    let positions: BTreeMap<TileKey, &[Vec3]> = tiles
        .iter()
        .map(|(k, d)| (*k, d.tile.positions()))
        .collect();
    let block = block_from(
        &positions,
        Channel::DISPLAY,
        key.face(),
        finer,
        origin,
        fine_side,
    )
    .ok_or(DisplayError::Unavailable)?;
    let means: BTreeMap<TileKey, &[Vec3]> = tiles
        .iter()
        .map(|(k, d)| (*k, d.means.as_slice()))
        .collect();
    let source = TileValues::new(Channel::DISPLAY, &means);
    let mean_normals = mip_normals_tile(mesh, &source, &key);
    if source.missed.get() {
        return Err(DisplayError::Unavailable);
    }
    build_tile(
        key,
        &Grid::new(fine_side, block)?,
        bound,
        cone,
        &mean_normals?,
    )
}

/// What the owner found a pending tile to need.
#[derive(Debug)]
enum Readiness {
    /// Everything it reads is there: hand it to a worker.
    Ready,
    /// Finest tiles it reads aren't evaluated: evaluate these first.
    Evaluate(Vec<TileKey>),
    /// Display tiles one level finer it reads are unknown or still being built: build these
    /// first (those being built, it waits for).
    Needs(Vec<TileKey>),
}

/// Where each side of [`TileStore::run_jobs`] stands within one call.
#[derive(Debug, Clone, Copy, Default)]
struct Sides {
    /// What the display jobs wait for, if they can do nothing on the owner thread now.
    display: Option<Progress>,
    /// What the queued evaluation waits for, or that it found nothing to do, if so.
    queue: Option<Turn>,
    /// Whether the next turn, both sides having work, is the queued evaluation's.
    queue_next: bool,
}

/// Counts a slice that took `took` in `report`.
pub(super) fn note_slice(report: &mut JobsReport, took: std::time::Duration) {
    report.longest_slice = report.longest_slice.max(took);
    if took > crate::jobs::SLICE_TARGET {
        report.long_slices += 1;
    }
    report.slices += 1;
}

/// What one step through a cone of finest tiles did (see [`TileStore::step_cone`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ConeStep {
    /// It evaluated a round of tiles of the cone.
    Round,
    /// It waits for the corrections it prefetched.
    Reading,
    /// The cone is current (or is to be evaluated whole): these finest tiles' normals are left
    /// to compute.
    Current(Vec<TileKey>),
}

/// What one step of evaluating finest tiles for a job did (see [`TileStore::evaluate_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// It evaluated these many finest tiles, with their normals, and copied them.
    Evaluated(usize),
    /// It evaluated a few tiles of their cone.
    Cone,
    /// It waits for the corrections it prefetched.
    Reading,
}

impl TileStore {
    /// Queues bringing up to date the display tiles under `finest` that writes since made
    /// unknown: into the refresh already queued (a tile made unknown by several steps is in it
    /// once, and built once, from what the store holds when it is), else as a new one. Looks
    /// only when the display book changed since it last did.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if settling the writes fails (never, for tiles of the store's
    /// mesh).
    fn queue_refresh(&mut self, finest: Level) -> Result<(), CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        let changes = self.display.changes();
        if changes == self.refresh_seen {
            return Ok(());
        }
        self.refresh_seen = changes;
        let unknown: Vec<TileKey> = self
            .display
            .unknown_keys()
            .into_iter()
            .filter(|key| !self.has_failed(key))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        if let Some(Job::Refresh(upkeep)) = self
            .jobs
            .iter_mut()
            .find(|job| matches!(job, Job::Refresh(_)))
        {
            for key in unknown {
                if !upkeep.in_flight.contains_key(&key) {
                    upkeep.push(key);
                }
            }
        } else {
            self.jobs.push_back(Job::Refresh(Upkeep::new(
                finest,
                unknown,
                self.cache_caps.copies,
            )));
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
        let missing: Vec<TileKey> = self
            .all_display_keys(finest)
            .into_iter()
            .filter(|key| self.where_held(key).is_none())
            .collect();
        if !missing.is_empty() {
            self.jobs.push_back(Job::Rebuild(Upkeep::new(
                finest,
                missing,
                self.cache_caps.copies,
            )));
        }
    }

    /// Where the store holds `key`'s display tile as known, without reading it: its record in
    /// the book, or (where the book doesn't say) a record in the files that isn't an unknown
    /// one (told by its length). `None` if it doesn't.
    fn where_held(&self, key: &TileKey) -> Option<Finer> {
        match self.display.kept(key) {
            Some(Kept::Known(record)) => Some(Finer::Record(record.clone())),
            Some(Kept::Unknown) => None,
            Some(Kept::Written) | None => {
                let unknown_len = unknown_display_len();
                let known = |loc: &Loc| Some(loc.len()) != unknown_len;
                if let Some(loc) = self.file.locate(key) {
                    return known(&loc).then(|| Finer::File(self.file.reader(), loc));
                }
                let base = self.base.as_ref()?;
                let loc = base.locate(key)?;
                known(&loc).then(|| Finer::File(base.reader(), loc))
            }
        }
    }

    /// Whether the store holds `key`'s display tile as known (see [`TileStore::where_held`]),
    /// without copying or reading anything.
    ///
    /// A record of the files is told known by its length, so a damaged record, or one built
    /// from another finest level, counts as held until a tile reading it decodes it: that
    /// decode fails ([`Outcome::Unreadable`]), and the tile is made unknown and built. A file
    /// whose floor holds a layer from another level has every tile made unknown on open
    /// ([`TileStore::forget_foreign_display`]).
    fn held(&self, key: &TileKey) -> bool {
        match self.display.kept(key) {
            Some(Kept::Known(_)) => true,
            Some(Kept::Unknown) => false,
            Some(Kept::Written) | None => {
                let unknown_len = unknown_display_len();
                self.file
                    .locate(key)
                    .or_else(|| self.base.as_ref().and_then(|b| b.locate(key)))
                    .is_some_and(|loc| Some(loc.len()) != unknown_len)
            }
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

    /// Makes every display tile under `finest` unknown, whatever the files hold: for a file
    /// holding a layer built from another finest level, whose records bound another surface.
    /// A record's level is in its body, so the jobs, which tell known records by their length
    /// alone ([`TileStore::held`]), would otherwise take them as built.
    pub(super) fn forget_foreign_display(&mut self, finest: Level) {
        for key in self.all_display_keys(finest) {
            self.display.set(key, None);
        }
    }

    /// Queues building every display tile under `finest` the store doesn't know, as a rebuild
    /// job: for a file without a display layer.
    pub fn queue_display_rebuild(&mut self, finest: Level) {
        let keys = self.all_display_keys(finest);
        self.jobs.push_back(Job::Rebuild(Upkeep::new(
            finest,
            keys,
            self.cache_caps.copies,
        )));
    }

    /// Runs the queued jobs until `deadline` (and, for a store keeping a display layer, first
    /// adds the display tiles writes left unknown to its refresh). Each job keeps its place; the
    /// next call goes on from there.
    ///
    /// The display tiles are built on the store's workers; the owner thread runs short slices
    /// that collect and record what they finished, hand them more, or evaluate a few finest
    /// tiles, so the deadline is honoured to within one slice ([`JobsReport::longest_slice`]).
    /// A tile the workers built from inputs a write has since changed is built again.
    ///
    /// The queued evaluation ([`TileStore::queue_evaluation`]) runs beside the display jobs,
    /// the two taking turns slice by slice: neither waits behind the other's whole job. A side
    /// with nothing to do on the owner thread (waiting for the workers, or for reads to land)
    /// gives its turn to the other; only when neither has anything to do does the call wait,
    /// until a display tile is finished, a moment for reads, or the deadline. The turn carries
    /// across calls (a call's first slice goes to the side that didn't run last), so neither
    /// side starves however short the calls.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing what a write made stale fails (never, for tiles of the
    /// store's mesh); [`CacheError::Store`] if a display record can't be read; the errors of
    /// [`TileStore::evaluate_many`]. The job stays queued, its tiles still to build or evaluate.
    /// A tile that won't build is no error: it is reported and skipped
    /// ([`TileStore::failed_display`]).
    pub fn run_jobs(&mut self, deadline: Instant) -> Result<JobsReport, CacheError> {
        let mut report = JobsReport::default();
        if let Some(finest) = self.display_finest {
            self.queue_refresh(finest)?;
        }
        if self.jobs.is_empty() && !self.display_decoded.tiles.is_empty() {
            // Nothing to do: the layer is up to date, and the decoded tiles kept for the next
            // refresh go, freed on the workers, until a refresh decodes them again.
            let decoded = self.display_decoded.take();
            self.workers.spawn(move || drop(decoded));
        }
        let mut sides = Sides {
            queue_next: self.evaluation_next,
            ..Sides::default()
        };
        let ran = self.run_sides(deadline, &mut sides, &mut report);
        self.evaluation_next = sides.queue_next;
        ran?;
        report.left = self.jobs.iter().map(Job::left).sum();
        report.queue_left = self.evaluation.len();
        Ok(report)
    }

    /// [`TileStore::run_jobs`]'s slices, taking turns between the display jobs and the queued
    /// evaluation as `sides` says, until `deadline`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::run_jobs`].
    fn run_sides(
        &mut self,
        deadline: Instant,
        sides: &mut Sides,
        report: &mut JobsReport,
    ) -> Result<(), CacheError> {
        while self.now() < deadline {
            let display_ready = !self.jobs.is_empty() && sides.display.is_none();
            let queue_ready = sides.queue.is_none() && self.evaluation.has_work();
            let run_queue = match (display_ready, queue_ready) {
                (true, true) => sides.queue_next,
                (false, true) => true,
                (true, false) => false,
                (false, false) => {
                    if !self.wait_for_sides(deadline, sides) {
                        break;
                    }
                    continue;
                }
            };
            sides.queue_next = !run_queue;
            #[cfg(test)]
            let work = self
                .hooks
                .begin_turn(run_queue, display_ready && queue_ready);
            let began = Instant::now();
            if run_queue {
                let turn = self.evaluation_slice(report);
                note_slice(report, began.elapsed());
                #[cfg(test)]
                self.hooks.end_turn(work);
                sides.queue = match turn? {
                    Turn::Worked => None,
                    waiting @ (Turn::Reading | Turn::Idle) => Some(waiting),
                };
                // The display jobs may have something to collect by now.
                sides.display = None;
                continue;
            }
            let Some(mut job) = self.jobs.pop_front() else {
                continue;
            };
            let progress = self.slice(&mut job, report);
            note_slice(report, began.elapsed());
            #[cfg(test)]
            self.hooks.end_turn(work);
            match progress {
                Ok(Progress::Done) => report.finished += 1,
                Ok(Progress::More) => self.jobs.push_front(job),
                Ok(wait @ (Progress::Waiting | Progress::Reading)) => {
                    self.jobs.push_front(job);
                    sides.display = Some(wait);
                }
                Err(e) => {
                    self.jobs.push_front(job);
                    return Err(e);
                }
            }
            // The queued evaluation's reads may have landed by now, and a slice that found
            // nothing to do may find something after the display jobs' slice.
            sides.queue = None;
        }
        Ok(())
    }

    /// With neither side of [`TileStore::run_jobs`] able to do anything on the owner thread,
    /// waits for one: until a worker finishes a display tile (when the display jobs wait on
    /// the workers), a moment (when a side waits on reads, which land in the store, not on the
    /// channel), or the deadline. Returns whether to go on: a side was waiting, and the deadline
    /// leaves time after the wait.
    fn wait_for_sides(&mut self, deadline: Instant, sides: &mut Sides) -> bool {
        let reading = sides.queue == Some(Turn::Reading);
        let display = sides.display;
        if display.is_none() && !reading {
            // Nothing to do, nothing to wait for.
            return false;
        }
        let now = self.now();
        let until = deadline
            .checked_sub(crate::jobs::WAKE_MARGIN)
            .unwrap_or(deadline);
        let until = if reading || display == Some(Progress::Reading) {
            until.min(now + crate::jobs::READ_WAIT)
        } else {
            until
        };
        // The wait itself is in real time, whatever clock the deadline is on.
        let left = until.saturating_duration_since(now);
        match self.jobs.front_mut() {
            Some(job) if display.is_some() => job.upkeep().wait_until(Instant::now() + left),
            // A sleep of nothing still gives the thread up, for as long as the scheduler likes.
            _ if left.is_zero() => {}
            _ => std::thread::sleep(left),
        }
        #[cfg(test)]
        self.hooks.wait_until(until);
        sides.display = None;
        if reading {
            sides.queue = None;
        }
        deadline.saturating_duration_since(self.now()) > crate::jobs::WAKE_MARGIN
    }

    /// The time [`TileStore::run_jobs`] reads its deadline against: now, or, in a test that
    /// sets one, the test's clock (in the store's test hooks), which moves only
    /// by whole slices and waits.
    #[cfg_attr(
        not(test),
        expect(clippy::unused_self, reason = "only the tests' clock reads the store")
    )]
    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = self.hooks.clock_now() {
            return now;
        }
        Instant::now()
    }

    /// Queues building the display tiles of `level` alone (a level below `finest`) the store
    /// doesn't know: for timing a rebuild level by level, finest first.
    #[cfg(feature = "bench")]
    pub fn queue_display_level(&mut self, finest: Level, level: Level) {
        let keys: Vec<TileKey> = self
            .all_display_keys(finest)
            .into_iter()
            .filter(|key| key.level() == level)
            .collect();
        self.jobs.push_back(Job::Rebuild(Upkeep::new(
            finest,
            keys,
            self.cache_caps.copies,
        )));
    }

    /// Drops every queued job, where it is: what they had built stays, and so do the tiles the
    /// queued evaluation finished, for [`TileStore::take_evaluated`]. Tiles the workers are
    /// building are dropped when they finish. The next [`TileStore::run_jobs`] looks for the
    /// display tiles writes left unknown again, so a refresh dropped here is queued anew.
    pub fn clear_jobs(&mut self) {
        self.jobs.clear();
        self.evaluation.clear();
        self.refresh_seen = u64::MAX;
    }

    /// Whether any job has work left: display tiles still to build (or being built), or tiles
    /// queued for evaluation with room to finish them (or finished tiles to bring up to date).
    #[must_use]
    pub fn has_jobs(&self) -> bool {
        !self.jobs.is_empty() || self.evaluation.has_work()
    }

    /// One slice of `job`: records what the workers finished, hands them the next tiles whose
    /// inputs the store holds (up to two a worker in flight), and, if it could hand them none,
    /// evaluates a few of the finest tiles those read.
    ///
    /// # Errors
    ///
    /// As [`TileStore::run_jobs`]; the tiles stay to build.
    fn slice(&mut self, job: &mut Job, report: &mut JobsReport) -> Result<Progress, CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        let upkeep = job.upkeep();
        // The finest copies a change outdated (any of them, C2) are freed on the workers: tens
        // of megabytes, at worst. Decoded display tiles stay: a record is only replaced by a
        // build, which its generation tells.
        if let Some(copies) = upkeep.copies.since(self.edits)
            && !copies.is_empty()
        {
            let mut rest = Vec::new();
            for (_, tile) in copies {
                if upkeep.spare_full() {
                    rest.push(tile);
                } else {
                    upkeep.recycle(tile);
                }
            }
            if !rest.is_empty() {
                self.workers.spawn(move || drop(rest));
            }
        }
        let collected = self.collect(upkeep, report)?;
        let workers = self.workers.current_num_threads().max(1);
        // A quarter of the workers (one, at least) are left free of display tiles, for the
        // evaluation a slice asks of the pool: a round finding none idle would wait for a
        // display tile to finish, and the owner thread with it.
        let reserve = (workers / 4).max(1);
        let room = workers
            .saturating_sub(reserve)
            .max(1)
            .saturating_sub(upkeep.busy());
        let (handed, evaluate) = self.dispatch(upkeep, room, report);
        let mut reading = false;
        let evaluated = handed == 0 && !evaluate.is_empty();
        if evaluated {
            // The evaluation runs on the workers not building display tiles (one, at least).
            let free = workers.saturating_sub(upkeep.busy()).max(1);
            let round = upkeep.steps.round_size(free);
            let evaluate: BTreeSet<TileKey> = evaluate.into_iter().take(round).collect();
            match self.evaluate_for(upkeep, &evaluate, round)? {
                Step::Evaluated(n) => report.evaluated += n,
                Step::Cone => {}
                Step::Reading => reading = true,
            }
        }
        Ok(if upkeep.is_empty() {
            Progress::Done
        } else if reading && collected == 0 {
            Progress::Reading
        } else if collected == 0 && handed == 0 && !evaluated && upkeep.busy() > 0 {
            Progress::Waiting
        } else {
            Progress::More
        })
    }

    /// Records the tiles the workers finished since the last slice, as one epoch, and keeps them
    /// (and the finer tiles they decoded) for the level above; puts back the ones built from
    /// inputs a change has since outdated, and those whose inputs turned out short; marks failed
    /// those that won't build. Returns how many finished.
    ///
    /// # Errors
    ///
    /// The first [`CacheError::Store`] a worker met, or the errors of
    /// [`TileStore::record_built`]: the tiles are put back; or that of
    /// [`TileStore::write_through_display`], with the tiles recorded (a worker's comes first).
    fn collect(
        &mut self,
        upkeep: &mut Upkeep,
        report: &mut JobsReport,
    ) -> Result<usize, CacheError> {
        let mut finished = std::mem::take(&mut upkeep.received);
        finished.extend(upkeep.receiver.try_iter());
        #[cfg(test)]
        {
            // The tests' hold: what the workers finished waits, as if they were still at it,
            // until it is let go.
            if self.hooks.hold_finished {
                self.hooks.held_finished.append(&mut finished);
                return Ok(0);
            }
            finished.append(&mut self.hooks.held_finished);
        }
        let count = finished.len();
        let mut built = Vec::new();
        let mut error = None;
        for done in finished {
            let ours = if done.decode {
                upkeep.decoding.remove(&done.key)
            } else {
                upkeep.in_flight.remove(&done.key).is_some()
            };
            if !ours {
                continue;
            }
            if done.decode && done.record != self.display_generation(&done.key) {
                // Decoded from a record since replaced: decoded again when a tile reads it.
                continue;
            }
            if !done.decode && done.as_of != self.edits {
                // Built from inputs a change has outdated since: built again.
                upkeep.push(done.key);
                continue;
            }
            match done.outcome {
                Outcome::Built(tile) => built.push(*tile),
                Outcome::Decoded(tile) => {
                    self.display_decoded
                        .insert(*tile.tile.key(), (done.record, Arc::new(*tile)));
                }
                Outcome::Unreadable => {
                    // A record the files hold that doesn't read: unknown, so it is rebuilt.
                    self.display.set(done.key, None);
                    upkeep.push(done.key);
                }
                Outcome::Failed(why) => {
                    // This tile won't build from what the store holds: unknown, and skipped, so
                    // one tile never holds up the rest.
                    self.fail_display(done.key, why);
                    report.failed += 1;
                }
                Outcome::Error(e) => {
                    upkeep.push(done.key);
                    error.get_or_insert(e);
                }
            }
        }
        if !built.is_empty() {
            if let Err(e) = self.record_built(&built) {
                for tile in &built {
                    upkeep.push(*tile.tile.key());
                }
                return Err(e);
            }
            report.built += built.len();
            // A slice is an operation: its display tiles are one epoch.
            self.close_epoch()?;
            for tile in built {
                let key = *tile.tile.key();
                let record = self.display_generation(&key);
                self.display_decoded.insert(key, (record, Arc::new(tile)));
            }
            // Recorded whatever comes of it: a failed write leaves them waiting in memory.
            if let Err(e) = self.write_through_display() {
                error.get_or_insert(e);
            }
        }
        error.map_or(Ok(count), Err)
    }

    /// Hands the workers up to `room` pending tiles whose inputs the store holds; returns how
    /// many it handed, and the finest tiles to evaluate for those it couldn't (all one tile
    /// lacks, and others' only while they stay few). Looks at a bounded number of tiles.
    fn dispatch(
        &mut self,
        upkeep: &mut Upkeep,
        room: usize,
        report: &mut JobsReport,
    ) -> (usize, BTreeSet<TileKey>) {
        let finest = upkeep.finest;
        let mut handed = 0;
        let mut evaluate: BTreeSet<TileKey> = BTreeSet::new();
        let mut later: Vec<TileKey> = Vec::new();
        let mut looked = 0;
        let mut allowance = COPY_BATCH;
        let began = Instant::now();
        while handed < room && looked < LOOK_AHEAD * room.max(1) && began.elapsed() < HAND_OUT_TIME
        {
            let Some(key) = upkeep.next() else {
                break;
            };
            looked += 1;
            if upkeep.in_flight.contains_key(&key) || self.has_failed(&key) || self.held(&key) {
                // Built, being built, or failed.
                continue;
            }
            let Some(reads) = tent_reads(&self.mesh, &key) else {
                continue;
            };
            match self.readiness(&key, &reads, upkeep) {
                Readiness::Ready if key.level().child() == Some(finest) => {
                    let Some(inputs) = self.finest_inputs(&reads, upkeep, &mut allowance) else {
                        later.push(key);
                        if allowance == 0 {
                            // This slice's copies are made: the tiles after it wait too.
                            break;
                        }
                        continue;
                    };
                    self.hand_out(key, inputs, upkeep);
                    handed += 1;
                }
                Readiness::Ready => {
                    // The finer tiles it reads, decoded each once for every tile reading them.
                    let missing = match self.decoded_inputs(&reads) {
                        Ok(inputs) => {
                            self.hand_out(key, inputs, upkeep);
                            handed += 1;
                            continue;
                        }
                        Err(missing) => missing,
                    };
                    later.push(key);
                    let mut asked = false;
                    for tile in missing {
                        if handed >= room || upkeep.decoding.contains(&tile) {
                            continue;
                        }
                        if let Some(place) = self.where_held(&tile) {
                            self.hand_out(tile, TaskInputs::Decode(place), upkeep);
                            handed += 1;
                            asked = true;
                        }
                    }
                    if !asked && handed < room {
                        // It waits for tiles being decoded, and so, coming in order, do most
                        // tiles after it.
                        break;
                    }
                }
                Readiness::Evaluate(missing) => {
                    later.push(key);
                    let joining: usize = missing.iter().filter(|t| !evaluate.contains(t)).count();
                    let fits = evaluate.len() + joining <= crate::jobs::ROUND_MAX;
                    if evaluate.is_empty() || fits {
                        evaluate.extend(missing);
                    } else {
                        // Enough to evaluate: the tiles after it wait for the next slices.
                        break;
                    }
                }
                Readiness::Needs(needs) => {
                    // A tile reading one that wouldn't build can't build either, until a change.
                    if let Some(why) = needs.iter().find_map(|dep| self.failure(dep)) {
                        self.fail_display(key, why);
                        report.failed += 1;
                        continue;
                    }
                    later.push(key);
                    let waiting = needs.iter().all(|need| upkeep.in_flight.contains_key(need));
                    for need in needs {
                        if !upkeep.in_flight.contains_key(&need) {
                            later.push(need);
                        }
                    }
                    if waiting {
                        // It waits for tiles the workers are building, and so, the tiles coming
                        // finest first, do those after it: none is looked at again until they
                        // are back.
                        break;
                    }
                }
            }
        }
        for key in later {
            upkeep.push(key);
        }
        let evaluate = evaluate.into_iter().take(crate::jobs::ROUND_MAX).collect();
        (handed, evaluate)
    }

    /// The decoded display tiles of `reads` (display tiles one level finer), for a worker
    /// building the tile above them.
    ///
    /// # Errors
    ///
    /// Those of `reads` not decoded as their records now are: to decode first.
    fn decoded_inputs(&self, reads: &BTreeSet<TileKey>) -> Result<TaskInputs, Vec<TileKey>> {
        let current = |t: &TileKey| {
            self.display_decoded
                .tiles
                .get(t)
                .filter(|(record, _)| *record == self.display_generation(t))
        };
        let missing: Vec<TileKey> = reads
            .iter()
            .filter(|t| current(t).is_none())
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(missing);
        }
        Ok(TaskInputs::Display(
            reads
                .iter()
                .filter_map(|t| Some((*t, Arc::clone(&current(t)?.1))))
                .collect(),
        ))
    }

    /// Hands display tile `key` to a worker, to build or decode with `inputs`, as of the store's
    /// count of correction changes now.
    fn hand_out(&self, key: TileKey, inputs: TaskInputs, upkeep: &mut Upkeep) {
        let decode = matches!(inputs, TaskInputs::Decode(_));
        if decode {
            upkeep.decoding.insert(key);
        } else {
            upkeep.in_flight.insert(key, self.edits);
        }
        #[cfg_attr(not(test), expect(unused_mut, reason = "only the tests' hook sets it"))]
        let mut task = Task {
            key,
            as_of: self.edits,
            record: self.display_generation(&key),
            mesh: Arc::clone(&self.mesh),
            finest: upkeep.finest,
            inputs,
            unbuildable: false,
        };
        #[cfg(test)]
        {
            task.unbuildable = !decode && self.hooks.unbuildable_display == Some(key);
        }
        let sender = upkeep.sender.clone();
        self.workers.spawn(move || {
            let outcome = run_task(&task);
            // The job may be gone (cleared): then nobody wants the tile.
            drop(sender.send(Finished {
                key: task.key,
                as_of: task.as_of,
                record: task.record,
                decode,
                outcome,
            }));
        });
    }

    /// Whether everything display tile `key`, which reads `reads` (see [`tent_reads`]), needs is
    /// there (see [`Readiness`]): the finest tiles it reads, cached or copied, one level above
    /// the finest; else the display tiles it reads, known and not being rebuilt.
    fn readiness(&self, key: &TileKey, reads: &BTreeSet<TileKey>, upkeep: &Upkeep) -> Readiness {
        if key.level().child() == Some(upkeep.finest) {
            let missing: Vec<TileKey> = reads
                .iter()
                .filter_map(|t| TileKey::displacement(t.face(), t.level(), t.tx(), t.ty()).ok())
                .filter(|t| {
                    !upkeep.copies.tiles.contains_key(t) && self.cached_normals(t).is_none()
                })
                .collect();
            if missing.is_empty() {
                Readiness::Ready
            } else {
                Readiness::Evaluate(missing)
            }
        } else {
            let needs: Vec<TileKey> = reads
                .iter()
                .filter(|t| upkeep.in_flight.contains_key(t) || !self.held(t))
                .copied()
                .collect();
            if needs.is_empty() {
                Readiness::Ready
            } else {
                Readiness::Needs(needs)
            }
        }
    }

    /// The finest tiles in `reads` (display keys one level finer: the finest), copied for a
    /// worker from `upkeep`'s copies or the store's cache (copying those into `upkeep` once, at
    /// most `allowance` of them, which it counts down); `None` if one isn't there, or the
    /// allowance ran out first (the copies made so far stay).
    fn finest_inputs(
        &self,
        reads: &BTreeSet<TileKey>,
        upkeep: &mut Upkeep,
        allowance: &mut usize,
    ) -> Option<TaskInputs> {
        let mut tiles = BTreeMap::new();
        for t in reads {
            let Ok(key) = TileKey::displacement(t.face(), t.level(), t.tx(), t.ty()) else {
                continue;
            };
            if !upkeep.copies.tiles.contains_key(&key) {
                *allowance = allowance.checked_sub(1)?;
                let positions = self.cached_positions(&key)?;
                let normals = self.cached_normals(&key)?;
                upkeep.copy_finest(key, positions, normals);
            }
            tiles.insert(key, upkeep.copies.tiles.get(&key)?.clone());
        }
        Some(TaskInputs::Finest(tiles))
    }

    /// Evaluates the finest tiles of `evaluate` (with their normals, on the workers) a step at a
    /// time (see [`TileStore::step_cone`]), and copies them for the workers building the display
    /// tiles that read them.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::prefetch`].
    fn evaluate_for(
        &mut self,
        upkeep: &mut Upkeep,
        evaluate: &BTreeSet<TileKey>,
        round: usize,
    ) -> Result<Step, CacheError> {
        let listed = || evaluate.iter().copied().collect();
        let keys = match self.step_cone(&mut upkeep.steps, listed, round)? {
            ConeStep::Round => return Ok(Step::Cone),
            ConeStep::Reading => return Ok(Step::Reading),
            ConeStep::Current(keys) => keys,
        };
        let t0 = Instant::now();
        self.evaluate_many(&keys, Evaluation::Normals)?;
        upkeep.steps.took(t0.elapsed(), keys.len());
        // Copied now, while cached: a later batch may evict them before their tiles are handed
        // out.
        for tile in &keys {
            if self.cached_positions(tile).is_none() {
                self.evaluate(*tile)?;
            }
            if self.cached_normals(tile).is_none() {
                self.normals(*tile)?;
            }
            // Both cached now: the normals are computed with the positions pinned.
            if let (Some(positions), Some(normals)) =
                (self.cached_positions(tile), self.cached_normals(tile))
            {
                upkeep.copy_finest(*tile, positions, normals);
            }
        }
        Ok(Step::Evaluated(keys.len()))
    }

    /// One step of evaluating finest tiles for a job, through the cone `steps` holds.
    ///
    /// The cone is of the finest tiles `keys` gives, listed when `steps` holds none or a
    /// correction change since it was listed outdated it: the positions their normals read,
    /// then the tiles those are predicted from, level by level. A step evaluates at most `round`
    /// tiles of the coarsest level of it that isn't current, so one slice runs one round of
    /// evaluation however cold the cone. The corrections of that level are prefetched first, and
    /// only tiles whose corrections are resident (or that have none) are evaluated, so a slice
    /// never waits on the disk; with none ready and reads in flight, it waits for them
    /// ([`ConeStep::Reading`]). Once the whole cone is current, it hands the finest tiles back
    /// ([`ConeStep::Current`]) and forgets the cone, for the caller to compute their normals. A
    /// step that would evaluate a tile an earlier step of this cone already did (a budget too
    /// small to keep the cone cached between slices) hands them back too, to be evaluated
    /// whole, so the work always ends.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::prefetch`].
    pub(super) fn step_cone(
        &mut self,
        steps: &mut ConeSteps,
        keys: impl FnOnce() -> Vec<TileKey>,
        round: usize,
    ) -> Result<ConeStep, CacheError> {
        // The tiles a cone is being stepped through for stay the ones evaluated until it is
        // done, so a step never starts over.
        let keys: Vec<TileKey> = match &steps.cone {
            Some(cone) if cone.as_of == self.edits => cone.keys.clone(),
            _ => {
                let keys = keys();
                let levels = self.normals_cone(&keys)?;
                steps.cone = Some(Cone {
                    keys: keys.clone(),
                    levels,
                    as_of: self.edits,
                });
                steps.seen.clear();
                keys
            }
        };
        let mut coarsest = None;
        if let Some(Cone { levels: cone, .. }) = &steps.cone {
            for level in cone {
                let stale: Vec<TileKey> = level
                    .iter()
                    .copied()
                    .filter(|key| !self.is_current(key))
                    .collect();
                if !stale.is_empty() {
                    coarsest = Some(stale);
                    break;
                }
            }
        }
        if let Some(stale) = coarsest
            && !stale.iter().any(|key| steps.seen.contains(key))
        {
            self.prefetch(
                stale
                    .get(..stale.len().min(PREFETCH_AHEAD))
                    .unwrap_or_default(),
            )?;
            self.poll()?;
            let mut ready: Vec<TileKey> = stale
                .iter()
                .filter(|key| self.corrections.find(key).is_some() || self.locate(key).is_none())
                .take(round.max(1))
                .copied()
                .collect();
            if ready.is_empty() {
                if self.prefetching() > 0 {
                    return Ok(ConeStep::Reading);
                }
                // The reads landed and weren't kept (no room for them): read one here.
                ready.extend(stale.first());
            }
            let t0 = Instant::now();
            self.evaluate_many(&ready, Evaluation::Positions)?;
            steps.took(t0.elapsed(), ready.len());
            steps.seen.extend(ready);
            return Ok(ConeStep::Round);
        }
        steps.reset();
        Ok(ConeStep::Current(keys))
    }

    /// Why display tile `key` wouldn't build, if it failed since the last correction change.
    fn failure(&self, key: &TileKey) -> Option<DisplayError> {
        self.display_failed
            .get(key)
            .filter(|(at, _)| *at == self.edits)
            .map(|&(_, why)| why)
    }

    /// Whether display tile `key` failed since the last correction change: it is skipped.
    fn has_failed(&self, key: &TileKey) -> bool {
        self.failure(key).is_some()
    }

    /// Records that display tile `key` wouldn't build, for `why`: it is unknown, and every job
    /// skips it until a correction changes.
    fn fail_display(&mut self, key: TileKey, why: DisplayError) {
        self.display.set(key, None);
        self.display_failed.insert(key, (self.edits, why));
    }

    /// The display tiles that wouldn't build since the last correction change, and why: each is
    /// unknown (drawn finer), and is tried again once a change lands.
    #[must_use]
    pub fn failed_display(&self) -> Vec<(TileKey, DisplayError)> {
        self.display_failed
            .iter()
            .filter(|(_, (at, _))| *at == self.edits)
            .map(|(&key, &(_, why))| (key, why))
            .collect()
    }
}
