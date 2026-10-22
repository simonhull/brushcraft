//! The tile store: a project held out of core.
//!
//! [`TileStore`] owns the base mesh, the working [`ProjectFile`], two budgeted pools (correction
//! tiles and evaluated tiles), the change feed and an op generator. Correction tiles are loaded
//! from the file on demand (an absent tile is all zero) and edited in their slots; evaluated
//! tiles are computed on demand from their parent tiles, recursively up to the base mesh, and
//! cached. Edits drop exactly the evaluations that followed from them (see [`crate::deps`]).
//!
//! Everything here belongs to one thread: the one calling the store. Pins are held by
//! [`PinGuard`]s, which may be dropped anywhere on that thread; the store applies the unpins at
//! its next call. The one exception is prefetch: [`TileStore::prefetch`] hands record locations
//! to worker threads (see [`crate::prefetch`]), which read and decode on their own and send the
//! tiles back for [`TileStore::poll`] to install.
//!
//! # Budget
//!
//! The two pools share one byte budget, granted in whole chunks. When one pool needs a slot and
//! has none to reuse (all of its slots are pinned) while the budget is spent, the store takes
//! whole chunks back from the pools: evaluated chunks first, since they are only a cache and cheap
//! to recompute, then correction chunks, whose dirty slots are written back to the working file
//! first. A chunk holding a pinned slot can't be given back, so the request fails only when no
//! chunk without a pin is left.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::io;
use std::rc::Rc;
use std::sync::Arc;

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionView, EvalError, FaceId, LayerId, Level,
    PositionSource, Region, TileKey, Vec3, normal_tiles, normals_into, parent_tiles,
    predict_region,
};
use thiserror::Error;

use crate::codec::CodecError;
use crate::deps::DepGraph;
use crate::feed::{ChangeFeed, ConsumerId};
use crate::file::{ProjectFile, ReadScratch, StoreError};
use crate::op::{OpId, RESERVED_SITE};
use crate::pool::{Budget, Freshness, Invalidated, Pool, PoolError, SlotId};
use crate::prefetch::{Fetched, Job, Prefetcher};
use crate::slot::{CorrectionSlot, EvaluatedSlot, SIDE};

/// How a [`TileStore`] is set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreConfig {
    /// Bytes the two pools may hold together. The pools never exceed it.
    pub budget_bytes: usize,
    /// Bytes per pool chunk: [`StoreConfig::DEFAULT_CHUNK_BYTES`] in the app, smaller in tests.
    pub chunk_bytes: usize,
    /// The site (machine or session) the store's op ids carry; any but [`RESERVED_SITE`].
    pub site: u16,
}

impl StoreConfig {
    /// The default chunk: 64 MiB, large enough that the allocator maps each chunk on its own
    /// and gives it back to the OS when it is dropped (see the pool's notes).
    pub const DEFAULT_CHUNK_BYTES: usize = 64 << 20;
}

/// Why a [`TileStore`] operation failed.
#[derive(Debug, Error)]
pub enum CacheError {
    /// Reading or writing the working file failed.
    #[error("project file operation failed")]
    Store(#[from] StoreError),
    /// A pool operation failed: [`PoolError::Full`] when every slot is pinned and no chunk can
    /// be taken back, or a dirty slot's write-back failed.
    #[error("tile pool operation failed")]
    Pool(#[from] PoolError),
    /// Evaluation failed.
    #[error("evaluation failed")]
    Eval(#[from] EvalError),
    /// A correction was rejected (not finite, level 0, unknown face, outside the face), or a
    /// slot didn't hold a valid tile.
    #[error("correction rejected")]
    Correction(#[from] CorrectionError),
    /// The key isn't a base-layer displacement tile above level 0 of a face of this mesh.
    #[error("{0:?} is not a base-layer displacement tile of this mesh above level 0")]
    NotATile(TileKey),
    /// The configured site is [`RESERVED_SITE`].
    #[error("site {RESERVED_SITE} is reserved")]
    ReservedSite,
    /// Every op id of this site has been handed out.
    #[error("the op ids of this site are used up")]
    OpsExhausted,
    /// A pin request can't be held within the budget, even with every unpinned slot freed.
    /// Nothing was evicted or pinned.
    #[error("pinning {tiles} tiles needs {needed} bytes of pool chunks; the budget is {budget}")]
    OverBudget {
        /// Distinct tiles requested.
        tiles: usize,
        /// Bytes of chunks the request and the pins already held would take.
        needed: usize,
        /// The budget.
        budget: usize,
    },
}

/// Pin sets whose guards have been dropped, waiting for the store to unpin them.
///
/// Shared between the store and its guards. It is borrowed only for a push (a guard's drop) or
/// a take (the store's next call), never across anything that could drop a guard, so the
/// borrows never overlap.
#[derive(Debug, Default)]
struct PinTable {
    /// Ids of released pin sets.
    released: Vec<u64>,
}

/// Keeps a set of tiles resident: their evaluated positions and their correction tiles stay in
/// the pools, in the same slots, until the guard is dropped.
///
/// An edit of a pinned tile (or of anything it was predicted from) doesn't move it: its slot is
/// recomputed in place on its next [`TileStore::evaluate`]. The store applies the unpin at its
/// next call after the drop.
#[derive(Debug)]
#[must_use = "dropping the guard unpins the tiles"]
pub struct PinGuard {
    /// The pin set in the store.
    id: u64,
    /// The pinned tiles, in key order.
    keys: Vec<TileKey>,
    /// Where the drop is reported.
    table: Rc<RefCell<PinTable>>,
}

impl PinGuard {
    /// The pinned tiles, in key order, each once.
    #[must_use]
    pub fn keys(&self) -> &[TileKey] {
        &self.keys
    }
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        self.table.borrow_mut().released.push(self.id);
    }
}

/// The slots one pinned tile holds: its evaluated tile and its correction tile.
type Pinned = (SlotId<EvaluatedSlot>, SlotId<CorrectionSlot>);

/// Evaluated tiles an evaluation pinned, with their keys: what its [`Resident`] source reads.
type Held = Vec<(TileKey, SlotId<EvaluatedSlot>)>;

/// What the store computed and read, for tests that count re-evaluations and file reads.
#[cfg(test)]
#[derive(Debug, Default)]
struct EvalLog {
    /// Tiles whose positions were computed, in order.
    positions: Vec<TileKey>,
    /// Tiles whose normals were computed, in order.
    normals: Vec<TileKey>,
    /// Correction tiles the owner thread looked up in the working file, in order (prefetch
    /// workers' reads aren't counted).
    file_reads: Vec<TileKey>,
    /// Prefetched correction tiles installed.
    prefetched: usize,
}

/// A project held out of core: the base mesh, the working file, budgeted pools of correction and
/// evaluated tiles, the change feed and an op generator.
///
/// Tiles are named by base-layer displacement [`TileKey`]s of level 1 or more; an evaluated tile
/// is named by the displacement tile at the same position. Level 0 is the base mesh, never a
/// tile.
pub struct TileStore {
    /// The prefetch workers. Declared first, so they are stopped and joined (when the store is
    /// dropped) before anything else goes, the working file included.
    prefetch: Prefetcher,
    /// The base mesh: level 0 and the topology.
    mesh: Arc<BaseMesh>,
    /// The working file: correction tiles are read from it and written back to it.
    file: ProjectFile,
    /// The bytes both pools may hold.
    budget: Budget,
    /// Correction tiles, loaded from the file or zero, and edited in place.
    corrections: Pool<CorrectionSlot>,
    /// Evaluated positions and normals, a cache.
    evaluated: Pool<EvaluatedSlot>,
    /// What each cached evaluated tile was computed from.
    deps: DepGraph,
    /// The op of the latest edit of each dirty correction slot: the op its record is written
    /// with, by write-back or [`TileStore::save`].
    unsaved: HashMap<TileKey, OpId>,
    /// Keys drained from the feed that [`TileStore::save`] hasn't written yet (left by a save
    /// that failed part-way, so the next one finishes them).
    to_save: VecDeque<TileKey>,
    /// The change feed: every key an edit changed, with its op.
    feed: ChangeFeed,
    /// Save's place in the feed.
    save_consumer: ConsumerId,
    /// The next op [`TileStore::next_op`] hands out; `None` once the site's ids are used up.
    next_op: Option<OpId>,
    /// Shared with the [`PinGuard`]s: the pin sets they have released.
    pins: Rc<RefCell<PinTable>>,
    /// The slots of each live pin set.
    pin_sets: HashMap<u64, Vec<Pinned>>,
    /// The id the next pin set gets.
    next_pin_set: u64,
    /// One tile of positions or normals: evaluation writes here, then copies into a slot.
    scratch: Vec<Vec3>,
    /// Buffers for reading correction tiles.
    read_scratch: ReadScratch,
    /// What was computed, for the tests.
    #[cfg(test)]
    log: EvalLog,
}

impl std::fmt::Debug for TileStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileStore")
            .field("file", &self.file)
            .field("budget", &self.budget)
            .field("corrections", &self.corrections)
            .field("evaluated", &self.evaluated)
            .field("unsaved", &self.unsaved.len())
            .field("pin_sets", &self.pin_sets.len())
            .field("prefetching", &self.prefetch.pending())
            .finish_non_exhaustive()
    }
}

/// Positions of the tiles an evaluation pinned, and the base mesh at level 0.
///
/// Each vertex is read from [`TileKey::containing`]'s tile, as [`PositionSource`] requires, so
/// differing copies on tile borders can't open a seam.
struct Resident<'a> {
    /// Level 0.
    mesh: &'a BaseMesh,
    /// The evaluated pool.
    pool: &'a Pool<EvaluatedSlot>,
    /// The pinned tiles; a dozen at most, so a linear search beats hashing.
    tiles: &'a [(TileKey, SlotId<EvaluatedSlot>)],
}

impl PositionSource for Resident<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        if level == Level::ZERO {
            return self.mesh.position(face, level, li, lj);
        }
        let &(_, id) = self.tiles.iter().find(|(k, _)| *k == key)?;
        let index = usize::try_from(lj * key.side() + li).ok()?;
        self.pool.get(id)?.positions.get(index).copied()
    }
}

/// Vertices in `key`'s tile: `side²`.
fn vertex_count(key: &TileKey) -> usize {
    let side = key.side() as usize;
    side * side
}

/// The first `n` values of `values`.
///
/// # Errors
///
/// [`EvalError::BufferSize`] if there are fewer (no slot or scratch buffer is smaller than the
/// largest tile, so this doesn't happen).
fn prefix(values: &[Vec3], n: usize) -> Result<&[Vec3], EvalError> {
    values.get(..n).ok_or(EvalError::BufferSize {
        expected: n,
        got: values.len(),
    })
}

/// [`prefix`], mutably.
///
/// # Errors
///
/// As [`prefix`].
fn prefix_mut(values: &mut [Vec3], n: usize) -> Result<&mut [Vec3], EvalError> {
    let got = values.len();
    values
        .get_mut(..n)
        .ok_or(EvalError::BufferSize { expected: n, got })
}

/// Records `next` in `outcome` unless `outcome` already holds an error: for cleanup loops that
/// must finish every step and report the first failure.
fn keep_first<E: Into<CacheError>>(outcome: &mut Result<(), CacheError>, next: Result<(), E>) {
    if outcome.is_ok() {
        *outcome = next.map_err(Into::into);
    }
}

/// Appends a dirty correction slot's tile to the working file, with the op of its latest edit
/// (no save point), and forgets the op: the slot is clean once this returns.
///
/// # Errors
///
/// [`StoreError::Io`] if the slot has no recorded op (every slot marked dirty gets one first,
/// so this doesn't happen), [`StoreError::Codec`] if the slot doesn't hold a valid tile, or the
/// file's write errors.
fn write_slot(
    file: &mut ProjectFile,
    unsaved: &mut HashMap<TileKey, OpId>,
    key: TileKey,
    slot: &CorrectionSlot,
) -> Result<(), StoreError> {
    let op = unsaved
        .get(&key)
        .copied()
        .ok_or_else(|| io::Error::other("a dirty correction tile has no op"))?;
    let tile = slot.to_tile().map_err(CodecError::from)?;
    file.write_displacement(key, op, &tile)?;
    unsaved.remove(&key);
    Ok(())
}

impl TileStore {
    /// A store over `file` (the working file) for `mesh`. The op generator starts after the
    /// file's head op if that op is of `config.site`, else at sequence 0 of the site.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::OpsExhausted`] if the file's head is the
    /// site's last op id; [`CacheError::Store`] if the read buffers can't be made.
    pub fn open(
        mesh: Arc<BaseMesh>,
        file: ProjectFile,
        config: StoreConfig,
    ) -> Result<Self, CacheError> {
        if config.site == RESERVED_SITE {
            return Err(CacheError::ReservedSite);
        }
        let next_op = match file.head() {
            Some(head) if head.site() == config.site => {
                Some(head.next().ok_or(CacheError::OpsExhausted)?)
            }
            _ => OpId::new(config.site, 0),
        };
        let mut feed = ChangeFeed::new();
        let save_consumer = feed.subscribe();
        Ok(Self {
            prefetch: Prefetcher::new(),
            mesh,
            file,
            budget: Budget::new(config.budget_bytes),
            corrections: Pool::new(config.chunk_bytes),
            evaluated: Pool::with_eviction_log(config.chunk_bytes),
            deps: DepGraph::new(),
            unsaved: HashMap::new(),
            to_save: VecDeque::new(),
            feed,
            save_consumer,
            next_op,
            pins: Rc::default(),
            pin_sets: HashMap::new(),
            next_pin_set: 0,
            scratch: vec![Vec3::ZERO; SIDE * SIDE],
            read_scratch: ReadScratch::new()?,
            #[cfg(test)]
            log: EvalLog::default(),
        })
    }

    /// A new op id for an edit: the site's next sequence number.
    ///
    /// # Errors
    ///
    /// [`CacheError::OpsExhausted`] past the last sequence number.
    pub fn next_op(&mut self) -> Result<OpId, CacheError> {
        let op = self.next_op.ok_or(CacheError::OpsExhausted)?;
        self.next_op = op.next();
        Ok(op)
    }

    /// Bytes the pools hold: whole chunks, never more than the budget.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.budget.used()
    }

    /// `key`'s correction tile: resident, else loaded from the working file, else all zero (an
    /// absent tile is zero).
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Store`] if the record can't be read;
    /// [`CacheError::Pool`] if no slot can be had.
    pub fn correction(&mut self, key: TileKey) -> Result<CorrectionView<'_>, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.load_correction(key)?;
        Ok(self.corrections.get(id).ok_or(PoolError::Stale)?.view()?)
    }

    /// Sets the correction at face-grid vertex `(i, j)` of `face` at `level` to `c`, as edit
    /// `op`: in every copy of the vertex (one per tile holding it, in every face sharing it).
    /// Each copy's slot becomes dirty and its key goes to the change feed with `op`; every cached
    /// evaluation that followed from those tiles is dropped (pinned ones are recomputed in place
    /// on their next [`TileStore::evaluate`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`],
    /// [`CorrectionError::LevelZero`], [`CorrectionError::UnknownFace`] or
    /// [`CorrectionError::OutOfTile`], before anything is written; [`CacheError::Store`] or
    /// [`CacheError::Pool`] if a copy's tile can't be loaded, also before anything is written.
    pub fn set_correction(
        &mut self,
        face: FaceId,
        level: Level,
        i: u32,
        j: u32,
        c: Vec3,
        op: OpId,
    ) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.poll()?;
        if !c.is_finite() {
            return Err(CorrectionError::NotFinite.into());
        }
        if level == Level::ZERO {
            return Err(CorrectionError::LevelZero.into());
        }
        let topology = self.mesh.topology();
        if !topology.contains(face) {
            return Err(CorrectionError::UnknownFace(face).into());
        }
        let copies = topology.copies(face, level, i, j);
        if copies.is_empty() {
            let side = level.face_quads() + 1;
            return Err(CorrectionError::OutOfTile { i, j, side }.into());
        }
        // Every copy is loaded and pinned before any is written, so a failed load leaves the
        // twins agreeing.
        let mut held = Vec::with_capacity(copies.len());
        let loaded = copies.iter().try_for_each(|copy| {
            let id = self.load_correction(copy.key(level))?;
            self.corrections.pin(id)?;
            held.push(id);
            Ok::<_, CacheError>(())
        });
        let written = loaded.and_then(|()| {
            for (copy, &id) in copies.iter().zip(&held) {
                let key = copy.key(level);
                let slot = self.corrections.get_mut(id).ok_or(PoolError::Stale)?;
                slot.view_mut()?.set(copy.i, copy.j, c)?;
                self.unsaved.insert(key, op);
                self.corrections.set_dirty(id, true)?;
                self.feed.push(key, op);
            }
            Ok(())
        });
        let released = held.iter().try_for_each(|&id| self.corrections.unpin(id));
        written?;
        released?;
        self.invalidate(copies.iter().map(|copy| copy.key(level)));
        Ok(())
    }

    /// `key`'s evaluated positions, row-major `side²`: cached, else computed from its parent
    /// tiles (each evaluated the same way, up to the base mesh) and its corrections.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Pool`] with [`PoolError::Full`] if the slots
    /// pinned at once (pins, and the parents of each level being evaluated) don't fit the
    /// budget; [`CacheError::Store`] if a correction tile can't be read; [`CacheError::Eval`]
    /// (not for a valid mesh).
    pub fn evaluate(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.ensure_positions(key)?;
        let slot = self.evaluated.get(id).ok_or(PoolError::Stale)?;
        Ok(prefix(&slot.positions, vertex_count(&key))?)
    }

    /// `key`'s unit normals, row-major `side²`: cached, else computed from the evaluated
    /// positions of its own level's [`normal_tiles`] (the tile, its apron and the rings at its
    /// corners).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    pub fn normals(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.ensure_normals(key)?;
        let slot = self.evaluated.get(id).ok_or(PoolError::Stale)?;
        Ok(prefix(&slot.normals, vertex_count(&key))?)
    }

    /// Evaluates every tile of `keys` and pins it, with its correction tile, until the guard is
    /// dropped.
    ///
    /// The request is first counted against the budget: the chunks needed to hold its tiles
    /// besides every slot pinned already. If that is more than the budget, nothing is evicted or
    /// pinned. Evaluation can still run out of room (it pins parent tiles while it works); then
    /// the pins this call took are released and the pins held before are untouched.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OverBudget`]; as [`TileStore::evaluate`].
    pub fn pin(&mut self, keys: &[TileKey]) -> Result<PinGuard, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        for key in &keys {
            self.check_key(key)?;
        }
        self.check_pin_budget(&keys)?;
        let mut set = Vec::with_capacity(keys.len());
        let pinned = keys.iter().try_for_each(|&key| {
            set.push(self.pin_one(key)?);
            Ok::<_, CacheError>(())
        });
        if let Err(e) = pinned {
            self.unpin_set(&set)?;
            return Err(e);
        }
        let id = self.next_pin_set;
        self.next_pin_set += 1;
        self.pin_sets.insert(id, set);
        Ok(PinGuard {
            id,
            keys: keys.into_iter().collect(),
            table: Rc::clone(&self.pins),
        })
    }

    /// Writes every correction tile edited since the last save to the working file and records
    /// a save point. Returns its offset.
    ///
    /// The save consumer's keys are drained from the change feed; each one whose slot is still
    /// dirty is written (a tile written back on eviction is in the file already). If a write
    /// fails, the keys not written yet are kept for the next save.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`]; [`CacheError::Pool`] (not after a successful edit).
    pub fn save(&mut self) -> Result<u64, CacheError> {
        self.sync_pins()?;
        self.to_save.extend(
            self.feed
                .drain(self.save_consumer)
                .into_iter()
                .map(|(key, _)| key),
        );
        while let Some(&key) = self.to_save.front() {
            if let Some(id) = self.corrections.find(&key)
                && self.corrections.is_dirty(id) == Some(true)
            {
                let slot = self.corrections.get(id).ok_or(PoolError::Stale)?;
                write_slot(&mut self.file, &mut self.unsaved, key, slot)?;
                self.corrections.set_dirty(id, false)?;
            }
            self.to_save.pop_front();
        }
        Ok(self.file.save()?)
    }

    /// Frees every unpinned slot of both pools, writing dirty correction tiles back first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] with [`PoolError::WriteBack`]: that tile and those not reached yet
    /// stay.
    pub fn evict_all(&mut self) -> Result<(), CacheError> {
        self.sync_pins()?;
        let evaluated = self.evaluated.evict_all(|_, _| Ok(()));
        self.note_evictions();
        evaluated?;
        let (file, unsaved) = (&mut self.file, &mut self.unsaved);
        self.corrections
            .evict_all(|key, slot| write_slot(file, unsaved, key, slot))?;
        Ok(())
    }

    /// Gives chunks back until the pools hold at most `bytes` or only chunks with pinned slots
    /// are left: evaluated chunks first, then correction chunks (dirty tiles written back).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evict_all`].
    pub fn shrink_to(&mut self, bytes: usize) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.give_back(bytes)?;
        Ok(())
    }

    /// Asks the prefetch workers to read the correction tiles of `keys` from the working file in
    /// the background, ahead of the store's use of them; returns how many reads were queued.
    ///
    /// Tiles already resident, absent from the file (all zero) or already on their way are
    /// skipped. This call's tiles are read before those of earlier calls still waiting, in the
    /// order given. They are installed by [`TileStore::poll`], which every method that reads
    /// corrections calls first. The workers start on the first call that queues anything.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`], before anything is queued; [`CacheError::Store`] if the
    /// workers can't be started; [`CacheError::Pool`] as for [`TileStore::evaluate`] (a pin
    /// guard's release).
    pub fn prefetch(&mut self, keys: &[TileKey]) -> Result<usize, CacheError> {
        self.sync_pins()?;
        for key in keys {
            self.check_key(key)?;
        }
        let mut seen = HashSet::with_capacity(keys.len());
        let jobs: Vec<Job> = keys
            .iter()
            .filter(|&&key| seen.insert(key) && self.corrections.find(&key).is_none())
            .filter_map(|&key| {
                let loc = self.file.locate(&key)?;
                Some(Job { key, loc })
            })
            .collect();
        Ok(self.prefetch.request(&self.file, &jobs)?)
    }

    /// Installs the correction tiles the prefetch workers have finished, and returns how many.
    ///
    /// A finished tile is installed only if it is still wanted and still current: not resident
    /// (a resident tile may have been edited since, and is what the store uses either way), its
    /// record still the latest (any write of the tile, by an edit's write-back or a save, moves
    /// it), and a slot to be had without failing (prefetch never takes a pinned slot). Anything
    /// else is dropped, as is a tile a worker couldn't read: a later load reads it again and
    /// reports the error.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] with [`PoolError::WriteBack`] if making room wrote a dirty tile back
    /// and that failed; the results not reached yet wait for the next poll.
    pub fn poll(&mut self) -> Result<usize, CacheError> {
        let mut installed = 0;
        while let Some(Fetched { job, slot }) = self.prefetch.next() {
            let Ok(slot) = slot else { continue };
            let current = self.file.locate(&job.key) == Some(job.loc);
            let outcome = if current {
                self.install(job.key, &slot)
            } else {
                Ok(false)
            };
            self.prefetch.recycle(slot);
            if outcome? {
                installed += 1;
            }
        }
        Ok(installed)
    }

    /// Prefetch requests not yet answered: queued, being read, or finished and waiting for
    /// [`TileStore::poll`].
    #[must_use]
    pub fn prefetching(&self) -> usize {
        self.prefetch.pending()
    }

    /// Puts a prefetched correction tile, read at its latest record, into a fresh slot; whether
    /// it did. A resident tile ([`PoolError::Present`]; every dirty tile is resident) is left
    /// alone, and a full pool ([`PoolError::Full`]) just drops the tile.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] with [`PoolError::WriteBack`], as for [`TileStore::poll`].
    fn install(&mut self, key: TileKey, tile: &CorrectionSlot) -> Result<bool, CacheError> {
        let id = match self.acquire_correction(key) {
            Ok(id) => id,
            Err(CacheError::Pool(PoolError::Present | PoolError::Full)) => return Ok(false),
            Err(e) => return Err(e),
        };
        *self.corrections.get_mut(id).ok_or(PoolError::Stale)? = *tile;
        #[cfg(test)]
        {
            self.log.prefetched += 1;
        }
        Ok(true)
    }

    /// Applies the unpins of dropped [`PinGuard`]s.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] if a pinned slot's handle went stale (pinned slots are never freed,
    /// so this doesn't happen). Every released set is unpinned either way.
    fn sync_pins(&mut self) -> Result<(), CacheError> {
        let released = std::mem::take(&mut self.pins.borrow_mut().released);
        let mut outcome = Ok(());
        for id in released {
            if let Some(set) = self.pin_sets.remove(&id) {
                keep_first(&mut outcome, self.unpin_set(&set));
            }
        }
        outcome
    }

    /// Unpins one pin set's slots.
    ///
    /// # Errors
    ///
    /// As [`TileStore::sync_pins`]; every slot is unpinned either way.
    fn unpin_set(&mut self, set: &[Pinned]) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &(evaluated, correction) in set {
            keep_first(&mut outcome, self.evaluated.unpin(evaluated));
            keep_first(&mut outcome, self.corrections.unpin(correction));
        }
        outcome
    }

    /// Evaluates `key` and pins its evaluated and correction slots.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; nothing stays pinned then.
    fn pin_one(&mut self, key: TileKey) -> Result<Pinned, CacheError> {
        let evaluated = self.ensure_positions(key)?;
        self.evaluated.pin(evaluated)?;
        let correction = self.load_correction(key).and_then(|id| {
            self.corrections.pin(id)?;
            Ok(id)
        });
        match correction {
            Ok(id) => Ok((evaluated, id)),
            Err(e) => {
                self.evaluated.unpin(evaluated)?;
                Err(e)
            }
        }
    }

    /// Whether pinning `keys` fits the budget: the chunks each pool needs to hold its pinned
    /// slots besides these tiles, plus one slot per tile, against the budget. Slots are counted
    /// as if packed into chunks; pins spread over chunks can still run a pool out of room later.
    ///
    /// # Errors
    ///
    /// [`CacheError::OverBudget`].
    fn check_pin_budget(&self, keys: &BTreeSet<TileKey>) -> Result<(), CacheError> {
        /// Bytes of chunks `pool` needs for its pinned slots outside `keys`, plus one per key.
        fn chunks_needed<T: bytemuck::Pod>(pool: &Pool<T>, keys: &BTreeSet<TileKey>) -> usize {
            let pinned_here = keys.iter().filter(|key| pool.is_pinned(key)).count();
            let slots = pool.pinned_slots().saturating_sub(pinned_here) + keys.len();
            slots.div_ceil(pool.per_chunk() as usize) * pool.chunk_bytes()
        }
        let needed = chunks_needed(&self.evaluated, keys) + chunks_needed(&self.corrections, keys);
        if needed > self.budget.limit() {
            return Err(CacheError::OverBudget {
                tiles: keys.len(),
                needed,
                budget: self.budget.limit(),
            });
        }
        Ok(())
    }

    /// Checks that `key` names a tile this store holds: base layer, displacement, level ≥ 1, a
    /// face of the mesh.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`].
    fn check_key(&self, key: &TileKey) -> Result<(), CacheError> {
        let ok = key.layer() == LayerId::BASE
            && key.channel() == Channel::DISPLACEMENT
            && key.level() != Level::ZERO
            && self.mesh.topology().contains(key.face());
        if ok {
            Ok(())
        } else {
            Err(CacheError::NotATile(*key))
        }
    }

    /// `key`'s correction slot: resident, else loaded from the file, else reset to zero.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if the record can't be read (the slot is freed again);
    /// [`CacheError::Pool`] if no slot can be had.
    fn load_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        if let Some((id, _)) = self.corrections.lookup(&key) {
            return Ok(id);
        }
        let id = self.acquire_correction(key)?;
        #[cfg(test)]
        self.log.file_reads.push(key);
        let slot = self.corrections.get_mut(id).ok_or(PoolError::Stale)?;
        let filled = match self
            .file
            .read_displacement_into(&key, slot, &mut self.read_scratch)
        {
            Ok(Some(_)) => Ok(()),
            Ok(None) => slot.reset(key.side()).map_err(CacheError::from),
            Err(e) => Err(e.into()),
        };
        if let Err(e) = filled {
            // The slot holds no tile: give it back (it is clean and unpinned).
            self.corrections.remove(&key)?;
            return Err(e);
        }
        Ok(id)
    }

    /// A fresh slot for `key`'s correction tile, taking chunks back from the pools if the
    /// correction pool is full (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`].
    fn acquire_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        for attempt in 0..2 {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            match self.corrections.acquire(key, &mut self.budget, |k, slot| {
                write_slot(file, unsaved, k, slot)
            }) {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.corrections.chunk_bytes())? {
                        return Err(PoolError::Full.into());
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(PoolError::Full.into())
    }

    /// A fresh slot for `key`'s evaluated tile, taking chunks back from the pools if the
    /// evaluated pool is full (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`].
    fn acquire_evaluated(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        for attempt in 0..2 {
            // Evaluated slots are never dirty: there is nothing to write back.
            let acquired = self.evaluated.acquire(key, &mut self.budget, |_, _| Ok(()));
            self.note_evictions();
            match acquired {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.evaluated.chunk_bytes())? {
                        return Err(PoolError::Full.into());
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(PoolError::Full.into())
    }

    /// Takes chunks back until another `need` bytes fit the budget; whether they do now.
    ///
    /// # Errors
    ///
    /// As [`TileStore::give_back`].
    fn make_room(&mut self, need: usize) -> Result<bool, CacheError> {
        let target = self.budget.limit().saturating_sub(need);
        self.give_back(target)?;
        Ok(self.budget.used() <= target)
    }

    /// Gives chunks without pinned slots back until the pools hold at most `bytes`: evaluated
    /// chunks first (a cache, cheap to recompute), then correction chunks, whose dirty tiles are
    /// written back to the working file first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] with [`PoolError::WriteBack`].
    fn give_back(&mut self, bytes: usize) -> Result<(), CacheError> {
        let evaluated = self
            .evaluated
            .shrink_to(&mut self.budget, bytes, |_, _| Ok(()));
        self.note_evictions();
        evaluated?;
        if self.budget.used() > bytes {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            self.corrections
                .shrink_to(&mut self.budget, bytes, |key, slot| {
                    write_slot(file, unsaved, key, slot)
                })?;
        }
        Ok(())
    }

    /// Tells the dependency graph which evaluated tiles the pool evicted.
    fn note_evictions(&mut self) {
        for key in self.evaluated.drain_evicted() {
            self.deps.left_pool(key);
        }
    }

    /// Drops every evaluation that followed from the `edited` tiles' corrections: positions
    /// through the recorded parents, transitively, and normals that read any of them. A pinned
    /// tile keeps its slot, marked stale, and is recomputed in place on its next evaluation.
    fn invalidate(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        let cascade = self.deps.cascade(edited);
        for key in cascade.normals {
            self.deps.drop_normals(key);
        }
        for key in cascade.positions {
            match self.evaluated.invalidate(&key) {
                Invalidated::Marked => {}
                Invalidated::Freed | Invalidated::Absent => self.deps.left_pool(key),
            }
        }
    }

    /// The slot of `key`'s current positions: cached and fresh, else computed (into a new slot,
    /// or in place over a stale one).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn ensure_positions(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        if let Some((id, Freshness::Fresh)) = self.evaluated.lookup(&key) {
            return Ok(id);
        }
        let region = Region::of_tile(&key);
        let parents = parent_tiles(&self.mesh, region)?;
        let mut held = Vec::with_capacity(parents.len());
        // The tile is stored and recorded before its parents are unpinned: a parent evicted in
        // between would take its node, and the edges to its own parents, out of the graph.
        let computed = self
            .predict_into_scratch(key, region, &parents, &mut held)
            .and_then(|()| self.store_positions(key, &parents));
        let released = self.unpin_held(&held);
        let id = computed?;
        released?;
        Ok(id)
    }

    /// Copies the positions in the scratch buffer into `key`'s slot (a new one, or its stale
    /// one in place), marks it fresh and records it as predicted from `parents`, which the
    /// caller still holds pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] if no slot can be had.
    fn store_positions(
        &mut self,
        key: TileKey,
        parents: &BTreeSet<TileKey>,
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        // A stale slot (pinned elsewhere, so still here) is overwritten in place: its handles
        // stay valid.
        let id = match self.evaluated.lookup(&key) {
            Some((id, _)) => id,
            None => self.acquire_evaluated(key)?,
        };
        let n = vertex_count(&key);
        let slot = self.evaluated.get_mut(id).ok_or(PoolError::Stale)?;
        prefix_mut(&mut slot.positions, n)?.copy_from_slice(prefix(&self.scratch, n)?);
        self.evaluated.mark_fresh(id)?;
        self.deps.record_positions(key, parents);
        #[cfg(test)]
        self.log.positions.push(key);
        Ok(id)
    }

    /// Evaluates and pins each parent tile (level 0 is the mesh), loads `key`'s corrections,
    /// predicts `region` and applies them into the scratch buffer. The pinned parents are in
    /// `held` for the caller to unpin, even on error.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn predict_into_scratch(
        &mut self,
        key: TileKey,
        region: Region,
        parents: &BTreeSet<TileKey>,
        held: &mut Held,
    ) -> Result<(), CacheError> {
        for &parent in parents {
            if parent.level() == Level::ZERO {
                continue;
            }
            let id = self.ensure_positions(parent)?;
            self.evaluated.pin(id)?;
            held.push((parent, id));
        }
        // Nothing acquires a slot between this load and the read below, so the correction slot
        // needn't be pinned.
        let correction = self.load_correction(key)?;
        let corrections = self
            .corrections
            .get(correction)
            .ok_or(PoolError::Stale)?
            .view()?;
        let source = Resident {
            mesh: &self.mesh,
            pool: &self.evaluated,
            tiles: held,
        };
        let prediction = predict_region(&self.mesh, &source, region)?;
        let out = prefix_mut(&mut self.scratch, vertex_count(&key))?;
        prediction.apply_into(|i, j| corrections.get(i, j).unwrap_or(Vec3::ZERO), out)?;
        Ok(())
    }

    /// The slot of `key`'s current positions and normals: cached, else the normals are computed
    /// from the positions of its [`normal_tiles`], each evaluated and pinned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn ensure_normals(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        if let Some((id, Freshness::Fresh)) = self.evaluated.lookup(&key)
            && self.deps.has_normals(&key)
        {
            return Ok(id);
        }
        let region = Region::of_tile(&key);
        let mut inputs = normal_tiles(&self.mesh, region)?;
        // The tile's own block is among them; inserting it makes that plain.
        inputs.insert(key);
        let mut held = Vec::with_capacity(inputs.len());
        // Recorded while the inputs are pinned, as positions are (see `ensure_positions`).
        let computed = self
            .normals_into_slot(key, region, &inputs, &mut held)
            .inspect(|_| {
                self.deps.record_normals(key, &inputs);
                #[cfg(test)]
                self.log.normals.push(key);
            });
        let released = self.unpin_held(&held);
        let id = computed?;
        released?;
        Ok(id)
    }

    /// Evaluates and pins each of `inputs` (which include `key`), computes `region`'s normals
    /// and writes them into `key`'s slot, which it returns. The pinned tiles are in `held` for
    /// the caller to unpin, even on error.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn normals_into_slot(
        &mut self,
        key: TileKey,
        region: Region,
        inputs: &BTreeSet<TileKey>,
        held: &mut Held,
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        let mut own = None;
        for &input in inputs {
            let id = self.ensure_positions(input)?;
            self.evaluated.pin(id)?;
            held.push((input, id));
            if input == key {
                own = Some(id);
            }
        }
        let own = own.ok_or(PoolError::Stale)?;
        let n = vertex_count(&key);
        let source = Resident {
            mesh: &self.mesh,
            pool: &self.evaluated,
            tiles: held,
        };
        normals_into(
            &self.mesh,
            &source,
            region,
            prefix_mut(&mut self.scratch, n)?,
        )?;
        let slot = self.evaluated.get_mut(own).ok_or(PoolError::Stale)?;
        prefix_mut(&mut slot.normals, n)?.copy_from_slice(prefix(&self.scratch, n)?);
        Ok(own)
    }

    /// Unpins the evaluated tiles an evaluation pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Pool`] if a handle went stale (pinned slots are never freed, so this
    /// doesn't happen); every tile is unpinned either way.
    fn unpin_held(&mut self, held: &Held) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &(_, id) in held {
            keep_first(&mut outcome, self.evaluated.unpin(id));
        }
        outcome
    }
}

#[cfg(test)]
mod tests;
