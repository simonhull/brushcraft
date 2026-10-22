//! The tile store: a project held out of core.
//!
//! [`TileStore`] owns the base mesh, the working [`ProjectFile`], two budgeted pools (correction
//! tiles and evaluated tiles), the change feed and an op generator. Correction tiles are loaded
//! from the file on demand (an absent tile is all zero) and edited in their slots; evaluated
//! tiles are computed on demand from their parent tiles and cached. Edits drop exactly the
//! evaluations that followed from them (see [`crate::deps`]).
//!
//! # Evaluation
//!
//! A tile is evaluated coarse to fine, never recursively. The store lists the tile's dependency
//! cone: the tile, then the parents of every tile of the level above, level by level, down to
//! the first level whose tiles are all cached (or level 1, predicted from the base mesh). It
//! pins that level, then builds each finer level tile by tile from the pinned level below,
//! pinning each tile it makes and unpinning the level below once the level is done. At most two
//! levels of the cone are pinned at once, and no level is wider than [`geom::cone_width`], so an
//! evaluation pins at most twice that many evaluated tiles, whatever its depth. Normals are
//! evaluated the same way, from the cone of the tiles they read.
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
//!
//! Within the budget the pools rebalance towards corrections. When a correction tile needs a
//! slot and the correction pool would have to evict a dirty one (a write to the working file
//! now, and a read back if the tile is wanted again), it first takes an evaluated chunk without
//! pins back, if there is one, and maps a correction chunk in its place: an evaluated tile is
//! recomputed from resident tiles without any I/O. A clean victim is evicted as before.
//!
//! A budget of [`StoreConfig::minimum_budget`] is enough for every evaluation and edit when
//! nothing else is pinned, and [`TileStore::open`] refuses less. An evaluation pins at most `E`
//! evaluated slots (two cone levels) and no correction slot:
//!
//! - The evaluated pool needs a chunk only when every one of its slots is pinned, so it holds fewer
//!   than `⌈E / per chunk⌉` chunks then, and the minimum leaves room for one more besides a
//!   correction chunk. Taking correction chunks back stops as soon as that room is there, so the
//!   correction pool keeps a chunk, however the evaluated pins are spread.
//! - The correction pool so always has a slot to reuse: none of its slots is pinned. (Its first
//!   chunk is mapped by an evaluation's first build, which loads a correction tile before it asks
//!   for an evaluated slot.)
//!
//! An edit pins correction slots only, at most one cone width of them, which the minimum's
//! correction chunks hold once every evaluated chunk is taken back.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::convert::Infallible;
#[cfg(test)]
use std::io;
use std::rc::Rc;
use std::sync::Arc;

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionView, EvalError, FaceId, LayerId, Level,
    MAX_COPIES, PositionSource, Region, TileKey, Vec3, normal_tiles, normals_into, parent_tiles,
    predict_region,
};
use thiserror::Error;

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
    /// Bytes the two pools may hold together. The pools never exceed it. At least
    /// [`StoreConfig::minimum_budget`] for the store's mesh.
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

    /// The least [`StoreConfig::budget_bytes`] a store over `mesh` with these chunks works
    /// with: enough that no evaluation, normals computation or edit runs out of room while
    /// nothing else is pinned.
    ///
    /// That is the larger of
    ///
    /// - evaluated chunks for twice the mesh's [`geom::cone_width`] (the two cone levels an
    ///   evaluation holds pinned, see the module notes) plus one correction chunk, and
    /// - correction chunks for one cone width of tiles (an edit holds every tile it writes a copy
    ///   of pinned: at most the tile's neighbourhood, which the cone width bounds).
    ///
    /// Pins taken with [`TileStore::pin`] come on top. With the default chunks the minimum is two
    /// or three chunks for a mesh of thousands of faces; it grows with the mesh's face
    /// neighbourhoods, never with the evaluated level.
    ///
    /// Computed from the mesh's face rings, in time linear in its faces: [`TileStore::open`] runs
    /// it once.
    #[must_use]
    pub fn minimum_budget(&self, mesh: &BaseMesh) -> usize {
        Headroom::of(mesh).minimum_budget(self.chunk_bytes)
    }
}

/// The most slots of each pool the store pins for its own work: what [`StoreConfig`]'s minimum
/// is made of, and what [`TileStore::pin`] counts besides the tiles asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Headroom {
    /// Evaluated slots one evaluation pins at most: two levels of its cone.
    evaluated: usize,
    /// Correction slots one edit pins at most: the tiles holding copies of its vertices.
    corrections: usize,
}

impl Headroom {
    /// The headroom for evaluating and editing `mesh`.
    fn of(mesh: &BaseMesh) -> Self {
        let width = geom::cone_width(mesh);
        Self {
            evaluated: 2 * width,
            corrections: width,
        }
    }

    /// The least budget, in chunks of `chunk_bytes`, that holds this headroom however the
    /// evaluated pins are spread (see the module notes).
    fn minimum_budget(self, chunk_bytes: usize) -> usize {
        let (per_evaluated, evaluated) = Pool::<EvaluatedSlot>::layout(chunk_bytes);
        let (per_correction, correction) = Pool::<CorrectionSlot>::layout(chunk_bytes);
        let chunks = |slots: usize, per: u32| slots.div_ceil(per as usize);
        let evaluating = chunks(self.evaluated, per_evaluated) * evaluated + correction;
        let editing = chunks(self.corrections, per_correction) * correction;
        evaluating.max(editing)
    }
}

/// Why a [`TileStore`] operation failed.
#[derive(Debug, Error)]
pub enum CacheError {
    /// Reading the working file, or recording a save point in it, failed.
    #[error("project file operation failed")]
    Store(#[from] StoreError),
    /// No slot to be had: every slot of the pool that needs one is pinned, and no chunk without a
    /// pin is left to take back. Only pins taken with [`TileStore::pin`] can crowd the store's
    /// own work this far (see [`StoreConfig::minimum_budget`]).
    #[error("every slot is pinned and no chunk can be taken back")]
    Full,
    /// Writing a dirty correction tile to the working file failed: when its slot was needed, or
    /// on [`TileStore::save`]. The tile stays resident and dirty.
    #[error("writing a dirty correction tile back failed")]
    WriteBack(#[source] StoreError),
    /// Evaluation failed (not for a valid mesh).
    #[error("evaluation failed")]
    Eval(#[from] EvalError),
    /// A correction was rejected: not finite, at level 0, on an unknown face, or outside the face
    /// or tile. Nothing was written.
    #[error("correction rejected")]
    Correction(#[source] CorrectionError),
    /// The key isn't a base-layer displacement tile above level 0 of a face of this mesh.
    #[error("{0:?} is not a base-layer displacement tile of this mesh above level 0")]
    NotATile(TileKey),
    /// The configured site is [`RESERVED_SITE`].
    #[error("site {RESERVED_SITE} is reserved")]
    ReservedSite,
    /// The configured budget is below [`StoreConfig::minimum_budget`] for the mesh.
    #[error("the budget is below the {minimum} bytes this mesh needs")]
    BudgetTooSmall {
        /// The least budget the mesh works with.
        minimum: usize,
    },
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
    /// The store found one of its own invariants broken: a bug in the store, never caused by
    /// its input or the file's contents.
    #[error("internal invariant broken")]
    Internal(#[from] Invariant),
}

/// An invariant of [`TileStore`] found broken (see [`CacheError::Internal`]).
#[derive(Debug, Error)]
pub enum Invariant {
    /// A dirty correction tile had no op to write it with.
    #[error("dirty correction tile {0:?} has no op")]
    MissingOp(TileKey),
    /// A correction slot didn't hold a valid tile, or refused a write the store had checked.
    #[error("a correction slot is broken")]
    CorrectionSlot(#[source] CorrectionError),
    /// A slot handle was used after its slot was freed.
    #[error("a slot handle outlived its slot")]
    StaleHandle,
    /// A tile was given a second slot in one pool.
    #[error("a tile was given a second slot")]
    SecondSlot,
    /// A slot was unpinned more often than it was pinned.
    #[error("a slot was unpinned more often than pinned")]
    NotPinned,
    /// A pinned slot was to be freed.
    #[error("a pinned slot was to be freed")]
    FreedPinned,
    /// A dirty slot was to be freed without being written back.
    #[error("a dirty slot was to be freed")]
    FreedDirty,
    /// A slot or buffer held fewer values than a tile has.
    #[error("a buffer of {got} values is shorter than a tile's {expected}")]
    ShortBuffer {
        /// Values a tile has.
        expected: usize,
        /// Values the buffer held.
        got: usize,
    },
}

/// No operation fails with [`Infallible`]: this lets `?` take pool results whose write-back
/// can't fail.
impl From<Infallible> for CacheError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

/// A pool's error, as the store reports it: [`CacheError::Full`], the write-back's own error,
/// or a broken invariant.
impl<W> From<PoolError<W>> for CacheError
where
    Self: From<W>,
{
    fn from(e: PoolError<W>) -> Self {
        match e {
            PoolError::Full => Self::Full,
            PoolError::WriteBack(w) => w.into(),
            PoolError::Present => Invariant::SecondSlot.into(),
            PoolError::Stale => Invariant::StaleHandle.into(),
            PoolError::NotPinned => Invariant::NotPinned.into(),
            PoolError::Pinned => Invariant::FreedPinned.into(),
            PoolError::Dirty => Invariant::FreedDirty.into(),
        }
    }
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

/// Evaluated tiles an evaluation pinned, with their keys, in key order: what its [`Resident`]
/// source reads.
type Held = Vec<(TileKey, SlotId<EvaluatedSlot>)>;

/// One level of a dependency cone: each tile, with the parents it is predicted from (left empty
/// for the coarsest level when its tiles are all cached, since nothing is built there).
type ConeLevel = BTreeMap<TileKey, Vec<TileKey>>;

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
    /// Invalidations run (dependency cascades walked) for edited tiles.
    invalidations: usize,
    /// Parent lists read back from the dependency graph. (Counted through `&self`, as the
    /// lists are looked up, hence the cells.)
    parents_reused: Cell<usize>,
    /// Parent lists listed with `parent_tiles`.
    parents_listed: Cell<usize>,
    /// Normal input lists read back from the dependency graph.
    normal_inputs_reused: Cell<usize>,
    /// Normal input lists listed with `normal_tiles`.
    normal_inputs_listed: Cell<usize>,
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
    /// What the store pins for its own work, besides [`TileStore::pin`]'s tiles.
    headroom: Headroom,
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
    /// A correction tile whose reads from the working file fail, for tests of error paths.
    #[cfg(test)]
    unreadable: Option<TileKey>,
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
    /// The pinned tiles, in key order.
    tiles: &'a [(TileKey, SlotId<EvaluatedSlot>)],
}

impl PositionSource for Resident<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        if level == Level::ZERO {
            return self.mesh.position(face, level, li, lj);
        }
        let at = self.tiles.binary_search_by_key(&key, |&(k, _)| k).ok()?;
        let &(_, id) = self.tiles.get(at)?;
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
/// [`Invariant::ShortBuffer`] if there are fewer (no slot or scratch buffer is smaller than the
/// largest tile, so this doesn't happen).
fn prefix(values: &[Vec3], n: usize) -> Result<&[Vec3], Invariant> {
    values.get(..n).ok_or(Invariant::ShortBuffer {
        expected: n,
        got: values.len(),
    })
}

/// [`prefix`], mutably.
///
/// # Errors
///
/// As [`prefix`].
fn prefix_mut(values: &mut [Vec3], n: usize) -> Result<&mut [Vec3], Invariant> {
    let got = values.len();
    values
        .get_mut(..n)
        .ok_or(Invariant::ShortBuffer { expected: n, got })
}

/// The evaluated pool's write-back: evaluated slots are never dirty, so there is nothing to
/// write.
///
/// # Errors
///
/// Never.
#[expect(
    clippy::unnecessary_wraps,
    reason = "it is the pools' write-back, whose signature returns a result"
)]
const fn nothing_to_write<T>(_: TileKey, _: &T) -> Result<(), Infallible> {
    Ok(())
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
/// [`CacheError::WriteBack`] if the file's write fails; [`CacheError::Internal`] if the slot has
/// no recorded op (every slot marked dirty gets one first) or doesn't hold a valid tile.
fn write_slot(
    file: &mut ProjectFile,
    unsaved: &mut HashMap<TileKey, OpId>,
    key: TileKey,
    slot: &CorrectionSlot,
) -> Result<(), CacheError> {
    let op = unsaved
        .get(&key)
        .copied()
        .ok_or(Invariant::MissingOp(key))?;
    let tile = slot.to_tile().map_err(Invariant::CorrectionSlot)?;
    file.write_displacement(key, op, &tile)
        .map_err(CacheError::WriteBack)?;
    unsaved.remove(&key);
    Ok(())
}

impl TileStore {
    /// A store over `file` (the working file) for `mesh`. The op generator starts after the
    /// file's head op if that op is of `config.site`, else at sequence 0 of the site.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::BudgetTooSmall`] below
    /// [`StoreConfig::minimum_budget`]; [`CacheError::OpsExhausted`] if the file's head is the
    /// site's last op id; [`CacheError::Store`] if the read buffers can't be made.
    pub fn open(
        mesh: Arc<BaseMesh>,
        file: ProjectFile,
        config: StoreConfig,
    ) -> Result<Self, CacheError> {
        if config.site == RESERVED_SITE {
            return Err(CacheError::ReservedSite);
        }
        let headroom = Headroom::of(&mesh);
        let minimum = headroom.minimum_budget(config.chunk_bytes);
        if config.budget_bytes < minimum {
            return Err(CacheError::BudgetTooSmall { minimum });
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
            headroom,
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
            #[cfg(test)]
            unreadable: None,
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
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if no slot can be had.
    pub fn correction(&mut self, key: TileKey) -> Result<CorrectionView<'_>, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.load_correction(key)?;
        let slot = self.corrections.get(id).ok_or(Invariant::StaleHandle)?;
        Ok(slot.view().map_err(Invariant::CorrectionSlot)?)
    }

    /// Sets the correction at face-grid vertex `(i, j)` of `face` at `level` to `c`, as edit
    /// `op`: in every copy of the vertex (one per tile holding it, in every face sharing it).
    ///
    /// Each copy's slot becomes dirty, and its key goes to the change feed with `op` unless it
    /// is there with `op` already (dirty since an earlier edit of the same op). Every cached
    /// evaluation that followed from those tiles is dropped (pinned ones are recomputed in
    /// place on their next [`TileStore::evaluate`]); a tile nothing has been computed from since
    /// it was last dropped is skipped, so repeated edits of one tile between evaluations walk
    /// its dependencies once. To edit many vertices of one tile, [`TileStore::edit_tile`] does
    /// the bookkeeping once for all of them.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`],
    /// [`CorrectionError::LevelZero`], [`CorrectionError::UnknownFace`] or
    /// [`CorrectionError::OutOfTile`], before anything is written; [`CacheError::Store`],
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if a copy's tile can't be loaded, also
    /// before anything is written. Should writing a copy fail all the same
    /// ([`CacheError::Internal`]: a broken slot), the copies written before it are kept, and
    /// their evaluations dropped, before the error is returned.
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
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        if level == Level::ZERO {
            return Err(CacheError::Correction(CorrectionError::LevelZero));
        }
        let topology = self.mesh.topology();
        if !topology.contains(face) {
            return Err(CacheError::Correction(CorrectionError::UnknownFace(face)));
        }
        let copies = topology.copies(face, level, i, j);
        if copies.is_empty() {
            let side = level.face_quads() + 1;
            return Err(CacheError::Correction(CorrectionError::OutOfTile {
                i,
                j,
                side,
            }));
        }
        // Every copy is loaded and pinned before any is written, so a failed load leaves the
        // twins agreeing.
        let mut held = [None; MAX_COPIES];
        let loaded = copies.iter().zip(&mut held).try_for_each(|(copy, place)| {
            let id = self.load_correction(copy.key(level))?;
            self.corrections.pin(id)?;
            *place = Some(id);
            Ok::<_, CacheError>(())
        });
        let mut written = 0;
        let outcome = loaded.and_then(|()| {
            for (copy, &id) in copies.iter().zip(held.iter().flatten()) {
                self.write_correction(copy.key(level), id, (copy.i, copy.j), c, op)?;
                written += 1;
            }
            Ok(())
        });
        let released = held
            .iter()
            .flatten()
            .try_for_each(|&id| self.corrections.unpin(id));
        // The copies written are invalidated even if a later one failed.
        self.invalidate_edited(copies.iter().take(written).map(|copy| copy.key(level)));
        outcome?;
        released?;
        Ok(())
    }

    /// Edits `key`'s corrections as edit `op`: `f` sets vertices through a [`TileEdit`], and
    /// its result is returned.
    ///
    /// This is [`TileStore::set_correction`] for many vertices of one tile, with the bookkeeping
    /// done once per tile instead of once per vertex: each tile written (the edited one, and the
    /// neighbours holding copies of its border vertices) is loaded and pinned when first
    /// touched, marked dirty and put on the change feed at its first write, and its evaluations
    /// are dropped once, when `f` returns. Errors of single writes go to `f`.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Store`], [`CacheError::Full`] or
    /// [`CacheError::WriteBack`] if the tile's corrections can't be loaded (nothing is written
    /// then); [`CacheError::Internal`] if a slot handle went stale.
    pub fn edit_tile<R>(
        &mut self,
        key: TileKey,
        op: OpId,
        f: impl FnOnce(&mut TileEdit<'_>) -> R,
    ) -> Result<R, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.load_correction(key)?;
        self.corrections.pin(id)?;
        let mut edit = TileEdit {
            store: self,
            key,
            op,
            touched: vec![Touched {
                key,
                id,
                written: false,
            }],
        };
        let result = f(&mut edit);
        let touched = edit.touched;
        let released = touched
            .iter()
            .try_for_each(|tile| self.corrections.unpin(tile.id));
        self.invalidate_edited(
            touched
                .iter()
                .filter(|tile| tile.written)
                .map(|tile| tile.key),
        );
        released?;
        Ok(result)
    }

    /// `key`'s evaluated positions, row-major `side²`: cached, else computed coarse to fine
    /// from the base mesh, or from the finest level of its cone that is cached (see the module
    /// notes), with its corrections.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Full`] if the pins taken with
    /// [`TileStore::pin`] leave too little of the budget for the evaluation's own (never without
    /// such pins); [`CacheError::Store`] if a correction tile can't be read;
    /// [`CacheError::WriteBack`] if making room for one fails; [`CacheError::Eval`] (not for a
    /// valid mesh).
    pub fn evaluate(&mut self, key: TileKey) -> Result<&[Vec3], CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.ensure_positions(key)?;
        let slot = self.evaluated.get(id).ok_or(Invariant::StaleHandle)?;
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
        let slot = self.evaluated.get(id).ok_or(Invariant::StaleHandle)?;
        Ok(prefix(&slot.normals, vertex_count(&key))?)
    }

    /// Evaluates every tile of `keys` and pins it, with its correction tile, until the guard is
    /// dropped.
    ///
    /// The request is first counted against the budget: the chunks needed to hold its tiles
    /// besides every slot pinned already, plus the slots an evaluation pins while it works (two
    /// levels of a cone, see [`StoreConfig::minimum_budget`]). If that is more than the budget,
    /// nothing is evicted or pinned. The count packs slots into chunks, so pins spread over
    /// more chunks than that can still leave an evaluation short of room; then, as when a
    /// correction tile can't be read, the pins this call took are released and the pins held
    /// before are untouched.
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
    /// [`CacheError::WriteBack`] if a tile's write fails; [`CacheError::Store`] if the save
    /// point's does.
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
                let slot = self.corrections.get(id).ok_or(Invariant::StaleHandle)?;
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
    /// [`CacheError::WriteBack`]: that tile and those not reached yet stay.
    pub fn evict_all(&mut self) -> Result<(), CacheError> {
        self.sync_pins()?;
        let evaluated = self.evaluated.evict_all(nothing_to_write);
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
    /// workers can't be started.
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
    /// [`CacheError::WriteBack`] if making room wrote a dirty tile back and that failed; the
    /// results not reached yet wait for the next poll.
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
    /// it did. A resident tile (every dirty tile is resident) is left alone, and a full pool
    /// ([`CacheError::Full`]) just drops the tile.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`], as for [`TileStore::poll`].
    fn install(&mut self, key: TileKey, tile: &CorrectionSlot) -> Result<bool, CacheError> {
        if self.corrections.find(&key).is_some() {
            return Ok(false);
        }
        let id = match self.acquire_correction(key) {
            Ok(id) => id,
            Err(CacheError::Full) => return Ok(false),
            Err(e) => return Err(e),
        };
        *self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)? = *tile;
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
    /// [`CacheError::Internal`] if a pinned slot's handle went stale (pinned slots are never
    /// freed, so this doesn't happen). Every released set is unpinned either way.
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
        let held = self.pin_current(BTreeSet::from([key]))?;
        let evaluated = held
            .first()
            .map(|&(_, id)| id)
            .ok_or(Invariant::StaleHandle)?;
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
    /// slots besides these tiles, one slot per tile, and the store's own headroom (an
    /// evaluation's pins, and the correction slot it loads), against the budget. Slots are
    /// counted as if packed into chunks; pins spread over chunks can still run a pool out of
    /// room later.
    ///
    /// # Errors
    ///
    /// [`CacheError::OverBudget`].
    fn check_pin_budget(&self, keys: &BTreeSet<TileKey>) -> Result<(), CacheError> {
        /// Bytes of chunks `pool` needs for its pinned slots outside `keys`, one per key, and
        /// `headroom` more.
        fn chunks_needed<T: bytemuck::Pod>(
            pool: &Pool<T>,
            keys: &BTreeSet<TileKey>,
            headroom: usize,
        ) -> usize {
            let pinned_here = keys.iter().filter(|key| pool.is_pinned(key)).count();
            let slots = pool.pinned_slots().saturating_sub(pinned_here) + keys.len() + headroom;
            slots.div_ceil(pool.per_chunk() as usize) * pool.chunk_bytes()
        }
        let needed = chunks_needed(&self.evaluated, keys, self.headroom.evaluated)
            + chunks_needed(&self.corrections, keys, 1);
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
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if no slot can be had.
    fn load_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        if let Some((id, _)) = self.corrections.lookup(&key) {
            return Ok(id);
        }
        let id = self.acquire_correction(key)?;
        #[cfg(test)]
        self.log.file_reads.push(key);
        let slot = self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)?;
        let read = self
            .file
            .read_displacement_into(&key, slot, &mut self.read_scratch);
        #[cfg(test)]
        let read = if self.unreadable == Some(key) {
            Err(io::Error::other("an unreadable tile, for the test").into())
        } else {
            read
        };
        let filled = match read {
            Ok(Some(_)) => Ok(()),
            Ok(None) => slot
                .reset(key.side())
                .map_err(|e| Invariant::CorrectionSlot(e).into()),
            Err(e) => Err(e.into()),
        };
        if let Err(e) = filled {
            // The slot holds no tile: give it back (it is clean and unpinned).
            self.corrections.remove(&key)?;
            return Err(e);
        }
        Ok(id)
    }

    /// A fresh slot for `key`'s correction tile: taking an evaluated chunk back rather than
    /// writing a dirty correction tile back, and taking chunks back from the pools if the
    /// correction pool is full (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`]; [`CacheError::WriteBack`] if a dirty tile's write-back fails.
    fn acquire_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        let chunk = self.corrections.chunk_bytes();
        let evicts_dirty = !self.corrections.has_free_slot()
            && !self.budget.fits(chunk)
            && self.corrections.victim_is_dirty();
        if evicts_dirty && self.evaluated.has_unpinned_chunk() {
            // Rebalancing (see the module notes): evaluated chunks only, so nothing is written.
            let target = self.budget.limit().saturating_sub(chunk);
            let shrunk = self
                .evaluated
                .shrink_to(&mut self.budget, target, nothing_to_write);
            self.note_evictions();
            shrunk?;
        }
        for attempt in 0..2 {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            match self.corrections.acquire(key, &mut self.budget, |k, slot| {
                write_slot(file, unsaved, k, slot)
            }) {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.corrections.chunk_bytes())? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
    }

    /// A fresh slot for `key`'s evaluated tile, taking chunks back from the pools if the
    /// evaluated pool is full (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`]; [`CacheError::WriteBack`] if taking a correction chunk back
    /// writes a dirty tile back and that fails.
    fn acquire_evaluated(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        for attempt in 0..2 {
            // Evaluated slots are never dirty: there is nothing to write back.
            let acquired = self
                .evaluated
                .acquire(key, &mut self.budget, nothing_to_write);
            self.note_evictions();
            match acquired {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.evaluated.chunk_bytes())? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
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
    /// [`CacheError::WriteBack`].
    fn give_back(&mut self, bytes: usize) -> Result<(), CacheError> {
        let evaluated = self
            .evaluated
            .shrink_to(&mut self.budget, bytes, nothing_to_write);
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

    /// Writes `c` at `(i, j)` of `key`'s correction slot `id` (resident and pinned) as edit `op`,
    /// and marks the slot written (see [`TileStore::mark_written`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle, or if the slot doesn't hold a valid tile or
    /// `(i, j)` is outside it (neither happens for a loaded slot and a copy's coordinates).
    fn write_correction(
        &mut self,
        key: TileKey,
        id: SlotId<CorrectionSlot>,
        (i, j): (u32, u32),
        c: Vec3,
        op: OpId,
    ) -> Result<(), CacheError> {
        let slot = self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)?;
        slot.view_mut()
            .and_then(|mut view| view.set(i, j, c))
            .map_err(Invariant::CorrectionSlot)?;
        self.mark_written(key, id, op)
    }

    /// Records that `key`'s correction slot `id` was changed by `op`: dirty, to be written with
    /// `op`, and on the change feed with it, unless it is there with `op` already (it was dirty
    /// with `op` before, and a dirty slot's key stays in the feed until it is written).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn mark_written(
        &mut self,
        key: TileKey,
        id: SlotId<CorrectionSlot>,
        op: OpId,
    ) -> Result<(), CacheError> {
        if self.unsaved.insert(key, op) != Some(op) {
            self.feed.push(key, op);
        }
        self.corrections.set_dirty(id, true)?;
        Ok(())
    }

    /// Drops the evaluations that followed from the `edited` tiles' corrections (see
    /// [`TileStore::invalidate`]), skipping each tile nothing can have followed from: one whose
    /// own positions aren't current and that nothing recorded reads.
    ///
    /// Anything computed from a tile's corrections is computed from its positions, and its
    /// positions only from the tile being evaluated, which makes them current. So once a tile's
    /// evaluations are dropped, edits of it have nothing more to drop until it is evaluated
    /// again, however many there are and whatever their ops.
    fn invalidate_edited(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        let reached: Vec<TileKey> = edited
            .into_iter()
            .filter(|key| {
                self.evaluated.freshness(key) == Some(Freshness::Fresh) || self.deps.is_read(key)
            })
            .collect();
        if !reached.is_empty() {
            self.invalidate(reached);
        }
    }

    /// Drops every evaluation that followed from the `edited` tiles' corrections: positions
    /// through the recorded parents, transitively, and normals that read any of them. A pinned
    /// tile keeps its slot, marked stale, and is recomputed in place on its next evaluation.
    fn invalidate(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        #[cfg(test)]
        {
            self.log.invalidations += 1;
        }
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

    /// The slot of `key`'s current positions: cached and fresh, else evaluated coarse to fine.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn ensure_positions(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        if let Some((id, Freshness::Fresh)) = self.evaluated.lookup(&key) {
            return Ok(id);
        }
        let held = self.pin_current(BTreeSet::from([key]))?;
        let id = held
            .first()
            .map(|&(_, id)| id)
            .ok_or(Invariant::StaleHandle)?;
        self.unpin_held(&held)?;
        Ok(id)
    }

    /// Makes every tile of `targets` (tiles of one level) current and pins it, evaluating its
    /// dependency cone coarse to fine (see the module notes). Returns the tiles pinned, for the
    /// caller to unpin with [`TileStore::unpin_held`]; nothing stays pinned on error.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn pin_current(&mut self, targets: BTreeSet<TileKey>) -> Result<Held, CacheError> {
        let cone = self.cone(targets)?;
        let mut below = Held::new();
        for level in cone.iter().rev() {
            let built = self.pin_level(level, &below);
            let released = self.unpin_held(&below);
            below = built?;
            if let Err(e) = released {
                // Pinned slots are never freed, so this doesn't happen; this level's pins go
                // too.
                self.unpin_held(&below)?;
                return Err(e);
            }
        }
        Ok(below)
    }

    /// The dependency cone of `targets`, finest level first: the targets, then the parents of
    /// every tile of the level above, down to the first level whose tiles are all current, or
    /// to level 1 (predicted from the mesh). Each tile comes with its parents, except on a
    /// current coarsest level.
    ///
    /// The parents of a current tile are listed with the rest of its level: a tile current now
    /// may be evicted while the coarser levels are built, and is then rebuilt from them.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn cone(&mut self, targets: BTreeSet<TileKey>) -> Result<Vec<ConeLevel>, CacheError> {
        let mut cone = Vec::new();
        let mut tiles = targets;
        loop {
            if tiles.iter().all(|key| self.is_current(key)) {
                cone.push(tiles.into_iter().map(|key| (key, Vec::new())).collect());
                return Ok(cone);
            }
            let level = tiles
                .into_iter()
                .map(|key| Ok((key, self.parents_of(key)?)))
                .collect::<Result<ConeLevel, CacheError>>()?;
            tiles = level.values().flatten().copied().collect();
            cone.push(level);
            if tiles.is_empty() {
                return Ok(cone);
            }
        }
    }

    /// Whether `key`'s evaluated positions are cached and current.
    fn is_current(&mut self, key: &TileKey) -> bool {
        matches!(self.evaluated.lookup(key), Some((_, Freshness::Fresh)))
    }

    /// The tiles `key`'s positions are predicted from, level 0 (the mesh) left out, in key order:
    /// as the dependency graph recorded them if `key` still has a node, else from
    /// [`parent_tiles`].
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn parents_of(&self, key: TileKey) -> Result<Vec<TileKey>, CacheError> {
        if let Some(parents) = self.deps.parents(&key) {
            #[cfg(test)]
            self.log
                .parents_reused
                .set(self.log.parents_reused.get() + 1);
            return Ok(parents.to_vec());
        }
        #[cfg(test)]
        self.log
            .parents_listed
            .set(self.log.parents_listed.get() + 1);
        let parents = parent_tiles(&self.mesh, Region::of_tile(&key))?;
        Ok(parents
            .into_iter()
            .filter(|parent| parent.level() != Level::ZERO)
            .collect())
    }

    /// The tiles `key`'s normals are computed from, the tile itself included: as the dependency
    /// graph recorded them if `key` still has a node that lists them, else from
    /// [`normal_tiles`].
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn normal_inputs_of(&self, key: TileKey) -> Result<BTreeSet<TileKey>, CacheError> {
        if let Some(inputs) = self.deps.normal_inputs(&key) {
            #[cfg(test)]
            self.log
                .normal_inputs_reused
                .set(self.log.normal_inputs_reused.get() + 1);
            return Ok(inputs.iter().copied().collect());
        }
        #[cfg(test)]
        self.log
            .normal_inputs_listed
            .set(self.log.normal_inputs_listed.get() + 1);
        let mut inputs = normal_tiles(&self.mesh, Region::of_tile(&key))?;
        // The tile's own block is among them; inserting it makes that plain.
        inputs.insert(key);
        Ok(inputs)
    }

    /// Makes every tile of one cone level current, building those that aren't from `below` (the
    /// level under it, pinned), and pins it. Returns them pinned, in key order; nothing this call
    /// pinned stays pinned on error.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn pin_level(&mut self, level: &ConeLevel, below: &Held) -> Result<Held, CacheError> {
        let mut held = Held::with_capacity(level.len());
        let pinned = level.iter().try_for_each(|(&key, parents)| {
            let id = match self.evaluated.lookup(&key) {
                Some((id, Freshness::Fresh)) => id,
                _ => self.build(key, parents, below)?,
            };
            self.evaluated.pin(id)?;
            held.push((key, id));
            Ok::<_, CacheError>(())
        });
        if let Err(e) = pinned {
            self.unpin_held(&held)?;
            return Err(e);
        }
        Ok(held)
    }

    /// Computes `key`'s positions from its `parents` (all pinned in `below`) and its
    /// corrections, and stores them in its slot (see [`TileStore::store_positions`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn build(
        &mut self,
        key: TileKey,
        parents: &[TileKey],
        below: &Held,
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        // Nothing acquires a slot between this load and the read below, so the correction slot
        // needn't be pinned.
        let correction = self.load_correction(key)?;
        let corrections = self
            .corrections
            .get(correction)
            .ok_or(Invariant::StaleHandle)?
            .view()
            .map_err(Invariant::CorrectionSlot)?;
        let source = Resident {
            mesh: &self.mesh,
            pool: &self.evaluated,
            tiles: below,
        };
        let prediction = predict_region(&self.mesh, &source, Region::of_tile(&key))?;
        let out = prefix_mut(&mut self.scratch, vertex_count(&key))?;
        prediction.apply_into(|i, j| corrections.get(i, j).unwrap_or(Vec3::ZERO), out)?;
        self.store_positions(key, parents)
    }

    /// Copies the positions in the scratch buffer into `key`'s slot (a new one, or its stale
    /// one in place), marks it fresh and records it as predicted from `parents`.
    ///
    /// The caller holds `parents` pinned: the tile is recorded before they can leave the pool,
    /// so their nodes, and their own parent edges, are still in the dependency graph (see
    /// [`crate::deps`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`] if no slot can be had.
    fn store_positions(
        &mut self,
        key: TileKey,
        parents: &[TileKey],
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        // A stale slot (pinned elsewhere, so still here) is overwritten in place: its handles
        // stay valid.
        let id = match self.evaluated.lookup(&key) {
            Some((id, _)) => id,
            None => self.acquire_evaluated(key)?,
        };
        let n = vertex_count(&key);
        let slot = self.evaluated.get_mut(id).ok_or(Invariant::StaleHandle)?;
        prefix_mut(&mut slot.positions, n)?.copy_from_slice(prefix(&self.scratch, n)?);
        self.evaluated.mark_fresh(id)?;
        self.deps.record_positions(key, parents);
        #[cfg(test)]
        self.log.positions.push(key);
        Ok(id)
    }

    /// The slot of `key`'s current positions and normals: cached, else the normals are computed
    /// from the positions of its [`normal_tiles`], made current together as one cone.
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
        let inputs = self.normal_inputs_of(key)?;
        let held = self.pin_current(inputs)?;
        let computed = self.normals_from(key, Region::of_tile(&key), &held);
        let released = self.unpin_held(&held);
        let id = computed?;
        released?;
        Ok(id)
    }

    /// Computes `region`'s normals from `held` (`key`'s normal inputs, pinned), writes them into
    /// `key`'s slot, which it returns, and records them while the inputs are still pinned (as
    /// [`TileStore::store_positions`] records positions).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn normals_from(
        &mut self,
        key: TileKey,
        region: Region,
        held: &Held,
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        let own = held
            .binary_search_by_key(&key, |&(k, _)| k)
            .ok()
            .and_then(|at| held.get(at))
            .map(|&(_, id)| id)
            .ok_or(Invariant::StaleHandle)?;
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
        let slot = self.evaluated.get_mut(own).ok_or(Invariant::StaleHandle)?;
        prefix_mut(&mut slot.normals, n)?.copy_from_slice(prefix(&self.scratch, n)?);
        self.deps.record_normals(key, held.iter().map(|(k, _)| k));
        #[cfg(test)]
        self.log.normals.push(key);
        Ok(own)
    }

    /// Unpins the evaluated tiles an evaluation pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale (pinned slots are never freed, so this
    /// doesn't happen); every tile is unpinned either way.
    fn unpin_held(&mut self, held: &Held) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &(_, id) in held {
            keep_first(&mut outcome, self.evaluated.unpin(id));
        }
        outcome
    }
}

/// A tile [`TileEdit`] has touched: loaded and pinned, perhaps written.
#[derive(Debug, Clone, Copy)]
struct Touched {
    /// The tile.
    key: TileKey,
    /// Its correction slot, pinned until the edit ends.
    id: SlotId<CorrectionSlot>,
    /// Whether a copy in it was written (it is dirty and on the feed).
    written: bool,
}

/// One tile's corrections being edited as one op: what [`TileStore::edit_tile`] hands its
/// closure.
///
/// Coordinates are the tile's own, `0..side` on each axis. Setting a vertex on the tile's border
/// sets every copy of it, in the neighbouring tiles of this face and of the faces across the
/// border too, so twins never disagree. The tiles written stay pinned until the edit ends; then
/// each one's evaluations are dropped, once.
pub struct TileEdit<'a> {
    /// The store, borrowed for the edit.
    store: &'a mut TileStore,
    /// The tile edited.
    key: TileKey,
    /// The op every write is recorded with.
    op: OpId,
    /// The tiles holding copies touched so far, the edited tile first. A handful (a border
    /// vertex's copies lie in the tile's neighbours), so a linear search finds them.
    touched: Vec<Touched>,
}

impl std::fmt::Debug for TileEdit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileEdit")
            .field("key", &self.key)
            .field("op", &self.op)
            .field("touched", &self.touched.len())
            .finish_non_exhaustive()
    }
}

impl TileEdit<'_> {
    /// The tile being edited.
    #[must_use]
    pub const fn key(&self) -> TileKey {
        self.key
    }

    /// The correction at `(i, j)` of the tile, as stored (interior values quantized).
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::OutOfTile`].
    pub fn get(&self, i: u32, j: u32) -> Result<Vec3, CacheError> {
        let side = self.key.side();
        let own = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get(own.id)
            .ok_or(Invariant::StaleHandle)?;
        let value = slot.view().map_err(Invariant::CorrectionSlot)?.get(i, j);
        value.ok_or(CacheError::Correction(CorrectionError::OutOfTile {
            i,
            j,
            side,
        }))
    }

    /// Sets the correction at `(i, j)` of the tile to `c`, in every copy of the vertex.
    ///
    /// An interior vertex has one copy and is written in place. A border vertex's copies are
    /// all loaded (and pinned) before any is written.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
    /// [`CorrectionError::OutOfTile`], before anything is written; [`CacheError::Store`],
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if a neighbouring tile holding a copy
    /// can't be loaded, also before anything of this vertex is written.
    pub fn set(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CacheError> {
        if !c.is_finite() {
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        let side = self.key.side();
        if i >= side || j >= side {
            return Err(CacheError::Correction(CorrectionError::OutOfTile {
                i,
                j,
                side,
            }));
        }
        let inside = 1..side - 1;
        if inside.contains(&i) && inside.contains(&j) {
            return self.write(0, (i, j), c);
        }
        let level = self.key.level();
        let region = Region::of_tile(&self.key);
        let copies = self.store.mesh.topology().copies(
            self.key.face(),
            level,
            region.x0() + i,
            region.y0() + j,
        );
        let mut at = [0; MAX_COPIES];
        for (copy, place) in copies.iter().zip(&mut at) {
            *place = self.touch(copy.key(level))?;
        }
        for (copy, &n) in copies.iter().zip(&at) {
            self.write(n, (copy.i, copy.j), c)?;
        }
        Ok(())
    }

    /// The index in `touched` of `key`'s tile, loaded and pinned if it is new.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]; [`CacheError::Internal`] for a stale handle.
    fn touch(&mut self, key: TileKey) -> Result<usize, CacheError> {
        if let Some(n) = self.touched.iter().position(|tile| tile.key == key) {
            return Ok(n);
        }
        let id = self.store.load_correction(key)?;
        self.store.corrections.pin(id)?;
        self.touched.push(Touched {
            key,
            id,
            written: false,
        });
        Ok(self.touched.len() - 1)
    }

    /// Writes `c` at `(i, j)` of touched tile `n`, marking the tile written at its first write.
    ///
    /// # Errors
    ///
    /// As [`TileStore::write_correction`].
    fn write(&mut self, n: usize, (i, j): (u32, u32), c: Vec3) -> Result<(), CacheError> {
        let tile = self.touched.get_mut(n).ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get_mut(tile.id)
            .ok_or(Invariant::StaleHandle)?;
        slot.view_mut()
            .and_then(|mut view| view.set(i, j, c))
            .map_err(Invariant::CorrectionSlot)?;
        if !tile.written {
            tile.written = true;
            self.store.mark_written(tile.key, tile.id, self.op)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
