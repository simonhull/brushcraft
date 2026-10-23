//! The tile store: a project held out of core.
//!
//! [`TileStore`] owns the base mesh, the working [`ProjectFile`], two budgeted pools (correction
//! tiles and evaluated tiles), the change feed and the op generator, and may borrow a
//! [`BaseFile`] under the working file. Correction tiles are loaded on demand (see below) and
//! edited in their slots; evaluated tiles are computed on demand from their parent tiles and
//! cached. Edits drop exactly the evaluations that followed from them (see [`crate::deps`]).
//!
//! # Layered reads
//!
//! A correction tile is read from the working file if it has a record there, else from the base
//! (the project file as of its last save), else it is all zero. The working file is the only one
//! the store writes: write-back, streaming (see [`crate::RecoveryWriter`]) and
//! [`TileStore::save`] append to it, and the base is read-only to the store. After the project file
//! is saved, [`TileStore::rebase`] swaps in the new save point as the base and a fresh working
//! file, keeping every resident tile.
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
//! evaluated the same way, from the cone of the tiles they read. An edit that moves the
//! positions a tile's normals read, but not the tile's own, leaves its normals stale rather than
//! dropping them, and only the border vertices that read a moved position are recomputed (see
//! [`crate::deps`]): after a brush step, that is all its neighbours need.
//!
//! The tiles of one cone level are independent of each other, so they are built in parallel,
//! on the store's worker threads ([`StoreConfig::threads`]), in rounds of at most [`ROUND`]
//! tiles: the owner thread loads and pins the round's correction tiles, the workers predict
//! every tile of the round from the pinned level below into scratch buffers, reading the pools
//! only through shared references, and the owner unpins the correction tiles and installs the
//! results one by one, in key order. Nothing the workers do touches the pools, the budget or the
//! dependency graph, and the rounds are cut the same way whatever the thread count, so the store
//! makes the same decisions (which slot, which victim) and computes the same bits with one
//! thread or thirty-two.
//!
//! [`TileStore::evaluate_many`] widens the cone to many target tiles at once, so the levels are
//! wide enough to keep every worker busy; [`TileStore::pin`] goes through it too. Such a cone's
//! levels are wider than [`geom::cone_width`]: a batch is sized to the room the budget has free
//! of pins, and halved, down to the one tile the bound above holds for, if it runs out.
//!
//! # Ops and writes
//!
//! The store issues the ops edits carry; callers never pass one in. [`TileStore::begin_op`]
//! starts a new op, and every edit from then on is part of it, so one stroke keeps one op
//! across all of its dabs, and ops only grow. Within the store, ops order nothing: every write
//! a correction slot takes is stamped from a counter (see [`crate::stamp`]), and the stamps
//! decide which slots a recovery flush may mark clean and which records its save point names.
//! A flush in the middle of a stroke so cuts cleanly between two dabs of one op.
//!
//! Everything here belongs to one thread: the one calling the store. Pins are held by
//! [`PinGuard`]s, which may be dropped anywhere on that thread; the store applies the unpins at
//! its next call. There are two exceptions. Prefetch: [`TileStore::prefetch`] hands record
//! locations to worker threads (see [`crate::prefetch`]), which read and decode on their own and
//! send the tiles back for [`TileStore::poll`] to install. And evaluation, whose workers only
//! compute, within a call, from inputs the owner thread has pinned (see above).
//!
//! # Budget
//!
//! The two pools share one byte budget, granted in whole chunks, and each is charged the same
//! [`StoreConfig::chunk_bytes`] per chunk, so a chunk one pool gives back is exactly the room the
//! other needs. [`TileStore`]'s memory policy orders what a pool that needs a slot takes: its own
//! free slots and new chunks first, then what is cheapest to lose. In short, a clean correction
//! tile is read again if it is wanted, an evaluated tile is recomputed from resident tiles without
//! any I/O, and a dirty correction tile costs a write now and a read later; chunks move between
//! the pools only within limits that keep each pool enough to work with.
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::convert::Infallible;
use std::io;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::sync::Arc;

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionView, EvalError, FaceId, LayerId, Level,
    MAX_COPIES, PositionSource, Region, TileKey, Vec3, normal_tiles, normals_into, parent_tiles,
    predict_region, refresh_normals_into,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use thiserror::Error;

use crate::deps::DepGraph;
use crate::feed::{ChangeFeed, ConsumerId};
use crate::file::{BaseFile, FileSyncer, FramedTile, Loc, ProjectFile, ReadScratch, StoreError};
use crate::meta::{Meta, Role, SavePoint};
use crate::op::{OpId, RESERVED_SITE};
use crate::pool::{Budget, Freshness, Invalidated, Pool, PoolError, SlotData, SlotId};
use crate::prefetch::{Fetched, Job, Prefetcher, Source};
use crate::slot::{CorrectionSlot, EvaluatedSlot, SIDE};
use crate::stamp::Stamp;

/// How a [`TileStore`] is set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreConfig {
    /// Bytes the two pools may hold together. The pools never exceed it. At least
    /// [`StoreConfig::minimum_budget`] for the store's mesh. How the pools share it is the
    /// store's memory policy (see [`TileStore`]): corrections take evaluated chunks only up to a
    /// quarter of it, and never below the evaluated pool's floor.
    pub budget_bytes: usize,
    /// Bytes per pool chunk, the unit both pools grow and shrink by and are charged in (whole
    /// slots fill it, the rest of a chunk is charged unused):
    /// [`StoreConfig::DEFAULT_CHUNK_BYTES`] in the app, smaller in tests.
    pub chunk_bytes: usize,
    /// The site (machine or session) the store's op ids carry; any but [`RESERVED_SITE`].
    pub site: u16,
    /// Worker threads that evaluate tiles in parallel (see [`TileStore::evaluate_many`]), or
    /// `None` for one per hardware thread ([`std::thread::available_parallelism`], one if that
    /// can't be told). Results are the same bits whatever the count; tests pin it to compare.
    pub threads: Option<NonZeroUsize>,
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

    /// The worker threads this configuration asks for: [`StoreConfig::threads`], else one per
    /// hardware thread, else one.
    fn worker_threads(&self) -> usize {
        self.threads
            .or_else(|| std::thread::available_parallelism().ok())
            .map_or(1, NonZeroUsize::get)
    }
}

/// Tiles of one cone level built in one parallel round at most (see the module notes).
///
/// The rounds are cut by this count alone, never by the thread count, so the store's pool
/// operations come in the same order however many workers it has. Each round pins its tiles'
/// correction slots while the workers read them (fewer, if the correction pool can't hold that
/// many besides its pins), and holds a scratch buffer of one tile per tile of the round.
const ROUND: usize = 64;

/// Target tiles [`TileStore::evaluate_many`] and [`TileStore::pin`] put in one batch at most.
///
/// One batch is one dependency cone: this is enough to keep every worker busy at every level of
/// the cone, and few enough that listing the cone stays cheap next to building it.
const MAX_BATCH: usize = 4096;

/// The share of the budget's free evaluated slots a batch's targets take: one in this many.
///
/// A batch's cone pins its targets and the level under them (about a quarter as many, plus a
/// border), and, for normals, the targets' neighbours too, so a quarter leaves room for all of
/// it. A batch that runs out of room all the same is halved and tried again.
const BATCH_SHARE: usize = 4;

/// What [`TileStore::evaluate_many`] makes current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evaluation {
    /// Evaluated positions, as [`TileStore::evaluate`] gives them.
    Positions,
    /// Evaluated positions and unit normals, as [`TileStore::normals`] gives them (normals are
    /// computed from positions, so these are made current too).
    Normals,
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

    /// The least the evaluated pool keeps when corrections take its chunks (see [`TileStore`]):
    /// the chunks that hold an evaluation's pins, and one more for the cache besides.
    fn evaluated_floor(self, chunk_bytes: usize) -> usize {
        let (per_chunk, charged) = Pool::<EvaluatedSlot>::layout(chunk_bytes);
        (self.evaluated.div_ceil(per_chunk as usize) + 1) * charged
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
    /// Writing the project file during a save failed: a tile, or the save point. The store is
    /// unchanged (its working file still holds every edit), and the save can be tried again.
    #[error("writing the project file during a save failed")]
    Save(#[source] StoreError),
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
    /// The working file and the base under it (or the store and a file given to a rebase or a
    /// save) are over different base meshes: their meta's mesh hashes differ.
    #[error("the working file and its base are over different meshes")]
    MeshMismatch,
    /// The working file isn't one that may be layered on the base given: a recovery file must
    /// name, in its meta, the project and the save point (offset and head op) of its base, and
    /// a file with no base must be a project file or a recovery file of a project never saved.
    /// Or a save was asked to write into a file that isn't a project file.
    #[error("the working file doesn't belong on this base")]
    WrongBase,
    /// [`TileStore::save`] was called while a recovery writer streams the working file, which
    /// writes its save points itself.
    #[error("the working file is streamed by a recovery writer, which writes its save points")]
    Streaming,
    /// A rebase was given a save that is out of date: the store was edited after the save was
    /// made, so the save doesn't hold every edit. Save again. Or a roll-over was given a new
    /// working file filled before the latest edit.
    #[error("the store was edited after the save it was to be rebased on")]
    StaleSave,
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
    /// The evaluation worker threads couldn't be started (see [`StoreConfig::threads`]).
    #[error("starting the evaluation worker threads failed")]
    Workers(#[source] io::Error),
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
    /// A rebase found a tile the store could read (in its working file or its base) that the
    /// save it was to be rebased on doesn't hold.
    #[error("{0:?} is not in the save a rebase was given")]
    NotInSave(TileKey),
    /// A save found a tile edited since the last rebase that is neither dirty nor in the
    /// working file: its edit is nowhere to copy from.
    #[error("edited tile {0:?} is neither dirty nor in the working file")]
    NothingToSave(TileKey),
    /// A project file had no save point right after a save wrote one.
    #[error("a project file had no save point after a save")]
    NotSaved,
    /// A correction tile was located in a base the store doesn't have.
    #[error("a tile was located in a missing base")]
    MissingBase,
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

/// What happens to a batch's targets once [`TileStore::in_batches`] has run it: what the
/// batches after it must leave room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Targets {
    /// They stay pinned ([`TileStore::pin`]): every target still to come is room taken.
    Pinned,
    /// They are unpinned, and stay only as a cache ([`TileStore::evaluate_many`]).
    Released,
}

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
    /// Tiles whose normals were computed, in order, in full or brought up to date.
    normals: Vec<TileKey>,
    /// Tiles whose stale normals were brought up to date (only the vertices reading a changed
    /// input recomputed), in order: also in `normals`.
    refreshed: Vec<TileKey>,
    /// Correction tiles the owner thread looked up in its files, in order (prefetch
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

#[cfg(test)]
impl EvalLog {
    /// Every tile whose positions (`false`) were computed, in order, then every tile whose
    /// normals (`true`) were: for comparing what two stores did.
    fn computed(&self) -> impl Iterator<Item = (TileKey, bool)> + '_ {
        let positions = self.positions.iter().map(|&key| (key, false));
        positions.chain(self.normals.iter().map(|&key| (key, true)))
    }
}

/// A project held out of core: the base mesh, its files, budgeted pools of correction and
/// evaluated tiles, the change feed and the op generator.
///
/// The files are the working file and, under it, a base: the project file as of its last save.
/// Corrections are read from the working file first, then the base, and an absent tile is zero
/// (see the module notes). The store writes only the working file.
///
/// Tiles are named by base-layer displacement [`TileKey`]s of level 1 or more; an evaluated tile
/// is named by the displacement tile at the same position. Level 0 is the base mesh, never a
/// tile.
///
/// Dropping the store doesn't write anything: edits in dirty correction slots (since the last
/// [`TileStore::save`] or recovery flush, and not written back on eviction) are lost with it. A
/// [`crate::RecoveryWriter`] streams them into the working file as the user works; without one,
/// call [`TileStore::save`] before dropping a store whose edits matter.
///
/// # Memory policy
///
/// The pools share [`StoreConfig::budget_bytes`] and grow by whole chunks of
/// [`StoreConfig::chunk_bytes`] while the budget allows another; once it is spent, memory moves
/// between them only a chunk at a time. A correction tile that needs a slot (to be read, edited,
/// or evaluated from) takes the first of:
///
/// 1. a free correction slot, or a new chunk if the budget allows one;
/// 2. the slot of a clean correction tile, the clock's pick: it is read again if it is wanted;
/// 3. an evaluated chunk taken back, while the correction pool holds less than a quarter of the
///    budget and the evaluated pool keeps its floor without that chunk (the chunks one evaluation
///    pins, see [`StoreConfig::minimum_budget`], plus one for the cache). Of the evaluated chunks
///    without a pin, the one with the fewest tiles in use goes (current, or referenced since the
///    clock last passed); its tiles are recomputed when next wanted;
/// 4. the slot of one dirty correction tile, written back to the working file first: one write per
///    slot taken, never a batch;
/// 5. only when every correction slot is pinned, chunks taken back from the pools (below).
///
/// The quarter and the floor keep dirty corrections from crowding out the evaluated cache that
/// every evaluation reads; past them, dirty tiles are written back one at a time.
///
/// A prefetched tile takes only a free slot, a new chunk or a clean tile's slot, and is dropped
/// otherwise: prefetch never takes an evaluated chunk or writes a tile back. An evaluated tile
/// takes a free slot, a new chunk or the clock's victim (evaluated tiles are never dirty), and
/// only when every evaluated slot is pinned, chunks taken back from the pools.
///
/// Chunks are taken back (for a pool whose slots are all pinned, and by
/// [`TileStore::shrink_to`]) evaluated chunks first, then correction chunks, the cheapest of
/// each pool first: the correction chunk with the fewest dirty tiles (each written back before
/// the chunk goes), the evaluated chunk with the fewest tiles in use. A chunk holding a pinned
/// slot stays.
pub struct TileStore {
    /// The prefetch workers. Declared first, so they are stopped and joined (when the store is
    /// dropped) before anything else goes, the working file included.
    prefetch: Prefetcher,
    /// The base mesh: level 0 and the topology. The working file's, and its base's.
    mesh: Arc<BaseMesh>,
    /// The mesh hash every file the store reads must carry in its meta.
    mesh_hash: u128,
    /// The working file: correction tiles are read from it first, and written back to it.
    file: ProjectFile,
    /// The project file as of its last save, read where the working file has no record; `None`
    /// for a project never saved.
    base: Option<BaseFile>,
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
    /// The latest write of each dirty correction slot: the op its record is written with, by
    /// write-back or [`TileStore::save`], and its stamp.
    unsaved: HashMap<TileKey, Dirty>,
    /// Every tile edited since the last rebase (or since the store opened), drained from the
    /// feed by save's consumer: what the next save copies into the project file.
    since_rebase: BTreeSet<TileKey>,
    /// The stamp of the latest write a correction slot took (see [`crate::stamp`]): where a
    /// recovery flush cuts ([`TileStore::open_cut`]); a save records it, and a rebase refuses a
    /// save made before a later write.
    last_write: Stamp,
    /// The op of the record each clean resident correction slot holds, where it has one:
    /// recorded when a slot is filled from a record, or cleaned by writing one. A clean slot
    /// whose record can't be read back any more is written again labelled with this op (see
    /// [`TileStore::fill_roll_over`]), never with one that didn't edit it. Entries of tiles
    /// no longer resident are never read, and go at the next rebase or roll-over.
    known_ops: HashMap<TileKey, OpId>,
    /// Counts the working files the store has had: a rebase moves it on, so a recovery flush
    /// started before can tell it would finish in the wrong file.
    generation: u64,
    /// The change feed: every key an edit changed, with its op.
    feed: ChangeFeed,
    /// Save's place in the feed.
    save_consumer: ConsumerId,
    /// The flush mark of each streaming consumer (the recovery writer): the keys it has drained
    /// from the feed whose latest edit it hasn't yet seen reach the working file. A key leaves
    /// the mark only once the store finds the tile in the working file at its latest edit (see
    /// [`TileStore::snapshot_dirty`]); until then every snapshot copies it again, so a flush
    /// that fails part-way loses nothing. Separate from the slots' dirty flag, which every
    /// writer of the working file clears, and from save's place in the feed.
    streams: HashMap<ConsumerId, BTreeSet<TileKey>>,
    /// The next op [`TileStore::begin_op`] hands out; `None` once the site's ids are used up.
    next_op: Option<OpId>,
    /// The op edits are made as: the one [`TileStore::begin_op`] last started, or `None` if
    /// there is none (the store just opened, or was rebased), and the next edit starts one.
    op: Option<OpId>,
    /// Shared with the [`PinGuard`]s: the pin sets they have released.
    pins: Rc<RefCell<PinTable>>,
    /// The slots of each live pin set.
    pin_sets: HashMap<u64, Vec<Pinned>>,
    /// The id the next pin set gets.
    next_pin_set: u64,
    /// One tile of positions or normals per tile of a round (see [`ROUND`]): the workers
    /// write here, then the owner thread copies each into its slot. Grown on demand, never past
    /// [`ROUND`] buffers.
    scratch: Vec<Vec<Vec3>>,
    /// The evaluation workers (see the module notes).
    workers: rayon::ThreadPool,
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
            .field("base", &self.base.as_ref().map(|base| base.keys().count()))
            .field("budget", &self.budget)
            .field("corrections", &self.corrections)
            .field("evaluated", &self.evaluated)
            .field("unsaved", &self.unsaved.len())
            .field("streams", &self.streams.len())
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

/// `f` of every item of `items`, in order: on `workers` when there are several, on the calling
/// thread for one (a single item gains nothing from a hand-off).
fn map_parallel<T: Sync, R: Send>(
    workers: &rayon::ThreadPool,
    items: &[T],
    f: impl Fn(&T) -> R + Sync + Send,
) -> Vec<R> {
    if items.len() <= 1 {
        return items.iter().map(f).collect();
    }
    workers.install(|| items.par_iter().map(f).collect())
}

/// Runs `f` on every item of `items` with a buffer of its own, as [`map_parallel`] does, and
/// returns the first error in item order.
///
/// Item `k` gets `buffers[k]`; there must be as many buffers as items. The error returned
/// doesn't depend on which worker finished first.
///
/// # Errors
///
/// The first item's error, in order; [`CacheError::Internal`] if there are fewer buffers than
/// items.
fn each_parallel<T: Sync>(
    workers: &rayon::ThreadPool,
    items: &[T],
    buffers: &mut [Vec<Vec3>],
    f: impl Fn(&T, &mut Vec<Vec3>) -> Result<(), CacheError> + Sync + Send,
) -> Result<(), CacheError> {
    let got = buffers.len();
    let buffers = buffers
        .get_mut(..items.len())
        .ok_or(Invariant::ShortBuffer {
            expected: items.len(),
            got,
        })?;
    let results: Vec<Result<(), CacheError>> = if items.len() <= 1 {
        items.iter().zip(buffers).map(|(t, b)| f(t, b)).collect()
    } else {
        workers.install(|| {
            items
                .par_iter()
                .zip(buffers.par_iter_mut())
                .map(|(t, b)| f(t, b))
                .collect()
        })
    };
    results.into_iter().collect()
}

/// The slot of `key` among the evaluated tiles an evaluation pinned.
///
/// # Errors
///
/// [`Invariant::StaleHandle`] if `key` isn't among them (the store asks only for tiles it
/// pinned).
fn held_slot(held: &Held, key: &TileKey) -> Result<SlotId<EvaluatedSlot>, Invariant> {
    held.binary_search_by_key(key, |&(k, _)| k)
        .ok()
        .and_then(|at| held.get(at))
        .map(|&(_, id)| id)
        .ok_or(Invariant::StaleHandle)
}

/// The tiles `key`'s positions are predicted from, listed with [`parent_tiles`], level 0 (the
/// mesh) left out, in key order.
///
/// # Errors
///
/// [`CacheError::Eval`] (not for a valid key).
fn list_parents(mesh: &BaseMesh, key: &TileKey) -> Result<Vec<TileKey>, CacheError> {
    let parents = parent_tiles(mesh, Region::of_tile(key))?;
    Ok(parents
        .into_iter()
        .filter(|parent| parent.level() != Level::ZERO)
        .collect())
}

/// The tiles `key`'s normals are computed from, listed with [`normal_tiles`], the tile itself
/// included.
///
/// # Errors
///
/// [`CacheError::Eval`] (not for a valid key).
fn list_normal_inputs(mesh: &BaseMesh, key: &TileKey) -> Result<BTreeSet<TileKey>, CacheError> {
    let mut inputs = normal_tiles(mesh, Region::of_tile(key))?;
    // The tile's own block is among them; inserting it makes that plain.
    inputs.insert(*key);
    Ok(inputs)
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

/// The latest write of a dirty correction slot: what its record is written with, and where that
/// write stands among the store's writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dirty {
    /// The op of the edit that made the write: the op its record carries.
    pub(crate) op: OpId,
    /// The write's stamp: a flush's copy of the slot leaves it clean only if no write came
    /// after (see [`TileStore::append_streamed`]).
    pub(crate) stamp: Stamp,
}

/// Appends a dirty correction slot's tile to the working file, with the op of its latest write
/// (no save point), and returns that op.
///
/// The write is forgotten: the slot is clean once this returns.
///
/// # Errors
///
/// [`CacheError::WriteBack`] if the file's write fails; [`CacheError::Internal`] if the slot has
/// no recorded write (every slot marked dirty gets one first) or doesn't hold a valid tile.
fn write_slot(
    file: &mut ProjectFile,
    unsaved: &mut HashMap<TileKey, Dirty>,
    key: TileKey,
    slot: &CorrectionSlot,
) -> Result<OpId, CacheError> {
    let Dirty { op, stamp } = unsaved
        .get(&key)
        .copied()
        .ok_or(Invariant::MissingOp(key))?;
    let tile = slot.to_tile().map_err(Invariant::CorrectionSlot)?;
    file.write_displacement_stamped(key, op, stamp, &tile)
        .map_err(CacheError::WriteBack)?;
    unsaved.remove(&key);
    Ok(op)
}

/// The first op the store hands out for `site` after the files' `heads`: the next after the
/// greatest head of `site`, or sequence 0 of `site` if none is of it.
///
/// # Errors
///
/// [`CacheError::OpsExhausted`] if that head is the site's last op id.
fn first_op(site: u16, heads: impl IntoIterator<Item = Option<OpId>>) -> Result<OpId, CacheError> {
    let latest = heads
        .into_iter()
        .flatten()
        .filter(|head| head.site() == site)
        .max();
    match latest {
        Some(head) => head.next(),
        None => OpId::new(site, 0),
    }
    .ok_or(CacheError::OpsExhausted)
}

/// A copy of a dirty correction slot, taken for a streaming consumer by
/// [`TileStore::snapshot_dirty`]: what the recovery writer encodes off the owner thread.
#[derive(Debug)]
pub(crate) struct Snapshot {
    /// The tile.
    pub(crate) key: TileKey,
    /// The op of the tile's latest write when the copy was taken: the record's op.
    pub(crate) op: OpId,
    /// The stamp of that write: what places the record against the flush's cut, and what
    /// [`TileStore::append_streamed`] compares against to tell whether the slot was written
    /// since.
    pub(crate) stamp: Stamp,
    /// The slot's contents, boxed so the hand-off moves a pointer, not 26.9 KB.
    pub(crate) slot: Box<CorrectionSlot>,
}

/// Proof that a save copied every tile edited since the last rebase into a project file, made
/// only by [`TileStore::save_into`] and taken by [`TileStore::rebase`].
///
/// It holds the project file's new save point, as the base to rebase on, the store's latest
/// write when the save was made (a rebase refuses it if the store has been written since, as
/// those writes are in no save), and the tiles the save found lost: edited, but readable
/// nowhere. There is no other way to make one, so a store can't be rebased past edits that
/// were never saved, except those it has listed as lost.
#[derive(Debug)]
pub(crate) struct Saved {
    /// The project file as of the save point the save wrote.
    base: BaseFile,
    /// The stamp of the store's latest write when the save was made.
    stamp: Stamp,
    /// The tiles edited since the last rebase whose latest edit couldn't be read (see
    /// [`TileStore::save_into`]): the save point holds their last saved state.
    lost: BTreeSet<TileKey>,
}

impl Saved {
    /// The save point the save wrote: what the fresh working file to rebase on must name.
    pub(crate) const fn save_point(&self) -> SavePoint {
        self.base.save_point()
    }

    /// The tiles whose latest edit the save couldn't read, and left out.
    pub(crate) const fn lost(&self) -> &BTreeSet<TileKey> {
        &self.lost
    }
}

/// A new working file [`TileStore::fill_roll_over`] has filled with every edit and saved,
/// waiting for [`TileStore::roll_over`] to swap it in.
#[derive(Debug)]
pub(crate) struct Filled {
    /// The new working file.
    file: ProjectFile,
    /// The tiles that couldn't be carried across.
    lost: Vec<TileKey>,
    /// The dirty slots written to it, clean once it is swapped in.
    cleaned: Vec<(TileKey, SlotId<CorrectionSlot>)>,
    /// The store's latest write when it was filled: a roll-over refuses it after a later one.
    stamp: Stamp,
}

/// `project` as of the save point a save has just written to it.
///
/// # Errors
///
/// [`CacheError::Save`] if the save point can't be read; [`CacheError::Internal`] with
/// [`Invariant::NotSaved`] if `project` has none, which a save that went through rules out.
fn saved_base(project: &ProjectFile) -> Result<BaseFile, CacheError> {
    project
        .base()
        .map_err(CacheError::Save)?
        .ok_or_else(|| Invariant::NotSaved.into())
}

/// Checks that a working file with `working`'s meta may be layered on `base`.
///
/// Over a base, the working file must be a recovery file of the base's project naming the
/// base's save point exactly: its project, its offset and its head op. A recovery file of an
/// earlier or later save, or of another project, would be read through the wrong saved state.
/// The base itself must be a project file. With no base, the working file must be a project
/// file, or a recovery file of a project never saved; a recovery file that names a save point
/// would be missing the state it extends.
///
/// # Errors
///
/// [`CacheError::WrongBase`].
fn check_layers(working: &Meta, base: Option<&BaseFile>) -> Result<(), CacheError> {
    let fits = match base {
        None => matches!(
            working.role(),
            Role::Project | Role::Recovery { base: None }
        ),
        Some(base) => {
            let point = base.save_point();
            base.meta().role() == Role::Project
                && working.project() == point.project
                && working.role() == Role::Recovery { base: Some(point) }
        }
    };
    if fits {
        Ok(())
    } else {
        Err(CacheError::WrongBase)
    }
}

impl TileStore {
    /// A store over `working`, the file it writes, with no base under it: a project file read
    /// and written directly, or the recovery file of a project never saved, which holds all of
    /// its work. The base mesh is the working file's.
    ///
    /// A project with a save is opened over its save point through [`crate::Project`], which
    /// layers a recovery file on it.
    ///
    /// The op generator starts after the working file's newest head op of `config.site`, else
    /// at sequence 0 of the site, so the store's edits are newer than everything it reads.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::WrongBase`] if `working` is a recovery file
    /// that names a save point (it needs that save point under it);
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`];
    /// [`CacheError::OpsExhausted`] if the file's head is the site's last op id;
    /// [`CacheError::Store`] if the read buffers can't be made; [`CacheError::Workers`] if the
    /// evaluation workers can't be started.
    pub fn open(working: ProjectFile, config: StoreConfig) -> Result<Self, CacheError> {
        Self::open_layered(working, None, config)
    }

    /// A store over `working`, the file it writes, and `base`, the project file as of its last
    /// save, read where `working` has no record (`None` for a project never saved). The base
    /// mesh is the working file's.
    ///
    /// The files must belong together (see [`check_layers`]): over a base, `working` is a
    /// recovery file whose meta names the base's project and save point, offset and head op.
    ///
    /// The op generator starts after the newest head op of `config.site` in either file (all of
    /// the base's file, records after its save point included), else at sequence 0 of the site,
    /// so the store's edits are newer than everything it reads or might copy.
    ///
    /// The tiles a recovery file already holds (one restored after a crash) count as edits no
    /// save has made part of the project file: the next save copies them.
    ///
    /// # Errors
    ///
    /// [`CacheError::ReservedSite`]; [`CacheError::MeshMismatch`] if the base's mesh hash isn't
    /// the working file's; [`CacheError::WrongBase`] if the files don't belong together;
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`];
    /// [`CacheError::OpsExhausted`] if a file's head is the site's last op id;
    /// [`CacheError::Store`] if the read buffers can't be made; [`CacheError::Workers`] if the
    /// evaluation workers can't be started.
    pub(crate) fn open_layered(
        working: ProjectFile,
        base: Option<BaseFile>,
        config: StoreConfig,
    ) -> Result<Self, CacheError> {
        if config.site == RESERVED_SITE {
            return Err(CacheError::ReservedSite);
        }
        let mesh_hash = working.meta().mesh_hash();
        if base
            .as_ref()
            .is_some_and(|base| base.meta().mesh_hash() != mesh_hash)
        {
            return Err(CacheError::MeshMismatch);
        }
        check_layers(working.meta(), base.as_ref())?;
        let mesh = working.mesh();
        let headroom = Headroom::of(&mesh);
        let minimum = headroom.minimum_budget(config.chunk_bytes);
        if config.budget_bytes < minimum {
            return Err(CacheError::BudgetTooSmall { minimum });
        }
        let next_op = first_op(
            config.site,
            [working.head(), base.as_ref().and_then(BaseFile::head)],
        )?;
        // A recovery file holds only edits no save has: a restored one's are still to save.
        let since_rebase = match working.meta().role() {
            Role::Recovery { .. } => working.keys().copied().collect(),
            Role::Project => BTreeSet::new(),
        };
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(config.worker_threads())
            .thread_name(|n| format!("tile-eval-{n}"))
            .build()
            .map_err(|e| CacheError::Workers(io::Error::other(e)))?;
        let mut feed = ChangeFeed::new();
        let save_consumer = feed.subscribe();
        Ok(Self {
            prefetch: Prefetcher::new(),
            mesh,
            mesh_hash,
            file: working,
            base,
            budget: Budget::new(config.budget_bytes),
            headroom,
            corrections: Pool::new(config.chunk_bytes),
            evaluated: Pool::with_eviction_log(config.chunk_bytes),
            deps: DepGraph::new(),
            unsaved: HashMap::new(),
            since_rebase,
            last_write: Stamp::ZERO,
            known_ops: HashMap::new(),
            generation: 0,
            feed,
            save_consumer,
            streams: HashMap::new(),
            next_op: Some(next_op),
            op: None,
            pins: Rc::default(),
            pin_sets: HashMap::new(),
            next_pin_set: 0,
            scratch: Vec::new(),
            workers,
            read_scratch: ReadScratch::new()?,
            #[cfg(test)]
            log: EvalLog::default(),
            #[cfg(test)]
            unreadable: None,
        })
    }

    /// Copies every tile edited since the last rebase (or since the store opened) into
    /// `project`, saves it, and returns the proof a rebase needs: the new save point as a base,
    /// and the store's latest write.
    ///
    /// Each tile is copied from where its latest edit is: a dirty slot is encoded and written
    /// with the op of its latest write, else the working file's record is copied as it is,
    /// keeping its original op, else, if that record no longer reads, a clean resident slot
    /// (which holds what the record did) is written with the record's op as the store last knew
    /// it. A tile whose latest edit none of these gives is lost, and listed in the proof
    /// ([`Saved::lost`]), so one lost tile never blocks the saves after it: a broken dirty slot
    /// is saved as the working file's record, if that reads, and a tile neither way left out,
    /// at its last saved state. Save's
    /// consumer of the change feed is drained first; the tiles it lists stay listed until a
    /// rebase, so a save that fails, or one followed by more edits, is simply made again.
    /// Nothing in the store changes: dirty slots stay dirty until the rebase, which is what
    /// makes the save's copies the saved state.
    ///
    /// A project being saved under a new name has its old save point copied in first (see
    /// [`ProjectFile::copy_base`]); this adds the edits.
    ///
    /// # Errors
    ///
    /// [`CacheError::WrongBase`] if `project` isn't a project file; [`CacheError::MeshMismatch`]
    /// if it is over another mesh; [`CacheError::Save`] if a write to it or its save point
    /// fails; [`CacheError::Internal`] for an edited tile that has no record in the working file
    /// and isn't dirty ([`Invariant::NothingToSave`]: every edit is in one or the other), or a
    /// tile of the recovery file that wasn't copied ([`Invariant::NotInSave`], checked before
    /// the save point is written).
    pub(crate) fn save_into(&mut self, project: &mut ProjectFile) -> Result<Saved, CacheError> {
        self.sync_pins()?;
        if project.meta().role() != Role::Project {
            return Err(CacheError::WrongBase);
        }
        if project.meta().mesh_hash() != self.mesh_hash {
            return Err(CacheError::MeshMismatch);
        }
        self.drain_saves();
        let mut lost = BTreeSet::new();
        for key in &self.since_rebase {
            let resident = self.corrections.find(key);
            let tile = |id| {
                self.corrections
                    .get(id)
                    .and_then(|slot| slot.to_tile().ok())
            };
            let dirty = resident.filter(|&id| self.corrections.is_dirty(id) == Some(true));
            // The dirty slot: the latest edit.
            if let Some(id) = dirty {
                let op = self.unsaved.get(key).map(|write| write.op);
                if let (Some(op), Some(tile)) = (op, tile(id)) {
                    project
                        .write_displacement(*key, op, &tile)
                        .map_err(CacheError::Save)?;
                    continue;
                }
                // Broken: the latest edit is lost, and the record is the best left of it.
                lost.insert(*key);
            }
            // The working file's record.
            match self.file.read_tile_record(key) {
                Ok(Some(raw)) => {
                    project.append_copy(key, &raw).map_err(CacheError::Save)?;
                    continue;
                }
                Ok(None) if dirty.is_none() => {
                    return Err(Invariant::NothingToSave(*key).into());
                }
                Ok(None) => continue,
                Err(_) => {}
            }
            // The record is unreadable, but a clean resident slot holds what it did.
            let clean = resident.filter(|_| dirty.is_none());
            let op = clean.and_then(|_| self.known_ops.get(key).copied());
            match (op, clean.and_then(tile)) {
                (Some(op), Some(tile)) => project
                    .write_displacement(*key, op, &tile)
                    .map_err(CacheError::Save)?,
                _ => {
                    lost.insert(*key);
                }
            }
        }
        // Every tile of a recovery file is an edit since the last save point, so it must have
        // been copied: a save point without one would lose it once the store is rebased.
        let recovery = matches!(self.file.meta().role(), Role::Recovery { .. });
        if let Some(missing) = self
            .file
            .keys()
            .find(|key| recovery && !self.since_rebase.contains(key))
        {
            return Err(Invariant::NotInSave(*missing).into());
        }
        project.save().map_err(CacheError::Save)?;
        let base = saved_base(project)?;
        Ok(Saved {
            base,
            stamp: self.last_write,
            lost,
        })
    }

    /// Swaps in the save point `saved` proves, as the base, and `working`, a fresh working file
    /// over it; returns the working file it replaces.
    ///
    /// `saved` must be current: made by [`TileStore::save_into`] with no write since. Then every
    /// tile edited since the last rebase is in the new save point, and the old working file's
    /// records are no longer needed. That is checked, not assumed: every tile of the old
    /// working file and of the old base must be in the new save point, but for those the save
    /// listed as lost ([`Saved::lost`]), whose last saved state the store reads from now on (a
    /// lost tile's resident slot is dropped, if no pin holds it). Dirty slots are clean
    /// after the rebase (the save wrote them), and every other resident correction tile is kept,
    /// as it is what was just saved; so are cached evaluations, which stay valid: nothing they
    /// read has changed. Prefetches in flight are dropped (their workers stop and are joined),
    /// and later ones read the new files. The op generator continues, past the base's head if
    /// that is newer, and the current op ends: the next edit starts a new one, so the edits
    /// after a save point are newer than its head.
    ///
    /// # Errors
    ///
    /// [`CacheError::StaleSave`] if the store was written after `saved` was made;
    /// [`CacheError::MeshMismatch`] if either file isn't over the store's mesh;
    /// [`CacheError::WrongBase`] if `working` isn't a recovery file naming `saved`'s save point;
    /// [`CacheError::Internal`] with [`Invariant::NotInSave`] if a tile the store could read is
    /// missing from the save (a bug in the save); [`CacheError::OpsExhausted`] if the base's
    /// head is the site's last op id. The store is unchanged then, and the files are dropped.
    pub(crate) fn rebase(
        &mut self,
        saved: Saved,
        working: ProjectFile,
    ) -> Result<ProjectFile, CacheError> {
        self.sync_pins()?;
        let Saved { base, stamp, lost } = saved;
        if stamp != self.last_write {
            return Err(CacheError::StaleSave);
        }
        if base.meta().mesh_hash() != self.mesh_hash || working.meta().mesh_hash() != self.mesh_hash
        {
            return Err(CacheError::MeshMismatch);
        }
        check_layers(working.meta(), Some(&base))?;
        let old_base = self.base.iter().flat_map(BaseFile::keys);
        if let Some(missing) = self
            .file
            .keys()
            .chain(old_base)
            .chain(&self.since_rebase)
            .find(|key| !base.contains(key) && !lost.contains(key))
        {
            return Err(Invariant::NotInSave(*missing).into());
        }
        let next_op = match self.next_op {
            Some(op) => {
                let after = first_op(op.site(), [working.head(), base.head()])?;
                Some(op.max(after))
            }
            None => None,
        };
        // The save wrote every dirty slot as it is now (no edit since), so none is unsaved.
        let dirty: Vec<(TileKey, Dirty)> = self.unsaved.drain().collect();
        for (key, write) in dirty {
            if let Some(id) = self.corrections.find(&key) {
                self.corrections.set_dirty(id, false)?;
                self.known_ops.insert(key, write.op);
            }
        }
        for key in &lost {
            // A lost tile's slot doesn't hold the saved state, and has nothing to write: it is
            // dropped, or, if a pin holds it, kept clean until the pin goes.
            if let Some(id) = self.corrections.find(key) {
                self.corrections.set_dirty(id, false)?;
                if !self.corrections.is_pinned(key) {
                    self.corrections.remove(key)?;
                }
            }
        }
        self.prune_known_ops();
        self.since_rebase.clear();
        self.generation += 1;
        // The old workers read the old files; dropping them joins them and drops their results.
        self.prefetch = Prefetcher::new();
        self.next_op = next_op;
        self.op = None;
        self.base = Some(base);
        Ok(std::mem::replace(&mut self.file, working))
    }

    /// Fills `fresh`, a new working file on the same base, with every edit, after the working
    /// file failed to sync: the first half of a roll-over, which changes nothing in the store.
    /// [`TileStore::roll_over`] swaps it in, once the caller has given it its name.
    ///
    /// Every tile of the old working file and every dirty slot is written to `fresh`, read
    /// through the store: a dirty slot as it is, with the op of its latest write; any other
    /// tile as the old file's record, copied after its checksum is checked. A clean resident
    /// tile whose record no longer reads is written from its slot, which holds what the record
    /// did, labelled with the record's op as the store last knew it (see
    /// [`TileStore::known_ops`]): never with an op that didn't edit it. A tile that can be read
    /// neither way, or a broken slot, is lost: listed, and left out. Then `fresh` is saved
    /// (synced, a save point, synced again).
    ///
    /// # Errors
    ///
    /// [`CacheError::MeshMismatch`] or [`CacheError::WrongBase`] if `fresh` isn't over the
    /// store's mesh, or doesn't name the same base as the working file; [`CacheError::WriteBack`]
    /// if writing or saving `fresh` fails.
    pub(crate) fn fill_roll_over(&mut self, mut fresh: ProjectFile) -> Result<Filled, CacheError> {
        self.sync_pins()?;
        if fresh.meta().mesh_hash() != self.mesh_hash {
            return Err(CacheError::MeshMismatch);
        }
        if fresh.meta().role() != self.file.meta().role()
            || fresh.meta().project() != self.file.meta().project()
        {
            return Err(CacheError::WrongBase);
        }
        check_layers(fresh.meta(), self.base.as_ref())?;
        let keys: BTreeSet<TileKey> = self
            .file
            .keys()
            .copied()
            .chain(self.unsaved.keys().copied())
            .collect();
        let mut lost = Vec::new();
        let mut cleaned = Vec::new();
        for key in keys {
            let resident = self.corrections.find(&key);
            let dirty = resident.filter(|&id| self.corrections.is_dirty(id) == Some(true));
            let from_slot = match dirty {
                Some(_) => self.unsaved.get(&key).map(|write| write.op),
                None => match self.file.read_tile_record(&key) {
                    Ok(Some(raw)) => {
                        fresh
                            .append_copy(&key, &raw)
                            .map_err(CacheError::WriteBack)?;
                        continue;
                    }
                    // Unreadable (or, impossibly, gone): only the slot holds it now.
                    Ok(None) | Err(_) if resident.is_some() => self.known_ops.get(&key).copied(),
                    Ok(None) | Err(_) => None,
                },
            };
            let tile = resident
                .and_then(|id| self.corrections.get(id))
                .and_then(|slot| slot.to_tile().ok());
            let (Some(op), Some(tile)) = (from_slot, tile) else {
                lost.push(key);
                continue;
            };
            fresh
                .write_displacement(key, op, &tile)
                .map_err(CacheError::WriteBack)?;
            if let Some(id) = dirty {
                cleaned.push((key, id));
            }
        }
        fresh.save().map_err(CacheError::WriteBack)?;
        Ok(Filled {
            file: fresh,
            lost,
            cleaned,
            stamp: self.last_write,
        })
    }

    /// Swaps in the working file [`TileStore::fill_roll_over`] filled; returns the old working
    /// file and the tiles that couldn't be carried across (lost).
    ///
    /// Every slot written to the new file is clean, streaming consumers' marks are emptied (all
    /// they held is in it), and the generation moves on, so a flush started before is
    /// abandoned. The base is unchanged, and so are the tiles to save into the project, but for
    /// the lost ones: they are in no file the store writes, so the next save leaves them at
    /// their last saved state instead of failing to find them.
    ///
    /// # Errors
    ///
    /// [`CacheError::StaleSave`] if the store was written since `filled` was: the new file
    /// doesn't hold those writes. The store is unchanged then, and the file is dropped.
    /// [`CacheError::Internal`] for a stale handle.
    pub(crate) fn roll_over(
        &mut self,
        filled: Filled,
    ) -> Result<(ProjectFile, Vec<TileKey>), CacheError> {
        let Filled {
            file,
            lost,
            cleaned,
            stamp,
        } = filled;
        if stamp != self.last_write {
            return Err(CacheError::StaleSave);
        }
        // Drained first, so the feed can't bring a lost key back.
        self.drain_saves();
        for key in &lost {
            self.since_rebase.remove(key);
        }
        for (key, id) in cleaned {
            self.corrections.set_dirty(id, false)?;
            if let Some(write) = self.unsaved.remove(&key) {
                self.known_ops.insert(key, write.op);
            }
        }
        self.prune_known_ops();
        let marks: Vec<ConsumerId> = self.streams.keys().copied().collect();
        for id in marks {
            if let Some(mark) = self.streams.get_mut(&id) {
                mark.clear();
            }
        }
        self.generation += 1;
        // The workers' jobs name records of the old file.
        self.prefetch = Prefetcher::new();
        Ok((std::mem::replace(&mut self.file, file), lost))
    }

    /// Moves every key save's consumer hasn't drained from the change feed into
    /// [`TileStore::since_rebase`].
    fn drain_saves(&mut self) {
        let drained = self.feed.drain(self.save_consumer);
        self.since_rebase
            .extend(drained.into_iter().map(|(key, _)| key));
    }

    /// Drops the [`TileStore::known_ops`] of tiles no longer resident.
    fn prune_known_ops(&mut self) {
        let corrections = &self.corrections;
        self.known_ops
            .retain(|key, _| corrections.find(key).is_some());
    }

    /// Whether the store holds edits no save has made part of a save point of the project
    /// file: any edit since the last rebase, or since it opened.
    pub(crate) fn has_unsaved_edits(&self) -> bool {
        !self.since_rebase.is_empty() || self.feed.has_pending(self.save_consumer)
    }

    /// The base mesh the store's tiles subdivide: level 0 and the topology.
    #[must_use]
    pub fn mesh(&self) -> &BaseMesh {
        &self.mesh
    }

    /// The working file the store reads first and writes.
    pub(crate) const fn working(&self) -> &ProjectFile {
        &self.file
    }

    /// The base under the working file, if the project has been saved.
    pub(crate) const fn base(&self) -> Option<&BaseFile> {
        self.base.as_ref()
    }

    /// Starts a new op, the site's next sequence number, and returns it: every edit from now
    /// on, [`TileStore::set_correction`] and [`TileStore::edit_tile`] alike, is part of it,
    /// until the next call. A stroke begins one and keeps it across all of its dabs.
    ///
    /// Ops are issued here only, never passed in by a caller, so they only grow. An edit with
    /// no op begun (the first after the store opens, or after a rebase) begins one itself.
    ///
    /// There is no way to make an edit as an op of the caller's choosing:
    ///
    /// ```compile_fail
    /// # fn edit(store: &mut store::TileStore, key: geom::TileKey, op: store::OpId) {
    /// store.edit_tile(key, op, |edit| edit.set(1, 1, geom::Vec3::ZERO));
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`CacheError::OpsExhausted`] past the site's last sequence number; the current op is
    /// kept then.
    pub fn begin_op(&mut self) -> Result<OpId, CacheError> {
        let op = self.next_op.ok_or(CacheError::OpsExhausted)?;
        self.next_op = op.next();
        self.op = Some(op);
        Ok(op)
    }

    /// Moves the op generator past every op of `heads` of the store's site, if it isn't already:
    /// for heads of files the store doesn't read, but whose edits its own must be newer than
    /// (other recovery files of the project, see [`crate::Project::open`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::OpsExhausted`] if one of them is the site's last op id; the generator is
    /// unchanged then.
    pub(crate) fn skip_ops_past(
        &mut self,
        heads: impl IntoIterator<Item = Option<OpId>>,
    ) -> Result<(), CacheError> {
        let Some(next) = self.next_op else {
            return Ok(());
        };
        let after = first_op(next.site(), heads)?;
        self.next_op = Some(next.max(after));
        Ok(())
    }

    /// The op edits are made as now: the one [`TileStore::begin_op`] last started; `None` if
    /// none is (the next edit begins one).
    #[must_use]
    pub const fn current_op(&self) -> Option<OpId> {
        self.op
    }

    /// The op for an edit: the current one, or a new one if none is current.
    ///
    /// # Errors
    ///
    /// As [`TileStore::begin_op`].
    fn edit_op(&mut self) -> Result<OpId, CacheError> {
        match self.op {
            Some(op) => Ok(op),
            None => self.begin_op(),
        }
    }

    /// Bytes the pools hold: whole chunks, never more than the budget.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.budget.used()
    }

    /// `key`'s correction tile: resident, else loaded from the working file, else from the base,
    /// else all zero (an absent tile is zero).
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

    /// Sets the correction at face-grid vertex `(i, j)` of `face` at `level` to `c`, as part of
    /// the current op (see [`TileStore::begin_op`]): in every copy of the vertex (one per tile
    /// holding it, in every face sharing it).
    ///
    /// Each copy's slot becomes dirty, and its key goes to the change feed with the op (once per
    /// op until a consumer drains it, see [`ChangeFeed::push`]). Every cached
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
    /// [`CorrectionError::OutOfTile`], before anything is written;
    /// [`CacheError::OpsExhausted`] if no op is current and none can be begun;
    /// [`CacheError::Store`], [`CacheError::Full`] or [`CacheError::WriteBack`] if a copy's tile
    /// can't be loaded, also before anything is written. Should writing a copy fail all the same
    /// ([`CacheError::Internal`]: a broken slot), the copies written before it are kept, and
    /// their evaluations dropped, before the error is returned.
    pub fn set_correction(
        &mut self,
        face: FaceId,
        level: Level,
        i: u32,
        j: u32,
        c: Vec3,
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
        let op = self.edit_op()?;
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

    /// Edits `key`'s corrections as part of the current op (see [`TileStore::begin_op`]): `f`
    /// sets vertices through a [`TileEdit`], and its result is returned.
    ///
    /// This is [`TileStore::set_correction`] for many vertices of one tile, with the bookkeeping
    /// done once per tile instead of once per vertex: each tile written (the edited one, and the
    /// neighbours holding copies of its border vertices) is loaded and pinned when first
    /// touched, marked dirty and put on the change feed at its first write, and its evaluations
    /// are dropped once, when `f` returns. Errors of single writes go to `f`.
    ///
    /// Should `f` panic, the same happens as the panic unwinds through this call: the touched
    /// tiles are unpinned, and the evaluations of those written are dropped. The writes made
    /// before the panic stay, dirty and on the change feed.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OpsExhausted`] if no op is current and none can
    /// be begun; [`CacheError::Store`], [`CacheError::Full`] or [`CacheError::WriteBack`] if the
    /// tile's corrections can't be loaded (nothing is written then); [`CacheError::Internal`] if
    /// a slot handle went stale.
    pub fn edit_tile<R>(
        &mut self,
        key: TileKey,
        f: impl FnOnce(&mut TileEdit<'_>) -> R,
    ) -> Result<R, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let op = self.edit_op()?;
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
        // Should `f` panic, the edit's drop finishes it all the same.
        let result = f(&mut edit);
        edit.finish()?;
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

    /// Makes every tile of `keys` current: its positions, as [`TileStore::evaluate`] gives them,
    /// and for [`Evaluation::Normals`] its normals too, as [`TileStore::normals`] does. The
    /// tiles are built on the store's worker threads, many at a time, with the same bits as one
    /// by one.
    ///
    /// The keys are taken level by level, coarse to fine, in batches: each batch's dependency
    /// cone is listed at once (the parents of the tiles not cached listed in parallel), then built
    /// coarse to fine, each level's tiles in parallel rounds (see the module notes); normals are
    /// computed in parallel once the positions of every tile they read are installed. A batch
    /// takes a share of the evaluated slots the budget has free of pins, at most a few
    /// thousand targets, and is halved and tried again if it runs out of room, down to one
    /// tile, which the budget always holds while nothing else is pinned.
    ///
    /// The tiles are left cached, not pinned: reading them right after with
    /// [`TileStore::evaluate`] or [`TileStore::normals`] finds them unless `keys` hold more than
    /// the budget does, when the first batches' tiles may be evicted again (and are recomputed
    /// when read). Duplicate keys are evaluated once.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: [`CacheError::NotATile`] before anything is evaluated. The
    /// batches made before an error stay cached.
    pub fn evaluate_many(&mut self, keys: &[TileKey], what: Evaluation) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.poll()?;
        for key in keys {
            self.check_key(key)?;
        }
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        self.in_batches(&keys, Targets::Released, |store, batch| match what {
            Evaluation::Positions => {
                let held = store.pin_current(batch.iter().copied().collect())?;
                store.unpin_held(&held)
            }
            Evaluation::Normals => store.normals_batch(batch),
        })
    }

    /// Evaluates every tile of `keys` and pins it, with its correction tile, until the guard is
    /// dropped. The tiles are evaluated as [`TileStore::evaluate_many`] does, in parallel
    /// batches.
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
        let pinned = self.in_batches(&keys, Targets::Pinned, |store, batch| {
            store.pin_batch(batch, &mut set)
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

    /// Writes every dirty correction tile to the working file and records a save point there.
    /// Returns its offset.
    ///
    /// This makes the working file hold every edit, as of a save point of its own; it doesn't
    /// save the project (see [`crate::Project::save`]). Save's consumer of the change feed is
    /// drained into the tiles the next project save copies. If a write fails, the tiles not
    /// written yet stay dirty for the next save.
    ///
    /// # Errors
    ///
    /// [`CacheError::Streaming`] while a recovery writer streams the working file: its save
    /// points are cuts at its snapshots, and one made here could name later edits than a save
    /// point the writer is about to write, so the durable state could go backwards.
    /// [`CacheError::WriteBack`] if a tile's write fails; [`CacheError::Store`] if the save
    /// point's does.
    pub fn save(&mut self) -> Result<u64, CacheError> {
        if !self.streams.is_empty() {
            return Err(CacheError::Streaming);
        }
        self.sync_pins()?;
        self.drain_saves();
        let mut dirty: Vec<TileKey> = self.unsaved.keys().copied().collect();
        dirty.sort_unstable();
        for key in dirty {
            if let Some(id) = self.corrections.find(&key)
                && self.corrections.is_dirty(id) == Some(true)
            {
                let slot = self.corrections.get(id).ok_or(Invariant::StaleHandle)?;
                let op = write_slot(&mut self.file, &mut self.unsaved, key, slot)?;
                self.corrections.set_dirty(id, false)?;
                self.known_ops.insert(key, op);
            }
        }
        Ok(self.file.save()?)
    }

    /// Subscribes a streaming consumer (the recovery writer) to the change feed, with an empty
    /// flush mark: it streams the edits made from now on.
    pub(crate) fn subscribe_stream(&mut self) -> ConsumerId {
        let id = self.feed.subscribe();
        self.streams.insert(id, BTreeSet::new());
        id
    }

    /// Removes a streaming consumer and its flush mark; entries only it was holding leave the
    /// feed.
    pub(crate) fn unsubscribe_stream(&mut self, id: ConsumerId) {
        self.feed.unsubscribe(id);
        self.streams.remove(&id);
    }

    /// Whether streaming consumer `id` has edits to flush: keys pushed to the feed since its last
    /// snapshot, or keys of its flush mark still unconfirmed.
    pub(crate) fn has_unflushed(&self, id: ConsumerId) -> bool {
        self.feed.has_pending(id) || self.streams.get(&id).is_some_and(|mark| !mark.is_empty())
    }

    /// Copies, for streaming consumer `id`, every correction slot whose latest edit isn't in the
    /// working file yet: the keys of its flush mark, with those the feed has pushed since its
    /// last call added, that are resident and dirty, each with the op and stamp of its latest
    /// write.
    ///
    /// A key of the mark whose slot is clean or not resident is in the working file already (a
    /// slot leaves the pool only once written back, and only a write clears it), so it leaves
    /// the mark without a copy. The rest stay in the mark until
    /// [`TileStore::append_streamed`] appends them, or finds them written by someone else.
    ///
    /// A dirty slot with no op to write it with (a broken slot: every slot marked dirty gets
    /// one first) is left out, dropped from the mark and listed in `skipped`, so it can't stall
    /// the flushes of the rest.
    ///
    /// This is the drawing thread's share of a flush: a copy of each slot, no encoding or I/O.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale. The mark keeps every key then.
    pub(crate) fn snapshot_dirty(
        &mut self,
        id: ConsumerId,
        skipped: &mut Vec<TileKey>,
    ) -> Result<Vec<Snapshot>, CacheError> {
        let drained = self.feed.drain(id);
        let Some(mark) = self.streams.get_mut(&id) else {
            return Ok(Vec::new());
        };
        mark.extend(drained.into_iter().map(|(key, _)| key));
        let (corrections, unsaved) = (&self.corrections, &self.unsaved);
        let mut snapshots = Vec::with_capacity(mark.len());
        let mut written = Vec::new();
        for &key in mark.iter() {
            let Some(slot_id) = corrections.find(&key) else {
                written.push(key);
                continue;
            };
            if corrections.is_dirty(slot_id) != Some(true) {
                written.push(key);
                continue;
            }
            let Some(&Dirty { op, stamp }) = unsaved.get(&key) else {
                // A broken slot: dropped, so the rest still flush.
                skipped.push(key);
                written.push(key);
                continue;
            };
            let mut copy = bytemuck::zeroed_box::<CorrectionSlot>();
            *copy = *corrections.get(slot_id).ok_or(Invariant::StaleHandle)?;
            snapshots.push(Snapshot {
                key,
                op,
                stamp,
                slot: copy,
            });
        }
        for key in &written {
            mark.remove(key);
        }
        Ok(snapshots)
    }

    /// Appends streaming consumer `id`'s encoded `tiles` to the working file: each one whose slot
    /// is still resident and dirty. A slot unchanged since its snapshot (its latest write is the
    /// one the record copies, by stamp) is clean once its record is appended, and leaves the
    /// mark. Ops can't tell: a later dab of the same stroke writes the slot with the same op.
    ///
    /// A slot written again since its snapshot is appended all the same, so the save point after
    /// this append holds every edit up to the snapshot, but stays dirty and in the mark: its
    /// newer edit isn't in the file, and whatever writes the slot next (the next flush, a save,
    /// or a write-back to make room) appends it after this record. Reads of a dirty tile come
    /// from its slot, never the file, meanwhile.
    ///
    /// A tile whose slot is clean or gone was written by someone else (a write-back to make
    /// room) after the snapshot, at a write at least as new. If that write is at or before the
    /// working file's cut, the record is the cut's state and the copy is skipped; if it is past
    /// the cut, the copy is appended for the cut's save point, and the newer record stays the
    /// latest (see [`ProjectFile::open_cut`]). Either way it leaves the mark.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if an append fails: that tile and those after it stay dirty
    /// and in the mark. [`CacheError::Internal`] for a stale handle.
    pub(crate) fn append_streamed(
        &mut self,
        id: ConsumerId,
        tiles: &[FramedTile],
    ) -> Result<(), CacheError> {
        for tile in tiles {
            let key = tile.key();
            let dirty = self
                .corrections
                .find(&key)
                .filter(|&found| self.corrections.is_dirty(found) == Some(true));
            let confirmed = match dirty {
                // Written since the snapshot by an edit past the cut: the cut still needs this
                // copy, which goes to the cut's save point, not over the newer record.
                None if self.file.is_past_cut(&key) => {
                    self.file
                        .append_framed(tile)
                        .map_err(CacheError::WriteBack)?;
                    true
                }
                None => true,
                Some(found) => {
                    self.file
                        .append_framed(tile)
                        .map_err(CacheError::WriteBack)?;
                    let unchanged = self
                        .unsaved
                        .get(&key)
                        .is_some_and(|write| write.stamp == tile.stamp());
                    if unchanged {
                        self.unsaved.remove(&key);
                        self.corrections.set_dirty(found, false)?;
                        self.known_ops.insert(key, tile.op());
                    }
                    unchanged
                }
            };
            if confirmed && let Some(mark) = self.streams.get_mut(&id) {
                mark.remove(&key);
            }
        }
        Ok(())
    }

    /// Opens a cut in the working file at the latest write so far: the next save point the
    /// recovery writer writes names the state as of now, whatever is written back meanwhile
    /// (see [`ProjectFile::open_cut`]). Called as a flush takes its snapshot.
    ///
    /// The cut is at a write stamp, not an op, so it falls between two dabs of one stroke.
    pub(crate) fn open_cut(&mut self) {
        self.file.open_cut(self.last_write);
    }

    /// The working-file generation: moved on by every rebase.
    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }

    /// Drops `key` from streaming consumer `id`'s flush mark: its copy couldn't be flushed, and
    /// is reported instead of tried again with every flush.
    pub(crate) fn drop_from_stream(&mut self, id: ConsumerId, key: &TileKey) {
        if let Some(mark) = self.streams.get_mut(&id) {
            mark.remove(key);
        }
    }

    /// A handle that syncs the working file from another thread.
    pub(crate) fn working_syncer(&self) -> FileSyncer {
        self.file.syncer()
    }

    /// Where the working file's next record goes: it moves with every append, so a caller can
    /// tell whether anything was written since it last looked.
    pub(crate) const fn working_len(&self) -> u64 {
        self.file.len()
    }

    /// Whether the working file has tiles written since its last save point.
    pub(crate) fn has_unsaved_tiles(&self) -> bool {
        self.file.has_unsaved_tiles()
    }

    /// Appends a save point to the working file naming every tile written to it so far, without
    /// syncing (see [`ProjectFile::write_save_point`]), and returns its offset. Dirty slots are
    /// not written: their edits are in no save point yet.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if the write fails.
    pub(crate) fn write_save_point(&mut self) -> Result<u64, CacheError> {
        Ok(self.file.write_save_point()?)
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
            .evict_all(|key, slot| write_slot(file, unsaved, key, slot).map(drop))?;
        Ok(())
    }

    /// Gives chunks back until the pools hold at most `bytes` or only chunks with pinned slots
    /// are left: evaluated chunks first, then correction chunks (dirty tiles written back), the
    /// cheapest of each pool first (see the memory policy).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evict_all`].
    pub fn shrink_to(&mut self, bytes: usize) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.give_back(bytes)?;
        Ok(())
    }

    /// Asks the prefetch workers to read the correction tiles of `keys` in the background, ahead
    /// of the store's use of them, each from the file a load would read (the working file, else
    /// the base); returns how many reads were queued.
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
                let (source, loc) = self.locate(&key)?;
                Some(Job { key, source, loc })
            })
            .collect();
        Ok(self
            .prefetch
            .request(&self.file, self.base.as_ref(), &jobs)?)
    }

    /// Installs the correction tiles the prefetch workers have finished, and returns how many.
    ///
    /// A finished tile is installed only if it is still wanted and still current: not resident
    /// (a resident tile may have been edited since, and is what the store uses either way), and
    /// its record still the one a load would read (any write of the tile, by an edit's
    /// write-back or a save, moves it to the working file). It goes into a free slot, a new chunk
    /// if the budget allows one, or the slot of a clean correction tile: prefetch never takes a
    /// pinned slot, an evaluated chunk or a dirty tile's slot, so it never writes anything
    /// back. Anything else is dropped, as is a tile a worker couldn't read: a later load reads
    /// it again and reports the error.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a fresh slot's handle went stale (it doesn't); the results not
    /// reached yet wait for the next poll.
    pub fn poll(&mut self) -> Result<usize, CacheError> {
        let mut installed = 0;
        while let Some(Fetched { job, slot }) = self.prefetch.next() {
            let Ok((op, slot)) = slot else { continue };
            let current = self.locate(&job.key) == Some((job.source, job.loc));
            let outcome = if current {
                self.install(job.key, op, &slot)
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

    /// Puts a prefetched correction tile, read at its latest record (written by `op`), into a
    /// fresh slot; whether it did. A resident tile (every dirty tile is resident) is left alone,
    /// and so is the tile when the correction pool has no slot to give without writing a dirty
    /// tile back or taking memory from the evaluated pool (see [`Pool::acquire_clean`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if the fresh slot's handle went stale (it doesn't).
    fn install(
        &mut self,
        key: TileKey,
        op: OpId,
        tile: &CorrectionSlot,
    ) -> Result<bool, CacheError> {
        if self.corrections.find(&key).is_some() {
            return Ok(false);
        }
        let id = match self.corrections.acquire_clean(key, &mut self.budget) {
            Ok(id) => id,
            Err(PoolError::Full) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        *self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)? = *tile;
        self.known_ops.insert(key, op);
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

    /// Evaluates the tiles of `batch` (of one level, in key order) and pins their evaluated and
    /// correction slots, adding them to `set`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; nothing this call pinned stays pinned then.
    fn pin_batch(&mut self, batch: &[TileKey], set: &mut Vec<Pinned>) -> Result<(), CacheError> {
        let held = self.pin_current(batch.iter().copied().collect())?;
        let mut corrections = Vec::with_capacity(held.len());
        let loaded = held.iter().try_for_each(|&(key, _)| {
            let id = self.load_correction(key)?;
            self.corrections.pin(id)?;
            corrections.push(id);
            Ok::<_, CacheError>(())
        });
        if let Err(e) = loaded {
            self.unpin_corrections(&corrections)?;
            self.unpin_held(&held)?;
            return Err(e);
        }
        set.extend(held.iter().map(|&(_, id)| id).zip(corrections));
        Ok(())
    }

    /// Runs `run` over `keys` in batches, for [`TileStore::evaluate_many`] and
    /// [`TileStore::pin`]: level by level, coarse to fine, each level's keys in key order, cut
    /// into batches of [`TileStore::batch_size`] tiles. A batch refused with
    /// [`CacheError::Full`] is halved and run again, until it is one tile; the size reached is
    /// kept for the batches after it.
    ///
    /// `run` must leave nothing it pinned pinned when it fails, so that a smaller batch finds
    /// the room the failed one had. With [`Targets::Pinned`], the targets still to come count
    /// against each batch's room, as they will be pinned beside it.
    ///
    /// # Errors
    ///
    /// The first error of `run` but [`CacheError::Full`] on a batch of more than one tile.
    fn in_batches(
        &mut self,
        keys: &BTreeSet<TileKey>,
        targets: Targets,
        mut run: impl FnMut(&mut Self, &[TileKey]) -> Result<(), CacheError>,
    ) -> Result<(), CacheError> {
        let mut levels: BTreeMap<Level, Vec<TileKey>> = BTreeMap::new();
        for &key in keys {
            levels.entry(key.level()).or_default().push(key);
        }
        let mut pending = match targets {
            Targets::Pinned => keys.len(),
            Targets::Released => 0,
        };
        for level in levels.values() {
            let mut rest = level.as_slice();
            let mut size = self.batch_size(pending);
            while !rest.is_empty() {
                let (batch, after) = rest.split_at(size.min(rest.len()));
                match run(self, batch) {
                    Err(CacheError::Full) if batch.len() > 1 => size = batch.len() / 2,
                    Err(e) => return Err(e),
                    Ok(()) => {
                        pending = pending.saturating_sub(batch.len());
                        rest = after;
                    }
                }
            }
        }
        Ok(())
    }

    /// The targets the next batch of [`TileStore::in_batches`] takes: a [`BATCH_SHARE`] of the
    /// evaluated slots the budget would hold besides those pinned, the `pending` targets still
    /// to come (which [`TileStore::pin`] has counted on keeping room for) and an evaluation's
    /// own headroom; at least one, at most [`MAX_BATCH`].
    ///
    /// It depends on the budget and the pins alone, never on the thread count, so the batches,
    /// like the rounds, are the same whatever the workers.
    fn batch_size(&self, pending: usize) -> usize {
        let chunks = self.budget.limit() / self.evaluated.chunk_bytes().max(1);
        let slots = chunks.saturating_mul(self.evaluated.per_chunk() as usize);
        let taken = self.evaluated.pinned_slots() + pending + self.headroom.evaluated;
        (slots.saturating_sub(taken) / BATCH_SHARE).clamp(1, MAX_BATCH)
    }

    /// Unpins correction slots a round or a pin batch pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale (pinned slots are never freed, so this
    /// doesn't happen); every slot is unpinned either way.
    fn unpin_corrections(&mut self, ids: &[SlotId<CorrectionSlot>]) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &id in ids {
            keep_first(&mut outcome, self.corrections.unpin(id));
        }
        outcome
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
        fn chunks_needed<T: SlotData>(
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

    /// Where a load of `key`'s correction tile reads it: its record in the working file, else in
    /// the base; `None` if neither has one (the tile is zero).
    fn locate(&self, key: &TileKey) -> Option<(Source, Loc)> {
        self.file
            .locate(key)
            .map(|loc| (Source::Working, loc))
            .or_else(|| {
                let loc = self.base.as_ref()?.locate(key)?;
                Some((Source::Base, loc))
            })
    }

    /// `key`'s correction slot: resident, else loaded from the working file, else from the base,
    /// else reset to zero.
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
        let found = self.locate(&key);
        let slot = self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)?;
        let scratch = &mut self.read_scratch;
        let read = match (found, &self.base) {
            (None, _) => Ok(None),
            (Some((Source::Working, loc)), _) => {
                self.file.read_at(&key, loc, slot, scratch).map(Some)
            }
            (Some((Source::Base, loc)), Some(base)) => {
                base.read_at(&key, loc, slot, scratch).map(Some)
            }
            (Some((Source::Base, _)), None) => {
                // The slot holds no tile: give it back (it is clean and unpinned).
                self.corrections.remove(&key)?;
                return Err(Invariant::MissingBase.into());
            }
        };
        #[cfg(test)]
        let read = if self.unreadable == Some(key) {
            Err(io::Error::other("an unreadable tile, for the test").into())
        } else {
            read
        };
        let filled = match read {
            Ok(Some(op)) => Ok(Some(op)),
            Ok(None) => slot
                .reset(key.side())
                .map(|()| None)
                .map_err(|e| Invariant::CorrectionSlot(e).into()),
            Err(e) => Err(e.into()),
        };
        match filled {
            Ok(Some(op)) => {
                self.known_ops.insert(key, op);
            }
            Ok(None) => {
                self.known_ops.remove(&key);
            }
            Err(e) => {
                // The slot holds no tile: give it back (it is clean and unpinned).
                self.corrections.remove(&key)?;
                return Err(e);
            }
        }
        Ok(id)
    }

    /// A fresh slot for `key`'s correction tile, as the memory policy orders it (see
    /// [`TileStore`]): a free slot, a new chunk, a clean victim, an evaluated chunk taken back,
    /// one dirty victim written back, and only when every correction slot is pinned, chunks
    /// taken back from the pools.
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`]; [`CacheError::WriteBack`] if a dirty tile's write-back fails.
    fn acquire_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        match self.corrections.acquire_clean(key, &mut self.budget) {
            Err(PoolError::Full) => {}
            other => return Ok(other?),
        }
        if self.may_take_evaluated_chunk() {
            let target = self
                .budget
                .limit()
                .saturating_sub(self.corrections.chunk_bytes());
            let shrunk = self
                .evaluated
                .shrink_to(&mut self.budget, target, nothing_to_write);
            self.note_evictions();
            shrunk?;
            match self.corrections.acquire_clean(key, &mut self.budget) {
                Err(PoolError::Full) => {}
                other => return Ok(other?),
            }
        }
        for attempt in 0..2 {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            match self.corrections.acquire(key, &mut self.budget, |k, slot| {
                write_slot(file, unsaved, k, slot).map(drop)
            }) {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.corrections.chunk_bytes(), 0)? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
    }

    /// Whether a correction acquire that found no free or clean slot may take an evaluated
    /// chunk back rather than write a dirty tile back: while the correction pool holds less
    /// than a quarter of the budget, and the evaluated pool keeps its floor (its share of the
    /// minimum budget plus one chunk) without the chunk, and has a chunk without pins to give.
    fn may_take_evaluated_chunk(&self) -> bool {
        let chunk = self.evaluated.chunk_bytes();
        let floor = self.headroom.evaluated_floor(chunk);
        self.corrections.bytes().saturating_mul(4) < self.budget.limit()
            && self.evaluated.bytes().saturating_sub(chunk) >= floor
            && self.evaluated.has_unpinned_chunk()
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
                    let keep = self.corrections.chunk_bytes();
                    if !self.make_room(self.evaluated.chunk_bytes(), keep)? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
    }

    /// Takes chunks back until another `need` bytes fit the budget, leaving the correction pool
    /// at least `keep` bytes of its chunks; whether they fit now.
    ///
    /// Evaluated chunks go first, then correction chunks (see [`TileStore::give_back`]), but
    /// never below `keep`. The evaluated pool asks with one correction chunk kept: the
    /// correction floor. Without it, a batch whose cone pins every evaluated chunk could take
    /// every correction chunk back, fail, and leave the one-tile batch it is halved to with no
    /// correction slot to load into, as the coarsest level's pins would then sit in every
    /// evaluated chunk (see the module notes). The correction pool asks with nothing kept: its
    /// own chunks are what it is short of.
    ///
    /// # Errors
    ///
    /// As [`TileStore::give_back`].
    fn make_room(&mut self, need: usize, keep: usize) -> Result<bool, CacheError> {
        let target = self.budget.limit().saturating_sub(need);
        let evaluated = self
            .evaluated
            .shrink_to(&mut self.budget, target, nothing_to_write);
        self.note_evictions();
        evaluated?;
        let floor = self.evaluated.bytes().saturating_add(keep);
        self.give_back(target.max(floor))?;
        Ok(self.budget.used() <= target)
    }

    /// Gives chunks without pinned slots back until the pools hold at most `bytes`: evaluated
    /// chunks first (a cache, cheap to recompute), then correction chunks, whose dirty tiles are
    /// written back to the working file first; the cheapest of each pool first (see
    /// [`TileStore`]'s memory policy).
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
                    write_slot(file, unsaved, key, slot).map(drop)
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

    /// Writes `c` at `(i, j)` of `key`'s correction slot `id` (resident and pinned) as part of
    /// `op`, and marks the slot written (see [`TileStore::mark_written`]).
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

    /// Records that `key`'s correction slot `id` was written as part of `op`: the write takes
    /// the next stamp, and the slot is dirty, to be written with `op`, and pushed to the change
    /// feed with it (the feed skips the push if no consumer has drained the same key and op
    /// yet).
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
        self.last_write = self.last_write.next();
        self.unsaved.insert(
            key,
            Dirty {
                op,
                stamp: self.last_write,
            },
        );
        self.feed.push(key, op);
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
    /// again, however many there are and whatever their ops. While the dependency graph holds
    /// an untracked node, nothing is skipped: the cascade takes such a node to follow from
    /// every coarser edit, recorded or not (see [`crate::deps`]).
    fn invalidate_edited(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        let reached: Vec<TileKey> = edited
            .into_iter()
            .filter(|key| {
                self.evaluated.freshness(key) == Some(Freshness::Fresh)
                    || self.deps.is_read(key)
                    || self.deps.has_untracked()
            })
            .collect();
        if !reached.is_empty() {
            self.invalidate(reached);
        }
    }

    /// Drops every evaluation that followed from the `edited` tiles' corrections: positions
    /// through the recorded parents, transitively, and the normals of those tiles. A pinned
    /// tile keeps its slot, marked stale, and is recomputed in place on its next evaluation.
    ///
    /// Normals that read a dropped tile's positions but whose own tile's positions stay are
    /// kept, marked stale since the inputs that changed (see [`crate::deps`]): their next
    /// computation recomputes only the vertices that read those.
    fn invalidate(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        #[cfg(test)]
        {
            self.log.invalidations += 1;
        }
        let cascade = self.deps.cascade(edited);
        for (key, changed) in cascade.normals {
            self.deps.stale_normals(key, changed);
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
            let level = self.parents_of_all(tiles)?;
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

    /// The tiles each of `keys` is predicted from, level 0 (the mesh) left out, in key order: as
    /// the dependency graph recorded them for a key that still has a node, else from
    /// [`parent_tiles`], listed on the workers.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn parents_of_all(&self, keys: BTreeSet<TileKey>) -> Result<ConeLevel, CacheError> {
        let keys: Vec<TileKey> = keys.into_iter().collect();
        #[cfg(test)]
        for key in &keys {
            let counter = if self.deps.parents(key).is_some() {
                &self.log.parents_reused
            } else {
                &self.log.parents_listed
            };
            counter.set(counter.get() + 1);
        }
        let (mesh, deps) = (&*self.mesh, &self.deps);
        let found = map_parallel(&self.workers, &keys, |key| match deps.parents(key) {
            Some(parents) => Ok(parents.to_vec()),
            None => list_parents(mesh, key),
        });
        keys.into_iter()
            .zip(found)
            .map(|(key, parents)| Ok((key, parents?)))
            .collect()
    }

    /// The tiles the normals of each of `keys` are computed from, the tile itself included: as
    /// the dependency graph recorded them for a key that still has a node listing them, else
    /// from [`normal_tiles`], listed on the workers.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn normal_inputs_of_all(&self, keys: &[TileKey]) -> Result<Vec<BTreeSet<TileKey>>, CacheError> {
        #[cfg(test)]
        for key in keys {
            let counter = if self.deps.normal_inputs(key).is_some() {
                &self.log.normal_inputs_reused
            } else {
                &self.log.normal_inputs_listed
            };
            counter.set(counter.get() + 1);
        }
        let (mesh, deps) = (&*self.mesh, &self.deps);
        map_parallel(&self.workers, keys, |key| match deps.normal_inputs(key) {
            Some(inputs) => Ok(inputs.iter().copied().collect()),
            None => list_normal_inputs(mesh, key),
        })
        .into_iter()
        .collect()
    }

    /// Makes every tile of one cone level current, building those that aren't from `below` (the
    /// level under it, pinned) in parallel rounds, and pins it. Returns them pinned, in key
    /// order; nothing this call pinned stays pinned on error.
    ///
    /// The level's current tiles are pinned first, so that building the others can't evict
    /// them.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn pin_level(&mut self, level: &ConeLevel, below: &Held) -> Result<Held, CacheError> {
        let mut held = Held::with_capacity(level.len());
        let mut stale = Vec::new();
        let pinned = level.iter().try_for_each(|(&key, parents)| {
            match self.evaluated.lookup(&key) {
                Some((id, Freshness::Fresh)) => {
                    self.evaluated.pin(id)?;
                    held.push((key, id));
                }
                _ => stale.push((key, parents.as_slice())),
            }
            Ok::<_, CacheError>(())
        });
        if let Err(e) = pinned.and_then(|()| self.build_level(&stale, below, &mut held)) {
            self.unpin_held(&held)?;
            return Err(e);
        }
        held.sort_unstable_by_key(|&(key, _)| key);
        Ok(held)
    }

    /// Builds `tiles` (each with its parents, all pinned in `below`) round by round, pinning
    /// each tile as it is installed and adding it to `held`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; the tiles installed before the error stay in `held`, pinned,
    /// for the caller to unpin.
    fn build_level(
        &mut self,
        tiles: &[(TileKey, &[TileKey])],
        below: &Held,
        held: &mut Held,
    ) -> Result<(), CacheError> {
        let mut rest = tiles;
        while !rest.is_empty() {
            let corrections = self.load_round(rest)?;
            let (round, after) = rest.split_at(corrections.len());
            self.build_round(round, &corrections, below, held)?;
            rest = after;
        }
        Ok(())
    }

    /// Loads and pins the correction tiles of the next round: the first [`ROUND`] tiles of
    /// `tiles`, or as many of them as the correction pool holds besides its pins (at least the
    /// first). Returns their slots, in order.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`], for the first tile only; nothing stays pinned then.
    fn load_round(
        &mut self,
        tiles: &[(TileKey, &[TileKey])],
    ) -> Result<Vec<SlotId<CorrectionSlot>>, CacheError> {
        let mut ids = Vec::with_capacity(tiles.len().min(ROUND));
        for &(key, _) in tiles.iter().take(ROUND) {
            let loaded = self.load_correction(key).and_then(|id| {
                self.corrections.pin(id)?;
                Ok(id)
            });
            match loaded {
                Ok(id) => ids.push(id),
                Err(CacheError::Full) if !ids.is_empty() => break,
                Err(e) => {
                    self.unpin_corrections(&ids)?;
                    return Err(e);
                }
            }
        }
        Ok(ids)
    }

    /// Builds one round: predicts every tile of `round` on the workers, from `below` and its
    /// pinned correction slot (`corrections`, in the same order), unpins the correction slots,
    /// then installs each tile in order (see [`TileStore::store_positions`]), pinning it and
    /// adding it to `held`.
    ///
    /// The correction slots are unpinned before anything is installed, as an evaluation one
    /// tile at a time holds no correction slot while it takes an evaluated one: the evaluated
    /// pool may need a correction chunk back to make room (see [`StoreConfig::minimum_budget`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: the first error of the round in key order. The tiles
    /// installed before it stay in `held`, pinned; the correction slots are unpinned either way.
    fn build_round(
        &mut self,
        round: &[(TileKey, &[TileKey])],
        corrections: &[SlotId<CorrectionSlot>],
        below: &Held,
        held: &mut Held,
    ) -> Result<(), CacheError> {
        let mut scratch = self.take_scratch(round.len());
        let predicted = self.predict_round(round, corrections, below, &mut scratch);
        let released = self.unpin_corrections(corrections);
        let built = predicted.and(released).and_then(|()| {
            round
                .iter()
                .zip(&scratch)
                .try_for_each(|(&(key, parents), positions)| {
                    let id = self.store_positions(key, parents, positions)?;
                    self.evaluated.pin(id)?;
                    held.push((key, id));
                    Ok(())
                })
        });
        self.scratch = scratch;
        built
    }

    /// Predicts every tile of `round` on the workers, from the pinned tiles of `below` and its
    /// correction slot (`corrections`, in the same order), into the buffers of `scratch`, one
    /// per tile. The workers only read: the pools are shared, not changed.
    ///
    /// # Errors
    ///
    /// The first tile's error in key order: [`CacheError::Eval`] (not for a valid mesh), or
    /// [`CacheError::Internal`] for a stale handle or a broken slot.
    fn predict_round(
        &self,
        round: &[(TileKey, &[TileKey])],
        corrections: &[SlotId<CorrectionSlot>],
        below: &Held,
        scratch: &mut [Vec<Vec3>],
    ) -> Result<(), CacheError> {
        let mesh = &*self.mesh;
        let source = Resident {
            mesh,
            pool: &self.evaluated,
            tiles: below,
        };
        let pool = &self.corrections;
        let tasks: Vec<(TileKey, SlotId<CorrectionSlot>)> = round
            .iter()
            .map(|&(key, _)| key)
            .zip(corrections.iter().copied())
            .collect();
        let predict = |&(key, id): &(TileKey, SlotId<CorrectionSlot>), out: &mut Vec<Vec3>| {
            let view = pool
                .get(id)
                .ok_or(Invariant::StaleHandle)?
                .view()
                .map_err(Invariant::CorrectionSlot)?;
            let prediction = predict_region(mesh, &source, Region::of_tile(&key))?;
            let out = prefix_mut(out, vertex_count(&key))?;
            prediction.apply_into(|i, j| view.get(i, j).unwrap_or(Vec3::ZERO), out)?;
            Ok(())
        };
        each_parallel(&self.workers, &tasks, scratch, predict)
    }

    /// The scratch buffers, taken out of the store so the owner thread can install from them
    /// while it changes the store, with at least `n` (at most [`ROUND`]) of them. The caller
    /// puts them back.
    fn take_scratch(&mut self, n: usize) -> Vec<Vec<Vec3>> {
        let mut scratch = std::mem::take(&mut self.scratch);
        let n = n.min(ROUND);
        if scratch.len() < n {
            scratch.resize_with(n, || vec![Vec3::ZERO; SIDE * SIDE]);
        }
        scratch
    }

    /// Copies `positions` into `key`'s slot (a new one, or its stale one in place), marks it
    /// fresh and records it as predicted from `parents`.
    ///
    /// The caller holds `parents` pinned: the tile is recorded before they can leave the pool,
    /// so their nodes, and their own parent edges, are still in the dependency graph (see
    /// [`crate::deps`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`] if no slot can be had; [`CacheError::Internal`] if `positions` is
    /// shorter than the tile.
    fn store_positions(
        &mut self,
        key: TileKey,
        parents: &[TileKey],
        positions: &[Vec3],
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        // A stale slot (pinned elsewhere, so still here) is overwritten in place: its handles
        // stay valid.
        let id = match self.evaluated.lookup(&key) {
            Some((id, _)) => id,
            None => self.acquire_evaluated(key)?,
        };
        let n = vertex_count(&key);
        let slot = self.evaluated.get_mut(id).ok_or(Invariant::StaleHandle)?;
        prefix_mut(&mut slot.positions, n)?.copy_from_slice(prefix(positions, n)?);
        self.evaluated.mark_fresh(id)?;
        self.deps.record_positions(key, parents);
        #[cfg(test)]
        self.log.positions.push(key);
        Ok(id)
    }

    /// The slot of `key`'s current positions and normals: cached, else the normals are computed
    /// (or, where they are stale, brought up to date) from the positions of its
    /// [`normal_tiles`], made current together as one cone.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn ensure_normals(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        self.normals_batch(&[key])?;
        // The normals were computed with the tile pinned, and nothing has taken a slot since.
        match self.evaluated.lookup(&key) {
            Some((id, Freshness::Fresh)) => Ok(id),
            _ => Err(Invariant::StaleHandle.into()),
        }
    }

    /// Makes the normals of every tile of `batch` (tiles of one level, in key order) current:
    /// those not cached are computed in parallel rounds from the positions of all their normal
    /// inputs, made current and pinned together as one cone, and each tile is recorded as
    /// reading its own inputs, while they are pinned (as [`TileStore::store_positions`]
    /// records positions).
    ///
    /// A tile whose normals are stale (see [`crate::deps`]) has only the vertices that read a
    /// changed input recomputed, with [`geom::refresh_normals_into`]: the border vertices next
    /// to a neighbour the brush moved, which is the same bits as computing them all. Whether
    /// they are stale is asked once the inputs are pinned: making them current may have evicted
    /// the tile, or rebuilt its positions, and its stale normals went with them.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; nothing stays pinned then.
    fn normals_batch(&mut self, batch: &[TileKey]) -> Result<(), CacheError> {
        let mut wanted = Vec::with_capacity(batch.len());
        for &key in batch {
            let current = matches!(self.evaluated.lookup(&key), Some((_, Freshness::Fresh)))
                && self.deps.has_normals(&key);
            if !current {
                wanted.push(key);
            }
        }
        if wanted.is_empty() {
            return Ok(());
        }
        let inputs = self.normal_inputs_of_all(&wanted)?;
        let held = self.pin_current(inputs.iter().flatten().copied().collect())?;
        let stale: Vec<Option<BTreeSet<TileKey>>> = wanted
            .iter()
            .map(|key| self.deps.stale_inputs(key).cloned())
            .collect();
        let computed = self.normals_rounds(&wanted, &inputs, &stale, &held);
        let released = self.unpin_held(&held);
        computed?;
        released
    }

    /// Computes the normals of `wanted` (each with its normal `inputs` and, where its normals
    /// are stale, the inputs they are stale since, in the same order) from `held`, every tile
    /// they read, pinned: on the workers, a round at a time, each round then installed in order
    /// (see [`TileStore::store_normals`]).
    ///
    /// A stale tile's normals are copied out of its slot into the scratch buffer and brought up
    /// to date there by [`geom::refresh_normals_into`]; any other's are computed in full.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: the first error in key order.
    fn normals_rounds(
        &mut self,
        wanted: &[TileKey],
        inputs: &[BTreeSet<TileKey>],
        stale: &[Option<BTreeSet<TileKey>>],
        held: &Held,
    ) -> Result<(), CacheError> {
        let mut scratch = self.take_scratch(wanted.len());
        let tasks: Vec<(TileKey, Option<&BTreeSet<TileKey>>)> = wanted
            .iter()
            .copied()
            .zip(stale.iter().map(Option::as_ref))
            .collect();
        let outcome =
            tasks
                .chunks(ROUND)
                .zip(inputs.chunks(ROUND))
                .try_for_each(|(round, inputs)| {
                    let mesh = &*self.mesh;
                    let pool = &self.evaluated;
                    let source = Resident {
                        mesh,
                        pool,
                        tiles: held,
                    };
                    let compute = |&(key, since): &(TileKey, Option<&BTreeSet<TileKey>>),
                                   out: &mut Vec<Vec3>| {
                        let n = vertex_count(&key);
                        let out = prefix_mut(out, n)?;
                        let region = Region::of_tile(&key);
                        let Some(since) = since else {
                            normals_into(mesh, &source, region, out)?;
                            return Ok(());
                        };
                        let id = held_slot(held, &key)?;
                        let slot = pool.get(id).ok_or(Invariant::StaleHandle)?;
                        out.copy_from_slice(prefix(&slot.normals, n)?);
                        refresh_normals_into(mesh, &source, region, |k| since.contains(k), out)?;
                        Ok(())
                    };
                    each_parallel(&self.workers, round, &mut scratch, compute)?;
                    #[cfg(test)]
                    self.log.refreshed.extend(
                        round
                            .iter()
                            .filter(|(_, since)| since.is_some())
                            .map(|&(key, _)| key),
                    );
                    round.iter().zip(inputs).zip(&scratch).try_for_each(
                        |((&(key, _), inputs), normals)| {
                            self.store_normals(key, inputs, held, normals)
                        },
                    )
                });
        self.scratch = scratch;
        outcome
    }

    /// Copies `normals` into `key`'s slot (pinned in `held`) and records them as read from
    /// `inputs`, which `held` holds pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if `key` isn't in `held`, its handle went stale, or `normals`
    /// is shorter than the tile.
    fn store_normals(
        &mut self,
        key: TileKey,
        inputs: &BTreeSet<TileKey>,
        held: &Held,
        normals: &[Vec3],
    ) -> Result<(), CacheError> {
        let own = held_slot(held, &key)?;
        let n = vertex_count(&key);
        let slot = self.evaluated.get_mut(own).ok_or(Invariant::StaleHandle)?;
        prefix_mut(&mut slot.normals, n)?.copy_from_slice(prefix(normals, n)?);
        self.deps.record_normals(key, inputs);
        #[cfg(test)]
        self.log.normals.push(key);
        Ok(())
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

/// One tile's corrections being edited as part of the current op: what
/// [`TileStore::edit_tile`] hands its closure.
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
    /// The tiles holding copies touched so far, the edited tile first; empty once the edit is
    /// finished. A handful (a border vertex's copies lie in the tile's neighbours), so a linear
    /// search finds them.
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

/// Finishes an edit whose closure panicked, so its tiles don't stay pinned and the evaluations
/// of those it wrote don't stay cached. After a normal return the edit is finished already, and
/// this does nothing.
impl Drop for TileEdit<'_> {
    fn drop(&mut self) {
        // While unwinding there is no caller to report to, and the only error is a broken
        // invariant: every touched tile is unpinned either way.
        drop(self.finish());
    }
}

impl TileEdit<'_> {
    /// Ends the edit: unpins every touched tile and drops the evaluations that followed from
    /// the tiles written. Touched tiles are forgotten, so a second call does nothing.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a touched slot's handle went stale (pinned slots are never
    /// freed, so this doesn't happen); every tile is unpinned and invalidated either way.
    fn finish(&mut self) -> Result<(), CacheError> {
        let touched = std::mem::take(&mut self.touched);
        let mut outcome = Ok(());
        for tile in &touched {
            keep_first(&mut outcome, self.store.corrections.unpin(tile.id));
        }
        self.store.invalidate_edited(
            touched
                .iter()
                .filter(|tile| tile.written)
                .map(|tile| tile.key),
        );
        outcome
    }

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
pub(crate) mod tests;
