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
//! The spatial index's bounds layer the same way, with the bounds changed since the last save on
//! top, in memory: the store keeps them consistent with the tiles across edits, flushes, saves and
//! rebases (see [`crate::bounds`]).
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
//! on the store's worker threads ([`StoreConfig::workers`]), in rounds of at most [`ROUND`]
//! tiles: the owner thread loads and pins the round's correction tiles, the workers predict
//! every tile of the round from the pinned level below, and the owner unpins the correction
//! tiles and installs the results one by one, in key order. A round of tiles that already have
//! slots (stale tiles recomputed in place, as after a brush step) is predicted straight into
//! those slots, each task writing its own; any other round into scratch buffers, which the
//! owner copies into the slots it takes as it installs them. Normals are computed straight into
//! their tiles' slots the same way. The workers write only the slots (or buffers) handed to
//! them, one per task, borrowed disjointly from the pool, and read the pinned tiles through
//! shared borrows; nothing they do touches the pools' bookkeeping, the budget or the dependency
//! graph, and the rounds are cut the same way whatever the thread count, so the store makes the
//! same decisions (which slot, which victim) and computes the same bits with one thread or
//! thirty-two.
//!
//! [`TileStore::evaluate_many`] widens the cone to many target tiles at once, so the levels are
//! wide enough to keep every worker busy; [`TileStore::pin`] goes through it too. Such a cone's
//! levels are wider than [`geom::cone_width`]: a batch is sized to the room the evaluated pool
//! can reach free of pins, and halved, down to the one tile the bound above holds for, if it
//! runs out.
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
//! nothing else is pinned, and [`TileStore::open`] refuses less. An evaluation of one tile pins
//! at most `E` evaluated slots (two cone levels), and correction slots only for a round, while
//! the workers predict it: a round unpins them before it installs anything, so no correction
//! slot of the evaluation's is pinned when it asks for an evaluated one.
//!
//! - The evaluated pool needs a chunk only when every one of its slots is pinned, so it holds fewer
//!   than `⌈E / per chunk⌉` chunks then, and the minimum leaves room for one more besides a
//!   correction chunk. Making that room takes evaluated chunks back first, then correction chunks,
//!   and stops as soon as the room is there, so the correction pool keeps a chunk, however the
//!   evaluated pins are spread.
//! - Besides, making room for the evaluated pool never takes the correction pool's last chunk: the
//!   correction floor (see [`TileStore::make_room`]). Within the bound above it never binds, as the
//!   room is there first; it binds for a batch of [`TileStore::evaluate_many`], whose wider cone
//!   can pin every evaluated chunk. Such a batch fails and is halved, and the coarsest level's pins
//!   of the one-tile batch it comes down to may then sit in every evaluated chunk: without the
//!   floor, no chunk would be left to load its correction tiles into.
//! - The correction pool so always has a slot to reuse when a round's first tile loads: none of its
//!   slots is pinned then. (Its first chunk is mapped by an evaluation's first build, which loads a
//!   correction tile before it asks for an evaluated slot.) A round's other tiles take only a free
//!   slot, a new chunk or a clean tile's slot, and the round is cut at the first that finds none
//!   ([`TileStore::load_round`]): a round that can't widen is a round of one tile, as one tile at a
//!   time would build it, and writes back no more.
//!
//! An edit pins correction slots only, at most one cone width of them, which the minimum's
//! correction chunks hold once every evaluated chunk is taken back.

use std::borrow::Cow;
#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionView, FaceId, LayerId, Level, MAX_COPIES,
    Prediction, Region, TileBorder, TileKey, Vec3, normals_into, refresh_normals_into,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use self::config::{BATCH_SHARE, Headroom, MAX_BATCH, ROUND};
use self::edit::{InteriorPass, Touched, refused_value, update_interior};
use self::layers::{Dirty, first_op, write_slot};
use self::pins::{PinTable, Pinned};
use self::round::{
    Acquire, ConeLevel, Fill, FillOutcomes, Held, Plan, Positions, Predict, Resident, RoundTask,
    Targets, Tiles, each_parallel, fill_slot, held_index, keep_first, list_parents, map_parallel,
    nothing_to_write, predict_tile, prefix, prefix_mut, recorded_or_listed, round_tasks,
    vertex_count,
};
use crate::activity::Activity;
use crate::bounds::Book;
use crate::deps::{DepGraph, list_normal_inputs};
use crate::feed::{ChangeFeed, ConsumerId};
#[cfg(test)]
use crate::file::StoreError;
use crate::file::{BaseFile, Loc, ProjectFile, ReadScratch};
use crate::hash::{TileHash, TileMap, TileSet};
use crate::index::{TileBounds, WaveBounds};
use crate::kept::Kept;
use crate::op::OpId;
use crate::pool::{Budget, Freshness, Invalidated, Pool, PoolError, SlotData, SlotId};
use crate::prefetch::{Fetched, Job, Prefetcher, Source};
use crate::slot::{CorrectionSlot, EvaluatedSlot, SIDE};
use crate::stamp::Stamp;

mod bounds;
mod config;
mod edit;
mod error;
mod layers;
mod persist;
mod pins;
mod round;
mod stream;

pub use config::{Evaluation, StoreConfig};
pub use edit::TileEdit;
pub use error::{CacheError, Invariant};
pub(crate) use layers::{Saved, Snapshot};
pub use pins::PinGuard;

/// A [`TileStore`]'s identity among the stores the process opens ([`TileStore::id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StoreId(u64);

impl StoreId {
    /// A store identity no store of this process has had yet.
    fn next() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};

        /// The next identity to hand out.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// What the store computed and read, for tests that count re-evaluations and file reads.
#[cfg(test)]
#[derive(Debug, Default)]
struct EvalLog {
    /// Tiles whose positions were computed, in order.
    positions: Vec<TileKey>,
    /// Tiles whose positions were predicted straight into their slots on the workers, in
    /// order: also in `positions`.
    in_place: Vec<TileKey>,
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
    /// The most predictions held at once past a round's predicting: those kept, and those the
    /// round made to keep, before they are inserted (see [`TileStore::plan_predictions`]).
    predictions_held: usize,
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
/// The pools share [`StoreConfig::budget_bytes`], less the reserved shares (the kept predictions',
/// below, and a brush stroke's state, see [`StoreConfig::stroke_tiles`]), and grow by whole chunks
/// of [`StoreConfig::chunk_bytes`] while the budget allows another; once it is spent, memory moves
/// between them only a chunk at a time. A correction tile that needs a slot (to be read, edited,
/// or evaluated from) takes the first of:
///
/// 1. a free correction slot, or a new chunk if the budget allows one;
/// 2. the slot of a clean correction tile, the clock's pick: it is read again if it is wanted;
/// 3. an evaluated chunk taken back, while the correction pool holds less than a quarter of the
///    pools' share (the budget less the reserved shares) and the evaluated pool keeps its floor
///    without that chunk (the chunks one evaluation pins, see [`StoreConfig::minimum_budget`], plus
///    one for the cache). Of the evaluated chunks without a pin, the one with the fewest tiles in
///    use goes (current, or referenced since the clock last passed); its tiles are recomputed when
///    next wanted;
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
/// slot stays, and so does the correction pool's last chunk when the room is for the evaluated
/// pool (the correction floor, see the module notes): the next evaluation loads its correction
/// tiles there.
///
/// # Kept predictions
///
/// A tile is evaluated in two parts: its prediction from the level below (the subdivided
/// positions and each vertex's tangent frame, most of the work) and its corrections applied in
/// those frames. Editing a tile changes only its corrections, so a tile predicted again because
/// it was edited is predicted from the same positions as before. The store keeps the predictions
/// of the tiles it predicts again in place after an edit of their own corrections alone, their
/// parents unchanged (pinned tiles a brush step edited), and, next time, only applies the
/// corrections, to the same bits: see [`StoreConfig::kept_predictions`] for how many.
///
/// Other tiles' predictions are dropped on the worker that made them, as soon as they are
/// applied: tiles paged in (thousands of them would cycle through the bound for nothing), and
/// pinned tiles predicted again because their parents moved (a view's pins under a coarser
/// edit), which a stroke wouldn't reuse.
///
/// A kept prediction goes when it may no longer be right: when the positions it was predicted
/// from are invalidated (an edit of the tile's parents, or of anything they follow from), when
/// the tile leaves the evaluated pool (its dependency record goes with it, so later edits of its
/// parents couldn't reach the prediction), or when it is the least recently used past the
/// bound. Room is made before a round predicts, so the bound holds with the round's new
/// predictions counted: the store never holds more than [`StoreConfig::kept_predictions`] of
/// them. They are charged to the budget at their worst case, set aside from it before the pools
/// share the rest ([`StoreConfig::kept_prediction_bytes`]); the evaluated pool maps a short
/// chunk, of as many slots as fit, in the room that leaves below a whole chunk, so the share
/// costs the pools its own bytes and no more.
///
/// # Stroke state
///
/// A brush stroke keeps, for each vertex of the tiles it is editing, the correction it first
/// found and the sum of its deltas, in `f32`, and writes their sum, so a dab's delta smaller
/// than a quantization step isn't rounded away. That state is the brush's, not the store's,
/// but it is charged to the budget the same way: [`StoreConfig::stroke_tiles`] tiles' worth at
/// the largest tile's size ([`StoreConfig::stroke_state_bytes`]) is set aside after the kept
/// predictions' share and before the pools share the rest. The store reports the count
/// ([`TileStore::stroke_tiles`]); one stroke at a time allocates it.
///
/// # Memory outside the budget
///
/// [`StoreConfig::budget_bytes`] bounds the two pools, the kept predictions and the stroke
/// state, which hold nearly all of a store's memory. Besides them, and not charged to the
/// budget:
///
/// - the evaluation scratch: one buffer of the largest tile's positions (about 50 KB) per tile of a
///   round not predicted in place, so about 3.2 MB at most (64 of them, once a round of new tiles
///   has been built), kept for the store's life;
/// - the buffers and zstd context the owner thread reads correction tiles with, each evaluation
///   worker's (made the first time a round's tiles are loaded on the workers) and each prefetch
///   worker's (at most four): about 150 KB a thread, so a few MB with dozens of workers, with the
///   prefetched tiles waiting for [`TileStore::poll`] and the spare boxes the workers decode into,
///   about 27 KB each;
/// - what each evaluation worker allocates while it predicts a tile or computes its normals: the
///   gathered parent block, the prediction and its tangent frames, a few tiles' worth (a few
///   hundred KB for the largest tiles), freed when the tile is done (a prediction to be kept is
///   charged to the budget instead);
/// - the thread stacks: Rust's default of 2 MiB of address space reserved per evaluation worker
///   ([`StoreConfig::workers`]) and per prefetch worker, of which only the pages touched are
///   committed;
/// - the bookkeeping: the dependency graph, the pools' indices and the change feed, which grow with
///   the tiles resident and edited, not with their size.
///
/// That is a few MB besides the budget, ten or so with dozens of workers, plus the stacks.
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
    /// The bytes both pools may hold: the configured budget less the reserved shares (the kept
    /// predictions' and the stroke state's).
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
    unsaved: TileMap<Dirty>,
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
    /// [`TileStore::fill_roll_over`]), never with one that didn't edit it. Only resident tiles
    /// have an entry: a tile's goes when its slot is evicted (the correction pool logs its
    /// evictions, see [`TileStore::forget_evicted_ops`]) or removed, so the map never grows
    /// past the pool, and no save walks it.
    known_ops: TileMap<OpId>,
    /// Counts the working files the store has had: a rebase moves it on, so a recovery flush
    /// started before can tell it would finish in the wrong file.
    generation: u64,
    /// Whether the working file is a recovery file whose writer failed (a failed sync, or a
    /// stopped writer thread), as its owner said ([`TileStore::fail_working`]). After a failed
    /// sync, what was appended since the last durable save point may not be on disk, so no save
    /// point may be written there again: [`TileStore::save`] refuses. Cleared when the store
    /// moves to another working file, by a rebase or a roll-over.
    working_failed: bool,
    /// The spatial index's bounds the store keeps in its files: which changed since the last
    /// save, which the working file lacks, and which writes may have made stale (see
    /// [`crate::bounds`]).
    bounds: Book,
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
    /// One tile of positions per tile of a round that isn't predicted in place (see [`ROUND`]
    /// and [`TileStore::build_round`]): the workers write here, then the owner thread copies
    /// each into the slot it takes for it. Grown on demand to the most such tiles a round has
    /// had, never past [`ROUND`] buffers; a round predicted wholly in place (a brush step's)
    /// takes none.
    scratch: Vec<Vec<Vec3>>,
    /// The evaluation workers (see the module notes).
    workers: rayon::ThreadPool,
    /// Buffers for reading correction tiles on the owner thread.
    read_scratch: ReadScratch,
    /// Buffers for reading correction tiles on the workers, one set per worker filling a round's
    /// slots at once (see [`TileStore::fill_round`]). Made on demand, never more than there are
    /// workers.
    worker_read_scratch: Vec<ReadScratch>,
    /// The predictions kept for tiles predicted again in place (see "Kept predictions" on
    /// [`TileStore`]): only of tiles resident in the evaluated pool, and only while their
    /// parents' positions stay as they were.
    kept: Kept<Prediction>,
    /// How many tiles' stroke state the budget set aside ([`StoreConfig::stroke_tiles`]): the
    /// most a brush stroke over this store may keep, which it reads through
    /// [`TileStore::stroke_tiles`]. The store holds none of that state itself.
    stroke_tiles: usize,
    /// This store's identity among every store the process opens ([`TileStore::id`]).
    id: StoreId,
    /// How many times a correction slot's values changed, as of now ([`TileStore::edits`]):
    /// while it stays the same, every tile evaluates to the same bits, so bounds computed from
    /// an evaluation are still the tile's.
    edits: u64,
    /// The pinned tiles an edit of their own corrections alone marked stale, their parents'
    /// positions unchanged: the tiles whose next prediction, made in place, is worth keeping
    /// (see "Kept predictions" on [`TileStore`]). A mark goes when the tile is next predicted,
    /// when its parents' positions change, and when it leaves the evaluated pool, so only
    /// resident tiles have one.
    own_edits: BTreeSet<TileKey>,
    /// What the store has done since it opened, for [`TileStore::activity`].
    activity: Activity,
    /// What was computed, for the tests.
    #[cfg(test)]
    log: EvalLog,
    /// A correction tile whose reads from the working file fail, for tests of error paths.
    #[cfg(test)]
    unreadable: Option<TileKey>,
    /// A tile whose normals computation fails on the workers, after every other task of its
    /// round has written its slot: for tests of that error path.
    #[cfg(test)]
    failing_normals: Option<TileKey>,
    /// Whether [`TileStore::tile_border`] gives `None` for every tile, as it would for a tile
    /// holding a border vertex twice: for tests of the edits' fallback, which no valid mesh
    /// reaches.
    #[cfg(test)]
    without_tile_borders: bool,
}

impl std::fmt::Debug for TileStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileStore")
            .field("file", &self.file)
            .field("base", &self.base.as_ref().map(BaseFile::len))
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

impl TileStore {
    /// Asserts that only tiles resident in the evaluated pool have a kept prediction or an
    /// [`TileStore::own_edits`] mark: that every eviction and every invalidation that freed a
    /// slot dropped its tile's.
    ///
    /// # Panics
    ///
    /// If a tile that isn't resident has one.
    #[cfg(test)]
    fn assert_kept_resident(&self) {
        let mut stale: Vec<&TileKey> = self
            .kept
            .keys()
            .chain(&self.own_edits)
            .filter(|key| self.evaluated.find(key).is_none())
            .collect();
        stale.sort_unstable();
        assert!(
            stale.is_empty(),
            "predictions kept for tiles not resident: {stale:?}"
        );
    }

    /// Asserts that only resident correction tiles have a [`TileStore::known_ops`] entry: that
    /// every eviction and removal dropped its tile's.
    ///
    /// # Panics
    ///
    /// If a tile that isn't resident has one.
    #[cfg(test)]
    fn assert_known_ops_resident(&self) {
        let mut stale: Vec<&TileKey> = self
            .known_ops
            .keys()
            .filter(|key| self.corrections.find(key).is_none())
            .collect();
        stale.sort_unstable();
        assert!(
            stale.is_empty(),
            "ops kept for tiles not resident: {stale:?}"
        );
    }

    /// The base mesh the store's tiles subdivide: level 0 and the topology.
    #[must_use]
    pub fn mesh(&self) -> &BaseMesh {
        &self.mesh
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

    /// How many tiles' stroke state the budget set aside for a brush stroke
    /// ([`StoreConfig::stroke_tiles`]): the most tiles one stroke over this store may keep a
    /// start and summed deltas for. Its bytes are already out of the pools' share, so a stroke
    /// that allocates this much, and no more, stays within the budget.
    #[must_use]
    pub const fn stroke_tiles(&self) -> usize {
        self.stroke_tiles
    }

    /// This store's identity: no other store the process opens has it, whatever its file or
    /// configuration, so state kept beside a store (a brush stroke's, say) can tell it is still
    /// over the same one. Op ids can't: every store of a site starts its ops at the same
    /// sequence number, and a project reopened after its edits were discarded hands out the
    /// discarded ops' ids again.
    #[must_use]
    pub const fn id(&self) -> StoreId {
        self.id
    }

    /// How many times a correction slot's values have changed since the store opened: every
    /// write ([`TileStore::mark_written`]) and every lost tile's write undone, by
    /// [`TileStore::rebase`], [`TileStore::roll_over`] or [`TileStore::lose_left_out`]
    /// ([`TileStore::drop_lost_slot`], [`TileStore::reload_lost`]). An ordinary page-in reads
    /// what the files hold and changes nothing. While it stays the same, every tile evaluates
    /// to the same bits: what [`WaveBounds`] checks before its bounds are used.
    pub(crate) const fn edits(&self) -> u64 {
        self.edits
    }

    /// Counts a change of a correction slot's values ([`TileStore::edits`]).
    ///
    /// Called where the values change, not where evaluations are dropped: an invalidation may
    /// skip a tile nothing cached follows from, and a path that drops evaluations some other
    /// way must still be counted.
    const fn corrections_changed(&mut self) {
        self.edits = self.edits.wrapping_add(1);
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

    /// The copies of every border vertex of tile `key`, found once
    /// ([`geom::Topology::tile_border`]), for an edit's border pass.
    ///
    /// `None` for a tile holding one of its border vertices twice, whose edits then find each
    /// vertex's copies one by one. No mesh [`BaseMesh`] accepts has such a tile: only a face
    /// glued to itself could hold a vertex twice, and a quad that repeats no vertex (which
    /// [`BaseMesh`] checks) can't share an edge or a corner with itself. The fallback is kept
    /// rather than made an error, as it is the definition the fast path must match; the tests
    /// reach it by forcing it.
    fn tile_border(&self, key: &TileKey) -> Option<TileBorder> {
        #[cfg(test)]
        if self.without_tile_borders {
            return None;
        }
        self.mesh.topology().tile_border(key)
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
    /// [`CorrectionError::OutOfTile`], before anything is written, or with
    /// [`CorrectionError::TooLarge`] for an interior value no tile's scale can hold (an interior
    /// vertex has one copy, so nothing is written then either);
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

    /// Rewrites the corrections of each tile of `keys` in turn, as part of the current op: the
    /// same as [`TileStore::edit_tile`] with [`TileEdit::update`] for each tile in order, with
    /// `f(n, i, j, c)` for vertex `(i, j)` of `keys[n]`, and the same bits; faster, as tiles
    /// the caller holds pinned (with [`TileStore::pin`]) have their interiors written on the
    /// store's workers.
    ///
    /// That is sound because a tile's interior vertices have no copies in other tiles, and its
    /// quantization scale covers its interior alone: what an interior write reads and changes
    /// is the tile's own, whichever tile is edited first. So each run of consecutive pinned
    /// tiles (no tile twice) has its interiors written in parallel, each tile's in row-major
    /// order (a value that outgrows the scale re-quantizes the interior, so the order within a
    /// tile matters); then, tile by tile in order, its border vertices are set with their
    /// copies and the tile is marked written where its first write falls in row-major order,
    /// as one tile at a time would. Other tiles are edited one at a time.
    ///
    /// What callers may rely on of the calls to `f`:
    ///
    /// - `f` is called exactly once per vertex of each entry of `keys` (at most once, where an edit
    ///   fails part way), with the vertex's correction as stored just before its write;
    /// - calls run on several threads, one tile's interior all on one thread.
    ///
    /// So `f` may keep state per vertex that only that vertex's call touches (a brush stroke's
    /// summed deltas, say), but must give the same result whatever the thread schedule. A
    /// change here that called `f` twice for a vertex would break such callers.
    ///
    /// # Errors
    ///
    /// As [`TileStore::edit_tile`] and [`TileEdit::update`], for the first tile that fails, in
    /// order: the tiles before it are edited. A tile edited one at a time is left as
    /// [`TileEdit::update`] leaves it; one of a run may have its whole interior pass applied
    /// (every interior vertex up to the one the pass failed at, if any), even where a border
    /// vertex earlier in row-major order failed. Tiles after it in its run may have had their
    /// interiors written too. Every tile of the run whose interior was written is marked
    /// written (dirty, on the change feed with the op) and its evaluations dropped, as for a
    /// whole edit, each of them even if marking another fails; the first error is returned.
    pub fn update_tiles(
        &mut self,
        keys: &[TileKey],
        f: impl Fn(usize, u32, u32, Vec3) -> Option<Vec3> + Sync,
    ) -> Result<(), CacheError> {
        self.sync_pins()?;
        let mut start = 0;
        while let Some(&next) = keys.get(start) {
            let mut run: Vec<TileKey> = Vec::new();
            for key in keys.get(start..).unwrap_or_default() {
                if run.contains(key) || !self.corrections.is_pinned(key) {
                    break;
                }
                run.push(*key);
            }
            if run.len() < 2 {
                self.edit_tile(next, |edit| edit.update(|i, j, c| f(start, i, j, c)))??;
                start += 1;
            } else {
                self.update_run(start, &run, &f)?;
                start += run.len();
            }
        }
        Ok(())
    }

    /// [`TileStore::update_tiles`] for `run`, tiles `first..` of its keys, each pinned by the
    /// caller and none twice: their interiors in parallel, then each tile's border and marking
    /// in order.
    ///
    /// # Errors
    ///
    /// As [`TileStore::update_tiles`].
    fn update_run(
        &mut self,
        first: usize,
        run: &[TileKey],
        f: &(impl Fn(usize, u32, u32, Vec3) -> Option<Vec3> + Sync),
    ) -> Result<(), CacheError> {
        self.check_keys_and_op(run)?;
        let ids = run
            .iter()
            .map(|key| {
                self.corrections
                    .lookup(key)
                    .map(|(id, _)| id)
                    .ok_or(Invariant::StaleHandle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut slots = self
            .corrections
            .get_many_mut(&ids)
            .ok_or(Invariant::StaleHandle)?;
        let interiors: Vec<InteriorPass> = self.workers.install(|| {
            run.par_iter()
                .zip(slots.par_iter_mut())
                .enumerate()
                .map(|(k, (key, slot))| {
                    update_interior(slot, key.side(), |i, j, c| f(first + k, i, j, c))
                })
                .collect()
        });
        let mut outcome = Ok(());
        for (k, (&key, pass)) in run.iter().zip(interiors).enumerate() {
            if outcome.is_err() {
                // Past the failing tile: one whose interior was written is marked and dropped,
                // each of them whatever marking another does, and the first error is kept.
                if pass.first_row.is_some() {
                    #[expect(
                        clippy::redundant_closure_for_method_calls,
                        reason = "the method path names one edit lifetime; edit_tile's closure \
                                  must take any"
                    )]
                    let marked = self
                        .edit_tile(key, |edit| edit.mark_own_written())
                        .and_then(std::convert::identity);
                    keep_first(&mut outcome, marked);
                }
                continue;
            }
            outcome = self
                .edit_tile(key, |edit| {
                    edit.update_border(pass, |i, j, c| f(first + k, i, j, c))
                })
                .and_then(std::convert::identity);
        }
        outcome
    }

    /// What the store has done since it opened: tiles loaded and evaluated, and evicted to make
    /// room (see [`Activity`]). Two counts around a call, through [`Activity::since`], give what
    /// that call did.
    #[must_use]
    pub const fn activity(&self) -> Activity {
        self.activity
    }

    /// `key`'s evaluated positions as cached, if they are current: no evaluation, and the read
    /// doesn't count as a use of the tile (which [`TileStore::evaluate`] does, for eviction).
    /// `None` if they aren't current; evaluate them first.
    ///
    /// With [`TileStore::par_map`], for reading many tiles' positions on the workers.
    #[must_use]
    pub fn cached_positions(&self, key: &TileKey) -> Option<&[Vec3]> {
        let id = self.fresh_slot(key)?;
        prefix(&self.evaluated.get(id)?.positions, vertex_count(key)).ok()
    }

    /// `key`'s unit normals as cached, if they are current, as [`TileStore::cached_positions`]
    /// gives positions (current normals go with current positions). `None` if they aren't
    /// current; compute them with [`TileStore::normals`] or [`TileStore::evaluate_many`] first.
    #[must_use]
    pub fn cached_normals(&self, key: &TileKey) -> Option<&[Vec3]> {
        let id = self.fresh_slot(key)?;
        if !self.deps.has_normals(key) {
            return None;
        }
        prefix(&self.evaluated.get(id)?.normals, vertex_count(key)).ok()
    }

    /// The tiles `key`'s normals are computed from, the tile itself included, in key order: the
    /// tile, its apron and the rings at its corners, as [`normal_tiles`](geom::normal_tiles) lists
    /// them for [`Region::of_tile`]. The list is structural, so it is read back from what the
    /// store recorded when it last computed the tile's normals, and listed from the mesh only
    /// for a tile it hasn't.
    ///
    /// Neighbourhood is symmetric, so these are also the tiles whose normals read `key`'s
    /// positions.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Eval`] (not for a valid key).
    pub fn normal_inputs(&self, key: &TileKey) -> Result<Cow<'_, [TileKey]>, CacheError> {
        self.check_key(key)?;
        Ok(self.deps.normal_inputs_or_list(&self.mesh, key)?)
    }

    /// `key`'s evaluated slot if its positions are current, found without counting as a use.
    fn fresh_slot(&self, key: &TileKey) -> Option<SlotId<EvaluatedSlot>> {
        (self.evaluated.freshness(key)? == Freshness::Fresh)
            .then(|| self.evaluated.find(key))
            .flatten()
    }

    /// `f` of every item of `items`, in order, computed on the store's evaluation workers
    /// ([`StoreConfig::workers`]); a single item is computed on the calling thread.
    ///
    /// For per-tile work over what the store has cached (with
    /// [`TileStore::cached_positions`] and [`TileStore::cached_normals`]) that is independent
    /// tile by tile: the results come back in item order however the work was spread.
    pub fn par_map<T: Sync, R: Send>(
        &self,
        items: &[T],
        f: impl Fn(&T) -> R + Sync + Send,
    ) -> Vec<R> {
        map_parallel(&self.workers, items, f)
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
    /// positions of its own level's [`normal_tiles`](geom::normal_tiles) (the tile, its apron and
    /// the rings at its corners).
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
        self.evaluate_many_into(keys, what, None)
    }

    /// [`TileStore::evaluate_many`] with [`Evaluation::Normals`], which also hands back the
    /// bounds of every tile whose normals it computed: a box and a cone over the tile's
    /// positions and normals, as [`crate::SpatialIndex::bounds_of`] gives them, computed on the
    /// worker that computed the normals, while they are in its cache.
    ///
    /// For [`crate::SpatialIndex::refit_with`], which uses them in place of reading every tile
    /// back, as long as no correction is written in between (see [`WaveBounds`]). A tile whose
    /// normals were current already, or that has no finite position, gets none.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`]. The bounds computed before an error are lost.
    #[must_use = "the bounds are what this adds to evaluate_many: pass them to refit_with"]
    pub fn evaluate_many_with_bounds(
        &mut self,
        keys: &[TileKey],
    ) -> Result<WaveBounds, CacheError> {
        let mut bounds = TileMap::default();
        self.evaluate_many_into(keys, Evaluation::Normals, Some(&mut bounds))?;
        Ok(WaveBounds {
            store: Some(self.id),
            edits: self.edits,
            bounds,
        })
    }

    /// [`TileStore::evaluate_many`], putting the bounds of every tile whose normals are
    /// computed in `bounds`, if given (see [`TileStore::evaluate_many_with_bounds`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`].
    fn evaluate_many_into(
        &mut self,
        keys: &[TileKey],
        what: Evaluation,
        mut bounds: Option<&mut TileMap<TileBounds>>,
    ) -> Result<(), CacheError> {
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
            Evaluation::Normals => store.normals_batch(batch, bounds.as_deref_mut()),
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
        let corrections = self
            .corrections
            .evict_all(|key, slot| write_slot(file, unsaved, key, slot).map(drop));
        self.forget_evicted_ops();
        Ok(corrections?)
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
        let mut seen = TileSet::with_capacity_and_hasher(keys.len(), TileHash);
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
        #[cfg(test)]
        {
            self.assert_known_ops_resident();
            self.assert_kept_resident();
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
        let acquired = self.corrections.acquire_clean(key, &mut self.budget);
        self.forget_evicted_ops();
        let id = match acquired {
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
    /// evaluated slots the pool can reach besides those pinned, the `pending` targets still to
    /// come (which [`TileStore::pin`] has counted on keeping room for) and an evaluation's own
    /// headroom; at least one, at most [`MAX_BATCH`].
    ///
    /// The evaluated pool reaches the budget less what the correction pool keeps from it: the
    /// correction floor (one chunk, see [`TileStore::make_room`]) and the chunks its pinned
    /// slots, which can't go, and its dirty ones, which go only after a write, fill. Those are
    /// counted as if packed into chunks, not by the chunks they sit in: where a tile sits
    /// depends on when prefetched tiles were installed, and the count must not. What the
    /// evaluated pool reaches is whole chunks and, in the room below a whole one, the short chunk
    /// it would map there ([`Pool::short_slots`]).
    ///
    /// It depends on the budget and the counts of pinned and dirty slots alone, never on the
    /// thread count or where tiles sit, so the batches, like the rounds, are the same whatever
    /// the workers.
    fn batch_size(&self, pending: usize) -> usize {
        let held = self.corrections.pinned_slots() + self.unsaved.len();
        let held_chunks = held.div_ceil(self.corrections.per_chunk() as usize);
        let kept = (held_chunks + 1) * self.corrections.chunk_bytes();
        let reachable = self.budget.limit().saturating_sub(kept);
        let chunk = self.evaluated.chunk_bytes().max(1);
        let (chunks, rest) = (reachable / chunk, reachable % chunk);
        let slots = chunks
            .saturating_mul(self.evaluated.per_chunk() as usize)
            .saturating_add(self.evaluated.short_slots(rest) as usize);
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

    /// Checks each of `keys` as [`TileStore::check_key`] does, and that an op is current or can
    /// be begun, before anything of an edit is written.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OpsExhausted`].
    fn check_keys_and_op(&mut self, keys: &[TileKey]) -> Result<(), CacheError> {
        keys.iter().try_for_each(|key| self.check_key(key))?;
        self.edit_op().map(drop)
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
        self.load_correction_as(key, Acquire::Any)
    }

    /// [`TileStore::load_correction`], getting a slot for a tile that isn't resident as
    /// `acquire` allows.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]; with [`Acquire::Clean`], [`CacheError::Full`] when
    /// no free slot, new chunk or clean tile's slot is to be had.
    fn load_correction_as(
        &mut self,
        key: TileKey,
        acquire: Acquire,
    ) -> Result<SlotId<CorrectionSlot>, CacheError> {
        if let Some((id, _)) = self.corrections.lookup(&key) {
            return Ok(id);
        }
        let fill = self.take_correction_slot(key, acquire)?;
        let slot = self
            .corrections
            .get_mut(fill.id)
            .ok_or(Invariant::StaleHandle)?;
        let filled = fill_slot(
            &self.file,
            self.base.as_ref(),
            &fill,
            slot,
            &mut self.read_scratch,
        );
        #[cfg(test)]
        let filled = self.as_tested(key, filled);
        self.settle_fill(&fill, filled)?;
        Ok(fill.id)
    }

    /// A slot for `key`'s correction tile, which isn't resident, taken as `acquire` allows, and
    /// where its record is: the first half of loading it, which [`fill_slot`] and
    /// [`TileStore::settle_fill`] finish.
    ///
    /// # Errors
    ///
    /// As [`TileStore::acquire_correction`] or, with [`Acquire::Clean`],
    /// [`CacheError::Full`] when no free slot, new chunk or clean tile's slot is to be had.
    fn take_correction_slot(&mut self, key: TileKey, acquire: Acquire) -> Result<Fill, CacheError> {
        let id = match acquire {
            Acquire::Any => self.acquire_correction(key),
            Acquire::Clean => self
                .corrections
                .acquire_clean(key, &mut self.budget)
                .map_err(CacheError::from),
        };
        // Taking a slot may have evicted others, whether or not it succeeded.
        self.forget_evicted_ops();
        let id = id?;
        self.activity.corrections_loaded += 1;
        #[cfg(test)]
        self.log.file_reads.push(key);
        Ok(Fill {
            key,
            id,
            found: self.locate(&key),
        })
    }

    /// `filled`, the outcome of filling `key`'s slot, as the tests see it: an error for the
    /// tile they made unreadable ([`TileStore::unreadable`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] for the unreadable tile; else `filled`'s.
    #[cfg(test)]
    fn as_tested(
        &self,
        key: TileKey,
        filled: Result<Option<OpId>, CacheError>,
    ) -> Result<Option<OpId>, CacheError> {
        if self.unreadable == Some(key) {
            Err(StoreError::from(std::io::Error::other("an unreadable tile, for the test")).into())
        } else {
            filled
        }
    }

    /// The last half of loading a correction tile: records the op of the record its slot was
    /// filled from (see [`TileStore::known_ops`]), or, if filling it failed, gives the slot back
    /// (it is clean, and unpinned by the caller) and returns the error.
    ///
    /// # Errors
    ///
    /// `filled`'s error; [`CacheError::Internal`] if the slot can't be given back.
    fn settle_fill(
        &mut self,
        fill: &Fill,
        filled: Result<Option<OpId>, CacheError>,
    ) -> Result<(), CacheError> {
        match filled {
            Ok(Some(op)) => {
                self.known_ops.insert(fill.key, op);
                Ok(())
            }
            Ok(None) => {
                self.known_ops.remove(&fill.key);
                Ok(())
            }
            Err(e) => {
                // The slot holds no tile: give it back.
                self.corrections.remove(&fill.key)?;
                Err(e)
            }
        }
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
    /// than a quarter of the pools' share (the budget less the reserved shares), and the
    /// evaluated pool keeps its floor (its share of the minimum budget plus one chunk) without
    /// the chunk, and has a chunk without pins to give.
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
            let shrunk = self
                .corrections
                .shrink_to(&mut self.budget, bytes, |key, slot| {
                    write_slot(file, unsaved, key, slot).map(drop)
                });
            self.forget_evicted_ops();
            shrunk?;
        }
        #[cfg(test)]
        self.assert_known_ops_resident();
        Ok(())
    }

    /// Drops the [`TileStore::known_ops`] of the correction tiles the pool has evicted since
    /// the last call, so that only resident tiles have one. A tile resident again keeps its
    /// entry: no flow reloads a tile between its eviction and the next drain today (each slot
    /// taken drains before it is filled), so that check only guards a future one.
    ///
    /// Called wherever the pool may evict: after taking a slot (to load a tile or install a
    /// prefetched one), evicting all, or giving chunks back, each whether or not it succeeded;
    /// and at a rebase or roll-over, which so find nothing left to drop. Tests assert after
    /// those, and after every poll, that only resident tiles have an op.
    fn forget_evicted_ops(&mut self) {
        for key in self.corrections.drain_evicted() {
            self.activity.corrections_evicted += 1;
            if self.corrections.find(&key).is_none() {
                self.known_ops.remove(&key);
            }
        }
    }

    /// Tells the dependency graph which evaluated tiles the pool evicted.
    fn note_evictions(&mut self) {
        for key in self.evaluated.drain_evicted() {
            self.activity.evaluated_evicted += 1;
            self.kept.remove(&key);
            self.own_edits.remove(&key);
            self.deps.left_pool(key);
        }
    }

    /// Writes `c` at `(i, j)` of `key`'s correction slot `id` (resident and pinned) as part of
    /// `op`, and marks the slot written (see [`TileStore::mark_written`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::TooLarge`] or
    /// [`CorrectionError::NotFinite`] for a value no tile can hold (the slot is unchanged);
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
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        view.set(i, j, c).map_err(refused_value)?;
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
        self.corrections_changed();
        self.last_write = self.last_write.next();
        self.unsaved.insert(
            key,
            Dirty {
                op,
                stamp: self.last_write,
            },
        );
        self.feed.push(key, op);
        self.bounds.note_write(key);
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
    ///
    /// A tile whose parents' positions changed is predicted from other positions now: its kept
    /// prediction goes. A pinned tile edited alone, its parents as they were, keeps its kept
    /// prediction, which its corrections don't change, and is marked for its next one to be
    /// kept ([`TileStore::own_edits`]). A tile whose parents the graph doesn't record (an
    /// untracked node, see [`crate::deps`]) counts as moved: the cascade takes it to follow
    /// from every coarser edit, and nothing says which.
    fn invalidate(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        #[cfg(test)]
        {
            self.log.invalidations += 1;
        }
        let edited: BTreeSet<TileKey> = edited.into_iter().collect();
        let cascade = self.deps.cascade(edited.iter().copied());
        for (key, changed) in cascade.normals {
            self.deps.stale_normals(key, changed);
        }
        let moved: BTreeSet<TileKey> = cascade
            .positions
            .iter()
            .filter(|key| {
                self.deps.parents(key).is_none_or(|parents| {
                    parents
                        .iter()
                        .any(|parent| cascade.positions.contains(parent))
                })
            })
            .copied()
            .collect();
        for key in cascade.positions {
            let parents_moved = moved.contains(&key);
            if parents_moved {
                self.kept.remove(&key);
            }
            match self.evaluated.invalidate(&key) {
                Invalidated::Marked if edited.contains(&key) && !parents_moved => {
                    self.own_edits.insert(key);
                }
                Invalidated::Marked => {
                    self.own_edits.remove(&key);
                }
                Invalidated::Freed | Invalidated::Absent => {
                    self.kept.remove(&key);
                    self.own_edits.remove(&key);
                    self.deps.left_pool(key);
                }
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
    /// [`parent_tiles`](geom::parent_tiles), listed on the workers (see [`recorded_or_listed`]).
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
        let mesh = &*self.mesh;
        let found = recorded_or_listed(
            &self.workers,
            &keys,
            |key| self.deps.parents(key).map(<[TileKey]>::to_vec),
            |key| list_parents(mesh, key),
        )?;
        Ok(keys.into_iter().zip(found).collect())
    }

    /// The tiles the normals of each of `keys` are computed from, the tile itself included:
    /// [`DepGraph::normal_inputs_or_list`]'s rule, with the lists not recorded made on the
    /// workers (see [`recorded_or_listed`]).
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
        let mesh = &*self.mesh;
        recorded_or_listed(
            &self.workers,
            keys,
            |key| {
                self.deps
                    .normal_inputs(key)
                    .map(|inputs| inputs.iter().copied().collect())
            },
            |key| Ok(list_normal_inputs(mesh, key)?),
        )
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
    /// `tiles`, cut at the first that can't be had cheaply (at least the first). Returns their
    /// slots, in order.
    ///
    /// The first tile gets its slot as the memory policy orders it, as one tile at a time
    /// would. The others take a free slot, a new chunk or a clean tile's slot only
    /// ([`Acquire::Clean`]): the round's pins keep the earlier tiles' slots from being reused,
    /// so going further would write back dirty tiles, or take evaluated chunks, that a tile at
    /// a time never would. The round is cut at the first refused instead, and that tile starts
    /// the next round.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]: any error for the first tile, any but
    /// [`CacheError::Full`] for the others; nothing stays pinned then.
    fn load_round(
        &mut self,
        tiles: &[(TileKey, &[TileKey])],
    ) -> Result<Vec<SlotId<CorrectionSlot>>, CacheError> {
        let mut ids = Vec::with_capacity(tiles.len().min(ROUND));
        let mut fills = Vec::new();
        for &(key, _) in tiles.iter().take(ROUND) {
            let taken = if let Some((id, _)) = self.corrections.lookup(&key) {
                Ok(id)
            } else {
                let acquire = if ids.is_empty() {
                    Acquire::Any
                } else {
                    Acquire::Clean
                };
                self.take_correction_slot(key, acquire).map(|fill| {
                    fills.push(fill);
                    fill.id
                })
            };
            let pinned = taken.and_then(|id| {
                self.corrections.pin(id)?;
                Ok(id)
            });
            match pinned {
                Ok(id) => ids.push(id),
                Err(CacheError::Full) if !ids.is_empty() => break,
                Err(e) => {
                    // A slot just taken and not pinned holds no tile: it goes with the rest.
                    self.unpin_corrections(&ids)?;
                    self.give_back_fills(&fills)?;
                    return Err(e);
                }
            }
        }
        let outcomes = match self.fill_round(&fills) {
            Ok(outcomes) => outcomes,
            Err(e) => {
                self.unpin_corrections(&ids)?;
                self.give_back_fills(&fills)?;
                return Err(e);
            }
        };
        for (n, (fill, filled)) in fills.iter().zip(outcomes).enumerate() {
            #[cfg(test)]
            let filled = self.as_tested(fill.key, filled);
            if filled.is_err() {
                // As loading one tile at a time would have: the tiles before it stay loaded,
                // and those after it, never loaded, give their slots back.
                self.unpin_corrections(&ids)?;
                let later = fills.get(n + 1..).unwrap_or_default();
                self.give_back_fills(later)?;
            }
            self.settle_fill(fill, filled)?;
        }
        Ok(ids)
    }

    /// Fills the slots of `fills` (taken and pinned by [`TileStore::load_round`]) from their
    /// records, on the workers: each worker reads, checks and decodes a run of consecutive
    /// tiles with read buffers of its own. Returns each tile's outcome, in the order of `fills`,
    /// for [`TileStore::settle_fill`]; a single tile is filled on the calling thread.
    ///
    /// The workers only read the files and write the slots handed to them, each its own: what
    /// lands in a slot is the same whichever worker fills it.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale; [`CacheError::Store`] if a worker's read
    /// buffers can't be made.
    fn fill_round(&mut self, fills: &[Fill]) -> Result<FillOutcomes, CacheError> {
        let ids: Vec<SlotId<CorrectionSlot>> = fills.iter().map(|fill| fill.id).collect();
        let parts = fills.len().min(self.workers.current_num_threads());
        if parts <= 1 {
            let mut outcomes = Vec::with_capacity(fills.len());
            for fill in fills {
                let slot = self
                    .corrections
                    .get_mut(fill.id)
                    .ok_or(Invariant::StaleHandle)?;
                let scratch = &mut self.read_scratch;
                outcomes.push(fill_slot(
                    &self.file,
                    self.base.as_ref(),
                    fill,
                    slot,
                    scratch,
                ));
            }
            return Ok(outcomes);
        }
        while self.worker_read_scratch.len() < parts {
            self.worker_read_scratch.push(ReadScratch::new()?);
        }
        let (file, base) = (&self.file, self.base.as_ref());
        let scratch = self
            .worker_read_scratch
            .get_mut(..parts)
            .ok_or(Invariant::StaleHandle)?;
        let mut slots = self
            .corrections
            .get_many_mut(&ids)
            .ok_or(Invariant::StaleHandle)?;
        let run = fills.len().div_ceil(parts);
        let runs: Vec<FillOutcomes> = self.workers.install(|| {
            fills
                .par_chunks(run)
                .zip(slots.par_chunks_mut(run))
                .zip(scratch.par_iter_mut())
                .map(|((fills, slots), scratch)| {
                    fills
                        .iter()
                        .zip(slots)
                        .map(|(fill, slot)| fill_slot(file, base, fill, slot, scratch))
                        .collect()
                })
                .collect()
        });
        Ok(runs.into_iter().flatten().collect())
    }

    /// Gives back the slots of `fills`, taken for tiles never loaded: each is unpinned if it is
    /// pinned, and freed.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle (the slots are the caller's, so it doesn't
    /// happen).
    fn give_back_fills(&mut self, fills: &[Fill]) -> Result<(), CacheError> {
        for fill in fills {
            if self.corrections.is_pinned(&fill.key) {
                self.corrections.unpin(fill.id)?;
            }
            self.corrections.remove(&fill.key)?;
        }
        Ok(())
    }

    /// Builds one round: predicts every tile of `round` on the workers, from `below` and its
    /// pinned correction slot (`corrections`, in the same order), unpins the correction slots,
    /// then installs each tile in key order, pinning it and adding it to `held`.
    ///
    /// A tile whose slot is pinned (a stale tile recomputed in place, as a brush step's tiles
    /// are: pinned by the step, so kept when their edit invalidated them) is predicted straight
    /// into that slot on the workers, and installing it is only its bookkeeping
    /// ([`TileStore::positions_installed`]). Only a pinned slot is safe to write ahead of the
    /// installs: a stale slot nothing pins any more (its pins dropped after the edit) is the
    /// clock's first victim, and an earlier tile of the round taking a slot as it installs
    /// would take it. Any other tile, such a one included, is predicted into a scratch buffer,
    /// and the owner copies it into a slot it takes as it installs the tile
    /// ([`TileStore::store_positions`]). Either way the pools see the same calls in the same
    /// order as when every tile went through scratch: a lookup of each tile's slot, or a slot
    /// taken for it, in key order, after the correction slots are unpinned. Slots are not taken
    /// earlier to predict into them: that would hold the round's correction pins while the
    /// evaluated pool makes room (see below), and fault fresh chunks' pages in on the workers,
    /// which once made paging in slower.
    ///
    /// The correction slots are unpinned before anything is installed, as an evaluation one
    /// tile at a time holds no correction slot while it takes an evaluated one: the evaluated
    /// pool may need a correction chunk back to make room (see [`StoreConfig::minimum_budget`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: the first error of the round in key order. The tiles
    /// installed before it stay in `held`, pinned; the correction slots are unpinned either way.
    /// A tile predicted in place but not installed keeps its slot, stale, to be recomputed.
    fn build_round(
        &mut self,
        round: &[(TileKey, &[TileKey])],
        corrections: &[SlotId<CorrectionSlot>],
        below: &Held,
        held: &mut Held,
    ) -> Result<(), CacheError> {
        let slots: Vec<Option<SlotId<EvaluatedSlot>>> = round
            .iter()
            .map(|(key, _)| {
                self.evaluated
                    .find(key)
                    .filter(|_| self.evaluated.is_pinned(key))
            })
            .collect();
        let through_scratch = slots.iter().filter(|slot| slot.is_none()).count();
        let mut scratch = self.take_scratch(through_scratch);
        let predicted = self.predict_round(round, corrections, &slots, below, &mut scratch);
        let released = self.unpin_corrections(corrections);
        let built = predicted
            .and_then(|reused| released.map(|()| reused))
            .and_then(|reused| {
                let mut buffers = scratch.iter();
                round.iter().zip(&slots).zip(reused).try_for_each(
                    |((&(key, parents), &slot), reused)| {
                        let id = if let Some(id) = slot {
                            self.install_in_place(key, id, parents)?;
                            if reused {
                                self.activity.predictions_reused += 1;
                            }
                            id
                        } else {
                            let positions = buffers.next().ok_or(Invariant::ShortBuffer {
                                expected: through_scratch,
                                got: scratch.len(),
                            })?;
                            self.store_positions(key, parents, positions)?
                        };
                        self.evaluated.pin(id)?;
                        held.push((key, id));
                        Ok(())
                    },
                )
            });
        self.scratch = scratch;
        built
    }

    /// Predicts every tile of `round` on the workers, from the pinned tiles of `below` and its
    /// correction slot (`corrections`, in the same order): straight into the tile's evaluated
    /// slot where `slots` names one (in the same order), else into the next buffer of
    /// `scratch`, which holds one for each tile without a slot.
    ///
    /// The slots written and the slots read are borrowed together, in one
    /// [`Pool::get_many_mut`], which hands out each slot once: the round's slots stay mutable,
    /// one per task, and its tiles' parents (each once, found in `below`) are reborrowed shared
    /// for every task to read. Only the parents are borrowed, not all of `below`: a wide level
    /// is built in many rounds, and borrowing the whole level below for each would cost each
    /// round as much as the level is wide. The round's tiles are a level finer than their
    /// parents, so no slot is named twice. Nothing of the pools' bookkeeping, the budget or the
    /// dependency graph changes; the owner installs the tiles afterwards, in key order.
    ///
    /// Of the tiles predicted in place, some reuse their kept prediction and some keep the one
    /// they make ([`TileStore::plan_predictions`]); every other prediction is dropped on its
    /// worker as soon as it is applied. Returns, in the round's order, which tiles reused one,
    /// for the installs to count.
    ///
    /// # Errors
    ///
    /// The first tile's error in key order: [`CacheError::Eval`] (not for a valid mesh), or
    /// [`CacheError::Internal`] for slots that can't be borrowed together (a stale handle or a
    /// slot named twice, [`Invariant::SlotsNotDisjoint`]), a parent not in `below`
    /// ([`Invariant::NotHeld`]), a broken slot, or too few buffers ([`Invariant::ShortBuffer`]). A
    /// slot whose tile failed holds whatever its prediction had written: it stays stale.
    fn predict_round(
        &mut self,
        round: &[(TileKey, &[TileKey])],
        corrections: &[SlotId<CorrectionSlot>],
        slots: &[Option<SlotId<EvaluatedSlot>>],
        below: &Held,
        scratch: &mut [Vec<Vec3>],
    ) -> Result<Vec<bool>, CacheError> {
        let tasks = round_tasks(round, corrections);
        if slots.iter().all(Option::is_none) {
            for (key, _) in round {
                self.own_edits.remove(key);
            }
            // Nothing to write in place: the workers read the pool shared, as paging in always
            // has, and write only scratch.
            let (mesh, pool) = (&*self.mesh, &self.corrections);
            let tiles = Tiles::Shared {
                pool: &self.evaluated,
                held: below,
            };
            each_parallel(&self.workers, &tasks, scratch, |&task, out| {
                predict_tile(mesh, pool, tiles, task, Predict::Drop, out).map(drop)
            })?;
            return Ok(vec![false; round.len()]);
        }
        let plan = self.plan_predictions(round, slots);
        let (mesh, pool) = (&*self.mesh, &self.corrections);
        let mut parent_keys: Vec<TileKey> = round
            .iter()
            .flat_map(|&(_, parents)| parents.iter().copied())
            .collect();
        parent_keys.sort_unstable();
        parent_keys.dedup();
        let read_from: Held = parent_keys
            .into_iter()
            .map(|key| {
                let at = held_index(below, &key)?;
                below.get(at).copied().ok_or(Invariant::NotHeld(key))
            })
            .collect::<Result<_, Invariant>>()?;
        let own: Vec<SlotId<EvaluatedSlot>> = slots.iter().flatten().copied().collect();
        let ids: Vec<SlotId<EvaluatedSlot>> = own
            .iter()
            .copied()
            .chain(read_from.iter().map(|&(_, id)| id))
            .collect();
        let mut written = self
            .evaluated
            .get_many_mut(&ids)
            .ok_or(Invariant::SlotsNotDisjoint)?;
        let read = written.split_off(own.len());
        let parents: Positions<'_> = read_from
            .iter()
            .zip(read)
            .map(|(&(key, _), slot)| {
                let slot: &EvaluatedSlot = slot;
                (key, slot.positions.as_slice())
            })
            .collect();
        let mut written = written.into_iter();
        let scratch_count = scratch.len();
        let mut buffers = scratch.iter_mut();
        let mut outs: Vec<(&mut [Vec3], Option<Prediction>)> = slots
            .iter()
            .map(|own_slot| match own_slot {
                Some(_) => written.next().map(|slot| slot.positions.as_mut_slice()),
                None => buffers.next().map(Vec::as_mut_slice),
            })
            .map(|out| out.map(|out| (out, None)))
            .collect::<Option<_>>()
            .ok_or(Invariant::ShortBuffer {
                expected: slots.len() - own.len(),
                got: scratch_count,
            })?;
        let tasks: Vec<(RoundTask, Predict<'_>)> = tasks
            .into_iter()
            .zip(&plan)
            .map(|(task, &(key, how))| {
                let how = match how {
                    Plan::Reuse => self.kept.get(&key).map_or(Predict::Drop, Predict::Reuse),
                    Plan::Keep => Predict::Keep,
                    Plan::Drop => Predict::Drop,
                };
                (task, how)
            })
            .collect();
        let reused = tasks
            .iter()
            .map(|(_, how)| matches!(how, Predict::Reuse(_)))
            .collect();
        let tiles = Tiles::Split(&parents);
        each_parallel(
            &self.workers,
            &tasks,
            &mut outs,
            |&(task, how), (out, made)| {
                *made = predict_tile(mesh, pool, tiles, task, how, out)?;
                Ok(())
            },
        )?;
        let made: Vec<Option<Prediction>> = outs.into_iter().map(|(_, made)| made).collect();
        self.keep_predictions(&plan, made);
        Ok(reused)
    }

    /// Keeps the predictions a round `made` for its tiles of `plan` (in the same order): those
    /// [`TileStore::plan_predictions`] made room for.
    fn keep_predictions(&mut self, plan: &[(TileKey, Plan)], made: Vec<Option<Prediction>>) {
        #[cfg(test)]
        {
            let held = self.kept.len() + made.iter().flatten().count();
            self.log.predictions_held = self.log.predictions_held.max(held);
        }
        for (&(key, _), prediction) in plan.iter().zip(made) {
            if let Some(prediction) = prediction {
                self.kept.insert(key, prediction);
            }
        }
    }

    /// What becomes of each prediction of `round` (whose evaluated slots are `slots`, in the
    /// same order), so that the kept predictions never number more than their bound, counting
    /// those the round makes (see "Kept predictions" on [`TileStore`]).
    ///
    /// A tile predicted in place with a kept prediction reuses it, which counts as its use. One
    /// predicted in place after an edit of its own corrections alone, its parents' positions as
    /// before ([`TileStore::own_edits`]), keeps the prediction it makes, as many of them as fit
    /// beside the ones reused, in key order; room is made for them now, the least recently used
    /// going first. Every other prediction is dropped on the worker that made it. The round's
    /// tiles lose their [`TileStore::own_edits`] mark either way: they are being predicted.
    fn plan_predictions(
        &mut self,
        round: &[(TileKey, &[TileKey])],
        slots: &[Option<SlotId<EvaluatedSlot>>],
    ) -> Vec<(TileKey, Plan)> {
        let mut plan: Vec<(TileKey, Plan)> = round
            .iter()
            .zip(slots)
            .map(|(&(key, _), slot)| {
                let own_edit = self.own_edits.remove(&key);
                let in_place = slot.is_some();
                let how = if in_place && self.kept.get(&key).is_some() {
                    Plan::Reuse
                } else if in_place && own_edit {
                    Plan::Keep
                } else {
                    Plan::Drop
                };
                (key, how)
            })
            .collect();
        let reused = plan.iter().filter(|(_, how)| *how == Plan::Reuse).count();
        let mut room = self.kept.capacity().saturating_sub(reused);
        for (key, how) in &mut plan {
            match how {
                Plan::Reuse => self.kept.used(key),
                Plan::Keep if room > 0 => room -= 1,
                Plan::Keep => *how = Plan::Drop,
                Plan::Drop => {}
            }
        }
        let keeping = plan.iter().filter(|(_, how)| *how == Plan::Keep).count();
        self.kept.make_room(keeping);
        plan
    }

    /// Installs `key`'s positions, predicted straight into its slot `id`: looks the slot up,
    /// as [`TileStore::store_positions`] finds a stale slot (marking it used), and records the
    /// tile as predicted from `parents` ([`TileStore::positions_installed`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if `key`'s slot is no longer `id` (only pinned slots are
    /// predicted in place, and nothing unpins them while the round is built, so this doesn't
    /// happen).
    fn install_in_place(
        &mut self,
        key: TileKey,
        id: SlotId<EvaluatedSlot>,
        parents: &[TileKey],
    ) -> Result<(), CacheError> {
        match self.evaluated.lookup(&key) {
            Some((found, _)) if found == id => {}
            _ => return Err(Invariant::StaleHandle.into()),
        }
        self.positions_installed(key, id, parents)?;
        #[cfg(test)]
        self.log.in_place.push(key);
        Ok(())
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
        // A stale slot still here (one that wasn't pinned when the round began, and that no
        // earlier tile of the round took) is overwritten in place: its handles stay valid.
        let id = match self.evaluated.lookup(&key) {
            Some((id, _)) => id,
            None => self.acquire_evaluated(key)?,
        };
        let n = vertex_count(&key);
        let slot = self.evaluated.get_mut(id).ok_or(Invariant::StaleHandle)?;
        prefix_mut(&mut slot.positions, n)?.copy_from_slice(prefix(positions, n)?);
        self.positions_installed(key, id, parents)?;
        Ok(id)
    }

    /// The bookkeeping of `key`'s new positions, written into its slot `id`: marks the slot
    /// fresh and records the tile as predicted from `parents`, which the caller holds pinned
    /// (see [`TileStore::store_positions`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn positions_installed(
        &mut self,
        key: TileKey,
        id: SlotId<EvaluatedSlot>,
        parents: &[TileKey],
    ) -> Result<(), CacheError> {
        self.evaluated.mark_fresh(id)?;
        self.deps.record_positions(key, parents);
        self.activity.positions_computed += 1;
        #[cfg(test)]
        self.log.positions.push(key);
        Ok(())
    }

    /// The slot of `key`'s current positions and normals: cached, else the normals are computed
    /// (or, where they are stale, brought up to date) from the positions of its
    /// [`normal_tiles`](geom::normal_tiles), made current together as one cone.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    fn ensure_normals(&mut self, key: TileKey) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        self.normals_batch(&[key], None)?;
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
    /// With `bounds`, each tile whose normals are computed has its bounds put there too (see
    /// [`TileStore::evaluate_many_with_bounds`]).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; nothing stays pinned then.
    fn normals_batch(
        &mut self,
        batch: &[TileKey],
        bounds: Option<&mut TileMap<TileBounds>>,
    ) -> Result<(), CacheError> {
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
        let computed = self.normals_rounds(&wanted, &inputs, &stale, &held, bounds);
        let released = self.unpin_held(&held);
        computed?;
        released
    }

    /// Computes the normals of `wanted` (each with its normal `inputs` and, where its normals
    /// are stale, the inputs they are stale since, in the same order) from `held`, every tile
    /// they read, pinned: on the workers, straight into each tile's slot
    /// ([`TileStore::normals_round`]), then recorded in key order (see
    /// [`TileStore::normals_installed`]). With `bounds`, each tile's bounds are computed beside
    /// its normals and put there once they are recorded.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]: the first error in key order. Nothing is recorded then: a
    /// tile whose normals were absent keeps none, and one whose normals were stale stays stale
    /// since the same inputs (any vertex rewritten was one its refresh recomputes).
    fn normals_rounds(
        &mut self,
        wanted: &[TileKey],
        inputs: &[BTreeSet<TileKey>],
        stale: &[Option<BTreeSet<TileKey>>],
        held: &Held,
        bounds: Option<&mut TileMap<TileBounds>>,
    ) -> Result<(), CacheError> {
        let tasks: Vec<(TileKey, Option<&BTreeSet<TileKey>>)> = wanted
            .iter()
            .copied()
            .zip(stale.iter().map(Option::as_ref))
            .collect();
        let computed = self.normals_round(&tasks, held, bounds.is_some())?;
        #[cfg(test)]
        self.log.refreshed.extend(
            tasks
                .iter()
                .filter(|(_, since)| since.is_some())
                .map(|&(key, _)| key),
        );
        for (&key, read) in wanted.iter().zip(inputs) {
            self.normals_installed(key, read);
        }
        if let Some(bounds) = bounds {
            bounds.extend(
                wanted
                    .iter()
                    .zip(computed)
                    .filter_map(|(&key, tile)| Some((key, tile?))),
            );
        }
        Ok(())
    }

    /// Computes the normals of each tile of `tasks` on the workers, straight into its slot
    /// among `held` (every tile the normals read, pinned, in key order; the tasks' tiles are
    /// among them, each once): brought up to date by [`geom::refresh_normals_into`] where they
    /// are stale since a set of inputs, computed in full by [`geom::normals_into`] otherwise.
    ///
    /// Every slot of `held` is borrowed at once ([`Pool::get_many_mut`]) and split by field:
    /// all the positions shared, for every task to read, and each task's tile's normals
    /// mutable, for that task alone. A task writes only its own tile's normals and reads only
    /// positions, so what lands in each slot is the same whichever worker computes it, and
    /// nothing of the pools' bookkeeping changes.
    ///
    /// With `with_bounds`, each task then computes its tile's bounds from its positions and the
    /// normals it just wrote ([`TileBounds::of_tile`], as [`crate::SpatialIndex::bounds_of`]
    /// does from the same slot), returned in task order; `None` for each tile without.
    ///
    /// # Errors
    ///
    /// The first task's error in order: [`CacheError::Eval`] (not for a valid mesh), or
    /// [`CacheError::Internal`] for slots of `held` that can't be borrowed together
    /// ([`Invariant::SlotsNotDisjoint`]), or a task whose tile isn't in `held`
    /// ([`Invariant::NotHeld`]) or is in `tasks` twice ([`Invariant::NamedTwice`]).
    fn normals_round(
        &mut self,
        tasks: &[(TileKey, Option<&BTreeSet<TileKey>>)],
        held: &Held,
        with_bounds: bool,
    ) -> Result<Vec<Option<TileBounds>>, CacheError> {
        let ids: Vec<SlotId<EvaluatedSlot>> = held.iter().map(|&(_, id)| id).collect();
        let slots = self
            .evaluated
            .get_many_mut(&ids)
            .ok_or(Invariant::SlotsNotDisjoint)?;
        let mut positions: Positions<'_> = Vec::with_capacity(held.len());
        let mut normals = Vec::with_capacity(held.len());
        for (&(key, _), slot) in held.iter().zip(slots) {
            let EvaluatedSlot {
                positions: own,
                normals: out,
            } = slot;
            let own: &[Vec3] = own;
            positions.push((key, own));
            normals.push(Some(out));
        }
        // Each task's buffer: its tile's normals to write, its tile's own positions, and the
        // bounds it computes.
        let mut outs = tasks
            .iter()
            .map(|(key, _)| {
                let at = held_index(held, key)?;
                let own = positions.get(at).map(|&(_, own)| own);
                let out = normals
                    .get_mut(at)
                    .and_then(Option::take)
                    .ok_or(Invariant::NamedTwice(*key))?;
                Ok((out, own.ok_or(Invariant::NotHeld(*key))?, None))
            })
            .collect::<Result<Vec<_>, Invariant>>()?;
        let mesh = &*self.mesh;
        #[cfg(test)]
        let failing = self.failing_normals;
        each_parallel(
            &self.workers,
            tasks,
            &mut outs,
            |&(key, since), (out, own, bounds)| {
                #[cfg(test)]
                if failing == Some(key) {
                    return Err(Invariant::StaleHandle.into());
                }
                let source = Resident::new(mesh, Tiles::Split(&positions));
                let n = vertex_count(&key);
                let out = prefix_mut(out.as_mut_slice(), n)?;
                let region = Region::of_tile(&key);
                match since {
                    None => normals_into(mesh, &source, region, out)?,
                    Some(since) => {
                        refresh_normals_into(mesh, &source, region, |k| since.contains(k), out)?;
                    }
                }
                if with_bounds {
                    *bounds = TileBounds::of_tile(prefix(own, n)?, out);
                }
                Ok(())
            },
        )?;
        Ok(outs.into_iter().map(|(_, _, bounds)| bounds).collect())
    }

    /// Records `key`'s normals, just written into its slot, as read from `inputs`, which the
    /// caller holds pinned.
    fn normals_installed(&mut self, key: TileKey, inputs: &BTreeSet<TileKey>) {
        self.deps.record_normals(key, inputs);
        self.activity.normals_computed += 1;
        #[cfg(test)]
        self.log.normals.push(key);
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

#[cfg(test)]
pub(crate) mod tests;
