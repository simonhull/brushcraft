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
//! on the store's worker threads ([`StoreConfig::workers`]), in rounds of at most
//! [`ROUND`](config::ROUND) tiles: the owner thread loads and pins the round's correction tiles,
//! the workers predict every tile of the round from the pinned level below, and the owner unpins
//! the correction tiles and installs the results one by one, in key order. A round of tiles that
//! already have slots (stale tiles recomputed in place, as after a brush step) is predicted
//! straight into those slots, each task writing its own; any other round into scratch buffers,
//! which the owner copies into the slots it takes as it installs them. Normals are computed
//! straight into their tiles' slots the same way. The workers write only the slots (or buffers)
//! handed to them, one per task, borrowed disjointly from the pool, and read the pinned tiles
//! through shared borrows; nothing they do touches the pools' bookkeeping, the budget or the
//! dependency graph, and the rounds are cut the same way whatever the thread count, so the store
//! makes the same decisions (which slot, which victim) and computes the same bits with one thread
//! or thirty-two.
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

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use geom::{BaseMesh, Prediction, TileKey, Vec3};

use self::config::Headroom;
#[cfg(test)]
use self::hooks::TestHooks;
use self::layers::Dirty;
use self::pins::{PinTable, Pinned};
use crate::activity::Activity;
use crate::bounds::Book;
use crate::deps::DepGraph;
use crate::display::Display;
use crate::feed::{ChangeFeed, ConsumerId};
use crate::file::{BaseFile, ProjectFile, ReadScratch};
use crate::hash::TileMap;
use crate::kept::KeptLru;
use crate::op::OpId;
use crate::pool::{Budget, Pool};
use crate::prefetch::Prefetcher;
use crate::slot::{CorrectionSlot, EvaluatedSlot};
use crate::stamp::Stamp;

mod bounds;
mod config;
mod display;
mod edit;
mod epochs;
mod error;
mod evaluate;
mod frames;
#[cfg(test)]
mod hooks;
mod layers;
mod page;
mod pen_down;
mod persist;
mod pins;
mod preview;
mod queued;
mod read;
mod read_ahead;
mod replay;
mod ring;
mod round;
mod stream;
mod strokes;
mod upkeep;

pub use config::{Evaluation, StoreConfig};
pub use edit::TileEdit;
pub use error::{CacheError, Invariant};
pub use frames::{ParentSlots, Predictor};
pub(crate) use layers::{Saved, Snapshot};
pub use pen_down::{MadePenDown, PenDownMaker, PenDownShape, PenDownTaker, TakenShape};
pub use pins::PinGuard;
pub use preview::DisplayPreview;
pub use read::WaveBounds;

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

/// A display tile a brush stroke is changing, as [`TileStore::bump_display_error`] keeps it.
#[derive(Debug, Clone)]
struct Bumped {
    /// Its record as it was before the stroke's first dab reached it.
    record: crate::display::DisplayRecord,
    /// What the dabs since have added to its bound: twice their displacements, summed.
    extra: f32,
    /// The finest tiles under it the dabs moved.
    finest: BTreeSet<TileKey>,
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
/// state, which hold nearly all of a store's memory, and, with a display layer, its floor and
/// its caches (the decoded display tiles, a job's finest copies, the records read ahead and the
/// readers' memo: [`StoreConfig::display_cache_bytes`], a sixteenth of the budget at most, each
/// cache holding a few dozen entries however small its share), and the display records
/// waiting for the working file ([`StoreConfig::display_waiting_bytes`], a thirty-second of the
/// budget, at most 64 MiB, written through down to half of it once past it; passed only by what
/// the store records while it can't write them through: a recovery flush is between its cut and
/// its save point, the working file is read-only or a failed recovery file, or a write through
/// failed since the last flush), and the queued evaluation's finished tiles and keys
/// ([`StoreConfig::evaluation_bytes`], a sixty-fourth of the budget, at most 64 MiB, holding a
/// round of tiles however small). Besides them, and not charged to the budget:
///
/// - the display tiles the workers are building or decoding, and the inputs handed to them: a slice
///   hands out a few at a time, at most one a worker in flight;
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
///   the tiles resident and edited, not with their size;
/// - a brush's tiles replayed away from the store (`sculpt`'s `apply_dabs`, through
///   [`TileStore::kept_prediction`], [`TileStore::cached_correction`] and
///   [`TileStore::replace_corrections`]): for each tile of a run, copies of its positions,
///   corrections and stroke state and its deltas, about 130 KB, for at most
///   [`StoreConfig::kept_predictions`] tiles at once (about 8.3 MB with the default 64), freed when
///   the run is written.
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
    /// The display tiles the store keeps in its files, under the bounds' rule (see
    /// [`crate::display`]).
    display: Book<Display>,
    /// The finest level the display layer is kept under ([`StoreConfig::display`]); `None` for
    /// a store that keeps none.
    display_finest: Option<geom::Level>,
    /// The bytes the display records waiting for the working file may hold before they are
    /// written through: the budget's share for them ([`StoreConfig::display_waiting_bytes`]).
    display_waiting_share: usize,
    /// Whether writing display records through to the working file failed since its last
    /// flush or save: no more are written through until one (see
    /// [`TileStore::write_through_display`]).
    display_through_failed: bool,
    /// The display tiles a brush stroke's dabs changed, as they were before the first of them,
    /// with what the dabs since have added to their bound (see
    /// [`TileStore::bump_display_error`]): drawn in their place until a refresh records them.
    /// Memory only: never written.
    bumped: std::collections::BTreeMap<TileKey, Bumped>,
    /// The display tiles that wouldn't build, each with why and the count of correction changes
    /// ([`TileStore::edits`]) it failed at: unknown, and left out of every job until a change
    /// lands (see [`TileStore::failed_display`]).
    display_failed: std::collections::BTreeMap<TileKey, (u64, geom::DisplayError)>,
    /// The display book's count of changes when the refresh last looked for unknown tiles
    /// ([`TileStore::run_jobs`]): it looks again only once that moves.
    refresh_seen: u64,
    /// How many times the store recorded built display tiles ([`TileStore::record_built`]).
    display_builds: u64,
    /// The generation of each display record built since the store opened: the value of
    /// `display_builds` that recorded it (see [`TileStore::display_generation`]).
    display_generations: std::collections::BTreeMap<TileKey, u64>,
    /// Display tiles the jobs built or decoded, for the coarser tiles that read them.
    display_decoded: crate::jobs::DecodedTiles,
    /// How many entries the display layer's caches hold, from the budget's share for them.
    cache_caps: crate::jobs::CacheCaps,
    /// Display records read ahead on the workers, for the next dabs' bumps.
    read_ahead: read_ahead::ReadAhead,
    /// The writes the last bump covered and the display tiles they can change, as it listed
    /// them: the next settle of those writes takes them (see [`TileStore::settle_display`]).
    bump_readers: Option<(BTreeSet<TileKey>, BTreeSet<TileKey>)>,
    /// The tiles whose corrections changed (a write, or a lost tile going back to its record)
    /// since a bump last covered them, while any tile is bumped: a bump they reach no longer
    /// bounds its tile, and is dropped before anything reads one
    /// ([`TileStore::drop_uncovered_bumps`]).
    unbumped: BTreeSet<TileKey>,
    /// The display layer's floor (see [`StoreConfig::floor_top`]): every face's display tiles up
    /// to [`TileStore::floor_top`], as last recorded, kept for the store's life.
    floor: std::collections::BTreeMap<TileKey, crate::display::DisplayRecord>,
    /// Where in the files each floor record loaded on open was read from, until the store
    /// records the tile again: while that place is still the files' latest record of the tile,
    /// the floor's copy is that record, and a draw takes it rather than reading the file again
    /// (see [`TileStore::floor_record`]).
    floor_read: std::collections::BTreeMap<TileKey, (crate::file::FileReader, crate::file::Loc)>,
    /// The top level of the floor.
    floor_top: geom::Level,
    /// The stale-tile feed (see [`crate::stale`]).
    stale: crate::stale::StaleFeed,
    /// The correction writes since the open epoch began, noted while a consumer listens.
    epoch_writes: BTreeSet<TileKey>,
    /// The other tiles the open epoch changed (display tiles recorded), noted while a consumer
    /// listens.
    epoch_keys: BTreeSet<TileKey>,
    /// The open epoch's brush steps, if it has any: the displacements their bumps raised
    /// display bounds by, summed (rounded up), and the correction tiles those bumps covered. The
    /// epoch is a step's if they cover every write in it ([`crate::EpochKind`]).
    /// The third: whether a write came after the last bump, which no bump bounds (a dab that
    /// failed after writing) and makes the epoch a revert's.
    epoch_step: Option<(f32, BTreeSet<TileKey>, bool)>,
    /// The previews: finest tiles a renderer shows ahead of their evaluation, never saved (see
    /// the `preview` module).
    previews: preview::Previews,
    /// The pen-down shapes of the tiles the current stroke changed (see the `pen_down` module).
    pen_down: pen_down::PenDown,
    /// The jobs queued (see [`crate::jobs`]), run in slices by [`TileStore::run_jobs`].
    jobs: std::collections::VecDeque<crate::jobs::Job>,
    /// The queued evaluation (see [`crate::queued`]): finest tiles wanted by priority, run in
    /// slices by [`TileStore::run_jobs`] beside the display jobs.
    evaluation: crate::queued::EvaluationQueue,
    /// Whether the next call to [`TileStore::run_jobs`] gives its first slice to the queued
    /// evaluation (else to the display jobs): the turn carries across calls, so neither side
    /// starves however short the calls.
    evaluation_next: bool,
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
    /// One tile of positions per tile of a round that isn't predicted in place (see
    /// [`ROUND`](config::ROUND) and [`TileStore::build_round`]): the workers write here, then
    /// the owner thread copies each into the slot it takes for it. Grown on demand to the most
    /// such tiles a round has had, never past [`ROUND`](config::ROUND) buffers; a round
    /// predicted wholly in place (a brush step's) takes none.
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
    kept: KeptLru<Prediction>,
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
    /// Stroke records framed at their strokes' ends, waiting for the next flush (or save) to
    /// append them (see [`TileStore::record_stroke`]).
    strokes_waiting: Vec<Vec<u8>>,
    /// What the tests read and set: the log of what was computed, and the faults they inject.
    #[cfg(test)]
    hooks: TestHooks,
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

#[cfg(test)]
pub(crate) mod tests;
