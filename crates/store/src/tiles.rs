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
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::convert::Infallible;
use std::rc::Rc;
use std::sync::Arc;

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionView, CorrectionViewMut, EvalError, FaceId,
    LayerId, Level, MAX_COPIES, PositionSource, Prediction, Region, RowUpdate, TileBorder, TileKey,
    Vec3, VertexCopy, normals_into, parent_tiles, predict_region, refresh_normals_into,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use thiserror::Error;

use crate::activity::Activity;
use crate::bounds::Book;
use crate::deps::{DepGraph, list_normal_inputs};
use crate::feed::{ChangeFeed, ConsumerId};
use crate::file::{BaseFile, FileSyncer, FramedTile, Loc, ProjectFile, ReadScratch, StoreError};
use crate::hash::{TileHash, TileMap, TileSet};
use crate::index::TileBounds;
use crate::kept::Kept;
use crate::meta::{Meta, Role, SavePoint};
use crate::op::{OpId, RESERVED_SITE};
use crate::pool::{Budget, Freshness, Invalidated, Pool, PoolError, SlotData, SlotId};
use crate::prefetch::{Fetched, Job, Prefetcher, Source};
use crate::slot::{CorrectionSlot, EvaluatedSlot, SIDE};
use crate::stamp::Stamp;
use crate::workers::{StartWorkersError, Workers};

/// How a [`TileStore`] is set up.
///
/// Made with [`StoreConfig::new`] and adjusted with its `with_` setters, so that a field added
/// later doesn't break a caller:
///
/// ```
/// # use std::num::NonZeroUsize;
/// # use store::{StoreConfig, Workers};
/// let four = Workers::Fixed(NonZeroUsize::new(4).unwrap());
/// let config = StoreConfig::new(2 << 30, 1).with_workers(four);
/// assert_eq!(config.chunk_bytes, StoreConfig::DEFAULT_CHUNK_BYTES);
/// assert_eq!(config.workers, four);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoreConfig {
    /// Bytes the two pools, the kept predictions and a brush stroke's state may hold together,
    /// never exceeded (the kept predictions are charged their positions' and frames'
    /// allocations, the stroke state its per-vertex buffers; their map entries and the
    /// allocator's rounding, a few KB at the default counts, sit outside it). At least
    /// [`StoreConfig::minimum_budget`] for the store's mesh.
    ///
    /// The kept predictions' share is set aside first, at its worst case
    /// ([`StoreConfig::kept_prediction_bytes`]), then the stroke state's
    /// ([`StoreConfig::stroke_state_bytes`]), and the pools share the rest, the evaluated pool
    /// filling the room below a whole chunk with a short one, so the shares cost only their own
    /// bytes. How the pools share it is the store's memory policy (see [`TileStore`]):
    /// corrections take evaluated chunks only up to a quarter of the pools' share (the budget less
    /// the reserved shares), and never below the evaluated pool's floor.
    ///
    /// The evaluation scratch (about 3 MB), the read buffers, the workers' own allocations (a
    /// few hundred KB each at most) and the thread stacks (2 MiB reserved per worker) sit
    /// outside it: a few MB besides the budget, plus the stacks (see "Memory outside the
    /// budget" on [`TileStore`]).
    pub budget_bytes: usize,
    /// Bytes per pool chunk, the unit both pools grow and shrink by and are charged in (whole
    /// slots fill it, the rest of a chunk is charged unused):
    /// [`StoreConfig::DEFAULT_CHUNK_BYTES`] in the app, smaller in tests.
    pub chunk_bytes: usize,
    /// The site (machine or session) the store's op ids carry; any but [`RESERVED_SITE`].
    pub site: u16,
    /// How many worker threads evaluate tiles in parallel (see [`TileStore::evaluate_many`]):
    /// by default one per physical core ([`Workers::PhysicalCores`], see [`crate::Workers`] for
    /// why). Results are the same bits whatever the count; tests pin it to compare.
    pub workers: Workers,
    /// How many tiles' predictions the store keeps for predicting them again in place (see
    /// "Kept predictions" on [`TileStore`]): [`StoreConfig::DEFAULT_KEPT_PREDICTIONS`] unless
    /// set. Each is charged to the budget at the largest tile's size,
    /// [`geom::Prediction::TILE_BYTES`] (about 206 KB); zero keeps none, and every tile is
    /// predicted in full. Results are the same bits either way. [`TileStore::open`] refuses a
    /// count whose share leaves the pools less than their minimum.
    pub kept_predictions: usize,
    /// How many tiles' stroke state the budget sets aside: the `f32` start and summed deltas a
    /// brush stroke keeps per vertex of the tiles it is editing, so that deltas smaller than a
    /// quantization step aren't rounded away dab by dab (the `sculpt` crate holds the state;
    /// the store only reserves for it and reports the count, [`TileStore::stroke_tiles`]).
    /// [`StoreConfig::DEFAULT_STROKE_TILES`] unless set. Each is charged to the budget at the
    /// largest tile's size, [`StoreConfig::STROKE_TILE_BYTES`]; zero keeps none, and a stroke
    /// writes every tile dab by dab. [`TileStore::open`] refuses a count whose share leaves the
    /// pools less than their minimum beside the kept predictions.
    pub stroke_tiles: usize,
}

impl StoreConfig {
    /// The default chunk: 64 MiB, large enough that the allocator maps each chunk on its own
    /// and gives it back to the OS when it is dropped (see the pool's notes).
    pub const DEFAULT_CHUNK_BYTES: usize = 64 << 20;
    /// The default [`StoreConfig::kept_predictions`]: 64, about 13 MB of the budget. A brush step
    /// re-predicts the 20 to 30 tiles its sphere reaches, and a stroke moves on by a tenth of a
    /// radius a step, so a stroke's tiles are reused for several steps before it leaves them.
    pub const DEFAULT_KEPT_PREDICTIONS: usize = 64;
    /// The default [`StoreConfig::stroke_tiles`]: 512, about 17.3 MB of the budget. A dab edits
    /// the 20 to 30 tiles its sphere reaches, so a stroke keeps the state of every tile its last
    /// several dabs edited with room to spare.
    pub const DEFAULT_STROKE_TILES: usize = 512;
    /// The bytes of the budget one tile's stroke state is charged:
    /// two `f32` (the start and the summed deltas) per vertex of the largest tile,
    /// `(TILE_QUADS + 1)²` of them, so 33,800 bytes.
    pub const STROKE_TILE_BYTES: usize = SIDE * SIDE * 2 * size_of::<f32>();

    /// A store of `budget_bytes` whose op ids carry `site`, with
    /// [`StoreConfig::DEFAULT_CHUNK_BYTES`] chunks and one evaluation worker per physical core
    /// ([`Workers::PhysicalCores`]).
    #[must_use]
    pub const fn new(budget_bytes: usize, site: u16) -> Self {
        Self {
            budget_bytes,
            chunk_bytes: Self::DEFAULT_CHUNK_BYTES,
            site,
            workers: Workers::PhysicalCores,
            kept_predictions: Self::DEFAULT_KEPT_PREDICTIONS,
            stroke_tiles: Self::DEFAULT_STROKE_TILES,
        }
    }

    /// This configuration with a budget of `budget_bytes` (see [`StoreConfig::budget_bytes`]).
    #[must_use]
    pub const fn with_budget_bytes(self, budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            ..self
        }
    }

    /// This configuration with chunks of `chunk_bytes` (see [`StoreConfig::chunk_bytes`]).
    #[must_use]
    pub const fn with_chunk_bytes(self, chunk_bytes: usize) -> Self {
        Self {
            chunk_bytes,
            ..self
        }
    }

    /// This configuration with op ids of `site` (see [`StoreConfig::site`]).
    #[must_use]
    pub const fn with_site(self, site: u16) -> Self {
        Self { site, ..self }
    }

    /// This configuration with its evaluation workers counted as `workers` asks (see
    /// [`StoreConfig::workers`]).
    #[must_use]
    pub const fn with_workers(self, workers: Workers) -> Self {
        Self { workers, ..self }
    }

    /// This configuration keeping up to `kept_predictions` tiles' predictions (see
    /// [`StoreConfig::kept_predictions`]).
    #[must_use]
    pub const fn with_kept_predictions(self, kept_predictions: usize) -> Self {
        Self {
            kept_predictions,
            ..self
        }
    }

    /// The bytes of the budget set aside for the kept predictions:
    /// [`StoreConfig::kept_predictions`] at [`geom::Prediction::TILE_BYTES`] each, the most
    /// they can hold. `None` if that is more than a `usize` counts (no budget holds it).
    #[must_use]
    pub const fn kept_prediction_bytes(&self) -> Option<usize> {
        self.kept_predictions.checked_mul(Prediction::TILE_BYTES)
    }

    /// This configuration setting aside stroke state for up to `stroke_tiles` tiles (see
    /// [`StoreConfig::stroke_tiles`]).
    #[must_use]
    pub const fn with_stroke_tiles(self, stroke_tiles: usize) -> Self {
        Self {
            stroke_tiles,
            ..self
        }
    }

    /// The bytes of the budget set aside for stroke state: [`StoreConfig::stroke_tiles`] at
    /// [`StoreConfig::STROKE_TILE_BYTES`] each. `None` if that is more than a `usize` counts
    /// (no budget holds it).
    #[must_use]
    pub const fn stroke_state_bytes(&self) -> Option<usize> {
        self.stroke_tiles.checked_mul(Self::STROKE_TILE_BYTES)
    }

    /// The least [`StoreConfig::budget_bytes`] a store over `mesh` with these chunks works
    /// with: enough that no evaluation, normals computation or edit runs out of room while
    /// nothing else is pinned, besides the kept predictions' share
    /// ([`StoreConfig::kept_prediction_bytes`]) and the stroke state's
    /// ([`StoreConfig::stroke_state_bytes`]; `usize::MAX` if either overflows).
    ///
    /// The pools' part is the larger of
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
        Headroom::of(mesh)
            .minimum_budget(self.chunk_bytes)
            .saturating_add(self.kept_prediction_bytes().unwrap_or(usize::MAX))
            .saturating_add(self.stroke_state_bytes().unwrap_or(usize::MAX))
    }

    /// The worker threads this configuration asks for on this machine (see
    /// [`StoreConfig::workers`]).
    fn worker_threads(&self) -> usize {
        self.workers.count().get()
    }
}

/// Tiles of one cone level built in one parallel round at most (see the module notes).
///
/// The rounds are cut by this count alone, never by the thread count, so the store's pool
/// operations come in the same order however many workers it has. Each round pins its tiles'
/// correction slots while the workers read them (fewer, cut at the first tile after the first
/// that would need a dirty tile written back or an evaluated chunk taken, see
/// [`TileStore::load_round`]), and holds a scratch buffer of one tile per tile of the round that
/// isn't predicted in place.
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

/// The bytes of the budget's reserved shares, once checked to fit beside the pools' minimum.
///
/// The shares are the kept predictions' ([`StoreConfig::kept_prediction_bytes`]) and the
/// stroke state's ([`StoreConfig::stroke_state_bytes`]); `config`'s budget is checked to hold
/// them beside the pools' minimum for `headroom`, the kept predictions first.
///
/// # Errors
///
/// [`CacheError::BudgetTooSmall`] if the budget is below the pools' minimum alone;
/// [`CacheError::TooManyKeptPredictions`] if it holds that but not the kept predictions besides;
/// [`CacheError::TooManyStrokeTiles`] if it holds both but not the stroke state besides.
fn reserved_shares(headroom: Headroom, config: &StoreConfig) -> Result<usize, CacheError> {
    let pools = headroom.minimum_budget(config.chunk_bytes);
    let Some(spare) = config.budget_bytes.checked_sub(pools) else {
        let kept = config.kept_prediction_bytes().unwrap_or(usize::MAX);
        let stroke = config.stroke_state_bytes().unwrap_or(usize::MAX);
        return Err(CacheError::BudgetTooSmall {
            minimum: pools.saturating_add(kept).saturating_add(stroke),
        });
    };
    let kept = config
        .kept_prediction_bytes()
        .filter(|&kept| kept <= spare)
        .ok_or(CacheError::TooManyKeptPredictions {
            asked: config.kept_predictions,
            most: spare / Prediction::TILE_BYTES,
        })?;
    let spare = spare - kept;
    let stroke = config
        .stroke_state_bytes()
        .filter(|&stroke| stroke <= spare)
        .ok_or(CacheError::TooManyStrokeTiles {
            asked: config.stroke_tiles,
            most: spare / StoreConfig::STROKE_TILE_BYTES,
        })?;
    Ok(kept + stroke)
}

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

/// Why a [`TileStore`] operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
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
    /// [`TileStore::save`] was called while the working file is a recovery file whose writer
    /// failed: a sync of it failed (or the writer's thread stopped), so no save point may be
    /// written there again. The project moves the store to a new file; save the project
    /// ([`crate::Project::save`]) instead.
    #[error("the working file is a recovery file whose sync failed: no save point goes there")]
    FailedWorkingFile,
    /// A rebase was given a save that is out of date: the store was edited after the save was
    /// made, so the save doesn't hold every edit. Save again. Or a roll-over was given a new
    /// working file filled before the latest edit.
    #[error("the store was edited after the save it was to be rebased on")]
    StaleSave,
    /// The configured budget is below [`StoreConfig::minimum_budget`] for the mesh, without
    /// counting the reserved shares (see [`CacheError::TooManyKeptPredictions`] and
    /// [`CacheError::TooManyStrokeTiles`]).
    #[error("the budget is below the {minimum} bytes this mesh needs")]
    BudgetTooSmall {
        /// The least budget the mesh works with, at the configured
        /// [`StoreConfig::kept_predictions`] and [`StoreConfig::stroke_tiles`]
        /// ([`StoreConfig::minimum_budget`]).
        minimum: usize,
    },
    /// The budget holds the pools' minimum, but not beside the share the configured
    /// [`StoreConfig::kept_predictions`] need ([`StoreConfig::kept_prediction_bytes`]).
    #[error("keeping {asked} predictions takes more of the budget than the {most} it can spare")]
    TooManyKeptPredictions {
        /// The kept predictions asked for.
        asked: usize,
        /// The most this budget can keep beside the pools' minimum.
        most: usize,
    },
    /// The budget holds the pools' minimum and the kept predictions' share, but not beside the
    /// share the configured [`StoreConfig::stroke_tiles`] need
    /// ([`StoreConfig::stroke_state_bytes`]).
    #[error("stroke state for {asked} tiles takes more of the budget than the {most} it can spare")]
    TooManyStrokeTiles {
        /// The tiles' stroke state asked for.
        asked: usize,
        /// The most tiles' stroke state this budget can set aside beside the pools' minimum
        /// and the kept predictions.
        most: usize,
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
    /// The evaluation worker threads couldn't be started (see [`StoreConfig::workers`]).
    #[error("starting the evaluation worker threads failed")]
    Workers(#[source] StartWorkersError),
}

/// An invariant of [`TileStore`] found broken (see [`CacheError::Internal`]).
#[derive(Debug, Error)]
#[non_exhaustive]
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
    /// Slots borrowed together weren't all live and distinct: a handle outlived its slot, or
    /// two named the same slot.
    #[error("slots borrowed together were stale or named one slot twice")]
    SlotsNotDisjoint,
    /// A tile an evaluation didn't pin was looked for among those it did.
    #[error("{0:?} is not among the tiles the evaluation pinned")]
    NotHeld(TileKey),
    /// A tile was named twice among the tasks of one round, which write one tile each.
    #[error("{0:?} is named twice in one round")]
    NamedTwice(TileKey),
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
    /// A full save point would have named a different number of records than the maps derived
    /// for it hold (see [`StoreError::SavePointMismatch`]). Nothing was written.
    #[error("a full save point names {named} records, but its derived maps hold {derived}")]
    SavePointMismatch {
        /// The records the full Index would name.
        named: usize,
        /// The records the derived maps hold.
        derived: usize,
    },
}

impl CacheError {
    /// `error`, from writing a save point, as the store reports it.
    ///
    /// [`StoreError::SavePointMismatch`] is a bug in deriving the save point, which trying again
    /// can't fix, so it is [`Invariant::SavePointMismatch`] whatever the operation was. Any other
    /// error is `wrap`ped as the operation reports file errors.
    fn from_save_point(error: StoreError, wrap: impl FnOnce(StoreError) -> Self) -> Self {
        if let StoreError::SavePointMismatch { named, derived } = error {
            return Invariant::SavePointMismatch { named, derived }.into();
        }
        wrap(error)
    }
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

/// How [`TileStore::load_correction_as`] may get a slot for a correction tile that isn't
/// resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Acquire {
    /// As the memory policy orders it (see [`TileStore::acquire_correction`]): it may take an
    /// evaluated chunk back or write a dirty tile back.
    Any,
    /// A free slot, a new chunk or a clean tile's slot only, as prefetch takes them: nothing is
    /// written back and the evaluated pool keeps its chunks. A round's tiles after its first
    /// load this way, so a round writes back no more than one tile at a time would.
    Clean,
}

/// A correction tile being loaded that wasn't resident: its slot, taken (see
/// [`TileStore::take_correction_slot`]) and not yet filled, and where its record is.
#[derive(Debug, Clone, Copy)]
struct Fill {
    /// The tile.
    key: TileKey,
    /// Its slot, which [`fill_slot`] fills.
    id: SlotId<CorrectionSlot>,
    /// Where its latest record is: the working file, else the base; `None` for a tile never
    /// written, which loads as zero.
    found: Option<(Source, Loc)>,
}

/// What filling each slot of a round gave, in the round's order: the op of the record read, or
/// `None` for a tile loaded as zero (see [`fill_slot`]).
type FillOutcomes = Vec<Result<Option<OpId>, CacheError>>;

/// Fills `slot` with `tile`'s tile: read from its record in `working` or `base`, with `scratch`'s
/// buffers, or reset to zero where it has none. Returns the op of the record read.
///
/// Touches nothing but `slot` and `scratch`, so a round's tiles are filled on several threads
/// at once (see [`TileStore::fill_round`]).
///
/// # Errors
///
/// [`CacheError::Store`] if the record can't be read or doesn't decode;
/// [`CacheError::Internal`] for a record in a base the store doesn't have, or a slot that can't
/// hold the tile's side. On error the slot holds no tile.
fn fill_slot(
    working: &ProjectFile,
    base: Option<&BaseFile>,
    tile: &Fill,
    slot: &mut CorrectionSlot,
    scratch: &mut ReadScratch,
) -> Result<Option<OpId>, CacheError> {
    let key = &tile.key;
    match (tile.found, base) {
        (None, _) => slot
            .reset(key.side())
            .map(|()| None)
            .map_err(|e| Invariant::CorrectionSlot(e).into()),
        (Some((Source::Working, loc)), _) => Ok(Some(working.read_at(key, loc, slot, scratch)?)),
        (Some((Source::Base, loc)), Some(base)) => Ok(Some(base.read_at(key, loc, slot, scratch)?)),
        (Some((Source::Base, _)), None) => Err(Invariant::MissingBase.into()),
    }
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

/// The positions of tiles an evaluation pinned, each tile with all `SIDE²` of its slot's
/// positions, in key order: what a [`Resident`] source reads.
type Positions<'a> = Vec<(TileKey, &'a [Vec3])>;

/// Where a [`Resident`] source finds the pinned tiles' positions.
#[derive(Clone, Copy)]
enum Tiles<'a> {
    /// Borrowed slices ([`Positions`]), found once per round by the owner thread, split off
    /// the same borrow that hands the workers the slots they write
    /// ([`TileStore::predict_round`], [`TileStore::normals_round`]).
    Split(&'a [(TileKey, &'a [Vec3])]),
    /// Looked up in the evaluated pool, shared, when a read first needs a tile: for a round
    /// whose workers write no slot (a round of new tiles, as paging in builds them), which so
    /// costs the owner nothing per tile of the level below, however wide that level is.
    Shared {
        /// The evaluated pool.
        pool: &'a Pool<EvaluatedSlot>,
        /// The pinned tiles, in key order.
        held: &'a [(TileKey, SlotId<EvaluatedSlot>)],
    },
}

/// Positions of the tiles an evaluation pinned, and the base mesh at level 0.
///
/// Each vertex is read from [`TileKey::containing`]'s tile, as [`PositionSource`] requires, so
/// differing copies on tile borders can't open a seam.
///
/// A gather reads a block row by row, so nearly every read falls in the tile the read before
/// it did: the source remembers that tile's positions, and finds a tile (a binary search of the
/// pinned tiles, see [`Tiles`]) only when a read leaves it. The memory is a [`Cell`], so a
/// source is `!Sync`: each task on the workers makes its own ([`Resident::new`]), which costs
/// nothing.
struct Resident<'a> {
    /// Level 0.
    mesh: &'a BaseMesh,
    /// The pinned tiles.
    tiles: Tiles<'a>,
    /// The tile the last read above level 0 resolved to, and its positions.
    last: Cell<Option<(TileKey, &'a [Vec3])>>,
}

impl<'a> Resident<'a> {
    /// A source over the pinned `tiles` and `mesh` at level 0, remembering no tile yet.
    const fn new(mesh: &'a BaseMesh, tiles: Tiles<'a>) -> Self {
        Self {
            mesh,
            tiles,
            last: Cell::new(None),
        }
    }

    /// The positions of pinned tile `key`, all `SIDE²` of its slot, or `None` if it isn't
    /// pinned here.
    fn positions(&self, key: TileKey) -> Option<&'a [Vec3]> {
        if let Some((last, positions)) = self.last.get()
            && last == key
        {
            return Some(positions);
        }
        let positions = match self.tiles {
            Tiles::Split(tiles) => {
                let at = tiles.binary_search_by_key(&key, |&(k, _)| k).ok()?;
                tiles.get(at)?.1
            }
            Tiles::Shared { pool, held } => {
                let at = held.binary_search_by_key(&key, |&(k, _)| k).ok()?;
                pool.get(held.get(at)?.1)?.positions.as_slice()
            }
        };
        self.last.set(Some((key, positions)));
        Some(positions)
    }
}

impl PositionSource for Resident<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        if level == Level::ZERO {
            return self.mesh.position(face, level, li, lj);
        }
        let index = usize::try_from(lj * key.side() + li).ok()?;
        self.positions(key)?.get(index).copied()
    }

    /// The run copied a tile at a time, answering each vertex from the tile `position` would.
    ///
    /// [`TileKey::containing`] picks a vertex's tile column from its column alone: with `tq`
    /// quads a tile, tile `tx` answers columns `tx·tq ..= tx·tq + tq − 1`, and its column `tq`
    /// belongs to the next tile, except in the face's last tile, which answers it too. So each
    /// tile's part of a run is one slice of its row.
    fn row(&self, face: FaceId, level: Level, i: u32, j: u32, out: &mut [Vec3]) -> Option<()> {
        // Level 0 is the base mesh, not tiles: read one by one, as the trait's default does.
        if level == Level::ZERO {
            for (slot, x) in out.iter_mut().zip(i..) {
                *slot = self.position(face, level, x, j)?;
            }
            return Some(());
        }
        let (mut x, mut rest) = (i, out);
        while !rest.is_empty() {
            let (key, li, lj) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, x, j)?;
            let side = key.side();
            // Up to column `tq - 1`, or just column `tq` in the face's last tile.
            let span = (side - 1).saturating_sub(li).max(1);
            let len = usize::try_from(span).ok()?.min(rest.len());
            let start = usize::try_from(lj * side + li).ok()?;
            let (part, tail) = rest.split_at_mut(len);
            part.copy_from_slice(self.positions(key)?.get(start..start + len)?);
            x += u32::try_from(len).ok()?;
            rest = tail;
        }
        Some(())
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
/// Item `k` gets `buffers[k]`; there must be as many buffers as items. A buffer is whatever
/// the item writes: a scratch vector, or a borrow of the slot the item fills, so the borrows
/// handed to the workers are disjoint by construction. The error returned doesn't depend on
/// which worker finished first.
///
/// # Errors
///
/// The first item's error, in order; [`CacheError::Internal`] if there are fewer buffers than
/// items.
fn each_parallel<T: Sync, B: Send>(
    workers: &rayon::ThreadPool,
    items: &[T],
    buffers: &mut [B],
    f: impl Fn(&T, &mut B) -> Result<(), CacheError> + Sync + Send,
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

/// One task of a round of predictions: a tile with its correction slot.
type RoundTask = (TileKey, SlotId<CorrectionSlot>);

/// The tasks of a round of predictions: each tile of `round` with its correction slot (from
/// `corrections`, in the same order).
fn round_tasks(
    round: &[(TileKey, &[TileKey])],
    corrections: &[SlotId<CorrectionSlot>],
) -> Vec<RoundTask> {
    round
        .iter()
        .map(|&(key, _)| key)
        .zip(corrections.iter().copied())
        .collect()
}

/// What [`predict_tile`] does about a tile's prediction (see "Kept predictions" on
/// [`TileStore`]).
#[derive(Debug, Clone, Copy)]
enum Predict<'a> {
    /// Applies the corrections to this prediction, the one the store kept for the tile, without
    /// predicting.
    Reuse(&'a Prediction),
    /// Predicts the tile and returns the prediction, for the store to keep.
    Keep,
    /// Predicts the tile and drops the prediction once it is applied, on the worker.
    Drop,
}

/// What [`TileStore::plan_predictions`] decides for one tile of a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// Reuse the tile's kept prediction ([`Predict::Reuse`]).
    Reuse,
    /// Keep the prediction made ([`Predict::Keep`]).
    Keep,
    /// Drop the prediction made ([`Predict::Drop`]).
    Drop,
}

/// Predicts tile `key` and applies its corrections, into the first `side²` values of `out`,
/// as `how` says: the prediction made, if it is to be kept ([`Predict::Keep`]), else `None`.
///
/// The parents are read from `tiles` (pinned, with the mesh at level 0), the corrections from
/// slot `id` of `pool`. Reads only, so it runs on any worker.
///
/// # Errors
///
/// [`CacheError::Eval`] (not for a valid mesh); [`CacheError::Internal`] for a stale handle, a
/// broken slot, or an `out` shorter than the tile.
fn predict_tile(
    mesh: &BaseMesh,
    pool: &Pool<CorrectionSlot>,
    tiles: Tiles<'_>,
    (key, id): RoundTask,
    how: Predict<'_>,
    out: &mut [Vec3],
) -> Result<Option<Prediction>, CacheError> {
    let view = pool
        .get(id)
        .ok_or(Invariant::StaleHandle)?
        .view()
        .map_err(Invariant::CorrectionSlot)?;
    let corrections = |i, j| view.get(i, j).unwrap_or(Vec3::ZERO);
    let out = prefix_mut(out, vertex_count(&key))?;
    if let Predict::Reuse(prediction) = how {
        prediction.apply_into(corrections, out)?;
        return Ok(None);
    }
    let source = Resident::new(mesh, tiles);
    let prediction = predict_region(mesh, &source, Region::of_tile(&key))?;
    prediction.apply_into(corrections, out)?;
    Ok(matches!(how, Predict::Keep).then_some(prediction))
}

/// A structural list for each of `keys`, in order: `recorded(key)` where the store recorded it,
/// else `list(key)`, computed on `workers`.
///
/// Reading a record back costs next to nothing, so those are read on the calling thread, and
/// only the keys without one go to the workers (none, in a brush step, whose tiles were all
/// evaluated before): a wave of trivial tasks costs more in waking the workers than it saves.
///
/// # Errors
///
/// The first error of `list`, in key order.
fn recorded_or_listed<T: Send>(
    workers: &rayon::ThreadPool,
    keys: &[TileKey],
    recorded: impl Fn(&TileKey) -> Option<T>,
    list: impl Fn(&TileKey) -> Result<T, CacheError> + Sync + Send,
) -> Result<Vec<T>, CacheError> {
    let mut found: Vec<Option<T>> = keys.iter().map(recorded).collect();
    let missing: Vec<TileKey> = keys
        .iter()
        .zip(&found)
        .filter(|(_, got)| got.is_none())
        .map(|(key, _)| *key)
        .collect();
    let mut listed = map_parallel(workers, &missing, list).into_iter();
    found
        .iter_mut()
        .map(|got| match got.take() {
            Some(value) => Ok(value),
            None => listed.next().unwrap_or_else(|| {
                Err(Invariant::ShortBuffer {
                    expected: missing.len() + 1,
                    got: missing.len(),
                }
                .into())
            }),
        })
        .collect()
}

/// Where `key` is among the evaluated tiles an evaluation pinned (in key order).
///
/// # Errors
///
/// [`Invariant::NotHeld`] if `key` isn't among them (the store asks only for tiles it pinned).
fn held_index(held: &Held, key: &TileKey) -> Result<usize, Invariant> {
    held.binary_search_by_key(key, |&(k, _)| k)
        .or(Err(Invariant::NotHeld(*key)))
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
    unsaved: &mut TileMap<Dirty>,
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
    /// The bounds changes the store had counted when the save was made: a rebase refuses it
    /// after a later one, which the save point doesn't hold.
    bounds: u64,
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
    /// The bounds changes the store had counted when it was filled: a roll-over refuses it
    /// after a later one, which the new file doesn't hold.
    bounds: u64,
}

/// The bounds book of a store over `working` and `base`.
///
/// It tracks every level either file holds bounds of and, for a recovery file, holds its bounds
/// as changed since the last save (no save has copied them into the project file yet). A bound of
/// the recovery file that doesn't read counts as unknown.
///
/// # Errors
///
/// [`CacheError::Store`] if reading the recovery file's bounds fails.
fn book_of(working: &ProjectFile, base: Option<&BaseFile>) -> Result<Book, CacheError> {
    let mut levels: BTreeSet<Level> = working
        .bounds_locations()
        .map(|(key, _)| key.level())
        .collect();
    if let Some(base) = base {
        levels.extend(base.bounds_locations().map(|(key, _)| key.level()));
    }
    let changed = match working.meta().role() {
        Role::Recovery { .. } => {
            let entries: Vec<(TileKey, Loc)> = working
                .bounds_locations()
                .map(|(key, loc)| (*key, loc))
                .collect();
            let read = working.read_bounds(&entries)?;
            entries
                .iter()
                .zip(read)
                .map(|((key, _), bounds)| (*key, bounds))
                .collect()
        }
        Role::Project => BTreeMap::new(),
    };
    Ok(Book::new(levels, changed))
}

/// `project` as of the save point a save has just written to it.
///
/// # Errors
///
/// [`CacheError::Internal`] with [`Invariant::NotSaved`] if `project` has none, which a save
/// that went through rules out.
fn saved_base(project: &ProjectFile) -> Result<BaseFile, CacheError> {
    project.base().ok_or_else(|| Invariant::NotSaved.into())
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
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`] without the reserved
    /// shares, [`CacheError::TooManyKeptPredictions`] below it with the kept predictions',
    /// [`CacheError::TooManyStrokeTiles`] below it with the stroke state's besides;
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
    /// [`CacheError::BudgetTooSmall`] below [`StoreConfig::minimum_budget`] without the reserved
    /// shares, [`CacheError::TooManyKeptPredictions`] below it with the kept predictions',
    /// [`CacheError::TooManyStrokeTiles`] below it with the stroke state's besides;
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
        let reserved_bytes = reserved_shares(headroom, &config)?;
        let next_op = first_op(
            config.site,
            [working.head(), base.as_ref().and_then(BaseFile::head)],
        )?;
        // A recovery file holds only edits no save has: a restored one's are still to save.
        let since_rebase = match working.meta().role() {
            Role::Recovery { .. } => working.keys().copied().collect(),
            Role::Project => BTreeSet::new(),
        };
        let bounds = book_of(&working, base.as_ref())?;
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(config.worker_threads())
            .thread_name(|n| format!("tile-eval-{n}"))
            .build()
            .map_err(|e| CacheError::Workers(StartWorkersError::new(e)))?;
        let mut feed = ChangeFeed::new();
        let save_consumer = feed.subscribe();
        Ok(Self {
            prefetch: Prefetcher::new(),
            mesh,
            mesh_hash,
            file: working,
            base,
            budget: Budget::new(config.budget_bytes - reserved_bytes),
            headroom,
            corrections: Pool::with_eviction_log(config.chunk_bytes),
            // Short chunks let the evaluated pool use the room the reserved shares
            // leave below a whole chunk (see "Kept predictions" on `TileStore`).
            evaluated: Pool::with_eviction_log(config.chunk_bytes).with_short_chunks(),
            deps: DepGraph::new(),
            unsaved: TileMap::default(),
            since_rebase,
            last_write: Stamp::ZERO,
            known_ops: TileMap::default(),
            generation: 0,
            working_failed: false,
            bounds,
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
            worker_read_scratch: Vec::new(),
            kept: Kept::new(config.kept_predictions),
            stroke_tiles: config.stroke_tiles,
            id: StoreId::next(),
            own_edits: BTreeSet::new(),
            activity: Activity::default(),
            #[cfg(test)]
            log: EvalLog::default(),
            #[cfg(test)]
            unreadable: None,
            #[cfg(test)]
            failing_normals: None,
            #[cfg(test)]
            without_tile_borders: false,
        })
    }

    /// Copies every tile edited since the last rebase (or since the store opened) into
    /// `project`, with every bound changed since, saves it, and returns the proof a rebase needs:
    /// the new save point as a base, and the store's latest write and bounds change.
    ///
    /// The bounds written describe the state the save point names: a tile whose latest edit is
    /// lost (below) has every bound its edit could change written as unknown.
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
    /// the save point is written), or a save point whose derived maps disagree with it
    /// ([`Invariant::SavePointMismatch`], with nothing written: not retryable).
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
        // The bounds of the state the save point names: the lost tiles' edits aren't in it.
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        project
            .write_bounds_many(
                self.bounds.all_changed().map(|(key, kept)| (*key, kept)),
                Stamp::ZERO,
            )
            .map_err(CacheError::Save)?;
        project
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Save))?;
        let base = saved_base(project)?;
        Ok(Saved {
            base,
            stamp: self.last_write,
            lost,
            bounds: self.bounds.changes(),
        })
    }

    /// Swaps in the save point `saved` proves, as the base, and `working`, a fresh working file
    /// over it; returns the working file it replaces.
    ///
    /// `saved` must be current: made by [`TileStore::save_into`] with no write since. Then every
    /// tile edited since the last rebase is in the new save point, and the old working file's
    /// records are no longer needed. That is checked, not assumed: every tile of the old
    /// working file and of the old base must be in the new save point (the old base's checked
    /// one by one only if the save point isn't known to extend it, see
    /// [`TileStore::missing_from_save`]), but for those the save listed as lost ([`Saved::lost`]),
    /// whose last saved state the store reads from now on (a lost tile's resident slot is
    /// dropped, or, if a pin holds it, filled again in place from the saved record, and every
    /// evaluation that followed from it is dropped too, as it may have come from the lost
    /// edit). Dirty slots are clean after the rebase (the save wrote them), and every other
    /// resident correction tile is kept, as it is what was just saved; so are the other cached
    /// evaluations, which stay valid: nothing they read has changed. Prefetches in flight are
    /// dropped (their workers stop and are joined), and later ones read the new files. The op
    /// generator continues, past the base's head if that is newer, and the current op ends: the
    /// next edit starts a new one, so the edits after a save point are newer than its head.
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
        let Saved {
            base,
            stamp,
            lost,
            bounds,
        } = saved;
        if stamp != self.last_write || bounds != self.bounds.changes() {
            return Err(CacheError::StaleSave);
        }
        if base.meta().mesh_hash() != self.mesh_hash || working.meta().mesh_hash() != self.mesh_hash
        {
            return Err(CacheError::MeshMismatch);
        }
        check_layers(working.meta(), Some(&base))?;
        if let Some(missing) = self.missing_from_save(&base, &lost) {
            return Err(Invariant::NotInSave(missing).into());
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
        // A lost tile's slot doesn't hold the saved state, and has nothing to write: it is
        // dropped, or, if a pin holds it, reloaded in place once the store reads the new files.
        let mut pinned = Vec::new();
        for key in &lost {
            if self.drop_lost_slot(key)? {
                pinned.push(*key);
            }
        }
        // What was evaluated from a lost edit isn't the saved state the store reads from now on.
        self.invalidate_edited(lost.iter().copied());
        self.forget_evicted_ops();
        #[cfg(test)]
        self.assert_known_ops_resident();
        self.since_rebase.clear();
        self.bounds.rebased();
        self.generation += 1;
        self.working_failed = false;
        // The old workers read the old files; dropping them joins them and drops their results.
        self.prefetch = Prefetcher::new();
        self.next_op = next_op;
        self.op = None;
        self.base = Some(base);
        let old = std::mem::replace(&mut self.file, working);
        for key in &pinned {
            self.reload_lost(key);
        }
        Ok(old)
    }

    /// The first tile the store could read that `saved`, a new save point, lacks, but for
    /// `lost`, the tiles the save listed as lost; `None` if it lacks none. What a rebase onto
    /// it checks, so that no tile the store reads is lost by moving on (a bug in a save).
    ///
    /// The tiles the store can read are the working file's, every tile edited since the last
    /// rebase, and the old base's. The first two are what the save wrote, and grow with the
    /// edits; the old base's are every tile of the project. So when the new save point is known
    /// to hold every tile of the old base ([`BaseFile::extends`]: it was made from the old
    /// base's save point, in the same open file, with records only put in), they aren't looked
    /// up one by one, and the check costs what the save wrote. Otherwise (a project saved under
    /// a new name, a save point loaded from the file, or one that doesn't descend from the
    /// base at all) every tile of the old base is looked up too.
    ///
    /// Tests run the full check as well whenever the short one passes.
    ///
    /// # Panics
    ///
    /// In tests only, if the full check finds a tile the short one missed: a save point taken
    /// to extend the base that doesn't.
    fn missing_from_save(&self, saved: &BaseFile, lost: &BTreeSet<TileKey>) -> Option<TileKey> {
        let old_base = self.base.as_ref().filter(|old| !saved.extends(old));
        let missing = self.missing_among(saved, lost, old_base);
        #[cfg(test)]
        if missing.is_none() {
            assert_eq!(
                self.missing_among(saved, lost, self.base.as_ref()),
                None,
                "a save point taken to extend the base lacks a tile of it"
            );
        }
        missing
    }

    /// The first tile of the working file, of those edited since the last rebase, or of
    /// `old_base` if given, that `saved` lacks and `lost` doesn't list (see
    /// [`TileStore::missing_from_save`]).
    fn missing_among(
        &self,
        saved: &BaseFile,
        lost: &BTreeSet<TileKey>,
        old_base: Option<&BaseFile>,
    ) -> Option<TileKey> {
        self.file
            .keys()
            .chain(&self.since_rebase)
            .chain(old_base.into_iter().flat_map(BaseFile::keys))
            .find(|key| !saved.contains(key) && !lost.contains(key))
            .copied()
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
    /// (synced, a save point, synced again). Every bound of the old working file and every bound
    /// changed since the last rebase go across too, as they are now; those a lost tile's edit could
    /// change, as unknown.
    ///
    /// # Errors
    ///
    /// [`CacheError::MeshMismatch`] or [`CacheError::WrongBase`] if `fresh` isn't over the
    /// store's mesh, or doesn't name the same base as the working file; [`CacheError::WriteBack`]
    /// if writing or saving `fresh` fails; [`CacheError::Internal`] if its save point's derived
    /// maps disagree with it ([`Invariant::SavePointMismatch`]).
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
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        let carried: BTreeSet<TileKey> =
            self.file.bounds_locations().map(|(key, _)| *key).collect();
        for &key in &carried {
            if self.bounds.changed(&key).is_none() {
                let copied = match self.file.read_bounds_record(&key) {
                    Ok(Some(raw)) => fresh.append_bounds_copy(&key, &raw),
                    // A bound is derived: one that doesn't read goes across as unknown.
                    Ok(None) | Err(_) => fresh.write_bounds(key, None, Stamp::ZERO),
                };
                copied.map_err(CacheError::WriteBack)?;
            }
        }
        // What the old file held, as it is now, and what a flush would write; not the derived
        // bounds, which no recovery file holds.
        let bounds = &self.bounds;
        fresh
            .write_bounds_many(
                bounds
                    .all_changed()
                    .filter(|(key, _)| carried.contains(key) || bounds.is_unflushed(key))
                    .map(|(key, kept)| (*key, kept)),
                Stamp::ZERO,
            )
            .map_err(CacheError::WriteBack)?;
        fresh
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::WriteBack))?;
        Ok(Filled {
            file: fresh,
            lost,
            cleaned,
            stamp: self.last_write,
            bounds: self.bounds.changes(),
        })
    }

    /// Swaps in the working file [`TileStore::fill_roll_over`] filled; returns the old working
    /// file and the tiles that couldn't be carried across (lost).
    ///
    /// Every slot written to the new file is clean, streaming consumers' marks are emptied (all
    /// they held is in it), and the generation moves on, so a flush started before is
    /// abandoned. The base is unchanged, and so are the tiles to save into the project, but for
    /// the lost ones: they are in no file the store writes, so the next save leaves them at
    /// their last saved state instead of failing to find them. A lost tile's slot, if it has
    /// one (a broken one), is dropped, or, if a pin holds it, filled again in place from the
    /// tile's last record, and every evaluation that followed from a lost tile is dropped, as it
    /// may have come from the lost edit.
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
            bounds,
        } = filled;
        if stamp != self.last_write || bounds != self.bounds.changes() {
            return Err(CacheError::StaleSave);
        }
        // Drained first, so the feed can't bring a lost key back.
        self.drain_saves();
        // A lost tile's slot, if it has one, is broken: dropped as a rebase drops one, or, if a
        // pin holds it, reloaded in place once the store reads the new file.
        let mut pinned = Vec::new();
        for key in &lost {
            self.since_rebase.remove(key);
            if self.drop_lost_slot(key)? {
                pinned.push(*key);
            }
        }
        // What was evaluated from a lost edit isn't what the store reads from now on.
        self.invalidate_edited(lost.iter().copied());
        for (key, id) in cleaned {
            self.corrections.set_dirty(id, false)?;
            if let Some(write) = self.unsaved.remove(&key) {
                self.known_ops.insert(key, write.op);
            }
        }
        self.forget_evicted_ops();
        #[cfg(test)]
        self.assert_known_ops_resident();
        let marks: Vec<ConsumerId> = self.streams.keys().copied().collect();
        for id in marks {
            if let Some(mark) = self.streams.get_mut(&id) {
                mark.clear();
            }
        }
        // The new file holds every bound as it is now.
        self.bounds.all_flushed();
        self.generation += 1;
        self.working_failed = false;
        // The workers' jobs name records of the old file.
        self.prefetch = Prefetcher::new();
        let old = std::mem::replace(&mut self.file, file);
        for key in &pinned {
            self.reload_lost(key);
        }
        Ok((old, lost))
    }

    /// Moves every key save's consumer hasn't drained from the change feed into
    /// [`TileStore::since_rebase`].
    ///
    /// Draining early changes nothing a save sees, as `since_rebase` is a set. A save drains it
    /// ([`TileStore::save_into`], [`TileStore::save`]), as do a roll-over and a lost tile, and so
    /// does every flush ([`TileStore::snapshot_dirty`]); with no recovery writer running, the
    /// project drains it on every call that would have advanced one
    /// ([`crate::Project::with_writer`]). So the feed's backlog stays bounded between saves.
    pub(crate) fn drain_saves(&mut self) {
        let drained = self.feed.drain(self.save_consumer);
        self.since_rebase
            .extend(drained.into_iter().map(|(key, _)| key));
    }

    /// The change feed's backlog: entries some consumer hasn't drained yet.
    #[cfg(test)]
    pub(crate) fn feed_backlog(&self) -> usize {
        self.feed.backlog()
    }

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

    /// Writes every dirty correction tile, and every spatial-index bound the working file lacks,
    /// to the working file and records a save point there. Returns its
    /// offset.
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
    /// [`CacheError::FailedWorkingFile`] while the working file is a recovery file whose
    /// writer failed, as the [`crate::Project`] that owns the store marks it: a save point there
    /// would vouch for records a failed sync may have lost.
    /// [`CacheError::WriteBack`] if a tile's or a bound's write fails; [`CacheError::Eval`] if
    /// listing what the edits made stale fails (never, for tiles of the store's mesh);
    /// [`CacheError::Store`] if the save point's write does; [`CacheError::Internal`] if the
    /// save point's derived maps disagree with it ([`Invariant::SavePointMismatch`]).
    pub fn save(&mut self) -> Result<u64, CacheError> {
        if !self.streams.is_empty() {
            return Err(CacheError::Streaming);
        }
        if self.working_failed {
            return Err(CacheError::FailedWorkingFile);
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
        self.write_unsaved_bounds()?;
        self.file
            .save()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Store))
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
    /// snapshot, or keys of its flush mark still unconfirmed; or bounds the working file lacks.
    pub(crate) fn has_unflushed(&self, id: ConsumerId) -> bool {
        self.feed.has_pending(id)
            || self.streams.get(&id).is_some_and(|mark| !mark.is_empty())
            || self.bounds.has_unflushed()
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
    /// Save's consumer is drained into [`TileStore::since_rebase`] here too, so the feed holds
    /// no more than one flush's worth of entries between saves.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale. The mark keeps every key then.
    pub(crate) fn snapshot_dirty(
        &mut self,
        id: ConsumerId,
        skipped: &mut Vec<TileKey>,
    ) -> Result<Vec<Snapshot>, CacheError> {
        self.drain_saves();
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

    /// The working file is a recovery file whose writer failed (see
    /// [`TileStore::working_failed`]): no save point is written there again, until the store
    /// moves to another working file.
    pub(crate) const fn fail_working(&mut self) {
        self.working_failed = true;
    }

    /// Whether the working file is a recovery file whose writer failed, which the store hasn't
    /// moved off yet ([`TileStore::fail_working`]).
    pub(crate) const fn is_working_failed(&self) -> bool {
        self.working_failed
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

    /// Where the last record the working file's next save point names ends
    /// ([`ProjectFile::named_end`]): a sync must cover it before that save point is written.
    pub(crate) fn working_named_end(&self) -> u64 {
        self.file.named_end()
    }

    /// Whether the working file has records (tiles or bounds) written since its last save
    /// point.
    pub(crate) fn has_unsaved_records(&self) -> bool {
        self.file.has_unsaved_records()
    }

    /// Appends a save point to the working file naming every tile and bound written to it so far,
    /// without syncing (see [`ProjectFile::write_save_point`]), and returns its offset. Dirty
    /// slots are not written: their edits are in no save point yet.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if the write fails; [`CacheError::Internal`] if the save point's
    /// derived maps disagree with it ([`Invariant::SavePointMismatch`]).
    pub(crate) fn write_save_point(&mut self) -> Result<u64, CacheError> {
        self.file
            .write_save_point()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Store))
    }

    /// Records `bounds`, computed from the store's current state (by the spatial index), as
    /// the bounds the store keeps for those tiles. Those replacing an edit's change still to be
    /// flushed go to the working file with the next flush; the rest are derived, and go to the
    /// project file with its next save (see [`crate::bounds`]).
    ///
    /// The writes since the last call are settled first (see [`crate::bounds`]), so the bounds
    /// they made unknown are replaced, not the other way round.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing a written tile's readers fails (never, for tiles of the
    /// store's mesh). Nothing is recorded then.
    pub(crate) fn record_bounds(
        &mut self,
        bounds: &[(TileKey, TileBounds)],
    ) -> Result<(), CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        for &(key, tile) in bounds {
            self.bounds.record(key, tile);
        }
        Ok(())
    }

    /// The bounds the store keeps for each of `keys`, in order: `None` where none is known (none
    /// was ever recorded, or a write since made it unknown), and the tile must be evaluated.
    ///
    /// Each is read as a correction tile is (see [`crate::bounds`]): the change since the last
    /// save if there is one, else the working file's record, else the base's. Records are read
    /// together, a file at a time; one that doesn't read counts as unknown.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::Store`] if a read
    /// fails.
    pub(crate) fn stored_bounds(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<TileBounds>>, CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let mut found = vec![None; keys.len()];
        let (mut working, mut base) = (Vec::new(), Vec::new());
        for (n, key) in keys.iter().enumerate() {
            if let Some(changed) = self.bounds.changed(key) {
                if let Some(slot) = found.get_mut(n) {
                    *slot = changed.bounds();
                }
            } else if let Some(loc) = self.file.locate_bounds(key) {
                working.push((n, (*key, loc)));
            } else if let Some(loc) = self.base.as_ref().and_then(|b| b.locate_bounds(key)) {
                base.push((n, (*key, loc)));
            }
        }
        let entries = |at: &[(usize, (TileKey, Loc))]| -> Vec<(TileKey, Loc)> {
            at.iter().map(|&(_, entry)| entry).collect()
        };
        let read_working = self.file.read_bounds(&entries(&working))?;
        let read_base = match &self.base {
            Some(file) if !base.is_empty() => file.read_bounds(&entries(&base))?,
            _ => Vec::new(),
        };
        for ((n, _), bounds) in working
            .iter()
            .zip(read_working)
            .chain(base.iter().zip(read_base))
        {
            if let Some(slot) = found.get_mut(*n) {
                *slot = bounds;
            }
        }
        Ok(found)
    }

    /// The bounds part of a recovery flush, at its cut: loses the `lost` tiles, left out of the
    /// flush (see [`TileStore::lose_left_out`]), settles the writes so far, makes unknown the
    /// bounds the lost tiles' edits could have changed, and
    /// appends, with one write, every bound edits changed that the working file lacks, placed at
    /// the cut. Derived bounds stay out (see [`crate::bounds`]), so the cost grows with the edits
    /// since the last flush.
    ///
    /// Called right after the flush's snapshot, on the owner thread, before any later write:
    /// the bounds appended are those of the state the snapshot copies, so the flush's save point
    /// names tiles and bounds of one state.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if an
    /// append fails: the bounds are kept for the next flush.
    pub(crate) fn flush_bounds(&mut self, lost: &[TileKey]) -> Result<(), CacheError> {
        self.lose_left_out(lost)?;
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        self.write_unflushed_bounds()
    }

    /// Loses the `lost` tiles a flush leaves out (broken slots), as a save loses a tile whose
    /// latest edit it can't read (see [`TileStore::rebase`]): the store stops reading the edit
    /// the recovery file doesn't hold, and drops every evaluation that followed from it, so
    /// nothing evaluated later (and no bound recorded from it) describes a state no save point
    /// holds.
    ///
    /// Each slot is dropped, or, if a pin holds it, filled again in place from the tile's last
    /// record (see [`TileStore::reload_lost`]), as a rebase does with a lost tile's. The tile is
    /// read from the working file's record from then on, else from the base: if the working
    /// file has no record of it, it is at its last saved state again, and no longer among the
    /// tiles the next save copies.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn lose_left_out(&mut self, lost: &[TileKey]) -> Result<(), CacheError> {
        if lost.is_empty() {
            return Ok(());
        }
        // Drained first, so the feed can't bring a lost key back among the tiles to save.
        self.drain_saves();
        for key in lost {
            if self.drop_lost_slot(key)? {
                self.reload_lost(key);
            }
            if !self.file.contains(key) {
                self.since_rebase.remove(key);
            }
        }
        self.invalidate_edited(lost.iter().copied());
        Ok(())
    }

    /// Forgets a lost tile's latest write and drops its correction slot, if it has one: its
    /// edit is in no file the store reads. Returns whether a pin holds the slot, which then
    /// stays, marked clean, for [`TileStore::reload_lost`] to fill again from the tile's last
    /// record once the store reads the files it is to read from.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn drop_lost_slot(&mut self, key: &TileKey) -> Result<bool, CacheError> {
        self.unsaved.remove(key);
        let Some(id) = self.corrections.find(key) else {
            return Ok(false);
        };
        self.corrections.set_dirty(id, false)?;
        if self.corrections.is_pinned(key) {
            return Ok(true);
        }
        self.corrections.remove(key)?;
        self.known_ops.remove(key);
        Ok(false)
    }

    /// Fills a lost tile's slot, kept because a pin holds it, again in place from the tile's
    /// last record (the working file's, else the base's, else zero), as loading it would: what
    /// the slot holds is then what the files hold, and an edit after the pin goes starts from
    /// it, never from the lost edit. The slot stays clean.
    ///
    /// If the record can't be read, the slot is left holding no tile, as a failed load leaves
    /// one: every read or edit of it fails until it leaves the pool (a clean slot, so dropping
    /// it writes nothing), and the tile is read from its record again after that.
    fn reload_lost(&mut self, key: &TileKey) {
        let Some(id) = self.corrections.find(key) else {
            return;
        };
        let fill = Fill {
            key: *key,
            id,
            found: self.locate(key),
        };
        let Some(slot) = self.corrections.get_mut(id) else {
            return;
        };
        let filled = fill_slot(
            &self.file,
            self.base.as_ref(),
            &fill,
            slot,
            &mut self.read_scratch,
        );
        match filled {
            Ok(Some(op)) => {
                self.known_ops.insert(*key, op);
            }
            Ok(None) | Err(_) => {
                self.known_ops.remove(key);
            }
        }
    }

    /// Appends, with one write, every bound edits changed that the working file lacks (not the
    /// derived ones, see [`crate::bounds`]), placed at the latest write: the state they
    /// describe. Settles the writes first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if
    /// the append fails: the bounds stay unflushed.
    fn write_unflushed_bounds(&mut self) -> Result<(), CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let unflushed: Vec<TileKey> = self.bounds.unflushed().map(|(key, _)| *key).collect();
        self.file
            .write_bounds_many(
                self.bounds.unflushed().map(|(key, kept)| (*key, kept)),
                self.last_write,
            )
            .map_err(CacheError::WriteBack)?;
        for key in &unflushed {
            self.bounds.flushed(key);
        }
        Ok(())
    }

    /// Appends every bound a save of the working file itself writes (see [`Book::unsaved`]):
    /// those a flush would, and the derived ones, with one write. Settles the writes first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if
    /// the append fails: the bounds stay as they were.
    fn write_unsaved_bounds(&mut self) -> Result<(), CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let unsaved: Vec<TileKey> = self.bounds.unsaved().map(|(key, _)| *key).collect();
        self.file
            .write_bounds_many(
                self.bounds.unsaved().map(|(key, kept)| (*key, kept)),
                self.last_write,
            )
            .map_err(CacheError::WriteBack)?;
        for key in &unsaved {
            self.bounds.saved(key);
        }
        Ok(())
    }

    /// Which of the tiles whose copies a flush couldn't encode (`broken`) are lost in memory:
    /// those whose live slot is dirty and broken too (holds no valid tile). A valid slot means
    /// the failure was transient (a compression error, say), whether or not it was edited since
    /// the snapshot: it stays dirty, in the flush mark, for the next flush to try again. A slot
    /// clean or gone was written by someone else since, encoded.
    pub(crate) fn left_out_at_reply(&self, broken: &[TileKey]) -> Vec<TileKey> {
        broken
            .iter()
            .filter(|key| {
                self.corrections
                    .find(key)
                    .filter(|&id| self.corrections.is_dirty(id) == Some(true))
                    .is_some_and(|id| {
                        self.corrections
                            .get(id)
                            .is_none_or(|slot| slot.to_tile().is_err())
                    })
            })
            .copied()
            .collect()
    }

    /// The two halves of a flush leaving out the `left_out` tiles, whose copies didn't encode.
    ///
    /// In the file: every bound their at-cut edits could have changed is appended as unknown,
    /// placed at the flush's cut, whatever their live slots hold now: the save point holds their
    /// older state, and the bounds the flush appended at its snapshot may describe the edits it
    /// lacks. The store's own are made unknown as well, and the next flush writes them again.
    ///
    /// In memory: only the `lost` ones (broken slots, see [`TileStore::left_out_at_reply`])
    /// are lost (see [`TileStore::lose_left_out`]); the others stay dirty for the next flush.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if an
    /// append fails.
    pub(crate) fn forget_bounds_at_cut(
        &mut self,
        left_out: &[TileKey],
        lost: &[TileKey],
    ) -> Result<(), CacheError> {
        self.lose_left_out(lost)?;
        if left_out.is_empty() {
            return Ok(());
        }
        let stale = self
            .bounds
            .forget(&self.mesh, &self.deps, left_out.iter().copied())?;
        let at = self.file.cut_at().unwrap_or(self.last_write);
        self.file
            .write_bounds_many(stale.iter().map(|&key| (key, None)), at)
            .map_err(CacheError::WriteBack)
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
    /// [`parent_tiles`], listed on the workers (see [`recorded_or_listed`]).
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
    /// they read, pinned: on the workers, straight into each tile's slot
    /// ([`TileStore::normals_round`]), then recorded in key order (see
    /// [`TileStore::normals_installed`]).
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
    ) -> Result<(), CacheError> {
        let tasks: Vec<(TileKey, Option<&BTreeSet<TileKey>>)> = wanted
            .iter()
            .copied()
            .zip(stale.iter().map(Option::as_ref))
            .collect();
        self.normals_round(&tasks, held)?;
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
    ) -> Result<(), CacheError> {
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
        let mut outs = tasks
            .iter()
            .map(|(key, _)| {
                let at = held_index(held, key)?;
                normals
                    .get_mut(at)
                    .and_then(Option::take)
                    .ok_or(Invariant::NamedTwice(*key))
            })
            .collect::<Result<Vec<_>, Invariant>>()?;
        let mesh = &*self.mesh;
        #[cfg(test)]
        let failing = self.failing_normals;
        each_parallel(&self.workers, tasks, &mut outs, |&(key, since), out| {
            #[cfg(test)]
            if failing == Some(key) {
                return Err(Invariant::StaleHandle.into());
            }
            let source = Resident::new(mesh, Tiles::Split(&positions));
            let out = prefix_mut(out.as_mut_slice(), vertex_count(&key))?;
            let region = Region::of_tile(&key);
            match since {
                None => normals_into(mesh, &source, region, out)?,
                Some(since) => {
                    refresh_normals_into(mesh, &source, region, |k| since.contains(k), out)?;
                }
            }
            Ok(())
        })
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

/// The border vertices of a tile with `side` vertices a side, in row-major order.
///
/// That is the first row, then the first and last vertex of each row inside, then the last row:
/// the order [`TileEdit::update`] visits them in.
fn border_order(side: u32) -> impl Iterator<Item = (u32, u32)> {
    let last = side.saturating_sub(1);
    (0..side).flat_map(move |j| {
        let step = if j == 0 || j == last { 1 } else { last.max(1) };
        (0..side)
            .step_by(usize::try_from(step).unwrap_or(1))
            .map(move |i| (i, j))
    })
}

/// A tile's border corrections, read at once in [`border_order`] for the tile's border writes
/// ([`TileEdit::update_border_one`]), instead of a slot lookup per vertex.
#[derive(Debug)]
struct BorderValues {
    /// The tile's vertices a side.
    side: u32,
    /// The corrections, in [`border_order`].
    values: Vec<Vec3>,
}

impl BorderValues {
    /// The correction read for border vertex `(i, j)`; `None` for a vertex inside the tile or
    /// outside it.
    fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        let last = self.side.checked_sub(1)?;
        if i > last || j > last {
            return None;
        }
        let at = if j == 0 {
            i
        } else if j == last {
            self.side + 2 * (last - 1) + i
        } else if i == 0 || i == last {
            self.side + 2 * (j - 1) + u32::from(i == last)
        } else {
            return None;
        };
        self.values.get(usize::try_from(at).ok()?).copied()
    }
}

/// What [`update_interior`] did to a tile's interior.
#[derive(Debug)]
struct InteriorPass {
    /// The first row with an interior vertex written: where, in row-major order, the tile's
    /// first interior write falls (after the row's first border vertex, before its last).
    first_row: Option<u32>,
    /// The row where a vertex failed, and why: nothing after it in row-major order was written.
    failed: Option<(u32, CacheError)>,
}

/// Runs `f` over the interior vertices of row `j` of `view`, in order, setting what it returns
/// ([`CorrectionViewMut::update_interior_row`]).
///
/// Returns whether anything was written, and the error of a value that isn't finite, which
/// stops the row (nothing is written for it).
///
/// # Errors
///
/// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
/// [`CorrectionError::TooLarge`] for a value no tile can hold; [`CacheError::Internal`] with
/// [`Invariant::CorrectionSlot`] and the slot's own error for a row outside the tile's interior
/// (the store asks only for interior rows).
fn update_row(
    view: &mut CorrectionViewMut<'_>,
    j: u32,
    f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
) -> (bool, Result<(), CacheError>) {
    let RowUpdate { wrote, error } = view.update_interior_row(j, f);
    (wrote, error.map_or(Ok(()), |e| Err(refused_value(e))))
}

/// The store's error for a value a correction slot refused to take.
///
/// The caller's ([`CacheError::Correction`]) for a value no tile can hold,
/// [`CorrectionError::TooLarge`] or [`CorrectionError::NotFinite`]; else the store's own
/// ([`Invariant::CorrectionSlot`]), as the store writes only inside a tile it holds.
fn refused_value(e: CorrectionError) -> CacheError {
    if matches!(e, CorrectionError::TooLarge | CorrectionError::NotFinite) {
        CacheError::Correction(e)
    } else {
        Invariant::CorrectionSlot(e).into()
    }
}

/// The half of [`TileStore::update_tiles`] that runs on the workers, for one tile.
///
/// Runs `f` over every interior vertex of the tile in `slot` (`side` vertices a side), row by
/// row in row-major order, setting what it returns, as [`TileEdit::update`] would. Stops at
/// the first vertex that fails.
fn update_interior(
    slot: &mut CorrectionSlot,
    side: u32,
    mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
) -> InteriorPass {
    let mut pass = InteriorPass {
        first_row: None,
        failed: None,
    };
    let mut view = match slot.view_mut() {
        Ok(view) => view,
        Err(e) => {
            pass.failed = Some((0, Invariant::CorrectionSlot(e).into()));
            return pass;
        }
    };
    for j in 1..side.saturating_sub(1) {
        let (wrote, row) = update_row(&mut view, j, &mut f);
        if wrote && pass.first_row.is_none() {
            pass.first_row = Some(j);
        }
        if let Err(e) = row {
            pass.failed = Some((j, e));
            break;
        }
    }
    pass
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
    /// [`CorrectionError::OutOfTile`], before anything is written, or with
    /// [`CorrectionError::TooLarge`] for an interior value no tile's scale can hold (the tile is
    /// unchanged); [`CacheError::Store`],
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
        let region = Region::of_tile(&self.key);
        let copies = self.store.mesh.topology().copies(
            self.key.face(),
            self.key.level(),
            region.x0() + i,
            region.y0() + j,
        );
        self.set_copies(&copies, c)
    }

    /// Sets every one of `copies`, the copies of one vertex in [`geom::Topology::copies`]'s order,
    /// to `c` (finite): each one's tile is loaded and pinned first, in that order, then each
    /// copy is written, as [`TileEdit::set`] does for a border vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`] for a neighbouring tile that can't be loaded, before anything of
    /// this vertex is written.
    fn set_copies(&mut self, copies: &[VertexCopy], c: Vec3) -> Result<(), CacheError> {
        let level = self.key.level();
        let mut at = [0; MAX_COPIES];
        for (copy, place) in copies.iter().zip(&mut at) {
            *place = self.touch(copy.key(level))?;
        }
        for (copy, &n) in copies.iter().zip(&at) {
            self.write(n, (copy.i, copy.j), c)?;
        }
        Ok(())
    }

    /// [`TileEdit::set`] for border vertex `(i, j)`, its copies read from `border`, the tile's
    /// [`TileBorder`]: the same writes in the same order, without finding the copies again for
    /// every vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn set_border(
        &mut self,
        border: &TileBorder,
        i: u32,
        j: u32,
        c: Vec3,
    ) -> Result<(), CacheError> {
        let Some(copies) = border.copies(i, j) else {
            return self.set(i, j, c);
        };
        if !c.is_finite() {
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        self.set_copies(&copies, c)
    }

    /// Rewrites the tile's corrections vertex by vertex: `f` sees every vertex in row-major
    /// order, with its coordinates and its correction as stored, and where it returns a new
    /// value, that is set as [`TileEdit::set`] sets it, before `f` sees the next vertex.
    ///
    /// The same as [`TileEdit::get`] then [`TileEdit::set`] for each vertex in that order, and
    /// so the same bits: order matters, as a value that outgrows the tile's scale re-quantizes
    /// its interior. It is cheaper, as each run of interior vertices of a row (which have no
    /// copies in other tiles) is read and written through one borrowed view instead of a
    /// lookup per call; border vertices go through [`TileEdit::set`], copies and all.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`], for the first vertex that fails; the vertices before it stay
    /// written.
    pub fn update(
        &mut self,
        mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let side = self.key.side();
        let last = side - 1;
        for j in 0..side {
            if j == 0 || j == last {
                for i in 0..side {
                    self.update_one(i, j, &mut f)?;
                }
            } else {
                self.update_one(0, j, &mut f)?;
                self.update_interior_row(j, &mut f)?;
                self.update_one(last, j, &mut f)?;
            }
        }
        Ok(())
    }

    /// [`TileEdit::update`] for vertex `(i, j)` alone, through [`TileEdit::get`] and
    /// [`TileEdit::set`].
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn update_one(
        &mut self,
        i: u32,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        match f(i, j, self.get(i, j)?) {
            Some(c) => self.set(i, j, c),
            None => Ok(()),
        }
    }

    /// [`TileEdit::update`] for the interior vertices of row `j` (`1..side - 1`, for a row
    /// inside the tile), through one view of the tile's slot. The tile is marked written after
    /// the row if anything in it was, before any other tile can be touched: as
    /// [`TileEdit::write`] marks it at its first write.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
    /// [`CorrectionError::TooLarge`] for a value `f` returns that no tile can hold (nothing is
    /// written for it, the row's earlier writes stay);
    /// [`CacheError::Internal`] for a stale handle or a broken slot.
    fn update_interior_row(
        &mut self,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let tile = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get_mut(tile.id)
            .ok_or(Invariant::StaleHandle)?;
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        let (wrote, row) = update_row(&mut view, j, f);
        if wrote {
            self.mark_own_written()?;
        }
        row
    }

    /// The other half of [`TileStore::update_tiles`] for this tile, whose interior `pass`
    /// wrote: runs `f` over the border vertices in row-major order, setting what it returns
    /// with [`TileEdit::set`], and marks the tile written where its first interior write falls
    /// in that order (if no border write marked it first). Stops where the interior pass
    /// failed, as [`TileEdit::update`] would have.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`] for a border vertex; else the interior pass's error. Whatever
    /// fails, and wherever in row-major order, a tile whose interior the pass wrote is marked
    /// written: its slot holds the pass's writes, before the failing vertex or not.
    fn update_border(
        &mut self,
        pass: InteriorPass,
        f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let wrote = pass.first_row.is_some();
        let mut outcome = self.border_writes(pass, f);
        if outcome.is_err() && wrote {
            keep_first(&mut outcome, self.mark_own_written());
        }
        outcome
    }

    /// [`TileEdit::update_border`]'s writes, in row-major order, stopping at the first that
    /// fails; the tile is marked written at its first interior row only if no write before it
    /// fails.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::update_border`].
    fn border_writes(
        &mut self,
        pass: InteriorPass,
        mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let side = self.key.side();
        let last = side - 1;
        let InteriorPass { first_row, failed } = pass;
        let (failed_row, mut failure) = match failed {
            Some((row, e)) => (Some(row), Some(e)),
            None => (None, None),
        };
        if failed_row == Some(0)
            && let Some(e) = failure.take()
        {
            return Err(e);
        }
        let border = match self.store.tile_border(&self.key) {
            Some(copies) => Some((self.border_values()?, copies)),
            None => None,
        };
        for j in 0..side {
            if j == 0 || j == last {
                for i in 0..side {
                    self.update_border_one(border.as_ref(), i, j, &mut f)?;
                }
                continue;
            }
            self.update_border_one(border.as_ref(), 0, j, &mut f)?;
            if first_row == Some(j) {
                self.mark_own_written()?;
            }
            if failed_row == Some(j)
                && let Some(e) = failure.take()
            {
                return Err(e);
            }
            self.update_border_one(border.as_ref(), last, j, &mut f)?;
        }
        Ok(())
    }

    /// The corrections of the tile's border vertices as they are now, read through one view
    /// (see [`BorderValues`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle or a broken slot.
    fn border_values(&self) -> Result<BorderValues, CacheError> {
        let side = self.key.side();
        let own = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get(own.id)
            .ok_or(Invariant::StaleHandle)?;
        let view = slot.view().map_err(Invariant::CorrectionSlot)?;
        let values = border_order(side)
            .map(|(i, j)| view.get(i, j).ok_or(Invariant::StaleHandle))
            .collect::<Result<_, _>>()?;
        Ok(BorderValues { side, values })
    }

    /// [`TileEdit::update_one`] for border vertex `(i, j)`; where the tile has a [`TileBorder`],
    /// with its correction from the [`BorderValues`] read before the tile's border writes and
    /// its copies from the border.
    ///
    /// The values read before are the values [`TileEdit::get`] would read now: the writes since
    /// went to other vertices' copies, and a tile with a [`TileBorder`] holds no vertex twice,
    /// so none of them landed on this vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn update_border_one(
        &mut self,
        border: Option<&(BorderValues, TileBorder)>,
        i: u32,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let Some((values, copies)) = border else {
            return self.update_one(i, j, f);
        };
        let current = match values.get(i, j) {
            Some(c) => c,
            None => self.get(i, j)?,
        };
        match f(i, j, current) {
            Some(c) => self.set_border(copies, i, j, c),
            None => Ok(()),
        }
    }

    /// Marks the edited tile written, if no write has yet: dirty, on the change feed with the
    /// edit's op, and its evaluations dropped when the edit ends (see [`TileEdit::write`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn mark_own_written(&mut self) -> Result<(), CacheError> {
        let tile = self.touched.first_mut().ok_or(Invariant::StaleHandle)?;
        if !tile.written {
            tile.written = true;
            let (key, id) = (tile.key, tile.id);
            self.store.mark_written(key, id, self.op)?;
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
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        view.set(i, j, c).map_err(refused_value)?;
        if !tile.written {
            tile.written = true;
            self.store.mark_written(tile.key, tile.id, self.op)?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;
