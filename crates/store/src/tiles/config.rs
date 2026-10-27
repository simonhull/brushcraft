//! How a [`TileStore`](super::TileStore) is set up ([`StoreConfig`]), the budget it needs, and the
//! fixed sizes of its evaluation rounds and pin batches.

use geom::{BaseMesh, Prediction};

use super::error::CacheError;
use crate::pool::Pool;
use crate::slot::{CorrectionSlot, EvaluatedSlot, SIDE};
use crate::workers::Workers;

/// How a [`TileStore`](super::TileStore) is set up.
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
    /// bytes. How the pools share it is the store's memory policy (see
    /// [`TileStore`](super::TileStore)): corrections take evaluated chunks only up to a
    /// quarter of the pools' share (the budget less the reserved shares), and never below the
    /// evaluated pool's floor.
    ///
    /// The evaluation scratch (about 3 MB), the read buffers, the workers' own allocations (a
    /// few hundred KB each at most) and the thread stacks (2 MiB reserved per worker) sit
    /// outside it: a few MB besides the budget, plus the stacks (see "Memory outside the
    /// budget" on [`TileStore`](super::TileStore)).
    pub budget_bytes: usize,
    /// Bytes per pool chunk, the unit both pools grow and shrink by and are charged in (whole
    /// slots fill it, the rest of a chunk is charged unused):
    /// [`StoreConfig::DEFAULT_CHUNK_BYTES`] in the app, smaller in tests.
    pub chunk_bytes: usize,
    /// The site (machine or session) the store's op ids carry; any but
    /// [`RESERVED_SITE`](crate::op::RESERVED_SITE).
    pub site: u16,
    /// How many worker threads evaluate tiles in parallel (see
    /// [`TileStore::evaluate_many`](super::TileStore::evaluate_many)): by default one per
    /// physical core ([`Workers::PhysicalCores`], see [`crate::Workers`] for why). Results are
    /// the same bits whatever the count; tests pin it to compare.
    pub workers: Workers,
    /// How many tiles' predictions the store keeps for predicting them again in place (see
    /// "Kept predictions" on [`TileStore`](super::TileStore)):
    /// [`StoreConfig::DEFAULT_KEPT_PREDICTIONS`] unless set. Each is charged to the budget at
    /// the largest tile's size, [`geom::Prediction::TILE_BYTES`] (about 206 KB); zero keeps
    /// none, and every tile is predicted in full. Results are the same bits either way.
    /// [`TileStore::open`](super::TileStore::open) refuses a count whose share leaves the
    /// pools less than their minimum.
    pub kept_predictions: usize,
    /// How many tiles' stroke state the budget sets aside: the `f32` start and summed deltas a
    /// brush stroke keeps per vertex of the tiles it is editing, so that deltas smaller than a
    /// quantization step aren't rounded away dab by dab (the `sculpt` crate holds the state;
    /// the store only reserves for it and reports the count,
    /// [`TileStore::stroke_tiles`](super::TileStore::stroke_tiles)).
    /// [`StoreConfig::DEFAULT_STROKE_TILES`] unless set. Each is charged to the budget at the
    /// largest tile's size, [`StoreConfig::STROKE_TILE_BYTES`]; zero keeps none, and a stroke
    /// writes every tile dab by dab. [`TileStore::open`](super::TileStore::open) refuses a count
    /// whose share leaves the pools less than their minimum beside the kept predictions.
    pub stroke_tiles: usize,
    /// The finest level the store keeps a display layer under (`geom::DisplayTile`s, stored as
    /// derived records): display tiles of every level below it, brought up to date by
    /// [`TileStore::run_jobs`](super::TileStore::run_jobs) after writes; `None` (the default)
    /// keeps none. A store opened over files without a display layer queues its rebuild.
    pub display: Option<geom::Level>,
    /// The most keys a consumer of the stale-tile feed may have pending before it loses them and
    /// is told to resync (see [`TileStore::drain_stale`](super::TileStore::drain_stale)):
    /// [`crate::DEFAULT_STALE_CAP`] unless set.
    pub stale_cap: usize,
    /// The top level of the display layer's floor: every face's display tiles at this level
    /// and below are loaded on open (and kept as a refresh or rebuild records them), never
    /// evicted, and charged to the budget as a reserved share at their largest
    /// ([`StoreConfig::floor_bytes`]), so a view always has something to draw.
    /// [`StoreConfig::DEFAULT_FLOOR_TOP`] unless set; only with a display layer.
    pub floor_top: geom::Level,
}

impl StoreConfig {
    /// The default chunk: 64 MiB, large enough that the allocator maps each chunk on its own
    /// and gives it back to the OS when it is dropped (see the pool's notes).
    pub const DEFAULT_CHUNK_BYTES: usize = 64 << 20;
    /// The default [`StoreConfig::floor_top`]: level 5, about 16 MB compressed for the
    /// benchmark's 2,400 faces (see the M1 spec's "Residency").
    pub const DEFAULT_FLOOR_TOP: geom::Level = match geom::Level::new(5) {
        Ok(level) => level,
        Err(_) => unreachable!(),
    };
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
            display: None,
            stale_cap: crate::stale::DEFAULT_STALE_CAP,
            floor_top: Self::DEFAULT_FLOOR_TOP,
        }
    }

    /// This configuration keeping the display floor up to `floor_top` (see
    /// [`StoreConfig::floor_top`]).
    #[must_use]
    pub const fn with_floor_top(self, floor_top: geom::Level) -> Self {
        Self { floor_top, ..self }
    }

    /// The bytes of the budget set aside for the display floor over `mesh`: every face's display
    /// tiles at levels up to [`StoreConfig::floor_top`] (and below the display layer's finest),
    /// each at the most its level's encoding can take. Zero without a display layer.
    #[must_use]
    pub fn floor_bytes(&self, mesh: &BaseMesh) -> usize {
        let Some(finest) = self.display else {
            return 0;
        };
        let per_face: usize = (0..finest.get().min(self.floor_top.get() + 1))
            .filter_map(|l| {
                let key =
                    geom::TileKey::display(geom::FaceId::new(0), geom::Level::new(l).ok()?, 0, 0)
                        .ok()?;
                Some(crate::display::max_body(&key))
            })
            .sum();
        per_face.saturating_mul(usize::try_from(mesh.face_count()).unwrap_or(usize::MAX))
    }

    /// This configuration letting a stale-feed consumer fall `stale_cap` keys behind (see
    /// [`StoreConfig::stale_cap`]).
    #[must_use]
    pub const fn with_stale_cap(self, stale_cap: usize) -> Self {
        Self { stale_cap, ..self }
    }

    /// This configuration keeping a display layer under `finest` (see
    /// [`StoreConfig::display`]).
    #[must_use]
    pub const fn with_display(self, finest: geom::Level) -> Self {
        Self {
            display: Some(finest),
            ..self
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
    /// Pins taken with [`TileStore::pin`](super::TileStore::pin) come on top. With the default
    /// chunks the minimum is two or three chunks for a mesh of thousands of faces; it grows
    /// with the mesh's face neighbourhoods, never with the evaluated level.
    ///
    /// Computed from the mesh's face rings, in time linear in its faces:
    /// [`TileStore::open`](super::TileStore::open) runs it once.
    #[must_use]
    pub fn minimum_budget(&self, mesh: &BaseMesh) -> usize {
        Headroom::of(mesh)
            .minimum_budget(self.chunk_bytes)
            .saturating_add(self.kept_prediction_bytes().unwrap_or(usize::MAX))
            .saturating_add(self.stroke_state_bytes().unwrap_or(usize::MAX))
    }

    /// The worker threads this configuration asks for on this machine (see
    /// [`StoreConfig::workers`]).
    pub(super) fn worker_threads(&self) -> usize {
        self.workers.count().get()
    }
}

/// Tiles of one cone level built in one parallel round at most (see the module notes).
///
/// The rounds are cut by this count alone, never by the thread count, so the store's pool
/// operations come in the same order however many workers it has. Each round pins its tiles'
/// correction slots while the workers read them (fewer, cut at the first tile after the first
/// that would need a dirty tile written back or an evaluated chunk taken, see
/// [`TileStore::load_round`](super::TileStore::load_round)), and holds a scratch buffer of one tile
/// per tile of the round that isn't predicted in place.
pub(super) const ROUND: usize = 64;

/// Target tiles [`TileStore::evaluate_many`](super::TileStore::evaluate_many) and
/// [`TileStore::pin`](super::TileStore::pin) put in one batch at most.
///
/// One batch is one dependency cone: this is enough to keep every worker busy at every level of
/// the cone, and few enough that listing the cone stays cheap next to building it.
pub(super) const MAX_BATCH: usize = 4096;

/// The share of the budget's free evaluated slots a batch's targets take: one in this many.
///
/// A batch's cone pins its targets and the level under them (about a quarter as many, plus a
/// border), and, for normals, the targets' neighbours too, so a quarter leaves room for all of
/// it. A batch that runs out of room all the same is halved and tried again.
pub(super) const BATCH_SHARE: usize = 4;

/// What [`TileStore::evaluate_many`](super::TileStore::evaluate_many) makes current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evaluation {
    /// Evaluated positions, as [`TileStore::evaluate`](super::TileStore::evaluate) gives them.
    Positions,
    /// Evaluated positions and unit normals, as [`TileStore::normals`](super::TileStore::normals)
    /// gives them (normals are computed from positions, so these are made current too).
    Normals,
}

/// The most slots of each pool the store pins for its own work: what [`StoreConfig`]'s minimum
/// is made of, and what [`TileStore::pin`](super::TileStore::pin) counts besides the tiles asked
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Headroom {
    /// Evaluated slots one evaluation pins at most: two levels of its cone.
    pub(super) evaluated: usize,
    /// Correction slots one edit pins at most: the tiles holding copies of its vertices.
    corrections: usize,
}

impl Headroom {
    /// The headroom for evaluating and editing `mesh`.
    pub(super) fn of(mesh: &BaseMesh) -> Self {
        let width = geom::cone_width(mesh);
        Self {
            evaluated: 2 * width,
            corrections: width,
        }
    }

    /// The least the evaluated pool keeps when corrections take its chunks (see
    /// [`TileStore`](super::TileStore)): the chunks that hold an evaluation's pins, and one
    /// more for the cache besides.
    pub(super) fn evaluated_floor(self, chunk_bytes: usize) -> usize {
        let (per_chunk, charged) = Pool::<EvaluatedSlot>::layout(chunk_bytes);
        (self.evaluated.div_ceil(per_chunk as usize) + 1) * charged
    }

    /// The least budget, in chunks of `chunk_bytes`, that holds this headroom however the
    /// evaluated pins are spread (see the module notes).
    pub(super) fn minimum_budget(self, chunk_bytes: usize) -> usize {
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
pub(super) fn reserved_shares(
    headroom: Headroom,
    config: &StoreConfig,
    floor: usize,
) -> Result<usize, CacheError> {
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
    let spare = spare - stroke;
    if floor > spare {
        return Err(CacheError::FloorTooLarge { floor, most: spare });
    }
    Ok(kept + stroke + floor)
}
