//! The pieces of an evaluation round: what a round targets and fills, the resident tiles its
//! workers read, the parallel maps that run it, and the predictions it plans.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::convert::Infallible;

use geom::{
    BaseMesh, Channel, FaceId, LayerId, Level, PositionSource, Prediction, Region, TileKey, Vec3,
    parent_tiles, predict_region,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::error::{CacheError, Invariant};
use crate::file::{BaseFile, Loc, ProjectFile, ReadScratch};
use crate::op::OpId;
use crate::pool::{Pool, SlotId};
use crate::prefetch::Source;
use crate::slot::{CorrectionSlot, EvaluatedSlot};

/// What happens to a batch's targets once [`TileStore::in_batches`](super::TileStore::in_batches)
/// has run it: what the batches after it must leave room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Targets {
    /// They stay pinned ([`TileStore::pin`](super::TileStore::pin)): every target still to come is
    /// room taken.
    Pinned,
    /// They are unpinned, and stay only as a cache
    /// ([`TileStore::evaluate_many`](super::TileStore::evaluate_many)).
    Released,
}

/// How [`TileStore::load_correction_as`](super::TileStore::load_correction_as) may get a slot for a
/// correction tile that isn't resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Acquire {
    /// As the memory policy orders it (see
    /// [`TileStore::acquire_correction`](super::TileStore::acquire_correction)): it may take an
    /// evaluated chunk back or write a dirty tile back.
    Any,
    /// A free slot, a new chunk or a clean tile's slot only, as prefetch takes them: nothing is
    /// written back and the evaluated pool keeps its chunks. A round's tiles after its first
    /// load this way, so a round writes back no more than one tile at a time would.
    Clean,
}

/// A correction tile being loaded that wasn't resident: its slot, taken (see
/// [`TileStore::take_correction_slot`](super::TileStore::take_correction_slot)) and not yet filled,
/// and where its record is.
#[derive(Debug, Clone, Copy)]
pub(super) struct Fill {
    /// The tile.
    pub(super) key: TileKey,
    /// Its slot, which [`fill_slot`] fills.
    pub(super) id: SlotId<CorrectionSlot>,
    /// Where its latest record is: the working file, else the base; `None` for a tile never
    /// written, which loads as zero.
    pub(super) found: Option<(Source, Loc)>,
}

/// What filling each slot of a round gave, in the round's order: the op of the record read, or
/// `None` for a tile loaded as zero (see [`fill_slot`]).
pub(super) type FillOutcomes = Vec<Result<Option<OpId>, CacheError>>;

/// Fills `slot` with `tile`'s tile: read from its record in `working` or `base`, with `scratch`'s
/// buffers, or reset to zero where it has none. Returns the op of the record read.
///
/// Touches nothing but `slot` and `scratch`, so a round's tiles are filled on several threads
/// at once (see [`TileStore::fill_round`](super::TileStore::fill_round)).
///
/// # Errors
///
/// [`CacheError::Store`] if the record can't be read or doesn't decode;
/// [`CacheError::Internal`] for a record in a base the store doesn't have, or a slot that can't
/// hold the tile's side. On error the slot holds no tile.
pub(super) fn fill_slot(
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
pub(super) type Held = Vec<(TileKey, SlotId<EvaluatedSlot>)>;

/// One level of a dependency cone: each tile, with the parents it is predicted from (left empty
/// for the coarsest level when its tiles are all cached, since nothing is built there).
pub(super) type ConeLevel = BTreeMap<TileKey, Vec<TileKey>>;

/// The positions of tiles an evaluation pinned, each tile with all `SIDE²` of its slot's
/// positions, in key order: what a [`Resident`] source reads.
pub(super) type Positions<'a> = Vec<(TileKey, &'a [Vec3])>;

/// Where a [`Resident`] source finds the pinned tiles' positions.
#[derive(Clone, Copy)]
pub(super) enum Tiles<'a> {
    /// Borrowed slices ([`Positions`]), found once per round by the owner thread, split off
    /// the same borrow that hands the workers the slots they write
    /// ([`TileStore::predict_round`](super::TileStore::predict_round),
    /// [`TileStore::normals_round`](super::TileStore::normals_round)).
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
pub(super) struct Resident<'a> {
    /// Level 0.
    mesh: &'a BaseMesh,
    /// The pinned tiles.
    tiles: Tiles<'a>,
    /// The tile the last read above level 0 resolved to, and its positions.
    last: Cell<Option<(TileKey, &'a [Vec3])>>,
}

impl<'a> Resident<'a> {
    /// A source over the pinned `tiles` and `mesh` at level 0, remembering no tile yet.
    pub(super) const fn new(mesh: &'a BaseMesh, tiles: Tiles<'a>) -> Self {
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
pub(super) fn map_parallel<T: Sync, R: Send>(
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
pub(super) fn each_parallel<T: Sync, B: Send>(
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
pub(super) type RoundTask = (TileKey, SlotId<CorrectionSlot>);

/// The tasks of a round of predictions: each tile of `round` with its correction slot (from
/// `corrections`, in the same order).
pub(super) fn round_tasks(
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
/// [`TileStore`](super::TileStore)).
#[derive(Debug, Clone, Copy)]
pub(super) enum Predict<'a> {
    /// Applies the corrections to this prediction, the one the store kept for the tile, without
    /// predicting.
    Reuse(&'a Prediction),
    /// Predicts the tile and returns the prediction, for the store to keep.
    Keep,
    /// Predicts the tile and drops the prediction once it is applied, on the worker.
    Drop,
}

/// What [`TileStore::plan_predictions`](super::TileStore::plan_predictions) decides for one tile of
/// a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Plan {
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
pub(super) fn predict_tile(
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
pub(super) fn recorded_or_listed<T: Send>(
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
pub(super) fn held_index(held: &Held, key: &TileKey) -> Result<usize, Invariant> {
    held.binary_search_by_key(key, |&(k, _)| k)
        .or(Err(Invariant::NotHeld(*key)))
}

/// The tiles `key`'s positions are predicted from, listed with [`parent_tiles`], level 0 (the
/// mesh) left out, in key order.
///
/// # Errors
///
/// [`CacheError::Eval`] (not for a valid key).
pub(super) fn list_parents(mesh: &BaseMesh, key: &TileKey) -> Result<Vec<TileKey>, CacheError> {
    let parents = parent_tiles(mesh, Region::of_tile(key))?;
    Ok(parents
        .into_iter()
        .filter(|parent| parent.level() != Level::ZERO)
        .collect())
}

/// Vertices in `key`'s tile: `side²`.
pub(super) fn vertex_count(key: &TileKey) -> usize {
    let side = key.side() as usize;
    side * side
}

/// The first `n` values of `values`.
///
/// # Errors
///
/// [`Invariant::ShortBuffer`] if there are fewer (no slot or scratch buffer is smaller than the
/// largest tile, so this doesn't happen).
pub(super) fn prefix(values: &[Vec3], n: usize) -> Result<&[Vec3], Invariant> {
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
pub(super) fn prefix_mut(values: &mut [Vec3], n: usize) -> Result<&mut [Vec3], Invariant> {
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
pub(super) const fn nothing_to_write<T>(_: TileKey, _: &T) -> Result<(), Infallible> {
    Ok(())
}

/// Records `next` in `outcome` unless `outcome` already holds an error: for cleanup loops that
/// must finish every step and report the first failure.
pub(super) fn keep_first<E: Into<CacheError>>(
    outcome: &mut Result<(), CacheError>,
    next: Result<(), E>,
) {
    if outcome.is_ok() {
        *outcome = next.map_err(Into::into);
    }
}
