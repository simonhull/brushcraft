//! Replaying edits away from the store: kept predictions, corrections as stored, whole writes.
//!
//! What a brush needs to replay several dabs on each tile, tiles in parallel, then write each
//! tile back whole.
//!
//! A tile's positions are its prediction (from the level below, which editing the tile doesn't
//! change) with its own corrections applied, vertex by vertex. So a caller holding a tile's
//! prediction and a copy of its corrections can edit the copy and know the positions every
//! edit leaves, without the store, and many tiles can be edited so at once.

use std::collections::BTreeSet;

use geom::{CorrectionError, CorrectionTile, CorrectionView, Prediction, TileKey, Vec3};

use super::TileStore;
use super::error::{CacheError, Invariant};
use super::round::{
    Held, Predict, RoundTask, Tiles, each_parallel, keep_first, list_parents, predict_tile,
};
use crate::pool::SlotId;
use crate::slot::CorrectionSlot;

impl TileStore {
    /// The most tiles whose predictions the store keeps
    /// ([`StoreConfig::kept_predictions`](super::StoreConfig::kept_predictions)): the most
    /// [`TileStore::keep_predictions_of`] can keep together.
    #[must_use]
    pub const fn kept_predictions(&self) -> usize {
        self.kept.capacity()
    }

    /// The prediction the store keeps for `key`, if it keeps one (see "Kept predictions" on
    /// [`TileStore`]): the tile's predicted positions and tangent frames, from its parents as
    /// they are now. Applied to the tile's corrections
    /// ([`Prediction::apply`](geom::Prediction::apply)), it gives the bits
    /// [`TileStore::evaluate`] gives.
    ///
    /// Reading it isn't a use of it. With [`TileStore::par_map`], for reading many on the
    /// workers.
    #[must_use]
    pub fn kept_prediction(&self, key: &TileKey) -> Option<&Prediction> {
        self.kept.get(key)
    }

    /// Makes the store keep the prediction of each of `keys` (see "Kept predictions" on
    /// [`TileStore`]), for [`TileStore::kept_prediction`]: each tile without one is predicted
    /// again from its parents, made current and pinned for it, on the workers. Each of `keys`
    /// counts as used, so the room the new ones need is made from other tiles' first.
    ///
    /// Every tile without a kept prediction must be pinned ([`TileStore::pin`]) and current.
    /// Returns whether every tile of `keys` has its prediction kept now: `false`, with nothing
    /// predicted, if they are more than the store keeps ([`TileStore::kept_predictions`]) or one
    /// without a kept prediction isn't pinned and current.
    ///
    /// The predictions stay kept as any other: until a parent's positions change, the tile
    /// leaves the evaluated pool, or it is the least recently used past the bound.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] before anything is done; as [`TileStore::evaluate`] for making
    /// the parents current. Nothing new is kept then.
    pub fn keep_predictions_of(&mut self, keys: &[TileKey]) -> Result<bool, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        let keys: BTreeSet<TileKey> = keys.iter().copied().collect();
        for key in &keys {
            self.check_key(key)?;
        }
        if keys.len() > self.kept.capacity() {
            return Ok(false);
        }
        let mut missing = Vec::new();
        for key in &keys {
            if self.kept.get(key).is_some() {
                self.kept.used(key);
                continue;
            }
            let held = self.evaluated.is_pinned(key) && self.corrections.is_pinned(key);
            if !held || !self.is_current(key) {
                return Ok(false);
            }
            missing.push(*key);
        }
        if missing.is_empty() {
            return Ok(true);
        }
        let mut parents = BTreeSet::new();
        for key in &missing {
            match self.deps.parents(key) {
                Some(found) => parents.extend(found.iter().copied()),
                None => parents.extend(list_parents(&self.mesh, key)?),
            }
        }
        let held = if parents.is_empty() {
            Held::new()
        } else {
            self.pin_current(parents)?
        };
        let made = self.predict_for_keeping(&missing, &held);
        let mut outcome = Ok(());
        keep_first(&mut outcome, self.unpin_held(&held));
        let made = made?;
        outcome?;
        self.kept.make_room(made.len());
        let kept = missing.into_iter().zip(made);
        for (key, prediction) in kept {
            self.kept.insert(key, prediction);
        }
        Ok(true)
    }

    /// The predictions of `keys` (pinned, their correction slots resident), from their parents
    /// in `held` (pinned and current), made on the workers a round of scratch buffers at a time.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid mesh); [`CacheError::Internal`] for a tile whose
    /// corrections aren't resident, or a parent not in `held`.
    fn predict_for_keeping(
        &mut self,
        keys: &[TileKey],
        held: &Held,
    ) -> Result<Vec<Prediction>, CacheError> {
        let tasks: Vec<RoundTask> = keys
            .iter()
            .map(|key| {
                let id: Option<SlotId<CorrectionSlot>> = self.corrections.find(key);
                id.map(|id| (*key, id)).ok_or(Invariant::NotHeld(*key))
            })
            .collect::<Result<_, _>>()?;
        let mut scratch = self.take_scratch(tasks.len());
        let made = self.predict_tasks(&tasks, held, &mut scratch);
        self.scratch = scratch;
        made
    }

    /// [`TileStore::predict_for_keeping`]'s rounds over `tasks`, into `scratch`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::predict_for_keeping`].
    fn predict_tasks(
        &self,
        tasks: &[RoundTask],
        held: &Held,
        scratch: &mut [Vec<Vec3>],
    ) -> Result<Vec<Prediction>, CacheError> {
        let (mesh, pool) = (&*self.mesh, &self.corrections);
        let tiles = Tiles::Shared {
            pool: &self.evaluated,
            held,
        };
        let mut made = Vec::with_capacity(tasks.len());
        for round in tasks.chunks(scratch.len().max(1)) {
            let mut outs: Vec<(&mut [Vec3], Option<Prediction>)> = scratch
                .iter_mut()
                .take(round.len())
                .map(|buffer| (buffer.as_mut_slice(), None))
                .collect();
            each_parallel(&self.workers, round, &mut outs, |&task, (out, kept)| {
                *kept = predict_tile(mesh, pool, tiles, task, Predict::Keep, out)?;
                Ok(())
            })?;
            let predictions = outs.into_iter().filter_map(|(_, kept)| kept);
            made.extend(predictions);
        }
        if made.len() == tasks.len() {
            Ok(made)
        } else {
            Err(Invariant::ShortBuffer {
                expected: tasks.len(),
                got: made.len(),
            }
            .into())
        }
    }

    /// `key`'s corrections as stored, if its correction tile is resident: no load, and the read
    /// isn't a use of the tile. With [`TileStore::par_map`], for reading many on the workers;
    /// pin them first ([`TileStore::pin`]) so they stay resident.
    #[must_use]
    pub fn cached_correction(&self, key: &TileKey) -> Option<CorrectionView<'_>> {
        let id = self.corrections.find(key)?;
        self.corrections.get(id)?.view().ok()
    }

    /// Writes each of `tiles` whole, as part of the current op (see [`TileStore::begin_op`]):
    /// side, scale, interior and border, the bits given. Each tile's corrections become those
    /// of its [`CorrectionTile`], and the tile is marked written as an edit marks it (dirty, on
    /// the change feed with the op, in the open epoch of the stale-tile feed) and its
    /// evaluations dropped, as [`TileStore::update_tiles`] does; in the order given, no tile
    /// twice.
    ///
    /// For a caller that edited copies of the tiles away from the store
    /// ([`CorrectionView::to_tile`]) with [`geom::CorrectionViewMut`]'s own writes, which so
    /// hold what the store's edits would have written. Unlike [`TileEdit::set`](super::TileEdit),
    /// it doesn't carry a border vertex's value to the copies other tiles hold: **the caller
    /// keeps twins agreeing**, writing every tile holding a copy of a border vertex it changes,
    /// with the same value, in the same call.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] or [`CacheError::OpsExhausted`], and
    /// [`CacheError::Correction`] with [`CorrectionError::BadSide`] for a tile whose side isn't
    /// its key's, before anything is written; as [`TileStore::edit_tile`] for a tile that
    /// can't be loaded, which stops the call there, the tiles before it written.
    pub fn replace_corrections(
        &mut self,
        tiles: &[(TileKey, &CorrectionTile)],
    ) -> Result<(), CacheError> {
        self.drop_uncovered_bumps()?;
        self.settle_display()?;
        self.sync_pins()?;
        self.poll()?;
        let keys: Vec<TileKey> = tiles.iter().map(|&(key, _)| key).collect();
        self.check_keys_and_op(&keys)?;
        if let Some(&(_, tile)) = tiles.iter().find(|(key, tile)| tile.side() != key.side()) {
            return Err(CacheError::Correction(CorrectionError::BadSide(
                tile.side(),
            )));
        }
        let op = self.edit_op()?;
        let mut written = Vec::with_capacity(tiles.len());
        let mut outcome = Ok(());
        for &(key, tile) in tiles {
            let replaced = self.replace_one(key, tile, op);
            if replaced.is_err() {
                outcome = replaced;
                break;
            }
            written.push(key);
        }
        self.invalidate_edited(written);
        outcome
    }

    /// [`TileStore::replace_corrections`] for one tile: loaded and pinned, its slot written,
    /// marked written, unpinned.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]; [`CacheError::Internal`] for a stale handle.
    fn replace_one(
        &mut self,
        key: TileKey,
        tile: &CorrectionTile,
        op: crate::op::OpId,
    ) -> Result<(), CacheError> {
        let id = self.load_correction(key)?;
        self.corrections.pin(id)?;
        let written = self
            .write_whole(id, tile)
            .and_then(|()| self.mark_written(key, id, op));
        let mut outcome = written;
        keep_first(&mut outcome, self.corrections.unpin(id));
        outcome
    }

    /// Copies `tile` into correction slot `id`, whose tile has the same side.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle or a slot too small for the tile (never: a
    /// slot holds the largest tile).
    fn write_whole(
        &mut self,
        id: SlotId<CorrectionSlot>,
        tile: &CorrectionTile,
    ) -> Result<(), CacheError> {
        let slot = self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)?;
        let short = || Invariant::ShortBuffer {
            expected: tile.interior().len() + tile.border().len(),
            got: slot_len(),
        };
        let interior = slot
            .interior
            .get_mut(..tile.interior().len())
            .ok_or_else(short)?;
        interior.copy_from_slice(tile.interior());
        let border = slot
            .border
            .get_mut(..tile.border().len())
            .ok_or_else(short)?;
        border.copy_from_slice(tile.border());
        slot.side = tile.side();
        slot.scale = tile.scale();
        Ok(())
    }
}

/// The values a correction slot holds: the largest tile's interior and border.
const fn slot_len() -> usize {
    let side = geom::TILE_QUADS as usize + 1;
    (side - 2) * (side - 2) + 4 * (side - 1)
}
