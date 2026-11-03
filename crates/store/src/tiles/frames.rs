//! Tangent frames: the axes a brush's deltas are written along.
//!
//! A tile's corrections live in each vertex's tangent frame of its predicted surface (from the
//! level below, see [`geom::Prediction`]): a brush that moves a vertex by a world vector writes
//! that vector in the vertex's frame. The frames depend on the level below alone: editing a
//! tile's own corrections leaves them as they are.

use std::collections::BTreeSet;

use geom::{BaseMesh, Frame, Prediction, Region, TileKey, predict_region};

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use super::round::{Resident, Tiles, list_parents, map_parallel};
use crate::pool::{Pool, SlotId};
use crate::slot::EvaluatedSlot;

/// The evaluated slots a prediction reads, by their tiles, in key order; `None` if one isn't
/// current.
type Parents = Option<Vec<(TileKey, SlotId<EvaluatedSlot>)>>;

impl TileStore {
    /// Each vertex's tangent frame, row-major `side²`, for each of `keys`, in order: the axes a
    /// brush writes a vertex's delta along.
    ///
    /// From the prediction the store keeps for the tile, if it keeps one, else predicted afresh
    /// from the level below on the store's workers (the level below made current first, together).
    /// The same bits as the frames the tile's evaluation uses: a vertex shared by several tiles
    /// has the same frame in each.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] for a key that isn't a tile of the store, before anything is
    /// done; as [`TileStore::evaluate_many`] for the level below; [`CacheError::Eval`] if a
    /// prediction fails (never, for tiles of the store's mesh).
    pub fn frames_many(&mut self, keys: &[TileKey]) -> Result<Vec<Vec<Frame>>, CacheError> {
        for key in keys {
            self.check_key(key)?;
        }
        let mut below: BTreeSet<TileKey> = BTreeSet::new();
        for key in keys.iter().filter(|key| self.kept.get(key).is_none()) {
            below.extend(list_parents(&self.mesh, key)?);
        }
        let below: Vec<TileKey> = below.into_iter().collect();
        if !below.is_empty() {
            self.evaluate_many(&below, Evaluation::Positions)?;
        }
        let mut parents: Vec<(TileKey, Parents)> = Vec::with_capacity(keys.len());
        for key in keys {
            if self.kept.get(key).is_some() {
                parents.push((*key, Some(Vec::new())));
                continue;
            }
            let held: Parents = list_parents(&self.mesh, key)?
                .into_iter()
                .map(|parent| Some((parent, self.fresh_slot(&parent)?)))
                .collect();
            parents.push((
                *key,
                held.map(|mut held| {
                    held.sort_unstable_by_key(|&(parent, _)| parent);
                    held
                }),
            ));
        }
        let (mesh, pool, kept) = (&*self.mesh, &self.evaluated, &self.kept);
        let predicted = map_parallel(&self.workers, &parents, |(key, held)| {
            let side = key.side();
            if let Some(prediction) = kept.get(key) {
                return Ok(Some(frames_of(side, |i, j| prediction.frame(i, j))));
            }
            let Some(held) = held else {
                return Ok(None);
            };
            let source = Resident::new(mesh, Tiles::Shared { pool, held });
            let prediction = predict_region(mesh, &source, Region::of_tile(key))?;
            Ok(Some(frames_of(side, |i, j| prediction.frame(i, j))))
        });
        let mut out = Vec::with_capacity(keys.len());
        for (key, each) in keys.iter().zip(predicted) {
            let frames: Option<Vec<Frame>> = each.map_err(CacheError::Eval)?;
            match frames {
                Some(frames) => out.push(frames),
                // A parent left the cache between being made current and read: on its own.
                None => out.push(self.frames_of_one(*key)?),
            }
        }
        Ok(out)
    }

    /// A predictor of the tiles whose parents are current now, for a brush's workers to predict
    /// tiles with while they brush others ([`Predictor`]): it reads the evaluated pool shared.
    ///
    /// Hold it only while nothing is evaluated or edited: it borrows the store.
    #[must_use]
    pub fn predictor(&self) -> Predictor<'_> {
        Predictor {
            mesh: &self.mesh,
            pool: &self.evaluated,
        }
    }

    /// The evaluated slots tile `key`'s prediction reads, its parents, as they are now
    /// ([`Predictor::predict`]); `None` if one isn't current, or `key` isn't a tile of the store
    /// above level 0.
    #[must_use]
    pub fn parent_slots(&self, key: &TileKey) -> Option<ParentSlots> {
        self.check_key(key).ok()?;
        let held: Option<Vec<(TileKey, SlotId<EvaluatedSlot>)>> = list_parents(&self.mesh, key)
            .ok()?
            .into_iter()
            .map(|parent| Some((parent, self.fresh_slot(&parent)?)))
            .collect();
        let mut held = held?;
        held.sort_unstable_by_key(|&(parent, _)| parent);
        Some(ParentSlots(held))
    }

    /// Keeps `prediction`, made by [`Predictor::predict`] from tile `key`'s parents as they are
    /// still (nothing evaluated or edited since), as [`TileStore::keep_predictions_of`] keeps
    /// one: for [`TileStore::kept_prediction`], and for the tile's evaluation to reuse. Returns
    /// whether it is kept now: not for a tile that isn't pinned and current, nor past the bound
    /// (a tile of the bound's least recently used gives up its room first).
    ///
    /// A tile that has a kept prediction already keeps that one (the same bits).
    pub fn keep_prediction(&mut self, key: TileKey, prediction: Prediction) -> bool {
        if self.kept.get(&key).is_some() {
            self.kept.used(&key);
            return true;
        }
        let held = self.evaluated.is_pinned(&key) && self.corrections.is_pinned(&key);
        if !held || !self.is_current(&key) || self.kept.capacity() == 0 {
            return false;
        }
        self.kept.make_room(1);
        self.kept.insert(key, prediction);
        true
    }

    /// [`TileStore::frames_many`] for one tile, its level below made current and pinned while it
    /// is predicted.
    ///
    /// # Errors
    ///
    /// As [`TileStore::frames_many`].
    fn frames_of_one(&mut self, key: TileKey) -> Result<Vec<Frame>, CacheError> {
        let prediction = self.prediction_of_one(key)?;
        Ok(frames_of(key.side(), |i, j| prediction.frame(i, j)))
    }

    /// Tile `key`'s prediction, made afresh from its parents, made current and pinned while it
    /// is predicted, and not kept: the bits a kept prediction of it would have.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`] for making the parents current; [`CacheError::Eval`] (not for a
    /// valid key).
    pub(super) fn prediction_of_one(&mut self, key: TileKey) -> Result<Prediction, CacheError> {
        let parents: BTreeSet<TileKey> = list_parents(&self.mesh, &key)?.into_iter().collect();
        let held = self.pin_current(parents)?;
        let source = Resident::new(
            &self.mesh,
            Tiles::Shared {
                pool: &self.evaluated,
                held: &held,
            },
        );
        let predicted = predict_region(&self.mesh, &source, Region::of_tile(&key));
        let unpinned = self.unpin_held(&held);
        let prediction = predicted?;
        unpinned?;
        Ok(prediction)
    }
}

/// A tile's parents' evaluated slots as current when [`TileStore::parent_slots`] found them, for
/// [`Predictor::predict`].
#[derive(Debug, Clone)]
pub struct ParentSlots(Vec<(TileKey, SlotId<EvaluatedSlot>)>);

/// Predicts tiles from their current parents, reading the store's evaluated pool shared.
///
/// From [`TileStore::predictor`]: a brush's workers predict the tiles it enters with it, each on
/// its own, while they brush the others.
#[derive(Clone, Copy)]
pub struct Predictor<'a> {
    /// The base mesh.
    mesh: &'a BaseMesh,
    /// The evaluated pool the parents are read from.
    pool: &'a Pool<EvaluatedSlot>,
}

impl std::fmt::Debug for Predictor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predictor").finish_non_exhaustive()
    }
}

impl Predictor<'_> {
    /// Tile `key`'s prediction from its parents `parents` ([`TileStore::parent_slots`]): the bits
    /// the store's evaluation predicts, frames and all. `None` if a parent's slot no longer holds
    /// its positions (never, while the predictor is held), or the prediction fails (never, for a
    /// tile of the store's mesh).
    #[must_use]
    pub fn predict(&self, key: &TileKey, parents: &ParentSlots) -> Option<Prediction> {
        let source = Resident::new(
            self.mesh,
            Tiles::Shared {
                pool: self.pool,
                held: &parents.0,
            },
        );
        predict_region(self.mesh, &source, Region::of_tile(key)).ok()
    }
}

/// The frames of a tile of `side` vertices a side, row-major, `frame(i, j)` giving each vertex's.
///
/// A vertex without one (never, for a tile and prediction of the same side) takes a frame whose
/// every axis is NaN, so a brush writing along it writes no finite value and is refused.
fn frames_of(side: u32, frame: impl Fn(u32, u32) -> Option<Frame>) -> Vec<Frame> {
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
    grid.map(|(i, j)| frame(i, j).unwrap_or(Frame::NAN))
        .collect()
}
