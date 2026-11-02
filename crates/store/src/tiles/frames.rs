//! Tangent frames: the axes a brush's deltas are written along.
//!
//! A tile's corrections live in each vertex's tangent frame of its predicted surface (from the
//! level below, see [`geom::Prediction`]): a brush that moves a vertex by a world vector writes
//! that vector in the vertex's frame. The frames depend on the level below alone: editing a
//! tile's own corrections leaves them as they are.

use std::collections::BTreeSet;

use geom::{Frame, Region, TileKey, predict_region};

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use super::round::{Resident, Tiles, list_parents, map_parallel};
use crate::pool::SlotId;
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

    /// [`TileStore::frames_many`] for one tile, its level below made current and pinned while it
    /// is predicted.
    ///
    /// # Errors
    ///
    /// As [`TileStore::frames_many`].
    fn frames_of_one(&mut self, key: TileKey) -> Result<Vec<Frame>, CacheError> {
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
        Ok(frames_of(key.side(), |i, j| prediction.frame(i, j)))
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
