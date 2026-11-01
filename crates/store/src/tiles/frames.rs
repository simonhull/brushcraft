//! Tangent-frame normals: the axis a brush's draw moves each vertex along.
//!
//! A tile's corrections live in each vertex's tangent frame of its predicted surface (from the
//! level below, see [`geom::Prediction`]), and the draw action pushes a vertex along that frame's
//! normal. A brush that shows its dabs ahead of the finest work moves the store's positions along
//! the same normals, so what it shows differs from what lands only by the stored encoding. The
//! frames depend on the level below alone: editing a tile's own corrections leaves them as they
//! are.

use std::collections::BTreeSet;

use geom::{Region, TileKey, Vec3, predict_region};

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
    /// The unit normal of each vertex's tangent frame, row-major `side²`, for each of `keys`, in
    /// order: the axis the draw action moves the vertex along as finest work.
    ///
    /// From the prediction the store keeps for the tile, if it keeps one, else predicted afresh
    /// from the level below on the store's workers (the level below made current first, together).
    /// The same bits as the frames the tile's evaluation and the brush's edits use: a vertex
    /// shared by several tiles has the same frame normal in each.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] for a key that isn't a tile of the store, before anything is
    /// done; as [`TileStore::evaluate_many`] for the level below; [`CacheError::Eval`] if a
    /// prediction fails (never, for tiles of the store's mesh).
    pub fn frame_normals_many(&mut self, keys: &[TileKey]) -> Result<Vec<Vec<Vec3>>, CacheError> {
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
                return Ok(Some(frame_normals(side, |i, j| prediction.frame(i, j))));
            }
            let Some(held) = held else {
                return Ok(None);
            };
            let source = Resident::new(mesh, Tiles::Shared { pool, held });
            let prediction = predict_region(mesh, &source, Region::of_tile(key))?;
            Ok(Some(frame_normals(side, |i, j| prediction.frame(i, j))))
        });
        let mut out = Vec::with_capacity(keys.len());
        for (key, each) in keys.iter().zip(predicted) {
            let normals: Option<Vec<Vec3>> = each.map_err(CacheError::Eval)?;
            match normals {
                Some(normals) => out.push(normals),
                // A parent left the cache between being made current and read: on its own.
                None => out.push(self.frame_normals_of(*key)?),
            }
        }
        Ok(out)
    }

    /// [`TileStore::frame_normals_many`] for one tile, its level below made current and pinned
    /// while it is predicted.
    ///
    /// # Errors
    ///
    /// As [`TileStore::frame_normals_many`].
    fn frame_normals_of(&mut self, key: TileKey) -> Result<Vec<Vec3>, CacheError> {
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
        Ok(frame_normals(key.side(), |i, j| prediction.frame(i, j)))
    }
}

/// The frame normals of a tile of `side` vertices a side, row-major, `frame(i, j)` giving each
/// vertex's frame (a vertex without one, never, takes no direction).
fn frame_normals(side: u32, frame: impl Fn(u32, u32) -> Option<geom::Frame>) -> Vec<Vec3> {
    let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
    grid.map(|(i, j)| frame(i, j).map_or(Vec3::ZERO, |frame| frame.n()))
        .collect()
}
