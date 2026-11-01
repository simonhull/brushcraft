//! Reading a tile with its one-vertex ring.
//!
//! The tile's evaluated positions, the ring of neighbouring vertices around it and the rings of
//! the base vertices at its corners, gathered from the cached evaluated tiles as `geom`'s normals
//! gather them: what a GPU needs to compute a finest tile's normals itself, with the bits the
//! store's own normals have (see [`GatheredBlock`] for the layout and the arithmetic).

use std::collections::BTreeSet;

use geom::{GatheredBlock, Region, TileKey, gather_block};

use super::TileStore;
use super::error::CacheError;
use super::round::{Resident, Tiles, map_parallel};
use crate::pool::SlotId;
use crate::slot::EvaluatedSlot;

/// The evaluated slots a gather reads, by their tiles, in key order.
type Held = Vec<(TileKey, SlotId<EvaluatedSlot>)>;

impl TileStore {
    /// `key`'s evaluated positions with its one-vertex ring and its base-vertex rings, as
    /// [`gather_block`] gathers them over the store's evaluated tiles: what `key`'s normals read.
    ///
    /// For a tile of 65² vertices that is 67² positions, row-major from `(-1, -1)`, and the ring
    /// of every base vertex at a tile corner, of any valence, owner first (the layout is
    /// [`GatheredBlock`]'s). Every vertex is read from its owner copy, so a vertex shared by two
    /// tiles, across a tile seam or a face seam or round a base vertex, has the same bits in
    /// both; and [`GatheredBlock::normals`] of the result is [`TileStore::normals`] of the tile,
    /// bit for bit.
    ///
    /// The tiles read ([`TileStore::normal_inputs`]: the tile, its apron and the rings at its
    /// corners) are read from the cache where they are all current; else they are evaluated
    /// coarse to fine and pinned while they are read, as computing the tile's normals does, so
    /// any budget the store opens with holds them.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`].
    pub fn positions_with_ring(&mut self, key: TileKey) -> Result<GatheredBlock, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        if let Some(block) = self.cached_positions_with_ring(&key)? {
            return Ok(block);
        }
        let inputs: BTreeSet<TileKey> = self.normal_inputs(&key)?.iter().copied().collect();
        // Made current and pinned as one cone level, as computing the tile's normals pins
        // them: within the room the budget keeps for an evaluation's own pins.
        let held = self.pin_current(inputs)?;
        let block = self.gather_held(&key, &held);
        self.unpin_held(&held)?;
        block
    }

    /// `key`'s positions with its ring, gathered from the evaluated tiles `held` (in key order,
    /// every tile `key`'s normals read).
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if `held` lacks a tile the gather reads.
    fn gather_held(
        &self,
        key: &TileKey,
        held: &[(TileKey, SlotId<EvaluatedSlot>)],
    ) -> Result<GatheredBlock, CacheError> {
        let tiles = Tiles::Shared {
            pool: &self.evaluated,
            held,
        };
        let source = Resident::new(&self.mesh, tiles);
        Ok(gather_block(&self.mesh, &source, Region::of_tile(key))?)
    }

    /// [`TileStore::positions_with_ring`] from the cache alone, never evaluating anything: `None`
    /// unless every tile it reads is cached and current. The read doesn't count as a use of the
    /// tiles, as [`TileStore::cached_positions`]'s doesn't.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Eval`] (not for a valid key).
    pub fn cached_positions_with_ring(
        &self,
        key: &TileKey,
    ) -> Result<Option<GatheredBlock>, CacheError> {
        let inputs = self.normal_inputs(key)?;
        let held: Option<Vec<(TileKey, SlotId<EvaluatedSlot>)>> = inputs
            .iter()
            .map(|input| Some((*input, self.fresh_slot(input)?)))
            .collect();
        let Some(mut held) = held else {
            return Ok(None);
        };
        // In key order, as the source finds them.
        held.sort_unstable_by_key(|&(input, _)| input);
        self.gather_held(key, &held).map(Some)
    }

    /// [`TileStore::cached_positions_with_ring`] for each of `keys`, in order, gathered on the
    /// store's workers: the same blocks, bit for bit.
    ///
    /// # Errors
    ///
    /// As [`TileStore::cached_positions_with_ring`], the first in `keys`' order.
    pub fn cached_positions_with_ring_many(
        &self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<GatheredBlock>>, CacheError> {
        let mut found: Vec<(TileKey, Option<Held>)> = Vec::with_capacity(keys.len());
        for key in keys {
            let inputs = self.normal_inputs(key)?;
            let held: Option<Held> = inputs
                .iter()
                .map(|input| Some((*input, self.fresh_slot(input)?)))
                .collect();
            found.push((
                *key,
                held.map(|mut held| {
                    held.sort_unstable_by_key(|&(input, _)| input);
                    held
                }),
            ));
        }
        let (mesh, pool) = (&*self.mesh, &self.evaluated);
        let gathered = map_parallel(&self.workers, &found, |(key, held)| {
            let Some(held) = held else {
                return Ok(None);
            };
            let tiles = Tiles::Shared { pool, held };
            let source = Resident::new(mesh, tiles);
            gather_block(mesh, &source, Region::of_tile(key))
                .map(Some)
                .map_err(CacheError::from)
        });
        gathered.into_iter().collect()
    }
}
