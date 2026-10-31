//! Requested finest tiles, evaluated in the store's jobs, or while a frame is staged when it has
//! the time.
//!
//! A finest tile has no stored record: its positions are evaluated from the corrections of its
//! cone, which may have to be read from the file first (about 0.8 ms a tile cold). Evaluated
//! while staging, that blocked the owner thread before the frame was handed off. Instead the
//! requested finest tiles go to the store's queued evaluation
//! ([`TileStore::queue_evaluation_as`], for their positions and ring alone: the GPU computes
//! their normals), which runs in the frame loop's jobs phase after the
//! hand-off ([`TileStore::run_jobs`], taking turns with the display jobs). What it finished is
//! taken ([`TileStore::take_evaluated`]) and held here, each tile a copy of its positions with
//! the ring ([`EvaluatedTile`]), until residency stages it: [`StoreSource`] reads a requested
//! finest tile from a copy held here first. Without one, it evaluates the tile while staging if
//! the frame has the time for it, at the cost measured lately
//! ([`StoreSource::evaluating_until`], [`FinestEvaluations::cost`]), and withdraws it from the
//! queue; else it defers it until the queue has made it ([`TileSource::deferred`]).
//!
//! # Never stale
//!
//! The store hands over a copy only while no write since it was made reaches it or its ring.
//! Once taken, a copy is outdated by the stale-tile feed: every epoch names each finest tile
//! whose positions or ring moved, so [`FinestEvaluations::outdate`], given each drain before
//! residency applies it, drops the copies the epochs name (all of them on a resync), and the
//! tile, still wanted, is queued again.
//!
//! [`TileSource::deferred`]: super::TileSource::deferred

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use geom::{Channel, TileKey};
use store::{CacheError, Drained, EvaluatedTile, Evaluation, TileStore};

#[cfg(doc)]
use super::StoreSource;
use crate::residency::Residency;

/// Requested finest tiles evaluated off the staging path: queued in the store's evaluation
/// job, and the copies it finished held until they are staged (see the module notes).
///
/// Kept across frames on the owner thread, beside the [`Residency`] whose requests it serves.
#[derive(Debug, Default)]
pub struct FinestEvaluations {
    /// Copies taken from the store and not outdated since, until staged.
    held: BTreeMap<TileKey, EvaluatedTile>,
    /// What a requested finest tile evaluated while staging has cost lately, if one has.
    cost: Option<Duration>,
}

impl FinestEvaluations {
    /// Nothing queued, nothing held.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops the copies `drained` outdates: those an epoch names (their positions, or the ring
    /// uploaded with them, moved), or every one on a resync. Call it with each drain of the
    /// stale-tile feed, before [`Residency::apply`] takes it.
    pub fn outdate(&mut self, drained: &Drained) {
        match drained {
            Drained::Epochs(epochs) => {
                let named = |key: &TileKey| epochs.iter().any(|epoch| epoch.keys.contains(key));
                self.held.retain(|key, _| !named(key));
            }
            Drained::Resync => self.held.clear(),
        }
    }

    /// Brings the store's queued evaluation in line with what `residency` wants, before it
    /// stages: takes the copies the job finished, keeps those of finest tiles still wanted,
    /// and queues every other wanted finest tile at its screen error, withdrawing the rest
    /// from the queue. Returns how many queued keys the store dropped to keep its queue within
    /// its budget share ([`TileStore::queue_evaluation_as`]).
    ///
    /// Call it every frame, after [`Residency::want`]: a tile whose copy was outdated is
    /// queued again here.
    ///
    /// # Errors
    ///
    /// As [`TileStore::queue_evaluation_as`] (never, for the tiles of the store's mesh that
    /// residency wants).
    pub fn request(
        &mut self,
        store: &mut TileStore,
        residency: &Residency,
    ) -> Result<usize, CacheError> {
        for tile in store.take_evaluated() {
            self.held.insert(tile.key(), tile);
        }
        let wanted = residency
            .wanted()
            .filter(|(key, _)| key.channel() != Channel::DISPLAY);
        let wanted: BTreeMap<TileKey, f32> = wanted.collect();
        self.held.retain(|key, _| wanted.contains_key(key));
        let queue = wanted
            .into_iter()
            .filter(|(key, _)| !self.held.contains_key(key));
        let queue: Vec<(TileKey, f32)> = queue.collect();
        let queued: BTreeSet<TileKey> = queue.iter().map(|(key, _)| *key).collect();
        store.retain_evaluation(|key| queued.contains(key));
        store.queue_evaluation_as(&queue, Evaluation::Positions)
    }

    /// The copy held of `key`, if one is: evaluated, and current as of the last drain given
    /// to [`FinestEvaluations::outdate`].
    #[must_use]
    pub fn get(&self, key: &TileKey) -> Option<&EvaluatedTile> {
        self.held.get(key)
    }

    /// What a requested finest tile evaluated while staging has cost lately, if one has been
    /// ([`StoreSource::evaluating_until`](super::StoreSource::evaluating_until)): what the time
    /// left is counted in. It follows a dearer tile at once and a cheaper one by a quarter, so
    /// it stays on the dear side.
    #[must_use]
    pub const fn cost(&self) -> Option<Duration> {
        self.cost
    }

    /// Notes that a requested finest tile evaluated while staging cost `each`.
    pub(crate) fn note_cost(&mut self, each: Duration) {
        self.cost = Some(match self.cost {
            Some(cost) if each < cost => (cost * 3 + each) / 4,
            _ => each,
        });
    }

    /// How many copies are held, waiting to be staged.
    #[must_use]
    pub fn held(&self) -> usize {
        self.held.len()
    }
}
