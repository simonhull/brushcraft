//! The queued evaluation's side of the store: queueing finest tiles by priority, withdrawing
//! them, a slice of the job, and handing over what it finished (see [`crate::queued`]).

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use geom::TileKey;

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use super::upkeep::ConeStep;
use crate::jobs::JobsReport;
use crate::pool::Freshness;
use crate::queued::{EvaluatedTile, EvaluationQueue, Priority};

/// About how long a slice spends bringing finished tiles up to date after a correction
/// changed (a copy each, a few dozen microseconds): it leaves the rest for the next slices.
const REFRESH_TIME: Duration = Duration::from_micros(300);

/// What a slice of the queued evaluation did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Turn {
    /// It evaluated, finished or brought up to date some tiles: run it again.
    Worked,
    /// It waits for the corrections it prefetched: nothing to do on the owner thread for a
    /// moment.
    Reading,
    /// It found nothing to do.
    Idle,
}

impl TileStore {
    /// Queues finest tiles to evaluate, each with a priority (a renderer's screen error, say):
    /// [`TileStore::run_jobs`] evaluates them in slices, highest priority first, and
    /// [`TileStore::take_evaluated`] hands over each finished tile. Returns how many queued keys
    /// were dropped to keep the queue within its share of the budget: the lowest priorities,
    /// which may be some of these.
    ///
    /// A tile queued already takes the new priority, and is evaluated once. A NaN priority is
    /// the lowest there is. Any tile of the store may be queued; the job is meant for the finest
    /// level, whose tiles a caller can't read from the display layer.
    ///
    /// A slice pages in what it evaluates itself (the corrections of the tile's cone are
    /// prefetched and installed a level at a time, never waited on), so a caller doesn't
    /// prefetch them first.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`] for a key that isn't a tile of the store, before anything is
    /// queued.
    pub fn queue_evaluation(&mut self, keys: &[(TileKey, f32)]) -> Result<usize, CacheError> {
        for (key, _) in keys {
            self.check_key(key)?;
        }
        for &(key, priority) in keys {
            self.evaluation.want(key, Priority::new(priority));
        }
        Ok(self.evaluation.trim())
    }

    /// Withdraws the queued tiles no longer wanted: keeps only those `keep` holds of, and only
    /// the finished tiles waiting to be taken it holds of. A tile being evaluated when it is
    /// withdrawn is dropped where it is (what was evaluated of it stays cached).
    pub fn retain_evaluation(&mut self, keep: impl FnMut(&TileKey) -> bool) {
        self.evaluation.retain(keep);
    }

    /// The finest tiles the queued evaluation finished since the last call, in the order they
    /// finished (highest priority first, within a round), each a copy of its positions with
    /// the ring and its normals as the store holds them now.
    ///
    /// A tile is kept back once a write since it finished reaches it: a write whose
    /// evaluations reach the tile or a tile of its ring, or any write while one of those has
    /// left the cache (nothing then tells whether the write reaches it), or a lost tile's write
    /// undone. The next slice of the job copies it again from the cache if it is current there,
    /// else queues it again, so the tile handed over is never older than the store's. Writes
    /// elsewhere leave it alone: during a brush stroke, the tiles away from the brush are
    /// handed over as they finish.
    ///
    /// A tile handed over can still change later without a write of its own, through its ring
    /// (see [`EvaluatedTile`]): the stale-tile feed tells.
    #[must_use = "the tiles are handed over once: take them to draw them"]
    pub fn take_evaluated(&mut self) -> Vec<EvaluatedTile> {
        let (current, outdated): (Vec<EvaluatedTile>, Vec<EvaluatedTile>) =
            std::mem::take(&mut self.evaluation.finished)
                .into_iter()
                .partition(|tile| tile.current);
        self.evaluation.finished = outdated;
        current
    }

    /// Outdates the finished tiles a write reaches, as the write's evaluations are dropped:
    /// those reading a tile of `moved` (the positions the write's cascade drops), and those
    /// reading a tile the cache doesn't hold current. The cascade follows only what the cache
    /// records, so a tile read from that has left the cache since may follow from the write
    /// without the cascade saying so.
    ///
    /// Called before the cascade's tiles are dropped, so every tile still current is one it
    /// would reach if the write reached it.
    pub(super) fn outdate_finished(&mut self, moved: &BTreeSet<TileKey>) {
        let evaluated = &self.evaluated;
        let reached = |inputs: &[TileKey]| {
            inputs.iter().any(|input| {
                moved.contains(input) || evaluated.freshness(input) != Some(Freshness::Fresh)
            })
        };
        self.evaluation.outdate(reached);
        #[cfg(test)]
        for tile in &mut self.hooks.held_evaluated {
            if reached(&tile.inputs) {
                tile.current = false;
            }
        }
    }

    /// Outdates every finished tile: for a lost tile's write undone, where the slot's values
    /// change outside an edit's cascade.
    pub(super) fn outdate_all_finished(&mut self) {
        self.evaluation.outdate(|_| true);
        #[cfg(test)]
        for tile in &mut self.hooks.held_evaluated {
            tile.current = false;
        }
    }

    /// How many tiles are queued for evaluation and not finished.
    #[must_use]
    pub fn queued_evaluations(&self) -> usize {
        self.evaluation.len()
    }

    /// Whether `key`'s positions and unit normals are cached and current: what
    /// [`TileStore::cached_positions`] and [`TileStore::cached_normals`] then read. Never
    /// evaluates, and doesn't count as a use of the tile.
    ///
    /// It says nothing of the tile's ring: [`TileStore::cached_positions_with_ring`] reads
    /// its neighbours too, and may find one of them evicted while this holds.
    #[must_use]
    pub fn is_cached(&self, key: &TileKey) -> bool {
        self.cached_normals(key).is_some()
    }

    /// One slice of the queued evaluation: brings finished tiles up to date after a correction
    /// changed, else steps through the cone of the highest-priority tiles wanted (see
    /// [`TileStore::step_cone`]) and, once it is current, computes their normals and copies
    /// them for the caller.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::prefetch`]; the tiles stay queued.
    pub(super) fn evaluation_slice(&mut self, report: &mut JobsReport) -> Result<Turn, CacheError> {
        let caps = self.evaluation.caps;
        let mut queue = std::mem::replace(&mut self.evaluation, EvaluationQueue::new(caps));
        let turn = self.evaluation_step(&mut queue, report);
        self.evaluation = queue;
        turn
    }

    /// [`TileStore::evaluation_slice`] on `queue`, taken out of the store for the slice.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluation_slice`].
    fn evaluation_step(
        &mut self,
        queue: &mut EvaluationQueue,
        report: &mut JobsReport,
    ) -> Result<Turn, CacheError> {
        #[cfg(test)]
        if !self.hooks.hold_evaluated {
            for tile in std::mem::take(&mut self.hooks.held_evaluated) {
                queue.keep(tile);
            }
        }
        if self.refresh_finished(queue)? {
            return Ok(Turn::Worked);
        }
        let room = queue.room();
        if room == 0 || queue.len() == 0 {
            return Ok(Turn::Idle);
        }
        // As a display job's own evaluation: a round fits the workers not building display
        // tiles (one at least) and the round's time at the cost measured lately, and the cone
        // is of that many tiles, so a slice evaluates no more than one round, its normals or,
        // past an evicted cone, that many tiles whole.
        let workers = self.workers.current_num_threads().max(1);
        let busy: usize = self.jobs.iter_mut().map(|job| job.upkeep().busy()).sum();
        let round = queue.steps.round_size(workers.saturating_sub(busy).max(1));
        let first = queue.first(room.min(round));
        let mut keys = match self.step_cone(&mut queue.steps, || first, round)? {
            ConeStep::Round => return Ok(Turn::Worked),
            ConeStep::Reading => return Ok(Turn::Reading),
            ConeStep::Current(keys) => keys,
        };
        // Highest priority first, as they are handed over; withdrawn ones left out.
        keys.retain(|key| queue.priority(key).is_some());
        keys.sort_by_key(|key| (Reverse(queue.priority(key)), *key));
        if keys.is_empty() {
            return Ok(Turn::Idle);
        }
        let t0 = Instant::now();
        self.evaluate_many(&keys, Evaluation::Normals)?;
        queue.steps.took(t0.elapsed(), keys.len());
        for key in keys {
            let priority = queue
                .priority(&key)
                .map_or(f32::NEG_INFINITY, Priority::get);
            let tile = if let Some(tile) = self.copy_evaluated(key, priority)? {
                tile
            } else {
                // Evicted by the round after it (a budget about the size of the batch).
                let ring = self.positions_with_ring(key)?;
                let normals = self.normals(key)?.to_vec();
                let inputs = self.normal_inputs(&key)?.into_owned();
                EvaluatedTile::new(key, priority, ring, normals, inputs)
            };
            queue.done(&key);
            report.queue_finished += 1;
            #[cfg(test)]
            if self.hooks.hold_evaluated {
                self.hooks.held_evaluated.push(tile);
                continue;
            }
            queue.keep(tile);
        }
        Ok(Turn::Worked)
    }

    /// Brings the finished tiles a write outdated up to date, for about [`REFRESH_TIME`]: each
    /// is copied again from the cache if it is current there, else queued again, behind the
    /// tiles queued for the first time (at its priority, unless it is queued already, at a
    /// newer one). Returns whether there were any.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn refresh_finished(&self, queue: &mut EvaluationQueue) -> Result<bool, CacheError> {
        let began = Instant::now();
        let mut any = false;
        let mut at = 0;
        while let Some(tile) = queue.finished.get(at) {
            if tile.current {
                at += 1;
                continue;
            }
            if any && began.elapsed() >= REFRESH_TIME {
                break;
            }
            any = true;
            let (key, priority) = (tile.key(), tile.priority());
            if let Some(fresh) = self.copy_evaluated(key, priority)? {
                if let Some(slot) = queue.finished.get_mut(at) {
                    *slot = fresh;
                }
                at += 1;
            } else {
                queue.finished.remove(at);
                queue.want_again(key, Priority::new(priority));
            }
        }
        Ok(any)
    }

    /// `key` copied out of the cache for the caller, at `priority`: its positions with the ring
    /// and its normals, as of now. `None` unless it, and every tile its ring reads, is cached
    /// and current.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn copy_evaluated(
        &self,
        key: TileKey,
        priority: f32,
    ) -> Result<Option<EvaluatedTile>, CacheError> {
        let Some(normals) = self.cached_normals(&key) else {
            return Ok(None);
        };
        let Some(ring) = self.cached_positions_with_ring(&key)? else {
            return Ok(None);
        };
        let inputs = self.normal_inputs(&key)?.into_owned();
        Ok(Some(EvaluatedTile::new(
            key,
            priority,
            ring,
            normals.to_vec(),
            inputs,
        )))
    }
}
