//! The queued evaluation's side of the store: queueing finest tiles by priority, withdrawing
//! them, a slice of the job, and handing over what it finished (see [`crate::queued`]).

use std::cmp::Reverse;
use std::time::{Duration, Instant};

use geom::TileKey;

use super::TileStore;
use super::config::Evaluation;
use super::error::CacheError;
use super::upkeep::ConeStep;
use crate::jobs::{JobsReport, ROUND_MAX};
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
    /// A tile finished before a correction changed is kept back: the next slice of the job
    /// copies it again from the cache if it is still current there, or queues it again
    /// (whatever the change moved, the tile handed over is never older than the store's).
    #[must_use = "the tiles are handed over once: take them to draw them"]
    pub fn take_evaluated(&mut self) -> Vec<EvaluatedTile> {
        let edits = self.edits;
        let (current, outdated): (Vec<EvaluatedTile>, Vec<EvaluatedTile>) =
            std::mem::take(&mut self.evaluation.finished)
                .into_iter()
                .partition(|tile| tile.as_of == edits);
        self.evaluation.finished = outdated;
        current
    }

    /// How many tiles are queued for evaluation and not finished.
    #[must_use]
    pub fn queued_evaluations(&self) -> usize {
        self.evaluation.len()
    }

    /// Whether `key`'s positions and unit normals are cached and current: what
    /// [`TileStore::cached_positions`] and [`TileStore::cached_normals`] then read. Never
    /// evaluates, and doesn't count as a use of the tile.
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
            queue.finished.append(&mut self.hooks.held_evaluated);
        }
        if self.refresh_finished(queue)? {
            return Ok(Turn::Worked);
        }
        let room = queue.room();
        if room == 0 || queue.len() == 0 {
            return Ok(Turn::Idle);
        }
        let workers = self.workers.current_num_threads().max(1);
        let round = queue.steps.round_size(workers);
        let first = queue.first(room.min(ROUND_MAX));
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
                EvaluatedTile::new(key, priority, ring, normals, self.edits)
            };
            queue.done(&key);
            report.queue_finished += 1;
            #[cfg(test)]
            if self.hooks.hold_evaluated {
                self.hooks.held_evaluated.push(tile);
                continue;
            }
            queue.finished.push(tile);
        }
        Ok(Turn::Worked)
    }

    /// Brings the finished tiles a correction change outdated up to date, for about
    /// [`REFRESH_TIME`]: each is copied again from the cache if it is current there, else
    /// queued again at its priority. Returns whether there were any.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn refresh_finished(&self, queue: &mut EvaluationQueue) -> Result<bool, CacheError> {
        let began = Instant::now();
        let mut any = false;
        let mut at = 0;
        while let Some(tile) = queue.finished.get(at) {
            if tile.as_of == self.edits {
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
                queue.want(key, Priority::new(priority));
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
        Ok(Some(EvaluatedTile::new(
            key,
            priority,
            ring,
            normals.to_vec(),
            self.edits,
        )))
    }
}
