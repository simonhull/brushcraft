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
use crate::jobs::{ConeSteps, JobsReport};
use crate::pool::Freshness;
use crate::queued::{EvaluatedTile, EvaluationQueue, Priority};

/// About how long a slice spends bringing finished tiles up to date after a correction
/// changed (a copy each, a few dozen microseconds): it leaves the rest for the next slices.
const REFRESH_TIME: Duration = Duration::from_micros(300);

/// The fewest tiles of a cone a step of the queued evaluation alone evaluates.
///
/// A round has a fixed cost (about 80–100 µs, handing work to the workers), so a step of one or
/// two tiles would spend a call's last moments mostly on it. With time for fewer left, a step
/// waits for the next call ([`Turn::Spent`]), unless it is the call's first, which takes these
/// whatever the time, so every call moves on.
const MIN_STEP_TILES: usize = 8;

/// The time a slice of the queued evaluation may take when it has the jobs to itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Alone {
    /// The time left before the call's deadline.
    pub(super) left: Duration,
    /// Whether it is the call's first slice of the queued evaluation: it moves on whatever the
    /// time.
    pub(super) first: bool,
}

/// What a step through a whole cone came to (see [`TileStore::step_whole_cone`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum WholeStep {
    /// As [`TileStore::step_cone`]'s step.
    Step(ConeStep),
    /// Too little time left for a step worth its fixed cost: the next call goes on.
    Spent,
}

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
    /// Too little time left in the call for a step worth its cost: it waits for the next call.
    Spent,
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
        self.queue_evaluation_as(keys, Evaluation::Normals)
    }

    /// [`TileStore::queue_evaluation`], for `what`: with [`Evaluation::Positions`], the tiles
    /// are evaluated for their positions and their ring's alone, and handed over without
    /// normals ([`EvaluatedTile::normals`] is empty), for a renderer that computes them itself
    /// from the ring; that skips the normals' work. A tile queued already takes the latest
    /// call's `what`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::queue_evaluation`].
    pub fn queue_evaluation_as(
        &mut self,
        keys: &[(TileKey, f32)],
        what: Evaluation,
    ) -> Result<usize, CacheError> {
        for (key, _) in keys {
            self.check_key(key)?;
        }
        for &(key, priority) in keys {
            self.evaluation.want_as(key, Priority::new(priority), what);
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
    /// `alone` is the time left before the call's deadline when the display jobs have nothing
    /// to do on the owner thread: the round then grows to fit it, up to
    /// [`ROUND_TIME_ALONE`](crate::jobs::ROUND_TIME_ALONE). Beside the display jobs (`None`),
    /// a round is sized as a display job's.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::prefetch`]; the tiles stay queued.
    pub(super) fn evaluation_slice(
        &mut self,
        report: &mut JobsReport,
        alone: Option<Alone>,
    ) -> Result<Turn, CacheError> {
        let caps = self.evaluation.caps;
        let mut queue = std::mem::replace(&mut self.evaluation, EvaluationQueue::new(caps));
        let turn = self.evaluation_step(&mut queue, report, alone);
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
        alone: Option<Alone>,
    ) -> Result<Turn, CacheError> {
        let began = Instant::now();
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
        let round = if alone.is_some() {
            queue.steps.round_alone()
        } else {
            let workers = self.workers.current_num_threads().max(1);
            let busy: usize = self.jobs.iter_mut().map(|job| job.upkeep().busy()).sum();
            queue.steps.round_size(workers.saturating_sub(busy).max(1))
        };
        let first = queue.first(room.min(round));
        let step = if let Some(alone) = alone {
            match self.step_whole_cone(&mut queue.steps, || first, alone)? {
                WholeStep::Step(step) => step,
                WholeStep::Spent => return Ok(Turn::Spent),
            }
        } else {
            self.step_cone(&mut queue.steps, || first, round)?
        };
        let mut keys = match step {
            ConeStep::Round => return Ok(Turn::Worked),
            ConeStep::Reading => return Ok(Turn::Reading),
            ConeStep::Current(keys) => keys,
        };
        // Highest priority first, as they are handed over; withdrawn ones left out.
        keys.retain(|key| queue.priority(key).is_some());
        keys.sort_by_key(|key| (Reverse(queue.priority(key)), *key));
        // A cone listed for a larger round (the job had the jobs to itself then) finishes a
        // round's worth now; the rest stay queued, current, for the next slices.
        keys.truncate(round);
        if keys.is_empty() {
            return Ok(Turn::Idle);
        }
        let t0 = Instant::now();
        // Positions are current through the cone: only the tiles wanted with normals have work
        // left.
        let with_normals: Vec<TileKey> = keys
            .iter()
            .copied()
            .filter(|key| queue.what(key) == Evaluation::Normals)
            .collect();
        if !with_normals.is_empty() {
            self.evaluate_many(&with_normals, Evaluation::Normals)?;
            queue.steps.took(t0.elapsed(), with_normals.len());
        }
        for (n, key) in keys.into_iter().enumerate() {
            // Alone, the copies keep to the time left too (one at least, so the slice moves
            // on): the rest stay queued, current in the cache, for the next slice to copy.
            if let Some(alone) = alone
                && n > 0
                && began.elapsed() >= alone.left
            {
                break;
            }
            let priority = queue
                .priority(&key)
                .map_or(f32::NEG_INFINITY, Priority::get);
            let what = queue.what(&key);
            let tile = if let Some(tile) = self.copy_evaluated(key, priority, what)? {
                tile
            } else {
                // Evicted by the round after it (a budget about the size of the batch).
                self.evaluated_copy(key, priority, what)?
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

    /// One step of the queued evaluation through the cone `steps` holds (listed as
    /// [`TileStore::step_cone`] lists it), when it has the jobs to itself, with `left` before
    /// the call's deadline: as much of the whole cone as can be evaluated without waiting on
    /// the disk and in the time left, in one round.
    ///
    /// Every correction the cone still lacks is prefetched at once. The tiles evaluated are
    /// the cone's, coarsest first, down to the first level not wholly in memory and those of
    /// that level that are, as many as the time left holds at the cost measured lately
    /// ([`ConeSteps::tiles_alone`]); with time for fewer than [`MIN_STEP_TILES`], none, and it
    /// waits for the next call ([`WholeStep::Spent`]), unless it is the call's first slice
    /// (`alone.first`), which evaluates that many. With the whole cone in memory
    /// and time for it, the cone is evaluated and its finest tiles handed back current
    /// ([`ConeStep::Current`]) in the same step. With nothing in memory to
    /// evaluate and reads in flight, it waits for them ([`ConeStep::Reading`]); with none in
    /// flight (their reads landed and weren't kept), it reads one here. A tile an earlier step
    /// of this cone evaluated, asked for again (a budget too small to keep the cone cached),
    /// hands the finest tiles back to be evaluated whole, so the work always ends.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate_many`] and [`TileStore::prefetch`].
    fn step_whole_cone(
        &mut self,
        steps: &mut ConeSteps,
        keys: impl FnOnce() -> Vec<TileKey>,
        alone: Alone,
    ) -> Result<WholeStep, CacheError> {
        let began = Instant::now();
        if !alone.first && alone.left < steps.least_step(MIN_STEP_TILES) {
            return Ok(WholeStep::Spent);
        }
        let keys = self.list_cone(steps, keys)?;
        let stale_levels = self.stale_levels(steps);
        if stale_levels.is_empty() {
            steps.reset();
            return Ok(WholeStep::Step(ConeStep::Current(keys)));
        }
        let wanting: Vec<TileKey> = stale_levels.iter().flatten().copied().collect();
        self.prefetch(&wanting)?;
        self.poll()?;
        // Coarsest first, down to the first level not wholly in memory and what of it is, as
        // many tiles as the time left holds (what the listing, the prefetch and installing the
        // reads took comes off it).
        let most = steps.tiles_alone(alone.left.saturating_sub(began.elapsed()));
        if most < MIN_STEP_TILES && !alone.first {
            steps.spent += began.elapsed();
            return Ok(WholeStep::Spent);
        }
        let most = most.max(MIN_STEP_TILES);
        let mut targets: Vec<TileKey> = Vec::new();
        let mut whole = true;
        for level in &stale_levels {
            let (ready, unread): (Vec<TileKey>, Vec<TileKey>) =
                level.iter().copied().partition(|key| {
                    self.corrections.find(key).is_some() || self.locate(key).is_none()
                });
            let room = most.saturating_sub(targets.len());
            if unread.is_empty() && ready.len() <= room {
                targets.extend(ready);
                continue;
            }
            whole = false;
            targets.extend(ready.into_iter().take(room));
            break;
        }
        if targets.is_empty() {
            if self.prefetching() > 0 {
                return Ok(WholeStep::Step(ConeStep::Reading));
            }
            // The reads landed and weren't kept (no room for them): read one here.
            targets.extend(stale_levels.first().and_then(|level| level.first()));
        }
        if targets.iter().any(|key| steps.seen.contains(key)) {
            steps.reset();
            return Ok(WholeStep::Step(ConeStep::Current(keys)));
        }
        let evaluating = Instant::now();
        self.evaluate_many(&targets, Evaluation::Positions)?;
        #[cfg(test)]
        std::thread::sleep(
            self.hooks.slow_cone_tile * u32::try_from(targets.len()).unwrap_or(u32::MAX),
        );
        steps.took_tiles_alone(evaluating.elapsed(), targets.len());
        steps.spent += began.elapsed();
        steps.seen.extend(targets);
        if whole {
            steps.took_keys(steps.spent, keys.len());
            steps.reset();
            return Ok(WholeStep::Step(ConeStep::Current(keys)));
        }
        Ok(WholeStep::Step(ConeStep::Round))
    }

    /// The tiles of `steps`' cone whose positions aren't current, level by level, coarsest
    /// first, the levels with none left out.
    fn stale_levels(&mut self, steps: &mut ConeSteps) -> Vec<Vec<TileKey>> {
        let mut stale_levels = Vec::new();
        if let Some(cone) = steps.cone.take() {
            for level in &cone.levels {
                let stale: Vec<TileKey> = level
                    .iter()
                    .copied()
                    .filter(|key| !self.is_current(key))
                    .collect();
                if !stale.is_empty() {
                    stale_levels.push(stale);
                }
            }
            steps.cone = Some(cone);
        }
        stale_levels
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
            let (key, priority, what) = (tile.key(), tile.priority(), tile.what());
            if let Some(fresh) = self.copy_evaluated(key, priority, what)? {
                if let Some(slot) = queue.finished.get_mut(at) {
                    *slot = fresh;
                }
                at += 1;
            } else {
                queue.finished.remove(at);
                queue.want_again(key, Priority::new(priority), what);
            }
        }
        Ok(any)
    }

    /// `key` copied out of the cache for the caller, at `priority`, for `what`: its positions
    /// with the ring, and its normals unless `what` is [`Evaluation::Positions`], as of now.
    /// `None` unless it (with its normals, if wanted), and every tile its ring reads, is cached
    /// and current.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] (not for a valid key).
    fn copy_evaluated(
        &self,
        key: TileKey,
        priority: f32,
        what: Evaluation,
    ) -> Result<Option<EvaluatedTile>, CacheError> {
        let normals = match what {
            Evaluation::Positions => None,
            Evaluation::Normals => match self.cached_normals(&key) {
                Some(normals) => Some(normals.to_vec()),
                None => return Ok(None),
            },
        };
        let Some(ring) = self.cached_positions_with_ring(&key)? else {
            return Ok(None);
        };
        let inputs = self.normal_inputs(&key)?.into_owned();
        Ok(Some(match normals {
            Some(normals) => EvaluatedTile::new(key, priority, ring, normals, inputs),
            None => EvaluatedTile::positions(key, priority, ring, inputs),
        }))
    }

    /// `key` evaluated and copied for the caller, at `priority`, for `what`, whatever the cache
    /// holds: for a tile the cache let go of before it was copied.
    ///
    /// # Errors
    ///
    /// As [`TileStore::positions_with_ring`] and [`TileStore::normals`].
    fn evaluated_copy(
        &mut self,
        key: TileKey,
        priority: f32,
        what: Evaluation,
    ) -> Result<EvaluatedTile, CacheError> {
        let ring = self.positions_with_ring(key)?;
        let inputs = self.normal_inputs(&key)?.into_owned();
        Ok(match what {
            Evaluation::Positions => EvaluatedTile::positions(key, priority, ring, inputs),
            Evaluation::Normals => {
                let normals = self.normals(key)?.to_vec();
                EvaluatedTile::new(key, priority, ring, normals, inputs)
            }
        })
    }
}
