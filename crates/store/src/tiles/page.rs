//! Paging: which tiles are resident, loading and prefetching correction tiles, taking and giving
//! back slots under the budget, writing dirty tiles back, and dropping what an edit made stale.

use std::collections::{BTreeMap, BTreeSet};

use geom::{Channel, LayerId, Level, TileKey, Vec3};

use super::TileStore;
use super::config::{BATCH_SHARE, MAX_BATCH};
use super::edit::refused_value;
use super::error::{CacheError, Invariant};
use super::layers::{Dirty, write_slot};
use super::pins::Pinned;
use super::round::{Acquire, Fill, Targets, fill_slot, keep_first, nothing_to_write};
#[cfg(test)]
use crate::file::FileError;
use crate::file::Loc;
use crate::hash::{TileHash, TileSet};
use crate::op::OpId;
use crate::pool::{Freshness, Invalidated, Pool, PoolError, SlotData, SlotId};
use crate::prefetch::{Fetched, Job, Source};
use crate::slot::{CorrectionSlot, EvaluatedSlot};

impl TileStore {
    /// Asserts that only tiles resident in the evaluated pool have a kept prediction or an
    /// [`TileStore::own_edits`] mark: that every eviction and every invalidation that freed a
    /// slot dropped its tile's.
    ///
    /// # Panics
    ///
    /// If a tile that isn't resident has one.
    #[cfg(test)]
    fn assert_kept_resident(&self) {
        let mut stale: Vec<&TileKey> = self
            .kept
            .keys()
            .chain(&self.own_edits)
            .filter(|key| self.evaluated.find(key).is_none())
            .collect();
        stale.sort_unstable();
        assert!(
            stale.is_empty(),
            "predictions kept for tiles not resident: {stale:?}"
        );
    }

    /// Asserts that only resident correction tiles have a [`TileStore::known_ops`] entry: that
    /// every eviction and removal dropped its tile's.
    ///
    /// # Panics
    ///
    /// If a tile that isn't resident has one.
    #[cfg(test)]
    pub(super) fn assert_known_ops_resident(&self) {
        let mut stale: Vec<&TileKey> = self
            .known_ops
            .keys()
            .filter(|key| self.corrections.find(key).is_none())
            .collect();
        stale.sort_unstable();
        assert!(
            stale.is_empty(),
            "ops kept for tiles not resident: {stale:?}"
        );
    }

    /// Frees every unpinned slot of both pools, writing dirty correction tiles back first.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`]: that tile and those not reached yet stay.
    pub fn evict_all(&mut self) -> Result<(), CacheError> {
        self.sync_pins()?;
        let evaluated = self.evaluated.evict_all(nothing_to_write);
        self.note_evictions();
        evaluated?;
        let (file, unsaved) = (&mut self.file, &mut self.unsaved);
        let corrections = self
            .corrections
            .evict_all(|key, slot| write_slot(file, unsaved, key, slot).map(drop));
        self.forget_evicted_ops();
        Ok(corrections?)
    }

    /// Gives chunks back until the pools hold at most `bytes` or only chunks with pinned slots
    /// are left: evaluated chunks first, then correction chunks (dirty tiles written back), the
    /// cheapest of each pool first (see the memory policy).
    ///
    /// # Errors
    ///
    /// As [`TileStore::evict_all`].
    pub fn shrink_to(&mut self, bytes: usize) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.give_back(bytes)?;
        Ok(())
    }

    /// Asks the prefetch workers to read the correction tiles of `keys` in the background, ahead
    /// of the store's use of them, each from the file a load would read (the working file, else
    /// the base); returns how many reads were queued.
    ///
    /// Tiles already resident, absent from the file (all zero) or already on their way are
    /// skipped. This call's tiles are read before those of earlier calls still waiting, in the
    /// order given. They are installed by [`TileStore::poll`], which every method that reads
    /// corrections calls first. The workers start on the first call that queues anything.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`], before anything is queued; [`CacheError::Store`] if the
    /// workers can't be started.
    pub fn prefetch(&mut self, keys: &[TileKey]) -> Result<usize, CacheError> {
        self.sync_pins()?;
        for key in keys {
            self.check_key(key)?;
        }
        let mut seen = TileSet::with_capacity_and_hasher(keys.len(), TileHash);
        let jobs: Vec<Job> = keys
            .iter()
            .filter(|&&key| seen.insert(key) && self.corrections.find(&key).is_none())
            .filter_map(|&key| {
                let (source, loc) = self.locate(&key)?;
                Some(Job { key, source, loc })
            })
            .collect();
        Ok(self
            .prefetch
            .request(&self.file, self.base.as_ref(), &jobs)?)
    }

    /// Installs the correction tiles the prefetch workers have finished, and returns how many.
    ///
    /// A finished tile is installed only if it is still wanted and still current: not resident
    /// (a resident tile may have been edited since, and is what the store uses either way), and
    /// its record still the one a load would read (any write of the tile, by an edit's
    /// write-back or a save, moves it to the working file). It goes into a free slot, a new chunk
    /// if the budget allows one, or the slot of a clean correction tile: prefetch never takes a
    /// pinned slot, an evaluated chunk or a dirty tile's slot, so it never writes anything
    /// back. Anything else is dropped, as is a tile a worker couldn't read: a later load reads
    /// it again and reports the error.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a fresh slot's handle went stale (it doesn't); the results not
    /// reached yet wait for the next poll.
    pub fn poll(&mut self) -> Result<usize, CacheError> {
        let mut installed = 0;
        while let Some(Fetched { job, slot }) = self.prefetch.next() {
            let Ok((op, slot)) = slot else { continue };
            let current = self.locate(&job.key) == Some((job.source, job.loc));
            let outcome = if current {
                self.install(job.key, op, &slot)
            } else {
                Ok(false)
            };
            self.prefetch.recycle(slot);
            if outcome? {
                installed += 1;
            }
        }
        #[cfg(test)]
        {
            self.assert_known_ops_resident();
            self.assert_kept_resident();
        }
        Ok(installed)
    }

    /// Prefetch requests not yet answered: queued, being read, or finished and waiting for
    /// [`TileStore::poll`].
    #[must_use]
    pub fn prefetching(&self) -> usize {
        self.prefetch.pending()
    }

    /// Puts a prefetched correction tile, read at its latest record (written by `op`), into a
    /// fresh slot; whether it did. A resident tile (every dirty tile is resident) is left alone,
    /// and so is the tile when the correction pool has no slot to give without writing a dirty
    /// tile back or taking memory from the evaluated pool (see [`Pool::acquire_clean`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if the fresh slot's handle went stale (it doesn't).
    fn install(
        &mut self,
        key: TileKey,
        op: OpId,
        tile: &CorrectionSlot,
    ) -> Result<bool, CacheError> {
        if self.corrections.find(&key).is_some() {
            return Ok(false);
        }
        let acquired = self.corrections.acquire_clean(key, &mut self.budget);
        self.forget_evicted_ops();
        let id = match acquired {
            Ok(id) => id,
            Err(PoolError::Full) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        *self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)? = *tile;
        self.known_ops.insert(key, op);
        #[cfg(test)]
        {
            self.hooks.log.prefetched += 1;
        }
        Ok(true)
    }

    /// Applies the unpins of dropped [`PinGuard`](super::pins::PinGuard)s.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a pinned slot's handle went stale (pinned slots are never
    /// freed, so this doesn't happen). Every released set is unpinned either way.
    pub(super) fn sync_pins(&mut self) -> Result<(), CacheError> {
        let released = std::mem::take(&mut self.pins.borrow_mut().released);
        let mut outcome = Ok(());
        for id in released {
            if let Some(set) = self.pin_sets.remove(&id) {
                keep_first(&mut outcome, self.unpin_set(&set));
            }
        }
        outcome
    }

    /// Unpins one pin set's slots.
    ///
    /// # Errors
    ///
    /// As [`TileStore::sync_pins`]; every slot is unpinned either way.
    pub(super) fn unpin_set(&mut self, set: &[Pinned]) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &(evaluated, correction) in set {
            keep_first(&mut outcome, self.evaluated.unpin(evaluated));
            keep_first(&mut outcome, self.corrections.unpin(correction));
        }
        outcome
    }

    /// Evaluates the tiles of `batch` (of one level, in key order) and pins their evaluated and
    /// correction slots, adding them to `set`.
    ///
    /// # Errors
    ///
    /// As [`TileStore::evaluate`]; nothing this call pinned stays pinned then.
    pub(super) fn pin_batch(
        &mut self,
        batch: &[TileKey],
        set: &mut Vec<Pinned>,
    ) -> Result<(), CacheError> {
        let held = self.pin_current(batch.iter().copied().collect())?;
        let mut corrections = Vec::with_capacity(held.len());
        let loaded = held.iter().try_for_each(|&(key, _)| {
            let id = self.load_correction(key)?;
            self.corrections.pin(id)?;
            corrections.push(id);
            Ok::<_, CacheError>(())
        });
        if let Err(e) = loaded {
            self.unpin_corrections(&corrections)?;
            self.unpin_held(&held)?;
            return Err(e);
        }
        set.extend(held.iter().map(|&(_, id)| id).zip(corrections));
        Ok(())
    }

    /// Runs `run` over `keys` in batches, for [`TileStore::evaluate_many`] and
    /// [`TileStore::pin`]: level by level, coarse to fine, each level's keys in key order, cut
    /// into batches of [`TileStore::batch_size`] tiles. A batch refused with
    /// [`CacheError::Full`] is halved and run again, until it is one tile; the size reached is
    /// kept for the batches after it.
    ///
    /// `run` must leave nothing it pinned pinned when it fails, so that a smaller batch finds
    /// the room the failed one had. With [`Targets::Pinned`], the targets still to come count
    /// against each batch's room, as they will be pinned beside it.
    ///
    /// # Errors
    ///
    /// The first error of `run` but [`CacheError::Full`] on a batch of more than one tile.
    pub(super) fn in_batches(
        &mut self,
        keys: &BTreeSet<TileKey>,
        targets: Targets,
        mut run: impl FnMut(&mut Self, &[TileKey]) -> Result<(), CacheError>,
    ) -> Result<(), CacheError> {
        let mut levels: BTreeMap<Level, Vec<TileKey>> = BTreeMap::new();
        for &key in keys {
            levels.entry(key.level()).or_default().push(key);
        }
        let mut pending = match targets {
            Targets::Pinned => keys.len(),
            Targets::Released => 0,
        };
        for level in levels.values() {
            let mut rest = level.as_slice();
            let mut size = self.batch_size(pending);
            while !rest.is_empty() {
                let (batch, after) = rest.split_at(size.min(rest.len()));
                match run(self, batch) {
                    Err(CacheError::Full) if batch.len() > 1 => size = batch.len() / 2,
                    Err(e) => return Err(e),
                    Ok(()) => {
                        pending = pending.saturating_sub(batch.len());
                        rest = after;
                    }
                }
            }
        }
        Ok(())
    }

    /// The targets the next batch of [`TileStore::in_batches`] takes: a [`BATCH_SHARE`] of the
    /// evaluated slots the pool can reach besides those pinned, the `pending` targets still to
    /// come (which [`TileStore::pin`] has counted on keeping room for) and an evaluation's own
    /// headroom; at least one, at most [`MAX_BATCH`].
    ///
    /// The evaluated pool reaches the budget less what the correction pool keeps from it: the
    /// correction floor (one chunk, see [`TileStore::make_room`]) and the chunks its pinned
    /// slots, which can't go, and its dirty ones, which go only after a write, fill. Those are
    /// counted as if packed into chunks, not by the chunks they sit in: where a tile sits
    /// depends on when prefetched tiles were installed, and the count must not. What the
    /// evaluated pool reaches is whole chunks and, in the room below a whole one, the short chunk
    /// it would map there ([`Pool::short_slots`]).
    ///
    /// It depends on the budget and the counts of pinned and dirty slots alone, never on the
    /// thread count or where tiles sit, so the batches, like the rounds, are the same whatever
    /// the workers.
    pub(super) fn batch_size(&self, pending: usize) -> usize {
        let held = self.corrections.pinned_slots() + self.unsaved.len();
        let held_chunks = held.div_ceil(self.corrections.per_chunk() as usize);
        let kept = (held_chunks + 1) * self.corrections.chunk_bytes();
        let reachable = self.budget.limit().saturating_sub(kept);
        let chunk = self.evaluated.chunk_bytes().max(1);
        let (chunks, rest) = (reachable / chunk, reachable % chunk);
        let slots = chunks
            .saturating_mul(self.evaluated.per_chunk() as usize)
            .saturating_add(self.evaluated.short_slots(rest) as usize);
        let taken = self.evaluated.pinned_slots() + pending + self.headroom.evaluated;
        (slots.saturating_sub(taken) / BATCH_SHARE).clamp(1, MAX_BATCH)
    }

    /// Unpins correction slots a round or a pin batch pinned.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale (pinned slots are never freed, so this
    /// doesn't happen); every slot is unpinned either way.
    pub(super) fn unpin_corrections(
        &mut self,
        ids: &[SlotId<CorrectionSlot>],
    ) -> Result<(), CacheError> {
        let mut outcome = Ok(());
        for &id in ids {
            keep_first(&mut outcome, self.corrections.unpin(id));
        }
        outcome
    }

    /// Whether pinning `keys` fits the budget: the chunks each pool needs to hold its pinned
    /// slots besides these tiles, one slot per tile, and the store's own headroom (an
    /// evaluation's pins, and the correction slot it loads), against the budget. Slots are
    /// counted as if packed into chunks; pins spread over chunks can still run a pool out of
    /// room later.
    ///
    /// # Errors
    ///
    /// [`CacheError::OverBudget`].
    pub(super) fn check_pin_budget(&self, keys: &BTreeSet<TileKey>) -> Result<(), CacheError> {
        /// Bytes of chunks `pool` needs for its pinned slots outside `keys`, one per key, and
        /// `headroom` more.
        fn chunks_needed<T: SlotData>(
            pool: &Pool<T>,
            keys: &BTreeSet<TileKey>,
            headroom: usize,
        ) -> usize {
            let pinned_here = keys.iter().filter(|key| pool.is_pinned(key)).count();
            let slots = pool.pinned_slots().saturating_sub(pinned_here) + keys.len() + headroom;
            slots.div_ceil(pool.per_chunk() as usize) * pool.chunk_bytes()
        }
        let needed = chunks_needed(&self.evaluated, keys, self.headroom.evaluated)
            + chunks_needed(&self.corrections, keys, 1);
        if needed > self.budget.limit() {
            return Err(CacheError::OverBudget {
                tiles: keys.len(),
                needed,
                budget: self.budget.limit(),
            });
        }
        Ok(())
    }

    /// Checks each of `keys` as [`TileStore::check_key`] does, and that an op is current or can
    /// be begun, before anything of an edit is written.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OpsExhausted`].
    pub(super) fn check_keys_and_op(&mut self, keys: &[TileKey]) -> Result<(), CacheError> {
        keys.iter().try_for_each(|key| self.check_key(key))?;
        self.edit_op().map(drop)
    }

    /// Checks that `key` names a tile this store holds: base layer, displacement, level ≥ 1, a
    /// face of the mesh.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`].
    pub(super) fn check_key(&self, key: &TileKey) -> Result<(), CacheError> {
        let ok = key.layer() == LayerId::BASE
            && key.channel() == Channel::DISPLACEMENT
            && key.level() != Level::ZERO
            && self.mesh.topology().contains(key.face());
        if ok {
            Ok(())
        } else {
            Err(CacheError::NotATile(*key))
        }
    }

    /// Where a load of `key`'s correction tile reads it: its record in the working file, else in
    /// the base; `None` if neither has one (the tile is zero).
    pub(super) fn locate(&self, key: &TileKey) -> Option<(Source, Loc)> {
        self.file
            .locate(key)
            .map(|loc| (Source::Working, loc))
            .or_else(|| {
                let loc = self.base.as_ref()?.locate(key)?;
                Some((Source::Base, loc))
            })
    }

    /// `key`'s correction slot: resident, else loaded from the working file, else from the base,
    /// else reset to zero.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if the record can't be read (the slot is freed again);
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if no slot can be had.
    pub(super) fn load_correction(
        &mut self,
        key: TileKey,
    ) -> Result<SlotId<CorrectionSlot>, CacheError> {
        self.load_correction_as(key, Acquire::Any)
    }

    /// [`TileStore::load_correction`], getting a slot for a tile that isn't resident as
    /// `acquire` allows.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]; with [`Acquire::Clean`], [`CacheError::Full`] when
    /// no free slot, new chunk or clean tile's slot is to be had.
    fn load_correction_as(
        &mut self,
        key: TileKey,
        acquire: Acquire,
    ) -> Result<SlotId<CorrectionSlot>, CacheError> {
        if let Some((id, _)) = self.corrections.lookup(&key) {
            return Ok(id);
        }
        let fill = self.take_correction_slot(key, acquire)?;
        let slot = self
            .corrections
            .get_mut(fill.id)
            .ok_or(Invariant::StaleHandle)?;
        let filled = fill_slot(
            &self.file,
            self.base.as_ref(),
            &fill,
            slot,
            &mut self.read_scratch,
        );
        #[cfg(test)]
        let filled = self.as_tested(key, filled);
        self.settle_fill(&fill, filled)?;
        Ok(fill.id)
    }

    /// A slot for `key`'s correction tile, which isn't resident, taken as `acquire` allows, and
    /// where its record is: the first half of loading it, which [`fill_slot`] and
    /// [`TileStore::settle_fill`] finish.
    ///
    /// # Errors
    ///
    /// As [`TileStore::acquire_correction`] or, with [`Acquire::Clean`],
    /// [`CacheError::Full`] when no free slot, new chunk or clean tile's slot is to be had.
    pub(super) fn take_correction_slot(
        &mut self,
        key: TileKey,
        acquire: Acquire,
    ) -> Result<Fill, CacheError> {
        let id = match acquire {
            Acquire::Any => self.acquire_correction(key),
            Acquire::Clean => self
                .corrections
                .acquire_clean(key, &mut self.budget)
                .map_err(CacheError::from),
        };
        // Taking a slot may have evicted others, whether or not it succeeded.
        self.forget_evicted_ops();
        let id = id?;
        self.activity.corrections_loaded += 1;
        #[cfg(test)]
        self.hooks.log.file_reads.push(key);
        Ok(Fill {
            key,
            id,
            found: self.locate(&key),
        })
    }

    /// `filled`, the outcome of filling `key`'s slot, as the tests see it: an error for the
    /// tile they made unreadable ([`TestHooks::unreadable`](super::hooks::TestHooks::unreadable)).
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] for the unreadable tile; else `filled`'s.
    #[cfg(test)]
    pub(super) fn as_tested(
        &self,
        key: TileKey,
        filled: Result<Option<OpId>, CacheError>,
    ) -> Result<Option<OpId>, CacheError> {
        if self.hooks.unreadable == Some(key) {
            Err(FileError::from(std::io::Error::other("an unreadable tile, for the test")).into())
        } else {
            filled
        }
    }

    /// The last half of loading a correction tile: records the op of the record its slot was
    /// filled from (see [`TileStore::known_ops`]), or, if filling it failed, gives the slot back
    /// (it is clean, and unpinned by the caller) and returns the error.
    ///
    /// # Errors
    ///
    /// `filled`'s error; [`CacheError::Internal`] if the slot can't be given back.
    pub(super) fn settle_fill(
        &mut self,
        fill: &Fill,
        filled: Result<Option<OpId>, CacheError>,
    ) -> Result<(), CacheError> {
        match filled {
            Ok(Some(op)) => {
                self.known_ops.insert(fill.key, op);
                Ok(())
            }
            Ok(None) => {
                self.known_ops.remove(&fill.key);
                Ok(())
            }
            Err(e) => {
                // The slot holds no tile: give it back.
                self.corrections.remove(&fill.key)?;
                Err(e)
            }
        }
    }

    /// A fresh slot for `key`'s correction tile, as the memory policy orders it (see
    /// [`TileStore`]): a free slot, a new chunk, a clean victim, an evaluated chunk taken back,
    /// one dirty victim written back, and only when every correction slot is pinned, chunks
    /// taken back from the pools.
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`]; [`CacheError::WriteBack`] if a dirty tile's write-back fails.
    fn acquire_correction(&mut self, key: TileKey) -> Result<SlotId<CorrectionSlot>, CacheError> {
        match self.corrections.acquire_clean(key, &mut self.budget) {
            Err(PoolError::Full) => {}
            other => return Ok(other?),
        }
        if self.may_take_evaluated_chunk() {
            let target = self
                .budget
                .limit()
                .saturating_sub(self.corrections.chunk_bytes());
            let shrunk = self
                .evaluated
                .shrink_to(&mut self.budget, target, nothing_to_write);
            self.note_evictions();
            shrunk?;
            match self.corrections.acquire_clean(key, &mut self.budget) {
                Err(PoolError::Full) => {}
                other => return Ok(other?),
            }
        }
        for attempt in 0..2 {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            match self.corrections.acquire(key, &mut self.budget, |k, slot| {
                write_slot(file, unsaved, k, slot).map(drop)
            }) {
                Err(PoolError::Full) if attempt == 0 => {
                    if !self.make_room(self.corrections.chunk_bytes(), 0)? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
    }

    /// Whether a correction acquire that found no free or clean slot may take an evaluated
    /// chunk back rather than write a dirty tile back: while the correction pool holds less
    /// than a quarter of the pools' share (the budget less the reserved shares), and the
    /// evaluated pool keeps its floor (its share of the minimum budget plus one chunk) without
    /// the chunk, and has a chunk without pins to give.
    pub(super) fn may_take_evaluated_chunk(&self) -> bool {
        let chunk = self.evaluated.chunk_bytes();
        let floor = self.headroom.evaluated_floor(chunk);
        self.corrections.bytes().saturating_mul(4) < self.budget.limit()
            && self.evaluated.bytes().saturating_sub(chunk) >= floor
            && self.evaluated.has_unpinned_chunk()
    }

    /// A fresh slot for `key`'s evaluated tile, taking chunks back from the pools if the
    /// evaluated pool is full (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::Full`]; [`CacheError::WriteBack`] if taking a correction chunk back
    /// writes a dirty tile back and that fails.
    pub(super) fn acquire_evaluated(
        &mut self,
        key: TileKey,
    ) -> Result<SlotId<EvaluatedSlot>, CacheError> {
        for attempt in 0..2 {
            // Evaluated slots are never dirty: there is nothing to write back.
            let acquired = self
                .evaluated
                .acquire(key, &mut self.budget, nothing_to_write);
            self.note_evictions();
            match acquired {
                Err(PoolError::Full) if attempt == 0 => {
                    let keep = self.corrections.chunk_bytes();
                    if !self.make_room(self.evaluated.chunk_bytes(), keep)? {
                        return Err(CacheError::Full);
                    }
                }
                other => return Ok(other?),
            }
        }
        Err(CacheError::Full)
    }

    /// Takes chunks back until another `need` bytes fit the budget, leaving the correction pool
    /// at least `keep` bytes of its chunks; whether they fit now.
    ///
    /// Evaluated chunks go first, then correction chunks (see [`TileStore::give_back`]), but
    /// never below `keep`. The evaluated pool asks with one correction chunk kept: the
    /// correction floor. Without it, a batch whose cone pins every evaluated chunk could take
    /// every correction chunk back, fail, and leave the one-tile batch it is halved to with no
    /// correction slot to load into, as the coarsest level's pins would then sit in every
    /// evaluated chunk (see the module notes). The correction pool asks with nothing kept: its
    /// own chunks are what it is short of.
    ///
    /// # Errors
    ///
    /// As [`TileStore::give_back`].
    fn make_room(&mut self, need: usize, keep: usize) -> Result<bool, CacheError> {
        let target = self.budget.limit().saturating_sub(need);
        let evaluated = self
            .evaluated
            .shrink_to(&mut self.budget, target, nothing_to_write);
        self.note_evictions();
        evaluated?;
        let floor = self.evaluated.bytes().saturating_add(keep);
        self.give_back(target.max(floor))?;
        Ok(self.budget.used() <= target)
    }

    /// Gives chunks without pinned slots back until the pools hold at most `bytes`: evaluated
    /// chunks first (a cache, cheap to recompute), then correction chunks, whose dirty tiles are
    /// written back to the working file first; the cheapest of each pool first (see
    /// [`TileStore`]'s memory policy).
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`].
    fn give_back(&mut self, bytes: usize) -> Result<(), CacheError> {
        let evaluated = self
            .evaluated
            .shrink_to(&mut self.budget, bytes, nothing_to_write);
        self.note_evictions();
        evaluated?;
        if self.budget.used() > bytes {
            let (file, unsaved) = (&mut self.file, &mut self.unsaved);
            let shrunk = self
                .corrections
                .shrink_to(&mut self.budget, bytes, |key, slot| {
                    write_slot(file, unsaved, key, slot).map(drop)
                });
            self.forget_evicted_ops();
            shrunk?;
        }
        #[cfg(test)]
        self.assert_known_ops_resident();
        Ok(())
    }

    /// Drops the [`TileStore::known_ops`] of the correction tiles the pool has evicted since
    /// the last call, so that only resident tiles have one. A tile resident again keeps its
    /// entry: no flow reloads a tile between its eviction and the next drain today (each slot
    /// taken drains before it is filled), so that check only guards a future one.
    ///
    /// Called wherever the pool may evict: after taking a slot (to load a tile or install a
    /// prefetched one), evicting all, or giving chunks back, each whether or not it succeeded;
    /// and at a rebase or roll-over, which so find nothing left to drop. Tests assert after
    /// those, and after every poll, that only resident tiles have an op.
    pub(super) fn forget_evicted_ops(&mut self) {
        for key in self.corrections.drain_evicted() {
            self.activity.corrections_evicted += 1;
            if self.corrections.find(&key).is_none() {
                self.known_ops.remove(&key);
            }
        }
    }

    /// Tells the dependency graph which evaluated tiles the pool evicted.
    pub(super) fn note_evictions(&mut self) {
        for key in self.evaluated.drain_evicted() {
            self.activity.evaluated_evicted += 1;
            self.kept.remove(&key);
            self.own_edits.remove(&key);
            self.deps.left_pool(key);
        }
    }

    /// Writes `c` at `(i, j)` of `key`'s correction slot `id` (resident and pinned) as part of
    /// `op`, and marks the slot written (see [`TileStore::mark_written`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with
    /// [`CorrectionError::TooLarge`](geom::CorrectionError::TooLarge) or
    /// [`CorrectionError::NotFinite`](geom::CorrectionError::NotFinite) for a value no tile can
    /// hold (the slot is unchanged); [`CacheError::Internal`] for a stale handle, or if the
    /// slot doesn't hold a valid tile or `(i, j)` is outside it (neither happens for a loaded
    /// slot and a copy's coordinates).
    pub(super) fn write_correction(
        &mut self,
        key: TileKey,
        id: SlotId<CorrectionSlot>,
        (i, j): (u32, u32),
        c: Vec3,
        op: OpId,
    ) -> Result<(), CacheError> {
        let slot = self.corrections.get_mut(id).ok_or(Invariant::StaleHandle)?;
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        view.set(i, j, c).map_err(refused_value)?;
        self.mark_written(key, id, op)
    }

    /// Records that `key`'s correction slot `id` was written as part of `op`: the write takes
    /// the next stamp, and the slot is dirty, to be written with `op`, and pushed to the change
    /// feed with it (the feed skips the push if no consumer has drained the same key and op
    /// yet).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    pub(super) fn mark_written(
        &mut self,
        key: TileKey,
        id: SlotId<CorrectionSlot>,
        op: OpId,
    ) -> Result<(), CacheError> {
        self.corrections_changed();
        self.last_write = self.last_write.next();
        self.unsaved.insert(
            key,
            Dirty {
                op,
                stamp: self.last_write,
            },
        );
        self.feed.push(key, op);
        self.bounds.note_write(key);
        self.display.note_write(key);
        self.corrections.set_dirty(id, true)?;
        Ok(())
    }

    /// Drops the evaluations that followed from the `edited` tiles' corrections (see
    /// [`TileStore::invalidate`]), skipping each tile nothing can have followed from: one whose
    /// own positions aren't current and that nothing recorded reads.
    ///
    /// Anything computed from a tile's corrections is computed from its positions, and its
    /// positions only from the tile being evaluated, which makes them current. So once a tile's
    /// evaluations are dropped, edits of it have nothing more to drop until it is evaluated
    /// again, however many there are and whatever their ops. While the dependency graph holds
    /// an untracked node, nothing is skipped: the cascade takes such a node to follow from
    /// every coarser edit, recorded or not (see [`crate::deps`]).
    pub(super) fn invalidate_edited(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        let reached: Vec<TileKey> = edited
            .into_iter()
            .filter(|key| {
                self.evaluated.freshness(key) == Some(Freshness::Fresh)
                    || self.deps.is_read(key)
                    || self.deps.has_untracked()
            })
            .collect();
        if !reached.is_empty() {
            self.invalidate(reached);
        }
    }

    /// Drops every evaluation that followed from the `edited` tiles' corrections: positions
    /// through the recorded parents, transitively, and the normals of those tiles. A pinned
    /// tile keeps its slot, marked stale, and is recomputed in place on its next evaluation.
    ///
    /// Normals that read a dropped tile's positions but whose own tile's positions stay are
    /// kept, marked stale since the inputs that changed (see [`crate::deps`]): their next
    /// computation recomputes only the vertices that read those.
    ///
    /// A tile whose parents' positions changed is predicted from other positions now: its kept
    /// prediction goes. A pinned tile edited alone, its parents as they were, keeps its kept
    /// prediction, which its corrections don't change, and is marked for its next one to be
    /// kept ([`TileStore::own_edits`]). A tile whose parents the graph doesn't record (an
    /// untracked node, see [`crate::deps`]) counts as moved: the cascade takes it to follow
    /// from every coarser edit, and nothing says which.
    fn invalidate(&mut self, edited: impl IntoIterator<Item = TileKey>) {
        #[cfg(test)]
        {
            self.hooks.log.invalidations += 1;
        }
        let edited: BTreeSet<TileKey> = edited.into_iter().collect();
        let cascade = self.deps.cascade(edited.iter().copied());
        for (key, changed) in cascade.normals {
            self.deps.stale_normals(key, changed);
        }
        let moved: BTreeSet<TileKey> = cascade
            .positions
            .iter()
            .filter(|key| {
                self.deps.parents(key).is_none_or(|parents| {
                    parents
                        .iter()
                        .any(|parent| cascade.positions.contains(parent))
                })
            })
            .copied()
            .collect();
        for key in cascade.positions {
            let parents_moved = moved.contains(&key);
            if parents_moved {
                self.kept.remove(&key);
            }
            match self.evaluated.invalidate(&key) {
                Invalidated::Marked if edited.contains(&key) && !parents_moved => {
                    self.own_edits.insert(key);
                }
                Invalidated::Marked => {
                    self.own_edits.remove(&key);
                }
                Invalidated::Freed | Invalidated::Absent => {
                    self.kept.remove(&key);
                    self.own_edits.remove(&key);
                    self.deps.left_pool(key);
                }
            }
        }
    }
}
