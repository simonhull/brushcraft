//! The store's side of the stale-tile feed (see [`crate::stale`]): noting what each operation
//! changes, and closing its epoch.
//!
//! While a consumer listens, every correction write is noted, and display tiles a refresh
//! records are too. Closing the epoch turns the writes into the finest tiles whose evaluated
//! positions they change, or whose one-vertex ring of neighbouring positions they change (the
//! same readers the spatial index's bounds are made unknown by, at the finest level), and hands
//! them out with the display tiles as one epoch. With no consumer, nothing is noted.

use std::collections::BTreeSet;

use geom::{Level, TileKey};

use super::TileStore;
use super::error::CacheError;
use crate::stale::{Drained, EpochKind, StaleCursor};

impl TileStore {
    /// A new consumer of the stale-tile feed: its first drain gives the epochs closed after now
    /// (what it shows already, it asks for itself).
    ///
    /// The cursor is this store's: a renderer subscribes again whenever the store it shows is
    /// replaced (a restore opens a new store, as does opening another project), since every
    /// drain of another store with it is [`Drained::Resync`].
    pub fn subscribe_stale(&mut self) -> StaleCursor {
        self.stale.subscribe()
    }

    /// Removes a consumer of the stale-tile feed.
    pub fn unsubscribe_stale(&mut self, cursor: StaleCursor) {
        self.stale.unsubscribe(cursor);
    }

    /// The epochs closed for `cursor` since its last drain, in order, each whole; or
    /// [`Drained::Resync`] if it fell more than [`StoreConfig::stale_cap`](super::StoreConfig)
    /// keys behind.
    pub fn drain_stale(&mut self, cursor: &StaleCursor) -> Drained {
        match self.stale.drain(cursor) {
            Drained::Resync(_) => Drained::Resync(self.unrefreshed_display()),
            epochs @ Drained::Epochs(_) => epochs,
        }
    }

    /// Whether [`TileStore::drain_stale`] with `cursor` would give anything now: epochs closed
    /// since its last drain, or a resync. Reads only, so a frame loop can ask after work that
    /// may close epochs (jobs, a poll) whether another frame is needed to take them.
    #[must_use]
    pub fn has_stale(&self, cursor: &StaleCursor) -> bool {
        self.stale.has_pending(cursor)
    }

    /// Closes the epoch of the operation since the last one: every finest tile whose evaluated
    /// positions, or ring, the writes since changed, and every display tile recorded since.
    ///
    /// Its kind ([`EpochKind`]): a step's if brush steps' bumps
    /// ([`TileStore::bump_display_error`]) covered every write in it, with their displacements
    /// summed (rounded up: the steps moved a vertex one after another); a revert's if it has
    /// writes no bump covered; else a refresh's. A refresh slice closes any epoch holding writes
    /// before it records its tiles, so a refresh's epoch holds refreshed tiles alone.
    /// What a brush step calls once its edits and their re-evaluation are done; the store calls
    /// it itself after a refresh slice and the reload of a lost tile.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing the writes' readers fails (never, for tiles of the
    /// store's mesh): the epoch stays open then.
    pub fn close_epoch(&mut self) -> Result<(), CacheError> {
        if !self.stale.is_listened() {
            self.epoch_writes.clear();
            self.epoch_keys.clear();
            self.epoch_step = None;
            return Ok(());
        }
        // Taken first: if listing the readers fails, the writes stay noted without it, and the
        // epoch they close in is a revert's (nothing bounds how far they moved).
        let step = self.epoch_step.take();
        // The fallible part next: on an error, the epoch's keys and writes stay noted.
        let moved = if self.epoch_writes.is_empty() {
            BTreeSet::new()
        } else {
            let finest = self
                .display_finest
                .or_else(|| self.epoch_writes.iter().map(TileKey::level).max())
                .unwrap_or(Level::ZERO);
            crate::bounds::readers(
                &self.mesh,
                &self.deps,
                &self.epoch_writes,
                &BTreeSet::from([finest]),
            )?
        };
        let kind = match step {
            Some((displacement, covered)) if self.epoch_writes.is_subset(&covered) => {
                EpochKind::Step { displacement }
            }
            _ if !self.epoch_writes.is_empty() => EpochKind::Revert,
            _ => EpochKind::Refresh,
        };
        self.epoch_writes.clear();
        let mut keys = std::mem::take(&mut self.epoch_keys);
        keys.extend(moved);
        self.stale.close(keys, kind);
        Ok(())
    }

    /// Notes a brush step's bump for the open epoch, while a consumer listens: it moved no finest
    /// vertex farther than `displacement`, with writes to the `edited` tiles. An epoch of
    /// several steps (a frame of dabs) sums them, rounded up: each moved a vertex after the
    /// last. A sum that isn't finite saturates at `f32::MAX`.
    pub(super) fn note_step(&mut self, edited: &BTreeSet<TileKey>, displacement: f32) {
        if self.stale.is_listened() {
            let (sum, covered) = self.epoch_step.get_or_insert_with(Default::default);
            let total = if *sum == 0.0 {
                displacement
            } else {
                (*sum + displacement).next_up()
            };
            // A sum past the largest finite value (or of one that isn't finite) saturates there:
            // still a bound, and one a reader can take.
            *sum = if total.is_finite() { total } else { f32::MAX };
            covered.extend(edited.iter().copied());
        }
    }

    /// Closes the open epoch if it holds writes (or a step's bump), before a refresh records
    /// its tiles: a refresh's epoch then holds refreshed tiles alone, and never passes writes
    /// off as a refresh, nor a refresh as a revert.
    ///
    /// # Errors
    ///
    /// As [`TileStore::close_epoch`].
    pub(super) fn close_writes(&mut self) -> Result<(), CacheError> {
        if self.epoch_writes.is_empty() && self.epoch_step.is_none() {
            return Ok(());
        }
        self.close_epoch()
    }

    /// Notes a correction write of `key` for the open epoch, while a consumer listens.
    pub(super) fn note_epoch_write(&mut self, key: TileKey) {
        if self.stale.is_listened() {
            self.epoch_writes.insert(key);
        }
    }

    /// Notes changed tiles for the open epoch (display tiles recorded), while a consumer listens.
    pub(super) fn note_epoch_keys(&mut self, keys: impl IntoIterator<Item = TileKey>) {
        if self.stale.is_listened() {
            self.epoch_keys.extend(keys);
        }
    }
}
