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
use crate::stale::{Drained, StaleCursor};

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
        self.stale.drain(cursor)
    }

    /// Closes the epoch of the operation since the last one: every finest tile whose evaluated
    /// positions, or ring, the writes since changed, and every display tile recorded since.
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
            return Ok(());
        }
        let mut keys = std::mem::take(&mut self.epoch_keys);
        if !self.epoch_writes.is_empty() {
            let finest = self
                .display_finest
                .or_else(|| self.epoch_writes.iter().map(TileKey::level).max())
                .unwrap_or(Level::ZERO);
            let moved = crate::bounds::readers(
                &self.mesh,
                &self.deps,
                &self.epoch_writes,
                &BTreeSet::from([finest]),
            )?;
            keys.extend(moved);
            self.epoch_writes.clear();
        }
        self.stale.close(keys);
        Ok(())
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
