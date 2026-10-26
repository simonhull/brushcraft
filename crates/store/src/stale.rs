//! The stale-tile feed: which evaluated tiles each store operation changed, in epochs, for a
//! renderer that keeps tiles of its own.
//!
//! Separate from the change feed ([`crate::ChangeFeed`]), which names correction writes for save
//! and recovery: this one names the tiles whose *evaluated positions* changed (the tiles edited,
//! the finer ones predicted from them, and the neighbours whose shared border vertices, or the
//! one-vertex ring a renderer uploads with each tile, moved) and the display tiles rebuilt. One
//! store operation (a brush step's edits and re-evaluation, a slice of a refresh, the reload of
//! a lost tile) closes one **epoch**: a renderer applies an epoch whole, so a seam never shows
//! one side new and the other old.
//!
//! Each consumer has a cursor ([`StaleCursor`]) and drains the epochs closed since its last
//! drain. A consumer that falls more than a cap of pending keys behind loses them: its next
//! drain is [`Drained::Resync`], and it asks for everything it shows again. So the feed's memory
//! stays bounded however long a consumer stalls.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use geom::TileKey;

/// The default cap on a consumer's pending keys ([`crate::StoreConfig::stale_cap`]).
pub const DEFAULT_STALE_CAP: usize = 65_536;

/// A consumer's place in the stale-tile feed, from
/// [`TileStore::subscribe_stale`](crate::TileStore::subscribe_stale): held by the one consumer
/// it was made for, and given back to unsubscribe.
#[derive(Debug, PartialEq, Eq)]
pub struct StaleCursor(u64);

/// The evaluated tiles one store operation changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Epoch {
    /// The epoch's number: one more than the epoch before it, store-wide.
    pub seq: u64,
    /// The finest tiles and display tiles whose decoded positions (or, for a finest tile, the
    /// ring of neighbouring positions uploaded with it) changed, in key order.
    pub keys: BTreeSet<TileKey>,
}

/// What a drain gives a consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drained {
    /// Every epoch closed since the last drain, in order, each whole.
    Epochs(Vec<Epoch>),
    /// The consumer fell more than the cap behind and lost its epochs: it must ask for
    /// everything it shows again. Its cursor goes on from now.
    Resync,
}

/// One consumer's pending epochs.
#[derive(Debug, Default)]
struct Consumer {
    /// The epochs closed since its last drain, oldest first.
    pending: VecDeque<Epoch>,
    /// How many keys they hold together.
    keys: usize,
    /// Whether it fell past the cap since its last drain.
    lost: bool,
}

/// The feed: the epoch being built and every consumer's pending epochs.
#[derive(Debug)]
pub(crate) struct StaleFeed {
    /// The most keys a consumer may have pending.
    cap: usize,
    /// The next epoch's number.
    next_seq: u64,
    /// The next cursor's number.
    next_cursor: u64,
    /// Each consumer's pending epochs.
    consumers: BTreeMap<u64, Consumer>,
}

impl StaleFeed {
    /// An empty feed whose consumers may each have `cap` keys pending.
    pub(crate) const fn new(cap: usize) -> Self {
        Self {
            cap,
            next_seq: 0,
            next_cursor: 0,
            consumers: BTreeMap::new(),
        }
    }

    /// Whether any consumer listens: the store skips building epochs while none does.
    pub(crate) fn is_listened(&self) -> bool {
        !self.consumers.is_empty()
    }

    /// A new consumer, whose first drain gives the epochs closed after now.
    pub(crate) fn subscribe(&mut self) -> StaleCursor {
        let cursor = self.next_cursor;
        self.next_cursor += 1;
        self.consumers.insert(cursor, Consumer::default());
        StaleCursor(cursor)
    }

    /// Removes a consumer and its pending epochs.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the cursor ends the subscription: nothing can drain with it again"
    )]
    pub(crate) fn unsubscribe(&mut self, cursor: StaleCursor) {
        self.consumers.remove(&cursor.0);
    }

    /// Closes an epoch of `keys`, handing it to every consumer; a consumer it would take past
    /// the cap loses its epochs instead. Nothing is closed for no keys.
    pub(crate) fn close(&mut self, keys: BTreeSet<TileKey>) {
        if keys.is_empty() || self.consumers.is_empty() {
            return;
        }
        let epoch = Epoch {
            seq: self.next_seq,
            keys,
        };
        self.next_seq += 1;
        let size = epoch.keys.len();
        for consumer in self.consumers.values_mut() {
            if consumer.lost {
                continue;
            }
            if consumer.keys + size > self.cap {
                consumer.pending.clear();
                consumer.keys = 0;
                consumer.lost = true;
                continue;
            }
            consumer.keys += size;
            consumer.pending.push_back(epoch.clone());
        }
    }

    /// The epochs closed for `cursor` since its last drain, or [`Drained::Resync`] if it fell
    /// past the cap; an unknown cursor (unsubscribed) is told to resync.
    pub(crate) fn drain(&mut self, cursor: &StaleCursor) -> Drained {
        let Some(consumer) = self.consumers.get_mut(&cursor.0) else {
            return Drained::Resync;
        };
        if std::mem::take(&mut consumer.lost) {
            return Drained::Resync;
        }
        consumer.keys = 0;
        Drained::Epochs(consumer.pending.drain(..).collect())
    }

    /// The most keys any consumer has pending: what the feed holds, at most the cap each.
    #[cfg(test)]
    pub(crate) fn most_pending(&self) -> usize {
        self.consumers.values().map(|c| c.keys).max().unwrap_or(0)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
