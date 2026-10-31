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
//!
//! A cursor belongs to one store's feed: given to another store (after a restore, which opens a
//! new store, or a project swapped), every drain is [`Drained::Resync`]. A renderer subscribes
//! again to each store it shows.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use geom::TileKey;

/// The default cap on a consumer's pending keys ([`crate::StoreConfig::stale_cap`]).
pub const DEFAULT_STALE_CAP: usize = 65_536;

/// A consumer's place in the stale-tile feed, from
/// [`TileStore::subscribe_stale`](crate::TileStore::subscribe_stale).
///
/// Held by the one consumer it was made for, and given back to unsubscribe. It names the feed it
/// came from, so another store's feed never mistakes it for one of its own.
#[derive(Debug, PartialEq, Eq)]
// Cloned only by the tests, to use a cursor after giving it back.
#[cfg_attr(test, derive(Clone))]
pub struct StaleCursor {
    /// The feed it came from.
    feed: u64,
    /// The consumer, among that feed's.
    consumer: u64,
}

/// What kind of operation closed an epoch: what a renderer may assume of the tiles it names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpochKind {
    /// One or more brush steps: their writes moved no finest vertex farther than
    /// `displacement` (world units; several steps' displacements summed), and the display
    /// tiles it names had only their bound raised
    /// ([`TileStore::bump_display_error`](crate::TileStore::bump_display_error)).
    Step {
        /// A bound on how far the steps moved any finest vertex.
        displacement: f32,
    },
    /// A refresh: the display tiles it names were rebuilt from the finest level as it is.
    Refresh,
    /// Anything else that moved finest vertices by an unknown amount: a lost tile going back
    /// to its record, a write outside a step. Display tiles above its finest tiles are only
    /// what the store holds for them.
    Revert,
}

/// A display tile a stroke changed since its last refresh, as a resync hands it over: what a
/// renderer needs to stand in for it again ([`Drained::Resync`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Unrefreshed {
    /// The display tile.
    pub tile: TileKey,
    /// A bound on how far the steps since its record moved any finest vertex under it.
    pub displacement: f32,
    /// The finest tiles under it those steps moved.
    pub finest: BTreeSet<TileKey>,
}

/// The evaluated tiles one store operation changed.
#[derive(Debug, Clone, PartialEq)]
pub struct Epoch {
    /// The epoch's number: one more than the epoch before it, store-wide.
    pub seq: u64,
    /// What closed it.
    pub kind: EpochKind,
    /// The finest tiles and display tiles whose decoded positions (or, for a finest tile, the
    /// ring of neighbouring positions uploaded with it) changed, in key order.
    pub keys: BTreeSet<TileKey>,
}

/// What a drain gives a consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum Drained {
    /// Every epoch closed since the last drain, in order, each whole.
    Epochs(Vec<Epoch>),
    /// The consumer fell more than the cap behind and lost its epochs: it must ask for
    /// everything it shows again. Its cursor goes on from now. It carries every display tile a
    /// stroke changed since its last refresh ([`Unrefreshed`]), which the lost epochs named.
    Resync(Vec<Unrefreshed>),
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
    /// This feed's identity among every feed of the process: its cursors carry it.
    id: u64,
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
    pub(crate) fn new(cap: usize) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};

        /// The next feed's identity.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
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
        StaleCursor {
            feed: self.id,
            consumer: cursor,
        }
    }

    /// Removes a consumer and its pending epochs; a cursor of another feed changes nothing.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the cursor ends the subscription: nothing can drain with it again"
    )]
    pub(crate) fn unsubscribe(&mut self, cursor: StaleCursor) {
        if cursor.feed == self.id {
            self.consumers.remove(&cursor.consumer);
        }
    }

    /// Closes an epoch of `keys`, of `kind`, handing it to every consumer; a consumer it would
    /// take past the cap loses its epochs instead. Nothing is closed for no keys.
    pub(crate) fn close(&mut self, keys: BTreeSet<TileKey>, kind: EpochKind) {
        if keys.is_empty() || self.consumers.is_empty() {
            return;
        }
        let epoch = Epoch {
            seq: self.next_seq,
            kind,
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
    /// past the cap; an unknown cursor (unsubscribed, or another feed's) is told to resync.
    pub(crate) fn drain(&mut self, cursor: &StaleCursor) -> Drained {
        if cursor.feed != self.id {
            return Drained::Resync(Vec::new());
        }
        let Some(consumer) = self.consumers.get_mut(&cursor.consumer) else {
            return Drained::Resync(Vec::new());
        };
        if std::mem::take(&mut consumer.lost) {
            return Drained::Resync(Vec::new());
        }
        consumer.keys = 0;
        Drained::Epochs(consumer.pending.drain(..).collect())
    }

    /// Whether a drain with `cursor` would give anything: epochs pending, or a resync (lost
    /// epochs, or a cursor this feed doesn't know).
    pub(crate) fn has_pending(&self, cursor: &StaleCursor) -> bool {
        if cursor.feed != self.id {
            return true;
        }
        self.consumers
            .get(&cursor.consumer)
            .is_none_or(|consumer| consumer.lost || !consumer.pending.is_empty())
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
