//! The predictions the store keeps for tiles it predicts again in place.
//!
//! What a brush step's tiles are predicted from doesn't change while the stroke edits them, so
//! their predictions (positions and tangent frames from the level below) are kept, and each step
//! only applies the new corrections.

use std::collections::BTreeMap;

use geom::TileKey;

use crate::hash::{TileHash, TileMap};

/// The most entries [`Kept::new`] makes room for up front: the default bound's worth. A larger
/// bound grows the map as values come instead.
const PREALLOCATED: usize = 64;

/// Up to a fixed number of values, one per tile, the least recently used dropped first to make
/// room: the store's kept predictions ([`geom::Prediction`]), generic so its rules are tested on
/// plain values.
///
/// Whether a value is still right is the store's business: it removes a tile's value when the
/// tile's parents change or the tile leaves the evaluated pool. Here are only the bound and the
/// order, both deterministic: a use or an insert stamps the tile with the next tick of a
/// counter, and the oldest stamp goes first. The ticks are kept in order beside the values, so
/// finding the oldest is a lookup, not a scan.
#[derive(Debug)]
pub(crate) struct Kept<T> {
    /// The most values kept; zero keeps none.
    capacity: usize,
    /// Each tile's value, with the tick of its last use.
    values: TileMap<(T, u64)>,
    /// The tile each live tick stamps: one entry per value, oldest first.
    order: BTreeMap<u64, TileKey>,
    /// The tick the next use or insert is stamped with.
    tick: u64,
}

impl<T> Kept<T> {
    /// Nothing kept yet, room for `capacity` values (allocated up front only up to
    /// [`PREALLOCATED`] of them).
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            values: TileMap::with_capacity_and_hasher(capacity.min(PREALLOCATED), TileHash),
            order: BTreeMap::new(),
            tick: 0,
        }
    }

    /// The most values kept.
    pub(crate) const fn capacity(&self) -> usize {
        self.capacity
    }

    /// `key`'s value, if it is kept. Reading it isn't a use: call [`Kept::used`] for that.
    pub(crate) fn get(&self, key: &TileKey) -> Option<&T> {
        self.values.get(key).map(|(value, _)| value)
    }

    /// Records a use of `key`'s value (none if it isn't kept): it is the last to go now.
    pub(crate) fn used(&mut self, key: &TileKey) {
        let tick = self.next_tick();
        if let Some((_, used)) = self.values.get_mut(key) {
            self.order.remove(used);
            *used = tick;
            self.order.insert(tick, *key);
        }
    }

    /// Drops the least recently used values until `n` more fit within the bound (all of them,
    /// if `n` is the bound or more).
    pub(crate) fn make_room(&mut self, n: usize) {
        while self.values.len().saturating_add(n) > self.capacity {
            let Some((_, oldest)) = self.order.pop_first() else {
                return;
            };
            self.values.remove(&oldest);
        }
    }

    /// Keeps `value` for `key`, as its most recent use, replacing any value it had; the least
    /// recently used other value goes if there is no room. With no capacity, nothing is kept.
    pub(crate) fn insert(&mut self, key: TileKey, value: T) {
        if self.capacity == 0 {
            return;
        }
        self.remove(&key);
        self.make_room(1);
        let tick = self.next_tick();
        self.order.insert(tick, key);
        self.values.insert(key, (value, tick));
    }

    /// Drops `key`'s value, if it is kept.
    pub(crate) fn remove(&mut self, key: &TileKey) {
        if let Some((_, used)) = self.values.remove(key) {
            self.order.remove(&used);
        }
    }

    /// How many values are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// The tiles with a value, in no order.
    #[cfg(test)]
    pub(crate) fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.values.keys()
    }

    /// The tick for a use or insert now.
    fn next_tick(&mut self) -> u64 {
        let tick = self.tick;
        self.tick += 1;
        tick
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level};

    use super::*;

    /// Tile `n` of face 0 at level 8: a distinct key per `n` below 4.
    fn key(n: u16) -> TileKey {
        TileKey::displacement(FaceId::new(0), Level::new(8).unwrap(), n % 2, n / 2).unwrap()
    }

    /// The tiles `kept` holds, sorted.
    fn held(kept: &Kept<u32>) -> Vec<TileKey> {
        let mut keys: Vec<TileKey> = kept.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn the_least_recently_used_goes_to_make_room() {
        let mut kept = Kept::new(2);
        kept.insert(key(0), 10);
        kept.insert(key(1), 11);
        kept.used(&key(0));
        kept.insert(key(2), 12);
        assert_eq!(held(&kept), [key(0), key(2)], "1 was used least recently");
        assert_eq!(kept.get(&key(0)), Some(&10), "a value read back");
        assert_eq!(kept.get(&key(1)), None, "gone");
        kept.insert(key(2), 22);
        assert_eq!(held(&kept), [key(0), key(2)], "a replacement makes no room");
        assert_eq!(kept.get(&key(2)), Some(&22), "the new value");
        kept.insert(key(3), 13);
        assert_eq!(
            held(&kept),
            [key(2), key(3)],
            "0's use was older than 2's replacement"
        );
        kept.remove(&key(3));
        kept.used(&key(3));
        assert_eq!(
            held(&kept),
            [key(2)],
            "removed, and a use doesn't bring it back"
        );
    }

    #[test]
    fn making_room_drops_the_least_recently_used() {
        let mut kept = Kept::new(3);
        for n in 0..3 {
            kept.insert(key(n), u32::from(n));
        }
        kept.used(&key(0));
        kept.make_room(1);
        assert_eq!(held(&kept), [key(0), key(2)], "1 was the oldest");
        kept.make_room(1);
        assert_eq!(held(&kept), [key(0), key(2)], "room for one already");
        kept.make_room(3);
        assert!(held(&kept).is_empty(), "room for the bound: none left");
        assert_eq!(kept.order.len(), 0, "no ticks left either");
        kept.insert(key(3), 13);
        assert_eq!(held(&kept), [key(3)], "and kept again");
    }

    /// Every value has exactly one tick in the order, whatever the uses, replacements and
    /// removals.
    #[test]
    fn each_value_has_one_tick() {
        let mut kept = Kept::new(2);
        kept.insert(key(0), 10);
        kept.insert(key(0), 20);
        kept.used(&key(0));
        kept.insert(key(1), 11);
        kept.insert(key(2), 12);
        kept.remove(&key(1));
        assert_eq!(kept.len(), 1, "one value");
        assert_eq!(
            kept.order.values().copied().collect::<Vec<_>>(),
            [key(2)],
            "its one tick"
        );
    }

    #[test]
    fn no_capacity_keeps_nothing() {
        let mut kept = Kept::new(0);
        kept.insert(key(0), 10);
        assert_eq!(kept.get(&key(0)), None, "nothing kept");
        assert!(held(&kept).is_empty(), "empty");
    }
}
