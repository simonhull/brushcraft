//! The predictions the store keeps for tiles it predicts again in place.
//!
//! What a brush step's tiles are predicted from doesn't change while the stroke edits them, so
//! their predictions (positions and tangent frames from the level below) are kept, and each step
//! only applies the new corrections.

use std::collections::HashMap;

use geom::TileKey;

/// Up to a fixed number of values, one per tile, the least recently used dropped first to make
/// room: the store's kept predictions ([`geom::Prediction`]), generic so its rules are tested on
/// plain values.
///
/// Whether a value is still right is the store's business: it removes a tile's value when the
/// tile's parents change or the tile leaves the evaluated pool. Here are only the bound and the
/// order, both deterministic: a use or an insert stamps the tile with the next tick of a
/// counter, and the oldest stamp goes first.
#[derive(Debug)]
pub(crate) struct Kept<T> {
    /// The most values kept; zero keeps none.
    capacity: usize,
    /// Each tile's value, with the tick of its last use.
    values: HashMap<TileKey, (T, u64)>,
    /// The tick the next use or insert is stamped with.
    tick: u64,
}

impl<T> Kept<T> {
    /// Nothing kept yet, room for `capacity` values.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            values: HashMap::with_capacity(capacity),
            tick: 0,
        }
    }

    /// `key`'s value, if it is kept. Reading it isn't a use: call [`Kept::used`] for that.
    pub(crate) fn get(&self, key: &TileKey) -> Option<&T> {
        self.values.get(key).map(|(value, _)| value)
    }

    /// Records a use of `key`'s value (none if it isn't kept): it is the last to go now.
    pub(crate) fn used(&mut self, key: &TileKey) {
        let tick = self.next_tick();
        if let Some((_, used)) = self.values.get_mut(key) {
            *used = tick;
        }
    }

    /// Keeps `value` for `key`, as its most recent use, replacing any value it had; the least
    /// recently used other value goes if there is no room. With no capacity, nothing is kept.
    pub(crate) fn insert(&mut self, key: TileKey, value: T) {
        if self.capacity == 0 {
            return;
        }
        if !self.values.contains_key(&key) && self.values.len() >= self.capacity {
            let oldest = self
                .values
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(oldest, _)| *oldest);
            if let Some(oldest) = oldest {
                self.values.remove(&oldest);
            }
        }
        let tick = self.next_tick();
        self.values.insert(key, (value, tick));
    }

    /// Drops `key`'s value, if it is kept.
    pub(crate) fn remove(&mut self, key: &TileKey) {
        self.values.remove(key);
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
    fn no_capacity_keeps_nothing() {
        let mut kept = Kept::new(0);
        kept.insert(key(0), 10);
        assert_eq!(kept.get(&key(0)), None, "nothing kept");
        assert!(held(&kept).is_empty(), "empty");
    }
}
