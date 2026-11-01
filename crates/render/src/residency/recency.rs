//! Which resident tiles were drawn least recently, per size class: the order eviction scans in.
//!
//! The order is every resident tile outside the floor by the frame it was last drawn in (or
//! arrived in), then by key. A frame's use ([`Recency::renew`]) moves every tile it drew to its
//! frame. Moving each one costs a few searches of trees of thousands, every frame, for tiles
//! that are mostly the ones drawn the frame before: so the tiles the latest use drew are
//! **held** apart, their frame the latest use's ([`Recency::held_at`]), and a use moves only the
//! tiles that start or stop being drawn. Telling which those are is one walk over the sorted
//! drawn tiles beside the held ones, with no searches. [`Recency::oldest`] merges the held tiles
//! back in at their frame, so the order eviction sees is the same as if every tile were moved.
//!
//! A tile's class here is its level's ([`SizeClass::of`]): every slot residency takes for a
//! tile is of that class.

use std::collections::{BTreeMap, BTreeSet};
use std::iter::Peekable;
use std::ops::Bound;

use geom::TileKey;

use super::{CLASSES, Entry, SizeClass};

/// The resident tiles outside the floor, by the frame each was last drawn in: see the module
/// notes.
#[derive(Debug, Clone)]
pub(super) struct Recency {
    /// Per class, the tiles not held, by the frame each was last drawn in (its
    /// [`Entry::last_drawn`]), then key.
    stamped: Vec<BTreeSet<(u64, TileKey)>>,
    /// Per class, the tiles the latest use drew: last drawn in `held_at` (their
    /// [`Entry::last_drawn`] is set when they stop being held).
    held: Vec<BTreeSet<TileKey>>,
    /// The frame of the latest use.
    held_at: u64,
    /// The tiles the latest use drew that aren't held: floor tiles, and tiles not resident
    /// then, until one arrives. A later use that draws them again has nothing to move for them.
    unheld: BTreeSet<TileKey>,
}

impl Default for Recency {
    fn default() -> Self {
        Self {
            stamped: vec![BTreeSet::new(); CLASSES],
            held: vec![BTreeSet::new(); CLASSES],
            held_at: 0,
            unheld: BTreeSet::new(),
        }
    }
}

/// The index of `key`'s class.
fn class_of(key: &TileKey) -> usize {
    usize::from(SizeClass::of(key.level()).get())
}

impl Recency {
    /// Notes that `key` arrived in `frame`: last drawn then, unless it is a floor tile
    /// (`floor`), which is never in the order.
    pub(super) fn arrived(&mut self, key: TileKey, frame: u64, floor: bool) {
        // Not resident when the latest use drew it: drawn again, it moves now.
        self.unheld.remove(&key);
        if !floor && let Some(stamped) = self.stamped.get_mut(class_of(&key)) {
            stamped.insert((frame, key));
        }
    }

    /// Notes that `key`, last drawn in `last_drawn` unless it is held, left.
    pub(super) fn left(&mut self, key: &TileKey, last_drawn: u64) {
        let class = class_of(key);
        let was_held = self
            .held
            .get_mut(class)
            .is_some_and(|held| held.remove(key));
        if !was_held && let Some(stamped) = self.stamped.get_mut(class) {
            stamped.remove(&(last_drawn, *key));
        }
    }

    /// A frame's use: every resident tile of `drawn` (in `entries`) was last drawn in `frame`.
    /// The tiles it held and `drawn` doesn't have are stamped with the frame they were held
    /// at; those `drawn` has that it didn't hold are looked up, and held if resident outside
    /// the floor.
    pub(super) fn renew(
        &mut self,
        drawn: &BTreeSet<TileKey>,
        frame: u64,
        entries: &mut BTreeMap<TileKey, Entry>,
    ) {
        let walk = self.walk(drawn);
        let held_at = self.held_at;
        for key in walk.released {
            let class = class_of(&key);
            if let Some(held) = self.held.get_mut(class) {
                held.remove(&key);
            }
            if let Some(stamped) = self.stamped.get_mut(class) {
                stamped.insert((held_at, key));
            }
            if let Some(entry) = entries.get_mut(&key) {
                entry.last_drawn = held_at;
            }
        }
        for key in &walk.forgotten {
            self.unheld.remove(key);
        }
        // A tile taken keeps its entry's frame: held, it is never used, and it is set when the
        // tile is released.
        for key in walk.taken {
            let class = class_of(&key);
            let Some(entry) = entries.get(&key).filter(|entry| !entry.floor) else {
                self.unheld.insert(key);
                continue;
            };
            if let Some(stamped) = self.stamped.get_mut(class) {
                stamped.remove(&(entry.last_drawn, key));
            }
            if let Some(held) = self.held.get_mut(class) {
                held.insert(key);
            }
        }
        self.held_at = frame;
    }

    /// Walks `drawn` beside the held and unheld tiles, all in key order: which held tiles it
    /// doesn't have, which unheld ones it doesn't have, and which of its tiles are neither.
    fn walk(&self, drawn: &BTreeSet<TileKey>) -> Walk {
        let cursors = self.held.iter().map(|class| class.iter().peekable());
        let mut held: Vec<Peekable<_>> = cursors.collect();
        let mut unheld = self.unheld.iter().peekable();
        let mut walk = Walk::default();
        for key in drawn {
            if let Some(cursor) = held.get_mut(class_of(key)) {
                while let Some(passed) = cursor.next_if(|before| *before < key) {
                    walk.released.push(*passed);
                }
                if cursor.next_if_eq(&key).is_some() {
                    continue;
                }
            }
            while let Some(passed) = unheld.next_if(|before| *before < key) {
                walk.forgotten.push(*passed);
            }
            if unheld.next_if_eq(&key).is_none() {
                walk.taken.push(*key);
            }
        }
        walk.released.extend(held.into_iter().flatten());
        walk.forgotten.extend(unheld);
        walk
    }

    /// The tiles of class `class` in the order they leave, least recently drawn first, each
    /// with the frame it was last drawn in: those after `after`, or all.
    pub(super) fn oldest(
        &self,
        class: usize,
        after: Option<(u64, TileKey)>,
    ) -> impl Iterator<Item = (u64, TileKey)> + '_ {
        let from = after.map_or(Bound::Unbounded, Bound::Excluded);
        let stamped = self.stamped.get(class).into_iter();
        let stamped = stamped.flat_map(move |stamped| stamped.range((from, Bound::Unbounded)));
        let mut stamped = stamped.copied().peekable();
        let held_at = self.held_at;
        // The held tiles all sit at `held_at`: after `after` all of them, those past its key, or
        // none.
        let held_from = match after {
            Some((stamp, key)) if stamp == held_at => Some(Bound::Excluded(key)),
            Some((stamp, _)) if stamp > held_at => None,
            Some(_) | None => Some(Bound::Unbounded),
        };
        let held = held_from.into_iter().flat_map(move |start| {
            let of_class = self.held.get(class).into_iter();
            of_class.flat_map(move |held| held.range((start, Bound::Unbounded)))
        });
        let mut held = held.map(move |&key| (held_at, key)).peekable();
        std::iter::from_fn(move || match (stamped.peek(), held.peek()) {
            (Some(older), Some(newer)) if older > newer => held.next(),
            (Some(_), _) => stamped.next(),
            (None, _) => held.next(),
        })
    }
}

/// What [`Recency::walk`] found.
#[derive(Debug, Default)]
struct Walk {
    /// Held tiles not drawn now.
    released: Vec<TileKey>,
    /// Unheld tiles not drawn now.
    forgotten: Vec<TileKey>,
    /// Drawn tiles neither held nor unheld: to look up.
    taken: Vec<TileKey>,
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level};

    use super::*;
    use crate::cull::tests::Rng;
    use crate::residency::Slot;

    /// The tiles of the tests: two faces, levels 3 to 8 (four classes), a few of each; the floor
    /// is levels 3 and 4.
    fn universe() -> Vec<TileKey> {
        let levels = (3..=8).filter_map(|l| Level::new(l).ok());
        let keys = levels.flat_map(|level| {
            let side = u16::try_from(level.tiles_per_side().min(3)).expect("small");
            (0..2).flat_map(move |face| {
                (0..side).flat_map(move |ty| {
                    (0..side).filter_map(move |tx| {
                        TileKey::display(FaceId::new(face), level, tx, ty).ok()
                    })
                })
            })
        });
        keys.collect()
    }

    /// Whether `key` is a floor tile of the tests.
    fn floor(key: &TileKey) -> bool {
        key.level().get() <= 4
    }

    /// A resident tile's entry, arrived in `frame`.
    fn entry(key: &TileKey, frame: u64) -> Entry {
        Entry {
            slot: Slot::for_tests(SizeClass::of(key.level()), 0),
            last_drawn: frame,
            floor: floor(key),
            version: 0,
            stored_error: 0.0,
            loaded_error: 0.0,
            loaded_after: None,
            bounded: 0,
        }
    }

    /// The order made from scratch: per class, every resident tile outside the floor by the
    /// frame the reference says it was last drawn in, then key.
    fn from_scratch(reference: &BTreeMap<TileKey, u64>, class: usize) -> Vec<(u64, TileKey)> {
        let of_class = reference.iter().filter(|(key, _)| class_of(key) == class);
        let order: BTreeSet<(u64, TileKey)> = of_class.map(|(key, &frame)| (frame, *key)).collect();
        order.into_iter().collect()
    }

    #[test]
    fn the_order_kept_by_moving_only_tiles_that_start_or_stop_being_drawn_is_the_one_made_from_scratch()
     {
        let keys = universe();
        let mut rng = Rng(7);
        let mut recency = Recency::default();
        let mut entries: BTreeMap<TileKey, Entry> = BTreeMap::new();
        // The tiles outside the floor by the frame each was last drawn in, every drawn tile
        // renewed one at a time, as residency once did.
        let mut reference: BTreeMap<TileKey, u64> = BTreeMap::new();
        let mut drawn: BTreeSet<TileKey> = BTreeSet::new();
        let mut frame = 0;
        // The last tile outside the floor to leave, with its frame.
        let mut gone: Option<(u64, TileKey)> = None;
        for step in 0..20_000 {
            match rng.below(10) {
                // A new frame.
                0 | 1 => frame += 1,
                // A tile arrives.
                2 | 3 => {
                    let key = keys[rng.below(keys.len())];
                    if let std::collections::btree_map::Entry::Vacant(vacant) = entries.entry(key) {
                        vacant.insert(entry(&key, frame));
                        recency.arrived(key, frame, floor(&key));
                        if !floor(&key) {
                            reference.insert(key, frame);
                        }
                    }
                }
                // A tile leaves (the floor too, as after a lost frame).
                4 => {
                    let key = keys[rng.below(keys.len())];
                    if let Some(entry) = entries.remove(&key) {
                        recency.left(&key, entry.last_drawn);
                        // As an evicted victim: where a scan resumes from, no longer there.
                        gone = reference.remove(&key).map(|last| (last, key)).or(gone);
                    }
                }
                // A frame's use: mostly the last one's drawn tiles, some gone, some new, resident
                // or not; now and then none.
                _ => {
                    drawn.retain(|_| rng.below(8) != 0);
                    for _ in 0..rng.below(6) {
                        drawn.insert(keys[rng.below(keys.len())]);
                    }
                    if rng.below(20) == 0 {
                        drawn.clear();
                    }
                    recency.renew(&drawn, frame, &mut entries);
                    let renewed = drawn.iter().filter(|key| reference.contains_key(key));
                    let renewed: Vec<TileKey> = renewed.copied().collect();
                    for key in renewed {
                        reference.insert(key, frame);
                    }
                }
            }
            for class in 0..CLASSES {
                let kept: Vec<(u64, TileKey)> = recency.oldest(class, None).collect();
                let scratch = from_scratch(&reference, class);
                assert_eq!(kept, scratch, "step {step}: class {class}'s order");
                // As an eviction's scan resumes: from a tile still in the order, from the last
                // to leave, and from any tile of the universe at a recent frame (the latest
                // use's often), there or not.
                let present = scratch.get(rng.below(scratch.len().max(1))).copied();
                let stamp = frame.saturating_sub(rng.next() % 3);
                let any = (stamp, keys[rng.below(keys.len())]);
                let resumes = [present, gone, Some(any)];
                for after in resumes.into_iter().flatten() {
                    let rest: Vec<(u64, TileKey)> = recency.oldest(class, Some(after)).collect();
                    let past = scratch.iter().filter(|&&tile| tile > after);
                    let past: Vec<(u64, TileKey)> = past.copied().collect();
                    assert_eq!(
                        rest, past,
                        "step {step}: class {class}'s order after {after:?}"
                    );
                }
            }
        }
    }
}
