//! The queued evaluation: finest tiles a caller wants evaluated, by priority, and what the job
//! made of them, waiting to be taken.
//!
//! A renderer asks for the finest tiles it wants drawn and lacks, each with its screen error
//! ([`TileStore::queue_evaluation`](crate::TileStore::queue_evaluation)); the store evaluates
//! them in slices on its owner thread, highest priority first, beside the display jobs
//! ([`TileStore::run_jobs`](crate::TileStore::run_jobs)); and the caller takes what is finished
//! ([`TileStore::take_evaluated`](crate::TileStore::take_evaluated)). A finished tile is handed
//! over as a copy ([`EvaluatedTile`]: its positions with the ring, and its normals), so nothing
//! the cache does between the slice that made it and the caller's read can take it away; the
//! copies waiting to be taken are held to a share of the budget
//! ([`StoreConfig::evaluation_bytes`](crate::StoreConfig::evaluation_bytes)).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet};

use geom::{GatheredBlock, TileKey, Vec3};

use crate::Evaluation;
use crate::jobs::ConeSteps;

/// How much a caller wants a tile evaluated: higher first.
///
/// Ordered as [`f32::total_cmp`] orders it, so every value has its place; a NaN priority is
/// taken as the lowest there is (negative infinity), so it never jumps the queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Priority(f32);

impl Priority {
    /// `value`, a NaN taken as negative infinity.
    pub(crate) const fn new(value: f32) -> Self {
        Self(if value.is_nan() {
            f32::NEG_INFINITY
        } else {
            value
        })
    }

    /// The value.
    pub(crate) const fn get(self) -> f32 {
        self.0
    }
}

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A finest tile the queued evaluation finished: its evaluated positions with the one-vertex
/// ring, and its unit normals unless it was queued for its positions alone, copied out of the
/// store's cache.
///
/// Handed over by [`TileStore::take_evaluated`](crate::TileStore::take_evaluated) only while
/// it is still the tile as the store holds it: a write whose evaluations reach the tile or any
/// tile its ring reads outdates the copy, and the job copies it again (or evaluates it again)
/// before handing it over. Writes elsewhere leave it alone, so tiles away from a brush stroke
/// are handed over during the stroke.
///
/// Once taken, a tile can change without its own corrections being written: a write in a
/// neighbour moves its ring, and with it the normals of its border vertices. The stale-tile
/// feed says so ([`TileStore::drain_stale`](crate::TileStore::drain_stale)): an epoch names
/// every evaluated tile whose positions moved, and a tile whose ring reads one of them has
/// normals to compute again.
///
/// The normals are the store's CPU normals. A renderer that computes them on the GPU from the
/// ring queues its tiles for their positions alone
/// ([`TileStore::queue_evaluation_as`](crate::TileStore::queue_evaluation_as)), and gets none.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedTile {
    /// The tile.
    key: TileKey,
    /// The priority it was queued with.
    priority: f32,
    /// Its positions with the ring and the base-vertex rings (see
    /// [`TileStore::positions_with_ring`](crate::TileStore::positions_with_ring)).
    ring: GatheredBlock,
    /// Its unit normals, row-major `side²`, as [`TileStore::normals`](crate::TileStore::normals)
    /// gives them; empty for a tile queued for its positions alone.
    normals: Vec<Vec3>,
    /// What it was evaluated for: its positions alone, or its normals too.
    what: Evaluation,
    /// The tiles its copy was read from (its own and its ring's, [`TileStore::normal_inputs`]
    /// in key order): a write whose evaluations reach one outdates the copy.
    ///
    /// [`TileStore::normal_inputs`]: crate::TileStore::normal_inputs
    pub(crate) inputs: Vec<TileKey>,
    /// Whether the copy is still the tile as the store holds it: no write has reached its
    /// inputs since it was made.
    pub(crate) current: bool,
}

impl EvaluatedTile {
    /// The most bytes one holds: a finest tile's positions with the ring (67²) and its normals
    /// (65²) on the heap, and itself (its four base-vertex rings are held inline, at the most
    /// points a ring has): what the budget's share is counted in.
    pub(crate) const BYTES: usize = {
        let side = geom::TILE_QUADS as usize + 1;
        let padded = side + 2;
        (padded * padded + side * side) * size_of::<Vec3>() + size_of::<Self>()
    };

    /// A finished tile with its normals, current, copied from `inputs`.
    pub(crate) const fn new(
        key: TileKey,
        priority: f32,
        ring: GatheredBlock,
        normals: Vec<Vec3>,
        inputs: Vec<TileKey>,
    ) -> Self {
        Self {
            key,
            priority,
            ring,
            normals,
            what: Evaluation::Normals,
            inputs,
            current: true,
        }
    }

    /// A finished tile of its positions alone, current, copied from `inputs`.
    pub(crate) const fn positions(
        key: TileKey,
        priority: f32,
        ring: GatheredBlock,
        inputs: Vec<TileKey>,
    ) -> Self {
        Self {
            key,
            priority,
            ring,
            normals: Vec::new(),
            what: Evaluation::Positions,
            inputs,
            current: true,
        }
    }

    /// What it was evaluated for: [`Evaluation::Positions`] for a tile queued for its positions
    /// alone ([`TileStore::queue_evaluation_as`](crate::TileStore::queue_evaluation_as)), whose
    /// normals are empty; else [`Evaluation::Normals`].
    #[must_use]
    pub const fn what(&self) -> Evaluation {
        self.what
    }

    /// The tile.
    #[must_use]
    pub const fn key(&self) -> TileKey {
        self.key
    }

    /// The priority it was queued with (the latest, if it was queued again).
    #[must_use]
    pub const fn priority(&self) -> f32 {
        self.priority
    }

    /// Its evaluated positions with the one-vertex ring and the base-vertex rings, in
    /// [`GatheredBlock`]'s layout: what a GPU computes the tile's normals from.
    #[must_use]
    pub const fn ring(&self) -> &GatheredBlock {
        &self.ring
    }

    /// Its evaluated positions, row by row: row `j` of the tile is `side` positions, the ring
    /// left out.
    pub fn rows(&self) -> impl Iterator<Item = &[Vec3]> {
        let side = usize::try_from(self.key.side()).unwrap_or(0);
        let rows = 0..i64::from(self.key.side());
        rows.map(move |j| self.ring.row(j).get(1..=side).unwrap_or_default())
    }

    /// Its unit normals, row-major `side²`: the store's, bit for bit. Empty for a tile queued
    /// for its positions alone.
    #[must_use]
    pub fn normals(&self) -> &[Vec3] {
        &self.normals
    }

    /// Its positions with the ring, and its normals, taken apart.
    #[must_use]
    pub fn into_parts(self) -> (GatheredBlock, Vec<Vec3>) {
        (self.ring, self.normals)
    }
}

/// How many entries the queued evaluation holds, from the bytes the budget sets aside for it
/// ([`StoreConfig::evaluation_bytes`](crate::StoreConfig::evaluation_bytes)).
///
/// Nearly all of the share goes to the finished tiles waiting to be taken, each counted at its
/// largest ([`EvaluatedTile::BYTES`]); a sixteenth to the keys queued. Each holds at least what
/// keeps the job going (a round of tiles; a frame's requests), which a tiny budget may take
/// past its share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvaluationCaps {
    /// Finished tiles waiting to be taken.
    pub(crate) finished: usize,
    /// Keys queued.
    pub(crate) queued: usize,
}

impl EvaluationCaps {
    /// The fewest finished tiles held: one round of the job's.
    pub(crate) const MIN_FINISHED: usize = crate::jobs::ROUND_MAX;
    /// The fewest keys queued: a frame's requests, many times over.
    pub(crate) const MIN_QUEUED: usize = 1024;
    /// One queued key: its entry by key and its entry by priority, with the trees' overhead.
    pub(crate) const QUEUED_BYTES: usize = 64;

    /// The caps that `bytes` hold (each at least its minimum).
    pub(crate) const fn of(bytes: usize) -> Self {
        let finished = bytes / 16 * 15 / EvaluatedTile::BYTES;
        let queued = bytes / 16 / Self::QUEUED_BYTES;
        Self {
            finished: if finished > Self::MIN_FINISHED {
                finished
            } else {
                Self::MIN_FINISHED
            },
            queued: if queued > Self::MIN_QUEUED {
                queued
            } else {
                Self::MIN_QUEUED
            },
        }
    }
}

/// A wanted tile's place in the queue: the tiles queued for the first time before those queued
/// again after a write outdated their copy, then the highest priority first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Rank {
    /// Whether the tile was queued again after a write outdated its copy.
    again: bool,
    /// Its priority, highest first.
    priority: Reverse<Priority>,
}

impl Rank {
    /// The place of a tile queued `again` or not, at `priority`.
    const fn new(again: bool, priority: Priority) -> Self {
        Self {
            again,
            priority: Reverse(priority),
        }
    }
}

/// The queued evaluation's state: the tiles wanted, the cone being stepped through for the
/// first of them, and the finished tiles waiting to be taken.
#[derive(Debug)]
pub(crate) struct EvaluationQueue {
    /// The tiles wanted and not finished, with their places in the queue.
    wanted: BTreeMap<TileKey, Rank>,
    /// Those of them wanted for their positions alone, without normals.
    positions_only: BTreeSet<TileKey>,
    /// The same, in the order they are evaluated: first by [`Rank`], then by key.
    order: BTreeSet<(Rank, TileKey)>,
    /// The cone of the tiles being evaluated (see `TileStore::step_cone`).
    pub(crate) steps: ConeSteps,
    /// The finished tiles, in the order they finished, waiting to be taken.
    pub(crate) finished: Vec<EvaluatedTile>,
    /// How many it holds.
    pub(crate) caps: EvaluationCaps,
}

impl EvaluationQueue {
    /// An empty queue holding at most `caps`.
    pub(crate) const fn new(caps: EvaluationCaps) -> Self {
        Self {
            wanted: BTreeMap::new(),
            positions_only: BTreeSet::new(),
            order: BTreeSet::new(),
            steps: ConeSteps::new(),
            finished: Vec::new(),
            caps,
        }
    }

    /// Wants `key` at `priority`: queued, or its priority changed if it is already. A tile
    /// finished and waiting to be taken is queued anew (its copy stays until taken).
    pub(crate) fn want(&mut self, key: TileKey, priority: Priority) {
        self.rank(key, Rank::new(false, priority));
    }

    /// [`EvaluationQueue::want`], for `what`: its positions alone, or its normals too (the
    /// latest call says which).
    pub(crate) fn want_as(&mut self, key: TileKey, priority: Priority, what: Evaluation) {
        self.want(key, priority);
        self.set_what(key, what);
    }

    /// Notes what `key` is wanted for.
    fn set_what(&mut self, key: TileKey, what: Evaluation) {
        match what {
            Evaluation::Positions => self.positions_only.insert(key),
            Evaluation::Normals => self.positions_only.remove(&key),
        };
    }

    /// What `key` is wanted for: its positions alone, or its normals too (the default).
    pub(crate) fn what(&self, key: &TileKey) -> Evaluation {
        if self.positions_only.contains(key) {
            Evaluation::Positions
        } else {
            Evaluation::Normals
        }
    }

    /// Wants `key` again at `priority`, for `what`, a write having outdated its copy: behind
    /// every tile queued for the first time, so a tile a brush keeps writing under can't keep
    /// the job from the rest. A tile queued already keeps its place.
    pub(crate) fn want_again(&mut self, key: TileKey, priority: Priority, what: Evaluation) {
        if !self.wanted.contains_key(&key) {
            self.rank(key, Rank::new(true, priority));
            self.set_what(key, what);
        }
    }

    /// Puts `key` in the queue at `rank`, out of any place it had.
    fn rank(&mut self, key: TileKey, rank: Rank) {
        if let Some(old) = self.wanted.insert(key, rank) {
            self.order.remove(&(old, key));
        }
        self.order.insert((rank, key));
    }

    /// Drops the lowest-priority keys past the cap; returns how many. A cone being stepped
    /// through for a key dropped is dropped too.
    pub(crate) fn trim(&mut self) -> usize {
        let mut dropped = 0;
        while self.wanted.len() > self.caps.queued {
            let Some((_, key)) = self.order.pop_last() else {
                break;
            };
            self.wanted.remove(&key);
            self.positions_only.remove(&key);
            dropped += 1;
        }
        self.drop_lost_cone();
        dropped
    }

    /// Forgets the cone being stepped through if a tile it is for is no longer wanted; returns
    /// whether it did.
    fn drop_lost_cone(&mut self) -> bool {
        let wanted = &self.wanted;
        let lost = self
            .steps
            .cone
            .as_ref()
            .is_some_and(|cone| cone.keys.iter().any(|key| !wanted.contains_key(key)));
        if lost {
            self.steps.reset();
        }
        lost
    }

    /// Keeps only the wanted tiles, and the finished ones, `keep` holds of; returns whether the
    /// tiles being evaluated lost one.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&TileKey) -> bool) -> bool {
        self.wanted.retain(|key, _| keep(key));
        let wanted = &self.wanted;
        self.order.retain(|(_, key)| wanted.contains_key(key));
        self.positions_only.retain(|key| wanted.contains_key(key));
        self.finished.retain(|tile| keep(&tile.key));
        self.drop_lost_cone()
    }

    /// The priority `key` is wanted at, if it is.
    pub(crate) fn priority(&self, key: &TileKey) -> Option<Priority> {
        self.wanted.get(key).map(|rank| rank.priority.0)
    }

    /// Marks `key` finished: no longer wanted.
    pub(crate) fn done(&mut self, key: &TileKey) {
        if let Some(rank) = self.wanted.remove(key) {
            self.order.remove(&(rank, *key));
        }
        self.positions_only.remove(key);
    }

    /// Keeps `tile` for the caller, in place of a copy of the same tile not taken yet (a tile
    /// queued again after it finished).
    pub(crate) fn keep(&mut self, tile: EvaluatedTile) {
        self.finished.retain(|older| older.key != tile.key);
        self.finished.push(tile);
    }

    /// Outdates the finished tiles `reached` holds of (by their inputs): a write reached them.
    pub(crate) fn outdate(&mut self, mut reached: impl FnMut(&[TileKey]) -> bool) {
        let current = self.finished.iter_mut().filter(|tile| tile.current);
        for tile in current {
            if reached(&tile.inputs) {
                tile.current = false;
            }
        }
    }

    /// The `n` wanted tiles of highest priority, highest first.
    pub(crate) fn first(&self, n: usize) -> Vec<TileKey> {
        let keys = self.order.iter().take(n).map(|&(_, key)| key);
        keys.collect()
    }

    /// How many tiles are wanted and not finished.
    pub(crate) fn len(&self) -> usize {
        self.wanted.len()
    }

    /// How many more finished tiles fit before the caller takes some.
    pub(crate) fn room(&self) -> usize {
        self.caps.finished.saturating_sub(self.finished.len())
    }

    /// Whether a slice has anything to do: tiles wanted with room to finish them, or finished
    /// tiles a write outdated, to copy again.
    pub(crate) fn has_work(&self) -> bool {
        (!self.wanted.is_empty() && self.room() > 0) || self.finished.iter().any(|t| !t.current)
    }

    /// Forgets every wanted tile and the cone; the finished tiles stay, to be taken.
    pub(crate) fn clear(&mut self) {
        self.wanted.clear();
        self.positions_only.clear();
        self.order.clear();
        self.steps.reset();
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level};

    use super::*;

    /// Level-8 tile `n` of face 0.
    fn key(n: u16) -> TileKey {
        TileKey::displacement(FaceId::new(0), Level::new(8).unwrap(), n % 4, n / 4).unwrap()
    }

    #[test]
    fn priorities_order_highest_first_and_nan_last() {
        let mut queue = EvaluationQueue::new(EvaluationCaps::of(0));
        queue.want(key(0), Priority::new(1.0));
        queue.want(key(1), Priority::new(f32::NAN));
        queue.want(key(2), Priority::new(5.0));
        queue.want(key(3), Priority::new(f32::NEG_INFINITY));
        queue.want(key(4), Priority::new(3.0));
        assert_eq!(queue.first(3), [key(2), key(4), key(0)], "highest first");
        queue.want(key(0), Priority::new(9.0));
        assert_eq!(
            queue.first(1),
            [key(0)],
            "queued again, at its new priority"
        );
        assert_eq!(queue.len(), 5, "once each");
        assert_eq!(
            queue.first(5)[3..],
            [key(1), key(3)],
            "NaN as negative infinity, the tie in key order"
        );
        assert_eq!(
            queue.priority(&key(1)).map(Priority::get),
            Some(f32::NEG_INFINITY),
            "NaN taken as negative infinity"
        );
        assert_eq!(Priority::new(2.0), Priority::new(2.0), "equal values");
        assert_ne!(Priority::new(2.0), Priority::new(3.0), "different values");
        assert_eq!(
            Priority::new(f32::NAN),
            Priority::new(f32::NEG_INFINITY),
            "NaN is negative infinity"
        );
    }

    #[test]
    fn past_the_cap_the_lowest_priorities_go() {
        let caps = EvaluationCaps {
            finished: 1,
            queued: 3,
        };
        let mut queue = EvaluationQueue::new(caps);
        for n in 0..5 {
            queue.want(key(n), Priority::new(f32::from(n)));
        }
        assert_eq!(queue.trim(), 2, "two past the cap");
        assert_eq!(queue.first(5), [key(4), key(3), key(2)], "the highest stay");
        queue.done(&key(3));
        assert_eq!(queue.first(5), [key(4), key(2)], "a finished tile leaves");
        assert!(queue.has_work(), "tiles wanted, room to finish them");
        queue.retain(|k| *k == key(2));
        assert_eq!(queue.first(5), [key(2)], "withdrawn");
        queue.clear();
        assert_eq!(queue.len(), 0, "cleared");
        assert!(!queue.has_work(), "nothing to do");
    }

    #[test]
    fn the_share_holds_finished_tiles_and_keys_each_at_least_its_minimum() {
        assert_eq!(
            EvaluationCaps::of(0),
            EvaluationCaps {
                finished: EvaluationCaps::MIN_FINISHED,
                queued: EvaluationCaps::MIN_QUEUED,
            },
            "no share: the least"
        );
        let share = 64 << 20;
        let caps = EvaluationCaps::of(share);
        assert_eq!(
            caps.finished,
            share / 16 * 15 / EvaluatedTile::BYTES,
            "fifteen sixteenths to the finished tiles"
        );
        assert_eq!(
            caps.queued,
            share / 16 / EvaluationCaps::QUEUED_BYTES,
            "a sixteenth to the keys"
        );
        assert!(
            caps.finished * EvaluatedTile::BYTES + caps.queued * EvaluationCaps::QUEUED_BYTES
                <= share,
            "within the share"
        );
        let least = 67 * 67 * 12 + 65 * 65 * 12 + 4 * 2 * 32 * 12;
        assert!(
            std::hint::black_box(EvaluatedTile::BYTES) > least,
            "a finished tile is counted at its positions, ring, normals and corner rings at least"
        );
        assert!(
            std::hint::black_box(EvaluatedTile::BYTES) < least + 4096,
            "and at little more"
        );
    }
}
