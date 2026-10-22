//! The change feed: the stream of tiles edits make dirty, read by several independent consumers
//! (save now; GPU upload, undo and the live link later), each at its own pace.

use std::collections::{HashMap, VecDeque};

use geom::TileKey;

use crate::OpId;

/// A subscriber's handle. Ids are never reused, so a stale one reads nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConsumerId(usize);

/// Dirty tiles, in edit order, with a cursor per consumer.
///
/// Consumers subscribe at the feed's current end. Each [`ChangeFeed::drain`] returns every key
/// pushed since that consumer's last drain, once, with its latest op. Entries every consumer has
/// passed are dropped, so memory is bounded by the slowest consumer's lag.
///
/// A push that repeats the key and op of an entry no consumer has drained yet is skipped: every
/// consumer would read the two as one anyway. Once any consumer has drained the entry, the push
/// is kept, so that consumer sees the key again.
#[derive(Debug, Default)]
pub struct ChangeFeed {
    /// The pushed keys with their ops, oldest first, from the slowest consumer's cursor on.
    entries: VecDeque<(TileKey, OpId)>,
    /// Sequence number of `entries[0]`.
    first: u64,
    /// Per consumer ever subscribed: the sequence number it reads next (`None`: unsubscribed).
    cursors: Vec<Option<u64>>,
    /// Per key in `entries`: the op and sequence number of its latest entry, for skipping
    /// repeated pushes.
    last: HashMap<TileKey, (OpId, u64)>,
}

impl ChangeFeed {
    /// An empty feed with no consumers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that `key` was changed by `op`. With no consumers, nothing is kept.
    ///
    /// Skipped if `key`'s latest entry has the same `op` and is ahead of every consumer's cursor:
    /// no consumer has drained it, so each one's next drain reports the key with `op` either way.
    pub fn push(&mut self, key: TileKey, op: OpId) {
        let Some(newest) = self.cursors.iter().flatten().max().copied() else {
            self.first += 1;
            return;
        };
        if self
            .last
            .get(&key)
            .is_some_and(|&(last_op, seq)| last_op == op && seq >= newest)
        {
            return;
        }
        self.last.insert(key, (op, self.end()));
        self.entries.push_back((key, op));
    }

    /// A new consumer, starting at the feed's current end.
    pub fn subscribe(&mut self) -> ConsumerId {
        self.cursors.push(Some(self.end()));
        ConsumerId(self.cursors.len() - 1)
    }

    /// Removes a consumer; entries only it was holding are dropped.
    pub fn unsubscribe(&mut self, id: ConsumerId) {
        if let Some(cursor) = self.cursors.get_mut(id.0) {
            *cursor = None;
        }
        self.trim();
    }

    /// Every key changed since `id`'s last drain, once each with its latest op, in order of first
    /// appearance. Empty for an unknown or unsubscribed id.
    pub fn drain(&mut self, id: ConsumerId) -> Vec<(TileKey, OpId)> {
        let end = self.end();
        let Some(Some(cursor)) = self.cursors.get_mut(id.0) else {
            return Vec::new();
        };
        let skip = usize::try_from(*cursor - self.first).unwrap_or(usize::MAX);
        *cursor = end;
        let mut order: Vec<(TileKey, OpId)> = Vec::new();
        let mut at: HashMap<TileKey, usize> = HashMap::new();
        for &(key, op) in self.entries.iter().skip(skip) {
            if let Some(slot) = at.get(&key).and_then(|&i| order.get_mut(i)) {
                slot.1 = op;
            } else {
                at.insert(key, order.len());
                order.push((key, op));
            }
        }
        self.trim();
        order
    }

    /// Entries held for the slowest consumer.
    #[must_use]
    pub fn backlog(&self) -> usize {
        self.entries.len()
    }

    /// The sequence number of the next push.
    fn end(&self) -> u64 {
        self.first + self.entries.len() as u64
    }

    /// Drops the entries every consumer has passed.
    fn trim(&mut self) {
        let oldest = self
            .cursors
            .iter()
            .flatten()
            .min()
            .copied()
            .unwrap_or_else(|| self.end());
        while self.first < oldest {
            let Some((key, _)) = self.entries.pop_front() else {
                break;
            };
            if self
                .last
                .get(&key)
                .is_some_and(|&(_, seq)| seq == self.first)
            {
                self.last.remove(&key);
            }
            self.first += 1;
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level};

    use super::*;

    fn key(face: u32) -> TileKey {
        TileKey::displacement(FaceId::new(face), Level::new(9).unwrap(), 0, 0).unwrap()
    }

    #[test]
    fn two_consumers_see_every_change_once_at_their_own_pace() {
        let mut feed = ChangeFeed::new();
        let fast = feed.subscribe();
        let slow = feed.subscribe();
        let mut x = 1u64;
        let mut pushed_since = [Vec::<TileKey>::new(), Vec::new()];
        for step in 0..2000u32 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let pushed = key(u32::try_from(x >> 59).unwrap()); // 32 distinct keys, many repeats
            feed.push(pushed, OpId::new(0, u64::from(step) + 1).unwrap());
            for p in &mut pushed_since {
                p.push(pushed);
            }
            for (slot, (id, every)) in [(fast, 3), (slow, 97)].into_iter().enumerate() {
                if step % every == 0 {
                    let got = feed.drain(id);
                    let mut want: Vec<TileKey> = Vec::new();
                    for k in &pushed_since[slot] {
                        if !want.contains(k) {
                            want.push(*k);
                        }
                    }
                    assert_eq!(
                        got.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                        want,
                        "consumer {slot} step {step}"
                    );
                    pushed_since[slot].clear();
                }
            }
        }
        assert!(
            feed.backlog() <= 97,
            "entries both consumers passed are dropped"
        );
    }

    #[test]
    fn drain_reports_the_latest_op_and_late_subscribers_start_at_the_end() {
        let mut feed = ChangeFeed::new();
        let a = feed.subscribe();
        feed.push(key(1), OpId::new(0, 1).unwrap());
        feed.push(key(1), OpId::new(0, 2).unwrap());
        let late = feed.subscribe();
        assert_eq!(
            feed.drain(a),
            vec![(key(1), OpId::new(0, 2).unwrap())],
            "latest op, once"
        );
        assert!(feed.drain(late).is_empty(), "nothing before subscribing");
        feed.unsubscribe(late);
        feed.push(key(2), OpId::new(0, 3).unwrap());
        assert_eq!(
            feed.drain(a).len(),
            1,
            "still works after another unsubscribes"
        );
        assert!(feed.drain(late).is_empty(), "unsubscribed");
        let again = feed.subscribe();
        assert_ne!(again, late, "ids are not reused");
    }

    #[test]
    fn unsubscribing_drops_what_only_that_consumer_held() {
        let mut feed = ChangeFeed::new();
        let fast = feed.subscribe();
        let slow = feed.subscribe();
        feed.push(key(1), OpId::new(0, 1).unwrap());
        feed.push(key(2), OpId::new(0, 2).unwrap());
        assert_eq!(feed.drain(fast).len(), 2, "the fast consumer reads both");
        assert_eq!(feed.backlog(), 2, "held for the slow one");
        feed.unsubscribe(slow);
        assert_eq!(feed.backlog(), 0, "nobody left behind");
    }

    #[test]
    fn a_repeated_push_is_skipped_only_until_a_consumer_drains_it() {
        let mut feed = ChangeFeed::new();
        let a = feed.subscribe();
        let b = feed.subscribe();
        let op = OpId::new(0, 1).unwrap();
        feed.push(key(1), op);
        feed.push(key(1), op);
        assert_eq!(feed.backlog(), 1, "undrained by anyone: one entry");
        assert_eq!(feed.drain(b), vec![(key(1), op)], "b sees the first push");
        feed.push(key(1), op);
        assert_eq!(
            feed.drain(b),
            vec![(key(1), op)],
            "b drained the first, so the repeat is kept for it"
        );
        assert_eq!(
            feed.drain(a),
            vec![(key(1), op)],
            "a, which drained neither, sees the key once"
        );
        feed.push(key(1), op);
        feed.push(key(1), op);
        assert_eq!(feed.backlog(), 1, "a fresh entry dedupes again");
        assert_eq!(feed.drain(a), vec![(key(1), op)], "a sees it");
        assert_eq!(feed.drain(b), vec![(key(1), op)], "and so does b");
        assert!(feed.last.is_empty(), "drained entries leave no trace");
    }

    #[test]
    fn no_consumers_keep_nothing() {
        let mut feed = ChangeFeed::new();
        feed.push(key(1), OpId::new(0, 1).unwrap());
        assert_eq!(feed.backlog(), 0, "nobody to read it");
        let a = feed.subscribe();
        feed.push(key(2), OpId::new(0, 2).unwrap());
        assert_eq!(
            feed.drain(a),
            vec![(key(2), OpId::new(0, 2).unwrap())],
            "only what came after"
        );
        assert_eq!(feed.backlog(), 0, "drained by the only consumer");
    }
}
