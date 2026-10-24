//! [`SharedMap`]: a sorted map whose versions share their storage, so a new version costs what
//! changed, not what the map holds.
//!
//! A save point of a project file names every live tile and bound, 369,600 records in the
//! benchmark project, while a save changes a few hundred. Each save point's map is the previous
//! one with the save's records put in, and the store keeps the previous one until it moves onto
//! the new one; copying the whole map for each save made a save's cost follow the project, not
//! the edit. Here a map is a list of sorted runs, each behind its own [`Arc`]: a new version
//! copies only the runs its entries land in, and shares the others with the version it came
//! from.

use std::sync::Arc;

/// The entries in a run that a build or a split makes. A run that grows past [`MAX_RUN`] is cut
/// into runs of this many.
///
/// Small enough that copying a run an insert lands in is cheap (256 entries of a tile key and a
/// location are about 8 KB), large enough that the list of runs stays short: the benchmark's
/// 216,000 tiles are 844 runs, a list a new version copies in a few microseconds.
const RUN: usize = 256;

/// The most entries a run holds: a run that an insert takes past this is split.
const MAX_RUN: usize = 2 * RUN;

/// The runs of a [`SharedMap`]: each shared, and the list of them shared too, so a clone of
/// the map copies one pointer.
type Runs<K, V> = Arc<[Arc<[(K, V)]>]>;

/// A map from `K` to `V`, sorted by key, that is never changed in place: [`SharedMap::with`]
/// makes a new version, which shares every run it doesn't change with this one.
///
/// Entries are stored as sorted, non-empty runs, each run's keys all below the next run's. A
/// lookup searches the list of runs by their last keys, then the run. Making a version with
/// `n` entries put in costs, besides those entries, a copy of each run they land in and of the
/// list of runs (one pointer a run). Entries are only ever put in, never taken out, which is
/// all a save point needs: a save never removes a record from the state it names. The format
/// allows it (an Index entry of zero length removes its record, and loading a chain applies
/// such removals), but this crate's writer never emits one: an Index names only records
/// written, so a save point's maps only grow.
#[derive(Debug, Clone)]
pub(crate) struct SharedMap<K, V> {
    /// The runs, in key order: each sorted by key, non-empty, its keys below the next run's.
    runs: Runs<K, V>,
    /// The entries in all runs.
    len: usize,
}

impl<K, V> Default for SharedMap<K, V> {
    fn default() -> Self {
        Self {
            runs: Arc::new([]),
            len: 0,
        }
    }
}

impl<K: Ord + Copy, V: Copy> SharedMap<K, V> {
    /// A map of `entries`, which must be sorted by key with no key twice, as a
    /// [`std::collections::BTreeMap`] lists them. Runs of [`RUN`] entries.
    pub(crate) fn from_sorted(entries: impl IntoIterator<Item = (K, V)>) -> Self {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        debug_assert!(
            entries.is_sorted_by(|a, b| a.0 < b.0),
            "a SharedMap is built from entries sorted by key, each key once"
        );
        let len = entries.len();
        let runs: Vec<Arc<[(K, V)]>> = entries.chunks(RUN).map(Arc::from).collect();
        Self {
            runs: runs.into(),
            len,
        }
    }

    /// A new version of the map with `entries` put in, each replacing the entry of its key if
    /// there is one. `entries` must be sorted by key with no key twice, as a
    /// [`std::collections::BTreeMap`] lists them.
    ///
    /// The new version shares every run no entry lands in with this one. An entry lands in the
    /// first run whose last key is at or above its own, or in the last run if there is none: so
    /// runs keep their key ranges, and a map built empty grows from its last run. A run that
    /// grows past [`MAX_RUN`] entries is split into runs of [`RUN`].
    pub(crate) fn with(&self, entries: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut entries = entries.into_iter().peekable();
        let mut runs: Vec<Arc<[(K, V)]>> = Vec::with_capacity(self.runs.len() + 1);
        let mut len = self.len;
        let last = self.runs.len().saturating_sub(1);
        for (at, run) in self.runs.iter().enumerate() {
            // The last run takes every entry left: its keys may grow past its current last.
            let ceiling = if at == last {
                None
            } else {
                run.last().map(|&(key, _)| key)
            };
            let mut landed = Vec::new();
            while let Some((key, _)) = entries.peek() {
                if ceiling.is_some_and(|ceiling| *key > ceiling) {
                    break;
                }
                landed.extend(entries.next());
            }
            if landed.is_empty() {
                runs.push(Arc::clone(run));
                continue;
            }
            let merged = merge(run, &landed);
            len += merged.len() - run.len();
            push_split(&mut runs, &merged);
        }
        // Only a map with no runs leaves entries over: they are all its entries.
        let rest: Vec<(K, V)> = entries.collect();
        debug_assert!(
            rest.is_sorted_by(|a, b| a.0 < b.0),
            "a SharedMap takes entries sorted by key, each key once"
        );
        len += rest.len();
        push_split(&mut runs, &rest);
        Self {
            runs: runs.into(),
            len,
        }
    }

    /// The value of `key`, if the map has it.
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        let at = self
            .runs
            .partition_point(|run| run.last().is_some_and(|(last, _)| last < key));
        let run = self.runs.get(at)?;
        let found = run.binary_search_by(|(probe, _)| probe.cmp(key)).ok()?;
        run.get(found).map(|(_, value)| value)
    }

    /// Whether the map has `key`.
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// The entries, in key order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.runs
            .iter()
            .flat_map(|run| run.iter().map(|(key, value)| (key, value)))
    }

    /// The keys, in order.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(key, _)| key)
    }

    /// How many entries the map holds.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Whether this version and `other` hold the same entries, however their runs are cut.
    #[cfg(test)]
    pub(crate) fn same_entries(&self, other: &Self) -> bool
    where
        V: PartialEq,
    {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

/// `run` with `landed` merged in, both sorted by key with each key once; where both have a key,
/// `landed`'s value wins.
fn merge<K: Ord + Copy, V: Copy>(run: &[(K, V)], landed: &[(K, V)]) -> Vec<(K, V)> {
    let mut merged = Vec::with_capacity(run.len() + landed.len());
    let (mut old, mut new) = (run.iter().peekable(), landed.iter().peekable());
    loop {
        let next = match (old.peek(), new.peek()) {
            (Some(a), Some(b)) => match a.0.cmp(&b.0) {
                std::cmp::Ordering::Less => old.next(),
                std::cmp::Ordering::Equal => {
                    old.next();
                    new.next()
                }
                std::cmp::Ordering::Greater => new.next(),
            },
            (Some(_), None) => old.next(),
            (None, Some(_)) => new.next(),
            (None, None) => break,
        };
        merged.extend(next.copied());
    }
    debug_assert!(
        merged.is_sorted_by(|a, b| a.0 < b.0),
        "a SharedMap takes entries sorted by key, each key once"
    );
    merged
}

/// Appends `entries` to `runs`: as one run if it holds at most [`MAX_RUN`], else cut into runs
/// of [`RUN`]. Nothing for no entries, so no run is ever empty.
fn push_split<K: Copy, V: Copy>(runs: &mut Vec<Arc<[(K, V)]>>, entries: &[(K, V)]) {
    if entries.len() <= MAX_RUN {
        if !entries.is_empty() {
            runs.push(Arc::from(entries));
        }
        return;
    }
    runs.extend(entries.chunks(RUN).map(Arc::from));
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A small deterministic generator (`SplitMix64`), so failures replay.
    struct Rng(u64);

    impl Rng {
        /// The next number below `n`.
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) % n
        }
    }

    /// Whether `map` holds exactly `oracle`'s entries, finds each of them, misses keys it lacks,
    /// and keeps its runs' invariants.
    fn matches(map: &SharedMap<u32, u64>, oracle: &BTreeMap<u32, u64>) -> bool {
        let runs_ok = map
            .runs
            .iter()
            .all(|run| !run.is_empty() && run.len() <= MAX_RUN)
            && map
                .runs
                .windows(2)
                .all(|pair| pair[0].last().map(|e| e.0) < pair[1].first().map(|e| e.0));
        let lookups_ok = (0..2_000).all(|key| map.get(&key) == oracle.get(&key));
        runs_ok
            && lookups_ok
            && map.len() == oracle.len()
            && map.iter().eq(oracle.iter())
            && map.keys().eq(oracle.keys())
    }

    #[test]
    fn versions_match_a_btreemap_through_random_inserts() {
        let mut rng = Rng(7);
        for seed_len in [0_u32, 1, 255, 256, 257, 700] {
            let mut oracle: BTreeMap<u32, u64> =
                (0..seed_len).map(|k| (k * 3, u64::from(k))).collect();
            let mut map = SharedMap::from_sorted(oracle.iter().map(|(&k, &v)| (k, v)));
            assert!(matches(&map, &oracle));
            for round in 0..60 {
                let before = oracle.clone();
                let previous = map.clone();
                // Clusters of keys (as an edit's tiles are) or keys spread over the whole range.
                let count = rng.below(if round % 7 == 0 { 600 } else { 40 });
                let centre = rng.below(2_000);
                let entries: BTreeMap<u32, u64> = (0..count)
                    .map(|_| {
                        let key = if round % 2 == 0 {
                            (centre + rng.below(50)).min(1_999)
                        } else {
                            rng.below(2_000)
                        };
                        (u32::try_from(key).unwrap(), rng.below(u64::MAX))
                    })
                    .collect();
                oracle.extend(entries.iter().map(|(&k, &v)| (k, v)));
                map = map.with(entries.iter().map(|(&k, &v)| (k, v)));
                assert!(matches(&map, &oracle), "seed {seed_len}, round {round}");
                // The version it came from is untouched.
                assert!(
                    matches(&previous, &before),
                    "seed {seed_len}, round {round}"
                );
            }
        }
    }

    /// [`RUN`] as a key.
    const RUN_KEY: u32 = 256;

    #[test]
    fn a_version_shares_every_run_its_entries_miss() {
        assert_eq!(usize::try_from(RUN_KEY).unwrap(), RUN);
        let map = SharedMap::from_sorted((0..10 * RUN_KEY).map(|k| (k, u64::from(k))));
        let next = map.with([(3 * RUN_KEY + 5, 0)]);
        let shared = map
            .runs
            .iter()
            .zip(next.runs.iter())
            .filter(|(a, b)| Arc::ptr_eq(a, b))
            .count();
        assert_eq!(next.runs.len(), map.runs.len());
        assert_eq!(shared, map.runs.len() - 1);
        assert_eq!(next.get(&(3 * RUN_KEY + 5)), Some(&0));
        assert_eq!(
            map.get(&(3 * RUN_KEY + 5)),
            Some(&u64::from(3 * RUN_KEY + 5))
        );
    }

    #[test]
    fn same_entries_ignores_how_runs_are_cut() {
        let built = SharedMap::from_sorted((0..1_000_u32).map(|k| (k, 1_u64)));
        let grown = (0..1_000_u32).fold(SharedMap::default(), |map, k| map.with([(k, 1_u64)]));
        assert!(built.same_entries(&grown));
        assert!(!built.same_entries(&grown.with([(5, 2)])));
        assert!(!built.same_entries(&grown.with([(1_000, 1)])));
    }
}
