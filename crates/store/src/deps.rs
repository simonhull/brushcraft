//! What each cached evaluated tile was computed from, so an edit drops exactly what it changes.
//!
//! An evaluated tile's positions are predicted from its parent tiles (one level coarser, as
//! [`geom::parent_tiles`] lists them) and its own corrections; its normals are computed from the
//! positions of its own level's [`geom::normal_tiles`]. The graph records both, with the reverse
//! edges ("who reads this tile"), so an edit can walk from the edited tiles to every cached tile
//! whose positions or normals followed from them.
//!
//! A tile's positions depend on its whole ancestry, not only its parents, so the walk must pass
//! through tiles that are no longer cached: if a level-5 tile was evicted while the level-6
//! tiles predicted from it stay cached, an edit at level 4 still has to reach level 6. A node
//! therefore stays in the graph while its tile is resident **or** anything recorded in the graph
//! still reads it; once neither holds, it is removed, and its parents are reconsidered. The graph
//! is so bounded by the cached tiles and their ancestry.
//!
//! Level-0 tiles are the base mesh, which never changes, so they are never recorded.
//!
//! # Recording order
//!
//! A tile's positions must be recorded while its parents' nodes still hold their own parents:
//! the store records a tile before it unpins the parents it was predicted from, so no parent can
//! have left the pool and been collected in between. Should a parent's node be missing all the
//! same, the graph doesn't know that parent's ancestry: the node it makes for it is *untracked*,
//! and every edit at a coarser level is taken to reach it (see [`DepGraph::cascade`]), so the
//! graph over-invalidates rather than lose an edge.

use std::collections::{BTreeSet, HashMap};

use geom::{Level, TileKey};

/// Whether a tile of `level` is predicted from other tiles: above level 1 (level-1 tiles are
/// predicted from the base mesh).
fn from_tiles(level: Level) -> bool {
    level.parent().is_some_and(|parent| parent != Level::ZERO)
}

/// One tile's place in the graph.
#[derive(Debug, Default)]
struct Node {
    /// The tile is in the evaluated pool (fresh or awaiting recomputation in place).
    resident: bool,
    /// The parent tiles its positions were predicted from (level ≥ 1 only); `None` while they
    /// haven't been recorded (a node made for a tile only because something reads it).
    parents: Option<Vec<TileKey>>,
    /// The same-level tiles its normals were computed from; `None` while the slot holds no
    /// current normals.
    normal_inputs: Option<Vec<TileKey>>,
    /// Tiles whose positions were predicted from this one's.
    readers: BTreeSet<TileKey>,
    /// Tiles whose normals were computed from this one's positions.
    normal_readers: BTreeSet<TileKey>,
}

impl Node {
    /// Whether the node can go: its tile isn't cached and nothing recorded reads it.
    fn is_garbage(&self) -> bool {
        !self.resident && self.readers.is_empty() && self.normal_readers.is_empty()
    }
}

/// What an edit invalidates, as [`DepGraph::cascade`] finds it.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Cascade {
    /// Tiles whose positions no longer follow from their inputs: the edited tiles and everything
    /// predicted from them, transitively, cached or not.
    pub(crate) positions: BTreeSet<TileKey>,
    /// Tiles whose normals were computed from any of those positions (positions dropped implies
    /// normals dropped, so this includes every cached tile of `positions` with normals).
    pub(crate) normals: BTreeSet<TileKey>,
}

/// Recorded inputs of the cached evaluated tiles, with reverse edges.
#[derive(Debug, Default)]
pub(crate) struct DepGraph {
    /// Every tile that is resident or read by a recorded tile.
    nodes: HashMap<TileKey, Node>,
    /// Nodes above level 1 whose parents were never recorded: their ancestry is unknown, so any
    /// coarser edit reaches them. Empty unless a tile was recorded after a parent's node went.
    untracked: BTreeSet<TileKey>,
}

impl DepGraph {
    /// An empty graph.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records that `key` is resident with positions predicted from `parents` (level-0 parents
    /// are skipped). Replaces whatever was recorded for its positions before; its normals, which
    /// read the old positions, are dropped.
    pub(crate) fn record_positions<'a>(
        &mut self,
        key: TileKey,
        parents: impl IntoIterator<Item = &'a TileKey>,
    ) {
        self.drop_normals(key);
        let parents: Vec<TileKey> = parents
            .into_iter()
            .filter(|p| p.level() != Level::ZERO)
            .copied()
            .collect();
        let old = self
            .nodes
            .get_mut(&key)
            .and_then(|node| node.parents.take())
            .unwrap_or_default();
        for parent in &old {
            if let Some(node) = self.nodes.get_mut(parent) {
                node.readers.remove(&key);
            }
        }
        for &parent in &parents {
            self.node(parent).readers.insert(key);
        }
        let node = self.nodes.entry(key).or_default();
        node.resident = true;
        node.parents = Some(parents);
        self.untracked.remove(&key);
        // Parents no longer read (none, while the inputs are structural) may be garbage now.
        for parent in old {
            self.collect(parent);
        }
    }

    /// `key`'s node, made if it is missing. A node made above level 1 is untracked until its
    /// own positions are recorded.
    fn node(&mut self, key: TileKey) -> &mut Node {
        let untracked = &mut self.untracked;
        self.nodes.entry(key).or_insert_with(|| {
            if from_tiles(key.level()) {
                untracked.insert(key);
            }
            Node::default()
        })
    }

    /// Records that `key`'s slot holds normals computed from the positions of `inputs`.
    pub(crate) fn record_normals<'a>(
        &mut self,
        key: TileKey,
        inputs: impl IntoIterator<Item = &'a TileKey>,
    ) {
        self.drop_normals(key);
        let inputs: Vec<TileKey> = inputs.into_iter().copied().collect();
        for &input in &inputs {
            self.node(input).normal_readers.insert(key);
        }
        self.nodes.entry(key).or_default().normal_inputs = Some(inputs);
    }

    /// Whether `key`'s slot holds current normals.
    pub(crate) fn has_normals(&self, key: &TileKey) -> bool {
        self.nodes
            .get(key)
            .is_some_and(|node| node.normal_inputs.is_some())
    }

    /// Forgets `key`'s normals: the slot's normals are no longer current.
    pub(crate) fn drop_normals(&mut self, key: TileKey) {
        let Some(inputs) = self
            .nodes
            .get_mut(&key)
            .and_then(|node| node.normal_inputs.take())
        else {
            return;
        };
        for input in inputs {
            if let Some(node) = self.nodes.get_mut(&input) {
                node.normal_readers.remove(&key);
            }
            self.collect(input);
        }
    }

    /// Records that `key`'s tile left the evaluated pool (evicted, or freed by an invalidation).
    /// Its normals went with it; its node stays while anything recorded still reads it.
    pub(crate) fn left_pool(&mut self, key: TileKey) {
        self.drop_normals(key);
        if let Some(node) = self.nodes.get_mut(&key) {
            node.resident = false;
        }
        self.collect(key);
    }

    /// Everything an edit of the `edited` tiles' corrections invalidates: those tiles, then every
    /// tile recorded as predicted from an invalidated one (cached or not, since the walk must
    /// reach cached descendants of evicted tiles), then the normals of every tile recorded as
    /// reading an invalidated tile's positions. An untracked node (see the module notes) finer
    /// than the coarsest edited tile is taken to be predicted from it. The graph isn't changed.
    pub(crate) fn cascade(&self, edited: impl IntoIterator<Item = TileKey>) -> Cascade {
        let mut cascade = Cascade::default();
        // Readers are one level finer, so a set ordered by level finishes each level before
        // the next: every key is visited once.
        let mut queue: BTreeSet<(Level, TileKey)> =
            edited.into_iter().map(|key| (key.level(), key)).collect();
        if let Some(&(coarsest, _)) = queue.first() {
            queue.extend(
                self.untracked
                    .iter()
                    .filter(|key| key.level() > coarsest)
                    .map(|key| (key.level(), *key)),
            );
        }
        while let Some((_, key)) = queue.pop_first() {
            if !cascade.positions.insert(key) {
                continue;
            }
            let Some(node) = self.nodes.get(&key) else {
                continue;
            };
            queue.extend(node.readers.iter().map(|r| (r.level(), *r)));
            cascade.normals.extend(node.normal_readers.iter().copied());
        }
        cascade
    }

    /// Removes `key`'s node if it is garbage, then the parents' nodes it was keeping, and so on
    /// up the levels.
    fn collect(&mut self, key: TileKey) {
        let mut pending = vec![key];
        while let Some(next) = pending.pop() {
            if !self.nodes.get(&next).is_some_and(Node::is_garbage) {
                continue;
            }
            let Some(node) = self.nodes.remove(&next) else {
                continue;
            };
            self.untracked.remove(&next);
            // A non-resident node has no normals (`left_pool` dropped them), so only its
            // position edges hold other nodes.
            for parent in node.parents.into_iter().flatten() {
                if let Some(up) = self.nodes.get_mut(&parent) {
                    up.readers.remove(&next);
                }
                pending.push(parent);
            }
        }
    }

    /// Nodes in the graph.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Checks the graph's invariants, and describes the first one broken:
    ///
    /// - every node above level 1 has its parents recorded (none is untracked), so an edit reaches
    ///   it through its ancestry;
    /// - every edge is recorded at both ends: a tile's parents list it as a reader and its readers
    ///   list it as a parent, and likewise for normals;
    /// - no node is garbage.
    ///
    /// # Errors
    ///
    /// The broken invariant, in words.
    #[cfg(test)]
    pub(crate) fn check_invariants(&self) -> Result<(), String> {
        /// Fails with `message` unless `holds`.
        ///
        /// # Errors
        ///
        /// `message`, if `holds` is false.
        fn check(holds: bool, message: impl FnOnce() -> String) -> Result<(), String> {
            if holds { Ok(()) } else { Err(message()) }
        }
        check(self.untracked.is_empty(), || {
            format!("untracked: {:?}", self.untracked)
        })?;
        let node = |key: &TileKey, of: &TileKey, what: &str| {
            self.nodes
                .get(key)
                .ok_or_else(|| format!("{of:?}'s {what} {key:?} has no node"))
        };
        // In key order, so the same graph always reports the same violation.
        let mut keys: Vec<&TileKey> = self.nodes.keys().collect();
        keys.sort_unstable();
        for key in keys {
            let n = node(key, key, "own")?;
            check(!n.is_garbage(), || format!("{key:?} is garbage"))?;
            let recorded = n.parents.as_ref().map(Vec::is_empty);
            check(
                recorded.is_some_and(|none| none != from_tiles(key.level()))
                    || (recorded.is_none() && !from_tiles(key.level())),
                || format!("{key:?}: parents {:?} at {:?}", n.parents, key.level()),
            )?;
            for parent in n.parents.iter().flatten() {
                check(node(parent, key, "parent")?.readers.contains(key), || {
                    format!("{parent:?} doesn't list its reader {key:?}")
                })?;
            }
            for reader in &n.readers {
                let parents = &node(reader, key, "reader")?.parents;
                check(parents.iter().flatten().any(|p| p == key), || {
                    format!("{reader:?} doesn't list its parent {key:?}")
                })?;
            }
            for input in n.normal_inputs.iter().flatten() {
                let readers = &node(input, key, "normal input")?.normal_readers;
                check(readers.contains(key), || {
                    format!("{input:?} doesn't list its normal reader {key:?}")
                })?;
            }
            for reader in &n.normal_readers {
                let inputs = &node(reader, key, "normal reader")?.normal_inputs;
                check(inputs.iter().flatten().any(|i| i == key), || {
                    format!("{reader:?} doesn't list its normal input {key:?}")
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::FaceId;

    use super::*;

    /// Tile `n` of level `level` (as face `n`, so keys of one level are distinct).
    fn key(level: u8, n: u32) -> TileKey {
        TileKey::displacement(FaceId::new(n), Level::new(level).unwrap(), 0, 0).unwrap()
    }

    /// The keys, as a set.
    fn set<const N: usize>(keys: [TileKey; N]) -> BTreeSet<TileKey> {
        keys.into_iter().collect()
    }

    #[test]
    fn edits_reach_cached_descendants_through_evicted_tiles() {
        let mut graph = DepGraph::new();
        let (a, b, c, d) = (key(1, 0), key(2, 0), key(3, 0), key(3, 1));
        graph.record_positions(a, &[key(0, 0)]);
        graph.record_positions(b, &[a]);
        graph.record_positions(c, &[b]);
        graph.record_positions(key(1, 7), &[]);
        graph.record_positions(key(2, 7), &[key(1, 7)]);
        graph.record_positions(d, &[key(2, 7)]);
        graph.left_pool(b);
        graph.left_pool(key(2, 7));
        graph.check_invariants().unwrap();
        assert_eq!(
            graph.cascade([a]).positions,
            set([a, b, c]),
            "c was predicted from b, which was predicted from a, though b is gone"
        );
        assert_eq!(
            graph.cascade([key(2, 7)]).positions,
            set([key(2, 7), d]),
            "an evicted parent still leads to its readers"
        );
    }

    #[test]
    fn normals_cascade_from_dropped_positions() {
        let mut graph = DepGraph::new();
        let (p, x, y, z) = (key(1, 0), key(2, 0), key(2, 1), key(2, 2));
        for n in 0..3 {
            graph.record_positions(key(1, n), &[]);
        }
        graph.record_positions(x, &[p]);
        graph.record_positions(y, &[key(1, 1)]);
        graph.record_positions(z, &[key(1, 2)]);
        graph.record_normals(x, &[x, y]);
        graph.record_normals(y, &[x, y, z]);
        graph.record_normals(z, &[y, z]);
        graph.check_invariants().unwrap();
        let cascade = graph.cascade([p]);
        assert_eq!(cascade.positions, set([p, x]), "x's positions");
        assert_eq!(
            cascade.normals,
            set([x, y]),
            "every tile whose normals read x"
        );
        let finest = graph.cascade([z]);
        assert_eq!(finest.positions, set([z]), "nothing reads z's positions");
        assert_eq!(finest.normals, set([y, z]), "but y's and z's normals do");
    }

    #[test]
    fn nodes_go_once_nothing_needs_them() {
        let mut graph = DepGraph::new();
        let (a, b, c) = (key(1, 0), key(2, 0), key(3, 0));
        graph.record_positions(a, &[]);
        graph.record_positions(b, &[a]);
        graph.record_positions(c, &[b]);
        graph.record_normals(c, &[c, b]);
        graph.check_invariants().unwrap();
        assert_eq!(graph.len(), 3, "three tiles");
        graph.left_pool(a);
        graph.left_pool(b);
        assert_eq!(graph.len(), 3, "a and b are still read by c");
        graph.check_invariants().unwrap();
        assert!(graph.has_normals(&c), "c's normals stay");
        graph.left_pool(c);
        assert_eq!(graph.len(), 0, "c gone: its whole ancestry goes with it");
    }

    #[test]
    fn dropping_normals_releases_their_inputs() {
        let mut graph = DepGraph::new();
        let (x, y) = (key(1, 0), key(1, 1));
        graph.record_positions(x, &[]);
        graph.record_positions(y, &[]);
        graph.record_normals(x, &[x, y]);
        graph.left_pool(y);
        graph.check_invariants().unwrap();
        assert_eq!(graph.len(), 2, "x's normals read y");
        graph.drop_normals(x);
        assert!(!graph.has_normals(&x), "dropped");
        assert_eq!(graph.len(), 1, "y's node went with the last reader");
        assert!(
            graph.cascade([y]).normals.is_empty(),
            "and edits of y no longer reach x's normals"
        );
    }

    #[test]
    fn a_tile_recorded_after_its_parent_left_is_still_reached() {
        let mut graph = DepGraph::new();
        let (g, p, c) = (key(5, 0), key(6, 0), key(7, 0));
        graph.record_positions(g, &[key(4, 0)]);
        graph.record_positions(p, &[g]);
        // Out of order: p leaves (its node and its edge to g go with it) before c, predicted
        // from p, is recorded. The store never does this; the graph must still not lose c.
        graph.left_pool(p);
        graph.record_positions(c, &[p]);
        assert!(
            graph.cascade([g]).positions.contains(&c),
            "an edit of g reaches c, though p's ancestry was forgotten"
        );
        assert!(
            !graph.cascade([key(7, 9)]).positions.contains(&c),
            "an edit at c's own level doesn't"
        );
        assert!(
            graph.check_invariants().is_err(),
            "and the invariant check reports p as untracked"
        );
    }

    #[test]
    fn a_tile_recorded_before_its_parent_leaves_keeps_every_edge() {
        // The store's order: c is recorded while p is pinned, so p's node and its edge to g
        // outlive p's slot.
        let mut graph = DepGraph::new();
        let (g, p, c) = (key(1, 0), key(2, 0), key(3, 0));
        graph.record_positions(g, &[key(0, 0)]);
        graph.record_positions(p, &[g]);
        graph.record_positions(c, &[p]);
        graph.left_pool(p);
        graph.check_invariants().unwrap();
        assert_eq!(graph.cascade([g]).positions, set([g, p, c]), "through p");
    }

    #[test]
    fn rerecording_positions_drops_normals() {
        let mut graph = DepGraph::new();
        let x = key(2, 0);
        graph.record_positions(key(1, 0), &[]);
        graph.record_positions(x, &[key(1, 0)]);
        graph.record_normals(x, &[x]);
        graph.record_positions(x, &[key(1, 0)]);
        graph.check_invariants().unwrap();
        assert!(!graph.has_normals(&x), "new positions, old normals");
        assert_eq!(
            graph.cascade([key(1, 0)]).positions,
            set([key(1, 0), x]),
            "edges kept once"
        );
    }
}
