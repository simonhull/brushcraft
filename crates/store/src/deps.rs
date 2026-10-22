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

use std::collections::{BTreeSet, HashMap};

use geom::{Level, TileKey};

/// One tile's place in the graph.
#[derive(Debug, Default)]
struct Node {
    /// The tile is in the evaluated pool (fresh or awaiting recomputation in place).
    resident: bool,
    /// The parent tiles its positions were predicted from (level ≥ 1 only).
    parents: Vec<TileKey>,
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
    nodes: HashMap<TileKey, Node>,
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
            .map(|node| std::mem::take(&mut node.parents))
            .unwrap_or_default();
        for parent in &old {
            if let Some(node) = self.nodes.get_mut(parent) {
                node.readers.remove(&key);
            }
        }
        for parent in &parents {
            self.nodes.entry(*parent).or_default().readers.insert(key);
        }
        let node = self.nodes.entry(key).or_default();
        node.resident = true;
        node.parents = parents;
        // Parents no longer read (none, while the inputs are structural) may be garbage now.
        for parent in old {
            self.collect(parent);
        }
    }

    /// Records that `key`'s slot holds normals computed from the positions of `inputs`.
    pub(crate) fn record_normals<'a>(
        &mut self,
        key: TileKey,
        inputs: impl IntoIterator<Item = &'a TileKey>,
    ) {
        self.drop_normals(key);
        let inputs: Vec<TileKey> = inputs.into_iter().copied().collect();
        for input in &inputs {
            self.nodes
                .entry(*input)
                .or_default()
                .normal_readers
                .insert(key);
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
    /// reading an invalidated tile's positions. The graph isn't changed.
    pub(crate) fn cascade(&self, edited: impl IntoIterator<Item = TileKey>) -> Cascade {
        let mut cascade = Cascade::default();
        // Readers are one level finer, so a set ordered by level finishes each level before
        // the next: every key is visited once.
        let mut queue: BTreeSet<(Level, TileKey)> =
            edited.into_iter().map(|key| (key.level(), key)).collect();
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
            // A non-resident node has no normals (`left_pool` dropped them), so only its
            // position edges hold other nodes.
            for parent in node.parents {
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
        graph.record_positions(d, &[key(2, 7)]);
        graph.left_pool(b);
        assert_eq!(
            graph.cascade([a]).positions,
            set([a, b, c]),
            "c was predicted from b, which was predicted from a, though b is gone"
        );
        assert_eq!(
            graph.cascade([key(2, 7)]).positions,
            set([key(2, 7), d]),
            "an uncached, never-evaluated parent still leads to its readers"
        );
    }

    #[test]
    fn normals_cascade_from_dropped_positions() {
        let mut graph = DepGraph::new();
        let (p, x, y, z) = (key(4, 0), key(5, 0), key(5, 1), key(5, 2));
        graph.record_positions(x, &[p]);
        graph.record_positions(y, &[key(4, 1)]);
        graph.record_positions(z, &[key(4, 2)]);
        graph.record_normals(x, &[x, y]);
        graph.record_normals(y, &[x, y, z]);
        graph.record_normals(z, &[y, z]);
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
        assert_eq!(graph.len(), 3, "three tiles");
        graph.left_pool(a);
        graph.left_pool(b);
        assert_eq!(graph.len(), 3, "a and b are still read by c");
        assert!(graph.has_normals(&c), "c's normals stay");
        graph.left_pool(c);
        assert_eq!(graph.len(), 0, "c gone: its whole ancestry goes with it");
    }

    #[test]
    fn dropping_normals_releases_their_inputs() {
        let mut graph = DepGraph::new();
        let (x, y) = (key(6, 0), key(6, 1));
        graph.record_positions(x, &[]);
        graph.record_positions(y, &[]);
        graph.record_normals(x, &[x, y]);
        graph.left_pool(y);
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
    fn rerecording_positions_drops_normals() {
        let mut graph = DepGraph::new();
        let x = key(7, 0);
        graph.record_positions(x, &[key(6, 0)]);
        graph.record_normals(x, &[x]);
        graph.record_positions(x, &[key(6, 0)]);
        assert!(!graph.has_normals(&x), "new positions, old normals");
        assert_eq!(
            graph.cascade([key(6, 0)]).positions,
            set([key(6, 0), x]),
            "edges kept once"
        );
    }
}
