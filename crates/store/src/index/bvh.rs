//! A bounding volume hierarchy over tile bounds, built with the binned surface area heuristic.
//!
//! # Layout
//!
//! The tree lives in two flat arrays and has no pointers. `items` holds every tile's key and
//! bounds, ordered so each leaf covers one contiguous range of it. `nodes` holds the nodes in
//! the order the build made them: the root first, and a node's two children always after it
//! (they are made together, as a pair, when the node is split). A refit uses that order:
//! walking the touched nodes from the highest index down updates every child before its
//! parent.
//!
//! # Build
//!
//! A node is split by the binned SAH (see Wald, "On fast Construction of SAH-based Bounding
//! Volume Hierarchies", 2007): along each axis, the items' centroids are sorted into [`BINS`]
//! equal-width bins, and of the `BINS − 1` planes between bins on each axis the build takes the
//! one with the lowest cost `area(left) · count(left) + area(right) · count(right)`. That is the
//! expected cost of a query that reaches the node, if the chance of entering a child is
//! proportional to its surface area. Nodes are split down to [`LEAF_SIZE`] items; where the
//! centroids leave nothing to bin (all equal), the items are halved as they lie.
//!
//! # Queries
//!
//! A sphere query walks the tree depth-first and yields every item whose box touches the
//! sphere. A raycast visits nodes and items nearest-first, by where the ray enters their boxes,
//! from a heap: the caller's precise test runs on the candidate tiles in that order, and stops
//! as soon as the next candidate's box starts beyond the nearest hit found.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use geom::{TileKey, Vec3};

use super::{IndexError, TileBounds};

/// Bins per axis the build sorts centroids into to find a split.
const BINS: usize = 16;

/// [`BINS`] as a float, for scaling centroid offsets into bin indices.
const BINS_F32: f32 = 16.0;

/// The most items a leaf holds: larger nodes are split.
///
/// Small, because the item a raycast finally tests is a whole tile's quads: a candidate the
/// tree could have ruled out costs far more than one more node visit.
const LEAF_SIZE: usize = 4;

/// A coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// The x axis.
    X,
    /// The y axis.
    Y,
    /// The z axis.
    Z,
}

impl Axis {
    /// All three axes, in order.
    const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// `v`'s component along this axis.
    const fn of(self, v: Vec3) -> f32 {
        match self {
            Self::X => v.x,
            Self::Y => v.y,
            Self::Z => v.z,
        }
    }
}

/// The component-wise minimum of `a` and `b`.
const fn component_min(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

/// The component-wise maximum of `a` and `b`.
const fn component_max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

/// An axis-aligned bounding box: every point between [`Aabb::min`] and [`Aabb::max`],
/// component by component.
///
/// Its corners are finite and `min ≤ max` in every component, so a box always holds at least
/// one point; the constructors check it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// The low corner.
    min: Vec3,
    /// The high corner.
    max: Vec3,
}

impl Aabb {
    /// The box from `min` to `max`, or `None` unless both are finite and `min ≤ max` in every
    /// component.
    #[must_use]
    pub fn new(min: Vec3, max: Vec3) -> Option<Self> {
        let ordered = Axis::ALL.iter().all(|&a| a.of(min) <= a.of(max));
        (min.is_finite() && max.is_finite() && ordered).then_some(Self { min, max })
    }

    /// The smallest box holding every finite point of `points`, or `None` if there is none.
    /// Points with a NaN or infinite component are skipped.
    #[must_use]
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        points
            .into_iter()
            .filter(|p| p.is_finite())
            .map(Self::point)
            .reduce(Self::union)
    }

    /// The box holding just `p`, which must be finite.
    const fn point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    /// The low corner.
    #[must_use]
    pub const fn min(&self) -> Vec3 {
        self.min
    }

    /// The high corner.
    #[must_use]
    pub const fn max(&self) -> Vec3 {
        self.max
    }

    /// The smallest box holding both boxes.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: component_min(self.min, other.min),
            max: component_max(self.max, other.max),
        }
    }

    /// The centre of the box.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The box's surface area: `2 (dx dy + dy dz + dz dx)`.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Whether `p` lies in the box (its faces included).
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        Axis::ALL
            .iter()
            .all(|&a| (a.of(self.min)..=a.of(self.max)).contains(&a.of(p)))
    }

    /// The squared distance from `p` to the nearest point of the box: 0 inside it.
    #[must_use]
    pub fn distance_squared(&self, p: Vec3) -> f32 {
        let nearest = component_min(component_max(p, self.min), self.max);
        let d = p - nearest;
        d.dot(d)
    }

    /// Where `ray` enters the box: the least `t ≥ 0` with `ray.at(t)` in it (0 if the origin
    /// is inside), or `None` if the ray misses it. `inverse` is the component-wise reciprocal
    /// of the ray's direction.
    ///
    /// The slab test: along each axis the ray is between the box's two planes for an interval
    /// of `t`, and it is in the box where all three intervals and `t ≥ 0` overlap. A direction
    /// component of zero has an infinite reciprocal, which makes that axis's interval all of
    /// `t` (the origin between the planes) or empty (outside them). A ray along one of the
    /// box's planes is the one case that gives a NaN; `f32::min` and `f32::max` skip it, so
    /// that axis doesn't constrain `t` then.
    fn entry(&self, ray: &Ray, inverse: Vec3) -> Option<f32> {
        let (mut near, mut far) = (0.0_f32, f32::INFINITY);
        for a in Axis::ALL {
            let t0 = (a.of(self.min) - a.of(ray.origin)) * a.of(inverse);
            let t1 = (a.of(self.max) - a.of(ray.origin)) * a.of(inverse);
            near = near.max(t0.min(t1));
            far = far.min(t0.max(t1));
        }
        (near <= far).then_some(near)
    }
}

/// A ray: the points `origin + t · direction` for `t ≥ 0`.
///
/// Its origin and direction are finite and the direction isn't zero; [`Ray::new`] checks it.
/// The direction needn't be a unit vector: a hit's `t` is in units of its length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Where the ray starts.
    origin: Vec3,
    /// Which way it goes, and how far one unit of `t` is.
    direction: Vec3,
}

impl Ray {
    /// The ray from `origin` along `direction`, or `None` unless both are finite and the
    /// direction isn't zero.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Option<Self> {
        (origin.is_finite() && direction.is_finite() && direction != Vec3::ZERO)
            .then_some(Self { origin, direction })
    }

    /// Where the ray starts.
    #[must_use]
    pub const fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Which way the ray goes.
    #[must_use]
    pub const fn direction(&self) -> Vec3 {
        self.direction
    }

    /// The point at parameter `t`: `origin + t · direction`.
    #[must_use]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// The component-wise reciprocal of the direction, for [`Aabb::entry`]: infinite along an
    /// axis the ray is parallel to.
    fn inverse_direction(&self) -> Vec3 {
        let d = self.direction;
        Vec3::new(d.x.recip(), d.y.recip(), d.z.recip())
    }
}

/// Where a ray first hits the surface: the tile it hit, and the ray parameter of the hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The tile the ray hits first.
    pub key: TileKey,
    /// The hit's ray parameter: the point is [`Ray::at`] it.
    pub t: f32,
}

/// What a node is: an inner node with two children, or a leaf over a range of items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    /// A node split in two. The children are made together, so `right` is `left + 1`; both
    /// come after the node in [`Bvh::nodes`].
    Inner {
        /// The index of the first child.
        left: usize,
        /// The index of the second child.
        right: usize,
    },
    /// A leaf over the items `first..end` of [`Bvh::items`]: at least one of them.
    Leaf {
        /// The first item.
        first: usize,
        /// One past the last item.
        end: usize,
    },
}

/// One node of the tree.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Node {
    /// The smallest box holding every item below the node (exactly that, after a build or a
    /// refit).
    aabb: Aabb,
    /// The node's parent; `None` for the root.
    parent: Option<usize>,
    /// Whether it is split, and into what.
    kind: NodeKind,
}

/// One item while the tree is built: a tile, its bounds and its box's centroid.
#[derive(Debug, Clone, Copy)]
struct Entry {
    /// The tile.
    key: TileKey,
    /// Its bounds.
    bounds: TileBounds,
    /// The centre of its box, which the build bins and partitions by.
    centroid: Vec3,
}

/// A bounding volume hierarchy over tiles' bounds (see the module notes): which tiles a
/// sphere touches, and which tile a ray hits first.
///
/// Built once over every tile with [`Bvh::build`], then kept in step with edits by
/// [`Bvh::refit`], which moves tiles' bounds without changing the tree's shape.
#[derive(Clone, Default)]
pub struct Bvh {
    /// The nodes: the root first, a node's children after it (see the module notes). Empty for
    /// a tree of no tiles.
    nodes: Vec<Node>,
    /// Every tile with its bounds, each leaf's items contiguous.
    items: Vec<(TileKey, TileBounds)>,
    /// The leaf node each item is in, by item index.
    leaf_of: Vec<usize>,
    /// The index in [`Bvh::items`] of each tile.
    index_of: HashMap<TileKey, usize>,
}

impl std::fmt::Debug for Bvh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bvh")
            .field("tiles", &self.items.len())
            .field("nodes", &self.nodes.len())
            .field("bounds", &self.nodes.first().map(|root| root.aabb))
            .finish_non_exhaustive()
    }
}

/// One bin of a split search: how many centroids fell in it, and the box of their items.
#[derive(Debug, Clone, Copy, Default)]
struct Bin {
    /// Items whose centroid is in the bin.
    count: usize,
    /// The box of those items; `None` while there are none.
    aabb: Option<Aabb>,
}

/// The union of two optional boxes: the other if one is `None`.
fn union_of(a: Option<Aabb>, b: Option<Aabb>) -> Option<Aabb> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.union(b)),
        (one, None) | (None, one) => one,
    }
}

/// The surface area of an optional box: 0 for `None`.
fn area_of(aabb: Option<Aabb>) -> f32 {
    aabb.map_or(0.0, |b| b.surface_area())
}

/// `count` as a float, for SAH costs.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts past 2^24 round, which only perturbs a cost estimate"
)]
const fn count_f32(count: usize) -> f32 {
    count as f32
}

/// The bin a centroid coordinate `c` falls in, of [`BINS`] equal bins starting at `low`, each
/// `1 / scale` wide. `c` is at least `low`; the top edge falls in the last bin.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the offset is in 0..=BINS by construction; `as` truncates it to its bin and saturates"
)]
fn bin_index(c: f32, low: f32, scale: f32) -> usize {
    (((c - low) * scale) as usize).min(BINS - 1)
}

/// Moves the entries `pred` holds for to the front of `entries`, and returns how many there
/// are. The order within each side is not kept.
fn partition(entries: &mut [Entry], mut pred: impl FnMut(&Entry) -> bool) -> usize {
    let mut front = 0;
    for k in 0..entries.len() {
        if entries.get(k).is_some_and(&mut pred) {
            entries.swap(front, k);
            front += 1;
        }
    }
    front
}

/// Where to split `entries`: how many go to the first child, after reordering them so the
/// first child's come first. `None` if the node should be a leaf.
///
/// Takes the cheapest binned-SAH plane over all three axes (see the module notes). An axis
/// along which every centroid is equal has nothing to bin; if every axis is like that, the
/// entries are halved as they lie.
fn split(entries: &mut [Entry]) -> Option<usize> {
    if entries.len() <= LEAF_SIZE {
        return None;
    }
    let centroids = Aabb::from_points(entries.iter().map(|e| e.centroid))?;
    // The cheapest plane so far: its cost, axis, and the last bin on its low side.
    let mut best: Option<(f32, Axis, usize)> = None;
    for axis in Axis::ALL {
        let low = axis.of(centroids.min);
        let extent = axis.of(centroids.max) - low;
        if extent <= 0.0 {
            continue;
        }
        let scale = BINS_F32 / extent;
        let mut bins = [Bin::default(); BINS];
        for e in entries.iter() {
            if let Some(bin) = bins.get_mut(bin_index(axis.of(e.centroid), low, scale)) {
                bin.count += 1;
                bin.aabb = union_of(bin.aabb, Some(e.bounds.aabb));
            }
        }
        // `above[i]`: the count and area of bins `i + 1..`, the high side of the plane after
        // bin `i`.
        let mut above = [(0, 0.0); BINS];
        let (mut high_count, mut high_aabb) = (0, None);
        for (bin, slot) in bins.iter().zip(above.iter_mut()).rev() {
            *slot = (high_count, area_of(high_aabb));
            high_count += bin.count;
            high_aabb = union_of(high_aabb, bin.aabb);
        }
        let (mut low_count, mut low_aabb) = (0, None);
        for (i, (bin, &(above_count, above_area))) in
            bins.iter().zip(&above).enumerate().take(BINS - 1)
        {
            low_count += bin.count;
            low_aabb = union_of(low_aabb, bin.aabb);
            if low_count == 0 || above_count == 0 {
                continue;
            }
            let cost =
                count_f32(low_count) * area_of(low_aabb) + count_f32(above_count) * above_area;
            if best.is_none_or(|(least, ..)| cost < least) {
                best = Some((cost, axis, i));
            }
        }
    }
    let half = entries.len() / 2;
    let Some((_, axis, last_low_bin)) = best else {
        return Some(half);
    };
    let low = axis.of(centroids.min);
    let scale = BINS_F32 / (axis.of(centroids.max) - low);
    let mid = partition(entries, |e| {
        bin_index(axis.of(e.centroid), low, scale) <= last_low_bin
    });
    // The plane had items on both sides when binned, and the partition bins them the same
    // way, so both sides are non-empty; halving keeps the build finite even if they weren't.
    Some(if mid == 0 || mid == entries.len() {
        half
    } else {
        mid
    })
}

/// The box of every entry of `entries`; `None` if there are none.
fn aabb_of(entries: &[Entry]) -> Option<Aabb> {
    entries.iter().map(|e| e.bounds.aabb).reduce(Aabb::union)
}

impl Bvh {
    /// The tree over `tiles` (see the module notes on the build).
    ///
    /// # Errors
    ///
    /// [`IndexError::DuplicateTile`] if a tile is listed twice.
    pub fn build(tiles: Vec<(TileKey, TileBounds)>) -> Result<Self, IndexError> {
        let mut entries: Vec<Entry> = tiles
            .into_iter()
            .map(|(key, bounds)| Entry {
                key,
                bounds,
                centroid: bounds.aabb.centroid(),
            })
            .collect();
        let mut nodes = Vec::with_capacity(2 * entries.len().div_ceil(LEAF_SIZE));
        if let Some(aabb) = aabb_of(&entries) {
            nodes.push(Node {
                aabb,
                parent: None,
                kind: NodeKind::Leaf {
                    first: 0,
                    end: entries.len(),
                },
            });
        }
        // Nodes still to split: each starts as a leaf over its entries (a valid node already),
        // and becomes an inner node if it splits. An explicit stack, so a lopsided tree can't
        // overflow the call stack.
        let mut work: Vec<(usize, usize, &mut [Entry])> = vec![(0, 0, entries.as_mut_slice())];
        while let Some((node, first, here)) = work.pop() {
            let end = first + here.len();
            let Some(mid) = split(here) else {
                continue;
            };
            let (low, high) = here.split_at_mut(mid);
            let (Some(low_aabb), Some(high_aabb)) = (aabb_of(low), aabb_of(high)) else {
                continue;
            };
            let left = nodes.len();
            let right = left + 1;
            let child = |aabb, from, to| Node {
                aabb,
                parent: Some(node),
                kind: NodeKind::Leaf {
                    first: from,
                    end: to,
                },
            };
            nodes.push(child(low_aabb, first, first + mid));
            nodes.push(child(high_aabb, first + mid, end));
            if let Some(parent) = nodes.get_mut(node) {
                parent.kind = NodeKind::Inner { left, right };
            }
            work.push((right, first + mid, high));
            work.push((left, first, low));
        }
        drop(work);
        let mut leaf_of = vec![0; entries.len()];
        for (index, node) in nodes.iter().enumerate() {
            if let NodeKind::Leaf { first, end } = node.kind
                && let Some(range) = leaf_of.get_mut(first..end)
            {
                range.fill(index);
            }
        }
        let mut index_of = HashMap::with_capacity(entries.len());
        for (index, e) in entries.iter().enumerate() {
            if index_of.insert(e.key, index).is_some() {
                return Err(IndexError::DuplicateTile(e.key));
            }
        }
        Ok(Self {
            nodes,
            items: entries.into_iter().map(|e| (e.key, e.bounds)).collect(),
            leaf_of,
            index_of,
        })
    }

    /// How many tiles the tree holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the tree holds no tiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The box of every tile, or `None` for an empty tree.
    #[must_use]
    pub fn aabb(&self) -> Option<Aabb> {
        self.nodes.first().map(|root| root.aabb)
    }

    /// `key`'s bounds, as last built or refitted; `None` if the tree doesn't hold it.
    #[must_use]
    pub fn bounds(&self, key: &TileKey) -> Option<&TileBounds> {
        let &index = self.index_of.get(key)?;
        self.items.get(index).map(|(_, bounds)| bounds)
    }

    /// Every tile with its bounds, in leaf order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(TileKey, TileBounds)> {
        self.items.iter()
    }

    /// Replaces the bounds of the tiles in `changed`, then updates the boxes of their leaves
    /// and of every node above them. The tree keeps its shape: a refit is cheap, but a tree
    /// refitted after large moves answers queries more slowly than a rebuilt one.
    ///
    /// # Errors
    ///
    /// [`IndexError::UnknownTile`] if a tile of `changed` isn't in the tree; nothing is changed
    /// then.
    pub fn refit(&mut self, changed: &[(TileKey, TileBounds)]) -> Result<(), IndexError> {
        let at = changed
            .iter()
            .map(|(key, _)| {
                self.index_of
                    .get(key)
                    .copied()
                    .ok_or(IndexError::UnknownTile(*key))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Every node whose box may change: the changed items' leaves and all their ancestors.
        // A walk up stops at a node already marked, whose ancestors are marked too.
        let mut stale = BTreeSet::new();
        for (&index, &(_, bounds)) in at.iter().zip(changed) {
            if let Some(item) = self.items.get_mut(index) {
                item.1 = bounds;
            }
            let mut node = self.leaf_of.get(index).copied();
            while let Some(n) = node {
                if !stale.insert(n) {
                    break;
                }
                node = self.nodes.get(n).and_then(|n| n.parent);
            }
        }
        // Children come after their parents, so going down the indices updates every child
        // before the parent that unites it.
        for &n in stale.iter().rev() {
            let aabb = self.nodes.get(n).and_then(|node| match node.kind {
                NodeKind::Leaf { first, end } => self
                    .items
                    .get(first..end)?
                    .iter()
                    .map(|(_, bounds)| bounds.aabb)
                    .reduce(Aabb::union),
                NodeKind::Inner { left, right } => union_of(
                    self.nodes.get(left).map(|c| c.aabb),
                    self.nodes.get(right).map(|c| c.aabb),
                ),
            });
            if let (Some(aabb), Some(node)) = (aabb, self.nodes.get_mut(n)) {
                node.aabb = aabb;
            }
        }
        Ok(())
    }

    /// Every tile whose box touches the sphere of `radius` around `center` (its surface
    /// included), each once, in no particular order. Nothing for a negative or NaN radius.
    #[must_use]
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> SphereQuery<'_> {
        let valid = radius >= 0.0;
        SphereQuery {
            bvh: self,
            center,
            radius_squared: radius * radius,
            stack: if valid && !self.nodes.is_empty() {
                vec![0]
            } else {
                Vec::new()
            },
            leaf: 0..0,
        }
    }

    /// The nearest hit of `ray` on the tiles, as `hit` finds them.
    ///
    /// `hit(key)` intersects `ray` with tile `key` precisely and returns the ray parameter of
    /// its nearest intersection, `None` if there is none. It is called only for tiles whose box
    /// the ray enters, nearest box first, and no more once the next box starts past the
    /// nearest hit so far. Its hits must lie in the tile's box, as the tile's surface does;
    /// a negative or NaN parameter counts as a miss.
    ///
    /// # Errors
    ///
    /// The first error `hit` returns; the search stops there.
    pub fn raycast<E>(
        &self,
        ray: &Ray,
        mut hit: impl FnMut(TileKey) -> Result<Option<f32>, E>,
    ) -> Result<Option<Hit>, E> {
        let inverse = ray.inverse_direction();
        let mut heap = BinaryHeap::new();
        // Queues `what` if the ray enters its box `aabb`, keyed by where it does.
        let enter = |queue: &mut BinaryHeap<Candidate>, aabb: &Aabb, what| {
            if let Some(entry) = aabb.entry(ray, inverse) {
                queue.push(Candidate { entry, what });
            }
        };
        if let Some(root) = self.nodes.first() {
            enter(&mut heap, &root.aabb, Visit::Node(0));
        }
        let mut best: Option<Hit> = None;
        while let Some(Candidate {
            entry: nearest,
            what,
        }) = heap.pop()
        {
            if best.is_some_and(|b| nearest >= b.t) {
                break;
            }
            match what {
                Visit::Item(index) => {
                    let Some(&(key, _)) = self.items.get(index) else {
                        continue;
                    };
                    if let Some(t) = hit(key)?
                        && t >= 0.0
                        && best.is_none_or(|b| t < b.t)
                    {
                        best = Some(Hit { key, t });
                    }
                }
                Visit::Node(n) => match self.nodes.get(n).map(|node| node.kind) {
                    Some(NodeKind::Inner { left, right }) => {
                        for child in [left, right] {
                            if let Some(c) = self.nodes.get(child) {
                                enter(&mut heap, &c.aabb, Visit::Node(child));
                            }
                        }
                    }
                    Some(NodeKind::Leaf { first, end }) => {
                        let items = self.items.get(first..end).unwrap_or_default();
                        for (index, (_, bounds)) in (first..).zip(items) {
                            enter(&mut heap, &bounds.aabb, Visit::Item(index));
                        }
                    }
                    None => {}
                },
            }
        }
        Ok(best)
    }
}

/// What a raycast visits next: a node, or an item to test precisely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Visit {
    /// A node of [`Bvh::nodes`].
    Node(usize),
    /// An item of [`Bvh::items`].
    Item(usize),
}

/// A raycast's heap entry: something to visit, and where the ray enters its box.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// The ray parameter where the ray enters the box: the heap pops the least first.
    entry: f32,
    /// What to visit.
    what: Visit,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Reversed on the entry, so [`BinaryHeap`], a max-heap, pops the nearest first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.entry.total_cmp(&self.entry)
    }
}

/// The tiles a sphere touches: the iterator [`Bvh::query_sphere`] returns.
///
/// It walks the tree depth-first as it is advanced, so taking only the first few tiles costs
/// only the walk to them.
#[derive(Debug, Clone)]
pub struct SphereQuery<'a> {
    /// The tree.
    bvh: &'a Bvh,
    /// The sphere's centre.
    center: Vec3,
    /// The square of its radius.
    radius_squared: f32,
    /// Nodes still to visit.
    stack: Vec<usize>,
    /// The items of the leaf being visited that are still to test.
    leaf: std::ops::Range<usize>,
}

impl SphereQuery<'_> {
    /// Whether `aabb` touches the sphere.
    fn touches(&self, aabb: &Aabb) -> bool {
        aabb.distance_squared(self.center) <= self.radius_squared
    }
}

impl Iterator for SphereQuery<'_> {
    type Item = TileKey;

    fn next(&mut self) -> Option<TileKey> {
        loop {
            if let Some(index) = self.leaf.next() {
                if let Some((key, bounds)) = self.bvh.items.get(index)
                    && self.touches(&bounds.aabb)
                {
                    return Some(*key);
                }
                continue;
            }
            let node = self.bvh.nodes.get(self.stack.pop()?)?;
            if !self.touches(&node.aabb) {
                continue;
            }
            match node.kind {
                NodeKind::Inner { left, right } => self.stack.extend([right, left]),
                NodeKind::Leaf { first, end } => self.leaf = first..end,
            }
        }
    }
}

#[cfg(test)]
pub(super) mod tests;
