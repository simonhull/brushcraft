//! A cluster's triangles: the one static index buffer every cluster draws, and the 16 edge
//! variants that keep its seams watertight beside a coarser tile.
//!
//! # The cluster
//!
//! A cluster is 8 × 8 quads of a tile (81 vertices, 128 triangles), or a whole tile under 8
//! quads a side ([`ClusterSize`]). Its vertices are the grid `(u, v)`, each `0..=8`, numbered
//! `9v + u`; `u` runs along the tile's `i`, `v` along its `j`. Each quad is split along its
//! `(u, v)`–`(u+1, v+1)` diagonal (plan decision 4, M1a2 decision 1: the triangulation `e` is
//! measured against), into `(p00, p10, p11)` and `(p00, p11, p01)`, counter-clockwise in the
//! grid, so facing out of a face whose quads face out in `(i, j)` order. Quad `(x, y)` holds
//! triangles `2(8y + x)` and `2(8y + x) + 1`: [`triangles`] in that order is [`indices`], the
//! static index buffer, and a triangle's place in it is the triangle id the visibility buffer
//! stores.
//!
//! # Edge variants
//!
//! Beside a neighbour a level coarser, a tile's odd border vertices on that edge (odd `u` on the
//! bottom and top edges, odd `v` on the left and right, corners excluded) don't exist on the
//! other side, so drawing them would leave a T-junction. The cluster's [`Variant`] names the
//! edges where that is so (bit `k` for edge `k`, in the tile tree's neighbour order, [`Edge`]),
//! and each odd border vertex of those edges is **replaced by an even neighbour on the same
//! edge**: the triangle that held both degenerates, and the others become the fan the bounds
//! pass already holds in its normal cone (`cull.rs`, "What the cone holds"). With `o` between
//! the even `a` and `b`, and `m`, `ma`, `mb` the vertices one row in from `o`, `a` and `b`, the
//! two border quads become `(a, b, m)`, `(a, m, ma)` and `(b, mb, m)`.
//!
//! Which neighbour follows from the diagonal: `o` moves to the even vertex of the border quad
//! whose diagonal ends at `o` (bottom: `u + 1`; right: `v - 1`; top: `u - 1`; left: `v + 1`).
//! Moving it the other way would make a fan the cone doesn't hold. The edge is then drawn
//! between even vertices only, the very vertices the coarser tile has there, read from the same
//! data: watertight by construction, with no position computed (snapping onto the coarse edge
//! could leave pinholes after rounding, spec).
//!
//! # Why a remap and not 16 index buffers
//!
//! The level choice lists every visible cluster for one instanced indexed indirect draw
//! ([`crate::lod::GpuCluster`]), whose index range is the same for every instance, so a
//! per-instance choice of index list can't come from the index buffer. Each variant is instead
//! a table, [`remap`], from a grid vertex to the vertex drawn in its place, which the vertex
//! shader applies to the index it is given: the variant's triangles are exactly [`triangles`]
//! mapped through it (degenerate ones are its padding), while the hardware still reuses vertices
//! by index (81 vertex-shader invocations a cluster at best, not 384). The same tables pad a
//! cluster of a smaller tile: vertices past its last quad clamp onto it, so the triangles beyond
//! it are degenerate. All of them, 4 sizes × 16 variants × 81 vertices, are one static buffer
//! ([`remap_table`]); a variant's table starts at [`table_offset`].

use std::fmt;

use crate::lod::CLUSTER_INDICES;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// Quads along a full cluster's side.
pub const CLUSTER_QUADS: u8 = 8;

/// Vertices along a full cluster's side.
pub const CLUSTER_SIDE: u8 = CLUSTER_QUADS + 1;

/// Vertices of a full cluster: its grid's.
pub const CLUSTER_VERTICES: usize = CLUSTER_SIDE as usize * CLUSTER_SIDE as usize;

/// Triangles of a full cluster: two a quad.
pub const CLUSTER_TRIANGLES: usize = 2 * CLUSTER_QUADS as usize * CLUSTER_QUADS as usize;

// The level choice's draw arguments ask for exactly this cluster's indices.
const _: () = assert!(
    CLUSTER_INDICES as usize == 3 * CLUSTER_TRIANGLES,
    "three indices a triangle"
);

/// Remap tables: one per cluster size and variant.
pub const TABLES: usize = ClusterSize::ALL.len() * Variant::COUNT;

/// An edge of a tile or of a cluster, in the order of the tile tree's neighbours
/// ([`crate::tree::GpuNode::neighbours`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Edge {
    /// The edge at `v = 0` (the tile's `j = 0`).
    Bottom,
    /// The edge at `u = last`.
    Right,
    /// The edge at `v = last`.
    Top,
    /// The edge at `u = 0`.
    Left,
}

impl Edge {
    /// Every edge, in neighbour order.
    pub const ALL: [Self; 4] = [Self::Bottom, Self::Right, Self::Top, Self::Left];

    /// Its number: its place in neighbour order and its bit in a [`Variant`].
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// Its bit in a [`Variant`] or an edge mask.
    #[must_use]
    pub const fn bit(self) -> u8 {
        1 << self.index()
    }

    /// Whether `vertex` of a cluster of `size` lies on this edge (corners on two edges).
    #[must_use]
    pub const fn holds(self, size: ClusterSize, vertex: GridVertex) -> bool {
        let last = size.quads();
        match self {
            Self::Bottom => vertex.v == 0,
            Self::Right => vertex.u == last,
            Self::Top => vertex.v == last,
            Self::Left => vertex.u == 0,
        }
    }
}

/// The edges of a cluster whose neighbour across them is a level coarser: bit `k` for
/// edge `k` of [`Edge::ALL`], as the level choice writes it into a cluster
/// ([`crate::lod::GpuCluster`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Variant(u8);

impl Variant {
    /// How many variants there are.
    pub const COUNT: usize = 16;
    /// No edge beside a coarser neighbour: the static triangles as they are.
    pub const PLAIN: Self = Self(0);

    /// The variant of `bits` (bit `k` for edge `k`), if below 16.
    #[must_use]
    pub const fn new(bits: u8) -> Option<Self> {
        if (bits as usize) < Self::COUNT {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// The variant skipping the odd vertices of every edge in `edges`.
    #[must_use]
    pub fn of_edges(edges: &[Edge]) -> Self {
        Self(edges.iter().fold(0, |bits, edge| bits | edge.bit()))
    }

    /// Its bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether the neighbour across `edge` is a level coarser.
    #[must_use]
    pub const fn has(self, edge: Edge) -> bool {
        self.0 & edge.bit() != 0
    }

    /// Every variant, in bit order.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).filter_map(|bits| u8::try_from(bits).ok().map(Self))
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04b}", self.0)
    }
}

/// Quads along a cluster's side: 8, or a whole tile of fewer (the levels below 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClusterSize {
    /// One quad: a level-0 tile.
    One,
    /// Two quads: a level-1 tile.
    Two,
    /// Four quads: a level-2 tile.
    Four,
    /// Eight quads: a full cluster, of any tile of level 3 or finer.
    Eight,
}

impl ClusterSize {
    /// Every size, smallest first: its place here is [`ClusterSize::index`].
    pub const ALL: [Self; 4] = [Self::One, Self::Two, Self::Four, Self::Eight];

    /// The size of `quads` quads a side, if it is 1, 2, 4 or 8.
    #[must_use]
    pub const fn of_quads(quads: u32) -> Option<Self> {
        match quads {
            1 => Some(Self::One),
            2 => Some(Self::Two),
            4 => Some(Self::Four),
            8 => Some(Self::Eight),
            _ => None,
        }
    }

    /// Quads a side.
    #[must_use]
    pub const fn quads(self) -> u8 {
        1 << self.index()
    }

    /// Its place in [`ClusterSize::ALL`]: the base-2 logarithm of its quads.
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }
}

/// A vertex of the cluster grid: `u` along the tile's `i`, `v` along its `j`, both `0 ..= 8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GridVertex {
    /// The column.
    u: u8,
    /// The row.
    v: u8,
}

impl GridVertex {
    /// The vertex `(u, v)`, if both are on the grid (at most 8).
    #[must_use]
    pub const fn new(u: u8, v: u8) -> Option<Self> {
        if u < CLUSTER_SIDE && v < CLUSTER_SIDE {
            Some(Self { u, v })
        } else {
            None
        }
    }

    /// The vertex numbered `index` (`v · 9 + u`), if below 81.
    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        Self::new(index % CLUSTER_SIDE, index / CLUSTER_SIDE)
    }

    /// The column.
    #[must_use]
    pub const fn u(self) -> u8 {
        self.u
    }

    /// The row.
    #[must_use]
    pub const fn v(self) -> u8 {
        self.v
    }

    /// Its number, `v · 9 + u`: the value an index holds.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.v * CLUSTER_SIDE + self.u
    }

    /// Every vertex of the grid, in index order.
    pub fn all() -> impl Iterator<Item = Self> {
        (0..CLUSTER_SIDE).flat_map(|v| (0..CLUSTER_SIDE).map(move |u| Self { u, v }))
    }
}

/// The static triangles of a full cluster, in index-buffer order: per quad `(x, y)`, row by row,
/// `(p00, p10, p11)` then `(p00, p11, p01)`, split along the `(x, y)`–`(x + 1, y + 1)` diagonal.
pub fn triangles() -> impl Iterator<Item = [GridVertex; 3]> {
    (0..CLUSTER_QUADS).flat_map(|y| {
        (0..CLUSTER_QUADS).flat_map(move |x| {
            let at = |dx: u8, dy: u8| GridVertex {
                u: x + dx,
                v: y + dy,
            };
            [
                [at(0, 0), at(1, 0), at(1, 1)],
                [at(0, 0), at(1, 1), at(0, 1)],
            ]
        })
    })
}

/// The static index buffer: [`triangles`]' vertex numbers, [`CLUSTER_INDICES`] of them.
#[must_use]
pub fn indices() -> Vec<u16> {
    triangles()
        .flatten()
        .map(|vertex| u16::from(vertex.index()))
        .collect()
}

/// The vertex a cluster of `size` and `variant` draws in place of `vertex`.
///
/// That is `vertex` clamped onto the cluster's quads, then, if it is an odd border vertex of an
/// edge the variant names, the even neighbour on that edge that takes its place (see the
/// module's rustdoc).
#[must_use]
pub const fn remap(size: ClusterSize, variant: Variant, vertex: GridVertex) -> GridVertex {
    let last = size.quads();
    let u = if vertex.u < last { vertex.u } else { last };
    let v = if vertex.v < last { vertex.v } else { last };
    let odd_u = u % 2 == 1 && u < last;
    let odd_v = v % 2 == 1 && v < last;
    if v == 0 && odd_u && variant.has(Edge::Bottom) {
        return GridVertex { u: u + 1, v };
    }
    if u == last && odd_v && variant.has(Edge::Right) {
        return GridVertex { u, v: v - 1 };
    }
    if v == last && odd_u && variant.has(Edge::Top) {
        return GridVertex { u: u - 1, v };
    }
    if u == 0 && odd_v && variant.has(Edge::Left) {
        return GridVertex { u, v: v + 1 };
    }
    GridVertex { u, v }
}

/// Where the remap table of `size` and `variant` starts in [`remap_table`], in entries (one a
/// grid vertex).
#[must_use]
pub const fn table_offset(size: ClusterSize, variant: Variant) -> usize {
    (size.index() as usize * Variant::COUNT + variant.bits() as usize) * CLUSTER_VERTICES
}

/// Every remap table, as the GPU reads them, four entries a little-endian word.
///
/// [`TABLES`] tables of [`CLUSTER_VERTICES`] entries, the table of `size` and `variant` at
/// [`table_offset`], each entry the number of the vertex [`remap`] draws.
#[must_use]
pub fn remap_table() -> Vec<u32> {
    let entries = ClusterSize::ALL.into_iter().flat_map(|size| {
        Variant::all().flat_map(move |variant| {
            GridVertex::all().map(move |vertex| remap(size, variant, vertex).index())
        })
    });
    let bytes: Vec<u8> = entries.collect();
    bytes
        .chunks(4)
        .map(|word| {
            word.iter()
                .rev()
                .fold(0_u32, |packed, &byte| (packed << 8) | u32::from(byte))
        })
        .collect()
}

/// The triangles a cluster of `size` and `variant` draws, in index-buffer order.
///
/// They are [`triangles`] mapped through [`remap`], `None` where one degenerates (two of its
/// vertices the same: the variant's or the size's padding).
pub fn drawn(size: ClusterSize, variant: Variant) -> impl Iterator<Item = Option<[GridVertex; 3]>> {
    triangles().map(move |triangle| {
        let [a, b, c] = triangle.map(|vertex| remap(size, variant, vertex));
        (a != b && b != c && c != a).then_some([a, b, c])
    })
}
