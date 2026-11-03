//! A block of one level's positions with its one-vertex ring and its base-vertex rings.
//!
//! As plain data: what evaluation gathers, for work done where `geom` itself doesn't run (normals
//! on the GPU, from a finest tile and its ring). [`gather_block`] reads what
//! [`predict_region`](super::predict_region) and [`normals`](fn@super::normals) read, through the
//! same owner copies (it is the same gather); [`vertex_stencil`] says which of those reads each
//! vertex's frame and normal take, and [`GatheredBlock::normals`] computes the normals from the
//! block alone, the bits [`normals`](fn@super::normals) gives.
//!
//! # Layout
//!
//! For a block of `q` quads a side (`q + 1` vertices; a finest tile of 65² vertices has `q = 64`):
//!
//! - **Positions:** `(q + 3)²` of them, row-major from `(-1, -1)`, so a side of 67 for a 65² tile.
//!   Block vertex `(i, j)`, `-1 ≤ i, j ≤ q + 1`, is at index `(j + 1) · (q + 3) + i + 1`. Rows and
//!   columns `-1` and `q + 1` are the ring: the vertices one grid step off the block, in the
//!   block's own grid orientation, across a face edge where the block touches one, each read from
//!   its owner copy, so every face holding a vertex holds its bits. The ring's four corner slots
//!   are NaN where the block's corner is a corner of its face: diagonally off a face corner there
//!   is no vertex (elsewhere they hold the diagonal neighbour).
//! - **Base vertices:** each block corner `c` (`0..4`: `(0, 0)`, `(q, 0)`, `(q, q)`, `(0, q)`) that
//!   is a corner of its face is a base vertex, of any valence, and carries its ring
//!   ([`GatheredBlock::corner`]): the vertex (the bits the positions hold there), then one edge
//!   neighbour and one diagonal neighbour per face around it, in ring order from the face owning
//!   the vertex. A ring of valence other than 4 reaches vertices the positions lack (at valence 3
//!   the missing diagonal, at 5 or more the faces that touch the block only at the vertex). A ring
//!   of valence 4 is given too: its points are in the positions, but the normal there sums the ring
//!   in that order, from the owner, and only that order gives every face's copy the same bits.
//!
//! # Normals from the layout
//!
//! What [`GatheredBlock::normals`] computes, vertex by vertex, as a GPU computing them must, in
//! `f32`, with no fused multiply-add and no reordering (the rule is in
//! [`normals`](fn@super::normals)):
//!
//! - `Axes([u, v])` ([`VertexStencil::Axes`]): `du = p(i + u) − p(i − u)`, `dv = p(i + v) − p(i −
//!   v)` from the positions, and `n = du × dv`;
//! - `Corner(c)` ([`VertexStencil::Corner`]): with `v` the ring's vertex and `e_k` its edge
//!   neighbours, `n = Σ_k (e_k − v) × (e_{k+1} − v)`, `k` from 0 (the owner) round the ring,
//!   wrapping, summed left to right from zero;
//! - `Outside` ([`VertexStencil::Outside`]): NaN (never, for a vertex of the block).
//!
//! A cross product is `(a.y·b.z − a.z·b.y, a.z·b.x − a.x·b.z, a.x·b.y − a.y·b.x)`. Then
//! `l = √(n.x·n.x + n.y·n.y + n.z·n.z)` (summed left to right), and the normal is `n · (1 / l)`
//! (the reciprocal once, then three products) where `l` is finite and above `f32::MIN_POSITIVE`;
//! else `(0, 0, 1)` where the inputs are finite (`du` and `dv`, or the ring), else NaN.
//!
//! Which stencil a vertex takes is structural: [`vertex_stencil`] gives it from the vertex's place
//! on its face and the mesh's topology, never from positions. Inside a face it is the block's own
//! axes; on a face border it is the axes of the face owning the vertex, as steps of this block's
//! grid.

use super::{EvalError, Grid, PositionSource, block_of, gather, normals_of};
use crate::frame::{Stencil, stencil};
use crate::subdiv::{CornerRing, Padded};
use crate::{BaseMesh, Region, Vec3};

/// A block of one level's positions with its one-vertex ring, and the rings of the base vertices
/// at its corners: what evaluation gathers (see the module notes for the layout).
#[derive(Debug, Clone, PartialEq)]
pub struct GatheredBlock {
    /// The block.
    region: Region,
    /// The block's positions with the ring, `(quads + 3)²`, NaN where there is no vertex.
    padded: Padded,
    /// Per block corner (`(0, 0)`, `(q, 0)`, `(q, q)`, `(0, q)`), the ring of the base vertex
    /// there, if the corner is one.
    corners: [Option<GatheredRing>; 4],
}

impl GatheredBlock {
    /// The block gathered, without its ring.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// Quads per side of the block, without the ring.
    #[must_use]
    pub const fn quads(&self) -> u32 {
        self.region.quads()
    }

    /// Positions per side, the ring included: `quads + 3` (67 for a tile of 65² vertices).
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.region.quads() + 3
    }

    /// The `(quads + 3)²` positions, row-major from `(-1, -1)`: block vertex `(i, j)` is at
    /// `(j + 1) · (quads + 3) + i + 1`. NaN where there is no vertex.
    #[must_use]
    pub fn padded(&self) -> &[Vec3] {
        self.padded.as_slice()
    }

    /// The position at block vertex `(i, j)`, `-1 ..= quads + 1` on both axes; NaN where there is
    /// no vertex, or off the block and its ring.
    #[must_use]
    pub fn get(&self, i: i64, j: i64) -> Vec3 {
        self.padded.get(i, j)
    }

    /// Row `j` (`-1 ..= quads + 1`) with its ring: `quads + 3` positions, column `i` at index
    /// `i + 1`. Empty off the block and its ring.
    #[must_use]
    pub fn row(&self, j: i64) -> &[Vec3] {
        self.padded.row(j)
    }

    /// Moves every point the block holds (its positions, ring included, and its base-vertex
    /// rings' points) through `f` of it and `reference`'s point in the same slot: its counterpart
    /// in a block gathered over the same region (the same surface at another time: a brush
    /// stroke's previews weigh each point's move at its position as the stroke began). Slots
    /// without a vertex in either block (NaN) stay as they are; `f` isn't called for them.
    ///
    /// Every block holds a vertex's owner bits, this one and `reference` alike, so an `f` that is
    /// a function of the two points' bits alone moves the vertex to the same new bits in every
    /// block: blocks so moved still meet bit for bit.
    ///
    /// Returns whether the blocks match: `false`, and nothing moved, if `reference` isn't
    /// gathered over this block's region, or its base-vertex rings aren't this block's.
    #[must_use = "a block not gathered over the same region is left unmoved"]
    pub fn displace_along(
        &mut self,
        reference: &Self,
        mut f: impl FnMut(Vec3, Vec3) -> Vec3,
    ) -> bool {
        let same_rings = self
            .corners
            .iter()
            .zip(&reference.corners)
            .all(|(own, other)| match (own, other) {
                (Some(own), Some(other)) => own.ring.valence() == other.ring.valence(),
                (None, None) => true,
                _ => false,
            });
        if self.region != reference.region || !same_rings {
            return false;
        }
        let finite = |v: &Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
        let slots = self
            .padded
            .as_mut_slice()
            .iter_mut()
            .zip(reference.padded.as_slice());
        let rings = self.corners.iter_mut().zip(&reference.corners);
        let ring_points = rings.flat_map(|(own, other)| {
            own.iter_mut()
                .zip(other)
                .flat_map(|(own, other)| own.ring.points_mut().zip(other.ring.points()))
        });
        for (point, at) in slots.chain(ring_points) {
            if finite(point) && finite(at) {
                *point = f(*point, *at);
            }
        }
        true
    }

    /// The ring of the base vertex at block corner `c` (`0..4`: `(0, 0)`, `(q, 0)`, `(q, q)`,
    /// `(0, q)`), or `None` if that corner isn't a base vertex.
    #[must_use]
    pub fn corner(&self, c: usize) -> Option<&GatheredRing> {
        self.corners.get(c)?.as_ref()
    }

    /// The block's unit normals, row-major `(quads + 1)²`, from the block alone: the bits
    /// [`normals`](fn@super::normals) gives for the block from the source it was gathered from.
    ///
    /// `mesh` must be the mesh the block was gathered over: the stencils come from its topology.
    #[must_use]
    pub fn normals(&self, mesh: &BaseMesh) -> Grid {
        let side = self.quads() + 1;
        let count = usize::try_from(side * side).unwrap_or(usize::MAX);
        let mut data = vec![Vec3::ZERO; count];
        self.write_normals(mesh, &mut data);
        Grid::from_parts(side, data)
    }

    /// [`GatheredBlock::normals`] into `out`, row-major `(quads + 1)²`, without allocating.
    ///
    /// # Errors
    ///
    /// [`EvalError::BufferSize`] unless `out` holds exactly `(quads + 1)²` values.
    pub fn normals_into(&self, mesh: &BaseMesh, out: &mut [Vec3]) -> Result<(), EvalError> {
        let side = self.quads() + 1;
        super::check_buffer(out, usize::try_from(side * side).unwrap_or(usize::MAX))?;
        self.write_normals(mesh, out);
        Ok(())
    }

    /// The normals into `out`, which holds one value per block vertex.
    fn write_normals(&self, mesh: &BaseMesh, out: &mut [Vec3]) {
        let rings = self
            .corners
            .each_ref()
            .map(|corner| corner.as_ref().map(|ring| &ring.ring));
        normals_of(mesh, self.region, &self.padded, rings, out);
    }
}

/// The one-ring of a base vertex at one level, owner first: quad `k` is
/// `(center, edges[k], diagonals[k], edges[k + 1])`, wrapping.
#[derive(Debug, Clone)]
pub struct GatheredRing {
    /// The ring, as evaluation gathered it.
    ring: CornerRing,
}

impl GatheredRing {
    /// The base vertex: the bits the block's positions hold at its corner.
    #[must_use]
    pub const fn center(&self) -> Vec3 {
        self.ring.center
    }

    /// The edge neighbours, one per face around the vertex, in ring order, owner first.
    #[must_use]
    pub fn edges(&self) -> &[Vec3] {
        self.ring.edges()
    }

    /// The diagonal neighbours (each face's vertex opposite the base vertex), in ring order,
    /// owner first.
    #[must_use]
    pub fn diagonals(&self) -> &[Vec3] {
        self.ring.diagonals()
    }

    /// How many faces meet at the vertex: the length of [`GatheredRing::edges`] and of
    /// [`GatheredRing::diagonals`].
    #[must_use]
    pub fn valence(&self) -> usize {
        self.ring.edges().len()
    }

    /// The owner face's corner at this vertex (`0..4`): which ring points lie along the
    /// owner's axes.
    #[must_use]
    pub const fn owner_corner(&self) -> u8 {
        self.ring.owner_corner
    }
}

impl PartialEq for GatheredRing {
    /// The same vertex, ring points and owner corner (as `f32`s compare: a NaN point differs).
    fn eq(&self, other: &Self) -> bool {
        self.center() == other.center()
            && self.edges() == other.edges()
            && self.diagonals() == other.diagonals()
            && self.owner_corner() == other.owner_corner()
    }
}

/// Gathers `region` at its own level from `src`, as evaluation does.
///
/// That is the block, its one-vertex ring and its corners' base-vertex rings, every position
/// read from its owner copy (see the module notes for the layout).
///
/// Gathering a tile's parent region gives what predicting the tile reads; gathering the tile's
/// own region (of evaluated positions) gives what its normals read, from the tiles
/// [`normal_tiles`](super::normal_tiles) lists.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a needed position.
pub fn gather_block(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<GatheredBlock, EvalError> {
    let parent = gather(mesh, src, region)?;
    Ok(GatheredBlock {
        region,
        padded: parent.padded,
        corners: parent
            .corners
            .map(|ring| ring.map(|ring| GatheredRing { ring })),
    })
}

/// What a vertex's frame and normal read, as [`vertex_stencil`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStencil {
    /// A base vertex at block corner `c` (`0..4`): it reads its ring.
    Corner(u8),
    /// Central differences along the block-grid steps `[u, v]` (the owner's `+i` and `+j`):
    /// vertex `(i, j)` reads `(i, j) ± u` and `(i, j) ± v`.
    Axes([[i32; 2]; 2]),
    /// Off the face: nothing to read.
    Outside,
}

/// The stencil of block vertex `(i, j)` of `region`, for `i` and `j` in `0..=region.quads()`.
///
/// It is structural, from the vertex's place on the face and the mesh's topology, never from
/// positions. [`VertexStencil::Outside`] for a vertex off the block (either coordinate past
/// `region.quads()`).
///
/// # Panics
///
/// Never: `geom`'s stencils hold block corners `0..4` and grid steps of `-1`, `0` or `1`, which
/// the conversions to `u8` and `i32` take.
#[must_use]
#[expect(
    clippy::expect_used,
    reason = "geom's stencils hold block corners 0..4 and grid steps of -1, 0 or 1"
)]
pub fn vertex_stencil(mesh: &BaseMesh, region: Region, i: u32, j: u32) -> VertexStencil {
    if i > region.quads() || j > region.quads() {
        return VertexStencil::Outside;
    }
    let step = |step: (i64, i64)| {
        <[i64; 2]>::from(step).map(|x| i32::try_from(x).expect("a grid step is -1, 0 or 1"))
    };
    match stencil(&block_of(mesh, region), i64::from(i), i64::from(j)) {
        Stencil::Corner(c) => {
            VertexStencil::Corner(u8::try_from(c).expect("a block corner is 0, 1, 2 or 3"))
        }
        Stencil::Axes([u, v]) => VertexStencil::Axes([step(u), step(v)]),
        Stencil::Outside => VertexStencil::Outside,
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
