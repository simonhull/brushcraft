//! A block's evaluation inputs as plain data, for evaluating it where `geom` itself doesn't run:
//! `render`'s GPU evaluation spike uploads them and runs the same stencils in a compute shader.
//!
//! [`gather_block`] reads what [`predict_region`](super::predict_region) and
//! [`normals`](super::normals) read, through the same owner copies; [`vertex_stencil`] says which
//! of those reads each vertex's frame and normal take. Both are structural copies of `geom`'s
//! internals, exposed for the spike, with no behaviour of their own.

use super::{EvalError, PositionSource, block_of, gather};
use crate::frame::{Stencil, stencil};
use crate::subdiv::CornerRing;
use crate::{BaseMesh, Region, Vec3};

/// A block of one level's positions with its one-vertex ring, and the rings of the base vertices
/// at its corners: what evaluation gathers.
#[derive(Debug, Clone, PartialEq)]
pub struct GatheredBlock {
    /// Quads per side of the block, without the ring.
    quads: u32,
    /// The `(quads + 3)²` positions, row-major from `(-1, -1)`; NaN where there is no vertex
    /// (diagonally off a face corner).
    padded: Vec<Vec3>,
    /// Per block corner (`(0, 0)`, `(q, 0)`, `(q, q)`, `(0, q)`), the ring of the base vertex
    /// there, if the corner is one.
    corners: [Option<GatheredRing>; 4],
}

impl GatheredBlock {
    /// Quads per side of the block, without the ring.
    #[must_use]
    pub const fn quads(&self) -> u32 {
        self.quads
    }

    /// The `(quads + 3)²` positions, row-major from `(-1, -1)`: block vertex `(i, j)` is at
    /// `(j + 1) · (quads + 3) + i + 1`. NaN where there is no vertex.
    #[must_use]
    pub fn padded(&self) -> &[Vec3] {
        &self.padded
    }

    /// The ring of the base vertex at block corner `c` (`0..4`: `(0, 0)`, `(q, 0)`, `(q, q)`,
    /// `(0, q)`), or `None` if that corner isn't a base vertex.
    #[must_use]
    pub fn corner(&self, c: usize) -> Option<&GatheredRing> {
        self.corners.get(c)?.as_ref()
    }
}

/// The one-ring of a base vertex at one level, owner first: quad `k` is
/// `(center, edges[k], diagonals[k], edges[k + 1])`.
#[derive(Debug, Clone, PartialEq)]
pub struct GatheredRing {
    /// The base vertex, as the block holds it.
    center: Vec3,
    /// The edge neighbours, in ring order.
    edges: Vec<Vec3>,
    /// The diagonal neighbours, in ring order.
    diagonals: Vec<Vec3>,
    /// The owner face's corner at this vertex (`0..4`).
    owner_corner: u8,
}

impl GatheredRing {
    /// The base vertex.
    #[must_use]
    pub const fn center(&self) -> Vec3 {
        self.center
    }

    /// The edge neighbours, in ring order, owner first.
    #[must_use]
    pub fn edges(&self) -> &[Vec3] {
        &self.edges
    }

    /// The diagonal neighbours, in ring order, owner first.
    #[must_use]
    pub fn diagonals(&self) -> &[Vec3] {
        &self.diagonals
    }

    /// The owner face's corner at this vertex (`0..4`): which ring points lie along the
    /// owner's axes.
    #[must_use]
    pub const fn owner_corner(&self) -> u8 {
        self.owner_corner
    }
}

impl From<&CornerRing> for GatheredRing {
    fn from(ring: &CornerRing) -> Self {
        Self {
            center: ring.center,
            edges: ring.edges().to_vec(),
            diagonals: ring.diagonals().to_vec(),
            owner_corner: ring.owner_corner,
        }
    }
}

/// Gathers `region` at its own level from `src`, as evaluation does: the block, its one-vertex
/// ring and its corners' base-vertex rings, every position read from its owner copy.
///
/// Gathering a tile's parent region gives what predicting the tile reads; gathering the tile's
/// own region (of evaluated positions) gives what its normals read.
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
        quads: region.quads(),
        padded: parent.padded.as_slice().to_vec(),
        corners: parent
            .corners
            .each_ref()
            .map(|ring| ring.as_ref().map(GatheredRing::from)),
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

/// The stencil of block vertex `(i, j)` of `region`: structural, from the vertex's place on the
/// face and the mesh's topology, never from positions.
#[must_use]
pub fn vertex_stencil(mesh: &BaseMesh, region: Region, i: u32, j: u32) -> VertexStencil {
    let step = |(a, b): (i64, i64)| {
        [
            i32::try_from(a).unwrap_or(i32::MAX),
            i32::try_from(b).unwrap_or(i32::MAX),
        ]
    };
    match stencil(&block_of(mesh, region), i64::from(i), i64::from(j)) {
        Stencil::Corner(c) => VertexStencil::Corner(u8::try_from(c).unwrap_or(u8::MAX)),
        Stencil::Axes([u, v]) => VertexStencil::Axes([step(u), step(v)]),
        Stencil::Outside => VertexStencil::Outside,
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::{CorrectionMap, DenseEvaluation, Level, cube_sphere};

    /// A cube sphere evaluated to level 2 without corrections, and one of its faces.
    fn sphere() -> (BaseMesh, DenseEvaluation) {
        let mesh = cube_sphere(1).expect("a valid mesh");
        let level = Level::new(2).expect("a valid level");
        let dense = DenseEvaluation::new(&mesh, level, &CorrectionMap::new()).expect("evaluates");
        (mesh, dense)
    }

    #[test]
    fn a_gathered_block_holds_the_padded_grid_and_the_corner_rings() {
        let (mesh, dense) = sphere();
        let face = mesh.faces().next().expect("a face");
        let level = Level::new(2).expect("a valid level");
        let region = Region::whole_face(face, level);
        let block = gather_block(&mesh, &dense, region).expect("gathers");
        let side = region.quads() as usize + 3;
        assert_eq!(block.padded().len(), side * side, "(quads + 3)² slots");
        let grid = dense.grid(face, level).expect("the face's grid");
        for j in 0..=region.quads() {
            for i in 0..=region.quads() {
                let slot = block.padded()[(j as usize + 1) * side + i as usize + 1];
                assert_eq!(
                    slot.to_bits(),
                    grid.get(i, j).expect("in the grid").to_bits(),
                    "({i}, {j}) is the face's own vertex"
                );
            }
        }
        assert!(
            block.padded()[0].x.is_nan(),
            "nothing diagonally off a face corner"
        );
        for c in 0..4 {
            let ring = block.corner(c).expect("every face corner is a base vertex");
            assert_eq!(ring.edges().len(), 3, "cube corners have valence 3");
            assert_eq!(ring.diagonals().len(), 3, "a diagonal per quad");
        }
        let corner = block.corner(0).expect("corner 0");
        assert_eq!(
            corner.center().to_bits(),
            block.padded()[side + 1].to_bits(),
            "the centre is the block's own corner"
        );
        assert!(block.corner(4).is_none(), "four corners only");
    }

    #[test]
    fn stencils_name_corners_borders_and_the_inside() {
        let (mesh, _) = sphere();
        let face = mesh.faces().next().expect("a face");
        let region = Region::whole_face(face, Level::new(2).expect("a valid level"));
        let q = region.quads();
        assert_eq!(
            vertex_stencil(&mesh, region, 0, 0),
            VertexStencil::Corner(0),
            "block corner 0"
        );
        assert_eq!(
            vertex_stencil(&mesh, region, q, q),
            VertexStencil::Corner(2),
            "block corner 2"
        );
        assert_eq!(
            vertex_stencil(&mesh, region, 1, 1),
            VertexStencil::Axes([[1, 0], [0, 1]]),
            "inside the face, the block's own axes"
        );
        assert!(
            matches!(vertex_stencil(&mesh, region, 1, 0), VertexStencil::Axes(_)),
            "on the face's border, some owner's axes"
        );
    }
}
