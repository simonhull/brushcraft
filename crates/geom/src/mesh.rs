//! The base mesh: a closed, consistently oriented, manifold quad mesh.

use thiserror::Error;

use crate::topology::Topology;
use crate::{FaceId, Vec3};

/// A closed quad base mesh with its adjacency.
#[derive(Debug, Clone)]
pub struct BaseMesh {
    positions: Vec<Vec3>,
    faces: Vec<[u32; 4]>,
    topology: Topology,
}

/// Why a base mesh was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MeshError {
    /// No faces.
    #[error("the mesh has no faces")]
    Empty,
    /// M0 base meshes are quads only.
    #[error("face {face} has {sides} sides; base meshes are quads")]
    NotQuad {
        /// Face index.
        face: usize,
        /// Its number of sides.
        sides: usize,
    },
    /// A face uses a vertex that doesn't exist.
    #[error("face {face} uses vertex {vertex}, but there are only {count} vertices")]
    VertexOutOfRange {
        /// Face index.
        face: usize,
        /// The vertex index.
        vertex: u32,
        /// Number of vertices.
        count: usize,
    },
    /// A face repeats a vertex.
    #[error("face {face} repeats a vertex")]
    DegenerateFace {
        /// Face index.
        face: usize,
    },
    /// A position is NaN or infinite.
    #[error("vertex {0} is not finite")]
    NotFinite(usize),
    /// An edge is used twice in the same direction.
    #[error(
        "edge {from}→{to} is used twice in the same direction (non-manifold, or a flipped face)"
    )]
    DuplicateEdge {
        /// Start vertex.
        from: u32,
        /// End vertex.
        to: u32,
    },
    /// An edge has no face on its other side.
    #[error(
        "edge {from}→{to} has no neighbour; base meshes must be closed (open meshes arrive in M2)"
    )]
    OpenEdge {
        /// Start vertex.
        from: u32,
        /// End vertex.
        to: u32,
    },
    /// A vertex no face uses.
    #[error("vertex {0} is not on any face")]
    IsolatedVertex(u32),
    /// The faces around a vertex don't form one fan.
    #[error("the faces around vertex {0} don't form a single fan")]
    NonManifoldVertex(u32),
    /// More faces or vertices than 32-bit ids hold.
    #[error("too many faces or vertices for 32-bit ids")]
    TooLarge,
}

/// Converts an id or count between integer types, failing with [`MeshError::TooLarge`] when it
/// doesn't fit.
///
/// # Errors
///
/// [`MeshError::TooLarge`] if `value` is out of range for `T`.
pub(crate) fn fit<T: TryFrom<U>, U>(value: U) -> Result<T, MeshError> {
    T::try_from(value).ok().ok_or(MeshError::TooLarge)
}

impl BaseMesh {
    /// Builds a base mesh from positions and polygons (vertex indices, counter-clockwise seen from
    /// outside).
    ///
    /// # Errors
    ///
    /// Any [`MeshError`]: non-quads, bad indices, degenerate faces, non-finite positions, or a mesh
    /// that is not closed, manifold and consistently oriented.
    pub fn from_polygons<P: AsRef<[u32]>>(
        positions: Vec<Vec3>,
        polygons: &[P],
    ) -> Result<Self, MeshError> {
        if polygons.is_empty() {
            return Err(MeshError::Empty);
        }
        if let Some(bad) = positions.iter().position(|p| !p.is_finite()) {
            return Err(MeshError::NotFinite(bad));
        }
        let count = positions.len();
        let faces = polygons
            .iter()
            .enumerate()
            .map(|(face, p)| {
                let p = p.as_ref();
                let Ok(quad) = <[u32; 4]>::try_from(p) else {
                    return Err(MeshError::NotQuad {
                        face,
                        sides: p.len(),
                    });
                };
                if let Some(&vertex) = quad
                    .iter()
                    .find(|&&v| usize::try_from(v).map_or(true, |v| v >= count))
                {
                    return Err(MeshError::VertexOutOfRange {
                        face,
                        vertex,
                        count,
                    });
                }
                let distinct = quad
                    .iter()
                    .enumerate()
                    .all(|(a, v)| !quad.iter().skip(a + 1).any(|w| w == v));
                if distinct {
                    Ok(quad)
                } else {
                    Err(MeshError::DegenerateFace { face })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Face ids are `u32`, so the face count must fit one too; `face_count` relies on it.
        fit::<u32, _>(faces.len())?;
        let vertex_count = fit(count)?;
        let topology = Topology::build(&faces, vertex_count)?;
        Ok(Self {
            positions,
            faces,
            topology,
        })
    }

    /// Number of faces.
    #[must_use]
    pub fn face_count(&self) -> u32 {
        // `from_polygons` checked that the count fits.
        u32::try_from(self.faces.len()).unwrap_or(u32::MAX)
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> u32 {
        // `from_polygons` checked that the count fits.
        u32::try_from(self.positions.len()).unwrap_or(u32::MAX)
    }

    /// Every face id.
    pub fn faces(&self) -> impl Iterator<Item = FaceId> {
        (0..self.face_count()).map(FaceId::new)
    }

    /// A vertex's position.
    #[must_use]
    pub fn position(&self, vertex: u32) -> Option<Vec3> {
        self.positions.get(usize::try_from(vertex).ok()?).copied()
    }

    /// The position of corner `corner` of `face`.
    #[must_use]
    pub fn corner_position(&self, face: FaceId, corner: u8) -> Option<Vec3> {
        let quad = self.faces.get(usize::try_from(face.index()).ok()?)?;
        self.position(*quad.get(usize::from(corner & 3))?)
    }

    /// Adjacency.
    #[must_use]
    pub const fn topology(&self) -> &Topology {
        &self.topology
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let v = vec![
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            p(-1.0, -1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
        ];
        // Outward-facing, counter-clockwise seen from outside.
        let f = vec![
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        (v, f)
    }

    #[test]
    fn cube_builds_with_valence_three_rings() {
        let (v, f) = unit_cube();
        let mesh = BaseMesh::from_polygons(v, &f).unwrap();
        assert_eq!(mesh.face_count(), 6, "faces");
        for face in mesh.faces() {
            for corner in 0..4 {
                assert_eq!(
                    mesh.topology().ring(face, corner).unwrap().entries.len(),
                    3,
                    "cube corners: valence 3"
                );
            }
        }
    }

    #[test]
    fn rejects_bad_meshes() {
        let (v, mut f) = unit_cube();
        assert!(
            matches!(
                BaseMesh::from_polygons(v.clone(), &[vec![0u32, 1, 2]]),
                Err(MeshError::NotQuad { face: 0, sides: 3 })
            ),
            "triangle"
        );
        assert!(
            matches!(
                BaseMesh::from_polygons(v.clone(), &f[..5]),
                Err(MeshError::OpenEdge { .. })
            ),
            "open"
        );
        f[0] = [0, 1, 2, 3]; // flipped
        assert!(
            matches!(
                BaseMesh::from_polygons(v.clone(), &f),
                Err(MeshError::DuplicateEdge { .. })
            ),
            "flipped face"
        );
        assert!(
            matches!(
                BaseMesh::from_polygons(v, &[[0u32, 0, 1, 2]]),
                Err(MeshError::DegenerateFace { face: 0 })
            ),
            "repeated vertex"
        );
    }
}
