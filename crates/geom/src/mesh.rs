//! The base mesh: a closed, consistently oriented, manifold quad mesh.

use thiserror::Error;

use crate::topology::{Topology, corner_at};
use crate::{FaceId, Level, PositionSource, Vec3, VertexId};

/// The most faces around one base vertex.
///
/// Prediction keeps a base vertex's ring in fixed arrays of this size, so evaluation doesn't
/// allocate per corner; [`BaseMesh::from_polygons`] rejects a vertex with more.
pub const MAX_VALENCE: usize = 32;

/// A closed quad base mesh with its adjacency.
#[derive(Debug, Clone)]
pub struct BaseMesh {
    /// Level 0's vertex positions, by vertex index.
    positions: Vec<Vec3>,
    /// Each face's four vertex indices, counter-clockwise seen from outside: corners 0–3, at
    /// grid positions `(0, 0)`, `(n, 0)`, `(n, n)` and `(0, n)`.
    faces: Vec<[u32; 4]>,
    /// How the faces fit together, built once from `faces`.
    topology: Topology,
}

/// Why a base mesh was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
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
        /// The vertex.
        vertex: VertexId,
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
    NotFinite(VertexId),
    /// An edge is used twice in the same direction.
    #[error(
        "edge {from}→{to} is used twice in the same direction (non-manifold, or a flipped face)"
    )]
    DuplicateEdge {
        /// Start vertex.
        from: VertexId,
        /// End vertex.
        to: VertexId,
    },
    /// An edge has no face on its other side.
    #[error(
        "edge {from}→{to} has no neighbour; base meshes must be closed (open meshes arrive in M2)"
    )]
    OpenEdge {
        /// Start vertex.
        from: VertexId,
        /// End vertex.
        to: VertexId,
    },
    /// A vertex no face uses.
    #[error("vertex {0} is not on any face")]
    IsolatedVertex(VertexId),
    /// The faces around a vertex don't form one fan.
    #[error("the faces around vertex {0} don't form a single fan")]
    NonManifoldVertex(VertexId),
    /// A vertex with fewer than three faces around it, as in a two-quad pillow. Subdivision
    /// stencils and tangent frames need valence 3 or more.
    #[error("vertex {0} has fewer than three faces around it")]
    LowValence(VertexId),
    /// A vertex with more than [`MAX_VALENCE`] faces around it.
    #[error("vertex {0} has more than {MAX_VALENCE} faces around it")]
    HighValence(VertexId),
    /// The faces are consistently oriented, but inward: the enclosed volume is not positive.
    #[error("the mesh is inside out (its faces point inward) or encloses no volume")]
    InsideOut,
    /// More faces or vertices than 32-bit ids hold.
    #[error("too many faces or vertices for 32-bit ids")]
    TooLarge,
    /// Bytes given to [`BaseMesh::from_bytes`] aren't a mesh's canonical encoding: too short for
    /// the counts they state, or with bytes left over.
    #[error("the bytes are not a canonical base mesh encoding")]
    Malformed,
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
    /// Any [`MeshError`]: non-quads, bad indices, degenerate faces, non-finite positions, low
    /// or high valence, or a mesh that is not closed, manifold and consistently oriented outward.
    pub fn from_polygons<P: AsRef<[u32]>>(
        positions: Vec<Vec3>,
        polygons: &[P],
    ) -> Result<Self, MeshError> {
        if polygons.is_empty() {
            return Err(MeshError::Empty);
        }
        let vertex_count: u32 = fit(positions.len())?;
        if let Some(bad) = (0..vertex_count)
            .zip(&positions)
            .find_map(|(v, p)| (!p.is_finite()).then_some(v))
        {
            return Err(MeshError::NotFinite(VertexId::new(bad)));
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
                if let Some(&vertex) = quad.iter().find(|&&v| v >= vertex_count) {
                    return Err(MeshError::VertexOutOfRange {
                        face,
                        vertex: VertexId::new(vertex),
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
        let topology = Topology::build(&faces, vertex_count)?;
        let mesh = Self {
            positions,
            faces,
            topology,
        };
        if mesh.signed_volume() <= 0.0 {
            return Err(MeshError::InsideOut);
        }
        Ok(mesh)
    }

    /// The mesh's canonical bytes: what a project file stores, and what its mesh hash is computed
    /// over.
    ///
    /// The layout, all little-endian: the vertex count as a `u32`, each vertex's position as three
    /// `f32`s (x, y, z), the face count as a `u32`, then each face's four vertex indices as `u32`s,
    /// corners 0–3. Vertices and faces keep their ids, so [`BaseMesh::from_bytes`] gives back the
    /// same mesh and encoding it again gives the same bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.positions.len() * 12 + self.faces.len() * 16);
        out.extend_from_slice(&self.vertex_count().to_le_bytes());
        for p in &self.positions {
            for c in [p.x, p.y, p.z] {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.face_count().to_le_bytes());
        for v in self.faces.iter().flatten() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Decodes [`BaseMesh::to_bytes`]'s encoding and validates the mesh as
    /// [`BaseMesh::from_polygons`] does.
    ///
    /// Each count is checked against the bytes that remain before anything is allocated for it,
    /// so hostile input can't make this allocate more than a small multiple of its own length.
    ///
    /// # Errors
    ///
    /// [`MeshError::Malformed`] if the bytes are shorter than their counts need or longer than
    /// the encoding; then any [`MeshError`] of [`BaseMesh::from_polygons`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MeshError> {
        let (vertex_count, rest) = split_count(bytes)?;
        let (position_words, rest) = split_words(rest, vertex_count, 3)?;
        let (face_count, rest) = split_count(rest)?;
        let (face_words, rest) = split_words(rest, face_count, 4)?;
        if !rest.is_empty() {
            return Err(MeshError::Malformed);
        }
        let floats: Vec<f32> = position_words
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect();
        let (xyz, _) = floats.as_chunks::<3>();
        let positions = xyz.iter().map(|&[x, y, z]| Vec3::new(x, y, z)).collect();
        let indices: Vec<u32> = face_words
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect();
        let (quads, _) = indices.as_chunks::<4>();
        Self::from_polygons(positions, quads)
    }

    /// Six times the enclosed volume, in `f64`: positive when the faces point outward. Each quad
    /// contributes the two triangles `(p0, p1, p2)` and `(p0, p2, p3)` as tetrahedra with the
    /// origin.
    fn signed_volume(&self) -> f64 {
        let wide = |p: Vec3| [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
        let triple = |[ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3], [cx, cy, cz]: [f64; 3]| {
            ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
        };
        self.faces()
            .filter_map(|f| {
                let [p0, p1, p2, p3] = [0, 1, 2, 3].map(|c| self.corner_position(f, c).map(wide));
                Some(triple(p0?, p1?, p2?) + triple(p0?, p2?, p3?))
            })
            .sum()
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

    /// Every vertex id.
    pub fn vertices(&self) -> impl Iterator<Item = VertexId> {
        (0..self.vertex_count()).map(VertexId::new)
    }

    /// A vertex's position, or `None` if `vertex` is not in this mesh.
    #[must_use]
    pub fn vertex_position(&self, vertex: VertexId) -> Option<Vec3> {
        self.positions
            .get(usize::try_from(vertex.index()).ok()?)
            .copied()
    }

    /// The vertex at corner `corner` (0–3) of `face`, or `None` if either is out of range.
    #[must_use]
    pub fn corner_vertex(&self, face: FaceId, corner: u8) -> Option<VertexId> {
        let quad = self.faces.get(usize::try_from(face.index()).ok()?)?;
        quad.get(usize::from(corner)).copied().map(VertexId::new)
    }

    /// The position of corner `corner` (0–3) of `face`, or `None` if either is out of range.
    #[must_use]
    pub fn corner_position(&self, face: FaceId, corner: u8) -> Option<Vec3> {
        self.vertex_position(self.corner_vertex(face, corner)?)
    }

    /// Adjacency.
    #[must_use]
    pub const fn topology(&self) -> &Topology {
        &self.topology
    }
}

/// Splits a little-endian `u32` count off the front of `bytes`.
///
/// # Errors
///
/// [`MeshError::Malformed`] if fewer than four bytes are left; [`MeshError::TooLarge`] if the
/// count doesn't fit in a `usize`.
fn split_count(bytes: &[u8]) -> Result<(usize, &[u8]), MeshError> {
    let (word, rest) = bytes.split_first_chunk::<4>().ok_or(MeshError::Malformed)?;
    Ok((fit(u32::from_le_bytes(*word))?, rest))
}

/// Splits `count` items of `words` four-byte words each off the front of `bytes`, as words.
///
/// The length is checked against `bytes` before anything is read, so a count far larger than
/// the input fails at once.
///
/// # Errors
///
/// [`MeshError::Malformed`] if `bytes` is shorter than the items need.
fn split_words(bytes: &[u8], count: usize, words: usize) -> Result<(&[[u8; 4]], &[u8]), MeshError> {
    let len = count.checked_mul(words * 4).ok_or(MeshError::Malformed)?;
    let (items, rest) = bytes.split_at_checked(len).ok_or(MeshError::Malformed)?;
    let (items, _) = items.as_chunks::<4>();
    Ok((items, rest))
}

/// Level 0: the base mesh's corners, `(0, 0)`, `(1, 0)`, `(1, 1)`, `(0, 1)` being corners 0–3.
/// Every other level, and anything outside the base quad, is `None`.
impl PositionSource for BaseMesh {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        if level != Level::ZERO || i > 1 || j > 1 {
            return None;
        }
        self.corner_position(face, corner_at(i, j, 1))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::cube_sphere;
    use crate::reference::prism;
    use crate::topology::Across;

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

    #[test]
    fn rejects_an_inside_out_mesh() {
        let (v, mut f) = unit_cube();
        for quad in &mut f {
            quad.reverse();
        }
        assert_eq!(
            BaseMesh::from_polygons(v, &f).unwrap_err(),
            MeshError::InsideOut,
            "all faces reversed"
        );
    }

    #[test]
    fn rejects_a_two_quad_pillow() {
        let (v, _) = unit_cube();
        let pillow = [[0u32, 1, 2, 3], [3, 2, 1, 0]];
        assert!(
            matches!(
                BaseMesh::from_polygons(v[..4].to_vec(), &pillow),
                Err(MeshError::LowValence(_))
            ),
            "valence-2 vertices"
        );
    }

    #[test]
    fn accepts_max_valence_and_rejects_more() {
        let mesh = prism(MAX_VALENCE).subdivide().to_base_mesh();
        let valence = mesh
            .faces()
            .flat_map(|f| (0..4).map(move |c| (f, c)))
            .map(|(f, c)| mesh.topology().ring(f, c).unwrap().entries.len())
            .max();
        assert_eq!(valence, Some(MAX_VALENCE), "the caps' centres");
        let Err(MeshError::HighValence(v)) = prism(MAX_VALENCE + 1).subdivide().try_to_base_mesh()
        else {
            panic!("valence {} is rejected", MAX_VALENCE + 1);
        };
        // Vertex `v` is not in a mesh, so find it among the polygon's positions: a cap centre.
        let poly = prism(MAX_VALENCE + 1).subdivide();
        let p = poly.positions[usize::try_from(v.index()).unwrap()];
        assert!(
            p[0].abs() < 1e-9 && p[1].abs() < 1e-9,
            "{v} at {p:?} is a cap centre"
        );
    }

    #[test]
    fn rejects_bad_indices_positions_and_connectivity() {
        let (v, f) = unit_cube();
        assert_eq!(
            BaseMesh::from_polygons(v.clone(), &[[0u32; 4]; 0]).unwrap_err(),
            MeshError::Empty,
            "no faces"
        );

        let mut out_of_range = f.clone();
        out_of_range[1] = [4, 5, 6, 8];
        assert_eq!(
            BaseMesh::from_polygons(v.clone(), &out_of_range).unwrap_err(),
            MeshError::VertexOutOfRange {
                face: 1,
                vertex: VertexId::new(8),
                count: 8
            },
            "vertex 8 of 8"
        );

        let mut nan = v.clone();
        nan[5] = Vec3::NAN;
        assert_eq!(
            BaseMesh::from_polygons(nan, &f).unwrap_err(),
            MeshError::NotFinite(VertexId::new(5)),
            "NaN position"
        );

        let mut extra = v.clone();
        extra.push(Vec3::ZERO);
        assert_eq!(
            BaseMesh::from_polygons(extra, &f).unwrap_err(),
            MeshError::IsolatedVertex(VertexId::new(8)),
            "unused vertex"
        );

        // Two cubes touching at one vertex: cube B's corner 0 is cube A's corner 6.
        let mut pinched = v.clone();
        pinched.extend(v.iter().skip(1).map(|&p| p + Vec3::new(2.0, 2.0, 2.0)));
        let b = |k: u32| if k == 0 { 6 } else { k + 7 };
        let mut faces = f.clone();
        faces.extend(f.iter().map(|q| q.map(b)));
        assert_eq!(
            BaseMesh::from_polygons(pinched, &faces).unwrap_err(),
            MeshError::NonManifoldVertex(VertexId::new(6)),
            "two fans at one vertex"
        );

        // A third face on edge 0→3, which face 0 already uses in that direction.
        let mut fin = v;
        fin.extend([Vec3::new(-2.0, -1.0, -2.0), Vec3::new(-2.0, 1.0, -2.0)]);
        let mut finned = f;
        finned.push([0, 3, 9, 8]);
        assert!(
            matches!(
                BaseMesh::from_polygons(fin, &finned),
                Err(MeshError::DuplicateEdge { .. })
            ),
            "three faces on one edge"
        );
    }

    #[test]
    fn rejects_out_of_range_corners() {
        let (v, f) = unit_cube();
        let mesh = BaseMesh::from_polygons(v, &f).unwrap();
        let face = FaceId::new(0);
        assert!(mesh.corner_position(face, 3).is_some(), "corner 3 exists");
        assert_eq!(mesh.corner_position(face, 4), None, "no corner 4");
        assert_eq!(mesh.corner_position(FaceId::new(6), 0), None, "no face 6");
        assert!(
            mesh.topology().ring(face, 4).is_none(),
            "no ring at corner 4"
        );
        assert_eq!(mesh.topology().across(face, 4), None, "no edge 4");
    }

    /// Checks every ring of `mesh`: it starts at its lowest face id, steps across each entry's
    /// incoming edge to the next (cyclically), and every entry's corner is the ring's vertex.
    fn assert_rings_are_canonical(mesh: &BaseMesh) {
        let t = mesh.topology();
        for face in mesh.faces() {
            for corner in 0..4 {
                let vertex = mesh.corner_vertex(face, corner).unwrap();
                let ring = &t.ring(face, corner).unwrap().entries;
                let lowest = ring.iter().map(|&(g, _)| g).min().unwrap();
                assert_eq!(
                    ring[0].0, lowest,
                    "ring of vertex {vertex} starts at its owner"
                );
                assert!(
                    ring.contains(&(face, corner)),
                    "ring of vertex {vertex} holds ({face:?}, {corner})"
                );
                for (k, &(f, c)) in ring.iter().enumerate() {
                    let (g, m) = ring[(k + 1) % ring.len()];
                    assert_eq!(
                        t.across(f, (c + 3) & 3),
                        Some(Across { face: g, edge: m }),
                        "vertex {vertex}: ({f:?}, {c}) steps to ({g:?}, {m})"
                    );
                    assert_eq!(
                        mesh.corner_vertex(g, m),
                        Some(vertex),
                        "entry ({g:?}, {m}) is at vertex {vertex}"
                    );
                    assert_eq!(
                        mesh.corner_position(g, m),
                        mesh.vertex_position(vertex),
                        "entry ({g:?}, {m}) is at vertex {vertex}'s position"
                    );
                }
            }
        }
    }

    #[test]
    fn level_zero_positions_are_the_face_corners() {
        let mesh = cube_sphere(2).unwrap();
        let at = |face, level, i, j| mesh.position(face, level, i, j);
        for face in mesh.faces() {
            for (c, (i, j)) in [(0, (0, 0)), (1, (1, 0)), (2, (1, 1)), (3, (0, 1))] {
                assert_eq!(
                    at(face, Level::ZERO, i, j),
                    mesh.corner_position(face, c),
                    "{face} corner {c} at ({i}, {j})"
                );
            }
            assert_eq!(at(face, Level::ZERO, 2, 0), None, "outside the base quad");
            assert_eq!(
                at(face, Level::new(1).unwrap(), 0, 0),
                None,
                "level 1 is not the mesh"
            );
        }
        assert_eq!(
            at(FaceId::new(24), Level::ZERO, 0, 0),
            None,
            "not a face of this mesh"
        );
    }

    #[test]
    fn canonical_bytes_round_trip() {
        for mesh in [cube_sphere(1).unwrap(), cube_sphere(3).unwrap()] {
            let bytes = mesh.to_bytes();
            assert_eq!(
                bytes.len(),
                8 + 12 * mesh.positions.len() + 16 * mesh.faces.len(),
                "the documented layout"
            );
            let back = BaseMesh::from_bytes(&bytes).unwrap();
            assert_eq!(back.positions, mesh.positions, "positions, bit for bit");
            assert_eq!(back.faces, mesh.faces, "faces, in order");
            assert_eq!(
                back.to_bytes(),
                bytes,
                "encoding again gives the same bytes"
            );
        }
    }

    #[test]
    fn hostile_bytes_are_rejected_without_panicking_or_overallocating() {
        let good = cube_sphere(1).unwrap().to_bytes();
        let vertex_count = u32::from_le_bytes(good[..4].try_into().unwrap());
        let faces_at = 4 + 12 * usize::try_from(vertex_count).unwrap();
        // Counts promising far more than the input holds: refused before any allocation.
        for (what, at, count) in [
            ("vertices", 0, u32::MAX),
            ("vertices, one too many", 0, vertex_count + 1),
            ("faces", faces_at, u32::MAX),
        ] {
            let mut bad = good.clone();
            bad[at..at + 4].copy_from_slice(&count.to_le_bytes());
            assert_eq!(
                BaseMesh::from_bytes(&bad).unwrap_err(),
                MeshError::Malformed,
                "{what}"
            );
        }
        assert_eq!(
            BaseMesh::from_bytes(&[good.as_slice(), &[0]].concat()).unwrap_err(),
            MeshError::Malformed,
            "a trailing byte"
        );
        for cut in 0..good.len() {
            assert!(BaseMesh::from_bytes(&good[..cut]).is_err(), "cut at {cut}");
        }
        // Well-formed bytes that aren't a valid mesh: the mesh's own checks catch them.
        let mut nan = good.clone();
        nan[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            BaseMesh::from_bytes(&nan).unwrap_err(),
            MeshError::NotFinite(VertexId::new(0)),
            "a NaN position"
        );
        let mut out_of_range = good;
        out_of_range[faces_at + 4..faces_at + 8].copy_from_slice(&vertex_count.to_le_bytes());
        assert!(
            matches!(
                BaseMesh::from_bytes(&out_of_range),
                Err(MeshError::VertexOutOfRange { face: 0, .. })
            ),
            "an index past the vertices"
        );
        // Noise of every short length, from a fixed seed.
        let mut rng = crate::test_rng::Rng::new(11);
        for len in 0..512 {
            let noise: Vec<u8> = (0..len).map(|_| rng.next_u64().to_le_bytes()[0]).collect();
            assert!(
                BaseMesh::from_bytes(&noise).is_err(),
                "noise of {len} bytes"
            );
        }
    }

    #[test]
    fn rings_start_at_the_owner_and_walk_across_incoming_edges() {
        let (v, f) = unit_cube();
        assert_rings_are_canonical(&BaseMesh::from_polygons(v, &f).unwrap());
        assert_rings_are_canonical(&cube_sphere(3).unwrap());
    }
}
