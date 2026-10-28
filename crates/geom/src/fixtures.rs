//! Meshes for tests: this crate's, and those of the crates above it, which turn on the
//! `test-fixtures` feature in their dev-dependencies. Release builds never compile it.

use std::f64::consts::TAU;

use crate::{BaseMesh, MeshError, Vec3};

/// A point in `f64`, as the subdivision below computes it.
type P = [f64; 3];

/// The mean of `points`.
fn mean<const N: usize>(points: [P; N]) -> P {
    let sum = points
        .iter()
        .fold([0.0; 3], |[x, y, z], &[a, b, c]| [x + a, y + b, z + c]);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a mean of a handful of points: N is exact in an f64"
    )]
    let n = N as f64;
    sum.map(|c| c / n)
}

/// `p` rounded to `f32`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the prism's points lie within a unit of the origin, well inside f32's range"
)]
const fn to_vec3([x, y, z]: P) -> Vec3 {
    Vec3::new(x as f32, y as f32, z as f32)
}

/// A pentagonal prism after one Catmull–Clark step: a closed quad mesh with two vertices of
/// valence 5.
///
/// The prism is two regular pentagons of circumradius 1 at `z = ±1`, joined by five quads. One
/// step makes it 30 quads over 32 vertices, the caps' centres of valence 5 and every other
/// vertex of valence 3 or 4: the smallest mesh the tests have with a vertex of valence other
/// than 3 and 4. The points are the textbook step's (`geom`'s own tests check them against an
/// independent reference), and are numbered so:
///
/// - vertices: the bottom ring's vertex points `0..5` and the top ring's `5..10`; the edge points
///   of the bottom ring's edges `10..15`, the top ring's `15..20` and the upright edges `20..25`;
///   the face points of the bottom cap (25), the top cap (26) and the sides `27..32`.
/// - faces: the bottom cap's five quads `0..5`, the top cap's `5..10`, then four for each side.
///   Face 5's corner 2 is the top cap's centre, vertex 26; face 0's corner 2 the bottom cap's.
///
/// # Errors
///
/// Never: the mesh is a valid base mesh. The [`MeshError`] is [`BaseMesh::from_polygons`]'s.
pub fn subdivided_pentagonal_prism() -> Result<BaseMesh, MeshError> {
    subdivided_prism(5)
}

/// A prism over a regular `sides`-gon after one Catmull–Clark step (see
/// [`subdivided_pentagonal_prism`] for the numbering, with `sides` for 5).
///
/// # Errors
///
/// As [`BaseMesh::from_polygons`]: for fewer than 3 sides, or more than a vertex's valence
/// allows.
fn subdivided_prism(sides: u32) -> Result<BaseMesh, MeshError> {
    let next = |k: u32| (k + 1) % sides;
    let prev = |k: u32| (k + sides - 1) % sides;
    // The prism's own vertices, faces and edges, then the step's points from them.
    let ring = |k: u32, z: f64| {
        let angle = TAU * f64::from(k) / f64::from(sides);
        [angle.cos(), angle.sin(), z]
    };
    let cap = |z: f64| {
        let sum = (0..sides)
            .map(|k| ring(k, z))
            .fold([0.0; 3], |[x, y, w], [a, b, c]| [x + a, y + b, w + c]);
        sum.map(|c| c / f64::from(sides))
    };
    let side = |k: u32| {
        mean([
            ring(k, -1.0),
            ring(next(k), -1.0),
            ring(next(k), 1.0),
            ring(k, 1.0),
        ])
    };
    let ring_edge = |k: u32, z: f64| mean([ring(k, z), ring(next(k), z), cap(z), side(k)]);
    let upright = |k: u32| mean([ring(k, -1.0), ring(k, 1.0), side(prev(k)), side(k)]);
    // A ring vertex has valence 3: its point is (Q + 2R) / 3, Q the mean of its faces' points
    // and R the mean of its edges' midpoints.
    let corner = |k: u32, z: f64| {
        let v = ring(k, z);
        let q = mean([cap(z), side(prev(k)), side(k)]);
        let r = mean([
            mean([v, ring(prev(k), z)]),
            mean([v, ring(next(k), z)]),
            mean([v, ring(k, -z)]),
        ]);
        mean([q, r, r])
    };

    let ks = || 0..sides;
    let positions: Vec<Vec3> = ks()
        .map(|k| corner(k, -1.0))
        .chain(ks().map(|k| corner(k, 1.0)))
        .chain(ks().map(|k| ring_edge(k, -1.0)))
        .chain(ks().map(|k| ring_edge(k, 1.0)))
        .chain(ks().map(upright))
        .chain([cap(-1.0), cap(1.0)])
        .chain(ks().map(side))
        .map(to_vec3)
        .collect();

    let s = sides;
    let (bottom, top) = (|k: u32| k, |k: u32| s + k);
    let (bottom_edge, top_edge, upright_edge) = (|k: u32| 2 * s + k, |k| 3 * s + k, |k| 4 * s + k);
    let (bottom_cap, top_cap, side_face) = (5 * s, 5 * s + 1, |k: u32| 5 * s + 2 + k);
    let faces: Vec<[u32; 4]> = ks()
        .map(|k| [bottom(k), bottom_edge(prev(k)), bottom_cap, bottom_edge(k)])
        .chain(ks().map(|k| [top(k), top_edge(k), top_cap, top_edge(prev(k))]))
        .chain(ks().flat_map(|k| {
            let (n, f) = (next(k), side_face(k));
            [
                [bottom(k), bottom_edge(k), f, upright_edge(k)],
                [bottom(n), upright_edge(n), f, bottom_edge(k)],
                [top(n), top_edge(k), f, upright_edge(n)],
                [top(k), upright_edge(k), f, top_edge(k)],
            ]
        }))
        .collect();
    BaseMesh::from_polygons(positions, &faces)
}

/// NaN in every component: a vertex [`ring_normals`] has no normal for.
const NO_NORMAL: Vec3 = Vec3::new(f32::NAN, f32::NAN, f32::NAN);

/// `n` at unit length: scaled by the reciprocal of its length where the length is finite and
/// above `f32::MIN_POSITIVE`; else `(0, 0, 1)` where the inputs were `finite`, else NaN.
fn unit_or_fallback(n: Vec3, finite: bool) -> Vec3 {
    let length = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
    if length.is_finite() && length > f32::MIN_POSITIVE {
        let r = 1.0 / length;
        Vec3::new(n.x * r, n.y * r, n.z * r)
    } else if finite {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        NO_NORMAL
    }
}

/// `a × b`, component by component.
const fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// `a − b`, component by component.
const fn minus(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// The normal at a base vertex from its ring: `Σ_k (e_k − v) × (e_{k+1} − v)` from the owner,
/// wrapping, at unit length; NaN for a ring that isn't finite or has fewer than two points.
fn ring_normal(ring: &crate::GatheredRing) -> Vec3 {
    let v = ring.center();
    let edges = ring.edges();
    let finite = v.is_finite() && edges.iter().all(|e| e.is_finite());
    if !finite || edges.len() < 2 {
        return NO_NORMAL;
    }
    let next = edges.iter().cycle().skip(1);
    let terms = edges
        .iter()
        .zip(next)
        .map(|(&a, &b)| cross(minus(a, v), minus(b, v)));
    let sum = terms.fold(Vec3::ZERO, |n, t| {
        Vec3::new(n.x + t.x, n.y + t.y, n.z + t.z)
    });
    unit_or_fallback(sum, true)
}

/// The unit normals of `block`'s vertices, row-major `(quads + 1)²`, as a GPU computes them.
///
/// One vertex at a time, from the block's layout alone: the arithmetic the notes of
/// [`gather_block`](crate::gather_block)'s module spell out, written out plainly, apart from
/// `geom`'s own normals code. The reference a shader computing normals from a finest tile and
/// its ring is checked against, and that `geom` and `store` check
/// [`GatheredBlock::normals`](crate::GatheredBlock::normals) and the store's normals against,
/// bit for bit.
///
/// `mesh` is the mesh the block was gathered over.
#[must_use]
pub fn ring_normals(mesh: &BaseMesh, block: &crate::GatheredBlock) -> Vec<Vec3> {
    use crate::{VertexStencil, vertex_stencil};

    let region = block.region();
    let q = region.quads();
    let vertices = (0..=q).flat_map(|j| (0..=q).map(move |i| (i, j)));
    vertices
        .map(|(i, j)| match vertex_stencil(mesh, region, i, j) {
            VertexStencil::Axes([[ui, uj], [vi, vj]]) => {
                let (i, j) = (i64::from(i), i64::from(j));
                let at = |di: i32, dj: i32| block.get(i + i64::from(di), j + i64::from(dj));
                let du = minus(at(ui, uj), at(-ui, -uj));
                let dv = minus(at(vi, vj), at(-vi, -vj));
                unit_or_fallback(cross(du, dv), du.is_finite() && dv.is_finite())
            }
            VertexStencil::Corner(c) => block.corner(usize::from(c)).map_or(NO_NORMAL, ring_normal),
            VertexStencil::Outside => NO_NORMAL,
        })
        .collect()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::reference::{Poly, assert_same_points, pentagonal_prism};
    use crate::{FaceId, VertexId};

    /// The corners of each face of `poly`, as points.
    fn face_points(poly: &Poly) -> Vec<Vec<P>> {
        poly.faces
            .iter()
            .map(|face| face.iter().map(|&v| poly.positions[v]).collect())
            .collect()
    }

    /// Whether `a` and `b` are the same face: the same points within `tol`, in the same
    /// cyclic order (so the same orientation), from any starting corner.
    fn same_face(a: &[P], b: &[P], tol: f64) -> bool {
        let close = |p: &P, q: &P| p.iter().zip(q).all(|(x, y)| (x - y).abs() <= tol);
        a.len() == b.len()
            && (0..b.len()).any(|shift| {
                a.iter()
                    .zip(b.iter().cycle().skip(shift))
                    .all(|(p, q)| close(p, q))
            })
    }

    #[test]
    fn the_subdivided_prism_is_the_reference_step() {
        let ours = subdivided_pentagonal_prism().unwrap();
        let reference = pentagonal_prism().subdivide();
        assert_eq!(
            (ours.vertex_count(), ours.face_count()),
            (32, 30),
            "V = 10 + 15 + 7, F = 2 · 5 + 5 · 4"
        );
        let points: Vec<Vec3> = ours
            .vertices()
            .map(|v| ours.vertex_position(v).unwrap())
            .collect();
        assert_same_points(&points, &reference.positions, 1e-6);
        let theirs = face_points(&reference);
        for face in face_points(&Poly::from_base_mesh(&ours)) {
            let matches = theirs
                .iter()
                .filter(|other| same_face(&face, other, 1e-6))
                .count();
            assert_eq!(matches, 1, "{face:?} is one of the reference's faces");
        }
    }

    #[test]
    fn the_caps_centres_are_where_the_docs_say() {
        let mesh = subdivided_pentagonal_prism().unwrap();
        assert_eq!(
            mesh.corner_vertex(FaceId::new(5), 2),
            Some(VertexId::new(26)),
            "the top cap's centre"
        );
        assert_eq!(
            mesh.corner_vertex(FaceId::new(0), 2),
            Some(VertexId::new(25)),
            "the bottom cap's centre"
        );
        let around_top = mesh
            .faces()
            .filter(|&f| (0..4).any(|c| mesh.corner_vertex(f, c) == Some(VertexId::new(26))))
            .count();
        assert_eq!(around_top, 5, "valence 5");
    }
}
