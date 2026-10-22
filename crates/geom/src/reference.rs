//! Test-only reference Catmull–Clark (closed meshes, any polygons, `f64`), test meshes and
//! point-set matching.

#![expect(clippy::missing_panics_doc, reason = "test helpers panic to fail")]

use std::collections::{BTreeMap, HashMap};

use crate::{BaseMesh, Vec3};

/// A point in `f64`.
pub(crate) type P = [f64; 3];

/// A closed polygon mesh in `f64`: positions and faces as vertex indices, counter-clockwise seen
/// from outside.
#[derive(Debug, Clone)]
pub(crate) struct Poly {
    pub(crate) positions: Vec<P>,
    pub(crate) faces: Vec<Vec<usize>>,
}

fn add(a: P, b: P) -> P {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: P, s: f64) -> P {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The mean of some points.
#[expect(clippy::cast_precision_loss, reason = "point counts are tiny")]
fn mean(points: impl ExactSizeIterator<Item = P>) -> P {
    let count = points.len() as f64;
    scale(points.fold([0.0; 3], add), 1.0 / count)
}

/// A point rounded to `f32`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "test meshes are well inside f32 range"
)]
fn to_vec3(p: P) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

/// An undirected edge, smaller vertex first.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

impl Poly {
    /// One Catmull–Clark step, from the textbook definition: face points are face centroids, edge
    /// points average the edge's ends and its two face points, and vertex points are
    /// `(Q + 2R + (n − 3)V) / n`.
    pub(crate) fn subdivide(&self) -> Self {
        let face_points: Vec<P> = self
            .faces
            .iter()
            .map(|f| mean(f.iter().map(|&v| self.positions[v])))
            .collect();
        // Sorted, so new vertex numbering is deterministic.
        let mut edge_faces: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for (&a, &b) in f.iter().zip(f.iter().cycle().skip(1)) {
                edge_faces.entry(edge_key(a, b)).or_default().push(fi);
            }
        }

        // Vertex points: (Q + 2R + (n − 3)V) / n, with Q the mean of adjacent face points and R the
        // mean of adjacent edge midpoints.
        let nv = self.positions.len();
        let mut vfaces: Vec<Vec<usize>> = vec![Vec::new(); nv];
        let mut vedges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];
        for (fi, f) in self.faces.iter().enumerate() {
            for &v in f {
                vfaces[v].push(fi);
            }
        }
        for &(a, b) in edge_faces.keys() {
            vedges[a].push((a, b));
            vedges[b].push((a, b));
        }
        let mut positions: Vec<P> = self
            .positions
            .iter()
            .zip(vfaces.iter().zip(&vedges))
            .map(|(&v, (fs, es))| {
                assert_eq!(
                    fs.len(),
                    es.len(),
                    "closed manifold: as many faces as edges at a vertex"
                );
                #[expect(clippy::cast_precision_loss, reason = "valences are tiny")]
                let n = fs.len() as f64;
                let q = mean(fs.iter().map(|&f| face_points[f]));
                let r = mean(
                    es.iter()
                        .map(|&(a, b)| scale(add(self.positions[a], self.positions[b]), 0.5)),
                );
                scale(add(add(q, scale(r, 2.0)), scale(v, n - 3.0)), 1.0 / n)
            })
            .collect();

        let mut edge_index: HashMap<(usize, usize), usize> = HashMap::new();
        for (&(a, b), fs) in &edge_faces {
            assert_eq!(fs.len(), 2, "closed mesh: every edge has two faces");
            let sum = add(
                add(self.positions[a], self.positions[b]),
                add(face_points[fs[0]], face_points[fs[1]]),
            );
            edge_index.insert((a, b), positions.len());
            positions.push(scale(sum, 0.25));
        }

        let face_base = positions.len();
        positions.extend(face_points.iter().copied());
        let edge = |a: usize, b: usize| edge_index[&edge_key(a, b)];
        let faces = self
            .faces
            .iter()
            .enumerate()
            .flat_map(|(fi, f)| {
                let len = f.len();
                (0..len).map(move |c| {
                    let (prev, v, next) = (f[(c + len - 1) % len], f[c], f[(c + 1) % len]);
                    vec![v, edge(v, next), face_base + fi, edge(prev, v)]
                })
            })
            .collect();
        Self { positions, faces }
    }

    /// This mesh, which must be all quads, as a base mesh (positions rounded to `f32`).
    pub(crate) fn to_base_mesh(&self) -> BaseMesh {
        let positions = self.positions.iter().copied().map(to_vec3).collect();
        let faces: Vec<Vec<u32>> = self
            .faces
            .iter()
            .map(|f| f.iter().map(|&v| u32::try_from(v).unwrap()).collect())
            .collect();
        BaseMesh::from_polygons(positions, &faces).unwrap()
    }

    /// A base mesh as a polygon mesh, with the same vertex and face numbering.
    pub(crate) fn from_base_mesh(mesh: &BaseMesh) -> Self {
        let positions = mesh
            .vertices()
            .map(|v| {
                let p = mesh.vertex_position(v).unwrap();
                [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
            })
            .collect();
        let faces = mesh
            .faces()
            .map(|f| {
                (0..4)
                    .map(|c| {
                        let v = mesh.corner_vertex(f, c).unwrap();
                        usize::try_from(v.index()).unwrap()
                    })
                    .collect()
            })
            .collect();
        Self { positions, faces }
    }
}

/// A pentagonal prism: two pentagons and five quads. One subdivision step gives a quad mesh with
/// valence-5 vertices.
pub(crate) fn pentagonal_prism() -> Poly {
    let mut positions = Vec::new();
    for z in [-1.0, 1.0] {
        for k in 0..5 {
            let a = std::f64::consts::TAU * f64::from(k) / 5.0;
            positions.push([a.cos(), a.sin(), z]);
        }
    }
    let mut faces = vec![vec![4, 3, 2, 1, 0], vec![5, 6, 7, 8, 9]];
    for i in 0..5 {
        let j = (i + 1) % 5;
        faces.push(vec![i, j, 5 + j, 5 + i]);
    }
    Poly { positions, faces }
}

/// Asserts that every point of `ours` is within `tol` of some point of `reference`, and that
/// `ours`, with bit-identical duplicates removed, has as many points as `reference`.
pub(crate) fn assert_same_points(ours: &[Vec3], reference: &[P], tol: f64) {
    let mut unique: Vec<[u32; 3]> = ours.iter().map(|p| p.to_bits()).collect();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        reference.len(),
        "vertex count (shared vertices must be bit-identical copies)"
    );
    for p in ours {
        let q = [f64::from(p.x), f64::from(p.y), f64::from(p.z)];
        let best = reference
            .iter()
            .map(|r| ((r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2) + (r[2] - q[2]).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best <= tol,
            "{p:?} is {best} from the nearest reference vertex"
        );
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn subdividing_a_cube_counts() {
        let cube = Poly::from_base_mesh(&crate::cube_sphere(1).unwrap());
        let once = cube.subdivide();
        assert_eq!(
            (once.positions.len(), once.faces.len()),
            (26, 24),
            "V = 8 + 12 + 6, F = 6 × 4"
        );
    }

    /// The cube `[−1, 1]³`, faces outward.
    fn textbook_cube() -> Poly {
        let positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        .map(Vec::from)
        .to_vec();
        Poly { positions, faces }
    }

    #[test]
    fn cube_corner_moves_to_five_ninths() {
        // Textbook value: (Q + 2R) / 3 = ((1/3) + 2·(2/3)) / 3 = 5/9 per coordinate.
        let once = textbook_cube().subdivide();
        for (got, want) in once.positions[6].iter().zip([5.0 / 9.0; 3]) {
            assert!(
                (got - want).abs() < 1e-12,
                "corner (1, 1, 1) moves to 5/9: got {got}"
            );
        }
    }

    #[test]
    fn cube_edge_points_average_ends_and_face_centres() {
        let cube = textbook_cube();
        let once = cube.subdivide();
        let centre = |f: &[usize]| {
            f.iter()
                .fold([0.0; 3], |s, &v| add(s, cube.positions[v]))
                .map(|x| x / 4.0)
        };
        // Each undirected edge with its two faces, found here independently of `subdivide`.
        let mut found = 0;
        for (fi, f) in cube.faces.iter().enumerate() {
            for k in 0..4 {
                let (a, b) = (f[k], f[(k + 1) % 4]);
                let other = cube
                    .faces
                    .iter()
                    .enumerate()
                    .find(|&(gi, g)| gi != fi && g.contains(&a) && g.contains(&b))
                    .map(|(_, g)| g)
                    .unwrap();
                let want = add(
                    add(cube.positions[a], cube.positions[b]),
                    add(centre(f), centre(other)),
                )
                .map(|x| x / 4.0);
                let matches = once
                    .positions
                    .iter()
                    .filter(|p| p.iter().zip(want).all(|(x, y)| (x - y).abs() < 1e-12))
                    .count();
                assert_eq!(
                    matches, 1,
                    "edge {a}–{b}: one edge point at (a + b + F1 + F2) / 4"
                );
                found += 1;
            }
        }
        assert_eq!(found, 24, "each of the 12 edges seen from both faces");
    }

    #[test]
    fn prism_subdivides_to_valence_five() {
        let quads = pentagonal_prism().subdivide();
        assert_eq!(quads.faces.len(), 30, "2×5 + 5×4");
        let mesh = quads.to_base_mesh();
        let valence_five = mesh
            .faces()
            .flat_map(|f| (0..4).map(move |c| (f, c)))
            .filter(|&(f, c)| mesh.topology().ring(f, c).unwrap().entries.len() == 5)
            .count();
        assert_eq!(
            valence_five, 10,
            "two pentagon centres, each seen from 5 faces"
        );
    }

    #[test]
    fn point_matching_accepts_bit_identical_copies() {
        let prism = pentagonal_prism();
        let mut ours: Vec<Vec3> = prism.positions.iter().copied().map(to_vec3).collect();
        ours.push(ours[3]);
        assert_same_points(&ours, &prism.positions, 1e-6);
    }

    #[test]
    #[should_panic(expected = "bit-identical")]
    fn point_matching_rejects_copies_that_differ_in_the_last_bit() {
        let prism = pentagonal_prism();
        let mut ours: Vec<Vec3> = prism.positions.iter().copied().map(to_vec3).collect();
        let p = ours[3];
        ours.push(Vec3::new(f32::from_bits(p.x.to_bits() + 1), p.y, p.z));
        assert_same_points(&ours, &prism.positions, 1e-6);
    }

    #[test]
    #[should_panic(expected = "from the nearest reference vertex")]
    fn point_matching_rejects_a_far_point() {
        let prism = pentagonal_prism();
        let mut ours: Vec<Vec3> = prism.positions.iter().copied().map(to_vec3).collect();
        ours[3] = ours[3] + Vec3::new(0.0, 0.0, 0.5);
        assert_same_points(&ours, &prism.positions, 1e-6);
    }
}
