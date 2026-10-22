//! The benchmark's base mesh: a subdivided cube projected onto the unit sphere.

use std::collections::HashMap;

use crate::mesh::fit;
use crate::{BaseMesh, MeshError, Vec3};

/// Largest `splits` whose `6·splits²` faces fit 32-bit ids (6 · 26 754² = 4 294 659 096). Its
/// lattice coordinates, `|x| ≤ splits`, then also fit an `i16`, so they convert to `f32` exactly.
const MAX_SPLITS: u16 = 26_754;

/// A cube with each side split `splits × splits`, projected onto the unit sphere: `6·splits²`
/// quads, all outward-facing. Corners have valence 3, every other vertex valence 4.
///
/// # Errors
///
/// [`MeshError::Empty`] for `splits == 0`; [`MeshError::TooLarge`] for `splits` above 26 754.
pub fn cube_sphere(splits: u32) -> Result<BaseMesh, MeshError> {
    let half = u16::try_from(splits)
        .ok()
        .filter(|&s| s <= MAX_SPLITS)
        .map(i32::from)
        .ok_or(MeshError::TooLarge)?;
    // (normal, u, v) with u × v = normal, so quads in (u, v) order face outward.
    let sides: [([i32; 3], [i32; 3], [i32; 3]); 6] = [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
        ([0, 1, 0], [0, 0, 1], [1, 0, 0]),
        ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
        ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
        ([0, 0, -1], [0, 1, 0], [1, 0, 0]),
    ];
    let mut index: HashMap<[i32; 3], u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut quads = Vec::new();
    for (normal, u, v) in sides {
        let mut vertex = |col: i32, row: i32| -> Result<u32, MeshError> {
            // Lattice point on the cube of half-size `half`: every coordinate is in `-half ..=
            // half`.
            let mut lattice = [0; 3];
            for (((out, nk), uk), vk) in lattice.iter_mut().zip(normal).zip(u).zip(v) {
                *out = nk * half + uk * (2 * col - half) + vk * (2 * row - half);
            }
            if let Some(&id) = index.get(&lattice) {
                return Ok(id);
            }
            let id = fit(positions.len())?;
            let coord = |c: i32| fit::<i16, _>(c).map(f32::from);
            let [x, y, z] = lattice;
            // Never zero: the `normal` coordinate is ±half, and half ≥ 1 whenever a quad is made.
            positions.push(
                Vec3::new(coord(x)?, coord(y)?, coord(z)?)
                    .normalized()
                    .unwrap_or(Vec3::ZERO),
            );
            index.insert(lattice, id);
            Ok(id)
        };
        for row in 0..half {
            for col in 0..half {
                quads.push([
                    vertex(col, row)?,
                    vertex(col + 1, row)?,
                    vertex(col + 1, row + 1)?,
                    vertex(col, row + 1)?,
                ]);
            }
        }
    }
    BaseMesh::from_polygons(positions, &quads)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn benchmark_sphere_has_2400_quads_on_the_unit_sphere() {
        let mesh = cube_sphere(20).unwrap();
        assert_eq!(mesh.face_count(), 2400, "6 × 20²");
        assert_eq!(
            mesh.vertex_count(),
            2402,
            "Euler: V = F + 2 for a closed quad sphere"
        );
        for v in mesh.vertices() {
            assert!(
                (mesh.vertex_position(v).unwrap().length() - 1.0).abs() < 1e-6,
                "on the sphere"
            );
        }
    }

    #[test]
    fn every_face_points_outward() {
        let mesh = cube_sphere(5).unwrap();
        for f in mesh.faces() {
            let [p0, p1, p2, p3] = [0, 1, 2, 3].map(|c| mesh.corner_position(f, c).unwrap());
            let centroid = (p0 + p1 + p2 + p3) * 0.25;
            let normal = (p1 - p0).cross(p3 - p0);
            assert!(normal.dot(centroid) > 0.0, "face {f:?} points inward");
        }
    }

    #[test]
    fn rejects_sizes_past_32_bit_ids() {
        assert_eq!(cube_sphere(0).unwrap_err(), MeshError::Empty, "no faces");
        assert_eq!(
            cube_sphere(26_755).unwrap_err(),
            MeshError::TooLarge,
            "6·26 755² faces overflow u32"
        );
        assert_eq!(
            cube_sphere(u32::MAX).unwrap_err(),
            MeshError::TooLarge,
            "far too large"
        );
    }
}
