//! The benchmark's base mesh: a subdivided cube projected onto the unit sphere.

use std::collections::HashMap;

use crate::mesh::fit;
use crate::{BaseMesh, MeshError, Vec3};

/// Largest `splits` whose lattice coordinates (`|x| ≤ splits`) fit an `i16`, and so convert to
/// `f32` exactly.
const MAX_SPLITS: u32 = 16383;

/// A cube with each side split `splits × splits`, projected onto the unit sphere: `6·splits²`
/// quads, all outward-facing. Corners have valence 3, every other vertex valence 4.
///
/// # Errors
///
/// [`MeshError::Empty`] for `splits == 0`; [`MeshError::TooLarge`] for `splits` above 16383.
pub fn cube_sphere(splits: u32) -> Result<BaseMesh, MeshError> {
    if splits > MAX_SPLITS {
        return Err(MeshError::TooLarge);
    }
    let half: i32 = fit(splits)?;
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
        for v in 0..mesh.vertex_count() {
            assert!(
                (mesh.position(v).unwrap().length() - 1.0).abs() < 1e-6,
                "on the sphere"
            );
        }
    }
}
