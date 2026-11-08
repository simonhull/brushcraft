//! A tall stroke on a rough surface folds no triangle over: draw moves every vertex of a dab
//! along its area normal.
//!
//! The surface is the `preview` bench's (`crates/sculpt/benches/preview.rs`): every level of the
//! sphere sculpted with an octave of value noise, so its normals turn by a few degrees over a few
//! quads, as the performance suite's projects' do. Moving each vertex along its own frame normal
//! there folds the surface over under a ridge taller than those normals' radius of curvature
//! (about 1.5 % of the bench's previews' triangles, up to a fifth of a tile; 1.1 % for the app's
//! strong stroke on the medium project). Moving all of a dab's vertices along one direction removes
//! that convergence, and here it folds nothing; it isn't a guarantee: a wall inside the dab's
//! sphere steep against that direction is sheared by its vertices' differing displacements and
//! can still flip.
//!
//! The setup is the bench's: `cube_sphere(4)` to level 8 (1,536 tiles of 64 × 64 quads, a quad
//! about 0.0015 across the unit sphere), and a ridge of 60 dabs of radius 0.2 (about 130 quads),
//! strength a tenth of it, spacing 0.15 of it, each placed on the stroke's pen-down surface by a
//! raycast and given its area normal there, then applied as finest work (one dab a call). Moving
//! each vertex along its own frame normal instead, the same ridge flips 17,444 of the 667,625
//! triangles it moves, 1,206 in the worst tile (2026-11-02, before draw took the area normal); at
//! level 7 it flips 7, at level 6 none: the folds need the finest octaves' turning normals under
//! a ridge many quads tall.

use std::collections::{BTreeMap, BTreeSet};

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use store::{Evaluation, Project, SpatialIndex, TileStore};
use stroke::{DVec3, Dab, Ray, Surface};

use super::super::apply_dabs;
use crate::StoreSurface;
use crate::state::Stroke;
use crate::testing::{TestDir, fixed, project_over};

/// The level sculpted and edited.
const LEVEL: u8 = 8;

/// The ridge's dabs' radius, on the unit sphere.
const RADIUS: f32 = 0.2;

/// The dabs of the ridge.
const RIDGE_DABS: u32 = 60;

/// The noise's amplitude at level 1, each level's octave half the one below's: the bench's.
const AMPLITUDE: f32 = 0.02;

#[test]
fn a_tall_stroke_on_a_rough_surface_folds_no_triangle() {
    let dir = TestDir::new();
    let level = Level::new(LEVEL).unwrap();
    let (mut project, _) = project_over(&dir, cube_sphere(4).unwrap(), level, 1024, fixed(4));
    for octave in 1..=LEVEL {
        sculpt_octave(&mut project, Level::new(octave).unwrap());
    }
    let mut index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
    // The tiles within two radii of the ridge's path: every one it can move.
    let near: BTreeSet<TileKey> = (0..RIDGE_DABS)
        .flat_map(|n| index.query_sphere(direction(n), 2.0 * RADIUS))
        .collect();
    let keys: Vec<TileKey> = near.into_iter().collect();
    let before = triangle_normals(project.store(), &keys);

    let mut stroke = Stroke::new(project.store());
    project.begin_stroke_and_wait().unwrap();
    let mut least = 1.0_f32;
    for n in 0..RIDGE_DABS {
        let dab = next_dab(project.store(), &index, n);
        let [x, y, z] = dab.position;
        let outward = Vec3::new(x, y, z).normalized().unwrap();
        let [nx, ny, nz] = dab.normal;
        least = least.min(outward.dot(Vec3::new(nx, ny, nz)));
        apply_dabs(project.store(), &mut index, &mut stroke, &[dab]).unwrap();
    }
    project.end_unrecorded_stroke().unwrap();
    let after = triangle_normals(project.store(), &keys);

    let (mut flipped, mut worst, mut moved) = (0_usize, 0_usize, 0_usize);
    for (key, was) in &before {
        let now = &after[key];
        let tile = was
            .iter()
            .zip(now)
            .filter(|&(a, b)| a.dot(*b) < 0.0)
            .count();
        flipped += tile;
        worst = worst.max(tile);
        moved += was
            .iter()
            .zip(now)
            .filter(|&(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    let total: usize = before.values().map(Vec::len).sum();
    let tilt = least.clamp(-1.0, 1.0).acos().to_degrees();
    println!(
        "{flipped} of {total} triangles flipped ({worst} in the worst tile), {moved} moved; \
         area normals at most {tilt:.1}° off the radial"
    );
    assert!(moved > 10_000, "the ridge moved the surface: {moved}");
    assert!(
        tilt < 10.0,
        "the area normals are the rough sphere's at pen-down, not the ridge's: {tilt}°"
    );
    assert_eq!(
        flipped, 0,
        "no triangle folds over ({worst} in the worst tile)"
    );
}

/// The direction of dab `n` of the ridge from the sphere's centre: along a great circle, 0.15
/// radii a dab, as the bench's.
fn direction(n: u32) -> Vec3 {
    #[expect(clippy::cast_precision_loss, reason = "a few dozen dabs")]
    let angle = 0.15 * RADIUS * n as f32;
    let (sin, cos) = angle.sin_cos();
    Vec3::new(0.6 * cos, 0.6 * sin, 0.8).normalized().unwrap()
}

/// The next dab of the ridge, number `n`: its centre on the stroke's pen-down surface by a
/// raycast from outside along the ridge's great circle, its normal its area normal there, full
/// strength.
fn next_dab(store: &mut TileStore, index: &SpatialIndex, n: u32) -> Dab {
    let direction = direction(n);
    let from = DVec3::from_f32(direction);
    let ray = Ray::new(from * 3.0, -from).unwrap();
    let mut surface = StoreSurface::new(store, index);
    let hit = surface.raycast(&ray).unwrap();
    let sum = surface
        .area_normal(hit.position, hit.normal, f64::from(RADIUS), 0.0)
        .unwrap();
    surface.finish().unwrap();
    let centre = hit.position.to_f32();
    let normal = sum.normalized().unwrap().to_f32();
    Dab {
        position: [centre.x, centre.y, centre.z],
        normal: [normal.x, normal.y, normal.z],
        tangent: [0.0, 0.0, 1.0],
        radius: RADIUS,
        strength: 0.1 * RADIUS,
        hardness: 0.0,
        seq: n,
        ..Dab::default()
    }
}

/// Each tile of `keys`'s triangles' unit normals (zero for a degenerate one).
///
/// Two a quad, row by row: `(i, j), (i + 1, j), (i + 1, j + 1)` then `(i, j), (i + 1, j + 1),
/// (i, j + 1)`, as the store's raycasts split them.
fn triangle_normals(store: &mut TileStore, keys: &[TileKey]) -> BTreeMap<TileKey, Vec<Vec3>> {
    store.evaluate_many(keys, Evaluation::Positions).unwrap();
    keys.iter()
        .map(|&key| {
            let positions = store.evaluate(key).unwrap();
            let side = key.side();
            let at = |i: u32, j: u32| positions[usize::try_from(j * side + i).unwrap()];
            let face =
                |a: Vec3, b: Vec3, c: Vec3| (b - a).cross(c - a).normalized().unwrap_or(Vec3::ZERO);
            let quads = (0..side - 1).flat_map(|j| (0..side - 1).map(move |i| (i, j)));
            let normals = quads
                .flat_map(|(i, j)| {
                    let (p, q, r, s) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                    [face(p, q, r), face(p, r, s)]
                })
                .collect();
            (key, normals)
        })
        .collect()
}

/// Sculpts `level`'s octave of the noise into every tile of it, as the `preview` bench does.
///
/// Each vertex the tile owns is pushed along its frame's normal by
/// `AMPLITUDE · 2^−(L−1) · noise(p · f_L)`, `p` its position with the levels below sculpted and
/// `f_L = 4 · 2^L / 2π` (features about four quads of the level across).
fn sculpt_octave(project: &mut Project, level: Level) {
    let l = i32::from(level.get());
    let amplitude = AMPLITUDE * 0.5_f32.powi(l - 1);
    let frequency = 4.0 * 2.0_f32.powi(l) / std::f32::consts::TAU;
    let per_side = u16::try_from(level.tiles_per_side()).unwrap();
    let faces = project.store().mesh().face_count();
    let keys: Vec<TileKey> = (0..faces)
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side)
                    .map(move |tx| TileKey::displacement(FaceId::new(face), level, tx, ty).unwrap())
            })
        })
        .collect();
    project.begin_stroke_and_wait().unwrap();
    let store = project.store();
    store.evaluate_many(&keys, Evaluation::Positions).unwrap();
    for key in keys {
        let positions = store.evaluate(key).unwrap().to_vec();
        let side = key.side();
        let topology = store.mesh().topology();
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let writes: Vec<(u32, u32, f32)> = grid
            .zip(&positions)
            .filter(|&((i, j), _)| topology.owns(&key, i, j))
            .map(|((i, j), &p)| (i, j, amplitude * value_noise(p * frequency, l)))
            .collect();
        store
            .edit_tile(key, |edit| {
                writes
                    .iter()
                    .try_for_each(|&(i, j, z)| edit.set(i, j, Vec3::new(0.0, 0.0, z)))
            })
            .unwrap()
            .unwrap();
    }
    project.end_unrecorded_stroke().unwrap();
}

/// Value noise in 3D, in `[-1, 1]`: a random value at every point of the integer lattice (of
/// octave `octave`), blended between the eight around `p` with smoothstep weights. The bench's.
fn value_noise(p: Vec3, octave: i32) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unit-sphere points times at most a few hundred"
    )]
    let split = |x: f32| (x.floor() as i32, x - x.floor());
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let ((x0, fx), (y0, fy), (z0, fz)) = (split(p.x), split(p.y), split(p.z));
    let lattice = |dx: i32, dy: i32, dz: i32| {
        let key = [x0 + dx, y0 + dy, z0 + dz, octave].into_iter().fold(
            0x9E37_79B9_7F4A_7C15_u64,
            |h, k| {
                let mut z = h ^ u64::from(k.cast_unsigned());
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            },
        );
        #[expect(clippy::cast_precision_loss, reason = "24 bits are exact in an f32")]
        let bits = (key >> 40) as f32;
        bits / 8_388_608.0 - 1.0
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let along_x = |dy, dz| lerp(lattice(0, dy, dz), lattice(1, dy, dz), smooth(fx));
    let along_y = |dz| lerp(along_x(0, dz), along_x(1, dz), smooth(fy));
    lerp(along_y(0), along_y(1), smooth(fz))
}
