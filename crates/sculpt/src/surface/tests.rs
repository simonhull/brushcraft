//! `StoreSurface` on generated projects.
//!
//! Picking's raycast, continuity across a seam and a gap, the closest point on a region, a kept
//! error, a stroke driven over it end to end, and (ignored, run in release) each method's cost.

use std::time::{Duration, Instant};

use geom::{Channel, FaceId, LayerId, Level, Region, TileKey, Vec3, cube_sphere};
use store::{Project, SpatialIndex, TileStore};
use stroke::{
    DVec3, Dab, Hit, Projector, Ray, Sample, ScreenPoint, Strength, Stroke as PenStroke, StrokeId,
    StrokeSettings, Surface, Unit,
};

use super::StoreSurface;
use crate::SculptError;
use crate::apply::apply_dabs;
use crate::state::Stroke;
use crate::testing::{TestDir, cube, dab, fixed, project_over, snapshot, tiles, two_cubes, vertex};

/// The level the tests read: 128 quads a face, four tiles of 64.
fn level() -> Level {
    Level::new(7).unwrap()
}

/// A few strong dabs over face 0 and its neighbours, so the surface isn't a smooth sphere.
fn bumps(project: &mut Project, index: &mut SpatialIndex) {
    let level = level();
    let mut frame = Vec::new();
    let faces = (0_u32..).zip([0, 1, 2, 0]);
    for (n, face) in faces {
        let key = TileKey::displacement(FaceId::new(face), level, 1, 0).unwrap();
        let centre = vertex(project.store(), key, 20 + 9 * n, 30);
        frame.push(dab(centre, 0.2, 0.03, 0.2, n));
    }
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), index, &mut state, &frame).unwrap();
    project.end_stroke().unwrap();
}

/// `count` directions spread evenly over the sphere (a Fibonacci lattice).
fn directions(count: u32) -> impl Iterator<Item = DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count).map(move |k| {
        let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(count);
        let r = (1.0 - z * z).sqrt();
        let a = golden * f64::from(k);
        DVec3::new(r * a.cos(), r * a.sin(), z)
    })
}

/// The ray from 3 units out along `direction` toward the origin.
fn inward(direction: DVec3) -> Ray {
    Ray::new(direction * 3.0, -direction).unwrap()
}

/// The hit at face-grid vertex `(gi, gj)` of `face` at the test level: its evaluated position
/// and normal, from the tile holding it.
fn vertex_hit(store: &mut TileStore, face: FaceId, gi: u32, gj: u32) -> Hit {
    let level = level();
    let (key, i, j) =
        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, gi, gj).unwrap();
    let at = usize::try_from(j * key.side() + i).unwrap();
    let normal = store.normals(key).unwrap()[at];
    let position = store.evaluate(key).unwrap()[at];
    let n = f64::from(level.face_quads());
    Hit {
        position: DVec3::from_f32(position),
        normal: DVec3::from_f32(normal),
        face,
        uv: [f64::from(gi) / n, f64::from(gj) / n],
    }
}

#[test]
fn raycasts_hit_what_picking_hits() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    bumps(&mut project, &mut index);
    let store = project.store();
    let mut hits = 0;
    for direction in directions(300) {
        let ray = inward(direction);
        let narrow = store::Ray::new(ray.origin().to_f32(), ray.direction().to_f32()).unwrap();
        let picked = index.raycast(store, &narrow).unwrap();
        let triangle = index.raycast_triangle(store, &narrow).unwrap();
        assert_eq!(
            triangle.map(|hit| (hit.key, hit.t.to_bits())),
            picked.map(|hit| (hit.key, hit.t.to_bits())),
            "the same tile at the same parameter, toward {direction:?}"
        );
        let mut surface = StoreSurface::new(store, &index);
        let (Some(hit), Some(picked)) = (surface.raycast(&ray), picked) else {
            continue;
        };
        hits += 1;
        let at = DVec3::from_f32(narrow.at(picked.t));
        assert!(
            hit.position.distance(at) < 1e-5,
            "the point on the triangle is where the ray crosses it: {hit:?} vs {at:?}"
        );
        assert_eq!(hit.face, picked.key.face(), "the hit tile's face");
        assert_eq!(
            surface.tile_of(&hit),
            Some(picked.key),
            "its coordinates are in the hit tile"
        );
        assert!(
            (hit.normal.length() - 1.0).abs() < 1e-9,
            "a unit normal: {hit:?}"
        );
        assert!(
            hit.normal.dot(ray.direction()) < 0.0,
            "facing the ray from outside: {hit:?}"
        );
        assert!(hit.uv.iter().all(|c| (0.0..=1.0).contains(c)), "{hit:?}");
        assert!(surface.error().is_none(), "no error");
    }
    assert_eq!(hits, 300, "every ray at the closed surface hits it");
}

#[test]
fn hits_either_side_of_a_seam_between_faces_are_continuous() {
    let dir = TestDir::new();
    let (mut project, index) = cube(&dir, level(), fixed(2));
    let level = level();
    let n = level.face_quads();
    let store = project.store();
    let a = vertex_hit(store, FaceId::new(0), n / 2, 2);
    let copies = store
        .mesh()
        .topology()
        .copies(FaceId::new(0), level, n / 2, 0);
    let across = copies
        .iter()
        .find(|copy| copy.face != FaceId::new(0))
        .unwrap();
    let region = Region::of_tile(&across.key(level));
    let (mut gi, mut gj) = (region.x0() + across.i, region.y0() + across.j);
    if gi == 0 {
        gi += 2;
    } else if gi == n {
        gi -= 2;
    } else if gj == 0 {
        gj += 2;
    } else {
        gj -= 2;
    }
    let b = vertex_hit(store, across.face, gi, gj);
    assert_ne!(a.face, b.face, "the hits are on two faces");
    let apart = a.position.distance(b.position);
    let mut surface = StoreSurface::new(store, &index);
    assert!(surface.continuous(&a, &b, 4.0 * apart), "across the seam");
    assert!(surface.continuous(&b, &a, 4.0 * apart), "and back");
    assert!(
        !surface.continuous(&a, &b, 0.5 * apart),
        "not with a reach short of the distance between them"
    );
    assert!(surface.error().is_none(), "no error");
}

/// The vertex of faces `faces` with the greatest `key(position)`, as a hit, over every tile.
fn extreme(store: &mut TileStore, faces: std::ops::Range<u32>, key: impl Fn(Vec3) -> f32) -> Hit {
    let level = level();
    let mut best: Option<(f32, TileKey, u32, u32)> = None;
    for tile in tiles(store.mesh(), level) {
        if !faces.contains(&tile.face().index()) {
            continue;
        }
        let side = tile.side();
        let positions = store.evaluate(tile).unwrap().to_vec();
        let numbered = (0_u32..).zip(positions);
        for (k, p) in numbered {
            if best.is_none_or(|(b, ..)| key(p) > b) {
                best = Some((key(p), tile, k % side, k / side));
            }
        }
    }
    let (_, tile, i, j) = best.unwrap();
    let region = Region::of_tile(&tile);
    vertex_hit(store, tile.face(), region.x0() + i, region.y0() + j)
}

#[test]
fn two_parts_across_a_gap_are_not_continuous_and_a_region_keeps_to_its_own() {
    let dir = TestDir::new();
    let (mut project, index) = project_over(&dir, two_cubes(1.3), level(), 128, fixed(2));
    let store = project.store();
    // The facing points of the two parts: the first's rightmost, the second's leftmost.
    let a = extreme(store, 0..6, |p| p.x);
    let b = extreme(store, 6..12, |p| -p.x);
    let gap = b.position.x - a.position.x;
    assert!(gap > 0.05, "the parts are apart: {gap}");
    let reach = 3.0 * gap;
    let mut surface = StoreSurface::new(store, &index);
    assert!(
        !surface.continuous(&a, &b, reach),
        "no surface joins the parts, however near"
    );

    // A ray that hits the second part, given the first's point as the region's anchor.
    let eye = (a.position + b.position) * 0.5 + DVec3::new(0.0, 0.0, 2.0);
    let ray = Ray::new(eye, b.position - eye).unwrap();
    let hit = surface.raycast(&ray).unwrap();
    assert!(
        hit.face.index() >= 6,
        "the ray hits the second part: {hit:?}"
    );
    let closest = surface.closest_on_region(&ray, &a, reach).unwrap();
    assert!(
        closest.face.index() < 6,
        "the region's point is on the first part: {closest:?}"
    );
    assert!(
        closest.position.distance(a.position) <= reach,
        "within reach of the anchor: {closest:?}"
    );
    assert!(
        surface.continuous(&a, &closest, reach),
        "continuous with the anchor, as the stroke engine requires"
    );
    assert_eq!(
        surface.closest_on_region(&ray, &a, reach),
        Some(closest),
        "the same answer again"
    );

    // No vertex of the first part within reach is nearer the ray.
    let reach_squared = reach * reach;
    let mut nearest = f64::INFINITY;
    for tile in tiles(surface.store.mesh(), level()) {
        if tile.face().index() >= 6 {
            continue;
        }
        for &p in surface.store.evaluate(tile).unwrap() {
            let p = DVec3::from_f32(p);
            if (p - a.position).dot(p - a.position) <= reach_squared {
                let off = p - ray.closest_to(p);
                nearest = nearest.min(off.dot(off));
            }
        }
    }
    let off = closest.position - ray.closest_to(closest.position);
    assert_eq!(
        off.dot(off).to_bits(),
        nearest.to_bits(),
        "the nearest vertex of the region to the ray"
    );
    assert!(surface.error().is_none(), "no error");
}

#[test]
fn a_reach_under_a_quad_answers_the_anchor() {
    let dir = TestDir::new();
    let (mut project, index) = cube(&dir, level(), fixed(2));
    let store = project.store();
    let a = vertex_hit(store, FaceId::new(2), 40, 50);
    let near = Hit {
        position: a.position + DVec3::new(0.0, 0.0, 1e-3),
        ..a
    };
    let ray = Ray::new(DVec3::new(0.0, 0.0, 5.0), DVec3::new(1.0, 0.0, 0.0)).unwrap();
    let mut surface = StoreSurface::new(store, &index);
    assert_eq!(
        surface.closest_on_region(&ray, &near, 1e-6),
        Some(near),
        "no vertex within reach: the anchor itself"
    );
}

#[test]
fn an_error_is_kept_and_every_later_call_answers_none() {
    let dir = TestDir::new();
    let (mut project, _) = cube(&dir, level(), fixed(2));
    // An index over another mesh, of twelve faces: its tiles past face 5 aren't this store's.
    let other = TestDir::new();
    let (_, index) = project_over(&other, two_cubes(1.3), level(), 128, fixed(2));
    let store = project.store();
    let a = vertex_hit(store, FaceId::new(0), 10, 10);
    let mut surface = StoreSurface::new(store, &index);
    let ray = Ray::new(DVec3::new(1.3, 0.0, 3.0), DVec3::new(0.0, 0.0, -1.0)).unwrap();
    assert_eq!(surface.raycast(&ray), None, "a tile the store lacks");
    assert!(surface.error().is_some(), "the error is kept");
    assert!(
        !surface.continuous(&a, &a, 1.0),
        "every later call answers false"
    );
    assert_eq!(surface.closest_on_region(&ray, &a, 1.0), None, "or none");
    assert!(surface.take_error().is_some(), "taken");
    assert!(surface.error().is_none(), "and gone");
    assert!(surface.continuous(&a, &a, 1.0), "answering again");
    assert!(surface.finish().is_ok(), "nothing kept");

    let mut again = StoreSurface::new(project.store(), &index);
    assert_eq!(again.raycast(&ray), None, "the same miss");
    assert!(
        matches!(again.finish(), Err(SculptError::Index(_))),
        "finishing gives the kept error back"
    );
}

/// The face-grid point `step` vertices inside face `face`'s grid from its border point
/// `(gi, gj)`, along each axis the point is on the border of.
fn inside(gi: u32, gj: u32, n: u32, step: u32) -> (u32, u32) {
    let inward = |g: u32| match g {
        0 => step,
        g if g == n => n - step,
        g => g,
    };
    (inward(gi), inward(gj))
}

/// Checks the faces around base vertex `(gi, gj)` of `face` are continuous there.
///
/// It hits a vertex two in from the corner on each face around the base vertex, and checks every
/// pair of them is continuous within four times their distance, and not within half of it.
/// Returns how many faces meet there.
fn around_a_corner(
    store: &mut TileStore,
    index: &SpatialIndex,
    face: FaceId,
    gi: u32,
    gj: u32,
) -> usize {
    let level = level();
    let n = level.face_quads();
    let copies = store.mesh().topology().copies(face, level, gi, gj);
    let hits: Vec<Hit> = copies
        .iter()
        .map(|copy| {
            let region = Region::of_tile(&copy.key(level));
            let (x, y) = inside(region.x0() + copy.i, region.y0() + copy.j, n, 2);
            vertex_hit(store, copy.face, x, y)
        })
        .collect();
    let mut surface = StoreSurface::new(store, index);
    for (k, a) in hits.iter().enumerate() {
        for b in hits.iter().skip(k + 1) {
            let apart = a.position.distance(b.position);
            assert_ne!(a.face, b.face, "two faces");
            assert!(surface.continuous(a, b, 4.0 * apart), "{a:?} to {b:?}");
            assert!(surface.continuous(b, a, 4.0 * apart), "{b:?} to {a:?}");
            assert!(
                !surface.continuous(a, b, 0.5 * apart),
                "short of the distance"
            );
        }
    }
    assert!(surface.finish().is_ok(), "no error");
    hits.len()
}

#[test]
fn hits_around_a_cube_corner_are_continuous() {
    let dir = TestDir::new();
    let (mut project, index) = cube(&dir, level(), fixed(2));
    let faces = around_a_corner(project.store(), &index, FaceId::new(0), 0, 0);
    assert_eq!(faces, 3, "three faces meet at a cube corner");
}

#[test]
fn hits_around_a_valence_5_vertex_are_continuous() {
    let dir = TestDir::new();
    let prism = geom::fixtures::subdivided_pentagonal_prism().unwrap();
    let (mut project, index) = project_over(&dir, prism, level(), 128, fixed(2));
    let n = level().face_quads();
    // Face 5's corner 2 is the top cap's centre, where five faces meet.
    let faces = around_a_corner(project.store(), &index, FaceId::new(5), n, n);
    assert_eq!(faces, 5, "five faces meet at the cap's centre");
}

/// A pinhole camera at `eye` looking down −z, `focal` pixels, its axis through screen point
/// `(400, 400)`: rays scaled so their parameter is the view depth.
#[derive(Debug, Clone, Copy)]
struct Pinhole {
    /// The eye.
    eye: DVec3,
    /// The focal length, in pixels.
    focal: f64,
}

impl Projector for Pinhole {
    fn ray(&self, screen: ScreenPoint) -> Option<Ray> {
        let direction = DVec3::new(
            (screen.x() - 400.0) / self.focal,
            -(screen.y() - 400.0) / self.focal,
            -1.0,
        );
        Ray::new(self.eye, direction)
    }

    fn world_per_pixel(&self, depth: f64) -> f64 {
        depth / self.focal
    }
}

/// The distance from `p` to the nearest vertex of the surface at the test level.
fn to_surface(store: &mut TileStore, index: &SpatialIndex, p: Vec3) -> f32 {
    let near: Vec<TileKey> = index.query_sphere(p, 0.05).collect();
    near.iter()
        .copied()
        .flat_map(|key| store.evaluate(key).unwrap().to_vec())
        .map(|q| (q - p).length())
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn a_stroke_over_the_store_edits_it_and_its_dabs_stay_on_the_surface() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let before = snapshot(project.store(), level());
    let camera = Pinhole {
        eye: DVec3::new(0.0, 0.0, 4.0),
        focal: 800.0,
    };
    let samples: Vec<Sample> = (0..40_u32)
        .map(|k| {
            let u = f64::from(k) / 39.0;
            let at = ScreenPoint::new(330.0 + 140.0 * u, 390.0 + 30.0 * u).unwrap();
            Sample::new(at, 0.5 * u, Unit::new(0.8).unwrap()).unwrap()
        })
        .collect();
    // The stroke engine's dabs are final: the brush's own strength times the pressure.
    let settings = StrokeSettings::default().with_strength(Strength::new(0.004).unwrap());
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    let (first, rest) = samples.split_first().unwrap();
    let mut pen = {
        let mut surface = StoreSurface::new(project.store(), &index);
        let pen = PenStroke::begin(&settings, StrokeId::new(1), &camera, &mut surface, *first)
            .expect("the first sample is over the cube");
        surface.finish().unwrap();
        pen
    };
    let mut placed = 0;
    let mut moved = 0;
    for sample in rest {
        let mut surface = StoreSurface::new(project.store(), &index);
        let dabs: Vec<Dab> = pen.push(&camera, &mut surface, *sample).collect();
        surface.finish().unwrap();
        for d in &dabs {
            assert!(
                (0.0032 - 1e-6..=0.0032 + 1e-6).contains(&d.strength),
                "the brush's strength at the pen's pressure: {}",
                d.strength
            );
            let [x, y, z] = d.position;
            let off = to_surface(project.store(), &index, Vec3::new(x, y, z));
            assert!(off < 0.01, "dab {} is on the surface: {off}", d.seq);
        }
        placed += dabs.len();
        moved += apply_dabs(project.store(), &mut index, &mut state, &dabs)
            .unwrap()
            .vertices;
    }
    let tail = {
        let mut surface = StoreSurface::new(project.store(), &index);
        let tail: Vec<Dab> = pen.end(&camera, &mut surface).collect();
        surface.finish().unwrap();
        tail
    };
    placed += tail.len();
    moved += apply_dabs(project.store(), &mut index, &mut state, &tail)
        .unwrap()
        .vertices;
    project.end_stroke().unwrap();
    assert!(placed >= 5, "the stroke placed dabs: {placed}");
    assert!(moved > 1000, "and they moved vertices: {moved}");
    assert!(
        snapshot(project.store(), level()) != before,
        "the project is edited"
    );
}

/// One timed call: the surface, and which of the prepared inputs.
type Call<'a> = dyn Fn(&mut StoreSurface<'_>, usize) + 'a;

/// Each method's cost on a generated project, warm and cold, in release: run with
/// `cargo nextest run -p sculpt --release --run-ignored only --no-capture surface_costs`.
///
/// The project is `cube_sphere(4)` (96 faces) at level 8: 1,536 tiles of 64 quads, 6.4 million
/// vertices, with a few bumps. The brush's radius is 0.08 (about 62 quads, 12,000 vertices), or
/// `SURFACE_RADIUS` if set (0.16 is about 50,000 vertices, the benchmark's footprint); the reach
/// is three radii. Each method runs 200 times: cold after evicting every tile, warm the same
/// calls again. It also prints the region's size in tiles and the walk's own cost.
#[test]
#[ignore = "a timing, in release: see the doc comment"]
fn surface_costs() {
    let dir = TestDir::new();
    let level = Level::new(8).unwrap();
    let (mut project, mut index) =
        project_over(&dir, cube_sphere(4).unwrap(), level, 1024, fixed(4));
    let radius: f64 = std::env::var("SURFACE_RADIUS").map_or(0.08, |r| r.parse().unwrap());
    let reach = 3.0 * radius;
    let calls = 200_u32;
    let rays: Vec<Ray> = directions(calls).map(inward).collect();
    let mut frame = Vec::new();
    let every_twentieth = (0_u32..).zip(rays.iter().step_by(20));
    for (n, ray) in every_twentieth {
        let mut surface = StoreSurface::new(project.store(), &index);
        let hit = surface.raycast(ray).unwrap();
        frame.push(dab(hit.position.to_f32(), 0.2, 0.02, 0.0, n));
    }
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), &mut index, &mut state, &frame).unwrap();
    project.end_stroke().unwrap();

    // Each call's inputs: a hit, a ray a radius aside from it (the next pen sample's), and
    // where that ray hits.
    let hits: Vec<(Hit, Ray, Hit)> = rays
        .iter()
        .map(|ray| {
            let mut surface = StoreSurface::new(project.store(), &index);
            let hit = surface.raycast(ray).unwrap();
            let side = hit.normal.cross(DVec3::Z).normalized().unwrap_or(DVec3::X);
            let next = Ray::new(ray.origin() + side * radius, ray.direction()).unwrap();
            let beside = surface.raycast(&next).unwrap_or(hit);
            (hit, next, beside)
        })
        .collect();
    let time = |store: &mut TileStore, cold: bool, f: &dyn Fn(&mut StoreSurface<'_>, usize)| {
        let mut total = Duration::ZERO;
        let mut worst = Duration::ZERO;
        for k in 0..hits.len() {
            if cold {
                store.evict_all().unwrap();
            }
            let mut surface = StoreSurface::new(store, &index);
            let start = Instant::now();
            f(&mut surface, k);
            let took = start.elapsed();
            assert!(surface.error().is_none(), "no error");
            total += took;
            worst = worst.max(took);
        }
        let n = u32::try_from(hits.len()).unwrap();
        (total / n, worst)
    };
    let raycast = |surface: &mut StoreSurface<'_>, k: usize| {
        surface.raycast(&hits[k].1);
    };
    let continuous = |surface: &mut StoreSurface<'_>, k: usize| {
        surface.continuous(&hits[k].0, &hits[k].2, reach);
    };
    let closest = |surface: &mut StoreSurface<'_>, k: usize| {
        surface.closest_on_region(&hits[k].1, &hits[k].0, reach);
    };
    let walk = |surface: &mut StoreSurface<'_>, k: usize| {
        let start = surface.tile_of(&hits[k].0).unwrap();
        surface
            .walk(start, hits[k].0.position, reach, None)
            .unwrap();
    };
    let costs: [(&str, &Call<'_>); 4] = [
        ("raycast", &raycast),
        ("continuous", &continuous),
        ("closest_on_region", &closest),
        ("walk", &walk),
    ];
    {
        let mut surface = StoreSurface::new(project.store(), &index);
        let sizes: Vec<usize> = hits
            .iter()
            .map(|(hit, _, _)| {
                let start = surface.tile_of(hit).unwrap();
                surface
                    .walk(start, hit.position, reach, None)
                    .unwrap()
                    .len()
            })
            .collect();
        println!(
            "region: mean {} tiles, most {}",
            sizes.iter().sum::<usize>() / sizes.len(),
            sizes.iter().max().unwrap()
        );
    }
    for (name, f) in costs {
        let (cold_mean, cold_worst) = time(project.store(), true, f);
        // Once to warm the tiles, then timed.
        time(project.store(), false, f);
        let (warm_mean, warm_worst) = time(project.store(), false, f);
        println!(
            "{name}: warm mean {:.1} µs, worst {:.1} µs; cold mean {:.1} µs, worst {:.1} µs",
            warm_mean.as_secs_f64() * 1e6,
            warm_worst.as_secs_f64() * 1e6,
            cold_mean.as_secs_f64() * 1e6,
            cold_worst.as_secs_f64() * 1e6,
        );
    }
}
