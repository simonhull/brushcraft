//! Normal cones, and the spatial index over a real store: built from evaluated tiles, raycast
//! onto the sphere, queried, and updated after an edit.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::BTreeSet;
use std::sync::Arc;

use geom::{FaceId, cube_sphere};

use super::bvh::tests::Rng;
use super::*;
use crate::file::ProjectFile;
use crate::fixture::meta_for;
use crate::slot::EvaluatedSlot;
use crate::test_dir::TestDir;
use crate::tiles::StoreConfig;
use crate::tiles::tests::{fixed, with_reserved_shares};

mod stored;

/// The bounds of every tile of `index`, in its order, as bits: what "bit for bit" compares.
pub(crate) fn index_bits(index: &SpatialIndex) -> Vec<(TileKey, [u32; TileBounds::BITS])> {
    index
        .bvh()
        .iter()
        .map(|(key, bounds)| (*key, bounds.to_bits()))
        .collect()
}

/// Loads the index of `level` from what `store` keeps, checks it is, bit for bit, the index
/// [`SpatialIndex::build`] makes of the store's state, and returns what the load found stored.
pub(crate) fn assert_loads_as_built(store: &mut TileStore, level: Level, what: &str) -> LoadStats {
    let (loaded, stats) = SpatialIndex::load(store, level, |_, _| {}).unwrap();
    let built = SpatialIndex::build(store, level, |_, _| {}).unwrap();
    assert!(
        index_bits(&loaded) == index_bits(&built),
        "{what}: the loaded index is the built one, bit for bit ({stats:?})"
    );
    assert_eq!(stats.stored + stats.evaluated, built.bvh().len(), "{what}");
    stats
}

/// The level the store-backed tests index: whole-face tiles of 16 × 16 quads.
const LEVEL: u8 = 4;

/// A store over a fresh file of `mesh`, with room for every tile of the tests' levels.
fn store_over(dir: &TestDir, mesh: &Arc<BaseMesh>) -> TileStore {
    store_with_workers(dir, mesh, 2)
}

/// [`store_over`], with `workers` evaluation workers.
fn store_with_workers(dir: &TestDir, mesh: &Arc<BaseMesh>, workers: usize) -> TileStore {
    let file = ProjectFile::create(&dir.path("index"), meta_for(mesh), Arc::clone(mesh)).unwrap();
    let chunk_bytes = 8 * size_of::<EvaluatedSlot>();
    let config = with_reserved_shares(
        StoreConfig::new(64 * chunk_bytes, 3)
            .with_chunk_bytes(chunk_bytes)
            .with_workers(fixed(workers)),
    );
    let config = config.with_budget_bytes(config.budget_bytes.max(config.minimum_budget(mesh)));
    TileStore::open(file, config).unwrap()
}

/// How far inside the band of its vertices' radii a hit on the level-4 sphere may fall.
///
/// The analytic surface is a sphere, but not the unit sphere its base vertices lie on:
/// subdividing a base this coarse (each cube side split 2 × 2) shrinks it to a radius of about
/// 0.84, varying by a few percent between the cube's corners and its face centres. So the hits
/// are held to the band of radii the evaluated vertices span, less what a flat quad sags inside
/// it: a level-4 quad spans about 4° of arc across its diagonal, so its middle falls
/// `0.84 · (1 − cos 2°)` ≈ 5e-4 short of the sphere.
const SAG: f32 = 1e-3;

/// The tests' level.
fn level() -> Level {
    Level::new(LEVEL).unwrap()
}

#[test]
fn a_cone_holds_its_normals() {
    // A fan of normals up to 30° from z.
    let (sin, cos) = 30.0_f32.to_radians().sin_cos();
    let normals: Vec<Vec3> = (0..12_u16)
        .map(|k| {
            let (s, c) = (f32::from(k) * 30.0_f32.to_radians()).sin_cos();
            Vec3::new(sin * c, sin * s, cos)
        })
        .chain([Vec3::new(0.0, 0.0, 1.0)])
        .collect();
    let cone = NormalCone::from_normals(&normals);
    assert!(
        (cone.axis() - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-5,
        "the axis is the mean: {:?}",
        cone.axis()
    );
    assert!(
        (cone.cos_half_angle() - cos).abs() < 1e-4,
        "the half-angle is the widest normal's: {}",
        cone.cos_half_angle()
    );
    assert!(
        normals.iter().all(|&n| cone.contains(n)),
        "every normal is inside"
    );
    assert!(
        !cone.contains(Vec3::new(1.0, 0.0, 0.0)),
        "a sideways normal isn't"
    );
    assert!(!cone.contains(Vec3::ZERO), "nor is a zero one");
}

#[test]
fn normals_a_cone_cannot_bound_give_everything() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    assert_eq!(
        NormalCone::from_normals(&[]),
        NormalCone::EVERYTHING,
        "no normals"
    );
    assert_eq!(
        NormalCone::from_normals(&[up, -up]),
        NormalCone::EVERYTHING,
        "opposite normals average to nothing"
    );
    assert_eq!(
        NormalCone::from_normals(&[up, Vec3::new(f32::NAN, 0.0, 0.0)]),
        NormalCone::EVERYTHING,
        "a NaN normal"
    );
    assert!(
        NormalCone::EVERYTHING.contains(-up) && NormalCone::EVERYTHING.contains(up),
        "everything holds every direction"
    );
    // Spread over more than a hemisphere, but not evenly: a wide cone that still holds them.
    let wide = [up, Vec3::new(1.0, 0.0, -0.5), Vec3::new(-1.0, 0.0, -0.5)];
    let cone = NormalCone::from_normals(&wide);
    assert!(
        wide.iter().all(|&n| cone.contains(n)),
        "a wide cone holds them all"
    );
}

#[test]
fn tile_bounds_need_a_finite_position() {
    let up = [Vec3::new(0.0, 0.0, 1.0)];
    assert!(
        TileBounds::of_tile(&[Vec3::new(f32::NAN, 0.0, 0.0)], &up).is_none(),
        "NaN"
    );
    let b = TileBounds::of_tile(&[Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)], &up).unwrap();
    assert_eq!(
        b.aabb.max(),
        Vec3::new(1.0, 1.0, 1.0),
        "the box of the positions"
    );
}

/// The cube sphere of 24 faces, its store, and the index over level 4, with every tile's
/// evaluated positions (by key).
struct Sphere {
    /// Keeps the store's file until the test ends.
    _dir: TestDir,
    /// The store the index reads.
    store: TileStore,
    /// The index under test.
    index: SpatialIndex,
    /// Every tile's evaluated positions.
    positions: Vec<(TileKey, Vec<Vec3>)>,
}

/// Builds [`Sphere`], checking the progress reports as it goes.
fn sphere() -> Sphere {
    let dir = TestDir::new();
    let mesh = Arc::new(cube_sphere(2).unwrap());
    let mut store = store_over(&dir, &mesh);
    let mut reports = Vec::new();
    let index = SpatialIndex::build(&mut store, level(), |done, total| {
        reports.push((done, total));
    })
    .unwrap();
    assert_eq!(index.level(), level(), "the level asked for");
    assert_eq!(index.bvh().len(), 24, "a whole-face tile per face");
    assert_eq!(
        reports,
        (1..=24).map(|done| (done, 24)).collect::<Vec<_>>(),
        "one report per tile, counting up to the total"
    );
    let positions = tiles_of(&mesh, level())
        .map(|key| (key, store.evaluate(key).unwrap().to_vec()))
        .collect();
    Sphere {
        _dir: dir,
        store,
        index,
        positions,
    }
}

#[test]
fn bounds_hold_every_evaluated_position_and_normal() {
    let mut fx = sphere();
    for (key, positions) in &fx.positions {
        let b = *fx.index.bvh().bounds(key).unwrap();
        assert!(
            positions.iter().all(|&p| b.aabb.contains(p)),
            "{key:?}'s box holds it"
        );
        let normals = fx.store.normals(*key).unwrap();
        assert!(
            normals.iter().all(|&n| b.cone.contains(n)),
            "{key:?}'s cone holds its normals"
        );
        assert!(
            b.cone.cos_half_angle() > 0.5,
            "a sixth of a sphere's normals fit in a cone well under 60°: {}",
            b.cone.cos_half_angle()
        );
    }
}

/// Where `ray` first meets the sphere of `radius` around the origin (or comes closest to it).
fn near_sphere_hit(ray: &Ray, radius: f32) -> f32 {
    let (origin, direction) = (ray.origin(), ray.direction());
    let quadratic = direction.dot(direction);
    let linear = 2.0 * direction.dot(origin);
    let constant = origin.dot(origin) - radius * radius;
    let discriminant = (linear * linear - 4.0 * quadratic * constant).max(0.0);
    (-linear - discriminant.sqrt()) / (2.0 * quadratic)
}

#[test]
fn raycasts_hit_the_sphere() {
    let mut fx = sphere();
    let radii: Vec<f32> = fx
        .positions
        .iter()
        .flat_map(|(_, p)| p.iter().map(|p| p.length()))
        .collect();
    let (least, most) = radii.iter().fold((f32::INFINITY, 0.0_f32), |(lo, hi), &r| {
        (lo.min(r), hi.max(r))
    });
    assert!(
        least > 0.8 && most < 0.9,
        "the surface is a shrunk sphere: radii {least}..{most}"
    );

    let mut rng = Rng(11);
    let mut hits = 0;
    for k in 0..300 {
        let toward = rng.point(-1.0, 1.0).normalized().unwrap();
        let origin = toward * 3.0;
        let aim = -toward + rng.point(-0.1, 0.1);
        let ray = Ray::new(origin, aim).unwrap();
        let hit = fx.index.raycast(&mut fx.store, &ray).unwrap();
        // Brute force: every tile's quads.
        let brute = fx
            .positions
            .iter()
            .filter_map(|(key, p)| {
                intersect_tile(&ray, p, key.side()).map(|t| Hit { key: *key, t })
            })
            .min_by(|a, b| a.t.total_cmp(&b.t));
        assert_eq!(hit, brute, "ray {k}: the index finds the nearest quad hit");
        let hit = hit.unwrap_or_else(|| panic!("ray {k} aimed at the sphere misses it"));
        hits += 1;
        let point = ray.at(hit.t);
        assert!(
            (least - SAG..=most + SAG).contains(&point.length()),
            "ray {k} hits at radius {} (surface {least}..{most})",
            point.length()
        );
        // The hit is on the near side: where the ray first meets a sphere of that radius.
        let near = near_sphere_hit(&ray, point.length());
        assert!(
            (hit.t - near).abs() < 1e-3,
            "ray {k} hits the near side: {} vs {near}",
            hit.t
        );
        let bounds = fx.index.bvh().bounds(&hit.key).unwrap();
        assert!(
            bounds.aabb.distance_squared(point) < 1e-10,
            "ray {k}'s hit is in its tile's box"
        );
    }
    assert_eq!(hits, 300, "every aimed ray hits");
    for _ in 0..50 {
        let away = rng.point(-1.0, 1.0).normalized().unwrap();
        let ray = Ray::new(away * 3.0, away).unwrap();
        assert_eq!(
            fx.index.raycast(&mut fx.store, &ray).unwrap(),
            None,
            "a ray going away misses"
        );
    }
}

#[test]
fn a_sphere_query_finds_every_tile_with_a_vertex_inside() {
    let fx = sphere();
    let mut rng = Rng(12);
    for _ in 0..200 {
        let center = rng.point(-1.0, 1.0).normalized().unwrap();
        let radius = rng.range(0.05, 0.6);
        let found: Vec<TileKey> = fx.index.query_sphere(center, radius).collect();
        for (key, p) in &fx.positions {
            if p.iter().any(|&p| (p - center).length() <= radius) {
                assert!(found.contains(key), "{key:?} has a vertex in the sphere");
            }
        }
    }
}

/// The bounds the normals wave computes are, bit for bit, the bounds read back from the store,
/// at one worker and at four.
///
/// After edits on faces' borders, corners and insides, every tile is evaluated with
/// [`TileStore::evaluate_many_with_bounds`]: the wave computes the normals of those the edits
/// reached (stale or dropped), and each bound it gives equals [`SpatialIndex::bounds_of`]'s. A
/// refit of every tile with them makes the index [`SpatialIndex::build`] makes. Once another
/// correction is written, they are no longer used.
#[test]
fn bounds_from_the_normals_wave_are_the_bounds_read_back() {
    for workers in [1, 4] {
        let dir = TestDir::new();
        let mesh = Arc::new(cube_sphere(2).unwrap());
        let mut store = store_with_workers(&dir, &mesh, workers);
        let mut index = SpatialIndex::build(&mut store, level(), |_, _| {}).unwrap();
        store.begin_op().unwrap();
        let side = Level::new(LEVEL).unwrap().tile_quads() + 1;
        for (face, i, j) in [(0, 0, 3), (0, side / 2, side / 2), (5, side - 1, side - 1)] {
            store
                .set_correction(FaceId::new(face), level(), i, j, Vec3::new(0.0, 0.0, 0.2))
                .unwrap();
        }
        let changed: Vec<TileKey> = tiles_of(&mesh, level()).collect();
        let wave = store.evaluate_many_with_bounds(&changed).unwrap();
        let edited: BTreeSet<TileKey> = [0, 5]
            .into_iter()
            .map(|face| TileKey::displacement(FaceId::new(face), level(), 0, 0).unwrap())
            .collect();
        assert!(
            edited.iter().all(|key| wave.current(&store, key).is_some()),
            "{workers} workers: the edited tiles' normals were computed"
        );
        assert!(
            wave.len() < changed.len(),
            "{workers} workers: not the rest"
        );
        for key in &changed {
            let Some(from_wave) = wave.current(&store, key) else {
                continue;
            };
            let read_back = SpatialIndex::bounds_of(&mut store, *key).unwrap();
            assert_eq!(
                from_wave.to_bits(),
                read_back.to_bits(),
                "{workers} workers: {key:?}'s bounds, bit for bit"
            );
        }
        index.refit_with(&mut store, &changed, &wave).unwrap();
        let built = SpatialIndex::build(&mut store, level(), |_, _| {}).unwrap();
        let (mut refitted, mut built) = (index_bits(&index), index_bits(&built));
        refitted.sort_unstable();
        built.sort_unstable();
        assert!(
            refitted == built,
            "{workers} workers: the refitted index holds the bounds a build finds"
        );
        store
            .set_correction(FaceId::new(0), level(), 1, 1, Vec3::new(0.0, 0.0, 0.1))
            .unwrap();
        assert!(
            changed
                .iter()
                .all(|key| wave.current(&store, key).is_none()),
            "{workers} workers: a write since leaves them unused"
        );
    }
}

/// After a refit, no bound the store keeps is stale: every tile whose normals read an edited
/// one has the bounds the refit recorded for it, or none known, never its old ones.
///
/// The edit is inside one tile, and only that tile is refitted, so its neighbours, whose cones
/// the edit may turn, must be left unknown.
#[test]
fn no_bound_is_stale_after_a_refit() {
    let dir = TestDir::new();
    let mesh = Arc::new(cube_sphere(2).unwrap());
    let mut store = store_over(&dir, &mesh);
    let mut index = SpatialIndex::build(&mut store, level(), |_, _| {}).unwrap();
    store.begin_op().unwrap();
    store
        .set_correction(FaceId::new(3), level(), 7, 4, Vec3::new(0.0, 0.0, 0.3))
        .unwrap();
    let edited = TileKey::displacement(FaceId::new(3), level(), 0, 0).unwrap();
    index.update(&mut store, &[edited]).unwrap();
    let all: Vec<TileKey> = tiles_of(&mesh, level()).collect();
    let stored = store.stored_bounds(&all).unwrap();
    let readers = store.normal_inputs(&edited).unwrap().into_owned();
    assert!(readers.len() > 1, "the edited tile has neighbours");
    for (key, kept) in all.iter().zip(stored) {
        let now = SpatialIndex::bounds_of(&mut store, *key).unwrap();
        match kept {
            Some(kept) => assert_eq!(kept.to_bits(), now.to_bits(), "{key:?}: current"),
            None => assert!(readers.contains(key), "{key:?}: only a reader is unknown"),
        }
    }
    let unknown = readers.iter().filter(|key| **key != edited);
    for key in unknown {
        assert!(
            store.stored_bounds(&[*key]).unwrap()[0].is_none(),
            "{key:?}: its cone may have turned, and it wasn't refitted"
        );
    }
}

#[test]
fn update_refits_to_an_edit() {
    let mut fx = sphere();
    let key = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let side = key.side();
    let (i, j) = (side / 2, side / 2);
    let at = usize::try_from(j * side + i).unwrap();
    fx.store.begin_op().unwrap();
    fx.store
        .set_correction(FaceId::new(0), level(), i, j, Vec3::new(0.0, 0.0, 0.5))
        .unwrap();
    let moved = fx.store.evaluate(key).unwrap()[at];
    let before = *fx.index.bvh().bounds(&key).unwrap();
    assert!(
        !before.aabb.contains(moved),
        "the edit moves the vertex out of the old box"
    );
    fx.index.update(&mut fx.store, &[key]).unwrap();
    let after = *fx.index.bvh().bounds(&key).unwrap();
    assert!(after.aabb.contains(moved), "the new box holds it");
    assert!(
        fx.index.bvh().aabb().unwrap().contains(moved),
        "so does the root's"
    );
    assert!(
        fx.index.query_sphere(moved, 1e-3).any(|k| k == key),
        "a query there finds the tile"
    );
    let unknown = TileKey::displacement(FaceId::new(0), Level::new(5).unwrap(), 0, 0).unwrap();
    assert!(
        matches!(
            fx.index.update(&mut fx.store, &[unknown]),
            Err(IndexError::UnknownTile(k)) if k == unknown
        ),
        "a tile of another level isn't in the index"
    );
}

#[test]
fn level_zero_is_not_indexed() {
    let dir = TestDir::new();
    let mesh = Arc::new(cube_sphere(1).unwrap());
    let mut store = store_over(&dir, &mesh);
    let err = SpatialIndex::build(&mut store, Level::ZERO, |_, _| {}).unwrap_err();
    assert!(
        matches!(err, IndexError::Cache(CacheError::NotATile(_))),
        "level 0 has no tiles: {err:?}"
    );
}

#[test]
fn rays_along_the_seams_hit_as_brute_force_does() {
    let mut fx = sphere();
    let on_seam = fx
        .index
        .bvh()
        .iter()
        .filter(|(_, b)| b.aabb.min().x == 0.0 || b.aabb.max().x == 0.0)
        .count();
    assert!(
        on_seam >= 8,
        "the tiles along the x = 0 seams have a face of their box on that plane: {on_seam}"
    );
    let mut rng = Rng(13);
    for k in 0..200 {
        // Within the plane x = 0: from outside the sphere towards it, aimed off the centre.
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let (sin, cos) = angle.sin_cos();
        let origin = Vec3::new(0.0, 3.0 * cos, 3.0 * sin);
        let aim = Vec3::new(
            0.0,
            -cos + rng.range(-0.2, 0.2),
            -sin + rng.range(-0.2, 0.2),
        );
        let ray = Ray::new(origin, aim).unwrap();
        let hit = fx.index.raycast(&mut fx.store, &ray).unwrap();
        let brute = fx
            .positions
            .iter()
            .filter_map(|(key, p)| intersect_tile(&ray, p, key.side()))
            .reduce(f32::min);
        let hit = hit.unwrap_or_else(|| panic!("seam ray {k} misses: {ray:?}"));
        assert_eq!(Some(hit.t), brute, "seam ray {k}: the nearest quad hit");
        let (_, positions) = fx
            .positions
            .iter()
            .find(|(key, _)| *key == hit.key)
            .unwrap();
        assert_eq!(
            intersect_tile(&ray, positions, hit.key.side()),
            Some(hit.t),
            "seam ray {k}: the tile named is one hit there"
        );
    }
}
