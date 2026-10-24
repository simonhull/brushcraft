//! The draw step against `geom`'s dense evaluation of the same edits, bit for bit.
//!
//! The oracle doesn't go through tiles at all: it evaluates every face of the mesh in full,
//! finds the deltas face by face, and writes them through a [`CorrectionMap`], which keeps twins
//! in step on its own. Its deltas come from the same brush function (so they match bit for bit),
//! and are checked against the smoothstep formula evaluated directly, in `f64`.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{
    BaseMesh, Channel, CorrectionMap, DenseEvaluation, FaceId, LayerId, Level, PositionSource,
    Region, cube_sphere, normals,
};
use store::{ProjectConfig, StoreConfig, Workers};

use super::*;

/// Exactly `n` evaluation workers.
///
/// # Panics
///
/// If `n` is 0.
fn fixed(n: usize) -> Workers {
    Workers::Fixed(std::num::NonZeroUsize::new(n).expect("a positive worker count"))
}

/// Distinguishes the directories of one process.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A fresh directory under the system temp dir, deleted with everything in it on drop.
#[derive(Debug)]
struct TestDir(PathBuf);

impl TestDir {
    /// Creates the directory.
    fn new() -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("brushcraft-sculpt-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// A new project over the cube (`cube_sphere(1)`, 6 faces) with room for every tile, recovery
/// files in `dir`, and its spatial index at `level`.
fn project(dir: &TestDir, level: Level) -> (Project, SpatialIndex) {
    project_over(dir, cube_sphere(1).unwrap(), level)
}

/// A new project over `mesh` with room for every tile of a few dozen faces, recovery files in
/// `dir`, and its spatial index at `level`.
fn project_over(dir: &TestDir, mesh: BaseMesh, level: Level) -> (Project, SpatialIndex) {
    let config = ProjectConfig {
        store: StoreConfig::new(64 << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(fixed(2)),
        recovery_dir: dir.0.join("recovery"),
    };
    let mut project = Project::new(mesh, config).unwrap();
    let index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
    (project, index)
}

/// Every tile of `level` of `mesh`, face by face.
fn tiles(mesh: &BaseMesh, level: Level) -> Vec<TileKey> {
    let per_side = u16::try_from(level.tiles_per_side()).unwrap();
    mesh.faces()
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).unwrap())
            })
        })
        .collect()
}

/// The delta the draw brush gives a point at `p`, from the smoothstep formula in `f64`.
///
/// `None` outside the brush. Inside is decided as the brush decides it, on the
/// squared distance in `f32`, so a point on the rim is outside for both.
fn formula(p: Vec3, params: DrawParams) -> Option<f64> {
    let center = params.center();
    let (dx, dy, dz) = (p.x - center.x, p.y - center.y, p.z - center.z);
    let radius = params.radius();
    if dx * dx + dy * dy + dz * dz >= radius * radius {
        return None;
    }
    let distance = f64::from(dx).hypot(f64::from(dy)).hypot(f64::from(dz));
    let t = distance / f64::from(radius);
    Some(f64::from(params.strength()) * (1.0 - t * t * (3.0 - 2.0 * t)))
}

/// One step of the oracle, written into `map`; returns the vertices moved.
///
/// Evaluates `map` densely at `level`, finds each vertex's delta, checks it against [`formula`],
/// and adds it to the `z` of the vertex's correction in `map`, once per vertex (a vertex on a face
/// edge is in the grids of every face sharing it).
///
/// Faces are visited in order and each grid row-major, as the store visits the vertices of each
/// of its tiles: an interior correction is quantized to a scale its tile's earlier writes set, so
/// the order of a tile's writes is part of the result.
fn oracle_step(
    mesh: &BaseMesh,
    level: Level,
    map: &mut CorrectionMap,
    params: DrawParams,
) -> usize {
    let dense = DenseEvaluation::new(mesh, level, map).unwrap();
    let topology = mesh.topology();
    let mut done = HashSet::new();
    let mut moved = 0;
    for face in mesh.faces() {
        let grid = dense.grid(face, level).unwrap();
        let mut deltas = vec![0.0; grid.as_slice().len()];
        let c = params.center();
        brush::draw(
            &Smoothstep,
            [c.x, c.y, c.z],
            params.radius(),
            params.strength(),
            bytemuck::cast_slice(grid.as_slice()),
            &mut deltas,
        );
        for ((i, j), &delta) in vertices(grid.side()).zip(&deltas) {
            let p = grid.get(i, j).unwrap();
            match formula(p, params) {
                Some(want) => assert!(
                    (f64::from(delta) - want).abs() <= 1e-6 * f64::from(params.strength().abs()),
                    "the delta at {p:?} is {delta}, the formula gives {want}"
                ),
                None => assert!(delta == 0.0, "{p:?} is outside, its delta is {delta}"),
            }
            let copies = topology.copies(face, level, i, j);
            let first = copies
                .iter()
                .map(|copy| (copy.face, copy.tx, copy.ty, copy.i, copy.j))
                .min();
            if delta == 0.0 || !done.insert(first.unwrap()) {
                continue;
            }
            let old = map.get(face, level, i, j);
            map.set(
                topology,
                face,
                level,
                i,
                j,
                Vec3::new(old.x, old.y, old.z + delta),
            )
            .unwrap();
            moved += 1;
        }
    }
    moved
}

/// Every vertex of every tile of `level` as `store` holds it: evaluated position and correction,
/// keyed by the tile and the vertex's coordinates within it.
fn snapshot(store: &mut TileStore, level: Level) -> BTreeMap<(TileKey, u32, u32), (Vec3, Vec3)> {
    let mut all = BTreeMap::new();
    for key in tiles(&store.mesh().clone(), level) {
        let positions = store.evaluate(key).unwrap().to_vec();
        let corrections = store.correction(key).unwrap();
        for ((i, j), p) in vertices(key.side()).zip(positions) {
            all.insert((key, i, j), (p, corrections.get(i, j).unwrap()));
        }
    }
    all
}

/// Asserts that `store` evaluates every tile of `level` as `map` does densely, and holds the same
/// corrections, bit for bit.
fn assert_as_dense(store: &mut TileStore, map: &CorrectionMap, level: Level, what: &str) {
    let mesh = store.mesh().clone();
    let dense = DenseEvaluation::new(&mesh, level, map).unwrap();
    for ((key, i, j), (p, c)) in snapshot(store, level) {
        let region = Region::of_tile(&key);
        let (gi, gj) = (region.x0() + i, region.y0() + j);
        let position = dense.grid(key.face(), level).unwrap().get(gi, gj).unwrap();
        assert_eq!(
            p.to_bits(),
            position.to_bits(),
            "{what}: position of {key:?} ({i}, {j})"
        );
        let correction = map.get(key.face(), level, gi, gj);
        assert_eq!(
            c.to_bits(),
            correction.to_bits(),
            "{what}: correction of {key:?} ({i}, {j})"
        );
    }
}

/// Asserts that every copy of every vertex of `level` agrees with the others, position and
/// correction, bit for bit.
fn assert_twins(store: &mut TileStore, level: Level) {
    let mesh = store.mesh().clone();
    let all = snapshot(store, level);
    let mut shared = 0;
    for (&(key, i, j), &(p, c)) in &all {
        let region = Region::of_tile(&key);
        let copies = mesh
            .topology()
            .copies(key.face(), level, region.x0() + i, region.y0() + j);
        for copy in copies.iter().filter(|copy| copy.key(level) != key) {
            let (q, d) = all[&(copy.key(level), copy.i, copy.j)];
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "positions of {key:?} ({i}, {j}) and {copy:?}"
            );
            assert_eq!(
                c.to_bits(),
                d.to_bits(),
                "corrections of {key:?} ({i}, {j}) and {copy:?}"
            );
            shared += 1;
        }
    }
    assert!(shared > 0, "the level has twins to compare");
}

/// Asserts that `index` holds, for every tile, the bounds an index built from scratch finds.
fn assert_bounds_fresh(store: &mut TileStore, index: &SpatialIndex) {
    let fresh = SpatialIndex::build(store, index.level(), |_, _| {}).unwrap();
    assert_eq!(fresh.bvh().len(), index.bvh().len(), "the same tiles");
    for (key, bounds) in fresh.bvh().iter() {
        assert_eq!(
            index.bvh().bounds(key),
            Some(bounds),
            "the bounds of {key:?}"
        );
    }
    assert_eq!(
        index.bvh().aabb(),
        fresh.bvh().aabb(),
        "the box of everything"
    );
}

/// The cube's face-grid vertex `(i, j)` of `face` at `level`, as evaluated before any edit.
fn vertex(level: Level, face: u32, i: u32, j: u32) -> Vec3 {
    vertex_of(&cube_sphere(1).unwrap(), level, face, i, j)
}

/// `mesh`'s face-grid vertex `(i, j)` of `face` at `level`, as evaluated before any edit.
fn vertex_of(mesh: &BaseMesh, level: Level, face: u32, i: u32, j: u32) -> Vec3 {
    let dense = DenseEvaluation::new(mesh, level, &CorrectionMap::new()).unwrap();
    dense
        .grid(FaceId::new(face), level)
        .unwrap()
        .get(i, j)
        .unwrap()
}

#[test]
fn one_step_moves_exactly_the_vertices_within_the_radius_by_the_falloff() {
    let dir = TestDir::new();
    let level = Level::new(6).unwrap();
    let (mut project, mut index) = project(&dir, level);
    // Near an edge of face 0, so the brush reaches across it into the next face.
    let params = DrawParams::new(vertex(level, 0, 60, 30), 0.3, 0.05).unwrap();
    let before = snapshot(project.store(), level);

    project.begin_stroke().unwrap();
    let stats = draw_step(&mut project, &mut index, params).unwrap();
    project.end_stroke().unwrap();

    let mut map = CorrectionMap::new();
    let moved = oracle_step(project.store().mesh(), level, &mut map, params);
    assert_eq!(stats.vertices, moved, "vertices moved");
    assert!(
        moved > 500,
        "the brush covers a good part of the level: {moved}"
    );
    assert!(
        stats.tiles_edited >= 2,
        "it reaches across a face edge: {stats:?}"
    );
    assert_as_dense(project.store(), &map, level, "after the step");

    // Moved means within the radius, and pushed along the frame normal alone.
    let after = snapshot(project.store(), level);
    let mut changed = 0;
    for (place, &(p, c)) in &after {
        let (old, old_c) = before[place];
        if p.to_bits() == old.to_bits() {
            continue;
        }
        changed += 1;
        let delta = formula(old, params).unwrap_or_else(|| panic!("{old:?} moved from outside"));
        assert_eq!(
            (c.x, c.y),
            (old_c.x, old_c.y),
            "only the normal axis changes"
        );
        let step = f64::from(c.z - old_c.z);
        // Interior corrections are quantized: within a step of the tile's final scale, at most
        // twice `strength / 32767`.
        assert!(
            (step - delta).abs() <= 4e-6,
            "{old:?} moved by {step}, the formula gives {delta}"
        );
    }
    assert!(changed > 0, "the step moved the surface");
}

#[test]
fn twins_stay_bit_identical() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    let (mut project, mut index) = project(&dir, level);
    // On a cube corner, where three faces and many tiles meet.
    let corner = vertex(level, 0, 0, 0);
    project.begin_stroke().unwrap();
    for (k, strength) in [0.04, -0.02, 0.03].into_iter().enumerate() {
        let shift = 0.05 * f32::from(u8::try_from(k).unwrap());
        let center = Vec3::new(corner.x + shift, corner.y, corner.z - shift);
        draw_step(
            &mut project,
            &mut index,
            DrawParams::new(center, 0.25, strength).unwrap(),
        )
        .unwrap();
    }
    project.end_stroke().unwrap();
    assert_twins(project.store(), level);
}

#[test]
fn bounds_after_a_step_equal_bounds_built_from_scratch() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    let (mut project, mut index) = project(&dir, level);
    let params = DrawParams::new(vertex(level, 2, 100, 128), 0.2, 0.1).unwrap();
    project.begin_stroke().unwrap();
    let stats = draw_step(&mut project, &mut index, params).unwrap();
    project.end_stroke().unwrap();
    assert!(
        stats.tiles_updated > stats.tiles_edited,
        "neighbours' normals turn too: {stats:?}"
    );
    assert_bounds_fresh(project.store(), &index);
}

#[test]
fn a_stroke_of_overlapping_steps_matches_dense_after_each_step() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    let (mut project, mut index) = project(&dir, level);
    let mesh = project.store().mesh().clone();
    let mut map = CorrectionMap::new();
    // Across face 4 and its tiles' borders (at 64 on each axis), each step under a radius on from
    // the last, some pushing in: every step reads the corrections the steps before it wrote.
    let steps = [0.05, 0.03, -0.04, 0.06, 0.02];
    project.begin_stroke().unwrap();
    for (k, strength) in (0_u32..).zip(steps) {
        let center = vertex(level, 4, 40 + 15 * k, 90 - 12 * k);
        let params = DrawParams::new(center, 0.1, strength).unwrap();
        let stats = draw_step(&mut project, &mut index, params).unwrap();
        let moved = oracle_step(&mesh, level, &mut map, params);
        assert_eq!(stats.vertices, moved, "step {k}: vertices moved");
        assert!(moved > 0, "step {k} reaches the surface");
        assert_as_dense(project.store(), &map, level, &format!("after step {k}"));
    }
    project.end_stroke().unwrap();
    assert_bounds_fresh(project.store(), &index);
}

#[test]
fn a_step_at_a_valence_5_vertex_matches_dense_and_keeps_its_five_copies_in_step() {
    let dir = TestDir::new();
    let level = Level::new(6).unwrap();
    let mesh = geom::fixtures::subdivided_pentagonal_prism().unwrap();
    let (mut project, mut index) = project_over(&dir, mesh.clone(), level);
    // Face 5's corner 2, at face-grid (n, n), is the top cap's centre, where five faces meet.
    let n = level.face_quads();
    let pole = vertex_of(&mesh, level, 5, n, n);
    let copies = mesh.topology().copies(FaceId::new(5), level, n, n);
    assert_eq!(copies.len(), 5, "a copy in each of the five faces");
    let mut map = CorrectionMap::new();
    project.begin_stroke().unwrap();
    // On the pole, then off it so the falloff crosses it unevenly, pushing out and then in.
    for (k, (shift, strength)) in [(0.0, 0.05), (0.08, -0.03)].into_iter().enumerate() {
        let center = Vec3::new(pole.x + shift, pole.y, pole.z);
        let params = DrawParams::new(center, 0.3, strength).unwrap();
        let stats = draw_step(&mut project, &mut index, params).unwrap();
        let moved = oracle_step(&mesh, level, &mut map, params);
        assert_eq!(stats.vertices, moved, "step {k}: vertices moved");
        assert!(
            stats.tiles_edited >= 5,
            "step {k}: the brush reaches all five faces: {stats:?}"
        );
        assert_as_dense(project.store(), &map, level, &format!("after step {k}"));
    }
    project.end_stroke().unwrap();
    assert_twins(project.store(), level);
    assert_bounds_fresh(project.store(), &index);
}

/// Evaluated positions of whole tiles, each vertex answered from its containing tile: the
/// source of a full recompute of normals.
struct Snapshot(BTreeMap<TileKey, Vec<Vec3>>);

impl PositionSource for Snapshot {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        let index = usize::try_from(lj * key.side() + li).ok()?;
        self.0.get(&key)?.get(index).copied()
    }
}

/// Asserts that the store's normals of every tile of `level` are a full recompute's from its
/// evaluated positions, bit for bit.
fn assert_normals_recomputed(store: &mut TileStore, level: Level, what: &str) {
    let mesh = store.mesh().clone();
    let keys = tiles(&mesh, level);
    let snapshot = Snapshot(
        keys.iter()
            .map(|&key| (key, store.evaluate(key).unwrap().to_vec()))
            .collect(),
    );
    for key in keys {
        let recomputed = normals(&mesh, &snapshot, Region::of_tile(&key)).unwrap();
        let got = store.normals(key).unwrap();
        for (k, (have, want)) in got.iter().zip(recomputed.as_slice()).enumerate() {
            assert_eq!(have.to_bits(), want.to_bits(), "{what}: {key:?} vertex {k}");
        }
    }
}

#[test]
fn normals_after_random_steps_equal_a_full_recompute_at_any_thread_count() {
    // The subdivided pentagonal prism at level 7: every tile has a face corner, of valence 3, 4
    // or 5, and tile corners on face edges. Small dabs near the corners and borders leave the
    // neighbours' normals to be brought up to date from their aprons and corner rings.
    let level = Level::new(7).unwrap();
    let mesh = geom::fixtures::subdivided_pentagonal_prism().unwrap();
    let n = level.face_quads();
    let hardware = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let mut finals = Vec::new();
    for threads in [1, hardware] {
        let dir = TestDir::new();
        let config = ProjectConfig {
            store: StoreConfig::new(64 << 20, 1)
                .with_chunk_bytes(1 << 20)
                .with_workers(fixed(threads)),
            recovery_dir: dir.0.join("recovery"),
        };
        let mut project = Project::new(mesh.clone(), config).unwrap();
        let mut index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
        let mut seed: u32 = 7;
        let mut next = |below: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % below
        };
        project.begin_stroke().unwrap();
        for k in 0..10 {
            // Near a face corner or a tile corner, a few vertices off it.
            let face = next(mesh.face_count());
            let corner = [0, n / 2, n][usize::try_from(next(3)).unwrap()];
            let other = [0, n / 2, n][usize::try_from(next(3)).unwrap()];
            let (i, j) = (
                corner.abs_diff(next(4)).min(n),
                other.abs_diff(next(4)).min(n),
            );
            let center = vertex_of(&mesh, level, face, i, j);
            let radius = 0.004 + 0.002 * f32::from(u8::try_from(next(4)).unwrap());
            let strength = if next(2) == 0 { 0.003 } else { -0.002 };
            let params = DrawParams::new(center, radius, strength).unwrap();
            draw_step(&mut project, &mut index, params).unwrap();
            assert_normals_recomputed(
                project.store(),
                level,
                &format!("{threads} threads, step {k} on face {face} near ({i}, {j})"),
            );
        }
        project.end_stroke().unwrap();
        let store = project.store();
        let all: Vec<Vec<[u32; 3]>> = tiles(&mesh, level)
            .into_iter()
            .map(|key| {
                store
                    .normals(key)
                    .unwrap()
                    .iter()
                    .map(|p| p.to_bits())
                    .collect()
            })
            .collect();
        finals.push(all);
    }
    assert!(
        finals.windows(2).all(|pair| pair[0] == pair[1]),
        "the same normals whatever the thread count"
    );
}

#[test]
fn a_brush_away_from_the_surface_changes_nothing() {
    let dir = TestDir::new();
    let level = Level::new(6).unwrap();
    let (mut project, mut index) = project(&dir, level);
    let before = snapshot(project.store(), level);
    project.begin_stroke().unwrap();
    let params = DrawParams::new(Vec3::new(0.0, 0.0, 0.0), 0.1, 1.0).unwrap();
    let stats = draw_step(&mut project, &mut index, params).unwrap();
    project.end_stroke().unwrap();
    assert_eq!(
        (
            stats.candidates,
            stats.tiles_edited,
            stats.vertices,
            stats.tiles_updated
        ),
        (0, 0, 0, 0),
        "inside the sphere, the brush reaches no tile"
    );
    assert_eq!(snapshot(project.store(), level), before, "nothing changed");
}

#[test]
fn draw_params_refuse_what_reaches_nowhere() {
    let center = Vec3::new(1.0, 0.0, 0.0);
    assert!(
        DrawParams::new(center, 0.1, -0.5).is_ok(),
        "a negative strength pushes in"
    );
    assert_eq!(
        DrawParams::new(Vec3::new(f32::NAN, 0.0, 0.0), 0.1, 1.0),
        Err(DrawParamsError::Center),
        "a NaN centre"
    );
    for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            DrawParams::new(center, radius, 1.0),
            Err(DrawParamsError::Radius),
            "radius {radius}"
        );
    }
    assert_eq!(
        DrawParams::new(center, 0.1, f32::INFINITY),
        Err(DrawParamsError::Strength),
        "an infinite strength"
    );
}

#[test]
fn phase_times_add_up() {
    let ms = Duration::from_millis;
    let mut times = PhaseTimes {
        query: ms(1),
        pin: ms(2),
        brush: ms(3),
        edit: ms(4),
        neighbours: ms(5),
        evaluate: ms(6),
        refit: ms(7),
    };
    assert_eq!(times.total(), ms(28), "the phases together");
    times += times;
    assert_eq!(times.total(), ms(56), "two steps together");
    assert_eq!(times.refit, ms(14), "phase by phase");
    assert_eq!(times.neighbours, ms(10), "the neighbours too");
}

/// A tile evicted before the workers could brush it is brushed from its positions read back
/// after evaluating it again, with the same result as brushing a copy of them.
#[test]
fn an_evicted_tile_is_brushed_as_from_its_positions() {
    let dir = TestDir::new();
    let level = Level::new(6).unwrap();
    let (mut project, _) = project(&dir, level);
    let params = DrawParams::new(vertex(level, 0, 60, 30), 0.3, 0.05).unwrap();
    let key = TileKey::displacement(FaceId::new(0), level, 0, 0).unwrap();
    let store = project.store();
    let copy = store.evaluate(key).unwrap().to_vec();
    let want = brush_tile(store.mesh().topology(), key, &copy, params);
    let got = brush_evicted(store, key, params).unwrap();
    assert!(want.reached, "the brush reaches the tile");
    assert_eq!(
        (got.reached, got.moved, got.deltas),
        (want.reached, want.moved, want.deltas),
        "the same brush"
    );
}
