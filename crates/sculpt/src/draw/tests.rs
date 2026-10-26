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
use store::{Project, ProjectConfig, StoreConfig, Workers};

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

/// The oracle's stroke state, as [`Stroke`] keeps it.
///
/// For each vertex the stroke has written, keyed by its least copy `(face, tx, ty, i, j)`: the
/// correction `z` it first found and the sum of its deltas. The oracle tests stay within the
/// default capacity, so no tile is ever released and the oracle doesn't model releases.
type OracleStroke = BTreeMap<(FaceId, u16, u16, u32, u32), (f32, f32)>;

/// One step of the oracle, written into `map`, as part of the stroke `stroke` holds the state
/// of; returns the vertices moved.
///
/// Evaluates `map` densely at `level`, finds each vertex's delta, checks it against [`formula`],
/// and writes the vertex's correction in `map` once per vertex (a vertex on a face edge is in the
/// grids of every face sharing it), as the store's stroke does: its `z` becomes the value the
/// stroke first found there plus the stroke's deltas for it so far, summed in `f32` in the same
/// order.
///
/// Faces are visited in order and each grid row-major, as the store visits the vertices of each
/// of its tiles: an interior correction is quantized to a scale its tile's earlier writes set, so
/// the order of a tile's writes is part of the result.
fn oracle_step(
    mesh: &BaseMesh,
    level: Level,
    map: &mut CorrectionMap,
    stroke: &mut OracleStroke,
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
            let first = first.unwrap();
            if delta == 0.0 || !done.insert(first) {
                continue;
            }
            let old = map.get(face, level, i, j);
            let (start, acc) = stroke.entry(first).or_insert((old.z, 0.0));
            *acc += delta;
            let z = *start + *acc;
            map.set(topology, face, level, i, j, Vec3::new(old.x, old.y, z))
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
    let mut stroke = Stroke::new(project.store());
    let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
    project.end_stroke().unwrap();
    assert_eq!(stats.tiles_unaccumulated, 0, "every tile keeps its state");

    let mut map = CorrectionMap::new();
    let moved = oracle_step(
        project.store().mesh(),
        level,
        &mut map,
        &mut OracleStroke::new(),
        params,
    );
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
    let mut stroke = Stroke::new(project.store());
    for (k, strength) in [0.04, -0.02, 0.03].into_iter().enumerate() {
        let shift = 0.05 * f32::from(u8::try_from(k).unwrap());
        let center = Vec3::new(corner.x + shift, corner.y, corner.z - shift);
        draw_step(
            project.store(),
            &mut index,
            &mut stroke,
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
    let mut stroke = Stroke::new(project.store());
    let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
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
    let mut oracle = OracleStroke::new();
    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    for (k, strength) in (0_u32..).zip(steps) {
        let center = vertex(level, 4, 40 + 15 * k, 90 - 12 * k);
        let params = DrawParams::new(center, 0.1, strength).unwrap();
        let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
        let moved = oracle_step(&mesh, level, &mut map, &mut oracle, params);
        assert_eq!(stats.vertices, moved, "step {k}: vertices moved");
        assert_eq!(
            stats.tiles_unaccumulated, 0,
            "step {k}: every tile keeps its state"
        );
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
    let mut oracle = OracleStroke::new();
    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    // On the pole, then off it so the falloff crosses it unevenly, pushing out and then in.
    for (k, (shift, strength)) in [(0.0, 0.05), (0.08, -0.03)].into_iter().enumerate() {
        let center = Vec3::new(pole.x + shift, pole.y, pole.z);
        let params = DrawParams::new(center, 0.3, strength).unwrap();
        let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
        let moved = oracle_step(&mesh, level, &mut map, &mut oracle, params);
        assert_eq!(stats.vertices, moved, "step {k}: vertices moved");
        assert_eq!(
            stats.tiles_unaccumulated, 0,
            "step {k}: every tile keeps its state"
        );
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
        let mut stroke = Stroke::new(project.store());
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
            draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
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
    let mut stroke = Stroke::new(project.store());
    let params = DrawParams::new(Vec3::new(0.0, 0.0, 0.0), 0.1, 1.0).unwrap();
    let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
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

/// Each vertex `(i, j)` of `key`: its correction's `z` and its evaluated position, row-major.
fn tile_state(store: &mut TileStore, key: TileKey) -> Vec<(u32, u32, f32, Vec3)> {
    let positions = store.evaluate(key).unwrap().to_vec();
    let corrections = store.correction(key).unwrap();
    vertices(key.side())
        .zip(positions)
        .map(|((i, j), p)| (i, j, corrections.get(i, j).unwrap().z, p))
        .collect()
}

/// Asserts that every vertex of `key` inside `light` moved from `before` by `dabs` times its delta.
///
/// Each within `steps` of the tile's final scale (and 1% of the expected move). Returns how many
/// vertices it checked, and how many of them are on the tile's border.
fn assert_moved_by_dabs(
    store: &mut TileStore,
    key: TileKey,
    before: &[(u32, u32, f32, Vec3)],
    light: DrawParams,
    dabs: f64,
    steps: f64,
) -> (usize, usize) {
    let side = key.side();
    let after = store.correction(key).unwrap();
    let final_scale = f64::from(after.scale());
    let (mut inside, mut border) = (0, 0);
    for &(i, j, z, p) in before {
        let Some(d) = formula(p, light) else {
            continue;
        };
        let moved = f64::from(after.get(i, j).unwrap().z) - f64::from(z);
        let expected = dabs * d;
        let on_border = i == 0 || j == 0 || i == side - 1 || j == side - 1;
        let place = if on_border { "border" } else { "interior" };
        assert!(
            (moved - expected).abs() <= steps * final_scale + 0.01 * expected.abs(),
            "{place} vertex ({i}, {j}) moved by {moved}, expected {expected} \
             (scale {final_scale})"
        );
        inside += 1;
        border += usize::from(on_border);
    }
    (inside, border)
}

#[test]
fn light_dabs_move_a_tiles_interior_as_far_as_its_border() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    let (mut project, mut index) = project(&dir, level);
    let key = TileKey::displacement(FaceId::new(4), level, 0, 0).unwrap();

    // A strong dab alone makes the tile's scale large.
    project.begin_stroke().unwrap();
    let mut first = Stroke::new(project.store());
    let strong = DrawParams::new(vertex(level, 4, 32, 32), 0.1, 0.05).unwrap();
    draw_step(project.store(), &mut index, &mut first, strong).unwrap();
    project.end_stroke().unwrap();
    let scale = project.store().correction(key).unwrap().scale();
    assert!(scale > 0.0, "the strong dab set a scale");

    // Light dabs near the tile's right border, each under half a step at most vertices.
    let light = DrawParams::new(vertex(level, 4, 60, 32), 0.06, 0.4 * scale).unwrap();
    let before = tile_state(project.store(), key);
    project.begin_stroke().unwrap();
    // A fresh stroke state, as a new stroke's.
    let mut stroke = Stroke::new(project.store());
    for _ in 0..10 {
        draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
    }
    project.end_stroke().unwrap();
    let (inside, border) = assert_moved_by_dabs(project.store(), key, &before, light, 10.0, 0.5);
    assert!(border > 0, "the brush reaches the tile's border");
    assert!(inside > border, "and its interior");
}

/// A light dab's strength in the tests below: 0.4 of the scale (about 2.5e-6) a strong dab
/// (radius 0.1, strength 0.05) gives a tile of the cube at level 7, so under half a step.
const LIGHT: f32 = 1.0e-6;

/// A new project over the cube with recovery files in `dir`, its spatial index at `level`, and
/// a store of `threads` workers setting aside stroke state for `stroke_tiles` tiles.
fn project_with(
    dir: &TestDir,
    level: Level,
    threads: usize,
    stroke_tiles: usize,
) -> (Project, SpatialIndex) {
    let config = ProjectConfig {
        store: StoreConfig::new(64 << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(fixed(threads))
            .with_stroke_tiles(stroke_tiles),
        recovery_dir: dir.0.join("recovery"),
    };
    let mut project = Project::new(cube_sphere(1).unwrap(), config).unwrap();
    let index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
    (project, index)
}

/// Makes the strong dab of [`light_dabs_move_a_tiles_interior_as_far_as_its_border`] at the
/// centre of tile `(face 4, 0, 0)` as a stroke of its own, and returns the tile and its scale.
///
/// It pushes the surface out by 0.05 around face-grid `(32, 32)`, so a light dab meant to reach
/// the tile goes where the surface hasn't moved: [`light_dab`]'s `(50, 32)`.
fn strong_stroke(project: &mut Project, index: &mut SpatialIndex, level: Level) -> (TileKey, f32) {
    let key = TileKey::displacement(FaceId::new(4), level, 0, 0).unwrap();
    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    let strong = DrawParams::new(vertex(level, 4, 32, 32), 0.1, 0.05).unwrap();
    draw_step(project.store(), index, &mut stroke, strong).unwrap();
    project.end_stroke().unwrap();
    let scale = project.store().correction(key).unwrap().scale();
    assert!(scale > 0.0, "the strong dab set a scale");
    (key, scale)
}

/// A light dab (radius 0.06, `0.4 × scale`) on `face` of the cube at `level`.
///
/// On face 4 it is at face-grid `(50, 32)`, in tile `(face 4, 0, 0)` clear of
/// [`strong_stroke`]'s dab; on another face at `(32, 32)`, the centre of its tile `(0, 0)`. Each
/// reaches that one tile alone.
fn light_dab(level: Level, face: u32, scale: f32) -> DrawParams {
    let i = if face == 4 { 50 } else { 32 };
    DrawParams::new(vertex(level, face, i, 32), 0.06, 0.4 * scale).unwrap()
}

#[test]
fn a_strokes_writes_are_the_same_at_any_worker_count() {
    let level = Level::new(7).unwrap();
    let mut finals = Vec::new();
    for threads in [1, 2, 8] {
        let dir = TestDir::new();
        let (mut project, mut index) = project_with(&dir, level, threads, 512);
        project.begin_stroke().unwrap();
        let mut stroke = Stroke::new(project.store());
        for k in 0..20 {
            let center = vertex(level, 4, 20 + 4 * k, 32);
            let params = if k % 2 == 0 {
                DrawParams::new(center, 0.1, 0.05)
            } else {
                DrawParams::new(center, 0.06, LIGHT)
            };
            draw_step(project.store(), &mut index, &mut stroke, params.unwrap()).unwrap();
        }
        project.end_stroke().unwrap();
        finals.push(snapshot(project.store(), level));
    }
    assert!(
        finals.windows(2).all(|pair| pair[0] == pair[1]),
        "the same corrections and positions at 1, 2 and 8 workers"
    );
}

#[test]
fn a_released_tile_revisited_stays_within_one_step() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    // Three tiles' state: the stroke below edits four tiles, one face at a time, so the first
    // is released for the fourth, then given state again on its revisit.
    let (mut project, mut index) = project_with(&dir, level, 2, 3);
    let (key, scale) = strong_stroke(&mut project, &mut index, level);
    let light = light_dab(level, 4, scale);
    let before = tile_state(project.store(), key);

    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    for (n, face) in [4, 1, 2, 3, 4].into_iter().enumerate() {
        if n == 4 {
            assert!(
                !stroke.holds(&key),
                "the tile was released before its revisit"
            );
        }
        let params = light_dab(level, face, scale);
        for _ in 0..5 {
            let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
            assert_eq!(stats.tiles_unaccumulated, 0, "a tile is free to release");
            assert!(stroke.tiles_held() <= 3, "no more than the store set aside");
        }
    }
    project.end_stroke().unwrap();
    let (inside, _) = assert_moved_by_dabs(project.store(), key, &before, light, 10.0, 1.0);
    assert!(inside > 0, "the brush reaches the tile");
}

#[test]
fn tiles_edited_in_the_last_dabs_keep_their_state() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    // Two tiles' state, and dabs cycling over three tiles: each tile is edited every third dab,
    // within the protected dabs, so the two tiles that got state first keep it, and the third
    // is written per dab.
    let (mut project, mut index) = project_with(&dir, level, 2, 2);
    let (key, scale) = strong_stroke(&mut project, &mut index, level);
    let light = light_dab(level, 4, scale);
    let before = tile_state(project.store(), key);

    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    for _ in 0..10 {
        for face in [4, 1, 2] {
            let params = light_dab(level, face, scale);
            let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
            let want = usize::from(face == 2);
            assert_eq!(stats.tiles_unaccumulated, want, "face {face}");
        }
    }
    project.end_stroke().unwrap();
    let (inside, _) = assert_moved_by_dabs(project.store(), key, &before, light, 10.0, 0.5);
    assert!(inside > 0, "the brush reaches the tile");
}

#[test]
fn a_dab_past_the_thrash_limit_writes_its_overflow_tiles_per_dab_and_does_not_fail() {
    let level = Level::new(7).unwrap();
    let params = DrawParams::new(vertex(level, 4, 64, 64), 0.3, 0.05).unwrap();
    let run = |threads: usize, stroke_tiles: usize| {
        let dir = TestDir::new();
        let (mut project, mut index) = project_with(&dir, level, threads, stroke_tiles);
        project.begin_stroke().unwrap();
        let mut stroke = Stroke::new(project.store());
        let stats = draw_step(project.store(), &mut index, &mut stroke, params).unwrap();
        project.end_stroke().unwrap();
        assert_eq!(
            stroke.tiles_held(),
            stroke_tiles.min(stats.tiles_edited),
            "as many tiles hold state as there is room for"
        );
        (stats, snapshot(project.store(), level))
    };
    let (stats, one) = run(1, 2);
    assert!(
        stats.tiles_edited >= 4,
        "the dab edits all four of face 4's tiles, past the two with room: {stats:?}"
    );
    assert_eq!(
        stats.tiles_unaccumulated,
        stats.tiles_edited - 2,
        "two tiles get state, the rest are written per dab"
    );
    let (_, eight) = run(8, 2);
    assert!(one == eight, "the same at 1 and 8 workers");
    let (none, per_dab) = run(1, 0);
    assert_eq!(
        none.tiles_unaccumulated, none.tiles_edited,
        "no state at all"
    );
    assert!(
        one == per_dab,
        "a first dab's start plus its delta is its stored value plus its delta"
    );
}

/// Three light dabs across the border between tiles `(face 4, 0, 0)` and `(face 4, 1, 0)`.
///
/// The border is face-grid `i = 64`, whose vertices the first tile owns. The dabs are near
/// `(61 + k, 30 + k)`, of `0.4 × scale`, `scale` the first tile's (see [`strong_stroke`]).
fn border_dabs(
    project: &mut Project,
    index: &mut SpatialIndex,
    stroke: &mut Stroke,
    level: Level,
    k: u32,
    scale: f32,
) {
    for n in 0..3 {
        let center = vertex(level, 4, 61 + n + k, 30 + k);
        let params = DrawParams::new(center, 0.06, 0.4 * scale).unwrap();
        draw_step(project.store(), index, stroke, params).unwrap();
    }
}

#[test]
fn a_new_stroke_starts_afresh() {
    let level = Level::new(7).unwrap();
    // One tile's state: the second stroke reuses the buffer the first left, over the same
    // vertices, the border ones included.
    let run = |reuse: bool| {
        let dir = TestDir::new();
        let (mut project, mut index) = project_with(&dir, level, 2, 1);
        let (key, scale) = strong_stroke(&mut project, &mut index, level);
        let mut stroke = Stroke::new(project.store());
        for k in 0..2 {
            project.begin_stroke().unwrap();
            if k == 1 && !reuse {
                stroke = Stroke::new(project.store());
            }
            border_dabs(&mut project, &mut index, &mut stroke, level, k, scale);
            assert!(
                stroke.holds(&key),
                "the tile owning the border has the state"
            );
            project.end_stroke().unwrap();
        }
        snapshot(project.store(), level)
    };
    assert!(
        run(true) == run(false),
        "a reused stroke state starts afresh"
    );
}

#[test]
fn a_reused_buffer_starts_with_no_sums() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    // One tile's state: tile (face 1, 0, 0) takes the buffer tile (face 4, 0, 0) leaves, and
    // its dabs touch the same tile coordinates as the sums left in it.
    let (mut project, mut index) = project_with(&dir, level, 2, 1);
    let (_, scale) = strong_stroke(&mut project, &mut index, level);
    let other = TileKey::displacement(FaceId::new(1), level, 0, 0).unwrap();
    let light = DrawParams::new(vertex(level, 1, 50, 32), 0.06, 0.4 * scale).unwrap();
    let before = tile_state(project.store(), other);

    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    for _ in 0..5 {
        draw_step(
            project.store(),
            &mut index,
            &mut stroke,
            light_dab(level, 4, scale),
        )
        .unwrap();
    }
    // Tile (4, 0, 0) was last edited at dab 5: protected through dab 8, released at dab 9.
    let mut unaccumulated = Vec::new();
    for _ in 0..5 {
        let stats = draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
        unaccumulated.push(stats.tiles_unaccumulated);
    }
    project.end_stroke().unwrap();
    assert_eq!(
        unaccumulated,
        [1, 1, 1, 0, 0],
        "written per dab while the buffer's tile is protected, then given its buffer"
    );
    assert!(stroke.holds(&other), "the tile has the buffer");
    let (inside, _) = assert_moved_by_dabs(project.store(), other, &before, light, 5.0, 1.0);
    assert!(inside > 0, "the brush reaches the tile");
}

#[test]
fn a_scale_that_grows_mid_stroke_keeps_vertices_within_a_step() {
    let dir = TestDir::new();
    let level = Level::new(7).unwrap();
    let (mut project, mut index) = project_with(&dir, level, 2, 512);
    let (key, scale) = strong_stroke(&mut project, &mut index, level);
    let light = light_dab(level, 4, scale);
    let before = tile_state(project.store(), key);

    project.begin_stroke().unwrap();
    let mut stroke = Stroke::new(project.store());
    for _ in 0..10 {
        draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
    }
    // A stronger dab in another corner of the tile, out of the light dabs' reach, grows the
    // tile's scale.
    let stronger = DrawParams::new(vertex(level, 4, 10, 55), 0.1, 0.2).unwrap();
    draw_step(project.store(), &mut index, &mut stroke, stronger).unwrap();
    for _ in 0..10 {
        draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
    }
    project.end_stroke().unwrap();
    let grown = project.store().correction(key).unwrap().scale();
    assert!(grown >= 2.0 * scale, "the scale grew: {scale} to {grown}");
    let (inside, _) = assert_moved_by_dabs(project.store(), key, &before, light, 20.0, 1.0);
    assert!(inside > 0, "the brush reaches the tile");
}

#[test]
fn a_stroke_used_over_another_project_starts_afresh() {
    let level = Level::new(7).unwrap();
    // The second project's store has the same site, so its first op has the same id as the
    // first's, and a reservation of another size.
    let second = |stroke: &mut Option<Stroke>| {
        let dir = TestDir::new();
        let (mut project, mut index) = project_with(&dir, level, 2, 3);
        let (_, scale) = strong_stroke(&mut project, &mut index, level);
        let mut stroke = stroke
            .take()
            .unwrap_or_else(|| Stroke::new(project.store()));
        project.begin_stroke().unwrap();
        border_dabs(&mut project, &mut index, &mut stroke, level, 0, scale);
        project.end_stroke().unwrap();
        assert_eq!(stroke.buffers(), 3, "the second store's reservation");
        snapshot(project.store(), level)
    };
    let dir = TestDir::new();
    let (mut project, mut index) = project_with(&dir, level, 2, 512);
    let (_, scale) = strong_stroke(&mut project, &mut index, level);
    let mut stroke = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    border_dabs(&mut project, &mut index, &mut stroke, level, 0, scale);
    // The first project's stroke is still in progress.
    let reused = second(&mut Some(stroke));
    assert!(
        reused == second(&mut None),
        "a stroke state carried to another project starts afresh"
    );
}

/// Asserts a save went through with crash protection running.
fn assert_saved(saved: Result<store::SaveOutcome, store::ProjectError>) {
    let outcome = saved.unwrap();
    assert!(outcome.protection.is_active(), "{outcome:?}");
}

/// A configuration for the reopening and saving tests: two workers, the default stroke state,
/// recovery files in `dir`.
fn reopen_config(dir: &TestDir) -> ProjectConfig {
    ProjectConfig {
        store: StoreConfig::new(64 << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(fixed(2)),
        recovery_dir: dir.0.join("recovery"),
    }
}

#[test]
fn a_project_reopened_after_discarding_a_stroke_starts_afresh() {
    let level = Level::new(7).unwrap();
    let run = |reuse: bool| {
        let dir = TestDir::new();
        let path = dir.0.join("project.bcp");
        let mut project = Project::new(cube_sphere(1).unwrap(), reopen_config(&dir)).unwrap();
        assert_saved(project.save_as(&path));
        let mut index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
        let (_, scale) = strong_stroke(&mut project, &mut index, level);
        let mut stroke = Stroke::new(project.store());
        project.begin_stroke().unwrap();
        border_dabs(&mut project, &mut index, &mut stroke, level, 0, scale);
        project.end_stroke().unwrap();
        // Closed unsaved, its recovery file deleted: the reopened project hands out the
        // discarded strokes' op ids again.
        project.close(false).unwrap();
        let mut reopened = Project::open(&path, reopen_config(&dir)).unwrap();
        let mut reindexed = SpatialIndex::build(reopened.store(), level, |_, _| {}).unwrap();
        let (_, rescale) = strong_stroke(&mut reopened, &mut reindexed, level);
        if !reuse {
            stroke = Stroke::new(reopened.store());
        }
        reopened.begin_stroke().unwrap();
        border_dabs(
            &mut reopened,
            &mut reindexed,
            &mut stroke,
            level,
            0,
            rescale,
        );
        reopened.end_stroke().unwrap();
        snapshot(reopened.store(), level)
    };
    assert!(
        run(true) == run(false),
        "the discarded stroke's sums don't come back"
    );
}

#[test]
fn a_stroke_saved_part_way_goes_on_from_the_save() {
    // A save moves the store to a new op, so the stroke starts its sums afresh from the saved
    // values, then keeps them from dab to dab: as the oracle does with a new stroke state at the
    // save, bit for bit.
    let level = Level::new(7).unwrap();
    let dir = TestDir::new();
    let path = dir.0.join("project.bcp");
    let mut project = Project::new(cube_sphere(1).unwrap(), reopen_config(&dir)).unwrap();
    assert_saved(project.save_as(&path));
    let mesh = project.store().mesh().clone();
    let mut index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
    let mut map = CorrectionMap::new();
    let (key, scale) = strong_stroke(&mut project, &mut index, level);
    let strong = DrawParams::new(vertex(level, 4, 32, 32), 0.1, 0.05).unwrap();
    oracle_step(&mesh, level, &mut map, &mut OracleStroke::new(), strong);
    let light = light_dab(level, 4, scale);
    let mut stroke = Stroke::new(project.store());
    let mut oracle = OracleStroke::new();
    project.begin_stroke().unwrap();
    for _ in 0..3 {
        draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
        oracle_step(&mesh, level, &mut map, &mut oracle, light);
    }
    assert_saved(project.save());
    oracle = OracleStroke::new();
    for k in 0..5 {
        draw_step(project.store(), &mut index, &mut stroke, light).unwrap();
        oracle_step(&mesh, level, &mut map, &mut oracle, light);
        assert!(stroke.holds(&key), "the tile keeps its state");
        assert_as_dense(
            project.store(),
            &map,
            level,
            &format!("dab {k} after the save"),
        );
    }
    project.end_stroke().unwrap();
}
