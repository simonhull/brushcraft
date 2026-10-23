//! The draw step against `geom`'s dense evaluation of the same edits, bit for bit.
//!
//! The oracle doesn't go through tiles at all: it evaluates every face of the mesh in full,
//! finds the deltas face by face, and writes them through a [`CorrectionMap`], which keeps twins
//! in step on its own. Its deltas come from the same kernel (so they match bit for bit), and are
//! checked against the smoothstep formula evaluated directly, in `f64`.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{BaseMesh, CorrectionMap, DenseEvaluation, FaceId, Level, cube_sphere};
use store::{ProjectConfig, StoreConfig};

use super::*;

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
    let config = ProjectConfig {
        store: StoreConfig {
            budget_bytes: 64 << 20,
            chunk_bytes: 1 << 20,
            site: 1,
        },
        recovery_dir: dir.0.join("recovery"),
    };
    let mut project = Project::new(cube_sphere(1).unwrap(), config).unwrap();
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
/// `None` outside the brush. Inside is decided as the kernel decides it, on the
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
        kernels::draw(
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
    let mesh = cube_sphere(1).unwrap();
    let dense = DenseEvaluation::new(&mesh, level, &CorrectionMap::new()).unwrap();
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
        kernel: ms(3),
        edit: ms(4),
        evaluate: ms(5),
        refit: ms(6),
    };
    assert_eq!(times.total(), ms(21), "the phases together");
    times += times;
    assert_eq!(times.total(), ms(42), "two steps together");
    assert_eq!(times.refit, ms(12), "phase by phase");
}
