//! `apply_dabs` against the same dabs applied one call each, bit for bit, at any worker count;
//! and its errors.

use geom::{CorrectionError, FaceId, Level, TileKey, Vec3};
use store::{CacheError, Project, SpatialIndex, StoreConfig, Workers};
use stroke::Dab;

use super::{ApplyStats, SculptError, apply_dabs};
use crate::draw::{DrawParams, DrawParamsError, draw_step};
use crate::state::Stroke;
use crate::testing::{TestDir, TileBits, cube, dab, fixed, snapshot, vertex};

/// The level the tests edit: 128 quads a face, four tiles of 64.
fn level() -> Level {
    Level::new(7).unwrap()
}

/// A stroke of `count` overlapping dabs across face 0's tiles and over its edge.
///
/// They are placed on the cube's surface as `project` found it: radius 0.15 (about 16 quads),
/// small strengths of both signs, varied hardness, and every fifth dab forty times stronger, so
/// tiles' scales grow mid-frame.
fn path(project: &mut Project, count: u32) -> Vec<Dab> {
    let level = level();
    let store = project.store();
    (0..count)
        .map(|n| {
            let tx = u16::try_from(n % 2).unwrap();
            let key = TileKey::displacement(FaceId::new(0), level, tx, 0).unwrap();
            let (i, j) = (4 + (n * 7) % 56, (n * 5) % 64);
            let centre = vertex(store, key, i, j);
            let sign = if n % 3 == 1 { -1.0 } else { 1.0 };
            let strength = if n % 5 == 4 { 0.04 } else { 0.001 } * sign;
            let hardness = [0.0, 0.3, 0.8][usize::try_from(n % 3).unwrap()];
            dab(centre, 0.15, strength, hardness, n)
        })
        .collect()
}

/// Applies `dabs` to `project` in frames of the sizes `frames` cycles through, in one stroke.
fn apply_in_frames(
    project: &mut Project,
    index: &mut SpatialIndex,
    dabs: &[Dab],
    frames: &[usize],
) -> ApplyStats {
    let mut brush_state = Stroke::new(project.store());
    apply_with(project, index, &mut brush_state, dabs, frames)
}

/// [`apply_in_frames`] with `brush_state`, the stroke's state.
fn apply_with(
    project: &mut Project,
    index: &mut SpatialIndex,
    brush_state: &mut Stroke,
    dabs: &[Dab],
    frames: &[usize],
) -> ApplyStats {
    project.begin_stroke().unwrap();
    let mut stats = ApplyStats::default();
    let mut rest = dabs;
    for &size in frames.iter().cycle() {
        if rest.is_empty() {
            break;
        }
        let (frame, after) = rest.split_at(size.min(rest.len()));
        stats += apply_dabs(project.store(), index, brush_state, frame).unwrap();
        rest = after;
    }
    project.end_unrecorded_stroke().unwrap();
    stats
}

/// A fresh cube with `workers`, the stroke's dabs planned on it, and those dabs applied in
/// frames of `frames`: the store's every tile afterwards, in bits, with its index's bounds.
fn run(workers: Workers, frames: &[usize]) -> (Snapshot, ApplyStats) {
    run_over(
        StoreConfig::new(64 << 20, 1).with_workers(workers),
        24,
        frames,
    )
}

/// [`run`] for a stroke of `count` dabs over a store of `config` (in 1 MiB chunks).
fn run_over(config: StoreConfig, count: u32, frames: &[usize]) -> (Snapshot, ApplyStats) {
    let dir = TestDir::new();
    let config = store::ProjectConfig {
        store: config.with_chunk_bytes(1 << 20),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::new(geom::cube_sphere(1).unwrap(), config).unwrap();
    let mut index = SpatialIndex::build(project.store(), level(), |_, _| {}).unwrap();
    let dabs = path(&mut project, count);
    let mut brush_state = Stroke::new(project.store());
    let stats = apply_with(&mut project, &mut index, &mut brush_state, &dabs, frames);
    let tiles = snapshot(project.store(), level());
    let bounds = index
        .bvh()
        .iter()
        .map(|(key, bounds)| (*key, format!("{bounds:?}")))
        .collect();
    let stroke = brush_state.bits();
    let snapshot = Snapshot {
        tiles,
        bounds,
        stroke,
    };
    (snapshot, stats)
}

/// A store's tiles in bits, its index's bounds and the stroke's state.
#[derive(Debug, PartialEq)]
struct Snapshot {
    /// Every tile's corrections, positions and normals.
    tiles: std::collections::BTreeMap<TileKey, TileBits>,
    /// Every tile's bounds in the index, printed (bit-exact for finite floats).
    bounds: std::collections::BTreeMap<TileKey, String>,
    /// The stroke's state after the dabs, in bits ([`Stroke::bits`]).
    stroke: StrokeBits,
}

/// What [`Stroke::bits`] gives.
type StrokeBits = (crate::state::Book, Vec<crate::state::TileBits>);

#[test]
fn a_frame_of_dabs_equals_its_dabs_one_call_each_bit_for_bit() {
    let (alone, one_each) = run(fixed(2), &[1]);
    assert_eq!(one_each.dabs, 24, "every dab applied");
    assert_eq!(one_each.tiled_dabs, 0, "a dab alone is applied as it was");
    assert!(
        one_each.vertices > 10_000,
        "the dabs move vertices: {one_each:?}"
    );
    assert!(
        one_each.tiles_edited >= 2,
        "the stroke crosses tiles: {one_each:?}"
    );
    for frames in [&[24][..], &[3, 5, 2], &[7]] {
        let (framed, stats) = run(fixed(2), frames);
        assert_eq!(stats.dabs, 24, "frames {frames:?}: every dab applied");
        assert_eq!(
            stats.tiled_dabs,
            24 - alone_in_frames(24, frames),
            "frames {frames:?}: every dab of a frame of two or more applied tile by tile"
        );
        assert_eq!(
            stats.vertices, one_each.vertices,
            "frames {frames:?}: the same vertex moves"
        );
        let differ = framed
            .tiles
            .iter()
            .filter(|(key, tile)| alone.tiles.get(key) != Some(tile))
            .count();
        assert_eq!(differ, 0, "frames {frames:?}: tiles that differ");
        assert!(
            framed.bounds == alone.bounds,
            "frames {frames:?}: the index's bounds too"
        );
        assert!(framed == alone, "frames {frames:?}: and the stroke's state");
    }
}

/// How many of `count` dabs applied in frames of the sizes `frames` cycles through are alone
/// in their frame: a frame of one dab is applied as it was before frames were tiled.
fn alone_in_frames(count: usize, frames: &[usize]) -> usize {
    let mut rest = count;
    let mut alone = 0;
    for &size in frames.iter().cycle() {
        if rest == 0 {
            break;
        }
        let taken = size.min(rest);
        alone += usize::from(taken == 1);
        rest -= taken;
    }
    alone
}

#[test]
fn a_frame_equals_its_dabs_one_call_each_when_the_stroke_state_runs_out() {
    // With a few buffers, the stroke releases tiles and writes others per dab mid-frame: a
    // run's first replay assumes neither, and the second takes the grants the first one's
    // dabs give.
    for buffers in [0, 1, 2, 3, 5] {
        let config = StoreConfig::new(64 << 20, 1)
            .with_workers(fixed(2))
            .with_stroke_tiles(buffers);
        let (alone, one_each) = run_over(config, 32, &[1]);
        assert!(
            buffers > 2 || one_each.tiles_unaccumulated > 0,
            "{buffers} buffers: some tiles written per dab: {one_each:?}"
        );
        for frames in [&[32][..], &[5, 3], &[9]] {
            let (framed, stats) = run_over(config, 32, frames);
            assert_eq!(
                stats.tiles_unaccumulated, one_each.tiles_unaccumulated,
                "{buffers} buffers, frames {frames:?}: the same tiles written per dab"
            );
            assert!(
                stats.tiled_dabs > 0,
                "{buffers} buffers, frames {frames:?}: dabs applied tile by tile"
            );
            assert!(
                framed == alone,
                "{buffers} buffers, frames {frames:?}: the same bits"
            );
        }
    }
}

#[test]
fn a_frame_equals_its_dabs_one_call_each_with_no_predictions_kept() {
    // No tile can have its prediction: every run is one dab.
    let config = StoreConfig::new(64 << 20, 1)
        .with_workers(fixed(2))
        .with_kept_predictions(0);
    let (alone, _) = run_over(config, 24, &[1]);
    let (framed, stats) = run_over(config, 24, &[24]);
    assert_eq!(stats.tiled_dabs, 24, "every dab tile by tile, one a run");
    assert!(framed == alone, "the same bits");
}

#[test]
fn a_frame_is_the_same_at_any_worker_count() {
    let (serial, _) = run(fixed(1), &[6]);
    for workers in [2, 4] {
        let (parallel, _) = run(fixed(workers), &[6]);
        assert!(parallel == serial, "{workers} workers against one");
    }
}

#[test]
fn draw_steps_equal_a_frame_of_their_dabs() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let mut dabs = path(&mut project, 8);
    for d in &mut dabs {
        d.hardness = 0.0;
    }
    apply_in_frames(&mut project, &mut index, &dabs, &[8]);
    let framed = snapshot(project.store(), level());
    assert!(
        stepped(&dabs) == framed,
        "a draw step a dab: the same tiles, bit for bit"
    );
}

/// The tiles of a fresh cube after a draw step at each of `dabs` (their hardness ignored), in
/// one stroke.
fn stepped(dabs: &[Dab]) -> std::collections::BTreeMap<TileKey, TileBits> {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    for d in dabs {
        let [x, y, z] = d.position;
        let params = DrawParams::new(Vec3::new(x, y, z), d.radius, d.strength).unwrap();
        draw_step(project.store(), &mut index, &mut state, params).unwrap();
    }
    project.end_unrecorded_stroke().unwrap();
    snapshot(project.store(), level())
}

/// The centre of the face farthest from `dabs`: the vertex in the middle of its grid, the face
/// the one whose middle is farthest from the nearest dab.
fn far_from(project: &mut Project, dabs: &[Dab]) -> (FaceId, Vec3) {
    let level = level();
    let store = project.store();
    let faces: Vec<FaceId> = store.mesh().faces().collect();
    let middles = faces.iter().map(|&face| {
        let key = TileKey::displacement(face, level, 1, 1).unwrap();
        (face, vertex(store, key, 0, 0))
    });
    let nearest = |p: Vec3| {
        dabs.iter()
            .map(|d| {
                let [x, y, z] = d.position;
                (Vec3::new(x, y, z) - p).length()
            })
            .fold(f32::INFINITY, f32::min)
    };
    let middles: Vec<(FaceId, Vec3)> = middles.collect();
    middles
        .into_iter()
        .max_by(|a, b| nearest(a.1).total_cmp(&nearest(b.1)))
        .unwrap()
}

#[test]
fn a_store_error_ends_the_call_and_keeps_the_dabs_before_it() {
    let level = level();
    // The reference: the first two dabs alone.
    let reference = {
        let dir = TestDir::new();
        let (mut project, mut index) = cube(&dir, level, fixed(2));
        let dabs = path(&mut project, 2);
        apply_in_frames(&mut project, &mut index, &dabs, &[2]);
        snapshot(project.store(), level)
    };

    // The same two, then a dab in the middle of the face farthest from them, too strong for
    // any tile to store: its tile takes the values near its rim, then refuses one.
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level, fixed(2));
    let mut frame = path(&mut project, 2);
    let (far, centre) = far_from(&mut project, &frame);
    frame.push(dab(centre, 0.15, 3e38, 0.0, 2));
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    let failed = apply_dabs(project.store(), &mut index, &mut state, &frame);
    assert!(
        matches!(
            failed,
            Err(SculptError::Cache(CacheError::Correction(
                CorrectionError::TooLarge
            )))
        ),
        "the third dab can't be stored: {failed:?}"
    );
    project.end_unrecorded_stroke().unwrap();
    let after = snapshot(project.store(), level);
    let near: Vec<&TileKey> = reference.keys().filter(|key| key.face() != far).collect();
    let kept = near
        .iter()
        .filter(|key| after.get(key) == reference.get(key))
        .count();
    assert_eq!(
        kept,
        near.len(),
        "every tile away from the failing dab is as the first two dabs left it"
    );
    assert!(
        reference
            .iter()
            .any(|(key, tile)| after.get(key) != Some(tile)),
        "the failing dab wrote what it could before it failed"
    );
    for key in near {
        let fresh = SpatialIndex::bounds_of(project.store(), *key).unwrap();
        assert_eq!(
            index.bvh().bounds(key),
            Some(&fresh),
            "{key:?}'s bounds in the index are current"
        );
    }
}

#[test]
fn an_invalid_dab_is_refused_before_anything_is_written() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let before = snapshot(project.store(), level());
    let mut frame = path(&mut project, 3);
    let cases = [
        (
            Dab {
                radius: 0.0,
                ..frame[2]
            },
            DrawParamsError::Radius,
        ),
        (
            Dab {
                strength: f32::NAN,
                ..frame[2]
            },
            DrawParamsError::Strength,
        ),
        (
            Dab {
                position: [f32::INFINITY, 0.0, 0.0],
                ..frame[2]
            },
            DrawParamsError::Center,
        ),
    ];
    let mut state = Stroke::new(project.store());
    for (bad, why) in cases {
        frame[2] = bad;
        let refused = apply_dabs(project.store(), &mut index, &mut state, &frame);
        assert!(
            matches!(refused, Err(SculptError::Dab { dab: 2, error }) if error == why),
            "{why:?}: {refused:?}"
        );
    }
    assert!(
        snapshot(project.store(), level()) == before,
        "nothing was written"
    );
}

#[test]
fn an_empty_frame_does_nothing() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let mut brush_state = Stroke::new(project.store());
    let applied = apply_dabs(project.store(), &mut index, &mut brush_state, &[]).unwrap();
    assert_eq!(applied, ApplyStats::default(), "no dabs, nothing done");
}

/// A dab lifting the seam vertex `(64, 32)` of face 0's tile `(0, 0)` by 0.2.
fn lift(project: &mut Project) -> Dab {
    let left = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    dab(vertex(project.store(), left, 64, 32), 0.15, 0.2, 0.0, 0)
}

#[test]
fn a_dab_reaches_a_tile_through_a_seam_vertex_the_dab_before_pushed_out() {
    let level = level();
    let left = TileKey::displacement(FaceId::new(0), level, 0, 0).unwrap();
    // The reference, one call per dab: the first lifts the seam vertex, the second is a small one
    // just past where it went, far above the surface as the frame finds it.
    let (reference, second) = {
        let dir = TestDir::new();
        let (mut project, mut index) = cube(&dir, level, fixed(2));
        let before = vertex(project.store(), left, 64, 32);
        let first = lift(&mut project);
        let mut state = Stroke::new(project.store());
        project.begin_stroke().unwrap();
        apply_dabs(project.store(), &mut index, &mut state, &[first]).unwrap();
        let after = vertex(project.store(), left, 64, 32);
        let second = dab(after + (after - before) * 0.05, 0.03, 0.001, 0.0, 1);
        let alone = apply_dabs(project.store(), &mut index, &mut state, &[second]).unwrap();
        project.end_unrecorded_stroke().unwrap();
        assert!(
            alone.vertices > 0,
            "the second dab reaches the lifted vertex"
        );
        (snapshot(project.store(), level), second)
    };

    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level, fixed(2));
    let [x, y, z] = second.position;
    assert_eq!(
        index
            .query_sphere(Vec3::new(x, y, z), second.radius)
            .count(),
        0,
        "no tile's box as the frame finds it touches the second dab"
    );
    let frame = [lift(&mut project), second];
    let framed = apply_in_frames(&mut project, &mut index, &frame, &[2]);
    assert_eq!(framed.frames, 1, "one frame");
    assert!(
        snapshot(project.store(), level) == reference,
        "the frame reaches the vertex too, and gives the same bits"
    );
}

#[test]
fn a_failed_call_closes_its_epoch() {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let cursor = project.store().subscribe_stale();
    let mut frame = path(&mut project, 2);
    let (_, centre) = far_from(&mut project, &frame);
    frame.push(dab(centre, 0.15, 3e38, 0.0, 2));
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    let failed = apply_dabs(project.store(), &mut index, &mut state, &frame);
    assert!(failed.is_err(), "the third dab can't be stored");
    assert!(
        project.store().has_stale(&cursor),
        "the call's epoch is closed, for a reader to drain"
    );
    let drained = project.store().drain_stale(&cursor);
    assert!(
        !project.store().has_stale(&cursor),
        "one epoch, drained: {drained:?}"
    );
    project.end_unrecorded_stroke().unwrap();
}

/// A project over the cube at the test level under `budget` bytes, or `None` if that is too
/// small for it or its index.
///
/// In 64 KiB chunks, with no kept predictions and a little stroke state, so the budget is mostly
/// the pools.
fn tight(dir: &TestDir, budget: usize) -> Option<(Project, SpatialIndex)> {
    let config = store::ProjectConfig {
        store: StoreConfig::new(budget, 1)
            .with_chunk_bytes(64 << 10)
            .with_kept_predictions(0)
            .with_stroke_tiles(8)
            .with_workers(fixed(2)),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::new(geom::cube_sphere(1).unwrap(), config).ok()?;
    let index = SpatialIndex::build(project.store(), level(), |_, _| {}).ok()?;
    Some((project, index))
}

/// Two dabs on `project`, far apart: in the middle of face 0's first tile, and on the face
/// farthest from it.
fn far_apart(project: &mut Project) -> [Dab; 2] {
    let near = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let first = dab(vertex(project.store(), near, 32, 32), 0.05, 0.01, 0.0, 0);
    let (_, centre) = far_from(project, &[first]);
    [first, dab(centre, 0.05, 0.01, 0.0, 1)]
}

/// The tiles of a fresh [`tight`] project under `budget` after [`far_apart`]'s dabs, one call
/// each.
fn far_apart_one_call_each(budget: usize) -> std::collections::BTreeMap<TileKey, TileBits> {
    let dir = TestDir::new();
    let (mut project, mut index) = tight(&dir, budget).unwrap();
    let dabs = far_apart(&mut project);
    apply_in_frames(&mut project, &mut index, &dabs, &[1]);
    snapshot(project.store(), level())
}

#[test]
fn a_frame_too_big_to_pin_is_applied_in_two_with_the_same_bits() {
    let mesh = geom::cube_sphere(1).unwrap();
    let minimum = StoreConfig::new(0, 1)
        .with_chunk_bytes(64 << 10)
        .with_kept_predictions(0)
        .with_stroke_tiles(8)
        .minimum_budget(&mesh);
    let budgets = (0..256).map(|k| minimum + k * (64 << 10));
    for budget in budgets {
        let dir = TestDir::new();
        let Some((mut project, mut index)) = tight(&dir, budget) else {
            continue;
        };
        let dabs = far_apart(&mut project);
        let mut brush_state = Stroke::new(project.store());
        project.begin_stroke().unwrap();
        let applied = apply_dabs(project.store(), &mut index, &mut brush_state, &dabs);
        project.end_unrecorded_stroke().unwrap();
        let Ok(done) = applied else {
            continue;
        };
        assert_eq!(
            done.frames, 2,
            "at the least budget that applies them, {budget} bytes, the two dabs don't fit together"
        );
        assert_eq!(done.dabs, 2, "both applied");
        assert!(
            snapshot(project.store(), level()) == far_apart_one_call_each(budget),
            "the same bits as a call per dab"
        );
        return;
    }
    panic!("no budget from the minimum up applies the two dabs");
}

#[test]
fn a_frame_of_dabs_closes_one_step_epoch_with_their_displacements_summed() {
    let dir = TestDir::new();
    let config = store::ProjectConfig {
        store: StoreConfig::new(64 << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(fixed(2))
            .with_display(level()),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::new(geom::cube_sphere(1).unwrap(), config).unwrap();
    let store = project.store();
    store.queue_display_rebuild(level());
    loop {
        store
            .run_jobs(std::time::Instant::now() + std::time::Duration::from_millis(50))
            .unwrap();
        if !store.has_jobs() {
            break;
        }
    }
    let mut index = SpatialIndex::build(project.store(), level(), |_, _| {}).unwrap();
    let cursor = project.store().subscribe_stale();
    // Four dabs each centred on a vertex, so each moves that vertex its whole strength.
    let key = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let strengths = [0.004_f32, 0.002, 0.003, 0.001];
    let places = [(20, 20), (30, 22), (24, 34), (40, 40)];
    let frame: Vec<Dab> = strengths
        .iter()
        .zip(places)
        .zip(0..)
        .map(|((&strength, (i, j)), seq)| {
            dab(vertex(project.store(), key, i, j), 0.03, strength, 0.0, seq)
        })
        .collect();
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), &mut index, &mut state, &frame).unwrap();
    project.end_unrecorded_stroke().unwrap();
    let store::Drained::Epochs(epochs) = project.store().drain_stale(&cursor) else {
        panic!("epochs");
    };
    let kinds: Vec<store::EpochKind> = epochs.iter().map(|epoch| epoch.kind).collect();
    let sum: f32 = strengths.iter().sum();
    assert!(
        matches!(kinds.as_slice(), [store::EpochKind::Step { displacement }] if *displacement >= sum),
        "one step's epoch, D at least the dabs' strengths summed ({sum}): {kinds:?}"
    );
}
