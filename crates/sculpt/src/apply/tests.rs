//! `apply_dabs` against the same dabs applied one call each, bit for bit, at any worker count;
//! and its errors.

use geom::{CorrectionError, FaceId, Level, TileKey, Vec3};
use store::{CacheError, Project, SpatialIndex, Workers};
use stroke::Dab;

use super::{ApplyStats, SculptError, apply_dabs};
use crate::draw::{DrawParams, DrawParamsError, draw_step};
use crate::state::Stroke;
use crate::testing::{TestDir, cube, dab, fixed, snapshot, vertex};

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
    project.begin_stroke().unwrap();
    let mut stats = ApplyStats::default();
    let mut rest = dabs;
    for &size in frames.iter().cycle() {
        if rest.is_empty() {
            break;
        }
        let (frame, after) = rest.split_at(size.min(rest.len()));
        stats += apply_dabs(project.store(), index, &mut brush_state, frame).unwrap();
        rest = after;
    }
    project.end_stroke().unwrap();
    stats
}

/// A fresh cube with `workers`, the stroke's dabs planned on it, and those dabs applied in
/// frames of `frames`: the store's every tile afterwards, in bits, with its index's bounds.
fn run(workers: Workers, frames: &[usize]) -> (Snapshot, ApplyStats) {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), workers);
    let dabs = path(&mut project, 24);
    let stats = apply_in_frames(&mut project, &mut index, &dabs, frames);
    let tiles = snapshot(project.store(), level());
    let bounds = index
        .bvh()
        .iter()
        .map(|(key, bounds)| (*key, format!("{bounds:?}")))
        .collect();
    (Snapshot { tiles, bounds }, stats)
}

/// A store's tiles in bits, and its index's bounds.
#[derive(Debug, PartialEq)]
struct Snapshot {
    /// Every tile's corrections, positions and normals.
    tiles: std::collections::BTreeMap<TileKey, crate::testing::TileBits>,
    /// Every tile's bounds in the index, printed (bit-exact for finite floats).
    bounds: std::collections::BTreeMap<TileKey, String>,
}

#[test]
fn a_frame_of_dabs_equals_its_dabs_one_call_each_bit_for_bit() {
    let (alone, one_each) = run(fixed(2), &[1]);
    assert_eq!(one_each.dabs, 24, "every dab applied");
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
            stats.vertices, one_each.vertices,
            "frames {frames:?}: the same vertex moves"
        );
        let differ = framed
            .tiles
            .iter()
            .filter(|(key, tile)| alone.tiles.get(key) != Some(tile))
            .count();
        assert_eq!(differ, 0, "frames {frames:?}: tiles that differ");
        assert!(framed == alone, "frames {frames:?}: the index's bounds too");
    }
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
fn stepped(dabs: &[Dab]) -> std::collections::BTreeMap<TileKey, crate::testing::TileBits> {
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let mut state = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    for d in dabs {
        let [x, y, z] = d.position;
        let params = DrawParams::new(Vec3::new(x, y, z), d.radius, d.strength).unwrap();
        draw_step(project.store(), &mut index, &mut state, params).unwrap();
    }
    project.end_stroke().unwrap();
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
    project.end_stroke().unwrap();
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

#[test]
fn the_frames_tiles_widen_with_the_strengths_before() {
    let imprints = [
        brush::Imprint::new([1.0, 0.0, 0.0], [0.0; 3], 0.5, 0.25, 0.0, [0.0; 4]),
        brush::Imprint::new([1.0, 0.0, 0.0], [0.0; 3], 0.5, -0.5, 0.0, [0.0; 4]),
        brush::Imprint::new([1.0, 0.0, 0.0], [0.0; 3], 0.5, 0.1, 0.0, [0.0; 4]),
    ];
    let reaches: Vec<f32> = super::widened(&imprints).map(|(_, r)| r).collect();
    assert_eq!(reaches[0], 0.5, "the first dab's own radius, exactly");
    assert!(
        reaches[1] >= 0.75 && reaches[1] < 0.751,
        "plus the first's strength: {reaches:?}"
    );
    assert!(
        reaches[2] >= 1.25 && reaches[2] < 1.252,
        "plus both strengths, whatever their signs: {reaches:?}"
    );
}
