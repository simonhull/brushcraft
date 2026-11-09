//! A tile's future against the tile once its strokes have landed, bit for bit; and that taking
//! one writes nothing.

use std::collections::{BTreeMap, BTreeSet};

use geom::{CorrectionTile, FaceId, Level, TileKey, Vec3};
use store::{OpId, SpatialIndex, TileStore};
use stroke::Dab;

use super::{FutureError, Remaining, future_of};
use crate::apply::SculptError;
use crate::apply_dabs;
use crate::state::Stroke;
use crate::testing::{TestDir, cube, dab, fixed, tiles, vertex};

/// The level the tests edit: 128 quads a face, four tiles of 64.
fn level() -> Level {
    Level::new(7).unwrap()
}

/// A small deterministic generator (xorshift64*), so each seed is one stroke on every run.
struct Rng(u64);

impl Rng {
    /// The generator for `seed`.
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// The next 64 random bits.
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A number in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next() % u64::from(n)).unwrap()
    }

    /// A number in `0..1`.
    fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.next() >> 48).unwrap()) / 65_536.0
    }
}

/// A random stroke of `count` dabs over faces 0 to 2 of the cube, numbered from `first_seq`.
///
/// Placed on the surface as `store` has it: radii 0.08 to 0.2 (9 to 22 quads, across tile and
/// face edges), small strengths of both signs, every few dabs a strong one (tiles' scales grow),
/// hardness anywhere.
fn random_stroke(store: &mut TileStore, rng: &mut Rng, count: u32, first_seq: u32) -> Vec<Dab> {
    let level = level();
    (0..count)
        .map(|n| {
            let face = FaceId::new(rng.below(3));
            let (tx, ty) = (rng.below(2), rng.below(2));
            let key = TileKey::displacement(
                face,
                level,
                u16::try_from(tx).unwrap(),
                u16::try_from(ty).unwrap(),
            )
            .unwrap();
            let centre = vertex(store, key, rng.below(65), rng.below(65));
            let radius = 0.08 + 0.12 * rng.unit();
            let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
            let strength = if rng.below(5) == 0 { 0.04 } else { 0.002 } * rng.unit() * sign;
            dab(centre, radius, strength, rng.unit(), first_seq + n)
        })
        .collect()
}

/// Lands `dabs` in frames of random sizes, 1 to 6 dabs.
fn land(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    state: &mut Stroke,
    dabs: &[Dab],
    rng: &mut Rng,
) {
    let mut rest = dabs;
    while !rest.is_empty() {
        let size = usize::try_from(1 + rng.below(6)).unwrap().min(rest.len());
        let (frame, after) = rest.split_at(size);
        apply_dabs(store, index, state, frame).unwrap();
        rest = after;
    }
}

/// For each tile, the dabs of `dabs` whose spheres touch it by its box in `index` (in order); and
/// every tile one touches: what a stroke's index of its remaining dabs holds.
fn bearing(index: &SpatialIndex, dabs: &[Dab]) -> (BTreeMap<TileKey, Vec<Dab>>, BTreeSet<TileKey>) {
    let mut per_tile: BTreeMap<TileKey, Vec<Dab>> = BTreeMap::new();
    for each in dabs {
        let [x, y, z] = each.position;
        let touched: BTreeSet<TileKey> = index
            .query_sphere(Vec3::new(x, y, z), each.radius)
            .collect();
        for key in touched {
            per_tile.entry(key).or_default().push(*each);
        }
    }
    let reach: BTreeSet<TileKey> = per_tile.keys().copied().collect();
    (per_tile, reach)
}

/// A correction tile in bits: its planes' scales, its interior and its border.
type Bits = ([u32; 3], Vec<[i16; 3]>, Vec<[u32; 3]>);

/// `tile` in bits.
fn bits(tile: &CorrectionTile) -> Bits {
    let border = tile
        .border()
        .iter()
        .map(|v| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()])
        .collect();
    (
        tile.scales().map(f32::to_bits),
        tile.interior().to_vec(),
        border,
    )
}

/// Tile `key`'s corrections as stored.
fn stored(store: &mut TileStore, key: TileKey) -> CorrectionTile {
    store.correction(key).unwrap().to_tile()
}

/// How many of `futures` differ from their tiles as stored now: the tiles the strokes still
/// change.
fn changed(store: &mut TileStore, futures: &BTreeMap<TileKey, CorrectionTile>) -> usize {
    futures
        .iter()
        .filter(|&(&key, future)| bits(future) != bits(&stored(store, key)))
        .count()
}

/// Tile `key`'s positions with `tile` as its corrections: its prediction with them applied, in
/// bits.
fn positions_of(store: &mut TileStore, key: TileKey, tile: &CorrectionTile) -> Vec<[u32; 3]> {
    let _pins = store.pin(&[key]).unwrap();
    assert!(store.keep_predictions_of(&[key]).unwrap());
    let view = tile.view();
    store
        .kept_prediction(&key)
        .unwrap()
        .apply(|i, j| view.get(i, j).unwrap())
        .into_vec()
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
        .collect()
}

/// Tile `key`'s positions as the store evaluates them, in bits.
fn evaluated(store: &mut TileStore, key: TileKey) -> Vec<[u32; 3]> {
    store
        .evaluate(key)
        .unwrap()
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
        .collect()
}

#[test]
fn a_future_is_the_tile_once_the_stroke_lands_bit_for_bit() {
    for seed in 1..=6 {
        let mut rng = Rng::new(seed);
        let dir = TestDir::new();
        let (mut project, mut index) = cube(&dir, level(), fixed(2));
        let dabs = random_stroke(project.store(), &mut rng, 24, 0);
        // The first seed lands nothing first: a stroke whose state hasn't begun.
        let landed = if seed == 1 {
            0
        } else {
            usize::try_from(rng.below(25)).unwrap()
        };
        let (first, rest) = dabs.split_at(landed);
        let mut state = Stroke::new(project.store());
        let op = project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, first, &mut rng);

        let (per_tile, reach) = bearing(&index, rest);
        let keys = tiles(project.store().mesh(), level());
        let futures: BTreeMap<TileKey, CorrectionTile> = keys
            .iter()
            .map(|&key| {
                let remaining = Remaining {
                    op,
                    dabs: per_tile.get(&key).map_or(&[][..], Vec::as_slice),
                    reach: &reach,
                };
                let future = future_of(project.store(), &state, &[remaining], key).unwrap();
                (key, future)
            })
            .collect();

        let moved = changed(project.store(), &futures);
        assert_eq!(
            moved > 0,
            !rest.is_empty(),
            "seed {seed}: {moved} tiles change"
        );
        land(project.store(), &mut index, &mut state, rest, &mut rng);
        for (&key, future) in &futures {
            let after = stored(project.store(), key);
            assert_eq!(bits(future), bits(&after), "seed {seed}, tile {key:?}");
            let made = positions_of(project.store(), key, future);
            assert_eq!(
                made,
                evaluated(project.store(), key),
                "seed {seed}, tile {key:?}"
            );
        }
        project.end_unrecorded_stroke().unwrap();
    }
}

#[test]
fn a_tile_no_remaining_dab_touches_is_its_stored_self() {
    let mut rng = Rng::new(7);
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let dabs = random_stroke(project.store(), &mut rng, 16, 0);
    let (first, rest) = dabs.split_at(8);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);

    let (per_tile, reach) = bearing(&index, rest);
    let keys = tiles(project.store().mesh(), level());
    let untouched: Vec<TileKey> = keys
        .iter()
        .filter(|key| !per_tile.contains_key(key))
        .copied()
        .collect();
    assert!(!untouched.is_empty(), "the stroke keeps to faces 0 to 2");
    for &key in &untouched {
        let as_stored = bits(&stored(project.store(), key));
        let none = Remaining {
            op,
            dabs: &[],
            reach: &reach,
        };
        let no_dabs = future_of(project.store(), &state, &[none], key).unwrap();
        assert_eq!(bits(&no_dabs), as_stored, "tile {key:?}, no dabs");
        // Dabs that reach none of its vertices change nothing either.
        let all = Remaining { dabs: rest, ..none };
        let far = future_of(project.store(), &state, &[all], key).unwrap();
        assert_eq!(bits(&far), as_stored, "tile {key:?}, dabs elsewhere");
        let no_chain = future_of(project.store(), &state, &[], key).unwrap();
        assert_eq!(bits(&no_chain), as_stored, "tile {key:?}, no strokes");
    }
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_future_of_a_chain_is_the_tile_once_both_land() {
    for seed in 11..=13 {
        let mut rng = Rng::new(seed);
        let dir = TestDir::new();
        let (mut project, mut index) = cube(&dir, level(), fixed(2));
        let a = random_stroke(project.store(), &mut rng, 20, 0);
        // B's dabs are placed on the surface as first found, not on A's futures: today's
        // surface can't read those (the store's work of the next task), so the chain's arithmetic
        // is tested with dabs whose placement doesn't depend on the surface.
        let b = random_stroke(project.store(), &mut rng, 12, 20);
        let landed = usize::try_from(rng.below(21)).unwrap();
        let (a_first, a_rest) = a.split_at(landed);
        let mut state = Stroke::new(project.store());
        let op_a = project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, a_first, &mut rng);

        let (a_per_tile, a_reach) = bearing(&index, a_rest);
        let keys = tiles(project.store().mesh(), level());
        // Every dab of B for every tile, and every tile its reach: B's dabs are measured on A's
        // futures, which the index's boxes don't hold yet.
        let b_reach: BTreeSet<TileKey> = keys.iter().copied().collect();
        let op_b: OpId = op_a.next().unwrap();
        let futures: BTreeMap<TileKey, CorrectionTile> = keys
            .iter()
            .map(|&key| {
                let chain = [
                    Remaining {
                        op: op_a,
                        dabs: a_per_tile.get(&key).map_or(&[][..], Vec::as_slice),
                        reach: &a_reach,
                    },
                    Remaining {
                        op: op_b,
                        dabs: &b,
                        reach: &b_reach,
                    },
                ];
                let future = future_of(project.store(), &state, &chain, key).unwrap();
                (key, future)
            })
            .collect();

        assert!(changed(project.store(), &futures) > 0, "seed {seed}");
        land(project.store(), &mut index, &mut state, a_rest, &mut rng);
        project.end_unrecorded_stroke().unwrap();
        project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, &b, &mut rng);
        project.end_unrecorded_stroke().unwrap();
        for (&key, future) in &futures {
            let after = stored(project.store(), key);
            assert_eq!(bits(future), bits(&after), "seed {seed}, tile {key:?}");
        }
    }
}

#[test]
fn taking_a_future_writes_nothing() {
    let mut rng = Rng::new(21);
    let dir = TestDir::new();
    let (mut project, mut index) = cube(&dir, level(), fixed(2));
    let dabs = random_stroke(project.store(), &mut rng, 20, 0);
    let (first, rest) = dabs.split_at(10);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);
    project.store().close_epoch().unwrap();
    let cursor = project.store().subscribe_stale();

    let keys = tiles(project.store().mesh(), level());
    let before: Vec<Bits> = keys
        .iter()
        .map(|&key| bits(&stored(project.store(), key)))
        .collect();
    let stroke_before = state.bits();
    let unsaved = project.has_unsaved_edits();
    let pen_down: BTreeSet<TileKey> = project.store().pen_down_keys().copied().collect();
    assert!(!pen_down.is_empty(), "the stroke has changed tiles");

    let (per_tile, reach) = bearing(&index, rest);
    for &key in &keys {
        let remaining = Remaining {
            op,
            dabs: per_tile.get(&key).map_or(&[][..], Vec::as_slice),
            reach: &reach,
        };
        let chain = [
            remaining,
            Remaining {
                op: op.next().unwrap(),
                dabs: rest,
                reach: &reach,
            },
        ];
        future_of(project.store(), &state, &chain, key).unwrap();
    }

    let after: Vec<Bits> = keys
        .iter()
        .map(|&key| bits(&stored(project.store(), key)))
        .collect();
    assert_eq!(before, after, "no correction is written");
    assert_eq!(
        state.bits(),
        stroke_before,
        "the stroke's state is as it was"
    );
    assert_eq!(project.store().current_op(), Some(op), "no op is begun");
    assert_eq!(project.has_unsaved_edits(), unsaved);
    let pen_down_after: BTreeSet<TileKey> = project.store().pen_down_keys().copied().collect();
    assert_eq!(pen_down_after, pen_down, "no pen-down shape is kept");
    project.store().close_epoch().unwrap();
    match project.store().drain_stale(&cursor) {
        store::Drained::Epochs(epochs) => {
            let written: Vec<&store::Epoch> = epochs
                .iter()
                .filter(|epoch| !epoch.keys.is_empty() || !epoch.written.is_empty())
                .collect();
            assert!(
                written.is_empty(),
                "no tile is on the stale feed: {written:?}"
            );
        }
        store::Drained::Resync(_) => panic!("the feed holds one epoch"),
    }
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_stroke_past_its_state_has_no_future() {
    let dir = TestDir::new();
    let config = store::ProjectConfig {
        store: store::StoreConfig::new(128 << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(fixed(2))
            .with_stroke_tiles(2),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = store::Project::new(geom::cube_sphere(1).unwrap(), config).unwrap();
    let index = SpatialIndex::build(project.store(), level(), |_, _| {}).unwrap();
    let mut rng = Rng::new(31);
    let dabs = random_stroke(project.store(), &mut rng, 8, 0);
    let state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    let (per_tile, reach) = bearing(&index, &dabs);
    assert!(
        reach.len() > 2,
        "the stroke reaches more tiles than its state holds"
    );
    let (&key, bearing) = per_tile.iter().next().unwrap();
    let remaining = Remaining {
        op,
        dabs: bearing,
        reach: &reach,
    };
    let refused = future_of(project.store(), &state, &[remaining], key);
    assert!(
        matches!(
            refused,
            Err(SculptError::Future(FutureError::StrokeState { stroke: 0 }))
        ),
        "{refused:?}"
    );
    project.end_unrecorded_stroke().unwrap();
}
