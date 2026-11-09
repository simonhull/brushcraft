//! A tile's future against the tile once its strokes have landed, bit for bit, whatever room the
//! stroke state has; and that taking one writes nothing.

use std::collections::{BTreeMap, BTreeSet};

use geom::{CorrectionTile, FaceId, Level, TileKey, Vec3};
use store::{OpId, Project, ProjectConfig, SpatialIndex, StoreConfig, TileStore};
use stroke::Dab;

use super::{FutureError, FuturePlan, Remaining};
use crate::apply::SculptError;
use crate::apply_dabs;
use crate::state::Stroke;
use crate::testing::{TestDir, dab, fixed, tiles, vertex};

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

/// A project over the cube with a 128 MiB budget of `config`'s other settings (two workers, 1
/// MiB chunks), and its index at [`level`].
fn project(
    dir: &TestDir,
    config: impl FnOnce(StoreConfig) -> StoreConfig,
) -> (Project, SpatialIndex) {
    let base = StoreConfig::new(128 << 20, 1)
        .with_chunk_bytes(1 << 20)
        .with_workers(fixed(2));
    let config = ProjectConfig {
        store: config(base),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::new(geom::cube_sphere(1).unwrap(), config).unwrap();
    let index = SpatialIndex::build(project.store(), level(), |_, _| {}).unwrap();
    (project, index)
}

/// Tile `(tx, ty)` of `face` at [`level`].
fn tile(face: u32, tx: u16, ty: u16) -> TileKey {
    TileKey::displacement(FaceId::new(face), level(), tx, ty).unwrap()
}

/// A random stroke of `count` dabs over faces 0 to 2 of the cube, numbered from `first_seq`.
///
/// Placed on the surface as `store` has it. Most dabs are of radii 0.08 to 0.2 (9 to 22 quads,
/// across tile and face edges); one in four sits on a tile's edge or corner with a radius of
/// 0.004 to 0.02 (a quad or two, or none). Small strengths of both signs, every few dabs a strong
/// one (tiles' scales grow), hardness anywhere.
fn random_stroke(store: &mut TileStore, rng: &mut Rng, count: u32, first_seq: u32) -> Vec<Dab> {
    (0..count)
        .map(|n| {
            let key = tile(
                rng.below(3),
                u16::try_from(rng.below(2)).unwrap(),
                u16::try_from(rng.below(2)).unwrap(),
            );
            let small = rng.below(4) == 0;
            let (i, j, radius) = if small {
                let corner = rng.below(2) == 0;
                let i = 64 * rng.below(2);
                let j = if corner {
                    64 * rng.below(2)
                } else {
                    rng.below(65)
                };
                (i, j, 0.004 + 0.016 * rng.unit())
            } else {
                (rng.below(65), rng.below(65), 0.08 + 0.12 * rng.unit())
            };
            let centre = vertex(store, key, i, j);
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

/// Every tile one of `dabs` touches by its box in `index`: a stroke's reach.
fn reach(index: &SpatialIndex, dabs: &[Dab]) -> BTreeSet<TileKey> {
    dabs.iter()
        .flat_map(|each| {
            let [x, y, z] = each.position;
            index.query_sphere(Vec3::new(x, y, z), each.radius)
        })
        .collect()
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

/// Every tile's future under `chain`, from one plan.
fn futures(
    store: &mut TileStore,
    state: &Stroke,
    chain: &[Remaining<'_>],
) -> BTreeMap<TileKey, CorrectionTile> {
    let plan = FuturePlan::new(store, state, chain).unwrap();
    let keys = tiles(store.mesh(), level());
    keys.into_iter()
        .map(|key| (key, plan.future_of(store, key).unwrap()))
        .collect()
}

/// How many of `futures` differ from their tiles as stored now: the tiles the strokes still
/// change.
fn changed(store: &mut TileStore, futures: &BTreeMap<TileKey, CorrectionTile>) -> usize {
    futures
        .iter()
        .filter(|&(&key, future)| bits(future) != bits(&stored(store, key)))
        .count()
}

/// Asserts each of `futures` is its tile as stored now, bit for bit.
fn assert_landed(store: &mut TileStore, futures: &BTreeMap<TileKey, CorrectionTile>, case: &str) {
    for (&key, future) in futures {
        let after = stored(store, key);
        assert_eq!(bits(future), bits(&after), "{case}, tile {key:?}");
    }
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

/// One property case, of `seed`, over a store of `stroke_tiles` tiles' stroke state.
///
/// After a stroke Z landed and ended first (if `before`), a random stroke A lands some of its
/// dabs; every tile's future is taken; the rest land; each future is the tile, bit for bit, and
/// (if `check_positions`) its positions are as evaluated too.
fn one_case(seed: u64, stroke_tiles: usize, before: bool, check_positions: bool) {
    let case = format!("seed {seed}, {stroke_tiles} tiles of state, stroke before: {before}");
    let mut rng = Rng::new(seed);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config.with_stroke_tiles(stroke_tiles));
    let mut state = Stroke::new(project.store());
    if before {
        let z = random_stroke(project.store(), &mut rng, 10, 1000);
        project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, &z, &mut rng);
        project.end_unrecorded_stroke().unwrap();
    }
    let dabs = random_stroke(project.store(), &mut rng, 24, 0);
    // Some cases land nothing first: a stroke whose state hasn't begun.
    let landed = if seed.is_multiple_of(4) {
        0
    } else {
        usize::try_from(rng.below(25)).unwrap()
    };
    let (first, rest) = dabs.split_at(landed);
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);

    let reach = reach(&index, rest);
    let remaining = Remaining {
        op,
        dabs: rest,
        reach: &reach,
    };
    let futures = futures(project.store(), &state, &[remaining]);
    let moved = changed(project.store(), &futures);
    assert_eq!(moved > 0, !rest.is_empty(), "{case}: {moved} tiles change");

    land(project.store(), &mut index, &mut state, rest, &mut rng);
    assert_landed(project.store(), &futures, &case);
    if check_positions {
        for (&key, future) in &futures {
            let made = positions_of(project.store(), key, future);
            assert_eq!(
                made,
                evaluated(project.store(), key),
                "{case}, tile {key:?}"
            );
        }
    }
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_future_is_the_tile_once_the_stroke_lands_bit_for_bit() {
    for seed in 1..=6 {
        one_case(seed, StoreConfig::DEFAULT_STROKE_TILES, seed == 3, true);
    }
}

#[test]
fn a_future_is_exact_past_the_stroke_states_room() {
    // Reservations of 4 to 16 tiles: tiles are released, re-taken and written per dab before
    // and during the rest, and the futures are exact all the same.
    let cases = (0..12_u64).map(|n| (100 + n, 4 + usize::try_from(n).unwrap() % 13, n % 3 == 1));
    for (seed, stroke_tiles, before) in cases {
        one_case(seed, stroke_tiles, before, false);
    }
}

/// Lands one dab of radius 0.05 (about 3 quads) at the middle of each of `keys`, in order, each
/// editing that tile alone, of strength `strength`.
fn middles(store: &mut TileStore, keys: &[TileKey], strength: f32, first_seq: u32) -> Vec<Dab> {
    let numbered = keys.iter().zip(first_seq..);
    numbered
        .map(|(&key, seq)| dab(vertex(store, key, 32, 32), 0.05, strength, 0.3, seq))
        .collect()
}

#[test]
fn the_room_boundary_exactly_full_and_one_past() {
    let [t1, t2, t3, t4, t5] = [
        tile(0, 0, 0),
        tile(0, 1, 0),
        tile(0, 0, 1),
        tile(0, 1, 1),
        tile(1, 0, 0),
    ];
    // Four buffers, two held after the first dabs: the rest needs exactly the two free, or one
    // more, which releases the oldest tile, edited again after.
    let cases: [(&str, Vec<TileKey>); 2] = [
        ("exactly the free buffers", vec![t3, t4, t3, t1, t2]),
        ("one past", vec![t3, t4, t5, t1, t1, t2, t3]),
    ];
    for (case, rest_keys) in cases {
        let mut rng = Rng::new(7);
        let dir = TestDir::new();
        let (mut project, mut index) = project(&dir, |config| config.with_stroke_tiles(4));
        let mut state = Stroke::new(project.store());
        // Strengths under a quantization step's worth, so a start captured again shows.
        let first = middles(project.store(), &[t1, t2], 0.0007, 0);
        let rest = middles(project.store(), &rest_keys, 0.0013, 2);
        let op = project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, &first, &mut rng);
        let reach = reach(&index, &rest);
        let remaining = Remaining {
            op,
            dabs: &rest,
            reach: &reach,
        };
        let futures = futures(project.store(), &state, &[remaining]);
        for each in &rest {
            apply_dabs(
                project.store(),
                &mut index,
                &mut state,
                std::slice::from_ref(each),
            )
            .unwrap();
        }
        assert_landed(project.store(), &futures, case);
        project.end_unrecorded_stroke().unwrap();
    }
}

#[test]
fn a_future_without_kept_predictions_is_the_landed_tile() {
    let mut rng = Rng::new(41);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config.with_kept_predictions(0));
    let a = random_stroke(project.store(), &mut rng, 16, 0);
    let b = random_stroke(project.store(), &mut rng, 8, 16);
    let (first, rest) = a.split_at(6);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);
    let a_reach = reach(&index, rest);
    let all: BTreeSet<TileKey> = tiles(project.store().mesh(), level()).into_iter().collect();
    let chain = [
        Remaining {
            op,
            dabs: rest,
            reach: &a_reach,
        },
        Remaining {
            op: op.next().unwrap(),
            dabs: &b,
            reach: &all,
        },
    ];
    let futures = futures(project.store(), &state, &chain);
    land(project.store(), &mut index, &mut state, rest, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, &b, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    assert_landed(project.store(), &futures, "no kept predictions");
}

#[test]
fn a_tile_no_remaining_dab_touches_is_its_stored_self() {
    let mut rng = Rng::new(7);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config);
    let dabs = random_stroke(project.store(), &mut rng, 16, 0);
    let (first, rest) = dabs.split_at(8);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);

    let reach = reach(&index, rest);
    let keys = tiles(project.store().mesh(), level());
    let untouched: Vec<TileKey> = keys
        .iter()
        .filter(|key| !reach.contains(key))
        .copied()
        .collect();
    assert!(!untouched.is_empty(), "the stroke keeps to faces 0 to 2");
    let remaining = Remaining {
        op,
        dabs: rest,
        reach: &reach,
    };
    let plan = FuturePlan::new(project.store(), &state, &[remaining]).unwrap();
    let no_dabs = Remaining {
        dabs: &[],
        ..remaining
    };
    let empty = FuturePlan::new(project.store(), &state, &[no_dabs]).unwrap();
    let no_chain = FuturePlan::new(project.store(), &state, &[]).unwrap();
    for &key in &untouched {
        let as_stored = bits(&stored(project.store(), key));
        let far = plan.future_of(project.store(), key).unwrap();
        assert_eq!(bits(&far), as_stored, "tile {key:?}, dabs elsewhere");
        let none = empty.future_of(project.store(), key).unwrap();
        assert_eq!(bits(&none), as_stored, "tile {key:?}, no dabs");
        let nothing = no_chain.future_of(project.store(), key).unwrap();
        assert_eq!(bits(&nothing), as_stored, "tile {key:?}, no strokes");
    }
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_future_of_a_chain_is_the_tile_once_both_land() {
    // The last seed with four tiles of state: both strokes release tiles.
    for (seed, stroke_tiles) in [(11, 512), (12, 512), (13, 4)] {
        let mut rng = Rng::new(seed);
        let dir = TestDir::new();
        let (mut project, mut index) =
            project(&dir, |config| config.with_stroke_tiles(stroke_tiles));
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

        let a_reach = reach(&index, a_rest);
        // B's reach is every tile: its dabs are measured on A's futures, which the index's boxes
        // don't hold yet.
        let b_reach: BTreeSet<TileKey> =
            tiles(project.store().mesh(), level()).into_iter().collect();
        let op_b: OpId = op_a.next().unwrap();
        let chain = [
            Remaining {
                op: op_a,
                dabs: a_rest,
                reach: &a_reach,
            },
            Remaining {
                op: op_b,
                dabs: &b,
                reach: &b_reach,
            },
        ];
        let futures = futures(project.store(), &state, &chain);
        assert!(changed(project.store(), &futures) > 0, "seed {seed}");

        land(project.store(), &mut index, &mut state, a_rest, &mut rng);
        project.end_unrecorded_stroke().unwrap();
        project.begin_stroke_and_wait().unwrap();
        land(project.store(), &mut index, &mut state, &b, &mut rng);
        project.end_unrecorded_stroke().unwrap();
        assert_landed(project.store(), &futures, &format!("seed {seed}"));
    }
}

#[test]
fn taking_a_future_writes_nothing() {
    let mut rng = Rng::new(21);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config);
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

    let reach = reach(&index, rest);
    let all: BTreeSet<TileKey> = keys.iter().copied().collect();
    let chain = [
        Remaining {
            op,
            dabs: rest,
            reach: &reach,
        },
        Remaining {
            op: op.next().unwrap(),
            dabs: rest,
            reach: &all,
        },
    ];
    futures(project.store(), &state, &chain);

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
fn another_strokes_pen_down_shapes_are_refused() {
    let mut rng = Rng::new(51);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config);
    let dabs = random_stroke(project.store(), &mut rng, 12, 0);
    let (first, rest) = dabs.split_at(6);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, first, &mut rng);
    let reach = reach(&index, rest);
    let shaped = reach
        .iter()
        .any(|key| project.store().pen_down_shape(key).is_some());
    assert!(shaped, "the rest reaches a tile the first dabs changed");
    // The oldest stroke named with an op not the store's: the shapes kept are the current op's.
    let foreign = Remaining {
        op: op.next().unwrap(),
        dabs: rest,
        reach: &reach,
    };
    let refused = FuturePlan::new(project.store(), &state, &[foreign]);
    assert!(
        matches!(
            refused,
            Err(SculptError::Future(FutureError::ForeignPenDown))
        ),
        "{refused:?}"
    );
    project.end_unrecorded_stroke().unwrap();
}
