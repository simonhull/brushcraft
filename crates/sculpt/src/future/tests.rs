//! A tile's future against the tile once its strokes have landed, bit for bit, whatever room the
//! stroke state has; and that taking one writes nothing.

use std::collections::{BTreeMap, BTreeSet};

use geom::{CorrectionTile, FaceId, Level, TileKey, Vec3};
use store::{OpId, Project, ProjectConfig, SpatialIndex, StoreConfig, TileStore};
use stroke::Dab;

use super::{FutureError, Futures, PenDownTiles, StrokeEdits};
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

/// One open stroke, for the tests: its op, its dabs not landed yet, and every tile they touch.
#[derive(Clone, Copy)]
struct Open<'d> {
    /// The stroke's op.
    op: OpId,
    /// Its dabs not landed.
    dabs: &'d [Dab],
    /// Every tile they touch by its box at the stroke's pen-down, or more.
    reach: &'d BTreeSet<TileKey>,
}

/// The edits of each stroke of `chain` (oldest first), placed in batches of random sizes.
///
/// The oldest at the store's pen-down, each later one at the futures of those before it, with
/// `state` the session's stroke state.
fn place_chain(
    store: &mut TileStore,
    state: &Stroke,
    chain: &[Open<'_>],
    rng: &mut Rng,
) -> Vec<StrokeEdits> {
    let mut placed: Vec<StrokeEdits> = Vec::with_capacity(chain.len());
    for open in chain {
        let pen_down = if placed.is_empty() {
            PenDownTiles::of_oldest(store, open.op, open.reach).unwrap()
        } else {
            for earlier in &mut placed {
                earlier.seal();
            }
            let earlier: Vec<&StrokeEdits> = placed.iter().collect();
            let before = Futures::new(store, state, &earlier).unwrap();
            PenDownTiles::of_futures(store, &before, open.reach).unwrap()
        };
        let after = placed.last().map(StrokeEdits::op);
        let mut edits = StrokeEdits::new(open.op, after);
        let mut rest = open.dabs;
        while !rest.is_empty() {
            let size = usize::try_from(1 + rng.below(8)).unwrap().min(rest.len());
            let (batch, later) = rest.split_at(size);
            edits.place(store, batch, &pen_down).unwrap();
            rest = later;
        }
        placed.push(edits);
    }
    placed
}

/// Every tile's future under the chain `edits` (oldest first), from one [`Futures`].
fn futures(
    store: &mut TileStore,
    state: &Stroke,
    edits: &[StrokeEdits],
) -> BTreeMap<TileKey, CorrectionTile> {
    let chain: Vec<&StrokeEdits> = edits.iter().collect();
    let futures = Futures::new(store, state, &chain).unwrap();
    let keys = tiles(store.mesh(), level());
    keys.into_iter()
        .map(|key| (key, futures.future_of(store, key).unwrap()))
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

/// Lands `count` dabs of `dabs` from `from` and drops them from `edits`: the catch-up's step.
fn land_some(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    state: &mut Stroke,
    edits: &mut StrokeEdits,
    dabs: &[Dab],
    rng: &mut Rng,
) {
    land(store, index, state, dabs, rng);
    edits.land(dabs.len());
}

/// One property case, of `seed`, over a store of `stroke_tiles` tiles' stroke state.
///
/// After a stroke Z landed and ended first (if `before`), a random stroke A is placed whole at
/// its pen-down, then lands some of its dabs; every tile's future is taken; the rest land; each
/// future is the tile, bit for bit, and (if `check_positions`) its positions are as evaluated
/// too.
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
    let reach = reach(&index, &dabs);
    let open = Open {
        op,
        dabs: &dabs,
        reach: &reach,
    };
    let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
    let [a] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(project.store(), &mut index, &mut state, a, first, &mut rng);

    let futures = futures(project.store(), &state, &edits);
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

/// One dab of radius 0.05 (about 3 quads) at the middle of each of `keys`, in order, each
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
        let all: Vec<Dab> = first.iter().chain(&rest).copied().collect();
        let reach = reach(&index, &all);
        let open = Open {
            op,
            dabs: &all,
            reach: &reach,
        };
        let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
        let [a] = edits.as_mut_slice() else {
            panic!("one stroke placed");
        };
        land_some(project.store(), &mut index, &mut state, a, &first, &mut rng);
        let futures = futures(project.store(), &state, &edits);
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
fn a_future_after_the_stores_op_changed_starts_the_state_afresh() {
    // The stroke's state last wrote in A's op, but the store has begun another: A's remaining
    // dabs land in that op, from a state released, and the future must assume the same.
    let mut rng = Rng::new(61);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config);
    let dabs = random_stroke(project.store(), &mut rng, 16, 0);
    let (first, rest) = dabs.split_at(8);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    let reach = reach(&index, &dabs);
    let open = Open {
        op,
        dabs: &dabs,
        reach: &reach,
    };
    let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
    let [a] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(project.store(), &mut index, &mut state, a, first, &mut rng);
    let next = project.store().begin_op().unwrap();
    assert_ne!(next, op);
    assert!(
        project.store().pen_down_keys().next().is_none(),
        "a new op drops the shapes"
    );

    let futures = futures(project.store(), &state, &edits);
    assert!(changed(project.store(), &futures) > 0);
    land(project.store(), &mut index, &mut state, rest, &mut rng);
    assert_landed(project.store(), &futures, "the store's op changed");
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
    let a_reach = reach(&index, &a);
    let all: BTreeSet<TileKey> = tiles(project.store().mesh(), level()).into_iter().collect();
    let mut edits = place_chain(
        project.store(),
        &state,
        &[Open {
            op,
            dabs: &a,
            reach: &a_reach,
        }],
        &mut rng,
    );
    let [a_edits] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(
        project.store(),
        &mut index,
        &mut state,
        a_edits,
        first,
        &mut rng,
    );
    let later = place_chain_after(
        project.store(),
        &state,
        &mut edits,
        op.next().unwrap(),
        &b,
        &all,
        &mut rng,
    );
    edits.push(later);
    let futures = futures(project.store(), &state, &edits);
    land(project.store(), &mut index, &mut state, rest, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, &b, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    assert_landed(project.store(), &futures, "no kept predictions");
}

/// The edits of a stroke of op `op` opened after the chain `earlier`, its `dabs` placed whole at
/// the futures of `earlier` over `reach`.
fn place_chain_after(
    store: &mut TileStore,
    state: &Stroke,
    earlier: &mut [StrokeEdits],
    op: OpId,
    dabs: &[Dab],
    reach: &BTreeSet<TileKey>,
    rng: &mut Rng,
) -> StrokeEdits {
    for each in &mut *earlier {
        each.seal();
    }
    let after = earlier.last().map(StrokeEdits::op);
    let chain: Vec<&StrokeEdits> = earlier.iter().collect();
    let before = Futures::new(store, state, &chain).unwrap();
    let pen_down = PenDownTiles::of_futures(store, &before, reach).unwrap();
    let mut edits = StrokeEdits::new(op, after);
    let mut rest = dabs;
    while !rest.is_empty() {
        let size = usize::try_from(1 + rng.below(8)).unwrap().min(rest.len());
        let (batch, later) = rest.split_at(size);
        edits.place(store, batch, &pen_down).unwrap();
        rest = later;
    }
    edits
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
    let reach_all = reach(&index, &dabs);
    let open = Open {
        op,
        dabs: &dabs,
        reach: &reach_all,
    };
    let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
    let [a] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(project.store(), &mut index, &mut state, a, first, &mut rng);

    let rest_reach = reach(&index, rest);
    let keys = tiles(project.store().mesh(), level());
    let untouched: Vec<TileKey> = keys
        .iter()
        .filter(|key| !rest_reach.contains(key))
        .copied()
        .collect();
    assert!(!untouched.is_empty(), "the stroke keeps to faces 0 to 2");
    let chain: Vec<&StrokeEdits> = edits.iter().collect();
    let some = Futures::new(project.store(), &state, &chain).unwrap();
    let empty_edits = StrokeEdits::new(op, None);
    let empty = Futures::new(project.store(), &state, &[&empty_edits]).unwrap();
    let no_chain = Futures::new(project.store(), &state, &[]).unwrap();
    for &key in &untouched {
        let as_stored = bits(&stored(project.store(), key));
        let far = some.future_of(project.store(), key).unwrap();
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
        let a_reach = reach(&index, &a);
        let open = Open {
            op: op_a,
            dabs: &a,
            reach: &a_reach,
        };
        let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
        let [a_edits] = edits.as_mut_slice() else {
            panic!("one stroke placed");
        };
        land_some(
            project.store(),
            &mut index,
            &mut state,
            a_edits,
            a_first,
            &mut rng,
        );

        // B's reach is every tile: its dabs are measured on A's futures, which the index's boxes
        // don't hold yet.
        let b_reach: BTreeSet<TileKey> =
            tiles(project.store().mesh(), level()).into_iter().collect();
        let op_b: OpId = op_a.next().unwrap();
        let later = place_chain_after(
            project.store(),
            &state,
            &mut edits,
            op_b,
            &b,
            &b_reach,
            &mut rng,
        );
        edits.push(later);
        let futures = futures(project.store(), &state, &edits);
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
    let reach = reach(&index, &dabs);
    let open = Open {
        op,
        dabs: &dabs,
        reach: &reach,
    };
    let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
    let [a] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(project.store(), &mut index, &mut state, a, first, &mut rng);
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

    let all: BTreeSet<TileKey> = keys.iter().copied().collect();
    let later = place_chain_after(
        project.store(),
        &state,
        &mut edits,
        op.next().unwrap(),
        rest,
        &all,
        &mut rng,
    );
    edits.push(later);
    futures(project.store(), &state, &edits);

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
fn a_reserved_stroke_reads_its_own_pen_down_once_it_is_the_oldest() {
    let mut rng = Rng::new(81);
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config);
    let a = random_stroke(project.store(), &mut rng, 16, 0);
    let b = random_stroke(project.store(), &mut rng, 10, 16);
    let (first, rest) = a.split_at(8);
    let mut state = Stroke::new(project.store());
    let op_a = project.begin_stroke_and_wait().unwrap();
    let a_reach = reach(&index, &a);
    let open = Open {
        op: op_a,
        dabs: &a,
        reach: &a_reach,
    };
    let mut edits = place_chain(project.store(), &state, &[open], &mut rng);
    let [a_edits] = edits.as_mut_slice() else {
        panic!("one stroke placed");
    };
    land_some(
        project.store(),
        &mut index,
        &mut state,
        a_edits,
        first,
        &mut rng,
    );

    // B is reserved while A lands: its pen-down is A's futures.
    let token = project.store().reserve_op().unwrap();
    let op_b = token.op();
    let all: BTreeSet<TileKey> = tiles(project.store().mesh(), level()).into_iter().collect();
    // Outside a reading of B, the store reads A's shapes: not B's own.
    let foreign = PenDownTiles::of_oldest(project.store(), op_b, &all);
    assert!(
        matches!(
            foreign,
            Err(SculptError::Future(FutureError::ForeignPenDown))
        ),
        "{foreign:?}"
    );
    let mut placed_b = place_chain_after(
        project.store(),
        &state,
        &mut edits,
        op_b,
        &b,
        &all,
        &mut rng,
    );
    placed_b.seal();
    edits.push(placed_b);
    let chained = futures(project.store(), &state, &edits);

    // A lands and ends: B is the oldest, reserved, and reads its own set.
    land(project.store(), &mut index, &mut state, rest, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    let mut b_edits = edits.pop().unwrap();
    b_edits.earlier_landed();
    let alone: BTreeMap<TileKey, CorrectionTile> = {
        let mut reading = project.store().read_pen_down_of(&token).unwrap();
        assert_eq!(reading.reading_op(), Some(op_b));
        let futures = Futures::new(&reading, &state, &[&b_edits]).unwrap();
        let keys = tiles(reading.mesh(), level());
        keys.into_iter()
            .map(|key| (key, futures.future_of(&mut reading, key).unwrap()))
            .collect()
    };
    // The store's op isn't ended at a stroke's end yet (the project's chained strokes, a later
    // task), so B's reserved op can't begin: B lands in an op of its own instead, the same
    // arithmetic (the stroke state starts afresh at either).
    let dropped = project.store().drop_reserved_op(token);
    assert_eq!(dropped, vec![op_b]);
    project.begin_stroke_and_wait().unwrap();
    land(project.store(), &mut index, &mut state, &b, &mut rng);
    project.end_unrecorded_stroke().unwrap();
    assert_landed(project.store(), &chained, "B's futures over A's");
    assert_landed(project.store(), &alone, "B's futures once the oldest");
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
    let refused = PenDownTiles::of_oldest(project.store(), op.next().unwrap(), &reach);
    assert!(
        matches!(
            refused,
            Err(SculptError::Future(FutureError::ForeignPenDown))
        ),
        "{refused:?}"
    );
    project.end_unrecorded_stroke().unwrap();
}

/// Asserts `edits`, the oldest open stroke's, counts exactly the tiles its remaining dabs edit.
///
/// And that its futures take the fast path exactly when those tiles holding no state fit the
/// stroke state's free buffers.
fn assert_fast_path(store: &TileStore, state: &Stroke, edits: &StrokeEdits, case: &str) {
    let edited: BTreeSet<TileKey> = edits
        .dabs
        .iter()
        .flat_map(|dab| dab.edits.iter().copied())
        .collect();
    let counted: BTreeSet<TileKey> = edits.edited.keys().copied().collect();
    assert_eq!(counted, edited, "{case}: the tiles counted as edited");
    let (unheld, free) = if state.continued(store, edits.op()).is_some() {
        let unheld = edited.iter().filter(|key| !state.holds(key)).count();
        (unheld, state.buffers() - state.tiles_held())
    } else {
        (edited.len(), store.stroke_tiles())
    };
    let futures = Futures::new(store, state, &[edits]).unwrap();
    assert_eq!(
        futures.continuous(),
        vec![unheld <= free],
        "{case}: {unheld} tiles to take, {free} free"
    );
}

/// One interleaved case, of `seed`, over 2 to 13 tiles of stroke state.
///
/// A stroke is placed in batches, each read at its pen-down after the dabs landed so far (tiles
/// changed already read from their shapes), random counts landed between batches. Every tile's
/// future is taken, more dabs land, and the futures taken again are the same; the rest land, and
/// each future is the tile, bit for bit.
fn interleaved_case(seed: u64) {
    let mut rng = Rng::new(seed);
    let stroke_tiles = 2 + usize::try_from(rng.below(12)).unwrap();
    let case = format!("seed {seed}, {stroke_tiles} tiles of state");
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config.with_stroke_tiles(stroke_tiles));
    let mut state = Stroke::new(project.store());
    let dabs = random_stroke(project.store(), &mut rng, 30, 0);
    let op = project.begin_stroke_and_wait().unwrap();
    let mut edits = StrokeEdits::new(op, None);
    let (mut placed, mut landed) = (0, 0);
    while placed < dabs.len() {
        let size = usize::try_from(1 + rng.below(6))
            .unwrap()
            .min(dabs.len() - placed);
        let batch = &dabs[placed..placed + size];
        let pen_down = PenDownTiles::of_oldest(project.store(), op, &reach(&index, batch)).unwrap();
        edits.place(project.store(), batch, &pen_down).unwrap();
        placed += size;
        let count =
            usize::try_from(rng.below(u32::try_from(placed - landed).unwrap() + 1)).unwrap();
        let landing = &dabs[landed..landed + count];
        land_some(
            project.store(),
            &mut index,
            &mut state,
            &mut edits,
            landing,
            &mut rng,
        );
        landed += count;
        assert_fast_path(project.store(), &state, &edits, &case);
    }

    let first = futures(project.store(), &state, std::slice::from_ref(&edits));
    let more = usize::try_from(rng.below(u32::try_from(dabs.len() - landed).unwrap() + 1)).unwrap();
    let landing = &dabs[landed..landed + more];
    land_some(
        project.store(),
        &mut index,
        &mut state,
        &mut edits,
        landing,
        &mut rng,
    );
    landed += more;
    assert_fast_path(project.store(), &state, &edits, &case);
    let again = futures(project.store(), &state, std::slice::from_ref(&edits));
    let unchanged = first.iter().zip(&again);
    for ((key, before), (_, after)) in unchanged {
        assert_eq!(
            bits(before),
            bits(after),
            "{case}: tile {key:?}'s future moved"
        );
    }

    land(
        project.store(),
        &mut index,
        &mut state,
        &dabs[landed..],
        &mut rng,
    );
    assert_landed(project.store(), &first, &case);
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_future_holds_while_dabs_are_placed_and_land_in_turn() {
    for seed in 200..216 {
        interleaved_case(seed);
    }
}

#[test]
fn strokes_out_of_chain_are_refused() {
    let mut rng = Rng::new(71);
    let dir = TestDir::new();
    let (mut project, index) = project(&dir, |config| config);
    let dabs = random_stroke(project.store(), &mut rng, 6, 0);
    let state = Stroke::new(project.store());
    let op_a = project.begin_stroke_and_wait().unwrap();
    let op_b = op_a.next().unwrap();
    let reach = reach(&index, &dabs);
    let is = |result: &Result<(), SculptError>, error: FutureError| matches!(result, Err(SculptError::Future(found)) if *found == error);

    let mut a = StrokeEdits::new(op_a, None);
    let oldest = PenDownTiles::of_oldest(project.store(), op_a, &reach).unwrap();
    a.place(project.store(), &dabs, &oldest).unwrap();
    let b = StrokeEdits::new(op_b, Some(op_a));
    // Out of order, or a later stroke alone, as if the one it follows had landed.
    let swapped = Futures::new(project.store(), &state, &[&b, &a]).map(|_| ());
    assert!(is(&swapped, FutureError::Chain), "{swapped:?}");
    let alone = Futures::new(project.store(), &state, &[&b]).map(|_| ());
    assert!(is(&alone, FutureError::Chain), "{alone:?}");

    // A's pen is still down: B's pen-down can't be read on its futures.
    let open = Futures::new(project.store(), &state, &[&a]).unwrap();
    let unsealed = PenDownTiles::of_futures(project.store(), &open, &reach).map(|_| ());
    assert!(is(&unsealed, FutureError::Unsealed), "{unsealed:?}");

    a.seal();
    let sealed = a.place(project.store(), &dabs, &oldest);
    assert!(is(&sealed, FutureError::Sealed), "{sealed:?}");
    let before = Futures::new(project.store(), &state, &[&a]).unwrap();
    let later = PenDownTiles::of_futures(project.store(), &before, &reach).unwrap();
    assert_eq!(later.after(), Some(op_a));
    // B placed on A's own pen-down, or a third stroke on B's: not where they are in the chain.
    let mut b = b;
    let wrong = b.place(project.store(), &dabs, &oldest);
    assert!(is(&wrong, FutureError::Chain), "{wrong:?}");
    let mut c = StrokeEdits::new(op_b.next().unwrap(), Some(op_b));
    let skipped = c.place(project.store(), &dabs, &later);
    assert!(is(&skipped, FutureError::Chain), "{skipped:?}");
    b.place(project.store(), &dabs, &later).unwrap();
    Futures::new(project.store(), &state, &[&a, &b]).unwrap();
    // Once A has landed, B is the oldest.
    b.earlier_landed();
    assert_eq!(b.after(), None);
    Futures::new(project.store(), &state, &[&b]).unwrap();
    project.end_unrecorded_stroke().unwrap();
}

/// How long placing, deciding and a future take for `count` remaining dabs over the cube, with
/// `stroke_tiles` tiles of state, against landing them.
fn time_case(count: u32, stroke_tiles: usize) {
    use std::time::Instant;
    let mut rng = Rng::new(u64::from(count));
    let dir = TestDir::new();
    let (mut project, mut index) = project(&dir, |config| config.with_stroke_tiles(stroke_tiles));
    let dabs = random_stroke(project.store(), &mut rng, count, 0);
    let mut state = Stroke::new(project.store());
    let op = project.begin_stroke_and_wait().unwrap();
    // A few landed, so the state is the stroke's own.
    let (first, rest) = dabs.split_at(8);
    land(project.store(), &mut index, &mut state, first, &mut rng);
    let reach = reach(&index, rest);
    let pen_down = PenDownTiles::of_oldest(project.store(), op, &reach).unwrap();
    let mut edits = StrokeEdits::new(op, None);
    let placing = Instant::now();
    edits.place(project.store(), rest, &pen_down).unwrap();
    let place = placing.elapsed();
    let deciding = Instant::now();
    let futures = Futures::new(project.store(), &state, &[&edits]).unwrap();
    let decide = deciding.elapsed();
    let keys = tiles(project.store().mesh(), level());
    let taking = Instant::now();
    for &key in &keys {
        futures.future_of(project.store(), key).unwrap();
    }
    let per_future = taking.elapsed() / u32::try_from(keys.len()).unwrap();
    let busiest = keys
        .iter()
        .map(|key| edits.touching(key).0.len())
        .max()
        .unwrap_or(0);
    drop(futures);
    let landing_from = Instant::now();
    land(project.store(), &mut index, &mut state, rest, &mut rng);
    let landing = landing_from.elapsed();
    println!(
        "{count} dabs, {stroke_tiles} tiles of state: place {place:?} ({:?} a dab), decide \
         {decide:?}, a future {per_future:?} (busiest tile {busiest} dabs), landing {landing:?}",
        place / count
    );
    project.end_unrecorded_stroke().unwrap();
}

#[test]
#[ignore = "a timing, not a check: run by hand"]
fn timings() {
    for (count, stroke_tiles) in [(2_000, 512), (2_000, 4), (16_000, 512), (16_000, 4)] {
        time_case(count, stroke_tiles);
    }
}
