//! The tile store against `geom`'s dense evaluation, the oracle: every tile bit for bit, with
//! room to spare and under constant eviction, across edits, saves and pins.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionMap, CorrectionTile, DenseEvaluation, FaceId,
    LayerId, Level, PositionSource, Region, TileKey, Vec3, cube_sphere, normal_tiles, normals,
};

use super::config::BATCH_SHARE;
use super::edit::{BorderValues, border_order, update_row};
use super::layers::{first_op, saved_base};
use super::round::{
    Held, Positions, Resident, Tiles, held_index, nothing_to_write, recorded_or_listed,
};
use super::*;
use crate::MAX_SEQ;
use crate::deps::list_normal_inputs;
use crate::file::StoreError;
use crate::fixture::{
    EVAL, Edit, Rng, SITE, TEST_THREADS, TOP, assert_positions, assert_tile, config, damage, dense,
    fixed, level, mesh, meta_for, pools, random_edits, roomy, split_prism, tight, tiles,
};
use crate::op::RESERVED_SITE;
use crate::pool::{Freshness, SlotData, SlotId};
use crate::test_dir::TestDir;
use crate::workers::Workers;

mod differential;
mod layered;
mod recovery;

/// Chunks in [`pinning`]'s budget.
const PINNING_CHUNKS: usize = 12;

/// [`tight`] with room for a few pinned tiles, or for the normals' inputs (pinned while a
/// normal tile is computed), besides an evaluation's own pins.
fn pinning() -> StoreConfig {
    config(PINNING_CHUNKS, 4)
}

/// A store over a new working file `name` in `dir`.
fn new_store(dir: &TestDir, name: &str, mesh: &Arc<BaseMesh>, config: StoreConfig) -> TileStore {
    let file = ProjectFile::create(&dir.path(name), meta_for(mesh), Arc::clone(mesh)).unwrap();
    TileStore::open(file, config).unwrap()
}

/// Applies `edits` to the store (each as a new op) and to the oracle's map.
pub(crate) fn apply(store: &mut TileStore, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        store.begin_op().unwrap();
        dab(store, map, e);
    }
}

/// Applies `edits` to the store as dabs of the current op, as one stroke makes them, and to the
/// oracle's map.
pub(crate) fn apply_dabs(store: &mut TileStore, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        dab(store, map, e);
    }
}

/// Applies `e` to the store as part of the current op, and to the oracle's map.
fn dab(store: &mut TileStore, map: &mut CorrectionMap, e: &Edit) {
    store
        .set_correction(e.face, e.level, e.i, e.j, e.c)
        .unwrap();
    map.set(store.mesh.topology(), e.face, e.level, e.i, e.j, e.c)
        .unwrap();
}

/// Frees `key`'s clean resident correction slot as an eviction does, op included, without
/// writing anything: its next read loads it from its record again.
pub(crate) fn evict(store: &mut TileStore, key: &TileKey) {
    assert!(
        store.corrections.remove(key).unwrap(),
        "{key:?} was resident"
    );
    store.known_ops.remove(key);
}

/// Breaks `key`'s resident correction slot: it no longer holds a valid tile, so it neither
/// encodes nor saves.
pub(crate) fn break_slot(store: &mut TileStore, key: &TileKey) {
    let id = store.corrections.find(key).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
}

/// Asserts that the store's normals of every tile of `keys` are geom's normals of the dense
/// evaluation.
fn assert_normals(
    store: &mut TileStore,
    mesh: &BaseMesh,
    dense: &DenseEvaluation,
    keys: &[TileKey],
    what: &str,
) {
    for key in keys {
        let want = normals(mesh, dense, Region::of_tile(key)).unwrap();
        let got = store.normals(*key).unwrap();
        let region = Region::of_tile(key);
        assert_tile(
            got,
            |i, j| want.get(i - region.x0(), j - region.y0()).unwrap(),
            key,
            what,
        );
    }
}

/// The level-[`TOP`] tile holding face-grid vertex `(i, j)` of `face`.
fn tile_at(face: FaceId, i: u32, j: u32) -> TileKey {
    TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level(TOP), i, j)
        .unwrap()
        .0
}

/// Applies the 300 random edits of the oracle test to a new store with `config`, then checks
/// every tile of levels 1–8 against the dense evaluation. Returns the store for more checks.
fn evaluation_equals_dense(config: StoreConfig) -> (TestDir, TileStore) {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 1));
    let dense = dense(&mesh, &map);
    let all = tiles(&mesh, 1, TOP);
    assert_positions(&mut store, &dense, &all, "first pass");
    // Again, coarse and fine interleaved the other way round: whatever was evicted is
    // recomputed from what is left.
    let reversed: Vec<TileKey> = all.iter().rev().copied().collect();
    assert_positions(&mut store, &dense, &reversed, "second pass");
    assert!(
        store.bytes() <= pools(&config),
        "{} bytes in a budget of {}",
        store.bytes(),
        pools(&config)
    );
    (dir, store)
}

#[test]
fn store_evaluation_equals_dense_with_room() {
    let (_dir, store) = evaluation_equals_dense(roomy());
    assert_eq!(
        store.log.positions.len(),
        tiles(&store.mesh, 1, TOP).len(),
        "with room, each tile is evaluated once"
    );
    assert_eq!(
        store.file.keys().count(),
        0,
        "nothing evicted, so nothing written back"
    );
}

#[test]
fn store_evaluation_equals_dense_under_pressure() {
    let (_dir, store) = evaluation_equals_dense(tight());
    let tiles = tiles(&store.mesh, 1, TOP).len();
    assert!(
        store.log.positions.len() > 2 * tiles,
        "evaluated {} times for {tiles} tiles: evicting constantly",
        store.log.positions.len()
    );
    assert!(
        store.file.keys().count() > 50,
        "dirty correction tiles were written back: {}",
        store.file.keys().count()
    );
}

#[test]
fn normals_equal_geoms_normals_of_the_dense_evaluation() {
    let dir = TestDir::new();
    let mesh = mesh();
    for config in [roomy(), pinning()] {
        let mut store = new_store(&dir, &format!("{}.bcp", config.budget_bytes), &mesh, config);
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 2));
        let dense = dense(&mesh, &map);
        let keys: Vec<TileKey> = [1, 6, 7, 8]
            .iter()
            .flat_map(|&l| tiles(&mesh, l, l))
            .collect();
        assert_normals(&mut store, &mesh, &dense, &keys, "normals");
    }
}

#[test]
fn evaluating_many_equals_the_dense_evaluation_at_any_thread_count() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let finest = tiles(&mesh, TOP, TOP);
    let threads = [
        1,
        2,
        thread::available_parallelism().map_or(1, NonZeroUsize::get),
    ];
    for (base, n) in [roomy(), tight()]
        .into_iter()
        .flat_map(|c| threads.map(|n| (c, n)))
    {
        let config = base.with_workers(fixed(n));
        let name = format!("{}-{n}.bcp", config.budget_bytes);
        let mut store = new_store(&dir, &name, &mesh, config);
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 4));
        let dense = dense(&mesh, &map);
        let what = format!("{} bytes, {n} threads", config.budget_bytes);
        // Every level at once, then the finest level's normals: each tile evaluated from a cold
        // cache in wide batches.
        store.evaluate_many(&all, Evaluation::Positions).unwrap();
        store.evaluate_many(&finest, Evaluation::Normals).unwrap();
        assert_positions(&mut store, &dense, &all, &what);
        assert_normals(&mut store, &mesh, &dense, &finest, &what);
        // Pinned in batches, too.
        let pinned: Vec<TileKey> = finest.iter().step_by(7).copied().collect();
        match store.pin(&pinned) {
            Ok(guard) => assert_positions(&mut store, &dense, guard.keys(), &what),
            Err(CacheError::OverBudget { .. }) => assert_eq!(base, tight(), "{what}: refused"),
            Err(e) => panic!("{what}: pin: {e:?}"),
        }
    }
}

/// Wide batches near the minimum budget leave the correction pool a chunk to load into.
///
/// A batch whose cone pins every evaluated chunk fails, and is halved. Without the correction
/// floor (see [`TileStore::make_room`]) the failed batch takes every correction chunk back, and
/// the one-tile batch it is halved to finds its coarsest level's pins in every evaluated chunk
/// and no correction slot, and fails too, with nothing else pinned. The wide batch is run
/// directly, so the test doesn't depend on how [`TileStore::batch_size`] sizes batches; then the
/// public batched calls must go through.
#[test]
fn wide_batches_near_the_minimum_keep_a_correction_chunk() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let finest = tiles(&mesh, TOP, TOP);
    let chunk = 4 * EVAL;
    let minimum = config(0, 4).minimum_budget(&mesh);
    let mut refused = 0;
    for extra in 0..6 {
        let config = config(0, 4).with_budget_bytes(minimum + extra * chunk);
        let mut store = new_store(&dir, &format!("{extra}.bcp"), &mesh, config);
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 21));
        let what = format!("{extra} chunks over the minimum");
        match store.pin_current(finest.iter().copied().collect()) {
            Ok(held) => store.unpin_held(&held).unwrap(),
            Err(CacheError::Full) => refused += 1,
            Err(e) => panic!("{what}: the wide batch: {e:?}"),
        }
        assert!(
            store.corrections.bytes() >= store.corrections.chunk_bytes(),
            "{what}: the correction pool kept a chunk"
        );
        for _ in 0..2 {
            store
                .evaluate_many(&all, Evaluation::Positions)
                .unwrap_or_else(|e| panic!("{what}: positions: {e:?}"));
            store
                .evaluate_many(&finest, Evaluation::Normals)
                .unwrap_or_else(|e| panic!("{what}: normals: {e:?}"));
            assert!(store.bytes() <= pools(&config), "{what}: over the budget");
        }
        let dense = dense(&mesh, &map);
        assert_positions(&mut store, &dense, &all, &what);
    }
    assert!(refused > 0, "a wide batch ran out of room");
}

/// A batch is sized from the evaluated pool's reachable room: the budget less the correction
/// floor and the correction chunks its pinned and dirty slots fill.
#[test]
fn batches_are_sized_from_the_room_the_evaluated_pool_can_reach() {
    let dir = TestDir::new();
    let mesh = mesh();
    let config = roomy();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let (per_evaluated, per_correction) = (
        store.evaluated.per_chunk() as usize,
        store.corrections.per_chunk() as usize,
    );
    let (chunk, headroom) = (config.chunk_bytes, store.headroom.evaluated);
    let size = |held_chunks: usize| {
        let chunks = (pools(&config) - (held_chunks + 1) * chunk) / chunk;
        (chunks * per_evaluated - headroom) / BATCH_SHARE
    };
    assert_eq!(store.batch_size(0), size(0), "the floor alone");
    // One more dirty tile than a correction chunk holds: two chunks' worth.
    store.begin_op().unwrap();
    for key in tiles(&mesh, TOP, TOP).into_iter().take(per_correction + 1) {
        store
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.001, 0.0, 0.0)))
            .unwrap()
            .unwrap();
    }
    assert_eq!(store.unsaved.len(), per_correction + 1, "dirty tiles");
    let expected = size(2);
    assert_eq!(
        store.batch_size(0),
        expected,
        "the floor and two dirty chunks"
    );
    store.save().unwrap();
    assert_eq!(store.batch_size(0), size(0), "clean again");
}

/// Of `keys`, how many have a slot in `pool`.
fn resident<T: SlotData>(pool: &Pool<T>, keys: &[TileKey]) -> u64 {
    keys.iter()
        .filter(|key| pool.find(key).is_some())
        .count()
        .try_into()
        .unwrap()
}

#[test]
fn activity_counts_what_the_store_did() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, tight());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 200, 5));
    store.log = EvalLog::default();
    let before = store.activity();
    for key in &tiles(&mesh, TOP, TOP) {
        store.normals(*key).unwrap();
    }
    let did = store.activity().since(&before);
    let count = |n: usize| u64::try_from(n).unwrap();
    assert_eq!(
        (
            did.positions_computed,
            did.normals_computed,
            did.corrections_loaded
        ),
        (
            count(store.log.positions.len()),
            count(store.log.normals.len()),
            count(store.log.file_reads.len())
        ),
        "the work the log saw, counted alike"
    );
    assert!(
        did.positions_computed > 0 && did.normals_computed > 0 && did.corrections_loaded > 0,
        "the evaluations did work: {did:?}"
    );
    assert!(
        did.evaluated_evicted > 0 && did.corrections_evicted > 0,
        "the tight budget evicted from both pools: {did:?}"
    );

    let all = tiles(&mesh, 1, TOP);
    let (evaluated, corrections) = (
        resident(&store.evaluated, &all),
        resident(&store.corrections, &all),
    );
    let full = store.activity();
    store.evict_all().unwrap();
    assert_eq!(
        store.activity().since(&full),
        Activity {
            evaluated_evicted: evaluated,
            corrections_evicted: corrections,
            ..Activity::default()
        },
        "emptying the pools counts each resident tile once, and nothing else"
    );
}

#[test]
fn edits_after_evaluation_invalidate_exactly() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 3));
    let all = tiles(&mesh, 1, TOP);
    let finest = tiles(&mesh, TOP, TOP);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    for key in &finest {
        store.normals(*key).unwrap();
    }
    store.log = EvalLog::default();

    // An interior vertex of one finest tile: its only copy.
    let (face, i, j) = (FaceId::new(4), 64 + 17, 128 + 40);
    let edited = tile_at(face, i, j);
    let edit = Edit {
        face,
        level: level(TOP),
        i,
        j,
        c: Vec3::new(0.001, -0.002, 0.0015),
    };
    apply(&mut store, &mut map, &[edit]);
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, "after the edit");
    assert_normals(&mut store, &mesh, &dense, &finest, "after the edit");

    assert_eq!(
        store.log.positions,
        vec![edited],
        "only the edited tile's positions were recomputed"
    );
    let readers: BTreeSet<TileKey> = finest
        .iter()
        .filter(|key| {
            normal_tiles(&mesh, Region::of_tile(key))
                .unwrap()
                .contains(&edited)
        })
        .copied()
        .collect();
    assert!(readers.len() > 1, "the edited tile has neighbours");
    let recomputed: BTreeSet<TileKey> = store.log.normals.iter().copied().collect();
    assert_eq!(
        recomputed, readers,
        "exactly the normals that read the edited tile were dropped"
    );
    assert_eq!(
        store.log.normals.len(),
        readers.len(),
        "each recomputed once"
    );
    let mut neighbours = readers.clone();
    neighbours.remove(&edited);
    assert_eq!(
        store.log.refreshed.iter().copied().collect::<BTreeSet<_>>(),
        neighbours,
        "the neighbours' normals, whose own positions stayed, were brought up to date; the \
         edited tile's were computed in full"
    );
    assert_eq!(
        (
            store.log.normal_inputs_reused.get(),
            store.log.normal_inputs_listed.get()
        ),
        (readers.len(), 0),
        "every input list was read back from the graph: the neighbours' stale normals kept the \
         edited tile's node there"
    );
}

/// Evaluated positions of whole tiles, each vertex answered from its containing tile: the
/// source of a full recompute of normals, as the store's own pinned tiles serve them.
struct Snapshot(BTreeMap<TileKey, Vec<Vec3>>);

impl PositionSource for Snapshot {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, li, lj) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        let index = usize::try_from(lj * key.side() + li).ok()?;
        self.0.get(&key)?.get(index).copied()
    }
}

/// Asserts that the store's normals of every tile of `keys` (all of one level) are a full
/// recompute's from its evaluated positions, bit for bit.
fn assert_normals_recomputed(store: &mut TileStore, keys: &[TileKey], what: &str) {
    let mesh = Arc::clone(&store.mesh);
    let snapshot = Snapshot(
        keys.iter()
            .map(|&key| (key, store.evaluate(key).unwrap().to_vec()))
            .collect(),
    );
    for &key in keys {
        let recomputed = normals(&mesh, &snapshot, Region::of_tile(&key)).unwrap();
        let got = store.normals(key).unwrap();
        for (k, (have, want)) in got.iter().zip(recomputed.as_slice()).enumerate() {
            assert_eq!(have.to_bits(), want.to_bits(), "{what}: {key:?} vertex {k}");
        }
    }
}

/// Every vertex of `face` at `level` within `reach` grid steps of `(ci, cj)` (a square, clipped
/// to the face) gets a random correction: a dab, as the store sees it.
fn square_dab(
    store: &mut TileStore,
    rng: &mut Rng,
    face: FaceId,
    level: Level,
    (ci, cj): (u32, u32),
    reach: u32,
) {
    let n = level.face_quads();
    let span = |c: u32| c.saturating_sub(reach)..=(c + reach).min(n);
    for j in span(cj) {
        for i in span(ci) {
            let c = rng.vec3(0.5 / f32::from(u16::try_from(n).unwrap()));
            store.set_correction(face, level, i, j, c).unwrap();
        }
    }
}

#[test]
fn stale_normals_brought_up_to_date_equal_a_full_recompute() {
    // The subdivided pentagonal prism at level 7: four tiles a face, so every tile has a face
    // corner (valence 3, 4 or 5) and two tile corners on face edges.
    //
    // A tile is only brought up to date when it holds no vertex a dab moved (one that does is
    // recomputed in full), so the dabs that matter move vertices one step off another tile's
    // border: a neighbour's apron. At a base vertex, they move a point of its ring but not the
    // vertex itself; at valence 5, the ring reaches faces that touch the dab's only there.
    let mesh = Arc::new(geom::fixtures::subdivided_pentagonal_prism().unwrap());
    let top = level(7);
    let n = top.face_quads();
    let half = n / 2;
    let all = tiles(&mesh, 7, 7);
    let dabs: Vec<(u32, (u32, u32), u32)> = vec![
        // Ring points of the caps' centres (valence 5), one on its own and one with the
        // vertices next to it, but never the centre.
        (5, (n - 1, n), 0),
        (0, (n, n - 2), 1),
        // A ring point of a corner of the prism (valence 3), and of a valence-4 vertex.
        (0, (1, 0), 0),
        (12, (n, 1), 0),
        // One step off a tile corner on a face edge, across the edge from the next face, and
        // along the edge from the next tile.
        (12, (half, 1), 0),
        (20, (1, half), 0),
        (12, (half + 1, 0), 0),
        // One step off a tile corner inside a face, along each border.
        (25, (half + 1, half), 0),
        (25, (half, half - 1), 0),
        // Squares across tile borders and around a valence-5 centre.
        (7, (half, 30), 3),
        (3, (100, half + 1), 1),
        (5, (n, n), 2),
    ];
    let dir = TestDir::new();
    let dab_count = dabs.len();
    let mut runs = Vec::new();
    for threads in [
        1,
        thread::available_parallelism().map_or(1, NonZeroUsize::get),
    ] {
        let config = config(80, 16).with_workers(fixed(threads));
        let mut store = new_store(&dir, &format!("{threads}.bcp"), &mesh, config);
        store.evaluate_many(&all, Evaluation::Normals).unwrap();
        store.begin_op().unwrap();
        let mut rng = Rng(21);
        let random = (0..8).map(|_| {
            let face = rng.below(mesh.face_count());
            (face, (rng.below(n + 1), rng.below(n + 1)), rng.below(3))
        });
        let dabs: Vec<_> = dabs.iter().copied().chain(random).collect();
        for (k, &(face, at, reach)) in dabs.iter().enumerate() {
            square_dab(&mut store, &mut rng, FaceId::new(face), top, at, reach);
            store.evaluate_many(&all, Evaluation::Normals).unwrap();
            let what = format!("{threads} threads, dab {k} on face {face} at {at:?}");
            assert_normals_recomputed(&mut store, &all, &what);
        }
        assert!(
            store.log.refreshed.len() > 2 * dab_count,
            "{threads} threads: neighbours were brought up to date, not recomputed: {}",
            store.log.refreshed.len()
        );
        runs.push((store.log.normals.clone(), store.log.refreshed.clone()));
    }
    assert!(
        runs.windows(2).all(|pair| pair[0] == pair[1]),
        "the same tiles computed and refreshed whatever the thread count"
    );
}

#[test]
fn structural_lists_are_read_back_from_the_graph() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 17));
    let finest = tiles(&mesh, TOP, TOP);
    for key in tiles(&mesh, 1, TOP) {
        store.evaluate(key).unwrap();
    }
    for key in &finest {
        store.normals(*key).unwrap();
    }
    // Levels 4–7 go, as eviction would take them, then one finest tile: their nodes stay, read
    // by the cached finest tiles and by the neighbours' normals.
    let gone = finest[21];
    for key in tiles(&mesh, 4, TOP - 1).into_iter().chain([gone]) {
        assert!(store.evaluated.remove(&key).unwrap(), "{key:?} was cached");
        store.deps.left_pool(key);
    }
    store.deps.check_invariants().unwrap();
    store.log = EvalLog::default();
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[gone], "rebuilt from levels 3 up");
    assert_normals(&mut store, &mesh, &dense, &[gone], "rebuilt");
    assert!(
        store.log.positions.len() > 4,
        "levels 4–8 were evaluated again: {:?}",
        store.log.positions
    );
    assert_eq!(
        store.log.parents_listed.get(),
        0,
        "every parent list was read back from the graph"
    );
    assert!(
        store.log.parents_reused.get() >= store.log.positions.len(),
        "one per tile built at least: {}",
        store.log.parents_reused.get()
    );
    assert_eq!(
        (
            store.log.normal_inputs_reused.get(),
            store.log.normal_inputs_listed.get()
        ),
        (1, 0),
        "and the normal inputs"
    );
}

#[test]
fn edits_at_a_coarse_level_reach_the_finest() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 4));
    for key in tiles(&mesh, 1, TOP) {
        store.evaluate(key).unwrap();
    }
    // The levels between go, as eviction would take them: the edit must still reach level 8
    // through tiles that are no longer cached.
    for key in tiles(&mesh, 4, TOP - 1) {
        assert!(store.evaluated.remove(&key).unwrap(), "{key:?} was cached");
        store.deps.left_pool(key);
    }
    store.log = EvalLog::default();
    let edits: Vec<Edit> = random_edits(&mesh, 3, 40, 5)
        .into_iter()
        .filter(|e| e.level == level(3))
        .collect();
    assert!(!edits.is_empty(), "some edits at level 3");
    apply(&mut store, &mut map, &edits);
    let dense = dense(&mesh, &map);
    let finest = tiles(&mesh, TOP, TOP);
    assert_positions(&mut store, &dense, &finest, "level 8 after level-3 edits");
    assert!(
        store.log.positions.iter().any(|k| k.level() == level(TOP)),
        "level-8 tiles were recomputed"
    );
}

#[test]
fn sparse_on_disk_equals_explicit_zero() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let edits = random_edits(&mesh, TOP, 120, 6);

    // Explicit: every tile has a record, all zero, before the edits.
    let mut explicit = ProjectFile::create(
        &dir.path("explicit.bcp"),
        meta_for(&mesh),
        Arc::clone(&mesh),
    )
    .unwrap();
    let op = OpId::new(SITE, 0).unwrap();
    for key in &all {
        explicit
            .write_displacement(*key, op, &CorrectionTile::zero(key.side()).unwrap())
            .unwrap();
    }
    explicit.save().unwrap();

    let mut map = CorrectionMap::new();
    let mut sparse = new_store(&dir, "sparse.bcp", &mesh, roomy());
    apply(&mut sparse, &mut map, &edits);
    sparse.save().unwrap();
    let mut explicit = TileStore::open(explicit, tight()).unwrap();
    apply(&mut explicit, &mut CorrectionMap::new(), &edits);
    explicit.save().unwrap();
    assert_eq!(
        explicit.file.keys().count(),
        all.len(),
        "every tile is on disk"
    );
    assert!(
        sparse.file.keys().count() < all.len() / 2,
        "most tiles are absent: {}",
        sparse.file.keys().count()
    );

    let dense = dense(&mesh, &map);
    for key in &all {
        let a: Vec<u32> = sparse
            .evaluate(*key)
            .unwrap()
            .iter()
            .flat_map(|v| v.to_bits())
            .collect();
        let b: Vec<u32> = explicit
            .evaluate(*key)
            .unwrap()
            .iter()
            .flat_map(|v| v.to_bits())
            .collect();
        assert_eq!(a, b, "{key:?}: absent and zero tiles evaluate alike");
    }
    assert_positions(&mut sparse, &dense, &all, "sparse");
}

#[test]
fn save_reopen_evaluate() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("work.bcp");
    let mut map = CorrectionMap::new();
    let last_op = {
        let mut store = TileStore::open(
            ProjectFile::create(&path, meta_for(&mesh), Arc::clone(&mesh)).unwrap(),
            tight(),
        )
        .unwrap();
        let edits = random_edits(&mesh, TOP, 300, 7);
        let (first, second) = edits.split_at(150);
        apply(&mut store, &mut map, first);
        // Evaluation between edits evicts, writing dirty tiles back without a save point.
        for key in tiles(&mesh, 1, TOP) {
            store.evaluate(key).unwrap();
        }
        apply(&mut store, &mut map, second);
        store.save().unwrap();
        assert!(store.unsaved.is_empty(), "every dirty tile written");
        let offset = store.file.len();
        store.save().unwrap();
        assert!(
            store.file.len() - offset < 1024,
            "a save with no edits writes no tiles"
        );
        store.begin_op().unwrap()
    };
    let file = ProjectFile::open(&path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_eq!(
        store.begin_op().unwrap(),
        last_op,
        "ops continue right after the file's head"
    );
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &tiles(&mesh, 1, TOP), "reopened");
}

#[test]
fn over_budget_pin_fails_without_touching_pins() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, pinning());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 100, 8));
    let finest = tiles(&mesh, TOP, TOP);
    let first: Vec<TileKey> = finest.iter().step_by(17).copied().collect();
    let guard = store.pin(&first).unwrap();
    let before: Vec<(SlotId<EvaluatedSlot>, Vec<Vec3>)> = first
        .iter()
        .map(|key| {
            let (id, freshness) = store.evaluated.lookup(key).unwrap();
            assert_eq!(freshness, Freshness::Fresh, "{key:?} fresh");
            (id, store.evaluated.get(id).unwrap().positions.to_vec())
        })
        .collect();
    let resident: Vec<TileKey> = tiles(&mesh, 1, TOP)
        .into_iter()
        .filter(|key| store.evaluated.find(key).is_some())
        .collect();
    let (bytes, written) = (store.bytes(), store.file.len());

    let too_many: Vec<TileKey> = finest.iter().take(4 * PINNING_CHUNKS).copied().collect();
    let refused = store.pin(&too_many);
    assert!(
        matches!(refused, Err(CacheError::OverBudget { .. })),
        "refused: {refused:?}"
    );
    assert_eq!(store.bytes(), bytes, "no chunk mapped or given back");
    assert_eq!(store.file.len(), written, "nothing written back");
    for key in &resident {
        assert!(store.evaluated.find(key).is_some(), "{key:?} not evicted");
    }
    for (key, (id, positions)) in first.iter().zip(&before) {
        assert_eq!(
            store.evaluated.lookup(key),
            Some((*id, Freshness::Fresh)),
            "{key:?} kept its slot"
        );
        assert!(store.evaluated.is_pinned(key), "{key:?} still pinned");
        assert!(store.corrections.is_pinned(key), "its corrections too");
        assert_eq!(
            store.evaluated.get(*id).unwrap().positions.as_slice(),
            positions.as_slice(),
            "{key:?} unchanged"
        );
    }
    assert_eq!(guard.keys(), first.as_slice(), "the guard names its tiles");
    drop(guard);
    store.evaluate(finest[0]).unwrap();
    assert!(
        first.iter().all(|key| !store.evaluated.is_pinned(key)),
        "dropping the guard unpins at the next call"
    );
}

#[test]
fn pins_survive_pressure() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, pinning());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 200, 9));
    let dense = dense(&mesh, &map);
    let finest = tiles(&mesh, TOP, TOP);
    let pinned = [finest[0], finest[33], finest[70]];
    let guard = store.pin(&pinned).unwrap();
    let ids: Vec<SlotId<EvaluatedSlot>> = pinned
        .iter()
        .map(|key| store.evaluated.find(key).unwrap())
        .collect();
    store.log = EvalLog::default();
    assert_positions(&mut store, &dense, &tiles(&mesh, 1, TOP), "the other tiles");
    assert!(
        store.log.positions.len() > 100,
        "under pressure: {} evaluations",
        store.log.positions.len()
    );
    for (key, id) in pinned.iter().zip(&ids) {
        assert_eq!(
            store.evaluated.find(key),
            Some(*id),
            "{key:?} stayed in its slot"
        );
        assert!(
            !store.log.positions.contains(key),
            "{key:?} was never recomputed"
        );
    }
    assert_positions(&mut store, &dense, &pinned, "pinned");
    drop(guard);
}

#[test]
fn corrections_take_chunks_back_from_evaluated_tiles() {
    // Chunks of one evaluated slot or three correction slots: the cube's minimum is eighteen
    // evaluated chunks and a correction chunk.
    let config = config(19, 1);
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let all = tiles(&mesh, 1, 3);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    assert!(
        store.bytes() + store.corrections.chunk_bytes() > pools(&config),
        "the evaluated pool has spent the budget"
    );
    // Every correction slot pinned: the next correction tile has no slot to reuse.
    let resident: Vec<SlotId<CorrectionSlot>> = all
        .iter()
        .filter_map(|key| store.corrections.find(key))
        .collect();
    for &id in &resident {
        store.corrections.pin(id).unwrap();
    }
    let (corrections, evaluated) = (store.corrections.bytes(), store.evaluated.bytes());
    let fresh = tiles(&mesh, 4, 4)[0];
    assert!(store.corrections.find(&fresh).is_none(), "not resident yet");
    store.correction(fresh).unwrap();
    assert!(
        store.corrections.bytes() > corrections && store.evaluated.bytes() < evaluated,
        "an evaluated chunk was taken back for a correction chunk"
    );
    assert!(store.bytes() <= pools(&config), "within the budget");
    for &id in &resident {
        assert!(
            store.corrections.get(id).is_some(),
            "pinned corrections stay"
        );
        store.corrections.unpin(id).unwrap();
    }
}

/// A store over `config` whose budget is spent and whose correction slots all hold dirty tiles.
///
/// Every tile of levels 1–8 is evaluated, then the pools are arranged directly so that the
/// correction pool holds `correction_chunks` chunks and the evaluated pool the rest of the
/// budget. The next correction load finds no free slot, no room for a chunk and no clean victim.
fn spent_on_dirty_corrections(
    dir: &TestDir,
    mesh: &Arc<BaseMesh>,
    config: StoreConfig,
    correction_chunks: usize,
) -> TileStore {
    let name = format!("{}-{correction_chunks}.bcp", config.budget_bytes);
    let mut store = new_store(dir, &name, mesh, config);
    let all = tiles(mesh, 1, TOP);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    let chunk = store.corrections.chunk_bytes();
    // Nothing is dirty yet: giving correction chunks back writes nothing.
    let target = store.budget.used() + correction_chunks * chunk - store.corrections.bytes();
    store
        .corrections
        .shrink_to(&mut store.budget, target, nothing_to_write)
        .unwrap();
    assert_eq!(
        store.corrections.bytes(),
        correction_chunks * chunk,
        "correction chunks"
    );
    // The evaluated pool takes the rest: slots of uncached tiles, freed again at once.
    let mut spare: Vec<TileKey> = all
        .iter()
        .filter(|key| store.evaluated.find(key).is_none())
        .copied()
        .collect();
    let mut taken = Vec::new();
    while store.budget.fits(chunk) {
        let key = spare.pop().unwrap();
        store
            .evaluated
            .acquire(key, &mut store.budget, nothing_to_write)
            .unwrap();
        taken.push(key);
    }
    for key in taken {
        store.evaluated.remove(&key).unwrap();
    }
    store.note_evictions();
    fill_and_dirty_corrections(&mut store, &all);
    assert_eq!(
        store.corrections.bytes() + store.evaluated.bytes(),
        pools(&config),
        "the budget is spent"
    );
    store
}

/// Loads correction tiles into every free correction slot, then dirties every resident one
/// with a new op: the next load finds neither a free slot nor a clean victim.
fn fill_and_dirty_corrections(store: &mut TileStore, all: &[TileKey]) {
    let mut absent: Vec<TileKey> = all
        .iter()
        .filter(|key| store.corrections.find(key).is_none())
        .copied()
        .collect();
    while store.corrections.has_free_slot() {
        store.correction(absent.pop().unwrap()).unwrap();
    }
    dirty_every_resident_correction(store, all);
}

/// What a correction load did for its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Made {
    /// Took an evaluated chunk back and mapped a correction chunk, writing nothing.
    Room,
    /// Wrote one dirty tile back and took its slot.
    WroteOne,
}

/// Loads a correction tile that isn't resident, and says what the store did for its slot.
fn load_one(store: &mut TileStore, all: &[TileKey]) -> Made {
    let key = *all
        .iter()
        .find(|key| store.corrections.find(key).is_none())
        .unwrap();
    let (corrections, evaluated, dirty) = (
        store.corrections.bytes(),
        store.evaluated.bytes(),
        store.unsaved.len(),
    );
    let chunk = store.corrections.chunk_bytes();
    store.correction(key).unwrap();
    let after = (
        store.corrections.bytes(),
        store.evaluated.bytes(),
        store.unsaved.len(),
    );
    if after == (corrections + chunk, evaluated - chunk, dirty) {
        Made::Room
    } else {
        assert_eq!(
            after,
            (corrections, evaluated, dirty - 1),
            "neither a chunk taken nor one tile written"
        );
        Made::WroteOne
    }
}

#[test]
fn a_correction_load_takes_a_clean_tile_before_an_evaluated_chunk() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = spent_on_dirty_corrections(&dir, &mesh, config(10, 4), 1);
    assert!(
        store.may_take_evaluated_chunk(),
        "an evaluated chunk could go"
    );
    let clean = *store.unsaved.keys().next().unwrap();
    store.save().unwrap();
    store.begin_op().unwrap();
    let dirty: Vec<TileKey> = tiles(&mesh, 1, TOP)
        .into_iter()
        .filter(|key| *key != clean && store.corrections.find(key).is_some())
        .collect();
    for key in &dirty {
        store
            .edit_tile(*key, |edit| edit.set(1, 1, Vec3::new(0.002, 0.0, 0.0)))
            .unwrap()
            .unwrap();
    }
    let (written, corrections, evaluated) = (
        store.file.len(),
        store.corrections.bytes(),
        store.evaluated.bytes(),
    );
    let next = tiles(&mesh, 1, TOP)
        .into_iter()
        .find(|key| store.corrections.find(key).is_none())
        .unwrap();
    store.correction(next).unwrap();
    assert!(
        store.corrections.find(&clean).is_none(),
        "the clean tile went"
    );
    assert_eq!(store.file.len(), written, "nothing written back");
    assert_eq!(
        (store.corrections.bytes(), store.evaluated.bytes()),
        (corrections, evaluated),
        "no chunk changed hands"
    );
    for key in &dirty {
        let id = store.corrections.find(key).unwrap();
        assert_eq!(store.corrections.is_dirty(id), Some(true), "{key:?} kept");
    }
}

/// Loading corrections in rounds writes back no more dirty tiles than a tile at a time does.
///
/// A round loads its tiles after the first only into slots to be had without a write-back; a
/// tile at a time unpins each tile before the next, so its slot can be reused.
#[test]
fn a_round_writes_back_no_more_than_a_tile_at_a_time() {
    let mesh = mesh();
    let (one_dir, round_dir) = (TestDir::new(), TestDir::new());
    // Seven chunks: the evaluated pool is at its floor, so every load past the free and clean
    // slots writes a dirty tile back.
    let mut one = spent_on_dirty_corrections(&one_dir, &mesh, config(7, 4), 1);
    let mut round = spent_on_dirty_corrections(&round_dir, &mesh, config(7, 4), 1);
    let cold: Vec<(TileKey, &[TileKey])> = tiles(&mesh, TOP, TOP)
        .into_iter()
        .filter(|key| one.corrections.find(key).is_none())
        .take(12)
        .map(|key| (key, &[][..]))
        .collect();
    assert_eq!(cold.len(), 12, "enough tiles not resident");
    let dirty = one.unsaved.len();
    assert_eq!(round.unsaved.len(), dirty, "the stores start alike");

    for &(key, _) in &cold {
        let id = one.load_correction(key).unwrap();
        one.corrections.pin(id).unwrap();
        one.unpin_corrections(&[id]).unwrap();
    }
    let mut rounds = 0;
    let mut rest = cold.as_slice();
    while !rest.is_empty() {
        let ids = round.load_round(rest).unwrap();
        round.unpin_corrections(&ids).unwrap();
        rest = &rest[ids.len()..];
        rounds += 1;
    }

    let written = |store: &TileStore| dirty - store.unsaved.len();
    assert!(written(&one) > 0, "a tile at a time writes back");
    assert!(
        written(&round) <= written(&one),
        "rounds wrote back {} tiles, a tile at a time {}",
        written(&round),
        written(&one)
    );
    assert!(rounds > 1, "the rounds were cut short");
}

#[test]
fn corrections_take_evaluated_chunks_up_to_a_quarter_of_the_budget() {
    let dir = TestDir::new();
    let mesh = mesh();
    // Ten chunks: corrections take evaluated chunks while they hold one or two, not three.
    let mut store = spent_on_dirty_corrections(&dir, &mesh, config(10, 4), 1);
    let all = tiles(&mesh, 1, TOP);
    let mut made = Vec::new();
    for _ in 0..4 {
        fill_and_dirty_corrections(&mut store, &all);
        made.push(load_one(&mut store, &all));
    }
    assert_eq!(
        made,
        [Made::Room, Made::Room, Made::WroteOne, Made::WroteOne],
        "chunks while under a quarter, then one write-back per load"
    );
    let chunk = store.corrections.chunk_bytes();
    assert_eq!(
        store.corrections.bytes(),
        3 * chunk,
        "three correction chunks"
    );
}

#[test]
fn corrections_leave_the_evaluated_pool_its_floor() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    // The cube's evaluated floor is six chunks: five for an evaluation's pins, and one more.
    for (chunks, want) in [(7, Made::WroteOne), (8, Made::Room)] {
        let mut store = spent_on_dirty_corrections(&dir, &mesh, config(chunks, 4), 1);
        let chunk = store.corrections.chunk_bytes();
        assert_eq!(
            store.headroom.evaluated_floor(chunk),
            6 * chunk,
            "the floor"
        );
        assert!(
            4 * store.corrections.bytes() < store.budget.limit(),
            "{chunks} chunks: under a quarter, so only the floor decides"
        );
        assert_eq!(
            load_one(&mut store, &all),
            want,
            "{chunks} chunks: {} evaluated",
            store.evaluated.bytes() / chunk
        );
    }
}

#[test]
fn a_pinned_tile_is_recomputed_in_place() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 200, 10));
    let (face, i, j) = (FaceId::new(2), 192 + 9, 64 + 50);
    let fine = tile_at(face, i, j);
    let coarse = TileKey::containing(
        LayerId::BASE,
        Channel::DISPLACEMENT,
        face,
        level(7),
        i / 2,
        j / 2,
    )
    .unwrap()
    .0;
    let guard = store.pin(&[fine, coarse]).unwrap();
    store.normals(fine).unwrap();
    let ids = [fine, coarse].map(|key| store.evaluated.find(&key).unwrap());

    // An edit of the fine tile, and one a level up that changes both.
    let edits = [
        Edit {
            face,
            level: level(TOP),
            i,
            j,
            c: Vec3::new(0.002, 0.001, -0.003),
        },
        Edit {
            face,
            level: level(6),
            i: i / 4,
            j: j / 4,
            c: Vec3::new(-0.004, 0.003, 0.001),
        },
    ];
    apply(&mut store, &mut map, &edits);
    for (key, id) in [fine, coarse].iter().zip(ids) {
        assert_eq!(
            store.evaluated.lookup(key),
            Some((id, Freshness::Stale)),
            "{key:?}: marked stale in its slot"
        );
    }
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[coarse, fine], "recomputed");
    assert_normals(&mut store, &mesh, &dense, &[fine], "recomputed");
    for (key, id) in [fine, coarse].iter().zip(ids) {
        assert_eq!(
            store.evaluated.lookup(key),
            Some((id, Freshness::Fresh)),
            "{key:?}: fresh in the same slot"
        );
    }
    drop(guard);
}

#[test]
fn bad_requests_are_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let file = ProjectFile::create(
        &dir.path("reserved.bcp"),
        meta_for(&mesh),
        Arc::clone(&mesh),
    )
    .unwrap();
    let reserved = roomy().with_site(RESERVED_SITE);
    assert!(
        matches!(
            TileStore::open(file, reserved),
            Err(CacheError::ReservedSite)
        ),
        "reserved site"
    );
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let base = TileKey::displacement(FaceId::new(0), Level::ZERO, 0, 0).unwrap();
    let elsewhere = TileKey::displacement(FaceId::new(6), level(1), 0, 0).unwrap();
    for key in [base, elsewhere] {
        assert!(
            matches!(store.evaluate(key), Err(CacheError::NotATile(_))),
            "{key:?}"
        );
    }
    store.begin_op().unwrap();
    let refused = [
        (FaceId::new(0), Level::ZERO, 0, 0, Vec3::ZERO),
        (FaceId::new(9), level(1), 0, 0, Vec3::ZERO),
        (FaceId::new(0), level(1), 3, 0, Vec3::ZERO),
        (
            FaceId::new(0),
            level(1),
            0,
            0,
            Vec3::new(f32::NAN, 0.0, 0.0),
        ),
    ];
    for (face, level, i, j, c) in refused {
        assert!(
            matches!(
                store.set_correction(face, level, i, j, c),
                Err(CacheError::Correction(_))
            ),
            "{face} {level} ({i}, {j})"
        );
    }
    assert_eq!(store.feed.backlog(), 0, "nothing pushed");
    assert!(store.unsaved.is_empty(), "nothing dirty");
}

#[test]
fn a_pin_without_room_for_its_evaluation_is_refused_up_front() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, tight());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 100, 15));
    let finest = tiles(&mesh, TOP, TOP);
    let held = store.pin(&finest[..1]).unwrap();
    let id = store.evaluated.find(&finest[0]).unwrap();
    let resident: Vec<TileKey> = tiles(&mesh, 1, TOP)
        .into_iter()
        .filter(|key| store.evaluated.find(key).is_some())
        .collect();
    let (bytes, written) = (store.bytes(), store.file.len());
    // Two tiles fit the budget as slots, but not with the room their evaluation pins besides.
    let wanted = [finest[9], finest[40]];
    let refused = store.pin(&wanted);
    assert!(
        matches!(refused, Err(CacheError::OverBudget { .. })),
        "refused before evaluating: {refused:?}"
    );
    assert_eq!(store.bytes(), bytes, "no chunk mapped or given back");
    assert_eq!(store.file.len(), written, "nothing written back");
    for key in &resident {
        assert!(store.evaluated.find(key).is_some(), "{key:?} not evicted");
    }
    assert_eq!(store.evaluated.find(&finest[0]), Some(id), "in its slot");
    assert!(store.evaluated.is_pinned(&finest[0]), "and pinned");
    drop(held);
}

#[test]
fn a_pin_that_fails_part_way_releases_only_its_own_pins() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = store_on_disk(&dir, "work.bcp", &mesh, pinning(), 100, 16);
    let finest = tiles(&mesh, TOP, TOP);
    let held = store.pin(&finest[..1]).unwrap();
    let id = store.evaluated.find(&finest[0]).unwrap();
    let on_disk = finest
        .iter()
        .skip(1)
        .find(|key| store.file.contains(key))
        .copied()
        .unwrap();
    store.unreadable = Some(on_disk);
    let wanted = [finest[1].min(on_disk), finest[1].max(on_disk)];
    let refused = store.pin(&wanted);
    assert!(
        matches!(refused, Err(CacheError::Store(_))),
        "the unreadable tile fails the pin: {refused:?}"
    );
    assert_eq!(
        (
            store.evaluated.pinned_slots(),
            store.corrections.pinned_slots()
        ),
        (1, 1),
        "only the first pin set's slots stay pinned"
    );
    assert_eq!(store.evaluated.find(&finest[0]), Some(id), "in its slot");
    drop(held);
}

#[test]
fn a_round_that_fails_part_way_keeps_only_the_tiles_before_the_failure() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = store_on_disk(&dir, "work.bcp", &mesh, roomy(), 300, 5);
    // Finest tiles on disk, in key order: one round of them, the second unreadable.
    let mut wanted: Vec<TileKey> = tiles(&mesh, TOP, TOP)
        .into_iter()
        .filter(|key| store.file.contains(key))
        .take(5)
        .collect();
    assert_eq!(wanted.len(), 5, "enough tiles on disk");
    wanted.sort_unstable();
    store.unreadable = Some(wanted[1]);
    let refused = store.pin(&wanted);
    assert!(
        matches!(refused, Err(CacheError::Store(_))),
        "the unreadable tile fails the pin: {refused:?}"
    );
    assert_eq!(store.corrections.pinned_slots(), 0, "nothing stays pinned");
    let resident: Vec<bool> = wanted
        .iter()
        .map(|key| store.corrections.find(key).is_some())
        .collect();
    assert_eq!(
        resident,
        [true, false, false, false, false],
        "as one tile at a time: the tile before the failure loaded, the rest given back"
    );
    assert!(
        store.known_ops.contains_key(&wanted[0]),
        "with its record's op"
    );
}

#[test]
fn a_budget_below_the_minimum_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let minimum = tight().minimum_budget(&mesh);
    assert!(
        minimum <= tight().budget_bytes,
        "the tests' tight budget is at least the minimum"
    );
    assert!(
        minimum > config(5, 4).budget_bytes,
        "and a chunk less is not"
    );
    let open = |name: &str, config: StoreConfig| {
        let file =
            ProjectFile::create(&dir.path(name), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
        TileStore::open(file, config)
    };
    let kept = tight().kept_prediction_bytes().unwrap();
    let stroke = tight().stroke_state_bytes().unwrap();
    let pools = minimum - kept - stroke;
    let refused = open("small.bcp", tight().with_budget_bytes(pools - 1));
    assert!(
        matches!(refused, Err(CacheError::BudgetTooSmall { minimum: m }) if m == minimum),
        "below the pools' minimum: {refused:?}"
    );
    let none_reserved = tight().with_kept_predictions(0).with_stroke_tiles(0);
    assert_eq!(
        none_reserved.minimum_budget(&mesh),
        pools,
        "nothing kept and no stroke state, nothing set aside"
    );
    assert!(
        open("none.bcp", none_reserved.with_budget_bytes(pools)).is_ok(),
        "the pools' minimum is enough with nothing reserved"
    );
    let short_kept = open("kept.bcp", tight().with_budget_bytes(pools + kept - 1));
    assert!(
        matches!(
            short_kept,
            Err(CacheError::TooManyKeptPredictions {
                asked: 64,
                most: 63
            })
        ),
        "a byte short of the kept predictions' share: {short_kept:?}"
    );
    let short = open("stroke.bcp", tight().with_budget_bytes(minimum - 1));
    assert!(
        matches!(
            short,
            Err(CacheError::TooManyStrokeTiles {
                asked: StoreConfig::DEFAULT_STROKE_TILES,
                most
            }) if most == StoreConfig::DEFAULT_STROKE_TILES - 1
        ),
        "a byte short of the stroke state's share, the kept predictions' set aside first: \
         {short:?}"
    );
    assert!(
        open("exact.bcp", tight().with_budget_bytes(minimum)).is_ok(),
        "the minimum is enough"
    );
}

/// A kept-prediction count beyond any budget is refused with what the budget can keep, not
/// allocated for.
#[test]
fn kept_predictions_beyond_the_budget_are_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    for (n, asked) in [usize::MAX, usize::MAX / Prediction::TILE_BYTES + 1, 1 << 20]
        .into_iter()
        .enumerate()
    {
        let config = roomy().with_kept_predictions(asked);
        let spare = config.budget_bytes - Headroom::of(&mesh).minimum_budget(config.chunk_bytes);
        let file = ProjectFile::create(
            &dir.path(&format!("{n}.bcp")),
            meta_for(&mesh),
            Arc::clone(&mesh),
        )
        .unwrap();
        let refused = TileStore::open(file, config);
        assert!(
            matches!(
                refused,
                Err(CacheError::TooManyKeptPredictions { asked: a, most })
                    if a == asked && most == spare / Prediction::TILE_BYTES
            ),
            "{asked}: {refused:?}"
        );
        assert_eq!(
            config.minimum_budget(&mesh) == usize::MAX,
            config.kept_prediction_bytes().is_none(),
            "{asked}: a share past usize is a minimum no budget meets"
        );
    }
}

/// A stroke-state count whose share doesn't fit beside the pools' minimum and the kept
/// predictions is refused with what the budget can set aside, not allocated for.
#[test]
fn a_stroke_share_that_does_not_fit_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    for (n, asked) in [
        usize::MAX,
        usize::MAX / StoreConfig::STROKE_TILE_BYTES + 1,
        1 << 20,
    ]
    .into_iter()
    .enumerate()
    {
        let config = roomy().with_stroke_tiles(asked);
        let spare = config.budget_bytes
            - Headroom::of(&mesh).minimum_budget(config.chunk_bytes)
            - config.kept_prediction_bytes().unwrap();
        let file = ProjectFile::create(
            &dir.path(&format!("{n}.bcp")),
            meta_for(&mesh),
            Arc::clone(&mesh),
        )
        .unwrap();
        let refused = TileStore::open(file, config);
        assert!(
            matches!(
                refused,
                Err(CacheError::TooManyStrokeTiles { asked: a, most })
                    if a == asked && most == spare / StoreConfig::STROKE_TILE_BYTES
            ),
            "{asked}: {refused:?}"
        );
        assert_eq!(
            config.minimum_budget(&mesh) == usize::MAX,
            config.stroke_state_bytes().is_none(),
            "{asked}: a share past usize is a minimum no budget meets"
        );
    }
    assert_eq!(
        StoreConfig::STROKE_TILE_BYTES,
        33_800,
        "two f32 a vertex of the largest tile"
    );
}

/// The stroke state's share comes out of the pools: a store with it holds exactly its bytes
/// less than one without it, and the store reports the count set aside.
#[test]
fn the_stroke_share_comes_out_of_the_pools() {
    let dir = TestDir::new();
    let mesh = mesh();
    let with = roomy();
    let without = with.with_stroke_tiles(0);
    let share = with.stroke_state_bytes().unwrap();
    assert_eq!(
        share,
        StoreConfig::DEFAULT_STROKE_TILES * StoreConfig::STROKE_TILE_BYTES,
        "the default share"
    );
    assert_eq!(without.stroke_state_bytes(), Some(0), "none set aside");
    let stroked = new_store(&dir, "with.bcp", &mesh, with);
    let plain = new_store(&dir, "without.bcp", &mesh, without);
    assert_eq!(
        plain.budget.limit() - stroked.budget.limit(),
        share,
        "the pools hold the share less"
    );
    assert_eq!(stroked.budget.limit(), pools(&with), "the pools' share");
    assert_eq!(
        stroked.stroke_tiles(),
        StoreConfig::DEFAULT_STROKE_TILES,
        "the count set aside"
    );
    assert_eq!(plain.stroke_tiles(), 0, "none");
}

#[test]
fn shrinking_gives_back_evaluated_chunks_before_corrections() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 11));
    let all = tiles(&mesh, 1, TOP);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    let (corrections, evaluated) = (store.corrections.bytes(), store.evaluated.bytes());
    assert!(
        evaluated > store.evaluated.chunk_bytes(),
        "several evaluated chunks"
    );
    let written = store.file.len();

    store.shrink_to(store.bytes() - evaluated / 2).unwrap();
    assert_eq!(
        store.corrections.bytes(),
        corrections,
        "corrections untouched while evaluated chunks are left to give"
    );
    assert_eq!(store.file.len(), written, "so nothing written back");
    assert!(store.evaluated.bytes() < evaluated, "evaluated chunks went");

    store.shrink_to(0).unwrap();
    assert_eq!(store.bytes(), 0, "everything unpinned goes");
    assert!(
        store.unsaved.is_empty(),
        "dirty tiles were written back first"
    );
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, "after shrinking to nothing");
}

/// Evaluates every tile of levels 1–8 cold, in a random order, on a new store with `config`.
///
/// The order and a few edits come from `seed`; every fifth tile's normals are computed too. The
/// dependency graph is checked after every step, and the positions against the dense evaluation
/// at the end if `check_dense`.
fn random_order_evaluation(config: StoreConfig, seed: u64, check_dense: bool) {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 20, seed));
    let mut order = tiles(&mesh, 1, TOP);
    let mut rng = Rng(seed);
    for n in (1..order.len()).rev() {
        order.swap(n, rng.below(u32::try_from(n + 1).unwrap()) as usize);
    }
    for (n, key) in order.iter().enumerate() {
        if let Err(e) = store.evaluate(*key) {
            panic!("seed {seed}: evaluating {key:?}: {e:?}");
        }
        if n % 5 == 0
            && let Err(e) = store.normals(*key)
        {
            panic!("seed {seed}: normals of {key:?}: {e:?}");
        }
        if let Err(broken) = store.deps.check_invariants() {
            panic!("seed {seed}: after {key:?}: dependency graph: {broken}");
        }
    }
    if check_dense {
        let dense = dense(&mesh, &map);
        assert_positions(&mut store, &dense, &order, &format!("seed {seed}"));
    }
}

#[test]
fn cold_evaluation_in_any_order_fits_the_minimum_budget() {
    // Each seed has a store of its own, so they run side by side.
    thread::scope(|scope| {
        for seed in 1..=40 {
            scope.spawn(move || random_order_evaluation(tight(), seed, seed % 10 == 0));
        }
    });
}

/// One tile and edits of it, in its own coordinates.
type TileBatch = (TileKey, Vec<(u32, u32, Vec3)>);

/// `count` tiles of levels 1–8 picked with `seed`, each with `per_tile` edits in its own
/// coordinates (a third of them on its border, where copies in other tiles follow).
fn tile_batches(mesh: &BaseMesh, count: u32, per_tile: u32, seed: u64) -> Vec<TileBatch> {
    let all = tiles(mesh, 1, TOP);
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let key = all[rng.below(u32::try_from(all.len()).unwrap()) as usize];
            let side = key.side();
            let amplitude = 0.5 / f32::from(u16::try_from(key.level().face_quads()).unwrap());
            let edits = (0..per_tile)
                .map(|_| {
                    let (mut i, mut j) = (rng.below(side), rng.below(side));
                    if rng.below(3) == 0 {
                        if rng.below(2) == 0 {
                            i = [0, side - 1][rng.below(2) as usize];
                        } else {
                            j = [0, side - 1][rng.below(2) as usize];
                        }
                    }
                    (i, j, rng.vec3(amplitude))
                })
                .collect();
            (key, edits)
        })
        .collect()
}

/// Applies `batches` to the store with [`TileStore::edit_tile`], a fresh op each, and to the
/// oracle's map.
fn apply_batches(store: &mut TileStore, map: &mut CorrectionMap, batches: &[TileBatch]) {
    for (key, writes) in batches {
        let region = Region::of_tile(key);
        store.begin_op().unwrap();
        store
            .edit_tile(*key, |edit| {
                for &(i, j, c) in writes {
                    edit.set(i, j, c).unwrap();
                }
            })
            .unwrap();
        for &(i, j, c) in writes {
            let (fi, fj) = (region.x0() + i, region.y0() + j);
            map.set(store.mesh.topology(), key.face(), key.level(), fi, fj, c)
                .unwrap();
        }
    }
}

#[test]
fn editing_a_tile_equals_setting_its_vertices() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut one_by_one = new_store(&dir, "one.bcp", &mesh, roomy());
    let mut batched = new_store(&dir, "batched.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    for (key, edits) in tile_batches(&mesh, 60, 25, 18) {
        let region = Region::of_tile(&key);
        let op = one_by_one.begin_op().unwrap();
        assert_eq!(
            batched.begin_op().unwrap(),
            op,
            "the stores hand out the same ops"
        );
        for &(i, j, c) in &edits {
            let (fi, fj) = (region.x0() + i, region.y0() + j);
            one_by_one
                .set_correction(key.face(), key.level(), fi, fj, c)
                .unwrap();
            map.set(mesh.topology(), key.face(), key.level(), fi, fj, c)
                .unwrap();
        }
        let touched = batched
            .edit_tile(key, |edit| {
                for &(i, j, c) in &edits {
                    edit.set(i, j, c).unwrap();
                }
                assert_eq!(
                    edit.get(0, 0).unwrap(),
                    {
                        let (i, j) = (region.x0(), region.y0());
                        map.get(key.face(), key.level(), i, j)
                    },
                    "reads its own writes"
                );
                edit.touched.len()
            })
            .unwrap();
        assert!(touched >= 1, "{key:?}");
        batched.deps.check_invariants().unwrap();
    }
    let edited: BTreeSet<TileKey> = one_by_one.unsaved.keys().copied().collect();
    assert_eq!(
        edited,
        batched.unsaved.keys().copied().collect(),
        "the same tiles are dirty"
    );
    let feed = |store: &mut TileStore| -> BTreeSet<(TileKey, OpId)> {
        store.feed.drain(store.save_consumer).into_iter().collect()
    };
    assert_eq!(feed(&mut one_by_one), feed(&mut batched), "and on the feed");
    for key in &edited {
        one_by_one.correction(*key).unwrap();
        batched.correction(*key).unwrap();
        assert_eq!(
            slot_resident(&one_by_one, key),
            slot_resident(&batched, key),
            "{key:?}: the same bits"
        );
    }
    let dense = dense(&mesh, &map);
    assert_positions(&mut batched, &dense, &tiles(&mesh, 1, TOP), "batched");
}

/// `evict_all` empties the store but keeps its pools' chunks mapped for the next pin.
///
/// The next pin reuses their slots without mapping another; `shrink_to(0)` is what gives the
/// chunks back. The benchmark's warm-pool page-in relies on the first.
#[test]
fn evicting_everything_keeps_the_chunks_for_the_next_pin() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 200, 31),
    );
    let finest = tiles(&mesh, TOP, TOP);
    let per_face = finest.len() / mesh.faces().count();
    let (first, second) = finest.split_at(per_face);
    let second = &second[..per_face];

    drop(store.pin(first).unwrap());
    let mapped = store.bytes();
    assert!(mapped > 0, "the pin mapped chunks");
    store.evict_all().unwrap();
    assert_eq!(store.bytes(), mapped, "evicting kept every chunk");
    assert!(
        tiles(&mesh, 1, TOP)
            .iter()
            .all(|key| store.evaluated.find(key).is_none() && store.corrections.find(key).is_none()),
        "nothing resident"
    );

    drop(store.pin(second).unwrap());
    assert_eq!(
        store.bytes(),
        mapped,
        "another face's tiles fit in the slots freed: no chunk mapped"
    );
    store.evict_all().unwrap();
    store.shrink_to(0).unwrap();
    assert_eq!(store.bytes(), 0, "shrinking gave every chunk back");
}

#[test]
fn a_failed_write_back_is_reported_and_keeps_the_tile() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("work.bcp");
    let saved = {
        let mut store = TileStore::open(
            ProjectFile::create(&path, meta_for(&mesh), Arc::clone(&mesh)).unwrap(),
            roomy(),
        )
        .unwrap();
        apply(
            &mut store,
            &mut CorrectionMap::new(),
            &random_edits(&mesh, 3, 20, 20),
        );
        store.save().unwrap()
    };
    // Opened at a save point, the file is read-only: edits stay in their slots, and writing
    // them back fails.
    let file = ProjectFile::open_at(&path, saved).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    let edit = random_edits(&mesh, 3, 1, 21);
    apply(&mut store, &mut CorrectionMap::new(), &edit);
    let dirty: Vec<TileKey> = store.unsaved.keys().copied().collect();
    assert!(!dirty.is_empty(), "the edit made tiles dirty");
    let evicted = store.evict_all();
    assert!(
        matches!(evicted, Err(CacheError::WriteBack(StoreError::ReadOnly))),
        "evicting a dirty tile reports the write: {evicted:?}"
    );
    let saving = store.save();
    assert!(
        matches!(saving, Err(CacheError::WriteBack(StoreError::ReadOnly))),
        "and so does saving it: {saving:?}"
    );
    for key in &dirty {
        let id = store.corrections.find(key).unwrap();
        assert_eq!(store.corrections.is_dirty(id), Some(true), "{key:?} kept");
    }
}

#[test]
fn a_tile_edited_by_one_op_goes_to_the_feed_once() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[5];
    let region = Region::of_tile(&key);
    store.begin_op().unwrap();
    for n in 0..50 {
        let (i, j) = (region.x0() + 1 + n, region.y0() + 7);
        store
            .set_correction(key.face(), key.level(), i, j, Vec3::new(0.001, 0.0, 0.0))
            .unwrap();
    }
    assert_eq!(
        store.feed.backlog(),
        1,
        "one interior tile, one op: one entry"
    );
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.backlog(), 1, "batched or not");
    let next = store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 4, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.backlog(), 2, "a new op is a new entry");
    store.save().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 5, Vec3::ZERO))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.feed.drain(store.save_consumer),
        vec![(key, next)],
        "written by the save, so clean: the same op goes to the feed again"
    );
}

#[test]
fn a_consumer_that_drained_a_tile_sees_its_next_edit_of_the_same_op() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A second consumer beside save, which drains nothing here.
    let other = store.feed.subscribe();
    let key = tiles(&mesh, TOP, TOP)[5];
    let op = store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(store.feed.drain(other), vec![(key, op)], "the first dab");
    store
        .edit_tile(key, |edit| edit.set(4, 4, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.feed.drain(other),
        vec![(key, op)],
        "the second dab of the same op, although the tile was dirty with it already"
    );
    assert_eq!(
        store.feed.drain(store.save_consumer),
        vec![(key, op)],
        "save, which drained neither, sees the tile once"
    );
}

#[test]
fn edits_between_evaluations_invalidate_once() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    let key = tiles(&mesh, TOP, TOP)[37];
    let region = Region::of_tile(&key);
    store.normals(key).unwrap();
    store.log = EvalLog::default();
    // One op throughout, as a brush stroke: the store is evaluated between its dabs.
    store.begin_op().unwrap();
    let mut rng = Rng(19);
    let mut dab = |into: &mut TileStore, oracle: &mut CorrectionMap| {
        for _ in 0..40 {
            let (i, j) = (
                region.x0() + 1 + rng.below(63),
                region.y0() + 1 + rng.below(63),
            );
            let c = rng.vec3(0.001);
            into.set_correction(key.face(), key.level(), i, j, c)
                .unwrap();
            oracle
                .set(mesh.topology(), key.face(), key.level(), i, j, c)
                .unwrap();
        }
    };
    dab(&mut store, &mut map);
    assert_eq!(
        store.log.invalidations, 1,
        "the first edit drops the tile's evaluations; the rest find nothing to drop"
    );
    assert_normals(
        &mut store,
        &mesh,
        &dense(&mesh, &map),
        &[key],
        "after a dab",
    );
    dab(&mut store, &mut map);
    assert_eq!(store.log.invalidations, 2, "evaluated since: dropped again");
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[key], "after the second dab, same op");
    assert_normals(
        &mut store,
        &mesh,
        &dense,
        &[key],
        "after the second dab, same op",
    );
    store
        .edit_tile(key, |edit| {
            for n in 0..64 {
                edit.set(n, n, Vec3::ZERO).unwrap();
            }
        })
        .unwrap();
    assert_eq!(store.log.invalidations, 3, "a whole tile edit drops once");
}

/// Which vertices of a tile a test changes, so that a tile's first write falls in each of the
/// places [`TileEdit::update`] and [`TileStore::update_tiles`] mark it written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Any vertex: the first write is on border row 0.
    Anywhere,
    /// Interior vertices only: the tile is marked by an interior write alone.
    InteriorOnly,
    /// Nothing before row 5; in row 5 the interior and the last vertex (a border vertex, whose
    /// copies lie in a neighbour); anything after. The first write is interior, and the border
    /// write right after it marks the neighbour: the order of the two marks is checked.
    InteriorThenBorder,
}

impl Pattern {
    /// The pattern for test case `n`.
    const fn of(n: u32) -> Self {
        match n % 3 {
            0 => Self::Anywhere,
            1 => Self::InteriorOnly,
            _ => Self::InteriorThenBorder,
        }
    }

    /// Whether vertex `(i, j)` of a tile with `side` vertices a side may change.
    const fn allows(self, side: u32, i: u32, j: u32) -> bool {
        let last = side - 1;
        let interior = i > 0 && i < last && j > 0 && j < last;
        match self {
            Self::Anywhere => true,
            Self::InteriorOnly => interior,
            Self::InteriorThenBorder => j > 5 || (j == 5 && (interior || i == last)),
        }
    }
}

/// Applies `change` to every vertex of `key`'s tile in row-major order, through `update` or
/// through `get` and `set` vertex by vertex.
///
/// # Errors
///
/// The edit's or the first write's.
fn change_tile(
    store: &mut TileStore,
    key: TileKey,
    through_update: bool,
    change: &dyn Fn(u32, u32, Vec3) -> Option<Vec3>,
) -> Result<(), CacheError> {
    store
        .edit_tile(key, |edit| {
            if through_update {
                return edit.update(change);
            }
            let side = key.side();
            for (i, j) in (0..side).flat_map(|j| (0..side).map(move |i| (i, j))) {
                if let Some(c) = change(i, j, edit.get(i, j)?) {
                    edit.set(i, j, c)?;
                }
            }
            Ok(())
        })
        .and_then(std::convert::identity)
}

#[test]
fn updating_a_tile_equals_getting_and_setting_each_vertex() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut by_vertex = new_store(&dir, "by_vertex.bcp", &mesh, roomy());
    let mut updated = new_store(&dir, "updated.bcp", &mesh, roomy());
    let all = tiles(&mesh, 1, TOP);
    let mut rng = Rng(0x000B_D7E5);
    for round in 0..24 {
        let key = all[rng.below(u32::try_from(all.len()).unwrap()) as usize];
        let side = key.side();
        let amplitude = 0.5 / f32::from(u16::try_from(key.level().face_quads()).unwrap());
        // About half the vertices change; now and then by a lot, so the scale grows (and the
        // interior is re-quantized) part-way through the tile.
        let deltas: Vec<Option<Vec3>> = (0..side * side)
            .map(|_| match rng.below(40) {
                0..20 => None,
                20 => Some(rng.vec3(amplitude * 40.0)),
                _ => Some(rng.vec3(amplitude)),
            })
            .collect();
        // Every few rounds, a value that isn't finite part-way through a row inside the tile.
        let poisoned = (round % 4 == 3).then_some((side / 2, side / 2));
        let pattern = Pattern::of(round);
        let change = |i: u32, j: u32, c: Vec3| {
            if poisoned == Some((i, j)) {
                return Some(Vec3::new(f32::NAN, 0.0, 0.0));
            }
            if !pattern.allows(side, i, j) {
                return None;
            }
            let d = deltas[(j * side + i) as usize]?;
            Some(Vec3::new(c.x + d.x, c.y + d.y, c.z + d.z))
        };
        let op = by_vertex.begin_op().unwrap();
        assert_eq!(updated.begin_op().unwrap(), op, "the same ops");
        let expected = change_tile(&mut by_vertex, key, false, &change);
        let got = change_tile(&mut updated, key, true, &change);
        assert_eq!(
            format!("{got:?}"),
            format!("{expected:?}"),
            "round {round}: the same outcome"
        );
        if poisoned.is_some() {
            assert!(
                matches!(got, Err(CacheError::Correction(CorrectionError::NotFinite))),
                "round {round}: {got:?}"
            );
        }
    }
    assert_eq!(
        updated.unsaved, by_vertex.unsaved,
        "the same tiles dirty, with the same ops and stamps (written in the same order)"
    );
    let drain = |store: &mut TileStore| store.feed.drain(store.save_consumer);
    assert_eq!(drain(&mut updated), drain(&mut by_vertex), "the same feed");
    let edited: BTreeSet<TileKey> = by_vertex.unsaved.keys().copied().collect();
    for key in &edited {
        assert_eq!(
            slot_resident(&updated, key),
            slot_resident(&by_vertex, key),
            "{key:?}: the same bits"
        );
    }
}

/// A change for [`updating_tiles_equals_updating_each_in_turn`], a pure function of its
/// arguments.
///
/// The round, the tile's place in the list and the vertex pick it, the tiles all of the finest
/// level. About half the vertices a [`Pattern`] allows move; now and then one by a lot (so the
/// scale grows part-way through a tile). In the `poisoned` round, tile 3 gets a value that
/// isn't finite part-way through row 5.
fn tiles_change(round: u32, poisoned: bool, tile: usize, i: u32, j: u32, c: Vec3) -> Option<Vec3> {
    let tile = u32::try_from(tile).unwrap();
    if poisoned && tile == 3 && (i, j) == (9, 5) {
        return Some(Vec3::new(0.0, f32::INFINITY, 0.0));
    }
    // Poisoned, the tiles up to the failing one write no border, so no copy of theirs reaches
    // a later tile: a later tile is marked only if the failure path marks it.
    let pattern = if poisoned && tile <= 3 {
        Pattern::InteriorOnly
    } else {
        Pattern::of(tile + round)
    };
    if !pattern.allows(level(TOP).tile_quads() + 1, i, j) {
        return None;
    }
    let seed = u64::from(round) << 40 ^ u64::from(tile) << 24 ^ u64::from(j) << 12 ^ u64::from(i);
    let mut rng = Rng(seed);
    let delta = match rng.below(40) {
        0..20 => return None,
        20 => rng.vec3(0.05),
        _ => rng.vec3(0.002),
    };
    Some(Vec3::new(c.x + delta.x, c.y + delta.y, c.z + delta.z))
}

/// Runs round `round` of [`updating_tiles_equals_updating_each_in_turn`] on `store`, as a new op.
///
/// The store named "None" edits one tile at a time (the definition), the others with
/// [`TileStore::update_tiles`]. Returns the outcome, printed.
fn update_round(
    name: &str,
    store: &mut TileStore,
    keys: &[TileKey],
    round: u32,
    poisoned: bool,
) -> String {
    store.begin_op().unwrap();
    let change = |n: usize, i: u32, j: u32, c: Vec3| tiles_change(round, poisoned, n, i, j, c);
    let outcome = if name == "None" {
        (0..).zip(keys).try_for_each(|(n, &key)| {
            store
                .edit_tile(key, |edit| edit.update(|i, j, c| change(n, i, j, c)))
                .and_then(std::convert::identity)
        })
    } else {
        store.update_tiles(keys, change)
    };
    format!("{outcome:?}")
}

#[test]
fn border_values_are_found_where_border_order_put_them() {
    for side in [2, 3, 4, 65] {
        let order: Vec<(u32, u32)> = border_order(side).collect();
        let last = side - 1;
        let border: Vec<(u32, u32)> = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || i == last || j == 0 || j == last)
            .collect();
        assert_eq!(
            order, border,
            "side {side}: every border vertex once, row-major"
        );
        let values = BorderValues {
            side,
            values: (0_u16..)
                .zip(&order)
                .map(|(n, _)| Vec3::new(f32::from(n), 0.0, 0.0))
                .collect(),
        };
        for (n, &(i, j)) in (0_u16..).zip(&order) {
            assert_eq!(
                values.get(i, j),
                Some(Vec3::new(f32::from(n), 0.0, 0.0)),
                "side {side}: ({i}, {j})"
            );
        }
        for (i, j) in [(1, 1), (side, 0), (0, side)] {
            if !order.contains(&(i, j)) {
                assert_eq!(
                    values.get(i, j),
                    None,
                    "side {side}: ({i}, {j}) isn't on it"
                );
            }
        }
    }
}

/// Updating tiles together ([`TileStore::update_tiles`]) is the same as updating each in turn.
///
/// The same bits, dirty set and feed, whatever the workers, and with or without the tiles'
/// [`TileBorder`]s (the fallback no valid mesh reaches, forced on one store).
#[test]
fn updating_tiles_equals_updating_each_in_turn() {
    let dir = TestDir::new();
    let mesh = mesh();
    // Neighbouring tiles of the finest level, sharing borders; one of them (5) isn't pinned.
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().take(12).collect();
    let pinned: Vec<TileKey> = (0..)
        .zip(&keys)
        .filter(|&(n, _)| n != 5)
        .map(|(_, &k)| k)
        .collect();
    // The last store's border passes find each vertex's copies one by one, as for a tile with
    // no `TileBorder`.
    let mut stores: Vec<(String, TileStore)> = [None, Some(1), Some(2), Some(5), Some(3)]
        .into_iter()
        .map(|threads| {
            let name = format!("{threads:?}");
            let config = roomy().with_workers(threads.map_or(Workers::PhysicalCores, fixed));
            let mut store = new_store(&dir, &format!("{name}.bcp"), &mesh, config);
            store.without_tile_borders = threads == Some(3);
            (name, store)
        })
        .collect();
    let guards: Vec<PinGuard> = stores
        .iter_mut()
        .map(|(_, store)| store.pin(&pinned).unwrap())
        .collect();
    for round in 0..5 {
        for (name, store) in &mut stores {
            assert_eq!(
                update_round(name, store, &keys, round, false),
                "Ok(())",
                "{name}"
            );
        }
    }
    let (reference, others) = stores.split_first_mut().unwrap();
    let reference = &mut reference.1;
    let edited: BTreeSet<TileKey> = reference.unsaved.keys().copied().collect();
    assert!(
        edited.len() > keys.len(),
        "border copies reached other tiles"
    );
    let expected_feed = reference.feed.drain(reference.save_consumer);
    for (name, store) in others.iter_mut() {
        assert_eq!(
            store.unsaved, reference.unsaved,
            "{name} workers: the same tiles dirty, with the same ops and stamps"
        );
        assert_eq!(
            store.feed.drain(store.save_consumer),
            expected_feed,
            "{name} workers: the same feed"
        );
        for key in &edited {
            assert_eq!(
                slot_resident(store, key),
                slot_resident(reference, key),
                "{name} workers, {key:?}: the same bits"
            );
        }
    }

    // A value that isn't finite in tile 3: the same error; the tiles up to it edited alike;
    // any later tile whose interior was written is dirty in the round's op.
    let before: Vec<Vec<u8>> = keys
        .iter()
        .map(|key| slot_resident(reference, key))
        .collect();
    let expected = update_round("None", reference, &keys, 5, true);
    assert!(expected.contains("NotFinite"), "{expected}");
    for (name, store) in others.iter_mut() {
        assert_eq!(
            update_round(name, store, &keys, 5, true),
            expected,
            "{name} workers"
        );
        let op = store.current_op().unwrap();
        for (n, key) in keys.iter().enumerate() {
            let bits = slot_resident(store, key);
            if n <= 3 {
                assert_eq!(
                    bits,
                    slot_resident(reference, key),
                    "{name} workers, tile {n}"
                );
            } else if n == 4 {
                assert_ne!(
                    bits, before[n],
                    "{name} workers: tile 4's interior was written"
                );
                assert_eq!(
                    store.unsaved.get(key).map(|dirty| dirty.op),
                    Some(op),
                    "{name} workers, tile 4: so it is dirty"
                );
            } else if bits != before[n] {
                assert_eq!(
                    store.unsaved.get(key).map(|dirty| dirty.op),
                    Some(op),
                    "{name} workers, tile {n}: written, so dirty"
                );
            }
        }
    }
    drop(guards);
}

/// Every evaluated and correction slot `store` holds for a tile of `all`, byte for byte, with
/// the slot's freshness and whether its normals are current: what a run's workers wrote, and
/// what the owner decided.
fn every_slot(store: &TileStore, all: &[TileKey]) -> Vec<(TileKey, String, Vec<u8>, Vec<u8>)> {
    all.iter()
        .map(|key| {
            let evaluated = store.evaluated.find(key).map_or_else(Vec::new, |id| {
                bytemuck::bytes_of(store.evaluated.get(id).unwrap()).to_vec()
            });
            let correction = store.corrections.find(key).map_or_else(Vec::new, |id| {
                bytemuck::bytes_of(store.corrections.get(id).unwrap()).to_vec()
            });
            let state = format!(
                "{:?} normals {} dirty {:?}",
                store.evaluated.freshness(key),
                store.deps.has_normals(key),
                store.unsaved.get(key),
            );
            (*key, state, evaluated, correction)
        })
        .collect()
}

/// A stroke as a brush makes it on `keys`, pinned if `pin` is set.
///
/// Each round is a new op of [`tiles_change`] through [`TileStore::update_tiles`], then the
/// positions and normals of the tiles and every tile whose normals read them (`around`)
/// brought up to date together.
fn brush_rounds(store: &mut TileStore, keys: &[TileKey], around: &[TileKey], pin: bool) {
    let guard = pin.then(|| store.pin(keys).unwrap());
    store.evaluate_many(around, Evaluation::Normals).unwrap();
    for round in 0..4 {
        assert_eq!(update_round("tiles", store, keys, round, false), "Ok(())");
        store.evaluate_many(around, Evaluation::Normals).unwrap();
    }
    drop(guard);
}

/// The finest-level tiles [`brush_rounds`] edits, and those it brings up to date.
fn brushed_tiles(mesh: &BaseMesh) -> (Vec<TileKey>, Vec<TileKey>) {
    let keys: Vec<TileKey> = tiles(mesh, TOP, TOP).into_iter().take(12).collect();
    let around: BTreeSet<TileKey> = keys
        .iter()
        .flat_map(|key| list_normal_inputs(mesh, key).unwrap())
        .collect();
    (keys, around.into_iter().collect())
}

/// Slots written on the workers are the same, byte for byte, whatever the thread count.
///
/// A brush's pinned tiles are predicted straight into their slots and every normal is written
/// straight into its slot; every slot, the dirty set and the change feed end up the same. The
/// edits reach unpinned neighbours too (through border copies), so each round mixes tiles
/// predicted in place with tiles predicted into scratch.
#[test]
fn slots_written_on_the_workers_are_the_same_at_any_thread_count() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (keys, around) = brushed_tiles(&mesh);
    let all = tiles(&mesh, 1, TOP);
    let mut runs = Vec::new();
    for threads in [
        1,
        2,
        thread::available_parallelism().map_or(1, NonZeroUsize::get),
    ] {
        let config = roomy().with_workers(fixed(threads));
        let mut store = new_store(&dir, &format!("{threads}.bcp"), &mesh, config);
        brush_rounds(&mut store, &keys, &around, true);
        assert_eq!(
            store.log.in_place.len(),
            4 * keys.len(),
            "{threads} threads: every pinned tile predicted in place each round"
        );
        // The cold evaluation before the first round goes through scratch: look after it.
        let in_place: BTreeSet<TileKey> = store.log.in_place.iter().copied().collect();
        let first = store.log.in_place.first().unwrap();
        let rounds = store.log.positions.iter().skip_while(|key| *key != first);
        assert!(
            rounds.filter(|key| !in_place.contains(key)).count() >= 4,
            "{threads} threads: unpinned tiles went through scratch in the rounds"
        );
        let feed = store.feed.drain(store.save_consumer);
        runs.push((threads, every_slot(&store, &all), feed, store.log.positions));
    }
    let ((_, slots, feed, positions), others) = runs.split_first().unwrap();
    assert!(!feed.is_empty(), "the rounds went to the feed");
    for (threads, other_slots, other_feed, other_positions) in others {
        assert_eq!(other_feed, feed, "{threads} threads: the same feed");
        assert_eq!(
            other_positions, positions,
            "{threads} threads: the same tiles computed, in the same order"
        );
        for (got, want) in other_slots.iter().zip(slots) {
            assert!(
                got == want,
                "{threads} threads, {:?}: the same slot",
                want.0
            );
        }
    }
}

/// A stroke's work on a store, as the tests compare it: every slot of `all`, the dirty tiles
/// with their ops and stamps, and the change feed.
type Worked = (
    Vec<(TileKey, String, Vec<u8>, Vec<u8>)>,
    TileMap<Dirty>,
    Vec<(TileKey, OpId)>,
);

/// A stroke around a parent's edit, on a store keeping `kept` predictions: what the store
/// worked out, and its activity.
///
/// Brush rounds, an edit of a brushed tile once unpinned, then an edit of every vertex of a
/// parent of the first brushed tile (one level coarser) while the brushed tiles are pinned,
/// the pinned tiles evaluated again (their parents moved, so nothing of theirs is kept), brush
/// rounds again, every tile evicted, and brush rounds once more. With a bound below the
/// brushed tiles, the store never holds more predictions than the bound.
fn rounds_around_a_parent_edit(
    dir: &TestDir,
    mesh: &Arc<BaseMesh>,
    kept: usize,
) -> (Worked, Activity) {
    let (keys, around) = brushed_tiles(mesh);
    let config = roomy().with_kept_predictions(kept);
    let mut store = new_store(dir, &format!("kept-{kept}.bcp"), mesh, config);
    brush_rounds(&mut store, &keys, &around, true);
    assert!(
        store.kept.keys().all(|key| keys.contains(key)),
        "only the pinned tiles, predicted in place, keep a prediction"
    );
    assert_eq!(
        store.kept.len(),
        kept.min(keys.len()),
        "every pinned tile keeps one, as far as the bound allows"
    );
    // A brushed tile edited unpinned loses its slot, and its prediction with it.
    store.begin_op().unwrap();
    store
        .edit_tile(keys[1], |edit| edit.set(1, 1, Vec3::new(0.0, 0.0, 0.001)))
        .unwrap()
        .unwrap();
    assert!(store.kept.get(&keys[1]).is_none(), "gone with its slot");
    // The brushed tiles stay pinned through their parent's edit, as a stroke's do: their slots
    // are kept, stale, and predicted again in place.
    let pins = store.pin(&keys).unwrap();
    let parent = *store.deps.parents(&keys[0]).unwrap().first().unwrap();
    store.begin_op().unwrap();
    store
        .edit_tile(parent, |edit| {
            edit.update(|_, _, c| Some(Vec3::new(c.x, c.y + 0.003, c.z - 0.002)))
        })
        .unwrap()
        .unwrap();
    let moved: Vec<TileKey> = keys
        .iter()
        .filter(|key| store.evaluated.freshness(key) == Some(Freshness::Stale))
        .copied()
        .collect();
    assert!(!moved.is_empty(), "the parent's edit reached brushed tiles");
    store.evaluate_many(&keys, Evaluation::Positions).unwrap();
    assert!(
        moved.iter().all(|key| store.kept.get(key).is_none()),
        "tiles predicted again because their parents moved keep nothing"
    );
    brush_rounds(&mut store, &keys, &around, true);
    assert!(
        store.log.predictions_held <= kept,
        "never more held than the bound, new ones included: {}",
        store.log.predictions_held
    );
    drop(pins);
    store.evict_all().unwrap();
    assert_eq!(
        store.kept.keys().count(),
        0,
        "nothing kept once nothing is resident"
    );
    brush_rounds(&mut store, &keys, &around, true);
    let feed = store.feed.drain(store.save_consumer);
    let all = tiles(mesh, 1, TOP);
    (
        (every_slot(&store, &all), store.unsaved.clone(), feed),
        store.activity(),
    )
}

/// Kept predictions change no bits.
///
/// A store keeping them works out the same slots, dirty set and feed as one predicting every
/// tile in full, through a parent's edit (which must drop the predictions of its children) and
/// an eviction of every tile.
#[test]
fn kept_predictions_change_no_bits() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (full, none) = rounds_around_a_parent_edit(&dir, &mesh, 0);
    let (few, bounded) = rounds_around_a_parent_edit(&dir, &mesh, 2);
    let (kept, some) = rounds_around_a_parent_edit(&dir, &mesh, 64);
    assert_eq!(none.predictions_reused, 0, "none kept, none reused");
    assert!(
        (1..some.predictions_reused).contains(&bounded.predictions_reused),
        "two kept: fewer reused, {bounded:?}"
    );
    assert!(few == full, "two kept: the same work");
    assert!(
        some.predictions_reused > 0,
        "the brushed tiles reused their predictions: {some:?}"
    );
    assert_eq!(
        (some.positions_computed, some.normals_computed),
        (none.positions_computed, none.normals_computed),
        "the same tiles evaluated"
    );
    assert_eq!(
        kept.1, full.1,
        "the same tiles dirty, with the same ops and stamps"
    );
    assert_eq!(kept.2, full.2, "the same feed");
    for (got, want) in kept.0.iter().zip(&full.0) {
        assert!(got == want, "{:?}: the same slot", want.0);
    }
    assert_eq!(kept.0.len(), full.0.len(), "the same tiles resident");
}

/// The set-up of the in-place error tests: finest-level edits on a few tiles.
///
/// The edits are applied to a store holding those tiles pinned and their normals' inputs
/// evaluated with normals, and to the oracle's map. Returns the store, the map, the pin, the
/// edited tiles and the tiles around them.
fn pinned_and_edited(
    dir: &TestDir,
    mesh: &Arc<BaseMesh>,
) -> (
    TileStore,
    CorrectionMap,
    PinGuard,
    Vec<TileKey>,
    Vec<TileKey>,
) {
    let mut store = new_store(dir, "work.bcp", mesh, roomy());
    let mut map = CorrectionMap::new();
    let edits: Vec<Edit> = random_edits(mesh, TOP, 200, 11)
        .into_iter()
        .filter(|edit| edit.level == level(TOP))
        .take(12)
        .collect();
    let keys: Vec<TileKey> = edits
        .iter()
        .map(|edit| tile_at(edit.face, edit.i, edit.j))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let around: Vec<TileKey> = keys
        .iter()
        .flat_map(|key| list_normal_inputs(mesh, key).unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let guard = store.pin(&keys).unwrap();
    store.evaluate_many(&around, Evaluation::Normals).unwrap();
    apply(&mut store, &mut map, &edits);
    (store, map, guard, keys, around)
}

/// The bytes of `key`'s evaluated slot, what the workers wrote into it; `None` if it has none.
fn evaluated_bytes(store: &TileStore, key: &TileKey) -> Option<Vec<u8>> {
    let id = store.evaluated.find(key)?;
    Some(bytemuck::bytes_of(store.evaluated.get(id)?).to_vec())
}

/// A round whose prediction fails part-way installs nothing.
///
/// The workers have written the other pinned tiles' slots in place by then: every tile stays
/// stale, and once the failing tile is readable again the store evaluates as the dense oracle
/// does.
#[test]
fn a_round_failing_after_writing_in_place_leaves_its_tiles_stale() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map, guard, keys, _) = pinned_and_edited(&dir, &mesh);
    assert!(keys.len() >= 4, "several tiles: {}", keys.len());
    let broken = keys[keys.len() / 2];
    let id = store.corrections.find(&broken).unwrap();
    let side = store.corrections.get(id).unwrap().side;
    store.corrections.get_mut(id).unwrap().side = 0;
    let before: Vec<Option<Vec<u8>>> = keys.iter().map(|k| evaluated_bytes(&store, k)).collect();
    let computed = store.log.positions.len();

    let failed = store.evaluate_many(&keys, Evaluation::Positions);
    assert!(matches!(failed, Err(CacheError::Internal(_))), "{failed:?}");
    assert_eq!(store.log.positions.len(), computed, "nothing installed");
    for key in &keys {
        assert_eq!(
            store.evaluated.freshness(key),
            Some(Freshness::Stale),
            "{key:?}: still stale"
        );
    }
    let written = keys
        .iter()
        .zip(&before)
        .filter(|&(key, bytes)| *key != broken && evaluated_bytes(&store, key) != *bytes)
        .count();
    assert!(
        written > 0,
        "the workers wrote other tiles' slots before the failure"
    );

    store.corrections.get_mut(id).unwrap().side = side;
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &keys, "recomputed after the failure");
    drop(guard);
}

/// A normals round that fails part-way records nothing.
///
/// The workers have written other tiles' normals in place by then: moved tiles keep no normals,
/// and neighbours stay stale since the same inputs (any vertex rewritten was one their refresh
/// recomputes). Computed again, every normal is the dense oracle's.
#[test]
fn normals_failing_after_writing_in_place_stay_stale() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map, guard, keys, around) = pinned_and_edited(&dir, &mesh);
    let stale: Vec<TileKey> = around
        .iter()
        .copied()
        .filter(|key| store.deps.stale_inputs(key).is_some())
        .collect();
    assert!(
        stale.len() >= 2,
        "neighbours with stale normals: {}",
        stale.len()
    );
    let since: Vec<BTreeSet<TileKey>> = stale
        .iter()
        .map(|key| store.deps.stale_inputs(key).unwrap().clone())
        .collect();
    store.failing_normals = Some(stale[stale.len() / 2]);
    let before: Vec<Option<Vec<u8>>> = around.iter().map(|k| evaluated_bytes(&store, k)).collect();
    let computed = store.log.normals.len();

    let failed = store.evaluate_many(&around, Evaluation::Normals);
    assert!(matches!(failed, Err(CacheError::Internal(_))), "{failed:?}");
    assert_eq!(store.log.normals.len(), computed, "nothing recorded");
    for key in &keys {
        assert!(
            !store.deps.has_normals(key),
            "{key:?}: a moved tile keeps no normals"
        );
    }
    for (key, inputs) in stale.iter().zip(&since) {
        assert_eq!(
            store.deps.stale_inputs(key),
            Some(inputs),
            "{key:?}: still stale"
        );
    }
    let written = around
        .iter()
        .zip(&before)
        .filter(|&(key, bytes)| bytes.is_some() && evaluated_bytes(&store, key) != *bytes)
        .count();
    assert!(written > 0, "the workers wrote slots before the failure");

    store.failing_normals = None;
    store.evaluate_many(&around, Evaluation::Normals).unwrap();
    let dense = dense(&mesh, &map);
    assert_normals(
        &mut store,
        &mesh,
        &dense,
        &around,
        "recomputed after the failure",
    );
    drop(guard);
}

/// A stale slot nothing pins any more isn't written in place: an earlier tile of the same round
/// can take it as the clock's victim when it installs.
///
/// Tiles A were pinned when an edit made them stale, and the pin was dropped after: their slots
/// stay, stale and unreferenced, the clock's first victims. With the pool full and the hand on
/// A's first slot, a round of B (no slot) then A takes A's slot for B. Predicting A in place
/// would have written a slot that is B's by the time A installs.
#[test]
fn a_stale_slot_that_lost_its_pin_is_not_predicted_in_place() {
    let dir = TestDir::new();
    let mesh = mesh();
    let finest = tiles(&mesh, TOP, TOP);
    let mut store = new_store(&dir, "work.bcp", &mesh, config(16, 4));
    let (b, a) = (finest[0], [finest[2], finest[3]]);
    store.evaluate(b).unwrap();
    let guard = store.pin(&a).unwrap();
    assert_eq!(
        update_round("tiles", &mut store, &[b, a[0], a[1]], 0, false),
        "Ok(())"
    );
    drop(guard);
    // Fill the free slots and the budget with tiles nothing else needs, without evicting.
    for key in finest.iter().skip(8) {
        let room =
            store.evaluated.has_free_slot() || store.budget.fits(store.evaluated.chunk_bytes());
        if !room {
            break;
        }
        store.evaluate(*key).unwrap();
    }
    assert!(!store.evaluated.has_free_slot(), "the pool is full");
    let stale = store
        .evaluated
        .find(&a[0])
        .expect("A's stale slot is still there");
    assert_eq!(store.evaluated.freshness(&a[0]), Some(Freshness::Stale));
    assert!(!store.evaluated.is_pinned(&a[0]), "and nothing pins it");
    store.evaluated.aim_hand_at(stale);

    store
        .evaluate_many(&[b, a[0], a[1]], Evaluation::Positions)
        .unwrap();
    for key in [b, a[0], a[1]] {
        assert_eq!(
            store.evaluated.freshness(&key),
            Some(Freshness::Fresh),
            "{key:?}"
        );
    }
    assert!(
        store.log.in_place.is_empty(),
        "nothing pinned, nothing in place"
    );
}

/// The normal inputs the store reads back from its records are those the mesh lists.
///
/// [`TileStore::normal_inputs`] and the bounds book's [`crate::bounds::readers`] read a tile's
/// neighbours from the dependency graph where it recorded them: after a brush's rounds, every
/// tile's list (recorded or listed) is the tile and [`normal_tiles`], and the readers of random
/// edits found through the graph are those found by listing.
#[test]
fn recorded_normal_inputs_are_the_listed_ones() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (keys, around) = brushed_tiles(&mesh);
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    brush_rounds(&mut store, &keys, &around, true);
    let finest = tiles(&mesh, TOP, TOP);
    let mut recorded = 0;
    for key in &finest {
        let mut want = normal_tiles(&mesh, Region::of_tile(key)).unwrap();
        want.insert(*key);
        let want: Vec<TileKey> = want.into_iter().collect();
        assert_eq!(*store.normal_inputs(key).unwrap(), *want, "{key:?}");
        if let Some(list) = store.deps.normal_inputs(key) {
            assert_eq!(list, want.as_slice(), "{key:?}: recorded");
            recorded += 1;
        }
    }
    assert!(
        recorded >= around.len(),
        "the brushed tiles' lists were recorded"
    );
    let levels = BTreeSet::from([level(TOP)]);
    let empty = DepGraph::new();
    let mut rng = Rng(7);
    for _ in 0..20 {
        let edited: BTreeSet<TileKey> = (0..3)
            .map(|_| finest[usize::try_from(rng.below(48)).unwrap()])
            .collect();
        assert_eq!(
            crate::bounds::readers(&mesh, &store.deps, &edited, &levels).unwrap(),
            crate::bounds::readers(&mesh, &empty, &edited, &levels).unwrap(),
            "{edited:?}"
        );
    }
}

/// The bounds book's readers through recorded neighbour lists equal readers by listing, for
/// edits at coarser levels than those tracked and several tracked levels at once.
///
/// Every tile of levels 5 to 8 has its normals computed, so its neighbour list is recorded;
/// coarser tiles have none. Edits at levels 3 to 8 then walk down through recorded and
/// unrecorded levels alike.
#[test]
fn readers_through_recorded_lists_equal_readers_by_listing_at_every_level() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    store
        .evaluate_many(&tiles(&mesh, 5, TOP), Evaluation::Normals)
        .unwrap();
    let recorded = tiles(&mesh, 5, TOP)
        .iter()
        .filter(|key| store.deps.normal_inputs(key).is_some())
        .count();
    assert!(recorded > 100, "levels 5 to 8 recorded: {recorded}");
    let empty = DepGraph::new();
    let candidates = tiles(&mesh, 3, TOP);
    let mut rng = Rng(23);
    for tracked in [
        BTreeSet::from([level(TOP)]),
        BTreeSet::from([level(6), level(TOP)]),
        BTreeSet::from([level(5), level(6), level(7), level(TOP)]),
    ] {
        for _ in 0..12 {
            let count = u32::try_from(candidates.len()).unwrap();
            let edited: BTreeSet<TileKey> = (0..=rng.below(3))
                .map(|_| candidates[usize::try_from(rng.below(count)).unwrap()])
                .collect();
            let through = crate::bounds::readers(&mesh, &store.deps, &edited, &tracked).unwrap();
            let listed = crate::bounds::readers(&mesh, &empty, &edited, &tracked).unwrap();
            assert!(!listed.is_empty(), "{edited:?} reaches a tracked level");
            assert_eq!(through, listed, "{edited:?}, tracking {tracked:?}");
        }
    }
}

/// Recorded lists and listed ones come back in key order, mixed, and the error is the first
/// in key order, whatever the thread count.
///
/// A recorded value and a listed one differ (`10n` against `10n + 1`), so a record ignored for
/// a listing shows; only keys without a record are listed, so a listing that would fail for a
/// recorded key is never run.
#[test]
fn recorded_or_listed_keeps_key_order() {
    let mesh = mesh();
    let keys = tiles(&mesh, TOP, TOP);
    let keys = &keys[..40];
    let index = |key: &TileKey| keys.iter().position(|k| k == key).unwrap();
    let is_recorded = |n: usize| n.is_multiple_of(3);
    for threads in [1, 4] {
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let recorded = |key: &TileKey| is_recorded(index(key)).then(|| index(key) * 10);
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let listed = recorded_or_listed(&workers, keys, recorded, |key| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match index(key) {
                n if is_recorded(n) => Err(CacheError::NotATile(*key)),
                n => Ok(n * 10 + 1),
            }
        });
        let want: Vec<usize> = (0..keys.len())
            .map(|n| if is_recorded(n) { n * 10 } else { n * 10 + 1 })
            .collect();
        assert_eq!(listed.unwrap(), want, "{threads} threads");
        let unrecorded = (0..keys.len()).filter(|&n| !is_recorded(n)).count();
        assert_eq!(
            calls.into_inner(),
            unrecorded,
            "{threads} threads: only the keys without a record are listed"
        );
        let failing = recorded_or_listed(&workers, keys, recorded, |key| match index(key) {
            n @ (7 | 20) => Err(CacheError::NotATile(keys[n])),
            n => Ok(n),
        });
        assert!(
            matches!(failing, Err(CacheError::NotATile(key)) if key == keys[7]),
            "{threads} threads: the first failure in key order"
        );
    }
}

/// Predicting a pinned tile straight into its slot gives the same bits as through scratch.
///
/// A tile that isn't pinned loses its slot to an edit, so it is predicted into scratch and
/// copied into a slot taken afresh.
#[test]
fn predicting_in_place_equals_predicting_into_scratch() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (keys, around) = brushed_tiles(&mesh);
    let mut stores = [true, false].map(|pin| {
        let mut store = new_store(&dir, &format!("{pin}.bcp"), &mesh, roomy());
        brush_rounds(&mut store, &keys, &around, pin);
        store
    });
    let [in_place, scratch] = &mut stores;
    assert!(!in_place.log.in_place.is_empty(), "pinned: in place");
    assert!(scratch.log.in_place.is_empty(), "not pinned: into scratch");
    for &key in &around {
        let positions = scratch.evaluate(key).unwrap().to_vec();
        let same = in_place.evaluate(key).unwrap().iter().zip(&positions);
        assert!(
            same.into_iter().all(|(a, b)| a.to_bits() == b.to_bits()),
            "{key:?}: positions"
        );
        let normals = scratch.normals(key).unwrap().to_vec();
        let alike = in_place.normals(key).unwrap().iter().zip(&normals);
        assert!(
            alike.into_iter().all(|(a, b)| a.to_bits() == b.to_bits()),
            "{key:?}: normals"
        );
    }
}

/// A tile of a run whose border fails before the row of its first interior write still has its
/// interior pass applied: it is marked written, never left changed and clean.
///
/// Tile 1 of three pinned tiles gets a value that isn't finite at border vertex `(0, 2)` and a
/// write at interior vertex `(4, 5)`, which the parallel pass makes before the border is
/// reached; tile 2, after the failure, an interior write too.
#[test]
fn a_failed_update_still_marks_what_it_wrote() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().take(3).collect();
    let guard = store.pin(&keys).unwrap();
    store.evaluate_many(&keys, Evaluation::Positions).unwrap();
    let before: Vec<Vec<u8>> = keys.iter().map(|key| slot_resident(&store, key)).collect();
    let op = store.begin_op().unwrap();
    let outcome = store.update_tiles(&keys, |n, i, j, c| match (n, i, j) {
        (1, 0, 2) => Some(Vec3::new(f32::NAN, 0.0, 0.0)),
        (1 | 2, 4, 5) => Some(Vec3::new(c.x, c.y, c.z + 0.01)),
        _ => None,
    });
    assert!(
        matches!(
            outcome,
            Err(CacheError::Correction(CorrectionError::NotFinite))
        ),
        "{outcome:?}"
    );
    let fed: Vec<TileKey> = store
        .feed
        .drain(store.save_consumer)
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    for (n, key) in keys.iter().enumerate() {
        if slot_resident(&store, key) == before[n] {
            assert!(
                !store.unsaved.contains_key(key),
                "tile {n}: unchanged, clean"
            );
            continue;
        }
        assert_eq!(
            store.unsaved.get(key).map(|dirty| dirty.op),
            Some(op),
            "tile {n}: changed, so dirty in the op"
        );
        assert!(fed.contains(key), "tile {n}: changed, so on the feed");
        assert_eq!(
            store.cached_positions(key),
            None,
            "tile {n}: changed, so its evaluation is dropped"
        );
    }
    assert_ne!(
        slot_resident(&store, &keys[1]),
        before[1],
        "tile 1's interior pass was applied"
    );
    drop(guard);
}

#[test]
fn the_cache_is_read_only_while_current() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[7];
    assert_eq!(store.cached_positions(&key), None, "nothing evaluated yet");
    let positions = store.evaluate(key).unwrap().to_vec();
    assert_eq!(store.cached_positions(&key), Some(positions.as_slice()));
    assert_eq!(store.cached_normals(&key), None, "positions alone");
    let normals = store.normals(key).unwrap().to_vec();
    assert_eq!(store.cached_normals(&key), Some(normals.as_slice()));
    store.begin_op().unwrap();
    store
        .edit_tile(key, |edit| edit.set(3, 3, Vec3::new(0.0, 0.0, 0.001)))
        .unwrap()
        .unwrap();
    assert_eq!(
        store.cached_positions(&key),
        None,
        "an edit makes them stale"
    );
    assert_eq!(store.cached_normals(&key), None, "and the normals");
    let squares = store.par_map(&[1_u32, 2, 3, 4, 5], |&n| n * n);
    assert_eq!(squares, [1, 4, 9, 16, 25], "in item order");
}

#[test]
fn an_unread_edit_still_reaches_untracked_tiles() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A tile recorded after its parent's node went: the parent's ancestry is unknown.
    let (fine, parent) = (tiles(&mesh, 4, 4)[0], tiles(&mesh, 3, 3)[0]);
    store.deps.record_positions(fine, &[parent]);
    assert!(store.deps.has_untracked(), "the parent is untracked");
    // Never evaluated, read by nothing recorded: skipped, but for the untracked node.
    let coarse = tiles(&mesh, 2, 2)[0];
    assert!(
        store.deps.cascade([coarse]).positions.contains(&fine),
        "an edit of it reaches the untracked node's reader"
    );
    store.log = EvalLog::default();
    store.begin_op().unwrap();
    store
        .edit_tile(coarse, |edit| edit.set(1, 1, Vec3::new(0.001, 0.0, 0.0)))
        .unwrap()
        .unwrap();
    assert_eq!(store.log.invalidations, 1, "the cascade ran");
    assert_eq!(store.deps.parents(&fine), None, "and dropped the reader");
}

#[test]
fn an_edit_that_panics_releases_its_pins() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    let finest = tiles(&mesh, TOP, TOP);
    // A corner tile: the corner vertex's copies lie in tiles of three faces.
    let key = finest[0];
    for tile in &finest {
        store.evaluate(*tile).unwrap();
    }
    store.begin_op().unwrap();
    let c = Vec3::new(0.001, -0.002, 0.0);
    let edited = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.edit_tile(key, |edit| {
            edit.set(0, 0, c).unwrap();
            panic!("the brush gave up");
        })
    }));
    let panic = edited.expect_err("the panic came through");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"the brush gave up"),
        "the closure's own panic"
    );
    assert_eq!(
        store.corrections.pinned_slots(),
        0,
        "no touched tile stays pinned"
    );
    let region = Region::of_tile(&key);
    map.set(
        mesh.topology(),
        key.face(),
        key.level(),
        region.x0(),
        region.y0(),
        c,
    )
    .unwrap();
    let dense = dense(&mesh, &map);
    assert_positions(
        &mut store,
        &dense,
        &finest,
        "the write before the panic, evaluated",
    );
}

#[test]
fn a_copy_that_fails_to_write_keeps_its_written_twins_invalidated() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    // A vertex on a face edge, on a tile border: four copies.
    let (face, level, i, j) = (FaceId::new(1), level(TOP), 64, 0);
    let copies = mesh.topology().copies(face, level, i, j);
    assert_eq!(copies.len(), 4, "four copies");
    for copy in &copies {
        store.evaluate(copy.key(level)).unwrap();
    }
    let (first, broken) = (copies[0].key(level), copies[2].key(level));
    store.correction(broken).unwrap();
    let id = store.corrections.find(&broken).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
    store.begin_op().unwrap();
    let c = Vec3::new(0.002, 0.0, 0.0);
    let failed = store.set_correction(face, level, i, j, c);
    assert!(
        matches!(
            failed,
            Err(CacheError::Internal(Invariant::CorrectionSlot(_)))
        ),
        "the broken copy fails the edit as a broken invariant, not as rejected input: {failed:?}"
    );
    assert_eq!(
        store
            .correction(first)
            .unwrap()
            .get(copies[0].i, copies[0].j),
        Some(c),
        "the first copy was written"
    );
    assert_ne!(
        store.evaluated.freshness(&first),
        Some(Freshness::Fresh),
        "and its evaluation dropped"
    );
}

/// Polls until every prefetch request is answered; returns how many tiles were installed.
///
/// Fails the test if that takes more than ten seconds: the reads are a few kilobytes each.
fn drain_prefetch(store: &mut TileStore) -> usize {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut installed = store.poll().unwrap();
    while store.prefetching() > 0 {
        assert!(Instant::now() < deadline, "prefetch didn't finish");
        thread::sleep(Duration::from_millis(1));
        installed += store.poll().unwrap();
    }
    installed
}

/// The bytes of `key`'s latest record in the working file, decoded into a slot (see
/// [`CorrectionSlot::tile_bytes`]).
fn slot_in_file(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let (_, tile) = store.file.read_displacement(key).unwrap().unwrap();
    CorrectionSlot::from_tile(&tile).tile_bytes()
}

/// The bytes of the tile `key`'s resident correction slot holds (see
/// [`CorrectionSlot::tile_bytes`]).
fn slot_resident(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let id = store.corrections.find(key).unwrap();
    store.corrections.get(id).unwrap().tile_bytes()
}

/// A store over a new working file `name` holding the correction tiles of `count` random edits
/// (seeded with `seed`), saved, with nothing resident; the oracle's map of the same edits.
fn store_on_disk(
    dir: &TestDir,
    name: &str,
    mesh: &Arc<BaseMesh>,
    config: StoreConfig,
    count: u32,
    seed: u64,
) -> (TileStore, CorrectionMap) {
    let mut store = new_store(dir, name, mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(mesh, TOP, count, seed));
    store.save().unwrap();
    store.shrink_to(0).unwrap();
    assert_eq!(store.bytes(), 0, "nothing resident");
    (store, map)
}

#[test]
fn prefetched_tiles_are_used_without_reading_the_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map) = store_on_disk(&dir, "work.bcp", &mesh, roomy(), 300, 12);
    let on_disk: Vec<TileKey> = store.file.keys().copied().collect();
    assert!(on_disk.len() > 50, "{} tiles on disk", on_disk.len());
    let absent = tiles(&mesh, 1, TOP)
        .into_iter()
        .find(|key| !store.file.contains(key))
        .unwrap();
    let mut wanted = on_disk.clone();
    // An absent tile (zero) and a repeated one aren't read.
    wanted.push(absent);
    wanted.push(on_disk[0]);
    assert_eq!(store.prefetch(&wanted).unwrap(), on_disk.len(), "queued");
    assert_eq!(
        store.prefetch(&on_disk).unwrap(),
        0,
        "pending requests aren't queued twice"
    );
    assert_eq!(drain_prefetch(&mut store), on_disk.len(), "all installed");
    assert_eq!(store.prefetch(&on_disk).unwrap(), 0, "resident: not queued");

    store.log = EvalLog::default();
    for key in &on_disk {
        store.correction(*key).unwrap();
        assert_eq!(
            slot_resident(&store, key),
            slot_in_file(&store, key),
            "{key:?}: the prefetched slot is the file's tile"
        );
    }
    assert_eq!(store.log.file_reads, [], "no read on the owner's thread");
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &tiles(&mesh, 1, TOP), "prefetched");
    assert!(
        store.log.file_reads.contains(&absent),
        "the absent tile was looked up"
    );
    assert!(
        store
            .log
            .file_reads
            .iter()
            .all(|key| !store.file.contains(key)),
        "and only absent tiles: every tile on disk was prefetched"
    );
}

#[test]
fn a_tile_edited_after_its_prefetch_keeps_the_edit() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = store_on_disk(&dir, "work.bcp", &mesh, roomy(), 300, 13);
    let (face, i, j) = (FaceId::new(4), 128 + 20, 64 + 30);
    let rewritten = tile_at(face, i, j);
    let (i2, j2) = (i + 64, j);
    let resident = tile_at(face, i2, j2);
    assert!(
        store.file.contains(&rewritten) && store.file.contains(&resident),
        "both on disk"
    );
    let edit = |at_i, at_j| Edit {
        face,
        level: level(TOP),
        i: at_i,
        j: at_j,
        c: Vec3::new(0.003, -0.002, 0.001),
    };

    // The workers are held back, so their results arrive only after the edits.
    store.prefetch.pause(true);
    assert_eq!(store.prefetch(&[rewritten, resident]).unwrap(), 2);
    apply(&mut store, &mut map, &[edit(i, j), edit(i2, j2)]);
    // `rewritten` is written back (its record moves) and leaves the pool; `resident` stays,
    // dirty, with its record unchanged.
    store.correction(resident).unwrap();
    let pin = store.pin(&[resident]).unwrap();
    store.evict_all().unwrap();
    assert!(store.corrections.find(&rewritten).is_none(), "written back");
    let id = store.corrections.find(&resident).unwrap();
    assert_eq!(store.corrections.is_dirty(id), Some(true), "dirty");
    store.prefetch.pause(false);
    assert_eq!(drain_prefetch(&mut store), 0, "both results were stale");

    assert_eq!(
        store.corrections.find(&resident),
        Some(id),
        "the edited tile kept its slot"
    );
    store.correction(rewritten).unwrap();
    assert_eq!(
        slot_resident(&store, &rewritten),
        slot_in_file(&store, &rewritten),
        "the edited tile, read again"
    );
    drop(pin);
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[rewritten, resident], "edited");
}

/// Dirties every resident correction tile of `store` with a new op, at an interior vertex of
/// each.
fn dirty_every_resident_correction(store: &mut TileStore, all: &[TileKey]) {
    store.begin_op().unwrap();
    let resident: Vec<TileKey> = all
        .iter()
        .filter(|key| store.corrections.find(key).is_some())
        .copied()
        .collect();
    for key in resident {
        store
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.001, 0.0, 0.0)))
            .unwrap()
            .unwrap();
    }
}

#[test]
fn prefetch_never_takes_an_evaluated_chunk_or_writes_a_tile_back() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = store_on_disk(&dir, "work.bcp", &mesh, config(10, 4), 300, 14);
    let all = tiles(&mesh, 1, TOP);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    let chunk = store.corrections.chunk_bytes();
    assert!(!store.budget.fits(chunk), "the budget is spent");
    // Every correction slot in use and dirty: none can be had without a write-back.
    let mut absent: Vec<TileKey> = all
        .iter()
        .filter(|key| !store.file.contains(key) && store.corrections.find(key).is_none())
        .copied()
        .collect();
    while store.corrections.has_free_slot() {
        store.correction(absent.pop().unwrap()).unwrap();
    }
    dirty_every_resident_correction(&mut store, &all);
    assert!(
        store.evaluated.has_unpinned_chunk(),
        "an evaluated chunk could be taken"
    );
    let on_disk: Vec<TileKey> = all
        .iter()
        .filter(|key| store.file.contains(key) && store.corrections.find(key).is_none())
        .copied()
        .collect();
    let (written, corrections, evaluated) = (
        store.file.len(),
        store.corrections.bytes(),
        store.evaluated.bytes(),
    );
    let dirty = store.unsaved.len();
    assert_eq!(store.prefetch(&on_disk[..1]).unwrap(), 1, "queued");
    assert_eq!(drain_prefetch(&mut store), 0, "dropped: no slot to spare");
    assert_eq!(store.file.len(), written, "nothing written back");
    assert_eq!(
        (store.corrections.bytes(), store.evaluated.bytes()),
        (corrections, evaluated),
        "no chunk changed hands"
    );
    assert_eq!(store.unsaved.len(), dirty, "every dirty tile stays");

    // Clean now: a prefetched tile takes a clean tile's slot, still without writing.
    store.save().unwrap();
    let saved = store.file.len();
    assert_eq!(store.prefetch(&on_disk[1..2]).unwrap(), 1, "queued");
    assert_eq!(drain_prefetch(&mut store), 1, "installed in a clean slot");
    assert_eq!(store.file.len(), saved, "nothing written");
    assert_eq!(
        (store.corrections.bytes(), store.evaluated.bytes()),
        (corrections, evaluated),
        "still no chunk changed hands"
    );
}

#[test]
fn dropping_a_store_with_queued_work_returns_promptly() {
    let (done, finished) = mpsc::channel();
    // The store isn't `Send`, so it lives and dies on a thread of its own, and a hang there
    // fails the test here instead of hanging it.
    thread::spawn(move || {
        let dir = TestDir::new();
        let mesh = mesh();
        for paused in [true, false] {
            let name = format!("paused-{paused}.bcp");
            let (mut store, _) = store_on_disk(&dir, &name, &mesh, roomy(), 300, 14);
            let on_disk: Vec<TileKey> = store.file.keys().copied().collect();
            store.prefetch.pause(paused);
            let queued = store.prefetch(&on_disk).unwrap();
            let workers = store.prefetch.liveness();
            let start = Instant::now();
            drop(store);
            let took = start.elapsed();
            assert!(
                workers.upgrade().is_none(),
                "paused {paused}: the workers were joined"
            );
            assert!(
                took < Duration::from_secs(1),
                "paused {paused}: dropping with {queued} reads queued took {took:?}"
            );
            let reopened = ProjectFile::open(&dir.path(&name));
            assert!(
                reopened.is_ok(),
                "paused {paused}: the file is free: {reopened:?}"
            );
        }
        // Removed before the test is told it is done: the test's process may exit at once,
        // ending this thread before its drops run.
        drop(dir);
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("the store dropped in time (a panic on its thread shows above)");
}

/// Most tiles the budget workload holds pinned at once.
const WORKLOAD_PINS: usize = 3;

/// Asserts what holds after every step of [`random_workload`]: the pools are within the budget,
/// which counts exactly their chunks, and the dependency graph is whole.
fn check_workload_step(store: &TileStore, config: StoreConfig, what: &str) {
    assert!(
        store.bytes() <= pools(&config),
        "{what}: {} bytes in a budget of {}",
        store.bytes(),
        pools(&config)
    );
    assert_eq!(
        store.bytes(),
        store.corrections.bytes() + store.evaluated.bytes(),
        "{what}: the budget counts the pools' chunks"
    );
    if let Err(broken) = store.deps.check_invariants() {
        panic!("{what}: dependency graph: {broken}");
    }
}

/// Runs `steps` random store operations, chosen with `seed`, under a budget of a few chunks.
///
/// The operations: evaluations, normals, edits (single vertices and tile batches), pins and unpins
/// (at most [`WORKLOAD_PINS`] tiles at once), prefetches and polls, saves, evictions and shrinks.
/// After every operation the pools are within the budget, the budget counts exactly their chunks
/// and the dependency graph is whole; at the end every tile of levels 1–8 evaluates as the dense
/// evaluation of the same edits does, bit for bit.
fn random_workload(seed: u64, steps: u32) {
    let config = pinning();
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let mut map = CorrectionMap::new();
    let all = tiles(&mesh, 1, TOP);
    let mut edits = random_edits(&mesh, TOP, steps, seed.wrapping_add(1)).into_iter();
    let mut rng = Rng(seed);
    let mut guards: Vec<PinGuard> = Vec::new();
    // How often the rarer paths ran, so a workload that never reaches them fails.
    let mut pin_sets = 0;
    let pick = |from: &mut Rng| all[from.below(u32::try_from(all.len()).unwrap()) as usize];
    for step in 0..steps {
        let what = match rng.below(100) {
            0..30 => {
                store.evaluate(pick(&mut rng)).unwrap();
                "evaluate"
            }
            30..40 => {
                store.normals(pick(&mut rng)).unwrap();
                "normals"
            }
            40..52 => {
                apply(&mut store, &mut map, &[edits.next().unwrap()]);
                "set_correction"
            }
            52..62 => {
                let batch = tile_batches(&mesh, 1, 1 + rng.below(8), rng.next_u64());
                apply_batches(&mut store, &mut map, &batch);
                "edit_tile"
            }
            62..70 => {
                let pinned: usize = guards.iter().map(|g| g.keys().len()).sum();
                let keys: Vec<TileKey> = (0..=rng.below(2)).map(|_| pick(&mut rng)).collect();
                if pinned + keys.len() <= WORKLOAD_PINS {
                    match store.pin(&keys) {
                        Ok(guard) => {
                            guards.push(guard);
                            pin_sets += 1;
                        }
                        Err(CacheError::OverBudget { .. }) => {}
                        Err(e) => panic!("step {step}: pin {keys:?}: {e:?}"),
                    }
                }
                "pin"
            }
            70..76 => {
                if !guards.is_empty() {
                    let n = rng.below(u32::try_from(guards.len()).unwrap()) as usize;
                    drop(guards.swap_remove(n));
                }
                "unpin"
            }
            76..88 => {
                let keys: Vec<TileKey> = (0..=rng.below(8)).map(|_| pick(&mut rng)).collect();
                store.prefetch(&keys).unwrap();
                store.poll().unwrap();
                "prefetch"
            }
            88..93 => {
                store.save().unwrap();
                "save"
            }
            93..96 => {
                store.evict_all().unwrap();
                "evict_all"
            }
            _ => {
                let budget = u32::try_from(pools(&config)).unwrap();
                store.shrink_to(rng.below(budget + 1) as usize).unwrap();
                "shrink_to"
            }
        };
        check_workload_step(&store, config, &format!("seed {seed} step {step} ({what})"));
    }
    drop(guards);
    drain_prefetch(&mut store);
    assert!(pin_sets > 10, "seed {seed}: only {pin_sets} pin sets held");
    // Installs happen at whichever call polls first, so they are counted by the store.
    assert!(
        store.log.prefetched > 10,
        "seed {seed}: only {} prefetched tiles installed",
        store.log.prefetched
    );
    assert!(
        store.log.positions.len() > 2 * all.len(),
        "seed {seed}: evicting, so tiles were evaluated again"
    );
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, &format!("seed {seed}"));
    let some: Vec<TileKey> = all.iter().step_by(7).copied().collect();
    assert_normals(&mut store, &mesh, &dense, &some, &format!("seed {seed}"));
}

#[test]
fn random_workloads_respect_the_budget() {
    // Seed 1 evicts a parent between its child's prediction and the child's slot: the window
    // where recording the child after unpinning its parents lost the parent's ancestry.
    for seed in [0x00B0_D6E7, 1] {
        random_workload(seed, 2_000);
    }
}

#[test]
#[ignore = "slow: more seeds of random_workloads_respect_the_budget"]
fn random_workloads_respect_the_budget_for_many_seeds() {
    for seed in 1..=16 {
        random_workload(seed, 2_000);
    }
}

/// A row read through the pinned tiles gives every vertex what reading it alone gives.
///
/// Runs cross tile borders (whose columns belong to the next tile) and reach the face's far
/// edge (whose column belongs to the last tile).
#[test]
fn resident_rows_read_as_single_vertices() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 100, 5));
    let top = level(TOP);
    let held = store
        .pin_current(tiles(&mesh, TOP, TOP).into_iter().collect())
        .unwrap();
    let n = top.face_quads();
    let tq = top.tile_quads();
    // Each tile's copy of the column it shares with the next tile along its row holds a value
    // no read may return: `TileKey::containing` gives that column to the next tile.
    let side = usize::try_from(tq + 1).unwrap();
    let planted = Vec3::new(1e9, -1e9, 1e9);
    for &(key, id) in &held {
        if u32::from(key.tx()) + 1 < top.tiles_per_side() {
            let slot = store.evaluated.get_mut(id).unwrap();
            for row in slot.positions[..side * side].chunks_mut(side) {
                row[side - 1] = planted;
            }
        }
    }
    let positions: Positions<'_> = held
        .iter()
        .map(|&(key, id)| (key, store.evaluated.get(id).unwrap().positions.as_slice()))
        .collect();
    let split = Resident::new(&store.mesh, Tiles::Split(&positions));
    let shared = Resident::new(
        &store.mesh,
        Tiles::Shared {
            pool: &store.evaluated,
            held: &held,
        },
    );
    let runs = [
        (0, n + 1),
        (1, tq - 1),
        (tq - 3, 6),
        (tq - 1, 2),
        (tq, 1),
        (n - 3, 4),
    ];
    for source in [&split, &shared] {
        for face in mesh.faces() {
            for j in [0, 1, tq - 1, tq, tq + 1, n / 2, n] {
                for (i, len) in runs {
                    let mut row = vec![
                        Vec3::new(f32::NAN, f32::NAN, f32::NAN);
                        usize::try_from(len).unwrap()
                    ];
                    source.row(face, top, i, j, &mut row).unwrap();
                    for (x, got) in (i..).zip(&row) {
                        let want = source.position(face, top, x, j).unwrap();
                        assert_ne!(want, planted, "a single read takes the next tile's copy");
                        assert_eq!(got.to_bits(), want.to_bits(), "{face:?} ({x}, {j})");
                        let other = shared.position(face, top, x, j).unwrap();
                        assert_eq!(other.to_bits(), want.to_bits(), "split and shared agree");
                    }
                }
            }
        }
        let mut past = [Vec3::ZERO; 2];
        assert!(
            source.row(FaceId::new(0), top, n, 0, &mut past).is_none(),
            "a run off the face isn't available"
        );
    }
    store.unpin_held(&held).unwrap();
}

/// A save point whose derived maps disagree with it is reported as a broken invariant.
///
/// It is a bug, which trying again can't fix, so it is never a save to retry, whichever
/// operation wrote the save point. Other file errors keep the operation's own variant.
#[test]
fn a_save_point_mismatch_is_an_internal_error_not_a_retryable_save() {
    let mismatch = || StoreError::SavePointMismatch {
        named: 2,
        derived: 1,
    };
    for wrap in [CacheError::Save, CacheError::WriteBack, CacheError::Store] {
        let routed = CacheError::from_save_point(mismatch(), wrap);
        assert!(
            matches!(
                routed,
                CacheError::Internal(Invariant::SavePointMismatch {
                    named: 2,
                    derived: 1
                })
            ),
            "{routed:?}"
        );
    }
    assert!(
        matches!(
            CacheError::from_save_point(StoreError::ReadOnly, CacheError::Save),
            CacheError::Save(StoreError::ReadOnly)
        ),
        "an I/O-class error stays a save to retry"
    );
}

/// Looking for a tile an evaluation didn't pin names the tile, not a stale handle.
#[test]
fn a_tile_not_held_is_named_as_such() {
    let key = tile_at(FaceId::new(0), 0, 0);
    let held: Held = Vec::new();
    assert!(
        matches!(held_index(&held, &key), Err(Invariant::NotHeld(k)) if k == key),
        "names the tile"
    );
}

/// A row the interior pass shouldn't reach is reported as the slot's own error.
///
/// The cause is kept: a row on the border, or past the tile.
#[test]
fn an_interior_row_outside_the_interior_keeps_its_cause() {
    let side = 9;
    let mut slot = CorrectionSlot::from_tile(&CorrectionTile::zero(side).unwrap());
    let mut view = slot.view_mut().unwrap();
    let mut f = |_, _, c: Vec3| Some(c + Vec3::new(0.0, 0.001, 0.0));
    for j in [0, side - 1, side] {
        let (wrote, row) = update_row(&mut view, j, &mut f);
        assert!(!wrote, "row {j}: nothing written");
        assert!(
            matches!(
                row,
                Err(CacheError::Internal(Invariant::CorrectionSlot(
                    CorrectionError::OutOfTile { .. }
                )))
            ),
            "row {j}: {row:?}"
        );
    }
    let (wrote, row) = update_row(&mut view, 1, &mut f);
    assert!(wrote && row.is_ok(), "an interior row is written: {row:?}");
}

/// A tile whose parents the dependency graph doesn't record counts as moved by a coarser edit.
///
/// Such a tile is an untracked node (see [`crate::deps`]): its kept prediction goes, and it is
/// predicted again from its parents' new positions.
///
/// The store records every tile's parents before they can leave the graph, so the test loses
/// the graph by hand: it is replaced by one where the brushed tile is known only as the parent
/// of a finer tile.
#[test]
fn a_tile_with_unrecorded_parents_loses_its_kept_prediction() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (keys, around) = brushed_tiles(&mesh);
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    brush_rounds(&mut store, &keys, &around, true);
    let key = keys[0];
    assert!(store.kept.get(&key).is_some(), "brushed: a prediction kept");
    let pins = store.pin(&[key]).unwrap();
    let parent = *store.deps.parents(&key).unwrap().first().unwrap();
    let child =
        TileKey::displacement(key.face(), level(TOP + 1), 2 * key.tx(), 2 * key.ty()).unwrap();
    store.deps = DepGraph::new();
    store.deps.record_positions(child, &[key]);
    assert!(
        store.deps.parents(&key).is_none() && store.deps.has_untracked(),
        "the tile is untracked"
    );
    store.begin_op().unwrap();
    store
        .edit_tile(parent, |edit| {
            edit.update(|_, _, c| Some(Vec3::new(c.x + 0.004, c.y, c.z - 0.003)))
        })
        .unwrap()
        .unwrap();
    assert!(store.kept.get(&key).is_none(), "its kept prediction went");
    let again: Vec<u32> = store
        .evaluate(key)
        .unwrap()
        .iter()
        .flat_map(|p| p.to_bits())
        .collect();
    drop(pins);
    store.evict_all().unwrap();
    let fresh: Vec<u32> = store
        .evaluate(key)
        .unwrap()
        .iter()
        .flat_map(|p| p.to_bits())
        .collect();
    assert!(again == fresh, "predicted from the parent's new positions");
}

/// The kept predictions' share costs only its own bytes.
///
/// The evaluated pool fills the room it leaves below a whole chunk with a short chunk, so the
/// pools reach the budget less the share, not a whole chunk less.
#[test]
fn the_kept_share_costs_only_its_bytes() {
    let dir = TestDir::new();
    let mesh = mesh();
    let chunk = 16 * EVAL;
    // Three kept predictions (617,904 bytes), and no stroke state, leave room for 9 of a
    // chunk's 16 evaluated slots.
    let config = StoreConfig::new(12 * chunk, SITE)
        .with_chunk_bytes(chunk)
        .with_workers(TEST_THREADS)
        .with_kept_predictions(3)
        .with_stroke_tiles(0);
    let room = 12 * chunk - config.kept_prediction_bytes().unwrap() - 11 * chunk;
    assert_eq!(room / EVAL, 9, "the set-up: room for nine slots");
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    // Batches count the short chunk's slots: ten whole chunks besides the correction floor,
    // and nine slots.
    let reach = 10 * 16 + 9 - store.headroom.evaluated;
    assert_eq!(store.batch_size(0), reach / BATCH_SHARE, "batches reach it");
    for key in tiles(&mesh, 1, TOP) {
        store.evaluate(key).unwrap();
    }
    assert_eq!(
        store.bytes(),
        11 * chunk + 9 * EVAL,
        "every whole chunk and a short one of nine slots"
    );
    assert!(store.bytes() <= pools(&config), "within the pools' share");
}

/// With no head of its site, a store's ops start at sequence 0, whatever other sites' heads.
#[test]
fn ops_start_at_zero_without_a_head_of_the_site() {
    let zero = OpId::new(SITE, 0).unwrap();
    assert_eq!(first_op(SITE, []).unwrap(), zero, "no heads");
    assert_eq!(first_op(SITE, [None, None]).unwrap(), zero, "no edits");
    let others = [OpId::new(SITE + 1, 900), OpId::new(SITE - 1, 5)];
    assert_eq!(first_op(SITE, others).unwrap(), zero, "other sites");
}

/// Ops start past the newest head of the site, ignoring other sites' heads, however much newer.
#[test]
fn ops_start_past_the_newest_head_of_the_site_only() {
    let heads = [
        OpId::new(SITE, 7),
        None,
        OpId::new(SITE + 1, MAX_SEQ),
        OpId::new(SITE, 41),
        OpId::new(SITE, 3),
    ];
    let next = OpId::new(SITE, 42).unwrap();
    assert_eq!(first_op(SITE, heads).unwrap(), next, "past 41");
}

/// A head at the last sequence of the site leaves no op to start at.
#[test]
fn a_head_at_the_last_op_of_the_site_exhausts_its_ops() {
    let heads = [OpId::new(SITE, 1), OpId::new(SITE, MAX_SEQ)];
    assert!(
        matches!(first_op(SITE, heads), Err(CacheError::OpsExhausted)),
        "the last sequence"
    );
    let below = [OpId::new(SITE, MAX_SEQ - 1)];
    let last = OpId::new(SITE, MAX_SEQ).unwrap();
    assert_eq!(
        first_op(SITE, below).unwrap(),
        last,
        "the last op is still given"
    );
}

/// A correction no tile's scale can hold is the caller's error, `TooLarge`, by every way the
/// store writes one, and nothing is written.
#[test]
fn a_correction_too_large_to_store_is_the_caller_s_error() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let key = tiles(&mesh, TOP, TOP)[0];
    let region = Region::of_tile(&key);
    // Finite, but past the largest interior value any scale decodes.
    let huge = Vec3::new(f32::MAX, 0.0, 0.0);
    let at = |i: u32, j: u32| ((i, j) == (5, 5)).then_some(huge);
    let too_large = |got: &Result<(), CacheError>| {
        matches!(got, Err(CacheError::Correction(CorrectionError::TooLarge)))
    };
    let (x, y) = (region.x0() + 5, region.y0() + 5);
    let one = store.set_correction(key.face(), key.level(), x, y, huge);
    assert!(too_large(&one), "set_correction: {one:?}");
    let edited = store.edit_tile(key, |edit| edit.set(5, 5, huge)).unwrap();
    assert!(too_large(&edited), "TileEdit::set: {edited:?}");
    let updated = store
        .edit_tile(key, |edit| edit.update(|i, j, _| at(i, j)))
        .unwrap();
    assert!(too_large(&updated), "TileEdit::update: {updated:?}");
    let guard = store.pin(&[key]).unwrap();
    let in_parallel = store.update_tiles(&[key], |_, i, j, _| at(i, j));
    assert!(too_large(&in_parallel), "update_tiles: {in_parallel:?}");
    drop(guard);
    assert!(store.unsaved.is_empty(), "nothing was written");
}
