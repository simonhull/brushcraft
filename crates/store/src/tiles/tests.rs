//! The tile store against `geom`'s dense evaluation, the oracle: every tile bit for bit, with
//! room to spare and under constant eviction, across edits, saves and pins.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use geom::{
    BaseMesh, CorrectionMap, CorrectionTile, DenseEvaluation, FaceId, Level, Region, TileKey, Vec3,
    cube_sphere, normal_tiles, normals,
};

use super::*;
use crate::fixture::meta_for;
use crate::test_dir::TestDir;

mod differential;
mod layered;
mod recovery;

/// Bytes of one evaluated slot: test chunks are sized in these.
const EVAL: usize = size_of::<EvaluatedSlot>();

/// The finest level the tests evaluate.
pub(crate) const TOP: u8 = 8;

/// The site the tests' ops carry.
const SITE: u16 = 3;

/// Deterministic random numbers (splitmix64).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`; `n > 0`.
    fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next_u64() % u64::from(n)).unwrap()
    }

    /// Uniform in `[-amplitude, amplitude)`, per component.
    fn vec3(&mut self, amplitude: f32) -> Vec3 {
        let mut signed =
            || f32::from(u16::try_from(self.next_u64() >> 48).unwrap()) / 32768.0 - 1.0;
        Vec3::new(
            signed() * amplitude,
            signed() * amplitude,
            signed() * amplitude,
        )
    }
}

/// One `set_correction` call.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edit {
    face: FaceId,
    level: Level,
    i: u32,
    j: u32,
    c: Vec3,
}

fn level(n: u8) -> Level {
    Level::new(n).unwrap()
}

/// The benchmark's sphere at its smallest: a cube, 6 faces.
pub(crate) fn mesh() -> Arc<BaseMesh> {
    Arc::new(cube_sphere(1).unwrap())
}

/// A configuration with chunks of `chunk_slots` evaluated slots and a budget of `chunks` of them.
fn config(chunks: usize, chunk_slots: usize) -> StoreConfig {
    StoreConfig {
        budget_bytes: chunks * chunk_slots * EVAL,
        chunk_bytes: chunk_slots * EVAL,
        site: SITE,
        threads: None,
    }
}

/// Room for every tile of levels 1–8 of the cube, twice over.
pub(crate) fn roomy() -> StoreConfig {
    config(40, 16)
}

/// Six chunks of four evaluated slots (or fifteen correction slots).
///
/// About a sixth of the cube's 156 tiles fit, so evaluation evicts constantly and dirty tiles
/// are written back. It is the cube's minimum budget rounded up to whole chunks: an evaluation
/// pins at most two cone levels of nine tiles, which take five evaluated chunks, and one
/// correction chunk goes beside them.
pub(crate) fn tight() -> StoreConfig {
    config(6, 4)
}

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

/// Every tile of levels `from..=to`, coarse levels first.
pub(crate) fn tiles(mesh: &BaseMesh, from: u8, to: u8) -> Vec<TileKey> {
    (from..=to)
        .flat_map(|l| {
            let level = level(l);
            let per_side = u16::try_from(level.tiles_per_side()).unwrap();
            mesh.faces().flat_map(move |face| {
                (0..per_side).flat_map(move |ty| {
                    (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).unwrap())
                })
            })
        })
        .collect()
}

/// `count` random edits at levels `1..=top`, half of them on face or tile borders, where twins
/// must stay equal.
pub(crate) fn random_edits(mesh: &BaseMesh, top: u8, count: u32, seed: u64) -> Vec<Edit> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let level = level(1 + u8::try_from(rng.below(u32::from(top))).unwrap());
            let (n, tq) = (level.face_quads(), level.tile_quads());
            let mut pick = || {
                if rng.below(2) == 0 {
                    let border = rng.below(level.tiles_per_side() + 1) * tq;
                    (border + rng.below(3)).saturating_sub(1).min(n)
                } else {
                    rng.below(n + 1)
                }
            };
            let (i, j) = (pick(), pick());
            let amplitude = 0.5 / f32::from(u16::try_from(n).unwrap());
            Edit {
                face,
                level,
                i,
                j,
                c: rng.vec3(amplitude),
            }
        })
        .collect()
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

/// Breaks `key`'s resident correction slot: it no longer holds a valid tile, so it neither
/// encodes nor saves.
pub(crate) fn break_slot(store: &mut TileStore, key: &TileKey) {
    let id = store.corrections.find(key).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
}

/// Flips a bit in the middle of the record at `loc` of the file at `path`, from outside, as a
/// bad sector would: the record fails its checksum.
pub(crate) fn damage(path: &std::path::Path, loc: Loc) {
    use std::os::unix::fs::FileExt;
    let mut bytes = std::fs::read(path).unwrap();
    let at = usize::try_from(loc.middle()).unwrap();
    bytes[at] ^= 0x40;
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .write_all_at(&bytes[at..=at], loc.middle())
        .unwrap();
}

/// Applies `edits` to the oracle's map alone: what a store over `mesh` that made them holds.
pub(crate) fn record(mesh: &BaseMesh, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        map.set(mesh.topology(), e.face, e.level, e.i, e.j, e.c)
            .unwrap();
    }
}

/// The oracle: every level up to [`TOP`], evaluated densely.
pub(crate) fn dense(mesh: &BaseMesh, map: &CorrectionMap) -> DenseEvaluation {
    DenseEvaluation::new(mesh, level(TOP), map).unwrap()
}

/// Asserts that `got`, tile `key`'s values, equal `want`'s at the same face-grid vertices, bit
/// for bit.
fn assert_tile(got: &[Vec3], want: impl Fn(u32, u32) -> Vec3, key: &TileKey, what: &str) {
    let region = Region::of_tile(key);
    let side = key.side();
    assert_eq!(got.len(), (side * side) as usize, "{what} {key:?}: length");
    for j in 0..side {
        for i in 0..side {
            let have = got[(j * side + i) as usize];
            assert_eq!(
                have.to_bits(),
                want(region.x0() + i, region.y0() + j).to_bits(),
                "{what} {key:?} ({i}, {j})"
            );
        }
    }
}

/// Asserts that the store evaluates every tile of `keys` as the dense evaluation does.
pub(crate) fn assert_positions(
    store: &mut TileStore,
    dense: &DenseEvaluation,
    keys: &[TileKey],
    what: &str,
) {
    for key in keys {
        let got = store.evaluate(*key).unwrap();
        let grid = dense.grid(key.face(), key.level()).unwrap();
        assert_tile(got, |i, j| grid.get(i, j).unwrap(), key, what);
    }
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
        store.bytes() <= config.budget_bytes,
        "{} bytes in a budget of {}",
        store.bytes(),
        config.budget_bytes
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
        let config = StoreConfig {
            threads: NonZeroUsize::new(n),
            ..base
        };
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
        let config = StoreConfig {
            threads: NonZeroUsize::new(threads),
            ..config(80, 16)
        };
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
        store.bytes() + store.corrections.chunk_bytes() > config.budget_bytes,
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
    assert!(store.bytes() <= config.budget_bytes, "within the budget");
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
        config.budget_bytes,
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
    let reserved = StoreConfig {
        site: RESERVED_SITE,
        ..roomy()
    };
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
    let file =
        ProjectFile::create(&dir.path("small.bcp"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let small = StoreConfig {
        budget_bytes: minimum - 1,
        ..tight()
    };
    let refused = TileStore::open(file, small);
    assert!(
        matches!(refused, Err(CacheError::BudgetTooSmall { minimum: m }) if m == minimum),
        "refused: {refused:?}"
    );
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

/// The bytes of `key`'s latest record in the working file, decoded into a slot.
fn slot_in_file(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let (_, tile) = store.file.read_displacement(key).unwrap().unwrap();
    bytemuck::bytes_of(&CorrectionSlot::from_tile(&tile)).to_vec()
}

/// The bytes of `key`'s resident correction slot.
fn slot_resident(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let id = store.corrections.find(key).unwrap();
    bytemuck::bytes_of(store.corrections.get(id).unwrap()).to_vec()
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
        store.bytes() <= config.budget_bytes,
        "{what}: {} bytes in a budget of {}",
        store.bytes(),
        config.budget_bytes
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
                let budget = u32::try_from(config.budget_bytes).unwrap();
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
