//! What an edit invalidates: exactly the tiles it reaches, at every level, once between
//! evaluations, with stale normals brought up to date.

use super::*;

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
    store.hooks.log = EvalLog::default();

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
        store.hooks.log.positions,
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
    let recomputed: BTreeSet<TileKey> = store.hooks.log.normals.iter().copied().collect();
    assert_eq!(
        recomputed, readers,
        "exactly the normals that read the edited tile were dropped"
    );
    assert_eq!(
        store.hooks.log.normals.len(),
        readers.len(),
        "each recomputed once"
    );
    let mut neighbours = readers.clone();
    neighbours.remove(&edited);
    assert_eq!(
        store
            .hooks
            .log
            .refreshed
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        neighbours,
        "the neighbours' normals, whose own positions stayed, were brought up to date; the \
         edited tile's were computed in full"
    );
    assert_eq!(
        (
            store.hooks.log.normal_inputs_reused.get(),
            store.hooks.log.normal_inputs_listed.get()
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
            store.hooks.log.refreshed.len() > 2 * dab_count,
            "{threads} threads: neighbours were brought up to date, not recomputed: {}",
            store.hooks.log.refreshed.len()
        );
        runs.push((
            store.hooks.log.normals.clone(),
            store.hooks.log.refreshed.clone(),
        ));
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
    store.hooks.log = EvalLog::default();
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[gone], "rebuilt from levels 3 up");
    assert_normals(&mut store, &mesh, &dense, &[gone], "rebuilt");
    assert!(
        store.hooks.log.positions.len() > 4,
        "levels 4–8 were evaluated again: {:?}",
        store.hooks.log.positions
    );
    assert_eq!(
        store.hooks.log.parents_listed.get(),
        0,
        "every parent list was read back from the graph"
    );
    assert!(
        store.hooks.log.parents_reused.get() >= store.hooks.log.positions.len(),
        "one per tile built at least: {}",
        store.hooks.log.parents_reused.get()
    );
    assert_eq!(
        (
            store.hooks.log.normal_inputs_reused.get(),
            store.hooks.log.normal_inputs_listed.get()
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
    store.hooks.log = EvalLog::default();
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
        store
            .hooks
            .log
            .positions
            .iter()
            .any(|k| k.level() == level(TOP)),
        "level-8 tiles were recomputed"
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
    store.hooks.log = EvalLog::default();
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
        store.hooks.log.invalidations, 1,
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
    assert_eq!(
        store.hooks.log.invalidations, 2,
        "evaluated since: dropped again"
    );
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
    assert_eq!(
        store.hooks.log.invalidations, 3,
        "a whole tile edit drops once"
    );
}
