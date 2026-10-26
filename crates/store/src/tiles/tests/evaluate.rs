//! The store's evaluation against the dense oracle: positions and normals, at any thread
//! count and batch size, in any order, from sparse files and after a reopen.

use super::*;

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
