//! Evaluation rounds on the workers and the kept predictions: the same bits at any thread
//! count, predicted in place or into scratch, and what a failing round leaves.

use super::*;

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
            store.hooks.log.in_place.len(),
            4 * keys.len(),
            "{threads} threads: every pinned tile predicted in place each round"
        );
        // The cold evaluation before the first round goes through scratch: look after it.
        let in_place: BTreeSet<TileKey> = store.hooks.log.in_place.iter().copied().collect();
        let first = store.hooks.log.in_place.first().unwrap();
        let rounds = store
            .hooks
            .log
            .positions
            .iter()
            .skip_while(|key| *key != first);
        assert!(
            rounds.filter(|key| !in_place.contains(key)).count() >= 4,
            "{threads} threads: unpinned tiles went through scratch in the rounds"
        );
        let feed = store.feed.drain(store.save_consumer);
        runs.push((
            threads,
            every_slot(&store, &all),
            feed,
            store.hooks.log.positions,
        ));
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
        store.hooks.log.predictions_held <= kept,
        "never more held than the bound, new ones included: {}",
        store.hooks.log.predictions_held
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
    let computed = store.hooks.log.positions.len();

    let failed = store.evaluate_many(&keys, Evaluation::Positions);
    assert!(matches!(failed, Err(CacheError::Internal(_))), "{failed:?}");
    assert_eq!(
        store.hooks.log.positions.len(),
        computed,
        "nothing installed"
    );
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
    store.hooks.failing_normals = Some(stale[stale.len() / 2]);
    let before: Vec<Option<Vec<u8>>> = around.iter().map(|k| evaluated_bytes(&store, k)).collect();
    let computed = store.hooks.log.normals.len();

    let failed = store.evaluate_many(&around, Evaluation::Normals);
    assert!(matches!(failed, Err(CacheError::Internal(_))), "{failed:?}");
    assert_eq!(store.hooks.log.normals.len(), computed, "nothing recorded");
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

    store.hooks.failing_normals = None;
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
        store.hooks.log.in_place.is_empty(),
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
    assert!(!in_place.hooks.log.in_place.is_empty(), "pinned: in place");
    assert!(
        scratch.hooks.log.in_place.is_empty(),
        "not pinned: into scratch"
    );
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
