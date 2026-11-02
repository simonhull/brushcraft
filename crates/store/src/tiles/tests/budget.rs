//! The memory budget: pins, the pools and the chunks moving between them, write-backs, the
//! shares set aside, and the requests refused for want of room.

use super::*;

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
    store.hooks.log = EvalLog::default();
    assert_positions(&mut store, &dense, &tiles(&mesh, 1, TOP), "the other tiles");
    assert!(
        store.hooks.log.positions.len() > 100,
        "under pressure: {} evaluations",
        store.hooks.log.positions.len()
    );
    for (key, id) in pinned.iter().zip(&ids) {
        assert_eq!(
            store.evaluated.find(key),
            Some(*id),
            "{key:?} stayed in its slot"
        );
        assert!(
            !store.hooks.log.positions.contains(key),
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
    store.hooks.unreadable = Some(on_disk);
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
    store.hooks.unreadable = Some(wanted[1]);
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
                asked: StoreConfig::DEFAULT_KEPT_PREDICTIONS,
                most
            }) if most == StoreConfig::DEFAULT_KEPT_PREDICTIONS - 1
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

/// The default kept predictions are 256 of the largest tile's, about 50.3 MiB of the budget.
#[test]
fn the_default_kept_predictions_share_is_256_tiles() {
    let config = roomy();
    assert_eq!(
        config.kept_predictions,
        StoreConfig::DEFAULT_KEPT_PREDICTIONS,
        "the default"
    );
    assert_eq!(
        StoreConfig::DEFAULT_KEPT_PREDICTIONS,
        256,
        "256 predictions"
    );
    assert_eq!(
        config.kept_prediction_bytes(),
        Some(256 * Prediction::TILE_BYTES),
        "each the largest tile's"
    );
    assert_eq!(
        config.kept_prediction_bytes(),
        Some(52_727_808),
        "about 50.3 MiB"
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
        50_700,
        "three f32 a vertex of the largest tile, one for each correction component"
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
    assert_eq!(
        share, 25_958_400,
        "512 tiles of 50,700 bytes: about 24.75 MiB"
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
        matches!(evicted, Err(CacheError::WriteBack(FileError::ReadOnly))),
        "evicting a dirty tile reports the write: {evicted:?}"
    );
    let saving = store.save();
    assert!(
        matches!(saving, Err(CacheError::WriteBack(FileError::ReadOnly))),
        "and so does saving it: {saving:?}"
    );
    for key in &dirty {
        let id = store.corrections.find(key).unwrap();
        assert_eq!(store.corrections.is_dirty(id), Some(true), "{key:?} kept");
    }
}

/// A save point whose derived maps disagree with it is reported as a broken invariant.
///
/// It is a bug, which trying again can't fix, so it is never a save to retry, whichever
/// operation wrote the save point. Other file errors keep the operation's own variant.
#[test]
fn a_save_point_mismatch_is_an_internal_error_not_a_retryable_save() {
    let mismatch = || FileError::SavePointMismatch {
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
            CacheError::from_save_point(FileError::ReadOnly, CacheError::Save),
            CacheError::Save(FileError::ReadOnly)
        ),
        "an I/O-class error stays a save to retry"
    );
}
