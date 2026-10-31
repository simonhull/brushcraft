//! Display records read ahead for drawing ([`TileStore::read_display_ahead`]), and the floor
//! drawn from memory after an open.

use super::display::saved_layer;
use super::*;
use crate::bounds::Kept;

/// The display tiles of the tests' mesh at `levels`, in key order.
fn display_at(mesh: &BaseMesh, levels: std::ops::RangeInclusive<u8>) -> Vec<TileKey> {
    let (first, last) = levels.into_inner();
    let mut keys: Vec<TileKey> = tiles(mesh, first, last)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect();
    keys.sort_unstable();
    keys
}

/// The store over a saved layer, opened again: every display record above the floor is in the
/// file only.
fn reopened(dir: &TestDir, mesh: &Arc<BaseMesh>) -> TileStore {
    saved_layer(dir, "work.bcp", mesh, 81);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    TileStore::open(file, roomy().with_display(level(TOP))).unwrap()
}

/// Waits until the workers have answered every read asked of them, and takes what they read.
fn settle_reads(store: &mut TileStore) {
    let began = Instant::now();
    loop {
        store.take_read_ahead();
        if store.read_ahead_pending() == 0 {
            return;
        }
        assert!(began.elapsed() < Duration::from_secs(10), "reads answered");
        thread::yield_now();
    }
}

/// `key`'s display tile as the files hold it.
fn from_file(store: &TileStore, key: &TileKey) -> Option<geom::DisplayTile> {
    crate::tiles::display::read_display(&store.file, store.base.as_ref(), key)
        .unwrap()
        .and_then(|record| record.decode(key))
}

#[test]
fn records_read_ahead_are_drawn_without_a_read_and_forgotten_once_staged() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let keys: Vec<TileKey> = display_at(&mesh, 6..=7).into_iter().take(4).collect();
    for key in &keys {
        assert!(store.display_needs_read(key), "{key:?}: in the file only");
    }
    assert_eq!(
        store.read_display_ahead(&keys),
        keys.len(),
        "every one read"
    );
    assert_eq!(store.read_display_ahead(&keys), 0, "asked once");
    settle_reads(&mut store);
    for key in &keys {
        assert!(!store.display_needs_read(key), "{key:?}: read ahead");
    }
    let drawn = store.display_for_draw(&keys).unwrap();
    for (key, tile) in keys.iter().zip(drawn) {
        assert_eq!(tile, from_file(&store, key), "{key:?}: the file's record");
        assert!(
            !store.display_needs_read(key),
            "{key:?}: drawn but not staged, kept for the next frame"
        );
    }
    // The first two staged, the others starved (no room, no slot) this frame.
    let (staged, starved) = keys.split_at(2);
    store.forget_drawn(staged);
    for key in staged {
        assert!(
            !store.read_ahead.read.tiles.contains_key(key),
            "{key:?}: forgotten once staged"
        );
        assert!(
            store.display_needs_read(key),
            "{key:?}: read again if drawn again"
        );
    }
    for key in starved {
        assert!(!store.display_needs_read(key), "{key:?}: still read ahead");
    }
}

/// A stroke's records read around its dabs survive the same tiles being requested, drawn and
/// staged, and are still there for the stroke's next dabs; and they don't take the draw's
/// share.
#[test]
fn a_strokes_records_read_around_survive_being_requested_and_drawn() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let reached = TileKey::display(FaceId::new(0), level(TOP - 1), 0, 0).unwrap();
    store.read_around(&BTreeSet::from([reached]));
    settle_reads(&mut store);
    let around: Vec<TileKey> = store.read_ahead.read.tiles.keys().copied().collect();
    assert!(!around.is_empty(), "the tiles around it were read");
    assert_eq!(
        store.read_display_ahead(&around),
        0,
        "a stroke's records are used as they are, not read again"
    );
    let drawn = store.display_for_draw(&around).unwrap();
    for (key, tile) in around.iter().zip(drawn) {
        assert_eq!(tile, from_file(&store, key), "{key:?}: the file's record");
    }
    store.forget_drawn(&around);
    // The request moves on, then goes.
    store.read_display_ahead(&display_at(&mesh, 6..=6));
    store.read_display_ahead(&[]);
    settle_reads(&mut store);
    for key in &around {
        assert!(
            store.read_ahead.read.tiles.contains_key(key),
            "{key:?}: the stroke's record survives"
        );
    }
    // And a tile the book knows takes no share: the whole share is left for the others.
    let layer = display_layer(&mut store);
    let tile = layer
        .iter()
        .find(|tile| tile.key().level() > level(5) && !around.contains(tile.key()))
        .unwrap();
    let known = *tile.key();
    store.record_display(std::slice::from_ref(tile)).unwrap();
    let others: Vec<TileKey> = display_at(&mesh, 6..=7)
        .into_iter()
        .filter(|key| *key != known && !around.contains(key))
        .collect();
    let mut asked = vec![known];
    asked.extend(&others);
    assert_eq!(
        store.read_display_ahead(&asked),
        store.draw_ahead_cap().min(others.len()),
        "the share, all of it for tiles needing a read"
    );
}

#[test]
fn reads_for_drawing_stay_within_their_share_of_the_cache() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let keys = display_at(&mesh, 6..=7);
    let cap = store.draw_ahead_cap();
    assert!(keys.len() > cap, "more wanted than the share holds");
    assert_eq!(store.read_display_ahead(&keys), cap, "the share, no more");
    settle_reads(&mut store);
    let held: Vec<TileKey> = keys
        .iter()
        .filter(|key| store.read_ahead.read.tiles.contains_key(key))
        .copied()
        .collect();
    assert_eq!(held, keys[..cap], "the most wanted first");
    assert!(store.read_ahead.read.tiles.len() <= store.read_ahead.read.cap());
    // Asked again, nothing more is read: the share is full.
    assert_eq!(store.read_display_ahead(&keys), 0, "still full");
    // Keys that aren't the layer's display tiles are never read.
    let finest = TileKey::displacement(FaceId::new(0), level(TOP), 0, 0).unwrap();
    let past = TileKey::display(FaceId::new(999), level(6), 0, 0).unwrap();
    assert_eq!(store.read_display_ahead(&[finest, past]), 0, "not ours");
}

#[test]
fn a_read_no_longer_wanted_goes() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let keys = display_at(&mesh, 6..=7);
    let (first, second) = (&keys[..4], &keys[4..8]);
    store.read_display_ahead(first);
    settle_reads(&mut store);
    assert!(
        first
            .iter()
            .all(|key| store.read_ahead.read.tiles.contains_key(key)),
        "read"
    );
    // The request moves on: the first tiles' records go, the second's are read.
    store.read_display_ahead(second);
    assert!(
        first
            .iter()
            .all(|key| !store.read_ahead.read.tiles.contains_key(key)),
        "dropped with the request"
    );
    // And at once nothing is wanted: whether the reads were done or skipped, nothing is kept.
    store.read_display_ahead(&[]);
    settle_reads(&mut store);
    assert!(
        store.read_ahead.read.tiles.is_empty(),
        "nothing kept for a request gone: {:?}",
        store.read_ahead.read.tiles.keys()
    );
}

#[test]
fn the_floor_loaded_on_open_is_drawn_without_reading_the_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let floor = display_at(&mesh, 0..=5);
    for key in &floor {
        assert!(!store.display_needs_read(key), "{key:?}: in the floor");
    }
    assert_eq!(store.read_display_ahead(&floor), 0, "nothing to read ahead");
    let drawn = store.display_for_draw(&floor).unwrap();
    for (key, tile) in floor.iter().zip(&drawn) {
        assert_eq!(*tile, from_file(&store, key), "{key:?}: the file's record");
    }
    // Recorded again and saved: the files' latest records are new ones, and drawn.
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 82),
    );
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    let redrawn = store.display_for_draw(&floor).unwrap();
    for (key, tile) in floor.iter().zip(&redrawn) {
        let built = layer.iter().find(|built| built.key() == key).unwrap();
        assert_eq!(tile.as_ref(), Some(built), "{key:?}: the new record");
        assert_eq!(
            *tile,
            from_file(&store, key),
            "{key:?}: as the file holds it"
        );
    }
}

// Staleness: a record held in memory (the floor's copy, or one read ahead) is drawn only while
// it is the record the store would read. Each test below compares every draw with `expected`,
// which reads the files (or the book) directly, never the floor's copy or the read-ahead.

/// What drawing `key` must give, worked out without the floor's copy or a record read ahead.
///
/// That is a stroke's bump, the book's record (none if unknown), else the files' latest record;
/// and where none is known, the floor's copy as last recorded, its bound the largest there is
/// and its cone covering everything. Call it after a draw, which settles the writes.
fn expected(store: &TileStore, key: &TileKey) -> Option<geom::DisplayTile> {
    if let Some(Bumped { record, extra, .. }) = store.bumped.get(key) {
        let bound = (record.error + extra).next_up();
        let everything = geom::FinestNormalCone::EVERYTHING;
        return geom::DisplayTile::decode(*key, bound, everything, &record.encoded).ok();
    }
    let record = match store.display.kept(key) {
        Some(Kept::Known(record)) => Some(record.clone()),
        Some(Kept::Unknown) => None,
        Some(Kept::Written) | None => {
            crate::tiles::display::read_display(&store.file, store.base.as_ref(), key).unwrap()
        }
    };
    match record.filter(|record| Some(record.finest) == store.display_finest) {
        Some(record) => record.decode(key),
        None => store.floor.get(key).and_then(|copy| {
            let everything = geom::FinestNormalCone::EVERYTHING;
            geom::DisplayTile::decode(*key, f32::MAX, everything, &copy.encoded).ok()
        }),
    }
}

/// Draws `keys` and checks each is what the files (or the book) give (see [`expected`]).
fn assert_drawn_as_stored(store: &mut TileStore, keys: &[TileKey], when: &str) {
    let drawn = store.display_for_draw(keys).unwrap();
    for (key, tile) in keys.iter().zip(drawn) {
        assert_eq!(tile, expected(store, key), "{when}: {key:?}");
    }
}

/// Whether `tile` is drawn as a tile nothing is trusted of: the largest bound, the cone covering
/// everything.
fn untrusted(tile: &geom::DisplayTile) -> bool {
    tile.error().to_bits() == f32::MAX.to_bits() && tile.cone().covers_everything()
}

/// The store's own fresh recovery file at `name` in `dir`, for a roll-over.
fn fresh_like(store: &TileStore, dir: &TestDir, name: &str) -> ProjectFile {
    let meta = store.file.meta().clone();
    ProjectFile::create_like(&dir.path(name), meta, &store.file).unwrap()
}

/// Rolls `store` over onto a fresh working file `name` in `dir`.
fn roll_over(store: &mut TileStore, dir: &TestDir, name: &str) {
    let fresh = fresh_like(store, dir, name);
    let filled = store.fill_roll_over(fresh).unwrap();
    let (old, lost) = store.roll_over(filled).unwrap();
    drop(old);
    assert!(lost.is_empty(), "nothing lost");
}

/// A store over a fresh recovery file layered on a saved display layer (the project file
/// `work.bcp` in `dir`, also returned, to save into): the floor and every record in the base.
fn layered(dir: &TestDir, mesh: &Arc<BaseMesh>) -> (TileStore, ProjectFile) {
    saved_layer(dir, "work.bcp", mesh, 81);
    let project = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let base = project.base().unwrap();
    let role = crate::meta::Role::Recovery {
        base: Some(base.save_point()),
    };
    let meta = project.meta().with_role(role);
    let working = ProjectFile::create_like(&dir.path("r1.bcrf"), meta, &project).unwrap();
    let config = roomy().with_display(level(TOP));
    let store = TileStore::open_layered(working, Some(base), config).unwrap();
    (store, project)
}

/// Saves `store` into `project` and moves it onto a fresh recovery file `name` in `dir`, as a
/// project's save does.
fn save_and_rebase(store: &mut TileStore, project: &mut ProjectFile, dir: &TestDir, name: &str) {
    let saved = store.save_into(project).unwrap();
    let base = project.base().unwrap();
    let role = crate::meta::Role::Recovery {
        base: Some(base.save_point()),
    };
    let meta = project.meta().with_role(role);
    let fresh = ProjectFile::create_like(&dir.path(name), meta, project).unwrap();
    drop(store.rebase(saved, fresh).unwrap());
}

/// The first `count` display tiles above the floor, read ahead and landed.
fn read_ahead_some(store: &mut TileStore, mesh: &BaseMesh, count: usize) -> Vec<TileKey> {
    let keys: Vec<TileKey> = display_at(mesh, 6..=7).into_iter().take(count).collect();
    assert_eq!(store.read_display_ahead(&keys), keys.len(), "read ahead");
    settle_reads(store);
    for key in &keys {
        assert!(!store.display_needs_read(key), "{key:?}: read ahead");
    }
    keys
}

#[test]
fn a_tile_recorded_after_its_read_ahead_is_drawn_as_recorded() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let keys = read_ahead_some(&mut store, &mesh, 6);
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 91),
    );
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    let drawn = store.display_for_draw(&keys).unwrap();
    for (key, tile) in keys.iter().zip(drawn) {
        let built = layer.iter().find(|built| built.key() == key).unwrap();
        assert_eq!(tile.as_ref(), Some(built), "{key:?}: the new record");
    }
    assert_drawn_as_stored(&mut store, &keys, "recorded");
}

/// A read for drawing still in flight when the record moves (written through at a new place)
/// lands, and is never taken for the tile: the draw reads the new record.
#[test]
fn a_read_landing_after_its_record_moved_is_rejected() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    store.hold_display_waiting_to(0);
    let keys: Vec<TileKey> = display_at(&mesh, 6..=7).into_iter().take(6).collect();
    assert_eq!(store.read_display_ahead(&keys), keys.len(), "read ahead");
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 92),
    );
    let layer = display_layer(&mut store);
    // Written through at once: every record at a new place in the working file.
    store.record_display(&layer).unwrap();
    for key in &keys {
        assert!(
            matches!(store.display.kept(key), Some(Kept::Written)),
            "{key:?}: written through"
        );
    }
    settle_reads(&mut store);
    for key in &keys {
        assert!(
            store.read_ahead_record(key).is_none(),
            "{key:?}: the read landed for the old place, rejected"
        );
    }
    let drawn = store.display_for_draw(&keys).unwrap();
    for (key, tile) in keys.iter().zip(drawn) {
        let built = layer.iter().find(|built| built.key() == key).unwrap();
        assert_eq!(tile.as_ref(), Some(built), "{key:?}: the new record");
    }
    assert_drawn_as_stored(&mut store, &keys, "written through");
}

#[test]
fn a_tile_bumped_after_its_read_ahead_is_drawn_bumped() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    let face = FaceId::new(0);
    let mut keys = vec![TileKey::display(face, level(6), 0, 0).unwrap()];
    for (tx, ty) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        keys.push(TileKey::display(face, level(7), tx, ty).unwrap());
    }
    assert_eq!(store.read_display_ahead(&keys), keys.len(), "read ahead");
    settle_reads(&mut store);
    let lift = 0.002_f32;
    let (edited, li, lj) = TileKey::containing(
        LayerId::BASE,
        Channel::DISPLACEMENT,
        face,
        level(TOP),
        40,
        50,
    )
    .unwrap();
    store.begin_op().unwrap();
    let c = store.correction(edited).unwrap().get(li, lj).unwrap();
    store
        .set_correction(face, level(TOP), 40, 50, c + Vec3::new(0.0, 0.0, lift))
        .unwrap();
    store.bump_display_error(&[edited], lift).unwrap();
    let bumped: Vec<TileKey> = keys
        .iter()
        .filter(|key| store.bumped.contains_key(key))
        .copied()
        .collect();
    assert!(
        bumped.len() >= 2,
        "the dab reaches tiles read ahead: {bumped:?}"
    );
    let drawn = store.display_for_draw(&bumped).unwrap();
    for (key, tile) in bumped.iter().zip(drawn) {
        let tile = tile.unwrap();
        let Bumped { record, extra, .. } = &store.bumped[key];
        assert_eq!(
            tile.error(),
            (record.error + extra).next_up(),
            "{key:?}: the bumped bound"
        );
        assert!(tile.cone().covers_everything(), "{key:?}: the cone");
    }
    assert_drawn_as_stored(&mut store, &keys, "bumped");
}

/// Both caches full (records read ahead, the floor's copies), then some tiles recorded again,
/// then a roll-over: every draw is the files' latest, the changed tiles' cached copies unused.
#[test]
fn draws_after_a_roll_over_are_the_files_latest() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _project) = layered(&dir, &mesh);
    let ahead = read_ahead_some(&mut store, &mesh, 8);
    let floor = display_at(&mesh, 0..=5);
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 93),
    );
    let layer = display_layer(&mut store);
    // Half the layer recorded again; the rest the edits made unknown, or left as it was.
    let half: Vec<&geom::DisplayTile> = layer.iter().step_by(2).collect();
    store.record_display(&half).unwrap();
    roll_over(&mut store, &dir, "r2.bcrf");
    let mut keys = floor;
    keys.extend(&ahead);
    assert_drawn_as_stored(&mut store, &keys, "rolled over");
    for tile in &half {
        if keys.contains(tile.key()) {
            let drawn = store.display_for_draw(&[*tile.key()]).unwrap();
            assert_eq!(drawn[0].as_ref(), Some(*tile), "{:?}: recorded", tile.key());
        }
    }
    // And again once the store has flushed nothing more: still the files' latest.
    roll_over(&mut store, &dir, "r3.bcrf");
    assert_drawn_as_stored(&mut store, &keys, "rolled over twice");
}

/// The same across a project's save: the store moves onto the new save point and a fresh
/// recovery file.
#[test]
fn draws_after_a_save_are_the_files_latest() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut project) = layered(&dir, &mesh);
    let ahead = read_ahead_some(&mut store, &mesh, 8);
    let floor = display_at(&mesh, 0..=5);
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 94),
    );
    let layer = display_layer(&mut store);
    let half: Vec<&geom::DisplayTile> = layer.iter().step_by(2).collect();
    store.record_display(&half).unwrap();
    save_and_rebase(&mut store, &mut project, &dir, "r2.bcrf");
    let mut keys = floor;
    keys.extend(&ahead);
    assert_drawn_as_stored(&mut store, &keys, "saved");
    for tile in &half {
        if keys.contains(tile.key()) {
            let drawn = store.display_for_draw(&[*tile.key()]).unwrap();
            assert_eq!(drawn[0].as_ref(), Some(*tile), "{:?}: recorded", tile.key());
        }
    }
}

/// A refresh through the jobs records floor tiles again, written through at once.
///
/// The book says the working file holds them ([`Kept::Written`]), so a draw looks past it to the
/// floor's copy, and must find the new record, not the one loaded on open.
#[test]
fn a_floor_tile_refreshed_and_written_through_is_drawn_as_refreshed() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = reopened(&dir, &mesh);
    store.hold_display_waiting_to(0);
    let floor = display_at(&mesh, 0..=5);
    let before = store.display_for_draw(&floor).unwrap();
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 95),
    );
    loop {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
        if !store.has_jobs() {
            break;
        }
    }
    let refreshed: Vec<TileKey> = floor
        .iter()
        .filter(|key| matches!(store.display.kept(key), Some(Kept::Written)))
        .copied()
        .collect();
    assert!(
        !refreshed.is_empty(),
        "floor tiles refreshed, written through"
    );
    for key in &refreshed {
        assert!(
            !store.floor_read.contains_key(key),
            "{key:?}: no longer named as the record loaded on open"
        );
    }
    assert_drawn_as_stored(&mut store, &floor, "refreshed");
    let drawn = store.display_for_draw(&floor).unwrap();
    let changed = floor
        .iter()
        .zip(before.iter().zip(&drawn))
        .filter(|(key, (old, new))| refreshed.contains(key) && old != new)
        .count();
    assert!(changed > 0, "the refresh changed floor tiles");
}

/// Asserts that every one of `unknown` is drawn untrusted, from the floor's copy.
fn assert_untrusted(store: &mut TileStore, unknown: &[TileKey], when: &str) {
    let drawn = store.display_for_draw(unknown).unwrap();
    for (key, tile) in unknown.iter().zip(drawn) {
        let tile = tile.unwrap_or_else(|| panic!("{when}: {key:?} drawn"));
        assert!(untrusted(&tile), "{when}: {key:?} drawn untrusted");
    }
    assert_drawn_as_stored(store, unknown, when);
}

/// Floor tiles an edit made unknown are drawn untrusted: the largest bound, the cone covering
/// everything.
///
/// So unflushed, after a roll-over, and after a save, when the files hold them as unknown and
/// the book no longer says so.
#[test]
fn a_floor_tile_made_unknown_is_drawn_untrusted_through_roll_over_and_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut project) = layered(&dir, &mesh);
    let floor = display_at(&mesh, 0..=5);
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 96),
    );
    drop(store.display_for_draw(&floor).unwrap());
    let unknown: Vec<TileKey> = floor
        .iter()
        .filter(|key| matches!(store.display.kept(key), Some(Kept::Unknown)))
        .copied()
        .collect();
    assert!(!unknown.is_empty(), "the edits made floor tiles unknown");
    assert_untrusted(&mut store, &unknown, "unflushed");
    roll_over(&mut store, &dir, "r2.bcrf");
    assert_untrusted(&mut store, &unknown, "rolled over");
    save_and_rebase(&mut store, &mut project, &dir, "r3.bcrf");
    for key in &unknown {
        assert!(
            store.display.kept(key).is_none(),
            "{key:?}: the book moved on"
        );
    }
    assert_untrusted(&mut store, &unknown, "saved");
    assert_drawn_as_stored(&mut store, &floor, "saved, the whole floor");
}

/// Runs the store's jobs to the end.
fn run_all(store: &mut TileStore) {
    loop {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
        if !store.has_jobs() {
            return;
        }
    }
}

/// A layer with another version's records among it is never drawn from the files on open.
///
/// A floor tile is drawn untrusted where the store holds a copy of it, else not at all, until
/// the rebuild records it.
#[test]
fn an_outdated_layer_on_open_is_drawn_untrusted_until_rebuilt() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 97);
    let floor = display_at(&mesh, 0..=5);
    // One floor record of an older version among the rest.
    let mut written = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let old = floor[3];
    let record = written
        .read_display(&old)
        .unwrap()
        .unwrap()
        .known()
        .unwrap();
    written
        .append_display_raw(&old, &crate::file::version_2_record(old, Some(&record)))
        .unwrap();
    written.save().unwrap();
    drop(written);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut store = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    let drawn = store.display_for_draw(&floor).unwrap();
    let mut held = 0;
    for (key, tile) in floor.iter().zip(&drawn) {
        if let Some(tile) = tile {
            assert!(untrusted(tile), "{key:?} drawn untrusted");
            held += 1;
        }
    }
    assert!(held > 0, "the current floor records are held");
    assert_drawn_as_stored(&mut store, &floor, "outdated");
    run_all(&mut store);
    assert_drawn_as_stored(&mut store, &floor, "rebuilt");
}

/// A layer built from another finest level is never drawn from the files on open: the store
/// holds no floor copy of it, and names none as loaded.
#[test]
fn a_foreign_layer_on_open_is_never_drawn_from_the_files() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 98);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let other = level(TOP - 1);
    let mut store = TileStore::open(file, roomy().with_display(other)).unwrap();
    let floor: Vec<TileKey> = display_at(&mesh, 0..=5)
        .into_iter()
        .filter(|k| k.level() < other)
        .collect();
    let drawn = store.display_for_draw(&floor).unwrap();
    for (key, tile) in floor.iter().zip(&drawn) {
        assert!(
            tile.as_ref().is_none_or(untrusted),
            "{key:?} never drawn from the files"
        );
    }
    assert!(store.floor_read.is_empty(), "nothing named as loaded");
    assert_drawn_as_stored(&mut store, &floor, "foreign");
    run_all(&mut store);
    assert_drawn_as_stored(&mut store, &floor, "rebuilt");
}

/// Asserts that every one of `floor` is drawn from the floor's copy, without a read, and as the
/// files hold it.
fn assert_floor_in_memory(store: &mut TileStore, floor: &[TileKey], when: &str) {
    for key in floor {
        assert!(!store.display_needs_read(key), "{when}: {key:?} in memory");
    }
    assert_drawn_as_stored(store, floor, when);
}

/// The floor's copies stay the files' records across a roll-over, which copies them into the
/// new working file: drawn from memory still.
#[test]
fn the_floor_stays_in_memory_across_a_roll_over() {
    let dir = TestDir::new();
    let mesh = mesh();
    let floor = display_at(&mesh, 0..=5);
    let mut store = reopened(&dir, &mesh);
    assert_floor_in_memory(&mut store, &floor, "opened");
    roll_over(&mut store, &dir, "copied.bcp");
    for key in &floor {
        assert!(store.file.locate(key).is_some(), "{key:?}: copied across");
    }
    assert_floor_in_memory(&mut store, &floor, "rolled over");
}

/// The floor's copies stay the files' records across a save, which moves the store onto a new
/// save point: drawn from memory still.
#[test]
fn the_floor_stays_in_memory_across_a_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let floor = display_at(&mesh, 0..=5);
    let (mut store, mut project) = layered(&dir, &mesh);
    assert_floor_in_memory(&mut store, &floor, "layered");
    save_and_rebase(&mut store, &mut project, &dir, "r2.bcrf");
    assert_floor_in_memory(&mut store, &floor, "saved");
}
