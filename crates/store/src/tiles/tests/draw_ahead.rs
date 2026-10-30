//! Display records read ahead for drawing ([`TileStore::read_display_ahead`]), and the floor
//! drawn from memory after an open.

use super::display::saved_layer;
use super::*;

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
fn records_read_ahead_are_drawn_without_a_read_and_then_forgotten() {
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
            !store.read_ahead.read.tiles.contains_key(key),
            "{key:?}: forgotten once drawn"
        );
        assert!(
            store.display_needs_read(key),
            "{key:?}: read again if drawn again"
        );
    }
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
