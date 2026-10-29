//! The display floor: loaded on open, kept through any workload under a tiny budget, and a
//! configuration it doesn't fit refused.

use std::time::Duration;

use super::*;

/// `config` keeping a display layer under [`TOP`], its budget grown by the floor's share so
/// its pools stay what they were.
fn with_floor(config: StoreConfig, mesh: &BaseMesh) -> StoreConfig {
    let display = config.with_display(level(TOP));
    display.with_budget_bytes(display.budget_bytes + display.floor_bytes(mesh))
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

/// Floor tiles per face of the tests' mesh: levels 0 to 5 are whole-face tiles.
const FLOOR_PER_FACE: usize = 6;

#[test]
fn the_floor_is_loaded_on_open() {
    let dir = TestDir::new();
    let mesh = mesh();
    let config = with_floor(roomy(), &mesh);
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    run_all(&mut store);
    let faces = usize::try_from(mesh.face_count()).unwrap();
    assert_eq!(
        store.floor_size().0,
        faces * FLOOR_PER_FACE,
        "built into the floor"
    );
    store.save().unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let reopened = TileStore::open(file, config).unwrap();
    let (tiles, bytes) = reopened.floor_size();
    assert_eq!(tiles, faces * FLOOR_PER_FACE, "loaded on open");
    assert!(bytes > 0, "with their encodings");
    assert!(!reopened.has_jobs(), "nothing to rebuild");
}

/// Under the tightest budget, random edits, evaluations and refreshes never take a floor tile
/// away: every one stays drawable, the refreshed ones as recorded.
#[test]
fn no_workload_evicts_a_floor_tile() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, with_floor(tight(), &mesh));
    run_all(&mut store);
    let floor: Vec<TileKey> = tiles(&mesh, 0, 5)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect();
    let mut rng = Rng(11);
    let all = tiles(&mesh, 1, TOP);
    for round in 0..8_u64 {
        let mut map = CorrectionMap::new();
        apply(
            &mut store,
            &mut map,
            &random_edits(&mesh, TOP, 2, 900 + round),
        );
        for _ in 0..20 {
            let key = all[usize::try_from(rng.below(u32::try_from(all.len()).unwrap())).unwrap()];
            store.evaluate(key).unwrap();
        }
        let drawn = store.display_for_draw(&floor).unwrap();
        let known = store.stored_display(&floor).unwrap();
        let mut unknown = 0;
        let with_known = floor.iter().zip(&drawn).zip(&known);
        for ((key, tile), kept) in with_known {
            let tile = tile
                .as_ref()
                .unwrap_or_else(|| panic!("round {round}: {key:?} still drawable"));
            if kept.is_none() {
                // Made unknown by the edits, and no stroke's bump covers it: drawn as it was,
                // nothing trusted of its bound or its cone.
                unknown += 1;
                assert_eq!(tile.error(), f32::MAX, "round {round}: {key:?}'s bound");
                assert!(
                    tile.cone().covers_everything(),
                    "round {round}: {key:?}'s cone"
                );
            }
        }
        assert!(unknown > 0, "round {round}: the edits outdate floor tiles");
        assert_eq!(
            store.floor_size().0,
            floor.len(),
            "round {round}: none evicted"
        );
        if round % 2 == 1 {
            run_all(&mut store);
            let stored = store.stored_display(&floor).unwrap();
            let refreshed = store.display_for_draw(&floor).unwrap();
            for ((key, kept), shown) in floor.iter().zip(stored).zip(refreshed) {
                assert_eq!(kept, shown, "round {round}: {key:?} drawn as refreshed");
            }
        }
    }
}

#[test]
fn a_budget_without_room_for_the_floor_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let config = tight().with_display(level(TOP));
    let floor = config.floor_bytes(&mesh);
    assert!(floor > 0, "a floor to keep");
    // The pools' minimum, the other shares and the floor, less a byte.
    let short = config.with_budget_bytes(tight().budget_bytes + floor - 1);
    let file = ProjectFile::create(&dir.path("a.bcp"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let refused = TileStore::open(file, short).map(|_| ());
    assert!(
        matches!(
            refused,
            Err(CacheError::FloorTooLarge { .. } | CacheError::BudgetTooSmall { .. })
        ),
        "{refused:?}"
    );
}

/// The floor's share counts every tile of every floor level at its largest, levels with several
/// tiles a face included, and holds the floor the store then loads.
#[test]
fn the_floor_share_counts_every_tile_of_each_level() {
    let mesh = mesh();
    let config = roomy()
        .with_display(level(TOP))
        .with_floor_top(level(TOP - 1));
    let want: usize = tiles(&mesh, 0, TOP - 1)
        .iter()
        .map(|k| {
            crate::display::max_body(
                &TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap(),
            )
        })
        .sum();
    assert_eq!(config.floor_bytes(&mesh), want, "every tile, levels 0 to 7");
    let dir = TestDir::new();
    let config = config.with_budget_bytes(config.budget_bytes + config.floor_bytes(&mesh));
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    run_all(&mut store);
    let (_, bytes) = store.floor_size();
    assert!(
        bytes <= config.floor_bytes(&mesh),
        "the floor fits its share"
    );
}
