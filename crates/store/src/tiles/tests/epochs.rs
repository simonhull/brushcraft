//! The stale-tile feed through the store: random edits at any level and refreshes, a fast and a
//! stalled consumer, against the tiles whose evaluated positions, normals and display tiles
//! really changed.

use std::time::Duration;

use geom::DisplayTile;

use super::*;
use crate::stale::Drained;

/// Every finest tile's positions and normals, by key.
fn finest_state(
    store: &mut TileStore,
    mesh: &BaseMesh,
) -> BTreeMap<TileKey, (Vec<Vec3>, Vec<Vec3>)> {
    tiles(mesh, TOP, TOP)
        .into_iter()
        .map(|key| {
            let p = store.evaluate(key).unwrap().to_vec();
            let n = store.normals(key).unwrap().to_vec();
            (key, (p, n))
        })
        .collect()
}

/// Every display tile the store holds, by key.
fn display_state(store: &mut TileStore, mesh: &BaseMesh) -> BTreeMap<TileKey, Option<DisplayTile>> {
    let keys: Vec<TileKey> = tiles(mesh, 0, TOP - 1)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect();
    let stored = store.stored_display(&keys).unwrap();
    keys.into_iter().zip(stored).collect()
}

/// Epochs name every changed tile, in order, and a stalled consumer resyncs.
///
/// Random operations (edits at random levels, each a closed epoch, and refreshes of the display
/// layer, an epoch a slice): a consumer draining after each gets, in order, epochs naming every
/// finest tile whose positions or normals changed and every display tile rebuilt; a stalled one
/// with a small cap is told to resync, and never holds more than the cap.
#[test]
fn epochs_name_every_changed_tile_and_a_stalled_consumer_resyncs() {
    let dir = TestDir::new();
    let mesh = mesh();
    let config = roomy().with_display(level(TOP)).with_stale_cap(200);
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
    }
    let fast = store.subscribe_stale();
    let stalled = store.subscribe_stale();
    let mut last_seq = None;
    let mut rng = Rng(5);
    for op in 0..10 {
        let finest_before = finest_state(&mut store, &mesh);
        let display_before = display_state(&mut store, &mesh);
        if op % 3 == 2 {
            // A refresh of what the edits so far left unknown.
            while {
                store
                    .run_jobs(Instant::now() + Duration::from_secs(60))
                    .unwrap();
                store.has_jobs()
            } {}
        } else {
            let edits = random_edits(&mesh, TOP, 1 + rng.below(3), 300 + op);
            store.begin_op().unwrap();
            for e in &edits {
                store
                    .set_correction(e.face, e.level, e.i, e.j, e.c)
                    .unwrap();
            }
            store.close_epoch().unwrap();
        }
        let mut named = BTreeSet::new();
        match store.drain_stale(&fast) {
            Drained::Epochs(epochs) => {
                for epoch in epochs {
                    assert!(
                        last_seq.is_none_or(|s| epoch.seq == s + 1),
                        "op {op}: epochs in order, none missed"
                    );
                    last_seq = Some(epoch.seq);
                    named.extend(epoch.keys);
                }
            }
            Drained::Resync => panic!("op {op}: the fast consumer keeps up"),
        }
        let finest_after = finest_state(&mut store, &mesh);
        for (key, was) in &finest_before {
            if finest_after[key] != *was {
                assert!(named.contains(key), "op {op}: {key:?} changed, so named");
            }
        }
        let display_after = display_state(&mut store, &mesh);
        for (key, now) in &display_after {
            if now.is_some() && *now != display_before[key] {
                assert!(
                    named.contains(key),
                    "op {op}: display {key:?} rebuilt, so named"
                );
            }
        }
        assert!(
            store.stale.most_pending() <= 200,
            "op {op}: memory within the cap"
        );
    }
    assert_eq!(
        store.drain_stale(&stalled),
        Drained::Resync,
        "the stalled consumer fell past the cap"
    );
    match store.drain_stale(&stalled) {
        Drained::Epochs(epochs) => assert!(epochs.is_empty(), "and goes on from now"),
        Drained::Resync => panic!("one resync"),
    }
}

/// An epoch whose close fails (listing its writes' readers fails, here for a write of a face the
/// mesh doesn't have) stays open whole: the next close hands out every key it noted.
#[test]
fn a_failed_close_keeps_the_epoch_open() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    let cursor = store.subscribe_stale();
    let shown = TileKey::display(FaceId::new(0), level(2), 0, 0).unwrap();
    store.note_epoch_keys([shown]);
    let foreign = TileKey::displacement(FaceId::new(999), level(TOP), 0, 0).unwrap();
    store.epoch_writes.insert(foreign);
    assert!(
        store.close_epoch().is_err(),
        "the readers of a foreign write fail"
    );
    store.epoch_writes.remove(&foreign);
    store.close_epoch().unwrap();
    match store.drain_stale(&cursor) {
        Drained::Epochs(epochs) => assert!(
            epochs.iter().any(|e| e.keys.contains(&shown)),
            "the key noted before the failed close: {epochs:?}"
        ),
        Drained::Resync => panic!("no resync expected"),
    }
}

/// A consumer of the feed in the model test, what it was last shown, and the epoch it expects
/// next.
struct Watcher {
    /// Its place in the feed.
    cursor: crate::stale::StaleCursor,
    /// The finest tiles as of its last drain.
    finest: BTreeMap<TileKey, (Vec<Vec3>, Vec<Vec3>)>,
    /// The display tiles as of its last drain.
    display: BTreeMap<TileKey, Option<DisplayTile>>,
    /// The epoch it expects next, unless it hasn't seen one since subscribing or resyncing.
    next: Option<u64>,
}

/// Drains `watcher` (consumer `n`, after operation `op`): its epochs come in order, none missed,
/// and name every finest tile and display tile changed since its last drain; or it resyncs.
fn observe(store: &mut TileStore, mesh: &BaseMesh, watcher: &mut Watcher, n: usize, op: u64) {
    let finest = finest_state(store, mesh);
    let display = display_state(store, mesh);
    match store.drain_stale(&watcher.cursor) {
        Drained::Epochs(epochs) => {
            let mut named = BTreeSet::new();
            for epoch in epochs {
                assert!(
                    watcher.next.is_none_or(|s| epoch.seq == s),
                    "op {op}, consumer {n}: in order, none missed"
                );
                watcher.next = Some(epoch.seq + 1);
                named.extend(epoch.keys);
            }
            for (key, was) in &watcher.finest {
                if finest[key] != *was {
                    assert!(named.contains(key), "op {op}, consumer {n}: {key:?}");
                }
            }
            for (key, now) in &display {
                if now.is_some() && *now != watcher.display[key] {
                    assert!(named.contains(key), "op {op}, consumer {n}: {key:?}");
                }
            }
        }
        // Past the cap: it shows everything again, from now on.
        Drained::Resync => watcher.next = None,
    }
    watcher.finest = finest;
    watcher.display = display;
}

/// The feed's model through the store, under random operations and drains.
///
/// The operations are edits at random levels, refreshes, saves, and roll-overs that lose a
/// tile; one consumer drains after each, one at random, and one is stalled.
///
/// Whenever the fast or the random consumer drains, its epochs come in order, none missed, and
/// name every finest tile whose positions or normals changed and every display tile rebuilt
/// since its previous drain; or, only past the cap, it is told to resync. The stalled one gets
/// every epoch at the end, or past the cap one resync, and memory stays within the cap
/// throughout.
#[test]
fn the_feed_names_every_change_under_random_operations_and_drains() {
    let dir = TestDir::new();
    let mesh = mesh();
    let cap = 400;
    let config = roomy().with_display(level(TOP)).with_stale_cap(cap);
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, config);
    let run_all = |s: &mut TileStore| {
        while s.has_jobs() {
            s.run_jobs(Instant::now() + Duration::from_secs(60))
                .unwrap();
        }
    };
    run_all(&mut store);
    let mut watchers: Vec<Watcher> = (0..2)
        .map(|_| Watcher {
            cursor: store.subscribe_stale(),
            finest: finest_state(&mut store, &mesh),
            display: display_state(&mut store, &mesh),
            next: None,
        })
        .collect();
    let stalled = store.subscribe_stale();
    let top = tiles(&mesh, TOP, TOP);
    let mut rng = Rng(23);
    let mut rolls = 0;
    for op in 0..16_u64 {
        match rng.below(4) {
            0 => run_all(&mut store),
            1 => drop(store.save().unwrap()),
            2 => {
                // A tile edited, saved, then lost to a roll-over (its record damaged).
                let lost = top[usize::try_from(rng.below(96)).unwrap()];
                let region = Region::of_tile(&lost);
                store.begin_op().unwrap();
                store
                    .set_correction(
                        lost.face(),
                        lost.level(),
                        region.x0() + 7,
                        region.y0() + 9,
                        Vec3::new(0.0, 0.0, 0.2),
                    )
                    .unwrap();
                store.close_epoch().unwrap();
                store.save().unwrap();
                // Shown with the edit, so its loss is a change of its own.
                for (n, watcher) in watchers.iter_mut().enumerate() {
                    observe(&mut store, &mesh, watcher, n, op);
                }
                if store.corrections.find(&lost).is_some() {
                    evict(&mut store, &lost);
                }
                damage(
                    &dir.path("recovery.bcrf"),
                    store.file.locate(&lost).unwrap(),
                );
                rolls += 1;
                let meta = store.file.meta().clone();
                let name = format!("fresh{rolls}.bcrf");
                let fresh = ProjectFile::create_like(&dir.path(&name), meta, &store.file).unwrap();
                let filled = store.fill_roll_over(fresh).unwrap();
                let (old, lost_tiles) = store.roll_over(filled).unwrap();
                drop(old);
                assert_eq!(lost_tiles, [lost], "op {op}: lost");
                std::fs::rename(dir.path(&name), dir.path("recovery.bcrf")).unwrap();
            }
            _ => {
                let edits = random_edits(&mesh, TOP, 1 + rng.below(3), 600 + op);
                store.begin_op().unwrap();
                for e in &edits {
                    store
                        .set_correction(e.face, e.level, e.i, e.j, e.c)
                        .unwrap();
                }
                store.close_epoch().unwrap();
            }
        }
        for (n, watcher) in watchers.iter_mut().enumerate() {
            // The first drains after every operation, the second now and then.
            if n == 1 && rng.below(3) != 0 {
                continue;
            }
            observe(&mut store, &mesh, watcher, n, op);
        }
        assert!(
            store.stale.most_pending() <= cap,
            "op {op}: memory within the cap"
        );
    }
    assert!(rolls > 0, "some tile was lost to a roll-over");
    // The stalled consumer: every epoch since it subscribed, in order, within the cap; or, past
    // the cap, one resync.
    match store.drain_stale(&stalled) {
        Drained::Epochs(epochs) => {
            let held: usize = epochs.iter().map(|e| e.keys.len()).sum();
            assert!(held <= cap, "within the cap");
            for (n, epoch) in epochs.iter().enumerate() {
                assert_eq!(
                    epoch.seq,
                    u64::try_from(n).unwrap(),
                    "every epoch, in order"
                );
            }
        }
        Drained::Resync => {}
    }
}
