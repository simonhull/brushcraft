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
