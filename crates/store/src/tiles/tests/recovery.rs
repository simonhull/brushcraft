//! The recovery writer over a store, against the dense evaluation.
//!
//! Every durable save point holds the edits up to its snapshot; ticks pace the flushes; edits
//! made while a flush is in flight reach the next one; a drop mid-flush is prompt and leaves a
//! file that opens; a failed sync is reported and leaves no save point.

use super::*;
use crate::recovery::{FLUSH_INTERVAL, RecoveryError, RecoveryWriter};

/// Blocks until the writer has no flush in flight or queued, advancing it reply by reply.
fn settle(writer: &mut RecoveryWriter, store: &mut TileStore) {
    while !writer.is_idle() {
        writer.step_blocking(store).unwrap();
    }
}

/// Asserts that the working file at `path`, opened at its save point `offset`, evaluates every
/// tile of levels 1–[`TOP`] as the dense evaluation of `map` does.
fn assert_save_point(path: &std::path::Path, offset: u64, mesh: &BaseMesh, map: &CorrectionMap) {
    let file = ProjectFile::open_at(path, offset).unwrap();
    let mut store = TileStore::open(file, None, roomy()).unwrap();
    let what = format!("save point {offset}");
    assert_positions(&mut store, &dense(mesh, map), &tiles(mesh, 1, TOP), &what);
}

/// Sets an interior vertex of `key` (a level-[`TOP`] tile) to `c` with a fresh op, in the store
/// and in the oracle's map; returns the op.
fn edit_interior(store: &mut TileStore, map: &mut CorrectionMap, key: TileKey, c: Vec3) -> OpId {
    let op = store.next_op().unwrap();
    store
        .edit_tile(key, op, |edit| edit.set(5, 7, c))
        .unwrap()
        .unwrap();
    let region = Region::of_tile(&key);
    map.set(
        store.mesh.topology(),
        key.face(),
        key.level(),
        region.x0() + 5,
        region.y0() + 7,
        c,
    )
    .unwrap();
    op
}

#[test]
fn a_stroke_end_yields_a_save_point_equal_to_the_edits_so_far() {
    for (name, config) in [("roomy", roomy()), ("tight", tight())] {
        let dir = TestDir::new();
        let mesh = mesh();
        let path = dir.path("recovery.bcrf");
        let mut store = new_store(&dir, "recovery.bcrf", &mesh, config);
        let mut writer = RecoveryWriter::new(&mut store).unwrap();
        let start = Instant::now();
        let mut map = CorrectionMap::new();
        let mut points = Vec::new();
        for (stroke, seed) in [31, 32, 33].into_iter().enumerate() {
            apply(&mut store, &mut map, &random_edits(&mesh, TOP, 120, seed));
            // Under the tight budget, evaluation evicts and writes dirty tiles back, so
            // streamed and written-back records interleave.
            for key in tiles(&mesh, TOP, TOP).iter().step_by(7) {
                store.evaluate(*key).unwrap();
            }
            let now = start + Duration::from_millis(10) * u32::try_from(stroke).unwrap();
            writer.end_stroke(&mut store, now).unwrap();
            settle(&mut writer, &mut store);
            assert!(
                store.unsaved.is_empty(),
                "{name} stroke {stroke}: every edit is in the file"
            );
            points.push((writer.durable_save_point().unwrap(), map.clone()));
        }
        assert_eq!(writer.flushes(), 3, "{name}: one flush per stroke");
        drop(writer);
        drop(store);
        for (offset, edits) in &points {
            assert_save_point(&path, *offset, &mesh, edits);
        }
    }
}

#[test]
fn records_appended_while_the_tiles_sync_are_synced_before_the_save_point() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 41));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    // More edits, all written back while the tiles sync: the file grows past what the sync
    // covers.
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 42));
    store.evict_all().unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        writer.phase(),
        "syncing tiles",
        "the file grew during the sync, so it is synced again before the save point"
    );
    settle(&mut writer, &mut store);
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
}

#[test]
fn a_long_stroke_flushes_about_once_a_second() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    let edits = random_edits(&mesh, TOP, 400, 51);
    let start = Instant::now();
    let tick = Duration::from_millis(100);
    let mut flushed_at = Vec::new();
    // Ten seconds of one stroke: a few dabs between ticks, a tick every 100 ms.
    for (n, dabs) in (1..=100u32).zip(edits.chunks(4)) {
        apply(&mut store, &mut map, dabs);
        let now = start + tick * n;
        let before = writer.flushes();
        writer.tick(&mut store, now).unwrap();
        settle(&mut writer, &mut store);
        if writer.flushes() > before {
            flushed_at.push(now - start);
        }
    }
    assert!(
        (9..=11).contains(&flushed_at.len()),
        "{} flushes in ten seconds: {flushed_at:?}",
        flushed_at.len()
    );
    for pair in flushed_at.windows(2) {
        let gap = pair[1].checked_sub(pair[0]).unwrap();
        assert!(
            gap >= FLUSH_INTERVAL,
            "flushes at least a second apart: {flushed_at:?}"
        );
        assert!(
            gap < FLUSH_INTERVAL + tick,
            "and at most a tick late: {flushed_at:?}"
        );
    }
    // The dabs after the last flush wait for the next tick a second on.
    let flushes = writer.flushes();
    let quiet = start + tick * 200;
    writer.tick(&mut store, quiet).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(
        writer.flushes(),
        flushes + 1,
        "the stroke's tail is flushed"
    );
    assert!(store.unsaved.is_empty(), "every edit is in the file");
    let tail = writer.flushes();
    writer.tick(&mut store, quiet + tick * 20).unwrap();
    assert!(writer.is_idle(), "nothing edited since, nothing flushed");
    writer.end_stroke(&mut store, quiet).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(
        writer.flushes(),
        tail,
        "a stroke end with nothing new writes nothing"
    );
}

#[test]
fn edits_during_a_flush_land_in_the_next_one_never_lost_never_doubled() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let start = Instant::now();
    let top = tiles(&mesh, TOP, TOP);
    let mut map = CorrectionMap::new();

    // Stroke A: two tiles of its own, and the edits of a random stroke.
    let (t, u) = (top[3], top[17]);
    edit_interior(&mut store, &mut map, t, Vec3::new(0.001, 0.0, 0.0));
    edit_interior(&mut store, &mut map, u, Vec3::new(0.0, 0.001, 0.0));
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 40, 61));
    let stroke_a = map.clone();
    let snapshot: BTreeSet<TileKey> = store.unsaved.keys().copied().collect();
    writer.end_stroke(&mut store, start).unwrap();
    assert_eq!(writer.phase(), "encoding", "the flush is in flight");

    // While it encodes: `t` again (it is in the snapshot), and a tile it doesn't hold.
    let fresh = *top.iter().find(|key| !snapshot.contains(key)).unwrap();
    let t_again = edit_interior(&mut store, &mut map, t, Vec3::new(0.002, 0.0, 0.0));
    edit_interior(&mut store, &mut map, fresh, Vec3::new(0.0, 0.0, 0.001));
    // A stroke ends mid-flush: the next flush is queued.
    writer.end_stroke(&mut store, start).unwrap();

    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    assert_eq!(
        store.file.unsaved_keys(),
        snapshot.iter().copied().collect::<Vec<_>>(),
        "the snapshot's tiles, each once, and nothing edited after it"
    );
    let t_slot = store.corrections.find(&t).unwrap();
    assert_eq!(
        store.corrections.is_dirty(t_slot),
        Some(true),
        "edited after its snapshot: still dirty"
    );
    assert_eq!(store.unsaved.get(&t), Some(&t_again), "with its newer edit");
    let u_slot = store.corrections.find(&u).unwrap();
    assert_eq!(
        store.corrections.is_dirty(u_slot),
        Some(false),
        "unchanged since its snapshot: clean"
    );

    // While the tiles sync, and while the save point syncs: more edits of `u`.
    edit_interior(&mut store, &mut map, u, Vec3::new(0.0, 0.002, 0.0));
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing save point", "the tiles are synced");
    edit_interior(&mut store, &mut map, u, Vec3::new(0.0, 0.003, 0.0));
    writer.step_blocking(&mut store).unwrap();
    let first = writer.durable_save_point().unwrap();
    assert_eq!(writer.flushes(), 1, "the first flush is durable");
    assert_eq!(
        writer.phase(),
        "encoding",
        "and the queued flush started at once"
    );
    let after_first: BTreeSet<TileKey> = store.unsaved.keys().copied().collect();
    assert_eq!(
        after_first,
        BTreeSet::from([t, u, fresh]),
        "the tiles edited during the first flush"
    );

    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        store.file.unsaved_keys(),
        after_first.iter().copied().collect::<Vec<_>>(),
        "the second flush appends those tiles, each once"
    );
    settle(&mut writer, &mut store);
    assert_eq!(writer.flushes(), 2, "the queued flush is durable too");
    let second = writer.durable_save_point().unwrap();
    assert!(store.unsaved.is_empty(), "nothing left dirty");
    assert_eq!(
        store.file.read_displacement(&t).unwrap().unwrap().0,
        t_again,
        "`t`'s latest record carries its latest edit"
    );

    // Nothing is lost from the slots either: every tile read back from the file.
    store.evict_all().unwrap();
    let dense_all = dense(&mesh, &map);
    assert_positions(&mut store, &dense_all, &top, "read back after the flushes");
    drop(writer);
    drop(store);
    assert_save_point(&path, first, &mesh, &stroke_a);
    assert_save_point(&path, second, &mesh, &map);
}

#[test]
fn dropping_the_writer_mid_flush_joins_promptly_and_the_file_reopens() {
    let (done, finished) = mpsc::channel();
    // The store isn't `Send`, so it lives and dies on a thread of its own, and a hang there
    // fails the test here instead of hanging it.
    thread::spawn(move || {
        let dir = TestDir::new();
        let mesh = mesh();
        for (n, phase) in ["encoding", "syncing tiles", "syncing save point"]
            .into_iter()
            .enumerate()
        {
            let name = format!("recovery-{n}.bcrf");
            let path = dir.path(&name);
            let mut store = new_store(&dir, &name, &mesh, roomy());
            let mut writer = RecoveryWriter::new(&mut store).unwrap();
            let mut map = CorrectionMap::new();
            apply(&mut store, &mut map, &random_edits(&mesh, TOP, 300, 71));
            writer.finish(&mut store, Instant::now()).unwrap();
            let durable = writer.durable_save_point().unwrap();
            let saved = map.clone();
            // Every tile of the top level, so the encoding takes a while.
            let mut later = map.clone();
            for (k, key) in tiles(&mesh, TOP, TOP).into_iter().enumerate() {
                let c = Vec3::new(0.0, 0.0, 1e-4 * f32::from(u16::try_from(k + 1).unwrap()));
                edit_interior(&mut store, &mut later, key, c);
            }
            writer.end_stroke(&mut store, Instant::now()).unwrap();
            while writer.phase() != phase {
                assert!(!writer.is_idle(), "the flush passes through {phase}");
                writer.step_blocking(&mut store).unwrap();
            }
            let thread = writer.liveness();
            let began = Instant::now();
            drop(writer);
            let took = began.elapsed();
            assert!(thread.upgrade().is_none(), "{phase}: the thread was joined");
            assert!(
                took < Duration::from_secs(1),
                "{phase}: dropping mid-flush took {took:?}"
            );
            drop(store);
            assert_save_point(&path, durable, &mesh, &saved);
            // The latest state opens too: the save point, and whole records after it.
            let file = ProjectFile::open(&path).unwrap();
            let mut reopened = TileStore::open(file, None, roomy()).unwrap();
            for key in tiles(&mesh, 1, TOP) {
                reopened.evaluate(key).unwrap();
            }
        }
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(120))
        .expect("the writer dropped in time (a panic on its thread shows above)");
}

#[test]
fn a_failed_sync_is_reported_and_writes_no_save_point_over_it() {
    // The tile sync fails (the first of the flush), then the save point's (the second).
    for (pass, phase) in [(0, "syncing tiles"), (1, "syncing save point")] {
        let dir = TestDir::new();
        let mesh = mesh();
        let path = dir.path("recovery.bcrf");
        let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
        let mut writer = RecoveryWriter::new(&mut store).unwrap();
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 81));
        let points = store.file.save_points().unwrap();
        writer.fail_syncs(pass, 1);
        writer.end_stroke(&mut store, Instant::now()).unwrap();
        let mut failed = None;
        while !writer.is_idle() {
            let at = writer.phase();
            if let Err(e) = writer.step_blocking(&mut store) {
                failed = Some((at, e));
            }
        }
        let (at, error) = failed.expect("the failed sync is reported");
        assert_eq!(at, phase, "it failed where the hook made it fail");
        assert!(
            matches!(error, RecoveryError::Sync(StoreError::Io(_))),
            "{phase}: reported as a sync failure: {error:?}"
        );
        assert_eq!(writer.flushes(), 0, "{phase}: no flush completed");
        assert_eq!(
            writer.durable_save_point(),
            None,
            "{phase}: no save point is durable"
        );
        if pass == 0 {
            assert_eq!(
                store.file.save_points().unwrap(),
                points,
                "no save point was written over tiles whose sync failed"
            );
        }
        // The next stroke end flushes again, the failed flush's edits with it, with nothing new
        // edited.
        writer.end_stroke(&mut store, Instant::now()).unwrap();
        settle(&mut writer, &mut store);
        assert_eq!(writer.flushes(), 1, "{phase}: the retry is durable");
        let offset = writer.durable_save_point().unwrap();
        drop(writer);
        drop(store);
        assert_save_point(&path, offset, &mesh, &map);
    }
}

#[test]
fn a_failed_sync_surfaces_through_poll_and_finish() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 91),
    );
    writer.fail_syncs(0, 1);
    let finished = writer.finish(&mut store, Instant::now());
    assert!(
        matches!(finished, Err(RecoveryError::Sync(_))),
        "finish reports it: {finished:?}"
    );
    writer.fail_syncs(0, 1);
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    let began = Instant::now();
    let polled = loop {
        match writer.poll(&mut store) {
            Ok(()) if !writer.is_idle() => {
                assert!(began.elapsed() < Duration::from_secs(30), "the flush ends");
                thread::yield_now();
            }
            other => break other,
        }
    };
    assert!(
        matches!(polled, Err(RecoveryError::Sync(_))),
        "poll reports it: {polled:?}"
    );
    writer.close(&mut store, Instant::now()).unwrap();
    assert!(store.streams.is_empty(), "closing unsubscribes");
}
