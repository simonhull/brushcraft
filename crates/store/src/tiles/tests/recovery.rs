//! The recovery writer over a store, against the dense evaluation.
//!
//! Every durable save point holds the edits up to its snapshot; ticks pace the flushes; edits
//! made while a flush is in flight reach the next one; a drop mid-flush is prompt and leaves a
//! file that opens; a failed sync is reported and leaves no save point; ticked, a flush's snapshot
//! stays proportional to the tiles dirtied per interval.

use super::*;
use crate::index::SpatialIndex;
use crate::index::tests::{assert_loads_as_built, wave_now};
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
    let mut store = TileStore::open(file, roomy()).unwrap();
    let what = format!("save point {offset}");
    assert_positions(&mut store, &dense(mesh, map), &tiles(mesh, 1, TOP), &what);
}

/// Sets an interior vertex of `key` (a level-[`TOP`] tile) to `c` as a new op, in the store
/// and in the oracle's map; returns the op.
fn edit_interior(store: &mut TileStore, map: &mut CorrectionMap, key: TileKey, c: Vec3) -> OpId {
    let op = store.begin_op().unwrap();
    dab_interior(store, map, key, c);
    op
}

/// [`edit_interior`] as part of the current op: another dab of the same stroke.
fn dab_interior(store: &mut TileStore, map: &mut CorrectionMap, key: TileKey, c: Vec3) {
    dab_interior_at(store, map, key, (5, 7), c);
}

/// Sets interior vertex `(i, j)` of `key` to `c` as part of the current op, in the store and in
/// the oracle's map.
fn dab_interior_at(
    store: &mut TileStore,
    map: &mut CorrectionMap,
    key: TileKey,
    (i, j): (u32, u32),
    c: Vec3,
) {
    store
        .edit_tile(key, |edit| edit.set(i, j, c))
        .unwrap()
        .unwrap();
    let region = Region::of_tile(&key);
    map.set(
        store.mesh.topology(),
        key.face(),
        key.level(),
        region.x0() + i,
        region.y0() + j,
        c,
    )
    .unwrap();
}

/// Every stroke end yields a durable save point holding exactly the edits so far.
///
/// Three strokes under a roomy budget and a tight one (where evaluation writes dirty tiles back,
/// so streamed and written-back records interleave): one flush per stroke, nothing left dirty
/// after each, and every save point, reopened, evaluates as the dense evaluation of its edits.
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

/// Records past the cut written back during the tile sync need no second sync.
///
/// Write-backs during the tile sync grow the file past what that sync covers, but they hold
/// edits past the flush's cut, which its save point doesn't name: the flush writes its save
/// point without syncing again. The save point still holds only the edits at its cut, and the
/// written-back edits land in the next one.
#[test]
fn records_past_the_cut_written_back_during_the_tile_sync_need_no_second_sync() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 41));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    let cut = map.clone();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    // More edits, all written back while the tiles sync: the file grows past what the sync
    // covers.
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 42));
    store.evict_all().unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        writer.phase(),
        "syncing save point",
        "the records written back during the sync are past the cut, so the save point needs no \
         second sync"
    );
    settle(&mut writer, &mut store);
    let first = writer.durable_save_point().unwrap();
    // The records written back belong to edits past the flush's cut: the next flush.
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    let second = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, first, &mesh, &cut);
    assert_save_point(&path, second, &mesh, &map);
}

/// A writer whose thread stopped (as a panic would stop it) is failed: no flush can be finished
/// there, so every trigger after the one that found it gone asks for a new file.
#[test]
fn a_writer_whose_thread_stopped_is_failed_and_needs_a_new_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 20, 51));
    writer.stop_thread();
    assert!(
        matches!(
            writer.end_stroke(&mut store, Instant::now()),
            Err(RecoveryError::Stopped)
        ),
        "the stroke's flush finds the thread gone"
    );
    assert!(
        writer.is_failed(),
        "a writer that can't finish a flush is failed"
    );
    assert!(
        matches!(
            writer.end_stroke(&mut store, Instant::now()),
            Err(RecoveryError::NeedsNewFile)
        ),
        "it needs a new file"
    );
}

/// The change feed stays bounded between saves: every flush drains it for every consumer.
///
/// Ten strokes over the same eight tiles, a new op each, each flushed: the feed holds nothing
/// after, though no save has drained save's consumer, and the edits still wait for a save.
#[test]
fn the_change_feed_stays_bounded_between_saves() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().take(8).collect();
    for k in 0_u8..10 {
        for &key in &keys {
            edit_interior(
                &mut store,
                &mut map,
                key,
                Vec3::new(0.0, 0.0, 0.001 * f32::from(k + 1)),
            );
        }
        writer.end_stroke(&mut store, Instant::now()).unwrap();
        settle(&mut writer, &mut store);
    }
    assert_eq!(
        store.feed.backlog(),
        0,
        "every consumer has drained what it was given"
    );
    assert!(store.has_unsaved_edits(), "the edits still wait for a save");
}

/// A record at the cut appended during the tile sync is synced before the save point.
///
/// A record the save point names, appended after the tile sync was asked for, isn't covered by
/// that sync: the flush syncs again before writing the save point.
#[test]
fn a_record_at_the_cut_appended_during_the_tile_sync_is_synced_before_the_save_point() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 41));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    let key = tiles(&mesh, TOP, TOP)[0];
    let tile = CorrectionTile::zero(key.side()).unwrap();
    let op = store.begin_op().unwrap();
    store
        .file
        .write_displacement_stamped(key, op, Stamp::ZERO, &tile)
        .unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        writer.phase(),
        "syncing tiles",
        "a record the save point names was appended after the sync was asked for"
    );
}

/// A tile at the cut, written back to make room during the tile sync, is synced before the save
/// point: the store's own write-back, end to end.
///
/// One copy fails to encode, so its tile stays dirty at its at-cut edit. Written back while the
/// tiles sync, its record is one the save point names, so the flush syncs again first; the save
/// point then holds every edit at the cut, that tile's included.
#[test]
fn an_at_cut_tile_written_back_during_the_tile_sync_is_synced_before_the_save_point() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 41));
    writer.fail_encodes(1);
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    let cut = map.clone();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    assert!(
        !store.unsaved.is_empty(),
        "the tile whose copy failed to encode stays dirty"
    );
    store.evict_all().unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        writer.phase(),
        "syncing tiles",
        "the tile written back is at the cut, so the save point names it: it is synced first"
    );
    settle(&mut writer, &mut store);
    let first = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, first, &mesh, &cut);
}

/// An at-cut record replacing a past-cut one during the tile sync is synced before the save
/// point.
///
/// A tile edited past the cut and written back leaves the cut a prior record; a record of the
/// same tile at the cut appended after it is the cut's state, so the save point names it, and
/// the flush syncs again before writing the save point.
#[test]
fn an_at_cut_record_replacing_a_past_cut_one_during_the_tile_sync_is_synced_before_the_save_point()
{
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 41));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the records are appended");
    let key = tiles(&mesh, TOP, TOP)[0];
    edit_interior(&mut store, &mut map, key, Vec3::new(0.0, 0.0, 0.002));
    // Written back past the cut: the cut remembers the tile's prior record.
    store.evict_all().unwrap();
    let tile = CorrectionTile::zero(key.side()).unwrap();
    let op = store.begin_op().unwrap();
    store
        .file
        .write_displacement_stamped(key, op, Stamp::ZERO, &tile)
        .unwrap();
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(
        writer.phase(),
        "syncing tiles",
        "the at-cut record that replaced the prior is named, and was appended after the sync was \
         asked for"
    );
}

/// Ticks during a ten-second stroke of one op flush about once a second.
///
/// A tick every 100 ms: the flushes are at least [`FLUSH_INTERVAL`] apart and at most a tick
/// late. The stroke's tail is flushed by the next due tick; with nothing edited since, neither a
/// tick nor a stroke end writes anything.
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
    // Ten seconds of one stroke, one op: a few dabs between ticks, a tick every 100 ms.
    store.begin_op().unwrap();
    for (n, dabs) in (1..=100u32).zip(edits.chunks(4)) {
        apply_dabs(&mut store, &mut map, dabs);
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

/// Edits made while a flush is in flight land in the next flush, never lost and never doubled.
///
/// Tiles edited during each phase (encoding, tile sync, save point sync) stay dirty and reach
/// the queued flush, which starts as soon as the first is durable; each flush appends each of
/// its tiles once, a tile's latest record carries its latest edit, and both save points reopen
/// to the state at their cuts.
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
    // Held back until the next stroke has ended, so that stroke end finds the flush still
    // encoding however the threads are scheduled.
    writer.pause(true);
    writer.end_stroke(&mut store, start).unwrap();
    assert_eq!(writer.phase(), "encoding", "the flush is in flight");

    // While it encodes: `t` again (it is in the snapshot), and a tile it doesn't hold.
    let fresh = *top.iter().find(|key| !snapshot.contains(key)).unwrap();
    let t_again = edit_interior(&mut store, &mut map, t, Vec3::new(0.002, 0.0, 0.0));
    edit_interior(&mut store, &mut map, fresh, Vec3::new(0.0, 0.0, 0.001));
    // A stroke ends mid-flush: the next flush is queued.
    writer.end_stroke(&mut store, start).unwrap();
    assert_eq!(writer.phase(), "encoding", "still in flight");
    assert!(writer.is_queued(), "the next flush is queued");
    writer.pause(false);

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
    assert_eq!(
        store.unsaved.get(&t).map(|write| write.op),
        Some(t_again),
        "with its newer edit"
    );
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

/// Dropping the writer in any phase of a flush joins its thread promptly, and the file reopens.
///
/// For each phase, the drop takes under a second and leaves no thread behind; the file opens at
/// its last durable save point, and its latest state is that save point, or the flush's own if
/// its save point was already written.
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
            // The latest state is the last save point in the file, and nothing after it: the
            // flush's records are discarded unless its save point was written (its sync was
            // in flight, but the page cache has it).
            let latest = if phase == "syncing save point" {
                &later
            } else {
                &saved
            };
            let file = ProjectFile::open(&path).unwrap();
            let mut reopened = TileStore::open(file, roomy()).unwrap();
            assert_positions(
                &mut reopened,
                &dense(&mesh, latest),
                &tiles(&mesh, 1, TOP),
                &format!("{phase}: the latest save point"),
            );
        }
        // Removed before the test is told it is done: the test's process may exit at once,
        // ending this thread before its drops run.
        drop(dir);
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(120))
        .expect("the writer dropped in time (a panic on its thread shows above)");
}

/// A failed sync, of the tiles or of the save point, fails the writer for good.
///
/// The failure is reported as a sync error where the hook made it fail; no save point is written
/// over tiles whose sync failed; every later trigger asks for a new file and writes nothing; and
/// the file still opens at its last durable save point.
#[test]
fn a_failed_sync_fails_the_writer_and_leaves_the_last_durable_save_point() {
    // The tile sync fails (the first of the flush), then the save point's (the second).
    for (pass, phase) in [(0, "syncing tiles"), (1, "syncing save point")] {
        let dir = TestDir::new();
        let mesh = mesh();
        let path = dir.path("recovery.bcrf");
        let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
        let mut writer = RecoveryWriter::new(&mut store).unwrap();
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 81));
        writer.finish(&mut store, Instant::now()).unwrap();
        let durable = writer.durable_save_point().unwrap();
        let saved = map.clone();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 82));
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
        assert!(writer.is_failed(), "{phase}: the writer is failed");
        assert_eq!(
            writer.flushes(),
            1,
            "{phase}: no flush completed after the first"
        );
        assert_eq!(
            writer.durable_save_point(),
            Some(durable),
            "{phase}: the last durable save point stands"
        );
        if pass == 0 {
            assert_eq!(
                store.file.save_points().unwrap(),
                points,
                "no save point was written over tiles whose sync failed"
            );
        }
        // Never again into this file: every trigger asks for a new one, and writes nothing.
        let len = store.working_len();
        let at_failure = map.clone();
        apply(&mut store, &mut map, &random_edits(&mesh, TOP, 10, 83));
        for (what, outcome) in [
            ("end_stroke", writer.end_stroke(&mut store, Instant::now())),
            ("finish", writer.finish(&mut store, Instant::now())),
            (
                "tick",
                writer.tick(&mut store, Instant::now() + 2 * FLUSH_INTERVAL),
            ),
        ] {
            assert!(
                matches!(outcome, Err(RecoveryError::NeedsNewFile)),
                "{phase}: {what}: {outcome:?}"
            );
        }
        assert_eq!(store.working_len(), len, "{phase}: nothing more written");
        drop(writer);
        drop(store);
        // The file opens at its last durable save point; if the failed sync was the save
        // point's, at that save point, which reached the page cache (on a real failure it may
        // not have reached the disk, and the open finds the one before).
        assert_save_point(&path, durable, &mesh, &saved);
        let file = ProjectFile::open(&path).unwrap();
        let mut reopened = TileStore::open(file, roomy()).unwrap();
        let latest = if pass == 0 { &saved } else { &at_failure };
        assert_positions(
            &mut reopened,
            &dense(&mesh, latest),
            &tiles(&mesh, 1, TOP),
            &format!("{phase}: the latest save point"),
        );
    }
}

/// A failed sync is reported by whichever call advances the flush to it.
///
/// That is [`RecoveryWriter::poll`] or [`RecoveryWriter::finish`]. Closing the failed writer says
/// it needs a new file, and unsubscribes it all the same.
#[test]
fn a_failed_sync_surfaces_through_poll_and_finish() {
    for through in ["finish", "poll"] {
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
        let outcome = if through == "finish" {
            writer.finish(&mut store, Instant::now())
        } else {
            writer.end_stroke(&mut store, Instant::now()).unwrap();
            let began = Instant::now();
            loop {
                match writer.poll(&mut store) {
                    Ok(()) if !writer.is_idle() => {
                        assert!(began.elapsed() < Duration::from_secs(30), "the flush ends");
                        thread::yield_now();
                    }
                    other => break other,
                }
            }
        };
        assert!(
            matches!(outcome, Err(RecoveryError::Sync(_))),
            "{through} reports it: {outcome:?}"
        );
        assert!(writer.is_failed(), "{through}: failed");
        let closed = writer.close(&mut store, Instant::now());
        assert!(
            matches!(closed, Err(RecoveryError::NeedsNewFile)),
            "{through}: closing a failed writer says so: {closed:?}"
        );
        assert!(
            store.streams.is_empty(),
            "{through}: closing unsubscribes all the same"
        );
    }
}

/// A sync that fails with a flush queued behind it settles the writer as failed in a few steps:
/// the queued flush is dropped rather than retried against the failed file.
///
/// The writer thread is held back until the second stroke has ended, so that stroke end meets
/// the first flush still encoding and queues the next. Unheld, the thread could encode and fail
/// the sync before it, and the stroke end would report the failure with nothing queued.
#[test]
fn a_failed_sync_with_a_flush_queued_does_not_spin() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    writer.fail_syncs(0, 1);
    writer.pause(true);
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 30, 91));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 30, 92));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    assert_eq!(writer.phase(), "encoding", "a flush in flight");
    assert!(writer.is_queued(), "and one queued");
    writer.pause(false);
    let mut steps = 0;
    let mut failed = false;
    while !writer.is_idle() {
        steps += 1;
        assert!(steps < 10, "the writer settles");
        failed |= writer.step_blocking(&mut store).is_err();
    }
    assert!(failed && writer.is_failed(), "the sync failed the writer");
    assert_eq!(
        steps, 2,
        "the encoding and the failed sync, nothing retried"
    );
    assert_eq!(writer.phase(), "failed", "and the queued flush was dropped");
    assert!(!writer.is_queued(), "dropped, not kept for later");
}

/// The reproduction of the bug a cut closes.
///
/// A flush snapshots tiles k and j at their edits, then edit X writes both while the flush is in
/// flight, and j, unpinned, is written back to make room before the flush's save point. That save
/// point must hold the state at the snapshot: k and j as of their earlier edits, neither at X.
#[test]
fn a_save_point_is_the_state_at_its_cut() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, pinning());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let (k, j) = (top[3], top[40]);
    let mut map = CorrectionMap::new();
    let k_op = edit_interior(&mut store, &mut map, k, Vec3::new(0.0, 0.0, 0.25));
    let j_op = edit_interior(&mut store, &mut map, j, Vec3::new(0.0, 0.0, -0.25));
    let guard = store.pin(&[k]).unwrap();
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    assert_eq!(writer.phase(), "encoding", "the snapshot is taken");
    let cut = map.clone();
    store.begin_op().unwrap();
    dab_interior(&mut store, &mut map, k, Vec3::new(0.5, 0.0, 0.0));
    dab_interior(&mut store, &mut map, j, Vec3::new(-0.5, 0.0, 0.0));
    store.evict_all().unwrap();
    assert!(
        store.corrections.find(&j).is_none(),
        "j was written back at X"
    );
    assert!(store.corrections.find(&k).is_some(), "k, pinned, stays");
    while !writer.is_idle() {
        writer.step_blocking(&mut store).unwrap();
    }
    let first = writer.durable_save_point().unwrap();
    drop(guard);
    // The next flush takes X.
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    while !writer.is_idle() {
        writer.step_blocking(&mut store).unwrap();
    }
    let second = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    let at_cut = ProjectFile::open_at(&path, first).unwrap();
    for (key, op) in [(k, k_op), (j, j_op)] {
        let (got, _) = at_cut.read_displacement(&key).unwrap().unwrap();
        assert_eq!(got, op, "{key:?}: as of the cut, not X");
    }
    drop(at_cut);
    assert_save_point(&path, first, &mesh, &cut);
    assert_save_point(&path, second, &mesh, &map);
}

/// Reproduction: one op across a flush.
///
/// A stroke's dab of `t` is snapshotted; a later dab of the same op writes `t` again before the
/// flush appends its copy. The copy must not leave the slot clean (the ops match, but the slot
/// has been written since), or the eviction that follows drops the later dab unwritten, and
/// neither the store nor a save of it holds it.
#[test]
fn a_later_dab_of_the_snapshotted_op_survives_eviction_and_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let t = tiles(&mesh, TOP, TOP)[3];
    let mut map = CorrectionMap::new();
    let op = edit_interior(&mut store, &mut map, t, Vec3::new(0.0, 0.0, 0.25));
    writer.tick(&mut store, Instant::now()).unwrap();
    assert_eq!(writer.phase(), "encoding", "a flush mid-stroke");
    let snapshot = map.clone();
    dab_interior(&mut store, &mut map, t, Vec3::new(0.5, 0.0, 0.0));
    assert_eq!(store.current_op(), Some(op), "the same stroke, the same op");
    writer.step_blocking(&mut store).unwrap();
    assert_eq!(writer.phase(), "syncing tiles", "the copy is appended");
    let slot = store.corrections.find(&t).unwrap();
    assert_eq!(
        store.corrections.is_dirty(slot),
        Some(true),
        "written since its snapshot: still dirty"
    );
    settle(&mut writer, &mut store);
    let first = writer.durable_save_point().unwrap();
    store.evict_all().unwrap();
    let all = tiles(&mesh, 1, TOP);
    assert_positions(&mut store, &dense(&mesh, &map), &all, "after the eviction");
    // The save: the writer finishes its flushes, then the store saves the working file.
    writer.close(&mut store, Instant::now()).unwrap();
    store.save().unwrap();
    drop(store);
    assert_save_point(&path, first, &mesh, &snapshot);
    let file = ProjectFile::open(&path).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_positions(&mut reopened, &dense(&mesh, &map), &all, "saved");
}

/// A tick during a long stroke cuts between two dabs of its one op.
///
/// Dabs go on while the flush is in flight, and are written back (to make room, under the tight
/// budget, or all at once) before its save point is written. The save point holds exactly the
/// dabs before the cut, to the dab, and the next one the rest.
#[test]
fn a_tick_mid_stroke_cuts_between_two_dabs_of_one_op() {
    /// The finest level edited and checked: coarse, so every save point can be checked in full.
    const LEVEL: u8 = 5;
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let check = tiles(&mesh, 1, LEVEL);
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, tight());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    let op = store.begin_op().unwrap();
    let start = Instant::now();
    let mut points = Vec::new();
    let mut past_cut = false;
    let mut seed = 300;
    for second in 0..4u32 {
        seed += 1;
        apply_dabs(&mut store, &mut map, &random_edits(&mesh, LEVEL, 40, seed));
        writer
            .tick(&mut store, start + FLUSH_INTERVAL * second)
            .unwrap();
        assert_ne!(writer.phase(), "idle", "second {second}: the tick flushes");
        let cut = map.clone();
        while !writer.is_idle() {
            seed += 1;
            apply_dabs(&mut store, &mut map, &random_edits(&mesh, LEVEL, 6, seed));
            for key in check.iter().step_by(5) {
                store.evaluate(*key).unwrap();
            }
            if seed % 2 == 0 {
                store.evict_all().unwrap();
            }
            past_cut |= check.iter().any(|key| store.file.is_past_cut(key));
            writer.step_blocking(&mut store).unwrap();
        }
        points.push((writer.durable_save_point().unwrap(), cut));
    }
    assert_eq!(store.current_op(), Some(op), "one op throughout");
    assert!(
        past_cut,
        "dabs after a cut reached the file before its save point"
    );
    writer.close(&mut store, Instant::now()).unwrap();
    drop(store);
    for (offset, cut) in &points {
        let file = ProjectFile::open_at(&path, *offset).unwrap();
        assert_eq!(
            file.head(),
            Some(op),
            "save point {offset}: the stroke's op"
        );
        let mut at_cut = TileStore::open(file, roomy()).unwrap();
        let dense = DenseEvaluation::new(&mesh, level(LEVEL), cut).unwrap();
        assert_positions(&mut at_cut, &dense, &check, &format!("save point {offset}"));
    }
    let file = ProjectFile::open(&path).unwrap();
    let mut closed = TileStore::open(file, roomy()).unwrap();
    let dense = DenseEvaluation::new(&mesh, level(LEVEL), &map).unwrap();
    assert_positions(&mut closed, &dense, &check, "the closed file");
}

/// A crash simulation under a tight budget.
///
/// Strokes, with edits, evaluations, write-backs and refits of the spatial index between every
/// phase of each flush. Every save point equals its flush's snapshot, and the file cut at every
/// record boundary opens at the last save point before the cut, with bounds that load as the
/// index built from its tiles: from the file alone (nothing evaluated) once a save point holds
/// them, as the index is refitted before every stroke ends.
#[test]
fn every_save_point_is_its_snapshot_and_every_cut_opens_at_one() {
    /// The finest level edited and checked: coarse, so every cut can be checked in full.
    const LEVEL: u8 = 5;
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let check = tiles(&mesh, 1, LEVEL);
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, tight());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut index = SpatialIndex::build(&mut store, level(LEVEL), |_, _| {}).unwrap();
    let indexed = tiles(&mesh, LEVEL, LEVEL);
    let mut map = CorrectionMap::new();
    let mut points = Vec::new();
    let mut seed = 200;
    for _ in 0..5 {
        seed += 1;
        apply(&mut store, &mut map, &random_edits(&mesh, LEVEL, 40, seed));
        index.update(&mut store, &indexed).unwrap();
        writer.end_stroke(&mut store, Instant::now()).unwrap();
        let snapshot = map.clone();
        while !writer.is_idle() {
            seed += 1;
            apply(&mut store, &mut map, &random_edits(&mesh, LEVEL, 6, seed));
            for key in check.iter().step_by(5) {
                store.evaluate(*key).unwrap();
            }
            // Bounds of a state past the flush's cut: never in its save point.
            index.update(&mut store, &indexed).unwrap();
            writer.step_blocking(&mut store).unwrap();
        }
        points.push((writer.durable_save_point().unwrap(), snapshot));
    }
    drop(writer);
    drop(store);
    let ends = ProjectFile::record_ends(&path);
    // Each save point's state, from where its record ends.
    let points: Vec<(u64, CorrectionMap)> = points
        .into_iter()
        .map(|(offset, snapshot)| (*ends.iter().find(|&&end| end > offset).unwrap(), snapshot))
        .collect();
    let bytes = std::fs::read(&path).unwrap();
    let empty = CorrectionMap::new();
    let cut_path = dir.path("cut.bcrf");
    for &cut in &ends {
        let want = points
            .iter()
            .rev()
            .find(|(end, _)| *end <= cut)
            .map_or(&empty, |(_, snapshot)| snapshot);
        std::fs::write(&cut_path, &bytes[..usize::try_from(cut).unwrap()]).unwrap();
        let file = ProjectFile::open(&cut_path).unwrap();
        let mut reopened = TileStore::open(file, roomy()).unwrap();
        let dense = DenseEvaluation::new(&mesh, level(LEVEL), want).unwrap();
        let what = format!("cut at {cut}");
        assert_positions(&mut reopened, &dense, &check, &what);
        let stats = assert_loads_as_built(&mut reopened, level(LEVEL), &what);
        let saved = points.first().is_some_and(|(end, _)| *end <= cut);
        assert_eq!(
            stats.evaluated == 0,
            saved,
            "{what}: every bound stored once a save point holds them, none before ({stats:?})"
        );
    }
    assert!(ends.len() > 100, "{} record boundaries", ends.len());
}

/// A store save is refused while a recovery writer streams from it, writing nothing; once the
/// writer is closed, the save goes through.
#[test]
fn a_store_save_is_refused_while_a_writer_streams() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let writer = RecoveryWriter::new(&mut store).unwrap();
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 101),
    );
    let len = store.working_len();
    let saved = store.save();
    assert!(matches!(saved, Err(CacheError::Streaming)), "{saved:?}");
    assert_eq!(store.working_len(), len, "nothing written");
    writer.close(&mut store, Instant::now()).unwrap();
    store.save().unwrap();
}

/// Broken slots are left out of a flush and reported once; every other tile still flushes.
///
/// One slot is dirty with no op, another holds no valid tile. Every other tile reaches the save
/// point, and the broken ones are dropped from the flush mark, so later flushes never stall on
/// them.
#[test]
fn a_broken_slot_is_reported_and_the_rest_still_flush() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let keys: Vec<TileKey> = tiles(&mesh, TOP, TOP).into_iter().step_by(9).collect();
    let (no_op, invalid) = (keys[1], keys[3]);
    // The oracle holds the edits of the good tiles only: the broken ones never reach the file.
    let (mut map, mut lost) = (CorrectionMap::new(), CorrectionMap::new());
    for (n, key) in keys.iter().enumerate() {
        let c = Vec3::new(0.0, 0.0, 0.01 * f32::from(u16::try_from(n + 1).unwrap()));
        let oracle = if [no_op, invalid].contains(key) {
            &mut lost
        } else {
            &mut map
        };
        edit_interior(&mut store, oracle, *key, c);
    }
    // The bounds kept describe the broken tiles' edits, which never reach the file.
    SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    store.unsaved.remove(&no_op);
    let id = store.corrections.find(&invalid).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;

    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(writer.flushes(), 1, "the flush completed");
    let mut skipped = writer.take_skipped();
    skipped.sort_unstable();
    let mut want = vec![no_op, invalid];
    want.sort_unstable();
    assert_eq!(skipped, want, "both broken slots reported");
    assert_eq!(writer.take_skipped(), [], "reported once");
    // Not retried: the next stroke end has nothing to flush, and never stalls on them.
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(writer.take_skipped(), [], "dropped from the mark");
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
    let file = ProjectFile::open_at(&path, offset).unwrap();
    let mut saved = TileStore::open(file, roomy()).unwrap();
    let stats = assert_loads_as_built(&mut saved, level(TOP), "the broken tiles left out");
    assert!(stats.evaluated > 0, "their bounds unknown: {stats:?}");
}

/// A tile a flush leaves out is lost as a save loses one: the store stops reading its lost edit
/// and drops what was evaluated from it.
///
/// After the flush that leaves the broken tiles out, an index is built and an unrelated edit
/// refitted: the bounds recorded then, flushed over the broken tiles' unknown markers, must be
/// those of the state the save point holds, without the lost edits.
#[test]
fn a_tile_left_out_of_a_flush_drops_what_was_evaluated_from_it() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let keys: Vec<TileKey> = top.iter().copied().step_by(9).collect();
    let (no_op, invalid) = (keys[1], keys[3]);
    let mut map = CorrectionMap::new();
    for (n, key) in keys.iter().enumerate() {
        let c = Vec3::new(0.0, 0.0, 0.01 * f32::from(u16::try_from(n + 1).unwrap()));
        let mut lost = CorrectionMap::new();
        let oracle = if [no_op, invalid].contains(key) {
            &mut lost
        } else {
            &mut map
        };
        edit_interior(&mut store, oracle, *key, c);
    }
    SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    let wave = wave_now(&mut store, &[no_op, invalid]);
    store.unsaved.remove(&no_op);
    let id = store.corrections.find(&invalid).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
    assert!(wave.current(&store, &no_op).is_some(), "current before");
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(writer.take_skipped().len(), 2, "both left out");
    assert!(
        [no_op, invalid]
            .iter()
            .all(|key| wave.current(&store, key).is_none()),
        "bounds from before the flush that lost the tiles are not used"
    );
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "the broken tiles at the state the save point holds",
    );
    // An index built now, and a refit after an unrelated edit, record what the store evaluates.
    let mut index = SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    edit_interior(&mut store, &mut map, keys[0], Vec3::new(0.0, 0.02, 0.0));
    index.update(&mut store, &top).unwrap();
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
    let file = ProjectFile::open_at(&path, offset).unwrap();
    let mut saved = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut saved, level(TOP), "flushed after the broken tiles");
}

/// A tile left out of a flush while a pin holds its slot is filled again in place.
///
/// The slot is refilled from the tile's last record: an edit after the pin goes starts from what
/// the file holds, never from the lost edit (whose quantization scale it would otherwise carry
/// into the next save point).
#[test]
fn a_pinned_tile_left_out_of_a_flush_is_reloaded_in_place() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let keys: Vec<TileKey> = top.iter().copied().step_by(9).collect();
    let no_op = keys[1];
    let mut map = CorrectionMap::new();
    for (n, key) in keys.iter().enumerate() {
        let c = Vec3::new(0.0, 0.0, 0.01 * f32::from(u16::try_from(n + 1).unwrap()));
        let mut lost = CorrectionMap::new();
        let oracle = if *key == no_op { &mut lost } else { &mut map };
        edit_interior(&mut store, oracle, *key, c);
    }
    // A large lost edit, so its quantization scale would show.
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        no_op,
        Vec3::new(0.0, 0.0, 0.5),
    );
    let pin = store.pin(&[no_op]).unwrap();
    store.unsaved.remove(&no_op);
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(writer.take_skipped(), [no_op], "left out");
    let mut index = SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    edit_interior(&mut store, &mut map, keys[0], Vec3::new(0.0, 0.02, 0.0));
    index.update(&mut store, &top).unwrap();
    drop(pin);
    // A small edit at the lost edit's vertex, after the pin goes.
    edit_interior(&mut store, &mut map, no_op, Vec3::new(0.0, 0.0, 0.001));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
    let file = ProjectFile::open_at(&path, offset).unwrap();
    let mut saved = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut saved, level(TOP), "flushed after the pinned lost tile");
}

/// A broken slot lost to a roll-over while a pin holds it is filled again in place.
///
/// It is refilled from the new file (which lacks the tile): no longer broken, and an edit after
/// the pin goes starts from there.
#[test]
fn a_pinned_broken_slot_lost_to_a_roll_over_is_reloaded_in_place() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let top = tiles(&mesh, TOP, TOP);
    let (broken, kept) = (top[5], top[30]);
    let mut want = CorrectionMap::new();
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        broken,
        Vec3::new(0.0, 0.0, 0.5),
    );
    edit_interior(&mut store, &mut want, kept, Vec3::new(0.25, 0.0, 0.0));
    let pin = store.pin(&[broken]).unwrap();
    let id = store.corrections.find(&broken).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
    let meta = store.file.meta().clone();
    let fresh = ProjectFile::create_like(&dir.path("fresh.bcrf"), meta, &store.file).unwrap();
    let filled = store.fill_roll_over(fresh).unwrap();
    let (old, lost) = store.roll_over(filled).unwrap();
    drop(old);
    assert_eq!(lost, [broken], "listed as lost");
    assert_eq!(
        store
            .corrections
            .is_dirty(store.corrections.find(&broken).unwrap()),
        Some(false),
        "kept, clean, while pinned"
    );
    drop(pin);
    edit_interior(&mut store, &mut want, broken, Vec3::new(0.0, 0.0, 0.001));
    store.save().unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("fresh.bcrf")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut reopened,
        &dense(&mesh, &want),
        &tiles(&mesh, 1, TOP),
        "the edit after the pin, from the new file's state",
    );
}

/// A copy that fails to encode while its live slot is valid loses nothing: the failure was
/// transient.
///
/// Two tiles' copies fail as a transient compression error would; one is edited after the
/// snapshot, the other not. Both are reported as unprotected by that flush's save point, both
/// slots stay dirty, and the next flush writes both, with every edit.
#[test]
fn a_failed_encode_of_a_valid_slot_is_retried_by_the_next_flush() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let (edited, unchanged) = (top[3], top[40]);
    let mut map = CorrectionMap::new();
    edit_interior(&mut store, &mut map, edited, Vec3::new(0.0, 0.0, 0.01));
    edit_interior(&mut store, &mut map, unchanged, Vec3::new(0.0, 0.01, 0.0));
    writer.fail_encodes(2);
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    // After the snapshot, before its reply is handled.
    dab_interior_at(
        &mut store,
        &mut map,
        edited,
        (9, 4),
        Vec3::new(0.02, 0.0, 0.0),
    );
    settle(&mut writer, &mut store);
    let mut skipped = writer.take_skipped();
    skipped.sort_unstable();
    assert_eq!(
        skipped,
        [edited, unchanged],
        "both unprotected by that save point"
    );
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    assert_eq!(writer.take_skipped(), [], "both flush");
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
}

/// A flush whose copy of a tile fails to encode names unknown bounds for whatever the tile's
/// at-cut edit could change, though the live slot, edited since, is kept.
///
/// The index was refitted over the edit before the flush, so the bounds the flush appended at
/// its snapshot describe an edit its save point then lacks.
#[test]
fn a_failed_encode_leaves_no_stale_bound_in_its_save_point() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let mut index = SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    settle(&mut writer, &mut store);
    let map = CorrectionMap::new();
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        top[5],
        Vec3::new(0.0, 0.0, 0.25),
    );
    index.update(&mut store, &top).unwrap();
    writer.fail_encodes(1);
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    // Edited again before the reply: the slot is kept, and the save point lacks both edits.
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        top[5],
        Vec3::new(0.0, 0.0, 0.3),
    );
    settle(&mut writer, &mut store);
    assert_eq!(
        writer.take_skipped(),
        [top[5]],
        "unprotected by that save point"
    );
    let offset = writer.durable_save_point().unwrap();
    drop(writer);
    drop(store);
    assert_save_point(&path, offset, &mesh, &map);
    let file = ProjectFile::open_at(&path, offset).unwrap();
    let mut saved = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut saved, level(TOP), "the save point without the edit");
}

/// A roll-over to a new recovery file carries every edit across, dirty or only in the file, and
/// lists the one tile whose damaged record it cannot read; a flush in flight is abandoned.
#[test]
fn a_roll_over_carries_every_edit_and_lists_what_it_cannot_read() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 80, 111));
    writer.finish(&mut store, Instant::now()).unwrap();
    // Some edits only in the file (written back), some dirty.
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 40, 112));
    store.evict_all().unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 40, 113));
    let dirty: Vec<TileKey> = store.unsaved.keys().copied().collect();
    assert!(!dirty.is_empty(), "some tiles dirty");
    // One tile only in the file, its record damaged by an outside write.
    let damaged = *store
        .file
        .keys()
        .find(|key| store.corrections.find(key).is_none())
        .unwrap();
    damage(&path, store.file.locate(&damaged).unwrap());

    let meta = store.file.meta().clone();
    let fresh = ProjectFile::create_like(&dir.path("fresh.bcrf"), meta, &store.file).unwrap();
    let generation = store.generation();
    let filled = store.fill_roll_over(fresh).unwrap();
    let (old, lost) = store.roll_over(filled).unwrap();
    drop(old);
    assert_eq!(lost, [damaged], "the damaged tile is lost, and listed");
    assert!(
        store.unsaved.is_empty(),
        "every dirty slot is in the new file"
    );
    assert_ne!(
        store.generation(),
        generation,
        "a flush in flight is abandoned"
    );
    writer.close(&mut store, Instant::now()).unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("fresh.bcrf")).unwrap();
    assert!(!file.contains(&damaged), "left out");
    for key in &dirty {
        assert!(file.contains(key), "{key:?}: a dirty slot carried across");
    }
}

/// A tile lost to a roll-over drops what was evaluated from its lost edit.
///
/// Afterwards the store evaluates the tile as no file it reads holds the edit, and an index
/// refitted then saves bounds of that state.
#[test]
fn a_tile_lost_to_a_roll_over_drops_what_was_evaluated_from_it() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let top = tiles(&mesh, TOP, TOP);
    let (lost, kept) = (top[5], top[30]);
    let mut want = CorrectionMap::new();
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        lost,
        Vec3::new(0.0, 0.0, 0.25),
    );
    edit_interior(&mut store, &mut want, kept, Vec3::new(0.25, 0.0, 0.0));
    store.save().unwrap();
    let mut index = SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    let wave = wave_now(&mut store, &[lost]);
    // The clean slot goes; the evaluations made from it stay cached.
    evict(&mut store, &lost);
    damage(&path, store.file.locate(&lost).unwrap());
    let meta = store.file.meta().clone();
    let fresh = ProjectFile::create_like(&dir.path("fresh.bcrf"), meta, &store.file).unwrap();
    let filled = store.fill_roll_over(fresh).unwrap();
    assert!(wave.current(&store, &lost).is_some(), "current before");
    let (old, lost_tiles) = store.roll_over(filled).unwrap();
    drop(old);
    assert_eq!(lost_tiles, [lost], "listed as lost");
    assert!(
        wave.current(&store, &lost).is_none(),
        "bounds from before the roll-over that lost the tile are not used"
    );
    assert_positions(
        &mut store,
        &dense(&mesh, &want),
        &tiles(&mesh, 1, TOP),
        "the lost tile and what reads it, without the lost edit",
    );
    index.update(&mut store, &top).unwrap();
    store.save().unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("fresh.bcrf")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut reopened, level(TOP), "saved after the roll-over");
}

/// A broken slot lost to a roll-over is dropped: the store reads the tile as the new file holds
/// it (without the lost edit), never from the broken slot.
#[test]
fn a_broken_slot_lost_to_a_roll_over_is_dropped() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let top = tiles(&mesh, TOP, TOP);
    let (broken, kept) = (top[5], top[30]);
    let mut want = CorrectionMap::new();
    edit_interior(
        &mut store,
        &mut CorrectionMap::new(),
        broken,
        Vec3::new(0.0, 0.0, 0.25),
    );
    edit_interior(&mut store, &mut want, kept, Vec3::new(0.25, 0.0, 0.0));
    // Dirty, but holding no valid tile: it can't be carried across.
    let id = store.corrections.find(&broken).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
    let meta = store.file.meta().clone();
    let fresh = ProjectFile::create_like(&dir.path("fresh.bcrf"), meta, &store.file).unwrap();
    let filled = store.fill_roll_over(fresh).unwrap();
    let (old, lost) = store.roll_over(filled).unwrap();
    drop(old);
    assert_eq!(lost, [broken], "listed as lost");
    assert!(
        store.corrections.find(&broken).is_none(),
        "its slot dropped"
    );
    assert_positions(
        &mut store,
        &dense(&mesh, &want),
        &tiles(&mesh, 1, TOP),
        "the broken tile as the new file holds it",
    );
}

/// A clean resident tile whose record no longer reads is carried across a roll-over from its
/// slot, labelled with the op of the record the slot holds: never a newer edit's, which never
/// touched it.
#[test]
fn a_roll_over_labels_an_unreadable_clean_tile_with_its_own_op() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("recovery.bcrf");
    let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let (t, later) = (top[3], top[20]);
    let mut map = CorrectionMap::new();
    let own = edit_interior(&mut store, &mut map, t, Vec3::new(0.0, 0.0, 0.25));
    writer.finish(&mut store, Instant::now()).unwrap();
    let slot = store.corrections.find(&t).unwrap();
    assert_eq!(
        store.corrections.is_dirty(slot),
        Some(false),
        "flushed: clean"
    );
    let newest = edit_interior(&mut store, &mut map, later, Vec3::new(0.0, 0.25, 0.0));
    assert!(newest > own, "a newer edit, of another tile");
    damage(&path, store.file.locate(&t).unwrap());
    assert!(
        store.file.read_tile_record(&t).is_err(),
        "t's record is damaged"
    );

    let meta = store.file.meta().clone();
    let fresh = ProjectFile::create_like(&dir.path("fresh.bcrf"), meta, &store.file).unwrap();
    let filled = store.fill_roll_over(fresh).unwrap();
    let (old, lost) = store.roll_over(filled).unwrap();
    drop(old);
    assert_eq!(lost, [], "nothing lost: the slot holds t");
    let (op, _) = store.file.read_displacement(&t).unwrap().unwrap();
    assert_eq!(op, own, "labelled with its own op");
    store.evict_all().unwrap();
    assert_positions(&mut store, &dense(&mesh, &map), &top, "carried across");
    writer.close(&mut store, Instant::now()).unwrap();
}

/// With a tick every 100 ms during a long stroke, the peak snapshot is the tiles dirtied per
/// flush interval, not the stroke.
///
/// One op dabs `PER_TICK` fresh tiles per tick until it has dirtied every top-level tile. Ticked,
/// each flush snapshots about a second's worth of tiles, and the peak bytes in flight to the
/// writer stay within an interval's dabs. The same stroke with no tick snapshots every tile it
/// dirtied at its end, all at once.
#[test]
fn ticks_bound_a_long_strokes_snapshot_to_the_tiles_of_one_interval() {
    /// Fresh tiles each tick dabs.
    const PER_TICK: usize = 2;
    /// The time between ticks.
    const TICK: Duration = Duration::from_millis(100);
    /// The bytes of one tile's snapshot: a correction slot's copy.
    const SLOT: usize = size_of::<CorrectionSlot>();
    let ticks_per_interval =
        usize::try_from(FLUSH_INTERVAL.as_millis() / TICK.as_millis()).unwrap();
    let mesh = mesh();
    let top = tiles(&mesh, TOP, TOP);
    assert!(
        top.len() >= 4 * ticks_per_interval * PER_TICK,
        "the stroke spans at least four intervals: {} tiles",
        top.len()
    );
    let mut peaks = Vec::new();
    for ticked in [true, false] {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "recovery.bcrf", &mesh, roomy());
        let mut writer = RecoveryWriter::new(&mut store).unwrap();
        let mut map = CorrectionMap::new();
        let start = Instant::now();
        store.begin_op().unwrap();
        for (n, keys) in (1..).zip(top.chunks(PER_TICK)) {
            for (k, key) in keys.iter().enumerate() {
                let c = Vec3::new(0.0, 1e-3 * f32::from(u8::try_from(k + 1).unwrap()), 0.0);
                dab_interior(&mut store, &mut map, *key, c);
            }
            if ticked {
                writer.tick(&mut store, start + TICK * n).unwrap();
                settle(&mut writer, &mut store);
            }
        }
        writer
            .end_stroke(&mut store, start + FLUSH_INTERVAL * 100)
            .unwrap();
        settle(&mut writer, &mut store);
        assert!(
            store.unsaved.is_empty(),
            "ticked {ticked}: every edit is in the file"
        );
        peaks.push(writer.peak_snapshot_bytes());
    }
    let (ticked, unticked) = (peaks[0], peaks[1]);
    assert!(ticked > 0, "the ticked stroke snapshotted something");
    assert!(
        ticked <= ticks_per_interval * PER_TICK * SLOT,
        "ticked: the peak, {} tiles, is at most an interval's dabs ({} tiles)",
        ticked / SLOT,
        ticks_per_interval * PER_TICK
    );
    assert_eq!(
        unticked,
        top.len() * SLOT,
        "never ticked: one snapshot of every tile the stroke dirtied"
    );
}
