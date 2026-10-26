//! Prefetching correction tiles: used without reading the file, never at the cost of an
//! evaluated chunk or a write-back.

use super::*;

/// The bytes of `key`'s latest record in the working file, decoded into a slot (see
/// [`CorrectionSlot::tile_bytes`]).
fn slot_in_file(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let (_, tile) = store.file.read_displacement(key).unwrap().unwrap();
    CorrectionSlot::from_tile(&tile).tile_bytes()
}

#[test]
fn prefetched_tiles_are_used_without_reading_the_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map) = store_on_disk(&dir, "work.bcp", &mesh, roomy(), 300, 12);
    let on_disk: Vec<TileKey> = store.file.keys().copied().collect();
    assert!(on_disk.len() > 50, "{} tiles on disk", on_disk.len());
    let absent = tiles(&mesh, 1, TOP)
        .into_iter()
        .find(|key| !store.file.contains(key))
        .unwrap();
    let mut wanted = on_disk.clone();
    // An absent tile (zero) and a repeated one aren't read.
    wanted.push(absent);
    wanted.push(on_disk[0]);
    assert_eq!(store.prefetch(&wanted).unwrap(), on_disk.len(), "queued");
    assert_eq!(
        store.prefetch(&on_disk).unwrap(),
        0,
        "pending requests aren't queued twice"
    );
    assert_eq!(drain_prefetch(&mut store), on_disk.len(), "all installed");
    assert_eq!(store.prefetch(&on_disk).unwrap(), 0, "resident: not queued");

    store.log = EvalLog::default();
    for key in &on_disk {
        store.correction(*key).unwrap();
        assert_eq!(
            slot_resident(&store, key),
            slot_in_file(&store, key),
            "{key:?}: the prefetched slot is the file's tile"
        );
    }
    assert_eq!(store.log.file_reads, [], "no read on the owner's thread");
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &tiles(&mesh, 1, TOP), "prefetched");
    assert!(
        store.log.file_reads.contains(&absent),
        "the absent tile was looked up"
    );
    assert!(
        store
            .log
            .file_reads
            .iter()
            .all(|key| !store.file.contains(key)),
        "and only absent tiles: every tile on disk was prefetched"
    );
}

#[test]
fn a_tile_edited_after_its_prefetch_keeps_the_edit() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = store_on_disk(&dir, "work.bcp", &mesh, roomy(), 300, 13);
    let (face, i, j) = (FaceId::new(4), 128 + 20, 64 + 30);
    let rewritten = tile_at(face, i, j);
    let (i2, j2) = (i + 64, j);
    let resident = tile_at(face, i2, j2);
    assert!(
        store.file.contains(&rewritten) && store.file.contains(&resident),
        "both on disk"
    );
    let edit = |at_i, at_j| Edit {
        face,
        level: level(TOP),
        i: at_i,
        j: at_j,
        c: Vec3::new(0.003, -0.002, 0.001),
    };

    // The workers are held back, so their results arrive only after the edits.
    store.prefetch.pause(true);
    assert_eq!(store.prefetch(&[rewritten, resident]).unwrap(), 2);
    apply(&mut store, &mut map, &[edit(i, j), edit(i2, j2)]);
    // `rewritten` is written back (its record moves) and leaves the pool; `resident` stays,
    // dirty, with its record unchanged.
    store.correction(resident).unwrap();
    let pin = store.pin(&[resident]).unwrap();
    store.evict_all().unwrap();
    assert!(store.corrections.find(&rewritten).is_none(), "written back");
    let id = store.corrections.find(&resident).unwrap();
    assert_eq!(store.corrections.is_dirty(id), Some(true), "dirty");
    store.prefetch.pause(false);
    assert_eq!(drain_prefetch(&mut store), 0, "both results were stale");

    assert_eq!(
        store.corrections.find(&resident),
        Some(id),
        "the edited tile kept its slot"
    );
    store.correction(rewritten).unwrap();
    assert_eq!(
        slot_resident(&store, &rewritten),
        slot_in_file(&store, &rewritten),
        "the edited tile, read again"
    );
    drop(pin);
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &[rewritten, resident], "edited");
}

#[test]
fn prefetch_never_takes_an_evaluated_chunk_or_writes_a_tile_back() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = store_on_disk(&dir, "work.bcp", &mesh, config(10, 4), 300, 14);
    let all = tiles(&mesh, 1, TOP);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    let chunk = store.corrections.chunk_bytes();
    assert!(!store.budget.fits(chunk), "the budget is spent");
    // Every correction slot in use and dirty: none can be had without a write-back.
    let mut absent: Vec<TileKey> = all
        .iter()
        .filter(|key| !store.file.contains(key) && store.corrections.find(key).is_none())
        .copied()
        .collect();
    while store.corrections.has_free_slot() {
        store.correction(absent.pop().unwrap()).unwrap();
    }
    dirty_every_resident_correction(&mut store, &all);
    assert!(
        store.evaluated.has_unpinned_chunk(),
        "an evaluated chunk could be taken"
    );
    let on_disk: Vec<TileKey> = all
        .iter()
        .filter(|key| store.file.contains(key) && store.corrections.find(key).is_none())
        .copied()
        .collect();
    let (written, corrections, evaluated) = (
        store.file.len(),
        store.corrections.bytes(),
        store.evaluated.bytes(),
    );
    let dirty = store.unsaved.len();
    assert_eq!(store.prefetch(&on_disk[..1]).unwrap(), 1, "queued");
    assert_eq!(drain_prefetch(&mut store), 0, "dropped: no slot to spare");
    assert_eq!(store.file.len(), written, "nothing written back");
    assert_eq!(
        (store.corrections.bytes(), store.evaluated.bytes()),
        (corrections, evaluated),
        "no chunk changed hands"
    );
    assert_eq!(store.unsaved.len(), dirty, "every dirty tile stays");

    // Clean now: a prefetched tile takes a clean tile's slot, still without writing.
    store.save().unwrap();
    let saved = store.file.len();
    assert_eq!(store.prefetch(&on_disk[1..2]).unwrap(), 1, "queued");
    assert_eq!(drain_prefetch(&mut store), 1, "installed in a clean slot");
    assert_eq!(store.file.len(), saved, "nothing written");
    assert_eq!(
        (store.corrections.bytes(), store.evaluated.bytes()),
        (corrections, evaluated),
        "still no chunk changed hands"
    );
}

#[test]
fn dropping_a_store_with_queued_work_returns_promptly() {
    let (done, finished) = mpsc::channel();
    // The store isn't `Send`, so it lives and dies on a thread of its own, and a hang there
    // fails the test here instead of hanging it.
    thread::spawn(move || {
        let dir = TestDir::new();
        let mesh = mesh();
        for paused in [true, false] {
            let name = format!("paused-{paused}.bcp");
            let (mut store, _) = store_on_disk(&dir, &name, &mesh, roomy(), 300, 14);
            let on_disk: Vec<TileKey> = store.file.keys().copied().collect();
            store.prefetch.pause(paused);
            let queued = store.prefetch(&on_disk).unwrap();
            let workers = store.prefetch.liveness();
            let start = Instant::now();
            drop(store);
            let took = start.elapsed();
            assert!(
                workers.upgrade().is_none(),
                "paused {paused}: the workers were joined"
            );
            assert!(
                took < Duration::from_secs(1),
                "paused {paused}: dropping with {queued} reads queued took {took:?}"
            );
            let reopened = ProjectFile::open(&dir.path(&name));
            assert!(
                reopened.is_ok(),
                "paused {paused}: the file is free: {reopened:?}"
            );
        }
        // Removed before the test is told it is done: the test's process may exit at once,
        // ending this thread before its drops run.
        drop(dir);
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("the store dropped in time (a panic on its thread shows above)");
}
