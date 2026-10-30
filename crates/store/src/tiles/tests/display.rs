//! The display layer the store keeps: through save and open, under edits (against a rebuild of
//! every display tile, across faces), and damaged.

use geom::DisplayTile;

use super::*;
use crate::bounds::Kept;
use crate::deps::DepGraph;
use crate::display::readers;
use crate::stamp::Stamp;
use crate::tiles::display::write_kept;

/// Every display tile key of the tests' mesh, in key order.
fn display_keys(mesh: &BaseMesh) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = tiles(mesh, 0, TOP - 1)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect();
    keys.sort_unstable();
    keys
}

/// Whether two display tiles hold the same bytes and bound.
fn same(a: &DisplayTile, b: &DisplayTile) -> bool {
    a.encoded() == b.encoded() && a.error().to_bits() == b.error().to_bits()
}

#[test]
fn display_tiles_round_trip_through_save_and_open() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 40, 3));
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut reopened = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    let keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    let stored = reopened.stored_display(&keys).unwrap();
    for (tile, back) in layer.iter().zip(stored) {
        let back = back.unwrap_or_else(|| panic!("{:?} kept", tile.key()));
        assert_eq!(&back, tile, "{:?} bit for bit", tile.key());
    }
}

#[test]
fn an_edit_makes_unknown_every_display_tile_it_changes_across_faces() {
    let mesh = mesh();
    let keys = display_keys(&mesh);
    for seed in 0..6_u64 {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
        let before = display_layer(&mut store);
        store.record_display(&before).unwrap();
        let edits = random_edits(&mesh, TOP, 1 + u32::try_from(seed % 3).unwrap(), 50 + seed);
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &edits);
        let stored = store.stored_display(&keys).unwrap();
        let after = display_layer(&mut store);
        let after: BTreeMap<TileKey, DisplayTile> =
            after.into_iter().map(|t| (*t.key(), t)).collect();
        let before: BTreeMap<TileKey, DisplayTile> =
            before.into_iter().map(|t| (*t.key(), t)).collect();
        let mut unknown = BTreeSet::new();
        let mut changed_faces = BTreeSet::new();
        for (key, kept) in keys.iter().zip(&stored) {
            let changed = !same(&before[key], &after[key]);
            if changed {
                changed_faces.insert(key.face());
            }
            match kept {
                None => {
                    unknown.insert(*key);
                }
                Some(kept) => {
                    assert!(
                        !changed,
                        "seed {seed}: {key:?} changed but the store still holds it"
                    );
                    assert!(same(kept, &before[key]), "seed {seed}: {key:?} as recorded");
                }
            }
        }
        // Every change made something unknown, and on more than one face where an edit sits on a
        // seam: the test reaches across faces.
        assert!(!unknown.is_empty(), "seed {seed}: something is unknown");
        // Every copy of each edited vertex is written, on every tile holding it.
        let edited: BTreeSet<TileKey> = edits
            .iter()
            .flat_map(|e| {
                mesh.topology()
                    .copies(e.face, e.level, e.i, e.j)
                    .iter()
                    .map(|copy| copy.key(e.level))
                    .collect::<Vec<_>>()
            })
            .collect();
        let levels: BTreeSet<Level> = (0..TOP).map(level).collect();
        let expected = readers(&mesh, &DepGraph::new(), &edited, &levels).unwrap();
        assert!(
            unknown.is_subset(&expected),
            "seed {seed}: only display readers of the edits are unknown"
        );
        assert!(
            changed_faces.len()
                <= unknown
                    .iter()
                    .map(TileKey::face)
                    .collect::<BTreeSet<_>>()
                    .len(),
            "seed {seed}: every face with a changed tile has an unknown one"
        );
    }
}

/// The finest-level corrections of `an_edit_makes_the_neighbours_display_tiles_unknown`'s
/// case `name`, on face 0: `(i, j, height)`.
///
/// The corner case is one vertex, a base vertex three faces share. The middle case is a broad
/// bump inside face 0's four middle finest tiles: no finest tile of a neighbour moves, and the
/// means it tilts survive quantization up to the coarsest levels, whose tents read across the
/// seams.
fn seam_edit(name: &str) -> Vec<(u32, u32, f32)> {
    if name == "corner" {
        return vec![(0, 0, 0.05)];
    }
    let middle = level(TOP).face_quads() / 2;
    let radius = 50_u16;
    let mut out = Vec::new();
    for j in middle - u32::from(radius)..=middle + u32::from(radius) {
        for i in middle - u32::from(radius)..=middle + u32::from(radius) {
            let d = |a: u32| f32::from(u16::try_from(a.abs_diff(middle)).unwrap());
            let r = (d(i) * d(i) + d(j) * d(j)).sqrt();
            out.push((i, j, 0.1 * (1.0 - r / f32::from(radius)).max(0.0)));
        }
    }
    out
}

/// An edit changes display tiles of neighbouring faces, and the store makes those unknown too.
///
/// Their normals' tent reads across the seam: at a corner (a base vertex three faces share),
/// and from the middle of a face, which reaches its neighbours' coarsest tiles only through the
/// mip (no finest tile of theirs moves).
#[test]
fn an_edit_makes_the_neighbours_display_tiles_unknown() {
    let mesh = mesh();
    let keys = display_keys(&mesh);
    for (name, faces) in [("corner", 3), ("middle", 2)] {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
        let before = display_layer(&mut store);
        store.record_display(&before).unwrap();
        store.begin_op().unwrap();
        for (i, j, height) in seam_edit(name) {
            store
                .set_correction(
                    FaceId::new(0),
                    level(TOP),
                    i,
                    j,
                    Vec3::new(0.0, 0.0, height),
                )
                .unwrap();
        }
        let stored = store.stored_display(&keys).unwrap();
        let after: BTreeMap<TileKey, DisplayTile> = display_layer(&mut store)
            .into_iter()
            .map(|t| (*t.key(), t))
            .collect();
        let mut changed_faces = BTreeSet::new();
        for ((key, kept), was) in keys.iter().zip(&stored).zip(&before) {
            if !same(was, &after[key]) {
                changed_faces.insert(key.face());
                assert!(kept.is_none(), "{name}: {key:?} changed and is unknown");
            }
        }
        assert!(
            changed_faces.len() >= faces,
            "{name}: tiles of {faces} faces changed: {changed_faces:?}"
        );
    }
}

#[test]
fn a_damaged_display_record_is_unknown_not_trusted() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    let victim = *layer[3].key();
    let loc = store.working().locate(&victim).unwrap();
    drop(store);
    damage(&dir.path("work.bcp"), loc);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut reopened = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    let keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    for (key, back) in keys.iter().zip(reopened.stored_display(&keys).unwrap()) {
        if *key == victim {
            assert!(back.is_none(), "{key:?}: damaged, so unknown");
        } else {
            assert!(back.is_some(), "{key:?}: intact");
        }
    }
}

/// A project file keeping a display layer under [`TOP`], every tile built and saved, at `name`
/// in `dir`, with the edits `seed` makes; returns the store's display tiles.
pub(super) fn saved_layer(
    dir: &TestDir,
    name: &str,
    mesh: &Arc<BaseMesh>,
    seed: u64,
) -> Vec<DisplayTile> {
    let mut store = new_store(dir, name, mesh, roomy().with_display(level(TOP)));
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(mesh, TOP, 20, seed),
    );
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    layer
}

/// Display keys the layer can't have, written into a file (a level at the deepest, a face past
/// the mesh), are ignored on open: they neither switch off invalidation nor count as tiles.
#[test]
fn display_keys_the_layer_cant_have_dont_switch_off_invalidation() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 41);
    let mut file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let deepest = Level::new(geom::MAX_LEVEL).unwrap();
    let hostile = [
        TileKey::display(FaceId::new(0), deepest, 0, 0).unwrap(),
        TileKey::display(FaceId::new(999), level(1), 0, 0).unwrap(),
    ];
    file.write_display_many(hostile.iter().map(|&key| (key, None)), Stamp::ZERO)
        .unwrap();
    file.save().unwrap();
    let mut store = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    let keys = display_keys(&mesh);
    assert!(
        store
            .stored_display(&keys)
            .unwrap()
            .iter()
            .all(Option::is_some),
        "the layer is known"
    );
    let edits = random_edits(&mesh, TOP, 3, 77);
    apply(&mut store, &mut CorrectionMap::new(), &edits);
    let edited: BTreeSet<TileKey> = edits.iter().flat_map(|e| e.keys(&mesh)).collect();
    let levels: BTreeSet<Level> = (0..TOP).map(level).collect();
    let reached = readers(&mesh, &DepGraph::new(), &edited, &levels).unwrap();
    assert!(!reached.is_empty(), "the edits reach display tiles");
    let reached: Vec<TileKey> = reached.into_iter().collect();
    for (key, kept) in reached.iter().zip(store.stored_display(&reached).unwrap()) {
        assert!(kept.is_none(), "{key:?} made unknown by the edits");
    }
}

/// A layer built from another finest level bounds another surface: a store keeping its layer
/// under another level reads none of it, and rebuilds it whole.
#[test]
fn a_layer_built_from_another_finest_level_is_unknown_and_rebuilt() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 43);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let other = level(TOP - 1);
    let mut store = TileStore::open(file, roomy().with_display(other)).unwrap();
    let keys: Vec<TileKey> = display_keys(&mesh)
        .into_iter()
        .filter(|k| k.level() < other)
        .collect();
    for (key, kept) in keys.iter().zip(store.stored_display(&keys).unwrap()) {
        assert!(kept.is_none(), "{key:?} built from level {TOP}: unknown");
    }
    assert_eq!(
        store.floor_size().0,
        0,
        "no floor tile from another surface"
    );
    assert!(store.has_jobs(), "rebuilt");
    let mut calls = 0;
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
        calls += 1;
        assert!(calls < 1000, "the rebuild ends");
    }
    let layer = crate::fixture::display_layer_at(&mut store, TOP - 1);
    let rebuilt: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    assert_eq!(rebuilt, keys, "the whole layer under level {}", TOP - 1);
    let stored = store.stored_display(&rebuilt).unwrap();
    let pairs = layer.iter().zip(stored);
    for (tile, kept) in pairs {
        assert_eq!(
            kept.as_ref(),
            Some(tile),
            "{:?} rebuilt, bit for bit",
            tile.key()
        );
    }
}

/// A file whose display layer an older build wrote is rebuilt whole on open.
///
/// The layer is version 2, from before the normal cones: the floor and the finer levels alike
/// are queued, and the rebuild gives the layer built from scratch.
///
/// Its records' lengths say they are known, so without the floor's check nothing would be
/// queued and the store would keep no floor.
#[test]
fn a_layer_an_older_version_wrote_is_rebuilt_whole() {
    let dir = TestDir::new();
    let mesh = mesh();
    let layer = saved_layer(&dir, "work.bcp", &mesh, 67);
    let mut file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let keys: Vec<TileKey> = file.display_keys().copied().collect();
    assert_eq!(keys.len(), layer.len(), "the file holds the whole layer");
    for key in &keys {
        let record = file.read_display(key).unwrap().unwrap().known().unwrap();
        let old = crate::file::version_2_record(*key, Some(&record));
        file.append_display_raw(key, &old).unwrap();
        assert_eq!(
            file.read_display(key).unwrap(),
            Some(crate::display::StoredDisplay::Unknown),
            "{key:?}: unread"
        );
    }
    file.save().unwrap();
    drop(file);
    let reopened = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut store = TileStore::open(reopened, roomy().with_display(level(TOP))).unwrap();
    assert!(store.has_jobs(), "the layer is queued for a rebuild");
    let mut calls = 0;
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
        calls += 1;
        assert!(calls < 1000, "the jobs end");
    }
    let layer_keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    let stored = store.stored_display(&layer_keys).unwrap();
    let pairs = layer.iter().zip(stored);
    for (tile, kept) in pairs {
        assert_eq!(kept.as_ref(), Some(tile), "{:?} rebuilt", tile.key());
    }
}

/// A damaged display record, or one built from another finest level, among the store's own is
/// rebuilt.
///
/// Each is found unknown (read on open, as a floor tile, or decoded by the tile above it), and
/// built; the layer ends as the one built from scratch.
#[test]
fn damaged_and_foreign_records_among_the_layer_are_rebuilt() {
    for foreign in [false, true] {
        let dir = TestDir::new();
        let mesh = mesh();
        saved_layer(&dir, "work.bcp", &mesh, 61);
        let mut file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
        let key = TileKey::display(FaceId::new(2), level(TOP - 1), 1, 1).unwrap();
        let record = file.read_display(&key).unwrap().unwrap().known().unwrap();
        if foreign {
            let other = crate::display::DisplayRecord {
                finest: level(TOP + 1),
                ..record
            };
            file.write_display_many([(key, Some(&other))], Stamp::ZERO)
                .unwrap();
            file.save().unwrap();
            drop(file);
        } else {
            file.save().unwrap();
            drop(file);
            damage(&dir.path("work.bcp"), {
                let opened = ProjectFile::open(&dir.path("work.bcp")).unwrap();
                opened.locate(&key).unwrap()
            });
        }
        let reopened = ProjectFile::open(&dir.path("work.bcp")).unwrap();
        let mut store = TileStore::open(reopened, roomy().with_display(level(TOP))).unwrap();
        // Make a tile above it unknown: its build reads the record and finds it unreadable.
        let above = TileKey::display(FaceId::new(2), level(TOP - 2), 0, 0).unwrap();
        store.display.set(above, None);
        let mut calls = 0;
        while store.has_jobs() || calls == 0 {
            store
                .run_jobs(Instant::now() + Duration::from_secs(60))
                .unwrap();
            calls += 1;
            assert!(calls < 1000, "the jobs end");
        }
        let layer = display_layer(&mut store);
        let keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
        let stored = store.stored_display(&keys).unwrap();
        let pairs = layer.iter().zip(stored);
        for (tile, kept) in pairs {
            assert_eq!(
                kept.as_ref(),
                Some(tile),
                "foreign {foreign}: {:?} current",
                tile.key()
            );
        }
    }
}

/// After a save by a store keeping no display layer, the file names no display tile.
///
/// That store can't keep a layer up to date, so one opened with a display layer again rebuilds
/// it rather than trusting tiles edits may have outdated.
#[test]
fn a_save_without_a_display_layer_names_no_display_tile() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 47);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut plain = TileStore::open(file, roomy()).unwrap();
    apply(
        &mut plain,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 3, 79),
    );
    plain.save().unwrap();
    drop(plain);
    let reopened = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut store = TileStore::open(reopened, roomy().with_display(level(TOP))).unwrap();
    let keys = display_keys(&mesh);
    for (key, kept) in keys.iter().zip(store.stored_display(&keys).unwrap()) {
        assert!(
            kept.is_none(),
            "{key:?}: unknown after a save that couldn't keep it"
        );
    }
}

/// [`readers`] against the display tiles a write really changes, found by building the whole
/// layer before and after random edits at every level, across faces.
///
/// Every changed tile must be among the readers (soundness, asserted). How many readers didn't
/// change is the listing's slack (its precision, printed): tiles made unknown and rebuilt to the
/// same bits.
#[test]
fn readers_hold_every_display_tile_a_write_changes() {
    let mesh = mesh();
    let levels: BTreeSet<Level> = (0..TOP).map(level).collect();
    let (mut listed, mut changed_total) = (0_usize, 0_usize);
    for seed in 0..10_u64 {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
        let before: BTreeMap<TileKey, DisplayTile> = display_layer(&mut store)
            .into_iter()
            .map(|t| (*t.key(), t))
            .collect();
        let edits = random_edits(&mesh, TOP, 1 + u32::try_from(seed % 3).unwrap(), 900 + seed);
        apply(&mut store, &mut CorrectionMap::new(), &edits);
        let edited: BTreeSet<TileKey> = edits.iter().flat_map(|e| e.keys(&mesh)).collect();
        let reached = readers(&mesh, &store.deps, &edited, &levels).unwrap();
        let after = display_layer(&mut store);
        let changed: Vec<TileKey> = after
            .iter()
            .filter(|t| !same(&before[t.key()], t))
            .map(|t| *t.key())
            .collect();
        assert!(
            !changed.is_empty(),
            "seed {seed}: the edits change the layer"
        );
        for key in &changed {
            assert!(
                reached.contains(key),
                "seed {seed}: {key:?} changed, not listed"
            );
        }
        listed += reached.len();
        changed_total += changed.len();
    }
    let precision = f64::from(u32::try_from(changed_total).unwrap())
        / f64::from(u32::try_from(listed).unwrap());
    eprintln!(
        "readers: {listed} listed, {changed_total} changed, precision {:.1} %",
        100.0 * precision
    );
}

/// A record read ahead stands in for the file's only while it is the file's latest: once the
/// tile is recorded again and saved, the store reads the new record, never the one read ahead.
#[test]
fn a_record_read_ahead_is_used_only_while_it_is_the_files_latest() {
    let dir = TestDir::new();
    let mesh = mesh();
    saved_layer(&dir, "work.bcp", &mesh, 53);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut store = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    let reached = TileKey::display(FaceId::new(0), level(TOP - 1), 0, 0).unwrap();
    store.read_around(&BTreeSet::from([reached]));
    let began = Instant::now();
    let read = loop {
        store.take_read_ahead();
        if let Some(key) = store.read_ahead.read.tiles.keys().next().copied() {
            break key;
        }
        assert!(began.elapsed() < Duration::from_secs(10), "read ahead");
        thread::yield_now();
    };
    assert_ne!(read, reached, "the tiles around it, not itself");
    let from_file = crate::tiles::display::read_display(&store.file, store.base.as_ref(), &read)
        .unwrap()
        .unwrap();
    assert_eq!(
        store
            .read_ahead_record(&read)
            .cloned()
            .and_then(crate::display::StoredDisplay::known),
        Some(from_file),
        "the file's latest record, read ahead"
    );
    // Recorded again and saved: the file's latest record is another one.
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 54),
    );
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    assert!(
        store.read_ahead_record(&read).is_none(),
        "superseded: read again, not taken from what was read ahead"
    );
    let rebuilt = layer.iter().find(|tile| *tile.key() == read).unwrap();
    let stored = store.stored_display(&[read]).unwrap();
    assert_eq!(
        stored.first(),
        Some(&Some(rebuilt.clone())),
        "the new record"
    );
}

/// A record written through is never placed at an open cut: while a flush's cut is open the
/// records wait in memory, and go through once its save point is written.
#[test]
fn nothing_is_written_through_while_a_cut_is_open() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 61),
    );
    store.hold_display_waiting_to(0);
    let layer = display_layer(&mut store);
    store.open_cut();
    store.record_display(&layer).unwrap();
    assert_eq!(
        store.working().display_keys().count(),
        0,
        "nothing written at the cut"
    );
    assert!(store.display_waiting().0 > 0, "the records wait");
    store.write_save_point().unwrap();
    store.record_display(&layer[..1]).unwrap();
    assert_eq!(
        store.working().display_keys().count(),
        layer.len(),
        "the save point written, every record goes through"
    );
    assert_eq!(store.display_waiting().0, 0, "none waits");
}

/// The display records a flush or a save writes, compressed on the workers, are the bytes
/// written one by one.
///
/// Those of [`write_kept`] land in the file as the same bytes, in the same places, as records
/// compressed one by one on the calling thread ([`ProjectFile::write_display_many`]): who
/// compresses changes nothing written.
#[test]
fn records_compressed_on_the_workers_are_the_bytes_written_one_by_one() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 20, 67),
    );
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    // An unknown one among them, framed as unknown on both paths.
    let unknown = *layer[layer.len() / 2].key();
    store.display.set(unknown, None);
    let keys: Vec<TileKey> = store
        .display
        .unflushed_kept()
        .map(|(key, _)| *key)
        .collect();
    assert_eq!(keys.len(), layer.len(), "every record waits for a flush");
    let held: Vec<(TileKey, Option<&crate::display::DisplayRecord>)> = keys
        .iter()
        .map(|key| match store.display.kept(key) {
            Some(Kept::Known(record)) => (*key, Some(record)),
            _ => (*key, None),
        })
        .collect();
    assert_eq!(
        held.iter().filter(|(_, record)| record.is_none()).count(),
        1,
        "one is unknown"
    );
    let stamp = Stamp::ZERO.next();
    let mut serial =
        ProjectFile::create(&dir.path("serial.bcp"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let mut parallel = ProjectFile::create(
        &dir.path("parallel.bcp"),
        meta_for(&mesh),
        Arc::clone(&mesh),
    )
    .unwrap();
    let start = std::fs::metadata(dir.path("serial.bcp")).unwrap().len();
    assert_eq!(
        std::fs::metadata(dir.path("parallel.bcp")).unwrap().len(),
        start,
        "the files start alike"
    );
    serial
        .write_display_many(held.iter().copied(), stamp)
        .unwrap();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    write_kept(&mut parallel, None, &workers, &store.display, &keys, stamp).unwrap();
    for key in &keys {
        assert_eq!(serial.locate(key), parallel.locate(key), "{key:?} in place");
        assert_eq!(
            serial.read_tile_record(key).unwrap(),
            parallel.read_tile_record(key).unwrap(),
            "{key:?} the same record"
        );
    }
    drop((serial, parallel));
    let serial_bytes = std::fs::read(dir.path("serial.bcp")).unwrap();
    let parallel_bytes = std::fs::read(dir.path("parallel.bcp")).unwrap();
    let at = usize::try_from(start).unwrap();
    assert!(serial_bytes.len() > at, "the records were written");
    assert_eq!(
        serial_bytes.get(at..),
        parallel_bytes.get(at..),
        "the same bytes"
    );
}
