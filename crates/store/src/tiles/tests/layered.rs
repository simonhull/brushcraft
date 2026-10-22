//! Layered reads: a working file over a base (the project file as of its last save), against the
//! dense evaluation of every edit in both, and across a rebase onto a new save.

use super::*;
use crate::meta::{Meta, Role};

/// A project file at `name` in `dir` holding `edits`, made through a store and saved, then
/// reopened as a save would leave it: every record named by its last save point.
fn saved_project(
    dir: &TestDir,
    name: &str,
    mesh: &Arc<BaseMesh>,
    map: &mut CorrectionMap,
    edits: &[Edit],
) -> ProjectFile {
    let path = dir.path(name);
    let mut store = new_store(dir, name, mesh, roomy());
    apply(&mut store, map, edits);
    store.save().unwrap();
    drop(store);
    ProjectFile::open(&path).unwrap()
}

/// A new, empty recovery file at `name` in `dir` over `mesh`.
fn recovery(dir: &TestDir, name: &str, mesh: &Arc<BaseMesh>) -> ProjectFile {
    let meta = Meta::new(crate::fixture::PROJECT, Role::Recovery { base: None }, mesh);
    ProjectFile::create(&dir.path(name), meta, Arc::clone(mesh)).unwrap()
}

/// A displacement tile for `key` with every correction `(0, seed, -seed)`.
fn flat_tile(key: &TileKey, seed: f32) -> CorrectionTile {
    let n = key.side() as usize;
    CorrectionTile::from_values(key.side(), &vec![Vec3::new(0.0, seed, -seed); n * n]).unwrap()
}

/// The bytes of `tile` in a correction slot, as [`slot_resident`] reads a resident one.
fn slot_of(tile: &CorrectionTile) -> Vec<u8> {
    bytemuck::bytes_of(&CorrectionSlot::from_tile(tile)).to_vec()
}

/// Every tile of `keys`, evaluated, as bits.
fn positions(store: &mut TileStore, keys: &[TileKey]) -> Vec<Vec<[u32; 3]>> {
    keys.iter()
        .map(|key| {
            let got = store.evaluate(*key).unwrap();
            got.iter().map(|v| v.to_bits()).collect()
        })
        .collect()
}

#[test]
fn a_tile_only_in_the_base_reads_through() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 200, 21),
    );
    let saved: Vec<TileKey> = project.keys().copied().collect();
    assert!(saved.len() > 30, "{} tiles saved", saved.len());
    let working = recovery(&dir, "r.bcrf", &mesh);
    let mut store = TileStore::open(working, Some(project.base().unwrap()), roomy()).unwrap();
    for key in &saved {
        store.correction(*key).unwrap();
        let (_, tile) = project.read_displacement(key).unwrap().unwrap();
        assert_eq!(
            slot_resident(&store, key),
            slot_of(&tile),
            "{key:?}: the base's tile"
        );
    }
    assert_eq!(store.file.keys().count(), 0, "nothing in the working file");
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "through the base",
    );
}

#[test]
fn a_tile_in_both_reads_the_working_copy() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, 3);
    let (both, base_only) = (all[7], all[15]);
    let mut project =
        ProjectFile::create(&dir.path("p.bcrf"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let op = |seq| OpId::new(SITE, seq).unwrap();
    project
        .write_displacement(both, op(1), &flat_tile(&both, 0.25))
        .unwrap();
    project
        .write_displacement(base_only, op(2), &flat_tile(&base_only, 0.5))
        .unwrap();
    project.save().unwrap();
    let mut unsaved = recovery(&dir, "r.bcrf", &mesh);
    unsaved
        .write_displacement(both, op(3), &flat_tile(&both, -0.125))
        .unwrap();
    drop(unsaved);

    // Loaded on the owner's thread, and prefetched by the workers.
    for prefetched in [false, true] {
        let what = if prefetched { "prefetched" } else { "loaded" };
        let working = ProjectFile::open(&dir.path("r.bcrf")).unwrap();
        let mut store = TileStore::open(working, Some(project.base().unwrap()), roomy()).unwrap();
        if prefetched {
            assert_eq!(store.prefetch(&[both, base_only]).unwrap(), 2, "queued");
            assert_eq!(drain_prefetch(&mut store), 2, "installed");
        }
        store.log = EvalLog::default();
        for key in [both, base_only] {
            store.correction(key).unwrap();
        }
        assert_eq!(
            slot_resident(&store, &both),
            slot_of(&flat_tile(&both, -0.125)),
            "{what}: the working file's copy"
        );
        assert_eq!(
            slot_resident(&store, &base_only),
            slot_of(&flat_tile(&base_only, 0.5)),
            "{what}: the base's tile"
        );
        let reads = if prefetched { 0 } else { 2 };
        assert_eq!(store.log.file_reads.len(), reads, "{what}: owner reads");
    }
}

#[test]
fn evaluation_over_layered_sources_equals_dense_with_the_combined_edits() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 150, 22),
    );
    let project_bytes = std::fs::read(dir.path("p.bcrf")).unwrap();
    // Under pressure: dirty tiles are written back, and tiles of both layers read again.
    let working = recovery(&dir, "r.bcrf", &mesh);
    let mut store = TileStore::open(working, Some(project.base().unwrap()), tight()).unwrap();
    let edits = random_edits(&mesh, TOP, 150, 23);
    let (first, second) = edits.split_at(75);
    apply(&mut store, &mut map, first);
    for key in &all {
        store.evaluate(*key).unwrap();
    }
    apply(&mut store, &mut map, second);
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, "layered");
    store.save().unwrap();
    assert!(
        store.file.keys().count() > 0,
        "the working file took the writes"
    );
    drop(store);
    assert_eq!(
        std::fs::read(dir.path("p.bcrf")).unwrap(),
        project_bytes,
        "the base is read-only to the store"
    );

    // The recovery file holds the second batch on top of the base.
    let reopened = ProjectFile::open(&dir.path("r.bcrf")).unwrap();
    let mut again = TileStore::open(reopened, Some(project.base().unwrap()), roomy()).unwrap();
    assert_positions(&mut again, &dense, &all, "reopened over the base");
}

#[test]
fn a_rebase_keeps_evaluations_valid_and_bit_identical() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut map = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 150, 24),
    );
    let working = recovery(&dir, "r1.bcrf", &mesh);
    let mut store = TileStore::open(working, Some(project.base().unwrap()), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 150, 25));
    let before = positions(&mut store, &all);

    // A save: every tile changed since the last save, copied into the project file.
    store.save().unwrap();
    for key in store.file.keys() {
        let t = store.file.read_tile(key).unwrap().unwrap();
        project.write_tile(*key, t.op, &t.header, &t.body).unwrap();
    }
    project.save().unwrap();
    let fresh = recovery(&dir, "r2.bcrf", &mesh);
    let old = store.rebase(project.base().unwrap(), fresh).unwrap();
    assert!(old.keys().count() > 0, "the old working file comes back");
    drop(old);
    std::fs::remove_file(dir.path("r1.bcrf")).unwrap();

    store.log = EvalLog::default();
    assert_eq!(positions(&mut store, &all), before, "cached, bit for bit");
    assert_eq!(store.log.positions, [], "nothing recomputed");
    store.shrink_to(0).unwrap();
    assert_eq!(
        positions(&mut store, &all),
        before,
        "read again from the new base"
    );
    assert_eq!(
        store.file.keys().count(),
        0,
        "nothing in the fresh working file"
    );

    // Editing goes on over the new base, with ops past every saved one.
    let op = store.next_op().unwrap();
    assert!(Some(op) > project.head(), "{op:?} after the saved head");
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 26));
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &all,
        "edited after the rebase",
    );
}

#[test]
fn a_rebase_with_dirty_tiles_or_another_mesh_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, 3);
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, 3, 20, 27),
    );
    let working = recovery(&dir, "r.bcrf", &mesh);
    let mut store = TileStore::open(working, Some(project.base().unwrap()), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, 3, 20, 28));
    let dirty = store.rebase(project.base().unwrap(), recovery(&dir, "r2.bcrf", &mesh));
    assert!(
        matches!(
            dirty,
            Err(CacheError::Internal(Invariant::DirtyAtRebase(n))) if n > 0
        ),
        "dirty: {dirty:?}"
    );
    assert_positions(&mut store, &dense(&mesh, &map), &all, "unchanged");

    let other = Arc::new(cube_sphere(2).unwrap());
    store.save().unwrap();
    let foreign = store.rebase(project.base().unwrap(), recovery(&dir, "r3.bcrf", &other));
    assert!(
        matches!(foreign, Err(CacheError::MeshMismatch)),
        "another mesh's working file: {foreign:?}"
    );
    drop(store);
    let elsewhere = saved_project(&dir, "q.bcrf", &other, &mut CorrectionMap::new(), &[]);
    let opened = TileStore::open(
        recovery(&dir, "r4.bcrf", &mesh),
        Some(elsewhere.base().unwrap()),
        roomy(),
    );
    assert!(
        matches!(opened, Err(CacheError::MeshMismatch)),
        "a base of another mesh: {opened:?}"
    );
}

#[test]
fn ops_continue_after_the_base_head() {
    let dir = TestDir::new();
    let mesh = mesh();
    let key = tiles(&mesh, 1, 1)[0];
    let mut project =
        ProjectFile::create(&dir.path("p.bcrf"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let saved = OpId::new(SITE, 41).unwrap();
    project
        .write_displacement(key, saved, &flat_tile(&key, 0.5))
        .unwrap();
    project.save().unwrap();
    // Unsaved, so not part of the base.
    project
        .write_displacement(key, OpId::new(SITE, 99).unwrap(), &flat_tile(&key, 0.75))
        .unwrap();
    let working = recovery(&dir, "r.bcrf", &mesh);
    let mut store = TileStore::open(working, Some(project.base().unwrap()), roomy()).unwrap();
    assert_eq!(
        store.next_op().unwrap(),
        saved.next().unwrap(),
        "after the save point's head"
    );
    store.correction(key).unwrap();
    assert_eq!(
        slot_resident(&store, &key),
        slot_of(&flat_tile(&key, 0.5)),
        "the saved tile, not the one written after the save"
    );
}
