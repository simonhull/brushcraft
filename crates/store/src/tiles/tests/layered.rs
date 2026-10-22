//! Layered reads: a working file over a base (the project file as of its last save), against the
//! dense evaluation of every edit in both, and across a rebase onto a new save.

use super::*;
use crate::meta::{Meta, ProjectId, Role, SavePoint};
use crate::recovery::{RecoveryError, RecoveryWriter};

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

/// A base for a store: `project` as of its last save point.
fn base_of(project: &ProjectFile) -> BaseFile {
    project
        .base()
        .unwrap()
        .expect("the project has a save point")
}

/// A new, empty recovery file at `name` in `dir`, layered on `project`'s last save point: its
/// meta names that save point, as a recovery file's must.
fn recovery(dir: &TestDir, name: &str, project: &ProjectFile) -> ProjectFile {
    let role = Role::Recovery {
        base: Some(base_of(project).save_point()),
    };
    let meta = project.meta().with_role(role);
    ProjectFile::create_like(&dir.path(name), meta, project).unwrap()
}

/// A recovery file at `name` in `dir` over `mesh` with `role`, of the fixture's project.
fn recovery_with(dir: &TestDir, name: &str, mesh: &Arc<BaseMesh>, role: Role) -> ProjectFile {
    let meta = Meta::new(crate::fixture::PROJECT, role, mesh);
    ProjectFile::create(&dir.path(name), meta, Arc::clone(mesh)).unwrap()
}

/// Saves `store` into `project` and rebases it on a fresh recovery file `name` in `dir`, as
/// [`crate::Project::save`] does; returns the old working file.
fn save_and_rebase(
    store: &mut TileStore,
    project: &mut ProjectFile,
    dir: &TestDir,
    name: &str,
) -> ProjectFile {
    let saved = store.save_into(project).unwrap();
    let fresh = recovery(dir, name, project);
    store.rebase(saved, fresh).unwrap()
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
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
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
    let mut unsaved = recovery(&dir, "r.bcrf", &project);
    unsaved
        .write_displacement(both, op(3), &flat_tile(&both, -0.125))
        .unwrap();
    unsaved.save().unwrap();
    drop(unsaved);

    // Loaded on the owner's thread, and prefetched by the workers.
    for prefetched in [false, true] {
        let what = if prefetched { "prefetched" } else { "loaded" };
        let working = ProjectFile::open(&dir.path("r.bcrf")).unwrap();
        let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
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
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), tight()).unwrap();
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
    let mut again = TileStore::open_layered(reopened, Some(base_of(&project)), roomy()).unwrap();
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
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    // Edits written back to the working file, then more left dirty: a save copies both.
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 150, 25));
    store.shrink_to(0).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 20, 23));
    let before = positions(&mut store, &all);
    assert!(!store.unsaved.is_empty(), "some tiles still dirty");
    assert!(store.file.keys().count() > 0, "some written back");
    let old = save_and_rebase(&mut store, &mut project, &dir, "r2.bcrf");
    assert!(old.keys().count() > 0, "the old working file comes back");
    assert!(store.unsaved.is_empty(), "the save holds every dirty tile");
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
fn a_stale_save_or_another_mesh_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, 3);
    let mut map = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, 3, 20, 27),
    );
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, 3, 20, 28));

    // Edits after the save: the save doesn't hold them.
    let stale = store.save_into(&mut project).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, 3, 5, 29));
    let refused = store.rebase(stale, recovery(&dir, "r2.bcrf", &project));
    assert!(
        matches!(refused, Err(CacheError::StaleSave)),
        "stale: {refused:?}"
    );
    assert_positions(&mut store, &dense(&mesh, &map), &all, "unchanged");

    let other = Arc::new(cube_sphere(2).unwrap());
    let saved = store.save_into(&mut project).unwrap();
    let role = Role::Recovery {
        base: Some(saved.save_point()),
    };
    let foreign = store.rebase(saved, recovery_with(&dir, "r3.bcrf", &other, role));
    assert!(
        matches!(foreign, Err(CacheError::MeshMismatch)),
        "another mesh's working file: {foreign:?}"
    );
    // The save is still good; a rebase on it goes through.
    drop(save_and_rebase(&mut store, &mut project, &dir, "r4.bcrf"));
    assert_positions(&mut store, &dense(&mesh, &map), &all, "rebased");
    drop(store);
    let elsewhere = saved_project(&dir, "q.bcrf", &other, &mut CorrectionMap::new(), &[]);
    let opened = TileStore::open_layered(
        recovery(&dir, "r5.bcrf", &project),
        Some(base_of(&elsewhere)),
        roomy(),
    );
    assert!(
        matches!(opened, Err(CacheError::MeshMismatch)),
        "a base of another mesh: {opened:?}"
    );
}

/// The reproduction of the bug a save proof closes.
///
/// Edits, every dirty tile written back to the working file (so no slot is dirty), then a
/// rebase. Before, `rebase` took any base and saw nothing dirty, and the edits, never saved,
/// were dropped with the old working file. Now the only way to a rebase is through a save that
/// copied them.
#[test]
fn edits_written_back_but_never_saved_reach_the_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut map = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 50, 41),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), tight()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 100, 42));
    store.shrink_to(0).unwrap();
    assert!(
        store.unsaved.is_empty(),
        "every edit written back, none dirty"
    );
    drop(save_and_rebase(&mut store, &mut project, &dir, "r2.bcrf"));
    std::fs::remove_file(dir.path("r1.bcrf")).unwrap();
    store.shrink_to(0).unwrap();
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, "read back over the new base");
    drop(store);
    drop(project);
    let reopened = ProjectFile::open(&dir.path("p.bcrf")).unwrap();
    let mut alone = TileStore::open(reopened, roomy()).unwrap();
    assert_positions(&mut alone, &dense, &all, "the project file alone");
}

#[test]
fn a_working_file_that_does_not_name_its_base_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, 3, 10, 51),
    );
    let point = base_of(&project).save_point();
    let earlier = SavePoint {
        offset: point.offset - 1,
        ..point
    };
    let other_head = SavePoint {
        head: point.head.and_then(OpId::next),
        ..point
    };
    let other_project = SavePoint {
        project: ProjectId::from_bits(0xBAD),
        ..point
    };
    let wrong = [
        (
            "another offset",
            Role::Recovery {
                base: Some(earlier),
            },
        ),
        (
            "another head",
            Role::Recovery {
                base: Some(other_head),
            },
        ),
        (
            "another project's save point",
            Role::Recovery {
                base: Some(other_project),
            },
        ),
        ("no save point", Role::Recovery { base: None }),
        ("a project file", Role::Project),
    ];
    for (n, (what, role)) in wrong.into_iter().enumerate() {
        let working = recovery_with(&dir, &format!("w{n}.bcrf"), &mesh, role);
        let opened = TileStore::open_layered(working, Some(base_of(&project)), roomy());
        assert!(
            matches!(opened, Err(CacheError::WrongBase)),
            "{what}: {opened:?}"
        );
    }
}

#[test]
fn a_recovery_file_of_another_project_alone_or_over_one_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, 3, 10, 51),
    );
    let point = base_of(&project).save_point();
    // The right save point, but the recovery file is of another project.
    let foreign = Meta::new(
        ProjectId::from_bits(0xBAD),
        Role::Recovery { base: Some(point) },
        &mesh,
    );
    let of_another =
        ProjectFile::create(&dir.path("foreign.bcrf"), foreign, Arc::clone(&mesh)).unwrap();
    let opened = TileStore::open_layered(of_another, Some(base_of(&project)), roomy());
    assert!(
        matches!(opened, Err(CacheError::WrongBase)),
        "another project's recovery file: {opened:?}"
    );
    // A recovery file that names a save point needs it under it.
    let needs_base = recovery(&dir, "needs.bcrf", &project);
    assert!(
        matches!(
            TileStore::open(needs_base, roomy()),
            Err(CacheError::WrongBase)
        ),
        "a recovery file of a save, alone"
    );
    // A recovery file as a base.
    let mut as_base_saved = recovery(&dir, "as-base.bcrf", &project);
    as_base_saved.save().unwrap();
    let role = Role::Recovery {
        base: Some(base_of(&as_base_saved).save_point()),
    };
    let over = recovery_with(&dir, "over-recovery.bcrf", &mesh, role);
    let over_recovery = TileStore::open_layered(over, Some(base_of(&as_base_saved)), roomy());
    assert!(
        matches!(over_recovery, Err(CacheError::WrongBase)),
        "a recovery file as a base: {over_recovery:?}"
    );
}

#[test]
fn a_rebase_or_save_into_the_wrong_file_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, 3, 10, 51),
    );
    let point = base_of(&project).save_point();
    // A rebase checks the same: a fresh file that doesn't name the save is refused.
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, 3, 10, 52));
    let saved = store.save_into(&mut project).unwrap();
    let stale_name = recovery_with(&dir, "r2.bcrf", &mesh, Role::Recovery { base: Some(point) });
    let rebased = store.rebase(saved, stale_name);
    assert!(
        matches!(rebased, Err(CacheError::WrongBase)),
        "the old save point: {rebased:?}"
    );
    // Saving into a recovery file is refused too.
    let mut not_a_project = recovery(&dir, "r3.bcrf", &project);
    let into = store.save_into(&mut not_a_project);
    assert!(
        matches!(into, Err(CacheError::WrongBase)),
        "into a recovery file: {into:?}"
    );
}

#[test]
fn a_recovery_file_opened_beside_its_project_shares_the_mesh() {
    let dir = TestDir::new();
    let mesh = mesh();
    let project = saved_project(&dir, "p.bcrf", &mesh, &mut CorrectionMap::new(), &[]);
    let mut working = recovery(&dir, "r.bcrf", &project);
    working.save().unwrap();
    drop(working);
    let shared = ProjectFile::open_sharing_mesh(&dir.path("r.bcrf"), &project).unwrap();
    assert!(
        Arc::ptr_eq(&shared.mesh(), &project.mesh()),
        "the project's mesh, shared"
    );
    drop(shared);
    let alone = ProjectFile::open(&dir.path("r.bcrf")).unwrap();
    assert!(
        !Arc::ptr_eq(&alone.mesh(), &project.mesh()),
        "opened alone, its own"
    );
    assert_eq!(
        alone.mesh().to_bytes(),
        project.mesh().to_bytes(),
        "the same mesh"
    );
    // A file over another mesh is read in full, and keeps its own.
    let other = Arc::new(cube_sphere(2).unwrap());
    drop(ProjectFile::create(&dir.path("o.bcrf"), meta_for(&other), Arc::clone(&other)).unwrap());
    let theirs = ProjectFile::open_sharing_mesh(&dir.path("o.bcrf"), &project).unwrap();
    assert_eq!(theirs.mesh().to_bytes(), other.to_bytes(), "its own mesh");
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
    // Unsaved, so not part of the base; but in the file, so no op is handed out twice there.
    let unsaved = OpId::new(SITE, 99).unwrap();
    project
        .write_displacement(key, unsaved, &flat_tile(&key, 0.75))
        .unwrap();
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    assert_eq!(
        store.next_op().unwrap(),
        unsaved.next().unwrap(),
        "after the file's head, not only the save point's"
    );
    assert_eq!(
        base_of(&project).save_point().head,
        Some(saved),
        "the save point names its own head"
    );
    store.correction(key).unwrap();
    assert_eq!(
        slot_resident(&store, &key),
        slot_of(&flat_tile(&key, 0.5)),
        "the saved tile, not the one written after the save"
    );
}

#[test]
fn a_flush_in_flight_across_a_rebase_is_abandoned() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 40, 61),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    let mut writer = RecoveryWriter::new(&mut store).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 60, 62));
    writer.end_stroke(&mut store, Instant::now()).unwrap();
    assert_eq!(writer.phase(), "encoding", "a flush in flight");
    // A save that skipped finishing the flush first.
    drop(save_and_rebase(&mut store, &mut project, &dir, "r2.bcrf"));
    let len = store.working_len();
    let mut outcome = Ok(());
    while !writer.is_idle() {
        if let Err(e) = writer.step_blocking(&mut store) {
            outcome = Err(e);
        }
    }
    assert!(
        matches!(outcome, Err(RecoveryError::OutOfStep)),
        "{outcome:?}"
    );
    assert_eq!(
        store.working_len(),
        len,
        "nothing of the old flush in the new file"
    );
    assert_eq!(store.file.save_points().unwrap(), [], "and no save point");
    assert!(
        !writer.is_failed(),
        "not a sync failure: the writer goes on"
    );
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 10, 63));
    writer.finish(&mut store, Instant::now()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "after",
    );
}
