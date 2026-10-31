//! Layered reads: a working file over a base (the project file as of its last save), against the
//! dense evaluation of every edit in both, and across a rebase onto a new save.

use super::*;
use crate::index::SpatialIndex;
use crate::index::tests::{assert_loads_as_built, wave_now};
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
    project.base().expect("the project has a save point")
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
    CorrectionSlot::from_tile(tile).tile_bytes()
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
        store.hooks.log = EvalLog::default();
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
        assert_eq!(
            store.hooks.log.file_reads.len(),
            reads,
            "{what}: owner reads"
        );
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

    store.hooks.log = EvalLog::default();
    assert_eq!(positions(&mut store, &all), before, "cached, bit for bit");
    assert_eq!(store.hooks.log.positions, [], "nothing recomputed");
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
    let op = store.begin_op().unwrap();
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
    // The refused rebase took its save; a new save, and a rebase on it, go through.
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

/// A tile lost to a save drops what was evaluated from its lost edit.
///
/// After the rebase the store evaluates the saved state, and the bounds an index then records
/// and saves are that state's.
///
/// The lost edit's tile was evaluated (by an index build) before its correction slot left the
/// pool, as pools evict independently under a budget, so cached positions came from the edit
/// the save couldn't read.
#[test]
fn a_tile_lost_to_a_save_drops_what_was_evaluated_from_it() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let top = tiles(&mesh, TOP, TOP);
    let mut saved = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut saved,
        &random_edits(&mesh, TOP, 40, 81),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    let lost = top[5];
    apply_batches(
        &mut store,
        &mut CorrectionMap::new(),
        &[(lost, vec![(5, 7, Vec3::new(0.0, 0.0, 0.25))])],
    );
    store.save().unwrap();
    let mut index = SpatialIndex::build(&mut store, level(TOP), |_, _| {}).unwrap();
    let wave = wave_now(&mut store, &[lost]);
    // The clean slot goes; the evaluations made from it stay cached.
    evict(&mut store, &lost);
    damage(&dir.path("r1.bcrf"), store.file.locate(&lost).unwrap());
    let proof = store.save_into(&mut project).unwrap();
    assert_eq!(proof.lost().iter().copied().collect::<Vec<_>>(), [lost]);
    assert!(wave.current(&store, &lost).is_some(), "current before");
    store
        .rebase(proof, recovery(&dir, "r2.bcrf", &project))
        .unwrap();
    assert!(
        wave.current(&store, &lost).is_none(),
        "bounds from before the rebase that lost the tile are not used"
    );
    assert_positions(
        &mut store,
        &dense(&mesh, &saved),
        &all,
        "the lost tile and what reads it, at the saved state",
    );
    index.update(&mut store, &top).unwrap();
    drop(save_and_rebase(&mut store, &mut project, &dir, "r3.bcrf"));
    drop(store);
    drop(project);
    let file = ProjectFile::open(&dir.path("p.bcrf")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut reopened, level(TOP), "saved after the lost tile");
}

/// A tile lost to a save while a pin holds its slot is filled again in place.
///
/// The slot is refilled from the saved record: an edit after the pin goes starts from the saved
/// state, never from the lost edit.
#[test]
fn a_pinned_tile_lost_to_a_save_is_reloaded_in_place() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut saved = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut saved,
        &random_edits(&mesh, TOP, 40, 91),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    let lost = tiles(&mesh, TOP, TOP)[5];
    // A large edit, dirty with no op to write it with: the save loses it.
    apply_batches(
        &mut store,
        &mut CorrectionMap::new(),
        &[(lost, vec![(5, 7, Vec3::new(0.0, 0.0, 0.5))])],
    );
    let pin = store.pin(&[lost]).unwrap();
    store.unsaved.remove(&lost);
    let proof = store.save_into(&mut project).unwrap();
    assert_eq!(proof.lost().iter().copied().collect::<Vec<_>>(), [lost]);
    store
        .rebase(proof, recovery(&dir, "r2.bcrf", &project))
        .unwrap();
    drop(pin);
    let mut want = saved.clone();
    apply_batches(
        &mut store,
        &mut want,
        &[(lost, vec![(5, 7, Vec3::new(0.0, 0.0, 0.001))])],
    );
    drop(save_and_rebase(&mut store, &mut project, &dir, "r3.bcrf"));
    drop(store);
    drop(project);
    let file = ProjectFile::open(&dir.path("p.bcrf")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut reopened,
        &dense(&mesh, &want),
        &all,
        "the edit after the pin, from the saved state",
    );
}

/// A save that can't read a tile's latest edit lists the tile as lost and leaves it at its last
/// saved state; the rebase and the saves after it go through.
#[test]
fn a_tile_lost_to_a_save_does_not_block_the_saves_after_it() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut saved = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut saved,
        &random_edits(&mesh, TOP, 40, 71),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    let top = tiles(&mesh, TOP, TOP);
    let (lost, kept) = (top[5], top[30]);
    // Interior vertices only: a lost tile's twins elsewhere would disagree with it.
    let mut map = saved.clone();
    apply_batches(
        &mut store,
        &mut map,
        &[(lost, vec![(5, 7, Vec3::new(0.0, 0.0, 0.25))])],
    );
    let mut want = saved.clone();
    for into in [&mut map, &mut want] {
        apply_batches(
            &mut store,
            into,
            &[(kept, vec![(9, 4, Vec3::new(0.25, 0.0, 0.0))])],
        );
    }
    // In the working file only, its record damaged there.
    store.save().unwrap();
    store.evict_all().unwrap();
    damage(&dir.path("r1.bcrf"), store.file.locate(&lost).unwrap());

    let proof = store.save_into(&mut project).unwrap();
    assert_eq!(
        proof.lost().iter().copied().collect::<Vec<_>>(),
        [lost],
        "listed as lost"
    );
    store
        .rebase(proof, recovery(&dir, "r2.bcrf", &project))
        .unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &want),
        &all,
        "its last saved state",
    );
    // The saves after it go through, the lost tile edited again too.
    let more = random_edits(&mesh, TOP, 30, 72);
    apply(&mut store, &mut want, &more);
    drop(save_and_rebase(&mut store, &mut project, &dir, "r3.bcrf"));
    drop(store);
    drop(project);
    let file = ProjectFile::open(&dir.path("p.bcrf")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    assert_positions(&mut reopened, &dense(&mesh, &want), &all, "saved after");
}

/// A store with `edits` made over a saved project, written to its working file: what the
/// broken invariants below are made from. Returns the project file and the store.
fn edited_over_a_save(dir: &TestDir, mesh: &Arc<BaseMesh>) -> (ProjectFile, TileStore) {
    let mut map = CorrectionMap::new();
    let project = saved_project(
        dir,
        "p.bcrf",
        mesh,
        &mut map,
        &random_edits(mesh, 3, 20, 81),
    );
    let working = recovery(dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(mesh, 3, 20, 82));
    store.save().unwrap();
    store.evict_all().unwrap();
    (project, store)
}

#[test]
fn a_tile_of_the_recovery_file_left_out_of_a_save_is_not_in_the_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut project, mut store) = edited_over_a_save(&dir, &mesh);
    store.drain_saves();
    let left_out = *store.since_rebase.first().unwrap();
    store.since_rebase.remove(&left_out);
    let saved = store.save_into(&mut project);
    assert!(
        matches!(
            saved,
            Err(CacheError::Internal(Invariant::NotInSave(key))) if key == left_out
        ),
        "{saved:?}"
    );
}

#[test]
fn a_rebase_onto_a_save_without_an_edited_tile_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut project, mut store) = edited_over_a_save(&dir, &mesh);
    let saved = store.save_into(&mut project).unwrap();
    // A tile neither saved nor lost: one of the next level, never edited.
    let unsaved = tiles(&mesh, 4, 4)[0];
    store.since_rebase.insert(unsaved);
    let len = store.working_len();
    let refused = store.rebase(saved, recovery(&dir, "r2.bcrf", &project));
    assert!(
        matches!(
            refused,
            Err(CacheError::Internal(Invariant::NotInSave(key))) if key == unsaved
        ),
        "{refused:?}"
    );
    assert_eq!(store.working_len(), len, "still on the old working file");
}

#[test]
fn a_rebase_onto_a_save_point_not_descending_from_the_base_is_refused() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut map,
        &random_edits(&mesh, TOP, 60, 91),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 5, 92));
    // Another file of the same project and mesh, which never held the base's records: its save
    // point holds the edits, but doesn't descend from the base.
    let mut other =
        ProjectFile::create_like(&dir.path("q.bcrf"), project.meta().clone(), &project).unwrap();
    let saved = store.save_into(&mut other).unwrap();
    let only_in_base = *project
        .keys()
        .find(|key| !other.contains(key))
        .expect("the base holds tiles the edits didn't touch");
    let len = store.working_len();
    let refused = store.rebase(saved, recovery(&dir, "r2.bcrf", &other));
    assert!(
        matches!(
            refused,
            Err(CacheError::Internal(Invariant::NotInSave(key))) if key == only_in_base
        ),
        "{refused:?}"
    );
    assert_eq!(store.working_len(), len, "still on the old working file");
}

#[test]
fn only_resident_tiles_keep_the_op_of_their_record() {
    let dir = TestDir::new();
    let mesh = mesh();
    let project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut CorrectionMap::new(),
        &random_edits(&mesh, TOP, 200, 93),
    );
    let saved: Vec<TileKey> = project.keys().copied().collect();
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), tight()).unwrap();
    // More tiles than fit: loading each evicts others, whose ops go with them.
    for key in &saved {
        store.correction(*key).unwrap();
        store.assert_known_ops_resident();
        let (op, _) = project.read_displacement(key).unwrap().unwrap();
        assert_eq!(
            store.known_ops.get(key),
            Some(&op),
            "{key:?}: its record's op"
        );
    }
    assert!(
        store.known_ops.len() < saved.len(),
        "the budget evicted some: {} of {} kept",
        store.known_ops.len(),
        saved.len()
    );
    store.evict_all().unwrap();
    assert!(store.known_ops.is_empty(), "nothing resident, no op kept");
}

#[test]
fn an_edited_tile_with_nothing_to_copy_is_nothing_to_save() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut project, mut store) = edited_over_a_save(&dir, &mesh);
    // Listed as edited, but neither dirty nor in the working file.
    let nowhere = tiles(&mesh, 4, 4)[0];
    store.since_rebase.insert(nowhere);
    let saved = store.save_into(&mut project);
    assert!(
        matches!(
            saved,
            Err(CacheError::Internal(Invariant::NothingToSave(key))) if key == nowhere
        ),
        "{saved:?}"
    );
}

#[test]
fn a_project_file_with_no_save_point_is_not_saved() {
    let dir = TestDir::new();
    let mesh = mesh();
    let never =
        ProjectFile::create(&dir.path("never.bcrf"), meta_for(&mesh), Arc::clone(&mesh)).unwrap();
    let base = saved_base(&never);
    assert!(
        matches!(base, Err(CacheError::Internal(Invariant::NotSaved))),
        "{base:?}"
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
    // Unsaved, so not part of the base; but in the file, so no op is handed out twice there.
    let unsaved = OpId::new(SITE, 99).unwrap();
    project
        .write_displacement(key, unsaved, &flat_tile(&key, 0.75))
        .unwrap();
    let working = recovery(&dir, "r.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    assert_eq!(
        store.begin_op().unwrap(),
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

/// A tile lost to a save between two slices of a display refresh: the refresh copied the lost
/// edit's evaluation, and must not build from that copy once the tile reads as its record.
///
/// The refresh, under the tightest budget, copies each finest tile it evaluates; the save loses
/// one of them (its working-file record is damaged) and the rebase reverts it, without a write.
/// The copies after the loss are of the saved state, and the layer the refresh then finishes
/// is the one the saved state builds.
#[test]
fn a_tile_lost_between_refresh_slices_is_not_built_from_its_copy() {
    use std::time::{Duration, Instant};
    let dir = TestDir::new();
    let mesh = mesh();
    let mut saved = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut saved,
        &random_edits(&mesh, TOP, 10, 90),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let display = tight().with_display(level(TOP));
    let config = display.with_budget_bytes(display.budget_bytes + display.floor_bytes(&mesh));
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), config).unwrap();
    let run_all = |s: &mut TileStore| {
        while s.has_jobs() {
            s.run_jobs(Instant::now() + Duration::from_secs(60))
                .unwrap();
        }
    };
    run_all(&mut store);
    let lost = tiles(&mesh, TOP, TOP)[9];
    apply_batches(
        &mut store,
        &mut CorrectionMap::new(),
        &[(lost, vec![(9, 11, Vec3::new(0.02, -0.03, 0.25))])],
    );
    store.save().unwrap();
    // Slices until the refresh has copied the lost tile's evaluation.
    let began = Instant::now();
    loop {
        store
            .run_jobs(Instant::now() + Duration::from_micros(50))
            .unwrap();
        let copied = store
            .jobs
            .front_mut()
            .is_some_and(|job| job.upkeep().copies.tiles.contains_key(&lost));
        if copied {
            break;
        }
        // Counted in time: a call this short returns at once while the workers build.
        assert!(
            began.elapsed() < Duration::from_secs(60) && store.has_jobs(),
            "the refresh copies the tile"
        );
    }
    if store.corrections.find(&lost).is_some() {
        evict(&mut store, &lost);
    }
    damage(&dir.path("r1.bcrf"), store.file.locate(&lost).unwrap());
    let proof = store.save_into(&mut project).unwrap();
    assert_eq!(proof.lost().iter().copied().collect::<Vec<_>>(), [lost]);
    store
        .rebase(proof, recovery(&dir, "r2.bcrf", &project))
        .unwrap();
    // The next slice holds no copy of the lost tile but its current evaluation.
    store
        .run_jobs(Instant::now() + Duration::from_micros(50))
        .unwrap();
    let copy = store.jobs.front_mut().and_then(|job| {
        job.upkeep()
            .copies
            .tiles
            .get(&lost)
            .map(|t| t.positions.to_vec())
    });
    if let Some(copy) = copy {
        let current = store.evaluate(lost).unwrap().to_vec();
        assert!(
            copy == current,
            "a copy of the saved state, not of the lost edit"
        );
    }
    run_all(&mut store);
    let built = display_layer(&mut store);
    let keys: Vec<TileKey> = built.iter().map(|t| *t.key()).collect();
    for (tile, kept) in built.iter().zip(store.stored_display(&keys).unwrap()) {
        assert_eq!(
            kept.as_ref(),
            Some(tile),
            "{:?} as the saved state builds it",
            tile.key()
        );
    }
}

/// A tile lost to a save goes back to its record at the rebase: its positions change, and the
/// stale-tile feed names it, in an epoch of its own.
#[test]
fn a_tile_lost_to_a_rebase_is_named_by_the_feed() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut saved = CorrectionMap::new();
    let mut project = saved_project(
        &dir,
        "p.bcrf",
        &mesh,
        &mut saved,
        &random_edits(&mesh, TOP, 10, 95),
    );
    let working = recovery(&dir, "r1.bcrf", &project);
    let mut store = TileStore::open_layered(working, Some(base_of(&project)), roomy()).unwrap();
    let cursor = store.subscribe_stale();
    let lost = tiles(&mesh, TOP, TOP)[17];
    apply_batches(
        &mut store,
        &mut CorrectionMap::new(),
        &[(lost, vec![(5, 7, Vec3::new(0.0, 0.0, 0.25))])],
    );
    store.close_epoch().unwrap();
    store.save().unwrap();
    let before = store.evaluate(lost).unwrap().to_vec();
    drop(store.drain_stale(&cursor));
    evict(&mut store, &lost);
    damage(&dir.path("r1.bcrf"), store.file.locate(&lost).unwrap());
    let proof = store.save_into(&mut project).unwrap();
    store
        .rebase(proof, recovery(&dir, "r2.bcrf", &project))
        .unwrap();
    assert!(
        store.evaluate(lost).unwrap() != before.as_slice(),
        "the tile went back to its record"
    );
    match store.drain_stale(&cursor) {
        crate::stale::Drained::Epochs(epochs) => assert!(
            epochs.iter().any(|e| e.keys.contains(&lost)),
            "the lost tile is named"
        ),
        crate::stale::Drained::Resync(_) => panic!("no resync expected"),
    }
}
