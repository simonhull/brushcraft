//! Projects end to end, against the dense evaluation: new, save as, save, close and open, the
//! recovery file each save starts, and what a failed save leaves.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

mod base;
mod bounds;
mod crash;
mod hostile;
mod restore;

use geom::{CorrectionMap, Level, Region, TileKey, Vec3};

use super::*;
use crate::index::SpatialIndex;
use crate::index::tests::assert_loads_as_built;
use crate::test_dir::TestDir;
use crate::tiles::tests::{
    TOP, apply, apply_dabs, assert_positions, break_slot, damage, dense, mesh, random_edits, roomy,
    tiles,
};

/// Asserts that a save went through with nothing to report: crash protection runs, every edit
/// is protected, and the old recovery file was deleted.
fn assert_saved(saved: Result<SaveOutcome, ProjectError>) {
    let outcome = saved.unwrap();
    assert!(
        outcome.protection.is_active() && outcome.unprotected.is_empty() && outcome.kept.is_empty(),
        "{outcome:?}"
    );
}

/// Whether `e` says crash protection paused because of an I/O error: a recovery file that
/// couldn't be put in place.
fn paused_by_io(e: &ProjectError) -> bool {
    matches!(e, ProjectError::ProtectionPaused(cause) if matches!(**cause, ProjectError::Io(_)))
}

/// A configuration with room for every tile, and recovery files in `dir`'s `recovery`.
fn config(dir: &TestDir) -> ProjectConfig {
    ProjectConfig {
        store: roomy(),
        recovery_dir: dir.path("recovery"),
    }
}

/// Whether `name` ends in `.tmp`: a temporary file.
fn is_temporary(name: &str) -> bool {
    Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
}

/// The names in `dir`'s recovery directory, sorted.
fn recovery_files(dir: &TestDir) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir.path("recovery"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// One stroke: begun, `count` random edits with `seed` applied as its dabs to the project's store
/// and to `map`, and ended.
fn stroke(project: &mut Project, map: &mut CorrectionMap, count: u32, seed: u64) {
    let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), map, &edits);
    project.end_stroke().unwrap();
}

/// Asserts that the index loaded from the project file at `path`, at its latest save point, is
/// the one built from its tiles: no bound it keeps is stale.
fn assert_project_bounds(path: &Path, what: &str) {
    let file = ProjectFile::open(path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_loads_as_built(&mut store, Level::new(TOP).unwrap(), what);
}

/// Asserts that the project file at `path`, opened at its latest save point, evaluates as the
/// dense evaluation of `map`.
fn assert_project_file(path: &Path, map: &CorrectionMap, what: &str) {
    let file = ProjectFile::open(path).unwrap();
    let mesh = file.mesh();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(&mut store, &dense(&mesh, map), &tiles(&mesh, 1, TOP), what);
}

#[test]
fn new_edit_save_as_close_open_evaluates_as_dense() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    assert!(!project.has_unsaved_edits(), "nothing edited yet");
    assert_eq!(project.path(), None, "never saved");
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 150, 1);
    stroke(&mut project, &mut map, 150, 2);
    assert!(project.has_unsaved_edits(), "edited, never saved");
    assert!(
        matches!(project.save(), Err(ProjectError::NoPath)),
        "a save needs a path first"
    );
    let path = dir.path("p.bcrf");
    assert_saved(project.save_as(&path));
    assert!(!project.has_unsaved_edits(), "all saved");
    assert_eq!(project.path(), Some(path.as_path()), "the new path");
    project.close(false).unwrap();
    assert_eq!(
        recovery_files(&dir),
        Vec::<String>::new(),
        "closed, deleted"
    );
    assert_project_file(&path, &map, "the saved file");

    let mut reopened = Project::open(&path, config(&dir)).unwrap();
    let dense = dense(&mesh, &map);
    assert_positions(reopened.store(), &dense, &tiles(&mesh, 1, TOP), "reopened");
    assert!(
        !reopened.has_unsaved_edits(),
        "nothing edited since the open"
    );
    assert_eq!(
        Project::open(&path, config(&dir))
            .map(drop)
            .unwrap_err()
            .to_string(),
        ProjectError::Store(StoreError::Locked).to_string(),
        "one instance at a time"
    );
}

#[test]
fn each_save_is_a_save_point_that_opens_to_its_state() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    let mut states = Vec::new();
    stroke(&mut project, &mut map, 120, 11);
    assert_saved(project.save_as(&path));
    states.push(map.clone());
    for seed in [12, 13] {
        stroke(&mut project, &mut map, 120, seed);
        assert_saved(project.save());
        states.push(map.clone());
    }
    project.close(false).unwrap();
    let points = ProjectFile::open(&path).unwrap().save_points().unwrap();
    assert_eq!(points.len(), states.len(), "one save point per save");
    for (point, state) in points.iter().zip(&states) {
        let file = ProjectFile::open_at(&path, *point).unwrap();
        let mut store = TileStore::open(file, roomy()).unwrap();
        let what = format!("save point {point}");
        assert_positions(
            &mut store,
            &dense(&mesh, state),
            &tiles(&mesh, 1, TOP),
            &what,
        );
    }
}

#[test]
fn a_save_starts_an_empty_recovery_file_on_its_save_point() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let id = project.id();
    let first = project.recovery_path().unwrap().to_path_buf();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 100, 21);
    assert_saved(project.save_as(&path));
    let new_id = project.id();
    assert_ne!(new_id, id, "a saved-as project has an id of its own");
    stroke(&mut project, &mut map, 100, 22);
    assert_saved(project.save());
    let recovery = project.recovery_path().unwrap().to_path_buf();
    assert!(!first.exists(), "the first recovery file is gone");
    assert_eq!(
        recovery_files(&dir),
        [recovery.file_name().unwrap().to_str().unwrap()],
        "one recovery file, and no temporary file"
    );
    let name = recovery.file_name().unwrap().to_str().unwrap().to_owned();
    let prefix = format!("{new_id}-");
    assert!(
        name.starts_with(&prefix)
            && name.len() == 32 + 1 + 16 + 5
            && recovery.extension() == Some("bcrf".as_ref()),
        "named by project id and a suffix: {name}"
    );
    let saved = ProjectFile::open(&path).map(drop);
    assert!(
        matches!(saved, Err(StoreError::Locked)),
        "the project holds its file: {saved:?}"
    );
    let point = *project
        .file
        .as_ref()
        .unwrap()
        .save_points()
        .unwrap()
        .last()
        .unwrap();
    let working = project.store().working();
    assert_eq!(
        working.keys().count(),
        0,
        "the recovery file holds no tiles"
    );
    let role = working.meta().role();
    let Role::Recovery { base: Some(base) } = role else {
        panic!("a recovery file on a save point: {role:?}");
    };
    assert_eq!(
        (base.project, base.offset),
        (new_id, point),
        "it names the project and its latest save point"
    );
    assert_eq!(working.meta().project(), new_id, "of the project");
    project.close(true).unwrap();
    assert!(recovery.exists(), "kept when asked");
}

#[test]
fn save_as_gives_a_new_id_and_leaves_the_old_file_untouched() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (old, new) = (dir.path("old.bcrf"), dir.path("new.bcrf"));
    let mut map = CorrectionMap::new();
    let mut first = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut first, &mut map, 150, 31);
    assert_saved(first.save_as(&old));
    let saved = map.clone();
    first.close(false).unwrap();
    let old_bytes = fs::read(&old).unwrap();
    let old_id = ProjectFile::open(&old).unwrap().meta().project();

    let mut project = Project::open(&old, config(&dir)).unwrap();
    stroke(&mut project, &mut map, 150, 32);
    assert!(
        matches!(project.save_as(&old), Err(ProjectError::Exists(_))),
        "an existing path is refused"
    );
    assert_saved(project.save_as(&new));
    assert_ne!(project.id(), old_id, "a new id");
    // The old file is no longer held: it opens elsewhere.
    assert_eq!(fs::read(&old).unwrap(), old_bytes, "the old file untouched");
    assert_project_file(&old, &saved, "the old file's state");
    stroke(&mut project, &mut map, 50, 33);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_eq!(fs::read(&old).unwrap(), old_bytes, "still untouched");
    assert_project_file(&new, &map, "the new file holds every tile");
    let names: Vec<_> = fs::read_dir(dir.path(""))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| is_temporary(name))
        .collect();
    assert_eq!(names, Vec::<String>::new(), "no temporary file left");
}

/// A save whose new recovery file can't be put in place (a crash before its rename, as far as
/// the file goes) goes through all the same: crash protection pauses, and the next save puts it
/// back.
#[test]
fn a_save_that_cannot_make_its_recovery_file_goes_through_and_pauses_protection() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 100, 41);
    assert_saved(project.save_as(&path));
    stroke(&mut project, &mut map, 100, 42);
    let old = project.recovery_path().unwrap().to_path_buf();

    project.fail_next_recovery();
    let outcome = project.save().unwrap();
    assert!(
        matches!(&outcome.protection, Protection::Paused(cause) if matches!(**cause, ProjectError::Io(_))),
        "paused, and why: {outcome:?}"
    );
    assert!(
        outcome.unprotected.is_empty() && outcome.kept.is_empty(),
        "nothing else to report: {outcome:?}"
    );
    assert!(!project.has_unsaved_edits(), "saved");
    assert_eq!(
        project.file.as_ref().unwrap().save_points().unwrap().len(),
        2,
        "the save point is written"
    );
    assert!(!project.protection().is_active(), "the project says so");
    assert_eq!(project.recovery_path(), None, "no recovery file");
    assert!(!old.exists(), "the old one, all saved, is gone");
    let names = recovery_files(&dir);
    assert!(
        names.iter().all(|name| is_temporary(name)),
        "no recovery file under a real name: {names:?}"
    );

    // Editing and saving go on, and the save puts protection back.
    stroke(&mut project, &mut map, 50, 43);
    assert_saved(project.save());
    assert!(project.protection().is_active(), "protected again");
    assert!(project.recovery_path().is_some(), "with a recovery file");
    project.close(false).unwrap();
    assert_project_file(&path, &map, "every edit saved");
}

/// Makes a directory read-only for as long as it lives, and writable again when dropped, even
/// by a failing test.
#[derive(Debug)]
struct Unwritable(PathBuf);

impl Unwritable {
    /// Makes `dir` read-only.
    fn new(dir: &Path) -> Self {
        set_mode(dir, 0o555);
        Self(dir.to_path_buf())
    }
}

/// Makes the directory writable again.
impl Drop for Unwritable {
    fn drop(&mut self) {
        set_mode(&self.0, 0o755);
    }
}

/// Sets `dir`'s permission bits to `mode`.
fn set_mode(dir: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt as _;
    fs::set_permissions(dir, fs::Permissions::from_mode(mode)).unwrap();
}

/// The number of save points of the project file `project` holds.
fn save_points(project: &Project) -> usize {
    project.file.as_ref().unwrap().save_points().unwrap().len()
}

/// A recovery directory that can't be written stops no save, and protection resumes after.
///
/// Each save goes through with crash protection paused, and edits and saves go on. Once the
/// directory can be written again, the next try puts protection back, holding every edit since
/// the last save, and deletes the old recovery file the first save couldn't.
#[test]
fn saves_go_through_an_unwritable_recovery_dir_and_protection_resumes_after() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 80, 171);
    assert_saved(project.save_as(&path));
    stroke(&mut project, &mut map, 80, 172);
    let old = project.recovery_path().unwrap().to_path_buf();

    let unwritable = Unwritable::new(&dir.path("recovery"));
    let outcome = project.save().unwrap();
    assert!(!outcome.protection.is_active(), "paused: {outcome:?}");
    let kept: Vec<&Path> = outcome.kept.iter().map(|k| k.path.as_path()).collect();
    assert_eq!(kept, [&old], "the old file can't be deleted: {outcome:?}");
    assert!(outcome.unprotected.is_empty(), "{outcome:?}");
    assert!(!project.has_unsaved_edits(), "saved");
    assert_eq!(save_points(&project), 2, "the save point is written");
    assert_eq!(project.recovery_path(), None, "no recovery file");
    assert!(!project.protection().is_active(), "the project says so");
    assert_eq!(
        find_recoveries(&dir.path("recovery")).unwrap(),
        [],
        "the old file is held, so not offered"
    );

    stroke(&mut project, &mut map, 80, 173);
    let again = project.save().unwrap();
    assert!(
        !again.protection.is_active() && again.kept.is_empty() && again.unprotected.is_empty(),
        "saved again, still paused: {again:?}"
    );
    assert!(!project.has_unsaved_edits(), "saved again");
    assert_eq!(save_points(&project), 3, "a save point more");

    drop(unwritable);
    stroke(&mut project, &mut map, 80, 174);
    assert!(
        !project.protection().is_active(),
        "not tried before the wait is over"
    );
    project.tick(Instant::now() + ROLL_OVER_RETRY_MAX).unwrap();
    assert!(project.protection().is_active(), "protected again");
    let recovery = project.recovery_path().unwrap().to_path_buf();
    assert_eq!(
        recovery_files(&dir),
        [recovery.file_name().unwrap().to_str().unwrap()],
        "one recovery file: the old one is gone too"
    );

    // A crash now loses nothing: the new recovery file holds the edits since the last save.
    drop(project);
    let candidates = find_recoveries(&dir.path("recovery")).unwrap();
    let [candidate] = candidates.as_slice() else {
        panic!("one recovery file: {candidates:?}");
    };
    assert_eq!(
        candidate.status,
        CandidateStatus::Restorable,
        "{candidate:?}"
    );
    let mut restored = Project::restore(candidate, config(&dir)).unwrap();
    assert_positions(
        restored.store(),
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "restored after protection resumed",
    );
    assert_saved(restored.save());
    restored.close(false).unwrap();
    assert_project_file(&path, &map, "every edit saved");
}

/// A save after a failed sync, with no roll-over possible, writes the project file.
///
/// The recovery writer has failed to sync, and the recovery directory can't be written, so no
/// roll-over can be made: the project file is written all the same, and holds every edit, the
/// failed flush's too.
#[test]
fn a_save_after_a_failed_sync_with_an_unwritable_recovery_dir_writes_the_project_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 80, 181);
    assert_saved(project.save_as(&path));
    stroke_failing_its_sync(&mut project, &mut map, 80, 182);
    let failed = project.recovery_path().unwrap().to_path_buf();

    let unwritable = Unwritable::new(&dir.path("recovery"));
    let outcome = project.save().unwrap();
    assert!(!outcome.protection.is_active(), "paused: {outcome:?}");
    let kept: Vec<&Path> = outcome.kept.iter().map(|k| k.path.as_path()).collect();
    assert_eq!(
        kept,
        [&failed],
        "the failed file can't be deleted: {outcome:?}"
    );
    assert!(!project.has_unsaved_edits(), "saved");
    assert_eq!(save_points(&project), 2, "the save point is written");

    drop(unwritable);
    project.close(false).unwrap();
    assert_eq!(recovery_files(&dir), Vec::<String>::new(), "all deleted");
    assert_project_file(&path, &map, "every edit, the failed flush's too");
}

/// A save as whose old recovery file can't be deleted: the session goes on with the new project
/// file, its path and its id, and the failure is reported, not fatal.
///
/// The old file is deleted from outside first: a file already gone is no failure at all. Then
/// a directory takes its name, which no file deletion removes.
#[test]
fn a_save_as_that_cannot_delete_the_old_recovery_file_goes_on_with_the_new_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let all = tiles(&mesh, 1, TOP);
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 100, 101);
    for (n, blocked) in [false, true].into_iter().enumerate() {
        let old = project.recovery_path().unwrap().to_path_buf();
        let id = project.id();
        fs::remove_file(&old).unwrap();
        if blocked {
            fs::create_dir_all(&old).unwrap();
        }
        let path = dir.path(&format!("p{n}.bcrf"));
        let saved = project.save_as(&path);
        if blocked {
            let outcome = saved.unwrap();
            let kept: Vec<&Path> = outcome.kept.iter().map(|k| k.path.as_path()).collect();
            assert_eq!(kept, [&old], "the old file is reported: {outcome:?}");
            assert!(
                outcome.protection.is_active() && outcome.unprotected.is_empty(),
                "and nothing else: {outcome:?}"
            );
            fs::remove_dir(&old).unwrap();
        } else {
            assert_saved(saved);
        }
        assert_eq!(project.path(), Some(path.as_path()), "{n}: the new path");
        assert_ne!(project.id(), id, "{n}: a new id");
        assert_ne!(
            project.recovery_path().unwrap(),
            old,
            "{n}: a new recovery file"
        );
        assert!(!project.has_unsaved_edits(), "{n}: saved");
        // Every tile read again, through the new project file.
        project.store().evict_all().unwrap();
        assert_positions(
            project.store(),
            &dense(&mesh, &map),
            &all,
            "after the save as",
        );
        stroke(&mut project, &mut map, 40, 102 + n as u64);
        assert_saved(project.save());
    }
    let path = project.path().unwrap().to_path_buf();
    let id = project.id();
    project.close(false).unwrap();
    assert_eq!(
        ProjectFile::open(&path).unwrap().meta().project(),
        id,
        "the project file is the project's"
    );
    assert_project_file(&path, &map, "every edit saved");
}

#[test]
fn close_keeps_the_recovery_file_only_when_asked() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 51);
    let kept = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    // A project never saved lives in its recovery file: it holds every edit.
    let file = ProjectFile::open(&kept).unwrap();
    assert_eq!(
        file.meta().role(),
        Role::Recovery { base: None },
        "never saved"
    );
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "kept",
    );
    drop(store);

    let other = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let gone = other.recovery_path().unwrap().to_path_buf();
    other.close(false).unwrap();
    assert!(!gone.exists(), "deleted when not asked to keep it");
    assert!(kept.exists(), "another project's file is left alone");
}

#[test]
fn dropping_a_project_closes_its_writer_and_keeps_the_recovery_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    // Edits with no stroke end: the drop's close flushes them.
    let edits = random_edits(&mesh, TOP, 60, 61);
    apply(project.store(), &mut map, &edits);
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    let file = ProjectFile::open(&path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "flushed on drop",
    );
}

#[test]
fn only_a_saved_project_file_opens_as_a_project() {
    let dir = TestDir::new();
    let mesh = mesh();
    let never = dir.path("never.bcrf");
    drop(ProjectFile::create(&never, crate::fixture::meta_for(&mesh), Arc::clone(&mesh)).unwrap());
    assert!(
        matches!(
            Project::open(&never, config(&dir)),
            Err(ProjectError::NeverSaved)
        ),
        "no save point"
    );
    let project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert!(
        matches!(
            Project::open(&recovery, config(&dir)),
            Err(ProjectError::NotAProject)
        ),
        "a recovery file"
    );
    assert_eq!(
        recovery_files(&dir).len(),
        1,
        "no recovery file made for either"
    );
}

#[test]
fn the_default_recovery_dir_follows_xdg() {
    let dir = default_recovery_dir();
    if let Some(dir) = dir {
        assert!(dir.is_absolute(), "{dir:?}");
        assert!(dir.ends_with("brushcraft/recovery"), "{dir:?}");
    }
}

/// Fails the writer's next sync, makes a stroke, and polls until the project has rolled over
/// to a new recovery file; returns the old one's path.
fn fail_and_roll_over(project: &mut Project, map: &mut CorrectionMap, seed: u64) -> PathBuf {
    let old = project.recovery_path().unwrap().to_path_buf();
    // The flushes of earlier strokes finish first, so the sync that fails is this stroke's.
    project.finish_flushes().unwrap();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    stroke(project, map, 80, seed);
    let began = Instant::now();
    while project.recovery_path().unwrap() == old {
        project.poll().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the roll-over happens"
        );
        std::thread::yield_now();
    }
    old
}

#[test]
fn a_failed_sync_rolls_over_to_a_new_recovery_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 71);
    let old = fail_and_roll_over(&mut project, &mut map, 72);
    assert!(!old.exists(), "the failed file is gone");
    let names = recovery_files(&dir);
    assert_eq!(names.len(), 1, "one recovery file: {names:?}");
    assert!(
        !project.writer.as_ref().unwrap().is_failed(),
        "a new writer"
    );
    assert_eq!(
        project.store().working().meta().role(),
        Role::Recovery { base: None },
        "on the same base"
    );
    // Edits go on into the new file.
    stroke(&mut project, &mut map, 40, 73);
    let path = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    let file = ProjectFile::open(&path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "every edit, the failed flush's too",
    );
}

/// A writer whose thread stopped, as a panic would stop it, is replaced by a roll-over.
///
/// The next stroke's end finds it gone, the project moves to a new recovery file with a new
/// writer, and protection carries on, every edit kept.
#[test]
fn a_writer_whose_thread_stopped_is_replaced_by_a_roll_over() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 60, 61);
    project.finish_flushes().unwrap();
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_mut().unwrap().stop_thread();
    stroke(&mut project, &mut map, 60, 62);
    assert_ne!(
        project.recovery_path().unwrap(),
        old,
        "rolled over to a new recovery file"
    );
    assert!(
        !project.writer.as_ref().unwrap().is_failed(),
        "a new writer"
    );
    assert!(project.protection().is_active(), "protection carries on");
    let path = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    let file = ProjectFile::open(&path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "every edit, both strokes",
    );
}

#[test]
fn a_saved_project_rolls_over_on_its_save_point_and_saves_after() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 81);
    assert_saved(project.save_as(&path));
    let role = project.store().working().meta().role();
    stroke(&mut project, &mut map, 80, 82);
    fail_and_roll_over(&mut project, &mut map, 83);
    assert_eq!(
        project.store().working().meta().role(),
        role,
        "the new file names the same save point"
    );
    assert!(
        project.has_unsaved_edits(),
        "nothing saved by the roll-over"
    );
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "saved after the roll-over");
    assert_eq!(recovery_files(&dir), Vec::<String>::new(), "all deleted");
}

#[test]
fn a_roll_over_that_cannot_make_its_file_keeps_the_old_one_and_tries_again() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 60, 91);
    // That stroke's flush finishes first: its sync failing would roll over (and fail) in the
    // next stroke's end, not in the polls below.
    project.finish_flushes().unwrap();
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.fail_next_recovery();
    stroke(&mut project, &mut map, 60, 92);
    let began = Instant::now();
    let failed = loop {
        match project.poll() {
            Ok(()) => {
                assert!(began.elapsed() < Duration::from_secs(30), "the sync fails");
                std::thread::yield_now();
            }
            Err(e) => break e,
        }
    };
    assert!(paused_by_io(&failed), "{failed:?}");
    assert!(!project.protection().is_active(), "paused");
    assert!(
        old.exists(),
        "the old file is kept until a new one is in place"
    );
    assert_eq!(
        project.recovery_path().unwrap(),
        old,
        "still on the old file"
    );
    // What the crash before the rename left: the old file, and the new one under its temporary
    // name only, already holding every edit. No file under a real name lacks them.
    let names = recovery_files(&dir);
    let old_name = old.file_name().unwrap().to_str().unwrap();
    let [first, second] = names.as_slice() else {
        panic!("the old file and a temporary one: {names:?}");
    };
    let tmp = if first == old_name { second } else { first };
    assert!(is_temporary(tmp), "the new file is temporary: {names:?}");
    let filled = ProjectFile::open(&dir.path("recovery").join(tmp)).unwrap();
    let mut temporary = TileStore::open(filled, roomy()).unwrap();
    assert_positions(
        &mut temporary,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "the temporary file, filled before its rename",
    );
    drop(temporary);
    // The calls of the frame loop wait a while before they try again.
    project.poll().unwrap();
    project.tick(Instant::now()).unwrap();
    assert_eq!(
        project.recovery_path().unwrap(),
        old,
        "not tried again at once"
    );
    // The first call once the wait is over tries again, and gets there.
    project.tick(Instant::now() + ROLL_OVER_RETRY).unwrap();
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    assert!(!old.exists(), "and only then the old file goes");
    let path = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    let file = ProjectFile::open(&path).unwrap();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "every edit",
    );
}

/// One stroke whose flush fails to sync.
///
/// Begun, `count` random edits with `seed` applied as its dabs to the project's store and to
/// `map`, the writer's next sync made to fail, and ended. The flushes of earlier strokes finish
/// first, so the sync that fails is this one's.
fn stroke_failing_its_sync(project: &mut Project, map: &mut CorrectionMap, count: u32, seed: u64) {
    project.finish_flushes().unwrap();
    let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), map, &edits);
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.end_stroke().unwrap();
}

/// A close whose last flush fails to sync rolls over first, so the recovery file it keeps
/// holds the stroke that ended just before it: for a project never saved, and a saved one.
#[test]
fn a_close_whose_last_sync_fails_keeps_the_ended_stroke() {
    for saved in [false, true] {
        let dir = TestDir::new();
        let mesh = mesh();
        let mut map = CorrectionMap::new();
        let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
        stroke(&mut project, &mut map, 80, 151);
        if saved {
            assert_saved(project.save_as(&dir.path("p.bcrf")));
        }
        stroke_failing_its_sync(&mut project, &mut map, 80, 152);
        let failed = project.recovery_path().unwrap().to_path_buf();
        project.close(true).unwrap();
        assert!(
            !failed.exists(),
            "{saved}: the file that failed to sync is gone"
        );
        let candidates = find_recoveries(&dir.path("recovery")).unwrap();
        let [candidate] = candidates.as_slice() else {
            panic!("{saved}: one recovery file: {candidates:?}");
        };
        let mut restored = Project::restore(candidate, config(&dir)).unwrap();
        assert_positions(
            restored.store(),
            &dense(&mesh, &map),
            &tiles(&mesh, 1, TOP),
            "every stroke, the one whose sync failed too",
        );
    }
}

/// A close that deletes the recovery file, whose last flush fails to sync, rolls over and then
/// deletes the new file: nothing is left behind.
#[test]
fn a_close_whose_last_sync_fails_still_deletes_the_recovery_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 161);
    assert_saved(project.save_as(&path));
    let saved = map.clone();
    stroke_failing_its_sync(&mut project, &mut map, 80, 162);
    project.close(false).unwrap();
    assert_eq!(recovery_files(&dir), Vec::<String>::new(), "all deleted");
    assert_project_file(&path, &saved, "the project file holds the save");
}

/// One stroke of one dab: begun, `c` set at an interior vertex of `key` (one copy, so a tile
/// lost to a save or a roll-over leaves no twin disagreeing) in the project's store and in
/// each of `maps`, and ended.
fn interior(project: &mut Project, maps: &mut [&mut CorrectionMap], key: TileKey, c: Vec3) {
    project.begin_stroke().unwrap();
    project
        .store()
        .edit_tile(key, |edit| edit.set(5, 7, c))
        .unwrap()
        .unwrap();
    let region = Region::of_tile(&key);
    let topology = project.store().working().mesh();
    for map in maps.iter_mut() {
        map.set(
            topology.topology(),
            key.face(),
            key.level(),
            region.x0() + 5,
            region.y0() + 7,
            c,
        )
        .unwrap();
    }
    project.end_stroke().unwrap();
}

/// A saved project with an edit readable nowhere.
///
/// Two tiles are edited since the save, the spatial index built over the edits (so the bounds
/// kept describe the lost edit too), written to the recovery file and evicted, and the first
/// one's record there damaged. Returns the project, the damaged tile, the edits of every
/// stroke, and the edits without the damaged tile's.
fn with_an_unreadable_edit(dir: &TestDir) -> (Project, TileKey, CorrectionMap, CorrectionMap) {
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(dir)).unwrap();
    stroke(&mut project, &mut map, 100, 121);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    let mut want = map.clone();
    let top = tiles(&mesh, TOP, TOP);
    let (lost, kept) = (top[5], top[30]);
    interior(
        &mut project,
        &mut [&mut map],
        lost,
        Vec3::new(0.0, 0.0, 0.25),
    );
    interior(
        &mut project,
        &mut [&mut map, &mut want],
        kept,
        Vec3::new(0.25, 0.0, 0.0),
    );
    SpatialIndex::build(project.store(), Level::new(TOP).unwrap(), |_, _| {}).unwrap();
    project.finish_flushes().unwrap();
    project.store().evict_all().unwrap();
    let loc = project.store().working().locate(&lost).unwrap();
    damage(project.recovery_path().unwrap(), loc);
    (project, lost, map, want)
}

#[test]
fn a_tile_lost_to_a_save_is_reported_and_the_saves_after_it_go_through() {
    let dir = TestDir::new();
    let (mut project, lost, _, mut want) = with_an_unreadable_edit(&dir);
    let outcome = project.save().unwrap();
    assert_eq!(outcome.unprotected, [lost], "{outcome:?}");
    assert!(
        outcome.protection.is_active() && outcome.kept.is_empty(),
        "{outcome:?}"
    );
    assert!(!project.has_unsaved_edits(), "the save went through");
    let other = tiles(&mesh(), TOP, TOP)[40];
    interior(
        &mut project,
        &mut [&mut want],
        other,
        Vec3::new(0.0, 0.25, 0.0),
    );
    assert_saved(project.save());
    let copy = dir.path("copy.bcrf");
    assert_saved(project.save_as(&copy));
    project.close(false).unwrap();
    for path in [dir.path("p.bcrf"), copy] {
        assert_project_file(&path, &want, "the lost tile at its last saved state");
        assert_project_bounds(&path, "the lost tile's bounds unknown");
    }
}

#[test]
fn a_tile_lost_to_a_roll_over_is_reported_and_saves_go_through() {
    let dir = TestDir::new();
    let (mut project, lost, _, mut want) = with_an_unreadable_edit(&dir);
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    let other = tiles(&mesh(), TOP, TOP)[40];
    interior(
        &mut project,
        &mut [&mut want],
        other,
        Vec3::new(0.0, 0.25, 0.0),
    );
    let began = Instant::now();
    let mut reported = Vec::new();
    while project.recovery_path().unwrap() == old {
        match project.poll() {
            Ok(()) => {}
            Err(ProjectError::Unprotected(tiles)) => reported.extend(tiles),
            Err(e) => panic!("{e:?}"),
        }
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the roll-over happens"
        );
        std::thread::yield_now();
    }
    assert_eq!(reported, [lost], "the lost tile is reported");
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(
        &dir.path("p.bcrf"),
        &want,
        "the lost tile at its last saved state",
    );
    assert_project_bounds(&dir.path("p.bcrf"), "the lost tile's bounds unknown");
}

/// A report waits for the save: tiles a flush left out (a broken slot) are reported after the
/// save is made, never instead of it.
#[test]
fn a_save_goes_through_and_then_reports_what_is_unprotected() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 100, 131);
    assert_saved(project.save_as(&path));
    let mut want = map.clone();
    let top = tiles(&mesh, TOP, TOP);
    let (broken, kept) = (top[5], top[30]);
    interior(
        &mut project,
        &mut [&mut map],
        broken,
        Vec3::new(0.0, 0.0, 0.25),
    );
    interior(
        &mut project,
        &mut [&mut map, &mut want],
        kept,
        Vec3::new(0.25, 0.0, 0.0),
    );
    project.finish_flushes().unwrap();
    // Edited again, its slot broken before the stroke ends: the flush leaves it out, and the
    // save copies its last record instead.
    project.begin_stroke().unwrap();
    project
        .store()
        .edit_tile(broken, |edit| edit.set(5, 7, Vec3::new(0.0, 0.0, 0.5)))
        .unwrap()
        .unwrap();
    break_slot(project.store(), &broken);
    project.end_stroke().unwrap();
    let outcome = project.save().unwrap();
    assert_eq!(
        outcome.unprotected,
        [broken],
        "reported once, by the save: {outcome:?}"
    );
    assert!(
        outcome.protection.is_active() && outcome.kept.is_empty(),
        "{outcome:?}"
    );
    assert!(!project.has_unsaved_edits(), "the save went through");
    project.poll().unwrap();
    project.close(false).unwrap();
    // The broken tile's first edit was flushed, so the save copied that; its second is lost.
    let mut first = want.clone();
    let region = Region::of_tile(&broken);
    first
        .set(
            mesh.topology(),
            broken.face(),
            broken.level(),
            region.x0() + 5,
            region.y0() + 7,
            Vec3::new(0.0, 0.0, 0.25),
        )
        .unwrap();
    assert_project_file(&path, &first, "every edit but the broken slot's last");
}

/// A roll-over that loses a tile and then fails (its old file can't be deleted) still reports
/// the lost tile: with the next call, as the failure takes this one.
#[test]
fn a_roll_over_that_fails_after_losing_a_tile_still_reports_it() {
    let dir = TestDir::new();
    let (mut project, lost, _, mut want) = with_an_unreadable_edit(&dir);
    let old = project.recovery_path().unwrap().to_path_buf();
    // The working file stays open, unlinked; a directory takes its name, so it can't be
    // deleted.
    fs::remove_file(&old).unwrap();
    fs::create_dir_all(&old).unwrap();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    let other = tiles(&mesh(), TOP, TOP)[40];
    interior(
        &mut project,
        &mut [&mut want],
        other,
        Vec3::new(0.0, 0.25, 0.0),
    );
    let began = Instant::now();
    let failed = loop {
        match project.poll() {
            Ok(()) => {
                assert!(began.elapsed() < Duration::from_secs(30), "the sync fails");
                std::thread::yield_now();
            }
            Err(e) => break e,
        }
    };
    assert!(
        matches!(&failed, ProjectError::OldRecoveryKept { path, .. } if *path == old),
        "{failed:?}"
    );
    assert_ne!(
        project.recovery_path().unwrap(),
        old,
        "rolled over all the same"
    );
    let next = project.poll();
    assert!(
        matches!(&next, Err(ProjectError::Unprotected(tiles)) if *tiles == [lost]),
        "the lost tile, reported next: {next:?}"
    );
    fs::remove_dir(&old).unwrap();
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(
        &dir.path("p.bcrf"),
        &want,
        "the lost tile at its last saved state",
    );
    assert_project_bounds(&dir.path("p.bcrf"), "the lost tile's bounds unknown");
}

/// A roll-over that keeps failing is tried less and less often by the frame loop's calls, the
/// wait doubling; a save tries at once.
#[test]
fn a_failing_roll_over_backs_off_and_a_save_tries_at_once() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 60, 141);
    assert_saved(project.save_as(&path));
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.fail_next_recovery();
    stroke(&mut project, &mut map, 60, 142);
    let began = Instant::now();
    let first = loop {
        match project.poll() {
            Ok(()) => {
                assert!(began.elapsed() < Duration::from_secs(30), "the sync fails");
                std::thread::yield_now();
            }
            Err(e) => break e,
        }
    };
    assert!(paused_by_io(&first), "{first:?}");
    // Due again a second on: it fails a second time, and the wait doubles.
    project.fail_next_recovery();
    let again = Instant::now() + ROLL_OVER_RETRY;
    let second = project.tick(again);
    assert!(second.as_ref().is_err_and(paused_by_io), "{second:?}");
    project.tick(again + ROLL_OVER_RETRY).unwrap();
    assert_eq!(
        project.recovery_path().unwrap(),
        old,
        "a second isn't enough now"
    );
    // A save needs every edit in a recovery file: it doesn't wait.
    assert_saved(project.save());
    assert_ne!(
        project.recovery_path().unwrap(),
        old,
        "rolled over by the save"
    );
    project.close(false).unwrap();
    assert_project_file(&path, &map, "every edit saved");
}

/// Placing a file never replaces one: not one there before, nor one made after a check for it
/// would have passed. Both files are left as they are.
#[test]
fn a_placed_file_never_replaces_another() {
    let dir = TestDir::new();
    let (tmp, path) = (dir.path("new.tmp"), dir.path("p.bcrf"));
    fs::write(&tmp, b"new").unwrap();
    fs::write(&path, b"someone else's").unwrap();
    let placed = place_new(&tmp, &path);
    assert!(
        placed
            .as_ref()
            .is_err_and(|e| e.kind() == io::ErrorKind::AlreadyExists),
        "{placed:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), b"someone else's", "not replaced");
    assert_eq!(fs::read(&tmp).unwrap(), b"new", "nor removed");
    fs::remove_file(&path).unwrap();
    place_new(&tmp, &path).unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"new", "placed");
    assert!(!tmp.exists(), "under its new name only");
}
