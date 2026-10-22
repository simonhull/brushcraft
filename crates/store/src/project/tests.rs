//! Projects end to end, against the dense evaluation: new, save as, save, close and open, the
//! recovery file each save starts, and what a failed save leaves.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use geom::CorrectionMap;

use super::*;
use crate::test_dir::TestDir;
use crate::tiles::tests::{TOP, apply, assert_positions, dense, mesh, random_edits, roomy, tiles};

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

/// `count` random edits with `seed` applied to the project's store and to `map`, then a stroke
/// end.
fn stroke(project: &mut Project, map: &mut CorrectionMap, count: u32, seed: u64) {
    let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
    apply(project.store(), map, &edits);
    project.end_stroke().unwrap();
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
    project.save_as(&path).unwrap();
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
    project.save_as(&path).unwrap();
    states.push(map.clone());
    for seed in [12, 13] {
        stroke(&mut project, &mut map, 120, seed);
        project.save().unwrap();
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
    let first = project.recovery_path().to_path_buf();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 100, 21);
    project.save_as(&path).unwrap();
    let new_id = project.id();
    assert_ne!(new_id, id, "a saved-as project has an id of its own");
    stroke(&mut project, &mut map, 100, 22);
    project.save().unwrap();
    let recovery = project.recovery_path().to_path_buf();
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
    first.save_as(&old).unwrap();
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
    project.save_as(&new).unwrap();
    assert_ne!(project.id(), old_id, "a new id");
    // The old file is no longer held: it opens elsewhere.
    assert_eq!(fs::read(&old).unwrap(), old_bytes, "the old file untouched");
    assert_project_file(&old, &saved, "the old file's state");
    stroke(&mut project, &mut map, 50, 33);
    project.save().unwrap();
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

#[test]
fn a_save_that_cannot_make_its_recovery_file_keeps_the_old_one() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let mut map = CorrectionMap::new();
    stroke(&mut project, &mut map, 100, 41);
    project.save_as(&path).unwrap();
    stroke(&mut project, &mut map, 100, 42);
    let old = project.recovery_path().to_path_buf();
    let old_name = old.file_name().unwrap().to_str().unwrap().to_owned();

    // A crash after the new recovery file is written, before it is renamed into place.
    project.fail_next_recovery();
    let failed = project.save();
    assert!(matches!(failed, Err(ProjectError::Io(_))), "{failed:?}");
    let names = recovery_files(&dir);
    assert!(old.exists(), "the old recovery file is kept");
    assert!(
        names
            .iter()
            .all(|name| *name == old_name || is_temporary(name)),
        "no new recovery file under a real name: {names:?}"
    );
    assert!(project.has_unsaved_edits(), "still unsaved");
    assert_eq!(project.recovery_path(), old, "still writing the old one");
    // The old file still holds every edit, at a durable save point, over the project's
    // previous save point.
    let working = project.store().working();
    assert!(working.keys().count() > 0, "the edits are still there");

    // Editing and saving go on.
    stroke(&mut project, &mut map, 50, 43);
    project.save().unwrap();
    assert!(
        !old.exists(),
        "the old file goes once the new one is in place"
    );
    project.close(false).unwrap();
    assert_project_file(&path, &map, "every edit saved");
}

#[test]
fn close_keeps_the_recovery_file_only_when_asked() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 51);
    let kept = project.recovery_path().to_path_buf();
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
    let gone = other.recovery_path().to_path_buf();
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
    let path = project.recovery_path().to_path_buf();
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
    let recovery = project.recovery_path().to_path_buf();
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
    let old = project.recovery_path().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    stroke(project, map, 80, seed);
    let began = Instant::now();
    while project.recovery_path() == old {
        project.poll().unwrap();
        assert!(
            began.elapsed() < std::time::Duration::from_secs(30),
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
    let path = project.recovery_path().to_path_buf();
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

#[test]
fn a_saved_project_rolls_over_on_its_save_point_and_saves_after() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 80, 81);
    project.save_as(&path).unwrap();
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
    project.save().unwrap();
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
    let old = project.recovery_path().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.fail_next_recovery();
    stroke(&mut project, &mut map, 60, 92);
    let began = Instant::now();
    let failed = loop {
        match project.poll() {
            Ok(()) => {
                assert!(
                    began.elapsed() < std::time::Duration::from_secs(30),
                    "the sync fails"
                );
                std::thread::yield_now();
            }
            Err(e) => break e,
        }
    };
    assert!(matches!(failed, ProjectError::Io(_)), "{failed:?}");
    assert!(
        old.exists(),
        "the old file is kept until a new one is in place"
    );
    assert_eq!(project.recovery_path(), old, "still on the old file");
    // The next call tries again, and gets there.
    project.poll().unwrap();
    assert_ne!(project.recovery_path(), old, "rolled over");
    assert!(!old.exists(), "and only then the old file goes");
    let path = project.recovery_path().to_path_buf();
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
