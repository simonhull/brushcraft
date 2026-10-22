//! Finding recovery files and restoring them, in one process: which files are listed, the
//! status each gets against its project file, and what a restore of each status does. The
//! crash itself is in `crash`.

use super::*;
use crate::OpId;

/// The recovery files `find_recoveries` lists in `dir`'s recovery directory.
fn found(dir: &TestDir) -> Vec<RecoveryCandidate> {
    find_recoveries(&dir.path("recovery")).unwrap()
}

/// The only candidate listed in `dir`'s recovery directory.
fn only(dir: &TestDir) -> RecoveryCandidate {
    let mut listed = found(dir);
    assert_eq!(listed.len(), 1, "one recovery file: {listed:?}");
    listed.pop().unwrap()
}

/// A project saved at `path` with one stroke, then a second stroke unsaved, dropped as a crash
/// would leave it: its recovery file kept, on the save point. Returns the edits of both.
fn saved_and_dropped(dir: &TestDir, path: &Path, seed: u64) -> CorrectionMap {
    saved_and_dropped_apart(dir, path, seed).1
}

/// [`saved_and_dropped`], returning the edits saved and the edits of both strokes apart.
fn saved_and_dropped_apart(
    dir: &TestDir,
    path: &Path,
    seed: u64,
) -> (CorrectionMap, CorrectionMap) {
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(dir)).unwrap();
    stroke(&mut project, &mut map, 80, seed);
    project.save_as(path).unwrap();
    let saved = map.clone();
    stroke(&mut project, &mut map, 80, seed + 1);
    drop(project);
    (saved, map)
}

/// Restoring `candidate` fails as not restorable with `status`.
fn assert_not_restorable(dir: &TestDir, candidate: &RecoveryCandidate, status: CandidateStatus) {
    match Project::restore(candidate, config(dir)) {
        Err(ProjectError::NotRestorable(got)) => assert_eq!(got, status, "the status"),
        other => panic!("restoring {candidate:?}: {other:?}"),
    }
}

#[test]
fn a_running_instance_s_recovery_file_is_not_listed() {
    let dir = TestDir::new();
    assert_eq!(
        find_recoveries(&dir.path("none")).unwrap(),
        [],
        "a missing directory has no recovery files"
    );
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 60, 1);
    assert_eq!(found(&dir), [], "held by the running project");
    let path = project.recovery_path().to_path_buf();
    let id = project.id();
    // Files that aren't recovery files by name are never opened.
    let recovery = dir.path("recovery");
    let tmp = recovery.join(format!(
        "{}.tmp",
        path.file_name().unwrap().to_str().unwrap()
    ));
    fs::copy(&path, &tmp).unwrap();
    fs::copy(&path, recovery.join("notes.bcrf")).unwrap();
    fs::copy(&path, recovery.join(format!("{id}.bcrf"))).unwrap();
    assert_eq!(found(&dir), [], "still none");

    drop(project);
    let candidate = only(&dir);
    assert_eq!(candidate.path, path, "the project's file, once released");
    assert_eq!(candidate.project, id, "its project");
    assert_eq!(candidate.status, CandidateStatus::NeverSaved, "never saved");
    assert_eq!(
        (candidate.base, candidate.base_path),
        (None, None),
        "no base"
    );
    assert!(candidate.last_edit.is_some(), "it holds edits");
}

#[test]
fn a_restored_project_holds_its_edits_unsaved_and_saves_them() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = saved_and_dropped(&dir, &path, 10);
    let candidate = only(&dir);
    assert_eq!(
        candidate.status,
        CandidateStatus::Restorable,
        "{candidate:?}"
    );
    assert_eq!(
        candidate.base_path.as_deref(),
        Some(path.as_path()),
        "names its project file"
    );
    let point = ProjectFile::open(&path)
        .unwrap()
        .base()
        .unwrap()
        .unwrap()
        .save_point();
    assert_eq!(candidate.base, Some(point), "on its latest save point");

    let mut project = Project::restore(&candidate, config(&dir)).unwrap();
    assert_eq!(project.path(), Some(path.as_path()), "the project file");
    assert_eq!(
        project.recovery_path(),
        candidate.path,
        "goes on in the file"
    );
    assert!(
        project.has_unsaved_edits(),
        "the recovered edits are unsaved"
    );
    let mesh = mesh();
    assert_positions(
        project.store(),
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "restored",
    );
    assert_eq!(found(&dir), [], "held by the restored project");
    stroke(&mut project, &mut map, 40, 12);
    project.save().unwrap();
    project.close(false).unwrap();
    assert_project_file(&path, &map, "the recovered and the new edits, saved");
    assert_eq!(recovery_files(&dir), Vec::<String>::new(), "all closed");
}

#[test]
fn a_deleted_project_file_leaves_its_recovery_file_project_missing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    saved_and_dropped(&dir, &path, 20);
    fs::remove_file(&path).unwrap();
    let candidate = only(&dir);
    assert_eq!(candidate.status, CandidateStatus::ProjectMissing, "deleted");
    assert_eq!(
        candidate.base_path.as_deref(),
        Some(path.as_path()),
        "where"
    );
    assert_not_restorable(&dir, &candidate, CandidateStatus::ProjectMissing);
    assert!(candidate.path.exists(), "kept");
}

#[test]
fn a_project_file_replaced_by_another_project_is_missing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    saved_and_dropped(&dir, &path, 30);
    let mut other = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    other.save_as(&dir.path("other.bcrf")).unwrap();
    other.close(false).unwrap();
    fs::rename(dir.path("other.bcrf"), &path).unwrap();
    let candidate = only(&dir);
    assert_eq!(
        candidate.status,
        CandidateStatus::ProjectMissing,
        "another project's file is under the path"
    );
    assert_not_restorable(&dir, &candidate, CandidateStatus::ProjectMissing);
}

#[test]
fn a_project_saved_since_restores_onto_the_earlier_save_point_and_saves_as_new() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (saved, recovered) = saved_and_dropped_apart(&dir, &path, 40);
    let stale = only(&dir);
    assert_eq!(stale.status, CandidateStatus::Restorable, "before");

    // Opened without restoring, edited and saved: the project moves past the save point.
    let mut project = Project::open(&path, config(&dir)).unwrap();
    let listed = found(&dir);
    assert_eq!(listed.len(), 1, "the open project's own file isn't listed");
    assert_eq!(
        listed[0].status,
        CandidateStatus::ProjectInUse,
        "the project file is held"
    );
    assert_not_restorable(&dir, &stale, CandidateStatus::ProjectInUse);
    let mut later = saved;
    stroke(&mut project, &mut later, 40, 42);
    project.save().unwrap();
    project.close(false).unwrap();
    let later_bytes = fs::read(&path).unwrap();

    let candidate = only(&dir);
    assert_eq!(candidate.path, stale.path, "the old recovery file");
    assert_eq!(
        candidate.status,
        CandidateStatus::RestorableOnEarlierSave,
        "saved since"
    );
    // Listed as restorable onto the latest save, it is restored onto the earlier one all the
    // same: the status is checked again.
    let mut restored = Project::restore(&stale, config(&dir)).unwrap();
    assert_eq!(restored.path(), None, "saves only as a new file");
    let mesh = mesh();
    assert_positions(
        restored.store(),
        &dense(&mesh, &recovered),
        &tiles(&mesh, 1, TOP),
        "the earlier save point and the recovered edits, not the later save",
    );
    assert!(restored.has_unsaved_edits(), "the recovered edits");
    assert!(
        matches!(restored.save(), Err(ProjectError::NoPath)),
        "not into the project file, which has moved on"
    );
    assert_eq!(
        found(&dir),
        [],
        "the restored project holds its recovery file"
    );
    let copy = dir.path("copy.bcrf");
    restored.save_as(&copy).unwrap();
    restored.close(false).unwrap();
    assert_project_file(&copy, &recovered, "saved as a new file");
    assert_eq!(
        fs::read(&path).unwrap(),
        later_bytes,
        "the project file untouched"
    );
    assert_project_file(&path, &later, "the later save is still the latest");
}

#[test]
fn of_two_files_on_one_save_point_the_newest_edit_comes_first() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    saved_and_dropped(&dir, &path, 60);
    let with_edits = only(&dir);
    // An empty file on the same save point, written later: what a crash mid-roll-over leaves.
    let head = with_edits.base.unwrap().head;
    let empty = forge(&dir, &path, head, 3);
    let listed = found(&dir);
    assert_eq!(listed.len(), 2, "both listed: {listed:?}");
    assert!(
        listed[1].modified >= listed[0].modified,
        "the empty one is written later"
    );
    assert_eq!(listed[0].path, with_edits.path, "the one with edits first");
    assert_eq!(listed[1].path, empty, "then the empty one");
    assert_eq!(listed[1].last_edit, None, "it holds no edits");
}

/// Two sessions on one save point, each crashing: the later one's file is listed first.
///
/// The second session opened the project file without restoring the first's recovery file. Its
/// ops start past the first's, so its file, the latest work, comes first, though it holds fewer
/// strokes.
#[test]
fn of_two_sessions_on_one_save_point_the_later_one_comes_first() {
    let dir = TestDir::new();
    let mesh = mesh();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    stroke(&mut project, &mut map, 60, 70);
    project.save_as(&path).unwrap();
    let saved = map.clone();
    for seed in 71..75 {
        stroke(&mut project, &mut map, 60, seed);
    }
    let first = project.recovery_path().to_path_buf();
    drop(project);

    let mut later = saved;
    let mut reopened = Project::open(&path, config(&dir)).unwrap();
    stroke(&mut reopened, &mut later, 60, 80);
    let second = reopened.recovery_path().to_path_buf();
    drop(reopened);

    let listed = found(&dir);
    let paths: Vec<&Path> = listed.iter().map(|c| c.path.as_path()).collect();
    assert_eq!(
        paths,
        [&second, &first],
        "the later session first: {listed:?}"
    );
    assert!(
        listed[0].last_edit > listed[1].last_edit,
        "its last edit is the newer: {listed:?}"
    );
    assert_eq!(listed[0].base, listed[1].base, "on one save point");
    let mut restored = Project::restore(&listed[0], config(&dir)).unwrap();
    assert_positions(
        restored.store(),
        &dense(&mesh, &later),
        &tiles(&mesh, 1, TOP),
        "the save and the later session's one stroke",
    );
}

#[test]
fn stale_temporary_files_are_swept() {
    let dir = TestDir::new();
    let project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let name = project
        .recovery_path()
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    project.close(false).unwrap();
    let recovery = dir.path("recovery");
    let stale = recovery.join(format!("{name}.tmp"));
    // Other recovery file names of the project: suffixes its random one is all but sure not
    // to be.
    let id = &name[..32];
    let fresh = recovery.join(format!("{id}-{:016x}.bcrf.tmp", 0));
    let held = recovery.join(format!("{id}-{:016x}.bcrf.tmp", 1));
    assert_ne!(stale, fresh, "distinct");
    assert_ne!(stale, held, "distinct");
    let other = recovery.join("notes.tmp");
    let old = SystemTime::now() - Duration::from_secs(120);
    for at in [&stale, &fresh, &held, &other] {
        fs::write(at, b"partial").unwrap();
    }
    for at in [&stale, &held, &other] {
        fs::File::options()
            .write(true)
            .open(at)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }
    let lock = fs::File::open(&held).unwrap();
    lock.lock().unwrap();
    assert_eq!(found(&dir), [], "no candidates");
    assert!(!stale.exists(), "a stale leftover is removed");
    assert!(fresh.exists(), "one written just now may be in the making");
    assert!(held.exists(), "one locked is in the making");
    assert!(other.exists(), "not a recovery file's");
}

/// Writes a recovery file in `dir`'s recovery directory whose meta names the project file at
/// `path`, at its latest save point's offset, with `head` as the save point's head op.
fn forge(dir: &TestDir, path: &Path, head: Option<OpId>, suffix: u64) -> PathBuf {
    let project = ProjectFile::open(path).unwrap();
    let point = SavePoint {
        head,
        ..project.base().unwrap().unwrap().save_point()
    };
    let meta = project.meta().on_save_point(point, path);
    let name = format!("{}-{suffix:016x}.bcrf", meta.project());
    let at = dir.path("recovery").join(name);
    drop(ProjectFile::create_like(&at, meta, &project).unwrap());
    at
}

#[test]
fn a_save_point_with_another_head_is_not_the_base() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let map = saved_and_dropped(&dir, &path, 50);
    let mut project = Project::restore(&only(&dir), config(&dir)).unwrap();
    project.save().unwrap();
    project.close(false).unwrap();
    let head = ProjectFile::open(&path).unwrap().head();
    let seq = head.unwrap().seq();
    let other = OpId::new(head.unwrap().site(), seq - 1).unwrap();
    let right = forge(&dir, &path, head, 1);
    let wrong = forge(&dir, &path, Some(other), 2);
    let listed = found(&dir);
    let status = |at: &Path| {
        listed
            .iter()
            .find(|candidate| candidate.path == at)
            .unwrap()
            .status
    };
    assert_eq!(status(&right), CandidateStatus::Restorable, "the same head");
    assert_eq!(
        status(&wrong),
        CandidateStatus::SavePointMissing,
        "the same offset, another head"
    );
    let wrong = listed.iter().find(|c| c.path == wrong).unwrap();
    assert_not_restorable(&dir, wrong, CandidateStatus::SavePointMissing);
    let right = listed.iter().find(|c| c.path == right).unwrap();
    let mut restored = Project::restore(right, config(&dir)).unwrap();
    let mesh = mesh();
    assert_positions(
        restored.store(),
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "the save point alone",
    );
    assert!(!restored.has_unsaved_edits(), "an empty recovery file");
}

#[test]
fn a_damaged_recovery_file_is_listed_damaged_and_not_restored() {
    let dir = TestDir::new();
    let project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let path = project.recovery_path().to_path_buf();
    let id = project.id();
    drop(project);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 0xFF;
    fs::write(&path, &bytes).unwrap();
    let candidate = only(&dir);
    assert_eq!(candidate.status, CandidateStatus::Damaged, "a bad header");
    assert_eq!(candidate.project, id, "named by its file name");
    assert_not_restorable(&dir, &candidate, CandidateStatus::Damaged);
    assert_eq!(fs::read(&path).unwrap(), bytes, "left as it is");
}
