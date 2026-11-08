//! Stroke records end to end: each stroke's record goes into the recovery file with its edits,
//! into the project file with a save, across a roll-over and a save as, and back after a restore.

use super::*;
use crate::StrokeRecord;
use crate::stroke_record::tests::wobbly;

/// One stroke, as [`stroke`] makes one, ended with a record of it; returns the record.
fn recorded_stroke(
    project: &mut Project,
    map: &mut CorrectionMap,
    count: u32,
    seed: u64,
) -> StrokeRecord {
    let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
    let op = project.begin_stroke_and_wait().unwrap();
    apply_dabs(project.store(), map, &edits);
    let record = wobbly(op, 30, seed);
    project.end_stroke(&record).unwrap();
    record
}

/// Every stroke record `file` holds, read back.
fn strokes_in(file: &ProjectFile) -> Vec<StrokeRecord> {
    file.strokes().map(Result::unwrap).collect()
}

/// The stroke records of the project's recovery file, its working file.
fn recovered(project: &mut Project) -> Vec<StrokeRecord> {
    strokes_in(project.store().working())
}

#[test]
fn each_strokes_record_goes_with_its_edits_into_every_file() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh).clone(), config(&dir)).unwrap();
    let first = recorded_stroke(&mut project, &mut map, 60, 1);
    crash::poll_until_idle(&mut project);
    assert_eq!(recovered(&mut project), vec![first.clone()], "flushed");
    assert!(
        !project.store().working().has_unsaved_records(),
        "a save point holds it"
    );
    // A stroke with no record keeps none.
    stroke(&mut project, &mut map, 20, 2);
    let third = recorded_stroke(&mut project, &mut map, 60, 3);
    let path = dir.path("p.bcrf");
    assert_saved(project.save_as(&path));
    let saved = vec![first, third];
    assert_eq!(
        strokes_in(project.file.as_ref().unwrap()),
        saved,
        "saved as"
    );
    assert!(recovered(&mut project).is_empty(), "a fresh recovery file");

    let fourth = recorded_stroke(&mut project, &mut map, 60, 4);
    assert_saved(project.save());
    let mut all = saved;
    all.push(fourth);
    assert_eq!(
        strokes_in(project.file.as_ref().unwrap()),
        all,
        "saved again"
    );
    project.close(false).unwrap();
    assert_eq!(
        strokes_in(&ProjectFile::open(&path).unwrap()),
        all,
        "reopened"
    );
    assert_project_file(&path, &map, "the edits beside them");

    // Save as copies the old file's records, then adds the new.
    let mut reopened = Project::open(&path, config(&dir)).unwrap();
    let fifth = recorded_stroke(&mut reopened, &mut map, 60, 5);
    let other = dir.path("q.bcrf");
    assert_saved(reopened.save_as(&other));
    reopened.close(false).unwrap();
    all.push(fifth);
    assert_eq!(
        strokes_in(&ProjectFile::open(&other).unwrap()),
        all,
        "saved under a new name"
    );
}

#[test]
fn a_record_of_another_stroke_is_refused() {
    let dir = TestDir::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let op = project.begin_stroke_and_wait().unwrap();
    let other = wobbly(op.next().unwrap(), 5, 1);
    assert!(
        matches!(
            project.end_stroke(&other),
            Err(ProjectError::NotTheStroke { record, stroke }) if record == other.op() && stroke == Some(op)
        ),
        "another op"
    );
    let own = wobbly(op, 5, 1);
    project.end_stroke(&own).unwrap();
    assert!(
        matches!(project.end_stroke(&own), Err(ProjectError::AlreadyRecorded(again)) if again == op),
        "a second record of the op"
    );
    project.finish_flushes().unwrap();
    assert_eq!(recovered(&mut project), vec![own], "kept once");
    project.close(false).unwrap();
}

/// Every stroke record of the project file at `path`, read back.
fn records_at(path: &Path) -> Vec<StrokeRecord> {
    strokes_in(&ProjectFile::open(path).unwrap())
}

#[test]
fn a_save_made_again_after_its_project_file_failed_writes_each_record_once() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let first = recorded_stroke(&mut project, &mut map, 60, 41);
    assert_saved(project.save_as(&path));
    let second = recorded_stroke(&mut project, &mut map, 60, 42);
    project.file.as_mut().unwrap().fail_next_save();
    assert!(project.save().is_err(), "the project file fails");
    // The records were copied before the failure, with no save point after them.
    assert_saved(project.save());
    let third = recorded_stroke(&mut project, &mut map, 60, 43);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_eq!(
        records_at(&path),
        vec![first, second, third],
        "each record once"
    );
    assert_project_file(&path, &map, "with every edit");
}

#[test]
fn a_save_whose_rebase_never_came_is_made_again_with_each_record_once() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let first = recorded_stroke(&mut project, &mut map, 60, 51);
    assert_saved(project.save_as(&path));
    let second = recorded_stroke(&mut project, &mut map, 60, 52);
    project.finish_flushes().unwrap();
    // A save point written, then the store left where it was, as a refused rebase leaves it.
    let saved = project
        .store
        .save_into(project.file.as_mut().unwrap())
        .unwrap();
    drop(saved);
    assert_eq!(records_count(&project), 2, "the save point holds both");
    assert_saved(project.save());
    assert_eq!(records_count(&project), 2, "not again");
    project.close(false).unwrap();
    assert_eq!(records_at(&path), vec![first, second], "each record once");
}

/// How many stroke records the project's project file holds.
fn records_count(project: &Project) -> usize {
    project.file.as_ref().unwrap().stroke_count()
}

#[test]
fn a_save_in_the_middle_of_a_stroke_keeps_its_record_once() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mesh = project.store().working().mesh();
    let op = project.begin_stroke_and_wait().unwrap();
    apply_dabs(project.store(), &mut map, &random_edits(&mesh, TOP, 40, 61));
    assert_saved(project.save_as(&path));
    // The save ended the store's op: the stroke's later edits take a new one.
    apply_dabs(project.store(), &mut map, &random_edits(&mesh, TOP, 40, 62));
    let record = wobbly(op, 30, 61);
    project.end_stroke(&record).unwrap();
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_eq!(records_at(&path), vec![record], "kept, once");
    assert_project_file(&path, &map, "with every edit of the stroke");
}

#[test]
fn a_roll_over_keeps_a_record_waiting_behind_an_in_flight_flush() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    project.finish_flushes().unwrap();
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    // The first stroke's flush is held on the writer thread; the second's record waits for it.
    project.writer.as_ref().unwrap().pause(true);
    let first = recorded_stroke(&mut project, &mut map, 60, 71);
    let second = recorded_stroke(&mut project, &mut map, 60, 72);
    assert!(project.store().has_waiting_strokes(), "behind the flush");
    project.writer.as_ref().unwrap().pause(false);
    let began = Instant::now();
    while project.recovery_path().unwrap() == old {
        project.poll().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the roll-over happens"
        );
        std::thread::yield_now();
    }
    // Before any later flush: the new file's save point holds both strokes' edits and records.
    assert!(!project.store().has_waiting_strokes(), "carried across");
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    let file = ProjectFile::open(&path).unwrap();
    assert_eq!(strokes_in(&file), vec![first, second], "both records");
    let mesh = file.mesh();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(
        &mut store,
        &dense(&mesh, &map),
        &tiles(&mesh, 1, TOP),
        "both strokes' edits",
    );
}

#[test]
fn no_open_or_restore_reads_a_stroke_record() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    recorded_stroke(&mut project, &mut map, 60, 81);
    assert_saved(project.save_as(&path));
    recorded_stroke(&mut project, &mut map, 60, 82);
    let recovery = project.recovery_path().unwrap().to_path_buf();
    // Dropped as a crash would leave it: the recovery file kept.
    drop(project);
    crate::file::take_stroke_reads();
    let looked = ProjectFile::inspect(&recovery).unwrap();
    assert_eq!(looked.stroke_count(), 1, "found by its head");
    drop(looked);
    let base = ProjectFile::open(&path).unwrap();
    let beside = ProjectFile::open_sharing_mesh(&recovery, &base).unwrap();
    drop((beside, base));
    let candidate = find_recoveries(&dir.path("recovery"))
        .unwrap()
        .pop()
        .unwrap();
    let restored = Project::restore(&candidate, config(&dir)).unwrap();
    restored.close(true).unwrap();
    let opened = Project::open(&path, config(&dir)).unwrap();
    opened.close(false).unwrap();
    assert_eq!(
        crate::file::take_stroke_reads(),
        0,
        "no stroke record read to open or restore"
    );
}

#[test]
fn compaction_drops_every_stroke_record() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    recorded_stroke(&mut project, &mut map, 60, 91);
    assert_saved(project.save_as(&path));
    project.close(false).unwrap();
    assert_eq!(records_at(&path).len(), 1, "saved");
    let compacted = dir.path("c.bcrf");
    crate::compact(&path, &compacted).unwrap();
    assert_eq!(
        ProjectFile::open(&compacted).unwrap().stroke_count(),
        0,
        "history, which compaction doesn't keep (yet)"
    );
    assert_project_file(&compacted, &map, "the edits kept");
}

#[test]
fn a_restored_recovery_file_keeps_its_strokes_records_and_saves_them() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let saved = recorded_stroke(&mut project, &mut map, 60, 11);
    assert_saved(project.save_as(&path));
    let unsaved = recorded_stroke(&mut project, &mut map, 60, 12);
    // Dropped as a crash would leave it: the recovery file kept.
    drop(project);
    let mut listed = find_recoveries(&dir.path("recovery")).unwrap();
    let candidate = listed.pop().unwrap();
    let mut restored = Project::restore(&candidate, config(&dir)).unwrap();
    assert_eq!(
        recovered(&mut restored),
        vec![unsaved.clone()],
        "the stroke since the save"
    );
    assert_saved(restored.save());
    restored.close(false).unwrap();
    assert_eq!(
        strokes_in(&ProjectFile::open(&path).unwrap()),
        vec![saved, unsaved],
        "saved after the restore"
    );
    assert_project_file(&path, &map, "with their edits");
}

#[test]
fn a_roll_over_carries_the_records_and_one_waiting_follows() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let first = recorded_stroke(&mut project, &mut map, 60, 21);
    project.finish_flushes().unwrap();
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    let second = recorded_stroke(&mut project, &mut map, 60, 22);
    let began = Instant::now();
    while project.recovery_path().unwrap() == old {
        project.poll().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the roll-over happens"
        );
        std::thread::yield_now();
    }
    crash::poll_until_idle(&mut project);
    assert_eq!(
        recovered(&mut project),
        vec![first.clone(), second.clone()],
        "both, in the new file"
    );
    let third = recorded_stroke(&mut project, &mut map, 60, 23);
    let path = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_eq!(
        strokes_in(&ProjectFile::open(&path).unwrap()),
        vec![first, second, third],
        "and the next"
    );
}

#[test]
fn records_wait_while_protection_is_paused_and_a_save_keeps_them() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let first = recorded_stroke(&mut project, &mut map, 60, 31);
    assert_saved(project.save_as(&path));
    project.fail_next_recovery();
    let outcome = project.save().unwrap();
    assert!(!outcome.protection.is_active(), "paused: {outcome:?}");
    let second = recorded_stroke(&mut project, &mut map, 60, 32);
    assert!(
        project.store().has_waiting_strokes(),
        "no writer to take it"
    );
    assert_saved(project.save());
    assert!(!project.store().has_waiting_strokes(), "the save took it");
    project.close(false).unwrap();
    assert_eq!(
        strokes_in(&ProjectFile::open(&path).unwrap()),
        vec![first, second],
        "both saved, once each"
    );
}
