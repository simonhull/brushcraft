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
    let op = project.begin_stroke().unwrap();
    apply_dabs(project.store(), map, &edits);
    let record = wobbly(op, 30, seed);
    project.end_stroke_with(&record).unwrap();
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
    let op = project.begin_stroke().unwrap();
    let other = wobbly(op.next().unwrap(), 5, 1);
    assert!(
        matches!(
            project.end_stroke_with(&other),
            Err(ProjectError::NotTheStroke { record, stroke }) if record == other.op() && stroke == Some(op)
        ),
        "another op"
    );
    project.end_stroke_with(&wobbly(op, 5, 1)).unwrap();
    project.close(false).unwrap();
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
