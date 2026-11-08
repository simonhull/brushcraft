//! A save's work done ahead of it, in slices: what the copies save, what an edit or a failure
//! makes of them, and that nothing past the last save point counts until the save.

use std::time::Instant;

use super::*;

/// A project saved at `dir`'s `p.bcrf`, then edited by one stroke, with `map` following it.
fn edited(dir: &TestDir, map: &mut CorrectionMap) -> Project {
    let mut project = Project::new((*mesh()).clone(), config(dir)).unwrap();
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    stroke(&mut project, map, 60, 7);
    project
}

/// Calls [`Project::save_ahead`] until it is ready, one tile at a time; returns how many calls
/// copied.
fn copy_all(project: &mut Project) -> usize {
    let mut copied = 0;
    loop {
        match project.save_ahead(Instant::now()).unwrap() {
            SaveAhead::Ready => return copied,
            SaveAhead::Copying(_) => copied += 1,
            SaveAhead::Flushing => {}
        }
    }
}

/// How long `path` is.
fn length(path: &Path) -> u64 {
    fs::metadata(path).unwrap().len()
}

#[test]
fn a_save_after_copying_ahead_holds_every_edit_and_copies_none_again() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    let before = length(&path);
    assert!(copy_all(&mut project) > 1, "in slices");
    let copied = length(&path) - before;
    let ahead = length(&path);
    assert_saved(project.save());
    let saved = length(&path) - ahead;
    assert!(
        saved < copied,
        "the save wrote {saved}, the copies {copied}"
    );
    assert!(!project.has_unsaved_edits());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "copied ahead");
}

#[test]
fn a_slice_past_its_deadline_copies_one_tile() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    let mut lefts = Vec::new();
    loop {
        match project.save_ahead(Instant::now()).unwrap() {
            SaveAhead::Ready => break,
            SaveAhead::Copying(left) => lefts.push(left),
            SaveAhead::Flushing => {}
        }
    }
    assert!(lefts.len() > 2, "{lefts:?}");
    let mut pairs = lefts.windows(2);
    assert!(
        pairs.all(|pair| matches!(pair, [before, after] if before - after == 1)),
        "{lefts:?}"
    );
    project.close(true).unwrap();
}

#[test]
fn the_next_save_copies_ahead_from_the_start_again() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    copy_all(&mut project);
    assert_saved(project.save());
    stroke(&mut project, &mut map, 60, 8);
    assert!(copy_all(&mut project) > 1, "in slices again");
    assert_saved(project.save());
    project.close(false).unwrap();
}

/// With crash protection paused nothing flushes: a tile edited again is dirty over its old record,
/// and copying that would hide the edit from the save.
#[test]
fn a_dirty_tile_is_left_for_the_save() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    let paused = fail_and_pause(&mut project, &mut map, 8, io::ErrorKind::Other);
    assert!(paused_by_io(&paused), "{paused:?}");
    stroke(&mut project, &mut map, 60, 8);
    copy_all(&mut project);
    drop(project.save().unwrap());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "the stroke no flush took");
}

/// A tile copied, then edited with its slot broken and lost by a roll-over, goes back to its last
/// saved state in the store, so the next save point mustn't name the copy.
#[test]
fn a_tile_lost_after_it_was_copied_stays_at_its_last_saved_state() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let [lost, other]: [TileKey; 2] = tiles(&mesh(), TOP, TOP)[..2].try_into().unwrap();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert_saved(project.save_as(&path));
    let mut want = CorrectionMap::new();
    let mut scratch = CorrectionMap::new();
    interior(
        &mut project,
        &mut [&mut scratch],
        lost,
        Vec3::new(0.0, 0.0, 0.25),
    );
    project.finish_flushes().unwrap();
    copy_all(&mut project);
    let old = project.recovery_path().unwrap().to_path_buf();
    // The sync of the next flush fails while the broken slot is dirty: the roll-over loses it.
    project.writer.as_ref().unwrap().pause(true);
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    interior(
        &mut project,
        &mut [&mut want],
        other,
        Vec3::new(0.25, 0.0, 0.0),
    );
    project.begin_stroke_and_wait().unwrap();
    project
        .store()
        .edit_tile(lost, |edit| edit.set(5, 7, Vec3::new(0.0, 0.0, 0.5)))
        .unwrap()
        .unwrap();
    break_slot(project.store(), &lost);
    project.end_unrecorded_stroke().unwrap();
    project.writer.as_ref().unwrap().pause(false);
    while project.recovery_path().unwrap() == old {
        drop(project.poll());
        std::thread::yield_now();
    }
    drop(project.save().unwrap());
    project.close(false).unwrap();
    assert_project_file(
        &path,
        &want,
        "the other tile, and the lost one as last saved",
    );
}

/// So too a tile no edit touched since its copy, its record found unreadable by a roll-over: the
/// store reads its last saved state from then on, and the copy would differ from that.
#[test]
fn a_tile_whose_record_went_bad_after_it_was_copied_stays_at_its_last_saved_state() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let top = tiles(&mesh(), TOP, TOP);
    let (lost, other) = (top[5], top[30]);
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert_saved(project.save_as(&path));
    let mut want = CorrectionMap::new();
    let mut scratch = CorrectionMap::new();
    let c = Vec3::new(0.0, 0.0, 0.25);
    interior(&mut project, &mut [&mut scratch], lost, c);
    project.finish_flushes().unwrap();
    project.store().evict_all().unwrap();
    copy_all(&mut project);
    let old = project.recovery_path().unwrap().to_path_buf();
    let loc = project.store().working().locate(&lost).unwrap();
    damage(&old, loc);
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    interior(
        &mut project,
        &mut [&mut want],
        other,
        Vec3::new(0.25, 0.0, 0.0),
    );
    while project.recovery_path().unwrap() == old {
        drop(project.poll());
        std::thread::yield_now();
    }
    drop(project.save().unwrap());
    project.close(false).unwrap();
    assert_project_file(
        &path,
        &want,
        "the other tile, and the lost one as last saved",
    );
}

#[test]
fn a_tile_evicted_after_it_was_copied_is_still_saved() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    copy_all(&mut project);
    project.store().evict_all().unwrap();
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "evicted after the copies");
}

#[test]
fn a_roll_over_between_slices_loses_nothing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    while !matches!(
        project.save_ahead(Instant::now()).unwrap(),
        SaveAhead::Copying(_)
    ) {}
    fail_and_roll_over(&mut project, &mut map, 8);
    copy_all(&mut project);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "rolled over between the slices");
}

#[test]
fn a_ticks_flush_between_slices_loses_nothing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    while !matches!(
        project.save_ahead(Instant::now()).unwrap(),
        SaveAhead::Copying(_)
    ) {}
    let edits = random_edits(&project.store().working().mesh(), TOP, 60, 8);
    project.begin_stroke_and_wait().unwrap();
    apply_dabs(project.store(), &mut map, &edits);
    project
        .tick(Instant::now() + Duration::from_secs(5))
        .unwrap();
    copy_all(&mut project);
    project.end_unrecorded_stroke().unwrap();
    copy_all(&mut project);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "a flush between the slices");
}

#[test]
fn an_edit_after_copying_ahead_reaches_the_save() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    copy_all(&mut project);
    stroke(&mut project, &mut map, 60, 8);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "edited after the copies");
}

#[test]
fn an_edit_between_slices_reaches_the_save() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    while !matches!(
        project.save_ahead(Instant::now()).unwrap(),
        SaveAhead::Copying(_)
    ) {}
    stroke(&mut project, &mut map, 60, 8);
    copy_all(&mut project);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "edited between the slices");
}

#[test]
fn a_save_that_failed_writes_what_was_copied_ahead_again() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    let before = length(&path);
    copy_all(&mut project);
    let copied = length(&path) - before;
    project.file.as_mut().unwrap().fail_next_save();
    assert!(project.save().is_err(), "the project file fails");
    assert!(project.has_unsaved_edits(), "nothing is saved");
    let failed = length(&path);
    assert_saved(project.save());
    let again = length(&path) - failed;
    assert!(again >= copied, "wrote {again} of the {copied} in doubt");
    project.close(false).unwrap();
    assert_project_file(&path, &map, "saved again");
}

#[test]
fn a_save_as_after_copying_ahead_holds_every_edit() {
    let dir = TestDir::new();
    let path = dir.path("other.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    copy_all(&mut project);
    assert_saved(project.save_as(&path));
    project.close(false).unwrap();
    assert_project_file(&path, &map, "saved as a new file");
}

#[test]
fn copies_with_no_save_after_them_are_not_part_of_the_project() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = edited(&dir, &mut map);
    copy_all(&mut project);
    assert!(project.has_unsaved_edits(), "copies aren't a save");
    project.close(false).unwrap();
    assert_project_file(&path, &CorrectionMap::new(), "as it was last saved");
}

#[test]
fn a_flush_under_way_is_waited_for_without_blocking() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    project.writer.as_ref().unwrap().pause(true);
    stroke(&mut project, &mut map, 60, 7);
    for _ in 0..3 {
        assert_eq!(
            project.save_ahead(Instant::now()).unwrap(),
            SaveAhead::Flushing,
            "the stroke's flush is held up"
        );
    }
    project.writer.as_ref().unwrap().pause(false);
    copy_all(&mut project);
    assert_saved(project.save());
    project.close(false).unwrap();
}

#[test]
fn a_project_never_saved_has_nowhere_to_copy_ahead_to() {
    let dir = TestDir::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert!(matches!(
        project.save_ahead(Instant::now()),
        Err(ProjectError::NoPath)
    ));
    project.close(false).unwrap();
}
