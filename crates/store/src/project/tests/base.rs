//! The store's base, checked against its save point rebuilt in full from the file.
//!
//! The base, the last save point as the store reads it, is derived from the save point before
//! plus what each save wrote. It is checked after every kind of save and every way a project
//! moves between files.

use super::bounds::{indexed_stroke, top};
use super::*;
use crate::file::{BaseFile, compact_in_place};

/// Asserts that the project's base is its project file's latest save point, rebuilt in full.
///
/// Both the base its store holds and the one its project file makes now are compared with the
/// save point rebuilt from the file's Index chain: the same tiles and bounds, bit for bit, and
/// the same save point.
fn assert_base_rebuilds(project: &Project, what: &str) {
    let file = project.file.as_ref().expect("a saved project has its file");
    let held = project.store.base().expect("a saved project has a base");
    let made = file.base().expect("the file has a save point");
    let rebuilt = file
        .base_at(held.save_point())
        .unwrap()
        .expect("the store's save point is the file's");
    assert!(held.same_state(&rebuilt), "{what}: the store's base");
    assert!(made.same_state(&rebuilt), "{what}: the file's base");
    assert_eq!(
        held.save_point().offset,
        *file.save_points().unwrap().last().unwrap(),
        "{what}: the latest save point"
    );
}

/// How many tiles a project's base holds.
fn base_len(project: &Project) -> usize {
    project.store.base().map_or(0, BaseFile::len)
}

#[test]
fn the_base_is_the_rebuilt_save_point_after_every_kind_of_save() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    indexed_stroke(&mut project, &mut index, true, 1);
    assert_saved(project.save_as(&path));
    assert_base_rebuilds(&project, "saved as");

    // More saves than a full Index's period, so the chain passes a full one (every 16th) and
    // the base goes on deriving across it. Some strokes refit the index (bounds written), some
    // don't, and some saves have nothing new.
    for k in 0..20_u64 {
        if k % 5 != 4 {
            indexed_stroke(&mut project, &mut index, k % 3 != 0, 10 + k);
        }
        let before = base_len(&project);
        assert_saved(project.save());
        assert_base_rebuilds(&project, &format!("save {k}"));
        assert!(
            base_len(&project) >= before,
            "save {k}: a save takes no tile out"
        );
    }

    // A roll-over moves the edits to a new recovery file on the same save point.
    let mut map = CorrectionMap::new();
    fail_and_roll_over(&mut project, &mut map, 40);
    assert_base_rebuilds(&project, "rolled over");
    assert_saved(project.save());
    assert_base_rebuilds(&project, "saved after the roll-over");

    // Compaction rewrites the file; the base is loaded from it, then derived again.
    project.close(false).unwrap();
    compact_in_place(&path).unwrap();
    project = Project::open(&path, config(&dir)).unwrap();
    assert_base_rebuilds(&project, "opened after compaction");
    (index, _) = project.load_index(top(), |_, _| {}).unwrap();
    indexed_stroke(&mut project, &mut index, true, 50);
    assert_saved(project.save());
    assert_base_rebuilds(&project, "saved after compaction");

    // A restore layers a recovery file on the latest save point; its edits then save.
    indexed_stroke(&mut project, &mut index, true, 60);
    drop(project);
    let candidate = find_recoveries(&dir.path("recovery")).unwrap().remove(0);
    project = Project::restore(&candidate, config(&dir)).unwrap();
    assert_base_rebuilds(&project, "restored");
    assert_saved(project.save());
    assert_base_rebuilds(&project, "saved after the restore");

    // Save as starts a new file with the last save point's records, then derives in it.
    let other = dir.path("q.bcrf");
    (index, _) = project.load_index(top(), |_, _| {}).unwrap();
    indexed_stroke(&mut project, &mut index, true, 70);
    assert_saved(project.save_as(&other));
    assert_base_rebuilds(&project, "saved as a new file");
    indexed_stroke(&mut project, &mut index, false, 71);
    assert_saved(project.save());
    assert_base_rebuilds(&project, "saved in the new file");
    project.close(false).unwrap();
}
