//! The spatial index of a project, loaded from the bounds its files keep: after a save, a save
//! as, and a restore that layers the recovery file's bounds over the project file's.

use geom::Level;

use super::crash::poll_until_idle;
use super::*;
use crate::index::SpatialIndex;
use crate::index::tests::{assert_loads_as_built, index_bits};

/// The level indexed: the tests' finest.
fn top() -> Level {
    Level::new(TOP).unwrap()
}

/// A stroke of `count` random edits with `seed`, its dabs then a refit of `index` to every tile
/// of the level (if `refit`), so the bounds recorded are those of the stroke's end.
fn indexed_stroke(project: &mut Project, index: &mut SpatialIndex, refit: bool, seed: u64) {
    let mesh = project.store().working().mesh();
    let edits = random_edits(&mesh, TOP, 40, seed);
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), &mut CorrectionMap::new(), &edits);
    if refit {
        index
            .update(project.store(), &tiles(&mesh, TOP, TOP))
            .unwrap();
    }
    project.end_stroke().unwrap();
}

#[test]
fn an_opened_project_loads_the_index_it_saved() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (mut index, fresh) = project.load_index(top(), |_, _| {}).unwrap();
    assert_eq!(fresh.stored, 0, "a new project keeps no bounds");
    indexed_stroke(&mut project, &mut index, true, 1);
    assert_saved(project.save_as(&path));
    indexed_stroke(&mut project, &mut index, true, 2);
    assert_saved(project.save());
    project.close(false).unwrap();

    let mut opened = Project::open(&path, config(&dir)).unwrap();
    let stats = assert_loads_as_built(opened.store(), top(), "opened");
    assert_eq!(stats.evaluated, 0, "every bound saved");
    opened.close(false).unwrap();
}

#[test]
fn a_save_as_carries_the_bounds_of_the_last_save_point() {
    let dir = TestDir::new();
    let (first, second) = (dir.path("p.bcrf"), dir.path("q.bcrf"));
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    indexed_stroke(&mut project, &mut index, true, 3);
    assert_saved(project.save_as(&first));
    // Edits since, refitted: the new file holds the saved bounds and these over them.
    indexed_stroke(&mut project, &mut index, true, 4);
    assert_saved(project.save_as(&second));
    project.close(false).unwrap();

    let mut opened = Project::open(&second, config(&dir)).unwrap();
    let stats = assert_loads_as_built(opened.store(), top(), "saved as");
    assert_eq!(stats.evaluated, 0, "every bound carried into the new file");
    opened.close(false).unwrap();
}

/// A restore layers the recovery file's bounds over the project file's.
///
/// After a save, strokes stream into the recovery file, and the project is dropped as a crash
/// would leave it. What the strokes changed is read from the recovery file, never the project
/// file's stale bounds.
#[test]
fn a_restored_project_loads_its_bounds_through_the_recovery_file() {
    for refit in [true, false] {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
        let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
        indexed_stroke(&mut project, &mut index, true, 5);
        assert_saved(project.save_as(&path));
        indexed_stroke(&mut project, &mut index, refit, 6);
        indexed_stroke(&mut project, &mut index, refit, 7);
        poll_until_idle(&mut project);
        drop(project);

        let mut listed = find_recoveries(&dir.path("recovery")).unwrap();
        assert_eq!(listed.len(), 1, "{listed:?}");
        let candidate = listed.pop().unwrap();
        assert_eq!(candidate.status, CandidateStatus::Restorable);
        let mut restored = Project::restore(&candidate, config(&dir)).unwrap();
        let what = format!("restored, refitted {refit}");
        // Loaded, not built: a build would record every bound again, and the save below
        // would copy them whatever the restore had kept.
        let (loaded, stats) = restored.load_index(top(), |_, _| {}).unwrap();
        if refit {
            assert_eq!(stats.evaluated, 0, "{what}: every bound kept");
        } else {
            assert!(
                stats.evaluated > 0,
                "{what}: the strokes' bounds are unknown"
            );
        }
        // The restored edits save into the project file, with the recovery file's bounds and
        // the load's.
        assert_saved(restored.save());
        restored.close(false).unwrap();
        let mut opened = Project::open(&path, config(&dir)).unwrap();
        let (reloaded, again) = opened.load_index(top(), |_, _| {}).unwrap();
        assert_eq!(again.evaluated, 0, "{what}: saved with the edits");
        assert!(
            index_bits(&reloaded) == index_bits(&loaded),
            "{what}: the saved index is the restored one"
        );
        let built = SpatialIndex::build(opened.store(), top(), |_, _| {}).unwrap();
        assert!(
            index_bits(&built) == index_bits(&loaded),
            "{what}: and the one built from the restored tiles, bit for bit"
        );
        opened.close(false).unwrap();
    }
}
