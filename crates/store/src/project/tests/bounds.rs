//! The spatial index of a project, loaded from the bounds its files keep: after a save, a save
//! as, and a restore that layers the recovery file's bounds over the project file's.

use geom::Level;

use super::crash::poll_until_idle;
use super::*;
use crate::index::SpatialIndex;
use crate::index::tests::{assert_loads_as_built, index_bits};

/// The level indexed: the tests' finest.
pub(super) fn top() -> Level {
    Level::new(TOP).unwrap()
}

/// A stroke of `count` random edits with `seed`, its dabs then a refit of `index` to every tile
/// of the level (if `refit`), so the bounds recorded are those of the stroke's end.
pub(super) fn indexed_stroke(
    project: &mut Project,
    index: &mut SpatialIndex,
    refit: bool,
    seed: u64,
) {
    let mesh = project.store().working().mesh();
    let edits = random_edits(&mesh, TOP, 40, seed);
    project.begin_stroke_and_wait().unwrap();
    apply_dabs(project.store(), &mut CorrectionMap::new(), &edits);
    if refit {
        index
            .update(project.store(), &tiles(&mesh, TOP, TOP))
            .unwrap();
    }
    project.end_unrecorded_stroke().unwrap();
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

/// The bounds an index computes without an edit pending stay out of the recovery file, so a
/// stroke end costs what its edits do; an explicit save writes them into the project file.
#[test]
fn derived_bounds_skip_the_recovery_file_and_reach_the_saved_one() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (_, loaded) = project.load_index(top(), |_, _| {}).unwrap();
    assert_eq!(loaded.stored, 0, "nothing stored: every bound computed");
    project.begin_stroke_and_wait().unwrap();
    project.end_unrecorded_stroke().unwrap();
    poll_until_idle(&mut project);
    assert_eq!(
        project.store().working().bounds_locations().count(),
        0,
        "the recovery file holds none of them"
    );
    assert_saved(project.save_as(&path));
    project.close(false).unwrap();
    let mut opened = Project::open(&path, config(&dir)).unwrap();
    let stats = assert_loads_as_built(opened.store(), top(), "saved");
    assert_eq!(stats.evaluated, 0, "the save wrote every one");
    opened.close(false).unwrap();
}

/// Sets an interior vertex of `key` (a level-[`TOP`] tile) as a stroke of its own, refitting
/// `index` to `refit` before the stroke ends.
fn edit_interior(project: &mut Project, index: &mut SpatialIndex, key: TileKey, refit: &[TileKey]) {
    project.begin_stroke_and_wait().unwrap();
    let region = Region::of_tile(&key);
    project
        .store()
        .set_correction(
            key.face(),
            key.level(),
            region.x0() + 5,
            region.y0() + 7,
            Vec3::new(0.0, 0.0, 0.25),
        )
        .unwrap();
    index.update(project.store(), refit).unwrap();
    project.end_unrecorded_stroke().unwrap();
}

/// A roll-over carries the bounds the failed recovery file held into the new one: those flushed
/// before the failure too, which hide the project file's stale bounds of the edited region.
#[test]
fn a_roll_over_carries_the_flushed_bounds_across() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    let all = tiles(&mesh(), TOP, TOP);
    assert_saved(project.save_as(&path));
    edit_interior(&mut project, &mut index, all[9], &all);
    poll_until_idle(&mut project);
    // A second edit, far off, whose flush fails to sync: the project rolls over.
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    edit_interior(&mut project, &mut index, all[60], &[]);
    let began = Instant::now();
    while project.recovery_path().unwrap() == old {
        project.poll().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the roll-over happens"
        );
        std::thread::yield_now();
    }
    poll_until_idle(&mut project);
    drop(project);
    let candidate = find_recoveries(&dir.path("recovery"))
        .unwrap()
        .pop()
        .unwrap();
    let mut restored = Project::restore(&candidate, config(&dir)).unwrap();
    assert_loads_as_built(restored.store(), top(), "restored after the roll-over");
    restored.close(false).unwrap();
}

/// A roll-over carries the bounds still waiting for a flush into the new recovery file.
///
/// They were edited after the failed flush began; without them the project file's stale bounds
/// would show through under the restored edit.
#[test]
fn a_roll_over_carries_the_bounds_waiting_for_a_flush() {
    for refit in [true, false] {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
        let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
        let all = tiles(&mesh(), TOP, TOP);
        assert_saved(project.save_as(&path));
        // A flush that fails to sync, then, before the roll-over, an edit far off.
        let old = project.recovery_path().unwrap().to_path_buf();
        project.writer.as_ref().unwrap().fail_syncs(0, 1);
        edit_interior(&mut project, &mut index, all[9], &[]);
        project.store().begin_op().unwrap();
        let later = all[60];
        let region = Region::of_tile(&later);
        project
            .store()
            .set_correction(
                later.face(),
                later.level(),
                region.x0() + 5,
                region.y0() + 7,
                Vec3::new(0.0, 0.25, 0.0),
            )
            .unwrap();
        if refit {
            index.update(project.store(), &all).unwrap();
        }
        let began = Instant::now();
        while project.recovery_path().unwrap() == old {
            project.poll().unwrap();
            assert!(
                began.elapsed() < Duration::from_secs(30),
                "the roll-over happens"
            );
            std::thread::yield_now();
        }
        drop(project);
        let candidate = find_recoveries(&dir.path("recovery"))
            .unwrap()
            .pop()
            .unwrap();
        let mut restored = Project::restore(&candidate, config(&dir)).unwrap();
        assert_loads_as_built(
            restored.store(),
            top(),
            &format!("restored after the roll-over, refitted {refit}"),
        );
        restored.close(false).unwrap();
    }
}

/// A bound an index records over an edit's unknown marker still waiting for a flush is flushed:
/// the recovery file must not keep an older known bound under the newer edit.
#[test]
fn a_bound_recorded_over_a_pending_edit_is_flushed() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    let mesh = mesh();
    let all = tiles(&mesh, TOP, TOP);
    // Saved, so the project file holds every bound; then an edit, refitted, streamed.
    assert_saved(project.save_as(&path));
    let key = all[9];
    project.begin_stroke_and_wait().unwrap();
    let region = Region::of_tile(&key);
    project
        .store()
        .set_correction(
            key.face(),
            key.level(),
            region.x0() + 5,
            region.y0() + 7,
            Vec3::new(0.0, 0.0, 0.25),
        )
        .unwrap();
    index.update(project.store(), &all).unwrap();
    project.end_unrecorded_stroke().unwrap();
    poll_until_idle(&mut project);
    let flushed = project.store().working().bounds_locations().count();
    assert!(
        flushed > 0 && flushed < all.len(),
        "the edited region's bounds, and only they, are flushed: {flushed}"
    );
    drop(project);
    let candidate = find_recoveries(&dir.path("recovery"))
        .unwrap()
        .pop()
        .unwrap();
    let mut restored = Project::restore(&candidate, config(&dir)).unwrap();
    let stats = assert_loads_as_built(restored.store(), top(), "restored");
    assert_eq!(stats.evaluated, 0, "every bound kept: {stats:?}");
    restored.close(false).unwrap();
}
