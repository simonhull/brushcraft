//! A roll-over carries the open stroke.
//!
//! The fresh file holds its pen-down surface in a save point of its own, written from the shapes
//! the store keeps in memory, and its journal from the landed count, so a crash after it restores
//! the stroke as a crash before it would have.

use super::*;
use crate::tiles::PenDownPoint;

/// Fails the writer's next sync and drives the flushes until the project has rolled over onto a
/// fresh recovery file. Returns the fresh file's save points.
fn fail_and_roll_over(project: &mut Project) -> Vec<u64> {
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.finish_flushes().unwrap();
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    project.store().working().save_points().unwrap()
}

/// Crashes `project` where it is: its writer is let go with nothing more flushed, and the
/// project dropped with its recovery file as the last durable save point left it.
fn crash_now(mut project: Project) {
    project.writer.take().unwrap().abandon(&mut project.store);
    drop(project);
}

/// The head op of the save point at `offset` of the recovery file at `path`.
fn head_at(path: &Path, offset: u64) -> Option<OpId> {
    ProjectFile::open_at(path, offset).unwrap().head()
}

/// Asserts that `restored` goes on with `stroke` as the live project would have.
///
/// Its open stroke is `stroke`'s, `landed` of its 30 dabs landed, its pen-down at `pen_down`,
/// and its shapes, once resumed, `live`'s of exactly the tiles its first `landed` dabs changed,
/// bit for bit; landing the rest gives `map`'s tiles with every dab landed. Closes it.
fn assert_resumes_as_live(
    mut restored: Project,
    stroke: Stroke,
    map: &mut CorrectionMap,
    live: &BTreeMap<TileKey, PenDownShape>,
    pen_down: PenDown,
    landed: u32,
) {
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!(open.op(), stroke.op, "its op");
    assert_eq!(open.landed(), landed, "its landed count");
    assert_eq!(open.dabs(), dabs(landed..30), "the dabs not landed");
    assert_eq!(open.dab_format(), FORMAT, "their format");
    assert_eq!(open.pen_down(), pen_down, "the fresh file's pen-down cut");
    let mesh = mesh();
    let changed = keys_of(&mesh, &stroke.edits[..landed as usize]);
    let keys: BTreeSet<TileKey> = open.pen_down_keys().copied().collect();
    assert_eq!(keys, changed, "the tiles the landed dabs changed");
    let mut resumed = going_on(stroke, &open);
    restored.resume_stroke(open).unwrap();
    for key in &changed {
        assert_eq!(
            restored.store().pen_down_shape(key),
            live.get(key),
            "{key:?}: the live shape, bit for bit"
        );
    }
    resumed.land(&mut restored, map, 30 - landed);
    restored.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut restored, map, "every dab landed");
    restored.close(false).unwrap();
}

/// A project with a stroke landing, its live shapes kept: the project, the stroke and its shapes.
///
/// A stroke of 300 dabs is flushed first, so the next stroke's tiles have records at its
/// pen-down; the stroke of 30, begun with `seed`, is placed whole, 6 landed and flushed, then 3
/// more landed.
fn landing_with_shapes(
    dir: &TestDir,
    map: &mut CorrectionMap,
    seed: u64,
) -> (Project, Stroke, BTreeMap<TileKey, PenDownShape>) {
    let mut project = new_project(dir);
    stroke(&mut project, map, 300, seed);
    flush(&mut project);
    let mut landing = begin_on_top(&mut project, 30, seed + 1);
    let live = live_shapes(&mut project, &landing);
    landing.place(&mut project, 30);
    landing.land(&mut project, map, 6);
    flush(&mut project);
    landing.land(&mut project, map, 3);
    (project, landing, live)
}

#[test]
fn a_roll_over_mid_stroke_keeps_the_pen_down_surface_and_the_journal() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let (mut project, mut landing, live) = landing_with_shapes(&dir, &mut map, 80);
    let points = fail_and_roll_over(&mut project);
    let [cut, _] = points[..] else {
        panic!("the pen-down cut and the fill's own: {points:?}");
    };
    let pen_down = PenDown::At(cut);
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(pen_down)),
        "the journal names the pen-down cut"
    );
    let op = landing.op;
    let carried = batch_at(op, pen_down, 9..30, 9).with_dab_format(FORMAT);
    assert_eq!(batches(&mut project), vec![carried.clone()], "from 9 on");
    // The journal goes on from what the fresh file holds.
    landing.land(&mut project, &mut map, 2);
    flush(&mut project);
    let count = batch_at(op, pen_down, 30..30, 11).with_dab_format(FORMAT);
    assert_eq!(
        batches(&mut project),
        vec![carried, count],
        "the count alone"
    );
    crash_now(project);
    assert_resumes_as_live(restore(&dir), landing, &mut map, &live, pen_down, 11);
}

#[test]
fn a_roll_over_before_the_next_flush_still_restores_the_stroke() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let (mut project, landing, live) = landing_with_shapes(&dir, &mut map, 82);
    let points = fail_and_roll_over(&mut project);
    let path = project.recovery_path().unwrap().to_path_buf();
    crash_now(project);
    let [cut, fill] = points[..] else {
        panic!("the pen-down cut and the fill's own: {points:?}");
    };
    assert!(
        head_at(&path, cut) < Some(landing.op),
        "the pen-down cut's head is before the stroke's op"
    );
    assert_eq!(head_at(&path, fill), Some(landing.op), "the fill's is its");
    assert_resumes_as_live(restore(&dir), landing, &mut map, &live, PenDown::At(cut), 9);
}

#[test]
fn a_roll_over_with_no_open_stroke_writes_no_pen_down_save_point() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 200, 84);
    flush(&mut project);
    stroke(&mut project, &mut map, 200, 85);
    let points = fail_and_roll_over(&mut project);
    assert_eq!(points.len(), 1, "the fill's alone: {points:?}");
    assert_eq!(batches(&mut project), Vec::new(), "no batch");
    crash_now(project);
    let mut restored = restore(&dir);
    assert!(restored.take_open_stroke().is_none(), "no stroke");
    assert_tiles(&mut restored, &map, "every edit");
    restored.close(false).unwrap();
}

#[test]
fn a_roll_over_while_the_stroke_waits_for_its_cut() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    // Stroke `a` lands behind a flush held in its tile sync, then ends: `b`, begun after,
    // waits for the cut of the flush queued behind it.
    let mut a = begin_on_top(&mut project, 30, 86);
    a.place(&mut project, 30);
    a.land(&mut project, &mut map, 20);
    project.writer.as_ref().unwrap().pause_syncs(true);
    project.tick(Instant::now() + FLUSH_INTERVAL * 100).unwrap();
    project
        .writer
        .as_mut()
        .unwrap()
        .step_blocking(&mut project.store)
        .unwrap();
    a.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    let edits = top_edits(&mesh(), 30, 87);
    let op = project.begin_stroke().unwrap();
    let mut b = Stroke {
        op,
        edits,
        placed: 0,
        landed: 0,
    };
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Next),
        "it waits for the queued flush's cut"
    );
    b.place(&mut project, 30);
    // The held sync fails: the project rolls over with `b` still waiting.
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.writer.as_ref().unwrap().pause_syncs(false);
    let old = project.recovery_path().unwrap().to_path_buf();
    project.finish_flushes().unwrap();
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    let points = project.store().working().save_points().unwrap();
    let cut = *points.first().unwrap();
    assert_eq!(points.len(), 2, "the fill and its batch's: {points:?}");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::At(cut))),
        "nothing landed: the fill's save point is the surface it found"
    );
    assert!(project.may_land(), "it may land at once");
    let carried = batch_at(op, PenDown::At(cut), 0..30, 0).with_dab_format(FORMAT);
    assert_eq!(batches(&mut project), vec![carried], "its dabs");
    let path = project.recovery_path().unwrap().to_path_buf();
    crash_now(project);
    {
        let file = ProjectFile::open_at(&path, cut).unwrap();
        let mesh = file.mesh();
        let mut store = TileStore::open(file, roomy()).unwrap();
        let found = dense(&mesh, &map);
        assert_positions(
            &mut store,
            &found,
            &tiles(&mesh, 1, TOP),
            "the tiles b found",
        );
    }
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!((open.op(), open.landed()), (op, 0), "nothing landed");
    assert_eq!(open.dabs(), dabs(0..30), "every dab placed");
    assert_eq!(open.pen_down_keys().count(), 0, "no tile changed");
    let mut resumed = going_on(b, &open);
    restored.resume_stroke(open).unwrap();
    resumed.land(&mut restored, &mut map, 30);
    restored.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut restored, &map, "every dab landed");
    restored.close(false).unwrap();
}

#[test]
fn a_roll_over_inside_begin_stroke() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 200, 88);
    flush(&mut project);
    let mesh = mesh();
    apply(project.store(), &mut map, &random_edits(&mesh, TOP, 20, 89));
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_mut().unwrap().stop_thread();
    let mut landing = begin_on_top(&mut project, 30, 90);
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    let points = project.store().working().save_points().unwrap();
    assert_eq!(points.len(), 1, "no stroke was open: the fill's alone");
    let pen_down = PenDown::At(points[0]);
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(pen_down)),
        "the new writer names the fresh file's save point"
    );
    let live = live_shapes(&mut project, &landing);
    landing.place(&mut project, 30);
    landing.land(&mut project, &mut map, 7);
    flush(&mut project);
    crash_now(project);
    assert_resumes_as_live(restore(&dir), landing, &mut map, &live, pen_down, 7);
}

#[test]
fn a_roll_over_of_a_stroke_whose_tiles_have_no_shapes_leaves_its_pen_down_unknown() {
    // The store's own strokes edit without keeping pen-down shapes: nothing in memory holds
    // the surface they found, so the fresh file names none.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut landing = begin_on_top(&mut project, 30, 91);
    landing.place(&mut project, 30);
    landing.land(&mut project, &mut map, 5);
    let points = fail_and_roll_over(&mut project);
    assert_eq!(points.len(), 1, "no pen-down cut: {points:?}");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::Unknown)),
        "unknown"
    );
    project.close(false).unwrap();
}
