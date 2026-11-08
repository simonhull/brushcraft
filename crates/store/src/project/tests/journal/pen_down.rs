//! A stroke's first landing waits for its cut, and its journal names its pen-down save point: the
//! one holding every tile as the stroke found them, whatever flush is in flight as it begins.

use super::*;
use crate::tiles::PenDownPoint;

/// How many snapshots the writer has taken so far.
fn snapshots(project: &Project) -> usize {
    project.writer.as_ref().unwrap().copied().len()
}

/// The pen-down every batch of `op` names, asserting they all name the same.
fn pen_down_of(project: &mut Project, op: OpId) -> PenDown {
    let mut named: Vec<PenDown> = batches(project)
        .iter()
        .filter(|batch| batch.op() == op)
        .map(DabBatch::pen_down)
        .collect();
    named.dedup();
    assert_eq!(named.len(), 1, "every batch of {op:?} names one: {named:?}");
    named.into_iter().next().unwrap()
}

/// Starts a flush with the writer thread held back from its syncs, and steps it past its
/// encoding: its records appended, it waits in its tile sync, in flight whatever the owner polls.
fn hold_in_tile_sync(project: &mut Project) {
    project.writer.as_ref().unwrap().pause_syncs(true);
    project.tick(Instant::now() + FLUSH_INTERVAL * 100).unwrap();
    let writer = project.writer.as_mut().unwrap();
    assert_eq!(writer.phase(), "encoding", "a flush started");
    writer.step_blocking(&mut project.store).unwrap();
    assert_eq!(
        project.writer.as_ref().unwrap().phase(),
        "syncing tiles",
        "its records appended"
    );
}

/// Lets a writer held back from its syncs go, and waits until every flush is durable.
fn release_syncs(project: &mut Project) {
    project.writer.as_ref().unwrap().pause_syncs(false);
    crash::poll_until_idle(project);
}

/// Asserts that the recovery file at `path`, opened at its save point `offset`, evaluates as
/// the dense evaluation of `map`.
fn assert_save_point(path: &Path, offset: u64, map: &CorrectionMap, what: &str) {
    let file = ProjectFile::open_at(path, offset).unwrap();
    let mesh = file.mesh();
    let mut store = TileStore::open(file, roomy()).unwrap();
    assert_positions(&mut store, &dense(&mesh, map), &tiles(&mesh, 1, TOP), what);
}

/// Stroke `a` is landing with a flush in its tile sync; the rest of it lands after that cut, and
/// it ends, its end's flush queued. Returns the map as `a` left it.
fn land_behind_a_flush_in_flight(
    project: &mut Project,
    map: &mut CorrectionMap,
    a: &mut Stroke,
) -> CorrectionMap {
    a.place(project, 30);
    a.land(project, map, 20);
    hold_in_tile_sync(project);
    a.land(project, map, 10);
    project.end_unrecorded_stroke().unwrap();
    assert!(
        project.writer.as_ref().unwrap().is_queued(),
        "the stroke end's flush waits"
    );
    map.clone()
}

#[test]
fn back_to_back_strokes_leave_a_save_point_at_each_pen_down() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 40, 31);
    let found = land_behind_a_flush_in_flight(&mut project, &mut map, &mut a);
    let taken = snapshots(&project);
    let mut b = Stroke::begin(&mut project, 40, 32);
    assert_eq!(snapshots(&project), taken, "nothing copied as it begins");
    assert!(!project.may_land(), "it lands nothing before its cut");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Next),
        "the flush queued holds it"
    );
    b.place(&mut project, 20);
    release_syncs(&mut project);
    assert!(project.may_land(), "the queued flush has cut");
    b.land(&mut project, &mut map, 15);
    flush(&mut project);

    assert_eq!(pen_down_of(&mut project, a.op), PenDown::Base, "the first");
    let PenDown::At(offset) = pen_down_of(&mut project, b.op) else {
        panic!("the second stroke's pen-down is in a save point");
    };
    let points = project.store().working().save_points().unwrap();
    assert_eq!(
        points.get(1),
        Some(&offset),
        "the second save point, after the one in flight: {points:?}"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_save_point(
        &recovery,
        offset,
        &found,
        "every tile the first stroke wrote, at its last edit, none of the second's",
    );
}

#[test]
fn a_waiting_stroke_may_land_within_the_flush_s_syncs_under_ticking() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 40, 50);
    land_behind_a_flush_in_flight(&mut project, &mut map, &mut a);
    let mut b = Stroke::begin(&mut project, 40, 51);
    b.place(&mut project, 10);
    assert!(!project.may_land(), "waiting");
    project.writer.as_ref().unwrap().pause_syncs(false);
    let began = Instant::now();
    // As the frame loop does: a tick and a poll a frame.
    while !project.may_land() {
        project.tick(Instant::now()).unwrap();
        project.poll().unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "it may land once the flush in flight is durable"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !project.writer.as_ref().unwrap().is_idle(),
        "its pen-down flush is in flight, its cut taken"
    );
    b.land(&mut project, &mut map, 5);
    flush(&mut project);
    let points = project.store().working().save_points().unwrap();
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::At(points[1]),
        "{points:?}"
    );
    project.close(false).unwrap();
}

#[test]
fn wait_to_land_drives_the_flushes_until_the_cut_is_taken() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 40, 52);
    land_behind_a_flush_in_flight(&mut project, &mut map, &mut a);
    let mut b = Stroke::begin(&mut project, 40, 53);
    assert!(!project.may_land(), "waiting");
    project.writer.as_ref().unwrap().pause_syncs(false);
    project.wait_to_land().unwrap();
    assert!(project.may_land(), "the cut is taken");
    assert!(
        matches!(
            project.store().open_pen_down(),
            Some(PenDownPoint::Flush(_))
        ),
        "by the flush that was queued"
    );
    b.place(&mut project, 3);
    b.land(&mut project, &mut map, 3);
    project.wait_to_land().unwrap();
    project.close(false).unwrap();
}

#[test]
fn an_op_beginning_with_nothing_unsnapshotted_takes_no_copy() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();

    // Nothing to flush and no flush in flight: nothing is copied, nothing starts.
    let quiet = Stroke::begin(&mut project, 10, 33);
    assert_eq!(snapshots(&project), 0, "no snapshot");
    assert!(project.writer.as_ref().unwrap().is_idle(), "no flush");
    assert!(project.may_land(), "it lands at once");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::Base)),
        "the file's base holds it"
    );
    project.end_unrecorded_stroke().unwrap();
    drop(quiet);

    // A flush in flight, its cut after the last edit: it holds the next stroke's pen-down, and
    // the op copies nothing and waits for nothing.
    let mut a = Stroke::begin(&mut project, 20, 34);
    a.place(&mut project, 10);
    a.land(&mut project, &mut map, 10);
    project.writer.as_ref().unwrap().pause(true);
    project.end_unrecorded_stroke().unwrap();
    assert!(
        !project.writer.as_ref().unwrap().is_idle(),
        "the stroke end's flush in flight"
    );
    let taken = snapshots(&project);
    let mut b = Stroke::begin(&mut project, 20, 35);
    assert_eq!(snapshots(&project), taken, "no copy taken");
    assert!(
        !project.writer.as_ref().unwrap().is_queued(),
        "no flush queued"
    );
    assert!(project.may_land(), "it lands at once");
    assert!(
        matches!(
            project.store().open_pen_down(),
            Some(PenDownPoint::Flush(_))
        ),
        "the flush in flight holds it"
    );
    b.place(&mut project, 4);
    let_go(&mut project);
    flush(&mut project);
    let points = project.store().working().save_points().unwrap();
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::At(points[0]),
        "the stroke end's save point: {points:?}"
    );
    project.close(false).unwrap();
}

#[test]
fn a_flush_that_left_a_valid_tile_out_holds_no_pen_down() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let edits = random_edits(&project.store().working().mesh(), TOP, 30, 54);
    apply(project.store(), &mut map, &edits);
    let found = map.clone();
    // Its first copy fails to encode, as a compression error would: no batch rides with it, so
    // it goes on to its tile sync, its save point naming that tile's older state.
    project.writer.as_ref().unwrap().fail_encodes(1);
    hold_in_tile_sync(&mut project);
    let mut b = Stroke::begin(&mut project, 20, 55);
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Next),
        "not the flush in flight, though nothing was edited since its cut"
    );
    assert!(!project.may_land(), "waiting");
    b.place(&mut project, 6);
    project.writer.as_ref().unwrap().pause_syncs(false);
    let began = Instant::now();
    // The tile left out is reported as late on the way: no error of the test's.
    while !project.writer.as_ref().unwrap().is_idle() {
        drop(project.poll());
        assert!(began.elapsed() < Duration::from_secs(30), "the flushes end");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(project.may_land(), "the next flush has cut");
    b.land(&mut project, &mut map, 4);
    flush(&mut project);
    let points = project.store().working().save_points().unwrap();
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::At(points[1]),
        "the next flush's save point: {points:?}"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_save_point(
        &recovery,
        points[1],
        &found,
        "every tile as the stroke found it",
    );
}

#[test]
fn a_stroke_on_a_quiet_file_names_the_latest_durable_save_point() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 38);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 20);
    project.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut project);
    let durable = project.writer.as_ref().unwrap().durable_save_point();
    let taken = snapshots(&project);
    let mut b = Stroke::begin(&mut project, 20, 39);
    assert_eq!(snapshots(&project), taken, "no copy");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::At(durable.unwrap()))),
        "the latest durable save point"
    );
    b.place(&mut project, 5);
    flush(&mut project);
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::At(durable.unwrap()),
        "and its batches name it"
    );
    project.close(false).unwrap();
}

#[test]
fn an_op_beginning_with_edits_waiting_starts_their_flush() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    // Edits with no stroke end and no tick: nothing flushed them yet.
    let edits = random_edits(&project.store().working().mesh(), TOP, 30, 40);
    apply(project.store(), &mut map, &edits);
    let found = map.clone();
    assert!(project.writer.as_ref().unwrap().is_idle(), "no flush");
    let mut b = Stroke::begin(&mut project, 20, 41);
    assert!(
        !project.writer.as_ref().unwrap().is_idle(),
        "the op's flush started"
    );
    assert!(project.may_land(), "its cut is taken: it lands at once");
    b.place(&mut project, 10);
    b.land(&mut project, &mut map, 6);
    flush(&mut project);
    let points = project.store().working().save_points().unwrap();
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::At(points[0]),
        "its save point: {points:?}"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_save_point(&recovery, points[0], &found, "the edits before the stroke");
}

#[test]
fn a_stroke_on_a_file_with_no_save_point_names_the_base() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert_eq!(
        project.store().working().last_save_point(),
        None,
        "a fresh recovery file"
    );
    let mut stroke = Stroke::begin(&mut project, 20, 42);
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::Base)),
        "the file's base"
    );
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 3);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch_at(stroke.op, PenDown::Base, 3..10, 3)],
        "named in its batch"
    );
    project.close(false).unwrap();
}

#[test]
fn a_stroke_whose_pen_down_flush_is_abandoned_names_none() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();

    // The flush in flight holds it, and fails at its save point.
    let mut a = Stroke::begin(&mut project, 20, 43);
    a.place(&mut project, 10);
    a.land(&mut project, &mut map, 10);
    project.writer.as_ref().unwrap().pause(true);
    project.end_unrecorded_stroke().unwrap();
    project.store().fail_next_save_point();
    let mut b = Stroke::begin(&mut project, 20, 44);
    b.place(&mut project, 5);
    let failed = let_go_to_fail(&mut project);
    assert!(
        matches!(failed, ProjectError::Recovery(RecoveryError::Store(_))),
        "{failed:?}"
    );
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Known(PenDown::Unknown)),
        "no save point holds it"
    );
    flush(&mut project);
    assert_eq!(
        pen_down_of(&mut project, b.op),
        PenDown::Unknown,
        "named so"
    );
    b.land(&mut project, &mut map, 5);
    project.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut project);

    // The flush queued behind a failing one holds it: both are abandoned, and the stroke lands.
    let mut c = Stroke::begin(&mut project, 20, 45);
    c.place(&mut project, 10);
    c.land(&mut project, &mut map, 4);
    start_held_flush(&mut project);
    c.land(&mut project, &mut map, 6);
    project.end_unrecorded_stroke().unwrap();
    project.store().fail_next_save_point();
    let mut d = Stroke::begin(&mut project, 20, 46);
    assert!(!project.may_land(), "waiting for the queued flush");
    d.place(&mut project, 5);
    let failed_again = let_go_to_fail(&mut project);
    assert!(
        matches!(
            failed_again,
            ProjectError::Recovery(RecoveryError::Store(_))
        ),
        "{failed_again:?}"
    );
    assert!(project.writer.as_ref().unwrap().is_idle(), "both dropped");
    assert!(project.may_land(), "it lands, its pen-down unknown");
    d.land(&mut project, &mut map, 2);
    flush(&mut project);
    assert_eq!(
        pen_down_of(&mut project, d.op),
        PenDown::Unknown,
        "named so"
    );
    project.close(false).unwrap();
}

#[test]
fn a_flush_failing_while_the_next_stroke_waits_leaves_no_gap_or_overlap() {
    for fail in ["save point", "batch encode"] {
        let dir = TestDir::new();
        let mut map = CorrectionMap::new();
        let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
        let mut a = Stroke::begin(&mut project, 20, 47);
        a.place(&mut project, 10);
        match fail {
            "save point" => project.store().fail_next_save_point(),
            _ => project.writer.as_ref().unwrap().fail_batch_encodes(1),
        }
        start_held_flush(&mut project);
        // Placed and landed after the cut in flight.
        a.place(&mut project, 5);
        a.land(&mut project, &mut map, 15);
        project.end_unrecorded_stroke().unwrap();
        let mut b = Stroke::begin(&mut project, 20, 48);
        assert!(!project.may_land(), "{fail}: waiting");
        b.place(&mut project, 6);
        let failed = let_go_to_fail(&mut project);
        assert!(
            matches!(failed, ProjectError::Recovery(_)),
            "{fail}: {failed:?}"
        );
        assert!(project.may_land(), "{fail}: abandoned, it lands");
        flush(&mut project);
        let written = match fail {
            // Its batch was appended before the save point failed: the rest follows, once.
            "save point" => vec![
                batch(a.op, 0..10, 0),
                batch(a.op, 15..15, 15),
                batch_at(b.op, PenDown::Unknown, 0..6, 0),
            ],
            // Nothing of the stroke reached the file, and every dab of it has landed: its
            // tiles hold them all.
            _ => vec![batch_at(b.op, PenDown::Unknown, 0..6, 0)],
        };
        assert_eq!(batches(&mut project), written, "{fail}: no gap, no overlap");
        flush(&mut project);
        assert_eq!(batches(&mut project), written, "{fail}: and once");
        assert!(
            !project.store().working().has_unsaved_records(),
            "{fail}: a save point holds them"
        );
        project.close(false).unwrap();
    }
}
