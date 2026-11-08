//! A flush asked for now ([`Project::request_flush`]), and knowing when everything is durable
//! ([`Project::flushed`]): what a close that finishes its landing with no window waits on.

use super::*;

/// The project's recovery writer.
fn writer(project: &Project) -> &RecoveryWriter {
    project.writer.as_ref().unwrap()
}

/// Polls `project` until [`Project::flushed`], as the app does between checks; returns how many
/// polls it took.
fn poll_until_flushed(project: &mut Project) -> u32 {
    let began = Instant::now();
    let mut polls = 0;
    while !project.flushed() {
        project.poll().unwrap();
        polls += 1;
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the flush becomes durable"
        );
        std::thread::yield_now();
    }
    polls
}

/// `count` random edits with `seed`, applied to the store and to `map` as dabs of the current op.
fn edit(project: &mut Project, map: &mut CorrectionMap, count: u32, seed: u64) {
    let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
    apply_dabs(project.store(), map, &edits);
}

#[test]
fn a_requested_flush_starts_at_once_and_the_timed_ones_keep_their_cadence() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    project.begin_stroke_and_wait().unwrap();
    let start = Instant::now();
    edit(&mut project, &mut map, 20, 110);
    project.tick(start).unwrap();
    crash::poll_until_idle(&mut project);
    assert_eq!(writer(&project).flushes(), 1, "the first tick flushes");

    edit(&mut project, &mut map, 20, 111);
    let half = start + FLUSH_INTERVAL / 2;
    project.tick(half).unwrap();
    assert!(
        writer(&project).is_idle(),
        "a tick before the interval doesn't"
    );
    project.request_flush(half).unwrap();
    assert!(!writer(&project).is_idle(), "a request starts one at once");
    crash::poll_until_idle(&mut project);
    assert_eq!(writer(&project).flushes(), 2, "flushed");

    // The interval runs from the requested flush's start, as from any flush's.
    edit(&mut project, &mut map, 20, 112);
    project.tick(half + FLUSH_INTERVAL / 2).unwrap();
    assert!(writer(&project).is_idle(), "not a second after it started");
    project.tick(half + FLUSH_INTERVAL).unwrap();
    assert!(!writer(&project).is_idle(), "a second after it started");
    crash::poll_until_idle(&mut project);
    project.end_unrecorded_stroke().unwrap();
    project.close(false).unwrap();
}

#[test]
fn flushed_turns_true_once_the_save_point_holding_the_rows_is_durable() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 113);
    assert!(project.flushed(), "nothing to flush");
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 4);
    assert!(!project.flushed(), "edits and rows not in the file");
    writer(&project).pause_syncs(true);
    project.request_flush(Instant::now()).unwrap();
    for _ in 0..20 {
        project.poll().unwrap();
        assert!(!project.flushed(), "its syncs are held: not durable");
    }
    writer(&project).pause_syncs(false);
    let polls = poll_until_flushed(&mut project);
    assert!(polls > 0, "it took polls");
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 4..10, 4)],
        "the rows placed before the request"
    );
    assert!(
        !project.store().working().has_unsaved_records(),
        "a save point holds them"
    );
    project.close(false).unwrap();
}

#[test]
fn a_request_while_a_flush_is_in_flight_queues_one_and_flushed_waits_for_it() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 114);
    stroke.place(&mut project, 6);
    stroke.land(&mut project, &mut map, 2);
    writer(&project).pause_syncs(true);
    project.request_flush(Instant::now()).unwrap();
    project
        .writer
        .as_mut()
        .unwrap()
        .step_blocking(&mut project.store)
        .unwrap();
    assert_eq!(writer(&project).phase(), "syncing tiles", "in flight");
    stroke.place(&mut project, 4);
    stroke.land(&mut project, &mut map, 3);
    project.request_flush(Instant::now()).unwrap();
    assert!(writer(&project).is_queued(), "queued behind it");
    assert!(!project.flushed(), "in flight and queued");
    writer(&project).pause_syncs(false);
    poll_until_flushed(&mut project);
    assert_eq!(writer(&project).flushes(), 2, "both");
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 2..6, 2), batch(stroke.op, 6..10, 5)],
        "the queued one holds what came after the first's cut"
    );
    project.close(false).unwrap();
}

#[test]
fn flushed_is_false_while_the_writer_has_failed() {
    // Failed: a sync failed, and nothing has rolled over yet.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    project.begin_stroke_and_wait().unwrap();
    edit(&mut project, &mut map, 20, 115);
    writer(&project).fail_syncs(0, 1);
    project.request_flush(Instant::now()).unwrap();
    let began = Instant::now();
    while !writer(&project).is_failed() {
        let stepped = project
            .writer
            .as_mut()
            .unwrap()
            .step_blocking(&mut project.store);
        assert!(
            stepped.is_ok() || writer(&project).is_failed(),
            "{stepped:?}"
        );
        assert!(began.elapsed() < Duration::from_secs(30), "the sync fails");
    }
    assert!(!project.flushed(), "failed");
    // The next call rolls over: the fresh file's save point holds every edit.
    project.poll().unwrap();
    assert!(project.flushed(), "rolled over, durable");
    project.end_unrecorded_stroke().unwrap();
    project.close(false).unwrap();
}

#[test]
fn flushed_is_false_while_protection_is_paused() {
    // The roll-over after a failed sync failed too.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    fail_and_pause(&mut project, &mut map, 116, io::ErrorKind::Other);
    assert!(!project.flushed(), "protection paused");
    // Before the next try is due, a request finds no writer to ask.
    project.request_flush(Instant::now()).unwrap();
    assert!(!project.flushed(), "still paused");
}
