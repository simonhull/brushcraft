//! The dab journal in every flush: each flush writes the open stroke's dabs placed since the last
//! and the count landed at its cut, into the recovery file only, and never a batch twice.

use std::ops::Range;

use super::*;
use crate::fixture::dab_at;
use crate::journal::{DAB_BYTES, DabBatch, MAX_JOURNAL_DABS};
use crate::{FLUSH_INTERVAL, JournalCountError};

/// The tests' dabs `range`, as [`dab_at`] makes them.
fn dabs(range: Range<u32>) -> Vec<[u8; DAB_BYTES]> {
    range.map(dab_at).collect()
}

/// The batch of `op` the tests expect: the dabs `range`, with `landed` of the stroke's landed.
fn batch(op: OpId, range: Range<u32>, landed: u32) -> DabBatch {
    let first = range.start;
    DabBatch::new(op, first, landed, dabs(range)).unwrap()
}

/// Every batch of the project's recovery file, its working file, read back.
fn batches(project: &mut Project) -> Vec<DabBatch> {
    project
        .store()
        .working()
        .batches()
        .map(Result::unwrap)
        .collect()
}

/// A stroke being journalled and landed: its op, the edits that stand for its dabs' work (one
/// edit a dab), and how far it has got.
struct Stroke {
    /// The stroke's op.
    op: OpId,
    /// The edit of each dab, by index.
    edits: Vec<crate::fixture::Edit>,
    /// The dabs placed so far.
    placed: u32,
    /// The dabs landed so far.
    landed: u32,
}

impl Stroke {
    /// Begins a stroke on `project` with room for `count` dabs, their edits from `seed`.
    fn begin(project: &mut Project, count: u32, seed: u64) -> Self {
        let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
        let op = project.begin_stroke().unwrap();
        Self {
            op,
            edits,
            placed: 0,
            landed: 0,
        }
    }

    /// Places the next `count` dabs: journals them.
    fn place(&mut self, project: &mut Project, count: u32) {
        let next = self.placed + count;
        project.journal(&dabs(self.placed..next)).unwrap();
        self.placed = next;
    }

    /// Lands the next `count` dabs: makes their edits in the store and in `map`, then says so.
    fn land(&mut self, project: &mut Project, map: &mut CorrectionMap, count: u32) {
        let next = self.landed + count;
        let edits = &self.edits[self.landed as usize..next as usize];
        apply_dabs(project.store(), map, edits);
        project.journal_landed(next).unwrap();
        self.landed = next;
    }
}

/// Flushes every edit and journal batch due, and waits until it is durable.
fn flush(project: &mut Project) {
    project.finish_flushes().unwrap();
}

#[test]
fn a_flush_writes_the_dabs_placed_since_the_last() {
    let dir = TestDir::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 1);
    stroke.place(&mut project, 10);
    // A tick flushes the journal alone: no tile was edited.
    project.tick(Instant::now() + FLUSH_INTERVAL).unwrap();
    crash::poll_until_idle(&mut project);
    let op = stroke.op;
    assert_eq!(
        batches(&mut project),
        vec![batch(op, 0..10, 0)],
        "the first ten"
    );
    stroke.place(&mut project, 5);
    flush(&mut project);
    let both = vec![batch(op, 0..10, 0), batch(op, 10..15, 0)];
    assert_eq!(batches(&mut project), both, "the five since");
    flush(&mut project);
    project.tick(Instant::now() + FLUSH_INTERVAL * 10).unwrap();
    crash::poll_until_idle(&mut project);
    assert_eq!(batches(&mut project), both, "nothing new, no batch");
    assert!(
        !project.store().working().has_unsaved_records(),
        "a save point holds them"
    );
    project.close(false).unwrap();
}

#[test]
fn a_batch_names_the_landed_count_at_its_cut() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 2);
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 4);
    flush(&mut project);
    let op = stroke.op;
    let first = batch(op, 4..10, 4);
    assert_eq!(
        batches(&mut project),
        vec![first.clone()],
        "the dabs not landed, and the count"
    );
    stroke.land(&mut project, &mut map, 3);
    stroke.place(&mut project, 5);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![first, batch(op, 10..15, 7)],
        "the dabs since, and the count at the second cut"
    );
    project.close(false).unwrap();
}

#[test]
fn landed_dabs_are_never_written() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 3);
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 10);
    flush(&mut project);
    assert_eq!(batches(&mut project), Vec::new(), "every dab landed");
    stroke.place(&mut project, 2);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 10..12, 10)],
        "from the landed count on"
    );
    project.close(false).unwrap();
}

#[test]
fn only_the_landed_count_moving_writes_an_empty_batch() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 4);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 3);
    flush(&mut project);
    let op = stroke.op;
    let written = vec![batch(op, 0..10, 0), batch(op, 10..10, 3)];
    assert_eq!(batches(&mut project), written, "the count alone");
    flush(&mut project);
    assert_eq!(batches(&mut project), written, "and once");
    project.close(false).unwrap();
}

#[test]
fn a_stroke_that_has_landed_writes_no_more() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 5);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 10);
    flush(&mut project);
    let op = stroke.op;
    let written = vec![batch(op, 0..10, 0), batch(op, 10..10, 10)];
    assert_eq!(batches(&mut project), written, "the last count");
    flush(&mut project);
    project.end_unrecorded_stroke().unwrap();
    project.poll().unwrap();
    crash::poll_until_idle(&mut project);
    assert_eq!(batches(&mut project), written, "nothing more");

    // A stroke that ends right after its last dabs land writes its last count with the flush
    // its end makes: no batch in the file is behind its tiles.
    let mut quick = Stroke::begin(&mut project, 20, 6);
    quick.place(&mut project, 8);
    flush(&mut project);
    quick.land(&mut project, &mut map, 8);
    project.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut project);
    let mut ended = written;
    ended.extend([batch(quick.op, 0..8, 0), batch(quick.op, 8..8, 8)]);
    assert_eq!(batches(&mut project), ended, "its end's flush");
    flush(&mut project);
    assert_eq!(batches(&mut project), ended, "and no more");
    project.close(false).unwrap();
}

#[test]
fn a_flush_of_more_dabs_than_a_batch_holds_writes_several() {
    let dir = TestDir::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 1, 7);
    let max = u32::try_from(MAX_JOURNAL_DABS).unwrap();
    stroke.place(&mut project, max + 10);
    flush(&mut project);
    let op = stroke.op;
    assert_eq!(
        batches(&mut project),
        vec![batch(op, 0..max, 0), batch(op, max..max + 10, 0)],
        "split at the limit"
    );
    project.close(false).unwrap();
}

#[test]
fn a_save_writes_no_batch_into_the_project_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 8);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    assert_saved(project.save_as(&path));
    assert_eq!(
        project.file.as_ref().unwrap().batches().count(),
        0,
        "no batch in the project file"
    );
    assert_eq!(batches(&mut project), Vec::new(), "a fresh recovery file");
    project.close(false).unwrap();
    assert_eq!(
        ProjectFile::open(&path).unwrap().batches().count(),
        0,
        "nor once reopened"
    );
    assert_project_file(&path, &map, "the landed edits saved");
    let compacted = dir.path("c.bcrf");
    crate::compact(&path, &compacted).unwrap();
    assert_eq!(
        ProjectFile::open(&compacted).unwrap().batches().count(),
        0,
        "compaction copies none"
    );
    assert_project_file(&compacted, &map, "the edits kept");
}

#[test]
fn a_save_waits_for_the_dabs_to_land() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 9);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 4);
    assert!(
        matches!(project.save_as(&path), Err(ProjectError::StillLanding)),
        "a save as, refused"
    );
    assert!(!path.exists(), "nothing written");
    stroke.land(&mut project, &mut map, 6);
    assert_saved(project.save_as(&path));
    assert_eq!(batches(&mut project), Vec::new(), "a fresh recovery file");
    stroke.place(&mut project, 2);
    assert!(
        matches!(project.save(), Err(ProjectError::StillLanding)),
        "a save, refused"
    );
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 10..12, 10)],
        "the journal goes on in the fresh file"
    );
    stroke.land(&mut project, &mut map, 2);
    assert_saved(project.save());
    project.close(false).unwrap();
    assert_project_file(&path, &map, "every dab landed and saved");
}

#[test]
fn the_journal_goes_when_the_stroke_ends_or_is_discarded() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 10);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 10);
    let record = crate::stroke_record::tests::wobbly(stroke.op, 10, 10);
    project.end_stroke(&record).unwrap();
    crash::poll_until_idle(&mut project);
    let op = stroke.op;
    let written = vec![batch(op, 0..10, 0), batch(op, 10..10, 10)];
    assert_eq!(batches(&mut project), written, "its last count, then none");
    assert!(
        matches!(project.journal(&dabs(10..11)), Err(ProjectError::NoStroke)),
        "no stroke to journal for"
    );
    assert!(
        matches!(project.journal_landed(10), Err(ProjectError::NoStroke)),
        "nor to land"
    );
    flush(&mut project);
    assert_eq!(batches(&mut project), written, "nothing after the end");

    // The next stroke's journal is its own.
    let mut next = Stroke::begin(&mut project, 20, 11);
    next.place(&mut project, 3);
    flush(&mut project);
    let mut both = written;
    both.push(batch(next.op, 0..3, 0));
    assert_eq!(batches(&mut project), both, "a new journal");
    next.place(&mut project, 3);
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.discard().unwrap();
    assert!(!recovery.exists(), "discarded with its recovery file");
}

#[test]
fn a_journal_refuses_counts_a_stroke_cannot_have() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 12);
    stroke.place(&mut project, 5);
    stroke.land(&mut project, &mut map, 3);
    assert!(
        matches!(
            project.journal_landed(6),
            Err(ProjectError::Journal(JournalCountError::LandedPastPlaced {
                landed: 6,
                placed: 5
            }))
        ),
        "past the dabs placed"
    );
    assert!(
        matches!(
            project.journal_landed(2),
            Err(ProjectError::Journal(JournalCountError::LandedBack {
                landed: 2,
                before: 3
            }))
        ),
        "back"
    );
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 3..5, 3)],
        "the journal as it was"
    );
    project.close(false).unwrap();
}

#[test]
fn a_batch_that_fails_to_encode_is_written_by_the_next_flush() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 13);
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 2);
    let flushes = project.writer.as_ref().unwrap().flushes();
    project.writer.as_ref().unwrap().fail_batch_encodes(1);
    let failed = project.finish_flushes();
    assert!(
        matches!(
            failed,
            Err(ProjectError::Recovery(RecoveryError::Journal(_)))
        ),
        "{failed:?}"
    );
    assert_eq!(
        project.writer.as_ref().unwrap().flushes(),
        flushes,
        "no save point names tiles past the file's last batch"
    );
    assert_eq!(batches(&mut project), Vec::new(), "the batch left out");
    stroke.land(&mut project, &mut map, 3);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 5..10, 5)],
        "the dabs not landed, once, with the count at the next cut"
    );
    project.close(false).unwrap();
}

#[test]
fn a_stroke_ending_with_dabs_not_landed_counts_them_landed() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 14);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 4);
    project.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut project);
    let op = stroke.op;
    let written = vec![batch(op, 0..10, 0), batch(op, 10..10, 10)];
    assert_eq!(
        batches(&mut project),
        written,
        "the dropped dabs never land"
    );

    // With no batch in the file yet, none is written: the tiles hold every dab landed.
    let mut unwritten = Stroke::begin(&mut project, 20, 15);
    unwritten.place(&mut project, 10);
    unwritten.land(&mut project, &mut map, 4);
    project.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut project);
    flush(&mut project);
    assert_eq!(batches(&mut project), written, "no batch of it");
    project.close(false).unwrap();
}

#[test]
fn a_flush_whose_save_point_fails_leaves_its_batches_for_the_next() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 16);
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 2);
    project.store().fail_next_save_point();
    let failed = project.finish_flushes();
    assert!(
        matches!(failed, Err(ProjectError::Recovery(RecoveryError::Store(_)))),
        "{failed:?}"
    );
    let op = stroke.op;
    let first = batch(op, 2..10, 2);
    assert_eq!(
        batches(&mut project),
        vec![first.clone()],
        "appended, unsaved"
    );
    assert!(
        project.store().working().has_unsaved_records(),
        "no save point holds it yet"
    );
    stroke.land(&mut project, &mut map, 3);
    stroke.place(&mut project, 4);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![first, batch(op, 10..14, 5)],
        "the next goes on after it: no gap, no overlap"
    );
    assert!(
        !project.store().working().has_unsaved_records(),
        "a save point holds both"
    );
    project.close(false).unwrap();
}

/// Starts a flush with the writer thread held back, so it stays in flight, its cut taken.
fn start_held_flush(project: &mut Project) {
    project.writer.as_ref().unwrap().pause(true);
    project.tick(Instant::now() + FLUSH_INTERVAL * 100).unwrap();
    assert!(
        !project.writer.as_ref().unwrap().is_idle(),
        "a flush in flight"
    );
}

/// Lets a held writer thread go, and waits until every flush is durable.
fn let_go(project: &mut Project) {
    project.writer.as_ref().unwrap().pause(false);
    crash::poll_until_idle(project);
}

#[test]
fn a_stroke_ending_while_its_cut_is_in_flight_writes_its_last_count_once() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 17);
    stroke.place(&mut project, 10);
    start_held_flush(&mut project);
    stroke.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    assert!(
        project.writer.as_ref().unwrap().is_queued(),
        "its flush queued"
    );
    let_go(&mut project);
    let op = stroke.op;
    let written = vec![batch(op, 0..10, 0), batch(op, 10..10, 10)];
    assert_eq!(
        batches(&mut project),
        written,
        "the queued flush writes its last count"
    );
    flush(&mut project);
    assert_eq!(batches(&mut project), written, "once");
    project.close(false).unwrap();
}

#[test]
fn a_stroke_begun_while_the_last_one_waits_writes_after_it() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut old = Stroke::begin(&mut project, 20, 18);
    old.place(&mut project, 10);
    start_held_flush(&mut project);
    old.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    let mut new = Stroke::begin(&mut project, 20, 19);
    new.place(&mut project, 3);
    let_go(&mut project);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![
            batch(old.op, 0..10, 0),
            batch(old.op, 10..10, 10),
            batch(new.op, 0..3, 0)
        ],
        "the old stroke's last count first"
    );
    project.close(false).unwrap();
}

#[test]
fn after_a_roll_over_the_next_flush_journals_the_dabs_not_landed_into_the_fresh_file() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 20);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 4);
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.finish_flushes().unwrap();
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 4..10, 4)],
        "from the landed count, with no gap"
    );
    project.close(false).unwrap();
}

#[test]
fn a_tile_left_out_of_a_flush_with_a_batch_abandons_it() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 21);
    stroke.place(&mut project, 10);
    stroke.land(&mut project, &mut map, 2);
    let flushes = project.writer.as_ref().unwrap().flushes();
    project.writer.as_ref().unwrap().fail_encodes(1);
    let failed = project.finish_flushes();
    assert!(
        matches!(failed, Err(ProjectError::Recovery(RecoveryError::Unpaired))),
        "{failed:?}"
    );
    assert_eq!(
        project.writer.as_ref().unwrap().flushes(),
        flushes,
        "no save point pairs the tile's older state with the count"
    );
    assert_eq!(batches(&mut project), Vec::new(), "no batch");
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 2..10, 2)],
        "the next flush writes both"
    );
    assert!(
        !project.store().working().has_unsaved_records(),
        "a save point holds them"
    );
    project.close(false).unwrap();
}

/// Lets a held writer thread go, and polls until the flush in flight fails: its batch's encode,
/// made to fail. Returns the error.
fn let_go_to_fail(project: &mut Project) -> ProjectError {
    project.writer.as_ref().unwrap().pause(false);
    let began = Instant::now();
    loop {
        if let Err(e) = project.poll() {
            return e;
        }
        assert!(began.elapsed() < Duration::from_secs(30), "the flush fails");
        std::thread::yield_now();
    }
}

#[test]
fn an_ended_stroke_whose_cut_fails_writes_its_last_count_only_if_a_batch_is_in_the_file() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();

    // No batch of it in the file: nothing to write, and the journal goes.
    let mut unwritten = Stroke::begin(&mut project, 20, 22);
    unwritten.place(&mut project, 10);
    project.writer.as_ref().unwrap().fail_batch_encodes(1);
    start_held_flush(&mut project);
    unwritten.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    let failed = let_go_to_fail(&mut project);
    assert!(
        matches!(failed, ProjectError::Recovery(RecoveryError::Journal(_))),
        "{failed:?}"
    );
    assert!(
        project.writer.as_ref().unwrap().is_idle(),
        "the stroke end's flush dropped"
    );
    assert!(!project.store().has_journal_work(), "the journal gone");
    flush(&mut project);
    assert_eq!(batches(&mut project), Vec::new(), "no batch, then or later");

    // A batch of it in the file: its last count waits for the next flush.
    let mut written = Stroke::begin(&mut project, 20, 23);
    written.place(&mut project, 4);
    flush(&mut project);
    written.place(&mut project, 6);
    project.writer.as_ref().unwrap().fail_batch_encodes(1);
    start_held_flush(&mut project);
    written.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    let failed_again = let_go_to_fail(&mut project);
    assert!(
        matches!(
            failed_again,
            ProjectError::Recovery(RecoveryError::Journal(_))
        ),
        "{failed_again:?}"
    );
    assert!(
        project.writer.as_ref().unwrap().is_idle(),
        "the stroke end's flush dropped"
    );
    let op = written.op;
    assert_eq!(batches(&mut project), vec![batch(op, 0..4, 0)], "not yet");
    assert!(project.store().has_journal_work(), "the last count waits");
    project
        .tick(Instant::now() + FLUSH_INTERVAL * 1_000)
        .unwrap();
    crash::poll_until_idle(&mut project);
    let last = vec![batch(op, 0..4, 0), batch(op, 10..10, 10)];
    assert_eq!(batches(&mut project), last, "the next tick writes it");
    flush(&mut project);
    assert_eq!(batches(&mut project), last, "once");
    project.close(false).unwrap();
}
