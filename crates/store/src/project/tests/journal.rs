//! The dab journal in every flush: each flush writes the open stroke's dabs placed since the last
//! and the count landed at its cut, into the recovery file only, and never a batch twice.

use std::ops::Range;

use super::*;
use crate::FLUSH_INTERVAL;
use crate::fixture::dab_at;
use crate::journal::{DAB_BYTES, DabBatch, JournalError, MAX_JOURNAL_DABS};

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
fn a_save_mid_stroke_journals_the_dabs_not_landed_into_the_fresh_recovery_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut stroke = Stroke::begin(&mut project, 20, 9);
    stroke.place(&mut project, 10);
    flush(&mut project);
    stroke.land(&mut project, &mut map, 4);
    assert_saved(project.save_as(&path));
    assert_eq!(batches(&mut project), Vec::new(), "a fresh recovery file");
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![batch(stroke.op, 4..10, 4)],
        "the dabs not landed, again"
    );
    project.close(false).unwrap();
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
            Err(ProjectError::Journal(JournalError::LandedPastPlaced {
                landed: 6,
                placed: 5
            }))
        ),
        "past the dabs placed"
    );
    assert!(
        matches!(
            project.journal_landed(2),
            Err(ProjectError::Journal(JournalError::LandedBack {
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
