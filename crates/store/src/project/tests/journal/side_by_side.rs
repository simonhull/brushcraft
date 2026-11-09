//! Strokes journal side by side.
//!
//! A stroke begun while the one before it still lands reserves its op and journals its dabs at
//! once, naming its pen-down as the stroke before's ([`PenDown::After`]) until its own cut is
//! durable, then that save point ([`PenDown::At`]).

use super::pen_down::{assert_save_point, hold_in_tile_sync, release_syncs};
use super::*;
use crate::tiles::{PenDownPoint, ReservedOp};

/// A stroke begun while the one before it lands: its op reserved, its dabs journalled and none
/// landed, until its op begins ([`Later::begin_op`]).
struct Later {
    /// The op reserved for it.
    reserved: ReservedOp,
    /// The edit of each dab, by index.
    edits: Vec<crate::fixture::Edit>,
    /// The dabs placed so far.
    placed: u32,
}

impl Later {
    /// Begins a stroke after the newest open one on `project`, with room for `count` dabs, their
    /// edits from `seed`.
    fn begin(project: &mut Project, count: u32, seed: u64) -> Self {
        let edits = random_edits(&project.store().working().mesh(), TOP, count, seed);
        let reserved = project.begin_stroke_after().unwrap();
        Self {
            reserved,
            edits,
            placed: 0,
        }
    }

    /// Its op.
    fn op(&self) -> OpId {
        self.reserved.op()
    }

    /// Places the next `count` dabs: journals them.
    fn place(&mut self, project: &mut Project, count: u32) {
        let next = self.placed + count;
        project
            .journal_placed_after(&self.reserved, &dabs(self.placed..next))
            .unwrap();
        self.placed = next;
    }

    /// Begins its op, the one before it ended: it is the open stroke from now on.
    fn begin_op(self, project: &mut Project) -> Stroke {
        let op = self.op();
        let begun = project.begin_reserved_stroke(self.reserved);
        assert_eq!(begun.map_err(|e| e.into_parts().1).unwrap(), op, "its op");
        Stroke {
            op,
            edits: self.edits,
            placed: self.placed,
            landed: 0,
        }
    }

    /// Ends the open stroke and begins this one's op; returns it.
    fn begin_op_after_end(self, project: &mut Project) -> OpId {
        project.end_unrecorded_stroke().unwrap();
        self.begin_op(project).op
    }
}

/// The pen-down each batch of `op` names, in file order.
fn pen_downs_of(project: &mut Project, op: OpId) -> Vec<PenDown> {
    let named = batches(project)
        .into_iter()
        .filter(|batch| batch.op() == op);
    named.map(|batch| batch.pen_down()).collect()
}

#[test]
fn two_open_strokes_journal_in_one_flush_each_its_own_batches() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 61);
    a.place(&mut project, 10);
    a.land(&mut project, &mut map, 4);
    let mut b = Later::begin(&mut project, 20, 62);
    assert!(b.op() > a.op, "ordered after the stroke before");
    b.place(&mut project, 6);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![
            batch(a.op, 4..10, 4),
            batch_at(b.op(), PenDown::After(a.op), 0..6, 0),
        ],
        "the writing stroke's with its landed count, the later one's after it, none landed"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    let point = project.store().working().last_save_point().unwrap();
    // The pairing holds: the save point's tiles are those of the writing stroke's landed count.
    let paired = map.clone();
    a.land(&mut project, &mut map, 6);
    project.end_unrecorded_stroke().unwrap();
    b.begin_op(&mut project);
    project.close(true).unwrap();
    assert_save_point(&recovery, point, &paired, "the first four dabs of a");
}

#[test]
fn a_later_stroke_names_after_until_its_cut_is_durable_then_at_never_back() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 63);
    a.place(&mut project, 10);
    a.land(&mut project, &mut map, 3);
    let mut b = Later::begin(&mut project, 20, 64);
    b.place(&mut project, 5);
    flush(&mut project);
    a.land(&mut project, &mut map, 7);
    b.place(&mut project, 2);
    flush(&mut project);
    project.end_unrecorded_stroke().unwrap();
    let found = map.clone();
    let mut b = b.begin_op(&mut project);
    b.place(&mut project, 3);
    flush(&mut project);
    assert!(project.may_land(), "its cut is durable");
    b.land(&mut project, &mut map, 4);
    b.place(&mut project, 2);
    flush(&mut project);
    let named = pen_downs_of(&mut project, b.op);
    let at: Vec<&PenDown> = named
        .iter()
        .skip_while(|pen_down| **pen_down == PenDown::After(a.op))
        .collect();
    assert!(
        named.len() > at.len() && !at.is_empty(),
        "after the stroke before, then a save point: {named:?}"
    );
    let Some(&&PenDown::At(offset)) = at.first() else {
        panic!("its own cut is a save point: {named:?}");
    };
    assert!(
        at.iter().all(|pen_down| **pen_down == PenDown::At(offset)),
        "never back: {named:?}"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_save_point(&recovery, offset, &found, "the surface a left");
}

#[test]
fn a_reserved_stroke_lands_nothing_and_its_batches_count_none_landed() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 65);
    a.place(&mut project, 10);
    let mut b = Later::begin(&mut project, 20, 66);
    b.place(&mut project, 4);
    a.land(&mut project, &mut map, 5);
    flush(&mut project);
    b.place(&mut project, 4);
    a.land(&mut project, &mut map, 5);
    flush(&mut project);
    let of_b: Vec<u32> = batches(&mut project)
        .iter()
        .filter(|batch| batch.op() == b.op())
        .map(DabBatch::landed)
        .collect();
    assert_eq!(of_b, vec![0, 0], "none of b's landed");
    let of_a: Vec<u32> = batches(&mut project)
        .iter()
        .filter(|batch| batch.op() == a.op)
        .map(DabBatch::landed)
        .collect();
    assert_eq!(of_a, vec![5, 10], "a's counts, as landed at each cut");
    let path = dir.path("saved.bcp");
    assert!(
        matches!(project.save_as(&path), Err(ProjectError::StillLanding)),
        "a save waits for the later stroke's dabs too"
    );
    project.end_unrecorded_stroke().unwrap();
    b.begin_op(&mut project);
    project.close(false).unwrap();
}

#[test]
fn the_later_stroke_s_first_landing_waits_for_its_cut_after_the_earlier_op_ends() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 40, 67);
    a.place(&mut project, 30);
    a.land(&mut project, &mut map, 20);
    let mut b = Later::begin(&mut project, 20, 68);
    b.place(&mut project, 5);
    hold_in_tile_sync(&mut project);
    a.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    let found = map.clone();
    let mut b = b.begin_op(&mut project);
    assert!(!project.may_land(), "it lands nothing before its cut");
    assert_eq!(
        project.store().open_pen_down(),
        Some(PenDownPoint::Next),
        "the flush queued holds it"
    );
    b.place(&mut project, 5);
    release_syncs(&mut project);
    assert!(project.may_land(), "the queued flush has cut");
    b.land(&mut project, &mut map, 6);
    flush(&mut project);
    let points = project.store().working().save_points().unwrap();
    let named = pen_downs_of(&mut project, b.op);
    assert_eq!(
        named.last(),
        Some(&PenDown::At(points[1])),
        "the queued flush's save point, after the one in flight: {named:?} {points:?}"
    );
    let recovery = project.recovery_path().unwrap().to_path_buf();
    project.close(true).unwrap();
    assert_save_point(&recovery, points[1], &found, "the surface a left");
}

#[test]
fn ending_a_stroke_with_a_later_one_reserved_ends_its_op() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 69);
    a.place(&mut project, 4);
    let b = Later::begin(&mut project, 20, 70);
    let refused = project
        .begin_reserved_stroke(b.reserved)
        .expect_err("the stroke before is open");
    let (reserved, error) = refused.into_parts();
    assert!(
        matches!(error, CacheError::OpBegun(op) if op == a.op),
        "{error:?}"
    );
    let b = Later { reserved, ..b };
    a.land(&mut project, &mut map, 4);
    let record = crate::stroke_record::tests::wobbly(a.op, 4, 69);
    project.end_stroke(&record).unwrap();
    assert_eq!(project.store().current_op(), None, "its op ended");
    let mut b = b.begin_op(&mut project);
    assert_eq!(project.store().current_op(), Some(b.op), "the later one's");
    let c = Later::begin(&mut project, 20, 71);
    b.place(&mut project, 2);
    project.end_unrecorded_stroke().unwrap();
    assert_eq!(
        project.store().current_op(),
        None,
        "the unrecorded one's too"
    );
    let c = c.begin_op(&mut project);
    project.end_unrecorded_stroke().unwrap();
    assert_eq!(
        project.store().current_op(),
        Some(c.op),
        "with none reserved, as before"
    );
    project.close(false).unwrap();
}

#[test]
fn dropping_a_reserved_stroke_closes_its_journal_and_those_after_it() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 72);
    a.place(&mut project, 10);
    let mut b = Later::begin(&mut project, 20, 73);
    b.place(&mut project, 4);
    let mut c = Later::begin(&mut project, 20, 74);
    c.place(&mut project, 3);
    let d = Later::begin(&mut project, 20, 75);
    flush(&mut project);
    let (b_op, c_op, d_op) = (b.op(), c.op(), d.op());
    b.place(&mut project, 2);
    assert_eq!(
        project.drop_stroke_after(b.reserved),
        vec![b_op, c_op, d_op],
        "b, and every stroke after it"
    );
    assert_eq!(project.drop_stroke_after(c.reserved), Vec::new(), "gone");
    assert_eq!(project.drop_stroke_after(d.reserved), Vec::new(), "gone");
    a.land(&mut project, &mut map, 10);
    flush(&mut project);
    assert_eq!(
        batches(&mut project),
        vec![
            batch(a.op, 0..10, 0),
            batch_at(b_op, PenDown::After(a.op), 0..4, 0),
            batch_at(c_op, PenDown::After(b_op), 0..3, 0),
            batch_at(b_op, PenDown::After(a.op), 6..6, 6),
            batch_at(c_op, PenDown::After(b_op), 3..3, 3),
            batch(a.op, 10..10, 10),
        ],
        "the dropped strokes close, their dabs counted landed; d wrote nothing, and closes none"
    );
    let mut reserved = Later::begin(&mut project, 20, 76);
    let last = Later::begin(&mut project, 2, 77);
    let dropped = project.drop_stroke_after(last.reserved);
    assert_eq!(dropped.len(), 1, "only the last");
    reserved.place(&mut project, 1);
    let begun = reserved.begin_op_after_end(&mut project);
    assert_eq!(project.store().current_op(), Some(begun), "it begins");
    project.close(false).unwrap();
}

#[test]
fn a_stroke_begins_after_another_only_while_one_is_open() {
    let dir = TestDir::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    assert!(
        matches!(project.begin_stroke_after(), Err(ProjectError::NoStroke)),
        "no stroke to begin after"
    );
    let a = Stroke::begin(&mut project, 4, 78);
    let b = Later::begin(&mut project, 4, 79);
    assert!(
        matches!(project.begin_stroke(), Err(ProjectError::Cache(CacheError::OpReserved(op))) if op == b.op()),
        "a stroke of its own begins after the one reserved"
    );
    assert_eq!(
        project.store().current_op(),
        Some(a.op),
        "and the open one goes on"
    );
    let placed = project.journal_placed(&dabs(0..2));
    assert!(placed.is_ok(), "the open stroke still journals: {placed:?}");
    let begun = b.begin_op_after_end(&mut project);
    assert!(begun > a.op, "begun");
    project.close(false).unwrap();
}

#[test]
fn a_roll_over_writes_a_reserved_stroke_s_dabs_again_after_the_stroke_before() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    let mut a = Stroke::begin(&mut project, 20, 80);
    a.place(&mut project, 10);
    let mut b = Later::begin(&mut project, 20, 81);
    b.place(&mut project, 4);
    flush(&mut project);
    a.land(&mut project, &mut map, 4);
    let old = project.recovery_path().unwrap().to_path_buf();
    project.writer.as_ref().unwrap().fail_syncs(0, 1);
    project.finish_flushes().unwrap();
    assert_ne!(project.recovery_path().unwrap(), old, "rolled over");
    flush(&mut project);
    let of_b: Vec<DabBatch> = batches(&mut project)
        .into_iter()
        .filter(|batch| batch.op() == b.op())
        .collect();
    assert_eq!(
        of_b,
        vec![batch_at(b.op(), PenDown::After(a.op), 0..4, 0)],
        "every dab of it again, still after the stroke before"
    );
    a.land(&mut project, &mut map, 6);
    let b_op = b.op();
    assert_eq!(b.begin_op_after_end(&mut project), b_op, "it begins");
    project.close(false).unwrap();
}
