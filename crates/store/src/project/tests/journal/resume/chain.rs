//! Restoring a chain of open strokes.
//!
//! A crash with several strokes open, the oldest landing and the later ones begun behind it,
//! their dabs placed and none landed, restores every one of them in op order: each resumes once
//! the one before it has landed and ended, on the surface it left. A stroke whose earlier one
//! can't be used can't be either, and a closed stroke never hides an open one.

use super::roll_over::{crash_now, fail_and_roll_over, landing_with_shapes};
use super::*;
use crate::tiles::ReservedOp;

/// A stroke begun behind the newest open one: its op reserved, its dabs journalled, none landed.
struct Behind {
    /// The op reserved for it.
    reserved: ReservedOp,
    /// The edit of each dab, by index.
    edits: Vec<Edit>,
    /// The dabs placed so far.
    placed: u32,
}

impl Behind {
    /// Begins a stroke behind the newest open one on `project`, with room for `count` dabs on
    /// the finest level, their edits from `seed`.
    fn begin(project: &mut Project, count: u32, seed: u64) -> Self {
        let edits = top_edits(&project.store().working().mesh(), count, seed);
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

    /// Its edits, the token given up: the project it was reserved on has crashed.
    fn crashed(self) -> Vec<Edit> {
        self.reserved.abandon_with_store_gone();
        self.edits
    }
}

/// Every stroke `found` offers, each usable: their ops, in the order offered.
fn usable(found: Vec<Result<OpenStroke, JournalUnusable>>) -> Vec<OpenStroke> {
    let usable = found
        .into_iter()
        .map(|stroke| stroke.expect("a stroke to land"));
    usable.collect()
}

/// Resumes `open`, a stroke a restore found whose dabs are `edits`, waits until it may land, lands
/// every dab not landed into `project` and `map`, and ends it.
fn resume_and_land(
    project: &mut Project,
    open: OpenStroke,
    edits: &[Edit],
    map: &mut CorrectionMap,
) {
    let (op, landed) = (open.op(), open.landed());
    let placed = landed + u32::try_from(open.dabs().len()).unwrap();
    project.resume_stroke(open).unwrap();
    assert_eq!(project.store().current_op(), Some(op), "its op again");
    project.wait_to_land().unwrap();
    let mut stroke = Stroke {
        op,
        edits: edits.to_vec(),
        placed,
        landed,
    };
    stroke.land(project, map, placed - landed);
    project.end_unrecorded_stroke().unwrap();
}

/// Asserts that a restore of `dir`'s recovery file again finds no stroke open, and the tiles of
/// `map`. Closes it.
fn assert_restores_finished(dir: &TestDir, map: &CorrectionMap) {
    let mut again = restore(dir);
    assert!(again.take_open_strokes().is_empty(), "every stroke ended");
    assert_tiles(&mut again, map, "restored again");
    again.close(false).unwrap();
}

#[test]
fn two_open_strokes_restore_in_op_order_and_the_later_lands_on_the_surface_the_earlier_left() {
    // The review's first scratch case: a 20 placed and 8 landed, b behind it with 3 placed, b's
    // batch the file's last.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 141);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 8);
    let mut b = Behind::begin(&mut project, 3, 142);
    b.place(&mut project, 3);
    flush(&mut project);
    let b_op = b.op();
    let last = batches(&mut project).pop().unwrap();
    assert_eq!(last.op(), b_op, "b's batch last");
    drop(project);
    let b_edits = b.crashed();

    let mut restored = restore(&dir);
    assert_tiles(&mut restored, &map, "a's 8 dabs landed");
    let [first, second]: [OpenStroke; 2] = usable(restored.take_open_strokes()).try_into().unwrap();
    assert_eq!((first.op(), first.landed()), (a.op, 8), "a first");
    assert_eq!(first.dabs(), dabs(8..20), "a's dabs not landed");
    assert_eq!(first.pen_down(), PenDown::Base, "a's own surface");
    assert_eq!(first.after(), None, "a waits for none");
    assert_eq!((second.op(), second.landed()), (b_op, 0), "then b");
    assert_eq!(second.dabs(), dabs(0..3), "b's dabs");
    assert_eq!(second.pen_down(), PenDown::After(a.op), "after a");
    assert_eq!(second.after(), Some(a.op), "b waits for a");
    assert!(restored.take_open_strokes().is_empty(), "taken once");

    // b resumes only once a has landed and ended.
    let early = restored.resume_stroke(second).unwrap_err();
    assert!(
        matches!(early.error(), ProjectError::NotResumable(op) if *op == b_op),
        "{early:?}"
    );
    let given_back = early.into_stroke();
    resume_and_land(&mut restored, first, &a.edits, &mut map);
    resume_and_land(&mut restored, given_back, &b_edits, &mut map);
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of a, then every dab of b");
    restored.close(true).unwrap();
    assert_restores_finished(&dir, &map);
}

#[test]
fn three_open_strokes_restore_in_op_order_each_on_the_surface_the_one_before_left() {
    // b's and c's batches are not the file's last: a's count alone came after them.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 30, 143);
    a.place(&mut project, 30);
    a.land(&mut project, &mut map, 5);
    let mut b = Behind::begin(&mut project, 12, 144);
    b.place(&mut project, 8);
    flush(&mut project);
    b.place(&mut project, 4);
    let mut c = Behind::begin(&mut project, 6, 145);
    c.place(&mut project, 6);
    a.land(&mut project, &mut map, 4);
    flush(&mut project);
    a.land(&mut project, &mut map, 3);
    flush(&mut project);
    let ops = [a.op, b.op(), c.op()];
    assert_eq!(
        batches(&mut project).pop().unwrap(),
        batch_of(a.op, 30..30, 12),
        "a's count last"
    );
    drop(project);
    let (b_edits, c_edits) = (b.crashed(), c.crashed());

    let mut restored = restore(&dir);
    assert_tiles(&mut restored, &map, "a's 12 dabs landed");
    let found = usable(restored.take_open_strokes());
    let offered: Vec<(OpId, u32, usize, PenDown)> = found
        .iter()
        .map(|open| (open.op(), open.landed(), open.dabs().len(), open.pen_down()))
        .collect();
    assert_eq!(
        offered,
        vec![
            (ops[0], 12, 18, PenDown::Base),
            (ops[1], 0, 12, PenDown::After(ops[0])),
            (ops[2], 0, 6, PenDown::After(ops[1])),
        ],
        "in op order, each after the one before"
    );
    let every_edit = [a.edits, b_edits, c_edits];
    let landing = found.into_iter().zip(&every_edit);
    for (open, edits) in landing {
        resume_and_land(&mut restored, open, edits, &mut map);
    }
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of each, in op order");
    restored.close(true).unwrap();
    assert_restores_finished(&dir, &map);
}

#[test]
fn strokes_never_resumed_keep_their_ops_from_being_handed_out_and_are_closed_by_the_next_op() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 146);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 4);
    let mut b = Behind::begin(&mut project, 3, 147);
    b.place(&mut project, 3);
    flush(&mut project);
    a.land(&mut project, &mut map, 2);
    flush(&mut project);
    let b_op = b.op();
    drop(project);
    drop(b.crashed());

    let mut restored = restore(&dir);
    assert_eq!(usable(restored.take_open_strokes()).len(), 2, "both found");
    let next = restored.begin_stroke().unwrap();
    assert!(
        next > b_op,
        "{next:?} after b's {b_op:?}, which only its batches hold"
    );
    restored.end_unrecorded_stroke().unwrap();
    restored.close(true).unwrap();
    assert_restores_finished(&dir, &map);
}

#[test]
fn an_unusable_oldest_stroke_makes_every_later_one_unusable() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    // A roll-over in the middle of a stroke that kept no shapes: its pen-down is unknown.
    let mut a = begin_on_top(&mut project, 20, 148);
    a.place(&mut project, 10);
    flush(&mut project);
    a.land(&mut project, &mut map, 4);
    fail_and_roll_over(&mut project);
    let mut b = Behind::begin(&mut project, 3, 149);
    b.place(&mut project, 3);
    let mut c = Behind::begin(&mut project, 2, 150);
    c.place(&mut project, 2);
    flush(&mut project);
    let b_op = b.op();
    drop(project);
    drop((b.crashed(), c.crashed()));

    let mut restored = restore(&dir);
    let found: Vec<Result<(), JournalUnusable>> = restored
        .take_open_strokes()
        .into_iter()
        .map(|found| found.map(drop))
        .collect();
    let a_why = JournalUnusable::PenDownUnknown;
    let b_why = JournalUnusable::EarlierUnusable {
        earlier: a.op,
        why: Box::new(a_why.clone()),
    };
    let c_why = JournalUnusable::EarlierUnusable {
        earlier: b_op,
        why: Box::new(b_why.clone()),
    };
    assert_eq!(
        found,
        vec![Err(a_why), Err(b_why), Err(c_why)],
        "each with its own reason, the earlier's behind it"
    );
    assert_tiles(&mut restored, &map, "the tiles restored alone");
    restored.close(false).unwrap();
}

#[test]
fn a_stroke_begun_after_the_one_before_ended_restores_from_its_own_cut() {
    // The review's second scratch case: a ended, b begun, its cut durable and 2 of its dabs
    // landed; its batches went from after a to its own save point.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 151);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 6);
    let mut b = Behind::begin(&mut project, 8, 152);
    b.place(&mut project, 5);
    flush(&mut project);
    a.land(&mut project, &mut map, 14);
    project.end_unrecorded_stroke().unwrap();
    let b_op = b.op();
    let begun = project.begin_reserved_stroke(b.reserved);
    assert_eq!(begun.map_err(|e| e.into_parts().1).unwrap(), b_op);
    let mut b = Stroke {
        op: b_op,
        edits: b.edits,
        placed: 5,
        landed: 0,
    };
    project.wait_to_land().unwrap();
    flush(&mut project);
    b.place(&mut project, 3);
    b.land(&mut project, &mut map, 2);
    flush(&mut project);
    let named: Vec<PenDown> = batches(&mut project)
        .iter()
        .filter(|batch| batch.op() == b_op)
        .map(DabBatch::pen_down)
        .collect();
    assert_eq!(named.first(), Some(&PenDown::After(a.op)), "{named:?}");
    let Some(&PenDown::At(cut)) = named.last() else {
        panic!("its own cut last: {named:?}");
    };
    drop(project);

    let mut restored = restore(&dir);
    let [open]: [OpenStroke; 1] = usable(restored.take_open_strokes()).try_into().unwrap();
    assert_eq!((open.op(), open.landed()), (b_op, 2), "b alone, 2 landed");
    assert_eq!(open.dabs(), dabs(2..8), "its dabs not landed");
    assert_eq!(open.pen_down(), PenDown::At(cut), "its own cut");
    assert_eq!(open.after(), None, "nothing to wait for");
    let edits = b.edits;
    resume_and_land(&mut restored, open, &edits, &mut map);
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of a, then of b");
    restored.close(true).unwrap();
    assert_restores_finished(&dir, &map);
}

#[test]
fn a_stroke_after_one_that_ended_in_the_file_resumes_at_once_on_the_restored_tiles() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 10, 153);
    a.place(&mut project, 10);
    let mut b = Behind::begin(&mut project, 4, 154);
    b.place(&mut project, 4);
    a.land(&mut project, &mut map, 10);
    project.end_unrecorded_stroke().unwrap();
    flush(&mut project);
    let b_op = b.op();
    drop(project);
    let b_edits = b.crashed();

    let mut restored = restore(&dir);
    assert_tiles(&mut restored, &map, "every dab of a");
    let [open]: [OpenStroke; 1] = usable(restored.take_open_strokes()).try_into().unwrap();
    assert_eq!(open.op(), b_op, "b alone: a's batches close it");
    assert_eq!(open.pen_down(), PenDown::After(a.op), "named after a");
    assert_eq!(
        open.after(),
        None,
        "a ended in the file: the tiles are its surface"
    );
    resume_and_land(&mut restored, open, &b_edits, &mut map);
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of b on them");
    restored.close(true).unwrap();
    assert_restores_finished(&dir, &map);
}

#[test]
fn a_dropped_stroke_s_closing_batch_last_never_hides_the_open_one_before_it() {
    // The review's third scratch case.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 155);
    a.place(&mut project, 10);
    a.land(&mut project, &mut map, 4);
    let mut b = Behind::begin(&mut project, 3, 156);
    b.place(&mut project, 3);
    flush(&mut project);
    let b_op = b.op();
    assert_eq!(project.drop_stroke_after(b.reserved), vec![b_op]);
    flush(&mut project);
    let last = batches(&mut project).pop().unwrap();
    assert_eq!(
        (last.op(), last.landed(), last.placed()),
        (b_op, 3, 3),
        "b's closing batch last"
    );
    drop(project);

    let mut restored = restore(&dir);
    let [open]: [OpenStroke; 1] = usable(restored.take_open_strokes()).try_into().unwrap();
    assert_eq!((open.op(), open.landed()), (a.op, 4), "a, still open");
    assert_eq!(open.dabs(), dabs(4..10), "its dabs not landed");
    resume_and_land(&mut restored, open, &a.edits, &mut map);
    assert_tiles(&mut restored, &map, "every dab of a");
    restored.close(false).unwrap();
}

#[test]
fn a_roll_over_with_strokes_open_behind_the_landing_one_keeps_them_all_restorable() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let (mut project, landing, live) = landing_with_shapes(&dir, &mut map, 157);
    let mut b = Behind::begin(&mut project, 4, 158);
    b.place(&mut project, 4);
    let mut c = Behind::begin(&mut project, 2, 159);
    c.place(&mut project, 2);
    let points = fail_and_roll_over(&mut project);
    let [cut, _] = points[..] else {
        panic!("the pen-down cut and the fill's own: {points:?}");
    };
    let (a_op, b_op, c_op) = (landing.op, b.op(), c.op());
    let carried = vec![
        batch_at(a_op, PenDown::At(cut), 9..30, 9).with_dab_format(FORMAT),
        batch_at(b_op, PenDown::After(a_op), 0..4, 0).with_dab_format(FORMAT),
        batch_at(c_op, PenDown::After(b_op), 0..2, 0).with_dab_format(FORMAT),
    ];
    assert_eq!(batches(&mut project), carried, "each carried, in op order");
    crash_now(project);
    let behind = [b.crashed(), c.crashed()];

    let mut restored = restore(&dir);
    let found = usable(restored.take_open_strokes());
    let offered: Vec<(OpId, PenDown)> = found
        .iter()
        .map(|open| (open.op(), open.pen_down()))
        .collect();
    assert_eq!(
        offered,
        vec![
            (a_op, PenDown::At(cut)),
            (b_op, PenDown::After(a_op)),
            (c_op, PenDown::After(b_op)),
        ],
        "every one restorable"
    );
    let mut found = found.into_iter();
    restored.resume_stroke(found.next().unwrap()).unwrap();
    let changed = keys_of(&mesh(), &landing.edits[..9]);
    for key in &changed {
        assert_eq!(
            restored.store().pen_down_shape(key),
            live.get(key),
            "{key:?}: a's live shape"
        );
    }
    let mut a = Stroke {
        op: a_op,
        edits: landing.edits,
        placed: 30,
        landed: 9,
    };
    a.land(&mut restored, &mut map, 21);
    restored.end_unrecorded_stroke().unwrap();
    let rest = found.zip(&behind);
    for (open, edits) in rest {
        resume_and_land(&mut restored, open, edits, &mut map);
    }
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of each, in op order");
    restored.close(false).unwrap();
}

#[test]
fn a_roll_over_with_a_landed_stroke_open_keeps_the_one_behind_it_restorable() {
    // Every dab of a has landed, but its stroke hasn't ended: the carry closes it in the fresh
    // file, so b's surface is the state it holds.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let (mut project, mut landing, _) = landing_with_shapes(&dir, &mut map, 160);
    // Every tile it changes keeps its shape before its first edit, so the roll-over carries it.
    let rest: Vec<TileKey> = keys_of(&mesh(), &landing.edits[12..]).into_iter().collect();
    project.store().keep_pen_down(&rest).unwrap();
    landing.land(&mut project, &mut map, 21);
    let mut b = Behind::begin(&mut project, 4, 161);
    b.place(&mut project, 4);
    fail_and_roll_over(&mut project);
    let a_op = landing.op;
    crash_now(project);
    let b_edits = b.crashed();

    let mut restored = restore(&dir);
    assert_tiles(&mut restored, &map, "every dab of a");
    let [open]: [OpenStroke; 1] = usable(restored.take_open_strokes()).try_into().unwrap();
    assert_eq!(open.pen_down(), PenDown::After(a_op), "after a");
    assert_eq!(open.after(), None, "a closed in the fresh file");
    resume_and_land(&mut restored, open, &b_edits, &mut map);
    assert_tiles(&mut restored, &map, "every dab of b on a's");
    restored.close(false).unwrap();
}

#[test]
fn a_stroke_named_after_one_the_file_holds_nothing_of_is_unusable() {
    let after_none = |op: OpId| {
        let later = next_op(next_op(op));
        vec![batch_at(later, PenDown::After(next_op(op)), 0..3, 0).with_dab_format(FORMAT)]
    };
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 162);
    append_and_save(&dir, &after_none(stroke.op));
    assert_eq!(
        offered_ops(restore(&dir)),
        vec![Ok(stroke.op), Err(JournalUnusable::Mismatch)],
        "the open stroke before it isn't the one it names"
    );

    // With no stroke open before it: its surface is in no save point.
    let alone = TestDir::new();
    let ended = crash_ended(&alone, &mut CorrectionMap::new(), 163);
    append_and_save(&alone, &after_none(ended));
    assert_eq!(
        offered_ops(restore(&alone)),
        vec![Err(JournalUnusable::PenDownUnknown)]
    );
}

/// The op of each stroke `restored` offers, or why it can't be landed. Closes it.
fn offered_ops(mut restored: Project) -> Vec<Result<OpId, JournalUnusable>> {
    let offered = restored
        .take_open_strokes()
        .into_iter()
        .map(|found| found.map(|open| open.op()));
    let ops: Vec<Result<OpId, JournalUnusable>> = offered.collect();
    restored.close(false).unwrap();
    ops
}

/// Crashes a project in `dir` whose one stroke, of 10 dabs with `seed`, has ended, every dab
/// landed before any flush: its edits are in the file, and no batch of it. Returns its op.
fn crash_ended(dir: &TestDir, map: &mut CorrectionMap, seed: u64) -> OpId {
    let mut project = new_project(dir);
    let mut ended = begin_on_top(&mut project, 10, seed);
    ended.place(&mut project, 10);
    ended.land(&mut project, map, 10);
    project.end_unrecorded_stroke().unwrap();
    flush(&mut project);
    drop(project);
    ended.op
}

/// Appends `extra` to the recovery file in `dir`, with a save point of their own.
fn append_and_save(dir: &TestDir, extra: &[DabBatch]) {
    let mut file = ProjectFile::open(&candidate(dir).path).unwrap();
    let framed: Vec<Vec<u8>> = extra
        .iter()
        .map(|batch| frame_batch(batch).unwrap())
        .collect();
    file.append_batches(&framed).unwrap();
    file.save().unwrap();
}

#[test]
fn a_damaged_batch_among_interleaved_ones_makes_only_the_strokes_it_may_end_unusable() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 164);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 3);
    let mut b = Behind::begin(&mut project, 6, 165);
    b.place(&mut project, 4);
    flush(&mut project);
    b.place(&mut project, 2);
    flush(&mut project);
    a.land(&mut project, &mut map, 2);
    flush(&mut project);
    // a 0..20, b 0..4, b 4..6, a's count: b's second may have been a's closing batch, and was
    // b's; a's count after it says a is open still, so only b, last before the damage, can't
    // be finished.
    let b_op = b.op();
    let ops: Vec<OpId> = batches(&mut project).iter().map(DabBatch::op).collect();
    assert_eq!(ops, vec![a.op, b_op, b_op, a.op]);
    let loc = project.store().working().batch_locations().nth(2).unwrap();
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    drop(b.crashed());
    damage(&path, loc);
    let mut restored = restore(&dir);
    let found: Vec<Result<OpId, JournalUnusable>> = restored
        .take_open_strokes()
        .into_iter()
        .map(|found| found.map(|open| open.op()))
        .collect();
    assert_eq!(found, vec![Ok(a.op), Err(JournalUnusable::Damaged)]);
    restored.close(false).unwrap();
}
