//! Strokes a restore finds left open, damaged or let go.
//!
//! A stroke left open in the file by an earlier session never makes a later chain unusable; a
//! damaged batch of the stroke before never lets a later one land on part of its surface; a
//! hostile journal restores in bounded time; a stroke that drops dabs takes every stroke behind
//! it with it, so none is offered on its partial surface; and a second crash after a resume finds
//! the chain as far as it got.

use std::time::{Duration, Instant};

use super::chain::{Behind, append_and_save, offered_ops, resume_and_land, usable};
use super::roll_over::{crash_now, fail_and_roll_over};
use super::*;
use crate::fixture::record;

#[test]
fn a_stroke_left_open_by_an_earlier_session_never_makes_a_later_one_unusable() {
    // The reviewed case: a's pen-down is unknown after a roll-over, so it can't be restored.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 171);
    a.place(&mut project, 10);
    flush(&mut project);
    a.land(&mut project, &mut map, 4);
    fail_and_roll_over(&mut project);
    flush(&mut project);
    drop(project);
    let mut restored = restore(&dir);
    let found: Vec<Result<OpId, JournalUnusable>> = restored
        .take_open_strokes()
        .into_iter()
        .map(|found| found.map(|open| open.op()))
        .collect();
    assert_eq!(found, vec![Err(JournalUnusable::PenDownUnknown)]);

    // The next stroke, fully restorable, closes a in the file as it begins.
    restored.set_dab_format(FORMAT);
    let mut d = begin_on_top(&mut restored, 20, 172);
    d.place(&mut restored, 10);
    restored.wait_to_land().unwrap();
    d.land(&mut restored, &mut map, 2);
    flush(&mut restored);
    let a_batches: Vec<DabBatch> = batches(&mut restored)
        .into_iter()
        .filter(|batch| batch.op() == a.op)
        .collect();
    let closing = a_batches.last().unwrap();
    assert_eq!(closing.landed(), closing.placed(), "a closed in the file");
    drop(restored);
    let mut again = restore(&dir);
    let [open]: [OpenStroke; 1] = usable(again.take_open_strokes()).try_into().unwrap();
    assert_eq!((open.op(), open.landed()), (d.op, 2), "d, a closed");
    resume_and_land(&mut again, open, &d.edits, &mut map);
    assert_tiles(&mut again, &map, "every dab of d");
    again.close(false).unwrap();
}

#[test]
fn a_stroke_left_open_and_never_closed_ends_at_the_next_chain() {
    // A file of an earlier build, which closed no unusable stroke: a later stroke naming a save
    // point of its own starts a chain, and the one left open has ended.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let left = crash_landing(new_project(&dir), &mut map, 173);
    let point = ProjectFile::open(&candidate(&dir).path)
        .unwrap()
        .last_save_point()
        .unwrap();
    let later = next_op(left.op);
    let own = batch_at(later, PenDown::At(point), 0..3, 0).with_dab_format(FORMAT);
    append_and_save(&dir, &[own]);
    assert_eq!(
        offered_ops(restore(&dir)),
        vec![Err(JournalUnusable::Ended), Ok(later)],
        "the left one ended, the later one usable"
    );
}

#[test]
fn a_damaged_batch_of_the_stroke_before_never_lets_the_later_one_land() {
    // a's only batch doesn't read: it may have left dabs not landed, so b's surface isn't
    // known, whatever the head says.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 174);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 8);
    let mut b = Behind::begin(&mut project, 3, 175);
    b.place(&mut project, 3);
    flush(&mut project);
    let ops: Vec<OpId> = batches(&mut project).iter().map(DabBatch::op).collect();
    assert_eq!(ops, vec![a.op, b.op()], "a's batch first");
    let loc = project.store().working().batch_locations().next().unwrap();
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    drop(b.crashed());
    damage(&path, loc);
    assert_eq!(
        offered_ops(restore(&dir)),
        vec![Err(JournalUnusable::Damaged)],
        "b can't be landed on part of a"
    );
}

/// The batches a hostile file appends: `count` of them.
const HOSTILE_BATCHES: u32 = 300_000;

/// The longest a restore of a hostile file may take: far more than it needs, but a scan doing
/// work that grows with the batches kept for each batch took minutes here.
const HOSTILE_RESTORE_WITHIN: Duration = Duration::from_secs(20);

/// Appends `batches` to a recovery file holding one ended stroke, restores it, and returns how
/// long the restore took and what it offered.
fn restore_hostile(seed: u64, batches: impl Fn(OpId) -> Vec<DabBatch>) -> (Duration, usize) {
    let dir = TestDir::new();
    let ended = crash_ended(&dir, seed);
    append_and_save(&dir, &batches(next_op(ended)));
    let start = Instant::now();
    let mut restored = restore(&dir);
    let took = start.elapsed();
    let offered = restored.take_open_strokes().len();
    restored.close(false).unwrap();
    (took, offered)
}

/// Crashes a project in `dir` whose one stroke, of 10 dabs with `seed`, has ended, every dab
/// landed. Returns its op.
fn crash_ended(dir: &TestDir, seed: u64) -> OpId {
    let mut project = new_project(dir);
    let mut ended = begin_on_top(&mut project, 10, seed);
    ended.place(&mut project, 10);
    ended.land(&mut project, &mut CorrectionMap::new(), 10);
    project.end_unrecorded_stroke().unwrap();
    flush(&mut project);
    drop(project);
    ended.op
}

#[test]
fn a_hostile_journal_of_many_batches_restores_in_bounded_time() {
    let one = |op: OpId, first: u32| batch_of(op, first..first + 1, 0);
    // One stroke, a dab a batch, none landed: every batch kept.
    let (took, offered) = restore_hostile(176, |op| {
        let run = (0..HOSTILE_BATCHES).map(|k| one(op, k));
        run.collect()
    });
    assert!(took < HOSTILE_RESTORE_WITHIN, "one long run: {took:?}");
    assert_eq!(offered, 1, "one stroke");
    // Every batch overlapping the one before: the run breaks at once, and nothing more is kept.
    let (overlaps_took, overlaps_offered) = restore_hostile(177, |op| {
        let run = (0..HOSTILE_BATCHES).map(|_| one(op, 0));
        run.collect()
    });
    assert!(
        overlaps_took < HOSTILE_RESTORE_WITHIN,
        "overlaps: {overlaps_took:?}"
    );
    assert_eq!(overlaps_offered, 1, "one stroke, unusable");
    // A stroke a batch, each naming its own surface: every one before the last has ended.
    let (ops_took, ops_offered) = restore_hostile(178, |op| {
        let ops = (0..HOSTILE_BATCHES).map(|k| {
            let each = OpId::new(op.site(), op.seq() + u64::from(k)).unwrap();
            one(each, 0)
        });
        ops.collect()
    });
    assert!(ops_took < HOSTILE_RESTORE_WITHIN, "many ops: {ops_took:?}");
    assert!(ops_offered >= 1, "{ops_offered}");
}

/// Crashes a project in `dir` with a stroke landing, 20 placed and 8 landed, and one behind it
/// with 3 placed, flushed. Returns the two strokes' ops and edits.
fn crash_two(dir: &TestDir, map: &mut CorrectionMap, seed: u64) -> [(OpId, Vec<Edit>); 2] {
    let mut project = new_project(dir);
    let mut a = begin_on_top(&mut project, 20, seed);
    a.place(&mut project, 20);
    a.land(&mut project, map, 8);
    let mut b = Behind::begin(&mut project, 3, seed + 1);
    b.place(&mut project, 3);
    flush(&mut project);
    drop(project);
    let b_op = b.op();
    [(a.op, a.edits), (b_op, b.crashed())]
}

#[test]
fn a_resumed_stroke_that_drops_dabs_lets_every_stroke_behind_it_go() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let [(a_op, a_edits), (b_op, _)] = crash_two(&dir, &mut map, 179);
    let mut restored = restore(&dir);
    let [first, second]: [OpenStroke; 2] = usable(restored.take_open_strokes()).try_into().unwrap();
    restored.resume_stroke(first).unwrap();
    let mut a = Stroke {
        op: a_op,
        edits: a_edits,
        placed: 20,
        landed: 8,
    };
    a.land(&mut restored, &mut map, 4);
    // A store error in its landing: its other dabs are dropped.
    let dropped = restored.end_stroke_dropping_later(None).unwrap();
    assert_eq!(dropped, vec![b_op], "b goes with it");
    let refused = restored.resume_stroke(second).unwrap_err();
    assert!(
        matches!(refused.error(), ProjectError::NotResumable(op) if *op == b_op),
        "{refused:?}"
    );
    flush(&mut restored);
    drop(restored);
    assert_eq!(
        offered_ops(restore(&dir)),
        Vec::new(),
        "neither offered: a's dropped dabs counted landed, b closed"
    );
}

#[test]
fn a_live_stroke_that_drops_dabs_lets_every_stroke_behind_it_go() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut a = begin_on_top(&mut project, 20, 180);
    a.place(&mut project, 20);
    a.land(&mut project, &mut map, 6);
    let mut b = Behind::begin(&mut project, 3, 181);
    b.place(&mut project, 3);
    let mut c = Behind::begin(&mut project, 2, 182);
    c.place(&mut project, 2);
    flush(&mut project);
    let ops = vec![b.op(), c.op()];
    assert_eq!(project.end_stroke_dropping_later(None).unwrap(), ops);
    for behind in [b, c] {
        assert_eq!(project.drop_stroke_after(behind.reserved), Vec::new());
    }
    flush(&mut project);
    drop(project);
    let mut restored = restore(&dir);
    assert!(restored.take_open_strokes().is_empty(), "none offered");
    assert_tiles(&mut restored, &map, "a's 6 dabs alone");
    restored.close(false).unwrap();
}

/// Lands every stroke `restored` offers, in op order, each its dabs from `edits` (by op), on a
/// map of its own; then asserts its tiles are `expected`. Closes it.
fn land_every_offered(
    mut restored: Project,
    edits: &[(OpId, Vec<Edit>)],
    expected: &CorrectionMap,
) {
    let offered = usable(restored.take_open_strokes());
    for open in offered {
        let (_, own) = edits.iter().find(|(op, _)| *op == open.op()).unwrap();
        resume_and_land(&mut restored, open, own, &mut CorrectionMap::new());
    }
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, expected, "every dab of each");
    restored.close(false).unwrap();
}

/// What landing every dab of `strokes`, in op order, `placed` dabs each, leaves.
fn every_dab(strokes: &[(OpId, Vec<Edit>)], placed: [usize; 2]) -> CorrectionMap {
    let mesh = mesh();
    let mut expected = CorrectionMap::new();
    let each = strokes.iter().zip(placed);
    for ((_, edits), count) in each {
        record(&mesh, &mut expected, &edits[..count]);
    }
    expected
}

#[test]
fn a_second_crash_after_a_resumed_stroke_s_cut_restores_the_chain_as_far_as_it_got() {
    for durable in [true, false] {
        let dir = TestDir::new();
        let mut map = CorrectionMap::new();
        let strokes = crash_two(&dir, &mut map, 183);
        let expected = every_dab(&strokes, [20, 3]);
        let mut restored = restore(&dir);
        let [first, second]: [OpenStroke; 2] =
            usable(restored.take_open_strokes()).try_into().unwrap();
        resume_and_land(&mut restored, first, &strokes[0].1, &mut map);
        restored.resume_stroke(second).unwrap();
        if durable {
            // Its cut durable, and 2 of its dabs landed after it.
            restored.wait_to_land().unwrap();
            flush(&mut restored);
            let mut b = Stroke {
                op: strokes[1].0,
                edits: strokes[1].1.clone(),
                placed: 3,
                landed: 0,
            };
            b.land(&mut restored, &mut map, 2);
            flush(&mut restored);
            drop(restored);
            let again = restore(&dir);
            land_every_offered(again, &strokes, &expected);
        } else {
            // Crashed as its cut is taken, whatever was durable by then.
            crash_now(restored);
            land_every_offered(restore(&dir), &strokes, &expected);
        }
    }
}

#[test]
fn a_resume_after_the_stroke_before_that_fails_leaves_it_to_resume_again() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let strokes = crash_two(&dir, &mut map, 185);
    let mut restored = restore(&dir);
    let [first, second]: [OpenStroke; 2] = usable(restored.take_open_strokes()).try_into().unwrap();
    resume_and_land(&mut restored, first, &strokes[0].1, &mut map);
    restored.fail_next_resume = true;
    let failed = restored.resume_stroke(second).unwrap_err();
    assert!(
        matches!(failed.error(), ProjectError::Cache(_)),
        "{failed:?}"
    );
    let b_op = strokes[1].0;
    assert!(
        !restored.store().journal_open_for(b_op),
        "its journal dropped"
    );
    resume_and_land(&mut restored, failed.into_stroke(), &strokes[1].1, &mut map);
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab of a, then of b");
    restored.close(false).unwrap();
}
