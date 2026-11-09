//! The journals of strokes reserved behind the writing one, on the store alone.
//!
//! Their op begins only once the writing stroke's journal has closed, and a move onto a fresh
//! file names their surface as the strokes before them leave it.

use super::*;
use crate::fixture::dab_at;
use crate::journal::{DAB_BYTES, DabBatch, PenDown};
use crate::tiles::journal::Rebase;

/// The tests' dabs `first..first + count`, as [`dab_at`] makes them.
fn rows(first: u32, count: u32) -> Vec<[u8; DAB_BYTES]> {
    (first..first + count).map(dab_at).collect()
}

/// The pen-down each batch of `op` the store's journals cut now names, dropping the cut after.
fn cut_pen_downs(store: &mut TileStore, op: OpId) -> Vec<PenDown> {
    let (_, batches) = store.cut_journal().unwrap();
    store.drop_journal_cut();
    let of_op = batches.iter().filter(|batch| batch.op() == op);
    of_op.map(DabBatch::pen_down).collect()
}

/// A store whose writing stroke `a` is open, its journal at the file's base, with stroke `b`
/// reserved behind it, two dabs of it journalled.
fn a_with_b_behind(dir: &TestDir) -> (TileStore, OpId, ReservedOp) {
    let mesh = mesh();
    let mut store = new_store(dir, "work.bcp", &mesh, roomy());
    let a = store.begin_op().unwrap();
    store.begin_journal(a, PenDownPoint::Known(PenDown::Base));
    let b = store.reserve_op().unwrap();
    store.begin_journal_after(b.op());
    store.journal(b.op(), &rows(0, 2)).unwrap();
    (store, a, b)
}

/// One begun stroke at a time: a reserved op doesn't begin while the writing stroke's journal is
/// open, even with no op current (as a save leaves the store), and begins once it closes.
#[test]
fn a_reserved_op_begins_only_once_the_writing_stroke_s_journal_closes() {
    let dir = TestDir::new();
    let (mut store, a, b) = a_with_b_behind(&dir);
    // What a save's rebase does: the store's op ends, the stroke's journal stays open.
    store.op = None;
    let refused = store.begin_reserved_op(b).unwrap_err();
    assert!(
        matches!(refused.error(), CacheError::OpBegun(op) if *op == a),
        "a's journal is open: {refused:?}"
    );
    store.end_journal();
    let b_op = store.begin_reserved_op(refused.into_reserved()).unwrap();
    assert_eq!(store.current_op(), Some(b_op), "begun once it closed");
}

/// A save's rebase: a reserved stroke goes on naming the stroke before while that one's journal
/// is open; once it has ended, the saved state, the file's base, is the surface it left.
#[test]
fn a_save_names_a_reserved_stroke_after_an_open_one_and_at_the_base_after_an_ended_one() {
    let dir = TestDir::new();
    let (mut store, a, b) = a_with_b_behind(&dir);
    store.rebase_journal(Rebase::Save);
    assert_eq!(
        cut_pen_downs(&mut store, b.op()),
        vec![PenDown::After(a)],
        "after a, still open"
    );
    store.end_journal();
    store.end_op();
    store.rebase_journal(Rebase::Save);
    assert_eq!(
        cut_pen_downs(&mut store, b.op()),
        vec![PenDown::Base],
        "a ended: the saved state"
    );
    let b_op = b.op();
    store.drop_reserved_op(b);
    assert!(!store.journal_open_for(b_op), "dropped, closed");
}

/// A stroke whose op began through a roll-over that carried it keeps the carried pen-down.
///
/// Its writes from then on are its own: a save after it, with no write since it began, finds its
/// surface in the saved state.
#[test]
fn a_reserved_stroke_carried_as_it_begins_is_begun_from_then_on() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, a, b) = a_with_b_behind(&dir);
    let key = tiles(&mesh, TOP, TOP)[0];
    let region = Region::of_tile(&key);
    let lift = Vec3::new(0.0, 0.0, 0.01);
    store
        .set_correction(
            key.face(),
            key.level(),
            region.x0() + 3,
            region.y0() + 3,
            lift,
        )
        .unwrap();
    store.end_journal();
    store.end_op();
    let b_op = store.begin_reserved_op(b).unwrap();
    let (_, carried) = store.carry_journal(7).unwrap().unwrap();
    store.rebase_journal(Rebase::RollOver(Some(carried)));
    store.journal_op_began(b_op, PenDownPoint::Known(PenDown::At(99)));
    store.journal(b_op, &rows(2, 1)).unwrap();
    assert_eq!(
        cut_pen_downs(&mut store, b_op),
        vec![PenDown::At(7)],
        "the carried pen-down stands"
    );
    store.rebase_journal(Rebase::Save);
    assert_eq!(
        cut_pen_downs(&mut store, b_op),
        vec![PenDown::Base],
        "nothing written since it began: a's write came before"
    );
    assert!(a < b_op);
}
