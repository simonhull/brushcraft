//! The open strokes' dab journals on the store (see [`crate::journal`]).
//!
//! Each holds the dabs its stroke placed and hasn't landed, how many have landed, what the working
//! file holds of them so far, and where the surface the stroke began on is.
//!
//! # Side by side
//!
//! One stroke writes at a time: the stroke of the store's current op, the **writing** stroke.
//! Strokes begun while it still lands have their ops reserved behind it
//! ([`TileStore::reserve_op`]), and their journals open at once, in op order after its
//! ([`TileStore::begin_journal_after`]), so their dabs reach the working file within a second as
//! any stroke's do. A reserved stroke lands nothing, so its landed count stays 0 until its op
//! begins; only the writing stroke's count moves ([`TileStore::journal_landed`]). Every flush cuts
//! every journal at once, so each cut pairs the store's tiles with the writing stroke's count, and
//! with none landed of the strokes after it.
//!
//! # Through a flush
//!
//! The recovery writer takes the journal's **cut** as it takes its tile snapshot, on the owner
//! thread, at the same moment ([`TileStore::cut_journal`]): a [`DabBatch`] of the dabs placed and
//! not yet written, from whichever is later of the last written and the last landed (a dab landed
//! is never written: the tiles hold it), with the landed count as it is then. The app passes the
//! landed count ([`TileStore::journal_landed`]) with each landing, on the owner thread, with no
//! tick (so no cut) between the landing's edits and its count, in either order: a cut after the
//! count and before the edits would count dabs landed that its tiles lack, and a restore would
//! lose them; a cut after the edits and before the count would name tiles holding dabs its count
//! leaves out, and a restore would land them twice. So at any cut the count is that of the dabs
//! whose edits the snapshot copies, and the batch's count describes exactly the tiles that
//! flush's save point names.
//!
//! The cut stays **in flight** on the journal until the writer appends its batches with the
//! flush's records ([`TileStore::append_journal`]), before the tile sync, so the save point that
//! names the cut's tiles holds its batch. Only then does the journal note it written: batches are
//! never deduplicated, so a flush abandoned before its append (a failed encode, an append that
//! fails, a flush overtaken by a rebase) drops the cut ([`TileStore::drop_journal_cut`]) and
//! leaves the journal as it was, and the next flush writes the same dabs, with the landed count it
//! finds, once, never overlapping a batch in the file. A flush abandoned after its append (a failed
//! save point) leaves its batches in the file for the next save point to hold, and the journal
//! goes on after them. One flush is in flight at a time, so a journal has at most one cut in
//! flight; each is named ([`CutId`]), so a pen-down can name the flush that holds it.
//!
//! A batch is due while a dab placed isn't written, or the landed count moved since the last
//! batch: then a batch of no dabs carries the count alone. A stroke none of whose batches is in
//! the file, or in flight, and whose dabs have all landed is due none: its tiles hold every dab.
//!
//! # The pen-down save point
//!
//! Each journal knows where its stroke's pen-down surface is ([`PenDownPoint`]), and every batch
//! it cuts names it ([`crate::PenDown`]). [`TileStore::begin_journal`] is told which flush holds
//! it: one started as the stroke began, or the one in flight if no edit followed its cut, or the
//! next one to start, queued behind the one in flight; or none, with every edit already named by
//! the working file's latest save point, which then holds it (the file's base, before its first).
//! The next flush's cut takes it up ([`TileStore::cut_journal`]); a named flush settles it, with
//! its save point's offset once durable ([`TileStore::pen_down_settled`]), or as unknown if it is
//! abandoned ([`TileStore::pen_downs_abandoned`]).
//!
//! **The stroke lands nothing before its pen-down flush has cut** ([`TileStore::may_land`]): the
//! cut must hold every earlier op's edits and none of the stroke's own. And **a journal cuts no
//! batch before its pen-down settles**: its dabs wait for the flush after (they are due none
//! until then), so every batch names a settled save point. Flushes run one at a time, so the
//! pen-down flush is durable or abandoned before the next one cuts.
//!
//! A stroke begun behind the writing one is the exception: until its own pen-down cut settles,
//! its surface is the one the stroke before it leaves, which its batches name
//! ([`PenDown::After`]), so they are cut from the first flush on. Its own cut is taken as its op
//! begins, once the op before has ended, as any stroke's is ([`TileStore::journal_op_began`]);
//! once that cut settles its later batches name its save point, never `After` again. It lands
//! nothing before that cut either, and the flush after the cut starts only once the cut has
//! settled, so a batch counting any of its dabs landed always names its own save point. Its
//! pen-down becomes unknown only as any stroke's does (its cut abandoned, or a save or a roll-over
//! it isn't carried through); no flush abandoned makes an `After` unknown, as it waits on none.
//!
//! # When the stroke ends
//!
//! The stroke's end ([`TileStore::end_journal`]) closes its journal, counting every dab landed:
//! dabs a stroke dropped (after an error) must never land on a restore. If that count isn't
//! written yet, or a cut of the journal is in flight, the journal waits among the ended ones for
//! the flushes that write its last count, so the file never ends a stroke on a batch whose count is
//! behind its tiles; then it goes. The next stroke's journal is a new one meanwhile
//! ([`TileStore::begin_journal`]), and a flush writes the ended strokes' batches before its own.
//! A reserved stroke let go before its op began ([`TileStore::drop_reserved_ops_from`]) ends the
//! same way: none of its dabs ever lands, and its last batch says so.
//!
//! # A restored stroke
//!
//! A restore finds the strokes still landing in the working file's batches (see
//! `crate::project::resume`), and resuming one opens its journal again as they left it
//! ([`TileStore::resume_journal`]): its landed count, its rows not landed (already written, so
//! never written again), its pen-down where the batches name it and its rows' format. It goes on
//! as any journal from there, its next batch carrying the landed count alone. One whose batches
//! name the stroke before it ([`PenDown::After`]), resumed once that one has ended, opens as a
//! reserved stroke's does, still naming it, and its op's cut is taken as a reserved stroke's op
//! begins ([`TileStore::resume_journal_after`], [`TileStore::journal_op_began`]).
//!
//! # Onto a fresh working file
//!
//! A save's rebase takes none of the batches with it ([`TileStore::rebase_journal`]): each open
//! stroke writes its dabs not landed again, from its landed count, with the next flush, so the
//! rows are kept at hand for it. The writing stroke's pen-down is the new file's base if the store
//! took no write since the stroke began, else unknown. A reserved stroke goes on naming the stroke
//! before it while that one's journal is open; else that one has ended, nothing has written since,
//! and the saved state, the new file's base, is the surface it left.
//!
//! A **roll-over** after a failed sync carries the open strokes instead
//! ([`TileStore::fill_roll_over`]): the fresh file gets a save point of its own holding the
//! writing stroke's pen-down surface, written from the shapes the store keeps in memory (the
//! **pen-down cut**), then the current state with every open stroke's batches, in op order
//! ([`TileStore::carry_journal`]): the writing stroke's from its landed count, naming that save
//! point, and each reserved behind it with every dab it placed, naming the stroke before it as
//! it does; and its own save point. The journals then go on from what the carry wrote, so the
//! strokes still landing survive the roll-over, a crash right after it included, and restore in
//! op order as before it. A writing stroke whose every dab has landed gets a batch of its count
//! alone if a stroke is reserved behind it, which closes it in the fresh file: the one behind
//! then lands on the state the file holds. A writing stroke whose surface the store can't write
//! from memory (a tile it changed with no shape kept, a tile lost to the roll-over, or a save in
//! its middle) isn't carried, nor are those behind it: as after a save, its next flush writes its
//! dabs not landed again, its pen-down unknown, and theirs still name it, as unusable as it is.
//! With no writing stroke (the one before has ended, the reserved one's op not yet begun), a
//! reserved stroke's next flush writes its dabs again naming its surface unknown, as the
//! roll-over names no save point of the surface the stroke before left.
//!
//! # Memory and cost
//!
//! A journal keeps the rows of the dabs not landed, [`DAB_BYTES`] each: the landing's backlog,
//! about 1.4 MB at the forced landing's 16,384 dabs. Every row a landed count passes goes at once.
//!
//! The owner thread's cost per flush is the cut: a copy of the rows not yet written, about a
//! second of placed dabs. Encoding, compression and syncing happen on the writer thread.

use std::collections::VecDeque;

use thiserror::Error;

use super::TileStore;
use super::error::{CacheError, Invariant};
use crate::OpId;
use crate::journal::{DAB_BYTES, DabBatch, MAX_JOURNAL_DABS, PenDown};
use crate::stamp::Stamp;

/// The name of a flush's cut of the journals ([`TileStore::cut_journal`]).
///
/// A journal tells its cut in flight by it, and a pen-down names the flush that holds it by it.
/// Each cut the store takes is named by the next number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CutId(u64);

/// Where a journal's pen-down surface is (see the module notes): in the save point of a flush not
/// yet settled, or known; or, for a stroke whose op is reserved, its cut not yet taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PenDownPoint {
    /// The stroke's op is reserved and hasn't begun: its cut is taken as it begins
    /// ([`TileStore::journal_op_began`]). Its batches name the stroke before it meanwhile.
    Reserved,
    /// The next flush to cut holds it: one is queued behind the one in flight. Its stroke lands
    /// nothing until that cut is taken.
    Next,
    /// The flush whose cut this is holds it: its save point, once durable.
    Flush(CutId),
    /// Settled: a save point, the file's base, or unknown.
    Known(PenDown),
}

/// How the store moved onto a fresh working file ([`TileStore::rebase_journal`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Rebase {
    /// A save: the new file's base is the saved state.
    Save,
    /// A roll-over after the old file failed: the new file holds every edit in a save point,
    /// and, if the roll-over carried the open strokes, the writing one's pen-down cut and their
    /// journals.
    RollOver(Option<Carried>),
}

/// What a roll-over carried of the open strokes into the fresh working file.
///
/// Made by [`TileStore::carry_journal`]: the writing stroke, the save point there holding its
/// pen-down surface (the pen-down cut), and what the fresh file's batches hold of it and of each
/// stroke reserved behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Carried {
    /// The writing stroke's op.
    op: OpId,
    /// The offset of the pen-down cut's save point in the fresh file.
    pen_down: u64,
    /// What the fresh file's batches hold of the writing stroke; `None` if it has none (every
    /// dab placed had landed, and no stroke is reserved behind it).
    written: Option<Written>,
    /// What they hold of each stroke reserved behind it that has a batch there, in op order.
    behind: Vec<(OpId, Written)>,
}

impl Carried {
    /// What the fresh file's batches hold of the stroke reserved as `op`, if it was carried.
    fn behind(&self, op: OpId) -> Option<Written> {
        self.behind
            .iter()
            .find(|(carried, _)| *carried == op)
            .map(|&(_, written)| written)
    }
}

/// Why the open stroke's journal refused what it was given. Nothing changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JournalCountError {
    /// The dabs would run past the last index a `u32` counts.
    #[error("the stroke's dabs would run past the last index")]
    PastLastIndex,
    /// The landed count is below the one the journal had: dabs never unland.
    #[error("{landed} dabs landed, after {before} had")]
    LandedBack {
        /// The landed count given.
        landed: u32,
        /// The count the journal had.
        before: u32,
    },
    /// The landed count is past the dabs journalled.
    #[error("{landed} dabs landed of the {placed} journalled")]
    LandedPastPlaced {
        /// The landed count given.
        landed: u32,
        /// The dabs journalled.
        placed: u32,
    },
}

/// Why the store refused a call on a stroke's journal ([`TileStore::journal`],
/// [`TileStore::journal_landed`]). Nothing changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JournalRefused {
    /// No journal of the stroke is open: no stroke began, or it has ended. [`crate::Project`]
    /// answers [`crate::ProjectError::NoStroke`] for it.
    NotOpen,
    /// The open journal refused the count.
    Count(JournalCountError),
}

impl From<JournalCountError> for JournalRefused {
    fn from(refused: JournalCountError) -> Self {
        Self::Count(refused)
    }
}

/// What the working file's batches of one stroke hold, or will once a cut's are appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Written {
    /// The dabs placed by the last batch's last: the next batch's first index.
    placed: u32,
    /// The last batch's landed count.
    landed: u32,
}

/// One stroke's journal: its dabs not yet landed, the counts, what the working file holds, and
/// where its pen-down surface is.
///
/// Holds `landed ≤ placed` and `rows.len() == placed - landed`: the rows are exactly the dabs not
/// landed, from index `landed`.
#[derive(Debug)]
struct Journal {
    /// The stroke's op.
    op: OpId,
    /// How many dabs the stroke has placed.
    placed: u32,
    /// How many of them have landed.
    landed: u32,
    /// The dabs `landed..placed`, in order, as opaque rows.
    rows: VecDeque<[u8; DAB_BYTES]>,
    /// What the working file's batches hold; `None` while it holds none of this stroke.
    written: Option<Written>,
    /// The cut in flight, and what the file holds once it is appended; `None` while none is.
    in_flight: Option<(CutId, Written)>,
    /// Where the stroke's pen-down surface is.
    pen_down: PenDownPoint,
    /// What the stroke's batches name while `pen_down` isn't known: [`PenDown::After`] the stroke
    /// before, for a stroke begun behind it (or what a move onto a fresh file left of that, see
    /// [`TileStore::rebase_journal`]); `None` for a stroke begun as the writing one, whose batches
    /// wait for its pen-down to settle.
    before_cut: Option<PenDown>,
    /// The format of its rows, as the app set it when the stroke began (see
    /// [`crate::journal`]): every batch of it carries it.
    dab_format: u32,
    /// The store's latest write when the stroke began: a write since (of the stroke's, or of any
    /// other op) changed the surface it found. `None` for a stroke resumed after a restore
    /// ([`TileStore::resume_journal`]), whose surface the store's tiles may hold some of its dabs
    /// over already: a save never finds its pen-down surface in the saved state.
    begun_at: Option<Stamp>,
}

impl Journal {
    /// An empty journal of the stroke made as `op`, its pen-down surface at `pen_down`, its rows
    /// in `dab_format`, begun with the store's writes at `begun_at`.
    const fn new(op: OpId, pen_down: PenDownPoint, dab_format: u32, begun_at: Stamp) -> Self {
        Self {
            dab_format,
            begun_at: Some(begun_at),
            op,
            placed: 0,
            landed: 0,
            rows: VecDeque::new(),
            written: None,
            in_flight: None,
            pen_down,
            before_cut: None,
        }
    }

    /// Whether the stroke's op is reserved and hasn't begun: it lands nothing.
    const fn is_reserved(&self) -> bool {
        matches!(self.pen_down, PenDownPoint::Reserved)
    }

    /// Whether the stroke's pen-down waits on a flush to cut it or settle it.
    const fn waits_on_flush(&self) -> bool {
        matches!(self.pen_down, PenDownPoint::Next | PenDownPoint::Flush(_))
    }

    /// Adds the dabs `placed`, the next the stroke placed.
    ///
    /// # Errors
    ///
    /// [`JournalCountError::PastLastIndex`] if they would run past the last index a `u32` counts;
    /// none is added then.
    fn place(&mut self, placed: &[[u8; DAB_BYTES]]) -> Result<(), JournalCountError> {
        self.placed = u32::try_from(placed.len())
            .ok()
            .and_then(|count| self.placed.checked_add(count))
            .ok_or(JournalCountError::PastLastIndex)?;
        self.rows.extend(placed.iter().copied());
        Ok(())
    }

    /// Sets the landed count to `landed`, dropping the rows it passes.
    ///
    /// # Errors
    ///
    /// [`JournalCountError::LandedBack`] below the count it has;
    /// [`JournalCountError::LandedPastPlaced`] past the dabs placed. The journal is unchanged then.
    fn land(&mut self, landed: u32) -> Result<(), JournalCountError> {
        if landed < self.landed {
            return Err(JournalCountError::LandedBack {
                landed,
                before: self.landed,
            });
        }
        if landed > self.placed {
            return Err(JournalCountError::LandedPastPlaced {
                landed,
                placed: self.placed,
            });
        }
        let passed = usize::try_from(landed - self.landed).unwrap_or(usize::MAX);
        self.rows.drain(..passed.min(self.rows.len()));
        self.landed = landed;
        Ok(())
    }

    /// Whether some of the dabs placed haven't landed.
    const fn is_landing(&self) -> bool {
        self.landed < self.placed
    }

    /// What the file will hold once everything up to now is appended: the cut in flight's, else
    /// the batches written.
    fn reached(&self) -> Option<Written> {
        self.in_flight.map(|(_, reached)| reached).or(self.written)
    }

    /// The index of the first dab the next batch holds: the later of the dabs written (or in
    /// flight) and those landed.
    fn unwritten_from(&self) -> u32 {
        self.reached()
            .map_or(0, |reached| reached.placed)
            .max(self.landed)
    }

    /// The pen-down a batch names now, if one may be cut: the settled one, else what the batches
    /// name until the stroke's own cut settles (see the module notes).
    const fn settled(&self) -> Option<PenDown> {
        match self.pen_down {
            PenDownPoint::Known(pen_down) => Some(pen_down),
            PenDownPoint::Reserved | PenDownPoint::Next | PenDownPoint::Flush(_) => self.before_cut,
        }
    }

    /// Whether a batch is due (see the module notes): with its pen-down settled, a dab placed and
    /// neither written nor landed, or a landed count that moved since the last batch written.
    fn is_due(&self) -> bool {
        self.settled().is_some()
            && (self.unwritten_from() < self.placed
                || self
                    .reached()
                    .is_some_and(|reached| reached.landed != self.landed))
    }

    /// Whether the journal has anything left to write: a batch due, or a cut in flight. An ended
    /// one whose pen-down hasn't settled has neither: nothing of it was cut, and every dab of it
    /// counts landed, so the tiles hold them all.
    fn is_pending(&self) -> bool {
        self.is_due() || self.in_flight.is_some()
    }

    /// The journal's cut, if a batch is due: the batches of the dabs neither written nor in
    /// flight, at most [`MAX_JOURNAL_DABS`] each (one of no dabs if only the landed count moved),
    /// each with the landed count as it is now, the stroke's pen-down save point and its rows'
    /// format, and what the file holds once they are in it.
    ///
    /// This copies the rows not yet written, on the calling thread.
    ///
    /// # Errors
    ///
    /// [`Invariant::Journal`] if [`DabBatch::new`] refuses a batch, and
    /// [`Invariant::JournalRows`] if the rows run out before the dabs placed: never, while the
    /// journal holds its invariants.
    fn cut(&self) -> Result<Option<(Vec<DabBatch>, Written)>, Invariant> {
        let Some(pen_down) = self.settled().filter(|_| self.is_due()) else {
            return Ok(None);
        };
        self.batches_from(self.unwritten_from(), pen_down).map(Some)
    }

    /// The batches of the dabs `from..placed` (`from` at or past the landed count), at most
    /// [`MAX_JOURNAL_DABS`] each (one of no dabs if there are none), each with the landed count
    /// as it is now, `pen_down` and the rows' format, and what a file holds once they are in it.
    ///
    /// # Errors
    ///
    /// As [`Journal::cut`].
    fn batches_from(
        &self,
        from: u32,
        pen_down: PenDown,
    ) -> Result<(Vec<DabBatch>, Written), Invariant> {
        let skip = usize::try_from(from.saturating_sub(self.landed)).unwrap_or(usize::MAX);
        let mut unwritten = self.rows.iter().skip(skip).copied();
        let mut batches = Vec::new();
        let mut first = from;
        loop {
            let left = usize::try_from(self.placed - first).unwrap_or(usize::MAX);
            let mut rows = Vec::with_capacity(left.min(MAX_JOURNAL_DABS));
            rows.extend(unwritten.by_ref().take(MAX_JOURNAL_DABS));
            // Every pass takes rows, but a batch of none (the count alone) can only be the
            // first and the last: so the loop ends however the rows and counts disagree.
            let empty = rows.is_empty();
            let batch = DabBatch::new(self.op, first, self.landed, pen_down, rows)
                .map_err(Invariant::Journal)?
                .with_dab_format(self.dab_format);
            first = batch.placed();
            batches.push(batch);
            if first >= self.placed {
                break;
            }
            if empty {
                return Err(Invariant::JournalRows);
            }
        }
        let reached = Written {
            placed: first,
            landed: self.landed,
        };
        Ok((batches, reached))
    }
}

/// The store's journals: the open strokes', and those of strokes ended with their last count not
/// yet written (see the module notes).
#[derive(Debug, Default)]
pub(super) struct Journals {
    /// The open strokes' journals, in op order: the writing stroke's, from
    /// [`TileStore::begin_journal`] to its end, then those of the strokes reserved behind it,
    /// from [`TileStore::begin_journal_after`]. Only the first may be of an op begun.
    open: Vec<Journal>,
    /// Journals of strokes that have ended, each waiting for a flush to write its last batch, in
    /// the order they ended.
    ended: Vec<Journal>,
    /// The number of the next cut ([`CutId`]).
    next_cut: u64,
    /// The version of the rows' layout the app set ([`TileStore::set_dab_format`]), which a
    /// stroke's journal takes as it begins.
    dab_format: u32,
}

impl Journals {
    /// The open journal of the stroke made as `op`.
    ///
    /// # Errors
    ///
    /// [`JournalRefused::NotOpen`] if no journal of `op` is open.
    fn open(&mut self, op: OpId) -> Result<&mut Journal, JournalRefused> {
        self.open
            .iter_mut()
            .find(|journal| journal.op == op)
            .ok_or(JournalRefused::NotOpen)
    }

    /// The open journal of a stroke whose op has begun, if one is: the writing stroke's.
    fn begun(&self) -> Option<&Journal> {
        self.open.iter().find(|journal| !journal.is_reserved())
    }

    /// Opens `journal`, of an op newer than every open one's.
    fn push(&mut self, journal: Journal) {
        debug_assert!(
            self.open.last().is_none_or(|last| last.op < journal.op),
            "journals open in op order"
        );
        self.open.push(journal);
    }

    /// Closes the open journals `close` picks, counting every dab of them landed: each waits for
    /// its last batch if one is due (or will be once its pen-down settles) or a cut of it is in
    /// flight, else goes.
    fn close_where(&mut self, close: impl Fn(&Journal) -> bool) {
        let (closed, open): (Vec<Journal>, Vec<Journal>) =
            std::mem::take(&mut self.open).into_iter().partition(close);
        self.open = open;
        for mut journal in closed {
            journal.rows.clear();
            journal.landed = journal.placed;
            if journal.is_pending() {
                self.ended.push(journal);
            }
        }
    }

    /// Closes the journals of strokes whose op has begun: the writing stroke's.
    fn close_begun(&mut self) {
        self.close_where(|journal| !journal.is_reserved());
    }

    /// Every journal: the ended ones in the order they ended, then the open ones in op order.
    fn all(&self) -> impl Iterator<Item = &Journal> {
        self.ended.iter().chain(&self.open)
    }

    /// Every journal, to change.
    fn all_mut(&mut self) -> impl Iterator<Item = &mut Journal> {
        self.ended.iter_mut().chain(&mut self.open)
    }

    /// Ended journals that have nothing left to write go.
    fn forget_written(&mut self) {
        self.ended.retain(Journal::is_pending);
    }
}

impl TileStore {
    /// The stroke made as `op` begins, its pen-down surface at `pen_down` (see the module
    /// notes): opens its journal, closing the writing stroke's if it is still open (see
    /// [`TileStore::end_journal`]). No op may be reserved: it would begin first.
    pub(crate) fn begin_journal(&mut self, op: OpId, pen_down: PenDownPoint) {
        self.journals.close_begun();
        debug_assert!(
            self.journals.open.is_empty(),
            "no stroke is reserved behind a stroke that begins"
        );
        let dab_format = self.journals.dab_format;
        let journal = Journal::new(op, pen_down, dab_format, self.last_write);
        self.journals.push(journal);
    }

    /// The op of the newest open stroke, which a stroke begun now would follow; `None` with none
    /// open.
    pub(crate) fn newest_open_stroke(&self) -> Option<OpId> {
        self.journals.open.last().map(|journal| journal.op)
    }

    /// The stroke made as `op`, an op just reserved ([`TileStore::reserve_op`]), begins behind
    /// the newest open stroke, which still lands: opens its journal, its batches naming
    /// [`PenDown::After`] that stroke until its own pen-down cut, taken as its op begins
    /// ([`TileStore::journal_op_began`]), settles (see the module notes).
    ///
    /// The stroke before is of `op`'s site and an earlier op, as a batch's [`PenDown::After`]
    /// must name (a debug assertion says so): every op is the store's own site's, and reserved
    /// after every open one. Edits merged from several sites would need an order of their own
    /// here, as the restore's head does (see `crate::project::resume`). With no stroke open,
    /// which the caller rules out first, its batches wait for its own cut instead.
    pub(crate) fn begin_journal_after(&mut self, op: OpId) {
        let after = self.newest_open_stroke();
        debug_assert!(
            after.is_some_and(|after| after.site() == op.site() && after < op),
            "a stroke begins behind an open stroke of its site, before it: {after:?}, {op:?}"
        );
        let dab_format = self.journals.dab_format;
        let mut journal = Journal::new(op, PenDownPoint::Reserved, dab_format, self.last_write);
        journal.before_cut = after.map(PenDown::After);
        self.journals.push(journal);
    }

    /// The reserved op `op` began ([`TileStore::begin_reserved_op`]) and its stroke's pen-down cut
    /// was taken: its pen-down surface is at `pen_down`, which its batches name once it settles;
    /// until then they go on naming the stroke before. A roll-over while the cut was taken, which
    /// carried the stroke and so settled its pen-down already, keeps that. Its writes from now on
    /// are its own, which a save in its middle looks for, as for any stroke
    /// ([`TileStore::rebase_journal`]). Nothing for an op whose journal isn't open.
    pub(crate) fn journal_op_began(&mut self, op: OpId, pen_down: PenDownPoint) {
        let last_write = self.last_write;
        if let Ok(journal) = self.journals.open(op) {
            journal.begun_at = Some(last_write);
            if journal.is_reserved() {
                journal.pen_down = pen_down;
            }
        }
    }

    /// The op of the writing stroke, if its journal is open: the one stroke whose op has begun.
    pub(super) fn begun_stroke(&self) -> Option<OpId> {
        self.journals.begun().map(|journal| journal.op)
    }

    /// The stroke made as `op`, restored after a crash, lands on (see
    /// [`crate::Project::resume_stroke`]): opens its journal as the working file's batches left
    /// it, closing any still open. `landed` of its dabs had landed, and `rows` are the rest, which
    /// the batches hold already, so they are never written again: the next batch carries the
    /// landed count alone, once it moves. Its pen-down surface stays where the batches name, and
    /// its rows keep their format, `dab_format`, whatever the app has set since.
    ///
    /// # Errors
    ///
    /// [`JournalCountError::PastLastIndex`] if the rows run past the last index a `u32` counts;
    /// nothing is opened then.
    ///
    /// The rows are copied: the stroke keeps its own until the resume is done.
    pub(crate) fn resume_journal(
        &mut self,
        op: OpId,
        landed: u32,
        rows: &[[u8; DAB_BYTES]],
        pen_down: PenDown,
        dab_format: u32,
    ) -> Result<(), JournalCountError> {
        let placed = u32::try_from(rows.len())
            .ok()
            .and_then(|count| landed.checked_add(count))
            .ok_or(JournalCountError::PastLastIndex)?;
        self.journals.close_begun();
        let written = Written { placed, landed };
        self.journals.push(Journal {
            op,
            placed,
            landed,
            rows: rows.iter().copied().collect(),
            written: Some(written),
            in_flight: None,
            pen_down: PenDownPoint::Known(pen_down),
            before_cut: None,
            dab_format,
            begun_at: None,
        });
        Ok(())
    }

    /// The stroke made as `op`, restored after a crash, its batches naming `earlier`, the stroke
    /// it began behind ([`PenDown::After`]), lands on what that one left, which has ended (see
    /// [`crate::Project::resume_stroke`]): opens its journal as a reserved stroke's, newer than
    /// every open one. None of its dabs has landed, and `rows` are all of them, which the
    /// batches hold already, so they are never written again. Its batches go on naming
    /// `earlier` until the cut its op takes as it begins ([`TileStore::journal_op_began`])
    /// settles; until then it lands nothing. Its rows keep their format, `dab_format`.
    ///
    /// # Errors
    ///
    /// [`JournalCountError::PastLastIndex`] if the rows run past the last index a `u32` counts;
    /// nothing is opened then.
    pub(crate) fn resume_journal_after(
        &mut self,
        op: OpId,
        rows: &[[u8; DAB_BYTES]],
        earlier: OpId,
        dab_format: u32,
    ) -> Result<(), JournalCountError> {
        let placed = u32::try_from(rows.len()).or(Err(JournalCountError::PastLastIndex))?;
        let written = Written { placed, landed: 0 };
        self.journals.push(Journal {
            op,
            placed,
            landed: 0,
            rows: rows.iter().copied().collect(),
            written: Some(written),
            in_flight: None,
            pen_down: PenDownPoint::Reserved,
            before_cut: Some(PenDown::After(earlier)),
            dab_format,
            begun_at: None,
        });
        Ok(())
    }

    /// Drops the open journal of `op` without ending it: no batch is written of it, as if it had
    /// never opened. For a resume that failed after [`TileStore::resume_journal`], which must
    /// leave the file saying nothing it didn't (ending it would write a batch counting every dab
    /// landed).
    pub(crate) fn discard_journal(&mut self, op: OpId) {
        self.journals.open.retain(|journal| journal.op != op);
    }

    /// The stroke made as `op`, restored still landing and never resumed, has ended: its dabs
    /// not landed are dropped. Queues the batch that says so for the next flush, of no dabs, its
    /// landed count `placed`, its dabs placed (see the module notes, "When the stroke ends"), so
    /// the file itself says the stroke is closed, and a second crash's restore never offers it
    /// over the edits that follow. `landed` is the count its last batch in the file has, its
    /// pen-down and its rows' format those the batches name.
    pub(crate) fn close_restored_journal(
        &mut self,
        op: OpId,
        landed: u32,
        placed: u32,
        pen_down: PenDown,
        dab_format: u32,
    ) {
        let journal = Journal {
            op,
            placed,
            landed: placed,
            rows: VecDeque::new(),
            written: Some(Written { placed, landed }),
            in_flight: None,
            pen_down: PenDownPoint::Known(pen_down),
            before_cut: None,
            dab_format,
            begun_at: None,
        };
        if journal.is_pending() {
            self.journals.ended.push(journal);
        }
    }

    /// The rows the app journals are laid out in dab format `dab_format`, the version of
    /// `stroke::Dab`'s layout, which the store never reads: every stroke begun from now on takes
    /// it, and every batch of the stroke carries it (see [`crate::journal`]). It is 0, unset,
    /// until set, and stays across a move onto a new working file.
    pub(crate) const fn set_dab_format(&mut self, dab_format: u32) {
        self.journals.dab_format = dab_format;
    }

    /// Adds `placed`, the next dabs the stroke made as `op` placed, to its journal: kept for the
    /// next flush to write (see the module notes).
    ///
    /// # Errors
    ///
    /// [`JournalRefused::NotOpen`] if no journal of `op` is open; [`JournalRefused::Count`] with
    /// [`JournalCountError::PastLastIndex`] if the stroke's dabs would run past the last index a
    /// `u32` counts. None is added then.
    pub(crate) fn journal(
        &mut self,
        op: OpId,
        placed: &[[u8; DAB_BYTES]],
    ) -> Result<(), JournalRefused> {
        Ok(self.journals.open(op)?.place(placed)?)
    }

    /// `landed` of the dabs the stroke made as `op` placed have landed: passed with each landing,
    /// on the owner thread, with no cut between the landing's edits and this count, in either
    /// order, so a flush's cut always pairs the store's tiles with the count that made them (see
    /// the module notes). Only the writing stroke lands: a reserved one's count stays 0.
    ///
    /// # Errors
    ///
    /// [`JournalRefused::NotOpen`] if no journal of `op` is open, or `op` is reserved and hasn't
    /// begun; [`JournalRefused::Count`] with [`JournalCountError::LandedBack`] below the count it
    /// had, or [`JournalCountError::LandedPastPlaced`] past the dabs journalled. Nothing changes
    /// then.
    pub(crate) fn journal_landed(&mut self, op: OpId, landed: u32) -> Result<(), JournalRefused> {
        let journal = self.journals.open(op)?;
        if journal.is_reserved() {
            return Err(JournalRefused::NotOpen);
        }
        Ok(journal.land(landed)?)
    }

    /// The writing stroke ended: its journal counts every dab landed (those it dropped never
    /// land), and goes once that count is written, if it isn't yet (see the module notes). The
    /// journals of the strokes reserved behind it stay open.
    pub(crate) fn end_journal(&mut self) {
        self.journals.close_begun();
    }

    /// The reserved ops `dropped` were let go before they began
    /// ([`TileStore::drop_reserved_ops_from`]): their strokes' journals close, every dab of them
    /// counted landed, as an ended stroke's are, so a restore never lands one.
    pub(super) fn end_dropped_journals(&mut self, dropped: &[OpId]) {
        self.journals
            .close_where(|journal| journal.is_reserved() && dropped.contains(&journal.op));
    }

    /// Whether an open stroke has dabs journalled and not landed, the writing one or one reserved
    /// behind it: a save must wait for them.
    pub(crate) fn is_landing(&self) -> bool {
        self.journals.open.iter().any(Journal::is_landing)
    }

    /// Whether the writing stroke may land, edit the store: its pen-down flush has taken its cut
    /// (see the module notes). True with no stroke open; a reserved stroke lands nothing until
    /// its op begins.
    pub(crate) fn may_land(&self) -> bool {
        self.journals
            .begun()
            .is_none_or(|journal| journal.pen_down != PenDownPoint::Next)
    }

    /// Whether the stroke made as `op` may land: its op has begun and its pen-down flush has
    /// taken its cut. False for a stroke reserved, or one whose journal isn't open.
    pub(crate) fn stroke_may_land(&self, op: OpId) -> bool {
        self.journals
            .begun()
            .is_some_and(|journal| journal.op == op && journal.pen_down != PenDownPoint::Next)
    }

    /// Whether a restore could go on with the open strokes, as far as their pen-down surfaces
    /// go: true with no stroke open, or with each one's pen-down in a save point, after the
    /// stroke before it or still to settle; false once one is unknown, which it stays (see
    /// [`crate::Project::open_stroke_restorable`]).
    pub(crate) fn open_stroke_restorable(&self) -> bool {
        self.journals
            .open
            .iter()
            .all(|journal| journal.settled() != Some(PenDown::Unknown))
    }

    /// Whether the open stroke made as `op` waits for its pen-down flush's cut before it may land
    /// ([`TileStore::may_land`]): an edit of `op` now would be a bug.
    pub(super) fn waits_to_land(&self, op: OpId) -> bool {
        self.journals
            .open
            .iter()
            .any(|journal| journal.op == op && journal.pen_down == PenDownPoint::Next)
    }

    /// Whether a flush now would write a batch.
    pub(crate) fn has_journal_work(&self) -> bool {
        self.journals.all().any(Journal::is_due)
    }

    /// The journals' cut, for the flush taking its snapshot now: the cut's name, and every batch
    /// due, the ended strokes' first. Each journal cut holds the cut in flight until
    /// [`TileStore::append_journal`] appends it or [`TileStore::drop_journal_cut`] drops it. A
    /// pen-down waiting for the next flush is this one's from now on.
    ///
    /// A cut left over from a flush that never ended (its writer gone) is dropped first, and a
    /// pen-down waiting on such a flush is unknown: it settles no more.
    ///
    /// The cost is a copy of the rows not yet written, on the owner thread.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] with [`Invariant::Journal`] or [`Invariant::JournalRows`] if a
    /// batch can't be made: a bug. No cut is in flight then.
    pub(crate) fn cut_journal(&mut self) -> Result<(CutId, Vec<DabBatch>), CacheError> {
        self.drop_journal_cut();
        for journal in self.journals.all_mut() {
            if matches!(journal.pen_down, PenDownPoint::Flush(_)) {
                journal.pen_down = PenDownPoint::Known(PenDown::Unknown);
            }
        }
        let id = CutId(self.journals.next_cut);
        self.journals.next_cut = self.journals.next_cut.saturating_add(1);
        let mut batches = Vec::new();
        let mut refused = None;
        for journal in self.journals.all_mut() {
            if journal.pen_down == PenDownPoint::Next {
                journal.pen_down = PenDownPoint::Flush(id);
            }
            match journal.cut() {
                Ok(Some((cut, reached))) => {
                    batches.extend(cut);
                    journal.in_flight = Some((id, reached));
                }
                Ok(None) => {}
                Err(e) => {
                    refused = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = refused {
            // The journals cut before it hold cuts no flush will append.
            self.drop_journal_cut();
            return Err(e.into());
        }
        Ok((id, batches))
    }

    /// Appends `framed`, the batches of the cut `id` as the writer thread framed them, to the
    /// working file with one write, then notes each journal written up to that cut.
    ///
    /// Noting them only once the write is done is what keeps a batch from being written twice:
    /// if it fails, the cut is dropped and the journals are as they were, so the next flush writes
    /// the same dabs. An ended journal whose last count is in goes.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if the append fails.
    pub(crate) fn append_journal(
        &mut self,
        id: CutId,
        framed: &[Vec<u8>],
    ) -> Result<(), CacheError> {
        if let Err(e) = self.file.append_batches(framed) {
            self.drop_journal_cut();
            return Err(CacheError::WriteBack(e));
        }
        for journal in self.journals.all_mut() {
            if let Some((cut, reached)) = journal.in_flight
                && cut == id
            {
                journal.in_flight = None;
                journal.written = Some(reached);
            }
        }
        self.journals.forget_written();
        Ok(())
    }

    /// Drops the cut in flight, if any: its flush was abandoned before its batches were appended,
    /// so the next flush writes them again. An ended journal that has nothing left to write goes.
    pub(crate) fn drop_journal_cut(&mut self) {
        for journal in self.journals.all_mut() {
            journal.in_flight = None;
        }
        self.journals.forget_written();
    }

    /// The flush whose cut is `id` settled the pen-down surface it holds, which is where
    /// `pen_down` says: its save point, made durable, or (a flush that found nothing to write)
    /// the working file's latest. Every journal waiting on it learns it.
    pub(crate) fn pen_down_settled(&mut self, id: CutId, pen_down: PenDown) {
        for journal in self.journals.all_mut() {
            if journal.pen_down == PenDownPoint::Flush(id) {
                journal.pen_down = PenDownPoint::Known(pen_down);
            }
        }
        self.journals.forget_written();
    }

    /// Whether a stroke's pen-down surface waits on the flush whose cut is `id`: its save point
    /// must then hold every tile at the cut (see `crate::recovery`).
    pub(crate) fn pen_down_on(&self, id: CutId) -> bool {
        self.journals
            .all()
            .any(|journal| journal.pen_down == PenDownPoint::Flush(id))
    }

    /// The flush in flight was abandoned, and any queued behind it, or a writer is gone: a
    /// journal waiting on one of them for its pen-down surface finds it in no save point, and its
    /// stroke may land.
    pub(crate) fn pen_downs_abandoned(&mut self) {
        for journal in self.journals.all_mut() {
            if journal.waits_on_flush() {
                journal.pen_down = PenDownPoint::Known(PenDown::Unknown);
            }
        }
        self.journals.forget_written();
    }

    /// The writing stroke's pen-down save point, if its journal is open: for tests.
    #[cfg(test)]
    pub(crate) fn open_pen_down(&self) -> Option<PenDownPoint> {
        self.journals.begun().map(|journal| journal.pen_down)
    }

    /// Test hook: makes the working file's next save point fail before anything is written, as
    /// a full disk would.
    #[cfg(test)]
    pub(crate) const fn fail_next_save_point(&mut self) {
        self.file.fail_next_save();
    }

    /// The store moved onto a fresh working file, by `how`, which holds no batch: the ended
    /// journals go (no batch in the new file is behind its tiles), and the open ones write their
    /// dabs not landed again, from their landed counts, with the next flush.
    ///
    /// The writing stroke's pen-down surface:
    ///
    /// - **After a save**, if the store took no write since the stroke began (none of the stroke's,
    ///   none of any other op's), the saved state is its surface, and the new file's base:
    ///   [`PenDown::Base`]. Otherwise the new file holds no save point of the surface it began on,
    ///   and its pen-down is unknown from then on (its journal can't be restored from the new
    ///   file). A save refuses while dabs are landing, and the app saves only after the pen lifts,
    ///   so the app never meets that; a store-level save in the middle of a stroke whose dabs have
    ///   all landed leaves the rest of the stroke's dabs unrestorable.
    /// - **After a roll-over** that carried the stroke ([`TileStore::carry_journal`]), the pen-down
    ///   cut in the new file, whose batches hold what the carry wrote: the journal goes on from
    ///   there, its next batch the dabs placed since, or its landed count alone. After one that
    ///   couldn't, unknown.
    ///
    /// A reserved stroke's, which the new file holds no save point of: it goes on naming the
    /// stroke before it while that one's journal is open, whatever became of that one's pen-down
    /// (the surface that stroke leaves is the same), and a roll-over that carried it holds its
    /// batches already. Else that stroke has ended, and nothing has written since (ops edit in
    /// order, and the reserved one hasn't begun): after a save, the saved state, the new file's
    /// base, is the surface it left; after a roll-over, no save point the store knows of is named
    /// for it, so it is unknown until its own cut settles.
    pub(super) fn rebase_journal(&mut self, how: &Rebase) {
        self.journals.ended.clear();
        let last_write = self.last_write;
        let open_ops: Vec<OpId> = self
            .journals
            .open
            .iter()
            .map(|journal| journal.op)
            .collect();
        for journal in &mut self.journals.open {
            journal.in_flight = None;
            match how {
                Rebase::RollOver(Some(carried)) if carried.op == journal.op => {
                    journal.pen_down = PenDownPoint::Known(PenDown::At(carried.pen_down));
                    journal.written = carried.written;
                }
                _ if journal.is_reserved() => {
                    let before_open = matches!(
                        journal.before_cut,
                        Some(PenDown::After(before)) if open_ops.contains(&before)
                    );
                    if !before_open {
                        journal.before_cut = Some(match how {
                            Rebase::Save => PenDown::Base,
                            Rebase::RollOver(_) => PenDown::Unknown,
                        });
                    }
                    journal.written = match how {
                        Rebase::RollOver(Some(carried)) => carried.behind(journal.op),
                        Rebase::RollOver(None) | Rebase::Save => None,
                    };
                }
                Rebase::Save => {
                    let found = journal
                        .begun_at
                        .is_some_and(|begun_at| last_write <= begun_at);
                    let pen_down = if found {
                        PenDown::Base
                    } else {
                        PenDown::Unknown
                    };
                    journal.pen_down = PenDownPoint::Known(pen_down);
                    journal.written = None;
                }
                Rebase::RollOver(_) => {
                    journal.pen_down = PenDownPoint::Known(PenDown::Unknown);
                    journal.written = None;
                }
            }
        }
    }

    /// Whether a journal of `op` is open: its stroke hasn't ended.
    pub(crate) fn journal_open_for(&self, op: OpId) -> bool {
        self.journals.open.iter().any(|journal| journal.op == op)
    }

    /// The op of the open stroke whose pen-down cut a roll-over writes into its fresh file, if one
    /// is open and the store's op is still its own (see [`TileStore::fill_roll_over`]): the
    /// writing stroke. The strokes reserved behind it are carried with it, or not at all (see
    /// the module notes).
    ///
    /// The stroke's edits are then exactly the writes of its op since it began (strokes don't
    /// overlap, see `crate::recovery`), which is how the roll-over tells the tiles it changed
    /// from those it found. A save in its middle moves its later edits to a new op: then nothing
    /// tells them apart, and the stroke isn't carried.
    pub(super) fn journal_to_carry(&self) -> Option<OpId> {
        let op = self.op?;
        self.journal_open_for(op).then_some(op)
    }

    /// The open strokes' journals as a roll-over carries them into its fresh file, whose pen-down
    /// cut is the save point at `pen_down` there, in op order (see the module notes), and what
    /// [`TileStore::rebase_journal`] sets the journals to once the file is swapped in; `None`
    /// with no writing stroke to carry ([`TileStore::journal_to_carry`]). Nothing in the store
    /// changes.
    ///
    /// The writing stroke's batches hold every dab not landed, from the landed count, naming that
    /// save point; if every dab placed has landed, none, or, with a stroke reserved behind it, one
    /// of its count alone, which closes it in the fresh file, so the stroke behind it finds its
    /// surface there. Each stroke reserved behind it that has placed dabs gets batches of every one
    /// of them, naming what its batches name now (the stroke before it); one whose batches can't
    /// be cut yet (its own cut, from a resume, still to settle with nothing before it named) gets
    /// none, and its next flush writes them.
    ///
    /// The cost is a copy of the rows not landed, on the calling thread.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] with [`Invariant::Journal`] or [`Invariant::JournalRows`] if a
    /// batch can't be made: a bug.
    pub(super) fn carry_journal(
        &self,
        pen_down: u64,
    ) -> Result<Option<(Vec<DabBatch>, Carried)>, CacheError> {
        let Some(op) = self.journal_to_carry() else {
            return Ok(None);
        };
        let open = &self.journals.open;
        let Some(writing) = open.iter().find(|journal| journal.op == op) else {
            return Ok(None);
        };
        let behind_it: Vec<&Journal> = open.iter().filter(|journal| journal.op > op).collect();
        let (mut batches, written) = if writing.is_landing() || !behind_it.is_empty() {
            let (batches, reached) = writing.batches_from(writing.landed, PenDown::At(pen_down))?;
            (batches, Some(reached))
        } else {
            (Vec::new(), None)
        };
        let mut behind = Vec::with_capacity(behind_it.len());
        for journal in behind_it {
            let Some(named) = journal.settled().filter(|_| journal.placed > 0) else {
                continue;
            };
            let (cut, reached) = journal.batches_from(journal.landed, named)?;
            batches.extend(cut);
            behind.push((journal.op, reached));
        }
        let carried = Carried {
            op,
            pen_down,
            written,
            behind,
        };
        Ok(Some((batches, carried)))
    }
}
