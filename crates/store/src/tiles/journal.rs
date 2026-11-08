//! The open stroke's dab journal on the store (see [`crate::journal`]): the dabs it placed and
//! hasn't landed, how many have landed, and what the working file holds of them so far.
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
//! goes on after them.
//!
//! A batch is due while a dab placed isn't written, or the landed count moved since the last
//! batch: then a batch of no dabs carries the count alone. A stroke none of whose batches is in
//! the file, or in flight, and whose dabs have all landed is due none: its tiles hold every dab.
//!
//! # When the stroke ends
//!
//! The stroke's end ([`TileStore::end_journal`]) closes its journal, counting every dab landed:
//! dabs a stroke dropped (after an error) must never land on a restore. If that count isn't
//! written yet, or a cut of the journal is in flight, the journal waits among the ended ones for
//! the flushes that write its last count, so the file never ends a stroke on a batch whose count is
//! behind its tiles; then it goes. The next stroke's journal is a new one meanwhile
//! ([`TileStore::begin_journal`]), and a flush writes the ended strokes' batches before its own.
//!
//! # Memory and cost
//!
//! A journal keeps the rows of the dabs not landed, [`DAB_BYTES`] each: the landing's backlog,
//! about 1.4 MB at the forced landing's 16,384 dabs. Every row a landed count passes goes at once.
//! A move onto a fresh working file (a save's rebase, or a roll-over) takes none of the batches
//! with it ([`TileStore::rebase_journal`]), so the rows a later flush must write again are still
//! at hand.
//!
//! The owner thread's cost per flush is the cut: a copy of the rows not yet written, about a
//! second of placed dabs. Encoding, compression and syncing happen on the writer thread.

use std::collections::VecDeque;

use thiserror::Error;

use super::TileStore;
use super::error::{CacheError, Invariant};
use crate::OpId;
use crate::journal::{DAB_BYTES, DabBatch, MAX_JOURNAL_DABS};

/// Why the open stroke's journal refused what it was given. Nothing changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JournalCountError {
    /// No journal of the stroke is open: no stroke began, or it has ended. [`crate::Project`]
    /// never meets it: it answers [`crate::ProjectError::NoStroke`] first, and opens a journal
    /// with every stroke it begins.
    #[error("no journal of the stroke is open")]
    NotOpen,
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

/// What the working file's batches of one stroke hold, or will once a cut's are appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Written {
    /// The dabs placed by the last batch's last: the next batch's first index.
    placed: u32,
    /// The last batch's landed count.
    landed: u32,
}

/// One stroke's journal: its dabs not yet landed, the counts, and what the working file holds.
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
    /// What the file holds once the cut in flight is appended; `None` while none is.
    in_flight: Option<Written>,
}

impl Journal {
    /// An empty journal of the stroke made as `op`.
    const fn new(op: OpId) -> Self {
        Self {
            op,
            placed: 0,
            landed: 0,
            rows: VecDeque::new(),
            written: None,
            in_flight: None,
        }
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
    const fn reached(&self) -> Option<Written> {
        match self.in_flight {
            Some(in_flight) => Some(in_flight),
            None => self.written,
        }
    }

    /// The index of the first dab the next batch holds: the later of the dabs written (or in
    /// flight) and those landed.
    fn unwritten_from(&self) -> u32 {
        self.reached()
            .map_or(0, |reached| reached.placed)
            .max(self.landed)
    }

    /// Whether a batch is due (see the module notes): a dab placed and neither written nor
    /// landed, or a landed count that moved since the last batch written.
    fn is_due(&self) -> bool {
        self.unwritten_from() < self.placed
            || self
                .reached()
                .is_some_and(|reached| reached.landed != self.landed)
    }

    /// Whether the journal has anything left to write: a batch due, or a cut in flight.
    fn is_pending(&self) -> bool {
        self.is_due() || self.in_flight.is_some()
    }

    /// The journal's cut, if a batch is due: the batches of the dabs not yet written, at most
    /// [`MAX_JOURNAL_DABS`] each (one of no dabs if only the landed count moved), each with the
    /// landed count as it is now, and what the file holds once they are in it.
    ///
    /// This copies the rows not yet written, on the calling thread.
    ///
    /// # Errors
    ///
    /// [`Invariant::Journal`] if [`DabBatch::new`] refuses a batch, and
    /// [`Invariant::JournalRows`] if the rows run out before the dabs placed: never, while the
    /// journal holds its invariants.
    fn cut(&self) -> Result<Option<(Vec<DabBatch>, Written)>, Invariant> {
        if !self.is_due() {
            return Ok(None);
        }
        let from = self.unwritten_from();
        let skip = usize::try_from(from - self.landed).unwrap_or(usize::MAX);
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
            let batch =
                DabBatch::new(self.op, first, self.landed, rows).map_err(Invariant::Journal)?;
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
        Ok(Some((batches, reached)))
    }
}

/// The store's journals: the open stroke's, and those of strokes ended with their last count not
/// yet written (see the module notes).
#[derive(Debug, Default)]
pub(super) struct Journals {
    /// The open stroke's journal, from [`TileStore::begin_journal`] to its end.
    open: Option<Journal>,
    /// Journals of strokes that have ended, each waiting for a flush to write its last batch, in
    /// the order they ended.
    ended: Vec<Journal>,
}

impl Journals {
    /// The open journal of the stroke made as `op`.
    ///
    /// # Errors
    ///
    /// [`JournalCountError::NotOpen`] if no journal of `op` is open.
    fn open(&mut self, op: OpId) -> Result<&mut Journal, JournalCountError> {
        self.open
            .as_mut()
            .filter(|journal| journal.op == op)
            .ok_or(JournalCountError::NotOpen)
    }

    /// Closes the open journal, if any, counting every dab landed: it waits for its last batch
    /// if one is due or a cut of it is in flight, else goes.
    fn close(&mut self) {
        if let Some(mut journal) = self.open.take() {
            journal.rows.clear();
            journal.landed = journal.placed;
            if journal.is_pending() {
                self.ended.push(journal);
            }
        }
    }

    /// Every journal: the ended ones in the order they ended, then the open one.
    fn all(&self) -> impl Iterator<Item = &Journal> {
        self.ended.iter().chain(&self.open)
    }

    /// Every journal, to change.
    fn all_mut(&mut self) -> impl Iterator<Item = &mut Journal> {
        self.ended.iter_mut().chain(&mut self.open)
    }
}

impl TileStore {
    /// The stroke made as `op` begins: opens its journal, closing any still open (see
    /// [`TileStore::end_journal`]).
    pub(crate) fn begin_journal(&mut self, op: OpId) {
        self.journals.close();
        self.journals.open = Some(Journal::new(op));
    }

    /// Adds `placed`, the next dabs the stroke made as `op` placed, to its journal: kept for the
    /// next flush to write (see the module notes).
    ///
    /// # Errors
    ///
    /// [`JournalCountError::NotOpen`] if no journal of `op` is open;
    /// [`JournalCountError::PastLastIndex`] if the stroke's dabs would run past the last index a
    /// `u32` counts. None is added then.
    pub(crate) fn journal(
        &mut self,
        op: OpId,
        placed: &[[u8; DAB_BYTES]],
    ) -> Result<(), JournalCountError> {
        self.journals.open(op)?.place(placed)
    }

    /// `landed` of the dabs the stroke made as `op` placed have landed: passed with each landing,
    /// on the owner thread, with no cut between the landing's edits and this count, in either
    /// order, so a flush's cut always pairs the store's tiles with the count that made them (see
    /// the module notes).
    ///
    /// # Errors
    ///
    /// [`JournalCountError::NotOpen`] if no journal of `op` is open;
    /// [`JournalCountError::LandedBack`] below the count it had;
    /// [`JournalCountError::LandedPastPlaced`] past the dabs journalled. Nothing changes then.
    pub(crate) fn journal_landed(
        &mut self,
        op: OpId,
        landed: u32,
    ) -> Result<(), JournalCountError> {
        self.journals.open(op)?.land(landed)
    }

    /// The open stroke ended: its journal counts every dab landed (those it dropped never land),
    /// and goes once that count is written, if it isn't yet (see the module notes).
    pub(crate) fn end_journal(&mut self) {
        self.journals.close();
    }

    /// Whether the open stroke has dabs journalled and not landed: a save must wait for them.
    pub(crate) fn is_landing(&self) -> bool {
        self.journals.open.as_ref().is_some_and(Journal::is_landing)
    }

    /// Whether a flush now would write a batch.
    pub(crate) fn has_journal_work(&self) -> bool {
        self.journals.all().any(Journal::is_due)
    }

    /// Whether a cut of the journals is in flight: a flush that carries batches.
    pub(crate) fn has_journal_cut(&self) -> bool {
        self.journals
            .all()
            .any(|journal| journal.in_flight.is_some())
    }

    /// The journals' cut, for the flush taking its snapshot now: every batch due, the ended
    /// strokes' first. Each journal cut holds the cut in flight until
    /// [`TileStore::append_journal`] appends it or [`TileStore::drop_journal_cut`] drops it; a
    /// cut left over from a flush that never ended (its writer gone) is dropped first.
    ///
    /// The cost is a copy of the rows not yet written, on the owner thread.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] with [`Invariant::Journal`] or [`Invariant::JournalRows`] if a
    /// batch can't be made: a bug. No cut is in flight then.
    pub(crate) fn cut_journal(&mut self) -> Result<Vec<DabBatch>, CacheError> {
        self.drop_journal_cut();
        let mut batches = Vec::new();
        let mut refused = None;
        for journal in self.journals.all_mut() {
            match journal.cut() {
                Ok(Some((cut, reached))) => {
                    batches.extend(cut);
                    journal.in_flight = Some(reached);
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
        Ok(batches)
    }

    /// Appends `framed`, the batches of the cut in flight as the writer thread framed them, to the
    /// working file with one write, then notes each journal written up to its cut.
    ///
    /// Noting them only once the write is done is what keeps a batch from being written twice:
    /// if it fails, the cut is dropped and the journals are as they were, so the next flush writes
    /// the same dabs. An ended journal whose last count is in goes.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if the append fails.
    pub(crate) fn append_journal(&mut self, framed: &[Vec<u8>]) -> Result<(), CacheError> {
        if let Err(e) = self.file.append_batches(framed) {
            self.drop_journal_cut();
            return Err(CacheError::WriteBack(e));
        }
        for journal in self.journals.all_mut() {
            if let Some(reached) = journal.in_flight.take() {
                journal.written = Some(reached);
            }
        }
        self.journals.ended.retain(Journal::is_pending);
        Ok(())
    }

    /// Drops the cut in flight, if any: its flush was abandoned before its batches were appended,
    /// so the next flush writes them again. An ended journal that has nothing left to write goes.
    pub(crate) fn drop_journal_cut(&mut self) {
        for journal in self.journals.all_mut() {
            journal.in_flight = None;
        }
        self.journals.ended.retain(Journal::is_pending);
    }

    /// Test hook: makes the working file's next save point fail before anything is written, as
    /// a full disk would.
    #[cfg(test)]
    pub(crate) const fn fail_next_save_point(&mut self) {
        self.file.fail_next_save();
    }

    /// The store moved onto a fresh working file (a save's rebase, or a roll-over), which holds
    /// no batch: the ended journals go (no batch in the new file is behind its tiles), and the
    /// open one writes its dabs not landed again, from its landed count, with the next flush.
    pub(super) fn rebase_journal(&mut self) {
        self.journals.ended.clear();
        if let Some(journal) = &mut self.journals.open {
            journal.written = None;
            journal.in_flight = None;
        }
    }
}
