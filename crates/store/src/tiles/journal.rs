//! The open stroke's dab journal on the store (see [`crate::journal`]): the dabs it placed and
//! hasn't landed, how many have landed, and what the working file holds of them so far.
//!
//! # Through a flush
//!
//! The recovery writer takes the journal's **cut** as it takes its tile snapshot, on the owner
//! thread, at the same moment ([`TileStore::cut_journal`]): a [`DabBatch`] of the dabs placed and
//! not yet written, from whichever is later of the last written and the last landed (a dab landed
//! is never written: the tiles hold it), with the landed count as it is then. The landed count is
//! moved by [`TileStore::journal_landed`] as the dabs' edits are made, so at any cut it is the
//! count of dabs whose edits the snapshot copies, and the batch's count describes exactly the
//! tiles that flush's save point names.
//!
//! The batch is encoded on the writer thread, with the tiles, and appended with the flush's
//! records ([`TileStore::append_journal`]), before the tile sync, so the save point that names the
//! cut's tiles holds its batch. Only then does the journal note it written: batches are never
//! deduplicated, so a flush abandoned before its append (a failed encode, an append that fails, a
//! flush overtaken by a rebase) leaves the journal as it was, and the next flush writes the same
//! dabs, with the landed count it finds, once, never overlapping a batch in the file.
//!
//! A batch is due while a dab placed isn't written, or the landed count moved since the last
//! batch: then a batch of no dabs carries the count alone. A stroke none of whose batches is in
//! the file yet and whose dabs have all landed is due none: its tiles hold every dab.
//!
//! # When the stroke ends
//!
//! The stroke's end ([`TileStore::end_journal`]) closes its journal. If its last count isn't
//! written yet, the journal waits for the flush the stroke's end triggers, which writes it, so
//! the file never ends a stroke on a batch whose count is behind its tiles; then it goes. The
//! next stroke's journal is a new one meanwhile.
//!
//! # Memory and cost
//!
//! A journal keeps the rows of the dabs not landed, [`DAB_BYTES`] each: the landing's backlog,
//! about 1.4 MB at the forced landing's 16,384 dabs. Every row a landed count passes goes at once.
//! A rebase onto a fresh working file (a save) takes none of the batches with it, so the rows a
//! later flush must write again are still at hand.
//!
//! The owner thread's cost per flush is the cut: a copy of the rows not yet written, about a
//! second of placed dabs. Encoding, compression and syncing happen on the writer thread.

use std::collections::VecDeque;

use super::TileStore;
use super::error::{CacheError, Invariant};
use crate::OpId;
use crate::journal::{DAB_BYTES, DabBatch, JournalError, MAX_JOURNAL_DABS};

/// What the working file's batches of one stroke hold so far.
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
        }
    }

    /// Adds the dabs `placed`, the next the stroke placed.
    ///
    /// # Errors
    ///
    /// [`JournalError::PastLastIndex`] if they would run past the last index a `u32` counts;
    /// none is added then.
    fn place(&mut self, placed: &[[u8; DAB_BYTES]]) -> Result<(), JournalError> {
        self.placed = u32::try_from(placed.len())
            .ok()
            .and_then(|count| self.placed.checked_add(count))
            .ok_or(JournalError::PastLastIndex)?;
        self.rows.extend(placed.iter().copied());
        Ok(())
    }

    /// Sets the landed count to `landed`, dropping the rows it passes.
    ///
    /// # Errors
    ///
    /// [`JournalError::LandedBack`] below the count it has; [`JournalError::LandedPastPlaced`]
    /// past the dabs placed. The journal is unchanged then.
    fn land(&mut self, landed: u32) -> Result<(), JournalError> {
        if landed < self.landed {
            return Err(JournalError::LandedBack {
                landed,
                before: self.landed,
            });
        }
        if landed > self.placed {
            return Err(JournalError::LandedPastPlaced {
                landed,
                placed: self.placed,
            });
        }
        let passed = usize::try_from(landed - self.landed).unwrap_or(usize::MAX);
        self.rows.drain(..passed.min(self.rows.len()));
        self.landed = landed;
        Ok(())
    }

    /// The index of the first dab the next batch holds: the later of the dabs written and those
    /// landed.
    fn unwritten_from(&self) -> u32 {
        self.written
            .map_or(0, |written| written.placed)
            .max(self.landed)
    }

    /// Whether a batch is due (see the module notes): a dab placed and neither written nor
    /// landed, or a landed count that moved since the last batch written.
    fn is_due(&self) -> bool {
        self.unwritten_from() < self.placed
            || self
                .written
                .is_some_and(|written| written.landed != self.landed)
    }

    /// The journal's cut, if a batch is due: the batches of the dabs not yet written, at most
    /// [`MAX_JOURNAL_DABS`] each (one of no dabs if only the landed count moved), each with the
    /// landed count as it is now, and what the file holds once they are in it.
    ///
    /// This copies the rows not yet written, on the calling thread.
    ///
    /// # Errors
    ///
    /// The [`JournalError`] of a batch [`DabBatch::new`] refuses: never, as the journal holds its
    /// counts within a batch's.
    fn cut(&self) -> Result<Option<(Vec<DabBatch>, JournalCut)>, JournalError> {
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
            let batch = DabBatch::new(self.op, first, self.landed, rows)?;
            first = batch.placed();
            batches.push(batch);
            if first >= self.placed {
                break;
            }
        }
        let cut = JournalCut {
            op: self.op,
            written: Written {
                placed: first,
                landed: self.landed,
            },
        };
        Ok(Some((batches, cut)))
    }
}

/// What a flush's cut took of one stroke's journal: what the working file holds of it once the
/// cut's batches are appended ([`TileStore::append_journal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JournalCut {
    /// The stroke's op.
    op: OpId,
    /// What the file holds of it then.
    written: Written,
}

/// The store's journals: the open stroke's, and those of strokes ended with their last count not
/// yet written (see the module notes).
#[derive(Debug, Default)]
pub(super) struct Journals {
    /// The open stroke's journal, from its first dab or landed count.
    open: Option<Journal>,
    /// Journals of strokes that have ended, each waiting for a flush to write its last batch, in
    /// the order they ended.
    ended: Vec<Journal>,
}

impl Journals {
    /// The open journal of the stroke made as `op`, made if there is none. One of another op
    /// is closed first, as its stroke's end would close it.
    fn open(&mut self, op: OpId) -> &mut Journal {
        if self.open.as_ref().is_some_and(|journal| journal.op != op) {
            self.close();
        }
        self.open.get_or_insert_with(|| Journal::new(op))
    }

    /// Closes the open journal, if any: it waits for its last batch if one is due, else goes.
    fn close(&mut self) {
        if let Some(journal) = self.open.take()
            && journal.is_due()
        {
            self.ended.push(journal);
        }
    }

    /// Every journal: the ended ones in the order they ended, then the open one.
    fn all(&self) -> impl Iterator<Item = &Journal> {
        self.ended.iter().chain(&self.open)
    }
}

impl TileStore {
    /// Adds `placed`, the next dabs the stroke made as `op` placed, to its journal: kept for the
    /// next flush to write (see the module notes). The stroke's first call opens its journal.
    ///
    /// # Errors
    ///
    /// [`JournalError::PastLastIndex`] if the stroke's dabs would run past the last index a `u32`
    /// counts; none is added then.
    pub(crate) fn journal(
        &mut self,
        op: OpId,
        placed: &[[u8; DAB_BYTES]],
    ) -> Result<(), JournalError> {
        self.journals.open(op).place(placed)
    }

    /// `landed` of the dabs the stroke made as `op` placed have landed: called as their edits are
    /// made, so a flush's cut always pairs the store's tiles with the count that made them.
    ///
    /// # Errors
    ///
    /// [`JournalError::LandedBack`] below the count it had; [`JournalError::LandedPastPlaced`]
    /// past the dabs journalled. Nothing changes then.
    pub(crate) fn journal_landed(&mut self, op: OpId, landed: u32) -> Result<(), JournalError> {
        self.journals.open(op).land(landed)
    }

    /// The open stroke ended: its journal goes, once its last count is written if it isn't yet
    /// (see the module notes).
    pub(crate) fn end_journal(&mut self) {
        self.journals.close();
    }

    /// Whether a flush now would write a batch.
    pub(crate) fn has_journal_work(&self) -> bool {
        self.journals.all().any(Journal::is_due)
    }

    /// The journals' cut, for the flush taking its snapshot now: every batch due, the ended
    /// strokes' first, and what each journal's file holds once they are appended. Nothing
    /// changes until [`TileStore::append_journal`] appends them.
    ///
    /// The cost is a copy of the rows not yet written, on the owner thread.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] with [`Invariant::Journal`] if a batch is refused: a bug.
    pub(crate) fn cut_journal(&self) -> Result<(Vec<DabBatch>, Vec<JournalCut>), CacheError> {
        let mut batches = Vec::new();
        let mut cuts = Vec::new();
        for journal in self.journals.all() {
            if let Some((cut_batches, cut)) = journal.cut().map_err(Invariant::Journal)? {
                batches.extend(cut_batches);
                cuts.push(cut);
            }
        }
        Ok((batches, cuts))
    }

    /// Appends `framed`, the batches of a flush's cut as the writer thread framed them, to the
    /// working file with one write, then notes each journal of `cuts` written to it.
    ///
    /// Noting them only once the write is done is what keeps a batch from being written twice:
    /// if it fails, the journals are as they were, and the next flush writes the same dabs. A
    /// journal gone since the cut (a rebase dropped it) takes nothing; an ended one whose last
    /// count is in goes.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if the append fails.
    pub(crate) fn append_journal(
        &mut self,
        framed: &[Vec<u8>],
        cuts: &[JournalCut],
    ) -> Result<(), CacheError> {
        self.file
            .append_batches(framed)
            .map_err(CacheError::WriteBack)?;
        let journals = &mut self.journals;
        for cut in cuts {
            let found = journals.ended.iter_mut().chain(&mut journals.open);
            for journal in found.filter(|journal| journal.op == cut.op) {
                journal.written = Some(cut.written);
            }
        }
        journals.ended.retain(Journal::is_due);
        Ok(())
    }

    /// The store moved onto a fresh working file, which holds no batch: the ended journals go
    /// (no batch in the new file is behind its tiles), and the open one writes its dabs not landed
    /// again, from its landed count, with the next flush.
    pub(super) fn rebase_journal(&mut self) {
        self.journals.ended.clear();
        if let Some(journal) = &mut self.journals.open {
            journal.written = None;
        }
    }
}
