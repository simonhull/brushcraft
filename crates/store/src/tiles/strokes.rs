//! Stroke records on their way to the files (see [`crate::stroke_record`]).
//!
//! A stroke's record is framed when the stroke ends ([`TileStore::record_stroke`]) and waits in
//! the store for the flush that stroke end triggers: the recovery writer appends it as the flush
//! opens its cut ([`TileStore::flush_strokes`]), before the save point, so the save point that
//! holds the stroke's last edits holds its record too, and none before it does. A save copies the
//! working file's records, and those still waiting, into the project file; a roll-over carries the
//! working file's into the new one, and those waiting stay for its first flush.

use super::TileStore;
use super::error::CacheError;
use crate::file::{FileError, ProjectFile, frame_stroke};
use crate::stroke_record::StrokeRecord;

impl TileStore {
    /// Frames `record` and keeps it for the next flush (or save) to append to the working file.
    ///
    /// The cost is the record's compression, on the calling thread: a few kilobytes of samples.
    /// It waits in memory, outside the budget, until appended: one stroke's while a recovery
    /// writer runs, every stroke's since the last save while crash protection is paused.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if compressing it fails; nothing is kept then.
    pub(crate) fn record_stroke(&mut self, record: &StrokeRecord) -> Result<(), CacheError> {
        let framed = frame_stroke(record).map_err(CacheError::Store)?;
        self.strokes_waiting.push(framed);
        Ok(())
    }

    /// Whether stroke records wait for the working file.
    pub(crate) const fn has_waiting_strokes(&self) -> bool {
        !self.strokes_waiting.is_empty()
    }

    /// Appends every stroke record waiting to the working file, with one write: called as a
    /// flush opens its cut, so the flush's save point holds them (see the module notes).
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if the append fails: they keep waiting, for the next flush.
    pub(crate) fn flush_strokes(&mut self) -> Result<(), CacheError> {
        self.file
            .append_strokes(&self.strokes_waiting)
            .map_err(CacheError::WriteBack)?;
        self.strokes_waiting.clear();
        Ok(())
    }

    /// Appends to `to` a copy of every stroke record of the working file, in order, then, if
    /// `waiting`, every one waiting (which stay waiting). Returns how many were waiting.
    ///
    /// A record of the working file that doesn't read is left out: history, which never stops a
    /// save or a roll-over.
    ///
    /// # Errors
    ///
    /// `wrap` of the error of an append to `to`.
    pub(super) fn copy_strokes(
        &self,
        to: &mut ProjectFile,
        waiting: bool,
        wrap: fn(FileError) -> CacheError,
    ) -> Result<usize, CacheError> {
        for loc in self.file.stroke_locations() {
            if let Ok(raw) = self.file.read_stroke_record(loc) {
                to.append_stroke_copy(&raw).map_err(wrap)?;
            }
        }
        if !waiting {
            return Ok(0);
        }
        to.append_strokes(&self.strokes_waiting).map_err(wrap)?;
        Ok(self.strokes_waiting.len())
    }

    /// Forgets the first `count` stroke records waiting: a save copied them into the project
    /// file, which the store has moved onto.
    pub(super) fn forget_saved_strokes(&mut self, count: usize) {
        self.strokes_waiting
            .drain(..count.min(self.strokes_waiting.len()));
    }
}
