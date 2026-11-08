//! Streaming the working file: the change-feed consumers a recovery writer subscribes, the dirty
//! tiles it snapshots and appends, and the save points that make its records count.
//!
//! # Protocol
//!
//! [`crate::RecoveryWriter`] (`recovery.rs`) drives these, on the owner thread:
//!
//! - It subscribes with [`TileStore::subscribe_stream`], leaves with
//!   [`TileStore::unsubscribe_stream`], and asks [`TileStore::has_unflushed`] and
//!   [`TileStore::has_unsaved_records`] whether a flush has anything to do.
//! - A flush starts with [`TileStore::open_cut`] and notes [`TileStore::generation`], which every
//!   rebase moves on: a flush that sees it change is abandoned. [`TileStore::snapshot_dirty`] then
//!   copies the dirty slots, with no encoding or I/O, [`TileStore::flush_bounds`] appends the
//!   bounds of the same state at the same cut, and [`TileStore::flush_strokes`] the records of the
//!   strokes ended by then.
//! - With the copies encoded, the tiles left out are dealt with (see [`super::bounds`]; a lost one
//!   leaves the flush mark through [`TileStore::drop_from_stream`]), and
//!   [`TileStore::append_streamed`] appends the rest.
//! - Once a sync ([`TileStore::working_syncer`]) covers [`TileStore::working_named_end`],
//!   [`TileStore::write_save_point`] appends the save point, unsynced, and another sync makes it
//!   durable.
//!
//! [`crate::Project`] marks a working file whose writer failed with [`TileStore::fail_working`]:
//! no save point is written there again until the store moves to another working file.

use std::collections::BTreeSet;

use geom::TileKey;

use super::TileStore;
use super::error::{CacheError, Invariant};
use super::layers::{Dirty, Snapshot};
use crate::feed::ConsumerId;
use crate::file::{FileSyncer, FramedTile};
use crate::slot::CorrectionSlot;
use crate::stamp::Stamp;

impl TileStore {
    /// Subscribes a streaming consumer (the recovery writer) to the change feed, with an empty
    /// flush mark: it streams the edits made from now on.
    pub(crate) fn subscribe_stream(&mut self) -> ConsumerId {
        let id = self.feed.subscribe();
        self.streams.insert(id, BTreeSet::new());
        id
    }

    /// Removes a streaming consumer and its flush mark; entries only it was holding leave the
    /// feed.
    pub(crate) fn unsubscribe_stream(&mut self, id: ConsumerId) {
        self.feed.unsubscribe(id);
        self.streams.remove(&id);
    }

    /// Whether streaming consumer `id` has edits to flush: keys pushed to the feed since its last
    /// snapshot, or keys of its flush mark still unconfirmed; or bounds the working file lacks;
    /// or stroke records waiting for it.
    pub(crate) fn has_unflushed(&self, id: ConsumerId) -> bool {
        self.feed.has_pending(id)
            || self.streams.get(&id).is_some_and(|mark| !mark.is_empty())
            || self.bounds.has_unflushed()
            || self.display.has_unflushed()
            || self.has_waiting_strokes()
    }

    /// Copies, for streaming consumer `id`, every correction slot whose latest edit isn't in the
    /// working file yet: the keys of its flush mark, with those the feed has pushed since its
    /// last call added, that are resident and dirty, each with the op and stamp of its latest
    /// write.
    ///
    /// A key of the mark whose slot is clean or not resident is in the working file already (a
    /// slot leaves the pool only once written back, and only a write clears it), so it leaves
    /// the mark without a copy. The rest stay in the mark until
    /// [`TileStore::append_streamed`] appends them, or finds them written by someone else.
    ///
    /// A dirty slot with no op to write it with (a broken slot: every slot marked dirty gets
    /// one first) is left out, dropped from the mark and listed in `skipped`, so it can't stall
    /// the flushes of the rest.
    ///
    /// This is the drawing thread's share of a flush: a copy of each slot, no encoding or I/O.
    ///
    /// Save's consumer is drained into [`TileStore::since_rebase`] here too, so the feed holds
    /// no more than one flush's worth of entries between saves.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a handle went stale. The mark keeps every key then.
    pub(crate) fn snapshot_dirty(
        &mut self,
        id: ConsumerId,
        skipped: &mut Vec<TileKey>,
    ) -> Result<Vec<Snapshot>, CacheError> {
        self.drain_saves();
        let drained = self.feed.drain(id);
        let Some(mark) = self.streams.get_mut(&id) else {
            return Ok(Vec::new());
        };
        mark.extend(drained.into_iter().map(|(key, _)| key));
        let (corrections, unsaved) = (&self.corrections, &self.unsaved);
        let mut snapshots = Vec::with_capacity(mark.len());
        let mut written = Vec::new();
        for &key in mark.iter() {
            let Some(slot_id) = corrections.find(&key) else {
                written.push(key);
                continue;
            };
            if corrections.is_dirty(slot_id) != Some(true) {
                written.push(key);
                continue;
            }
            let Some(&Dirty { op, stamp }) = unsaved.get(&key) else {
                // A broken slot: dropped, so the rest still flush.
                skipped.push(key);
                written.push(key);
                continue;
            };
            let mut copy = bytemuck::zeroed_box::<CorrectionSlot>();
            *copy = *corrections.get(slot_id).ok_or(Invariant::StaleHandle)?;
            snapshots.push(Snapshot {
                key,
                op,
                stamp,
                slot: copy,
            });
        }
        for key in &written {
            mark.remove(key);
        }
        Ok(snapshots)
    }

    /// Appends streaming consumer `id`'s encoded `tiles` to the working file: each one whose slot
    /// is still resident and dirty. A slot unchanged since its snapshot (its latest write is the
    /// one the record copies, by stamp) is clean once its record is appended, and leaves the
    /// mark. Ops can't tell: a later dab of the same stroke writes the slot with the same op.
    ///
    /// A slot written again since its snapshot is appended all the same, so the save point after
    /// this append holds every edit up to the snapshot, but stays dirty and in the mark: its
    /// newer edit isn't in the file, and whatever writes the slot next (the next flush, a save,
    /// or a write-back to make room) appends it after this record. Reads of a dirty tile come
    /// from its slot, never the file, meanwhile.
    ///
    /// A tile whose slot is clean or gone was written by someone else (a write-back to make
    /// room) after the snapshot, at a write at least as new. If that write is at or before the
    /// working file's cut, the record is the cut's state and the copy is skipped; if it is past
    /// the cut, the copy is appended for the cut's save point, and the newer record stays the
    /// latest (see [`ProjectFile::open_cut`](crate::file::ProjectFile::open_cut)). Either way it
    /// leaves the mark.
    ///
    /// # Errors
    ///
    /// [`CacheError::WriteBack`] if an append fails: that tile and those after it stay dirty
    /// and in the mark. [`CacheError::Internal`] for a stale handle.
    pub(crate) fn append_streamed(
        &mut self,
        id: ConsumerId,
        tiles: &[FramedTile],
    ) -> Result<(), CacheError> {
        for tile in tiles {
            let key = tile.key();
            let dirty = self
                .corrections
                .find(&key)
                .filter(|&found| self.corrections.is_dirty(found) == Some(true));
            let confirmed = match dirty {
                // Written since the snapshot by an edit past the cut: the cut still needs this
                // copy, which goes to the cut's save point, not over the newer record.
                None if self.file.is_past_cut(&key) => {
                    self.file
                        .append_framed(tile)
                        .map_err(CacheError::WriteBack)?;
                    true
                }
                None => true,
                Some(found) => {
                    self.file
                        .append_framed(tile)
                        .map_err(CacheError::WriteBack)?;
                    let unchanged = self
                        .unsaved
                        .get(&key)
                        .is_some_and(|write| write.stamp == tile.stamp());
                    if unchanged {
                        self.unsaved.remove(&key);
                        self.corrections.set_dirty(found, false)?;
                        self.known_ops.insert(key, tile.op());
                    }
                    unchanged
                }
            };
            if confirmed && let Some(mark) = self.streams.get_mut(&id) {
                mark.remove(&key);
            }
        }
        Ok(())
    }

    /// Opens a cut in the working file at the latest write so far: the next save point the
    /// recovery writer writes names the state as of now, whatever is written back meanwhile
    /// (see [`ProjectFile::open_cut`](crate::file::ProjectFile::open_cut)). Called as a flush takes
    /// its snapshot. Returns the write the cut is at.
    ///
    /// The cut is at a write stamp, not an op, so it falls between two dabs of one stroke.
    pub(crate) fn open_cut(&mut self) -> Stamp {
        self.file.open_cut(self.last_write);
        self.last_write
    }

    /// Whether a correction slot took a write after `stamp`: an edit a cut at `stamp` doesn't
    /// hold.
    pub(crate) fn written_since(&self, stamp: Stamp) -> bool {
        self.last_write > stamp
    }

    /// The offset of the working file's latest save point; `None` before its first.
    pub(crate) fn working_last_save_point(&self) -> Option<u64> {
        self.file.last_save_point()
    }

    /// Drops the cut open in the working file, if any: the flush that opened it was abandoned,
    /// or had nothing to flush, so no save point follows it (see
    /// [`ProjectFile::drop_cut`](crate::file::ProjectFile::drop_cut)).
    pub(crate) fn drop_cut(&mut self) {
        self.file.drop_cut();
    }

    /// The working-file generation: moved on by every rebase.
    pub(crate) const fn generation(&self) -> u64 {
        self.generation
    }

    /// The working file is a recovery file whose writer failed (see
    /// [`TileStore::working_failed`]): no save point is written there again, until the store
    /// moves to another working file.
    pub(crate) const fn fail_working(&mut self) {
        self.working_failed = true;
    }

    /// Whether the working file is a recovery file whose writer failed, which the store hasn't
    /// moved off yet ([`TileStore::fail_working`]).
    pub(crate) const fn is_working_failed(&self) -> bool {
        self.working_failed
    }

    /// Drops `key` from streaming consumer `id`'s flush mark: its copy couldn't be flushed, and
    /// is reported instead of tried again with every flush.
    pub(crate) fn drop_from_stream(&mut self, id: ConsumerId, key: &TileKey) {
        if let Some(mark) = self.streams.get_mut(&id) {
            mark.remove(key);
        }
    }

    /// A handle that syncs the working file from another thread.
    pub(crate) fn working_syncer(&self) -> FileSyncer {
        self.file.syncer()
    }

    /// Where the working file's next record goes: it moves with every append, so a caller can
    /// tell whether anything was written since it last looked.
    pub(crate) const fn working_len(&self) -> u64 {
        self.file.len()
    }

    /// Where the last record the working file's next save point names ends
    /// ([`ProjectFile::named_end`](crate::file::ProjectFile::named_end)): a sync must cover it
    /// before that save point is written.
    pub(crate) fn working_named_end(&self) -> u64 {
        self.file.named_end()
    }

    /// Whether the working file has records (tiles or bounds) written since its last save
    /// point.
    pub(crate) fn has_unsaved_records(&self) -> bool {
        self.file.has_unsaved_records()
    }

    /// Appends a save point to the working file naming every tile and bound written to it so far,
    /// without syncing (see
    /// [`ProjectFile::write_save_point`](crate::file::ProjectFile::write_save_point)), and returns
    /// its offset. Dirty slots are not written: their edits are in no save point yet.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if the write fails; [`CacheError::Internal`] if the save point's
    /// derived maps disagree with it ([`Invariant::SavePointMismatch`]).
    pub(crate) fn write_save_point(&mut self) -> Result<u64, CacheError> {
        self.file
            .write_save_point()
            .map_err(|e| CacheError::from_save_point(e, CacheError::Store))
    }
}
