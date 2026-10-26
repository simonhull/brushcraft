//! The spatial index's bounds as the store keeps them: recorded with the tiles, layered over the
//! saved bounds, written out when tiles are, and dropped or reloaded with the tiles they bound.
//!
//! # Protocol
//!
//! [`crate::SpatialIndex`] (`index.rs`) reads the bounds the store keeps with
//! [`TileStore::stored_bounds`] (`None` where none is known, and the tile must be evaluated),
//! and records those it computes from the store's current state with
//! [`TileStore::record_bounds`], which settles the writes since its last call first.
//!
//! [`crate::RecoveryWriter`] (`recovery.rs`) calls the rest within a flush (see
//! [`super::stream`]): [`TileStore::flush_bounds`] right after the snapshot, before any later
//! write, so the flush's save point names tiles and bounds of one state; then, for the copies
//! that didn't encode, [`TileStore::left_out_at_reply`] to tell which are lost in memory, and
//! [`TileStore::forget_bounds_at_cut`] to append as unknown every bound their at-cut edits could
//! have changed.

use geom::TileKey;

use super::TileStore;
use super::error::CacheError;
use super::round::{Fill, fill_slot};
use crate::file::Loc;
use crate::tile_bounds::TileBounds;

impl TileStore {
    /// Records `bounds`, computed from the store's current state (by the spatial index), as
    /// the bounds the store keeps for those tiles. Those replacing an edit's change still to be
    /// flushed go to the working file with the next flush; the rest are derived, and go to the
    /// project file with its next save (see [`crate::bounds`]).
    ///
    /// The writes since the last call are settled first (see [`crate::bounds`]), so the bounds
    /// they made unknown are replaced, not the other way round.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing a written tile's readers fails (never, for tiles of the
    /// store's mesh). Nothing is recorded then.
    pub(crate) fn record_bounds(
        &mut self,
        bounds: &[(TileKey, TileBounds)],
    ) -> Result<(), CacheError> {
        Ok(self
            .bounds
            .settle_and_record(&self.mesh, &self.deps, bounds)?)
    }

    /// The bounds the store keeps for each of `keys`, in order: `None` where none is known (none
    /// was ever recorded, or a write since made it unknown), and the tile must be evaluated.
    ///
    /// Each is read as a correction tile is (see [`crate::bounds`]): the change since the last
    /// save if there is one, else the working file's record, else the base's. Records are read
    /// together, a file at a time; one that doesn't read counts as unknown.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::Store`] if a read
    /// fails.
    pub(crate) fn stored_bounds(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<TileBounds>>, CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let mut found = vec![None; keys.len()];
        let (mut working, mut base) = (Vec::new(), Vec::new());
        for (n, key) in keys.iter().enumerate() {
            if let Some(changed) = self.bounds.changed(key) {
                if let Some(slot) = found.get_mut(n) {
                    *slot = changed.bounds();
                }
            } else if let Some(loc) = self.file.locate_bounds(key) {
                working.push((n, (*key, loc)));
            } else if let Some(loc) = self.base.as_ref().and_then(|b| b.locate_bounds(key)) {
                base.push((n, (*key, loc)));
            }
        }
        let entries = |at: &[(usize, (TileKey, Loc))]| -> Vec<(TileKey, Loc)> {
            at.iter().map(|&(_, entry)| entry).collect()
        };
        let read_working = self.file.read_bounds(&entries(&working))?;
        let read_base = match &self.base {
            Some(file) if !base.is_empty() => file.read_bounds(&entries(&base))?,
            _ => Vec::new(),
        };
        for ((n, _), bounds) in working
            .iter()
            .zip(read_working)
            .chain(base.iter().zip(read_base))
        {
            if let Some(slot) = found.get_mut(*n) {
                *slot = bounds;
            }
        }
        Ok(found)
    }

    /// The bounds part of a recovery flush, at its cut: loses the `lost` tiles, left out of the
    /// flush (see [`TileStore::lose_left_out`]), settles the writes so far, makes unknown the
    /// bounds the lost tiles' edits could have changed, and
    /// appends, with one write, every bound edits changed that the working file lacks, placed at
    /// the cut. Derived bounds stay out (see [`crate::bounds`]), so the cost grows with the edits
    /// since the last flush.
    ///
    /// Called right after the flush's snapshot, on the owner thread, before any later write:
    /// the bounds appended are those of the state the snapshot copies, so the flush's save point
    /// names tiles and bounds of one state.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if an
    /// append fails: the bounds are kept for the next flush.
    pub(crate) fn flush_bounds(&mut self, lost: &[TileKey]) -> Result<(), CacheError> {
        self.lose_left_out(lost)?;
        self.bounds.settle(&self.mesh, &self.deps)?;
        self.bounds
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        self.write_unflushed_bounds()
    }

    /// Loses the `lost` tiles a flush leaves out (broken slots), as a save loses a tile whose
    /// latest edit it can't read (see [`TileStore::rebase`]): the store stops reading the edit
    /// the recovery file doesn't hold, and drops every evaluation that followed from it, so
    /// nothing evaluated later (and no bound recorded from it) describes a state no save point
    /// holds.
    ///
    /// Each slot is dropped, or, if a pin holds it, filled again in place from the tile's last
    /// record (see [`TileStore::reload_lost`]), as a rebase does with a lost tile's. The tile is
    /// read from the working file's record from then on, else from the base: if the working
    /// file has no record of it, it is at its last saved state again, and no longer among the
    /// tiles the next save copies.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn lose_left_out(&mut self, lost: &[TileKey]) -> Result<(), CacheError> {
        if lost.is_empty() {
            return Ok(());
        }
        // Drained first, so the feed can't bring a lost key back among the tiles to save.
        self.drain_saves();
        for key in lost {
            if self.drop_lost_slot(key)? {
                self.reload_lost(key);
            }
            if !self.file.contains(key) {
                self.since_rebase.remove(key);
            }
        }
        self.invalidate_edited(lost.iter().copied());
        Ok(())
    }

    /// Forgets a lost tile's latest write and drops its correction slot, if it has one: its
    /// edit is in no file the store reads. Returns whether a pin holds the slot, which then
    /// stays, marked clean, for [`TileStore::reload_lost`] to fill again from the tile's last
    /// record once the store reads the files it is to read from.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    pub(super) fn drop_lost_slot(&mut self, key: &TileKey) -> Result<bool, CacheError> {
        // The tile reads as its last record from now on, not as the lost write.
        self.corrections_changed();
        self.unsaved.remove(key);
        let Some(id) = self.corrections.find(key) else {
            return Ok(false);
        };
        self.corrections.set_dirty(id, false)?;
        if self.corrections.is_pinned(key) {
            return Ok(true);
        }
        self.corrections.remove(key)?;
        self.known_ops.remove(key);
        Ok(false)
    }

    /// Fills a lost tile's slot, kept because a pin holds it, again in place from the tile's
    /// last record (the working file's, else the base's, else zero), as loading it would: what
    /// the slot holds is then what the files hold, and an edit after the pin goes starts from
    /// it, never from the lost edit. The slot stays clean.
    ///
    /// If the record can't be read, the slot is left holding no tile, as a failed load leaves
    /// one: every read or edit of it fails until it leaves the pool (a clean slot, so dropping
    /// it writes nothing), and the tile is read from its record again after that.
    pub(super) fn reload_lost(&mut self, key: &TileKey) {
        self.corrections_changed();
        let Some(id) = self.corrections.find(key) else {
            return;
        };
        let fill = Fill {
            key: *key,
            id,
            found: self.locate(key),
        };
        let Some(slot) = self.corrections.get_mut(id) else {
            return;
        };
        let filled = fill_slot(
            &self.file,
            self.base.as_ref(),
            &fill,
            slot,
            &mut self.read_scratch,
        );
        match filled {
            Ok(Some(op)) => {
                self.known_ops.insert(*key, op);
            }
            Ok(None) | Err(_) => {
                self.known_ops.remove(key);
            }
        }
    }

    /// Appends, with one write, every bound edits changed that the working file lacks (not the
    /// derived ones, see [`crate::bounds`]), placed at the latest write: the state they
    /// describe. Settles the writes first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if
    /// the append fails: the bounds stay unflushed.
    fn write_unflushed_bounds(&mut self) -> Result<(), CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let unflushed: Vec<TileKey> = self.bounds.unflushed().map(|(key, _)| *key).collect();
        self.file
            .write_bounds_many(
                self.bounds.unflushed().map(|(key, kept)| (*key, kept)),
                self.last_write,
            )
            .map_err(CacheError::WriteBack)?;
        for key in &unflushed {
            self.bounds.flushed(key);
        }
        Ok(())
    }

    /// Appends every bound a save of the working file itself writes (see
    /// [`Book::unsaved`](crate::bounds::Book::unsaved)): those a flush would, and the derived
    /// ones, with one write. Settles the writes first.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if
    /// the append fails: the bounds stay as they were.
    pub(super) fn write_unsaved_bounds(&mut self) -> Result<(), CacheError> {
        self.bounds.settle(&self.mesh, &self.deps)?;
        let unsaved: Vec<TileKey> = self.bounds.unsaved().map(|(key, _)| *key).collect();
        self.file
            .write_bounds_many(
                self.bounds.unsaved().map(|(key, kept)| (*key, kept)),
                self.last_write,
            )
            .map_err(CacheError::WriteBack)?;
        for key in &unsaved {
            self.bounds.saved(key);
        }
        Ok(())
    }

    /// Which of the tiles whose copies a flush couldn't encode (`broken`) are lost in memory:
    /// those whose live slot is dirty and broken too (holds no valid tile). A valid slot means
    /// the failure was transient (a compression error, say), whether or not it was edited since
    /// the snapshot: it stays dirty, in the flush mark, for the next flush to try again. A slot
    /// clean or gone was written by someone else since, encoded.
    pub(crate) fn left_out_at_reply(&self, broken: &[TileKey]) -> Vec<TileKey> {
        broken
            .iter()
            .filter(|key| {
                self.corrections
                    .find(key)
                    .filter(|&id| self.corrections.is_dirty(id) == Some(true))
                    .is_some_and(|id| {
                        self.corrections
                            .get(id)
                            .is_none_or(|slot| slot.to_tile().is_err())
                    })
            })
            .copied()
            .collect()
    }

    /// The two halves of a flush leaving out the `left_out` tiles, whose copies didn't encode.
    ///
    /// In the file: every bound their at-cut edits could have changed is appended as unknown,
    /// placed at the flush's cut, whatever their live slots hold now: the save point holds their
    /// older state, and the bounds the flush appended at its snapshot may describe the edits it
    /// lacks. The store's own are made unknown as well, and the next flush writes them again.
    ///
    /// In memory: only the `lost` ones (broken slots, see [`TileStore::left_out_at_reply`])
    /// are lost (see [`TileStore::lose_left_out`]); the others stay dirty for the next flush.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_bounds`]; [`CacheError::WriteBack`] if an
    /// append fails.
    pub(crate) fn forget_bounds_at_cut(
        &mut self,
        left_out: &[TileKey],
        lost: &[TileKey],
    ) -> Result<(), CacheError> {
        self.lose_left_out(lost)?;
        if left_out.is_empty() {
            return Ok(());
        }
        let stale = self
            .bounds
            .forget(&self.mesh, &self.deps, left_out.iter().copied())?;
        let at = self.file.cut_at().unwrap_or(self.last_write);
        self.file
            .write_bounds_many(stale.iter().map(|&key| (key, None)), at)
            .map_err(CacheError::WriteBack)
    }
}
