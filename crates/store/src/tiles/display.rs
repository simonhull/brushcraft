//! The display layer as the store keeps it (see [`crate::display`]).
//!
//! Recorded with the tiles, layered over the saved display records, written out when tiles are,
//! and made unknown by the writes that change it.
//!
//! It follows the bounds' protocol ([`super::bounds`]), with one difference: a recorded display
//! tile always goes to the working file with the next flush, and leaves memory once there.

use std::collections::{BTreeMap, BTreeSet};

use geom::{DisplayTile, Level, TileKey};

use super::TileStore;
use super::error::CacheError;
use crate::bounds::{Book, Kept};
use crate::display::{Display, DisplayRecord, StoredDisplay};
use crate::file::{BaseFile, FileError, ProjectFile};
use crate::meta::Role;
use crate::stamp::Stamp;

impl TileStore {
    /// Records `tiles`, built from the store's current state, as the display tiles the store
    /// keeps: they go to the working file with the next flush, and to the project file with its
    /// next save.
    ///
    /// The writes since the last call are settled first, so the display tiles they made unknown
    /// are replaced, not the other way round.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing a written tile's readers fails (never, for tiles of the
    /// store's mesh). Nothing is recorded then.
    pub fn record_display(&mut self, tiles: &[DisplayTile]) -> Result<(), CacheError> {
        for tile in tiles {
            self.bumped.remove(tile.key());
        }
        self.note_epoch_keys(tiles.iter().map(|tile| *tile.key()));
        let records: Vec<(TileKey, DisplayRecord)> = tiles
            .iter()
            .map(|tile| (*tile.key(), DisplayRecord::of(tile)))
            .collect();
        Ok(self
            .display
            .settle_and_record(&self.mesh, &self.deps, &records)?)
    }

    /// The display tiles the store keeps for each of `keys`, in order: `None` where none is
    /// known (none was ever recorded, a write since made it unknown, or its record doesn't
    /// read), and the tile must be built.
    ///
    /// Each is read as a correction tile is: the change since the last save if the store holds
    /// it, else the working file's record, else the base's.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_display`]; [`CacheError::Store`] if a
    /// read fails.
    pub fn stored_display(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<DisplayTile>>, CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        keys.iter()
            .map(|key| {
                let record = self.display_record(key)?;
                Ok(record.and_then(|r| DisplayTile::decode(*key, r.error, &r.encoded).ok()))
            })
            .collect()
    }

    /// Raises the bound of every display tile a dab's writes of the `edited` tiles can change by
    /// twice `max_displacement`, the farthest the dab moved a finest vertex: in memory only,
    /// never written, so the display tiles as they were before the stroke stay drawable, their
    /// bound still a bound, until a refresh replaces them (see [`TileStore::display_for_draw`]).
    ///
    /// Twice: the finest vertex and the display vertex compared with it can each move that far
    /// (the renderer's stand-ins move display vertices too). Call it after each dab's writes,
    /// before anything reads the store's display tiles: the first dab of a stroke keeps each
    /// tile as it was before the stroke.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing the writes' readers fails (never, for tiles of the
    /// store's mesh); [`CacheError::Store`] if a display record can't be read.
    pub fn bump_display_error(
        &mut self,
        edited: &[TileKey],
        max_displacement: f32,
    ) -> Result<(), CacheError> {
        if self.display_finest.is_none() || self.display.levels().is_empty() || edited.is_empty() {
            return Ok(());
        }
        let edited: BTreeSet<TileKey> = edited.iter().copied().collect();
        let readers =
            crate::display::readers(&self.mesh, &self.deps, &edited, self.display.levels())?;
        let raise = 2.0 * max_displacement.abs();
        for key in readers {
            if let Some((_, extra)) = self.bumped.get_mut(&key) {
                *extra += raise;
            } else if let Some(record) = self.display_record(&key)? {
                self.bumped.insert(key, (record, raise));
            }
        }
        Ok(())
    }

    /// The display tiles to draw for each of `keys`, in order: the stored tile, or one a stroke
    /// is changing as it was before the stroke with its bound raised (see
    /// [`TileStore::bump_display_error`]); `None` where neither is known, and the tile must be
    /// drawn finer.
    ///
    /// # Errors
    ///
    /// As [`TileStore::stored_display`].
    pub fn display_for_draw(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<DisplayTile>>, CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        keys.iter()
            .map(|key| {
                if let Some((record, extra)) = self.bumped.get(key) {
                    // Rounded up, so it stays a bound.
                    let bound = (record.error + extra).next_up();
                    return Ok(DisplayTile::decode(*key, bound, &record.encoded).ok());
                }
                let record = self.display_record(key)?;
                Ok(record.and_then(|r| DisplayTile::decode(*key, r.error, &r.encoded).ok()))
            })
            .collect()
    }

    /// Whether the store knows `key`'s display tile (it would read one back), without decoding
    /// it. Settles the writes first.
    ///
    /// # Errors
    ///
    /// As [`TileStore::stored_display`].
    pub fn knows_display(&mut self, key: &TileKey) -> Result<bool, CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        Ok(self.display_record(key)?.is_some())
    }

    /// `key`'s display record as the store reads it (see [`TileStore::stored_display`]);
    /// settled by the caller.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if a read fails.
    fn display_record(&self, key: &TileKey) -> Result<Option<DisplayRecord>, CacheError> {
        match self.display.kept(key) {
            Some(Kept::Known(record)) => return Ok(Some(record.clone())),
            Some(Kept::Unknown) => return Ok(None),
            Some(Kept::Written) | None => {}
        }
        if let Some(found) = self.file.read_display(key)? {
            return Ok(found.known());
        }
        Ok(match &self.base {
            Some(base) => base.read_display(key)?.and_then(StoredDisplay::known),
            None => None,
        })
    }

    /// The display part of a recovery flush, at its cut (see [`TileStore::flush_bounds`]): makes
    /// unknown the display tiles the `lost` tiles' edits could have changed, then appends every
    /// display tile the working file lacks, placed at the store's latest write.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_display`]; [`CacheError::WriteBack`] if
    /// the append fails: the tiles stay unflushed.
    pub(super) fn flush_display(&mut self, lost: &[TileKey]) -> Result<(), CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        self.display
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        let unflushed: Vec<TileKey> = self.display.unflushed_kept().map(|(key, _)| *key).collect();
        write_kept(
            &mut self.file,
            None,
            &self.display,
            &unflushed,
            self.last_write,
        )
        .map_err(CacheError::WriteBack)?;
        for key in &unflushed {
            self.display.flushed(key);
        }
        Ok(())
    }

    /// Appends every display tile a save of the working file itself writes: those a flush
    /// would. Settles the writes first.
    ///
    /// # Errors
    ///
    /// As [`TileStore::flush_display`].
    pub(super) fn write_unsaved_display(&mut self) -> Result<(), CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        let unsaved: Vec<TileKey> = self.display.unsaved_kept().map(|(key, _)| *key).collect();
        write_kept(
            &mut self.file,
            None,
            &self.display,
            &unsaved,
            self.last_write,
        )
        .map_err(CacheError::WriteBack)?;
        for key in &unsaved {
            self.display.saved(key);
        }
        Ok(())
    }

    /// Makes unknown, in the store and in the working file at its cut, every display tile the
    /// edits of the `left_out` tiles (left out of a flush) could have changed.
    ///
    /// # Errors
    ///
    /// As [`TileStore::flush_display`].
    pub(super) fn forget_display_at_cut(&mut self, left_out: &[TileKey]) -> Result<(), CacheError> {
        if left_out.is_empty() {
            return Ok(());
        }
        let stale = self
            .display
            .forget(&self.mesh, &self.deps, left_out.iter().copied())?;
        let at = self.file.cut_at().unwrap_or(self.last_write);
        self.file
            .write_display_many(stale.iter().map(|&key| (key, None)), at)
            .map_err(CacheError::WriteBack)
    }

    /// Writes every display tile changed since the last rebase into `project` (a save), from
    /// the store's memory or the working file; makes unknown first those the `lost` tiles' edits
    /// could have changed.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_display`]; [`CacheError::Save`] if a write
    /// to `project` fails.
    pub(super) fn save_display(
        &mut self,
        project: &mut ProjectFile,
        lost: &BTreeSet<TileKey>,
    ) -> Result<(), CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        self.display
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        let all: Vec<TileKey> = self.display.all_kept().map(|(key, _)| *key).collect();
        write_kept(project, Some(&self.file), &self.display, &all, Stamp::ZERO)
            .map_err(CacheError::Save)
    }

    /// Carries the display layer into `fresh`, a new working file (a roll-over): every record
    /// of the old working file the store hasn't changed since, copied, and every change a flush
    /// or the old file holds.
    ///
    /// # Errors
    ///
    /// As [`TileStore::flush_display`].
    pub(super) fn carry_display(
        &mut self,
        fresh: &mut ProjectFile,
        lost: &[TileKey],
    ) -> Result<(), CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        self.display
            .forget(&self.mesh, &self.deps, lost.iter().copied())?;
        let carried: Vec<TileKey> = self.file.display_keys().copied().collect();
        for key in &carried {
            if self.display.kept(key).is_none() {
                copy_record(fresh, &self.file, key).map_err(CacheError::WriteBack)?;
            }
        }
        let carried: BTreeSet<TileKey> = carried.into_iter().collect();
        let changed: Vec<TileKey> = self
            .display
            .all_kept()
            .filter(|(key, _)| carried.contains(key) || self.display.is_unflushed(key))
            .map(|(key, _)| *key)
            .collect();
        write_kept(
            fresh,
            Some(&self.file),
            &self.display,
            &changed,
            Stamp::ZERO,
        )
        .map_err(CacheError::WriteBack)
    }
}

/// Writes `keys`' display tiles as `book` keeps them into `to`, placed at `stamp`.
///
/// A known one is written as it is, an unknown one as unknown, one the working file holds
/// ([`Kept::Written`]) copied from `working` (as unknown if it doesn't read; with no `working`,
/// `to` is the working file itself, which has it).
///
/// # Errors
///
/// The [`FileError`] of a failed write.
fn write_kept(
    to: &mut ProjectFile,
    working: Option<&ProjectFile>,
    book: &Book<Display>,
    keys: &[TileKey],
    stamp: Stamp,
) -> Result<(), FileError> {
    let mut held = Vec::new();
    for key in keys {
        match book.kept(key) {
            Some(Kept::Known(record)) => held.push((*key, Some(record))),
            Some(Kept::Unknown) => held.push((*key, None)),
            Some(Kept::Written) => {
                if let Some(working) = working {
                    copy_record(to, working, key)?;
                }
            }
            None => {}
        }
    }
    to.write_display_many(held, stamp)
}

/// Copies `key`'s display record from `from` into `to`, as unknown if it doesn't read.
///
/// # Errors
///
/// The [`FileError`] of a failed write.
fn copy_record(to: &mut ProjectFile, from: &ProjectFile, key: &TileKey) -> Result<(), FileError> {
    match from.read_tile_record(key) {
        Ok(Some(raw)) => to.append_display_copy(key, &raw),
        Ok(None) | Err(_) => to.write_display_many([(*key, None)], Stamp::ZERO),
    }
}

/// The display book of a store over `working` and `base`: every level either file holds display
/// tiles of and, for a recovery file, its display tiles as changed since the last save, held by
/// the working file.
pub(super) fn display_book_of(working: &ProjectFile, base: Option<&BaseFile>) -> Book<Display> {
    let mut levels: BTreeSet<Level> = working.display_keys().map(TileKey::level).collect();
    if let Some(base) = base {
        levels.extend(base.display_keys().map(TileKey::level));
    }
    let changed: BTreeMap<TileKey, Kept<DisplayRecord>> = match working.meta().role() {
        Role::Recovery { .. } => working
            .display_keys()
            .map(|key| (*key, Kept::Written))
            .collect(),
        Role::Project => BTreeMap::new(),
    };
    Book::with(levels, changed)
}
