//! The display layer as the store keeps it (see [`crate::display`]).
//!
//! Recorded with the tiles, layered over the saved display records, written out when tiles are,
//! and made unknown by the writes that change it.
//!
//! It follows the bounds' protocol ([`super::bounds`]), with one difference: a recorded display
//! tile always goes to the working file with the next flush, and leaves memory once there.
//!
//! # Written through
//!
//! A display record is tens of kilobytes, and a rebuild records the whole layer (about 1 GB on
//! the benchmark project), so the records waiting for the working file are held to a share of
//! the budget ([`StoreConfig::display_waiting_bytes`](super::StoreConfig::display_waiting_bytes)):
//! past it, [`TileStore::write_through_display`] appends some to the working file between
//! flushes, and they leave memory as a flush's do. That changes nothing a save point names, nor
//! what a crash leaves:
//!
//! - a record appended past the last save point is named by none until the next one, and a crash
//!   before it discards it with the rest of the file's tail (the tile is rebuilt, as it would be
//!   had it waited in memory);
//! - it is placed at the store's latest write, the state it was built from, and the writes are
//!   settled first, so it is current there; the next save point, a flush's cut or a save, is at
//!   that write or later, and a later write that changes the tile makes it unknown in the book,
//!   which that save point's flush or save then writes over it;
//! - nothing is written while a flush's cut is open: a tile left out of the flush (its copy didn't
//!   encode, its slot still dirty) is in the store's state at the cut but not in the cut's save
//!   point, so a record built from it and placed at the cut would be named stale. The records wait
//!   until the save point is written.

use std::collections::{BTreeMap, BTreeSet};

use geom::{DisplayTile, FinestNormalCone, Level, TileKey};
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

use super::TileStore;
use super::error::CacheError;
use super::read_ahead::Held;
use crate::bounds::{Book, Derived as _, Kept};
use crate::display::{Display, DisplayRecord, StoredDisplay};
use crate::file::{
    BaseFile, FileError, FramedDisplay, ProjectFile, display_record, frame_display,
    unknown_display_len,
};
use crate::meta::Role;
use crate::stamp::Stamp;

/// Whether the display layer the files hold can be trusted, as [`TileStore::load_floor`] finds
/// its floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Floor {
    /// Every floor record the files hold reads, built from the store's finest level (or there
    /// are none): the layer's records are taken as they are, and the missing ones built.
    Current,
    /// A floor record is an intact record of another version, which this build doesn't read
    /// (every file written before the records held normal cones has them). The layer's other
    /// records are likely of that version too, and the jobs, telling known records by their
    /// length alone, would take them as built: the layer is rebuilt whole. (A damaged record is
    /// its own tile's alone: it is found unknown when read, and built.)
    Outdated,
    /// A floor record was built from another finest level: a layer built for another surface,
    /// which is rebuilt whole.
    Foreign,
}

impl TileStore {
    /// [`TileStore::record_built`], then [`TileStore::write_through_display`], as a job's slice
    /// records what the workers built: for tests that record a layer they built themselves.
    ///
    /// # Errors
    ///
    /// As [`TileStore::record_built`], and then nothing is recorded; as
    /// [`TileStore::write_through_display`], and then the tiles are recorded.
    #[cfg(test)]
    pub(crate) fn record_display<T: std::borrow::Borrow<DisplayTile>>(
        &mut self,
        tiles: &[T],
    ) -> Result<(), CacheError> {
        self.record_built(tiles)?;
        self.write_through_display()
    }

    /// Records `tiles`, built from the store's current state, as the display tiles the store
    /// keeps: they go to the working file with the next flush (or sooner, written through, see
    /// [`TileStore::write_through_display`], which the caller runs next), and to the project
    /// file with its next save.
    ///
    /// The writes since the last call are settled first, so the display tiles they made unknown
    /// are replaced, not the other way round.
    ///
    /// # Errors
    ///
    /// [`CacheError::Display`] ([`geom::DisplayError::NoDisplayLevel`]) if the store keeps no
    /// display layer; [`CacheError::Eval`] if listing a written tile's readers fails (never,
    /// for tiles of the store's mesh). Nothing is recorded then.
    pub(super) fn record_built<T: std::borrow::Borrow<DisplayTile>>(
        &mut self,
        tiles: &[T],
    ) -> Result<(), CacheError> {
        let finest = self
            .display_finest
            .ok_or(geom::DisplayError::NoDisplayLevel(Level::ZERO))?;
        self.display_builds += 1;
        let tiles: Vec<&DisplayTile> = tiles.iter().map(std::borrow::Borrow::borrow).collect();
        for &tile in &tiles {
            self.display_generations
                .insert(*tile.key(), self.display_builds);
            self.bumped.remove(tile.key());
            if tile.key().level() <= self.floor_top {
                self.floor
                    .insert(*tile.key(), DisplayRecord::of(tile, finest));
            }
        }
        self.note_epoch_keys(tiles.iter().map(|tile| *tile.key()));
        let records: Vec<(TileKey, DisplayRecord)> = tiles
            .iter()
            .map(|tile| (*tile.key(), DisplayRecord::of(tile, finest)))
            .collect();
        Ok(self
            .display
            .settle_and_record(&self.mesh, &self.deps, &records)?)
    }

    /// The generation of `key`'s display record: the count of [`TileStore::record_built`]
    /// calls when it was last recorded, 0 for a record read from the files and never built
    /// since. A known record is only ever replaced by a build, so two decodes of a key at one
    /// generation give the same tile.
    pub(super) fn display_generation(&self, key: &TileKey) -> u64 {
        self.display_generations.get(key).copied().unwrap_or(0)
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
    /// [`CacheError::Eval`] if listing what the writes since the last settle made unknown fails
    /// (never, for tiles of the store's mesh); [`CacheError::Store`] if a read fails.
    pub fn stored_display(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<DisplayTile>>, CacheError> {
        self.display.settle(&self.mesh, &self.deps)?;
        keys.iter()
            .map(|key| {
                let record = self.display_record(key)?;
                Ok(record.and_then(|r| r.decode(key)))
            })
            .collect()
    }

    /// Loads the floor: every face's display tiles up to the floor's top level the files hold.
    /// Returns whether the layer the files hold can be trusted ([`Floor`]): one built for
    /// another surface, or one holding records this build doesn't read, is rebuilt whole.
    ///
    /// # Errors
    ///
    /// [`CacheError::Store`] if a record can't be read.
    pub(super) fn load_floor(&mut self) -> Result<Floor, CacheError> {
        let Some(finest) = self.display_finest else {
            return Ok(Floor::Current);
        };
        let top = self.floor_top.get().min(finest.get().saturating_sub(1));
        let mut keys = Vec::new();
        for face in self.mesh.faces() {
            for l in 0..=top {
                let level = Level::new(l).map_err(|_level| geom::DisplayError::Shape)?;
                let per_side = u16::try_from(level.tiles_per_side()).unwrap_or(0);
                for ty in 0..per_side {
                    for tx in 0..per_side {
                        keys.push(
                            TileKey::display(face, level, tx, ty)
                                .map_err(|_range| geom::DisplayError::Shape)?,
                        );
                    }
                }
            }
        }
        // Read and decompressed on the workers: thousands of records (14,400 on the benchmark
        // project), each a read and a zstd frame.
        // Nothing is kept in memory yet on open: the files hold every record.
        let (file, base) = (&self.file, self.base.as_ref());
        let records = self.workers.install(|| {
            keys.par_iter()
                .map(|key| Ok((*key, read_display(file, base, key)?)))
                .collect::<Result<Vec<_>, CacheError>>()
        })?;
        let mut state = Floor::Current;
        let unknown_len = unknown_display_len();
        for (key, record) in records {
            match record {
                Some(record) if record.finest == finest => {
                    self.floor.insert(key, record);
                }
                Some(_) => state = Floor::Foreign,
                // Unknown, or a record that doesn't read. One whose length says known is read
                // again (a few, on an open): intact but of another version, the layer's other
                // records are likely of it too, and the jobs, telling known records by their
                // length alone, would count them built. A damaged one is that tile's alone.
                None => {
                    let in_file = file.locate(&key);
                    let located = in_file.or_else(|| base.and_then(|b| b.locate(&key)));
                    if located.is_none_or(|loc| Some(loc.len()) == unknown_len) {
                        continue;
                    }
                    let outdated = match (in_file, base) {
                        (Some(_), _) => file.display_outdated(&key)?,
                        (None, Some(base)) => base.display_outdated(&key)?,
                        (None, None) => false,
                    };
                    if outdated {
                        state = state.max(Floor::Outdated);
                    }
                }
            }
        }
        Ok(state)
    }

    /// The level the store's display layer is built from (its display tiles sit at the levels
    /// below it), or `None` if the store keeps no display layer
    /// ([`StoreConfig::display`](super::StoreConfig::display)).
    #[must_use]
    pub const fn display_finest(&self) -> Option<Level> {
        self.display_finest
    }

    /// The display floor the store keeps: how many tiles, and their encodings' bytes.
    #[must_use]
    pub fn floor_size(&self) -> (usize, usize) {
        (
            self.floor.len(),
            self.floor.values().map(|r| r.encoded.len()).sum(),
        )
    }

    /// Raises the bound of every display tile a dab's writes of the `edited` tiles can change by
    /// twice `max_displacement`, the farthest the dab moved a finest vertex: in memory only,
    /// never written, so the display tiles as they were before the stroke stay drawable, their
    /// bound still a bound, until a refresh replaces them (see [`TileStore::display_for_draw`]).
    ///
    /// A dab turns the finest normals under it by an amount no displacement bounds, so a bumped
    /// tile is drawn with the normal cone that covers everything (never facing away) until the
    /// refresh rebuilds its cone with the rest of it.
    ///
    /// Twice: the finest vertex and the display vertex compared with it can each move that far
    /// (the renderer's stand-ins move display vertices too). Call it after each dab's writes
    /// ([`TileStore::update_tiles`], which settles every write before them), before anything
    /// reads the store's display tiles: the first dab of a stroke keeps each tile as it was
    /// before the stroke. A write since the last bump that isn't one of `edited` (a lost tile's
    /// revert, a write outside the dab) drops the bumps it reaches: those tiles are unknown.
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
        // Changes the bump doesn't cover: the tiles they reach are unknown, never bumped.
        let uncovered: BTreeSet<TileKey> = self.unbumped.difference(&edited).copied().collect();
        self.unbumped.clear();
        let levels = self.display.levels().clone();
        let stale = crate::display::readers(&self.mesh, &self.deps, &uncovered, &levels)?;
        for key in &stale {
            self.bumped.remove(key);
        }
        let readers = crate::display::readers(&self.mesh, &self.deps, &edited, &levels)?;
        let raise = 2.0 * max_displacement.abs();
        self.take_read_ahead();
        let mut reached = BTreeSet::new();
        let mut to_read = Vec::new();
        for key in readers.difference(&stale) {
            if let Some((_, extra)) = self.bumped.get_mut(key) {
                *extra += raise;
                continue;
            }
            // The record as it was before the dab's writes: every write before them is
            // settled, theirs not yet.
            match self.bump_record(key) {
                Held::Record(record) => {
                    self.bumped.insert(*key, (record, raise));
                    reached.insert(*key);
                }
                Held::In(reader, loc) => to_read.push((*key, reader, loc)),
                Held::None => {}
            }
        }
        // Records neither in memory nor read ahead (a stroke's first dab, mostly): read
        // together on the workers, not one after another here.
        let finest = self.display_finest;
        let read: Vec<Result<Option<DisplayRecord>, FileError>> = self.workers.install(|| {
            to_read
                .par_iter()
                .map(|(key, reader, loc)| match reader.read_record(*loc) {
                    Ok(raw) => Ok(display_record(&raw, key).and_then(StoredDisplay::known)),
                    Err(FileError::Damaged { .. }) => Ok(None),
                    Err(e) => Err(e),
                })
                .collect()
        });
        let reads = to_read.iter().zip(read);
        for ((key, _, _), record) in reads {
            if let Some(record) = record?.filter(|record| Some(record.finest) == finest) {
                self.bumped.insert(*key, (record, raise));
                reached.insert(*key);
            }
        }
        // The next dabs reach the tiles around those this one reached first: their records
        // are read on the workers meanwhile.
        if !reached.is_empty() {
            self.read_around(&reached);
        }
        // The next settle makes the same tiles unknown, for the same writes: it takes them
        // from here instead of listing them again.
        self.bump_readers = Some((edited, readers));
        Ok(())
    }

    /// Settles the display book's writes (see [`Book::settle`]), with the tiles the last bump
    /// listed for them if they are the same writes.
    ///
    /// # Errors
    ///
    /// As [`Book::settle`].
    pub(super) fn settle_display(&mut self) -> Result<(), CacheError> {
        if let Some((edited, readers)) = self.bump_readers.take()
            && self.display.settle_listed(&edited, &readers)
        {
            return Ok(());
        }
        Ok(self.display.settle(&self.mesh, &self.deps)?)
    }

    /// Notes a change of `key`'s corrections no bump has covered yet, while any tile is bumped.
    pub(super) fn note_unbumped(&mut self, key: TileKey) {
        if !self.bumped.is_empty() {
            self.unbumped.insert(key);
        }
    }

    /// Drops every bump the corrections changed since the last bump reach: those tiles' bumped
    /// bounds no longer bound them, and they are unknown (see
    /// [`TileStore::bump_display_error`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if listing the changes' readers fails (never, for tiles of the
    /// store's mesh); they stay noted then.
    pub(super) fn drop_uncovered_bumps(&mut self) -> Result<(), CacheError> {
        if self.unbumped.is_empty() {
            return Ok(());
        }
        let stale = crate::display::readers(
            &self.mesh,
            &self.deps,
            &self.unbumped,
            self.display.levels(),
        )?;
        self.unbumped.clear();
        for key in &stale {
            self.bumped.remove(key);
        }
        Ok(())
    }

    /// The display tiles to draw for each of `keys`, in order: the stored tile, or one a stroke
    /// is changing as it was before the stroke with its bound raised and its normal cone the one
    /// that covers everything (see [`TileStore::bump_display_error`]), or a floor tile writes made
    /// unknown as it was last recorded, its bound the largest there is and its cone covering
    /// everything; `None` where none of these is known, and the tile must be drawn finer.
    ///
    /// # Errors
    ///
    /// As [`TileStore::stored_display`].
    pub fn display_for_draw(
        &mut self,
        keys: &[TileKey],
    ) -> Result<Vec<Option<DisplayTile>>, CacheError> {
        self.drop_uncovered_bumps()?;
        self.display.settle(&self.mesh, &self.deps)?;
        keys.iter()
            .map(|key| {
                if let Some((record, extra)) = self.bumped.get(key) {
                    // Rounded up, so it stays a bound.
                    let bound = (record.error + extra).next_up();
                    // A dab turns normals too: the cone before it holds them no longer.
                    let cone = FinestNormalCone::EVERYTHING;
                    return Ok(DisplayTile::decode(*key, bound, cone, &record.encoded).ok());
                }
                if let Some(record) = self.display_record(key)? {
                    return Ok(record.decode(key));
                }
                // A floor tile writes made unknown, and no stroke's bump covers: drawn as it was,
                // its bound the largest there is and its cone covering everything, so a view finer
                // than it is chosen whenever there is one, and never nothing.
                Ok(self.floor.get(key).and_then(|r| {
                    DisplayTile::decode(*key, f32::MAX, FinestNormalCone::EVERYTHING, &r.encoded)
                        .ok()
                }))
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
    pub(super) fn display_record(
        &self,
        key: &TileKey,
    ) -> Result<Option<DisplayRecord>, CacheError> {
        match self.display.kept(key) {
            Some(Kept::Known(record)) => return Ok(Some(record.clone())),
            Some(Kept::Unknown) => return Ok(None),
            Some(Kept::Written) | None => {}
        }
        // A record built from another finest level bounds another surface: unknown here.
        let record = match self.read_ahead_record(key) {
            Some(read) => read.known(),
            None => read_display(&self.file, self.base.as_ref(), key)?,
        };
        Ok(record.filter(|record| Some(record.finest) == self.display_finest))
    }

    /// The display part of a recovery flush, at its cut (see [`TileStore::flush_bounds`]): makes
    /// unknown the display tiles the `lost` tiles' edits could have changed, then appends every
    /// display tile the working file lacks, placed at the store's latest write.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] as for [`TileStore::record_built`]; [`CacheError::WriteBack`] if
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
        self.display_through_failed = false;
        Ok(())
    }

    /// Writes display records through to the working file once those waiting in memory pass
    /// the budget's share for them
    /// ([`StoreConfig::display_waiting_bytes`](super::StoreConfig::display_waiting_bytes)):
    /// as many, in key order, as bring them down to half of it, so the writes come one per half
    /// a share recorded, not one per slice. They are compressed and framed on the workers, then
    /// appended with one write as a flush appends them, placed at the store's latest write, and
    /// leave memory as a flush's do (see the module notes). Settles the writes first.
    ///
    /// Writes nothing, and the records wait in memory past the share, while a recovery flush's
    /// cut is open, while the working file is a failed recovery file (whose records a roll-over
    /// would have to read back) or one opened read-only, and after a write through failed, until
    /// the next flush or save of the working file (which meets the failure itself, and reports
    /// it) or a new working file.
    ///
    /// # Errors
    ///
    /// [`CacheError::Eval`] if settling the writes fails (never, for tiles of the store's mesh).
    /// A failed write is no error here (see above).
    pub(super) fn write_through_display(&mut self) -> Result<(), CacheError> {
        let share = self.display_waiting_share;
        if self.display.waiting_bytes() <= share
            || self.display_through_failed
            || self.file.cut_at().is_some()
            || self.working_failed
            || !self.file.is_writable()
        {
            return Ok(());
        }
        self.settle_display()?;
        let mut over = self.display.waiting_bytes().saturating_sub(share / 2);
        let mut records: Vec<(TileKey, &DisplayRecord)> = Vec::new();
        for (key, record) in self.display.waiting() {
            if over == 0 {
                break;
            }
            over = over.saturating_sub(Display::bytes(record));
            records.push((*key, record));
        }
        // Compressing them is the costly part: a few milliseconds a megabyte on one thread.
        let framed: Result<Vec<FramedDisplay>, FileError> = self.workers.install(|| {
            records
                .par_iter()
                .map(|&(key, record)| frame_display(key, Some(record)))
                .collect()
        });
        let keys: Vec<TileKey> = records.iter().map(|&(key, _)| key).collect();
        let appended =
            framed.and_then(|framed| self.file.append_display_framed(&framed, self.last_write));
        if appended.is_err() {
            self.display_through_failed = true;
            return Ok(());
        }
        for key in &keys {
            self.display.flushed(key);
        }
        Ok(())
    }

    /// The bytes of the display records waiting in memory for the working file, and the share
    /// of the budget they are held to (see [`TileStore::write_through_display`]).
    #[cfg(test)]
    pub(crate) const fn display_waiting(&self) -> (usize, usize) {
        (self.display.waiting_bytes(), self.display_waiting_share)
    }

    /// Holds the display records waiting for the working file to `bytes` instead of the
    /// budget's share: for tests of the writing through.
    #[cfg(test)]
    pub(crate) const fn hold_display_waiting_to(&mut self, bytes: usize) {
        self.display_waiting_share = bytes;
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
        self.display_through_failed = false;
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
    /// [`CacheError::Eval`] as for [`TileStore::record_built`]; [`CacheError::Save`] if a write
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

/// The display book of a store over `working` and `base` keeping a display layer under
/// `finest` (none if `None`), on a mesh of `faces` faces.
///
/// Its levels are the store's, every one below `finest`, whatever keys the files hold, so a
/// write always makes unknown the display tiles it changes. A display key the layer doesn't
/// have (a level at or above `finest`, a face the mesh doesn't) is left out, as if absent. For
/// a recovery file, the working file's display tiles are changes since the last save, held by
/// it. A store keeping no display layer can't keep one up to date: every display tile the files
/// hold is unknown to it, so its next save names none.
pub(super) fn display_book_of(
    working: &ProjectFile,
    base: Option<&BaseFile>,
    finest: Option<Level>,
    faces: u32,
) -> Book<Display> {
    let on_mesh = |key: &&TileKey| key.face().index() < faces;
    let Some(finest) = finest else {
        let all: Vec<TileKey> = working
            .display_keys()
            .chain(base.into_iter().flat_map(BaseFile::display_keys))
            .filter(on_mesh)
            .copied()
            .collect();
        return Book::all_unknown(all);
    };
    let levels: BTreeSet<Level> = (0..finest.get())
        .filter_map(|l| Level::new(l).ok())
        .collect();
    let changed: BTreeMap<TileKey, Kept<DisplayRecord>> = match working.meta().role() {
        Role::Recovery { .. } => working
            .display_keys()
            .filter(on_mesh)
            .filter(|key| key.level() < finest)
            .map(|key| (*key, Kept::Written))
            .collect(),
        Role::Project => BTreeMap::new(),
    };
    Book::with(levels, changed)
}

/// `key`'s display record as the files hold it: the working file's, else the base's; `None`
/// where neither knows it.
///
/// # Errors
///
/// [`CacheError::Store`] if a read fails.
pub(super) fn read_display(
    file: &ProjectFile,
    base: Option<&BaseFile>,
    key: &TileKey,
) -> Result<Option<DisplayRecord>, CacheError> {
    if let Some(found) = file.read_display(key)? {
        return Ok(found.known());
    }
    Ok(match base {
        Some(base) => base.read_display(key)?.and_then(StoredDisplay::known),
        None => None,
    })
}
