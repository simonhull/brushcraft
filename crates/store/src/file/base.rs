//! The base under a working file: the project file as of its last save, read where the working
//! file has no record.

use geom::TileKey;

use super::error::FileError;
use super::reader::{FileReader, ReadScratch};
use super::{Loc, read_bounds_at};
use crate::OpId;
use crate::meta::{Meta, SavePoint};
use crate::shared_map::SharedMap;
use crate::slot::CorrectionSlot;
use crate::tile_bounds::TileBounds;

/// A project file as of a save point, lent to a [`crate::TileStore`] as the read-only layer under
/// its working file: corrections the working file doesn't hold are read from here. Made by
/// [`ProjectFile::base`](super::ProjectFile::base).
///
/// It holds a reader of the open file, a frozen map of where each saved tile's record is, and
/// what the store checks it against: the file's meta and the save point, which a working file
/// layered on it must name. It reads only while its [`ProjectFile`](super::ProjectFile) stays open.
#[derive(Debug, Clone)]
pub(crate) struct BaseFile {
    /// Reads the project file while its owner keeps it open.
    pub(super) reader: FileReader,
    /// Each saved tile's record as of the save point. Records are never moved while the file
    /// is open, so these stay valid as the owner appends. Shared with the file's own record of
    /// the save point, and with the bases of the save points before and after it, where they
    /// agree (see [`SavedState`](super::SavedState)).
    pub(super) locations: SharedMap<TileKey, Loc>,
    /// Each saved bound's record as of the save point.
    pub(super) bounds: SharedMap<TileKey, Loc>,
    /// Each stroke record the save point holds (every one before its Index), by offset.
    pub(super) strokes: SharedMap<u64, Loc>,
    /// The project file's meta: its project, its role and its mesh.
    pub(super) meta: Meta,
    /// The save point: the project, the Index's offset and the head op it recorded.
    pub(super) save_point: SavePoint,
    /// The newest edit in the whole file when the base was made, records after the save point
    /// included: a store's op ids continue after it, so no op is handed out twice in one file.
    pub(super) head: Option<OpId>,
    /// The save point this one was derived from in memory, with records only put in (see
    /// [`SavedState::parent`](super::SavedState::parent)); `None` for one loaded from the file.
    pub(super) extends: Option<SavePoint>,
}

impl BaseFile {
    /// Where `key`'s saved record is, if the save point has one.
    pub(crate) fn locate(&self, key: &TileKey) -> Option<Loc> {
        self.locations.get(key).copied()
    }

    /// Whether the save point holds a record of `key`.
    pub(crate) fn contains(&self, key: &TileKey) -> bool {
        self.locations.contains_key(key)
    }

    /// The reader the store's prefetch workers read the base through.
    pub(crate) fn reader(&self) -> FileReader {
        self.reader.clone()
    }

    /// Reads `key`'s displacement tile recorded at `loc` into `slot` (see
    /// [`FileReader::read_displacement_into`]).
    ///
    /// # Errors
    ///
    /// As [`FileReader::read_displacement_into`]; [`FileError::Closed`] once the project file
    /// is dropped. On error `slot` holds no tile.
    pub(crate) fn read_at(
        &self,
        key: &TileKey,
        loc: Loc,
        slot: &mut CorrectionSlot,
        scratch: &mut ReadScratch,
    ) -> Result<OpId, FileError> {
        self.reader.read_displacement_into(key, loc, slot, scratch)
    }

    /// The project file's meta.
    pub(crate) const fn meta(&self) -> &Meta {
        &self.meta
    }

    /// The save point the base is: what a recovery file layered on it names in its meta.
    pub(crate) const fn save_point(&self) -> SavePoint {
        self.save_point
    }

    /// The newest edit in the project file when the base was made, including any written after
    /// the save point; `None` before any edit. Op ids for new edits start after it.
    pub(crate) const fn head(&self) -> Option<OpId> {
        self.head
    }

    /// The saved tiles, in key order.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.locations.keys()
    }

    /// How many tiles the save point holds.
    pub(crate) const fn len(&self) -> usize {
        self.locations.len()
    }

    /// Whether this save point is known to hold every tile and bound `older` does, without
    /// looking at them: it was derived in the same open [`ProjectFile`](super::ProjectFile) from
    /// `older`'s save point by putting records in (see [`SavedState`](super::SavedState)), so
    /// its Index's parent is `older`'s Index (the same project, offset and head op), and
    /// nothing was taken out on the way.
    ///
    /// `false` proves nothing either way: a save point loaded from the file, one two or more
    /// saves on, or one of another file (a project saved under a new name) may still hold
    /// them all, and has to be checked key by key.
    pub(crate) fn extends(&self, older: &Self) -> bool {
        self.extends == Some(older.save_point) && self.reader.same_file(&older.reader)
    }

    /// Whether `other` is the same save point with the same records: every tile's and every
    /// bound's, bit for bit, however each was made.
    #[cfg(test)]
    pub(crate) fn same_state(&self, other: &Self) -> bool {
        self.save_point == other.save_point
            && self.locations.same_entries(&other.locations)
            && self.bounds.same_entries(&other.bounds)
    }

    /// Where `key`'s saved bounds record is, if the save point has one.
    pub(crate) fn locate_bounds(&self, key: &TileKey) -> Option<Loc> {
        self.bounds.get(key).copied()
    }

    /// Where each saved bound's record is, in key order.
    pub(crate) fn bounds_locations(&self) -> impl Iterator<Item = (&TileKey, Loc)> {
        self.bounds.iter().map(|(key, &loc)| (key, loc))
    }

    /// Where each stroke record the save point holds is, in file order.
    pub(crate) fn stroke_locations(&self) -> impl Iterator<Item = Loc> + '_ {
        self.strokes.iter().map(|(_, &loc)| loc)
    }

    /// The bounds the records at `entries` hold (see [`read_bounds_at`]).
    ///
    /// # Errors
    ///
    /// [`FileError::Closed`] once the project file is dropped; [`FileError::Io`].
    pub(crate) fn read_bounds(
        &self,
        entries: &[(TileKey, Loc)],
    ) -> Result<Vec<Option<TileBounds>>, FileError> {
        let file = self.reader.file.upgrade().ok_or(FileError::Closed)?;
        Ok(read_bounds_at(&file, entries)?)
    }
}
