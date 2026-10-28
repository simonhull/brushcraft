//! Display records in the file: tile records of the display channel (see [`crate::display`]),
//! read as derived data (one that doesn't read counts as unknown) and written many at a time.

use std::os::unix::fs::FileExt as _;

use geom::TileKey;

use super::base::BaseFile;
use super::format::{tile_payload, tile_record};
use super::{FileError, Loc, Name, ProjectFile};
use crate::codec::{compress, decompress};
use crate::display::{
    DISPLAY_OP, DisplayRecord, StoredDisplay, display_body, display_header, is_display, max_body,
    parse_display,
};
use crate::record::{KIND_TILE, frame};
use crate::stamp::Stamp;

/// What a raw record holds as `key`'s display tile.
///
/// `None` if it doesn't verify as a display record of `key` this build reads (it counts as
/// unknown). The body is decompressed only if its stated size is plausible for the key's level
/// ([`max_body`]).
pub(crate) fn display_record(raw: &[u8], key: &TileKey) -> Option<StoredDisplay> {
    let tile = tile_record(raw, key)?;
    if tile.raw_len > max_body(key) {
        return None;
    }
    let body = decompress(tile.packed, tile.raw_len).ok()?;
    parse_display(key, tile.header, &body)
}

impl ProjectFile {
    /// `key`'s display tile as its latest record holds it: `None` if it has no record, else the
    /// record, unknown if it says so or doesn't read (a display tile is derived, so damage only
    /// means it is rebuilt).
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    pub(crate) fn read_display(&self, key: &TileKey) -> Result<Option<StoredDisplay>, FileError> {
        let Some(&Loc { offset, len }) = self.live.get(&Name::Tile(*key)) else {
            return Ok(None);
        };
        match self.read_record(offset, len) {
            Ok(raw) => Ok(Some(
                display_record(&raw, key).unwrap_or(StoredDisplay::Unknown),
            )),
            Err(FileError::Damaged { .. }) => Ok(Some(StoredDisplay::Unknown)),
            Err(e) => Err(e),
        }
    }

    /// The display tiles the file holds a record of, in key order.
    pub(crate) fn display_keys(&self) -> impl Iterator<Item = &TileKey> {
        self.keys().filter(|key| is_display(key))
    }

    /// Appends a display record for each of `tiles` (`None`: unknown), in order, framed into one
    /// buffer and written with one write: each is its key's latest record, placed against an
    /// open cut by `stamp`, as a bound is. The records carry [`DISPLAY_OP`], and no op is noted:
    /// they move no head.
    ///
    /// If the write fails, none of them is recorded (the bytes written past the end, if any, are
    /// overwritten by the next append).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`].
    pub(crate) fn write_display_many<'a>(
        &mut self,
        tiles: impl IntoIterator<Item = (TileKey, Option<&'a DisplayRecord>)>,
        stamp: Stamp,
    ) -> Result<(), FileError> {
        if self.access == super::lock::Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let size = |n: usize| u64::try_from(n).or(Err(FileError::TooLarge));
        let mut buffer = Vec::new();
        let mut placed = Vec::new();
        for (key, record) in tiles {
            let body = display_body(record);
            let packed = compress(body)?;
            let payload = tile_payload(
                key,
                DISPLAY_OP,
                &display_header(record),
                body.len(),
                &packed,
            )?;
            let framed = frame(KIND_TILE, &payload).ok_or(FileError::TooLarge)?;
            placed.push((key, size(buffer.len())?, size(framed.len())?));
            buffer.extend_from_slice(&framed);
        }
        if buffer.is_empty() {
            return Ok(());
        }
        let start = self.end;
        self.file.write_all_at(&buffer, start)?;
        self.end += size(buffer.len())?;
        for (key, at, len) in placed {
            let loc = Loc {
                offset: start + at,
                len,
            };
            self.note(Name::Tile(key), None, stamp, loc);
        }
        Ok(())
    }

    /// Appends `raw`, a display record of `key` read from another file, as `key`'s latest record
    /// if it reads as one; else an unknown record in its place (a display tile is derived, and
    /// must never stop a save).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`].
    pub(crate) fn append_display_copy(
        &mut self,
        key: &TileKey,
        raw: &[u8],
    ) -> Result<(), FileError> {
        if display_record(raw, key).is_some() {
            if self.access == super::lock::Access::ReadOnly {
                return Err(FileError::ReadOnly);
            }
            let loc = self.append_record(raw)?;
            self.note(Name::Tile(*key), None, Stamp::ZERO, loc);
            Ok(())
        } else {
            self.write_display_many([(*key, None)], Stamp::ZERO)
        }
    }
}

impl BaseFile {
    /// `key`'s display tile as the save point holds it: as [`ProjectFile::read_display`].
    ///
    /// # Errors
    ///
    /// [`FileError::Closed`] once the project file is dropped; [`FileError::Io`].
    pub(crate) fn read_display(&self, key: &TileKey) -> Result<Option<StoredDisplay>, FileError> {
        let Some(loc) = self.locate(key) else {
            return Ok(None);
        };
        match self.reader.read_record(loc) {
            Ok(raw) => Ok(Some(
                display_record(&raw, key).unwrap_or(StoredDisplay::Unknown),
            )),
            Err(FileError::Damaged { .. }) => Ok(Some(StoredDisplay::Unknown)),
            Err(e) => Err(e),
        }
    }

    /// The display tiles the save point holds a record of, in key order.
    pub(crate) fn display_keys(&self) -> impl Iterator<Item = &TileKey> {
        self.keys().filter(|key| is_display(key))
    }
}

/// The length of an unknown display record, framing included.
///
/// The same for every key (a key, an op, a two-byte header and an empty body), so a store can
/// tell unknown records from the file's map alone. `None` if the empty body won't compress
/// (never).
pub(crate) fn unknown_display_len() -> Option<u64> {
    let key = TileKey::display(geom::FaceId::new(0), geom::Level::ZERO, 0, 0).ok()?;
    let packed = compress(&[]).ok()?;
    let payload = tile_payload(key, DISPLAY_OP, &display_header(None), 0, &packed).ok()?;
    u64::try_from(frame(KIND_TILE, &payload)?.len()).ok()
}
