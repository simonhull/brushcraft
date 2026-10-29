//! Display records in the file: tile records of the display channel (see [`crate::display`]),
//! read as derived data (one that doesn't read counts as unknown) and written many at a time.

use std::os::unix::fs::FileExt as _;

use geom::TileKey;

use super::base::BaseFile;
use super::format::{tile_payload, tile_record};
use super::{FileError, Loc, Name, ProjectFile};
use crate::codec::{compress, decompress};
use crate::display::{
    DISPLAY_OP, DisplayRecord, StoredDisplay, display_body, display_header, is_current_version,
    is_display, max_body, parse_display,
};
use crate::record::{KIND_TILE, frame};
use crate::stamp::Stamp;

/// `key`'s display record holding `record` (`None`: unknown), compressed and framed.
///
/// Ready to append ([`ProjectFile::append_display_framed`]): the costly part of writing one,
/// which needs no file, so it can run on any thread.
///
/// # Errors
///
/// [`FileError::TooLarge`], or [`FileError::Io`] if the compressor fails.
pub(crate) fn frame_display(
    key: TileKey,
    record: Option<&DisplayRecord>,
) -> Result<FramedDisplay, FileError> {
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
    Ok(FramedDisplay { key, framed })
}

/// A display record framed by [`frame_display`], waiting to be appended.
#[derive(Debug)]
pub(crate) struct FramedDisplay {
    /// The display tile it is the record of.
    key: TileKey,
    /// The whole record, framing included.
    framed: Vec<u8>,
}

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
    /// Appends `raw` as `key`'s latest display record, as it is, whatever it holds: for tests
    /// planting records an older build wrote ([`ProjectFile::append_display_copy`] writes an
    /// unknown one in place of one that doesn't read).
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::append_display_copy`].
    #[cfg(test)]
    pub(crate) fn append_display_raw(
        &mut self,
        key: &TileKey,
        raw: &[u8],
    ) -> Result<(), FileError> {
        let loc = self.append_record(raw)?;
        self.note(Name::Tile(*key), None, Stamp::ZERO, loc);
        Ok(())
    }

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

    /// Whether `key`'s latest record is an intact display record of another version than this
    /// build's ([`is_outdated`]); `false` if there is none, or it is damaged.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    pub(crate) fn display_outdated(&self, key: &TileKey) -> Result<bool, FileError> {
        let Some(Loc { offset, len }) = self.live.get(&Name::Tile(*key)).copied() else {
            return Ok(false);
        };
        match self.read_record(offset, len) {
            Ok(raw) => Ok(is_outdated(&raw, key)),
            Err(FileError::Damaged { .. }) => Ok(false),
            Err(e) => Err(e),
        }
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
        let framed: Result<Vec<FramedDisplay>, FileError> = tiles
            .into_iter()
            .map(|(key, record)| frame_display(key, record))
            .collect();
        self.append_display_framed(&framed?, stamp)
    }

    /// Appends the records [`frame_display`] framed, in order, with one write, as
    /// [`ProjectFile::write_display_many`] does once it has framed them.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::write_display_many`].
    pub(crate) fn append_display_framed(
        &mut self,
        framed: &[FramedDisplay],
        stamp: Stamp,
    ) -> Result<(), FileError> {
        if self.access == super::lock::Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let size = |n: usize| u64::try_from(n).or(Err(FileError::TooLarge));
        let total: usize = framed.iter().map(|f| f.framed.len()).sum();
        let mut buffer = Vec::with_capacity(total);
        let mut placed = Vec::with_capacity(framed.len());
        for record in framed {
            placed.push((record.key, size(buffer.len())?, size(record.framed.len())?));
            buffer.extend_from_slice(&record.framed);
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

    /// Whether `key`'s record is an intact display record of another version: as
    /// [`ProjectFile::display_outdated`].
    ///
    /// # Errors
    ///
    /// [`FileError::Closed`] once the project file is dropped; [`FileError::Io`].
    pub(crate) fn display_outdated(&self, key: &TileKey) -> Result<bool, FileError> {
        let Some(loc) = self.locate(key) else {
            return Ok(false);
        };
        match self.reader.read_record(loc) {
            Ok(raw) => Ok(is_outdated(&raw, key)),
            Err(FileError::Damaged { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Whether `raw`, a record that verified, is a tile record of `key` whose display header is of
/// another version than this build's: written by another build, intact, and unread here.
fn is_outdated(raw: &[u8], key: &TileKey) -> bool {
    tile_record(raw, key).is_some_and(|tile| !is_current_version(tile.header))
}

/// `key`'s display record as version 2 wrote it, framed: the finest level and the bound of
/// `record` (known) or nothing (unknown), no normal cone. For tests of how a file from before
/// the cones reads.
///
/// # Panics
///
/// Never: a display record of a tile [`max_body`] holds always frames.
#[cfg(test)]
pub(crate) fn version_2_record(key: TileKey, record: Option<&DisplayRecord>) -> Vec<u8> {
    let (header, body): (Vec<u8>, &[u8]) = match record {
        Some(record) => {
            let mut header = vec![2, 1, record.finest.get()];
            header.extend_from_slice(&record.error.to_le_bytes());
            (header, &record.encoded)
        }
        None => (vec![2, 0], &[]),
    };
    let packed = compress(body).expect("a display body compresses");
    let payload =
        tile_payload(key, DISPLAY_OP, &header, body.len(), &packed).expect("a display payload");
    frame(KIND_TILE, &payload).expect("a display record frames")
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
