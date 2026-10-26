//! Reading tile records off the thread that owns the file: a shared reader, a syncer, framed
//! records, and the scratch buffers a read decodes into.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Weak;

use geom::TileKey;

use super::error::FileError;
use super::format::{MAX_RAW, read_exact, tile_payload, tile_record};
use super::lock::LockedFile;
use super::{Loc, check_displacement};
use crate::OpId;
use crate::codec::{
    CodecError, compress, decode_displacement_into, displacement_body_len, encode_displacement,
};
use crate::record::{KIND_TILE, frame};
use crate::slot::CorrectionSlot;
use crate::stamp::Stamp;

/// Reads tiles of an open [`ProjectFile`](super::ProjectFile) from any thread.
///
/// It reads at locations the owner looked up ([`ProjectFile::locate`](super::ProjectFile::locate)).
/// The file is append-only while open, so a location stays valid. A reader doesn't keep the file
/// open: once its [`ProjectFile`](super::ProjectFile) is dropped, the file closes and its lock is
/// released (after any read in flight, which holds the file for its duration), and reads fail with
/// [`FileError::Closed`].
#[derive(Debug, Clone)]
pub(crate) struct FileReader {
    /// The owner's open file, upgraded for the length of each read.
    pub(super) file: Weak<LockedFile>,
}

impl FileReader {
    /// Whether `other` reads the same open file: the same [`ProjectFile`](super::ProjectFile), not
    /// merely the same path. Each holds the file weakly, which keeps its allocation, so readers
    /// of two files opened apart never compare equal, even after one is closed.
    pub(crate) fn same_file(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.file, &other.file)
    }

    /// The whole record at `loc`, framing included, as it is in the file: for copying it into
    /// another. Not checked here; the caller verifies it.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if the file ends first, [`FileError::Closed`] if the
    /// [`ProjectFile`](super::ProjectFile) has been dropped, [`FileError::Io`].
    pub(crate) fn read_record(&self, Loc { offset, len }: Loc) -> Result<Vec<u8>, FileError> {
        let file = self.file.upgrade().ok_or(FileError::Closed)?;
        match read_exact(&file, offset, len) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                Err(FileError::Damaged { offset })
            }
            other => Ok(other?),
        }
    }

    /// Reads the displacement tile of `key` recorded at `loc` straight into `slot`, and returns
    /// the edit that wrote it. Allocates nothing once `scratch` has grown to the largest record.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if the record fails its checksum, isn't a tile of `key` or doesn't
    /// decompress; [`FileError::NotDisplacement`], [`FileError::WrongSide`],
    /// [`FileError::Codec`] as for
    /// [`ProjectFile::read_displacement`](super::ProjectFile::read_displacement);
    /// [`FileError::Closed`] if the [`ProjectFile`](super::ProjectFile) has been dropped;
    /// [`FileError::Io`]. On error `slot` holds no tile.
    pub(crate) fn read_displacement_into(
        &self,
        key: &TileKey,
        loc: Loc,
        slot: &mut CorrectionSlot,
        scratch: &mut ReadScratch,
    ) -> Result<OpId, FileError> {
        let file = check_displacement(key, key.side())
            .and_then(|()| self.file.upgrade().ok_or(FileError::Closed));
        match file {
            Ok(file) => read_displacement_at(&file, key, loc, slot, scratch),
            Err(e) => {
                bytemuck::bytes_of_mut(slot).fill(0);
                Err(e)
            }
        }
    }
}

/// Syncs an open [`ProjectFile`](super::ProjectFile) from another thread: the recovery writer's way
/// of making appended records durable without blocking the thread that owns the file.
///
/// Like a [`FileReader`], it holds the file weakly and doesn't keep it (or its lock) open after
/// its [`ProjectFile`](super::ProjectFile) is dropped; a sync in flight holds the file until it
/// returns.
#[derive(Debug, Clone)]
pub(crate) struct FileSyncer {
    /// The owner's open file, upgraded for the length of each sync.
    pub(super) file: Weak<LockedFile>,
}

impl FileSyncer {
    /// Makes every record written to the file so far durable (`fdatasync`): their data, and the
    /// file length that reaches them.
    ///
    /// # Errors
    ///
    /// [`FileError::Closed`] if the [`ProjectFile`](super::ProjectFile) has been dropped;
    /// [`FileError::Io`] if the sync fails, after which what was written since the last good
    /// sync may not be on disk.
    pub(crate) fn sync_data(&self) -> Result<(), FileError> {
        let file = self.file.upgrade().ok_or(FileError::Closed)?;
        file.sync_data()?;
        Ok(())
    }
}

/// A tile record, compressed and framed, ready for
/// [`ProjectFile::append_framed`](super::ProjectFile::append_framed): the
/// encoding [`ProjectFile::write_displacement`](super::ProjectFile::write_displacement) does, done
/// ahead of the append (on the recovery writer's thread).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FramedTile {
    /// The tile.
    pub(super) key: TileKey,
    /// The edit the record is written with.
    pub(super) op: OpId,
    /// The stamp of the store's write the record copies: what places it against an open cut,
    /// and tells whether its slot was written again since.
    pub(super) stamp: Stamp,
    /// The whole record, framing and checksum included.
    pub(super) record: Vec<u8>,
}

impl FramedTile {
    /// The tile the record holds.
    pub(crate) const fn key(&self) -> TileKey {
        self.key
    }

    /// The edit the record is written with.
    pub(crate) const fn op(&self) -> OpId {
        self.op
    }

    /// The stamp of the store's write the record copies.
    pub(crate) const fn stamp(&self) -> Stamp {
        self.stamp
    }
}

/// Encodes, compresses and frames the displacement tile in `slot` as `key`'s record, written
/// by `op`: the record [`ProjectFile::write_displacement`](super::ProjectFile::write_displacement)
/// would append. `stamp` is the store's write the slot was copied at.
///
/// # Errors
///
/// [`FileError::NotDisplacement`] if `key` is of another channel, [`FileError::WrongSide`] if
/// the slot's side isn't the key's, [`FileError::Codec`] if the slot holds no valid tile,
/// [`FileError::TooLarge`], [`FileError::Io`] if compression fails.
pub(crate) fn frame_displacement(
    key: TileKey,
    op: OpId,
    stamp: Stamp,
    slot: &CorrectionSlot,
) -> Result<FramedTile, FileError> {
    let tile = slot.to_tile().map_err(CodecError::from)?;
    check_displacement(&key, tile.side())?;
    let (header, body) = encode_displacement(&tile)?;
    if !u32::try_from(body.len()).is_ok_and(|n| n <= MAX_RAW) {
        return Err(FileError::TooLarge);
    }
    let packed = compress(&body)?;
    let payload = tile_payload(key, op, &header, body.len(), &packed)?;
    let record = frame(KIND_TILE, &payload).ok_or(FileError::TooLarge)?;
    Ok(FramedTile {
        key,
        op,
        stamp,
        record,
    })
}

/// Reusable buffers for reading tiles: the record, its decompressed body and a zstd context.
/// One per reading thread.
pub(crate) struct ReadScratch {
    /// The record as read from the file, framing included; grown to the largest record read.
    record: Vec<u8>,
    /// The decompressed body; grown to the largest body read.
    raw: Vec<u8>,
    /// The decompression context, kept so each read needn't make one.
    zstd: zstd::bulk::Decompressor<'static>,
}

impl ReadScratch {
    /// Empty buffers and a fresh zstd context.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`] if zstd can't allocate its context.
    pub(crate) fn new() -> Result<Self, FileError> {
        Ok(Self {
            record: Vec::new(),
            raw: Vec::new(),
            zstd: zstd::bulk::Decompressor::new()?,
        })
    }
}

impl std::fmt::Debug for ReadScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadScratch")
            .field("record", &self.record.capacity())
            .field("raw", &self.raw.capacity())
            .finish_non_exhaustive()
    }
}

/// Reads the displacement record of `key` at `loc` into `slot`: verifies it, checks the header's
/// side and stated size before decompressing, then decodes and validates.
///
/// # Errors
///
/// As [`FileReader::read_displacement_into`]. On error `slot` holds no tile.
pub(super) fn read_displacement_at(
    file: &File,
    key: &TileKey,
    loc: Loc,
    slot: &mut CorrectionSlot,
    scratch: &mut ReadScratch,
) -> Result<OpId, FileError> {
    let read = decode_record_at(file, key, loc, slot, scratch);
    if read.is_err() {
        bytemuck::bytes_of_mut(slot).fill(0);
    }
    read
}

/// [`read_displacement_at`] without clearing `slot` on error.
///
/// # Errors
///
/// As [`read_displacement_at`].
fn decode_record_at(
    file: &File,
    key: &TileKey,
    Loc { offset, len }: Loc,
    slot: &mut CorrectionSlot,
    scratch: &mut ReadScratch,
) -> Result<OpId, FileError> {
    let ReadScratch { record, raw, zstd } = scratch;
    let damaged = FileError::Damaged { offset };
    record.resize(usize::try_from(len).or(Err(FileError::TooLarge))?, 0);
    match file.read_exact_at(record, offset) {
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Err(damaged),
        other => other?,
    }
    let tile = tile_record(record, key).ok_or(damaged)?;
    let (side, body_len) = displacement_body_len(tile.header)?;
    check_displacement(key, side)?;
    if tile.raw_len != body_len {
        return Err(CodecError::Size.into());
    }
    raw.resize(body_len, 0);
    match zstd.decompress_to_buffer(tile.packed, raw.as_mut_slice()) {
        Ok(n) if n == body_len => {}
        _ => return Err(FileError::Damaged { offset }),
    }
    decode_displacement_into(tile.header, raw, slot)?;
    Ok(tile.op)
}
