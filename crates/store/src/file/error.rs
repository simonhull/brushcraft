//! Why a project file operation fails: [`FileError`].

use std::io;

use geom::MeshError;
use thiserror::Error;

use crate::codec::CodecError;

/// Why a project file operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FileError {
    /// An I/O call failed.
    #[error("project file I/O failed")]
    Io(#[from] io::Error),
    /// Not a BrushCraft project, or a version or flags this build can't read; or its leading
    /// `Meta` or `Mesh` record is damaged, which is treated like a damaged header (the file is
    /// left untouched).
    #[error(
        "not a BrushCraft project file this build can read (bad magic, version, flags, or a \
         damaged meta or mesh record)"
    )]
    BadHeader,
    /// The file doesn't begin with a `Meta` record followed by a `Mesh` record.
    #[error("the file doesn't begin with its meta and mesh records")]
    MissingMeta,
    /// The base mesh doesn't match the meta's mesh hash: in a file's leading records, or a mesh
    /// and meta given to [`ProjectFile::create`](super::ProjectFile::create).
    #[error("the base mesh doesn't match the mesh hash of the file's meta")]
    MeshMismatch,
    /// The `Mesh` record matches its hash but isn't a valid base mesh.
    #[error("the file's base mesh is not valid")]
    BadMesh(#[source] MeshError),
    /// A record's checksum or structure is wrong.
    #[error("the record at byte {offset} is damaged")]
    Damaged {
        /// Where the record starts.
        offset: u64,
    },
    /// Recovery stopped at byte `offset`, but a good save point (Index) follows, or the Index
    /// it stopped at verifies but doesn't load: the file is damaged there, not torn at its end.
    /// It was left untouched.
    #[error("the project file is damaged at byte {offset}, before later good records")]
    DamagedMidFile {
        /// Where recovery stopped.
        offset: u64,
    },
    /// `open_at` was given an offset that isn't a good Index.
    #[error("byte {0} is not a save point")]
    NotASavePoint(u64),
    /// The file was opened read-only (at an earlier save point, or without write permission)
    /// and can't be written.
    #[error("the project file is open read-only")]
    ReadOnly,
    /// Records follow the last save point from byte `offset` (a torn tail, or an operation that
    /// never finished), which must be cut off, but the file can't be written. It was left
    /// untouched.
    #[error("the project file needs its torn tail from byte {offset} cut off, but is read-only")]
    NeedsRepair {
        /// Where the torn tail starts.
        offset: u64,
    },
    /// Another open of the file holds a conflicting lock: one writer, or any number of readers.
    #[error("the project file is open elsewhere")]
    Locked,
    /// The project file a reader was made from has been closed (its owner dropped).
    #[error("the project file has been closed")]
    Closed,
    /// A payload doesn't fit the format's length fields, or a tile body is larger than
    /// [`MAX_RAW`](super::MAX_RAW).
    #[error("record payload too large")]
    TooLarge,
    /// A tile payload didn't encode or decode.
    #[error("tile payload didn't encode or decode")]
    Codec(#[from] CodecError),
    /// A displacement read or write named a key of another channel.
    #[error("channel {0} is not displacement")]
    NotDisplacement(u8),
    /// A displacement tile's side isn't the side of its key's level.
    #[error("a displacement tile of side {tile} under a key of side {key}")]
    WrongSide {
        /// The key's side.
        key: u32,
        /// The tile's side.
        tile: u32,
    },
    /// A full Index would name a different number of records than the save point derived in
    /// memory (from the last one and the records written since) holds: a bug in deriving it,
    /// not an I/O failure. Nothing was written, and the file's last save point stands.
    ///
    /// Trying the save again doesn't help: the same derivation fails the same way. A
    /// [`crate::TileStore`] reports it as [`crate::Invariant::SavePointMismatch`], never as a
    /// save that can be retried.
    #[error("a full save point names {named} records, but its derived maps hold {derived}")]
    SavePointMismatch {
        /// The records the full Index would name.
        named: usize,
        /// The records the derived maps hold.
        derived: usize,
    },
}
