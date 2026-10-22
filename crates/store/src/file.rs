//! The project file: a header, a `Meta` and a `Mesh` record that make it self-contained, then
//! append-only records (see `record`), with save points that form a chain.
//!
//! The two leading records are part of the file's identity, like the header: they are written
//! and synced when the file is created and never change. Recovery after a crash starts after
//! them, and damage to them fails the open rather than being cut off as a torn tail.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, TryLockError};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use geom::{BaseMesh, Channel, CorrectionTile, MeshError, TileKey};
use thiserror::Error;

use crate::OpId;
use crate::bytes::{KEY_LEN, Reader, put_key};
use crate::codec::{
    CodecError, compress, decode_displacement_into, decompress, displacement_body_len,
    encode_displacement,
};
use crate::meta::{Meta, mesh_hash};
use crate::record::{
    HEAD, Head, KIND_INDEX, KIND_MESH, KIND_META, KIND_TILE, TAIL, frame, parse_head, verify,
};
use crate::slot::CorrectionSlot;

/// The file's first bytes.
const MAGIC: [u8; 4] = *b"BCRF";
/// The format version this build reads and writes.
const VERSION: u16 = 1;
/// File header bytes: magic, version `u16`, flags `u16`, reserved `u64`.
pub(crate) const HEADER_LEN: usize = 16;
/// Offset of the first record: the `Meta` record.
const FIRST_RECORD: u64 = 16;
/// Every this-many-th index is full; the rest are deltas.
const FULL_EVERY: u32 = 16;
/// "None" for an index's parent offset and head op.
const NONE: u64 = u64::MAX;
/// Bytes per index entry: key, offset `u64`, len `u32`.
const ENTRY_LEN: usize = KEY_LEN + 12;

/// The largest decompressed tile body, in bytes (1 MiB). A displacement tile needs 26,886.
///
/// Readers reject a record that states a larger size, so a damaged or hostile file can't make
/// them allocate more.
pub const MAX_RAW: u32 = 1 << 20;

/// The largest base mesh, in canonical bytes ([`BaseMesh::to_bytes`]): 64 MiB, room for about
/// 2.8 million vertices and their faces.
///
/// Creating a file with a larger mesh fails, and readers reject a `Mesh` record that states a
/// larger size, so a damaged or hostile file can't make them allocate more.
pub const MAX_MESH_RAW: u32 = 64 << 20;

/// The largest `Meta` record payload a reader accepts. A meta is at most 66 bytes; the rest is
/// room for a later version's fields.
const MAX_META_PAYLOAD: u32 = 1024;

/// The largest `Mesh` record payload a reader accepts: the stated size, then a zstd frame of at
/// most [`MAX_MESH_RAW`] bytes, which zstd never expands by more than 1/128 plus a small
/// constant.
const MAX_MESH_PAYLOAD: u32 = 4 + MAX_MESH_RAW + MAX_MESH_RAW / 128 + 1024;

/// Why a project file operation failed.
#[derive(Debug, Error)]
pub enum StoreError {
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
    /// and meta given to [`ProjectFile::create`].
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
    /// The latest state ends before a torn tail from byte `offset`, which must be cut off, but
    /// the file can't be written. It was left untouched.
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
    /// [`MAX_RAW`].
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
}

/// A tile record's contents, decompressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBytes {
    /// The edit that wrote it.
    pub op: OpId,
    /// The channel header.
    pub header: Vec<u8>,
    /// The decompressed body.
    pub body: Vec<u8>,
}

/// Where a record lives: its first byte and its whole length. Records are never moved or
/// overwritten while the file is open, so a location stays valid after later appends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Loc {
    /// The record's first byte, from the start of the file.
    offset: u64,
    /// The record's length in bytes, framing included.
    len: u64,
}

/// Whether a [`ProjectFile`] accepts writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Opened at the latest state: writes append. Holds the file's exclusive lock.
    ReadWrite,
    /// Opened at an earlier save point, without write permission, or as a compaction source.
    /// Holds a shared lock (an exclusive one as the source of an in-place compaction).
    ReadOnly,
}

/// A lock on the project file: exclusive to write (or replace) it, shared to read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lock {
    /// One holder only: the writer, or the source of an in-place compaction.
    Exclusive,
    /// Any number of readers, and no writer.
    Shared,
}

impl Lock {
    /// Takes this lock on `file` without waiting.
    ///
    /// # Errors
    ///
    /// [`StoreError::Locked`] if a conflicting lock is held, [`StoreError::Io`].
    fn take(self, file: &File) -> Result<(), StoreError> {
        let locked = match self {
            Self::Exclusive => file.try_lock(),
            Self::Shared => file.try_lock_shared(),
        };
        locked.map_err(|e| match e {
            TryLockError::WouldBlock => StoreError::Locked,
            TryLockError::Error(e) => e.into(),
        })
    }
}

/// Opens `path` with `open` and takes `lock` on it (see [`lock_at`]).
///
/// # Errors
///
/// As [`lock_at`], and `open`'s error.
fn open_locked(
    path: &Path,
    open: impl Fn(&Path) -> io::Result<File>,
    lock: Lock,
) -> Result<File, StoreError> {
    lock_at(path, open(path)?, open, lock)
}

/// Takes `lock` on `file`, opened from `path`, and checks that `path` still names it.
///
/// A lock belongs to the file, not the path, and compacting in place renames a new file over the
/// path: a handle opened before that rename and locked after it would hold the old, unlinked file.
/// So if `path` names another file once the lock is held, `file` is dropped and `path` opened
/// (with `open`) and locked once more.
///
/// # Errors
///
/// [`StoreError::Locked`] if a conflicting lock is held, or `path` is replaced again during the
/// retry; `open`'s error; [`StoreError::Io`].
fn lock_at(
    path: &Path,
    file: File,
    open: impl Fn(&Path) -> io::Result<File>,
    lock: Lock,
) -> Result<File, StoreError> {
    lock.take(&file)?;
    if is_at(&file, path)? {
        return Ok(file);
    }
    drop(file);
    let reopened = open(path)?;
    lock.take(&reopened)?;
    if is_at(&reopened, path)? {
        Ok(reopened)
    } else {
        Err(StoreError::Locked)
    }
}

/// Whether `path` names `file` (the same device and inode); `false` if it names nothing.
///
/// # Errors
///
/// The I/O error of either `stat`.
fn is_at(file: &File, path: &Path) -> io::Result<bool> {
    let held = file.metadata()?;
    match std::fs::metadata(path) {
        Ok(now) => Ok(held.dev() == now.dev() && held.ino() == now.ino()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Opens `path` for reading.
///
/// # Errors
///
/// The I/O error of the open.
fn open_readable(path: &Path) -> io::Result<File> {
    File::open(path)
}

/// Opens `path` for reading and writing.
///
/// # Errors
///
/// The I/O error of the open.
fn open_writable(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

/// Why replay stopped before the end of the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// At an Index that verifies but didn't load as the base: damage.
    Index,
    /// At the end of the heads, or a record that doesn't verify.
    Other,
}

/// What opening at the latest state does with a torn tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repair {
    /// Cut it off, so the file is writable again. Opens read-write.
    Truncate,
    /// Fail with [`StoreError::NeedsRepair`]: the file can't be written. Opens read-only.
    Refuse,
    /// Leave it in place and open the state before it read-only, for compaction.
    Ignore,
}

impl Repair {
    /// The access an open with this repair has.
    const fn access(self) -> Access {
        match self {
            Self::Truncate => Access::ReadWrite,
            Self::Refuse | Self::Ignore => Access::ReadOnly,
        }
    }
}

/// Whether an Index lists every live tile or only those changed since its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    /// Every live tile: loading stops at this Index.
    Full,
    /// Only the tiles changed since the parent Index: loading reads the parent chain first.
    Delta,
}

/// A parsed Index payload.
#[derive(Debug)]
struct Index {
    /// Whether the entries are every live tile or a delta on the parent.
    coverage: Coverage,
    /// The offset of the Index saved before this one, if any.
    parent: Option<u64>,
    /// The file's head op when this Index was saved; `None` before any edit.
    head: Option<OpId>,
    /// Sorted by key; `None` marks a removed tile.
    entries: Vec<(TileKey, Option<Loc>)>,
}

impl Index {
    /// Whether every entry lies between the header and `at`, where this Index starts: an Index
    /// names only records written before it, and so bounds what loading them reads.
    fn entries_precede(&self, at: u64) -> bool {
        self.entries.iter().all(|(_, loc)| {
            loc.is_none_or(|loc| {
                loc.offset >= FIRST_RECORD
                    && loc.offset.checked_add(loc.len).is_some_and(|end| end <= at)
            })
        })
    }
}

/// The live map an Index and its parent chain describe.
#[derive(Debug)]
struct Chain {
    /// Each live tile's latest record.
    live: BTreeMap<TileKey, Loc>,
    /// Delta indexes after the chain's full one.
    deltas: u32,
    /// The head op of the chain's newest Index.
    head: Option<OpId>,
}

/// A parsed Tile payload, borrowing the record.
#[derive(Debug)]
struct TileRecord<'a> {
    /// The tile written.
    key: TileKey,
    /// The edit that wrote it.
    op: OpId,
    /// The channel header, stored uncompressed.
    header: &'a [u8],
    /// The body's length once decompressed.
    raw_len: usize,
    /// The zstd-compressed body.
    packed: &'a [u8],
}

/// An open project file.
pub struct ProjectFile {
    /// The file's identity, from its `Meta` record.
    meta: Meta,
    /// The base mesh, from its `Mesh` record.
    mesh: Arc<BaseMesh>,
    /// The open file, which holds the lock. [`FileReader`]s hold it weakly, so dropping the
    /// `ProjectFile` closes the file and releases the lock (once any read in flight ends).
    file: Arc<File>,
    /// Where the next record goes.
    end: u64,
    /// Each live tile's latest record, saved or not: what reads find.
    live: BTreeMap<TileKey, Loc>,
    /// Entries written since the last Index.
    pending: BTreeMap<TileKey, Loc>,
    /// The offset of the latest Index, the parent of the next one; `None` until the first save.
    last_index: Option<u64>,
    /// Delta indexes since the last full one.
    deltas: u32,
    /// The greatest op written (see [`ProjectFile::head`]).
    head: Option<OpId>,
    /// Whether writes are accepted.
    access: Access,
}

/// Leaves the mesh out (its size is enough) and counts the maps rather than listing them.
impl std::fmt::Debug for ProjectFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectFile")
            .field("meta", &self.meta)
            .field("mesh_faces", &self.mesh.face_count())
            .field("end", &self.end)
            .field("live", &self.live.len())
            .field("pending", &self.pending.len())
            .field("last_index", &self.last_index)
            .field("deltas", &self.deltas)
            .field("head", &self.head)
            .field("access", &self.access)
            .finish_non_exhaustive()
    }
}

impl ProjectFile {
    /// Creates a new project at `path`, open read-write, holding `meta` and `mesh` and no tiles.
    /// The file and its directory entry are durable when this returns.
    ///
    /// The file begins with its header, then a `Meta` record (`meta`) and a `Mesh` record
    /// (`mesh`, compressed), so it opens on its own. `meta` must have been made for `mesh`
    /// ([`Meta::new`]); this is checked before the file is created.
    ///
    /// The file is created, then locked: in between, another open of the same path can find it
    /// without its header (and fail with [`StoreError::BadHeader`]) or lock it first (and this
    /// fails with [`StoreError::Locked`], leaving the empty file behind). Only two processes
    /// creating and opening one new path at the same instant can meet this.
    ///
    /// # Errors
    ///
    /// [`StoreError::MeshMismatch`] if `meta`'s mesh hash isn't `mesh`'s;
    /// [`StoreError::TooLarge`] if the mesh's canonical bytes exceed [`MAX_MESH_RAW`] (nothing is
    /// created then); [`StoreError::Io`], including when `path` already exists;
    /// [`StoreError::Locked`].
    pub fn create(path: &Path, meta: Meta, mesh: Arc<BaseMesh>) -> Result<Self, StoreError> {
        let leading = leading_bytes(&meta, &mesh)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        Lock::Exclusive.take(&file)?;
        file.write_all_at(&leading, 0)?;
        file.sync_all()?;
        sync_parent(path)?;
        Ok(Self {
            meta,
            mesh,
            file: Arc::new(file),
            end: leading.len() as u64,
            live: BTreeMap::new(),
            pending: BTreeMap::new(),
            last_index: None,
            deltas: 0,
            head: None,
            access: Access::ReadWrite,
        })
    }

    /// Opens the project at `path` at its latest state, recovering from a crash: the state is
    /// the last good save point plus every complete record after it, and a torn tail is cut off.
    ///
    /// The file must begin with its `Meta` and `Mesh` records, whose mesh must match the meta's
    /// hash. They are checked before anything else: damage there fails the open and leaves the
    /// file untouched, as a damaged header does. Recovery starts after them.
    ///
    /// Where recovery stops before the end of the file, the rest is searched for a save point:
    /// an Index that verifies, parses and has a plausible parent. Saves sync their tiles before
    /// writing the Index, so everything before a good Index was durable: the stop was damage, and
    /// the file is left as it is. So is a stop at an Index that verifies but doesn't load.
    /// Otherwise the rest is a torn or unsynced tail of unsaved edits (even if some of its tile
    /// records verify, since writeback may have kept a later write and lost an earlier one), and
    /// is cut off.
    ///
    /// The file is opened read-write and locked exclusively. If it can't be opened for writing
    /// (no permission, or a read-only file system), it is opened read-only with a shared lock
    /// instead, which works only if there is no torn tail to cut off. If the path was replaced
    /// (by [`compact_in_place`]) between opening and locking, the new file is opened instead.
    ///
    /// # Errors
    ///
    /// [`StoreError::DamagedMidFile`], [`StoreError::NeedsRepair`], [`StoreError::Locked`],
    /// [`StoreError::BadHeader`], [`StoreError::MissingMeta`], [`StoreError::MeshMismatch`],
    /// [`StoreError::BadMesh`], [`StoreError::Io`].
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        match open_locked(path, open_writable, Lock::Exclusive) {
            Ok(file) => Self::open_latest(file, Repair::Truncate),
            Err(StoreError::Io(e))
                if matches!(
                    e.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
                ) =>
            {
                Self::open_latest(
                    open_locked(path, open_readable, Lock::Shared)?,
                    Repair::Refuse,
                )
            }
            Err(e) => Err(e),
        }
    }

    /// Opens `file`, already locked, at its latest state, as [`ProjectFile::open`] does; `repair`
    /// says what happens to a torn tail, and with it the access. `file` must be writable and
    /// locked exclusively for [`Repair::Truncate`].
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::open`].
    fn open_latest(file: File, repair: Repair) -> Result<Self, StoreError> {
        let access = repair.access();
        let len = file.metadata()?.len();
        check_header(&file, len)?;
        let (meta, mesh, start) = read_leading(&file, len)?;
        let heads = scan_heads(&file, start, len)?;
        let mut this = Self {
            meta,
            mesh,
            file: Arc::new(file),
            end: start,
            live: BTreeMap::new(),
            pending: BTreeMap::new(),
            last_index: None,
            deltas: 0,
            head: None,
            access,
        };
        let replay_from = this.load_base(&heads, len)?;
        let stop = this.replay(heads.get(replay_from..).unwrap_or_default())?;
        if this.end < len {
            if stop == Stop::Index || find_index(&this.file, this.end, len)?.is_some() {
                return Err(StoreError::DamagedMidFile { offset: this.end });
            }
            match repair {
                Repair::Truncate => {
                    this.file.set_len(this.end)?;
                    this.file.sync_all()?;
                }
                Repair::Refuse => return Err(StoreError::NeedsRepair { offset: this.end }),
                Repair::Ignore => {}
            }
        }
        Ok(this)
    }

    /// Opens the project at `path` read-only, with a shared lock, at the save point (an Index)
    /// at byte `index_offset`.
    ///
    /// # Errors
    ///
    /// [`StoreError::NotASavePoint`] if no good Index starts there, [`StoreError::Damaged`] if
    /// its parent chain is, [`StoreError::Locked`] if the file is open for writing,
    /// [`StoreError::BadHeader`], [`StoreError::MissingMeta`], [`StoreError::MeshMismatch`],
    /// [`StoreError::BadMesh`], [`StoreError::Io`].
    pub fn open_at(path: &Path, index_offset: u64) -> Result<Self, StoreError> {
        let file = open_locked(path, open_readable, Lock::Shared)?;
        let len = file.metadata()?.len();
        check_header(&file, len)?;
        let (meta, mesh, start) = read_leading(&file, len)?;
        if index_offset < start || read_index(&file, index_offset, len)?.is_none() {
            return Err(StoreError::NotASavePoint(index_offset));
        }
        let chain = load_chain(&file, index_offset, len)?.ok_or(StoreError::Damaged {
            offset: index_offset,
        })?;
        Ok(Self {
            meta,
            mesh,
            file: Arc::new(file),
            end: len,
            live: chain.live,
            pending: BTreeMap::new(),
            last_index: Some(index_offset),
            deltas: chain.deltas,
            head: chain.head,
            access: Access::ReadOnly,
        })
    }

    /// Loads the last Index of `heads` whose record and parent chain are good, and returns the
    /// position in `heads` of the first record after it (0 if there is none).
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`].
    fn load_base(&mut self, heads: &[(u64, Head)], len: u64) -> Result<usize, StoreError> {
        for (at, &(offset, head)) in heads.iter().enumerate().rev() {
            if head.kind != KIND_INDEX {
                continue;
            }
            if let Some(chain) = load_chain(&self.file, offset, len)? {
                self.live = chain.live;
                self.deltas = chain.deltas;
                self.head = chain.head;
                self.last_index = Some(offset);
                self.end = offset + head.total();
                return Ok(at + 1);
            }
        }
        Ok(0)
    }

    /// Replays the records after the base Index while they verify, moving `end` past each one,
    /// and says why it stopped. A record that verifies but doesn't parse (a tile key from a
    /// newer version, say) is skipped, but an Index stops replay: any Index this late failed to
    /// load as the base, whether or not it parses, and is damage.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`].
    fn replay(&mut self, heads: &[(u64, Head)]) -> Result<Stop, StoreError> {
        for &(offset, head) in heads {
            let raw = read_exact(&self.file, offset, head.total())?;
            let Some((kind, payload)) = verify(&raw) else {
                break;
            };
            let loc = Loc {
                offset,
                len: head.total(),
            };
            match kind {
                KIND_TILE => {
                    if let Some(tile) = parse_tile_payload(payload) {
                        self.live.insert(tile.key, loc);
                        self.pending.insert(tile.key, loc);
                        self.head = self.head.max(Some(tile.op));
                    }
                }
                KIND_INDEX => return Ok(Stop::Index),
                // A kind from a newer version: skipped.
                _ => {}
            }
            self.end = offset + head.total();
        }
        Ok(Stop::Other)
    }

    /// Appends a tile record: `key`'s new content, written by `op`. The body is compressed.
    /// Durable once [`ProjectFile::save`] returns.
    ///
    /// # Errors
    ///
    /// [`StoreError::ReadOnly`], [`StoreError::TooLarge`] (including a body larger than
    /// [`MAX_RAW`]), [`StoreError::Io`].
    pub fn write_tile(
        &mut self,
        key: TileKey,
        op: OpId,
        header: &[u8],
        body: &[u8],
    ) -> Result<(), StoreError> {
        if self.access == Access::ReadOnly {
            return Err(StoreError::ReadOnly);
        }
        // Checked before compressing: no reader would accept it.
        if !u32::try_from(body.len()).is_ok_and(|n| n <= MAX_RAW) {
            return Err(StoreError::TooLarge);
        }
        let packed = compress(body)?;
        let payload = tile_payload(key, op, header, body.len(), &packed)?;
        let loc = self.append(KIND_TILE, &payload)?;
        self.live.insert(key, loc);
        self.pending.insert(key, loc);
        self.head = self.head.max(Some(op));
        Ok(())
    }

    /// Appends a displacement tile (see [`ProjectFile::write_tile`]).
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::write_tile`]; [`StoreError::NotDisplacement`] if `key` is of another
    /// channel, [`StoreError::WrongSide`] if the tile's side isn't the key's.
    pub fn write_displacement(
        &mut self,
        key: TileKey,
        op: OpId,
        tile: &CorrectionTile,
    ) -> Result<(), StoreError> {
        check_displacement(&key, tile.side())?;
        let (header, body) = encode_displacement(tile)?;
        self.write_tile(key, op, &header, &body)
    }

    /// Frames `payload` and writes it at the end.
    ///
    /// # Errors
    ///
    /// [`StoreError::TooLarge`], [`StoreError::Io`].
    fn append(&mut self, kind: u8, payload: &[u8]) -> Result<Loc, StoreError> {
        let record = frame(kind, payload).ok_or(StoreError::TooLarge)?;
        self.append_record(&record)
    }

    /// Writes an already framed record at the end.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`].
    fn append_record(&mut self, record: &[u8]) -> Result<Loc, StoreError> {
        self.file.write_all_at(record, self.end)?;
        let loc = Loc {
            offset: self.end,
            len: record.len() as u64,
        };
        self.end += loc.len;
        Ok(loc)
    }

    /// `key`'s latest tile record, decompressed; `None` if the tile is absent.
    ///
    /// # Errors
    ///
    /// [`StoreError::Damaged`] if the record fails its checksum or doesn't decode as a tile of
    /// `key`, [`StoreError::Io`].
    pub fn read_tile(&self, key: &TileKey) -> Result<Option<TileBytes>, StoreError> {
        self.read_tile_checked(key, |_| Ok(()))
    }

    /// As [`ProjectFile::read_tile`], with `check` run on the parsed record before its body is
    /// decompressed.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_tile`], and `check`'s error.
    fn read_tile_checked(
        &self,
        key: &TileKey,
        check: impl FnOnce(&TileRecord<'_>) -> Result<(), StoreError>,
    ) -> Result<Option<TileBytes>, StoreError> {
        let Some(&Loc { offset, len }) = self.live.get(key) else {
            return Ok(None);
        };
        let raw = self.read_record(offset, len)?;
        let tile = tile_record(&raw, key).ok_or(StoreError::Damaged { offset })?;
        check(&tile)?;
        let body = decompress(tile.packed, tile.raw_len)
            .ok()
            .ok_or(StoreError::Damaged { offset })?;
        Ok(Some(TileBytes {
            op: tile.op,
            header: tile.header.to_vec(),
            body,
        }))
    }

    /// The `len` bytes of the record at `offset`, which the live map promises exist.
    ///
    /// # Errors
    ///
    /// [`StoreError::Damaged`] if the file ends first, [`StoreError::Io`].
    fn read_record(&self, offset: u64, len: u64) -> Result<Vec<u8>, StoreError> {
        match read_exact(&self.file, offset, len) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                Err(StoreError::Damaged { offset })
            }
            other => Ok(other?),
        }
    }

    /// `key`'s latest displacement tile and the edit that wrote it; `None` if absent.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_tile`]; [`StoreError::Codec`] if the payload isn't a valid
    /// displacement tile, [`StoreError::NotDisplacement`] if `key` is of another channel,
    /// [`StoreError::WrongSide`] if the stored side isn't the key's. The side and the stated body
    /// size are checked before decompressing.
    pub fn read_displacement(
        &self,
        key: &TileKey,
    ) -> Result<Option<(OpId, CorrectionTile)>, StoreError> {
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        let Some(op) = self.read_displacement_into(key, &mut slot, &mut ReadScratch::new()?)?
        else {
            return Ok(None);
        };
        Ok(Some((op, slot.to_tile().map_err(CodecError::from)?)))
    }

    /// Reads `key`'s latest displacement tile straight into `slot`, as a [`FileReader`] does;
    /// returns the edit that wrote it, or `None` (and leaves `slot` alone) if the tile is absent.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_displacement`]. On error `slot` holds no tile.
    pub(crate) fn read_displacement_into(
        &self,
        key: &TileKey,
        slot: &mut CorrectionSlot,
        scratch: &mut ReadScratch,
    ) -> Result<Option<OpId>, StoreError> {
        if let Err(e) = check_displacement(key, key.side()) {
            bytemuck::bytes_of_mut(slot).fill(0);
            return Err(e);
        }
        let Some(loc) = self.locate(key) else {
            return Ok(None);
        };
        read_displacement_at(&self.file, key, loc, slot, scratch).map(Some)
    }

    /// Where `key`'s latest record is, if it has one.
    pub(crate) fn locate(&self, key: &TileKey) -> Option<Loc> {
        self.live.get(key).copied()
    }

    /// A reader for other threads. It reads this open file while it stays open, and doesn't keep
    /// it (or its lock) open after this `ProjectFile` is dropped.
    pub(crate) fn reader(&self) -> FileReader {
        FileReader {
            file: Arc::downgrade(&self.file),
        }
    }

    /// Makes every write so far durable and records a save point. Returns its offset, for
    /// [`ProjectFile::open_at`].
    ///
    /// The save point is an Index naming the live tiles: all of them the first time and every 16th
    /// time, otherwise only those written since the last save.
    ///
    /// # Errors
    ///
    /// [`StoreError::ReadOnly`], [`StoreError::TooLarge`], [`StoreError::Io`].
    pub fn save(&mut self) -> Result<u64, StoreError> {
        if self.access == Access::ReadOnly {
            return Err(StoreError::ReadOnly);
        }
        // Tiles are durable before the Index that names them.
        self.file.sync_data()?;
        let coverage = if self.last_index.is_none() || self.deltas + 1 >= FULL_EVERY {
            Coverage::Full
        } else {
            Coverage::Delta
        };
        let entries = match coverage {
            Coverage::Full => &self.live,
            Coverage::Delta => &self.pending,
        };
        let payload = index_payload(coverage, self.last_index, self.head, entries)?;
        let loc = self.append(KIND_INDEX, &payload)?;
        self.file.sync_data()?;
        self.last_index = Some(loc.offset);
        self.deltas = match coverage {
            Coverage::Full => 0,
            Coverage::Delta => self.deltas + 1,
        };
        self.pending.clear();
        Ok(loc.offset)
    }

    /// The save points of the current state's chain, oldest first.
    ///
    /// # Errors
    ///
    /// [`StoreError::Damaged`] if an Index of the chain is, [`StoreError::Io`].
    pub fn save_points(&self) -> Result<Vec<u64>, StoreError> {
        let len = self.file.metadata()?.len();
        let mut points = Vec::new();
        let mut next = self.last_index;
        while let Some(offset) = next {
            let index =
                read_index(&self.file, offset, len)?.ok_or(StoreError::Damaged { offset })?;
            points.push(offset);
            next = match index.parent {
                Some(parent) if parent < offset => Some(parent),
                Some(_) => return Err(StoreError::Damaged { offset }),
                None => None,
            };
        }
        points.reverse();
        Ok(points)
    }

    /// The file's identity, from its `Meta` record.
    #[must_use]
    pub const fn meta(&self) -> &Meta {
        &self.meta
    }

    /// The base mesh, from its `Mesh` record.
    #[must_use]
    pub fn mesh(&self) -> Arc<BaseMesh> {
        Arc::clone(&self.mesh)
    }

    /// The newest edit in the current state: the greatest [`OpId`] written, which is the newest
    /// only while every edit comes from one site (see [`OpId`]'s ordering).
    #[must_use]
    pub const fn head(&self) -> Option<OpId> {
        self.head
    }

    /// The live tiles, in key order.
    pub fn keys(&self) -> impl Iterator<Item = &TileKey> {
        self.live.keys()
    }

    /// Whether `key` has a live tile.
    #[must_use]
    pub fn contains(&self, key: &TileKey) -> bool {
        self.live.contains_key(key)
    }

    /// The file's length in bytes (for a read-only file, as opened). A new file's length is
    /// that of its header and leading records.
    #[must_use]
    #[expect(
        clippy::len_without_is_empty,
        reason = "a project file always holds its header"
    )]
    pub const fn len(&self) -> u64 {
        self.end
    }
}

/// Test-only access to the format's internals.
#[cfg(test)]
impl ProjectFile {
    /// Where the first record after the leading `Meta` and `Mesh` records starts, in a file
    /// made with the test fixture's meta and mesh.
    ///
    /// # Panics
    ///
    /// Never: the fixture's meta is made for its mesh.
    fn fixture_start() -> u64 {
        let leading = leading_bytes(&crate::fixture::meta(), &crate::fixture::mesh()).unwrap();
        u64::try_from(leading.len()).unwrap()
    }

    /// Payload bytes of the Index at `offset`.
    ///
    /// # Errors
    ///
    /// [`StoreError::Damaged`] if no record head is there, [`StoreError::Io`].
    fn index_len(&self, offset: u64) -> Result<u64, StoreError> {
        let len = self.file.metadata()?.len();
        Ok(read_head(&self.file, offset, len)?
            .ok_or(StoreError::Damaged { offset })?
            .len
            .into())
    }

    /// Payload bytes of the last Index.
    fn last_index_len(&self) -> Option<u64> {
        self.index_len(self.last_index?).ok()
    }

    /// Frames and appends an arbitrary record.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::append`].
    fn append_raw(&mut self, kind: u8, payload: &[u8]) -> Result<(), StoreError> {
        self.append(kind, payload).map(drop)
    }
}

/// Reads tiles of an open [`ProjectFile`] from any thread.
///
/// It reads at locations the owner looked up ([`ProjectFile::locate`]). The file is append-only
/// while open, so a location stays valid. A reader doesn't keep the file open: once its
/// [`ProjectFile`] is dropped, the file closes and its lock is released (after any read in
/// flight, which holds the file for its duration), and reads fail with [`StoreError::Closed`].
#[derive(Debug, Clone)]
pub(crate) struct FileReader {
    /// The owner's open file, upgraded for the length of each read.
    file: Weak<File>,
}

impl FileReader {
    /// Reads the displacement tile of `key` recorded at `loc` straight into `slot`, and returns
    /// the edit that wrote it. Allocates nothing once `scratch` has grown to the largest record.
    ///
    /// # Errors
    ///
    /// [`StoreError::Damaged`] if the record fails its checksum, isn't a tile of `key` or doesn't
    /// decompress; [`StoreError::NotDisplacement`], [`StoreError::WrongSide`],
    /// [`StoreError::Codec`] as for [`ProjectFile::read_displacement`]; [`StoreError::Closed`] if
    /// the [`ProjectFile`] has been dropped; [`StoreError::Io`]. On error `slot` holds no tile.
    pub(crate) fn read_displacement_into(
        &self,
        key: &TileKey,
        loc: Loc,
        slot: &mut CorrectionSlot,
        scratch: &mut ReadScratch,
    ) -> Result<OpId, StoreError> {
        let file = check_displacement(key, key.side())
            .and_then(|()| self.file.upgrade().ok_or(StoreError::Closed));
        match file {
            Ok(file) => read_displacement_at(&file, key, loc, slot, scratch),
            Err(e) => {
                bytemuck::bytes_of_mut(slot).fill(0);
                Err(e)
            }
        }
    }
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
    /// [`StoreError::Io`] if zstd can't allocate its context.
    pub(crate) fn new() -> Result<Self, StoreError> {
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
fn read_displacement_at(
    file: &File,
    key: &TileKey,
    loc: Loc,
    slot: &mut CorrectionSlot,
    scratch: &mut ReadScratch,
) -> Result<OpId, StoreError> {
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
) -> Result<OpId, StoreError> {
    let ReadScratch { record, raw, zstd } = scratch;
    let damaged = StoreError::Damaged { offset };
    record.resize(usize::try_from(len).or(Err(StoreError::TooLarge))?, 0);
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
        _ => return Err(StoreError::Damaged { offset }),
    }
    decode_displacement_into(tile.header, raw, slot)?;
    Ok(tile.op)
}

/// Writes a new project at `dst` holding `src`'s meta and mesh, its live tiles (raw records
/// copied, not recompressed) and one full Index.
///
/// `src` is recovered as by [`ProjectFile::open`], but never modified: a torn tail is left in
/// place, not cut off. It is held with a shared lock, so it can't be open for writing. History
/// before the latest state is dropped; which history to keep is decided with stroke history
/// (M2). On error, `dst` may be left partly written.
///
/// # Errors
///
/// Any [`StoreError`] from reading `src` (a damaged live tile is [`StoreError::Damaged`]; one
/// open for writing, [`StoreError::Locked`]) or writing `dst`, which must not exist.
pub fn compact(src: &Path, dst: &Path) -> Result<(), StoreError> {
    let from = ProjectFile::open_latest(
        open_locked(src, open_readable, Lock::Shared)?,
        Repair::Ignore,
    )?;
    compact_into(&from, dst)
}

/// Compacts the project at `path` in place, as [`compact`] does.
///
/// It compacts into a temporary file `<name>.compact-*.tmp` in the same directory, gives it the
/// project's permissions, then renames it over `path` (resolved first, so a symlink keeps
/// pointing at the project rather than being replaced). Replacing the project is a write, so
/// the source is locked exclusively until after the rename: no writer or other compaction runs
/// meanwhile, and one that opened the old file locks it only to find the path replaced (see
/// [`ProjectFile::open`]). The directory is synced after the rename, so the swap is durable.
///
/// On error, `path` is unchanged (unless the rename succeeded and only the directory sync
/// failed) and the temporary file is removed. A crash before the rename can leave the temporary
/// file behind; it is never read, and can be deleted.
///
/// # Errors
///
/// As [`compact`], with [`StoreError::Locked`] if the project is open at all.
pub fn compact_in_place(path: &Path) -> Result<(), StoreError> {
    let path = path.canonicalize()?;
    let from = ProjectFile::open_latest(
        open_locked(&path, open_readable, Lock::Exclusive)?,
        Repair::Ignore,
    )?;
    let permissions = from.file.metadata()?.permissions();
    let tmp = temp_sibling(&path)?;
    let swapped = compact_into(&from, &tmp).and_then(|()| {
        std::fs::set_permissions(&tmp, permissions)?;
        std::fs::rename(&tmp, &path)?;
        sync_parent(&path)
    });
    if swapped.is_err() {
        // Best effort: gone already if the rename succeeded.
        drop(std::fs::remove_file(&tmp));
    }
    swapped
}

/// Writes `from`'s live tiles and one full Index to a new project at `dst`.
///
/// # Errors
///
/// As [`compact`].
fn compact_into(from: &ProjectFile, dst: &Path) -> Result<(), StoreError> {
    let mut to = ProjectFile::create(dst, from.meta, Arc::clone(&from.mesh))?;
    for (key, &Loc { offset, len }) in &from.live {
        let raw = from.read_record(offset, len)?;
        if tile_record(&raw, key).is_none() {
            return Err(StoreError::Damaged { offset });
        }
        let loc = to.append_record(&raw)?;
        to.live.insert(*key, loc);
    }
    to.head = from.head;
    to.save()?;
    Ok(())
}

/// A path for a temporary file next to `path`, unique within this process and among processes.
///
/// # Errors
///
/// [`StoreError::Io`] if `path` has no file name.
fn temp_sibling(path: &Path) -> Result<PathBuf, StoreError> {
    /// Distinguishes this process's temporary files.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the project path has no file name",
        )
    })?;
    let mut tmp = name.to_os_string();
    tmp.push(format!(
        ".compact-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(path.with_file_name(tmp))
}

/// Syncs the directory holding `path`, making a new or renamed entry in it durable.
///
/// # Errors
///
/// [`StoreError::Io`].
fn sync_parent(path: &Path) -> Result<(), StoreError> {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    File::open(dir)?.sync_all()?;
    Ok(())
}

/// Checks that `key` is a displacement key whose tiles have `side` vertices per side.
///
/// # Errors
///
/// [`StoreError::NotDisplacement`], [`StoreError::WrongSide`].
fn check_displacement(key: &TileKey, side: u32) -> Result<(), StoreError> {
    if key.channel() != Channel::DISPLACEMENT {
        Err(StoreError::NotDisplacement(key.channel().raw()))
    } else if side != key.side() {
        Err(StoreError::WrongSide {
            key: key.side(),
            tile: side,
        })
    } else {
        Ok(())
    }
}

/// The tile in a raw record, if the record verifies and holds a tile of `key`.
fn tile_record<'a>(raw: &'a [u8], key: &TileKey) -> Option<TileRecord<'a>> {
    match verify(raw) {
        Some((KIND_TILE, payload)) => parse_tile_payload(payload),
        _ => None,
    }
    .filter(|tile| tile.key == *key)
}

/// Checks the file header.
///
/// # Errors
///
/// [`StoreError::BadHeader`] if it is short or has the wrong magic or version, or flags or
/// reserved bits this version doesn't know (a later version may set them to mean something),
/// [`StoreError::Io`].
fn check_header(file: &File, len: u64) -> Result<(), StoreError> {
    if len < FIRST_RECORD {
        return Err(StoreError::BadHeader);
    }
    let mut header = [0; HEADER_LEN];
    file.read_exact_at(&mut header, 0)?;
    let mut r = Reader::new(&header);
    if r.take(MAGIC.len()) == Some(&MAGIC[..])
        && r.u16() == Some(VERSION)
        && r.u16() == Some(0)
        && r.u64() == Some(0)
    {
        Ok(())
    } else {
        Err(StoreError::BadHeader)
    }
}

/// The header and leading records of a new file holding `meta` and `mesh`.
///
/// # Errors
///
/// [`StoreError::MeshMismatch`] if `meta` wasn't made for `mesh`; [`StoreError::TooLarge`] if
/// the mesh's bytes exceed [`MAX_MESH_RAW`]; [`StoreError::Io`] if compression fails.
fn leading_bytes(meta: &Meta, mesh: &BaseMesh) -> Result<Vec<u8>, StoreError> {
    let raw = mesh.to_bytes();
    if mesh_hash(&raw) != meta.mesh_hash() {
        return Err(StoreError::MeshMismatch);
    }
    let raw_len = u32::try_from(raw.len())
        .ok()
        .filter(|&n| n <= MAX_MESH_RAW)
        .ok_or(StoreError::TooLarge)?;
    let mut mesh_payload = raw_len.to_le_bytes().to_vec();
    mesh_payload.extend_from_slice(&compress(&raw)?);
    let meta_record = frame(KIND_META, &meta.to_payload()).ok_or(StoreError::TooLarge)?;
    let mesh_record = frame(KIND_MESH, &mesh_payload).ok_or(StoreError::TooLarge)?;
    let mut header = [0; HEADER_LEN];
    let (magic, rest) = header.split_at_mut(MAGIC.len());
    magic.copy_from_slice(&MAGIC);
    if let Some(version) = rest.get_mut(..2) {
        version.copy_from_slice(&VERSION.to_le_bytes());
    }
    Ok([&header[..], &meta_record, &mesh_record].concat())
}

/// Reads and checks the leading `Meta` and `Mesh` records: the file's identity and base mesh,
/// and the offset where its other records start.
///
/// # Errors
///
/// [`StoreError::MissingMeta`] if either record isn't where it belongs;
/// [`StoreError::BadHeader`] if either is damaged or doesn't parse (or states a mesh larger
/// than [`MAX_MESH_RAW`]); [`StoreError::MeshMismatch`] if the mesh doesn't match the meta's
/// hash; [`StoreError::BadMesh`] if it isn't a valid mesh; [`StoreError::Io`].
fn read_leading(file: &File, len: u64) -> Result<(Meta, Arc<BaseMesh>, u64), StoreError> {
    let (meta_payload, mesh_at) =
        leading_record(file, FIRST_RECORD, len, KIND_META, MAX_META_PAYLOAD)?;
    let meta = Meta::from_payload(&meta_payload).ok_or(StoreError::BadHeader)?;
    let (mesh_payload, start) = leading_record(file, mesh_at, len, KIND_MESH, MAX_MESH_PAYLOAD)?;
    let mut r = Reader::new(&mesh_payload);
    let raw_len = r
        .u32()
        .filter(|&n| n <= MAX_MESH_RAW)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(StoreError::BadHeader)?;
    let raw = decompress(r.rest(), raw_len).or(Err(StoreError::BadHeader))?;
    if mesh_hash(&raw) != meta.mesh_hash() {
        return Err(StoreError::MeshMismatch);
    }
    let mesh = BaseMesh::from_bytes(&raw).map_err(StoreError::BadMesh)?;
    Ok((meta, Arc::new(mesh), start))
}

/// The payload of the leading record at `offset`, which must be of `kind` with a payload of at
/// most `max` bytes, and the offset of the record after it.
///
/// # Errors
///
/// [`StoreError::MissingMeta`] if the file ends before the record's head or the head is of
/// another kind; [`StoreError::BadHeader`] if the head is malformed, the record states more
/// than `max` bytes or runs past the end, or fails its checksum; [`StoreError::Io`].
fn leading_record(
    file: &File,
    offset: u64,
    len: u64,
    kind: u8,
    max: u32,
) -> Result<(Vec<u8>, u64), StoreError> {
    if offset.saturating_add(HEAD as u64) > len {
        return Err(StoreError::MissingMeta);
    }
    let mut bytes = [0; HEAD];
    file.read_exact_at(&mut bytes, offset)?;
    let head = parse_head(bytes).ok_or(StoreError::BadHeader)?;
    if head.kind != kind {
        return Err(StoreError::MissingMeta);
    }
    let end = offset.saturating_add(head.total());
    if head.len > max || end > len {
        return Err(StoreError::BadHeader);
    }
    let raw = read_exact(file, offset, head.total())?;
    let (_, payload) = verify(&raw).ok_or(StoreError::BadHeader)?;
    Ok((payload.to_vec(), end))
}

/// Every record head from `start` on, trusting lengths, up to the first that is invalid or runs
/// past `len`.
///
/// # Errors
///
/// The I/O error of a read.
fn scan_heads(file: &File, start: u64, len: u64) -> io::Result<Vec<(u64, Head)>> {
    let mut heads = Vec::new();
    let mut offset = start;
    while let Some(head) = read_head(file, offset, len)? {
        heads.push((offset, head));
        offset += head.total();
    }
    Ok(heads)
}

/// The first offset in `[from, len)` where a save point starts (see [`save_point`]): proof that the
/// file is damaged, not torn, at `from`. `None` if there is none.
///
/// Every byte offset is a candidate, so the checks beyond the checksum keep bytes inside another
/// record from passing for one.
///
/// The range is read in windows, so a long tail costs one pass and little memory; a candidate
/// longer than its window is read on its own.
///
/// # Errors
///
/// The I/O error of a read.
fn find_index(file: &File, from: u64, len: u64) -> io::Result<Option<u64>> {
    /// Candidate head positions per window.
    const WINDOW: u64 = 1 << 20;
    let mut start = from;
    while start.saturating_add((HEAD + TAIL) as u64) <= len {
        // Enough bytes for a head at each of the window's positions.
        let buf = read_exact(file, start, (len - start).min(WINDOW + HEAD as u64 - 1))?;
        for (at, bytes) in (start..).zip(buf.windows(HEAD)) {
            let Some(head) = bytes
                .try_into()
                .ok()
                .and_then(parse_head)
                .filter(|h| h.kind == KIND_INDEX)
                .filter(|h| at.saturating_add(h.total()) <= len)
            else {
                continue;
            };
            let in_buf = usize::try_from(at - start)
                .ok()
                .zip(usize::try_from(head.total()).ok())
                .and_then(|(lo, n)| buf.get(lo..lo.checked_add(n)?));
            let good = match in_buf {
                Some(record) => save_point(record, at).is_some(),
                None => save_point(&read_exact(file, at, head.total())?, at).is_some(),
            };
            if good {
                return Ok(Some(at));
            }
        }
        start = start.saturating_add(WINDOW);
    }
    Ok(None)
}

/// The head at `offset`, if it is valid and its record ends by `len`.
///
/// # Errors
///
/// The I/O error of the read.
fn read_head(file: &File, offset: u64, len: u64) -> io::Result<Option<Head>> {
    if offset.saturating_add(HEAD as u64) > len {
        return Ok(None);
    }
    let mut bytes = [0; HEAD];
    file.read_exact_at(&mut bytes, offset)?;
    Ok(parse_head(bytes).filter(|head| offset.saturating_add(head.total()) <= len))
}

/// `len` bytes at `offset`.
///
/// # Errors
///
/// The I/O error of the read (`UnexpectedEof` past the end), or `OutOfMemory` if `len` doesn't
/// fit in memory.
fn read_exact(file: &File, offset: u64, len: u64) -> io::Result<Vec<u8>> {
    let len = usize::try_from(len).map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, e))?;
    let mut buf = vec![0; len];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

/// The save point at `offset` (see [`save_point`]), if a whole one starts there.
///
/// # Errors
///
/// The I/O error of a read.
fn read_index(file: &File, offset: u64, len: u64) -> io::Result<Option<Index>> {
    let Some(head) = read_head(file, offset, len)? else {
        return Ok(None);
    };
    if head.kind != KIND_INDEX {
        return Ok(None);
    }
    Ok(save_point(&read_exact(file, offset, head.total())?, offset))
}

/// The Index in `record`, which starts at byte `at`, if it is a save point: it verifies, parses
/// as an Index, and has no parent or one between the first record and `at`.
fn save_point(record: &[u8], at: u64) -> Option<Index> {
    let index = match verify(record)? {
        (KIND_INDEX, payload) => parse_index_payload(payload)?,
        _ => return None,
    };
    let parent_ok = index
        .parent
        .is_none_or(|parent| (FIRST_RECORD..at).contains(&parent));
    parent_ok.then_some(index)
}

/// The live map of the Index at `offset`, or `None` if any Index of its chain is bad.
///
/// Follows parent links back to a full Index, then applies it and the deltas oldest first. A chain
/// that ends without a full Index is bad, and so is one with an entry outside the records before
/// its Index.
///
/// # Errors
///
/// The I/O error of a read.
fn load_chain(file: &File, offset: u64, len: u64) -> io::Result<Option<Chain>> {
    let mut chain = Vec::new();
    let mut at = offset;
    loop {
        let Some(index) = read_index(file, at, len)? else {
            return Ok(None);
        };
        if !index.entries_precede(at) {
            return Ok(None);
        }
        let (coverage, parent) = (index.coverage, index.parent);
        chain.push(index);
        match (coverage, parent) {
            (Coverage::Full, _) => break,
            // Parents come strictly earlier, so the walk ends.
            (Coverage::Delta, Some(parent)) if parent < at => at = parent,
            (Coverage::Delta, _) => return Ok(None),
        }
    }
    let head = chain.first().and_then(|newest| newest.head);
    let Ok(deltas) = u32::try_from(chain.len() - 1) else {
        return Ok(None);
    };
    let mut live = BTreeMap::new();
    for index in chain.into_iter().rev() {
        for (key, loc) in index.entries {
            match loc {
                Some(loc) => live.insert(key, loc),
                None => live.remove(&key),
            };
        }
    }
    Ok(Some(Chain { live, deltas, head }))
}

/// A Tile payload: `key | op: u64 | header_len: u16 | header | raw_len: u32 | zstd body`.
///
/// # Errors
///
/// [`StoreError::TooLarge`] if the header exceeds its length field or the body [`MAX_RAW`].
fn tile_payload(
    key: TileKey,
    op: OpId,
    header: &[u8],
    raw_len: usize,
    packed: &[u8],
) -> Result<Vec<u8>, StoreError> {
    let header_len = u16::try_from(header.len()).or(Err(StoreError::TooLarge))?;
    let raw_len = u32::try_from(raw_len)
        .ok()
        .filter(|&n| n <= MAX_RAW)
        .ok_or(StoreError::TooLarge)?;
    let mut out = Vec::with_capacity(KEY_LEN + 14 + header.len() + packed.len());
    put_key(&mut out, &key);
    out.extend_from_slice(&op.to_bits().to_le_bytes());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(&raw_len.to_le_bytes());
    out.extend_from_slice(packed);
    Ok(out)
}

/// Parses [`tile_payload`]'s output; `None` on any shortfall, an invalid key or a decompressed
/// size over [`MAX_RAW`].
fn parse_tile_payload(payload: &[u8]) -> Option<TileRecord<'_>> {
    let mut r = Reader::new(payload);
    let key = r.key()?;
    let op = OpId::from_bits(r.u64()?)?;
    let header_len = r.u16()?;
    let header = r.take(usize::from(header_len))?;
    let raw_len = usize::try_from(r.u32().filter(|&n| n <= MAX_RAW)?).ok()?;
    Some(TileRecord {
        key,
        op,
        header,
        raw_len,
        packed: r.rest(),
    })
}

/// An Index payload: `full: u8 | reserved [u8; 7] | parent: u64 | head: u64 | count: u32 |
/// count × (key | offset: u64 | len: u32)`, entries in key order.
///
/// # Errors
///
/// [`StoreError::TooLarge`] if there are more than `u32::MAX` entries or a record is longer than
/// `u32::MAX` bytes.
fn index_payload(
    coverage: Coverage,
    parent: Option<u64>,
    head: Option<OpId>,
    entries: &BTreeMap<TileKey, Loc>,
) -> Result<Vec<u8>, StoreError> {
    let count = u32::try_from(entries.len()).or(Err(StoreError::TooLarge))?;
    let mut out = Vec::with_capacity(28 + entries.len() * ENTRY_LEN);
    out.push(match coverage {
        Coverage::Full => 1,
        Coverage::Delta => 0,
    });
    out.extend_from_slice(&[0; 7]);
    out.extend_from_slice(&parent.unwrap_or(NONE).to_le_bytes());
    out.extend_from_slice(&head.map_or(NONE, OpId::to_bits).to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    for (key, loc) in entries {
        put_key(&mut out, key);
        out.extend_from_slice(&loc.offset.to_le_bytes());
        let len = u32::try_from(loc.len).or(Err(StoreError::TooLarge))?;
        out.extend_from_slice(&len.to_le_bytes());
    }
    Ok(out)
}

/// Parses [`index_payload`]'s output; `None` on any shortfall, an invalid key or trailing bytes.
fn parse_index_payload(payload: &[u8]) -> Option<Index> {
    let mut r = Reader::new(payload);
    let coverage = match r.u8()? {
        0 => Coverage::Delta,
        1 => Coverage::Full,
        _ => return None,
    };
    r.take(7)?;
    let parent = Some(r.u64()?).filter(|&p| p != NONE);
    let head = match r.u64()? {
        NONE => None,
        bits => Some(OpId::from_bits(bits)?),
    };
    let count = usize::try_from(r.u32()?).ok()?;
    if r.rest().len() != count.checked_mul(ENTRY_LEN)? {
        return None;
    }
    let entries = (0..count)
        .map(|_| {
            let key = r.key()?;
            let offset = r.u64()?;
            let len = r.u32()?;
            Some((
                key,
                (len != 0).then_some(Loc {
                    offset,
                    len: len.into(),
                }),
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Index {
        coverage,
        parent,
        head,
        entries,
    })
}

#[cfg(test)]
mod adversarial;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::path::Path;

    use geom::{Channel, FaceId, LayerId, Level, Vec3};

    use super::*;
    use crate::fixture::{self, mesh, meta};
    use crate::test_dir::TestDir;

    fn key(face: u32, tx: u16) -> TileKey {
        TileKey::displacement(FaceId::new(face), Level::new(9).unwrap(), tx, 0).unwrap()
    }

    fn tile(seed: f32) -> CorrectionTile {
        let values: Vec<Vec3> = (0..65 * 65u16)
            .map(|k| Vec3::new(0.0, seed, seed * f32::from(k % 7)))
            .collect();
        CorrectionTile::from_values(65, &values).unwrap()
    }

    fn op(seq: u64) -> OpId {
        OpId::new(0, seq).unwrap()
    }

    #[test]
    fn tiles_survive_save_and_reopen() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
        f.write_displacement(key(2, 3), op(2), &tile(-0.25))
            .unwrap();
        f.save().unwrap();
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(
            opened.read_displacement(&key(1, 0)).unwrap(),
            Some((op(1), tile(0.5))),
            "tile 1"
        );
        assert_eq!(
            opened.read_displacement(&key(2, 3)).unwrap(),
            Some((op(2), tile(-0.25))),
            "tile 2"
        );
        assert_eq!(
            opened.read_displacement(&key(3, 0)).unwrap(),
            None,
            "absent"
        );
        assert_eq!(opened.head(), Some(op(2)), "head op");
    }

    #[test]
    fn unsaved_tiles_are_replayed_on_open() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
        f.save().unwrap();
        f.write_displacement(key(1, 0), op(2), &tile(0.75)).unwrap(); // no save
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(
            opened.read_displacement(&key(1, 0)).unwrap(),
            Some((op(2), tile(0.75))),
            "replayed"
        );
    }

    #[test]
    fn delta_indexes_stay_small_and_chain_correctly() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        for k in 0..200 {
            f.write_displacement(key(k, 0), op(u64::from(k) + 1), &tile(0.1))
                .unwrap();
        }
        let full = f.save().unwrap();
        let mut seq = 1000;
        for round in 0..40u32 {
            seq += 1;
            let t = tile(f32::from(u16::try_from(round).unwrap()));
            f.write_displacement(key(round % 200, 0), op(seq), &t)
                .unwrap();
            f.save().unwrap();
        }
        assert!(f.index_len(full).unwrap() > 200 * 24, "first index is full");
        assert!(
            f.last_index_len().unwrap() < 200,
            "later indexes are deltas (one entry)"
        );
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(opened.keys().count(), 200, "all tiles live");
        assert_eq!(
            opened.read_displacement(&key(39, 0)).unwrap().unwrap().1,
            tile(39.0),
            "last write wins"
        );
    }

    #[test]
    fn bad_header_is_rejected() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        std::fs::write(&path, b"NOTAPROJECTFILE!").unwrap();
        assert!(
            matches!(ProjectFile::open(&path), Err(StoreError::BadHeader)),
            "magic"
        );
    }

    #[test]
    fn a_damaged_old_tile_is_caught_when_read() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
        f.write_displacement(key(2, 0), op(2), &tile(0.5)).unwrap();
        f.save().unwrap();
        drop(f);
        let mut bytes = std::fs::read(&path).unwrap();
        let start = usize::try_from(ProjectFile::fixture_start()).unwrap();
        bytes[start + 40] ^= 0xFF; // inside the first tile's payload
        std::fs::write(&path, &bytes).unwrap();
        let opened = ProjectFile::open(&path).unwrap();
        assert!(
            matches!(
                opened.read_displacement(&key(1, 0)),
                Err(StoreError::Damaged { .. })
            ),
            "checksum"
        );
        assert!(
            opened.read_displacement(&key(2, 0)).unwrap().is_some(),
            "others still readable"
        );
    }
    /// Every live tile: its op and content.
    type State = BTreeMap<TileKey, (OpId, CorrectionTile)>;

    /// Writes a project with saves, overwrites and trailing unsaved tiles, recording after every
    /// record the file length and the expected live state (key → (op, tile)).
    fn scripted(path: &Path) -> Vec<(u64, State)> {
        let mut f = ProjectFile::create(path, meta(), mesh()).unwrap();
        let mut state = BTreeMap::new();
        let mut log = vec![(f.len(), state.clone())];
        let mut seq = 0;
        for round in 0..6u16 {
            for k in 0..5u32 {
                seq += 1;
                let t = tile(f32::from(round) + 0.1 * f32::from(u16::try_from(k).unwrap()));
                let at = key(k + u32::from(round % 2), 0);
                f.write_displacement(at, op(seq), &t).unwrap();
                state.insert(at, (op(seq), t));
                log.push((f.len(), state.clone()));
            }
            if round < 5 {
                f.save().unwrap(); // the last round stays unsaved
                log.push((f.len(), state.clone()));
            }
        }
        log
    }

    fn state_of(f: &ProjectFile) -> State {
        f.keys()
            .map(|k| (*k, f.read_displacement(k).unwrap().unwrap()))
            .collect()
    }

    #[test]
    fn truncation_anywhere_opens_with_the_last_complete_state() {
        let dir = TestDir::new();
        let full = dir.path("full.bcrf");
        let log = scripted(&full);
        let bytes = std::fs::read(&full).unwrap();
        let mut cuts: Vec<u64> = log.iter().map(|(len, _)| *len).collect();
        let mut x = 0x9E37_79B9_u64;
        let start = ProjectFile::fixture_start();
        for _ in 0..60 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            cuts.push(start + x % (bytes.len() as u64 - start));
        }
        for cut in cuts {
            let (good_len, want) = log.iter().rev().find(|(len, _)| *len <= cut).unwrap();
            let kept = &bytes[..usize::try_from(cut).unwrap()];
            // A crash can also leave the file at its full length with the unwritten blocks zeroed.
            let zero_filled = [kept, &vec![0; bytes.len() - kept.len()]].concat();
            for (variant, contents) in [("cut", kept.to_vec()), ("zero-filled", zero_filled)] {
                let path = dir.path("cut.bcrf");
                std::fs::write(&path, &contents).unwrap();
                let mut f = ProjectFile::open(&path).unwrap();
                assert_eq!(&state_of(&f), want, "{variant} at {cut}");
                // The torn tail is gone and the file is writable again.
                assert_eq!(
                    std::fs::metadata(&path).unwrap().len(),
                    *good_len,
                    "tail cut off: {variant} at {cut}"
                );
                f.write_displacement(key(99, 0), op(10_000), &tile(9.0))
                    .unwrap();
                f.save().unwrap();
                drop(f);
                let reopened = ProjectFile::open(&path).unwrap();
                let mut want = want.clone();
                want.insert(key(99, 0), (op(10_000), tile(9.0)));
                assert_eq!(
                    state_of(&reopened),
                    want,
                    "written after the {variant} at {cut}"
                );
            }
        }
    }

    #[test]
    fn every_save_point_opens_to_its_own_state() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        let mut snapshots = Vec::new();
        for round in 0..40u16 {
            f.write_displacement(
                key(u32::from(round % 7), 0),
                op(u64::from(round) + 1),
                &tile(f32::from(round)),
            )
            .unwrap();
            let point = f.save().unwrap();
            snapshots.push((point, state_of(&f)));
        }
        assert_eq!(
            f.save_points().unwrap(),
            snapshots.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            "chain"
        );
        drop(f); // save points are opened with a shared lock
        for (point, want) in &snapshots {
            let mut old = ProjectFile::open_at(&path, *point).unwrap();
            assert_eq!(&state_of(&old), want, "save point {point}");
            assert!(
                matches!(
                    old.write_displacement(key(0, 0), op(1), &tile(0.0)),
                    Err(StoreError::ReadOnly)
                ),
                "read-only"
            );
        }
        assert!(
            matches!(
                ProjectFile::open_at(&path, 17),
                Err(StoreError::NotASavePoint(17))
            ),
            "not an index"
        );
    }

    #[test]
    fn unknown_record_kinds_are_skipped() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        f.save().unwrap();
        f.append_raw(9, b"from a newer version").unwrap();
        f.write_displacement(key(2, 0), op(2), &tile(2.0)).unwrap();
        f.save().unwrap();
        f.append_raw(200, &[0; 64]).unwrap(); // unsaved, after the last index
        f.write_displacement(key(3, 0), op(3), &tile(3.0)).unwrap();
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(
            opened.keys().count(),
            3,
            "all three tiles, unknown records skipped"
        );
    }

    /// A project with three saves of five tiles each, and its bytes.
    fn saved_three_times(path: &Path) -> Vec<u8> {
        let mut f = ProjectFile::create(path, meta(), mesh()).unwrap();
        for seq in 1..=15u16 {
            f.write_displacement(
                key(u32::from(seq % 5), 0),
                op(u64::from(seq)),
                &tile(f32::from(seq)),
            )
            .unwrap();
            if seq % 5 == 0 {
                f.save().unwrap();
            }
        }
        drop(f);
        std::fs::read(path).unwrap()
    }

    #[test]
    fn damage_before_later_records_fails_open_and_leaves_the_file() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let good = saved_three_times(&path);
        let start = ProjectFile::fixture_start();
        let first = usize::try_from(start).unwrap();
        // The first record's head: a reserved byte (the head no longer parses), then the low
        // byte of its length (the scan runs off the record boundaries).
        for (what, byte) in [("reserved byte", first + 1), ("length", first + 4)] {
            let mut bad = good.clone();
            bad[byte] ^= 0x01;
            std::fs::write(&path, &bad).unwrap();
            let opened = ProjectFile::open(&path);
            assert!(
                matches!(
                    opened,
                    Err(StoreError::DamagedMidFile { offset }) if offset == start
                ),
                "{what}: {opened:?}"
            );
            assert_eq!(std::fs::read(&path).unwrap(), bad, "{what}: file untouched");
        }
    }

    #[test]
    fn a_torn_tail_is_still_cut_off() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let good = saved_three_times(&path);
        let want = state_of(&ProjectFile::open(&path).unwrap());
        // Half of a further tile record, then bytes that never were a record.
        let record = frame(
            KIND_TILE,
            &tile_payload(key(7, 0), op(99), b"h", 3, &[1, 2, 3]).unwrap(),
        )
        .unwrap();
        let torn = [
            good.as_slice(),
            &record[..record.len() / 2],
            &[0xA5; 300][..],
        ]
        .concat();
        std::fs::write(&path, &torn).unwrap();
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(state_of(&opened), want, "the saved state");
        assert_eq!(std::fs::read(&path).unwrap(), good, "the tail is cut off");
    }

    #[test]
    fn unsaved_writes_lost_out_of_order_are_cut_off() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        saved_three_times(&path);
        let mut f = ProjectFile::open(&path).unwrap();
        let want = state_of(&f);
        let saved = f.len();
        // Unsaved, so not synced: writeback may keep B and lose A.
        f.write_displacement(key(7, 0), op(100), &tile(7.0))
            .unwrap();
        let a_end = f.len();
        f.write_displacement(key(8, 0), op(101), &tile(8.0))
            .unwrap();
        drop(f);
        let mut bytes = std::fs::read(&path).unwrap();
        let a = usize::try_from(saved).unwrap()..usize::try_from(a_end).unwrap();
        bytes[a].fill(0);
        std::fs::write(&path, &bytes).unwrap();
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(state_of(&opened), want, "the last save, nothing after");
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            saved,
            "cut off at A, with B after it"
        );
    }

    #[test]
    fn tile_records_that_do_not_parse_are_skipped() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        f.save().unwrap();
        // Unsaved, so replayed: a payload too short to hold a key, and a tile whose op is
        // `u64::MAX` (the reserved site, as no real op can be).
        f.append_raw(KIND_TILE, b"short").unwrap();
        let mut reserved = tile_payload(key(2, 0), op(2), b"", 0, &[]).unwrap();
        reserved[KEY_LEN..KEY_LEN + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        f.append_raw(KIND_TILE, &reserved).unwrap();
        f.write_displacement(key(3, 0), op(3), &tile(3.0)).unwrap();
        let len = f.len();
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(
            opened.keys().copied().collect::<Vec<_>>(),
            vec![key(1, 0), key(3, 0)],
            "replay goes on past them"
        );
        assert_eq!(opened.head(), Some(op(3)), "head op");
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            len,
            "nothing cut off"
        );
    }

    #[test]
    fn compaction_leaves_its_source_untouched() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let good = saved_three_times(&path);
        let want = state_of(&ProjectFile::open(&path).unwrap());
        let torn = [good.as_slice(), &[0x5A; 100][..]].concat();
        std::fs::write(&path, &torn).unwrap();
        let out = dir.path("c.bcrf");
        compact(&path, &out).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            torn,
            "source bytes unchanged"
        );
        assert_eq!(
            state_of(&ProjectFile::open(&out).unwrap()),
            want,
            "the recovered state"
        );
    }

    #[test]
    fn other_layers_and_channels_round_trip_through_save_open_and_compaction() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let odd = TileKey::new(
            LayerId::new(3),
            Channel::from_raw(7),
            FaceId::new(5),
            Level::new(4).unwrap(),
            0,
            0,
        )
        .unwrap();
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_tile(odd, op(1), b"future header", b"future body")
            .unwrap();
        f.write_displacement(key(1, 0), op(2), &tile(1.0)).unwrap();
        f.save().unwrap();
        drop(f);
        let compacted = dir.path("c.bcrf");
        compact(&path, &compacted).unwrap();
        for p in [&path, &compacted] {
            let opened = ProjectFile::open(p).unwrap();
            let t = opened.read_tile(&odd).unwrap().unwrap();
            assert_eq!(
                (t.op, t.header.as_slice(), t.body.as_slice()),
                (op(1), &b"future header"[..], &b"future body"[..]),
                "{p:?}"
            );
            assert_eq!(opened.head(), Some(op(2)), "head op {p:?}");
        }
    }

    #[test]
    fn compaction_keeps_the_latest_state_and_shrinks_the_file() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        for seq in 1..=100u16 {
            f.write_displacement(
                key(u32::from(seq % 4), 0),
                op(u64::from(seq)),
                &tile(f32::from(seq)),
            )
            .unwrap();
            f.save().unwrap();
        }
        let before = state_of(&f);
        let big = f.len();
        drop(f);
        let out = dir.path("c.bcrf");
        compact(&path, &out).unwrap();
        let c = ProjectFile::open(&out).unwrap();
        assert_eq!(state_of(&c), before, "same latest state");
        assert!(
            c.len() < big / 10,
            "only live records kept: {} vs {big}",
            c.len()
        );
        assert_eq!(c.save_points().unwrap().len(), 1, "one full index");
    }

    fn small_key(face: u32) -> TileKey {
        TileKey::displacement(FaceId::new(face), Level::new(3).unwrap(), 0, 0).unwrap()
    }

    fn small_tile(seed: f32) -> CorrectionTile {
        let values: Vec<Vec3> = (0..9 * 9u16)
            .map(|k| Vec3::new(seed * f32::from(k % 5), -seed, 0.25))
            .collect();
        CorrectionTile::from_values(9, &values).unwrap()
    }

    #[test]
    fn a_reader_on_another_thread_reads_what_the_owner_does() {
        let dir = TestDir::new();
        let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
        let tiles = [(key(1, 0), tile(0.5)), (small_key(2), small_tile(0.125))];
        for (seq, (k, t)) in (1..).zip(&tiles) {
            f.write_displacement(*k, op(seq), t).unwrap();
        }
        f.save().unwrap();
        let jobs: Vec<(TileKey, Loc)> = tiles
            .iter()
            .map(|(k, _)| (*k, f.locate(k).unwrap()))
            .collect();
        let reader = f.reader();
        let read = std::thread::spawn(move || {
            let mut scratch = ReadScratch::new().unwrap();
            let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
            jobs.iter()
                .map(|(k, loc)| {
                    let op = reader
                        .read_displacement_into(k, *loc, &mut slot, &mut scratch)
                        .unwrap();
                    (op, slot.to_tile().unwrap())
                })
                .collect::<Vec<_>>()
        })
        .join()
        .unwrap();
        for ((k, t), got) in tiles.iter().zip(read) {
            let owner = f.read_displacement(k).unwrap().unwrap();
            assert_eq!(got, owner, "{k:?}: reader and owner agree");
            assert_eq!(&got.1, t, "{k:?}: the tile written");
        }
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        let mut scratch = ReadScratch::new().unwrap();
        slot.reset(9).unwrap();
        assert_eq!(
            f.read_displacement_into(&small_key(3), &mut slot, &mut scratch)
                .unwrap(),
            None,
            "absent"
        );
        assert_eq!(slot.side, 9, "an absent tile leaves the slot alone");
    }

    #[test]
    fn a_damaged_record_read_through_a_reader_is_damaged() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        f.write_displacement(key(2, 0), op(2), &tile(2.0)).unwrap();
        f.save().unwrap();
        let loc = f.locate(&key(1, 0)).unwrap();
        let reader = f.reader();
        let mut scratch = ReadScratch::new().unwrap();
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();

        // The record of another key at this location.
        let other = f.locate(&key(2, 0)).unwrap();
        assert!(
            matches!(
                reader.read_displacement_into(&key(1, 0), other, &mut slot, &mut scratch),
                Err(StoreError::Damaged { offset }) if offset == other.offset
            ),
            "wrong key"
        );
        assert!(
            matches!(
                reader.read_displacement_into(&small_key(1), loc, &mut slot, &mut scratch),
                Err(StoreError::Damaged { .. })
            ),
            "a key of another level"
        );

        reader
            .read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch)
            .unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let at = usize::try_from(loc.offset + loc.len / 2).unwrap();
        bytes[at] ^= 0x10;
        // The lock is advisory: an outside writer can still damage the file.
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&bytes[at..=at], loc.offset + loc.len / 2)
            .unwrap();
        assert!(
            matches!(
                reader.read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch),
                Err(StoreError::Damaged { offset }) if offset == loc.offset
            ),
            "flipped bit"
        );
        assert!(slot.view().is_err(), "a failed read leaves no tile");
        assert!(
            matches!(
                f.read_displacement_into(&key(1, 0), &mut slot, &mut scratch),
                Err(StoreError::Damaged { .. })
            ),
            "the owner's read agrees"
        );
    }

    #[test]
    fn a_reader_lets_go_of_the_file_when_its_owner_drops() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        f.save().unwrap();
        let loc = f.locate(&key(1, 0)).unwrap();
        let reader = f.reader();
        let clone = reader.clone();
        drop(reader);
        drop(f);
        let reopened = ProjectFile::open(&path);
        assert!(
            reopened.is_ok(),
            "the lock went with the owner: {reopened:?}"
        );
        let mut scratch = ReadScratch::new().unwrap();
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        slot.reset(65).unwrap();
        assert!(
            matches!(
                clone.read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch),
                Err(StoreError::Closed)
            ),
            "a reader of a closed file reads nothing"
        );
        assert!(slot.view().is_err(), "and leaves no tile");
    }

    #[test]
    fn a_body_that_decompresses_short_is_damaged() {
        let dir = TestDir::new();
        let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
        let k = small_key(1);
        let (header, body) = encode_displacement(&small_tile(0.5)).unwrap();
        // States the right size, but the frame holds one border value less.
        let short = &body[..body.len() - 4];
        let payload =
            tile_payload(k, op(1), &header, body.len(), &compress(short).unwrap()).unwrap();
        let loc = f.append(KIND_TILE, &payload).unwrap();
        let mut scratch = ReadScratch::new().unwrap();
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        assert!(
            matches!(
                f.reader()
                    .read_displacement_into(&k, loc, &mut slot, &mut scratch),
                Err(StoreError::Damaged { offset }) if offset == loc.offset
            ),
            "the reader rejects it"
        );
        assert!(slot.view().is_err(), "and leaves no tile");
        f.live.insert(k, loc);
        assert!(
            matches!(
                f.read_displacement(&k),
                Err(StoreError::Damaged { offset }) if offset == loc.offset
            ),
            "so does the owner"
        );
        assert!(
            matches!(f.read_tile(&k), Err(StoreError::Damaged { .. })),
            "and the raw read"
        );
    }

    #[test]
    fn meta_and_mesh_survive_create_and_open() {
        let dir = TestDir::new();
        let path = dir.path("r.bcrf");
        let mesh = Arc::new(geom::cube_sphere(2).unwrap());
        let project = crate::ProjectId::from_bits(42);
        let role = crate::Role::Recovery {
            base: Some(crate::SavePoint {
                project,
                offset: 1234,
                head: Some(op(9)),
            }),
        };
        let meta = Meta::new(project, role, &mesh);
        let mut f = ProjectFile::create(&path, meta, Arc::clone(&mesh)).unwrap();
        assert_eq!(*f.meta(), meta, "as created");
        assert_eq!(f.keys().count(), 0, "no tiles");
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        let point = f.save().unwrap();
        drop(f);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(*opened.meta(), meta, "the meta, reopened");
        assert_eq!(
            opened.mesh().to_bytes(),
            mesh.to_bytes(),
            "the mesh, reopened"
        );
        assert_eq!(opened.keys().count(), 1, "the tile");
        drop(opened);
        let old = ProjectFile::open_at(&path, point).unwrap();
        assert_eq!(*old.meta(), meta, "the meta, at a save point");
        assert_eq!(
            old.mesh().to_bytes(),
            mesh.to_bytes(),
            "the mesh, at a save point"
        );
    }

    #[test]
    fn a_meta_made_for_another_mesh_creates_nothing() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let other = geom::cube_sphere(2).unwrap();
        let created = ProjectFile::create(&path, fixture::meta_for(&other), mesh());
        assert!(
            matches!(created, Err(StoreError::MeshMismatch)),
            "{created:?}"
        );
        assert!(!path.exists(), "no file");
    }

    #[test]
    fn compaction_keeps_the_meta_and_the_mesh() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mesh = Arc::new(geom::cube_sphere(3).unwrap());
        let meta = fixture::meta_for(&mesh);
        let mut f = ProjectFile::create(&path, meta, Arc::clone(&mesh)).unwrap();
        for seq in 1..=10 {
            f.write_displacement(key(u32::try_from(seq % 3).unwrap(), 0), op(seq), &tile(1.0))
                .unwrap();
            f.save().unwrap();
        }
        drop(f);
        let out = dir.path("c.bcrf");
        compact(&path, &out).unwrap();
        compact_in_place(&path).unwrap();
        for p in [&out, &path] {
            let c = ProjectFile::open(p).unwrap();
            assert_eq!(*c.meta(), meta, "{p:?}: the meta");
            assert_eq!(c.mesh().to_bytes(), mesh.to_bytes(), "{p:?}: the mesh");
            assert_eq!(c.keys().count(), 3, "{p:?}: the tiles");
        }
    }

    #[test]
    fn locations_stay_valid_after_more_appends() {
        let dir = TestDir::new();
        let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
        let first = f.locate(&key(1, 0)).unwrap();
        let reader = f.reader();
        for seq in 2..6 {
            f.write_displacement(key(1, 0), op(seq), &tile(0.5))
                .unwrap();
            f.write_displacement(key(u32::try_from(seq).unwrap(), 0), op(seq), &tile(2.0))
                .unwrap();
            f.save().unwrap();
        }
        let mut scratch = ReadScratch::new().unwrap();
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        assert_eq!(
            reader
                .read_displacement_into(&key(1, 0), first, &mut slot, &mut scratch)
                .unwrap(),
            op(1),
            "the old record is still there"
        );
        assert_eq!(slot.to_tile().unwrap(), tile(1.0), "with its tile");
        let latest = f.locate(&key(1, 0)).unwrap();
        assert_ne!(latest, first, "a newer record");
        assert_eq!(
            reader
                .read_displacement_into(&key(1, 0), latest, &mut slot, &mut scratch)
                .unwrap(),
            op(5),
            "the newest edit"
        );
        assert_eq!(slot.to_tile().unwrap(), tile(0.5), "the newest tile");
    }
}
