//! The project file: a header, a `Meta` and a `Mesh` record that make it self-contained, then
//! append-only records (see `record`), with save points that form a chain.
//!
//! The two leading records are part of the file's identity, like the header: they are written
//! and synced when the file is created and never change. Recovery after a crash starts after
//! them, and damage to them fails the open rather than being cut off as a torn tail.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt as _};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use geom::{BaseMesh, Channel, CorrectionTile, TileKey};

use self::bounds_io::{bounds_record, read_bounds_at};
#[cfg(test)]
use self::format::Work;
#[cfg(test)]
use self::format::read_head;
#[cfg(test)]
use self::format::take_work;
use self::format::{
    Coverage, FULL_EVERY, Index, Search, TileRecord, Version, chain_from, check_header, find_index,
    index_payload, leading_bytes, load_chain, mesh_record, read_exact, read_index, read_leading,
    read_save_point, scan_heads, tile_payload, tile_record, upgrade_header,
};
use self::journal_io::batch_locs;
use self::lock::{Access, Lock, LockedFile, open_locked, open_readable, open_writable};
use self::reader::read_displacement_at;
use self::stroke_io::{read_stroke_at, stroke_locs};
use crate::OpId;
use crate::bounds::bounds_payload;
use crate::codec::{CodecError, compress, decompress, encode_displacement};
use crate::meta::{Meta, SavePoint};
use crate::record::{Head, KIND_BOUNDS, KIND_INDEX, KIND_TILE, frame, verify};
use crate::shared_map::SharedMap;
use crate::slot::CorrectionSlot;
use crate::stamp::Stamp;
use crate::tile_bounds::TileBounds;

mod base;
mod bounds_io;
mod compact;
mod display_io;
mod error;
mod format;
mod journal_io;
mod lock;
mod reader;
mod stroke_io;

pub(crate) use base::BaseFile;
pub(crate) use compact::sync_parent;
pub use compact::{compact, compact_in_place};
#[cfg(test)]
pub(crate) use display_io::version_2_record;
pub(crate) use display_io::{FramedDisplay, display_record, frame_display_on, unknown_display_len};
pub use error::FileError;
#[cfg(any(feature = "fuzzing", test))]
pub(crate) use format::HEADER_LEN;
pub use format::{MAX_MESH_RAW, MAX_RAW};
pub(crate) use journal_io::frame_batch;
pub(crate) use lock::open_regular;
pub(crate) use reader::{FileReader, FileSyncer, FramedTile, ReadScratch, frame_displacement};
pub(crate) use stroke_io::frame_stroke;
#[cfg(test)]
pub(crate) use stroke_io::take_stroke_reads;

/// The mode a user's project file is created with: the umask decides who else may read it, as
/// for any file the user saves.
pub(crate) const SHARED_MODE: u32 = 0o666;
/// The mode recovery, scratch and temporary files are created with: only the user may read or
/// write them, whatever the umask, as they hold the user's unsaved work.
pub(crate) const PRIVATE_MODE: u32 = 0o600;
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

impl Loc {
    /// The record's length in bytes, framing included.
    pub(crate) const fn len(self) -> u64 {
        self.len
    }

    /// The byte halfway through the record: for tests that damage one.
    #[cfg(test)]
    pub(crate) const fn middle(self) -> u64 {
        self.offset + self.len / 2
    }
}

/// What a save point names a record as: a tile's corrections, or the bounds of a tile of the
/// spatial index's level (see [`crate::bounds`]). Each kind is keyed by its tile, and a tile can
/// have one of each.
///
/// Tiles order before bounds, so a map keyed by names lists every tile first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Name {
    /// The tile record of a key.
    Tile(TileKey),
    /// The bounds record of a key.
    Bounds(TileKey),
}

impl Name {
    /// The tile, if this names a tile record.
    const fn tile(&self) -> Option<&TileKey> {
        match self {
            Self::Tile(key) => Some(key),
            Self::Bounds(_) => None,
        }
    }

    /// The tile, if this names a bounds record.
    const fn bounds(&self) -> Option<&TileKey> {
        match self {
            Self::Bounds(key) => Some(key),
            Self::Tile(_) => None,
        }
    }
}

/// Records of one kind, by the key of their tile, in key order.
type Records = Vec<(TileKey, Loc)>;

/// Tiles changed since an earlier save point, in key order, each with its record there, if it
/// had one ([`ProjectFile::tiles_changed_since`]).
pub(crate) type Changed = Vec<(TileKey, Option<Loc>)>;

/// Splits a map keyed by [`Name`] into its tile records and its bounds records, each in key
/// order.
///
/// One pass over the map for each kind, with no lookups: a map keyed by names lists each kind
/// in key order, so each half comes out sorted, as [`SharedMap`] needs.
fn split_names(names: &BTreeMap<Name, Loc>) -> (Records, Records) {
    let tiles = names
        .iter()
        .filter_map(|(name, &loc)| name.tile().map(|key| (*key, loc)))
        .collect();
    let bounds = names
        .iter()
        .filter_map(|(name, &loc)| name.bounds().map(|key| (*key, loc)))
        .collect();
    (tiles, bounds)
}

/// Why the walk of the records after the last save point stopped (see
/// [`ProjectFile::walk_tail`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// At an Index that verifies but didn't load as the save point: damage.
    Index,
    /// At the end of the heads, or a record that doesn't verify.
    Other,
}

/// What opening at the latest state does with the records after the last save point (a torn
/// tail, or an operation that never finished).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Repair {
    /// Cut them off, so the file is writable again. Opens read-write.
    Truncate,
    /// Fail with [`FileError::NeedsRepair`]: the file can't be written. Opens read-only.
    Refuse,
    /// Leave them in place and open the save point read-only, for compaction and inspection.
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

/// The latest save point of a [`ProjectFile`], kept as the maps a [`BaseFile`] of it reads, so
/// that making one copies nothing (see [`ProjectFile::base`]).
///
/// It is derived, not reread: when a save point is loaded (opening the file), its maps are
/// built from the Index chain once; each save point written after that has the previous one's
/// maps with the entries the new Index adds put in ([`SharedMap::with`]), whether that Index
/// is a delta or full. So a save costs what it wrote, not what the project holds, and the maps
/// of the save point before stay valid and shared with any [`BaseFile`] still holding them.
/// Tests check every derived save point against the chain read back from the file.
#[derive(Debug, Clone)]
struct SavedState {
    /// Where the save point's Index starts.
    offset: u64,
    /// The head op its Index records; `None` before any edit.
    head: Option<OpId>,
    /// Each tile record it names, by key.
    tiles: SharedMap<TileKey, Loc>,
    /// Each bounds record it names, by key.
    bounds: SharedMap<TileKey, Loc>,
    /// Each stroke record it holds, by offset: every one before its Index in the file (no Index
    /// names them, see [`crate::stroke_record`]).
    strokes: SharedMap<u64, Loc>,
    /// Each dab batch it holds, by offset: every one before its Index in the file (no Index
    /// names them, see [`crate::journal`]).
    batches: SharedMap<u64, Loc>,
    /// The save point it was derived from by putting entries in, never taking one out; `None`
    /// for one loaded from the file, or the file's first. With the file it was derived in, this
    /// proves the save point holds every record its parent does (see [`BaseFile::extends`]).
    parent: Option<SavePoint>,
}

impl SavedState {
    /// The save point at `offset`, with head op `head`, whose live records are `live` and
    /// whose stroke records and dab batches are `records`: loaded from the file, with no parent.
    fn loaded(offset: u64, head: Option<OpId>, live: &BTreeMap<Name, Loc>, records: Held) -> Self {
        let (tiles, bounds) = split_names(live);
        Self {
            offset,
            head,
            tiles: SharedMap::from_sorted(tiles),
            bounds: SharedMap::from_sorted(bounds),
            strokes: by_offset(records.strokes),
            batches: by_offset(records.batches),
            parent: None,
        }
    }
}

/// The records a save point holds without its Index naming them, each in file order: every
/// stroke record and dab batch before the Index.
#[derive(Debug, Default)]
struct Held {
    /// The stroke records (see [`crate::stroke_record`]).
    strokes: Vec<Loc>,
    /// The dab batches (see [`crate::journal`]).
    batches: Vec<Loc>,
}

impl Held {
    /// The stroke records and dab batches of `heads`, the heads of every record before an Index.
    fn of(heads: &[(u64, Head)]) -> Self {
        Self {
            strokes: stroke_locs(heads).collect(),
            batches: batch_locs(heads).collect(),
        }
    }

    /// The stroke records of `latest`, the latest save point, before `index`, the offset of an
    /// Index at or before its own: the ones that Index's save point holds. No batch: this is for
    /// a base, which a store reads tiles and copies stroke records through, while a journal is
    /// read from the recovery file it is in, never through a base.
    fn strokes_before(latest: &SavedState, index: u64) -> Self {
        let strokes = latest
            .strokes
            .iter()
            .map(|(_, &loc)| loc)
            .take_while(|loc| loc.offset < index);
        Self {
            strokes: strokes.collect(),
            batches: Vec::new(),
        }
    }
}

/// `locs`, in file order, keyed by their offsets.
fn by_offset(locs: Vec<Loc>) -> SharedMap<u64, Loc> {
    SharedMap::from_sorted(locs.into_iter().map(|loc| (loc.offset, loc)))
}

/// An Index appended but not yet the latest save point: what [`ProjectFile::commit_index`]
/// needs to make it so.
#[derive(Debug)]
struct Appended {
    /// Whether it lists every live record or only those changed since its parent.
    coverage: Coverage,
    /// Where it is.
    loc: Loc,
    /// The head op it records.
    head: Option<OpId>,
    /// Each tile record the new save point names: the last save point's with those written
    /// since put in, as of the cut it was written for, if any (see [`SavedState`]).
    tiles: SharedMap<TileKey, Loc>,
    /// The same for bounds records.
    bounds: SharedMap<TileKey, Loc>,
}

/// A tile record in the file that nothing names yet ([`ProjectFile::append_unnamed`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnnamedCopy {
    /// The op of the record.
    op: OpId,
    /// Where it is.
    loc: Loc,
}

/// An open project file.
pub struct ProjectFile {
    /// The file's identity, from its `Meta` record.
    meta: Meta,
    /// The base mesh, from its `Mesh` record.
    mesh: Arc<BaseMesh>,
    /// The `Mesh` record itself, framed and compressed: a file made over the same mesh (a
    /// recovery file, a project saved under a new name) copies it rather than compressing the
    /// mesh again.
    mesh_record: Arc<[u8]>,
    /// The open file, which holds the lock. [`FileReader`]s hold it weakly, so dropping the
    /// `ProjectFile` releases the lock and closes the file (once any read in flight ends).
    file: Arc<LockedFile>,
    /// Where the next record goes.
    end: u64,
    /// Each live tile's and bounds' latest record, saved or not: what reads find.
    ///
    /// **Invariant:** `live` is `saved`'s records with `pending`'s put in over them (just
    /// `pending` before the first save point). Each save point's maps are derived from the
    /// last one's and the records written since (see [`SavedState`]), which is right only while
    /// this holds. So every record appended goes into `live` and `pending` together, through
    /// [`ProjectFile::note`]; loading a save point sets `live` and `saved` from the same chain,
    /// with `pending` empty; and a save point leaves in `pending` only the records past its
    /// cut, which `live` holds too. Tests assert it before and after every save point.
    live: BTreeMap<Name, Loc>,
    /// Entries written since the last Index (see the invariant on `live`).
    pending: BTreeMap<Name, Loc>,
    /// The stroke records appended since the last save point, in file order: the next one holds
    /// them, as it holds every record before it (see [`crate::stroke_record`]).
    strokes: Vec<Loc>,
    /// The ops of the stroke records the file holds, as packed bits: a stroke record of an op
    /// already here isn't appended again (a save retried after a failure, say). Read from the
    /// records' op fields the first time a stroke record is appended, never on open; `None`
    /// until then.
    stroke_ops: Option<BTreeSet<u64>>,
    /// The dab batches appended since the last save point, in file order: the next one holds
    /// them, as it holds every record before it (see [`crate::journal`]).
    batches: Vec<Loc>,
    /// The latest save point, the parent of the next one; `None` until the first save (see the
    /// invariant on `live`).
    saved: Option<SavedState>,
    /// Delta indexes since the last full one.
    deltas: u32,
    /// The greatest op written (see [`ProjectFile::head`]).
    head: Option<OpId>,
    /// Whether writes are accepted.
    access: Access,
    /// The format version the file's header states: an older file's is rewritten to the
    /// current one before the first record appended to it (see [`upgrade_header`]).
    version: Version,
    /// The cut the next save point is written for, while one is open (see [`Cut`]).
    cut: Option<Cut>,
    /// Makes the next save point's derivation leave out its first new tile record: for tests
    /// of the guard in [`ProjectFile::append_index`].
    #[cfg(test)]
    skew_derivation: bool,
    /// Makes the next [`ProjectFile::save`] fail with an I/O error after its records' sync,
    /// before its Index, or the next [`ProjectFile::write_save_point`] before it writes anything:
    /// for tests of a save, or a flush, made again.
    #[cfg(test)]
    fail_next_save: bool,
}

/// A save point in the making: the state of the file as of a write of the store (the cut),
/// which records appended since may have moved past.
///
/// A recovery flush snapshots the store's dirty tiles as of its latest write, and its save
/// point must name exactly that state: no more, or a crash could reopen half of a later stroke,
/// or the later dabs of the stroke in progress. But while the flush is in flight, a later
/// write's tile may be written back to make room, and land in the file before the save point.
/// So the cut remembers, for each tile, where its record was before the first record past the
/// cut was appended, and the save point names that one; the later records wait for the next
/// save point. A record at or before the cut appended after one past it (the flush's own copy
/// of a tile written back since) is the cut's state, and takes the remembered place instead.
///
/// Records are placed by their [`Stamp`], not their op: one stroke keeps one op across many
/// dabs, and a cut during a long stroke falls between two dabs of the same op.
#[derive(Debug)]
struct Cut {
    /// The newest write the cut holds; records of later writes (greater stamps) are past it.
    at: Stamp,
    /// The head op of the cut's state: the newest op of a record at or before the cut.
    head: Option<OpId>,
    /// For each tile or bounds with a record past the cut, where its record at the cut is.
    prior: BTreeMap<Name, Prior>,
    /// Where the last record the cut's save point will name ends: the file's end when the cut
    /// opened, moved by every record appended at or before the cut since. A sync must cover it
    /// before the save point is written.
    named_end: u64,
}

/// Where a tile's record at a cut is, for each kind of Index the cut's save point may be.
#[derive(Debug, Clone, Copy)]
struct Prior {
    /// Its latest record at the cut, for a full Index; `None` if it had none (it is left out).
    live: Option<Loc>,
    /// Its latest record written since the last save point at the cut, for a delta Index;
    /// `None` if it had none (the parent's entry stands, so it is left out).
    pending: Option<Loc>,
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
            .field("last_index", &self.last_index())
            .field("deltas", &self.deltas)
            .field("head", &self.head)
            .field("access", &self.access)
            .field("cut", &self.cut.as_ref().map(|cut| cut.at))
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
    /// without its header (and fail with [`FileError::BadHeader`]) or lock it first (and this
    /// fails with [`FileError::Locked`], leaving the empty file behind). Only two processes
    /// creating and opening one new path at the same instant can meet this.
    ///
    /// # Errors
    ///
    /// [`FileError::MeshMismatch`] if `meta`'s mesh hash isn't `mesh`'s;
    /// [`FileError::TooLarge`] if the mesh's canonical bytes exceed [`MAX_MESH_RAW`] (nothing is
    /// created then); [`FileError::Io`], including when `path` already exists;
    /// [`FileError::Locked`].
    pub fn create(path: &Path, meta: Meta, mesh: Arc<BaseMesh>) -> Result<Self, FileError> {
        let record = mesh_record(&meta, &mesh)?;
        Self::create_with(path, meta, mesh, record.into(), SHARED_MODE)
    }

    /// Creates a new file at `path`, as [`ProjectFile::create`] does, readable and writable by
    /// the user only ([`PRIVATE_MODE`]): a recovery file.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::create`].
    pub(crate) fn create_private(
        path: &Path,
        meta: Meta,
        mesh: Arc<BaseMesh>,
    ) -> Result<Self, FileError> {
        let record = mesh_record(&meta, &mesh)?;
        Self::create_with(path, meta, mesh, record.into(), PRIVATE_MODE)
    }

    /// [`ProjectFile::create_like_mode`] with [`SHARED_MODE`], for tests that make files beside
    /// a project and don't care who may read them.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::create_like_mode`].
    #[cfg(test)]
    pub(crate) fn create_like(path: &Path, meta: Meta, like: &Self) -> Result<Self, FileError> {
        Self::create_like_mode(path, meta, like, SHARED_MODE)
    }

    /// Creates a new file at `path`, as [`ProjectFile::create`] does, over `like`'s mesh: its
    /// `Mesh` record is copied as it is, without compressing (or hashing) the mesh again.
    /// `meta` must be over the same mesh (made from `like`'s meta with [`Meta::with_role`],
    /// say). The file is created with permission bits `mode` (less the umask's):
    /// [`PRIVATE_MODE`] for a recovery, scratch or temporary file, [`SHARED_MODE`] for a
    /// user's project file.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::create`]; [`FileError::MeshMismatch`] if `meta`'s mesh hash isn't
    /// `like`'s.
    pub(crate) fn create_like_mode(
        path: &Path,
        meta: Meta,
        like: &Self,
        mode: u32,
    ) -> Result<Self, FileError> {
        if meta.mesh_hash() != like.meta.mesh_hash() {
            return Err(FileError::MeshMismatch);
        }
        Self::create_with(
            path,
            meta,
            Arc::clone(&like.mesh),
            Arc::clone(&like.mesh_record),
            mode,
        )
    }

    /// Creates the file at `path` with `meta` and the framed `Mesh` record `mesh_record` of
    /// `mesh`, which the caller has checked against `meta`, with permission bits `mode` (less
    /// the umask's), set as the file is made: it is never more open than that.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::create`].
    fn create_with(
        path: &Path,
        meta: Meta,
        mesh: Arc<BaseMesh>,
        mesh_record: Arc<[u8]>,
        mode: u32,
    ) -> Result<Self, FileError> {
        let leading = leading_bytes(&meta, &mesh_record)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(mode)
            .open(path)?;
        let file = LockedFile::new(file, Lock::Exclusive)?;
        file.write_all_at(&leading, 0)?;
        file.sync_all()?;
        sync_parent(path)?;
        Ok(Self {
            meta,
            mesh,
            mesh_record,
            file: Arc::new(file),
            end: leading.len() as u64,
            live: BTreeMap::new(),
            pending: BTreeMap::new(),
            strokes: Vec::new(),
            stroke_ops: None,
            batches: Vec::new(),
            saved: None,
            deltas: 0,
            head: None,
            access: Access::ReadWrite,
            version: Version::Current,
            cut: None,
            #[cfg(test)]
            skew_derivation: false,
            #[cfg(test)]
            fail_next_save: false,
        })
    }

    /// Opens the file at `path` at its latest state, recovering from a crash: the state is the
    /// last good save point, and whatever follows it is discarded.
    ///
    /// **A save point is the unit of durability.** Records after the last good save point
    /// belong to an operation that never finished (a save, or a recovery flush, cut short):
    /// replaying them could bring back half a stroke or half a save. So they are cut off, and
    /// the file opens at the save point, as it was when that operation began. Records before a
    /// good save point were complete when it was written; damage inside them is caught by their
    /// checksums when they are read.
    ///
    /// The file must begin with its `Meta` and `Mesh` records, whose mesh must match the meta's
    /// hash. They are checked before anything else: damage there fails the open and leaves the
    /// file untouched, as a damaged header does. Recovery starts after them.
    ///
    /// Before anything is cut off, the records after the save point are walked while they
    /// verify, and the rest searched for a save point: an Index that verifies, parses and has a
    /// plausible parent. Saves sync their records before writing the Index, so everything
    /// before a good Index was durable: finding one means the walk stopped at damage, not at a
    /// torn tail, and the open fails, leaving the file as it is. So does an Index the walk
    /// reaches that verifies but didn't load as the save point. Otherwise the rest is a torn or
    /// unsynced tail (even if some of its records verify, since writeback may have kept a later
    /// write and lost an earlier one), and is cut off.
    ///
    /// The file is opened read-write and locked exclusively. If it can't be opened for writing
    /// (no permission, or a read-only file system), it is opened read-only with a shared lock
    /// instead, which works only if nothing follows its last save point. If the path was
    /// replaced (by [`compact_in_place`]) between opening and locking, the new file is opened
    /// instead.
    ///
    /// # Errors
    ///
    /// [`FileError::DamagedMidFile`], [`FileError::NeedsRepair`], [`FileError::Locked`],
    /// [`FileError::BadHeader`], [`FileError::MissingMeta`], [`FileError::MeshMismatch`],
    /// [`FileError::BadMesh`], [`FileError::Io`].
    pub fn open(path: &Path) -> Result<Self, FileError> {
        Self::open_sharing(path, None)
    }

    /// Opens the file at `path` as [`ProjectFile::open`] does, sharing `like`'s base mesh if
    /// the file's `Mesh` record is byte for byte `like`'s: a recovery file opened beside its
    /// project, whose mesh is then neither decompressed nor parsed again. A file whose record
    /// differs is read in full and checked as usual.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::open`].
    pub(crate) fn open_sharing_mesh(path: &Path, like: &Self) -> Result<Self, FileError> {
        Self::open_sharing(path, Some(like))
    }

    /// [`ProjectFile::open`], sharing `like`'s mesh where its `Mesh` record matches.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::open`].
    fn open_sharing(path: &Path, like: Option<&Self>) -> Result<Self, FileError> {
        match open_locked(path, open_writable, Lock::Exclusive) {
            Ok(file) => Self::open_latest(file, Repair::Truncate, like),
            Err(FileError::Io(e))
                if matches!(
                    e.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
                ) =>
            {
                Self::open_latest(
                    open_locked(path, open_readable, Lock::Shared)?,
                    Repair::Refuse,
                    like,
                )
            }
            Err(e) => Err(e),
        }
    }

    /// Opens `file`, already locked, at its latest state, as [`ProjectFile::open`] does; `repair`
    /// says what happens to a tail after the last save point, and with it the access. `file`
    /// must be writable and locked exclusively for [`Repair::Truncate`]. The mesh is shared
    /// with `like` where its `Mesh` record matches.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::open`].
    fn open_latest(
        file: LockedFile,
        repair: Repair,
        like: Option<&Self>,
    ) -> Result<Self, FileError> {
        let access = repair.access();
        let len = file.metadata()?.len();
        let version = check_header(&file, len)?;
        let leading = read_leading(&file, len, like)?;
        let heads = scan_heads(&file, leading.end, len)?;
        let mut this = Self {
            meta: leading.meta,
            mesh: leading.mesh,
            mesh_record: leading.mesh_record,
            file: Arc::new(file),
            end: leading.end,
            live: BTreeMap::new(),
            pending: BTreeMap::new(),
            strokes: Vec::new(),
            stroke_ops: None,
            batches: Vec::new(),
            saved: None,
            deltas: 0,
            head: None,
            access,
            version,
            cut: None,
            #[cfg(test)]
            skew_derivation: false,
            #[cfg(test)]
            fail_next_save: false,
        };
        let tail = this.load_base(&heads, len)?;
        if this.end < len {
            let (stop, stopped_at) = this.walk_tail(heads.get(tail..).unwrap_or_default())?;
            let damaged = stop == Stop::Index
                || match find_index(&this.file, stopped_at, len)? {
                    Search::Index(_) | Search::TooCostly => true,
                    Search::Nothing => false,
                };
            if damaged {
                return Err(FileError::DamagedMidFile { offset: stopped_at });
            }
            match repair {
                Repair::Truncate => {
                    this.file.set_len(this.end)?;
                    this.file.sync_all()?;
                }
                Repair::Refuse => return Err(FileError::NeedsRepair { offset: this.end }),
                Repair::Ignore => {}
            }
        }
        Ok(this)
    }

    /// Opens the file at `path` to look at, never to write: read-only, with a shared lock, at
    /// its latest save point as [`ProjectFile::open`] finds it, but with whatever follows that
    /// save point left in place. Listing recovery files after a crash opens them so: a file a
    /// running instance holds is locked, and one being restored isn't repaired by a look. A
    /// project restored onto an earlier save point holds its project file so, as it never
    /// writes it.
    ///
    /// # Errors
    ///
    /// [`FileError::Locked`] if the file is open for writing; as [`ProjectFile::open`]
    /// otherwise, except [`FileError::NeedsRepair`].
    pub(crate) fn inspect(path: &Path) -> Result<Self, FileError> {
        Self::open_latest(
            open_locked(path, open_readable, Lock::Shared)?,
            Repair::Ignore,
            None,
        )
    }

    /// Opens the project at `path` read-only, with a shared lock, at the save point (an Index)
    /// at byte `index_offset`.
    ///
    /// Besides the Index chain, it walks every record head before the Index (8 bytes each, as an
    /// open at the latest save point walks every head of the file) to find the stroke records the
    /// save point holds; it reads none of them.
    ///
    /// # Errors
    ///
    /// [`FileError::NotASavePoint`] if no good Index starts there, [`FileError::Damaged`] if
    /// its parent chain is, [`FileError::Locked`] if the file is open for writing,
    /// [`FileError::BadHeader`], [`FileError::MissingMeta`], [`FileError::MeshMismatch`],
    /// [`FileError::BadMesh`], [`FileError::Io`].
    pub fn open_at(path: &Path, index_offset: u64) -> Result<Self, FileError> {
        let file = open_locked(path, open_readable, Lock::Shared)?;
        let len = file.metadata()?.len();
        let version = check_header(&file, len)?;
        let leading = read_leading(&file, len, None)?;
        if index_offset < leading.end || read_index(&file, index_offset, len)?.is_none() {
            return Err(FileError::NotASavePoint(index_offset));
        }
        let chain = load_chain(&file, index_offset, len)?.ok_or(FileError::Damaged {
            offset: index_offset,
        })?;
        // The stroke records and dab batches the save point holds: every one before its Index.
        // Only their heads are read, as on any open.
        let before = scan_heads(&file, leading.end, index_offset)?;
        let saved = SavedState::loaded(index_offset, chain.head, &chain.live, Held::of(&before));
        Ok(Self {
            meta: leading.meta,
            mesh: leading.mesh,
            mesh_record: leading.mesh_record,
            file: Arc::new(file),
            end: len,
            live: chain.live,
            pending: BTreeMap::new(),
            strokes: Vec::new(),
            stroke_ops: None,
            batches: Vec::new(),
            saved: Some(saved),
            deltas: chain.deltas,
            head: chain.head,
            access: Access::ReadOnly,
            version,
            cut: None,
            #[cfg(test)]
            skew_derivation: false,
            #[cfg(test)]
            fail_next_save: false,
        })
    }

    /// Loads the last Index of `heads` whose record and parent chain are good, moves `end` past
    /// it, and returns the position in `heads` of the first record after it (0 if there is
    /// none, and `end` stays after the leading records).
    ///
    /// A parent is followed only to an Index of `heads`, where every parent a writer names is,
    /// and each Index is read at most once however many chains lead to it, so a file of many
    /// Indexes naming one large or many made-up parents costs one pass, not one per Index.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    fn load_base(&mut self, heads: &[(u64, Head)], len: u64) -> Result<usize, FileError> {
        let indexes: BTreeSet<u64> = heads
            .iter()
            .filter(|(_, head)| head.kind == KIND_INDEX)
            .map(|&(offset, _)| offset)
            .collect();
        let mut read: BTreeMap<u64, Option<Rc<Index>>> = BTreeMap::new();
        let file = &self.file;
        let mut index_at = |at: u64| -> io::Result<Option<Rc<Index>>> {
            if !indexes.contains(&at) {
                return Ok(None);
            }
            if let Some(known) = read.get(&at) {
                return Ok(known.clone());
            }
            let index = read_save_point(file, at, len)?.map(Rc::new);
            read.insert(at, index.clone());
            Ok(index)
        };
        for (at, &(offset, head)) in heads.iter().enumerate().rev() {
            if head.kind != KIND_INDEX {
                continue;
            }
            if let Some(chain) = chain_from(offset, &mut index_at)? {
                let before = heads.get(..at).unwrap_or_default();
                self.saved = Some(SavedState::loaded(
                    offset,
                    chain.head,
                    &chain.live,
                    Held::of(before),
                ));
                self.live = chain.live;
                self.deltas = chain.deltas;
                self.head = chain.head;
                self.end = offset + head.total();
                return Ok(at + 1);
            }
        }
        Ok(0)
    }

    /// Walks the records after the save point (`heads`, from `end` on) while they verify, and
    /// says why it stopped and where. Nothing is loaded from them: they are about to be
    /// discarded, and the walk only looks for proof that they are damage rather than a torn
    /// tail. A record that verifies is stepped over, whatever its kind (a tile, or a kind from
    /// a newer version), but an Index stops the walk: any Index this late failed to load as
    /// the save point, whether or not it parses, and is damage.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    fn walk_tail(&self, heads: &[(u64, Head)]) -> Result<(Stop, u64), FileError> {
        let mut at = self.end;
        for &(offset, head) in heads {
            let raw = read_exact(&self.file, offset, head.total())?;
            match verify(&raw) {
                None => break,
                Some((KIND_INDEX, _)) => return Ok((Stop::Index, offset)),
                Some(_) => at = offset + head.total(),
            }
        }
        Ok((Stop::Other, at))
    }

    /// Appends a tile record: `key`'s new content, written by `op`. The body is compressed.
    /// Durable once [`ProjectFile::save`] returns.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`] (including a body larger than
    /// [`MAX_RAW`]), [`FileError::Io`].
    pub fn write_tile(
        &mut self,
        key: TileKey,
        op: OpId,
        header: &[u8],
        body: &[u8],
    ) -> Result<(), FileError> {
        self.write_tile_stamped(key, op, Stamp::ZERO, header, body)
    }

    /// [`ProjectFile::write_tile`] for a tile store's write stamped `stamp`, which places the
    /// record against an open cut (see [`Cut`]).
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::write_tile`].
    fn write_tile_stamped(
        &mut self,
        key: TileKey,
        op: OpId,
        stamp: Stamp,
        header: &[u8],
        body: &[u8],
    ) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        // Checked before compressing: no reader would accept it.
        if !u32::try_from(body.len()).is_ok_and(|n| n <= MAX_RAW) {
            return Err(FileError::TooLarge);
        }
        let packed = compress(body)?;
        let payload = tile_payload(key, op, header, body.len(), &packed)?;
        let loc = self.append(KIND_TILE, &payload)?;
        self.note(Name::Tile(key), Some(op), stamp, loc);
        Ok(())
    }

    /// Appends a displacement tile (see [`ProjectFile::write_tile`]).
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::write_tile`]; [`FileError::NotDisplacement`] if `key` is of another
    /// channel, [`FileError::WrongSide`] if the tile's side isn't the key's.
    pub fn write_displacement(
        &mut self,
        key: TileKey,
        op: OpId,
        tile: &CorrectionTile,
    ) -> Result<(), FileError> {
        self.write_displacement_stamped(key, op, Stamp::ZERO, tile)
    }

    /// [`ProjectFile::write_displacement`] for a tile store's write stamped `stamp`, which
    /// places the record against an open cut (see [`Cut`]): a dirty slot written back.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::write_displacement`].
    pub(crate) fn write_displacement_stamped(
        &mut self,
        key: TileKey,
        op: OpId,
        stamp: Stamp,
        tile: &CorrectionTile,
    ) -> Result<(), FileError> {
        check_displacement(&key, tile.side())?;
        let (header, body) = encode_displacement(tile)?;
        self.write_tile_stamped(key, op, stamp, &header, &body)
    }

    /// Appends a tile record framed elsewhere (by [`frame_displacement`], on another thread),
    /// and records it as [`ProjectFile::write_tile`] does: `key`'s latest record, written since
    /// the last save point, with its op counted towards the head, and placed against an open
    /// cut by the stamp of the write it copies.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::Io`].
    pub(crate) fn append_framed(&mut self, tile: &FramedTile) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let loc = self.append_record(&tile.record)?;
        self.note(Name::Tile(tile.key), Some(tile.op), tile.stamp, loc);
        Ok(())
    }

    /// Records that `name`'s record, written by `op` (a tile record) or by no op (a bounds
    /// record), was appended at `loc`: its latest record, written since the last save point, with
    /// `op` counted towards the head. While a cut is open (see [`Cut`]), `stamp` places the
    /// record: the first record of `name` past the cut leaves the cut its place before, and a
    /// record at or before the cut after one past it is the cut's state, not the latest.
    fn note(&mut self, name: Name, op: Option<OpId>, stamp: Stamp, loc: Loc) {
        if let Some(cut) = &mut self.cut {
            if stamp > cut.at {
                let before = Prior {
                    live: self.live.get(&name).copied(),
                    pending: self.pending.get(&name).copied(),
                };
                cut.prior.entry(name).or_insert(before);
            } else {
                // First, as the branch may return early: every record at the cut is named.
                cut.named_end = cut.named_end.max(loc.offset + loc.len);
                cut.head = cut.head.max(op);
                if let Some(prior) = cut.prior.get_mut(&name) {
                    *prior = Prior {
                        live: Some(loc),
                        pending: Some(loc),
                    };
                    return;
                }
            }
        }
        self.live.insert(name, loc);
        self.pending.insert(name, loc);
        self.head = self.head.max(op);
    }

    /// Opens a cut at the store's write `at`: the next [`ProjectFile::write_save_point`] names
    /// the state as of `at` (see [`Cut`]). Replaces any cut open already; records written before
    /// this call are all at or before `at`, so nothing of the old cut carries over.
    pub(crate) fn open_cut(&mut self, at: Stamp) {
        self.cut = Some(Cut {
            at,
            head: self.head,
            prior: BTreeMap::new(),
            named_end: self.end,
        });
    }

    /// Drops the open cut, if any: the next save point names the latest state, as with no cut
    /// opened. For a flush that writes no save point after its cut; the live maps already hold
    /// every record's latest location, so nothing is lost.
    pub(crate) fn drop_cut(&mut self) {
        self.cut = None;
    }

    /// The offset of the latest save point's Index, the parent of the next; `None` before the
    /// file's first.
    pub(crate) fn last_save_point(&self) -> Option<u64> {
        self.last_index()
    }

    /// Where the last record the next save point names ends: with a cut open, the end of the
    /// last record at or before it (see [`Cut`]), else the file's end. A sync that covers it
    /// covers every record the save point names; records past the cut beyond it need none.
    pub(crate) fn named_end(&self) -> u64 {
        self.cut.as_ref().map_or(self.end, |cut| cut.named_end)
    }

    /// Whether a record of `key` past the open cut has been appended: a copy of its tile as of
    /// the cut must still be appended for the cut's save point to hold it.
    pub(crate) fn is_past_cut(&self, key: &TileKey) -> bool {
        self.cut
            .as_ref()
            .is_some_and(|cut| cut.prior.contains_key(&Name::Tile(*key)))
    }

    /// Whether records can be appended: the file wasn't opened read-only.
    pub(crate) fn is_writable(&self) -> bool {
        self.access != Access::ReadOnly
    }

    /// The write the open cut is at, if one is open: a record appended with this stamp is part
    /// of the cut's save point.
    pub(crate) fn cut_at(&self) -> Option<Stamp> {
        self.cut.as_ref().map(|cut| cut.at)
    }

    /// Frames `payload` and writes it at the end.
    ///
    /// # Errors
    ///
    /// [`FileError::TooLarge`], [`FileError::Io`].
    fn append(&mut self, kind: u8, payload: &[u8]) -> Result<Loc, FileError> {
        let record = frame(kind, payload).ok_or(FileError::TooLarge)?;
        self.append_record(&record)
    }

    /// Writes an already framed record at the end, first upgrading an older file's header to
    /// the version this build writes (see [`upgrade_header`]): every tile record goes through
    /// here.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    fn append_record(&mut self, record: &[u8]) -> Result<Loc, FileError> {
        if self.version != Version::Current {
            upgrade_header(&self.file)?;
            self.version = Version::Current;
        }
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
    /// [`FileError::Damaged`] if the record fails its checksum or doesn't decode as a tile of
    /// `key`, [`FileError::Io`].
    pub fn read_tile(&self, key: &TileKey) -> Result<Option<TileBytes>, FileError> {
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
        check: impl FnOnce(&TileRecord<'_>) -> Result<(), FileError>,
    ) -> Result<Option<TileBytes>, FileError> {
        let Some(&Loc { offset, len }) = self.live.get(&Name::Tile(*key)) else {
            return Ok(None);
        };
        let raw = self.read_record(offset, len)?;
        let tile = tile_record(&raw, key).ok_or(FileError::Damaged { offset })?;
        check(&tile)?;
        let body = decompress(tile.packed, tile.raw_len)
            .ok()
            .ok_or(FileError::Damaged { offset })?;
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
    /// [`FileError::Damaged`] if the file ends first, [`FileError::Io`].
    fn read_record(&self, offset: u64, len: u64) -> Result<Vec<u8>, FileError> {
        match read_exact(&self.file, offset, len) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                Err(FileError::Damaged { offset })
            }
            other => Ok(other?),
        }
    }

    /// `key`'s latest displacement tile and the edit that wrote it; `None` if absent.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_tile`]; [`FileError::Codec`] if the payload isn't a valid
    /// displacement tile, [`FileError::NotDisplacement`] if `key` is of another channel,
    /// [`FileError::WrongSide`] if the stored side isn't the key's. The side and the stated body
    /// size are checked before decompressing.
    pub fn read_displacement(
        &self,
        key: &TileKey,
    ) -> Result<Option<(OpId, CorrectionTile)>, FileError> {
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
    ) -> Result<Option<OpId>, FileError> {
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
        self.live.get(&Name::Tile(*key)).copied()
    }

    /// Reads the displacement tile of `key` recorded at `loc` (which [`ProjectFile::locate`]
    /// gave) straight into `slot`, as a [`FileReader`] does, and returns the edit that wrote it.
    ///
    /// # Errors
    ///
    /// As [`FileReader::read_displacement_into`], but never [`FileError::Closed`]. On error
    /// `slot` holds no tile.
    pub(crate) fn read_at(
        &self,
        key: &TileKey,
        loc: Loc,
        slot: &mut CorrectionSlot,
        scratch: &mut ReadScratch,
    ) -> Result<OpId, FileError> {
        if let Err(e) = check_displacement(key, key.side()) {
            bytemuck::bytes_of_mut(slot).fill(0);
            return Err(e);
        }
        read_displacement_at(&self.file, key, loc, slot, scratch)
    }

    /// This file as of its last save point, as a [`BaseFile`] for a tile store to read
    /// corrections through, under a working file of its own; `None` if the file has no save
    /// point (a project never saved has no base).
    ///
    /// Records written since the last save point are left out: the base is the saved state.
    /// Its locations are a snapshot, so writes and saves after this call don't change what it
    /// reads; a later save needs a new base. The base reads this open file
    /// and doesn't keep it open: once this `ProjectFile` is dropped, reads through the base fail
    /// with [`FileError::Closed`].
    ///
    /// Nothing is copied or read: the base shares the maps the file keeps of its latest save
    /// point (see [`SavedState`]), so this costs the same whatever the project's size. A base
    /// of a save point this `ProjectFile` wrote names the save point before it as the one it
    /// extends (see [`BaseFile::extends`]).
    pub(crate) fn base(&self) -> Option<BaseFile> {
        let saved = self.saved.as_ref()?;
        Some(BaseFile {
            reader: self.reader(),
            locations: saved.tiles.clone(),
            bounds: saved.bounds.clone(),
            strokes: saved.strokes.clone(),
            meta: self.meta.clone(),
            save_point: SavePoint {
                project: self.meta.project(),
                offset: saved.offset,
                head: saved.head,
            },
            head: self.head,
            extends: saved.parent,
        })
    }

    /// The project file as of `point`, one of its save points, as a base to layer a recovery
    /// file on ([`ProjectFile::base`] at the latest one); `None` if the file has no such save
    /// point: none at `point`'s offset in the chain of its current state, or one whose project
    /// or head op isn't `point`'s.
    ///
    /// Every save is kept as a save point, so a recovery file on a save the project has moved
    /// past since still has its base. The base's head is the whole file's, as with
    /// [`ProjectFile::base`], so a store's op ids continue after every op in the file.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if an Index of the current chain is; [`FileError::Io`].
    pub(crate) fn base_at(&self, point: SavePoint) -> Result<Option<BaseFile>, FileError> {
        if point.project != self.meta.project() || !self.save_points()?.contains(&point.offset) {
            return Ok(None);
        }
        let len = self.file.metadata()?.len();
        let chain = load_chain(&self.file, point.offset, len)?.ok_or(FileError::Damaged {
            offset: point.offset,
        })?;
        if chain.head != point.head {
            return Ok(None);
        }
        // Every save point of the chain is at or before the latest, which holds every stroke
        // record before it: this one's are those before its Index. A base needs no batches.
        let held = self
            .saved
            .as_ref()
            .map(|latest| Held::strokes_before(latest, point.offset))
            .unwrap_or_default();
        let saved = SavedState::loaded(point.offset, chain.head, &chain.live, held);
        Ok(Some(BaseFile {
            reader: self.reader(),
            locations: saved.tiles,
            bounds: saved.bounds,
            strokes: saved.strokes,
            meta: self.meta.clone(),
            save_point: point,
            head: self.head,
            extends: None,
        }))
    }

    /// The tiles whose records differ between the save point at `earlier` and the latest one,
    /// in key order, each with its record at `earlier` (`None` where it had none there, and
    /// reads what lies under the file): what a stroke begun at `earlier` changed, and where the
    /// tiles were as it found them (see [`crate::journal`]). `earlier` is `None` for the state
    /// the file starts from, before its first save point, which holds no record.
    ///
    /// Only the tiles a stroke keeps pen-down shapes of are listed
    /// ([`crate::tiles::keeps_pen_down`]), never display tiles. Records written since the latest
    /// save point are left out: the latest save point is the state compared.
    ///
    /// `Ok(None)` if `earlier` isn't a save point of the latest one's chain, or the file has no
    /// save point at all.
    ///
    /// The cost is a walk of the chain's Indexes and the load of `earlier`'s state, as
    /// [`ProjectFile::open_at`] loads it, then one pass over both states' tiles, in memory.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if an Index of the chain is; [`FileError::Io`].
    pub(crate) fn tiles_changed_since(
        &self,
        earlier: Option<u64>,
    ) -> Result<Option<Changed>, FileError> {
        let Some(latest) = self.saved.as_ref() else {
            return Ok(None);
        };
        let before: BTreeMap<TileKey, Loc> = match earlier {
            None => BTreeMap::new(),
            Some(offset) => {
                if !self.save_points()?.contains(&offset) {
                    return Ok(None);
                }
                let len = self.file.metadata()?.len();
                let chain =
                    load_chain(&self.file, offset, len)?.ok_or(FileError::Damaged { offset })?;
                let tiles = chain.live.into_iter().filter_map(|(name, loc)| match name {
                    Name::Tile(key) => Some((key, loc)),
                    Name::Bounds(_) => None,
                });
                tiles.collect()
            }
        };
        let gone = before
            .iter()
            .filter(|&(key, loc)| latest.tiles.get(key) != Some(loc))
            .map(|(&key, &loc)| (key, Some(loc)));
        let new = latest
            .tiles
            .iter()
            .filter(|(key, _)| !before.contains_key(key))
            .map(|(&key, _)| (key, None));
        let mut changed: Changed = gone
            .chain(new)
            .filter(|(key, _)| crate::tiles::keeps_pen_down(key))
            .collect();
        changed.sort_unstable_by_key(|&(key, _)| key);
        Ok(Some(changed))
    }

    /// `key`'s latest record, framing included, after checking that it verifies and is a tile
    /// of `key`; `None` if the tile is absent. For [`ProjectFile::append_copy`] into another
    /// file.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if the record fails its checksum or isn't a tile of `key`;
    /// [`FileError::Io`].
    pub(crate) fn read_tile_record(&self, key: &TileKey) -> Result<Option<Vec<u8>>, FileError> {
        Ok(self.read_tile_record_op(key)?.map(|(raw, _)| raw))
    }

    /// `key`'s latest record, as [`ProjectFile::read_tile_record`] gives it, with the op it
    /// carries: for a roll-over, which tells the tiles a stroke changed by it.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_tile_record`].
    pub(crate) fn read_tile_record_op(
        &self,
        key: &TileKey,
    ) -> Result<Option<(Vec<u8>, OpId)>, FileError> {
        let Some(Loc { offset, len }) = self.locate(key) else {
            return Ok(None);
        };
        let raw = self.read_record(offset, len)?;
        let op = tile_record(&raw, key)
            .ok_or(FileError::Damaged { offset })?
            .op;
        Ok(Some((raw, op)))
    }

    /// Appends a copy of every tile and every bound `base` holds, each record as it is (not
    /// decoded or recompressed), checked first: the saved state of another project, carried
    /// into this one.
    ///
    /// A bounds record that doesn't read is written as unknown instead: a bound is derived
    /// data, which loading the index computes again, and must never stop a save. So is a display
    /// record (see [`crate::display`]). Then every stroke record `base` holds is copied, in
    /// order; one that doesn't read is left out (history, which never stops a save either).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`]; [`FileError::Damaged`] if a tile record of `base` fails its
    /// checksum or isn't a tile of its key; [`FileError::Io`]; [`FileError::Closed`] if
    /// `base`'s file has been dropped.
    pub(crate) fn copy_base(&mut self, base: &BaseFile) -> Result<(), FileError> {
        for (key, &loc) in base.locations.iter() {
            if crate::display::is_display(key) {
                // Derived: one that doesn't read goes across as unknown.
                match base.reader.read_record(loc) {
                    Ok(raw) => self.append_display_copy(key, &raw)?,
                    Err(FileError::Damaged { .. }) => {
                        self.write_display_many([(*key, None)], Stamp::ZERO)?;
                    }
                    Err(e) => return Err(e),
                }
                continue;
            }
            let raw = base.reader.read_record(loc)?;
            if tile_record(&raw, key).is_none() {
                return Err(FileError::Damaged { offset: loc.offset });
            }
            self.append_copy(key, &raw)?;
        }
        for (key, &loc) in base.bounds.iter() {
            match base.reader.read_record(loc) {
                Ok(raw) if bounds_record(&raw, key).is_some() => {
                    self.append_bounds_copy(key, &raw)?;
                }
                Ok(_) | Err(FileError::Damaged { .. }) => {
                    self.write_bounds(*key, None, Stamp::ZERO)?;
                }
                Err(e) => return Err(e),
            }
        }
        // History, not state: a stroke record that doesn't read is left out, never a reason to
        // stop the save.
        for loc in base.stroke_locations() {
            let Ok(raw) = read_stroke_at(&base.reader, loc) else {
                continue;
            };
            match self.append_stroke_copy(&raw) {
                Ok(()) | Err(FileError::Damaged { .. }) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Appends `key`'s bound, `None` if unknown: its latest bounds record, placed against an
    /// open cut by `stamp`, the store's write whose state it describes (see [`Cut`]).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::Io`].
    pub(crate) fn write_bounds(
        &mut self,
        key: TileKey,
        bounds: Option<&TileBounds>,
        stamp: Stamp,
    ) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let loc = self.append(KIND_BOUNDS, &bounds_payload(&key, bounds))?;
        self.note(Name::Bounds(key), None, stamp, loc);
        Ok(())
    }

    /// Appends a bounds record for each of `bounds` (`None`: unknown), in order, framed into one
    /// buffer and written with one write: each is its key's latest bounds record, placed against
    /// an open cut by `stamp`, as [`ProjectFile::write_bounds`] places one.
    ///
    /// If the write fails, none of them is recorded (the bytes written past the end, if any, are
    /// overwritten by the next append).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`].
    pub(crate) fn write_bounds_many<'a>(
        &mut self,
        bounds: impl IntoIterator<Item = (TileKey, Option<&'a TileBounds>)>,
        stamp: Stamp,
    ) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let size = |n: usize| u64::try_from(n).or(Err(FileError::TooLarge));
        let mut buffer = Vec::new();
        let mut placed = Vec::new();
        for (key, kept) in bounds {
            let record =
                frame(KIND_BOUNDS, &bounds_payload(&key, kept)).ok_or(FileError::TooLarge)?;
            placed.push((key, size(buffer.len())?, size(record.len())?));
            buffer.extend_from_slice(&record);
        }
        if buffer.is_empty() {
            return Ok(());
        }
        let start = self.end;
        let written = size(buffer.len())?;
        self.file.write_all_at(&buffer, start)?;
        self.end += written;
        for (key, at, len) in placed {
            let loc = Loc {
                offset: start + at,
                len,
            };
            self.note(Name::Bounds(key), None, stamp, loc);
        }
        Ok(())
    }

    /// Appends `raw`, a bounds record of `key` read from another file and checked there, as
    /// `key`'s latest bounds record.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`]; [`FileError::Damaged`] if `raw` isn't a whole bounds record
    /// of `key` after all; [`FileError::Io`].
    pub(crate) fn append_bounds_copy(
        &mut self,
        key: &TileKey,
        raw: &[u8],
    ) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        if bounds_record(raw, key).is_none() {
            return Err(FileError::Damaged { offset: self.end });
        }
        let loc = self.append_record(raw)?;
        self.note(Name::Bounds(*key), None, Stamp::ZERO, loc);
        Ok(())
    }

    /// `key`'s latest bounds record, framing included, if it has one that verifies and is a
    /// bounds record of `key`; `None` if it has none, or one that doesn't read. For copying it
    /// into another file.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    pub(crate) fn read_bounds_record(&self, key: &TileKey) -> Result<Option<Vec<u8>>, FileError> {
        let Some(&Loc { offset, len }) = self.live.get(&Name::Bounds(*key)) else {
            return Ok(None);
        };
        match self.read_record(offset, len) {
            Ok(raw) => Ok(bounds_record(&raw, key).is_some().then_some(raw)),
            Err(FileError::Damaged { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Where `key`'s latest bounds record is, if it has one.
    pub(crate) fn locate_bounds(&self, key: &TileKey) -> Option<Loc> {
        self.live.get(&Name::Bounds(*key)).copied()
    }

    /// Where each tile's latest bounds record is, in key order.
    pub(crate) fn bounds_locations(&self) -> impl Iterator<Item = (&TileKey, Loc)> {
        self.live
            .iter()
            .filter_map(|(name, &loc)| Some((name.bounds()?, loc)))
    }

    /// The bounds the records at `entries` hold (see [`read_bounds_at`]).
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    pub(crate) fn read_bounds(
        &self,
        entries: &[(TileKey, Loc)],
    ) -> Result<Vec<Option<TileBounds>>, FileError> {
        Ok(read_bounds_at(&self.file, entries)?)
    }

    /// Appends `raw`, a tile record of `key` read from another file and checked there, as
    /// `key`'s latest record.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`]; [`FileError::Damaged`] if `raw` isn't a whole tile record of
    /// `key` after all; [`FileError::Io`].
    pub(crate) fn append_copy(&mut self, key: &TileKey, raw: &[u8]) -> Result<(), FileError> {
        let copy = self.append_unnamed(key, raw)?;
        self.name_copy(key, copy);
        Ok(())
    }

    /// Appends `raw` as [`ProjectFile::append_copy`] does, but as nothing yet: no save point names
    /// it until [`ProjectFile::name_copy`] does. So a copy made ahead of a save that turns out
    /// not to be wanted is only bytes past the end, whatever the next save point holds.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::append_copy`].
    pub(crate) fn append_unnamed(
        &mut self,
        key: &TileKey,
        raw: &[u8],
    ) -> Result<UnnamedCopy, FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let op = tile_record(raw, key)
            .ok_or(FileError::Damaged { offset: self.end })?
            .op;
        let loc = self.append_record(raw)?;
        Ok(UnnamedCopy { op, loc })
    }

    /// Makes `copy`, appended by [`ProjectFile::append_unnamed`], `key`'s latest record.
    pub(crate) fn name_copy(&mut self, key: &TileKey, copy: UnnamedCopy) {
        self.note(Name::Tile(*key), Some(copy.op), Stamp::ZERO, copy.loc);
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
    /// The save point is an Index naming the live tiles and bounds: all of them the first time and
    /// every 16th time, otherwise only those written since the last save. It names the latest
    /// state: an open cut is dropped.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`];
    /// [`FileError::SavePointMismatch`] (a bug), with nothing written.
    pub fn save(&mut self) -> Result<u64, FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        self.cut = None;
        // Tiles are durable before the Index that names them.
        self.file.sync_data()?;
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_save) {
            return Err(io::Error::other("a save failing, for the test").into());
        }
        let appended = self.append_index()?;
        self.file.sync_data()?;
        Ok(self.commit_index(appended))
    }

    /// Appends a save point naming the tiles and bounds written so far, without syncing anything,
    /// and returns its offset: [`ProjectFile::save`] for a caller that syncs on another thread
    /// (through a [`FileSyncer`]). If a cut is open, the save point names the state at the cut
    /// (see [`Cut`]) and closes it; the records past it wait for the next save point.
    ///
    /// The caller must have synced every record written before this call (the tiles the Index
    /// names) and must sync again after it before the save point is durable. Until that second
    /// sync completes, a crash can lose the Index; opening the file then finds the previous
    /// save point and discards the tiles after it.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`];
    /// [`FileError::SavePointMismatch`] (a bug), with nothing written.
    pub(crate) fn write_save_point(&mut self) -> Result<u64, FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_save) {
            return Err(io::Error::other("a save point failing, for the test").into());
        }
        let appended = self.append_index()?;
        Ok(self.commit_index(appended))
    }

    /// Appends the next Index: full the first time and every [`FULL_EVERY`]th, otherwise a
    /// delta of the tiles written since the last one. Returns what
    /// [`ProjectFile::commit_index`] needs to make it the latest save point: with it, the new
    /// save point's maps, derived before anything is written (see [`SavedState`]).
    ///
    /// For a full Index, the derived maps must hold as many records as the Index names, or
    /// nothing is written: a check of the derivation that costs nothing in release builds
    /// beyond two lengths.
    ///
    /// # Errors
    ///
    /// [`FileError::SavePointMismatch`] if that check fails; [`FileError::TooLarge`],
    /// [`FileError::Io`].
    fn append_index(&mut self) -> Result<Appended, FileError> {
        #[cfg(test)]
        self.check_live();
        let parent = self.last_index();
        let coverage = if parent.is_none() || self.deltas + 1 >= FULL_EVERY {
            Coverage::Full
        } else {
            Coverage::Delta
        };
        let head = match &self.cut {
            Some(cut) if !cut.prior.is_empty() => cut.head,
            _ => self.head,
        };
        let delta = self.at_cut(&self.pending, Coverage::Delta);
        // What the Index names, and the records new since the last save point (all of them for
        // the first), from which the new save point's maps are derived.
        let (payload, named, (tiles, bounds)) = match (coverage, parent) {
            (Coverage::Full, None) => {
                let entries = self.at_cut(&self.live, coverage);
                (
                    index_payload(coverage, parent, head, &entries)?,
                    Some(entries.len()),
                    split_names(&entries),
                )
            }
            (Coverage::Full, Some(_)) => {
                let entries = self.at_cut(&self.live, coverage);
                (
                    index_payload(coverage, parent, head, &entries)?,
                    Some(entries.len()),
                    split_names(&delta),
                )
            }
            (Coverage::Delta, _) => (
                index_payload(coverage, parent, head, &delta)?,
                None,
                split_names(&delta),
            ),
        };
        #[cfg(test)]
        let tiles = if std::mem::take(&mut self.skew_derivation) {
            tiles.into_iter().skip(1).collect()
        } else {
            tiles
        };
        let (old_tiles, old_bounds) = self
            .saved
            .as_ref()
            .map(|saved| (saved.tiles.clone(), saved.bounds.clone()))
            .unwrap_or_default();
        let (tiles, bounds) = (old_tiles.with(tiles), old_bounds.with(bounds));
        // A full Index names every record of the new save point: a cheap check of the
        // derivation, before anything is written.
        let derived = tiles.len() + bounds.len();
        if let Some(named) = named.filter(|&named| named != derived) {
            return Err(FileError::SavePointMismatch { named, derived });
        }
        let loc = self.append(KIND_INDEX, &payload)?;
        Ok(Appended {
            coverage,
            loc,
            head,
            tiles,
            bounds,
        })
    }

    /// `latest`, the latest record of each name (all of them for a full Index, or those written
    /// since the last save point for a delta), as of the open cut, if any (see [`Cut`]): each
    /// name with a record past the cut takes its place at the cut for `coverage`'s kind of
    /// Index, or is left out if it had none.
    fn at_cut<'a>(
        &self,
        latest: &'a BTreeMap<Name, Loc>,
        coverage: Coverage,
    ) -> Cow<'a, BTreeMap<Name, Loc>> {
        let Some(cut) = self.cut.as_ref().filter(|cut| !cut.prior.is_empty()) else {
            return Cow::Borrowed(latest);
        };
        let mut at_cut = latest.clone();
        for (key, prior) in &cut.prior {
            let place = match coverage {
                Coverage::Full => prior.live,
                Coverage::Delta => prior.pending,
            };
            match place {
                Some(loc) => at_cut.insert(*key, loc),
                None => at_cut.remove(key),
            };
        }
        Cow::Owned(at_cut)
    }

    /// Makes the Index `appended` the latest save point, the parent of the next, and returns
    /// its offset, with the maps [`ProjectFile::append_index`] derived for it. What it named leaves
    /// `pending`; the records past a cut it was written for stay, for the next one, and the cut
    /// closes.
    fn commit_index(&mut self, appended: Appended) -> u64 {
        let Appended {
            coverage,
            loc,
            head,
            tiles,
            bounds,
        } = appended;
        let parent = self.saved.as_ref().map(|saved| SavePoint {
            project: self.meta.project(),
            offset: saved.offset,
            head: saved.head,
        });
        // Every stroke record appended so far is before the Index, so the save point holds it.
        let new_strokes = self.strokes.drain(..).map(|stroke| (stroke.offset, stroke));
        let strokes = self
            .saved
            .as_ref()
            .map(|saved| saved.strokes.clone())
            .unwrap_or_default()
            .with(new_strokes);
        // So is every batch.
        let new_batches = self.batches.drain(..).map(|batch| (batch.offset, batch));
        let batches = self
            .saved
            .as_ref()
            .map(|saved| saved.batches.clone())
            .unwrap_or_default()
            .with(new_batches);
        self.saved = Some(SavedState {
            offset: loc.offset,
            head,
            tiles,
            bounds,
            strokes,
            batches,
            parent,
        });
        self.deltas = match coverage {
            Coverage::Full => 0,
            Coverage::Delta => self.deltas + 1,
        };
        match self.cut.take() {
            Some(cut) => self.pending.retain(|key, _| cut.prior.contains_key(key)),
            None => self.pending.clear(),
        }
        #[cfg(test)]
        {
            self.check_saved();
            self.check_live();
        }
        loc.offset
    }

    /// The offset of the latest save point's Index, the parent of the next; `None` until the
    /// first save.
    fn last_index(&self) -> Option<u64> {
        self.saved.as_ref().map(|saved| saved.offset)
    }

    /// Whether records (tiles, bounds, strokes or dab batches) have been written since the last
    /// save point (or, before the first, at all): what the next save point would hold.
    pub(crate) fn has_unsaved_records(&self) -> bool {
        !self.pending.is_empty() || !self.strokes.is_empty() || !self.batches.is_empty()
    }

    /// A handle that syncs this file from another thread while it stays open (see
    /// [`FileSyncer`]).
    pub(crate) fn syncer(&self) -> FileSyncer {
        FileSyncer {
            file: Arc::downgrade(&self.file),
        }
    }

    /// The save points of the current state's chain, oldest first.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if an Index of the chain is, [`FileError::Io`].
    pub fn save_points(&self) -> Result<Vec<u64>, FileError> {
        let len = self.file.metadata()?.len();
        let mut points = Vec::new();
        let mut next = self.last_index();
        while let Some(offset) = next {
            let index =
                read_index(&self.file, offset, len)?.ok_or(FileError::Damaged { offset })?;
            points.push(offset);
            next = match index.parent {
                Some(parent) if parent < offset => Some(parent),
                Some(_) => return Err(FileError::Damaged { offset }),
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
        // Tiles order first: the walk stops at the first bounds record.
        self.live.keys().map_while(Name::tile)
    }

    /// Whether `key` has a live tile.
    #[must_use]
    pub fn contains(&self, key: &TileKey) -> bool {
        self.live.contains_key(&Name::Tile(*key))
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
        let meta = crate::fixture::meta();
        let record = mesh_record(&meta, &crate::fixture::mesh()).unwrap();
        u64::try_from(leading_bytes(&meta, &record).unwrap().len()).unwrap()
    }

    /// Payload bytes of the Index at `offset`.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if no record head is there, [`FileError::Io`].
    fn index_len(&self, offset: u64) -> Result<u64, FileError> {
        let len = self.file.metadata()?.len();
        Ok(read_head(&self.file, offset, len)?
            .ok_or(FileError::Damaged { offset })?
            .len
            .into())
    }

    /// The lengths a crash between two records can leave the file at `path` with: the end of
    /// its leading records, then the end of each record after them, in order.
    ///
    /// # Panics
    ///
    /// If the file can't be read or its leading records are bad.
    pub(crate) fn record_ends(path: &Path) -> Vec<u64> {
        let file = File::open(path).unwrap();
        let len = file.metadata().unwrap().len();
        let leading = read_leading(&file, len, None).unwrap();
        let heads = scan_heads(&file, leading.end, len).unwrap();
        std::iter::once(leading.end)
            .chain(
                heads
                    .into_iter()
                    .map(|(offset, head)| offset + head.total()),
            )
            .collect()
    }

    /// The tiles written since the last save point, in key order: what the next one names.
    pub(crate) fn unsaved_keys(&self) -> Vec<TileKey> {
        self.pending
            .keys()
            .filter_map(Name::tile)
            .copied()
            .collect()
    }

    /// Checks the latest save point's maps, derived in memory (see [`SavedState`]), against
    /// the same save point's chain read back from the file and rebuilt in full: the same tiles
    /// and bounds, bit for bit, and the same head op. Run after every save point a test writes.
    ///
    /// # Panics
    ///
    /// If they differ, or the chain can't be read back.
    fn check_saved(&self) {
        let Some(saved) = &self.saved else {
            return;
        };
        let len = self.file.metadata().unwrap().len();
        let chain = load_chain(&self.file, saved.offset, len)
            .unwrap()
            .expect("a save point just written loads");
        let leading = read_leading(&self.file, len, None).unwrap();
        let before = scan_heads(&self.file, leading.end, saved.offset).unwrap();
        let rebuilt = SavedState::loaded(saved.offset, chain.head, &chain.live, Held::of(&before));
        assert!(
            saved.tiles.same_entries(&rebuilt.tiles),
            "the derived save point's tiles differ from its chain's"
        );
        assert!(
            saved.bounds.same_entries(&rebuilt.bounds),
            "the derived save point's bounds differ from its chain's"
        );
        assert!(
            saved.strokes.same_entries(&rebuilt.strokes),
            "the derived save point's stroke records differ from the file's"
        );
        assert!(
            saved.batches.same_entries(&rebuilt.batches),
            "the derived save point's dab batches differ from the file's"
        );
        assert_eq!(saved.head, rebuilt.head, "the derived save point's head");
    }

    /// Checks the invariant on [`ProjectFile::live`]: it holds the latest save point's records
    /// with those written since put in over them, and nothing else. Run before and after every
    /// save point a test writes.
    ///
    /// # Panics
    ///
    /// If it doesn't.
    fn check_live(&self) {
        let saved_loc = |name: &Name| {
            let saved = self.saved.as_ref()?;
            match name {
                Name::Tile(key) => saved.tiles.get(key),
                Name::Bounds(key) => saved.bounds.get(key),
            }
            .copied()
        };
        for (name, loc) in &self.live {
            let expected = self.pending.get(name).copied().or_else(|| saved_loc(name));
            assert_eq!(
                Some(*loc),
                expected,
                "{name:?}: live, against saved and pending"
            );
        }
        let saved_len = self
            .saved
            .as_ref()
            .map_or(0, |saved| saved.tiles.len() + saved.bounds.len());
        let only_pending = self
            .pending
            .keys()
            .filter(|name| saved_loc(name).is_none())
            .count();
        assert_eq!(
            self.live.len(),
            saved_len + only_pending,
            "live holds a name neither saved nor pending, or lacks one"
        );
    }

    /// Makes the next [`ProjectFile::save`] fail after syncing its records, before its Index, or
    /// the next [`ProjectFile::write_save_point`] before it writes anything.
    pub(crate) const fn fail_next_save(&mut self) {
        self.fail_next_save = true;
    }

    /// Payload bytes of the last Index.
    fn last_index_len(&self) -> Option<u64> {
        self.index_len(self.last_index()?).ok()
    }

    /// Frames and appends an arbitrary record.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::append`].
    fn append_raw(&mut self, kind: u8, payload: &[u8]) -> Result<(), FileError> {
        self.append(kind, payload).map(drop)
    }
}

/// Checks that `key` is a displacement key whose tiles have `side` vertices per side.
///
/// # Errors
///
/// [`FileError::NotDisplacement`], [`FileError::WrongSide`].
fn check_displacement(key: &TileKey, side: u32) -> Result<(), FileError> {
    if key.channel() != Channel::DISPLACEMENT {
        Err(FileError::NotDisplacement(key.channel().raw()))
    } else if side != key.side() {
        Err(FileError::WrongSide {
            key: key.side(),
            tile: side,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod adversarial;

#[cfg(test)]
mod fuzz;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
