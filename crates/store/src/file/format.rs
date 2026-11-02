//! The project file's on-disk format.
//!
//! Its constants and limits, the header and leading records, the search for save points, and the
//! tile and Index payloads. This is the surface the fuzzers read.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::rc::Rc;
use std::sync::Arc;

use geom::{BaseMesh, TileKey};

use super::error::FileError;
use super::{Loc, Name, ProjectFile};
use crate::bytes::{KEY_LEN, Reader, put_key};
use crate::codec::{compress, decompress};
use crate::crc::Crc32;
use crate::meta::{MAX_BASE_PATH, Meta, mesh_hash};
use crate::record::{
    HEAD, Head, KIND_INDEX, KIND_MESH, KIND_META, KIND_TILE, TAIL, frame, parse_head, verify,
};
use crate::{OpId, bounds};

/// The file's first bytes.
const MAGIC: [u8; 4] = *b"BCRF";
/// The format version this build writes: 2, where a displacement tile's channel header holds a
/// scale per plane.
const VERSION: u16 = 2;
/// The format version before per-plane scales, which this build still reads: its displacement
/// tiles hold one scale, read as three equal ones.
const ONE_SCALE_VERSION: u16 = 1;
/// Offset of the header's version field, after the magic.
const VERSION_AT: u64 = MAGIC.len() as u64;

/// The format version a file's header states, of those this build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Version {
    /// Version 1: every displacement tile in it holds one scale for its three planes. A file
    /// this build appends a record to is upgraded to [`Version::Current`] first
    /// ([`upgrade_header`]), as the records it writes hold three.
    OneScale,
    /// Version 2, the one this build writes.
    Current,
}
/// File header bytes: magic, version `u16`, flags `u16`, reserved `u64`.
pub(crate) const HEADER_LEN: usize = 16;
/// Offset of the first record: the `Meta` record.
const FIRST_RECORD: u64 = 16;
/// Every this-many-th index is full; the rest are deltas.
pub(super) const FULL_EVERY: u32 = 16;
/// The longest parent chain a reader follows: writers make a full Index every
/// [`FULL_EVERY`]th, so a chain is at most one full Index and `FULL_EVERY - 1` deltas.
pub(super) const MAX_CHAIN: usize = FULL_EVERY as usize;
/// "None" for an index's parent offset and head op.
pub(super) const NONE: u64 = u64::MAX;
/// Bytes per index entry: key, offset `u64`, len `u32`.
pub(super) const ENTRY_LEN: usize = KEY_LEN + 12;
/// Bytes of an Index payload before its entries: `full: u8`, seven reserved, `parent: u64`,
/// `head: u64`, `count: u32`.
pub(super) const INDEX_FIXED: usize = 28;

/// The largest decompressed tile body, in bytes (1 MiB). A displacement tile needs 26,886.
///
/// Readers reject a record that states a larger size, so a damaged or hostile file can't make
/// them allocate more.
pub const MAX_RAW: u32 = 1 << 20;

/// The longest framed tile record a reader accepts.
///
/// That is framing, key, op, header length, the largest header (`u16::MAX`), the stated size, and
/// a zstd frame of at most [`MAX_RAW`] bytes (zstd never expands by more than 1/128 plus a small
/// constant).
pub(super) const MAX_TILE_RECORD: u64 = (HEAD + TAIL + KEY_LEN + 8 + 2 + 4) as u64
    + u16::MAX as u64
    + MAX_RAW as u64
    + MAX_RAW as u64 / 128
    + 1024;

/// The largest base mesh, in canonical bytes ([`BaseMesh::to_bytes`]): 64 MiB, room for about
/// 2.8 million vertices and their faces.
///
/// Creating a file with a larger mesh fails, and readers reject a `Mesh` record that states a
/// larger size, so a damaged or hostile file can't make them allocate more.
pub const MAX_MESH_RAW: u32 = 64 << 20;

/// The largest `Meta` record payload a reader accepts. A meta is at most 70 bytes plus its base
/// path's [`MAX_BASE_PATH`]; the rest is room for a later version's fields.
#[expect(
    clippy::cast_possible_truncation,
    reason = "MAX_BASE_PATH is 4096, far below u32::MAX"
)]
const MAX_META_PAYLOAD: u32 = 1024 + MAX_BASE_PATH as u32;

/// The largest `Mesh` record payload a reader accepts: the stated size, then a zstd frame of at
/// most [`MAX_MESH_RAW`] bytes, which zstd never expands by more than 1/128 plus a small
/// constant.
const MAX_MESH_PAYLOAD: u32 = 4 + MAX_MESH_RAW + MAX_MESH_RAW / 128 + 1024;

/// Whether an Index lists every live tile and bound or only those changed since its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Coverage {
    /// Every live tile: loading stops at this Index.
    Full,
    /// Only the tiles changed since the parent Index: loading reads the parent chain first.
    Delta,
}

/// A parsed Index payload.
#[derive(Debug)]
pub(super) struct Index {
    /// Whether the entries are every live tile and bound, or a delta on the parent.
    pub(super) coverage: Coverage,
    /// The offset of the Index saved before this one, if any.
    pub(super) parent: Option<u64>,
    /// The file's head op when this Index was saved; `None` before any edit.
    pub(super) head: Option<OpId>,
    /// Sorted by name (every tile record, then every bounds record); `None` marks a removed
    /// one.
    pub(super) entries: Vec<(Name, Option<Loc>)>,
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
pub(super) struct Chain {
    /// Each live tile's and bounds' latest record.
    pub(super) live: BTreeMap<Name, Loc>,
    /// Delta indexes after the chain's full one.
    pub(super) deltas: u32,
    /// The head op of the chain's newest Index.
    pub(super) head: Option<OpId>,
}

/// A parsed Tile payload, borrowing the record.
#[derive(Debug)]
pub(super) struct TileRecord<'a> {
    /// The tile written.
    pub(super) key: TileKey,
    /// The edit that wrote it.
    pub(super) op: OpId,
    /// The channel header, stored uncompressed.
    pub(super) header: &'a [u8],
    /// The body's length once decompressed.
    pub(super) raw_len: usize,
    /// The zstd-compressed body.
    pub(super) packed: &'a [u8],
}

/// The tile in a raw record, if the record verifies and holds a tile of `key`.
pub(super) fn tile_record<'a>(raw: &'a [u8], key: &TileKey) -> Option<TileRecord<'a>> {
    match verify(raw) {
        Some((KIND_TILE, payload)) => parse_tile_payload(payload),
        _ => None,
    }
    .filter(|tile| tile.key == *key)
}

/// Checks the file header, and says which of the versions this build reads it states.
///
/// # Errors
///
/// [`FileError::BadHeader`] if it is short or has the wrong magic or a version this build
/// doesn't read, or flags or reserved bits this version doesn't know (a later version may set
/// them to mean something), [`FileError::Io`].
pub(super) fn check_header(file: &File, len: u64) -> Result<Version, FileError> {
    if len < FIRST_RECORD {
        return Err(FileError::BadHeader);
    }
    let mut header = [0; HEADER_LEN];
    file.read_exact_at(&mut header, 0)?;
    let mut r = Reader::new(&header);
    let magic = r.take(MAGIC.len()) == Some(&MAGIC[..]);
    let version = match r.u16() {
        Some(VERSION) => Some(Version::Current),
        Some(ONE_SCALE_VERSION) => Some(Version::OneScale),
        _ => None,
    };
    match version {
        Some(version) if magic && r.u16() == Some(0) && r.u64() == Some(0) => Ok(version),
        _ => Err(FileError::BadHeader),
    }
}

/// Rewrites the header of a [`Version::OneScale`] file to say [`VERSION`].
///
/// Done before the first record this build appends to it: a reader that knows only version 1
/// then refuses the file as a whole ([`FileError::BadHeader`]) rather than finding tiles it
/// can't decode in it.
///
/// Not synced here: the version is two bytes of the file's first block, synced with the
/// records after it, before any save point that names them.
///
/// # Errors
///
/// [`FileError::Io`].
pub(super) fn upgrade_header(file: &File) -> Result<(), FileError> {
    file.write_all_at(&VERSION.to_le_bytes(), VERSION_AT)?;
    Ok(())
}

/// The framed `Mesh` record of `mesh`: its canonical bytes' length, then the bytes compressed.
///
/// # Errors
///
/// [`FileError::MeshMismatch`] if `meta` wasn't made for `mesh`; [`FileError::TooLarge`] if
/// the mesh's bytes exceed [`MAX_MESH_RAW`]; [`FileError::Io`] if compression fails.
pub(super) fn mesh_record(meta: &Meta, mesh: &BaseMesh) -> Result<Vec<u8>, FileError> {
    let raw = mesh.to_bytes();
    if mesh_hash(&raw) != meta.mesh_hash() {
        return Err(FileError::MeshMismatch);
    }
    let raw_len = u32::try_from(raw.len())
        .ok()
        .filter(|&n| n <= MAX_MESH_RAW)
        .ok_or(FileError::TooLarge)?;
    let mut payload = raw_len.to_le_bytes().to_vec();
    payload.extend_from_slice(&compress(&raw)?);
    frame(KIND_MESH, &payload).ok_or(FileError::TooLarge)
}

/// The header and leading records of a new file holding `meta`, then `mesh_record` (made by
/// [`mesh_record`] for a mesh `meta` was made for).
///
/// # Errors
///
/// [`FileError::TooLarge`] if the meta doesn't fit a record (it does).
pub(super) fn leading_bytes(meta: &Meta, mesh_record: &[u8]) -> Result<Vec<u8>, FileError> {
    let meta_record = frame(KIND_META, &meta.to_payload()).ok_or(FileError::TooLarge)?;
    let mut header = [0; HEADER_LEN];
    let (magic, rest) = header.split_at_mut(MAGIC.len());
    magic.copy_from_slice(&MAGIC);
    if let Some(version) = rest.get_mut(..2) {
        version.copy_from_slice(&VERSION.to_le_bytes());
    }
    Ok([&header[..], &meta_record, mesh_record].concat())
}

/// A file's leading records, read and checked by [`read_leading`].
pub(super) struct Leading {
    /// The file's identity.
    pub(super) meta: Meta,
    /// The base mesh.
    pub(super) mesh: Arc<BaseMesh>,
    /// The `Mesh` record as it is in the file, framing included.
    pub(super) mesh_record: Arc<[u8]>,
    /// Where the records after them start.
    pub(super) end: u64,
}

/// Reads and checks the leading `Meta` and `Mesh` records: the file's identity and base mesh,
/// and the offset where its other records start.
///
/// If the `Mesh` record is byte for byte `like`'s (a file made with
/// [`ProjectFile::create_like_mode`]) and the meta states `like`'s mesh hash, `like`'s mesh is
/// shared: the record verified, so it holds what `like`'s held, and `like` checked that against
/// the same hash. Otherwise the mesh is decompressed, hashed and parsed.
///
/// # Errors
///
/// [`FileError::MissingMeta`] if either record isn't where it belongs;
/// [`FileError::BadHeader`] if either is damaged or doesn't parse (or states a mesh larger
/// than [`MAX_MESH_RAW`]); [`FileError::MeshMismatch`] if the mesh doesn't match the meta's
/// hash; [`FileError::BadMesh`] if it isn't a valid mesh; [`FileError::Io`].
pub(super) fn read_leading(
    file: &File,
    len: u64,
    like: Option<&ProjectFile>,
) -> Result<Leading, FileError> {
    let (meta_record, mesh_at) =
        leading_record(file, FIRST_RECORD, len, KIND_META, MAX_META_PAYLOAD)?;
    let meta = verify(&meta_record)
        .and_then(|(_, payload)| Meta::from_payload(payload))
        .ok_or(FileError::BadHeader)?;
    let (mesh_record, end) = leading_record(file, mesh_at, len, KIND_MESH, MAX_MESH_PAYLOAD)?;
    if let Some(like) = like
        && like.meta.mesh_hash() == meta.mesh_hash()
        && *like.mesh_record == *mesh_record
    {
        return Ok(Leading {
            meta,
            mesh: Arc::clone(&like.mesh),
            mesh_record: Arc::clone(&like.mesh_record),
            end,
        });
    }
    let (_, payload) = verify(&mesh_record).ok_or(FileError::BadHeader)?;
    let mut r = Reader::new(payload);
    let raw_len = r
        .u32()
        .filter(|&n| n <= MAX_MESH_RAW)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(FileError::BadHeader)?;
    let raw = decompress(r.rest(), raw_len).or(Err(FileError::BadHeader))?;
    if mesh_hash(&raw) != meta.mesh_hash() {
        return Err(FileError::MeshMismatch);
    }
    let mesh = BaseMesh::from_bytes(&raw).map_err(FileError::BadMesh)?;
    Ok(Leading {
        meta,
        mesh: Arc::new(mesh),
        mesh_record: mesh_record.into(),
        end,
    })
}

/// The leading record at `offset`, framing included, which must be of `kind` with a payload of
/// at most `max` bytes and must verify, and the offset of the record after it.
///
/// # Errors
///
/// [`FileError::MissingMeta`] if the file ends before the record's head or the head is of
/// another kind; [`FileError::BadHeader`] if the head is malformed, the record states more
/// than `max` bytes or runs past the end, or fails its checksum; [`FileError::Io`].
fn leading_record(
    file: &File,
    offset: u64,
    len: u64,
    kind: u8,
    max: u32,
) -> Result<(Vec<u8>, u64), FileError> {
    if offset.saturating_add(HEAD as u64) > len {
        return Err(FileError::MissingMeta);
    }
    let mut bytes = [0; HEAD];
    file.read_exact_at(&mut bytes, offset)?;
    let head = parse_head(bytes).ok_or(FileError::BadHeader)?;
    if head.kind != kind {
        return Err(FileError::MissingMeta);
    }
    let end = offset.saturating_add(head.total());
    if head.len > max || end > len {
        return Err(FileError::BadHeader);
    }
    let raw = read_exact(file, offset, head.total())?;
    verify(&raw).ok_or(FileError::BadHeader)?;
    Ok((raw, end))
}

/// Every record head from `start` on, trusting lengths, up to the first that is invalid or runs
/// past `len`.
///
/// # Errors
///
/// The I/O error of a read.
pub(super) fn scan_heads(file: &File, start: u64, len: u64) -> io::Result<Vec<(u64, Head)>> {
    let mut heads = Vec::new();
    let mut offset = start;
    while let Some(head) = read_head(file, offset, len)? {
        heads.push((offset, head));
        offset += head.total();
    }
    Ok(heads)
}

/// Test hook: the work an open did on this thread, so adversarial tests can bound it.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Work {
    /// Indexes read by [`read_index`].
    pub(super) indexes: u64,
    /// Bytes read (and checksummed) by [`read_index`].
    pub(super) index_bytes: u64,
    /// Bytes checksummed by [`find_index`].
    pub(super) searched: u64,
}

#[cfg(test)]
thread_local! {
    /// Test hook: the [`Work`] done on this thread since [`take_work`] last ran.
    static WORK: std::cell::Cell<Work> = const {
        std::cell::Cell::new(Work { indexes: 0, index_bytes: 0, searched: 0 })
    };
}

/// Test hook: the work counted in [`WORK`] since the last call on this thread, and starts
/// counting again from zero.
#[cfg(test)]
pub(super) fn take_work() -> Work {
    WORK.with(|work| work.replace(Work::default()))
}

/// Test hook: counts one more Index of `bytes` read in [`WORK`].
#[cfg(test)]
fn count_index_read(bytes: u64) {
    WORK.with(|work| {
        let mut now = work.get();
        now.indexes = now.indexes.saturating_add(1);
        now.index_bytes = now.index_bytes.saturating_add(bytes);
        work.set(now);
    });
}

/// Test hook: counts `n` more bytes checksummed by [`find_index`] in [`WORK`].
#[cfg(test)]
fn count_checksummed(n: u64) {
    WORK.with(|work| {
        let mut now = work.get();
        now.searched = now.searched.saturating_add(n);
        work.set(now);
    });
}

/// What [`find_index`] found after the walk stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Search {
    /// A save point starts at this offset: proof of mid-file damage.
    Index(u64),
    /// No save point: the tail is torn, and is cut back.
    Nothing,
    /// The search would cost more than its budget: treated as damage, so nothing is cut.
    TooCostly,
}

/// Whether a byte offset is worth checksumming as an Index: `head`, an Index head, and
/// `payload_start`, the first [`INDEX_FIXED`] bytes after it, could begin what [`index_payload`]
/// writes.
///
/// That is, the `full` byte is 0 or 1, the seven reserved bytes are zero (every writer writes
/// zeros), and the stated length is exactly the fixed part and `count` tile entries, then either
/// nothing or a bounds section of one entry or more.
///
/// Checking this costs a few bytes; checksumming a candidate costs its whole stated length.
pub(super) fn plausible_index(head: Head, payload_start: &[u8]) -> bool {
    let (Some(&full), Some(reserved), Some(count)) = (
        payload_start.first(),
        payload_start.get(1..8),
        payload_start
            .get(24..INDEX_FIXED)
            .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok()),
    ) else {
        return false;
    };
    if full > 1 || reserved.iter().any(|&b| b != 0) {
        return false;
    }
    let entry = ENTRY_LEN as u64;
    let rest = u64::from(u32::from_le_bytes(count))
        .checked_mul(entry)
        .and_then(|tiles| tiles.checked_add(INDEX_FIXED as u64))
        .and_then(|fixed| u64::from(head.len).checked_sub(fixed));
    rest.is_some_and(|rest| rest == 0 || (rest >= 4 + entry && (rest - 4) % entry == 0))
}

/// Where a save point starts in `[from, len)` (see [`save_point`]): proof that the file is
/// damaged, not torn, at `from`.
///
/// Every byte offset is a candidate, so the checks beyond the checksum keep bytes inside another
/// record from passing for one. Only a candidate [`plausible_index`] accepts is checksummed.
///
/// The range is read in windows, so a long tail costs one pass and a window of memory. A
/// candidate longer than its window is checksummed in window-sized pieces, which costs one more
/// window; only one whose checksum matches is then read whole, its stated length, which the file
/// holds.
///
/// The bytes checksummed over all candidates are capped at four times the range's length, or
/// 64 MiB if that is more. A search that would checksum more stops with [`Search::TooCostly`],
/// which the caller treats as damage, as it does a save point found: it fails closed, so a file
/// that costs this much to search (hostile, or badly damaged) is refused, never cut back.
///
/// # Errors
///
/// The I/O error of a read.
pub(super) fn find_index(file: &File, from: u64, len: u64) -> io::Result<Search> {
    /// Candidate head positions per window, and the longest piece checksummed at once.
    const WINDOW: u64 = 1 << 20;
    /// The smallest budget of bytes checksummed.
    const MIN_BUDGET: u64 = 64 << 20;
    let budget = len.saturating_sub(from).saturating_mul(4).max(MIN_BUDGET);
    let mut spent = 0u64;
    let mut piece = Vec::new();
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
                .filter(|h| u64::from(h.len) >= INDEX_FIXED as u64)
            else {
                continue;
            };
            let lo = usize::try_from(at - start).ok();
            let mut fixed = [0; INDEX_FIXED];
            let fixed_in_buf = lo
                .and_then(|lo| buf.get(lo.checked_add(HEAD)?..lo.checked_add(HEAD + INDEX_FIXED)?));
            match fixed_in_buf {
                Some(payload_start) => fixed.copy_from_slice(payload_start),
                None => file.read_exact_at(&mut fixed, at + HEAD as u64)?,
            }
            if !plausible_index(head, &fixed) {
                continue;
            }
            if !charge(&mut spent, head.total(), budget) {
                return Ok(Search::TooCostly);
            }
            let in_buf = lo
                .zip(usize::try_from(head.total()).ok())
                .and_then(|(lo, n)| buf.get(lo..lo.checked_add(n)?));
            let good = if let Some(record) = in_buf {
                save_point(record, at).is_some()
            } else {
                if !checksums(file, at, head, &mut piece, WINDOW)? {
                    continue;
                }
                // A match is checksummed once more, as the whole record is checked.
                if !charge(&mut spent, head.total(), budget) {
                    return Ok(Search::TooCostly);
                }
                save_point(&read_exact(file, at, head.total())?, at).is_some()
            };
            if good {
                return Ok(Search::Index(at));
            }
        }
        start = start.saturating_add(WINDOW);
    }
    Ok(Search::Nothing)
}

/// Adds `n` bytes to `spent` if the total stays within `budget`, and says whether it did.
fn charge(spent: &mut u64, n: u64, budget: u64) -> bool {
    match spent.checked_add(n).filter(|&total| total <= budget) {
        Some(total) => {
            *spent = total;
            #[cfg(test)]
            count_checksummed(n);
            true
        }
        None => false,
    }
}

/// Whether the record `head` heads at `at` matches its checksum, read in pieces of at most `most`
/// bytes into `piece`: checking it costs no more memory than one piece, whatever its length.
///
/// # Errors
///
/// The I/O error of a read.
fn checksums(file: &File, at: u64, head: Head, piece: &mut Vec<u8>, most: u64) -> io::Result<bool> {
    let body = head.total() - TAIL as u64;
    let mut crc = Crc32::new();
    let mut done = 0;
    while done < body {
        let n = (body - done).min(most);
        let n_bytes =
            usize::try_from(n).map_err(|e| io::Error::new(io::ErrorKind::OutOfMemory, e))?;
        piece.resize(n_bytes, 0);
        file.read_exact_at(piece, at + done)?;
        crc.update(piece);
        done += n;
    }
    let mut stored = [0; TAIL];
    file.read_exact_at(&mut stored, at + body)?;
    Ok(crc.finish() == u32::from_le_bytes(stored))
}

/// The head at `offset`, if it is valid and its record ends by `len`.
///
/// # Errors
///
/// The I/O error of the read.
pub(super) fn read_head(file: &File, offset: u64, len: u64) -> io::Result<Option<Head>> {
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
pub(super) fn read_exact(file: &File, offset: u64, len: u64) -> io::Result<Vec<u8>> {
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
pub(super) fn read_index(file: &File, offset: u64, len: u64) -> io::Result<Option<Index>> {
    let Some(head) = read_head(file, offset, len)? else {
        return Ok(None);
    };
    if head.kind != KIND_INDEX {
        return Ok(None);
    }
    #[cfg(test)]
    count_index_read(head.total());
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

/// The live map of the Index at `offset`, or `None` if any Index of its chain is bad (see
/// [`chain_from`]).
///
/// # Errors
///
/// The I/O error of a read.
pub(super) fn load_chain(file: &File, offset: u64, len: u64) -> io::Result<Option<Chain>> {
    chain_from(
        offset,
        |at| Ok(read_save_point(file, at, len)?.map(Rc::new)),
    )
}

/// The save point at `offset` (see [`read_index`]), if its entries all lie between the header
/// and `offset`: an Index names only records written before it, and so bounds what loading them
/// reads.
///
/// # Errors
///
/// The I/O error of a read.
pub(super) fn read_save_point(file: &File, offset: u64, len: u64) -> io::Result<Option<Index>> {
    Ok(read_index(file, offset, len)?.filter(|index| index.entries_precede(offset)))
}

/// The live map of the Index `index_at` gives at `offset`, or `None` if any Index of its chain
/// is bad.
///
/// Follows parent links back to a full Index, then applies it and the deltas oldest first. A chain
/// that ends without a full Index is bad, or that is longer than any writer makes ([`MAX_CHAIN`]),
/// and so is one with an Index `index_at` doesn't give (`None`: not a save point, or one with an
/// entry outside the records before it).
///
/// # Errors
///
/// `index_at`'s I/O error.
pub(super) fn chain_from(
    offset: u64,
    mut index_at: impl FnMut(u64) -> io::Result<Option<Rc<Index>>>,
) -> io::Result<Option<Chain>> {
    let mut chain = Vec::new();
    let mut at = offset;
    loop {
        let Some(index) = index_at(at)? else {
            return Ok(None);
        };
        let (coverage, parent) = (index.coverage, index.parent);
        chain.push(index);
        if chain.len() > MAX_CHAIN {
            return Ok(None);
        }
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
    for index in chain.iter().rev() {
        for &(key, loc) in &index.entries {
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
/// [`FileError::TooLarge`] if the header exceeds its length field, the body [`MAX_RAW`], or the
/// framed record [`MAX_TILE_RECORD`].
pub(super) fn tile_payload(
    key: TileKey,
    op: OpId,
    header: &[u8],
    raw_len: usize,
    packed: &[u8],
) -> Result<Vec<u8>, FileError> {
    let header_len = u16::try_from(header.len()).or(Err(FileError::TooLarge))?;
    let raw_len = u32::try_from(raw_len)
        .ok()
        .filter(|&n| n <= MAX_RAW)
        .ok_or(FileError::TooLarge)?;
    let payload_len = KEY_LEN + 14 + header.len() + packed.len();
    let framed = u64::try_from(HEAD + TAIL + payload_len).or(Err(FileError::TooLarge))?;
    if framed > MAX_TILE_RECORD {
        return Err(FileError::TooLarge);
    }
    let mut out = Vec::with_capacity(payload_len);
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
pub(super) fn parse_tile_payload(payload: &[u8]) -> Option<TileRecord<'_>> {
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

/// An Index payload, naming tile records and then bounds records.
///
/// `full: u8 | reserved [u8; 7] | parent: u64 | head: u64 | count: u32 | count × (key | offset:
/// u64 | len: u32)`, the tile entries in key order; then, if any bounds record is named,
/// `bounds: u32 | bounds × (key | offset: u64 | len: u32)`, the bounds entries in key order. An
/// Index naming no bounds ends after its tile entries, as one written before bounds were kept
/// does.
///
/// # Errors
///
/// [`FileError::TooLarge`] if either section has more than `u32::MAX` entries or a record is
/// longer than `u32::MAX` bytes.
pub(super) fn index_payload(
    coverage: Coverage,
    parent: Option<u64>,
    head: Option<OpId>,
    entries: &BTreeMap<Name, Loc>,
) -> Result<Vec<u8>, FileError> {
    let tiles = entries
        .keys()
        .take_while(|name| name.tile().is_some())
        .count();
    let bounds = entries.len() - tiles;
    let mut out = Vec::with_capacity(32 + entries.len() * ENTRY_LEN);
    out.push(match coverage {
        Coverage::Full => 1,
        Coverage::Delta => 0,
    });
    out.extend_from_slice(&[0; 7]);
    out.extend_from_slice(&parent.unwrap_or(NONE).to_le_bytes());
    out.extend_from_slice(&head.map_or(NONE, OpId::to_bits).to_le_bytes());
    let count = |n: usize| u32::try_from(n).or(Err(FileError::TooLarge));
    out.extend_from_slice(&count(tiles)?.to_le_bytes());
    for (n, (name, loc)) in entries.iter().enumerate() {
        if n == tiles {
            out.extend_from_slice(&count(bounds)?.to_le_bytes());
        }
        let (Name::Tile(key) | Name::Bounds(key)) = name;
        put_key(&mut out, key);
        out.extend_from_slice(&loc.offset.to_le_bytes());
        let len = u32::try_from(loc.len).or(Err(FileError::TooLarge))?;
        out.extend_from_slice(&len.to_le_bytes());
    }
    Ok(out)
}

/// Parses `count` Index entries of `r`, each named by `name` of its key; `None` if a record's
/// length isn't one that `fits` its kind (a zero length, a removed entry, always is).
fn parse_entries(
    r: &mut Reader<'_>,
    count: usize,
    name: fn(TileKey) -> Name,
    fits: fn(u64) -> bool,
) -> Option<Vec<(Name, Option<Loc>)>> {
    (0..count)
        .map(|_| {
            let key = r.key()?;
            let offset = r.u64()?;
            let len = r.u32().filter(|&len| len == 0 || fits(len.into()))?;
            Some((
                name(key),
                (len != 0).then_some(Loc {
                    offset,
                    len: len.into(),
                }),
            ))
        })
        .collect()
}

/// Parses [`index_payload`]'s output.
///
/// `None` on any shortfall, a reserved byte that isn't zero, an invalid key, a bounds section that
/// is empty or doesn't fill the rest exactly, a section's keys out of order, or an entry whose
/// record length no record of its kind has (a tile record over [`MAX_TILE_RECORD`], a bounds record
/// of neither length [`bounds::is_record_len`] accepts).
pub(super) fn parse_index_payload(payload: &[u8]) -> Option<Index> {
    let mut r = Reader::new(payload);
    let coverage = match r.u8()? {
        0 => Coverage::Delta,
        1 => Coverage::Full,
        _ => return None,
    };
    // Every writer writes zeros here; a reader that skipped them would let a later version's
    // meaning for them pass unnoticed.
    r.take(7)
        .filter(|reserved| reserved.iter().all(|&b| b == 0))?;
    let parent = Some(r.u64()?).filter(|&p| p != NONE);
    let head = match r.u64()? {
        NONE => None,
        bits => Some(OpId::from_bits(bits)?),
    };
    let tiles = usize::try_from(r.u32()?).ok()?;
    let mut entries = parse_entries(&mut r, tiles, Name::Tile, |len| len <= MAX_TILE_RECORD)?;
    if !r.rest().is_empty() {
        let bounds = usize::try_from(r.u32()?).ok()?;
        if bounds == 0 || r.rest().len() != bounds.checked_mul(ENTRY_LEN)? {
            return None;
        }
        entries.extend(parse_entries(
            &mut r,
            bounds,
            Name::Bounds,
            bounds::is_record_len,
        )?);
    }
    // In order, so loading a chain applies each name's entries in the order they were made.
    let ordered = entries.windows(2).all(|pair| match pair {
        [(a, _), (b, _)] => a < b,
        _ => true,
    });
    ordered.then_some(Index {
        coverage,
        parent,
        head,
        entries,
    })
}
