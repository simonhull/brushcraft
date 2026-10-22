//! The project file: a header and append-only records (see `record`), with save points that form
//! a chain.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use geom::{CorrectionTile, TileKey};
use thiserror::Error;

use crate::OpId;
use crate::bytes::{KEY_LEN, Reader, put_key};
use crate::codec::{CodecError, compress, decode_displacement, decompress, encode_displacement};
use crate::record::{HEAD, Head, KIND_INDEX, KIND_TILE, frame, parse_head, verify};

/// The file's first bytes.
const MAGIC: [u8; 4] = *b"BCRF";
/// The format version this build reads and writes.
const VERSION: u16 = 1;
/// File header bytes: magic, version `u16`, flags `u16`, reserved `u64`.
pub(crate) const HEADER_LEN: usize = 16;
/// Offset of the first record.
const FIRST_RECORD: u64 = 16;
/// Every this-many-th index is full; the rest are deltas.
const FULL_EVERY: u32 = 16;
/// "None" for an index's parent offset and head op.
const NONE: u64 = u64::MAX;
/// Bytes per index entry: key, offset `u64`, len `u32`.
const ENTRY_LEN: usize = KEY_LEN + 12;

/// Why a project file operation failed.
#[derive(Debug, Error)]
pub enum StoreError {
    /// An I/O call failed.
    #[error("project file I/O failed")]
    Io(#[from] io::Error),
    /// Not a BrushCraft project, or a version this build can't read.
    #[error("not a BrushCraft project file (bad magic or version)")]
    BadHeader,
    /// A record's checksum or structure is wrong.
    #[error("the record at byte {offset} is damaged")]
    Damaged {
        /// Where the record starts.
        offset: u64,
    },
    /// `open_at` was given an offset that isn't a good Index.
    #[error("byte {0} is not a save point")]
    NotASavePoint(u64),
    /// The file was opened at an earlier save point and can't be written.
    #[error("opened read-only at an earlier save point")]
    ReadOnly,
    /// A payload doesn't fit the format's length fields.
    #[error("record payload too large")]
    TooLarge,
    /// A tile payload didn't encode or decode.
    #[error("tile payload didn't encode or decode")]
    Codec(#[from] CodecError),
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

/// Where a record lives: its first byte and its whole length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Loc {
    offset: u64,
    len: u64,
}

/// Whether a [`ProjectFile`] accepts writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Opened at the latest state: writes append.
    ReadWrite,
    /// Opened at an earlier save point.
    ReadOnly,
}

/// Whether an Index lists every live tile or only those changed since its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    Full,
    Delta,
}

/// A parsed Index payload.
#[derive(Debug)]
struct Index {
    coverage: Coverage,
    parent: Option<u64>,
    head: Option<OpId>,
    /// Sorted by key; `None` marks a removed tile.
    entries: Vec<(TileKey, Option<Loc>)>,
}

/// The live map an Index and its parent chain describe.
#[derive(Debug)]
struct Chain {
    live: BTreeMap<TileKey, Loc>,
    /// Delta indexes after the chain's full one.
    deltas: u32,
    head: Option<OpId>,
}

/// A parsed Tile payload, borrowing the record.
#[derive(Debug)]
struct TileRecord<'a> {
    key: TileKey,
    op: OpId,
    header: &'a [u8],
    raw_len: usize,
    packed: &'a [u8],
}

/// An open project file.
#[derive(Debug)]
pub struct ProjectFile {
    file: File,
    /// Where the next record goes.
    end: u64,
    live: BTreeMap<TileKey, Loc>,
    /// Entries written since the last Index.
    pending: BTreeMap<TileKey, Loc>,
    last_index: Option<u64>,
    /// Delta indexes since the last full one.
    deltas: u32,
    head: Option<OpId>,
    access: Access,
}

impl ProjectFile {
    /// Creates a new, empty project at `path`.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`], including when `path` already exists.
    pub fn create(path: &Path) -> Result<Self, StoreError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut header = [0; HEADER_LEN];
        let (magic, rest) = header.split_at_mut(MAGIC.len());
        magic.copy_from_slice(&MAGIC);
        if let Some(version) = rest.get_mut(..2) {
            version.copy_from_slice(&VERSION.to_le_bytes());
        }
        file.write_all_at(&header, 0)?;
        file.sync_all()?;
        Ok(Self {
            file,
            end: FIRST_RECORD,
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
    /// # Errors
    ///
    /// [`StoreError::BadHeader`], [`StoreError::Io`].
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        check_header(&file, len)?;
        let heads = scan_heads(&file, len)?;
        let mut this = Self {
            file,
            end: FIRST_RECORD,
            live: BTreeMap::new(),
            pending: BTreeMap::new(),
            last_index: None,
            deltas: 0,
            head: None,
            access: Access::ReadWrite,
        };
        let replay_from = this.load_base(&heads, len)?;
        this.replay(heads.get(replay_from..).unwrap_or_default())?;
        if this.end < len {
            this.file.set_len(this.end)?;
            this.file.sync_all()?;
        }
        Ok(this)
    }

    /// Opens the project at `path` read-only, at the save point (an Index) at byte `index_offset`.
    ///
    /// # Errors
    ///
    /// [`StoreError::NotASavePoint`] if no good Index starts there, [`StoreError::Damaged`] if
    /// its parent chain is, [`StoreError::BadHeader`], [`StoreError::Io`].
    pub fn open_at(path: &Path, index_offset: u64) -> Result<Self, StoreError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        check_header(&file, len)?;
        if index_offset < FIRST_RECORD || read_index(&file, index_offset, len)?.is_none() {
            return Err(StoreError::NotASavePoint(index_offset));
        }
        let chain = load_chain(&file, index_offset, len)?.ok_or(StoreError::Damaged {
            offset: index_offset,
        })?;
        Ok(Self {
            file,
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

    /// Replays the records after the base Index while they verify, moving `end` past each one.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`].
    fn replay(&mut self, heads: &[(u64, Head)]) -> Result<(), StoreError> {
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
                    let Some(tile) = parse_tile_payload(payload) else {
                        break;
                    };
                    self.live.insert(tile.key, loc);
                    self.pending.insert(tile.key, loc);
                    self.head = self.head.max(Some(tile.op));
                }
                // Any Index this late failed to load as the base.
                KIND_INDEX => break,
                // A kind from a newer version: skipped.
                _ => {}
            }
            self.end = offset + head.total();
        }
        Ok(())
    }

    /// Appends a tile record: `key`'s new content, written by `op`. The body is compressed.
    /// Durable once [`ProjectFile::save`] returns.
    ///
    /// # Errors
    ///
    /// [`StoreError::ReadOnly`], [`StoreError::TooLarge`], [`StoreError::Io`].
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
    /// As [`ProjectFile::write_tile`], and [`StoreError::Codec`] if the tile is too wide.
    pub fn write_displacement(
        &mut self,
        key: TileKey,
        op: OpId,
        tile: &CorrectionTile,
    ) -> Result<(), StoreError> {
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
        self.file.write_all_at(&record, self.end)?;
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
        let Some(&Loc { offset, len }) = self.live.get(key) else {
            return Ok(None);
        };
        let damaged = StoreError::Damaged { offset };
        let raw = match read_exact(&self.file, offset, len) {
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Err(damaged),
            other => other?,
        };
        let tile = match verify(&raw) {
            Some((KIND_TILE, payload)) => parse_tile_payload(payload),
            _ => None,
        }
        .filter(|tile| tile.key == *key)
        .ok_or(damaged)?;
        let body = decompress(tile.packed, tile.raw_len)
            .ok()
            .ok_or(StoreError::Damaged { offset })?;
        Ok(Some(TileBytes {
            op: tile.op,
            header: tile.header.to_vec(),
            body,
        }))
    }

    /// `key`'s latest displacement tile and the edit that wrote it; `None` if absent.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_tile`], and [`StoreError::Codec`] if the payload isn't a valid
    /// displacement tile.
    pub fn read_displacement(
        &self,
        key: &TileKey,
    ) -> Result<Option<(OpId, CorrectionTile)>, StoreError> {
        let Some(tile) = self.read_tile(key)? else {
            return Ok(None);
        };
        Ok(Some((
            tile.op,
            decode_displacement(&tile.header, &tile.body)?,
        )))
    }

    /// Makes every write so far durable and records a save point: an Index naming the live tiles
    /// (in full every [`FULL_EVERY`]th time and the first, else only those written since the last
    /// save). Returns the save point's offset, for [`ProjectFile::open_at`].
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

    /// The newest edit in the current state.
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

    /// The file's length in bytes (for a read-only file, as opened).
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
}

/// Checks the file header.
///
/// # Errors
///
/// [`StoreError::BadHeader`] if it is short or has the wrong magic or version,
/// [`StoreError::Io`].
fn check_header(file: &File, len: u64) -> Result<(), StoreError> {
    if len < FIRST_RECORD {
        return Err(StoreError::BadHeader);
    }
    let mut header = [0; HEADER_LEN];
    file.read_exact_at(&mut header, 0)?;
    let mut r = Reader::new(&header);
    if r.take(MAGIC.len()) == Some(&MAGIC[..]) && r.u16() == Some(VERSION) {
        Ok(())
    } else {
        Err(StoreError::BadHeader)
    }
}

/// Every record head from the first record on, trusting lengths, up to the first that is invalid
/// or runs past `len`.
///
/// # Errors
///
/// The I/O error of a read.
fn scan_heads(file: &File, len: u64) -> io::Result<Vec<(u64, Head)>> {
    let mut heads = Vec::new();
    let mut offset = FIRST_RECORD;
    while let Some(head) = read_head(file, offset, len)? {
        heads.push((offset, head));
        offset += head.total();
    }
    Ok(heads)
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

/// The Index at `offset`, if a whole record that verifies and parses as an Index starts there.
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
    let raw = read_exact(file, offset, head.total())?;
    Ok(match verify(&raw) {
        Some((KIND_INDEX, payload)) => parse_index_payload(payload),
        _ => None,
    })
}

/// The live map of the Index at `offset`, or `None` if any Index of its chain is bad.
///
/// Follows parent links back to a full Index, then applies it and the deltas oldest first. A chain
/// that ends without a full Index is bad.
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
/// [`StoreError::TooLarge`] if the header or body exceeds its length field.
fn tile_payload(
    key: TileKey,
    op: OpId,
    header: &[u8],
    raw_len: usize,
    packed: &[u8],
) -> Result<Vec<u8>, StoreError> {
    let header_len = u16::try_from(header.len()).or(Err(StoreError::TooLarge))?;
    let raw_len = u32::try_from(raw_len).or(Err(StoreError::TooLarge))?;
    let mut out = Vec::with_capacity(KEY_LEN + 14 + header.len() + packed.len());
    put_key(&mut out, &key);
    out.extend_from_slice(&op.to_bits().to_le_bytes());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(&raw_len.to_le_bytes());
    out.extend_from_slice(packed);
    Ok(out)
}

/// Parses [`tile_payload`]'s output; `None` on any shortfall or an invalid key.
fn parse_tile_payload(payload: &[u8]) -> Option<TileRecord<'_>> {
    let mut r = Reader::new(payload);
    let key = r.key()?;
    let op = OpId::from_bits(r.u64()?);
    let header_len = r.u16()?;
    let header = r.take(usize::from(header_len))?;
    let raw_len = usize::try_from(r.u32()?).ok()?;
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
    let head = Some(r.u64()?).filter(|&h| h != NONE).map(OpId::from_bits);
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
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level, Vec3};

    use super::*;
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
        let mut f = ProjectFile::create(&path).unwrap();
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
        let mut f = ProjectFile::create(&path).unwrap();
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
        let mut f = ProjectFile::create(&path).unwrap();
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
        let mut f = ProjectFile::create(&path).unwrap();
        f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
        f.write_displacement(key(2, 0), op(2), &tile(0.5)).unwrap();
        f.save().unwrap();
        drop(f);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_LEN + 40] ^= 0xFF; // inside the first tile's payload
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
}
