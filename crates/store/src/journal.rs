//! The dab journal: the dabs an open stroke has placed and not yet landed, kept in the recovery
//! file so that a crash, or a close that leaves the landing to finish unseen, loses none of them.
//!
//! While a stroke's op is open and some of its dabs haven't landed, each recovery flush writes
//! one [`DabBatch`]: the dabs placed since the last batch (none, if only the landed count moved)
//! and how many of the stroke's dabs had landed at the flush's cut. The batches of an op, in file
//! order, hold every dab the stroke placed, so restoring needs only them and the tiles as the
//! stroke found them, never the stroke engine. Each batch names where those tiles are: the
//! stroke's pen-down save point ([`PenDown`]), which a restore reads them through, never guessing
//! it from the order of ops. A dab (`stroke::Dab`) is [`DAB_BYTES`] bytes of
//! plain data, which the store keeps as opaque rows and never reads, as it keeps a stroke's
//! settings.
//!
//! The project file never holds a journal: a save lands every dab first, and compaction keeps the
//! latest state only.
//!
//! # Payload
//!
//! A dab-batch record (kind 7) holds, all numbers little-endian:
//!
//! - the record's version, `1`, as a `u8`, then three reserved zero bytes;
//! - the dab format, a `u32`: the version of the rows' layout, as the app set it
//!   ([`DabBatch::dab_format`]), which the store never reads;
//! - the op, as `OpId::to_bits` packs it, a `u64`;
//! - the stroke's index of the batch's first dab, a `u32`: dabs counted in placement order,
//!   symmetry copies included;
//! - how many of the stroke's dabs had landed at the flush's cut, a `u32`;
//! - the dabs' count, a `u32`, at most [`MAX_JOURNAL_DABS`];
//! - the stroke's pen-down save point ([`PenDown`]): a `u8` tag, `0` for [`PenDown::Base`], `1` for
//!   [`PenDown::At`] and `2` for [`PenDown::Unknown`], then a `u64`, the save point's offset for
//!   `At` and `0` otherwise;
//! - a zstd frame of the dabs transposed into [`DAB_BYTES`] byte planes (byte `j` of dab `i` at `j
//!   × count + i`): `DAB_BYTES × count` bytes once decompressed, exactly.
//!
//! A reader refuses any other version, a reserved byte set, a reserved site's op, a count past
//! its limit, a pen-down tag it doesn't know or an offset beside a tag that takes none, anything
//! but exactly one zstd frame after the fixed fields (a skippable frame, or
//! bytes after the frame), a frame that doesn't decompress to the stated size, a batch whose dabs
//! run past the last `u32` index, and a landed count past the dabs placed: the first index plus
//! the count.
//!
//! # The dab format
//!
//! The rows are `stroke::Dab`'s bytes as the app laid them out, native-endian and unversioned,
//! and a stroke's record (which names the engine's version) is written only at its end: a crashed
//! stroke has none. So each batch carries the version of the rows' layout the app set
//! (`crate::Project::set_dab_format`), as it was when the stroke began, and a restore reports it,
//! for the app to refuse rows of a layout it doesn't read. The store only carries it. **0 means
//! unset**, the app never said: a restore must refuse rows of format 0.
//!
//! # Where batches live
//!
//! No Index names a batch: a save point holds every batch before it in the file, as it holds
//! stroke records (see `crate::file::journal_io`), so the save point that names a flush's tiles
//! holds its batch, and the batch's landed count describes exactly that cut. Opening a file only
//! steps over their heads.
//!
//! # The pen-down save point
//!
//! A stroke's dabs are placed on its pen-down surface, the tiles as the stroke found them, so a
//! restore needs that surface besides the dabs. Each batch names the save point that holds it
//! ([`PenDown`]): the store makes sure one does, as a stroke's first landing waits until a
//! flush's cut holds every earlier op's edits and none of its own (see `crate::recovery`, and
//! `crate::Project::may_land`). The save point is
//! named, never found from the order of ops, because the save point just before a stroke's op may
//! have been cut midway through the stroke before it. A batch whose pen-down is
//! [`PenDown::Unknown`] can't be restored: the tiles restore alone.

use thiserror::Error;

use crate::bytes::{Reader, transpose, untranspose};
use crate::codec::{compress, decompress_frame};
use crate::{FileError, OpId};

/// The bytes of one dab as the journal keeps it: `stroke::Dab`'s, which the store never reads.
pub const DAB_BYTES: usize = 84;

/// The most dabs one batch holds; a flush with more writes several batches.
pub const MAX_JOURNAL_DABS: usize = 65_536;

/// The record's version this build writes and reads.
const VERSION: u8 = 1;

/// Payload bytes before the compressed frame: version and reserved bytes, dab format, op, first
/// index, landed count, dab count, and the pen-down's tag and offset.
const FIXED: usize = 4 + 4 + 8 + 4 + 4 + 4 + 1 + 8;

/// The longest payload a writer makes: the fixed fields, then a zstd frame of the most dabs,
/// which zstd never expands by more than 1/128 plus a small constant.
pub(crate) const MAX_JOURNAL_PAYLOAD: usize = {
    let raw = DAB_BYTES * MAX_JOURNAL_DABS;
    FIXED + raw + raw / 128 + 1024
};

/// Where a journalled stroke's pen-down surface is: the save point of the recovery file whose
/// state is every tile as the stroke found them (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PenDown {
    /// The stroke began on a recovery file with no save point of its own yet: its pen-down
    /// surface is the state the file starts from, the project file's save point its meta names,
    /// or, for a project never saved, the base mesh with no sculpting.
    Base,
    /// The save point (Index) at this offset in the recovery file holds it.
    At(u64),
    /// No save point holds it: the flush that was to was abandoned, or the stroke began before
    /// the recovery file's own start (a save, or a roll-over, in the middle of the stroke). The
    /// stroke's journal can't be restored.
    Unknown,
}

impl PenDown {
    /// The payload's tag of [`PenDown::At`].
    const AT: u8 = 1;
    /// The payload's tag of [`PenDown::Base`].
    const BASE: u8 = 0;
    /// The payload's tag of [`PenDown::Unknown`].
    const UNKNOWN: u8 = 2;

    /// The payload's tag and offset (see the module notes).
    const fn to_parts(self) -> (u8, u64) {
        match self {
            Self::Base => (Self::BASE, 0),
            Self::At(offset) => (Self::AT, offset),
            Self::Unknown => (Self::UNKNOWN, 0),
        }
    }

    /// The pen-down a payload's `tag` and `offset` state; `None` for a tag this build doesn't
    /// know, or an offset beside a tag that takes none.
    const fn from_parts(tag: u8, offset: u64) -> Option<Self> {
        match (tag, offset) {
            (Self::BASE, 0) => Some(Self::Base),
            (Self::AT, offset) => Some(Self::At(offset)),
            (Self::UNKNOWN, 0) => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// Why a [`DabBatch`] can't be made, or a payload isn't one this build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JournalError {
    /// More than [`MAX_JOURNAL_DABS`] dabs.
    #[error("{0} dabs are more than a batch holds")]
    TooManyDabs(usize),
    /// The batch's dabs run past the last index a `u32` counts.
    #[error("the batch's dabs run past the last index of a stroke")]
    PastLastIndex,
    /// More of the stroke's dabs landed than it had placed by the batch's last dab.
    #[error("{landed} dabs landed of the {placed} placed")]
    LandedPastPlaced {
        /// The landed count.
        landed: u32,
        /// The dabs placed: the batch's first index plus its count.
        placed: u32,
    },
    /// The payload ends before its fixed fields do.
    #[error("the dab batch is cut short")]
    Short,
    /// A version this build doesn't read.
    #[error("dab batch version {0} isn't one this build reads")]
    Version(u8),
    /// A reserved byte is set.
    #[error("a reserved byte of the dab batch is set")]
    Reserved,
    /// The op is of the reserved site, which has none.
    #[error("the dab batch's op is of the reserved site")]
    ReservedSite,
    /// The pen-down's tag isn't one this build knows, or an offset stands beside a tag that
    /// takes none.
    #[error("the dab batch's pen-down save point isn't one this build reads")]
    PenDown,
    /// The frame doesn't decompress to exactly the stated dabs.
    #[error("the dab batch's frame doesn't hold its dabs")]
    Frame,
}

/// One flush's part of an open stroke's journal (record kind 7).
///
/// It holds the dabs the stroke placed since the batch before, from index [`DabBatch::first`],
/// how many of its dabs had landed at the flush's cut, and where its pen-down surface is
/// ([`DabBatch::pen_down`]).
///
/// Made only by the store, which holds it to what a reader of the record takes, so every batch
/// the store writes reads back as the same batch; the app reads them back, for tools and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DabBatch {
    /// The stroke's op.
    op: OpId,
    /// The index in the stroke of `dabs[0]`: dabs counted in placement order, symmetry copies
    /// included.
    first: u32,
    /// How many of the stroke's dabs had landed at the flush's cut; at most `first + dabs.len()`.
    landed: u32,
    /// The save point holding the stroke's pen-down surface.
    pen_down: PenDown,
    /// The version of the rows' layout, as the app set it (see the module notes).
    dab_format: u32,
    /// Opaque rows of [`DAB_BYTES`], at most [`MAX_JOURNAL_DABS`], ending at or before the last
    /// `u32` index.
    dabs: Vec<[u8; DAB_BYTES]>,
}

impl DabBatch {
    /// The batch of the stroke made as `op` holding `dabs` (each a `stroke::Dab`'s bytes) from
    /// its index `first`, with `landed` of its dabs landed, its pen-down surface at `pen_down`, in
    /// dab format 0 ([`DabBatch::with_dab_format`] gives another).
    ///
    /// # Errors
    ///
    /// [`JournalError::TooManyDabs`] past [`MAX_JOURNAL_DABS`]; [`JournalError::PastLastIndex`]
    /// if `first` plus the count is past `u32::MAX`; [`JournalError::LandedPastPlaced`] if
    /// `landed` is past that.
    pub(crate) fn new(
        op: OpId,
        first: u32,
        landed: u32,
        pen_down: PenDown,
        dabs: Vec<[u8; DAB_BYTES]>,
    ) -> Result<Self, JournalError> {
        if dabs.len() > MAX_JOURNAL_DABS {
            return Err(JournalError::TooManyDabs(dabs.len()));
        }
        let placed = u32::try_from(dabs.len())
            .ok()
            .and_then(|count| first.checked_add(count))
            .ok_or(JournalError::PastLastIndex)?;
        if landed > placed {
            return Err(JournalError::LandedPastPlaced { landed, placed });
        }
        Ok(Self {
            op,
            first,
            landed,
            pen_down,
            dab_format: 0,
            dabs,
        })
    }

    /// The batch with its rows in dab format `dab_format` (see the module notes).
    #[must_use]
    pub(crate) const fn with_dab_format(mut self, dab_format: u32) -> Self {
        self.dab_format = dab_format;
        self
    }

    /// The version of the rows' layout the app set, which the store never reads (see the module
    /// notes).
    #[must_use]
    pub const fn dab_format(&self) -> u32 {
        self.dab_format
    }

    /// The stroke's op.
    #[must_use]
    pub const fn op(&self) -> OpId {
        self.op
    }

    /// The stroke's index of the batch's first dab.
    #[must_use]
    pub const fn first(&self) -> u32 {
        self.first
    }

    /// How many of the stroke's dabs had landed at the flush's cut.
    #[must_use]
    pub const fn landed(&self) -> u32 {
        self.landed
    }

    /// The save point holding the stroke's pen-down surface (see the module notes).
    #[must_use]
    pub const fn pen_down(&self) -> PenDown {
        self.pen_down
    }

    /// How many dabs the stroke had placed by the batch's last: its first index plus its count,
    /// the next batch's first index.
    #[must_use]
    pub fn placed(&self) -> u32 {
        // `new` holds the sum within a u32.
        self.first.saturating_add(self.count())
    }

    /// The dabs, in placement order, each a `stroke::Dab`'s bytes.
    #[must_use]
    pub fn dabs(&self) -> &[[u8; DAB_BYTES]] {
        &self.dabs
    }

    /// How many dabs the batch holds, as the payload states it.
    fn count(&self) -> u32 {
        // `new` holds it within MAX_JOURNAL_DABS, far below u32::MAX.
        u32::try_from(self.dabs.len()).unwrap_or(u32::MAX)
    }

    /// The batch's payload (see the module notes).
    ///
    /// # Errors
    ///
    /// [`FileError::Io`] if compressing fails.
    pub(crate) fn encode(&self) -> Result<Vec<u8>, FileError> {
        let packed = compress(&transpose(self.dabs.iter().copied()))?;
        let mut out = Vec::with_capacity(FIXED + packed.len());
        out.extend_from_slice(&[VERSION, 0, 0, 0]);
        out.extend_from_slice(&self.dab_format.to_le_bytes());
        out.extend_from_slice(&self.op.to_bits().to_le_bytes());
        out.extend_from_slice(&self.first.to_le_bytes());
        out.extend_from_slice(&self.landed.to_le_bytes());
        out.extend_from_slice(&self.count().to_le_bytes());
        let (tag, offset) = self.pen_down.to_parts();
        out.push(tag);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&packed);
        Ok(out)
    }

    /// The batch `payload` holds.
    ///
    /// # Errors
    ///
    /// The [`JournalError`] of the first thing refused (see the module notes):
    /// [`JournalError::Short`], [`JournalError::Version`], [`JournalError::Reserved`],
    /// [`JournalError::ReservedSite`], [`JournalError::TooManyDabs`], [`JournalError::PenDown`],
    /// [`JournalError::Frame`], [`JournalError::PastLastIndex`] or
    /// [`JournalError::LandedPastPlaced`].
    pub(crate) fn decode(payload: &[u8]) -> Result<Self, JournalError> {
        let head = Self::decode_head(payload)?;
        let count = usize::try_from(head.count).unwrap_or(usize::MAX);
        let frame = payload.get(FIXED..).ok_or(JournalError::Short)?;
        let planes = decompress_frame(frame, count * DAB_BYTES).or(Err(JournalError::Frame))?;
        let dabs: Vec<[u8; DAB_BYTES]> = untranspose(&planes, count)
            .ok_or(JournalError::Frame)?
            .collect();
        Ok(
            Self::new(head.op, head.first, head.landed, head.pen_down, dabs)?
                .with_dab_format(head.dab_format),
        )
    }

    /// The fixed fields at the start of `payload`, a batch's payload, checked as
    /// [`DabBatch::decode`] checks them, without decompressing the dabs.
    ///
    /// # Errors
    ///
    /// As [`DabBatch::decode`], but for [`JournalError::Frame`]: the dabs aren't read.
    pub(crate) fn decode_head(payload: &[u8]) -> Result<BatchHead, JournalError> {
        let fixed = payload.get(..FIXED).ok_or(JournalError::Short)?;
        let mut fields = Reader::new(fixed);
        let [version, r0, r1, r2] = fields.u32().ok_or(JournalError::Short)?.to_le_bytes();
        if version != VERSION {
            return Err(JournalError::Version(version));
        }
        if (r0, r1, r2) != (0, 0, 0) {
            return Err(JournalError::Reserved);
        }
        let dab_format = fields.u32().ok_or(JournalError::Short)?;
        let bits = fields.u64().ok_or(JournalError::Short)?;
        let op = OpId::from_bits(bits).ok_or(JournalError::ReservedSite)?;
        let mut next = || fields.u32().ok_or(JournalError::Short);
        let (first, landed, stated) = (next()?, next()?, next()?);
        let count = usize::try_from(stated).unwrap_or(usize::MAX);
        if count > MAX_JOURNAL_DABS {
            return Err(JournalError::TooManyDabs(count));
        }
        let tag = fields.u8().ok_or(JournalError::Short)?;
        let offset = fields.u64().ok_or(JournalError::Short)?;
        let pen_down = PenDown::from_parts(tag, offset).ok_or(JournalError::PenDown)?;
        let placed = first
            .checked_add(stated)
            .ok_or(JournalError::PastLastIndex)?;
        if landed > placed {
            return Err(JournalError::LandedPastPlaced { landed, placed });
        }
        Ok(BatchHead {
            op,
            first,
            landed,
            count: stated,
            pen_down,
            dab_format,
        })
    }
}

/// A batch's fixed fields, read without its dabs ([`DabBatch::decode_head`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchHead {
    /// The stroke's op.
    pub(crate) op: OpId,
    /// The index of the batch's first dab.
    pub(crate) first: u32,
    /// The stroke's dabs landed at the flush's cut.
    pub(crate) landed: u32,
    /// The dabs it holds.
    pub(crate) count: u32,
    /// The stroke's pen-down save point.
    pub(crate) pen_down: PenDown,
    /// The rows' format.
    pub(crate) dab_format: u32,
}

impl BatchHead {
    /// The dabs the stroke had placed by the batch's last: its first index plus its count. The
    /// head was checked not to run past the last index.
    pub(crate) const fn placed(&self) -> u32 {
        self.first.saturating_add(self.count)
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// [`DabBatch::decode_head`] never panics or overflows, whatever the payload's bytes and
    /// length up to the fixed fields and two more, and what it accepts passes every check
    /// [`DabBatch::decode`] makes of the fixed fields: a payload shorter than them is refused,
    /// and an accepted head's version is this build's, its reserved bytes clear, its count within
    /// [`MAX_JOURNAL_DABS`], its dabs' indices within a `u32`, and its landed count within them,
    /// so [`DabBatch::new`] takes the head as is.
    #[kani::proof]
    #[kani::unwind(3)]
    fn decode_head_is_in_range_and_refuses_what_decode_refuses() {
        let bytes: [u8; FIXED + 2] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len());
        let Some(payload) = bytes.get(..len) else {
            return;
        };
        let decoded = DabBatch::decode_head(payload);
        if len < FIXED {
            assert_eq!(
                decoded,
                Err(JournalError::Short),
                "short of the fixed fields"
            );
            return;
        }
        let Ok(head) = decoded else {
            return;
        };
        let count = usize::try_from(head.count).unwrap_or(usize::MAX);
        assert!(count <= MAX_JOURNAL_DABS, "the count within the limit");
        let placed = head.first.checked_add(head.count);
        assert!(placed.is_some(), "the indices within a u32");
        assert!(
            placed.is_some_and(|placed| head.landed <= placed),
            "landed within the dabs"
        );
        assert_eq!(
            payload.get(..4),
            Some([VERSION, 0, 0, 0].as_slice()),
            "this build's version, the reserved bytes clear"
        );
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::fixture::dab_at;

    /// The tests' batch: op 9 of site 2, three dabs of distinct bytes from index 37, 12 of the
    /// stroke's dabs landed, its pen-down surface at offset 4,096.
    fn batch() -> DabBatch {
        let dabs: Vec<[u8; DAB_BYTES]> = (1..=3_u8)
            .map(|k| std::array::from_fn(|j| u8::try_from(j).unwrap().wrapping_mul(k) ^ k))
            .collect();
        DabBatch::new(OpId::new(2, 9).unwrap(), 37, 12, PenDown::At(4_096), dabs)
            .unwrap()
            .with_dab_format(0x0102_0304)
    }

    #[test]
    fn a_batch_round_trips() {
        let batch = batch();
        assert_eq!(batch.first(), 37, "first");
        assert_eq!(batch.landed(), 12, "landed");
        assert_eq!(batch.placed(), 40, "placed");
        assert_eq!(batch.dabs().len(), 3, "dabs");
        assert_eq!(batch.pen_down(), PenDown::At(4_096), "pen-down");
        assert_eq!(batch.dab_format(), 0x0102_0304, "dab format");
        let payload = batch.encode().unwrap();
        assert_eq!(DabBatch::decode(&payload), Ok(batch), "round trip");
    }

    /// An empty skippable zstd frame: its magic number, then a length of zero.
    const SKIPPABLE: [u8; 8] = [0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0];

    #[test]
    fn an_empty_batch_s_frame_is_a_real_one() {
        let empty =
            DabBatch::new(OpId::new(0, 4).unwrap(), 3, 3, PenDown::Base, Vec::new()).unwrap();
        let mut payload = empty.encode().unwrap();
        payload.truncate(FIXED);
        payload.extend_from_slice(&SKIPPABLE);
        assert_eq!(
            DabBatch::decode(&payload),
            Err(JournalError::Frame),
            "a skippable frame alone"
        );
    }

    #[test]
    fn an_empty_batch_round_trips() {
        let op = OpId::new(0, 4).unwrap();
        let batch = DabBatch::new(op, 500, 480, PenDown::Unknown, Vec::new()).unwrap();
        let payload = batch.encode().unwrap();
        assert_eq!(
            DabBatch::decode(&payload),
            Ok(batch),
            "only the landed count"
        );
    }

    #[test]
    fn a_batch_a_reader_would_refuse_is_never_made() {
        let op = OpId::new(0, 1).unwrap();
        assert_eq!(
            DabBatch::new(
                op,
                0,
                0,
                PenDown::Base,
                vec![[0; DAB_BYTES]; MAX_JOURNAL_DABS + 1]
            ),
            Err(JournalError::TooManyDabs(MAX_JOURNAL_DABS + 1)),
            "too many dabs"
        );
        assert_eq!(
            DabBatch::new(op, 10, 13, PenDown::Base, vec![[0; DAB_BYTES]; 2]),
            Err(JournalError::LandedPastPlaced {
                landed: 13,
                placed: 12
            }),
            "more landed than placed"
        );
        assert_eq!(
            DabBatch::new(op, u32::MAX, 0, PenDown::Base, vec![[0; DAB_BYTES]; 1]),
            Err(JournalError::PastLastIndex),
            "an index past u32"
        );
        assert!(
            DabBatch::new(op, u32::MAX, u32::MAX, PenDown::Base, Vec::new()).is_ok(),
            "the last index itself"
        );
    }

    #[test]
    fn the_reader_refuses_what_a_writer_never_makes() {
        let payload = batch().encode().unwrap();
        let refused = |change: &dyn Fn(&mut Vec<u8>), error: JournalError| {
            let mut bad = payload.clone();
            change(&mut bad);
            assert_eq!(DabBatch::decode(&bad), Err(error), "{error}");
        };
        refused(&|p| p[0] = 2, JournalError::Version(2));
        refused(&|p| p[0] = 0, JournalError::Version(0));
        refused(&|p| p[3] = 1, JournalError::Reserved);
        refused(
            &|p| p[8..16].copy_from_slice(&u64::MAX.to_le_bytes()),
            JournalError::ReservedSite,
        );
        let count = u32::try_from(MAX_JOURNAL_DABS + 1).unwrap();
        refused(
            &|p| p[24..28].copy_from_slice(&count.to_le_bytes()),
            JournalError::TooManyDabs(MAX_JOURNAL_DABS + 1),
        );
        refused(
            &|p| p[24..28].copy_from_slice(&4_u32.to_le_bytes()),
            JournalError::Frame,
        );
        refused(&|p| p.truncate(p.len() - 1), JournalError::Frame);
        refused(&|p| p.push(0), JournalError::Frame);
        refused(
            &|p| p.extend_from_slice(&compress(&[]).unwrap()),
            JournalError::Frame,
        );
        refused(&|p| p.extend_from_slice(&SKIPPABLE), JournalError::Frame);
        refused(
            &|p| p[20..24].copy_from_slice(&41_u32.to_le_bytes()),
            JournalError::LandedPastPlaced {
                landed: 41,
                placed: 40,
            },
        );
        refused(
            &|p| p[16..20].copy_from_slice(&u32::MAX.to_le_bytes()),
            JournalError::PastLastIndex,
        );
        refused(&|p| p.truncate(FIXED - 1), JournalError::Short);
        refused(&|p| p[28] = 3, JournalError::PenDown);
        refused(&|p| p[28] = u8::MAX, JournalError::PenDown);
        // The tests' batch is at an offset, which neither of the others takes.
        refused(&|p| p[28] = 0, JournalError::PenDown);
        refused(&|p| p[28] = 2, JournalError::PenDown);
        refused(
            &|p| {
                p[28] = 0;
                p[29..37].copy_from_slice(&0_u64.to_le_bytes());
                p[36] = 1;
            },
            JournalError::PenDown,
        );
    }

    #[test]
    fn the_dab_format_is_carried_as_set() {
        let op = OpId::new(1, 3).unwrap();
        let plain = DabBatch::new(op, 0, 0, PenDown::Base, vec![[1; DAB_BYTES]; 2]).unwrap();
        assert_eq!(plain.dab_format(), 0, "format 0 unless set");
        for format in [0, 1, 77, u32::MAX] {
            let batch = plain.clone().with_dab_format(format);
            let payload = batch.encode().unwrap();
            assert_eq!(
                payload[4..8],
                format.to_le_bytes(),
                "{format}: after the reserved bytes"
            );
            assert_eq!(DabBatch::decode(&payload), Ok(batch), "{format}: read back");
        }
    }

    #[test]
    fn every_pen_down_round_trips() {
        let op = OpId::new(1, 3).unwrap();
        let pen_downs = [
            PenDown::Base,
            PenDown::At(0),
            PenDown::At(77),
            PenDown::At(u64::MAX),
            PenDown::Unknown,
        ];
        for pen_down in pen_downs {
            let batch = DabBatch::new(op, 2, 1, pen_down, vec![[5; DAB_BYTES]; 2]).unwrap();
            let payload = batch.encode().unwrap();
            let (tag, offset) = pen_down.to_parts();
            assert_eq!(payload[28], tag, "{pen_down:?}'s tag");
            assert_eq!(
                payload[29..37],
                offset.to_le_bytes(),
                "{pen_down:?}'s offset"
            );
            assert_eq!(DabBatch::decode(&payload), Ok(batch), "{pen_down:?}");
        }
    }

    #[test]
    fn a_pen_down_reads_only_as_written() {
        assert_eq!(PenDown::from_parts(0, 0), Some(PenDown::Base), "base");
        assert_eq!(PenDown::from_parts(1, 9), Some(PenDown::At(9)), "at");
        assert_eq!(PenDown::from_parts(2, 0), Some(PenDown::Unknown), "unknown");
        assert_eq!(PenDown::from_parts(0, 9), None, "base with an offset");
        assert_eq!(PenDown::from_parts(2, 1), None, "unknown with an offset");
        assert_eq!(PenDown::from_parts(3, 0), None, "another tag");
    }

    #[test]
    fn a_thousand_dabs_compress() {
        let dabs: Vec<[u8; DAB_BYTES]> = (0..1_000).map(dab_at).collect();
        let batch = DabBatch::new(OpId::new(0, 7).unwrap(), 0, 0, PenDown::Base, dabs).unwrap();
        let payload = batch.encode().unwrap();
        let raw = 1_000 * DAB_BYTES;
        assert!(
            payload.len() * 10 < raw * 6,
            "{} bytes for {raw} of dabs",
            payload.len()
        );
        assert_eq!(DabBatch::decode(&payload), Ok(batch), "and reads back");
    }
}
