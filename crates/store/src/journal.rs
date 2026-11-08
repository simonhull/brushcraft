//! The dab journal: the dabs an open stroke has placed and not yet landed, kept in the recovery
//! file so that a crash, or a close that leaves the landing to finish unseen, loses none of them.
//!
//! While a stroke's op is open and some of its dabs haven't landed, each recovery flush writes
//! one [`DabBatch`]: the dabs placed since the last batch (none, if only the landed count moved)
//! and how many of the stroke's dabs had landed at the flush's cut. The batches of an op, in file
//! order, hold every dab the stroke placed, so restoring needs only them and the tiles as the
//! stroke found them, never the stroke engine. A dab (`stroke::Dab`) is [`DAB_BYTES`] bytes of
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
//! - the op, as `OpId::to_bits` packs it, a `u64`;
//! - the stroke's index of the batch's first dab, a `u32`: dabs counted in placement order,
//!   symmetry copies included;
//! - how many of the stroke's dabs had landed at the flush's cut, a `u32`;
//! - the dabs' count, a `u32`, at most [`MAX_JOURNAL_DABS`];
//! - a zstd frame of the dabs transposed into [`DAB_BYTES`] byte planes (byte `j` of dab `i` at `j
//!   × count + i`): `DAB_BYTES × count` bytes once decompressed, exactly.
//!
//! A reader refuses any other version, a reserved byte set, a reserved site's op, a count past
//! its limit, anything but exactly one zstd frame after the fixed fields (a skippable frame, or
//! bytes after the frame), a frame that doesn't decompress to the stated size, a batch whose dabs
//! run past the last `u32` index, and a landed count past the dabs placed: the first index plus
//! the count.
//!
//! # Where batches live
//!
//! No Index names a batch: a save point holds every batch before it in the file, as it holds
//! stroke records (see `crate::file::journal_io`), so the save point that names a flush's tiles
//! holds its batch, and the batch's landed count describes exactly that cut. Opening a file only
//! steps over their heads.

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

/// Payload bytes before the compressed frame: version and reserved bytes, op, first index,
/// landed count and dab count.
const FIXED: usize = 4 + 8 + 4 + 4 + 4;

/// The longest payload a writer makes: the fixed fields, then a zstd frame of the most dabs,
/// which zstd never expands by more than 1/128 plus a small constant.
pub(crate) const MAX_JOURNAL_PAYLOAD: usize = {
    let raw = DAB_BYTES * MAX_JOURNAL_DABS;
    FIXED + raw + raw / 128 + 1024
};

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
    /// A landed count given to the open stroke's journal is below the one it had: dabs never
    /// unland. Only the store's journal refuses this; no reader meets it.
    #[error("{landed} dabs landed, after {before} had")]
    LandedBack {
        /// The landed count given.
        landed: u32,
        /// The count the journal had.
        before: u32,
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
    /// The frame doesn't decompress to exactly the stated dabs.
    #[error("the dab batch's frame doesn't hold its dabs")]
    Frame,
}

/// One flush's part of an open stroke's journal (record kind 7): the dabs the stroke placed since
/// the batch before, from index [`DabBatch::first`], and how many of its dabs had landed at the
/// flush's cut.
///
/// Made by [`DabBatch::new`], which holds it to what a reader of the record takes, so every batch
/// the store writes reads back as the same batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DabBatch {
    /// The stroke's op.
    op: OpId,
    /// The index in the stroke of `dabs[0]`: dabs counted in placement order, symmetry copies
    /// included.
    first: u32,
    /// How many of the stroke's dabs had landed at the flush's cut; at most `first + dabs.len()`.
    landed: u32,
    /// Opaque rows of [`DAB_BYTES`], at most [`MAX_JOURNAL_DABS`], ending at or before the last
    /// `u32` index.
    dabs: Vec<[u8; DAB_BYTES]>,
}

impl DabBatch {
    /// The batch of the stroke made as `op` holding `dabs` (each a `stroke::Dab`'s bytes) from
    /// its index `first`, with `landed` of its dabs landed.
    ///
    /// # Errors
    ///
    /// [`JournalError::TooManyDabs`] past [`MAX_JOURNAL_DABS`]; [`JournalError::PastLastIndex`]
    /// if `first` plus the count is past `u32::MAX`; [`JournalError::LandedPastPlaced`] if
    /// `landed` is past that.
    pub fn new(
        op: OpId,
        first: u32,
        landed: u32,
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
            dabs,
        })
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
        out.extend_from_slice(&self.op.to_bits().to_le_bytes());
        out.extend_from_slice(&self.first.to_le_bytes());
        out.extend_from_slice(&self.landed.to_le_bytes());
        out.extend_from_slice(&self.count().to_le_bytes());
        out.extend_from_slice(&packed);
        Ok(out)
    }

    /// The batch `payload` holds.
    ///
    /// # Errors
    ///
    /// The [`JournalError`] of the first thing refused (see the module notes):
    /// [`JournalError::Short`], [`JournalError::Version`], [`JournalError::Reserved`],
    /// [`JournalError::ReservedSite`], [`JournalError::TooManyDabs`], [`JournalError::Frame`],
    /// [`JournalError::PastLastIndex`] or [`JournalError::LandedPastPlaced`].
    pub(crate) fn decode(payload: &[u8]) -> Result<Self, JournalError> {
        let mut reader = Reader::new(payload);
        let fixed = reader.take(FIXED).ok_or(JournalError::Short)?;
        let mut fields = Reader::new(fixed);
        let [version, r0, r1, r2] = fields.u32().ok_or(JournalError::Short)?.to_le_bytes();
        if version != VERSION {
            return Err(JournalError::Version(version));
        }
        if (r0, r1, r2) != (0, 0, 0) {
            return Err(JournalError::Reserved);
        }
        let bits = fields.u64().ok_or(JournalError::Short)?;
        let op = OpId::from_bits(bits).ok_or(JournalError::ReservedSite)?;
        let mut next = || fields.u32().ok_or(JournalError::Short);
        let (first, landed, stated) = (next()?, next()?, next()?);
        let count = usize::try_from(stated).unwrap_or(usize::MAX);
        if count > MAX_JOURNAL_DABS {
            return Err(JournalError::TooManyDabs(count));
        }
        let planes =
            decompress_frame(reader.rest(), count * DAB_BYTES).or(Err(JournalError::Frame))?;
        let dabs: Vec<[u8; DAB_BYTES]> = untranspose(&planes, count)
            .ok_or(JournalError::Frame)?
            .collect();
        Self::new(op, first, landed, dabs)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::fixture::dab_at;

    /// The tests' batch: op 9 of site 2, three dabs of distinct bytes from index 37, 12 of the
    /// stroke's dabs landed.
    fn batch() -> DabBatch {
        let dabs: Vec<[u8; DAB_BYTES]> = (1..=3_u8)
            .map(|k| std::array::from_fn(|j| u8::try_from(j).unwrap().wrapping_mul(k) ^ k))
            .collect();
        DabBatch::new(OpId::new(2, 9).unwrap(), 37, 12, dabs).unwrap()
    }

    #[test]
    fn a_batch_round_trips() {
        let batch = batch();
        assert_eq!(batch.first(), 37, "first");
        assert_eq!(batch.landed(), 12, "landed");
        assert_eq!(batch.placed(), 40, "placed");
        assert_eq!(batch.dabs().len(), 3, "dabs");
        let payload = batch.encode().unwrap();
        assert_eq!(DabBatch::decode(&payload), Ok(batch), "round trip");
    }

    /// An empty skippable zstd frame: its magic number, then a length of zero.
    const SKIPPABLE: [u8; 8] = [0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0];

    #[test]
    fn an_empty_batch_s_frame_is_a_real_one() {
        let empty = DabBatch::new(OpId::new(0, 4).unwrap(), 3, 3, Vec::new()).unwrap();
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
        let batch = DabBatch::new(op, 500, 480, Vec::new()).unwrap();
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
            DabBatch::new(op, 0, 0, vec![[0; DAB_BYTES]; MAX_JOURNAL_DABS + 1]),
            Err(JournalError::TooManyDabs(MAX_JOURNAL_DABS + 1)),
            "too many dabs"
        );
        assert_eq!(
            DabBatch::new(op, 10, 13, vec![[0; DAB_BYTES]; 2]),
            Err(JournalError::LandedPastPlaced {
                landed: 13,
                placed: 12
            }),
            "more landed than placed"
        );
        assert_eq!(
            DabBatch::new(op, u32::MAX, 0, vec![[0; DAB_BYTES]; 1]),
            Err(JournalError::PastLastIndex),
            "an index past u32"
        );
        assert!(
            DabBatch::new(op, u32::MAX, u32::MAX, Vec::new()).is_ok(),
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
            &|p| p[4..12].copy_from_slice(&u64::MAX.to_le_bytes()),
            JournalError::ReservedSite,
        );
        let count = u32::try_from(MAX_JOURNAL_DABS + 1).unwrap();
        refused(
            &|p| p[20..24].copy_from_slice(&count.to_le_bytes()),
            JournalError::TooManyDabs(MAX_JOURNAL_DABS + 1),
        );
        refused(
            &|p| p[20..24].copy_from_slice(&4_u32.to_le_bytes()),
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
            &|p| p[16..20].copy_from_slice(&41_u32.to_le_bytes()),
            JournalError::LandedPastPlaced {
                landed: 41,
                placed: 40,
            },
        );
        refused(
            &|p| p[12..16].copy_from_slice(&u32::MAX.to_le_bytes()),
            JournalError::PastLastIndex,
        );
        refused(&|p| p.truncate(FIXED - 1), JournalError::Short);
    }

    #[test]
    fn a_thousand_dabs_compress() {
        let dabs: Vec<[u8; DAB_BYTES]> = (0..1_000).map(dab_at).collect();
        let batch = DabBatch::new(OpId::new(0, 7).unwrap(), 0, 0, dabs).unwrap();
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
