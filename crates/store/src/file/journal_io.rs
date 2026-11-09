//! Dab batches in a recovery file (see [`crate::journal`]): appended with a flush's records,
//! before the save point that holds them, and read back to restore an open stroke.
//!
//! No Index names a batch: a save point holds every batch before it in the file, as it holds
//! stroke records. Opening a file finds them by the record heads it scans anyway, and reads none
//! of them. An op has many batches, one a flush, so unlike stroke records they are never
//! deduplicated: each append is a flush's own.

use std::os::unix::fs::FileExt;

use super::error::FileError;
use super::lock::Access;
use super::{Loc, ProjectFile};
use crate::journal::{BatchHead, DabBatch, MAX_JOURNAL_PAYLOAD};
use crate::record::{HEAD, Head, KIND_JOURNAL, TAIL, frame, verify};

/// The longest framed batch: framing around the longest payload a writer makes.
///
/// A head stating a longer one isn't taken for a batch, so a hostile file can't make a read
/// allocate its whole length.
const MAX_JOURNAL_RECORD: u64 = (HEAD + MAX_JOURNAL_PAYLOAD + TAIL) as u64;

#[cfg(test)]
thread_local! {
    /// Test hook: reads of batches on this thread, all counted by [`count_batch_read`].
    static BATCH_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test hook: the batch reads on this thread since the last call, counting again from zero.
#[cfg(test)]
pub(crate) fn take_batch_reads() -> u64 {
    BATCH_READS.with(|reads| reads.replace(0))
}

/// Counts a read of a batch, for the tests that check opening a file reads none: every such read
/// goes through here.
fn count_batch_read() {
    #[cfg(test)]
    BATCH_READS.with(|reads| reads.set(reads.get() + 1));
}

/// Where each batch of `heads` is, in file order. A head of the journal kind stating a record
/// longer than [`MAX_JOURNAL_RECORD`] is passed over, as data of a kind this build doesn't know
/// would be.
pub(super) fn batch_locs(heads: &[(u64, Head)]) -> impl Iterator<Item = Loc> + '_ {
    heads
        .iter()
        .filter(|(_, head)| head.kind == KIND_JOURNAL && head.total() <= MAX_JOURNAL_RECORD)
        .map(|&(offset, head)| Loc {
            offset,
            len: head.total(),
        })
}

/// `batch` framed as a dab-batch record, ready to append.
///
/// # Errors
///
/// [`FileError::Io`] if compressing fails; [`FileError::TooLarge`] never, for a batch within its
/// limits.
pub(crate) fn frame_batch(batch: &DabBatch) -> Result<Vec<u8>, FileError> {
    frame(KIND_JOURNAL, &batch.encode()?).ok_or(FileError::TooLarge)
}

impl ProjectFile {
    /// Appends `framed`, batches [`frame_batch`] made, in order, with one write. The next save
    /// point holds them, so a sync before it must cover them: they move the open cut's named end,
    /// if one is open.
    ///
    /// If the write fails, none of them is recorded (the bytes written past the end, if any, are
    /// overwritten by the next append).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`].
    pub(crate) fn append_batches(&mut self, framed: &[Vec<u8>]) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        let size = |n: usize| u64::try_from(n).or(Err(FileError::TooLarge));
        let buffer: Vec<u8> = framed.concat();
        if buffer.is_empty() {
            return Ok(());
        }
        let start = self.end;
        self.file.write_all_at(&buffer, start)?;
        self.end += size(buffer.len())?;
        let mut offset = start;
        for record in framed {
            let len = size(record.len())?;
            self.note_batch(Loc { offset, len });
            offset += len;
        }
        Ok(())
    }

    /// Records that a batch was appended at `loc`: the next save point holds it, and a sync
    /// before that save point must cover it.
    fn note_batch(&mut self, loc: Loc) {
        if let Some(cut) = &mut self.cut {
            cut.named_end = cut.named_end.max(loc.offset + loc.len);
        }
        self.batches.push(loc);
    }

    /// Where each batch of the file's state is, in file order: those the latest save point holds,
    /// then those appended since.
    pub(crate) fn batch_locations(&self) -> impl Iterator<Item = Loc> + '_ {
        let saved = self
            .saved
            .iter()
            .flat_map(|saved| saved.batches.iter().map(|(_, &loc)| loc));
        saved.chain(self.batches.iter().copied())
    }

    /// The batch at `loc` (one of [`ProjectFile::batch_locations`]), read and decoded. Every read
    /// of a batch goes through here.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if the record fails its checksum, isn't a batch, or isn't one this
    /// build reads (a newer version's, say); [`FileError::Closed`] if the file has been dropped;
    /// [`FileError::Io`].
    pub(crate) fn read_batch(&self, loc: Loc) -> Result<DabBatch, FileError> {
        count_batch_read();
        let raw = self.reader().read_record(loc)?;
        match verify(&raw) {
            Some((KIND_JOURNAL, payload)) => DabBatch::decode(payload).ok(),
            _ => None,
        }
        .ok_or(FileError::Damaged { offset: loc.offset })
    }

    /// The fixed fields of the batch at `loc` (one of [`ProjectFile::batch_locations`]): its
    /// whole record read and its checksum checked, as [`ProjectFile::read_batch`] does, but its
    /// dabs never decompressed ([`DabBatch::decode_head`]). For a restore, which looks over every
    /// batch to find the strokes left open.
    ///
    /// # Errors
    ///
    /// As [`ProjectFile::read_batch`].
    pub(crate) fn read_batch_head(&self, loc: Loc) -> Result<BatchHead, FileError> {
        count_batch_read();
        let raw = self.reader().read_record(loc)?;
        match verify(&raw) {
            Some((KIND_JOURNAL, payload)) => DabBatch::decode_head(payload).ok(),
            _ => None,
        }
        .ok_or(FileError::Damaged { offset: loc.offset })
    }

    /// The op of the stroke the file's journal leaves landing: that of its last batch, if that
    /// batch has dabs not landed (no closing batch came after). `None` with no batch, a last
    /// batch whose dabs have all landed, or one that doesn't read.
    ///
    /// Such a stroke is work a restore brings back even with no tile edit in the file (a stroke
    /// that had landed nothing), so the file isn't empty (see
    /// [`crate::RecoveryCandidate::last_edit`]). Whether a restore can use it is the restore's to
    /// say (see `crate::project::resume`).
    ///
    /// This reads the last batch's whole record (no longer than [`MAX_JOURNAL_RECORD`], as every
    /// batch location is) and checks its checksum, then decodes its fixed fields alone
    /// ([`DabBatch::decode_head`]), never decompressing its dabs. A record that fails its
    /// checksum counts as no stroke landing: an op read from damaged bytes could be one no
    /// stroke had, and one at its site's last sequence number would leave a new session no op
    /// to start, refusing to open the project. One batch read a file is the price.
    pub(crate) fn landing_stroke(&self) -> Option<crate::OpId> {
        let last = self.batch_locations().last()?;
        let raw = self.reader().read_record(last).ok()?;
        let (kind, payload) = verify(&raw)?;
        if kind != KIND_JOURNAL {
            return None;
        }
        let batch = DabBatch::decode_head(payload).ok()?;
        (batch.landed < batch.placed()).then_some(batch.op)
    }

    /// The newest op the file's state holds work of: its head ([`ProjectFile::head`]), or the
    /// op of the stroke its journal leaves landing ([`ProjectFile::landing_stroke`]) if that is
    /// greater. A stroke that has landed nothing has no tile record, so the head alone misses
    /// it; a restore brings it back all the same.
    ///
    /// Every reader of a file's newest op takes it from here: the op generator, which must never
    /// hand that stroke's op out again, and the recovery listing's last edit.
    pub(crate) fn last_op(&self) -> Option<crate::OpId> {
        self.head().max(self.landing_stroke())
    }

    /// Reads every batch of the file's state, in the order they were written: for tools, tests
    /// and the fuzz target. Opening a file reads none (see [`DabBatch`]).
    ///
    /// Each batch is read when the iterator reaches it, and one that doesn't read is an error in
    /// its place; the rest still read.
    ///
    /// # Errors
    ///
    /// Each item: [`FileError::Damaged`] if the batch fails its checksum, or isn't one this build
    /// reads (a newer version's, say); [`FileError::Closed`] if the file has been dropped;
    /// [`FileError::Io`].
    pub fn batches(&self) -> impl Iterator<Item = Result<DabBatch, FileError>> + '_ {
        self.batch_locations().map(|loc| self.read_batch(loc))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level, TileKey};

    use super::*;
    use crate::OpId;
    use crate::fixture::{dab_at, damage, mesh, meta};
    use crate::journal::{DAB_BYTES, PenDown};
    use crate::test_dir::TestDir;

    /// The `k`th test batch: op 5 of site 0, `k + 1` dabs from index `10 × k`, all but the last
    /// landed, its pen-down surface at the file's start.
    fn batch(k: u32) -> DabBatch {
        let dabs: Vec<[u8; DAB_BYTES]> = (10 * k..=11 * k).map(dab_at).collect();
        DabBatch::new(
            OpId::new(0, 5).unwrap(),
            10 * k,
            11 * k,
            PenDown::Base,
            dabs,
        )
        .unwrap()
    }

    /// Appends batches `ks` to `file`, each after a tile written by its op, a flush apiece.
    fn write_batches(file: &mut ProjectFile, ks: std::ops::Range<u32>) {
        let key = TileKey::displacement(FaceId::new(1), Level::new(1).unwrap(), 0, 0).unwrap();
        let tile = geom::CorrectionTile::from_values(3, &[geom::Vec3::ZERO; 9]).unwrap();
        for k in ks {
            let batch = batch(k);
            file.write_displacement(key, batch.op(), &tile).unwrap();
            file.append_batches(&[frame_batch(&batch).unwrap()])
                .unwrap();
        }
    }

    #[test]
    fn batches_round_trip_through_a_file_in_order() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_batches(&mut file, 0..2);
        file.save().unwrap();
        let framed = [
            frame_batch(&batch(2)).unwrap(),
            frame_batch(&batch(3)).unwrap(),
        ];
        file.append_batches(&framed).unwrap();
        assert!(file.has_unsaved_records(), "a batch alone is unsaved");
        file.save().unwrap();
        assert!(!file.has_unsaved_records(), "the save point holds it");
        let written: Vec<DabBatch> = (0..4).map(batch).collect();
        let while_open: Vec<DabBatch> = file.batches().map(Result::unwrap).collect();
        assert_eq!(while_open, written, "read back while open");
        drop(file);
        let reopened = ProjectFile::open(&path).unwrap();
        let reopened_batches: Vec<DabBatch> = reopened.batches().map(Result::unwrap).collect();
        assert_eq!(reopened_batches, written, "read back after a reopen");
    }

    #[test]
    fn a_save_point_holds_the_batches_before_it() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_batches(&mut file, 0..2);
        let first = file.save().unwrap();
        write_batches(&mut file, 2..3);
        file.save().unwrap();
        // A batch after the last save point: the tail of a flush that never finished.
        write_batches(&mut file, 3..4);
        assert_eq!(file.batch_locations().count(), 4, "counted while open");
        drop(file);
        let earlier = ProjectFile::open_at(&path, first).unwrap();
        let earlier_batches: Vec<DabBatch> = earlier.batches().map(Result::unwrap).collect();
        assert_eq!(
            earlier_batches,
            vec![batch(0), batch(1)],
            "the earlier save point's"
        );
        drop(earlier);
        let reopened = ProjectFile::open(&path).unwrap();
        let latest_batches: Vec<DabBatch> = reopened.batches().map(Result::unwrap).collect();
        let saved: Vec<DabBatch> = (0..3).map(batch).collect();
        assert_eq!(latest_batches, saved, "the tail is cut off");
    }

    #[test]
    fn opening_reads_no_batch_even_a_damaged_one() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_batches(&mut file, 0..3);
        let point = file.save().unwrap();
        let damaged = file.batch_locations().nth(1).unwrap();
        drop(file);
        damage(&path, damaged);
        take_batch_reads();
        drop(ProjectFile::open(&path).unwrap());
        drop(ProjectFile::open_at(&path, point).unwrap());
        assert_eq!(take_batch_reads(), 0, "no batch read to open");
        let reopened = ProjectFile::open(&path).unwrap();
        let read: Vec<Result<DabBatch, FileError>> = reopened.batches().collect();
        assert_eq!(take_batch_reads(), 3, "each read when asked for");
        assert!(
            matches!(read[1], Err(FileError::Damaged { offset }) if offset == damaged.offset),
            "the damaged one: {:?}",
            read[1]
        );
        assert_eq!(read[0].as_ref().unwrap(), &batch(0), "the others read");
        assert_eq!(read[2].as_ref().unwrap(), &batch(2), "the others read");
    }

    #[test]
    fn a_batch_of_a_newer_version_is_damage_only_when_read() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        let mut payload = batch(1).encode().unwrap();
        payload[0] = 2;
        let newer = frame(KIND_JOURNAL, &payload).unwrap();
        file.append_batches(&[newer]).unwrap();
        write_batches(&mut file, 2..3);
        file.save().unwrap();
        drop(file);
        let opened = ProjectFile::open(&path).unwrap();
        let read: Vec<Result<DabBatch, FileError>> = opened.batches().collect();
        assert_eq!(read.len(), 2, "both found by their heads");
        assert!(
            matches!(read[0], Err(FileError::Damaged { .. })),
            "a version this build doesn't read: {:?}",
            read[0]
        );
        assert_eq!(read[1].as_ref().unwrap(), &batch(2), "the next still reads");
    }

    #[test]
    fn a_record_of_another_kind_is_no_batch() {
        let dir = TestDir::new();
        let mut file = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
        write_batches(&mut file, 0..1);
        let key = TileKey::displacement(FaceId::new(1), Level::new(1).unwrap(), 0, 0).unwrap();
        let tile = file.locate(&key).unwrap();
        assert!(
            matches!(file.read_batch(tile), Err(FileError::Damaged { offset }) if offset == tile.offset),
            "a tile record"
        );
    }
}
