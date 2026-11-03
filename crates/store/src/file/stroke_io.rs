//! Stroke records in a project file (see [`crate::stroke_record`]): appended before the save point
//! that holds them, copied as they are between files, and read back for tools and tests only.
//!
//! No Index names a stroke record: a save point holds every stroke record before it in the file.
//! Opening a file finds them by the record heads it scans anyway, and reads none of them.
//!
//! A file holds at most one stroke record of an op: an append of a record whose op the file
//! holds already is skipped, so a save that failed and is made again, or one whose rebase was
//! refused, never writes a stroke's record twice.

use std::collections::BTreeSet;
use std::io;
use std::os::unix::fs::FileExt;

use super::error::FileError;
use super::lock::Access;
use super::reader::FileReader;
use super::{Loc, ProjectFile};
use crate::record::{HEAD, Head, KIND_STROKE, TAIL, frame, verify};
use crate::stroke_record::{MAX_STROKE_PAYLOAD, OP_AT, StrokeRecord};

/// The longest framed stroke record: framing around the longest payload a writer makes.
///
/// A head stating a longer one isn't taken for a stroke record, so a hostile file can't make a
/// copy or a read allocate its whole length.
const MAX_STROKE_RECORD: u64 = (HEAD + MAX_STROKE_PAYLOAD + TAIL) as u64;

/// Where a framed stroke record's op is, from its first byte: past the head.
const FRAMED_OP_AT: usize = HEAD + OP_AT;

#[cfg(test)]
thread_local! {
    /// Test hook: reads of stroke records (whole, or their op) on this thread, all counted by
    /// [`count_stroke_read`].
    static STROKE_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test hook: the stroke record reads on this thread since the last call, counting again from
/// zero.
#[cfg(test)]
pub(crate) fn take_stroke_reads() -> u64 {
    STROKE_READS.with(|reads| reads.replace(0))
}

/// Counts a read of a stroke record, whole or in part, for the tests that check opening a file
/// reads none: every such read goes through here.
fn count_stroke_read() {
    #[cfg(test)]
    STROKE_READS.with(|reads| reads.set(reads.get() + 1));
}

/// Where each stroke record of `heads` is, in file order. A head of the stroke kind stating a
/// record longer than [`MAX_STROKE_RECORD`] is passed over, as data of a kind this build doesn't
/// know would be.
pub(super) fn stroke_locs(heads: &[(u64, Head)]) -> impl Iterator<Item = Loc> + '_ {
    heads
        .iter()
        .filter(|(_, head)| head.kind == KIND_STROKE && head.total() <= MAX_STROKE_RECORD)
        .map(|&(offset, head)| Loc {
            offset,
            len: head.total(),
        })
}

/// `record` framed as a stroke record, ready to append.
///
/// # Errors
///
/// [`FileError::Io`] if compressing fails; [`FileError::TooLarge`] never, for a record within its
/// limits.
pub(crate) fn frame_stroke(record: &StrokeRecord) -> Result<Vec<u8>, FileError> {
    frame(KIND_STROKE, &record.to_payload()?).ok_or(FileError::TooLarge)
}

/// The stroke record at `loc` of the file `reader` reads, framing included, after checking that
/// it verifies and is a stroke record. Every whole read of a stroke record goes through here.
///
/// # Errors
///
/// [`FileError::Damaged`] if it doesn't, or the file ends first; [`FileError::Closed`] if the
/// file has been dropped; [`FileError::Io`].
pub(super) fn read_stroke_at(reader: &FileReader, loc: Loc) -> Result<Vec<u8>, FileError> {
    count_stroke_read();
    let raw = reader.read_record(loc)?;
    if is_stroke_record(&raw) {
        Ok(raw)
    } else {
        Err(FileError::Damaged { offset: loc.offset })
    }
}

/// Whether `raw` is a whole stroke record: its framing and checksum are good and its kind is a
/// stroke's. The payload isn't parsed: a copy carries it as it is.
fn is_stroke_record(raw: &[u8]) -> bool {
    matches!(verify(raw), Some((KIND_STROKE, _)))
}

/// The op bits of the framed stroke record `raw`, unchecked; `None` if it is too short to hold
/// them.
fn op_bits(raw: &[u8]) -> Option<u64> {
    let bytes = raw.get(FRAMED_OP_AT..FRAMED_OP_AT + 8)?;
    bytes.try_into().ok().map(u64::from_le_bytes)
}

impl ProjectFile {
    /// Appends `framed`, stroke records [`frame_stroke`] made, in order, with one write, but for
    /// those of an op the file holds already (or one earlier in `framed`). The next save point
    /// holds them, so a sync before it must cover them: they move the open cut's named end, if
    /// one is open.
    ///
    /// If the write fails, none of them is recorded (the bytes written past the end, if any, are
    /// overwritten by the next append).
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`], [`FileError::TooLarge`], [`FileError::Io`] (reading the ops the
    /// file holds, the first time, too).
    pub(crate) fn append_strokes(&mut self, framed: &[Vec<u8>]) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        if framed.is_empty() {
            return Ok(());
        }
        let size = |n: usize| u64::try_from(n).or(Err(FileError::TooLarge));
        let held = self.stroke_ops()?;
        let mut new_ops = BTreeSet::new();
        let fresh: Vec<&Vec<u8>> = framed
            .iter()
            .filter(|record| {
                op_bits(record).is_none_or(|op| !held.contains(&op) && new_ops.insert(op))
            })
            .collect();
        let buffer: Vec<u8> = fresh
            .iter()
            .flat_map(|record| record.iter().copied())
            .collect();
        if buffer.is_empty() {
            return Ok(());
        }
        let start = self.end;
        self.file.write_all_at(&buffer, start)?;
        self.end += size(buffer.len())?;
        let mut offset = start;
        for record in fresh {
            let len = size(record.len())?;
            self.note_stroke(Loc { offset, len }, op_bits(record));
            offset += len;
        }
        Ok(())
    }

    /// Appends `raw`, a stroke record read from another file, as it is, unless the file holds a
    /// stroke record of its op already.
    ///
    /// # Errors
    ///
    /// [`FileError::ReadOnly`]; [`FileError::Damaged`] if `raw` isn't a whole stroke record;
    /// [`FileError::Io`] (reading the ops the file holds, the first time, too).
    pub(crate) fn append_stroke_copy(&mut self, raw: &[u8]) -> Result<(), FileError> {
        if self.access == Access::ReadOnly {
            return Err(FileError::ReadOnly);
        }
        if !is_stroke_record(raw) {
            return Err(FileError::Damaged { offset: self.end });
        }
        let op = op_bits(raw);
        if let Some(op) = op
            && self.stroke_ops()?.contains(&op)
        {
            return Ok(());
        }
        let loc = self.append_record(raw)?;
        self.note_stroke(loc, op);
        Ok(())
    }

    /// The ops of the stroke records the file holds, read from each record's op field (8 bytes,
    /// unchecked) the first time they are wanted. A record too short to hold one is passed over.
    ///
    /// # Errors
    ///
    /// [`FileError::Io`].
    fn stroke_ops(&mut self) -> Result<&BTreeSet<u64>, FileError> {
        if self.stroke_ops.is_none() {
            let mut ops = BTreeSet::new();
            let locs: Vec<Loc> = self.stroke_locations().collect();
            for loc in locs {
                count_stroke_read();
                let mut bits = [0; 8];
                match self
                    .file
                    .read_exact_at(&mut bits, loc.offset + FRAMED_OP_AT as u64)
                {
                    Ok(()) => {
                        ops.insert(u64::from_le_bytes(bits));
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {}
                    Err(e) => return Err(e.into()),
                }
            }
            self.stroke_ops = Some(ops);
        }
        Ok(self.stroke_ops.get_or_insert_with(BTreeSet::new))
    }

    /// Records that a stroke record of `op` was appended at `loc`: the next save point holds it,
    /// and a sync before that save point must cover it.
    fn note_stroke(&mut self, loc: Loc, op: Option<u64>) {
        if let Some(cut) = &mut self.cut {
            cut.named_end = cut.named_end.max(loc.offset + loc.len);
        }
        self.strokes.push(loc);
        if let (Some(ops), Some(op)) = (&mut self.stroke_ops, op) {
            ops.insert(op);
        }
    }

    /// Where each stroke record of the file's state is, in file order: those the latest save
    /// point holds, then those appended since.
    pub(crate) fn stroke_locations(&self) -> impl Iterator<Item = Loc> + '_ {
        let saved = self
            .saved
            .iter()
            .flat_map(|saved| saved.strokes.iter().map(|(_, &loc)| loc));
        saved.chain(self.strokes.iter().copied())
    }

    /// The stroke record at `loc` (one of [`ProjectFile::stroke_locations`]), framing included,
    /// after checking that it verifies and is a stroke record.
    ///
    /// # Errors
    ///
    /// [`FileError::Damaged`] if it doesn't; [`FileError::Io`].
    pub(crate) fn read_stroke_record(&self, loc: Loc) -> Result<Vec<u8>, FileError> {
        read_stroke_at(&self.reader(), loc)
    }

    /// How many stroke records the file's state holds: the latest save point's and those
    /// appended since.
    #[must_use]
    pub fn stroke_count(&self) -> usize {
        self.saved.as_ref().map_or(0, |saved| saved.strokes.len()) + self.strokes.len()
    }

    /// Reads every stroke record of the file's state, in the order they were written: for tools
    /// and tests. Opening a file reads none (see [`StrokeRecord`]).
    ///
    /// Each record is read when the iterator reaches it, and a record that doesn't read is an
    /// error in its place; the rest still read.
    ///
    /// # Errors
    ///
    /// Each item: [`FileError::Damaged`] if the record fails its checksum, or isn't a stroke
    /// record this build reads (a newer version's, say); [`FileError::Io`].
    pub fn strokes(&self) -> impl Iterator<Item = Result<StrokeRecord, FileError>> + '_ {
        self.stroke_locations().map(|loc| {
            let raw = self.read_stroke_record(loc)?;
            verify(&raw)
                .and_then(|(_, payload)| StrokeRecord::from_payload(payload))
                .ok_or(FileError::Damaged { offset: loc.offset })
        })
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::path::Path;

    use geom::{FaceId, Level, TileKey};

    use super::*;
    use crate::OpId;
    use crate::fixture::{damage, mesh, meta};
    use crate::meta::SavePoint;
    use crate::stroke_record::tests::wobbly;
    use crate::test_dir::TestDir;

    /// A displacement tile of the fixture's mesh, all zero.
    fn zero_tile() -> (TileKey, geom::CorrectionTile) {
        let key = TileKey::displacement(FaceId::new(1), Level::new(1).unwrap(), 0, 0).unwrap();
        let values = vec![geom::Vec3::ZERO; 9];
        let tile = geom::CorrectionTile::from_values(key.side(), &values).unwrap();
        (key, tile)
    }

    /// The `k`th test stroke's record: op `k` of site 0, and `20 + k` samples.
    fn record(k: u32) -> StrokeRecord {
        wobbly(OpId::new(0, k.into()).unwrap(), 20 + k, k.into())
    }

    /// Appends the records of strokes `ks` to `file`, each after a tile written by its op.
    fn write_strokes(file: &mut ProjectFile, ks: std::ops::Range<u32>) {
        let (key, tile) = zero_tile();
        for k in ks {
            let stroke = record(k);
            file.write_displacement(key, stroke.op(), &tile).unwrap();
            file.append_strokes(&[frame_stroke(&stroke).unwrap()])
                .unwrap();
        }
    }

    /// Every stroke record `path` holds at its latest save point, read back.
    fn strokes_of(path: &Path) -> Vec<StrokeRecord> {
        let file = ProjectFile::open(path).unwrap();
        file.strokes().map(Result::unwrap).collect()
    }

    #[test]
    fn stroke_records_round_trip_through_a_file() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_strokes(&mut file, 0..2);
        file.save().unwrap();
        file.append_strokes(&[frame_stroke(&record(2)).unwrap()])
            .unwrap();
        assert!(
            file.has_unsaved_records(),
            "a stroke record alone is unsaved"
        );
        file.save().unwrap();
        assert!(!file.has_unsaved_records(), "the save point holds it");
        write_strokes(&mut file, 3..5);
        file.save().unwrap();
        let written: Vec<StrokeRecord> = (0..5).map(record).collect();
        let read: Vec<StrokeRecord> = file.strokes().map(Result::unwrap).collect();
        assert_eq!(read, written, "read back while open");
        drop(file);
        assert_eq!(strokes_of(&path), written, "read back after a reopen");
    }

    #[test]
    fn a_save_point_holds_the_stroke_records_before_it() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_strokes(&mut file, 0..2);
        let first = file.save().unwrap();
        write_strokes(&mut file, 2..3);
        file.save().unwrap();
        // A stroke after the last save point: the tail of a flush that never finished.
        write_strokes(&mut file, 3..4);
        assert_eq!(file.stroke_count(), 4, "counted while open");
        drop(file);
        let earlier = ProjectFile::open_at(&path, first).unwrap();
        let held: Vec<StrokeRecord> = earlier.strokes().map(Result::unwrap).collect();
        assert_eq!(held, vec![record(0), record(1)], "the earlier save point's");
        drop(earlier);
        let reopened = ProjectFile::open(&path).unwrap();
        assert_eq!(reopened.stroke_count(), 3, "the tail is cut off");
        let at_first = reopened
            .base_at(SavePoint {
                project: reopened.meta().project(),
                offset: first,
                head: Some(OpId::new(0, 1).unwrap()),
            })
            .unwrap()
            .unwrap();
        let latest = reopened.base().unwrap();
        assert_eq!(latest.stroke_locations().count(), 3, "the latest base's");
        assert_eq!(at_first.stroke_locations().count(), 2, "an earlier base's");
    }

    #[test]
    fn opening_reads_no_stroke_record_even_a_damaged_one() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        write_strokes(&mut file, 0..3);
        let point = file.save().unwrap();
        let damaged = file.stroke_locations().nth(1).unwrap();
        drop(file);
        damage(&path, damaged);
        take_stroke_reads();
        let opened = ProjectFile::open(&path).unwrap();
        let (key, _) = zero_tile();
        assert!(
            opened.read_displacement(&key).unwrap().is_some(),
            "the edits read"
        );
        drop(opened);
        drop(ProjectFile::open_at(&path, point).unwrap());
        assert_eq!(take_stroke_reads(), 0, "no stroke record read to open");
        let reopened = ProjectFile::open(&path).unwrap();
        let read: Vec<Result<StrokeRecord, FileError>> = reopened.strokes().collect();
        assert_eq!(take_stroke_reads(), 3, "each read when asked for");
        assert!(
            matches!(read[1], Err(FileError::Damaged { offset }) if offset == damaged.offset),
            "the damaged one: {:?}",
            read[1]
        );
        assert_eq!(read[0].as_ref().unwrap(), &record(0), "the others read");
        assert_eq!(read[2].as_ref().unwrap(), &record(2), "the others read");
    }

    #[test]
    fn a_stroke_record_of_a_newer_version_is_damage_only_when_read() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        let mut payload = record(0).to_payload().unwrap();
        payload[0] = 3;
        let newer = frame(KIND_STROKE, &payload).unwrap();
        file.append_strokes(&[newer]).unwrap();
        file.save().unwrap();
        drop(file);
        let opened = ProjectFile::open(&path).unwrap();
        assert_eq!(opened.stroke_count(), 1, "found by its head");
        assert!(
            matches!(
                opened.strokes().next(),
                Some(Err(FileError::Damaged { .. }))
            ),
            "a version this build doesn't read"
        );
    }

    #[test]
    fn copies_refuse_what_isnt_a_whole_stroke_record() {
        let dir = TestDir::new();
        let mut file = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
        let mut good = frame_stroke(&record(0)).unwrap();
        file.append_stroke_copy(&good).unwrap();
        let last = good.len() - 1;
        good[last] ^= 1;
        assert!(
            matches!(
                file.append_stroke_copy(&good),
                Err(FileError::Damaged { .. })
            ),
            "a bad checksum"
        );
        let tile = frame(crate::record::KIND_TILE, b"x").unwrap();
        assert!(
            matches!(
                file.append_stroke_copy(&tile),
                Err(FileError::Damaged { .. })
            ),
            "another kind"
        );
        assert_eq!(file.stroke_count(), 1, "only the good one");
    }
}
