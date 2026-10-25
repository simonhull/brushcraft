//! What the whole-file fuzz target (`fuzz/fuzz_targets/project_open.rs`) needs from inside the
//! crate. Built only under cargo-fuzz's `cfg(fuzzing)` (and for its own tests), so no other
//! build has it.
//!
//! Every record is checksummed, so a fuzzer's change to a file fails [`crate::record::verify`] at
//! the first record it touches and nothing behind the checksum is ever reached. The target lets
//! the fuzzer change any byte, then recomputes each record's checksum before opening the file:
//! the framing is the store's own, so this is the one thing it can't do from outside.

use crate::crc::crc32;
use crate::file::HEADER_LEN;
use crate::record::{HEAD, TAIL, parse_head};

/// Recomputes the checksum of every record of the project file `file`, in place.
///
/// Walks the records by their heads from the end of the file header, as reading does, and stops
/// at the first head that doesn't parse or the first record that runs past the end; what follows
/// is left as it is.
pub fn fix_checksums(file: &mut [u8]) {
    let mut at = HEADER_LEN;
    while let Some(head) = file
        .get(at..at.saturating_add(HEAD))
        .and_then(|head| parse_head(head.try_into().ok()?))
    {
        let Some(end) = usize::try_from(head.total())
            .ok()
            .and_then(|len| at.checked_add(len))
        else {
            return;
        };
        let Some((body, crc)) = file
            .get_mut(at..end)
            .and_then(|record| record.split_at_mut_checked(end - at - TAIL))
        else {
            return;
        };
        crc.copy_from_slice(&crc32(body).to_le_bytes());
        at = end;
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use geom::{FaceId, Level, TileKey};

    use super::*;
    use crate::bytes::{KEY_LEN, put_key};
    use crate::fixture::{mesh, meta};
    use crate::test_dir::TestDir;
    use crate::{OpId, ProjectFile};

    /// A changed payload byte fails its record's read until the checksums are fixed; then the
    /// change is read as the file's content (here, a tile's op).
    #[test]
    fn fixed_checksums_let_changed_payloads_through() {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let key = TileKey::displacement(FaceId::new(1), Level::new(1).unwrap(), 0, 0).unwrap();
        let tile = geom::CorrectionTile::from_values(3, &[geom::Vec3::ZERO; 9]).unwrap();
        let mut file = ProjectFile::create(&path, meta(), mesh()).unwrap();
        file.write_displacement(key, OpId::new(0, 1).unwrap(), &tile)
            .unwrap();
        file.save().unwrap();
        drop(file);
        let mut bytes = std::fs::read(&path).unwrap();
        // The tile record's op, after its key: sequence 1 becomes 2.
        let mut needle = Vec::new();
        put_key(&mut needle, &key);
        needle.extend_from_slice(&1_u64.to_le_bytes());
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .unwrap();
        bytes[at + KEY_LEN] = 2;
        std::fs::write(&path, &bytes).unwrap();
        let stale = ProjectFile::open(&path).and_then(|opened| opened.read_displacement(&key));
        assert!(stale.is_err(), "a stale checksum is damage: {stale:?}");
        fix_checksums(&mut bytes);
        std::fs::write(&path, &bytes).unwrap();
        let reopened = ProjectFile::open(&path).unwrap();
        let (op, _) = reopened.read_displacement(&key).unwrap().unwrap();
        assert_eq!(op, OpId::new(0, 2).unwrap(), "the changed op is read");
    }
}
