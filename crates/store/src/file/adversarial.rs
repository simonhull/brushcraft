//! Damaged, hostile and concurrent use of the project file: each test pins a recovery or
//! validation rule.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use geom::{FaceId, Level, Vec3};

use super::*;
use crate::test_dir::TestDir;

fn key(face: u32) -> TileKey {
    TileKey::displacement(FaceId::new(face), Level::new(9).unwrap(), 0, 0).unwrap()
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

/// Saves S1 (full), S2 and S3 (deltas), one tile each; returns the bytes and the three offsets.
fn three(path: &Path) -> (Vec<u8>, [u64; 3]) {
    let mut f = ProjectFile::create(path).unwrap();
    let mut points = [0; 3];
    for (seq, point) in (1..).zip(&mut points) {
        f.write_displacement(key(u32::try_from(seq).unwrap()), op(seq), &tile(1.0))
            .unwrap();
        *point = f.save().unwrap();
    }
    drop(f);
    (std::fs::read(path).unwrap(), points)
}

/// Byte offset of a tile payload's `raw_len` field, for a header of `header_len` bytes.
const fn raw_len_at(header_len: usize) -> usize {
    KEY_LEN + 8 + 2 + header_len
}

#[test]
fn a_huge_stated_size_is_rejected_without_allocating() {
    let packed = compress(&[0; 16]).unwrap();
    let mut payload = tile_payload(key(2), op(2), b"", 16, &packed).unwrap();
    let at = raw_len_at(0);
    for (raw_len, parses) in [(MAX_RAW, true), (MAX_RAW + 1, false), (1 << 30, false)] {
        payload[at..at + 4].copy_from_slice(&raw_len.to_le_bytes());
        assert_eq!(
            parse_tile_payload(&payload).is_some(),
            parses,
            "raw_len {raw_len}"
        );
    }
    // The 1 GiB claim, in a record a save point names.
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    f.write_displacement(key(1), op(1), &tile(1.0)).unwrap();
    let parent = f.save().unwrap();
    let offset = f.len();
    f.append_raw(KIND_TILE, &payload).unwrap();
    let entries = BTreeMap::from([(
        key(2),
        Loc {
            offset,
            len: f.len() - offset,
        },
    )]);
    f.append_raw(
        KIND_INDEX,
        &index_payload(Coverage::Delta, Some(parent), Some(op(2)), &entries).unwrap(),
    )
    .unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert!(opened.contains(&key(2)), "named by the last save point");
    assert!(
        matches!(opened.read_tile(&key(2)), Err(StoreError::Damaged { offset: o }) if o == offset),
        "unparseable, so damaged"
    );
    assert!(
        opened.read_tile(&key(1)).unwrap().is_some(),
        "others still readable"
    );
}

#[test]
fn a_huge_unsaved_record_is_skipped_on_replay() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    f.write_displacement(key(1), op(1), &tile(1.0)).unwrap();
    f.save().unwrap();
    let packed = compress(&[0; 16]).unwrap();
    let mut payload = tile_payload(key(2), op(2), b"", 16, &packed).unwrap();
    let at = raw_len_at(0);
    payload[at..at + 4].copy_from_slice(&(1u32 << 30).to_le_bytes());
    f.append_raw(KIND_TILE, &payload).unwrap();
    f.write_displacement(key(3), op(3), &tile(3.0)).unwrap();
    let len = f.len();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.keys().copied().collect::<Vec<_>>(),
        vec![key(1), key(3)],
        "skipped, and replay goes on"
    );
    assert_eq!(opened.len(), len, "nothing cut off");
}

#[test]
fn bodies_over_the_cap_are_not_written() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    let cap = usize::try_from(MAX_RAW).unwrap();
    let before = f.len();
    assert!(
        matches!(
            f.write_tile(key(1), op(1), b"", &vec![0; cap + 1]),
            Err(StoreError::TooLarge)
        ),
        "one byte over"
    );
    assert_eq!(f.len(), before, "nothing appended");
    f.write_tile(key(1), op(1), b"", &vec![7; cap]).unwrap();
    assert_eq!(
        f.read_tile(&key(1)).unwrap().unwrap().body.len(),
        cap,
        "exactly the cap"
    );
}

#[test]
fn displacement_reads_check_the_stated_size_before_decompressing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    let (header, raw) = encode_displacement(&tile(2.0)).unwrap();
    // A good header and frame, but a stated size the side doesn't need.
    let mut payload =
        tile_payload(key(1), op(1), &header, raw.len(), &compress(&raw).unwrap()).unwrap();
    let at = raw_len_at(header.len());
    payload[at..at + 4].copy_from_slice(&MAX_RAW.to_le_bytes());
    let loc = f.append(KIND_TILE, &payload).unwrap();
    let offset = loc.offset;
    f.live.insert(key(1), loc);
    f.pending.insert(key(1), loc);
    f.save().unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert!(
        matches!(
            opened.read_displacement(&key(1)),
            Err(StoreError::Codec(CodecError::Size))
        ),
        "the side needs 26,886 bytes, not MAX_RAW"
    );
    assert!(
        matches!(opened.read_tile(&key(1)), Err(StoreError::Damaged { offset: o }) if o == offset),
        "and the frame doesn't hold MAX_RAW bytes"
    );
}

#[test]
fn decompress_checks_the_length() {
    let packed = compress(b"abc").unwrap();
    assert_eq!(decompress(&packed, 3).unwrap(), b"abc", "exact");
    assert!(decompress(&packed, 4).is_err(), "shorter than stated");
    assert!(decompress(&packed, 2).is_err(), "longer than stated");
}

#[test]
fn index_entries_must_point_before_their_index() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, ..]) = three(&path);
    let end = u64::try_from(good.len()).unwrap();
    for (what, loc) in [
        ("in the header", Loc { offset: 0, len: 40 }),
        (
            "at the Index itself",
            Loc {
                offset: end,
                len: 40,
            },
        ),
        (
            "running into the Index",
            Loc {
                offset: end - 20,
                len: 40,
            },
        ),
        (
            "overflowing",
            Loc {
                offset: u64::MAX - 8,
                len: 40,
            },
        ),
    ] {
        std::fs::write(&path, &good).unwrap();
        let mut f = ProjectFile::open(&path).unwrap();
        let entries = BTreeMap::from([(key(1), loc)]);
        f.append_raw(
            KIND_INDEX,
            &index_payload(Coverage::Full, None, None, &entries).unwrap(),
        )
        .unwrap();
        drop(f);
        assert!(
            matches!(ProjectFile::open_at(&path, end), Err(StoreError::Damaged { offset }) if offset == end),
            "{what}"
        );
        assert!(
            ProjectFile::open_at(&path, s1).is_ok(),
            "{what}: earlier save points still open"
        );
    }
}
