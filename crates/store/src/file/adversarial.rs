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

/// Every live tile and the op that wrote it.
fn state(f: &ProjectFile) -> Vec<(TileKey, OpId)> {
    f.keys()
        .map(|k| (*k, f.read_tile(k).unwrap().unwrap().op))
        .collect()
}

/// The names in `dir`'s directory, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

#[test]
fn one_writer_or_many_readers() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, ..]) = three(&path);
    let writer = ProjectFile::open(&path).unwrap();
    assert!(
        matches!(ProjectFile::open(&path), Err(StoreError::Locked)),
        "a second writer"
    );
    assert!(
        matches!(ProjectFile::open_at(&path, s1), Err(StoreError::Locked)),
        "a reader while a writer is open"
    );
    assert!(
        matches!(compact(&path, &dir.path("c.bcrf")), Err(StoreError::Locked)),
        "compaction while a writer is open"
    );
    assert!(
        !dir.path("c.bcrf").exists(),
        "nothing written when the source is locked"
    );
    drop(writer);
    let a = ProjectFile::open_at(&path, s1).unwrap();
    let b = ProjectFile::open_at(&path, s1).unwrap();
    assert!(
        matches!(ProjectFile::open(&path), Err(StoreError::Locked)),
        "a writer while readers are open"
    );
    drop((a, b));
    drop(ProjectFile::open(&path).unwrap());
    assert_eq!(std::fs::read(&path).unwrap(), good, "never modified");
}

#[test]
fn a_file_that_cannot_be_written_opens_read_only_if_its_tail_is_clean() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    let want = state(&ProjectFile::open(&path).unwrap());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    if OpenOptions::new().write(true).open(&path).is_ok() {
        // Running with the power to ignore permissions (root): nothing to test.
        return;
    }
    let mut f = ProjectFile::open(&path).unwrap();
    assert_eq!(state(&f), want, "the latest state");
    assert!(
        matches!(
            f.write_displacement(key(9), op(9), &tile(9.0)),
            Err(StoreError::ReadOnly)
        ),
        "no writes"
    );
    assert!(matches!(f.save(), Err(StoreError::ReadOnly)), "no saves");
    drop(f);
    let torn = [good.as_slice(), &[KIND_TILE, 0, 0, 0, 100, 0, 0, 0]].concat();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::write(&path, &torn).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    let end = u64::try_from(good.len()).unwrap();
    assert!(
        matches!(ProjectFile::open(&path), Err(StoreError::NeedsRepair { offset }) if offset == end),
        "a torn tail it can't cut off"
    );
    assert_eq!(std::fs::read(&path).unwrap(), torn, "left untouched");
}

#[test]
fn compacting_in_place_replaces_the_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    for seq in 1..=40 {
        f.write_displacement(key(u32::try_from(seq % 3).unwrap()), op(seq), &tile(1.0))
            .unwrap();
        f.save().unwrap();
    }
    let want = state(&f);
    let big = f.len();
    assert!(
        matches!(compact_in_place(&path), Err(StoreError::Locked)),
        "not while open for writing"
    );
    assert_eq!(
        names(dir.path("").as_path()),
        ["p.bcrf"],
        "no temporary file left"
    );
    drop(f);
    compact_in_place(&path).unwrap();
    assert_eq!(
        names(dir.path("").as_path()),
        ["p.bcrf"],
        "renamed over the original"
    );
    let c = ProjectFile::open(&path).unwrap();
    assert_eq!(state(&c), want, "same latest state");
    assert!(c.len() < big / 10, "only live records kept");
    assert_eq!(c.save_points().unwrap().len(), 1, "one full index");
}

#[test]
fn a_damaged_middle_save_point_fails_open_and_earlier_ones_still_open() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, s2, s3]) = three(&path);
    let mut bad = good;
    // S2's parent link: S2 no longer verifies, so S3's chain doesn't load.
    bad[usize::try_from(s2).unwrap() + HEAD + 8] ^= 0x10;
    std::fs::write(&path, &bad).unwrap();
    let opened = ProjectFile::open(&path);
    assert!(
        matches!(opened, Err(StoreError::DamagedMidFile { offset }) if offset == s2),
        "S3 still follows: {opened:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bad, "file untouched");
    let at_s1 = ProjectFile::open_at(&path, s1).unwrap();
    assert_eq!(state(&at_s1), [(key(1), op(1))], "S1's state");
    drop(at_s1);
    assert!(
        matches!(ProjectFile::open_at(&path, s2), Err(StoreError::NotASavePoint(o)) if o == s2),
        "S2 is damaged"
    );
    assert!(
        matches!(ProjectFile::open_at(&path, s3), Err(StoreError::Damaged { offset }) if offset == s3),
        "S3's chain runs through S2"
    );
}

#[test]
fn a_damaged_last_save_point_is_cut_off_with_the_tail() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (_, [_, _, s3]) = three(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    f.write_displacement(key(9), op(50), &tile(2.0)).unwrap();
    drop(f);
    let mut bad = std::fs::read(&path).unwrap();
    bad[usize::try_from(s3).unwrap() + HEAD + 2] ^= 1;
    std::fs::write(&path, &bad).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        state(&opened),
        [(key(1), op(1)), (key(2), op(2)), (key(3), op(3))],
        "S2 plus the tile S3 named, replayed"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        s3,
        "no save point follows S3, so it and the unsaved tile are a torn tail"
    );
}

#[test]
fn full_indexes_are_the_first_and_every_sixteenth() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    let points: Vec<u64> = (1..=40)
        .map(|seq| {
            f.write_displacement(key(u32::try_from(seq % 5).unwrap()), op(seq), &tile(1.0))
                .unwrap();
            f.save().unwrap()
        })
        .collect();
    drop(f);
    let bytes = std::fs::read(&path).unwrap();
    for (n, &point) in points.iter().enumerate() {
        let coverage = bytes[usize::try_from(point).unwrap() + HEAD];
        assert_eq!(coverage, u8::from(n % 16 == 0), "save {n}");
    }
}

#[test]
fn a_checksummed_index_that_does_not_load_stops_recovery_as_damage() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    let stop = u64::try_from(good.len()).unwrap();
    // Each verifies; none loads.
    let mut newer = index_payload(Coverage::Full, None, None, &BTreeMap::new()).unwrap();
    newer[0] = 2; // a coverage this version doesn't know
    let cases = [
        (
            "a delta whose parent is a tile",
            index_payload(Coverage::Delta, Some(FIRST_RECORD), None, &BTreeMap::new()).unwrap(),
        ),
        (
            "a delta with no parent",
            index_payload(Coverage::Delta, None, None, &BTreeMap::new()).unwrap(),
        ),
        ("an unknown coverage", newer),
        ("an empty payload", Vec::new()),
    ];
    for (what, payload) in cases {
        std::fs::write(&path, &good).unwrap();
        let mut f = ProjectFile::open(&path).unwrap();
        f.append_raw(KIND_INDEX, &payload).unwrap();
        f.write_displacement(key(9), op(50), &tile(2.0)).unwrap();
        drop(f);
        let bad = std::fs::read(&path).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(opened, Err(StoreError::DamagedMidFile { offset }) if offset == stop),
            "{what}: {opened:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bad, "{what}: file untouched");
    }
}

#[test]
fn a_save_point_across_a_search_window_boundary_is_found() {
    for d in -12i64..=4 {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path).unwrap();
        f.append_raw(9, &[]).unwrap(); // 12 bytes at FIRST_RECORD
        let target = 16 + (1i64 << 20) + d;
        let filler = usize::try_from(target - 16 - 12 - 12).unwrap();
        f.append_raw(9, &vec![0x77; filler]).unwrap();
        let point = f.save().unwrap();
        assert_eq!(i64::try_from(point).unwrap(), target, "d {d}: placed");
        drop(f);
        let mut bad = std::fs::read(&path).unwrap();
        bad[17] = 1; // a reserved byte of the first record's head
        std::fs::write(&path, &bad).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(
                opened,
                Err(StoreError::DamagedMidFile {
                    offset: FIRST_RECORD
                })
            ),
            "d {d}: {opened:?}"
        );
    }
}

#[test]
fn only_a_plausible_index_proves_damage() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    let saved = u64::try_from(good.len()).unwrap();
    // Unsaved: tile A, lost to writeback, then a record carrying a framed Index.
    let a_len = {
        let mut f = ProjectFile::open(&path).unwrap();
        f.write_displacement(key(20), op(60), &tile(3.0)).unwrap();
        f.len() - saved
    };
    let carrier = saved + a_len;
    let fake_at = carrier + u64::try_from(HEAD).unwrap();
    for (what, parent, proof) in [
        ("its own offset", Some(fake_at), false),
        ("past it", Some(fake_at + 100), false),
        ("in the header", Some(8), false),
        // Indistinguishable from a real first save: the file is left as it is.
        ("none", None, true),
    ] {
        let fake = frame(
            KIND_INDEX,
            &index_payload(Coverage::Full, parent, None, &BTreeMap::new()).unwrap(),
        )
        .unwrap();
        std::fs::write(&path, &good).unwrap();
        let mut f = ProjectFile::open(&path).unwrap();
        f.write_displacement(key(20), op(60), &tile(3.0)).unwrap();
        f.append_raw(9, &fake).unwrap();
        drop(f);
        let mut bad = std::fs::read(&path).unwrap();
        bad[usize::try_from(saved).unwrap()..usize::try_from(carrier).unwrap()].fill(0);
        std::fs::write(&path, &bad).unwrap();
        assert_eq!(
            ProjectFile::open_at(&path, fake_at).is_ok(),
            proof,
            "parent {what}: open_at"
        );
        let opened = ProjectFile::open(&path);
        if proof {
            assert!(
                matches!(opened, Err(StoreError::DamagedMidFile { offset }) if offset == saved),
                "parent {what}: {opened:?}"
            );
            assert_eq!(
                std::fs::read(&path).unwrap(),
                bad,
                "parent {what}: untouched"
            );
        } else {
            assert!(opened.is_ok(), "parent {what}: {opened:?}");
            assert_eq!(
                std::fs::read(&path).unwrap(),
                good,
                "parent {what}: no proof, so a torn tail"
            );
        }
    }
}

#[test]
fn an_entry_naming_another_keys_record_is_damage() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (_, [.., s3]) = three(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    let theirs = f.live[&key(2)];
    let entries = BTreeMap::from([(key(1), theirs)]);
    f.append_raw(
        KIND_INDEX,
        &index_payload(Coverage::Delta, Some(s3), Some(op(3)), &entries).unwrap(),
    )
    .unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert!(
        matches!(opened.read_tile(&key(1)), Err(StoreError::Damaged { offset }) if offset == theirs.offset),
        "key 1 names key 2's record"
    );
    assert_eq!(
        opened.read_tile(&key(2)).unwrap().unwrap().op,
        op(2),
        "key 2 reads its own"
    );
}

#[test]
fn an_unknown_record_kind_at_the_very_end_is_kept() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    three(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    f.append_raw(9, b"from a newer version").unwrap();
    drop(f);
    let bytes = std::fs::read(&path).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        state(&opened),
        [(key(1), op(1)), (key(2), op(2)), (key(3), op(3))],
        "skipped"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes, "not cut off");
}

#[test]
fn a_file_ending_inside_a_head_is_cut_back() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    for (what, head) in [
        (
            "a tile head promising more",
            [KIND_TILE, 0, 0, 0, 100, 0, 0, 0],
        ),
        (
            "an Index head with no checksum",
            [KIND_INDEX, 0, 0, 0, 0, 0, 0, 0],
        ),
    ] {
        std::fs::write(&path, [good.as_slice(), &head].concat()).unwrap();
        drop(ProjectFile::open(&path).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), good, "{what}");
    }
}

#[test]
fn unknown_header_flags_or_reserved_bits_are_rejected() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, ..]) = three(&path);
    for byte in [6, 7, 8, 15] {
        let mut bad = good.clone();
        bad[byte] = 1;
        std::fs::write(&path, &bad).unwrap();
        assert!(
            matches!(ProjectFile::open(&path), Err(StoreError::BadHeader)),
            "open, byte {byte}"
        );
        assert!(
            matches!(ProjectFile::open_at(&path, s1), Err(StoreError::BadHeader)),
            "open_at, byte {byte}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bad, "byte {byte}: untouched");
    }
}

#[test]
fn displacement_tiles_must_fit_their_keys() {
    use geom::{Channel, LayerId};

    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path).unwrap();
    let other = TileKey::new(
        LayerId::new(0),
        Channel::from_raw(1),
        FaceId::new(1),
        Level::new(9).unwrap(),
        0,
        0,
    )
    .unwrap();
    assert!(
        matches!(
            f.write_displacement(other, op(1), &tile(1.0)),
            Err(StoreError::NotDisplacement(1))
        ),
        "write under another channel"
    );
    let small = CorrectionTile::zero(2).unwrap();
    assert!(
        matches!(
            f.write_displacement(key(1), op(1), &small),
            Err(StoreError::WrongSide { key: 65, tile: 2 })
        ),
        "write a side-2 tile under a level-9 key"
    );
    assert_eq!(f.len(), FIRST_RECORD, "nothing written");
    // Stored anyway, as raw bytes: caught on read, before decompressing.
    let (header, body) = encode_displacement(&small).unwrap();
    f.write_tile(key(1), op(1), &header, &body).unwrap();
    f.write_tile(other, op(2), &header, &body).unwrap();
    assert!(
        matches!(
            f.read_displacement(&key(1)),
            Err(StoreError::WrongSide { key: 65, tile: 2 })
        ),
        "read a side-2 tile under a level-9 key"
    );
    assert!(
        matches!(
            f.read_displacement(&other),
            Err(StoreError::NotDisplacement(1))
        ),
        "read under another channel"
    );
    assert_eq!(
        f.read_tile(&other).unwrap().unwrap().body,
        body,
        "the raw bytes are still there"
    );
}

#[test]
fn keys_differing_only_in_layer_or_channel_stay_apart() {
    use geom::{Channel, LayerId};

    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let at = |layer: u16, channel: u8| {
        TileKey::new(
            LayerId::new(layer),
            Channel::from_raw(channel),
            FaceId::new(3),
            Level::new(9).unwrap(),
            1,
            2,
        )
        .unwrap()
    };
    let keys = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)];
    assert_eq!(
        TileKey::displacement(FaceId::new(3), Level::new(9).unwrap(), 1, 2).unwrap(),
        keys[0],
        "layer 0, channel 0 is the base displacement key"
    );
    let mut f = ProjectFile::create(&path).unwrap();
    for (n, k) in (0u8..).zip(&keys) {
        f.write_tile(*k, op(u64::from(n) + 1), &[n], b"x").unwrap();
    }
    f.save().unwrap();
    drop(f);
    let compacted = dir.path("c.bcrf");
    compact(&path, &compacted).unwrap();
    for p in [&path, &compacted] {
        let opened = ProjectFile::open(p).unwrap();
        for (n, k) in (0u8..).zip(&keys) {
            assert_eq!(
                opened.read_tile(k).unwrap().unwrap().header,
                [n],
                "{p:?}: key {n}"
            );
        }
    }
}
