//! Damaged, hostile and concurrent use of the project file: each test pins a recovery or
//! validation rule.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::os::unix::fs::MetadataExt;

use geom::{FaceId, Level, Vec3};

use super::*;
use crate::fixture::{mesh, meta};
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
    let mut f = ProjectFile::create(path, meta(), mesh()).unwrap();
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
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1), op(1), &tile(1.0)).unwrap();
    let parent = f.save().unwrap();
    let offset = f.len();
    f.append_raw(KIND_TILE, &payload).unwrap();
    let entries = BTreeMap::from([(
        Name::Tile(key(2)),
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
fn a_huge_stated_size_never_stops_recovery() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1), op(1), &tile(1.0)).unwrap();
    f.save().unwrap();
    let packed = compress(&[0; 16]).unwrap();
    let mut payload = tile_payload(key(2), op(2), b"", 16, &packed).unwrap();
    let at = raw_len_at(0);
    payload[at..at + 4].copy_from_slice(&(1u32 << 30).to_le_bytes());
    f.append_raw(KIND_TILE, &payload).unwrap();
    f.write_displacement(key(3), op(3), &tile(3.0)).unwrap();
    // Before a save point, the record is stepped over; after the last one, it is discarded
    // with the rest of the tail, and the walk over the tail steps over it too.
    let point = f.save().unwrap();
    let len = f.len();
    f.append_raw(KIND_TILE, &payload).unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.save_points().unwrap().last(),
        Some(&point),
        "recovery reaches the save point after the record"
    );
    assert_eq!(
        opened.keys().copied().collect::<Vec<_>>(),
        vec![key(1), key(3)],
        "skipped, and the tiles before and after it are read"
    );
    assert_eq!(opened.len(), len, "the record after the save point cut off");
}

#[test]
fn bodies_over_the_cap_are_not_written() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    let (header, raw) = encode_displacement(&tile(2.0)).unwrap();
    // A good header and frame, but a stated size the side doesn't need.
    let mut payload =
        tile_payload(key(1), op(1), &header, raw.len(), &compress(&raw).unwrap()).unwrap();
    let at = raw_len_at(header.len());
    payload[at..at + 4].copy_from_slice(&MAX_RAW.to_le_bytes());
    let loc = f.append(KIND_TILE, &payload).unwrap();
    let offset = loc.offset;
    f.live.insert(Name::Tile(key(1)), loc);
    f.pending.insert(Name::Tile(key(1)), loc);
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
        let entries = BTreeMap::from([(Name::Tile(key(1)), loc)]);
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

/// A closed file is unlocked even while a duplicate of its descriptor lives on, as one does in a
/// child this process is spawning until the child's `exec`: reopening it doesn't find it locked.
#[test]
fn a_closed_file_is_unlocked_though_its_descriptor_was_duplicated() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (_, [s1, ..]) = three(&path);
    let writer = ProjectFile::open(&path).unwrap();
    // Shares the open file description, and with it the lock, as a forked child's copy does.
    let duplicate = writer.file.try_clone().unwrap();
    drop(writer);
    drop(ProjectFile::open(&path).unwrap());
    let reader = ProjectFile::open_at(&path, s1).unwrap();
    let shared = reader.file.try_clone().unwrap();
    drop(reader);
    drop(ProjectFile::open(&path).unwrap());
    drop((duplicate, shared));
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
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
    let (_, [_, s2, s3]) = three(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    let after_s2 = s2 + u64::try_from(HEAD + TAIL).unwrap() + f.index_len(s2).unwrap();
    f.write_displacement(key(9), op(50), &tile(2.0)).unwrap();
    drop(f);
    let mut bad = std::fs::read(&path).unwrap();
    bad[usize::try_from(s3).unwrap() + HEAD + 2] ^= 1;
    std::fs::write(&path, &bad).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        state(&opened),
        [(key(1), op(1)), (key(2), op(2))],
        "S2's state: the tile S3 named came after S2, and is discarded with it"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        after_s2,
        "no save point follows S2: S3's tile, S3 and the unsaved tile are a torn tail"
    );
}

#[test]
fn full_indexes_are_the_first_and_every_sixteenth() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
            index_payload(
                Coverage::Delta,
                Some(ProjectFile::fixture_start()),
                None,
                &BTreeMap::new(),
            )
            .unwrap(),
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
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        let start = i64::try_from(f.len()).unwrap();
        f.append_raw(9, &[]).unwrap(); // 12 bytes at the first record
        let target = start + (1i64 << 20) + d;
        let filler = usize::try_from(target - start - 12 - 12).unwrap();
        f.append_raw(9, &vec![0x77; filler]).unwrap();
        let point = f.save().unwrap();
        assert_eq!(i64::try_from(point).unwrap(), target, "d {d}: placed");
        drop(f);
        let mut bad = std::fs::read(&path).unwrap();
        let first = usize::try_from(start).unwrap();
        bad[first + 1] = 1; // a reserved byte of the first record's head
        std::fs::write(&path, &bad).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(
                opened,
                Err(StoreError::DamagedMidFile { offset }) if i64::try_from(offset).unwrap() == start
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
    let theirs = f.live[&Name::Tile(key(2))];
    let entries = BTreeMap::from([(Name::Tile(key(1)), theirs)]);
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
fn an_unknown_record_kind_after_the_last_save_point_is_discarded() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    f.append_raw(9, b"from a newer version").unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        state(&opened),
        [(key(1), op(1)), (key(2), op(2)), (key(3), op(3))],
        "the last save point's state"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        good,
        "nothing after a save point survives an open, whatever its kind"
    );
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
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
    assert_eq!(f.len(), ProjectFile::fixture_start(), "nothing written");
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
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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

#[test]
fn a_writer_that_opened_before_an_in_place_compaction_writes_to_the_new_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    three(&path);
    // As `open` does: open, then lock. The compaction runs in between.
    let early = open_writable(&path).unwrap();
    let old_inode = early.metadata().unwrap().ino();
    compact_in_place(&path).unwrap();
    let file = lock_at(&path, early, open_writable, Lock::Exclusive).unwrap();
    assert_ne!(
        file.metadata().unwrap().ino(),
        old_inode,
        "the replaced file was let go"
    );
    let mut f = ProjectFile::open_latest(file, Repair::Truncate, None).unwrap();
    f.write_displacement(key(9), op(9), &tile(9.0)).unwrap();
    f.save().unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        state(&opened),
        [
            (key(1), op(1)),
            (key(2), op(2)),
            (key(3), op(3)),
            (key(9), op(9))
        ],
        "the write landed in the file at the path"
    );
}

#[test]
fn in_place_compactions_cannot_interleave() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, ..]) = three(&path);
    // What a running in-place compaction holds on its source.
    let source = open_locked(&path, open_readable, Lock::Exclusive).unwrap();
    assert!(
        matches!(compact_in_place(&path), Err(StoreError::Locked)),
        "a second in-place compaction"
    );
    assert!(
        matches!(compact(&path, &dir.path("c.bcrf")), Err(StoreError::Locked)),
        "a compaction to another file"
    );
    assert!(
        matches!(ProjectFile::open(&path), Err(StoreError::Locked)),
        "a writer"
    );
    assert!(
        matches!(ProjectFile::open_at(&path, s1), Err(StoreError::Locked)),
        "a reader"
    );
    drop(source);
    let reader = ProjectFile::open_at(&path, s1).unwrap();
    assert!(
        matches!(compact_in_place(&path), Err(StoreError::Locked)),
        "not while a reader is open either"
    );
    drop(reader);
    assert_eq!(
        names(dir.path("").as_path()),
        ["p.bcrf"],
        "no temporary file left"
    );
    assert_eq!(std::fs::read(&path).unwrap(), good, "untouched");
}

#[test]
fn compacting_in_place_keeps_permissions_and_symlinks() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    three(&path);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let link = dir.path("link.bcrf");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    compact_in_place(&link).unwrap();
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the link is still a link"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600,
        "the project's mode"
    );
    assert_eq!(
        ProjectFile::open(&path)
            .unwrap()
            .save_points()
            .unwrap()
            .len(),
        1,
        "the project itself was compacted"
    );
    assert_eq!(
        names(dir.path("").as_path()),
        ["link.bcrf", "p.bcrf"],
        "no temporary file left"
    );
}

#[test]
fn an_empty_first_save_is_a_parent() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    let first = f.save().unwrap();
    assert_eq!(first, ProjectFile::fixture_start(), "the first record");
    f.write_displacement(key(1), op(1), &tile(1.0)).unwrap();
    let second = f.save().unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.save_points().unwrap(),
        [first, second],
        "the chain reaches the first record"
    );
    assert_eq!(state(&opened), [(key(1), op(1))], "the latest state");
    drop(opened);
    assert_eq!(
        ProjectFile::open_at(&path, first).unwrap().keys().count(),
        0,
        "the empty save"
    );
}

/// A file's header and leading records with `meta`, and a `Mesh` record holding `raw` (canonical
/// mesh bytes, compressed) under a stated size of `stated`: for files whose leading records lie.
fn leading(meta: &Meta, raw: &[u8], stated: u32) -> Vec<u8> {
    let header = leading_bytes(&crate::fixture::meta(), &[]).unwrap();
    let mut payload = stated.to_le_bytes().to_vec();
    payload.extend_from_slice(&compress(raw).unwrap());
    [
        &header[..HEADER_LEN],
        &frame(KIND_META, &meta.to_payload()).unwrap(),
        &frame(KIND_MESH, &payload).unwrap(),
    ]
    .concat()
}

/// The length of `raw` as a mesh record's stated size.
fn stated(raw: &[u8]) -> u32 {
    u32::try_from(raw.len()).unwrap()
}

#[test]
fn a_mesh_that_does_not_match_its_hash_is_rejected() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, [s1, ..]) = three(&path);
    let start = usize::try_from(ProjectFile::fixture_start()).unwrap();
    let raw = mesh().to_bytes();
    let lying = meta().with_mesh_hash(meta().mesh_hash() ^ 1);
    let bad = [
        leading(&lying, &raw, stated(&raw)).as_slice(),
        &good[start..],
    ]
    .concat();
    assert_eq!(bad.len(), good.len(), "the same layout");
    std::fs::write(&path, &bad).unwrap();
    let latest = ProjectFile::open(&path);
    assert!(
        matches!(latest, Err(StoreError::MeshMismatch)),
        "open: {latest:?}"
    );
    let saved = ProjectFile::open_at(&path, s1);
    assert!(
        matches!(saved, Err(StoreError::MeshMismatch)),
        "open_at: {saved:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bad, "untouched");
}

#[test]
fn a_mesh_record_that_is_no_valid_mesh_is_rejected() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut nan = mesh().to_bytes();
    nan[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    let honest = meta().with_mesh_hash(mesh_hash(&nan));
    std::fs::write(&path, leading(&honest, &nan, stated(&nan))).unwrap();
    let invalid = ProjectFile::open(&path);
    assert!(
        matches!(invalid, Err(StoreError::BadMesh(MeshError::NotFinite(_)))),
        "{invalid:?}"
    );
    // A stated size over the cap is refused before decompressing.
    let raw = mesh().to_bytes();
    for size in [MAX_MESH_RAW + 1, u32::MAX] {
        std::fs::write(&path, leading(&meta(), &raw, size)).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(opened, Err(StoreError::BadHeader)),
            "stated {size}: {opened:?}"
        );
    }
}

#[test]
fn a_file_without_its_meta_or_mesh_record_is_rejected() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (good, _) = three(&path);
    let start = usize::try_from(ProjectFile::fixture_start()).unwrap();
    let header = &good[..HEADER_LEN];
    let meta_end = HEADER_LEN + HEAD + meta().to_payload().len() + TAIL;
    let (meta_record, mesh_record) = (&good[HEADER_LEN..meta_end], &good[meta_end..start]);
    let records = &good[start..];
    for (what, bytes) in [
        ("a header alone", header.to_vec()),
        ("no mesh", [header, meta_record].concat()),
        (
            "no mesh, then tiles",
            [header, meta_record, records].concat(),
        ),
        ("no meta", [header, mesh_record, records].concat()),
        ("neither", [header, records].concat()),
        (
            "in the wrong order",
            [header, mesh_record, meta_record, records].concat(),
        ),
    ] {
        std::fs::write(&path, &bytes).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(opened, Err(StoreError::MissingMeta)),
            "{what}: {opened:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "{what}: untouched");
    }
}

#[test]
fn damage_in_the_leading_records_fails_open_and_is_never_cut_off() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let (_, [s1, ..]) = three(&path);
    // An unsaved tile after the last save: a torn tail if recovery ever began at the damage.
    let mut f = ProjectFile::open(&path).unwrap();
    f.write_displacement(key(9), op(50), &tile(2.0)).unwrap();
    drop(f);
    let good = std::fs::read(&path).unwrap();
    std::fs::write(&path, &good).unwrap();
    assert!(ProjectFile::open(&path).is_ok(), "undamaged, it opens");
    let start = usize::try_from(ProjectFile::fixture_start()).unwrap();
    for byte in HEADER_LEN..start {
        for bit in [0, 7] {
            let mut bad = good.clone();
            bad[byte] ^= 1 << bit;
            std::fs::write(&path, &bad).unwrap();
            let opened = ProjectFile::open(&path);
            assert!(
                matches!(
                    opened,
                    Err(StoreError::BadHeader | StoreError::MissingMeta | StoreError::MeshMismatch)
                ),
                "byte {byte} bit {bit}: {opened:?}"
            );
            assert!(
                ProjectFile::open_at(&path, s1).is_err(),
                "byte {byte} bit {bit}: open_at"
            );
            assert_eq!(
                std::fs::read(&path).unwrap(),
                bad,
                "byte {byte} bit {bit}: untouched"
            );
        }
    }
    // A file cut inside its leading records (it was never fully created) is left as it is too.
    for cut in HEADER_LEN..start {
        std::fs::write(&path, &good[..cut]).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(opened, Err(StoreError::BadHeader | StoreError::MissingMeta)),
            "cut at {cut}: {opened:?}"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            good[..cut],
            "cut at {cut}: untouched"
        );
    }
}
