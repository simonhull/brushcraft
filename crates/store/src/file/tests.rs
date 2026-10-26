//! Tests of the project file: its format, locking, reads, saves and compaction.

use std::path::Path;

use geom::{Channel, FaceId, LayerId, Level, Vec3};

use super::format::{ENTRY_LEN, parse_index_payload};
use super::*;
use crate::bounds;
use crate::bytes::KEY_LEN;
use crate::fixture::{self, mesh, meta};
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

/// The next number of a `SplitMix64` sequence at `state`, so failures replay.
fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Writes `count` random records to `f`, tiles and bounds of a few dozen keys, each with the
/// stamp after `stamp`'s (which it moves on) or, with `at`, that stamp; returns their names.
fn random_writes(
    f: &mut ProjectFile,
    rng: &mut u64,
    stamp: &mut Stamp,
    at: Option<Stamp>,
    count: u64,
) -> Vec<Name> {
    (0..count)
        .map(|_| {
            let r = next_random(rng);
            let face = u32::try_from(r % 6).unwrap();
            let tx = u16::try_from((r >> 8) % 8).unwrap();
            let k = key(face, tx);
            let write = at.unwrap_or_else(|| {
                *stamp = stamp.next();
                *stamp
            });
            #[expect(
                clippy::cast_precision_loss,
                reason = "a small seed for a tile's values"
            )]
            let seed = ((r >> 16) % 1000) as f32 / 1000.0;
            if (r >> 32).is_multiple_of(3) {
                let known = !(r >> 40).is_multiple_of(4);
                f.write_bounds(k, known.then(|| bound(seed)).as_ref(), write)
                    .unwrap();
                Name::Bounds(k)
            } else {
                f.write_displacement_stamped(k, op(r >> 48), write, &tile(seed))
                    .unwrap();
                Name::Tile(k)
            }
        })
        .collect()
}

/// Every save point derived through cuts and full Indexes is its chain read back, bit for
/// bit, and extends the one before.
///
/// Rounds alternate at random between a plain save and a recovery flush's save point at a
/// cut: records written past the cut, then some written at it after them (the flush's own
/// copies), so the save point names the cut's state and the records past it wait. Every
/// full Index is written at a cut with records past it, so the derivation of a full Index
/// from the cut's delta is covered, not only the delta's.
#[test]
fn derived_save_points_are_their_chains_through_random_cuts_and_full_indexes() {
    for seed in 1..=6_u64 {
        let dir = TestDir::new();
        let path = dir.path("p.bcrf");
        let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
        let mut rng = seed;
        let mut stamp = Stamp::ZERO;
        let mut previous: Option<BaseFile> = None;
        let mut full_at_cut = 0;
        for round in 0..50 {
            let what = format!("seed {seed}, round {round}");
            let before = 1 + next_random(&mut rng) % 6;
            random_writes(&mut f, &mut rng, &mut stamp, None, before);
            let full = f.last_index().is_none() || f.deltas + 1 >= FULL_EVERY;
            if full || next_random(&mut rng).is_multiple_of(2) {
                let at = stamp;
                f.open_cut(at);
                let past = 1 + next_random(&mut rng) % 5;
                random_writes(&mut f, &mut rng, &mut stamp, None, past);
                let copies = next_random(&mut rng) % 4;
                random_writes(&mut f, &mut rng, &mut stamp, Some(at), copies);
                f.write_save_point().unwrap();
                full_at_cut += usize::from(full);
            } else {
                f.save().unwrap();
            }
            let base = f.base().unwrap();
            let rebuilt = f.base_at(base.save_point()).unwrap().unwrap();
            assert!(base.same_state(&rebuilt), "{what}: the chain's state");
            if let Some(previous) = &previous {
                assert!(base.extends(previous), "{what}: extends the one before");
            }
            previous = Some(base);
        }
        assert!(
            full_at_cut >= 4,
            "seed {seed}: {full_at_cut} full Indexes at a cut"
        );
        let last = previous.unwrap();
        drop(f);
        let reopened = ProjectFile::open(&path).unwrap();
        assert!(
            reopened.base().unwrap().same_state(&last),
            "seed {seed}: reopened at the last save point"
        );
    }
}

#[test]
fn a_full_index_its_derived_maps_disagree_with_is_refused_before_it_is_written() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
    f.save().unwrap();
    // Delta saves up to the one before the next full Index.
    for seq in 2..=FULL_EVERY {
        f.write_displacement(key(1, 0), op(u64::from(seq)), &tile(0.25))
            .unwrap();
        f.save().unwrap();
    }
    let points = f.save_points().unwrap();
    // A new tile, which the skewed derivation leaves out.
    f.write_displacement(key(2, 0), op(100), &tile(0.125))
        .unwrap();
    f.skew_derivation = true;
    let refused = f.save();
    assert!(
        matches!(
            refused,
            Err(FileError::SavePointMismatch {
                named: 2,
                derived: 1
            })
        ),
        "{refused:?}"
    );
    assert_eq!(f.save_points().unwrap(), points, "no Index written");
    // The next save derives right, and is the full Index.
    let at = f.save().unwrap();
    assert_eq!(f.deltas, 0, "a full Index");
    let base = f.base().unwrap();
    assert_eq!(base.save_point().offset, at);
    assert!(base.same_state(&f.base_at(base.save_point()).unwrap().unwrap()));
    drop(f);
    let reopened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        reopened.read_displacement(&key(2, 0)).unwrap(),
        Some((op(100), tile(0.125))),
        "saved by the save after"
    );
}

#[test]
fn a_save_point_extends_only_the_one_before_it_in_the_same_open_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
    f.save().unwrap();
    let first = f.base().unwrap();
    assert!(!first.extends(&first), "the first save point extends none");

    // A byte-for-byte copy holds the same save point, but not as this file made it.
    let copy_path = dir.path("q.bcrf");
    std::fs::copy(&path, &copy_path).unwrap();
    let mut copy = ProjectFile::open(&copy_path).unwrap();
    let loaded = copy.base().unwrap();
    assert_eq!(
        loaded.save_point(),
        first.save_point(),
        "the same save point"
    );
    assert!(!loaded.extends(&first), "loaded from the file: no parent");

    f.write_displacement(key(2, 0), op(2), &tile(0.25)).unwrap();
    f.save().unwrap();
    let second = f.base().unwrap();
    assert!(second.extends(&first), "derived from the first");
    assert!(!first.extends(&second), "not the other way");

    // The copy saves the same records: a save point with the same parent, in another file.
    copy.write_displacement(key(2, 0), op(2), &tile(0.25))
        .unwrap();
    copy.save().unwrap();
    let copied = copy.base().unwrap();
    assert_eq!(
        copied.save_point(),
        second.save_point(),
        "the same save point"
    );
    assert!(!copied.extends(&first), "another open file");

    f.save().unwrap();
    let third = f.base().unwrap();
    assert!(third.extends(&second), "the one before");
    assert!(!third.extends(&first), "only the one before");
    let reloaded = f.base_at(second.save_point()).unwrap().unwrap();
    assert!(third.extends(&reloaded), "the save point before, read back");
    assert!(!reloaded.extends(&first), "read back: no parent");
}

#[test]
fn tiles_survive_save_and_reopen() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
fn records_after_the_last_save_point_are_discarded_on_open() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
    f.save().unwrap();
    let saved = f.len();
    // Whole records, but no save point: an operation that never finished.
    f.write_displacement(key(1, 0), op(2), &tile(0.75)).unwrap();
    f.write_displacement(key(2, 0), op(3), &tile(0.25)).unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.read_displacement(&key(1, 0)).unwrap(),
        Some((op(1), tile(0.5))),
        "the saved tile, not the later one"
    );
    assert!(
        !opened.contains(&key(2, 0)),
        "a tile written after it is gone"
    );
    assert_eq!(opened.head(), Some(op(1)), "the save point's head");
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        saved,
        "cut back to the save point"
    );
}

#[test]
fn delta_indexes_stay_small_and_chain_correctly() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
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
        matches!(ProjectFile::open(&path), Err(FileError::BadHeader)),
        "magic"
    );
}

#[test]
fn a_damaged_old_tile_is_caught_when_read() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(0.5)).unwrap();
    f.write_displacement(key(2, 0), op(2), &tile(0.5)).unwrap();
    f.save().unwrap();
    drop(f);
    let mut bytes = std::fs::read(&path).unwrap();
    let start = usize::try_from(ProjectFile::fixture_start()).unwrap();
    bytes[start + 40] ^= 0xFF; // inside the first tile's payload
    std::fs::write(&path, &bytes).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert!(
        matches!(
            opened.read_displacement(&key(1, 0)),
            Err(FileError::Damaged { .. })
        ),
        "checksum"
    );
    assert!(
        opened.read_displacement(&key(2, 0)).unwrap().is_some(),
        "others still readable"
    );
}
/// Every live tile: its op and content.
type State = BTreeMap<TileKey, (OpId, CorrectionTile)>;

/// Every live bound, as bits (`None` if unknown).
type Bounded = BTreeMap<TileKey, Option<[u32; TileBounds::BITS]>>;

/// Bounds made from `seed`: a box around it and, for odd seeds, the cone of everything.
fn bound(seed: f32) -> TileBounds {
    let at = Vec3::new(seed, -seed, 2.0 * seed);
    let cone = if seed.to_bits() % 2 == 1 {
        crate::index::NormalCone::EVERYTHING
    } else {
        crate::index::NormalCone::from_normals(&[Vec3::new(0.0, 0.0, 1.0)])
    };
    TileBounds {
        aabb: crate::index::Aabb::new(at, at + Vec3::new(1.0, 1.0, 1.0)).unwrap(),
        cone,
    }
}

/// The live bounds of `f`, every record read and checked.
fn bounds_of(f: &ProjectFile) -> Bounded {
    let entries: Vec<(TileKey, Loc)> = f.bounds_locations().map(|(key, loc)| (*key, loc)).collect();
    let read = f.read_bounds(&entries).unwrap();
    entries
        .iter()
        .zip(read)
        .map(|((key, _), bounds)| (*key, bounds.map(TileBounds::to_bits)))
        .collect()
}

/// Writes a project with saves, overwrites and trailing unsaved tiles and bounds; returns
/// the states an open can find.
///
/// Recorded at creation and after every save point: the file length and the saved live
/// state (key → (op, tile)), and the saved bounds, known and unknown, each written before or
/// after the tiles of its round.
fn scripted(path: &Path) -> Vec<(u64, State, Bounded)> {
    let mut f = ProjectFile::create(path, meta(), mesh()).unwrap();
    let (mut state, mut bounds) = (BTreeMap::new(), BTreeMap::new());
    let mut log = vec![(f.len(), state.clone(), bounds.clone())];
    let mut seq = 0;
    for round in 0..6u16 {
        let write_bounds = |file: &mut ProjectFile, saved: &mut Bounded| {
            for k in 0..3u32 {
                let at = key(k + u32::from(round % 3), 0);
                let b = (k != u32::from(round) % 3).then(|| bound(f32::from(round + 1)));
                file.write_bounds(at, b.as_ref(), Stamp::ZERO).unwrap();
                saved.insert(at, b.map(TileBounds::to_bits));
            }
        };
        if round % 2 == 0 {
            write_bounds(&mut f, &mut bounds);
        }
        for k in 0..5u32 {
            seq += 1;
            let t = tile(f32::from(round) + 0.1 * f32::from(u16::try_from(k).unwrap()));
            let at = key(k + u32::from(round % 2), 0);
            f.write_displacement(at, op(seq), &t).unwrap();
            state.insert(at, (op(seq), t));
        }
        if round % 2 == 1 {
            write_bounds(&mut f, &mut bounds);
        }
        if round < 5 {
            f.save().unwrap(); // the last round stays unsaved
            log.push((f.len(), state.clone(), bounds.clone()));
        }
    }
    log
}

fn state_of(f: &ProjectFile) -> State {
    f.keys()
        .map(|k| (*k, f.read_displacement(k).unwrap().unwrap()))
        .collect()
}

#[test]
fn truncation_anywhere_opens_at_the_last_complete_save_point() {
    let dir = TestDir::new();
    let full = dir.path("full.bcrf");
    let log = scripted(&full);
    let bytes = std::fs::read(&full).unwrap();
    let mut cuts: Vec<u64> = log.iter().map(|(len, _, _)| *len).collect();
    // Every record boundary too: a crash between any two records.
    cuts.extend(ProjectFile::record_ends(&full));
    let mut x = 0x9E37_79B9_u64;
    let start = ProjectFile::fixture_start();
    for _ in 0..60 {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        cuts.push(start + x % (bytes.len() as u64 - start));
    }
    for cut in cuts {
        let (good_len, want, want_bounds) =
            log.iter().rev().find(|(len, _, _)| *len <= cut).unwrap();
        let kept = &bytes[..usize::try_from(cut).unwrap()];
        // A crash can also leave the file at its full length with the unwritten blocks zeroed.
        let zero_filled = [kept, &vec![0; bytes.len() - kept.len()]].concat();
        for (variant, contents) in [("cut", kept.to_vec()), ("zero-filled", zero_filled)] {
            let path = dir.path("cut.bcrf");
            std::fs::write(&path, &contents).unwrap();
            let mut f = ProjectFile::open(&path).unwrap();
            assert_eq!(&state_of(&f), want, "{variant} at {cut}");
            assert_eq!(&bounds_of(&f), want_bounds, "{variant} at {cut}: bounds");
            // The torn tail is gone and the file is writable again.
            assert_eq!(
                std::fs::metadata(&path).unwrap().len(),
                *good_len,
                "tail cut off: {variant} at {cut}"
            );
            f.write_displacement(key(99, 0), op(10_000), &tile(9.0))
                .unwrap();
            f.save().unwrap();
            drop(f);
            let reopened = ProjectFile::open(&path).unwrap();
            let mut want = want.clone();
            want.insert(key(99, 0), (op(10_000), tile(9.0)));
            assert_eq!(
                state_of(&reopened),
                want,
                "written after the {variant} at {cut}"
            );
            assert_eq!(
                &bounds_of(&reopened),
                want_bounds,
                "bounds after the {variant} at {cut}"
            );
        }
    }
}

#[test]
fn every_save_point_opens_to_its_own_state() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    let mut snapshots = Vec::new();
    for round in 0..40u16 {
        f.write_displacement(
            key(u32::from(round % 7), 0),
            op(u64::from(round) + 1),
            &tile(f32::from(round)),
        )
        .unwrap();
        let point = f.save().unwrap();
        snapshots.push((point, state_of(&f)));
    }
    assert_eq!(
        f.save_points().unwrap(),
        snapshots.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        "chain"
    );
    drop(f); // save points are opened with a shared lock
    for (point, want) in &snapshots {
        let mut old = ProjectFile::open_at(&path, *point).unwrap();
        assert_eq!(&state_of(&old), want, "save point {point}");
        assert!(
            matches!(
                old.write_displacement(key(0, 0), op(1), &tile(0.0)),
                Err(FileError::ReadOnly)
            ),
            "read-only"
        );
    }
    assert!(
        matches!(
            ProjectFile::open_at(&path, 17),
            Err(FileError::NotASavePoint(17))
        ),
        "not an index"
    );
}

#[test]
fn unknown_record_kinds_are_skipped() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    f.write_bounds(key(1, 0), Some(&bound(1.0)), Stamp::ZERO)
        .unwrap();
    f.save().unwrap();
    f.append_raw(9, b"from a newer version").unwrap();
    f.write_displacement(key(2, 0), op(2), &tile(2.0)).unwrap();
    f.append_raw(200, &[0; 64]).unwrap();
    f.write_bounds(key(2, 0), None, Stamp::ZERO).unwrap();
    f.write_displacement(key(3, 0), op(3), &tile(3.0)).unwrap();
    f.save().unwrap();
    let len = f.len();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.keys().count(),
        3,
        "all three tiles, unknown records skipped"
    );
    assert_eq!(
        bounds_of(&opened),
        BTreeMap::from([(key(1, 0), Some(bound(1.0).to_bits())), (key(2, 0), None)]),
        "and both bounds"
    );
    assert_eq!(opened.len(), len, "nothing cut off");
}

/// A bound record cut short by the end of the file reads as unknown, not as an error, and the
/// records read with it in one window still read as written.
///
/// The window over all of them runs past the end, so each record is read again on its own: the
/// whole ones give their bounds, the cut one nothing.
#[test]
fn a_bound_cut_short_by_the_end_of_the_file_reads_as_unknown() {
    let dir = TestDir::new();
    let path = dir.path("cut.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    for k in 0..3u16 {
        f.write_bounds(key(1, k), Some(&bound(f32::from(k + 1))), Stamp::ZERO)
            .unwrap();
    }
    let mut entries: Vec<(TileKey, Loc)> =
        f.bounds_locations().map(|(key, loc)| (*key, loc)).collect();
    entries.sort_by_key(|(_, loc)| loc.offset);
    let &(_, last) = entries.last().unwrap();
    assert_eq!(
        last.offset + last.len,
        f.len(),
        "the set-up: the last record ends the file"
    );
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(last.offset + last.len / 2)
        .unwrap();
    let read = f.read_bounds(&entries).unwrap();
    let want: Vec<Option<[u32; TileBounds::BITS]>> = (0..3u16)
        .map(|k| (k < 2).then(|| bound(f32::from(k + 1)).to_bits()))
        .collect();
    assert_eq!(
        read.into_iter()
            .map(|b| b.map(TileBounds::to_bits))
            .collect::<Vec<_>>(),
        want,
        "the whole records read, the cut one is unknown"
    );
    let alone = f.read_bounds(&entries[2..]).unwrap();
    assert_eq!(alone, vec![None], "read on its own, too");
}

#[test]
fn an_index_names_its_bounds_strictly() {
    let at = |offset| Loc { offset, len: 40 };
    let entries = BTreeMap::from([
        (Name::Tile(key(1, 0)), at(100)),
        (Name::Tile(key(2, 0)), at(140)),
        (
            Name::Bounds(key(1, 0)),
            Loc {
                offset: 180,
                len: bounds::KNOWN_RECORD_LEN,
            },
        ),
    ]);
    let good = index_payload(Coverage::Full, None, None, &entries).unwrap();
    let parsed = parse_index_payload(&good).unwrap();
    assert_eq!(
        parsed.entries,
        entries
            .iter()
            .map(|(name, loc)| (*name, Some(*loc)))
            .collect::<Vec<_>>(),
        "both sections round trip"
    );
    let tiles_only: BTreeMap<Name, Loc> = entries
        .iter()
        .filter(|(name, _)| name.tile().is_some())
        .map(|(name, loc)| (*name, *loc))
        .collect();
    let old = index_payload(Coverage::Full, None, None, &tiles_only).unwrap();
    assert_eq!(
        old.len(),
        good.len() - 4 - ENTRY_LEN,
        "no bounds, no section: the layout of an Index before bounds were kept"
    );
    assert!(parse_index_payload(&old).is_some(), "which still parses");
    let empty_section = [old.as_slice(), &0_u32.to_le_bytes()].concat();
    assert!(
        parse_index_payload(&empty_section).is_none(),
        "an empty bounds section"
    );
    assert!(
        parse_index_payload(&good[..good.len() - 1]).is_none(),
        "a short section"
    );
    // The two tile entries swapped: out of order.
    let mut swapped = good;
    let first = 28;
    let (a, b) = swapped[first..first + 2 * ENTRY_LEN].split_at_mut(ENTRY_LEN);
    a.swap_with_slice(b);
    assert!(parse_index_payload(&swapped).is_none(), "keys out of order");
}

/// A project with three saves of five tiles each, and its bytes.
fn saved_three_times(path: &Path) -> Vec<u8> {
    let mut f = ProjectFile::create(path, meta(), mesh()).unwrap();
    for seq in 1..=15u16 {
        f.write_displacement(
            key(u32::from(seq % 5), 0),
            op(u64::from(seq)),
            &tile(f32::from(seq)),
        )
        .unwrap();
        if seq % 5 == 0 {
            f.save().unwrap();
        }
    }
    drop(f);
    std::fs::read(path).unwrap()
}

#[test]
fn damage_before_later_records_fails_open_and_leaves_the_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let good = saved_three_times(&path);
    let start = ProjectFile::fixture_start();
    let first = usize::try_from(start).unwrap();
    // The first record's head: a reserved byte (the head no longer parses), then the low
    // byte of its length (the scan runs off the record boundaries).
    for (what, byte) in [("reserved byte", first + 1), ("length", first + 4)] {
        let mut bad = good.clone();
        bad[byte] ^= 0x01;
        std::fs::write(&path, &bad).unwrap();
        let opened = ProjectFile::open(&path);
        assert!(
            matches!(
                opened,
                Err(FileError::DamagedMidFile { offset }) if offset == start
            ),
            "{what}: {opened:?}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bad, "{what}: file untouched");
    }
}

#[test]
fn a_torn_tail_is_still_cut_off() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let good = saved_three_times(&path);
    let want = state_of(&ProjectFile::open(&path).unwrap());
    // Half of a further tile record, then bytes that never were a record.
    let record = frame(
        KIND_TILE,
        &tile_payload(key(7, 0), op(99), b"h", 3, &[1, 2, 3]).unwrap(),
    )
    .unwrap();
    let torn = [
        good.as_slice(),
        &record[..record.len() / 2],
        &[0xA5; 300][..],
    ]
    .concat();
    std::fs::write(&path, &torn).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(state_of(&opened), want, "the saved state");
    assert_eq!(std::fs::read(&path).unwrap(), good, "the tail is cut off");
}

#[test]
fn unsaved_writes_lost_out_of_order_are_cut_off() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    saved_three_times(&path);
    let mut f = ProjectFile::open(&path).unwrap();
    let want = state_of(&f);
    let saved = f.len();
    // Unsaved, so not synced: writeback may keep B and lose A.
    f.write_displacement(key(7, 0), op(100), &tile(7.0))
        .unwrap();
    let a_end = f.len();
    f.write_displacement(key(8, 0), op(101), &tile(8.0))
        .unwrap();
    drop(f);
    let mut bytes = std::fs::read(&path).unwrap();
    let a = usize::try_from(saved).unwrap()..usize::try_from(a_end).unwrap();
    bytes[a].fill(0);
    std::fs::write(&path, &bytes).unwrap();
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(state_of(&opened), want, "the last save, nothing after");
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        saved,
        "cut off at A, with B after it"
    );
}

#[test]
fn tile_records_that_do_not_parse_are_skipped() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    f.save().unwrap();
    // Before the next save point: a payload too short to hold a key, and a tile whose op is
    // `u64::MAX` (the reserved site, as no real op can be).
    f.append_raw(KIND_TILE, b"short").unwrap();
    let mut reserved = tile_payload(key(2, 0), op(2), b"", 0, &[]).unwrap();
    reserved[KEY_LEN..KEY_LEN + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    f.append_raw(KIND_TILE, &reserved).unwrap();
    f.write_displacement(key(3, 0), op(3), &tile(3.0)).unwrap();
    f.save().unwrap();
    let len = f.len();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(
        opened.keys().copied().collect::<Vec<_>>(),
        vec![key(1, 0), key(3, 0)],
        "recovery goes on past them to the save point"
    );
    assert_eq!(opened.head(), Some(op(3)), "head op");
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        len,
        "nothing cut off"
    );
}

#[test]
fn compaction_leaves_its_source_untouched() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let good = saved_three_times(&path);
    let want = state_of(&ProjectFile::open(&path).unwrap());
    let torn = [good.as_slice(), &[0x5A; 100][..]].concat();
    std::fs::write(&path, &torn).unwrap();
    let out = dir.path("c.bcrf");
    compact(&path, &out).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        torn,
        "source bytes unchanged"
    );
    assert_eq!(
        state_of(&ProjectFile::open(&out).unwrap()),
        want,
        "the recovered state"
    );
}

#[test]
fn other_layers_and_channels_round_trip_through_save_open_and_compaction() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let odd = TileKey::new(
        LayerId::new(3),
        Channel::from_raw(7),
        FaceId::new(5),
        Level::new(4).unwrap(),
        0,
        0,
    )
    .unwrap();
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_tile(odd, op(1), b"future header", b"future body")
        .unwrap();
    f.write_displacement(key(1, 0), op(2), &tile(1.0)).unwrap();
    f.save().unwrap();
    drop(f);
    let compacted = dir.path("c.bcrf");
    compact(&path, &compacted).unwrap();
    for p in [&path, &compacted] {
        let opened = ProjectFile::open(p).unwrap();
        let t = opened.read_tile(&odd).unwrap().unwrap();
        assert_eq!(
            (t.op, t.header.as_slice(), t.body.as_slice()),
            (op(1), &b"future header"[..], &b"future body"[..]),
            "{p:?}"
        );
        assert_eq!(opened.head(), Some(op(2)), "head op {p:?}");
    }
}

#[test]
fn compaction_keeps_the_latest_state_and_shrinks_the_file() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    for seq in 1..=100u16 {
        f.write_displacement(
            key(u32::from(seq % 4), 0),
            op(u64::from(seq)),
            &tile(f32::from(seq)),
        )
        .unwrap();
        f.save().unwrap();
    }
    let before = state_of(&f);
    let big = f.len();
    drop(f);
    let out = dir.path("c.bcrf");
    compact(&path, &out).unwrap();
    let c = ProjectFile::open(&out).unwrap();
    assert_eq!(state_of(&c), before, "same latest state");
    assert!(
        c.len() < big / 10,
        "only live records kept: {} vs {big}",
        c.len()
    );
    assert_eq!(c.save_points().unwrap().len(), 1, "one full index");
}

fn small_key(face: u32) -> TileKey {
    TileKey::displacement(FaceId::new(face), Level::new(3).unwrap(), 0, 0).unwrap()
}

fn small_tile(seed: f32) -> CorrectionTile {
    let values: Vec<Vec3> = (0..9 * 9u16)
        .map(|k| Vec3::new(seed * f32::from(k % 5), -seed, 0.25))
        .collect();
    CorrectionTile::from_values(9, &values).unwrap()
}

#[test]
fn a_reader_on_another_thread_reads_what_the_owner_does() {
    let dir = TestDir::new();
    let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
    let tiles = [(key(1, 0), tile(0.5)), (small_key(2), small_tile(0.125))];
    for (seq, (k, t)) in (1..).zip(&tiles) {
        f.write_displacement(*k, op(seq), t).unwrap();
    }
    f.save().unwrap();
    let jobs: Vec<(TileKey, Loc)> = tiles
        .iter()
        .map(|(k, _)| (*k, f.locate(k).unwrap()))
        .collect();
    let reader = f.reader();
    let read = std::thread::spawn(move || {
        let mut scratch = ReadScratch::new().unwrap();
        let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
        jobs.iter()
            .map(|(k, loc)| {
                let op = reader
                    .read_displacement_into(k, *loc, &mut slot, &mut scratch)
                    .unwrap();
                (op, slot.to_tile().unwrap())
            })
            .collect::<Vec<_>>()
    })
    .join()
    .unwrap();
    for ((k, t), got) in tiles.iter().zip(read) {
        let owner = f.read_displacement(k).unwrap().unwrap();
        assert_eq!(got, owner, "{k:?}: reader and owner agree");
        assert_eq!(&got.1, t, "{k:?}: the tile written");
    }
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
    let mut scratch = ReadScratch::new().unwrap();
    slot.reset(9).unwrap();
    assert_eq!(
        f.read_displacement_into(&small_key(3), &mut slot, &mut scratch)
            .unwrap(),
        None,
        "absent"
    );
    assert_eq!(slot.side, 9, "an absent tile leaves the slot alone");
}

#[test]
fn a_damaged_record_read_through_a_reader_is_damaged() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    f.write_displacement(key(2, 0), op(2), &tile(2.0)).unwrap();
    f.save().unwrap();
    let loc = f.locate(&key(1, 0)).unwrap();
    let reader = f.reader();
    let mut scratch = ReadScratch::new().unwrap();
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();

    // The record of another key at this location.
    let other = f.locate(&key(2, 0)).unwrap();
    assert!(
        matches!(
            reader.read_displacement_into(&key(1, 0), other, &mut slot, &mut scratch),
            Err(FileError::Damaged { offset }) if offset == other.offset
        ),
        "wrong key"
    );
    assert!(
        matches!(
            reader.read_displacement_into(&small_key(1), loc, &mut slot, &mut scratch),
            Err(FileError::Damaged { .. })
        ),
        "a key of another level"
    );

    reader
        .read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch)
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let at = usize::try_from(loc.offset + loc.len / 2).unwrap();
    bytes[at] ^= 0x10;
    // The lock is advisory: an outside writer can still damage the file.
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .write_all_at(&bytes[at..=at], loc.offset + loc.len / 2)
        .unwrap();
    assert!(
        matches!(
            reader.read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch),
            Err(FileError::Damaged { offset }) if offset == loc.offset
        ),
        "flipped bit"
    );
    assert!(slot.view().is_err(), "a failed read leaves no tile");
    assert!(
        matches!(
            f.read_displacement_into(&key(1, 0), &mut slot, &mut scratch),
            Err(FileError::Damaged { .. })
        ),
        "the owner's read agrees"
    );
}

#[test]
fn a_reader_lets_go_of_the_file_when_its_owner_drops() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mut f = ProjectFile::create(&path, meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    f.save().unwrap();
    let loc = f.locate(&key(1, 0)).unwrap();
    let reader = f.reader();
    let clone = reader.clone();
    drop(reader);
    drop(f);
    let reopened = ProjectFile::open(&path);
    assert!(
        reopened.is_ok(),
        "the lock went with the owner: {reopened:?}"
    );
    let mut scratch = ReadScratch::new().unwrap();
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
    slot.reset(65).unwrap();
    assert!(
        matches!(
            clone.read_displacement_into(&key(1, 0), loc, &mut slot, &mut scratch),
            Err(FileError::Closed)
        ),
        "a reader of a closed file reads nothing"
    );
    assert!(slot.view().is_err(), "and leaves no tile");
}

#[test]
fn a_body_that_decompresses_short_is_damaged() {
    let dir = TestDir::new();
    let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
    let k = small_key(1);
    let (header, body) = encode_displacement(&small_tile(0.5)).unwrap();
    // States the right size, but the frame holds one border value less.
    let short = &body[..body.len() - 4];
    let payload = tile_payload(k, op(1), &header, body.len(), &compress(short).unwrap()).unwrap();
    let loc = f.append(KIND_TILE, &payload).unwrap();
    let mut scratch = ReadScratch::new().unwrap();
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
    assert!(
        matches!(
            f.reader()
                .read_displacement_into(&k, loc, &mut slot, &mut scratch),
            Err(FileError::Damaged { offset }) if offset == loc.offset
        ),
        "the reader rejects it"
    );
    assert!(slot.view().is_err(), "and leaves no tile");
    f.live.insert(Name::Tile(k), loc);
    assert!(
        matches!(
            f.read_displacement(&k),
            Err(FileError::Damaged { offset }) if offset == loc.offset
        ),
        "so does the owner"
    );
    assert!(
        matches!(f.read_tile(&k), Err(FileError::Damaged { .. })),
        "and the raw read"
    );
}

#[test]
fn meta_and_mesh_survive_create_and_open() {
    let dir = TestDir::new();
    let path = dir.path("r.bcrf");
    let mesh = Arc::new(geom::cube_sphere(2).unwrap());
    let project = crate::ProjectId::from_bits(42);
    let role = crate::Role::Recovery {
        base: Some(SavePoint {
            project,
            offset: 1234,
            head: Some(op(9)),
        }),
    };
    let meta = Meta::new(project, role, &mesh);
    let mut f = ProjectFile::create(&path, meta.clone(), Arc::clone(&mesh)).unwrap();
    assert_eq!(*f.meta(), meta, "as created");
    assert_eq!(f.keys().count(), 0, "no tiles");
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    let point = f.save().unwrap();
    drop(f);
    let opened = ProjectFile::open(&path).unwrap();
    assert_eq!(*opened.meta(), meta, "the meta, reopened");
    assert_eq!(
        opened.mesh().to_bytes(),
        mesh.to_bytes(),
        "the mesh, reopened"
    );
    assert_eq!(opened.keys().count(), 1, "the tile");
    drop(opened);
    let old = ProjectFile::open_at(&path, point).unwrap();
    assert_eq!(*old.meta(), meta, "the meta, at a save point");
    assert_eq!(
        old.mesh().to_bytes(),
        mesh.to_bytes(),
        "the mesh, at a save point"
    );
}

#[test]
fn a_meta_made_for_another_mesh_creates_nothing() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let other = geom::cube_sphere(2).unwrap();
    let created = ProjectFile::create(&path, fixture::meta_for(&other), mesh());
    assert!(
        matches!(created, Err(FileError::MeshMismatch)),
        "{created:?}"
    );
    assert!(!path.exists(), "no file");
}

#[test]
fn compaction_keeps_the_meta_and_the_mesh() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mesh = Arc::new(geom::cube_sphere(3).unwrap());
    let meta = fixture::meta_for(&mesh);
    let mut f = ProjectFile::create(&path, meta.clone(), Arc::clone(&mesh)).unwrap();
    for seq in 1..=10 {
        f.write_displacement(key(u32::try_from(seq % 3).unwrap(), 0), op(seq), &tile(1.0))
            .unwrap();
        f.save().unwrap();
    }
    drop(f);
    let out = dir.path("c.bcrf");
    compact(&path, &out).unwrap();
    compact_in_place(&path).unwrap();
    for p in [&out, &path] {
        let c = ProjectFile::open(p).unwrap();
        assert_eq!(*c.meta(), meta, "{p:?}: the meta");
        assert_eq!(c.mesh().to_bytes(), mesh.to_bytes(), "{p:?}: the mesh");
        assert_eq!(c.keys().count(), 3, "{p:?}: the tiles");
    }
}

#[test]
fn locations_stay_valid_after_more_appends() {
    let dir = TestDir::new();
    let mut f = ProjectFile::create(&dir.path("p.bcrf"), meta(), mesh()).unwrap();
    f.write_displacement(key(1, 0), op(1), &tile(1.0)).unwrap();
    let first = f.locate(&key(1, 0)).unwrap();
    let reader = f.reader();
    for seq in 2..6 {
        f.write_displacement(key(1, 0), op(seq), &tile(0.5))
            .unwrap();
        f.write_displacement(key(u32::try_from(seq).unwrap(), 0), op(seq), &tile(2.0))
            .unwrap();
        f.save().unwrap();
    }
    let mut scratch = ReadScratch::new().unwrap();
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
    assert_eq!(
        reader
            .read_displacement_into(&key(1, 0), first, &mut slot, &mut scratch)
            .unwrap(),
        op(1),
        "the old record is still there"
    );
    assert_eq!(slot.to_tile().unwrap(), tile(1.0), "with its tile");
    let latest = f.locate(&key(1, 0)).unwrap();
    assert_ne!(latest, first, "a newer record");
    assert_eq!(
        reader
            .read_displacement_into(&key(1, 0), latest, &mut slot, &mut scratch)
            .unwrap(),
        op(5),
        "the newest edit"
    );
    assert_eq!(slot.to_tile().unwrap(), tile(0.5), "the newest tile");
}
