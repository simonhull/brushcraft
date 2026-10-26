//! Property harnesses for the parsers a project or recovery file feeds: every one takes any bytes
//! without panicking, and every one with a writer gives back what its writer wrote.
//!
//! They are `bolero` checks: under `cargo nextest` each runs for bolero's default second on random
//! inputs; under `cargo bolero test -p store <name>` the same check is coverage-guided fuzzing.
//! Most harnesses start from a writer's output and change one byte of it, since random bytes
//! almost never get past a parser's first field.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use geom::{FaceId, LayerId, Level, Vec3};

use super::format::{ENTRY_LEN, MAX_TILE_RECORD, NONE, parse_index_payload, parse_tile_payload};
use super::*;
use crate::bounds::{KNOWN_RECORD_LEN, UNKNOWN_RECORD_LEN, parse_bounds_payload};
use crate::codec::decode_displacement_into;
use crate::meta::{MAX_BASE_PATH, ProjectId, Role};
use crate::op::{MAX_SEQ, RESERVED_SITE};
use crate::record::{HEAD, TAIL};
use crate::test_dir::TestDir;
use crate::{bounds, fixture};

/// A key's raw fields as a generator makes them: layer, channel, face, level, tx, ty.
type RawKey = (u16, u8, u32, u8, u16, u16);

/// One byte to change: `(position, xor)`, the position taken modulo the length.
type Flip = Option<(u16, u8)>;

/// A valid key from any raw fields: the level and tile wrapped into range.
fn key_of((layer, channel, face, level, tx, ty): RawKey) -> TileKey {
    let level = Level::new(level % (geom::MAX_LEVEL + 1)).unwrap();
    let per_side = level.tiles_per_side();
    let wrap = |t: u16| u16::try_from(u32::from(t) % per_side).unwrap();
    TileKey::new(
        LayerId::new(layer),
        Channel::from_raw(channel),
        FaceId::new(face),
        level,
        wrap(tx),
        wrap(ty),
    )
    .unwrap()
}

/// `bytes` with `flip` applied, if any (and if there is a byte to change).
fn flipped(mut bytes: Vec<u8>, flip: Flip) -> Vec<u8> {
    if let Some((at, xor)) = flip
        && !bytes.is_empty()
    {
        let at = usize::from(at) % bytes.len();
        bytes[at] ^= xor;
    }
    bytes
}

/// An op id from any bits: the site wrapped below the reserved one.
fn op_of(site: u16, seq: u64) -> OpId {
    OpId::new(site % RESERVED_SITE, seq & MAX_SEQ).unwrap()
}

/// What generates an Index: coverage, parent, head, then tile and bounds entries as
/// `(key, offset, length choice)`.
type IndexParts = (
    bool,
    Option<u64>,
    Option<(u16, u64)>,
    Vec<(RawKey, u64, u32)>,
    Vec<(RawKey, u64, bool)>,
);

/// [`index_payload`] of the Index `parts` describe, every entry a live record of a length its
/// kind can have.
fn index_of((full, parent, head, tiles, bounds): IndexParts) -> Vec<u8> {
    let tile_len = |len: u32| u64::from(len) % MAX_TILE_RECORD + 1;
    let bounds_len = |known: bool| {
        if known {
            KNOWN_RECORD_LEN
        } else {
            UNKNOWN_RECORD_LEN
        }
    };
    let entries: BTreeMap<Name, Loc> = tiles
        .into_iter()
        .map(|(key, offset, len)| {
            let loc = Loc {
                offset,
                len: tile_len(len),
            };
            (Name::Tile(key_of(key)), loc)
        })
        .chain(bounds.into_iter().map(|(key, offset, known)| {
            let loc = Loc {
                offset,
                len: bounds_len(known),
            };
            (Name::Bounds(key_of(key)), loc)
        }))
        .collect();
    let coverage = if full {
        Coverage::Full
    } else {
        Coverage::Delta
    };
    let parent = parent.filter(|&p| p != NONE);
    let head = head.map(|(site, seq)| op_of(site, seq));
    index_payload(coverage, parent, head, &entries).unwrap()
}

/// What [`index_payload`] writes for a parsed Index, or `None` if it names a removed entry,
/// which the writer never writes (its zero length drops its offset).
fn rewritten(index: &Index) -> Option<Vec<u8>> {
    let entries = index
        .entries
        .iter()
        .map(|&(name, loc)| loc.map(|loc| (name, loc)))
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(index_payload(index.coverage, index.parent, index.head, &entries).unwrap())
}

/// The Index payload's reserved bytes, after the coverage byte.
const RESERVED: std::ops::Range<usize> = 1..8;

/// `payload` with its reserved bytes zeroed, as the writer writes them.
fn reserved_zeroed(mut payload: Vec<u8>) -> Vec<u8> {
    if let Some(reserved) = payload.get_mut(RESERVED) {
        reserved.fill(0);
    }
    payload
}

/// The Index parser takes any bytes.
#[test]
fn index_parsing_never_panics() {
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Some(index) = parse_index_payload(bytes) {
            // What the entries take is bounded by the bytes they were read from.
            assert!(
                index.entries.len() * ENTRY_LEN <= bytes.len(),
                "entries from bytes"
            );
        }
    });
}

/// Every accepted Index payload is its own rewrite, but for the reserved bytes, which the parser
/// skips unchecked (see [`index_payloads_round_trip_exactly`]).
#[test]
fn index_payloads_round_trip_but_for_the_reserved_bytes() {
    bolero::check!()
        .with_type::<(IndexParts, Flip)>()
        .for_each(|(parts, flip)| {
            let payload = flipped(index_of(parts.clone()), *flip);
            if let Some(again) = parse_index_payload(&payload).as_ref().and_then(rewritten) {
                assert_eq!(again, reserved_zeroed(payload), "the Index's rewrite");
            }
        });
}

/// Every accepted Index payload is its own rewrite, byte for byte.
///
/// Fails today: the parser reads the 7 reserved bytes after the coverage byte without checking
/// they are zero, so a payload with any of them set parses and rewrites with zeros. Whether
/// readers should require zeros is a file-format decision.
#[test]
#[ignore = "pending the reserved-bytes decision (plan 008, TEST-06)"]
fn index_payloads_round_trip_exactly() {
    bolero::check!()
        .with_type::<(IndexParts, Flip)>()
        .for_each(|(parts, flip)| {
            let payload = flipped(index_of(parts.clone()), *flip);
            if let Some(again) = parse_index_payload(&payload).as_ref().and_then(rewritten) {
                assert_eq!(again, payload, "the Index's rewrite");
            }
        });
}

/// What generates a tile payload: key, op, header, the body's stated length, and the packed body.
type TileParts = (RawKey, (u16, u64), Vec<u8>, u32, Vec<u8>);

/// The tile record parser takes any bytes, and every tile payload it accepts is its own rewrite.
#[test]
fn tile_payloads_never_panic_and_round_trip() {
    bolero::check!().with_type::<(TileParts, Flip)>().for_each(
        |((key, (site, seq), header, raw_len, packed), flip)| {
            let raw_len = usize::try_from(raw_len % (MAX_RAW + 1)).unwrap();
            let written =
                tile_payload(key_of(*key), op_of(*site, *seq), header, raw_len, packed).unwrap();
            let payload = flipped(written, *flip);
            if let Some(tile) = parse_tile_payload(&payload) {
                let again = tile_payload(tile.key, tile.op, tile.header, tile.raw_len, tile.packed);
                assert_eq!(again.unwrap(), payload, "the tile record's rewrite");
            }
        },
    );
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Some(tile) = parse_tile_payload(bytes) {
            assert!(
                tile.raw_len <= MAX_RAW as usize,
                "a stated size within the cap"
            );
            let borrowed = tile.header.len() + tile.packed.len();
            assert!(borrowed <= bytes.len(), "header and body from the payload");
        }
    });
}

/// Record verification takes any bytes, accepts every framed record, and rejects one with any
/// byte changed.
#[test]
fn record_verification_never_panics_and_accepts_what_is_framed() {
    bolero::check!()
        .with_type::<(u8, Vec<u8>, Flip)>()
        .for_each(|(kind, payload, flip)| {
            let framed = frame(*kind, payload).unwrap();
            assert_eq!(
                verify(&framed),
                Some((*kind, payload.as_slice())),
                "a framed record verifies"
            );
            let damaged = flipped(framed.clone(), *flip);
            if damaged != framed {
                assert_eq!(verify(&damaged), None, "one changed byte fails the check");
            }
        });
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Some((_, payload)) = verify(bytes) {
            assert_eq!(HEAD + payload.len() + TAIL, bytes.len(), "the whole record");
        }
    });
}

/// What generates a meta: project, then for a recovery file its base, as `(project, offset,
/// head, path)`.
type MetaParts = (
    u128,
    Option<Option<(u128, u64, Option<(u16, u64)>, Vec<u8>)>>,
);

/// The meta parser takes any bytes, and every meta payload it accepts is its own rewrite.
#[test]
fn meta_payloads_never_panic_and_round_trip() {
    use std::os::unix::ffi::OsStrExt;

    bolero::check!()
        .with_type::<(MetaParts, Flip)>()
        .for_each(|((project, role), flip)| {
            let meta = fixture::meta().with_project(ProjectId::from_bits(*project));
            let meta = match role {
                None => meta,
                Some(None) => meta.with_role(Role::Recovery { base: None }),
                Some(Some((base, offset, head, path))) => {
                    let point = SavePoint {
                        project: ProjectId::from_bits(*base),
                        offset: *offset,
                        head: head.map(|(site, seq)| op_of(site, seq)),
                    };
                    let path = Path::new(std::ffi::OsStr::from_bytes(path));
                    meta.with_role(Role::Recovery { base: Some(point) })
                        .on_save_point(point, path)
                }
            };
            let payload = flipped(meta.to_payload(), *flip);
            if let Some(back) = Meta::from_payload(&payload) {
                assert_eq!(back.to_payload(), payload, "the meta's rewrite");
            }
        });
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Some(meta) = Meta::from_payload(bytes) {
            let path = meta.base_path().map_or(0, |path| path.as_os_str().len());
            assert!(path <= MAX_BASE_PATH, "a base path within the cap");
        }
    });
}

/// What generates a bound: the key, and for a known one the positions and normals of a tile.
type BoundsParts = (RawKey, Option<Vec<([f32; 3], [f32; 3])>>);

/// The bounds parser takes any bytes, and every bound it accepts is its own rewrite, of a
/// length a bounds record can have.
#[test]
fn bounds_payloads_never_panic_and_round_trip() {
    bolero::check!()
        .with_type::<(BoundsParts, Flip)>()
        .for_each(|((key, points), flip)| {
            let bound = points.as_ref().and_then(|points| {
                let (positions, normals): (Vec<Vec3>, Vec<Vec3>) = points
                    .iter()
                    .map(|&([px, py, pz], [nx, ny, nz])| {
                        (Vec3::new(px, py, pz), Vec3::new(nx, ny, nz))
                    })
                    .unzip();
                TileBounds::of_tile(&positions, &normals)
            });
            let payload = flipped(bounds_payload(&key_of(*key), bound.as_ref()), *flip);
            if let Some((parsed, kept)) = parse_bounds_payload(&payload) {
                let again = bounds_payload(&parsed, kept.bounds().as_ref());
                assert_eq!(again, payload, "the bound's rewrite");
            }
        });
    bolero::check!().for_each(|bytes: &[u8]| {
        if parse_bounds_payload(bytes).is_some() {
            let framed = u64::try_from(HEAD + TAIL + bytes.len()).unwrap();
            assert!(
                bounds::is_record_len(framed),
                "an accepted bound's record length"
            );
        }
    });
}

/// A displacement tile's header and a body of the length its side needs, from any side, scale
/// and body bytes (repeated to length).
fn displacement_of(side: u8, scale: f32, bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let side = u16::from(side % 70);
    let mut header = side.to_le_bytes().to_vec();
    header.extend_from_slice(&scale.to_le_bytes());
    let n = usize::from(side.saturating_sub(2));
    let ring = 4 * usize::from(side.saturating_sub(1));
    let len = n * n * 6 + ring * 12;
    let body = bytes.iter().copied().cycle().take(len).collect();
    (header, body)
}

/// The displacement decoder takes any header and body, and a tile it accepts encodes and decodes
/// again to the same slot, bit for bit.
#[test]
fn displacement_decoding_never_panics_and_round_trips() {
    bolero::check!()
        .with_type::<(u8, f32, Vec<u8>, Flip)>()
        .for_each(|(side, scale, bytes, flip)| {
            let (header, body) = displacement_of(*side, *scale, bytes);
            let body = flipped(body, *flip);
            let mut slot: Box<CorrectionSlot> = bytemuck::zeroed_box();
            if decode_displacement_into(&header, &body, &mut slot).is_ok() {
                let tile: CorrectionTile = slot.to_tile().unwrap();
                let (header_again, body_again) = encode_displacement(&tile).unwrap();
                let mut again: Box<CorrectionSlot> = bytemuck::zeroed_box();
                decode_displacement_into(&header_again, &body_again, &mut again).unwrap();
                assert!(
                    bytemuck::bytes_of(&*again) == bytemuck::bytes_of(&*slot),
                    "the decoded tile's re-encoding decodes to the same slot"
                );
            }
        });
    bolero::check!()
        .with_type::<(Vec<u8>, Vec<u8>)>()
        .for_each(|(header, body)| {
            let mut slot: Box<CorrectionSlot> = bytemuck::zeroed_box();
            if decode_displacement_into(header, body, &mut slot).is_err() {
                let cleared = bytemuck::bytes_of(&*slot).iter().all(|&b| b == 0);
                assert!(cleared, "a slot that failed to decode holds no tile");
            }
        });
}

/// Decompression takes any frame and any stated length up to [`MAX_RAW`], and gives back a
/// frame's content at its own length.
#[test]
fn decompression_never_panics() {
    bolero::check!()
        .with_type::<(Vec<u8>, u32, Flip)>()
        .for_each(|(raw, len, flip)| {
            let packed = flipped(compress(raw).unwrap(), *flip);
            let len = usize::try_from(len % (MAX_RAW + 1)).unwrap();
            if let Ok(back) = decompress(&packed, raw.len()) {
                assert_eq!(back.len(), raw.len(), "the stated length");
            }
            if let Ok(back) = decompress(&packed, len) {
                assert_eq!(back.len(), len, "exactly the stated length");
            }
        });
}

/// A small project file at `path` with `meta`: three save points (one full Index, two deltas)
/// of a tile each at levels 1 to 3, a known and an unknown bound, and an unsaved tile after the
/// last save point.
fn seed_file(path: &Path, meta: Meta) {
    let mut file = ProjectFile::create(path, meta, fixture::mesh()).unwrap();
    for (seq, level) in (1..).zip(1..=3_u8) {
        let face = FaceId::new(level.into());
        let key = TileKey::displacement(face, Level::new(level).unwrap(), 0, 0).unwrap();
        let side = key.side();
        let values: Vec<Vec3> = (0..side * side)
            .map(|k| Vec3::new(0.0, f32::from(u8::try_from(k % 5).unwrap()) * 1e-3, 0.0))
            .collect();
        let tile = CorrectionTile::from_values(side, &values).unwrap();
        file.write_displacement(key, op_of(0, seq), &tile).unwrap();
        let bound = TileBounds::of_tile(
            &[Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)],
            &[Vec3::new(0.0, 0.0, 1.0)],
        );
        let known = seq == 2;
        file.write_bounds(key, bound.as_ref().filter(|_| known), Stamp::ZERO)
            .unwrap();
        file.save().unwrap();
    }
    let key = TileKey::displacement(FaceId::new(4), Level::new(1).unwrap(), 0, 0).unwrap();
    let tile = CorrectionTile::from_values(3, &[Vec3::ZERO; 9]).unwrap();
    file.write_displacement(key, op_of(0, 4), &tile).unwrap();
}

/// Writes the seed corpora of the cargo-fuzz targets (`fuzz/corpus/<target>/seed-*`).
///
/// Each is a few KiB: the tests' mesh, and small project and recovery files over it. Run by hand
/// when the format changes: `cargo nextest run -p store --run-ignored only write_fuzz_seeds`.
#[test]
#[ignore = "writes the fuzz seed corpora; run by hand"]
fn write_fuzz_seeds() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/corpus");
    let mesh = fixture::mesh().to_bytes();
    std::fs::write(corpus.join("mesh_from_bytes/seed-cube"), mesh).unwrap();
    let dir = TestDir::new();
    let project = dir.path("project");
    seed_file(&project, fixture::meta());
    let point = ProjectFile::open(&project)
        .unwrap()
        .base()
        .unwrap()
        .save_point();
    let recovery = fixture::meta()
        .with_role(Role::Recovery { base: Some(point) })
        .on_save_point(point, Path::new("/home/someone/sculpts/head.project"));
    let recovery_path = dir.path("recovery");
    seed_file(&recovery_path, recovery);
    let seeds = corpus.join("project_open");
    for (from, name) in [(project, "seed-project"), (recovery_path, "seed-recovery")] {
        let bytes = std::fs::read(from).unwrap();
        assert!(bytes.len() <= 64 << 10, "{name}: a seed stays small");
        std::fs::write(seeds.join(name), bytes).unwrap();
    }
}
