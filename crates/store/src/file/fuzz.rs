//! Property harnesses for the parsers a project or recovery file feeds: every one takes any bytes
//! without panicking, and every one with a writer gives back what its writer wrote.
//!
//! They are `bolero` checks: under `cargo nextest` each runs for bolero's default second on random
//! inputs; under `cargo bolero test -p store <name>` the same check is coverage-guided fuzzing.
//! Most harnesses start from a writer's output and change one byte of it, since random bytes
//! almost never get past a parser's first field.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use geom::{
    CorrectionMap, DenseEvaluation, DisplayTile, FaceGrids, FaceId, LayerId, Level, Vec3,
    build_mesh,
};

use super::format::{
    ENTRY_LEN, MAX_TILE_RECORD, NONE, index_payload, parse_index_payload, parse_tile_payload,
};
use super::*;
use crate::bounds::{KNOWN_RECORD_LEN, UNKNOWN_RECORD_LEN, parse_bounds_payload};
use crate::codec::decode_displacement_into;
use crate::display::{DisplayRecord, StoredDisplay, display_body, display_header, parse_display};
use crate::meta::{MAX_BASE_PATH, ProjectId, Role};
use crate::op::{MAX_SEQ, RESERVED_SITE};
use crate::record::{HEAD, TAIL};
use crate::stroke_record::{MAX_STROKE_SETTINGS, PenSample, PenTilt, StrokeCamera, StrokeRecord};
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

/// Every accepted Index payload is its own rewrite, byte for byte: the parser refuses a payload
/// whose reserved bytes aren't the zeros the writer writes.
#[test]
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

/// An Index payload with any of its seven reserved bytes set, the bytes after the coverage byte,
/// is refused: a later version may give them a meaning this one can't read.
#[test]
fn an_index_with_a_reserved_byte_set_is_refused() {
    let written = index_payload(Coverage::Full, None, None, &BTreeMap::new()).unwrap();
    assert!(parse_index_payload(&written).is_some(), "the writer's own");
    for byte in 1..8 {
        let mut payload = written.clone();
        payload[byte] = 1;
        assert!(
            parse_index_payload(&payload).is_none(),
            "reserved byte {byte}"
        );
    }
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

/// A displacement tile's header and a body of the length its side needs.
///
/// From any side, scales and body bytes (repeated to length). With `one_scale`, the header is
/// an older file's, with the first scale alone.
fn displacement_of(
    side: u8,
    scales: [f32; 3],
    one_scale: bool,
    bytes: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let side = u16::from(side % 70);
    let mut header = side.to_le_bytes().to_vec();
    let written = if one_scale { &scales[..1] } else { &scales[..] };
    for scale in written {
        header.extend_from_slice(&scale.to_le_bytes());
    }
    let n = usize::from(side.saturating_sub(2));
    let ring = 4 * usize::from(side.saturating_sub(1));
    let len = n * n * 6 + ring * 12;
    let body = bytes.iter().copied().cycle().take(len).collect();
    (header, body)
}

/// The displacement decoder takes any header and body, an older file's one-scale header
/// included, and a tile it accepts encodes and decodes again to the same slot, bit for bit.
#[test]
fn displacement_decoding_never_panics_and_round_trips() {
    bolero::check!()
        .with_type::<(u8, [f32; 3], bool, Vec<u8>, Flip)>()
        .for_each(|(side, scales, one_scale, bytes, flip)| {
            let (header, body) = displacement_of(*side, *scales, *one_scale, bytes);

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

/// What generates a stroke record.
///
/// The op's site and sequence, the engine version, the settings, the camera's ten `f64`s and its
/// size, then each sample's position, time, pressure, tilt amount and direction, twist and
/// buttons.
type StrokeParts = (
    (u16, u64),
    u32,
    Vec<u8>,
    [f64; 10],
    [u32; 2],
    Vec<(f64, f64, f64, f32, Option<(f32, f32)>, Option<f32>, u8)>,
);

/// The longest stroke a generator makes: enough samples for every delta and plane to matter.
const FUZZ_SAMPLES: usize = 600;

/// The stroke record `parts` describe, its samples cut at [`FUZZ_SAMPLES`].
fn stroke_of((op, engine, settings, floats, size, samples): &StrokeParts) -> StrokeRecord {
    let [ex, ey, ez, tx, ty, tz, ux, uy, uz, fov_y] = *floats;
    let camera = StrokeCamera {
        eye: [ex, ey, ez],
        target: [tx, ty, tz],
        up: [ux, uy, uz],
        fov_y,
        size: *size,
    };
    let settings: Vec<u8> = settings.iter().copied().take(MAX_STROKE_SETTINGS).collect();
    let mut record = StrokeRecord::new(op_of(op.0, op.1), *engine, settings, camera).unwrap();
    let pushed = samples.iter().take(FUZZ_SAMPLES);
    for &(x, y, time, pressure, tilt, twist, buttons) in pushed {
        let sample = PenSample {
            x,
            y,
            time,
            pressure,
            tilt: tilt.map(|(amount, direction)| PenTilt { amount, direction }),
            twist,
            buttons,
        };
        record.push(&sample).unwrap();
    }
    record
}

/// Whether two stroke records write the same payload: equal, NaNs of the camera included, which
/// `PartialEq` on `f64` can't say.
fn same_record(a: &StrokeRecord, b: &StrokeRecord) -> bool {
    a.to_payload().unwrap() == b.to_payload().unwrap()
}

/// The stroke record parser takes any bytes without panicking, and reads back what was written.
///
/// It is given a writer's payload with any one byte changed, and random bytes. Whatever it
/// accepts writes a payload it reads back as the same record.
#[test]
fn stroke_payloads_never_panic_and_round_trip() {
    bolero::check!()
        .with_type::<(StrokeParts, Flip)>()
        .for_each(|(parts, flip)| {
            let record = stroke_of(parts);
            let payload = record.to_payload().unwrap();
            let back = StrokeRecord::from_payload(&payload).expect("a writer's payload reads");
            assert!(same_record(&back, &record), "the record written");
            assert_eq!(back.samples().len(), record.len(), "every sample");
            if let Some(changed) = StrokeRecord::from_payload(&flipped(payload, *flip)) {
                let again = StrokeRecord::from_payload(&changed.to_payload().unwrap())
                    .expect("its own payload reads");
                assert!(
                    same_record(&again, &changed),
                    "an accepted record's rewrite"
                );
            }
        });
    bolero::check!().for_each(|bytes: &[u8]| {
        if let Some(record) = StrokeRecord::from_payload(bytes) {
            let again = StrokeRecord::from_payload(&record.to_payload().unwrap())
                .expect("its own payload reads");
            assert!(same_record(&again, &record), "an accepted record's rewrite");
        }
    });
}

/// A small project file at `path` with `meta`, of every kind of record.
///
/// Three save points (one full Index, two deltas) of a tile each at levels 1 to 3, a known and an
/// unknown bound, and an unsaved tile after the last save point. The first and third tiles'
/// strokes leave records, and so does the unsaved tile's.
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
        if seq != 2 {
            seed_stroke(&mut file, op_of(0, seq));
        }
        file.save().unwrap();
    }
    // Display records, named by a save point: two known, one unknown.
    let display = small_display_tiles();
    let records: Vec<DisplayRecord> = display
        .iter()
        .take(2)
        .map(|tile| DisplayRecord::of(tile, SMALL_FINEST))
        .collect();
    let unknown = *display[2].key();
    file.write_display_many(
        records
            .iter()
            .zip(&display)
            .map(|(r, t)| (*t.key(), Some(r)))
            .chain([(unknown, None)]),
        Stamp::ZERO,
    )
    .unwrap();
    file.save().unwrap();
    let key = TileKey::displacement(FaceId::new(4), Level::new(1).unwrap(), 0, 0).unwrap();
    let tile = CorrectionTile::from_values(3, &[Vec3::ZERO; 9]).unwrap();
    file.write_displacement(key, op_of(0, 4), &tile).unwrap();
    seed_stroke(&mut file, op_of(0, 4));
}

/// Appends a short stroke record of `op` to `file`: a few samples, every field moving.
fn seed_stroke(file: &mut ProjectFile, op: OpId) {
    let record = crate::stroke_record::tests::wobbly(op, 12, op.seq());
    file.append_strokes(&[frame_stroke(&record).unwrap()])
        .unwrap();
}

/// The finest level [`small_display_tiles`] are built from.
const SMALL_FINEST: Level = match Level::new(2) {
    Ok(level) => level,
    Err(_) => unreachable!(),
};

/// Display tiles of the tests' mesh at levels 0 and 1, built from its level-2 surface: small,
/// real tiles for the display records' harnesses and seeds.
fn small_display_tiles() -> Vec<DisplayTile> {
    let mesh = fixture::mesh();
    let finest = SMALL_FINEST;
    let dense = DenseEvaluation::new(&mesh, finest, &CorrectionMap::new()).unwrap();
    let (mut positions, mut normals) = (FaceGrids::new(finest), FaceGrids::new(finest));
    for face in mesh.faces() {
        positions
            .insert(face, dense.grid(face, finest).unwrap().clone())
            .unwrap();
        let unit = geom::normals(&mesh, &dense, geom::Region::whole_face(face, finest)).unwrap();
        normals.insert(face, unit).unwrap();
    }
    build_mesh(&mesh, &positions, &normals).unwrap()
}

/// [`parse_display`] takes any header and body for any key without panicking, and what it
/// accepts is a display tile of that key.
#[test]
fn display_records_parse_anything() {
    bolero::check!()
        .with_type::<(RawKey, Vec<u8>, Vec<u8>)>()
        .for_each(|(key, header, body)| {
            let key = key_of(*key);
            if let Some(StoredDisplay::Known(record)) = parse_display(&key, header, body) {
                assert!(record.decode(&key).is_some(), "accepted means decodes");
            }
        });
}

/// A display record reads back as its writer wrote it; with one byte of its header or body
/// changed, it reads as a tile that decodes (a change that keeps it valid) or not at all, never
/// as a panic.
#[test]
fn display_records_round_trip_or_are_refused() {
    let tiles = small_display_tiles();
    bolero::check!()
        .with_type::<(u8, Flip, bool)>()
        .for_each(|&(pick, flip, in_header)| {
            let tile = &tiles[usize::from(pick) % tiles.len()];
            let record = DisplayRecord::of(tile, SMALL_FINEST);
            let header = display_header(Some(&record));
            let body = display_body(Some(&record)).to_vec();
            let parsed = if in_header {
                parse_display(tile.key(), &flipped(header, flip), &body)
            } else {
                parse_display(tile.key(), &header, &flipped(body, flip))
            };
            if flip.is_none_or(|(_, xor)| xor == 0) {
                assert_eq!(
                    parsed,
                    Some(StoredDisplay::Known(record)),
                    "{:?} round-trips",
                    tile.key()
                );
            }
        });
    assert_eq!(
        parse_display(tiles[0].key(), &display_header(None), &[]),
        Some(StoredDisplay::Unknown),
        "an unknown record reads as unknown"
    );
}

/// A display record as version 2 wrote it (the finest level and the bound, no normal cone) reads
/// as unknown, known or not, so its tile is rebuilt with a cone.
#[test]
fn a_version_2_display_record_reads_as_unknown() {
    let tiles = small_display_tiles();
    let tile = &tiles[0];
    let record = DisplayRecord::of(tile, SMALL_FINEST);
    let mut known = vec![2, 1, SMALL_FINEST.get()];
    known.extend_from_slice(&record.error.to_le_bytes());
    let unknown = vec![2, 0];
    let cases = [
        (known, record.encoded.as_slice(), Some(&record)),
        (unknown, &[][..], None),
    ];
    for (header, body, held) in cases {
        assert_eq!(
            parse_display(tile.key(), &header, body),
            None,
            "{header:?}: not a header this build reads"
        );
        let raw = version_2_record(*tile.key(), held);
        assert_eq!(display_record(&raw, tile.key()), None, "{header:?}");
        // As a file holds it: read back as unknown, not as a missing record.
        let dir = TestDir::new();
        let mut file =
            ProjectFile::create(&dir.path("old.bcp"), fixture::meta(), fixture::mesh()).unwrap();
        file.append_display_raw(tile.key(), &raw).unwrap();
        assert_eq!(
            file.read_display(tile.key()).unwrap(),
            Some(StoredDisplay::Unknown),
            "{header:?}"
        );
    }
    let current = display_header(Some(&record));
    assert_eq!(current.first(), Some(&3), "this build writes version 3");
    assert_eq!(
        parse_display(tile.key(), &current, &record.encoded),
        Some(StoredDisplay::Known(record)),
        "and reads it"
    );
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
