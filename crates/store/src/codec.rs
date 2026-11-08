//! The displacement channel's payload: header `side: u16, scales: 3 × f32` (one per plane,
//! `(t, b, n)`), body planar delta-coded interior `i16`s then exact border `f32`s, compressed
//! with zstd.
//!
//! Tiles written before the planes had scales of their own have a header `side: u16, scale:
//! f32`; they read as three equal scales, so older files open.

use geom::{CorrectionError, CorrectionTile, Vec3};
use thiserror::Error;

use crate::bytes::Reader;
use crate::slot::{CorrectionSlot, SIDE};

/// zstd level: fast, and smooth correction data compresses well at it.
const LEVEL: i32 = 3;

/// Why a displacement payload didn't encode or decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CodecError {
    /// The header or body is the wrong size.
    #[error("displacement payload has the wrong size")]
    Size,
    /// The tile is wider than the format's `u16` side.
    #[error("a tile of {0} vertices per side is too wide to store")]
    TooWide(u32),
    /// The values don't form a valid tile.
    #[error("displacement payload is not a valid tile")]
    Tile(#[from] CorrectionError),
}

/// Header and uncompressed body of a displacement tile.
///
/// # Errors
///
/// [`CodecError::TooWide`] if the tile's side doesn't fit in a `u16`.
pub(crate) fn encode_displacement(tile: &CorrectionTile) -> Result<(Vec<u8>, Vec<u8>), CodecError> {
    let side = u16::try_from(tile.side())
        .ok()
        .ok_or_else(|| CodecError::TooWide(tile.side()))?;
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&side.to_le_bytes());
    for scale in tile.scales() {
        header.extend_from_slice(&scale.to_le_bytes());
    }
    let interior = tile.interior();
    let border = tile.border();
    let mut body = Vec::with_capacity(interior.len() * 6 + border.len() * 12);
    let interior_planes: [fn(&[i16; 3]) -> i16; 3] =
        [|&[x, _, _]| x, |&[_, y, _]| y, |&[_, _, z]| z];
    for pick in interior_planes {
        let mut prev = 0i16;
        for q in interior {
            let v = pick(q);
            body.extend_from_slice(&v.wrapping_sub(prev).to_le_bytes());
            prev = v;
        }
    }
    let border_planes: [fn(&Vec3) -> f32; 3] = [|v| v.x, |v| v.y, |v| v.z];
    for pick in border_planes {
        for v in border {
            body.extend_from_slice(&pick(v).to_le_bytes());
        }
    }
    Ok((header, body))
}

/// Decodes [`encode_displacement`]'s output into a [`CorrectionTile`]. Readers decode into slots
/// ([`decode_displacement_into`]); this is the tests' reference.
///
/// # Errors
///
/// As [`decode_displacement_into`].
#[cfg(test)]
pub(crate) fn decode_displacement(
    header: &[u8],
    body: &[u8],
) -> Result<CorrectionTile, CodecError> {
    let mut slot: CorrectionSlot = bytemuck::Zeroable::zeroed();
    decode_displacement_into(header, body, &mut slot)?;
    Ok(slot.to_tile()?)
}

/// Decodes [`encode_displacement`]'s output straight into `slot`, without allocating, and
/// validates it as [`CorrectionTile::from_parts`] does (see [`CorrectionSlot::validate`]).
///
/// Every value of the prefixes the tile's side uses is written, so `slot` needn't be zeroed
/// first; past them it keeps what it held (see [`CorrectionSlot::reset`]).
///
/// # Errors
///
/// [`CodecError::Size`] if the header or body length doesn't match the side; [`CodecError::Tile`]
/// if the values are invalid (side out of range, bad scale, `i16::MIN`, non-finite border, or a
/// plane with a zero scale and a nonzero value). On error the slot holds no tile.
pub(crate) fn decode_displacement_into(
    header: &[u8],
    body: &[u8],
    slot: &mut CorrectionSlot,
) -> Result<(), CodecError> {
    let decoded = fill_slot(header, body, slot);
    if decoded.is_err() {
        bytemuck::bytes_of_mut(slot).fill(0);
    }
    decoded
}

/// [`decode_displacement_into`] without clearing `slot` on error.
///
/// # Errors
///
/// As [`decode_displacement_into`].
fn fill_slot(header: &[u8], body: &[u8], slot: &mut CorrectionSlot) -> Result<(), CodecError> {
    let Header { side, scales } = parse_header(header)?;
    let (n, ring) = counts(side)?;
    let (interior, border) = body
        .split_at_checked(n * 6)
        .filter(|(_, border)| border.len() == ring * 12)
        .ok_or(CodecError::Size)?;
    // Every value of the side's prefixes is written below: nothing need be zeroed first.
    slot.side = side;
    slot.scales = scales;
    let interior_planes: [fn(&mut [i16; 3], i16); 3] =
        [|q, v| q[0] = v, |q, v| q[1] = v, |q, v| q[2] = v];
    let (deltas, _) = interior.as_chunks::<2>();
    for (c, put) in interior_planes.iter().enumerate() {
        let mut prev = 0i16;
        for (q, delta) in slot
            .interior
            .iter_mut()
            .zip(deltas.iter().skip(c * n).take(n))
        {
            prev = prev.wrapping_add(i16::from_le_bytes(*delta));
            put(q, prev);
        }
    }
    let border_planes: [fn(&mut Vec3, f32); 3] = [|v, x| v.x = x, |v, y| v.y = y, |v, z| v.z = z];
    let (values, _) = border.as_chunks::<4>();
    for (c, put) in border_planes.iter().enumerate() {
        for (v, bytes) in slot
            .border
            .iter_mut()
            .zip(values.iter().skip(c * ring).take(ring))
        {
            put(v, f32::from_le_bytes(*bytes));
        }
    }
    Ok(slot.validate()?)
}

/// Bytes of a displacement channel header: `side: u16, scales: 3 × f32`.
const HEADER_LEN: usize = 14;
/// Bytes of a header written before the planes had scales of their own: `side: u16, scale: f32`.
const ONE_SCALE_HEADER_LEN: usize = 6;

/// A displacement channel header.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Header {
    /// Vertices per side.
    side: u32,
    /// Each interior plane's quantization step, `(t, b, n)`.
    scales: [f32; 3],
}

/// Parses a displacement channel header: `side: u16, scales: 3 × f32`, or an older tile's
/// `side: u16, scale: f32`, whose one scale is every plane's. The header's length tells them
/// apart.
///
/// # Errors
///
/// [`CodecError::Size`] if it is neither.
fn parse_header(header: &[u8]) -> Result<Header, CodecError> {
    let mut h = Reader::new(header);
    let side = h.u16().ok_or(CodecError::Size)?;
    let mut scale = || h.f32().ok_or(CodecError::Size);
    let scales = match header.len() {
        HEADER_LEN => [scale()?, scale()?, scale()?],
        ONE_SCALE_HEADER_LEN => [scale()?; 3],
        _ => return Err(CodecError::Size),
    };
    Ok(Header {
        side: u32::from(side),
        scales,
    })
}

/// Interior and border vertex counts of a tile of `side`.
///
/// # Errors
///
/// [`CodecError::Tile`] if `side` is out of range (below 2, or wider than the largest tile).
fn counts(side: u32) -> Result<(usize, usize), CodecError> {
    let n = usize::try_from(side)
        .ok()
        .filter(|n| (2..=SIDE).contains(n))
        .ok_or(CorrectionError::BadSide(side))?;
    Ok(((n - 2) * (n - 2), 4 * (n - 1)))
}

/// The side a displacement header states, and the uncompressed body length that side needs, so
/// a reader can check a record's stated size before decompressing it.
///
/// # Errors
///
/// [`CodecError::Size`] if the header is malformed, [`CodecError::Tile`] if its side is out of
/// range.
pub(crate) fn displacement_body_len(header: &[u8]) -> Result<(u32, usize), CodecError> {
    let side = parse_header(header)?.side;
    let (n, ring) = counts(side)?;
    Ok((side, n * 6 + ring * 12))
}

/// zstd-compresses `raw`.
///
/// # Errors
///
/// The compressor's I/O error (allocation failure).
pub(crate) fn compress(raw: &[u8]) -> std::io::Result<Vec<u8>> {
    zstd::bulk::compress(raw, LEVEL)
}

/// Decompresses a zstd frame whose content is exactly `len` bytes.
///
/// # Errors
///
/// The decompressor's error for a damaged frame or one larger than `len`, or `InvalidData` if it
/// is smaller.
pub(crate) fn decompress(packed: &[u8], len: usize) -> std::io::Result<Vec<u8>> {
    let raw = zstd::bulk::decompress(packed, len)?;
    if raw.len() == len {
        Ok(raw)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "decompressed size differs",
        ))
    }
}

/// The magic number a zstd frame that holds data begins with, little-endian; a skippable frame
/// has another.
const FRAME_MAGIC: u32 = 0xFD2F_B528;

/// Decompresses `packed`, which must be one zstd frame holding exactly `len` bytes, and nothing
/// else.
///
/// A record's writer makes one frame holding data, so its reader takes nothing else: not a
/// skippable frame, and nothing after the frame.
///
/// # Errors
///
/// `InvalidData` if `packed` isn't one such frame; as [`decompress`] otherwise.
pub(crate) fn decompress_frame(packed: &[u8], len: usize) -> std::io::Result<Vec<u8>> {
    let magic = packed
        .first_chunk::<4>()
        .map(|bytes| u32::from_le_bytes(*bytes));
    let size = zstd::zstd_safe::find_frame_compressed_size(packed).ok();
    if magic != Some(FRAME_MAGIC) || size != Some(packed.len()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not exactly one zstd frame",
        ));
    }
    decompress(packed, len)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use bytemuck::Zeroable;

    use super::*;

    #[test]
    fn tiles_round_trip_bit_for_bit() {
        let mut rng = 0x1234_5678_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            f32::from(u16::try_from(rng >> 48).unwrap()) / 65536.0 - 0.5
        };
        for side in [2, 3, 9, 65] {
            let values: Vec<Vec3> = (0..side * side)
                .map(|_| Vec3::new(next(), next(), next()))
                .collect();
            let tile = CorrectionTile::from_values(side, &values).unwrap();
            let (header, raw) = encode_displacement(&tile).unwrap();
            let packed = compress(&raw).unwrap();
            let back =
                decode_displacement(&header, &decompress(&packed, raw.len()).unwrap()).unwrap();
            assert_eq!(back, tile, "side {side}");
            assert_eq!(
                back.border()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                tile.border()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "border bits"
            );
        }
    }

    #[test]
    fn smooth_tiles_compress_well() {
        let side = 65;
        let values: Vec<Vec3> = (0..side * side)
            .map(|k| {
                let (i, j) = (k % side, k / side);
                let (i, j) = (
                    f32::from(u16::try_from(i).unwrap()),
                    f32::from(u16::try_from(j).unwrap()),
                );
                Vec3::new(0.0, 0.0, 1e-3 * ((i * 0.1).sin() + (j * 0.07).cos()))
            })
            .collect();
        let tile = CorrectionTile::from_values(side, &values).unwrap();
        let (_, raw) = encode_displacement(&tile).unwrap();
        let packed = compress(&raw).unwrap();
        // Report, don't gate: the real B/vertex goal is measured in M0c.
        println!("smooth 65² tile: {} → {} bytes", raw.len(), packed.len());
        assert!(
            packed.len() < raw.len() / 2,
            "smooth data compresses at least 2×: {} → {}",
            raw.len(),
            packed.len()
        );
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        assert!(decode_displacement(&[1, 0], &[]).is_err(), "short header");
        let (header, raw) = encode_displacement(&CorrectionTile::zero(9).unwrap()).unwrap();
        assert!(
            decode_displacement(&header, &raw[1..]).is_err(),
            "short body"
        );
        let mut slot = CorrectionSlot::zeroed();
        for extra in [1, 2, 12, 64] {
            slot.reset(9).unwrap();
            let mut long = raw.clone();
            long.resize(raw.len() + extra, 0);
            assert_eq!(
                decode_displacement_into(&header, &long, &mut slot),
                Err(CodecError::Size),
                "body {extra} bytes too long"
            );
            assert!(
                slot.view().is_err(),
                "{extra} too long: the slot holds no tile"
            );
        }
        assert!(decompress(b"not zstd", 100).is_err(), "not a zstd frame");
    }

    /// A side-3 payload (one interior vertex, eight border ones): the scales (`(t, b, n)`), the
    /// interior vertex's x, and border x0.
    fn side3(scales: [f32; 3], x: i16, border_x: f32) -> (Vec<u8>, Vec<u8>) {
        let mut header = 3u16.to_le_bytes().to_vec();
        for scale in scales {
            header.extend_from_slice(&scale.to_le_bytes());
        }
        let mut body = vec![0; 6 + 8 * 12];
        body[..2].copy_from_slice(&x.to_le_bytes());
        body[6..10].copy_from_slice(&border_x.to_le_bytes());
        (header, body)
    }

    #[test]
    fn decoding_into_a_slot_is_the_gate() {
        let mut slot = CorrectionSlot::zeroed();
        for (what, scales, x, border_x) in [
            ("i16::MIN", [1.0; 3], i16::MIN, 0.0),
            ("NaN border", [1.0; 3], 1, f32::NAN),
            ("infinite border", [1.0; 3], 1, f32::INFINITY),
            ("negative scale", [1.0, 1.0, -1.0], 1, 0.0),
            ("infinite scale", [1.0, f32::INFINITY, 1.0], 1, 0.0),
            (
                "subnormal scale",
                [f32::MIN_POSITIVE / 2.0, 1.0, 1.0],
                1,
                0.0,
            ),
            ("zero scale, nonzero interior", [0.0, 1.0, 1.0], 1, 0.0),
        ] {
            let (header, body) = side3(scales, x, border_x);
            slot.reset(3).unwrap();
            assert!(
                matches!(
                    decode_displacement_into(&header, &body, &mut slot),
                    Err(CodecError::Tile(CorrectionError::BadParts { side: 3 }))
                ),
                "{what} rejected"
            );
            assert!(slot.view().is_err(), "{what}: the slot holds no tile");
            assert!(
                decode_displacement(&header, &body).is_err(),
                "{what}: the tile decoder agrees"
            );
        }
        let (zero_header, zero_body) = side3([-0.0; 3], 0, -0.0);
        decode_displacement_into(&zero_header, &zero_body, &mut slot).unwrap();
        assert_eq!(
            slot.scales.map(f32::to_bits),
            [0; 3],
            "-0.0 scale stored as +0.0"
        );
        assert_eq!(slot.border[0].x.to_bits(), 0, "-0.0 border stored as +0.0");
        let (good_header, good_body) = side3([0.5, 4.0, 4.0], 7, 1.5);
        decode_displacement_into(&good_header, &good_body, &mut slot).unwrap();
        assert_eq!(
            slot.view().unwrap().get(1, 1),
            Some(Vec3::new(3.5, 0.0, 0.0)),
            "interior decoded"
        );
        assert_eq!(
            slot.view().unwrap().get(0, 0),
            Some(Vec3::new(1.5, 0.0, 0.0)),
            "border"
        );
        let mut wide = 66u16.to_le_bytes().to_vec();
        wide.extend_from_slice(&[0; 12]);
        assert!(
            matches!(
                decode_displacement_into(&wide, &[], &mut slot),
                Err(CodecError::Tile(CorrectionError::BadSide(66)))
            ),
            "too wide"
        );
    }

    /// A decode writes every value of the side's prefixes, so a slot an earlier tile left
    /// values in decodes to the same tile as a zeroed one.
    #[test]
    fn decoding_overwrites_everything_the_side_uses() {
        let mut rng = 0x9E37_79B9_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            f32::from(u16::try_from(rng >> 48).unwrap()) / 65536.0 - 0.5
        };
        for side in [2, 3, 9, 33, 65] {
            let values: Vec<Vec3> = (0..side * side)
                .map(|_| Vec3::new(next(), next(), next()))
                .collect();
            let tile = CorrectionTile::from_values(side, &values).unwrap();
            let (header, raw) = encode_displacement(&tile).unwrap();
            let mut clean = CorrectionSlot::zeroed();
            decode_displacement_into(&header, &raw, &mut clean).unwrap();
            // Every byte set, padding and values past any prefix included, to what no decoded
            // value is: whatever the decode leaves unwritten shows.
            let mut used = CorrectionSlot::zeroed();
            bytemuck::bytes_of_mut(&mut used).fill(0x7F);
            decode_displacement_into(&header, &raw, &mut used).unwrap();
            assert_eq!(used.tile_bytes(), clean.tile_bytes(), "side {side}");
            assert_eq!(used.to_tile().unwrap(), tile, "side {side}");
        }
    }

    /// A one-scale header, as files before per-plane scales hold it, reads as three equal scales.
    ///
    /// The values decode the same, bit for bit, and the tile re-encodes with three scales. Any
    /// other header length is refused.
    #[test]
    fn one_scale_headers_decode_as_three_equal_scales() {
        let scale = 0.25_f32;
        let mut old = 3u16.to_le_bytes().to_vec();
        old.extend_from_slice(&scale.to_le_bytes());
        let (_, mut body) = side3([scale; 3], 7, 1.5);
        body[2..4].copy_from_slice(&(-3_i16).to_le_bytes());
        body[4..6].copy_from_slice(&5_i16.to_le_bytes());
        let tile = decode_displacement(&old, &body).unwrap();
        assert_eq!(tile.scales(), [scale; 3], "three equal scales");
        assert_eq!(
            tile.get(1, 1).map(Vec3::to_bits),
            Some(Vec3::new(1.75, -0.75, 1.25).to_bits()),
            "the interior as one scale decodes it"
        );
        let (header, again) = encode_displacement(&tile).unwrap();
        assert_eq!(header.len(), HEADER_LEN, "written with three scales");
        assert_eq!(again, body, "the same body");
        assert_eq!(decode_displacement(&header, &again).unwrap(), tile);
        for len in [0, 2, 5, 7, 10, 13, 15, 18] {
            let mut odd = 3u16.to_le_bytes().to_vec();
            odd.resize(len, 0);
            assert_eq!(
                displacement_body_len(&odd).unwrap_err(),
                CodecError::Size,
                "a {len}-byte header"
            );
        }
    }

    /// Each interior component decodes within half its own plane's step: the normal plane's
    /// fine step isn't spent on the coarser tangential parts.
    #[test]
    fn each_plane_decodes_within_half_its_step() {
        let mut rng = 0x5EED_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            f32::from(u16::try_from(rng >> 48).unwrap()) / 65536.0 - 0.5
        };
        let side = 33_u32;
        let values: Vec<Vec3> = (0..side * side)
            .map(|_| Vec3::new(next() * 0.3, next() * 0.01, next() * 1e-3))
            .collect();
        let tile = CorrectionTile::from_values(side, &values).unwrap();
        let (header, raw) = encode_displacement(&tile).unwrap();
        let back = decode_displacement(&header, &raw).unwrap();
        let [t, b, n] = back.scales();
        assert!(
            t > 8.0 * n && b >= 8.0 * n && n > 0.0,
            "{:?}",
            back.scales()
        );
        let half = back.max_error();
        for j in 1..side - 1 {
            for i in 1..side - 1 {
                let want = values[usize::try_from(j * side + i).unwrap()];
                let got = back.get(i, j).unwrap();
                let planes = [(want.x, got.x), (want.y, got.y), (want.z, got.z)];
                for ((w, g), bound) in planes.into_iter().zip(half) {
                    let slack = w.abs() * 4.0 * f32::EPSILON;
                    assert!((w - g).abs() <= bound + slack, "|{w} − {g}| > {bound}");
                }
            }
        }
    }

    #[test]
    fn the_smallest_tile_decodes() {
        let tile = CorrectionTile::from_values(2, &[Vec3::new(1.0, 2.0, 3.0); 4]).unwrap();
        let (header, raw) = encode_displacement(&tile).unwrap();
        assert_eq!(decode_displacement(&header, &raw).unwrap(), tile, "side 2");
    }
}
