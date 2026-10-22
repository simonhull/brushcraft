//! The displacement channel's payload: header `side: u16, scale: f32`, body planar delta-coded
//! interior `i16`s then exact border `f32`s, compressed with zstd.

use geom::{CorrectionError, CorrectionTile, Vec3};
use thiserror::Error;

use crate::bytes::Reader;
use crate::slot::{CorrectionSlot, SIDE};

/// zstd level: fast, and smooth correction data compresses well at it.
const LEVEL: i32 = 3;

/// Why a displacement payload didn't encode or decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
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
    let mut header = Vec::with_capacity(6);
    header.extend_from_slice(&side.to_le_bytes());
    header.extend_from_slice(&tile.scale().to_le_bytes());
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
/// # Errors
///
/// [`CodecError::Size`] if the header or body length doesn't match the side; [`CodecError::Tile`]
/// if the values are invalid (side out of range, bad scale, `i16::MIN`, non-finite border, or a
/// zero scale with a nonzero interior). On error the slot holds no tile.
pub(crate) fn decode_displacement_into(
    header: &[u8],
    body: &[u8],
    slot: &mut CorrectionSlot,
) -> Result<(), CodecError> {
    let Header { side, scale } = parse_header(header)?;
    let (n, ring) = counts(side)?;
    let (interior, border) = body
        .split_at_checked(n * 6)
        .filter(|(_, border)| border.len() == ring * 12)
        .ok_or(CodecError::Size)?;
    slot.reset(side)?;
    slot.scale = scale;
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

/// A displacement channel header.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Header {
    /// Vertices per side.
    side: u32,
    /// The interior quantization step.
    scale: f32,
}

/// Parses a displacement channel header.
///
/// # Errors
///
/// [`CodecError::Size`] if it isn't exactly `side: u16, scale: f32`.
fn parse_header(header: &[u8]) -> Result<Header, CodecError> {
    let mut h = Reader::new(header);
    let side = h.u16().ok_or(CodecError::Size)?;
    let scale = h.f32().ok_or(CodecError::Size)?;
    if h.rest().is_empty() {
        Ok(Header {
            side: u32::from(side),
            scale,
        })
    } else {
        Err(CodecError::Size)
    }
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
        assert!(decompress(b"not zstd", 100).is_err(), "not a zstd frame");
    }

    /// A side-3 payload (one interior vertex, eight border ones): scale, interior x, border x0.
    fn side3(scale: f32, x: i16, border_x: f32) -> (Vec<u8>, Vec<u8>) {
        let mut header = 3u16.to_le_bytes().to_vec();
        header.extend_from_slice(&scale.to_le_bytes());
        let mut body = vec![0; 6 + 8 * 12];
        body[..2].copy_from_slice(&x.to_le_bytes());
        body[6..10].copy_from_slice(&border_x.to_le_bytes());
        (header, body)
    }

    #[test]
    fn decoding_into_a_slot_is_the_gate() {
        let mut slot = CorrectionSlot::zeroed();
        for (what, scale, x, border_x) in [
            ("i16::MIN", 1.0, i16::MIN, 0.0),
            ("NaN border", 1.0, 1, f32::NAN),
            ("infinite border", 1.0, 1, f32::INFINITY),
            ("negative scale", -1.0, 1, 0.0),
            ("infinite scale", f32::INFINITY, 1, 0.0),
            ("subnormal scale", f32::MIN_POSITIVE / 2.0, 1, 0.0),
            ("zero scale, nonzero interior", 0.0, 1, 0.0),
        ] {
            let (header, body) = side3(scale, x, border_x);
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
        let (zero_header, zero_body) = side3(-0.0, 0, -0.0);
        decode_displacement_into(&zero_header, &zero_body, &mut slot).unwrap();
        assert_eq!(slot.scale.to_bits(), 0, "-0.0 scale stored as +0.0");
        assert_eq!(slot.border[0].x.to_bits(), 0, "-0.0 border stored as +0.0");
        let (good_header, good_body) = side3(0.5, 7, 1.5);
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
        wide.extend_from_slice(&0f32.to_le_bytes());
        assert!(
            matches!(
                decode_displacement_into(&wide, &[], &mut slot),
                Err(CodecError::Tile(CorrectionError::BadSide(66)))
            ),
            "too wide"
        );
    }

    #[test]
    fn the_smallest_tile_decodes() {
        let tile = CorrectionTile::from_values(2, &[Vec3::new(1.0, 2.0, 3.0); 4]).unwrap();
        let (header, raw) = encode_displacement(&tile).unwrap();
        assert_eq!(decode_displacement(&header, &raw).unwrap(), tile, "side 2");
    }
}
