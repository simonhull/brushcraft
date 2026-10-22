//! The displacement channel's payload: header `side: u16, scale: f32`, body planar delta-coded
//! interior `i16`s then exact border `f32`s, compressed with zstd.

use geom::{CorrectionError, CorrectionTile, Vec3};
use thiserror::Error;

use crate::bytes::Reader;

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

/// Decodes [`encode_displacement`]'s output.
///
/// # Errors
///
/// [`CodecError::Size`] if the header or body length doesn't match the side; [`CodecError::Tile`]
/// if the values are invalid (non-finite border, bad scale).
pub(crate) fn decode_displacement(
    header: &[u8],
    body: &[u8],
) -> Result<CorrectionTile, CodecError> {
    let mut h = Reader::new(header);
    let side = h.u16().ok_or(CodecError::Size)?;
    let scale = h.f32().ok_or(CodecError::Size)?;
    if !h.rest().is_empty() {
        return Err(CodecError::Size);
    }
    // Validates `side >= 2`; the counts follow from it.
    let empty = CorrectionTile::zero(u32::from(side))?;
    let (n, ring) = (empty.interior().len(), empty.border().len());
    if body.len() != n * 6 + ring * 12 {
        return Err(CodecError::Size);
    }
    let mut r = Reader::new(body);
    let (ix, iy, iz) = (
        delta_plane(&mut r, n)?,
        delta_plane(&mut r, n)?,
        delta_plane(&mut r, n)?,
    );
    let (bx, by, bz) = (
        f32_plane(&mut r, ring)?,
        f32_plane(&mut r, ring)?,
        f32_plane(&mut r, ring)?,
    );
    let interior = ix
        .into_iter()
        .zip(iy)
        .zip(iz)
        .map(|((x, y), z)| [x, y, z])
        .collect();
    let border = bx
        .into_iter()
        .zip(by)
        .zip(bz)
        .map(|((x, y), z)| Vec3::new(x, y, z))
        .collect();
    Ok(CorrectionTile::from_parts(
        u32::from(side),
        scale,
        interior,
        border,
    )?)
}

/// `n` delta-coded `i16`s, summed back to values.
///
/// # Errors
///
/// [`CodecError::Size`] if `r` runs out.
fn delta_plane(r: &mut Reader<'_>, n: usize) -> Result<Vec<i16>, CodecError> {
    let mut prev = 0i16;
    (0..n)
        .map(|_| {
            prev = prev.wrapping_add(r.i16().ok_or(CodecError::Size)?);
            Ok(prev)
        })
        .collect()
}

/// `n` exact `f32`s.
///
/// # Errors
///
/// [`CodecError::Size`] if `r` runs out.
fn f32_plane(r: &mut Reader<'_>, n: usize) -> Result<Vec<f32>, CodecError> {
    (0..n).map(|_| r.f32().ok_or(CodecError::Size)).collect()
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
}
