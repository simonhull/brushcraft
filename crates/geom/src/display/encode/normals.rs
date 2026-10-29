//! Normals and roughness: each vertex's unit normal, as octahedral coordinates or as a residual
//! from the normal of the tile's own decoded surface, then one roughness byte per vertex,
//! delta-coded.
//!
//! A vertex two tiles share must decode to the same bits in both, as positions do: its mean
//! normal is the same in both (the mip is computed once, as the owner computes it), so it is
//! stored the same way in both, octahedral at [`BORDER_BITS`], never predicted (each tile's
//! prediction differs there).

use super::bytes::{
    Reader, deltas_u8, deltas_u16, put_u16_planes, take_u16_planes, undeltas_u8, undeltas_u16,
    unzigzag, zigzag,
};
use super::pow2_ceil;
use crate::display::DisplayError;
use crate::{Grid, Vec3};

/// How a display tile's normals and roughness are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalEncoding {
    /// Octahedral coordinates, two signed fractions per vertex of this many bits each (2–16).
    Octahedral(u8),
    /// Interior vertices as a residual from the normal of the tile's decoded surface there (its
    /// central differences), in that normal's tangent plane, two signed components of this many
    /// bits each (2–16) with one power-of-two step per tile. Border vertices octahedral at
    /// 12 bits. A tile whose normals stray more than 60° from its surface's is stored octahedral
    /// at 12 bits throughout.
    Predicted(u8),
}

/// The octahedral precision of the vertices [`NormalEncoding::Predicted`] doesn't predict: a
/// tile's border, and every vertex of a tile it can't predict. Its largest error is about 0.06°.
const BORDER_BITS: u8 = 12;

/// A predicted tile's first byte: interior predicted.
const PREDICTED: u8 = 0;
/// A predicted tile's first byte: every vertex octahedral at [`BORDER_BITS`].
const FALLBACK: u8 = 1;

/// The least cosine between a normal and its prediction that [`NormalEncoding::Predicted`]
/// codes as a residual: 60°.
const PREDICTABLE: f32 = 0.5;

/// The unit normal and roughness byte of each of `means`: the normalized mean (`(0, 0, 1)` where
/// it is too short to normalize), and its length, clamped to `0..=1`, in 255ths.
pub(in crate::display) fn shade(means: &[Vec3]) -> (Vec<Vec3>, Vec<u8>) {
    means
        .iter()
        .map(|m| {
            let unit = m.normalized().unwrap_or(UP);
            let len = m.length().clamp(0.0, 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a value in 0..=255 after the clamp, rounded"
            )]
            let rough = (len * 255.0).round() as u8;
            (unit, rough)
        })
        .unzip()
}

/// The mean normal a decoded unit normal and roughness byte stand for, as the next coarser
/// level's mip reads it: the normal scaled to the roughness's length.
pub(in crate::display) fn mean_of(unit: Vec3, roughness: u8) -> Vec3 {
    unit * (f32::from(roughness) / 255.0)
}

/// The fallback normal where none can be computed.
const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

impl NormalEncoding {
    /// The tile's `normals` (unit) and `roughness`, one each per vertex of `positions` (its
    /// decoded positions, which [`NormalEncoding::Predicted`] predicts from), encoded.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if the lengths don't match `positions` or the bits are out of
    /// range; [`DisplayError::NotFinite`] for a normal that isn't finite.
    pub fn encode(
        self,
        normals: &[Vec3],
        roughness: &[u8],
        positions: &Grid,
    ) -> Result<Vec<u8>, DisplayError> {
        let count = positions.as_slice().len();
        if normals.len() != count || roughness.len() != count || !(2..=16).contains(&self.bits()) {
            return Err(DisplayError::Shape);
        }
        if !normals.iter().all(|n| n.is_finite()) {
            return Err(DisplayError::NotFinite);
        }
        let mut out = Vec::new();
        match self {
            Self::Octahedral(bits) => put_octahedral(&mut out, normals, bits),
            Self::Predicted(bits) => encode_predicted(&mut out, normals, positions, bits)?,
        }
        // Delta-coded along rows. Measured on the benchmark project, raw roughness compresses
        // smaller alone (0.92 % of the file against 0.99 %) but not beside the normals in one
        // zstd frame (+53.88 % in all against +53.87 %), which is how the store writes it.
        out.extend(deltas_u8(roughness));
        Ok(out)
    }

    /// The unit normals and roughness bytes of a tile with decoded `positions`, from `bytes`
    /// that [`NormalEncoding::encode`] gave.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if the bytes don't hold a tile of that many vertices, or the
    /// bits are out of range.
    pub fn decode(
        self,
        positions: &Grid,
        bytes: &[u8],
    ) -> Result<(Vec<Vec3>, Vec<u8>), DisplayError> {
        if !(2..=16).contains(&self.bits()) {
            return Err(DisplayError::Malformed);
        }
        let count = positions.as_slice().len();
        let mut reader = Reader::new(bytes);
        let normals = match self {
            Self::Octahedral(bits) => take_octahedral(&mut reader, count, bits)?,
            Self::Predicted(bits) => decode_predicted(&mut reader, positions, bits)?,
        };
        let roughness = undeltas_u8(reader.take(count)?);
        reader.finish()?;
        Ok((normals, roughness))
    }

    /// Bits per component.
    const fn bits(self) -> u8 {
        match self {
            Self::Octahedral(bits) | Self::Predicted(bits) => bits,
        }
    }
}

/// The largest magnitude a signed `bits`-bit component holds: `2^(bits − 1) − 1`.
pub(in crate::display) fn levels(bits: u8) -> f32 {
    f32::from((1_u16 << (bits.clamp(2, 16) - 1)) - 1)
}

/// `c` in `[−1, 1]` as a signed `bits`-bit fraction.
pub(in crate::display) fn quantize_unit(c: f32, bits: u8) -> i16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "within ±(2^15 − 1) after the clamp, rounded"
    )]
    let q = (c.clamp(-1.0, 1.0) * levels(bits)).round() as i16;
    q
}

/// Picks one coordinate of a pair.
type Component = fn(&(f32, f32)) -> f32;

/// Writes `normals` as octahedral coordinates of `bits` bits, each plane delta-coded in the
/// width it is stored in: one byte per value up to 8 bits, two byte planes past.
fn put_octahedral(out: &mut Vec<u8>, normals: &[Vec3], bits: u8) {
    let coordinates: Vec<(f32, f32)> = normals.iter().map(|&n| octahedral(n)).collect();
    let pick: [Component; 2] = [|c| c.0, |c| c.1];
    for component in pick {
        let q = coordinates
            .iter()
            .map(|c| quantize_unit(component(c), bits).to_le_bytes());
        if bits <= 8 {
            let bytes: Vec<u8> = q.map(|[low, _]| low).collect();
            out.extend(deltas_u8(&bytes));
        } else {
            let words: Vec<u16> = q.map(u16::from_le_bytes).collect();
            put_u16_planes(out, &deltas_u16(&words));
        }
    }
}

/// Reads `count` normals [`put_octahedral`] wrote.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the reader holds fewer.
fn take_octahedral(
    reader: &mut Reader<'_>,
    count: usize,
    bits: u8,
) -> Result<Vec<Vec3>, DisplayError> {
    let mut plane = || -> Result<Vec<f32>, DisplayError> {
        let scale = levels(bits);
        Ok(if bits <= 8 {
            undeltas_u8(reader.take(count)?)
                .into_iter()
                .map(|b| f32::from(i8::from_le_bytes([b])) / scale)
                .collect()
        } else {
            undeltas_u16(&take_u16_planes(reader, count)?)
                .into_iter()
                .map(|w| f32::from(i16::from_le_bytes(w.to_le_bytes())) / scale)
                .collect()
        })
    };
    let us = plane()?;
    let vs = plane()?;
    Ok(us
        .into_iter()
        .zip(vs)
        .map(|(u, v)| from_octahedral(u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)))
        .collect())
}

/// Whether `(i, j)` lies on the border of a tile with `side` vertices per side.
const fn on_border(side: u32, i: u32, j: u32) -> bool {
    i == 0 || j == 0 || i + 1 == side || j + 1 == side
}

/// The vertex coordinates of a tile with `side` vertices per side, row-major.
fn cells(side: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..side).flat_map(move |j| (0..side).map(move |i| (i, j)))
}

/// The prediction at interior vertex `(col, row)` of `positions`, with a tangent basis of it.
///
/// The prediction is the unit normal of the vertex's central differences,
/// `(p(i+1, j) − p(i−1, j)) × (p(i, j+1) − p(i, j−1))` (`(0, 0, 1)` if degenerate); the result
/// is `[tangent, bitangent, normal]`.
fn prediction(positions: &Grid, col: u32, row: u32) -> Option<[Vec3; 3]> {
    let at = |x: u32, y: u32| positions.get(x, y);
    let du = at(col + 1, row)? - at(col.checked_sub(1)?, row)?;
    let dv = at(col, row + 1)? - at(col, row.checked_sub(1)?)?;
    let normal = du.cross(dv).normalized().unwrap_or(UP);
    // Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017): branch-free but for
    // the sign, and the same bits wherever it is computed.
    let sign = if normal.z >= 0.0 { 1.0 } else { -1.0 };
    let scale = -1.0 / (sign + normal.z);
    let cross = normal.x * normal.y * scale;
    let tangent = Vec3::new(
        1.0 + sign * normal.x * normal.x * scale,
        sign * cross,
        -sign * normal.x,
    );
    let bitangent = Vec3::new(cross, sign + normal.y * normal.y * scale, -normal.y);
    Some([tangent, bitangent, normal])
}

/// The unit normal the residual `(along, across)` stands for in the tangent plane of `basis`.
fn from_residual([tangent, bitangent, normal]: [Vec3; 3], along: f32, across: f32) -> Vec3 {
    let height = (1.0 - along * along - across * across).max(0.0).sqrt();
    (tangent * along + bitangent * across + normal * height)
        .normalized()
        .unwrap_or(normal)
}

/// Writes `normals` of a tile with decoded `positions` as [`NormalEncoding::Predicted`] does.
///
/// # Errors
///
/// [`DisplayError::Shape`] if `normals` doesn't match `positions`.
fn encode_predicted(
    out: &mut Vec<u8>,
    normals: &[Vec3],
    positions: &Grid,
    bits: u8,
) -> Result<(), DisplayError> {
    let side = positions.side();
    let mut border = Vec::new();
    let mut residuals = Vec::new();
    let mut predictable = true;
    for ((i, j), &m) in cells(side).zip(normals) {
        if on_border(side, i, j) {
            border.push(m);
            continue;
        }
        let basis = prediction(positions, i, j).ok_or(DisplayError::Shape)?;
        let [t, b, n] = basis;
        predictable &= m.dot(n) >= PREDICTABLE;
        residuals.push((m.dot(t), m.dot(b)));
    }
    if !predictable {
        out.push(FALLBACK);
        put_octahedral(out, normals, BORDER_BITS);
        return Ok(());
    }
    out.push(PREDICTED);
    put_octahedral(out, &border, BORDER_BITS);
    let peak = residuals
        .iter()
        .fold(0.0_f32, |p, &(x, y)| p.max(x.abs()).max(y.abs()));
    let step = pow2_ceil(peak / levels(bits)).unwrap_or(1.0);
    out.extend_from_slice(&step.to_le_bytes());
    let scale = levels(bits);
    let quantize = |r: f32| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "within ±(2^15 − 1) after the clamp, rounded"
        )]
        let q = (r / step).round().clamp(-scale, scale) as i16;
        q
    };
    for pick in [|r: &(f32, f32)| r.0, |r: &(f32, f32)| r.1] {
        let q: Vec<i16> = residuals.iter().map(|r| quantize(pick(r))).collect();
        put_zigzag(out, &q, bits);
    }
    Ok(())
}

/// Reads the normals [`encode_predicted`] wrote for a tile with decoded `positions`.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the bytes don't hold such a tile.
fn decode_predicted(
    reader: &mut Reader<'_>,
    positions: &Grid,
    bits: u8,
) -> Result<Vec<Vec3>, DisplayError> {
    let side = positions.side();
    let count = positions.as_slice().len();
    match reader.u8()? {
        FALLBACK => return take_octahedral(reader, count, BORDER_BITS),
        PREDICTED => {}
        _ => return Err(DisplayError::Malformed),
    }
    let borders = cells(side).filter(|&(i, j)| on_border(side, i, j)).count();
    let interior = count - borders;
    let border = take_octahedral(reader, borders, BORDER_BITS)?;
    let step = reader.f32()?;
    if !(step.is_finite() && step > 0.0) {
        return Err(DisplayError::Malformed);
    }
    let xs = take_zigzag(reader, interior, bits)?;
    let ys = take_zigzag(reader, interior, bits)?;
    let (mut border, mut inner) = (border.into_iter(), xs.into_iter().zip(ys));
    cells(side)
        .map(|(i, j)| {
            if on_border(side, i, j) {
                return border.next().ok_or(DisplayError::Malformed);
            }
            let (x, y) = inner.next().ok_or(DisplayError::Malformed)?;
            let basis = prediction(positions, i, j).ok_or(DisplayError::Malformed)?;
            Ok(from_residual(
                basis,
                f32::from(x) * step,
                f32::from(y) * step,
            ))
        })
        .collect()
}

/// Writes signed `values` zigzagged, so small magnitudes of either sign are small: one byte
/// plane if `bits` fit a byte, else two.
fn put_zigzag(out: &mut Vec<u8>, values: &[i16], bits: u8) {
    let folded: Vec<u16> = values
        .iter()
        .map(|&q| u16::try_from(zigzag(i32::from(q))).unwrap_or(u16::MAX))
        .collect();
    if bits <= 8 {
        out.extend(folded.iter().map(|&z| u8::try_from(z).unwrap_or(u8::MAX)));
    } else {
        put_u16_planes(out, &folded);
    }
}

/// Reads `count` values [`put_zigzag`] wrote.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the reader holds fewer.
fn take_zigzag(reader: &mut Reader<'_>, count: usize, bits: u8) -> Result<Vec<i16>, DisplayError> {
    let folded: Vec<u16> = if bits <= 8 {
        reader.take(count)?.iter().map(|&b| u16::from(b)).collect()
    } else {
        take_u16_planes(reader, count)?
    };
    folded
        .into_iter()
        .map(|z| i16::try_from(unzigzag(u32::from(z))).map_err(|_overflow| DisplayError::Malformed))
        .collect()
}

/// Unit vector `n` mapped onto the octahedron and unfolded into `[−1, 1]²`.
pub(in crate::display) fn octahedral(n: Vec3) -> (f32, f32) {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    if l1 <= 0.0 {
        return (0.0, 0.0);
    }
    let (u, v) = (n.x / l1, n.y / l1);
    if n.z >= 0.0 {
        (u, v)
    } else {
        ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v))
    }
}

/// [`octahedral`]'s inverse: the unit vector at `(u, v)`.
pub(in crate::display) fn from_octahedral(across: f32, up: f32) -> Vec3 {
    let height = 1.0 - across.abs() - up.abs();
    let (x, y) = if height >= 0.0 {
        (across, up)
    } else {
        (
            (1.0 - up.abs()) * sign(across),
            (1.0 - across.abs()) * sign(up),
        )
    };
    Vec3::new(x, y, height).normalized().unwrap_or(UP)
}

/// `1` for zero and up, `−1` below: the octahedral fold's sign (`f32::signum` gives `−1` for
/// `−0.0`, which would fold a vector on the equator the wrong way).
fn sign(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}
