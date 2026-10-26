//! How display tiles are stored: the position codecs measured against each other, and the
//! normals' octahedral encoding.
//!
//! Decision 4 of the display-layer plan asks of the stored positions, in order: border vertices
//! decode to exactly the same `f32`s in every tile holding them (seams exact without reading a
//! neighbour); a tile decodes on its own (no ancestor, no base level); the stored `e` comes from
//! the decoded values (which [`build_face_with`](super::build_face_with) guarantees for any
//! codec); and the file grows by no more than the design note's estimate. The candidates:
//!
//! - [`Interpolating`]: each vertex coded against the midpoint of its two parents in the
//!   triangulation (its edge's ends, or its quad's diagonal), coarsest first, so a smooth surface
//!   leaves small residuals. Border vertices are stored losslessly (their bits against their
//!   prediction's), interior ones quantized with one power-of-two step per tile ([`Precision`]).
//!   Decodes on its own; borders exact by construction.
//! - [`FixedPoint`]: interior vertices as 16-bit fixed point in the tile's bounding box, border
//!   vertices as [`Interpolating`] stores them.
//! - [`BaseOffsets`]: the design note's first proposal: offsets from the base level's own position,
//!   in its tangent frame, quantized as correction tiles are. It needs the base level to decode, so
//!   it fails the second constraint by construction; it is kept for the study.
//!
//! # The choice
//!
//! Measured on the benchmark project (all 2,400 faces, `brushcraft-bench display-study`; the
//! table is in the docs repository's `research/platform/m1a2-display-encoding.md`), as growth of
//! the project file: [`Interpolating`] at `Precision::QuadFraction(6)` (a step of about a
//! 64th of the level's quad edge) with [`NormalEncoding::Oct8`], +38.6 % (positions 24.5 %,
//! normals and roughness 13.6 %). It meets the first three constraints by construction; its
//! quantization stays under 1.4 % of a quad edge and raises the median `e` by about 2 %. A
//! coarser step (a 16th) saves 7 points more but raises the median `e` by about 20 %, which the
//! renderer pays in triangles. 16-bit normals cost 41.4 % on their own, which no position codec
//! leaves room for under the 65 % budget without that coarser step; 8-bit octahedral normals
//! are off by at most 0.96°.
//!
//! Integer streams are written as byte planes (every value's low byte, then the next, …), which
//! the store's zstd compresses far better than interleaved bytes: small residuals leave the high
//! planes zero.

use self::bytes::{
    Reader, put_u16_planes, put_u32_planes, take_u16_planes, take_u32_planes, unzigzag, zigzag,
};
pub use self::normals::NormalEncoding;
pub(super) use self::normals::shade;
#[cfg(any(test, feature = "display-study"))]
pub use self::study::{BaseOffsets, FixedPoint};
use super::DisplayError;
use crate::{Grid, TileKey, Vec3};

mod bytes;
mod normals;
#[cfg(any(test, feature = "display-study"))]
mod study;

/// The position codec [`build_face`](super::build_face) and [`DisplayTile::decode`](
/// super::DisplayTile::decode) use.
pub(super) const CHOSEN: Interpolating = Interpolating::new(Precision::QuadFraction(6));

/// The normal encoding [`build_face`](super::build_face) and [`DisplayTile::decode`](
/// super::DisplayTile::decode) use.
pub(super) const CHOSEN_NORMALS: NormalEncoding = NormalEncoding::Oct8;

/// How a display tile's positions are stored.
///
/// `encode` then `decode` gives the positions the tile draws, which the bound is computed from;
/// `decode` of the same bytes always gives the same bits.
pub trait DisplayCodec {
    /// The positions of tile `key`, row-major `side²` (its side), encoded.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if `positions` isn't the key's side;
    /// [`DisplayError::NotFinite`] for a position that isn't finite;
    /// [`DisplayError::TooLarge`] if a value is out of the codec's range.
    fn encode(&self, key: &TileKey, positions: &Grid) -> Result<Vec<u8>, DisplayError>;

    /// The positions of tile `key` from `bytes` that [`DisplayCodec::encode`] gave.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Malformed`] if the bytes don't hold a tile of the key's side;
    /// [`DisplayError::NotFinite`] if a position decodes to a value that isn't finite.
    fn decode(&self, key: &TileKey, bytes: &[u8]) -> Result<Grid, DisplayError>;
}

/// How finely [`Interpolating`] quantizes interior residuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Not at all: every vertex stored losslessly, as border vertices are. The reference the
    /// lossy precisions are measured against.
    Lossless,
    /// The finest power-of-two step that keeps every residual in an `i16`: the tile's largest
    /// residual sets it, as a correction tile's largest value sets its scale.
    Bits16,
    /// A power-of-two step of about `2⁻ⁿ` of the tile's mean quad edge (the largest power of two
    /// at or below it), or [`Precision::Bits16`]'s if that is coarser.
    QuadFraction(u8),
}

/// Positions coded against the midpoint of their parents in the triangulation, coarsest first
/// (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interpolating {
    /// How interior residuals are quantized.
    precision: Precision,
}

/// The positions' and normals' parts joined: a `u32` length of `positions`, then both.
///
/// # Errors
///
/// [`DisplayError::TooLarge`] if `positions` is longer than a `u32` counts.
pub(super) fn join_parts(positions: &[u8], normals: &[u8]) -> Result<Vec<u8>, DisplayError> {
    let len = u32::try_from(positions.len()).map_err(|_too_long| DisplayError::TooLarge)?;
    let mut out = Vec::with_capacity(4 + positions.len() + normals.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(positions);
    out.extend_from_slice(normals);
    Ok(out)
}

/// [`join_parts`]'s two parts.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the length runs past the bytes.
pub(super) fn split_parts(bytes: &[u8]) -> Result<(&[u8], &[u8]), DisplayError> {
    let mut reader = Reader::new(bytes);
    let len = usize::try_from(reader.u32()?).map_err(|_too_long| DisplayError::Malformed)?;
    let positions = reader.take(len)?;
    Ok((positions, reader.rest()))
}

/// One step of the refinement order: vertex `at`, predicted from the midpoint of `parents`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Refined {
    /// The vertex, `(i, j)` in the tile.
    at: (u32, u32),
    /// Its two parents in the triangulation one sub-level coarser: its edge's ends, or the ends
    /// of its quad's `(i, j)`–`(i + 1, j + 1)` diagonal.
    parents: [(u32, u32); 2],
    /// Whether it lies on the tile's border (and so do its parents).
    border: bool,
}

/// The four corners of a tile with `quads` quads per side.
const fn corners(quads: u32) -> [(u32, u32); 4] {
    [(0, 0), (quads, 0), (0, quads), (quads, quads)]
}

/// Every vertex of a tile with `quads` quads per side (a power of two) but its corners, in
/// refinement order: sub-level by sub-level, coarsest first, so each vertex's parents come
/// before it.
fn refinement(quads: u32) -> Vec<Refined> {
    let mut order = Vec::new();
    let mut span = quads;
    while span >= 2 {
        let half = span / 2;
        for j in (0..=quads).step_by(usize::try_from(half).unwrap_or(usize::MAX)) {
            for i in (0..=quads).step_by(usize::try_from(half).unwrap_or(usize::MAX)) {
                let (odd_i, odd_j) = (!i.is_multiple_of(span), !j.is_multiple_of(span));
                let parents = match (odd_i, odd_j) {
                    (false, false) => continue,
                    (true, false) => [(i - half, j), (i + half, j)],
                    (false, true) => [(i, j - half), (i, j + half)],
                    (true, true) => [(i - half, j - half), (i + half, j + half)],
                };
                let border = i == 0 || j == 0 || i == quads || j == quads;
                order.push(Refined {
                    at: (i, j),
                    parents,
                    border,
                });
            }
        }
        span = half;
    }
    order
}

/// The tile's quads per side, checked against `key` and that it is a power of two.
///
/// # Errors
///
/// [`DisplayError::Shape`] otherwise.
fn tile_quads(key: &TileKey, side: u32) -> Result<u32, DisplayError> {
    let quads = side.checked_sub(1).ok_or(DisplayError::Shape)?;
    if side == key.side() && quads.is_power_of_two() {
        Ok(quads)
    } else {
        Err(DisplayError::Shape)
    }
}

/// The midpoint of `a` and `b` as the renderer computes it: `(a + b) × ½`, in `f32`.
fn mid(a: Vec3, b: Vec3) -> Vec3 {
    (a + b) * 0.5
}

/// `v`'s components.
const fn parts(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// `f` of each axis's components of `a` and `b`.
fn per_axis(a: [f32; 3], b: [f32; 3], f: impl Fn(f32, f32) -> f32) -> [f32; 3] {
    let ([a0, a1, a2], [b0, b1, b2]) = (a, b);
    [f(a0, b0), f(a1, b1), f(a2, b2)]
}

/// The vertex with components `c`.
const fn join([x, y, z]: [f32; 3]) -> Vec3 {
    Vec3::new(x, y, z)
}

/// A tile's positions while it is coded or decoded: row-major, written as vertices are reached.
struct Tile {
    /// Vertices per side.
    side: u32,
    /// The positions.
    data: Vec<Vec3>,
}

impl Tile {
    /// A tile of `side²` positions copied from `grid`.
    fn from_grid(grid: &Grid) -> Self {
        Self {
            side: grid.side(),
            data: grid.as_slice().to_vec(),
        }
    }

    /// A tile of `side²` zero positions, to decode into.
    fn zero(side: u32) -> Self {
        let count = usize::try_from(u64::from(side) * u64::from(side)).unwrap_or(0);
        Self {
            side,
            data: vec![Vec3::ZERO; count],
        }
    }

    /// The index of `(i, j)`.
    fn index(&self, (i, j): (u32, u32)) -> usize {
        usize::try_from(u64::from(j) * u64::from(self.side) + u64::from(i)).unwrap_or(usize::MAX)
    }

    /// The position at `(i, j)`.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] outside the tile.
    fn get(&self, at: (u32, u32)) -> Result<Vec3, DisplayError> {
        self.data
            .get(self.index(at))
            .copied()
            .ok_or(DisplayError::Shape)
    }

    /// Writes the position at `(i, j)`.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] outside the tile.
    fn set(&mut self, at: (u32, u32), v: Vec3) -> Result<(), DisplayError> {
        let index = self.index(at);
        *self.data.get_mut(index).ok_or(DisplayError::Shape)? = v;
        Ok(())
    }

    /// The prediction for `r`: the midpoint of its parents as held now.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if a parent is outside the tile.
    fn predict(&self, r: &Refined) -> Result<Vec3, DisplayError> {
        let [a, b] = r.parents;
        Ok(mid(self.get(a)?, self.get(b)?))
    }

    /// The tile as a grid.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] never: the tile holds `side²` positions.
    fn into_grid(self) -> Result<Grid, DisplayError> {
        Ok(Grid::new(self.side, self.data)?)
    }
}

/// `actual`'s bits against `predicted`'s, per component, zigzagged: lossless, and small when
/// they are close.
fn bit_residual(actual: Vec3, predicted: Vec3) -> [u32; 3] {
    let ([a0, a1, a2], [p0, p1, p2]) = (actual.to_bits(), predicted.to_bits());
    let residual = |a: u32, p: u32| zigzag(i32::from_le_bytes(a.wrapping_sub(p).to_le_bytes()));
    [residual(a0, p0), residual(a1, p1), residual(a2, p2)]
}

/// [`bit_residual`]'s inverse: the value whose bits are `predicted`'s plus `residual`.
///
/// # Errors
///
/// [`DisplayError::NotFinite`] if the value isn't finite.
fn apply_bit_residual(predicted: Vec3, residual: [u32; 3]) -> Result<Vec3, DisplayError> {
    let ([p0, p1, p2], [r0, r1, r2]) = (predicted.to_bits(), residual);
    let add = |p: u32, r: u32| {
        f32::from_bits(p.wrapping_add(u32::from_le_bytes(unzigzag(r).to_le_bytes())))
    };
    let v = Vec3::new(add(p0, r0), add(p1, r1), add(p2, r2));
    if v.is_finite() {
        Ok(v)
    } else {
        Err(DisplayError::NotFinite)
    }
}

/// Writes a tile's corners (raw bits) and border vertices (lossless, against their prediction
/// along the border) to `out`, from `tile`, whose other vertices aren't read.
///
/// # Errors
///
/// [`DisplayError::Shape`] if `order` doesn't fit the tile.
fn encode_border(
    tile: &Tile,
    quads: u32,
    order: &[Refined],
    out: &mut Vec<u8>,
) -> Result<(), DisplayError> {
    let mut corner_bits = Vec::with_capacity(12);
    for at in corners(quads) {
        corner_bits.extend(tile.get(at)?.to_bits());
    }
    put_u32_planes(out, &corner_bits);
    let mut residuals = Vec::new();
    for r in order.iter().filter(|r| r.border) {
        residuals.extend(bit_residual(tile.get(r.at)?, tile.predict(r)?));
    }
    put_u32_planes(out, &residuals);
    Ok(())
}

/// Reads what [`encode_border`] wrote into `tile`'s corners and border.
///
/// # Errors
///
/// [`DisplayError::Malformed`] if the bytes run short; [`DisplayError::NotFinite`] for a value
/// that isn't finite.
fn decode_border(
    tile: &mut Tile,
    quads: u32,
    order: &[Refined],
    reader: &mut Reader<'_>,
) -> Result<(), DisplayError> {
    let corner_bits = take_u32_planes(reader, 12)?;
    let (corner_bits, _) = corner_bits.as_chunks::<3>();
    for (at, bits) in corners(quads).into_iter().zip(corner_bits) {
        let v = join(bits.map(f32::from_bits));
        if !v.is_finite() {
            return Err(DisplayError::NotFinite);
        }
        tile.set(at, v)?;
    }
    let count = order.iter().filter(|r| r.border).count();
    let residuals = take_u32_planes(reader, count * 3)?;
    let (residuals, _) = residuals.as_chunks::<3>();
    for (r, &res) in order.iter().filter(|r| r.border).zip(residuals) {
        let v = apply_bit_residual(tile.predict(r)?, res)?;
        tile.set(r.at, v)?;
    }
    Ok(())
}

/// The largest power of two at or above positive, finite `x` (`f32::MIN_POSITIVE` for a
/// subnormal), or `None` past the largest finite one.
fn pow2_ceil(x: f32) -> Option<f32> {
    if !(x.is_finite() && x > 0.0) {
        return None;
    }
    let bits = x.to_bits();
    let exponent = bits >> 23;
    if exponent == 0 {
        return Some(f32::MIN_POSITIVE);
    }
    if bits.trailing_zeros() >= 23 {
        return Some(x);
    }
    let up = f32::from_bits((exponent + 1) << 23);
    up.is_finite().then_some(up)
}

/// The largest power of two at or below positive, finite `x` (`f32::MIN_POSITIVE` for a
/// subnormal).
fn pow2_floor(x: f32) -> Option<f32> {
    if !(x.is_finite() && x > 0.0) {
        return None;
    }
    let exponent = x.to_bits() >> 23;
    Some(if exponent == 0 {
        f32::MIN_POSITIVE
    } else {
        f32::from_bits(exponent << 23)
    })
}

/// The mean length of a tile's row and column edges: its level's quad edge there.
fn mean_edge(tile: &Tile) -> f32 {
    let (mut sum, mut count) = (0.0_f64, 0_u32);
    for j in 0..tile.side {
        for i in 0..tile.side {
            let Ok(p) = tile.get((i, j)) else { continue };
            for q in [(i + 1, j), (i, j + 1)] {
                if q.0 < tile.side
                    && q.1 < tile.side
                    && let Ok(q) = tile.get(q)
                {
                    sum += f64::from((q - p).length());
                    count += 1;
                }
            }
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a mean edge length, well inside f32's range"
    )]
    let mean = (sum / f64::from(count.max(1))) as f32;
    mean
}

/// The largest residual component quantization will be asked to hold.
const Q_MAX: f32 = 32767.0;

/// `r` in steps of `step`, rounded, if within an `i16`'s symmetric range.
fn quantize(r: f32, step: f32) -> Option<i16> {
    let q = (r / step).round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "checked within ±32767 first"
    )]
    (q.abs() <= Q_MAX).then_some(q as i16)
}

/// Byte marking [`Precision::Lossless`] interiors in [`Interpolating`]'s header.
const LOSSLESS: u8 = 0;
/// Byte marking quantized interiors in [`Interpolating`]'s header.
const QUANTIZED: u8 = 1;

impl Interpolating {
    /// The codec at `precision`.
    #[must_use]
    pub const fn new(precision: Precision) -> Self {
        Self { precision }
    }

    /// The quantization step for `tile`'s interior, from the residuals against its true
    /// positions (the closed loop of [`Interpolating::encode_interior`] may need it doubled).
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if `order` doesn't fit the tile.
    fn step(self, tile: &Tile, order: &[Refined]) -> Result<f32, DisplayError> {
        let mut peak = 0.0_f32;
        for r in order.iter().filter(|r| !r.border) {
            let residual = tile.get(r.at)? - tile.predict(r)?;
            peak = parts(residual)
                .into_iter()
                .fold(peak, |a, c| a.max(c.abs()));
        }
        let fitting = pow2_ceil(peak / Q_MAX);
        let step = match self.precision {
            Precision::Lossless => None,
            Precision::Bits16 => fitting,
            Precision::QuadFraction(n) => {
                // 2⁻ⁿ as an `f32` (its biased exponent `127 − n`), down to the least normal.
                let scale = f32::from_bits(127_u32.saturating_sub(u32::from(n)).max(1) << 23);
                let fraction = mean_edge(tile) * scale;
                match (pow2_floor(fraction), fitting) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                }
            }
        };
        Ok(step.unwrap_or(1.0))
    }

    /// Quantizes the interior of `tile` in refinement order against the decoded parents (closed
    /// loop), writing the decoded values back into `tile`. `None` if a residual overflows the
    /// step.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if `order` doesn't fit the tile; [`DisplayError::TooLarge`] if a
    /// decoded value isn't finite.
    fn encode_interior(
        tile: &mut Tile,
        order: &[Refined],
        step: f32,
    ) -> Result<Option<Vec<u16>>, DisplayError> {
        let mut out = Vec::new();
        for r in order.iter().filter(|r| !r.border) {
            let predicted = tile.predict(r)?;
            let residual = parts(tile.get(r.at)? - predicted);
            let mut decoded = parts(predicted);
            for (d, c) in decoded.iter_mut().zip(residual) {
                let Some(q) = quantize(c, step) else {
                    return Ok(None);
                };
                *d += f32::from(q) * step;
                out.push(u16::from_le_bytes(q.to_le_bytes()));
            }
            let decoded = join(decoded);
            if !decoded.is_finite() {
                return Err(DisplayError::TooLarge);
            }
            tile.set(r.at, decoded)?;
        }
        Ok(Some(out))
    }
}

impl DisplayCodec for Interpolating {
    fn encode(&self, key: &TileKey, positions: &Grid) -> Result<Vec<u8>, DisplayError> {
        let quads = tile_quads(key, positions.side())?;
        if !positions.as_slice().iter().all(|v| v.is_finite()) {
            return Err(DisplayError::NotFinite);
        }
        let order = refinement(quads);
        let original = Tile::from_grid(positions);
        let mut out = Vec::new();
        if self.precision == Precision::Lossless {
            out.push(LOSSLESS);
            encode_border(&original, quads, &order, &mut out)?;
            let mut residuals = Vec::new();
            for r in order.iter().filter(|r| !r.border) {
                residuals.extend(bit_residual(original.get(r.at)?, original.predict(r)?));
            }
            put_u32_planes(&mut out, &residuals);
            return Ok(out);
        }
        let mut step = self.step(&original, &order)?;
        // The closed loop's residuals can exceed the open loop's by a step's rounding: double
        // the step until they fit (once at most, in practice).
        for _ in 0..8 {
            let mut tile = Tile::from_grid(positions);
            if let Some(interior) = Self::encode_interior(&mut tile, &order, step)? {
                out.push(QUANTIZED);
                out.extend_from_slice(&step.to_le_bytes());
                encode_border(&original, quads, &order, &mut out)?;
                put_u16_planes(&mut out, &interior);
                return Ok(out);
            }
            step *= 2.0;
        }
        Err(DisplayError::TooLarge)
    }

    fn decode(&self, key: &TileKey, bytes: &[u8]) -> Result<Grid, DisplayError> {
        let side = key.side();
        let quads = tile_quads(key, side)?;
        let order = refinement(quads);
        let mut reader = Reader::new(bytes);
        let mode = reader.u8()?;
        let step = match mode {
            LOSSLESS => None,
            QUANTIZED => {
                let step = reader.f32()?;
                if !(step.is_finite() && step > 0.0) {
                    return Err(DisplayError::Malformed);
                }
                Some(step)
            }
            _ => return Err(DisplayError::Malformed),
        };
        let mut tile = Tile::zero(side);
        decode_border(&mut tile, quads, &order, &mut reader)?;
        let interior: Vec<&Refined> = order.iter().filter(|r| !r.border).collect();
        if let Some(step) = step {
            let values = take_u16_planes(&mut reader, interior.len() * 3)?;
            let (values, _) = values.as_chunks::<3>();
            for (r, q) in interior.iter().zip(values) {
                let steps = q.map(|q| f32::from(i16::from_le_bytes(q.to_le_bytes())) * step);
                let decoded = join(per_axis(parts(tile.predict(r)?), steps, |p, d| p + d));
                if !decoded.is_finite() {
                    return Err(DisplayError::NotFinite);
                }
                tile.set(r.at, decoded)?;
            }
        } else {
            let values = take_u32_planes(&mut reader, interior.len() * 3)?;
            let (values, _) = values.as_chunks::<3>();
            for (r, &res) in interior.iter().zip(values) {
                let v = apply_bit_residual(tile.predict(r)?, res)?;
                tile.set(r.at, v)?;
            }
        }
        reader.finish()?;
        tile.into_grid()
    }
}
