//! The codecs the encoding study measured and didn't choose, kept for `brushcraft-bench
//! display-study` and the tests (see the encoding module's notes).

use super::bytes::{Reader, deltas_u16, put_u16_planes, take_u16_planes, undeltas_u16};
use super::{
    DisplayCodec, Tile, decode_border, encode_border, join, parts, per_axis, refinement, tile_quads,
};
use crate::display::DisplayError;
use crate::{CorrectionTile, Grid, Region, TileKey, Vec3};

/// Interior vertices as 16-bit fixed point in the tile's bounding box, border vertices
/// lossless (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedPoint;

/// Offsets from the base level's own positions, in its tangent frame, quantized as correction
/// tiles are and laid out as the store writes them (see the module notes).
///
/// Holds the face's base grids, `base[l]` the face's whole grid at level `l`, which it needs to
/// encode and to decode: a tile doesn't decode on its own.
#[derive(Debug, Clone, Copy)]
pub struct BaseOffsets<'a> {
    /// The face's base grids, by level.
    base: &'a [Grid],
}

impl<'a> BaseOffsets<'a> {
    /// The codec over one face's base grids, `base[l]` its whole grid at level `l`.
    #[must_use]
    pub const fn new(base: &'a [Grid]) -> Self {
        Self { base }
    }
}

/// The interior vertices of a tile with `side` vertices per side, row-major.
fn interior(side: u32) -> impl Iterator<Item = (u32, u32)> {
    let last = side.saturating_sub(1);
    (1..last).flat_map(move |j| (1..last).map(move |i| (i, j)))
}

/// Largest fixed-point value [`FixedPoint`] stores.
const FIXED_MAX: f32 = 65535.0;

impl DisplayCodec for FixedPoint {
    fn encode(&self, key: &TileKey, positions: &Grid) -> Result<Vec<u8>, DisplayError> {
        let quads = tile_quads(key, positions.side())?;
        if !positions.as_slice().iter().all(|v| v.is_finite()) {
            return Err(DisplayError::NotFinite);
        }
        let order = refinement(quads);
        let tile = Tile::from_grid(positions);
        let (lo, hi) = positions.as_slice().iter().fold(
            ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
            |(lo, hi), v| {
                let c = parts(*v);
                (per_axis(lo, c, f32::min), per_axis(hi, c, f32::max))
            },
        );
        let step = per_axis(hi, lo, |h, l| (h - l) / FIXED_MAX);
        if !step.iter().all(|s| s.is_finite()) {
            return Err(DisplayError::TooLarge);
        }
        let mut out = Vec::new();
        for v in lo.into_iter().chain(step) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        encode_border(&tile, quads, &order, &mut out)?;
        let mut planes: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for at in interior(positions.side()) {
            let offset = per_axis(parts(tile.get(at)?), lo, |c, l| c - l);
            let fixed = per_axis(offset, step, |d, s| {
                if s > 0.0 {
                    (d / s).round().clamp(0.0, FIXED_MAX)
                } else {
                    0.0
                }
            });
            for (plane, q) in planes.iter_mut().zip(fixed) {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "within 0..=65535 after the clamp, rounded"
                )]
                plane.push(q as u16);
            }
        }
        for plane in planes {
            put_u16_planes(&mut out, &deltas_u16(&plane));
        }
        Ok(out)
    }

    fn decode(&self, key: &TileKey, bytes: &[u8]) -> Result<Grid, DisplayError> {
        let side = key.side();
        let quads = tile_quads(key, side)?;
        let order = refinement(quads);
        let mut reader = Reader::new(bytes);
        let lo = [reader.f32()?, reader.f32()?, reader.f32()?];
        let step = [reader.f32()?, reader.f32()?, reader.f32()?];
        if !lo.iter().chain(&step).all(|v| v.is_finite()) || step.iter().any(|&s| s < 0.0) {
            return Err(DisplayError::Malformed);
        }
        let mut tile = Tile::zero(side);
        decode_border(&mut tile, quads, &order, &mut reader)?;
        let count = interior(side).count();
        let planes = [
            undeltas_u16(&take_u16_planes(&mut reader, count)?),
            undeltas_u16(&take_u16_planes(&mut reader, count)?),
            undeltas_u16(&take_u16_planes(&mut reader, count)?),
        ];
        reader.finish()?;
        let [xs, ys, zs] = &planes;
        for (at, ((&x, &y), &z)) in interior(side).zip(xs.iter().zip(ys).zip(zs)) {
            let scaled = per_axis([x, y, z].map(f32::from), step, |q, s| q * s);
            let v = join(per_axis(lo, scaled, |l, d| l + d));
            if !v.is_finite() {
                return Err(DisplayError::NotFinite);
            }
            tile.set(at, v)?;
        }
        tile.into_grid()
    }
}

impl BaseOffsets<'_> {
    /// The face's base grid at `key`'s level, with the tile's place on it.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] if the codec holds no whole-face grid at that level.
    fn base_of(&self, key: &TileKey) -> Result<(&Grid, Region), DisplayError> {
        let grid = self
            .base
            .get(usize::from(key.level().get()))
            .filter(|g| g.side() == key.level().face_quads() + 1)
            .ok_or(DisplayError::Shape)?;
        Ok((grid, Region::of_tile(key)))
    }
}

/// The base position at face-grid `(x, y)` of `grid`, and its tangent frame there.
///
/// The frame is `t` along the grid's `i` (central differences, one-sided at the face's border),
/// `n` the normal `du × dv`, `b = n × t`: the world axes where the grid is degenerate.
///
/// # Errors
///
/// [`DisplayError::Shape`] if `(x, y)` is outside the grid.
fn base_frame(grid: &Grid, x: u32, y: u32) -> Result<(Vec3, [Vec3; 3]), DisplayError> {
    let here = grid.get(x, y).ok_or(DisplayError::Shape)?;
    let along = |ahead: Option<Vec3>, behind: Option<Vec3>| match (ahead, behind) {
        (Some(a), Some(b)) => a - b,
        (Some(a), None) => a - here,
        (None, Some(b)) => here - b,
        (None, None) => Vec3::ZERO,
    };
    let du = along(
        grid.get(x + 1, y),
        x.checked_sub(1).and_then(|p| grid.get(p, y)),
    );
    let dv = along(
        grid.get(x, y + 1),
        y.checked_sub(1).and_then(|p| grid.get(x, p)),
    );
    let world = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let (Some(n), Some(t)) = (du.cross(dv).normalized(), du.normalized()) else {
        return Ok((here, world));
    };
    Ok((here, [t, n.cross(t), n]))
}

impl DisplayCodec for BaseOffsets<'_> {
    fn encode(&self, key: &TileKey, positions: &Grid) -> Result<Vec<u8>, DisplayError> {
        let side = positions.side();
        tile_quads(key, side)?;
        let (grid, region) = self.base_of(key)?;
        let mut offsets = Vec::with_capacity(positions.as_slice().len());
        for j in 0..side {
            for i in 0..side {
                let (here, [t, b, n]) = base_frame(grid, region.x0() + i, region.y0() + j)?;
                let d = positions.get(i, j).ok_or(DisplayError::Shape)? - here;
                offsets.push(Vec3::new(d.dot(t), d.dot(b), d.dot(n)));
            }
        }
        let tile = CorrectionTile::from_values(side, &offsets).map_err(|e| match e {
            crate::CorrectionError::NotFinite => DisplayError::NotFinite,
            crate::CorrectionError::TooLarge => DisplayError::TooLarge,
            crate::CorrectionError::OutOfTile { .. }
            | crate::CorrectionError::UnknownFace(_)
            | crate::CorrectionError::LevelZero
            | crate::CorrectionError::BadSide(_)
            | crate::CorrectionError::BadParts { .. } => DisplayError::Shape,
        })?;
        // As the store writes a correction tile: side, scale, the interior's three planes
        // delta-coded, then the border's three planes of exact `f32`s.
        let mut out = Vec::new();
        out.extend_from_slice(
            &u16::try_from(side)
                .map_err(|_too_wide| DisplayError::Shape)?
                .to_le_bytes(),
        );
        out.extend_from_slice(&tile.scale().to_le_bytes());
        for k in 0..3 {
            let mut prev = 0_i16;
            for q in tile.interior() {
                let v = q.get(k).copied().unwrap_or(0);
                out.extend_from_slice(&v.wrapping_sub(prev).to_le_bytes());
                prev = v;
            }
        }
        for k in 0..3 {
            for v in tile.border() {
                out.extend_from_slice(&parts(*v).get(k).copied().unwrap_or(0.0).to_le_bytes());
            }
        }
        Ok(out)
    }

    fn decode(&self, key: &TileKey, bytes: &[u8]) -> Result<Grid, DisplayError> {
        let side = key.side();
        tile_quads(key, side)?;
        let (grid, region) = self.base_of(key)?;
        let mut reader = Reader::new(bytes);
        let stored_side = u16::from_le_bytes(*reader.take_array::<2>()?);
        if u32::from(stored_side) != side {
            return Err(DisplayError::Malformed);
        }
        let scale = reader.f32()?;
        let inner =
            usize::try_from(side.saturating_sub(2)).map_err(|_too_wide| DisplayError::Shape)?;
        let (count, ring) = (inner * inner, 4 * (inner + 1));
        let mut interior = vec![[0_i16; 3]; count];
        for k in 0..3 {
            let mut prev = 0_i16;
            for q in &mut interior {
                let d = i16::from_le_bytes(*reader.take_array::<2>()?);
                prev = prev.wrapping_add(d);
                if let Some(slot) = q.get_mut(k) {
                    *slot = prev;
                }
            }
        }
        let mut border = vec![[0.0_f32; 3]; ring];
        for k in 0..3 {
            for v in &mut border {
                if let Some(slot) = v.get_mut(k) {
                    *slot = reader.f32()?;
                }
            }
        }
        reader.finish()?;
        let tile = CorrectionTile::from_parts(
            side,
            scale,
            interior,
            border.into_iter().map(join).collect(),
        )
        .map_err(|_invalid| DisplayError::Malformed)?;
        let mut out = Vec::with_capacity(count + ring);
        for j in 0..side {
            for i in 0..side {
                let (here, [tangent, bitangent, normal]) =
                    base_frame(grid, region.x0() + i, region.y0() + j)?;
                let offset = tile.get(i, j).ok_or(DisplayError::Shape)?;
                let position = here + tangent * offset.x + bitangent * offset.y + normal * offset.z;
                if !position.is_finite() {
                    return Err(DisplayError::NotFinite);
                }
                out.push(position);
            }
        }
        Ok(Grid::new(side, out)?)
    }
}
