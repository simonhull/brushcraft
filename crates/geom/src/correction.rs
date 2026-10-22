//! Correction tiles: the displacement channel's payload.

use thiserror::Error;

use crate::Vec3;

/// Largest quantized magnitude.
const Q_MAX: f32 = 32767.0;

/// One tile's corrections, `(t, b, n)` per vertex (`x`, `y`, `z`) in the tangent frame of the predicted surface.
///
/// Interior vertices are stored as `[i16; 3]` with one scale per tile, so each component is off by at most
/// [`CorrectionTile::max_error`]. Border vertices are shared with neighbouring tiles that have their own scales, so
/// they are stored exactly: every copy decodes to the same bits, which keeps seams closed.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionTile {
    side: u32,
    scale: f32,
    interior: Vec<[i16; 3]>,
    border: Vec<Vec3>,
}

/// Why a correction tile operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CorrectionError {
    /// The vertex is outside the tile.
    #[error("vertex ({i}, {j}) is outside a tile of {side} vertices per side")]
    OutOfTile {
        /// Column.
        i: u32,
        /// Row.
        j: u32,
        /// Vertices per side.
        side: u32,
    },
    /// A component is NaN or infinite.
    #[error("correction is not finite")]
    NotFinite,
    /// Tiles have at least 2 vertices per side.
    #[error("a tile needs at least 2 vertices per side, got {0}")]
    BadSide(u32),
    /// Parts given to [`CorrectionTile::from_parts`] don't fit together.
    #[error("parts don't match a tile of {side} vertices per side")]
    BadParts {
        /// Vertices per side.
        side: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Interior(usize),
    Border(usize),
}

impl CorrectionTile {
    /// A tile of zero corrections.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] below 2.
    pub fn zero(side: u32) -> Result<Self, CorrectionError> {
        let (interior, border) = Self::counts(side)?;
        Ok(Self { side, scale: 0.0, interior: vec![[0; 3]; interior], border: vec![Vec3::ZERO; border] })
    }

    /// A tile from `side²` row-major values, quantized in one pass.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`], [`CorrectionError::BadParts`] if `values` has the wrong length, or
    /// [`CorrectionError::NotFinite`].
    pub fn from_values(side: u32, values: &[Vec3]) -> Result<Self, CorrectionError> {
        let mut tile = Self::zero(side)?;
        if values.len() != tile.interior.len() + tile.border.len() + Self::overlap(side) {
            return Err(CorrectionError::BadParts { side });
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(CorrectionError::NotFinite);
        }
        let mut cells = (0..side).flat_map(|j| (0..side).map(move |i| (i, j))).zip(values);
        let max = values
            .iter()
            .zip((0..side).flat_map(|j| (0..side).map(move |i| (i, j))))
            .filter(|(_, (i, j))| matches!(tile.slot(*i, *j), Some(Slot::Interior(_))))
            .map(|(v, _)| max_abs(*v))
            .fold(0.0, f32::max);
        tile.scale = max / Q_MAX;
        cells.try_for_each(|((i, j), v)| tile.store(i, j, *v))?;
        Ok(tile)
    }

    /// A tile from stored parts (the file format's decode path).
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] or [`CorrectionError::BadParts`] if the lengths don't match `side`, the scale is
    /// negative or not finite, or a border value is not finite.
    pub fn from_parts(
        side: u32,
        scale: f32,
        interior: Vec<[i16; 3]>,
        border: Vec<Vec3>,
    ) -> Result<Self, CorrectionError> {
        let (ni, nb) = Self::counts(side)?;
        let ok = interior.len() == ni
            && border.len() == nb
            && scale.is_finite()
            && scale >= 0.0
            && border.iter().all(|v| v.is_finite());
        if ok { Ok(Self { side, scale, interior, border }) } else { Err(CorrectionError::BadParts { side }) }
    }

    /// Vertices per side.
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.side
    }

    /// The interior quantization step.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// The largest error per interior component after one quantization: half a step.
    #[must_use]
    pub fn max_error(&self) -> f32 {
        self.scale * 0.5
    }

    /// Quantized interior values, row-major.
    #[must_use]
    pub fn interior(&self) -> &[[i16; 3]] {
        &self.interior
    }

    /// Exact border values, in ring order.
    #[must_use]
    pub fn border(&self) -> &[Vec3] {
        &self.border
    }

    /// Whether every correction is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.interior.iter().all(|q| *q == [0; 3]) && self.border.iter().all(|v| *v == Vec3::ZERO)
    }

    /// The correction at `(i, j)`, or `None` outside the tile.
    #[must_use]
    pub fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        match self.slot(i, j)? {
            Slot::Interior(k) => self.interior.get(k).map(|q| decode(*q, self.scale)),
            Slot::Border(k) => self.border.get(k).copied(),
        }
    }

    /// Sets the correction at `(i, j)`. If an interior value doesn't fit the current scale, the tile is re-quantized
    /// with a larger one (earlier values are then within one new step).
    ///
    /// # Errors
    ///
    /// [`CorrectionError::OutOfTile`] or [`CorrectionError::NotFinite`].
    pub fn set(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CorrectionError> {
        if !c.is_finite() {
            return Err(CorrectionError::NotFinite);
        }
        if let Some(Slot::Interior(_)) = self.slot(i, j) {
            let needed = max_abs(c);
            if needed > Q_MAX * self.scale {
                self.requantize(needed / Q_MAX);
            }
        }
        self.store(i, j, c)
    }

    /// Writes `c` with the current scale.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::OutOfTile`].
    fn store(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CorrectionError> {
        let side = self.side;
        match self.slot(i, j) {
            Some(Slot::Interior(k)) => {
                let scale = self.scale;
                let q = self.interior.get_mut(k).ok_or(CorrectionError::OutOfTile { i, j, side })?;
                *q = [quantize(c.x, scale), quantize(c.y, scale), quantize(c.z, scale)];
            }
            Some(Slot::Border(k)) => {
                *self.border.get_mut(k).ok_or(CorrectionError::OutOfTile { i, j, side })? = c;
            }
            None => return Err(CorrectionError::OutOfTile { i, j, side }),
        }
        Ok(())
    }

    fn requantize(&mut self, scale: f32) {
        let old = self.scale;
        self.scale = scale;
        for q in &mut self.interior {
            let v = decode(*q, old);
            *q = [quantize(v.x, scale), quantize(v.y, scale), quantize(v.z, scale)];
        }
    }

    fn slot(&self, i: u32, j: u32) -> Option<Slot> {
        let s = self.side;
        if i >= s || j >= s {
            return None;
        }
        let last = s - 1;
        let k = |n: u32| usize::try_from(n).ok();
        if i > 0 && i < last && j > 0 && j < last {
            return k((j - 1) * (s - 2) + (i - 1)).map(Slot::Interior);
        }
        let ring = if j == 0 && i < last {
            i
        } else if i == last && j < last {
            last + j
        } else if j == last && i > 0 {
            2 * last + (last - i)
        } else {
            3 * last + (last - j)
        };
        k(ring).map(Slot::Border)
    }

    /// Interior and border counts for `side`.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] below 2.
    fn counts(side: u32) -> Result<(usize, usize), CorrectionError> {
        if side < 2 {
            return Err(CorrectionError::BadSide(side));
        }
        let to = |n: u32| usize::try_from(n).map_err(|_| CorrectionError::BadSide(side));
        Ok((to((side - 2) * (side - 2))?, to(4 * (side - 1))?))
    }

    /// Always 0: interior and border partition the tile. Kept so `from_values`' length check reads as a sum.
    const fn overlap(_side: u32) -> usize {
        0
    }
}

fn max_abs(v: Vec3) -> f32 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

#[expect(clippy::cast_possible_truncation, reason = "rounded and clamped to ±32767 first")]
fn quantize(x: f32, scale: f32) -> i16 {
    if scale <= 0.0 {
        return 0;
    }
    (x / scale).round().clamp(-Q_MAX, Q_MAX) as i16
}

fn decode(q: [i16; 3], scale: f32) -> Vec3 {
    Vec3::new(f32::from(q[0]) * scale, f32::from(q[1]) * scale, f32::from(q[2]) * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    #[test]
    fn border_ring_indexes_every_border_vertex_once() {
        let tile = CorrectionTile::zero(5).unwrap();
        let mut seen = vec![false; 16];
        for j in 0..5 {
            for i in 0..5 {
                if let Some(Slot::Border(k)) = tile.slot(i, j) {
                    assert!(!seen[k], "border slot {k} used twice");
                    seen[k] = true;
                }
            }
        }
        assert!(seen.iter().all(|s| *s), "every border slot used");
    }

    #[test]
    fn quantization_error_is_at_most_half_a_step() {
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let side = 2 + rng.below(64);
            let amplitude = 10f32.powf(rng.signed() * 4.0);
            let values: Vec<Vec3> = (0..side * side).map(|_| rng.vec3(amplitude)).collect();
            let tile = CorrectionTile::from_values(side, &values).unwrap();
            let bound = tile.max_error();
            for j in 0..side {
                for i in 0..side {
                    let want = values[(j * side + i) as usize];
                    let got = tile.get(i, j).unwrap();
                    for (w, g) in [(want.x, got.x), (want.y, got.y), (want.z, got.z)] {
                        let slack = w.abs() * 4.0 * f32::EPSILON;
                        assert!((w - g).abs() <= bound + slack, "|{w} − {g}| > {bound}");
                    }
                }
            }
        }
    }

    #[test]
    fn border_values_are_exact() {
        let mut tile = CorrectionTile::zero(65).unwrap();
        let c = Vec3::new(0.123_456_7, -9.876_543, 1e-9);
        tile.set(0, 17, c).unwrap();
        tile.set(64, 64, c).unwrap();
        assert_eq!(tile.get(0, 17).unwrap().to_bits(), c.to_bits(), "left border exact");
        assert_eq!(tile.get(64, 64).unwrap().to_bits(), c.to_bits(), "corner exact");
    }

    #[test]
    fn set_grows_the_scale_when_needed() {
        let mut tile = CorrectionTile::zero(9).unwrap();
        tile.set(4, 4, Vec3::new(0.001, 0.0, 0.0)).unwrap();
        tile.set(3, 3, Vec3::new(5.0, 0.0, 0.0)).unwrap();
        let big = tile.get(3, 3).unwrap();
        assert!((big.x - 5.0).abs() <= tile.max_error(), "large value kept");
        // The small value was re-quantized once more: within one (new) step.
        assert!((tile.get(4, 4).unwrap().x - 0.001).abs() <= tile.scale(), "small value within a step");
    }

    #[test]
    fn rejects_bad_input() {
        let mut tile = CorrectionTile::zero(4).unwrap();
        assert_eq!(tile.set(4, 0, Vec3::ZERO), Err(CorrectionError::OutOfTile { i: 4, j: 0, side: 4 }), "outside");
        assert_eq!(tile.set(1, 1, Vec3::NAN), Err(CorrectionError::NotFinite), "NaN");
        assert_eq!(CorrectionTile::zero(1), Err(CorrectionError::BadSide(1)), "too small");
        assert!(tile.is_zero(), "untouched");
    }
}
