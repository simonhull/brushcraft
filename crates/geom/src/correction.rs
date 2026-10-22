//! Correction tiles: the displacement channel's payload.

use thiserror::Error;

use crate::{TILE_QUADS, Vec3};

/// Largest quantized magnitude.
const Q_MAX: f32 = 32767.0;
/// Interior magnitudes below this would need a subnormal scale; such interiors are stored as zero.
const TINY: f32 = Q_MAX * f32::MIN_POSITIVE;

/// One tile's corrections, `(t, b, n)` per vertex (`x`, `y`, `z`) in the tangent frame of the
/// predicted surface.
///
/// Interior vertices are stored as `[i16; 3]` with one scale per tile, so each component is off by
/// at most [`CorrectionTile::max_error`]. Border vertices are shared with neighbouring tiles that
/// have their own scales, so they are stored exactly: every copy decodes to the same bits, which
/// keeps seams closed.
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
        Ok(Self {
            side,
            scale: 0.0,
            interior: vec![[0; 3]; interior],
            border: vec![Vec3::ZERO; border],
        })
    }

    /// A tile from `side²` row-major values. The scale comes from the interior values alone
    /// (borders are stored exactly), then every value is stored once.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`], [`CorrectionError::BadParts`] if `values` has the wrong
    /// length, or [`CorrectionError::NotFinite`].
    pub fn from_values(side: u32, values: &[Vec3]) -> Result<Self, CorrectionError> {
        let mut tile = Self::zero(side)?;
        if values.len() != tile.interior.len() + tile.border.len() {
            return Err(CorrectionError::BadParts { side });
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(CorrectionError::NotFinite);
        }
        let cells = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .zip(values);
        let peak = cells
            .clone()
            .filter(|((i, j), _)| matches!(tile.slot(*i, *j), Some(Slot::Interior(_))))
            .map(|(_, v)| max_abs(*v))
            .fold(0.0, f32::max);
        tile.scale = if peak < TINY { 0.0 } else { peak / Q_MAX };
        cells
            .into_iter()
            .try_for_each(|((i, j), v)| tile.store(i, j, *v))?;
        Ok(tile)
    }

    /// A tile from stored parts (the file format's decode path).
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] or [`CorrectionError::BadParts`] if the lengths don't match
    /// `side`, the scale is negative or not finite, or a border value is not finite.
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
            && border.iter().all(|v| v.is_finite())
            && interior.iter().flatten().all(|q| *q != i16::MIN);
        if ok {
            Ok(Self {
                side,
                scale: scale + 0.0,
                interior,
                border: border.into_iter().map(without_negative_zero).collect(),
            })
        } else {
            Err(CorrectionError::BadParts { side })
        }
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
    ///
    /// This holds for tiles built by [`CorrectionTile::from_values`] or
    /// [`CorrectionTile::from_parts`]. After [`CorrectionTile::set`] has grown the scale,
    /// values are only guaranteed within one [`CorrectionTile::scale`].
    #[must_use]
    pub const fn max_error(&self) -> f32 {
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

    /// Sets the correction at `(i, j)`.
    ///
    /// If an interior value doesn't fit the current scale, the scale grows geometrically (doubling
    /// until it fits; from zero it becomes exactly what the value needs) and the existing
    /// interior is re-quantized. Each growth adds at most half the new step of error to earlier
    /// values, so after any number of growths they are within one
    /// final [`CorrectionTile::scale`] of what was stored, not within
    /// [`CorrectionTile::max_error`]. Interior values below a tiny threshold on a zero-scale
    /// tile are stored as zero.
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
                if self.scale == 0.0 {
                    if needed >= TINY {
                        self.requantize(needed / Q_MAX);
                    }
                } else {
                    self.requantize(grown_scale(self.scale, needed));
                }
            }
        }
        self.store(i, j, c)
    }

    /// Writes `value` with the current scale.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::OutOfTile`] if `(i, j)` is outside the tile. The `get_mut` misses below
    /// are unreachable: `slot` only returns indices inside the vectors `counts` sized.
    fn store(&mut self, i: u32, j: u32, value: Vec3) -> Result<(), CorrectionError> {
        let side = self.side;
        match self.slot(i, j) {
            Some(Slot::Interior(index)) => {
                let scale = self.scale;
                let q = self
                    .interior
                    .get_mut(index)
                    .ok_or(CorrectionError::OutOfTile { i, j, side })?;
                *q = [
                    quantize(value.x, scale),
                    quantize(value.y, scale),
                    quantize(value.z, scale),
                ];
            }
            Some(Slot::Border(index)) => {
                *self
                    .border
                    .get_mut(index)
                    .ok_or(CorrectionError::OutOfTile { i, j, side })? =
                    without_negative_zero(value);
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
            *q = [
                quantize(v.x, scale),
                quantize(v.y, scale),
                quantize(v.z, scale),
            ];
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
    /// [`CorrectionError::BadSide`] below 2 or above `TILE_QUADS + 1`, the largest tile there is.
    fn counts(side: u32) -> Result<(usize, usize), CorrectionError> {
        if !(2..=TILE_QUADS + 1).contains(&side) {
            return Err(CorrectionError::BadSide(side));
        }
        let to = |n: u32| {
            usize::try_from(n)
                .ok()
                .ok_or(CorrectionError::BadSide(side))
        };
        Ok((to((side - 2) * (side - 2))?, to(4 * (side - 1))?))
    }
}

/// Grows `scale` by the smallest power of two that makes `needed` fit.
fn grown_scale(scale: f32, needed: f32) -> f32 {
    (1..=128)
        .map(|k| scale * 2.0_f32.powi(k))
        .find(|grown| grown.is_finite() && grown * Q_MAX >= needed)
        .unwrap_or(needed / Q_MAX)
}

/// Maps `-0.0` components to `+0.0`, so equal corrections have equal bits.
fn without_negative_zero(v: Vec3) -> Vec3 {
    Vec3::new(v.x + 0.0, v.y + 0.0, v.z + 0.0)
}

fn max_abs(v: Vec3) -> f32 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounded and clamped to ±32767 first"
)]
fn quantize(x: f32, scale: f32) -> i16 {
    if scale <= 0.0 {
        return 0;
    }
    (x / scale).round().clamp(-Q_MAX, Q_MAX) as i16
}

fn decode(q: [i16; 3], scale: f32) -> Vec3 {
    Vec3::new(
        f32::from(q[0]) * scale,
        f32::from(q[1]) * scale,
        f32::from(q[2]) * scale,
    )
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::test_rng::Rng;

    #[test]
    fn slots_are_a_bijection_onto_the_storage() {
        for side in [2_u32, 3, 65] {
            let tile = CorrectionTile::zero(side).unwrap();
            let mut interior = vec![false; tile.interior.len()];
            let mut border = vec![false; tile.border.len()];
            for j in 0..side {
                for i in 0..side {
                    let (seen, k) = match tile.slot(i, j).unwrap() {
                        Slot::Interior(k) => (&mut interior, k),
                        Slot::Border(k) => (&mut border, k),
                    };
                    assert!(!seen[k], "side {side}: slot {k} used twice");
                    seen[k] = true;
                }
            }
            assert!(
                interior.iter().all(|s| *s),
                "side {side}: every interior slot used"
            );
            assert!(
                border.iter().all(|s| *s),
                "side {side}: every border slot used"
            );
        }
    }

    #[test]
    fn slots_are_laid_out_as_documented() {
        let tile = CorrectionTile::zero(5).unwrap();
        let last: u32 = 4;
        let ring = |n: u32| Slot::Border(usize::try_from(n).unwrap());
        assert_eq!(tile.slot(1, 1), Some(Slot::Interior(0)), "first interior");
        assert_eq!(tile.slot(2, 1), Some(Slot::Interior(1)), "row-major");
        assert_eq!(tile.slot(0, 0), Some(Slot::Border(0)), "corner 0");
        assert_eq!(tile.slot(last, 0), Some(ring(last)), "corner 1");
        assert_eq!(tile.slot(last, last), Some(ring(2 * last)), "corner 2");
        assert_eq!(tile.slot(0, last), Some(ring(3 * last)), "corner 3");
        assert_eq!(
            tile.slot(last - 1, last),
            Some(ring(2 * last + 1)),
            "top row runs backwards"
        );
    }

    #[test]
    fn scale_comes_from_the_interior_only() {
        let side = 5;
        let mut values = vec![Vec3::new(1e3, 1e3, 1e3); 25];
        for j in 1..4 {
            for i in 1..4 {
                values[j * 5 + i] = Vec3::new(1.0, 0.0, -0.5);
            }
        }
        let tile = CorrectionTile::from_values(side, &values).unwrap();
        assert_eq!(tile.scale(), 1.0 / 32767.0, "borders don't set the scale");
        assert_eq!(tile.get(0, 0).unwrap().x, 1e3, "border exact");
    }

    #[test]
    fn tiny_interiors_are_stored_as_zero() {
        let values = vec![Vec3::new(1e-40, 0.0, 0.0); 9];
        let tile = CorrectionTile::from_values(3, &values).unwrap();
        assert_eq!(tile.scale(), 0.0, "no subnormal scale");
        assert_eq!(tile.get(1, 1).unwrap(), Vec3::ZERO, "interior zero");
    }

    #[test]
    fn negative_zero_is_normalised() {
        let mut tile = CorrectionTile::zero(4).unwrap();
        tile.set(0, 0, Vec3::new(-0.0, -0.0, -0.0)).unwrap();
        assert_eq!(
            tile.get(0, 0).unwrap().to_bits(),
            [0; 3],
            "border −0 becomes +0"
        );
        assert!(tile.is_zero(), "still zero");
        let parts = CorrectionTile::from_parts(2, -0.0, vec![], vec![Vec3::new(-0.0, 0.0, 0.0); 4])
            .unwrap();
        assert_eq!(parts.scale().to_bits(), 0, "scale normalised");
        assert_eq!(parts.border()[0].to_bits(), [0; 3], "border normalised");
    }

    #[test]
    fn from_parts_rejects_i16_min_and_oversized_sides() {
        let bad = vec![[i16::MIN, 0, 0]];
        assert_eq!(
            CorrectionTile::from_parts(3, 1.0, bad, vec![Vec3::ZERO; 8]),
            Err(CorrectionError::BadParts { side: 3 }),
            "−32768 is not a valid quantized value"
        );
        assert_eq!(
            CorrectionTile::zero(66),
            Err(CorrectionError::BadSide(66)),
            "above a full tile"
        );
        assert!(CorrectionTile::zero(65).is_ok(), "a full tile");
    }

    #[test]
    fn repeated_small_growth_does_not_drift() {
        let mut tile = CorrectionTile::zero(9).unwrap();
        let small = Vec3::new(0.01, 0.0, 0.0);
        tile.set(2, 2, small).unwrap();
        let mut v = 0.01_f32;
        for _ in 0..100 {
            v *= 1.0005;
            tile.set(5, 5, Vec3::new(v, 0.0, 0.0)).unwrap();
        }
        assert!(
            (tile.get(2, 2).unwrap().x - small.x).abs() <= tile.scale(),
            "small value within one scale"
        );
        assert!(
            (tile.get(5, 5).unwrap().x - v).abs() <= tile.scale(),
            "latest value within one scale"
        );
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
        assert_eq!(
            tile.get(0, 17).unwrap().to_bits(),
            c.to_bits(),
            "left border exact"
        );
        assert_eq!(
            tile.get(64, 64).unwrap().to_bits(),
            c.to_bits(),
            "corner exact"
        );
    }

    #[test]
    fn set_grows_the_scale_when_needed() {
        let mut tile = CorrectionTile::zero(9).unwrap();
        tile.set(4, 4, Vec3::new(0.001, 0.0, 0.0)).unwrap();
        tile.set(3, 3, Vec3::new(5.0, 0.0, 0.0)).unwrap();
        let big = tile.get(3, 3).unwrap();
        assert!((big.x - 5.0).abs() <= tile.max_error(), "large value kept");
        // The small value was re-quantized once more: within one (new) step.
        assert!(
            (tile.get(4, 4).unwrap().x - 0.001).abs() <= tile.scale(),
            "small value within a step"
        );
    }

    #[test]
    fn rejects_bad_input() {
        let mut tile = CorrectionTile::zero(4).unwrap();
        assert_eq!(
            tile.set(4, 0, Vec3::ZERO),
            Err(CorrectionError::OutOfTile {
                i: 4,
                j: 0,
                side: 4
            }),
            "outside"
        );
        assert_eq!(
            tile.set(1, 1, Vec3::NAN),
            Err(CorrectionError::NotFinite),
            "NaN"
        );
        assert_eq!(
            CorrectionTile::zero(1),
            Err(CorrectionError::BadSide(1)),
            "too small"
        );
        assert!(tile.is_zero(), "untouched");
    }
}
