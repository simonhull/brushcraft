//! Pool slot layouts: plain data, so the same bytes can be uploaded to a GPU storage buffer.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use geom::{CorrectionError, CorrectionTile, CorrectionView, CorrectionViewMut, TILE_QUADS, Vec3};

/// Vertices per side of the largest tile.
pub(crate) const SIDE: usize = TILE_QUADS as usize + 1;
/// Interior vertices of the largest tile.
const INTERIOR: usize = (SIDE - 2) * (SIDE - 2);
/// Border vertices of the largest tile.
const BORDER: usize = 4 * (SIDE - 1);

/// A correction tile of any level (smaller tiles use a prefix of each array), laid out as
/// [`CorrectionTile`] stores it: quantized interior, exact border.
///
/// A zeroed slot has side 0 and no valid view until [`CorrectionSlot::reset`] or a fill.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CorrectionSlot {
    /// Vertices per side.
    pub(crate) side: u32,
    /// The interior quantization step.
    pub(crate) scale: f32,
    /// Quantized interior values, row-major over `(side − 2)²`.
    pub(crate) interior: [[i16; 3]; INTERIOR],
    /// Keeps `border` 4-byte aligned without implicit padding (required by `Pod`).
    _pad: [u8; 2],
    /// Exact border values in ring order, `4(side − 1)` of them.
    pub(crate) border: [Vec3; BORDER],
}

/// An evaluated tile: positions and normals, row-major over `side²` (a prefix for smaller tiles).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct EvaluatedSlot {
    /// Limit-surface positions.
    pub(crate) positions: [Vec3; SIDE * SIDE],
    /// Unit normals.
    pub(crate) normals: [Vec3; SIDE * SIDE],
}

impl CorrectionSlot {
    /// Zeroes the slot and gives it `side` vertices per side: a tile of zero corrections.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] for a side outside `2..=65`; the slot is zeroed either way.
    pub(crate) fn reset(&mut self, side: u32) -> Result<(), CorrectionError> {
        bytemuck::bytes_of_mut(self).fill(0);
        // Zero scale and values: only the side can be wrong.
        CorrectionView::new(side, 0.0, &self.interior, &self.border)?;
        self.side = side;
        Ok(())
    }

    /// The tile in the slot.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] or [`CorrectionError::BadParts`] if the slot doesn't hold a
    /// tile (a zeroed slot has side 0).
    pub(crate) fn view(&self) -> Result<CorrectionView<'_>, CorrectionError> {
        CorrectionView::new(self.side, self.scale, &self.interior, &self.border)
    }

    /// The tile in the slot, for editing.
    ///
    /// # Errors
    ///
    /// As [`CorrectionSlot::view`].
    pub(crate) fn view_mut(&mut self) -> Result<CorrectionViewMut<'_>, CorrectionError> {
        CorrectionViewMut::new(
            self.side,
            &mut self.scale,
            &mut self.interior,
            &mut self.border,
        )
    }

    /// A slot holding a copy of `tile`.
    #[must_use]
    pub(crate) fn from_tile(tile: &CorrectionTile) -> Self {
        let mut slot = Self::zeroed();
        slot.side = tile.side();
        slot.scale = tile.scale();
        // A tile's arrays are never longer than the largest tile's, so `zip` copies all of them.
        for (to, from) in slot.interior.iter_mut().zip(tile.interior()) {
            *to = *from;
        }
        for (to, from) in slot.border.iter_mut().zip(tile.border()) {
            *to = *from;
        }
        slot
    }

    /// The slot's tile as a [`CorrectionTile`], validated as the file decode validates it.
    ///
    /// # Errors
    ///
    /// As [`CorrectionTile::from_parts`]; [`CorrectionSlot::view`]'s errors for a slot without a
    /// tile.
    #[expect(
        clippy::wrong_self_convention,
        reason = "a 26 KB slot is converted by reference, not copied"
    )]
    pub(crate) fn to_tile(&self) -> Result<CorrectionTile, CorrectionError> {
        let (interior, border) = self.parts()?;
        CorrectionTile::from_parts(self.side, self.scale, interior.to_vec(), border.to_vec())
    }

    /// The interior and border prefixes the slot's side uses.
    ///
    /// # Errors
    ///
    /// As [`CorrectionSlot::view`].
    fn parts(&self) -> Result<(&[[i16; 3]], &[Vec3]), CorrectionError> {
        let side = self.view()?.side();
        let bad = CorrectionError::BadParts { side };
        // `view` accepted the side, so it is in `2..=65` and the counts fit the arrays.
        let n = usize::try_from(side).or(Err(bad))?;
        let interior = self.interior.get(..(n - 2) * (n - 2)).ok_or(bad)?;
        let border = self.border.get(..4 * (n - 1)).ok_or(bad)?;
        Ok((interior, border))
    }

    /// The gate for values written straight into the slot (the file decode): checks them by
    /// [`CorrectionTile::from_parts`]'s rules (a valid side and scale, no `i16::MIN`, a finite
    /// border, and a zero scale only with a zero interior) and turns `-0.0` into `+0.0` as it
    /// does, so views, which trust their input, see only what a [`CorrectionTile`] could hold.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] or [`CorrectionError::BadParts`]; the slot is zeroed, so it
    /// holds no tile.
    pub(crate) fn validate(&mut self) -> Result<(), CorrectionError> {
        let checked = self.parts().and_then(|(interior, border)| {
            let ok = (self.scale != 0.0 || interior.iter().all(|q| *q == [0; 3]))
                && interior.iter().flatten().all(|q| *q != i16::MIN)
                && border.iter().all(|v| v.is_finite());
            if ok {
                Ok(border.len())
            } else {
                Err(CorrectionError::BadParts { side: self.side })
            }
        });
        match checked {
            Ok(ring) => {
                self.scale += 0.0;
                for v in self.border.iter_mut().take(ring) {
                    *v = Vec3::new(v.x + 0.0, v.y + 0.0, v.z + 0.0);
                }
                Ok(())
            }
            Err(e) => {
                bytemuck::bytes_of_mut(self).fill(0);
                Err(e)
            }
        }
    }
}

impl fmt::Debug for CorrectionSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CorrectionSlot")
            .field("side", &self.side)
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for EvaluatedSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvaluatedSlot").finish_non_exhaustive()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A tile of `side` with pseudo-random values from `seed`.
    fn random_tile(side: u32, seed: u64) -> CorrectionTile {
        let mut rng = seed;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            f32::from(u16::try_from(rng >> 48).unwrap()) / 65536.0 - 0.5
        };
        let values: Vec<Vec3> = (0..side * side)
            .map(|_| Vec3::new(next(), next(), next()))
            .collect();
        CorrectionTile::from_values(side, &values).unwrap()
    }

    #[test]
    fn slots_have_the_gpu_sizes() {
        assert_eq!(size_of::<CorrectionSlot>(), 26_896, "correction slot");
        assert_eq!(size_of::<EvaluatedSlot>(), 101_400, "evaluated slot");
        let slot = EvaluatedSlot::zeroed();
        assert_eq!(
            (slot.positions.len(), slot.normals.len()),
            (65 * 65, 65 * 65),
            "a 65² grid each"
        );
    }

    #[test]
    fn tiles_round_trip_through_slots() {
        for (side, seed) in [(65, 0x1234_5678), (9, 0x9ABC_DEF0)] {
            let tile = random_tile(side, seed);
            let slot = CorrectionSlot::from_tile(&tile);
            assert_eq!(slot.to_tile().unwrap(), tile, "side {side}");
            let view = slot.view().unwrap();
            for j in 0..side {
                for i in 0..side {
                    assert_eq!(
                        view.get(i, j).map(Vec3::to_bits),
                        tile.get(i, j).map(Vec3::to_bits),
                        "side {side} ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn views_edit_the_slot_in_place() {
        let mut slot = CorrectionSlot::zeroed();
        assert!(slot.view().is_err(), "a zeroed slot holds no tile");
        slot.reset(9).unwrap();
        assert!(slot.view().unwrap().is_zero(), "reset to zero corrections");
        let c = Vec3::new(0.25, -0.5, 1.0);
        slot.view_mut().unwrap().set(3, 4, c).unwrap();
        let mut tile = CorrectionTile::zero(9).unwrap();
        tile.set(3, 4, c).unwrap();
        assert_eq!(slot.to_tile().unwrap(), tile, "same as editing a tile");
        assert!(slot.reset(66).is_err(), "too wide");
        assert!(slot.view().is_err(), "a failed reset leaves no tile");
    }
}
