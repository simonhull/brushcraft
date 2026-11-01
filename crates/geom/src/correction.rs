//! Correction tiles: the displacement channel's payload.

use thiserror::Error;

use crate::{FaceId, TILE_QUADS, Vec3};

/// Largest quantized magnitude.
const Q_MAX: f32 = 32767.0;
/// Interior magnitudes below this would need a subnormal scale; such interiors are stored as zero.
const TINY: f32 = Q_MAX * f32::MIN_POSITIVE;
/// The largest scale a tile can have: every one of its `±32767` steps decodes finite, as
/// `Q_MAX × MAX_SCALE < f32::MAX`. A power-of-two divisor keeps it exact.
const MAX_SCALE: f32 = f32::MAX / 65536.0;

/// One tile's corrections, `(t, b, n)` per vertex (`x`, `y`, `z`) in the tangent frame of the
/// predicted surface.
///
/// Interior vertices are stored as `[i16; 3]` with one scale per tile, so each component is off by
/// at most [`CorrectionTile::max_error`]. Border vertices are shared with neighbouring tiles that
/// have their own scales, so they are stored exactly: every copy decodes to the same bits, which
/// keeps seams closed.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionTile {
    /// Vertices per side, at least 2.
    side: u32,
    /// What one quantization step of the interior is worth: a component decodes as `q × scale`.
    /// `+0.0` while the interior is all zero, else finite, positive and normal.
    scale: f32,
    /// The `(side − 2)²` interior vertices, row-major, quantized with `scale`.
    interior: Vec<[i16; 3]>,
    /// The `4 × (side − 1)` border vertices, exact: once round the tile counter-clockwise from
    /// `(0, 0)`, along the bottom row, up the right column, back along the top row and down the
    /// left column.
    border: Vec<Vec3>,
}

/// Why a correction tile operation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
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
    /// The face isn't in the mesh.
    #[error("face {0} is not in the mesh")]
    UnknownFace(FaceId),
    /// A component is NaN or infinite.
    #[error("correction is not finite")]
    NotFinite,
    /// Level 0 is the base mesh; it has no corrections.
    #[error("level 0 is the base mesh and has no corrections")]
    LevelZero,
    /// Tiles have at least 2 vertices per side.
    #[error("a tile needs at least 2 vertices per side, got {0}")]
    BadSide(u32),
    /// Parts given to [`CorrectionTile::from_parts`] don't fit together.
    #[error("parts don't match a tile of {side} vertices per side")]
    BadParts {
        /// Vertices per side.
        side: u32,
    },
    /// An interior value would need a scale above the largest a tile can have, past which its
    /// steps decode to infinity.
    #[error("correction is too large to store")]
    TooLarge,
}

/// What [`CorrectionViewMut::update_interior_row`] did to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowUpdate {
    /// Whether any vertex of the row was written.
    pub wrote: bool,
    /// Why the row stopped before its end, if it did: nothing after that vertex was written.
    pub error: Option<CorrectionError>,
}

/// A correction tile stored elsewhere (a pool slot): `side` vertices per side, interior quantized
/// with `scale`, border exact. Same layout and rules as [`CorrectionTile`], whose
/// [`CorrectionTile::view`] is one.
///
/// The constructor checks the shape and the scale; the values themselves are trusted to have
/// been written through [`CorrectionViewMut::set`] or checked by [`CorrectionView::validate`]
/// (which [`CorrectionTile::from_parts`] and a decode through
/// [`CorrectionViewMut::normalize_and_validate`] run).
#[derive(Debug, Clone, Copy)]
pub struct CorrectionView<'a> {
    /// Vertices per side, at least 2.
    side: u32,
    /// The interior's quantization step, as [`CorrectionTile`]'s.
    scale: f32,
    /// The interior, row-major, at least `(side − 2)²` values (any more are ignored).
    interior: &'a [[i16; 3]],
    /// The border ring, in [`CorrectionTile`]'s order, at least `4 × (side − 1)` values.
    border: &'a [Vec3],
}

/// The mutable form of [`CorrectionView`]; [`CorrectionViewMut::set`] re-quantizes in place with
/// the same geometric growth as [`CorrectionTile::set`].
///
/// Values written straight into the storage (a decode) are checked and normalised with
/// [`CorrectionViewMut::normalize_and_validate`] before anything reads them.
#[derive(Debug)]
pub struct CorrectionViewMut<'a> {
    /// Vertices per side, at least 2.
    side: u32,
    /// The interior's quantization step, grown in place when a write needs more range.
    scale: &'a mut f32,
    /// The interior, row-major, re-quantized in place when the scale grows.
    interior: &'a mut [[i16; 3]],
    /// The border ring, in [`CorrectionTile`]'s order.
    border: &'a mut [Vec3],
}

/// Where a tile vertex is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Index into the quantized interior.
    Interior(usize),
    /// Index into the exact border ring.
    Border(usize),
}

/// Where `(i, j)` of a tile with `side` vertices per side is stored, or `None` outside it.
/// Interior vertices are row-major; border vertices run counter-clockwise from `(0, 0)`.
fn slot(side: u32, i: u32, j: u32) -> Option<Slot> {
    let s = side;
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

/// Whether `scale` can be a tile's scale: `+0.0`, or finite, positive, normal and at most
/// [`MAX_SCALE`], so every interior value decodes finite.
///
/// `-0.0` is not one: it would decode every interior value to a signed zero, so equal tiles
/// could differ in their bits. Callers that accept it normalise it to `+0.0` first.
fn valid_scale(scale: f32) -> bool {
    scale.to_bits() == 0 || (f32::MIN_POSITIVE..=MAX_SCALE).contains(&scale)
}

/// Whether `v` has a `-0.0` component.
fn has_negative_zero(v: Vec3) -> bool {
    [v.x, v.y, v.z]
        .iter()
        .any(|c| c.to_bits() == (-0.0_f32).to_bits())
}

impl<'a> CorrectionView<'a> {
    /// A view of a tile with `side` vertices per side over `interior` and `border`, which may be
    /// longer than the tile needs (a slot sized for the largest tile): the view uses the first
    /// `(side − 2)²` and `4(side − 1)` entries.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] for a side outside `2..=65`, or
    /// [`CorrectionError::BadParts`] if either slice is too short or the scale is negative
    /// (`-0.0` included), subnormal, not finite or so large its steps decode to infinity.
    pub fn new(
        side: u32,
        scale: f32,
        interior: &'a [[i16; 3]],
        border: &'a [Vec3],
    ) -> Result<Self, CorrectionError> {
        let (ni, nb) = counts(side)?;
        let bad = CorrectionError::BadParts { side };
        if !valid_scale(scale) {
            return Err(bad);
        }
        Ok(Self {
            side,
            scale,
            interior: interior.get(..ni).ok_or(bad)?,
            border: border.get(..nb).ok_or(bad)?,
        })
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

    /// The largest error per interior component after one quantization: half a step. As
    /// [`CorrectionTile::max_error`].
    #[must_use]
    pub const fn max_error(&self) -> f32 {
        self.scale * 0.5
    }

    /// Whether every correction is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.interior.iter().all(|q| *q == [0; 3]) && self.border.iter().all(|v| *v == Vec3::ZERO)
    }

    /// Checks the values against the rules every tile keeps, the ones a tile written only through
    /// [`CorrectionViewMut::set`] keeps by construction:
    ///
    /// - the scale is `+0.0`, or finite, positive and normal, and small enough that every interior
    ///   value decodes finite;
    /// - a zero scale has a zero interior (anything else would decode to zero);
    /// - no interior value is `i16::MIN`, which quantization never produces (its range is
    ///   symmetric, `±32767`);
    /// - every border value is finite and none is `-0.0`, so equal corrections have equal bits.
    ///
    /// [`CorrectionTile::from_parts`] and every decode run these; views otherwise trust their
    /// storage.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadParts`] if any rule is broken.
    pub fn validate(&self) -> Result<(), CorrectionError> {
        let ok = valid_scale(self.scale)
            && (self.scale != 0.0 || self.interior.iter().all(|q| *q == [0; 3]))
            && self.interior.iter().flatten().all(|q| *q != i16::MIN)
            && self
                .border
                .iter()
                .all(|v| v.is_finite() && !has_negative_zero(*v));
        if ok {
            Ok(())
        } else {
            Err(CorrectionError::BadParts { side: self.side })
        }
    }

    /// A [`CorrectionTile`] holding a copy of this tile: the same side, scale, interior and
    /// border, so the same bits for every vertex. For editing a tile away from its storage (a
    /// brush replaying several dabs on one tile), then writing it back whole.
    #[must_use]
    pub fn to_tile(self) -> CorrectionTile {
        CorrectionTile {
            side: self.side,
            scale: self.scale,
            interior: self.interior.to_vec(),
            border: self.border.to_vec(),
        }
    }

    /// The correction at `(i, j)`, or `None` outside the tile.
    #[must_use]
    pub fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        match slot(self.side, i, j)? {
            Slot::Interior(k) => self.interior.get(k).map(|q| decode(*q, self.scale)),
            Slot::Border(k) => self.border.get(k).copied(),
        }
    }
}

impl<'a> CorrectionViewMut<'a> {
    /// A mutable view; the same rules as [`CorrectionView::new`], except that a `-0.0` scale is
    /// normalised to `+0.0` in place rather than rejected (the storage is the view's to write).
    ///
    /// # Errors
    ///
    /// As [`CorrectionView::new`].
    pub fn new(
        side: u32,
        scale: &'a mut f32,
        interior: &'a mut [[i16; 3]],
        border: &'a mut [Vec3],
    ) -> Result<Self, CorrectionError> {
        let (ni, nb) = counts(side)?;
        let bad = CorrectionError::BadParts { side };
        *scale += 0.0;
        if !valid_scale(*scale) {
            return Err(bad);
        }
        Ok(Self {
            side,
            scale,
            interior: interior.get_mut(..ni).ok_or(bad)?,
            border: border.get_mut(..nb).ok_or(bad)?,
        })
    }

    /// The read-only view of the same tile.
    #[must_use]
    pub fn as_view(&self) -> CorrectionView<'_> {
        CorrectionView {
            side: self.side,
            scale: *self.scale,
            interior: self.interior,
            border: self.border,
        }
    }

    /// The correction at `(i, j)`, or `None` outside the tile.
    #[must_use]
    pub fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        self.as_view().get(i, j)
    }

    /// The gate for values written straight into the storage (a decode): turns `-0.0` border
    /// components into `+0.0` (the scale was normalised by [`CorrectionViewMut::new`]), then
    /// checks [`CorrectionView::validate`]'s rules.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadParts`] as [`CorrectionView::validate`]. The border is normalised
    /// either way.
    pub fn normalize_and_validate(&mut self) -> Result<(), CorrectionError> {
        for v in self.border.iter_mut() {
            *v = without_negative_zero(*v);
        }
        self.as_view().validate()
    }

    /// Sets the correction at `(i, j)`.
    ///
    /// If an interior value doesn't fit the current scale, the scale grows geometrically (doubling
    /// until it fits; from zero it becomes exactly what the value needs) and the existing
    /// interior is re-quantized. Each growth adds at most half the new step of error to earlier
    /// values, so after any number of growths they are within one
    /// final [`CorrectionView::scale`] of what was stored, not within
    /// [`CorrectionView::max_error`]. Interior values below a tiny threshold on a zero-scale
    /// tile are stored as zero.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::OutOfTile`] or [`CorrectionError::NotFinite`];
    /// [`CorrectionError::TooLarge`] for an interior value no scale can hold with every step
    /// decoding finite. Nothing is written then.
    pub fn set(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CorrectionError> {
        if !c.is_finite() {
            return Err(CorrectionError::NotFinite);
        }
        if let Some(Slot::Interior(_)) = slot(self.side, i, j) {
            self.fit_scale(c)?;
        }
        self.store(i, j, c)
    }

    /// Rewrites the interior vertices of row `j` (`1..side - 1`, a row inside the tile; any
    /// `j` is checked against that without overflow, as `side` is at least 2), in order: `f` sees
    /// each vertex's coordinates and its correction as stored, and where it returns a new
    /// value, that is set as [`CorrectionViewMut::set`] sets it, before `f` sees
    /// the next vertex.
    ///
    /// The same bits as [`CorrectionViewMut::get`] then [`CorrectionViewMut::set`] for each
    /// vertex in that order (a value that outgrows the scale re-quantizes the interior, which
    /// the vertices after it then read), without finding each vertex's place in the tile again
    /// for the read and the write: what an edit of a whole interior, row by row, costs per
    /// vertex.
    ///
    /// It stops at the first value `f` returns that isn't finite, or is too large to store,
    /// which isn't written; the writes before it stay. The outcome says whether anything was
    /// written, and why it stopped if it did: [`CorrectionError::NotFinite`],
    /// [`CorrectionError::TooLarge`] (as [`CorrectionViewMut::set`]), or
    /// [`CorrectionError::OutOfTile`] for a row that isn't inside the tile (nothing is read
    /// then).
    pub fn update_interior_row(
        &mut self,
        j: u32,
        mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> RowUpdate {
        let side = self.side;
        let mut outcome = RowUpdate {
            wrote: false,
            error: None,
        };
        let first = (j >= 1 && j < side - 1)
            .then(|| usize::try_from((j - 1) * (side - 2)).ok())
            .flatten();
        let Some(first) = first else {
            outcome.error = Some(CorrectionError::OutOfTile { i: 0, j, side });
            return outcome;
        };
        for (i, k) in (1..side - 1).zip(first..) {
            let Some(&q) = self.interior.get(k) else {
                outcome.error = Some(CorrectionError::OutOfTile { i, j, side });
                return outcome;
            };
            let Some(c) = f(i, j, decode(q, *self.scale)) else {
                continue;
            };
            if !c.is_finite() {
                outcome.error = Some(CorrectionError::NotFinite);
                return outcome;
            }
            if let Err(e) = self.fit_scale(c) {
                outcome.error = Some(e);
                return outcome;
            }
            let scale = *self.scale;
            if let Some(stored) = self.interior.get_mut(k) {
                *stored = [
                    quantize(c.x, scale),
                    quantize(c.y, scale),
                    quantize(c.z, scale),
                ];
            }
            outcome.wrote = true;
        }
        outcome
    }

    /// Grows the scale, re-quantizing the interior, if interior value `c` (finite) doesn't fit
    /// it: as [`CorrectionViewMut::set`] describes.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::TooLarge`] if `c` needs a scale above [`MAX_SCALE`]; the tile is
    /// unchanged then.
    fn fit_scale(&mut self, c: Vec3) -> Result<(), CorrectionError> {
        let needed = max_abs(c);
        let scale = *self.scale;
        if needed > Q_MAX * scale {
            if needed > Q_MAX * MAX_SCALE {
                return Err(CorrectionError::TooLarge);
            }
            if scale == 0.0 {
                if needed >= TINY {
                    self.requantize(needed / Q_MAX);
                }
            } else {
                self.requantize(grown_scale(scale, needed));
            }
        }
        Ok(())
    }

    /// Writes `value` with the current scale.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::OutOfTile`] if `(i, j)` is outside the tile. The `get_mut` misses below
    /// are unreachable: `slot` only returns indices inside the slices `counts` sized.
    fn store(&mut self, i: u32, j: u32, value: Vec3) -> Result<(), CorrectionError> {
        let side = self.side;
        let outside = CorrectionError::OutOfTile { i, j, side };
        match slot(side, i, j).ok_or(outside)? {
            Slot::Interior(index) => {
                let scale = *self.scale;
                *self.interior.get_mut(index).ok_or(outside)? = [
                    quantize(value.x, scale),
                    quantize(value.y, scale),
                    quantize(value.z, scale),
                ];
            }
            Slot::Border(index) => {
                *self.border.get_mut(index).ok_or(outside)? = without_negative_zero(value);
            }
        }
        Ok(())
    }

    /// Makes `scale` the tile's scale and re-quantizes every interior value to it.
    fn requantize(&mut self, scale: f32) {
        let old = *self.scale;
        *self.scale = scale;
        for q in self.interior.iter_mut() {
            let v = decode(*q, old);
            *q = [
                quantize(v.x, scale),
                quantize(v.y, scale),
                quantize(v.z, scale),
            ];
        }
    }
}

impl CorrectionTile {
    /// A tile of zero corrections.
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`] below 2.
    pub fn zero(side: u32) -> Result<Self, CorrectionError> {
        let (interior, border) = counts(side)?;
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
    /// length, [`CorrectionError::NotFinite`], or [`CorrectionError::TooLarge`] if an interior
    /// value needs a scale above the largest a tile can have.
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
            .filter(|((i, j), _)| matches!(slot(side, *i, *j), Some(Slot::Interior(_))))
            .map(|(_, v)| max_abs(*v))
            .fold(0.0, f32::max);
        if peak > Q_MAX * MAX_SCALE {
            return Err(CorrectionError::TooLarge);
        }
        tile.scale = if peak < TINY { 0.0 } else { peak / Q_MAX };
        let mut view = tile.view_mut();
        cells
            .into_iter()
            .try_for_each(|((i, j), v)| view.store(i, j, *v))?;
        Ok(tile)
    }

    /// A tile from stored parts (the file format's decode path). `-0.0` in the scale or the
    /// border becomes `+0.0`; everything else must pass [`CorrectionView::validate`].
    ///
    /// # Errors
    ///
    /// [`CorrectionError::BadSide`], or [`CorrectionError::BadParts`] if the lengths don't match
    /// `side` or a rule of [`CorrectionView::validate`] is broken.
    pub fn from_parts(
        side: u32,
        scale: f32,
        interior: Vec<[i16; 3]>,
        border: Vec<Vec3>,
    ) -> Result<Self, CorrectionError> {
        let (ni, nb) = counts(side)?;
        if interior.len() != ni || border.len() != nb {
            return Err(CorrectionError::BadParts { side });
        }
        let mut tile = Self {
            side,
            scale,
            interior,
            border,
        };
        CorrectionViewMut::new(side, &mut tile.scale, &mut tile.interior, &mut tile.border)?
            .normalize_and_validate()?;
        Ok(tile)
    }

    /// The tile as a [`CorrectionView`].
    #[must_use]
    pub fn view(&self) -> CorrectionView<'_> {
        CorrectionView {
            side: self.side,
            scale: self.scale,
            interior: &self.interior,
            border: &self.border,
        }
    }

    /// The tile as a [`CorrectionViewMut`].
    #[must_use]
    pub fn view_mut(&mut self) -> CorrectionViewMut<'_> {
        CorrectionViewMut {
            side: self.side,
            scale: &mut self.scale,
            interior: &mut self.interior,
            border: &mut self.border,
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
        self.view().is_zero()
    }

    /// The correction at `(i, j)`, or `None` outside the tile.
    #[must_use]
    pub fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        self.view().get(i, j)
    }

    /// Sets the correction at `(i, j)`, growing the scale as [`CorrectionViewMut::set`] does:
    /// after a growth, earlier interior values are within one final [`CorrectionTile::scale`] of
    /// what was stored, not within [`CorrectionTile::max_error`].
    ///
    /// # Errors
    ///
    /// As [`CorrectionViewMut::set`].
    pub fn set(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CorrectionError> {
        self.view_mut().set(i, j, c)
    }
}

/// Grows `scale` by the smallest power of two that makes `needed` fit, staying at most
/// [`MAX_SCALE`].
///
/// [`MAX_SCALE`] itself if no power of two below it does, which fits any `needed` up to
/// `Q_MAX × MAX_SCALE` (the caller refuses more).
fn grown_scale(scale: f32, needed: f32) -> f32 {
    (1..=128)
        .map(|k| scale * 2.0_f32.powi(k))
        .find(|grown| *grown <= MAX_SCALE && grown * Q_MAX >= needed)
        .unwrap_or(MAX_SCALE)
}

/// Maps `-0.0` components to `+0.0`, so equal corrections have equal bits.
fn without_negative_zero(v: Vec3) -> Vec3 {
    Vec3::new(v.x + 0.0, v.y + 0.0, v.z + 0.0)
}

/// The largest component magnitude: what a value needs of the interior's `±32767` steps.
fn max_abs(v: Vec3) -> f32 {
    v.x.abs().max(v.y.abs()).max(v.z.abs())
}

/// `x` in steps of `scale`, rounded half away from zero and clamped to `±32767`; zero for a
/// zero scale.
///
/// The same as `(x / scale).round().clamp(-Q_MAX, Q_MAX) as i16`, without [`f32::round`],
/// which is a library call on targets without a rounding instruction (x86-64 before SSE4.1):
/// clamping first changes nothing, as rounding is monotonic and the bounds are whole; the cast
/// truncates toward zero, and the fraction left, `v - trunc(v)`, is exact for `|v| < 2²³`, so
/// comparing it with `±0.5` decides the rounding exactly. (A NaN casts to 0 either way.) Checked
/// against the library rounding for every `f32` at scale 1.
#[expect(
    clippy::cast_possible_truncation,
    reason = "clamped to ±32767 first; the cast truncates toward zero, as intended"
)]
fn quantize(x: f32, scale: f32) -> i16 {
    if scale <= 0.0 {
        return 0;
    }
    let v = (x / scale).clamp(-Q_MAX, Q_MAX);
    let toward_zero = v as i16;
    let fraction = v - f32::from(toward_zero);
    toward_zero + i16::from(fraction >= 0.5) - i16::from(fraction <= -0.5)
}

/// A quantized interior value back in tangent-frame units.
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

    /// What [`quantize`] must give: the library's rounding, then the clamp.
    #[expect(clippy::cast_possible_truncation, reason = "rounded and clamped first")]
    fn quantize_by_rounding(x: f32, scale: f32) -> i16 {
        if scale <= 0.0 {
            return 0;
        }
        (x / scale).round().clamp(-Q_MAX, Q_MAX) as i16
    }

    #[test]
    fn quantize_rounds_half_away_from_zero_as_the_library_does() {
        let mut values = vec![
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN_POSITIVE,
            Q_MAX,
            Q_MAX + 0.5,
            32_766.5,
            65_535.0,
        ];
        // Each half, and the floats either side of it, across the whole range.
        for k in (-32_768..=32_768).step_by(97).chain(-3..=3) {
            #[expect(clippy::cast_precision_loss, reason = "|k| < 2^24 converts exactly")]
            let half = k as f32 + 0.5;
            values.extend([half, half.next_up(), half.next_down()]);
        }
        let mut rng = Rng::new(0x0071_7AB5);
        values.extend((0..100_000).map(|_| rng.signed() * 40_000.0));
        for scale in [1.0, 0.25, 3.0e-5, 1.0e-7, 7.0] {
            for &v in &values {
                let x = v * scale;
                assert_eq!(
                    quantize(x, scale),
                    quantize_by_rounding(x, scale),
                    "{x:e} at scale {scale:e}"
                );
            }
        }
        assert_eq!(quantize(1.0, 0.0), 0, "zero scale");
    }

    #[test]
    fn slots_are_a_bijection_onto_the_storage() {
        for side in [2_u32, 3, 65] {
            let tile = CorrectionTile::zero(side).unwrap();
            let mut interior = vec![false; tile.interior.len()];
            let mut border = vec![false; tile.border.len()];
            for j in 0..side {
                for i in 0..side {
                    let (seen, k) = match slot(side, i, j).unwrap() {
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
        let last: u32 = 4;
        let ring = |n: u32| Slot::Border(usize::try_from(n).unwrap());
        assert_eq!(slot(5, 1, 1), Some(Slot::Interior(0)), "first interior");
        assert_eq!(slot(5, 2, 1), Some(Slot::Interior(1)), "row-major");
        assert_eq!(slot(5, 0, 0), Some(Slot::Border(0)), "corner 0");
        assert_eq!(slot(5, last, 0), Some(ring(last)), "corner 1");
        assert_eq!(slot(5, last, last), Some(ring(2 * last)), "corner 2");
        assert_eq!(slot(5, 0, last), Some(ring(3 * last)), "corner 3");
        assert_eq!(
            slot(5, last - 1, last),
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
    fn from_parts_rejects_scales_that_lose_the_interior() {
        let border = || vec![Vec3::ZERO; 8];
        assert_eq!(
            CorrectionTile::from_parts(3, 0.0, vec![[1, 0, 0]], border()),
            Err(CorrectionError::BadParts { side: 3 }),
            "zero scale with a nonzero interior would decode to zero"
        );
        assert_eq!(
            CorrectionTile::from_parts(3, 1e-40, vec![[0; 3]], border()),
            Err(CorrectionError::BadParts { side: 3 }),
            "subnormal scale"
        );
        assert!(
            CorrectionTile::from_parts(3, 0.0, vec![[0; 3]], border()).is_ok(),
            "zero scale, zero interior"
        );
        assert!(
            CorrectionTile::from_parts(3, f32::MIN_POSITIVE, vec![[1, 0, 0]], border()).is_ok(),
            "smallest normal scale"
        );
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
    fn updating_an_interior_row_equals_getting_and_setting_each_vertex() {
        let mut rng = Rng::new(23);
        for case in 0..400_u32 {
            let side = 3 + rng.below(63);
            let amplitude = 10f32.powf(rng.signed() * 3.0);
            let values: Vec<Vec3> = (0..side * side).map(|_| rng.vec3(amplitude)).collect();
            let start = if case % 5 == 0 {
                CorrectionTile::zero(side).unwrap()
            } else {
                CorrectionTile::from_values(side, &values).unwrap()
            };
            let j = 1 + rng.below(side - 2);
            let seed = rng.next_u64();
            // Pure in its arguments, as the edit's `f` must be: no write, a small change, a
            // change that outgrows the scale, from time to time a value that isn't finite.
            let f = |i: u32, row: u32, c: Vec3| {
                let mut r = Rng::new(seed ^ (u64::from(i) << 20) ^ u64::from(row));
                match r.below(30) {
                    0..10 => None,
                    10..13 => Some(c + r.vec3(amplitude * 1000.0)),
                    13 if case % 7 == 0 => Some(Vec3::new(f32::NAN, 0.0, 0.0)),
                    _ => Some(c + r.vec3(amplitude * 0.1)),
                }
            };
            let mut fast = start.clone();
            let got = fast.view_mut().update_interior_row(j, f);
            let mut slow = start.clone();
            let mut want = RowUpdate {
                wrote: false,
                error: None,
            };
            let mut view = slow.view_mut();
            for i in 1..side - 1 {
                let Some(c) = f(i, j, view.get(i, j).unwrap()) else {
                    continue;
                };
                if let Err(e) = view.set(i, j, c) {
                    want.error = Some(e);
                    break;
                }
                want.wrote = true;
            }
            assert_eq!(got, want, "case {case}: the same outcome");
            assert_eq!(
                fast.view().scale().to_bits(),
                slow.view().scale().to_bits(),
                "case {case}: the same scale"
            );
            assert_eq!(fast, slow, "case {case}: the same values");
        }
        let mut tile = CorrectionTile::zero(5).unwrap();
        for j in [0, 4, 9, u32::MAX] {
            let outcome = tile.view_mut().update_interior_row(j, |_, _, c| Some(c));
            assert_eq!(
                outcome,
                RowUpdate {
                    wrote: false,
                    error: Some(CorrectionError::OutOfTile { i: 0, j, side: 5 }),
                },
                "row {j} isn't inside the tile"
            );
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

    #[test]
    fn a_border_only_tile_is_not_zero() {
        for c in [Vec3::new(1e-30, 0.0, 0.0), Vec3::new(0.0, 0.0, -2.0)] {
            let mut tile = CorrectionTile::zero(5).unwrap();
            tile.set(4, 2, c).unwrap();
            assert_eq!(tile.scale(), 0.0, "borders don't set the scale");
            assert!(!tile.is_zero(), "{c:?} on the border");
            assert!(!tile.view().is_zero(), "{c:?}: the view agrees");
        }
    }

    #[test]
    fn growth_from_zero_is_exactly_what_the_value_needs() {
        for needed in [1.0_f32, 0.003, 1234.5] {
            let mut tile = CorrectionTile::zero(9).unwrap();
            tile.set(3, 5, Vec3::new(0.0, -needed, 0.0)).unwrap();
            assert_eq!(
                tile.scale().to_bits(),
                (needed / 32767.0).to_bits(),
                "needed {needed}"
            );
        }
    }

    #[test]
    fn a_value_that_fits_exactly_does_not_grow_the_scale() {
        let mut tile = CorrectionTile::zero(9).unwrap();
        tile.set(2, 2, Vec3::new(32767.0, 0.0, 0.0)).unwrap();
        assert_eq!(tile.scale(), 1.0, "from zero: 32767 needs a step of 1");
        tile.set(4, 4, Vec3::new(0.0, 0.0, -32767.0)).unwrap();
        assert_eq!(tile.scale(), 1.0, "32767 steps of 1 fit: no growth");
    }

    /// Every interior vertex of a tile of `side`, row-major.
    fn interior_cells(side: u32) -> impl Iterator<Item = (u32, u32)> {
        (1..side - 1).flat_map(move |j| (1..side - 1).map(move |i| (i, j)))
    }

    #[test]
    fn a_view_copied_to_a_tile_is_the_same_tile() {
        let side = 9;
        let mut tile = CorrectionTile::zero(side).unwrap();
        let mut rng = Rng::new(5);
        let cells = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        for (i, j) in cells {
            tile.set(i, j, rng.vec3(0.02)).unwrap();
        }
        let copy = tile.view().to_tile();
        assert_eq!(copy, tile, "side, scale, interior and border");
        let mut edited = copy;
        edited.set(4, 4, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        assert_ne!(edited, tile, "the copy is its own");
    }

    #[test]
    fn growth_requantizes_every_interior_value() {
        let side = 9;
        let mut tile = CorrectionTile::zero(side).unwrap();
        let mut rng = Rng::new(3);
        let written: Vec<((u32, u32), Vec3)> = interior_cells(side)
            .map(|cell| (cell, rng.vec3(0.01)))
            .collect();
        for &((i, j), c) in &written {
            tile.set(i, j, c).unwrap();
        }
        let before = tile.scale();
        // A value far outside the scale, at the last interior vertex, so the first (index 0) is
        // only touched by the re-quantization.
        tile.set(side - 2, side - 2, Vec3::new(50.0, 0.0, 0.0))
            .unwrap();
        assert!(tile.scale() > before, "the scale grew");
        for &((i, j), c) in written
            .iter()
            .filter(|(cell, _)| *cell != (side - 2, side - 2))
        {
            let got = tile.get(i, j).unwrap();
            for (w, g) in [(c.x, got.x), (c.y, got.y), (c.z, got.z)] {
                assert!(
                    (w - g).abs() <= tile.scale(),
                    "({i}, {j}): {g} is not within one scale of {w}"
                );
            }
        }
    }

    #[test]
    fn repeated_growth_keeps_every_value_within_one_final_scale() {
        let side = 17;
        let mut rng = Rng::new(19);
        let mut tile = CorrectionTile::zero(side).unwrap();
        let mut written = std::collections::BTreeMap::new();
        for (i, j) in interior_cells(side) {
            let c = rng.vec3(1e-3);
            tile.set(i, j, c).unwrap();
            written.insert((i, j), c);
        }
        // Grow many times, by small factors and large, at random vertices.
        let mut amplitude = 1e-3_f32;
        for _ in 0..60 {
            amplitude *= 1.0 + 3.0 * rng.unit();
            let (i, j) = (1 + rng.below(side - 2), 1 + rng.below(side - 2));
            let c = Vec3::new(amplitude, rng.signed() * amplitude, 0.0);
            tile.set(i, j, c).unwrap();
            written.insert((i, j), c);
        }
        let scale = tile.scale();
        for ((i, j), c) in written {
            let got = tile.get(i, j).unwrap();
            for (w, g) in [(c.x, got.x), (c.y, got.y), (c.z, got.z)] {
                assert!(
                    (w - g).abs() <= scale,
                    "({i}, {j}): |{w} − {g}| > final scale {scale}"
                );
            }
        }
    }

    #[test]
    fn validate_holds_the_value_rules() {
        let bad = Err(CorrectionError::BadParts { side: 3 });
        let check = |scale: f32, q: [i16; 3], b: Vec3| {
            let mut border = vec![Vec3::ZERO; 8];
            border[3] = b;
            CorrectionView::new(3, scale, &[q], &border)?.validate()
        };
        assert_eq!(
            check(1.0, [1, 2, 3], Vec3::new(0.5, 0.0, 0.0)),
            Ok(()),
            "valid"
        );
        assert_eq!(check(0.0, [0; 3], Vec3::ZERO), Ok(()), "zero");
        assert_eq!(
            check(0.0, [0, 1, 0], Vec3::ZERO),
            bad,
            "zero scale, nonzero interior"
        );
        assert_eq!(check(1.0, [0, 0, i16::MIN], Vec3::ZERO), bad, "i16::MIN");
        assert_eq!(
            check(1.0, [0; 3], Vec3::new(0.0, f32::NAN, 0.0)),
            bad,
            "NaN border"
        );
        assert_eq!(
            check(1.0, [0; 3], Vec3::new(0.0, 0.0, f32::NEG_INFINITY)),
            bad,
            "infinite border"
        );
        assert_eq!(
            check(1.0, [0; 3], Vec3::new(-0.0, 0.0, 0.0)),
            bad,
            "-0.0 border"
        );
    }

    #[test]
    fn a_negative_zero_scale_is_normalised_or_rejected() {
        let (_, mut interior, mut border) = full_slot();
        assert_eq!(
            CorrectionView::new(3, -0.0, &interior, &border).unwrap_err(),
            CorrectionError::BadParts { side: 3 },
            "a read-only view can't normalise it, so it rejects it"
        );
        let mut scale = -0.0_f32;
        let mut view = CorrectionViewMut::new(3, &mut scale, &mut interior, &mut border).unwrap();
        assert_eq!(
            view.as_view().scale().to_bits(),
            0,
            "a mutable view normalises it"
        );
        view.normalize_and_validate().unwrap();
        assert_eq!(scale.to_bits(), 0, "in the storage");
    }

    #[test]
    fn normalize_and_validate_clears_negative_zero_borders_only() {
        let (mut scale, mut interior, mut border) = full_slot();
        border[2] = Vec3::new(-0.0, 1.0, -0.0);
        border[5] = Vec3::new(-3.0, 0.0, 0.0);
        let mut view = CorrectionViewMut::new(3, &mut scale, &mut interior, &mut border).unwrap();
        view.normalize_and_validate().unwrap();
        assert_eq!(
            border[2].to_bits(),
            Vec3::new(0.0, 1.0, 0.0).to_bits(),
            "-0.0 → +0.0"
        );
        assert_eq!(border[5].x, -3.0, "other values kept");
        border[1].y = f32::INFINITY;
        let mut infinite =
            CorrectionViewMut::new(3, &mut scale, &mut interior, &mut border).unwrap();
        assert_eq!(
            infinite.normalize_and_validate(),
            Err(CorrectionError::BadParts { side: 3 }),
            "still validated"
        );
    }

    /// The largest tile's storage, as a pool slot holds it.
    fn full_slot() -> (f32, Vec<[i16; 3]>, Vec<Vec3>) {
        let (ni, nb) = counts(TILE_QUADS + 1).unwrap();
        (0.0, vec![[0; 3]; ni], vec![Vec3::ZERO; nb])
    }

    #[test]
    fn a_view_over_a_full_slot_behaves_like_a_small_tile() {
        let bits = |v: Option<Vec3>| v.unwrap().to_bits();
        for side in [2, 3, 9, 33, 65] {
            let (mut scale, mut interior, mut border) = full_slot();
            let mut view =
                CorrectionViewMut::new(side, &mut scale, &mut interior, &mut border).unwrap();
            let mut tile = CorrectionTile::zero(side).unwrap();
            let mut rng = Rng::new(u64::from(side));
            for step in 0..400 {
                let (i, j) = (rng.below(side), rng.below(side));
                // Amplitudes over six decades, so the scale grows from zero and then doubles.
                let amplitude = 10f32.powf(rng.signed() * 3.0);
                let c = rng.vec3(amplitude);
                tile.set(i, j, c).unwrap();
                view.set(i, j, c).unwrap();
                assert_eq!(
                    view.as_view().scale().to_bits(),
                    tile.scale().to_bits(),
                    "side {side} step {step}: scale"
                );
            }
            for j in 0..side {
                for i in 0..side {
                    assert_eq!(
                        bits(view.get(i, j)),
                        bits(tile.get(i, j)),
                        "side {side} ({i}, {j})"
                    );
                }
            }
            let read = view.as_view();
            assert_eq!(read.side(), side, "side");
            assert_eq!(read.is_zero(), tile.is_zero(), "is_zero");
            assert_eq!(
                read.max_error().to_bits(),
                tile.max_error().to_bits(),
                "max_error"
            );
            assert_eq!(read.get(side, 0), None, "outside");
            assert_eq!(
                view.set(0, side, Vec3::ZERO),
                Err(CorrectionError::OutOfTile {
                    i: 0,
                    j: side,
                    side
                }),
                "set outside"
            );
            // The slot's entries past the tile's are untouched.
            let (ni, nb) = counts(side).unwrap();
            assert!(interior[ni..].iter().all(|q| *q == [0; 3]), "interior tail");
            assert!(border[nb..].iter().all(|v| *v == Vec3::ZERO), "border tail");
            // And a read-only view of the same storage decodes the same.
            let again = CorrectionView::new(side, scale, &interior, &border).unwrap();
            assert_eq!(
                bits(again.get(1, 1)),
                bits(tile.get(1, 1)),
                "read-only view"
            );
            assert_eq!(
                tile.view().get(side - 1, side - 1).map(Vec3::to_bits),
                again.get(side - 1, side - 1).map(Vec3::to_bits),
                "tile view"
            );
        }
    }

    #[test]
    fn views_check_shape_and_scale() {
        let (_, interior, border) = full_slot();
        assert_eq!(
            CorrectionView::new(66, 0.0, &interior, &border).unwrap_err(),
            CorrectionError::BadSide(66),
            "side above a full tile"
        );
        assert_eq!(
            CorrectionView::new(1, 0.0, &interior, &border).unwrap_err(),
            CorrectionError::BadSide(1),
            "side below 2"
        );
        assert_eq!(
            CorrectionView::new(9, 0.0, &interior[..48], &border).unwrap_err(),
            CorrectionError::BadParts { side: 9 },
            "interior one short of 7²"
        );
        assert_eq!(
            CorrectionView::new(9, 0.0, &interior, &border[..31]).unwrap_err(),
            CorrectionError::BadParts { side: 9 },
            "border one short of 4·8"
        );
        assert!(
            CorrectionView::new(9, 0.0, &interior[..49], &border[..32]).is_ok(),
            "exact lengths"
        );
        for scale in [-1.0, 1e-40, f32::NAN, f32::INFINITY] {
            assert_eq!(
                CorrectionView::new(9, scale, &interior, &border).unwrap_err(),
                CorrectionError::BadParts { side: 9 },
                "scale {scale}"
            );
            let (mut s, mut i, mut b) = (scale, interior.clone(), border.clone());
            assert_eq!(
                CorrectionViewMut::new(9, &mut s, &mut i, &mut b).unwrap_err(),
                CorrectionError::BadParts { side: 9 },
                "mutable, scale {scale}"
            );
        }
    }

    /// A stored tile with a scale past [`MAX_SCALE`] is refused, as scales not far past it
    /// decode the largest step to infinity; at [`MAX_SCALE`] every step decodes finite.
    #[test]
    fn from_parts_rejects_scales_that_decode_to_infinity() {
        assert!(
            (Q_MAX * MAX_SCALE).is_finite(),
            "the largest step decodes finite"
        );
        assert!(
            !(Q_MAX * MAX_SCALE * 4.0).is_finite(),
            "four times the cap would not"
        );
        assert_eq!(
            CorrectionTile::from_parts(
                3,
                MAX_SCALE * 2.0,
                vec![[32767, 0, 0]],
                vec![Vec3::ZERO; 8]
            )
            .unwrap_err(),
            CorrectionError::BadParts { side: 3 },
        );
        let tile =
            CorrectionTile::from_parts(3, MAX_SCALE, vec![[32767, 0, 0]], vec![Vec3::ZERO; 8])
                .unwrap();
        assert!(tile.get(1, 1).unwrap().is_finite());
    }

    /// A value no allowed scale can hold is refused before anything is written, through `set`,
    /// a row update and `from_values` alike; a large value that fits is stored.
    #[test]
    fn values_needing_a_scale_past_the_cap_are_refused() {
        let huge = Vec3::new(f32::MAX, 0.0, 0.0);
        let mut tile = CorrectionTile::zero(3).unwrap();
        assert_eq!(tile.set(1, 1, huge), Err(CorrectionError::TooLarge));
        assert_eq!(tile, CorrectionTile::zero(3).unwrap(), "unchanged");

        // From a non-zero scale, growth is capped too.
        tile.set(1, 1, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let before = tile.clone();
        assert_eq!(tile.set(1, 1, huge), Err(CorrectionError::TooLarge));
        assert_eq!(tile, before, "unchanged");
        let outcome = tile.view_mut().update_interior_row(1, |_, _, _| Some(huge));
        assert_eq!(
            outcome,
            RowUpdate {
                wrote: false,
                error: Some(CorrectionError::TooLarge)
            }
        );
        assert_eq!(tile, before, "unchanged by the row either");

        tile.set(1, 1, Vec3::new(1e30, 0.0, 0.0)).unwrap();
        assert!(tile.scale() <= MAX_SCALE);
        tile.view().validate().unwrap();
        let got = tile.get(1, 1).unwrap();
        assert!((got.x - 1e30).abs() <= tile.scale(), "{got:?}");

        // The largest value that fits, from a small scale: growth stops at the cap.
        let mut capped = CorrectionTile::zero(3).unwrap();
        capped.set(1, 1, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        capped
            .set(1, 1, Vec3::new(Q_MAX * MAX_SCALE, 0.0, 0.0))
            .unwrap();
        assert!(capped.scale() <= MAX_SCALE && capped.get(1, 1).unwrap().is_finite());

        let mut values = vec![Vec3::ZERO; 9];
        values[4] = huge;
        assert_eq!(
            CorrectionTile::from_values(3, &values).unwrap_err(),
            CorrectionError::TooLarge
        );
    }
}
