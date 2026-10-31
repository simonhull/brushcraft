//! `Unit`: a value in `0.0..=1.0`, the range pressures, curve points and dynamics inputs share.

/// A value in `0.0..=1.0`, such as a pressure, a curve point or a dynamics input read on its
/// curve.
///
/// It can only hold a value in range: [`Unit::new`] refuses anything else, NaN included, and
/// [`Unit::saturating`] clamps. Negative zero is stored as zero, so equal units have equal bits.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Unit(f32);

/// A value outside `0.0..=1.0` (or NaN) offered as a [`Unit`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} is outside 0..=1")]
pub struct NotUnit(pub f32);

impl Unit {
    /// The top of the range.
    pub const ONE: Self = Self(1.0);
    /// The bottom of the range.
    pub const ZERO: Self = Self(0.0);

    /// The value as a unit.
    ///
    /// # Errors
    ///
    /// [`NotUnit`] if `value` is outside `0.0..=1.0` or NaN.
    pub fn new(value: f32) -> Result<Self, NotUnit> {
        if (0.0..=1.0).contains(&value) {
            // `+ 0.0` turns -0.0 into 0.0.
            Ok(Self(value + 0.0))
        } else {
            Err(NotUnit(value))
        }
    }

    /// `value` clamped to `0.0..=1.0`; NaN becomes zero.
    #[must_use]
    pub fn saturating(value: f32) -> Self {
        if value.is_nan() {
            Self::ZERO
        } else {
            Self(value.clamp(0.0, 1.0) + 0.0)
        }
    }

    /// The value, in `0.0..=1.0`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Unit {
    type Error = NotUnit;

    fn try_from(value: f32) -> Result<Self, NotUnit> {
        Self::new(value)
    }
}

impl From<Unit> for f32 {
    fn from(unit: Unit) -> Self {
        unit.get()
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn holds_only_values_in_range() {
        assert_eq!(Unit::new(0.25).map(Unit::get), Ok(0.25), "inside");
        assert_eq!(Unit::new(0.0), Ok(Unit::ZERO), "bottom end");
        assert_eq!(Unit::new(1.0), Ok(Unit::ONE), "top end");
        assert!(Unit::new(-0.01).is_err(), "below");
        assert!(Unit::new(1.01).is_err(), "above");
        assert!(Unit::new(f32::NAN).is_err(), "NaN");
        assert_eq!(
            Unit::new(-0.0).map(|u| u.get().to_bits()),
            Ok(0),
            "negative zero is stored as zero"
        );
    }

    #[test]
    fn saturating_clamps() {
        assert_eq!(Unit::saturating(-3.0), Unit::ZERO, "below");
        assert_eq!(Unit::saturating(7.0), Unit::ONE, "above");
        assert_eq!(Unit::saturating(f32::NAN), Unit::ZERO, "NaN");
        assert_eq!(Unit::saturating(0.5).get(), 0.5, "inside");
    }
}
