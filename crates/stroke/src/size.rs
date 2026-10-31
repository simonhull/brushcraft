//! The brush's size: a [`Radius`], and whether it is measured on the screen or in the world.

/// A radius that isn't positive and finite.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("a radius must be positive and finite, not {0}")]
pub struct RadiusError(pub f32);

/// A brush radius: positive and finite.
///
/// Its unit depends on the brush's [`SizeMode`]: physical pixels for [`SizeMode::Screen`], world
/// units for [`SizeMode::World`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Radius(f32);

impl Radius {
    /// The default brush's radius: 40, in screen pixels as [`SizeMode::Screen`] measures it.
    pub const DEFAULT: Self = Self(40.0);

    /// The radius `value`.
    ///
    /// # Errors
    ///
    /// [`RadiusError`] if `value` is zero, negative, infinite or NaN.
    pub fn new(value: f32) -> Result<Self, RadiusError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(RadiusError(value))
        }
    }

    /// The radius, positive and finite.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Radius {
    type Error = RadiusError;

    fn try_from(value: f32) -> Result<Self, RadiusError> {
        Self::new(value)
    }
}

/// What a brush's [`Radius`] is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SizeMode {
    /// Physical pixels on the screen, the default. Zooming between strokes changes how much of the
    /// model a stroke covers; within a stroke the size is frozen at the first hit's depth, so
    /// zooming mid-stroke can't make it uneven.
    #[default]
    Screen,
    /// World units, whatever the zoom: for detail that must match across views (pores, scales).
    World,
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn a_radius_is_positive_and_finite() {
        assert!(Radius::new(0.5).is_ok(), "positive");
        assert!(Radius::new(f32::MIN_POSITIVE).is_ok(), "tiny but positive");
        for bad in [0.0, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(Radius::new(bad).is_err(), "{bad} refused");
        }
    }

    #[test]
    fn screen_is_the_default_mode() {
        assert_eq!(SizeMode::default(), SizeMode::Screen, "screen pixels");
    }
}
