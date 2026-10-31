//! What the stroke engine reads from the pen: one [`Sample`] per pen report.
//!
//! The pen backend in `input` reports logical pixels and `Instant`s; the app converts each
//! de-jittered sample into a [`Sample`] (physical pixels, seconds on the pen's clock), because
//! `input` carries the platform's pen backends and `stroke` stays a headless lower crate that
//! depends on no windowing code.

use std::f32::consts::{FRAC_PI_2, TAU};

use crate::Unit;

/// A sample field that isn't a usable value.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum SampleError {
    /// A screen coordinate is NaN or infinite.
    #[error("the screen position ({x}, {y}) is not finite")]
    Position {
        /// The horizontal coordinate offered.
        x: f64,
        /// The vertical coordinate offered.
        y: f64,
    },
    /// The time is NaN or infinite.
    #[error("the time {0} s is not finite")]
    Time(f64),
    /// The tilt amount is outside `0..=π/2` radians, or NaN.
    #[error("the tilt amount {0} rad is outside 0..=π/2")]
    TiltAmount(f32),
    /// A direction or rotation is NaN or infinite.
    #[error("the angle {0} rad is not finite")]
    Angle(f32),
}

/// A point on the screen, in physical pixels from the top-left corner of the view.
///
/// Both coordinates are finite: [`ScreenPoint::new`] refuses anything else.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScreenPoint {
    /// Pixels right of the left edge.
    x: f64,
    /// Pixels below the top edge.
    y: f64,
}

impl ScreenPoint {
    /// The point at `(x, y)` physical pixels.
    ///
    /// # Errors
    ///
    /// [`SampleError::Position`] if either coordinate is NaN or infinite.
    pub fn new(x: f64, y: f64) -> Result<Self, SampleError> {
        if x.is_finite() && y.is_finite() {
            Ok(Self { x, y })
        } else {
            Err(SampleError::Position { x, y })
        }
    }

    /// Pixels right of the left edge.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Pixels below the top edge.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// The straight-line distance to `other`, in pixels.
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        (other.x - self.x).hypot(other.y - self.y)
    }

    /// The point a fraction `t` of the way from `self` to `to` (`0` is `self`, `1` is `to`).
    ///
    /// Only the engine's own filters call this, with a `t` in `0..=1` and finite points, so the
    /// result stays finite.
    pub(crate) fn lerp(self, to: Self, t: f64) -> Self {
        Self {
            x: self.x + (to.x - self.x) * t,
            y: self.y + (to.y - self.y) * t,
        }
    }
}

/// Which way the pen leans and how far.
///
/// The amount is the angle from upright, in `0..=π/2` radians (`0` upright, `π/2` lying flat).
/// The direction is the way the pen leans on the screen, in `0..2π` radians from `+x` (right)
/// towards `+y` (down); it is meaningless when the pen is upright and reads `0` then by
/// convention.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tilt {
    /// Radians from upright, in `0..=π/2`.
    amount: f32,
    /// Radians from `+x` towards `+y`, in `0..2π`.
    direction: f32,
}

impl Tilt {
    /// A tilt of `amount` radians from upright, leaning towards `direction` radians from `+x`
    /// towards `+y`. Any finite direction is accepted and wrapped into `0..2π`.
    ///
    /// # Errors
    ///
    /// [`SampleError::TiltAmount`] if `amount` is outside `0..=π/2` or NaN;
    /// [`SampleError::Angle`] if `direction` is not finite.
    pub fn new(amount: f32, direction: f32) -> Result<Self, SampleError> {
        if !(0.0..=FRAC_PI_2).contains(&amount) {
            return Err(SampleError::TiltAmount(amount));
        }
        Ok(Self {
            amount: amount + 0.0,
            direction: wrap_angle(direction)?,
        })
    }

    /// Radians from upright, in `0..=π/2`.
    #[must_use]
    pub const fn amount(self) -> f32 {
        self.amount
    }

    /// Radians from `+x` (right) towards `+y` (down), in `0..2π`.
    #[must_use]
    pub const fn direction(self) -> f32 {
        self.direction
    }
}

/// The pen's barrel rotation, in `0..2π` radians, for pens that report it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Twist(f32);

impl Twist {
    /// A barrel rotation of `radians`, wrapped into `0..2π`.
    ///
    /// # Errors
    ///
    /// [`SampleError::Angle`] if `radians` is not finite.
    pub fn new(radians: f32) -> Result<Self, SampleError> {
        wrap_angle(radians).map(Self)
    }

    /// The rotation, in `0..2π` radians.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Wraps a finite angle into `0..2π`.
///
/// # Errors
///
/// [`SampleError::Angle`] if `radians` is not finite.
fn wrap_angle(radians: f32) -> Result<f32, SampleError> {
    if !radians.is_finite() {
        return Err(SampleError::Angle(radians));
    }
    let wrapped = radians.rem_euclid(TAU);
    // A tiny negative angle wraps to TAU itself after rounding; that is the same direction as 0.
    Ok(if wrapped >= TAU { 0.0 } else { wrapped + 0.0 })
}

/// Pen buttons held down, as a small bit set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Buttons(u8);

impl Buttons {
    /// No button held.
    pub const NONE: Self = Self(0);
    /// The first barrel button.
    pub const PRIMARY: Self = Self(1);
    /// The second barrel button.
    pub const SECONDARY: Self = Self(1 << 1);
    /// A third button, where the pen has one.
    pub const TERTIARY: Self = Self(1 << 2);

    /// A set from its bits: bit 0 the primary button, bit 1 the secondary, bit 2 the tertiary.
    /// Higher bits are kept, for buttons the engine doesn't name.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// The set's bits, as [`Buttons::from_bits`] takes them.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every button in `other` is held.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// This set with `button` held or released.
    #[must_use]
    pub const fn with(self, button: Self, held: bool) -> Self {
        if held {
            Self(self.0 | button.0)
        } else {
            Self(self.0 & !button.0)
        }
    }
}

/// One pen sample, as the stroke engine reads it.
///
/// Built with [`Sample::new`] from the parts every pen sends, then [`Sample::with_tilt`],
/// [`Sample::with_twist`] and [`Sample::with_buttons`] for what this pen also reports. Every
/// field holds a valid value: the types of its parts refuse the rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Where the pen is, in physical pixels.
    position: ScreenPoint,
    /// When the pen sampled it: seconds on the pen's clock (de-jittered), from any fixed epoch.
    time: f64,
    /// Tip pressure.
    pressure: Unit,
    /// The pen's tilt, if it reports one.
    tilt: Option<Tilt>,
    /// The pen's barrel rotation, if it reports one (few pens do).
    twist: Option<Twist>,
    /// Barrel buttons held.
    buttons: Buttons,
}

impl Sample {
    /// A sample at `position`, taken at `time` seconds on the pen's clock, with `pressure`; no
    /// tilt, no twist and no buttons until they are added.
    ///
    /// The time's epoch doesn't matter (only differences are used), but it is the pen's clock as
    /// `input`'s de-jitter filter recovers it, not the time the sample arrived.
    ///
    /// # Errors
    ///
    /// [`SampleError::Time`] if `time` is NaN or infinite.
    pub fn new(position: ScreenPoint, time: f64, pressure: Unit) -> Result<Self, SampleError> {
        if !time.is_finite() {
            return Err(SampleError::Time(time));
        }
        Ok(Self {
            position,
            time,
            pressure,
            tilt: None,
            twist: None,
            buttons: Buttons::NONE,
        })
    }

    /// This sample with the pen's tilt.
    #[must_use]
    pub const fn with_tilt(self, tilt: Tilt) -> Self {
        Self {
            tilt: Some(tilt),
            ..self
        }
    }

    /// This sample with the pen's barrel rotation.
    #[must_use]
    pub const fn with_twist(self, twist: Twist) -> Self {
        Self {
            twist: Some(twist),
            ..self
        }
    }

    /// This sample with the buttons held.
    #[must_use]
    pub const fn with_buttons(self, buttons: Buttons) -> Self {
        Self { buttons, ..self }
    }

    /// This sample moved to `position`, everything else kept: how the path hands on its
    /// smoothed point with the pen's other readings.
    pub(crate) const fn at(self, position: ScreenPoint) -> Self {
        Self { position, ..self }
    }

    /// Where the pen is, in physical pixels.
    #[must_use]
    pub const fn position(&self) -> ScreenPoint {
        self.position
    }

    /// When the pen sampled it, in seconds on the pen's clock.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Tip pressure.
    #[must_use]
    pub const fn pressure(&self) -> Unit {
        self.pressure
    }

    /// The pen's tilt, or `None` if it doesn't report one.
    #[must_use]
    pub const fn tilt(&self) -> Option<Tilt> {
        self.tilt
    }

    /// The pen's barrel rotation, or `None` if it doesn't report one.
    #[must_use]
    pub const fn twist(&self) -> Option<Twist> {
        self.twist
    }

    /// Barrel buttons held.
    #[must_use]
    pub const fn buttons(&self) -> Buttons {
        self.buttons
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    #[test]
    fn positions_and_times_must_be_finite() {
        assert!(ScreenPoint::new(f64::NAN, 0.0).is_err(), "NaN x");
        assert!(ScreenPoint::new(0.0, f64::INFINITY).is_err(), "infinite y");
        let at = ScreenPoint::new(3.0, 4.0).unwrap();
        assert!(Sample::new(at, f64::NAN, Unit::ONE).is_err(), "NaN time");
        let sample = Sample::new(at, 1.5, Unit::ONE).unwrap();
        assert_eq!(sample.position(), at, "position kept");
        assert!(
            sample.tilt().is_none() && sample.twist().is_none(),
            "none yet"
        );
    }

    #[test]
    fn tilt_is_checked_and_its_direction_wrapped() {
        assert!(Tilt::new(-0.1, 0.0).is_err(), "negative amount");
        assert!(Tilt::new(FRAC_PI_2 + 0.01, 0.0).is_err(), "past flat");
        assert!(Tilt::new(f32::NAN, 0.0).is_err(), "NaN amount");
        assert!(Tilt::new(0.5, f32::INFINITY).is_err(), "infinite direction");
        let tilt = Tilt::new(0.5, -PI / 2.0).unwrap();
        assert!(
            (tilt.direction() - 1.5 * PI).abs() < 1e-5,
            "-π/2 wraps to 3π/2: {}",
            tilt.direction()
        );
        let tiny = Tilt::new(0.5, -1e-9).unwrap();
        assert!(
            (0.0..TAU).contains(&tiny.direction()),
            "a tiny negative angle stays below 2π: {}",
            tiny.direction()
        );
    }

    #[test]
    fn twist_wraps() {
        assert!(Twist::new(f32::NAN).is_err(), "NaN");
        let twist = Twist::new(TAU + 1.0).unwrap();
        assert!((twist.get() - 1.0).abs() < 1e-5, "wrapped: {}", twist.get());
    }

    #[test]
    fn buttons_are_a_set() {
        let held = Buttons::NONE
            .with(Buttons::PRIMARY, true)
            .with(Buttons::TERTIARY, true);
        assert!(held.contains(Buttons::PRIMARY), "primary held");
        assert!(!held.contains(Buttons::SECONDARY), "secondary not held");
        assert_eq!(Buttons::from_bits(held.bits()), held, "bits round-trip");
        assert_eq!(
            held.with(Buttons::PRIMARY, false),
            Buttons::TERTIARY,
            "released"
        );
    }

    #[test]
    fn lerp_and_distance() {
        let a = ScreenPoint::new(0.0, 0.0).unwrap();
        let b = ScreenPoint::new(3.0, 4.0).unwrap();
        assert!((a.distance(b) - 5.0).abs() < 1e-12, "3-4-5");
        let mid = a.lerp(b, 0.5);
        assert!(
            (mid.x() - 1.5).abs() < 1e-12 && (mid.y() - 2.0).abs() < 1e-12,
            "half way"
        );
    }
}
