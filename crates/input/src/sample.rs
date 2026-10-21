//! What `input` hands to the stroke engine: pen samples and pad events.

use std::f64::consts::TAU;
use std::time::Instant;

/// A value normalized to `0.0..=1.0`, such as pressure or hover distance.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Unit(f32);

impl Unit {
    /// The top of the range.
    pub const ONE: Self = Self(1.0);
    /// The bottom of the range.
    pub const ZERO: Self = Self(0.0);

    /// Maps a raw 16-bit device value (`0..=65535`, as the tablet protocol sends pressure,
    /// distance and strip position) onto `0.0..=1.0`. Larger values saturate.
    #[must_use]
    pub fn from_u16_range(raw: u32) -> Self {
        let raw = u16::try_from(raw).unwrap_or(u16::MAX);
        Self(f32::from(raw) / f32::from(u16::MAX))
    }

    /// The value, in `0.0..=1.0`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// A value outside `0.0..=1.0` (or NaN) offered as a [`Unit`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("{0} is outside 0..=1")]
pub struct OutOfRange(pub f32);

impl TryFrom<f32> for Unit {
    type Error = OutOfRange;

    fn try_from(value: f32) -> Result<Self, OutOfRange> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value + 0.0))
        } else {
            Err(OutOfRange(value))
        }
    }
}

/// Pen tilt in degrees, as tablets report it: each axis in `-90.0..=90.0`, `0` upright,
/// positive `x` leaning right and positive `y` leaning towards the user (screen down).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tilt {
    x: f64,
    y: f64,
}

impl Tilt {
    /// Clamps each axis to `-90.0..=90.0`; NaN becomes `0`.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        let clamp = |v: f64| {
            if v.is_nan() {
                0.0
            } else {
                // `+ 0.0` turns -0.0 into 0.0, so an upright pen has azimuth 0, not π.
                v.clamp(-90.0, 90.0) + 0.0
            }
        };
        Self {
            x: clamp(x),
            y: clamp(y),
        }
    }

    /// Tilt along x, in degrees.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Tilt along y, in degrees.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Angle between the pen and the surface, in radians: `π/2` upright, `0` lying flat.
    #[must_use]
    pub fn altitude(self) -> f64 {
        let (tx, ty) = self.tangents();
        1.0_f64.atan2(tx.hypot(ty))
    }

    /// Direction the pen leans, in radians from `+x` towards `+y`, in `0..2π`; `0` when upright.
    #[must_use]
    pub fn azimuth(self) -> f64 {
        let (tx, ty) = self.tangents();
        ty.atan2(tx).rem_euclid(TAU)
    }

    /// `tan(90°)` is about 1.6e16 in `f64`, not infinite, so the formulas above need no special
    /// cases.
    fn tangents(self) -> (f64, f64) {
        (self.x.to_radians().tan(), self.y.to_radians().tan())
    }
}

/// A position on a surface, in logical pixels from its top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    /// Pixels right of the left edge.
    pub x: f64,
    /// Pixels below the top edge.
    pub y: f64,
}

/// The kind of tool, as the tablet reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Tool {
    /// The pen tip.
    #[default]
    Pen,
    /// The eraser end of a pen.
    Eraser,
    /// A brush-shaped tool.
    Brush,
    /// A pencil-shaped tool.
    Pencil,
    /// An airbrush.
    Airbrush,
    /// A finger on a tablet that tracks touch as a tool.
    Finger,
    /// A tablet puck.
    Mouse,
    /// A tablet lens cursor.
    Lens,
}

/// Where a sample sits in the tool's lifecycle over a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// The tool came into range over the surface.
    Enter,
    /// Moving in range, not touching.
    Hover,
    /// The tip touched the surface: a stroke starts.
    Down,
    /// Moving while touching.
    Drag,
    /// The tip lifted: the stroke ends.
    Up,
    /// The tool left range or the surface.
    Leave,
}

impl Phase {
    /// Whether this sample belongs to a stroke: from the tip touching (`Down`) through lifting
    /// (`Up`), inclusive.
    #[must_use]
    pub const fn in_stroke(self) -> bool {
        matches!(self, Self::Down | Self::Drag | Self::Up)
    }
}

/// Pen buttons held down, as a small bit set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Buttons(u8);

impl Buttons {
    /// The first barrel button.
    pub const PRIMARY: Self = Self(1);
    /// The second barrel button.
    pub const SECONDARY: Self = Self(1 << 1);
    /// A third button, where the pen has one.
    pub const TERTIARY: Self = Self(1 << 2);

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

/// Identifies the window surface a sample belongs to.
///
/// Compare it with the one `wayland::surface_id` returns for a window.
///
/// It is an address, so it is unique only among live surfaces and may be reused after one is
/// destroyed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceId(usize);

impl SurfaceId {
    /// Wraps an opaque surface identifier (a surface's address). Only compared, never
    /// dereferenced.
    #[must_use]
    pub const fn from_raw(raw: usize) -> Self {
        Self(raw)
    }
}

/// One pen sample: everything the tablet reported in one hardware frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeSample {
    /// Where this sample sits in the tool's lifecycle.
    pub phase: Phase,
    /// Which tool produced it.
    pub tool: Tool,
    /// The surface it's over.
    pub surface: SurfaceId,
    /// Position on that surface, in logical pixels (the protocol's 1/256 px precision).
    pub pos: Position,
    /// Tip pressure; zero when not touching.
    pub pressure: Unit,
    /// Hover distance, if the tablet reports it.
    pub distance: Option<Unit>,
    /// Tilt, if the tablet reports it.
    pub tilt: Option<Tilt>,
    /// Barrel rotation in degrees, if the pen reports it.
    pub roll: Option<f64>,
    /// Barrel buttons held.
    pub buttons: Buttons,
    /// When the pen sampled this, as best we can tell: de-jittered for contact phases, the
    /// receipt time otherwise.
    pub timestamp: Instant,
    /// When the input thread decoded it.
    pub received: Instant,
    /// The compositor's frame time: milliseconds, unspecified epoch, wraps. Diagnostics only.
    pub device_ms: u32,
    /// Whether this sample is a prediction rather than a measurement. Always false for now.
    pub predicted: bool,
}

/// One input from a tablet pad (the buttons, rings, strips and dials around a tablet).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadEvent {
    /// When the input thread decoded it.
    pub received: Instant,
    /// What happened.
    pub kind: PadEventKind,
}

/// Identifies one ring, strip or dial for as long as it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ControlId(u32);

impl ControlId {
    /// Wraps an opaque control identifier. Only compared, never interpreted.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// What a pad reported.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadEventKind {
    /// A pad button changed state.
    Button {
        /// The button's index on the pad.
        index: u32,
        /// Whether it's now held.
        pressed: bool,
    },
    /// A ring moved to `angle` degrees (clockwise from the top), or the finger lifted (`None`).
    Ring {
        /// Which ring.
        control: ControlId,
        /// The new angle, or `None` when the interaction stopped.
        angle: Option<f64>,
    },
    /// A strip moved to `position`, or the finger lifted (`None`).
    Strip {
        /// Which strip.
        control: ControlId,
        /// The new position, or `None` when the interaction stopped.
        position: Option<Unit>,
    },
    /// A dial turned; one notch is 120.
    Dial {
        /// Which dial.
        control: ControlId,
        /// Rotation since the last event, in 1/120 notches.
        delta: i32,
    },
}

/// Anything a backend reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// A pen sample.
    Sample(StrokeSample),
    /// A pad input.
    Pad(PadEvent),
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use super::*;

    #[test]
    fn unit_maps_the_16_bit_range() {
        assert_eq!(Unit::from_u16_range(0), Unit::ZERO);
        assert_eq!(Unit::from_u16_range(65_535), Unit::ONE);
        assert_eq!(
            Unit::from_u16_range(1_000_000),
            Unit::ONE,
            "above the range saturates"
        );
    }

    #[test]
    fn unit_rejects_out_of_range_and_nan() {
        Unit::try_from(0.5).unwrap();
        Unit::try_from(-0.01).unwrap_err();
        Unit::try_from(1.01).unwrap_err();
        Unit::try_from(f32::NAN).unwrap_err();
    }

    #[test]
    fn tilt_clamps_and_drops_nan() {
        let t = Tilt::new(120.0, f64::NAN);
        assert_eq!((t.x(), t.y()), (90.0, 0.0));
    }

    #[test]
    fn upright_pen_points_straight_up() {
        for t in [Tilt::new(0.0, 0.0), Tilt::new(-0.0, -0.0)] {
            assert!((t.altitude() - FRAC_PI_2).abs() < 1e-12);
            assert!(t.azimuth().abs() < 1e-12);
        }
    }

    #[test]
    fn tilt_clamps_infinities() {
        let t = Tilt::new(f64::INFINITY, f64::NEG_INFINITY);
        assert_eq!((t.x(), t.y()), (90.0, -90.0));
    }

    #[test]
    fn unit_accepts_its_endpoints() {
        assert_eq!(Unit::try_from(0.0).unwrap(), Unit::ZERO);
        assert_eq!(Unit::try_from(1.0).unwrap(), Unit::ONE);
        assert!(Unit::try_from(-0.0).unwrap().get().is_sign_positive());
    }

    #[test]
    fn only_down_through_up_are_in_a_stroke() {
        let expected = [
            (Phase::Enter, false),
            (Phase::Hover, false),
            (Phase::Down, true),
            (Phase::Drag, true),
            (Phase::Up, true),
            (Phase::Leave, false),
        ];
        for (phase, in_stroke) in expected {
            assert_eq!(phase.in_stroke(), in_stroke, "{phase:?}");
        }
    }

    #[test]
    fn tilt_converts_like_pointer_events() {
        // W3C Pointer Events: azimuth from +x towards +y, altitude from the surface.
        let cases = [
            ((45.0, 0.0), (FRAC_PI_4, 0.0)),
            ((0.0, 45.0), (FRAC_PI_4, FRAC_PI_2)),
            ((-45.0, 0.0), (FRAC_PI_4, PI)),
            ((0.0, -45.0), (FRAC_PI_4, 3.0 * FRAC_PI_2)),
            ((90.0, 0.0), (0.0, 0.0)),
        ];
        for ((x, y), (altitude, azimuth)) in cases {
            let t = Tilt::new(x, y);
            assert!(
                (t.altitude() - altitude).abs() < 1e-9,
                "altitude for ({x}, {y})"
            );
            assert!(
                (t.azimuth() - azimuth).abs() < 1e-9,
                "azimuth for ({x}, {y})"
            );
        }
    }

    #[test]
    fn tilt_keeps_fractions() {
        assert!(Tilt::new(30.5, 0.0).altitude() < Tilt::new(30.0, 0.0).altitude());
    }

    #[test]
    fn buttons_set_and_clear() {
        let b = Buttons::default()
            .with(Buttons::PRIMARY, true)
            .with(Buttons::TERTIARY, true);
        assert!(b.contains(Buttons::PRIMARY));
        assert!(!b.contains(Buttons::SECONDARY));
        assert!(!b.with(Buttons::PRIMARY, false).contains(Buttons::PRIMARY));
    }
}
