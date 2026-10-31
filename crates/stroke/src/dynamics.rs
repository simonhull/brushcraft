//! Dynamics: how the pen's readings shape each dab, through a table of curves.
//!
//! A brush's [`Dynamics`] is a list of [`Entry`]s, each mapping one [`Input`] (pressure, tilt,
//! speed, …) through a [`Curve`] onto one [`Param`] of the dab (size, strength, …), scaled into
//! the entry's `min..=max`. Entries on one parameter multiply. Curves are evaluated on the CPU
//! as each dab is placed, so a dab carries final values and nothing downstream sees a curve.
//!
//! **Inputs at rest.** Every input is read as a value in `0..=1` whose `0` is the input at rest:
//! no pressure, an upright pen, no twist, standing still, the stroke's start. An input the pen
//! doesn't send (twist on most pens, tilt on a mouse) reads `0`, so its entry holds its curve's
//! rest value ([`Curve::rest`]) scaled into `min..=max`.

use std::f32::consts::{FRAC_PI_2, TAU};

use crate::{Curve, CurveError, Sample, Tilt, Twist, Unit};

/// How many parameters an action may define for dynamics to drive ([`Param::Action`]).
pub const ACTION_SLOTS: usize = 4;
/// The most entries a dynamics table holds: far more than a brush needs, few enough that
/// evaluating them per dab costs nothing and a hostile file can't make the table large.
pub const MAX_ENTRIES: usize = 32;

/// A dynamics table or entry that can't be built.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DynamicsError {
    /// An entry's range is not allowed for its parameter (see [`Param::allows`]).
    #[error("{param:?} can't range over {min}..={max}")]
    Range {
        /// The parameter.
        param: Param,
        /// The range's start.
        min: f32,
        /// The range's end.
        max: f32,
    },
    /// More than [`MAX_ENTRIES`] entries.
    #[error("a dynamics table has at most {MAX_ENTRIES} entries, not {0}")]
    TooManyEntries(usize),
    /// A full scale that isn't positive and finite.
    #[error("an input's full scale must be positive and finite, not {0}")]
    FullScale(f32),
    /// An action slot outside `0..ACTION_SLOTS`.
    #[error("action slot {0} is outside 0..{ACTION_SLOTS}")]
    ActionSlot(u8),
    /// An invalid curve.
    #[error(transparent)]
    Curve(#[from] CurveError),
}

/// The reading at which an unbounded input ([`Input::Speed`], [`Input::Distance`]) reaches the end
/// of its curve: positive and finite. Readings beyond it hold the curve's last value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullScale(f32);

impl FullScale {
    /// The full scale `value`.
    ///
    /// # Errors
    ///
    /// [`DynamicsError::FullScale`] if `value` is zero, negative, infinite or NaN.
    pub fn new(value: f32) -> Result<Self, DynamicsError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(DynamicsError::FullScale(value))
        }
    }

    /// The value, positive and finite.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// What a dynamics entry reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Input {
    /// Tip pressure, as the pen reports it.
    Pressure,
    /// How far the pen leans: `0` upright, `1` lying flat (π/2 radians).
    TiltAmount,
    /// Which way the pen leans on the screen: `0..1` for `0..2π` radians from right towards down.
    TiltDirection,
    /// The barrel rotation: `0..1` for `0..2π` radians.
    Twist,
    /// How fast the brush moves along the surface, in radii (of the stroke's world radius) per
    /// second, read as a fraction of the full scale. Measuring in radii keeps a brush's feel the
    /// same at any size.
    Speed(FullScale),
    /// How far along the surface the stroke has come, in radii, read as a fraction of the full
    /// scale: for tapering a stroke's start.
    Distance(FullScale),
    /// A value in `0..1` per dab, from the stroke's seed ([`StrokeRandom`]): deterministic, so a
    /// replayed stroke places the same dabs.
    Random,
}

/// One of the parameters an action defines for dynamics to drive, `0..ACTION_SLOTS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionSlot(u8);

impl ActionSlot {
    /// The slot `index`.
    ///
    /// # Errors
    ///
    /// [`DynamicsError::ActionSlot`] if `index` is not below [`ACTION_SLOTS`].
    pub fn new(index: u8) -> Result<Self, DynamicsError> {
        if usize::from(index) < ACTION_SLOTS {
            Ok(Self(index))
        } else {
            Err(DynamicsError::ActionSlot(index))
        }
    }

    /// The slot's index, below [`ACTION_SLOTS`].
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// What a dynamics entry drives.
///
/// Every parameter but the angle is a factor on the brush's own value (its radius, its spacing,
/// the action's strength, hardness and parameters), so with no entry it stays `1`. The angle is
/// absolute, in radians, and stays `0` with no entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    /// A factor on the dab's radius.
    Size,
    /// A factor on the action's strength.
    Strength,
    /// A factor on the spacing between dabs.
    Spacing,
    /// The alpha's rotation, in radians.
    Angle,
    /// A factor on the falloff's hardness.
    Hardness,
    /// A factor on one of the action's own parameters.
    Action(ActionSlot),
}

impl Param {
    /// Whether an entry on this parameter may range over `min..=max`.
    ///
    /// Both ends must be finite. Size, strength and hardness factors can't be negative; a spacing
    /// factor must be positive, so dabs never pile up on one spot. The angle and the action's
    /// parameters take any finite value.
    #[must_use]
    pub fn allows(self, min: f32, max: f32) -> bool {
        let finite = min.is_finite() && max.is_finite();
        finite
            && match self {
                Self::Size | Self::Strength | Self::Hardness => min >= 0.0 && max >= 0.0,
                Self::Spacing => min > 0.0 && max > 0.0,
                Self::Angle | Self::Action(_) => true,
            }
    }
}

/// One row of a dynamics table: `param` gets `min + (max − min) · curve(input)`.
///
/// `min` may exceed `max`, to invert the response without a falling curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    /// What the entry reads.
    input: Input,
    /// What it drives.
    param: Param,
    /// The response to the input, in `0..=1`.
    curve: Curve,
    /// The value at the curve's `0`.
    min: f32,
    /// The value at the curve's `1`.
    max: f32,
}

impl Entry {
    /// An entry reading `input` through `curve` and driving `param` over `min..=max`.
    ///
    /// # Errors
    ///
    /// [`DynamicsError::Range`] if `param` doesn't allow the range ([`Param::allows`]).
    pub fn new(
        input: Input,
        param: Param,
        curve: Curve,
        min: f32,
        max: f32,
    ) -> Result<Self, DynamicsError> {
        if !param.allows(min, max) {
            return Err(DynamicsError::Range { param, min, max });
        }
        Ok(Self {
            input,
            param,
            curve,
            min: min + 0.0,
            max: max + 0.0,
        })
    }

    /// What the entry reads.
    #[must_use]
    pub const fn input(&self) -> Input {
        self.input
    }

    /// What it drives.
    #[must_use]
    pub const fn param(&self) -> Param {
        self.param
    }

    /// The response curve.
    #[must_use]
    pub const fn curve(&self) -> &Curve {
        &self.curve
    }

    /// The value at the curve's `0`.
    #[must_use]
    pub const fn min(&self) -> f32 {
        self.min
    }

    /// The value at the curve's `1`.
    #[must_use]
    pub const fn max(&self) -> f32 {
        self.max
    }

    /// The entry's value for these readings: finite, between `min` and `max`.
    #[must_use]
    pub fn value(&self, inputs: &Inputs) -> f32 {
        let t = self.curve.eval(inputs.read(self.input).get());
        // This form stays finite for any finite ends, where `min + (max − min)·t` could overflow.
        self.min * (1.0 - t) + self.max * t
    }
}

/// The readings dynamics evaluates a dab with.
///
/// The pen's readings come from the [`Sample`] the dab was placed at; the stroke engine adds
/// what it measures (speed and distance along the surface) and the dab's random value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Inputs {
    /// Tip pressure.
    pub pressure: Unit,
    /// The pen's tilt, or `None` if it doesn't report one.
    pub tilt: Option<Tilt>,
    /// The barrel rotation, or `None` if the pen doesn't report one.
    pub twist: Option<Twist>,
    /// Speed along the surface, in radii per second. Negative and NaN read as `0`.
    pub speed: f32,
    /// Distance along the surface since the stroke began, in radii. Negative and NaN read as `0`.
    pub distance: f32,
    /// The dab's random value ([`StrokeRandom::at`]).
    pub random: Unit,
}

impl Inputs {
    /// The pen's readings from `sample`; speed, distance and the random value at `0`.
    #[must_use]
    pub const fn from_sample(sample: &Sample) -> Self {
        Self {
            pressure: sample.pressure(),
            tilt: sample.tilt(),
            twist: sample.twist(),
            speed: 0.0,
            distance: 0.0,
            random: Unit::ZERO,
        }
    }

    /// `input`'s reading, in `0..=1` with `0` at rest. A missing tilt or twist reads `0`.
    #[must_use]
    pub fn read(&self, input: Input) -> Unit {
        match input {
            Input::Pressure => self.pressure,
            Input::TiltAmount => {
                Unit::saturating(self.tilt.map_or(0.0, |t| t.amount() / FRAC_PI_2))
            }
            Input::TiltDirection => {
                Unit::saturating(self.tilt.map_or(0.0, |t| t.direction() / TAU))
            }
            Input::Twist => Unit::saturating(self.twist.map_or(0.0, |t| t.get() / TAU)),
            Input::Speed(full) => Unit::saturating(self.speed / full.get()),
            Input::Distance(full) => Unit::saturating(self.distance / full.get()),
            Input::Random => self.random,
        }
    }
}

/// Every parameter's value for one dab, from [`Dynamics::evaluate`].
///
/// Each is the product of its entries' values, or its neutral value with no entry: `1` for the
/// factors, `0` radians for the angle ([`Values::NEUTRAL`]). Every value is finite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Values {
    /// A factor on the radius.
    pub size: f32,
    /// A factor on the strength.
    pub strength: f32,
    /// A factor on the spacing; positive.
    pub spacing: f32,
    /// The alpha's rotation, in radians.
    pub angle: f32,
    /// A factor on the hardness.
    pub hardness: f32,
    /// Factors on the action's parameters.
    pub action: [f32; ACTION_SLOTS],
}

impl Values {
    /// What a table with no entries gives: every factor `1`, the angle `0`.
    pub const NEUTRAL: Self = Self {
        size: 1.0,
        strength: 1.0,
        spacing: 1.0,
        angle: 0.0,
        hardness: 1.0,
        action: [1.0; ACTION_SLOTS],
    };
}

impl Default for Values {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// A brush's dynamics table: up to [`MAX_ENTRIES`] entries.
///
/// The default maps pressure to size and pressure to strength, each linearly over `0..=1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dynamics {
    /// The entries, in the order they were given.
    entries: Vec<Entry>,
}

impl Dynamics {
    /// A table of `entries`.
    ///
    /// # Errors
    ///
    /// [`DynamicsError::TooManyEntries`] for more than [`MAX_ENTRIES`] entries.
    pub fn new(entries: Vec<Entry>) -> Result<Self, DynamicsError> {
        if entries.len() > MAX_ENTRIES {
            return Err(DynamicsError::TooManyEntries(entries.len()));
        }
        Ok(Self { entries })
    }

    /// A table with no entries: every dab gets [`Values::NEUTRAL`].
    #[must_use]
    pub const fn none() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The entries, in order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Every parameter's value for a dab with these readings.
    ///
    /// Entries on one parameter multiply. A product that would overflow is held at the largest
    /// finite `f32`, so every value stays finite.
    #[must_use]
    pub fn evaluate(&self, inputs: &Inputs) -> Values {
        let finite = |v: f32| v.clamp(-f32::MAX, f32::MAX);
        // The factors start at 1, so their first entry sets them exactly. The angle's neutral
        // value is 0, which would swallow every product, so it starts unset.
        let mut values = Values::NEUTRAL;
        let mut angle: Option<f32> = None;
        for entry in &self.entries {
            let value = entry.value(inputs);
            let factor = match entry.param {
                Param::Size => &mut values.size,
                Param::Strength => &mut values.strength,
                Param::Spacing => &mut values.spacing,
                Param::Hardness => &mut values.hardness,
                Param::Angle => {
                    angle = Some(finite(angle.map_or(value, |a| a * value)));
                    continue;
                }
                Param::Action(ActionSlot(index)) => {
                    // `ActionSlot::new` keeps the index below ACTION_SLOTS.
                    let Some(factor) = values.action.get_mut(usize::from(index)) else {
                        continue;
                    };
                    factor
                }
            };
            *factor = finite(*factor * value);
        }
        values.angle = angle.unwrap_or(Values::NEUTRAL.angle);
        values
    }
}

impl Default for Dynamics {
    fn default() -> Self {
        let pressure = |param| Entry {
            input: Input::Pressure,
            param,
            curve: Curve::LINEAR,
            min: 0.0,
            max: 1.0,
        };
        Self {
            entries: vec![pressure(Param::Size), pressure(Param::Strength)],
        }
    }
}

/// The per-stroke source of [`Input::Random`]: one value per dab, from the stroke's seed and the
/// dab's sequence number.
///
/// Each value depends only on the seed and the sequence number, not on how many were drawn
/// before, so replaying a stroke with the same seed gives the same values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrokeRandom {
    /// The stroke's seed.
    seed: u64,
}

impl StrokeRandom {
    /// The random values of a stroke seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The value for the dab with sequence number `seq`, in `0..1`.
    #[must_use]
    pub fn at(self, seq: u32) -> Unit {
        // splitmix64's finaliser over the seed and the sequence number: well mixed for nearby
        // seeds and numbers.
        let mut z = self
            .seed
            .wrapping_add(u64::from(seq).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 24 bits, which f32 holds exactly: a value in 0..1.
        #[expect(
            clippy::cast_precision_loss,
            reason = "a 24-bit integer converts to f32 exactly"
        )]
        let top = (z >> 40) as f32;
        Unit::saturating(top / 16_777_216.0)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::ScreenPoint;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn table(entries: Vec<Entry>) -> Dynamics {
        Dynamics::new(entries).unwrap()
    }

    fn linear(input: Input, param: Param, min: f32, max: f32) -> Entry {
        Entry::new(input, param, Curve::LINEAR, min, max).unwrap()
    }

    fn unit(value: f32) -> Unit {
        Unit::new(value).unwrap()
    }

    #[test]
    fn each_input_reads_on_its_scale() {
        let full = FullScale::new(10.0).unwrap();
        let inputs = Inputs {
            pressure: unit(0.4),
            tilt: Some(Tilt::new(FRAC_PI_2 / 2.0, TAU / 4.0).unwrap()),
            twist: Some(Twist::new(TAU * 0.75).unwrap()),
            speed: 5.0,
            distance: 25.0,
            random: unit(0.9),
        };
        let cases = [
            (Input::Pressure, 0.4),
            (Input::TiltAmount, 0.5),
            (Input::TiltDirection, 0.25),
            (Input::Twist, 0.75),
            (Input::Speed(full), 0.5),
            (Input::Distance(full), 1.0),
            (Input::Random, 0.9),
        ];
        for (input, expected) in cases {
            let got = inputs.read(input).get();
            assert!(close(got, expected), "{input:?}: {got}, not {expected}");
            let values = table(vec![linear(input, Param::Size, 0.0, 2.0)]).evaluate(&inputs);
            assert!(
                close(values.size, expected * 2.0),
                "{input:?} drives size: {}",
                values.size
            );
        }
    }

    #[test]
    fn each_param_is_driven_and_the_rest_stay_neutral() {
        let inputs = Inputs {
            pressure: unit(0.5),
            ..Inputs::default()
        };
        let params = [
            Param::Size,
            Param::Strength,
            Param::Spacing,
            Param::Angle,
            Param::Hardness,
            Param::Action(ActionSlot::new(0).unwrap()),
            Param::Action(ActionSlot::new(3).unwrap()),
        ];
        for param in params {
            let values = table(vec![linear(Input::Pressure, param, 1.0, 3.0)]).evaluate(&inputs);
            let mut expected = Values::NEUTRAL;
            match param {
                Param::Size => expected.size = 2.0,
                Param::Strength => expected.strength = 2.0,
                Param::Spacing => expected.spacing = 2.0,
                Param::Angle => expected.angle = 2.0,
                Param::Hardness => expected.hardness = 2.0,
                Param::Action(slot) => expected.action[usize::from(slot.index())] = 2.0,
            }
            assert_eq!(values, expected, "{param:?}");
        }
    }

    #[test]
    fn entries_on_one_param_multiply() {
        let inputs = Inputs {
            pressure: unit(0.5),
            random: unit(0.25),
            ..Inputs::default()
        };
        let values = table(vec![
            linear(Input::Pressure, Param::Size, 0.0, 1.0),
            linear(Input::Random, Param::Size, 0.0, 4.0),
            linear(Input::Pressure, Param::Angle, 0.0, 2.0),
            linear(Input::Random, Param::Angle, 0.0, 4.0),
        ])
        .evaluate(&inputs);
        assert!(close(values.size, 0.5), "0.5 × 1.0: {}", values.size);
        assert!(close(values.angle, 1.0), "1.0 × 1.0: {}", values.angle);
        assert!(close(values.strength, 1.0), "untouched");
    }

    #[test]
    fn a_huge_product_stays_finite() {
        let inputs = Inputs {
            pressure: Unit::ONE,
            ..Inputs::default()
        };
        let entries = vec![linear(Input::Pressure, Param::Size, 0.0, f32::MAX); 3];
        let values = table(entries).evaluate(&inputs);
        assert!(values.size.is_finite(), "{}", values.size);
    }

    #[test]
    fn missing_twist_and_tilt_hold_the_rest_value() {
        let curve = Curve::new(&[(0.0, 0.3), (1.0, 1.0)]).unwrap();
        let dynamics = table(vec![
            Entry::new(Input::Twist, Param::Angle, curve, 0.0, 1.0).unwrap(),
            Entry::new(Input::TiltAmount, Param::Hardness, curve, 0.0, 1.0).unwrap(),
        ]);
        let at = ScreenPoint::new(0.0, 0.0).unwrap();
        let sample = Sample::new(at, 0.0, Unit::ONE).unwrap();
        let values = dynamics.evaluate(&Inputs::from_sample(&sample));
        assert!(close(values.angle, curve.rest()), "twist at rest");
        assert!(close(values.hardness, curve.rest()), "tilt at rest");
    }

    #[test]
    fn the_default_maps_pressure_to_size_and_strength() {
        let inputs = Inputs {
            pressure: unit(0.6),
            ..Inputs::default()
        };
        let values = Dynamics::default().evaluate(&inputs);
        assert!(close(values.size, 0.6), "size");
        assert!(close(values.strength, 0.6), "strength");
        assert!(close(values.spacing, 1.0), "spacing untouched");
        assert_eq!(
            Dynamics::none().evaluate(&inputs),
            Values::NEUTRAL,
            "no entries"
        );
    }

    #[test]
    fn invalid_entries_and_tables_are_refused() {
        let refused = |param, min, max| Entry::new(Input::Pressure, param, Curve::LINEAR, min, max);
        assert!(refused(Param::Size, -1.0, 1.0).is_err(), "negative size");
        assert!(refused(Param::Spacing, 0.0, 1.0).is_err(), "zero spacing");
        assert!(refused(Param::Angle, 0.0, f32::NAN).is_err(), "NaN angle");
        assert!(refused(Param::Angle, -3.0, 3.0).is_ok(), "signed angle");
        assert!(ActionSlot::new(4).is_err(), "slot 4");
        assert!(FullScale::new(0.0).is_err(), "zero full scale");
        let too_many = vec![linear(Input::Pressure, Param::Size, 0.0, 1.0); MAX_ENTRIES + 1];
        assert_eq!(
            Dynamics::new(too_many),
            Err(DynamicsError::TooManyEntries(MAX_ENTRIES + 1)),
            "too many"
        );
    }

    #[test]
    fn random_is_deterministic_per_seed() {
        let a = StrokeRandom::new(7);
        let b = StrokeRandom::new(7);
        let first: Vec<Unit> = (0..64).map(|seq| a.at(seq)).collect();
        let again: Vec<Unit> = (0..64).map(|seq| b.at(seq)).collect();
        assert_eq!(first, again, "same seed, same values");
        let other: Vec<Unit> = (0..64).map(|seq| StrokeRandom::new(8).at(seq)).collect();
        assert_ne!(first, other, "another seed, other values");
        let mean = first.iter().map(|u| u.get()).sum::<f32>() / 64.0;
        assert!((0.3..0.7).contains(&mean), "roughly uniform: mean {mean}");
        assert!(first.iter().all(|u| u.get() < 1.0), "below 1");
    }
}
