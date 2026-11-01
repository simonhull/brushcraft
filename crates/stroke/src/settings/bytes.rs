//! The settings' stable byte form: [`StrokeSettings::to_bytes`] and
//! [`StrokeSettings::from_bytes`].

use super::{
    Axis, DecodeError, Hardness, MirrorPlanes, Radial, Spacing, Strength, StrengthMode,
    StrokeSettings, Symmetry,
};
use crate::{
    ACTION_SLOTS, ActionSlot, Curve, CurveError, Dynamics, DynamicsError, Entry, FullScale, Input,
    MAX_CURVE_POINTS, MAX_ENTRIES, Param, PathSettings, Radius, SizeMode,
};

/// The byte form's current version.
///
/// Version 1 held one kind of symmetry at a time; no stroke record was ever written with it, so
/// this build doesn't read it. Version 2 had no strength or hardness: it is read with the
/// defaults ([`Strength::DEFAULT`], [`Hardness::DEFAULT`]). Version 3 had no strength mode: it is
/// read in model units ([`StrengthMode::Model`]).
pub(super) const VERSION: u8 = 4;

/// The version before the brush's strength and hardness, still read.
const VERSION_2: u8 = 2;

/// The version before the strength's mode, still read.
const VERSION_3: u8 = 3;

/// The bits of the symmetry byte: which kinds are on. The rest are refused.
mod symmetry_bit {
    /// Mirror planes follow: [`Symmetry::mirror`](super::Symmetry::mirror).
    pub(super) const MIRROR: u8 = 1;
    /// Radial copies follow: [`Symmetry::radial`](super::Symmetry::radial).
    pub(super) const RADIAL: u8 = 1 << 1;
    /// Reserved for topological symmetry; refused until it exists.
    pub(super) const TOPOLOGICAL: u8 = 1 << 2;
}

/// The tag of each [`Input`] in the byte form.
const fn input_tag(input: Input) -> u8 {
    match input {
        Input::Pressure => 0,
        Input::TiltAmount => 1,
        Input::TiltDirection => 2,
        Input::Twist => 3,
        Input::Speed(_) => 4,
        Input::Distance(_) => 5,
        Input::Random => 6,
    }
}

/// The tag of each [`Param`] in the byte form: the action's slots are `5 + slot`.
const fn param_tag(param: Param) -> u8 {
    match param {
        Param::Size => 0,
        Param::Strength => 1,
        Param::Spacing => 2,
        Param::Angle => 3,
        Param::Hardness => 4,
        Param::Action(slot) => 5 + slot.index(),
    }
}

/// The tag of each [`Axis`] in the byte form.
const fn axis_tag(axis: Axis) -> u8 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

impl StrokeSettings {
    /// The settings' stable byte form, for a stroke's record.
    ///
    /// The layout (version 4), all numbers little-endian:
    /// - the version, `4`, as a `u8`;
    /// - the path: the minimum cutoff, the cutoff slope, the derivative cutoff and the lazy string,
    ///   each an `f64`;
    /// - the spacing as an `f32`; the size mode as a `u8` (`0` screen, `1` world); the radius as an
    ///   `f32`;
    /// - the symmetry: a `u8` of the kinds that are on (bit 0 mirror, bit 1 radial, bit 2 reserved
    ///   for topological), then for mirror the plane bits as a `u8`, then for radial the axis (`0`
    ///   X, `1` Y, `2` Z) and the copy count as `u8`s;
    /// - the action's parameters, four `f32`s;
    /// - the brush's strength and hardness, each an `f32` (new in version 3; version 2 has neither,
    ///   and is read with the defaults);
    /// - the strength's mode as a `u8`, `0` model units, `1` radii (new in version 4; version 3 is
    ///   read in model units);
    /// - the dynamics: the entry count as a `u8`, then each entry: the input's tag (`0` pressure,
    ///   `1` tilt amount, `2` tilt direction, `3` twist, `4` speed, `5` distance, `6` random)
    ///   followed for speed and distance by the full scale as an `f32`; the parameter's tag (`0`
    ///   size, `1` strength, `2` spacing, `3` angle, `4` hardness, `5 + slot` an action parameter);
    ///   the curve's point count as a `u8` and its points as `(input, output)` `f32` pairs; then
    ///   `min` and `max` as `f32`s.
    ///
    /// [`StrokeSettings::from_bytes`] gives back equal settings.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Writer(Vec::with_capacity(64 + self.dynamics.entries().len() * 80));
        out.u8(VERSION);
        let path = &self.path;
        for value in [
            path.min_cutoff(),
            path.cutoff_slope(),
            path.derivative_cutoff(),
            path.lazy(),
        ] {
            out.f64(value);
        }
        out.f32(self.spacing.get());
        out.u8(match self.size_mode {
            SizeMode::Screen => 0,
            SizeMode::World => 1,
        });
        out.f32(self.radius.get());
        let Symmetry { mirror, radial } = self.symmetry;
        let kinds =
            mirror.map_or(0, |_| symmetry_bit::MIRROR) | radial.map_or(0, |_| symmetry_bit::RADIAL);
        out.u8(kinds);
        if let Some(planes) = mirror {
            out.u8(planes.bits());
        }
        if let Some(radial) = radial {
            out.u8(axis_tag(radial.axis()));
            out.u8(radial.copies());
        }
        for value in self.action_params {
            out.f32(value);
        }
        out.f32(self.strength.get());
        out.f32(self.hardness.get());
        out.u8(match self.strength_mode {
            StrengthMode::Model => 0,
            StrengthMode::Radius => 1,
        });
        let entries = self.dynamics.entries();
        // `Dynamics::new` holds the count to MAX_ENTRIES, which fits a byte.
        out.u8(u8::try_from(entries.len()).unwrap_or(u8::MAX));
        for entry in entries {
            out.u8(input_tag(entry.input()));
            if let Input::Speed(full) | Input::Distance(full) = entry.input() {
                out.f32(full.get());
            }
            out.u8(param_tag(entry.param()));
            let points = entry.curve().points();
            // `Curve::new` holds the count to MAX_CURVE_POINTS, which fits a byte.
            out.u8(u8::try_from(points.len()).unwrap_or(u8::MAX));
            for point in points {
                out.f32(point.input.get());
                out.f32(point.output.get());
            }
            out.f32(entry.min());
            out.f32(entry.max());
        }
        out.0
    }

    /// Decodes [`StrokeSettings::to_bytes`]'s form, validating every value as the constructors
    /// do.
    ///
    /// Counts are checked against their limits before anything is read for them, so hostile
    /// bytes can't make this allocate more than a small table.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::Version`] for a version this build doesn't read (it reads 2 to 4);
    /// - [`DecodeError::Truncated`] or [`DecodeError::Trailing`] if the bytes are shorter or longer
    ///   than the settings;
    /// - [`DecodeError::Tag`] for an unknown tag;
    /// - [`DecodeError::Invalid`] for values the constructors refuse.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut reader = Reader(bytes);
        let version = reader.u8()?;
        if ![VERSION_2, VERSION_3, VERSION].contains(&version) {
            return Err(DecodeError::Version(version));
        }
        let path = PathSettings::new(reader.f64()?, reader.f64()?, reader.f64()?, reader.f64()?)?;
        let spacing = Spacing::new(reader.f32()?)?;
        let size_mode = match reader.u8()? {
            0 => SizeMode::Screen,
            1 => SizeMode::World,
            tag => {
                return Err(DecodeError::Tag {
                    what: "size mode",
                    tag,
                });
            }
        };
        let radius = Radius::new(reader.f32()?)?;
        let symmetry = read_symmetry(&mut reader)?;
        let mut action_params = [0.0; ACTION_SLOTS];
        for value in &mut action_params {
            *value = reader.f32()?;
        }
        let (strength, hardness) = if version == VERSION_2 {
            (Strength::DEFAULT, Hardness::DEFAULT)
        } else {
            (Strength::new(reader.f32()?)?, Hardness::new(reader.f32()?)?)
        };
        let strength_mode = if version == VERSION {
            match reader.u8()? {
                0 => StrengthMode::Model,
                1 => StrengthMode::Radius,
                tag => {
                    return Err(DecodeError::Tag {
                        what: "strength mode",
                        tag,
                    });
                }
            }
        } else {
            StrengthMode::Model
        };
        let count = usize::from(reader.u8()?);
        if count > MAX_ENTRIES {
            return Err(DynamicsError::TooManyEntries(count).into());
        }
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(read_entry(&mut reader)?);
        }
        let dynamics = Dynamics::new(entries)?;
        if !reader.0.is_empty() {
            return Err(DecodeError::Trailing(reader.0.len()));
        }
        Ok(Self::new(
            path,
            spacing,
            size_mode,
            radius,
            dynamics,
            symmetry,
            action_params,
        )?
        .with_strength(strength)
        .with_strength_mode(strength_mode)
        .with_hardness(hardness))
    }
}

/// Reads an [`Axis`] tag.
///
/// # Errors
///
/// [`DecodeError::Truncated`] at the end of the bytes; [`DecodeError::Tag`] for an unknown tag.
fn read_axis(reader: &mut Reader<'_>) -> Result<Axis, DecodeError> {
    match reader.u8()? {
        0 => Ok(Axis::X),
        1 => Ok(Axis::Y),
        2 => Ok(Axis::Z),
        tag => Err(DecodeError::Tag { what: "axis", tag }),
    }
}

/// Reads the symmetry: its kinds byte, then each kind's values.
///
/// # Errors
///
/// [`DecodeError::Truncated`] at the end of the bytes; [`DecodeError::Tag`] for a kinds byte with
/// the topological bit (which this build can't apply) or an unknown one, or an unknown axis;
/// [`DecodeError::Invalid`] for planes or a copy count the constructors refuse.
fn read_symmetry(reader: &mut Reader<'_>) -> Result<Symmetry, DecodeError> {
    let kinds = reader.u8()?;
    let known = symmetry_bit::MIRROR | symmetry_bit::RADIAL;
    if kinds & !known != 0 || kinds & symmetry_bit::TOPOLOGICAL != 0 {
        return Err(DecodeError::Tag {
            what: "symmetry",
            tag: kinds,
        });
    }
    let mirror = if kinds & symmetry_bit::MIRROR == 0 {
        None
    } else {
        Some(MirrorPlanes::from_bits(reader.u8()?)?)
    };
    let radial = if kinds & symmetry_bit::RADIAL == 0 {
        None
    } else {
        let axis = read_axis(reader)?;
        Some(Radial::new(axis, reader.u8()?)?)
    };
    Ok(Symmetry { mirror, radial })
}

/// Reads one dynamics entry.
///
/// # Errors
///
/// Any [`DecodeError`] but the version and trailing bytes.
fn read_entry(reader: &mut Reader<'_>) -> Result<Entry, DecodeError> {
    let input = match reader.u8()? {
        0 => Input::Pressure,
        1 => Input::TiltAmount,
        2 => Input::TiltDirection,
        3 => Input::Twist,
        4 => Input::Speed(FullScale::new(reader.f32()?)?),
        5 => Input::Distance(FullScale::new(reader.f32()?)?),
        6 => Input::Random,
        tag => return Err(DecodeError::Tag { what: "input", tag }),
    };
    let param = match reader.u8()? {
        0 => Param::Size,
        1 => Param::Strength,
        2 => Param::Spacing,
        3 => Param::Angle,
        4 => Param::Hardness,
        tag @ 5.. => Param::Action(
            ActionSlot::new(tag - 5)
                .ok()
                .ok_or(DecodeError::Tag { what: "param", tag })?,
        ),
    };
    let count = usize::from(reader.u8()?);
    if count > MAX_CURVE_POINTS {
        return Err(CurveError::PointCount(count).into());
    }
    let mut points = [(0.0, 0.0); MAX_CURVE_POINTS];
    let used = points.get_mut(..count).unwrap_or_default();
    for point in &mut *used {
        *point = (reader.f32()?, reader.f32()?);
    }
    let curve = Curve::new(used)?;
    Ok(Entry::new(
        input,
        param,
        curve,
        reader.f32()?,
        reader.f32()?,
    )?)
}

/// Appends little-endian numbers to a byte buffer.
#[derive(Debug)]
struct Writer(Vec<u8>);

impl Writer {
    /// Appends a byte.
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    /// Appends an `f32`'s bits.
    fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    /// Appends an `f64`'s bits.
    fn f64(&mut self, value: f64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }
}

/// Reads little-endian numbers off the front of a byte slice.
#[derive(Debug)]
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    /// Takes the next `N` bytes.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] if fewer than `N` remain.
    fn take<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let (head, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or(DecodeError::Truncated)?;
        self.0 = rest;
        Ok(*head)
    }

    /// Reads a byte.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] at the end of the bytes.
    fn u8(&mut self) -> Result<u8, DecodeError> {
        self.take::<1>().map(|[b]| b)
    }

    /// Reads an `f32`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] if fewer than four bytes remain.
    fn f32(&mut self) -> Result<f32, DecodeError> {
        self.take().map(f32::from_le_bytes)
    }

    /// Reads an `f64`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] if fewer than eight bytes remain.
    fn f64(&mut self) -> Result<f64, DecodeError> {
        self.take().map(f64::from_le_bytes)
    }
}
