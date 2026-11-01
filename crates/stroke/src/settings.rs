//! `StrokeSettings`: everything that shapes a stroke, as plain data with a stable byte form.
//!
//! The byte form ([`StrokeSettings::to_bytes`]) is what a stroke's record stores with its
//! samples. It starts with a version byte; a later version adds a decoder, it never changes
//! what an earlier version's bytes mean.

mod bytes;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

use crate::{
    ACTION_SLOTS, CurveError, Dynamics, DynamicsError, PathError, PathSettings, Radius,
    RadiusError, SizeMode,
};

/// A stroke setting out of range.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum SettingsError {
    /// The path's settings.
    #[error(transparent)]
    Path(#[from] PathError),
    /// The spacing (see [`Spacing::new`]).
    #[error("the spacing must be in {min}..={max} radii, not {0}", min = Spacing::MIN, max = Spacing::MAX)]
    Spacing(f32),
    /// The radius.
    #[error(transparent)]
    Radius(#[from] RadiusError),
    /// The dynamics table.
    #[error(transparent)]
    Dynamics(#[from] DynamicsError),
    /// The symmetry.
    #[error(transparent)]
    Symmetry(#[from] SymmetryError),
    /// The brush's strength is NaN or infinite (see [`Strength::new`]).
    #[error("the strength must be finite, not {0}")]
    Strength(f32),
    /// The brush's hardness is outside `0..=1` or NaN (see [`Hardness::new`]).
    #[error("the hardness must be in 0..=1, not {0}")]
    Hardness(f32),
    /// An action parameter is NaN or infinite.
    #[error("action parameter {index} is not finite: {value}")]
    ActionParam {
        /// Which parameter.
        index: usize,
        /// Its value.
        value: f32,
    },
}

impl From<CurveError> for SettingsError {
    fn from(error: CurveError) -> Self {
        Self::Dynamics(DynamicsError::Curve(error))
    }
}

/// Bytes [`StrokeSettings::from_bytes`] can't decode.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// The version byte names no version this build reads.
    #[error("stroke settings version {0} is not one this build reads")]
    Version(u8),
    /// The bytes end before the settings do.
    #[error("the stroke settings are cut short")]
    Truncated,
    /// Bytes follow the settings.
    #[error("{0} bytes follow the stroke settings")]
    Trailing(usize),
    /// A tag byte names no variant.
    #[error("{what} tag {tag} is not known")]
    Tag {
        /// What the tag picks.
        what: &'static str,
        /// The tag.
        tag: u8,
    },
    /// The values decode but aren't valid settings.
    #[error(transparent)]
    Invalid(#[from] SettingsError),
}

impl From<PathError> for DecodeError {
    fn from(error: PathError) -> Self {
        Self::Invalid(error.into())
    }
}

impl From<RadiusError> for DecodeError {
    fn from(error: RadiusError) -> Self {
        Self::Invalid(error.into())
    }
}

impl From<DynamicsError> for DecodeError {
    fn from(error: DynamicsError) -> Self {
        Self::Invalid(error.into())
    }
}

impl From<CurveError> for DecodeError {
    fn from(error: CurveError) -> Self {
        Self::Invalid(error.into())
    }
}

impl From<SymmetryError> for DecodeError {
    fn from(error: SymmetryError) -> Self {
        Self::Invalid(error.into())
    }
}

/// The gap between dabs, as a fraction of the brush's radius, measured along the surface.
///
/// In `Spacing::MIN..=Spacing::MAX`: the floor keeps a stroke from placing thousands of dabs per
/// radius. Dynamics can scale it per dab.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Spacing(f32);

impl Spacing {
    /// The default: 0.15 of the radius.
    pub const DEFAULT: Self = Self(0.15);
    /// The widest spacing, ten radii apart.
    pub const MAX: f32 = 10.0;
    /// The closest spacing, a hundredth of the radius.
    pub const MIN: f32 = 0.01;

    /// A spacing of `fraction` of the radius.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Spacing`] if `fraction` is outside `MIN..=MAX` or NaN.
    pub fn new(fraction: f32) -> Result<Self, SettingsError> {
        if (Self::MIN..=Self::MAX).contains(&fraction) {
            Ok(Self(fraction))
        } else {
            Err(SettingsError::Spacing(fraction))
        }
    }

    /// The fraction of the radius.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for Spacing {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The brush's own strength, in the unit its [`StrengthMode`] says.
///
/// It is the action's strength for a dab whose dynamics leave it at its full value (a factor of
/// 1): for draw, how far a vertex at the centre moves. Finite; negative works the other way
/// (draw pushes in).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Strength(f32);

impl Strength {
    /// The default, `0.001` model units: a placeholder until the built-in brushes set their
    /// own.
    pub const DEFAULT: Self = Self(0.001);

    /// A strength of `value`, in model units or radii as the [`StrengthMode`] says.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Strength`] if `value` is NaN or infinite.
    pub fn new(value: f32) -> Result<Self, SettingsError> {
        if value.is_finite() {
            Ok(Self(value + 0.0))
        } else {
            Err(SettingsError::Strength(value))
        }
    }

    /// The strength, in model units or radii as the [`StrengthMode`] says.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for Strength {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What a brush's [`Strength`] is measured in.
///
/// Either way a [`Dab`](crate::Dab)'s strength is final, in model units: a strength in radii is
/// turned into model units once, where the stroke freezes its world radius, so it holds for the
/// whole stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StrengthMode {
    /// Model units, whatever the brush's size: the default.
    #[default]
    Model,
    /// A fraction of the stroke's frozen world radius ([`Stroke::radius`](crate::Stroke::radius)):
    /// a stroke raises the surface by the same share of its width at any zoom, and a brush made
    /// larger is as much stronger.
    Radius,
}

/// The brush's own falloff hardness, in `0..=1`: `0` the action's falloff as it is, `1` a disc
/// of full strength (see `brush::Hardened`). A dab's hardness is this times its dynamics'
/// factor, clamped to `0..=1`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Hardness(f32);

impl Hardness {
    /// The default: `0`, the falloff as it is.
    pub const DEFAULT: Self = Self(0.0);

    /// A hardness of `value`.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Hardness`] if `value` is outside `0..=1` or NaN.
    pub fn new(value: f32) -> Result<Self, SettingsError> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value + 0.0))
        } else {
            Err(SettingsError::Hardness(value))
        }
    }

    /// The hardness, in `0..=1`.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

impl Default for Hardness {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Symmetry that can't be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SymmetryError {
    /// A set of mirror planes with no plane, or bits naming no plane.
    #[error("mirror plane bits {0:#04b} name no plane or an unknown one")]
    MirrorPlanes(u8),
    /// A radial count outside `2..=Radial::MAX_COPIES`.
    #[error("radial symmetry makes 2 to {max} copies, not {0}", max = Radial::MAX_COPIES)]
    RadialCopies(u8),
}

/// A world axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// World X.
    X,
    /// World Y.
    Y,
    /// World Z.
    Z,
}

/// Which world mirror planes are active: at least one of the planes through the origin
/// perpendicular to X, Y and Z.
///
/// Built from [`MirrorPlanes::X`], [`MirrorPlanes::Y`] and [`MirrorPlanes::Z`], combined with
/// `|`, so it never names no plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MirrorPlanes(u8);

impl MirrorPlanes {
    /// The plane `x = 0`: mirrors X.
    pub const X: Self = Self(1);
    /// The plane `y = 0`: mirrors Y.
    pub const Y: Self = Self(1 << 1);
    /// The plane `z = 0`: mirrors Z.
    pub const Z: Self = Self(1 << 2);

    /// The planes from their bits (X bit 0, Y bit 1, Z bit 2), as [`MirrorPlanes::bits`] gives
    /// them.
    ///
    /// # Errors
    ///
    /// [`SymmetryError::MirrorPlanes`] if no plane bit is set, or a higher bit is.
    pub const fn from_bits(bits: u8) -> Result<Self, SymmetryError> {
        if bits == 0 || bits & !0b111 != 0 {
            Err(SymmetryError::MirrorPlanes(bits))
        } else {
            Ok(Self(bits))
        }
    }

    /// The planes' bits: X bit 0, Y bit 1, Z bit 2.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether the plane mirroring `axis` is active.
    #[must_use]
    pub const fn mirrors(self, axis: Axis) -> bool {
        let bit = match axis {
            Axis::X => Self::X.0,
            Axis::Y => Self::Y.0,
            Axis::Z => Self::Z.0,
        };
        self.0 & bit != 0
    }
}

impl core::ops::BitOr for MirrorPlanes {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Radial symmetry: copies evenly spaced around a world axis through the origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Radial {
    /// The axis the copies turn about.
    axis: Axis,
    /// How many copies, the original included, in `2..=MAX_COPIES`.
    copies: u8,
}

impl Radial {
    /// The most copies radial symmetry makes, the original included.
    pub const MAX_COPIES: u8 = 64;

    /// `copies` copies (the original included) evenly spaced around `axis`.
    ///
    /// # Errors
    ///
    /// [`SymmetryError::RadialCopies`] if `copies` is outside `2..=MAX_COPIES`.
    pub const fn new(axis: Axis, copies: u8) -> Result<Self, SymmetryError> {
        if copies < 2 || copies > Self::MAX_COPIES {
            Err(SymmetryError::RadialCopies(copies))
        } else {
            Ok(Self { axis, copies })
        }
    }

    /// The axis the copies turn about.
    #[must_use]
    pub const fn axis(self) -> Axis {
        self.axis
    }

    /// How many copies, the original included.
    #[must_use]
    pub const fn copies(self) -> u8 {
        self.copies
    }
}

/// How each placed dab is copied before any action runs: through mirror planes, radial copies,
/// or both.
///
/// With both, the copies are the product: every radial turn of every mirror image, so mirroring
/// in X with six radial copies makes twelve. [`Symmetry::copies`] counts them. The default has
/// neither: each dab is placed once.
///
/// **Copy indices** ([`Dab::copy`](crate::Dab::copy)): the mirror images are numbered `0` (the
/// dab itself) then each non-empty subset of the planes in the order of their bits (X, Y, XY, Z,
/// XZ, YZ, XYZ, those active), and the turns `0..n`; copy `image · n + turn` is mirror image
/// `image` turned `turn` steps, so an index always names the same map. A copy that coincides with
/// an earlier one of the same dab (within 1 % of its radius: a dab on a mirror plane or the
/// radial axis) is dropped, leaving a gap in the indices.
///
/// Each copy's frame goes through its map (normal and tangent alike), and a copy mirrored through
/// an odd number of planes has its alpha angle negated, since a reflection reverses rotation.
///
/// Topological symmetry (through a correspondence between base faces) will join these as a third
/// kind; its bit in the byte form is reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Symmetry {
    /// The mirror planes, or `None` for no mirroring.
    pub mirror: Option<MirrorPlanes>,
    /// The radial copies, or `None` for none.
    pub radial: Option<Radial>,
}

impl Symmetry {
    /// No symmetry: each dab is placed once.
    pub const NONE: Self = Self {
        mirror: None,
        radial: None,
    };

    /// Mirroring through `planes`, with no radial copies.
    #[must_use]
    pub const fn mirror(planes: MirrorPlanes) -> Self {
        Self {
            mirror: Some(planes),
            radial: None,
        }
    }

    /// Radial copies, with no mirroring.
    #[must_use]
    pub const fn radial(radial: Radial) -> Self {
        Self {
            mirror: None,
            radial: Some(radial),
        }
    }

    /// How many copies each dab makes, the original included: `2^planes` mirror images times
    /// the radial count. `1` with no symmetry; at most `8 · Radial::MAX_COPIES`.
    #[must_use]
    pub fn copies(self) -> u32 {
        let mirrors = match self.mirror {
            Some(planes) => 1 << planes.bits().count_ones(),
            None => 1,
        };
        let turns = match self.radial {
            Some(radial) => u32::from(radial.copies()),
            None => 1,
        };
        mirrors * turns
    }
}

/// Everything that shapes a stroke: the path's smoothing, the spacing, the size and how it is
/// measured, the dynamics, the symmetry, and the action's strength, hardness and parameters.
///
/// Plain data, validated as it is built: every part's type holds only valid values, and
/// [`StrokeSettings::new`] checks the action's parameters, so settings that exist are usable and a
/// stroke never fails on them mid-way. [`StrokeSettings::to_bytes`] and
/// [`StrokeSettings::from_bytes`] give it a stable byte form.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeSettings {
    /// How the screen path is smoothed.
    path: PathSettings,
    /// The gap between dabs, in radii.
    spacing: Spacing,
    /// Whether the radius is in screen pixels or world units.
    size_mode: SizeMode,
    /// The brush's radius, in the size mode's unit.
    radius: Radius,
    /// How the pen's readings shape each dab.
    dynamics: Dynamics,
    /// How each dab is copied.
    symmetry: Symmetry,
    /// The brush's own strength, before dynamics scales it.
    strength: Strength,
    /// What the strength is measured in.
    strength_mode: StrengthMode,
    /// The brush's own hardness, before dynamics scales it.
    hardness: Hardness,
    /// The action's own parameters, before dynamics scales them; all finite.
    action_params: [f32; ACTION_SLOTS],
}

impl StrokeSettings {
    /// Settings from their parts, with the default [`Strength`] and [`Hardness`]
    /// ([`StrokeSettings::with_strength`] and [`StrokeSettings::with_hardness`] set them).
    ///
    /// # Errors
    ///
    /// [`SettingsError::ActionParam`] if an action parameter is NaN or infinite.
    pub fn new(
        path: PathSettings,
        spacing: Spacing,
        size_mode: SizeMode,
        radius: Radius,
        dynamics: Dynamics,
        symmetry: Symmetry,
        action_params: [f32; ACTION_SLOTS],
    ) -> Result<Self, SettingsError> {
        if let Some((index, &value)) = action_params
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(SettingsError::ActionParam { index, value });
        }
        Ok(Self {
            path,
            spacing,
            size_mode,
            radius,
            dynamics,
            symmetry,
            strength: Strength::DEFAULT,
            strength_mode: StrengthMode::Model,
            hardness: Hardness::DEFAULT,
            action_params: action_params.map(|v| v + 0.0),
        })
    }

    /// These settings with the brush's own `strength`.
    #[must_use]
    pub fn with_strength(self, strength: Strength) -> Self {
        Self { strength, ..self }
    }

    /// These settings with the brush's strength measured in `mode`.
    #[must_use]
    pub fn with_strength_mode(self, mode: StrengthMode) -> Self {
        Self {
            strength_mode: mode,
            ..self
        }
    }

    /// These settings with the brush's own `hardness`.
    #[must_use]
    pub fn with_hardness(self, hardness: Hardness) -> Self {
        Self { hardness, ..self }
    }

    /// The brush's own strength, before dynamics scales it.
    #[must_use]
    pub const fn strength(&self) -> Strength {
        self.strength
    }

    /// What the brush's strength is measured in.
    #[must_use]
    pub const fn strength_mode(&self) -> StrengthMode {
        self.strength_mode
    }

    /// The brush's own hardness, before dynamics scales it.
    #[must_use]
    pub const fn hardness(&self) -> Hardness {
        self.hardness
    }

    /// How the screen path is smoothed.
    #[must_use]
    pub const fn path(&self) -> &PathSettings {
        &self.path
    }

    /// The gap between dabs, in radii.
    #[must_use]
    pub const fn spacing(&self) -> Spacing {
        self.spacing
    }

    /// Whether the radius is in screen pixels or world units.
    #[must_use]
    pub const fn size_mode(&self) -> SizeMode {
        self.size_mode
    }

    /// The brush's radius, in the size mode's unit.
    #[must_use]
    pub const fn radius(&self) -> Radius {
        self.radius
    }

    /// How the pen's readings shape each dab.
    #[must_use]
    pub const fn dynamics(&self) -> &Dynamics {
        &self.dynamics
    }

    /// How each dab is copied.
    #[must_use]
    pub const fn symmetry(&self) -> Symmetry {
        self.symmetry
    }

    /// The action's own parameters, before dynamics scales them.
    #[must_use]
    pub const fn action_params(&self) -> [f32; ACTION_SLOTS] {
        self.action_params
    }
}

impl Default for StrokeSettings {
    /// The default path, spacing 0.15, a 40-pixel screen radius, pressure to size and strength,
    /// no symmetry, the default strength (in model units) and hardness, and the action's
    /// parameters at zero.
    fn default() -> Self {
        Self {
            path: PathSettings::default(),
            spacing: Spacing::DEFAULT,
            size_mode: SizeMode::Screen,
            radius: Radius::DEFAULT,
            dynamics: Dynamics::default(),
            symmetry: Symmetry::NONE,
            strength: Strength::DEFAULT,
            strength_mode: StrengthMode::Model,
            hardness: Hardness::DEFAULT,
            action_params: [0.0; ACTION_SLOTS],
        }
    }
}
