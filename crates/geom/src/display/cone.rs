//! Normal cones: a direction and a half-angle holding every finest normal under a display tile.
//!
//! Level choice refines a tile by its cone, not by its own coarse triangles: a tile whose
//! triangles face away can still hold finest detail facing the camera (just past a silhouette),
//! and the cone says so. So a cone must **hold every finest normal under its tile as stored**:
//! the angle from each to the decoded axis is at most the decoded half-angle.
//!
//! # Building one
//!
//! One level below the finest, a tile's cone is taken from the finest normals under it
//! ([`NormalCone::covering`]); above, from its children's decoded cones
//! ([`NormalCone::holding`]), so it chains up the levels as `e` does. Either way, an axis is
//! chosen (the normals' sum, or the children's cones merged two at a time), quantized, and
//! decoded; the half-angle is then measured against the *decoded* axis, from the inputs
//! themselves (`angle(axis, n)` for a normal, `angle(axis, child's axis) + child's half-angle`
//! for a child, which holds everything the child holds by the triangle inequality on the
//! sphere), each angle raised by [`SLACK`] unless it is exactly 0, and rounded **up** to its
//! stored step. So whatever the axis's quantization error, the stored cone holds its inputs, and
//! the chosen axis only decides how tight it is. A half-angle that reaches 180° (or past) is
//! stored as the cone that covers everything.
//!
//! The chaining costs width: a parent's cone holds its children's whole cones, not just the
//! normals in them, so it is wider than the least cone of those normals (on the display tests'
//! smooth sphere, about 6° wider than the exact cone around the same axis, where four children
//! of 30.6° meet).
//!
//! # Stored
//!
//! In one `u32` ([`NormalCone::to_bits`]): the axis octahedral in the low [`AXIS_BITS`] × 2
//! (two signed fractions), the half-angle in the high [`HALF_BITS`] as a count of
//! `π / (2^HALF_BITS − 1)` steps, the largest count meaning 180°: everything. Every `u32` is a
//! cone. The split of the 32 bits is the measured one (see [`AXIS_BITS`]).

use core::f64::consts::PI;
use core::fmt;

use super::DisplayError;
use super::encode::{from_octahedral, levels, octahedral, quantize_unit};
use crate::Vec3;

/// Bits per octahedral component of the axis.
///
/// A cone grows, at each level, by up to the axis's quantization error (its half-angle is
/// measured from the decoded axis: under 1.3° at 8 bits, 0.33° at 10, 0.085° at 12) and by up
/// to one half-angle step (it is rounded up: 0.003° at 16 bits, 0.044° at 12, 0.71° at 8), and
/// the levels add these up. Of the splits of the 32 bits, 10 + 12 gives the tightest cones on
/// the display tests' fixtures (`cone_widths`): the smooth sphere's cones one level below the
/// finest are 30.59° (8 + 16 bits: 30.92°; 12 + 8: 31.06°), and above them 68.57° (68.88°,
/// 69.18°).
const AXIS_BITS: u8 = 10;

/// Bits of the half-angle's step count (see [`AXIS_BITS`]).
const HALF_BITS: u8 = 12;

// The cone fills one `u32` exactly, so every one of them is a cone.
const _: () = assert!(2 * AXIS_BITS + HALF_BITS == 32, "a cone is one u32");

/// The mask of one axis component's bits.
const AXIS_MASK: u32 = (1 << AXIS_BITS) - 1;

/// Where the half-angle's bits start.
const HALF_SHIFT: u32 = 2 * AXIS_BITS as u32;

/// The half-angle's largest step count: 180°, the cone that covers everything.
const HALF_MAX: u32 = (1 << HALF_BITS) - 1;

/// One half-angle step, in radians: `π / HALF_MAX`.
const HALF_STEP: f32 = core::f32::consts::PI / 4095.0;

// `HALF_STEP`'s literal is `HALF_MAX`.
const _: () = assert!(HALF_MAX == 4095, "HALF_STEP divides by HALF_MAX");

/// What each angle a half-angle is measured from is raised by, in radians.
///
/// Far more than the `f64` arithmetic measuring it can lose (an `atan2`, a few products and a
/// sum, each off by about 2⁻⁵² of π), and far less than a step. An angle of exactly 0 (a direction
/// the axis itself, the cross product exactly zero) is exact, and isn't raised, so a cone alone is
/// its own union and a level whose tiles each hold one child doesn't widen.
const SLACK: f64 = 1e-9;

/// A cone of directions: an axis and a half-angle, as a display tile stores the cone holding
/// every finest normal under it (see the module notes).
///
/// Every `u32` is one ([`NormalCone::from_bits`]); a half-angle of 180° holds every direction
/// ([`NormalCone::EVERYTHING`], also the [`Default`], what a tile whose normals aren't known is
/// given).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormalCone(u32);

/// A direction in `f64`, not necessarily unit.
type Wide = [f64; 3];

impl NormalCone {
    /// The cone that holds every direction: a half-angle of 180°, around `+z`.
    pub const EVERYTHING: Self = Self(HALF_MAX << HALF_SHIFT);

    /// The cone `bits` store, as [`NormalCone::to_bits`] gave them: every `u32` is one.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The cone as stored: the axis's two octahedral components in the low bits, the
    /// half-angle's step count in the high ones (see the module notes).
    #[must_use]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Whether the cone holds every direction (its half-angle is 180°).
    #[must_use]
    pub const fn covers_everything(self) -> bool {
        self.steps() == HALF_MAX
    }

    /// The decoded axis, a unit vector (`+z` for [`NormalCone::EVERYTHING`]).
    #[must_use]
    pub fn axis(self) -> Vec3 {
        let component = |shift: u32| {
            let raw = (self.0 >> shift) & AXIS_MASK;
            // Sign-extended from its `AXIS_BITS`: the top bit is the sign.
            let unused = 32 - u32::from(AXIS_BITS);
            let signed = (raw << unused).cast_signed() >> unused;
            let small = i16::try_from(signed).unwrap_or(0);
            (f32::from(small) / levels(AXIS_BITS)).clamp(-1.0, 1.0)
        };
        from_octahedral(component(0), component(u32::from(AXIS_BITS)))
    }

    /// The decoded half-angle, in radians: a whole number of steps of `π / 4095`, up to π.
    #[must_use]
    pub fn half_angle(self) -> f32 {
        if self.covers_everything() {
            return core::f32::consts::PI;
        }
        half_of(self.steps())
    }

    /// Whether `direction` lies within the cone: its angle to the decoded axis is at most the
    /// decoded half-angle, measured in `f64` (a zero vector, which has no direction, is held by
    /// every cone).
    #[must_use]
    pub fn contains(self, direction: Vec3) -> bool {
        self.covers_everything()
            || angle(wide(self.axis()), wide(direction)) <= f64::from(self.half_angle())
    }

    /// Whether the cone holds every direction `other` holds: the angle between the axes plus
    /// `other`'s half-angle is at most this one's.
    #[must_use]
    pub fn holds(self, other: Self) -> bool {
        if self.covers_everything() {
            return true;
        }
        !other.covers_everything()
            && angle(wide(self.axis()), wide(other.axis())) + f64::from(other.half_angle())
                <= f64::from(self.half_angle())
    }

    /// The cone holding every one of `normals`, as stored: around their sum (summed in order,
    /// so the same normals give the same bits), its half-angle the largest angle from the
    /// decoded axis to any of them, rounded up (see the module notes).
    ///
    /// A zero normal has no direction, so it widens nothing. Normals that cancel out (a zero
    /// sum) get the axis `+z`; normals spread over more than a hemisphere around the axis may
    /// need the cone that covers everything.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] for no normals; [`DisplayError::NotFinite`] for a normal that
    /// isn't finite.
    pub fn covering(normals: &[Vec3]) -> Result<Self, DisplayError> {
        if normals.is_empty() {
            return Err(DisplayError::Shape);
        }
        if !normals.iter().all(|n| n.is_finite()) {
            return Err(DisplayError::NotFinite);
        }
        let sum = normals.iter().fold([0.0; 3], |[x, y, z], &n| {
            let [nx, ny, nz] = wide(n);
            [x + nx, y + ny, z + nz]
        });
        Ok(Self::around(sum, |axis| {
            normals
                .iter()
                .map(|&n| reach(axis, wide(n)))
                .fold(0.0, f64::max)
        }))
    }

    /// The cone holding every one of `cones`, as stored: their union, as a parent tile's cone
    /// holds its children's.
    ///
    /// The axis is the cones merged two at a time (the least cone holding two cones, in `f64`),
    /// in the order of their bits, so the same cones give the same bits in any order; its
    /// half-angle is the largest `angle(axis, child's axis) + child's half-angle` from the
    /// decoded axis, rounded up (see the module notes). Any of them covering everything makes
    /// the union cover everything.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] for no cones.
    pub fn holding(cones: &[Self]) -> Result<Self, DisplayError> {
        if cones.iter().any(|c| c.covers_everything()) {
            return Ok(Self::EVERYTHING);
        }
        let mut sorted: Vec<Self> = cones.to_vec();
        sorted.sort_unstable_by_key(|c| c.0);
        let (first, rest) = sorted.split_first().ok_or(DisplayError::Shape)?;
        let start = (unit(wide(first.axis())), f64::from(first.half_angle()));
        let (axis, _) = rest.iter().fold(start, |cone, c| {
            merge(cone, (unit(wide(c.axis())), f64::from(c.half_angle())))
        });
        Ok(Self::around(axis, |decoded| {
            sorted
                .iter()
                .map(|c| reach(decoded, wide(c.axis())) + f64::from(c.half_angle()))
                .fold(0.0, f64::max)
        }))
    }

    /// The cone around `axis` (quantized) whose half-angle is `needs` of the decoded axis
    /// (measured with [`reach`]), rounded up to a step; the cone that covers everything if that
    /// reaches 180°.
    fn around(axis: Wide, needs: impl Fn(Wide) -> f64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "only the direction matters, and it is quantized far coarser than an f32"
        )]
        let narrow = Vec3::new(axis[0] as f32, axis[1] as f32, axis[2] as f32);
        let (u, v) = octahedral(narrow);
        let pack = |c: f32| u32::from(quantize_unit(c, AXIS_BITS).cast_unsigned()) & AXIS_MASK;
        let axis_bits = pack(u) | pack(v) << AXIS_BITS;
        let need = needs(wide(Self(axis_bits).axis()));
        steps_up(need).map_or(Self::EVERYTHING, |steps| {
            Self(axis_bits | steps << HALF_SHIFT)
        })
    }

    /// The half-angle's step count.
    const fn steps(self) -> u32 {
        self.0 >> HALF_SHIFT
    }
}

impl Default for NormalCone {
    /// [`NormalCone::EVERYTHING`]: the cone of a tile whose normals aren't known.
    fn default() -> Self {
        Self::EVERYTHING
    }
}

impl fmt::Debug for NormalCone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.covers_everything() {
            return f.write_str("NormalCone::EVERYTHING");
        }
        f.debug_struct("NormalCone")
            .field("axis", &self.axis())
            .field("half_degrees", &self.half_angle().to_degrees())
            .finish()
    }
}

/// The half-angle `steps` steps stand for, in radians, as decoded (an `f32` product, so it can
/// be computed the same way anywhere).
fn half_of(steps: u32) -> f32 {
    f32::from(u16::try_from(steps).unwrap_or(u16::MAX)) * HALF_STEP
}

/// The fewest steps whose decoded half-angle is at least `need` radians, or `None` if that
/// would be 180° or more (or `need` isn't a number): the cone that covers everything.
fn steps_up(need: f64) -> Option<u32> {
    if need.is_nan() || need >= PI {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a non-negative count under HALF_MAX + 1, rounded down"
    )]
    let below = (need.max(0.0) / f64::from(HALF_STEP)).floor() as u32;
    // From a step under the quotient, up to the first decoded half-angle (an `f32` product,
    // which can round either way from the `f64` one) at or past `need`: one or two steps.
    let mut steps = below.saturating_sub(1);
    while steps < HALF_MAX && f64::from(half_of(steps)) < need {
        steps += 1;
    }
    (steps < HALF_MAX).then_some(steps)
}

/// The least cone (axis, half-angle in radians) holding the cones `a` and `b`, whose axes are
/// unit.
///
/// One of them if it holds the other, else the cone spanning both, its axis on the great circle
/// through theirs. Opposite axes give a cone of 180° around `a`'s.
fn merge(a: (Wide, f64), b: (Wide, f64)) -> (Wide, f64) {
    let ((a_axis, a_half), (b_axis, b_half)) = (a, b);
    let between = angle(a_axis, b_axis);
    if between + b_half <= a_half {
        return a;
    }
    if between + a_half <= b_half {
        return b;
    }
    let half = (between + a_half + b_half) / 2.0;
    // `b`'s axis less its part along `a`'s: the way from `a`'s axis towards `b`'s.
    let along = dot(a_axis, b_axis);
    let towards = [
        b_axis[0] - a_axis[0] * along,
        b_axis[1] - a_axis[1] * along,
        b_axis[2] - a_axis[2] * along,
    ];
    let length = dot(towards, towards).sqrt();
    if length <= 0.0 {
        return (a_axis, PI);
    }
    let turn = half - a_half;
    let (sin, cos) = turn.sin_cos();
    let axis = [
        a_axis[0] * cos + towards[0] / length * sin,
        a_axis[1] * cos + towards[1] / length * sin,
        a_axis[2] * cos + towards[2] / length * sin,
    ];
    (axis, half)
}

/// [`angle`] raised by [`SLACK`], or exactly 0 where the angle is: what a half-angle must reach
/// to hold `b` around `a`.
fn reach(a: Wide, b: Wide) -> f64 {
    let between = angle(a, b);
    if between > 0.0 { between + SLACK } else { 0.0 }
}

/// `v` in `f64`, exactly.
fn wide(v: Vec3) -> Wide {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// The dot product of `a` and `b`.
fn dot(a: Wide, b: Wide) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `v` scaled to unit length (`v` itself if it is zero).
fn unit(v: Wide) -> Wide {
    let length = dot(v, v).sqrt();
    if length > 0.0 {
        [v[0] / length, v[1] / length, v[2] / length]
    } else {
        v
    }
}

/// The angle between `a` and `b` in radians, from `atan2(|a × b|, a · b)` (an `acos` of the dot
/// product loses small angles to rounding); 0 if either is zero. Neither needs to be unit.
fn angle(a: Wide, b: Wide) -> f64 {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    dot(cross, cross).sqrt().atan2(dot(a, b))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
