//! `Vec3` and the scalar tests evaluation needs, with `geom`'s arithmetic.
//!
//! Every operation is the same `f32` operations in the same order and grouping as
//! `geom::Vec3`'s, so a GPU that rounds like the CPU gives the same bits.

use core::ops::{Add, Mul, Sub};

use spirv_std::num_traits::Float;

/// The bits of an `f32`'s exponent field: all set for `±∞` and NaN, and only for them.
const EXPONENT_MASK: u32 = 0x7F80_0000;

/// The smallest positive normal `f32`, `2^-126`: lengths at or below it don't normalize.
const MIN_POSITIVE: f32 = 1.175_494_4e-38;

/// Whether `x` is finite, tested on its exponent bits: no infinite constant for `naga` to
/// reject, and nothing a compiler that assumes finite floats can fold away.
pub(crate) const fn is_finite(x: f32) -> bool {
    x.to_bits() & EXPONENT_MASK != EXPONENT_MASK
}

/// Whether a vector of length `len` can be normalized: `geom::math::normalizable`.
pub(crate) const fn normalizable(len: f32) -> bool {
    is_finite(len) && len > MIN_POSITIVE
}

/// A 3D point or vector: `geom::Vec3`'s arithmetic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Vec3 {
    /// X component.
    pub(crate) x: f32,
    /// Y component.
    pub(crate) y: f32,
    /// Z component.
    pub(crate) z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub(crate) const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// A vector from its components.
    pub(crate) const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector from a buffer element.
    pub(crate) const fn from_array([x, y, z]: [f32; 3]) -> Self {
        Self::new(x, y, z)
    }

    /// The vector as a buffer element.
    pub(crate) const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Every component `value`.
    pub(crate) const fn splat(value: f32) -> Self {
        Self::new(value, value, value)
    }

    /// Dot product: `x·x' + y·y' + z·z'`, summed left to right.
    pub(crate) fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product, each component `p·q − r·s`.
    pub(crate) fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Whether every component is finite.
    pub(crate) const fn is_finite(self) -> bool {
        is_finite(self.x) && is_finite(self.y) && is_finite(self.z)
    }

    /// The length: the square root of `self · self`.
    pub(crate) fn length(self) -> f32 {
        Float::sqrt(self.dot(self))
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
