//! `DVec3`: the stroke engine's world-space vector, in `f64`.
//!
//! Placement works in `f64` (rays, hits, spacing), so a long stroke's measured distances don't
//! drift and a ray from a far camera keeps its precision; a [`Dab`](crate::Dab) is rounded to
//! `f32` once, when it is made.

use core::ops::{Add, Mul, Neg, Sub};

/// A point or a direction in world space, `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DVec3 {
    /// The x component.
    pub x: f64,
    /// The y component.
    pub y: f64,
    /// The z component.
    pub z: f64,
}

impl DVec3 {
    /// World X, unit length.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// World Y, unit length.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// World Z, unit length.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// The vector `(x, y, z)`.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The dot product.
    #[must_use]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The cross product, `self × o`.
    #[must_use]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// The length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// The distance to `o`.
    #[must_use]
    pub fn distance(self, o: Self) -> f64 {
        (o - self).length()
    }

    /// The vector scaled to unit length, or `None` for a vector too short to have a direction or
    /// not finite.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let length = self.length();
        (length.is_finite() && length > f64::MIN_POSITIVE).then(|| self * length.recip())
    }

    /// Whether every component is finite.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// The point a fraction `t` of the way from `self` to `to`.
    #[must_use]
    pub fn lerp(self, to: Self, t: f64) -> Self {
        self + (to - self) * t
    }

    /// The vector of `geom`'s `f32` vector, exactly.
    #[must_use]
    pub fn from_f32(v: geom::Vec3) -> Self {
        Self::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
    }

    /// The vector rounded to `geom`'s `f32` vector, as a [`Dab`](crate::Dab) holds it.
    #[must_use]
    pub fn to_f32(self) -> geom::Vec3 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a dab is f32: placement rounds its result once, here"
        )]
        geom::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// The components as an array, `[x, y, z]`, rounded to `f32` as a [`Dab`](crate::Dab)
    /// holds them.
    #[must_use]
    pub fn to_f32_array(self) -> [f32; 3] {
        let v = self.to_f32();
        [v.x, v.y, v.z]
    }
}

impl Add for DVec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for DVec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for DVec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Neg for DVec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl From<geom::Vec3> for DVec3 {
    fn from(v: geom::Vec3) -> Self {
        Self::from_f32(v)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_products() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, DVec3::new(5.0, 7.0, 9.0), "add");
        assert_eq!(b - a, DVec3::new(3.0, 3.0, 3.0), "sub");
        assert_eq!(a * 2.0, DVec3::new(2.0, 4.0, 6.0), "scale");
        assert_eq!(-a, DVec3::new(-1.0, -2.0, -3.0), "neg");
        assert_eq!(a.dot(b), 32.0, "dot");
        assert_eq!(DVec3::X.cross(DVec3::Y), DVec3::Z, "x × y = z");
        assert_eq!(a.lerp(b, 0.5), DVec3::new(2.5, 3.5, 4.5), "half way");
        assert_eq!(
            DVec3::ZERO.distance(DVec3::new(3.0, 4.0, 0.0)),
            5.0,
            "3-4-5"
        );
    }

    #[test]
    fn normalized_refuses_zero_and_nan() {
        assert_eq!(DVec3::ZERO.normalized(), None, "zero");
        assert_eq!(DVec3::new(f64::NAN, 0.0, 0.0).normalized(), None, "NaN");
        let n = DVec3::new(0.0, 3.0, 4.0).normalized().unwrap();
        assert!((n.length() - 1.0).abs() < 1e-15, "unit length");
    }

    #[test]
    fn converts_to_and_from_f32() {
        let v = geom::Vec3::new(1.5, -2.25, 0.125);
        assert_eq!(DVec3::from(v).to_f32(), v, "exact both ways");
        assert_eq!(
            DVec3::new(1.0, 2.0, 3.0).to_f32_array(),
            [1.0, 2.0, 3.0],
            "as an array"
        );
    }
}
