//! `Vec3`: the small amount of vector math geometry needs, in `f32`.

use core::ops::{Add, Mul, Neg, Sub};

/// A 3D point or vector.
///
/// Plain data (`#[repr(C)]`, three `f32`s, 12 bytes): pool slots and GPU buffers hold it as is.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// All components NaN: marks a slot with no value.
    pub(crate) const NAN: Self = Self::new(f32::NAN, f32::NAN, f32::NAN);
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// A vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// This vector at unit length, or `None` if it is zero, too small to normalise, or not finite.
    #[must_use]
    pub fn normalized(self) -> Option<Self> {
        let len = self.length();
        (len.is_finite() && len > f32::MIN_POSITIVE).then(|| self * len.recip())
    }

    /// Whether every component is finite.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// The components' bit patterns, for exact comparisons.
    #[must_use]
    pub const fn to_bits(self) -> [u32; 3] {
        [self.x.to_bits(), self.y.to_bits(), self.z.to_bits()]
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

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn cross_follows_the_right_hand_rule() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0), "x × y = z");
        assert_eq!(y.cross(x), Vec3::new(0.0, 0.0, -1.0), "y × x = −z");
    }

    #[test]
    fn normalized_rejects_zero_and_nan() {
        assert_eq!(Vec3::ZERO.normalized(), None, "zero has no direction");
        assert_eq!(Vec3::NAN.normalized(), None, "NaN has no direction");
        let n = Vec3::new(3.0, 0.0, 4.0).normalized().unwrap();
        assert!((n.length() - 1.0).abs() < 1e-6, "unit length");
    }

    #[test]
    fn is_plain_data() {
        assert_eq!(size_of::<Vec3>(), 12, "three f32s, no padding");
        assert_eq!(align_of::<Vec3>(), 4, "f32 alignment");
        let v = Vec3::new(1.0, -2.0, 3.5);
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(
            bytes,
            [1.0_f32, -2.0, 3.5]
                .iter()
                .flat_map(|c| c.to_ne_bytes())
                .collect::<Vec<_>>(),
            "x, y, z in order"
        );
        assert_eq!(
            <Vec3 as bytemuck::Zeroable>::zeroed(),
            Vec3::ZERO,
            "zeroed is the zero vector"
        );
    }

    #[test]
    fn arithmetic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0), "add");
        assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0), "sub");
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0), "scale");
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0), "neg");
        assert!((a.dot(b) - 32.0).abs() < 1e-6, "dot");
    }
}
