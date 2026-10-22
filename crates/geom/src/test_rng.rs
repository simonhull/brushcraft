//! Deterministic random numbers for tests.

#![expect(clippy::missing_panics_doc, reason = "test helpers panic to fail")]

/// splitmix64.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, 16 bits of resolution.
    pub(crate) fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.next_u64() >> 48).unwrap()) / 65536.0
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn signed(&mut self) -> f32 {
        self.unit().mul_add(2.0, -1.0)
    }

    /// Uniform in `0..n`; `n > 0`.
    pub(crate) fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next_u64() % u64::from(n)).unwrap()
    }

    pub(crate) fn vec3(&mut self, amplitude: f32) -> crate::Vec3 {
        crate::Vec3::new(
            self.signed() * amplitude,
            self.signed() * amplitude,
            self.signed() * amplitude,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_and_in_range() {
        let (mut a, mut b) = (Rng::new(1), Rng::new(1));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64(), "same seed, same stream");
        }
        let mut rng = Rng::new(2);
        for _ in 0..1000 {
            assert!((0.0..1.0).contains(&rng.unit()), "unit in [0, 1)");
            assert!((-1.0..1.0).contains(&rng.signed()), "signed in [-1, 1)");
            assert!(rng.below(7) < 7, "below stays below");
            assert!(rng.vec3(2.0).x.abs() <= 2.0, "vec3 within amplitude");
        }
    }
}
