//! Randomness: a seeded generator for the harness's choices, and the value noise `generate`
//! sculpts the sphere with. Our own, a few lines each: nothing here needs a crate.

use geom::Vec3;

/// A seeded random generator (splitmix64): the same seed gives the same run.
#[derive(Debug, Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub(crate) const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// Uniform in `[0, 1)`, with 24 bits of resolution.
    pub(crate) fn unit(&mut self) -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "24 bits are exact in an f32")]
        let bits = (self.next_u64() >> 40) as f32;
        bits / 16_777_216.0
    }

    /// Uniform in `[low, high)`.
    pub(crate) fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// A direction uniform over the unit sphere: a point uniform in the cube, kept if it lies
    /// in the unit ball (and isn't too near its centre to normalise), scaled onto the sphere.
    pub(crate) fn direction(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let length = p.length();
            if (1e-3..=1.0).contains(&length)
                && let Some(d) = p.normalized()
            {
                return d;
            }
        }
    }
}

/// splitmix64's finaliser: 64 well-mixed bits from `z`.
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Value noise in 3D: a random value in `[-1, 1]` at every point of the integer lattice,
/// blended between the eight lattice points around a point with smoothstep weights.
///
/// It is smooth (C¹) and band-limited to about one lattice cell, so scaling the input by a
/// frequency `f` gives features about `1 / f` across. Each octave has its own lattice values,
/// so the octaves of an fBm sum are independent.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValueNoise {
    /// Picks the lattice values.
    seed: u64,
}

impl ValueNoise {
    /// The noise of `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The noise of octave `octave` at `p`, in `[-1, 1]`.
    pub(crate) fn at(self, p: Vec3, octave: u32) -> f32 {
        let (x0, fx) = split(p.x);
        let (y0, fy) = split(p.y);
        let (z0, fz) = split(p.z);
        let (wx, wy, wz) = (smooth(fx), smooth(fy), smooth(fz));
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let corner = |dx: i32, dy: i32, dz: i32| {
            self.lattice(
                x0.wrapping_add(dx),
                y0.wrapping_add(dy),
                z0.wrapping_add(dz),
                octave,
            )
        };
        let along_x = |dy, dz| lerp(corner(0, dy, dz), corner(1, dy, dz), wx);
        let along_y = |dz| lerp(along_x(0, dz), along_x(1, dz), wy);
        lerp(along_y(0), along_y(1), wz)
    }

    /// The value at lattice point `(x, y, z)` of octave `octave`, in `[-1, 1]`.
    fn lattice(self, x: i32, y: i32, z: i32, octave: u32) -> f32 {
        let key = [
            x.cast_unsigned(),
            y.cast_unsigned(),
            z.cast_unsigned(),
            octave,
        ]
        .into_iter()
        .fold(self.seed, |h, k| mix(h ^ u64::from(k)));
        #[expect(clippy::cast_precision_loss, reason = "24 bits are exact in an f32")]
        let bits = (key >> 40) as f32;
        bits / 8_388_608.0 - 1.0
    }
}

/// `x` split into its lattice cell (the integer below it) and its offset in the cell.
fn split(x: f32) -> (i32, f32) {
    let floor = x.floor();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`as` saturates, and the harness's inputs (unit-sphere points times at most a \
                  few thousand) are far inside i32"
    )]
    let cell = floor as i32;
    (cell, x - floor)
}

/// The smoothstep weight `3t² − 2t³`, for blending across a cell without a crease.
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn the_generator_is_seeded_and_uniformish() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        assert_eq!(a.next_u64(), b.next_u64(), "same seed, same bits");
        let mean = (0..10_000).map(|_| a.unit()).sum::<f32>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean of unit() {mean}");
        for _ in 0..100 {
            let d = a.direction();
            assert!((d.length() - 1.0).abs() < 1e-5, "a unit direction: {d:?}");
        }
    }

    #[test]
    fn the_noise_is_bounded_continuous_and_lattice_valued() {
        let noise = ValueNoise::new(3);
        let mut rng = Rng::new(4);
        for _ in 0..1000 {
            let p = Vec3::new(
                rng.range(-50.0, 50.0),
                rng.range(-50.0, 50.0),
                rng.range(-50.0, 50.0),
            );
            let v = noise.at(p, 2);
            assert!((-1.0..=1.0).contains(&v), "bounded: {v}");
            let near = noise.at(p + Vec3::new(1e-4, 0.0, 0.0), 2);
            assert!((v - near).abs() < 1e-2, "continuous: {v} vs {near}");
        }
        let at_lattice = noise.at(Vec3::new(3.0, -2.0, 5.0), 0);
        assert!(
            (at_lattice - noise.lattice(3, -2, 5, 0)).abs() < 1e-6,
            "a lattice point takes its own value"
        );
        assert!(
            (noise.at(Vec3::new(0.3, 0.6, 0.9), 0) - noise.at(Vec3::new(0.3, 0.6, 0.9), 1)).abs()
                > 0.0,
            "octaves differ"
        );
    }
}
