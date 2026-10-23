//! The little vector and scalar math the kernels need, on plain arrays.
//!
//! `core` has no square root on stable Rust (`f32::sqrt` lives in `std`, which a GPU target
//! doesn't have), so [`sqrt`] is our own: Newton steps on the reciprocal square root, which take
//! only multiplications and so vectorise on the CPU and compile anywhere.

/// The squared distance between `a` and `b`.
#[inline]
pub(crate) fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    let (dx, dy, dz) = (ax - bx, ay - by, az - bz);
    dx * dx + dy * dy + dz * dz
}

/// `2^24`, which lifts a subnormal input into the normal range for [`sqrt`].
const SUBNORMAL_LIFT: f32 = 16_777_216.0;

/// `2^-12`: the square root of `1 / SUBNORMAL_LIFT`, which undoes the lift on the result.
const SUBNORMAL_UNDO: f32 = 1.0 / 4096.0;

/// The square root of `x`, to within a few units in the last place.
///
/// `0` for zero and negative inputs (the kernels only take roots of squared distances), `x`
/// itself for `+∞` and NaN.
///
/// It starts from the classic bit-level estimate of `1/√x` (halve the exponent by shifting the
/// bits, subtract from a magic constant: within about 3.5 %), refines it with three Newton
/// steps `y ← y (1.5 − ½ x y²)`, each of which roughly squares the relative error, and returns
/// `x · y`. A subnormal `x` would defeat the estimate's exponent trick, so it is scaled into the
/// normal range first and the result scaled back.
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return x;
    }
    if x < f32::MIN_POSITIVE {
        return rsqrt_newton(x * SUBNORMAL_LIFT) * SUBNORMAL_UNDO;
    }
    rsqrt_newton(x)
}

/// `√x` for a positive, finite, normal `x`, as `x · (1/√x)` (see [`sqrt`]).
#[inline]
fn rsqrt_newton(x: f32) -> f32 {
    let half = 0.5 * x;
    let mut y = f32::from_bits(0x5F37_5A86 - (x.to_bits() >> 1));
    y *= 1.5 - half * y * y;
    y *= 1.5 - half * y * y;
    y *= 1.5 - half * y * y;
    x * y
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    extern crate std;

    use super::*;

    /// How far `ours` is from `exact`, in units of the last place of `exact`.
    fn ulps(ours: f32, exact: f32) -> u32 {
        ours.to_bits().abs_diff(exact.to_bits())
    }

    #[test]
    fn sqrt_matches_std_within_a_few_ulps() {
        // Every exponent of the normal and subnormal range, at several mantissas each.
        for exponent in 0..255_u32 {
            for mantissa in [0, 1, 0x12_3456, 0x40_0000, 0x7F_FFFF] {
                let x = f32::from_bits((exponent << 23) | mantissa);
                if x == 0.0 {
                    continue;
                }
                let (ours, exact) = (sqrt(x), f32::sqrt(x));
                assert!(
                    ulps(ours, exact) <= 4,
                    "√{x:e}: {ours:e} vs {exact:e} ({} ulps)",
                    ulps(ours, exact)
                );
            }
        }
    }

    #[test]
    fn sqrt_special_cases() {
        assert_eq!(sqrt(0.0), 0.0, "√0");
        assert_eq!(sqrt(-0.0), 0.0, "√−0");
        assert_eq!(sqrt(-4.0), 0.0, "negative inputs give 0");
        assert_eq!(sqrt(f32::INFINITY), f32::INFINITY, "√∞");
        assert!(sqrt(f32::NAN).is_nan(), "√NaN");
        assert!(ulps(sqrt(4.0), 2.0) <= 1, "√4 is within an ulp of 2");
    }

    #[test]
    fn distance_squared_sums_the_squares() {
        assert_eq!(
            distance_squared([1.0, 2.0, 3.0], [4.0, 6.0, 3.0]),
            25.0,
            "3² + 4² + 0²"
        );
        assert_eq!(
            distance_squared([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            0.0,
            "same point"
        );
    }
}
