//! Falloffs: how a brush's effect fades from its centre to its rim.

/// How strongly a brush acts at a normalised distance from its centre.
///
/// `t` is the distance divided by the brush radius, so the brush covers `t` in `[0, 1]`. A
/// falloff gives `1` at the centre and `0` at the rim and beyond, and is `0` for `t ≥ 1` (and
/// for a NaN `t`), so a brush may evaluate it for any vertex without checking first.
pub trait Falloff {
    /// The weight at normalised distance `t`: `1` at `t = 0`, `0` for `t ≥ 1`.
    fn weight(&self, t: f32) -> f32;
}

/// The smoothstep falloff, `1 − (3t² − 2t³)`.
///
/// It falls monotonically from `1` at the centre to `0` at the rim, with zero slope at both ends,
/// so it is C¹ across the rim (where it meets the untouched surface) and has no crease at the
/// centre. Inside the centre (`t ≤ 0`) it is `1`; at and past the rim it is `0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Smoothstep;

impl Falloff for Smoothstep {
    #[inline]
    fn weight(&self, t: f32) -> f32 {
        if t <= 0.0 {
            1.0
        } else if t < 1.0 {
            1.0 - t * t * (3.0 - 2.0 * t)
        } else {
            // `t ≥ 1`, or NaN.
            0.0
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// Samples across `[0, 1]` the tests check the falloff at.
    const STEPS: u16 = 1000;

    /// The `k`-th of [`STEPS`] evenly spaced points of `[0, 1]`.
    fn at(k: u16) -> f32 {
        f32::from(k) / f32::from(STEPS)
    }

    #[test]
    fn smoothstep_is_one_at_the_centre_and_zero_from_the_rim() {
        let f = Smoothstep;
        assert_eq!(f.weight(0.0), 1.0, "full strength at the centre");
        assert_eq!(
            f.weight(-0.5),
            1.0,
            "a negative distance counts as the centre"
        );
        assert_eq!(f.weight(1.0), 0.0, "nothing at the rim");
        assert_eq!(f.weight(1.5), 0.0, "nothing past the rim");
        assert_eq!(f.weight(f32::INFINITY), 0.0, "nothing at infinity");
        assert_eq!(f.weight(f32::NAN), 0.0, "nothing for a NaN distance");
        assert!(
            (f.weight(0.5) - 0.5).abs() < 1e-7,
            "half strength halfway, by symmetry"
        );
    }

    #[test]
    fn smoothstep_falls_monotonically() {
        let f = Smoothstep;
        for k in 0..STEPS {
            let (a, b) = (f.weight(at(k)), f.weight(at(k + 1)));
            assert!(
                b < a,
                "falls from t = {} to t = {}: {a} → {b}",
                at(k),
                at(k + 1)
            );
        }
    }

    #[test]
    fn smoothstep_is_c1_at_both_ends() {
        let f = Smoothstep;
        let h = 1e-3_f32;
        // Continuous: the value just inside each end is close to the value at it.
        assert!((f.weight(h) - 1.0).abs() < 1e-5, "continuous at the centre");
        assert!(f.weight(1.0 - h).abs() < 1e-5, "continuous at the rim");
        // Flat: one-sided finite differences vanish like h (the slope is 6t(t − 1)).
        let centre_slope = (f.weight(h) - f.weight(0.0)) / h;
        let rim_slope = (f.weight(1.0) - f.weight(1.0 - h)) / h;
        assert!(
            centre_slope.abs() < 1e-2,
            "flat at the centre: {centre_slope}"
        );
        assert!(rim_slope.abs() < 1e-2, "flat at the rim: {rim_slope}");
        // Smooth inside: the slope matches the analytic derivative everywhere.
        for k in 1..STEPS - 1 {
            let t = at(k);
            let numeric = (f.weight(t + h) - f.weight(t - h)) / (2.0 * h);
            let analytic = 6.0 * t * (t - 1.0);
            assert!(
                (numeric - analytic).abs() < 1e-2,
                "slope at t = {t}: {numeric} vs {analytic}"
            );
        }
    }
}
