//! The draw kernel: push the surface out along its normal, strongest at the brush centre.

use crate::falloff::Falloff;
use crate::math::{distance_squared, sqrt};

/// The draw brush's per-vertex step: how far each vertex moves along its frame normal.
///
/// For each `positions[k]` strictly within `radius` of `center` (distance `d < radius`), writes
/// `strength · falloff(d / radius)` into `out[k]`; every other `out[k]` gets `0`, so `out` is fully
/// written either way. Returns how many positions were within the radius.
///
/// The value is the delta along the normal axis of the vertex's tangent frame: the caller adds it
/// to the `z` of the vertex's correction. A point exactly on the rim gets `0` and doesn't count.
/// A `radius` that isn't positive and finite covers nothing: `out` is zeroed and the count is 0.
///
/// `positions` and `out` should have the same length. If they don't, only the shorter prefix is
/// processed (the rest of `out` is left as it was); the kernel never panics.
///
/// ```
/// use kernels::{Smoothstep, draw};
///
/// let positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]];
/// let mut out = [9.0; 3];
/// let inside = draw(&Smoothstep, [0.0; 3], 1.0, 0.1, &positions, &mut out);
/// assert_eq!(inside, 2);
/// assert!((out[0] - 0.1).abs() < 1e-7, "full strength at the centre");
/// assert!((out[1] - 0.05).abs() < 1e-7, "half strength halfway");
/// assert!(out[2] == 0.0, "nothing outside");
/// ```
#[inline]
pub fn draw<F: Falloff>(
    falloff: &F,
    center: [f32; 3],
    radius: f32,
    strength: f32,
    positions: &[[f32; 3]],
    out: &mut [f32],
) -> usize {
    let pairs = positions.iter().zip(out.iter_mut());
    if !(radius > 0.0 && radius.is_finite()) {
        pairs.for_each(|(_, o)| *o = 0.0);
        return 0;
    }
    let radius_squared = radius * radius;
    let inv_radius_squared = radius_squared.recip();
    let mut inside = 0;
    for (&p, o) in pairs {
        let d2 = distance_squared(p, center);
        *o = if d2 < radius_squared {
            inside += 1;
            strength * falloff.weight(sqrt(d2 * inv_radius_squared))
        } else {
            0.0
        };
    }
    inside
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::*;
    use crate::falloff::Smoothstep;

    /// A `n × n` grid of points with spacing `step` in the plane `z = 0`, centred on the origin.
    fn grid(n: u16, step: f32) -> Vec<[f32; 3]> {
        let half = f32::from(n - 1) / 2.0;
        (0..n)
            .flat_map(|j| {
                (0..n).map(move |i| {
                    [
                        (f32::from(i) - half) * step,
                        (f32::from(j) - half) * step,
                        0.0,
                    ]
                })
            })
            .collect()
    }

    /// The draw delta at `p`, computed directly with `std`'s square root and the smoothstep
    /// formula: the oracle the kernel is checked against.
    fn direct(p: [f32; 3], center: [f32; 3], radius: f32, strength: f32) -> Option<f32> {
        let [dx, dy, dz] = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        let d = f64::from(dx * dx + dy * dy + dz * dz).sqrt();
        let t = d / f64::from(radius);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the oracle works in f64 and is compared at f32 precision"
        )]
        let weight = (1.0 - (3.0 * t * t - 2.0 * t * t * t)) as f32;
        (t < 1.0).then_some(strength * weight)
    }

    #[test]
    fn draw_matches_a_direct_computation_on_a_grid() {
        let positions = grid(101, 0.02);
        let (center, radius, strength) = ([0.13, -0.07, 0.05], 0.6, 0.25);
        let mut out = std::vec![f32::NAN; positions.len()];
        let inside = draw(&Smoothstep, center, radius, strength, &positions, &mut out);
        let mut expected_inside = 0;
        for (&p, &o) in positions.iter().zip(&out) {
            match direct(p, center, radius, strength) {
                Some(want) => {
                    expected_inside += 1;
                    assert!((o - want).abs() < 1e-6, "at {p:?}: {o} vs {want}");
                }
                None => assert!(o == 0.0, "outside at {p:?}, got {o}"),
            }
        }
        assert_eq!(inside, expected_inside, "the count of positions inside");
        assert!(
            inside > 1000,
            "the brush covers a good part of the grid: {inside}"
        );
        assert!(inside < positions.len(), "and not all of it");
    }

    #[test]
    fn a_point_exactly_on_the_rim_gets_zero() {
        let positions = [[3.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 3.0]];
        let mut out = [1.0; 3];
        let inside = draw(&Smoothstep, [0.0; 3], 3.0, 1.0, &positions, &mut out);
        assert_eq!(inside, 0, "the rim is outside");
        assert!(out.iter().all(|&o| o == 0.0), "and gets nothing: {out:?}");
    }

    #[test]
    fn an_empty_input_is_fine() {
        assert_eq!(
            draw(&Smoothstep, [0.0; 3], 1.0, 1.0, &[], &mut []),
            0,
            "nothing to do"
        );
    }

    #[test]
    fn a_length_mismatch_processes_the_shorter_prefix() {
        let positions = [[0.0; 3]; 4];
        let mut out = [7.0; 6];
        assert_eq!(
            draw(&Smoothstep, [0.0; 3], 1.0, 2.0, &positions, &mut out),
            4,
            "four positions"
        );
        assert_eq!(
            out,
            [2.0, 2.0, 2.0, 2.0, 7.0, 7.0],
            "the tail of out is untouched"
        );
        let mut short = [7.0; 2];
        assert_eq!(
            draw(&Smoothstep, [0.0; 3], 1.0, 2.0, &positions, &mut short),
            2,
            "only as many as out holds"
        );
        assert_eq!(short, [2.0, 2.0], "both written");
    }

    #[test]
    fn a_degenerate_radius_covers_nothing() {
        let positions = [[0.0; 3]; 2];
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut out = [5.0; 2];
            assert_eq!(
                draw(&Smoothstep, [0.0; 3], radius, 1.0, &positions, &mut out),
                0,
                "radius {radius}"
            );
            assert_eq!(out, [0.0; 2], "radius {radius} zeroes out");
        }
    }
}
