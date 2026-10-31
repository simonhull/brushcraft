//! A dab's imprint: the values an action reads from one placed dab, and the hardness that
//! shapes a falloff.

use crate::draw::draw_vertex;
use crate::falloff::Falloff;

/// One dab as an action reads it: where it is, its frame's normal, how far it reaches, how
/// strongly it acts, how hard its falloff is, and the action's own parameters.
///
/// Plain `f32` data (`#[repr(C)]`), the same on the CPU and the GPU. It is the `brush` side of
/// the stroke engine's dab: `brush` depends on nothing, so the caller copies the dab's fields in.
/// Every value is final: the caller has applied its brush's own strength and hardness.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imprint {
    /// The centre, in model space.
    pub center: [f32; 3],
    /// The surface normal at the centre, unit length, for actions that move along it (draw
    /// moves each vertex along its own frame normal and doesn't read it).
    pub normal: [f32; 3],
    /// How far the dab reaches from its centre: positive and finite for the dab to cover
    /// anything.
    pub radius: f32,
    /// `radius * radius`, computed once in `f32` ([`Imprint::new`] does).
    pub radius_squared: f32,
    /// The action's strength: for draw, how far a vertex at the centre moves, in model units.
    pub strength: f32,
    /// The falloff's hardness, `0..=1` (see [`Hardened`]): `0` is the falloff itself, `1` a
    /// disc of full strength.
    pub hardness: f32,
    /// The action's own parameters, as it defines them (draw reads none).
    pub params: [f32; 4],
}

impl Imprint {
    /// The imprint of a dab at `center` with `normal`, reaching `radius`, of `strength` and
    /// `hardness`, with the action's `params`; its squared radius computed here.
    #[must_use]
    pub fn new(
        center: [f32; 3],
        normal: [f32; 3],
        radius: f32,
        strength: f32,
        hardness: f32,
        params: [f32; 4],
    ) -> Self {
        Self {
            center,
            normal,
            radius,
            radius_squared: radius * radius,
            strength,
            hardness,
            params,
        }
    }
}

/// A falloff made harder: full strength out to `hardness` of the radius, then the falloff
/// squeezed into what is left of it.
///
/// `weight(t)` is `1` for `t ≤ h`, else `falloff.weight((t − h) / (1 − h))`, with `h` the
/// hardness clamped to `0..=1` (a NaN counts as `0`). At `h = 0` that is the falloff itself, bit
/// for bit: `(t − 0) / (1 − 0)` is `t` exactly, and the falloff is `1` at the centre. At `h = 1`
/// every point inside the radius gets full strength (past it, `(t − 1) / 0` is `+∞` and the
/// falloff `0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hardened<'a, F> {
    /// The falloff made harder.
    falloff: &'a F,
    /// The hardness, clamped to `0..=1`.
    hardness: f32,
}

impl<'a, F: Falloff> Hardened<'a, F> {
    /// `falloff` with `hardness`, clamped to `0..=1`; a NaN counts as `0`.
    #[must_use]
    pub fn new(falloff: &'a F, hardness: f32) -> Self {
        // Written with comparisons rather than `f32::clamp`, which panics on a NaN bound and
        // passes a NaN through: a NaN hardness fails both tests and becomes 0.
        let hardness = if hardness > 0.0 {
            if hardness < 1.0 { hardness } else { 1.0 }
        } else {
            0.0
        };
        Self { falloff, hardness }
    }
}

impl<F: Falloff> Falloff for Hardened<'_, F> {
    #[inline]
    fn weight(&self, t: f32) -> f32 {
        let h = self.hardness;
        if t <= h {
            1.0
        } else {
            self.falloff.weight((t - h) / (1.0 - h))
        }
    }
}

/// The displacement `imprint` gives the vertex at `p`, or `None` outside its sphere: the draw
/// brush's step for one vertex, [`draw_vertex`] with the imprint's values and its falloff made
/// [`Hardened`].
///
/// At hardness `0` it is [`draw_vertex`] with `falloff`, bit for bit. `imprint.radius` must be
/// positive and finite and `imprint.radius_squared` its square ([`Imprint::new`]);
/// [`draw_imprint`] checks the radius once for all vertices.
///
/// ```
/// use brush::{Imprint, Smoothstep, draw_imprint_vertex};
///
/// let soft = Imprint::new([0.0; 3], [0.0, 0.0, 1.0], 1.0, 0.1, 0.0, [0.0; 4]);
/// let hard = Imprint {
///     hardness: 0.5,
///     ..soft
/// };
/// let at = |imprint: &Imprint, p| draw_imprint_vertex(&Smoothstep, imprint, p).unwrap();
/// assert!(
///     (at(&soft, [0.5, 0.0, 0.0]) - 0.05).abs() < 1e-7,
///     "half strength halfway"
/// );
/// assert_eq!(
///     at(&hard, [0.5, 0.0, 0.0]),
///     0.1,
///     "full strength to half the radius"
/// );
/// assert!(
///     (at(&hard, [0.75, 0.0, 0.0]) - 0.05).abs() < 1e-7,
///     "then the falloff"
/// );
/// ```
#[inline]
pub fn draw_imprint_vertex<F: Falloff>(falloff: &F, imprint: &Imprint, p: [f32; 3]) -> Option<f32> {
    draw_vertex(
        &Hardened::new(falloff, imprint.hardness),
        imprint.center,
        imprint.radius,
        imprint.radius_squared,
        imprint.strength,
        p,
    )
}

/// [`draw`](crate::draw()) for an [`Imprint`]: writes [`draw_imprint_vertex`]'s displacement
/// of each of `positions` into `out` (`0` outside the sphere) and returns how many were inside.
///
/// The same rules as [`draw`](crate::draw()): a radius that isn't positive and finite covers
/// nothing (`out` zeroed, count `0`), and of mismatched lengths only the shorter prefix is
/// processed. At hardness `0` it is [`draw`](crate::draw()), bit for bit.
#[inline]
pub fn draw_imprint<F: Falloff>(
    falloff: &F,
    imprint: &Imprint,
    positions: &[[f32; 3]],
    out: &mut [f32],
) -> usize {
    let pairs = positions.iter().zip(out.iter_mut());
    let radius = imprint.radius;
    if !(radius > 0.0 && radius.is_finite()) {
        pairs.for_each(|(_, o)| *o = 0.0);
        return 0;
    }
    let hardened = Hardened::new(falloff, imprint.hardness);
    let mut inside = 0;
    for (&p, o) in pairs {
        let moved = draw_vertex(
            &hardened,
            imprint.center,
            radius,
            imprint.radius_squared,
            imprint.strength,
            p,
        );
        match moved {
            Some(w) => {
                inside += 1;
                *o = w;
            }
            None => *o = 0.0,
        }
    }
    inside
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use super::*;
    use crate::draw::draw;
    use crate::falloff::Smoothstep;

    /// A line of `n` points along x, `step` apart from the origin.
    fn line(n: u16, step: f32) -> Vec<[f32; 3]> {
        (0..n).map(|k| [f32::from(k) * step, 0.0, 0.0]).collect()
    }

    #[test]
    fn hardness_zero_is_draw_bit_for_bit() {
        let positions: Vec<[f32; 3]> = (0..2000_u16)
            .map(|k| {
                // Scattered in a box around the centre by multiplicative hashing.
                let at = |m: u32| f32::from(u16::try_from((u32::from(k) * m) % 1000).unwrap());
                [
                    at(7919) * 0.0012 - 0.6,
                    at(104_729) * 0.0008 - 0.4,
                    at(1_299_709) * 0.0006 - 0.3,
                ]
            })
            .collect();
        let (center, radius, strength) = ([0.05, -0.02, 0.01], 0.55, -0.125);
        let mut plain = std::vec![f32::NAN; positions.len()];
        let mut imprinted = std::vec![f32::NAN; positions.len()];
        let inside = draw(
            &Smoothstep,
            center,
            radius,
            strength,
            &positions,
            &mut plain,
        );
        let imprint = Imprint::new(center, [0.0, 0.0, 1.0], radius, strength, 0.0, [0.0; 4]);
        let counted = draw_imprint(&Smoothstep, &imprint, &positions, &mut imprinted);
        assert_eq!(inside, counted, "the same vertices inside");
        assert!(inside > 100 && inside < positions.len(), "a mix: {inside}");
        for (k, (a, b)) in plain.iter().zip(&imprinted).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "vertex {k}");
        }
    }

    #[test]
    fn a_hard_imprint_is_full_strength_inside_its_hardness_then_falls() {
        let positions = line(101, 0.01);
        let imprint = Imprint::new([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0, 0.4, [0.0; 4]);
        let mut out = std::vec![f32::NAN; positions.len()];
        let inside = draw_imprint(&Smoothstep, &imprint, &positions, &mut out);
        assert_eq!(inside, 100, "every point but the rim");
        for (k, (&p, &o)) in positions.iter().zip(&out).enumerate() {
            let t = p[0];
            if t <= 0.4 {
                assert_eq!(o, 2.0, "full strength at t = {t}");
            } else if k < 100 {
                let s = (t - 0.4) / 0.6;
                let want = 2.0 * (1.0 - s * s * (3.0 - 2.0 * s));
                assert!(
                    (o - want).abs() < 1e-5,
                    "squeezed falloff at t = {t}: {o} vs {want}"
                );
            } else {
                assert_eq!(o, 0.0, "nothing on the rim");
            }
        }
        let falling = out.windows(2).all(|w| matches!(w, [a, b] if b <= a));
        assert!(falling, "never rises outwards: {out:?}");
    }

    #[test]
    fn hardness_is_clamped_and_a_nan_is_soft() {
        let positions = line(10, 0.1);
        let base = Imprint::new([0.0; 3], [0.0, 0.0, 1.0], 1.0, 1.0, 0.0, [0.0; 4]);
        let run = |hardness: f32| {
            let mut out = [f32::NAN; 10];
            draw_imprint(
                &Smoothstep,
                &Imprint { hardness, ..base },
                &positions,
                &mut out,
            );
            out
        };
        assert_eq!(run(-3.0), run(0.0), "a negative hardness is 0");
        assert_eq!(run(f32::NAN), run(0.0), "a NaN hardness is 0");
        assert_eq!(run(7.0), run(1.0), "past 1 is 1");
        assert_eq!(run(1.0), [1.0; 10], "hardness 1 is a disc of full strength");
    }

    #[test]
    fn a_degenerate_radius_covers_nothing() {
        let positions = [[0.0; 3]; 2];
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let imprint = Imprint::new([0.0; 3], [0.0, 0.0, 1.0], radius, 1.0, 0.5, [0.0; 4]);
            let mut out = [5.0; 2];
            assert_eq!(
                draw_imprint(&Smoothstep, &imprint, &positions, &mut out),
                0,
                "radius {radius}"
            );
            assert_eq!(out, [0.0; 2], "radius {radius} zeroes out");
        }
    }
}
