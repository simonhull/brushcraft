//! A FLIP-style perceptual difference between two images, ours, for the golden-image tests.
//!
//! It follows the shape of FLIP (Andersson et al., 2020, "FLIP: A Difference Evaluator for
//! Alternating Images"): a colour difference of the images as the eye resolves them at a viewing
//! distance, raised to a power that a difference in edges and points lowers. Each pixel gets an
//! error in `[0, 1]`, 0 where the two can't be told apart, 1 the most different (pure green
//! against pure blue). The constants and filters are this module's own, set below.
//!
//! # The formula
//!
//! Inputs are linear RGB in `[0, 1]` (the frame after the tone map), seen at `ppd` pixels per
//! degree of visual angle ([`PPD`]: a 24-inch 4K monitor at 70 cm).
//!
//! **Colour.**
//!
//! 1. Each image goes to linearized CIELAB, `YyCxCz` (opponent space, linear in light): from linear
//!    sRGB to CIE XYZ (D65), then `y = 116 Y/Yₙ − 16`, `cx = 500 (X/Xₙ − Y/Yₙ)`, `cz = 200 (Y/Yₙ −
//!    Z/Zₙ)`.
//! 2. Each channel is blurred with a normalized Gaussian of the eye's resolving power for it, `σ`
//!    in degrees: achromatic [`SIGMA_Y`], red–green [`SIGMA_CX`], blue–yellow [`SIGMA_CZ`] (colour
//!    is resolved more coarsely than brightness), times `ppd` for pixels; filters reach `3σ`, edges
//!    clamped.
//! 3. Back to linear RGB, clamped to `[0, 1]`, and on to CIELAB `(L, a, b)`; `a` and `b` are scaled
//!    by `L / 100` (the Hunt effect: colour is seen less in the dark).
//! 4. The `HyAB` distance between the two, `ΔE = |ΔL| + √(Δa² + Δb²)`, compressed, `ΔE^0.7`, and
//!    mapped to `[0, 1]` against `c = HyAB(green, blue)^0.7` (the largest in the gamut): linearly
//!    to `0.95` at `0.4 c`, then linearly to 1 at `c`: `e_c`.
//!
//! **Features.**
//!
//! 5. On each image's achromatic channel, normalized (`(y + 16) / 116`), the edge and point
//!    responses: the gradient's length after the first derivatives of a Gaussian of `σ = ½ ·`
//!    [`FEATURE`] `· ppd`, and the length of `(∂²/∂x², ∂²/∂y²)` after its second derivatives, each
//!    filter scaled so its positive weights sum to 1 (and the second's negative ones to −1).
//! 6. `Δf = max(|edge_R − edge_T|, |point_R − point_T|)`, and `e_f = (Δf / √2)^½`.
//!
//! **Together:** `E = e_c^(1 − e_f)`: a feature difference (an edge moved, a point lost) raises
//! the colour error towards 1, so a shifted detail counts for more than a change of tint of the
//! same size. [`FlipMap`] holds `E` per pixel, with its mean, a percentile and its largest.

/// Pixels per degree the images are seen at: 3840 pixels across a 52 cm wide screen at 70 cm.
pub(crate) const PPD: f32 = 67.0;

/// The achromatic channel's blur, degrees.
const SIGMA_Y: f32 = 0.0085;

/// The red–green channel's, degrees.
const SIGMA_CX: f32 = 0.025;

/// The blue–yellow channel's, degrees.
const SIGMA_CZ: f32 = 0.04;

/// The feature detectors' width, degrees.
const FEATURE: f32 = 0.082;

/// D65's white in XYZ.
const WHITE: [f32; 3] = [0.950_47, 1.0, 1.088_83];

/// An image: linear RGB in `[0, 1]`, row by row from the top.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Linear {
    /// Its width.
    pub(crate) width: usize,
    /// Its height.
    pub(crate) height: usize,
    /// Its pixels.
    pub(crate) pixels: Vec<[f32; 3]>,
}

/// The per-pixel error of a comparison, and its summaries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FlipMap {
    /// `E` per pixel, in `[0, 1]`.
    pub(crate) errors: Vec<f32>,
}

impl FlipMap {
    /// The mean error.
    pub(crate) fn mean(&self) -> f32 {
        let sum: f64 = self.errors.iter().map(|&e| f64::from(e)).sum();
        let count = u32::try_from(self.errors.len()).map_or(f64::INFINITY, f64::from);
        #[expect(clippy::cast_possible_truncation, reason = "a mean of errors in 0..1")]
        let mean = (sum / count.max(1.0)) as f32;
        mean
    }

    /// The error at `percent` (0 to 100) of the pixels, sorted.
    pub(crate) fn percentile(&self, percent: usize) -> f32 {
        let mut sorted = self.errors.clone();
        sorted.sort_by(f32::total_cmp);
        let at = (sorted.len().saturating_sub(1) * percent) / 100;
        sorted.get(at).copied().unwrap_or(0.0)
    }

    /// The largest error.
    pub(crate) fn max(&self) -> f32 {
        self.errors.iter().copied().fold(0.0, f32::max)
    }
}

/// Linear sRGB to XYZ.
fn xyz(c: [f32; 3]) -> [f32; 3] {
    [
        0.412_456_4 * c[0] + 0.357_576_1 * c[1] + 0.180_437_5 * c[2],
        0.212_672_9 * c[0] + 0.715_152_2 * c[1] + 0.072_175 * c[2],
        0.019_333_9 * c[0] + 0.119_192 * c[1] + 0.950_304_1 * c[2],
    ]
}

/// XYZ to linear sRGB.
fn rgb(c: [f32; 3]) -> [f32; 3] {
    [
        3.240_454_2 * c[0] - 1.537_138_5 * c[1] - 0.498_531_4 * c[2],
        -0.969_266 * c[0] + 1.876_010_8 * c[1] + 0.041_556 * c[2],
        0.055_643_4 * c[0] - 0.204_025_9 * c[1] + 1.057_225_2 * c[2],
    ]
}

/// Linear RGB to `YyCxCz`.
fn ycxcz(c: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz(c);
    let (x, y, z) = (x / WHITE[0], y / WHITE[1], z / WHITE[2]);
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}

/// `YyCxCz` back to linear RGB.
fn from_ycxcz(c: [f32; 3]) -> [f32; 3] {
    let y = (c[0] + 16.0) / 116.0;
    let x = c[1] / 500.0 + y;
    let z = y - c[2] / 200.0;
    rgb([x * WHITE[0], y * WHITE[1], z * WHITE[2]])
}

/// Linear RGB (clamped) to CIELAB, `a` and `b` scaled by `L / 100` (Hunt).
fn hunt_lab(colour: [f32; 3]) -> [f32; 3] {
    let lab_curve = |ratio: f32| {
        let delta: f32 = 6.0 / 29.0;
        if ratio > delta * delta * delta {
            ratio.cbrt()
        } else {
            ratio / (3.0 * delta * delta) + 4.0 / 29.0
        }
    };
    let [x, y, z] = xyz(colour.map(|v| v.clamp(0.0, 1.0)));
    let (fx, fy, fz) = (
        lab_curve(x / WHITE[0]),
        lab_curve(y / WHITE[1]),
        lab_curve(z / WHITE[2]),
    );
    let lightness = 116.0 * fy - 16.0;
    let (red_green, blue_yellow) = (500.0 * (fx - fy), 200.0 * (fy - fz));
    [
        lightness,
        0.01 * lightness * red_green,
        0.01 * lightness * blue_yellow,
    ]
}

/// The `HyAB` distance between two Hunt-adjusted colours.
fn hyab(p: [f32; 3], q: [f32; 3]) -> f32 {
    (p[0] - q[0]).abs() + (p[1] - q[1]).hypot(p[2] - q[2])
}

/// A normalized Gaussian filter of `sigma` pixels, reaching `3σ` (at least one tap a side).
fn gaussian(sigma: f32) -> Vec<f32> {
    let radius = radius_of(sigma);
    let taps = (-radius..=radius).map(|k| {
        let x = f32::from(i16::try_from(k).expect("a small filter"));
        (-x * x / (2.0 * sigma * sigma)).exp()
    });
    let taps: Vec<f32> = taps.collect();
    let sum: f32 = taps.iter().sum();
    taps.iter().map(|t| t / sum).collect()
}

/// The filter's radius for `sigma` pixels: `⌈3σ⌉`, at least 1.
fn radius_of(sigma: f32) -> i32 {
    #[expect(clippy::cast_possible_truncation, reason = "a filter of a few pixels")]
    let radius = (3.0 * sigma).ceil() as i32;
    radius.max(1)
}

/// `plane` (`width` a row) convolved with `weights` along rows (`along_rows`) or columns, edges
/// clamped.
fn convolve(plane: &[f32], width: usize, weights: &[f32], along_rows: bool) -> Vec<f32> {
    let height = plane.len() / width;
    let radius = i64::try_from(weights.len() / 2).expect("small");
    let mut out = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, weight) in (-radius..=radius).zip(weights) {
                let (sx, sy) = if along_rows {
                    (clamp(x, k, width), y)
                } else {
                    (x, clamp(y, k, height))
                };
                sum += weight * plane[sy * width + sx];
            }
            out[y * width + x] = sum;
        }
    }
    out
}

/// `at + offset`, clamped to `0..len`.
fn clamp(at: usize, offset: i64, len: usize) -> usize {
    let moved = i64::try_from(at).expect("small") + offset;
    usize::try_from(moved.clamp(0, i64::try_from(len).expect("small") - 1)).expect("not negative")
}

/// The edge and point responses of `plane` (normalized luminance): per pixel, the gradient's
/// length and the length of the two second derivatives.
fn features(plane: &[f32], width: usize, sigma: f32) -> (Vec<f32>, Vec<f32>) {
    let radius = radius_of(sigma);
    let xs: Vec<f32> = (-radius..=radius)
        .map(|k| f32::from(i16::try_from(k).expect("small")))
        .collect();
    let g: Vec<f32> = xs
        .iter()
        .map(|x| (-x * x / (2.0 * sigma * sigma)).exp())
        .collect();
    let g_sum: f32 = g.iter().sum();
    let smooth: Vec<f32> = g.iter().map(|v| v / g_sum).collect();
    // First derivative: −x g(x), positive weights scaled to sum to 1.
    let first: Vec<f32> = xs.iter().zip(&g).map(|(x, v)| -x * v).collect();
    let first_pos: f32 = first.iter().filter(|v| **v > 0.0).sum();
    let first: Vec<f32> = first.iter().map(|v| v / first_pos).collect();
    // Second derivative: (x² / σ² − 1) g(x), positive and negative parts scaled to ±1.
    let second: Vec<f32> = xs
        .iter()
        .zip(&g)
        .map(|(x, v)| (x * x / (sigma * sigma) - 1.0) * v)
        .collect();
    let pos: f32 = second.iter().filter(|v| **v > 0.0).sum();
    let neg: f32 = -second.iter().filter(|v| **v < 0.0).sum::<f32>();
    let second: Vec<f32> = second
        .iter()
        .map(|&v| if v > 0.0 { v / pos } else { v / neg })
        .collect();
    let dx = convolve(&convolve(plane, width, &first, true), width, &smooth, false);
    let dy = convolve(&convolve(plane, width, &smooth, true), width, &first, false);
    let dxx = convolve(
        &convolve(plane, width, &second, true),
        width,
        &smooth,
        false,
    );
    let dyy = convolve(
        &convolve(plane, width, &smooth, true),
        width,
        &second,
        false,
    );
    let edges = dx.iter().zip(&dy).map(|(a, b)| a.hypot(*b)).collect();
    let points = dxx.iter().zip(&dyy).map(|(a, b)| a.hypot(*b)).collect();
    (edges, points)
}

/// The three blurred `YyCxCz` planes of `image`.
fn filtered(image: &Linear) -> [Vec<f32>; 3] {
    let opponent: Vec<[f32; 3]> = image.pixels.iter().map(|&c| ycxcz(c)).collect();
    let sigmas = [SIGMA_Y, SIGMA_CX, SIGMA_CZ];
    [0, 1, 2].map(|k| {
        let plane: Vec<f32> = opponent.iter().map(|c| c[k]).collect();
        let filter = gaussian(sigmas[k] * PPD);
        let rows = convolve(&plane, image.width, &filter, true);
        convolve(&rows, image.width, &filter, false)
    })
}

/// The FLIP-style error of `test` against `reference` (the same size), per pixel.
///
/// # Panics
///
/// If the two differ in size.
pub(crate) fn flip(reference: &Linear, test: &Linear) -> FlipMap {
    assert_eq!(
        (reference.width, reference.height),
        (test.width, test.height),
        "images of one size"
    );
    let width = reference.width;
    let max = hyab(hunt_lab([0.0, 1.0, 0.0]), hunt_lab([0.0, 0.0, 1.0])).powf(0.7);
    let (pc, pt) = (0.4, 0.95);
    let [ry, rx, rz] = filtered(reference);
    let [ty, tx, tz] = filtered(test);
    let colour = (0..ry.len()).map(|i| {
        let r = hunt_lab(from_ycxcz([ry[i], rx[i], rz[i]]));
        let t = hunt_lab(from_ycxcz([ty[i], tx[i], tz[i]]));
        let d = hyab(r, t).powf(0.7);
        if d < pc * max {
            pt / (pc * max) * d
        } else {
            pt + (d - pc * max) / (max - pc * max) * (1.0 - pt)
        }
    });
    let colour: Vec<f32> = colour.collect();
    let luminance = |image: &Linear| -> Vec<f32> {
        let y = image.pixels.iter().map(|&c| (ycxcz(c)[0] + 16.0) / 116.0);
        y.collect()
    };
    let sigma = 0.5 * FEATURE * PPD;
    let (re, rp) = features(&luminance(reference), width, sigma);
    let (te, tp) = features(&luminance(test), width, sigma);
    let errors = colour
        .iter()
        .enumerate()
        .map(|(i, &ec)| {
            let df = (re[i] - te[i]).abs().max((rp[i] - tp[i]).abs());
            let ef = (df / std::f32::consts::SQRT_2).sqrt().min(1.0);
            ec.clamp(0.0, 1.0).powf(1.0 - ef)
        })
        .collect();
    FlipMap { errors }
}

#[cfg(test)]
mod checks {
    use super::*;

    /// A `width` × `height` image, `paint` giving each pixel.
    fn image(width: usize, height: usize, paint: impl Fn(usize, usize) -> [f32; 3]) -> Linear {
        let pixels = (0..height).flat_map(|y| (0..width).map(move |x| (x, y)));
        Linear {
            width,
            height,
            pixels: pixels.map(|(x, y)| paint(x, y)).collect(),
        }
    }

    #[test]
    fn the_flip_measure_is_zero_for_equal_images_and_ranks_differences_as_the_eye_does() {
        let grey = image(48, 32, |_, _| [0.2; 3]);
        assert!(flip(&grey, &grey).max() < 1e-6, "equal images");
        // The round trip through the opponent space is the identity.
        let c = [0.3, 0.6, 0.1];
        let back = from_ycxcz(ycxcz(c));
        assert!(
            c.iter().zip(back).all(|(a, b)| (a - b).abs() < 1e-5),
            "{back:?}"
        );
        // Green against blue is the most different there is.
        let green = image(48, 32, |_, _| [0.0, 1.0, 0.0]);
        let blue = image(48, 32, |_, _| [0.0, 0.0, 1.0]);
        assert!(flip(&green, &blue).mean() > 0.99, "the extreme");
        // A slight tint everywhere is less than a sharp edge moved by two pixels.
        let edge = |at: usize| {
            image(
                48,
                32,
                move |x, _| if x < at { [0.05; 3] } else { [0.8; 3] },
            )
        };
        let tinted = image(48, 32, |x, _| if x < 24 { [0.055; 3] } else { [0.78; 3] });
        let moved = flip(&edge(24), &edge(26));
        let tint = flip(&edge(24), &tinted);
        assert!(
            moved.max() > 2.0 * tint.max(),
            "{} against {}",
            moved.max(),
            tint.max()
        );
        // Fine noise a viewer can't resolve counts for little.
        let noise = image(48, 32, |x, y| {
            if (x + y) % 2 == 0 {
                [0.21; 3]
            } else {
                [0.19; 3]
            }
        });
        assert!(
            flip(&grey, &noise).mean() < 0.05,
            "noise: {}",
            flip(&grey, &noise).mean()
        );
        assert!(flip(&grey, &noise).percentile(50) <= flip(&grey, &noise).max());
    }
}
