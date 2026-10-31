//! `Curve`: the piecewise-linear response curve a dynamics entry reads its input through.

use crate::Unit;

/// The fewest points a curve has: a straight line.
pub const MIN_CURVE_POINTS: usize = 2;
/// The most points a curve has. Enough for an S-curve with a plateau; small enough that a curve
/// is plain data with no allocation.
pub const MAX_CURVE_POINTS: usize = 8;

/// A curve that can't be built.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum CurveError {
    /// Fewer than [`MIN_CURVE_POINTS`] or more than [`MAX_CURVE_POINTS`] points.
    #[error("a curve has {MIN_CURVE_POINTS} to {MAX_CURVE_POINTS} points, not {0}")]
    PointCount(usize),
    /// A point's input or output is outside `0..=1`, or NaN.
    #[error("point {index} ({input}, {output}) is outside 0..=1")]
    OutOfRange {
        /// The point's position in the list.
        index: usize,
        /// Its input.
        input: f32,
        /// Its output.
        output: f32,
    },
    /// A point's input is not greater than the one before.
    #[error("point {index}'s input is not greater than the point before's")]
    NotIncreasing {
        /// The point's position in the list.
        index: usize,
    },
}

/// One point of a [`Curve`]: the output it gives at an input.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CurvePoint {
    /// Where on the input axis the point sits.
    pub input: Unit,
    /// The output there.
    pub output: Unit,
}

/// A piecewise-linear response curve from `0..=1` to `0..=1`.
///
/// It has [`MIN_CURVE_POINTS`] to [`MAX_CURVE_POINTS`] points whose inputs strictly increase and
/// whose inputs and outputs all lie in `0..=1`; [`Curve::new`] refuses anything else, so a
/// `Curve` is always valid. Between points it interpolates linearly; outside the first and last
/// input it holds the end point's output ([`Curve::eval`] clamps).
///
/// The output need not be monotone: a curve may rise and fall. Only the inputs must increase,
/// so each input has one output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    /// The points, the first [`Curve::len`] of them in use; the rest are zero, so equal curves
    /// compare equal.
    points: [CurvePoint; MAX_CURVE_POINTS],
    /// How many points are in use, in `MIN_CURVE_POINTS..=MAX_CURVE_POINTS`.
    len: u8,
}

impl Curve {
    /// The identity: output equals input.
    pub const LINEAR: Self = {
        let mut points = [CurvePoint {
            input: Unit::ZERO,
            output: Unit::ZERO,
        }; MAX_CURVE_POINTS];
        points[1] = CurvePoint {
            input: Unit::ONE,
            output: Unit::ONE,
        };
        Self { points, len: 2 }
    };

    /// A curve through `points`, given as `(input, output)` pairs.
    ///
    /// # Errors
    ///
    /// - [`CurveError::PointCount`] for fewer than [`MIN_CURVE_POINTS`] or more than
    ///   [`MAX_CURVE_POINTS`] points;
    /// - [`CurveError::OutOfRange`] for a coordinate outside `0..=1` or NaN;
    /// - [`CurveError::NotIncreasing`] for an input not greater than the one before.
    pub fn new(points: &[(f32, f32)]) -> Result<Self, CurveError> {
        let len = points.len();
        if !(MIN_CURVE_POINTS..=MAX_CURVE_POINTS).contains(&len) {
            return Err(CurveError::PointCount(len));
        }
        let mut curve = Self {
            points: [CurvePoint::default(); MAX_CURVE_POINTS],
            len: u8::try_from(len).ok().ok_or(CurveError::PointCount(len))?,
        };
        let mut previous: Option<Unit> = None;
        let slots = curve.points.iter_mut().zip(points).enumerate();
        for (index, (slot, &(input, output))) in slots {
            let (Ok(x), Ok(y)) = (Unit::new(input), Unit::new(output)) else {
                return Err(CurveError::OutOfRange {
                    index,
                    input,
                    output,
                });
            };
            if previous.is_some_and(|p| x <= p) {
                return Err(CurveError::NotIncreasing { index });
            }
            previous = Some(x);
            *slot = CurvePoint {
                input: x,
                output: y,
            };
        }
        Ok(curve)
    }

    /// The points, in order of input.
    #[must_use]
    pub fn points(&self) -> &[CurvePoint] {
        // `len` is at most MAX_CURVE_POINTS by construction, so the slice is never empty.
        self.points
            .get(..usize::from(self.len))
            .unwrap_or(&self.points)
    }

    /// The curve's output at `input`, in `0..=1`.
    ///
    /// Inputs before the first point give the first point's output, inputs after the last give
    /// the last's. NaN reads as `0`, the rest position of every input (see [`Curve::rest`]).
    #[must_use]
    pub fn eval(&self, input: f32) -> f32 {
        let x = if input.is_nan() { 0.0 } else { input };
        let points = self.points();
        let (Some(first), Some(last)) = (points.first(), points.last()) else {
            return 0.0;
        };
        if x <= first.input.get() {
            return first.output.get();
        }
        if x >= last.input.get() {
            return last.output.get();
        }
        let segment = points
            .windows(2)
            .find_map(|pair| match *pair {
                [a, b] if x <= b.input.get() => Some((a, b)),
                _ => None,
            })
            .unwrap_or((*last, *last));
        let (a, b) = segment;
        let (x0, x1) = (a.input.get(), b.input.get());
        let (y0, y1) = (a.output.get(), b.output.get());
        // Inputs strictly increase, so the span is positive.
        let t = (x - x0) / (x1 - x0);
        (y0 + (y1 - y0) * t).clamp(0.0, 1.0)
    }

    /// The output at input `0`: what the curve gives for an input the pen doesn't send.
    ///
    /// Every dynamics input reads `0` at rest (no pressure, an upright pen, no twist, standing
    /// still, the stroke's start), so a missing input holds its curve at this value.
    #[must_use]
    pub fn rest(&self) -> f32 {
        self.eval(0.0)
    }
}

impl Default for Curve {
    fn default() -> Self {
        Self::LINEAR
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn endpoints_are_exact() {
        let curve = Curve::new(&[(0.1, 0.2), (0.5, 0.9), (0.8, 0.4)]).unwrap();
        assert!(close(curve.eval(0.1), 0.2), "first point");
        assert!(close(curve.eval(0.5), 0.9), "middle point");
        assert!(close(curve.eval(0.8), 0.4), "last point");
        assert_eq!(curve.points().len(), 3, "three points kept");
    }

    #[test]
    fn interpolates_linearly_between_points() {
        let curve = Curve::new(&[(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]).unwrap();
        assert!(close(curve.eval(0.25), 0.5), "rising half");
        assert!(close(curve.eval(0.75), 0.5), "falling half");
        assert!(close(Curve::LINEAR.eval(0.3), 0.3), "identity");
    }

    #[test]
    fn clamps_outside_the_points() {
        let curve = Curve::new(&[(0.2, 0.3), (0.6, 0.7)]).unwrap();
        assert!(close(curve.eval(0.0), 0.3), "before the first input");
        assert!(close(curve.eval(-5.0), 0.3), "below the range");
        assert!(close(curve.eval(1.0), 0.7), "after the last input");
        assert!(close(curve.eval(42.0), 0.7), "above the range");
        assert!(close(curve.eval(f32::NAN), 0.3), "NaN reads as rest");
        assert!(close(curve.rest(), 0.3), "rest is the value at 0");
    }

    #[test]
    fn refuses_every_invalid_curve() {
        assert_eq!(
            Curve::new(&[(0.0, 0.0)]),
            Err(CurveError::PointCount(1)),
            "one point"
        );
        assert_eq!(Curve::new(&[]), Err(CurveError::PointCount(0)), "none");
        let nine: Vec<(f32, f32)> = (0..9_u8).map(|i| (f32::from(i) / 8.0, 0.5)).collect();
        assert_eq!(
            Curve::new(&nine),
            Err(CurveError::PointCount(9)),
            "nine points"
        );
        let eight = &nine[..8];
        assert!(Curve::new(eight).is_ok(), "eight points are fine");
        assert!(
            matches!(
                Curve::new(&[(0.0, 0.0), (1.1, 1.0)]),
                Err(CurveError::OutOfRange { index: 1, .. })
            ),
            "input above 1"
        );
        assert!(
            matches!(
                Curve::new(&[(-0.1, 0.0), (1.0, 1.0)]),
                Err(CurveError::OutOfRange { index: 0, .. })
            ),
            "input below 0"
        );
        assert!(
            matches!(
                Curve::new(&[(0.0, 0.0), (1.0, 1.5)]),
                Err(CurveError::OutOfRange { index: 1, .. })
            ),
            "output above 1"
        );
        assert!(
            matches!(
                Curve::new(&[(0.0, f32::NAN), (1.0, 1.0)]),
                Err(CurveError::OutOfRange { index: 0, .. })
            ),
            "NaN output"
        );
        assert_eq!(
            Curve::new(&[(0.0, 0.0), (0.5, 1.0), (0.5, 0.0)]),
            Err(CurveError::NotIncreasing { index: 2 }),
            "repeated input"
        );
        assert_eq!(
            Curve::new(&[(0.6, 0.0), (0.5, 1.0)]),
            Err(CurveError::NotIncreasing { index: 1 }),
            "decreasing input"
        );
    }

    #[test]
    fn output_stays_in_range_for_any_input() {
        bolero::check!()
            .with_type::<(Vec<(u8, u8)>, f32)>()
            .for_each(|(raw, x)| {
                let points: Vec<(f32, f32)> = raw
                    .iter()
                    .map(|&(i, o)| (f32::from(i) / 255.0, f32::from(o) / 255.0))
                    .collect();
                if let Ok(curve) = Curve::new(&points) {
                    let y = curve.eval(*x);
                    assert!((0.0..=1.0).contains(&y), "{y} out of range at {x}");
                }
            });
    }
}
