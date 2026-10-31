//! The screen path: the pen's samples smoothed by a speed-adaptive filter, with an optional lazy
//! string, completed to the last sample when the pen lifts.
//!
//! **The filter** is a One-Euro-style low-pass on the screen position (after `input`'s Dejitter
//! has fixed the timing). Each step is an exponential smoothing whose cutoff frequency rises
//! with the pen's speed: `cutoff = min_cutoff + cutoff_slope · speed`. Slow, careful motion is
//! smoothed strongly; fast motion hardly at all. The speed is the raw sample-to-sample velocity,
//! itself smoothed at `derivative_cutoff`, so jitter doesn't read as speed.
//!
//! **The lag it costs.** For a pen moving steadily at speed `v`, an exponential smoothing with
//! time constant `τ = 1 / (2π · cutoff)` settles `v · τ` behind the pen, and with the cutoff
//! above that is `v / (2π (min_cutoff + cutoff_slope · v))`, which never reaches
//! `1 / (2π · cutoff_slope)` pixels however fast the pen goes: [`PathSettings::lag_bound`]. Until
//! the speed estimate has caught up with a sudden start (a few of its time constants,
//! `1 / (2π · derivative_cutoff)`) the lag is larger.
//!
//! **Measured** with the defaults, on a simulated 200 Hz pen (`path::tests`, `report`):
//! - jitter of ±1 px on each axis (0.58 px RMS off the line) leaves 0.12 px at 5 px/s, 0.13 px at
//!   20 px/s and 0.16 px at 50 px/s: 21–27 % of the input;
//! - a straight line settles 2.65 px behind the pen at 100 px/s, 3.12 px at 1000 px/s and 3.18 px
//!   at 10 000 px/s, against the bound of 3.18 px;
//! - from a standing start at full speed, the worst lag is 4.4 px at 300 px/s, 7.1 px at 1000 px/s
//!   and 10.8 px at 3000 px/s, in the first tens of milliseconds.
//!
//! **The lazy string** ([`PathSettings::lazy`], `0` = off) makes the brush point trail the filtered
//! point by a set distance: the brush moves only when the filtered point pulls the string taut,
//! and then along the string, for deliberate clean lines.
//!
//! **The tail.** [`Path::end`] hands back the last raw sample, so a stroke always reaches where
//! the pen lifted, whatever the filter or the string held back.

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

use std::f64::consts::TAU;

use crate::{Sample, ScreenPoint};

/// A path setting out of range.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum PathError {
    /// The minimum cutoff isn't positive and finite.
    #[error("the minimum cutoff must be positive and finite, not {0} Hz")]
    MinCutoff(f64),
    /// The cutoff slope is negative or not finite.
    #[error("the cutoff slope must be zero or more and finite, not {0}")]
    CutoffSlope(f64),
    /// The speed estimate's cutoff isn't positive and finite.
    #[error("the derivative cutoff must be positive and finite, not {0} Hz")]
    DerivativeCutoff(f64),
    /// The lazy string's length is negative or not finite.
    #[error("the lazy string must be zero or more and finite, not {0} px")]
    Lazy(f64),
}

/// How the screen path is smoothed: the filter's constants and the lazy string.
///
/// The defaults are a starting point, not tuned: tuning them with the user on the tablet is an
/// open question of the brush-engine spec. With them, jitter on a slow line falls to about a
/// quarter, and the lag stays under 3.18 px at any speed once the speed estimate has settled
/// (measured in the `path` module's notes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSettings {
    /// The cutoff at rest, in Hz: how strongly slow motion is smoothed (lower is smoother).
    min_cutoff: f64,
    /// How much the cutoff rises per pixel per second of speed, in Hz per (px/s), that is, per
    /// pixel. Higher means less lag on fast motion, and sets the lag bound.
    cutoff_slope: f64,
    /// The cutoff of the speed estimate, in Hz: lower ignores more jitter, higher follows a
    /// sudden start sooner.
    derivative_cutoff: f64,
    /// The lazy string's length, in physical pixels; `0` turns it off.
    lazy: f64,
}

impl PathSettings {
    /// The default cutoff slope: 0.05 Hz per px/s, a lag bound of about 3.2 px.
    pub const DEFAULT_CUTOFF_SLOPE: f64 = 0.05;
    /// The default cutoff of the speed estimate: 5 Hz.
    pub const DEFAULT_DERIVATIVE_CUTOFF: f64 = 5.0;
    /// The default cutoff at rest: 1 Hz.
    pub const DEFAULT_MIN_CUTOFF: f64 = 1.0;

    /// Settings with these constants (see the fields' meanings on [`PathSettings`]'s getters).
    ///
    /// # Errors
    ///
    /// - [`PathError::MinCutoff`] or [`PathError::DerivativeCutoff`] if a cutoff isn't positive and
    ///   finite;
    /// - [`PathError::CutoffSlope`] if the slope is negative or not finite;
    /// - [`PathError::Lazy`] if the string's length is negative or not finite.
    pub fn new(
        min_cutoff: f64,
        cutoff_slope: f64,
        derivative_cutoff: f64,
        lazy: f64,
    ) -> Result<Self, PathError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        if !positive(min_cutoff) {
            return Err(PathError::MinCutoff(min_cutoff));
        }
        if !non_negative(cutoff_slope) {
            return Err(PathError::CutoffSlope(cutoff_slope));
        }
        if !positive(derivative_cutoff) {
            return Err(PathError::DerivativeCutoff(derivative_cutoff));
        }
        if !non_negative(lazy) {
            return Err(PathError::Lazy(lazy));
        }
        Ok(Self {
            min_cutoff,
            cutoff_slope: cutoff_slope + 0.0,
            derivative_cutoff,
            lazy: lazy + 0.0,
        })
    }

    /// These settings with a lazy string of `length` physical pixels (`0` = off).
    ///
    /// # Errors
    ///
    /// [`PathError::Lazy`] if `length` is negative or not finite.
    pub fn with_lazy(self, length: f64) -> Result<Self, PathError> {
        Self::new(
            self.min_cutoff,
            self.cutoff_slope,
            self.derivative_cutoff,
            length,
        )
    }

    /// The cutoff at rest, in Hz: how strongly slow motion is smoothed (lower is smoother).
    #[must_use]
    pub const fn min_cutoff(&self) -> f64 {
        self.min_cutoff
    }

    /// How much the cutoff rises with speed, in Hz per px/s. Higher means less lag on fast motion.
    #[must_use]
    pub const fn cutoff_slope(&self) -> f64 {
        self.cutoff_slope
    }

    /// The cutoff of the speed estimate, in Hz.
    #[must_use]
    pub const fn derivative_cutoff(&self) -> f64 {
        self.derivative_cutoff
    }

    /// The lazy string's length, in physical pixels; `0` when it is off.
    #[must_use]
    pub const fn lazy(&self) -> f64 {
        self.lazy
    }

    /// The most the filter lags a pen moving steadily at any speed, in pixels, once its speed
    /// estimate has settled: `1 / (2π · cutoff_slope)`. Infinite with a zero slope. The lazy
    /// string adds its length on top.
    #[must_use]
    pub fn lag_bound(&self) -> f64 {
        (TAU * self.cutoff_slope).recip()
    }

    /// The cutoff for a pen moving at `speed` pixels per second, in Hz.
    fn cutoff(&self, speed: f64) -> f64 {
        self.min_cutoff + self.cutoff_slope * speed
    }
}

impl Default for PathSettings {
    fn default() -> Self {
        Self {
            min_cutoff: Self::DEFAULT_MIN_CUTOFF,
            cutoff_slope: Self::DEFAULT_CUTOFF_SLOPE,
            derivative_cutoff: Self::DEFAULT_DERIVATIVE_CUTOFF,
            lazy: 0.0,
        }
    }
}

/// The fastest speed the filter believes, in pixels per second: far beyond any pen, and small
/// enough that the velocity arithmetic can't overflow.
const MAX_SPEED: f64 = 1.0e9;

/// The smoothing weight of one step of `dt` seconds for a low-pass at `cutoff` Hz: the fraction
/// of the way the output moves toward the input, `dt / (dt + τ)` with `τ = 1 / (2π · cutoff)`.
fn smoothing(cutoff: f64, dt: f64) -> f64 {
    let tau = (TAU * cutoff).recip();
    dt / (dt + tau)
}

/// One stroke's screen path: feed it the stroke's samples with [`Path::push`], then call
/// [`Path::end`] when the pen lifts.
#[derive(Debug, Clone)]
pub struct Path {
    /// The filter's constants and the lazy string.
    settings: PathSettings,
    /// The filter's state; `None` before the stroke's first sample.
    state: Option<State>,
}

/// The filter's state within a stroke.
#[derive(Debug, Clone, Copy)]
struct State {
    /// The latest raw sample: what the speed is measured from, and the tail.
    last: Sample,
    /// The filtered pen point.
    filtered: ScreenPoint,
    /// The smoothed velocity, in pixels per second, `(x, y)`.
    velocity: [f64; 2],
    /// The brush point: the filtered point, held back by the lazy string when it is on.
    brush: ScreenPoint,
}

impl Path {
    /// A path that has seen no sample yet.
    #[must_use]
    pub const fn new(settings: PathSettings) -> Self {
        Self {
            settings,
            state: None,
        }
    }

    /// The settings it smooths with.
    #[must_use]
    pub const fn settings(&self) -> &PathSettings {
        &self.settings
    }

    /// Takes the stroke's next sample and returns the brush point for it: the sample with its
    /// position smoothed (and held back by the lazy string), its other readings as they were.
    ///
    /// The first sample passes through unchanged. A sample no later than the one before (on the
    /// pen's clock) carries no timing to filter with: it becomes the latest raw sample, for the
    /// next step's speed and for the tail, and the brush point stays where it was.
    pub fn push(&mut self, sample: Sample) -> Sample {
        let Some(state) = &mut self.state else {
            let position = sample.position();
            self.state = Some(State {
                last: sample,
                filtered: position,
                velocity: [0.0, 0.0],
                brush: position,
            });
            return sample;
        };
        let dt = sample.time() - state.last.time();
        let (from, to) = (state.last.position(), sample.position());
        state.last = sample;
        // Also false for NaN, which finite times can't give, but costs nothing to rule out.
        if dt.is_nan() || dt <= 0.0 {
            return sample.at(state.brush);
        }
        // Clamped so a near-zero step can't overflow the speed: no pen moves faster.
        let raw = [(to.x() - from.x()) / dt, (to.y() - from.y()) / dt]
            .map(|v| v.clamp(-MAX_SPEED, MAX_SPEED));
        let follow = smoothing(self.settings.derivative_cutoff, dt);
        let [vx, vy] = state.velocity;
        let [rx, ry] = raw;
        state.velocity = [vx + (rx - vx) * follow, vy + (ry - vy) * follow];
        let speed = state.velocity[0].hypot(state.velocity[1]);
        let weight = smoothing(self.settings.cutoff(speed), dt);
        state.filtered = state.filtered.lerp(to, weight);
        state.brush = lazy_string(state.brush, state.filtered, self.settings.lazy);
        sample.at(state.brush)
    }

    /// Ends the stroke: returns its last raw sample, so the stroke can be completed to where the
    /// pen lifted, and readies the path for the next stroke. `None` if no sample was pushed.
    pub fn end(&mut self) -> Option<Sample> {
        self.state.take().map(|state| state.last)
    }
}

/// Where the brush point goes when the pen's filtered point is at `pen`.
///
/// It stays put while the string of `length` is slack, and is pulled along the string to
/// `length` from the pen once it is taut. A zero length makes the brush the pen.
fn lazy_string(brush: ScreenPoint, pen: ScreenPoint, length: f64) -> ScreenPoint {
    let distance = brush.distance(pen);
    if distance <= length {
        return if length > 0.0 { brush } else { pen };
    }
    pen.lerp(brush, length / distance)
}
