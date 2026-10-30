//! The scripted camera path (`--script`): the measured run of the renderer.
//!
//! A path is a start pose and a list of keyframes ([`Keyframe`]): each names its segment of the
//! run, lasts a fixed time, and ends at its pose, moving there from the one before as its easing
//! says. The camera is a pure function of time ([`Script::at`]), and the frame loop drives it with
//! its own clock (each frame's stamp, [`crate::frame::FrameUpdate::time`]), so a run moves the
//! same way however fast its frames come: a slow frame sees the camera further on, as a person
//! orbiting at that speed would.
//!
//! The benchmark path ([`Script::benchmark`]) looks at the project from three units away (a unit
//! sphere, as the benchmark project is, fills two thirds of the view's height), 20° above its
//! equator, and runs five segments:
//!
//! 1. **open**, 3 s still: from the project opening until every visible tile is correct (the spec's
//!    open-to-correct, ≤ 1 s provisional on desktop);
//! 2. **orbit**, 8 s: a full turn about the vertical axis at that distance (orbit frame times, and
//!    the tiles over budget during it, 0 on desktop);
//! 3. **zoom**, 4 s: in from the whole object to 0.1 above its surface, at a steady rate in
//!    log-distance, easing in and out;
//! 4. **flick**, 0.15 s: half a turn at that distance, to the far side, never resident that close;
//! 5. **settle**, 3 s still: the spin catch-up, until no tile is wanted or requested (≤ 100 ms).
//!
//! `--script-speed` runs the same path faster (or slower), every duration divided by it: the
//! gate's test runs it on lavapipe in a few seconds.

use std::f64::consts::{PI, TAU};
use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use thiserror::Error;

use crate::camera::Camera;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// Where a scripted camera is: turned `yaw` radians about the vertical axis (from +z towards
/// +x), `pitch` radians above the horizon, `distance` from the origin it looks at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pose {
    /// Radians about the vertical axis, not wrapped: a whole turn ends at τ.
    pub(crate) yaw: f64,
    /// Radians above the horizon.
    pub(crate) pitch: f64,
    /// From the origin.
    pub(crate) distance: f64,
}

impl Pose {
    /// The camera at this pose.
    pub(crate) fn camera(self) -> Camera {
        Camera::orbiting(self.yaw, self.pitch, self.distance)
    }

    /// The pose a fraction `s` (0 to 1) of the way from this one to `to`: yaw and pitch linear,
    /// distance geometric (a steady rate in log-distance, so a zoom looks as fast near as far).
    fn toward(self, to: Self, s: f64) -> Self {
        let lerp = |a: f64, b: f64| a + (b - a) * s;
        Self {
            yaw: lerp(self.yaw, to.yaw),
            pitch: lerp(self.pitch, to.pitch),
            distance: self.distance * (to.distance / self.distance).powf(s),
        }
    }
}

/// How a keyframe moves from the pose before it to its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ease {
    /// At a steady rate: an orbit, a flick.
    Linear,
    /// Easing in and out (smoothstep): a zoom.
    Smooth,
}

impl Ease {
    /// The fraction of the way along at `u` of the time (both 0 to 1).
    fn apply(self, u: f64) -> f64 {
        match self {
            Self::Linear => u,
            Self::Smooth => u * u * (3.0 - 2.0 * u),
        }
    }
}

/// The segments of the benchmark run, as records name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Segment {
    /// Still, from the project opening.
    Open,
    /// A whole turn about the object.
    Orbit,
    /// From the whole object to close up.
    Zoom,
    /// Half a turn, fast.
    Flick,
    /// Still, after the flick.
    Settle,
}

impl Segment {
    /// The segment's name in records.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Orbit => "orbit",
            Self::Zoom => "zoom",
            Self::Flick => "flick",
            Self::Settle => "settle",
        }
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One step of a path: its segment, how long it lasts at speed 1, where it ends and how it gets
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Keyframe {
    /// The segment it is.
    pub(crate) segment: Segment,
    /// How long it lasts at speed 1, in seconds.
    pub(crate) seconds: f64,
    /// The pose it ends at.
    pub(crate) to: Pose,
    /// How it moves there.
    pub(crate) ease: Ease,
}

/// The benchmark path's pitch: 20° above the equator.
const PITCH: f64 = 20.0 * PI / 180.0;

/// The benchmark path's start: the whole object, three units away.
const START: Pose = Pose {
    yaw: 0.0,
    pitch: PITCH,
    distance: 3.0,
};

/// After the orbit: a whole turn on.
const ORBITED: Pose = Pose { yaw: TAU, ..START };

/// After the zoom: 0.1 above a unit sphere's surface.
const CLOSE: Pose = Pose {
    distance: 1.1,
    ..ORBITED
};

/// After the flick: half a turn on, close.
const FLICKED: Pose = Pose {
    yaw: TAU + PI,
    ..CLOSE
};

/// The benchmark path ([`Script::benchmark`]; the module notes say why each segment).
const BENCHMARK: [Keyframe; 5] = [
    Keyframe {
        segment: Segment::Open,
        seconds: 3.0,
        to: START,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Orbit,
        seconds: 8.0,
        to: ORBITED,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Zoom,
        seconds: 4.0,
        to: CLOSE,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::Flick,
        seconds: 0.15,
        to: FLICKED,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Settle,
        seconds: 3.0,
        to: FLICKED,
        ease: Ease::Linear,
    },
];

/// How fast a script runs: every duration is divided by it. A positive, finite number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Speed(f64);

/// Why a text isn't a [`Speed`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0:?} isn't a speed: a positive number such as 1 or 2.5")]
pub(crate) struct BadSpeed(String);

impl Speed {
    /// Real time.
    pub(crate) const ONE: Self = Self(1.0);

    /// The speed as a number.
    pub(crate) const fn get(self) -> f64 {
        self.0
    }
}

impl FromStr for Speed {
    type Err = BadSpeed;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.parse::<f64>() {
            Ok(speed) if speed.is_finite() && speed > 0.0 => Ok(Self(speed)),
            _ => Err(BadSpeed(text.to_owned())),
        }
    }
}

/// Where a script's camera is at a moment of its run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Moment {
    /// The camera's pose.
    pub(crate) pose: Pose,
    /// The segment running (the last, once the script is over).
    pub(crate) segment: Segment,
    /// Whether the script is over: the moment is at or past its end.
    pub(crate) over: bool,
}

/// A camera path: a start pose and its keyframes, run at a speed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Script {
    /// Where the camera starts.
    start: Pose,
    /// The keyframes, in order; never empty.
    keyframes: Vec<Keyframe>,
    /// How fast it runs.
    speed: Speed,
}

impl Script {
    /// The benchmark path (the module notes), run at `speed`.
    pub(crate) fn benchmark(speed: Speed) -> Self {
        Self {
            start: START,
            keyframes: BENCHMARK.to_vec(),
            speed,
        }
    }

    /// How fast it runs.
    pub(crate) const fn speed(&self) -> Speed {
        self.speed
    }

    /// How long keyframe `keyframe` lasts at this speed.
    fn length(&self, keyframe: &Keyframe) -> Duration {
        Duration::from_secs_f64(keyframe.seconds / self.speed.get())
    }

    /// Each segment's span of the run: its name, when it starts and when it ends, since the run
    /// began.
    pub(crate) fn segments(&self) -> Vec<(Segment, Duration, Duration)> {
        let mut start = Duration::ZERO;
        let spans = self.keyframes.iter().map(|keyframe| {
            let end = start + self.length(keyframe);
            let span = (keyframe.segment, start, end);
            start = end;
            span
        });
        spans.collect()
    }

    /// How long the whole run lasts.
    #[cfg(test)]
    pub(crate) fn duration(&self) -> Duration {
        self.keyframes.iter().map(|k| self.length(k)).sum()
    }

    /// Where the camera is `time` into the run.
    pub(crate) fn at(&self, time: Duration) -> Moment {
        let mut from = self.start;
        let mut start = Duration::ZERO;
        for keyframe in &self.keyframes {
            let length = self.length(keyframe);
            let end = start + length;
            if time < end {
                let into = time.saturating_sub(start).as_secs_f64();
                let u = (into / length.as_secs_f64()).clamp(0.0, 1.0);
                return Moment {
                    pose: from.toward(keyframe.to, keyframe.ease.apply(u)),
                    segment: keyframe.segment,
                    over: false,
                };
            }
            from = keyframe.to;
            start = end;
        }
        Moment {
            pose: from,
            segment: self.keyframes.last().map_or(Segment::Settle, |k| k.segment),
            over: true,
        }
    }
}
