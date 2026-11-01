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
//!
//! # The strokes path
//!
//! `--script-path strokes` ([`Script::strokes`]) draws two pen strokes on the sphere, for the
//! pen's latency (sample to present submitted) and a stroke seen from a distance:
//!
//! 1. **open**, 3 s still, three units away, as the benchmark path starts;
//! 2. **approach**, 1.5 s: in to 0.3 above the surface, easing in and out;
//! 3. **`stroke_near`**, 2 s still: the pen comes into range, then draws a stroke 1.2 s long across
//!    the point facing the camera, a 40-pixel brush;
//! 4. **retreat**, 1.5 s: out to three units again, a quarter turn round, to a part of the surface
//!    not yet drawn close, easing in and out;
//! 5. **`stroke_far`**, 2 s still: the same stroke seen from a distance, its tiles drawn at display
//!    levels (the stand-ins' case), a 6-pixel brush: about the near stroke's world size in a 1280 ×
//!    800 view, below the footprint limit F on the benchmark project;
//! 6. **settle**, 2 s still: the edits' refresh lands.
//!
//! The pen's samples come at the Kamvas's measured 173 Hz, in real time whatever the speed (a
//! faster run has fewer of them): hovering over the stroke's start from 0.3 s before it touches
//! (so the brush warms the surface under it), then down, along the path with the pressure rising
//! and falling, up, and out of range 0.1 s later. Each sample's point lies on the unit sphere, as
//! the benchmark project is; the frame loop projects it through the camera at the sample's time
//! ([`Script::pen`]).

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use input::{Phase, Unit};
use render::camera::Vec3 as WorldPoint;
use thiserror::Error;

use crate::camera::{Camera, Vec3};

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
    /// Still, after the flick (or the far stroke).
    Settle,
    /// In close, to draw a stroke near the camera.
    Approach,
    /// Still, close, while a stroke is drawn.
    StrokeNear,
    /// Out again, to draw a stroke from a distance.
    Retreat,
    /// Still, far, while a stroke is drawn.
    StrokeFar,
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
            Self::Approach => "approach",
            Self::StrokeNear => "stroke_near",
            Self::Retreat => "retreat",
            Self::StrokeFar => "stroke_far",
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

/// The strokes path's close pose: 0.3 above a unit sphere's surface.
const NEAR: Pose = Pose {
    distance: 1.3,
    ..START
};

/// The strokes path's far pose: three units away again, a quarter turn round.
const FAR: Pose = Pose {
    yaw: FRAC_PI_2,
    ..START
};

/// The strokes path's camera ([`Script::strokes`]; the module notes say why each segment).
const STROKES: [Keyframe; 6] = [
    Keyframe {
        segment: Segment::Open,
        seconds: 3.0,
        to: START,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Approach,
        seconds: 1.5,
        to: NEAR,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::StrokeNear,
        seconds: 2.0,
        to: NEAR,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Retreat,
        seconds: 1.5,
        to: FAR,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::StrokeFar,
        seconds: 2.0,
        to: FAR,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Settle,
        seconds: 2.0,
        to: FAR,
        ease: Ease::Linear,
    },
];

/// The strokes path's pen strokes, one in each still stroke segment (which start at 4.5 s and
/// 8 s).
const PEN_STROKES: [PenStroke; 2] = [
    PenStroke {
        hover: 4.7,
        down: 5.0,
        up: 6.2,
        radius: 40.0,
        span: 0.25,
    },
    PenStroke {
        hover: 8.2,
        down: 8.5,
        up: 9.7,
        radius: 6.0,
        span: 0.6,
    },
];

/// How often a scripted pen reports: the Kamvas's measured rate, in hertz.
pub(crate) const PEN_RATE: f64 = 173.0;

/// How long after lifting a scripted pen leaves the tablet's range, in seconds.
const LEAVE_AFTER: f64 = 0.1;

/// One scripted pen stroke: when the pen hovers, touches and lifts, the brush's size, and how
/// far the stroke runs.
///
/// The stroke is drawn across the point of the unit sphere facing the camera (the camera
/// looks at the sphere's centre), from left to right as the view sees it, bowing up by a
/// fifth of its span at its middle ([`PenStroke::point`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PenStroke {
    /// When the pen comes into range over the stroke's start: seconds into the run at speed 1.
    hover: f64,
    /// When the tip touches, as `hover`.
    down: f64,
    /// When it lifts, as `hover`; after `down`.
    up: f64,
    /// The brush's radius for the stroke, in physical pixels, set as the pen comes into range.
    radius: f32,
    /// How far the stroke runs across the view, in world units, at the facing point.
    span: f64,
}

impl PenStroke {
    /// The stroke's point a fraction `along` (0 to 1) of the way, seen by `camera`: on the unit
    /// sphere, about the point facing the camera.
    fn point(&self, camera: &Camera, along: f64) -> WorldPoint {
        let eye = camera.eye();
        let facing = eye * eye.dot(eye).sqrt().recip();
        let right = camera.axis(Vec3::new(1.0, 0.0, 0.0));
        let up = camera.axis(Vec3::new(0.0, 1.0, 0.0));
        let bow = 0.2 * (PI * along).sin();
        let p = facing + right * ((along - 0.5) * self.span) + up * (bow * self.span);
        let length = p.dot(p).sqrt();
        WorldPoint::new(p.x / length, p.y / length, p.z / length)
    }

    /// The tip's pressure a fraction `along` (0 to 1) of the way: rising from 0.35 to 0.8 at the
    /// middle and falling back.
    fn pressure(along: f64) -> Unit {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a pressure in 0.35..=0.8: f32 holds it"
        )]
        let pressure = (0.35 + 0.45 * (PI * along).sin()) as f32;
        Unit::try_from(pressure).unwrap_or(Unit::ONE)
    }
}

/// One sample of a scripted pen: when, in what phase, where on the sphere, and how hard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScriptPen {
    /// When the pen took it, since the run began.
    pub(crate) time: Duration,
    /// Where it is in the stroke's lifecycle.
    pub(crate) phase: Phase,
    /// The camera at its time, which `point` is seen through.
    pub(crate) camera: Camera,
    /// The point of the surface the pen is over.
    pub(crate) point: WorldPoint,
    /// Tip pressure; zero out of contact.
    pub(crate) pressure: Unit,
    /// The brush's radius for the stroke, in physical pixels: on the stroke's first sample only.
    pub(crate) radius: Option<f32>,
}

/// Which camera path a script runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ScriptPath {
    /// The renderer's measured run ([`Script::benchmark`]).
    #[default]
    Benchmark,
    /// Two pen strokes, near and far ([`Script::strokes`]).
    Strokes,
}

/// Why a text isn't a [`ScriptPath`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0:?} isn't a script path: benchmark or strokes")]
pub(crate) struct BadPath(String);

impl FromStr for ScriptPath {
    type Err = BadPath;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "benchmark" => Ok(Self::Benchmark),
            "strokes" => Ok(Self::Strokes),
            _ => Err(BadPath(text.to_owned())),
        }
    }
}

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
    /// The pen's strokes, in order; none on the benchmark path.
    strokes: Vec<PenStroke>,
    /// Every sample of the pen's strokes, in order of time: made once, as the script is.
    pen: Vec<ScriptPen>,
}

impl Script {
    /// The path `path` names, run at `speed`.
    pub(crate) fn new(path: ScriptPath, speed: Speed) -> Self {
        match path {
            ScriptPath::Benchmark => Self::benchmark(speed),
            ScriptPath::Strokes => Self::strokes(speed),
        }
    }

    /// The benchmark path (the module notes), run at `speed`.
    pub(crate) fn benchmark(speed: Speed) -> Self {
        Self {
            start: START,
            keyframes: BENCHMARK.to_vec(),
            speed,
            strokes: Vec::new(),
            pen: Vec::new(),
        }
    }

    /// The strokes path (the module notes), run at `speed`.
    pub(crate) fn strokes(speed: Speed) -> Self {
        let mut script = Self {
            start: START,
            keyframes: STROKES.to_vec(),
            speed,
            strokes: PEN_STROKES.to_vec(),
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The scripted pen's samples taken after `from` and up to `to` into the run, in order (see
    /// the module notes).
    ///
    /// Samples come every 1/[`PEN_RATE`] s from each stroke's hover, in real time; the first
    /// is the pen entering range (with the stroke's brush radius), those before the stroke's
    /// touch hover, the first at or after it is the `Down`, then `Drag`s, then an `Up` at the
    /// lift, and a `Leave` [`LEAVE_AFTER`] later.
    ///
    /// The samples are made once, as the script is; this only finds them.
    pub(crate) fn pen(&self, from: Duration, to: Duration) -> &[ScriptPen] {
        let start = self.pen.partition_point(|sample| sample.time <= from);
        let end = self.pen.partition_point(|sample| sample.time <= to);
        self.pen.get(start..end.max(start)).unwrap_or_default()
    }

    /// Every sample of the pen's strokes, in order of time (the strokes are in order, and don't
    /// overlap).
    fn every_pen_sample(&self) -> Vec<ScriptPen> {
        let samples = self.strokes.iter().flat_map(|stroke| {
            self.stroke_samples(stroke)
                .into_iter()
                .map(move |(time, phase)| self.sample(stroke, time, phase))
        });
        samples.collect()
    }

    /// Every sample of `stroke` at this speed, from its hover to its leave: its time in seconds
    /// into the run, and its phase.
    fn stroke_samples(&self, stroke: &PenStroke) -> Vec<(f64, Phase)> {
        let speed = self.speed.get();
        let (hover, down, up) = (stroke.hover / speed, stroke.down / speed, stroke.up / speed);
        let period = PEN_RATE.recip();
        let mut touched = false;
        let regular = (0_u32..)
            .map(|k| (k, hover + f64::from(k) * period))
            .take_while(|&(_, time)| time < up)
            .map(|(k, time)| {
                let phase = if time < down {
                    if k == 0 { Phase::Enter } else { Phase::Hover }
                } else if std::mem::replace(&mut touched, true) {
                    Phase::Drag
                } else {
                    Phase::Down
                };
                (time, phase)
            });
        let ends = [(up, Phase::Up), (up + LEAVE_AFTER, Phase::Leave)];
        let samples = regular.chain(ends);
        samples.collect()
    }

    /// `stroke`'s sample at `time` seconds into the run, in `phase` (one of
    /// [`Script::stroke_samples`]).
    fn sample(&self, stroke: &PenStroke, time: f64, phase: Phase) -> ScriptPen {
        let speed = self.speed.get();
        let (down, up) = (stroke.down / speed, stroke.up / speed);
        let along = ((time - down) / (up - down)).clamp(0.0, 1.0);
        let at = Duration::from_secs_f64(time);
        let camera = self.at(at).pose.camera();
        ScriptPen {
            time: at,
            phase,
            camera,
            point: stroke.point(&camera, along),
            pressure: if matches!(phase, Phase::Down | Phase::Drag) {
                PenStroke::pressure(along)
            } else {
                Unit::ZERO
            },
            radius: (phase == Phase::Enter).then_some(stroke.radius),
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
