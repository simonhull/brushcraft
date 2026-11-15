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
//! `--script-path strokes` ([`Script::strokes`]) draws five pen strokes on the sphere, for the
//! pen's latency (sample to present submitted), a stroke seen from a distance, a brush that
//! keeps up with a fast pen, lightly and pressed hard, and the M1 spec's big brush of a million
//! vertices with its catch-up:
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
//! 6. **turn**, 1 s: a quarter turn more at that distance, to surface not yet drawn, easing in and
//!    out;
//! 7. **`stroke_long`**, 2 s still: a fast, long stroke at the default 40-pixel brush across the
//!    whole sphere as the view sees it (from about 39° left of the point facing the camera to 39°
//!    right, two thirds of the sphere's width on the screen), in 0.8 s: the brush must keep up, its
//!    queue of dabs never growing over the stroke and drained within two frames of pen-up. At the
//!    performance suite's 4K a dab touches about 60 finest tiles of the benchmark project, below F
//!    (in a 1280 × 800 view, about F);
//! 8. **turn**, 1 s: a quarter turn more, to surface not yet drawn;
//! 9. **`stroke_heavy`**, 2 s still: the long stroke again, pressed at full pressure all the way
//!    (the others rise from 0.35 to 0.8 and fall back): bigger, stronger dabs, a quarter wider than
//!    the long stroke's widest, so more tiles a dab; the case a user pressing hard draws;
//! 10. **turn**, 1 s: a quarter turn more, back round to the start's side, and down to 20° below
//!     the equator, to surface not yet drawn;
//! 11. **`stroke_huge`**, 8 s still: a stroke 2 s long at full pressure with a brush covering
//!     [`HUGE_VERTICES`] (a million) finest vertices about its start ([`BrushSize::Vertices`],
//!     sized on the project as the pen comes into range), over 0.6 of the view at the point facing
//!     the camera: far above the footprint limit F, so every dab previews and catches up. The 5.5 s
//!     after pen-up leave room to time the catch-up until the finest level is exact (the spec's
//!     provisional target is 2 s after a 2 s stroke);
//! 12. **settle**, 2 s still: the edits' refresh lands.
//!
//! The pen's samples come at the Kamvas's measured 173 Hz, in real time whatever the speed (a
//! faster run has fewer of them): hovering over the stroke's start from 0.3 s before it touches
//! (so the brush warms the surface under it), then down, along the path with the pressure rising
//! and falling (or held at full, or rising all the way, [`Pressure`]), up, and out of range 0.1 s
//! later. Each sample's point lies on the unit sphere, as the benchmark project is; the frame
//! loop projects it through the camera at the sample's time ([`Script::pen`]).
//!
//! # The spiral paths
//!
//! `--script-path spiral` ([`Script::spiral`]) draws one stroke 25 s long at the default brush,
//! pressed hard, spiralling out across the sphere onto fresh ground: what a stroke's length costs,
//! read against the dabs placed so far. `--script-path spiral-light` ([`Script::spiral_light`])
//! touches down lightly and presses hard over its first 2 s, as a pen does, so its dabs start a
//! fraction of the size of the rest.
//!
//! # The over path
//!
//! `--script-path over` ([`Script::over`]) draws strokes over strokes still landing (the
//! stroke-while-landing spec): six pairs, each on ground of its own, an earlier stroke and then a
//! later one drawn along it. Two weights of earlier stroke ([`Weight`]), each met three ways
//! ([`Before`]):
//!
//! - **ordinary**: a brush of [`ORDINARY_VERTICES`] finest vertices (the performance suite's brush
//!   footprint) run fast across the sphere and back, [`ORDINARY_PASSES`] passes, the later stroke
//!   drawn along its last pass, the part that lands last: some tens of dabs not landed on each tile
//!   under it. A smaller brush lands within its own frames on a fast machine (the default 40 pixels
//!   left none at all on one ground and hundreds on another), so it gives no stroke still landing
//!   to draw over;
//! - **heavy**: a brush of [`HEAVY_VERTICES`] scribbled to and fro over one span, [`HEAVY_PASSES`]
//!   times, the later stroke drawn from its middle: hundreds of dabs not landed, piled on the same
//!   tiles;
//! - **cold**: the pen leaves the tablet's range as it lifts and comes down [`OVER_GAP`] later with
//!   no hover before, so the later stroke's `Down` makes every future it reads;
//! - **warmed**: it hovers where it will come down first ([`Weight::hover`]), the earlier stroke
//!   landing on meanwhile, so the futures and the seams are made ahead;
//! - **landed**: it comes down once the earlier stroke has had [`Weight::landing`] to land, after a
//!   hover as the strokes path's: the same `Down` on a stroke alone, for reference.
//!
//! After **open** (3 s still), each pair is a **turn** (1 s, to its ground: a ninth of a turn
//! round from the last, 35° above the equator, on it and 35° below it in turn), **`over_first`**
//! (still: the earlier stroke, from its hover to [`OVER_EDGE`] past its lift), and the later
//! stroke's segment (`over_ordinary_cold` to `over_heavy_landed`, still: from there until both
//! strokes have had time to land, [`Weight::landing`] after the later stroke's lift). The later
//! stroke's brush is the default 40 pixels, drawn for [`OVER_DRAWN`] at 0.8 of full pressure,
//! and every stroke's pen leaves the tablet's range as it lifts.
//!
//! Then a **chain** on a seventh ground: an ordinary stroke, a second at once along it (both in
//! `over_first`), and a third at once over both (**`over_chain`**): three strokes landing
//! together. The run ends with 2 s of **settle**.
//!
//! # The sweep path
//!
//! `--script-path sweep` ([`Script::sweep`]) draws one long, sweeping stroke for looking at, its
//! pressure rising evenly from none to full: how a stroke looks as the pen presses harder, seen at
//! a slant, where a ridge that breaks into separate leaning spikes shows clearest. With
//! `--screenshot` it leaves the result in a PNG.
//!
//! 1. **open**, 3 s still, three units away, as the benchmark path starts;
//! 2. **approach**, 1.5 s: in to 1.3 from the centre, the camera slid 0.6 down its own up axis, so
//!    it looks past the point facing it at surface below, easing in and out;
//! 3. **sweep**, 3 s still: the pen comes into range, then draws a stroke 1.6 s long at the default
//!    40-pixel brush, left to right along a circle 12° of the sphere about the point facing the
//!    camera, 60° each side of the circle's lowest point as the view sees it: a smile across the
//!    middle three fifths of the view's width, about 0.5 from the eye, every point of it seen at
//!    about 37° between the view's ray and the surface's normal;
//! 4. **settle**, 2 s still: the edits' refresh lands, and the run ends.

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

/// Where a scripted camera is: turned about the origin, at a distance, perhaps slid down.
///
/// It is turned `yaw` radians about the vertical axis (from +z towards +x), `pitch` radians above
/// the horizon, `distance` from the origin it looks at, then slid `lowered` down its own up axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pose {
    /// Radians about the vertical axis, not wrapped: a whole turn ends at τ.
    pub(crate) yaw: f64,
    /// Radians above the horizon.
    pub(crate) pitch: f64,
    /// From the origin.
    pub(crate) distance: f64,
    /// How far the camera then slides down its own up axis, without turning, in world units
    /// ([`Camera::slide`]): 0 looks at the origin; more puts surface below the point facing the
    /// camera at the view's centre, seen at a slant.
    pub(crate) lowered: f64,
}

impl Pose {
    /// The camera at this pose.
    pub(crate) fn camera(self) -> Camera {
        let mut camera = Camera::orbiting(self.yaw, self.pitch, self.distance);
        camera.slide(0.0, -self.lowered);
        camera
    }

    /// The pose a fraction `s` (0 to 1) of the way from this one to `to`: yaw, pitch and the
    /// lowering linear, distance geometric (a steady rate in log-distance, so a zoom looks as fast
    /// near as far).
    fn toward(self, to: Self, s: f64) -> Self {
        let lerp = |a: f64, b: f64| a + (b - a) * s;
        Self {
            yaw: lerp(self.yaw, to.yaw),
            pitch: lerp(self.pitch, to.pitch),
            distance: self.distance * (to.distance / self.distance).powf(s),
            lowered: lerp(self.lowered, to.lowered),
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
    /// Round the object at the same distance, to surface not yet drawn.
    Turn,
    /// Still, while a fast, long stroke is drawn across the whole object.
    StrokeLong,
    /// The spiral path's one long stroke ([`Script::spiral`]).
    StrokeSpiral,
    /// Still, while the fast, long stroke is drawn again at full pressure.
    StrokeHeavy,
    /// Still, while a stroke of a million-vertex brush is drawn and caught up.
    StrokeHuge,
    /// Still, while one long stroke is drawn at a slant, its pressure rising from none to full.
    Sweep,
    /// Still, while the earlier stroke of a pair is drawn ([`Script::over`]).
    OverFirst,
    /// Still, while the later stroke of a pair is drawn over the earlier one, of this weight and
    /// met this way, and both land ([`Script::over`]).
    Over(Weight, Before),
    /// Still, while a third stroke is drawn over two still landing, and all three land
    /// ([`Script::over`]).
    OverChain,
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
            Self::Turn => "turn",
            Self::StrokeLong => "stroke_long",
            Self::StrokeHeavy => "stroke_heavy",
            Self::StrokeHuge => "stroke_huge",
            Self::Sweep => "sweep",
            Self::StrokeSpiral => "stroke_spiral",
            Self::OverFirst => "over_first",
            Self::OverChain => "over_chain",
            Self::Over(Weight::Ordinary, Before::Cold) => "over_ordinary_cold",
            Self::Over(Weight::Ordinary, Before::Warmed) => "over_ordinary_warmed",
            Self::Over(Weight::Ordinary, Before::Landed) => "over_ordinary_landed",
            Self::Over(Weight::Heavy, Before::Cold) => "over_heavy_cold",
            Self::Over(Weight::Heavy, Before::Warmed) => "over_heavy_warmed",
            Self::Over(Weight::Heavy, Before::Landed) => "over_heavy_landed",
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
    lowered: 0.0,
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

/// The strokes path's pose for the long stroke: three units away, half a turn round from the
/// start.
const ROUND: Pose = Pose { yaw: PI, ..START };

/// The strokes path's last pose, for the heavy stroke: three quarters of a turn round from the
/// start.
const LAST: Pose = Pose {
    yaw: PI + FRAC_PI_2,
    ..START
};

/// The strokes path's pose for the huge stroke: three units away, back round at the start's
/// yaw (a whole turn on), 20° below the equator, where none of the other strokes reach.
const BELOW: Pose = Pose {
    yaw: TAU,
    pitch: -PITCH,
    ..START
};

/// The strokes path's camera ([`Script::strokes`]; the module notes say why each segment).
const STROKES: [Keyframe; 12] = [
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
        segment: Segment::Turn,
        seconds: 1.0,
        to: ROUND,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::StrokeLong,
        seconds: 2.0,
        to: ROUND,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Turn,
        seconds: 1.0,
        to: LAST,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::StrokeHeavy,
        seconds: 2.0,
        to: LAST,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Turn,
        seconds: 1.0,
        to: BELOW,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::StrokeHuge,
        seconds: 8.0,
        to: BELOW,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Settle,
        seconds: 2.0,
        to: BELOW,
        ease: Ease::Linear,
    },
];

/// The finest vertices the huge stroke's brush covers about its start: the M1 spec's "brush of
/// 1M vertices".
///
/// Sized by its vertices, the brush does the same work on any project, whatever its density
/// there: about 250 finest level-9 tiles edited a dab (256 on the medium project).
pub(crate) const HUGE_VERTICES: u32 = 1_000_000;

/// The strokes path's pen strokes, one in each still stroke segment (which start at 4.5 s, 8 s,
/// 11 s, 14 s and 17 s).
const PEN_STROKES: [PenStroke; 5] = [
    PenStroke {
        hover: 4.7,
        down: 5.0,
        up: 6.2,
        size: BrushSize::Pixels(40.0),
        shape: Shape::Across { span: 0.25 },
        pressure: Pressure::Swelling,
        part: WHOLE,
        leave: LEAVE_AFTER,
    },
    PenStroke {
        hover: 8.2,
        down: 8.5,
        up: 9.7,
        size: BrushSize::Pixels(6.0),
        shape: Shape::Across { span: 0.6 },
        pressure: Pressure::Swelling,
        part: WHOLE,
        leave: LEAVE_AFTER,
    },
    PenStroke {
        hover: 11.2,
        down: 11.5,
        up: 12.3,
        size: BrushSize::Pixels(40.0),
        shape: Shape::Across { span: 1.6 },
        pressure: Pressure::Swelling,
        part: WHOLE,
        leave: LEAVE_AFTER,
    },
    PenStroke {
        hover: 14.2,
        down: 14.5,
        up: 15.3,
        size: BrushSize::Pixels(40.0),
        shape: Shape::Across { span: 1.6 },
        pressure: Pressure::Full,
        part: WHOLE,
        leave: LEAVE_AFTER,
    },
    PenStroke {
        hover: 17.2,
        down: 17.5,
        up: 19.5,
        size: BrushSize::Vertices(HUGE_VERTICES),
        shape: Shape::Across { span: 0.6 },
        pressure: Pressure::Full,
        part: WHOLE,
        leave: LEAVE_AFTER,
    },
];

/// The sweep path's pose: 1.3 from the centre, slid 0.6 down, so the view's centre falls on
/// surface below the point facing the camera, seen at a slant.
const SLANTED: Pose = Pose {
    distance: 1.3,
    lowered: 0.6,
    ..START
};

/// The sweep path's camera ([`Script::sweep`]; the module notes say why each segment).
const SWEEP: [Keyframe; 4] = [
    Keyframe {
        segment: Segment::Open,
        seconds: 3.0,
        to: START,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Approach,
        seconds: 1.5,
        to: SLANTED,
        ease: Ease::Smooth,
    },
    Keyframe {
        segment: Segment::Sweep,
        seconds: 3.0,
        to: SLANTED,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Settle,
        seconds: 2.0,
        to: SLANTED,
        ease: Ease::Linear,
    },
];

/// The sweep path's one stroke, in its still segment (which starts at 4.5 s): 12° of the sphere
/// about the point facing the camera, seen there at about 37°, 60° each side of the circle's
/// lowest point.
const SWEEP_STROKE: PenStroke = PenStroke {
    hover: 4.7,
    down: 5.0,
    up: 6.6,
    size: BrushSize::Pixels(40.0),
    shape: Shape::Arc {
        angle: 12.0 * PI / 180.0,
        sweep: 60.0 * PI / 180.0,
    },
    pressure: Pressure::Rising,
    part: WHOLE,
    leave: LEAVE_AFTER,
};

/// The spiral path's camera ([`Script::spiral`]): the sphere whole in view, the stroke held still
/// under it long enough to feel what its length costs.
const SPIRAL: [Keyframe; 3] = [
    Keyframe {
        segment: Segment::Open,
        seconds: 3.0,
        to: START,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::StrokeSpiral,
        seconds: 27.0,
        to: START,
        ease: Ease::Linear,
    },
    Keyframe {
        segment: Segment::Settle,
        seconds: 20.0,
        to: START,
        ease: Ease::Linear,
    },
];

/// The spiral path's one stroke: 25 s at the default brush, pressed hard, out from near the
/// facing point in five turns, onto fresh ground all the way.
const SPIRAL_STROKE: PenStroke = PenStroke {
    hover: 3.2,
    down: 3.5,
    up: 28.5,
    size: BrushSize::Pixels(40.0),
    shape: Shape::Spiral {
        from: 6.0 * PI / 180.0,
        to: 60.0 * PI / 180.0,
        turns: 5.0,
    },
    pressure: Pressure::Full,
    part: WHOLE,
    leave: LEAVE_AFTER,
};

/// The light spiral's stroke: [`SPIRAL_STROKE`], touching down lightly.
const SPIRAL_LIGHT_STROKE: PenStroke = PenStroke {
    pressure: Pressure::Touchdown,
    ..SPIRAL_STROKE
};

/// How much of an earlier stroke is still to land under a later one drawn along it, on the over
/// path ([`Script::over`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Weight {
    /// Some tens of dabs on a tile: the last pass of a quick scribble across the sphere with a
    /// brush of half the footprint limit.
    Ordinary,
    /// Hundreds of dabs piled on the same tiles: a scribble to and fro with a brush near the
    /// footprint limit.
    Heavy,
}

/// The finest vertices the over path's ordinary earlier stroke's brush covers: the performance
/// suite's brush footprint, a dab reaching some 25 to 35 finest level-9 tiles, half the footprint
/// limit F.
pub(crate) const ORDINARY_VERTICES: u32 = 50_000;

/// The finest vertices the over path's heavy earlier stroke's brush covers: a dab reaches some
/// 60 to 75 finest level-9 tiles, about the footprint limit F.
pub(crate) const HEAVY_VERTICES: u32 = 250_000;

/// How many times the over path's ordinary earlier stroke runs across its span.
pub(crate) const ORDINARY_PASSES: u32 = 3;

/// How many times the over path's heavy earlier stroke runs across its span.
pub(crate) const HEAVY_PASSES: u32 = 8;

impl Weight {
    /// Both weights, in the order the over path draws them.
    const ALL: [Self; 2] = [Self::Ordinary, Self::Heavy];

    /// The earlier stroke's brush.
    const fn brush(self) -> BrushSize {
        match self {
            Self::Ordinary => BrushSize::Vertices(ORDINARY_VERTICES),
            Self::Heavy => BrushSize::Vertices(HEAVY_VERTICES),
        }
    }

    /// Where the earlier stroke runs.
    const fn shape(self) -> Shape {
        match self {
            Self::Ordinary => Shape::ToAndFro {
                span: 1.2,
                passes: ORDINARY_PASSES,
            },
            Self::Heavy => Shape::ToAndFro {
                span: 0.4,
                passes: HEAVY_PASSES,
            },
        }
    }

    /// How long the earlier stroke is drawn for, in seconds at speed 1.
    const fn drawn(self) -> f64 {
        match self {
            Self::Ordinary => 0.9,
            Self::Heavy => 2.0,
        }
    }

    /// The part of the earlier stroke's shape the later stroke is drawn along ([`PenStroke`]'s
    /// `part`): the middle of the ordinary scribble's last pass, whose dabs land last; a quarter
    /// of one pass of the heavy scribble, from its middle.
    const fn over(self) -> (f64, f64) {
        match self {
            Self::Ordinary => {
                let passes = ORDINARY_PASSES as f64;
                ((passes - 0.6) / passes, (passes - 0.3) / passes)
            }
            Self::Heavy => {
                let passes = HEAVY_PASSES as f64;
                (0.5 / passes, 0.75 / passes)
            }
        }
    }

    /// How long the pen hovers where it will come down on the warmed pair, in seconds at speed 1:
    /// long enough for the futures and the seams there to be made ahead, in the time the
    /// earlier stroke's landing leaves, and short enough that it still lands as the pen comes
    /// down.
    const fn hover(self) -> f64 {
        match self {
            Self::Ordinary => 0.2,
            Self::Heavy => 0.6,
        }
    }

    /// How long both strokes of a pair are given to land, in seconds at speed 1: the wait before
    /// the reference `Down` ([`Before::Landed`]), and after each later stroke's lift, before the
    /// next pair begins on a brush with nothing under way.
    ///
    /// Several times what the landing took on the machine the path was written on (the ordinary
    /// strokes up to a second after the later one's lift, the heavy ones 2 to 5 s), and for the
    /// ordinary pair long enough that the display tiles' refresh after the stroke is over too
    /// (2 s after it, the reference `Down` still cost several times a stroke alone). A machine
    /// too slow for it gives no sample of the metrics that need the stroke landed, and the suite
    /// says so.
    const fn landing(self) -> f64 {
        match self {
            Self::Ordinary => 5.0,
            Self::Heavy => 12.0,
        }
    }
}

/// How a later stroke's `Down` meets the earlier stroke, on the over path ([`Script::over`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Before {
    /// At once, with no hover before: the pen left the tablet's range as it lifted.
    Cold,
    /// At once, after a hover where it comes down ([`Weight::hover`]).
    Warmed,
    /// Once the earlier stroke has landed: a stroke alone, the reference.
    Landed,
}

impl Before {
    /// Every way, in the order the over path draws them.
    const ALL: [Self; 3] = [Self::Cold, Self::Warmed, Self::Landed];
}

/// How long after the earlier stroke's lift the pen is back, on the over path's cold and warmed
/// pairs, in seconds at speed 1: a hand's fastest hop.
pub(crate) const OVER_GAP: f64 = 0.05;

/// How long after the earlier stroke's lift its segment ends and the later stroke's begins.
///
/// On the over path, in seconds at speed 1: half the gap, so a frame begun in the later stroke's
/// segment has none of the earlier stroke's samples to show.
pub(crate) const OVER_EDGE: f64 = OVER_GAP / 2.0;

/// The brush of the over path's later strokes: the default 40 pixels.
const OVER_BRUSH: BrushSize = BrushSize::Pixels(40.0);

/// How long the over path's later strokes are drawn for, in seconds at speed 1.
pub(crate) const OVER_DRAWN: f64 = 0.4;

/// How far above and below the equator the over path's upper and lower rows of grounds lie: 35°.
const OVER_PITCH: f64 = 35.0 * PI / 180.0;

/// The over path as it is built ([`Script::over`]): its keyframes and strokes so far.
#[derive(Debug)]
struct OverPath {
    /// The keyframes so far.
    keyframes: Vec<Keyframe>,
    /// The pen's strokes so far, in order.
    strokes: Vec<PenStroke>,
    /// Where the last keyframe ends, in seconds at speed 1.
    now: f64,
    /// How many grounds it has turned to.
    grounds: u32,
    /// The pose of the ground it is at.
    pose: Pose,
}

impl OverPath {
    /// The path through its open segment: 3 s still, as the benchmark path starts.
    fn new() -> Self {
        let open = Keyframe {
            segment: Segment::Open,
            seconds: 3.0,
            to: START,
            ease: Ease::Linear,
        };
        Self {
            keyframes: vec![open],
            strokes: Vec::new(),
            now: 3.0,
            grounds: 0,
            pose: START,
        }
    }

    /// A turn of 1 s to the next ground: a ninth of a turn round from the last, above the
    /// equator, on it and below it in turn ([`OVER_PITCH`]), so the grounds of one row are a third
    /// of a turn apart and no stroke of the path reaches another's.
    fn turn(&mut self) {
        let row = f64::from(1_i32 - i32::try_from(self.grounds % 3).unwrap_or(0));
        self.pose = Pose {
            yaw: f64::from(self.grounds) * TAU / 9.0,
            pitch: row * OVER_PITCH,
            ..START
        };
        self.grounds += 1;
        self.segment(Segment::Turn, self.now + 1.0);
    }

    /// A keyframe of `segment` at the ground's pose, from where the last ends until `until`.
    fn segment(&mut self, segment: Segment, until: f64) {
        self.keyframes.push(Keyframe {
            segment,
            seconds: until - self.now,
            to: self.pose,
            ease: if segment == Segment::Turn {
                Ease::Smooth
            } else {
                Ease::Linear
            },
        });
        self.now = until;
    }

    /// The earlier stroke of `weight`, hovering from just after the last keyframe's end and
    /// touching 0.3 s after it; returns when it lifts.
    fn earlier(&mut self, weight: Weight) -> f64 {
        let down = self.now + 0.3;
        let up = down + weight.drawn();
        self.strokes.push(PenStroke {
            hover: self.now + 0.05,
            down,
            up,
            size: weight.brush(),
            shape: weight.shape(),
            pressure: Pressure::Full,
            part: WHOLE,
            // Out of range as it lifts: where it lifted is no hover.
            leave: 0.0,
        });
        up
    }

    /// A later stroke along the earlier one of `weight` ([`Weight::over`]), the pen back in range
    /// at `back` and touching after `hovering`; returns when it lifts.
    fn later(&mut self, weight: Weight, back: f64, hovering: f64) -> f64 {
        let down = back + hovering;
        let up = down + OVER_DRAWN;
        self.strokes.push(PenStroke {
            hover: back,
            down,
            up,
            size: OVER_BRUSH,
            shape: weight.shape(),
            pressure: Pressure::Firm,
            part: weight.over(),
            leave: 0.0,
        });
        up
    }
}

/// How hard a scripted pen presses along its stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pressure {
    /// Rising from 0.35 to 0.8 at the middle and falling back: a stroke drawn lightly.
    Swelling,
    /// Full pressure all the way: a stroke pressed hard.
    Full,
    /// Rising evenly from none at the touch to full at the lift: every pressure along one stroke.
    Rising,
    /// From a touch of 0.02 to full in the first twelfth of the stroke, then full.
    Touchdown,
    /// 0.8 of full all the way: a stroke drawn firmly.
    Firm,
}

/// Where a scripted stroke runs on the unit sphere, about the point facing the camera (the
/// sphere's point nearest the eye), left to right as the view sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Shape {
    /// Across the facing point, bowing up by a fifth of its span at its middle: seen square on
    /// at its middle, more and more at a slant towards its ends.
    Across {
        /// How far the stroke runs across the view, in world units, at the facing point.
        span: f64,
    },
    /// Straight across the facing point and back, over and over: a scribble whose dabs pile up
    /// on the same ground.
    ToAndFro {
        /// How far each pass runs across the view, in world units, at the facing point.
        span: f64,
        /// How many times it runs across, each pass the other way from the last.
        passes: u32,
    },
    /// Along a circle about the facing point, round from one side of the circle's lowest point
    /// (as the view sees it) to the other: a smile, every point of it seen at the same slant.
    Arc {
        /// The circle's radius as an angle at the sphere's centre, from the facing point, in
        /// radians: the further, the steeper the slant.
        angle: f64,
        /// How far round the circle the stroke runs each side of its lowest point, in radians.
        sweep: f64,
    },
    /// Round the facing point, the angle at the sphere's centre growing evenly: a stroke that
    /// never stops reaching fresh ground.
    Spiral {
        /// The angle it starts at, in radians.
        from: f64,
        /// The angle it ends at, in radians.
        to: f64,
        /// How many times it goes round.
        turns: f64,
    },
}

/// How often a scripted pen reports: the Kamvas's measured rate, in hertz.
pub(crate) const PEN_RATE: f64 = 173.0;

/// How long after lifting a scripted pen leaves the tablet's range, in seconds.
const LEAVE_AFTER: f64 = 0.1;

/// The whole of a stroke's [`Shape`], from its start to its end ([`PenStroke::part`]).
const WHOLE: (f64, f64) = (0.0, 1.0);

/// A scripted stroke's brush size, set as the pen comes into range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BrushSize {
    /// A radius in physical pixels, as the brush's keys set it.
    Pixels(f32),
    /// The radius that covers about this many finest vertices about the stroke's start, found
    /// on the project (`sculpt::radius_covering`) and set as the pixels it spans there, so the
    /// brush does the same work on any project and at any view size.
    Vertices(u32),
}

/// One scripted pen stroke: when the pen hovers, touches and lifts, the brush's size, where the
/// stroke runs, and how hard it presses.
///
/// The stroke is drawn about the point of the unit sphere facing the camera, from left to right
/// as the view sees it, as its [`Shape`] says ([`PenStroke::point`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PenStroke {
    /// When the pen comes into range over the stroke's start: seconds into the run at speed 1.
    hover: f64,
    /// When the tip touches, as `hover`.
    down: f64,
    /// When it lifts, as `hover`; after `down`.
    up: f64,
    /// The brush's size for the stroke, set as the pen comes into range.
    size: BrushSize,
    /// Where it runs.
    shape: Shape,
    /// How hard the pen presses along it.
    pressure: Pressure,
    /// The part of its [`Shape`] it runs over, as fractions of the way along it, from its touch
    /// to its lift: [`WHOLE`], or a part of another stroke's shape, to draw along that stroke.
    part: (f64, f64),
    /// How long after its lift the pen leaves the tablet's range, in seconds of real time
    /// whatever the script's speed: until then it hovers where it lifted.
    leave: f64,
}

impl PenStroke {
    /// The stroke's point a fraction `along` (0 to 1) of the way, seen by `camera`: on the unit
    /// sphere, about the point facing the camera, as its [`Shape`] says.
    fn point(&self, camera: &Camera, along: f64) -> WorldPoint {
        let facing = unit(camera.eye());
        let right = camera.axis(Vec3::new(1.0, 0.0, 0.0));
        let up = camera.axis(Vec3::new(0.0, 1.0, 0.0));
        let along = self.part.0 + (self.part.1 - self.part.0) * along;
        let p = match self.shape {
            Shape::Across { span } => {
                let bow = 0.2 * (PI * along).sin();
                unit(facing + right * ((along - 0.5) * span) + up * (bow * span))
            }
            Shape::ToAndFro { span, passes } => {
                // Each pass runs the span one way, the next back.
                let gone = (along * f64::from(passes)).rem_euclid(2.0);
                let across = gone.min(2.0 - gone) - 0.5;
                unit(facing + right * (across * span))
            }
            Shape::Arc { angle, sweep } => {
                // The view's right and down, square to the facing point's normal: the circle's
                // plane's axes.
                let across = unit(right - facing * right.dot(facing));
                let down = unit(-up - facing * (-up).dot(facing) - across * (-up).dot(across));
                let round = (2.0 * along - 1.0) * sweep;
                let outward = down * round.cos() + across * round.sin();
                facing * angle.cos() + outward * angle.sin()
            }
            Shape::Spiral { from, to, turns } => {
                let across = unit(right - facing * right.dot(facing));
                let above = unit(up - facing * up.dot(facing) - across * up.dot(across));
                // Both by the root of how far along: the pen's speed stays about even.
                let angle = from + (to - from) * along.sqrt();
                let round = 2.0 * PI * turns * along.sqrt();
                let outward = across * round.cos() + above * round.sin();
                facing * angle.cos() + outward * angle.sin()
            }
        };
        WorldPoint::new(p.x, p.y, p.z)
    }

    /// The tip's pressure a fraction `along` (0 to 1) of the way, as its [`Pressure`] says:
    /// rising from 0.35 to 0.8 at the middle and falling back, full, or rising from none to full.
    fn pressure(&self, along: f64) -> Unit {
        match self.pressure {
            Pressure::Swelling => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a pressure in 0.35..=0.8: f32 holds it"
                )]
                let pressure = (0.35 + 0.45 * (PI * along).sin()) as f32;
                Unit::try_from(pressure).unwrap_or(Unit::ONE)
            }
            Pressure::Full => Unit::ONE,
            Pressure::Firm => Unit::try_from(0.8).unwrap_or(Unit::ONE),
            Pressure::Rising => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a pressure in 0.0..=1.0: f32 holds it"
                )]
                let pressure = along.clamp(0.0, 1.0) as f32;
                Unit::try_from(pressure).unwrap_or(Unit::ONE)
            }
            Pressure::Touchdown => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a pressure in 0.02..=1.0: f32 holds it"
                )]
                let pressure = (0.02 + 12.0 * along).min(1.0) as f32;
                Unit::try_from(pressure).unwrap_or(Unit::ONE)
            }
        }
    }
}

/// `v` scaled to unit length; `v` is never zero here (a camera's eye, outside the sphere, or an
/// axis with a component square to another).
fn unit(v: Vec3) -> Vec3 {
    v * v.dot(v).sqrt().recip()
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
    /// The brush's size for the stroke: on the stroke's first sample only (the pen entering
    /// range; its `Down`, for a stroke with no hover).
    pub(crate) size: Option<BrushSize>,
}

/// Which camera path a script runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ScriptPath {
    /// The renderer's measured run ([`Script::benchmark`]).
    #[default]
    Benchmark,
    /// Five pen strokes: near, far, fast and long, that again pressed hard, and a million-vertex
    /// brush ([`Script::strokes`]).
    Strokes,
    /// One long stroke seen at a slant, pressed from none to full ([`Script::sweep`]).
    Sweep,
    /// One stroke 25 s long, out across the sphere ([`Script::spiral`]).
    Spiral,
    /// That stroke touching down lightly ([`Script::spiral_light`]).
    SpiralLight,
    /// The first of the strokes alone ([`Script::near`]).
    Near,
    /// Strokes over strokes still landing: six pairs ([`Script::over`]).
    Over,
}

/// Why a text isn't a [`ScriptPath`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0:?} isn't a script path: benchmark, strokes, sweep, spiral, spiral-light, near or over")]
pub(crate) struct BadPath(String);

impl FromStr for ScriptPath {
    type Err = BadPath;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "benchmark" => Ok(Self::Benchmark),
            "strokes" => Ok(Self::Strokes),
            "sweep" => Ok(Self::Sweep),
            "spiral" => Ok(Self::Spiral),
            "spiral-light" => Ok(Self::SpiralLight),
            "near" => Ok(Self::Near),
            "over" => Ok(Self::Over),
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
            ScriptPath::Sweep => Self::sweep(speed),
            ScriptPath::Spiral => Self::spiral(speed),
            ScriptPath::SpiralLight => Self::spiral_light(speed),
            ScriptPath::Near => Self::near(speed),
            ScriptPath::Over => Self::over(speed),
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

    /// The sweep path (the module notes), run at `speed`.
    pub(crate) fn sweep(speed: Speed) -> Self {
        let mut script = Self {
            start: START,
            keyframes: SWEEP.to_vec(),
            speed,
            strokes: vec![SWEEP_STROKE],
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The spiral path (the module notes), run at `speed`.
    pub(crate) fn spiral(speed: Speed) -> Self {
        let mut script = Self {
            start: START,
            keyframes: SPIRAL.to_vec(),
            speed,
            strokes: vec![SPIRAL_STROKE],
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The light spiral path (the module notes), run at `speed`.
    pub(crate) fn spiral_light(speed: Speed) -> Self {
        let mut script = Self {
            start: START,
            keyframes: SPIRAL.to_vec(),
            speed,
            strokes: vec![SPIRAL_LIGHT_STROKE],
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The strokes path up to the end of its first stroke, run at `speed`: as [`Script::strokes`]
    /// so far, for measuring a stroke with no more of the path than it needs.
    pub(crate) fn near(speed: Speed) -> Self {
        let through_the_stroke = STROKES
            .iter()
            .position(|key| key.segment == Segment::StrokeNear);
        let keyframes = STROKES
            .iter()
            .take(through_the_stroke.map_or(0, |at| at + 1));
        let mut script = Self {
            start: START,
            keyframes: keyframes.copied().collect(),
            speed,
            strokes: PEN_STROKES.iter().take(1).copied().collect(),
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The over path (the module notes), run at `speed`: for each [`Weight`] and each [`Before`],
    /// a turn to ground of its own, the earlier stroke, and the later one along it; then the
    /// chain of three.
    pub(crate) fn over(speed: Speed) -> Self {
        let mut path = OverPath::new();
        let pairs = Weight::ALL
            .into_iter()
            .flat_map(|weight| Before::ALL.map(|before| (weight, before)));
        for (weight, before) in pairs {
            path.turn();
            let lift = path.earlier(weight);
            path.segment(Segment::OverFirst, lift + OVER_EDGE);
            let (back, hovering) = match before {
                Before::Cold => (OVER_GAP, 0.0),
                Before::Warmed => (OVER_GAP, weight.hover()),
                Before::Landed => (weight.landing(), 0.3),
            };
            let lift = path.later(weight, lift + back, hovering);
            path.segment(Segment::Over(weight, before), lift + weight.landing());
        }
        path.turn();
        let first = path.earlier(Weight::Ordinary);
        let second = path.later(Weight::Ordinary, first + OVER_GAP, 0.0);
        path.segment(Segment::OverFirst, second + OVER_EDGE);
        let third = path.later(Weight::Ordinary, second + OVER_GAP, 0.0);
        path.segment(Segment::OverChain, third + Weight::Ordinary.landing());
        path.segment(Segment::Settle, path.now + 2.0);
        let mut script = Self {
            start: START,
            keyframes: path.keyframes,
            speed,
            strokes: path.strokes,
            pen: Vec::new(),
        };
        script.pen = script.every_pen_sample();
        script
    }

    /// The scripted pen's samples taken after `from` and up to `to` into the run, in order (see
    /// the module notes).
    ///
    /// Samples come every 1/[`PEN_RATE`] s from each stroke's hover, in real time; the first
    /// is the pen entering range (with the stroke's brush size), those before the stroke's
    /// touch hover, the first at or after it is the `Down`, then `Drag`s, then an `Up` at the
    /// lift, and a `Leave` its [`PenStroke::leave`] later ([`LEAVE_AFTER`], but for the over
    /// path's). A stroke whose hover is its touch has no hover: its first sample is its `Down`.
    ///
    /// The samples are made once, as the script is; this only finds them.
    pub(crate) fn pen(&self, from: Duration, to: Duration) -> &[ScriptPen] {
        let start = self.pen.partition_point(|sample| sample.time <= from);
        let end = self.pen.partition_point(|sample| sample.time <= to);
        self.pen.get(start..end.max(start)).unwrap_or_default()
    }

    /// When the scripted pen first touches (its first `Down`) at or after `from` and before `to`
    /// into the run, if it does: a segment's stroke, for its record.
    pub(crate) fn first_down(&self, from: Duration, to: Duration) -> Option<Duration> {
        let mut within = self
            .pen
            .iter()
            .filter(|sample| (from..to).contains(&sample.time));
        within
            .find(|sample| sample.phase == Phase::Down)
            .map(|sample| sample.time)
    }

    /// When the scripted pen's first sample after `from` into the run is taken, if there is one.
    pub(crate) fn next_pen_after(&self, from: Duration) -> Option<Duration> {
        let next = self.pen.partition_point(|sample| sample.time <= from);
        self.pen.get(next).map(|sample| sample.time)
    }

    /// Every sample of the pen's strokes, in order of time (the strokes are in order, and don't
    /// overlap).
    fn every_pen_sample(&self) -> Vec<ScriptPen> {
        let samples = self.strokes.iter().flat_map(|stroke| {
            self.stroke_samples(stroke)
                .into_iter()
                .enumerate()
                .map(move |(nth, (time, phase))| {
                    let size = (nth == 0).then_some(stroke.size);
                    self.sample(stroke, time, phase, size)
                })
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
        let ends = [(up, Phase::Up), (up + stroke.leave, Phase::Leave)];
        let samples = regular.chain(ends);
        samples.collect()
    }

    /// `stroke`'s sample at `time` seconds into the run, in `phase` (one of
    /// [`Script::stroke_samples`]), setting the brush's size to `size` if given (the stroke's
    /// first does).
    fn sample(
        &self,
        stroke: &PenStroke,
        time: f64,
        phase: Phase,
        size: Option<BrushSize>,
    ) -> ScriptPen {
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
                stroke.pressure(along)
            } else {
                Unit::ZERO
            },
            size,
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
