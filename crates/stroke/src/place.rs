//! Placement: the smoothed screen path walked into dabs on the surface, a set surface distance
//! apart, each with its size, dynamics and frame.
//!
//! **The size freeze.** At the stroke's first hit, a [`SizeMode::Screen`] radius becomes a world
//! radius once, at the hit's view depth through [`Projector::world_per_pixel`], and holds until
//! pen-up; a [`SizeMode::World`] radius is used as given. Zooming mid-stroke can't make the
//! stroke uneven. Dynamics' size factor scales each dab's radius from it; spacing, the reach and
//! the speed and distance inputs are all measured against it. A strength in radii
//! ([`StrengthMode::Radius`]) becomes model units here too, once, from the frozen radius.
//!
//! **The walk.** Each new brush point (from the [`Path`](crate::Path)) makes a screen segment
//! from the one before. The walk steps along it in screen space: each candidate point is
//! interpolated on the segment, its ray cast onto the surface, and the straight distance from
//! the last accepted point (the *chord*) added to the surface distance travelled. A step's
//! chord is held to at most the smaller of half the spacing and one radius (so the sum of chords
//! follows a curved surface and no step can outrun the reach), and to what is left before the
//! next dab: a step that overshoots is shortened by the secant rule (the chord is near
//! proportional to the screen step locally) and tried again. A dab is placed where the distance
//! travelled since the last dab reaches the spacing, to within 1 %
//! ([`SPACING_TOLERANCE`]). Steps grow again after short chords, so a slant view (where a pixel
//! spans more of the surface) and a pen moving fast or standing still all cost a few raycasts
//! per dab. The spacing is the brush's spacing × the frozen radius × the dynamics' spacing
//! factor at the last dab, never less than [`Spacing::MIN`](crate::Spacing::MIN) radii.
//!
//! **The stroke's region.** The first hit anchors the stroke. Each candidate's hit must be
//! [`Surface::continuous`] with the last accepted point within [`REACH_RADII`] radii; a ray that
//! misses, or hits a part that isn't continuous (across a gap, a separate part behind a
//! silhouette, the far face of a thin slab), goes to [`Surface::closest_on_region`] instead: the
//! point of the stroke's own surface closest to the pen's ray, near the last accepted point. Past
//! a silhouette the resolved points stop advancing, so no dabs pile up there, and the stroke
//! never jumps to a separate part. Checking against the last accepted point (a chord at most a
//! radius back) rather than the last dab keeps every link short.
//!
//! **The frame.** Each dab carries its area normal and a tangent along the stroke's direction
//! there: the last step's direction, projected into the plane the normal is normal to. The area
//! normal is the surface's normals inside the dab's sphere, weighted by its falloff, summed and
//! normalised ([`Surface::area_normal`]), taken as the dab is placed, its final radius and
//! hardness known; where that sum is degenerate (zero or not finite), the surface normal at the
//! hit. The
//! first dab waits for the pen's first movement to learn its direction (it is emitted by the
//! first [`Stroke::push`](crate::Stroke::push) either way); where there is no direction (a still
//! pen, a step along the normal), the previous dab's tangent is used, and failing that a world
//! axis projected into the tangent plane.
//!
//! **Dynamics.** A dab's readings are the pen's at its point on the segment (pressure and time
//! interpolated, tilt and twist from the nearer sample), plus the speed along the surface since
//! the last dab and the distance from the start, both in radii, and the stroke's random value
//! for its sequence number.
//!
//! **Dabs too small to move anything.** A dab whose dynamics' size factor is below
//! [`MIN_DAB_SIZE`] (a sample with no pressure, under the default pressure-to-size curve) is not
//! placed: the walk goes on as if it were (the spacing to the next dab counts from its place),
//! but nothing is emitted and it takes no sequence number, so the dabs that are placed keep their
//! spacing and number on from each other. No dab of radius 0 ever leaves the engine.
//!
//! **The tail.** [`Stroke::end`](crate::Stroke::end) walks the last segment to the pen's last
//! raw sample, then places a final dab there unless one already lies within 1 % of the spacing
//! of it, so a fast stroke's dabs always reach where the pen lifted.
//!
//! **Determinism.** Everything is computed in `f64` from the samples, the settings, the
//! projector and the surface in a fixed order, with no clock and no unseeded randomness, so the
//! same inputs give bit-identical dabs.

use crate::surface::{Hit, Projector, Ray, Surface};
use crate::symmetry::Placed;
use crate::{
    ACTION_SLOTS, DVec3, Dynamics, Inputs, Sample, ScreenPoint, SizeMode, Spacing, StrengthMode,
    StrokeRandom, StrokeSettings, Unit,
};

/// The smallest size factor a placed dab has: a dab whose dynamics make it smaller than this
/// share of the stroke's frozen radius, a thousandth, is not placed (see the module's notes).
///
/// It would move next to nothing (at most a thousandth of the brush's footprint's width) and,
/// at 0, nothing at all, which an action refuses as a dab.
pub const MIN_DAB_SIZE: f32 = 1e-3;

/// How far a hit may be from the last accepted point and still continue the stroke, in radii of
/// the stroke's frozen world radius: the reach given to [`Surface::continuous`] and
/// [`Surface::closest_on_region`].
///
/// A few radii: far enough that an ordinary step (at most a radius) never fails it, near enough
/// that a separate part a little way behind a silhouette or across a thin gap does. Not tuned:
/// the brush-engine spec leaves it to tune with the user on the tablet.
pub const REACH_RADII: f64 = 3.0;

/// How close to its target spacing a dab is placed, as a fraction of the spacing: the walk
/// shortens a step that would overshoot by more, and places the dab once it is within this short.
pub const SPACING_TOLERANCE: f64 = 0.01;

/// The most times one step is shortened before it is taken anyway: only where the surface
/// gives a point the walk can't approach smoothly (a resolved point that jumps).
const MAX_TRIES: u32 = 32;

/// The shortest screen step, in physical pixels: a step this short is taken whatever its chord,
/// so the walk always advances.
const MIN_STEP_PX: f64 = 1e-3;

/// A screen segment of the path, from one brush point to the next, with the pen's readings at
/// each end.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    /// The start: the previous brush point.
    pub(crate) from: Sample,
    /// The end: the new brush point.
    pub(crate) to: Sample,
}

impl Segment {
    /// The screen point a fraction `u` of the way along; exactly the end at `u ≥ 1`.
    fn point(&self, u: f64) -> ScreenPoint {
        if u >= 1.0 {
            self.to.position()
        } else {
            self.from.position().lerp(self.to.position(), u)
        }
    }

    /// The pen's clock a fraction `u` of the way along.
    fn time(&self, u: f64) -> f64 {
        if u >= 1.0 {
            self.to.time()
        } else {
            self.from.time() + (self.to.time() - self.from.time()) * u
        }
    }

    /// The pen's readings a fraction `u` of the way along: the pressure interpolated, the tilt
    /// and twist from the nearer end (angles don't interpolate cleanly across their wrap).
    fn inputs(&self, u: f64) -> Inputs {
        let (a, b) = (self.from.pressure().get(), self.to.pressure().get());
        #[expect(
            clippy::cast_possible_truncation,
            reason = "u is in 0..=1: a fraction for an f32 interpolation"
        )]
        let t = u.clamp(0.0, 1.0) as f32;
        let nearer = if u < 0.5 { &self.from } else { &self.to };
        Inputs {
            pressure: Unit::saturating(a + (b - a) * t),
            ..Inputs::from_sample(nearer)
        }
    }
}

/// A unit tangent at a point with unit normal `normal`, as close to `direction` as it can be.
///
/// That is `direction` projected into the tangent plane; failing that, `previous` projected;
/// failing that, the world axis least aligned with the normal, projected.
fn tangent_along(normal: DVec3, direction: DVec3, previous: Option<DVec3>) -> DVec3 {
    let project = |d: DVec3| (d - normal * normal.dot(d)).normalized();
    project(direction)
        .or_else(|| previous.and_then(project))
        .unwrap_or_else(|| {
            let (ax, ay, az) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
            let axis = if ax <= ay && ax <= az {
                DVec3::X
            } else if ay <= az {
                DVec3::Y
            } else {
                DVec3::Z
            };
            // The least aligned axis is at most 1/√3 along a unit normal: always projectable.
            project(axis).unwrap_or(DVec3::X)
        })
}

/// The area normal of a dab at `hit` of `radius` and `hardness` on `surface`
/// ([`Surface::area_normal`]), at unit length; the hit's normal where the surface gives no sum,
/// or one that is zero or not finite.
fn area_normal<S: Surface>(surface: &mut S, hit: &Hit, radius: f64, hardness: f32) -> DVec3 {
    surface
        .area_normal(hit.position, radius, hardness)
        .and_then(DVec3::normalized)
        .unwrap_or(hit.normal)
}

/// `hit` with its normal at unit length, or `None` if its position isn't finite or its normal
/// has no direction: what the walk accepts from a [`Surface`].
fn checked(hit: Hit) -> Option<Hit> {
    let normal = hit.normal.normalized()?;
    hit.position.is_finite().then_some(Hit { normal, ..hit })
}

/// The placement state of one stroke: the frozen radius, where the walk has reached, and what
/// the next dab needs.
#[derive(Debug, Clone)]
pub(crate) struct Placer {
    /// The brush's dynamics table.
    dynamics: Dynamics,
    /// The brush's spacing, in radii.
    spacing: f64,
    /// The action's parameters, before dynamics.
    action_params: [f32; ACTION_SLOTS],
    /// The brush's own strength before dynamics, in model units: a strength in radii is turned
    /// into model units with the frozen radius.
    strength: f32,
    /// The brush's own hardness, before dynamics.
    hardness: f32,
    /// The frozen world radius: positive and finite.
    radius: f64,
    /// The stroke's random values.
    random: StrokeRandom,
    /// The pen's clock at the first sample: dab times count from it.
    start: f64,
    /// The last accepted point on the surface: where the walk has reached.
    cursor: Hit,
    /// The direction of the last step that moved, if any has.
    heading: Option<DVec3>,
    /// Surface distance travelled since the last dab.
    travelled: f64,
    /// Surface distance travelled since the stroke began.
    distance: f64,
    /// The surface distance from the last dab to the next.
    next_spacing: f64,
    /// The pen's clock at the last dab.
    last_time: f64,
    /// [`Placer::distance`] at the last dab.
    last_distance: f64,
    /// The speed at the last dab, in radii per second: kept when no time passes between dabs.
    speed: f64,
    /// The last dab's tangent.
    tangent: Option<DVec3>,
    /// The next dab's sequence number.
    seq: u32,
    /// The first dab, made at the first hit and waiting for the pen's first movement to give it
    /// a direction; `None` once emitted.
    pending: Option<Placed>,
}

impl Placer {
    /// The placement of a stroke starting at `first`, or `None` if nothing is under it (or the
    /// camera gives no ray or no usable pixel size there).
    pub(crate) fn begin<P: Projector, S: Surface>(
        settings: &StrokeSettings,
        random: StrokeRandom,
        projector: &P,
        surface: &mut S,
        first: &Sample,
    ) -> Option<Self> {
        let ray = projector.ray(first.position())?;
        let hit = checked(surface.raycast(&ray)?)?;
        let given = f64::from(settings.radius().get());
        let radius = match settings.size_mode() {
            SizeMode::Screen => given * projector.world_per_pixel(ray.depth_of(hit.position)),
            SizeMode::World => given,
        };
        if !(radius.is_finite() && radius > 0.0) {
            return None;
        }
        let given_strength = settings.strength().get();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a dab's strength is f32: rounded once, as the radius freezes"
        )]
        let strength = match settings.strength_mode() {
            StrengthMode::Model => given_strength,
            StrengthMode::Radius => (f64::from(given_strength) * radius) as f32,
        };
        let mut placer = Self {
            dynamics: settings.dynamics().clone(),
            spacing: f64::from(settings.spacing().get()),
            action_params: settings.action_params(),
            strength,
            hardness: settings.hardness().get(),
            radius,
            random,
            start: first.time(),
            cursor: hit,
            heading: None,
            travelled: 0.0,
            distance: 0.0,
            next_spacing: 0.0,
            last_time: first.time(),
            last_distance: 0.0,
            speed: 0.0,
            tangent: None,
            seq: 0,
            pending: None,
        };
        let anchor = Segment {
            from: *first,
            to: *first,
        };
        placer.pending = placer.make(surface, &anchor, 0.0, DVec3::ZERO);
        Some(placer)
    }

    /// The frozen world radius.
    pub(crate) const fn radius(&self) -> f64 {
        self.radius
    }

    /// The reach, in world units: [`REACH_RADII`] radii.
    fn reach(&self) -> f64 {
        REACH_RADII * self.radius
    }

    /// The longest chord a step may take before the next dab is due: half the spacing, at most
    /// a radius, and no more than what is left.
    fn target(&self) -> f64 {
        let longest = (0.5 * self.next_spacing).min(self.radius);
        (self.next_spacing - self.travelled).min(longest)
    }

    /// Where the pen's ray through `screen` lands on the stroke's region: its hit if that is
    /// continuous with the cursor, otherwise the region's point closest to the ray. `None` if the
    /// camera gives no ray or the surface no point.
    fn resolve<P: Projector, S: Surface>(
        &self,
        projector: &P,
        surface: &mut S,
        screen: ScreenPoint,
    ) -> Option<Hit> {
        let ray: Ray = projector.ray(screen)?;
        let reach = self.reach();
        let direct = surface
            .raycast(&ray)
            .and_then(checked)
            .filter(|hit| surface.continuous(&self.cursor, hit, reach));
        direct.or_else(|| {
            surface
                .closest_on_region(&ray, &self.cursor, reach)
                .and_then(checked)
        })
    }

    /// The first screen step along `segment` (of `length` pixels), as a fraction: about the
    /// pixels a target chord spans at the cursor's depth.
    fn first_step<P: Projector>(&self, projector: &P, segment: &Segment, length: f64) -> f64 {
        let per_pixel = projector
            .ray(segment.from.position())
            .map(|ray| projector.world_per_pixel(ray.depth_of(self.cursor.position)))
            .filter(|w| w.is_finite() && *w > 0.0);
        per_pixel.map_or(1.0, |w| (self.target() / w / length).clamp(0.0, 1.0))
    }

    /// Walks `segment`, appending the dabs it places to `out` (the pending first dab among them,
    /// once the pen has moved).
    pub(crate) fn walk<P: Projector, S: Surface>(
        &mut self,
        projector: &P,
        surface: &mut S,
        segment: &Segment,
        out: &mut Vec<Placed>,
    ) {
        let length = segment.from.position().distance(segment.to.position());
        // Also false for NaN, which finite screen points can't give.
        if length.is_nan() || length <= 0.0 {
            return;
        }
        let min_step = MIN_STEP_PX / length;
        let mut u = 0.0;
        let mut step = self.first_step(projector, segment, length).max(min_step);
        // A `loop` rather than `while u < 1.0`: `u` starts at 0 and ends exactly at 1.
        loop {
            let target = self.target();
            let mut tries = 0;
            let (at, found) = loop {
                let at = (u + step).min(1.0);
                let Some(hit) = self.resolve(projector, surface, segment.point(at)) else {
                    break (at, None);
                };
                let chord = self.cursor.position.distance(hit.position);
                tries += 1;
                if chord <= target * (1.0 + SPACING_TOLERANCE)
                    || at - u <= min_step
                    || tries >= MAX_TRIES
                {
                    break (at, Some((hit, chord)));
                }
                // The chord overshoots: shorten the step in proportion, a little more, so the
                // next try lands just inside.
                step = ((at - u) * (0.95 * target / chord).clamp(0.05, 0.9)).max(min_step);
            };
            let taken = at - u;
            u = at;
            // With nothing to stand on (a failing projector or surface), the walk moves on along
            // the screen with the same step.
            if let Some((hit, chord)) = found {
                self.advance(surface, segment, at, hit, chord, out);
                // The next step: scaled toward the next target from this one's chord.
                let grow = if chord > 0.0 {
                    (self.target() / chord).clamp(0.25, 4.0)
                } else {
                    4.0
                };
                step = (taken * grow).max(min_step);
            }
            if u >= 1.0 {
                return;
            }
        }
    }

    /// Moves the cursor to `hit`, a `chord` away and `u` of the way along `segment`, emitting
    /// the waiting first dab once the pen has moved and a dab where the spacing is reached, its
    /// area normal taken from `surface`.
    fn advance<S: Surface>(
        &mut self,
        surface: &mut S,
        segment: &Segment,
        u: f64,
        hit: Hit,
        chord: f64,
        out: &mut Vec<Placed>,
    ) {
        let direction = hit.position - self.cursor.position;
        self.cursor = hit;
        self.travelled += chord;
        self.distance += chord;
        if chord > 0.0 {
            self.heading = Some(direction);
            self.flush(out);
        }
        if self.travelled >= self.next_spacing * (1.0 - SPACING_TOLERANCE) {
            out.extend(self.make(surface, segment, u, direction));
        }
    }

    /// Emits the first dab if it is still waiting, its tangent along the stroke's heading if the
    /// pen has moved (see the module's notes on the frame).
    pub(crate) fn flush(&mut self, out: &mut Vec<Placed>) {
        if let Some(mut first) = self.pending.take() {
            first.tangent = tangent_along(first.normal, self.heading.unwrap_or(DVec3::ZERO), None);
            self.tangent = Some(first.tangent);
            out.push(first);
        }
    }

    /// Places the tail: a dab at the cursor (where the last segment ended) unless one already
    /// lies there, to within the spacing tolerance, its area normal taken from `surface`.
    /// `segment` is the last segment walked.
    pub(crate) fn tail<S: Surface>(
        &mut self,
        surface: &mut S,
        segment: &Segment,
        out: &mut Vec<Placed>,
    ) {
        self.flush(out);
        if self.travelled > self.next_spacing * SPACING_TOLERANCE {
            let heading = self.heading.unwrap_or(DVec3::ZERO);
            out.extend(self.make(surface, segment, 1.0, heading));
        }
    }

    /// The dab at the cursor, `u` of the way along `segment`, heading along `direction`, its
    /// normal its area normal on `surface` ([`area_normal`]); the spacing, speed and distance
    /// bookkeeping move on to it. `None` if its size factor is below [`MIN_DAB_SIZE`]: the
    /// bookkeeping moves on all the same, but it takes no sequence number (nor is its area normal
    /// taken).
    fn make<S: Surface>(
        &mut self,
        surface: &mut S,
        segment: &Segment,
        u: f64,
        direction: DVec3,
    ) -> Option<Placed> {
        let time = segment.time(u);
        let dt = time - self.last_time;
        if dt > 0.0 {
            self.speed = (self.distance - self.last_distance) / dt / self.radius;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dynamics reads its inputs as f32"
        )]
        let inputs = Inputs {
            speed: self.speed as f32,
            distance: (self.distance / self.radius) as f32,
            random: self.random.at(self.seq),
            ..segment.inputs(u)
        };
        let values = self.dynamics.evaluate(&inputs);
        // `>=` is false for NaN too, which finite dynamics can't give.
        let sized = values.size >= MIN_DAB_SIZE;
        let radius = self.radius * f64::from(values.size);
        let hardness = (self.hardness * values.hardness).clamp(0.0, 1.0);
        let normal = if sized {
            area_normal(surface, &self.cursor, radius, hardness)
        } else {
            self.cursor.normal
        };
        let tangent = tangent_along(normal, direction, self.tangent);
        let mut params = self.action_params;
        for (param, factor) in params.iter_mut().zip(values.action) {
            *param *= factor;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a dab's time is f32 seconds since its stroke began"
        )]
        let since_start = (time - self.start) as f32;
        let placed = sized.then_some(Placed {
            position: self.cursor.position,
            normal,
            tangent,
            radius,
            strength: self.strength * values.strength,
            angle: values.angle,
            hardness,
            params,
            seq: self.seq,
            time: since_start,
        });
        let floor = f64::from(Spacing::MIN) * self.radius;
        self.next_spacing = (self.spacing * self.radius * f64::from(values.spacing)).max(floor);
        if sized {
            self.seq = self.seq.wrapping_add(1);
        }
        self.travelled = 0.0;
        self.last_time = time;
        self.last_distance = self.distance;
        self.tangent = Some(tangent);
        placed
    }
}
