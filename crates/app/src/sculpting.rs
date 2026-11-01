//! Sculpting with the pen: the app's [`Brush`], strokes through the stroke engine into the store,
//! shown in the same frame (the brush-engine spec, §3 "In the app").
//!
//! # A stroke
//!
//! - **Routing** ([`Brush::route`]): a pen `Down` whose ray hits the surface is the brush's stroke;
//!   off the surface it navigates, so no stroke starts off the surface. The ray is the one the
//!   stroke engine casts ([`ViewProjector`]), through the store's spatial index.
//! - **Pen down:** [`Stroke::begin`] with the camera's projector and a [`StoreSurface`], which
//!   freezes the brush's world radius at the first hit (the size is in screen pixels); then
//!   [`Project::begin_stroke`], so every dab of the stroke is one op. A `Down` the engine finds
//!   nothing under (the route's hit lost to rounding) starts no stroke, and its samples are
//!   dropped.
//! - **Each sample:** [`Stroke::push`], through a fresh [`StoreSurface`] each call, ended with
//!   [`StoreSurface::finish`]; the dabs it places join the stroke's queue, but those of radius 0 (a
//!   sample with no pressure), which reach nothing.
//! - **Each frame:** the queue's dabs are applied by [`apply_dabs`], as many as the frame's budget
//!   allows ([`DabBudget`]), in order; the rest wait for the next frame. The edit closes the
//!   store's epoch, which the frame loop drains after the brush phase and uploads in the same
//!   frame.
//! - **Pen up:** [`Stroke::end`] gives the tail; once the queue is empty the stroke's record
//!   ([`StrokeRecord`]: the samples the engine read, the settings' bytes, the camera at the
//!   stroke's start, [`stroke::ENGINE_VERSION`]) ends the op ([`Project::end_stroke`]). A stroke
//!   whose `Up` was lost ends at the next sample out of a stroke.
//!
//! A sample's time is seconds from a fixed epoch (the brush's creation) on the pen's
//! de-jittered clock.
//!
//! # The budget: dabs per frame
//!
//! Applying a frame's dabs costs about linearly in dabs and in the vertices each covers (Task 3
//! measured four dabs of 50,000 vertices at 5.4 ms a call, one at 1.73 ms), which would eat the
//! pen's 8 ms budget from sample to present. So a frame applies dabs while their summed
//! footprint, estimated as `π (r / h)²` vertices for a dab of radius `r` over the index level's
//! mean quad spacing `h`, stays within [`DabBudget::DEFAULT`] (150,000 vertices: about 4 ms on the
//! development desktop by those numbers), and always at least one. The rest are carried to the
//! next frame, in order: no dab is dropped, and the stroke's tail waits behind them. A frame that
//! carries dabs wants the next. A new stroke starting while the last still has dabs queued first
//! applies all of them and ends it.
//!
//! The footprint limit F (the M1 spec's "Big brushes", about 180,000 vertices a dab,
//! [`LIMIT_VERTICES`]) is reported per frame ([`BrushReport::above_limit`]): the latency target
//! holds below it; above it is the big-brush path, still to come.
//!
//! Carried dabs were placed on the surface as it was when they were placed, before the dabs
//! ahead of them were applied; a stroke's later dabs always are (a frame's dabs apply after its
//! samples are all placed), so carrying changes when a dab lands, not where.
//!
//! # Warming under a hovering pen
//!
//! A stroke's first sample finds the surface under it cold: evaluating the reach ball's tiles
//! costs up to 6.8 ms for `closest_on_region` and 2.4 ms for the raycast (Task 3). So while the
//! pen hovers, the brush warms the region under it, once per position and view: each frame's step
//! notes where it hovers ([`PenFrame::hover`]), and [`Brush::prepare`], after the frame's
//! hand-off and so off the latency path, raycasts there and evaluates the tiles within the reach
//! ([`stroke::REACH_RADII`] brush radii) of the hit, [`WARM_BATCH`] at a time on the store's
//! workers, until the frame's deadline. What is left goes on in the next frame's `prepare`; a
//! pen that moves on, touches or leaves cancels it. A still pen is warmed once.
//!
//! # Errors
//!
//! A store error ends the stroke cleanly (the brush-engine spec, §3 "Errors"): the dabs of a push
//! that ended in an error are dropped, the edits so far stay, the dabs still queued are applied,
//! and the op ends with the stroke's record; a failed [`apply_dabs`] drops the dabs left (its
//! own are applied up to the failing one; applying them again would count some twice). Either
//! way the error shows in the overlay ([`Brush::status`]) and the app goes on.

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI};
use std::time::Instant;

use anyhow::Result;
use geom::{Level, TileKey};
use input::Phase;
use render::camera::View;
use sculpt::{StoreSurface, apply_dabs};
use store::{
    Evaluation, PenSample as RecordSample, PenTilt, Project, SpatialIndex, StrokeCamera,
    StrokeRecord, TileStore,
};
use stroke::{
    DVec3, Dab, Hit, Projector, REACH_RADII, Radius, Ray, Sample, ScreenPoint, Strength, Stroke,
    StrokeId, StrokeSettings, Surface, Tilt, Twist, Unit,
};

use crate::camera::{Camera, Pixel};
use crate::frame::{Brush, BrushChange, BrushContext, BrushReport, SampleTimes};
use crate::pen::{PenFrame, PenSample, Route};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The default base strength, as a fraction of the project's size (its bounding radius).
///
/// 0.002, so a draw stroke at full pressure on the benchmark sphere (radius 1) moves the
/// surface by about 0.013 at its centre line (spacing 0.15 radii stacks about 6.7 dabs' worth), a
/// low ridge under the default 40-pixel brush seen whole. The keys scale it
/// ([`STRENGTH_STEP`]).
pub(crate) const DEFAULT_STRENGTH_PER_SIZE: f64 = 0.002;

/// How much one press of a strength key scales the strength.
pub(crate) const STRENGTH_STEP: f32 = 1.25;

/// The strength's range, as factors of the default: a 64th to 64 times.
pub(crate) const STRENGTH_FACTORS: (f32, f32) = (1.0 / 64.0, 64.0);

/// How much one press of a size key scales the brush's radius.
pub(crate) const SIZE_STEP: f32 = 1.25;

/// The brush radius's range, in physical pixels.
pub(crate) const RADIUS_PIXELS: (f32, f32) = (1.0, 2000.0);

/// The footprint limit F, as the vertices a dab covers: the M1 spec's "Big brushes" (about 150
/// tiles updated, about 180,000 vertices). The pen's latency target holds below it.
pub(crate) const LIMIT_VERTICES: f64 = 180_000.0;

/// Tiles the warming evaluates together on the store's workers between checks of its deadline.
pub(crate) const WARM_BATCH: usize = 16;

/// How many dabs a frame applies: as many as fit a budget of vertices covered, summed over the
/// dabs, and always at least one (see "The budget" in the module notes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DabBudget {
    /// The most vertices a frame's dabs cover together, by their estimated footprints.
    vertices: f64,
}

impl DabBudget {
    /// 150,000 vertices a frame: about 4 ms of [`apply_dabs`] on the development desktop by Task
    /// 3's measurements (27 ns a vertex covered, four 50,000-vertex dabs in 5.4 ms), half the
    /// pen's 8 ms from sample to present.
    pub(crate) const DEFAULT: Self = Self {
        vertices: 150_000.0,
    };

    /// A budget of `vertices` covered a frame.
    #[cfg(test)]
    pub(crate) const fn new(vertices: f64) -> Self {
        Self { vertices }
    }

    /// How many of `footprints` (the queued dabs' estimated vertices, in order) to apply this
    /// frame: the longest run from the front whose sum fits, at least one if there are any.
    pub(crate) fn take(self, footprints: impl IntoIterator<Item = f64>) -> usize {
        let mut sum = 0.0;
        let mut count = 0;
        for footprint in footprints {
            sum += footprint;
            if count > 0 && sum > self.vertices {
                break;
            }
            count += 1;
        }
        count
    }
}

impl Default for DabBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The camera as the stroke engine's [`Projector`]: rays from the eye through a viewport's
/// pixels, scaled so their parameter is the view depth, as [`crate::pick::PickRay`] aims them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewProjector {
    /// The view.
    view: View,
}

impl ViewProjector {
    /// The projector of `view`.
    pub(crate) const fn new(view: &View) -> Self {
        Self { view: *view }
    }
}

impl Projector for ViewProjector {
    fn ray(&self, screen: ScreenPoint) -> Option<Ray> {
        let view = &self.view;
        let size = view.size();
        let (width, height) = (f64::from(size.width()), f64::from(size.height()));
        let per_unit = view.pixels_per_unit();
        let across = (screen.x() - width / 2.0) / per_unit;
        let down = (height / 2.0 - screen.y()) / per_unit;
        let direction = view.forward() + view.right() * across + view.up() * down;
        Ray::new(world(view.eye()), world(direction))
    }

    fn world_per_pixel(&self, depth: f64) -> f64 {
        depth / self.view.pixels_per_unit()
    }
}

/// A renderer's `f64` vector as the stroke engine's.
const fn world(v: render::camera::Vec3) -> DVec3 {
    DVec3::new(v.x, v.y, v.z)
}

/// A dab waiting to be applied, with the pen sample it came from, for the frame's latency.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Queued {
    /// The dab.
    dab: Dab,
    /// The sample whose push (or the pen-up's end) placed it.
    sample: SampleTimes,
}

/// The stroke in progress: the engine's stroke until the pen lifts, and its record.
#[derive(Debug)]
struct Active {
    /// The engine's stroke; `None` once it has ended (pen up, or an error), while its queued
    /// dabs are applied.
    stroke: Option<Stroke>,
    /// The record of its samples, settings and camera, written as it ends.
    record: StrokeRecord,
    /// The settings it began with: the brush's, its radius frozen at the first hit.
    settings: StrokeSettings,
}

/// What the warming under a hovering pen is doing.
#[derive(Debug, Default)]
struct Warm {
    /// Where the pen hovers now, and the view it is seen through: what to warm.
    wanted: Option<(Pixel, View)>,
    /// What is warm (or being warmed): a still pen is warmed once.
    target: Option<(Pixel, View)>,
    /// Tiles still to evaluate for `target`.
    todo: Vec<TileKey>,
}

/// The app's brush: pen strokes through the stroke engine into the store (see the module notes).
///
/// One brush for M1a4: the draw action with the spec's defaults, its size in screen pixels
/// frozen per stroke, its size and strength set with keys.
#[derive(Debug)]
pub(crate) struct Sculpting {
    /// The brush's settings: draw, the stroke engine's defaults, the radius and strength the keys
    /// set.
    settings: StrokeSettings,
    /// The default base strength, in model units, for the project's size.
    base_strength: f32,
    /// The strength, as a factor of `base_strength`.
    strength_factor: f32,
    /// The stroke state `apply_dabs` keeps across a stroke's dabs.
    state: sculpt::Stroke,
    /// The stroke in progress, if any.
    active: Option<Active>,
    /// Dabs placed and not yet applied, oldest first.
    queue: VecDeque<Queued>,
    /// How many dabs a frame applies.
    budget: DabBudget,
    /// The mean spacing of the index level's vertices, in world units, for footprints; zero
    /// before the first frame with an index.
    spacing: f64,
    /// The next stroke's number.
    next_stroke: u32,
    /// The epoch sample times count from.
    epoch: Instant,
    /// The warming under a hovering pen.
    warm: Warm,
    /// The last error, for the overlay.
    error: Option<String>,
    /// The last frame that applied dabs, for the overlay.
    last: BrushReport,
}

impl Sculpting {
    /// The brush for `project`: its default strength from the project's size
    /// ([`DEFAULT_STRENGTH_PER_SIZE`]), its stroke state from the store's reservation.
    pub(crate) fn new(project: &mut Project) -> Self {
        let store = project.store();
        let base_strength = default_strength(store);
        let state = sculpt::Stroke::new(store);
        let settings = StrokeSettings::default().with_strength(strength(base_strength, 1.0));
        Self {
            settings,
            base_strength,
            strength_factor: 1.0,
            state,
            active: None,
            queue: VecDeque::new(),
            budget: DabBudget::DEFAULT,
            spacing: 0.0,
            next_stroke: 0,
            epoch: Instant::now(),
            warm: Warm::default(),
            error: None,
            last: BrushReport::default(),
        }
    }

    /// This brush with `budget` dabs a frame.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn with_budget(mut self, budget: DabBudget) -> Self {
        self.budget = budget;
        self
    }

    /// This brush with the strength `factor` times the default.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_strength(mut self, factor: f32) -> Self {
        self.strength_factor = factor;
        self.settings = self
            .settings
            .with_strength(strength(self.base_strength, factor));
        self
    }

    /// The brush's settings.
    #[cfg(test)]
    pub(crate) const fn settings(&self) -> &StrokeSettings {
        &self.settings
    }

    /// The last error, if one happened.
    #[cfg(test)]
    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// This brush with its sample times counted from `epoch`: two brushes given the same samples
    /// and epoch place the same dabs, bit for bit.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn with_epoch(mut self, epoch: Instant) -> Self {
        self.epoch = epoch;
        self
    }

    /// What the last frame that applied dabs did.
    #[cfg(test)]
    pub(crate) const fn last(&self) -> BrushReport {
        self.last
    }

    /// Whether a stroke is in progress or its dabs are still being applied.
    #[cfg(test)]
    pub(crate) const fn stroking(&self) -> bool {
        self.active.is_some()
    }

    /// A pen sample as the stroke engine reads it: physical pixels, seconds since the brush's
    /// epoch on the pen's clock, and its pressure, tilt, twist and buttons; `None` if its
    /// position isn't finite.
    fn engine_sample(&self, sample: &PenSample) -> Option<Sample> {
        let time = match sample.timestamp.checked_duration_since(self.epoch) {
            Some(after) => after.as_secs_f64(),
            None => -self.epoch.duration_since(sample.timestamp).as_secs_f64(),
        };
        let position = ScreenPoint::new(sample.at.x, sample.at.y).ok()?;
        let pressure = Unit::saturating(sample.pressure.get());
        let mut engine = Sample::new(position, time, pressure).ok()?;
        if let Some(tilt) = sample.tilt.and_then(engine_tilt) {
            engine = engine.with_tilt(tilt);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a barrel rotation in radians: f32 holds it"
        )]
        let twist = sample
            .roll
            .and_then(|degrees| Twist::new(degrees.to_radians() as f32).ok());
        if let Some(twist) = twist {
            engine = engine.with_twist(twist);
        }
        Some(engine.with_buttons(stroke::Buttons::from_bits(button_bits(sample.buttons))))
    }

    /// Ends the stroke in progress, if any, at once: its engine stroke ended and its tail
    /// queued, every dab queued applied whatever the budget, and its op ended with its record.
    fn finish_now(&mut self, context: &mut Scene<'_>, report: &mut BrushReport) {
        self.lift(context, None);
        let count = self.queue.len();
        self.apply(context, report, count);
        self.end_if_done(context.project);
    }

    /// Starts a stroke at `down` (routed to the brush), if the engine finds the surface under it.
    fn begin(&mut self, context: &mut Scene<'_>, down: &PenSample) {
        self.warm = Warm::default();
        let Some(first) = self.engine_sample(down) else {
            return;
        };
        let projector = context.projector;
        let id = StrokeId::new(self.next_stroke);
        self.next_stroke = self.next_stroke.wrapping_add(1);
        let mut surface = StoreSurface::new(context.project.store(), context.index);
        let stroke = Stroke::begin(&self.settings, id, &projector, &mut surface, first);
        if let Err(error) = surface.finish() {
            self.fail("starting a stroke", &error);
            return;
        }
        let Some(stroke) = stroke else {
            return;
        };
        let op = match context.project.begin_stroke() {
            Ok(op) => op,
            Err(error) => {
                self.fail("starting a stroke", &error);
                return;
            }
        };
        let camera = stroke_camera(context.camera, &projector.view);
        let record =
            StrokeRecord::new(op, stroke::ENGINE_VERSION, self.settings.to_bytes(), camera);
        let mut record = match record {
            Ok(record) => record,
            Err(error) => {
                self.fail("recording a stroke", &error);
                return;
            }
        };
        self.keep_sample(&mut record, &first);
        self.active = Some(Active {
            stroke: Some(stroke),
            record,
            settings: self.settings.clone(),
        });
    }

    /// Feeds `sample` (a `Drag` of the stroke in progress) to the engine, queueing the dabs it
    /// places; a store error ends the stroke there, its dabs dropped.
    fn push(&mut self, context: &mut Scene<'_>, sample: &PenSample) {
        let Some(engine) = self.engine_sample(sample) else {
            return;
        };
        let times = SampleTimes {
            taken: sample.timestamp,
            received: sample.received,
        };
        let Some(active) = &mut self.active else {
            return;
        };
        let Some(stroke) = &mut active.stroke else {
            return;
        };
        let projector = context.projector;
        let mut surface = StoreSurface::new(context.project.store(), context.index);
        let placed = stroke.push(&projector, &mut surface, engine);
        let finished = surface.finish();
        keep_sample(&mut active.record, &engine, &mut self.error);
        match finished {
            Ok(()) => {
                let reaching = placed.filter(reaches);
                let queued = reaching.map(|dab| Queued { dab, sample: times });
                self.queue.extend(queued);
            }
            Err(error) => {
                drop(placed);
                active.stroke = None;
                self.fail("placing dabs", &error);
            }
        }
    }

    /// Ends the engine's stroke (the pen lifted, at `up` if its sample came), queueing its tail.
    fn lift(&mut self, context: &mut Scene<'_>, up: Option<&PenSample>) {
        let Some(stroke) = self.active.as_mut().and_then(|active| active.stroke.take()) else {
            return;
        };
        let times = up.map_or_else(
            || {
                let now = Instant::now();
                SampleTimes {
                    taken: now,
                    received: now,
                }
            },
            |up| SampleTimes {
                taken: up.timestamp,
                received: up.received,
            },
        );
        let projector = context.projector;
        let mut surface = StoreSurface::new(context.project.store(), context.index);
        let tail = stroke.end(&projector, &mut surface);
        match surface.finish() {
            Ok(()) => {
                let reaching = tail.filter(reaches);
                let queued = reaching.map(|dab| Queued { dab, sample: times });
                self.queue.extend(queued);
            }
            Err(error) => self.fail("placing the stroke's last dabs", &error),
        }
    }

    /// Applies the first `count` queued dabs in one [`apply_dabs`], adding to `report`. A store
    /// error drops every queued dab and ends the engine's stroke.
    fn apply(&mut self, context: &mut Scene<'_>, report: &mut BrushReport, count: usize) {
        let count = count.min(self.queue.len());
        if count == 0 {
            return;
        }
        let batch: Vec<Dab> = self.queue.iter().take(count).map(|q| q.dab).collect();
        let sources = self.queue.iter().take(count).map(|q| q.sample);
        let oldest = sources.reduce(SampleTimes::earliest);
        report.oldest = match (report.oldest, oldest) {
            (Some(a), Some(b)) => Some(a.earliest(b)),
            (a, b) => a.or(b),
        };
        let footprints = batch.iter().map(|dab| footprint(dab.radius, self.spacing));
        report.above_limit |= footprints.fold(0.0, f64::max) > LIMIT_VERTICES;
        let applied = apply_dabs(
            context.project.store(),
            context.index,
            &mut self.state,
            &batch,
        );
        match applied {
            Ok(stats) => {
                self.queue.drain(..count);
                report.dabs_applied += stats.dabs;
                report.tiles_edited += stats.tiles_edited;
            }
            Err(error) => {
                self.queue.clear();
                if let Some(active) = &mut self.active {
                    active.stroke = None;
                }
                self.fail("applying dabs", &error);
            }
        }
    }

    /// Ends the stroke's op with its record once its engine stroke has ended and its dabs are
    /// all applied.
    fn end_if_done(&mut self, project: &mut Project) {
        let done = self
            .active
            .as_ref()
            .is_some_and(|active| active.stroke.is_none() && self.queue.is_empty());
        if !done {
            return;
        }
        let Some(active) = self.active.take() else {
            return;
        };
        if let Err(error) = project.end_stroke(&active.record) {
            self.fail("ending the stroke", &error);
        }
    }

    /// Keeps `error`, from `doing`, for the overlay.
    fn fail(&mut self, doing: &str, error: &dyn std::error::Error) {
        let causes = std::iter::successors(Some(error), |error| error.source());
        let chain: Vec<String> = causes.map(ToString::to_string).collect();
        self.error = Some(format!("{doing}: {}", chain.join(": ")));
    }

    /// Adds `sample` to `record`, keeping the error if the record is full.
    fn keep_sample(&mut self, record: &mut StrokeRecord, sample: &Sample) {
        keep_sample(record, sample, &mut self.error);
    }

    /// The radius in screen pixels, scaled by `factor` within [`RADIUS_PIXELS`].
    fn scale_radius(&mut self, factor: f32) -> bool {
        let (least, most) = RADIUS_PIXELS;
        self.set_radius_pixels((self.settings.radius().get() * factor).clamp(least, most))
    }

    /// Sets the radius to `pixels`; returns whether it changed.
    fn set_radius_pixels(&mut self, pixels: f32) -> bool {
        let Ok(radius) = Radius::new(pixels) else {
            return false;
        };
        if radius == self.settings.radius() {
            return false;
        }
        let Ok(settings) = StrokeSettings::new(
            *self.settings.path(),
            self.settings.spacing(),
            self.settings.size_mode(),
            radius,
            self.settings.dynamics().clone(),
            self.settings.symmetry(),
            self.settings.action_params(),
        ) else {
            return false;
        };
        self.settings = settings
            .with_strength(self.settings.strength())
            .with_hardness(self.settings.hardness());
        true
    }

    /// The strength factor scaled by `factor` within [`STRENGTH_FACTORS`]; returns whether it
    /// changed.
    fn scale_strength(&mut self, factor: f32) -> bool {
        let (least, most) = STRENGTH_FACTORS;
        let scaled = (self.strength_factor * factor).clamp(least, most);
        if scaled.total_cmp(&self.strength_factor).is_eq() {
            return false;
        }
        self.strength_factor = scaled;
        self.settings = self
            .settings
            .clone()
            .with_strength(strength(self.base_strength, scaled));
        true
    }

    /// Notes where the pen hovers in this frame's view, for [`Brush::prepare`] to warm.
    fn note_hover(&mut self, pen: &PenFrame, view: Option<&View>) {
        self.warm.wanted = match (pen.hover(), view, &self.active) {
            (Some(hover), Some(view), None) => Some((hover.at, *view)),
            _ => None,
        };
        if self.warm.wanted != self.warm.target {
            // Moved on (or touched, or left): what was left of the last warming is dropped.
            self.warm.todo.clear();
        }
    }

    /// Starts warming the hover's `wanted` position: raycasts there and lists the tiles within
    /// reach of the hit that aren't evaluated yet.
    fn start_warming(&mut self, context: &mut Scene<'_>, at: Pixel) {
        let projector = context.projector;
        let Ok(screen) = ScreenPoint::new(at.x, at.y) else {
            return;
        };
        let Some(ray) = projector.ray(screen) else {
            return;
        };
        let mut surface = StoreSurface::new(context.project.store(), context.index);
        let hit = surface.raycast(&ray);
        if let Err(error) = surface.finish() {
            self.fail("warming the surface under the pen", &error);
            return;
        }
        let Some(hit) = hit else {
            return;
        };
        let depth = ray.depth_of(hit.position);
        let radius = f64::from(self.settings.radius().get()) * projector.world_per_pixel(depth);
        let reach = REACH_RADII * radius;
        self.warm.todo = cold_tiles(context.project.store(), context.index, &hit, reach);
    }
}

/// What the brush works with in one call: the context's parts, the camera's projector made.
#[derive(Debug)]
struct Scene<'a> {
    /// The project.
    project: &'a mut Project,
    /// Its spatial index.
    index: &'a mut SpatialIndex,
    /// The camera.
    camera: &'a Camera,
    /// The camera's projector, for the frame's size.
    projector: ViewProjector,
}

impl<'a> Scene<'a> {
    /// `context`'s scene; `None` with no index (nothing to brush) or no view (a degenerate
    /// camera).
    fn of(context: BrushContext<'a>) -> Option<Self> {
        let BrushContext {
            project,
            index,
            camera,
            size,
        } = context;
        Some(Self {
            project,
            index: index?,
            projector: ViewProjector::new(&camera.view(size)?),
            camera,
        })
    }
}

impl Brush for Sculpting {
    fn step(&mut self, context: BrushContext<'_>, pen: &PenFrame) -> Result<BrushReport> {
        let view = context.camera.view(context.size);
        let mut report = BrushReport::default();
        let Some(mut scene) = Scene::of(context) else {
            self.note_hover(pen, None);
            return Ok(report);
        };
        if self.spacing == 0.0 {
            self.spacing = mean_spacing(scene.project.store(), scene.index.level());
        }
        let samples = pen.for_brush();
        for sample in samples {
            match sample.phase {
                Phase::Down => {
                    self.finish_now(&mut scene, &mut report);
                    self.begin(&mut scene, sample);
                }
                Phase::Drag => self.push(&mut scene, sample),
                Phase::Up => self.lift(&mut scene, Some(sample)),
                // Out of a stroke: a stroke whose `Up` was lost ends here.
                Phase::Enter | Phase::Hover | Phase::Leave => self.lift(&mut scene, None),
            }
        }
        let footprints = self
            .queue
            .iter()
            .map(|queued| footprint(queued.dab.radius, self.spacing));
        let count = self.budget.take(footprints);
        self.apply(&mut scene, &mut report, count);
        self.end_if_done(scene.project);
        self.note_hover(pen, view.as_ref());
        report.dabs_carried = self.queue.len();
        report.busy = !self.queue.is_empty();
        if report.dabs_applied > 0 {
            self.last = report;
        }
        Ok(report)
    }

    fn route(&mut self, context: BrushContext<'_>, down: &PenSample) -> Route {
        let Some(scene) = Scene::of(context) else {
            return Route::Navigate;
        };
        let Some(ray) = ScreenPoint::new(down.at.x, down.at.y)
            .ok()
            .and_then(|screen| scene.projector.ray(screen))
        else {
            return Route::Navigate;
        };
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let hit = surface.raycast(&ray);
        if let Err(error) = surface.finish() {
            self.fail("picking under the pen", &error);
            return Route::Navigate;
        }
        if hit.is_some() {
            Route::Brush
        } else {
            Route::Navigate
        }
    }

    fn prepare(&mut self, context: BrushContext<'_>, deadline: Instant) -> Result<bool> {
        let Some(mut scene) = Scene::of(context) else {
            return Ok(false);
        };
        if self.warm.wanted != self.warm.target {
            self.warm.target = self.warm.wanted;
            if let Some((at, _)) = self.warm.wanted {
                self.start_warming(&mut scene, at);
            }
        }
        while !self.warm.todo.is_empty() && Instant::now() < deadline {
            let split = self.warm.todo.len().saturating_sub(WARM_BATCH);
            let batch = self.warm.todo.split_off(split);
            let evaluated = scene
                .project
                .store()
                .evaluate_many(&batch, Evaluation::Positions);
            if let Err(error) = evaluated {
                self.warm.todo.clear();
                self.fail("warming the surface under the pen", &error);
            }
        }
        Ok(!self.warm.todo.is_empty())
    }

    fn flush(&mut self, context: BrushContext<'_>) -> Result<()> {
        if let Some(mut scene) = Scene::of(context) {
            self.finish_now(&mut scene, &mut BrushReport::default());
        }
        Ok(())
    }

    fn change(&mut self, change: BrushChange) -> bool {
        match change {
            BrushChange::Smaller => self.scale_radius(SIZE_STEP.recip()),
            BrushChange::Larger => self.scale_radius(SIZE_STEP),
            BrushChange::Weaker => self.scale_strength(STRENGTH_STEP.recip()),
            BrushChange::Stronger => self.scale_strength(STRENGTH_STEP),
        }
    }

    fn set_radius(&mut self, pixels: f32) {
        self.set_radius_pixels(pixels);
    }

    fn status(&self) -> Vec<String> {
        let settings = self
            .active
            .as_ref()
            .map_or(&self.settings, |active| &active.settings);
        let mut lines = vec![format!(
            "brush      draw  radius {:.1} px  strength {:.5} (x{:.2})  keys: [ ] size, - = \
             strength",
            settings.radius().get(),
            settings.strength().get(),
            self.strength_factor
        )];
        let last = self.last;
        lines.push(format!(
            "stroke     {}  last frame with dabs: {} applied, {} carried, {} tiles edited{}",
            if self.active.is_some() {
                "drawing"
            } else {
                "none"
            },
            last.dabs_applied,
            last.dabs_carried,
            last.tiles_edited,
            if last.above_limit { ", above F" } else { "" }
        ));
        if let Some(error) = &self.error {
            lines.push(format!("brush err  {error}"));
        }
        lines
    }
}

/// Whether `dab` reaches any vertex: its radius is positive.
///
/// The engine scales a dab's radius by its dynamics' size factor, so a sample with no pressure
/// places a dab of radius 0 (the default pressure-to-size curve starts at 0). It moves nothing, and
/// [`apply_dabs`] refuses a frame holding one, so it is left out: leaving it out is applying it.
fn reaches(dab: &Dab) -> bool {
    dab.radius > 0.0
}

/// `sample` added to `record`; a full record leaves it out and keeps the error in `error`.
fn keep_sample(record: &mut StrokeRecord, sample: &Sample, error: &mut Option<String>) {
    if let Err(full) = record.push(&record_sample(sample)) {
        *error = Some(format!("recording the stroke: {full}"));
    }
}

/// The engine's sample as a stroke record keeps it.
fn record_sample(sample: &Sample) -> RecordSample {
    RecordSample {
        x: sample.position().x(),
        y: sample.position().y(),
        time: sample.time(),
        pressure: sample.pressure().get(),
        tilt: sample.tilt().map(|tilt| PenTilt {
            amount: tilt.amount(),
            direction: tilt.direction(),
        }),
        twist: sample.twist().map(Twist::get),
        buttons: sample.buttons().bits(),
    }
}

/// The pen's tilt as the engine takes it: the angle from upright, and the way it leans.
fn engine_tilt(tilt: input::Tilt) -> Option<Tilt> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "angles in radians: f32 holds them"
    )]
    let (amount, direction) = (
        (FRAC_PI_2 - tilt.altitude()).clamp(0.0, FRAC_PI_2) as f32,
        tilt.azimuth() as f32,
    );
    Tilt::new(amount.min(std::f32::consts::FRAC_PI_2), direction).ok()
}

/// The bits of the barrel buttons held, as the engine's [`stroke::Buttons::from_bits`] reads
/// them.
fn button_bits(buttons: input::Buttons) -> u8 {
    let named = [
        (input::Buttons::PRIMARY, stroke::Buttons::PRIMARY),
        (input::Buttons::SECONDARY, stroke::Buttons::SECONDARY),
        (input::Buttons::TERTIARY, stroke::Buttons::TERTIARY),
    ];
    let held = named
        .into_iter()
        .filter(|(pen, _)| buttons.contains(*pen))
        .map(|(_, engine)| engine.bits());
    held.fold(0, |bits, bit| bits | bit)
}

/// The camera a stroke was drawn through, as its record keeps it: `camera`'s eye, target, up and
/// field of view, and `view`'s size.
fn stroke_camera(camera: &Camera, view: &View) -> StrokeCamera {
    let triple = |v: crate::camera::Vec3| [v.x, v.y, v.z];
    let size = view.size();
    StrokeCamera {
        eye: triple(camera.eye()),
        target: triple(camera.target()),
        up: triple(camera.axis(crate::camera::Vec3::new(0.0, 1.0, 0.0))),
        fov_y: camera.fov_y(),
        size: [size.width(), size.height()],
    }
}

/// The brush's base strength, `factor` times `base`; the default if that isn't finite.
fn strength(base: f32, factor: f32) -> Strength {
    Strength::new(base * factor).unwrap_or_default()
}

/// The default base strength for `store`'s project.
///
/// [`DEFAULT_STRENGTH_PER_SIZE`] of its base mesh's bounding radius (the farthest vertex from its
/// box's centre: 1 for the unit sphere); the engine's default for an empty mesh.
fn default_strength(store: &TileStore) -> f32 {
    let mesh = store.mesh();
    let points = || {
        mesh.vertices()
            .filter_map(|v| mesh.vertex_position(v))
            .map(|p| DVec3::new(p.x.into(), p.y.into(), p.z.into()))
    };
    let bounds = points().fold(None, |bounds: Option<(DVec3, DVec3)>, p| {
        Some(bounds.map_or((p, p), |(low, high)| {
            let low = DVec3::new(low.x.min(p.x), low.y.min(p.y), low.z.min(p.z));
            let high = DVec3::new(high.x.max(p.x), high.y.max(p.y), high.z.max(p.z));
            (low, high)
        }))
    });
    let Some((low, high)) = bounds else {
        return Strength::DEFAULT.get();
    };
    let centre = (low + high) * 0.5;
    let radius = points().map(|p| p.distance(centre)).fold(0.0, f64::max);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a strength in model units: f32 holds it"
    )]
    let base = (DEFAULT_STRENGTH_PER_SIZE * radius) as f32;
    if base.is_finite() && base > 0.0 {
        base
    } else {
        Strength::DEFAULT.get()
    }
}

/// The mean spacing of `level`'s vertices over `store`'s base mesh.
///
/// The mean length of the base faces' edges over the quads a face's side holds at `level`: a
/// footprint estimate, not exact on a curved or uneven mesh.
fn mean_spacing(store: &TileStore, level: Level) -> f64 {
    let mesh = store.mesh();
    let corners = |face| {
        let corner = |c| {
            mesh.corner_vertex(face, c)
                .and_then(|v| mesh.vertex_position(v))
        };
        [corner(0), corner(1), corner(2), corner(3)]
    };
    let edges = mesh.faces().flat_map(|face| {
        let [a, b, c, d] = corners(face);
        [(a, b), (b, c), (c, d), (d, a)]
    });
    let lengths = edges.filter_map(|(a, b)| Some(f64::from((a? - b?).length())));
    let (sum, count) = lengths.fold((0.0, 0_u32), |(sum, count), l| (sum + l, count + 1));
    if count == 0 {
        return 0.0;
    }
    sum / f64::from(count) / f64::from(level.face_quads())
}

/// The vertices a dab of `radius` covers over vertices `spacing` apart: `π (r / h)²`, at least
/// one. With no spacing known, one.
fn footprint(radius: f32, spacing: f64) -> f64 {
    if spacing > 0.0 {
        (PI * (f64::from(radius) / spacing).powi(2)).max(1.0)
    } else {
        1.0
    }
}

/// The tiles warming evaluates: those of `index`'s level not evaluated in `store` whose boxes
/// touch the ball of `reach` around `hit`.
///
/// In the index's order reversed: warming takes its batches from the end, so the index's first
/// come first.
fn cold_tiles(store: &TileStore, index: &SpatialIndex, hit: &Hit, reach: f64) -> Vec<TileKey> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the index's query is f32: a tile it misses by rounding is evaluated by the \
                  stroke when it reaches it"
    )]
    let radius = reach as f32;
    let center = hit.position.to_f32();
    let found = index
        .query_sphere(center, radius)
        .filter(|key| store.cached_positions(key).is_none());
    let mut tiles: Vec<TileKey> = found.collect();
    tiles.reverse();
    tiles
}
