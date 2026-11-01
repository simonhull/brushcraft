//! Sculpting with the pen: the app's [`Brush`], strokes through the stroke engine into the store,
//! shown in the same frame (the brush-engine spec, §3 "In the app").
//!
//! # A stroke
//!
//! - **Routing** ([`Brush::route`]): a pen `Down` whose ray hits the surface is the brush's stroke;
//!   off the surface it navigates, so no stroke starts off the surface. The ray is the one the
//!   stroke engine casts ([`ViewProjector`]), through the store's spatial index, and its hit is
//!   kept for [`Stroke::begin`], so a `Down` raycasts once.
//! - **Pen down:** [`Stroke::begin`] with the camera's projector and a [`StoreSurface`], which
//!   freezes the brush's world radius at the first hit (the size is in screen pixels), and with it
//!   a strength in radii ([`StrengthMode::Radius`]). A `Down` the engine finds nothing under (the
//!   route's hit lost to a moved camera) starts no stroke, and its samples are dropped.
//! - **Each sample:** [`Stroke::push`], through a fresh [`StoreSurface`] each call, ended with
//!   [`StoreSurface::finish`]; the dabs it places join the stroke's queue. A sample with no
//!   pressure places none: the engine places no dab below [`stroke::MIN_DAB_SIZE`].
//! - **Each frame:** the queued dabs are applied by [`apply_dabs`], oldest stroke first, as many as
//!   the frame's budget allows ([`DabBudget`]); the rest wait for the next frame. The edit closes
//!   the store's epoch, which the frame loop drains after the brush phase and uploads in the same
//!   frame.
//! - **The op:** a stroke's op begins ([`Project::begin_stroke`]) when it is the oldest stroke
//!   left, so every edit of the store's current op is that stroke's: a stroke started while the
//!   last still has dabs queued waits behind them, its dabs placed but not applied, and its op
//!   begins once the last one's has ended.
//! - **Pen up:** [`Stroke::end`] gives the tail; once every dab of the stroke is applied, its
//!   record ([`StrokeRecord`]: the samples the engine read, the settings' bytes, the camera at the
//!   stroke's start, [`stroke::ENGINE_VERSION`]) ends the op ([`Project::end_stroke`]). A stroke
//!   whose `Up` was lost ends at the next sample out of a stroke, or as another stroke starts (one
//!   the brush doesn't take included, [`ForBrush::StrokeElsewhere`]); its tail's time is its last
//!   sample's.
//!
//! A sample's time is seconds from a fixed epoch (the brush's creation) on the pen's
//! de-jittered clock.
//!
//! # The budget: dabs per frame
//!
//! A frame's dabs cost [`apply_dabs`] about linearly in the tiles they touch: each dab re-evaluates
//! every tile its sphere reaches, whole, and pays a query and a dispatch to the workers besides.
//! So each dab is costed as the tiles its sphere touches in the index (queried as it is queued)
//! times a tile's [`TILE_VERTICES`], plus a fixed overhead, and a frame applies dabs while their
//! summed cost fits [`DabBudget::DEFAULT`], and always at least one. The rest are carried to the
//! next frame, in order: no dab is dropped, and the stroke's tail waits behind them. A frame that
//! carries dabs wants the next.
//!
//! The footprint limit F (the M1 spec's "Big brushes": about 150 tiles a step, [`LIMIT_TILES`]) is
//! reported per frame ([`BrushReport::above_limit`]): the latency target holds below it; above it
//! is the big-brush path, still to come.
//!
//! A dab is placed on the surface as it is when the dab is placed: the dabs queued ahead of it
//! haven't landed yet. Within a frame that was always so (a frame's dabs apply after all its
//! samples are placed); carrying makes it so for longer, so a carried dab may be placed on a
//! surface its predecessors have yet to raise.
//!
//! # Warming under a hovering pen
//!
//! A stroke's first sample finds the surface under it cold: evaluating the reach ball's tiles
//! costs up to 6.8 ms for `closest_on_region` and 2.4 ms for the raycast (Task 3). So while the
//! pen hovers, the brush warms the region under it, once per position and view: each frame's step
//! notes where it hovers ([`PenFrame::hover`]), and [`Brush::prepare`], after the frame's
//! hand-off and so off the latency path, raycasts there and evaluates the tiles within the reach
//! ([`stroke::REACH_RADII`] brush radii) of the hit, [`WARM_BATCH`] at a time on the store's
//! workers, until the frame's deadline. Each call does at least one piece of that (the raycast
//! and the tiles' list, or a batch), so it finishes even in frames that overrun their deadline.
//! What is left goes on in the next frame's `prepare`; a pen that moves on, touches or leaves
//! cancels it. A still pen is warmed once.
//!
//! # Errors
//!
//! A store error ends the stroke cleanly (the brush-engine spec, §3 "Errors"), and shows in the
//! overlay ([`Brush::status`]) while the app goes on:
//!
//! - a push that ended in an error has its dabs dropped; the dabs queued before it are still
//!   applied, and the op ends with the stroke's record;
//! - a failed [`apply_dabs`] drops the stroke's dabs left (its own are applied up to the failing
//!   one, and applying them again would count some twice), and the spatial index is rebuilt from
//!   the bounds the project keeps (`apply_dabs` may have left it stale), retried each frame until
//!   it loads;
//! - a record [`Project::end_stroke`] refuses ([`ProjectError::Cache`],
//!   [`ProjectError::NotTheStroke`], or one that can't be made) ends the stroke unrecorded
//!   ([`Project::end_unrecorded_stroke`]), so its edits are still flushed.

use std::collections::VecDeque;
use std::f64::consts::FRAC_PI_2;
use std::time::Instant;

use anyhow::Result;
use geom::TileKey;
use input::Phase;
use render::camera::View;
use sculpt::{StoreSurface, apply_dabs};
use store::{
    Evaluation, OpId, PenSample as RecordSample, PenTilt, Project, ProjectError, SpatialIndex,
    StrokeCamera, StrokeRecord, StrokeRecordError, TileStore,
};
use stroke::{
    DVec3, Dab, Hit, Projector, REACH_RADII, Radius, Ray, Sample, ScreenPoint, Strength,
    StrengthMode, Stroke, StrokeId, StrokeSettings, Surface, Tilt, Twist, Unit,
};

use crate::camera::{Camera, Pixel};
use crate::frame::{Brush, BrushChange, BrushContext, BrushReport, SampleTimes};
use crate::pen::{ForBrush, PenFrame, PenSample, Route};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The default base strength, as a fraction of the stroke's frozen world radius
/// ([`StrengthMode::Radius`]): a tenth.
///
/// A stroke then raises the surface by the same share of its width at any zoom. At spacing 0.15
/// radii a stroke's centre line sums about `p / 0.15` dabs' worth (pressure `p` scales both a
/// dab's size, so how many overlap, and its strength), so the ridge stands about `0.1 · p² /
/// 0.15` radii: a sixth of the radius at half pressure, two thirds at full. The keys scale it
/// ([`STRENGTH_STEP`]).
pub(crate) const DEFAULT_STRENGTH_FRACTION: f32 = 0.1;

/// How much one press of a strength key scales the strength.
pub(crate) const STRENGTH_STEP: f32 = 1.25;

/// The strength's range, as factors of the default: a 64th to 64 times (0.0016 to 6.4 radii).
pub(crate) const STRENGTH_FACTORS: (f32, f32) = (1.0 / 64.0, 64.0);

/// How much one press of a size key scales the brush's radius.
pub(crate) const SIZE_STEP: f32 = 1.25;

/// The brush radius's range, in physical pixels.
pub(crate) const RADIUS_PIXELS: (f32, f32) = (1.0, 2000.0);

/// The footprint limit F, as the tiles a dab touches: the M1 spec's "Big brushes" (about 150
/// tiles a step, about 180,000 vertices). The pen's latency target holds below it.
pub(crate) const LIMIT_TILES: usize = 150;

/// The vertices a tile holds, `65²` at any level ([`geom::TILE_QUADS`] + 1 a side): what a dab
/// re-evaluates for each tile its sphere touches.
pub(crate) const TILE_VERTICES: f64 = 4225.0;

/// Tiles the warming evaluates together on the store's workers between checks of its deadline.
pub(crate) const WARM_BATCH: usize = 16;

/// How many dabs a frame applies: as many as fit a budget, each dab costed by the tiles it
/// touches plus a fixed overhead, and always at least one (see "The budget" in the module notes).
///
/// Costs are in tile vertices: a dab touching `t` tiles costs `t ·` [`TILE_VERTICES`] plus the
/// overhead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DabBudget {
    /// The most a frame's dabs cost together.
    units: f64,
    /// What every dab costs besides its tiles: its query, a dispatch to the workers.
    per_dab: f64,
}

impl DabBudget {
    /// 160,000 tile vertices a frame, and two tiles' worth (8,450) a dab besides.
    ///
    /// Sized against Task 3's p99, not its median: four dabs of 50,000 vertices (each touching
    /// about 24 tiles, about 100,000 tile vertices) took 7.5 ms a call at p99, about 18 ns a tile
    /// vertex, so the budget is about 3 ms at p99, leaving the rest of the pen's 8 ms to the
    /// frame's input, feed, uploads and encoding (3 to 4 ms of uploads and encoding at p99 in the
    /// first scripted run). Both constants are provisional: the overhead is a guess until
    /// `brushcraft-bench brush --dabs 32 --footprint small` (many small dabs) and `--dabs 4
    /// --footprint 50000` calibrate them.
    pub(crate) const DEFAULT: Self = Self {
        units: 160_000.0,
        per_dab: 2.0 * TILE_VERTICES,
    };

    /// A budget of `units` tile vertices a frame, and `per_dab` a dab besides its tiles.
    #[cfg(test)]
    pub(crate) const fn new(units: f64, per_dab: f64) -> Self {
        Self { units, per_dab }
    }

    /// What a dab touching `tiles` tiles costs.
    pub(crate) fn cost(self, tiles: usize) -> f64 {
        let count = u32::try_from(tiles).map_or_else(|_| f64::from(u32::MAX), f64::from);
        count * TILE_VERTICES + self.per_dab
    }

    /// How many of `costs` (queued dabs' costs, in order) fit `room`, the budget the frame has
    /// left, which they use up: the longest run from the front that fits, and at least one if
    /// `at_least_one` and there are any.
    pub(crate) fn take(
        costs: impl IntoIterator<Item = f64>,
        room: &mut f64,
        at_least_one: bool,
    ) -> usize {
        let mut count = 0;
        for cost in costs {
            if cost > *room && !(at_least_one && count == 0) {
                break;
            }
            *room -= cost;
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

/// A surface whose first raycast is answered already: the route's hit, so a `Down` raycasts once.
///
/// The kept hit answers only a ray equal to its own, bit for bit (the same pixel, the same view);
/// any other, and every later call, goes to the surface.
struct Primed<'s, S> {
    /// The surface.
    surface: &'s mut S,
    /// The route's ray and its hit, until the first raycast.
    first: Option<(Ray, Hit)>,
}

impl<S: Surface> Surface for Primed<'_, S> {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        match self.first.take() {
            Some((routed, hit)) if routed == *ray => Some(hit),
            _ => self.surface.raycast(ray),
        }
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        self.surface.closest_on_region(ray, near, reach)
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        self.surface.continuous(a, b, reach)
    }
}

/// Where a queued dab came from, and what it costs.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Source {
    /// The sample whose push (or the pen-up's end) placed it: the frame's latency runs from it.
    times: SampleTimes,
    /// The tiles its sphere touches, as it was queued.
    tiles: usize,
    /// Its cost against the frame's budget.
    cost: f64,
}

/// A stroke, from its `Down` until its op ends: its engine stroke while the pen is down, what its
/// record keeps, and its dabs not yet applied.
#[derive(Debug)]
struct Pending {
    /// The engine's stroke; `None` once it has ended (pen up, or an error), while its queued
    /// dabs are applied.
    engine: Option<Stroke>,
    /// The brush's settings as the stroke began, which its record keeps (the engine holds its own
    /// copy, its radius frozen at the first hit).
    settings: StrokeSettings,
    /// The camera at the stroke's start, which its record keeps.
    camera: StrokeCamera,
    /// The samples the engine read, which its record keeps.
    samples: Vec<RecordSample>,
    /// Its op, once begun: once it is the oldest stroke left.
    op: Option<OpId>,
    /// Its dabs not yet applied, oldest first.
    dabs: VecDeque<Dab>,
    /// Where each of `dabs` came from, and its cost.
    sources: VecDeque<Source>,
    /// Its last sample's times: a lost `Up`'s tail is timed from it.
    last: SampleTimes,
}

impl Pending {
    /// Whether the pen still draws it.
    const fn open(&self) -> bool {
        self.engine.is_some()
    }

    /// Whether it is over: the pen lifted (or an error ended it) and every dab is applied.
    fn done(&self) -> bool {
        self.engine.is_none() && self.dabs.is_empty()
    }
}

/// Queues `placed` on a stroke's `queue`: dabs placed by the sample at `times`, each costed by
/// `budget` from the tiles of `index` its sphere touches.
fn enqueue(
    (dabs, sources): (&mut VecDeque<Dab>, &mut VecDeque<Source>),
    placed: impl Iterator<Item = Dab>,
    times: SampleTimes,
    index: &SpatialIndex,
    budget: DabBudget,
) {
    for dab in placed {
        let [x, y, z] = dab.position;
        let touched = index
            .query_sphere(geom::Vec3::new(x, y, z), dab.radius)
            .count();
        dabs.push_back(dab);
        sources.push_back(Source {
            times,
            tiles: touched,
            cost: budget.cost(touched),
        });
    }
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

/// What the overlay's lines say: they are made again only when it changes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StatusKey {
    /// The radius, in pixels.
    radius: f32,
    /// The strength, in radii.
    strength: f32,
    /// The strength's factor of the default.
    factor: f32,
    /// Whether a stroke is pending.
    drawing: bool,
    /// The last frame that applied dabs.
    last: BrushReport,
    /// How many errors have been kept (or cleared).
    errors: u64,
}

/// The app's brush: pen strokes through the stroke engine into the store (see the module notes).
///
/// One brush for M1a4: the draw action with the spec's defaults, its size in screen pixels
/// frozen per stroke, its strength a share of that size, both set with keys.
#[derive(Debug)]
pub(crate) struct Sculpting {
    /// The brush's settings: draw, the stroke engine's defaults, the radius and strength the keys
    /// set, the strength in radii.
    settings: StrokeSettings,
    /// The strength, as a factor of [`DEFAULT_STRENGTH_FRACTION`].
    strength_factor: f32,
    /// The stroke state `apply_dabs` keeps across a stroke's dabs.
    state: sculpt::Stroke,
    /// The strokes not over yet, oldest first: only the last can be open.
    strokes: VecDeque<Pending>,
    /// How many dabs a frame applies.
    budget: DabBudget,
    /// The next stroke's number.
    next_stroke: u32,
    /// The epoch sample times count from.
    epoch: Instant,
    /// The ray and hit of the last `Down` routed to the brush, for its stroke's first raycast;
    /// cleared at the end of each step.
    routed: Option<(Ray, Hit)>,
    /// Pen samples held over from a step with no scene (a degenerate view) while strokes were
    /// pending, in order; `None` for a stroke started elsewhere.
    held: Vec<Option<PenSample>>,
    /// Whether the spatial index must be rebuilt: an `apply_dabs` failed, and its rebuild hasn't
    /// succeeded yet.
    index_stale: bool,
    /// The warming under a hovering pen.
    warm: Warm,
    /// The last error, for the overlay; cleared as a stroke starts cleanly.
    error: Option<String>,
    /// Errors kept or cleared so far, for the overlay to see a change.
    errors: u64,
    /// The last frame that applied dabs, for the overlay.
    last: BrushReport,
    /// The overlay's lines, and what they were made from.
    status: (Vec<String>, Option<StatusKey>),
}

impl Sculpting {
    /// The brush for `project`: its strength [`DEFAULT_STRENGTH_FRACTION`] of the stroke's
    /// radius, its stroke state from the store's reservation.
    pub(crate) fn new(project: &mut Project) -> Self {
        let state = sculpt::Stroke::new(project.store());
        let settings = StrokeSettings::default()
            .with_strength_mode(StrengthMode::Radius)
            .with_strength(strength(1.0));
        let mut brush = Self {
            settings,
            strength_factor: 1.0,
            state,
            strokes: VecDeque::new(),
            budget: DabBudget::DEFAULT,
            next_stroke: 0,
            epoch: Instant::now(),
            routed: None,
            held: Vec::new(),
            index_stale: false,
            warm: Warm::default(),
            error: None,
            errors: 0,
            last: BrushReport::default(),
            status: (Vec::new(), None),
        };
        brush.refresh_status();
        brush
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
        self.settings = self.settings.with_strength(strength(factor));
        self.refresh_status();
        self
    }

    /// The brush's settings.
    #[cfg(test)]
    pub(crate) const fn settings(&self) -> &StrokeSettings {
        &self.settings
    }

    /// The last error, if one happened since a stroke last started cleanly.
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
    pub(crate) fn stroking(&self) -> bool {
        !self.strokes.is_empty()
    }

    /// The strokes not over yet.
    #[cfg(test)]
    pub(crate) fn strokes_pending(&self) -> usize {
        self.strokes.len()
    }

    /// Dabs placed and not yet applied, over every stroke.
    fn queued(&self) -> usize {
        self.strokes.iter().map(|stroke| stroke.dabs.len()).sum()
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

    /// Takes one pen event: a sample for the brush, or (`None`) a stroke the brush doesn't take
    /// starting, which ends an open stroke whose `Up` was lost.
    fn handle(&mut self, scene: &mut Scene<'_>, event: Option<&PenSample>) {
        let Some(sample) = event else {
            self.lift(scene, None);
            return;
        };
        match sample.phase {
            Phase::Down => {
                self.lift(scene, None);
                self.begin(scene, sample);
            }
            Phase::Drag => self.push(scene, sample),
            Phase::Up => self.lift(scene, Some(sample)),
            // Out of a stroke: a stroke whose `Up` was lost ends here.
            Phase::Enter | Phase::Hover | Phase::Leave => self.lift(scene, None),
        }
    }

    /// Starts a stroke at `down` (routed to the brush), if the engine finds the surface under it;
    /// its op begins once the strokes before it are over.
    fn begin(&mut self, scene: &mut Scene<'_>, down: &PenSample) {
        self.warm = Warm::default();
        let routed = self.routed.take();
        let Some(first) = self.engine_sample(down) else {
            return;
        };
        let projector = scene.projector;
        let id = StrokeId::new(self.next_stroke);
        self.next_stroke = self.next_stroke.wrapping_add(1);
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let mut primed = Primed {
            surface: &mut surface,
            first: routed,
        };
        let engine = Stroke::begin(&self.settings, id, &projector, &mut primed, first);
        if let Err(error) = surface.finish() {
            self.fail("starting a stroke", &error);
            return;
        }
        let Some(engine) = engine else {
            return;
        };
        if self.error.take().is_some() {
            self.errors += 1;
        }
        self.strokes.push_back(Pending {
            engine: Some(engine),
            settings: self.settings.clone(),
            camera: stroke_camera(scene.camera, &projector.view),
            samples: vec![record_sample(&first)],
            op: None,
            dabs: VecDeque::new(),
            sources: VecDeque::new(),
            last: times_of(down),
        });
    }

    /// Feeds `sample` (a `Drag` of the open stroke) to the engine, queueing the dabs it places;
    /// a store error ends the stroke there, its dabs dropped.
    fn push(&mut self, scene: &mut Scene<'_>, sample: &PenSample) {
        let Some(engine_sample) = self.engine_sample(sample) else {
            return;
        };
        let budget = self.budget;
        let Some(pending) = self.strokes.back_mut().filter(|stroke| stroke.open()) else {
            return;
        };
        let Some(engine) = &mut pending.engine else {
            return;
        };
        let projector = scene.projector;
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let placed = engine.push(&projector, &mut surface, engine_sample);
        let finished = surface.finish();
        let times = times_of(sample);
        pending.samples.push(record_sample(&engine_sample));
        pending.last = times;
        match finished {
            Ok(()) => {
                let queue = (&mut pending.dabs, &mut pending.sources);
                enqueue(queue, placed, times, scene.index, budget);
            }
            Err(error) => {
                drop(placed);
                pending.engine = None;
                self.fail("placing dabs", &error);
            }
        }
    }

    /// Ends the open stroke's engine stroke, if there is one, queueing its tail: the pen lifted
    /// at `up`, or (`None`) its `Up` was lost, and the tail is timed from its last sample.
    fn lift(&mut self, scene: &mut Scene<'_>, up: Option<&PenSample>) {
        let budget = self.budget;
        let Some(pending) = self.strokes.back_mut() else {
            return;
        };
        let Some(engine) = pending.engine.take() else {
            return;
        };
        let times = up.map_or(pending.last, times_of);
        let projector = scene.projector;
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let tail = engine.end(&projector, &mut surface);
        match surface.finish() {
            Ok(()) => {
                let queue = (&mut pending.dabs, &mut pending.sources);
                enqueue(queue, tail, times, scene.index, budget);
            }
            Err(error) => {
                drop(tail);
                self.fail("placing the stroke's last dabs", &error);
            }
        }
    }

    /// Applies queued dabs within `room` of the budget, oldest stroke first, beginning each
    /// stroke's op as it comes to the front and ending it once it is over; at least one dab if
    /// any is queued.
    fn apply(&mut self, scene: &mut Scene<'_>, report: &mut BrushReport, mut room: f64) {
        loop {
            let Some(front) = self.strokes.front_mut() else {
                return;
            };
            if front.op.is_none() {
                match scene.project.begin_stroke() {
                    Ok(op) => front.op = Some(op),
                    Err(error) => {
                        // No op, so nothing of it can be applied: it is dropped.
                        self.strokes.pop_front();
                        self.fail("starting a stroke's op", &error);
                        continue;
                    }
                }
            }
            let first = report.dabs_applied == 0;
            let costs = front.sources.iter().map(|source| source.cost);
            let count = DabBudget::take(costs, &mut room, first);
            if count > 0 {
                self.apply_front(scene, report, count);
            }
            if !self.strokes.front().is_some_and(Pending::done) {
                return;
            }
            if let Some(over) = self.strokes.pop_front() {
                self.end(scene.project, &over);
            }
        }
    }

    /// Applies the oldest stroke's first `count` queued dabs in one [`apply_dabs`], adding to
    /// `report`. A store error drops the stroke's dabs left, ends its engine stroke, and has the
    /// spatial index rebuilt.
    fn apply_front(&mut self, scene: &mut Scene<'_>, report: &mut BrushReport, count: usize) {
        let Some(front) = self.strokes.front_mut() else {
            return;
        };
        let sources = front.sources.iter().take(count);
        for source in sources {
            report.oldest = Some(
                report
                    .oldest
                    .map_or(source.times, |oldest| oldest.earliest(source.times)),
            );
            report.above_limit |= source.tiles > LIMIT_TILES;
        }
        let dabs: &[Dab] = front.dabs.make_contiguous();
        let batch = dabs.get(..count).unwrap_or(dabs);
        let applied = apply_dabs(scene.project.store(), scene.index, &mut self.state, batch);
        match applied {
            Ok(stats) => {
                front.dabs.drain(..count);
                front.sources.drain(..count);
                report.dabs_applied += stats.dabs;
                report.tiles_edited += stats.tiles_edited;
            }
            Err(error) => {
                front.dabs.clear();
                front.sources.clear();
                front.engine = None;
                self.index_stale = true;
                self.fail("applying dabs", &error);
                self.rebuild_index(scene);
            }
        }
    }

    /// Rebuilds the spatial index from the bounds the project keeps, after a failed
    /// [`apply_dabs`]; on failure it is tried again next frame.
    fn rebuild_index(&mut self, scene: &mut Scene<'_>) {
        let level = scene.index.level();
        match scene.project.load_index(level, |_, _| {}) {
            Ok((index, _)) => {
                *scene.index = index;
                self.index_stale = false;
            }
            Err(error) => self.fail("rebuilding the spatial index", &error),
        }
    }

    /// Ends `over`'s op with its record; unrecorded if the record can't be made or the project
    /// refuses it, so its edits are flushed all the same.
    fn end(&mut self, project: &mut Project, over: &Pending) {
        let Some(op) = over.op else {
            return;
        };
        let record = match self.record_of(op, over) {
            Ok(record) => record,
            Err(error) => {
                self.fail("recording the stroke", &error);
                self.end_unrecorded(project);
                return;
            }
        };
        match project.end_stroke(&record) {
            Ok(()) => {}
            Err(error @ (ProjectError::Cache(_) | ProjectError::NotTheStroke { .. })) => {
                self.fail("recording the stroke", &error);
                self.end_unrecorded(project);
            }
            Err(error) => self.fail("ending the stroke", &error),
        }
    }

    /// Ends the open op with no record.
    fn end_unrecorded(&mut self, project: &mut Project) {
        if let Err(error) = project.end_unrecorded_stroke() {
            self.fail("ending the stroke", &error);
        }
    }

    /// `over`'s record under its `op`: its samples (as many as a record holds), settings,
    /// camera and the engine's version.
    ///
    /// # Errors
    ///
    /// As [`StrokeRecord::new`]: settings too long for a record.
    fn record_of(&mut self, op: OpId, over: &Pending) -> Result<StrokeRecord, StrokeRecordError> {
        let mut record = StrokeRecord::new(
            op,
            stroke::ENGINE_VERSION,
            over.settings.to_bytes(),
            over.camera,
        )?;
        let samples = over.samples.iter();
        for sample in samples {
            if let Err(full) = record.push(sample) {
                self.fail("recording the stroke", &full);
                break;
            }
        }
        Ok(record)
    }

    /// Keeps `error`, from `doing`, for the overlay.
    fn fail(&mut self, doing: &str, error: &dyn std::error::Error) {
        let causes = std::iter::successors(Some(error), |error| error.source());
        let chain: Vec<String> = causes.map(ToString::to_string).collect();
        self.error = Some(format!("{doing}: {}", chain.join(": ")));
        self.errors += 1;
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
        self.settings = self.settings.clone().with_radius(radius);
        self.refresh_status();
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
        self.settings = self.settings.clone().with_strength(strength(scaled));
        self.refresh_status();
        true
    }

    /// Makes the overlay's lines again if what they say changed.
    fn refresh_status(&mut self) {
        let key = StatusKey {
            radius: self.settings.radius().get(),
            strength: self.settings.strength().get(),
            factor: self.strength_factor,
            drawing: !self.strokes.is_empty(),
            last: self.last,
            errors: self.errors,
        };
        if self.status.1 == Some(key) {
            return;
        }
        let mut lines = vec![format!(
            "brush      draw  radius {:.1} px  strength {:.4} of the radius (x{:.2})  keys: [ ] \
             size, - = strength",
            key.radius, key.strength, key.factor
        )];
        let last = key.last;
        lines.push(format!(
            "stroke     {}  last frame with dabs: {} applied, {} carried, {} tiles edited{}",
            if key.drawing { "drawing" } else { "none" },
            last.dabs_applied,
            last.dabs_carried,
            last.tiles_edited,
            if last.above_limit { ", above F" } else { "" }
        ));
        if let Some(error) = &self.error {
            lines.push(format!("brush err  {error}"));
        }
        self.status = (lines, Some(key));
    }

    /// Notes where the pen hovers in this frame's view, for [`Brush::prepare`] to warm.
    fn note_hover(&mut self, pen: &PenFrame, view: Option<&View>) {
        let open = self.strokes.back().is_some_and(Pending::open);
        self.warm.wanted = match (pen.hover(), view, open) {
            (Some(hover), Some(view), false) => Some((hover.at, *view)),
            _ => None,
        };
        if self.warm.wanted != self.warm.target {
            // Moved on (or touched, or left): what was left of the last warming is dropped.
            self.warm.todo.clear();
        }
    }

    /// Starts warming the hover's `wanted` position: raycasts there and lists the tiles within
    /// reach of the hit that aren't evaluated yet.
    fn start_warming(&mut self, scene: &mut Scene<'_>, at: Pixel) {
        let projector = scene.projector;
        let Ok(screen) = ScreenPoint::new(at.x, at.y) else {
            return;
        };
        let Some(ray) = projector.ray(screen) else {
            return;
        };
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
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
        self.warm.todo = cold_tiles(scene.project.store(), scene.index, &hit, reach);
    }

    /// Evaluates the last [`WARM_BATCH`] tiles still to warm.
    fn warm_batch(&mut self, scene: &mut Scene<'_>) {
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
}

/// When `sample` was taken and when it reached the app.
const fn times_of(sample: &PenSample) -> SampleTimes {
    SampleTimes {
        taken: sample.timestamp,
        received: sample.received,
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
            if !self.strokes.is_empty() {
                // Kept for a frame that can place them: the strokes they belong to go on.
                let events = pen.brush_events().map(|event| match event {
                    ForBrush::Sample(sample) => Some(*sample),
                    ForBrush::StrokeElsewhere => None,
                });
                self.held.extend(events);
                report.dabs_carried = self.queued();
                report.busy = true;
            }
            self.routed = None;
            self.note_hover(pen, None);
            self.refresh_status();
            return Ok(report);
        };
        if self.index_stale {
            self.rebuild_index(&mut scene);
        }
        let held = std::mem::take(&mut self.held);
        for event in &held {
            self.handle(&mut scene, event.as_ref());
        }
        let events = pen.brush_events();
        for event in events {
            let sample = match event {
                ForBrush::Sample(sample) => Some(sample),
                ForBrush::StrokeElsewhere => None,
            };
            self.handle(&mut scene, sample);
        }
        self.routed = None;
        self.apply(&mut scene, &mut report, self.budget.units);
        self.note_hover(pen, view.as_ref());
        report.dabs_carried = self.queued();
        report.busy = report.dabs_carried > 0;
        if report.dabs_applied > 0 {
            self.last = report;
        }
        self.refresh_status();
        Ok(report)
    }

    fn route(&mut self, context: BrushContext<'_>, down: &PenSample) -> Route {
        self.routed = None;
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
        match hit {
            Some(hit) => {
                self.routed = Some((ray, hit));
                Route::Brush
            }
            None => Route::Navigate,
        }
    }

    fn prepare(&mut self, context: BrushContext<'_>, deadline: Instant) -> Result<bool> {
        let Some(mut scene) = Scene::of(context) else {
            return Ok(false);
        };
        // At least one piece of work each call, so warming finishes even in frames that overrun.
        let mut worked = false;
        if self.warm.wanted != self.warm.target {
            self.warm.target = self.warm.wanted;
            if let Some((at, _)) = self.warm.wanted {
                self.start_warming(&mut scene, at);
                worked = true;
            }
        }
        while !self.warm.todo.is_empty() && (!worked || Instant::now() < deadline) {
            self.warm_batch(&mut scene);
            worked = true;
        }
        Ok(!self.warm.todo.is_empty())
    }

    fn flush(&mut self, context: BrushContext<'_>) -> Result<()> {
        if let Some(mut scene) = Scene::of(context) {
            let held = std::mem::take(&mut self.held);
            for event in &held {
                self.handle(&mut scene, event.as_ref());
            }
            self.lift(&mut scene, None);
            self.apply(&mut scene, &mut BrushReport::default(), f64::INFINITY);
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

    fn status(&self) -> &[String] {
        &self.status.0
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

/// The brush's base strength, `factor` times [`DEFAULT_STRENGTH_FRACTION`] of the radius.
fn strength(factor: f32) -> Strength {
    Strength::new(DEFAULT_STRENGTH_FRACTION * factor).unwrap_or_default()
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
