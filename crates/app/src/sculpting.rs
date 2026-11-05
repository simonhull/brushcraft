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
//! - **Each step:** the queued dabs are applied by [`apply_dabs`], oldest first, as many as the
//!   time the step has fits ([`BrushBudget`], [`DabCost`]); the rest are shown at once ahead of
//!   their finest work ([`Preview`], "Big brushes" below) and wait for later steps. The edit closes
//!   the store's epoch, which the frame loop drains after the brush phase and uploads in the same
//!   frame (a step between frames: in the next).
//! - **The op:** a stroke's op begins ([`Project::begin_stroke`]) as its first dabs are applied, so
//!   every edit of the store's current op is that stroke's.
//! - **A stroke begins on what the last one left:** it reads its pen-down surface (the brush-engine
//!   spec), which is the surface once every stroke before it has landed. So a pen `Down` while an
//!   earlier stroke still has dabs to apply (its pen lifted) waits, with every event after it, and
//!   is taken once that stroke's op has ended ([`Sculpting::admit`]): its first dab is shown no
//!   sooner, but every dab it places is the same however fast the finest work went. So one stroke
//!   at most is under way. Whether such a `Down` brushes or navigates is decided as it comes, from
//!   the surface as it is then too (the earlier stroke's dabs landed so far), as a `Down` on that
//!   stroke's fresh ridge means to brush ([`Brush::route`]); the stroke begins, or finds nothing
//!   under it and doesn't, on the surface the earlier one left. A `Down` after Ctrl+S waits the
//!   same way, for the save ([`crate::saving`]).
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
//! # The budget: dabs per step, by the time left
//!
//! A step applies dabs for as long as its frame has left ([`BrushBudget`]: the pen's latency aim
//! from the frame's start, less the frame's own work after the brush; up to a refresh interval
//! once the oldest queued dab is past the aim, so a brush behind catches up). The time is turned
//! into dabs by one cost model, [`DabCost`]: [`apply_dabs`] applies a batch tile by tile, so a
//! batch costs a fixed part, a part a dab, a part for each tile each dab reaches, and a part for
//! each tile of their union (each dab's tiles queried in the index as it is queued), fitted to the
//! bench and scaled by a factor learnt on the machine. A step applies at least one dab.
//! The rest wait for the next steps, in order: no dab is dropped, and the stroke's tail waits
//! behind them. A frame that leaves dabs queued wants the next, on input.
//!
//! **Between frames** the frame loop steps the brush too, in short slices, while a stroke is under
//! way and the loop would otherwise wait (for the render thread, the GPU, or the refresh slot): so
//! a frame's own step has only the samples since, and a 60 Hz frame's idle time isn't lost to a
//! fast stroke ([`crate::frame::Owner::between_frames`]).
//!
//! # Big brushes: what doesn't fit is shown, then caught up
//!
//! A frame never waits on finest work that doesn't fit its time (the M1 spec's "Big brushes",
//! decided per frame, 2026-11-01). Whatever a frame's step can't apply in its time is **shown**
//! at once on the finest tiles it reaches ([`Preview::show`]): the draw action's own amounts on
//! the tiles as the store holds them, kept in memory beside the store, never saved, and drawn by
//! the renderer from the store's preview epochs, the display levels through stand-ins. A frame
//! within its time (nothing shown before still to catch up, and all its dabs fitting) is finest
//! work alone, as ever.
//!
//! The dabs shown are the queue's front; they **catch up** as finest work, through
//! [`apply_dabs`] in order, merged per tile in each batch, off the frame's latency path: between
//! frames and after the hand-off ([`Brush::prepare`], until the frame's deadline), never in a
//! frame's own step, which waits on no catch-up. While the pen draws ([`STROKE_QUIET`] since its
//! last sample), the catch-up takes only batches the cost model fits before the pen's next sample
//! is due (its smoothed interval, less a margin), so it doesn't hold a sample up; while it rests
//! on the surface, batches of a sample's interval; once the pen lifts, each step takes at least
//! one dab, in batches of up to [`CATCH_UP_BATCH`] ([`WAITED_BATCH`] while the next stroke waits),
//! a frame between batches.
//! A frame's own step takes what fits the slack its time leaves, if it showed nothing. While the
//! pen is down progress is guaranteed all the same, off a frame's step and whatever the dab costs:
//! under a pen resting [`RESTING_QUIET`] a catch-up that has landed nothing for
//! [`CATCH_UP_STARVED`] takes one dab,
//! and under a drawing pen one holding [`CATCH_UP_BACKLOG`] dabs takes a batch every quarter of
//! that. Only then does a batch's time come before a sample. The backlog isn't bounded: a pen
//! placing dabs faster than they land outruns even these, but the batches grow with the backlog
//! ([`backlog_batch`]) and write each tile once, so the landing gains on it as it grows.
//! Each batch that lands makes the previews it moved again over what landed ([`Preview::land`]);
//! once every dab has landed none is left and the surface is exact, which the record's
//! `dabs_catching_up` counts down to. The saved result is that of applying the dabs at once, bit
//! for bit. The previews are never saved, so a save lands what the brush holds first
//! ([`crate::saving`]).
//!
//! The footprint limit F ([`LIMIT_TILES`]) is the dab one frame's time can't fit at all, by the
//! cost model: such a dab is always shown and caught up. It is reported per frame
//! ([`BrushReport::above_limit`]).
//!
//! A dab is placed on the stroke's pen-down surface ([`StoreSurface`]): the surface as the stroke
//! began, never its own dabs, whether they have landed, wait in the queue or are shown ahead. So
//! what a stroke places doesn't depend on how far its finest work has got.
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

use std::collections::{BTreeSet, VecDeque};
use std::f64::consts::FRAC_PI_2;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use geom::TileKey;
use input::Phase;
use render::camera::View;
use sculpt::{Preview, StoreSurface, apply_dabs};
use store::{
    Evaluation, OpId, PenSample as RecordSample, PenTilt, Project, ProjectError, SpatialIndex,
    StrokeCamera, StrokeRecord, StrokeRecordError, TileStore,
};
use stroke::{
    DVec3, Dab, Hit, Projector, REACH_RADII, Radius, Ray, Sample, ScreenPoint, Strength,
    StrengthMode, Stroke, StrokeId, StrokeSettings, Surface, Tilt, Twist, Unit,
};

use crate::camera::{Camera, Pixel};
use crate::frame::{
    Brush, BrushBudget, BrushChange, BrushContext, BrushReport, Drawing, Kept, LATENCY_AIM,
    Landing, STROKE_QUIET, SampleTimes,
};
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

/// The footprint limit F, as the tiles a dab touches: the most a dab can touch and still fit a
/// frame's time on its own, by the brush's cost model ([`DabCost::limit_tiles`]).
///
/// Derived from [`DabCost::CALIBRATED`] and the latency aim ([`LATENCY_AIM`], 6 ms): a frame
/// with nothing else to do has at most the aim for its brush, and one dab of `t` tiles (its
/// union is its own tiles) costs `per_batch + per_dab + (per_dab_tile + per_tile) · t`, so `t ≤
/// (6 − 0.41 − 0.024) ms / (0.0264 + 0.0589) ms = 65.2`: **65 tiles** (about 270,000 vertices
/// of 65² at the benchmark's level 9). The M1 spec's provisional "about 150 tiles" came from the
/// M0 step model (0.25 ms + 24 µs a tile against 4 ms); the brush's own calibrated model is
/// the one the budget uses, so F follows it. A dab above F never fits a frame: it is always shown
/// ahead and caught up. Below it, the per-frame switch decides (see "Big brushes" in the module
/// notes).
pub(crate) const LIMIT_TILES: usize = DabCost::CALIBRATED.limit_tiles(LATENCY_AIM);

/// The time a catch-up batch is given once the pen has lifted or rests (see "Big brushes" in the
/// module notes): about a 60 Hz frame's.
///
/// [`apply_dabs`] merges a batch's dabs per tile, so a batch of many costs far less than as many
/// batches of one (32 dabs of 50,000 vertices: 16 ms against 38 ms, Task 5b's A/B), and with no
/// pen sample to wait behind it, the catch-up takes batches this long: a step between frames, or
/// after a frame's hand-off, runs to it rather than to its own short slice. A pen that touches
/// again waits at most one batch.
pub(crate) const CATCH_UP_BATCH: Duration = Duration::from_millis(12);

/// How many dabs a catch-up holding `backlog` takes at once under a drawing pen.
///
/// Past `limit`: more as the backlog grows, four at the limit, one at least. A batch writes each
/// tile once, so it lands many dabs for little more of a sample's wait than one.
const fn backlog_batch(backlog: usize, limit: usize) -> usize {
    let batch = backlog / (limit / 4);
    if batch == 0 { 1 } else { batch }
}

/// How long the catch-up's batches are while the pen's next stroke waits for it.
///
/// Dabs overlap, a batch writes each tile once, and landing is the whole wait, so longer batches
/// cut it. A frame comes between batches, however long; measured on the heavy stroke's landing, a
/// batch of one big dab (33 or 66 ms) gives 15 frames a second where this gives 11, for a wait
/// 15% longer and a scribble's slowest tenth of dabs four times later.
pub(crate) const WAITED_BATCH: Duration = Duration::from_millis(100);

/// How often a stroke's samples are taken to arrive before its own say: the Kamvas's measured
/// 173 Hz.
const FIRST_INTERVAL: Duration = Duration::from_micros(5_780);

/// The range a stroke's smoothed sample interval is held to: a gap past it (the pen resting) or
/// below it (samples bunched by the compositor) says little about the next.
const INTERVAL_RANGE: (Duration, Duration) = (Duration::from_millis(2), Duration::from_millis(20));

/// How much earlier than the next sample is due the catch-up must be done while the pen draws:
/// the cost model's misses and the sample's own jitter.
const SAMPLE_MARGIN: Duration = Duration::from_micros(700);

/// A stroke's sample interval `interval` after a sample that came `gap` after the last: an eighth
/// of the way toward the gap, held to [`INTERVAL_RANGE`].
fn smoothed_interval(interval: Duration, gap: Duration) -> Duration {
    let (least, most) = INTERVAL_RANGE;
    let gap = gap.clamp(least, most);
    let toward = match gap.checked_sub(interval) {
        Some(longer) => interval + longer / 8,
        None => interval.saturating_sub(interval.saturating_sub(gap) / 8),
    };
    toward.clamp(least, most)
}

/// Tiles the warming evaluates together on the store's workers between checks of its deadline.
pub(crate) const WARM_BATCH: usize = 16;

/// How long a catch-up may land nothing under a pen resting on the surface before it takes a dab
/// all the same (off a frame's step), whatever it costs: no sample waits behind it. About three
/// 60 Hz frames.
pub(crate) const CATCH_UP_STARVED: Duration = Duration::from_millis(50);

/// How long a pen must have placed no dab before it counts as resting, for a starved catch-up.
///
/// It may go on reporting: what counts is the dabs, not the samples. Much longer than
/// [`STROKE_QUIET`]: a frame held up past that, the samples waiting unread, isn't a resting pen,
/// and a dab forced then held the samples up further. Every 50 ms of stall on a busy machine took
/// a dab of up to 450 ms, and a fast stroke came out seconds behind.
pub(crate) const RESTING_QUIET: Duration = Duration::from_millis(250);

/// How many dabs a catch-up may hold under a drawing pen before it takes them all the same, four
/// times as often as [`CATCH_UP_STARVED`].
///
/// A dab shown costs next to nothing to keep, so this only says when a stroke has gone on for
/// minutes. Forced at 512, a fast stroke fell from 170 frames a second to five after eight
/// seconds, for good. The store's budget plays no part: forcing dabs to land only adds pen-down
/// shapes, which go at the stroke's end, so past the budget it would never let go.
pub(crate) const CATCH_UP_BACKLOG: usize = 16_384;

/// How many dab spacings ahead of an open stroke's last dab the warming ahead of it looks
/// ([`Sculpting::warm_ahead`]): as many as a fast stroke places in a frame or two.
pub(crate) const AHEAD_DABS: usize = 6;

/// What applying dabs costs: the one cost model the brush's budget turns time into dabs with (see
/// "The budget" in the module notes), in one place to retune.
///
/// [`apply_dabs`] applies a batch's dabs tile by tile, tiles in parallel, each tile taking the
/// dabs reaching it in turn. So a batch costs a fixed part, a part for each dab, a part for each
/// tile a dab reaches (counted once per dab: the dab's work on it), and a part for each tile any
/// of them reaches (counted once: predicting, evaluating and writing it):
///
/// ```text
/// time = per_batch + per_dab · dabs + per_dab_tile · Σ tiles each dab reaches + per_tile · |∪ tiles|
/// ```
///
/// A frame of `n` dabs over the same few tiles thus costs well under `n` times one. The terms are
/// fitted ([`DabCost::CALIBRATED`]); the time is then scaled by a factor learnt on the machine.
///
/// **Learnt on the machine:** the predicted time is the calibrated one times a scale learnt from
/// each batch applied ([`DabCost::observe`], the way [`crate::pace::Pacer`] learns its sleeps): a
/// batch slower than predicted raises it at once, but at most doubling it a batch (one cold or
/// stalled batch says little about the next), and a faster one lowers it an eighth of the way.
/// The benchmark project's dabs pin and evaluate more neighbours than the medium project's the
/// fit is made on, so a fixed model would overrun its budgets there; learnt, the budget holds on
/// any project and machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DabCost {
    /// What every batch costs besides its dabs and tiles, in nanoseconds: the call's set-up.
    per_batch: f64,
    /// What every dab costs besides its tiles, in nanoseconds: its query, its share of the
    /// dispatch.
    per_dab: f64,
    /// What every tile a dab reaches costs that dab, in nanoseconds: its work on the tile.
    per_dab_tile: f64,
    /// What every tile any of the batch's dabs reaches costs once, in nanoseconds: predicting,
    /// evaluating and writing it.
    per_tile: f64,
    /// How much longer than calibrated batches have taken on this machine and project, learnt.
    scale: f64,
}

/// The range a [`DabCost`]'s learnt scale is held to: a batch far off says little about the next
/// (a cold start, the process stopped).
const COST_SCALE: (f64, f64) = (0.25, 64.0);

/// The most a [`DabCost`]'s scale rises for one batch: it doubles at most.
const COST_RISE: f64 = 2.0;

impl DabCost {
    /// The model fitted at p99 to part A's interleaved A/B of tile-by-tile `apply_dabs` on the
    /// medium project (`brushcraft-bench brush --dabs N --footprint small|50000`, 16 workers).
    ///
    /// Per call, p99, in ms: a small brush (about 940 vertices, 2 tiles a dab) 0.52, 1.06, 1.10
    /// and 3.09 for 1, 4, 8 and 32 dabs; a 50,000-vertex brush (21 tiles a dab) 2.55, 4.15, 6.08
    /// and 22.0. The union of the batch's tiles, as the bench's stroke (a tenth of a radius a
    /// dab) reaches them: 2, about 2.5, 3 and 4 small tiles; 21, about 23, 27 and 51 large
    /// (measured with the same bench on the benchmark sphere, level 9). Least squares on those
    /// eight gives 0.41 ms a batch, 0.024 ms a dab, 0.026 ms a dab's tile and 0.059 ms a tile of
    /// the union, within 0.55 ms of every point (most within 0.2). A model by the most-hit tile's
    /// dab count fits worse (0.9 ms off at 8 large dabs): the 50,000-vertex brush's time keeps
    /// growing with its dabs after its most-hit tile has all the dabs it can take.
    ///
    /// Retune here when `apply_dabs` changes, with the same bench modes.
    pub(crate) const CALIBRATED: Self = Self {
        per_batch: 410_000.0,
        per_dab: 24_000.0,
        per_dab_tile: 26_400.0,
        per_tile: 58_900.0,
        scale: 1.0,
    };

    /// The most tiles one dab can touch and fit `time` as a batch of its own, unscaled: what F
    /// is ([`LIMIT_TILES`]). Zero if not even a dab of no tile fits.
    pub(crate) const fn limit_tiles(self, time: Duration) -> usize {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a frame's time in nanoseconds: far below where f64 loses integers"
        )]
        let fits = time.as_nanos() as f64;
        let left = fits - self.per_batch - self.per_dab;
        let per_tile = self.per_dab_tile + self.per_tile;
        if left <= 0.0 || per_tile <= 0.0 {
            return 0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a positive count of tiles, rounded down: what fits"
        )]
        let tiles = (left / per_tile) as usize;
        tiles
    }

    /// A model of the given terms, in nanoseconds (as [`DabCost`]'s fields), not yet scaled.
    #[cfg(test)]
    pub(crate) const fn new(
        per_batch: f64,
        per_dab: f64,
        per_dab_tile: f64,
        per_tile: f64,
    ) -> Self {
        Self {
            per_batch,
            per_dab,
            per_dab_tile,
            per_tile,
            scale: 1.0,
        }
    }

    /// A model under which a step applies one dab at a time (each takes far longer than any
    /// budget), for tests of carrying.
    #[cfg(test)]
    pub(crate) const fn one_at_a_time() -> Self {
        Self::new(0.0, 1e15, 0.0, 0.0)
    }

    /// A model under which any number of dabs fit any time, for tests that apply every dab
    /// placed.
    #[cfg(test)]
    pub(crate) const fn any_number() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Learns from a batch predicted (unscaled) at `predicted` nanoseconds that took `took`: a
    /// slower one than predicted raises the scale at once, at most [`COST_RISE`] times; a faster
    /// one lowers it an eighth of the way.
    pub(crate) fn observe(&mut self, predicted: f64, took: Duration) {
        if predicted <= 0.0 {
            return;
        }
        let (least, most) = COST_SCALE;
        let seen = (took.as_secs_f64() * 1e9 / predicted).clamp(least, most);
        self.scale = if seen > self.scale {
            seen.min(self.scale * COST_RISE)
        } else {
            self.scale + (seen - self.scale) / 8.0
        };
    }

    /// How many of the queued dabs, each given by the tiles it touches, in order, fit `room` as
    /// one batch: the longest run from the front whose batch time, scaled as learnt, fits, and at
    /// least one if `at_least_one` and there are any. Returns the count and the batch's predicted
    /// time in nanoseconds, unscaled (none if it takes no dab).
    pub(crate) fn take<'a>(
        self,
        dabs: impl IntoIterator<Item = &'a [TileKey]>,
        room: Duration,
        at_least_one: bool,
    ) -> (usize, f64) {
        let fits = room.as_secs_f64() * 1e9 / self.scale;
        let mut union: BTreeSet<TileKey> = BTreeSet::new();
        let mut used = self.per_batch;
        let mut count = 0;
        for tiles in dabs {
            let new = tiles.iter().filter(|&&key| union.insert(key)).count();
            let cost = self.per_dab
                + self.per_dab_tile * count_of(tiles.len())
                + self.per_tile * count_of(new);
            if used + cost > fits && !(at_least_one && count == 0) {
                break;
            }
            used += cost;
            count += 1;
        }
        (count, if count == 0 { 0.0 } else { used })
    }
}

impl Default for DabCost {
    fn default() -> Self {
        Self::CALIBRATED
    }
}

/// `count` as a number to cost.
fn count_of(count: usize) -> f64 {
    u32::try_from(count).map_or_else(|_| f64::from(u32::MAX), f64::from)
}

/// The camera as the stroke engine's [`Projector`]: rays from the eye through a viewport's
/// pixels, scaled so their parameter is the view depth, as [`crate::pick::PickRay`] aims them,
/// magnified or not.
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
    /// The ray from the eye through `screen`, through the view's principal point
    /// ([`View::through`]), so a magnified view's rays go where its pixels show.
    fn ray(&self, screen: ScreenPoint) -> Option<Ray> {
        let view = &self.view;
        let direction = view.through(screen.x(), screen.y());
        Ray::new(world(view.eye()), world(direction))
    }

    /// A pixel's world size at `depth`: the view's own scale, which a magnified view's
    /// magnification is in ([`View::pixels_per_unit`]), so the brush's screen radius and the
    /// world radius it freezes at are the magnified image's pixels.
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

    fn area_normal(
        &mut self,
        center: DVec3,
        facing: DVec3,
        radius: f64,
        hardness: f32,
    ) -> Option<DVec3> {
        self.surface.area_normal(center, facing, radius, hardness)
    }
}

/// Where a queued dab came from, and what it touches.
#[derive(Debug, Clone, PartialEq)]
struct Source {
    /// The sample whose push (or the pen-up's end) placed it: the frame's latency runs from it.
    times: SampleTimes,
    /// The tiles its sphere touches in the index, as it was queued: its cost ([`DabCost`]).
    tiles: Vec<TileKey>,
    /// Whether it is shown ahead of its finest work ([`Preview::show`]), waiting to catch up.
    shown: bool,
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
    /// Its op, once begun: as its first dabs are applied.
    op: Option<OpId>,
    /// Its dabs not yet applied, oldest first.
    dabs: VecDeque<Dab>,
    /// Where each of `dabs` came from, and its cost.
    sources: VecDeque<Source>,
    /// Its last sample's times: a lost `Up`'s tail is timed from it.
    last: SampleTimes,
    /// When a sample last placed a dab: a held pen goes on reporting, but isn't drawing.
    placed_at: Instant,
    /// How often its samples arrive, smoothed: when the next is due, which the catch-up keeps
    /// clear of ([`Sculpting::before_next_sample`]).
    interval: Duration,
    /// Whether its start (its first dab's centre) has been noted for the camera's pivot.
    start_noted: bool,
    /// When its `Down` was received: what its start is dated by, for the pivot.
    down_at: Instant,
    /// Its last two dabs placed, the latest last: where it heads, for the warming ahead of it.
    heading: [Option<Dab>; 2],
}

impl Pending {
    /// Notes the stroke's start in `start` once its first dab is queued, dated by its `Down`:
    /// the first queued is the first placed, since none is applied before it.
    fn note_start(&mut self, start: &mut Option<([f32; 3], Instant)>) {
        if !self.start_noted
            && let Some(first) = self.dabs.front()
        {
            self.start_noted = true;
            *start = Some((first.position, self.down_at));
        }
    }

    /// Whether the pen still draws it.
    const fn open(&self) -> bool {
        self.engine.is_some()
    }

    /// Whether it is over: the pen lifted (or an error ended it) and every dab is applied.
    fn done(&self) -> bool {
        self.engine.is_none() && self.dabs.is_empty()
    }
}

/// Queues `placed` on a stroke's `queue`: dabs placed by the sample at `times`, each with the
/// tiles of `index` its sphere touches.
fn enqueue(
    (dabs, sources, heading): (
        &mut VecDeque<Dab>,
        &mut VecDeque<Source>,
        &mut [Option<Dab>; 2],
    ),
    placed: impl Iterator<Item = Dab>,
    times: SampleTimes,
    index: &SpatialIndex,
) {
    for dab in placed {
        *heading = [heading[1], Some(dab)];
        let [x, y, z] = dab.position;
        let touched = index.query_sphere(geom::Vec3::new(x, y, z), dab.radius);
        let keys: Vec<TileKey> = touched.collect();
        dabs.push_back(dab);
        sources.push_back(Source {
            times,
            tiles: keys,
            shown: false,
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
    /// What the stroke keeps of the surface it began on.
    kept: Kept,
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
    /// The stroke not over yet, if one is: its pen down, or its dabs still to apply. A stroke
    /// started meanwhile waits in `deferred`.
    stroke: Option<Pending>,
    /// What the stroke keeps of the surface it began on, as the last step left it.
    kept: Kept,
    /// What applying dabs costs: how a step's time becomes dabs.
    cost: DabCost,
    /// The dabs shown ahead of their finest work, and the tiles they reach (see "Big brushes" in
    /// the module notes); `None` for a brush that never shows dabs ahead (F = ∞: everything is
    /// finest work, carried to later steps where it doesn't fit).
    preview: Option<Preview>,
    /// The catch-up done after the hand-off since the last step, which the next step reports.
    caught_up: BrushReport,
    /// Whether a step between frames has shown or applied dabs no frame has drawn yet: while the
    /// pen draws, the catch-up then waits for that frame, which it would hold up.
    unframed: bool,
    /// The next stroke's number.
    next_stroke: u32,
    /// The epoch sample times count from.
    epoch: Instant,
    /// The ray and hit of the last `Down` routed to the brush, for its stroke's first raycast;
    /// cleared at the end of each step.
    routed: Option<(Ray, Hit)>,
    /// Pen samples held over from a step with no scene (a degenerate view) while a stroke was
    /// under way or waited to begin, in order; `None` for a stroke started elsewhere.
    held: Vec<Option<PenSample>>,
    /// Pen events waiting for the stroke before them to end, in order, from a `Down` that came
    /// while an earlier stroke still had dabs to apply (see "A stroke begins on what the last one
    /// left" in the module notes); `None` for a stroke started elsewhere.
    deferred: Vec<Option<PenSample>>,
    /// Whether the spatial index must be rebuilt: an `apply_dabs` failed, and its rebuild hasn't
    /// succeeded yet.
    index_stale: bool,
    /// The warming under a hovering pen.
    warm: Warm,
    /// When the catch-up last landed a dab (or had none to land): what [`CATCH_UP_STARVED`]
    /// counts from.
    landed_at: Instant,
    /// How dabs land: a close and a save land them in steps of their own.
    landing: Landing,
    /// When the save under way was asked for: strokes down after it wait for the save.
    save_asked: Option<Instant>,
    /// The last projector a step had: what a close lands with if the camera has gone degenerate.
    last_projector: Option<ViewProjector>,
    /// The backlog past which a drawing pen waits for the catch-up ([`CATCH_UP_BACKLOG`]).
    backlog_limit: usize,
    /// Where the warming ahead of the open stroke last listed its tiles.
    ahead_warmed: Option<geom::Vec3>,
    /// The tiles ahead of the open stroke still to evaluate.
    ahead_todo: Vec<TileKey>,
    /// The last error, for the overlay; cleared as a stroke starts cleanly.
    error: Option<String>,
    /// Errors kept or cleared so far, for the overlay to see a change.
    errors: u64,
    /// How many errors have dropped dabs, and the latest of them: a save since can't hold them.
    drops: (u64, Option<String>),
    /// The last frame that applied dabs, for the overlay.
    last: BrushReport,
    /// The overlay's lines, and what they were made from.
    status: (Vec<String>, Option<StatusKey>),
    /// The start of the last stroke begun (its first dab's centre) and when its `Down` was
    /// received, until the frame loop takes it for the camera's pivot
    /// ([`Brush::take_stroke_start`]).
    stroke_start: Option<([f32; 3], Instant)>,
    /// Whether the last route raycast its `Down` and found no surface
    /// ([`Brush::routed_off_surface`]).
    route_missed: bool,
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
            stroke: None,
            kept: Kept::default(),
            cost: DabCost::CALIBRATED,
            preview: Some(Preview::new()),
            caught_up: BrushReport::default(),
            unframed: false,
            next_stroke: 0,
            epoch: Instant::now(),
            routed: None,
            held: Vec::new(),
            deferred: Vec::new(),
            index_stale: false,
            warm: Warm::default(),
            landed_at: Instant::now(),
            landing: Landing::InFrames,
            save_asked: None,
            last_projector: None,
            backlog_limit: CATCH_UP_BACKLOG,
            ahead_warmed: None,
            ahead_todo: Vec::new(),
            error: None,
            errors: 0,
            drops: (0, None),
            last: BrushReport::default(),
            status: (Vec::new(), None),
            stroke_start: None,
            route_missed: false,
        };
        brush.refresh_status();
        brush
    }

    /// This brush costing its dabs by `cost`.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn with_cost(mut self, cost: DabCost) -> Self {
        self.cost = cost;
        self
    }

    /// This brush with its dabs shown ahead of their finest work where a step can't fit them
    /// (`true`, the default: F = 0 for whatever doesn't fit), or never (`false`: F = ∞, every
    /// dab finest work, carried to later steps where it doesn't fit).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_previews(mut self, previews: bool) -> Self {
        self.preview = previews.then(Preview::new);
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

    /// Whether a stroke is in progress or its dabs are still being applied, or a stroke waits to
    /// begin.
    #[cfg(test)]
    pub(crate) fn stroking(&self) -> bool {
        self.stroke.is_some() || !self.deferred.is_empty()
    }

    /// Whether a stroke is under way (not one that waits to begin).
    #[cfg(test)]
    pub(crate) const fn under_way(&self) -> bool {
        self.stroke.is_some()
    }

    /// Dabs placed and not yet applied.
    fn queued(&self) -> usize {
        self.stroke.as_ref().map_or(0, |stroke| stroke.dabs.len())
    }

    /// Dabs placed and neither applied nor shown: the stroke's lag.
    fn unshown(&self) -> usize {
        let sources = self.stroke.iter().flat_map(|stroke| &stroke.sources);
        sources.filter(|source| !source.shown).count()
    }

    /// Whether the pen is drawing: the last stroke's last sample arrived within
    /// [`STROKE_QUIET`] (its `Up` included, a few milliseconds after it). The catch-up then takes
    /// only dabs that fit their step's time: a dab forced past it would hold up the samples
    /// arriving meanwhile, and the frame showing a lifted stroke's tail.
    fn pen_moving(&self) -> bool {
        let last = self.stroke.as_ref();
        last.is_some_and(|stroke| stroke.last.received.elapsed() < STROKE_QUIET)
    }

    /// When a catch-up batch must be done by while the pen draws, so as not to hold up its next
    /// sample: [`SAMPLE_MARGIN`] before that is due, by its stroke's smoothed interval. `None`
    /// with no stroke.
    fn before_next_sample(&self) -> Option<Instant> {
        let stroke = self.stroke.as_ref()?;
        let due = stroke.last.received + stroke.interval;
        Some(due.checked_sub(SAMPLE_MARGIN).unwrap_or(due))
    }

    /// Dabs shown and not yet applied: the catch-up still to do.
    fn catching_up(&self) -> usize {
        self.preview.as_ref().map_or(0, Preview::shown)
    }

    /// The report's state as a step ends: what is carried and catching up, whether work is left,
    /// whether a stroke is open, what it keeps.
    fn finish_report(&self, report: &mut BrushReport) {
        report.dabs_carried = self.unshown();
        report.dabs_catching_up = self.catching_up();
        report.busy = self.queued() > 0 || !self.deferred.is_empty();
        report.open = self.stroke.as_ref().is_some_and(Pending::open) || !self.deferred.is_empty();
        report.kept = self.kept;
    }

    /// Notes what the stroke keeps of the surface it began on, from `store` (see
    /// [`BrushReport::kept`]).
    fn note_kept(&mut self, store: &TileStore) {
        self.kept = Kept {
            tiles: store.pen_down_tiles(),
            bytes: store.pen_down_bytes(),
            beyond_budget: store.pen_down_beyond_budget(),
        };
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

    /// Takes one pen event as [`Sculpting::handle`] does, unless it must wait: a `Down` (ending
    /// an open stroke whose `Up` was lost) while an earlier stroke still has dabs to apply, and
    /// every event after one that waits, are deferred until the stroke before them has ended
    /// ([`Sculpting::release_deferred`]).
    fn admit(&mut self, scene: &mut Scene<'_>, event: Option<&PenSample>) {
        let down = event.is_some_and(|sample| sample.phase == Phase::Down);
        if down && self.deferred.is_empty() {
            self.lift(scene, None);
        }
        let waits = !self.deferred.is_empty()
            || (down && self.stroke.is_some())
            || self.waits_for_save(event);
        if !waits {
            self.handle(scene, event);
            return;
        }
        if down {
            // Cast on the surface an earlier stroke had at its pen-down: the stroke raycasts
            // again once it begins.
            self.routed = None;
        }
        self.deferred.push(event.copied());
    }

    /// Whether `event` is the `Down` of a stroke begun after the save under way was asked for: it
    /// waits for the save, and isn't in it.
    fn waits_for_save(&self, event: Option<&PenSample>) -> bool {
        let down = event.filter(|sample| sample.phase == Phase::Down);
        let after = |asked| down.is_some_and(|sample| sample.received > asked);
        self.save_asked.is_some_and(after)
    }

    /// Whether the next events deferred are a stroke waiting for the save under way.
    fn deferred_for_save(&self) -> bool {
        let next = self.deferred.first().copied().flatten();
        self.waits_for_save(next.as_ref())
    }

    /// The events deferred that a save holds: those before a stroke that waits for it.
    fn deferred_in_save(&self) -> impl Iterator<Item = &Option<PenSample>> {
        let events = self.deferred.iter();
        events.take_while(|event| !self.waits_for_save(event.as_ref()))
    }

    /// Takes the events deferred, once the stroke before them has ended: through
    /// [`Sculpting::admit`] again, so a later stroke among them waits for the first in turn, and
    /// one begun after a save was asked for waits for it.
    fn release_deferred(&mut self, scene: &mut Scene<'_>) {
        if self.stroke.is_some() || self.deferred_for_save() {
            return;
        }
        let deferred = std::mem::take(&mut self.deferred);
        for event in &deferred {
            self.admit(scene, event.as_ref());
        }
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

    /// Starts a stroke at `down` (routed to the brush), if the engine finds the surface under it:
    /// with none under way ([`Sculpting::admit`] holds a `Down` back until then).
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
            self.dropped("starting a stroke", &error);
            return;
        }
        let Some(engine) = engine else {
            return;
        };
        if self.error.take().is_some() {
            self.errors += 1;
        }
        self.stroke = Some(Pending {
            engine: Some(engine),
            settings: self.settings.clone(),
            camera: stroke_camera(scene.camera, &projector.view),
            samples: vec![record_sample(&first)],
            op: None,
            dabs: VecDeque::new(),
            sources: VecDeque::new(),
            last: times_of(down),
            placed_at: Instant::now(),
            interval: FIRST_INTERVAL,
            start_noted: false,
            down_at: down.received,
            heading: [None, None],
        });
    }

    /// Feeds `sample` (a `Drag` of the open stroke) to the engine, queueing the dabs it places;
    /// a store error ends the stroke there, its dabs dropped.
    fn push(&mut self, scene: &mut Scene<'_>, sample: &PenSample) {
        let Some(engine_sample) = self.engine_sample(sample) else {
            return;
        };
        let Some(pending) = self.stroke.as_mut().filter(|stroke| stroke.open()) else {
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
        let gap = times
            .received
            .saturating_duration_since(pending.last.received);
        pending.interval = smoothed_interval(pending.interval, gap);
        pending.last = times;
        match finished {
            Ok(()) => {
                let queue = (
                    &mut pending.dabs,
                    &mut pending.sources,
                    &mut pending.heading,
                );
                let before = queue.0.len();
                enqueue(queue, placed, times, scene.index);
                if pending.dabs.len() > before {
                    pending.placed_at = Instant::now();
                }
                pending.note_start(&mut self.stroke_start);
            }
            Err(error) => {
                drop(placed);
                pending.engine = None;
                self.dropped("placing dabs", &error);
            }
        }
    }

    /// Ends the open stroke's engine stroke, if there is one, queueing its tail: the pen lifted
    /// at `up`, or (`None`) its `Up` was lost, and the tail is timed from its last sample.
    fn lift(&mut self, scene: &mut Scene<'_>, up: Option<&PenSample>) {
        let Some(pending) = self.stroke.as_mut() else {
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
                let queue = (
                    &mut pending.dabs,
                    &mut pending.sources,
                    &mut pending.heading,
                );
                enqueue(queue, tail, times, scene.index);
                pending.note_start(&mut self.stroke_start);
            }
            Err(error) => {
                drop(tail);
                self.dropped("placing the stroke's last dabs", &error);
            }
        }
    }

    /// One step's dabs: applied as `budget` leaves time for (all of them with none), and, for a
    /// brush that shows dabs ahead, the rest shown (see "Big brushes" in the module notes).
    ///
    /// A step within its time is finest work alone: nothing shown waits to catch up and the
    /// cost model fits every queued dab. Any other shows its unshown dabs first (they are what
    /// the frame is for) and then catches up with the time left.
    fn work(
        &mut self,
        scene: &mut Scene<'_>,
        report: &mut BrushReport,
        budget: Option<&BrushBudget>,
    ) {
        let showing = self.preview.is_some() && budget.is_some();
        if showing && !self.fits(budget) {
            self.show(scene, report);
        }
        self.apply(scene, report, budget);
        if showing {
            // What the step's finest work didn't take after all.
            self.show(scene, report);
        }
    }

    /// Whether every queued dab fits `budget` as finest work, with nothing shown waiting: then a
    /// step needs no preview.
    fn fits(&self, budget: Option<&BrushBudget>) -> bool {
        let Some(budget) = budget else {
            return true;
        };
        if self.catching_up() > 0 {
            return false;
        }
        let Some(front) = self.stroke.as_ref() else {
            return true;
        };
        let Some(oldest) = front.sources.front() else {
            return true;
        };
        let room = budget
            .until(oldest.times.taken)
            .saturating_duration_since(Instant::now());
        let tiles = front.sources.iter().map(|source| source.tiles.as_slice());
        let (count, _) = self.cost.take(tiles, room, false);
        count == front.dabs.len()
    }

    /// Shows every queued dab not yet shown, in order, ahead of its finest work, counting them in
    /// `report` (their samples among those whose latency the frame carries). A preview that
    /// fails leaves the dabs it didn't show to be shown (or applied) by a later step.
    fn show(&mut self, scene: &mut Scene<'_>, report: &mut BrushReport) {
        let Some(preview) = &mut self.preview else {
            return;
        };
        let unshown = self.stroke.iter().flat_map(|stroke| {
            let pairs = stroke.dabs.iter().zip(&stroke.sources);
            pairs.filter(|(_, source)| !source.shown)
        });
        let dabs: Vec<Dab> = unshown.map(|(dab, _)| *dab).collect();
        if dabs.is_empty() {
            return;
        }
        let before = preview.shown();
        let drawn = scene.drawn;
        let shown = preview.show(
            scene.project.store(),
            scene.index,
            &dabs,
            &|key: &TileKey| drawn.draws(key),
        );
        let count = preview.shown() - before;
        let sources = self
            .stroke
            .iter_mut()
            .flat_map(|stroke| &mut stroke.sources);
        let newly = sources.filter(|source| !source.shown).take(count);
        for source in newly {
            source.shown = true;
            report.dabs_shown += 1;
            report.oldest = Some(
                report
                    .oldest
                    .map_or(source.times, |oldest| oldest.earliest(source.times)),
            );
            report.above_limit |= source.tiles.len() > LIMIT_TILES;
        }
        if let Err(error) = shown {
            self.fail("showing dabs ahead of their finest work", &error);
        }
    }

    /// Applies queued dabs while `budget` leaves time for them (all of them with none), oldest
    /// first, beginning the stroke's op with its first and ending it once the stroke is over; at
    /// least one dab if any is queued and the step must make progress (always for a brush that
    /// shows nothing ahead, else as the budget says).
    ///
    /// The time left is read from the clock before the batch. Dabs shown ahead keep the aim's time:
    /// a step catching up never takes the "behind" time, which is for a brush whose stroke lags
    /// the pen.
    fn apply(
        &mut self,
        scene: &mut Scene<'_>,
        report: &mut BrushReport,
        budget: Option<&BrushBudget>,
    ) {
        // A close or a save lands through its own steps, whatever frames do.
        if self.landing == Landing::InSteps {
            return;
        }
        let moving = self.landing == Landing::InFrames && self.pen_moving();
        // While the pen is down, resting or not, the catch-up never holds up its next sample:
        // a dab is forced, and long batches taken, only once it has lifted.
        let down = self.stroke.as_ref().is_some_and(Pending::open);
        let forced = match (&self.preview, budget) {
            (None, _) | (_, None) => true,
            (Some(_), Some(budget)) => budget.forces_progress() && !moving && !down,
        };
        // Once the pen lifts, a step that must make progress catches up in batches long enough
        // to merge dabs per tile.
        let batch = self.preview.is_some() && forced && budget.is_some() && !moving;
        // A pen resting on the surface sends no samples to wait for: the catch-up takes a
        // sample's interval at a time, so one arriving waits at most that.
        let resting = down && !moving;
        let interval = self
            .stroke
            .as_ref()
            .map_or(FIRST_INTERVAL, |stroke| stroke.interval);
        // While the pen draws, the catch-up waits for the frame of what this step (or one before
        // it, between frames) showed, then has until the next sample is due.
        let waits = self.unframed || report.dabs_shown > 0 || report.dabs_applied > 0;
        let before_next = self.before_next_sample().filter(|_| !waits);
        let waited = self.landed_at.elapsed();
        let backlog = self.catching_up();
        let limit = self.backlog_limit;
        let drawing = self
            .stroke
            .as_ref()
            .is_some_and(|stroke| stroke.placed_at.elapsed() < RESTING_QUIET);
        let behind = if drawing {
            backlog >= self.backlog_limit && waited >= CATCH_UP_STARVED / 4
        } else {
            waited >= CATCH_UP_STARVED
        };
        let Some(front) = self.stroke.as_mut() else {
            return;
        };
        if front.op.is_none() {
            match scene.project.begin_stroke() {
                Ok(op) => front.op = Some(op),
                Err(error) => {
                    // No op, so nothing of it can be applied: it is dropped.
                    self.stroke = None;
                    self.dropped("starting a stroke's op", &error);
                    return;
                }
            }
        }
        let first = report.dabs_applied == 0;
        let waiting_stroke = self.landing != Landing::InFrames || !self.deferred.is_empty();
        // A frame shows each batch before the next starts, so the frames keep coming.
        if batch && self.unframed {
            return;
        }
        let unshown = front.sources.iter().find(|source| !source.shown);
        let room = match (budget, front.sources.front(), unshown) {
            (Some(budget), Some(_), Some(oldest)) => budget
                .until(oldest.times.taken)
                .saturating_duration_since(Instant::now()),
            (Some(budget), Some(_), None) => budget.aim().saturating_duration_since(Instant::now()),
            (Some(_), None, _) => Duration::ZERO,
            (None, _, _) => Duration::MAX,
        };
        let catching_up = front.sources.front().is_some_and(|source| source.shown);
        // A frame's own step catches up only with the slack its time leaves, up to the pen's
        // next sample; what it shows is shown already.
        let frame_step = budget.is_some_and(|budget| !budget.forces_progress());
        // While the pen is down, a catch-up that has landed nothing for a while (or holds many
        // dabs) takes one dab all the same, off a frame's step: the backlog never grows
        // unbounded, and the tiles it holds are let go.
        let starved = catching_up && down && !frame_step && behind;
        let until_sample = before_next.map_or(Duration::ZERO, |by| {
            by.saturating_duration_since(Instant::now())
        });
        let room = match (catching_up, batch, forced) {
            (true, true, _) if waiting_stroke => room.max(WAITED_BATCH),
            (true, true, _) => room.max(CATCH_UP_BATCH),
            (true, false, false) if frame_step => room.min(until_sample),
            // The pen rests: a sample's interval, less the margin.
            (true, false, false) if resting => interval.saturating_sub(SAMPLE_MARGIN),
            // The pen draws: the catch-up has until its next sample is due.
            (true, false, false) => until_sample,
            _ => room,
        };
        let tiles = front.sources.iter().map(|source| source.tiles.as_slice());
        let at_least_one = first && (forced || starved);
        let (count, predicted) = if starved && drawing {
            self.cost.take(
                tiles.take(backlog_batch(backlog, limit)),
                Duration::MAX,
                true,
            )
        } else {
            self.cost.take(tiles, room, at_least_one)
        };
        if count > 0
            && let Some(took) = self.apply_front(scene, report, count)
        {
            self.cost.observe(predicted, took);
        }
        if count > 0 || !catching_up {
            self.landed_at = Instant::now();
        }
        if !self.stroke.as_ref().is_some_and(Pending::done) {
            return;
        }
        if let Some(over) = self.stroke.take() {
            self.end(scene.project, &over);
        }
    }

    /// Applies the stroke's first `count` queued dabs in one [`apply_dabs`], adding to
    /// `report`, and lands those of them shown. A store error drops the stroke's dabs left, ends
    /// its engine stroke, and has the spatial index rebuilt.
    ///
    /// Returns how long [`apply_dabs`] took, what the cost model learns from ([`DabCost`] models
    /// the finest work alone, not the landing's previews made again); `None` if there was no
    /// stroke.
    fn apply_front(
        &mut self,
        scene: &mut Scene<'_>,
        report: &mut BrushReport,
        count: usize,
    ) -> Option<Duration> {
        let front = self.stroke.as_mut()?;
        // Dabs shown already count for the frame that showed them.
        let sources = front
            .sources
            .iter()
            .take(count)
            .filter(|source| !source.shown);
        for source in sources {
            report.oldest = Some(
                report
                    .oldest
                    .map_or(source.times, |oldest| oldest.earliest(source.times)),
            );
            report.above_limit |= source.tiles.len() > LIMIT_TILES;
        }
        let landing = front.sources.iter().take(count).filter(|s| s.shown).count();
        let dabs: &[Dab] = front.dabs.make_contiguous();
        let batch = dabs.get(..count).unwrap_or(dabs);
        let began = Instant::now();
        let applied = apply_dabs(scene.project.store(), scene.index, &mut self.state, batch);
        let took = began.elapsed();
        match applied {
            Ok(stats) => {
                let direct: Vec<Dab> = batch.get(landing..).unwrap_or_default().to_vec();
                front.dabs.drain(..count);
                front.sources.drain(..count);
                report.dabs_applied += stats.dabs;
                report.tiles_edited += stats.tiles_edited;
                self.land(scene, landing, false);
                if let Some(preview) = &mut self.preview
                    && !direct.is_empty()
                    // Applied at once: the stroke's previews count them.
                    && let Err(error) =
                        preview.note_applied(scene.project.store(), scene.index, &direct)
                {
                    self.fail("moving the previews by dabs applied at once", &error);
                }
            }
            Err(error) => {
                // Every dab of the stroke left is dropped, those shown with them.
                let dropped = front.sources.iter().filter(|s| s.shown).count();
                front.dabs.clear();
                front.sources.clear();
                front.engine = None;
                self.index_stale = true;
                self.dropped("applying dabs", &error);
                self.rebuild_index(scene);
                self.land(scene, dropped, true);
            }
        }
        Some(took)
    }

    /// Lands the oldest `count` dabs shown (applied, or dropped), making the previews they moved
    /// again over what landed; with `rebase`, every preview, for a store a failed batch left
    /// changed past what its dabs say.
    fn land(&mut self, scene: &mut Scene<'_>, count: usize, rebase: bool) {
        let Some(preview) = &mut self.preview else {
            return;
        };
        if count == 0 && !rebase {
            return;
        }
        let store = scene.project.store();
        let drawn = scene.drawn;
        let mut landed = preview.land(store, scene.index, count, &|key: &TileKey| drawn.draws(key));
        if rebase && landed.is_ok() {
            landed = preview.rebase(store, scene.index);
        }
        if let Err(error) = landed {
            self.fail("catching the shown dabs up", &error);
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
        if let Some(preview) = &mut self.preview
            && let Err(error) = preview.end_stroke(project.store())
        {
            self.fail("ending the stroke's previews", &error);
        }
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

    /// A batch of a close's or a save's landing, whether or not a frame can be drawn: `lifting`
    /// the pen of any open stroke for a close, which doesn't wait for it. Returns whether any
    /// landed.
    ///
    /// # Errors
    ///
    /// The error kept, if dabs were dropped.
    fn landing_step(&mut self, context: BrushContext<'_>, lifting: bool) -> Result<bool> {
        let errors = self.errors_kept();
        let Some(mut scene) = Scene::closing(context, self.last_projector.as_ref()) else {
            // No scene ever: no stroke began, and what waits to begin can't.
            self.held.clear();
            self.deferred.clear();
            return Ok(false);
        };
        let held = std::mem::take(&mut self.held);
        for event in &held {
            self.admit(&mut scene, event.as_ref());
        }
        self.landing = Landing::Stepping;
        self.unframed = false;
        let budget = BrushBudget::slice(Instant::now(), WAITED_BATCH);
        let mut report = BrushReport::default();
        if lifting {
            self.lift(&mut scene, None);
        }
        self.work(&mut scene, &mut report, Some(&budget));
        if !self.deferred.is_empty() && self.stroke.is_none() {
            self.release_deferred(&mut scene);
            if lifting {
                self.lift(&mut scene, None);
            }
            self.work(&mut scene, &mut report, Some(&budget));
        }
        self.landing = Landing::InSteps;
        self.note_kept(scene.project.store());
        let landed = report.dabs_applied > 0;
        self.caught_up = std::mem::take(&mut self.caught_up).then(report);
        self.failed_since(errors)?;
        Ok(landed)
    }

    /// How many errors have been kept: for [`Sculpting::failed_since`].
    const fn errors_kept(&self) -> u64 {
        self.errors
    }

    /// An error if one was kept since there were `before`: landing for a close has no overlay left
    /// to say it on, and dabs dropped must reach the process's exit; a save can't say it holds
    /// them.
    ///
    /// # Errors
    ///
    /// The error kept.
    fn failed_since(&self, before: u64) -> Result<()> {
        match &self.error {
            Some(error) if self.errors != before => Err(anyhow!("{error}")),
            _ => Ok(()),
        }
    }

    /// Keeps `error`, from `doing`, for the overlay.
    fn fail(&mut self, doing: &str, error: &dyn std::error::Error) {
        let causes = std::iter::successors(Some(error), |error| error.source());
        let chain: Vec<String> = causes.map(ToString::to_string).collect();
        self.error = Some(format!("{doing}: {}", chain.join(": ")));
        self.errors += 1;
    }

    /// Keeps `error`, from `doing`, as [`Sculpting::fail`] does: one that dropped dabs.
    fn dropped(&mut self, doing: &str, error: &dyn std::error::Error) {
        self.fail(doing, error);
        self.drops = (self.drops.0 + 1, self.error.clone());
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
            drawing: self.stroke.is_some(),
            last: self.last,
            kept: self.kept,
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
            "stroke     {}  last frame with dabs: {} applied, {} shown, {} carried, {} catching \
             up, {} tiles edited{}",
            if key.drawing { "drawing" } else { "none" },
            last.dabs_applied,
            last.dabs_shown,
            last.dabs_carried,
            last.dabs_catching_up,
            last.tiles_edited,
            if last.above_limit { ", above F" } else { "" }
        ));
        let kept = key.kept;
        lines.push(format!(
            "pen-down   {} tiles kept  {:.1} MiB in the budget{}",
            kept.tiles,
            mib(kept.bytes),
            if kept.beyond_budget > 0 {
                format!(", {:.1} MiB beyond it", mib(kept.beyond_budget))
            } else {
                String::new()
            }
        ));
        if let Some(error) = &self.error {
            lines.push(format!("brush err  {error}"));
        }
        self.status = (lines, Some(key));
    }

    /// Notes where the pen hovers in this frame's view, for [`Brush::prepare`] to warm.
    fn note_hover(&mut self, pen: &PenFrame, view: Option<&View>) {
        let open = self.stroke.as_ref().is_some_and(Pending::open);
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

    /// Warms a batch of the ground ahead of the open stroke: the tiles within the reach
    /// ([`stroke::REACH_RADII`] radii) of where its dabs head, [`AHEAD_DABS`] spacings past its
    /// last, not evaluated yet. A heading met before (within a radius of the last one warmed) is
    /// not listed again.
    fn warm_ahead(&mut self, scene: &mut Scene<'_>) {
        let Some([Some(previous), Some(last)]) = self.stroke.as_ref().map(|stroke| stroke.heading)
        else {
            return;
        };
        let at = |dab: &Dab| geom::Vec3::new(dab.position[0], dab.position[1], dab.position[2]);
        let step = at(&last) - at(&previous);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a few spacings ahead, a small count"
        )]
        let ahead = at(&last) + step * (AHEAD_DABS as f32);
        let near_last = self
            .ahead_warmed
            .is_some_and(|warmed| (warmed - ahead).length() < last.radius);
        if !near_last {
            self.ahead_warmed = Some(ahead);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the index's query is f32, as the dab's radius"
            )]
            let reach = (REACH_RADII * f64::from(last.radius)) as f32;
            let store = &*scene.project.store();
            let found = scene.index.query_sphere(ahead, reach);
            self.ahead_todo = found
                .filter(|key| store.cached_positions(key).is_none())
                .collect();
        }
        let split = self.ahead_todo.len().saturating_sub(WARM_BATCH);
        let batch = self.ahead_todo.split_off(split);
        if batch.is_empty() {
            return;
        }
        if let Err(error) = scene
            .project
            .store()
            .evaluate_many(&batch, Evaluation::Positions)
        {
            self.ahead_todo.clear();
            self.fail("warming the ground ahead of the stroke", &error);
        }
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
    /// Whether the renderer holds or wants a tile: what the previews are made of.
    drawn: Drawing<'a>,
    /// The tiles the renderer waits for previews of.
    waiting: &'a [TileKey],
}

impl<'a> Scene<'a> {
    /// `context`'s scene for a close: a degenerate camera gives `fallback`, the last good
    /// projector, as what is left to land hardly needs a view. `None` if there is no index, or
    /// nothing ever gave a view: no stroke began then.
    fn closing(context: BrushContext<'a>, fallback: Option<&ViewProjector>) -> Option<Self> {
        let BrushContext {
            project,
            index,
            camera,
            size,
            drawn,
            waiting,
        } = context;
        let projector = camera.view(size).as_ref().map(ViewProjector::new);
        Some(Self {
            project,
            index: index?,
            projector: projector.or_else(|| fallback.copied())?,
            camera,
            drawn,
            waiting,
        })
    }

    /// `context`'s scene; `None` with no index (nothing to brush) or no view (a degenerate
    /// camera).
    fn of(context: BrushContext<'a>) -> Option<Self> {
        let BrushContext {
            project,
            index,
            camera,
            size,
            drawn,
            waiting,
        } = context;
        Some(Self {
            project,
            index: index?,
            projector: ViewProjector::new(&camera.view(size)?),
            camera,
            drawn,
            waiting,
        })
    }
}

impl Brush for Sculpting {
    fn step(
        &mut self,
        context: BrushContext<'_>,
        pen: &PenFrame,
        budget: &BrushBudget,
    ) -> Result<BrushReport> {
        let view = context.camera.view(context.size);
        let mut report = BrushReport::default();
        let Some(mut scene) = Scene::of(context) else {
            if self.stroke.is_some() || !self.deferred.is_empty() {
                // Kept for a frame that can place them: the strokes they belong to go on, and so
                // does a stroke that waits to begin.
                let events = pen.brush_events().map(|event| match event {
                    ForBrush::Sample(sample) => Some(*sample),
                    ForBrush::StrokeElsewhere => None,
                });
                self.held.extend(events);
                self.finish_report(&mut report);
                report.busy = true;
            }
            self.routed = None;
            self.note_hover(pen, None);
            self.refresh_status();
            return Ok(report);
        };
        self.last_projector = Some(scene.projector);
        if self.index_stale {
            self.rebuild_index(&mut scene);
        }
        let held = std::mem::take(&mut self.held);
        for event in &held {
            self.admit(&mut scene, event.as_ref());
        }
        let events = pen.brush_events();
        for event in events {
            let sample = match event {
                ForBrush::Sample(sample) => Some(sample),
                ForBrush::StrokeElsewhere => None,
            };
            self.admit(&mut scene, sample);
        }
        self.work(&mut scene, &mut report, Some(budget));
        if !self.deferred.is_empty() && self.stroke.is_none() {
            // The stroke before them has ended: the deferred stroke begins on what it left.
            self.release_deferred(&mut scene);
            self.work(&mut scene, &mut report, Some(budget));
        }
        self.note_kept(scene.project.store());
        self.routed = None;
        // A frame draws what the steps before it did; one between frames leaves it to the next.
        self.unframed = budget.forces_progress()
            && (self.unframed || report.dabs_shown > 0 || report.dabs_applied > 0);
        self.note_hover(pen, view.as_ref());
        let mut report = std::mem::take(&mut self.caught_up).then(report);
        self.finish_report(&mut report);
        if report.dabs_applied > 0 || report.dabs_shown > 0 {
            self.last = report;
        }
        self.refresh_status();
        Ok(report)
    }

    fn route(&mut self, context: BrushContext<'_>, down: &PenSample) -> Route {
        self.routed = None;
        self.route_missed = false;
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
            // A `Down` that waits for a stroke still landing begins on what that stroke leaves:
            // one on its fresh ridge, off the surface it began on, brushes too.
            None if self.stroke.is_some() || !self.deferred.is_empty() => {
                let narrow = store::Ray::new(ray.origin().to_f32(), ray.direction().to_f32());
                let landed = narrow
                    .map(|narrow| scene.index.raycast_triangle(scene.project.store(), &narrow));
                match landed {
                    Some(Ok(Some(_))) => Route::Brush,
                    Some(Err(error)) => {
                        self.fail("picking under the pen", &error);
                        Route::Navigate
                    }
                    Some(Ok(None)) => {
                        self.route_missed = true;
                        Route::Navigate
                    }
                    None => Route::Navigate,
                }
            }
            None => {
                self.route_missed = true;
                Route::Navigate
            }
        }
    }

    fn prepare(&mut self, context: BrushContext<'_>, deadline: Instant) -> Result<bool> {
        let Some(mut scene) = Scene::of(context) else {
            return Ok(false);
        };
        // The previews the renderer waits for, a few a call, off the pen's path: it draws their
        // ancestors meanwhile.
        if let Some(preview) = &mut self.preview
            && !scene.waiting.is_empty()
            && let Err(error) = preview.make_waiting(scene.project.store(), scene.waiting)
        {
            self.fail("making the previews the renderer waits for", &error);
        }
        // Ahead of a drawing pen, the ground its next dabs will reach is warmed first, a batch a
        // call: placement's reads there (the area normal, the raycasts) find it evaluated.
        if self.stroke.as_ref().is_some_and(Pending::open) {
            self.warm_ahead(&mut scene);
        }
        if self.catching_up() > 0 {
            // The catch-up: finest work of the dabs shown, at least a dab a call, off the
            // latency path. No warming under a hover meanwhile: the pen is down, or its stroke
            // not landed.
            let now = Instant::now();
            let budget = BrushBudget::slice(now, deadline.saturating_duration_since(now));
            let mut report = BrushReport::default();
            self.apply(&mut scene, &mut report, Some(&budget));
            self.note_kept(scene.project.store());
            let progress = report.dabs_applied > 0;
            self.unframed |= progress;
            self.caught_up = std::mem::take(&mut self.caught_up).then(report);
            // More of it wanted only while it moves on: a pen drawing takes only dabs that fit,
            // and the next frame (wanted while dabs are queued) tries again.
            return Ok(progress && self.queued() > 0);
        }
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

    fn next_sample_due(&self) -> Option<Instant> {
        self.before_next_sample().filter(|_| self.pen_moving())
    }

    fn begin_closing(&mut self, context: BrushContext<'_>) -> Result<()> {
        self.landing = Landing::InSteps;
        // Nothing waits for a save any more: a close lands every stroke there is, and the save
        // holds them all.
        self.save_asked = None;
        if let Some(mut scene) = Scene::closing(context, self.last_projector.as_ref()) {
            let held = std::mem::take(&mut self.held);
            for event in &held {
                self.admit(&mut scene, event.as_ref());
            }
            self.lift(&mut scene, None);
        }
        Ok(())
    }

    fn close_step(&mut self, context: BrushContext<'_>) -> Result<bool> {
        self.landing_step(context, true)
    }

    fn landing(&self) -> usize {
        let under_way = usize::from(self.stroke.is_some());
        self.queued().max(under_way) + self.deferred.len() + self.held.len()
    }

    fn begin_saving(&mut self, asked: Instant) {
        self.save_asked = Some(asked);
    }

    fn pen_down(&self) -> bool {
        let events = self.deferred.iter().chain(&self.held).flatten();
        // A stroke after the save began: the pen lifted from any before it, `Up` lost or not.
        if events.clone().any(|event| self.waits_for_save(Some(event))) {
            return false;
        }
        match events.last() {
            Some(event) => matches!(event.phase, Phase::Down | Phase::Drag),
            None => self.stroke.as_ref().is_some_and(Pending::open),
        }
    }

    fn begin_landing(&mut self) {
        self.landing = Landing::InSteps;
    }

    fn landing_for_save(&self) -> usize {
        let under_way = usize::from(self.stroke.is_some());
        self.queued().max(under_way) + self.deferred_in_save().count() + self.held.len()
    }

    fn save_step(&mut self, context: BrushContext<'_>) -> Result<bool> {
        self.landing_step(context, false)
    }

    fn end_saving(&mut self) {
        self.landing = Landing::InFrames;
        self.save_asked = None;
    }

    fn abandon(&mut self) {
        self.stroke = None;
        self.held.clear();
        self.deferred.clear();
        self.landing = Landing::InFrames;
        self.save_asked = None;
    }

    fn drops(&self) -> u64 {
        self.drops.0
    }

    fn dropped_since(&self, before: u64) -> Result<()> {
        match &self.drops {
            (count, Some(error)) if *count != before => Err(anyhow!("{error}")),
            _ => Ok(()),
        }
    }

    fn flush(&mut self, context: BrushContext<'_>) -> Result<()> {
        let errors = self.errors_kept();
        self.landing = Landing::InFrames;
        self.save_asked = None;
        if let Some(mut scene) = Scene::closing(context, self.last_projector.as_ref()) {
            let held = std::mem::take(&mut self.held);
            for event in &held {
                self.admit(&mut scene, event.as_ref());
            }
            // In batches as a close's: one of everything held pins every tile it reaches, which
            // a tight budget may refuse though dab by dab it fits.
            self.lift(&mut scene, None);
            self.landing = Landing::Stepping;
            loop {
                self.unframed = false;
                let budget = BrushBudget::slice(Instant::now(), WAITED_BATCH);
                let mut report = BrushReport::default();
                self.work(&mut scene, &mut report, Some(&budget));
                if report.dabs_applied == 0 {
                    break;
                }
            }
            self.landing = Landing::InFrames;
            loop {
                self.lift(&mut scene, None);
                self.work(&mut scene, &mut BrushReport::default(), None);
                if self.deferred.is_empty() || self.stroke.is_some() {
                    break;
                }
                self.release_deferred(&mut scene);
            }
        }
        self.failed_since(errors)
    }

    fn change(&mut self, change: BrushChange) -> bool {
        match change {
            BrushChange::Smaller => self.scale_radius(SIZE_STEP.recip()),
            BrushChange::Larger => self.scale_radius(SIZE_STEP),
            BrushChange::Weaker => self.scale_strength(STRENGTH_STEP.recip()),
            BrushChange::Stronger => self.scale_strength(STRENGTH_STEP),
        }
    }

    fn take_stroke_start(&mut self) -> Option<(crate::camera::Vec3, Instant)> {
        let ([x, y, z], at) = self.stroke_start.take()?;
        let start = crate::camera::Vec3::new(f64::from(x), f64::from(y), f64::from(z));
        Some((start, at))
    }

    fn routed_off_surface(&self) -> bool {
        self.route_missed
    }

    fn set_radius(&mut self, pixels: f32) {
        self.set_radius_pixels(pixels);
    }

    fn take_settings(&mut self, from: &Self) {
        self.settings = from.settings.clone();
        self.strength_factor = from.strength_factor;
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

/// `bytes` in MiB, for the overlay.
fn mib(bytes: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "an overlay's figure: a few decimals of MiB"
    )]
    let bytes = bytes as f64;
    bytes / f64::from(1_u32 << 20)
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

/// The camera a stroke was drawn through, as its record keeps it.
///
/// `camera`'s eye, target, up, unmagnified field of view and lens (its magnification and the
/// image's shift), and `view`'s size. `recorded_view` (in tests) rebuilds the view from it.
fn stroke_camera(camera: &Camera, view: &View) -> StrokeCamera {
    let triple = |v: crate::camera::Vec3| [v.x, v.y, v.z];
    let size = view.size();
    StrokeCamera {
        eye: triple(camera.eye()),
        target: triple(camera.target()),
        up: triple(camera.axis(crate::camera::Vec3::new(0.0, 1.0, 0.0))),
        fov_y: camera.fov_y(),
        size: [size.width(), size.height()],
        magnification: camera.magnification(),
        shift: camera.shift(),
    }
}

/// The view a stroke record's `camera` was drawn through.
///
/// The plain view from its eye towards its target, its near plane at
/// [`render::camera::NEAR_FRACTION`] of their distance as [`Camera::view`] puts it, magnified by
/// its lens ([`crate::camera::lensed`]). A version 1 record reads as unmagnified. `None` for a
/// camera no view can be made of. For tests and replay tools: nothing in the app replays a stroke
/// yet.
#[cfg(test)]
pub(crate) fn recorded_view(camera: &StrokeCamera) -> Option<View> {
    let point = |[x, y, z]: [f64; 3]| render::camera::Vec3::new(x, y, z);
    let (eye, target) = (point(camera.eye), point(camera.target));
    let size = render::surface::Size::new(camera.size[0], camera.size[1])?;
    let near = (eye - target).length() * render::camera::NEAR_FRACTION;
    let plain = View::look_at(eye, target, point(camera.up), camera.fov_y, size, near)?;
    #[expect(
        clippy::float_cmp,
        reason = "an unmagnified record rebuilds the plain view bit for bit, as Camera::view gives it"
    )]
    if camera.magnification == 1.0 && camera.shift == [0.0, 0.0] {
        return Some(plain);
    }
    crate::camera::lensed(&plain, camera.magnification, camera.shift)
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
