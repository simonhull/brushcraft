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
//! - **The op:** a stroke's op begins ([`Project::begin_stroke`]) as the stroke does, at its `Down`
//!   once the engine finds the surface under it, before it places a dab: so every edit of the
//!   store's current op is that stroke's, and its journal is open for its first dab ("The journal"
//!   below). It begins without waiting; its first landing may have to wait for its pen-down cut
//!   ("The first landing waits for its cut", below). A stroke whose op can't begin doesn't begin
//!   either, its samples dropped.
//! - **A stroke begins on what the last one left:** it reads its pen-down surface (the brush-engine
//!   spec), which is the surface once every stroke before it has landed. A pen `Down` while earlier
//!   strokes still have dabs to apply (their pens lifted) begins at once over them, reading the
//!   surface they will leave ("Strokes over strokes still landing" below), so every dab it places
//!   is the same however fast the finest work went. Past the open strokes' memory share, or if that
//!   surface can't be had, it waits instead, with every event after it, and is taken once they have
//!   all ended ([`Sculpting::admit`]). Whether such a `Down` brushes or navigates is decided as it
//!   comes, from the surface as it is then (the earlier strokes' dabs landed so far), as a `Down`
//!   on a fresh ridge means to brush ([`Brush::route`]). A `Down` after Ctrl+S waits for the save
//!   ([`crate::saving`]).
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
//! # The journal: what is placed and what has landed
//!
//! Dabs placed and not yet landed would live only in memory, so the brush journals them into the
//! recovery file (the dab journal spec, "The journal"): a crash then loses about a second of them,
//! and never a landed part of the stroke.
//!
//! - **Placed:** each dab goes to [`Project::journal_placed`] once, as it joins the stroke's queue,
//!   in placement order (symmetry copies included), as its 84 bytes ([`DabRow`]): the queue's own
//!   memory, cast in place, so the frame's path copies and allocates nothing for it. A dab shown
//!   ahead is placed like any other: a big brush's stroke is journalled from its first dab, however
//!   long it is shown before any lands.
//! - **Landed:** the stroke counts the dabs [`apply_dabs`] has applied (shown or not: showing lands
//!   nothing), and passes the count to [`Project::journal_landed`] right after each [`apply_dabs`]
//!   that succeeds, in the same call on the owner thread. The brush never ticks the project
//!   ([`Project::tick`], which can cut the journal at a flush) and the frame loop ticks only
//!   between brush calls, so no cut ever falls between a landing's edits and its count.
//! - **Dropped:** dabs dropped after an error are never counted landed by the brush: a failed
//!   [`apply_dabs`] passes no count (how far it got is unknown), and its stroke ends in that same
//!   call, where the store closes the journal counting every dab landed so that a restore never
//!   lands them. The dabs of a push that failed never join the queue, so they are never journalled.
//! - A journal call that fails means the brush's bookkeeping broke: it ends the stroke as a store
//!   error does ("Errors" below).
//! - **The first landing waits for its cut.** A stroke's op begins ahead
//!   ([`Project::begin_stroke`]): when strokes come back to back, a save point at the stroke's
//!   pen-down surface is the flush's queued behind the one in flight, and until that flush has
//!   taken its cut the stroke may not land ([`Project::may_land`]). Its dabs are placed, journalled
//!   and shown ahead meanwhile, as when the budget doesn't fit, and none is applied; the frame
//!   loop's ticks and polls drive the flushes ([`Project::tick`]), so it may land within the flush
//!   in flight's syncs, and lands then. A close's or a save's landing, which has no frames to wait
//!   for, waits for the cut ([`Project::wait_to_land`]).
//! - **The op begins at the `Down`,** before any dab is placed: a stroke that places no dab (a tap
//!   off the brush's reach, say) still records an op, with no edits. And [`Project::begin_stroke`]
//!   may roll the recovery file over then, if its writer failed since the last call: the `Down`
//!   takes that roll-over's time.
//!
//! # Strokes over strokes still landing
//!
//! A `Down` while earlier strokes still land doesn't wait for them (the stroke-while-landing
//! spec): it begins at once, on exactly the surface they will leave, and shows its dabs in its
//! first frame. The strokes under way are a list, oldest first; only the newest's pen is ever down
//! (a `Down` lifts the stroke before it), and they land one at a time, in op order.
//!
//! - **Its op is reserved** ([`Project::begin_stroke_after`], [`StrokeOp::Reserved`]): ordered
//!   after theirs, its journal open at once, its dabs journalled as placed
//!   ([`Project::journal_placed_after`], their batches naming the stroke before as their pen-down
//!   until the stroke's own cut is durable). Past the open strokes' share of the store's budget
//!   ([`store::CacheError::OpenStrokesFull`]) nothing is reserved and the `Down` waits, as before
//!   ([`Begin::Waits`]).
//! - **What it reads of them:** as a stroke begins over others, each of them not yet read gets its
//!   [`Remaining`]: its dabs not landed, brushed over its pen-down tiles without writing
//!   ([`StrokeEdits`], sealed: its pen is up), and the tiles they reach with how far they can move
//!   each. Its landing keeps both up ([`Remaining::land`]).
//! - **Its surface is their futures:** each tile as stored with every earlier stroke's remaining
//!   dabs applied ([`Futures::future_of`], bit for bit what their landing leaves), installed as the
//!   stroke's pen-down shapes ([`store::TileStore::install_pen_down_of`]) and refitted in the
//!   spatial index, read through [`StoreSurface::of_reserved`]. A tile they don't change is read as
//!   stored, which it stays.
//! - **Which futures:** each call into the stroke engine runs on a copy of its engine stroke over a
//!   surface that notes what it reads ([`Recorded`]): rays up to their hits, and balls (a walk's
//!   reach, an area normal's radius, a hit's normals). Every tile the earlier strokes change whose
//!   box, grown by how far their dabs can move it, one of those reaches, and every such tile in the
//!   ring of a tile a ball reaches, must have its future installed ([`needed`]); if one hadn't,
//!   they are made (in one [`Futures`] a batch) and the call runs again, until a run read only
//!   installed futures ([`on_futures`]). The engine is deterministic, so that run is the one the
//!   finished surface gives. Each future is made once: the stroke keeps which are installed until
//!   its op begins (a future stays valid while its stroke lands).
//! - **Shown at once:** its dabs are shown over its own pen-down shapes, in a reading of its op
//!   ([`store::TileStore::read_pen_down_of`]), after the earlier strokes' in the one [`Preview`],
//!   whose dabs land in that same order. The previews end with the last stroke.
//! - **Landing in order:** the catch-up lands the oldest stroke only. As it ends, the next's op
//!   begins ([`Project::begin_reserved_stroke`]), its futures its own pen-down shapes now, and its
//!   first landing waits for its pen-down cut as any stroke's ([`Project::may_land`]). A close, a
//!   flush or a save lands them one after another the same way.
//! - **Futures that can't be had** at the `Down` make it wait, nothing reserved kept; later in the
//!   stroke, every stroke before it lands then and there, blocking, and it goes on over the surface
//!   they left, its op begun ([`Sculpting::land_before_newest`]).
//! - **A store error** while the oldest lands drops its dabs left, and every later stroke with
//!   them, their reserved ops given back ([`Project::drop_stroke_after`]): they read the surface
//!   those dabs were to leave.
//!
//! # A stroke a restore found
//!
//! A crash during a landing leaves the stroke's journal in the recovery file. A restore finds it
//! ([`Project::take_open_stroke`]), and the frame loop hands it to the brush before its first
//! frame ([`crate::frame::Owner::resume_stroke`], [`Brush::resume`]; the dab journal spec,
//! "Restoring a journalled stroke"):
//!
//! - **Taken on:** its rows are read back as dabs (each is a [`Dab`]'s bytes, in the layout
//!   [`stroke::DAB_FORMAT`] names), and the project resumes it ([`Project::resume_stroke`]): its op
//!   reopened, its pen-down shapes installed, its journal counting on from its landed count. The
//!   brush holds it as a stroke whose pen has lifted, with no engine stroke and nothing to record:
//!   its dabs queued, each with the tiles it touches, as placed dabs are, and never journalled
//!   again (the file holds them).
//! - **Shown at once,** every dab ahead of its finest work ([`Preview::show`], over what the
//!   renderer draws from the start: its floor), so the whole stroke is on screen in the first
//!   frame, which reports them shown.
//! - **Landed as a lifted stroke:** the catch-up lands it as any stroke whose pen has lifted, and a
//!   `Down` meanwhile begins over it, as over any landing. Once every dab has landed its op ends
//!   unrecorded ([`Project::end_unrecorded_stroke`]): the journal holds no pen samples, and a
//!   stroke record is history, never needed to open.
//! - **Not taken on** with rows of another layout ([`OtherDabFormat`](resume::OtherDabFormat)), a
//!   row that can't be a dab of the model (a damaged or hostile file,
//!   [`HostileDab`](resume::HostileDab)), or by a project that won't resume it (asked twice if the
//!   first refusal is the cache's): the restored tiles stand, and the project's next stroke closes
//!   it in the file. The restore note says so.
//! - **With no window** (a copy of found work, [`finish_restored`]) it is taken on the same way and
//!   landed whole at once, blocking, before the copy is saved.
//!
//! It lands close to what the live landing would have made, not bit for bit: the stroke state
//! ([`sculpt::Stroke`]) starts each vertex it touches from the stored, rounded value, where the
//! live one kept it unrounded, so each is rounded once more, by at most half its tile's
//! quantization step.
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
//! - a journal that refuses the dabs a push or the pen's lift placed has them dropped, and ends the
//!   engine stroke; one that refuses a landed count drops the stroke's dabs left after the batch
//!   applied, so the stroke ends in that call, its journal closed by the store before any cut could
//!   pair the batch's edits with the count before it;
//! - an op that can't begin ([`Project::begin_stroke`]) starts no stroke;
//! - a record [`Project::end_stroke`] refuses ([`ProjectError::Cache`],
//!   [`ProjectError::NotTheStroke`], or one that can't be made) ends the stroke unrecorded
//!   ([`Project::end_unrecorded_stroke`]), so its edits are still flushed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::f64::consts::FRAC_PI_2;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use geom::TileKey;
use input::Phase;
use render::camera::View;
use sculpt::{Futures, PenDownTiles, Preview, SculptError, StoreSurface, StrokeEdits, apply_dabs};
use store::{
    Evaluation, OpId, OpenStroke, PenSample as RecordSample, PenTilt, Project, ProjectError,
    ReservedOp, SpatialIndex, StrokeCamera, StrokeRecord, StrokeRecordError, TileStore,
};
use stroke::{
    DVec3, Dab, Hit, Projector, REACH_RADII, Radius, Ray, Sample, ScreenPoint, Strength,
    StrengthMode, Stroke, StrokeId, StrokeSettings, Surface, Tilt, Twist, Unit,
};

pub(crate) use self::resume::{finish_restored, unseen_view};
use crate::camera::{Camera, Pixel};
use crate::frame::{
    Brush, BrushBudget, BrushChange, BrushContext, BrushReport, Drawing, Kept, LATENCY_AIM,
    Landing, STROKE_QUIET, SampleTimes, recovery_report,
};
use crate::pen::{ForBrush, PenFrame, PenSample, Route};
use crate::saving::DabsDropped;

mod resume;
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

impl Source {
    /// Where `dab`, placed from the sample at `times`, came from: the tiles of `index` its sphere
    /// touches, not shown yet.
    fn of(dab: &Dab, times: SampleTimes, index: &SpatialIndex) -> Self {
        let [x, y, z] = dab.position;
        let touched = index.query_sphere(geom::Vec3::new(x, y, z), dab.radius);
        Self {
            times,
            tiles: touched.collect(),
            shown: false,
        }
    }
}

/// What a stroke drawn with the pen keeps for its record ([`StrokeRecord`]).
#[derive(Debug)]
struct Recording {
    /// The brush's settings as the stroke began (the engine holds its own copy, its radius frozen
    /// at the first hit).
    settings: StrokeSettings,
    /// The camera at the stroke's start.
    camera: StrokeCamera,
    /// The samples the engine read.
    samples: Vec<RecordSample>,
}

/// A stroke's op: begun, the store's current op, or reserved behind the strokes before it while
/// they still land (see "Strokes over strokes still landing" in the module notes).
#[derive(Debug)]
enum StrokeOp {
    /// Begun ([`Project::begin_stroke`], [`Project::begin_reserved_stroke`], or resumed): the
    /// store's current op, every edit made as it, and its dabs land once it may.
    Begun(OpId),
    /// Reserved behind the strokes before it ([`Project::begin_stroke_after`]): it reads their
    /// futures, installed as its pen-down shapes, and lands nothing until it begins, once they
    /// have all ended.
    Reserved {
        /// The op's token, given back to the project as it begins or is let go.
        token: ReservedOp,
        /// The tiles whose futures are installed for it: each computed once, and read as its
        /// pen-down shapes until it begins, when they become its own.
        installed: BTreeSet<TileKey>,
    },
}

impl StrokeOp {
    /// The op.
    const fn id(&self) -> OpId {
        match self {
            Self::Begun(op) => *op,
            Self::Reserved { token, .. } => token.op(),
        }
    }
}

/// How far an earlier stroke's dabs still to land reach a tile: the tiles a later stroke must read
/// at their futures, and how far those futures can lie from the tile as the index holds it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Reach {
    /// The dabs not landed whose spheres touch the tile in the index, as each was queued.
    dabs: usize,
    /// Their strengths' magnitudes summed, over every dab that ever counted here (landing
    /// lowers only the count): how far they can move one of its vertices together, as draw moves
    /// a vertex at most its strength a dab.
    strength: f64,
}

/// What a later stroke reads of an earlier one still landing (the stroke-while-landing spec,
/// "What a new stroke reads").
///
/// Made once a stroke begins over it, its pen up by then so its dabs are final: the futures of
/// the tiles its remaining dabs touch are made from `edits` ([`Futures`]), and which tiles those
/// are, with how far they can move, from `reach`. Both follow its landing.
#[derive(Debug)]
struct Remaining {
    /// Its dabs not landed, with the tiles each touches and edits at its pen-down, sealed.
    edits: StrokeEdits,
    /// Every tile its dabs not landed reach in the index, as they were queued.
    reach: BTreeMap<TileKey, Reach>,
}

impl Remaining {
    /// What a later stroke reads of `stroke`, which follows `earlier` (each with its own made
    /// already), every one of them still landing, oldest first. Its pen-down tiles are the store's
    /// for the oldest, the futures of those before it for any other, so `state` is the session's
    /// stroke state the futures are made from.
    ///
    /// # Errors
    ///
    /// [`SculptError`] as [`PenDownTiles::of_oldest`], [`PenDownTiles::of_futures`],
    /// [`Futures::new`] and [`StrokeEdits::place`]; [`sculpt::FutureError::Chain`] if one of
    /// `earlier` has none made.
    fn of(
        store: &mut TileStore,
        state: &sculpt::Stroke,
        earlier: &[Pending],
        stroke: &mut Pending,
    ) -> Result<Self, SculptError> {
        let touched = stroke.sources.iter().flat_map(|source| &source.tiles);
        let keys: BTreeSet<TileKey> = touched.copied().collect();
        let op = stroke.op.id();
        let pen_down = if earlier.is_empty() {
            PenDownTiles::of_oldest(store, op, &keys)?
        } else {
            let chain = chain_of(earlier).ok_or(sculpt::FutureError::Chain)?;
            let futures = Futures::new(store, state, &chain)?;
            PenDownTiles::of_futures(store, &futures, &keys)?
        };
        let after = earlier.last().map(|before| before.op.id());
        let mut edits = StrokeEdits::new(op, after);
        let dabs: &[Dab] = stroke.dabs.make_contiguous();
        edits.place(store, dabs, &pen_down)?;
        edits.seal();
        let mut reach: BTreeMap<TileKey, Reach> = BTreeMap::new();
        let queued = stroke.dabs.iter().zip(&stroke.sources);
        for (dab, source) in queued {
            for key in &source.tiles {
                let tile = reach.entry(*key).or_default();
                tile.dabs += 1;
                tile.strength += f64::from(dab.strength.abs());
            }
        }
        Ok(Self { edits, reach })
    }

    /// The oldest of its dabs, from `sources` (theirs, oldest first), have landed: they leave its
    /// edits and its reach.
    fn land<'a>(&mut self, sources: impl ExactSizeIterator<Item = &'a Source>) {
        self.edits.land(sources.len());
        for source in sources {
            for key in &source.tiles {
                if let Some(tile) = self.reach.get_mut(key) {
                    tile.dabs = tile.dabs.saturating_sub(1);
                    if tile.dabs == 0 {
                        self.reach.remove(key);
                    }
                }
            }
        }
    }
}

/// The edits of `strokes`, oldest first, if each has its [`Remaining`] made: the chain of futures
/// a stroke after them reads.
fn chain_of(strokes: &[Pending]) -> Option<Vec<&StrokeEdits>> {
    let edits = strokes
        .iter()
        .map(|stroke| stroke.remaining.as_ref().map(|remaining| &remaining.edits));
    edits.collect()
}

/// How many rounds of reading a later stroke's surface and installing the futures it found
/// missing ([`on_futures`]) may go before it gives up.
///
/// Each installs at least one tile, and the tiles that can be missing are those the earlier
/// strokes still change, so a few always do.
const FUTURE_ROUNDS: usize = 32;

/// Why a call into the stroke engine on a reserved stroke's surface ([`on_futures`]) gave nothing.
#[derive(Debug)]
enum OnFutures {
    /// The surface met a store error: the call's dabs are dropped, and its stroke ends, as any
    /// stroke's after a store error.
    Surface(SculptError),
    /// The futures it read couldn't be had (past the open strokes' share, a tile that can't be
    /// evaluated): it must wait for the strokes before it to land.
    Futures(anyhow::Error),
}

/// What a call into the stroke engine read of a surface, as far as which tiles it can have read
/// goes ([`Recorded`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Read {
    /// A raycast: the tiles along `ray` up to `until`, its hit's parameter (infinite for a miss).
    Ray {
        /// The ray.
        ray: Ray,
        /// The parameter its hit lies at; infinite if it hit nothing.
        until: f64,
    },
    /// The surface within `radius` of `center`, normals included: a walk, a region's closest
    /// point, an area normal, a hit's normal.
    Ball {
        /// The centre.
        center: DVec3,
        /// The radius.
        radius: f64,
    },
}

/// A surface that notes what each call reads ([`Read`]): what a reserved stroke's call is checked
/// against ([`on_futures`]).
struct Recorded<'s, S> {
    /// The surface.
    surface: &'s mut S,
    /// What was read, in order.
    reads: Vec<Read>,
}

impl<S: Surface> Surface for Recorded<'_, S> {
    fn raycast(&mut self, ray: &Ray) -> Option<Hit> {
        let hit = self.surface.raycast(ray);
        let until = hit.map_or(f64::INFINITY, |hit| ray.depth_of(hit.position));
        self.reads.push(Read::Ray { ray: *ray, until });
        if let Some(hit) = hit {
            // Its normal reads the tile it hit and that tile's ring.
            self.reads.push(Read::Ball {
                center: hit.position,
                radius: 0.0,
            });
        }
        hit
    }

    fn closest_on_region(&mut self, ray: &Ray, near: &Hit, reach: f64) -> Option<Hit> {
        self.reads.push(Read::Ball {
            center: near.position,
            radius: reach,
        });
        self.surface.closest_on_region(ray, near, reach)
    }

    fn continuous(&mut self, a: &Hit, b: &Hit, reach: f64) -> bool {
        self.reads.push(Read::Ball {
            center: a.position,
            radius: reach,
        });
        self.surface.continuous(a, b, reach)
    }

    fn area_normal(
        &mut self,
        center: DVec3,
        facing: DVec3,
        radius: f64,
        hardness: f32,
    ) -> Option<DVec3> {
        self.reads.push(Read::Ball { center, radius });
        self.surface.area_normal(center, facing, radius, hardness)
    }
}

/// The share of a tile's box diagonal its box is grown by besides the dabs' reach, for the
/// rounding of the positions a future holds against the moves that make it.
const BOX_SLACK: f64 = 1e-3;

/// The tiles `reads` may have read whose futures differ from the store's tiles.
///
/// Of those the strokes `earlier` still change: each whose box in `index`, grown by how far
/// their dabs can move it, a read reaches, and those in the ring of any tile a read reaches by
/// its normals.
///
/// What a later stroke's surface must have installed before that read for it to read the
/// surface those strokes leave (the store's "Ops reserved ahead"): a tile they don't change is
/// read as it is, which it stays.
///
/// # Errors
///
/// As [`TileStore::normal_inputs`].
fn needed(
    store: &TileStore,
    index: &SpatialIndex,
    earlier: &[Pending],
    reads: &[Read],
) -> Result<BTreeSet<TileKey>, store::CacheError> {
    let reaches = earlier
        .iter()
        .filter_map(|stroke| stroke.remaining.as_ref());
    let maps: Vec<&BTreeMap<TileKey, Reach>> = reaches.map(|remaining| &remaining.reach).collect();
    // How far the earlier strokes' dabs can move tile `key`; `None` for one they don't reach.
    let growth = |key: &TileKey| -> Option<f64> {
        let reaching = maps.iter().filter_map(|map| map.get(key));
        reaching.map(|reach| reach.strength).reduce(|a, b| a + b)
    };
    let changed: BTreeSet<TileKey> = maps.iter().flat_map(|map| map.keys()).copied().collect();
    let most = changed.iter().filter_map(growth).fold(0.0_f64, f64::max);
    // The box of tile `key` grown by how far it can move, and by its slack.
    let grown = |key: &TileKey| -> Option<(DVec3, DVec3)> {
        let aabb = index.bvh().bounds(key)?.aabb;
        let (low, high) = (aabb.min(), aabb.max());
        let (low, high) = (
            DVec3::new(f64::from(low.x), f64::from(low.y), f64::from(low.z)),
            DVec3::new(f64::from(high.x), f64::from(high.y), f64::from(high.z)),
        );
        let slack = (high - low).length() * BOX_SLACK + growth(key).unwrap_or(0.0);
        let by = DVec3::new(slack, slack, slack);
        Some((low - by, high + by))
    };
    let mut needed: BTreeSet<TileKey> = BTreeSet::new();
    for read in reads {
        match *read {
            Read::Ray { ray, until } => {
                let crossed = changed.iter().filter(|key| {
                    grown(key).is_none_or(|(low, high)| ray_meets_box(&ray, until, low, high))
                });
                needed.extend(crossed);
            }
            Read::Ball { center, radius } => {
                let slack = radius + most + center.length().max(1.0) * BOX_SLACK;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the index's query is f32; the boxes are grown past its rounding"
                )]
                let (at, within) = (center.to_f32(), (slack * 1.01) as f32);
                let near = index.query_sphere(at, within);
                for key in near {
                    let touched = grown(&key).is_none_or(|(low, high)| {
                        box_distance_squared(center, low, high) <= radius * radius
                    });
                    if !touched {
                        continue;
                    }
                    if changed.contains(&key) {
                        needed.insert(key);
                    }
                    let ring = store.normal_inputs(&key)?;
                    needed.extend(ring.iter().filter(|input| changed.contains(input)));
                }
            }
        }
    }
    Ok(needed)
}

/// Whether `ray`, from its origin up to parameter `until`, meets the box from `low` to `high`
/// (the slab test, its boundary included).
fn ray_meets_box(ray: &Ray, until: f64, low: DVec3, high: DVec3) -> bool {
    let (origin, direction) = (ray.origin(), ray.direction());
    let axes = [
        (origin.x, direction.x, low.x, high.x),
        (origin.y, direction.y, low.y, high.y),
        (origin.z, direction.z, low.z, high.z),
    ];
    let mut span = (0.0_f64, until);
    for (from, along, least, most) in axes {
        if along == 0.0 {
            if from < least || from > most {
                return false;
            }
            continue;
        }
        let (a, b) = ((least - from) / along, (most - from) / along);
        span = (span.0.max(a.min(b)), span.1.min(a.max(b)));
        if span.0 > span.1 {
            return false;
        }
    }
    true
}

/// The squared distance from `point` to the box from `low` to `high`: 0 inside it.
fn box_distance_squared(point: DVec3, low: DVec3, high: DVec3) -> f64 {
    let outside = |at: f64, least: f64, most: f64| (least - at).max(at - most).max(0.0);
    let d = DVec3::new(
        outside(point.x, low.x, high.x),
        outside(point.y, low.y, high.y),
        outside(point.z, low.z, high.z),
    );
    d.dot(d)
}

/// Installs the futures of `keys` not `installed` yet for the stroke reserved as `token`.
///
/// They are the futures after the strokes `earlier` (each with its [`Remaining`]), made from the
/// session's stroke state `state` in one [`Futures`]; each is labelled with the newest of them
/// whose dabs reach the tile, and refitted in the scene's index so the stroke's raycasts find it
/// there.
///
/// # Errors
///
/// [`OnFutures::Futures`] if a future can't be made or installed (past the open strokes' share)
/// or a tile refitted: those installed before it stay.
fn install(
    scene: &mut Scene<'_>,
    state: &sculpt::Stroke,
    earlier: &[Pending],
    token: &ReservedOp,
    installed: &mut BTreeSet<TileKey>,
    keys: &BTreeSet<TileKey>,
) -> Result<(), OnFutures> {
    let fresh: Vec<TileKey> = keys.difference(installed).copied().collect();
    if fresh.is_empty() {
        return Ok(());
    }
    let chain = chain_of(earlier)
        .ok_or_else(|| OnFutures::Futures(anyhow!("an earlier stroke's dabs aren't known")))?;
    let store = scene.project.store();
    let futures = Futures::new(store, state, &chain).map_err(futures_error)?;
    let mut added: Vec<TileKey> = Vec::with_capacity(fresh.len());
    let mut outcome = Ok(());
    for key in fresh {
        let tile = match futures.future_of(store, key) {
            Ok(tile) => tile,
            Err(error) => {
                outcome = Err(futures_error(error));
                break;
            }
        };
        let reaching = earlier.iter().rev().find(|stroke| {
            let remaining = stroke.remaining.as_ref();
            remaining.is_some_and(|remaining| remaining.reach.contains_key(&key))
        });
        let written_by = reaching.map(|stroke| stroke.op.id());
        if let Err(error) = store.install_pen_down_of(token, key, tile, written_by) {
            outcome = Err(futures_error(error));
            break;
        }
        installed.insert(key);
        added.push(key);
    }
    drop(futures);
    let refitted = scene.index.refit(scene.project.store(), &added);
    outcome?;
    refitted.map_err(futures_error)
}

/// `error`, met making or installing a future, as the reason a stroke waits.
fn futures_error(error: impl std::error::Error + Send + Sync + 'static) -> OnFutures {
    OnFutures::Futures(error.into())
}

/// Runs `call`, a call into the stroke engine, on the surface of the stroke reserved as `token`,
/// after the strokes `earlier` still landing, until it has every future it reads.
///
/// It runs again after installing every future its reads found missing ([`needed`]), until one
/// run reads only tiles whose futures it had, and returns that run's result. The stroke engine is
/// deterministic, so that run is what the call gives on the surface those strokes leave; `call`
/// must start each run afresh (from a copy of the engine's stroke, say).
///
/// # Errors
///
/// [`OnFutures::Surface`] for a store error the surface met; [`OnFutures::Futures`] if the
/// futures can't be had, or a run still finds some missing after [`FUTURE_ROUNDS`].
fn on_futures<T>(
    scene: &mut Scene<'_>,
    state: &sculpt::Stroke,
    earlier: &[Pending],
    token: &ReservedOp,
    installed: &mut BTreeSet<TileKey>,
    mut call: impl FnMut(&mut Recorded<'_, StoreSurface<'_>>) -> T,
) -> Result<T, OnFutures> {
    for _ in 0..FUTURE_ROUNDS {
        let (made, reads) = {
            let mut surface = StoreSurface::of_reserved(scene.project.store(), scene.index, token)
                .map_err(OnFutures::Surface)?;
            let mut recorded = Recorded {
                surface: &mut surface,
                reads: Vec::new(),
            };
            let made = call(&mut recorded);
            let reads = recorded.reads;
            surface.finish().map_err(OnFutures::Surface)?;
            (made, reads)
        };
        let wanted =
            needed(scene.project.store(), scene.index, earlier, &reads).map_err(futures_error)?;
        if wanted.is_subset(installed) {
            return Ok(made);
        }
        install(scene, state, earlier, token, installed, &wanted)?;
    }
    Err(OnFutures::Futures(anyhow!(
        "the surface under the pen kept reaching tiles still to land"
    )))
}

/// A stroke, from its `Down` (or its resuming, for one a restore found) until its op ends: its
/// engine stroke while the pen is down, what its record keeps, and its dabs not yet applied.
#[derive(Debug)]
struct Pending {
    /// The engine's stroke; `None` once it has ended (pen up, or an error), while its queued
    /// dabs are applied, and for a stroke a restore found, whose pen has lifted.
    engine: Option<Stroke>,
    /// What its record keeps; `None` for a stroke a restore found ([`Sculpting::resume`]): the
    /// journal holds no pen samples, so it ends unrecorded
    /// ([`Project::end_unrecorded_stroke`]).
    record: Option<Recording>,
    /// Its op: begun with it, or reserved behind the strokes before it until they have ended
    /// (see "Strokes over strokes still landing" in the module notes). Every edit of its dabs is
    /// part of it, and its journal is the op's.
    op: StrokeOp,
    /// What a later stroke reads of it while it lands: its remaining dabs' edits and reach, made
    /// once a stroke begins over it (its pen up by then), and kept as it lands.
    remaining: Option<Remaining>,
    /// Its dabs not yet applied, oldest first.
    dabs: VecDeque<Dab>,
    /// How many of its dabs [`apply_dabs`] has applied, in placement order: the last landed count
    /// its journal accepted ([`Project::journal_landed`]). Placed, it has `landed + dabs.len()`
    /// until the journal refuses a count, which ends the stroke.
    landed: u32,
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

/// A dab as the stroke's journal keeps it ([`Project::journal_placed`]): its bytes, one opaque row
/// of [`store::DAB_BYTES`].
type DabRow = [u8; store::DAB_BYTES];

// A dab is exactly one row, so casting dabs to rows ([`Pending::place`]) never meets a size it
// can't cast; a row's alignment is 1, a dab's 4, so any dab is a row in place.
const _: () = assert!(
    size_of::<Dab>() == store::DAB_BYTES,
    "a dab is one journal row"
);

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

    /// Its op.
    #[cfg(test)]
    const fn op_id(&self) -> OpId {
        self.op.id()
    }

    /// Notes `sample`, taken at `times`, the open stroke's next: its record keeps it, and its
    /// sample interval and last sample follow it.
    fn note_sample(&mut self, sample: &Sample, times: SampleTimes) {
        if let Some(record) = &mut self.record {
            record.samples.push(record_sample(sample));
        }
        let gap = times.received.saturating_duration_since(self.last.received);
        self.interval = smoothed_interval(self.interval, gap);
        self.last = times;
    }

    /// How many of its dabs not yet applied are shown ahead of their finest work.
    fn shown(&self) -> usize {
        self.sources.iter().filter(|source| source.shown).count()
    }

    /// Whether the pen still draws it.
    const fn open(&self) -> bool {
        self.engine.is_some()
    }

    /// Whether it is over: the pen lifted (or an error ended it) and every dab is applied.
    fn done(&self) -> bool {
        self.engine.is_none() && self.dabs.is_empty()
    }

    /// Queues `placed`, the dabs the sample at `times` placed, each with the tiles of `index` its
    /// sphere touches, and journals them in `project` ([`Project::journal_placed`], or
    /// [`Project::journal_placed_after`] while its op is reserved), in order, as
    /// their bytes in the queue itself: the journal is given the queue's memory, cast to rows,
    /// so it costs no copy and no allocation here. Returns how many joined the queue.
    ///
    /// Making the queue contiguous moves its dabs only when its ring has wrapped, as
    /// [`apply_dabs`]'s batch makes it so anyway.
    ///
    /// # Errors
    ///
    /// As [`Project::journal_placed`] (or [`Project::journal_placed_after`]): none of them joins
    /// the queue then, and it is as it was.
    fn place(
        &mut self,
        placed: impl Iterator<Item = Dab>,
        times: SampleTimes,
        index: &SpatialIndex,
        project: &mut Project,
    ) -> Result<usize, ProjectError> {
        let (before, heading) = (self.dabs.len(), self.heading);
        for dab in placed {
            self.heading = [self.heading[1], Some(dab)];
            self.dabs.push_back(dab);
            self.sources.push_back(Source::of(&dab, times, index));
        }
        let queue: &[Dab] = self.dabs.make_contiguous();
        let new = queue.get(before..).unwrap_or_default();
        if new.is_empty() {
            return Ok(0);
        }
        let rows: &[DabRow] = bytemuck::cast_slice(new);
        let count = rows.len();
        let journalled = match &self.op {
            StrokeOp::Begun(_) => project.journal_placed(rows),
            StrokeOp::Reserved { token, .. } => project.journal_placed_after(token, rows),
        };
        if let Err(error) = journalled {
            self.dabs.truncate(before);
            self.sources.truncate(before);
            self.heading = heading;
            return Err(error);
        }
        Ok(count)
    }
}

/// What became of a pen `Down` ([`Sculpting::begin`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Begin {
    /// Taken: a stroke began, or none could (nothing under it, an error kept).
    Done,
    /// It can't begin over the strokes still landing: it waits for them, deferred.
    Waits,
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
    /// The strokes not over yet, oldest first: the oldest's op begun, every later one's reserved
    /// behind it, and only the newest's pen ever down (see "Strokes over strokes still landing" in
    /// the module notes). A stroke that can't begin over them waits in `deferred`.
    strokes: Vec<Pending>,
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
    /// What stroke ends said of the recovery file ([`Brush::take_recovery_reports`]).
    recovery_reports: Vec<ProjectError>,
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
    /// Test record: how many steps that land at once found the stroke waiting for its cut, and
    /// waited ([`Sculpting::may_land`]).
    #[cfg(test)]
    waited_to_land: u64,
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
            strokes: Vec::new(),
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
            recovery_reports: Vec::new(),
            last: BrushReport::default(),
            status: (Vec::new(), None),
            stroke_start: None,
            route_missed: false,
            #[cfg(test)]
            waited_to_land: 0,
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
        !self.strokes.is_empty() || !self.deferred.is_empty()
    }

    /// Whether a stroke is under way (not one that waits to begin).
    #[cfg(test)]
    pub(crate) const fn under_way(&self) -> bool {
        !self.strokes.is_empty()
    }

    /// The strokes under way, oldest first: each from its `Down` until its op ends.
    #[cfg(test)]
    fn pending(&self) -> impl Iterator<Item = &Pending> {
        self.strokes.iter()
    }

    /// Dabs placed and not yet applied.
    fn queued(&self) -> usize {
        self.strokes.iter().map(|stroke| stroke.dabs.len()).sum()
    }

    /// Dabs placed and neither applied nor shown: the stroke's lag.
    fn unshown(&self) -> usize {
        let sources = self.strokes.iter().flat_map(|stroke| &stroke.sources);
        sources.filter(|source| !source.shown).count()
    }

    /// Whether the pen is drawing: the last stroke's last sample arrived within
    /// [`STROKE_QUIET`] (its `Up` included, a few milliseconds after it). The catch-up then takes
    /// only dabs that fit their step's time: a dab forced past it would hold up the samples
    /// arriving meanwhile, and the frame showing a lifted stroke's tail.
    fn pen_moving(&self) -> bool {
        let last = self.strokes.last();
        last.is_some_and(|stroke| stroke.last.received.elapsed() < STROKE_QUIET)
    }

    /// Whether the pen is down on a stroke: the newest, the only one whose pen can be.
    fn pen_on_stroke(&self) -> bool {
        self.strokes.last().is_some_and(Pending::open)
    }

    /// When a catch-up batch must be done by while the pen draws, so as not to hold up its next
    /// sample: [`SAMPLE_MARGIN`] before that is due, by the newest stroke's smoothed interval.
    /// `None` with no stroke.
    fn before_next_sample(&self) -> Option<Instant> {
        let stroke = self.strokes.last()?;
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
        report.open = self.pen_on_stroke() || !self.deferred.is_empty();
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

    /// Takes one pen event as [`Sculpting::handle`] does, unless it must wait: every event after
    /// one that waits, and a stroke's `Down` begun after a save was asked for, are deferred until
    /// the strokes before them have ended ([`Sculpting::release_deferred`]). A `Down` (ending an
    /// open stroke whose `Up` was lost) while earlier strokes still have dabs to apply begins at
    /// once over them, unless it can't ([`Begin::Waits`]): it is deferred then too.
    fn admit(&mut self, scene: &mut Scene<'_>, event: Option<&PenSample>) {
        let down = event.is_some_and(|sample| sample.phase == Phase::Down);
        if down && self.deferred.is_empty() {
            self.lift(scene, None);
        }
        let waits = !self.deferred.is_empty() || self.waits_for_save(event);
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

    /// Takes the events deferred, once the strokes before them have ended: through
    /// [`Sculpting::admit`] again, so a later stroke among them begins over the first in turn (or
    /// waits for it), and one begun after a save was asked for waits for it.
    fn release_deferred(&mut self, scene: &mut Scene<'_>) {
        if !self.strokes.is_empty() || self.deferred_for_save() {
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
                if self.begin(scene, sample) == Begin::Waits {
                    // It begins once the strokes before it have ended, raycasting again then.
                    self.routed = None;
                    self.deferred.push(Some(*sample));
                }
            }
            Phase::Drag => self.push(scene, sample),
            Phase::Up => self.lift(scene, Some(sample)),
            // Out of a stroke: a stroke whose `Up` was lost ends here.
            Phase::Enter | Phase::Hover | Phase::Leave => self.lift(scene, None),
        }
    }

    /// Starts a stroke at `down` (routed to the brush), if the engine finds the surface under it,
    /// and begins its op ahead ([`Project::begin_stroke`]); with strokes still landing, over them
    /// ([`Sculpting::begin_after`]). It lands once it may (see "The journal" in the module notes).
    ///
    /// Returns [`Begin::Waits`] for a stroke that can't begin over those landing: nothing of it is
    /// done, its number not taken, and the caller defers it.
    fn begin(&mut self, scene: &mut Scene<'_>, down: &PenSample) -> Begin {
        self.warm = Warm::default();
        let routed = self.routed.take();
        let Some(first) = self.engine_sample(down) else {
            return Begin::Done;
        };
        let id = StrokeId::new(self.next_stroke);
        if !self.strokes.is_empty() {
            let begun = self.begin_after(scene, down, first, id);
            if begun == Begin::Done {
                self.next_stroke = self.next_stroke.wrapping_add(1);
            }
            return begun;
        }
        self.next_stroke = self.next_stroke.wrapping_add(1);
        let projector = scene.projector;
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let mut primed = Primed {
            surface: &mut surface,
            first: routed,
        };
        let engine = Stroke::begin(&self.settings, id, &projector, &mut primed, first);
        if let Err(error) = surface.finish() {
            self.dropped("starting a stroke", &error);
            return Begin::Done;
        }
        let Some(engine) = engine else {
            return Begin::Done;
        };
        // Before its first dab: the op's journal takes every dab it places.
        let op = match scene.project.begin_stroke() {
            Ok(op) => op,
            Err(error) => {
                // No op, so nothing of it can be applied: it is dropped.
                self.dropped("starting a stroke's op", &error);
                return Begin::Done;
            }
        };
        self.started(scene, down, first, engine, StrokeOp::Begun(op));
        Begin::Done
    }

    /// Starts a stroke at `down`, numbered `id`, its first sample `first`, over the strokes still
    /// landing (see "Strokes over strokes still landing" in the module notes): its op reserved
    /// behind theirs ([`Project::begin_stroke_after`]), it begins on their futures. `routed` is
    /// left unused: it was cast on the pen-down surface of the oldest of them.
    ///
    /// [`Begin::Waits`], with nothing reserved, past the open strokes' share, or if the futures
    /// can't be had.
    fn begin_after(
        &mut self,
        scene: &mut Scene<'_>,
        down: &PenSample,
        first: Sample,
        id: StrokeId,
    ) -> Begin {
        let Ok(token) = scene.project.begin_stroke_after() else {
            // Past the share, typically: it begins once those before it have landed.
            return Begin::Waits;
        };
        if self.make_remaining(scene).is_err() {
            drop(scene.project.drop_stroke_after(token));
            return Begin::Waits;
        }
        let mut installed: BTreeSet<TileKey> = BTreeSet::new();
        let (settings, projector) = (&self.settings, scene.projector);
        let begun = on_futures(
            scene,
            &self.state,
            &self.strokes,
            &token,
            &mut installed,
            |surface| Stroke::begin(settings, id, &projector, surface, first),
        );
        let engine = match begun {
            Ok(Some(engine)) => engine,
            Ok(None) => {
                // Nothing under it: no stroke, as with none landing.
                drop(scene.project.drop_stroke_after(token));
                return Begin::Done;
            }
            Err(OnFutures::Surface(error)) => {
                drop(scene.project.drop_stroke_after(token));
                self.dropped("starting a stroke", &error);
                return Begin::Done;
            }
            Err(OnFutures::Futures(_)) => {
                drop(scene.project.drop_stroke_after(token));
                return Begin::Waits;
            }
        };
        let op = StrokeOp::Reserved { token, installed };
        self.started(scene, down, first, engine, op);
        Begin::Done
    }

    /// Makes what a later stroke reads ([`Remaining`]) of every stroke still landing that has none,
    /// oldest first: a stroke begins over them.
    ///
    /// # Errors
    ///
    /// As [`Remaining::of`]: those made before it stay.
    fn make_remaining(&mut self, scene: &mut Scene<'_>) -> Result<(), SculptError> {
        for at in 0..self.strokes.len() {
            let Some((earlier, rest)) = self.strokes.split_at_mut_checked(at) else {
                break;
            };
            let Some(stroke) = rest.first_mut().filter(|stroke| stroke.remaining.is_none()) else {
                continue;
            };
            let made = Remaining::of(scene.project.store(), &self.state, earlier, stroke)?;
            stroke.remaining = Some(made);
        }
        Ok(())
    }

    /// The stroke begun at `down`, its first sample `first`, its engine stroke `engine` and its op
    /// `op`, joins the strokes under way, the newest: its pen is down.
    fn started(
        &mut self,
        scene: &Scene<'_>,
        down: &PenSample,
        first: Sample,
        engine: Stroke,
        op: StrokeOp,
    ) {
        if self.error.take().is_some() {
            self.errors += 1;
        }
        self.strokes.push(Pending {
            engine: Some(engine),
            record: Some(Recording {
                settings: self.settings.clone(),
                camera: stroke_camera(scene.camera, &scene.projector.view),
                samples: vec![record_sample(&first)],
            }),
            op,
            remaining: None,
            dabs: VecDeque::new(),
            landed: 0,
            sources: VecDeque::new(),
            last: times_of(down),
            placed_at: Instant::now(),
            interval: FIRST_INTERVAL,
            start_noted: false,
            down_at: down.received,
            heading: [None, None],
        });
    }

    /// Feeds `sample` (a `Drag` of the open stroke) to the engine, queueing and journalling the
    /// dabs it places; a store error (the journal's included) ends the stroke there, its dabs
    /// dropped. A stroke over strokes still landing reads their futures ([`Sculpting::on_newest`]).
    fn push(&mut self, scene: &mut Scene<'_>, sample: &PenSample) {
        let Some(engine_sample) = self.engine_sample(sample) else {
            return;
        };
        let times = times_of(sample);
        let Some(pending) = self.strokes.last_mut().filter(|stroke| stroke.open()) else {
            return;
        };
        pending.note_sample(&engine_sample, times);
        let projector = scene.projector;
        if matches!(pending.op, StrokeOp::Reserved { .. }) {
            let pushed = self.on_newest(scene, |engine, surface| {
                let placed: Vec<Dab> = engine.push(&projector, surface, engine_sample).collect();
                placed
            });
            if let Some(placed) = pushed {
                self.queue_newest(
                    scene,
                    placed.into_iter(),
                    times,
                    "journalling the dabs placed",
                );
            }
            return;
        }
        // Out while its dabs are queued, which borrow it; back only if they all were.
        let Some(mut engine) = pending.engine.take() else {
            return;
        };
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let placed = engine.push(&projector, &mut surface, engine_sample);
        if let Err(error) = surface.finish() {
            drop(placed);
            self.dropped("placing dabs", &error);
            return;
        }
        let queued = pending.place(placed, times, scene.index, scene.project);
        match queued {
            Ok(count) => {
                pending.engine = Some(engine);
                if count > 0 {
                    pending.placed_at = Instant::now();
                }
                pending.note_start(&mut self.stroke_start);
            }
            Err(error) => self.dropped("journalling the dabs placed", &error),
        }
    }

    /// Ends the open stroke's engine stroke, if there is one, queueing and journalling its tail:
    /// the pen lifted at `up`, or (`None`) its `Up` was lost, and the tail is timed from its last
    /// sample.
    fn lift(&mut self, scene: &mut Scene<'_>, up: Option<&PenSample>) {
        let Some(pending) = self.strokes.last_mut().filter(|stroke| stroke.open()) else {
            return;
        };
        let times = up.map_or(pending.last, times_of);
        let projector = scene.projector;
        if matches!(pending.op, StrokeOp::Reserved { .. }) {
            let ended = self.on_newest(scene, |engine, surface| {
                let tail: Vec<Dab> = engine.clone().end(&projector, surface).collect();
                tail
            });
            if let Some(newest) = self.strokes.last_mut() {
                // Lifted, whatever its tail met.
                newest.engine = None;
            }
            if let Some(tail) = ended {
                self.queue_newest(
                    scene,
                    tail.into_iter(),
                    times,
                    "journalling the stroke's last dabs",
                );
            }
            return;
        }
        let Some(engine) = pending.engine.take() else {
            return;
        };
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let tail = engine.end(&projector, &mut surface);
        if let Err(error) = surface.finish() {
            drop(tail);
            self.dropped("placing the stroke's last dabs", &error);
            return;
        }
        let queued = pending.place(tail, times, scene.index, scene.project);
        match queued {
            Ok(_) => pending.note_start(&mut self.stroke_start),
            Err(error) => self.dropped("journalling the stroke's last dabs", &error),
        }
    }

    /// Runs `call` on the newest stroke's engine stroke, its pen down and its op reserved over the
    /// strokes still landing, on their futures ([`on_futures`]), each run on a copy of the engine
    /// stroke: the engine stroke becomes the run's that stands, and its dabs are returned.
    ///
    /// A store error ends the stroke (its pen lifted, these dabs dropped) and returns `None`.
    /// Futures that can't be had land every stroke before it now, as if it had waited for them
    /// ([`Sculpting::land_before_newest`]); then its op has begun, and `call` runs on its own
    /// pen-down surface, the surface they left. If they can't land, its pen is lifted, and it ends
    /// once they have.
    fn on_newest(
        &mut self,
        scene: &mut Scene<'_>,
        mut call: impl FnMut(&mut Stroke, &mut Recorded<'_, StoreSurface<'_>>) -> Vec<Dab>,
    ) -> Option<Vec<Dab>> {
        let (pending, earlier) = self.strokes.split_last_mut()?;
        let engine = pending.engine.as_ref()?;
        let StrokeOp::Reserved { token, installed } = &mut pending.op else {
            return None;
        };
        let run = on_futures(scene, &self.state, earlier, token, installed, |surface| {
            let mut moved = engine.clone();
            let placed = call(&mut moved, surface);
            (moved, placed)
        });
        let why = match run {
            Ok((moved, placed)) => {
                pending.engine = Some(moved);
                return Some(placed);
            }
            Err(OnFutures::Surface(error)) => {
                pending.engine = None;
                self.dropped("placing dabs", &error);
                return None;
            }
            Err(OnFutures::Futures(why)) => why,
        };
        self.land_before_newest(scene);
        let newest = self.strokes.last_mut()?;
        if !matches!(newest.op, StrokeOp::Begun(_)) {
            newest.engine = None;
            let why = why.context("the strokes before it couldn't land");
            self.fail("placing dabs over strokes still landing", &*why);
            return None;
        }
        let own = newest.engine.as_mut()?;
        let mut surface = StoreSurface::new(scene.project.store(), scene.index);
        let mut recorded = Recorded {
            surface: &mut surface,
            reads: Vec::new(),
        };
        let placed = call(own, &mut recorded);
        if let Err(error) = surface.finish() {
            newest.engine = None;
            self.dropped("placing dabs", &error);
            return None;
        }
        Some(placed)
    }

    /// Queues and journals `placed`, the newest stroke's dabs placed from the sample at `times`;
    /// a journal that refuses them ends its engine stroke, the error kept from `doing`.
    fn queue_newest(
        &mut self,
        scene: &mut Scene<'_>,
        placed: impl Iterator<Item = Dab>,
        times: SampleTimes,
        doing: &str,
    ) {
        let Some(pending) = self.strokes.last_mut() else {
            return;
        };
        match pending.place(placed, times, scene.index, scene.project) {
            Ok(count) => {
                if count > 0 {
                    pending.placed_at = Instant::now();
                }
                pending.note_start(&mut self.stroke_start);
            }
            Err(error) => {
                pending.engine = None;
                self.dropped(doing, &error);
            }
        }
    }

    /// Lands every stroke before the newest now, blocking, oldest first, each beginning the next
    /// as it ends: for a newest stroke whose futures can't be had, which then reads the surface
    /// they left, as if it had waited for them. Stops if a landing makes no progress.
    fn land_before_newest(&mut self, scene: &mut Scene<'_>) {
        let landing = std::mem::replace(&mut self.landing, Landing::InFrames);
        while self.strokes.len() > 1 {
            let before = (self.strokes.len(), self.queued());
            self.unframed = false;
            self.apply(scene, &mut BrushReport::default(), None);
            if (self.strokes.len(), self.queued()) == before {
                break;
            }
        }
        self.landing = landing;
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
    /// step needs no preview. Never with strokes over the oldest: their dabs can't land before
    /// its op ends, so they are shown.
    fn fits(&self, budget: Option<&BrushBudget>) -> bool {
        let Some(budget) = budget else {
            return true;
        };
        if self.catching_up() > 0 || self.strokes.len() > 1 {
            return false;
        }
        let Some(front) = self.strokes.first() else {
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
        self.show_on(scene.project, scene.index, scene.drawn, report);
    }

    /// [`Sculpting::show`] on `project`'s tiles, found by `index`, previews made of those `drawn`:
    /// what showing needs of a scene, for a stroke taken on before any view
    /// ([`Sculpting::resume`]).
    ///
    /// Stroke by stroke, oldest first, as they land: a stroke whose op is reserved over strokes
    /// still landing shows its dabs over its own pen-down shapes, their futures
    /// ([`TileStore::read_pen_down_of`]), as its dabs were placed.
    fn show_on(
        &mut self,
        project: &mut Project,
        index: &SpatialIndex,
        drawn: Drawing<'_>,
        report: &mut BrushReport,
    ) {
        let Some(preview) = &mut self.preview else {
            return;
        };
        let wanted = |key: &TileKey| drawn.draws(key);
        let mut failed = None;
        for stroke in &mut self.strokes {
            let pairs = stroke.dabs.iter().zip(&stroke.sources);
            let unshown = pairs.filter(|(_, source)| !source.shown);
            let dabs: Vec<Dab> = unshown.map(|(dab, _)| *dab).collect();
            if dabs.is_empty() {
                continue;
            }
            let before = preview.shown();
            let shown = match &stroke.op {
                StrokeOp::Begun(_) => preview.show(project.store(), index, &dabs, &wanted),
                StrokeOp::Reserved { token, .. } => match project.store().read_pen_down_of(token) {
                    Ok(mut reading) => preview.show(&mut reading, index, &dabs, &wanted),
                    Err(error) => Err(error.into()),
                },
            };
            let count = preview.shown() - before;
            let sources = stroke.sources.iter_mut();
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
                failed = Some(error);
                break;
            }
        }
        if let Some(error) = failed {
            self.fail("showing dabs ahead of their finest work", &error);
        }
    }

    /// Applies queued dabs while `budget` leaves time for them (all of them with none), oldest
    /// first, of the oldest stroke, ending its op once it is over (in the same call as the batch
    /// that leaves it over, a failed one's included), and beginning the next's; at least one dab if
    /// any is queued and the step must make progress (always for a brush that shows nothing
    /// ahead, else as the budget says).
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
        // A step that must land everything (no budget: a flush's) waits, as a close's does.
        let at_once = self.landing == Landing::Stepping || budget.is_none();
        if !self.may_land(scene, at_once) {
            return;
        }
        let moving = self.landing == Landing::InFrames && self.pen_moving();
        // While the pen is down, resting or not, the catch-up never holds up its next sample:
        // a dab is forced, and long batches taken, only once it has lifted.
        let down = self.pen_on_stroke();
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
            .strokes
            .last()
            .map_or(FIRST_INTERVAL, |stroke| stroke.interval);
        // While the pen draws, the catch-up waits for the frame of what this step (or one before
        // it, between frames) showed, then has until the next sample is due.
        let waits = self.unframed || report.dabs_shown > 0 || report.dabs_applied > 0;
        let before_next = self.before_next_sample().filter(|_| !waits);
        let waited = self.landed_at.elapsed();
        let backlog = self.catching_up();
        let limit = self.backlog_limit;
        let drawing = self
            .strokes
            .last()
            .is_some_and(|stroke| stroke.placed_at.elapsed() < RESTING_QUIET);
        let behind = if drawing {
            backlog >= self.backlog_limit && waited >= CATCH_UP_STARVED / 4
        } else {
            waited >= CATCH_UP_STARVED
        };
        // The oldest stroke lands first: one stroke writes at a time, in op order.
        let Some(front) = self.strokes.first_mut() else {
            return;
        };
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
        if !self.strokes.first().is_some_and(Pending::done) {
            return;
        }
        if !self.strokes.is_empty() {
            let over = self.strokes.remove(0);
            self.end(scene, &over);
        }
    }

    /// Whether the stroke may land now ([`Project::may_land`]): its pen-down flush has taken its
    /// cut. A frame's step goes on without landing until it has (what it would land is shown
    /// ahead); a step that lands `at_once`, with no frames to wait for (a close's, a save's, a
    /// flush's), waits for the cut ([`Project::wait_to_land`]), at most the flush in flight's
    /// syncs. What waiting meets of the recovery file is reported as a stroke end's is; the stroke
    /// may land then all the same.
    fn may_land(&mut self, scene: &mut Scene<'_>, at_once: bool) -> bool {
        if scene.project.may_land() {
            return true;
        }
        if !at_once {
            return false;
        }
        #[cfg(test)]
        {
            self.waited_to_land += 1;
        }
        if let Err(error) = scene.project.wait_to_land() {
            self.ended_with(error);
        }
        scene.project.may_land()
    }

    /// Applies the oldest stroke's first `count` queued dabs in one [`apply_dabs`], adding to
    /// `report`, passes its landed count to its journal ([`Project::journal_landed`]) right
    /// after, and lands those of them shown; what a later stroke reads of it follows
    /// ([`Remaining::land`]). A store error drops the stroke's dabs left (and every later
    /// stroke, [`Sculpting::drop_left`]), ends its engine stroke, and has the spatial index
    /// rebuilt; a count the journal refuses drops the dabs left too.
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
        let front = self.strokes.first_mut()?;
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
                // The dabs `apply_dabs` was given: what landed.
                let taken = batch.len();
                if let Some(remaining) = &mut front.remaining {
                    remaining.land(front.sources.iter().take(taken));
                }
                front.dabs.drain(..taken);
                front.sources.drain(..taken);
                // The count goes with the edits, in this call: nothing ticks the project between
                // them (see "The journal" in the module notes). A count past a `u32` saturates,
                // and the journal refuses it.
                let batch_dabs = u32::try_from(taken).unwrap_or(u32::MAX);
                let reached = front.landed.saturating_add(batch_dabs);
                let counted = scene.project.journal_landed(reached);
                if counted.is_ok() {
                    front.landed = reached;
                }
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
                if let Err(error) = counted {
                    // No cut may pair the batch's edits with the count before it: the stroke
                    // ends in this call, and its journal closes counting every dab landed.
                    let dropped =
                        self.drop_left(scene.project, "journalling the dabs landed", &error);
                    self.land(scene, dropped, false);
                }
            }
            Err(error) => {
                // No count: how far the batch got is unknown, and the stroke ends in this call.
                let dropped = self.drop_left(scene.project, "applying dabs", &error);
                self.index_stale = true;
                self.rebuild_index(scene);
                self.land(scene, dropped, true);
            }
        }
        Some(took)
    }

    /// Drops every dab the oldest stroke has left to apply, those shown with them, and ends its
    /// engine stroke, keeping `error` from `doing` as one that dropped dabs: the stroke is then
    /// over, and [`Sculpting::apply`] ends its op in the same call. Every later stroke goes too
    /// ([`Sculpting::drop_later`]): it read the surface those dabs were to leave. Returns how many
    /// of the dabs dropped were shown: theirs are the previews to land ([`Sculpting::land`]).
    fn drop_left(
        &mut self,
        project: &mut Project,
        doing: &str,
        error: &dyn std::error::Error,
    ) -> usize {
        let shown = self.strokes.first_mut().map_or(0, |front| {
            let shown = front.shown();
            front.dabs.clear();
            front.sources.clear();
            front.engine = None;
            shown
        });
        let later = self.drop_later(project);
        self.dropped(doing, error);
        shown + later
    }

    /// Lets every stroke after the oldest go, none of its dabs landed: each reserved op is given
    /// back to `project` ([`Project::drop_stroke_after`], which closes its journal counting every
    /// dab landed, so a restore never lands them). Returns how many of their dabs were shown,
    /// which come right after the oldest's among those the previews hold.
    fn drop_later(&mut self, project: &mut Project) -> usize {
        let later = self.strokes.split_off(self.strokes.len().min(1));
        let mut shown = 0;
        for stroke in later {
            shown += stroke.shown();
            if let StrokeOp::Reserved { token, .. } = stroke.op {
                drop(project.drop_stroke_after(token));
            }
        }
        shown
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

    /// Ends `over`'s op, the oldest stroke's, taken out of the strokes under way, and begins the
    /// next's ([`Sculpting::begin_next`]).
    ///
    /// The previews end with the last stroke: a later stroke's dabs shown stay. The op ends with
    /// its record; unrecorded if it has none (a stroke a restore found), if the record can't be
    /// made or the project refuses it, so its edits are flushed all the same.
    fn end(&mut self, scene: &mut Scene<'_>, over: &Pending) {
        if self.strokes.is_empty() {
            self.end_previews(scene.project);
        }
        self.end_op(scene.project, over);
        if !self.strokes.is_empty() {
            self.begin_next(scene);
            if self.strokes.is_empty() {
                // The next couldn't begin, and went with every one after it.
                self.end_previews(scene.project);
            }
        }
    }

    /// Ends the previews' stroke: every dab shown has landed, or was dropped.
    fn end_previews(&mut self, project: &mut Project) {
        if let Some(preview) = &mut self.preview
            && let Err(error) = preview.end_stroke(project.store())
        {
            self.fail("ending the stroke's previews", &error);
        }
    }

    /// Ends `over`'s op (see [`Sculpting::end`]).
    fn end_op(&mut self, project: &mut Project, over: &Pending) {
        let Some(recording) = &over.record else {
            self.end_unrecorded(project);
            return;
        };
        let record = match self.record_of(over.op.id(), recording) {
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
            Err(error) => self.ended_with(error),
        }
    }

    /// Begins the op of the oldest stroke left, reserved behind the one that just ended
    /// ([`Project::begin_reserved_stroke`]): its futures become its own pen-down shapes, the
    /// strokes after it read it as the oldest ([`StrokeEdits::earlier_landed`]), and it lands
    /// once its pen-down cut is taken ([`Project::may_land`]), as any stroke. One that can't
    /// begin goes, with every stroke after it, their dabs dropped.
    fn begin_next(&mut self, scene: &mut Scene<'_>) {
        let Some(next) = self.strokes.first_mut() else {
            return;
        };
        let id = next.op.id();
        let StrokeOp::Reserved { token, .. } = std::mem::replace(&mut next.op, StrokeOp::Begun(id))
        else {
            return;
        };
        match scene.project.begin_reserved_stroke(token) {
            Ok(op) => {
                next.op = StrokeOp::Begun(op);
                if let Some(remaining) = &mut next.remaining {
                    remaining.edits.earlier_landed();
                }
            }
            Err(refused) => {
                let (given_back, error) = refused.into_parts();
                let shown = next.shown();
                drop(scene.project.drop_stroke_after(given_back));
                let later = self.drop_later(scene.project);
                self.strokes.clear();
                self.dropped("beginning a stroke's op", &error);
                self.land(scene, shown + later, false);
            }
        }
    }

    /// Ends the open op with no record.
    fn end_unrecorded(&mut self, project: &mut Project) {
        if let Err(error) = project.end_unrecorded_stroke() {
            self.ended_with(error);
        }
    }

    /// A stroke's end said `error`. Of the recovery file, it is the frame loop's to hear: the
    /// stroke did end, and nothing was dropped, so it can't fail the save or the close it landed
    /// for.
    fn ended_with(&mut self, error: ProjectError) {
        if recovery_report(&error) {
            self.recovery_reports.push(error);
        } else {
            self.fail("ending the stroke", &error);
        }
    }

    /// The record of the stroke `over` kept for, under its `op`: its samples (as many as a record
    /// holds), settings, camera and the engine's version.
    ///
    /// # Errors
    ///
    /// As [`StrokeRecord::new`]: settings too long for a record.
    fn record_of(&mut self, op: OpId, over: &Recording) -> Result<StrokeRecord, StrokeRecordError> {
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
        if !self.deferred.is_empty() && self.strokes.is_empty() {
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
            drawing: !self.strokes.is_empty(),
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
        let open = self.pen_on_stroke();
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
        let Some([Some(previous), Some(last)]) = self.strokes.last().map(|stroke| stroke.heading)
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
    /// nothing ever gave a view: no stroke began then. A stroke a restore found begins with no
    /// step, and so with no view of one; taking it on keeps the view it was taken on through as
    /// that projector, and one with no view isn't taken on ([`Sculpting::take_on`]), so a stroke
    /// under way always has one here.
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
            if !self.strokes.is_empty() || !self.deferred.is_empty() {
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
        if !self.deferred.is_empty() && self.strokes.is_empty() {
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
            None if !self.strokes.is_empty() || !self.deferred.is_empty() => {
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
        if self.pen_on_stroke() {
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

    fn resume(&mut self, context: BrushContext<'_>, stroke: OpenStroke) -> Result<()> {
        self.take_on(context, stroke)
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
        self.queued().max(self.strokes.len()) + self.deferred.len() + self.held.len()
    }

    fn journals(&self) -> bool {
        true
    }

    /// The events held or deferred, and every stroke reserved over one still landing: its dabs
    /// are journalled, but no restore takes a stroke behind another yet, so the window stays for
    /// the save as it does for a stroke not placed.
    fn unplaced_for_save(&self) -> usize {
        let reserved = self.strokes.iter().skip(1).count();
        self.deferred_in_save().count() + self.held.len() + reserved
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
            None => self.pen_on_stroke(),
        }
    }

    fn begin_landing(&mut self) {
        self.landing = Landing::InSteps;
    }

    fn landing_for_save(&self) -> usize {
        self.queued().max(self.strokes.len()) + self.deferred_in_save().count() + self.held.len()
    }

    fn save_step(&mut self, context: BrushContext<'_>) -> Result<bool> {
        self.landing_step(context, false)
    }

    fn end_saving(&mut self) {
        self.landing = Landing::InFrames;
        self.save_asked = None;
    }

    fn abandon(&mut self) {
        // The project is discarded next, its store with it: a reserved op has nothing to go
        // back to.
        for stroke in std::mem::take(&mut self.strokes) {
            if let StrokeOp::Reserved { token, .. } = stroke.op {
                token.abandon_with_store_gone();
            }
        }
        self.held.clear();
        self.deferred.clear();
        self.landing = Landing::InFrames;
        self.save_asked = None;
    }

    fn drops(&self) -> u64 {
        self.drops.0
    }

    fn take_recovery_reports(&mut self) -> Vec<ProjectError> {
        std::mem::take(&mut self.recovery_reports)
    }

    fn dropped_since(&self, before: u64) -> Result<(), DabsDropped> {
        match &self.drops {
            (count, Some(error)) if *count != before => Err(DabsDropped(error.clone())),
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
                let before = (self.strokes.len(), self.queued());
                self.work(&mut scene, &mut BrushReport::default(), Some(&budget));
                // A stroke ending, and the next beginning, is progress too.
                if (self.strokes.len(), self.queued()) == before {
                    break;
                }
            }
            self.landing = Landing::InFrames;
            loop {
                self.lift(&mut scene, None);
                let before = (self.strokes.len(), self.queued());
                self.work(&mut scene, &mut BrushReport::default(), None);
                if !self.strokes.is_empty() {
                    // Each work lands the oldest stroke whole, and begins the next's op.
                    if (self.strokes.len(), self.queued()) == before {
                        break;
                    }
                    continue;
                }
                if self.deferred.is_empty() {
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
