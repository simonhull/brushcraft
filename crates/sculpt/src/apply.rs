//! Applying a frame's dabs: [`apply_dabs`], the draw brush for every dab a frame placed, each
//! dab's tiles brushed in parallel on the store's workers.
//!
//! [`apply_dabs`] runs, in order:
//!
//! 1. **Query**: for each dab, the tiles whose boxes its sphere touches, from the spatial index as
//!    the frame found it.
//! 2. **Pin**: every tile of those queries is evaluated and pinned, together, so they stay resident
//!    for the frame; and with them every tile step 5 can evaluate or read, so that wave needs no
//!    room the pins haven't counted (see "Errors").
//! 3. **Each dab, in order:**
//!    - **Brush**: `brush`'s draw action ([`brush::draw_imprint`]) over the evaluated positions of
//!      each of the dab's tiles, the tiles on the store's workers, giving each vertex within the
//!      radius its delta along the normal axis of its tangent frame. Every delta of the dab is
//!      computed before any correction is written: an edit moves the positions of every tile
//!      holding a copy of the vertex, and a later tile's brush must see the surface as the dab
//!      found it.
//!    - **Edit**: each delta is added to the sum of the stroke's deltas for its vertex, and the
//!      vertex's correction `z` set to the value the stroke first found there plus that sum (see
//!      [`Stroke`]), through [`TileStore::update_tiles`] ([`TileStore::edit_tile`] for each tile,
//!      with seam twins kept in step), which writes the pinned tiles' interiors in parallel. The
//!      stored value is so rounded to its tile's quantization once per stroke, not once per dab.
//!    - **Positions**: the tiles whose positions the dab moved (those holding a vertex within its
//!      radius) get their positions again, in parallel ([`TileStore::evaluate_many`]). They are
//!      pinned and only their corrections changed, so the store applies the new corrections to the
//!      prediction it kept of each (see "Kept predictions" on [`TileStore`]): the prediction, most
//!      of a tile's evaluation, is made once per frame. Their boxes are taken again, for the next
//!      dabs' tiles (see "The frame's tiles"). The last dab skips this: step 5 evaluates its tiles.
//! 4. **Neighbours**: the tiles whose bounds may have changed are listed, once for the frame: every
//!    tile holding a vertex within a dab's radius, and every tile whose normals read one of those
//!    ([`TileStore::stale_after`]).
//! 5. **Evaluate**: those tiles are evaluated again, positions and normals, together through
//!    [`TileStore::evaluate_many_with_bounds`], each tile's bounds on the worker that computed its
//!    normals.
//! 6. **Refit**: the index's tree is refitted to those bounds; the frame closes its epoch of the
//!    store's stale-tile feed ([`TileStore::close_epoch`]), exactly one a call, whatever failed.
//!
//! # The same as one dab at a time
//!
//! The frame's corrections, and so its positions and normals, are those of applying its dabs
//! one at a time, a call each, bit for bit, at any worker count:
//!
//! - **Each dab sees the surface the dabs before it left.** Its brush reads its tiles' positions as
//!   evaluated after the previous dab's writes: the tiles that dab moved are evaluated again before
//!   the next is brushed (step 3), and the store's evaluation of a tile is a function of its
//!   corrections and the coarser levels', which the brush never edits. So each dab's deltas are the
//!   ones it would compute alone.
//! - **Its tiles hold every vertex it reaches.** A one-at-a-time call queries the index refitted
//!   after the dab before it; this one knows each tile's box as it is now (see "The frame's
//!   tiles"), so every tile holding a vertex within the dab's radius is brushed. A tile brushed
//!   that no vertex of falls within the radius gives nothing: the brush finds no vertex to move,
//!   and the tile is neither reached nor edited.
//! - **Its writes are the ones it would make alone.** The edit is the same call with the same
//!   tiles, deltas and stroke state, dab by dab in sequence order: [`Stroke`] counts and releases
//!   per dab as it does across calls, and the store's edit gives the same bits at any worker count.
//!
//! The tests check it on a generated project, against one call per dab.
//!
//! # The frame's tiles
//!
//! A tile's positions move only if a dab reaches it: if it holds a vertex within the dab's
//! radius (and so is brushed, and counted as reached), whether it owns the vertex and is edited,
//! or holds a copy of one its neighbour edits. So dab `k` brushes:
//!
//! - every tile its sphere touches by its box in the index as the frame found it, but those the
//!   frame has reached already: the others haven't moved since, so that box still bounds them;
//! - every tile the frame has reached whose box now, taken after the dab that last moved it,
//!   touches its sphere.
//!
//! By induction over the dabs, that is every tile holding a vertex within dab `k`'s radius,
//! whatever its action does to the vertices it reaches, with no bound on how far they move. Only
//! the first set is known before the frame runs; it is what the frame pins, and the second is
//! within it, as a tile reached was brushed and so pinned.
//!
//! # Errors
//!
//! A store error ends the call. The dabs before it stay applied (the failing dab as
//! [`TileStore::update_tiles`] leaves it), and the index is still brought up to date for every
//! tile a dab of the frame reached (steps 4 to 6); should that fail too, the first error is
//! returned and the index must be rebuilt. The stale-tile epoch is closed either way. If the
//! frame's tiles don't fit the budget's pins together ([`CacheError::OverBudget`], or
//! [`CacheError::Full`] for pins that fit the count but not the chunks), the frame is applied in
//! two halves, each a frame of its own, down to one dab: nothing has been written when the pin
//! fails. The pins also hold the tiles the neighbours' wave evaluates (the tiles whose normals
//! read a queried tile) and those their normals read, so after the edits that wave finds every
//! tile it touches pinned, and can't run out of room. Releasing the pins before the wave instead
//! would lose the predictions the store keeps for pinned tiles edited in place, which the next
//! frames reuse.

use std::collections::BTreeSet;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

use brush::{Imprint, Smoothstep, draw_imprint};
use geom::{TileKey, Vec3};
use store::{Aabb, CacheError, Evaluation, IndexError, SpatialIndex, TileStore};
use stroke::Dab;
use thiserror::Error;

use crate::draw::{DrawParamsError, check_dab};
use crate::state::{Stroke, TileStroke};

/// Why [`apply_dabs`] or [`draw_step`](crate::draw_step) failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SculptError {
    /// A dab reaches nowhere or pushes by a value that isn't finite: refused before anything is
    /// written.
    #[error("dab {dab} is not one the brush can apply")]
    Dab {
        /// The dab's place in the slice given.
        dab: usize,
        /// What is wrong with it.
        #[source]
        error: DrawParamsError,
    },
    /// The store couldn't pin, evaluate or edit a tile.
    #[error("the tile store failed")]
    Cache(#[from] CacheError),
    /// The spatial index couldn't be brought up to date.
    #[error("updating the spatial index failed")]
    Index(#[from] IndexError),
}

/// How long each phase of an [`apply_dabs`] or a [`draw_step`](crate::draw_step) took (see the
/// module notes for the phases).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseTimes {
    /// Querying the spatial index for the tiles the dabs touch.
    pub query: Duration,
    /// Evaluating and pinning those tiles.
    pub pin: Duration,
    /// Running the brush over their evaluated positions.
    pub brush: Duration,
    /// Writing the deltas into the corrections.
    pub edit: Duration,
    /// Listing the tiles whose bounds the edits may change: those reached, and their
    /// neighbours whose normals read them.
    pub neighbours: Duration,
    /// Evaluating the changed tiles again: their positions between dabs, then positions and
    /// normals once for the frame.
    pub evaluate: Duration,
    /// Recomputing their bounds and refitting the index.
    pub refit: Duration,
}

impl PhaseTimes {
    /// The whole call: every phase together.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.query
            + self.pin
            + self.brush
            + self.edit
            + self.neighbours
            + self.evaluate
            + self.refit
    }
}

impl AddAssign for PhaseTimes {
    /// Adds each phase's time, to total the phases of several calls.
    fn add_assign(&mut self, rhs: Self) {
        self.query += rhs.query;
        self.pin += rhs.pin;
        self.brush += rhs.brush;
        self.edit += rhs.edit;
        self.neighbours += rhs.neighbours;
        self.evaluate += rhs.evaluate;
        self.refit += rhs.refit;
    }
}

/// What one [`apply_dabs`] did, and how long each phase took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplyStats {
    /// Dabs applied.
    pub dabs: usize,
    /// The frames they were applied in: one, unless the frame's tiles didn't fit the budget's
    /// pins together and it was split (see "Errors" in the module notes).
    pub frames: usize,
    /// Tiles the dabs' spheres touch by their boxes as the frame found them: queried and pinned
    /// (summed over the frames, should the frame be split to fit the budget).
    pub candidates: usize,
    /// Tiles whose corrections a dab edited, each counted once: those owning a vertex a dab
    /// moved (each also writes the copies its neighbours hold of its border vertices).
    pub tiles_edited: usize,
    /// Vertex moves: for each dab, the vertices it moved, each counted once however many tiles
    /// hold a copy of it.
    pub vertices: usize,
    /// Tiles evaluated again and refitted in the index: those holding a moved vertex, and those
    /// whose normals read one of them.
    pub tiles_updated: usize,
    /// For each dab, the tiles it wrote per dab, the stored value plus the dab's delta, because
    /// the stroke had no room to keep their state (see "The thrash limit" on [`Stroke`]).
    pub tiles_unaccumulated: usize,
    /// How long each phase took.
    pub times: PhaseTimes,
}

impl AddAssign for ApplyStats {
    /// Adds two frames' counts and times: a frame applied in parts.
    fn add_assign(&mut self, rhs: Self) {
        self.dabs += rhs.dabs;
        self.frames += rhs.frames;
        self.candidates += rhs.candidates;
        self.tiles_edited += rhs.tiles_edited;
        self.vertices += rhs.vertices;
        self.tiles_updated += rhs.tiles_updated;
        self.tiles_unaccumulated += rhs.tiles_unaccumulated;
        self.times += rhs.times;
    }
}

/// Applies `dabs`, a frame's dabs in sequence order, with the draw action.
///
/// Each dab moves every vertex of the index's level within its radius along the vertex's frame
/// normal by `strength · falloff(d / radius)`, the falloff [`Smoothstep`] made as hard as the
/// dab's hardness ([`brush::Hardened`]), and `index` is brought up to date once for the frame
/// (see the module notes for the phases).
///
/// Each dab's values are taken as final: `strength` is the displacement at its centre in model
/// units, `hardness` in `0..=1` (`0`, the plain falloff); the dab's normal, tangent, angle and
/// params aren't read by the draw action. The result is that of applying the dabs one at a time,
/// a call each, bit for bit (see "The same as one dab at a time" in the module notes).
///
/// `store` is the project's ([`Project::store`](store::Project::store)). The dabs edit as part
/// of the store's current op: call it between
/// [`Project::begin_stroke`](store::Project::begin_stroke) and
/// [`Project::end_stroke`](store::Project::end_stroke), so the stroke is one op. `stroke` keeps
/// each edited vertex's start and summed deltas across the stroke's dabs (see [`Stroke`]); keep
/// one for the session and pass it to every call.
///
/// # Errors
///
/// [`SculptError::Dab`] for a dab whose centre or strength isn't finite or whose radius isn't
/// positive and finite, before anything is written. [`SculptError::Cache`] if the store can't
/// pin the tiles a dab touches (more than its budget holds, for one dab), evaluate them or edit
/// them; [`SculptError::Index`] if the index can't be updated. A store error ends the call: the
/// dabs before the failing one stay applied, and the index is updated for them (see "Errors" in
/// the module notes). The failing dab's edit is left as [`TileStore::update_tiles`] leaves it,
/// and the deltas it summed into `stroke` but didn't write are written by the next dab that
/// writes those vertices; don't apply it again, or its deltas already summed count twice.
pub fn apply_dabs(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    stroke: &mut Stroke,
    dabs: &[Dab],
) -> Result<ApplyStats, SculptError> {
    let imprints: Result<Vec<Imprint>, SculptError> = dabs
        .iter()
        .enumerate()
        .map(|(n, dab)| imprint_of(dab).map_err(|error| SculptError::Dab { dab: n, error }))
        .collect();
    apply_imprints(store, index, stroke, &imprints?)
}

/// The draw action's imprint of `dab`, checked as [`DrawParams::new`](crate::DrawParams::new)
/// checks its values.
///
/// # Errors
///
/// As [`DrawParams::new`](crate::DrawParams::new).
fn imprint_of(dab: &Dab) -> Result<Imprint, DrawParamsError> {
    let [x, y, z] = dab.position;
    check_dab(Vec3::new(x, y, z), dab.radius, dab.strength)?;
    Ok(Imprint::new(
        dab.position,
        dab.normal,
        dab.radius,
        dab.strength,
        dab.hardness,
        dab.params,
    ))
}

/// [`apply_dabs`] for imprints already checked: positive, finite radii and finite centres and
/// strengths. An empty frame does nothing.
///
/// # Errors
///
/// As [`apply_dabs`].
pub(crate) fn apply_imprints(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    stroke: &mut Stroke,
    imprints: &[Imprint],
) -> Result<ApplyStats, SculptError> {
    if imprints.is_empty() {
        return Ok(ApplyStats::default());
    }
    let mut laps = Laps::start();
    let mut times = PhaseTimes::default();

    let queries: Vec<Vec<TileKey>> = imprints
        .iter()
        .map(|imprint| {
            index
                .query_sphere(centre(imprint), imprint.radius)
                .collect()
        })
        .collect();
    let frame_tiles: BTreeSet<TileKey> = queries.iter().flatten().copied().collect();
    let frame_tiles: Vec<TileKey> = frame_tiles.into_iter().collect();
    times.query = laps.lap();

    let held = headroom(store, &frame_tiles)?;
    let pins = match store.pin(&held) {
        Ok(pins) => pins,
        Err(CacheError::OverBudget { .. } | CacheError::Full) if imprints.len() > 1 => {
            let (first, second) = imprints.split_at(imprints.len() / 2);
            let mut stats = apply_imprints(store, index, stroke, first)?;
            stats += apply_imprints(store, index, stroke, second)?;
            return Ok(stats);
        }
        Err(e) => return Err(e.into()),
    };
    times.pin = laps.lap();

    let mut frame = Frame::default();
    let applied = run_dabs(
        store, stroke, imprints, &queries, &mut frame, &mut times, &mut laps,
    );
    let finished = finish(store, index, &frame.reached, &mut times, &mut laps);
    drop(pins);
    applied?;
    let tiles_updated = finished?;

    Ok(ApplyStats {
        dabs: imprints.len(),
        frames: 1,
        candidates: frame_tiles.len(),
        tiles_edited: frame.edited.len(),
        vertices: frame.vertices,
        tiles_updated,
        tiles_unaccumulated: frame.unaccumulated,
        times,
    })
}

/// The tiles a frame over `frame_tiles` pins, in key order.
///
/// Those, every tile whose normals read one of them (what the neighbours' wave can evaluate
/// again), and every tile those tiles' normals read (what the wave reads): two rings of
/// neighbours.
///
/// # Errors
///
/// As [`TileStore::stale_after`].
fn headroom(store: &TileStore, frame_tiles: &[TileKey]) -> Result<Vec<TileKey>, CacheError> {
    let wave = store.stale_after(frame_tiles)?;
    store.stale_after(&wave)
}

/// `imprint`'s centre as a vector.
fn centre(imprint: &Imprint) -> Vec3 {
    let [x, y, z] = imprint.center;
    Vec3::new(x, y, z)
}

/// What a frame's dabs have done so far.
#[derive(Debug, Default)]
struct Frame {
    /// Every tile holding a vertex within a dab's radius: those whose positions moved.
    reached: BTreeSet<TileKey>,
    /// The box of each tile of `reached` as it is now, taken after the dab that last moved it;
    /// `None` for one with no finite position.
    boxes: std::collections::BTreeMap<TileKey, Option<Aabb>>,
    /// Every tile whose corrections a dab edited.
    edited: BTreeSet<TileKey>,
    /// Vertex moves, summed over the dabs.
    vertices: usize,
    /// Tiles written per dab for want of stroke state, summed over the dabs.
    unaccumulated: usize,
}

/// Step 3 of the module notes for each of `imprints` in turn.
///
/// `queries[k]` holds the tiles imprint `k`'s sphere touches by their boxes as the frame found
/// them, every one pinned. Each dab is brushed and written, and the tiles it moved made current
/// for the next, their boxes taken again. Stops at the first error, `frame` holding what the
/// dabs before it (and the failing one's brush, if it ran) reached.
///
/// # Errors
///
/// As [`TileStore::evaluate`], [`TileStore::update_tiles`] and
/// [`TileStore::bump_display_error`].
fn run_dabs(
    store: &mut TileStore,
    stroke: &mut Stroke,
    imprints: &[Imprint],
    queries: &[Vec<TileKey>],
    frame: &mut Frame,
    times: &mut PhaseTimes,
    laps: &mut Laps,
) -> Result<(), CacheError> {
    let dabs = imprints.iter().zip(queries).enumerate();
    for (k, (imprint, query)) in dabs {
        let keys = frame.tiles_for(imprint, query);
        let found = run_brush(store, &keys, imprint)?;
        frame.reached.extend(&found.reached);
        times.brush += laps.lap();

        let edited: Vec<TileKey> = found.edits.iter().map(|tile| tile.key).collect();
        frame.unaccumulated += stroke.begin_dab(store, &edited);
        let states = stroke.states(&edited);
        let edit = store.update_tiles(&edited, |n, i, j, c| {
            stroke_write(&found.edits, &states, n, i, j, c)
        });
        stroke.end_dab(store);
        edit?;
        frame.edited.extend(&edited);
        frame.vertices += found.vertices;
        raise_display_bounds(store, &found)?;
        times.edit += laps.lap();

        if k + 1 < imprints.len() && !found.reached.is_empty() {
            store.evaluate_many(&found.reached, Evaluation::Positions)?;
            frame.rebox(store, &found.reached);
            times.evaluate += laps.lap();
        }
    }
    Ok(())
}

impl Frame {
    /// The tiles `imprint` is brushed over, in key order (see "The frame's tiles" in the module
    /// notes): those of `query`, its tiles by their boxes as the frame found them, that the frame
    /// hasn't reached, and those it has reached whose boxes now touch its sphere.
    fn tiles_for(&self, imprint: &Imprint, query: &[TileKey]) -> Vec<TileKey> {
        let centre = centre(imprint);
        let radius_squared = imprint.radius * imprint.radius;
        let unmoved = query.iter().filter(|key| !self.reached.contains(key));
        let moved = self.boxes.iter().filter_map(|(key, aabb)| {
            aabb.filter(|aabb| aabb.distance_squared(centre) <= radius_squared)
                .map(|_| key)
        });
        let keys: BTreeSet<TileKey> = unmoved.chain(moved).copied().collect();
        keys.into_iter().collect()
    }

    /// Takes again the boxes of `moved`, tiles just made current, from their cached positions,
    /// on the store's workers.
    fn rebox(&mut self, store: &TileStore, moved: &[TileKey]) {
        let cached: Vec<Option<&[Vec3]>> = moved
            .iter()
            .map(|key| store.cached_positions(key))
            .collect();
        let boxes: Vec<Option<Aabb>> = store.par_map(&cached, |positions| {
            positions.and_then(|positions| Aabb::from_points(positions.iter().copied()))
        });
        let fresh = moved.iter().copied().zip(boxes);
        self.boxes.extend(fresh);
    }
}

/// Steps 4 to 6 of the module notes for the tiles `reached`.
///
/// Lists the tiles whose bounds may have changed, evaluates them, refits `index` and closes the
/// store's epoch, which it closes even if an earlier step failed. Returns how many tiles were
/// evaluated and refitted.
///
/// # Errors
///
/// As [`TileStore::stale_after`], [`TileStore::evaluate_many_with_bounds`],
/// [`SpatialIndex::refit_with`] and [`TileStore::close_epoch`]: the first that failed.
fn finish(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    reached: &BTreeSet<TileKey>,
    times: &mut PhaseTimes,
    laps: &mut Laps,
) -> Result<usize, SculptError> {
    let reached: Vec<TileKey> = reached.iter().copied().collect();
    let refitted = refit(store, index, &reached, times, laps);
    // The frame's edits and their re-evaluation are one epoch of the stale-tile feed, closed
    // whatever failed before: a reader relies on one closed epoch a call.
    let closed = store.close_epoch();
    times.refit += laps.lap();
    let updated = refitted?;
    closed?;
    Ok(updated)
}

/// Steps 4 to 6 of the module notes but the epoch: lists the tiles whose bounds may have changed
/// after `reached`, evaluates them and refits `index`. Returns how many tiles were evaluated and
/// refitted.
///
/// # Errors
///
/// As [`TileStore::stale_after`], [`TileStore::evaluate_many_with_bounds`] and
/// [`SpatialIndex::refit_with`].
fn refit(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    reached: &[TileKey],
    times: &mut PhaseTimes,
    laps: &mut Laps,
) -> Result<usize, SculptError> {
    let changed = store.stale_after(reached)?;
    times.neighbours += laps.lap();

    let wave = store.evaluate_many_with_bounds(&changed)?;
    times.evaluate += laps.lap();

    // The tiles were made current together just above, their bounds computed with their
    // normals: the refit takes those, and reads any other tile from the cache.
    index.refit_with(store, &changed, &wave)?;
    Ok(changed.len())
}

/// The deltas of one tile a dab edits: one per vertex of the tile, row-major, zero for each
/// vertex the tile doesn't move (outside the radius, or edited through another copy).
#[derive(Debug)]
struct TileDeltas {
    /// The tile.
    key: TileKey,
    /// The `side²` deltas along the frame normal.
    deltas: Vec<f32>,
}

/// What a dab's brush phase found: the deltas to write, and the tiles whose positions they move.
#[derive(Debug, Default)]
struct BrushOutput {
    /// The tiles owning a vertex to move, with their deltas, in key order.
    edits: Vec<TileDeltas>,
    /// Every tile holding a vertex within the radius, owned or a copy: the tiles whose positions
    /// the edits move, in key order.
    reached: Vec<TileKey>,
    /// Vertices to move, each once.
    vertices: usize,
}

/// Times the phases of a call, one lap after another.
#[derive(Debug)]
struct Laps(Instant);

impl Laps {
    /// Starts the first lap now.
    fn start() -> Self {
        Self(Instant::now())
    }

    /// Ends the current lap, returning its length, and starts the next.
    fn lap(&mut self) -> Duration {
        let now = Instant::now();
        let lap = now.duration_since(self.0);
        self.0 = now;
        lap
    }
}

/// A dab's brush phase, the draw action for `imprint` over every tile of `keys`.
///
/// It runs over the tiles' evaluated positions (pinned, so cached; in key order), and keeps
/// the deltas of the vertices each tile owns (see [`geom::Topology::owns`]).
///
/// Each tile is made current first, in order, as [`TileStore::evaluate`] makes it (a lookup,
/// as it is pinned and the tiles the previous dab moved were evaluated again together); then the
/// tiles are brushed on the store's workers ([`TileStore::par_map`]) from their cached
/// positions, each on its own, and gathered in key order. A tile whose positions aren't cached
/// is brushed on its own afterwards.
///
/// # Errors
///
/// As [`TileStore::evaluate`].
fn run_brush(
    store: &mut TileStore,
    keys: &[TileKey],
    imprint: &Imprint,
) -> Result<BrushOutput, CacheError> {
    for &key in keys {
        store.evaluate(key)?;
    }
    let brushed: Vec<Option<TileBrush>> = {
        let cache = &*store;
        let topology = cache.mesh().topology();
        let cached: Vec<(TileKey, Option<&[Vec3]>)> = keys
            .iter()
            .map(|key| (*key, cache.cached_positions(key)))
            .collect();
        cache.par_map(&cached, |&(key, positions)| {
            positions.map(|positions| brush_tile(topology, key, positions, imprint))
        })
    };
    let mut found = BrushOutput::default();
    let tiles = keys.iter().zip(brushed);
    for (&key, tile) in tiles {
        let tile = tile.map_or_else(|| brush_evicted(store, key, imprint), Ok)?;
        if !tile.reached {
            continue;
        }
        found.reached.push(key);
        if tile.moved > 0 {
            found.vertices += tile.moved;
            found.edits.push(TileDeltas {
                key,
                deltas: tile.deltas,
            });
        }
    }
    Ok(found)
}

/// What the brush does to `key`, a tile a later tile's evaluation pushed out of the cache
/// before the workers could brush it: evaluated again, then brushed here.
///
/// The positions are read back shared ([`TileStore::cached_positions`]) beside the mesh, not
/// copied out of the evaluation's mutable borrow. They are current right after the evaluation,
/// so the copy below it is a fallback that keeps the result right should that ever change.
///
/// # Errors
///
/// As [`TileStore::evaluate`].
pub(crate) fn brush_evicted(
    store: &mut TileStore,
    key: TileKey,
    imprint: &Imprint,
) -> Result<TileBrush, CacheError> {
    store.evaluate(key)?;
    if let Some(positions) = store.cached_positions(&key) {
        return Ok(brush_tile(store.mesh().topology(), key, positions, imprint));
    }
    let positions = store.evaluate(key)?.to_vec();
    Ok(brush_tile(
        store.mesh().topology(),
        key,
        &positions,
        imprint,
    ))
}

/// What the brush does to one tile: see [`brush_tile`].
#[derive(Debug)]
pub(crate) struct TileBrush {
    /// Whether a vertex of the tile is within the radius: its positions move.
    pub(crate) reached: bool,
    /// The deltas of the vertices the tile owns, one per vertex, row-major; zero elsewhere.
    pub(crate) deltas: Vec<f32>,
    /// How many of them aren't zero.
    pub(crate) moved: usize,
}

/// Runs the draw action for `imprint` over `positions`, tile `key`'s evaluated positions, and
/// keeps the deltas of the vertices the tile owns (see [`geom::Topology::owns`]).
pub(crate) fn brush_tile(
    topology: &geom::Topology,
    key: TileKey,
    positions: &[Vec3],
    imprint: &Imprint,
) -> TileBrush {
    let mut deltas = vec![0.0; positions.len()];
    let inside = draw_imprint(
        &Smoothstep,
        imprint,
        bytemuck::cast_slice(positions),
        &mut deltas,
    );
    if inside == 0 {
        return TileBrush {
            reached: false,
            deltas,
            moved: 0,
        };
    }
    let grid = vertices(key.side()).zip(&mut deltas);
    for ((i, j), delta) in grid {
        if *delta != 0.0 && !topology.owns(&key, i, j) {
            *delta = 0.0;
        }
    }
    let moved = deltas.iter().filter(|&&delta| delta != 0.0).count();
    TileBrush {
        reached: true,
        deltas,
        moved,
    }
}

/// The edit for vertex `(i, j)` of tile `edits[n]`, whose correction is stored as `c`.
///
/// `None` (no write) where its delta is zero; else `c` with its `z` the stroke's start for the
/// vertex plus the stroke's deltas for it so far, this one included ([`TileStroke::write`]), or,
/// for a tile without state (`states[n]` is `None`, see "The thrash limit" on [`Stroke`]),
/// `c.z` plus the delta.
///
/// Interior and border vertices so take the same formula: the stored value is rounded to the
/// tile's quantization when it is written, never added to dab by dab, and the interior's error
/// over the stroke is bounded per capture of the start (see "Error bound" on [`Stroke`]).
///
/// This keeps state per vertex, which is sound because [`TileStore::update_tiles`] calls its
/// closure exactly once per vertex of each tile, and only the vertex's owner has a non-zero
/// delta for it (copies are zeroed by [`brush_tile`]): the vertex's entries are read and written
/// by one thread a dab, and `c` is its value before this dab writes it. The same bits at any
/// worker count.
fn stroke_write(
    edits: &[TileDeltas],
    states: &[Option<&TileStroke>],
    n: usize,
    i: u32,
    j: u32,
    c: Vec3,
) -> Option<Vec3> {
    let tile = edits.get(n)?;
    let at = usize::try_from(j * tile.key.side() + i).ok()?;
    let delta = tile.deltas.get(at).copied()?;
    if delta == 0.0 {
        return None;
    }
    let written = match states.get(n)? {
        Some(state) => state.write(at, c.z, delta)?,
        None => c.z + delta,
    };
    Some(Vec3::new(c.x, c.y, written))
}

/// Raises the bound of every display tile a dab's writes can change.
///
/// By twice the farthest the dab moved a vertex (see [`TileStore::bump_display_error`]), so
/// the display tiles drawn until the store's refresh stay bounds.
///
/// The farthest is the largest delta plus the quantization step of the tiles it was written
/// into: a stroke rounds each written value to its tile's step once, so a dab's change can
/// exceed its delta by that much.
///
/// # Errors
///
/// As [`TileStore::bump_display_error`], and [`TileStore::correction`] for reading a tile's
/// step.
fn raise_display_bounds(store: &mut TileStore, found: &BrushOutput) -> Result<(), CacheError> {
    // Without a display layer there is nothing to raise, and no reason to scan every delta.
    if store.display_finest().is_none() {
        return Ok(());
    }
    let delta = found
        .edits
        .iter()
        .flat_map(|tile| tile.deltas.iter())
        .fold(0.0_f32, |m, d| m.max(d.abs()));
    if delta == 0.0 {
        return Ok(());
    }
    let mut step = 0.0_f32;
    for tile in &found.edits {
        step = step.max(store.correction(tile.key)?.scale());
    }
    store.bump_display_error(&found.reached, delta + step)
}

/// The vertex coordinates `(i, j)` of a tile with `side` vertices per side, in row-major order
/// (the order of its evaluated positions).
pub(crate) fn vertices(side: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..side).flat_map(move |j| (0..side).map(move |i| (i, j)))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
