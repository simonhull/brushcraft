//! Applying a frame's dabs: [`apply_dabs`], the draw brush for every dab a frame placed, the
//! dabs applied tile by tile, tiles in parallel on the store's workers.
//!
//! [`apply_dabs`] runs, in order:
//!
//! 1. **Query**: for each dab, the tiles whose boxes its sphere touches, from the spatial index as
//!    the frame found it.
//! 2. **Pin**: every tile of those queries is evaluated and pinned, together, so they stay resident
//!    for the frame; and with them every tile step 5 can evaluate or read, so that wave needs no
//!    room the pins haven't counted (see "Errors").
//! 3. **The dabs, in runs**: the frame's dabs are cut into runs of consecutive dabs whose tiles
//!    number at most the predictions the store keeps ([`TileStore::kept_predictions`]), and each
//!    run is applied tile by tile (see "Tile by tile"):
//!    - **Brush and edit**: each of the run's tiles, on the store's workers, takes every dab of the
//!      run that reaches it, in order, on a copy of its own corrections: `brush`'s draw action
//!      ([`brush::draw_imprint`], [`brush::draw_move`]) gives each vertex within the radius its
//!      move, its displacement along the dab's normal (its area normal: one direction for every
//!      vertex), and the move is written in the vertex's tangent frame ([`frame_delta`]) as its
//!      delta; the vertex's correction is set to the value the stroke builds for it, the correction
//!      it first found there plus every delta since (see [`Stroke`]), so the stored value is
//!      rounded to the tile's quantization once per stroke, not once per dab; and the tile's
//!      positions are made again from the prediction the store keeps of it
//!      ([`TileStore::kept_prediction`]), for the next dab, whose frames the deltas take too.
//!    - **Write**: the tiles are written back whole, together ([`TileStore::replace_corrections`]);
//!      the stroke's state takes what the run did ([`Stroke::end_frame`]).
//!    - **Positions**: unless it is the frame's last, the tiles the run reached get their positions
//!      again ([`TileStore::evaluate_many`], from the same kept predictions), and their boxes are
//!      taken again, for the next run's tiles (see "The frame's tiles").
//!
//!    A frame of one dab is applied as it always was, the dab's tiles brushed on the workers and
//!    written through [`TileStore::update_tiles`]; so is a run whose replay can't be exact (see
//!    "One dab at a time, still").
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
//! one at a time, a call each, bit for bit, at any worker count. The tests check it on a
//! generated project, against one call per dab, with strong dabs that grow tiles' scales
//! mid-frame, and with a stroke state too small for the stroke.
//!
//! ## Tile by tile
//!
//! Applying a run's dabs tile by tile, each tile on its own, is exact because nothing a dab
//! does to a tile depends on another tile, given the run's dabs:
//!
//! - **A tile's positions are its own.** A finest tile's positions are its prediction, from the
//!   level below, which the brush never edits, with its own corrections applied vertex by vertex in
//!   each vertex's tangent frame ([`geom::Frame::apply`]). No stencil reaches across tiles at the
//!   edited level. So a tile's copy of its corrections and its kept prediction give its positions
//!   after any of the run's writes, the bits the store's evaluation gives.
//! - **A vertex's delta is its own.** The draw action's delta for a vertex is a function of the
//!   vertex's position, its tangent frame (from the level below, which the brush never edits) and
//!   the dab alone.
//! - **A tile's quantization is its own.** The interior's scale covers its interior alone, and
//!   grows (re-quantizing the interior) only for its own writes, in the order the store's edit
//!   makes them: the replay writes through [`geom::CorrectionViewMut::update_interior_row`], the
//!   store's own edit of an interior, row by row, so a scale grows mid-run at the same vertex, with
//!   the same values re-quantized, as one dab at a time.
//! - **Shared vertices agree.** A border vertex is held by every tile it lies on, each copy's
//!   prediction and frame the same bits (geom's seam guarantee) and its correction the same, so
//!   each copy's position is the same, and every tile holding one finds the same delta for it,
//!   written in the same frame. Border values are stored exactly. The vertex's owner writes it from
//!   its stroke state; each other tile writes its copy from a copy of that state, taken from the
//!   stroke as the run began and carried along with the same arithmetic
//!   ([`crate::state::stroke_sum`]), so every copy gets the bits the owner's write gives all of
//!   them one at a time. The store checks it before writing ([`TileStore::replace_corrections`]):
//!   every copy of a border vertex written is among the run's tiles and holds the same bits, or
//!   nothing is written.
//! - **The stroke's state is decided ahead.** Which tiles hold stroke state, and which are released
//!   or written per dab, depends on the tiles every dab edits, and so on every tile. The run
//!   assumes no tile it edits loses its state or goes without, replays its tiles, then makes the
//!   stroke's decisions for the tiles each dab edited, dab by dab, on a copy of the stroke's
//!   bookkeeping ([`crate::state::Book`]): if they differ from what the replay assumed, the run is
//!   replayed under the decisions made, and those are checked again. The decisions made from the
//!   tiles each dab edits are the stroke's own, so a run whose decisions are what its replay
//!   assumed is the dabs one at a time. A tile written per dab writes the stored value plus the
//!   delta, a value of its own copy too.
//!
//! What stays once a frame: the neighbours, normals, bounds and refit, and the epoch's close.
//! The display tiles' bounds are raised once a run, by the run's dabs' displacements summed
//! rounding up ([`TileStore::bump_display_error`], whose own sums round up too): each display
//! tile's raise is at least the exact sum of the displacements that reach it, so a bound, as a
//! dab at a time gives one. The two needn't be the same bits: a dab at a time rounds once a
//! dab, a run once, and a run's sum also covers dabs that reach only some of its tiles.
//!
//! ## One dab at a time, still
//!
//! A run is applied one dab at a time instead, nothing written by its replay, when its replay
//! can't stand for it: a tile refuses a value ([`geom::CorrectionError::TooLarge`] or
//! [`geom::CorrectionError::NotFinite`]: one dab at a time, the dabs before it stay applied, as
//! "Errors" says), a tile lacks the positions, corrections or prediction it needs, the store
//! has no room to keep the run's predictions, the stroke's decisions differ again after the
//! second replay, or the store refuses its twins ([`CacheError::TwinsDisagree`], none should).
//! A frame of one dab is applied so too, as a run of one dab needs no prediction and gains
//! nothing.
//!
//! # The frame's tiles
//!
//! A tile's positions move only if a dab reaches it: if it holds a vertex within the dab's
//! radius (and so is brushed, and counted as reached), whether it owns the vertex and is edited,
//! or holds a copy of one its neighbour edits. So dab `k` brushes:
//!
//! - every tile its sphere touches by its box in the index as the frame found it, but those the
//!   frame has reached already: the others haven't moved since, so that box still bounds them;
//! - every tile the frame has reached whose box now, taken after the run that last moved it,
//!   touches its sphere.
//!
//! By induction over the dabs, that is every tile holding a vertex within dab `k`'s radius,
//! whatever its action does to the vertices it reaches, with no bound on how far they move. Only
//! the first set is known before the frame runs; it is what the frame pins, and the second is
//! within it, as a tile reached was brushed and so pinned.
//!
//! A run's tiles are its dabs' tiles so chosen at its start. Within the run, each tile keeps a
//! box for each of its rows, grown with every position the run gives it, and takes a dab only
//! over the rows whose box touches the dab's sphere: the same induction, row by row, inside the
//! tile. A tile that isn't among the run's tiles doesn't move during it: its first vertex to
//! come within a dab's radius would lie in its box as the frame found it, which so touches that
//! dab's sphere, and it would be among them.
//!
//! # Errors
//!
//! A store error ends the call. The dabs before it stay applied (the failing dab as
//! [`TileStore::update_tiles`] leaves it: a value a tile refuses stops a run's replay before
//! anything is written, and the run, applied one dab at a time, then fails at the same dab as
//! before; a run whose write fails part way is left as [`TileStore::replace_corrections`] leaves
//! it, the tiles before the failing one written), and the index is still brought up to date for
//! every tile a dab of the frame reached (steps 4 to 6); should that fail too, the first error is
//! returned and the index must be rebuilt. The stale-tile epoch is closed either way. If the
//! frame's tiles don't fit the budget's pins together ([`CacheError::OverBudget`], or
//! [`CacheError::Full`] for pins that fit the count but not the chunks), the frame is applied in
//! two halves, each a frame of its own, down to one dab: nothing has been written when the pin
//! fails. The pins also hold the tiles the neighbours' wave evaluates (the tiles whose normals
//! read a queried tile) and those their normals read, so after the edits that wave finds every
//! tile it touches pinned, and can't run out of room. Releasing the pins before the wave instead
//! would lose the predictions the store keeps for pinned tiles edited in place, which the next
//! frames reuse.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::AddAssign;
use std::time::{Duration, Instant};

use brush::{Imprint, Smoothstep, draw_imprint, draw_move};
use geom::{Prediction, TileKey, Vec3};
use store::{Aabb, CacheError, Evaluation, IndexError, SpatialIndex, TileStore};
use stroke::Dab;
use thiserror::Error;

use self::replay::{Plan, Run, TileOutcome};
use crate::draw::{DrawParamsError, check_dab};
use crate::state::{Book, Stroke, TileStroke};

mod replay;

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
    /// Dabs applied tile by tile, in runs (see "Tile by tile" in the module notes); the others
    /// were applied one at a time.
    pub tiled_dabs: usize,
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
        self.tiled_dabs += rhs.tiled_dabs;
        self.times += rhs.times;
    }
}

/// Applies `dabs`, a frame's dabs in sequence order, with the draw action.
///
/// Each dab moves every vertex of the index's level within its radius along the dab's normal by
/// `strength · falloff(d / radius)`, the falloff [`Smoothstep`] made as hard as the dab's hardness
/// ([`brush::Hardened`]), and `index` is brought up to date once for the frame (see the module
/// notes for the phases). The normal is the dab's area normal, as placement takes it: one
/// direction for every vertex the dab moves ([`brush::draw_move`]).
///
/// Each dab's values are taken as final: `strength` is the displacement at its centre in model
/// units, `hardness` in `0..=1` (`0`, the plain falloff), the normal any finite, non-zero vector
/// (used at unit length); the dab's tangent, angle and params aren't read by the draw action. The
/// result is that of applying the dabs one at a time, a call each, bit for bit (see "The same as
/// one dab at a time" in the module notes).
///
/// `store` is the project's ([`Project::store`](store::Project::store)). The dabs edit as part
/// of the store's current op: call it between
/// [`Project::begin_stroke`](store::Project::begin_stroke) and
/// [`Project::end_stroke`](store::Project::end_stroke), so the stroke is one op. `stroke` keeps
/// the value each edited vertex's correction is being built to across the stroke's dabs (see
/// [`Stroke`]); keep
/// one for the session and pass it to every call.
///
/// # Errors
///
/// [`SculptError::Dab`] for a dab whose centre or strength isn't finite, whose radius isn't
/// positive and finite, or whose normal has no direction, before anything is written.
/// [`SculptError::Cache`] if the store can't pin the tiles a dab touches (more than its budget
/// holds, for one dab), evaluate them or edit them; [`SculptError::Index`] if the index can't be
/// updated. A store error ends the call: the dabs before the failing one stay applied, and the
/// index is updated for them (see "Errors" in the module notes). The failing dab's edit is left as
/// [`TileStore::update_tiles`] leaves it, and the deltas it added to `stroke`'s values but didn't
/// write are written by the next dab that writes those vertices; don't apply it again, or its
/// deltas already added count twice.
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
pub(crate) fn imprint_of(dab: &Dab) -> Result<Imprint, DrawParamsError> {
    let [x, y, z] = dab.position;
    let [nx, ny, nz] = dab.normal;
    let normal = check_dab(
        Vec3::new(x, y, z),
        Vec3::new(nx, ny, nz),
        dab.radius,
        dab.strength,
    )?;
    Ok(Imprint::new(
        dab.position,
        [normal.x, normal.y, normal.z],
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
        tiled_dabs: frame.tiled,
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
    boxes: BTreeMap<TileKey, Option<Aabb>>,
    /// Every tile whose corrections a dab edited.
    edited: BTreeSet<TileKey>,
    /// Vertex moves, summed over the dabs.
    vertices: usize,
    /// Tiles written per dab for want of stroke state, summed over the dabs.
    unaccumulated: usize,
    /// Dabs applied tile by tile.
    tiled: usize,
}

/// Step 3 of the module notes for `imprints`, in runs.
///
/// `queries[k]` holds the tiles imprint `k`'s sphere touches by their boxes as the frame found
/// them, every one pinned. Each run is applied tile by tile ([`run_tiled`]), or one dab at a
/// time where it can't be ([`run_dab`]), and the tiles it moved made current for the next, their
/// boxes taken again; a frame of one dab is applied one dab at a time. Stops at the first error,
/// `frame` holding what the dabs before it (and the failing one's brush, if it ran) reached.
///
/// # Errors
///
/// As [`TileStore::evaluate`], [`TileStore::keep_predictions_of`], [`TileStore::frames_many`],
/// [`TileStore::update_tiles`], [`TileStore::replace_corrections`] and
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
    if imprints.len() == 1 {
        return run_dabs_one_by_one(store, stroke, imprints, queries, frame, times, laps);
    }
    let mut first = 0;
    while first < imprints.len() {
        let (end, keys) = frame.next_run(store.kept_predictions(), imprints, queries, first);
        let run = imprints.get(first..end).unwrap_or_default();
        let tiled = run_tiled(store, stroke, run, &keys, frame, times, laps)?;
        let more = end < imprints.len();
        if tiled.is_some() {
            frame.tiled += run.len();
        }
        match tiled {
            Some(reached) if more && !reached.is_empty() => {
                store.evaluate_many(&reached, Evaluation::Positions)?;
                frame.rebox(store, &reached);
                times.evaluate += laps.lap();
            }
            Some(_) => {}
            None => {
                let run_queries = queries.get(first..end).unwrap_or_default();
                let dabs = run.iter().zip(run_queries).enumerate();
                for (k, (imprint, query)) in dabs {
                    let last = !more && first + k + 1 == imprints.len();
                    run_dab(store, stroke, imprint, query, last, frame, times, laps)?;
                }
            }
        }
        first = end;
    }
    Ok(())
}

/// [`run_dabs`] one dab at a time (see "One dab at a time, still" in the module notes).
///
/// # Errors
///
/// As [`run_dabs`].
fn run_dabs_one_by_one(
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
        let last = k + 1 == imprints.len();
        run_dab(store, stroke, imprint, query, last, frame, times, laps)?;
    }
    Ok(())
}

/// One dab of step 3 of the module notes, on its own.
///
/// Brushed over its tiles on the workers, written through [`TileStore::update_tiles`], and,
/// unless it is the frame's `last`, the tiles it moved made current for the next, their boxes
/// taken again.
///
/// # Errors
///
/// As [`run_dabs`].
#[expect(
    clippy::too_many_arguments,
    reason = "one dab of the frame's loop: the frame's state, the dab and the phase timers"
)]
fn run_dab(
    store: &mut TileStore,
    stroke: &mut Stroke,
    imprint: &Imprint,
    query: &[TileKey],
    last: bool,
    frame: &mut Frame,
    times: &mut PhaseTimes,
    laps: &mut Laps,
) -> Result<(), CacheError> {
    let keys = frame.tiles_for(imprint, query);
    let mut found = run_brush(store, &keys, imprint)?;
    frame.reached.extend(&found.reached);
    times.brush += laps.lap();

    let edited: Vec<TileKey> = found.edits.iter().map(|tile| tile.key).collect();
    let deltas = frame_deltas(store, imprint, &mut found.edits)?;
    frame.unaccumulated += stroke.begin_dab(store, &edited);
    let states = stroke.states(&edited);
    let edit = store.update_tiles(&edited, |n, i, j, c| {
        stroke_write(&found.edits, &deltas, &states, n, i, j, c)
    });
    stroke.end_dab(store);
    edit?;
    frame.edited.extend(&edited);
    frame.vertices += found.vertices;
    raise_display_bounds(store, &found)?;
    times.edit += laps.lap();

    if !last && !found.reached.is_empty() {
        store.evaluate_many(&found.reached, Evaluation::Positions)?;
        frame.rebox(store, &found.reached);
        times.evaluate += laps.lap();
    }
    Ok(())
}

/// A run of the frame's dabs, `imprints`, applied tile by tile over `keys`, the run's tiles.
///
/// See "Tile by tile" in the module notes. Returns the tiles the run reached, in key order, or
/// `None` if the run must be applied one dab at a time instead, nothing having been written.
///
/// # Errors
///
/// As [`TileStore::evaluate_many`], [`TileStore::frames_many`] and
/// [`TileStore::keep_predictions_of`] before anything is written (but [`CacheError::Full`] and
/// [`CacheError::OverBudget`] from keeping the predictions, which leave the run to be applied one
/// dab at a time); as [`TileStore::replace_corrections`] (nothing written, but the stroke's state
/// readied for the frame) and [`TileStore::bump_display_error`] after.
fn run_tiled(
    store: &mut TileStore,
    stroke: &mut Stroke,
    imprints: &[Imprint],
    keys: &[TileKey],
    frame: &mut Frame,
    times: &mut PhaseTimes,
    laps: &mut Laps,
) -> Result<Option<Vec<TileKey>>, CacheError> {
    let predicted = imprints.len() > 1;
    store.evaluate_many(keys, Evaluation::Positions)?;
    if predicted {
        // Too little room to keep the run's predictions isn't the frame's failure: one dab at
        // a time needs no more room than its pins, which it has.
        let kept = store.keep_predictions_of(keys);
        #[cfg(test)]
        let kept = if replay::injected(store.id(), replay::Fault::StarvePredictions) {
            Err(CacheError::Full)
        } else {
            kept
        };
        match kept {
            Ok(true) => {}
            Ok(false) | Err(CacheError::Full | CacheError::OverBudget { .. }) => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    // A run of one dab has no kept predictions to take its frames from: they are listed.
    let frames = if predicted {
        Vec::new()
    } else {
        store.frames_many(keys)?
    };
    times.evaluate += laps.lap();
    stroke.begin_frame(store);
    let book = stroke.book().clone();
    let mut plan = Plan::Continuous;
    // A second replay at most: under the grants the first one's dabs give, which hold unless
    // the second's dabs edit other tiles.
    for _ in 0..2 {
        let replayed = replay_run(store, stroke, imprints, keys, &frames, &plan, predicted);
        let Some(outcomes) = replayed else {
            return Ok(None);
        };
        if !replay::all_done(&outcomes) {
            return Ok(None);
        }
        let (grants, after, without) = Plan::of(&book, &outcomes, imprints.len());
        if replay::used(&plan, &book, &outcomes) != grants {
            plan = Plan::Granted(grants);
            continue;
        }
        times.brush += laps.lap();
        let written = replay::written(&outcomes);
        if !written.is_empty() {
            // The store checks the twins before it writes anything; none should disagree (see
            // "Tile by tile" in the module notes), and a run that broke them is applied one dab
            // at a time.
            match store.replace_corrections(&written) {
                Ok(()) => {}
                Err(CacheError::TwinsDisagree { .. }) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
        let reached = commit_run(store, stroke, outcomes, after, without, frame)?;
        times.edit += laps.lap();
        return Ok(Some(reached));
    }
    Ok(None)
}

/// Each tile of `keys` replayed over the run's dabs `imprints` under `plan`, on the store's
/// workers, in key order.
///
/// `None` if a tile's positions, corrections or (when `predicted`) prediction aren't cached. A
/// tile without a kept prediction takes its frames from `frames`, aligned with `keys`.
fn replay_run(
    store: &TileStore,
    stroke: &Stroke,
    imprints: &[Imprint],
    keys: &[TileKey],
    frames: &[Vec<geom::Frame>],
    plan: &Plan,
    predicted: bool,
) -> Option<Vec<TileOutcome>> {
    let inputs = replay::inputs(store, keys, predicted, frames)?;
    let run = Run {
        imprints,
        plan,
        stroke,
        topology: store.mesh().topology(),
        #[cfg(test)]
        break_twins: replay::injected(store.id(), replay::Fault::BreakTwins),
    };
    Some(store.par_map(&inputs, |&input| replay::replay_tile(run, input)))
}

/// Takes a run's `outcomes` (in key order), its tiles written, as done, and returns the tiles it
/// reached, in key order.
///
/// The stroke's state, its book `after` the run's dabs; the display tiles' bounds; and what the
/// frame has done.
///
/// # Errors
///
/// As [`TileStore::bump_display_error`].
fn commit_run(
    store: &mut TileStore,
    stroke: &mut Stroke,
    outcomes: Vec<TileOutcome>,
    after: Book,
    without: usize,
    frame: &mut Frame,
) -> Result<Vec<TileKey>, CacheError> {
    let reached: Vec<TileKey> = outcomes
        .iter()
        .filter(|outcome| outcome.reached)
        .map(|outcome| outcome.key)
        .collect();
    let displacement = run_displacement(&outcomes);
    frame.reached.extend(&reached);
    frame.unaccumulated += without;
    let mut finals = Vec::new();
    for outcome in outcomes {
        if !outcome.edits.is_empty() {
            frame.edited.insert(outcome.key);
        }
        let moved: usize = outcome.edits.iter().map(|edit| edit.moved).sum();
        frame.vertices += moved;
        finals.extend(outcome.state);
    }
    stroke.end_frame(store, after, finals);
    if store.display_finest().is_some() && displacement > 0.0 {
        store.bump_display_error(&reached, displacement)?;
    }
    Ok(reached)
}

/// The farthest a run's dabs moved a finest vertex, summed over its dabs.
///
/// For [`TileStore::bump_display_error`]: for each dab, its largest delta plus the largest
/// quantization step of the tiles it was written into (as [`raise_display_bounds`] takes a
/// dab's), summed rounding up, as the store sums a frame's.
fn run_displacement(outcomes: &[TileOutcome]) -> f32 {
    let mut dabs: BTreeMap<usize, (f32, f32)> = BTreeMap::new();
    let edits = outcomes.iter().flat_map(|outcome| outcome.edits.iter());
    for edit in edits {
        let (delta, step) = dabs.entry(edit.dab).or_insert((0.0, 0.0));
        *delta = delta.max(edit.delta);
        *step = step.max(edit.scale);
    }
    dabs.values()
        .filter(|&&(delta, _)| delta != 0.0)
        .map(|&(delta, step)| delta + step)
        .fold(
            0.0_f32,
            |sum, d| {
                if sum == 0.0 { d } else { (sum + d).next_up() }
            },
        )
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

    /// The next run of the frame's dabs from dab `first`, and its tiles, in key order (see
    /// "Tile by tile" in the module notes): `first` and the dabs after it while the run's tiles
    /// number at most `kept`, the predictions the store keeps, so each of them can have its
    /// own. Each dab's tiles are [`Frame::tiles_for`]'s. Returns the end of the run (the dab
    /// after its last) and its tiles; a run of one dab may have more tiles than that, as its
    /// tiles need no prediction.
    fn next_run(
        &self,
        kept: usize,
        imprints: &[Imprint],
        queries: &[Vec<TileKey>],
        first: usize,
    ) -> (usize, Vec<TileKey>) {
        let tiles_of = |k: usize| {
            let imprint = imprints.get(k)?;
            let query = queries.get(k)?;
            Some(self.tiles_for(imprint, query))
        };
        let mut keys: BTreeSet<TileKey> = tiles_of(first).unwrap_or_default().into_iter().collect();
        let mut end = first + 1;
        while let Some(more) = tiles_of(end) {
            let grown: BTreeSet<TileKey> = keys.iter().chain(&more).copied().collect();
            if grown.len() > kept {
                break;
            }
            keys = grown;
            end += 1;
        }
        (end, keys.into_iter().collect())
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

/// The displacements of one tile a dab edits: one per vertex of the tile, row-major, zero for
/// each vertex the tile doesn't move (outside the radius, or edited through another copy).
#[derive(Debug)]
struct TileDeltas {
    /// The tile.
    key: TileKey,
    /// The `side²` displacements along the dab's normal ([`brush::draw_imprint`]).
    deltas: Vec<f32>,
    /// The `side²` deltas, the dab's moves written in each vertex's frame ([`frame_delta`]),
    /// where the brush had the tile's frames at hand (a kept prediction); else made by
    /// [`frame_deltas`].
    moves: Option<Vec<Vec3>>,
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
    // The frames the deltas are written along: kept predictions, so the brush writes each tile's
    // deltas as it finds its displacements (see `frame_deltas` for the rest).
    match store.keep_predictions_of(keys) {
        Ok(_) | Err(CacheError::Full | CacheError::OverBudget { .. }) => {}
        Err(e) => return Err(e),
    }
    let brushed: Vec<Option<(TileBrush, Option<Vec<Vec3>>)>> = {
        let cache = &*store;
        let topology = cache.mesh().topology();
        let cached: Vec<CachedTile<'_>> = keys
            .iter()
            .map(|key| {
                (
                    *key,
                    cache.cached_positions(key),
                    cache.kept_prediction(key),
                )
            })
            .collect();
        cache.par_map(&cached, |&(key, positions, prediction)| {
            let tile = brush_tile(topology, key, positions?, imprint);
            let moves = prediction.filter(|_| tile.moved > 0).map(|prediction| {
                tile_frame_deltas(key, &tile.deltas, imprint, |i, j| prediction.frame(i, j))
            });
            Some((tile, moves))
        })
    };
    let mut found = BrushOutput::default();
    let tiles = keys.iter().zip(brushed);
    for (&key, outcome) in tiles {
        let (tile, moves) = match outcome {
            Some(done) => done,
            None => (brush_evicted(store, key, imprint)?, None),
        };
        if !tile.reached {
            continue;
        }
        found.reached.push(key);
        if tile.moved > 0 {
            found.vertices += tile.moved;
            found.edits.push(TileDeltas {
                key,
                deltas: tile.deltas,
                moves,
            });
        }
    }
    Ok(found)
}

/// A tile as [`run_brush`] reads it from the cache: its key, its positions and its kept
/// prediction, each if cached.
type CachedTile<'a> = (TileKey, Option<&'a [Vec3]>, Option<&'a Prediction>);

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
    /// The displacements of the vertices the tile owns, along the dab's normal, one per vertex,
    /// row-major; zero elsewhere.
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

/// `imprint`'s move of a vertex whose displacement is `w` ([`brush::draw_move`]: `w` along the
/// dab's normal), written in the vertex's tangent `frame`: the delta its correction takes.
///
/// The frame is orthonormal, so the components are the move's along its axes, and the corrected
/// position ([`geom::Frame::apply`]) moves by the move itself, up to rounding. The one projection
/// every path that writes draw's deltas makes (a dab at a time, a run tile by tile), so they give
/// the same bits; and a vertex shared by several tiles has the same frame in each, so every copy
/// gets the same delta.
pub(crate) fn frame_delta(frame: &geom::Frame, imprint: &Imprint, w: f32) -> Vec3 {
    let [x, y, z] = draw_move(imprint, w);
    let moved = Vec3::new(x, y, z);
    Vec3::new(
        frame.t().dot(moved),
        frame.b().dot(moved),
        frame.n().dot(moved),
    )
}

/// The deltas of a dab's edits: `imprint`'s move of each vertex it moves, written in the vertex's
/// tangent frame ([`frame_delta`]).
///
/// For each of `edits`, in order, one per vertex of its tile, row-major, zero where its
/// displacement is zero. The brush made them already for the tiles whose predictions the store
/// keeps ([`run_brush`] has it keep them, [`TileStore::keep_predictions_of`], which the frame's
/// evaluation of the tiles after the edit reuses, so it costs no prediction the dab didn't make
/// already). The rest, for edits of more tiles than the store keeps predictions for, take copies of
/// their frames ([`TileStore::frames_many`]), on the store's workers. Either way the frames are the
/// bits the tiles' evaluation uses.
///
/// # Errors
///
/// As [`TileStore::frames_many`].
fn frame_deltas(
    store: &mut TileStore,
    imprint: &Imprint,
    edits: &mut [TileDeltas],
) -> Result<Vec<Vec<Vec3>>, CacheError> {
    let missing: Vec<TileKey> = edits
        .iter()
        .filter(|tile| tile.moves.is_none())
        .map(|tile| tile.key)
        .collect();
    if !missing.is_empty() {
        let copies = store.frames_many(&missing)?;
        let lacking = edits.iter_mut().filter(|tile| tile.moves.is_none());
        let listed: Vec<(&mut TileDeltas, &[geom::Frame])> = lacking
            .zip(&copies)
            .map(|(tile, frames)| (tile, frames.as_slice()))
            .collect();
        let made: Vec<Vec<Vec3>> = store.par_map(&listed, |(tile, frames)| {
            let side = tile.key.side();
            tile_frame_deltas(tile.key, &tile.deltas, imprint, |i, j| {
                frames.get(usize::try_from(j * side + i).ok()?).copied()
            })
        });
        for ((tile, _), moves) in listed.into_iter().zip(made) {
            tile.moves = Some(moves);
        }
    }
    Ok(edits
        .iter_mut()
        .map(|tile| tile.moves.take().unwrap_or_default())
        .collect())
}

/// The deltas of tile `key` for its displacements `deltas` by `imprint`, `frame(i, j)` giving
/// vertex `(i, j)`'s frame (see [`frame_deltas`]).
///
/// A vertex it gives none (never, for a tile and frames of the same side) takes a delta that isn't
/// finite, which the store refuses to write.
fn tile_frame_deltas(
    key: TileKey,
    deltas: &[f32],
    imprint: &Imprint,
    frame: impl Fn(u32, u32) -> Option<geom::Frame>,
) -> Vec<Vec3> {
    let moves = vertices(key.side()).zip(deltas);
    moves
        .map(|((i, j), &w)| {
            if w == 0.0 {
                return Vec3::ZERO;
            }
            frame(i, j).map_or(Vec3::new(f32::NAN, f32::NAN, f32::NAN), |frame| {
                frame_delta(&frame, imprint, w)
            })
        })
        .collect()
}

/// The edit for vertex `(i, j)` of tile `edits[n]`, whose correction is stored as `c`, its delta
/// `deltas[n]`'s (the dab's move in its frame, [`frame_deltas`]).
///
/// `None` (no write) where its displacement is zero; else the value the stroke builds for the
/// vertex, the correction it first found there plus every delta since, this one included
/// ([`TileStroke::write`]), or, for a tile without state (`states[n]` is `None`, see "The thrash
/// limit" on [`Stroke`]), `c` plus the delta.
///
/// Interior and border vertices so take the same formula: the stored value is rounded to the
/// tile's quantization when it is written, never added to dab by dab, and the interior's error
/// over the stroke is bounded per capture of the start (see "Error bound" on [`Stroke`]).
///
/// This keeps state per vertex, which is sound because [`TileStore::update_tiles`] calls its
/// closure exactly once per vertex of each tile, and only the vertex's owner has a non-zero
/// displacement for it (copies are zeroed by [`brush_tile`]): the vertex's entries are read and
/// written by one thread a dab, and `c` is its value before this dab writes it. The same bits at
/// any worker count.
#[expect(
    clippy::many_single_char_names,
    reason = "the store's edit closure's names: tile n, vertex (i, j), correction c, and w"
)]
fn stroke_write(
    edits: &[TileDeltas],
    deltas: &[Vec<Vec3>],
    states: &[Option<&TileStroke>],
    n: usize,
    i: u32,
    j: u32,
    c: Vec3,
) -> Option<Vec3> {
    let tile = edits.get(n)?;
    let at = usize::try_from(j * tile.key.side() + i).ok()?;
    let w = tile.deltas.get(at).copied()?;
    if w == 0.0 {
        return None;
    }
    let delta = deltas.get(n)?.get(at).copied()?;
    match states.get(n)? {
        Some(state) => state.write(at, c, delta),
        None => Some(c + delta),
    }
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
