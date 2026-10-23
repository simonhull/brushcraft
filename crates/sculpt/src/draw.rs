//! The draw step: one dab of the draw brush, pushing the surface out along its normal.
//!
//! [`draw_step`] runs, in order:
//!
//! 1. **Query**: the tiles whose boxes the brush's sphere touches, from the spatial index.
//! 2. **Pin**: those tiles are evaluated and pinned, so they stay resident for the step.
//! 3. **Kernel**: `kernels::draw` over each tile's evaluated positions, giving each vertex within
//!    the radius its delta along the normal axis of its tangent frame. Every delta is computed
//!    before any correction is written: an edit moves the positions of every tile holding a copy of
//!    the vertex, and a later tile's kernel must see the surface as the step found it.
//! 4. **Edit**: each delta is added to the `z` of its vertex's correction (read-modify-write, `c.z
//!    += delta`) through [`TileStore::edit_tile`], which keeps seam twins in step.
//! 5. **Neighbours**: the tiles whose bounds may have changed are listed. That is every tile
//!    holding a vertex within the radius, whose positions moved, and every tile whose normals read
//!    one of those, whose normal cones may have turned: the tiles each reached tile's normals read,
//!    gathered from the base mesh's topology.
//! 6. **Evaluate**: those tiles are evaluated again, positions and normals.
//! 7. **Refit**: their bounds are recomputed and the index's tree refitted.
//!
//! Corrections live in each vertex's tangent frame `(t, b, n)` of the predicted surface, so a
//! push along the frame normal `n` is a change of the correction's `z` alone, with no frames to
//! fetch. (Brushes that move along the evaluated normal need the frames; they come later.)
//!
//! # Seam twins
//!
//! A vertex on a tile's border is held by every tile that shares it, each with its own copy of
//! its position and correction, and the copies agree bit for bit. Each of those tiles is a
//! candidate, and its kernel finds the vertex within the radius with the same delta. Adding the
//! delta once per copy would push the vertex once per tile that holds it, so exactly one copy
//! edits it: the least of its copies (see [`geom::Topology::owns`]). `edit_tile` then writes the
//! new value into every copy.

use std::collections::BTreeSet;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

use geom::{EvalError, Region, TileKey, Vec3, normal_tiles};
use kernels::Smoothstep;
use store::{CacheError, IndexError, Project, SpatialIndex, TileEdit, TileStore};
use thiserror::Error;

/// Why [`DrawParams::new`] refused its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DrawParamsError {
    /// The brush's centre has a component that isn't finite.
    #[error("the brush centre is not finite")]
    Center,
    /// The brush's radius isn't positive and finite.
    #[error("the brush radius is not positive and finite")]
    Radius,
    /// The brush's strength isn't finite.
    #[error("the brush strength is not finite")]
    Strength,
}

/// One dab of the draw brush: where it is, how far it reaches, and how far it pushes.
///
/// Every value is checked by [`DrawParams::new`], so a step never sees a brush that reaches
/// nowhere or pushes by a NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawParams {
    /// The brush's centre, in model space.
    center: Vec3,
    /// How far the brush reaches from its centre: positive and finite.
    radius: f32,
    /// How far a vertex at the centre is pushed along its frame normal: finite, and negative to
    /// push inwards.
    strength: f32,
}

impl DrawParams {
    /// A dab centred at `center`, reaching `radius` and pushing a vertex at its centre by
    /// `strength` along the frame normal (inwards if negative), fading with the smoothstep
    /// falloff to nothing at the rim.
    ///
    /// # Errors
    ///
    /// [`DrawParamsError::Center`] if `center` isn't finite, [`DrawParamsError::Radius`] if
    /// `radius` isn't positive and finite, [`DrawParamsError::Strength`] if `strength` isn't
    /// finite.
    pub fn new(center: Vec3, radius: f32, strength: f32) -> Result<Self, DrawParamsError> {
        if !center.is_finite() {
            return Err(DrawParamsError::Center);
        }
        if !(radius > 0.0 && radius.is_finite()) {
            return Err(DrawParamsError::Radius);
        }
        if !strength.is_finite() {
            return Err(DrawParamsError::Strength);
        }
        Ok(Self {
            center,
            radius,
            strength,
        })
    }

    /// The brush's centre, in model space.
    #[must_use]
    pub const fn center(&self) -> Vec3 {
        self.center
    }

    /// How far the brush reaches from its centre.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        self.radius
    }

    /// How far a vertex at the centre is pushed along its frame normal.
    #[must_use]
    pub const fn strength(&self) -> f32 {
        self.strength
    }
}

/// Why a [`draw_step`] failed.
#[derive(Debug, Error)]
pub enum SculptError {
    /// The store couldn't pin, evaluate or edit a tile.
    #[error("the tile store failed")]
    Cache(#[from] CacheError),
    /// The spatial index couldn't be brought up to date.
    #[error("updating the spatial index failed")]
    Index(#[from] IndexError),
    /// Listing the tiles whose normals read an edited tile failed (not for a valid mesh).
    #[error("listing the tiles an edit reaches failed")]
    Eval(#[from] EvalError),
}

/// How long each phase of a [`draw_step`] took (see the module notes for the phases).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseTimes {
    /// Querying the spatial index for the tiles the brush touches.
    pub query: Duration,
    /// Evaluating and pinning those tiles.
    pub pin: Duration,
    /// Running the kernel over their evaluated positions.
    pub kernel: Duration,
    /// Writing the deltas into the corrections.
    pub edit: Duration,
    /// Listing the tiles whose bounds the edits may change: those reached, and their
    /// neighbours whose normals read them.
    pub neighbours: Duration,
    /// Evaluating the changed tiles again, positions and normals.
    pub evaluate: Duration,
    /// Recomputing their bounds and refitting the index.
    pub refit: Duration,
}

impl PhaseTimes {
    /// The whole step: every phase together.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.query
            + self.pin
            + self.kernel
            + self.edit
            + self.neighbours
            + self.evaluate
            + self.refit
    }
}

impl AddAssign for PhaseTimes {
    /// Adds each phase's time, to total the phases of several steps.
    fn add_assign(&mut self, rhs: Self) {
        self.query += rhs.query;
        self.pin += rhs.pin;
        self.kernel += rhs.kernel;
        self.edit += rhs.edit;
        self.neighbours += rhs.neighbours;
        self.evaluate += rhs.evaluate;
        self.refit += rhs.refit;
    }
}

/// What one [`draw_step`] did, and how long each phase took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepStats {
    /// Tiles the brush's sphere touches by their boxes: queried, pinned and run through the
    /// kernel.
    pub candidates: usize,
    /// Tiles whose corrections the step edited: those owning a vertex it moved (each also
    /// writes the copies its neighbours hold of its border vertices).
    pub tiles_edited: usize,
    /// Vertices the step moved, each counted once however many tiles hold a copy of it.
    pub vertices: usize,
    /// Tiles evaluated again and refitted in the index: those holding a moved vertex, and those
    /// whose normals read one of them.
    pub tiles_updated: usize,
    /// How long each phase took.
    pub times: PhaseTimes,
}

/// The deltas of one tile the step edits: one per vertex of the tile, row-major, zero for each
/// vertex the tile doesn't move (outside the radius, or edited through another copy).
#[derive(Debug)]
struct TileDeltas {
    /// The tile.
    key: TileKey,
    /// The `side²` deltas along the frame normal.
    deltas: Vec<f32>,
}

/// What the kernel phase found: the deltas to write, and the tiles whose positions they move.
#[derive(Debug, Default)]
struct KernelOutput {
    /// The tiles owning a vertex to move, with their deltas, in key order.
    edits: Vec<TileDeltas>,
    /// Every tile holding a vertex within the radius, owned or a copy: the tiles whose positions
    /// the edits move.
    reached: Vec<TileKey>,
    /// Vertices to move, each once.
    vertices: usize,
}

/// Times the phases of a step, one lap after another.
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

/// One dab of the draw brush: pushes the surface out along its frame normal near `params`.
///
/// Every vertex of the index's level within the radius moves by `strength · smoothstep(d /
/// radius)`, and `index` is brought up to date (see the module notes for the phases).
///
/// The step edits as part of the store's current op: call it between
/// [`Project::begin_stroke`] and [`Project::end_stroke`], so the stroke's steps are one op. The
/// vertices edited are those of the level `index` holds, the finest the store evaluates.
///
/// # Errors
///
/// [`SculptError::Cache`] if the store can't pin the tiles the brush touches (more than its
/// budget holds, for one), evaluate them or edit them; [`SculptError::Index`] if the index
/// can't be updated; [`SculptError::Eval`] never, for a valid mesh. An error in the edit phase
/// leaves the tiles edited before it edited (each tile's edit is whole), and the index not
/// updated for them: rebuild it.
pub fn draw_step(
    project: &mut Project,
    index: &mut SpatialIndex,
    params: DrawParams,
) -> Result<StepStats, SculptError> {
    let store = project.store();
    let mut laps = Laps::start();
    let mut times = PhaseTimes::default();

    let candidates: Vec<TileKey> = index.query_sphere(params.center, params.radius).collect();
    times.query = laps.lap();

    let pins = store.pin(&candidates)?;
    times.pin = laps.lap();

    let found = run_kernel(store, pins.keys(), params)?;
    times.kernel = laps.lap();

    for tile in &found.edits {
        store.edit_tile(tile.key, |edit| add_deltas(edit, &tile.deltas))??;
    }
    times.edit = laps.lap();

    let changed = bounds_changed(store.mesh(), &found.reached)?;
    times.neighbours = laps.lap();

    for &key in &changed {
        store.evaluate(key)?;
        store.normals(key)?;
    }
    times.evaluate = laps.lap();

    index.update(store, &changed)?;
    times.refit = laps.lap();
    drop(pins);

    Ok(StepStats {
        candidates: candidates.len(),
        tiles_edited: found.edits.len(),
        vertices: found.vertices,
        tiles_updated: changed.len(),
        times,
    })
}

/// The kernel phase: runs the draw kernel over the evaluated positions of every tile of `keys`
/// (pinned, so cached), and keeps the deltas of the vertices each tile owns (see
/// [`geom::Topology::owns`]).
///
/// # Errors
///
/// As [`TileStore::evaluate`].
fn run_kernel(
    store: &mut TileStore,
    keys: &[TileKey],
    params: DrawParams,
) -> Result<KernelOutput, CacheError> {
    let center = [params.center.x, params.center.y, params.center.z];
    let mut found = KernelOutput::default();
    for &key in keys {
        let positions = store.evaluate(key)?;
        let mut deltas = vec![0.0; positions.len()];
        let inside = kernels::draw(
            &Smoothstep,
            center,
            params.radius,
            params.strength,
            bytemuck::cast_slice(positions),
            &mut deltas,
        );
        if inside == 0 {
            continue;
        }
        found.reached.push(key);
        let topology = store.mesh().topology();
        for ((i, j), delta) in vertices(key.side()).zip(&mut deltas) {
            if *delta != 0.0 && !topology.owns(&key, i, j) {
                *delta = 0.0;
            }
        }
        let moved = deltas.iter().filter(|&&delta| delta != 0.0).count();
        if moved > 0 {
            found.vertices += moved;
            found.edits.push(TileDeltas { key, deltas });
        }
    }
    Ok(found)
}

/// The edit phase for one tile: adds each nonzero delta to the `z` of its vertex's correction,
/// in row-major order.
///
/// # Errors
///
/// As [`TileEdit::get`] and [`TileEdit::set`].
fn add_deltas(edit: &mut TileEdit<'_>, deltas: &[f32]) -> Result<(), CacheError> {
    for ((i, j), &delta) in vertices(edit.key().side()).zip(deltas) {
        if delta != 0.0 {
            let c = edit.get(i, j)?;
            edit.set(i, j, Vec3::new(c.x, c.y, c.z + delta))?;
        }
    }
    Ok(())
}

/// The tiles whose bounds an edit of the `reached` tiles may change, each once, in key order.
///
/// That is each of them (their positions move), and every tile whose normals read one of them
/// (their normal cones may turn).
///
/// The tiles a tile's normals read ([`normal_tiles`]: the tile, its apron and the rings at its
/// corners) are its neighbours, and neighbourhood goes both ways, so the tiles reading a
/// reached tile are among the tiles it reads.
///
/// # Errors
///
/// As [`normal_tiles`].
fn bounds_changed(mesh: &geom::BaseMesh, reached: &[TileKey]) -> Result<Vec<TileKey>, EvalError> {
    let mut changed = BTreeSet::new();
    for key in reached {
        changed.extend(normal_tiles(mesh, Region::of_tile(key))?);
    }
    Ok(changed.into_iter().collect())
}

/// The vertex coordinates `(i, j)` of a tile with `side` vertices per side, in row-major order
/// (the order of its evaluated positions).
fn vertices(side: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..side).flat_map(move |j| (0..side).map(move |i| (i, j)))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
