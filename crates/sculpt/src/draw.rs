//! The draw step: one dab of the draw brush, pushing the surface out along the dab's normal.
//!
//! [`draw_step`] is [`apply_dabs`](crate::apply_dabs) for one dab, with the plain smoothstep
//! falloff: the step the M0 benchmark measured, kept so its history stays comparable. See the
//! `apply` module notes for its phases (`query`, `pin`, `brush`, `edit`, `neighbours`,
//! `evaluate`, `refit`, as [`PhaseTimes`] times them).
//!
//! Every vertex the dab reaches moves along the one direction, the dab's normal (for a placed
//! dab, its area normal). Corrections live in each vertex's tangent frame `(t, b, n)` of the
//! predicted surface, which isn't the dab's, so the move is written in the vertex's frame, all
//! three components of its correction (see `frame_delta` in the `apply` module), with the frames
//! from the level below: those of the predictions the store keeps for the tiles a stroke edits
//! ([`TileStore::kept_prediction`]), read as each vertex is written
//! ([`TileStore::update_tiles_framed`]); a tile the stroke enters is predicted by the workers
//! brushing the dab's tiles ([`TileStore::predictor`]) and kept, and the frame's evaluation after
//! the edit reuses them all. Frames are listed afresh ([`TileStore::frames_many`]) only for a dab
//! of more tiles than the store keeps predictions for.
//!
//! # Seam twins
//!
//! A vertex on a tile's border is held by every tile that shares it, each with its own copy of
//! its position and correction, and the copies agree bit for bit. Each of those tiles is a
//! candidate, and its brush finds the vertex within the radius with the same delta. Adding the
//! delta once per copy would push the vertex once per tile that holds it, so exactly one copy
//! edits it: the least of its copies (see [`geom::Topology::owns`]). The edit then writes the
//! new value into every copy.

use brush::Imprint;
use geom::Vec3;
use store::{SpatialIndex, TileStore};
use thiserror::Error;

use crate::apply::{ApplyStats, PhaseTimes, SculptError, apply_imprints};
use crate::state::Stroke;

/// Why [`DrawParams::new`] refused its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DrawParamsError {
    /// The brush's centre has a component that isn't finite.
    #[error("the brush centre is not finite")]
    Center,
    /// The brush's radius isn't positive and finite.
    #[error("the brush radius is not positive and finite")]
    Radius,
    /// The brush's normal has no direction: zero, too small to normalise, or not finite.
    #[error("the brush normal has no direction")]
    Normal,
    /// The brush's strength isn't finite.
    #[error("the brush strength is not finite")]
    Strength,
}

/// One dab of the draw brush: where it is, which way and how far it pushes, and how far it
/// reaches.
///
/// Every value is checked by [`DrawParams::new`], so a step never sees a brush that reaches
/// nowhere or pushes by a NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawParams {
    /// The brush's centre, in model space.
    center: Vec3,
    /// The direction every vertex is pushed along: unit length.
    normal: Vec3,
    /// How far the brush reaches from its centre: positive and finite.
    radius: f32,
    /// How far a vertex at the centre is pushed along the normal: finite, and negative to push
    /// inwards.
    strength: f32,
}

impl DrawParams {
    /// A dab centred at `center`, reaching `radius` and pushing a vertex at its centre by
    /// `strength` along `normal` (inwards if negative), every vertex along the same direction,
    /// fading with the smoothstep falloff to nothing at the rim. `normal` is taken at unit length.
    ///
    /// # Errors
    ///
    /// [`DrawParamsError::Center`] if `center` isn't finite, [`DrawParamsError::Normal`] if
    /// `normal` has no direction, [`DrawParamsError::Radius`] if `radius` isn't positive and
    /// finite, [`DrawParamsError::Strength`] if `strength` isn't finite.
    pub fn new(
        center: Vec3,
        normal: Vec3,
        radius: f32,
        strength: f32,
    ) -> Result<Self, DrawParamsError> {
        let normal = check_dab(center, normal, radius, strength)?;
        Ok(Self {
            center,
            normal,
            radius,
            strength,
        })
    }

    /// The brush's centre, in model space.
    #[must_use]
    pub const fn center(&self) -> Vec3 {
        self.center
    }

    /// The direction every vertex is pushed along, unit length.
    #[must_use]
    pub const fn normal(&self) -> Vec3 {
        self.normal
    }

    /// How far the brush reaches from its centre.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        self.radius
    }

    /// How far a vertex at the centre is pushed along the normal.
    #[must_use]
    pub const fn strength(&self) -> f32 {
        self.strength
    }

    /// The draw action's imprint of this dab: its centre, normal, radius and strength, hardness
    /// `0` (the plain smoothstep falloff). The params are placeholders the draw action doesn't
    /// read.
    pub(crate) fn imprint(&self) -> Imprint {
        let Vec3 { x, y, z } = self.center;
        let n = self.normal;
        Imprint::new(
            [x, y, z],
            [n.x, n.y, n.z],
            self.radius,
            self.strength,
            0.0,
            [0.0; 4],
        )
    }
}

/// Checks a dab's values as [`DrawParams::new`] and [`apply_dabs`](crate::apply_dabs) take
/// them: a finite centre, a normal with a direction, a positive and finite radius and a finite
/// strength. Returns the normal at unit length.
///
/// # Errors
///
/// [`DrawParamsError::Center`], [`DrawParamsError::Normal`], [`DrawParamsError::Radius`] or
/// [`DrawParamsError::Strength`], for the first value that isn't so.
pub(crate) fn check_dab(
    center: Vec3,
    normal: Vec3,
    radius: f32,
    strength: f32,
) -> Result<Vec3, DrawParamsError> {
    if !center.is_finite() {
        return Err(DrawParamsError::Center);
    }
    let normal = normal.normalized().ok_or(DrawParamsError::Normal)?;
    if !(radius > 0.0 && radius.is_finite()) {
        return Err(DrawParamsError::Radius);
    }
    if !strength.is_finite() {
        return Err(DrawParamsError::Strength);
    }
    Ok(normal)
}

/// What one [`draw_step`] did, and how long each phase took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepStats {
    /// Tiles the brush's sphere touches by their boxes: queried, pinned and run through the
    /// brush.
    pub candidates: usize,
    /// Tiles whose corrections the step edited: those owning a vertex it moved (each also
    /// writes the copies its neighbours hold of its border vertices).
    pub tiles_edited: usize,
    /// Vertices the step moved, each counted once however many tiles hold a copy of it.
    pub vertices: usize,
    /// Tiles evaluated again and refitted in the index: those holding a moved vertex, and those
    /// whose normals read one of them.
    pub tiles_updated: usize,
    /// Tiles of [`StepStats::tiles_edited`] this step wrote per dab, the stored value plus the
    /// dab's delta, because the stroke had no room to keep their state (see "The thrash limit"
    /// on [`Stroke`]): their small deltas may be rounded away on their interiors.
    pub tiles_unaccumulated: usize,
    /// How long each phase took.
    pub times: PhaseTimes,
}

/// One dab of the draw brush: pushes the surface out along the dab's normal near `params`.
///
/// Every vertex of the index's level within the radius moves by `strength · smoothstep(d /
/// radius)` along `params`' normal, and `index` is brought up to date:
/// [`apply_dabs`](crate::apply_dabs) for this one dab, at hardness `0`, with the same bits.
///
/// `store` is the project's ([`Project::store`](store::Project::store)). The step edits as part
/// of the store's current op: call it between
/// [`Project::begin_stroke`](store::Project::begin_stroke)
/// and [`Project::end_stroke`](store::Project::end_stroke), so the stroke's steps are one op. The
/// vertices edited are those of the level `index` holds, the finest the store evaluates.
///
/// `stroke` keeps the value each edited vertex's correction is being built to across the stroke's
/// steps, the correction it first found plus every delta since, and the step writes it (see
/// [`Stroke`]); it starts afresh when the store's op changes, or
/// when it is used over another store. Keep one for the session and pass it to every step.
///
/// # Errors
///
/// As [`apply_dabs`](crate::apply_dabs): [`SculptError::Cache`] if the store can't pin the tiles
/// the brush touches (more than its budget holds, for one), evaluate them or edit them;
/// [`SculptError::Index`] if the index can't be updated. An error in the edit phase leaves the
/// tiles edited as [`TileStore::update_tiles`] says (the failing tile may have its whole
/// interior pass applied, and is then marked dirty), and the index is still updated for the
/// tiles the brush reached, unless that fails too (then rebuild it). The deltas the edit added
/// to `stroke`'s values but didn't write (the failing vertex's, and the interiors of later tiles of
/// its parallel run) are written by the next step that writes those vertices, as part of their
/// values; values no later step writes are lost. Tiles after the failing one get no deltas for
/// their border vertices, nor tiles after its run for any vertex. Don't retry a failed step: its
/// deltas already added would be counted twice.
pub fn draw_step(
    store: &mut TileStore,
    index: &mut SpatialIndex,
    stroke: &mut Stroke,
    params: DrawParams,
) -> Result<StepStats, SculptError> {
    Ok(apply_imprints(store, index, stroke, &[params.imprint()])?.into())
}

impl From<ApplyStats> for StepStats {
    /// The counts and times of an [`apply_dabs`](crate::apply_dabs) call as a step's: for one
    /// dab, what the step did; for more, the frame's (its vertices and unaccumulated tiles
    /// summed over the dabs).
    fn from(stats: ApplyStats) -> Self {
        Self {
            candidates: stats.candidates,
            tiles_edited: stats.tiles_edited,
            vertices: stats.vertices,
            tiles_updated: stats.tiles_updated,
            tiles_unaccumulated: stats.tiles_unaccumulated,
            times: stats.times,
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
