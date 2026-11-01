//! The draw step: one dab of the draw brush, pushing the surface out along its normal.
//!
//! [`draw_step`] is [`apply_dabs`](crate::apply_dabs) for one dab, with the plain smoothstep
//! falloff: the step the M0 benchmark measured, kept so its history stays comparable. See the
//! `apply` module notes for its phases (`query`, `pin`, `brush`, `edit`, `neighbours`,
//! `evaluate`, `refit`, as [`PhaseTimes`] times them).
//!
//! Corrections live in each vertex's tangent frame `(t, b, n)` of the predicted surface, so a
//! push along the frame normal `n` is a change of the correction's `z` alone, with no frames to
//! fetch. (Brushes that move along the evaluated normal need the frames; they come later.)
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
        check_dab(center, radius, strength)?;
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

    /// The draw action's imprint of this dab: its centre, radius and strength, hardness `0`
    /// (the plain smoothstep falloff). The normal and params are placeholders the draw action
    /// doesn't read.
    pub(crate) fn imprint(&self) -> Imprint {
        let Vec3 { x, y, z } = self.center;
        Imprint::new(
            [x, y, z],
            [0.0, 0.0, 1.0],
            self.radius,
            self.strength,
            0.0,
            [0.0; 4],
        )
    }
}

/// Checks a dab's values as [`DrawParams::new`] and [`apply_dabs`](crate::apply_dabs) take
/// them: a finite centre, a positive and finite radius and a finite strength.
///
/// # Errors
///
/// [`DrawParamsError::Center`], [`DrawParamsError::Radius`] or [`DrawParamsError::Strength`],
/// for the first value that isn't so.
pub(crate) fn check_dab(center: Vec3, radius: f32, strength: f32) -> Result<(), DrawParamsError> {
    if !center.is_finite() {
        return Err(DrawParamsError::Center);
    }
    if !(radius > 0.0 && radius.is_finite()) {
        return Err(DrawParamsError::Radius);
    }
    if !strength.is_finite() {
        return Err(DrawParamsError::Strength);
    }
    Ok(())
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

/// One dab of the draw brush: pushes the surface out along its frame normal near `params`.
///
/// Every vertex of the index's level within the radius moves by `strength · smoothstep(d /
/// radius)`, and `index` is brought up to date: [`apply_dabs`](crate::apply_dabs) for this one
/// dab, at hardness `0`, with the same bits.
///
/// `store` is the project's ([`Project::store`](store::Project::store)). The step edits as part
/// of the store's current op: call it between
/// [`Project::begin_stroke`](store::Project::begin_stroke)
/// and [`Project::end_stroke`](store::Project::end_stroke), so the stroke's steps are one op. The
/// vertices edited are those of the level `index` holds, the finest the store evaluates.
///
/// `stroke` keeps each edited vertex's start and summed deltas across the stroke's steps, and
/// the step writes their sum (see [`Stroke`]); it starts afresh when the store's op changes, or
/// when it is used over another store. Keep one for the session and pass it to every step.
///
/// # Errors
///
/// As [`apply_dabs`](crate::apply_dabs): [`SculptError::Cache`] if the store can't pin the tiles
/// the brush touches (more than its budget holds, for one), evaluate them or edit them;
/// [`SculptError::Index`] if the index can't be updated. An error in the edit phase leaves the
/// tiles edited as [`TileStore::update_tiles`] says (the failing tile may have its whole
/// interior pass applied, and is then marked dirty), and the index is still updated for the
/// tiles the brush reached, unless that fails too (then rebuild it). The deltas the edit summed
/// into `stroke` but didn't write (the failing vertex's, and the interiors of later tiles of its
/// parallel run) are written by the next step that writes those vertices, as part of their
/// sums; sums no later step writes are lost. Tiles after the failing one get no sums for their
/// border vertices, nor tiles after its run for any vertex. Don't retry a failed step: its
/// deltas already summed would be counted twice.
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
