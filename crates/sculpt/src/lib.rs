//! Brushes: dabs that edit a project's corrections through its store and spatial index, and the
//! store as the stroke engine's surface.
//!
//! A brush stroke is a sequence of dabs, each one an edit of the corrections near the brush. A
//! [`Stroke`] keeps, across the dabs, each edited vertex's correction as the stroke found it and
//! the sum of the stroke's deltas for it, and each dab writes their sum: a tile's quantized
//! interior is rounded once per stroke, not once per dab, so small dabs move it as far as its
//! exactly stored border.
//!
//! A stroke reads the surface as it was when it began, its **pen-down surface** (the brush-engine
//! spec): placement's every read, and the distances each dab's falloff is measured over. The store
//! keeps the pen-down shape of each tile a stroke changes ([`store::TileStore::keep_pen_down`]), so
//! a stroke never reads its own dabs: a ridge is the sum of its dabs over the surface the stroke
//! began on, and what a stroke places doesn't depend on how far its dabs have landed.
//!
//! The per-vertex math lives in `brush`, shared with the GPU; this crate does the rest:
//! - [`apply_dabs`] applies a frame's dabs (the stroke engine's [`stroke::Dab`]s): it finds the
//!   tiles they reach through the [`store::SpatialIndex`], pins and evaluates them, runs each dab's
//!   brush over its tiles' pen-down positions on the store's workers, writes the deltas into the
//!   corrections through [`store::TileStore::edit_tile`], and brings the index back up to date once
//!   for the frame. The result is that of applying the dabs one at a time, bit for bit.
//! - [`draw_step`] is one dab of the draw brush, as M0 measured it: [`apply_dabs`] for one dab.
//! - [`Preview`] shows dabs at once on the finest tiles they reach, ahead of the finest work, for a
//!   brush whose frame can't fit it all; the same dabs land later through [`apply_dabs`].
//! - [`radius_covering`] sizes a brush by the vertices it covers about a point: the radius whose
//!   ball holds that many, as the M1 spec's "brush of 1M vertices" is named.
//! - [`FuturePlan`] gives a tile as it will be once the open strokes still landing have landed
//!   their remaining dabs, made by the landing's own per-tile arithmetic without writing: the
//!   surface a stroke started while earlier ones land reads.
//! - [`StoreSurface`] is the store's pen-down surface as the stroke engine's [`stroke::Surface`]:
//!   raycasts through the index, the stroke's connected region near a point, and area normals.
//!
//! Dabs are part of the stroke their caller has begun: [`store::Project::begin_stroke_and_wait`]
//! before the first (or [`store::Project::begin_stroke`], and the first only once
//! [`store::Project::may_land`]), [`store::Project::end_stroke`] after the last, so every dab of
//! the stroke is one op, and the store keeps the stroke's pen-down shapes from the first until the
//! last. Every call takes the session's one [`Stroke`], which starts afresh when the op changes.
//!
//! No GPU or windowing code: everything here runs headless.

mod apply;
mod draw;
mod footprint;
mod future;
mod preview;
mod state;
mod surface;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod testing;

pub use apply::{ApplyStats, PhaseTimes, SculptError, apply_dabs};
pub use draw::{DrawParams, DrawParamsError, StepStats, draw_step};
pub use footprint::radius_covering;
pub use future::{FutureError, FuturePlan, Remaining};
pub use preview::{Drawn, DrawsAll, Preview, PreviewStats, WAITING_A_CALL};
pub use state::Stroke;
pub use surface::StoreSurface;
