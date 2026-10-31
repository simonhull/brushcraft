//! Brushes: dabs that edit a project's corrections through its store and spatial index, and the
//! store as the stroke engine's surface.
//!
//! A brush stroke is a sequence of dabs, each one an edit of the corrections near the brush. A
//! [`Stroke`] keeps, across the dabs, each edited vertex's correction as the stroke found it and
//! the sum of the stroke's deltas for it, and each dab writes their sum: a tile's quantized
//! interior is rounded once per stroke, not once per dab, so small dabs move it as far as its
//! exactly stored border.
//!
//! The per-vertex math lives in `brush`, shared with the GPU; this crate does the rest:
//! - [`apply_dabs`] applies a frame's dabs (the stroke engine's [`stroke::Dab`]s): it finds the
//!   tiles they reach through the [`store::SpatialIndex`], pins and evaluates them, runs each dab's
//!   brush over its tiles on the store's workers, writes the deltas into the corrections through
//!   [`store::TileStore::edit_tile`], and brings the index back up to date once for the frame. The
//!   result is that of applying the dabs one at a time, bit for bit.
//! - [`draw_step`] is one dab of the draw brush, as M0 measured it: [`apply_dabs`] for one dab.
//! - [`StoreSurface`] is the store as the stroke engine's [`stroke::Surface`]: raycasts through the
//!   index, and the stroke's connected region near a point.
//!
//! Dabs are part of the stroke their caller has begun: [`store::Project::begin_stroke`] before
//! the first, [`store::Project::end_stroke`] after the last, so every dab of the stroke is one
//! op. Every call takes the session's one [`Stroke`], which starts afresh when the op changes.
//!
//! No GPU or windowing code: everything here runs headless.

mod apply;
mod draw;
mod state;
mod surface;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod testing;

pub use apply::{ApplyStats, PhaseTimes, SculptError, apply_dabs};
pub use draw::{DrawParams, DrawParamsError, StepStats, draw_step};
pub use state::Stroke;
pub use surface::StoreSurface;
