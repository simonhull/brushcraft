//! Brushes: steps that edit a project's corrections through its store and spatial index.
//!
//! A brush stroke is a sequence of steps (dabs), each one a read-modify-write of the corrections
//! near the brush. The per-vertex math lives in `kernels`, shared with the GPU; this crate does
//! the rest of a step: finding the tiles the brush reaches through the [`store::SpatialIndex`],
//! pinning and evaluating them, writing the kernel's deltas into the corrections through
//! [`store::TileStore::edit_tile`], and bringing the index back up to date.
//!
//! M0 has one brush, the [`draw_step`], to measure edit latency with. It runs on one thread: the
//! per-tile work runs in parallel later, measured against this baseline.
//!
//! A step is part of the stroke its caller has begun: [`store::Project::begin_stroke`] before
//! the first step, [`store::Project::end_stroke`] after the last, so every step of the stroke is
//! one op.
//!
//! No GPU or windowing code: everything here runs headless.

mod draw;

pub use draw::{DrawParams, DrawParamsError, PhaseTimes, SculptError, StepStats, draw_step};
