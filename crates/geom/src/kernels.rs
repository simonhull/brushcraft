//! The stages of predicting a block, one at a time, for the per-kernel benchmarks
//! (`benches/kernels.rs`).
//!
//! Compiled only with the `bench-kernels` feature, which only the benchmarks turn on: it is not
//! part of the crate's API, and nothing here is needed to evaluate.

use crate::eval::{Parent, gather, parent_region, predicted_frames};
use crate::subdiv::{Padded, predict};
use crate::{BaseMesh, EvalError, Frame, PositionSource, Region};

/// One block's gathered parent and predicted positions, kept so each later stage can be run (and
/// timed) alone on the same inputs.
#[derive(Debug)]
pub struct Stages<'a> {
    /// The mesh the block lies on.
    mesh: &'a BaseMesh,
    /// The block predicted.
    region: Region,
    /// The parent level's block, ring and base-vertex rings, as [`crate::predict_region`]
    /// gathers them.
    parent: Parent,
    /// The block predicted from `parent`, with its one-vertex ring.
    predicted: Padded,
}

impl<'a> Stages<'a> {
    /// Gathers and predicts `region` (level ≥ 1) from `src` once.
    ///
    /// # Errors
    ///
    /// As [`crate::predict_region`].
    pub fn new(
        mesh: &'a BaseMesh,
        src: &impl PositionSource,
        region: Region,
    ) -> Result<Self, EvalError> {
        let parent = gather(mesh, src, parent_region(region)?)?;
        let predicted = predict(&parent.padded, &parent.corners);
        Ok(Self {
            mesh,
            region,
            parent,
            predicted,
        })
    }

    /// The gather stage again: reads the parent block, its ring and the corners' rings from
    /// `src`. Returns the block's ring-included vertex count, so the work can't be optimised out.
    ///
    /// # Errors
    ///
    /// As [`crate::predict_region`].
    pub fn gather(&self, src: &impl PositionSource) -> Result<usize, EvalError> {
        let parent = gather(self.mesh, src, parent_region(self.region)?)?;
        Ok(parent.padded.len())
    }

    /// The subdivision stage again: predicts the block from the gathered parent. Returns the
    /// predicted block's ring-included vertex count.
    #[must_use]
    pub fn subdivide(&self) -> usize {
        predict(&self.parent.padded, &self.parent.corners).len()
    }

    /// The frames stage again: every vertex's tangent frame from the predicted block.
    #[must_use]
    pub fn frames(&self) -> Vec<Frame> {
        predicted_frames(self.mesh, self.region, &self.parent, &self.predicted)
    }
}
