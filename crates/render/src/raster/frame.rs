//! A frame's level choice and raster in their order, with the two depth pyramids the occlusion
//! test reads (the order `lod.rs` documents, "A frame", and its pyramids "decided for Task 4").
//!
//! [`FramePasses::encode`] encodes, after the frame's uploads: the bounds of the tiles whose data
//! changed; the level choice's first phase, its occlusion test against the previous frame's
//! pyramid seen from the previous frame's view; the stamps; the first draw; this frame's pyramid
//! from that depth; the second phase against it; the second draw; the pyramid again, from the
//! whole frame's depth; and the readback. The pyramids then swap, so the next frame's first phase
//! tests against everything this frame drew. A frame of another size than the last has no
//! previous pyramid: nothing is occlusion-culled in its first phase.
//!
//! This is the composition [`crate::Renderer::draw`] runs once the app hands it the tile tree
//! and residency (not wired yet: see the plan's Task 4 notes); the tests draw through it.

use std::time::Duration;

use geom::Level;
use thiserror::Error;
use wgpu::{Buffer, CommandEncoder};

use super::{ClusterDraw, Phase, Raster, RasterConfig, RasterError, RasterFrame};
use crate::camera::View;
use crate::cull::{BoundsUpdate, DepthPyramid, PyramidPass, SlotBuffers};
use crate::gpu::{Gpu, GpuError};
use crate::lod::{CutReadback, Lod, LodConfig, LodError, MainInputs};

/// Why a frame's passes couldn't be set up or encoded.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FrameError {
    /// The level choice failed.
    #[error(transparent)]
    Lod(#[from] LodError),
    /// The raster failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
    /// The GPU failed: a pyramid, a bind group.
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// What one frame's passes read.
#[derive(Debug, Clone, Copy)]
pub struct SceneInputs<'a> {
    /// The tile tree's buffer ([`crate::tree::TileTree::buffer`]).
    pub tree: &'a Buffer,
    /// Residency's request buffer ([`crate::residency::Residency::request_buffer`]).
    pub requests: &'a Buffer,
    /// The slot buffers the tree's slots name.
    pub slots: &'a SlotBuffers,
    /// The tiles whose data changed since the bounds were last computed.
    pub bounds: &'a BoundsUpdate,
    /// The frame's view.
    pub view: &'a View,
    /// The frame's time, for the morph ([`RasterFrame::time`]).
    pub time: Duration,
    /// Whether the camera moves this frame.
    pub moving: bool,
    /// The project's finest level.
    pub finest: Level,
}

/// The level choice, the raster and the depth pyramids of a frame.
#[derive(Debug)]
pub struct FramePasses {
    /// The level choice.
    lod: Lod,
    /// The raster.
    raster: Raster,
    /// The pyramid's passes.
    pyramids: PyramidPass,
    /// The pyramid this frame builds, once made for the frame's size.
    current: Option<DepthPyramid>,
    /// The previous frame's view and its whole depth's pyramid.
    previous: Option<(View, DepthPyramid)>,
}

impl FramePasses {
    /// The passes for a tile tree of `capacity` nodes.
    ///
    /// # Errors
    ///
    /// As [`Lod::new`], [`Raster::new`] and [`PyramidPass::new`].
    pub fn new(
        gpu: &Gpu,
        capacity: u32,
        lod: LodConfig,
        raster: RasterConfig,
    ) -> Result<Self, FrameError> {
        Ok(Self {
            lod: Lod::new(gpu, capacity, lod)?,
            raster: Raster::new(gpu, capacity, raster)?,
            pyramids: PyramidPass::new(gpu)?,
            current: None,
            previous: None,
        })
    }

    /// The level choice: its cluster lists, and [`Lod::collect`] for the readbacks.
    #[must_use]
    pub const fn lod(&self) -> &Lod {
        &self.lod
    }

    /// The level choice, to collect its readbacks.
    pub const fn lod_mut(&mut self) -> &mut Lod {
        &mut self.lod
    }

    /// The raster: its visibility and depth buffers.
    #[must_use]
    pub const fn raster(&self) -> &Raster {
        &self.raster
    }

    /// Encodes the frame's passes into `encoder` (after its uploads) and returns its readback,
    /// `None` if every readback buffer is in flight. Settings go through the queue, so encode
    /// one frame per submission, and call [`CutReadback::submitted`] once it is submitted.
    ///
    /// # Errors
    ///
    /// As the passes' own encoding: [`FrameError::Lod`], [`FrameError::Raster`],
    /// [`FrameError::Gpu`].
    pub fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        inputs: &SceneInputs<'_>,
    ) -> Result<Option<CutReadback>, FrameError> {
        let size = inputs.view.size();
        let wanted = (size.width(), size.height());
        if self
            .current
            .as_ref()
            .is_none_or(|p| p.depth_size() != wanted)
        {
            self.current = Some(DepthPyramid::new(gpu, size)?);
        }
        if self
            .previous
            .as_ref()
            .is_some_and(|(_, p)| p.depth_size() != wanted)
        {
            self.previous = None;
        }
        self.lod
            .encode_bounds(gpu, encoder, inputs.tree, inputs.bounds, inputs.slots)?;
        let previous = self
            .previous
            .as_ref()
            .map(|(view, pyramid)| (view, pyramid));
        let main = MainInputs {
            tree: inputs.tree,
            requests: inputs.requests,
            view: inputs.view,
            previous,
            finest: inputs.finest,
        };
        self.lod.encode_main(gpu, encoder, &main)?;
        let frame = RasterFrame {
            view: inputs.view,
            time: inputs.time,
            moving: inputs.moving,
        };
        self.raster.begin_frame(gpu, &frame)?;
        self.raster
            .encode_stamp(gpu, encoder, inputs.tree, self.lod.cut_list())?;
        let current = self.current.take().ok_or(RasterError::NoFrame)?;
        let drawn = self.draw_both(gpu, encoder, inputs, &current);
        // The swap: this frame's pyramid is the next one's previous, and the old previous is
        // built over next.
        let older = self.previous.replace((*inputs.view, current));
        self.current = older.map(|(_, pyramid)| pyramid);
        drawn?;
        Ok(self.lod.encode_readback(gpu, encoder)?)
    }

    /// The two draws around the second phase, building `current` after each.
    ///
    /// # Errors
    ///
    /// As [`FramePasses::encode`].
    fn draw_both(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        inputs: &SceneInputs<'_>,
        current: &DepthPyramid,
    ) -> Result<(), FrameError> {
        for (post, phase) in [(false, Phase::First), (true, Phase::Second)] {
            if post {
                self.lod.encode_post(gpu, encoder, current)?;
            }
            let draw = ClusterDraw {
                tree: inputs.tree,
                slots: inputs.slots,
                clusters: self.lod.clusters(post),
                args: self.lod.draw_args(post),
            };
            self.raster.encode_draw(gpu, encoder, &draw, phase)?;
            let depth = self.raster.depth_view().ok_or(RasterError::NoFrame)?;
            self.pyramids.build(gpu, encoder, depth, current, None)?;
        }
        Ok(())
    }
}
