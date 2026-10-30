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
//! [`FramePasses::encode_shading`] then shades the visibility buffer ([`crate::shade`]) into the
//! linear HDR image the tone map shows, and [`FramePasses::encode_shading_again`] shades it again
//! with no raster, for a change of material or wireframe alone.
//!
//! [`crate::Renderer::draw`] runs them each frame the app hands it a scene, after the frame's
//! uploads and before the image is shown; the tests draw through it too.

use std::time::Duration;

use geom::Level;
use thiserror::Error;
use wgpu::{Buffer, CommandEncoder};

use super::{ClusterDraw, Phase, Raster, RasterConfig, RasterError, RasterFrame};
use crate::Look;
use crate::camera::View;
use crate::cull::{BoundsUpdate, DepthPyramid, PyramidPass, SlotBuffers};
use crate::gpu::{Gpu, GpuError};
use crate::lod::{CutReadback, Lod, LodConfig, LodError, MainInputs};
use crate::material::MatCapStyle;
use crate::residency::SlotLayout;
use crate::shade::{Shade, ShadeError, ShadePath, ShadeScene};

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
    /// The shading pass failed.
    #[error(transparent)]
    Shade(#[from] ShadeError),
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
    /// The number the frame's readback carries ([`crate::lod::LodOutput::frame`]): the app's
    /// residency frame, so the cut comes back beside that frame's requests.
    pub frame: u64,
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
    /// The shading pass.
    shade: Shade,
}

impl FramePasses {
    /// The passes for a tile tree of `capacity` nodes and slot buffers of `slots` (residency's,
    /// [`crate::residency::Residency::slot_layout`]), the level choice set up by `lod` and the
    /// geomorph holding the drawn error to its refine threshold
    /// ([`RasterConfig::for_level_choice`]), level changes made with the camera still morphing
    /// over `morph`, and the shading pass running as `path` ([`ShadePath::for_tier`]).
    ///
    /// # Errors
    ///
    /// As [`Lod::new`], [`Raster::new`], [`PyramidPass::new`] and `Shade::new`.
    pub fn new(
        gpu: &Gpu,
        capacity: u32,
        slots: SlotLayout,
        lod: LodConfig,
        morph: Duration,
        path: ShadePath,
    ) -> Result<Self, FrameError> {
        let raster = RasterConfig::for_level_choice(&lod).with_morph(morph);
        Ok(Self {
            lod: Lod::new(gpu, capacity, lod)?,
            raster: Raster::new(gpu, capacity, slots, raster)?,
            pyramids: PyramidPass::new(gpu)?,
            current: None,
            previous: None,
            shade: Shade::new(gpu, slots, path, MatCapStyle::default())?,
        })
    }

    /// The level choice: its cluster lists, and [`Lod::collect`] (or [`Lod::take_reader`]) for
    /// the readbacks.
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

    /// The shading pass: its target, its settings.
    #[must_use]
    pub(crate) const fn shade(&self) -> &Shade {
        &self.shade
    }

    /// Sets how the shading shows the scene, `look`'s wireframe and MatCap style, the wireframe's
    /// lines `pixel_scale` frame pixels a point (the window's scale factor), from the next shading
    /// pass on.
    pub fn set_shading(&mut self, gpu: &Gpu, look: Look, pixel_scale: u32) {
        self.shade.set_settings(look.shading);
        self.shade.set_material(gpu, look.material);
        self.shade.set_pixel_scale(pixel_scale);
    }

    /// Encodes the shading pass over the visibility buffer [`FramePasses::encode`] just drew
    /// from `inputs`, in the same submission.
    ///
    /// # Errors
    ///
    /// [`FrameError::Shade`] as `Shade::encode`.
    pub fn encode_shading(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        inputs: &SceneInputs<'_>,
    ) -> Result<(), FrameError> {
        let scene = ShadeScene {
            tree: inputs.tree,
            slots: inputs.slots,
            raster: &self.raster,
            view: inputs.view,
        };
        Ok(self.shade.encode(gpu, encoder, &scene)?)
    }

    /// Encodes the shading pass again over the last visibility buffer shaded, with no raster:
    /// `Shade::encode_again`. Returns whether there was one.
    ///
    /// # Errors
    ///
    /// [`FrameError::Shade`] as `Shade::encode_again`.
    pub fn encode_shading_again(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
    ) -> Result<bool, FrameError> {
        Ok(self.shade.encode_again(gpu, encoder)?)
    }

    /// How long a level change made with the camera still morphs: frames should keep coming
    /// for that long after the cut changes, so the morph is seen through to its end.
    #[must_use]
    pub const fn morph(&self) -> Duration {
        self.raster.config().morph()
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
        let current = match self.current.take() {
            Some(pyramid) if pyramid.depth_size() == wanted => pyramid,
            _ => DepthPyramid::new(gpu, size)?,
        };
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
        let drawn = self.draw_both(gpu, encoder, inputs, &current);
        // The swap: this frame's pyramid is the next one's previous, and the old previous is
        // built over next.
        let older = self.previous.replace((*inputs.view, current));
        self.current = older.map(|(_, pyramid)| pyramid);
        drawn?;
        Ok(self.lod.encode_readback(gpu, encoder, inputs.frame)?)
    }

    /// The two draws around the second phase, building `current` after each.
    ///
    /// # Errors
    ///
    /// As [`FramePasses::encode`].
    fn draw_both(
        &self,
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
            let depth = self.raster.depth_view();
            self.pyramids.build(gpu, encoder, depth, current, None)?;
        }
        Ok(())
    }
}
