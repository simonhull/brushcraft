//! Anti-aliasing: accumulation over frames while the camera is still, a filter of our own while
//! it moves (spec, "Frame pipeline", 6).
//!
//! # Accumulation, in linear HDR
//!
//! While the camera is still, each frame draws the scene at a different point inside every pixel
//! (a [`Jitter`]) and adds its shaded sample to the pixel's running mean. The mean is taken of
//! the shading pass's **linear HDR** output, premultiplied (colour times coverage, coverage 1 on
//! the surface and 0 off it), **before the tone map**: the mean of radiance over the pixel is
//! what a box filter of the scene gives, and the tone map then shows it once. Averaging after the
//! tone map would darken the highlights the curve compresses (the mean of `tone(c)` is below
//! `tone(mean c)` past the knee) and make an edge's blend depend on the curve. The tone map
//! (`crate::tone`) composites the mean over the frame's clear colour by its coverage.
//!
//! # Jitter
//!
//! Sample `k` of a pixel's history is drawn at [`jitter`] of `k mod SAMPLES`: the Halton sequence
//! in bases 2 and 3, shifted by half a pixel (a Cranley–Patterson rotation, which keeps it low
//! discrepancy) so that `jitter(0)` is the pixel's centre, where a moving frame samples too: the
//! first still frame after a move draws exactly what the last moving frame drew. Any
//! [`SAMPLES`] frames in a row take each offset once, so a pixel whose history starts mid-cycle
//! (reset alone, or restarted by a look change) still takes every offset.
//!
//! The jitter moves the raster's projection only ([`crate::camera::View::jittered`], whose depth,
//! near plane and pixel lengths are bit for bit the unjittered view's). **The level choice and its
//! culling see the unjittered view**, so the cut, the morph and every vertex position are the same
//! at every jitter: only which triangle covers each pixel's sample moves. That is conservative for
//! culling: a jittered sample lies inside its pixel, so inside the unjittered frustum; the
//! occlusion test dilates each tile's rectangle by a pixel, more than the half pixel a sample
//! moves, and its pyramid is built from this frame's jittered depth.
//!
//! # The reset rule
//!
//! A sample is stale once the surface it saw has changed. With the camera still, what a tile
//! draws changes only when one of its inputs does, and the passes here track them per node of
//! the tile tree (`aa.wgsl`): its data (an epoch, an arrival: the bounds update's list); its key,
//! the inputs of `place.wgsl` and the shading that can change while still (its slot, `e`, its
//! parent's `e`, how far its morph by time has run, whether it is in the cut and its drawn
//! edges, its links); and the seam inputs (`e`, parent `e`, morph clock) of its edge neighbours,
//! which its seam vertices read. Each node gets the latest frame its drawing changed.
//!
//! A pixel keeps the frame its history started and the tiles it saw: up to four, as many as meet
//! at a valence-4 tile corner, which a pixel holding it sees within a cycle; past that (a corner
//! of higher valence, or one under a silhouette) it is marked crowded. **A pixel resets** when a
//! tile it saw changed after its start, when the tile this frame's sample sees changed after its
//! start, or, crowded, when any tile changed after its start.
//!
//! *Why it is enough.* Fix a jitter offset. With every tile's drawing unchanged, the nearest
//! surface along that offset's ray is the same tile at the same point, so the sample is the same.
//! If the sample there changed, then the tile seen before (`a`) or the tile seen now (`b`)
//! changed: had neither, `b` would have been in front of `a` before too. If `a` changed, `a` is
//! in the record (or the pixel is crowded) and the pixel resets the frame it changes. If only
//! `b` changed, the pixel resets the first frame it samples that offset again, within
//! [`SAMPLES`] frames (the stale mean is shown until then).
//!
//! *Why the jitter alone never resets a pixel.* With the scene still, no node's key or data
//! changes (the key holds nothing the jitter moves: the cut and the morph come from the
//! unjittered view, and the morph's clock stops at its duration), so no node's change frame
//! passes any pixel's start: the rule only ever compares node change frames, never ids, so the
//! different tiles a pixel's samples see at different offsets don't touch it.
//!
//! A look change restarts every pixel: the shading itself changed. Its first sample is the last
//! visibility buffer shaded again (one shading pass, the spec's light or material target), the
//! rest are drawn at the following jitters: one visibility buffer is kept, not one per jitter
//! (16 or 32 times the memory).
//!
//! # Convergence
//!
//! A pixel's mean takes [`SAMPLES`] samples, then stops changing. The app keeps frames coming
//! ([`SAMPLES`] after the camera stops or the look changes, twice that after a scene change:
//! one cycle for every pixel the change reaches to notice it, one to converge), then stops.
//!
//! # When it can't
//!
//! The histories are two storage buffers bound whole, so an image whose buffers pass the device's
//! `max_storage_buffer_binding_size` (often 128 MiB on mobile: 5K and up) isn't accumulated:
//! [`Accumulation::fits`] says so when the size is known, and still frames are filtered as moving
//! ones are ([`AaReport::filtered_instead`]). So is a frame whose accumulation the GPU refuses: the
//! frame is shown, never failed. Splitting the buffers into bands of rows would lift the limit;
//! nothing needs it yet.
//!
//! # Moving
//!
//! A frame with the camera moving isn't jittered or accumulated: the tone map filters it
//! (`tone.wgsl`), an edge filter of our own after FXAA's idea (find a luma edge and its
//! orientation, walk along it to its ends, blend across it by where the pixel lies on it, plus a
//! sub-pixel blend), and the next still frame starts the history again.
//!
//! # Memory
//!
//! Per pixel, [`PIXEL_BYTES`] (24): the mean as four half floats (8) and the record (16). At 4K
//! (3840 × 2160, 8.3 M pixels) that is 199 MB: 66 MB of means and 132.7 MB of records, the
//! larger just under 128 MiB (134.2 MB), the binding limit of the smallest devices we open (the
//! device takes the adapter's limits; past them see "When it can't"). At 1080p, 50 MB. The tone map
//! reads the mean directly, so there is no resolved image. Per node of the tile tree, 56 bytes (its
//! key, its change frames, and its change frame for the pixels).

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, CommandEncoder,
    ComputePassDescriptor, ComputePipeline, ShaderStages, Texture, TextureSampleType,
    TextureViewDescriptor, TextureViewDimension,
};

use crate::camera::Jitter;
use crate::cull::{BoundsUpdate, entry, pipeline, storage_buffer, uniform_buffer};
use crate::draw::{Look, SceneView};
use crate::gpu::{Gpu, GpuError};
use crate::raster::{MORPH_WGSL, Raster};
use crate::surface::Size;
use crate::tree::NODE_WGSL;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The samples a pixel's history takes: then it is converged, and frames stop.
///
/// 32 jittered samples put an edge's coverage within 1/32 of its area (Halton points, well spread);
/// at 60 Hz they take about half a second.
pub const SAMPLES: u32 = 32;

/// Bytes per pixel the accumulation keeps: the mean (four half floats) and the record (four
/// words).
pub const PIXEL_BYTES: u64 = MEAN_BYTES + RECORD_BYTES;

/// Bytes per pixel of the mean.
const MEAN_BYTES: u64 = 8;

/// Bytes per pixel of the record.
const RECORD_BYTES: u64 = 16;

/// Words per node of `aa.wgsl`'s `AaNode`: its key (11) and its two change frames.
const NODE_WORDS: u64 = 13;

/// The accumulation's frame number past which it starts over (pixel records hold 31 bits of it).
const FRAME_LIMIT: u32 = 1 << 30;

/// Invocations a compute group over nodes.
const NODE_GROUP: u32 = 64;

/// Pixels a side of a compute group over pixels.
const PIXEL_GROUP: u32 = 8;

/// The radical inverse of `index` in `base`: its digits mirrored about the point, in `[0, 1)`.
fn radical_inverse(base: u32, mut index: u32) -> f64 {
    let mut value = 0.0;
    let mut scale = 1.0 / f64::from(base);
    while index > 0 {
        value += f64::from(index % base) * scale;
        index /= base;
        scale /= f64::from(base);
    }
    value
}

/// The jitter of sample `index` (taken modulo [`SAMPLES`]): Halton in bases 2 and 3, shifted by
/// half a pixel so that sample 0 is the pixel's centre.
#[must_use]
pub fn jitter(index: u32) -> Jitter {
    let k = index % SAMPLES;
    let shifted = |base| (radical_inverse(base, k) + 0.5).fract() - 0.5;
    Jitter::new(shifted(2), shifted(3)).unwrap_or(Jitter::CENTRE)
}

/// What a frame does to show its scene smoothly: the renderer's choice ([`AaState`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum AaFrame {
    /// No anti-aliasing: the debug view, which shows ids.
    Plain,
    /// The camera moves: drawn unjittered, the tone map filters it.
    Filter,
    /// The camera is still: drawn at `jitter`, its sample added to every pixel's history
    /// (started again at every pixel if `restart`).
    Accumulate {
        /// Where the frame samples its pixels.
        jitter: Jitter,
        /// Whether every pixel's history starts again with this sample.
        restart: bool,
    },
    /// Nothing to draw or shade: the histories are shown as they are (a frame wanted for the
    /// overlay alone, the look unchanged).
    Show,
}

/// The renderer's side of accumulation: whether the histories are of the look and size being
/// shown, and the jitter the last raster used.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AaState {
    /// The look and size the histories hold; `None` when there are none to continue.
    held: Option<(Look, Size)>,
    /// The index of the jitter the last raster used.
    index: u32,
}

impl AaState {
    /// A frame drawing its scene with the camera `still` or not, shown as `look` at `size`; with
    /// `accumulable` false (the histories don't fit), filtered even when still.
    pub(crate) fn draw(
        &mut self,
        still: bool,
        accumulable: bool,
        look: Look,
        size: Size,
    ) -> AaFrame {
        if look.view != SceneView::Shaded || !still || !accumulable {
            self.held = None;
            self.index = 0;
            return if look.view == SceneView::Shaded {
                AaFrame::Filter
            } else {
                AaFrame::Plain
            };
        }
        let continuing = self.held == Some((look, size));
        self.index = if continuing {
            (self.index + 1) % SAMPLES
        } else {
            0
        };
        self.held = Some((look, size));
        AaFrame::Accumulate {
            jitter: jitter(self.index),
            restart: !continuing,
        }
    }

    /// A frame shading the last scene again, shown as `look` at `size`: nothing to do if the
    /// histories are already of it ([`AaFrame::Show`]); otherwise the shading again is the first
    /// sample of new histories, at the last raster's jitter. With `accumulable` false, the shading
    /// again is filtered.
    pub(crate) fn reshade(&mut self, accumulable: bool, look: Look, size: Size) -> AaFrame {
        if look.view != SceneView::Shaded {
            self.held = None;
            return AaFrame::Plain;
        }
        if !accumulable {
            self.held = None;
            return AaFrame::Filter;
        }
        if self.held == Some((look, size)) {
            return AaFrame::Show;
        }
        self.held = Some((look, size));
        AaFrame::Accumulate {
            jitter: jitter(self.index),
            restart: true,
        }
    }

    /// The frame showed no scene, or failed: there are no histories to continue.
    pub(crate) fn lost(&mut self) {
        self.held = None;
    }
}

/// What a frame's anti-aliasing did, for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AaReport {
    /// Every pixel's history started again with this frame's sample: [`SAMPLES`] − 1 frames more
    /// converge it.
    pub restarted: bool,
    /// The camera was still, but the image couldn't accumulate (its histories don't fit the
    /// device's bindings, or the GPU refused them): it was filtered instead.
    pub filtered_instead: bool,
}

/// Why accumulation couldn't be set up or encoded.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AaError {
    /// The GPU refused a buffer, pipeline or bind group.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// A history buffer for the image is larger than the device binds.
    #[error("a {bytes}-byte history buffer is over the device's {limit}-byte binding limit")]
    TooLarge {
        /// The buffer's size.
        bytes: u64,
        /// The device's limit.
        limit: u64,
    },
}

/// The accumulation's settings as the GPU reads them: WGSL's `AaParams`, 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuAaParams {
    /// The image's width and height.
    size: [u32; 2],
    /// This frame's number, from 1.
    frame: u32,
    /// [`RESTART`].
    flags: u32,
    /// [`SAMPLES`].
    samples: u32,
    /// How many marks are this frame's.
    marks: u32,
    /// Zero.
    reserved: [u32; 2],
}

/// [`GpuAaParams::flags`]: every pixel's history starts again.
const RESTART: u32 = 1;

/// The WGSL declarations both accumulation shaders share.
const PARAMS_WGSL: &str = r"
// The accumulation's settings (render::aa::GpuAaParams): 32 bytes.
struct AaParams {
    // The image's width and height.
    size: vec2<u32>,
    // This frame's number, from 1.
    frame: u32,
    // RESTART.
    flags: u32,
    // The samples a pixel's history takes, at most.
    samples: u32,
    // How many entries of `marks` are this frame's.
    marks: u32,
    // Zero.
    reserved: vec2<u32>,
}

// AaParams.flags: every pixel's history starts again this frame.
const RESTART: u32 = 1u;
";

/// The node passes' source.
const NODES_WGSL: &str = include_str!("aa.wgsl");

/// The pixel pass's source.
const PIXELS_WGSL: &str = include_str!("aa_pixels.wgsl");

/// The per-pixel histories, for one image size.
#[derive(Debug)]
struct History {
    /// The image's size.
    size: (u32, u32),
    /// Per pixel, the mean: four half floats.
    mean: Buffer,
    /// Per pixel, the record: four words.
    record: Buffer,
}

/// What a frame's accumulation reads.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AccumulateInputs<'a> {
    /// The tile tree's buffer; `None` for the last one given (a frame shading the last scene
    /// again).
    pub(crate) tree: Option<&'a Buffer>,
    /// The raster, its frame drawn at the jitter: its visibility buffer, stamps, drawn edges and
    /// settings.
    pub(crate) raster: &'a Raster,
    /// The shaded image of that frame.
    pub(crate) shaded: &'a Texture,
    /// The tiles whose data changed this frame; `None` for none.
    pub(crate) changes: Option<&'a BoundsUpdate>,
    /// Whether every pixel's history starts again.
    pub(crate) restart: bool,
}

/// The accumulation's passes and buffers.
#[derive(Debug)]
pub(crate) struct Accumulation {
    /// The node passes' layout.
    node_layout: BindGroupLayout,
    /// The pixel pass's layout.
    pixel_layout: BindGroupLayout,
    /// Marks the nodes whose data changed.
    mark: ComputePipeline,
    /// Compares each node's key with last frame's.
    compare: ComputePipeline,
    /// Gives each node the latest frame its drawing changed.
    spread: ComputePipeline,
    /// Resets and accumulates each pixel.
    accumulate: ComputePipeline,
    /// The tile tree's capacity.
    capacity: u32,
    /// Per node, `AaNode`.
    nodes: Buffer,
    /// Per node, the latest frame its drawing changed.
    changed: Buffer,
    /// The latest frame any node's drawing changed.
    latest: Buffer,
    /// The nodes whose data changed this frame.
    marks: Buffer,
    /// [`GpuAaParams`].
    params: Buffer,
    /// The histories, once a frame was accumulated.
    history: Option<History>,
    /// The tile tree's buffer last given.
    tree: Option<Buffer>,
    /// The last frame's number; 0 before the first.
    frame: u32,
    /// The largest buffer the device binds.
    limit: u64,
}

impl Accumulation {
    /// The passes for a tile tree of `capacity` nodes.
    ///
    /// # Errors
    ///
    /// [`AaError::Gpu`] if a buffer, layout or pipeline is refused.
    pub(crate) fn new(gpu: &Gpu, capacity: u32) -> Result<Self, AaError> {
        Self::build(gpu, capacity, &[])
    }

    /// [`Accumulation::new`] with text replacements in the pixel pass's shader, for the tests'
    /// bite checks.
    ///
    /// # Errors
    ///
    /// As [`Accumulation::new`].
    #[cfg(test)]
    pub(crate) fn patched(
        gpu: &Gpu,
        capacity: u32,
        patches: &[(&str, &str)],
    ) -> Result<Self, AaError> {
        Self::build(gpu, capacity, patches)
    }

    /// [`Accumulation::new`], `patches` made in the pixel pass's shader.
    ///
    /// # Errors
    ///
    /// As [`Accumulation::new`].
    ///
    /// # Panics
    ///
    /// If a patch doesn't apply (only the tests make any).
    #[expect(
        clippy::panic_in_result_fn,
        reason = "a patch that doesn't apply is a broken test, not an error to return"
    )]
    fn build(gpu: &Gpu, capacity: u32, patches: &[(&str, &str)]) -> Result<Self, AaError> {
        let device = gpu.device();
        let storage = |binding, read_only| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding| BindGroupLayoutEntry {
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            ..storage(binding, true)
        };
        let texture = |binding, sample_type| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let node_layout = gpu.scoped("aa node layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("aa nodes"),
                entries: &[
                    storage(0, true),
                    storage(1, true),
                    storage(2, true),
                    uniform(3),
                    storage(4, false),
                    storage(5, false),
                    storage(6, false),
                    storage(7, true),
                    uniform(8),
                ],
            })
        })?;
        let pixel_layout = gpu.scoped("aa pixel layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("aa pixels"),
                entries: &[
                    texture(0, TextureSampleType::Uint),
                    texture(1, TextureSampleType::Float { filterable: false }),
                    storage(2, true),
                    storage(3, true),
                    storage(4, false),
                    storage(5, false),
                    uniform(6),
                ],
            })
        })?;
        let nodes_source = format!("{NODE_WGSL}{MORPH_WGSL}{PARAMS_WGSL}{NODES_WGSL}");
        let node_pipeline =
            |name: &str| pipeline(gpu, "aa nodes", &nodes_source, &[&node_layout], name);
        let mark = node_pipeline("mark")?;
        let compare = node_pipeline("compare")?;
        let spread = node_pipeline("spread")?;
        let pixels_source = format!("{PARAMS_WGSL}{PIXELS_WGSL}");
        let pixels_source = patches.iter().fold(pixels_source, |source, (from, to)| {
            assert!(source.contains(from), "the patch {from:?} applies");
            source.replace(from, to)
        });
        let accumulate = pipeline(
            gpu,
            "aa pixels",
            &pixels_source,
            &[&pixel_layout],
            "accumulate",
        )?;
        let nodes_bytes = u64::from(capacity.max(1)) * NODE_WORDS * 4;
        Ok(Self {
            node_layout,
            pixel_layout,
            mark,
            compare,
            spread,
            accumulate,
            capacity,
            nodes: storage_buffer(gpu, "aa nodes", nodes_bytes)?,
            changed: storage_buffer(gpu, "aa changed", u64::from(capacity.max(1)) * 4)?,
            latest: storage_buffer(gpu, "aa latest", 4)?,
            marks: storage_buffer(gpu, "aa marks", 4 * u64::from(NODE_GROUP))?,
            params: uniform_buffer(gpu, "aa params", size_of::<GpuAaParams>() as u64)?,
            history: None,
            tree: None,
            frame: 0,
            limit: device.limits().max_storage_buffer_binding_size,
        })
    }

    /// Whether an image of `size` can accumulate: both its history buffers fit a binding.
    pub(crate) fn fits(&self, size: Size) -> bool {
        let pixels = u64::from(size.width()) * u64::from(size.height());
        pixels * MEAN_BYTES.max(RECORD_BYTES) <= self.limit
    }

    /// Binds at most `limit` bytes a buffer from now on, as a smaller device would.
    #[cfg(test)]
    pub(crate) const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// The per-pixel means, four half floats a pixel (red and green, blue and coverage, each
    /// pair a word), premultiplied linear colour: the image to show while still. `None` before
    /// the first frame accumulated.
    pub(crate) fn mean(&self) -> Option<&Buffer> {
        self.history.as_ref().map(|history| &history.mean)
    }

    /// The per-pixel records, four words a pixel (`aa_pixels.wgsl`), for the tests.
    #[cfg(test)]
    pub(crate) fn record(&self) -> Option<&Buffer> {
        self.history.as_ref().map(|history| &history.record)
    }

    /// The last frame's number, from 1: a pixel whose record starts at `s` holds
    /// `min(frame − s + 1, SAMPLES)` samples.
    #[cfg(test)]
    pub(crate) const fn frame(&self) -> u32 {
        self.frame
    }

    /// Encodes a frame's accumulation into `encoder`, after its shading: the node passes, then
    /// the pixel pass over `inputs.shaded`. Settings go through the queue: one frame a
    /// submission.
    ///
    /// # Errors
    ///
    /// [`AaError::TooLarge`] if the image's histories don't fit a binding; [`AaError::Gpu`] if
    /// a buffer or bind group is refused. With no tree given yet, nothing is encoded.
    pub(crate) fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        inputs: &AccumulateInputs<'_>,
    ) -> Result<(), AaError> {
        if let Some(tree) = inputs.tree {
            self.tree = Some(tree.clone());
        }
        let Some(tree) = self.tree.clone() else {
            return Ok(());
        };
        let visibility = inputs.raster.visibility();
        let size = (visibility.width(), visibility.height());
        let resized = self
            .history
            .as_ref()
            .is_none_or(|history| history.size != size);
        if resized {
            self.history = None;
            self.history = Some(new_history(gpu, size, self.limit)?);
        }
        let renumbered = self.next_frame(encoder);
        let restart = inputs.restart || resized || renumbered;
        let marks = self.write_marks(gpu, inputs.changes)?;
        let params = GpuAaParams {
            size: size.into(),
            frame: self.frame,
            flags: if restart { RESTART } else { 0 },
            samples: SAMPLES,
            marks,
            reserved: [0; 2],
        };
        gpu.queue()
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let Some(history) = &self.history else {
            return Ok(());
        };
        let (node_group, pixel_group) = self.groups(gpu, &tree, inputs, history)?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("aa"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &node_group, &[]);
        if marks > 0 {
            pass.set_pipeline(&self.mark);
            pass.dispatch_workgroups(marks.div_ceil(NODE_GROUP), 1, 1);
        }
        let node_groups = self.capacity.max(1).div_ceil(NODE_GROUP);
        pass.set_pipeline(&self.compare);
        pass.dispatch_workgroups(node_groups, 1, 1);
        pass.set_pipeline(&self.spread);
        pass.dispatch_workgroups(node_groups, 1, 1);
        pass.set_pipeline(&self.accumulate);
        pass.set_bind_group(0, &pixel_group, &[]);
        pass.dispatch_workgroups(
            size.0.div_ceil(PIXEL_GROUP),
            size.1.div_ceil(PIXEL_GROUP),
            1,
        );
        Ok(())
    }

    /// Moves on to the next frame number; past [`FRAME_LIMIT`], starts the numbering over
    /// (every node and pixel forgets its frames: the clears go into `encoder`) and returns true.
    fn next_frame(&mut self, encoder: &mut CommandEncoder) -> bool {
        let over = self.frame >= FRAME_LIMIT;
        if over {
            encoder.clear_buffer(&self.nodes, 0, None);
            encoder.clear_buffer(&self.changed, 0, None);
            encoder.clear_buffer(&self.latest, 0, None);
            self.frame = 0;
        }
        self.frame += 1;
        over
    }

    /// The node passes' and the pixel pass's bind groups for a frame of `inputs` over `tree`,
    /// into `history`.
    ///
    /// # Errors
    ///
    /// [`AaError::Gpu`] if one is refused.
    fn groups(
        &self,
        gpu: &Gpu,
        tree: &Buffer,
        inputs: &AccumulateInputs<'_>,
        history: &History,
    ) -> Result<(BindGroup, BindGroup), AaError> {
        let visibility = inputs.raster.visibility();
        let placing = inputs.raster.placing();
        let device = gpu.device();
        let node_group = gpu.scoped("aa node bind group", || {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("aa nodes"),
                layout: &self.node_layout,
                entries: &[
                    entry(0, tree),
                    entry(1, placing.stamps),
                    entry(2, placing.drawn_edges),
                    entry(3, placing.params),
                    entry(4, &self.nodes),
                    entry(5, &self.changed),
                    entry(6, &self.latest),
                    entry(7, &self.marks),
                    entry(8, &self.params),
                ],
            })
        })?;
        let pixel_group = gpu.scoped("aa pixel bind group", || {
            let ids = visibility.create_view(&TextureViewDescriptor::default());
            let shaded = inputs.shaded.create_view(&TextureViewDescriptor::default());
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("aa pixels"),
                layout: &self.pixel_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&ids),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&shaded),
                    },
                    entry(2, &self.changed),
                    entry(3, &self.latest),
                    entry(4, &history.mean),
                    entry(5, &history.record),
                    entry(6, &self.params),
                ],
            })
        })?;
        Ok((node_group, pixel_group))
    }

    /// Writes the nodes of `changes` into the marks buffer (grown to hold them) and returns how
    /// many there are.
    ///
    /// # Errors
    ///
    /// [`AaError::Gpu`] if a larger buffer is refused.
    fn write_marks(&mut self, gpu: &Gpu, changes: Option<&BoundsUpdate>) -> Result<u32, AaError> {
        let nodes: Vec<u32> = changes
            .map(|update| update.nodes().collect())
            .unwrap_or_default();
        let bytes = (nodes.len() as u64 * 4).max(4);
        if bytes > self.marks.size() {
            self.marks = storage_buffer(gpu, "aa marks", bytes.next_power_of_two())?;
        }
        if !nodes.is_empty() {
            gpu.queue()
                .write_buffer(&self.marks, 0, bytemuck::cast_slice(&nodes));
        }
        Ok(u32::try_from(nodes.len()).unwrap_or(u32::MAX))
    }
}

/// Histories for an image of `size`, zeroed (the first frame restarts every pixel).
///
/// # Errors
///
/// [`AaError::TooLarge`] past `limit`, the device's binding limit; [`AaError::Gpu`] if refused.
fn new_history(gpu: &Gpu, size: (u32, u32), limit: u64) -> Result<History, AaError> {
    let pixels = u64::from(size.0) * u64::from(size.1);
    for bytes in [pixels * MEAN_BYTES, pixels * RECORD_BYTES] {
        if bytes > limit {
            return Err(AaError::TooLarge { bytes, limit });
        }
    }
    Ok(History {
        size,
        mean: storage_buffer(gpu, "aa means", (pixels * MEAN_BYTES).max(8))?,
        record: storage_buffer(gpu, "aa records", (pixels * RECORD_BYTES).max(RECORD_BYTES))?,
    })
}
