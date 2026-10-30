//! The raster: the level choice's visible clusters drawn into a visibility buffer, watertight at
//! every seam, with the geomorph that keeps level changes from popping.
//!
//! # What it draws
//!
//! Each phase of the level choice ([`crate::lod`]) lists its visible clusters
//! ([`crate::lod::GpuCluster`]: a tile's node, the cluster's place in the tile, its edge variant
//! and its finer edges) with indexed indirect draw arguments. The raster draws a list in **one
//! instanced indexed indirect draw** ([`Raster::encode_draw`]), one instance a cluster, over one
//! static index buffer of the cluster's 128 triangles ([`crate::indices`]). The vertex shader
//! (`raster.wgsl`) remaps the index it is given by the cluster's edge variant (the 16 variants are
//! tables of one static buffer, [`crate::indices::remap_table`]), reads the vertex from its
//! tile's slot and geomorphs it; the fragment shader writes the visibility id.
//!
//! **Which variant** is the level choice's: `emit` sets bit `k` of a cluster on its tile's edge
//! `k` where the neighbour across it at the tile's level isn't reached, so the area there is drawn
//! by an ancestor, a level coarser by the neighbour rule (or off-screen, where it doesn't
//! matter); its finer bits where that neighbour is refined. This is what the bounds pass's cone
//! assumes (the variant's fans; see [`crate::indices`] for the triangles).
//!
//! # The visibility buffer
//!
//! [`VISIBILITY_FORMAT`] (`R32Uint`): per pixel, a [`VisibilityId`], the tile tree's node
//! (bits 13 to 32), the cluster within the tile (7 to 13) and the triangle within the cluster
//! (0 to 7, its place in the static index buffer), or [`BACKGROUND`]. The node, not the slot,
//! because a large epoch can move a tile to a shadow slot while its node stays (plan, "Gaps found
//! while executing"), so the ids are stable while the camera is still. Nineteen bits of node
//! allow [`MAX_NODES`] nodes, past the benchmark tree's 218,400. The triangle comes from the
//! primitive index, which restarts with each instance: [`wgpu::Features::PRIMITIVE_INDEX`], which
//! Vulkan has where the device has geometry shaders (lavapipe has; Metal on some GPUs, to check on
//! the M4; the tablet's in M1c). Depth is reversed-Z [`DEPTH_FORMAT`]
//! (`Depth32Float`), cleared to 0 (infinitely far), nearer passing; back faces are culled, as the
//! cone culling assumes.
//!
//! # The geomorph
//!
//! An odd vertex of a tile (odd `i` or `j`: not in its parent level's grid) is drawn at `mix(mid,
//! p, t)`, from `mid`, the midpoint of its two parents in decision 4's triangulation (where the
//! parent level draws that point), to `p`, its position. Even vertices never move. `t` is the
//! smaller of two factors:
//!
//! - **Distance:** 0 while the parent's `e` (the bound the level change trades) projects at `p` to
//!   under [`RasterConfig::with_ramp`] pixels (by default the level choice's 1 px refine
//!   threshold), rising to 1 at twice that. The level choice refines by a conservative bound over
//!   the tile's sphere, so a tile is usually drawn as its parent at first, and grows its detail as
//!   the camera nears; the drawn surface stays within the threshold of the finest surface meanwhile
//!   (it is the parent's surface while `t` is 0).
//! - **Time:** for a level change made while the camera is still, 0 when the tile came into the
//!   cut, rising to 1 over [`RasterConfig::with_morph`] (150 ms, spec). Where it lives (the design
//!   question): **on the GPU, a stamp per node** (WGSL's `MorphState`: the tile, the frame it was
//!   last in the cut, the time it came in), written by a stamp pass over the level choice's cut
//!   each frame (`morph.wgsl`), and the frame's time from the CPU in the frame's uniform
//!   ([`RasterFrame`]). The cut is only known on the GPU in the frame it is drawn (the CPU reads it
//!   back a frame or two later), so the stamps can only be kept there; the time is the CPU's, so a
//!   still camera at a given time always draws the same, and once every stamp is older than the
//!   duration every `t` is exactly 1: the ids are stable. A tile new to the cut is stamped "now"
//!   only when it refines what was drawn last frame and the camera is still; otherwise it is
//!   stamped a whole duration ago (no morph): moving, the distance factor alone morphs; a tile
//!   replacing finer ones, or one with nothing drawn before it, has nothing to morph from. **Not
//!   covered:** a coarsening made with the camera still (a refresh that lowers `e`) still pops, by
//!   at most the threshold: morphing out would need the finer tiles drawn for the duration after
//!   the level choice dropped them.
//!
//! **At seams** a border vertex's factors come from inputs both tiles share, so both draw it at
//! the same bits: the larger of the two tiles' parent `e`, the less advanced of their stamps,
//! and the vertex's own position (bit-identical on both sides, spec "Display layer"); the
//! midpoint's two parents are border vertices too. Where the neighbour is drawn finer the vertex
//! isn't morphed (`t` = 1): the finer side draws it as an even vertex, which never moves. Beside a
//! coarser one, the variant has already replaced it by an even vertex.
//!
//! **The cone:** a morphing triangle's normal stays in the tile's cone when one of its vertices
//! moves (the cone holds both ends, `cull.rs`), but this grid's triangles have two odd vertices
//! each, moving together, whose normal is quadratic in `t` and can leave it slightly while a morph
//! runs. The level choice doesn't widen its drawing cone test for that yet: a tile all of whose
//! triangles face away at both ends of a morph could, mid-morph, have a sliver facing the eye at
//! the silhouette and still be culled. To settle with Task 5's golden images.
//!
//! # Slot buffers
//!
//! One draw reads tiles from every slot buffer, so the raster binds them all at once, one binding
//! each (group 1), and selects between them in WGSL by class and buffer: the shader is generated
//! for the frame's buffer counts per class and rebuilt when they change (residency only ever adds
//! buffers, a few times a session). Every tier measured has room in
//! `max_storage_buffers_per_shader_stage` (desktop: 8 buffers; lavapipe, the gate: 38 of 48);
//! past it [`RasterError::TooManySlotBuffers`].
//!
//! # A frame
//!
//! [`Raster::begin_frame`] (the frame's view, time and whether the camera moves, written through
//! the queue: once per submission), then after the level choice's first phase
//! [`Raster::encode_stamp`] over its cut, then [`Raster::encode_draw`] of each phase's list,
//! clearing first ([`Phase::First`]), keeping after ([`Phase::Second`]).

use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoder, CompareFunction, ComputePassDescriptor,
    ComputePipeline, DepthBiasState, DepthStencilState, Extent3d, Face, Features, FragmentState,
    FrontFace, IndexFormat, LoadOp, MultisampleState, Operations, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StencilState, StoreOp, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor, VertexState,
};

use crate::camera::{GpuView, View};
use crate::cull::{CULL_WGSL, SlotBuffers, entry, pipeline, storage_buffer, uniform_buffer};
use crate::gpu::{Gpu, GpuError};
use crate::indices::{self, INDICES_WGSL};
use crate::lod::{CLUSTER_WGSL, CutList};
use crate::surface::Size;
use crate::tree::NODE_WGSL;
use crate::upload::VERTEX_WGSL;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests;

/// The raster's own source, after the declarations it uses and the generated slot bindings.
const RASTER_WGSL: &str = include_str!("raster.wgsl");

/// The stamp pass's own source, after [`NODE_WGSL`] and [`MORPH_WGSL`].
const STAMP_WGSL: &str = include_str!("morph.wgsl");

/// The WGSL declarations the raster and its stamp pass share: `RasterParams`
/// ([`GpuRasterParams`]), `MorphState` ([`GpuMorphState`]) and `MOVING`.
const MORPH_WGSL: &str = r"
// The raster's frame settings (render::raster::GpuRasterParams): 32 bytes.
struct RasterParams {
    // The frame's number: stamps of this frame are this, of the last one this minus one.
    frame: u32,
    // The frame's time, milliseconds (wrapping).
    now: u32,
    // The morph's duration, milliseconds; 0 for none.
    duration: u32,
    // MOVING while the camera moves.
    flags: u32,
    // The distance factor's ramp: 0 while e projects under this many pixels, 1 from twice it
    // on; 0 to morph by time alone.
    ramp: f32,
    // The stamp pass: the cut's count's word in its counts buffer.
    count_word: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
}

// Per node, when its tile came into the cut (render::raster::GpuMorphState): 16 bytes.
struct MorphState {
    // The tile's face, as TileNode.face: the stamp is its tile's only if both match.
    face: u32,
    // The tile's place, as TileNode.place.
    place: u32,
    // The frame the tile was last in the cut.
    frame: u32,
    // When it came into the cut, milliseconds, as RasterParams.now.
    start: u32,
}

// RasterParams.flags: the camera moves this frame.
const MOVING: u32 = 1u;
";

/// The visibility buffer's format: one [`VisibilityId`] a pixel.
pub const VISIBILITY_FORMAT: TextureFormat = TextureFormat::R32Uint;

/// The depth buffer's format: reversed Z, 1 at the near plane, 0 infinitely far.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// The visibility buffer where nothing is drawn.
pub const BACKGROUND: u32 = u32::MAX;

/// Bits of a [`VisibilityId`] below its node.
const NODE_SHIFT: u32 = 13;

/// Bits of a [`VisibilityId`] below its cluster.
const CLUSTER_SHIFT: u32 = 7;

/// The most nodes the raster covers: those whose ids fit in 19 bits, all but the last (whose
/// last id would be [`BACKGROUND`]).
pub const MAX_NODES: u32 = (u32::MAX >> NODE_SHIFT) - 1;

/// The morph's duration by default (spec, "Seams without cracks").
pub const MORPH: Duration = Duration::from_millis(150);

/// The first frame the raster numbers: stamps start zeroed, so no zeroed stamp is ever last
/// frame's.
const FIRST_FRAME: u32 = 2;

/// Threads per workgroup of the stamp pass.
const GROUP: u32 = 64;

/// Storage buffers of group 0 the vertex stage reads: the tree, the clusters, the remap tables
/// and the stamps.
const FIXED_STORAGE: u32 = 4;

/// One pixel of the visibility buffer, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VisibilityId {
    /// The tile's node in the tile tree.
    pub node: u32,
    /// The cluster within the tile, `cy · per + cx` for a tile of `per × per` clusters.
    pub cluster: u32,
    /// The triangle within the cluster: its place in the static index buffer
    /// ([`crate::indices::triangles`]).
    pub triangle: u32,
}

impl VisibilityId {
    /// The id a pixel holds, `None` for [`BACKGROUND`].
    #[must_use]
    pub const fn decode(pixel: u32) -> Option<Self> {
        if pixel == BACKGROUND {
            return None;
        }
        Some(Self {
            node: pixel >> NODE_SHIFT,
            cluster: (pixel >> CLUSTER_SHIFT) & ((1 << (NODE_SHIFT - CLUSTER_SHIFT)) - 1),
            triangle: pixel & ((1 << CLUSTER_SHIFT) - 1),
        })
    }

    /// The pixel's value.
    #[must_use]
    pub const fn encode(self) -> u32 {
        (self.node << NODE_SHIFT) | (self.cluster << CLUSTER_SHIFT) | self.triangle
    }
}

/// How the raster morphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterConfig {
    /// The time factor's duration.
    morph: Duration,
    /// The distance factor's ramp, pixels.
    ramp: f32,
}

impl RasterConfig {
    /// The morph over `duration` (zero: no time factor).
    #[must_use]
    pub const fn with_morph(self, duration: Duration) -> Self {
        Self {
            morph: duration,
            ..self
        }
    }

    /// The distance factor 0 while the parent's `e` projects under `pixels`, 1 from twice that
    /// (0 or less: no distance factor).
    #[must_use]
    pub const fn with_ramp(self, pixels: f32) -> Self {
        Self {
            ramp: pixels,
            ..self
        }
    }
}

impl Default for RasterConfig {
    /// [`MORPH`], and the ramp at the level choice's refine threshold, 1 px.
    fn default() -> Self {
        Self {
            morph: MORPH,
            ramp: crate::lod::Thresholds::SPEC.refine_error,
        }
    }
}

/// What a frame's raster needs from the CPU.
#[derive(Debug, Clone, Copy)]
pub struct RasterFrame<'a> {
    /// The view: its size is the visibility buffer's.
    pub view: &'a View,
    /// The frame's time, from any fixed start (the morph compares times of the same clock).
    pub time: Duration,
    /// Whether the camera moves this frame: level changes then don't morph by time.
    pub moving: bool,
}

/// Which of a frame's two draws: the first clears the targets, the second draws over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The first phase's list: the targets are cleared first.
    First,
    /// The second phase's: drawn over the first's.
    Second,
}

/// One draw's inputs.
#[derive(Debug, Clone, Copy)]
pub struct ClusterDraw<'a> {
    /// The tile tree's buffer.
    pub tree: &'a Buffer,
    /// The slot buffers the tree's slots name.
    pub slots: &'a SlotBuffers,
    /// The cluster list ([`crate::lod::Lod::clusters`]).
    pub clusters: &'a Buffer,
    /// The indexed indirect draw arguments and their offset ([`crate::lod::Lod::draw_args`]).
    pub args: (&'a Buffer, u64),
}

/// Why the raster couldn't be set up or encoded.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterError {
    /// The GPU failed: a buffer, a pipeline, a bind group.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// More nodes than a visibility id holds.
    #[error("{0} nodes: the visibility buffer's ids hold at most {MAX_NODES}")]
    Capacity(u32),
    /// The device can't give a fragment its primitive index.
    #[error("the device has no primitive index (wgpu's PRIMITIVE_INDEX feature)")]
    NoPrimitiveIndex,
    /// More slot buffers than one draw can bind.
    #[error("{buffers} slot buffers: one draw binds at most {limit} beside its own")]
    TooManySlotBuffers {
        /// The slot buffers.
        buffers: u32,
        /// The most the device allows.
        limit: u32,
    },
    /// [`Raster::encode_stamp`] or [`Raster::encode_draw`] before [`Raster::begin_frame`].
    #[error("the raster has no frame begun")]
    NoFrame,
}

/// The frame's settings as the GPU reads them: WGSL's `RasterParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuRasterParams {
    /// The frame's number.
    frame: u32,
    /// Its time, milliseconds, wrapping.
    now: u32,
    /// The morph's duration, milliseconds.
    duration: u32,
    /// [`MOVING`] while the camera moves.
    flags: u32,
    /// The distance factor's ramp, pixels.
    ramp: f32,
    /// The cut's count's word.
    count_word: u32,
    /// Zero.
    reserved: [u32; 2],
}

/// [`GpuRasterParams::flags`]: the camera moves.
const MOVING: u32 = 1;

/// A node's stamp as the GPU keeps it: WGSL's `MorphState`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuMorphState {
    /// Its tile's face.
    face: u32,
    /// Its tile's place.
    place: u32,
    /// The frame it was last in the cut.
    frame: u32,
    /// When it came in, milliseconds.
    start: u32,
}

/// The render pipeline for one set of slot-buffer counts.
#[derive(Debug)]
struct SlotPipeline {
    /// Buffers per class it binds.
    counts: Vec<usize>,
    /// Group 1's layout: one binding a slot buffer.
    slot_layout: BindGroupLayout,
    /// The pipeline.
    pipeline: RenderPipeline,
    /// The shader source, for the tests' probe.
    #[cfg(test)]
    source: String,
}

/// The visibility and depth images.
#[derive(Debug)]
struct Targets {
    /// Their size.
    size: Size,
    /// The visibility buffer.
    visibility: Texture,
    /// A view of it.
    visibility_view: TextureView,
    /// The depth buffer.
    depth: Texture,
    /// A view of it.
    depth_view: TextureView,
}

/// The raster: its pipelines, the static buffers, the stamps and the frame's targets.
#[derive(Debug)]
pub struct Raster {
    /// Nodes covered.
    capacity: u32,
    /// How it morphs.
    config: RasterConfig,
    /// The current frame's number, once begun.
    frame: Option<u32>,
    /// The static index buffer.
    indices: Buffer,
    /// The remap tables.
    remap: Buffer,
    /// Per node, [`GpuMorphState`].
    stamps: Buffer,
    /// [`GpuRasterParams`].
    params: Buffer,
    /// The frame's [`GpuView`].
    view: Buffer,
    /// Group 0's layout.
    layout: BindGroupLayout,
    /// The stamp pass's layout.
    stamp_layout: BindGroupLayout,
    /// The stamp pass.
    stamp: ComputePipeline,
    /// The render pipeline for the slot buffers last drawn from.
    pipeline: Option<SlotPipeline>,
    /// The targets, sized to the frame's view.
    targets: Option<Targets>,
}

/// A layout entry for a read-only storage buffer seen by the vertex and compute stages.
const fn read_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Storage { read_only: true })
}

/// A layout entry for a buffer binding of `ty` seen by the vertex and compute stages.
const fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX.union(ShaderStages::COMPUTE),
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A layout entry for a storage buffer the compute stage writes (the vertex stage may not).
const fn written_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        visibility: ShaderStages::COMPUTE,
        ..buffer_entry(binding, BufferBindingType::Storage { read_only: false })
    }
}

impl Raster {
    /// The raster for a tile tree of `capacity` nodes, morphing by `config`.
    ///
    /// # Errors
    ///
    /// [`RasterError::Capacity`] above [`MAX_NODES`]; [`RasterError::NoPrimitiveIndex`] if the
    /// device lacks the feature; [`RasterError::Gpu`] if a buffer or pipeline can't be made.
    pub fn new(gpu: &Gpu, capacity: u32, config: RasterConfig) -> Result<Self, RasterError> {
        if capacity > MAX_NODES {
            return Err(RasterError::Capacity(capacity));
        }
        if !gpu.device().features().contains(Features::PRIMITIVE_INDEX) {
            return Err(RasterError::NoPrimitiveIndex);
        }
        let device = gpu.device();
        let indices = gpu.scoped("cluster indices", || {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("cluster indices"),
                contents: bytemuck::cast_slice(&indices::indices()),
                usage: BufferUsages::INDEX,
            })
        })?;
        let remap = gpu.scoped("cluster remap tables", || {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("cluster remap tables"),
                contents: bytemuck::cast_slice(&indices::remap_table()),
                usage: BufferUsages::STORAGE,
            })
        })?;
        let stamp_bytes = u64::from(capacity.max(1)) * size_of::<GpuMorphState>() as u64;
        let stamps = storage_buffer(gpu, "raster stamps", stamp_bytes)?;
        let params = uniform_buffer(gpu, "raster params", size_of::<GpuRasterParams>() as u64)?;
        let view = uniform_buffer(gpu, "raster view", size_of::<GpuView>() as u64)?;
        let uniform = BufferBindingType::Uniform;
        let layout = gpu.scoped("raster layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("raster"),
                entries: &[
                    read_entry(0),
                    read_entry(1),
                    read_entry(2),
                    read_entry(3),
                    buffer_entry(4, uniform),
                    buffer_entry(5, uniform),
                ],
            })
        })?;
        let stamp_layout = gpu.scoped("raster stamp layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("raster stamp"),
                entries: &[
                    read_entry(0),
                    read_entry(1),
                    read_entry(2),
                    written_entry(3),
                    buffer_entry(4, uniform),
                ],
            })
        })?;
        let stamp_source = format!("{NODE_WGSL}{MORPH_WGSL}{STAMP_WGSL}");
        let stamp = pipeline(
            gpu,
            "raster stamp",
            &stamp_source,
            &[&stamp_layout],
            "stamp",
        )?;
        Ok(Self {
            capacity,
            config,
            frame: None,
            indices,
            remap,
            stamps,
            params,
            view,
            layout,
            stamp_layout,
            stamp,
            pipeline: None,
            targets: None,
        })
    }

    /// How it morphs.
    #[must_use]
    pub const fn config(&self) -> &RasterConfig {
        &self.config
    }

    /// Begins a frame: numbers it, writes its settings and view through the queue (so once per
    /// submission), and sizes the targets to the view.
    ///
    /// # Errors
    ///
    /// [`RasterError::Gpu`] if a target can't be made.
    pub fn begin_frame(&mut self, gpu: &Gpu, frame: &RasterFrame<'_>) -> Result<(), RasterError> {
        let number = self.frame.map_or(FIRST_FRAME, |f| f.wrapping_add(1));
        self.frame = Some(number);
        let millis = |d: Duration| u32::try_from(d.as_millis() % (1 << 32)).unwrap_or(0);
        let params = GpuRasterParams {
            frame: number,
            now: millis(frame.time),
            duration: u32::try_from(self.config.morph.as_millis()).unwrap_or(u32::MAX),
            flags: if frame.moving { MOVING } else { 0 },
            ramp: self.config.ramp,
            count_word: 0,
            reserved: [0; 2],
        };
        let queue = gpu.queue();
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.view, 0, bytemuck::bytes_of(&frame.view.gpu()));
        let size = frame.view.size();
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Targets::new(gpu, size)?);
        }
        Ok(())
    }

    /// Encodes the stamp pass over `cut`, the frame's cut (after the level choice's first
    /// phase, before either draw), reading `tree`.
    ///
    /// # Errors
    ///
    /// [`RasterError::NoFrame`] before [`Raster::begin_frame`]; [`RasterError::Gpu`] if a bind
    /// group can't be made.
    pub fn encode_stamp(
        &self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        tree: &Buffer,
        cut: CutList<'_>,
    ) -> Result<(), RasterError> {
        if self.frame.is_none() {
            return Err(RasterError::NoFrame);
        }
        // The count's word goes into the frame's settings (the queue's writes land before the
        // submission).
        gpu.queue().write_buffer(
            &self.params,
            std::mem::offset_of!(GpuRasterParams, count_word) as u64,
            bytemuck::bytes_of(&cut.count_word),
        );
        let group = gpu.scoped("raster stamp bind group", || {
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("raster stamp"),
                layout: &self.stamp_layout,
                entries: &[
                    entry(0, tree),
                    entry(1, cut.entries),
                    entry(2, cut.counts),
                    entry(3, &self.stamps),
                    entry(4, &self.params),
                ],
            })
        })?;
        let entries = u32::try_from(cut.entries.size() / 16).unwrap_or(u32::MAX);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("raster stamp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.stamp);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(entries.min(self.capacity).div_ceil(GROUP).max(1), 1, 1);
        Ok(())
    }

    /// Encodes one phase's draw: its clusters, one instance each, into the visibility and depth
    /// buffers, cleared first for [`Phase::First`].
    ///
    /// # Errors
    ///
    /// [`RasterError::NoFrame`] before [`Raster::begin_frame`];
    /// [`RasterError::TooManySlotBuffers`] if the slot buffers don't fit beside the raster's own;
    /// [`RasterError::Gpu`] if the pipeline or a bind group can't be made.
    pub fn encode_draw(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        draw: &ClusterDraw<'_>,
        phase: Phase,
    ) -> Result<(), RasterError> {
        if self.frame.is_none() {
            return Err(RasterError::NoFrame);
        }
        self.prepare_pipeline(gpu, draw.slots)?;
        let (Some(targets), Some(slot_pipeline)) = (&self.targets, &self.pipeline) else {
            return Err(RasterError::NoFrame);
        };
        let (group, slot_group) = self.groups(gpu, draw, slot_pipeline)?;
        let clear = phase == Phase::First;
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("raster"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &targets.visibility_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: if clear {
                        LoadOp::Clear(Color {
                            r: f64::from(BACKGROUND),
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        })
                    } else {
                        LoadOp::Load
                    },
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: Some(Operations {
                    load: if clear {
                        LoadOp::Clear(0.0)
                    } else {
                        LoadOp::Load
                    },
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..RenderPassDescriptor::default()
        });
        pass.set_pipeline(&slot_pipeline.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &slot_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), IndexFormat::Uint16);
        pass.draw_indexed_indirect(draw.args.0, draw.args.1);
        Ok(())
    }

    /// Group 0 and group 1 of a draw.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn groups(
        &self,
        gpu: &Gpu,
        draw: &ClusterDraw<'_>,
        slot_pipeline: &SlotPipeline,
    ) -> Result<(BindGroup, BindGroup), GpuError> {
        let device = gpu.device();
        let group = gpu.scoped("raster bind group", || {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("raster"),
                layout: &self.layout,
                entries: &[
                    entry(0, draw.tree),
                    entry(1, draw.clusters),
                    entry(2, &self.remap),
                    entry(3, &self.stamps),
                    entry(4, &self.params),
                    entry(5, &self.view),
                ],
            })
        })?;
        let buffers = draw.slots.classes().iter().flatten();
        let entries = buffers
            .zip(0..)
            .map(|(buffer, binding)| entry(binding, buffer));
        let entries: Vec<BindGroupEntry<'_>> = entries.collect();
        let slot_group = gpu.scoped("raster slot bind group", || {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("raster slots"),
                layout: &slot_pipeline.slot_layout,
                entries: &entries,
            })
        })?;
        Ok((group, slot_group))
    }

    /// Makes sure the pipeline binds `slots`' buffer counts, rebuilding it if they changed.
    ///
    /// # Errors
    ///
    /// [`RasterError::TooManySlotBuffers`]; [`RasterError::Gpu`] if the pipeline can't be made.
    fn prepare_pipeline(&mut self, gpu: &Gpu, slots: &SlotBuffers) -> Result<(), RasterError> {
        let counts: Vec<usize> = slots.classes().iter().map(Vec::len).collect();
        if self.pipeline.as_ref().is_some_and(|p| p.counts == counts) {
            return Ok(());
        }
        let buffers = u32::try_from(counts.iter().sum::<usize>()).unwrap_or(u32::MAX);
        // The probe's two buffers count against the compute stage's limit too.
        let limit = gpu
            .device()
            .limits()
            .max_storage_buffers_per_shader_stage
            .saturating_sub(FIXED_STORAGE + 2);
        if buffers > limit {
            return Err(RasterError::TooManySlotBuffers { buffers, limit });
        }
        self.pipeline = Some(SlotPipeline::new(gpu, &self.layout, counts)?);
        Ok(())
    }

    /// The visibility buffer, once a frame has begun: [`VISIBILITY_FORMAT`], the view's size.
    #[must_use]
    pub fn visibility(&self) -> Option<&Texture> {
        self.targets.as_ref().map(|t| &t.visibility)
    }

    /// The depth buffer, once a frame has begun: [`DEPTH_FORMAT`], the view's size.
    #[must_use]
    pub fn depth(&self) -> Option<&Texture> {
        self.targets.as_ref().map(|t| &t.depth)
    }

    /// A view of the depth buffer, for the depth pyramid ([`crate::cull::PyramidPass::build`]).
    #[must_use]
    pub fn depth_view(&self) -> Option<&TextureView> {
        self.targets.as_ref().map(|t| &t.depth_view)
    }
}

impl SlotPipeline {
    /// The pipeline binding `counts[c]` slot buffers of each class `c`, group 0 on `layout`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new(gpu: &Gpu, layout: &BindGroupLayout, counts: Vec<usize>) -> Result<Self, GpuError> {
        let device = gpu.device();
        let bindings = u32::try_from(counts.iter().sum::<usize>()).unwrap_or(0);
        let entries = (0..bindings).map(read_entry);
        let entries: Vec<BindGroupLayoutEntry> = entries.collect();
        let slot_layout = gpu.scoped("raster slot layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("raster slots"),
                entries: &entries,
            })
        })?;
        let source = format!(
            "enable primitive_index;\n{NODE_WGSL}{VERTEX_WGSL}{CULL_WGSL}{CLUSTER_WGSL}\
             {INDICES_WGSL}{MORPH_WGSL}{}{RASTER_WGSL}",
            slot_source(&counts)
        );
        let pipeline = gpu.scoped("raster pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("raster"),
                source: ShaderSource::Wgsl(source.as_str().into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("raster"),
                bind_group_layouts: &[Some(layout), Some(&slot_layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("raster"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("cluster_vertex"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(Face::Back),
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::Greater),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("visibility"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: VISIBILITY_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })?;
        Ok(Self {
            counts,
            slot_layout,
            pipeline,
            #[cfg(test)]
            source,
        })
    }
}

/// The generated WGSL of the slot bindings, for `counts[c]` buffers of each class `c`.
///
/// `slots_<class>_<buffer>` at group 1, one binding each in class then buffer order, and
/// `slot_vertex(size_class, slot_buffer, i)` reading vertex `i` of that buffer (a zeroed vertex
/// for a buffer not bound).
fn slot_source(counts: &[usize]) -> String {
    let buffers = counts
        .iter()
        .enumerate()
        .flat_map(|(class, &count)| (0..count).map(move |buffer| (class, buffer)));
    let buffers: Vec<(usize, usize)> = buffers.collect();
    let declarations = buffers
        .iter()
        .zip(0_u32..)
        .map(|((class, buffer), binding)| {
            format!(
                "@group(1) @binding({binding}) var<storage, read> slots_{class}_{buffer}: \
             array<TileVertex>;\n"
            )
        });
    let declarations: String = declarations.collect();
    let cases = buffers.iter().map(|(class, buffer)| {
        let key = class * 256 + buffer;
        format!("        case {key}u: {{ return slots_{class}_{buffer}[i]; }}\n")
    });
    let cases: String = cases.collect();
    format!(
        "
// The frame's slot buffers (generated by render::raster for its buffer counts per class).
{declarations}
// Vertex `i` of slot buffer `slot_buffer` of size class `size_class`.
fn slot_vertex(size_class: u32, slot_buffer: u32, i: u32) -> TileVertex {{
    switch size_class * 256u + slot_buffer {{
{cases}        default: {{}}
    }}
    return TileVertex();
}}
"
    )
}

impl Targets {
    /// Visibility and depth images of `size`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new(gpu: &Gpu, size: Size) -> Result<Self, GpuError> {
        let device = gpu.device();
        let extent = Extent3d {
            width: size.width(),
            height: size.height(),
            depth_or_array_layers: 1,
        };
        let image = |label, format| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        gpu.scoped("raster targets", || {
            let visibility = image("visibility", VISIBILITY_FORMAT);
            let depth = image("depth", DEPTH_FORMAT);
            Self {
                size,
                visibility_view: visibility.create_view(&TextureViewDescriptor::default()),
                visibility,
                depth_view: depth.create_view(&TextureViewDescriptor::default()),
                depth,
            }
        })
    }
}
