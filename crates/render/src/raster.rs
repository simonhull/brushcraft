//! The raster: the level choice's visible clusters drawn into a visibility buffer, watertight at
//! every seam, with the geomorph that keeps level changes from popping.
//!
//! # What it draws
//!
//! Each phase of the level choice ([`crate::lod`]) lists its visible clusters
//! ([`crate::lod::GpuCluster`]: a tile's node, the cluster's place in the tile, its edge variant
//! and its finer edges) with indexed indirect draw arguments. The raster draws a list in **one
//! instanced indexed indirect draw** ([`Raster::encode_draw`]), one instance a cluster, over one
//! static index buffer of the cluster's 128 triangles (`indices.rs`). The vertex shader
//! (`raster.wgsl`) remaps the index it is given by the cluster's edge variant (the 16 variants are
//! tables of one static buffer, `indices.rs`' `remap_table`), reads the vertex from its
//! tile's slot and geomorphs it; the fragment shader writes the visibility id.
//!
//! **Which variant** is the level choice's: `emit` sets bit `k` of a cluster on its tile's edge
//! `k` where the neighbour across it at the tile's level isn't reached, so the area there is drawn
//! by an ancestor, a level coarser by the neighbour rule (or off-screen, where it doesn't
//! matter); its finer bits where that neighbour is refined. This is what the bounds pass's cone
//! assumes (the variant's fans; see `indices.rs` for the triangles).
//!
//! # The visibility buffer
//!
//! [`VISIBILITY_FORMAT`] (`R32Uint`): per pixel, a [`VisibilityId`], the tile tree's node
//! (bits 12 to 32), the cluster within the tile (6 to 12) and the quad within the cluster (0 to
//! 6, `8y + x`), or [`BACKGROUND`] (user decision, 2026-10-29). The node, not the slot, because a
//! large epoch can move a tile to a shadow slot while its node stays (plan, "Gaps found while
//! executing"), so the ids are stable while the camera is still. Twenty bits of node allow
//! [`MAX_NODES`] nodes, past the benchmark tree's 218,400.
//!
//! The quad comes flat from each triangle's provoking vertex, its first (WGSL's
//! `@interpolate(flat, first)`): both of quad `(x, y)`'s triangles start at its `p00`, grid
//! vertex `(x, y)`, and every vertex hands on the quad its *unremapped* grid index starts, so the
//! edge variants don't change it (a vertex a variant collapses still provokes its quad's
//! surviving triangle). No geometry-shader feature is needed (`PRIMITIVE_INDEX` is missing on
//! Mali and `PowerVR` GPUs). Which of the quad's two triangles a pixel is in, the shading pass
//! tells by one test against the quad's diagonal: [`quad_triangle`], and [`QUAD_WGSL`] for the GPU.
//!
//! Depth is reversed-Z [`DEPTH_FORMAT`] (`Depth32Float`), cleared to 0 (infinitely far), nearer
//! passing; back faces are culled, as the cone culling assumes.
//!
//! # The geomorph
//!
//! An odd vertex of a tile (odd `i` or `j`: not in its parent level's grid) is drawn at `mix(mid,
//! p, t)`, from `mid`, the midpoint of its two parents in decision 4's triangulation (where the
//! parent level draws that point), to `p`, its position. Even vertices never move. `t` is the
//! smaller of two factors:
//!
//! - **Distance** (user decision, 2026-10-29): `t = (x − r) / (x − y)`, clamped to `[0, 1]`, with
//!   `x` the parent's `e` (the bound the level change trades) and `y` the tile's own, both
//!   projected at `p` in pixels, and `r` the level choice's refine threshold
//!   ([`RasterConfig::for_level_choice`], 1 px by default). The drawn surface's error, about `(1 −
//!   t) x + t y`, then stays at `r` while it can, and the morph ends just as the tile's own `e`
//!   reaches `r`, where the level choice refines the tile (by its conservative bound, a little
//!   earlier): the tile's odd vertices are where its children's even ones will be, so the
//!   refinement doesn't pop but by that bound's slack. Where `x − y` is 0 or less (a parent bound
//!   no larger than the tile's own), `t` is 1 if `y` is within `r`, else 0. At `t = 0` the surface
//!   is the tile's even vertices, which are the parent's vertices re-quantized at the parent's step
//!   (≤ 1/64 of a quad, about 0.06 px at the threshold), not the parent's bits.
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
//!   covered, recorded (user decision, 2026-10-29; plan, "Gaps found while executing", the "Task 4
//!   review" bullet):** a coarsening made with the camera still (a refresh that lowers `e`) pops,
//!   by at most the threshold: morphing out would need the finer tiles drawn for the duration after
//!   the level choice dropped them. To revisit with Task 5's golden images.
//!
//! **At seams** a border vertex's factors come from inputs both tiles share, so both draw it at
//! the same bits: the larger of the two tiles' parent `e`, the larger of their own `e`, the less
//! advanced of their stamps, and the vertex's own position (bit-identical on both sides, spec
//! "Display layer"); the
//! midpoint's two parents are border vertices too. Where the neighbour is drawn finer the vertex
//! isn't morphed (`t` = 1): the finer side draws it as an even vertex, which never moves. Beside a
//! coarser one, the variant has already replaced it by an even vertex.
//!
//! **The cone:** this grid's triangles have two or three odd vertices each, every one with its
//! own factor, so a morphing triangle's normal isn't on the arc between its two ends and can leave
//! the tile's cone, which holds its drawn triangles at either end of the morph only (`cull.rs`,
//! "What the cone holds"). The level choice reads this module's stamps and frame settings and
//! culls a tile by its cone only when every odd vertex is certainly at one end
//! ([`crate::lod::MorphInputs`]; `lod.rs`, "Cone culling and the morph"), so no sliver facing the
//! eye is dropped mid-morph. Nothing else here depends on the cone: the shading pass rebuilds only
//! what was drawn, and the anti-aliasing's reset rule (`aa.wgsl`) needs nothing new, since a
//! settled tile its cone faces away draws only back faces, which the raster culls: culled or drawn,
//! the image is the same.
//!
//! # For the shading pass
//!
//! The shading pass ([`crate::shade`]) rebuilds the triangles drawn at each pixel from its
//! [`VisibilityId`], which names the node, the cluster and the quad, but not what else placed the
//! quad's corners: the cluster's edge variant and finer edges, and the morph. Both come from
//! buffers the raster keeps per node, read with the same code the vertex shader runs
//! (`place.wgsl`'s `place_vertex`):
//!
//! - **The drawn edges** (`crate::lod::DrawnEdges`): the level choice's `emit` records, in each cut
//!   entry, across which of the tile's edges the neighbour is drawn coarser and finer, and the
//!   stamp pass keeps that byte per node. A cluster's variant and finer bits are the tile's on the
//!   edges the cluster lies on, so the byte and the cluster's place give back the very
//!   `Cluster.cell` emit listed (`lod.rs`' `CLUSTER_WGSL`, `cluster_cell`, which emit lists
//!   clusters with too).
//! - **The morph:** the stamps, the frame's settings and its view, as the frame drawn left them.
//!   They change only when the next frame is begun, so a shading pass run again without a raster (a
//!   material change) still places every vertex as it was drawn.
//!
//! # Slot buffers
//!
//! One draw reads tiles from every slot buffer, so the raster binds them all at once, one binding
//! each (group 1), and selects between them in WGSL by class and buffer. The pipeline is built
//! once, in [`Raster::new`], for the most buffers each class can ever have
//! ([`crate::residency::SlotLayout`], from the budget and each class's tiles with their shadow
//! headroom), a placeholder bound for each not made yet, so nothing is recompiled mid-session.
//! Residency refuses a budget whose most buffers don't fit `max_storage_buffers_per_shader_stage`
//! beside the raster's own four ([`slot_buffer_limit`]; on Metal its two uniform buffers count
//! against the same table); every tier measured fits (desktop: 14 buffers for the benchmark
//! project; lavapipe, the gate: 40 of its 44). A tier past it would draw one indirect draw per slot
//! buffer from cluster lists split by buffer: not built (plan, "Gaps found while executing", the
//! "Task 4 review" bullet).
//!
//! # A frame
//!
//! [`Raster::begin_frame`] (the frame's view, time and whether the camera moves, written through
//! the queue: once per submission), then after the level choice's first phase
//! [`Raster::encode_stamp`] over its cut (the stamps and the drawn edges), then
//! [`Raster::encode_draw`] of each phase's list, clearing first ([`Phase::First`]), keeping after
//! ([`Phase::Second`]). [`FramePasses`] runs the whole frame in that order with the level choice
//! and the two depth pyramids.

use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoder, CompareFunction, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipeline, DepthBiasState, DepthStencilState, Extent3d, Face,
    FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPassTimestampWrites, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, StoreOp, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexState,
};

use crate::camera::{GpuView, View};
use crate::cull::{CULL_WGSL, SlotBuffers, entry, pipeline, storage_buffer, uniform_buffer};
use crate::gpu::{Gpu, GpuError};
use crate::indices::{self, INDICES_WGSL};
use crate::lod::{CLUSTER_WGSL, CutList, LodConfig, MorphInputs};
use crate::residency::{CLASSES, SizeClass, SlotLayout};
use crate::tree::NODE_WGSL;
use crate::upload::{VERTEX_BYTES, VERTEX_WGSL};

mod frame;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests;

pub use frame::{FrameError, FramePasses, SceneInputs};

/// The raster's own source, after the declarations it uses and the generated slot bindings.
const RASTER_WGSL: &str = include_str!("raster.wgsl");

/// Where a cluster vertex is drawn: the geomorph.
///
/// Shared by the raster's vertex shader and the shading pass ([`crate::shade`]), so what is
/// shaded is what was drawn. After the declarations [`RASTER_WGSL`] lists; the including shader
/// declares the globals it reads.
pub(crate) const PLACE_WGSL: &str = include_str!("place.wgsl");

/// The stamp pass's own source, after [`NODE_WGSL`], [`CLUSTER_WGSL`] and [`MORPH_WGSL`].
const STAMP_WGSL: &str = include_str!("morph.wgsl");

/// The WGSL declarations the raster and its stamp pass share: `RasterParams`
/// ([`GpuRasterParams`]), `MorphState` ([`GpuMorphState`]) and `MOVING`.
pub(crate) const MORPH_WGSL: &str = r"
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
const NODE_SHIFT: u32 = 12;

/// Bits of a [`VisibilityId`] below its cluster.
const CLUSTER_SHIFT: u32 = 6;

/// The most nodes the raster covers: those whose ids fit in 20 bits, all but the last (whose
/// last id would be [`BACKGROUND`]).
pub const MAX_NODES: u32 = (u32::MAX >> NODE_SHIFT) - 1;

/// The WGSL of [`quad_triangle`], for the shading pass: `quad_triangle(p00, p10, p11, p01,
/// pixel) -> u32`, the corners and the pixel in any one 2D frame (pixels).
pub const QUAD_WGSL: &str = r"
// Which triangle of a quad a pixel is in (render::raster::quad_triangle): 0 for (p00, p10,
// p11), 1 for (p00, p11, p01). The corners are where they are drawn (remapped and morphed),
// projected; the two triangles lie on either side of their diagonal p00-p11, and one an edge
// variant collapsed covers nothing.
fn quad_triangle(
    p00: vec2<f32>,
    p10: vec2<f32>,
    p11: vec2<f32>,
    p01: vec2<f32>,
    pixel: vec2<f32>,
) -> u32 {
    let d = p11 - p00;
    let side = d.x * (pixel.y - p00.y) - d.y * (pixel.x - p00.x);
    let lower = d.x * (p10.y - p00.y) - d.y * (p10.x - p00.x);
    let upper = d.x * (p01.y - p00.y) - d.y * (p01.x - p00.x);
    if lower == 0.0 {
        return 1u;
    }
    if upper == 0.0 {
        return 0u;
    }
    return select(0u, 1u, (side > 0.0) == (upper > 0.0));
}
";

/// Which triangle of a quad a pixel is in: 0 for `(p00, p10, p11)`, 1 for `(p00, p11, p01)`,
/// their places in the quad's pair of the static index buffer (`indices.rs`).
///
/// `corners` are `[p00, p10, p11, p01]` where they are drawn (remapped by the cluster's edge
/// variant and morphed), projected, and `pixel` in the same frame. The two triangles lie on either
/// side of their shared diagonal `p00`–`p11`, so one side test tells them apart; one an edge
/// variant collapsed (its `p10` or `p01` on the diagonal) covers nothing, and the other is chosen.
/// A pixel on the diagonal itself goes to triangle 0; the rasterizer's tie rule may differ there.
#[must_use]
pub fn quad_triangle(corners: [[f32; 2]; 4], pixel: [f32; 2]) -> u32 {
    let [p00, p10, p11, p01] = corners;
    let d = [p11[0] - p00[0], p11[1] - p00[1]];
    let cross = |q: [f32; 2]| d[0] * (q[1] - p00[1]) - d[1] * (q[0] - p00[0]);
    let (side, lower, upper) = (cross(pixel), cross(p10), cross(p01));
    if lower == 0.0 {
        return 1;
    }
    if upper == 0.0 {
        return 0;
    }
    u32::from((side > 0.0) == (upper > 0.0))
}

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

/// Uniform buffers of group 0 the vertex stage reads: the settings and the view.
const FIXED_UNIFORMS: u32 = 2;

/// The most slot buffers the raster's one draw, and the shading pass, can bind on `gpu`'s device.
///
/// That is the smaller of the two passes' room beside their own buffers (the shading pass with
/// the renderer's material, the MatCap), within the device's per-stage limits.
#[must_use]
pub fn slot_buffer_limit(gpu: &Gpu) -> u32 {
    let (storage, uniforms) = crate::shade::fixed_buffers::<crate::material::MatCap>();
    slots_beside(gpu, FIXED_STORAGE, FIXED_UNIFORMS).min(slots_beside(gpu, storage, uniforms))
}

/// The most slot buffers a stage can bind on `gpu`'s device beside other buffers.
///
/// Those are `storage` other storage buffers and `uniforms` uniform buffers. The slots stay
/// within `max_storage_buffers_per_shader_stage`, and within
/// `max_buffers_and_acceleration_structures_per_shader_stage`, the table Metal's storage and
/// uniform buffers share (other backends report no such bound).
pub(crate) fn slots_beside(gpu: &Gpu, storage: u32, uniforms: u32) -> u32 {
    let limits = gpu.device().limits();
    let alone = limits
        .max_storage_buffers_per_shader_stage
        .saturating_sub(storage);
    let shared = limits
        .max_buffers_and_acceleration_structures_per_shader_stage
        .saturating_sub(storage + uniforms);
    alone.min(shared)
}

/// One pixel of the visibility buffer, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VisibilityId {
    /// The tile's node in the tile tree.
    pub node: u32,
    /// The cluster within the tile, `cy · per + cx` for a tile of `per × per` clusters.
    pub cluster: u32,
    /// The quad within the cluster, `8y + x`: its triangles are the static index buffer's
    /// `2 (8y + x)` and the one after ([`quad_triangle`] tells which).
    pub quad: u32,
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
            quad: pixel & ((1 << CLUSTER_SHIFT) - 1),
        })
    }

    /// The pixel's value.
    #[must_use]
    pub const fn encode(self) -> u32 {
        (self.node << NODE_SHIFT) | (self.cluster << CLUSTER_SHIFT) | self.quad
    }
}

/// How the raster morphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterConfig {
    /// The time factor's duration.
    morph: Duration,
    /// The distance factor's threshold.
    ramp: MorphThreshold,
}

/// The error, in pixels, the distance factor holds the drawn surface to: finite and not
/// negative; [`MorphThreshold::OFF`] for no distance factor (the morph by time alone).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct MorphThreshold(f32);

impl MorphThreshold {
    /// No distance factor: odd vertices morph by time alone.
    pub const OFF: Self = Self(0.0);

    /// `pixels`, if finite and not negative (0 is [`MorphThreshold::OFF`]).
    #[must_use]
    pub fn new(pixels: f32) -> Option<Self> {
        (pixels.is_finite() && pixels >= 0.0).then_some(Self(pixels))
    }

    /// The pixels.
    #[must_use]
    pub const fn pixels(self) -> f32 {
        self.0
    }
}

impl RasterConfig {
    /// The raster for `lod`'s level choice: the distance factor at its refine threshold (no
    /// distance factor if that isn't a valid threshold), the time factor over [`MORPH`].
    #[must_use]
    pub fn for_level_choice(lod: &LodConfig) -> Self {
        Self {
            morph: MORPH,
            ramp: MorphThreshold::new(lod.thresholds().refine_error).unwrap_or(MorphThreshold::OFF),
        }
    }

    /// The time factor's duration: how long a level change made with the camera still morphs.
    #[must_use]
    pub const fn morph(&self) -> Duration {
        self.morph
    }

    /// The morph over `duration` (zero: no time factor).
    #[must_use]
    pub const fn with_morph(self, duration: Duration) -> Self {
        Self {
            morph: duration,
            ..self
        }
    }

    /// The distance factor at `threshold` instead of the level choice's: for tests and tools
    /// ([`MorphThreshold::OFF`] morphs by time alone).
    #[must_use]
    pub const fn with_ramp(self, threshold: MorphThreshold) -> Self {
        Self {
            ramp: threshold,
            ..self
        }
    }
}

impl Default for RasterConfig {
    /// The raster for the default level choice: [`RasterConfig::for_level_choice`].
    fn default() -> Self {
        Self::for_level_choice(&LodConfig::default())
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

    /// More slot buffers than one draw can bind.
    #[error("{buffers} slot buffers: one draw binds at most {limit} beside its own")]
    TooManySlotBuffers {
        /// The slot buffers.
        buffers: u32,
        /// The most the device allows.
        limit: u32,
    },
    /// A draw's slot buffers hold more of a class than the layout the raster was built for.
    #[error("{buffers} slot buffers of size class {class}, past the {most} the raster binds")]
    PastLayout {
        /// The size class.
        class: u8,
        /// Its buffers in the draw.
        buffers: usize,
        /// The most the layout has.
        most: u16,
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

/// The raster's buffers [`PLACE_WGSL`] reads beside the tile tree and the slots, as the last
/// frame drawn left them: the shading pass binds them to place each vertex where it was drawn.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placing<'a> {
    /// The edge variants' remap tables.
    pub(crate) remap: &'a Buffer,
    /// Per node, [`GpuMorphState`].
    pub(crate) stamps: &'a Buffer,
    /// Per node, its [`crate::lod::DrawnEdges`].
    pub(crate) drawn_edges: &'a Buffer,
    /// [`GpuRasterParams`].
    pub(crate) params: &'a Buffer,
    /// The frame's [`GpuView`].
    pub(crate) view: &'a Buffer,
}

/// The render pipeline, for the most slot buffers each class can have.
#[derive(Debug)]
struct SlotPipeline {
    /// Buffers per class it binds.
    layout: SlotLayout,
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
    /// Their width and height.
    size: (u32, u32),
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
    /// Per node, its tile's [`crate::lod::DrawnEdges`] as the stamp pass last kept them, one
    /// `u32` each: for the shading pass.
    drawn_edges: Buffer,
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
    /// The render pipeline.
    pipeline: SlotPipeline,
    /// A small buffer bound for each slot buffer of the layout not made yet (nothing reads it:
    /// no slot names it).
    placeholder: Buffer,
    /// The targets, sized to the last frame's view (1 × 1 before the first).
    targets: Targets,
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
    /// The raster for a tile tree of `capacity` nodes, morphing by `config`, binding the slot
    /// buffers of `layout` (residency's, [`crate::residency::Residency::slot_layout`]).
    ///
    /// # Errors
    ///
    /// [`RasterError::Capacity`] above [`MAX_NODES`]; [`RasterError::TooManySlotBuffers`] if
    /// `layout`'s buffers don't fit [`slot_buffer_limit`]; [`RasterError::Gpu`] if a buffer or
    /// pipeline can't be made.
    pub fn new(
        gpu: &Gpu,
        capacity: u32,
        layout: SlotLayout,
        config: RasterConfig,
    ) -> Result<Self, RasterError> {
        Self::build(gpu, capacity, layout, config, &[])
    }

    /// [`Raster::new`] with text replacements in the render pipeline's shader, for the tests'
    /// bite checks, which break a rule on purpose to see a test fail.
    ///
    /// # Errors
    ///
    /// As [`Raster::new`].
    #[cfg(test)]
    pub(crate) fn patched(
        gpu: &Gpu,
        capacity: u32,
        layout: SlotLayout,
        config: RasterConfig,
        patches: &[(&str, &str)],
    ) -> Result<Self, RasterError> {
        Self::build(gpu, capacity, layout, config, patches)
    }

    /// [`Raster::new`], `patches` made in the shader.
    ///
    /// # Errors
    ///
    /// As [`Raster::new`].
    fn build(
        gpu: &Gpu,
        capacity: u32,
        slot_layout: SlotLayout,
        config: RasterConfig,
        patches: &[(&str, &str)],
    ) -> Result<Self, RasterError> {
        if capacity > MAX_NODES {
            return Err(RasterError::Capacity(capacity));
        }
        let limit = slot_buffer_limit(gpu);
        if slot_layout.total() > limit {
            return Err(RasterError::TooManySlotBuffers {
                buffers: slot_layout.total(),
                limit,
            });
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
        let drawn_edges =
            storage_buffer(gpu, "raster drawn edges", u64::from(capacity.max(1)) * 4)?;
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
                    written_entry(5),
                ],
            })
        })?;
        let stamp_source = format!("{NODE_WGSL}{CLUSTER_WGSL}{MORPH_WGSL}{STAMP_WGSL}");
        let stamp = pipeline(
            gpu,
            "raster stamp",
            &stamp_source,
            &[&stamp_layout],
            "stamp",
        )?;
        let placeholder = storage_buffer(gpu, "raster slot placeholder", VERTEX_BYTES)?;
        let source = shader_source(&slot_layout, patches);
        let slot_pipeline = SlotPipeline::new(gpu, &layout, slot_layout, &source)?;
        Ok(Self {
            capacity,
            config,
            frame: None,
            indices,
            remap,
            stamps,
            drawn_edges,
            params,
            view,
            layout,
            stamp_layout,
            stamp,
            pipeline: slot_pipeline,
            placeholder,
            targets: Targets::new(gpu, 1, 1)?,
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
            ramp: self.config.ramp.pixels(),
            count_word: 0,
            reserved: [0; 2],
        };
        let queue = gpu.queue();
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.view, 0, bytemuck::bytes_of(&frame.view.gpu()));
        let size = frame.view.size();
        if self.targets.size != (size.width(), size.height()) {
            self.targets = Targets::new(gpu, size.width(), size.height())?;
        }
        Ok(())
    }

    /// Encodes the stamp pass over `cut`, the frame's cut (after the level choice's first
    /// phase, before either draw), reading `tree`; the pass writes `timestamps` if given
    /// ([`crate::timing`]).
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
        timestamps: Option<ComputePassTimestampWrites<'_>>,
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
                    entry(5, &self.drawn_edges),
                ],
            })
        })?;
        let entries = u32::try_from(cut.entries.size() / 16).unwrap_or(u32::MAX);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("raster stamp"),
            timestamp_writes: timestamps,
        });
        pass.set_pipeline(&self.stamp);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(entries.min(self.capacity).div_ceil(GROUP).max(1), 1, 1);
        Ok(())
    }

    /// Encodes one phase's draw: its clusters, one instance each, into the visibility and depth
    /// buffers, cleared first for [`Phase::First`]; the pass writes `timestamps` if given
    /// ([`crate::timing`]).
    ///
    /// # Errors
    ///
    /// [`RasterError::NoFrame`] before [`Raster::begin_frame`]; [`RasterError::PastLayout`] if
    /// `draw`'s slot buffers have more of a class than the raster was built for;
    /// [`RasterError::Gpu`] if a bind group can't be made.
    pub fn encode_draw(
        &self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        draw: &ClusterDraw<'_>,
        phase: Phase,
        timestamps: Option<RenderPassTimestampWrites<'_>>,
    ) -> Result<(), RasterError> {
        if self.frame.is_none() {
            return Err(RasterError::NoFrame);
        }
        let targets = &self.targets;
        let slot_pipeline = &self.pipeline;
        let (group, slot_group) = self.groups(gpu, draw)?;
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
            timestamp_writes: timestamps,
            ..RenderPassDescriptor::default()
        });
        pass.set_pipeline(&slot_pipeline.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &slot_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), IndexFormat::Uint16);
        pass.draw_indexed_indirect(draw.args.0, draw.args.1);
        Ok(())
    }

    /// Group 0 and group 1 of a draw: group 1 binds every buffer of the layout, the
    /// placeholder for each not made yet.
    ///
    /// # Errors
    ///
    /// [`RasterError::PastLayout`]; as [`Gpu::scoped`].
    fn groups(
        &self,
        gpu: &Gpu,
        draw: &ClusterDraw<'_>,
    ) -> Result<(BindGroup, BindGroup), RasterError> {
        let device = gpu.device();
        let slot_group = slot_group(
            gpu,
            &self.pipeline.slot_layout,
            self.pipeline.layout,
            draw.slots,
            &self.placeholder,
        )?;
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
        Ok((group, slot_group))
    }

    /// What the level choice reads to cone-cull only tiles whose morph is settled: the stamps and
    /// the frame's settings ([`crate::lod::MainInputs::morph`]).
    #[must_use]
    pub const fn morph_inputs(&self) -> MorphInputs<'_> {
        MorphInputs {
            stamps: &self.stamps,
            params: &self.params,
        }
    }

    /// What the shading pass reads of the last frame drawn to rebuild its triangles: the remap
    /// tables, the stamps, the drawn edges, the frame's settings and its view.
    pub(crate) const fn placing(&self) -> Placing<'_> {
        Placing {
            remap: &self.remap,
            stamps: &self.stamps,
            drawn_edges: &self.drawn_edges,
            params: &self.params,
            view: &self.view,
        }
    }

    /// The visibility buffer: [`VISIBILITY_FORMAT`], the last begun frame's size (1 × 1 before
    /// the first).
    #[must_use]
    pub const fn visibility(&self) -> &Texture {
        &self.targets.visibility
    }

    /// The depth buffer: [`DEPTH_FORMAT`], the size of [`Raster::visibility`].
    #[must_use]
    pub const fn depth(&self) -> &Texture {
        &self.targets.depth
    }

    /// A view of the depth buffer, for the depth pyramid ([`crate::cull::PyramidPass::build`]).
    #[must_use]
    pub const fn depth_view(&self) -> &TextureView {
        &self.targets.depth_view
    }

    /// The visibility buffer's pixels, row by row from the top, once everything submitted so far
    /// has finished. It blocks, so it is for tests and tools ([`Gpu::read_buffer`]).
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer`], and as [`Gpu::scoped`] for the copy out of the texture.
    pub fn read_visibility(&self, gpu: &Gpu) -> Result<Vec<u32>, GpuError> {
        let texture = self.visibility();
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let device = gpu.device();
        let buffer = gpu.scoped("visibility readback", || {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visibility readback"),
                size: u64::from(row) * u64::from(height),
                usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visibility readback"),
            });
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
                    },
                },
                texture.size(),
            );
            gpu.queue().submit([encoder.finish()]);
            buffer
        })?;
        let padded: Vec<u32> = gpu.read_buffer(&buffer)?;
        let stride = usize::try_from(row / 4).unwrap_or(usize::MAX);
        let width = usize::try_from(width).unwrap_or(usize::MAX);
        let rows = padded
            .chunks(stride)
            .flat_map(|pixels| pixels.iter().take(width).copied());
        Ok(rows.collect())
    }
}

impl SlotPipeline {
    /// The pipeline binding `slots`' buffers, group 0 on `layout`, from `source`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new(
        gpu: &Gpu,
        layout: &BindGroupLayout,
        slots: SlotLayout,
        source: &str,
    ) -> Result<Self, GpuError> {
        let device = gpu.device();
        let bindings = slots.total();
        let entries = (0..bindings).map(read_entry);
        let entries: Vec<BindGroupLayoutEntry> = entries.collect();
        let slot_layout = gpu.scoped("raster slot layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("raster slots"),
                entries: &entries,
            })
        })?;
        let pipeline = gpu.scoped("raster pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("raster"),
                source: ShaderSource::Wgsl(source.into()),
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
            layout: slots,
            slot_layout,
            pipeline,
            #[cfg(test)]
            source: source.to_owned(),
        })
    }
}

/// The bind group of `slots` for a pass whose slot bindings (`bind_layout`, generated by
/// [`slot_source`]) are those of `layout`: every buffer of the layout, `placeholder` for each not
/// made yet.
///
/// # Errors
///
/// [`RasterError::PastLayout`] if `slots` hold more buffers of a class than `layout`; as
/// [`Gpu::scoped`].
pub(crate) fn slot_group(
    gpu: &Gpu,
    bind_layout: &BindGroupLayout,
    layout: SlotLayout,
    slots: &SlotBuffers,
    placeholder: &Buffer,
) -> Result<BindGroup, RasterError> {
    let classes = slots.classes();
    let every = (0..CLASSES).filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new));
    for class in every {
        let buffers = classes.get(usize::from(class.get())).map_or(0, Vec::len);
        let most = layout.buffers(class);
        if buffers > usize::from(most) {
            return Err(RasterError::PastLayout {
                class: class.get(),
                buffers,
                most,
            });
        }
    }
    let bound = layout
        .per_class()
        .iter()
        .enumerate()
        .flat_map(|(class, &most)| {
            let made = classes.get(class).map_or(&[][..], Vec::as_slice);
            (0..usize::from(most)).map(move |b| made.get(b))
        });
    let entries = bound
        .zip(0..)
        .map(|(buffer, binding)| entry(binding, buffer.unwrap_or(placeholder)));
    let entries: Vec<BindGroupEntry<'_>> = entries.collect();
    Ok(gpu.scoped("slot bind group", || {
        gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some("slots"),
            layout: bind_layout,
            entries: &entries,
        })
    })?)
}

/// The render pipeline's shader for `layout`'s slot buffers, `patches` made (only the tests
/// make any).
///
/// # Panics
///
/// If a patch doesn't apply.
fn shader_source(layout: &SlotLayout, patches: &[(&str, &str)]) -> String {
    let source = format!(
        "{NODE_WGSL}{VERTEX_WGSL}{CULL_WGSL}{CLUSTER_WGSL}\
         {INDICES_WGSL}{MORPH_WGSL}{}{PLACE_WGSL}{RASTER_WGSL}",
        slot_source(layout)
    );
    patches.iter().fold(source, |source, (from, to)| {
        assert!(source.contains(from), "the patch {from:?} applies");
        source.replace(from, to)
    })
}

/// The generated WGSL of the slot bindings, for `layout`'s buffers of each class.
///
/// `slots_<class>_<buffer>` at group 1, one binding each in class then buffer order, and
/// `slot_vertex(size_class, slot_buffer, i)` reading vertex `i` of that buffer (a zeroed vertex
/// for a buffer not bound).
pub(crate) fn slot_source(layout: &SlotLayout) -> String {
    let buffers = layout
        .per_class()
        .iter()
        .enumerate()
        .flat_map(|(class, &count)| (0..usize::from(count)).map(move |buffer| (class, buffer)));
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
    /// Visibility and depth images `width` × `height`, both positive.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Self, GpuError> {
        let device = gpu.device();
        let extent = Extent3d {
            width,
            height,
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
                size: (width, height),
                visibility_view: visibility.create_view(&TextureViewDescriptor::default()),
                visibility,
                depth_view: depth.create_view(&TextureViewDescriptor::default()),
                depth,
            }
        })
    }
}
