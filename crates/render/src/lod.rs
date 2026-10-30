//! Level choice: which resident tiles a frame draws, and which it wants and finds missing.
//!
//! The choice runs on the GPU over the resident tile tree ([`crate::tree`], plan decision 3),
//! one thread per node, in `lod.wgsl` (WGSL: no source shared with the CPU, so rust-gpu buys
//! nothing here). Its rules are the spec's ("Level choice and seams"):
//!
//! - **Screen error.** A tile is refined while its `e` projects above 1 px or its longest quad edge
//!   above 4 px, and once refined is coarsened only when both drop below 0.5 px and 2 px
//!   (hysteresis: the memory of last frame's choice is kept per node, checked against the node's
//!   tile so a reused node starts afresh). A length's pixels are bounded over the tile's bounding
//!   sphere grown by `e` (the finest surface lies within `e` of what the tile draws): `l · f · d /
//!   z²` with `d` the sphere's farthest distance from the eye and `z` its nearest view depth,
//!   floored at the near plane, which bounds the projection's stretch in any direction and anywhere
//!   in the image (off the axis it is up to `(d / z)²` more than `l · f / d`), so the error is
//!   never underestimated (checked against measured segments, `cull/tests/truth.rs`). A tile out of
//!   view (frustum, or facing away, both with the sphere grown by `e`) isn't refined. A display
//!   tile faces away when the cone of the finest normals under it does
//!   ([`crate::tree::GpuNode::finest_cone`], spec "Normal cone"), not its own triangles: a coarse
//!   tile at a silhouette can face away while finer detail under it faces the camera. Drawing culls
//!   every tile by its own triangles, so a display tile is skipped for refining only when drawing
//!   culls it too: one whose finest normals face away but whose own triangles face the eye would be
//!   drawn coarse, and is refined. A finest tile, which has nothing to refine into, is tested by
//!   its own triangles' cone.
//! - **Completeness.** A tile is refined only when all its children are resident, so the cut covers
//!   every face exactly once; tiles under a missing one are unreachable and ignored. The pinned
//!   floor means level 0 is always there.
//! - **Budget.** The tier's triangle budget caps the refinements: every possible refinement's added
//!   triangles go into a histogram by its screen error (capped by its ancestors', so the refined
//!   set stays a tree), and refinement stops at the error where the budget runs out. Tiles left
//!   over 1 px are counted and reported ([`LodStats::over_pixel`]); higher errors are refined
//!   first.
//! - **Neighbours at most one level apart.** A fixed number of relaxation passes
//!   ([`LodConfig::with_relax_passes`]) raises any tile in the cut that touches an area two levels
//!   finer. Then a sweep, coarse to fine, keeps a refinement only where every edge neighbour is
//!   reached at the same level, which makes the invariant hold whatever the passes left: an area
//!   the passes didn't settle draws a level coarser, counted as the shortfall
//!   ([`LodStats::shortfall`]). The rule is for seams between drawn tiles: a seam where either tile
//!   lies outside the frustum (by the sphere of what it draws) is off-screen and exempt
//!   ([`LodConfig::with_offscreen_seams`]); a tile culled by its cone is not, its seams may be on
//!   the silhouette.
//! - **Requests.** A tile in the cut that wants refining and lacks children requests them, with its
//!   screen error, in residency's request buffer ([`crate::residency::REQUESTS_WGSL`]).
//!
//! # Outputs
//!
//! - **For the raster (Task 4):** two lists of visible clusters ([`Lod::clusters`]: 8 × 8-quad
//!   clusters, [`GpuCluster`], [`CLUSTER_WGSL`]) and their indexed indirect draw arguments
//!   ([`Lod::draw_args`]: one draw per list, [`CLUSTER_INDICES`] indices an instance, an instance
//!   per cluster). The first list holds the tiles that passed the first-phase occlusion test; the
//!   second, written by [`Lod::encode_post`] after the first is drawn and this frame's depth
//!   pyramid is built, the tiles that failed it and pass against the new pyramid.
//! - **For residency:** the cut and the tiles drawn, read back as a [`LodOutput`] whose
//!   [`LodOutput::frame_use`] is the [`FrameUse`] residency takes. **Latency:** the readback is
//!   copied at the end of frame `F` and mapped when the GPU finishes it, so [`Lod::collect`] (or
//!   the [`CutReader`] moved to the thread that polls the device) has it, tagged `F`
//!   ([`LodOutput::frame`]), at the start of frame `F + 1` at best, `F + 2` when the GPU runs a
//!   frame behind; residency then protects a cut a frame or two old, with every ancestor of it (the
//!   refined set). What that covers: the GPU never reads a slot that was freed, since residency
//!   changes reach the tree in the same submission as the tile data they point at and each frame's
//!   level choice sees the tree as it is; and the cut never has a hole, since a tile whose ancestor
//!   left is unreachable and its ancestor's ancestor is drawn instead (the floor is never evicted).
//!   What it doesn't: a tile the GPU started using in the last frame or two isn't protected yet, so
//!   under budget pressure it can be evicted and drawn coarser for a frame, then requested again.
//!
//! # A frame
//!
//! On the render thread, after the frame's uploads are encoded:
//! [`Lod::encode_bounds`] (the tiles whose data changed), [`Lod::encode_main`], the raster's
//! stamp pass over the cut ([`crate::raster::Raster::encode_stamp`], [`Lod::cut_list`]: when
//! each tile came into the cut, for the geomorph), the first raster, the depth pyramid
//! ([`crate::cull::PyramidPass`]), [`Lod::encode_post`], the second raster, the pyramid again,
//! [`Lod::encode_readback`]; submit; then [`CutReadback::submitted`]. The owner thread reads the
//! output a frame or two later through the [`CutReader`] ([`Lod::take_reader`]), where it polls
//! the device, beside residency's requests of the same frame. [`crate::raster::FramePasses`]
//! encodes it all in this order.
//!
//! **The pyramids (decided for, and built in, Task 4):** the renderer keeps two, `current` and
//! `previous`.
//! After the first raster, `current` is built from the depth and [`Lod::encode_post`] tests
//! against it; after the second raster, `current` is **built again**, from the frame's whole
//! depth, and the two swap, so the next frame's first phase tests against everything this frame
//! drew, seen from this frame's view. The extra build is a few dispatches; testing against the
//! first raster's depth alone would leave the next frame's first phase drawing tiles the second
//! raster hid. After a resize there is no previous pyramid (pass `None`).
//!
//! **Readback cost:** each frame copies the counts (≈ 1.4 KB) and the whole cut list, 16 bytes
//! a node of capacity: 3.5 MB a frame at the benchmark's 218,400 nodes, about 1 MB at a
//! desktop budget's ~60k. On a tablet (unified memory, but a slower copy) that is worth a
//! readback only every few frames, or of the cut's changes; residency tolerates a stale cut.

use std::collections::BTreeSet;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use geom::{Level, TileKey};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferSize, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipeline, MapMode, QUERY_SIZE, QuerySet, QuerySetDescriptor,
    QueryType, ShaderStages, TextureSampleType, TextureViewDimension,
};

use crate::camera::{GpuView, View};
use crate::cull::{
    BoundsPass, BoundsUpdate, CULL_WGSL, CullError, DepthPyramid, MAX_GROUPS, SlotBuffers,
    UNIFORM_STRIDE, entry, pipeline, storage_buffer, uniform_buffer,
};
use crate::gpu::{Gpu, GpuError};
use crate::residency::{FrameUse, REQUESTS_WGSL, Tier};
use crate::tree::{NODE_BYTES, NODE_WGSL, Place};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The level choice's own source, after [`NODE_WGSL`], [`REQUESTS_WGSL`], [`CULL_WGSL`] and
/// [`CLUSTER_WGSL`].
const LOD_WGSL: &str = include_str!("lod.wgsl");

/// The WGSL declaration of [`GpuCluster`], for the raster that reads the cluster lists.
pub const CLUSTER_WGSL: &str = r"
// One visible cluster (render::lod::GpuCluster): 8 bytes.
struct Cluster {
    // The tile's node in the tile tree: the cluster id's tile part, stable while resident.
    node: u32,
    // Bits 0..8: the cluster's column in the tile, 8..16 its row, 16..24 its quads a side
    // (8, or fewer for a tile under 8 quads), 24..28 its edge variant: bit k set where the
    // cluster lies on the tile's edge k and the neighbour across it is a level coarser; 28..32
    // bit 28 + k set where it lies on edge k and the neighbour across it is refined (finer).
    cell: u32,
}
";

/// Indices one cluster draws: 8 × 8 quads, two triangles each. A cluster of a smaller tile, or
/// an edge variant with fewer triangles, pads with degenerate triangles.
pub const CLUSTER_INDICES: u32 = 384;

/// Triangles the triangle budget counts in a unit: a full cluster's.
///
/// Refinements at level 3 and finer are whole units; coarser ones round up to one, so the budget is
/// respected with up to a unit of slack a coarse refinement. Counting in units keeps the GPU's sums
/// from wrapping a `u32`.
pub const BUDGET_UNIT: u32 = 128;

/// The most relaxation passes a frame may run.
pub const MAX_RELAX_PASSES: u32 = 16;

/// Histogram buckets of the budget (see `lod.wgsl`).
const BUCKETS: usize = 256;

/// Threads per workgroup of the per-node passes.
const GROUP: u32 = 64;

/// Readback buffers in flight at most.
const READBACKS: usize = 3;

/// Step uniforms: levels for the sweep and passes for the relaxation, one per dynamic offset.
const STEPS: u32 = 32;

/// Timed stages: bounds, choose, relax, sweep, emit, post.
const STAGES: u32 = 6;

/// One visible cluster as the raster reads it ([`CLUSTER_WGSL`]'s `Cluster`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuCluster {
    /// The tile's node in the tile tree.
    pub node: u32,
    /// Column (bits 0–8), row (8–16), quads a side (16–24), edge variant (24–28) and finer
    /// edges (28–32).
    pub cell: u32,
}

impl GpuCluster {
    /// The cluster's column and row in its tile.
    #[must_use]
    pub const fn position(self) -> (u32, u32) {
        (self.cell & 0xFF, (self.cell >> 8) & 0xFF)
    }

    /// The cluster's quads a side: 8, or fewer for a tile under 8 quads a side.
    #[must_use]
    pub const fn quads(self) -> u32 {
        (self.cell >> 16) & 0xFF
    }

    /// The edge variant: bit `k` set where the cluster lies on its tile's edge `k` and the
    /// neighbour across it is a level coarser.
    #[must_use]
    pub const fn variant(self) -> u32 {
        (self.cell >> 24) & 0xF
    }

    /// The finer edges: bit `k` set where the cluster lies on its tile's edge `k` and the
    /// neighbour across it is refined, so drawn finer. The raster draws that edge's odd vertices
    /// where they are, unmorphed, since the finer side draws them as its even vertices.
    #[must_use]
    pub const fn finer(self) -> u32 {
        self.cell >> 28
    }
}

/// One tile of the cut as the GPU writes it: WGSL's `CutEntry`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuCutEntry {
    /// Its face.
    pub(crate) face: u32,
    /// Its place ([`Place::packed`]).
    pub(crate) place: u32,
    /// Its node.
    pub(crate) node: u32,
    /// [`CUT_DRAWN`] once drawn.
    pub(crate) flags: u32,
}

/// [`GpuCutEntry::flags`]: drawn by either phase.
const CUT_DRAWN: u32 = 1;

/// The bytes of one [`GpuCutEntry`].
const CUT_BYTES: u64 = size_of::<GpuCutEntry>() as u64;

/// Word offset of [`GpuFrame::cut_count`] in the frame buffer.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a word offset early in a struct of a few hundred words"
)]
const CUT_COUNT_WORD: u32 = (std::mem::offset_of!(GpuFrame, cut_count) / 4) as u32;

/// The cut as the GPU holds it after [`Lod::encode_main`], for the passes after the level
/// choice in the same frame (the raster's stamp pass, [`crate::raster`]).
///
/// The entries are `CutEntry`s of 16 bytes (face, place packed as [`Place::packed`], node,
/// flags), the first `counts[count_word]` of them written.
#[derive(Debug, Clone, Copy)]
pub struct CutList<'a> {
    /// The entries.
    pub entries: &'a Buffer,
    /// The buffer holding the count, as 32-bit words.
    pub counts: &'a Buffer,
    /// The count's word in `counts`.
    pub count_word: u32,
}

/// Per node, WGSL's `NodeWork`: last frame's choice and this frame's state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct NodeWork {
    /// The face `memory` is about.
    face: u32,
    /// The place `memory` is about.
    place: u32,
    /// Last frame's choice.
    memory: u32,
    /// This frame's state.
    flags: u32,
    /// The projected `e`, pixels.
    priority: f32,
    /// The priority capped by the ancestors'.
    effective: f32,
    /// Its entry in the cut.
    cut_index: u32,
    /// Zero.
    reserved: u32,
}

/// Indexed indirect draw arguments, WGSL's `DrawArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuDrawArgs {
    /// Indices per instance.
    index_count: u32,
    /// Instances: clusters.
    instance_count: u32,
    /// Zero.
    first_index: u32,
    /// Zero.
    base_vertex: i32,
    /// Zero.
    first_instance: u32,
}

/// The frame's results, WGSL's `LodFrame`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuFrame {
    /// The first phase's draw.
    main: GpuDrawArgs,
    /// The second phase's draw.
    post: GpuDrawArgs,
    /// The second phase's dispatch.
    post_groups: [u32; 3],
    /// The lowest histogram bucket refined.
    threshold: u32,
    /// Triangles of the level-0 tiles.
    base: u32,
    /// Budget units ([`BUDGET_UNIT`] triangles) the refined choice is charged.
    triangles: u32,
    /// Tiles in the cut.
    cut_count: u32,
    /// Tiles to test again.
    retry_count: u32,
    /// Triangles of the cut.
    cut_triangles: u32,
    /// Triangles drawn.
    drawn_triangles: u32,
    /// Tiles drawn in the first phase.
    drawn_main: u32,
    /// Tiles drawn in the second.
    drawn_post: u32,
    /// Tiles in the cut, in view, over 1 px.
    over_pixel: u32,
    /// ... waiting for children.
    waiting: u32,
    /// ... held back by the budget.
    budget_limited: u32,
    /// ... held back by the neighbour rule.
    held_by_neighbours: u32,
    /// Tiles requested.
    requested: u32,
    /// Culled by the frustum.
    culled_frustum: u32,
    /// Culled by the normal cone.
    culled_cone: u32,
    /// Culled by the first phase's occlusion test.
    culled_occlusion: u32,
    /// Culled by the second's.
    culled_occlusion_post: u32,
    /// Clusters past the lists' end.
    cluster_overflow: u32,
    /// Tiles past the retry list's end.
    retry_overflow: u32,
    /// Tiles past the cut list's end.
    cut_overflow: u32,
    /// Refinements the sweep took back.
    shortfall: u32,
    /// Raises the budget refused.
    refused: u32,
    /// Raises missing children.
    stuck: u32,
    /// Zero.
    reserved: u32,
    /// Tiles raised per relaxation pass.
    raises: [u32; MAX_RELAX_PASSES as usize],
    /// The budget's histogram.
    histogram: [u32; BUCKETS],
}

/// The bytes of [`GpuFrame`].
const FRAME_BYTES: u64 = size_of::<GpuFrame>() as u64;

/// Byte offset of [`GpuFrame::post`] in the frame buffer: the second draw's arguments.
const POST_ARGS: u64 = size_of::<GpuDrawArgs>() as u64;

/// Byte offset of [`GpuFrame::post_groups`]: the second phase's dispatch.
const POST_GROUPS: u64 = 2 * POST_ARGS;

/// The dispatch-arguments entry the second phase's dispatch is copied to, past the relaxation
/// passes'.
const POST_SLOT: u32 = MAX_RELAX_PASSES + 1;

/// The frame's settings, WGSL's `LodParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct GpuParams {
    /// Nodes.
    capacity: u32,
    /// The finest level.
    finest: u32,
    /// The triangle budget.
    budget: u32,
    /// Relaxation passes.
    passes: u32,
    /// Clusters a list holds.
    cluster_capacity: u32,
    /// Entries the cut holds.
    cut_capacity: u32,
    /// Entries the retry list holds.
    retry_capacity: u32,
    /// 1 with a previous frame's pyramid.
    previous: u32,
    /// Refine above this projected `e`.
    refine_error: f32,
    /// ... or this projected quad edge.
    refine_quad: f32,
    /// Coarsen below this projected `e`.
    coarsen_error: f32,
    /// ... and this projected quad edge.
    coarsen_quad: f32,
    /// [`OPTION_OFFSCREEN_SEAMS`] when set.
    options: u32,
    /// Zero.
    reserved: [u32; 3],
}

/// [`GpuParams::options`]: seams of tiles outside the frustum are exempt from the neighbour
/// rule.
const OPTION_OFFSCREEN_SEAMS: u32 = 1;

/// A pass index, WGSL's `Step`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuStep {
    /// The level or pass.
    index: u32,
    /// Zero.
    reserved: [u32; 3],
}

/// The relaxation gate (`gate.wgsl`), after the word offset of [`GpuFrame::raises`].
const GATE_WGSL: &str = include_str!("gate.wgsl");

/// Word offset of [`GpuFrame::raises`] in the frame buffer.
const RAISES_WORD: usize = std::mem::offset_of!(GpuFrame, raises) / 4;

/// Bytes of one dispatch's indirect arguments: three `u32` workgroup counts.
const DISPATCH_BYTES: u64 = 12;

/// The screen-error thresholds, pixels (spec "Level choice and seams").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Refine while `e` projects above this.
    pub refine_error: f32,
    /// ... or the longest quad edge above this.
    pub refine_quad: f32,
    /// Coarsen only once `e` projects below this.
    pub coarsen_error: f32,
    /// ... and the longest quad edge below this.
    pub coarsen_quad: f32,
}

impl Thresholds {
    /// The spec's: refine above 1 px of error or 4 px of quad edge, coarsen below 0.5 px and
    /// 2 px.
    pub const SPEC: Self = Self {
        refine_error: 1.0,
        refine_quad: 4.0,
        coarsen_error: 0.5,
        coarsen_quad: 2.0,
    };
}

/// How the level choice is set up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodConfig {
    /// The most triangles the cut may have.
    triangle_budget: u32,
    /// Relaxation passes a frame.
    relax_passes: u32,
    /// The screen-error thresholds.
    thresholds: Thresholds,
    /// Whether seams of tiles outside the frustum are exempt from the neighbour rule.
    offscreen_seams: bool,
}

/// The relaxation passes a frame may run by default: all of them.
///
/// The benchmark orbit's tree shape (the ignored `lod::tests::orbit` measurement, RTX 4090,
/// 2026-10-28) needed up to 13 in every frame, one per level between the front's finest tiles and
/// the culled back faces at level 0, and settled within them (no shortfall). Passes after the
/// settling are gated off on the GPU, so only those needed cost anything.
pub const RELAX_PASSES: u32 = MAX_RELAX_PASSES;

impl LodConfig {
    /// The configuration of `tier`. The triangle budgets are provisional until measured (plan
    /// Task 7): 32M on desktop, 8M on a high-end tablet.
    #[must_use]
    pub const fn for_tier(tier: Tier) -> Self {
        let triangle_budget = match tier {
            Tier::Desktop => 32 << 20,
            Tier::MobileHigh => 8 << 20,
        };
        Self {
            triangle_budget,
            relax_passes: RELAX_PASSES,
            thresholds: Thresholds::SPEC,
            offscreen_seams: true,
        }
    }

    /// This configuration with a budget of `triangles`.
    #[must_use]
    pub const fn with_triangle_budget(self, triangles: u32) -> Self {
        Self {
            triangle_budget: triangles,
            ..self
        }
    }

    /// This configuration running `passes` relaxation passes a frame.
    ///
    /// # Errors
    ///
    /// [`LodError::RelaxPasses`] above [`MAX_RELAX_PASSES`].
    pub const fn with_relax_passes(self, passes: u32) -> Result<Self, LodError> {
        if passes > MAX_RELAX_PASSES {
            return Err(LodError::RelaxPasses(passes));
        }
        Ok(Self {
            relax_passes: passes,
            ..self
        })
    }

    /// This configuration exempting seams of tiles outside the frustum from the neighbour rule
    /// (`true`, the default) or not.
    ///
    /// The rule matters only for seams between drawn tiles, and a tile outside the frustum by
    /// its conservative sphere has every seam off-screen, so exempting them saves the
    /// relaxation raising areas nobody sees. Tiles culled by their cone aren't exempt: their
    /// seams may lie on the silhouette.
    #[must_use]
    pub const fn with_offscreen_seams(self, exempt: bool) -> Self {
        Self {
            offscreen_seams: exempt,
            ..self
        }
    }

    /// This configuration with `thresholds`.
    #[must_use]
    pub const fn with_thresholds(self, thresholds: Thresholds) -> Self {
        Self { thresholds, ..self }
    }

    /// The most triangles the cut may have.
    #[must_use]
    pub const fn triangle_budget(&self) -> u32 {
        self.triangle_budget
    }

    /// The screen-error thresholds: the raster's geomorph holds the drawn error to the refine
    /// threshold ([`crate::raster::RasterConfig::for_level_choice`]).
    #[must_use]
    pub const fn thresholds(&self) -> &Thresholds {
        &self.thresholds
    }

    /// Relaxation passes a frame.
    #[must_use]
    pub const fn relax_passes(&self) -> u32 {
        self.relax_passes
    }
}

impl Default for LodConfig {
    fn default() -> Self {
        Self::for_tier(Tier::Desktop)
    }
}

/// Why the level choice failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LodError {
    /// The GPU failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The bounds pass failed.
    #[error(transparent)]
    Cull(#[from] CullError),
    /// More relaxation passes than a frame may run.
    #[error("{0} relaxation passes: at most {MAX_RELAX_PASSES}")]
    RelaxPasses(u32),
    /// The tile tree is larger than one dispatch covers.
    #[error("{0} nodes: one dispatch covers at most {MAX_NODES}")]
    Capacity(u32),
    /// The tile tree's buffer holds fewer nodes than the level choice covers.
    #[error("a tile tree buffer of {bytes} bytes can't hold the {nodes} nodes it was made for")]
    TreeTooSmall {
        /// The buffer's size.
        bytes: u64,
        /// The nodes the level choice covers.
        nodes: u32,
    },
    /// The previous frame's pyramid wasn't built for depth images of its view's size.
    #[error("a depth pyramid for {pyramid:?} images against a {view:?} view")]
    PyramidSize {
        /// The depth image size the pyramid is for.
        pyramid: (u32, u32),
        /// The view's size.
        view: (u32, u32),
    },
}

/// The most nodes the level choice covers: one dispatch of 64 threads a workgroup.
pub const MAX_NODES: u32 = MAX_GROUPS * GROUP;

/// What a frame's level choice counted, read back with its cut.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LodStats {
    /// Tiles in the cut.
    pub cut: u32,
    /// Triangles of the cut.
    pub cut_triangles: u32,
    /// Triangles the refinements were charged, within the budget: in [`BUDGET_UNIT`]s, so an
    /// upper bound on [`LodStats::cut_triangles`].
    pub refined_triangles: u32,
    /// Triangles drawn, both phases.
    pub drawn_triangles: u32,
    /// Tiles drawn in the first phase and in the second.
    pub drawn: (u32, u32),
    /// Clusters in the first list and the second.
    pub clusters: (u32, u32),
    /// Tiles in the cut, in view, whose `e` projects over 1 px: the over-budget count, 0 on
    /// desktop in the benchmark orbit by the spec's target.
    pub over_pixel: u32,
    /// ... of which their children aren't resident yet (requested).
    pub waiting: u32,
    /// ... of which the triangle budget held them back.
    pub budget_limited: u32,
    /// ... of which the neighbour rule held them back.
    pub held_by_neighbours: u32,
    /// Tiles requested (not counting those the request buffer had no room for).
    pub requested: u32,
    /// Tiles in the cut culled by the frustum, the normal cone, the first phase's occlusion
    /// test (then tested again), and the second's.
    pub culled: (u32, u32, u32, u32),
    /// Refinements the sweep took back because the relaxation hadn't settled: the area drew a
    /// level coarser.
    pub shortfall: u32,
    /// Tiles a relaxation pass wanted to raise and the budget refused.
    pub refused: u32,
    /// Tiles a relaxation pass wanted to raise and whose children weren't resident.
    pub stuck: u32,
    /// Tiles each relaxation pass raised, for the passes run.
    pub raises: Vec<u32>,
    /// Clusters, retries and cut entries past their lists' ends: zero unless a list is too
    /// small.
    pub overflow: (u32, u32, u32),
}

impl LodStats {
    /// The relaxation passes that raised anything: how many the frame needed, if it settled
    /// within those run (the last pass run raising something means it may not have).
    #[must_use]
    pub fn passes_used(&self) -> usize {
        self.raises
            .iter()
            .rposition(|&raised| raised > 0)
            .map_or(0, |last| last + 1)
    }
}

/// A frame's cut and the tiles it drew, read back: what residency takes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LodOutput {
    /// The frame it is of: the number its readback was encoded with
    /// ([`Lod::encode_readback`]).
    pub frame: u64,
    /// The tiles of the cut.
    pub cut: BTreeSet<TileKey>,
    /// The tiles drawn, by either phase.
    pub drawn: BTreeSet<TileKey>,
    /// The cut's node indices with their places, as the GPU wrote them: for tests and tools.
    pub nodes: Vec<(u32, Place)>,
    /// Counts.
    pub stats: LodStats,
}

impl LodOutput {
    /// The cut and drawn sets as residency takes them ([`crate::residency::Residency::use_frame`]).
    #[must_use]
    pub fn frame_use(&self) -> FrameUse {
        FrameUse::new(self.cut.clone(), self.drawn.clone())
    }
}

/// The inputs of a frame's first phase.
#[derive(Debug, Clone, Copy)]
pub struct MainInputs<'a> {
    /// The tile tree's buffer ([`crate::tree::TileTree::buffer`]).
    pub tree: &'a Buffer,
    /// Residency's request buffer ([`crate::residency::Residency::request_buffer`]).
    pub requests: &'a Buffer,
    /// This frame's view.
    pub view: &'a View,
    /// The previous frame's view and depth pyramid, for the first phase's occlusion test; `None`
    /// on the first frame or after a resize (nothing is occlusion-culled in the first phase).
    pub previous: Option<(&'a View, &'a DepthPyramid)>,
    /// The project's finest level.
    pub finest: Level,
}

/// A readback buffer coming back to its reader.
#[derive(Debug)]
enum Readback {
    /// Mapped, the results of frame `1` in it.
    Mapped(Buffer, u64),
    /// Mapping failed; the buffer is gone.
    Failed,
}

/// A readback buffer going back to the level choice that made it.
#[derive(Debug)]
enum Returned {
    /// Free for another frame: read, or never submitted.
    Free(Buffer),
    /// Gone: its mapping failed.
    Lost,
}

/// A frame's readback, encoded: call [`CutReadback::submitted`] once its command buffer is
/// submitted, or drop it to give its buffer back unread.
#[derive(Debug)]
#[must_use = "a readback not marked submitted is never read"]
pub struct CutReadback {
    /// The buffer the results were copied to.
    buffer: Option<Buffer>,
    /// The frame it is of.
    frame: u64,
    /// Where it goes once mapped: to the reader.
    done: Sender<Readback>,
    /// Where an unread one goes back.
    unused: Sender<Returned>,
}

impl CutReadback {
    /// Maps the buffer, now that its frame is submitted: [`CutReader::collect`] reads it once
    /// the device is polled after the GPU finishes the frame.
    pub fn submitted(mut self) {
        if let Some(buffer) = self.buffer.take() {
            let back = buffer.clone();
            let done = self.done.clone();
            let frame = self.frame;
            buffer.map_async(MapMode::Read, .., move |result| {
                let readback = if result.is_ok() {
                    Readback::Mapped(back, frame)
                } else {
                    Readback::Failed
                };
                // A level choice that's gone has nothing to read.
                drop(done.send(readback));
            });
        }
    }
}

impl Drop for CutReadback {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // A level choice that's gone has nothing to take back.
            drop(self.unused.send(Returned::Free(buffer)));
        }
    }
}

/// Where a level choice's readbacks arrive, read on whichever thread polls the device.
///
/// The level choice holds it until [`Lod::take_reader`] moves it out: the app reads each frame's
/// cut on the owner thread, which polls the device, while the render thread keeps the level
/// choice. Read buffers go back to the level choice for later frames.
#[derive(Debug)]
pub struct CutReader {
    /// Where mapped readbacks arrive.
    mapped: Receiver<Readback>,
    /// Where read ones go back.
    back: Sender<Returned>,
}

impl CutReader {
    /// The latest frame's results read back, if a readback completed since the last call (the
    /// device must have been polled since; older completed readbacks are dropped): its cut and
    /// drawn tiles as keys of a project whose finest level is `finest`, its counts, and the frame
    /// it is of.
    ///
    /// # Errors
    ///
    /// [`LodError::Gpu`] if a mapped buffer can't be viewed.
    pub fn collect(&self, finest: Level) -> Result<Option<LodOutput>, LodError> {
        let mut latest = None;
        while let Ok(back) = self.mapped.try_recv() {
            let Readback::Mapped(buffer, frame) = back else {
                // A level choice that's gone has no count to correct.
                drop(self.back.send(Returned::Lost));
                continue;
            };
            let read = read_output(&buffer, finest, frame);
            buffer.unmap();
            // A level choice that's gone has nothing to take back.
            drop(self.back.send(Returned::Free(buffer)));
            latest = Some(read?);
        }
        Ok(latest)
    }
}

/// How long each stage took on the GPU, by timestamp queries ([`Lod::enable_timing`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LodTimings {
    /// The bounds pass.
    pub bounds: Duration,
    /// Choose, close, threshold and mark.
    pub choose: Duration,
    /// The relaxation passes.
    pub relax: Duration,
    /// The sweep.
    pub sweep: Duration,
    /// Emit and finish.
    pub emit: Duration,
    /// The second phase.
    pub post: Duration,
}

/// Timestamp queries around each stage.
#[derive(Debug)]
struct Timer {
    /// Two queries a stage.
    queries: QuerySet,
    /// Where they are resolved.
    resolved: Buffer,
    /// Where the resolved ticks are read through.
    staging: Buffer,
    /// Nanoseconds a tick.
    period: f32,
}

/// The level choice: its passes, the buffers it writes, and its readbacks.
#[derive(Debug)]
pub struct Lod {
    /// How it is set up.
    config: LodConfig,
    /// Nodes it covers.
    capacity: u32,
    /// Clusters each list holds.
    cluster_capacity: u32,
    /// The bounds and their pass.
    bounds: BoundsPass,
    /// Per node, [`NodeWork`].
    work: Buffer,
    /// [`GpuFrame`]: counts, histogram, draw and dispatch arguments.
    frame: Buffer,
    /// The cut, [`GpuCutEntry`]s.
    cut: Buffer,
    /// Node indices to test again.
    retry: Buffer,
    /// The first phase's clusters.
    clusters_main: Buffer,
    /// The second phase's clusters.
    clusters_post: Buffer,
    /// [`GpuParams`].
    params: Buffer,
    /// This frame's [`GpuView`].
    view: Buffer,
    /// The previous frame's [`GpuView`], for the first phase's occlusion test.
    previous_view: Buffer,
    /// [`GpuStep`]s, one per [`UNIFORM_STRIDE`].
    steps: Buffer,
    /// Indirect dispatch arguments: entry 0 every node, entry `k` relaxation pass `k`'s (set
    /// by the gate after pass `k - 1`), entry [`POST_SLOT`] the second phase's.
    dispatches: Buffer,
    /// The gate's bind group: the frame buffer, the dispatch arguments, the steps.
    gate_group: BindGroup,
    /// A 1 × 1 pyramid holding the far depth, bound when there is no previous one.
    empty_pyramid: DepthPyramid,
    /// Group 0's layout.
    layout: BindGroupLayout,
    /// Group 1's layout.
    phase_layout: BindGroupLayout,
    /// The passes, by entry point.
    pipelines: Pipelines,
    /// Group 0 of the frame being encoded, kept for [`Lod::encode_post`].
    group: Option<BindGroup>,
    /// Readback buffers free.
    free: Vec<Buffer>,
    /// Readback buffers made.
    made: usize,
    /// Given to each readback, to send its buffer to the reader once mapped.
    done: Sender<Readback>,
    /// Where mapped readbacks arrive, until [`Lod::take_reader`] moves it out.
    reader: Option<CutReader>,
    /// Given to each readback and to the reader, to send buffers back.
    returns: Sender<Returned>,
    /// Where buffers come back: read, unread, or lost.
    returned: Receiver<Returned>,
    /// Timestamp queries, once [`Lod::enable_timing`] is called on a device that has them.
    timer: Option<Timer>,
}

/// The level choice's compute pipelines.
#[derive(Debug)]
struct Pipelines {
    /// `begin`.
    begin: ComputePipeline,
    /// `choose`.
    choose: ComputePipeline,
    /// `close`.
    close: ComputePipeline,
    /// `threshold`.
    threshold: ComputePipeline,
    /// `mark`.
    mark: ComputePipeline,
    /// `relax`.
    relax: ComputePipeline,
    /// `sweep`.
    sweep: ComputePipeline,
    /// `emit`.
    emit: ComputePipeline,
    /// `finish`.
    finish: ComputePipeline,
    /// `post`.
    post: ComputePipeline,
    /// `post_finish`.
    post_finish: ComputePipeline,
    /// `gate`, with its own layout.
    gate: ComputePipeline,
}

/// A layout entry for a buffer binding.
const fn buffer_entry(binding: u32, ty: BufferBindingType, dynamic: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A read-only storage binding.
const READ: BufferBindingType = BufferBindingType::Storage { read_only: true };
/// A read-write storage binding.
const WRITE: BufferBindingType = BufferBindingType::Storage { read_only: false };

impl Lod {
    /// The level choice for a tile tree of `capacity` nodes, set up by `config`.
    ///
    /// # Errors
    ///
    /// [`LodError::Capacity`] above [`MAX_NODES`]; [`LodError::Gpu`] or [`LodError::Cull`] if a
    /// buffer or pipeline can't be made.
    pub fn new(gpu: &Gpu, capacity: u32, config: LodConfig) -> Result<Self, LodError> {
        if capacity > MAX_NODES {
            return Err(LodError::Capacity(capacity));
        }
        let nodes = u64::from(capacity.max(1));
        // Clusters of tiles with at least 8 quads a side carry 128 triangles each; smaller tiles
        // one cluster each, at most one per node.
        let cluster_capacity = (config.triangle_budget / 128).saturating_add(capacity.max(1));
        let device = gpu.device();
        let bounds = BoundsPass::new(gpu, capacity)?;
        let work = storage_buffer(gpu, "lod work", nodes * size_of::<NodeWork>() as u64)?;
        let frame = gpu.scoped("lod frame", || {
            device.create_buffer(&BufferDescriptor {
                label: Some("lod frame"),
                size: FRAME_BYTES,
                usage: BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })?;
        let cut = storage_buffer(gpu, "lod cut", nodes * CUT_BYTES)?;
        let retry = storage_buffer(gpu, "lod retry", nodes * 4)?;
        let cluster_bytes = u64::from(cluster_capacity) * size_of::<GpuCluster>() as u64;
        let clusters_main = storage_buffer(gpu, "lod clusters main", cluster_bytes)?;
        let clusters_post = storage_buffer(gpu, "lod clusters post", cluster_bytes)?;
        let params = uniform_buffer(gpu, "lod params", size_of::<GpuParams>() as u64)?;
        let view = uniform_buffer(gpu, "lod view", size_of::<GpuView>() as u64)?;
        let previous_view = uniform_buffer(gpu, "lod previous view", size_of::<GpuView>() as u64)?;
        let steps = steps(gpu)?;
        let empty_pyramid = DepthPyramid::empty(gpu)?;
        let (layout, phase_layout) = layouts(gpu)?;
        let gate_layout = gate_layout(gpu)?;
        let pipelines = Pipelines::new(gpu, &layout, &phase_layout, &gate_layout)?;
        let dispatches = gpu.scoped("lod dispatches", || {
            device.create_buffer(&BufferDescriptor {
                label: Some("lod dispatches"),
                size: (u64::from(POST_SLOT) + 1) * DISPATCH_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })?;
        let gate_group = gpu.scoped("lod gate bind group", || {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("lod gate"),
                layout: &gate_layout,
                entries: &[
                    entry(0, &frame),
                    entry(1, &dispatches),
                    step_entry(2, &steps),
                ],
            })
        })?;
        let (done, mapped) = mpsc::channel();
        let (returns, returned) = mpsc::channel();
        let reader = CutReader {
            mapped,
            back: returns.clone(),
        };
        Ok(Self {
            config,
            capacity,
            cluster_capacity,
            bounds,
            work,
            frame,
            cut,
            retry,
            clusters_main,
            clusters_post,
            params,
            view,
            previous_view,
            steps,
            dispatches,
            gate_group,
            empty_pyramid,
            layout,
            phase_layout,
            pipelines,
            group: None,
            free: Vec::new(),
            made: 0,
            done,
            reader: Some(reader),
            returns,
            returned,
            timer: None,
        })
    }

    /// How it is set up.
    #[must_use]
    pub const fn config(&self) -> &LodConfig {
        &self.config
    }

    /// The bounds and the pass that computes them.
    #[must_use]
    pub const fn bounds(&self) -> &BoundsPass {
        &self.bounds
    }

    /// A phase's cluster list: `false` the first phase's, `true` the second's; each
    /// [`Lod::cluster_capacity`] [`GpuCluster`]s, the first `instance_count` of
    /// [`Lod::draw_args`] written.
    #[must_use]
    pub const fn clusters(&self, post: bool) -> &Buffer {
        if post {
            &self.clusters_post
        } else {
            &self.clusters_main
        }
    }

    /// Clusters each list holds.
    #[must_use]
    pub const fn cluster_capacity(&self) -> u32 {
        self.cluster_capacity
    }

    /// The buffer holding a phase's indexed indirect draw arguments
    /// (`wgpu::util::DrawIndexedIndirectArgs`), and their offset in it: `false` the first
    /// phase's, `true` the second's.
    #[must_use]
    pub const fn draw_args(&self, post: bool) -> (&Buffer, u64) {
        (&self.frame, if post { POST_ARGS } else { 0 })
    }

    /// The frame's cut, as [`Lod::encode_main`] writes it.
    #[must_use]
    pub const fn cut_list(&self) -> CutList<'_> {
        CutList {
            entries: &self.cut,
            counts: &self.frame,
            count_word: CUT_COUNT_WORD,
        }
    }

    /// Encodes the bounds pass for `update`'s tiles ([`BoundsPass::encode`]), timed when timing
    /// is on.
    ///
    /// # Errors
    ///
    /// As [`BoundsPass::encode`].
    pub fn encode_bounds(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        tree: &Buffer,
        update: &BoundsUpdate,
        slots: &SlotBuffers,
    ) -> Result<(), LodError> {
        let writes = self.timer.as_ref().map(|timer| timer.writes(0));
        self.bounds
            .encode(gpu, encoder, tree, update, slots, writes)?;
        Ok(())
    }

    /// Encodes the first phase: the level choice, the cut, the requests, culling, and the first
    /// cluster list with its draw arguments. The settings and views are written through the
    /// queue, so encode this once per submission.
    ///
    /// # Errors
    ///
    /// [`LodError::TreeTooSmall`] if the tree buffer holds fewer nodes than the level choice
    /// covers; [`LodError::PyramidSize`] if the previous pyramid wasn't built for depth images
    /// of the previous view's size (nothing is encoded then); [`LodError::Gpu`] if a bind group
    /// can't be made.
    pub fn encode_main(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        inputs: &MainInputs<'_>,
    ) -> Result<(), LodError> {
        self.check(inputs)?;
        self.write_settings(gpu, inputs);
        let passes = self.config.relax_passes;
        let group = self.group(gpu, inputs.tree, inputs.requests)?;
        let pyramid = inputs
            .previous
            .map_or(&self.empty_pyramid, |(_, pyramid)| pyramid);
        let phase = self.phase_group(gpu, false, pyramid)?;
        let nodes = self.capacity.div_ceil(GROUP);
        let step = |index: u32| index * u32::try_from(UNIFORM_STRIDE).unwrap_or(256);
        let p = &self.pipelines;
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lod choose"),
                timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(1)),
            });
            pass.set_bind_group(0, &group, &[0]);
            pass.set_bind_group(1, &phase, &[]);
            pass.set_pipeline(&p.begin);
            pass.dispatch_workgroups(1, 1, 1);
            for pipeline in [&p.choose, &p.close] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(nodes, 1, 1);
            }
            pass.set_pipeline(&p.threshold);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&p.mark);
            pass.dispatch_workgroups(nodes, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lod relax"),
                timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(2)),
            });
            for index in 0..passes {
                pass.set_pipeline(&p.relax);
                pass.set_bind_group(0, &group, &[step(index)]);
                pass.set_bind_group(1, &phase, &[]);
                pass.dispatch_workgroups_indirect(
                    &self.dispatches,
                    u64::from(index) * DISPATCH_BYTES,
                );
                pass.set_pipeline(&p.gate);
                pass.set_bind_group(0, &self.gate_group, &[step(index)]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lod sweep"),
                timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(3)),
            });
            pass.set_bind_group(1, &phase, &[]);
            pass.set_pipeline(&p.sweep);
            for level in 0..u32::from(inputs.finest.get()) {
                pass.set_bind_group(0, &group, &[step(level)]);
                pass.dispatch_workgroups(nodes, 1, 1);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lod emit"),
                timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(4)),
            });
            pass.set_bind_group(0, &group, &[0]);
            pass.set_bind_group(1, &phase, &[]);
            pass.set_pipeline(&p.emit);
            pass.dispatch_workgroups(nodes, 1, 1);
            pass.set_pipeline(&p.finish);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // The second phase's dispatch, out of the buffer its pass writes.
        encoder.copy_buffer_to_buffer(
            &self.frame,
            POST_GROUPS,
            &self.dispatches,
            u64::from(POST_SLOT) * DISPATCH_BYTES,
            DISPATCH_BYTES,
        );
        self.group = Some(group);
        Ok(())
    }

    /// Checks a frame's inputs against what the level choice was made for.
    ///
    /// # Errors
    ///
    /// As [`Lod::encode_main`]: [`LodError::TreeTooSmall`], [`LodError::PyramidSize`].
    fn check(&self, inputs: &MainInputs<'_>) -> Result<(), LodError> {
        let needed = u64::from(self.capacity) * NODE_BYTES;
        if inputs.tree.size() < needed {
            return Err(LodError::TreeTooSmall {
                bytes: inputs.tree.size(),
                nodes: self.capacity,
            });
        }
        if let Some((view, pyramid)) = inputs.previous {
            let size = view.size();
            let wanted = (size.width(), size.height());
            if pyramid.depth_size() != wanted {
                return Err(LodError::PyramidSize {
                    pyramid: pyramid.depth_size(),
                    view: wanted,
                });
            }
        }
        Ok(())
    }

    /// Writes a frame's settings, its views and the first relaxation pass's dispatch through
    /// the queue.
    fn write_settings(&self, gpu: &Gpu, inputs: &MainInputs<'_>) {
        let queue = gpu.queue();
        let passes = self.config.relax_passes;
        let thresholds = self.config.thresholds;
        let params = GpuParams {
            capacity: self.capacity,
            finest: u32::from(inputs.finest.get()),
            budget: self.config.triangle_budget,
            passes,
            cluster_capacity: self.cluster_capacity,
            cut_capacity: self.capacity,
            retry_capacity: self.capacity,
            previous: u32::from(inputs.previous.is_some()),
            refine_error: thresholds.refine_error,
            refine_quad: thresholds.refine_quad,
            coarsen_error: thresholds.coarsen_error,
            coarsen_quad: thresholds.coarsen_quad,
            options: if self.config.offscreen_seams {
                OPTION_OFFSCREEN_SEAMS
            } else {
                0
            },
            reserved: [0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let nodes = self.capacity.div_ceil(GROUP);
        queue.write_buffer(&self.dispatches, 0, bytemuck::cast_slice(&[nodes, 1, 1]));
        queue.write_buffer(&self.view, 0, bytemuck::bytes_of(&inputs.view.gpu()));
        let previous = inputs.previous.map_or(inputs.view, |(view, _)| view);
        queue.write_buffer(&self.previous_view, 0, bytemuck::bytes_of(&previous.gpu()));
    }

    /// Encodes the second phase, after the first phase's clusters are drawn and `pyramid` is
    /// built from that depth: the tiles the first phase's test hid, tested against `pyramid`
    /// from this frame's view, into the second cluster list. Encode it whenever the first phase
    /// had a previous pyramid, or the tiles it hid are never drawn; without one it draws nothing.
    ///
    /// # Errors
    ///
    /// [`LodError::Gpu`] if a bind group can't be made.
    pub fn encode_post(
        &self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        pyramid: &DepthPyramid,
    ) -> Result<(), LodError> {
        let Some(group) = self.group.as_ref() else {
            return Ok(());
        };
        let phase = self.phase_group(gpu, true, pyramid)?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("lod post"),
            timestamp_writes: self.timer.as_ref().map(|timer| timer.writes(5)),
        });
        pass.set_bind_group(0, group, &[0]);
        pass.set_bind_group(1, &phase, &[]);
        pass.set_pipeline(&self.pipelines.post);
        pass.dispatch_workgroups_indirect(&self.dispatches, u64::from(POST_SLOT) * DISPATCH_BYTES);
        pass.set_pipeline(&self.pipelines.post_finish);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    /// Encodes copying the frame's counts and cut to a readback buffer, after both phases, as the
    /// results of `frame` (any number the caller tells its frames by: [`LodOutput::frame`]);
    /// `None` if every readback buffer is in flight (this frame's results are skipped). Call
    /// [`CutReadback::submitted`] on it once the command buffer is submitted.
    ///
    /// # Errors
    ///
    /// [`LodError::Gpu`] if a readback buffer can't be made.
    pub fn encode_readback(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        frame: u64,
    ) -> Result<Option<CutReadback>, LodError> {
        self.take_unused();
        let buffer = match self.free.pop() {
            Some(buffer) => buffer,
            None if self.made < READBACKS => {
                let size = FRAME_BYTES + self.cut.size();
                let buffer = gpu.scoped("lod readback", || {
                    gpu.device().create_buffer(&BufferDescriptor {
                        label: Some("lod readback"),
                        size,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })?;
                self.made += 1;
                buffer
            }
            None => return Ok(None),
        };
        encoder.copy_buffer_to_buffer(&self.frame, 0, &buffer, 0, FRAME_BYTES);
        encoder.copy_buffer_to_buffer(&self.cut, 0, &buffer, FRAME_BYTES, self.cut.size());
        if let Some(timer) = &self.timer {
            encoder.resolve_query_set(&timer.queries, 0..2 * STAGES, &timer.resolved, 0);
        }
        Ok(Some(CutReadback {
            buffer: Some(buffer),
            frame,
            done: self.done.clone(),
            unused: self.returns.clone(),
        }))
    }

    /// The latest frame's results read back, as [`CutReader::collect`]; `None` once
    /// [`Lod::take_reader`] has moved the reader out.
    ///
    /// # Errors
    ///
    /// [`LodError::Gpu`] if a mapped buffer can't be viewed.
    pub fn collect(&mut self, finest: Level) -> Result<Option<LodOutput>, LodError> {
        let latest = match &self.reader {
            Some(reader) => reader.collect(finest)?,
            None => None,
        };
        self.take_unused();
        Ok(latest)
    }

    /// Moves where the readbacks arrive out of the level choice, to read them on another thread
    /// ([`CutReader`]); `None` if it was taken already.
    pub const fn take_reader(&mut self) -> Option<CutReader> {
        self.reader.take()
    }

    /// Takes back readback buffers read, dropped unread, or lost.
    fn take_unused(&mut self) {
        while let Ok(returned) = self.returned.try_recv() {
            match returned {
                Returned::Free(buffer) => self.free.push(buffer),
                Returned::Lost => self.made = self.made.saturating_sub(1),
            }
        }
    }

    /// Turns on timing every stage with timestamp queries, if the device has them; whether it
    /// does.
    ///
    /// # Errors
    ///
    /// [`LodError::Gpu`] if the queries or their buffers can't be made.
    pub fn enable_timing(&mut self, gpu: &Gpu) -> Result<bool, LodError> {
        let Some(period) = gpu.timestamp_period() else {
            return Ok(false);
        };
        let device = gpu.device();
        let size = u64::from(2 * STAGES * QUERY_SIZE);
        let timer = gpu.scoped("lod timer", || {
            let buffer = |label, usage| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };
            Timer {
                queries: device.create_query_set(&QuerySetDescriptor {
                    label: Some("lod timestamps"),
                    ty: QueryType::Timestamp,
                    count: 2 * STAGES,
                }),
                resolved: buffer(
                    "lod timestamps resolved",
                    BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                ),
                staging: buffer(
                    "lod timestamps readback",
                    BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                ),
                period,
            }
        })?;
        self.timer = Some(timer);
        Ok(true)
    }

    /// The stages' times of the last frame encoded with [`Lod::encode_readback`], waiting for
    /// the GPU (for tools and benchmarks); `None` without timing. A stage not encoded that
    /// frame reads as zero.
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer`].
    pub fn read_timings(&self, gpu: &Gpu) -> Result<Option<LodTimings>, LodError> {
        let Some(timer) = &self.timer else {
            return Ok(None);
        };
        let ticks: Vec<u64> = gpu.read_buffer_with(&timer.resolved, &timer.staging)?;
        let stage = |index: usize| {
            let begin = ticks.get(2 * index).copied().unwrap_or(0);
            let end = ticks.get(2 * index + 1).copied().unwrap_or(0);
            let elapsed = end.saturating_sub(begin);
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a stage's ticks are far below 2^52, and the product is nanoseconds"
            )]
            let nanos = (elapsed as f64 * f64::from(timer.period)) as u64;
            Duration::from_nanos(nanos)
        };
        Ok(Some(LodTimings {
            bounds: stage(0),
            choose: stage(1),
            relax: stage(2),
            sweep: stage(3),
            emit: stage(4),
            post: stage(5),
        }))
    }

    /// Group 0 for this frame: the tree and request buffers are the caller's.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn group(&self, gpu: &Gpu, tree: &Buffer, requests: &Buffer) -> Result<BindGroup, GpuError> {
        gpu.scoped("lod bind group", || {
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("lod"),
                layout: &self.layout,
                entries: &[
                    entry(0, tree),
                    entry(1, self.bounds.buffer()),
                    entry(2, &self.work),
                    entry(3, &self.frame),
                    entry(4, requests),
                    entry(5, &self.cut),
                    entry(6, &self.retry),
                    entry(7, &self.params),
                    entry(8, &self.view),
                    step_entry(9, &self.steps),
                ],
            })
        })
    }

    /// Group 1 for a phase: its cluster list, its pyramid, and the view the pyramid was made
    /// from (the previous frame's for the first phase, this frame's for the second).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn phase_group(
        &self,
        gpu: &Gpu,
        post: bool,
        pyramid: &DepthPyramid,
    ) -> Result<BindGroup, GpuError> {
        let view = if post {
            &self.view
        } else {
            &self.previous_view
        };
        gpu.scoped("lod phase bind group", || {
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("lod phase"),
                layout: &self.phase_layout,
                entries: &[
                    entry(0, self.clusters(post)),
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(pyramid.view()),
                    },
                    entry(2, view),
                ],
            })
        })
    }
}

/// The step uniforms: entry `k`, at `k` · [`UNIFORM_STRIDE`], is level or pass `k`.
///
/// # Errors
///
/// As [`Gpu::scoped`].
fn steps(gpu: &Gpu) -> Result<Buffer, GpuError> {
    let steps = (0..STEPS).flat_map(|index| {
        let step = GpuStep {
            index,
            reserved: [0; 3],
        };
        let mut bytes = bytemuck::bytes_of(&step).to_vec();
        bytes.resize(usize::try_from(UNIFORM_STRIDE).unwrap_or(256), 0);
        bytes
    });
    let steps: Vec<u8> = steps.collect();
    gpu.scoped("lod steps", || {
        gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("lod steps"),
            contents: &steps,
            usage: BufferUsages::UNIFORM,
        })
    })
}

/// The level choice's bind group layouts: group 0 (the frame's buffers), group 1 (a phase's
/// cluster list, pyramid and view).
///
/// # Errors
///
/// As [`Gpu::scoped`].
fn layouts(gpu: &Gpu) -> Result<(BindGroupLayout, BindGroupLayout), GpuError> {
    let device = gpu.device();
    let uniform = BufferBindingType::Uniform;
    let layout = gpu.scoped("lod layout", || {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("lod"),
            entries: &[
                buffer_entry(0, READ, false),
                buffer_entry(1, READ, false),
                buffer_entry(2, WRITE, false),
                buffer_entry(3, WRITE, false),
                buffer_entry(4, WRITE, false),
                buffer_entry(5, WRITE, false),
                buffer_entry(6, WRITE, false),
                buffer_entry(7, uniform, false),
                buffer_entry(8, uniform, false),
                step_layout_entry(9),
            ],
        })
    })?;
    let phase_layout = gpu.scoped("lod phase layout", || {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("lod phase"),
            entries: &[
                buffer_entry(0, WRITE, false),
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer_entry(2, uniform, false),
            ],
        })
    })?;
    Ok((layout, phase_layout))
}

/// The gate's bind group layout: the frame buffer as words (read), the dispatch arguments
/// (written), the step.
///
/// # Errors
///
/// As [`Gpu::scoped`].
fn gate_layout(gpu: &Gpu) -> Result<BindGroupLayout, GpuError> {
    gpu.scoped("lod gate layout", || {
        gpu.device()
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("lod gate"),
                entries: &[
                    buffer_entry(0, READ, false),
                    buffer_entry(1, WRITE, false),
                    step_layout_entry(2),
                ],
            })
    })
}

/// The layout entry of a [`GpuStep`] uniform at a dynamic offset.
const fn step_layout_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: BufferSize::new(size_of::<GpuStep>() as u64),
        },
        count: None,
    }
}

/// The bind group entry of one [`GpuStep`] of `steps`, its offset dynamic.
const fn step_entry(binding: u32, steps: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer(BufferBinding {
            buffer: steps,
            offset: 0,
            size: BufferSize::new(size_of::<GpuStep>() as u64),
        }),
    }
}

impl Pipelines {
    /// Every entry point of `lod.wgsl`, on `layout` and `phase_layout`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::compute_pipeline`].
    fn new(
        gpu: &Gpu,
        layout: &BindGroupLayout,
        phase_layout: &BindGroupLayout,
        gate_layout: &BindGroupLayout,
    ) -> Result<Self, GpuError> {
        let gate_source = format!("const RAISES: u32 = {RAISES_WORD}u;\n{GATE_WGSL}");
        let gate = pipeline(gpu, "lod gate", &gate_source, &[gate_layout], "gate")?;
        let source = format!("{NODE_WGSL}{REQUESTS_WGSL}{CULL_WGSL}{CLUSTER_WGSL}{LOD_WGSL}");
        let groups = [layout, phase_layout];
        let make = |entry_point: &str| {
            pipeline(
                gpu,
                &format!("lod {entry_point}"),
                &source,
                &groups,
                entry_point,
            )
        };
        Ok(Self {
            begin: make("begin")?,
            choose: make("choose")?,
            close: make("close")?,
            threshold: make("threshold")?,
            mark: make("mark")?,
            relax: make("relax")?,
            sweep: make("sweep")?,
            emit: make("emit")?,
            finish: make("finish")?,
            post: make("post")?,
            post_finish: make("post_finish")?,
            gate,
        })
    }
}

impl Timer {
    /// Stage `stage`'s timestamp writes.
    fn writes(&self, stage: u32) -> ComputePassTimestampWrites<'_> {
        ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(2 * stage),
            end_of_pass_write_index: Some(2 * stage + 1),
        }
    }
}

/// The results in a mapped readback buffer, of frame `tag`.
///
/// # Errors
///
/// [`LodError::Gpu`] if it can't be viewed.
fn read_output(buffer: &Buffer, finest: Level, tag: u64) -> Result<LodOutput, LodError> {
    let view = buffer.get_mapped_range(..).map_err(GpuError::from)?;
    let frame_end = usize::try_from(FRAME_BYTES).unwrap_or(usize::MAX);
    let frame: GpuFrame = view
        .get(..frame_end)
        .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
        .unwrap_or_else(GpuFrame::zeroed);
    // Only the entries written: the buffer holds the whole capacity's.
    let written_bytes = u64::from(frame.cut_count).saturating_mul(CUT_BYTES);
    let cut_end = usize::try_from(written_bytes)
        .unwrap_or(usize::MAX)
        .saturating_add(frame_end)
        .min(view.len());
    let entries: Vec<GpuCutEntry> = view
        .get(frame_end..cut_end)
        .map(bytemuck::pod_collect_to_vec)
        .unwrap_or_default();
    let written = entries.iter();
    let mut output = LodOutput {
        frame: tag,
        stats: stats(&frame),
        ..LodOutput::default()
    };
    for entry in written {
        let Some(place) = Place::from_packed(entry.face, entry.place) else {
            continue;
        };
        output.nodes.push((entry.node, place));
        let Some(key) = place.key(finest) else {
            continue;
        };
        output.cut.insert(key);
        if entry.flags & CUT_DRAWN != 0 {
            output.drawn.insert(key);
        }
    }
    Ok(output)
}

/// The counts of a frame's results.
fn stats(frame: &GpuFrame) -> LodStats {
    LodStats {
        cut: frame.cut_count,
        cut_triangles: frame.cut_triangles,
        refined_triangles: frame.triangles.saturating_mul(BUDGET_UNIT),
        drawn_triangles: frame.drawn_triangles,
        drawn: (frame.drawn_main, frame.drawn_post),
        clusters: (frame.main.instance_count, frame.post.instance_count),
        over_pixel: frame.over_pixel,
        waiting: frame.waiting,
        budget_limited: frame.budget_limited,
        held_by_neighbours: frame.held_by_neighbours,
        requested: frame.requested,
        culled: (
            frame.culled_frustum,
            frame.culled_cone,
            frame.culled_occlusion,
            frame.culled_occlusion_post,
        ),
        shortfall: frame.shortfall,
        refused: frame.refused,
        stuck: frame.stuck,
        raises: frame.raises.to_vec(),
        overflow: (
            frame.cluster_overflow,
            frame.retry_overflow,
            frame.cut_overflow,
        ),
    }
}
