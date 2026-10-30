//! Culling: what a tile's bounds are, and the tests the level choice runs on them.
//!
//! # Bounds
//!
//! Every resident tile has [`GpuBounds`], by node index, in a buffer the level choice reads: a
//! sphere holding its vertices, its normal cone (from its own triangles: a coarse tile's cone
//! comes from the display tile, never from its children's cones, since a coarse quad's normal
//! can fall outside them) and its longest quad edge. Residency uploads no bounds, so the
//! [`BoundsPass`] computes them on the GPU from the tile's slot, in the frame its data arrives
//! or changes: the tile tree stamps each node when its data changes
//! ([`crate::tree::TileTree::changed_since`]), the owner thread gathers those nodes into a
//! [`BoundsUpdate`], and the render thread encodes the pass right after the frame's uploads.
//!
//! **What the cone holds**, and so what the raster may draw from a tile, morphing or not,
//! without a culled tile leaving slivers at the silhouette:
//!
//! 1. the tile's own triangles: each quad split along `(i, j)–(i + 1, j + 1)` (decision 4);
//! 2. its **parent triangulation**: each 2 × 2-quad block split along its `(0, 0)–(2, 2)` diagonal.
//!    This is where the geomorph starts: with every odd vertex at its decision-4 midpoint, each of
//!    the tile's triangles lies in one of these, so has its normal;
//! 3. every **edge variant**: on an edge whose neighbour is a level coarser, each odd border vertex
//!    `o` (between the even border vertices `a` and `b`; `m`, `ma` and `mb` the vertices one row in
//!    from `o`, `a` and `b`) is dropped, and the two border quads sharing it become the fan `(a, b,
//!    m)`, `(a, m, ma)`, `(b, mb, m)`. **The variant index buffers must use exactly this
//!    triangulation** (`bounds.wgsl`'s `extra_normal`);
//! 4. every **morph state** (`morph_normal`): each own and fan triangle's moving vertices are its
//!    two or three odd ones, each with its own factor (the raster's seams and distance factor vary
//!    it per vertex). The triangle's normal is linear in each vertex's place, so with the vertices
//!    anywhere on their ways it is a sum, with weights not negative, of the normals of its eight
//!    corners: each vertex at its start or its end. All at the start lie in the parent
//!    triangulation's plane (2), all at the end are the triangle itself (1, 3); the cone also holds
//!    the six mixed ones, so it holds every morph state (a cone narrower than a half-space is
//!    convex). That is twelve normals a quad and 48 an odd border vertex, about 3.4 times the
//!    tile's own triangles for the bounds pass to fold in (a cost per arrival, not per frame; not
//!    measured).
//!
//! # Tests
//!
//! All conservative, in WGSL ([`CULL_WGSL`]) with CPU twins in the tests:
//!
//! - **frustum:** the sphere against the view's planes;
//! - **normal cone:** every triangle faces away when `cos(θ + α) > r / d` (θ from the cone's axis
//!   to the tile's centre seen from the eye, α the cone's half-angle, `r` the sphere's radius, `d`
//!   its distance);
//! - **occlusion:** the sphere's screen rectangle against a [`DepthPyramid`]: occluded when its
//!   nearest depth is farther than the farthest depth the pyramid holds over the rectangle. Two
//!   phases (the level choice's `main` and `post`): first against the previous frame's pyramid seen
//!   from the previous frame's view, then the tiles that failed it against this frame's.
//!
//! Shaders are WGSL: these passes share no source with the CPU code, so rust-gpu would buy
//! nothing here (plan: "chosen per pass and recorded").

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferSize, BufferUsages, CommandEncoder, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipeline, ComputePipelineDescriptor, Extent3d,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

use crate::gpu::{Gpu, GpuError};
use crate::residency::{CLASSES, Residency, SizeClass};
use crate::surface::Size;
use crate::tree::TileTree;
use crate::upload::VERTEX_WGSL;

#[cfg(test)]
pub(crate) mod reference;
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests;

/// The WGSL declarations and functions of culling: `TileBounds` ([`GpuBounds`]), `View`
/// ([`crate::camera::GpuView`]), and the frustum and cone tests. No bindings.
pub const CULL_WGSL: &str = include_str!("cull.wgsl");

/// The bounds pass's own source, after [`crate::tree::NODE_WGSL`], [`VERTEX_WGSL`] and
/// [`CULL_WGSL`].
const BOUNDS_WGSL: &str = include_str!("bounds.wgsl");

/// The depth pyramid's passes.
const PYRAMID_WGSL: &str = include_str!("pyramid.wgsl");

/// A tile's bounds as the GPU reads them: 48 bytes, WGSL's `TileBounds`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GpuBounds {
    /// The centre of a sphere holding every vertex the tile draws, world units.
    pub center: [f32; 3],
    /// Its radius; negative while unknown ([`GpuBounds::UNKNOWN`]: nothing is culled then).
    pub radius: f32,
    /// The normal cone's axis, unit length; zero without a cone.
    pub axis: [f32; 3],
    /// The cosine of the cone's half-angle; at or below 0 the cone culls nothing.
    pub cone_cos: f32,
    /// The longest quad edge, world units.
    pub quad_edge: f32,
    /// Zero.
    pub reserved: [u32; 3],
}

impl GpuBounds {
    /// Bounds not computed yet: a negative radius, no cone. Such a tile is never culled and
    /// never refined.
    pub const UNKNOWN: Self = Self {
        center: [0.0; 3],
        radius: -1.0,
        axis: [0.0; 3],
        cone_cos: -1.0,
        quad_edge: 0.0,
        reserved: [0; 3],
    };
}

/// The bytes of one [`GpuBounds`].
pub const BOUNDS_BYTES: u64 = size_of::<GpuBounds>() as u64;

/// Why a culling pass couldn't be set up or encoded.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CullError {
    /// The GPU failed: a buffer, a pipeline, a bind group.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// A tile's slot names a slot buffer the frame didn't pass.
    #[error("no slot buffer {buffer} of size class {class}")]
    MissingSlotBuffer {
        /// The size class.
        class: u8,
        /// The buffer within it.
        buffer: u16,
    },
}

/// Bytes between the dynamic offsets of a uniform array: wgpu's portable alignment.
pub(crate) const UNIFORM_STRIDE: u64 = 256;

/// The most workgroups one dispatch dimension may have (`wgpu`'s default limit).
pub(crate) const MAX_GROUPS: u32 = 65_535;

/// The slot buffers of every size class, as the passes that read tile data bind them: one
/// buffer per bind group, by (class, buffer index) as a [`crate::residency::Slot`] names them.
///
/// Cloned from [`Residency::slot_buffers`] on the owner thread and handed to the render thread
/// with the frame ([`wgpu::Buffer`] is a shared handle).
#[derive(Debug, Clone, Default)]
pub struct SlotBuffers {
    /// Per class, its buffers in buffer-index order.
    classes: Vec<Vec<Buffer>>,
}

impl SlotBuffers {
    /// The slot buffers `residency` holds now.
    #[must_use]
    pub fn of(residency: &Residency) -> Self {
        let classes = (0..CLASSES)
            .filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new))
            .map(|class| residency.slot_buffers(class).to_vec());
        Self {
            classes: classes.collect(),
        }
    }

    /// Slot buffers given directly: `classes[c]` is class `c`'s, in buffer-index order.
    #[must_use]
    pub const fn new(classes: Vec<Vec<Buffer>>) -> Self {
        Self { classes }
    }

    /// Every class's buffers: `classes()[c]` is class `c`'s, in buffer-index order.
    pub(crate) fn classes(&self) -> &[Vec<Buffer>] {
        &self.classes
    }

    /// Buffer `buffer` of class `class`.
    fn get(&self, class: u8, buffer: u16) -> Option<&Buffer> {
        self.classes
            .get(usize::from(class))
            .and_then(|buffers| buffers.get(usize::from(buffer)))
    }
}

/// One slot buffer's tiles to bound: (class, buffer index), node indices.
type Run = ((u8, u16), Vec<u32>);

/// The tiles whose bounds a frame recomputes, grouped by slot buffer: gathered on the owner
/// thread from the tile tree after the frame's uploads are staged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundsUpdate {
    /// Per slot buffer, by (class, buffer), its tiles' node indices.
    runs: Vec<Run>,
    /// The tree's generation this update brings the bounds to.
    generation: u64,
}

impl BoundsUpdate {
    /// Every resident tile of `tree` whose data changed after `generation`: pass the previous
    /// update's [`BoundsUpdate::generation`] once its frame was submitted (a frame dropped
    /// unsubmitted leaves its tiles to the next update), 0 for every resident tile.
    #[must_use]
    pub fn since(tree: &TileTree, generation: u64) -> Self {
        let mut runs: BTreeMap<(u8, u16), Vec<u32>> = BTreeMap::new();
        for index in tree.changed_since(generation) {
            let slot = tree.gpu_node(index).slot;
            let class = u8::try_from(slot >> 29).unwrap_or(u8::MAX);
            let buffer = u16::try_from((slot >> 21) & 0xFF).unwrap_or(u16::MAX);
            runs.entry((class, buffer)).or_default().push(index);
        }
        Self {
            runs: runs.into_iter().collect(),
            generation: tree.generation(),
        }
    }

    /// The tree's generation the bounds are at once this update's frame is submitted.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// How many tiles it bounds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.runs.iter().map(|(_, nodes)| nodes.len()).sum()
    }

    /// The node indices of the tiles it bounds: those whose data changed.
    pub fn nodes(&self) -> impl Iterator<Item = u32> + '_ {
        self.runs
            .iter()
            .flat_map(|(_, nodes)| nodes.iter().copied())
    }

    /// Whether it bounds no tile.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.iter().all(|(_, nodes)| nodes.is_empty())
    }
}

/// A run of the bounds pass's node list, as its uniform: WGSL's `Run`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
struct GpuRun {
    /// The first node of the run in the list.
    first: u32,
    /// How many.
    count: u32,
    /// Zero.
    reserved: [u32; 2],
}

/// The bounds of every resident tile, and the pass that computes them from the tiles' slots.
#[derive(Debug)]
pub struct BoundsPass {
    /// The bounds, by node index ([`GpuBounds`]): [`GpuBounds::UNKNOWN`] until computed.
    bounds: Buffer,
    /// How many nodes it holds.
    capacity: u32,
    /// The node list the pass reads, grown as a frame needs.
    nodes: Buffer,
    /// The runs' uniforms, one per [`UNIFORM_STRIDE`], grown as a frame needs.
    runs: Buffer,
    /// The pass's bind group layout.
    layout: BindGroupLayout,
    /// The pass.
    pipeline: ComputePipeline,
}

impl BoundsPass {
    /// Bounds for `capacity` nodes, every one unknown, and the pass.
    ///
    /// # Errors
    ///
    /// [`CullError::Gpu`] if a buffer or the pipeline can't be made.
    pub fn new(gpu: &Gpu, capacity: u32) -> Result<Self, CullError> {
        let device = gpu.device();
        let unknown = vec![GpuBounds::UNKNOWN; usize::try_from(capacity.max(1)).unwrap_or(1)];
        let bounds = gpu.scoped("tile bounds", || {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("tile bounds"),
                contents: bytemuck::cast_slice(&unknown),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            })
        })?;
        let nodes = storage_buffer(gpu, "bounds node list", 4096)?;
        let runs = uniform_buffer(gpu, "bounds runs", UNIFORM_STRIDE * 16)?;
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
        let layout = gpu.scoped("bounds layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("bounds"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    storage(2, true),
                    storage(3, true),
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: BufferSize::new(size_of::<GpuRun>() as u64),
                        },
                        count: None,
                    },
                ],
            })
        })?;
        let source = format!(
            "{}{VERTEX_WGSL}{CULL_WGSL}{BOUNDS_WGSL}",
            crate::tree::NODE_WGSL
        );
        let pipeline = pipeline(gpu, "bounds", &source, &[&layout], "bound_tiles")?;
        Ok(Self {
            bounds,
            capacity,
            nodes,
            runs,
            layout,
            pipeline,
        })
    }

    /// The bounds buffer: [`BoundsPass::capacity`] [`GpuBounds`], by node index.
    #[must_use]
    pub const fn buffer(&self) -> &Buffer {
        &self.bounds
    }

    /// How many nodes it holds.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Writes bounds known on the CPU for the nodes from `first` on, through the queue (ahead
    /// of the next submission): for tools and tests whose tiles have no slot data.
    pub fn write(&self, gpu: &Gpu, first: u32, bounds: &[GpuBounds]) {
        let room = usize::try_from(self.capacity.saturating_sub(first)).unwrap_or(0);
        let bounds = bounds.get(..room.min(bounds.len())).unwrap_or(&[]);
        gpu.queue().write_buffer(
            &self.bounds,
            u64::from(first) * BOUNDS_BYTES,
            bytemuck::cast_slice(bounds),
        );
    }

    /// Encodes the pass for `update`'s tiles, reading `tree` (the tile tree's buffer) and their
    /// slots in `slots`. The node list and runs are written through the queue, so encode this
    /// once per submission, after the frame's uploads and before the level choice.
    ///
    /// A failure leaves no stale bounds: every tile of `update` is reset to
    /// [`GpuBounds::UNKNOWN`] (drawn, never culled or refined) until a later update bounds it.
    ///
    /// # Errors
    ///
    /// [`CullError::MissingSlotBuffer`] if a tile's slot buffer isn't in `slots` (nothing is
    /// encoded then); [`CullError::Gpu`] if a buffer or bind group can't be made.
    pub fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        tree: &Buffer,
        update: &BoundsUpdate,
        slots: &SlotBuffers,
        timestamps: Option<ComputePassTimestampWrites<'_>>,
    ) -> Result<(), CullError> {
        let outcome = self.encode_runs(gpu, encoder, tree, update, slots, timestamps);
        if outcome.is_err() {
            let nodes = update.runs.iter().flat_map(|(_, nodes)| nodes);
            for &node in nodes {
                self.write(gpu, node, &[GpuBounds::UNKNOWN]);
            }
        }
        outcome
    }

    /// [`BoundsPass::encode`] but for resetting the bounds on failure.
    ///
    /// # Errors
    ///
    /// As [`BoundsPass::encode`].
    fn encode_runs(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        tree: &Buffer,
        update: &BoundsUpdate,
        slots: &SlotBuffers,
        timestamps: Option<ComputePassTimestampWrites<'_>>,
    ) -> Result<(), CullError> {
        // One dispatch per run, each at most MAX_GROUPS tiles.
        let mut list: Vec<u32> = Vec::with_capacity(update.len());
        let mut dispatches: Vec<(u8, u16, GpuRun)> = Vec::new();
        for ((class, buffer), nodes) in &update.runs {
            if slots.get(*class, *buffer).is_none() {
                return Err(CullError::MissingSlotBuffer {
                    class: *class,
                    buffer: *buffer,
                });
            }
            let chunks = nodes.chunks(usize::try_from(MAX_GROUPS).unwrap_or(usize::MAX));
            for chunk in chunks {
                let first = u32::try_from(list.len()).unwrap_or(u32::MAX);
                let count = u32::try_from(chunk.len()).unwrap_or(0);
                list.extend_from_slice(chunk);
                let run = GpuRun {
                    first,
                    count,
                    reserved: [0; 2],
                };
                dispatches.push((*class, *buffer, run));
            }
        }
        if dispatches.is_empty() {
            return Ok(());
        }
        let list_bytes = (list.len() as u64 * 4).max(4);
        if self.nodes.size() < list_bytes {
            self.nodes = storage_buffer(gpu, "bounds node list", list_bytes.next_power_of_two())?;
        }
        let runs_bytes = dispatches.len() as u64 * UNIFORM_STRIDE;
        if self.runs.size() < runs_bytes {
            self.runs = uniform_buffer(gpu, "bounds runs", runs_bytes.next_power_of_two())?;
        }
        let queue = gpu.queue();
        queue.write_buffer(&self.nodes, 0, bytemuck::cast_slice(&list));
        let mut strided = vec![0_u8; usize::try_from(runs_bytes).unwrap_or(0)];
        let stride = usize::try_from(UNIFORM_STRIDE).unwrap_or(256);
        let chunks = strided.chunks_mut(stride).zip(&dispatches);
        for (chunk, (_, _, run)) in chunks {
            if let Some(head) = chunk.get_mut(..size_of::<GpuRun>()) {
                head.copy_from_slice(bytemuck::bytes_of(run));
            }
        }
        queue.write_buffer(&self.runs, 0, &strided);
        let groups = dispatches
            .iter()
            .map(|(class, buffer, run)| -> Result<_, CullError> {
                let vertices = slots
                    .get(*class, *buffer)
                    .ok_or(CullError::MissingSlotBuffer {
                        class: *class,
                        buffer: *buffer,
                    })?;
                let group = gpu.scoped("bounds bind group", || {
                    gpu.device().create_bind_group(&BindGroupDescriptor {
                        label: Some("bounds"),
                        layout: &self.layout,
                        entries: &[
                            entry(0, tree),
                            entry(1, &self.bounds),
                            entry(2, vertices),
                            entry(3, &self.nodes),
                            BindGroupEntry {
                                binding: 4,
                                resource: BindingResource::Buffer(BufferBinding {
                                    buffer: &self.runs,
                                    offset: 0,
                                    size: BufferSize::new(size_of::<GpuRun>() as u64),
                                }),
                            },
                        ],
                    })
                })?;
                Ok((group, run.count))
            });
        let groups: Result<Vec<(wgpu::BindGroup, u32)>, CullError> = groups.collect();
        let groups = groups?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("bounds"),
            timestamp_writes: timestamps,
        });
        pass.set_pipeline(&self.pipeline);
        let offsets = (0_u32..).step_by(usize::try_from(UNIFORM_STRIDE).unwrap_or(256));
        for ((group, count), offset) in groups.iter().zip(offsets) {
            pass.set_bind_group(0, group, &[offset]);
            pass.dispatch_workgroups(*count, 1, 1);
        }
        Ok(())
    }
}

/// A bind group entry for the whole of `buffer`.
pub(crate) fn entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A storage buffer of `size` bytes, zeroed, that can be written and read back.
///
/// # Errors
///
/// As [`Gpu::scoped`].
pub(crate) fn storage_buffer(gpu: &Gpu, label: &str, size: u64) -> Result<Buffer, GpuError> {
    gpu.scoped(label, || {
        gpu.device().create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    })
}

/// A uniform buffer of `size` bytes, written through the queue.
///
/// # Errors
///
/// As [`Gpu::scoped`].
pub(crate) fn uniform_buffer(gpu: &Gpu, label: &str, size: u64) -> Result<Buffer, GpuError> {
    gpu.scoped(label, || {
        gpu.device().create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    })
}

/// A compute pipeline running `entry_point` of `source`, with an explicit layout of `groups`.
///
/// # Errors
///
/// As [`Gpu::compute_pipeline`].
pub(crate) fn pipeline(
    gpu: &Gpu,
    label: &str,
    source: &str,
    groups: &[&BindGroupLayout],
    entry_point: &str,
) -> Result<ComputePipeline, GpuError> {
    let device = gpu.device();
    gpu.scoped(label, || {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layouts: Vec<Option<&BindGroupLayout>> = groups.iter().copied().map(Some).collect();
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    })
}

/// The pyramid's texture format: one float of depth per texel.
const PYRAMID_FORMAT: TextureFormat = TextureFormat::R32Float;

/// A depth pyramid: the farthest depth over ever larger squares of a depth image, for
/// occlusion tests (see `pyramid.wgsl`).
///
/// Mip 0 is half the depth image's size, rounded up to a power of two (so every mip after it
/// is exactly half the one before, down to 1 × 1); a texel of mip `m` covers `2^(m + 1)` depth
/// texels a side, those past the image's edge standing for its last row or column. Built from
/// the frame's
/// `Depth32Float` image by [`PyramidPass::build`]. A renderer keeps two, this frame's and the
/// previous frame's, and swaps them.
#[derive(Debug)]
pub struct DepthPyramid {
    /// The texture, all mips.
    texture: Texture,
    /// One view per mip, in mip order.
    mips: Vec<TextureView>,
    /// A view of every mip, for the occlusion test.
    view: TextureView,
    /// The size, width and height, of the depth images it is built from.
    depth_size: (u32, u32),
}

impl DepthPyramid {
    /// A pyramid for depth images of `size`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn new(gpu: &Gpu, size: Size) -> Result<Self, GpuError> {
        Self::with_extent(gpu, size.width(), size.height())
    }

    /// A 1 × 1 pyramid for a 1 × 1 depth image (zeroed: the far depth), bound where a pass
    /// has no pyramid to test against.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub(crate) fn empty(gpu: &Gpu) -> Result<Self, GpuError> {
        Self::with_extent(gpu, 1, 1)
    }

    /// A pyramid for depth images `depth_width` × `depth_height`, both positive.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn with_extent(gpu: &Gpu, depth_width: u32, depth_height: u32) -> Result<Self, GpuError> {
        let (width, height) = mip_zero(depth_width, depth_height);
        let levels = width.max(height).trailing_zeros() + 1;
        let texture = gpu.scoped("depth pyramid", || {
            gpu.device().create_texture(&TextureDescriptor {
                label: Some("depth pyramid"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: PYRAMID_FORMAT,
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        })?;
        let mips = (0..levels).map(|level| {
            texture.create_view(&TextureViewDescriptor {
                label: Some("depth pyramid mip"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..TextureViewDescriptor::default()
            })
        });
        let mips: Vec<TextureView> = mips.collect();
        let view = texture.create_view(&TextureViewDescriptor::default());
        Ok(Self {
            texture,
            mips,
            view,
            depth_size: (depth_width, depth_height),
        })
    }

    /// The size, width and height, of the depth images it is built from.
    #[must_use]
    pub const fn depth_size(&self) -> (u32, u32) {
        self.depth_size
    }

    /// How many mips it has.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.texture.mip_level_count()
    }

    /// The texture, every mip (`R32Float`).
    #[must_use]
    pub const fn texture(&self) -> &Texture {
        &self.texture
    }

    /// A view of every mip, as the occlusion test binds it.
    #[must_use]
    pub const fn view(&self) -> &TextureView {
        &self.view
    }
}

/// The passes that build a [`DepthPyramid`].
#[derive(Debug)]
pub struct PyramidPass {
    /// Mip 0 from the depth image.
    from_depth: ComputePipeline,
    /// Each later mip from the one below.
    from_mip: ComputePipeline,
}

impl PyramidPass {
    /// The pyramid's passes.
    ///
    /// # Errors
    ///
    /// As [`Gpu::compute_pipeline`].
    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let from_depth = gpu.compute_pipeline(
            "depth pyramid from depth",
            ShaderSource::Wgsl(PYRAMID_WGSL.into()),
            "from_depth",
        )?;
        let from_mip = gpu.compute_pipeline(
            "depth pyramid from mip",
            ShaderSource::Wgsl(PYRAMID_WGSL.into()),
            "from_mip",
        )?;
        Ok(Self {
            from_depth,
            from_mip,
        })
    }

    /// Encodes building `pyramid` from `depth`, a view of a `Depth32Float` image of the
    /// pyramid's size with `TEXTURE_BINDING` usage.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], making the bind groups.
    pub fn build(
        &self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        depth: &TextureView,
        pyramid: &DepthPyramid,
        timestamps: Option<ComputePassTimestampWrites<'_>>,
    ) -> Result<(), GpuError> {
        let device = gpu.device();
        let groups = pyramid.mips.iter().enumerate().map(|(level, target)| {
            let (pipeline, source) = match level.checked_sub(1).and_then(|b| pyramid.mips.get(b)) {
                None => (&self.from_depth, (0, depth)),
                Some(below) => (&self.from_mip, (1, below)),
            };
            let group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("depth pyramid"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: source.0,
                        resource: BindingResource::TextureView(source.1),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(target),
                    },
                ],
            });
            (pipeline, group, level)
        });
        let groups: Vec<(&ComputePipeline, wgpu::BindGroup, usize)> =
            gpu.scoped("depth pyramid bind groups", || groups.collect())?;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("depth pyramid"),
            timestamp_writes: timestamps,
        });
        let (depth_width, depth_height) = pyramid.depth_size;
        let (width, height) = mip_zero(depth_width, depth_height);
        for (pipeline, group, level) in &groups {
            let shift = u32::try_from(*level).unwrap_or(31).min(31);
            let (width, height) = ((width >> shift).max(1), (height >> shift).max(1));
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        Ok(())
    }
}

/// The size of mip 0 of a pyramid for depth images `width` × `height`: half, rounded up to a
/// power of two.
pub(crate) const fn mip_zero(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(2).next_power_of_two(),
        height.div_ceil(2).next_power_of_two(),
    )
}
