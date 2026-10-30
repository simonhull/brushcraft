//! The shading pass: the visibility buffer turned into linear HDR colour, through a material.
//!
//! # What it does
//!
//! For each pixel the raster drew ([`crate::raster`]), the pass (`shade.wgsl`) rebuilds the
//! triangle drawn there and the surface at the pixel's centre, and calls the material
//! ([`crate::material`]) with it; the colour goes, linear, into a 16-bit float target
//! (`HDR_FORMAT`), which the tone map (`crate::tone`) then brings to the frame's format
//! (spec, "Shading": a linear HDR pipeline from the first frame).
//!
//! # Rebuilding what was drawn
//!
//! A pixel's [`crate::raster::VisibilityId`] names its tile's node, the cluster and the quad. The
//! quad's corners are placed by **the raster's own code**, `place.wgsl`'s `place_vertex`, from
//! the same inputs the vertex shader had:
//!
//! 1. **The cluster's `cell`:** its place in the tile (from the id's cluster and the tile's side),
//!    its edge variant and its finer edges. The id doesn't carry the last two, so the level choice
//!    records each tile's `crate::lod::DrawnEdges` in its cut entry and the raster's stamp pass
//!    keeps them per node; `cluster_cell` (the function `emit` lists clusters with) rebuilds the
//!    very cell from them.
//! 2. **The remap:** each corner's grid vertex through the variant's table.
//! 3. **The morph:** the stamps, the frame's settings (time, morph duration, threshold) and its
//!    view are the raster's buffers, as its last frame left them, so the time and distance factors
//!    are the same numbers.
//!
//! So each corner is the same function of the same data as in the vertex shader, and projects
//! through the same expression (`view_proj · (p − eye)`). **Which triangle** of the quad holds the
//! pixel is [`crate::raster::quad_triangle`]'s answer (the side of the diagonal the pixel's
//! centre is on, tested against the rasterizer in the raster's tests), checked against what the
//! rasterizer can have drawn: a triangle facing the eye (back faces are culled; facing is
//! tested in world space, so it holds for corners behind the eye), not degenerate (an edge
//! variant collapses one triangle of each border quad; its barycentric coordinates are none), and
//! holding the centre in front of the eye, the nearer where both do. The other triangle is taken
//! where only it can have been drawn, where the chosen one doesn't hold the centre and the other
//! holds it better, or where both hold it and the other is nearer. That catches the
//! rasterizer's tie rule on the diagonal (a centre within rounding of it; on a grazing quad the
//! other plane, extrapolated, lies far off: 2 pixels in 95,000 of the tests' scenes, 8 % off
//! in depth, before the check); a quad folded over itself, seen grazing, whose hidden or
//! back-facing triangle lies on the centre's side of the diagonal; and a corner behind the eye,
//! which the screen can't hold, where the choice is the triangle holding the centre best. The
//! point, the normal and the roughness term are
//! the corners' mixed by the **perspective-correct barycentric coordinates** of the pixel's centre,
//! solved in homogeneous clip coordinates (Cramer's rule: no division by a corner's `w`), as the
//! rasterizer interpolates. A morphing vertex's normal and roughness term are carried from its
//! parents' midpoint by its factor, as its position is. That midpoint is of the tile's own even
//! vertices' normals, not the parent tile's (whose averaged normals and roughness are its own,
//! over a wider tent), so as a morph starts the shading can step slightly where the geometry
//! doesn't. The tests check the rebuilt point against
//! the depth the raster wrote, pixel by pixel, with edge variants and morphs in the scene, and that
//! the check bites when either is ignored.
//!
//! On hardware that compiles the vertex and compute stages' arithmetic differently (fused
//! multiply-adds in one and not the other) the rebuilt corners may differ from the drawn ones in
//! their last bits. That is harmless: positions, normals and the roughness term are continuous
//! across a triangle's edges, so a pixel near the diagonal shaded from the other triangle gets
//! the same values to within that rounding.
//!
//! # The material and the overlay
//!
//! The material sees WGSL's `Surface` (the point, the unit normal, the roughness term, channels)
//! and `Eye` (`crate::material::SURFACE_WGSL`), and returns linear colour. The **wireframe**
//! is analytic: each barycentric coordinate over its rate of change across the screen is the
//! pixel's distance to the triangle's edge opposite that corner, and a line of
//! `WIRE_WIDTH` pixels with a pixel's ramp is mixed over the colour
//! ([`ShadeSettings::wireframe`]). Background pixels are written as zero (alpha 0), for the tone
//! map to leave the frame's clear colour there.
//!
//! # Paths
//!
//! [`ShadePath::Compute`] (desktop) runs one invocation a pixel in 8 × 8 groups, writing the
//! target as a storage image. [`ShadePath::Fragment`] (tile-based mobile GPUs, M1c) draws one
//! fullscreen triangle into it as a colour attachment, running the same `shade_pixel`; the tests
//! run both on lavapipe and compare them. The tier chooses ([`ShadePath::for_tier`]).
//!
//! # A shading pass alone
//!
//! A change of material or of the wireframe needs no raster: `Shade::encode_again` runs the
//! pass over the visibility buffer last shaded, with the bind groups it kept, the material's
//! new settings and the same view. Its cost is one pass over the pixels, independent of the mesh
//! (spec target, "Light or material change").

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, Color,
    ColorTargetState, ColorWrites, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Extent3d, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, StoreOp, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};

use crate::camera::{Vec3, View};
use crate::cull::{CULL_WGSL, SlotBuffers, entry, storage_buffer, uniform_buffer};
use crate::gpu::{Gpu, GpuError};
use crate::indices::INDICES_WGSL;
use crate::lod::CLUSTER_WGSL;
use crate::material::{
    MATERIAL_GROUP, MatCap, MatCapStyle, Material, SHADING_STAGES, SURFACE_WGSL,
};
use crate::raster::{
    MORPH_WGSL, PLACE_WGSL, QUAD_WGSL, Raster, RasterError, slot_group, slot_source,
};
use crate::residency::{SlotLayout, Tier};
use crate::tree::NODE_WGSL;
use crate::upload::{VERTEX_BYTES, VERTEX_WGSL};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests;

/// The shading pass's own source, after the declarations it uses (see its header).
const SHADE_WGSL: &str = include_str!("shade.wgsl");

/// The format the pass shades into: linear, 16-bit float, HDR.
pub(crate) const HDR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Storage buffers the pass binds itself beside the slot buffers and its material's: the tree,
/// the remap tables, the stamps and the drawn edges.
const OWN_STORAGE: u32 = 4;

/// Uniform buffers it binds itself beside its material's: the raster's settings and view, and
/// its own settings.
const OWN_UNIFORMS: u32 = 3;

/// The storage and uniform buffers the pass binds beside the slot buffers with material `M`:
/// its own and the material's ([`Material::STORAGE_BUFFERS`], [`Material::UNIFORM_BUFFERS`]).
pub(crate) const fn fixed_buffers<M: Material>() -> (u32, u32) {
    (
        OWN_STORAGE + M::STORAGE_BUFFERS,
        OWN_UNIFORMS + M::UNIFORM_BUFFERS,
    )
}

/// Pixels a side of a compute workgroup.
const GROUP_SIDE: u32 = 8;

/// The wireframe's lines' width, pixels.
pub(crate) const WIRE_WIDTH: f32 = 1.0;

/// The wireframe's colour, linear: a dark slate, readable over every MatCap.
const WIRE_COLOUR: [f32; 3] = [0.012, 0.016, 0.024];

/// How the pass runs: a compute pass or a fullscreen fragment pass (spec, "Frame pipeline", 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadePath {
    /// One compute invocation a pixel, writing the target as a storage image: desktop GPUs.
    Compute,
    /// A fullscreen triangle drawn into the target: tile-based mobile GPUs, which keep a
    /// fragment pass's target on chip. Exercised on lavapipe; finished in M1c.
    Fragment,
}

impl ShadePath {
    /// The path for `tier` (spec, "Capability tiers"): compute on desktop, fragment on mobile.
    #[must_use]
    pub const fn for_tier(tier: Tier) -> Self {
        match tier {
            Tier::Desktop => Self::Compute,
            Tier::MobileHigh => Self::Fragment,
        }
    }
}

/// What the shading shows beside its material's own settings (`Material::Settings`): a
/// change of these alone needs only the shading pass (`Shade::encode_again`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShadeSettings {
    /// Whether the wireframe is drawn over the shading.
    pub wireframe: bool,
}

/// The pass's settings as the GPU reads them: WGSL's `ShadeParams`, 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct GpuShadeParams {
    /// The view's up, `w` 0.
    up: [f32; 4],
    /// The view's right, `w` 0.
    right: [f32; 4],
    /// The wireframe's colour and its width, pixels.
    wire: [f32; 4],
    /// [`WIREFRAME`].
    flags: u32,
    /// Zero.
    reserved: [u32; 3],
}

/// [`GpuShadeParams::flags`]: draw the wireframe.
const WIREFRAME: u32 = 1;

/// Why the pass couldn't be set up or encoded.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ShadeError {
    /// The GPU failed: a buffer, a pipeline, a bind group, the target.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The slot buffers don't fit the layout the pass was built for.
    #[error(transparent)]
    Slots(#[from] RasterError),
    /// The slot layout's buffers don't fit beside the pass's own and its material's.
    #[error("{buffers} slot buffers: the shading pass binds at most {limit} beside its own")]
    TooManySlotBuffers {
        /// The slot buffers.
        buffers: u32,
        /// The most the device allows beside the pass's own.
        limit: u32,
    },
}

/// What a frame's shading reads: the scene the raster drew.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShadeScene<'a> {
    /// The tile tree's buffer.
    pub(crate) tree: &'a Buffer,
    /// The slot buffers the tree's slots name.
    pub(crate) slots: &'a SlotBuffers,
    /// The raster, its frame drawn: its visibility buffer, stamps, drawn edges and settings.
    pub(crate) raster: &'a Raster,
    /// The frame's view (the raster's).
    pub(crate) view: &'a View,
}

/// The pipeline, by path.
#[derive(Debug)]
enum Pipeline {
    /// [`ShadePath::Compute`].
    Compute(ComputePipeline),
    /// [`ShadePath::Fragment`].
    Fragment(RenderPipeline),
}

/// The target, and what binds it.
#[derive(Debug)]
struct Target {
    /// The image.
    texture: Texture,
    /// A view of it.
    view: TextureView,
    /// Group 3, the compute path's storage image; `None` on the fragment path.
    output: Option<BindGroup>,
}

/// The last scene shaded: its bind groups, kept for [`Shade::encode_again`].
#[derive(Debug)]
struct Bound {
    /// Group 0: the frame's buffers and the visibility buffer.
    frame: BindGroup,
    /// Group 1: the slot buffers.
    slots: BindGroup,
    /// The visibility buffer's size.
    size: (u32, u32),
    /// The view's up and right.
    basis: ([f32; 4], [f32; 4]),
}

/// The shading pass: its pipeline (for the slot layout it was built for, and its material), its
/// settings and its target.
#[derive(Debug)]
pub(crate) struct Shade<M: Material = MatCap> {
    /// How it runs.
    path: ShadePath,
    /// The material.
    material: M,
    /// The slot buffers' layout it binds.
    slots: SlotLayout,
    /// Group 0's layout.
    layout: BindGroupLayout,
    /// Group 1's layout: one binding a slot buffer.
    slot_layout: BindGroupLayout,
    /// Group 3's layout: the storage image (compute only).
    output_layout: BindGroupLayout,
    /// The pipeline.
    pipeline: Pipeline,
    /// [`GpuShadeParams`].
    params: Buffer,
    /// A small buffer bound for each slot buffer of the layout not made yet.
    placeholder: Buffer,
    /// The wireframe and the style.
    settings: ShadeSettings,
    /// The target, once a frame was shaded.
    target: Option<Target>,
    /// The last scene shaded.
    bound: Option<Bound>,
    /// The shader source, for the tests' probes.
    #[cfg(test)]
    source: String,
}

impl Shade<MatCap> {
    /// The pass with a [`MatCap`] showing `style`, for slot buffers of `slots` (residency's, as
    /// the raster's), running as `path`.
    ///
    /// # Errors
    ///
    /// As [`Shade::with_material`] and [`MatCap::new`].
    pub(crate) fn new(
        gpu: &Gpu,
        slots: SlotLayout,
        path: ShadePath,
        style: MatCapStyle,
    ) -> Result<Self, ShadeError> {
        let material = MatCap::new(gpu, style)?;
        Self::with_material(gpu, slots, path, material)
    }
}

impl<M: Material> Shade<M> {
    /// Sets what the shading shows beside the material, from the next pass on (a uniform write:
    /// no new pipeline).
    pub(crate) fn set_settings(&mut self, settings: ShadeSettings) {
        self.settings = settings;
    }

    /// Sets the material's settings, from the next pass on (its own buffers: no new pipeline).
    pub(crate) fn set_material(&mut self, gpu: &Gpu, settings: M::Settings) {
        self.material.set_settings(gpu, settings);
    }

    /// The pass shading through `material`, for slot buffers of `slots`, running as `path`.
    ///
    /// # Errors
    ///
    /// [`ShadeError::Gpu`] if a buffer, layout or pipeline is refused.
    pub(crate) fn with_material(
        gpu: &Gpu,
        slots: SlotLayout,
        path: ShadePath,
        material: M,
    ) -> Result<Self, ShadeError> {
        Self::build(gpu, slots, path, material, &[])
    }

    /// [`Shade::with_material`] with text replacements in the shader, for the tests' bite
    /// checks.
    ///
    /// # Errors
    ///
    /// As [`Shade::with_material`].
    #[cfg(test)]
    pub(crate) fn patched(
        gpu: &Gpu,
        slots: SlotLayout,
        path: ShadePath,
        material: M,
        patches: &[(&str, &str)],
    ) -> Result<Self, ShadeError> {
        Self::build(gpu, slots, path, material, patches)
    }

    /// [`Shade::with_material`], `patches` made in the shader.
    ///
    /// # Errors
    ///
    /// As [`Shade::with_material`].
    fn build(
        gpu: &Gpu,
        slots: SlotLayout,
        path: ShadePath,
        material: M,
        patches: &[(&str, &str)],
    ) -> Result<Self, ShadeError> {
        let (storage, uniforms) = fixed_buffers::<M>();
        let limit = crate::raster::slots_beside(gpu, storage, uniforms);
        if slots.total() > limit {
            return Err(ShadeError::TooManySlotBuffers {
                buffers: slots.total(),
                limit,
            });
        }
        let layouts = Layouts::new(gpu, slots)?;
        let source = shade_source(slots, material.source(), patches);
        let pipeline = layouts.pipeline(gpu, path, &source, material.layout())?;
        let params = uniform_buffer(gpu, "shade params", size_of::<GpuShadeParams>() as u64)?;
        let placeholder = storage_buffer(gpu, "shade slot placeholder", VERTEX_BYTES)?;
        let Layouts {
            frame: layout,
            slots: slot_layout,
            output: output_layout,
        } = layouts;
        Ok(Self {
            path,
            material,
            slots,
            layout,
            slot_layout,
            output_layout,
            pipeline,
            params,
            placeholder,
            settings: ShadeSettings::default(),
            target: None,
            bound: None,
            #[cfg(test)]
            source,
        })
    }

    /// How the pass runs.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn path(&self) -> ShadePath {
        self.path
    }

    /// The target: [`HDR_FORMAT`], the size of the visibility buffer last shaded; `None` before
    /// the first pass.
    #[must_use]
    pub(crate) fn target(&self) -> Option<&Texture> {
        self.target.as_ref().map(|target| &target.texture)
    }

    /// Encodes the pass over `scene`'s visibility buffer, after the raster's draws in the same
    /// submission (the settings go through the queue: one pass a submission), and keeps its bind
    /// groups for [`Shade::encode_again`].
    ///
    /// # Errors
    ///
    /// [`ShadeError::Slots`] if the slot buffers don't fit the layout the pass was built for;
    /// [`ShadeError::Gpu`] if a bind group or the target is refused.
    pub(crate) fn encode(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
        scene: &ShadeScene<'_>,
    ) -> Result<(), ShadeError> {
        let visibility = scene.raster.visibility();
        let size = (visibility.width(), visibility.height());
        let placing = scene.raster.placing();
        let slots = slot_group(
            gpu,
            &self.slot_layout,
            self.slots,
            scene.slots,
            &self.placeholder,
        )?;
        let frame = gpu.scoped("shade bind group", || {
            let ids = visibility.create_view(&TextureViewDescriptor::default());
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("shade"),
                layout: &self.layout,
                entries: &[
                    entry(0, scene.tree),
                    entry(1, placing.remap),
                    entry(2, placing.stamps),
                    entry(3, placing.drawn_edges),
                    entry(4, placing.params),
                    entry(5, placing.view),
                    BindGroupEntry {
                        binding: 6,
                        resource: BindingResource::TextureView(&ids),
                    },
                    entry(7, &self.params),
                ],
            })
        })?;
        let basis = (direction(scene.view.up()), direction(scene.view.right()));
        self.bound = Some(Bound {
            frame,
            slots,
            size,
            basis,
        });
        self.encode_again(gpu, encoder)?;
        Ok(())
    }

    /// Encodes the pass again over the visibility buffer last shaded, with the settings and
    /// material as they are now: the shading alone, no raster (a material or wireframe change
    /// with the scene unchanged). Returns whether there was a scene to shade.
    ///
    /// # Errors
    ///
    /// [`ShadeError::Gpu`] if the target is refused.
    pub(crate) fn encode_again(
        &mut self,
        gpu: &Gpu,
        encoder: &mut CommandEncoder,
    ) -> Result<bool, ShadeError> {
        let Some(bound) = &self.bound else {
            return Ok(false);
        };
        let size = bound.size;
        let (up, right) = bound.basis;
        let params = GpuShadeParams {
            up,
            right,
            wire: [WIRE_COLOUR[0], WIRE_COLOUR[1], WIRE_COLOUR[2], WIRE_WIDTH],
            flags: if self.settings.wireframe {
                WIREFRAME
            } else {
                0
            },
            reserved: [0; 3],
        };
        gpu.queue()
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let target = match self.target.take() {
            Some(target) if (target.texture.width(), target.texture.height()) == size => target,
            _ => self.new_target(gpu, size)?,
        };
        match &self.pipeline {
            Pipeline::Compute(pipeline) => {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("shade"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bound.frame, &[]);
                pass.set_bind_group(1, &bound.slots, &[]);
                pass.set_bind_group(MATERIAL_GROUP, self.material.bind_group(), &[]);
                pass.set_bind_group(3, target.output.as_ref(), &[]);
                pass.dispatch_workgroups(
                    size.0.div_ceil(GROUP_SIDE),
                    size.1.div_ceil(GROUP_SIDE),
                    1,
                );
            }
            Pipeline::Fragment(pipeline) => {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("shade"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &target.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::TRANSPARENT),
                            store: StoreOp::Store,
                        },
                    })],
                    ..RenderPassDescriptor::default()
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bound.frame, &[]);
                pass.set_bind_group(1, &bound.slots, &[]);
                pass.set_bind_group(MATERIAL_GROUP, self.material.bind_group(), &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.target = Some(target);
        Ok(true)
    }

    /// A target of `size`, used as the pass's path writes it: a storage image on the compute
    /// path, a colour attachment on the fragment path; read by the tone map, and copied out by
    /// the tests.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new_target(&self, gpu: &Gpu, size: (u32, u32)) -> Result<Target, GpuError> {
        let device = gpu.device();
        let written = match self.path {
            ShadePath::Compute => TextureUsages::STORAGE_BINDING,
            ShadePath::Fragment => TextureUsages::RENDER_ATTACHMENT,
        };
        gpu.scoped("shade target", || {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("shaded"),
                size: Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: HDR_FORMAT,
                usage: written | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let output = (self.path == ShadePath::Compute).then(|| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("shade output"),
                    layout: &self.output_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    }],
                })
            });
            Target {
                texture,
                view,
                output,
            }
        })
    }
}

/// The pass's own bind group layouts.
#[derive(Debug)]
struct Layouts {
    /// Group 0: the frame's buffers and the visibility buffer.
    frame: BindGroupLayout,
    /// Group 1: one binding a slot buffer.
    slots: BindGroupLayout,
    /// Group 3: the storage image (compute only).
    output: BindGroupLayout,
}

impl Layouts {
    /// The layouts for slot buffers of `slots`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn new(gpu: &Gpu, slots: SlotLayout) -> Result<Self, GpuError> {
        let device = gpu.device();
        let read = |binding| BindGroupLayoutEntry {
            binding,
            visibility: SHADING_STAGES,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
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
            ..read(binding)
        };
        let ids = BindGroupLayoutEntry {
            binding: 6,
            visibility: SHADING_STAGES,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let frame = gpu.scoped("shade layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("shade"),
                entries: &[
                    read(0),
                    read(1),
                    read(2),
                    read(3),
                    uniform(4),
                    uniform(5),
                    ids,
                    uniform(7),
                ],
            })
        })?;
        let slot_entries = (0..slots.total()).map(read);
        let slot_entries: Vec<BindGroupLayoutEntry> = slot_entries.collect();
        let slot_layout = gpu.scoped("shade slot layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("shade slots"),
                entries: &slot_entries,
            })
        })?;
        let output = gpu.scoped("shade output layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("shade output"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            })
        })?;
        Ok(Self {
            frame,
            slots: slot_layout,
            output,
        })
    }

    /// The pipeline running `source` as `path`, the material's group on `material`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn pipeline(
        &self,
        gpu: &Gpu,
        path: ShadePath,
        source: &str,
        material: &BindGroupLayout,
    ) -> Result<Pipeline, GpuError> {
        let device = gpu.device();
        gpu.scoped("shade pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("shade"),
                source: ShaderSource::Wgsl(source.into()),
            });
            let mut groups = vec![Some(&self.frame), Some(&self.slots), Some(material)];
            if path == ShadePath::Compute {
                groups.push(Some(&self.output));
            }
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("shade"),
                bind_group_layouts: &groups,
                immediate_size: 0,
            });
            match path {
                ShadePath::Compute => {
                    Pipeline::Compute(device.create_compute_pipeline(&ComputePipelineDescriptor {
                        label: Some("shade"),
                        layout: Some(&layout),
                        module: &module,
                        entry_point: Some("shade_compute"),
                        compilation_options: PipelineCompilationOptions::default(),
                        cache: None,
                    }))
                }
                ShadePath::Fragment => {
                    Pipeline::Fragment(device.create_render_pipeline(&RenderPipelineDescriptor {
                        label: Some("shade"),
                        layout: Some(&layout),
                        vertex: VertexState {
                            module: &module,
                            entry_point: Some("fullscreen"),
                            compilation_options: PipelineCompilationOptions::default(),
                            buffers: &[],
                        },
                        primitive: PrimitiveState {
                            topology: PrimitiveTopology::TriangleList,
                            ..PrimitiveState::default()
                        },
                        depth_stencil: None,
                        multisample: MultisampleState::default(),
                        fragment: Some(FragmentState {
                            module: &module,
                            entry_point: Some("shade_fragment"),
                            compilation_options: PipelineCompilationOptions::default(),
                            targets: &[Some(ColorTargetState {
                                format: HDR_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    }))
                }
            }
        })
    }
}

/// The pass's shader for slot buffers of `slots` and a material of `material` source,
/// `patches` made (only the tests make any).
///
/// # Panics
///
/// If a patch doesn't apply.
fn shade_source(slots: SlotLayout, material: &str, patches: &[(&str, &str)]) -> String {
    let source = format!(
        "{NODE_WGSL}{VERTEX_WGSL}{CULL_WGSL}{CLUSTER_WGSL}{INDICES_WGSL}{MORPH_WGSL}{}\
         {PLACE_WGSL}{QUAD_WGSL}{SURFACE_WGSL}{material}{SHADE_WGSL}",
        slot_source(&slots),
    );
    patches.iter().fold(source, |source, (from, to)| {
        assert!(source.contains(from), "the patch {from:?} applies");
        source.replace(from, to)
    })
}

/// `v` as the GPU takes a direction: `f32`, `w` 0.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the GPU works in f32: a unit direction rounded once"
)]
const fn direction(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}
