//! The UI pass: the frame's last, drawing what the `ui` crate hands over, in display colour.
//!
//! Frames are sRGB, so blending on them happens in linear light: glass at 66 % over bright clay
//! comes out far lighter than the mockups, and white text on it falls under 4.5:1. So the pass
//! draws on a view of the frame that isn't sRGB ([`crate::surface::Frame::display_view`]), where
//! blending happens in the display values the UI's colours are written in. Where a surface can't
//! offer that view the pass draws on the sRGB view ([`Blend::Linear`]) and the shader decodes the
//! colours. The `ui` crate's list then already holds the theme for that blend, its glass raised,
//! because the app asks [`UiPass::blend_on`] when it makes its `ui::Ui`: this pass substitutes
//! nothing.
//!
//! Rounded rectangles are one instanced draw, shaded by distance to the edge, and glyphs one draw
//! from the atlas (`ui.wgsl`). Instances upload only when the list changes and the atlas only when
//! its version does; fades are computed from the time and written each frame, so a fade costs the
//! owner thread nothing. With nothing to draw, the frame encodes no pass at all.

use std::sync::Arc;
use std::time::Instant;

use ::ui::{AtlasImage, Blend, Colour, DrawList, UiFrame};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendState, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Extent3d, FragmentState,
    MultisampleState, Origin3d, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension, VertexAttribute, VertexBufferLayout, VertexState, VertexStepMode,
    vertex_attr_array,
};

use crate::gpu::{Gpu, GpuError};
use crate::surface::{Size, Target};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The least an instance buffer holds, bytes.
const FIRST_BUFFER_BYTES: u64 = 4096;

/// The shader's `RectIn`.
const RECT_ATTRIBUTES: [VertexAttribute; 4] = vertex_attr_array![
    0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4
];

/// The shader's `GlyphIn`.
const GLYPH_ATTRIBUTES: [VertexAttribute; 4] = vertex_attr_array![
    0 => Float32x4, 1 => Float32x2, 2 => Float32x4, 3 => Float32
];

/// The shader's `View`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    /// The frame's size.
    size: [f32; 2],
    /// 1 where the shader must hand linear colours to an sRGB view.
    linear: f32,
    /// Pads the uniform.
    pad: f32,
    /// The opacity of each layer.
    fades: [f32; 4],
}

/// The shader's `RectIn`: one rounded rectangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RectInstance {
    /// x, y, width and height.
    rect: [f32; 4],
    /// Radius, stroke width, layer, and one unused.
    shape: [f32; 4],
    /// The fill, 0 to 1, straight alpha.
    fill: [f32; 4],
    /// The stroke, likewise.
    stroke: [f32; 4],
}

/// The shader's `GlyphIn`: one glyph.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GlyphInstance {
    /// x, y, width and height.
    rect: [f32; 4],
    /// Its top left texel.
    atlas: [f32; 2],
    /// Its colour, 0 to 1, straight alpha.
    colour: [f32; 4],
    /// Its layer.
    layer: f32,
}

/// `colour` as the shader takes it.
fn channels(colour: Colour) -> [f32; 4] {
    [colour.r, colour.g, colour.b, colour.a].map(|channel| f32::from(channel) / 255.0)
}

/// The list's rectangles, as instances.
fn rect_instances(list: &DrawList) -> Vec<RectInstance> {
    let instances = list.rects.iter().map(|shape| RectInstance {
        rect: [shape.rect.x, shape.rect.y, shape.rect.w, shape.rect.h],
        shape: [
            shape.radius,
            shape.stroke_width,
            f32::from(shape.layer as u8),
            0.0,
        ],
        fill: channels(shape.fill),
        stroke: channels(shape.stroke),
    });
    instances.collect()
}

/// The list's glyphs, as instances.
fn glyph_instances(list: &DrawList) -> Vec<GlyphInstance> {
    let instances = list.glyphs.iter().map(|glyph| {
        #[expect(
            clippy::cast_precision_loss,
            reason = "texels of an atlas at most 2048 across, exact in f32"
        )]
        let atlas = [glyph.atlas.0 as f32, glyph.atlas.1 as f32];
        GlyphInstance {
            rect: [glyph.rect.x, glyph.rect.y, glyph.rect.w, glyph.rect.h],
            atlas,
            colour: channels(glyph.colour),
            layer: f32::from(glyph.layer as u8),
        }
    });
    instances.collect()
}

/// A vertex buffer of instances, doubling as it must.
#[derive(Debug)]
struct Instances {
    /// What it's called.
    label: &'static str,
    /// The buffer.
    buffer: Buffer,
    /// How many instances it holds now.
    count: u32,
}

impl Instances {
    /// An empty buffer of the first size.
    fn new(gpu: &Gpu, label: &'static str) -> Self {
        Self {
            label,
            buffer: instance_buffer(gpu, label, FIRST_BUFFER_BYTES),
            count: 0,
        }
    }

    /// Replaces the instances with `instances`.
    fn write<T: Pod>(&mut self, gpu: &Gpu, instances: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(instances);
        let needed = bytes.len() as u64;
        if needed > self.buffer.size() {
            let size = needed.next_power_of_two();
            self.buffer = instance_buffer(gpu, self.label, size);
        }
        gpu.queue().write_buffer(&self.buffer, 0, bytes);
        self.count = u32::try_from(instances.len()).unwrap_or(u32::MAX);
    }
}

/// A vertex buffer of `size` bytes.
fn instance_buffer(gpu: &Gpu, label: &str, size: u64) -> Buffer {
    gpu.device().create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The atlas on the GPU, and the bind group that reads it.
#[derive(Debug)]
struct Uploaded {
    /// The texture, of the atlas's size.
    texture: Texture,
    /// The atlas version it holds.
    version: u64,
    /// The view uniforms and the texture, bound for the shader.
    bind_group: BindGroup,
}

/// The pipelines for one target format.
#[derive(Debug)]
struct Pipelines {
    /// Draws the rounded rectangles.
    rects: RenderPipeline,
    /// Draws the glyphs.
    glyphs: RenderPipeline,
}

/// What the pass has done, for tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UiCounts {
    /// Passes drawn.
    pub(crate) drawn: u64,
    /// Lists uploaded.
    pub(crate) lists: u64,
    /// Atlases uploaded.
    pub(crate) atlases: u64,
}

/// The UI pass: its pipelines, its uniforms, and the list and atlas it last uploaded.
#[derive(Debug)]
pub(crate) struct UiPass {
    /// The shader.
    module: ShaderModule,
    /// What both pipelines bind.
    layout: PipelineLayout,
    /// The bind group's layout.
    bind_layout: BindGroupLayout,
    /// The frames' format, which may be sRGB.
    format: TextureFormat,
    /// The pipelines for the display view.
    display: Pipelines,
    /// The pipelines for the frame's sRGB view, made the first time a surface needs them.
    linear: Option<Pipelines>,
    /// How this frame blends.
    blend: Blend,
    /// The shader's `View`, written each frame.
    view: Buffer,
    /// The atlas, once uploaded.
    atlas: Option<Uploaded>,
    /// The list uploaded, held so a new list can't be mistaken for it.
    list: Option<Arc<DrawList>>,
    /// The rectangles of the list.
    rects: Instances,
    /// The glyphs of the list.
    glyphs: Instances,
    /// What it has uploaded.
    #[cfg(test)]
    counts: UiCounts,
    /// The passes it has drawn.
    #[cfg(test)]
    drawn: std::cell::Cell<u64>,
}

impl UiPass {
    /// The UI's pipelines, drawing into frames of `format` through its display view.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the shader or a pipeline is refused.
    pub(crate) fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        let device = gpu.device();
        gpu.scoped("ui pipelines", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("ui"),
                source: ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
            });
            let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("ui"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ui"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });
            let display = pipelines(gpu, &module, &layout, format.remove_srgb_suffix());
            let view = device.create_buffer(&BufferDescriptor {
                label: Some("ui view"),
                size: size_of::<ViewUniform>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Self {
                rects: Instances::new(gpu, "ui rects"),
                glyphs: Instances::new(gpu, "ui glyphs"),
                module,
                layout,
                bind_layout,
                format,
                display,
                linear: None,
                blend: Blend::Display,
                view,
                atlas: None,
                list: None,
                #[cfg(test)]
                counts: UiCounts::default(),
                #[cfg(test)]
                drawn: std::cell::Cell::new(0),
            }
        })
    }

    /// How frames of `target` blend: what a `ui::Ui` drawn on it must be told.
    pub(crate) const fn blend_on(target: &Target) -> Blend {
        if target.has_display_view() {
            Blend::Display
        } else {
            Blend::Linear
        }
    }

    /// Gets this frame's UI ready to draw at `now` in a frame of `size`, blending as `blend` says:
    /// the fades, and the list and the atlas if they changed.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if a pipeline, texture or buffer is refused.
    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        frame: &UiFrame,
        now: Instant,
        size: Size,
        blend: Blend,
    ) -> Result<(), GpuError> {
        self.blend = blend;
        gpu.scoped("ui upload", || {
            if blend == Blend::Linear && self.linear.is_none() {
                self.linear = Some(pipelines(gpu, &self.module, &self.layout, self.format));
            }
            self.write_view(gpu, frame, now, size);
            self.upload_atlas(gpu, &frame.atlas);
            let current = self.list.as_ref();
            if current.is_none_or(|list| !Arc::ptr_eq(list, &frame.list)) {
                self.rects.write(gpu, &rect_instances(&frame.list));
                self.glyphs.write(gpu, &glyph_instances(&frame.list));
                self.list = Some(Arc::clone(&frame.list));
                #[cfg(test)]
                {
                    self.counts.lists += 1;
                }
            }
        })
    }

    /// Writes the shader's `View`.
    fn write_view(&self, gpu: &Gpu, frame: &UiFrame, now: Instant, size: Size) {
        let [bar, chip] = frame.fades.map(|fade| fade.at(now));
        let linear = match self.blend {
            Blend::Display => 0.0,
            Blend::Linear => 1.0,
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "a frame's size is far below 2^24, exact in f32"
        )]
        let uniform = ViewUniform {
            size: [size.width() as f32, size.height() as f32],
            linear,
            pad: 0.0,
            fades: [bar, chip, 0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.view, 0, bytemuck::bytes_of(&uniform));
    }

    /// Uploads `atlas` if it isn't what the GPU holds, making a texture if its size changed.
    fn upload_atlas(&mut self, gpu: &Gpu, atlas: &AtlasImage) {
        let held = self.atlas.as_ref();
        if held.is_some_and(|held| held.version == atlas.version) {
            return;
        }
        let side = atlas.size;
        if held.is_none_or(|held| held.texture.size().width != side) {
            self.atlas = Some(self.atlas_texture(gpu, side, atlas.version));
        }
        let Some(uploaded) = &mut self.atlas else {
            return;
        };
        gpu.queue().write_texture(
            TexelCopyTextureInfo {
                texture: &uploaded.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &atlas.texels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side),
                rows_per_image: Some(side),
            },
            Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
        uploaded.version = atlas.version;
        #[cfg(test)]
        {
            self.counts.atlases += 1;
        }
    }

    /// A texture for an atlas `side` texels square, and its bind group.
    fn atlas_texture(&self, gpu: &Gpu, side: u32, version: u64) -> Uploaded {
        let device = gpu.device();
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("ui atlas"),
            size: Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ui"),
            layout: &self.bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.view.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&view),
                },
            ],
        });
        Uploaded {
            texture,
            version,
            bind_group,
        }
    }

    /// Draws this frame's UI into `pass`, a render pass of its own on the view
    /// [`UiPass::prepare`] was told of.
    pub(crate) fn draw(&self, pass: &mut RenderPass<'_>) {
        let (Some(atlas), Some(pipelines)) = (&self.atlas, self.pipelines()) else {
            return;
        };
        pass.set_bind_group(0, &atlas.bind_group, &[]);
        if self.rects.count > 0 {
            pass.set_pipeline(&pipelines.rects);
            pass.set_vertex_buffer(0, self.rects.buffer.slice(..));
            pass.draw(0..4, 0..self.rects.count);
        }
        if self.glyphs.count > 0 {
            pass.set_pipeline(&pipelines.glyphs);
            pass.set_vertex_buffer(0, self.glyphs.buffer.slice(..));
            pass.draw(0..4, 0..self.glyphs.count);
        }
        #[cfg(test)]
        self.drawn.set(self.drawn.get() + 1);
    }

    /// The pipelines this frame blends with.
    const fn pipelines(&self) -> Option<&Pipelines> {
        match self.blend {
            Blend::Display => Some(&self.display),
            Blend::Linear => self.linear.as_ref(),
        }
    }

    /// What it has done.
    #[cfg(test)]
    pub(crate) fn counts(&self) -> UiCounts {
        UiCounts {
            drawn: self.drawn.get(),
            ..self.counts
        }
    }
}

/// The two pipelines, drawing premultiplied into `format`.
fn pipelines(
    gpu: &Gpu,
    module: &ShaderModule,
    layout: &PipelineLayout,
    format: TextureFormat,
) -> Pipelines {
    let pipeline = |label, vertex, fragment, stride, attributes: &[VertexAttribute]| {
        gpu.device()
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: VertexState {
                    module,
                    entry_point: Some(vertex),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(VertexBufferLayout {
                        array_stride: stride,
                        step_mode: VertexStepMode::Instance,
                        attributes,
                    })],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module,
                    entry_point: Some(fragment),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    };
    Pipelines {
        rects: pipeline(
            "ui rects",
            "rect_vertex",
            "rect_fragment",
            size_of::<RectInstance>() as u64,
            &RECT_ATTRIBUTES,
        ),
        glyphs: pipeline(
            "ui glyphs",
            "glyph_vertex",
            "glyph_fragment",
            size_of::<GlyphInstance>() as u64,
            &GLYPH_ATTRIBUTES,
        ),
    }
}
