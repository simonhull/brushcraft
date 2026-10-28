//! The overlay pass, the frame's last: a one-byte-per-texel text image drawn over the frame's
//! top-left corner at a whole-number scale, with a translucent backing so it reads over any
//! scene.
//!
//! The caller lays the text out ([`TextImage`]: ink where a glyph is, zero on the backing);
//! the app does that with its own bitmap font. This pass only uploads the image, places it
//! and shades it (`overlay.wgsl`).

use std::fmt;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, BlendState, Buffer,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Extent3d, FragmentState,
    MultisampleState, Origin3d, PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, VertexState,
};

use crate::gpu::{Gpu, GpuError};
use crate::surface::Size;

/// The overlay's offset from the frame's top-left corner, in texels of the text image (frame
/// pixels at scale 1).
pub const MARGIN: u32 = 8;

/// A laid-out block of text: one byte per texel, row by row from the top, nonzero where a
/// glyph has ink (drawn at full ink from 255 down to none at 0), zero on the backing.
#[derive(Clone, PartialEq, Eq)]
pub struct TextImage {
    /// Texels across.
    width: u32,
    /// Texels down.
    height: u32,
    /// The texels, `width × height` of them.
    texels: Vec<u8>,
}

impl fmt::Debug for TextImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl TextImage {
    /// An image of `width × height` texels; `None` unless `texels` holds exactly that many.
    #[must_use]
    pub fn new(width: u32, height: u32, texels: Vec<u8>) -> Option<Self> {
        let count = usize::try_from(u64::from(width) * u64::from(height)).ok()?;
        (texels.len() == count).then_some(Self {
            width,
            height,
            texels,
        })
    }

    /// Texels across.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Texels down.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The texel at column `x`, row `y`; `None` outside the image.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x)).ok()?;
        self.texels.get(index).copied()
    }
}

/// The text image on the GPU, and the bind group that reads it.
#[derive(Debug)]
struct Uploaded {
    /// The texture: at least the image's size, grown as the image does.
    texture: Texture,
    /// The placement uniforms and the texture, bound for the shader.
    bind_group: BindGroup,
}

/// The overlay pass: its pipeline, its placement uniforms and the text image's texture.
#[derive(Debug)]
pub(crate) struct OverlayPass {
    /// Draws the quad.
    pipeline: RenderPipeline,
    /// The shader's `Placement`, written each frame.
    placement: Buffer,
    /// The image's texture, once one has been uploaded.
    uploaded: Option<Uploaded>,
    /// Whether this frame's image is uploaded and should be drawn.
    ready: bool,
}

/// Bytes of the shader's `Placement`: eight `f32`.
const PLACEMENT_BYTES: u64 = 32;

impl OverlayPass {
    /// The overlay's pipeline, drawing into frames of `format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the shader or pipeline is refused.
    pub(crate) fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        let device = gpu.device();
        gpu.scoped("overlay pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("overlay"),
                source: ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("overlay"),
                layout: None,
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("quad"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..PrimitiveState::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("shade"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            let placement = device.create_buffer(&BufferDescriptor {
                label: Some("overlay placement"),
                size: PLACEMENT_BYTES,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Self {
                pipeline,
                placement,
                uploaded: None,
                ready: false,
            }
        })
    }

    /// Uploads `image` and places it at the top-left of a frame of `frame` size, `scale` frame
    /// pixels to the texel (at least 1); with no image, or an empty one, nothing is drawn this
    /// frame.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the texture can't be created or written.
    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        image: Option<&TextImage>,
        frame: Size,
        scale: u32,
    ) -> Result<(), GpuError> {
        self.ready = false;
        let Some(image) = image.filter(|image| image.width > 0 && image.height > 0) else {
            return Ok(());
        };
        let (width, height) = (image.width, image.height);
        let grow = self.uploaded.as_ref().is_none_or(|uploaded| {
            let size = uploaded.texture.size();
            size.width < width || size.height < height
        });
        if grow {
            self.uploaded = Some(self.upload_target(gpu, width, height)?);
        }
        let Some(uploaded) = &self.uploaded else {
            return Ok(());
        };
        let scale = scale.max(1);
        #[expect(
            clippy::cast_precision_loss,
            reason = "frame and image sizes are far below 2^24, exact in f32"
        )]
        let placement: [f32; 8] = [
            (MARGIN * scale) as f32,
            (MARGIN * scale) as f32,
            (width * scale) as f32,
            (height * scale) as f32,
            frame.width() as f32,
            frame.height() as f32,
            scale as f32,
            0.0,
        ];
        gpu.scoped("overlay upload", || {
            gpu.queue()
                .write_buffer(&self.placement, 0, bytemuck::cast_slice(&placement));
            gpu.queue().write_texture(
                TexelCopyTextureInfo {
                    texture: &uploaded.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &image.texels,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(height),
                },
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        })?;
        self.ready = true;
        Ok(())
    }

    /// A texture for a text image of at least `width × height`, rounded up so a slowly
    /// growing image doesn't make one each frame, and its bind group.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn upload_target(&self, gpu: &Gpu, width: u32, height: u32) -> Result<Uploaded, GpuError> {
        let device = gpu.device();
        gpu.scoped("overlay texture", || {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some("overlay text"),
                size: Extent3d {
                    width: width.next_multiple_of(64),
                    height: height.next_multiple_of(16),
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
                label: Some("overlay"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.placement.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&view),
                    },
                ],
            });
            Uploaded {
                texture,
                bind_group,
            }
        })
    }

    /// Draws this frame's overlay into `pass`, if [`OverlayPass::prepare`] uploaded one.
    pub(crate) fn draw(&self, pass: &mut RenderPass<'_>) {
        if let (true, Some(uploaded)) = (self.ready, &self.uploaded) {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &uploaded.bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}
