//! The debug overlay: what the frame loop reports each frame, as lines of text drawn over the
//! frame's top-left corner in our own bitmap font ([`font`]).
//!
//! Three parts: [`lines`] says what the overlay shows, from a [`Snapshot`] of the frame loop;
//! [`TextImage`] lays the lines out into a one-byte-per-texel image on the CPU; and
//! [`OverlayPass`] uploads that image and draws it, scaled by a whole number, with a translucent
//! backing so it reads over any scene. No UI toolkit: the app starts with this overlay only.

mod font;

use std::time::Duration;

use render::gpu::{Gpu, GpuError};
use render::surface::{Presentation, Size};
use store::JobsReport;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, BlendState, Buffer,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, Extent3d, FragmentState,
    MultisampleState, Origin3d, PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, VertexState,
};

use crate::camera::{Camera, Vec3};
use crate::frame::{FeedStats, OwnerPhases};
use crate::render_thread::RenderPhases;

/// Everything the overlay reports about one frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Snapshot<'a> {
    /// The frame's number, from 1.
    pub(crate) frame: u64,
    /// The display's refresh rate the frame loop paces itself to, in hertz.
    pub(crate) refresh_hz: f64,
    /// How frames reach the screen.
    pub(crate) presentation: Presentation,
    /// The GPU's name and backend.
    pub(crate) adapter: &'a str,
    /// The frame's size.
    pub(crate) size: Size,
    /// How long this frame's phases took on the owner thread, up to the jobs.
    pub(crate) owner: OwnerPhases,
    /// How long the render thread took over the last frame it finished; `None` before it has
    /// finished one.
    pub(crate) render: Option<RenderPhases>,
    /// What the stale-tile feed has delivered so far.
    pub(crate) feed: FeedStats,
    /// What this frame's store jobs did.
    pub(crate) jobs: JobsReport,
    /// What the GPU holds (tiles per level, over-budget count, residency), from the uploads;
    /// `None` while nothing is uploaded.
    pub(crate) tiles: Option<&'a str>,
    /// The camera.
    pub(crate) camera: Camera,
}

/// `duration` in milliseconds, to two places.
fn ms(duration: Duration) -> String {
    format!("{:.2}", duration.as_secs_f64() * 1e3)
}

/// `v` as `(x, y, z)` to two places.
fn vec3(v: Vec3) -> String {
    format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z)
}

/// The overlay's lines for `snapshot`: plain ASCII, one fact per field.
pub(crate) fn lines(snapshot: &Snapshot<'_>) -> Vec<String> {
    let Snapshot {
        frame,
        refresh_hz,
        presentation,
        adapter,
        size,
        owner,
        render,
        feed,
        jobs,
        tiles,
        camera,
    } = *snapshot;
    let render_line = render.map_or_else(
        || "render ms  -".to_owned(),
        |phases| {
            format!(
                "render ms  acquire {}  encode {}  present {}",
                ms(phases.acquire),
                ms(phases.encode),
                ms(phases.present)
            )
        },
    );
    vec![
        format!(
            "BrushCraft  frame {frame}  {}x{}  {presentation} at {refresh_hz:.1} Hz  {adapter}",
            size.width(),
            size.height()
        ),
        format!(
            "owner ms   input {}  brush {}  feed {}  uploads {}  upkeep {}  jobs {}",
            ms(owner.input),
            ms(owner.brush),
            ms(owner.feed),
            ms(owner.uploads),
            ms(owner.upkeep),
            ms(owner.jobs)
        ),
        render_line,
        format!(
            "feed       {} epochs  {} tiles  {} resyncs",
            feed.epochs, feed.keys, feed.resyncs
        ),
        format!(
            "jobs       {} slices (longest {} ms)  {} built  {} display tiles left",
            jobs.slices,
            ms(jobs.longest_slice),
            jobs.built,
            jobs.left
        ),
        format!("tiles      {}", tiles.unwrap_or("none on the GPU yet")),
        format!(
            "camera     eye {}  target {}  distance {:.3}",
            vec3(camera.eye()),
            vec3(camera.target()),
            camera.distance()
        ),
    ]
}

/// Pixels between glyphs on a line.
const GLYPH_GAP: usize = 1;
/// Pixels between lines.
const LINE_GAP: usize = 2;
/// Pixels of backing around the text.
const PADDING: usize = 3;
/// The overlay's offset from the frame's top-left corner, in frame pixels.
const MARGIN: f32 = 8.0;

/// Lines of text laid out as an image: one byte per texel, 255 where a glyph has ink, 0 on the
/// backing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextImage {
    /// Texels across.
    width: usize,
    /// Texels down.
    height: usize,
    /// The texels, row by row from the top.
    texels: Vec<u8>,
}

impl TextImage {
    /// `lines` laid out top to bottom, each from the left edge, in [`font`]'s glyphs, with
    /// [`PADDING`] all round.
    pub(crate) fn new(lines: &[String]) -> Self {
        let advance = font::WIDTH + GLYPH_GAP;
        let line_height = font::HEIGHT + LINE_GAP;
        let longest = lines
            .iter()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0);
        let width = 2 * PADDING + (longest * advance).saturating_sub(GLYPH_GAP);
        let height = 2 * PADDING + (lines.len() * line_height).saturating_sub(LINE_GAP);
        let mut image = Self {
            width,
            height,
            texels: vec![0; width * height],
        };
        let numbered_lines = lines.iter().enumerate();
        for (row, line) in numbered_lines {
            let top = PADDING + row * line_height;
            let numbered_chars = line.chars().enumerate();
            for (column, c) in numbered_chars {
                image.stamp(PADDING + column * advance, top, font::glyph(c));
            }
        }
        image
    }

    /// Draws `glyph`'s ink with its top-left corner at texel (`left`, `top`); whatever falls
    /// outside the image is left out.
    fn stamp(&mut self, left: usize, top: usize, glyph: font::Glyph) {
        let glyph_rows = glyph.iter().enumerate();
        for (dy, bits) in glyph_rows {
            let start = (top + dy) * self.width + left;
            let Some(span) = self.texels.get_mut(start..start + font::WIDTH) else {
                continue;
            };
            let pixels = span.iter_mut().enumerate();
            for (dx, texel) in pixels {
                if bits & (1 << (font::WIDTH - 1 - dx)) != 0 {
                    *texel = u8::MAX;
                }
            }
        }
    }

    /// Texels across.
    #[cfg(test)]
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// Texels down.
    #[cfg(test)]
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    /// Whether the texel at column `x`, row `y` is ink; `false` outside the image.
    #[cfg(test)]
    pub(crate) fn ink(&self, x: usize, y: usize) -> bool {
        x < self.width && self.texels.get(y * self.width + x).is_some_and(|&t| t != 0)
    }
}

/// The overlay's shader: one quad over the overlay's rectangle, each pixel reading its texel
/// of the text image.
const SHADER: &str = r"
// The debug overlay: the text image drawn over the frame's corner at a whole-number scale,
// ink near-white and the backing translucent black, premultiplied.

// Where the overlay goes. All in frame pixels, but `scale`, frame pixels per texel.
struct Placement {
    // The overlay's top-left corner.
    origin: vec2<f32>,
    // The overlay's size: the image's times the scale.
    size: vec2<f32>,
    // The frame's size.
    frame: vec2<f32>,
    // Frame pixels per texel of the text image.
    scale: f32,
    // Unused: pads the struct to the 16-byte multiple uniforms need.
    pad: f32,
}

// Where the overlay goes, written each frame.
@group(0) @binding(0) var<uniform> placement: Placement;
// The text image: red 1 where a glyph has ink, 0 on the backing. Larger than the image when
// the image has shrunk; only the part `placement.size` covers is read.
@group(0) @binding(1) var text: texture_2d<f32>;

// The four corners of the overlay's rectangle as a triangle strip, in clip space.
@vertex
fn quad(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    let pixel = placement.origin + corner * placement.size;
    let clip = vec2<f32>(pixel.x / placement.frame.x * 2.0 - 1.0, 1.0 - pixel.y / placement.frame.y * 2.0);
    return vec4<f32>(clip, 0.0, 1.0);
}

// Each pixel's texel, unfiltered, as premultiplied ink over the backing.
@fragment
fn shade(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>((position.xy - placement.origin) / placement.scale);
    let ink = textureLoad(text, texel, 0).r;
    let backing = vec4<f32>(0.0, 0.0, 0.0, 0.6);
    let glyph = vec4<f32>(0.85, 0.85, 0.8, 1.0);
    return mix(backing, glyph, ink);
}
";

/// The text image on the GPU, and the bind group that reads it.
#[derive(Debug)]
struct Uploaded {
    /// The texture: at least the image's size, grown as the image does.
    texture: Texture,
    /// The placement uniforms and the texture, bound for the shader.
    bind_group: BindGroup,
}

/// The GPU side of the overlay: its pipeline, its placement uniforms and the text image.
#[derive(Debug)]
pub(crate) struct OverlayPass {
    /// Draws the quad.
    pipeline: RenderPipeline,
    /// The [`SHADER`]'s `Placement`, written each frame.
    placement: Buffer,
    /// The image's texture, once one has been prepared.
    uploaded: Option<Uploaded>,
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
                source: ShaderSource::Wgsl(SHADER.into()),
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
            }
        })
    }

    /// Uploads `image` and places it at the top-left of a frame of `frame` size, `scale` frame
    /// pixels to the texel (at least 1).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the texture can't be created or written.
    pub(crate) fn prepare(
        &mut self,
        gpu: &Gpu,
        image: &TextImage,
        frame: Size,
        scale: u32,
    ) -> Result<(), GpuError> {
        let (Ok(width), Ok(height)) = (u32::try_from(image.width), u32::try_from(image.height))
        else {
            // An image wider than the GPU could hold isn't one the overlay makes.
            return Ok(());
        };
        if width == 0 || height == 0 {
            return Ok(());
        }
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
            MARGIN * scale as f32,
            MARGIN * scale as f32,
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
        })
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

    /// Draws the prepared overlay into `pass`; nothing before [`OverlayPass::prepare`] has
    /// uploaded an image.
    pub(crate) fn draw(&self, pass: &mut RenderPass<'_>) {
        if let Some(uploaded) = &self.uploaded {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &uploaded.bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn text_is_laid_out_in_cells_inside_the_padding() {
        let image = TextImage::new(&["AB".to_owned(), "i".to_owned()]);
        // Two glyphs and a gap across, two lines and a gap down, padding all round.
        assert_eq!(image.width(), 2 * PADDING + 2 * font::WIDTH + GLYPH_GAP);
        assert_eq!(image.height(), 2 * PADDING + 2 * font::HEIGHT + LINE_GAP);
        // 'A' starts with a blank pixel then ink on its top row; 'B' with ink.
        assert!(!image.ink(PADDING, PADDING));
        assert!(image.ink(PADDING + 1, PADDING));
        assert!(image.ink(PADDING + font::WIDTH + GLYPH_GAP, PADDING), "B");
        // The gap column between them is blank all the way down.
        let gap = PADDING + font::WIDTH;
        let column = (0..image.height()).map(|y| image.ink(gap, y));
        assert!(!column.into_iter().any(|ink| ink), "the gap between glyphs");
        // 'i's dot sits on the second line's top row, in its middle column.
        let second = PADDING + font::HEIGHT + LINE_GAP;
        assert!(image.ink(PADDING + 2, second));
        // Nothing in the padding.
        let border =
            (0..image.width()).map(|x| image.ink(x, 0) || image.ink(x, image.height() - 1));
        assert!(!border.into_iter().any(|ink| ink), "the padding is backing");
    }

    #[test]
    fn no_lines_make_an_image_of_padding_only() {
        let image = TextImage::new(&[]);
        assert_eq!((image.width(), image.height()), (2 * PADDING, 2 * PADDING));
    }

    #[test]
    fn the_overlay_reports_every_field() {
        let mut jobs = JobsReport::default();
        jobs.left = 1234;
        let snapshot = Snapshot {
            frame: 7,
            refresh_hz: 59.94,
            presentation: Presentation::Mailbox,
            adapter: "llvmpipe (Vulkan)",
            size: Size::new(640, 480).expect("non-zero"),
            owner: OwnerPhases::default(),
            render: Some(RenderPhases {
                acquire: Duration::from_micros(1500),
                ..RenderPhases::default()
            }),
            feed: FeedStats {
                epochs: 3,
                keys: 45,
                resyncs: 1,
            },
            jobs,
            tiles: None,
            camera: Camera::default(),
        };
        let text = lines(&snapshot).join("\n");
        for expected in [
            "frame 7",
            "640x480",
            "mailbox at 59.9 Hz",
            "llvmpipe (Vulkan)",
            "acquire 1.50",
            "3 epochs  45 tiles  1 resyncs",
            "1234 display tiles left",
            "none on the GPU yet",
            "eye (0.00, 0.00, 3.00)",
        ] {
            assert!(text.contains(expected), "{expected:?} in:\n{text}");
        }
        assert!(text.is_ascii(), "the font draws ASCII");
        let without_render = Snapshot {
            render: None,
            tiles: Some("12 tiles at level 5"),
            ..snapshot
        };
        let rest = lines(&without_render).join("\n");
        assert!(rest.contains("render ms  -"), "{rest}");
        assert!(rest.contains("12 tiles at level 5"), "{rest}");
    }
}
