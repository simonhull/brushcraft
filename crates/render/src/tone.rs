//! The tone map: the scene's linear HDR image brought to the frame's format, in the frame's own
//! render pass, before the overlay; while the camera moves, with an edge filter.
//!
//! The image is premultiplied (colour times coverage, and the coverage): the shading pass's
//! target ([`crate::shade::HDR_FORMAT`], coverage 0 or 1) or, while the camera is still,
//! accumulation's per-pixel means ([`crate::aa`]), whose coverage is the share of a pixel's
//! samples that hit the surface ([`ToneSource`]).
//!
//! The curve, per channel `c` (`tone.wgsl`): the identity up to the knee `k` = 0.8, then a
//! shoulder rising towards 1, `k + (1 − k)(1 − e^{−(c − k)/(1 − k)})`, whose slope is 1 at the
//! knee, so it bends without a crease. Everything a MatCap paints under the knee is shown as it
//! is; a highlight past it (the glossy MatCap reaches about 5) rolls off to white instead of
//! clipping. Exposure is 1. A pixel shows `a · tone(c / a) + (1 − a) · clear`, its colour over the
//! frame's clear colour by its coverage `a`, linear (exactly `tone(c)` where `a` is 1); a pixel
//! with no coverage is discarded, so the clear colour the pass began with shows there. On a frame
//! format that isn't sRGB (the hardware then stores what it is given) the pass encodes sRGB
//! itself; on an sRGB one the hardware does.
//!
//! # The edge filter
//!
//! While the camera moves a frame isn't accumulated, so the pass smooths its edges itself
//! (`tone.wgsl`'s `filtered`), our own filter after FXAA's idea. It compares lumas of the colour
//! as shown (square-rooted, about perceptual). A pixel whose four neighbours' lumas span less than
//! an eighth of their largest (or less than 0.0312) is left exactly as it is: flat and gently
//! shaded regions don't change. On an edge, the second differences across rows and across
//! columns say which way it runs, the neighbour across it that differs more is the one to blend
//! with, and a walk along the edge (up to 12 pixels each way) finds its two ends: a pixel blends
//! by about `0.5 − d / span`, `d` its distance to the nearer end, so the steps of a stair-stepped
//! edge become a ramp. A pixel standing out from its 3 × 3 neighbourhood (a sub-pixel feature)
//! blends by how much it does, up to 0.75. Blending is of the premultiplied HDR colour, so an edge
//! against the background blends coverage.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, Color,
    ColorTargetState, ColorWrites, FragmentState, MultisampleState, PipelineCompilationOptions,
    PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, Texture, TextureFormat, TextureViewDescriptor,
    VertexState,
};

use crate::cull::{entry, storage_buffer, uniform_buffer};
use crate::gpu::{Gpu, GpuError};

/// The curve's knee: below it, colour is shown as it is.
#[cfg(test)]
pub(crate) const KNEE: f32 = 0.8;

/// The curve on the CPU, for the tests: `c` (linear, not negative) as the tone map shows it.
#[must_use]
#[cfg(test)]
pub(crate) fn tone(c: f32) -> f32 {
    if c <= KNEE {
        return c.max(0.0);
    }
    KNEE + (1.0 - KNEE) * (1.0 - (-(c - KNEE) / (1.0 - KNEE)).exp())
}

/// What the tone map shows.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ToneSource<'a> {
    /// The shading pass's target, filtered or not.
    Shaded {
        /// The target.
        image: &'a Texture,
        /// Whether the edge filter runs (the camera moves).
        filter: bool,
    },
    /// Accumulation's means (four half floats a pixel), of an image the size of `shaded`.
    Means {
        /// The shading pass's target: the image's size.
        shaded: &'a Texture,
        /// The means.
        means: &'a Buffer,
    },
}

/// The pass's settings as the GPU reads them: WGSL's `ToneSettings`, 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct GpuToneSettings {
    /// The clear colour, linear; the fourth unused.
    clear: [f32; 4],
    /// The image's size.
    size: [u32; 2],
    /// [`SOURCE_MEANS`] or 0.
    source: u32,
    /// 1 to filter.
    filter: u32,
}

/// [`GpuToneSettings::source`]: the means.
const SOURCE_MEANS: u32 = 1;

/// The tone map's pipeline for one frame format, and the bind group of the image it last showed.
#[derive(Debug)]
pub(crate) struct ToneMap {
    /// Draws the fullscreen triangle.
    pipeline: RenderPipeline,
    /// [`GpuToneSettings`].
    settings: Buffer,
    /// Bound in place of the means when the shaded image is shown.
    placeholder: Buffer,
    /// The image and means bound, and their bind group: kept while they are the same.
    bound: Option<(Texture, Option<Buffer>, BindGroup)>,
}

impl ToneMap {
    /// The tone map drawing into frames of `format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the shader, pipeline or a buffer is refused.
    pub(crate) fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        let device = gpu.device();
        let encode = if format.is_srgb() { 0.0 } else { 1.0 };
        let constants = [("ENCODE_SRGB", encode)];
        let options = PipelineCompilationOptions {
            constants: &constants,
            ..PipelineCompilationOptions::default()
        };
        let pipeline = gpu.scoped("tone map pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("tone map"),
                source: ShaderSource::Wgsl(include_str!("tone.wgsl").into()),
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("tone map"),
                layout: None,
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
                    entry_point: Some("map"),
                    compilation_options: options,
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })?;
        Ok(Self {
            pipeline,
            settings: uniform_buffer(gpu, "tone settings", size_of::<GpuToneSettings>() as u64)?,
            placeholder: storage_buffer(gpu, "tone means placeholder", 8)?,
            bound: None,
        })
    }

    /// Binds `source` for [`ToneMap::draw`], shown over `clear` (linear): the settings are
    /// written through the queue, and the bind group is made only when the image or the means
    /// differ from the last bound.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub(crate) fn bind(
        &mut self,
        gpu: &Gpu,
        source: ToneSource<'_>,
        clear: Color,
    ) -> Result<(), GpuError> {
        let (image, means, filter) = match source {
            ToneSource::Shaded { image, filter } => (image, None, filter),
            ToneSource::Means { shaded, means } => (shaded, Some(means), false),
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a colour to the GPU's f32, rounded once"
        )]
        let clear = [clear.r as f32, clear.g as f32, clear.b as f32, 1.0];
        let settings = GpuToneSettings {
            clear,
            size: [image.width(), image.height()],
            source: if means.is_some() { SOURCE_MEANS } else { 0 },
            filter: u32::from(filter),
        };
        gpu.queue()
            .write_buffer(&self.settings, 0, bytemuck::bytes_of(&settings));
        let same = self
            .bound
            .as_ref()
            .is_some_and(|(texture, buffer, _)| texture == image && buffer.as_ref() == means);
        if same {
            return Ok(());
        }
        self.bound = None;
        let group = gpu.scoped("tone map bind group", || {
            let view = image.create_view(&TextureViewDescriptor::default());
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("tone map"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    entry(1, means.unwrap_or(&self.placeholder)),
                    entry(2, &self.settings),
                ],
            })
        })?;
        self.bound = Some((image.clone(), means.cloned(), group));
        Ok(())
    }

    /// Draws the image last bound ([`ToneMap::bind`]) into `pass`; nothing if none is.
    pub(crate) fn draw(&self, pass: &mut RenderPass<'_>) {
        if let Some((_, _, group)) = &self.bound {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn the_curve_is_the_identity_to_the_knee_then_bends_smoothly_towards_white() {
        assert!((tone(0.5) - 0.5).abs() < f32::EPSILON, "identity below");
        assert!(
            (tone(KNEE) - KNEE).abs() < f32::EPSILON,
            "continuous at the knee"
        );
        let slope = (tone(KNEE + 1e-3) - tone(KNEE)) / 1e-3;
        assert!((slope - 1.0).abs() < 1e-2, "slope 1 at the knee: {slope}");
        let samples = [0.9, 1.0, 1.5, 2.0, 3.0].map(tone);
        assert!(
            samples.windows(2).all(|w| w[0] < w[1]),
            "rising: {samples:?}"
        );
        assert!(samples.iter().all(|&c| c < 1.0), "under white: {samples:?}");
        assert!(tone(50.0) > 0.999 && tone(50.0) <= 1.0, "towards white");
        assert!(tone(-1.0).abs() < f32::EPSILON, "nothing negative");
    }
}
