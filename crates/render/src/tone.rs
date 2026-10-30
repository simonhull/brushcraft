//! The tone map: the shading pass's linear HDR image ([`crate::shade::HDR_FORMAT`]) brought to
//! the frame's format, in the frame's own render pass, before the overlay.
//!
//! The curve, per channel `c` (`tone.wgsl`): the identity up to the knee `k` = 0.8, then a
//! shoulder rising towards 1, `k + (1 − k)(1 − e^{−(c − k)/(1 − k)})`, whose slope is 1 at the
//! knee, so it bends without a crease. Everything a MatCap paints under the knee is shown as it
//! is; a highlight past it (the glossy MatCap reaches about 5) rolls off to white instead of
//! clipping. Exposure is 1. On a frame format that isn't sRGB (the hardware then stores what it
//! is given) the pass encodes sRGB itself; on an sRGB one the hardware does. Background pixels
//! (alpha 0) are discarded, so the frame's clear colour shows there.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, ColorTargetState, ColorWrites,
    FragmentState, MultisampleState, PipelineCompilationOptions, PrimitiveState, PrimitiveTopology,
    RenderPass, RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    Texture, TextureFormat, TextureViewDescriptor, VertexState,
};

use crate::gpu::{Gpu, GpuError};

/// The curve's knee: below it, colour is shown as it is.
pub const KNEE: f32 = 0.8;

/// The curve, on the CPU: `c` (linear, not negative) as the tone map shows it, linear.
#[must_use]
pub fn tone(c: f32) -> f32 {
    if c <= KNEE {
        return c.max(0.0);
    }
    KNEE + (1.0 - KNEE) * (1.0 - (-(c - KNEE) / (1.0 - KNEE)).exp())
}

/// The tone map's pipeline for one frame format, and the bind group of the image it last showed.
#[derive(Debug)]
pub struct ToneMap {
    /// Draws the fullscreen triangle.
    pipeline: RenderPipeline,
    /// The shaded image bound, and its bind group: kept while the shading pass keeps it.
    bound: Option<(Texture, BindGroup)>,
}

impl ToneMap {
    /// The tone map drawing into frames of `format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the shader or pipeline is refused.
    pub fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        let device = gpu.device();
        let encode = if format.is_srgb() { 0.0 } else { 1.0 };
        let constants = [("ENCODE_SRGB", encode)];
        let options = PipelineCompilationOptions {
            constants: &constants,
            ..PipelineCompilationOptions::default()
        };
        gpu.scoped("tone map pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("tone map"),
                source: ShaderSource::Wgsl(include_str!("tone.wgsl").into()),
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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
            });
            Self {
                pipeline,
                bound: None,
            }
        })
    }

    /// Binds `shaded` (the shading pass's target, the frame's size) for [`ToneMap::draw`]: the
    /// bind group is made only when it is another image than the last one bound.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub fn bind(&mut self, gpu: &Gpu, shaded: &Texture) -> Result<(), GpuError> {
        if self
            .bound
            .as_ref()
            .is_some_and(|(texture, _)| texture == shaded)
        {
            return Ok(());
        }
        self.bound = None;
        let group = gpu.scoped("tone map bind group", || {
            let view = shaded.create_view(&TextureViewDescriptor::default());
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("tone map"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            })
        })?;
        self.bound = Some((shaded.clone(), group));
        Ok(())
    }

    /// Draws the image last bound ([`ToneMap::bind`]) into `pass`; nothing if none is.
    pub fn draw(&self, pass: &mut RenderPass<'_>) {
        if let Some((_, group)) = &self.bound {
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
