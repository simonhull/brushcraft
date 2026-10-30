//! The visibility buffer shown as colours: the frame's debug view of the scene until shading
//! (M1a3 Task 5) replaces it.
//!
//! A fullscreen pass (`visibility_view.wgsl`) reads each pixel's [`crate::raster::VisibilityId`]
//! and colours it by its cluster, hashed, so every cluster the raster drew shows as a patch of its
//! own colour, its quads a checker of two shades of it. Background pixels are discarded, so the
//! frame's clear colour shows there. It draws into the frame's own render pass, before the
//! overlay.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, ColorTargetState, ColorWrites,
    FragmentState, MultisampleState, PrimitiveState, PrimitiveTopology, RenderPass, RenderPipeline,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, Texture, TextureFormat,
    TextureViewDescriptor, VertexState,
};

use crate::gpu::{Gpu, GpuError};

/// The debug view's pipeline, and the bind group of the visibility buffer it last showed.
#[derive(Debug)]
pub(crate) struct VisibilityView {
    /// Draws the fullscreen triangle.
    pipeline: RenderPipeline,
    /// The visibility buffer bound, and its bind group: kept while the raster keeps the buffer
    /// (it makes a new one only when the frame's size changes).
    bound: Option<(Texture, BindGroup)>,
}

impl VisibilityView {
    /// The view's pipeline, drawing into frames of `format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the shader or pipeline is refused.
    pub(crate) fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        let device = gpu.device();
        gpu.scoped("visibility view pipeline", || {
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("visibility view"),
                source: ShaderSource::Wgsl(include_str!("visibility_view.wgsl").into()),
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("visibility view"),
                layout: None,
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
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
                    entry_point: Some("shade"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
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

    /// Binds `visibility` (the raster's visibility buffer, the frame's size) for
    /// [`VisibilityView::draw`]: the bind group is made only when the buffer is another than
    /// the last one bound.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    pub(crate) fn bind(&mut self, gpu: &Gpu, visibility: &Texture) -> Result<(), GpuError> {
        if self
            .bound
            .as_ref()
            .is_some_and(|(texture, _)| texture == visibility)
        {
            return Ok(());
        }
        self.bound = None;
        let group = gpu.scoped("visibility view bind group", || {
            let view = visibility.create_view(&TextureViewDescriptor::default());
            gpu.device().create_bind_group(&BindGroupDescriptor {
                label: Some("visibility view"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            })
        })?;
        self.bound = Some((visibility.clone(), group));
        Ok(())
    }

    /// Draws the visibility buffer last bound ([`VisibilityView::bind`]) into `pass`; nothing
    /// if none is.
    pub(crate) fn draw(&self, pass: &mut RenderPass<'_>) {
        if let Some((_, group)) = &self.bound {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
