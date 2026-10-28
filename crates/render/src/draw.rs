//! A frame's passes, in order: [`Renderer::draw`] owns the composition, so the app, the
//! pinhole test and the golden images all draw a frame the same way.
//!
//! The passes today: the target cleared to the frame's colour, then the overlay
//! ([`crate::overlay`]), always last. The culling, raster, shading and anti-aliasing passes of
//! M1a3's later tasks go between them; the uploads of Task 2 are encoded before them.

use std::time::{Duration, Instant};

use wgpu::{
    Color, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureFormat,
};

use crate::gpu::{Gpu, GpuError};
use crate::overlay::{OverlayPass, TextImage};
use crate::surface::{Acquired, Size, SurfaceError, Target};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// What one frame draws.
#[derive(Debug, Clone, Copy)]
pub struct FrameInputs<'a> {
    /// The size to draw at: the target is resized to it first.
    pub size: Size,
    /// The colour the frame is cleared to, linear.
    pub clear: Color,
    /// The overlay's text image, drawn last over the top-left corner; `None` for no overlay.
    pub overlay: Option<&'a TextImage>,
    /// Frame pixels per overlay texel: the window's scale factor, rounded, at least 1.
    pub overlay_scale: u32,
}

/// How long a frame's phases took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawTimes {
    /// Waiting for the target's next image (FIFO waits here for a free one).
    pub acquire: Duration,
    /// Uploading the frame's inputs, encoding its passes and submitting them.
    pub encode: Duration,
    /// Presenting the image.
    pub present: Duration,
}

/// What became of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// It was drawn and presented (or, offscreen, drawn).
    Presented,
    /// No image this time: the surface changed or the wait timed out, and the surface has been
    /// set up again. Draw the frame again.
    Retry,
    /// The window can't be seen: nothing was drawn.
    Hidden,
}

/// A drawn frame's outcome and timings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Drawn {
    /// What became of it.
    pub outcome: Outcome,
    /// How long its phases took (only acquiring, if it wasn't drawn).
    pub times: DrawTimes,
}

/// Draws frames: the passes and their GPU state.
#[derive(Debug)]
pub struct Renderer {
    /// The format the passes draw into.
    format: TextureFormat,
    /// The overlay pass.
    overlay: OverlayPass,
}

impl Renderer {
    /// The passes for frames of `format` (the target's, [`Target::format`]).
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if a pass's shader or pipeline is refused.
    pub fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        Ok(Self {
            format,
            overlay: OverlayPass::new(gpu, format)?,
        })
    }

    /// Draws one frame into `target` and presents it: resizes the target to the frame,
    /// acquires its next image, encodes and submits the passes, calls `before_present` (a
    /// window's `pre_present_notify`), and presents.
    ///
    /// # Errors
    ///
    /// As [`Target::resize`] and [`Target::acquire`]; [`SurfaceError::Gpu`] if a pass can't be
    /// rebuilt for a target of another format, or the GPU reports an error.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        target: &mut Target,
        inputs: &FrameInputs<'_>,
        before_present: impl FnOnce(),
    ) -> Result<Drawn, SurfaceError> {
        if target.format() != self.format {
            *self = Self::new(gpu, target.format())?;
        }
        let mut times = DrawTimes::default();
        target.resize(gpu, inputs.size)?;
        let began = Instant::now();
        let acquired = target.acquire(gpu)?;
        times.acquire = began.elapsed();
        let image = match acquired {
            Acquired::Frame(image) => image,
            Acquired::Retry => {
                return Ok(Drawn {
                    outcome: Outcome::Retry,
                    times,
                });
            }
            Acquired::Hidden => {
                return Ok(Drawn {
                    outcome: Outcome::Hidden,
                    times,
                });
            }
        };
        let encoding = Instant::now();
        self.overlay
            .prepare(gpu, inputs.overlay, inputs.size, inputs.overlay_scale)?;
        gpu.scoped("encoding a frame", || {
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("clear and overlay"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: image.view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(inputs.clear),
                            store: StoreOp::Store,
                        },
                    })],
                    ..RenderPassDescriptor::default()
                });
                // The last pass: the overlay over everything.
                self.overlay.draw(&mut pass);
            }
            gpu.queue().submit([encoder.finish()]);
        })?;
        times.encode = encoding.elapsed();
        let presenting = Instant::now();
        before_present();
        image.present(gpu);
        gpu.check()?;
        times.present = presenting.elapsed();
        Ok(Drawn {
            outcome: Outcome::Presented,
            times,
        })
    }
}
