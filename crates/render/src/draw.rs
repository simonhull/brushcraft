//! A frame's passes, in order: [`Renderer::draw`] owns the composition, so the app, the
//! pinhole test and the golden images all draw a frame the same way.
//!
//! A frame is one submission: the uploads the owner thread staged for it
//! ([`crate::upload::FrameUploads`]) first, then the passes: today the target cleared to the
//! frame's colour, then the overlay ([`crate::overlay`]), always last. The culling, raster,
//! shading and anti-aliasing passes of M1a3's later tasks go between them.

use std::time::{Duration, Instant};

use wgpu::{
    Color, CommandEncoder, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureFormat,
};

use crate::gpu::{Gpu, GpuError};
use crate::overlay::{OverlayPass, TextImage};
use crate::surface::{Acquired, Size, SurfaceError, Target};
use crate::upload::FrameUploads;

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
    /// acquires its next image, encodes the frame's `uploads` (staged by the owner thread's
    /// [`crate::residency::Residency::stage`]) and then the passes, submits them together,
    /// calls `before_present` (a window's `pre_present_notify`), and presents.
    ///
    /// Uploads don't need the image: a frame that gets none ([`Outcome::Retry`],
    /// [`Outcome::Hidden`]) submits its uploads alone, so the tiles still arrive.
    ///
    /// # Errors
    ///
    /// As [`Target::resize`] and [`Target::acquire`]; [`SurfaceError::Gpu`] if a pass can't be
    /// rebuilt for a target of another format, or the GPU reports an error. The uploads are
    /// then dropped unsubmitted, which their residency hears of and recovers from.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        target: &mut Target,
        inputs: &FrameInputs<'_>,
        uploads: Option<FrameUploads>,
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
            Acquired::Retry => return not_drawn(gpu, uploads, Outcome::Retry, times),
            Acquired::Hidden => return not_drawn(gpu, uploads, Outcome::Hidden, times),
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
            // The uploads first: every pass after them reads the tiles they bring.
            if let Some(uploads) = &uploads {
                uploads.encode(&mut encoder);
            }
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
            submit(gpu, uploads, encoder);
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

/// Submits `encoder`: with `uploads`, through [`FrameUploads::submit`], which adds the request
/// readback and hands the staging buffers back once submitted.
fn submit(gpu: &Gpu, uploads: Option<FrameUploads>, encoder: CommandEncoder) {
    match uploads {
        Some(uploads) => uploads.submit(gpu.queue(), encoder),
        None => {
            gpu.queue().submit([encoder.finish()]);
        }
    }
}

/// A frame that got no image (`outcome`): its `uploads`, if any, are submitted alone, so the
/// tiles still arrive. `times` holds the wait for the image; the submission is added as the
/// encode time.
///
/// # Errors
///
/// [`SurfaceError::Gpu`] if the GPU reports an error.
fn not_drawn(
    gpu: &Gpu,
    uploads: Option<FrameUploads>,
    outcome: Outcome,
    mut times: DrawTimes,
) -> Result<Drawn, SurfaceError> {
    if let Some(uploads) = uploads {
        let encoding = Instant::now();
        gpu.scoped("submitting a frame's uploads", || {
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("uploads"),
                });
            uploads.encode(&mut encoder);
            uploads.submit(gpu.queue(), encoder);
        })?;
        times.encode = encoding.elapsed();
    }
    Ok(Drawn { outcome, times })
}
