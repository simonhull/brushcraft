//! A frame's passes, in order: [`Renderer::draw`] owns the composition, so the app, the
//! pinhole test and the golden images all draw a frame the same way.
//!
//! A frame is one submission: the uploads the owner thread staged for it
//! ([`crate::upload::FrameUploads`]) first, then the passes:
//!
//! 1. the scene's, when the frame has one and the renderer has its [`FramePasses`]
//!    ([`Renderer::set_passes`]): the bounds of the tiles whose data changed, the level choice and
//!    the raster into the visibility buffer, and the cut's readback;
//! 2. the target cleared to the frame's colour, with the visibility buffer shown over it as colours
//!    ([`crate::visibility_view`], a debug view until shading);
//! 3. the overlay ([`crate::overlay`]), always last.
//!
//! Shading and anti-aliasing (M1a3's later tasks) take the debug view's place.

use std::time::{Duration, Instant};

use thiserror::Error;
use wgpu::{
    BindGroup, Color, CommandEncoder, CommandEncoderDescriptor, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureFormat,
};

use crate::gpu::{Gpu, GpuError};
use crate::lod::CutReadback;
use crate::overlay::{OverlayPass, TextImage};
use crate::raster::{FrameError, FramePasses, SceneInputs};
use crate::surface::{Acquired, Size, SurfaceError, Target};
use crate::upload::FrameUploads;
use crate::visibility_view::VisibilityView;

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
    /// The scene: the tiles the frame draws, its view (of the frame's size) and its time.
    /// `None`, or a renderer without [`FramePasses`], draws no scene.
    pub scene: Option<&'a SceneInputs<'a>>,
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
    /// How long its phases took. A frame that wasn't drawn has no present time, and its
    /// encode time is submitting its uploads alone, if it had any.
    pub times: DrawTimes,
    /// Whether the scene's passes were submitted: the bounds update the frame carried is then on
    /// the GPU, and its cut is read back. False for a frame with no scene, or no image.
    pub scene: bool,
}

/// Why a frame couldn't be drawn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DrawError {
    /// The target (resizing, acquiring, presenting), or the GPU under it.
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    /// The scene's passes couldn't be encoded. The frame's uploads were submitted all the same.
    #[error("encoding the scene's passes")]
    Scene(#[from] FrameError),
}

impl From<GpuError> for DrawError {
    fn from(error: GpuError) -> Self {
        Self::Surface(SurfaceError::Gpu(error))
    }
}

/// The passes that draw into the frame's image, built for its format.
#[derive(Debug)]
struct ImagePasses {
    /// The format they draw into.
    format: TextureFormat,
    /// The visibility buffer's debug view.
    view: VisibilityView,
    /// The overlay pass.
    overlay: OverlayPass,
}

impl ImagePasses {
    /// The passes for frames of `format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if a pass's shader or pipeline is refused.
    fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        Ok(Self {
            format,
            view: VisibilityView::new(gpu, format)?,
            overlay: OverlayPass::new(gpu, format)?,
        })
    }
}

/// A frame's scene, encoded: its readback, and how its visibility buffer is shown.
struct EncodedScene {
    /// The cut's readback, to mark submitted once the frame is; `None` if every readback buffer
    /// was in flight.
    readback: Option<CutReadback>,
    /// The debug view's bind group over the frame's visibility buffer.
    view: BindGroup,
}

/// Draws frames: the passes and their GPU state.
#[derive(Debug)]
pub struct Renderer {
    /// The passes into the frame's image.
    image: ImagePasses,
    /// The scene's level choice and raster, once given ([`Renderer::set_passes`]).
    passes: Option<FramePasses>,
}

impl Renderer {
    /// The passes for frames of `format` (the target's, [`Target::format`]), with no scene's
    /// passes yet.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if a pass's shader or pipeline is refused.
    pub fn new(gpu: &Gpu, format: TextureFormat) -> Result<Self, GpuError> {
        Ok(Self {
            image: ImagePasses::new(gpu, format)?,
            passes: None,
        })
    }

    /// Draws each frame's scene with `passes` from now on. They are built where residency is,
    /// for its tile tree and slot layout ([`FramePasses::new`]), so a device that can't take
    /// them fails at startup, not mid-session.
    pub fn set_passes(&mut self, passes: FramePasses) {
        self.passes = Some(passes);
    }

    /// The scene's passes, once given: their visibility buffer, for tests and tools.
    #[must_use]
    pub const fn passes(&self) -> Option<&FramePasses> {
        self.passes.as_ref()
    }

    /// Draws one frame into `target` and presents it: resizes the target to the frame,
    /// acquires its next image, encodes the frame's `uploads` (staged by the owner thread's
    /// [`crate::residency::Residency::stage`]) and then the passes (the module notes say which),
    /// submits them together, marks the scene's readback submitted, calls `before_present` (a
    /// window's `pre_present_notify`), and presents.
    ///
    /// Uploads don't need the image: a frame that gets none ([`Outcome::Retry`],
    /// [`Outcome::Hidden`]) submits its uploads alone, so the tiles still arrive; its scene
    /// isn't drawn ([`Drawn::scene`]).
    ///
    /// # Errors
    ///
    /// [`DrawError::Surface`] as [`Target::resize`] and [`Target::acquire`], if a pass can't be
    /// rebuilt for a target of another format, or if the GPU reports an error; the uploads are
    /// then dropped unsubmitted, which their residency hears of and recovers from.
    /// [`DrawError::Scene`] if the scene's passes can't be encoded (the uploads are submitted).
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        target: &mut Target,
        inputs: &FrameInputs<'_>,
        uploads: Option<FrameUploads>,
        before_present: impl FnOnce(),
    ) -> Result<Drawn, DrawError> {
        if target.format() != self.image.format {
            self.image = ImagePasses::new(gpu, target.format())?;
        }
        let mut times = DrawTimes::default();
        target.resize(gpu, inputs.size)?;
        let began = Instant::now();
        let acquired = target.acquire(gpu)?;
        times.acquire = began.elapsed();
        let image = match acquired {
            Acquired::Frame(image) => image,
            Acquired::Retry => return Ok(not_drawn(gpu, uploads, Outcome::Retry, times)?),
            Acquired::Hidden => return Ok(not_drawn(gpu, uploads, Outcome::Hidden, times)?),
        };
        let encoding = Instant::now();
        let Self {
            image: image_passes,
            passes: scene_passes,
        } = self;
        image_passes
            .overlay
            .prepare(gpu, inputs.overlay, inputs.size, inputs.overlay_scale)?;
        let to_draw = inputs.scene.zip(scene_passes.as_mut());
        // A frame with no scene runs no level choice: its request readback reads nothing.
        let uploads = uploads.map(|uploads| {
            if to_draw.is_some() {
                uploads
            } else {
                uploads.without_level_choice()
            }
        });
        let outcome = gpu.scoped("encoding a frame", || {
            let commands = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            let mut drawn = None;
            let passes = |frame: &mut CommandEncoder| {
                drawn = to_draw.map(|(scene, passes)| {
                    encode_scene(gpu, frame, scene, passes, &image_passes.view)
                });
                let mut pass = frame.begin_render_pass(&RenderPassDescriptor {
                    label: Some("visibility view and overlay"),
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
                if let Some(Ok(encoded)) = &drawn {
                    image_passes.view.draw(&mut pass, &encoded.view);
                }
                // The last pass: the overlay over everything.
                image_passes.overlay.draw(&mut pass);
            };
            submit(gpu, uploads, commands, passes);
            drawn
        })?;
        let scene = match outcome {
            Some(Ok(encoded)) => {
                if let Some(readback) = encoded.readback {
                    readback.submitted();
                }
                true
            }
            Some(Err(error)) => return Err(error),
            None => false,
        };
        times.encode = encoding.elapsed();
        let presenting = Instant::now();
        before_present();
        image.present(gpu);
        gpu.check()?;
        times.present = presenting.elapsed();
        Ok(Drawn {
            outcome: Outcome::Presented,
            times,
            scene,
        })
    }
}

/// Encodes `scene` with `passes` into `frame`, after the uploads, and binds its visibility
/// buffer for the debug `view`.
///
/// # Errors
///
/// [`DrawError::Scene`] as [`FramePasses::encode`]; [`DrawError::Surface`] if the debug view
/// can't bind the visibility buffer.
fn encode_scene(
    gpu: &Gpu,
    frame: &mut CommandEncoder,
    scene: &SceneInputs<'_>,
    passes: &mut FramePasses,
    view: &VisibilityView,
) -> Result<EncodedScene, DrawError> {
    let readback = passes.encode(gpu, frame, scene)?;
    let view = view.bind(gpu, passes.raster().visibility())?;
    Ok(EncodedScene { readback, view })
}

/// Encodes `passes` into `encoder` and submits it.
///
/// With `uploads`, through [`FrameUploads::submit`], which puts the uploads' copies first (every
/// pass reads the tiles they bring), adds the request readback, and hands the staging buffers back
/// once submitted.
fn submit(
    gpu: &Gpu,
    uploads: Option<FrameUploads>,
    mut encoder: CommandEncoder,
    passes: impl FnOnce(&mut CommandEncoder),
) {
    if let Some(uploads) = uploads {
        uploads.submit(gpu.queue(), encoder, passes);
    } else {
        passes(&mut encoder);
        gpu.queue().submit([encoder.finish()]);
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
            let encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("uploads"),
                });
            uploads
                .without_level_choice()
                .submit(gpu.queue(), encoder, |_| {});
        })?;
        times.encode = encoding.elapsed();
    }
    Ok(Drawn {
        outcome,
        times,
        scene: false,
    })
}
