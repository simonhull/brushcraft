//! A frame's passes, in order: [`Renderer::draw`] owns the composition, so the app, the
//! pinhole test and the golden images all draw a frame the same way.
//!
//! A frame is one submission: the uploads the owner thread staged for it
//! ([`crate::upload::FrameUploads`]) first, then the passes:
//!
//! 1. the scene's, when the frame has one and the renderer has its [`FramePasses`]
//!    ([`Renderer::set_passes`]): the bounds of the tiles whose data changed, the level choice and
//!    the raster into the visibility buffer, and the cut's readback;
//! 2. the shading pass ([`crate::shade`]), shading the visibility buffer into the linear HDR image;
//!    a frame whose scene is unchanged and whose shading isn't ([`Scene::Reshade`]: a material or
//!    wireframe change) runs this pass alone, with no level choice or raster;
//! 3. the target cleared to the frame's colour, with the scene shown over it: the shaded image tone
//!    mapped ([`crate::tone`]), or, as a debug view ([`SceneView::Clusters`]), the visibility
//!    buffer as a colour per cluster ([`crate::visibility_view`]);
//! 4. the overlay ([`crate::overlay`]), always last.
//!
//! Anti-aliasing (M1a3 Task 6) goes between the shading and the tone map.

use std::time::{Duration, Instant};

use thiserror::Error;
use wgpu::{
    Color, CommandEncoder, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureFormat,
};

use crate::gpu::{Gpu, GpuError};
use crate::lod::CutReadback;
use crate::overlay::{OverlayPass, TextImage};
use crate::raster::{FrameError, FramePasses, SceneInputs};
use crate::shade::ShadeSettings;
use crate::surface::{Acquired, Size, SurfaceError, Target};
use crate::tone::ToneMap;
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
    /// The scene: drawn, shaded again as it was, or none. A renderer without [`FramePasses`]
    /// draws none.
    pub scene: Scene<'a>,
    /// How the scene is shown.
    pub look: Look,
}

/// What a frame does with the scene.
#[derive(Debug, Clone, Copy, Default)]
pub enum Scene<'a> {
    /// No scene: the clear colour and the overlay.
    #[default]
    None,
    /// Draws it: the tiles the frame draws, its view (of the frame's size) and its time.
    Draw(&'a SceneInputs<'a>),
    /// Shows the last scene drawn again, shaded with this frame's [`Look`]: the shading pass
    /// alone, no level choice or raster, for a change of material or wireframe with nothing else
    /// changed (spec target: a shading pass only, independent of the mesh). Shows no scene if the
    /// last one drawn was of another size than the frame, or there was none.
    Reshade,
}

/// How the scene is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Look {
    /// Shaded, or the debug view.
    pub view: SceneView,
    /// The shading's settings: the MatCap's style and the wireframe.
    pub shading: ShadeSettings,
}

/// Which image of the scene a frame shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SceneView {
    /// The shaded image, tone mapped.
    #[default]
    Shaded,
    /// The debug view: the visibility buffer as a colour per cluster, its quads a checker.
    Clusters,
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
    /// the GPU, and its cut is read back. False for a frame with no scene, or no image, or one
    /// that only shaded the last scene again ([`Scene::Reshade`]).
    pub scene: bool,
    /// Whether a scene was shown: drawn, or shaded again.
    pub shown: bool,
}

/// Why a frame couldn't be drawn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DrawError {
    /// The target: resizing, acquiring or presenting it. The frame's uploads are dropped
    /// unsubmitted (their residency hears of it and recovers).
    #[error(transparent)]
    Surface(#[from] SurfaceError),
    /// The GPU refused the frame's own work: rebuilding a pass for a new format, uploading the
    /// overlay, or a validation error in the frame's submission (the frame was submitted).
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// The scene's passes (the level choice, the raster, the shading, the tone map's or the
    /// debug view's binding) couldn't be encoded. The frame was submitted all the same, its uploads
    /// first, with the passes encoded before the failure (whole passes only), so its tiles
    /// arrive; its cut readback is given back unread, and the acquired image is dropped
    /// unpresented.
    #[error("encoding the scene's passes")]
    Scene(#[from] FrameError),
}

/// The passes that draw into the frame's image, built for its format.
#[derive(Debug)]
struct ImagePasses {
    /// The format they draw into.
    format: TextureFormat,
    /// The visibility buffer's debug view.
    view: VisibilityView,
    /// The tone map.
    tone: ToneMap,
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
            tone: ToneMap::new(gpu, format)?,
            overlay: OverlayPass::new(gpu, format)?,
        })
    }
}

/// A frame's scene, encoded, its image bound for the pass that shows it.
#[derive(Debug)]
struct EncodedScene {
    /// Its readback, to mark submitted once the frame is; `None` if every readback buffer was
    /// in flight, or the scene was only shaded again.
    readback: Option<CutReadback>,
    /// Whether there is an image to show.
    shown: bool,
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
        let ImagePasses {
            view,
            tone,
            overlay,
            ..
        } = image_passes;
        overlay.prepare(gpu, inputs.overlay, inputs.size, inputs.overlay_scale)?;
        if let Some(passes) = scene_passes.as_mut() {
            passes.set_shading(gpu, inputs.look.shading);
        }
        let to_draw = scene_passes.as_mut().map(|passes| (inputs.scene, passes));
        let drawing = matches!(to_draw, Some((Scene::Draw(_), _)));
        // A frame drawing no scene runs no level choice: its request readback reads nothing.
        let uploads = uploads.map(|uploads| {
            if drawing {
                uploads
            } else {
                uploads.without_level_choice()
            }
        });
        let showing = Showing {
            look: inputs.look,
            size: inputs.size,
        };
        let outcome = gpu.scoped("encoding a frame", || {
            let commands = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame"),
                });
            let mut encoded = None;
            let passes = |frame: &mut CommandEncoder| {
                encoded = to_draw.and_then(|(scene, passes)| {
                    encode_scene(gpu, frame, scene, passes, (view, tone), showing)
                });
                let mut pass = frame.begin_render_pass(&RenderPassDescriptor {
                    label: Some("scene and overlay"),
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
                if let Some(Ok(EncodedScene { shown: true, .. })) = &encoded {
                    match inputs.look.view {
                        SceneView::Shaded => tone.draw(&mut pass),
                        SceneView::Clusters => view.draw(&mut pass),
                    }
                }
                // The last pass: the overlay over everything.
                overlay.draw(&mut pass);
            };
            submit(gpu, uploads, commands, passes);
            encoded
        })?;
        let (scene, visible) = match outcome {
            Some(Ok(encoded)) => {
                if let Some(readback) = encoded.readback {
                    readback.submitted();
                }
                (drawing, encoded.shown)
            }
            Some(Err(error)) => return Err(error.into()),
            None => (false, false),
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
            shown: visible,
        })
    }
}

/// How a frame shows its scene: its look, and its size (a scene shaded again must be of it).
#[derive(Debug, Clone, Copy)]
struct Showing {
    /// The look.
    look: Look,
    /// The frame's size.
    size: Size,
}

/// Encodes `scene` with `passes` into `frame`, after the uploads (drawn, or shaded again), and
/// binds the image the frame shows ([`Showing::look`]) for the tone map or the debug view;
/// `None` for no scene.
///
/// A scene drawn is always shaded, the debug view or not, so the shading pass's view and bind
/// groups are the last scene's when a later frame shades it again.
///
/// # Errors
///
/// As [`FramePasses::encode`] and [`FramePasses::encode_shading`]; [`FrameError::Gpu`] if the
/// tone map or the debug view can't bind its image.
fn encode_scene(
    gpu: &Gpu,
    frame: &mut CommandEncoder,
    scene: Scene<'_>,
    passes: &mut FramePasses,
    (view, tone): (&mut VisibilityView, &mut ToneMap),
    showing: Showing,
) -> Option<Result<EncodedScene, FrameError>> {
    let shaded = showing.look.view == SceneView::Shaded;
    let encoded = match scene {
        Scene::None => return None,
        Scene::Draw(scene) => (|| {
            let readback = passes.encode(gpu, frame, scene)?;
            passes.encode_shading(gpu, frame, scene)?;
            Ok(EncodedScene {
                readback,
                shown: true,
            })
        })(),
        Scene::Reshade => (|| {
            let visibility = passes.raster().visibility();
            let same_size = (visibility.width(), visibility.height())
                == (showing.size.width(), showing.size.height());
            let visible = same_size && (!shaded || passes.encode_shading_again(gpu, frame)?);
            Ok(EncodedScene {
                readback: None,
                shown: visible,
            })
        })(),
    };
    Some(encoded.and_then(|encoded| {
        if encoded.shown {
            if shaded {
                if let Some(target) = passes.shade().target() {
                    tone.bind(gpu, target)?;
                }
            } else {
                view.bind(gpu, passes.raster().visibility())?;
            }
        }
        Ok(encoded)
    }))
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
        shown: false,
    })
}
