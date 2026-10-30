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
//! 3. with the camera still, the accumulation ([`crate::aa`]): the frame was drawn at the next
//!    jitter, and its sample goes into each pixel's history;
//! 4. the target cleared to the frame's colour, with the scene shown over it: the histories' means
//!    (still) or the shaded image filtered (moving), tone mapped ([`crate::tone`]), or, as a debug
//!    view ([`SceneView::Clusters`]), the visibility buffer as a colour per cluster
//!    ([`crate::visibility_view`]), neither jittered nor accumulated;
//! 5. the overlay ([`crate::overlay`]), always last.
//!
//! A frame shading the last scene again with the look the histories already hold (one wanted for
//! the overlay alone, such as a pick line) shades nothing: it shows the histories as they are, so
//! it neither restarts nor adds to them.

use std::time::{Duration, Instant};

use thiserror::Error;
use wgpu::{
    Buffer, Color, CommandEncoder, CommandEncoderDescriptor, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureFormat,
};

use crate::aa::{AaFrame, AaReport, AaState};
use crate::camera::Jitter;
use crate::cull::BoundsUpdate;
use crate::gpu::{Gpu, GpuError};
use crate::lod::CutReadback;
use crate::material::MatCapStyle;
use crate::overlay::{OverlayPass, TextImage};
use crate::raster::{FrameError, FramePasses, SceneInputs};
use crate::shade::ShadeSettings;
use crate::surface::{Acquired, Size, SurfaceError, Target};
use crate::tone::{ToneMap, ToneSource};
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
    /// Frame pixels per overlay texel: the window's scale factor, rounded, at least 1. The
    /// wireframe's lines are as many pixels a point.
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
    /// alone (or, with the camera still and the look unchanged, nothing: the accumulated image as
    /// it is), no level choice or raster, for a change of material or wireframe with nothing else
    /// changed (spec target: a shading pass only, independent of the mesh). Shows no scene
    /// ([`Drawn::shown`] false) if the last one drawn was of another size than the frame, or there
    /// was none, or the frame brings uploads: those may change the tiles the visibility buffer's
    /// ids name, so shading the last scene again would read data it wasn't drawn from. Send such
    /// a frame as [`Scene::Draw`].
    Reshade,
}

/// How the scene is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Look {
    /// Shaded, or the debug view.
    pub view: SceneView,
    /// The shading's settings beside the material: the wireframe.
    pub shading: ShadeSettings,
    /// The MatCap's style.
    pub material: MatCapStyle,
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
    /// What its anti-aliasing did ([`crate::aa`]).
    pub aa: AaReport,
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
    /// What its anti-aliasing did.
    aa: AaReport,
}

/// Draws frames: the passes and their GPU state.
#[derive(Debug)]
pub struct Renderer {
    /// The passes into the frame's image.
    image: ImagePasses,
    /// The scene's level choice and raster, once given ([`Renderer::set_passes`]).
    passes: Option<FramePasses>,
    /// Whether the accumulation's histories carry on, and the last jitter.
    aa: AaState,
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
            aa: AaState::default(),
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
            aa,
        } = self;
        let ImagePasses {
            view,
            tone,
            overlay,
            ..
        } = image_passes;
        overlay.prepare(gpu, inputs.overlay, inputs.size, inputs.overlay_scale)?;
        if let Some(passes) = scene_passes.as_mut() {
            passes.set_shading(gpu, inputs.look, inputs.overlay_scale);
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
            staged: uploads.as_ref().is_some_and(|uploads| !uploads.is_empty()),
            clear: inputs.clear,
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
                    encode_scene(gpu, frame, scene, passes, (view, tone, &mut *aa), showing)
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
        let (scene, visible, report) = settle(outcome, drawing, &mut self.aa)?;
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
            aa: report,
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
    /// Whether the frame brings uploads: a scene shaded again must not.
    staged: bool,
    /// The frame's clear colour, which the tone map composites over.
    clear: Color,
}

/// Encodes `scene` with `passes` into `frame`, after the uploads; `None` for no scene.
///
/// The scene is drawn or shaded again, with its accumulation as `aa` decides, and the image the
/// frame shows ([`Showing::look`]) is bound for the tone map or the debug view.
///
/// A scene drawn is always shaded, the debug view or not, so the shading pass's view and bind
/// groups are the last scene's when a later frame shades it again.
///
/// # Errors
///
/// As [`FramePasses::encode`], [`FramePasses::encode_shading`] and
/// [`FramePasses::encode_accumulation`]; [`FrameError::Gpu`] if the tone map or the debug view
/// can't bind its image.
fn encode_scene(
    gpu: &Gpu,
    frame: &mut CommandEncoder,
    scene: Scene<'_>,
    passes: &mut FramePasses,
    (view, tone, aa): (&mut VisibilityView, &mut ToneMap, &mut AaState),
    showing: Showing,
) -> Option<Result<EncodedScene, FrameError>> {
    let shaded = showing.look.view == SceneView::Shaded;
    let encoded = match scene {
        Scene::None => return None,
        Scene::Draw(scene) => (|| {
            let accumulable = passes.aa().fits(showing.size);
            let still = !scene.moving;
            let step = aa.draw(still, accumulable, showing.look, showing.size);
            let jitter = match step {
                AaFrame::Accumulate { jitter, .. } => jitter,
                AaFrame::Plain | AaFrame::Filter | AaFrame::Show => Jitter::CENTRE,
            };
            let readback = passes.encode_jittered(gpu, frame, scene, jitter)?;
            passes.encode_shading(gpu, frame, scene)?;
            let tree = Some(scene.tree);
            let (step, mut report) =
                accumulate(gpu, frame, passes, aa, step, (tree, Some(scene.bounds)));
            report.filtered_instead |= shaded && still && !accumulable;
            let encoded = EncodedScene {
                readback,
                shown: true,
                aa: report,
            };
            Ok((encoded, step))
        })(),
        Scene::Reshade => (|| {
            let visibility = passes.raster().visibility();
            let same_size = (visibility.width(), visibility.height())
                == (showing.size.width(), showing.size.height());
            let unchanged = same_size && !showing.staged;
            if !unchanged {
                let nothing = EncodedScene {
                    readback: None,
                    shown: false,
                    aa: AaReport::default(),
                };
                return Ok((nothing, AaFrame::Plain));
            }
            let accumulable = passes.aa().fits(showing.size);
            let step = aa.reshade(accumulable, showing.look, showing.size);
            let (visible, step, mut report) = match step {
                AaFrame::Show => (passes.aa().mean().is_some(), step, AaReport::default()),
                AaFrame::Accumulate { .. } | AaFrame::Filter => {
                    let again = passes.encode_shading_again(gpu, frame)?;
                    let (step, report) = if again {
                        accumulate(gpu, frame, passes, aa, step, (None, None))
                    } else {
                        (step, AaReport::default())
                    };
                    (again, step, report)
                }
                AaFrame::Plain => (true, step, AaReport::default()),
            };
            report.filtered_instead |= shaded && !accumulable;
            let encoded = EncodedScene {
                readback: None,
                shown: visible,
                aa: report,
            };
            Ok((encoded, step))
        })(),
    };
    Some(encoded.and_then(|(encoded, step)| {
        if encoded.shown {
            if shaded {
                bind_shaded(gpu, passes, tone, step, showing.clear)?;
            } else {
                view.bind(gpu, passes.raster().visibility())?;
            }
        }
        Ok(encoded)
    }))
}

/// Encodes the accumulation of the frame just shaded, if `step` accumulates; returns the step
/// taken and what it did.
///
/// It reads `inputs`: the tile tree's buffer and the tiles whose data changed (`None` for the
/// last tree, and none). A refusal never fails the frame: the histories are dropped (`aa`
/// restarts next time) and the frame is filtered instead.
fn accumulate(
    gpu: &Gpu,
    frame: &mut CommandEncoder,
    passes: &mut FramePasses,
    aa: &mut AaState,
    step: AaFrame,
    (tree, changes): (Option<&Buffer>, Option<&BoundsUpdate>),
) -> (AaFrame, AaReport) {
    let AaFrame::Accumulate { restart, .. } = step else {
        return (step, AaReport::default());
    };
    match passes.encode_accumulation(gpu, frame, tree, changes, restart) {
        Ok(()) => (
            step,
            AaReport {
                restarted: restart,
                filtered_instead: false,
            },
        ),
        Err(_refused) => {
            aa.lost();
            let report = AaReport {
                restarted: false,
                filtered_instead: true,
            };
            (AaFrame::Filter, report)
        }
    }
}

/// Binds the image a shaded frame shows for the tone map, as `step` made it: the histories'
/// means while still, else the shaded image, filtered while moving.
///
/// # Errors
///
/// [`FrameError::Gpu`] if the tone map can't bind it.
fn bind_shaded(
    gpu: &Gpu,
    passes: &FramePasses,
    tone: &mut ToneMap,
    step: AaFrame,
    clear: Color,
) -> Result<(), FrameError> {
    let Some(target) = passes.shade().target() else {
        return Ok(());
    };
    let aa = passes.aa();
    let source = match (step, aa.mean(), aa.record()) {
        (AaFrame::Accumulate { .. } | AaFrame::Show, Some(means), Some(records)) => {
            ToneSource::Means {
                shaded: target,
                means,
                records,
                frame: aa.frame(),
            }
        }
        _ => ToneSource::Shaded {
            image: target,
            filter: step == AaFrame::Filter,
        },
    };
    tone.bind(gpu, source, clear)?;
    Ok(())
}

/// What a submitted frame's scene came to: drawn, shown, and its anti-aliasing.
///
/// Drawn means `drawing`, if it was encoded. Its readback is marked submitted; a scene not shown
/// leaves `aa` no history to continue.
///
/// # Errors
///
/// The scene's encoding error, if it had one.
fn settle(
    outcome: Option<Result<EncodedScene, FrameError>>,
    drawing: bool,
    aa: &mut AaState,
) -> Result<(bool, bool, AaReport), FrameError> {
    let settled = match outcome {
        Some(Ok(encoded)) => {
            if let Some(readback) = encoded.readback {
                readback.submitted();
            }
            (drawing, encoded.shown, encoded.aa)
        }
        Some(Err(error)) => {
            aa.lost();
            return Err(error);
        }
        None => (false, false, AaReport::default()),
    };
    if !settled.1 {
        aa.lost();
    }
    Ok(settled)
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
        aa: AaReport::default(),
    })
}
