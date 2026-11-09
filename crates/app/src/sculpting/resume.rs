//! A stroke a restore found still landing: taken on by the brush, or finished with no frame loop
//! for a copy of found work.
//!
//! See the [`super`] module notes, "A stroke a restore found", and the dab journal spec,
//! "Restoring a journalled stroke".

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use store::{OpenStroke, Project, ProjectError};
use stroke::Dab;

use super::{DabRow, FIRST_INTERVAL, Pending, Sculpting, Source, StrokeOp, ViewProjector};
use crate::camera::Camera;
use crate::dialogs::LastStroke;
use crate::frame::{Brush, BrushContext, BrushReport, Drawing, SampleTimes};

/// Why a stroke a restore found isn't taken on: its rows are of another dab layout than the
/// brush's ([`stroke::DAB_FORMAT`]), so they can't be read as its dabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "its dabs are of layout {found}, and the brush reads layout {}",
    stroke::DAB_FORMAT
)]
pub(super) struct OtherDabFormat {
    /// The layout its rows were written in.
    found: u32,
}

/// Why a stroke a restore found isn't taken on: one of its rows can't be a dab of this model, so
/// the file holding it is damaged or hostile, and its numbers mustn't reach the index's queries.
///
/// A dab is placed on the surface, so its numbers are finite, its radius positive and no wider
/// than the model (twice the radius of the sphere holding the index's tiles,
/// [`store::Bvh::bounding_sphere`]), and its sphere reaches that sphere. The rest of what a dab
/// must be (a normal that has a direction) is checked as it lands
/// ([`apply_dabs`](sculpt::apply_dabs)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("its dab {at} can't be a dab of this model: {why}")]
pub(super) struct HostileDab {
    /// The dab's place among those not landed.
    at: usize,
    /// What is wrong with it.
    why: &'static str,
}

impl HostileDab {
    /// Checks `dab`, the `at`th not landed, against the sphere holding the model's tiles
    /// (`model`; `None` for an empty index, which nothing can be a dab of).
    ///
    /// # Errors
    ///
    /// Why it can't be a dab of this model.
    fn check(at: usize, dab: &Dab, model: Option<([f64; 3], f64)>) -> Result<(), Self> {
        let hostile = |why| Err(Self { at, why });
        let numbers = dab
            .position
            .iter()
            .chain(&dab.normal)
            .chain(&dab.tangent)
            .chain(&dab.params)
            .chain([
                &dab.radius,
                &dab.strength,
                &dab.angle,
                &dab.hardness,
                &dab.time,
            ]);
        if !numbers.copied().all(f32::is_finite) {
            return hostile("a number that isn't finite");
        }
        let Some((centre, model_radius)) = model else {
            return hostile("the model has no tiles");
        };
        let radius = f64::from(dab.radius);
        if radius <= 0.0 || radius > 2.0 * model_radius {
            return hostile("a radius out of the model's range");
        }
        let offsets = dab.position.iter().zip(centre);
        let away: f64 = offsets.map(|(&p, c)| (f64::from(p) - c).powi(2)).sum();
        if away.sqrt() > model_radius + radius {
            return hostile("a sphere off the model");
        }
        Ok(())
    }
}

impl Sculpting {
    /// Takes on `found`, the stroke a restore found still landing, through `context` (see "A
    /// stroke a restore found" in the module notes): resumed in the project, its dabs queued as
    /// a stroke whose pen has lifted and shown at once.
    ///
    /// A project that refuses it for a transient reason (the store's cache) is asked once more.
    ///
    /// # Errors
    ///
    /// Why it isn't taken on, kept as the brush's error too: a stroke under way or waiting
    /// (never, before any step), no spatial index (nothing can be brushed), no view to land it
    /// through, rows of another layout ([`OtherDabFormat`]), a row that can't be a dab of this
    /// model ([`HostileDab`]), or a project that won't resume it.
    pub(super) fn take_on(&mut self, context: BrushContext<'_>, found: OpenStroke) -> Result<()> {
        let taken = self.try_take_on(context, found);
        if let Err(why) = &taken {
            self.error = Some(format!("taking on the stroke a restore found: {why:#}"));
            self.errors += 1;
        }
        taken
    }

    /// [`Sculpting::take_on`], its error not yet kept.
    ///
    /// # Errors
    ///
    /// As [`Sculpting::take_on`].
    fn try_take_on(&mut self, context: BrushContext<'_>, found: OpenStroke) -> Result<()> {
        let BrushContext {
            project,
            index,
            camera,
            size,
            drawn,
            ..
        } = context;
        if !self.strokes.is_empty() || !self.deferred.is_empty() || !self.held.is_empty() {
            return Err(anyhow!("a stroke is under way"));
        }
        let index = index.ok_or_else(|| anyhow!("there is no spatial index to brush"))?;
        // What a close or a flush lands it through if the camera goes degenerate: it has had no
        // step to leave one ([`Scene::closing`]).
        let projector = camera.view(size).as_ref().map(ViewProjector::new);
        let projector = projector
            .or(self.last_projector)
            .ok_or_else(|| anyhow!("there is no view to land it through"))?;
        if found.dab_format() != stroke::DAB_FORMAT {
            return Err(OtherDabFormat {
                found: found.dab_format(),
            }
            .into());
        }
        // Read out first: resuming takes the stroke. A row is a dab's bytes, so any row casts.
        let rows = found.dabs().iter();
        let dabs: VecDeque<Dab> = rows.map(|row: &DabRow| bytemuck::cast(*row)).collect();
        let model = index.bvh().bounding_sphere();
        let checked = dabs.iter().enumerate();
        for (at, dab) in checked {
            HostileDab::check(at, dab, model)?;
        }
        let (op, landed) = (found.op(), found.landed());
        match project.resume_stroke(found) {
            Ok(()) => {}
            Err(refused) if matches!(refused.error(), ProjectError::Cache(_)) => {
                project.resume_stroke(refused.into_stroke())?;
            }
            Err(refused) => return Err(refused.into()),
        }
        self.last_projector = Some(projector);
        // No pen sample placed them: they date from now, as a lift does.
        let now = Instant::now();
        let times = SampleTimes {
            taken: now,
            received: now,
        };
        let sources = dabs.iter().map(|dab| Source::of(dab, times, index));
        let sources: VecDeque<Source> = sources.collect();
        self.strokes.push(Pending {
            engine: None,
            record: None,
            op: StrokeOp::Begun(op),
            remaining: None,
            display_reads: None,
            dabs,
            landed,
            sources,
            last: times,
            placed_at: now,
            interval: FIRST_INTERVAL,
            // Its start isn't the pivot: opening frames the model.
            start_noted: true,
            down_at: now,
            heading: [None, None],
        });
        let mut report = BrushReport::default();
        self.show_on(project, index, drawn, &mut report);
        // Shown before any frame, which reports them; no pen sample times a latency from them.
        report.oldest = None;
        self.caught_up = std::mem::take(&mut self.caught_up).then(report);
        self.note_kept(project.store());
        self.refresh_status();
        Ok(())
    }
}

/// Finishes the stroke a restore of `project` found still landing, with no frame loop.
///
/// Taken on as [`Brush::resume`] takes it, and landed whole, its op ended, before anything else
/// edits or saves the project (a copy of found work, [`crate::ProjectToOpen::reopen`]). It is
/// shown nowhere: nothing draws.
///
/// It blocks the thread for the whole landing, the window's event loop included, with no frames
/// and in full batches: a copy is saved in one call that holds the thread anyway (the bar says
/// "Saving" first), the landing is at most a stroke's backlog, and a copy that lacked the stroke
/// would lose work the user asked to keep. The window is frozen meanwhile, for minutes with a
/// large backlog at scripted rates; accepted for now.
///
/// Returns what became of it: [`LastStroke::Whole`] if there was none or it landed, and
/// [`LastStroke::Unfinished`] if it can't be (a journal the project can't use, rows of another
/// layout, a project keeping no display layer to brush), said on stderr, the restored tiles
/// standing.
///
/// # Errors
///
/// If the spatial index can't be loaded, or landing it dropped dabs: what it would hold isn't all
/// there. The stroke has ended then, its dabs not landed counted landed, so a close that flushes
/// would leave a recovery file a retry finds no stroke in: close the project with
/// [`Project::close_untouched`] to keep the found work as it was.
pub(crate) fn finish_restored(project: &mut Project) -> Result<LastStroke> {
    let Some(found) = project.take_open_stroke() else {
        return Ok(LastStroke::Whole);
    };
    let stroke = match found {
        Ok(stroke) => stroke,
        Err(why) => {
            let why = anyhow::Error::new(why);
            crate::saving::log(&format!("brushcraft: the last stroke's journal: {why:#}"));
            return Ok(LastStroke::Unfinished);
        }
    };
    let Some(level) = project.store().display_finest() else {
        crate::saving::log("brushcraft: the last stroke can't land with no display layer");
        return Ok(LastStroke::Unfinished);
    };
    let (mut index, _) = project.load_index(level, |_, _| {})?;
    let mut brush = Sculpting::new(project);
    // Nothing draws, so nothing is shown ahead: every dab is finest work.
    brush.preview = None;
    let camera = Camera::default();
    let size = unseen_view()?;
    let taking = BrushContext {
        project: &mut *project,
        index: Some(&mut index),
        camera: &camera,
        size,
        drawn: Drawing::NONE,
        waiting: &[],
    };
    if let Err(why) = brush.resume(taking, stroke) {
        crate::saving::log(&format!("brushcraft: the last stroke: {why:#}"));
        return Ok(LastStroke::Unfinished);
    }
    let landing = BrushContext {
        project,
        index: Some(&mut index),
        camera: &camera,
        size,
        drawn: Drawing::NONE,
        waiting: &[],
    };
    brush.flush(landing)?;
    if brush.landing() > 0 {
        bail!("{} of the last stroke's dabs didn't land", brush.landing());
    }
    let reports = brush.take_recovery_reports().into_iter();
    for report in reports {
        let why = anyhow::Error::new(report);
        crate::saving::log(&format!("brushcraft: landing the last stroke: {why:#}"));
    }
    Ok(LastStroke::Whole)
}

/// The size of the view a stroke a restore found is taken on through before any window has a
/// size, or with none ([`finish_restored`]): its dabs are placed already, so any view does.
const UNSEEN_VIEW: (u32, u32) = (320, 200);

/// [`UNSEEN_VIEW`] as a size.
///
/// # Errors
///
/// Never: it isn't empty.
pub(crate) fn unseen_view() -> Result<render::surface::Size> {
    let (width, height) = UNSEEN_VIEW;
    render::surface::Size::new(width, height).ok_or_else(|| anyhow!("a view of {width} × {height}"))
}
