//! The strokes a restore found still landing: taken on by the brush, one after another in op
//! order, or finished with no frame loop for a copy of found work.
//!
//! See the [`super`] module notes, "Strokes a restore found", the dab journal spec, "Restoring a
//! journalled stroke", and the stroke-while-landing spec, "Showing and landing" (restore).

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use store::{OpenStroke, Project, ProjectError, SpatialIndex};
use stroke::Dab;

use super::{DabRow, FIRST_INTERVAL, Pending, Scene, Sculpting, Source, StrokeOp, ViewProjector};
use crate::camera::Camera;
use crate::dialogs::LastStroke;
use crate::frame::{Brush, BrushContext, BrushReport, Drawing, NotTakenOn, SampleTimes};

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

/// What a restored stroke's dabs are checked against ([`HostileDab::check`]).
#[derive(Debug, Clone, Copy)]
enum Against {
    /// The model as the index holds it now, the sphere holding its tiles (`None` for an empty
    /// index, which nothing can be a dab of): for a stroke resuming now, which lands on it.
    Model(Option<([f64; 3], f64)>),
    /// Their numbers alone: for a later stroke of a chain, its surface the one the strokes before
    /// it leave, which the index doesn't hold yet. It is checked against the model as it resumes.
    Numbers,
}

impl HostileDab {
    /// Checks `dab`, the `at`th not landed, `against` the model or for its numbers alone.
    ///
    /// # Errors
    ///
    /// Why it can't be a dab of this model.
    fn check(at: usize, dab: &Dab, against: Against) -> Result<(), Self> {
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
        let radius = f64::from(dab.radius);
        if radius <= 0.0 {
            return hostile("a radius out of the model's range");
        }
        let Against::Model(model) = against else {
            return Ok(());
        };
        let Some((centre, model_radius)) = model else {
            return hostile("the model has no tiles");
        };
        if radius > 2.0 * model_radius {
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

/// The dabs of `found`, a stroke a restore found, as the brush lands them: its rows read back as
/// dabs, each checked `against` the model or for its numbers alone.
///
/// # Errors
///
/// [`OtherDabFormat`] for rows of another layout than the brush's, [`HostileDab`] for a row that
/// can't be a dab of the model.
fn dabs_of(found: &OpenStroke, against: Against) -> Result<VecDeque<Dab>> {
    if found.dab_format() != stroke::DAB_FORMAT {
        return Err(OtherDabFormat {
            found: found.dab_format(),
        }
        .into());
    }
    // A row is a dab's bytes, so any row casts.
    let rows = found.dabs().iter();
    let dabs: VecDeque<Dab> = rows.map(|row: &DabRow| bytemuck::cast(*row)).collect();
    let checked = dabs.iter().enumerate();
    for (at, dab) in checked {
        HostileDab::check(at, dab, against)?;
    }
    Ok(dabs)
}

impl Sculpting {
    /// Takes on `found`, the strokes a restore found still landing, in op order, through
    /// `context` (see "Strokes a restore found" in the module notes): the first resumed in the
    /// project, its dabs queued as a stroke whose pen has lifted and shown at once; the rest kept
    /// to resume in turn, each once the one before has landed and ended
    /// ([`Sculpting::resume_next`]).
    ///
    /// Each is checked first: the first whose rows the brush can't read, and every one after it,
    /// which lands on its surface, isn't taken on. With every stroke it takes resumed, the
    /// project lets the rest go ([`Project::let_go_of_restored_strokes`]), closing them in the
    /// file. A project that refuses a stroke for a transient reason (the store's cache) is asked
    /// once more.
    ///
    /// # Errors
    ///
    /// [`NotTakenOn`], kept as the brush's error too: the strokes from the first it doesn't take
    /// on, and why. All of them with a stroke under way or waiting (never, before any step), no
    /// spatial index (nothing can be brushed), no view to land them through, or a project that
    /// won't resume the first; those from the first with rows of another layout
    /// ([`OtherDabFormat`]) or a row that can't be a dab of this model ([`HostileDab`]).
    pub(super) fn take_on(
        &mut self,
        context: BrushContext<'_>,
        found: Vec<OpenStroke>,
    ) -> Result<(), NotTakenOn> {
        let taken = self.try_take_on(context, found);
        if let Err(not_taken) = &taken {
            self.error = Some(format!(
                "taking on the strokes a restore found: {:#}",
                not_taken.why
            ));
            self.errors += 1;
        }
        taken
    }

    /// [`Sculpting::take_on`], its error not yet kept.
    ///
    /// # Errors
    ///
    /// As [`Sculpting::take_on`].
    fn try_take_on(
        &mut self,
        context: BrushContext<'_>,
        mut found: Vec<OpenStroke>,
    ) -> Result<(), NotTakenOn> {
        let BrushContext {
            project,
            index,
            camera,
            size,
            drawn,
            ..
        } = context;
        let count = found.len();
        let none_taken = |why: anyhow::Error| NotTakenOn::new(count, why).map_or(Ok(()), Err);
        let busy = !self.strokes.is_empty()
            || !self.deferred.is_empty()
            || !self.held.is_empty()
            || !self.restored.is_empty();
        if busy {
            return none_taken(anyhow!("a stroke is under way"));
        }
        let Some(index) = index else {
            return none_taken(anyhow!("there is no spatial index to brush"));
        };
        // What a close or a flush lands them through if the camera goes degenerate: they have had
        // no step to leave one ([`Scene::closing`]).
        let projector = camera.view(size).as_ref().map(ViewProjector::new);
        let Some(projector) = projector.or(self.last_projector) else {
            return none_taken(anyhow!("there is no view to land them through"));
        };
        // The first lands on the model as the index holds it; each later one on what the strokes
        // before it leave, checked against that as it resumes.
        let model = Against::Model(index.bvh().bounding_sphere());
        let checks = found.iter().enumerate().map(|(at, stroke)| {
            let against = if at == 0 { model } else { Against::Numbers };
            dabs_of(stroke, against).map(drop)
        });
        let first_refused = checks
            .enumerate()
            .find_map(|(at, checked)| checked.err().map(|why| (at, why)));
        let refused = first_refused.and_then(|(at, why)| NotTakenOn::new(count - at, why));
        if let Some(refused) = &refused {
            found.truncate(count - refused.strokes.get());
        }
        let mut found = VecDeque::from(found);
        let Some(first) = found.pop_front() else {
            // None to take: none of them ever resumes.
            drop(project.let_go_of_restored_strokes());
            return refused.map_or(Ok(()), Err);
        };
        if let Err(why) = self.resume_found(project, index, &projector, drawn, first) {
            drop(project.let_go_of_restored_strokes());
            return none_taken(why);
        }
        self.restored = found;
        self.after_resuming(project);
        refused.map_or(Ok(()), Err)
    }

    /// The next stroke a restore found resumes, the one before it having landed and ended (its op
    /// ended in this call), on the surface it left: queued and shown as the first was
    /// ([`Sculpting::take_on`]). With every one resumed, the rest are let go
    /// ([`Sculpting::after_resuming`]).
    ///
    /// One the project won't resume ends the chain: it and every one after it are let go
    /// ([`Project::let_go_of_restored_strokes`], closing them in the file), their dabs never
    /// landed, kept as an error that dropped dabs, which the next save owns up to.
    pub(super) fn resume_next(&mut self, scene: &mut Scene<'_>) {
        let Some(next) = self.restored.pop_front() else {
            return;
        };
        let projector = scene.projector;
        let resumed = self.resume_found(scene.project, scene.index, &projector, scene.drawn, next);
        if let Err(why) = resumed {
            self.restored.clear();
            drop(scene.project.let_go_of_restored_strokes());
            self.dropped("resuming a stroke a restore found", why.as_ref());
            return;
        }
        self.after_resuming(scene.project);
    }

    /// Once every stroke a restore found that the brush takes on has resumed, the project lets
    /// the rest go (those it didn't take on, or couldn't use), closing them in the file: none
    /// resumes after the last taken, and a later crash's restore never offers them again.
    fn after_resuming(&self, project: &mut Project) {
        if self.restored.is_empty() {
            drop(project.let_go_of_restored_strokes());
        }
    }

    /// Resumes `found`, a stroke a restore found, in `project`, and holds it as a stroke whose pen
    /// has lifted, with no engine stroke and nothing to record: its dabs queued, each with the
    /// tiles of `index` it touches, and shown at once, previews made of the tiles `drawn`.
    /// `projector` is kept as the last a step had: what a close lands it through.
    ///
    /// A project that refuses it for a transient reason (the store's cache) is asked once more.
    ///
    /// # Errors
    ///
    /// As [`dabs_of`], or the project's refusal ([`Project::resume_stroke`]); nothing is kept
    /// then.
    fn resume_found(
        &mut self,
        project: &mut Project,
        index: &mut SpatialIndex,
        projector: &ViewProjector,
        drawn: Drawing<'_>,
        found: OpenStroke,
    ) -> Result<()> {
        // Read out first: resuming takes the stroke.
        let dabs = dabs_of(&found, Against::Model(index.bvh().bounding_sphere()))?;
        let (op, landed) = (found.op(), found.landed());
        match project.resume_stroke(found) {
            Ok(()) => {}
            Err(refused) if matches!(refused.error(), ProjectError::Cache(_)) => {
                project.resume_stroke(refused.into_stroke())?;
            }
            Err(refused) => return Err(refused.into()),
        }
        self.last_projector = Some(*projector);
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
        // Shown before any frame (or between them), which reports them; no pen sample times a
        // latency from them.
        report.oldest = None;
        self.caught_up = std::mem::take(&mut self.caught_up).then(report);
        self.note_kept(project.store());
        self.refresh_status();
        Ok(())
    }
}

/// Finishes the strokes a restore of `project` found still landing, with no frame loop.
///
/// Taken on as [`Brush::resume`] takes them, and landed whole, one after another in op order,
/// each op ended, before anything else edits or saves the project (a copy of found work,
/// [`crate::ProjectToOpen::reopen`]). They are shown nowhere: nothing draws.
///
/// It blocks the thread for the whole landing, the window's event loop included, with no frames
/// and in full batches: a copy is saved in one call that holds the thread anyway (the bar says
/// "Saving" first), the landing is at most the strokes' backlog, and a copy that lacked them
/// would lose work the user asked to keep. The window is frozen meanwhile, for minutes with a
/// large backlog at scripted rates; accepted for now.
///
/// Returns what became of them: [`LastStroke::Whole`] if there were none or all landed, and
/// [`LastStroke::Unfinished`], one or several, for those that can't be (a journal the project
/// can't use, rows of another layout, a project keeping no display layer to brush, or a stroke
/// after one of those), said on stderr, the restored tiles standing with what the strokes before
/// them landed.
///
/// # Errors
///
/// If the spatial index can't be loaded, or landing them dropped dabs: what they would hold isn't
/// all there. The stroke has ended then, its dabs not landed counted landed, so a close that
/// flushes would leave a recovery file a retry finds no stroke in: close the project with
/// [`Project::close_untouched`] to keep the found work as it was.
pub(crate) fn finish_restored(project: &mut Project) -> Result<LastStroke> {
    let found = project.take_open_strokes();
    if found.is_empty() {
        return Ok(LastStroke::Whole);
    }
    let mut unfinished = 0;
    let mut usable: Vec<OpenStroke> = Vec::with_capacity(found.len());
    for stroke in found {
        match stroke {
            Ok(stroke) => usable.push(stroke),
            Err(why) => {
                let why = anyhow::Error::new(why);
                crate::saving::log(&format!("brushcraft: a last stroke's journal: {why:#}"));
                unfinished += 1;
            }
        }
    }
    if usable.is_empty() {
        return Ok(LastStroke::of(unfinished));
    }
    let Some(level) = project.store().display_finest() else {
        crate::saving::log("brushcraft: the last strokes can't land with no display layer");
        return Ok(LastStroke::of(unfinished + usable.len()));
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
    if let Err(not_taken) = brush.resume(taking, usable) {
        crate::saving::log(&format!(
            "brushcraft: {} of the last strokes: {:#}",
            not_taken.strokes, not_taken.why
        ));
        unfinished += not_taken.strokes.get();
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
        bail!("{} of the last strokes' dabs didn't land", brush.landing());
    }
    let reports = brush.take_recovery_reports().into_iter();
    for report in reports {
        let why = anyhow::Error::new(report);
        crate::saving::log(&format!("brushcraft: landing the last strokes: {why:#}"));
    }
    Ok(LastStroke::of(unfinished))
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
