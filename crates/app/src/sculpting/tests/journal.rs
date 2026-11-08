//! The dab journal from the brush's side, read back from the recovery file's batches.
//!
//! Every dab placed goes to its stroke's journal once, in order, from the stroke's first; the
//! landed count follows the finest work, never the dabs shown; dabs an error drops are never
//! replayed; and a journal that refuses ends the stroke cleanly.

use store::{DabBatch, FLUSH_INTERVAL, SaveAhead};
use stroke::{MirrorPlanes, Symmetry};

use super::*;

/// Waits until `project`'s recovery writer has no flush under way: every cut taken so far is in
/// the recovery file, durable. A saved project's [`Project::save_ahead`] says so, and copies
/// nothing a reader sees.
fn settle(project: &mut Project) {
    let began = Instant::now();
    while matches!(
        project.save_ahead(Instant::now()).expect("a slice"),
        SaveAhead::Flushing
    ) {
        std::thread::sleep(Duration::from_millis(1));
        assert!(began.elapsed() < Duration::from_secs(60), "the flush ends");
    }
}

/// Has `project` cut its journal now, between brush calls as the frame loop does.
///
/// Ticks it at `at`, moved a flush interval past every tick before (so the tick starts a flush),
/// with no flush under way first (so it starts now, not queued behind one), and waits until that
/// flush is durable.
fn cut(project: &mut Project, at: &mut Instant) {
    settle(project);
    *at = (*at).max(Instant::now()) + FLUSH_INTERVAL * 2;
    project.tick(*at).expect("ticked");
    settle(project);
}

/// The batches of the stroke made as `op` in the recovery file at `path`, closed, in the order
/// they were written.
fn batches_at(path: &std::path::Path, op: OpId) -> Vec<DabBatch> {
    let file = ProjectFile::open(path).expect("the recovery file opens");
    let batches = file.batches().map(|batch| batch.expect("a batch"));
    let of_op = batches.filter(|batch| batch.op() == op);
    of_op.collect()
}

/// `dab` as the journal's row: its bytes.
fn row_of(dab: Dab) -> DabRow {
    bytemuck::cast(dab)
}

/// `brush` with its settings copied by `symmetry`, all else as it was.
fn symmetric(mut brush: Sculpting, symmetry: Symmetry) -> Sculpting {
    let settings = &brush.settings;
    let copied = StrokeSettings::new(
        *settings.path(),
        settings.spacing(),
        settings.size_mode(),
        settings.radius(),
        settings.dynamics().clone(),
        symmetry,
        settings.action_params(),
    )
    .expect("the brush's own settings");
    brush.settings = copied
        .with_strength(settings.strength())
        .with_strength_mode(settings.strength_mode())
        .with_hardness(settings.hardness());
    brush
}

/// A brush stepped directly over a project saved in a directory of its own.
///
/// By default a dab at a time is too dear for any time ([`DabCost::one_at_a_time`]): a frame's
/// step shows every dab it can't fit and lands none, and the tests land them one by one
/// ([`Journalling::land_one`]).
struct Journalling {
    /// The project's directory, kept until the batches are read.
    dir: TestDir,
    /// The project, saved: a saved project says when its flushes are done ([`settle`]).
    project: Project,
    /// Its spatial index.
    picker: Picker,
    /// The brush.
    brush: Sculpting,
    /// The camera, the default one.
    camera: Camera,
    /// When the last cut was ticked at ([`cut`]).
    ticked: Instant,
}

impl Journalling {
    /// The default brush over a fresh project, saved.
    fn new() -> Self {
        Self::with(|brush| brush)
    }

    /// The default brush as `made` makes it, over a fresh project, saved.
    fn with(made: impl FnOnce(Sculpting) -> Sculpting) -> Self {
        let dir = TestDir::new();
        let mut project = project(&dir);
        drop(
            project
                .save_as(&dir.path().join("journal.bcrf"))
                .expect("saved"),
        );
        let picker = Picker::load(&mut project)
            .expect("the index loads")
            .expect("a display layer");
        let brush = made(Sculpting::new(&mut project).with_cost(DabCost::one_at_a_time()));
        Self {
            dir,
            project,
            picker,
            brush,
            camera: Camera::default(),
            ticked: Instant::now(),
        }
    }

    /// A frame's step of the brush given `samples`, all routed to it: with the default brush, no
    /// dab fits, and none lands.
    fn frame(&mut self, samples: &[PenSample]) -> BrushReport {
        let mut pen = PenFrame::default();
        for sample in samples {
            pen.push(*sample, Some(Route::Brush));
        }
        let budget = BrushBudget::frame(Instant::now(), Duration::ZERO, setup().interval);
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        self.brush.step(context, &pen, &budget).expect("a step")
    }

    /// A step that lands one dab of the stroke under way, however dear: the pen (down or lifted)
    /// quiet for long, and the catch-up starved, so a step that must make progress takes one.
    fn land_one(&mut self) -> BrushReport {
        let long_ago = Instant::now()
            .checked_sub(RESTING_QUIET * 2)
            .expect("after the epoch");
        self.brush.landed_at = long_ago;
        self.brush.unframed = false;
        if let Some(stroke) = &mut self.brush.stroke {
            stroke.placed_at = long_ago;
            stroke.last.received = long_ago;
        }
        let context = context(&mut self.project, self.picker.index_mut(), &self.camera);
        let budget = BrushBudget::unlimited();
        self.brush
            .step(context, &PenFrame::default(), &budget)
            .expect("a step")
    }

    /// The stroke under way.
    fn stroke(&self) -> &Pending {
        self.brush.stroke.as_ref().expect("a stroke under way")
    }

    /// The rows of the stroke's dabs not yet landed, in order.
    fn queued_rows(&self) -> Vec<DabRow> {
        let rows = self.stroke().dabs.iter().copied().map(row_of);
        rows.collect()
    }

    /// Cuts the project's journal now ([`cut`]).
    fn cut(&mut self) {
        cut(&mut self.project, &mut self.ticked);
    }

    /// Closes the project, keeping its recovery file, and reads back the batches of the stroke
    /// made as `op`.
    fn close(self, op: OpId) -> Vec<DabBatch> {
        let Self { dir, project, .. } = self;
        let path = project.recovery_path().expect("a recovery file").to_owned();
        project.close(true).expect("closed");
        let batches = batches_at(&path, op);
        drop(dir);
        batches
    }

    /// Two strokes back to back, the second waiting to begin: the first (`Down` to `Up`) is
    /// shown and none of it lands; a tick cuts the journal, its flush held in its tile sync until
    /// a stroke waits for its cut ([`Project::hold_recovery_syncs_until_waited`]), so it is in
    /// flight whatever polls and however the threads are scheduled; the second's `Down` to `Up`
    /// waits for the first to land. Whatever lands the first lands it after that cut, so the
    /// second begins waiting for the next flush's, and its wait lets the held flush go.
    fn second_waits(&mut self) {
        let first = self.frame(&stroke(ACROSS, 12));
        assert!(first.dabs_shown > 0, "{first:?}");
        self.project.hold_recovery_syncs_until_waited();
        self.ticked = self.ticked.max(Instant::now()) + FLUSH_INTERVAL * 2;
        self.project.tick(self.ticked).expect("ticked");
        self.frame(&below());
        assert!(!self.brush.deferred.is_empty(), "the second waits to begin");
    }

    /// Asserts that the second stroke began waiting for its cut and the step waited for it, that
    /// every dab of both strokes has landed, and that no stroke waits to.
    fn assert_both_landed(&self) {
        assert!(
            self.brush.waited_to_land > 0,
            "the second stroke began waiting for its cut, and the step waited"
        );
        assert_eq!(self.brush.landing(), 0, "nothing left to land");
        assert!(self.brush.stroke.is_none(), "no stroke under way");
        assert!(self.project.may_land(), "and none waits");
    }
}

/// [`stroke()`] along [`ACROSS`] in 12 drags, without its `Up`: the pen stays down.
fn pen_left_down() -> Vec<PenSample> {
    let mut samples = stroke(ACROSS, 12);
    samples.pop();
    samples
}

#[test]
fn every_dab_placed_is_journalled_in_order() {
    let mut run = Journalling::new();
    let mut samples = stroke(ACROSS, 12);
    let later = samples.split_off(7);
    let (drags, up) = later.split_at(later.len() - 1);
    // Three pushes' worth, each cut apart: the Down and six drags, six more, then the pen's lift.
    let first = run.frame(&samples);
    assert!(first.dabs_shown > 0, "{first:?}");
    let op = run.stroke().op;
    run.cut();
    run.frame(drags);
    run.cut();
    run.frame(up);
    let placed = run.queued_rows();
    assert!(placed.len() > first.dabs_shown, "the tail placed dabs");
    assert!(!run.stroke().open(), "the pen lifted");
    let batches = run.close(op);
    assert_eq!(batches.len(), 3, "a batch a cut, and the close's");
    let mut next = 0;
    for batch in &batches {
        assert_eq!(batch.first(), next, "each from where the last left off");
        assert!(!batch.dabs().is_empty(), "each with the dabs since");
        assert_eq!(batch.landed(), 0, "none landed");
        next = batch.placed();
    }
    let rows = batches
        .iter()
        .flat_map(|batch| batch.dabs().iter().copied());
    let journalled: Vec<DabRow> = rows.collect();
    assert_eq!(journalled, placed, "every dab placed, once, in order");
}

#[test]
fn the_landed_count_follows_the_finest_work() {
    let mut run = Journalling::new();
    let shown = run.frame(&pen_left_down());
    assert!(shown.dabs_shown > 3, "{shown:?}");
    assert_eq!(shown.dabs_applied, 0, "a frame lands none of them");
    let op = run.stroke().op;
    run.cut();
    let mut applied = 0;
    for landing in [1, 2] {
        for _ in 0..landing {
            let landed = run.land_one();
            assert_eq!(landed.dabs_applied, 1, "{landed:?}");
            applied += landed.dabs_applied;
        }
        assert_eq!(
            run.brush.catching_up(),
            shown.dabs_shown - applied,
            "the rest still shown"
        );
        run.cut();
    }
    assert_eq!(run.brush.queued(), shown.dabs_shown - applied);
    let batches = run.close(op);
    let landed: Vec<u32> = batches.iter().map(DabBatch::landed).collect();
    assert_eq!(
        landed,
        [0, 1, 3],
        "at each cut, the dabs applied and none of those only shown"
    );
    let placed = u32::try_from(shown.dabs_shown).expect("a few");
    let all_placed = batches.iter().all(|batch| batch.placed() == placed);
    assert!(all_placed, "every dab written by the first cut");
}

#[test]
fn a_stroke_shown_ahead_is_journalled_before_any_dab_lands() {
    let mut run = Journalling::new();
    let samples = pen_left_down();
    let (down, drags) = samples.split_at(1);
    // The op begins at the `Down`, before its first dab: the journal is open for it.
    run.frame(down);
    let op = run.stroke().op;
    // Frame after frame of a big brush: every dab shown at once, none landed.
    let mut shown = 0;
    let frames = drags.chunks(3);
    for frame in frames {
        let report = run.frame(frame);
        assert_eq!(report.dabs_applied, 0, "{report:?}");
        shown += report.dabs_shown;
    }
    assert!(shown > 3, "{shown} dabs shown");
    assert!(!run.project.has_unsaved_edits(), "no dab has landed");
    let placed = run.queued_rows();
    assert_eq!(placed.len(), shown);
    run.cut();
    let batches = run.close(op);
    let [batch] = batches.as_slice() else {
        panic!("one batch, the cut's: {batches:?}");
    };
    assert_eq!((batch.first(), batch.landed()), (0, 0));
    assert_eq!(batch.dabs(), placed.as_slice(), "every dab shown, in order");
}

#[test]
fn dabs_dropped_after_an_error_are_never_replayed() {
    let (dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    let light = owner.brush().stroke.as_ref().expect("its pen down");
    let op = light.op;
    assert!(light.dabs.is_empty(), "the light half has landed");
    // Held from landing while they are placed, shown and cut: the journal holds them unlanded.
    owner.brush_mut().landing = Landing::InSteps;
    queue(&mut owner, heavy);
    run(&mut owner, 1);
    let waiting = owner
        .brush()
        .stroke
        .as_ref()
        .expect("its dabs wait")
        .dabs
        .len();
    let waiting = u32::try_from(waiting).expect("a few");
    assert!(waiting > 0, "the heavy dabs wait to land");
    cut(owner.project(), &mut Instant::now());
    owner.brush_mut().landing = Landing::InFrames;
    run(&mut owner, 10_000);
    let error = owner.brush().error().map(str::to_owned);
    let error = error.expect("the heavy dabs don't fit the least budget the light ones do");
    assert!(error.contains("applying dabs"), "{error}");
    assert!(!owner.brush().stroking(), "the stroke ended");
    let project = owner.project();
    let path = project.recovery_path().expect("a recovery file").to_owned();
    owner.close().expect("closed");
    let batches = batches_at(&path, op);
    let counts: Vec<(u32, u32)> = batches
        .iter()
        .map(|batch| (batch.placed(), batch.landed()))
        .collect();
    // The cut's batch: every dab but the heavy ones waiting had landed.
    let at_cut = counts
        .iter()
        .find(|&&(placed, landed)| placed == landed + waiting);
    let &(placed, _) =
        at_cut.unwrap_or_else(|| panic!("the cut journalled them unlanded: {counts:?}"));
    assert_eq!(
        counts.last(),
        Some(&(placed, placed)),
        "the stroke's end counts every dab landed: a restore lands none again"
    );
    drop(dir);
}

#[test]
fn a_journal_that_refuses_ends_the_stroke_cleanly_and_says_so() {
    let mut run = Journalling::new();
    let mut samples = stroke(ACROSS, 12);
    samples.pop();
    let later = samples.split_off(7);
    run.frame(&samples);
    let queued = run.queued_rows();
    assert!(!queued.is_empty(), "dabs placed");
    let drops = run.brush.drops();
    // The project's stroke ends behind the brush's back: its journal is gone.
    run.project.end_unrecorded_stroke().expect("ended");
    run.frame(&later);
    let error = run.brush.error().expect("said").to_owned();
    assert!(error.contains("journalling the dabs placed"), "{error}");
    assert_eq!(
        run.queued_rows(),
        queued,
        "the push's dabs dropped, not queued"
    );
    assert!(!run.stroke().open(), "the engine stroke ended");
    assert_eq!(run.brush.drops(), drops + 1, "dabs dropped");
    // The dabs queued before land, and the journal refuses their count: the stroke ends there.
    let landed = run.land_one();
    assert_eq!(landed.dabs_applied, 1, "{landed:?}");
    assert!(!run.brush.stroking(), "the stroke ended in that step");
    assert_eq!(run.brush.drops(), drops + 2, "the rest dropped");
    assert!(run.brush.error().is_some(), "said");
    run.project.close(false).expect("closed");
}

/// The common path, with `symmetry`: no previews, the dabs landing as finest work in the steps
/// that place them, a cut while some wait and one once none does.
///
/// Its oracles are independent of the brush's own count: the dabs come from the same samples
/// through a brush that lands none (placement reads the pen-down surface, so the dabs are the
/// same, bit for bit), and the landed counts from what each step says it applied.
fn landed_at_once_is_counted_as_applied(symmetry: Symmetry) {
    let epoch = Instant::now();
    let samples = stroke(ACROSS, 12);
    let (first, rest) = samples.split_at(7);
    let (drags, up) = rest.split_at(rest.len() - 1);
    let mut oracle = Journalling::with(|brush| symmetric(brush.with_epoch(epoch), symmetry));
    oracle.frame(&samples);
    let placed = oracle.queued_rows();
    oracle.project.close(false).expect("closed");
    // A dab a step while the cost is dear (a brush showing nothing makes progress), then all.
    let mut run = Journalling::with(|brush| {
        symmetric(brush.with_epoch(epoch).with_previews(false), symmetry)
    });
    let one = run.frame(first).dabs_applied;
    assert_eq!(one, 1, "a dab at a time");
    assert!(!run.stroke().dabs.is_empty(), "some wait");
    let op = run.stroke().op;
    run.cut();
    run.brush.cost = DabCost::any_number();
    let more = run.frame(drags).dabs_applied;
    assert!(run.stroke().dabs.is_empty(), "the rest landed at once");
    run.cut();
    let tail = run.frame(up).dabs_applied;
    assert!(!run.brush.stroking(), "the stroke ended with its tail");
    let batches = run.close(op);
    let counts = batches
        .iter()
        .map(|batch| usize::try_from(batch.landed()).expect("a few"));
    let landed: Vec<usize> = counts.collect();
    assert_eq!(
        landed,
        [one, one + more, one + more + tail],
        "the cuts', then the end's: what the steps applied"
    );
    assert_eq!(
        landed.last(),
        Some(&placed.len()),
        "every dab placed landed"
    );
    for batch in &batches {
        let index = |at: u32| usize::try_from(at).expect("a few");
        let span = index(batch.first())..index(batch.placed());
        let expected = placed.get(span.clone()).expect("dabs placed");
        assert_eq!(batch.dabs(), expected, "the dabs {span:?} as placed");
    }
    assert!(
        batches.iter().any(|batch| !batch.dabs().is_empty()),
        "the dabs waiting at the first cut were written"
    );
    let copies = placed
        .iter()
        .map(|&row| {
            let dab: Dab = bytemuck::cast(row);
            dab.copy
        })
        .filter(|&copy| copy != 0)
        .count();
    assert_eq!(
        copies > 0,
        symmetry.copies() > 1,
        "the copies are rows of their own: {copies} of {}",
        placed.len()
    );
}

#[test]
fn dabs_landed_as_they_are_placed_are_counted_by_what_each_step_applied() {
    landed_at_once_is_counted_as_applied(Symmetry::NONE);
}

#[test]
fn symmetry_copies_are_journalled_and_counted_landed_as_dabs_of_their_own() {
    landed_at_once_is_counted_as_applied(Symmetry::mirror(MirrorPlanes::X));
}

#[test]
fn a_stroke_shows_its_dabs_and_lands_none_until_its_cut_is_taken() {
    let mut run = Journalling::new();
    // A stroke of the project's own, before the brush's: a tick cuts it midway, and it goes on
    // editing after that cut, the flush still in flight (nothing polls the project between).
    let level = Level::new(LEVEL).expect("a level");
    let face = geom::FaceId::new(0);
    let before = run.project.begin_stroke_and_wait().expect("a stroke");
    let edit = |project: &mut Project, i: u32, z: f32| {
        project
            .store()
            .set_correction(face, level, i, 9, geom::Vec3::new(0.0, 0.0, z))
            .expect("an edit");
    };
    edit(&mut run.project, 9, 1e-3);
    run.ticked = run.ticked.max(Instant::now()) + FLUSH_INTERVAL * 2;
    run.project.tick(run.ticked).expect("ticked");
    edit(&mut run.project, 11, -1e-3);
    assert!(
        run.project.may_land(),
        "the project's own stroke edits freely"
    );

    let shown = run.frame(&pen_left_down());
    let op = run.stroke().op;
    assert_ne!(op, before, "the brush's own stroke");
    assert!(
        !run.project.may_land(),
        "back to back: it waits for its cut"
    );
    assert!(shown.dabs_shown > 1, "its dabs shown ahead: {shown:?}");
    let waiting = run.land_one();
    assert_eq!(waiting.dabs_applied, 0, "none lands before its cut");
    assert_eq!(run.brush.catching_up(), shown.dabs_shown, "all still shown");

    // As the frame loop does between steps, until the flush queued behind has cut.
    let began = Instant::now();
    while !run.project.may_land() {
        run.project.tick(Instant::now()).expect("ticked");
        run.project.poll().expect("polled");
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "it may land within the flush's syncs"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let landed = run.land_one();
    assert_eq!(landed.dabs_applied, 1, "then it lands: {landed:?}");
    run.cut();
    let batches = run.close(op);
    assert!(!batches.is_empty(), "journalled");
    assert!(
        batches
            .iter()
            .all(|batch| matches!(batch.pen_down(), store::PenDown::At(_))),
        "naming the save point its cut made: {batches:?}"
    );
}

/// A stroke along the sphere's middle lower down than [`ACROSS`], `Down` to `Up`.
fn below() -> Vec<PenSample> {
    stroke(((130.0, 120.0), (190.0, 130.0)), 12)
}

#[test]
fn a_flush_lands_a_stroke_that_begins_waiting_for_its_cut() {
    let mut run = Journalling::new();
    run.second_waits();
    let flushing = context(&mut run.project, run.picker.index_mut(), &run.camera);
    run.brush.flush(flushing).expect("flushed");
    run.assert_both_landed();
}

#[test]
fn a_close_lands_a_stroke_that_begins_waiting_for_its_cut() {
    let mut run = Journalling::new();
    run.second_waits();
    let closing = context(&mut run.project, run.picker.index_mut(), &run.camera);
    run.brush.begin_closing(closing).expect("closing");
    let began = Instant::now();
    while run.brush.landing() > 0 {
        let step = context(&mut run.project, run.picker.index_mut(), &run.camera);
        run.brush.close_step(step).expect("a step");
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the close lands it all"
        );
    }
    run.assert_both_landed();
}

#[test]
fn a_save_lands_a_stroke_that_begins_waiting_for_its_cut() {
    let mut run = Journalling::new();
    run.second_waits();
    run.brush.begin_saving(Instant::now());
    run.brush.begin_landing();
    let began = Instant::now();
    while run.brush.landing_for_save() > 0 {
        let step = context(&mut run.project, run.picker.index_mut(), &run.camera);
        run.brush.save_step(step).expect("a step");
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the save lands it all"
        );
    }
    run.brush.end_saving();
    run.assert_both_landed();
}
