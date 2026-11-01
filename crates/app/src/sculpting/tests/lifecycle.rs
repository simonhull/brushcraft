//! A stroke's lifecycle at its edges: a lost `Up`, a stroke started while dabs are carried, the
//! project closing mid-stroke, a store error mid-stroke; and warming's deadline and cancellation.

use super::*;

/// `samples` with the last (its `Up`) replaced by `end` at the same place: a stroke whose `Up`
/// was lost.
fn ended_by(mut samples: Vec<PenSample>, end: Phase) -> Vec<PenSample> {
    if let Some(last) = samples.last_mut() {
        last.phase = end;
    }
    samples
}

/// `samples` without the last (its `Up`): a stroke still open.
fn open(mut samples: Vec<PenSample>) -> Vec<PenSample> {
    samples.pop();
    samples
}

#[test]
fn a_lost_up_ends_the_stroke_at_the_next_sample_out_of_a_stroke() {
    for end in [Phase::Hover, Phase::Leave, Phase::Enter] {
        let dir = TestDir::new();
        let mut owner = owner(&dir);
        run(&mut owner, 10_000);
        queue(&mut owner, ended_by(stroke(ACROSS, 12), end));
        run(&mut owner, 10_000);
        assert!(!owner.brush().stroking(), "{end:?} ended it");
        assert!(owner.brush().last().dabs_applied > 0, "{end:?}: it drew");
        assert_eq!(saved_strokes(owner, &dir, "lost.bcrf").len(), 1, "{end:?}");
    }
}

#[test]
fn a_lost_up_ends_the_stroke_as_a_navigation_stroke_starts() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, open(stroke(ACROSS, 12)));
    run(&mut owner, 1);
    assert!(owner.brush().stroking(), "open");
    // A barrel held: the next stroke navigates, and the brush never sees its samples.
    let barrel = |sample: PenSample| PenSample {
        buttons: input::Buttons::default().with(input::Buttons::SECONDARY, true),
        ..sample
    };
    let navigation: Vec<PenSample> = stroke(((160.0, 60.0), (200.0, 60.0)), 4)
        .into_iter()
        .map(barrel)
        .collect();
    queue(&mut owner, navigation);
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "the brush's stroke ended");
    assert_eq!(saved_strokes(owner, &dir, "nav.bcrf").len(), 1);
}

#[test]
fn a_stroke_started_while_dabs_are_carried_waits_behind_them() {
    let dir = TestDir::new();
    let mut owner = owner_of(
        project(&dir),
        |project| Sculpting::new(project).with_cost(DabCost::one_at_a_time()),
        setup(),
    );
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    assert_eq!(run(&mut owner, 1), 1);
    let carried = owner.brush().last().dabs_carried;
    assert!(carried > 2, "the first stroke carries dabs: {carried}");
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    assert_eq!(run(&mut owner, 1), 1);
    assert_eq!(
        owner.brush().strokes_pending(),
        2,
        "the second waits behind the first"
    );
    assert_eq!(
        owner.brush().last().dabs_applied,
        1,
        "still one dab a frame"
    );
    let mut frames = 0;
    while owner.brush().stroking() {
        assert_eq!(run(&mut owner, 1), 1, "frames while dabs are carried");
        assert_eq!(owner.brush().last().dabs_applied, 1, "never all at once");
        frames += 1;
        assert!(frames < 10_000, "the strokes end");
    }
    assert_eq!(owner.brush().error(), None);
    assert_eq!(
        saved_strokes(owner, &dir, "behind.bcrf").len(),
        2,
        "a record each"
    );
}

#[test]
fn closing_mid_stroke_applies_it_and_records_it_once() {
    let dir = TestDir::new();
    let mut owner = owner_of(
        project(&dir),
        |project| Sculpting::new(project).with_cost(DabCost::one_at_a_time()),
        setup(),
    );
    run(&mut owner, 10_000);
    queue(&mut owner, open(stroke(ACROSS, 12)));
    assert_eq!(run(&mut owner, 1), 1);
    assert!(
        owner.brush().last().dabs_carried > 0,
        "dabs carried as it closes"
    );
    let recovery = owner
        .project()
        .recovery_path()
        .expect("a recovery file")
        .to_path_buf();
    owner.close().expect("closed");
    let file = ProjectFile::open(&recovery).expect("the recovery file opens");
    let records: Vec<StrokeRecord> = file
        .strokes()
        .map(|record| record.expect("a record"))
        .collect();
    assert_eq!(records.len(), 1, "one record, by the flush");
    assert_eq!(records[0].len(), 13, "the down and the twelve drags");
}

/// A store under `budget` bytes, in small chunks with little set aside, keeping a display layer
/// under level 7: 96 tiles, so a light dab pins a few of them and a dab over the whole sphere
/// all.
fn tight(budget: usize) -> StoreConfig {
    StoreConfig::new(budget, 1)
        .with_chunk_bytes(64 << 10)
        .with_kept_predictions(0)
        .with_stroke_tiles(8)
        .with_display(Level::new(7).expect("a level"))
}

/// A stroke on a fresh project under `tight(budget)`, light first, then heavy.
///
/// Its brush 400 pixels, its pressure light (0.01: small dabs) for its first half and full (dabs
/// over the whole sphere) after. Returns the frame loop with its light half drawn, and the heavy
/// half's samples. `None` if the project doesn't open or pick under that budget, or the light half
/// doesn't fit it.
fn light_then_heavy(
    dir: &TestDir,
    budget: usize,
) -> Option<(Owner<Sculpting, NoUploads>, Vec<PenSample>)> {
    let config = ProjectConfig {
        store: tight(budget),
        recovery_dir: dir.path().join("recovery"),
    };
    let mut project = Project::new(cube_sphere(2).ok()?, config).ok()?;
    let Ok(Some(picker)) = Picker::load(&mut project) else {
        project.close(false).expect("closed");
        return None;
    };
    let mut brush = Sculpting::new(&mut project);
    brush.set_radius(400.0);
    let mut owner =
        Owner::new(project, brush, NoUploads, setup(), Instant::now()).with_picker(picker);
    run(&mut owner, 10_000);
    let mut samples = stroke(ACROSS, 24);
    let count = samples.len();
    let pressed = samples.iter_mut().enumerate().take(count - 1);
    for (k, sample) in pressed {
        let pressure = if k < count / 2 { 0.01 } else { 1.0 };
        sample.pressure = input::Unit::try_from(pressure).expect("a pressure");
    }
    let light: Vec<PenSample> = samples.drain(..count / 2).collect();
    queue(&mut owner, light);
    run(&mut owner, 10_000);
    if owner.brush().last().dabs_applied == 0 || owner.brush().error().is_some() {
        owner.close().expect("closed");
        return None;
    }
    Some((owner, samples))
}

#[test]
fn a_store_error_mid_stroke_ends_it_with_one_record_and_nothing_after() {
    // The least budget the light dabs fit under, found by halving: there the heavy dabs, over
    // the whole sphere, can't pin their tiles.
    let minimum = tight(0).minimum_budget(&cube_sphere(2).expect("a cube sphere"));
    let budget = |step: usize| minimum + step * (64 << 10);
    let fits = |step: usize| {
        let dir = TestDir::new();
        light_then_heavy(&dir, budget(step))
            .map(|(owner, _)| owner.close().expect("closed"))
            .is_some()
    };
    let (mut low, mut high) = (0, 512);
    assert!(fits(high), "the light dabs fit a generous budget");
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if fits(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    let dir = TestDir::new();
    let (mut owner, heavy) = light_then_heavy(&dir, budget(high)).expect("it fits, as found");
    queue(&mut owner, heavy);
    run(&mut owner, 10_000);
    let error = owner.brush().error().map(str::to_owned);
    let error = error.expect("the heavy dabs don't fit the least budget the light ones do");
    assert!(error.contains("applying dabs"), "{error}");
    assert!(!owner.brush().stroking(), "the stroke ended");
    let applied = owner.brush().last();
    run(&mut owner, 10_000);
    assert_eq!(
        owner.brush().last(),
        applied,
        "nothing applied after the error"
    );
    assert_eq!(
        saved_strokes(owner, &dir, "failed.bcrf").len(),
        1,
        "one record"
    );
}

/// The brush's step and preparation over `project` and `index` with the pen hovering at `at`
/// (`None`: out of range), the preparation's deadline `deadline`; returns whether warming is
/// left.
fn hover_frame(
    brush: &mut Sculpting,
    (project, index): (&mut Project, &mut SpatialIndex),
    at: Option<(f64, f64)>,
    deadline: Instant,
) -> bool {
    let camera = Camera::default();
    let mut pen = PenFrame::default();
    let hover = at.map_or_else(
        || sample(Phase::Leave, (0.0, 0.0), 0.0),
        |at| sample(Phase::Hover, at, 0.0),
    );
    pen.push(hover, None);
    brush
        .step(
            context(project, index, &camera),
            &pen,
            &BrushBudget::unlimited(),
        )
        .expect("a step");
    brush
        .prepare(context(project, index, &camera), deadline)
        .expect("prepared")
}

#[test]
fn warming_progresses_past_its_deadline_and_a_moved_pen_cancels_it() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    project.store().evict_all().expect("evicted");
    let past = Instant::now();
    let centre = Some((160.0, 100.0));
    // A deadline already passed: each call still does one piece, the list, then a batch.
    let first = hover_frame(&mut brush, (&mut project, picker.index_mut()), centre, past);
    let listed = brush.warm.todo.len();
    assert!(first && listed > 0, "listed, nothing more: {listed}");
    let second = hover_frame(&mut brush, (&mut project, picker.index_mut()), centre, past);
    assert_eq!(
        brush.warm.todo.len(),
        listed.saturating_sub(WARM_BATCH),
        "a batch"
    );
    assert_eq!(second, !brush.warm.todo.is_empty());
    // Moved on: what was left is dropped, and the new place is listed afresh.
    project.store().evict_all().expect("evicted");
    let moved = Some((150.0, 90.0));
    hover_frame(&mut brush, (&mut project, picker.index_mut()), moved, past);
    assert_eq!(brush.warm.target, brush.warm.wanted, "the new place");
    // Out of range: nothing to warm.
    let left = hover_frame(&mut brush, (&mut project, picker.index_mut()), None, past);
    assert!(!left && brush.warm.todo.is_empty(), "cancelled");
    assert_eq!(brush.warm.wanted, None);
    project.close(false).expect("closed");
}
