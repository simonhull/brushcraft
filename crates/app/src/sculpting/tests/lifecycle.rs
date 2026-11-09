//! A stroke's lifecycle at its edges, and warming's deadline and cancellation.
//!
//! The edges: a lost `Up`, a stroke started while dabs are carried (it waits for them, its
//! pen-down surface what they leave, its samples held over a step with no scene, and a `Down` on
//! their fresh ridge brushing), the project closing mid-stroke, a store error mid-stroke.

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
fn a_stroke_started_while_dabs_are_carried_begins_at_once_and_lands_after_them() {
    let dir = TestDir::new();
    // F = ∞: nothing shown ahead, every dab carried until its finest work.
    let mut owner = owner_of(
        project(&dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_previews(false)
        },
        setup(),
    );
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    assert_eq!(run(&mut owner, 1), 1);
    let carried = owner.brush().last().dabs_carried;
    assert!(carried > 2, "the first stroke carries dabs: {carried}");
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    assert_eq!(run(&mut owner, 1), 1);
    assert!(
        owner.brush().deferred.is_empty() && owner.brush().strokes.len() == 2,
        "the second begins at once, on what the first will leave"
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
    let last = owner.brush().last();
    assert_eq!(last.dabs_carried, 0, "every dab shown");
    assert!(last.dabs_catching_up > 0, "dabs to catch up as it closes");
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

/// A stroke on a fresh project, saved as `edge.bcrf`, under `tight(budget)`, light first, then
/// heavy.
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
    drop(project.save_as(&dir.path().join("edge.bcrf")).ok()?);
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
        drop(owner.close());
        return None;
    }
    Some((owner, samples))
}

/// A frame loop on about the least budget the light dabs of [`light_then_heavy`] fit, with its
/// directory, and the heavy dabs, which don't.
///
/// Found by halving, then going up from there: how a run's dabs fall into batches is up to the
/// clock, a batch pins the tiles of all its dabs, and at the very edge one run fits where the next
/// doesn't. Nothing is asked of the close of a run that's only looked at.
pub(super) fn at_the_budgets_edge() -> (TestDir, Owner<Sculpting, NoUploads>, Vec<PenSample>) {
    let minimum = tight(0).minimum_budget(&cube_sphere(2).expect("a cube sphere"));
    let budget = |step: usize| minimum + step * (64 << 10);
    let fits = |step: usize| {
        let dir = TestDir::new();
        light_then_heavy(&dir, budget(step))
            .map(|(owner, _)| drop(owner.close()))
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
    (high..=512)
        .find_map(|step| {
            let dir = TestDir::new();
            let made = light_then_heavy(&dir, budget(step));
            made.map(|(owner, heavy)| (dir, owner, heavy))
        })
        .expect("it fits a generous budget")
}

/// A store failing as a close lands reaches the process: the dabs it drops aren't lost in silence.
#[test]
fn a_store_error_while_closing_is_an_error_of_the_close() {
    let (_dir, mut owner, heavy) = at_the_budgets_edge();
    // Held for a frame that never came: the close takes them in, and they don't fit.
    owner.brush_mut().held.extend(heavy.into_iter().map(Some));
    owner.request_close().expect("a close");
    let mut steps = 0;
    let failed = loop {
        match owner.close_step() {
            Err(error) => break error,
            Ok(_) => assert!(!owner.ready_to_close(), "landed with no failure"),
        }
        steps += 1;
        assert!(steps < 10_000, "the landing ends");
    };
    assert!(failed.to_string().contains("applying dabs"), "{failed:#}");
}

#[test]
fn a_store_error_mid_stroke_ends_it_with_one_record_and_nothing_after() {
    let (dir, mut owner, heavy) = at_the_budgets_edge();
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

/// A stroke of `op` under way with no dabs left to place, standing for one whose dabs still land:
/// what a `Down` waits behind.
fn landing_stroke(camera: &Camera, brush: &Sculpting, op: OpId) -> Pending {
    let view = camera.view(size()).expect("a view");
    let down = sample(Phase::Down, (160.0, 100.0), 0.5);
    Pending {
        engine: None,
        record: Some(Recording {
            settings: brush.settings.clone(),
            camera: stroke_camera(camera, &view),
            samples: Vec::new(),
        }),
        op: StrokeOp::Begun(op),
        remaining: None,
        display_reads: None,
        landed: 0,
        dabs: VecDeque::from([Dab::default()]),
        sources: VecDeque::new(),
        last: times_of(&down),
        placed_at: Instant::now(),
        interval: FIRST_INTERVAL,
        start_noted: true,
        down_at: down.received,
        heading: [None, None],
    }
}

#[test]
fn a_waiting_strokes_samples_are_held_over_a_step_with_no_scene() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    let camera = Camera::default();
    // The earlier stroke ended between frames; the next one's `Down` still waits.
    brush
        .deferred
        .push(Some(sample(Phase::Down, (150.0, 100.0), 0.5)));
    let mut pen = PenFrame::default();
    let drags = stroke(((150.0, 100.0), (180.0, 104.0)), 6);
    let count = drags.len();
    for drag in drags.into_iter().skip(1) {
        pen.push(drag, Some(Route::Brush));
    }
    let blind = BrushContext {
        project: &mut project,
        index: None,
        camera: &camera,
        size: size(),
        drawn: Drawing::ALL,
        waiting: &[],
    };
    let report = brush
        .step(blind, &pen, &BrushBudget::unlimited())
        .expect("a step");
    assert!(report.busy, "a stroke waits: the next frame is wanted");
    assert_eq!(
        brush.held.len(),
        count - 1,
        "its samples held, none dropped"
    );
    brush
        .step(
            context(&mut project, picker.index_mut(), &camera),
            &PenFrame::default(),
            &BrushBudget::unlimited(),
        )
        .expect("a step");
    assert!(!brush.stroking(), "it began, took every sample and ended");
    project.close(false).expect("closed");
}

#[test]
fn a_down_on_a_landing_strokes_fresh_ridge_brushes() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    let camera = Camera::default();
    let projector = ViewProjector::new(&camera.view(size()).expect("a view"));
    let ray_at = |x: f64| {
        projector
            .ray(ScreenPoint::new(x, 100.0).expect("a point"))
            .expect("a ray")
    };
    let pen_down_hit = |of: &mut Project, index: &SpatialIndex, x: f64| {
        let mut surface = StoreSurface::new(of.store(), index);
        let hit = surface.raycast(&ray_at(x));
        surface.finish().expect("no store error");
        hit
    };
    // The sphere's silhouette along the middle row.
    let edge = (160..320)
        .map(f64::from)
        .take_while(|&x| pen_down_hit(&mut project, picker.index_mut(), x).is_some())
        .last()
        .expect("the sphere is under the middle");
    let near = pen_down_hit(&mut project, picker.index_mut(), edge - 4.0).expect("a hit");
    // The earlier stroke raises a bump at the silhouette, its pen-down shapes kept.
    let op = project.begin_stroke_and_wait().expect("a stroke");
    let mut state = sculpt::Stroke::new(project.store());
    let centre = near.position.to_f32();
    let outward = near.normal.to_f32();
    let bump = Dab {
        position: [centre.x, centre.y, centre.z],
        normal: [outward.x, outward.y, outward.z],
        radius: 0.3,
        strength: 0.25,
        ..Dab::default()
    };
    apply_dabs(project.store(), picker.index_mut(), &mut state, &[bump]).expect("applied");
    brush.strokes = vec![landing_stroke(&camera, &brush, op)];
    // A pixel past the silhouette the stroke began on, on the bump.
    let on_bump = (1..40).map(|d| edge + f64::from(d)).find(|&x| {
        let missed = pen_down_hit(&mut project, picker.index_mut(), x).is_none();
        let ray = ray_at(x);
        let narrow =
            store::Ray::new(ray.origin().to_f32(), ray.direction().to_f32()).expect("a ray");
        let landed = picker
            .index_mut()
            .raycast_triangle(project.store(), &narrow)
            .expect("casts");
        missed && landed.is_some()
    });
    let x = on_bump.expect("the bump stands past the silhouette");
    let down = sample(Phase::Down, (x, 100.0), 0.5);
    let waiting = brush.route(context(&mut project, picker.index_mut(), &camera), &down);
    assert_eq!(waiting, Route::Brush, "a Down on the fresh ridge brushes");
    brush.strokes.clear();
    let alone = brush.route(context(&mut project, picker.index_mut(), &camera), &down);
    assert_eq!(
        alone,
        Route::Navigate,
        "with no stroke landing, the pen-down surface decides"
    );
    project.end_unrecorded_stroke().expect("ended");
    project.close(false).expect("closed");
}
