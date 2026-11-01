//! The brush on a small generated sphere, with no GPU.
//!
//! Routing on and off the surface, strokes into the store and their records, the dab budget,
//! warming under a hovering pen, the keys, and a scripted run's record.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use geom::{Level, TileKey, cube_sphere};
use render::surface::{Presentation, Size};
use render::{DrawTimes, Outcome};
use store::{ProjectConfig, ProjectFile, StoreConfig};

use super::*;
use crate::frame::schedule::Next;
use crate::frame::{NoPen, NoUploads, Owner, RecordSetup, Setup};
use crate::pick::Picker;
use crate::render_thread::Presented;
use crate::script::{Script, Speed};
use crate::test_dir::TestDir;

mod lifecycle;

/// The level the test projects are sculpted at: a 24-face sphere's 24 tiles of 32 × 32 quads.
const LEVEL: u8 = 5;

/// The view the tests draw into.
fn size() -> Size {
    Size::new(320, 200).expect("non-zero")
}

/// The frame loop's setup for a 320 × 200 offscreen run.
fn setup() -> Setup {
    Setup {
        size: Some(size()),
        interval: Duration::from_micros(16_667),
        presentation: Presentation::Offscreen,
        adapter: "test".to_owned(),
        frames: None,
        script: None,
        record: None,
    }
}

/// A never-saved project over a 24-face cube sphere (radius about 1), keeping a display layer
/// under [`LEVEL`], its recovery file in `dir`.
fn project(dir: &TestDir) -> Project {
    let config = ProjectConfig {
        store: StoreConfig::new(256 << 20, 1).with_display(Level::new(LEVEL).expect("a level")),
        recovery_dir: dir.path().join("recovery"),
    };
    Project::new(cube_sphere(2).expect("a cube sphere"), config).expect("a new project")
}

/// The frame loop over `project`, its brush made by `brush`, picking by its index, set up by
/// `setup`.
fn owner_of(
    mut project: Project,
    brush: impl FnOnce(&mut Project) -> Sculpting,
    setup: Setup,
) -> Owner<Sculpting, NoUploads> {
    let picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let brush = brush(&mut project);
    Owner::new(project, brush, NoUploads, setup, Instant::now()).with_picker(picker)
}

/// The frame loop over a fresh project in `dir` with the default brush.
fn owner(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    owner_of(project(dir), Sculpting::new, setup())
}

/// The render thread's report of frame `number`: shown, its present submitted now.
fn presented(number: u64) -> Presented {
    Presented {
        number,
        outcome: Outcome::Presented,
        times: DrawTimes::default(),
        scene: false,
        shown: false,
        aa: render::aa::AaReport::default(),
        gpu: Vec::new(),
        dropped: render::DroppedReadbacks::default(),
        gpu_finished: number,
        submitted: Instant::now(),
    }
}

/// Makes frames on the real clock while they are wanted, up to `most`, each shown at once;
/// returns how many it made.
fn run(owner: &mut Owner<Sculpting, NoUploads>, most: u64) -> u64 {
    let mut made = 0;
    while made < most && !owner.done() {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                let mut number = 0;
                owner
                    .frame(now, deadline, |request| {
                        number = request.number;
                        Ok(())
                    })
                    .expect("a frame");
                owner.shown(presented(number));
                made += 1;
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu => panic!("every frame was shown and finished"),
            Next::Idle { .. } => break,
        }
    }
    made
}

/// A pen sample in `phase` at `(x, y)` with `pressure`, taken and received now.
fn sample(phase: Phase, (x, y): (f64, f64), pressure: f32) -> PenSample {
    let now = Instant::now();
    PenSample {
        phase,
        tool: input::Tool::Pen,
        at: Pixel { x, y },
        pressure: input::Unit::try_from(pressure).expect("a pressure"),
        distance: None,
        tilt: None,
        roll: None,
        buttons: input::Buttons::default(),
        timestamp: now,
        received: now,
    }
}

/// A stroke from `from` to `to` in `steps` drags, then up: its samples, oldest first, taken
/// at the scripted pen's rate up to now.
fn stroke((from, to): ((f64, f64), (f64, f64)), steps: u32) -> Vec<PenSample> {
    let at = |k: u32| {
        let t = f64::from(k) / f64::from(steps);
        (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t)
    };
    let drags = (1..=steps).map(|k| sample(Phase::Drag, at(k), 0.8));
    let samples = std::iter::once(sample(Phase::Down, from, 0.8))
        .chain(drags)
        .chain(std::iter::once(sample(Phase::Up, to, 0.0)));
    let mut samples: Vec<PenSample> = samples.collect();
    let now = Instant::now();
    let period = Duration::from_secs_f64(crate::script::PEN_RATE.recip());
    let count = u32::try_from(samples.len()).expect("a few");
    let numbered = samples.iter_mut().zip((0..count).rev());
    for (sample, before) in numbered {
        sample.timestamp = now.checked_sub(period * before).expect("after the epoch");
        sample.received = sample.timestamp;
    }
    samples
}

/// Queues `samples` for the next frame.
fn queue(owner: &mut Owner<Sculpting, NoUploads>, samples: Vec<PenSample>) {
    for sample in samples {
        owner.pen(sample);
    }
}

/// A stroke across the sphere's middle, as the default camera sees it.
const ACROSS: ((f64, f64), (f64, f64)) = ((130.0, 100.0), (190.0, 110.0));

/// Every tile of [`LEVEL`] of `project`'s mesh, face by face.
fn tile_keys(project: &mut Project) -> Vec<TileKey> {
    let level = Level::new(LEVEL).expect("a level");
    let per_side = u16::try_from(level.tiles_per_side()).expect("a few");
    let keys = project.store().mesh().faces().flat_map(|face| {
        (0..per_side).flat_map(move |ty| {
            (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).expect("a tile"))
        })
    });
    keys.collect()
}

/// Every tile of [`LEVEL`]'s evaluated positions, in bits.
fn positions(project: &mut Project) -> BTreeMap<TileKey, Vec<[u32; 3]>> {
    let keys = tile_keys(project);
    let store = project.store();
    let tiles = keys.into_iter().map(|key| {
        let bits = store
            .evaluate(key)
            .expect("evaluated")
            .iter()
            .map(|p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
            .collect();
        (key, bits)
    });
    tiles.collect()
}

/// The brush's context over `project` and `index`, seen by `camera` in a view of [`size`].
fn context<'a>(
    project: &'a mut Project,
    index: &'a mut SpatialIndex,
    camera: &'a Camera,
) -> BrushContext<'a> {
    BrushContext {
        project,
        index: Some(index),
        camera,
        size: size(),
    }
}

/// The stroke records `owner`'s project holds once saved as `name` in `dir`, read back once
/// it has closed.
fn saved_strokes(
    mut owner: Owner<Sculpting, NoUploads>,
    dir: &TestDir,
    name: &str,
) -> Vec<StrokeRecord> {
    let path = dir.path().join(name);
    let saved = owner.project().save_as(&path).expect("saved");
    assert!(saved.unprotected.is_empty(), "{saved:?}");
    owner.close().expect("closed");
    let file = ProjectFile::open(&path).expect("opened");
    let records = file.strokes().map(|record| record.expect("a record"));
    records.collect()
}

#[test]
fn a_down_on_the_surface_brushes_and_off_it_navigates() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    let camera = Camera::default();
    let mut route = |index: Option<&mut SpatialIndex>, at| {
        let context = BrushContext {
            project: &mut project,
            index,
            camera: &camera,
            size: size(),
        };
        brush.route(context, &sample(Phase::Down, at, 0.5))
    };
    assert_eq!(
        route(Some(picker.index_mut()), (160.0, 100.0)),
        Route::Brush
    );
    assert_eq!(route(Some(picker.index_mut()), (3.0, 3.0)), Route::Navigate);
    assert_eq!(route(None, (160.0, 100.0)), Route::Navigate, "no index");
    project.close(false).expect("closed");
}

#[test]
fn no_stroke_starts_off_the_surface() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    // From a corner, off the sphere, sweeping onto it: the stroke orbits, whole.
    queue(&mut owner, stroke(((5.0, 5.0), (160.0, 100.0)), 10));
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "no stroke");
    assert_eq!(owner.brush().last().dabs_applied, 0, "no dab");
    assert!(!owner.project().has_unsaved_edits(), "nothing edited");
    assert_ne!(*owner.camera(), Camera::default(), "the stroke orbited");
    owner.close().expect("closed");
}

#[test]
fn a_stroke_edits_the_project_and_is_recorded_once() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    let samples = stroke(ACROSS, 12);
    queue(&mut owner, samples);
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "the stroke ended");
    assert!(owner.brush().last().dabs_applied > 0, "dabs applied");
    assert!(owner.brush().last().tiles_edited > 0, "tiles edited");
    assert!(
        owner.project().has_unsaved_edits(),
        "the project was edited"
    );
    assert_eq!(
        *owner.camera(),
        Camera::default(),
        "the stroke didn't orbit"
    );
    assert_eq!(owner.brush().error(), None);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 10_000);
    let settings = owner.brush().settings().to_bytes();
    let records = saved_strokes(owner, &dir, "two.bcrf");
    assert_eq!(records.len(), 2, "a record a stroke");
    let counts: Vec<usize> = records.iter().map(StrokeRecord::len).collect();
    // The engine reads the down and the drags; the up only ends the stroke.
    assert_eq!(counts, [13, 7], "each stroke's samples");
    for record in &records {
        assert_eq!(record.engine(), stroke::ENGINE_VERSION);
        assert_eq!(record.settings(), settings.as_slice());
        assert_eq!(record.camera().size, [320, 200]);
        assert_eq!(record.camera().fov_y, Camera::default().fov_y());
    }
    let first: Vec<RecordSample> = records[0].samples().collect();
    assert!(
        (first[0].x - 130.0).abs() < 0.1 && (first[0].pressure - 0.8).abs() < 1e-3,
        "the first sample: {:?}",
        first[0]
    );
}

#[test]
fn a_drag_with_no_pressure_reaches_nothing_and_the_stroke_goes_on() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    let mut samples = stroke(ACROSS, 12);
    // Every drag of the stroke's first half without pressure: the engine places no dab there.
    for sample in samples.iter_mut().skip(1).take(6) {
        sample.pressure = input::Unit::ZERO;
    }
    queue(&mut owner, samples);
    run(&mut owner, 10_000);
    assert_eq!(owner.brush().error(), None, "no dab refused");
    assert!(
        owner.brush().last().dabs_applied > 0,
        "the pressed half drew"
    );
    assert!(!owner.brush().stroking(), "the stroke ended");
    assert_eq!(saved_strokes(owner, &dir, "light.bcrf").len(), 1);
}

/// The overlay's picks `owner` has made so far.
fn picks(owner: &Owner<Sculpting, NoUploads>) -> u64 {
    owner.picker().expect("a picker").picks()
}

#[test]
fn the_overlays_pick_waits_for_the_stroke_to_end() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    let samples = stroke(ACROSS, 12);
    let (first, rest) = samples.split_at(6);
    let before = picks(&owner);
    // The stroke's first half in one frame, its second (with the up) in the next.
    queue(&mut owner, first.to_vec());
    assert_eq!(run(&mut owner, 1), 1);
    assert_eq!(picks(&owner), before, "no pick in a frame of the stroke");
    queue(&mut owner, rest.to_vec());
    // Its last frame, and those that apply the dabs it carried.
    for _ in 0..1000 {
        assert_eq!(run(&mut owner, 1), 1, "frames while it is stroking");
        assert_eq!(picks(&owner), before, "no pick while it is stroking");
        if !owner.brush().stroking() {
            break;
        }
    }
    assert!(!owner.brush().stroking(), "the stroke is over");
    run(&mut owner, 10_000);
    assert_eq!(picks(&owner), before + 1, "then the pick, once");
    owner.close().expect("closed");
}

/// `n` distinct display tiles in a row along one face's first row, from `first` on.
fn tiles(first: u16, n: u16) -> Vec<TileKey> {
    let level = Level::new(12).expect("a level");
    let keys = (first..first + n)
        .map(|tx| TileKey::display(geom::FaceId::new(0), level, tx, 0).expect("on the face"));
    keys.collect()
}

#[test]
fn the_time_left_takes_dabs_from_the_front_and_at_least_one() {
    // A nanosecond a dab's tile, nothing else: a dab of one tile takes 1 ns.
    let cost = DabCost::new(0.0, 0.0, 1.0, 0.0);
    let one = tiles(0, 1);
    let ns = Duration::from_nanos;
    let take = |dabs: &[&[TileKey]], room: Duration, at_least_one: bool| {
        cost.take(dabs.iter().copied(), room, at_least_one).0
    };
    assert_eq!(take(&[&one, &one, &one], ns(2), true), 2, "what fits");
    assert_eq!(take(&[&one, &one], ns(0), true), 1, "at least one");
    assert_eq!(take(&[&one, &one], ns(0), false), 0, "none once applied");
    assert_eq!(take(&[], ns(10), true), 0, "none to take");
    let big = tiles(0, 30);
    assert_eq!(
        take(&[&one, &big, &one], ns(5), true),
        1,
        "a run from the front, not a selection"
    );
}

#[test]
fn a_batch_pays_its_part_its_dabs_their_tiles_and_their_union_once() {
    let (a, b) = (tiles(0, 4), tiles(2, 4));
    let room = Duration::from_secs(1);
    let cost = |model: DabCost| model.take([a.as_slice(), &b, &a], room, true);
    assert_eq!(
        cost(DabCost::new(7.0, 0.0, 0.0, 0.0)),
        (3, 7.0),
        "once a batch"
    );
    assert_eq!(
        cost(DabCost::new(0.0, 5.0, 0.0, 0.0)),
        (3, 15.0),
        "each dab"
    );
    assert_eq!(
        cost(DabCost::new(0.0, 0.0, 1.0, 0.0)),
        (3, 12.0),
        "each dab's tiles"
    );
    assert_eq!(
        cost(DabCost::new(0.0, 0.0, 0.0, 1.0)),
        (3, 6.0),
        "six tiles among them, once each"
    );
    assert_eq!(
        DabCost::new(7.0, 0.0, 0.0, 0.0).take(std::iter::empty::<&[TileKey]>(), room, true),
        (0, 0.0),
        "no dab, no batch"
    );
}

#[test]
fn the_calibration_fits_the_bench_within_its_spread() {
    // Part A's p99 per call, in ms, with the tiles a dab reaches and their union as the bench's
    // stroke reaches them: (dabs, tiles a dab, union, ms).
    let points: [(u16, u16, u16, f64); 6] = [
        (1, 2, 2, 0.52),
        (8, 2, 3, 1.10),
        (32, 2, 4, 3.09),
        (1, 21, 21, 2.55),
        (8, 21, 27, 6.08),
        (32, 21, 51, 22.0),
    ];
    for (dabs, each, union, ms) in points {
        // Each dab's tiles a window sliding along the union, as the stroke moves.
        let slide = union - each;
        let windows = (0..dabs).map(|k| {
            let offset = if dabs > 1 { k * slide / (dabs - 1) } else { 0 };
            tiles(offset, each)
        });
        let batch: Vec<Vec<TileKey>> = windows.collect();
        let (count, predicted) =
            DabCost::CALIBRATED.take(batch.iter().map(Vec::as_slice), Duration::MAX, true);
        assert_eq!(count, usize::from(dabs));
        let fitted = predicted / 1e6;
        assert!(
            (fitted - ms).abs() < 0.6,
            "{dabs} dabs of {each} tiles: {fitted:.2} ms against {ms} ms"
        );
    }
}

#[test]
fn the_cost_learns_how_long_batches_take_here() {
    // A microsecond a dab's tile: a dab of one tile takes 1 us as calibrated.
    let mut cost = DabCost::new(0.0, 0.0, 1000.0, 0.0);
    let one = tiles(0, 1);
    let ns = |us: u64| Duration::from_nanos(us * 1000);
    // Four and a half dabs' time, as calibrated.
    let room = Duration::from_nanos(4500);
    assert_eq!(cost.take([one.as_slice(); 8], room, true).0, 4);
    // A batch took four times its prediction: the scale rises, but only doubles.
    cost.observe(1000.0, ns(4));
    assert_eq!(cost.take([one.as_slice(); 8], room, true).0, 2, "doubled");
    cost.observe(1000.0, ns(4));
    assert_eq!(
        cost.take([one.as_slice(); 8], room, true).0,
        1,
        "four times"
    );
    // Faster batches lower it an eighth of the way each.
    for _ in 0..40 {
        cost.observe(1000.0, ns(1));
    }
    assert_eq!(cost.take([one.as_slice(); 8], room, true).0, 4, "back");
    // One far off is believed only to double, however far.
    cost.observe(1000.0, Duration::from_secs(3600));
    assert_eq!(cost.take([one.as_slice(); 8], room, true).0, 2);
}

#[test]
fn a_frames_brush_has_the_aim_less_the_frames_other_work_and_more_once_behind() {
    let start = Instant::now();
    let interval = Duration::from_micros(16_667);
    let after = Duration::from_millis(2);
    let budget = BrushBudget::frame(start, after, interval);
    let fresh = start
        .checked_sub(Duration::from_millis(1))
        .expect("after boot");
    assert_eq!(
        budget.until(fresh),
        start + LATENCY_AIM.saturating_sub(after),
        "the time left before the frame must be submitted"
    );
    let late = start
        .checked_sub(LATENCY_AIM + Duration::from_millis(1))
        .expect("after boot");
    assert_eq!(
        budget.until(late),
        start + interval,
        "behind: up to a refresh interval, to catch up"
    );
    let full = BrushBudget::frame(start, Duration::from_secs(1), interval);
    assert!(
        full.until(fresh) > start,
        "a frame whose other work fills the aim still brushes a little"
    );
    let between = BrushBudget::slice(start, Duration::from_millis(1));
    assert_eq!(between.until(late), start + Duration::from_millis(1));
}

#[test]
fn the_budget_carries_dabs_across_frames_in_order_and_loses_none() {
    let (capped_dir, whole_dir) = (TestDir::new(), TestDir::new());
    let epoch = Instant::now();
    // F = ∞: nothing shown ahead, every dab carried until its finest work.
    let mut capped = owner_of(
        project(&capped_dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_epoch(epoch)
                .with_previews(false)
        },
        setup(),
    );
    let mut whole = owner_of(
        project(&whole_dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::any_number())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut capped, 10_000);
    run(&mut whole, 10_000);
    // The same samples, all in one frame: the same dabs placed.
    let samples = stroke(ACROSS, 12);
    queue(&mut capped, samples.clone());
    queue(&mut whole, samples);
    run(&mut whole, 10_000);
    let placed = whole.brush().last().dabs_applied;
    assert!(placed > 3, "a stroke of several dabs: {placed}");
    assert_eq!(whole.brush().last().dabs_carried, 0, "all in one frame");
    // One dab a frame, in order, until none is left; the stroke ends after the last.
    let mut applied = Vec::new();
    let mut carried = Vec::new();
    while applied.len() <= placed {
        assert_eq!(run(&mut capped, 1), 1, "frames come while dabs are carried");
        let last = capped.brush().last();
        applied.push(last.dabs_applied);
        carried.push(last.dabs_carried);
        if last.dabs_carried == 0 {
            break;
        }
        assert!(capped.brush().stroking(), "the stroke waits for its dabs");
    }
    assert!(applied.iter().all(|&n| n == 1), "one a frame: {applied:?}");
    let expected: Vec<usize> = (0..placed).rev().collect();
    assert_eq!(carried, expected, "carried down to none");
    run(&mut capped, 10_000);
    assert!(!capped.brush().stroking(), "then the stroke ended");
    assert_eq!(
        positions(capped.project()),
        positions(whole.project()),
        "the same surface, bit for bit"
    );
    whole.close().expect("closed");
    assert_eq!(saved_strokes(capped, &capped_dir, "c.bcrf").len(), 1);
}

#[test]
fn between_frames_a_strokes_dabs_are_applied_once_and_shown_by_the_next_frame() {
    let dir = TestDir::new();
    let mut owner = owner_of(
        project(&dir),
        |project| Sculpting::new(project).with_cost(DabCost::any_number()),
        setup(),
    );
    run(&mut owner, 10_000);
    assert!(
        !owner
            .between_frames(Instant::now(), &mut NoPen)
            .expect("a step"),
        "no stroke: nothing to do between frames"
    );
    // A frame in flight: the render thread has it, and the loop would wait.
    let now = Instant::now();
    let mut number = 0;
    owner
        .frame(now, now + Duration::from_secs(1), |request| {
            number = request.number;
            Ok(())
        })
        .expect("a frame");
    assert_eq!(owner.next(Instant::now()), Next::Render);
    // A down and drags arrive meanwhile (the stroke stays open).
    let mut samples = stroke(ACROSS, 12);
    samples.pop();
    let first = samples.first().expect("a down").timestamp;
    let last = samples.last().expect("drags").timestamp;
    queue(&mut owner, samples);
    let mut steps = 0;
    while owner
        .between_frames(Instant::now(), &mut NoPen)
        .expect("a step")
    {
        steps += 1;
        assert!(steps < 1000, "the loop would spin");
    }
    let between = owner.between_report().expect("the brush stepped");
    assert!(steps > 0 && between.dabs_applied > 0, "{between:?}");
    assert_eq!(between.dabs_carried, 0, "all of them, with time for any");
    let oldest = between.oldest.expect("a sample applied");
    assert!(
        (first..=last).contains(&oldest.taken),
        "one of the stroke's samples"
    );
    assert!(
        !owner
            .between_frames(Instant::now(), &mut NoPen)
            .expect("a step"),
        "nothing queued or carried: no more work, no spin"
    );
    owner.shown(presented(number));
    let epochs = owner.feed_stats().epochs;
    let next = owner.next(Instant::now());
    let Next::Frame { deadline } = next else {
        panic!("the edits want their frame at once: {next:?}");
    };
    owner
        .frame(Instant::now(), deadline, |request| {
            number = request.number;
            Ok(())
        })
        .expect("a frame");
    assert!(
        owner.feed_stats().epochs > epochs,
        "the next frame drained the epochs the steps closed"
    );
    assert_eq!(
        owner.latency_from_of(number),
        Some(oldest),
        "its latency runs from the sample applied between frames"
    );
    assert_eq!(owner.between_report(), None, "counted once, by that frame");
    assert_eq!(
        owner.brush().last().dabs_applied,
        between.dabs_applied,
        "the frame applied none again"
    );
    owner.shown(presented(number));
    owner.close().expect("closed");
}

#[test]
fn hovering_warms_the_surface_under_the_pen_once() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    let mut brush = Sculpting::new(&mut project);
    let camera = Camera::default();
    project.store().evict_all().expect("evicted");
    let keys = tile_keys(&mut project);
    let cached = |store: &TileStore| {
        let warm = keys
            .iter()
            .filter(|key| store.cached_positions(key).is_some());
        warm.count()
    };
    assert_eq!(cached(project.store()), 0, "cold");
    let mut pen = PenFrame::default();
    pen.push(sample(Phase::Hover, (160.0, 100.0), 0.0), None);
    let call = |warming: &mut Sculpting, open: &mut Project, index: &mut SpatialIndex| {
        warming
            .step(
                context(open, index, &camera),
                &pen,
                &BrushBudget::unlimited(),
            )
            .expect("a step");
        let deadline = Instant::now() + Duration::from_secs(10);
        warming
            .prepare(context(open, index, &camera), deadline)
            .expect("prepared")
    };
    let more = call(&mut brush, &mut project, picker.index_mut());
    assert!(!more, "warmed within the deadline");
    let warmed = cached(project.store());
    assert!(warmed > 0, "the tiles under the pen were evaluated");
    // A still pen isn't warmed again: nothing is left, and nothing is listed again.
    assert!(!call(&mut brush, &mut project, picker.index_mut()));
    assert!(brush.warm.todo.is_empty(), "nothing listed");
    assert_eq!(brush.warm.target, brush.warm.wanted, "the same target");
    project.close(false).expect("closed");
}

#[test]
fn keys_change_the_size_and_the_strength_within_bounds() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let mut brush = Sculpting::new(&mut project);
    let strength = brush.settings().strength().get();
    assert!(brush.change(BrushChange::Larger));
    assert_eq!(brush.settings().radius().get(), 50.0);
    assert_eq!(
        brush.settings().strength_mode(),
        StrengthMode::Radius,
        "a new size keeps the strength in radii"
    );
    assert!(brush.change(BrushChange::Smaller));
    assert!(brush.change(BrushChange::Smaller));
    assert_eq!(brush.settings().radius().get(), 32.0);
    assert!(brush.change(BrushChange::Stronger));
    assert!((brush.settings().strength().get() - strength * 1.25).abs() < 1e-9);
    let weakest = (0..100)
        .filter(|_| brush.change(BrushChange::Weaker))
        .count();
    assert!(weakest < 100, "the strength stops at its least");
    let least = brush.settings().strength().get();
    assert!((least - strength / 64.0).abs() < 1e-9, "{least}");
    let text = brush.status().join("\n");
    assert!(text.contains("radius 32.0 px"), "{text}");
    assert!(text.contains("keys: [ ] size, - = strength"), "{text}");
    project.close(false).expect("closed");
}

#[test]
fn the_default_strength_is_a_tenth_of_the_strokes_radius() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let brush = Sculpting::new(&mut project);
    let settings = brush.settings();
    assert_eq!(settings.strength_mode(), StrengthMode::Radius, "in radii");
    assert_eq!(settings.strength().get(), DEFAULT_STRENGTH_FRACTION);
    let text = brush.status().join("\n");
    assert!(
        text.contains("strength 0.1000 of the radius (x1.00)"),
        "{text}"
    );
    project.close(false).expect("closed");
}

/// The value of `field` in a record line, as written.
fn field<'a>(line: &'a str, field: &str) -> &'a str {
    let key = format!("\"{field}\":");
    let start = line.find(&key).expect("the field") + key.len();
    let rest = line.get(start..).expect("its value");
    let end = rest.find([',', '}']).expect("the value's end");
    rest.get(..end).expect("the value")
}

#[test]
fn a_scripted_run_strokes_the_project_and_records_its_latency() {
    let dir = TestDir::new();
    let setup = Setup {
        script: Some(Script::strokes("8".parse::<Speed>().expect("a speed"))),
        record: Some(RecordSetup {
            project: "test".to_owned(),
            gpu_timing: false,
            lod_timing: false,
            opening: Instant::now(),
        }),
        ..setup()
    };
    // A cost model that fits every dab: the queue's mechanics, whatever this machine's speed (the
    // numbers are the performance suite's, on the benchmark project).
    let mut owner = owner_of(
        project(&dir),
        |project| Sculpting::new(project).with_cost(DabCost::any_number()),
        setup,
    );
    run(&mut owner, 100_000);
    assert!(owner.done(), "the script ran through");
    let settings = owner.brush().settings();
    assert_eq!(
        settings.radius().get(),
        40.0,
        "the long stroke's size, set as it entered"
    );
    assert_eq!(
        settings.strength_mode(),
        StrengthMode::Radius,
        "still in radii after the script set the size"
    );
    assert!(owner.project().has_unsaved_edits(), "the strokes edited");
    let lines = owner.record_lines();
    let segments: Vec<&str> = lines
        .iter()
        .filter(|line| field(line, "result") == "\"segment\"")
        .map(|line| field(line, "segment"))
        .collect();
    assert!(segments.contains(&"\"stroke_near\""), "{segments:?}");
    assert!(segments.contains(&"\"stroke_far\""), "{segments:?}");
    let frames: Vec<&String> = lines
        .iter()
        .filter(|line| field(line, "result") == "\"frame\"")
        .collect();
    let brushed: Vec<&&String> = frames
        .iter()
        .filter(|line| field(line, "dabs_applied") != "0")
        .collect();
    assert!(!brushed.is_empty(), "frames applied dabs");
    for line in &brushed {
        let latency: f64 = field(line, "sample_to_present_ms")
            .parse()
            .expect("a latency where dabs applied");
        assert!((0.0..10_000.0).contains(&latency), "{line}");
        assert_ne!(field(line, "received_to_present_ms"), "null", "{line}");
        assert_ne!(field(line, "submitted_ms"), "null", "{line}");
        assert_ne!(field(line, "above_f"), "null", "{line}");
    }
    let quiet = frames
        .iter()
        .find(|line| field(line, "dabs_applied") == "0");
    let quiet = quiet.expect("frames without dabs");
    assert_eq!(field(quiet, "sample_to_present_ms"), "null", "{quiet}");
    let (most, drained) = queue_over(&frames, "\"stroke_long\"");
    assert_eq!(
        most, 0,
        "with time for every dab, the long stroke's queue never holds one past its frame"
    );
    assert_eq!(drained, Some(1), "the frame after pen-up carries none");
    let records = saved_strokes(owner, &dir, "scripted.bcrf");
    assert_eq!(records.len(), 4, "four strokes, a record each");
}

/// Over the record's `frames` in `segment`: the most dabs any carried, and how many frames it took
/// to drain.
///
/// The drain is counted from pen-up (the first frame after the last with its stroke open) to the
/// first carrying none, inclusively; `None` if it never drained there.
fn queue_over(frames: &[&String], segment: &str) -> (usize, Option<usize>) {
    let ours: Vec<&&String> = frames
        .iter()
        .filter(|line| field(line, "segment") == segment)
        .collect();
    let carried = |line: &str| -> usize { field(line, "dabs_carried").parse().expect("a count") };
    let most = ours.iter().map(|line| carried(line)).max().unwrap_or(0);
    let last_open = ours
        .iter()
        .rposition(|line| field(line, "stroke_open") == "true")
        .expect("the pen touched in the segment");
    let after = ours.get(last_open + 1..).unwrap_or_default();
    let drained = after.iter().position(|line| carried(line) == 0);
    (most, drained.map(|at| at + 1))
}

#[test]
fn a_step_within_its_time_is_finest_work_alone_and_one_past_it_shows_the_rest() {
    let (within_dir, past_dir) = (TestDir::new(), TestDir::new());
    let epoch = Instant::now();
    let mut within = owner_of(
        project(&within_dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::any_number())
                .with_epoch(epoch)
        },
        setup(),
    );
    // Nothing fits a frame: every dab is shown, and catches up between frames and after the
    // hand-off once the pen rests.
    let mut past = owner_of(
        project(&past_dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut within, 10_000);
    run(&mut past, 10_000);
    let samples = stroke(ACROSS, 12);
    queue(&mut within, samples.clone());
    queue(&mut past, samples);

    assert_eq!(run(&mut within, 1), 1);
    let finest = within.brush().last();
    assert!(finest.dabs_applied > 3, "{finest:?}");
    assert_eq!(finest.dabs_shown, 0, "within its time: nothing shown ahead");
    assert_eq!(finest.dabs_catching_up, 0);
    assert!(!within.project().store().has_previews());

    assert_eq!(run(&mut past, 1), 1);
    let shown = past.brush().last();
    assert_eq!(shown.dabs_applied, 0, "a frame waits on no finest work");
    assert_eq!(
        shown.dabs_shown, finest.dabs_applied,
        "every dab placed is shown in the frame of its sample"
    );
    assert_eq!(shown.dabs_carried, 0, "none left behind the pen");
    assert_eq!(shown.dabs_catching_up, shown.dabs_shown);
    assert!(shown.oldest.is_some(), "the frame's latency runs from them");
    assert!(past.project().store().has_previews(), "drawn from previews");

    run(&mut within, 10_000);
    let mut frames = 0;
    while past.brush().stroking() {
        run(&mut past, 1);
        frames += 1;
        assert!(frames < 10_000, "the catch-up ends");
    }
    assert!(
        !past.project().store().has_previews(),
        "no preview once every dab has landed"
    );
    assert_eq!(past.brush().error(), None);
    assert_eq!(
        positions(past.project()),
        positions(within.project()),
        "shown then caught up: the surface of finest work alone, bit for bit"
    );
    within.close().expect("closed");
    assert_eq!(saved_strokes(past, &past_dir, "p.bcrf").len(), 1);
}

#[test]
fn f_is_the_dab_one_frames_time_fits_by_the_cost_model() {
    let cost = DabCost::CALIBRATED;
    let level = Level::new(12).expect("a level");
    let keys = |n: u16| -> Vec<TileKey> {
        let keys = (0..n).map(|k| {
            TileKey::display(geom::FaceId::new(0), level, k % 64, k / 64).expect("on the face")
        });
        keys.collect()
    };
    let fits = |n: u16| cost.take([keys(n).as_slice()], LATENCY_AIM, false).0 == 1;
    let limit = u16::try_from(LIMIT_TILES).expect("a few tiles");
    assert!(fits(limit), "F itself fits the aim");
    assert!(!fits(limit + 1), "one tile more doesn't");
    assert_eq!(LIMIT_TILES, 65, "the derivation in LIMIT_TILES' notes");
}
