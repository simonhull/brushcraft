//! The frame loop on a small project, on the real clock: the order of its phases, render on
//! demand, and closing.
//!
//! Mostly with no GPU; with GPU residency on lavapipe, the floor uploaded, a lost frame
//! recovered, and an idle residency asking for no frames; and with the renderer too, the surface
//! drawn through the level choice and the raster, its cut read back into residency, and the same
//! frames drawn from the same inputs.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::bail;
use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use render::gpu::{AdapterRequest, Gpu};
use render::material::MatCapStyle;
use render::raster::{BACKGROUND, VisibilityId};
use render::surface::{Offscreen, Target};
use render::tree::Place;
use render::{DroppedReadbacks, Outcome, Renderer};
use store::{Epoch, ProjectConfig, StoreConfig};

use super::*;
use crate::camera::Pixel;
use crate::navigate::Button;
use crate::pen::PenFrame;
use crate::render_thread::draw_frame;
use crate::sculpting::Sculpting;
use crate::test_dir::TestDir;
use crate::uploads::GpuResidency;

/// The finest level the test projects keep a display layer under: small, so a rebuild takes
/// a few frames.
const FINEST: u8 = 3;

/// A 60 Hz display's interval.
const INTERVAL: Duration = Duration::from_micros(16_667);

/// A never-saved project over a six-face cube sphere, keeping a display layer under level
/// [`FINEST`], its recovery file in `dir`.
fn project(dir: &TestDir) -> Project {
    project_to(dir, FINEST)
}

/// [`project`], its display layer under level `finest`.
fn project_to(dir: &TestDir, finest: u8) -> Project {
    let mesh = cube_sphere(1).expect("a cube sphere");
    let finest = Level::new(finest).expect("a level");
    let config = ProjectConfig {
        store: StoreConfig::new(512 << 20, 1).with_display(finest),
        recovery_dir: dir.path().join("recovery"),
    };
    Project::new(mesh, config).expect("a new project")
}

/// The setup of a 320 × 200 offscreen run, scripted for `frames` if given.
fn setup(frames: Option<u64>) -> Setup {
    Setup {
        size: Size::new(320, 200),
        interval: INTERVAL,
        presentation: Presentation::Offscreen,
        adapter: "test".to_owned(),
        frames,
        script: None,
        record: None,
    }
}

/// The render thread's report of frame `number`, shown, with no scene drawn, and finished on the
/// GPU at once, its present submitted now.
fn presented(number: u64) -> Presented {
    Presented {
        number,
        outcome: Outcome::Presented,
        times: DrawTimes {
            acquire: Duration::ZERO,
            encode: Duration::ZERO,
            present: Duration::ZERO,
        },
        scene: false,
        shown: false,
        aa: aa::AaReport {
            restarted: false,
            filtered_instead: false,
        },
        gpu: Vec::new(),
        dropped: DroppedReadbacks { cut: 0, timing: 0 },
        gpu_finished: number,
        submitted: Instant::now(),
    }
}

/// Makes one frame now, as [`Owner::next`] said with `deadline`, shown at once; returns
/// what it handed the render thread.
fn frame<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, deadline: Instant) -> FrameRequest {
    let mut handed = None;
    owner
        .frame(Instant::now(), deadline, |request| {
            handed = Some(request);
            Ok(())
        })
        .expect("a frame");
    let request = handed.expect("the frame was handed off");
    owner.shown(presented(request.number));
    request
}

/// Makes frames on the real clock for as long as the owner wants them, up to `most`, each
/// shown at once, sleeping where the owner says to wait; returns how many it made.
fn run<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, most: u64) -> u64 {
    let mut made = 0;
    while made < most {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                frame(owner, deadline);
                made += 1;
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu => panic!("the frame was shown and finished"),
            Next::Idle { .. } => break,
        }
    }
    made
}

/// The first frame's deadline, which the owner wants at once.
fn first_deadline<B: Brush, U: Uploads>(owner: &mut Owner<B, U>) -> Instant {
    let Next::Frame { deadline } = owner.next(Instant::now()) else {
        panic!("the first frame is wanted");
    };
    deadline
}

/// A brush that, on its first step only, edits one vertex of one tile and closes the epoch,
/// and records the camera each step saw.
#[derive(Debug, Default)]
struct OneEdit {
    /// Whether it has made its edit.
    done: bool,
    /// The camera each step was given.
    cameras: Vec<Camera>,
}

impl Brush for OneEdit {
    fn step(
        &mut self,
        context: BrushContext<'_>,
        _pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        let BrushContext {
            project, camera, ..
        } = context;
        self.cameras.push(*camera);
        if std::mem::replace(&mut self.done, true) {
            return Ok(BrushReport::default());
        }
        let key = TileKey::displacement(FaceId::new(0), Level::new(FINEST)?, 0, 0)?;
        project.begin_stroke()?;
        project
            .store()
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.0, 0.0, 0.01)))??;
        project.store().close_epoch()?;
        project.end_unrecorded_stroke()?;
        Ok(BrushReport::default())
    }
}

/// Uploads that record every frame's drained epochs and when they finished, and can claim to
/// be converging.
#[derive(Debug, Default)]
struct Recording {
    /// The epochs each frame's apply was given, frame by frame.
    frames: Vec<Vec<Epoch>>,
    /// When each frame's apply returned, and the deadline it was given.
    times: Vec<(Instant, Instant)>,
    /// Frames still to report converging.
    converging_for: u32,
}

impl Uploads for Recording {
    fn apply(&mut self, update: FrameUpdate, _store: &mut TileStore) -> Result<UploadStatus> {
        let Drained::Epochs(epochs) = update.drained else {
            bail!("no resync expected");
        };
        self.frames.push(epochs);
        let converging = self.converging_for > 0;
        self.converging_for = self.converging_for.saturating_sub(1);
        self.times.push((Instant::now(), update.deadline));
        Ok(UploadStatus {
            uploading: converging,
            summary: Some("3 tiles".to_owned()),
            ..UploadStatus::default()
        })
    }
}

#[test]
fn a_frame_runs_input_brush_feed_uploads_then_hands_off_then_the_jobs() {
    let dir = TestDir::new();
    // A slow display, so the first frame's jobs surely record some of the display rebuild a new
    // project queues, and close epochs of it.
    let slow = Setup {
        interval: Duration::from_millis(100),
        ..setup(None)
    };
    let mut owner = Owner::new(
        project(&dir),
        OneEdit::default(),
        Recording::default(),
        slow,
        Instant::now(),
    );
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: Pixel { x: 160.0, y: 100.0 },
    });
    owner.input(Input::Move {
        at: Pixel { x: 200.0, y: 100.0 },
    });
    let deadline = first_deadline(&mut owner);
    let started = Instant::now();
    let mut handed = None;
    owner
        .frame(started, deadline, |request| {
            handed = Some((Instant::now(), request));
            Ok(())
        })
        .expect("a frame");
    let ended = Instant::now();
    let (handed_at, first) = handed.expect("the frame was handed off");
    owner.shown(presented(first.number));

    // The uploads, then the hand-off, then the jobs: the uploads had returned when the frame
    // was handed off, and the jobs, which ran until the frame returned, began after it.
    let &(uploaded_at, uploads_deadline) = owner.uploads().times.first().expect("uploads ran");
    assert!(
        uploaded_at <= handed_at,
        "the uploads come before the hand-off"
    );
    let jobs = owner.after_hand_off.jobs;
    assert!(owner.after_hand_off.report.slices >= 1, "the jobs ran");
    assert!(
        handed_at + jobs <= ended,
        "the jobs ({jobs:?}) run after the hand-off ({:?} before the frame returned)",
        ended - handed_at
    );
    assert!(
        uploads_deadline < deadline && uploads_deadline > started,
        "the uploads get a share of the frame's time, the jobs the rest"
    );

    // Input, then the brush: the brush saw the camera the drag had already turned.
    let seen = owner
        .brush
        .cameras
        .first()
        .copied()
        .expect("the brush stepped");
    assert_ne!(seen, Camera::default(), "the brush sees this frame's input");
    // The brush, then the feed and the uploads: its epoch reached the uploads that frame, and
    // only it: the jobs, which run after, hadn't closed theirs yet.
    let uploaded = owner.uploads().frames.first().expect("uploads ran");
    assert_eq!(uploaded.len(), 1, "the brush's epoch alone: {uploaded:?}");
    assert!(!uploaded.iter().all(|epoch| epoch.keys.is_empty()));
    // The jobs ran after and closed epochs of the rebuild, which the next frame drains.
    assert!(owner.has_stale(), "the jobs closed epochs after the drain");
    // The hand-off came before the jobs: its overlay reports the jobs of the frame before
    // (none), the next frame's those of this one.
    let first_text = first.overlay.join("\n");
    assert!(first_text.contains("jobs       0 slices"), "{first_text}");
    let Next::Frame { deadline: later } = owner.next(Instant::now() + Duration::from_millis(100))
    else {
        panic!("the jobs' epochs, undrained, want the next frame");
    };
    let second = frame(&mut owner, later);
    let second_text = second.overlay.join("\n");
    assert!(
        !second_text.contains("jobs       0 slices"),
        "{second_text}"
    );
    let next = owner
        .uploads()
        .frames
        .get(1)
        .expect("the second frame's uploads");
    assert!(
        !next.is_empty(),
        "the jobs' epochs, drained by the next frame"
    );
    assert!(
        second_text.contains("tiles      3 tiles"),
        "the uploads' summary: {second_text}"
    );
    owner.close().expect("closed");
    let kept: Vec<_> = std::fs::read_dir(dir.path().join("recovery"))
        .expect("the recovery directory")
        .collect();
    assert_eq!(kept.len(), 1, "unsaved edits keep their recovery file");
}

#[test]
fn a_frame_runs_a_slice_of_jobs_however_late_it_is() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    // A deadline already past: the jobs still get one slice.
    let past = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("a second after boot");
    frame(&mut owner, past);
    assert!(
        owner.after_hand_off.report.slices >= 1,
        "{:?}",
        owner.after_hand_off.report
    );
    owner.close().expect("closed");
}

#[test]
fn store_jobs_keep_frames_coming_until_they_are_done_then_none() {
    let dir = TestDir::new();
    // A 4 kHz display: three quarters of its interval is too short to rebuild the layer in
    // one frame, so it takes several.
    let fast = Setup {
        interval: Duration::from_micros(250),
        ..setup(None)
    };
    let mut owner = Owner::new(project(&dir), NoBrush, NoUploads, fast, Instant::now());
    assert!(
        owner.project().store().has_jobs(),
        "a new project builds its display layer"
    );
    let made = run(&mut owner, 100_000);
    assert!(
        made > 1,
        "the rebuild took more than the first frame: {made}"
    );
    assert!(!owner.project().store().has_jobs(), "the rebuild finished");
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "and nothing more is wanted"
    );
    owner.close().expect("closed");
    let left = std::fs::read_dir(dir.path().join("recovery"))
        .expect("the recovery directory")
        .count();
    assert_eq!(left, 0, "no edits: the recovery file goes");
}

#[test]
fn uploads_still_converging_want_the_next_frame() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        Recording {
            converging_for: 4,
            ..Recording::default()
        },
        setup(None),
        Instant::now(),
    );
    // Let the display rebuild finish first, so only the uploads want frames.
    run(&mut owner, 10_000);
    owner.uploads.converging_for = 4;
    owner.expose();
    assert_eq!(run(&mut owner, 100), 5, "one exposed, four more converging");
    owner.close().expect("closed");
}

#[test]
fn a_drag_wants_a_frame_and_moves_the_camera_in_it_but_hovering_wants_none() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    run(&mut owner, 10_000);
    owner.input(Input::Move {
        at: Pixel { x: 10.0, y: 10.0 },
    });
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "hovering"
    );
    let before = *owner.camera();
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: Pixel { x: 160.0, y: 100.0 },
    });
    owner.input(Input::Move {
        at: Pixel { x: 200.0, y: 100.0 },
    });
    assert_eq!(*owner.camera(), before, "input waits for the frame");
    assert_eq!(run(&mut owner, 100), 1, "one frame for the drag");
    assert_ne!(*owner.camera(), before, "the frame's input phase orbited");
    owner.close().expect("closed");
}

/// A brush that claims the strokes it is set to, and records what each frame handed it: the
/// phases of the samples it reads, and where the pen hovered.
#[derive(Debug, Default)]
struct PenLog {
    /// Whether it claims the strokes not started with a barrel held.
    claims: bool,
    /// Each step's samples' phases, and the hover it was handed.
    frames: Vec<(Vec<input::Phase>, Option<Pixel>)>,
}

impl Brush for PenLog {
    fn step(
        &mut self,
        _context: BrushContext<'_>,
        pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        let phases: Vec<input::Phase> = pen.for_brush().map(|sample| sample.phase).collect();
        self.frames
            .push((phases, pen.hover().map(|hover| hover.at)));
        Ok(BrushReport::default())
    }

    fn route(&mut self, _context: BrushContext<'_>, _down: &PenSample) -> Route {
        if self.claims {
            Route::Brush
        } else {
            Route::Navigate
        }
    }
}

/// A pen sample in `phase` at `(x, y)`, now.
fn pen_sample(phase: input::Phase, (x, y): (f64, f64)) -> PenSample {
    let now = Instant::now();
    PenSample {
        phase,
        tool: input::Tool::Pen,
        at: Pixel { x, y },
        pressure: input::Unit::ZERO,
        distance: None,
        tilt: None,
        roll: None,
        buttons: input::Buttons::default(),
        timestamp: now,
        received: now,
    }
}

/// A frame loop over a fresh project with a [`PenLog`] brush claiming strokes or not, settled.
fn pen_owner(dir: &TestDir, claims: bool) -> Owner<PenLog, NoUploads> {
    let brush = PenLog {
        claims,
        ..PenLog::default()
    };
    let mut owner = Owner::new(project(dir), brush, NoUploads, setup(None), Instant::now());
    run(&mut owner, 10_000);
    owner.brush.frames.clear();
    owner
}

#[test]
fn a_navigation_stroke_orbits_and_never_reaches_the_brush() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, false);
    owner.pen(pen_sample(Phase::Enter, (160.0, 100.0)));
    owner.pen(pen_sample(Phase::Hover, (161.0, 100.0)));
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "hovering with no picker wants no frame"
    );
    let before = *owner.camera();
    owner.pen(pen_sample(Phase::Down, (161.0, 100.0)));
    owner.pen(pen_sample(Phase::Drag, (200.0, 100.0)));
    assert_eq!(run(&mut owner, 100), 1, "one frame for the stroke");
    assert_ne!(*owner.camera(), before, "the tip's drag orbited");
    let phases: Vec<Phase> = owner
        .pen_frame()
        .samples()
        .iter()
        .map(|sample| sample.phase)
        .collect();
    assert_eq!(
        phases,
        [Phase::Hover, Phase::Down, Phase::Drag],
        "every sample in order, the enter coalesced into the hover after it"
    );
    assert_eq!(
        owner.brush.frames,
        [(vec![Phase::Hover], None)],
        "the brush reads the hover, not the navigation stroke; touching isn't hovering"
    );

    owner.pen(pen_sample(Phase::Up, (200.0, 100.0)));
    owner.pen(pen_sample(Phase::Hover, (205.0, 90.0)));
    owner.pen(pen_sample(Phase::Hover, (210.0, 90.0)));
    assert_eq!(run(&mut owner, 100), 1);
    let after = *owner.camera();
    let last = owner.brush.frames.last().cloned();
    assert_eq!(
        last,
        Some((vec![Phase::Hover], Some(Pixel { x: 210.0, y: 90.0 }))),
        "lifted and hovering: the latest hover"
    );
    owner.pen(pen_sample(Phase::Drag, (300.0, 10.0)));
    run(&mut owner, 100);
    assert_eq!(*owner.camera(), after, "the drag ended at the lift");
    owner.close().expect("closed");
}

#[test]
fn a_brush_strokes_samples_start_a_frame_at_once_and_with_fifo_wait_for_the_slot() {
    use input::Phase;
    let on_input = TestDir::new();
    let mut owner = pen_owner(&on_input, true);
    // The last frame started a moment ago: the refresh slot is most of an interval away.
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    assert!(
        matches!(owner.next(Instant::now()), Next::Frame { .. }),
        "the down starts a frame at once"
    );
    frame(&mut owner, Instant::now());
    owner.pen(pen_sample(Phase::Drag, (170.0, 100.0)));
    assert!(
        matches!(owner.next(Instant::now()), Next::Frame { .. }),
        "and each sample of the brush's stroke"
    );
    frame(&mut owner, Instant::now());
    owner.pen(pen_sample(Phase::Up, (170.0, 100.0)));
    frame(&mut owner, Instant::now());
    owner.close().expect("closed");

    // A navigation stroke's samples after its down are paced to the refresh, as ever.
    let navigating = TestDir::new();
    let mut orbit = pen_owner(&navigating, false);
    orbit.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    frame(&mut orbit, Instant::now());
    orbit.pen(pen_sample(Phase::Drag, (200.0, 100.0)));
    assert!(
        matches!(orbit.next(Instant::now()), Next::Wait { .. }),
        "an orbit by the pen keeps its pacing"
    );
    orbit.close().expect("closed");

    // Where mailbox isn't offered (FIFO), a stroke's frames wait for their slot.
    let paced = TestDir::new();
    let fifo = Setup {
        presentation: Presentation::Fifo,
        ..setup(None)
    };
    let brush = PenLog {
        claims: true,
        ..PenLog::default()
    };
    let mut fallback = Owner::new(project(&paced), brush, NoUploads, fifo, Instant::now());
    run(&mut fallback, 10_000);
    fallback.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    frame(&mut fallback, Instant::now());
    fallback.pen(pen_sample(Phase::Drag, (170.0, 100.0)));
    assert!(
        matches!(fallback.next(Instant::now()), Next::Wait { .. }),
        "paced to the refresh"
    );
    let size = Size::new(320, 200).expect("non-zero");
    let overlay = fallback.overlay(size, OwnerPhases::default(), None);
    assert!(
        overlay.join("\n").contains("no mailbox"),
        "and the overlay says why"
    );
    fallback.close().expect("closed");
}

/// A pen stream that delivers `sample` once, at its first pull at or after `from`.
#[derive(Debug)]
struct LateSample {
    /// When it may deliver.
    from: Instant,
    /// The sample, until delivered.
    sample: Option<PenSample>,
}

impl LateSample {
    /// A stream delivering `sample` at its first pull at or after `from`.
    const fn new(from: Instant, sample: PenSample) -> Self {
        Self {
            from,
            sample: Some(sample),
        }
    }
}

impl PenFeed for LateSample {
    fn pull(&mut self, out: &mut Vec<PenSample>) {
        if Instant::now() >= self.from {
            out.extend(self.sample.take());
        }
    }
}

/// A pen stream that delivers a drag of an open stroke at every pull: a stroke whose samples
/// always wait.
#[derive(Debug, Default)]
struct EveryPull {
    /// Pulls so far.
    pulls: u32,
}

impl PenFeed for EveryPull {
    fn pull(&mut self, out: &mut Vec<PenSample>) {
        self.pulls += 1;
        let x = 160.0 + f64::from(self.pulls % 40);
        out.push(pen_sample(input::Phase::Drag, (x, 100.0)));
    }
}

#[test]
fn a_moving_strokes_held_jobs_want_no_frame_and_run_once_it_holds_still() {
    use input::Phase;
    let dir = TestDir::new();
    let brush = PenLog {
        claims: true,
        ..PenLog::default()
    };
    let mut owner = Owner::new(
        project_to(&dir, 6),
        brush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    assert!(owner.project().store().has_jobs(), "a rebuild to run");
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    frame(&mut owner, Instant::now() + Duration::from_secs(1));
    assert_eq!(owner.after_hand_off.report.slices, 0, "held by the stroke");
    let now = Instant::now();
    let Next::Idle { upkeep } = owner.next(now) else {
        panic!("no frame for the held jobs while the stroke moves");
    };
    assert!(
        upkeep <= now + STROKE_QUIET,
        "but the loop wakes as the stroke falls quiet"
    );
    // The pen holds still (or its `Up` was lost): the jobs want their frame, and run.
    std::thread::sleep(STROKE_QUIET + Duration::from_millis(5));
    assert!(matches!(owner.next(Instant::now()), Next::Frame { .. }));
    frame(&mut owner, Instant::now() + Duration::from_millis(20));
    assert!(
        owner.after_hand_off.report.slices > 0,
        "a still stroke lets the jobs run"
    );
    owner.close().expect("closed");
}

#[test]
fn the_upkeep_runs_when_due_however_often_a_stroke_waits() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, true);
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    frame(&mut owner, Instant::now());
    let before = owner.upkept_at;
    let mut pen = EveryPull::default();
    let until = Instant::now() + UPKEEP_INTERVAL * 2;
    let mut number = owner.started;
    while Instant::now() < until {
        let now = Instant::now();
        owner
            .frame_fed(now, now + Duration::from_millis(1), &mut pen, |_| Ok(()))
            .expect("a frame");
        number += 1;
        owner.shown(presented(number));
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        owner.upkept_at > before,
        "a sample waited at every look, yet the due upkeep ran"
    );
    assert!(
        Instant::now() < owner.upkept_at + UPKEEP_INTERVAL * 2,
        "and keeps running"
    );
    owner.close().expect("closed");
}

#[test]
fn the_jobs_yield_to_a_strokes_sample_and_wait_while_the_stroke_is_open() {
    use input::Phase;
    let dir = TestDir::new();
    let brush = PenLog {
        claims: true,
        ..PenLog::default()
    };
    // A larger display layer to build: its jobs take many steps.
    let mut owner = Owner::new(
        project_to(&dir, 6),
        brush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    assert!(owner.project().store().has_jobs(), "a rebuild to run");
    // The sample arrives as the jobs run: a few milliseconds into the frame, past its brush
    // phase, upkeep and preparation (all quick here).
    let started = Instant::now();
    let mut pen = LateSample::new(
        started + Duration::from_millis(3),
        pen_sample(Phase::Down, (160.0, 100.0)),
    );
    owner
        .frame_fed(started, started + Duration::from_secs(30), &mut pen, |_| {
            Ok(())
        })
        .expect("a frame");
    owner.shown(presented(1));
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(1),
        "the jobs stopped at the sample, long before their deadline: {took:?}"
    );
    assert!(owner.project().store().has_jobs(), "left for later");
    let ran = owner.after_hand_off.report.slices;
    assert!(ran >= 1, "a step ran before the sample came");
    assert!(
        matches!(owner.next(Instant::now()), Next::Frame { .. }),
        "and the sample's frame starts at once"
    );
    // That frame opens the brush's stroke: no jobs at all while it moves.
    frame(&mut owner, Instant::now() + Duration::from_secs(1));
    assert_eq!(
        owner.after_hand_off.report.slices, 0,
        "none while the stroke moves"
    );
    owner.pen(pen_sample(Phase::Up, (160.0, 100.0)));
    std::thread::sleep(STROKE_QUIET);
    frame(&mut owner, Instant::now() + Duration::from_millis(50));
    assert!(
        owner.after_hand_off.report.slices > 0,
        "the pen lifted: they go on"
    );
    owner.close().expect("closed");
}

#[test]
fn a_brush_stroke_never_orbits_and_a_barrel_stroke_still_navigates() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, true);
    let before = *owner.camera();
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    owner.pen(pen_sample(Phase::Drag, (200.0, 100.0)));
    assert_eq!(run(&mut owner, 100), 1);
    owner.pen(pen_sample(Phase::Drag, (240.0, 100.0)));
    owner.pen(pen_sample(Phase::Up, (240.0, 100.0)));
    assert_eq!(run(&mut owner, 100), 1);
    assert_eq!(*owner.camera(), before, "the brush's stroke moved nothing");
    let read: Vec<Vec<Phase>> = owner
        .brush
        .frames
        .iter()
        .map(|(phases, _)| phases.clone())
        .collect();
    assert_eq!(
        read,
        [vec![Phase::Down, Phase::Drag], vec![Phase::Drag, Phase::Up]],
        "the brush has its stroke across both frames"
    );

    let barrel = |phase, at| PenSample {
        buttons: input::Buttons::SECONDARY,
        ..pen_sample(phase, at)
    };
    owner.pen(barrel(Phase::Down, (160.0, 100.0)));
    owner.pen(barrel(Phase::Drag, (200.0, 100.0)));
    run(&mut owner, 100);
    assert_ne!(
        owner.camera().target(),
        before.target(),
        "a barrel held at the down pans, though the brush claims strokes"
    );
    owner.close().expect("closed");
}

#[test]
fn the_mouse_moves_no_view_while_a_brush_stroke_is_open() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, true);
    let before = *owner.camera();
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    assert_eq!(run(&mut owner, 100), 1);
    // A mouse drag while the pen's stroke is open: the view holds still.
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: Pixel { x: 100.0, y: 100.0 },
    });
    owner.input(Input::Move {
        at: Pixel { x: 140.0, y: 100.0 },
    });
    owner.pen(pen_sample(Phase::Drag, (200.0, 100.0)));
    run(&mut owner, 100);
    assert_eq!(*owner.camera(), before, "frozen while the brush draws");
    // The stroke over, the same drag goes on from where the pointer is, with no jump.
    owner.pen(pen_sample(Phase::Up, (200.0, 100.0)));
    run(&mut owner, 100);
    assert_eq!(*owner.camera(), before, "no catching up");
    owner.input(Input::Move {
        at: Pixel { x: 180.0, y: 100.0 },
    });
    run(&mut owner, 100);
    assert_ne!(*owner.camera(), before, "then the drag orbits");
    owner.close().expect("closed");
}

#[test]
fn hovering_while_no_frame_comes_keeps_one_sample() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, false);
    let moves = (0..1000).map(f64::from);
    for x in moves {
        owner.pen(pen_sample(Phase::Hover, (x, 10.0)));
    }
    assert_eq!(owner.pen_queue.len(), 1, "coalesced");
    owner.expose();
    run(&mut owner, 100);
    assert_eq!(
        owner.pen_frame().hover().map(|hover| hover.at),
        Some(Pixel { x: 999.0, y: 10.0 }),
        "the latest hover"
    );
    owner.close().expect("closed");
}

#[test]
fn a_scripted_run_makes_its_frames_then_is_done() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(Some(3)),
        Instant::now(),
    );
    assert!(!owner.done());
    assert_eq!(run(&mut owner, 100), 3);
    assert!(owner.done());
    assert_eq!(owner.finished(), 3);
    owner.close().expect("closed");
}

#[test]
fn no_frames_while_there_is_no_frame_size() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(Some(3)),
        Instant::now(),
    );
    owner.resize(None);
    assert!(matches!(owner.next(Instant::now()), Next::Idle { .. }));
    owner.resize(Size::new(64, 64));
    assert!(
        matches!(owner.next(Instant::now()), Next::Frame { .. }),
        "a size again: the window is drawn"
    );
    owner.close().expect("closed");
}

#[test]
fn epochs_closed_between_frames_want_a_frame_to_take_them() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    run(&mut owner, 10_000);
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "settled"
    );
    // Work between frames closes an epoch, as a poll installing prefetched tiles does.
    let context = BrushContext {
        project: owner.project(),
        index: None,
        camera: &Camera::default(),
        size: Size::new(320, 200).expect("a size"),
    };
    OneEdit::default()
        .step(context, &PenFrame::default(), &BrushBudget::unlimited())
        .expect("an edit");
    owner.upkeep(Instant::now()).expect("upkeep");
    assert!(
        matches!(
            owner.next(Instant::now()),
            Next::Frame { .. } | Next::Wait { .. }
        ),
        "the undrained epoch wants a frame"
    );
    // That frame drains it (and the refresh the edit queues takes a few more).
    assert!(run(&mut owner, 10_000) >= 1);
    assert!(!owner.has_stale(), "drained");
    owner.close().expect("closed");
}

#[test]
fn a_frame_due_with_the_gpu_behind_waits_running_jobs_and_upkeep_then_starts() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(Some(100)),
        Instant::now(),
    );
    // Frames handed off and shown, the GPU finishing none: each runs a single slice of jobs.
    let bound = u64::try_from(render::FRAMES_IN_FLIGHT).expect("small");
    for number in 1..=bound {
        let now = loop {
            let now = Instant::now();
            match owner.next(now) {
                Next::Frame { .. } => break now,
                Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
                other @ (Next::Gpu | Next::Render | Next::Idle { .. }) => {
                    panic!("frame {number} is due: {other:?}")
                }
            }
        };
        owner.frame(now, now, |_| Ok(())).expect("a frame");
        owner.shown(Presented {
            gpu_finished: 0,
            ..presented(number)
        });
    }
    assert!(
        owner.project().store().has_jobs(),
        "the display layer still building"
    );
    let due = loop {
        let now = Instant::now();
        match owner.next(now) {
            Next::Gpu => break now,
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            other @ (Next::Frame { .. } | Next::Render | Next::Idle { .. }) => {
                panic!("the next frame waits for the GPU: {other:?}")
            }
        }
    };
    let busy = owner
        .wait_for_gpu(due, &mut NoPen, || Ok(0))
        .expect("waiting");
    assert_eq!(busy, GpuWait::Worked, "the store's jobs ran meanwhile");
    assert!(
        owner.scheduler.wanted().contains(Reason::Converging),
        "and want the frame for what they leave"
    );
    let later = owner.scheduler.upkeep_due() + Duration::from_millis(1);
    owner
        .wait_for_gpu(later, &mut NoPen, || Ok(0))
        .expect("waiting");
    assert_eq!(
        owner.scheduler.upkeep_due(),
        later + UPKEEP_INTERVAL,
        "the upkeep ran, due meanwhile"
    );
    let ready = owner
        .wait_for_gpu(later, &mut NoPen, || Ok(bound))
        .expect("waiting");
    assert_eq!(ready, GpuWait::Ready, "the GPU finished");
    let Next::Frame { deadline } = owner.next(later) else {
        panic!("the frame starts");
    };
    owner.frame(later, deadline, |_| Ok(())).expect("a frame");
    let log = owner.phase_log().expect("a scripted run's log");
    let waited = log.owner.last().expect("logged").gpu_wait;
    assert_eq!(
        waited,
        later - due,
        "from when it was due to when it started"
    );
    owner.close().expect("closed");
}

#[test]
fn a_wait_for_the_gpu_that_leads_to_no_frame_is_forgotten() {
    let dir = TestDir::new();
    let start = Instant::now();
    let mut owner = Owner::new(project(&dir), NoBrush, NoUploads, setup(Some(2)), start);
    let wait = owner
        .wait_for_gpu(start, &mut NoPen, || Ok(0))
        .expect("waiting");
    assert_eq!(wait, GpuWait::Ready, "nothing in flight");
    let later = start + Duration::from_secs(5);
    assert!(matches!(owner.next(later), Next::Frame { .. }));
    assert_eq!(
        owner.gpu_wait_from,
        Some(start),
        "a frame is due: the wait leads to it"
    );
    // Minimised: no frame, so the wait is over, and the frame once shown again didn't wait.
    owner.resize(None);
    assert!(matches!(owner.next(later), Next::Idle { .. }));
    assert_eq!(owner.gpu_wait_from, None, "no frame: no wait to count");
    owner.close().expect("closed");
}

#[test]
fn a_minimised_window_still_runs_its_upkeep() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        NoBrush,
        NoUploads,
        setup(None),
        Instant::now(),
    );
    run(&mut owner, 10_000);
    // An edit the recovery writer has yet to flush, then the window is minimised.
    let context = BrushContext {
        project: owner.project(),
        index: None,
        camera: &Camera::default(),
        size: Size::new(320, 200).expect("a size"),
    };
    OneEdit::default()
        .step(context, &PenFrame::default(), &BrushBudget::unlimited())
        .expect("an edit");
    owner.resize(None);
    let later = Instant::now() + UPKEEP_INTERVAL;
    let Next::Idle { upkeep } = owner.next(later) else {
        panic!("no frames while minimised");
    };
    assert!(upkeep <= later, "upkeep falls due however long it waits");
    owner
        .upkeep(later)
        .expect("upkeep: the recovery writer ticked");
    let Next::Idle { upkeep: after } = owner.next(later) else {
        panic!("still no frames");
    };
    assert_eq!(after, later + UPKEEP_INTERVAL, "and is due again later");
    owner.close().expect("closed");
}

/// A software GPU: lavapipe, in the gate.
fn software_gpu() -> Arc<Gpu> {
    Arc::new(Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe"))
}

/// The frame loop over a test project in `dir` whose display layer is built already, its
/// tiles kept on `gpu`, its clock started now.
fn resident_owner(dir: &TestDir, gpu: &Arc<Gpu>) -> Owner<NoBrush, GpuResidency> {
    resident_owner_at(dir, gpu, Instant::now())
}

/// [`resident_owner`], its clock started at `clock`.
fn resident_owner_at(
    dir: &TestDir,
    gpu: &Arc<Gpu>,
    clock: Instant,
) -> Owner<NoBrush, GpuResidency> {
    let mut project = project(dir);
    while project.store().has_jobs() {
        project
            .store()
            .run_jobs(Instant::now() + Duration::from_secs(1))
            .expect("the display layer builds");
    }
    let residency = GpuResidency::new(Arc::clone(gpu), project.store()).expect("residency");
    Owner::new(project, NoBrush, residency, setup(None), clock)
}

/// Submits `request`'s uploads alone, as the render thread does ahead of its passes, and waits
/// for the GPU; returns the bytes they staged.
fn submit(gpu: &Gpu, request: FrameRequest) -> u64 {
    let Some(uploads) = request.uploads else {
        return 0;
    };
    let bytes = uploads.bytes();
    gpu.scoped("a test frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("the uploads submit");
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finishes");
    bytes
}

/// Makes one frame now with `deadline`, hands it to `deliver` (which says what became of it),
/// and reports it shown; returns the frame's number.
fn frame_with<U: Uploads>(
    owner: &mut Owner<NoBrush, U>,
    deadline: Instant,
    deliver: impl FnOnce(FrameRequest) -> Outcome,
) -> u64 {
    let mut handed = None;
    owner
        .frame(Instant::now(), deadline, |request| {
            handed = Some(request);
            Ok(())
        })
        .expect("a frame");
    let request = handed.expect("the frame was handed off");
    let number = request.number;
    let outcome = deliver(request);
    owner.shown(Presented {
        outcome,
        ..presented(number)
    });
    number
}

/// Makes frames for as long as the owner wants them, up to `most`, each submitted to `gpu`
/// and shown at once; returns how many it made.
fn run_on_gpu(gpu: &Gpu, owner: &mut Owner<NoBrush, GpuResidency>, most: u64) -> u64 {
    let mut made = 0;
    while made < most {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                frame_with(owner, deadline, |request| {
                    submit(gpu, request);
                    Outcome::Presented
                });
                made += 1;
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu => panic!("the frame was shown and finished"),
            Next::Idle { .. } => break,
        }
    }
    made
}

#[test]
fn the_first_frame_uploads_the_floor_and_then_nothing_spins() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let floor = owner.uploads().floor_tiles();
    assert!(floor > 0, "a floor to upload");
    let deadline = first_deadline(&mut owner);
    let mut first = None;
    frame_with(&mut owner, deadline, |request| {
        first = Some((request.overlay.join("\n"), submit(&gpu, request)));
        Outcome::Presented
    });
    let (text, bytes) = first.expect("the first frame");
    assert!(bytes > 0, "its uploads travelled with the frame");
    assert!(
        text.contains(&format!("floor {floor}/{floor}")),
        "the overlay's tiles line says the floor is resident: {text}"
    );
    let stats = owner.uploads().stats();
    assert_eq!(stats.floor, floor);
    assert_eq!(stats.lost_frames, 0);
    // The floor is up and nothing else is asked for: once the floor frame's readback is read
    // (it agrees: nothing requested) and the image has accumulated its samples (twice SAMPLES
    // after the frame that uploaded), no more frames are wanted.
    let settling = run_on_gpu(&gpu, &mut owner, 200);
    let accumulating = u64::from(2 * aa::SAMPLES);
    assert!(
        (accumulating..=accumulating + 3).contains(&settling),
        "a frame or so to read the readback, the samples, then none: {settling}"
    );
    assert!(matches!(owner.next(Instant::now()), Next::Idle { .. }));
    owner.close().expect("closed");
}

#[test]
fn a_frame_whose_uploads_are_dropped_is_recovered_by_the_next() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let floor = owner.uploads().floor_tiles();
    // The first frame's uploads are dropped unsubmitted (as by a render thread that gave up on
    // its frame), and nothing else asks for another: no retry, no input.
    let mut dropped = 0;
    let deadline = first_deadline(&mut owner);
    frame_with(&mut owner, deadline, |request| {
        dropped = request.uploads.as_ref().map_or(0, FrameUploads::bytes);
        drop(request);
        Outcome::Presented
    });
    assert!(dropped > 0, "the floor was staged, and lost");
    // Its readback never comes, so residency still wants a frame, which hears of the loss.
    let Next::Frame { deadline: again_by } = owner.next(Instant::now() + INTERVAL) else {
        panic!("a frame whose uploads were lost is followed by another");
    };
    let mut again = 0;
    frame_with(&mut owner, again_by, |request| {
        again = submit(&gpu, request);
        Outcome::Presented
    });
    assert!(
        again >= dropped,
        "the next frame stages the floor again, whole: {again} of {dropped} bytes"
    );
    run_on_gpu(&gpu, &mut owner, 100);
    let stats = owner.uploads().stats();
    assert_eq!(stats.lost_frames, 1, "the dropped frame was heard of");
    assert_eq!(stats.floor, floor, "and the floor is resident again");
    assert!(matches!(owner.next(Instant::now()), Next::Idle { .. }));
    owner.close().expect("closed");
}

/// The request buffer's bytes as a level choice that wants `key` alone leaves them: the count
/// 1, nothing dropped, then the one entry ([`render::residency::REQUESTS_WGSL`]).
fn one_request(key: &TileKey) -> Vec<u8> {
    let place = Place::of(key);
    let words = [1_u32, 0, 0, 0, key.face().index(), place.packed()];
    let mut bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    bytes.extend(100.0_f32.to_le_bytes());
    bytes.extend(0_u32.to_le_bytes());
    bytes
}

#[test]
fn the_last_moves_requests_are_read_back_with_no_further_event() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    run_on_gpu(&gpu, &mut owner, 100);
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "settled"
    );
    // The camera moves once, and that frame's level choice asks for a finest tile; no frame
    // after writes any request, and no event follows.
    let wanted = TileKey::displacement(FaceId::new(2), Level::new(FINEST).expect("a level"), 0, 0)
        .expect("a key");
    owner.input(Input::Press {
        button: Button::MouseLeft,
        at: Pixel { x: 160.0, y: 100.0 },
    });
    owner.input(Input::Move {
        at: Pixel { x: 170.0, y: 100.0 },
    });
    owner.input(Input::Release {
        button: Button::MouseLeft,
    });
    let Next::Frame { deadline } = owner.next(Instant::now() + INTERVAL) else {
        panic!("the move wants a frame");
    };
    let buffer = owner.uploads().residency().request_buffer().clone();
    frame_with(&mut owner, deadline, |request| {
        // As the level choice's pass would, before the frame's readback copies the buffer.
        gpu.queue().write_buffer(&buffer, 0, &one_request(&wanted));
        submit(&gpu, request);
        Outcome::Presented
    });
    let moved = owner.uploads().residency().frame();
    // That frame's readback is still to be read: frames keep coming until it is (and until the
    // next one, which asks for nothing, agrees).
    let made = run_on_gpu(&gpu, &mut owner, 100);
    assert!(made >= 1, "frames came with no event: {made}");
    assert!(
        owner.uploads().residency().requests_heard() >= Some(moved),
        "the moving frame's requests were read back"
    );
    assert!(
        matches!(owner.next(Instant::now()), Next::Idle { .. }),
        "and once the readbacks agree, nothing more is wanted"
    );
    owner.close().expect("closed");
}

/// Where the GPU tests draw frames as the render thread does: a renderer and an offscreen
/// target.
struct Screen {
    /// The renderer: the scene's passes arrive with the first frame.
    renderer: Renderer,
    /// The target.
    target: Target,
}

impl Screen {
    /// A renderer and an offscreen target of [`setup`]'s size on `gpu`.
    fn new(gpu: &Gpu) -> Self {
        let size = setup(None).size.expect("a size");
        let target = Target::Offscreen(Offscreen::new(gpu, size).expect("a target"));
        let renderer = Renderer::new(gpu, target.format()).expect("a renderer");
        Self { renderer, target }
    }

    /// Draws `request` as the render thread would, and waits for the GPU, so its readbacks are
    /// in by the next frame and it is reported finished.
    fn draw(&mut self, gpu: &Gpu, request: FrameRequest) -> Presented {
        let presented = draw_frame(&mut self.renderer, gpu, &mut self.target, request, || {})
            .expect("the frame is drawn");
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
        Presented {
            gpu_finished: presented.number,
            ..presented
        }
    }

    /// The frame's pixels, as the last frame left them.
    fn pixels(&self, gpu: &Gpu) -> Vec<[u8; 4]> {
        let Target::Offscreen(offscreen) = &self.target else {
            unreachable!("made offscreen");
        };
        offscreen.read_pixels(gpu).expect("read back")
    }

    /// The visibility buffer, as the last frame left it.
    fn ids(&self, gpu: &Gpu) -> Vec<u32> {
        self.renderer
            .passes()
            .expect("the first frame brought the passes")
            .raster()
            .read_visibility(gpu)
            .expect("read back")
    }
}

/// Makes a frame started at `now`, its jobs and uploads given until `deadline`, draws it on
/// `screen` and reports it shown; returns whether its scene was drawn.
fn draw_one<B: Brush>(
    gpu: &Gpu,
    owner: &mut Owner<B, GpuResidency>,
    screen: &mut Screen,
    now: Instant,
    deadline: Instant,
) -> bool {
    let mut handed = None;
    owner
        .frame(now, deadline, |request| {
            handed = Some(request);
            Ok(())
        })
        .expect("a frame");
    let presented = screen.draw(gpu, handed.expect("the frame was handed off"));
    let scene = presented.scene;
    owner.shown(presented);
    scene
}

#[test]
fn the_app_draws_the_surface_through_residency_and_hears_its_cuts() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut screen = Screen::new(&gpu);
    let floor = owner.uploads().floor_tiles();
    // On the real clock, for as long as frames are wanted.
    let (mut made, mut most_requested) = (0, 0);
    loop {
        assert!(made < 300, "frames never stopped");
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                assert!(draw_one(&gpu, &mut owner, &mut screen, now, deadline));
                made += 1;
                let requested = owner.uploads().scene().map_or(0, |scene| scene.requested);
                most_requested = most_requested.max(requested);
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu => panic!("every frame was shown and finished"),
            // Nothing converging: frames stopped by themselves.
            Next::Idle { .. } => break,
        }
    }
    // The cut came back, frame by frame, into residency.
    let residency = owner.uploads().residency();
    assert!(owner.uploads().cuts_heard() > 1, "cuts were read back");
    assert!(!residency.cut().is_empty(), "residency holds the cut");
    // The level choice asked for finest tiles, which arrived and are drawn.
    assert!(most_requested > 0, "the level choice requested tiles");
    let scene = owner.uploads().scene().expect("a cut").clone();
    assert_eq!(scene.requested, 0, "every request served: {scene:?}");
    let stats = residency.stats();
    assert!(stats.resident > floor, "requested tiles arrived: {stats:?}");
    assert_eq!(stats.wanted, 0, "and none is wanted still");
    // The surface is in the visibility buffer, and every id names a resident tile.
    let finest = Level::new(FINEST).expect("a level");
    let ids: Vec<VisibilityId> = screen
        .ids(&gpu)
        .into_iter()
        .filter_map(VisibilityId::decode)
        .collect();
    assert!(
        ids.len() > 1000,
        "the sphere covers the frame: {}",
        ids.len()
    );
    let tree = residency.tree();
    let keys: BTreeSet<TileKey> = ids
        .iter()
        .map(|id| {
            let node = tree.gpu_node(id.node);
            Place::from_packed(node.face, node.place)
                .and_then(|place| place.key(finest))
                .expect("a live node's tile")
        })
        .collect();
    for key in &keys {
        assert!(residency.slot(key).is_some(), "{key:?} is resident");
    }
    assert!(
        keys.iter().any(|key| key.level() == finest),
        "finest tiles are drawn: {keys:?}"
    );
    owner.close().expect("closed");
}

/// Each frame's visibility buffer, of `frames` frames of a fresh test project on `gpu`.
///
/// The camera is still; each frame starts an interval after the last on a clock begun at
/// `clock`, and has all the time it wants.
fn scripted_run(gpu: &Arc<Gpu>, clock: Instant, frames: u32) -> Vec<Vec<u32>> {
    let dir = TestDir::new();
    let mut owner = resident_owner_at(&dir, gpu, clock);
    let mut screen = Screen::new(gpu);
    let buffers = (0..frames).map(|k| {
        let now = clock + INTERVAL * k;
        let drawn = draw_one(
            gpu,
            &mut owner,
            &mut screen,
            now,
            now + Duration::from_secs(600),
        );
        assert!(drawn, "frame {k}'s scene was drawn");
        screen.ids(gpu)
    });
    let buffers: Vec<Vec<u32>> = buffers.collect();
    owner.close().expect("closed");
    buffers
}

#[test]
fn the_same_inputs_draw_the_same_frames() {
    let gpu = software_gpu();
    // Long enough for the requested tiles to arrive and morph in, the camera still.
    let frames = 14;
    let first = scripted_run(&gpu, Instant::now(), frames);
    let second = scripted_run(&gpu, Instant::now(), frames);
    let drawn = |buffer: &[u32]| buffer.iter().filter(|&&id| id != BACKGROUND).count();
    assert!(
        first.iter().all(|buffer| drawn(buffer) > 1000),
        "every frame draws the surface"
    );
    assert_ne!(
        first.first(),
        first.last(),
        "the frames changed as tiles arrived"
    );
    for (k, (a, b)) in first.iter().zip(&second).enumerate() {
        assert!(a == b, "frame {k} differs between the runs");
    }
}

/// `project`'s display layer, built.
fn build_display(project: &mut Project) {
    while project.store().has_jobs() {
        project
            .store()
            .run_jobs(Instant::now() + Duration::from_secs(1))
            .expect("the display layer builds");
    }
}

/// Makes a frame at `now` with all the time it wants, hands it off and returns it unshown.
fn hand_off(owner: &mut Owner<NoBrush, GpuResidency>, now: Instant) -> FrameRequest {
    let mut handed = None;
    owner
        .frame(now, now + Duration::from_secs(600), |request| {
            handed = Some(request);
            Ok(())
        })
        .expect("a frame");
    handed.expect("the frame was handed off")
}

/// The bounds a frame's scene asks to recompute: how many tiles.
fn bounds_of(request: &FrameRequest) -> usize {
    request
        .scene
        .as_ref()
        .expect("a scene")
        .resident
        .bounds
        .len()
}

/// Makes frames on `screen` until they stop by themselves: everything resident, every morph
/// over.
fn settle<B: Brush>(gpu: &Gpu, owner: &mut Owner<B, GpuResidency>, screen: &mut Screen) {
    let mut made = 0;
    loop {
        assert!(made < 300, "frames never stopped");
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                draw_one(gpu, owner, screen, now, deadline);
                made += 1;
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Render | Next::Gpu => panic!("every frame was shown and finished"),
            Next::Idle { .. } => break,
        }
    }
}

/// Something else a frame is wanted for: its name, and how the scheduler hears of it.
type Also = (&'static str, fn(&mut Scheduler));

#[test]
fn a_change_of_look_is_drawn_whole_while_anything_else_wants_the_frame() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut screen = Screen::new(&gpu);
    settle(&gpu, &mut owner, &mut screen);
    // Beside uploads in flight, requests settling, or a morph under way, the frame draws its
    // scene whole, in the new look.
    let others: [Also; 3] = [
        ("uploads", |s| s.want(Reason::Uploads)),
        ("requests", |s| s.want(Reason::Requests)),
        ("a morph", |s| s.cut_changed(Instant::now())),
    ];
    for (what, also) in others {
        owner.change_look(LookChange::MatCap);
        also(&mut owner.scheduler);
        let request = hand_off(&mut owner, Instant::now());
        assert!(!request.reshade, "with {what}: drawn whole");
        assert!(request.scene.is_some(), "with {what}: its scene");
        let presented = screen.draw(&gpu, request);
        assert!(presented.scene, "with {what}: drawn");
        owner.shown(presented);
        settle(&gpu, &mut owner, &mut screen);
    }
    owner.close().expect("closed");
}

#[test]
fn a_converged_image_drawn_again_unchanged_wants_no_more_frames() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut screen = Screen::new(&gpu);
    settle(&gpu, &mut owner, &mut screen);
    // The window exposed, nothing changed: one frame, which carries the image on.
    owner.expose();
    let mut made = 0;
    while let Next::Frame { deadline } = owner.next(Instant::now() + INTERVAL) {
        draw_one(&gpu, &mut owner, &mut screen, Instant::now(), deadline);
        made += 1;
        assert!(made < 10, "frames never stopped");
    }
    assert_eq!(made, 1, "no samples asked for: the image goes on");
    owner.close().expect("closed");
}

#[test]
fn only_uploads_with_nothing_new_leave_a_frame_quiet_enough_to_shade_again() {
    assert!(quiet(&UploadStatus::default()), "nothing at all");
    let busy = [
        UploadStatus {
            uploading: true,
            ..UploadStatus::default()
        },
        UploadStatus {
            requesting: true,
            ..UploadStatus::default()
        },
        UploadStatus {
            cut_changed: Some(Duration::from_millis(5)),
            ..UploadStatus::default()
        },
    ];
    for status in &busy {
        assert!(!quiet(status), "{status:?}");
    }
}

#[test]
fn a_change_of_look_alone_shades_the_last_scene_again_and_draws_nothing_new() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut screen = Screen::new(&gpu);
    settle(&gpu, &mut owner, &mut screen);
    let clay = screen.pixels(&gpu);
    let ids = screen.ids(&gpu);
    owner.change_look(LookChange::MatCap);
    assert_eq!(owner.look().material, MatCapStyle::Glossy);
    assert!(
        matches!(owner.next(Instant::now() + INTERVAL), Next::Frame { .. }),
        "the change wants a frame"
    );
    let request = hand_off(&mut owner, Instant::now());
    assert!(request.reshade, "the shading alone");
    assert!(request.scene.is_none(), "no level choice or raster");
    assert_eq!(request.look.material, MatCapStyle::Glossy);
    let presented = screen.draw(&gpu, request);
    assert!(
        !presented.scene && presented.shown,
        "shaded again: {presented:?}"
    );
    owner.shown(presented);
    assert_eq!(screen.ids(&gpu), ids, "nothing drawn anew");
    let glossy = screen.pixels(&gpu);
    let changed = clay.iter().zip(&glossy).filter(|(a, b)| a != b).count();
    assert!(
        changed > clay.len() / 20,
        "the new MatCap shows: {changed} pixels"
    );
    // The shading again restarted the image's samples: frames come until they are all taken.
    assert!(
        matches!(owner.next(Instant::now() + INTERVAL), Next::Frame { .. }),
        "accumulating"
    );
    settle(&gpu, &mut owner, &mut screen);
    // The debug view, the same way.
    owner.change_look(LookChange::Clusters);
    let debug = hand_off(&mut owner, Instant::now());
    assert!(debug.reshade, "the debug view shows the same scene");
    let debug_shown = screen.draw(&gpu, debug);
    assert!(debug_shown.shown, "shown");
    owner.shown(debug_shown);
    assert_ne!(screen.pixels(&gpu), glossy, "as clusters");
    // A change of look with the view changed too draws the scene whole.
    owner.change_look(LookChange::Clusters);
    owner.resize(Size::new(300, 200));
    let whole = hand_off(&mut owner, Instant::now());
    assert!(!whole.reshade, "drawn whole");
    assert!(whole.scene.is_some(), "with its scene");
    let whole_shown = screen.draw(&gpu, whole);
    assert!(
        whole_shown.scene && whole_shown.shown,
        "drawn: {whole_shown:?}"
    );
    owner.shown(whole_shown);
    owner.close().expect("closed");
}

#[test]
fn each_key_changes_the_look_its_own_way() {
    let look = Look::default();
    let wired = LookChange::Wireframe.apply(look);
    assert!(wired.shading.wireframe && !LookChange::Wireframe.apply(wired).shading.wireframe);
    let styles: Vec<MatCapStyle> =
        std::iter::successors(Some(look), |l| Some(LookChange::MatCap.apply(*l)))
            .take(4)
            .map(|l| l.material)
            .collect();
    assert_eq!(
        styles,
        [
            MatCapStyle::Clay,
            MatCapStyle::Glossy,
            MatCapStyle::Pale,
            MatCapStyle::Clay
        ],
        "round the styles"
    );
    let debug = LookChange::Clusters.apply(look);
    assert_eq!(debug.view, SceneView::Clusters);
    assert_eq!(LookChange::Clusters.apply(debug), look, "and back");
}

#[test]
fn a_frame_whose_scene_wasnt_drawn_leaves_its_bounds_to_the_next() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut screen = Screen::new(&gpu);
    let floor = owner.uploads().floor_tiles();
    let first = hand_off(&mut owner, Instant::now());
    assert_eq!(bounds_of(&first), floor, "the floor's bounds, all of them");
    // Drawn, but reported with no scene (as for a frame with no image).
    let presented = screen.draw(&gpu, first);
    owner.shown(Presented {
        scene: false,
        ..presented
    });
    let again = hand_off(&mut owner, Instant::now());
    assert!(
        bounds_of(&again) >= floor,
        "the floor's bounds asked for again: {}",
        bounds_of(&again)
    );
    let drawn = screen.draw(&gpu, again);
    assert!(drawn.scene, "drawn this time");
    owner.shown(drawn);
    let after = hand_off(&mut owner, Instant::now());
    assert!(
        bounds_of(&after) < floor,
        "committed: only tiles new since: {}",
        bounds_of(&after)
    );
    owner.shown(screen.draw(&gpu, after));
    owner.close().expect("closed");
}

#[test]
fn the_passes_reach_the_renderer_with_the_first_frame_even_with_no_scene() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut owner = resident_owner(&dir, &gpu);
    let mut first = hand_off(&mut owner, Instant::now());
    assert!(first.passes.is_some(), "the first frame brings the passes");
    // As if the camera had no view to draw: no scene, the passes still delivered.
    first.scene = None;
    let mut screen = Screen::new(&gpu);
    let presented = screen.draw(&gpu, first);
    assert!(!presented.scene, "no scene drawn");
    assert!(
        screen.renderer.passes().is_some(),
        "but the passes are kept"
    );
    owner.shown(presented);
    let second = hand_off(&mut owner, Instant::now());
    assert!(second.passes.is_none(), "sent once");
    let next = screen.draw(&gpu, second);
    assert!(next.scene, "the next frame draws its scene");
    owner.shown(next);
    owner.close().expect("closed");
}

/// One frame of `residency` on its own, the default camera's view at `stamp`'s time, moving or
/// not, drawn on `screen` and shown; returns the cut change the uploads told.
fn residency_frame(
    gpu: &Gpu,
    residency: &mut GpuResidency,
    project: &mut Project,
    screen: &mut Screen,
    stamp: (Duration, bool),
) -> Told {
    let (time, moving) = stamp;
    let size = setup(None).size.expect("a size");
    let update = FrameUpdate {
        drained: Drained::Epochs(Vec::new()),
        deadline: Instant::now() + Duration::from_secs(600),
        moved: moving,
        time,
    };
    let mut status = residency
        .apply(update, project.store())
        .expect("residency's frame");
    let scene = status.scene.take().map(|resident| SceneRequest {
        resident,
        view: Camera::default().view(size).expect("a view"),
        time,
        moving,
    });
    let request = FrameRequest {
        number: 1,
        size,
        overlay_scale: 1,
        overlay: Vec::new(),
        uploads: status.frame.take(),
        scene,
        reshade: false,
        look: Look::default(),
        passes: status.passes.take(),
    };
    let presented = screen.draw(gpu, request);
    residency.shown(presented.scene);
    Told(status.cut_changed)
}

/// The cut change a frame's uploads told: the time of the frame whose cut changed, if one did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Told(Option<Duration>);

#[test]
fn a_cut_change_is_told_with_its_own_frames_time_and_only_for_a_still_camera() {
    let gpu = software_gpu();
    for moving in [false, true] {
        let dir = TestDir::new();
        let mut project = project(&dir);
        build_display(&mut project);
        let mut residency =
            GpuResidency::new(Arc::clone(&gpu), project.store()).expect("residency");
        let mut screen = Screen::new(&gpu);
        let first = Duration::from_secs(1);
        let unheard = residency_frame(
            &gpu,
            &mut residency,
            &mut project,
            &mut screen,
            (first, moving),
        );
        assert_eq!(unheard, Told(None), "no cut heard yet");
        // The first frame's cut (new: there was none) is heard in the second, a second later.
        let stamp = (Duration::from_secs(2), false);
        let changed = residency_frame(&gpu, &mut residency, &mut project, &mut screen, stamp);
        if moving {
            assert_eq!(
                changed,
                Told(None),
                "a moving frame's cut change morphs by distance only"
            );
        } else {
            assert_eq!(
                changed,
                Told(Some(first)),
                "told with the cut's own frame's time"
            );
        }
        project.close(false).expect("closed");
    }
}

#[test]
fn eviction_under_a_tiny_budget_keeps_the_cut_read_back_and_its_ancestors() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    // Display tiles to level 8: the finest class (side 65) has 126 tiles; the budget holds the
    // floor's classes and twelve of them, in buffers of four.
    let mut project = project_to(&dir, 8);
    build_display(&mut project);
    let big = render::residency::SizeClass::new(6)
        .expect("a class")
        .slot_bytes();
    let floor_bytes: u64 = (0..6)
        .filter_map(render::residency::SizeClass::new)
        .map(|class| 2 * 6 * class.slot_bytes())
        .sum();
    let config = render::residency::ResidencyConfig::default()
        .with_budget(floor_bytes + 12 * big)
        .and_then(|config| config.with_buffer_bytes(4 * big))
        .expect("a valid configuration");
    let small =
        GpuResidency::with_config(Arc::clone(&gpu), project.store(), config).expect("residency");
    let mut owner = Owner::new(project, NoBrush, small, setup(None), Instant::now());
    owner.camera.zoom(0.5);
    let mut screen = Screen::new(&gpu);
    let turn = crate::camera::Rotation::about(crate::camera::Vec3::new(0.0, 1.0, 0.0), 0.15);
    let mut deepest = 0;
    for _ in 0..40 {
        owner.camera.orbit(turn);
        let now = Instant::now();
        let deadline = now + Duration::from_secs(600);
        assert!(draw_one(&gpu, &mut owner, &mut screen, now, deadline));
        // The cut residency heard this frame, and every ancestor of it, survived its staging.
        let residency = owner.uploads().residency();
        for key in residency.cut() {
            assert!(
                residency.slot(key).is_some(),
                "{key:?} of the cut is resident"
            );
            let mut above = Place::of(key).parent();
            while let Some(place) = above {
                let ancestor = place.key(residency.finest()).expect("a tile");
                assert!(
                    residency.slot(&ancestor).is_some(),
                    "{ancestor:?} above the cut"
                );
                above = place.parent();
            }
            deepest = deepest.max(key.level().get());
        }
    }
    let stats = owner.uploads().stats();
    assert!(
        stats.evictions > 0,
        "the budget was tight enough to evict: {stats:?}"
    );
    assert!(deepest > 6, "cuts reached past the floor: level {deepest}");
    owner.close().expect("closed");
}

/// `field`'s value in the record line `line`, as written.
fn record_field<'a>(line: &'a str, field: &str) -> &'a str {
    let key = format!("\"{field}\":");
    let (_, rest) = line
        .split_once(&key)
        .unwrap_or_else(|| panic!("no {field} in {line}"));
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    &rest[..end]
}

/// The real loop on lavapipe, its display far faster than lavapipe draws (2000 Hz).
///
/// It makes a set number of frames (the camera on the benchmark path at real speed, which takes
/// far longer), so how many it makes doesn't depend on the machine's load.
///
/// The completions the render thread asks the queue for reach the owner as it polls (else the
/// loop would wait for the GPU forever after two frames), every frame's cut and GPU times are
/// read back, and none is dropped.
///
/// Lavapipe can't show the GPU falling behind: its submission waits for the work before it, so
/// it never has more than a frame or so unfinished. The model in `schedule`'s tests shows the
/// bound holding against a GPU that queues.
#[test]
fn a_display_faster_than_the_gpu_gets_every_frame_read_back() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut project = project(&dir);
    while project.store().has_jobs() {
        project
            .store()
            .run_jobs(Instant::now() + Duration::from_secs(1))
            .expect("the display layer builds");
    }
    let size = setup(None).size.expect("a size");
    let target = Target::Offscreen(Offscreen::new(&gpu, size).expect("a target"));
    let hooks = crate::render_thread::Hooks::none();
    let mut render =
        crate::render_thread::RenderThread::spawn(Arc::clone(&gpu), target, hooks, true)
            .expect("the render thread");
    let mut residency = GpuResidency::new(Arc::clone(&gpu), project.store()).expect("residency");
    let lod_timing = residency.enable_timing().expect("the level choice's timer");
    let opening = Instant::now();
    let made = 40;
    let setup = Setup {
        interval: Duration::from_micros(500),
        frames: Some(made),
        script: Some(Script::benchmark(crate::script::Speed::ONE)),
        record: Some(RecordSetup {
            project: "test".to_owned(),
            gpu_timing: render.timing(),
            lod_timing,
            opening,
        }),
        ..setup(None)
    };
    let mut owner = Owner::new(project, NoBrush, residency, setup, Instant::now());
    crate::drive_offscreen(&mut owner, &mut render).expect("the run");
    let stopped = render.stop().expect("stopped cleanly");
    owner.last_gpu_times(stopped.gpu);
    let lines = owner.record_lines();
    owner.close().expect("closed");
    let run = &lines[0];
    assert_eq!(record_field(run, "readbacks_dropped"), "0", "{run}");
    let frames = lines.get(6..).expect("frame lines");
    assert_eq!(
        frames.len(),
        usize::try_from(made).expect("small"),
        "the frames asked for"
    );
    // The last frames' readbacks may still be on their way as the run ends; none before them.
    // Their GPU times are read as the render thread stops.
    for frame in frames {
        assert_ne!(
            record_field(frame, "gpu_ms"),
            "null",
            "its times came back: {frame}"
        );
    }
    let settled = frames.len() - render::FRAMES_IN_FLIGHT;
    for frame in frames.get(..settled).expect("frames") {
        assert_eq!(
            record_field(frame, "drawn"),
            "true",
            "its cut came back: {frame}"
        );
        for readback in ["cut", "request", "timing"] {
            let name = format!("{readback}_readback_dropped");
            assert_eq!(record_field(frame, &name), "false", "{frame}");
        }
    }
}

/// Draws a frame of `owner` on `screen` as soon as one is wanted; returns its visibility buffer.
fn draw_when_wanted<B: Brush>(
    gpu: &Gpu,
    owner: &mut Owner<B, GpuResidency>,
    screen: &mut Screen,
) -> Vec<u32> {
    loop {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => {
                assert!(draw_one(gpu, owner, screen, now, deadline), "drawn");
                return screen.ids(gpu);
            }
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            other @ (Next::Render | Next::Gpu | Next::Idle { .. }) => {
                panic!("a frame is wanted: {other:?}")
            }
        }
    }
}

#[test]
fn a_dab_shows_in_the_visibility_buffer_of_its_own_frame() {
    let dir = TestDir::new();
    let gpu = software_gpu();
    let mut project = project(&dir);
    build_display(&mut project);
    let picker = Picker::load(&mut project)
        .expect("the index loads")
        .expect("a display layer");
    // Strong (0.15 of the radius), so the dab moves the surface by pixels in a 320 × 200 frame.
    let brush = Sculpting::new(&mut project).with_strength(1.5);
    let residency = GpuResidency::new(Arc::clone(&gpu), project.store()).expect("residency");
    let mut owner =
        Owner::new(project, brush, residency, setup(None), Instant::now()).with_picker(picker);
    let mut screen = Screen::new(&gpu);
    settle(&gpu, &mut owner, &mut screen);
    // A still frame is drawn at the next of the accumulation's jitters, so frames are compared
    // drawn unjittered: each just after the camera came back from a nudge, so moving.
    let home = owner.camera;
    let mut nudged = home;
    nudged.zoom(1.01);
    let mut moving_frame = |sculpting: &mut Owner<Sculpting, GpuResidency>| {
        sculpting.camera = nudged;
        sculpting.expose();
        draw_when_wanted(&gpu, sculpting, &mut screen);
        sculpting.camera = home;
        sculpting.expose();
        draw_when_wanted(&gpu, sculpting, &mut screen)
    };
    let before = moving_frame(&mut owner);
    let again = moving_frame(&mut owner);
    assert!(before == again, "the same view draws the same ids");
    // A stroke across the sphere's middle, all of it taken by the next frame at `home`.
    let drags = (1..6).map(|k| (input::Phase::Drag, 150.0 + 4.0 * f64::from(k)));
    let stroke = std::iter::once((input::Phase::Down, 150.0)).chain(drags);
    owner.camera = nudged;
    owner.expose();
    draw_when_wanted(&gpu, &mut owner, &mut screen);
    owner.camera = home;
    for (phase, x) in stroke {
        owner.pen(PenSample {
            pressure: input::Unit::ONE,
            ..pen_sample(phase, (x, 100.0))
        });
    }
    let after = draw_when_wanted(&gpu, &mut owner, &mut screen);
    // Applied as finest work or, where the frame's time ran short (a loaded machine), shown
    // ahead of it: either way the frame draws them.
    let report = owner.brush.last();
    assert!(
        report.dabs_applied + report.dabs_shown > 0,
        "the frame applied or showed dabs: {report:?}"
    );
    let changed = before.iter().zip(&after).filter(|(a, b)| a != b).count();
    assert!(changed > 0, "the dab's frame drew the moved surface");
    owner.close().expect("closed");
}
