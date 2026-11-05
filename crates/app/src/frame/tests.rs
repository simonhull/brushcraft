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
use ui::{Bar, Control, Emphasis, Theme, UiFrame};

use super::*;
use crate::camera::Pixel;
use crate::dialogs::Action;
use crate::navigate::Button;
use crate::pen::PenFrame;
use crate::render_thread::draw_frame;
use crate::saving::{SaveStep, Stage};
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
        blend: Blend::Display,
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
    let overlay = fallback.overlay(size, OwnerPhases::default(), None, false);
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
fn a_moving_strokes_frames_run_the_jobs_time_boxed_and_want_frames_for_them() {
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
    let began = Instant::now();
    frame(&mut owner, began + Duration::from_secs(1));
    let report = owner.after_hand_off.report;
    assert!(
        report.slices > 0,
        "the refresh runs in a stroke's frame too"
    );
    assert!(
        owner.after_hand_off.jobs < Duration::from_millis(200),
        "time-boxed to its share, not the frame's deadline: {:?}",
        owner.after_hand_off.jobs
    );
    assert!(
        matches!(
            owner.next(Instant::now()),
            Next::Frame { .. } | Next::Wait { .. } | Next::Gpu
        ),
        "the jobs left want the next frame while the stroke moves"
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
fn a_late_strokes_frame_starts_no_slice_of_the_jobs() {
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
    // A frame already past the latency aim when it started: no slice, however long it has.
    let began = Instant::now().checked_sub(LATENCY_AIM * 2).expect("a past");
    owner
        .frame_fed(
            began,
            began + Duration::from_secs(1),
            &mut NoPen,
            |_| Ok(()),
        )
        .expect("a frame");
    owner.shown(presented(1));
    assert_eq!(owner.after_hand_off.report.slices, 0, "the pen comes first");
    owner.close().expect("closed");
}

#[test]
fn the_jobs_yield_to_a_strokes_sample_and_keep_a_share_while_the_stroke_is_open() {
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
    // That frame opens the brush's stroke: the jobs keep a share of it, no more.
    frame(&mut owner, Instant::now() + Duration::from_secs(1));
    assert!(
        owner.after_hand_off.jobs < Duration::from_millis(200),
        "time-boxed while the stroke moves: {:?}",
        owner.after_hand_off.jobs
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
        drawn: Drawing::ALL,
        waiting: &[],
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
        drawn: Drawing::ALL,
        waiting: &[],
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
        stroke_open: false,
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
        ui: None,
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
            path: None,
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

/// A brush that holds a dab from its first step on, as a stroke does: the project is unsaved from
/// that frame, and not before.
#[derive(Debug, Default)]
struct HoldsADab {
    /// Whether it has stepped.
    stepped: bool,
}

impl Brush for HoldsADab {
    fn step(
        &mut self,
        _context: BrushContext<'_>,
        _pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        self.stepped = true;
        Ok(BrushReport::default())
    }

    fn landing(&self) -> usize {
        usize::from(self.stepped)
    }
}

/// An owner of a small project with `brush`, its frames on demand.
fn ui_owner<B: Brush>(dir: &TestDir, brush: B) -> Owner<B, NoUploads> {
    Owner::new(project(dir), brush, NoUploads, setup(None), Instant::now())
}

#[test]
fn a_frame_with_nothing_to_show_carries_no_ui() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, NoBrush);
    let deadline = first_deadline(&mut owner);
    let request = frame(&mut owner, deadline);
    assert!(request.ui.is_none(), "nothing to say");
}

#[test]
fn the_chip_is_worked_out_after_the_brush_in_the_frame_it_stepped() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    assert_eq!(owner.landing(), 0, "nothing yet");
    let deadline = first_deadline(&mut owner);
    let request = frame(&mut owner, deadline);
    assert!(owner.brush().stepped, "the brush phase ran");
    assert!(
        request.ui.is_some(),
        "this frame's chip knows what this frame's brush did, so it came after it"
    );
}

#[test]
fn a_chip_fading_wants_frames_until_it_settles_then_none() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    let fading = owner.next(Instant::now());
    assert!(
        matches!(fading, Next::Frame { .. } | Next::Wait { .. }),
        "the fade wants a frame: {fading:?}"
    );

    let started = Instant::now();
    let made = run(&mut owner, 200);
    assert!(made >= 2, "frames while it fades: {made}");
    let fade = Theme::GLASS.fade;
    assert!(started.elapsed() >= fade / 2, "they lasted the fade");
    let settled = owner.next(Instant::now());
    assert!(matches!(settled, Next::Idle { .. }), "{settled:?}");
}

#[test]
fn a_pen_down_touches_the_fades_and_rebuilds_nothing() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    let deadline = first_deadline(&mut owner);
    let before = frame(&mut owner, deadline).ui.expect("the chip");
    owner.pen(pen_sample(input::Phase::Down, (10.0, 10.0)));
    let during = frame(&mut owner, Instant::now() + INTERVAL).ui;
    let during = during.expect("the chip");
    assert!(Arc::ptr_eq(&before.list, &during.list), "no rebuild");
    let dimmed = Theme::GLASS.chip_pen_down;
    assert!((during.fades[1].to - dimmed).abs() < 1e-6, "dimming");

    owner.pen(pen_sample(input::Phase::Up, (10.0, 10.0)));
    let after = frame(&mut owner, Instant::now() + INTERVAL).ui;
    let after = after.expect("the chip");
    assert!(
        Arc::ptr_eq(&before.list, &after.list),
        "no rebuild on the up"
    );
    assert!((after.fades[1].to - 1.0).abs() < 1e-6, "coming back");
}

#[test]
fn a_saved_chip_wakes_the_idle_loop_to_leave() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, NoBrush);
    let now = Instant::now();
    owner.dialogs.saved(now);
    let leaves = owner.dialogs.next_change().expect("\"Saved\" will go");
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(run(&mut owner, 200) < 200, "the fade ends");

    // The upkeep just done, so only the chip is left to wake for.
    let just_before = now + Theme::GLASS.saved_hold.mul_f32(0.99);
    owner.upkeep(just_before).expect("upkeep");
    let Next::Idle { upkeep: wake } = owner.next(just_before) else {
        panic!("settled, with nothing wanted");
    };
    assert_eq!(wake, leaves, "it wakes for the chip, before the upkeep");
    let after = owner.next(leaves);
    assert!(
        matches!(after, Next::Frame { .. } | Next::Wait { .. }),
        "{after:?}"
    );
}

#[test]
fn a_window_moved_to_a_fractional_scale_is_drawn_again() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, NoBrush);
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(run(&mut owner, 10) < 10, "settled");
    owner.set_scale(1.25);
    assert_eq!(owner.overlay_scale, 1, "the overlay rounds it away");
    let next = owner.next(Instant::now());
    assert!(
        matches!(next, Next::Frame { .. } | Next::Wait { .. }),
        "{next:?}"
    );
}

/// A frame of `owner`'s, wanted or not; what it hands over of the UI.
fn ui_of<B: Brush, U: Uploads>(owner: &mut Owner<B, U>) -> Option<UiFrame> {
    frame(owner, Instant::now() + INTERVAL).ui
}

/// Asserts `owner` wants a frame.
fn wants_a_frame<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, what: &str) {
    let next = owner.next(Instant::now());
    assert!(
        matches!(next, Next::Frame { .. } | Next::Wait { .. }),
        "{what}: {next:?}"
    );
}

#[test]
fn a_stroke_ended_by_a_leave_brings_the_chip_back() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(run(&mut owner, 200) < 200, "the chip settles");

    owner.pen(pen_sample(input::Phase::Down, (10.0, 10.0)));
    let down = ui_of(&mut owner).expect("the chip");
    let dimmed = Theme::GLASS.chip_pen_down;
    assert!((down.fades[1].to - dimmed).abs() < 1e-6, "dimmed");
    assert!(run(&mut owner, 200) < 200, "dim, and settled");

    owner.pen(pen_sample(input::Phase::Leave, (10.0, 10.0)));
    wants_a_frame(&mut owner, "the fade back needs drawing");
    let after = ui_of(&mut owner).expect("the chip");
    assert!((after.fades[1].to - 1.0).abs() < 1e-6, "the pen is up");
}

#[test]
fn a_leave_with_no_stroke_changes_nothing() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(run(&mut owner, 200) < 200, "the chip settles");
    owner.pen(pen_sample(input::Phase::Leave, (10.0, 10.0)));
    let next = owner.next(Instant::now());
    assert!(matches!(next, Next::Idle { .. }), "{next:?}");
}

#[test]
fn a_save_that_ends_between_frames_still_says_saved() {
    let dir = TestDir::new();
    let mut project = project(&dir);
    let path = dir.path().join("saved.bcrf");
    drop(project.save_as(&path).expect("saved"));
    let mut owner = Owner::new(
        project,
        OneEdit::default(),
        NoUploads,
        setup(None),
        Instant::now(),
    );
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(owner.unsaved(), "the edit");

    owner.request_save(Instant::now());
    for _ in 0..100_000 {
        if owner.save_stage().is_none() {
            break;
        }
        owner.save_step();
        owner.upkeep(Instant::now()).expect("upkeep");
    }
    assert_eq!(owner.save_stage(), None, "saved, and no frame saw it");
    assert!(!owner.unsaved(), "nothing left");

    wants_a_frame(&mut owner, "to say so");
    assert!(ui_of(&mut owner).is_some(), "a chip");
    assert!(owner.dialogs.next_change().is_some(), "that will leave");
}

#[test]
fn the_first_chip_waits_for_the_up() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    owner.pen(pen_sample(input::Phase::Down, (10.0, 10.0)));
    let deadline = first_deadline(&mut owner);
    let first = frame(&mut owner, deadline);
    assert!(owner.unsaved(), "the first dab");
    assert!(first.ui.is_none(), "no chip, and no atlas, under the pen");
    owner.pen(pen_sample(input::Phase::Drag, (11.0, 10.0)));
    assert!(ui_of(&mut owner).is_none(), "nor on the next frame");

    owner.pen(pen_sample(input::Phase::Up, (11.0, 10.0)));
    let up = ui_of(&mut owner).expect("the chip, on the up");
    let now = Instant::now();
    assert!((up.fades[1].from).abs() < 1e-6, "it fades in");
    assert!((up.fades[1].to - 1.0).abs() < 1e-6, "to full");
    assert!(owner.dialogs.needs_frame(now), "over frames");
}

#[test]
fn a_first_chip_waiting_comes_on_a_leave_too() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    owner.pen(pen_sample(input::Phase::Down, (10.0, 10.0)));
    let deadline = first_deadline(&mut owner);
    assert!(frame(&mut owner, deadline).ui.is_none(), "held back");
    assert!(run(&mut owner, 200) < 200, "nothing wants a frame for it");

    owner.pen(pen_sample(input::Phase::Leave, (11.0, 10.0)));
    wants_a_frame(&mut owner, "the chip that waited");
    assert!(ui_of(&mut owner).is_some(), "the chip");
}

/// A bar of two quiet controls: one puts it away, the other does nothing yet.
fn test_bar() -> Bar<Action> {
    let quiet = |label: &str, action| Control {
        label: label.to_owned(),
        action,
        emphasis: Emphasis::Quiet,
    };
    let controls = vec![
        quiet("Put it away", Action::Dismiss),
        quiet("Leave it", Action::Ignore),
    ];
    Bar::new("A question", controls)
}

/// A window big enough for a bar's controls, its frames on demand.
fn big_setup() -> Setup {
    Setup {
        size: Size::new(1280, 800),
        ..setup(None)
    }
}

/// [`pen_owner`] in a window big enough for a bar, showing [`test_bar`].
fn bar_owner(dir: &TestDir, claims: bool) -> Owner<PenLog, NoUploads> {
    let brush = PenLog {
        claims,
        ..PenLog::default()
    };
    let mut owner = Owner::new(project(dir), brush, NoUploads, big_setup(), Instant::now());
    run(&mut owner, 10_000);
    owner.dialogs.show_bar(test_bar());
    run(&mut owner, 10_000);
    owner.brush.frames.clear();
    owner
}

/// A pen sample in `phase` at `at`.
fn pen_at(phase: input::Phase, at: Pixel) -> PenSample {
    pen_sample(phase, (at.x, at.y))
}

/// Every sample the brush was handed, frame after frame.
fn brush_saw(owner: &Owner<PenLog, NoUploads>) -> Vec<input::Phase> {
    let frames = owner.brush.frames.iter();
    frames
        .flat_map(|(phases, _)| phases.iter().copied())
        .collect()
}

/// A point well off any bar.
const ON_THE_MODEL: Pixel = Pixel { x: 640.0, y: 200.0 };

#[test]
fn a_down_on_a_control_never_reaches_the_brush() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let on = owner.dialogs.control(Action::Dismiss);
    owner.pen(pen_at(Phase::Down, on));
    assert!(owner.pen_queue.is_empty(), "no sample is queued");
    owner.expose();
    run(&mut owner, 100);
    assert!(brush_saw(&owner).is_empty(), "{:?}", owner.brush.frames);
    assert_eq!(owner.pen_router.stroke(), None, "no stroke began");
    assert!(!owner.dialogs.pen_is_down(), "the bar stays up under it");
    owner.close().expect("closed");
}

#[test]
fn the_rest_of_that_stroke_never_reaches_the_brush_either() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let on = owner.dialogs.control(Action::Ignore);
    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Drag, on));
    owner.pen(pen_at(Phase::Up, on));
    assert!(owner.pen_queue.is_empty(), "none of it is queued");
    owner.expose();
    run(&mut owner, 100);
    assert!(brush_saw(&owner).is_empty(), "{:?}", owner.brush.frames);
    assert_eq!(owner.pen_router.stroke(), None, "no stroke began");

    owner.brush.frames.clear();
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    run(&mut owner, 100);
    assert_eq!(
        brush_saw(&owner),
        [Phase::Down],
        "the next stroke is the brush's"
    );
    owner.close().expect("closed");
}

#[test]
fn the_control_acts_on_the_up_inside_it_and_not_outside() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let (on, other) = (
        owner.dialogs.control(Action::Dismiss),
        owner.dialogs.control(Action::Ignore),
    );
    owner.pen(pen_at(Phase::Down, on));
    assert!(owner.dialogs.bar_showing(), "the Down does nothing");
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    assert!(owner.dialogs.bar_showing(), "the Up was off it");

    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Up, other));
    assert!(owner.dialogs.bar_showing(), "the Up was on another control");

    owner.pen(pen_at(Phase::Down, other));
    owner.pen(pen_at(Phase::Up, on));
    assert!(owner.dialogs.bar_showing(), "and the other way about");

    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Up, on));
    assert!(!owner.dialogs.bar_showing(), "back inside it: it acts");
    wants_a_frame(&mut owner, "the bar's fade");
    owner.close().expect("closed");
}

#[test]
fn a_claimed_stroke_ended_by_a_leave_drops_its_action_and_the_leave_goes_on() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let on = owner.dialogs.control(Action::Dismiss);
    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Leave, on));
    assert!(owner.dialogs.bar_showing(), "no Up, no action");
    assert_eq!(owner.pen_queue.len(), 1, "the Leave is the pen's news");
    owner.pen(pen_at(Phase::Up, on));
    assert!(owner.dialogs.bar_showing(), "a stray Up is nobody's");

    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Hover, on));
    owner.pen(pen_at(Phase::Up, on));
    assert!(owner.dialogs.bar_showing(), "a Hover ends it just the same");

    owner.pen(pen_at(Phase::Down, on));
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    assert!(
        owner.dialogs.pen_is_down(),
        "a Down with the Up lost starts over"
    );
    assert!(owner.dialogs.bar_showing(), "and acts on nothing");
    owner.close().expect("closed");
}

#[test]
fn a_down_on_the_bars_glass_off_any_control_is_swallowed() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let glass = owner.dialogs.bare_glass();
    owner.pen(pen_at(Phase::Down, glass));
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    owner.expose();
    run(&mut owner, 100);
    assert!(brush_saw(&owner).is_empty(), "{:?}", owner.brush.frames);
    assert!(owner.dialogs.bar_showing(), "and nothing was pressed");
    owner.close().expect("closed");
}

#[test]
fn a_down_off_the_bar_goes_where_it_went_before() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    run(&mut owner, 100);
    assert_eq!(
        brush_saw(&owner),
        [Phase::Down, Phase::Drag, Phase::Up],
        "the whole stroke"
    );
    owner.close().expect("closed");
}

#[test]
fn a_stroke_that_began_on_the_model_keeps_going_over_the_bar() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, true);
    let on = owner.dialogs.control(Action::Dismiss);
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    owner.pen(pen_at(Phase::Drag, on));
    owner.pen(pen_at(Phase::Up, on));
    run(&mut owner, 100);
    assert_eq!(
        brush_saw(&owner),
        [Phase::Down, Phase::Drag, Phase::Up],
        "every sample"
    );
    assert!(owner.dialogs.bar_showing(), "and the Up pressed nothing");
    owner.close().expect("closed");
}

/// The mouse's left button.
const LEFT: Button = Button::MouseLeft;

#[test]
fn a_mouse_press_on_a_control_never_reaches_navigation() {
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, false);
    let before = *owner.camera();
    let at = owner.dialogs.control(Action::Ignore);
    owner.input(Input::Press { button: LEFT, at });
    assert!(owner.input.is_empty(), "the press stops here");
    owner.input(Input::Move { at: ON_THE_MODEL });
    run(&mut owner, 100);
    assert_eq!(*owner.camera(), before, "no drag began, so nothing orbits");
    owner.input(Input::Release { button: LEFT });
    assert!(
        !owner
            .input
            .iter()
            .any(|input| matches!(input, Input::Release { .. })),
        "nor does its release go on"
    );
    owner.close().expect("closed");
}

#[test]
fn the_mouse_acts_on_a_release_inside_the_control_only() {
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, false);
    let at = owner.dialogs.control(Action::Dismiss);
    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Move { at: ON_THE_MODEL });
    owner.input(Input::Release { button: LEFT });
    assert!(owner.dialogs.bar_showing(), "released off it");

    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Cancel);
    owner.input(Input::Release { button: LEFT });
    assert!(
        owner.dialogs.bar_showing(),
        "focus went, and the press with it"
    );

    owner.input(Input::Press { button: LEFT, at });
    owner.pointer_left();
    owner.input(Input::Release { button: LEFT });
    assert!(owner.dialogs.bar_showing(), "released outside the window");

    owner.input(Input::Move { at });
    owner.input(Input::Press {
        button: Button::MouseRight,
        at,
    });
    owner.input(Input::Release {
        button: Button::MouseRight,
    });
    assert!(owner.dialogs.bar_showing(), "only the left button presses");

    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Release { button: LEFT });
    assert!(!owner.dialogs.bar_showing(), "a click");
    owner.close().expect("closed");
}

#[test]
fn a_press_off_the_bar_navigates_as_ever() {
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, false);
    let before = *owner.camera();
    owner.input(Input::Press {
        button: LEFT,
        at: ON_THE_MODEL,
    });
    owner.input(Input::Move {
        at: Pixel { x: 700.0, y: 240.0 },
    });
    run(&mut owner, 100);
    assert_ne!(*owner.camera(), before, "it orbits");
    owner.close().expect("closed");
}

#[test]
fn with_no_bar_routing_costs_one_branch() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = pen_owner(&dir, true);
    assert!(!owner.dialogs.routes(), "nothing to route to");
    owner.pen(pen_sample(Phase::Down, (160.0, 100.0)));
    owner.pen(pen_sample(Phase::Drag, (170.0, 100.0)));
    owner.pen(pen_sample(Phase::Up, (170.0, 100.0)));
    owner.pen(pen_sample(Phase::Hover, (170.0, 100.0)));
    let at = Pixel { x: 160.0, y: 100.0 };
    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Move { at });
    owner.input(Input::Release { button: LEFT });
    assert_eq!(owner.dialogs.hit_tests(), 0, "nothing was hit tested");
    owner.close().expect("closed");
}

#[test]
fn hovering_a_control_wants_a_frame_and_only_with_a_bar() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = bar_owner(&dir, false);
    let on = owner.dialogs.control(Action::Dismiss);
    let hover = Theme::GLASS.hover;
    let hovered = |ui: Option<UiFrame>| {
        let frame = ui.expect("the bar");
        frame.list.rects.iter().any(|rect| rect.fill == hover)
    };

    owner.input(Input::Move { at: on });
    wants_a_frame(&mut owner, "the mouse's hover");
    assert!(hovered(ui_of(&mut owner)), "it fills");
    owner.pen(pen_at(Phase::Hover, ON_THE_MODEL));
    wants_a_frame(&mut owner, "the pen moved off");
    assert!(!hovered(ui_of(&mut owner)), "it clears");
    owner.pen(pen_at(Phase::Hover, on));
    assert!(hovered(ui_of(&mut owner)), "the pen's hover too");
    owner.pen(pen_at(Phase::Leave, on));
    assert!(!hovered(ui_of(&mut owner)), "the pen left");
    owner.input(Input::Move { at: on });
    owner.pointer_left();
    assert!(!hovered(ui_of(&mut owner)), "the mouse left");
    owner.close().expect("closed");
}

/// A brush with dabs to land, one for each of a save's steps, that edits the project on its first
/// step, as a stroke does.
#[derive(Debug, Default)]
struct Lands {
    /// Whether it has made its edit.
    edited: bool,
    /// The dabs it holds.
    held: usize,
    /// Whether the pen is down on a stroke it holds.
    pen: bool,
    /// Whether a close has begun.
    closing: bool,
    /// When the save holding its strokes back was asked for.
    saving: Option<Instant>,
    /// The strokes it took: a `Down` each.
    strokes: usize,
    /// How many times it has dropped dabs.
    drops: u64,
    /// Where it sends strokes, if not to itself: as off the surface.
    routes_to: Option<Route>,
}

impl Brush for Lands {
    fn step(
        &mut self,
        context: BrushContext<'_>,
        pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        let downs = pen
            .for_brush()
            .filter(|sample| sample.phase == input::Phase::Down);
        self.strokes += downs.count();
        if std::mem::replace(&mut self.edited, true) {
            return Ok(BrushReport::default());
        }
        let key = TileKey::displacement(FaceId::new(0), Level::new(FINEST)?, 0, 0)?;
        let project = context.project;
        project.begin_stroke()?;
        project
            .store()
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.0, 0.0, 0.01)))??;
        project.store().close_epoch()?;
        project.end_unrecorded_stroke()?;
        Ok(BrushReport::default())
    }

    fn route(&mut self, _context: BrushContext<'_>, _down: &PenSample) -> Route {
        self.routes_to.unwrap_or(Route::Brush)
    }

    fn begin_closing(&mut self, _context: BrushContext<'_>) -> Result<()> {
        self.closing = true;
        Ok(())
    }

    fn landing(&self) -> usize {
        self.held
    }

    fn begin_saving(&mut self, asked: Instant) {
        self.saving = Some(asked);
    }

    fn pen_down(&self) -> bool {
        self.pen
    }

    fn landing_for_save(&self) -> usize {
        self.held
    }

    fn save_step(&mut self, _context: BrushContext<'_>) -> Result<bool> {
        let landed = self.held > 0;
        self.held = self.held.saturating_sub(1);
        Ok(landed)
    }

    fn end_saving(&mut self) {
        self.saving = None;
    }

    fn drops(&self) -> u64 {
        self.drops
    }

    fn dropped_since(&self, before: u64) -> Result<()> {
        if self.drops == before {
            Ok(())
        } else {
            bail!("no room for the dabs");
        }
    }
}

/// The frame loop over a saved project in `dir`, in a window big enough for a bar, the brush's
/// first edit made and `held` dabs left to land, settled.
fn closing_owner(dir: &TestDir, held: usize) -> Owner<Lands, NoUploads> {
    let mut project = project(dir);
    drop(
        project
            .save_as(&dir.path().join("first-sculpt.bcrf"))
            .expect("saved"),
    );
    let brush = Lands {
        held,
        ..Lands::default()
    };
    let mut owner = Owner::new(project, brush, NoUploads, big_setup(), Instant::now());
    run(&mut owner, 10_000);
    assert!(owner.unsaved(), "the edit");
    owner
}

/// Lets the bar's fade, and any frames it wants, run out.
fn see_the_bar<B: Brush, U: Uploads>(owner: &mut Owner<B, U>) {
    std::thread::sleep(Theme::GLASS.fade);
    run(owner, 10_000);
}

/// [`closing_owner`], its close bar showing.
fn asked_owner(dir: &TestDir, held: usize) -> Owner<Lands, NoUploads> {
    let mut owner = closing_owner(dir, held);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    see_the_bar(&mut owner);
    owner
}

/// A pen tap at `at`: `Down` then `Up`.
fn tap<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, at: Pixel) {
    owner.pen(pen_at(input::Phase::Down, at));
    owner.pen(pen_at(input::Phase::Up, at));
}

/// Passes of the window's loop until the save under way is over, frames among them.
fn finish_saving<B: Brush, U: Uploads>(owner: &mut Owner<B, U>) {
    for _ in 0..100_000 {
        if owner.save_stage().is_none() {
            break;
        }
        assert_eq!(owner.take_request(), None, "nothing goes during the save");
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(owner, 1);
        owner.upkeep(Instant::now()).expect("upkeep");
    }
    assert_eq!(owner.save_stage(), None, "the save ended");
    owner.save_step();
}

#[test]
fn nothing_unsaved_closes_at_once() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, NoBrush);
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Now);
    assert!(!owner.dialogs.bar_showing(), "no question");
}

#[test]
fn unsaved_edits_ask_and_nothing_lands_or_lifts() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 3);
    owner.brush.pen = true;
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.dialogs.asking_close(), "the question");
    assert!(!owner.closing(), "the window isn't going");
    assert!(!owner.brush.closing, "the pen wasn't lifted");
    assert_eq!(owner.landing(), 3, "nothing landed");
    assert_eq!(owner.save_stage(), None, "no save began");
    assert_eq!(owner.take_request(), None);
    owner.pen(pen_sample(input::Phase::Hover, (10.0, 10.0)));
    assert_eq!(owner.pen_queue.len(), 1, "the pen is still taken");
}

#[test]
fn dabs_still_landing_count_as_unsaved() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, HoldsADab::default());
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(!owner.project.has_unsaved_edits(), "no edit yet");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
}

#[test]
fn a_stroke_not_routed_yet_counts_as_unsaved() {
    let dir = TestDir::new();
    let mut owner = ui_owner(&dir, NoBrush);
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    owner.pen(pen_sample(input::Phase::Down, (10.0, 10.0)));
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
}

#[test]
fn a_second_close_while_the_bar_shows_does_nothing_new() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    let before = ui_of(&mut owner).expect("the bar");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked, "still");
    assert!(owner.dialogs.asking_close(), "the same question");
    let after = ui_of(&mut owner).expect("the bar");
    assert!(Arc::ptr_eq(&before.list, &after.list), "no rebuild");
    assert_eq!(before.fades[0], after.fades[0], "no fade");
    assert_eq!(owner.take_request(), None);
}

#[test]
fn keep_sculpting_clears_the_bar() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let keep = owner.dialogs.control(Action::KeepSculpting);
    tap(&mut owner, keep);
    assert!(!owner.dialogs.bar_showing(), "gone");
    assert_eq!(owner.take_request(), None, "the window stays");
    assert_eq!(owner.save_stage(), None, "and nothing saves");
    assert_eq!(owner.landing(), 2, "or lands");
}

#[test]
fn a_down_off_the_bar_dismisses_it_and_sculpts() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    assert!(owner.dialogs.pen_is_down(), "the stroke goes on");
    run(&mut owner, 100);
    assert!(!owner.dialogs.bar_showing(), "gone once it is the brush's");
    owner.pen(pen_at(input::Phase::Up, ON_THE_MODEL));
    run(&mut owner, 100);
    assert_eq!(owner.brush.strokes, 1, "the brush had it");
    assert!(!owner.dialogs.bar_showing(), "and it doesn't come back");
    assert_eq!(owner.take_request(), None, "the window stays");
}

/// Orbiting to look at the model before deciding keeps the question.
#[test]
fn a_mouse_press_off_the_bar_navigates_and_leaves_it() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    owner.input(Input::Press {
        button: LEFT,
        at: ON_THE_MODEL,
    });
    assert_eq!(owner.input.len(), 1, "the press navigates");
    run(&mut owner, 100);
    assert!(owner.dialogs.asking_close(), "and the question stands");
}

/// A pen's navigation is no answer either: a barrel held, the stroke is the camera's.
#[test]
fn a_barrel_stroke_off_the_bar_navigates_and_leaves_it() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    let barrel = |phase| PenSample {
        buttons: input::Buttons::PRIMARY,
        ..pen_at(phase, ON_THE_MODEL)
    };
    owner.pen(barrel(Phase::Hover));
    owner.pen(barrel(Phase::Down));
    run(&mut owner, 100);
    assert!(owner.dialogs.asking_close(), "still asking");
    owner.pen(barrel(Phase::Up));
    run(&mut owner, 100);
    assert!(owner.dialogs.asking_close(), "and after the Up");
    assert_eq!(owner.brush.strokes, 0, "the brush saw none of it");
    let frame = ui_of(&mut owner).expect("the bar");
    assert!((frame.fades[0].to - 1.0).abs() < 1e-6, "back on the Up");
}

/// Nor is the tip off the surface: the brush sends it to navigation, and the question stands.
#[test]
fn a_tip_stroke_off_the_surface_leaves_the_bar() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    owner.brush.routes_to = Some(Route::Navigate);
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    run(&mut owner, 100);
    assert!(owner.dialogs.asking_close(), "still asking");
}

#[test]
fn navigating_while_finishing_keeps_the_exit() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 3);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    see_the_bar(&mut owner);
    owner.input(Input::Press {
        button: LEFT,
        at: ON_THE_MODEL,
    });
    owner.input(Input::Release { button: LEFT });
    finish_saving(&mut owner);
    assert_eq!(
        owner.take_request(),
        Some(WindowRequest::Exit),
        "not cancelled"
    );
}

#[test]
fn save_finishes_then_exits() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 3);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    assert!(owner.dialogs.finishing(), "the bar becomes the wait");
    assert!(owner.save_stage().is_some(), "a save began");
    assert!(
        !owner.closing(),
        "an ordinary save: it can still be cancelled"
    );
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
    assert_eq!(owner.take_request(), None, "once");
    assert!(!owner.unsaved(), "it is all in the save");
}

#[test]
fn save_shows_finishing_with_progress_falling_from_where_it_started() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 4);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    let bar_of = |progress| {
        let cancel = Control {
            label: "Cancel".to_owned(),
            action: Action::Cancel,
            emphasis: Emphasis::Quiet,
        };
        Bar::new("Finishing your stroke", vec![cancel]).with_progress(progress)
    };
    assert_eq!(owner.dialogs.bar(), Some(bar_of(0.0)), "nothing landed yet");
    owner.save_step();
    owner.save_step();
    owner.save_step();
    assert_eq!(owner.landing(), 2, "half");
    std::thread::sleep(Theme::GLASS.progress_step);
    ui_of(&mut owner);
    assert_eq!(owner.dialogs.bar(), Some(bar_of(0.5)), "half way along");
}

#[test]
fn cancel_keeps_the_window_and_the_save_finishes() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 3);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    see_the_bar(&mut owner);
    owner.save_step();
    let cancel = owner.dialogs.control(Action::Cancel);
    tap(&mut owner, cancel);
    assert!(!owner.dialogs.bar_showing(), "the bar goes");
    assert!(owner.save_stage().is_some(), "and the save goes on");
    finish_saving(&mut owner);
    assert!(!owner.unsaved(), "it finished, harmlessly");
    assert_eq!(owner.take_request(), None, "and the window stays");
}

#[test]
fn a_down_while_finishing_is_a_cancel_and_the_stroke_waits_for_the_save() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 3);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    see_the_bar(&mut owner);
    owner.save_step();
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    assert_eq!(
        owner.pen_queue.len(),
        1,
        "the stroke goes to the brush to wait"
    );
    run(&mut owner, 1);
    assert!(
        !owner.dialogs.bar_showing(),
        "cancelled, once it is the brush's"
    );
    assert!(
        owner.brush.saving.is_some(),
        "the save still holds strokes back"
    );
    owner.pen(pen_at(input::Phase::Up, ON_THE_MODEL));
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "the window stays");
}

#[test]
fn a_down_on_the_glass_while_finishing_changes_nothing() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 3);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    see_the_bar(&mut owner);
    let glass = owner.dialogs.bare_glass();
    tap(&mut owner, glass);
    assert!(owner.dialogs.finishing(), "still finishing");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
}

#[test]
fn a_close_during_a_save_shows_finishing_and_never_asks() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 3);
    owner.request_save(Instant::now());
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.dialogs.finishing(), "no question");
    assert!(!owner.dialogs.asking_close());
    let first = owner.dialogs.bar();
    owner.save_step();
    owner.save_step();
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked, "again");
    assert_eq!(owner.dialogs.bar(), first, "from where it started");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
}

#[test]
fn a_close_with_the_pen_down_shows_the_bar_on_the_up() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    // Dated ahead: a pen reporting all the while, however long the test takes.
    let reporting = Instant::now() + Duration::from_secs(60);
    owner.pen(pen_received(Phase::Down, reporting));
    run(&mut owner, 100);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);

    let faint = |frame: Option<UiFrame>| {
        frame.is_none_or(|frame| frame.fades[0].at(Instant::now()).abs() < 1e-6)
    };
    assert!(faint(ui_of(&mut owner)), "nothing shows under the pen");
    owner.pen(pen_received(Phase::Drag, reporting));
    assert!(faint(ui_of(&mut owner)), "on any frame");
    std::thread::sleep(Theme::GLASS.fade);
    assert!(faint(ui_of(&mut owner)), "however long the stroke");
    assert!(
        !owner.key(UiKey::Enter, Instant::now()),
        "nor can a key answer it"
    );
    assert_eq!(owner.save_stage(), None, "what can't be seen isn't asked");

    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    wants_a_frame(&mut owner, "the bar's fade");
    let frame = ui_of(&mut owner).expect("the bar comes");
    assert!((frame.fades[0].to - 1.0).abs() < 1e-6, "rising from the Up");
    assert!(owner.dialogs.asking_close());
}

#[test]
fn a_close_with_a_saves_pen_down_shows_finishing_on_the_up() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    let reporting = Instant::now() + Duration::from_secs(60);
    owner.pen(pen_received(Phase::Down, reporting));
    run(&mut owner, 100);
    owner.brush.pen = true;
    owner.request_save(Instant::now());
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let held_back = ui_of(&mut owner);
    assert!(
        held_back.is_none_or(|ui| ui.fades[0].to.abs() < 1e-6),
        "not yet"
    );
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    owner.brush.pen = false;
    let frame = ui_of(&mut owner).expect("the bar comes");
    assert!((frame.fades[0].to - 1.0).abs() < 1e-6, "on the Up");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
}

#[test]
fn a_scripted_run_never_asks() {
    let dir = TestDir::new();
    let brush = HoldsADab::default();
    let mut owner = Owner::new(
        project(&dir),
        brush,
        NoUploads,
        setup(Some(5)),
        Instant::now(),
    );
    let deadline = first_deadline(&mut owner);
    frame(&mut owner, deadline);
    assert!(owner.unsaved(), "it would ask");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Now);
    assert!(!owner.dialogs.bar_showing());
}

#[test]
fn a_window_with_no_size_goes_rather_than_ask_where_nothing_shows() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 0);
    owner.resize(None);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Now);
}

#[test]
fn enter_saves_and_escape_keeps_sculpting_only_with_the_bar_showing() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 0);
    let now = Instant::now();
    assert!(!owner.key(UiKey::Enter, now), "no bar: the key goes on");
    assert!(!owner.key(UiKey::Escape, now), "the same");
    assert_eq!(owner.save_stage(), None, "and no save");

    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Escape, now), "keeps sculpting");
    assert!(!owner.dialogs.bar_showing(), "the bar goes");
    assert_eq!(owner.take_request(), None);

    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "saves");
    assert!(owner.dialogs.finishing());
    assert!(
        !owner.key(UiKey::Enter, now),
        "Enter does nothing while finishing"
    );
    assert!(owner.key(UiKey::Escape, now), "Escape cancels");
    assert!(!owner.dialogs.bar_showing());
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "cancelled: the window stays");
}

#[test]
fn enter_never_means_dont_save() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 0);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, Instant::now()));
    finish_saving(&mut owner);
    assert_eq!(
        owner.take_request(),
        Some(WindowRequest::Exit),
        "a save, and out"
    );
    assert!(!owner.unsaved(), "saved");
}

#[test]
fn dont_save_runs_only_from_the_close_bar() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 0);
    let now = Instant::now();
    owner.act(Action::DontSave, now);
    assert_eq!(owner.take_request(), None, "no bar asked");

    owner.request_save(now);
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.dialogs.finishing());
    owner.act(Action::DontSave, now);
    assert_eq!(owner.take_request(), None, "finishing isn't asking");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
}

#[test]
fn the_close_bars_dont_save_asks_the_window_to_discard() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    let dont = owner.dialogs.control(Action::DontSave);
    tap(&mut owner, dont);
    assert_eq!(owner.take_request(), Some(WindowRequest::ExitDiscarding));
}

#[test]
fn a_tap_on_dont_save_needs_its_up_inside_it() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    let dont = owner.dialogs.control(Action::DontSave);
    owner.pen(pen_at(Phase::Down, dont));
    assert_eq!(owner.take_request(), None, "not on the Down");
    owner.pen(pen_at(
        Phase::Drag,
        owner.dialogs.control(Action::KeepSculpting),
    ));
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    assert_eq!(owner.take_request(), None, "nor off it");
    assert!(owner.dialogs.asking_close(), "the bar stays");
}

#[test]
fn a_mouse_click_answers_the_bar_too() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    let at = owner.dialogs.control(Action::Save);
    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Release { button: LEFT });
    assert!(owner.dialogs.finishing());
}

#[test]
fn a_bar_just_asked_cant_be_hit_before_it_can_be_seen() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 0);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    // Where Don't save will be, in the first instant of the fade: the Down is the stroke's.
    owner.pen(pen_at(input::Phase::Down, Pixel { x: 640.0, y: 750.0 }));
    assert_eq!(owner.take_request(), None, "nothing was pressed");
}

/// A brush that says when it is told to forget what it holds, and never lands.
#[derive(Debug, Default)]
struct Forgets {
    /// Told by [`Brush::abandon`], and seen after the owner has gone.
    abandoned: std::rc::Rc<std::cell::Cell<bool>>,
}

impl Brush for Forgets {
    fn step(
        &mut self,
        _context: BrushContext<'_>,
        _pen: &PenFrame,
        _budget: &BrushBudget,
    ) -> Result<BrushReport> {
        Ok(BrushReport::default())
    }

    fn flush(&mut self, _context: BrushContext<'_>) -> Result<()> {
        bail!("Don't save lands nothing");
    }

    fn abandon(&mut self) {
        self.abandoned.set(true);
    }
}

#[test]
fn discard_tells_the_brush_to_forget_and_never_to_land() {
    let dir = TestDir::new();
    let brush = Forgets::default();
    let abandoned = std::rc::Rc::clone(&brush.abandoned);
    let owner = ui_owner(&dir, brush);
    owner.discard().expect("discarded");
    assert!(abandoned.get(), "it was told");
}

/// The save can't say what it doesn't hold: `saving.start` counts drops from the Save, so dabs
/// dropped under the bar before it pass as nothing.
#[test]
fn dabs_dropped_under_the_bar_keep_the_window_and_the_title_says_so() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    owner.brush.drops = 1;
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    drop(owner.title_change());
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "the window stays");
    assert!(!owner.dialogs.bar_showing(), "the bar goes");
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("no room for the dabs"), "{title}");

    assert_eq!(
        owner.ask_close(Instant::now()),
        CloseAnswer::Now,
        "told, and with nothing unsaved a further close leaves"
    );
}

#[test]
fn a_further_save_leaves_once_the_drop_was_said() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    owner.brush.drops = 1;
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "the first time, it stays");

    owner.brush.held = 2;
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit), "told once");
}

#[test]
fn dabs_dropped_while_a_failed_save_keeps_the_window_count_as_told() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    owner.brush.drops = 1;
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "that save failed");

    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    finish_saving(&mut owner);
    assert_eq!(
        owner.take_request(),
        Some(WindowRequest::Exit),
        "said already"
    );
}

/// The mouse's release goes by the mouse's place alone: a pen in the other hand steers nothing.
#[test]
fn a_pen_hover_doesnt_steer_a_mouse_release() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let dont = owner.dialogs.control(Action::DontSave);
    owner.input(Input::Press {
        button: LEFT,
        at: dont,
    });
    owner.input(Input::Move { at: ON_THE_MODEL });
    owner.pen(pen_at(input::Phase::Hover, dont));
    owner.input(Input::Release { button: LEFT });
    assert_eq!(owner.take_request(), None, "released off the control");
    assert!(owner.dialogs.asking_close(), "still asking");
}

/// And the other way about: a pen's hover going elsewhere doesn't take a click away.
#[test]
fn a_pen_hover_doesnt_spoil_a_mouse_click() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let at = owner.dialogs.control(Action::KeepSculpting);
    owner.input(Input::Press { button: LEFT, at });
    owner.pen(pen_at(input::Phase::Leave, ON_THE_MODEL));
    owner.input(Input::Release { button: LEFT });
    assert!(!owner.dialogs.bar_showing(), "the click stands");
}

/// `save_step` sets the Exit, then the same pass's `drive` hears a stroke, a Cancel: the window
/// takes its request only after that.
#[test]
fn a_cancel_in_the_same_pass_as_the_exit_keeps_the_window() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 0);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    see_the_bar(&mut owner);
    let mut steps = 0;
    while owner.request.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        owner.upkeep(Instant::now()).expect("upkeep");
        steps += 1;
        assert!(steps < 100_000, "the save ends");
    }
    assert_eq!(owner.request, Some(WindowRequest::Exit), "the save is over");
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    run(&mut owner, 1);
    assert!(!owner.dialogs.bar_showing(), "cancelled");
    assert_eq!(owner.take_request(), None, "and that takes the exit back");
}

/// A stroke after Don't save, in one batch: the bar is answered twice, and the last answer stands.
#[test]
fn a_stroke_after_dont_save_in_one_batch_keeps_the_window() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let dont = owner.dialogs.control(Action::DontSave);
    tap(&mut owner, dont);
    assert_eq!(owner.request, Some(WindowRequest::ExitDiscarding));
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    run(&mut owner, 1);
    assert!(!owner.dialogs.bar_showing(), "dismissed");
    assert_eq!(
        owner.take_request(),
        None,
        "bar gone, and the discard with it"
    );
}

/// A pen sample in `phase` on the model, received at `at`.
fn pen_received(phase: input::Phase, at: Instant) -> PenSample {
    PenSample {
        received: at,
        ..pen_at(phase, ON_THE_MODEL)
    }
}

/// Whether the bar is on its way up, or up.
fn bar_rising<B: Brush, U: Uploads>(owner: &mut Owner<B, U>) -> bool {
    ui_of(owner).is_some_and(|ui| ui.fades[0].to > 0.5)
}

/// A pen whose `Up` was lost, and no sample since: the bar comes all the same, or nothing could
/// ever answer it. Asked for after the quiet, it comes at once.
#[test]
fn a_close_long_after_the_pens_last_sample_shows_the_bar() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.pen(pen_at(input::Phase::Down, ON_THE_MODEL));
    run(&mut owner, 100);
    std::thread::sleep(STROKE_QUIET);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(!owner.dialogs.pen_is_down(), "taken as lifted");
    assert!(bar_rising(&mut owner), "the bar comes");
    assert!(
        owner.key(UiKey::Escape, Instant::now()),
        "and can be answered"
    );
}

/// Asked for during the quiet, the bar comes as it ends, the idle loop waking for it.
#[test]
fn a_close_as_the_pen_goes_quiet_shows_the_bar_when_it_has() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    // Dated ahead, so nothing here depends on how long the test takes.
    let last = Instant::now() + Duration::from_secs(60);
    owner.pen(pen_received(Phase::Down, last));
    run(&mut owner, 100);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    run(&mut owner, 100);
    assert!(owner.dialogs.pen_is_down(), "not quiet yet");
    assert!(!bar_rising(&mut owner), "so no bar");
    run(&mut owner, 100);

    let due = last + STROKE_QUIET;
    owner
        .upkeep(due.checked_sub(Duration::from_millis(1)).expect("later"))
        .expect("upkeep");
    let Next::Idle { upkeep: wake } = owner.next(Instant::now()) else {
        panic!("settled");
    };
    assert_eq!(wake, due, "the wait ends for it");
    wants_a_frame_at(&mut owner, due, "the bar's frame");
    assert!(!owner.dialogs.pen_is_down(), "quiet: taken as lifted");
    assert!(bar_rising(&mut owner), "the bar comes");
}

/// Asserts `owner` wants a frame at `now`.
fn wants_a_frame_at<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, now: Instant, what: &str) {
    let next = owner.next(now);
    assert!(
        matches!(next, Next::Frame { .. } | Next::Wait { .. }),
        "{what}: {next:?}"
    );
}

/// A pen still reporting holds the bar back, a bar never appearing under a stroke.
#[test]
fn a_pen_still_reporting_holds_the_bar_back() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let later = Instant::now() + Duration::from_secs(60);
    owner.pen(pen_received(Phase::Drag, later));
    owner.next(later);
    assert!(owner.dialogs.pen_is_down(), "this sample is the latest");
    owner.next(later + STROKE_QUIET);
    assert!(!owner.dialogs.pen_is_down(), "quiet since it");
}

/// A pen that was only resting, and draws on: the bar goes again until the Up.
#[test]
fn a_pen_moving_again_puts_the_bar_back_down() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    run(&mut owner, 100);
    std::thread::sleep(STROKE_QUIET);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(bar_rising(&mut owner), "the bar came");
    owner.pen(pen_at(Phase::Drag, ON_THE_MODEL));
    assert!(owner.dialogs.pen_is_down(), "the stroke goes on");
    assert!(!bar_rising(&mut owner), "so the bar goes");
    owner.pen(pen_at(Phase::Up, ON_THE_MODEL));
    assert!(bar_rising(&mut owner), "back on the Up");
    assert!(owner.dialogs.asking_close(), "still asking");
}

/// Under a navigation stroke too, the bar already up: it fades, and mustn't stay gone.
#[test]
fn a_navigation_stroke_gone_quiet_gives_the_bar_back() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let last = Instant::now() + Duration::from_secs(60);
    let barrel = |phase| PenSample {
        buttons: input::Buttons::PRIMARY,
        ..pen_received(phase, last)
    };
    owner.pen(barrel(Phase::Hover));
    owner.pen(barrel(Phase::Down));
    run(&mut owner, 100);
    assert!(!bar_rising(&mut owner), "it faded under the stroke");
    owner.next(last + STROKE_QUIET);
    assert!(bar_rising(&mut owner), "and comes back");
}

/// With no pen down nothing is latched, and the check costs the idle loop nothing.
#[test]
fn a_bar_with_no_pen_down_wakes_nobody() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    let far = Instant::now() + Duration::from_secs(1);
    owner.upkeep(far).expect("upkeep");
    let next = owner.next(Instant::now());
    assert!(matches!(next, Next::Idle { .. }), "{next:?}");
}

/// The same with a save holding at its pen: Finishing shows, and Cancel gets out.
#[test]
fn a_lost_up_under_finishing_has_a_way_out() {
    use input::Phase;
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.pen(pen_at(Phase::Down, ON_THE_MODEL));
    run(&mut owner, 100);
    owner.brush.pen = true;
    owner.request_save(Instant::now());
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    std::thread::sleep(STROKE_QUIET);
    for _ in 0..10 {
        owner.save_step();
        run(&mut owner, 1);
    }
    assert_eq!(owner.save_stage(), Some(Stage::PenDown), "it can't go on");
    assert!(bar_rising(&mut owner), "but the bar shows");
    assert!(owner.key(UiKey::Escape, Instant::now()), "Cancel");
    assert_eq!(owner.save_stage(), None, "a save that hadn't begun goes");
}

/// A save stuck at its pen can't be waited for: Cancel lets go of it, so a close reaches the
/// question, and Don't save.
#[test]
fn cancel_lets_go_of_a_save_still_waiting_for_the_pen() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.brush.pen = true;
    owner.request_save(Instant::now());
    assert!(owner.brush.saving.is_some(), "strokes wait for it");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    see_the_bar(&mut owner);
    assert!(owner.key(UiKey::Escape, Instant::now()), "Cancel");
    assert_eq!(owner.save_stage(), None, "nothing had been done");
    assert!(owner.brush.saving.is_none(), "strokes go on");
    assert_eq!(owner.landing(), 2, "and nothing landed");

    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    assert!(owner.dialogs.asking_close(), "the question, not the wait");
    see_the_bar(&mut owner);
    let dont = owner.dialogs.control(Action::DontSave);
    tap(&mut owner, dont);
    assert_eq!(owner.take_request(), Some(WindowRequest::ExitDiscarding));
}

/// The question was about saving: the save key answers it.
#[test]
fn the_save_key_under_the_close_bar_is_its_save() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 2);
    owner.save_key(Instant::now());
    assert!(owner.dialogs.finishing(), "the bar's Save");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
}

#[test]
fn the_save_key_with_no_bar_is_a_save_and_the_window_stays() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    owner.save_key(Instant::now());
    assert!(!owner.dialogs.bar_showing(), "nothing asked");
    finish_saving(&mut owner);
    assert_eq!(owner.take_request(), None, "it stays");
}

/// A bar the pen holds back hasn't been asked yet: the key saves, as it would with none.
#[test]
fn the_save_key_doesnt_answer_a_bar_it_cant_see() {
    let dir = TestDir::new();
    let mut owner = closing_owner(&dir, 2);
    let reporting = Instant::now() + Duration::from_secs(60);
    owner.pen(pen_received(input::Phase::Down, reporting));
    run(&mut owner, 100);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    owner.save_key(Instant::now());
    assert!(owner.save_stage().is_some(), "a save");
    assert!(owner.dialogs.asking_close(), "and the question stands");
}

/// The second click lands where Cancel just came: it must stop on the bar, and undo nothing.
#[test]
fn a_double_click_on_save_stays_a_save() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 50);
    let at = owner.dialogs.control(Action::Save);
    owner.input(Input::Press { button: LEFT, at });
    owner.input(Input::Release { button: LEFT });
    owner.input(Input::Press { button: LEFT, at });
    assert!(owner.input.is_empty(), "the bar took it");
    owner.input(Input::Release { button: LEFT });
    assert!(
        owner.dialogs.finishing(),
        "the second click undid the first"
    );
    assert_eq!(owner.after_save, AfterSave::Exit);
}

/// A pen's double tap the same: its second Down isn't a stroke, or it would answer the bar.
#[test]
fn a_double_tap_on_save_stays_a_save() {
    let dir = TestDir::new();
    let mut owner = asked_owner(&dir, 50);
    let save = owner.dialogs.control(Action::Save);
    tap(&mut owner, save);
    tap(&mut owner, save);
    run(&mut owner, 2);
    assert!(owner.dialogs.finishing(), "the second tap undid the first");
    assert_eq!(owner.brush.strokes, 0, "and no stroke began");
}
