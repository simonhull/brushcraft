//! The frame loop on a small project, with no GPU, on the real clock: the order of its phases,
//! render on demand, and closing.

use anyhow::bail;
use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use render::Outcome;
use store::{Epoch, ProjectConfig, StoreConfig};

use super::*;
use crate::camera::Pixel;
use crate::navigate::Button;
use crate::test_dir::TestDir;

/// The finest level the test projects keep a display layer under: small, so a rebuild takes
/// a few frames.
const FINEST: u8 = 3;

/// A 60 Hz display's interval.
const INTERVAL: Duration = Duration::from_micros(16_667);

/// A never-saved project over a six-face cube sphere, keeping a display layer under level
/// [`FINEST`], its recovery file in `dir`.
fn project(dir: &TestDir) -> Project {
    let mesh = cube_sphere(1).expect("a cube sphere");
    let finest = Level::new(FINEST).expect("a level");
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
    }
}

/// The render thread's report of frame `number`, shown.
const fn presented(number: u64) -> Presented {
    Presented {
        number,
        outcome: Outcome::Presented,
        times: DrawTimes {
            acquire: Duration::ZERO,
            encode: Duration::ZERO,
            present: Duration::ZERO,
        },
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
            Next::Render => panic!("the frame was shown"),
            Next::Idle { .. } => break,
        }
    }
    made
}

/// The first frame's deadline, which the owner wants at once.
fn first_deadline<B: Brush, U: Uploads>(owner: &Owner<B, U>) -> Instant {
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
    fn step(&mut self, project: &mut Project, camera: &Camera) -> Result<()> {
        self.cameras.push(*camera);
        if std::mem::replace(&mut self.done, true) {
            return Ok(());
        }
        let key = TileKey::displacement(FaceId::new(0), Level::new(FINEST)?, 0, 0)?;
        project.begin_stroke()?;
        project
            .store()
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.0, 0.0, 0.01)))??;
        project.store().close_epoch()?;
        project.end_stroke()?;
        Ok(())
    }
}

/// Uploads that record every frame's drained epochs, and can claim to be converging.
#[derive(Debug, Default)]
struct Recording {
    /// The epochs each frame's apply was given, frame by frame.
    frames: Vec<Vec<Epoch>>,
    /// Frames still to report converging.
    converging_for: u32,
}

impl Uploads for Recording {
    fn apply(&mut self, drained: Drained, _store: &mut TileStore) -> Result<UploadStatus> {
        let Drained::Epochs(epochs) = drained else {
            bail!("no resync expected");
        };
        self.frames.push(epochs);
        let converging = self.converging_for > 0;
        self.converging_for = self.converging_for.saturating_sub(1);
        Ok(UploadStatus {
            converging,
            summary: Some("3 tiles".to_owned()),
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
    let deadline = first_deadline(&owner);
    let first = frame(&mut owner, deadline);

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
    owner.expose();
    let Next::Frame { deadline: later } = owner.next(Instant::now() + Duration::from_millis(100))
    else {
        panic!("the next frame is wanted");
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
