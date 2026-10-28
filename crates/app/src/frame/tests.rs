//! The frame loop on a small project, with no GPU: the order of its phases, render on demand,
//! and closing.

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

/// Makes frames for as long as the owner wants them, up to `most`, each shown at once;
/// returns how many it made.
fn run<B: Brush, U: Uploads>(owner: &mut Owner<B, U>, most: u64) -> u64 {
    let mut made = 0;
    let mut now = Instant::now();
    while made < most {
        match owner.next(now) {
            Next::Frame { deadline } => {
                let request = owner.frame(now, deadline).expect("a frame");
                owner.shown(presented(request.number));
                made += 1;
            }
            Next::Wait { until } => now = until,
            Next::Render => panic!("the frame was shown"),
            Next::Idle { .. } => break,
        }
    }
    made
}

/// A brush that, on its first step only, edits one vertex of one tile and closes the epoch.
#[derive(Debug, Default)]
struct OneEdit {
    /// Whether it has made its edit.
    done: bool,
}

impl Brush for OneEdit {
    fn step(&mut self, project: &mut Project) -> Result<()> {
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
fn the_brush_steps_before_the_feed_is_drained_so_its_epoch_is_uploaded_that_frame() {
    let dir = TestDir::new();
    let mut owner = Owner::new(
        project(&dir),
        OneEdit::default(),
        Recording::default(),
        setup(Some(1)),
        Instant::now(),
    );
    let now = Instant::now();
    let Next::Frame { deadline } = owner.next(now) else {
        panic!("the first frame is wanted");
    };
    let request = owner.frame(now, deadline).expect("a frame");
    let first = owner.uploads().frames.first().expect("uploads ran");
    let keys: usize = first.iter().map(|epoch| epoch.keys.len()).sum();
    assert!(
        !first.is_empty() && keys > 0,
        "the brush's epoch, in the frame it was made: {first:?}"
    );
    assert_eq!(owner.feed().epochs, first.len() as u64);
    let text = request.overlay.join("\n");
    assert!(
        text.contains("tiles      3 tiles"),
        "the uploads' summary: {text}"
    );
    owner.shown(presented(request.number));
    assert!(owner.done());
    owner.close().expect("closed");
    let kept: Vec<_> = std::fs::read_dir(dir.path().join("recovery"))
        .expect("the recovery directory")
        .collect();
    assert_eq!(kept.len(), 1, "unsaved edits keep their recovery file");
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
