//! The frame loop on a small project, on the real clock: the order of its phases, render on
//! demand, and closing.
//!
//! Mostly with no GPU; with GPU residency on lavapipe, the floor uploaded, a lost frame
//! recovered, and an idle residency asking for no frames.

use std::sync::Arc;

use anyhow::bail;
use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use render::Outcome;
use render::gpu::{AdapterRequest, Gpu};
use store::{Epoch, ProjectConfig, StoreConfig};

use super::*;
use crate::camera::Pixel;
use crate::navigate::Button;
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
            converging,
            summary: Some("3 tiles".to_owned()),
            frame: None,
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
    OneEdit::default()
        .step(owner.project(), &Camera::default())
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
    OneEdit::default()
        .step(owner.project(), &Camera::default())
        .expect("an edit");
    owner.resize(None);
    let later = Instant::now() + schedule::UPKEEP_INTERVAL;
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
    assert_eq!(
        after,
        later + schedule::UPKEEP_INTERVAL,
        "and is due again later"
    );
    owner.close().expect("closed");
}

/// A software GPU: lavapipe, in the gate.
fn software_gpu() -> Arc<Gpu> {
    Arc::new(Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe"))
}

/// The frame loop over a test project in `dir` whose display layer is built already, its
/// tiles kept on `gpu`.
fn resident_owner(dir: &TestDir, gpu: &Arc<Gpu>) -> Owner<NoBrush, GpuResidency> {
    let mut project = project(dir);
    while project.store().has_jobs() {
        project
            .store()
            .run_jobs(Instant::now() + Duration::from_secs(1))
            .expect("the display layer builds");
    }
    let residency = GpuResidency::new(Arc::clone(gpu), project.store()).expect("residency");
    Owner::new(project, NoBrush, residency, setup(None), Instant::now())
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
            Next::Render => panic!("the frame was shown"),
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
    let deadline = first_deadline(&owner);
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
    // (it agrees: nothing requested), no more frames are wanted.
    let settling = run_on_gpu(&gpu, &mut owner, 100);
    assert!(
        (1..=3).contains(&settling),
        "a frame or so to read the readback, then none: {settling}"
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
    let deadline = first_deadline(&owner);
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
    let place = render::tree::Place::of(key);
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
