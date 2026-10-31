//! Requested finest tiles through the store's queued evaluation: queued when requested,
//! evaluated only by the jobs, staged from their copies, never stale.

use std::time::{Duration, Instant};

use geom::{Channel, FaceId, LayerId, Level, TileKey, Vec3, cube_sphere};
use store::{Drained, Project, ProjectConfig, StaleCursor, StoreConfig, TileStore};
use wgpu::CommandEncoderDescriptor;

use super::{REPEATS, TempDir, array, level, report, slot_contents, with_floor};
use crate::gpu::{Gpu, GpuTest};
use crate::residency::{Request, Requests, Residency};
use crate::upload::{FinestEvaluations, StoreSource, prefetch_requests};

/// The finest level of the small tests' projects.
const FINEST: u8 = 7;

/// A project of the cube's six faces, finest level [`FINEST`], its display layer built, in
/// `dir`; and a cursor on its stale-tile feed.
fn small_project(dir: &TempDir) -> (Project, StaleCursor) {
    let (mut project, cursor) = unbuilt_project(dir);
    let store = project.store();
    store.queue_display_rebuild(level(FINEST));
    run_all_jobs(store);
    drop(store.drain_stale(&cursor));
    (project, cursor)
}

/// [`small_project`] with no display layer built yet.
fn unbuilt_project(dir: &TempDir) -> (Project, StaleCursor) {
    let mesh = cube_sphere(1).expect("a mesh");
    let config = ProjectConfig {
        store: StoreConfig::new(256 << 20, 1).with_display(level(FINEST)),
        recovery_dir: dir.0.clone(),
    };
    let mut project = Project::new(mesh, config).expect("a project");
    let cursor = project.store().subscribe_stale();
    (project, cursor)
}

/// Runs the store's jobs, the queued evaluation's included, until none is left.
fn run_all_jobs(store: &mut TileStore) {
    let began = Instant::now();
    while store.has_jobs() {
        assert!(began.elapsed() < Duration::from_secs(60), "the jobs end");
        store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .expect("the jobs run");
    }
}

/// A requested finest tile of face 2, and the requests for it.
fn requested() -> (TileKey, Requests) {
    let key = TileKey::displacement(FaceId::new(2), level(FINEST), 1, 0).expect("a finest tile");
    let requests = Requests::new(
        [Request {
            key,
            screen_error: 3.0,
        }],
        0,
    );
    (key, requests)
}

/// One frame of the owner thread before the hand-off, with no time to evaluate while staging.
///
/// The feed's drain (given to `evaluations` first), `requests` wanted, the queued evaluation
/// brought in line, and the uploads staged with requested finest tiles read only from their
/// copies, then submitted. Returns how many finest tiles staging read through the store's own
/// evaluation.
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    evaluations: &mut FinestEvaluations,
    store: &mut TileStore,
    cursor: &StaleCursor,
    requests: &Requests,
) -> usize {
    let staging = Staging {
        until: None,
        slow: Duration::ZERO,
    };
    hybrid_frame(
        gpu,
        residency,
        evaluations,
        store,
        cursor,
        requests,
        staging,
    )
    .0
}

/// How a test frame stages requested finest tiles without a copy.
#[derive(Debug, Clone, Copy)]
struct Staging {
    /// The staging deadline they may be evaluated until, if they may be at all.
    until: Option<Instant>,
    /// A delay added to each evaluated while staging.
    slow: Duration,
}

/// [`frame`], staging as `staging` says, as the app does: returns how many finest tiles were
/// read through the store's evaluation while staging, and when staging ended.
fn hybrid_frame(
    gpu: &Gpu,
    residency: &mut Residency,
    evaluations: &mut FinestEvaluations,
    store: &mut TileStore,
    cursor: &StaleCursor,
    requests: &Requests,
    staging: Staging,
) -> (usize, Instant) {
    residency.begin_frame(gpu).expect("begins");
    let drained = store.drain_stale(cursor);
    evaluations.outdate(&drained);
    residency.apply(drained);
    residency.want(requests);
    evaluations
        .request(store, residency)
        .expect("queues the requests");
    let deadline = staging
        .until
        .unwrap_or_else(|| Instant::now() + Duration::from_secs(60));
    let mut source = StoreSource::with_evaluations(store, evaluations);
    if let Some(until) = staging.until {
        source = source.evaluating_until(until);
    }
    source.slow = staging.slow;
    let uploads = residency.stage(gpu, &mut source, deadline).expect("staged");
    let ended = Instant::now();
    let evaluated_here = source.evaluated_here();
    gpu.scoped("frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("submits");
    // The GPU done with the frame, as it is by the next in the app: its staging buffers come
    // back to the ring instead of new ones being made.
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finishes");
    (evaluated_here, ended)
}

/// `key`'s positions as the store evaluates them now.
fn evaluated(store: &mut TileStore, key: TileKey) -> Vec<[f32; 3]> {
    let positions = store.evaluate(key).expect("evaluates");
    positions.iter().map(|&p| array(p)).collect()
}

/// `key`'s positions as its slot holds them.
fn shown(gpu: &Gpu, residency: &Residency, key: &TileKey) -> Vec<[f32; 3]> {
    let vertices = slot_contents(gpu, residency, key);
    vertices.iter().map(|v| v.position).collect()
}

/// Lifts the correction of finest vertex `(90, 20)` of face 2, inside [`requested`]'s tile, by
/// `lift`, as one operation, its epoch closed.
fn edit(store: &mut TileStore, lift: f32) {
    let face = FaceId::new(2);
    let (i, j) = (90, 20);
    let (key, li, lj) = TileKey::containing(
        LayerId::BASE,
        Channel::DISPLACEMENT,
        face,
        level(FINEST),
        i,
        j,
    )
    .expect("a finest tile");
    assert_eq!(key, requested().0, "the edit is in the requested tile");
    store.begin_op().expect("an op");
    let c = store
        .correction(key)
        .expect("reads")
        .get(li, lj)
        .expect("a vertex");
    store
        .set_correction(face, level(FINEST), i, j, c + Vec3::new(0.0, 0.0, lift))
        .expect("written");
    store.close_epoch().expect("closed");
}

#[test]
fn a_requested_finest_tile_is_queued_evaluated_by_the_jobs_and_staged_from_its_copy() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("finest-queued");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let (key, requests) = requested();

        // Heard: queued, not staged (it waits a frame, as every finest tile does).
        let here = frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        assert_eq!(here, 0, "staging evaluated nothing");
        assert_eq!(store.queued_evaluations(), 1, "queued when requested");
        assert!(residency.slot(&key).is_none(), "not staged yet");
        // With no jobs run, staging never evaluates it: it stays queued and wanted.
        for _ in 0..3 {
            let here_now = frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
            );
            assert_eq!(here_now, 0, "staging evaluated nothing");
            assert_eq!(store.queued_evaluations(), 1, "still queued");
            assert!(residency.slot(&key).is_none(), "deferred, not read");
            assert_eq!(residency.stats().wanted, 1, "and still wanted");
        }
        // The jobs evaluate it; the next frame stages it from the copy.
        let report = store
            .run_jobs(Instant::now() + Duration::from_secs(10))
            .expect("the jobs run");
        assert_eq!(report.queue_finished, 1, "the jobs evaluated it");
        let from_copy = frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        assert_eq!(
            from_copy, 0,
            "staged from its copy, nothing evaluated while staging"
        );
        assert!(residency.slot(&key).is_some(), "staged");
        assert_eq!(residency.stats().wanted, 0, "no longer wanted");
        assert_eq!(shown(gpu, &residency, &key), evaluated(store, key));
        // Staged and no longer wanted, its copy goes.
        let none = Requests::new([], 0);
        frame(gpu, &mut residency, &mut evaluations, store, &cursor, &none);
        assert_eq!(evaluations.held(), 0, "the copy let go");
        drop(project.close(false));
    });
}

/// A write reaching a tile after it was queued never reaches the GPU stale.
///
/// Not when it lands before the job finishes the tile, nor after it finished and before it is
/// taken, nor after it is taken and before it is staged.
#[test]
fn a_tile_edited_after_it_was_queued_is_never_staged_stale() {
    GpuTest::run(|gpu| {
        // When the write lands: 0 before the job runs; 1 after it finished; 2 after the copy
        // is taken.
        for when in 0..3 {
            let dir = TempDir::new(&format!("finest-edited-{when}"));
            let (mut project, cursor) = small_project(&dir);
            let store = project.store();
            let mesh = store.mesh().clone();
            let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
            let mut evaluations = FinestEvaluations::new();
            let (key, requests) = requested();
            frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
            );
            let before = evaluated(store, key);
            if when > 0 {
                store
                    .run_jobs(Instant::now() + Duration::from_secs(10))
                    .expect("the jobs run");
            }
            if when > 1 {
                // As the next frame does before it stages: the copy is taken and held.
                evaluations
                    .request(store, &residency)
                    .expect("queues the requests");
                assert_eq!(evaluations.held(), 1, "{when}: the copy is held");
            }
            edit(store, 0.01);
            let after = evaluated(store, key);
            assert_ne!(before, after, "the edit moves the tile");
            // The next frame has no current copy to stage: the store withholds one it finished
            // (1), and the drain naming the tile drops one taken (2).
            let here = frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
            );
            assert_eq!(here, 0, "{when}: nothing evaluated while staging");
            assert!(
                residency.slot(&key).is_none(),
                "{when}: no stale copy staged"
            );
            assert_eq!(evaluations.held(), 0, "{when}: no copy held");
            let began = Instant::now();
            while residency.slot(&key).is_none() {
                assert!(began.elapsed() < Duration::from_secs(60), "{when}: staged");
                run_all_jobs(store);
                let later = frame(
                    gpu,
                    &mut residency,
                    &mut evaluations,
                    store,
                    &cursor,
                    &requests,
                );
                assert_eq!(later, 0, "{when}: staged only from a copy");
            }
            assert_eq!(
                shown(gpu, &residency, &key),
                after,
                "{when}: the edited tile, never the stale copy"
            );
            drop(project.close(false));
        }
    });
}

/// A copy held is dropped by the drain that names its tile, and by a resync.
#[test]
fn a_drain_naming_a_held_tile_drops_its_copy() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("finest-outdated");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let (key, requests) = requested();
        frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        run_all_jobs(store);
        evaluations
            .request(store, &residency)
            .expect("queues the requests");
        assert!(evaluations.get(&key).is_some(), "held");
        evaluations.outdate(&Drained::Epochs(Vec::new()));
        assert!(evaluations.get(&key).is_some(), "no epoch: still held");
        evaluations.outdate(&Drained::Resync);
        assert_eq!(evaluations.held(), 0, "a resync drops every copy");
        drop(project.close(false));
    });
}

#[test]
fn a_request_withdrawn_drops_its_queued_work_and_its_copy() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("finest-withdrawn");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let (key, requests) = requested();
        let none = Requests::new([], 0);

        // Withdrawn before it is evaluated: its queued work goes.
        frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        assert_eq!(store.queued_evaluations(), 1, "queued");
        frame(gpu, &mut residency, &mut evaluations, store, &cursor, &none);
        assert_eq!(store.queued_evaluations(), 0, "withdrawn: no longer queued");
        let report = store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .expect("the jobs run");
        assert_eq!(report.queue_finished, 0, "nothing evaluated for it");

        // Withdrawn once evaluated: its copy, taken or not, goes.
        frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        run_all_jobs(store);
        frame(gpu, &mut residency, &mut evaluations, store, &cursor, &none);
        assert_eq!(evaluations.held(), 0, "no copy held");
        assert!(store.take_evaluated().is_empty(), "nor left in the store");
        assert!(residency.slot(&key).is_none(), "and nothing staged");
        drop(project.close(false));
    });
}

/// Requests for every finest tile of `store`'s mesh.
fn every_finest(store: &TileStore) -> Requests {
    let mesh = store.mesh();
    let keys = mesh.faces().flat_map(|face| {
        (0..4_u16)
            .filter_map(move |n| TileKey::displacement(face, level(FINEST), n % 2, n / 2).ok())
    });
    Requests::new(
        keys.map(|key| Request {
            key,
            screen_error: 2.0,
        }),
        0,
    )
}

/// A staging deadline that stages copies but leaves no time to evaluate a tile: less than
/// [`STAGE_MARGIN`](crate::upload::STAGE_MARGIN) and [`PROBE_COST`](crate::upload::PROBE_COST)
/// together.
const NO_SLACK: Duration = Duration::from_micros(1500);

/// Staging that may evaluate until `slack` from now, each tile `slow` slower.
fn slack(slack: Duration, slow: Duration) -> Staging {
    Staging {
        until: Some(Instant::now() + slack),
        slow,
    }
}

/// With time in the frame, a requested tile the queue hasn't made is evaluated while staging
/// in the frame it is due, and withdrawn from the queue: the jobs never evaluate it again.
#[test]
fn with_slack_a_due_tile_is_staged_in_its_frame_while_staging() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("hybrid-slack");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let (key, requests) = requested();
        let roomy = || slack(Duration::from_secs(1), Duration::ZERO);
        let (here, _) = hybrid_frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
            roomy(),
        );
        assert_eq!(here, 0, "not due in the frame it is requested in");
        assert_eq!(store.queued_evaluations(), 1, "queued meanwhile");
        let (due, _) = hybrid_frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
            roomy(),
        );
        assert_eq!(due, 1, "evaluated while staging, the frame it is due");
        assert!(residency.slot(&key).is_some(), "staged");
        assert_eq!(shown(gpu, &residency, &key), evaluated(store, key));
        assert_eq!(store.queued_evaluations(), 0, "withdrawn from the queue");
        let report = store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .expect("the jobs run");
        assert_eq!(report.queue_finished, 0, "never evaluated again");
        assert!(evaluations.cost().is_some(), "its cost measured");
        drop(project.close(false));
    });
}

/// With no time left in the frame, a due tile waits for the queue, and is staged from its copy.
#[test]
fn without_slack_a_due_tile_is_staged_from_the_queue() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("hybrid-no-slack");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let (key, requests) = requested();
        let tight = || slack(NO_SLACK, Duration::ZERO);
        for _ in 0..3 {
            let (here, _) = hybrid_frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
                tight(),
            );
            assert_eq!(here, 0, "no time: nothing evaluated while staging");
            assert!(residency.slot(&key).is_none(), "deferred");
        }
        run_all_jobs(store);
        let (here, _) = hybrid_frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
            tight(),
        );
        assert_eq!(here, 0, "staged from its copy");
        assert!(residency.slot(&key).is_some(), "staged");
        assert_eq!(shown(gpu, &residency, &key), evaluated(store, key));
        drop(project.close(false));
    });
}

/// Some tiles evaluated while staging, the rest left to the queue and staged from its copies:
/// every tile is evaluated once, by one or the other, and staged.
#[test]
fn no_tile_is_evaluated_twice_between_staging_and_the_queue() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("hybrid-once");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let requests = every_finest(store);
        let count = requests.iter().count();
        // Heard: queued, not due.
        frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        let mut by_queue = 0;
        // Due, with time for a few while staging (each made slow), the rest left over.
        let (mut by_staging, _) = hybrid_frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
            slack(Duration::from_millis(15), Duration::from_millis(3)),
        );
        assert!(by_staging > 0, "some evaluated while staging");
        let began = Instant::now();
        while requests.iter().any(|r| residency.slot(&r.key).is_none()) {
            assert!(
                began.elapsed() < Duration::from_secs(60),
                "every tile staged"
            );
            by_queue += store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the jobs run")
                .queue_finished;
            let (here, _) = hybrid_frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
                slack(NO_SLACK, Duration::ZERO),
            );
            by_staging += here;
        }
        assert!(by_queue > 0, "some made by the queue");
        assert_eq!(
            by_queue + by_staging,
            count,
            "each tile evaluated once: {by_queue} by the queue, {by_staging} while staging"
        );
        drop(project.close(false));
    });
}

/// Evaluating while staging keeps to the staging deadline however slow the evaluation: the
/// batches are sized from the cost measured, the first tile alone.
#[test]
fn staging_keeps_its_deadline_under_a_slow_evaluation() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("hybrid-slow");
        let (mut project, cursor) = small_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        let requests = every_finest(store);
        frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        let mut staged_here = 0;
        for _ in 0..4 {
            let staging = slack(Duration::from_millis(12), Duration::from_millis(4));
            let until = staging.until.expect("a deadline");
            let (here, ended) = hybrid_frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
                staging,
            );
            assert!(
                ended <= until,
                "staging ended {:?} past its deadline, {here} tiles evaluated",
                ended.saturating_duration_since(until)
            );
            staged_here += here;
        }
        assert!(staged_here > 0, "some evaluated while staging");
        drop(project.close(false));
    });
}

/// The queued evaluation keeps to the jobs' turns.
///
/// With a display rebuild queued beside it, each call of the jobs ends near its deadline, and
/// both move on together; staging between calls evaluates nothing.
#[test]
fn the_queued_evaluation_keeps_to_the_jobs_share() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("finest-share");
        let (mut project, cursor) = unbuilt_project(&dir);
        let store = project.store();
        let mesh = store.mesh().clone();
        let mut residency = with_floor(gpu, &mesh, level(FINEST), store);
        let mut evaluations = FinestEvaluations::new();
        // Every finest tile, and the whole display layer to build.
        let finest = mesh.faces().flat_map(|face| {
            (0..4_u16)
                .filter_map(move |n| TileKey::displacement(face, level(FINEST), n % 2, n / 2).ok())
        });
        let requests = Requests::new(
            finest.map(|key| Request {
                key,
                screen_error: 2.0,
            }),
            0,
        );
        let count = requests.iter().count();
        store.queue_display_rebuild(level(FINEST));
        let budget = Duration::from_millis(2);
        let (mut together, mut calls) = (0, Vec::new());
        let began = Instant::now();
        loop {
            assert!(began.elapsed() < Duration::from_secs(60), "the jobs end");
            let here = frame(
                gpu,
                &mut residency,
                &mut evaluations,
                store,
                &cursor,
                &requests,
            );
            assert_eq!(here, 0, "staging evaluated nothing");
            let staged = requests
                .iter()
                .filter(|r| residency.slot(&r.key).is_some())
                .count();
            if staged == count && !store.has_jobs() {
                break;
            }
            let start = Instant::now();
            let report = store.run_jobs(start + budget).expect("the jobs run");
            let spent = start.elapsed();
            assert!(
                spent <= budget + report.longest_slice + Duration::from_millis(50),
                "{spent:?} for {budget:?}, longest slice {:?}",
                report.longest_slice
            );
            // The rebuild's progress: tiles built, or the finest tiles it reads evaluated.
            let display = report.built + report.evaluated;
            if display > 0 && report.queue_finished > 0 {
                together += 1;
            }
            calls.push((display, report.queue_finished));
        }
        assert!(
            together > 0,
            "the display jobs and the queue took turns within a call: (display, queue) by call {calls:?}"
        );
        drop(project.close(false));
    });
}

/// A project of `mesh` with a correction in every correction tile, saved; its path and
/// configuration.
///
/// Its display layer is kept under `finest`; a correction is written at the middle of every
/// correction tile at every level from 1 (so a finest tile's cone has something to read at
/// each), and it is saved as `name` in `dir`.
fn saved_edited(
    dir: &TempDir,
    name: &str,
    mesh: &geom::BaseMesh,
    finest: Level,
) -> (std::path::PathBuf, ProjectConfig) {
    let config = ProjectConfig {
        store: StoreConfig::new(1 << 30, 1).with_display(finest),
        recovery_dir: dir.0.clone(),
    };
    let mut project = Project::new(mesh.clone(), config.clone()).expect("a project");
    let store = project.store();
    store.begin_op().expect("an op");
    let mut n = 0_u32;
    for face in mesh.faces() {
        for l in 1..=finest.get() {
            let at = level(l);
            let q = at.tile_quads();
            let per_side = at.tiles_per_side();
            for ty in 0..per_side {
                for tx in 0..per_side {
                    n += 1;
                    let lift = 1e-3 * f32::from(u16::try_from(n % 7).expect("small"));
                    let (i, j) = (tx * q + q / 2, ty * q + q / 2);
                    store
                        .set_correction(face, at, i, j, Vec3::new(lift, -lift, lift))
                        .expect("written");
                }
            }
        }
    }
    store.queue_display_rebuild(finest);
    run_all_jobs(store);
    let path = dir.0.join(name);
    drop(project.save_as(&path).expect("saved"));
    drop(project.close(false));
    (path, config)
}

/// The finest tiles of `mesh`'s first `faces` faces at `finest`.
fn finest_tiles(mesh: &geom::BaseMesh, finest: Level, faces: u32) -> Vec<TileKey> {
    let per_side = u16::try_from(finest.tiles_per_side()).expect("small");
    let mut keys = Vec::new();
    for face in mesh.faces().take(usize::try_from(faces).expect("small")) {
        for ty in 0..per_side {
            for tx in 0..per_side {
                keys.push(TileKey::displacement(face, finest, tx, ty).expect("a tile"));
            }
        }
    }
    keys
}

/// Stages `keys`, each finest tile evaluated while staging, as before the queued evaluation.
///
/// They are requested at once and paged in on request, from a store opened afresh (its cones
/// cold). Returns the owner's staging time and the tiles staged.
fn staged_evaluating_while_staging(
    gpu: &Gpu,
    path: &std::path::Path,
    config: &ProjectConfig,
    keys: &[TileKey],
) -> (Duration, u64) {
    let mut project = Project::open(path, config.clone()).expect("opened");
    let store = project.store();
    let mesh = store.mesh().clone();
    let finest = store.display_finest().expect("a display layer");
    let mut residency = with_floor(gpu, &mesh, finest, store);
    let requests = super::requests_of(keys);
    prefetch_requests(store, &requests).expect("paged in");
    let (mut took, mut staged) = (Duration::ZERO, 0);
    for _ in 0..2 {
        // A frame's interval for the page-in to land, as in the app.
        std::thread::sleep(Duration::from_micros(16_700));
        took += super::frame(gpu, &mut residency, store, &requests);
        staged += residency.stats().staged_tiles;
    }
    assert_eq!(staged, keys.len() as u64, "every tile staged");
    drop(project.close(false));
    (took, staged)
}

/// [`staged_evaluating_while_staging`] through the queued evaluation.
///
/// The jobs are given `jobs` after each frame's staging, frames until every tile is staged.
/// Returns the owner's staging time, the tiles staged, and the jobs' time.
fn staged_from_the_queue(
    gpu: &Gpu,
    path: &std::path::Path,
    config: &ProjectConfig,
    keys: &[TileKey],
    jobs: Duration,
) -> ((Duration, u64), Duration) {
    let mut project = Project::open(path, config.clone()).expect("opened");
    let store = project.store();
    let cursor = store.subscribe_stale();
    let mesh = store.mesh().clone();
    let finest = store.display_finest().expect("a display layer");
    let mut residency = with_floor(gpu, &mesh, finest, store);
    let mut evaluations = FinestEvaluations::new();
    let requests = super::requests_of(keys);
    prefetch_requests(store, &requests).expect("paged in");
    let (mut took, mut staged, mut in_jobs) = (Duration::ZERO, 0, Duration::ZERO);
    let mut frames = 0;
    while staged < keys.len() as u64 {
        frames += 1;
        assert!(frames < 1000, "every tile staged");
        let began = Instant::now();
        let here = frame(
            gpu,
            &mut residency,
            &mut evaluations,
            store,
            &cursor,
            &requests,
        );
        took += began.elapsed();
        assert_eq!(here, 0, "staging evaluated nothing");
        staged += residency.stats().staged_tiles;
        let jobs_began = Instant::now();
        store.run_jobs(jobs_began + jobs).expect("the jobs run");
        in_jobs += jobs_began.elapsed();
    }
    eprintln!(
        "  through the queue: {staged} tiles staged over {frames} frames, the jobs given {:.1} ms \
         a frame",
        jobs.as_secs_f64() * 1e3
    );
    drop(project.close(false));
    ((took, staged), in_jobs)
}

/// How a paced run stages requested finest tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Every one evaluated while staging, as before the queued evaluation.
    Staging,
    /// Every one through the queue, staged from its copy.
    Queue,
    /// Evaluated while staging while the frame has the time, else through the queue.
    Hybrid,
}

/// What a paced run came to.
#[derive(Debug, Clone, Copy, Default)]
struct Paced {
    /// Frames until every tile was staged.
    frames: u32,
    /// The owner's staging time, over every frame.
    staging: Duration,
    /// The most a frame's staging ran past its deadline.
    worst_over: Duration,
    /// Tiles evaluated while staging.
    here: usize,
    /// Tiles staged.
    staged: u64,
}

/// A frame's interval, at 60 Hz.
const INTERVAL: Duration = Duration::from_micros(16_700);

/// Where a frame's staging deadline falls, from its start (the app's, as the investigation
/// measured it).
const STAGING: Duration = Duration::from_micros(6_250);

/// How long a frame's jobs phase runs after staging (about the idle time a frame has during
/// a catch-up).
const JOBS: Duration = Duration::from_millis(8);

/// Stages `keys` in frames paced as the app's, as `mode` says.
///
/// They are requested at once and paged in on request, from a store opened afresh (its cones
/// cold). Each frame stages until [`STAGING`] into the frame, then runs the jobs for [`JOBS`]
/// (but in [`Mode::Staging`]), then idles out the rest of the [`INTERVAL`].
fn paced(
    gpu: &Gpu,
    path: &std::path::Path,
    config: &ProjectConfig,
    keys: &[TileKey],
    mode: Mode,
) -> Paced {
    let mut project = Project::open(path, config.clone()).expect("opened");
    let store = project.store();
    let cursor = store.subscribe_stale();
    let mesh = store.mesh().clone();
    let finest = store.display_finest().expect("a display layer");
    let mut residency = with_floor(gpu, &mesh, finest, store);
    let mut evaluations = FinestEvaluations::new();
    let requests = super::requests_of(keys);
    prefetch_requests(store, &requests).expect("paged in");
    let mut run = Paced::default();
    while run.staged < keys.len() as u64 {
        run.frames += 1;
        assert!(run.frames < 1000, "every tile staged");
        let began = Instant::now();
        let deadline = began + STAGING;
        residency.begin_frame(gpu).expect("begins");
        let drained = store.drain_stale(&cursor);
        evaluations.outdate(&drained);
        residency.apply(drained);
        residency.want(&requests);
        let uploads = if mode == Mode::Staging {
            let mut source = StoreSource::new(store);
            let uploads = residency.stage(gpu, &mut source, deadline);
            run.here += source.evaluated_here();
            uploads
        } else {
            evaluations
                .request(store, &residency)
                .expect("queues the requests");
            let mut source = StoreSource::with_evaluations(store, &mut evaluations);
            if mode == Mode::Hybrid {
                source = source.evaluating_until(deadline);
            }
            let uploads = residency.stage(gpu, &mut source, deadline);
            run.here += source.evaluated_here();
            uploads
        }
        .expect("staged");
        let ended = Instant::now();
        run.staging += ended - began;
        run.worst_over = run
            .worst_over
            .max(ended.saturating_duration_since(deadline));
        run.staged += residency.stats().staged_tiles;
        gpu.scoped("frame", || {
            let encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            uploads.submit(gpu.queue(), encoder, |_| {});
        })
        .expect("submits");
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finishes");
        if mode != Mode::Staging {
            store
                .run_jobs(Instant::now().max(deadline) + JOBS)
                .expect("the jobs run");
        }
        std::thread::sleep((began + INTERVAL).saturating_duration_since(Instant::now()));
    }
    drop(project.close(false));
    run
}

/// Prints the paced runs `runs` of `what`: the one with the fewest frames, then the least
/// staging time, and the worst overrun of any.
fn report_paced(what: &str, runs: &[Paced]) {
    let best = runs
        .iter()
        .min_by_key(|run| (run.frames, run.staging))
        .copied()
        .expect("a run");
    let worst = runs
        .iter()
        .map(|run| run.worst_over)
        .max()
        .unwrap_or_default();
    let per_tile =
        best.staging.as_secs_f64() * 1e6 / f64::from(u32::try_from(best.staged).expect("small"));
    eprintln!(
        "{what}: {} tiles staged by frame {}, staging {:.1} ms in all ({per_tile:.1} µs a tile), \
         {} evaluated while staging; worst staging overrun {:.2} ms (of {} runs)",
        best.staged,
        best.frames,
        best.staging.as_secs_f64() * 1e3,
        best.here,
        worst.as_secs_f64() * 1e3,
        runs.len()
    );
}

/// The owner thread's time staging requested finest tiles from a cold store.
///
/// Evaluated while staging (as before the queued evaluation) against evaluated in the jobs and
/// staged from their copies, then the three ways in frames paced as the app's (staging
/// deadline, jobs phase, interval). A timing, not a check: run it in release, with
/// `--run-ignored only`.
#[test]
#[ignore = "a timing, for release builds"]
fn owner_time_staging_finest_tiles() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("finest-owner-time");
        let mesh = cube_sphere(3).expect("a mesh");
        let finest = level(8);
        let (path, config) = saved_edited(&dir, "finest.bcp", &mesh, finest);
        let keys = finest_tiles(&mesh, finest, 4);
        let jobs = Duration::from_millis(8);
        let (mut before, mut after, mut in_jobs) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..REPEATS {
            before.push(staged_evaluating_while_staging(gpu, &path, &config, &keys));
            let (run, jobs_took) = staged_from_the_queue(gpu, &path, &config, &keys, jobs);
            after.push(run);
            in_jobs.push((jobs_took, run.1));
        }
        report("finest tiles evaluated while staging (before)", &before);
        report(
            "finest tiles staged from the queue's copies (after)",
            &after,
        );
        report(
            "finest tiles evaluated in the jobs phase, after the hand-off",
            &in_jobs,
        );
        // The 64 tiles, and 256 at once: more than a jobs phase evaluates.
        let many = finest_tiles(&mesh, finest, 16);
        for tiles in [&keys, &many] {
            for (mode, what) in [
                (Mode::Staging, "paced, evaluated while staging (main)"),
                (Mode::Queue, "paced, through the queue"),
                (Mode::Hybrid, "paced, hybrid"),
            ] {
                let runs: Vec<Paced> = (0..REPEATS)
                    .map(|_| paced(gpu, &path, &config, tiles, mode))
                    .collect();
                report_paced(what, &runs);
            }
        }
    });
}
