//! Requested finest tiles through the store's queued evaluation: queued when requested,
//! evaluated only by the jobs, staged from their copies, never stale.

use std::time::{Duration, Instant};

use geom::{Channel, FaceId, LayerId, TileKey, Vec3, cube_sphere};
use store::{Drained, Project, ProjectConfig, StaleCursor, StoreConfig, TileStore};
use wgpu::CommandEncoderDescriptor;

use super::{TempDir, array, level, slot_contents, with_floor};
use crate::gpu::{Gpu, GpuTest};
use crate::residency::{Request, Requests, Residency};
use crate::upload::{FinestEvaluations, StoreSource};

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

/// One frame of the owner thread before the hand-off, as the app runs it.
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
    residency.begin_frame(gpu).expect("begins");
    let drained = store.drain_stale(cursor);
    evaluations.outdate(&drained);
    residency.apply(drained);
    residency.want(requests);
    evaluations
        .request(store, residency)
        .expect("queues the requests");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut source = StoreSource::with_evaluations(store, evaluations);
    let uploads = residency.stage(gpu, &mut source, deadline).expect("staged");
    let evaluated_here = source.evaluated_here();
    gpu.scoped("frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("submits");
    evaluated_here
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
