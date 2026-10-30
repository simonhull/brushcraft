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

/// The owner thread's time staging requested finest tiles from a cold store.
///
/// Evaluated while staging (as before the queued evaluation) against evaluated in the jobs and
/// staged from their copies. A timing, not a check: run it in release, with
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
    });
}
