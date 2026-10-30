//! Uploads from a real store: what lands in a slot is what the store gives.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use geom::{FaceId, FinestNormalCone, Level, TileKey, cube_sphere};
use store::{Project, ProjectConfig, StoreConfig};
use wgpu::CommandEncoderDescriptor;

use super::*;
use crate::gpu::GpuTest;
use crate::residency::{Request, Requests, Residency, ResidencyConfig, SizeClass};

/// A fresh recovery directory under the system temp dir, removed with everything in it on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// Names a directory for this process and `name` (the project creates it).
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-render-{}-{name}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Level `l`.
fn level(l: u8) -> Level {
    Level::new(l).expect("a level")
}

/// `key`'s vertices as the GPU holds them.
fn slot_contents(gpu: &Gpu, residency: &Residency, key: &TileKey) -> Vec<GpuVertex> {
    let slot = residency
        .slot(key)
        .unwrap_or_else(|| panic!("{key:?} resident"));
    let class = slot.class();
    let buffer = &residency.slot_buffers(class)[usize::from(slot.buffer())];
    let all: Vec<GpuVertex> = gpu.read_buffer(buffer).expect("reads back");
    let n = usize::try_from(class.vertices()).expect("small");
    let first = usize::try_from(slot.index()).expect("small") * n;
    all[first..first + n].to_vec()
}

/// `v` as an array.
const fn array(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

#[test]
fn the_stores_tiles_land_in_their_slots_their_positions_bit_for_bit() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("store-source");
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(7);
        let config = ProjectConfig {
            store: StoreConfig::new(256 << 20, 1).with_display(finest),
            recovery_dir: dir.0.clone(),
        };
        let mut project = Project::new(mesh.clone(), config).expect("a project");
        let store = project.store();
        store.queue_display_rebuild(finest);
        while store.has_jobs() {
            store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the display layer builds");
        }
        let mut residency =
            Residency::new(gpu, &mesh, finest, ResidencyConfig::default()).expect("residency");
        let face = FaceId::new(2);
        let fine = TileKey::displacement(face, finest, 1, 0).expect("a finest tile");
        let coarse = TileKey::display(face, level(6), 0, 0).expect("a display tile");
        let floor = TileKey::display(face, level(3), 0, 0).expect("a floor tile");
        let requests = Requests::new(
            [fine, coarse].map(|key| Request {
                key,
                screen_error: 2.0,
            }),
            0,
        );
        assert_eq!(
            prefetch_requests(store, &requests).ok(),
            Some(0),
            "nothing to page in"
        );
        // Requests are staged from the frame after they're made: the floor arrives in the
        // first, the requests in the second.
        for _ in 0..2 {
            residency.begin_frame(gpu).expect("begins");
            residency.want(&requests);
            let deadline = Instant::now() + Duration::from_secs(60);
            let uploads = residency
                .stage(gpu, &mut StoreSource::new(store), deadline)
                .expect("staged");
            gpu.scoped("frame", || {
                let encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                uploads.submit(gpu.queue(), encoder, |_| {});
            })
            .expect("submits");
        }
        assert!(residency.floor_ready(), "the floor arrived");

        let positions: Vec<[f32; 3]> = store
            .evaluate(fine)
            .expect("evaluates")
            .iter()
            .map(|&p| array(p))
            .collect();
        let normals: Vec<[f32; 3]> = store
            .normals(fine)
            .expect("normals")
            .iter()
            .map(|&n| array(n))
            .collect();
        let held = slot_contents(gpu, &residency, &fine);
        let held_positions: Vec<[f32; 3]> = held.iter().map(|v| v.position).collect();
        assert_eq!(held_positions, positions, "the finest tile's positions");
        // Its normals are the GPU's, from its ring: the store's formula, its rounding the GPU's
        // (`normals::tests` checks the seams), within the same angle.
        for (vertex, &[x, y, z]) in held.iter().zip(&normals) {
            let off = crate::normals::tests::angle(vertex.normal, Vec3::new(x, y, z));
            assert!(
                off <= crate::normals::tests::MAX_ANGLE,
                "its normals, computed on the GPU: {off:e} rad off the store's"
            );
        }

        for key in [coarse, floor] {
            let tile = store
                .display_for_draw(&[key])
                .expect("reads")
                .pop()
                .flatten()
                .unwrap_or_else(|| panic!("{key:?} built"));
            let shown = slot_contents(gpu, &residency, &key);
            let shown_positions: Vec<[f32; 3]> = shown.iter().map(|v| v.position).collect();
            let stored: Vec<[f32; 3]> = tile.positions().iter().map(|&p| array(p)).collect();
            assert_eq!(shown_positions, stored, "{key:?}'s positions");
            let roughness = shown.iter().zip(tile.roughness());
            for (vertex, &r) in roughness {
                assert_eq!(
                    vertex.roughness,
                    f32::from(r) / 255.0,
                    "{key:?}'s roughness"
                );
            }
        }
        drop(project.close(false));
    });
}

/// Stages one frame of `residency` from `store`, wanting `requests`, and submits its uploads.
///
/// Returns the owner thread's time: the frame's beginning, its requests and its staging, not
/// the submission (the render thread's).
fn frame(
    gpu: &Gpu,
    residency: &mut Residency,
    store: &mut TileStore,
    requests: &Requests,
) -> Duration {
    let began = Instant::now();
    residency.begin_frame(gpu).expect("begins");
    residency.want(requests);
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency
        .stage(gpu, &mut StoreSource::new(store), deadline)
        .expect("staged");
    let took = began.elapsed();
    gpu.scoped("frame", || {
        let encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        uploads.submit(gpu.queue(), encoder, |_| {});
    })
    .expect("submits");
    took
}

/// The `e` and cone resident tile `key`'s node holds.
fn node_detail(residency: &Residency, key: &TileKey) -> (f32, FinestNormalCone) {
    let tree = residency.tree();
    let index = tree
        .node(crate::tree::Place::of(key))
        .unwrap_or_else(|| panic!("{key:?} resident"));
    let node = tree.gpu_node(index);
    (node.error, FinestNormalCone::from_bits(node.finest_cone))
}

/// Runs frames, each after applying the feed's epochs, until `done` holds of `key`'s node (at
/// most a few).
fn frames_until(
    gpu: &Gpu,
    residency: &mut Residency,
    store: &mut TileStore,
    cursor: &store::StaleCursor,
    key: &TileKey,
    done: impl Fn((f32, FinestNormalCone)) -> bool,
) -> (f32, FinestNormalCone) {
    let none = Requests::new([], 0);
    for _ in 0..4 {
        residency.apply(store.drain_stale(cursor));
        frame(gpu, residency, store, &none);
        let detail = node_detail(residency, key);
        if done(detail) {
            return detail;
        }
    }
    panic!(
        "{key:?}'s node never changed: {:?}",
        node_detail(residency, key)
    );
}

#[test]
fn a_dab_makes_a_resident_display_tile_cover_everything_until_its_refresh() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("dab-cone");
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(7);
        let config = ProjectConfig {
            store: StoreConfig::new(256 << 20, 1).with_display(finest),
            recovery_dir: dir.0.clone(),
        };
        let mut project = Project::new(mesh.clone(), config).expect("a project");
        let store = project.store();
        store.queue_display_rebuild(finest);
        while store.has_jobs() {
            store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the display layer builds");
        }
        let cursor = store.subscribe_stale();
        let mut residency =
            Residency::new(gpu, &mesh, finest, ResidencyConfig::default()).expect("residency");
        let face = FaceId::new(2);
        // A whole-face display tile above the floor: every dab on the face reaches it.
        let coarse = TileKey::display(face, level(6), 0, 0).expect("a display tile");
        let requests = Requests::new(
            [Request {
                key: coarse,
                screen_error: 2.0,
            }],
            0,
        );
        for _ in 0..2 {
            frame(gpu, &mut residency, store, &requests);
        }
        let built = store
            .display_for_draw(&[coarse])
            .expect("reads")
            .pop()
            .flatten()
            .expect("built");
        let (e, cone) = node_detail(&residency, &coarse);
        assert_eq!((e, cone), (built.error(), built.cone()), "the record's");
        assert!(!cone.covers_everything(), "a tight cone to begin with");

        // A dab: a small bump on the face, its display tiles' bounds raised.
        let lift = 0.002_f32;
        let (key, li, lj) = TileKey::containing(
            geom::LayerId::BASE,
            Channel::DISPLACEMENT,
            face,
            finest,
            40,
            50,
        )
        .expect("a finest tile");
        store.begin_op().expect("an op");
        let c = store
            .correction(key)
            .expect("reads")
            .get(li, lj)
            .expect("a vertex");
        store
            .set_correction(face, finest, 40, 50, c + Vec3::new(0.0, 0.0, lift))
            .expect("written");
        store.bump_display_error(&[key], lift).expect("bumped");
        store.close_epoch().expect("closed");
        let (bumped_e, _) = frames_until(gpu, &mut residency, store, &cursor, &coarse, |d| {
            d.1.covers_everything()
        });
        assert!(bumped_e > e, "its bound raised: {bumped_e} over {e}");

        // The refresh's epochs bring its rebuilt record, a tight cone again. The first call
        // queues it (the dab's writes settle there), so it runs at least once.
        loop {
            store
                .run_jobs(Instant::now() + Duration::from_millis(50))
                .expect("the refresh runs");
            if !store.has_jobs() {
                break;
            }
        }
        let rebuilt = store
            .display_for_draw(&[coarse])
            .expect("reads")
            .pop()
            .flatten()
            .expect("rebuilt");
        assert!(
            !rebuilt.cone().covers_everything(),
            "the refresh's cone is tight: e {} against {bumped_e}, jobs {}",
            rebuilt.error(),
            store.has_jobs()
        );
        let record = (rebuilt.error(), rebuilt.cone());
        frames_until(gpu, &mut residency, store, &cursor, &coarse, |d| {
            d == record
        });
        drop(project.close(false));
    });
}

/// Builds a project over `mesh` keeping its display layer under `finest` and saves it as `name`
/// in `dir`; returns its path and configuration, to open it with.
fn saved(
    dir: &TempDir,
    name: &str,
    mesh: &geom::BaseMesh,
    finest: Level,
) -> (PathBuf, ProjectConfig) {
    let config = ProjectConfig {
        store: StoreConfig::new(1 << 30, 1).with_display(finest),
        recovery_dir: dir.0.clone(),
    };
    let mut project = Project::new(mesh.clone(), config.clone()).expect("a project");
    let store = project.store();
    store.queue_display_rebuild(finest);
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .expect("the display layer builds");
    }
    let path = dir.0.join(name);
    drop(project.save_as(&path).expect("saved"));
    drop(project.close(false));
    (path, config)
}

/// Every display tile of `mesh` at `levels`.
fn display_tiles(mesh: &geom::BaseMesh, levels: &[u8]) -> Vec<TileKey> {
    let mut keys = Vec::new();
    for face in mesh.faces() {
        for &l in levels {
            let per_side = u16::try_from(level(l).tiles_per_side()).expect("small");
            for ty in 0..per_side {
                for tx in 0..per_side {
                    keys.push(TileKey::display(face, level(l), tx, ty).expect("a tile"));
                }
            }
        }
    }
    keys
}

/// Requests for `keys`, of one screen error.
fn requests_of(keys: &[TileKey]) -> Requests {
    Requests::new(
        keys.iter().map(|&key| Request {
            key,
            screen_error: 2.0,
        }),
        0,
    )
}

/// How many times each timing is taken: the least is printed.
const REPEATS: usize = 3;

/// Prints `what`'s least time of `runs`, each the time and the tiles it staged.
fn report(what: &str, runs: &[(Duration, u64)]) {
    let (took, tiles) = runs
        .iter()
        .min_by_key(|(took, _)| *took)
        .copied()
        .expect("a run");
    let per_tile = took.as_secs_f64() * 1e6 / f64::from(u32::try_from(tiles).expect("small"));
    eprintln!(
        "{what}: {tiles} tiles in {:.1} ms, {per_tile:.1} µs a tile, {:.2} tiles/ms (least of {})",
        took.as_secs_f64() * 1e3,
        1e3 / per_tile,
        runs.len()
    );
}

/// A display tile prepared but not staged (its frame ran out of room or slots) keeps its record
/// read ahead for the next frame; one staged lets it go.
#[test]
fn a_starved_tile_keeps_its_record_read_ahead_for_the_next_frame() {
    let dir = TempDir::new("starved");
    let mesh = cube_sphere(1).expect("a mesh");
    let finest = level(7);
    let (path, config) = saved(&dir, "starved.bcp", &mesh, finest);
    let mut project = Project::open(&path, config).expect("opened");
    let store = project.store();
    let face = FaceId::new(1);
    let staged = TileKey::display(face, level(6), 0, 0).expect("a display tile");
    let starved = TileKey::display(FaceId::new(2), level(6), 0, 0).expect("a display tile");
    let requests = requests_of(&[staged, starved]);
    assert_eq!(
        prefetch_requests(store, &requests).ok(),
        Some(2),
        "both read ahead"
    );
    let began = Instant::now();
    while store.display_needs_read(&staged) || store.display_needs_read(&starved) {
        assert!(began.elapsed() < Duration::from_secs(10), "read ahead");
        std::thread::yield_now();
    }
    let mut source = StoreSource::new(store);
    source.prepare(&[staged, starved]).expect("prepared");
    let mut data = TileData::default();
    data.reset(usize::try_from(SizeClass::of(level(6)).vertices()).expect("small"));
    let detail = source.read(staged, &mut data).expect("reads");
    assert!(detail.is_some(), "the staged tile has data");
    drop(source);
    assert!(
        store.display_needs_read(&staged),
        "staged: its record let go"
    );
    assert!(
        !store.display_needs_read(&starved),
        "starved: its record kept for the next frame"
    );
    drop(project.close(false));
}

/// Frame 1's owner time on a project of the benchmark project's 2,400 faces, opened afresh
/// each run: the floor, read on open, staged whole.
fn time_the_floor(gpu: &Gpu, dir: &TempDir) {
    let mesh = cube_sphere(20).expect("a mesh");
    let finest = level(6);
    let (path, config) = saved(dir, "floor.bcp", &mesh, finest);
    let mut runs = Vec::new();
    for _ in 0..REPEATS {
        let mut project = Project::open(&path, config.clone()).expect("opened");
        let store = project.store();
        let mut residency =
            Residency::new(gpu, &mesh, finest, ResidencyConfig::default()).expect("residency");
        let took = frame(gpu, &mut residency, store, &Requests::new([], 0));
        runs.push((took, residency.stats().staged_tiles));
        assert!(residency.floor_ready(), "the floor arrived");
        drop(project.close(false));
    }
    report("frame 1, the floor", &runs);
}

/// A residency over `mesh` with the floor staged from `store`.
fn with_floor(gpu: &Gpu, mesh: &geom::BaseMesh, finest: Level, store: &mut TileStore) -> Residency {
    let mut residency =
        Residency::new(gpu, mesh, finest, ResidencyConfig::default()).expect("residency");
    frame(gpu, &mut residency, store, &Requests::new([], 0));
    residency
}

/// Stages `keys` all requested at once, each record read on the owner thread as it is staged:
/// the owner time and the tiles staged.
fn staged_reading_on_the_owner(
    gpu: &Gpu,
    mesh: &geom::BaseMesh,
    finest: Level,
    store: &mut TileStore,
    keys: &[TileKey],
) -> (Duration, u64) {
    let mut residency = with_floor(gpu, mesh, finest, store);
    let requests = requests_of(keys);
    let (mut took, mut staged) = (Duration::ZERO, 0);
    for _ in 0..2 {
        took += frame(gpu, &mut residency, store, &requests);
        staged += residency.stats().staged_tiles;
    }
    assert_eq!(staged, keys.len() as u64, "every tile staged");
    (took, staged)
}

/// Stages `keys` requested a few at a time, fewer than the read-ahead holds, each record read on
/// the workers before the frame that stages it: the owner time and the tiles staged.
fn staged_read_ahead(
    gpu: &Gpu,
    mesh: &geom::BaseMesh,
    finest: Level,
    store: &mut TileStore,
    keys: &[TileKey],
) -> (Duration, u64) {
    let mut residency = with_floor(gpu, mesh, finest, store);
    let (mut took, mut staged) = (Duration::ZERO, 0);
    for chunk in keys.chunks(32) {
        let requests = requests_of(chunk);
        assert_eq!(
            prefetch_requests(store, &requests).ok(),
            Some(chunk.len()),
            "every one read ahead"
        );
        std::thread::sleep(Duration::from_millis(100));
        took += frame(gpu, &mut residency, store, &requests);
        staged += residency.stats().staged_tiles;
        assert!(
            chunk.iter().all(|key| residency.slot(key).is_some()),
            "staged in the frame they were requested in"
        );
    }
    (took, staged)
}

/// The owner thread's time staging display tiles read from the file.
///
/// Frame 1's floor (on a mesh of the benchmark project's 2,400 faces), and requested level-6
/// and level-7 tiles, read on the owner thread and read ahead. A timing, not a check: run it in
/// release, with `--run-ignored only`.
#[test]
#[ignore = "a timing, for release builds"]
fn owner_time_staging_display_tiles() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("owner-time");
        time_the_floor(gpu, &dir);

        let mesh = cube_sphere(3).expect("a mesh");
        let finest = level(8);
        let (path, config) = saved(&dir, "display.bcp", &mesh, finest);
        let mut project = Project::open(&path, config).expect("opened");
        let store = project.store();
        let keys = display_tiles(&mesh, &[6, 7]);
        let (mut on_owner, mut ahead) = (Vec::new(), Vec::new());
        for _ in 0..REPEATS {
            on_owner.push(staged_reading_on_the_owner(
                gpu, &mesh, finest, store, &keys,
            ));
            ahead.push(staged_read_ahead(gpu, &mesh, finest, store, &keys));
        }
        report("requested display tiles, read on the owner", &on_owner);
        report("requested display tiles, read ahead", &ahead);
        drop(project.close(false));
    });
}

mod finest;
