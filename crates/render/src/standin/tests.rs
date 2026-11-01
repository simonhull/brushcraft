//! Stand-ins end to end over the store.
//!
//! A project with its display layer built, seen from a distance (every display tile resident,
//! no finest tile), edited by corrections the way a
//! brush step writes them (its display bounds bumped too), and the step's epoch applied with
//! [`Residency::apply`]: the edited finest tiles reach the scratch as the store's rings,
//! the store's own epoch names every finest tile holding a moved vertex (across faces, round a
//! base vertex), and every resident display tile's border is the store's finest bits after it.
//! Then the refresh replaces the stand-ins.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use geom::{BaseMesh, Channel, FaceId, Level, TileKey, Vec3, cube_sphere};
use store::{Drained, EpochKind, Project, ProjectConfig, StaleCursor, StoreConfig, TileStore};
use wgpu::{CommandEncoderDescriptor, PollType};

use super::*;
use crate::gpu::GpuTest;
use crate::residency::{FrameUse, Residency, ResidencyConfig};
use crate::tree::GpuNode;
use crate::upload::StoreSource;

/// The finest level: four finest tiles a face.
const FINEST: u8 = 7;

/// A fresh recovery directory under the system temp dir, removed with everything in it on drop.
struct TempDir(PathBuf);

impl TempDir {
    /// Names a directory for this process and `name` (the project creates it).
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-standins-{}-{name}", std::process::id()));
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

/// Runs the store's jobs until there are none: the display layer built, or refreshed. At least
/// once: the first run queues the refresh of the writes since.
fn run_jobs(store: &mut TileStore) {
    loop {
        store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .expect("the jobs run");
        if !store.has_jobs() {
            break;
        }
    }
}

/// One frame over the store: the drain applied, staged, submitted and waited for.
fn frame(gpu: &Gpu, residency: &mut Residency, store: &mut TileStore, drained: Drained) {
    residency.begin_frame(gpu).expect("a frame begins");
    residency.apply(drained);
    residency.use_frame(FrameUse::default());
    let mut source = StoreSource::new(store);
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency.stage(gpu, &mut source, deadline).expect("staged");
    let encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    uploads.submit(gpu.queue(), encoder, |_| {});
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .expect("the GPU finishes");
    gpu.check().expect("no GPU error");
}

/// Every face's finest grid as the store evaluates it now, row-major.
fn finest_grids(store: &mut TileStore, mesh: &BaseMesh) -> BTreeMap<u32, Vec<Vec3>> {
    let finest = level(FINEST);
    let face_quads = finest.face_quads();
    let quads = finest.tile_quads();
    let padded = quads + 3;
    let mut grids = BTreeMap::new();
    for face in mesh.faces() {
        let mut grid =
            vec![Vec3::ZERO; usize::try_from((face_quads + 1) * (face_quads + 1)).expect("small")];
        let per_side = u16::try_from(finest.tiles_per_side()).expect("small");
        for ty in 0..per_side {
            for tx in 0..per_side {
                let key = TileKey::displacement(face, finest, tx, ty).expect("a key");
                let block = store.positions_with_ring(key).expect("evaluated");
                for j in 0..=quads {
                    for i in 0..=quads {
                        let point = block.padded()
                            [usize::try_from((j + 1) * padded + i + 1).expect("small")];
                        let (x, y) = (u32::from(tx) * quads + i, u32::from(ty) * quads + j);
                        grid[usize::try_from(y * (face_quads + 1) + x).expect("small")] = point;
                    }
                }
            }
        }
        grids.insert(face.index(), grid);
    }
    grids
}

/// What the GPU holds for each resident display tile: its vertices and its node.
fn display_tiles(gpu: &Gpu, residency: &Residency) -> BTreeMap<Place, (Vec<GpuVertex>, GpuNode)> {
    let nodes: Vec<GpuNode> = gpu
        .read_buffer(residency.tree().buffer())
        .expect("the tree reads back");
    let mut buffers: BTreeMap<(u8, u16), Vec<GpuVertex>> = BTreeMap::new();
    let mut tiles = BTreeMap::new();
    for key in residency.resident() {
        if key.channel() != Channel::DISPLAY {
            continue;
        }
        let slot = residency.slot(key).expect("resident");
        let data = buffers
            .entry((slot.class().get(), slot.buffer()))
            .or_insert_with(|| {
                let buffer = &residency.slot_buffers(slot.class())[usize::from(slot.buffer())];
                gpu.read_buffer(buffer).expect("slots read back")
            });
        let count = usize::try_from(slot.class().vertices()).expect("small");
        let first = usize::try_from(slot.index()).expect("small") * count;
        let place = Place::of(key);
        let node = residency.tree().node(place).expect("a node");
        let node = nodes[usize::try_from(node).expect("small")];
        tiles.insert(place, (data[first..first + count].to_vec(), node));
    }
    tiles
}

/// Checks every resident display tile against the store's finest level `grids`.
///
/// Each border vertex holds the finest vertex's bits there (so every seam between two of them,
/// across faces too, closes), and every finest vertex under each tile in `bounded` lies within the
/// tile's bound of its triangles (measured in `f64`). Returns how many border vertices were
/// checked.
fn check(
    gpu: &Gpu,
    residency: &Residency,
    grids: &BTreeMap<u32, Vec<Vec3>>,
    bounded: &BTreeSet<Place>,
) -> usize {
    let finest = level(FINEST);
    let n = finest.face_quads();
    let at = |face: FaceId, x: u32, y: u32| {
        grids[&face.index()][usize::try_from(y * (n + 1) + x).expect("small")]
    };
    let mut checked = 0;
    for (place, (vertices, node)) in display_tiles(gpu, residency) {
        let size = n / place.level().tiles_per_side();
        let (x0, y0) = (u32::from(place.tx()) * size, u32::from(place.ty()) * size);
        let quads = place.level().tile_quads();
        let scale = size / quads;
        let vertex =
            |i: u32, j: u32| vertices[usize::try_from(j * (quads + 1) + i).expect("small")];
        let border = (0..=quads)
            .flat_map(|j| (0..=quads).map(move |i| (i, j)))
            .filter(|&(i, j)| i == 0 || j == 0 || i == quads || j == quads);
        for (i, j) in border {
            let finest_bits = at(place.face(), x0 + i * scale, y0 + j * scale).to_bits();
            assert_eq!(
                vertex(i, j).position.map(f32::to_bits),
                finest_bits,
                "{place:?} ({i}, {j}) isn't the finest vertex's bits"
            );
            checked += 1;
        }
        if !bounded.contains(&place) {
            continue;
        }
        let mut worst = 0.0_f64;
        for y in 0..=size {
            for x in 0..=size {
                let fine = at(place.face(), x0 + x, y0 + y);
                let fine = [fine.x, fine.y, fine.z].map(f64::from);
                let (qx, qy) = ((x / scale).min(quads - 1), (y / scale).min(quads - 1));
                let fx = f64::from(x - qx * scale) / f64::from(scale);
                let fy = f64::from(y - qy * scale) / f64::from(scale);
                let p = |i, j| vertex(i, j).position.map(f64::from);
                let (p00, p10, p11, p01) =
                    (p(qx, qy), p(qx + 1, qy), p(qx + 1, qy + 1), p(qx, qy + 1));
                let drawn: [f64; 3] = [0, 1, 2].map(|a| {
                    if fx >= fy {
                        p00[a] + fx * (p10[a] - p00[a]) + fy * (p11[a] - p10[a])
                    } else {
                        p00[a] + fx * (p11[a] - p01[a]) + fy * (p01[a] - p00[a])
                    }
                });
                let d = [0, 1, 2].map(|a| fine[a] - drawn[a]);
                worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
        assert!(
            worst <= f64::from(node.error),
            "{place:?}: a finest vertex {worst} from its triangles, past its bound {}",
            node.error
        );
    }
    checked
}

/// The display tiles above `keys`, finest tiles.
fn above(keys: &BTreeSet<TileKey>) -> BTreeSet<Place> {
    let mut above = BTreeSet::new();
    for key in keys {
        let mut at = Place::of(key).parent();
        while let Some(place) = at {
            above.insert(place);
            at = place.parent();
        }
    }
    above
}

/// The finest tiles of `drained`'s epochs.
fn finest_of(drained: &Drained) -> BTreeSet<TileKey> {
    let Drained::Epochs(epochs) = drained else {
        panic!("epochs, not a resync");
    };
    let keys = epochs.iter().flat_map(|e| e.keys.iter().copied());
    keys.filter(|key| key.channel() != Channel::DISPLAY && key.level() == level(FINEST))
        .collect()
}

/// A brush step on `face` by corrections, as the edit path writes one.
///
/// It lifts by `lift` the face's corner (a base vertex), a vertex along its border and two at
/// its centre, with the display bounds bumped as the edit path bumps them, by the displacement
/// measured on the store's finest level. Returns its drain.
fn step(
    store: &mut TileStore,
    mesh: &BaseMesh,
    cursor: &StaleCursor,
    face: FaceId,
    lift: Vec3,
) -> Drained {
    let finest = level(FINEST);
    let n = finest.face_quads();
    let before = finest_grids(store, mesh);
    store.begin_op().expect("an op");
    let mut edited = Vec::new();
    for (fi, fj) in [
        (0, 0),
        (n / 2 + 9, 0),
        (n / 2, n / 2),
        (n / 2 + 3, n / 2 - 5),
    ] {
        let (key, li, lj) = TileKey::containing(
            geom::LayerId::BASE,
            Channel::DISPLACEMENT,
            face,
            finest,
            fi,
            fj,
        )
        .expect("a finest tile");
        let c = store
            .correction(key)
            .expect("reads")
            .get(li, lj)
            .expect("a vertex");
        store
            .set_correction(face, finest, fi, fj, c + lift)
            .expect("written");
        // The write goes to the vertex's owner copy, which may be another face's tile.
        let copies = mesh.topology().copies(face, finest, fi, fj);
        let holders = copies
            .iter()
            .filter_map(|copy| TileKey::displacement(copy.face, finest, copy.tx, copy.ty).ok());
        edited.extend(holders);
        edited.push(key);
    }
    let after = finest_grids(store, mesh);
    let mut worst = 0.0_f64;
    for (index, grid) in &after {
        for (p, q) in grid.iter().zip(&before[index]) {
            let d = [p.x - q.x, p.y - q.y, p.z - q.z].map(f64::from);
            worst = worst.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded to an f32, then up a step"
    )]
    let worst = (worst as f32).next_up();
    store.bump_display_error(&edited, worst).expect("bumped");
    store.close_epoch().expect("closed");
    store.drain_stale(cursor)
}

#[test]
fn a_step_on_the_store_stands_in_for_every_display_tile_above_it_and_the_refresh_replaces_them() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("store");
        let mesh = cube_sphere(1).expect("a mesh");
        let finest = level(FINEST);
        let config = ProjectConfig {
            store: StoreConfig::new(256 << 20, 1).with_display(finest),
            recovery_dir: dir.0.clone(),
        };
        let mut project = Project::new(mesh.clone(), config).expect("a project");
        let store = project.store();
        run_jobs_after_rebuild(store, finest);
        let cursor = store.subscribe_stale();
        let budget = ResidencyConfig::new(1 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(6))
            .expect("a configuration");
        let mut residency = Residency::new(gpu, &mesh, finest, budget).expect("residency");
        frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
        let built = finest_grids(store, &mesh);
        let checked = check(gpu, &residency, &built, &BTreeSet::new());
        assert!(checked > 1000, "{checked} border vertices");

        let face = FaceId::new(2);
        for lift in [Vec3::new(0.0, 0.01, 0.03), Vec3::new(0.02, -0.01, 0.0)] {
            let drained = step(store, &mesh, &cursor, face, lift);
            assert_step(&drained);
            let edited = finest_of(&drained);
            let faces: BTreeSet<FaceId> = edited.iter().map(TileKey::face).collect();
            assert!(faces.len() >= 3, "the corner's three faces: {faces:?}");
            frame(gpu, &mut residency, store, drained);
            assert!(
                residency
                    .resident()
                    .all(|key| key.channel() == Channel::DISPLAY),
                "seen from a distance: no finest tile resident"
            );
            let stepped = finest_grids(store, &mesh);
            check(gpu, &residency, &stepped, &above(&edited));
        }
        let stats = residency.stats();
        assert!(
            stats.stand_ins > 0 && stats.stand_ins_skipped == 0,
            "{stats:?}"
        );
        // The refresh: its epochs replace every stand-in.
        run_jobs(store);
        let drained = store.drain_stale(&cursor);
        frame(gpu, &mut residency, store, drained);
        frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
        let refreshed = finest_grids(store, &mesh);
        check(gpu, &residency, &refreshed, &BTreeSet::new());
        for (place, (_, node)) in display_tiles(gpu, &residency) {
            let key = place.key(finest).expect("a key");
            let record = store
                .display_for_draw(&[key])
                .expect("reads")
                .pop()
                .flatten()
                .expect("rebuilt");
            assert_eq!(
                node.error,
                record.error(),
                "{place:?} draws its refreshed bound"
            );
        }
        store.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

/// Checks that `drained` holds one epoch, a step's.
fn assert_step(drained: &Drained) {
    let Drained::Epochs(epochs) = drained else {
        panic!("epochs, not a resync");
    };
    let kinds: Vec<EpochKind> = epochs.iter().map(|epoch| epoch.kind).collect();
    assert!(
        matches!(kinds.as_slice(), [EpochKind::Step { displacement }] if *displacement > 0.0),
        "one step's epoch: {kinds:?}"
    );
}

/// Builds the store's display layer whole.
fn run_jobs_after_rebuild(store: &mut TileStore, finest: Level) {
    store.queue_display_rebuild(finest);
    run_jobs(store);
}

#[test]
fn the_passes_index_no_local_array_at_run_time() {
    let prefix = shared_source();
    for own in [REPLACE_WGSL, SETTLE_WGSL] {
        let indexed = crate::shader_check::run_time_indexed_locals(&format!("{prefix}{own}"));
        assert!(indexed.is_empty(), "{indexed:?}");
    }
}

/// A project over the cube sphere to [`FINEST`], its display layer built, with `store`'s
/// configuration changed by `configure`.
fn project(
    dir: &TempDir,
    configure: impl FnOnce(StoreConfig) -> StoreConfig,
) -> (Project, BaseMesh) {
    let mesh = cube_sphere(1).expect("a mesh");
    let finest = level(FINEST);
    let config = ProjectConfig {
        store: configure(StoreConfig::new(256 << 20, 1).with_display(finest)),
        recovery_dir: dir.0.clone(),
    };
    let mut project = Project::new(mesh.clone(), config).expect("a project");
    run_jobs_after_rebuild(project.store(), finest);
    (project, mesh)
}

/// Residency seen from a distance (every display tile resident, no finest tile), its floor up.
fn far_residency(gpu: &Gpu, mesh: &BaseMesh, store: &mut TileStore) -> Residency {
    let config = ResidencyConfig::new(1 << 30, 16 << 20, 64 << 20, 8 << 20, 64, level(6))
        .expect("a configuration");
    let mut residency = Residency::new(gpu, mesh, level(FINEST), config).expect("residency");
    frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
    residency
}

#[test]
fn a_resync_hands_over_what_the_stroke_changed_and_the_stand_ins_come_back() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("resync");
        // A feed holding a few keys: a step's epoch loses the consumer its epochs.
        let (mut project, mesh) = project(&dir, |config| config.with_stale_cap(4));
        let store = project.store();
        let cursor = store.subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, store);
        let face = FaceId::new(2);
        let drained = step(store, &mesh, &cursor, face, Vec3::new(0.0, 0.01, 0.03));
        let Drained::Resync(unrefreshed) = &drained else {
            panic!("a resync");
        };
        assert!(
            !unrefreshed.is_empty(),
            "the stroke's display tiles handed over"
        );
        let edited: BTreeSet<TileKey> = unrefreshed
            .iter()
            .flat_map(|tile| tile.finest.iter().copied())
            .collect();
        frame(gpu, &mut residency, store, drained);
        frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
        assert_eq!(residency.stats().resyncs, 1);
        assert!(residency.stats().stand_ins > 0, "stood in again");
        let grids = finest_grids(store, &mesh);
        check(gpu, &residency, &grids, &above(&edited));
        store.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

#[test]
fn a_failed_step_before_an_upkeep_refresh_never_passes_for_a_refresh() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("failed-step");
        let (mut project, mesh) = project(&dir, |config| config);
        let store = project.store();
        let cursor = store.subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, store);
        // A step that failed after its writes, before its bump: no displacement known.
        store.begin_op().expect("an op");
        let finest = level(FINEST);
        let (key, li, lj) = TileKey::containing(
            geom::LayerId::BASE,
            Channel::DISPLACEMENT,
            FaceId::new(2),
            finest,
            40,
            50,
        )
        .expect("a finest tile");
        let c = store
            .correction(key)
            .expect("reads")
            .get(li, lj)
            .expect("a vertex");
        store
            .set_correction(
                FaceId::new(2),
                finest,
                40,
                50,
                c + Vec3::new(0.0, 0.0, 0.02),
            )
            .expect("written");
        // The upkeep's refresh runs next, closing epochs of its own.
        run_jobs(store);
        let drained = store.drain_stale(&cursor);
        let Drained::Epochs(epochs) = &drained else {
            panic!("epochs");
        };
        for epoch in epochs {
            let moved = epoch
                .keys
                .iter()
                .any(|named| named.channel() != Channel::DISPLAY);
            if moved {
                assert_eq!(
                    epoch.kind,
                    EpochKind::Revert,
                    "the writes' epoch: {epoch:?}"
                );
            }
        }
        assert!(epochs.iter().any(|epoch| epoch.kind == EpochKind::Revert));
        assert!(
            epochs
                .iter()
                .filter(|epoch| epoch.kind == EpochKind::Refresh)
                .all(|epoch| epoch.keys.iter().all(|k| k.channel() == Channel::DISPLAY)),
            "a refresh's epoch holds refreshed tiles alone"
        );
        frame(gpu, &mut residency, store, drained);
        run_jobs(store);
        let refreshed = store.drain_stale(&cursor);
        frame(gpu, &mut residency, store, refreshed);
        frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
        let grids = finest_grids(store, &mesh);
        check(gpu, &residency, &grids, &BTreeSet::new());
        // The refresh unblocked the places the revert reached: a step there stands in again.
        assert!(residency.blocked_places() == 0, "unblocked by the refresh");
        let before = residency.stats().stand_ins;
        let again = step(
            store,
            &mesh,
            &cursor,
            FaceId::new(2),
            Vec3::new(0.0, 0.01, 0.0),
        );
        assert_step(&again);
        let edited = finest_of(&again);
        frame(gpu, &mut residency, store, again);
        assert!(residency.stats().stand_ins > before, "stood in again");
        let stepped = finest_grids(store, &mesh);
        check(gpu, &residency, &stepped, &above(&edited));
        store.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

#[test]
fn an_epoch_of_several_dabs_sums_their_displacements() {
    let dir = TempDir::new("summed");
    let (mut project, _) = project(&dir, |config| config);
    let store = project.store();
    let cursor = store.subscribe_stale();
    let finest = level(FINEST);
    store.begin_op().expect("an op");
    let face = FaceId::new(2);
    let (key, li, lj) = TileKey::containing(
        geom::LayerId::BASE,
        Channel::DISPLACEMENT,
        face,
        finest,
        40,
        50,
    )
    .expect("a finest tile");
    for lift in [0.002_f32, 0.003] {
        let c = store
            .correction(key)
            .expect("reads")
            .get(li, lj)
            .expect("a vertex");
        store
            .set_correction(face, finest, 40, 50, c + Vec3::new(0.0, 0.0, lift))
            .expect("written");
        store.bump_display_error(&[key], lift).expect("bumped");
    }
    store.close_epoch().expect("closed");
    let Drained::Epochs(epochs) = store.drain_stale(&cursor) else {
        panic!("epochs");
    };
    let kinds: Vec<EpochKind> = epochs.iter().map(|epoch| epoch.kind).collect();
    assert!(
        matches!(kinds.as_slice(), [EpochKind::Step { displacement }] if *displacement >= 0.005),
        "one step's epoch, its displacements summed: {kinds:?}"
    );
    store.unsubscribe_stale(cursor);
    drop(project.close(false));
}

// Its world holds the sculpt crate's types, a dev-dependency rustdoc doesn't link.
#[cfg(not(doc))]
mod model;

// A diagnosis probe (ignored): a strong stroke's stand-ins against the refresh.
#[cfg(not(doc))]
mod diag;
