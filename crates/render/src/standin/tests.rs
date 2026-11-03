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
    for own in [
        REPLACE_WGSL,
        &format!("{REPLACE_WGSL}{NORMALS_WGSL}"),
        SETTLE_WGSL,
    ] {
        let indexed = crate::shader_check::run_time_indexed_locals(&format!("{prefix}{own}"));
        assert!(indexed.is_empty(), "{indexed:?}");
    }
}

#[test]
fn a_link_s_map_finds_the_tile_across_every_edge_on_a_face_and_across_its_edges() {
    let mesh = cube_sphere(1).expect("a mesh");
    let finest = level(8);
    let per_side = u16::try_from(finest.tiles_per_side()).expect("small");
    let places: Vec<Place> = mesh
        .faces()
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    Place::of(&TileKey::displacement(face, finest, tx, ty).expect("a tile"))
                })
            })
        })
        .collect();
    let links = Links {
        scratch: places.iter().zip(0..).map(|(&p, i)| (p, i)).collect(),
        topology: mesh.topology(),
        finest,
    };
    let (mut inside, mut across) = (0, 0);
    for &place in &places {
        let neighbours = place.neighbours(mesh.topology());
        for (edge, other) in neighbours.iter().enumerate() {
            let other = other.expect("a closed mesh");
            // Whichever way a map is found, the topology's agrees with it.
            let by_topology = links.map(place, edge, other).expect("a map");
            if let Some((on_face, code)) = links.on_face(place, edge) {
                assert_eq!(on_face, other, "{place:?} edge {edge}");
                assert_eq!(code, by_topology, "{place:?} edge {edge}");
                inside += 1;
            } else {
                assert_ne!(other.face(), place.face(), "{place:?} edge {edge}");
                across += 1;
            }
            let (found, maps) = links.around(place);
            assert_eq!(found[edge], links.scratch[&other], "{place:?} edge {edge}");
            assert_eq!(maps[edge], by_topology, "{place:?} edge {edge}");
        }
    }
    assert!(inside > 0 && across > 0, "{inside} inside, {across} across");
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

// A strong stroke's stand-ins against the refresh that replaces them; ignored variants print.
#[cfg(not(doc))]
mod strong_stroke;

// Uses the sculpt crate's types, a dev-dependency rustdoc doesn't link.
#[cfg(not(doc))]
#[test]
fn dabs_shown_from_a_distance_stand_in_at_once_and_catch_up_to_the_display_build() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("preview");
        let (mut project, mesh) = project(&dir, |config| config);
        let finest = level(FINEST);
        let store = project.store();
        let cursor = store.subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, store);
        let mut index = store::SpatialIndex::build(store, finest, |_, _| {}).expect("an index");
        // A stroke of dabs across face 2's corner and centre, shown ahead of their finest work.
        let grids = finest_grids(store, &mesh);
        let n = finest.face_quads();
        let dabs: Vec<stroke::Dab> = (0..6_u32)
            .map(|k| {
                let (i, j) = (k * n / 6, k * n / 7);
                let p = grids[&2][usize::try_from(j * (n + 1) + i).expect("small")];
                let length = p.dot(p).sqrt();
                stroke::Dab {
                    position: [p.x, p.y, p.z],
                    normal: [p.x / length, p.y / length, p.z / length],
                    radius: 0.12,
                    strength: 0.01,
                    seq: k,
                    ..stroke::Dab::default()
                }
            })
            .collect();
        let mut preview = sculpt::Preview::new();
        preview
            .show(store, &index, &dabs, &finest_only)
            .expect("shown");
        let previewed = store.drain_stale(&cursor);
        let Drained::Epochs(epochs) = &previewed else {
            panic!("no resync");
        };
        assert!(
            epochs.iter().all(|epoch| epoch.kind == EpochKind::Preview),
            "the preview's epoch alone: nothing landed"
        );
        frame(gpu, &mut residency, store, previewed);
        let truth = finest_grids(store, &mesh);
        assert_eq!(truth, grids, "the store's surface hasn't moved");
        let shown = model::drawn_grids(store, &truth);
        assert_ne!(shown, truth, "the previews moved what is drawn");
        check(gpu, &residency, &shown, &BTreeSet::new());
        let stats = residency.stats();
        assert!(
            stats.stand_ins > 0 && stats.stand_ins_skipped == 0,
            "seen from a distance, shown by stand-ins at once: {stats:?}"
        );

        // The catch-up lands them in two batches, then the refresh runs.
        let mut stroke = sculpt::Stroke::new(store);
        project.begin_stroke().expect("a stroke");
        for batch in dabs.chunks(3) {
            let landing = project.store();
            sculpt::apply_dabs(landing, &mut index, &mut stroke, batch).expect("applied");
            preview
                .land(landing, &index, batch.len(), &finest_only)
                .expect("landed");
            let landed = landing.drain_stale(&cursor);
            frame(gpu, &mut residency, landing, landed);
            let now = finest_grids(landing, &mesh);
            let drawn = model::drawn_grids(landing, &now);
            check(gpu, &residency, &drawn, &BTreeSet::new());
        }
        project.end_unrecorded_stroke().expect("ended");
        let refreshing = project.store();
        assert!(!refreshing.has_previews(), "caught up: no preview left");
        run_jobs(refreshing);
        let refreshed = refreshing.drain_stale(&cursor);
        frame(gpu, &mut residency, refreshing, refreshed);
        frame(gpu, &mut residency, refreshing, Drained::Epochs(Vec::new()));
        let exact = finest_grids(refreshing, &mesh);
        check(gpu, &residency, &exact, &BTreeSet::new());
        let mut compared = 0;
        for (place, (vertices, node)) in display_tiles(gpu, &residency) {
            let key = place.key(finest).expect("a key");
            let built = refreshing
                .display_for_draw(&[key])
                .expect("reads")
                .pop()
                .flatten()
                .expect("rebuilt");
            assert_eq!(node.error, built.error(), "{place:?} draws the built bound");
            for (drawn, stored) in vertices.iter().zip(built.positions()) {
                assert_eq!(
                    drawn.position.map(f32::to_bits),
                    [stored.x, stored.y, stored.z].map(f32::to_bits),
                    "{place:?} draws the display build of the finest result"
                );
                compared += 1;
            }
        }
        assert!(compared > 1000, "{compared} vertices");
        refreshing.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

// Uses the sculpt crate's types, a dev-dependency rustdoc doesn't link.
#[cfg(not(doc))]
#[test]
fn dabs_shown_on_display_tiles_are_drawn_from_them_with_closed_seams_until_the_refresh_replaces_them()
 {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("display-preview");
        let (mut project, mesh) = project(&dir, |config| config);
        let finest = level(FINEST);
        let store = project.store();
        let cursor = store.subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, store);
        let mut index = store::SpatialIndex::build(store, finest, |_, _| {}).expect("an index");
        let grids = finest_grids(store, &mesh);
        let n = finest.face_quads();
        let dabs: Vec<stroke::Dab> = (0..6_u32)
            .map(|k| {
                let (i, j) = (k * n / 6, k * n / 7);
                let p = grids[&2][usize::try_from(j * (n + 1) + i).expect("small")];
                let length = p.dot(p).sqrt();
                stroke::Dab {
                    position: [p.x, p.y, p.z],
                    normal: [p.x / length, p.y / length, p.z / length],
                    radius: 0.12,
                    strength: 0.01,
                    seq: k,
                    ..stroke::Dab::default()
                }
            })
            .collect();
        let resident: BTreeSet<Place> = display_tiles(gpu, &residency).keys().copied().collect();
        let mut preview = sculpt::Preview::new();
        preview
            .show(store, &index, &dabs, &sculpt::DrawsAll)
            .expect("shown");
        let previewed = store.drain_stale(&cursor);
        frame(gpu, &mut residency, store, previewed);
        let truth = finest_grids(store, &mesh);
        let shown = model::drawn_grids(store, &truth);
        assert_ne!(shown, truth, "the previews moved what is drawn");
        // Every resident display tile meets the finest previews bit for bit at its border and
        // bounds them; none stands in (the brush makes the held ones).
        check(gpu, &residency, &shown, &resident);
        let held = display_tiles(gpu, &residency)
            .keys()
            .filter(|place| {
                place
                    .key(finest)
                    .is_some_and(|key| store.display_preview(&key).is_some())
            })
            .count();
        assert!(held > 0, "display previews drawn");
        assert_eq!(
            residency.stats().stand_ins,
            0,
            "no stand-in over a display preview"
        );

        let mut stroke = sculpt::Stroke::new(store);
        project.begin_stroke().expect("a stroke");
        for batch in dabs.chunks(3) {
            let landing = project.store();
            sculpt::apply_dabs(landing, &mut index, &mut stroke, batch).expect("applied");
            preview
                .land(landing, &index, batch.len(), &sculpt::DrawsAll)
                .expect("landed");
            let landed = landing.drain_stale(&cursor);
            frame(gpu, &mut residency, landing, landed);
            let now = finest_grids(landing, &mesh);
            let drawn = model::drawn_grids(landing, &now);
            check(gpu, &residency, &drawn, &resident);
        }
        preview.end_stroke(project.store()).expect("ended");
        project.end_unrecorded_stroke().expect("ended");
        let refreshing = project.store();
        let released = refreshing.drain_stale(&cursor);
        frame(gpu, &mut residency, refreshing, released);
        let exact = finest_grids(refreshing, &mesh);
        check(gpu, &residency, &exact, &resident);
        run_jobs(refreshing);
        assert!(
            refreshing.display_previewed().next().is_none(),
            "the refresh replaced the previews"
        );
        let refreshed = refreshing.drain_stale(&cursor);
        frame(gpu, &mut residency, refreshing, refreshed);
        frame(gpu, &mut residency, refreshing, Drained::Epochs(Vec::new()));
        check(gpu, &residency, &exact, &resident);
        for (place, (vertices, node)) in display_tiles(gpu, &residency) {
            let key = place.key(finest).expect("a key");
            let built = refreshing
                .display_for_draw(&[key])
                .expect("reads")
                .pop()
                .flatten()
                .expect("rebuilt");
            assert_eq!(node.error, built.error(), "{place:?} draws the built bound");
            for (drawn, stored) in vertices.iter().zip(built.positions()) {
                assert_eq!(
                    drawn.position.map(f32::to_bits),
                    [stored.x, stored.y, stored.z].map(f32::to_bits),
                    "{place:?} draws the display build of the finest result"
                );
            }
        }
        refreshing.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

// Uses the sculpt crate's types, a dev-dependency rustdoc doesn't link.
#[cfg(not(doc))]
#[test]
fn a_long_previewed_stroke_keeps_every_drawn_bound_true_and_flat_frame_by_frame_level_0_included() {
    GpuTest::run(|gpu| {
        let dir = TempDir::new("long-preview");
        let (mut project, mesh) = project(&dir, |config| config);
        let finest = level(FINEST);
        let store = project.store();
        let cursor = store.subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, store);
        let mut index = store::SpatialIndex::build(store, finest, |_, _| {}).expect("an index");
        let grids = finest_grids(store, &mesh);
        let n = finest.face_quads();
        // A long stroke across face 2, a dab a frame, landing three frames behind.
        let dabs: Vec<stroke::Dab> = (0..18_u32)
            .map(|k| {
                let (i, j) = (n / 8 + k * n / 24, n / 2);
                let p = grids[&2][usize::try_from(j * (n + 1) + i).expect("small")];
                let length = p.dot(p).sqrt();
                stroke::Dab {
                    position: [p.x, p.y, p.z],
                    normal: [p.x / length, p.y / length, p.z / length],
                    radius: 0.1,
                    strength: 0.01,
                    seq: k,
                    ..stroke::Dab::default()
                }
            })
            .collect();
        let resident: BTreeSet<Place> = display_tiles(gpu, &residency).keys().copied().collect();
        let mut preview = sculpt::Preview::new();
        let mut stroke = sculpt::Stroke::new(store);
        project.begin_stroke().expect("a stroke");
        let mut largest = Vec::new();
        let mut every = Vec::new();
        for (k, dab) in dabs.iter().enumerate() {
            let stepping = project.store();
            preview
                .show(
                    stepping,
                    &index,
                    std::slice::from_ref(dab),
                    &sculpt::DrawsAll,
                )
                .expect("shown");
            if k >= 3 {
                let landing = dabs.get(k - 3..=k - 3).expect("a dab");
                sculpt::apply_dabs(stepping, &mut index, &mut stroke, landing).expect("applied");
                preview
                    .land(stepping, &index, 1, &sculpt::DrawsAll)
                    .expect("landed");
            }
            let drained = stepping.drain_stale(&cursor);
            frame(gpu, &mut residency, stepping, drained);
            let now = finest_grids(stepping, &mesh);
            let drawn = model::drawn_grids(stepping, &now);
            check(gpu, &residency, &drawn, &resident);
            // The previewed tiles' bounds, and every tile's: a tile no dab reaches stands in only
            // as a reader of a moved ring, with no displacement, at any level.
            let tiles = display_tiles(gpu, &residency);
            let everyone = tiles
                .iter()
                .map(|(place, (_, node))| (place.level(), node.error))
                .fold((Level::ZERO, 0.0_f32), |most, (level, error)| {
                    if error > most.1 { (level, error) } else { most }
                });
            every.push(everyone);
            let previewed = tiles.iter().filter(|(place, _)| {
                place
                    .key(finest)
                    .is_some_and(|key| stepping.display_preview(&key).is_some())
            });
            let bound = previewed
                .map(|(_, (_, node))| node.error)
                .fold(0.0_f32, f32::max);
            largest.push(bound);
        }
        // Released tiles stand in for the steps after their release, as any data would.
        eprintln!("the previewed tiles' largest bound, frame by frame: {largest:?}");
        eprintln!("every tile's largest bound, with its level: {every:?}");
        let (first, last) = (
            every.first().expect("frames").1,
            every.last().expect("frames").1,
        );
        assert!(
            last <= 1.5 * first,
            "no level compounds, level 0 included: {every:?}"
        );
        let early = largest.get(5).copied().expect("frames");
        let late = largest.last().copied().expect("frames");
        assert!(
            late <= 1.5 * early,
            "the bounds follow the dabs over each place, not the frames: {largest:?}"
        );
        project.store().unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}

/// A renderer drawing only finest tiles, as these tests' stand-ins over previews assume: the
/// previews are of finest tiles alone, and display tiles above them stand in.
pub(crate) fn finest_only(key: &TileKey) -> bool {
    key.channel() != Channel::DISPLAY
}
