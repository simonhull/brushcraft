//! What the `apply` and `surface` tests share: projects in temporary directories, meshes, dabs
//! and snapshots of a store.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{BaseMesh, Level, TileKey, Vec3, cube_sphere};
use store::{Project, ProjectConfig, SpatialIndex, StoreConfig, TileStore, Workers};
use stroke::Dab;

/// Exactly `n` evaluation workers.
///
/// # Panics
///
/// If `n` is 0.
pub(crate) fn fixed(n: usize) -> Workers {
    Workers::Fixed(std::num::NonZeroUsize::new(n).expect("a positive worker count"))
}

/// Distinguishes the directories of one process.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// A fresh directory under the system temp dir, deleted with everything in it on drop.
#[derive(Debug)]
pub(crate) struct TestDir(PathBuf);

impl TestDir {
    /// Creates the directory.
    ///
    /// # Panics
    ///
    /// If it can't be created.
    pub(crate) fn new() -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-sculpt-frame-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// A new project over `mesh` with a `budget_mib` budget and `workers`, recovery files in
/// `dir`, and its spatial index at `level`.
///
/// # Panics
///
/// If the project or its index can't be made.
pub(crate) fn project_over(
    dir: &TestDir,
    mesh: BaseMesh,
    level: Level,
    budget_mib: usize,
    workers: Workers,
) -> (Project, SpatialIndex) {
    let config = ProjectConfig {
        store: StoreConfig::new(budget_mib << 20, 1)
            .with_chunk_bytes(1 << 20)
            .with_workers(workers),
        recovery_dir: dir.0.join("recovery"),
    };
    let mut project = Project::new(mesh, config).unwrap();
    let index = SpatialIndex::build(project.store(), level, |_, _| {}).unwrap();
    (project, index)
}

/// The cube (`cube_sphere(1)`, 6 faces) at `level` under a 64 MiB budget, with `workers`.
pub(crate) fn cube(dir: &TestDir, level: Level, workers: Workers) -> (Project, SpatialIndex) {
    project_over(dir, cube_sphere(1).unwrap(), level, 64, workers)
}

/// Two copies of the cube (`cube_sphere(1)`), the second moved `offset` along x: a mesh of two
/// separate parts, faces `0..6` the first and `6..12` the second.
pub(crate) fn two_cubes(offset: f32) -> BaseMesh {
    let one = cube_sphere(1).unwrap();
    let count = one.vertex_count();
    let positions: Vec<Vec3> = one
        .vertices()
        .map(|v| one.vertex_position(v).unwrap())
        .collect();
    let moved = positions.iter().map(|p| Vec3::new(p.x + offset, p.y, p.z));
    let positions: Vec<Vec3> = positions.iter().copied().chain(moved).collect();
    let quads: Vec<[u32; 4]> = one
        .faces()
        .map(|f| [0, 1, 2, 3].map(|c| one.corner_vertex(f, c).unwrap().index()))
        .collect();
    let shifted = quads.iter().map(|q| q.map(|v| v + count));
    let polygons: Vec<[u32; 4]> = quads.iter().copied().chain(shifted).collect();
    BaseMesh::from_polygons(positions, &polygons).unwrap()
}

/// Every tile of `level` of `mesh`, face by face.
pub(crate) fn tiles(mesh: &BaseMesh, level: Level) -> Vec<TileKey> {
    let per_side = u16::try_from(level.tiles_per_side()).unwrap();
    mesh.faces()
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| TileKey::displacement(face, level, tx, ty).unwrap())
            })
        })
        .collect()
}

/// A draw dab at `position` of `radius`, `strength` and `hardness`, number `seq`.
pub(crate) fn dab(position: Vec3, radius: f32, strength: f32, hardness: f32, seq: u32) -> Dab {
    Dab {
        position: [position.x, position.y, position.z],
        normal: [0.0, 0.0, 1.0],
        tangent: [1.0, 0.0, 0.0],
        radius,
        strength,
        hardness,
        seq,
        ..Dab::default()
    }
}

/// One tile as the store holds it, in bits: its scale, each vertex's correction, position and
/// normal, row-major.
pub(crate) type TileBits = (u32, Vec<[u32; 9]>);

/// Every tile of `level`'s corrections, positions and normals, in bits.
///
/// # Panics
///
/// If a tile can't be read.
pub(crate) fn snapshot(store: &mut TileStore, level: Level) -> BTreeMap<TileKey, TileBits> {
    let keys = tiles(store.mesh(), level);
    let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
    keys.into_iter()
        .map(|key| {
            let side = key.side();
            let view = store.correction(key).unwrap();
            let scale = view.scale().to_bits();
            let corrections: Vec<Vec3> = (0..side)
                .flat_map(|j| (0..side).map(move |i| (i, j)))
                .map(|(i, j)| view.get(i, j).unwrap())
                .collect();
            let normals = store.normals(key).unwrap().to_vec();
            let positions = store.evaluate(key).unwrap().to_vec();
            let vertices = corrections
                .iter()
                .zip(&positions)
                .zip(&normals)
                .map(|((&c, &p), &n)| {
                    let mut all = [0; 9];
                    let parts = [bits(c), bits(p), bits(n)];
                    for (slot, value) in all.iter_mut().zip(parts.iter().flatten()) {
                        *slot = *value;
                    }
                    all
                })
                .collect();
            (key, (scale, vertices))
        })
        .collect()
}

/// The evaluated position of vertex `(i, j)` of tile `key`.
///
/// # Panics
///
/// If the tile can't be evaluated or holds no such vertex.
pub(crate) fn vertex(store: &mut TileStore, key: TileKey, i: u32, j: u32) -> Vec3 {
    let side = key.side();
    store.evaluate(key).unwrap()[usize::try_from(j * side + i).unwrap()]
}
