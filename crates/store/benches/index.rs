//! Building the spatial index.
//!
//! - [`bvh_build_153600`]: the tree alone over 153,600 tile boxes laid out like the benchmark
//!   sphere's level-9 tiles (small boxes on the unit sphere), from bounds already known.
//! - [`spatial_index_build`]: the whole build through a store, evaluating every tile of a level
//!   from a cold cache (positions and normals) and building the tree over them: a 96-face sphere at
//!   level 8, 1,536 tiles.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use store::{
    Aabb, Bvh, Meta, NormalCone, ProjectFile, ProjectId, Role, SpatialIndex, StoreConfig,
    TileBounds, TileStore,
};

fn main() {
    divan::main();
}

/// Tiles of the benchmark sphere at level 9: 2,400 faces of 64 tiles.
const SPHERE_TILES: u32 = 153_600;

/// 153,600 boxes of a level-9 tile's size (about 0.01 across) around random points of the unit
/// sphere, keyed as the sphere's tiles are.
///
/// # Panics
///
/// Never: the fixed keys and boxes are valid.
fn sphere_tiles() -> Vec<(TileKey, TileBounds)> {
    let level = Level::new(9).expect("level");
    let mut state = 1_u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        f32::from(u16::try_from(state >> 48).expect("16 bits")) / 32768.0 - 1.0
    };
    (0..SPHERE_TILES)
        .map(|i| {
            let key = TileKey::displacement(
                FaceId::new(i / 64),
                level,
                u16::try_from(i % 8).expect("tx"),
                u16::try_from((i / 8) % 8).expect("ty"),
            )
            .expect("key");
            let center = Vec3::new(unit(), unit(), unit())
                .normalized()
                .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            let half = Vec3::new(0.005, 0.005, 0.005);
            let aabb = Aabb::new(center - half, center + half).expect("box");
            let cone = NormalCone::from_normals(&[center]);
            (key, TileBounds { aabb, cone })
        })
        .collect()
}

/// [`Bvh::build`] over [`sphere_tiles`].
///
/// # Panics
///
/// Never: the tiles are distinct.
#[divan::bench(sample_count = 10)]
fn bvh_build_153600(bencher: divan::Bencher<'_, '_>) {
    let tiles = sphere_tiles();
    bencher
        .with_inputs(|| tiles.clone())
        .bench_local_values(|tiles| Bvh::build(tiles).expect("build"));
}

/// A store over a fresh, empty working file in its own temporary directory, removed on drop.
struct Fresh {
    /// The store.
    store: TileStore,
    /// The directory holding its working file.
    dir: PathBuf,
}

impl Fresh {
    /// Makes the store over a 96-face sphere with room for the build's working set.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-index-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let mesh = Arc::new(cube_sphere(4).expect("sphere"));
        let meta = Meta::new(
            ProjectId::from_bits(1),
            Role::Recovery { base: None },
            &mesh,
        );
        let file =
            ProjectFile::create(&dir.join("index.bcrf"), meta, Arc::clone(&mesh)).expect("file");
        let mut config = StoreConfig {
            budget_bytes: 0,
            chunk_bytes: 4 << 20,
            site: 1,
        };
        config.budget_bytes = config.minimum_budget(&mesh).max(256 << 20);
        let store = TileStore::open(file, config).expect("store");
        Self { store, dir }
    }
}

impl Drop for Fresh {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// [`SpatialIndex::build`] at level 8 of a 96-face sphere (1,536 tiles), from a cold cache.
///
/// # Panics
///
/// If the fixed setup is rejected or a tile can't be evaluated, which would be a bug.
#[divan::bench(sample_count = 5)]
fn spatial_index_build(bencher: divan::Bencher<'_, '_>) {
    let level = Level::new(8).expect("level");
    bencher
        .with_inputs(Fresh::new)
        .bench_local_values(|mut fresh| {
            let index = SpatialIndex::build(&mut fresh.store, level, |_, _| {}).expect("build");
            (fresh, index)
        });
}
