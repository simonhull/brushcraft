//! Building the spatial index.
//!
//! - [`bvh_build_153600`]: the tree alone over the benchmark sphere's 153,600 level-9 tiles, from
//!   bounds already known (worked out analytically, see [`sphere_tiles`]).
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
    Bvh, Meta, ProjectFile, ProjectId, Role, SpatialIndex, StoreConfig, TileBounds, TileStore,
};

fn main() {
    divan::main();
}

/// `cube_sphere`'s splits for the benchmark sphere: `6 · 20²` = 2,400 faces.
const SPLITS: u16 = 20;

/// Tiles per side of a face at level 9: 512 quads per side in tiles of 64.
const TILES_PER_SIDE: u16 = 8;

/// Samples per side of a tile its box is taken over.
const SAMPLES: u16 = 5;

/// The benchmark sphere's 153,600 level-9 tiles with their bounds, worked out analytically.
///
/// That is 2,400 faces of 8 × 8 tiles, each with the box and normal cone of its patch of the
/// sphere, computed directly rather than by evaluating the tiles (which takes minutes).
/// The base mesh is a cube with each side split 20 × 20, projected onto the unit sphere (see
/// `geom::cube_sphere`, whose face order and side axes this follows), and a tile covers an
/// eighth of a face's side each way. The tile's patch is taken as the projection of its square
/// of the cube: its box is that of a 5 × 5 grid of projected points, and its normals are those
/// points' directions. The subdivided surface lies a little inside the unit sphere, and between
/// the samples the patch bulges a little outside their box; neither changes the boxes' sizes,
/// overlaps or layout, which is what the build's cost depends on.
///
/// # Panics
///
/// Never: the keys are in range and every box is finite.
fn sphere_tiles() -> Vec<(TileKey, TileBounds)> {
    let level = Level::new(9).expect("level");
    // (normal, u, v) per cube side, as `geom::cube_sphere` lists them.
    let sides: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    let vec = |[x, y, z]: [f32; 3]| Vec3::new(x, y, z);
    // The face-grid step of one tile, on a cube of half-size 1.
    let step = 2.0 / f32::from(SPLITS * TILES_PER_SIDE);
    let mut tiles = Vec::with_capacity(153_600);
    let mut face = 0;
    for (normal, u, v) in sides {
        let (normal, u, v) = (vec(normal), vec(u), vec(v));
        for row in 0..SPLITS {
            for col in 0..SPLITS {
                for ty in 0..TILES_PER_SIDE {
                    for tx in 0..TILES_PER_SIDE {
                        let s0 = f32::from(col * TILES_PER_SIDE + tx) * step - 1.0;
                        let t0 = f32::from(row * TILES_PER_SIDE + ty) * step - 1.0;
                        let samples: Vec<Vec3> = (0..SAMPLES)
                            .flat_map(|j| (0..SAMPLES).map(move |i| (i, j)))
                            .map(|(i, j)| {
                                let a = s0 + step * f32::from(i) / f32::from(SAMPLES - 1);
                                let b = t0 + step * f32::from(j) / f32::from(SAMPLES - 1);
                                (normal + u * a + v * b)
                                    .normalized()
                                    .expect("off the centre")
                            })
                            .collect();
                        let key =
                            TileKey::displacement(FaceId::new(face), level, tx, ty).expect("key");
                        let bounds = TileBounds::of_tile(&samples, &samples).expect("bounds");
                        tiles.push((key, bounds));
                    }
                }
                face += 1;
            }
        }
    }
    tiles
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
        let config = StoreConfig::new(0, 1).with_chunk_bytes(4 << 20);
        let config = config.with_budget_bytes(config.minimum_budget(&mesh).max(256 << 20));
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
