//! Raycasts through the spatial index: picking's and the brush's
//! [`SpatialIndex::raycast_triangle`], on a 96-face sphere at level 8 (1,536 tiles of 64 × 64
//! quads, 12.6 million triangles), every tile evaluated and cached first, so a raycast costs
//! the tree's walk and the candidate tiles' triangle tests.
//!
//! The index keeps the blocks it raycasts a tile through until the store's next write, for the
//! last 64 tiles raycast. The benches whose rays go all round the sphere meet each tile on first
//! sight, its blocks built; the local ones meet the same few tiles again, their blocks kept
//! unless a write comes between.
//!
//! - [`raycast_spread`]: rays from all around aimed at the sphere, each at other tiles, on first
//!   sight.
//! - [`raycast_local`]: rays aimed within a small patch, a few tiles, again and again with no write
//!   between: the blocks kept.
//! - [`raycast_local_edited`]: [`raycast_local`]'s rays with a write to the tile each hits before
//!   each raycast, the tile evaluated again untimed: a stroke, each raycast after a dab.
//! - [`raycast_grazing`]: rays from all around aimed at the sphere's rim, nearly along the surface,
//!   each crossing a long row of quads, on first sight: the most a silhouette costs.
//! - [`raycast_grazing_edited`]: grazing rays at one place on the rim with a write before each, as
//!   [`raycast_local_edited`]: a stroke along the silhouette.
//! - [`raycast_miss`]: rays from all around passing beside the sphere, close to it, on first sight.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{Level, TILE_QUADS, TileKey, Vec3, cube_sphere};
use store::{Meta, ProjectFile, ProjectId, Ray, Role, SpatialIndex, StoreConfig, TileStore};

fn main() {
    divan::main();
}

/// The level raycast.
const LEVEL: u8 = 8;

/// Rays in each bench's cycle.
const RAYS: usize = 512;

/// A store over the sphere with every tile of [`LEVEL`] evaluated and its index built, in its
/// own temporary directory, removed on drop.
struct Fresh {
    /// The store, every tile of [`LEVEL`] cached.
    store: TileStore,
    /// The index at [`LEVEL`].
    index: SpatialIndex,
    /// The directory holding its working file.
    dir: PathBuf,
}

impl Fresh {
    /// Makes the store under a budget that keeps every tile of [`LEVEL`], and builds the index.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-raycast-{}-{}",
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
            ProjectFile::create(&dir.join("raycast.bcrf"), meta, Arc::clone(&mesh)).expect("file");
        let config = StoreConfig::new(0, 1).with_chunk_bytes(4 << 20);
        let config = config.with_budget_bytes(config.minimum_budget(&mesh).max(512 << 20));
        let mut store = TileStore::open(file, config).expect("store");
        let level = Level::new(LEVEL).expect("level");
        let index = SpatialIndex::build(&mut store, level, |_, _| {}).expect("index");
        Self { store, index, dir }
    }

    /// Raycasts each of `rays` in turn, cycling, once per iteration.
    ///
    /// # Panics
    ///
    /// If a raycast fails, which would be a bug.
    fn bench(mut self, bencher: divan::Bencher<'_, '_>, rays: &[Ray]) {
        let mut next = rays.iter().cycle();
        bencher.bench_local(|| {
            let ray = next.next().expect("a cycle of rays");
            self.index
                .raycast_triangle(&mut self.store, ray)
                .expect("raycast")
        });
    }

    /// Raycasts each of `rays` in turn, cycling, once per iteration, each after a write to a
    /// vertex in the middle of the tile it hits and that tile's evaluation (both untimed), the
    /// index refitted to it: every raycast meets the store just written, as a stroke's do after
    /// each dab. The write moves the vertex by a millionth and back, in turn.
    ///
    /// # Panics
    ///
    /// If a ray misses, or a write, an evaluation or a raycast fails, which would be a bug.
    fn bench_edited(mut self, bencher: divan::Bencher<'_, '_>, rays: &[Ray]) {
        let targets: Vec<(Ray, TileKey)> = rays
            .iter()
            .map(|ray| {
                let hit = self
                    .index
                    .raycast(&mut self.store, ray)
                    .expect("raycast")
                    .expect("the ray hits");
                (*ray, hit.key)
            })
            .collect();
        self.store.begin_op().expect("op");
        let fresh = RefCell::new(self);
        let mut next = targets.iter().cycle();
        let mut lift = 0.0;
        bencher
            .with_inputs(|| {
                let &(ray, key) = next.next().expect("a cycle of rays");
                let mut fresh = fresh.borrow_mut();
                let Self { store, index, .. } = &mut *fresh;
                let middle = TILE_QUADS / 2;
                let (i, j) = (
                    u32::from(key.tx()) * TILE_QUADS + middle,
                    u32::from(key.ty()) * TILE_QUADS + middle,
                );
                lift = if lift == 0.0 { 1e-6 } else { 0.0 };
                store
                    .set_correction(key.face(), key.level(), i, j, Vec3::new(0.0, 0.0, lift))
                    .expect("write");
                store.evaluate(key).expect("evaluate");
                index.update(store, &[key]).expect("refit");
                ray
            })
            .bench_local_values(|ray| {
                let mut fresh = fresh.borrow_mut();
                let Self { store, index, .. } = &mut *fresh;
                index.raycast_triangle(store, &ray).expect("raycast")
            });
    }
}

impl Drop for Fresh {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// A small deterministic generator (`SplitMix64`), so every run casts the same rays.
struct Rng(u64);

impl Rng {
    /// A number in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        #[expect(
            clippy::cast_precision_loss,
            reason = "24 random bits, which an f32 holds exactly"
        )]
        let bits = (z >> 40) as f32;
        bits / 16_777_216.0
    }

    /// A number in `[low, high)`.
    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// A unit vector, about uniformly spread.
    ///
    /// # Panics
    ///
    /// Never in practice: a point of the cube that rounds to zero length is drawn again.
    fn direction(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if let Some(unit) = p.normalized() {
                return unit;
            }
        }
    }
}

/// A unit vector at right angles to `v`.
///
/// # Panics
///
/// If `v` is zero, which no caller passes.
fn across(v: Vec3) -> Vec3 {
    let other = if v.x.abs() < 0.5 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    v.cross(other).normalized().expect("v is a unit vector")
}

/// [`RAYS`] rays, each from `aim(rng, toward)` (an origin and a direction), `toward` a unit
/// direction drawn afresh for each ray.
///
/// # Panics
///
/// If `aim` gives an origin or direction a ray can't have, which the benches don't.
fn rays(seed: u64, mut aim: impl FnMut(&mut Rng, Vec3) -> (Vec3, Vec3)) -> Vec<Ray> {
    let mut rng = Rng(seed);
    (0..RAYS)
        .map(|_| {
            let toward = rng.direction();
            let (origin, direction) = aim(&mut rng, toward);
            Ray::new(origin, direction).expect("ray")
        })
        .collect()
}

/// The sphere's radius about in direction `toward`: how far from the centre a ray from near
/// the centre leaves it. The ray starts a little off the centre, off the planes the faces'
/// seams lie in.
///
/// # Panics
///
/// If the ray misses the closed surface, which would be a bug.
fn radius(fresh: &mut Fresh, toward: Vec3) -> f32 {
    let ray = Ray::new(Vec3::new(1e-3, 2e-3, 3e-3), toward).expect("ray");
    let hit = fresh
        .index
        .raycast(&mut fresh.store, &ray)
        .expect("raycast")
        .expect("the ray leaves the closed surface");
    ray.at(hit.t).length()
}

/// Rays from all around, each aimed near the centre: every one hits, at other tiles, on first
/// sight.
#[divan::bench(sample_count = 50, sample_size = 512)]
fn raycast_spread(bencher: divan::Bencher<'_, '_>) {
    let rays = rays(1, |rng, toward| {
        let aim = -toward + Vec3::new(rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), 0.0);
        (toward * 3.0, aim)
    });
    Fresh::new().bench(bencher, &rays);
}

/// Rays from one side, aimed within a patch a hundredth of the radius across, at `fresh`'s
/// sphere: a few tiles.
///
/// # Panics
///
/// Never: the direction is a fixed nonzero vector.
fn local_rays(fresh: &mut Fresh) -> Vec<Ray> {
    let toward = Vec3::new(0.3, 0.5, 0.8).normalized().expect("nonzero");
    let r = radius(fresh, toward);
    let side = across(toward);
    let up = toward.cross(side);
    rays(2, |rng, _| {
        let target = toward * r + side * rng.range(-0.01, 0.01) + up * rng.range(-0.01, 0.01);
        let origin = toward * 3.0;
        (origin, target - origin)
    })
}

/// [`local_rays`], raycast again and again with nothing written between: each raycast after
/// the first few finds its tiles' blocks kept.
#[divan::bench(sample_count = 50, sample_size = 512)]
fn raycast_local(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let rays = local_rays(&mut fresh);
    fresh.bench(bencher, &rays);
}

/// [`local_rays`], each after a write to the tile it hits ([`Fresh::bench_edited`]): a
/// stroke's raycasts, each after a dab, its tiles' blocks built again.
#[divan::bench(sample_count = 2000, sample_size = 1)]
fn raycast_local_edited(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let rays = local_rays(&mut fresh);
    fresh.bench_edited(bencher, &rays);
}

/// Rays from one side grazing the sphere at one place on its rim, within 2 % of the radius
/// inside it, each after a write to the tile it hits ([`Fresh::bench_edited`]): a stroke along
/// the silhouette.
///
/// # Panics
///
/// Never: the directions are fixed nonzero vectors.
#[divan::bench(sample_count = 2000, sample_size = 1)]
fn raycast_grazing_edited(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let toward = Vec3::new(0.3, 0.5, 0.8).normalized().expect("nonzero");
    let side = across(toward);
    let up = toward.cross(side);
    let r = radius(&mut fresh, side);
    let rays = rays(5, |rng, _| {
        let origin = toward * 3.0 + side * r * rng.range(0.98, 1.0) + up * rng.range(-0.01, 0.01);
        (origin, -toward)
    });
    fresh.bench_edited(bencher, &rays);
}

/// Rays from all around grazing the sphere, on first sight: the worst case.
///
/// They pass within 2 % of the radius inside the rim, nearly along the surface where they hit
/// it, each at other tiles: a long run of quads close to the ray.
#[divan::bench(sample_count = 50, sample_size = 512)]
fn raycast_grazing(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let rays = rays(3, |rng, toward| {
        let side = across(toward);
        let r = radius(&mut fresh, side);
        (toward * 3.0 + side * r * rng.range(0.98, 1.0), -toward)
    });
    fresh.bench(bencher, &rays);
}

/// Rays from all around passing just outside the rim, within 2 % of the radius, on first
/// sight: they walk candidate tiles and hit none.
#[divan::bench(sample_count = 50, sample_size = 512)]
fn raycast_miss(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let rays = rays(4, |rng, toward| {
        let side = across(toward);
        let r = radius(&mut fresh, side);
        (toward * 3.0 + side * r * rng.range(1.005, 1.02), -toward)
    });
    fresh.bench(bencher, &rays);
}
