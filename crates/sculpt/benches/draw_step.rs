//! One draw step over a 50,000-vertex footprint: the M0 benchmark's brush step, on a mesh small
//! enough to set up in a benchmark run.
//!
//! The mesh is the benchmark's sphere at 1,536 faces (`cube_sphere(16)`), edited at level 8:
//! 24,576 tiles, 100 million vertices, 2.5 GB evaluated: more than the 2 GiB budget holds at
//! once. The
//! project and its spatial index are built once; the steps then run one after another in a
//! single stroke, along a path on the surface, so each one finds the tiles the step before it
//! edited still resident.
//!
//! The radius is set where the path starts to cover about 50,000 vertices. The vertices are
//! denser towards the cube's edges, so the footprint grows along the path: the bench prints its
//! mean and range over the steps it ran, with the mean time of each phase (see
//! `sculpt::PhaseTimes`), besides divan's timings.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use geom::{Level, Vec3, cube_sphere};
use sculpt::{DrawParams, PhaseTimes, StepStats, draw_step};
use store::{Project, ProjectConfig, Ray, SpatialIndex, StoreConfig};

fn main() {
    divan::main();
}

/// `cube_sphere`'s splits: `6 · 16²` = 1,536 faces.
const SPLITS: u32 = 16;

/// The level edited.
const LEVEL: u8 = 8;

/// The footprint aimed for, in vertices.
const FOOTPRINT: usize = 50_000;

/// How far the path moves the brush each step, in radii.
const STEP_RADII: f32 = 0.1;

/// The draw brush's strength: small, so the surface stays close to the sphere however many
/// steps the bench runs.
const STRENGTH: f32 = 1e-4;

/// A project over the sphere, in its own temporary directory, removed on drop.
struct Fresh {
    /// The project.
    project: Project,
    /// Its spatial index at [`LEVEL`].
    index: SpatialIndex,
    /// The directory holding its recovery file.
    dir: PathBuf,
}

impl Fresh {
    /// Makes the project, under a 2 GiB budget, and builds its index.
    ///
    /// The budget doesn't hold every tile evaluated (24,576 of 101 KB, 2.5 GB), so the build
    /// evicts as it goes, as it would on the benchmark sphere; it holds a step's tiles many
    /// times over, so the stroke's steps find what the steps before them evaluated.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-sculpt-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let config = ProjectConfig {
            store: StoreConfig {
                budget_bytes: 2 << 30,
                chunk_bytes: StoreConfig::DEFAULT_CHUNK_BYTES,
                site: 1,
            },
            recovery_dir: dir.join("recovery"),
        };
        let mut project =
            Project::new(cube_sphere(SPLITS).expect("sphere"), config).expect("project");
        let level = Level::new(LEVEL).expect("level");
        let index = SpatialIndex::build(project.store(), level, |_, _| {}).expect("index");
        Self {
            project,
            index,
            dir,
        }
    }
}

impl Drop for Fresh {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// The radius that covers about [`FOOTPRINT`] vertices around `center`, found by bisection on
/// the count of evaluated positions within it.
///
/// The count is of tile copies (a vertex on a tile border counts once per tile holding it, a
/// few percent more), so the step's own count, which the bench prints, is the one to quote.
///
/// # Panics
///
/// If a tile can't be evaluated, which would be a bug.
fn calibrated_radius(fresh: &mut Fresh, center: Vec3) -> f32 {
    let mut count = |radius: f32| {
        let keys: Vec<_> = fresh.index.query_sphere(center, radius).collect();
        let store = fresh.project.store();
        keys.iter()
            .map(|&key| {
                let positions = store.evaluate(key).expect("evaluate");
                positions
                    .iter()
                    .filter(|&&p| {
                        let d = p - center;
                        d.dot(d) < radius * radius
                    })
                    .count()
            })
            .sum::<usize>()
    };
    let (mut low, mut high) = (0.0_f32, 0.5_f32);
    for _ in 0..30 {
        let mid = f32::midpoint(low, high);
        if count(mid) < FOOTPRINT {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

/// Steps of the brush's path, after which it starts over.
const PATH_STEPS: u16 = 1000;

/// The direction from the sphere's centre to the brush at step `k`: along a circle of latitude,
/// [`STEP_RADII`] radii a step, starting inside a face away from its corners.
///
/// # Panics
///
/// Never: the direction is never zero.
fn direction(k: u16, radius: f32) -> Vec3 {
    let start = Vec3::new(0.8, 0.35, 0.45);
    let angle = STEP_RADII * radius * f32::from(k);
    let (sin, cos) = angle.sin_cos();
    let p = Vec3::new(
        start.x * cos - start.y * sin,
        start.x * sin + start.y * cos,
        start.z,
    );
    p.normalized().expect("nonzero")
}

/// Where the surface lies in `direction` from the sphere's centre: the brush's centre there.
///
/// The surface is close to the unit sphere but not on it, and a brush centred off the surface
/// covers a smaller cap, so the centres are put on it.
///
/// # Panics
///
/// If the ray misses the closed surface, which would be a bug.
fn on_surface(fresh: &mut Fresh, direction: Vec3) -> Vec3 {
    let ray = Ray::new(Vec3::ZERO, direction).expect("ray");
    let hit = fresh
        .index
        .raycast(fresh.project.store(), &ray)
        .expect("raycast")
        .expect("the ray leaves the closed surface");
    ray.at(hit.t)
}

/// Prints the steps' footprints (mean and range of vertices moved, mean tiles) and the mean
/// time of each phase.
///
/// # Panics
///
/// Never: the step count fits a `u32`, and nothing is printed for no steps.
#[expect(
    clippy::print_stderr,
    reason = "the per-phase breakdown is part of the bench's output"
)]
fn report(all: &[StepStats]) {
    let Some(steps) = u32::try_from(all.len()).ok().filter(|&n| n > 0) else {
        return;
    };
    let mut times = PhaseTimes::default();
    for stats in all {
        times += stats.times;
    }
    let mean = |d: Duration| d / steps;
    let count = |f: fn(&StepStats) -> usize| all.iter().map(f).sum::<usize>() / all.len();
    let vertices = all.iter().map(|stats| stats.vertices);
    eprintln!(
        "draw_step_50k: {steps} steps; mean {} vertices (from {} to {}), {} candidate tiles, {} \
         edited, {} updated",
        count(|stats| stats.vertices),
        vertices.clone().min().unwrap_or(0),
        vertices.max().unwrap_or(0),
        count(|stats| stats.candidates),
        count(|stats| stats.tiles_edited),
        count(|stats| stats.tiles_updated),
    );
    eprintln!(
        "  mean per step: query {:?}, pin {:?}, brush {:?}, edit {:?}, neighbours {:?}, evaluate \
         {:?}, refit {:?}, total {:?}",
        mean(times.query),
        mean(times.pin),
        mean(times.brush),
        mean(times.edit),
        mean(times.neighbours),
        mean(times.evaluate),
        mean(times.refit),
        mean(times.total()),
    );
}

/// One [`draw_step`] of about 50,000 vertices at level 8 of the 1,536-face sphere, inside one
/// stroke, the brush moving a tenth of its radius each step.
///
/// # Panics
///
/// If the setup is rejected or a step fails, which would be a bug.
#[divan::bench(sample_count = 50)]
fn draw_step_50k(bencher: divan::Bencher<'_, '_>) {
    let mut fresh = Fresh::new();
    let start = on_surface(&mut fresh, direction(0, 0.0));
    let radius = calibrated_radius(&mut fresh, start);
    let path: Vec<Vec3> = (0..PATH_STEPS)
        .map(|k| on_surface(&mut fresh, direction(k, radius)))
        .collect();
    fresh.project.begin_stroke().expect("stroke");
    let mut all: Vec<StepStats> = Vec::new();
    bencher.bench_local(|| {
        let center = path
            .get(all.len() % path.len())
            .copied()
            .expect("a path of PATH_STEPS centres");
        let params = DrawParams::new(center, radius, STRENGTH).expect("params");
        let stats = draw_step(&mut fresh.project, &mut fresh.index, params).expect("draw step");
        all.push(stats);
    });
    fresh.project.end_stroke().expect("stroke");
    report(&all);
}
