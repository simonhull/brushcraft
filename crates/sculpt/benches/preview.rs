//! A strong stroke shown ahead of its finest work: the raycasts placement makes over its previews,
//! and showing its next dab.
//!
//! The mesh is `cube_sphere(4)` at level 8 (1,536 tiles of 64 × 64 quads, a quad about 0.0015
//! across the unit sphere). A stroke of [`RIDGE_DABS`] dabs runs along a great circle, every dab
//! shown ([`Preview::show`], two at a time, as a brush whose finest work lags) and none landed:
//! radius [`RADIUS`] (about 130 quads, some 20 tiles a dab, as a large brush on the medium
//! project), strength a tenth of it, spacing 0.15 of it, each dab's centre placed on the stroke's
//! own surface by a raycast, as placement places it. The previews stand as a ridge about two thirds
//! of a radius tall, its flanks steep.
//!
//! - [`raycast_ridge`]: [`SpatialIndex::raycast_surface`] from a camera off to the ridge's side, so
//!   its far flank is seen edge on, aimed around the ridge's end, where a pen places the next dabs:
//!   the raycasts placement makes, about a dozen a dab. The app's strong stroke on the medium
//!   project costs several times as much a raycast (50–70 µs, 2026-11-01): its previews' flanks run
//!   along its rays over far more blocks than this smooth ridge's, each tested whole.
//! - [`show_ridge`]: [`Preview::show`] of the stroke's next dab, the ridge growing a dab a sample.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{Level, Vec3, cube_sphere};
use sculpt::Preview;
use store::{Project, ProjectConfig, Ray, SpatialIndex, StoreConfig};
use stroke::Dab;

fn main() {
    divan::main();
}

/// The level edited.
const LEVEL: u8 = 8;

/// The dabs' radius, on the unit sphere.
const RADIUS: f32 = 0.2;

/// The dabs of the ridge shown before the benches.
const RIDGE_DABS: u32 = 60;

/// Rays in [`raycast_ridge`]'s cycle.
const RAYS: usize = 256;

/// A project over the sphere with a ridge shown on it, in its own temporary directory, removed on
/// drop.
struct Ridge {
    /// The project.
    project: Project,
    /// Its spatial index at [`LEVEL`].
    index: SpatialIndex,
    /// The ridge's previews.
    preview: Preview,
    /// The directory holding its recovery file.
    dir: PathBuf,
    /// The dabs placed so far.
    placed: u32,
}

impl Ridge {
    /// The project, its index, and [`RIDGE_DABS`] dabs shown.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-preview-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let config = ProjectConfig {
            store: StoreConfig::new(1 << 30, 1),
            recovery_dir: dir.join("recovery"),
        };
        let mut project = Project::new(cube_sphere(4).expect("sphere"), config).expect("project");
        let level = Level::new(LEVEL).expect("level");
        let index = SpatialIndex::build(project.store(), level, |_, _| {}).expect("index");
        let mut ridge = Self {
            project,
            index,
            preview: Preview::new(),
            dir,
            placed: 0,
        };
        while ridge.placed < RIDGE_DABS {
            let pair = [ridge.next_dab(), ridge.next_dab()];
            ridge
                .preview
                .show(ridge.project.store(), &mut ridge.index, &pair)
                .expect("show");
        }
        ridge
    }

    /// The direction of dab `n` from the sphere's centre: along a great circle, 0.15 radii a dab.
    ///
    /// # Panics
    ///
    /// Never: the direction is never zero.
    fn direction(n: u32) -> Vec3 {
        #[expect(clippy::cast_precision_loss, reason = "a few hundred dabs")]
        let angle = 0.15 * RADIUS * n as f32;
        let (sin, cos) = angle.sin_cos();
        Vec3::new(0.6 * cos, 0.6 * sin, 0.8)
            .normalized()
            .expect("nonzero")
    }

    /// The next dab: its centre on the stroke's surface (previews in place) in its direction,
    /// the outward normal, full strength.
    ///
    /// # Panics
    ///
    /// If the raycast fails or misses the closed surface, which would be a bug.
    fn next_dab(&mut self) -> Dab {
        let direction = Self::direction(self.placed);
        let ray = Ray::new(direction * 3.0, direction * -1.0).expect("ray");
        let hit = self
            .index
            .raycast_surface(self.project.store(), &ray)
            .expect("raycast")
            .expect("the ray meets the closed surface");
        let centre = ray.at(hit.t);
        let dab = Dab {
            position: [centre.x, centre.y, centre.z],
            normal: [direction.x, direction.y, direction.z],
            tangent: [0.0, 0.0, 1.0],
            radius: RADIUS,
            strength: 0.1 * RADIUS,
            hardness: 0.0,
            seq: self.placed,
            ..Dab::default()
        };
        self.placed += 1;
        dab
    }

    /// Rays from a camera off to the ridge's side and above it, aimed at its last radius and
    /// a radius to either side, as a pen there casts them.
    ///
    /// # Panics
    ///
    /// Never: the path moves, and the camera is off the surface.
    fn rays(&self) -> Vec<Ray> {
        let end = Self::direction(self.placed);
        let along = (Self::direction(self.placed + 1) - end)
            .normalized()
            .expect("the path moves");
        let side = end.cross(along).normalized().expect("across the path");
        let camera = end * 1.6 + side * 1.2;
        (0..RAYS)
            .map(|k| {
                #[expect(clippy::cast_precision_loss, reason = "a few hundred rays")]
                let (a, b) = ((k % 16) as f32 / 15.0, (k / 16) as f32 / 15.0 - 0.5);
                let target = end - along * (RADIUS * a) + side * (2.0 * RADIUS * b);
                Ray::new(camera, target - camera).expect("ray")
            })
            .collect()
    }
}

impl Drop for Ridge {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// [`SpatialIndex::raycast_surface`] at the ridge's end from its side: see the module notes.
///
/// # Panics
///
/// If a raycast fails, which would be a bug.
#[divan::bench(sample_count = 50, sample_size = 256)]
fn raycast_ridge(bencher: divan::Bencher<'_, '_>) {
    let mut ridge = Ridge::new();
    let rays = ridge.rays();
    let mut k = 0;
    bencher.bench_local(|| {
        let ray = rays.get(k % rays.len()).expect("a ray");
        k += 1;
        ridge
            .index
            .raycast_surface(ridge.project.store(), ray)
            .expect("raycast")
    });
}

/// [`Preview::show`] of the ridge's next dab: see the module notes. Placing the dab (a raycast)
/// is left out of the time.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 100, sample_size = 1)]
fn show_ridge(bencher: divan::Bencher<'_, '_>) {
    let ridge = RefCell::new(Ridge::new());
    bencher
        .with_inputs(|| ridge.borrow_mut().next_dab())
        .bench_local_values(|dab| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .show(ridge.project.store(), &mut ridge.index, &[dab])
                .expect("show")
        });
}
