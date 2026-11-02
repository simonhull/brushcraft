//! A strong stroke shown ahead of its finest work: the raycasts placement makes over its previews,
//! and showing its next dab.
//!
//! The mesh is `cube_sphere(4)` at level 8 (1,536 tiles of 64 × 64 quads, a quad about 0.0015
//! across the unit sphere), every level sculpted with an octave of noise as the performance
//! suite's projects are ([`sculpt_octave`]). A stroke of [`RIDGE_DABS`] dabs runs along a great
//! circle, every dab shown ([`Preview::show`], two at a time, as a brush whose finest work lags)
//! and none landed: radius [`RADIUS`] (about 130 quads, some 20 tiles a dab, as a large brush on
//! the medium project), strength a tenth of it, spacing 0.15 of it, each dab's centre placed on
//! the stroke's own surface by a raycast, as placement places it. The previews stand as a ridge
//! about nine tenths of a radius tall, its flanks steep.
//!
//! The noise is what makes the ridge as hard to raycast as the app's strong stroke. The draw
//! action moves each vertex along its tangent frame's normal, which the level below gives, and
//! the noise turns those normals by a few degrees over a few quads: where they converge, a
//! height past their radius of curvature (about half the dab's radius here) folds the surface
//! over, about 1.5 % of the previews' triangles (2026-11-02; the app's strong stroke on the
//! medium project folds 1.1 %, up to a fifth of a tile). A folded triangle faces every ray both
//! ways at once, and a flank seen edge on faces the rays along it so, and the raycast tests their
//! blocks whole. Over the smooth sphere the ridge folds nowhere, and a raycast costs a fifth of
//! what it does over the noise (5.1 against 26.6 µs, 2026-11-02; the app's, 50–70 µs).
//!
//! - [`raycast_ridge`]: [`SpatialIndex::raycast_surface`] from a camera off to the ridge's side, so
//!   its far flank is seen edge on, aimed around the ridge's end, where a pen places the next dabs:
//!   the raycasts placement makes, about a dozen a dab.
//! - [`show_ridge`]: [`Preview::show`] of the stroke's next dab, the ridge growing a dab a sample.
//! - [`dab_ridge`]: a whole dab as the app makes it, placed, shown and raycast around: the previews
//!   change every dab, so their raycast blocks are built afresh each time (about ten tiles a dab,
//!   20 µs each, 2026-11-02), which [`raycast_ridge`] leaves out. A raycast's saving that needs
//!   bounds built on top of a preview's blocks has to pay for itself within a dab.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use sculpt::Preview;
use store::{Evaluation, Project, ProjectConfig, Ray, SpatialIndex, StoreConfig};
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
        for octave in 1..=LEVEL {
            sculpt_octave(&mut project, Level::new(octave).expect("level"));
        }
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

/// The noise's amplitude at level 1, each level's octave half the one below's.
const AMPLITUDE: f32 = 0.02;

/// Sculpts `level`'s octave of the noise into every tile of it, as `brushcraft-bench generate`
/// sculpts the performance suite's projects.
///
/// Each vertex the tile owns is pushed along its frame's normal by
/// `AMPLITUDE · 2^−(L−1) · noise(p · f_L)`, `p` its position with the levels below sculpted and
/// `f_L = 4 · 2^L / 2π` (features about four quads of the level across).
///
/// # Panics
///
/// If the store refuses a tile of its own mesh.
fn sculpt_octave(project: &mut Project, level: Level) {
    let l = i32::from(level.get());
    let amplitude = AMPLITUDE * 0.5_f32.powi(l - 1);
    let frequency = 4.0 * 2.0_f32.powi(l) / std::f32::consts::TAU;
    let per_side = u16::try_from(level.tiles_per_side()).expect("tiles a side");
    let faces = project.store().mesh().face_count();
    let keys: Vec<TileKey> = (0..faces)
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    TileKey::displacement(FaceId::new(face), level, tx, ty).expect("tile")
                })
            })
        })
        .collect();
    project.begin_stroke().expect("stroke");
    let store = project.store();
    store
        .evaluate_many(&keys, Evaluation::Positions)
        .expect("evaluate");
    for key in keys {
        let positions = store.evaluate(key).expect("evaluate").to_vec();
        let side = key.side();
        let topology = store.mesh().topology();
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let writes: Vec<(u32, u32, f32)> = grid
            .zip(&positions)
            .filter(|&((i, j), _)| topology.owns(&key, i, j))
            .map(|((i, j), &p)| (i, j, amplitude * value_noise(p * frequency, l)))
            .collect();
        store
            .edit_tile(key, |edit| {
                writes
                    .iter()
                    .try_for_each(|&(i, j, z)| edit.set(i, j, Vec3::new(0.0, 0.0, z)))
            })
            .expect("edit")
            .expect("set");
    }
    project.end_unrecorded_stroke().expect("stroke");
}

/// Value noise in 3D, in `[-1, 1]`: a random value at every point of the integer lattice (of
/// octave `octave`), blended between the eight around `p` with smoothstep weights.
fn value_noise(p: Vec3, octave: i32) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unit-sphere points times at most a few hundred"
    )]
    let split = |x: f32| (x.floor() as i32, x - x.floor());
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let ((x0, fx), (y0, fy), (z0, fz)) = (split(p.x), split(p.y), split(p.z));
    let lattice = |dx: i32, dy: i32, dz: i32| {
        let key = [x0 + dx, y0 + dy, z0 + dz, octave].into_iter().fold(
            0x9E37_79B9_7F4A_7C15_u64,
            |h, k| {
                let mut z = h ^ u64::from(k.cast_unsigned());
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            },
        );
        #[expect(clippy::cast_precision_loss, reason = "24 bits are exact in an f32")]
        let bits = (key >> 40) as f32;
        bits / 8_388_608.0 - 1.0
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let along_x = |dy, dz| lerp(lattice(0, dy, dz), lattice(1, dy, dz), smooth(fx));
    let along_y = |dz| lerp(along_x(0, dz), along_x(1, dz), smooth(fy));
    lerp(along_y(0), along_y(1), smooth(fz))
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

/// Raycasts a dab makes in [`dab_ridge`]: about as many as placement makes a dab in the app.
const RAYS_A_DAB: usize = 14;

/// A dab of the ridge as the app's strong stroke makes it.
///
/// Placed by a raycast, shown ([`Preview::show`]), then [`RAYS_A_DAB`] raycasts more around the
/// ridge's new end, from its side, as placement makes for the next. Each raycast after a show
/// builds the blocks of the previews it meets afresh, which [`raycast_ridge`], over previews that
/// stay, leaves out.
///
/// # Panics
///
/// If placing, showing or a raycast fails, which would be a bug.
#[divan::bench(sample_count = 100, sample_size = 1)]
fn dab_ridge(bencher: divan::Bencher<'_, '_>) {
    let mut ridge = Ridge::new();
    bencher.bench_local(|| {
        let dab = ridge.next_dab();
        ridge
            .preview
            .show(ridge.project.store(), &mut ridge.index, &[dab])
            .expect("show");
        let rays = ridge.rays();
        let every = rays
            .iter()
            .step_by(rays.len() / RAYS_A_DAB)
            .take(RAYS_A_DAB);
        for ray in every {
            ridge
                .index
                .raycast_surface(ridge.project.store(), ray)
                .expect("raycast");
        }
    });
}
