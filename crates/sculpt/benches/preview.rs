//! A strong stroke's ridge: the raycasts and area normals placement makes over the stroke's
//! pen-down surface, and showing its next dab ahead of its finest work.
//!
//! The mesh is `cube_sphere(4)` at level 8 (1,536 tiles of 64 × 64 quads, a quad about 0.0015
//! across the unit sphere), every level sculpted with an octave of noise as the performance
//! suite's projects are ([`sculpt_octave`]). One stroke of [`RIDGE_DABS`] dabs runs along a great
//! circle, two at a time applied through [`apply_dabs`]: radius [`RADIUS`] (about 130 quads, some
//! 20 tiles a dab, as a large brush on the medium project), strength a tenth of it, spacing 0.15
//! of it, each dab's centre placed on the stroke's pen-down surface by a raycast and its normal its
//! area normal there ([`Surface::area_normal`] over [`StoreSurface`]), as placement places them.
//! The ridge stands some six tenths of a radius tall, and the stroke stays open: every tile it
//! changed keeps its pen-down shape, which placement reads (made into positions, with raycast
//! blocks built for the call) in place of the ridge.
//!
//! Before placement read the pen-down surface (2026-11-02), these benches measured placement over
//! the stroke's own surface, the ridge shown as previews: the same work a dab, over the other
//! surface.
//!
//! - [`raycast_ridge`]: [`SpatialIndex::raycast_pen_down`] from a camera off to the ridge's side,
//!   aimed around the ridge's end, where a pen places the next dabs: the raycasts placement makes,
//!   about a dozen a dab.
//! - [`show_ridge`]: [`Preview::show`] of the stroke's next dab, over the ridge, each point weighed
//!   at pen-down.
//! - [`area_normal_ridge`]: [`Surface::area_normal`] of the ridge's next dab: the per-dab work the
//!   area normal adds to placement, over the changed tiles' pen-down shapes.
//! - [`dab_ridge`]: a whole dab as the app makes it, placed, shown and raycast around.
//! - [`big_dab_ridge`]: a whole dab of a big brush (radius [`BIG_RADIUS`], about 300 tiles), placed
//!   and applied, on a ridge of its dabs: more changed tiles a dab than the store keeps placement's
//!   pen-down positions made for.
//! - [`area_normal_cold`]: [`Surface::area_normal`] of a big dab (about 70 tiles, [`FAR_RADIUS`])
//!   where nothing has been evaluated yet: what placement pays as a big stroke moves onto fresh
//!   ground.
//! - [`show_far`]: [`Preview::show`] of the next dab of a stroke whose earlier dabs are all still
//!   shown (none landed: the catch-up far behind), at about 70 tiles a dab ([`FAR_RADIUS`]) and for
//!   the big brush (about 300 tiles, past a million vertices), drawn as a renderer far off draws
//!   it: on the finest tiles alone (the preview before display tiles, every tile the dab reaches
//!   moved and its display levels left to stand-ins), or on the display tiles of a coarse cut
//!   ([`CUT`] levels below the finest) and every level above it, no finest tile drawn.
//! - [`show_first_of_many`]: [`Preview::show`] of the first dab shown of a stroke that has changed
//!   more than a thousand tiles, every dab before it applied at once: every display preview it
//!   reaches made there and then, replaying the stroke and gathering what it changed.
//! - [`show_late_in_stroke`] and [`land_late_in_stroke`]: [`Preview::show`] of a dab, and
//!   [`Preview::land`] of the oldest, with [`LATE_STROKE`]'s dabs shown before them: what a dab
//!   pays for the backlog, on the finest tiles and on display tiles.
//! - [`note_late_in_stroke`]: [`Preview::note_applied`] of one dab with [`LONG_STROKE`]'s dabs
//!   before it: what a dab pays for the stroke's length in `push_stroke`.
//! - [`over_first`], [`over_first_noted`] and [`over_steady`]: [`Preview::show_over`] of a dab of a
//!   small stroke drawn along a small stroke's ridge still landing ([`Over`]: both of
//!   [`LATE_RADIUS`], the earlier one's [`OVER_DABS`] dabs shown and none landed, the later one's
//!   op reserved over its futures), drawn at every level: its first dab, where its previews first
//!   meet the earlier stroke's and the seams between them are made (with what the stitch looks up
//!   noted ahead, as under a hovering pen, or not), and a dab well into it, where they are kept.
//!   [`alone_first`] and [`alone_steady`] are the same dabs of a stroke with no other under way:
//!   what the showing costs without the seams.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use sculpt::{
    DrawsAll, Futures, OpenStrokes, PenDownTiles, Preview, StoreSurface, Stroke, StrokeEdits,
    apply_dabs,
};
use store::{
    CacheError, Evaluation, OpId, PenDownReading, Project, ProjectConfig, Ray, ReservedOp,
    SpatialIndex, StoreConfig, TileStore,
};
use stroke::{DVec3, Dab, Surface};

fn main() {
    divan::main();
}

/// The level edited.
const LEVEL: u8 = 8;

/// The dabs shown before the one timed in [`show_late_in_stroke`], [`land_late_in_stroke`].
const LATE_STROKE: [usize; 2] = [1_370, 5_000];

/// The radius of those dabs: the benchmark's small brush, a few tiles a dab.
const LATE_RADIUS: f32 = 0.05;

/// The earlier stroke's dabs in the benches of a stroke over one still landing ([`Over`]).
const OVER_DABS: u32 = 120;

/// Which of the earlier stroke's dabs the later stroke begins on ([`Over`]): the ridge's middle.
const OVER_FROM: u32 = 60;

/// The later stroke's dabs shown before the one [`over_steady`] times.
const OVER_SHOWN: u32 = 10;

/// The dabs before the one noted in [`note_late_in_stroke`].
const LONG_STROKE: [usize; 3] = [100, 1_000, 10_000];

/// The dabs' radius, on the unit sphere.
const RADIUS: f32 = 0.2;

/// The dabs of the ridge shown before the benches.
const RIDGE_DABS: u32 = 60;

/// The big brush's radius, on the unit sphere: each dab reaches about 300 tiles, past the tiles
/// the store keeps pen-down positions made for placement.
const BIG_RADIUS: f32 = 0.9;

/// The radius of [`show_far`]'s smaller brush: about 70 tiles a dab.
const FAR_RADIUS: f32 = 0.35;

/// The radius of [`show_far`]'s big brush: about a million vertices, some 240 tiles a dab.
const MILLION_RADIUS: f32 = 0.73;

/// How many fresh places [`area_normal_cold`] spreads over the sphere: as many as its samples.
const COLD_PLACES: u32 = 12;

/// How many levels below the finest [`show_far`]'s cut on display tiles is: a big brush seen from
/// far enough that its 70 tiles fit a 40-pixel radius.
const CUT: u8 = 4;

/// The big brush's dabs applied before [`big_dab_ridge`].
const BIG_DABS: u32 = 8;

/// Rays in [`raycast_ridge`]'s cycle.
const RAYS: usize = 256;

/// A project over the sphere with a ridge drawn on it by a stroke still open, in its own
/// temporary directory, removed on drop.
struct Ridge {
    /// The project.
    project: Project,
    /// Its spatial index at [`LEVEL`].
    index: SpatialIndex,
    /// The dabs shown past the ridge.
    preview: Preview,
    /// The directory holding its recovery file.
    dir: PathBuf,
    /// The dabs placed so far.
    placed: u32,
    /// The surface normal at the last dab's hit: the side its area normal counts.
    facing: DVec3,
    /// The dabs' radius.
    radius: f32,
    /// The stroke's state, the stroke open.
    stroke: Stroke,
}

impl Ridge {
    /// The project, its index, and [`RIDGE_DABS`] dabs of [`RADIUS`] applied in an open stroke.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, or the temporary directory can't be made.
    fn new() -> Self {
        Self::of(RADIUS, RIDGE_DABS)
    }

    /// The project, its index, and `dabs` dabs of `radius` applied in an open stroke, two at a
    /// time.
    ///
    /// # Panics
    ///
    /// As [`Ridge::new`].
    fn of(radius: f32, dabs: u32) -> Self {
        Self::made(radius, dabs, false)
    }

    /// [`Ridge::of`], with its display layer built first if `display`.
    ///
    /// # Panics
    ///
    /// As [`Ridge::new`].
    fn made(radius: f32, dabs: u32, display: bool) -> Self {
        /// Distinguishes this process's directories.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "brushcraft-bench-preview-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temporary directory");
        let level = Level::new(LEVEL).expect("level");
        let store = StoreConfig::new(1 << 30, 1);
        let config = ProjectConfig {
            store: if display {
                store.with_display(level)
            } else {
                store
            },
            recovery_dir: dir.join("recovery"),
        };
        let mut project = Project::new(cube_sphere(4).expect("sphere"), config).expect("project");
        for octave in 1..=LEVEL {
            sculpt_octave(&mut project, Level::new(octave).expect("level"));
        }
        if display {
            project.store().queue_display_rebuild(level);
            while project.store().has_jobs() {
                project
                    .store()
                    .run_jobs(std::time::Instant::now() + std::time::Duration::from_millis(100))
                    .expect("the display layer builds");
            }
        }
        let index = SpatialIndex::build(project.store(), level, |_, _| {}).expect("index");
        let stroke = Stroke::new(project.store());
        let mut ridge = Self {
            project,
            index,
            preview: Preview::new(),
            dir,
            placed: 0,
            facing: DVec3::Z,
            radius,
            stroke,
        };
        ridge.project.begin_stroke_and_wait().expect("stroke");
        while ridge.placed < dabs {
            let pair = [ridge.next_dab(), ridge.next_dab()];
            ridge.apply(&pair);
        }
        ridge
    }

    /// Applies `dabs` in the ridge's stroke.
    ///
    /// # Panics
    ///
    /// If applying fails, which would be a bug.
    fn apply(&mut self, dabs: &[Dab]) {
        apply_dabs(
            self.project.store(),
            &mut self.index,
            &mut self.stroke,
            dabs,
        )
        .expect("apply");
    }

    /// The direction of dab `n` from the sphere's centre: along a great circle, 0.15 radii a dab.
    ///
    /// # Panics
    ///
    /// Never: the direction is never zero.
    fn direction(&self, n: u32) -> Vec3 {
        #[expect(clippy::cast_precision_loss, reason = "a few hundred dabs")]
        let angle = 0.15 * self.radius * n as f32;
        let (sin, cos) = angle.sin_cos();
        Vec3::new(0.6 * cos, 0.6 * sin, 0.8)
            .normalized()
            .expect("nonzero")
    }

    /// The next dab: its centre on the stroke's pen-down surface in its direction, its normal its
    /// area normal there, full strength.
    ///
    /// # Panics
    ///
    /// If the raycast fails or misses the closed surface, or the area normal can't be taken,
    /// which would be a bug.
    fn next_dab(&mut self) -> Dab {
        let direction = self.direction(self.placed);
        let ray = Ray::new(direction * 3.0, direction * -1.0).expect("ray");
        let ray = stroke::Ray::new(
            DVec3::from_f32(ray.origin()),
            DVec3::from_f32(ray.direction()),
        )
        .expect("ray");
        let mut surface = StoreSurface::new(self.project.store(), &self.index);
        let hit = surface
            .raycast(&ray)
            .expect("the ray meets the closed surface");
        let sum = surface.area_normal(hit.position, hit.normal, f64::from(self.radius), 0.0);
        surface.finish().expect("the raycast and the area normal");
        let centre = hit.position.to_f32();
        let normal = sum
            .and_then(DVec3::normalized)
            .map_or(direction, DVec3::to_f32);
        self.facing = hit.normal;
        let dab = Dab {
            position: [centre.x, centre.y, centre.z],
            normal: [normal.x, normal.y, normal.z],
            tangent: [0.0, 0.0, 1.0],
            radius: self.radius,
            strength: 0.1 * self.radius,
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
        let end = self.direction(self.placed);
        let along = (self.direction(self.placed + 1) - end)
            .normalized()
            .expect("the path moves");
        let side = end.cross(along).normalized().expect("across the path");
        let camera = end * 1.6 + side * 1.2;
        (0..RAYS)
            .map(|k| {
                #[expect(clippy::cast_precision_loss, reason = "a few hundred rays")]
                let (a, b) = ((k % 16) as f32 / 15.0, (k / 16) as f32 / 15.0 - 0.5);
                let target = end - along * (self.radius * a) + side * (2.0 * self.radius * b);
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
    project.begin_stroke_and_wait().expect("stroke");
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

/// [`SpatialIndex::raycast_pen_down`] at the ridge's end from its side: see the module notes.
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
            .raycast_pen_down(ridge.project.store(), ray)
            .expect("raycast")
    });
}

/// [`Preview::show`] of the ridge's next dab: see the module notes. Placing the dab (a raycast
/// and its area normal) is left out of the time.
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
                .show(ridge.project.store(), &ridge.index, &[dab], &DrawsAll)
                .expect("show")
        });
}

/// Raycasts a dab makes in [`dab_ridge`]: about as many as placement makes a dab in the app.
const RAYS_A_DAB: usize = 14;

/// A dab of the ridge as the app's strong stroke makes it.
///
/// Placed by a raycast and its area normal, shown ([`Preview::show`]), then [`RAYS_A_DAB`]
/// raycasts more around the ridge's new end, from its side, as placement makes for the next.
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
            .show(ridge.project.store(), &ridge.index, &[dab], &DrawsAll)
            .expect("show");
        let rays = ridge.rays();
        let every = rays
            .iter()
            .step_by(rays.len() / RAYS_A_DAB)
            .take(RAYS_A_DAB);
        for ray in every {
            ridge
                .index
                .raycast_pen_down(ridge.project.store(), ray)
                .expect("raycast");
        }
    });
}

/// [`Surface::area_normal`] of the ridge's next dab over [`StoreSurface`], just after its last dab
/// was shown (untimed): see the module notes.
///
/// # Panics
///
/// If showing or the area normal fails, which would be a bug.
#[divan::bench(sample_count = 100, sample_size = 1)]
fn area_normal_ridge(bencher: divan::Bencher<'_, '_>) {
    let ridge = RefCell::new(Ridge::new());
    bencher
        .with_inputs(|| {
            let ridge = &mut *ridge.borrow_mut();
            let dab = ridge.next_dab();
            ridge
                .preview
                .show(ridge.project.store(), &ridge.index, &[dab], &DrawsAll)
                .expect("show");
            let [x, y, z] = dab.position;
            (DVec3::from_f32(Vec3::new(x, y, z)), ridge.facing)
        })
        .bench_local_values(|(centre, facing)| {
            let ridge = &mut *ridge.borrow_mut();
            let mut surface = StoreSurface::new(ridge.project.store(), &ridge.index);
            let sum = surface.area_normal(centre, facing, f64::from(RADIUS), 0.0);
            surface.finish().expect("the area normal");
            sum
        });
}

/// A whole dab of a big brush, about 300 tiles, on a ridge of [`BIG_DABS`] of them: placed (a
/// raycast and its area normal over the stroke's changed tiles) and applied.
///
/// # Panics
///
/// If placing or applying fails, which would be a bug.
#[divan::bench(sample_count = 20, sample_size = 1)]
#[expect(
    clippy::print_stderr,
    reason = "a bench reports the footprint it measured beside divan's timings"
)]
fn big_dab_ridge(bencher: divan::Bencher<'_, '_>) {
    let mut ridge = Ridge::of(BIG_RADIUS, BIG_DABS);
    bencher.bench_local(|| {
        let dab = ridge.next_dab();
        ridge.apply(&[dab]);
    });
    let tiles = ridge
        .index
        .query_sphere(ridge.direction(ridge.placed), BIG_RADIUS)
        .count();
    eprintln!(
        "big_dab_ridge: {tiles} tiles a dab, {} changed, {:.1} MiB of pen-down data",
        ridge.project.store().pen_down_tiles(),
        f64::from(u32::try_from(ridge.project.store().pen_down_bytes()).unwrap_or(u32::MAX))
            / f64::from(1_u32 << 20)
    );
}

/// How [`show_far`] draws: the tiles a renderer holds.
#[derive(Debug, Clone, Copy)]
enum Drawing {
    /// The finest tiles alone.
    Finest,
    /// The display tiles [`CUT`] levels below the finest and above.
    Display,
}

impl Drawing {
    /// Whether it draws `key`.
    fn draws(self, key: &TileKey) -> bool {
        match self {
            Self::Finest => key.channel() != geom::Channel::DISPLAY,
            Self::Display => {
                key.channel() == geom::Channel::DISPLAY && key.level().get() + CUT <= LEVEL
            }
        }
    }
}

/// [`Preview::show`] of the first dab shown of a long big stroke: see the module notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 10, sample_size = 1)]
#[expect(
    clippy::print_stderr,
    reason = "a bench reports the footprint it measured beside divan's timings"
)]
fn show_first_of_many(bencher: divan::Bencher<'_, '_>) {
    let ridge = RefCell::new(Ridge::made(BIG_RADIUS, 60, true));
    let earlier: Vec<Dab> = {
        let ridge = &mut *ridge.borrow_mut();
        let dabs: Vec<Dab> = (0..40).map(|_| ridge.next_dab()).collect();
        ridge.apply(&dabs);
        dabs
    };
    bencher
        .with_inputs(|| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .end_stroke(ridge.project.store())
                .expect("end");
            ridge.preview = Preview::new();
            ridge
                .preview
                .note_applied(ridge.project.store(), &ridge.index, &earlier)
                .expect("note");
            ridge.next_dab()
        })
        .bench_local_values(|dab| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .show(
                    ridge.project.store(),
                    &ridge.index,
                    &[dab],
                    &|key: &TileKey| Drawing::Display.draws(key),
                )
                .expect("show")
        });
    let ridge = &mut *ridge.borrow_mut();
    let changed = ridge.project.store().pen_down_tiles();
    let displays = ridge.project.store().display_previewed().count();
    eprintln!("show_first_of_many: {changed} tiles changed, {displays} display previews");
}

/// A ridge of `shown` dabs shown as `drawing` draws them, none landed.
///
/// # Panics
///
/// If showing fails, which would be a bug.
fn shown_ridge(shown: usize, drawing: Drawing) -> (Ridge, Vec<Dab>) {
    let mut ridge = Ridge::made(LATE_RADIUS, 0, true);
    let mut dabs = Vec::with_capacity(shown);
    for _ in 0..shown {
        let dab = ridge.next_dab();
        ridge
            .preview
            .show(
                ridge.project.store(),
                &ridge.index,
                &[dab],
                &|key: &TileKey| drawing.draws(key),
            )
            .expect("show");
        dabs.push(dab);
    }
    (ridge, dabs)
}

/// [`Preview::show`] of a dab with many shown before it: see the module notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(args = LATE_STROKE, sample_count = 30, sample_size = 1)]
fn show_late_in_stroke(bencher: divan::Bencher<'_, '_>, shown: usize) {
    let ridge = RefCell::new(shown_ridge(shown, Drawing::Finest).0);
    bencher
        .with_inputs(|| ridge.borrow_mut().next_dab())
        .bench_local_values(|dab| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .show(
                    ridge.project.store(),
                    &ridge.index,
                    &[dab],
                    &|key: &TileKey| Drawing::Finest.draws(key),
                )
                .expect("show")
        });
}

/// [`Preview::land`] of a dab with many still shown behind it: see the module notes.
///
/// # Panics
///
/// If applying or landing fails, which would be a bug.
#[divan::bench(args = LATE_STROKE, sample_count = 30, sample_size = 1)]
fn land_late_in_stroke(bencher: divan::Bencher<'_, '_>, shown: usize) {
    let (ridge, dabs) = shown_ridge(shown, Drawing::Display);
    let ridge = RefCell::new(ridge);
    let next = std::cell::Cell::new(0);
    bencher
        .with_inputs(|| {
            let ridge = &mut *ridge.borrow_mut();
            let dab = dabs.get(next.replace(next.get() + 1)).expect("a dab");
            ridge.apply(std::slice::from_ref(dab));
        })
        .bench_local_values(|()| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .land(ridge.project.store(), &ridge.index, 1, &|key: &TileKey| {
                    Drawing::Display.draws(key)
                })
                .expect("land")
        });
}

/// [`Preview::note_applied`] of a dab late in a long stroke: see the module notes.
///
/// # Panics
///
/// If noting fails, which would be a bug.
#[divan::bench(args = LONG_STROKE, sample_count = 50, sample_size = 1)]
fn note_late_in_stroke(bencher: divan::Bencher<'_, '_>, dabs: usize) {
    let ridge = RefCell::new(Ridge::of(RADIUS, 2));
    let one = {
        let ridge = &mut *ridge.borrow_mut();
        let one = ridge.next_dab();
        let many = vec![one; dabs];
        ridge
            .preview
            .note_applied(ridge.project.store(), &ridge.index, &many)
            .expect("note");
        one
    };
    bencher.bench_local(|| {
        let ridge = &mut *ridge.borrow_mut();
        ridge
            .preview
            .note_applied(ridge.project.store(), &ridge.index, &[one])
            .expect("note");
    });
}

/// [`Preview::show`] of a stroke's next dab, the earlier ones all shown: see the module notes.
/// Its first eight dabs are shown before the timing, and each sample shows one more.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(
    sample_count = 30,
    sample_size = 1,
    args = [("70 tiles, finest", FAR_RADIUS, Drawing::Finest),
            ("70 tiles, display", FAR_RADIUS, Drawing::Display),
            ("1M vertices, finest", MILLION_RADIUS, Drawing::Finest),
            ("1M vertices, display", MILLION_RADIUS, Drawing::Display)]
)]
#[expect(
    clippy::print_stderr,
    reason = "a bench reports the footprint it measured beside divan's timings"
)]
fn show_far(bencher: divan::Bencher<'_, '_>, (what, radius, drawing): (&str, f32, Drawing)) {
    let ridge = RefCell::new(Ridge::made(radius, 0, true));
    {
        let ridge = &mut *ridge.borrow_mut();
        let first: Vec<Dab> = (0..8).map(|_| ridge.next_dab()).collect();
        ridge
            .preview
            .show(
                ridge.project.store(),
                &ridge.index,
                &first,
                &|key: &TileKey| drawing.draws(key),
            )
            .expect("show");
    }
    bencher
        .with_inputs(|| ridge.borrow_mut().next_dab())
        .bench_local_values(|dab| {
            let ridge = &mut *ridge.borrow_mut();
            ridge
                .preview
                .show(
                    ridge.project.store(),
                    &ridge.index,
                    &[dab],
                    &|key: &TileKey| drawing.draws(key),
                )
                .expect("show")
        });
    let ridge = &mut *ridge.borrow_mut();
    let tiles = ridge
        .index
        .query_sphere(ridge.direction(ridge.placed), radius)
        .count();
    let displays = ridge.project.store().display_previewed().count();
    eprintln!("show_far {what}: {tiles} tiles a dab, {displays} display previews");
}

/// [`Surface::area_normal`] of a dab of [`FAR_RADIUS`] over cold ground: see the module notes.
///
/// Every sample starts with the store's pools emptied ([`store::TileStore::evict_all`]), at a fresh
/// place: [`COLD_PLACES`] spread over the sphere by a golden spiral.
///
/// # Panics
///
/// If the area normal fails, which would be a bug.
#[divan::bench(sample_count = 12, sample_size = 1)]
fn area_normal_cold(bencher: divan::Bencher<'_, '_>) {
    let ridge = RefCell::new(Ridge::made(FAR_RADIUS, 0, false));
    let mut k = 0_u32;
    bencher
        .with_inputs(|| {
            k += 1;
            // The golden spiral's place `k` of `COLD_PLACES`.
            #[expect(clippy::cast_precision_loss, reason = "a dozen places")]
            let (n, total) = ((k % COLD_PLACES) as f32 + 0.5, COLD_PLACES as f32);
            let z = 1.0 - 2.0 * n / total;
            let around = n * std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
            let flat = (1.0 - z * z).max(0.0).sqrt();
            ridge
                .borrow_mut()
                .project
                .store()
                .evict_all()
                .expect("emptied");
            Vec3::new(flat * around.cos(), flat * around.sin(), z)
        })
        .bench_local_values(|direction| {
            let ridge = &mut *ridge.borrow_mut();
            let centre = DVec3::from_f32(direction);
            let mut surface = StoreSurface::new(ridge.project.store(), &ridge.index);
            let sum = surface.area_normal(centre, centre, f64::from(FAR_RADIUS), 0.0);
            surface.finish().expect("the area normal");
            sum
        });
}

/// The strokes of [`Over`] as the previews read them: the earlier one's op begun, the later
/// one's reserved over futures all installed.
struct Two<'a> {
    /// The store.
    store: &'a mut TileStore,
    /// The spatial index, refitted over the futures.
    index: &'a SpatialIndex,
    /// The earlier stroke's op.
    earlier: OpId,
    /// The later stroke's.
    later: &'a ReservedOp,
}

impl OpenStrokes for Two<'_> {
    fn current(&mut self) -> (PenDownReading<'_>, &SpatialIndex) {
        (self.store.read_pen_down(), self.index)
    }

    fn read(&mut self, op: OpId) -> Result<(PenDownReading<'_>, &SpatialIndex), CacheError> {
        if op == self.earlier {
            return Ok(self.current());
        }
        Ok((self.store.read_pen_down_of(self.later)?, self.index))
    }

    fn reserved(&self, op: OpId) -> bool {
        op == self.later.op()
    }

    fn ready(&mut self, _op: OpId, _dabs: &[Dab], _displays: &[TileKey]) -> bool {
        true
    }
}

/// A small stroke about to be drawn along a small stroke's ridge.
///
/// The earlier stroke's [`OVER_DABS`] dabs of [`LATE_RADIUS`] are shown and none landed; with
/// `landing`, the later stroke's op is reserved behind it, the future of every tile the earlier
/// one changes installed and the index refitted over them, as the app has them by the time a
/// stroke's dabs are shown. Without, the earlier stroke has landed and ended, and the later one
/// is alone.
///
/// The futures installed are those of the tiles the earlier stroke's dabs reach by the index,
/// no more: the tiles its landing changes.
struct Over {
    /// The project, with the earlier stroke's dabs shown (or landed).
    ridge: Ridge,
    /// The strokes' ops, the later one's reserved, while the earlier stroke still lands.
    ops: Option<(OpId, ReservedOp)>,
    /// How many dabs of the later stroke are placed.
    drawn: u32,
}

impl Over {
    /// The strokes, the later one with no dab yet.
    ///
    /// # Panics
    ///
    /// If the fixed setup is rejected, which would be a bug.
    fn new(landing: bool) -> Self {
        let mut ridge = Ridge::made(LATE_RADIUS, 0, true);
        let earlier = ridge
            .project
            .store()
            .current_op()
            .expect("the ridge's stroke");
        let dabs: Vec<Dab> = (0..OVER_DABS).map(|_| ridge.next_dab()).collect();
        if !landing {
            ridge.apply(&dabs);
            ridge.project.end_unrecorded_stroke().expect("ended");
            ridge.project.begin_stroke_and_wait().expect("stroke");
            ridge.stroke = Stroke::new(ridge.project.store());
            return Self {
                ridge,
                ops: None,
                drawn: 0,
            };
        }
        let later = ridge.project.begin_stroke_after().expect("reserved");
        let reached = dabs.iter().flat_map(|dab| {
            let [x, y, z] = dab.position;
            ridge
                .index
                .query_sphere(Vec3::new(x, y, z), dab.radius * 1.01)
        });
        let reached: BTreeSet<TileKey> = reached.collect();
        let keys: Vec<TileKey> = reached.iter().copied().collect();
        let mut edits = StrokeEdits::new(earlier, None);
        {
            let store = ridge.project.store();
            let pen_down = PenDownTiles::of_oldest(store, earlier, &reached).expect("pen-down");
            edits.place(store, &dabs, &pen_down).expect("placed");
            edits.seal();
            let futures = Futures::new(store, &ridge.stroke, &[&edits]).expect("futures");
            let made = futures.futures_of(store, &keys);
            for (key, future) in keys.iter().zip(made) {
                let tile = future.expect("a future").into_tile();
                store
                    .install_pen_down_of(&later, *key, tile, Some(earlier))
                    .expect("installed");
            }
        }
        ridge
            .index
            .refit(ridge.project.store(), &keys)
            .expect("refitted");
        let mut over = Self {
            ridge,
            ops: Some((earlier, later)),
            drawn: 0,
        };
        over.show_of(earlier, &dabs);
        over
    }

    /// Shows `dabs` of stroke `op`, drawn at every level.
    ///
    /// # Panics
    ///
    /// If showing fails or shows fewer, which would be a bug.
    fn show_of(&mut self, op: OpId, dabs: &[Dab]) {
        let (earlier, later) = self.ops.as_ref().expect("two strokes");
        let mut two = Two {
            store: self.ridge.project.store(),
            index: &self.ridge.index,
            earlier: *earlier,
            later,
        };
        let shown = self
            .ridge
            .preview
            .show_over(&mut two, op, dabs, &DrawsAll)
            .expect("show");
        assert_eq!(shown.dabs, dabs.len(), "every dab shown");
    }

    /// Shows `dab` of the later stroke.
    ///
    /// # Panics
    ///
    /// As [`Over::show_of`].
    fn show(&mut self, dab: &Dab) {
        let dabs = std::slice::from_ref(dab);
        if let Some(later) = self.ops.as_ref().map(|(_, later)| later.op()) {
            self.show_of(later, dabs);
            return;
        }
        let ridge = &mut self.ridge;
        let store = ridge.project.store();
        let shown = ridge.preview.show(store, &ridge.index, dabs, &DrawsAll);
        assert_eq!(shown.expect("show").dabs, 1, "the dab shown");
    }

    /// The later stroke's next dab: along the earlier stroke's path from its middle, its centre
    /// on the later stroke's pen-down surface (the ridge as it will land), its normal outward.
    ///
    /// # Panics
    ///
    /// If the raycast fails or misses the closed surface, which would be a bug.
    fn next_dab(&mut self) -> Dab {
        let direction = self.ridge.direction(OVER_FROM + self.drawn);
        let ray = stroke::Ray::new(
            DVec3::from_f32(direction * 3.0),
            DVec3::from_f32(direction * -1.0),
        )
        .expect("ray");
        let (store, index) = (self.ridge.project.store(), &self.ridge.index);
        let mut surface = match &self.ops {
            Some((_, later)) => StoreSurface::of_reserved(store, index, later).expect("reserved"),
            None => StoreSurface::new(store, index),
        };
        let hit = surface
            .raycast(&ray)
            .expect("the ray meets the closed surface");
        surface.finish().expect("the raycast");
        let centre = hit.position.to_f32();
        let dab = Dab {
            position: [centre.x, centre.y, centre.z],
            normal: [direction.x, direction.y, direction.z],
            tangent: [0.0, 0.0, 1.0],
            radius: LATE_RADIUS,
            strength: 0.1 * LATE_RADIUS,
            hardness: 0.0,
            seq: self.drawn,
            ..Dab::default()
        };
        self.drawn += 1;
        dab
    }

    /// The finest tiles `dab` reaches by the index.
    fn reach(&self, dab: &Dab) -> Vec<TileKey> {
        let [x, y, z] = dab.position;
        let reached = self
            .ridge
            .index
            .query_sphere(Vec3::new(x, y, z), dab.radius * 1.01);
        reached.collect()
    }
}

/// The first dab of the later stroke of an [`Over`] made per sample (`landing` as [`Over::new`]),
/// shown: what [`over_first`] and [`alone_first`] time.
///
/// # Panics
///
/// If showing fails, which would be a bug.
fn first_dab(bencher: divan::Bencher<'_, '_>, landing: bool) {
    bencher
        .with_inputs(|| {
            let mut over = Over::new(landing);
            let dab = over.next_dab();
            (over, dab)
        })
        .bench_local_values(|(mut over, dab)| {
            over.show(&dab);
            over
        });
}

/// A dab of the later stroke of an [`Over`] with [`OVER_SHOWN`] of its dabs shown before
/// (`landing` as [`Over::new`]), shown: what [`over_steady`] and [`alone_steady`] time.
///
/// # Panics
///
/// If showing fails, which would be a bug, or the samples run past the earlier stroke's ridge
/// ([`OVER_FROM`] and those drawn, against [`OVER_DABS`]).
fn steady_dab(bencher: divan::Bencher<'_, '_>, landing: bool) {
    let over = RefCell::new(Over::new(landing));
    for _ in 0..OVER_SHOWN {
        let over = &mut *over.borrow_mut();
        let dab = over.next_dab();
        over.show(&dab);
    }
    bencher
        .with_inputs(|| {
            let over = &mut *over.borrow_mut();
            let on_ridge = OVER_FROM + over.drawn < OVER_DABS;
            assert!(
                on_ridge,
                "the stroke has run off the earlier stroke's ridge"
            );
            over.next_dab()
        })
        .bench_local_values(|dab| over.borrow_mut().show(&dab));
}

/// [`Preview::show_over`] of the first dab of a stroke over one still landing: see the module
/// notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 7, sample_size = 1)]
fn over_first(bencher: divan::Bencher<'_, '_>) {
    first_dab(bencher, true);
}

/// [`Preview::show_over`] of the first dab of a stroke over one still landing, the seams about
/// it noted ahead: see the module notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 7, sample_size = 1)]
fn over_first_noted(bencher: divan::Bencher<'_, '_>) {
    bencher
        .with_inputs(|| {
            let mut over = Over::new(true);
            let dab = over.next_dab();
            let near = over.reach(&dab);
            let store = over.ridge.project.store();
            assert!(over.ridge.preview.warm_seams(store, &near), "room to note");
            (over, dab)
        })
        .bench_local_values(|(mut over, dab)| {
            over.show(&dab);
            over
        });
}

/// [`Preview::show_over`] of a dab of a stroke well into its way over one still landing: see the
/// module notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 40, sample_size = 1)]
fn over_steady(bencher: divan::Bencher<'_, '_>) {
    steady_dab(bencher, true);
}

/// [`Preview::show`] of the first dab of the same stroke with no other under way: see the module
/// notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 7, sample_size = 1)]
fn alone_first(bencher: divan::Bencher<'_, '_>) {
    first_dab(bencher, false);
}

/// [`Preview::show`] of a dab of the same stroke well into its way, with no other under way: see
/// the module notes.
///
/// # Panics
///
/// If showing fails, which would be a bug.
#[divan::bench(sample_count = 40, sample_size = 1)]
fn alone_steady(bencher: divan::Bencher<'_, '_>) {
    steady_dab(bencher, false);
}
