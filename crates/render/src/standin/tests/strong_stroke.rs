//! A strong stroke's stand-ins against the display tiles the refresh builds for the same edit.
//!
//! The regression test of the spiky stroke (spec, "A stroke seen from a distance", amended
//! 2026-11-01), and the probe that diagnosed it.
//!
//! A project over the cube sphere to [`FINEST`], its display layer built (optionally over a
//! detailed surface: noise on one face's finest corrections), seen from a distance (every display
//! tile resident, no finest tile). A stroke of draw dabs across that face, spaced as the app
//! spaces them (0.15 of the radius), strength 0.1 of the radius (the app's `StrengthMode::Radius`
//! default at full pressure), applied a few dabs a frame as the app's steps are, each frame
//! standing in. Then the snapshot of every display tile above the stroke (the stand-ins), the
//! refresh, and the snapshot again. Per level: how far the refresh's swap moves a drawn vertex,
//! how far the stand-ins' normals are from the refreshed tiles', their bounds, and where a morph
//! from the refreshed tile's coarse interpolation would have started (what the swap replaced).
//!
//! The test checks the swap, the normals and the bounds along a row and across the quads' split
//! ([`check`]): every stand-in's bound at least its real error (in `f64`, against the finest
//! level it showed), and the measured ones within a small factor of the refreshed tile's. The swap
//! itself drawing at once, with no morph, is the raster's
//! `the_refresh_swaps_in_over_the_stand_ins_at_once_with_no_crack`. The ignored probe prints the
//! figures for the case the environment sets: `cargo nextest run -p render --run-ignored only
//! standin::tests::strong_stroke --no-capture`, with `BRUSHCRAFT_DIAG_RADIUS` (finest quads),
//! `BRUSHCRAFT_DIAG_NOISE` (finest quads, the detail's amplitude), `BRUSHCRAFT_DIAG_BATCH` (dabs
//! a frame) and `BRUSHCRAFT_DIAG_DIAGONAL` (0: along the face's middle row; 1: along its
//! diagonal, the quads' split; 2: along the other diagonal, across the split).

use std::f64::consts::PI;

use super::*;

/// A number from the environment, or `default`.
fn env_or(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// `[f32; 3]` as `f64`s.
fn wide(p: [f32; 3]) -> [f64; 3] {
    p.map(f64::from)
}

/// `a - b`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `|a|`.
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// `a × b`.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The angle between `a` and `b`, degrees (0 if either is zero).
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (la, lb) = (norm(a), norm(b));
    if la == 0.0 || lb == 0.0 {
        return 0.0;
    }
    let cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
    cos.clamp(-1.0, 1.0).acos() * 180.0 / PI
}

/// A hash of a face-grid vertex to `-1..1`: the probe's detail.
fn noise(face: u32, i: u32, j: u32) -> f32 {
    let mut h = u64::from(face) << 40 ^ u64::from(i) << 20 ^ u64::from(j);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    #[expect(clippy::cast_precision_loss, reason = "a probe's noise, 24 bits kept")]
    let unit = (h >> 40) as f32 / (1_u64 << 24) as f32;
    unit.mul_add(2.0, -1.0)
}

/// Writes noise of `amplitude` (world units) into every finest correction's normal part on
/// `face`, then refreshes the display layer whole.
fn detail(store: &mut TileStore, face: FaceId, amplitude: f32) {
    let finest = level(FINEST);
    let quads = finest.face_quads();
    store.begin_op().expect("an op");
    for j in 0..=quads {
        for i in 0..=quads {
            let (key, li, lj) = TileKey::containing(
                geom::LayerId::BASE,
                Channel::DISPLACEMENT,
                face,
                finest,
                i,
                j,
            )
            .expect("a finest tile");
            let correction = store
                .correction(key)
                .expect("reads")
                .get(li, lj)
                .expect("a vertex");
            let lift = Vec3::new(0.0, 0.0, amplitude * noise(face.index(), i, j));
            store
                .set_correction(face, finest, i, j, correction + lift)
                .expect("written");
        }
    }
    store.close_epoch().expect("closed");
    run_jobs(store);
}

/// The vertex normal of `positions` (a tile `side` a row) at `(i, j)`, from central differences.
///
/// One-sided at the borders: what recomputing a stand-in's normals from its own positions would
/// give.
fn geometric_normal(positions: &[[f64; 3]], side: u32, i: u32, j: u32) -> [f64; 3] {
    let at = |x: u32, y: u32| positions[usize::try_from(y * side + x).expect("small")];
    let (i0, i1) = (i.saturating_sub(1), (i + 1).min(side - 1));
    let (j0, j1) = (j.saturating_sub(1), (j + 1).min(side - 1));
    cross(sub(at(i1, j), at(i0, j)), sub(at(i, j1), at(i, j0)))
}

/// The case the environment sets.
#[derive(Debug, Clone, Copy)]
struct Case {
    /// The brush's radius, finest quads.
    radius_quads: f64,
    /// The detail's amplitude, finest quads (0: none).
    noise_quads: f64,
    /// Dabs a frame.
    batch: usize,
    /// The stroke's direction: 0 a row, 1 the diagonal along the split, 2 the other.
    direction: u8,
    /// A finest quad's edge, about, world units.
    quad: f64,
}

impl Default for Case {
    /// The checked case: a brush of 8 finest quads along the face's middle row, 4 dabs a
    /// frame, no detail.
    fn default() -> Self {
        Self {
            radius_quads: 8.0,
            noise_quads: 0.0,
            batch: 4,
            direction: 0,
            quad: PI / 2.0 / f64::from(level(FINEST).face_quads()),
        }
    }
}

impl Case {
    /// The case from the environment, the default's settings where it sets none.
    fn from_env() -> Self {
        let default = Self::default();
        #[expect(clippy::cast_precision_loss, reason = "a probe's small settings")]
        let batch = default.batch as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a probe's small settings"
        )]
        let (batch, direction) = (
            env_or("BRUSHCRAFT_DIAG_BATCH", batch).max(1.0) as usize,
            env_or("BRUSHCRAFT_DIAG_DIAGONAL", f64::from(default.direction)).clamp(0.0, 2.0) as u8,
        );
        Self {
            radius_quads: env_or("BRUSHCRAFT_DIAG_RADIUS", default.radius_quads),
            noise_quads: env_or("BRUSHCRAFT_DIAG_NOISE", default.noise_quads),
            batch,
            direction,
            ..default
        }
    }

    /// The brush's radius, world units.
    fn radius(self) -> f32 {
        #[expect(clippy::cast_possible_truncation, reason = "a probe's small settings")]
        let radius = (self.radius_quads * self.quad) as f32;
        radius
    }

    /// The stroke's dabs on the finest grid `grid` (row-major, `quads` quads a side): from a
    /// quarter to three quarters across the face, in the case's direction.
    fn dabs(self, grid: &[Vec3], quads: u32) -> Vec<stroke::Dab> {
        let vertex = |i: u32, j: u32| {
            let v = grid[usize::try_from(j * (quads + 1) + i).expect("small")];
            wide([v.x, v.y, v.z])
        };
        // The finest grid's bilinear point at face-grid place (x, y).
        let point = |x: f64, y: f64| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "within the face"
            )]
            let (i, j) = (x.floor() as u32, y.floor() as u32);
            let (s, t) = (x - f64::from(i), y - f64::from(j));
            let (p00, p10) = (vertex(i, j), vertex(i + 1, j));
            let (p01, p11) = (vertex(i, j + 1), vertex(i + 1, j + 1));
            [0, 1, 2].map(|k| {
                let low = p00[k] + s * (p10[k] - p00[k]);
                let high = p01[k] + s * (p11[k] - p01[k]);
                low + t * (high - low)
            })
        };
        let side = f64::from(quads);
        // Along a diagonal, a face-grid step is about √2 quads.
        let step = 0.15 * self.radius_quads
            / if self.direction > 0 {
                2_f64.sqrt()
            } else {
                1.0
            };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a few hundred dabs"
        )]
        let count = (side / 2.0 / step).floor() as u32 + 1;
        let radius = self.radius();
        (0..count)
            .map(|seq| {
                let x = side / 4.0 + f64::from(seq) * step;
                let y = match self.direction {
                    0 => side / 2.0,
                    1 => x,
                    _ => side - x,
                };
                #[expect(clippy::cast_possible_truncation, reason = "f32 positions")]
                let position = point(x, y).map(|v| v as f32);
                // Placeholder until the dab is applied: it takes its area normal then
                // ([`area_normals`]), as placement gives it.
                stroke::Dab {
                    position,
                    normal: [0.0, 0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    radius,
                    strength: 0.1 * radius,
                    seq,
                    ..stroke::Dab::default()
                }
            })
            .collect()
    }
}

/// `dabs`, each with its area normal on the stroke's pen-down surface, as placement gives it.
///
/// The surface is `store`'s as the stroke began, whatever of its dabs were applied since. Where the
/// sum is degenerate, the normal is the surface normal's stand-in, outward from the sphere's
/// centre.
fn area_normals(
    store: &mut TileStore,
    index: &store::SpatialIndex,
    dabs: &[stroke::Dab],
) -> Vec<stroke::Dab> {
    let mut surface = sculpt::StoreSurface::new(store, index);
    let placed = dabs
        .iter()
        .map(|dab| {
            let [x, y, z] = dab.position;
            let centre = stroke::DVec3::from_f32(Vec3::new(x, y, z));
            // The near side: facing outward from the sphere's centre, about the hit's normal.
            let outward = centre.normalized().expect("a dab off the sphere's centre");
            let normal = stroke::Surface::area_normal(
                &mut surface,
                centre,
                outward,
                f64::from(dab.radius),
                dab.hardness,
            )
            .and_then(stroke::DVec3::normalized)
            .unwrap_or(outward);
            stroke::Dab {
                normal: normal.to_f32_array(),
                ..*dab
            }
        })
        .collect();
    surface.finish().expect("the area normals");
    placed
}

/// A distribution's largest value and sum of squares.
#[derive(Debug, Default, Clone, Copy)]
struct Spread {
    /// The largest value.
    max: f64,
    /// The sum of squares.
    squares: f64,
    /// How many values.
    count: usize,
}

impl Spread {
    /// Adds `value`, returning whether it is the largest so far.
    fn add(&mut self, value: f64) -> bool {
        self.squares += value * value;
        self.count += 1;
        let largest = value > self.max;
        self.max = self.max.max(value);
        largest
    }

    /// The root mean square.
    fn rms(self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let count = self.count.max(1) as f64;
        (self.squares / count).sqrt()
    }
}

/// One level's figures, over the vertices the stroke moved by more than a hundredth of the radius.
#[derive(Debug, Default)]
struct Figures {
    /// Tiles compared.
    tiles: usize,
    /// How far the stroke moved a vertex at most, world units.
    lift_max: f64,
    /// The quad edges summed over the tiles (one edge a tile), world units.
    edges: f64,
    /// Position distance, stand-in vs refreshed: how far the swap moves a drawn vertex.
    position: Spread,
    /// Where the largest is.
    position_at: String,
    /// Normal angle, the stand-in's (as the pass computed it) vs the refreshed one, degrees.
    normal: Spread,
    /// Where the largest is.
    normal_at: String,
    /// Normal angle, the stand-in's positions' own normal (central differences on the CPU,
    /// one-sided at the tile's borders) vs the refreshed one, degrees: what the pass should get.
    recomputed: Spread,
    /// Normal angle, the refreshed tile's positions' own normal vs its stored one: the floor a
    /// recomputation from positions can reach against the stored (averaged) normals.
    floor: Spread,
    /// How far a morph from the refreshed tile's own coarse interpolation would start from it:
    /// an odd vertex's distance from the midpoint of its two parents. What the refresh drew
    /// before it swapped in place (spec, amended 2026-11-01); kept to show what the swap avoids.
    coarse: Spread,
    /// The largest stand-in bound.
    bound_standin: f64,
    /// The largest refreshed bound.
    bound_refreshed: f64,
    /// The largest ratio of the two, tile by tile.
    bound_ratio: f64,
    /// The largest real error of a stand-in: a finest vertex's distance from its triangles, in
    /// `f64`, against the finest level the stand-ins showed.
    error_standin: f64,
    /// The stand-ins whose bound is below their real error.
    dishonest: usize,
    /// Where the first one is.
    dishonest_at: String,
}

/// What a snapshot holds: each resident display tile's vertices and node.
type Snapshot = BTreeMap<Place, (Vec<GpuVertex>, GpuNode)>;

/// The three snapshots of one tile: before the stroke, stood in, refreshed.
struct Tile<'a> {
    /// The tile.
    place: Place,
    /// Before the stroke.
    before: &'a [GpuVertex],
    /// Its stand-in.
    stood: &'a [GpuVertex],
    /// Refreshed.
    refreshed: &'a [GpuVertex],
}

impl Figures {
    /// Adds `tile`'s vertices: those the stroke moved by more than `0.01 radius`.
    fn add(&mut self, tile: &Tile<'_>, radius: f64) {
        let side = tile.place.level().tile_quads() + 1;
        let stood: Vec<[f64; 3]> = tile.stood.iter().map(|v| wide(v.position)).collect();
        let fresh: Vec<[f64; 3]> = tile.refreshed.iter().map(|v| wide(v.position)).collect();
        let index = |i: u32, j: u32| usize::try_from(j * side + i).expect("small");
        self.tiles += 1;
        self.edges += norm(sub(fresh[1], fresh[0]));
        let grid: Vec<(u32, u32)> = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .collect();
        for (i, j) in grid {
            let k = index(i, j);
            let lift = norm(sub(stood[k], wide(tile.before[k].position)));
            self.lift_max = self.lift_max.max(lift);
            if lift <= 0.01 * radius {
                continue;
            }
            let odd = i % 2 == 1 || j % 2 == 1;
            let at = format!("{:?} ({i}, {j}) lift {:.2} R", tile.place, lift / radius);
            if self.position.add(norm(sub(stood[k], fresh[k]))) {
                self.position_at = format!("{at}{}", if odd { " odd" } else { " even" });
            }
            let normal = wide(tile.refreshed[k].normal);
            if self.normal.add(angle(wide(tile.stood[k].normal), normal)) {
                self.normal_at = at;
            }
            self.recomputed
                .add(angle(geometric_normal(&stood, side, i, j), normal));
            self.floor
                .add(angle(geometric_normal(&fresh, side, i, j), normal));
            if odd {
                let (ai, bi) = if i % 2 == 1 { (i - 1, i + 1) } else { (i, i) };
                let (aj, bj) = if j % 2 == 1 { (j - 1, j + 1) } else { (j, j) };
                let (pa, pb) = (fresh[index(ai, aj)], fresh[index(bi, bj)]);
                let mid = [0, 1, 2].map(|c| f64::midpoint(pa[c], pb[c]));
                self.coarse.add(norm(sub(fresh[k], mid)));
            }
        }
    }

    /// The average quad edge of the level's tiles, world units.
    fn edge(&self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let tiles = self.tiles.max(1) as f64;
        self.edges / tiles
    }

    /// Prints the figures of level `l`, in radii `radius` and quad edges too.
    fn print(&self, l: u8, radius: f64) {
        let edge = self.edge();
        let (position, normal, coarse) = (self.position, self.normal, self.coarse);
        println!(
            "level {l}: {} tiles, {} moved vertices (lift max {:.2} R); edge {edge:.5} = {:.3} R",
            self.tiles,
            position.count,
            self.lift_max / radius,
            edge / radius,
        );
        println!(
            "  position stand-in vs refreshed (the swap): max {:.2e} ({:.4} R, {:.4} edge), rms \
             {:.4} R, {:.4} edge; worst at {}",
            position.max,
            position.max / radius,
            position.max / edge,
            position.rms() / radius,
            position.rms() / edge,
            self.position_at,
        );
        println!(
            "  normal, the stand-in's vs refreshed: max {:.1} deg, rms {:.1} deg; worst at {}",
            normal.max,
            normal.rms(),
            self.normal_at,
        );
        println!(
            "  normal, recomputed on the CPU from the stand-in vs refreshed: max {:.1} deg, rms \
             {:.1} deg (floor, from the refreshed positions: max {:.1}, rms {:.1})",
            self.recomputed.max,
            self.recomputed.rms(),
            self.floor.max,
            self.floor.rms(),
        );
        println!(
            "  a morph from the coarse interpolation would start (odd vertices; none now): max \
             {:.3} R ({:.3} edge), rms {:.3} R",
            coarse.max / radius,
            coarse.max / edge,
            coarse.rms() / radius,
        );
        println!(
            "  bound: stand-in max {:.3} R, refreshed max {:.3} R, ratio max {:.2}; stand-in real \
             error max {:.3} R, {} bounds below it",
            self.bound_standin / radius,
            self.bound_refreshed / radius,
            self.bound_ratio,
            self.error_standin / radius,
            self.dishonest,
        );
    }
}

/// The real error of the display tile at `place`, its `vertices`, against the finest level `grids`.
///
/// The largest distance, in `f64`, of a finest vertex under the tile (`grids` row-major, a face
/// each) from its triangles, at its place on the quad holding it: the bound's pointwise measure
/// (the stand-in module's notes, "The bound").
fn real_error(place: Place, vertices: &[GpuVertex], grids: &BTreeMap<u32, Vec<Vec3>>) -> f64 {
    let n = level(FINEST).face_quads();
    let grid = &grids[&place.face().index()];
    let size = n / place.level().tiles_per_side();
    let (x0, y0) = (u32::from(place.tx()) * size, u32::from(place.ty()) * size);
    let quads = place.level().tile_quads();
    let scale = size / quads;
    let p = |i: u32, j: u32| {
        wide(vertices[usize::try_from(j * (quads + 1) + i).expect("small")].position)
    };
    let places = (0..=size).flat_map(|y| (0..=size).map(move |x| (x, y)));
    places.fold(0.0, |worst: f64, (x, y)| {
        let fine = grid[usize::try_from((y0 + y) * (n + 1) + x0 + x).expect("small")];
        let (qx, qy) = ((x / scale).min(quads - 1), (y / scale).min(quads - 1));
        let fx = f64::from(x - qx * scale) / f64::from(scale);
        let fy = f64::from(y - qy * scale) / f64::from(scale);
        let (p00, p10, p11, p01) = (p(qx, qy), p(qx + 1, qy), p(qx + 1, qy + 1), p(qx, qy + 1));
        let drawn: [f64; 3] = [0, 1, 2].map(|a| {
            if fx >= fy {
                p00[a] + fx * (p10[a] - p00[a]) + fy * (p11[a] - p10[a])
            } else {
                p00[a] + fy * (p01[a] - p00[a]) + fx * (p11[a] - p01[a])
            }
        });
        worst.max(norm(sub(wide([fine.x, fine.y, fine.z]), drawn)))
    })
}

/// The figures by level of every display tile above `edited` in all three snapshots, the
/// stand-ins' real error against `stepped`, the finest level they showed.
fn compare(
    edited: &BTreeSet<TileKey>,
    before: &Snapshot,
    stood: &Snapshot,
    refreshed: &Snapshot,
    stepped: &BTreeMap<u32, Vec<Vec3>>,
    radius: f64,
) -> BTreeMap<u8, Figures> {
    let mut by_level: BTreeMap<u8, Figures> = BTreeMap::new();
    for place in above(edited) {
        let (Some((old, _)), Some((stand_in, stood_node)), Some((fresh, fresh_node))) =
            (before.get(&place), stood.get(&place), refreshed.get(&place))
        else {
            continue;
        };
        let figures = by_level.entry(place.level().get()).or_default();
        let (bound, fresh_bound) = (f64::from(stood_node.error), f64::from(fresh_node.error));
        figures.bound_standin = figures.bound_standin.max(bound);
        figures.bound_refreshed = figures.bound_refreshed.max(fresh_bound);
        figures.bound_ratio = figures.bound_ratio.max(bound / fresh_bound.max(1e-12));
        let real = real_error(place, stand_in, stepped);
        figures.error_standin = figures.error_standin.max(real);
        if real > bound {
            if figures.dishonest == 0 {
                figures.dishonest_at = format!("{place:?}: real {real:e}, bound {bound:e}");
            }
            figures.dishonest += 1;
        }
        let tile = Tile {
            place,
            before: old,
            stood: stand_in,
            refreshed: fresh,
        };
        figures.add(&tile, radius);
    }
    by_level
}

/// Runs `case` on `gpu`: the stroke stood in a step a frame, then refreshed. Returns the
/// figures by level.
fn run(gpu: &Gpu, case: Case) -> BTreeMap<u8, Figures> {
    run_with(gpu, case, &[])
}

/// [`run`] with the stand-in pass's sources patched by `patches`, for the bites.
fn run_with(gpu: &Gpu, case: Case, patches: &[(&str, &str)]) -> BTreeMap<u8, Figures> {
    let finest = level(FINEST);
    let dir = TempDir::new("strong-stroke");
    let (mut project, mesh) = project(&dir, |config| config);
    let face = FaceId::new(2);
    if case.noise_quads > 0.0 {
        #[expect(clippy::cast_possible_truncation, reason = "a probe's small settings")]
        let amplitude = (case.noise_quads * case.quad) as f32;
        detail(project.store(), face, amplitude);
    }
    let cursor = project.store().subscribe_stale();
    let mut residency = far_residency(gpu, &mesh, project.store());
    residency
        .patch_stand_in_pass(gpu, patches)
        .expect("the stand-in pass");
    let before = display_tiles(gpu, &residency);
    let grids = finest_grids(project.store(), &mesh);
    let dabs = case.dabs(&grids[&face.index()], finest.face_quads());

    // The stroke, a step a frame.
    let mut index =
        store::SpatialIndex::build(project.store(), finest, |_, _| {}).expect("an index");
    let mut stroke_state = sculpt::Stroke::new(project.store());
    project.begin_stroke().expect("a stroke");
    let mut edited = BTreeSet::new();
    for step in dabs.chunks(case.batch) {
        let step = area_normals(project.store(), &index, step);
        sculpt::apply_dabs(project.store(), &mut index, &mut stroke_state, &step)
            .expect("the dabs apply");
        let drained = project.store().drain_stale(&cursor);
        edited.extend(finest_of(&drained));
        frame(gpu, &mut residency, project.store(), drained);
    }
    project.end_unrecorded_stroke().expect("the stroke ends");
    let store = project.store();
    let last = store.drain_stale(&cursor);
    frame(gpu, &mut residency, store, last);
    let stats = residency.stats();
    println!(
        "case: {case:?}; radius {:.5} world, {} dabs; stand-ins {} skipped {}",
        case.radius(),
        dabs.len(),
        stats.stand_ins,
        stats.stand_ins_skipped
    );
    assert!(stats.stand_ins > 0, "nothing stood in");
    let stood = display_tiles(gpu, &residency);
    let stepped = finest_grids(store, &mesh);

    // The refresh.
    run_jobs(store);
    let refresh = store.drain_stale(&cursor);
    frame(gpu, &mut residency, store, refresh);
    frame(gpu, &mut residency, store, Drained::Epochs(Vec::new()));
    let refreshed = display_tiles(gpu, &residency);
    assert_eq!(
        stepped,
        finest_grids(store, &mesh),
        "the refresh moves no finest vertex"
    );

    let by_level = compare(
        &edited,
        &before,
        &stood,
        &refreshed,
        &stepped,
        f64::from(case.radius()),
    );
    store.unsubscribe_stale(cursor);
    drop(project.close(false));
    by_level
}

/// Checks `case`'s figures, printing them.
///
/// At every level the refresh swaps in within [`SWAP_EDGES`] of a quad
/// edge of what the stand-in drew, and the stand-in's normals are within [`NORMAL_SLACK`] of the
/// floor (what the refreshed tile's own positions give against its stored normals) and
/// [`NORMAL_RMS`] rms of the refreshed tile's. Every stand-in's bound is at least its real error,
/// and, at the levels whose bound is measured, at most [`BOUND_RATIO`] times the refreshed tile's.
fn check(case: Case, by_level: &BTreeMap<u8, Figures>) {
    let radius = f64::from(case.radius());
    let mut compared = 0;
    for (l, figures) in by_level {
        figures.print(*l, radius);
        assert_eq!(
            figures.dishonest, 0,
            "level {l}: stand-in bounds below their real error, the first {}",
            figures.dishonest_at
        );
        if *l >= MEASURED_FROM {
            assert!(
                figures.bound_ratio <= BOUND_RATIO,
                "level {l}: a stand-in's bound {:.2} times the refreshed tile's",
                figures.bound_ratio
            );
        }
        if figures.position.count == 0 {
            continue;
        }
        compared += figures.position.count;
        let swap = figures.position.max / figures.edge();
        assert!(
            swap <= SWAP_EDGES,
            "level {l}: the swap moves a vertex {swap:.4} of an edge, at {}",
            figures.position_at
        );
        let (normal, floor) = (figures.normal, figures.floor);
        assert!(
            normal.max <= floor.max + NORMAL_SLACK && normal.rms() <= floor.rms() + NORMAL_SLACK,
            "level {l}: the stand-in's normals are max {:.1} deg, rms {:.1} deg from the \
             refreshed tile's, the floor max {:.1}, rms {:.1}; worst at {}",
            normal.max,
            normal.rms(),
            floor.max,
            floor.rms(),
            figures.normal_at
        );
        assert!(
            normal.rms() <= NORMAL_RMS,
            "level {l}: the stand-in's normals are {:.1} deg rms from the refreshed tile's",
            normal.rms()
        );
    }
    assert!(compared > 500, "{compared} moved vertices compared");
}

/// The coarsest level whose stand-ins' bounds are measured: six above the finest, where a quad
/// spans a finest tile (the module's notes, "The bound"). Coarser ones take the formula.
const MEASURED_FROM: u8 = FINEST - 6;

/// The most a measured stand-in bound may be over the refreshed tile's: it measures the stand-in's
/// own error, about the refreshed tile's, which adds its quantization and chains its levels.
const BOUND_RATIO: f64 = 4.0;

/// The most a vertex moves when the refresh swaps in for its stand-in, in quad edges of its
/// level: the stand-in is the finest bits, the refresh their quantized encoding, within about
/// 0.03 of an edge.
const SWAP_EDGES: f64 = 0.05;

/// How far, degrees, the stand-in's normals may be from the refreshed ones beyond the floor
/// (max and rms): the encoding's step and the one-sided differences' change at a seam.
const NORMAL_SLACK: f64 = 1.0;

/// The most the stand-in's normals may be from the refreshed ones, rms, degrees.
///
/// The strong stroke's floor (the refreshed tile's own positions against its stored normals) is
/// 8.6 at its worst level, level 3 across the quads' split, since a stroke weighs its dabs at
/// pen-down and its ridge is their sum, 1.37 radii tall (2026-11-02; 5.2 there before, on a ridge
/// of 0.81 radii, and about 4 when each dab moved its vertices along their own frame normals);
/// the stand-in meets the floor at every level. Its stored normals were 30–34 off.
const NORMAL_RMS: f64 = 10.0;

#[test]
fn a_strong_stroke_s_refresh_swaps_in_close_to_its_stand_ins_along_a_row_and_a_diagonal() {
    GpuTest::run(|gpu| {
        for direction in [0, 2] {
            let case = Case {
                direction,
                ..Case::default()
            };
            check(case, &run(gpu, case));
        }
    });
}

#[test]
fn the_strong_stroke_test_bites_stand_ins_that_keep_their_stored_normals() {
    GpuTest::run(|gpu| {
        let case = Case::default();
        let by_level = run_with(gpu, case, &[("slots[at].normal = ", "let unused = ")]);
        let stale = by_level
            .values()
            .map(|f| f.normal.rms())
            .fold(0.0, f64::max);
        assert!(
            stale > 2.0 * NORMAL_RMS,
            "stored normals {stale:.1} deg rms off"
        );
    });
}

#[test]
fn the_strong_stroke_test_bites_a_measured_bound_that_halves() {
    GpuTest::run(|gpu| {
        let case = Case::default();
        let halved = [(
            "worst = max(worst, bound_bits(fine - drawn, magnitude));",
            "worst = max(worst, bound_bits(0.5 * (fine - drawn), 0.0));",
        )];
        let by_level = run_with(gpu, case, &halved);
        let dishonest: usize = by_level.values().map(|f| f.dishonest).sum();
        assert!(dishonest > 0, "no bound below its real error");
    });
}

#[test]
#[ignore = "a diagnosis probe: prints figures for the case the environment sets, checks nothing"]
fn stand_ins_against_the_refresh_after_a_strong_stroke() {
    GpuTest::run(|gpu| {
        let case = Case::from_env();
        let radius = f64::from(case.radius());
        for (l, figures) in &run(gpu, case) {
            figures.print(*l, radius);
        }
    });
}
