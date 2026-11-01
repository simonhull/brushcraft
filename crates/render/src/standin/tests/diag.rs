//! A diagnosis probe, not a check: a strong stroke's stand-ins against the display tiles the
//! refresh builds for the same edit.
//!
//! A project over the cube sphere to [`FINEST`], its display layer built (optionally over a
//! detailed surface: noise on one face's finest corrections), seen from a distance (every display
//! tile resident, no finest tile). A stroke of draw dabs across that face, spaced as the app
//! spaces them (0.15 of the radius), strength 0.1 of the radius (the app's `StrengthMode::Radius`
//! default at full pressure), applied a few dabs a frame as the app's steps are, each frame
//! standing in. Then the snapshot of every display tile above the stroke (the stand-ins), the
//! refresh, and the snapshot again. Per level: how far the stand-ins' positions and normals are
//! from the refreshed tiles', where the worst sit, their bounds, and how far the refreshed tile's
//! morph-in starts from its own shape.
//!
//! Run: `cargo nextest run -p render --run-ignored only standin::tests::diag --no-capture`.
//! `BRUSHCRAFT_DIAG_RADIUS` (finest quads), `BRUSHCRAFT_DIAG_NOISE` (finest quads, the detail's
//! amplitude), `BRUSHCRAFT_DIAG_BATCH` (dabs a frame) and `BRUSHCRAFT_DIAG_DIAGONAL` (0: along
//! the face's middle row; 1: along its diagonal, the quads' split; 2: along the other diagonal,
//! across the split) vary the case.

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

impl Case {
    /// The case from the environment, with the defaults the report quotes.
    fn from_env() -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a probe's small settings"
        )]
        let (batch, direction) = (
            env_or("BRUSHCRAFT_DIAG_BATCH", 4.0).max(1.0) as usize,
            env_or("BRUSHCRAFT_DIAG_DIAGONAL", 0.0).clamp(0.0, 2.0) as u8,
        );
        Self {
            radius_quads: env_or("BRUSHCRAFT_DIAG_RADIUS", 8.0),
            noise_quads: env_or("BRUSHCRAFT_DIAG_NOISE", 0.0),
            batch,
            direction,
            quad: PI / 2.0 / f64::from(level(FINEST).face_quads()),
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
    /// Position distance, stand-in vs refreshed.
    position: Spread,
    /// Where the largest is.
    position_at: String,
    /// Normal angle, the stand-in's stored normal vs the refreshed one, degrees.
    stale: Spread,
    /// Where the largest is.
    stale_at: String,
    /// Normal angle, the stand-in's positions' own normal vs the refreshed one, degrees.
    recomputed: Spread,
    /// Normal angle, the refreshed tile's positions' own normal vs its stored one: the floor a
    /// recomputation from positions can reach against the mip.
    floor: Spread,
    /// How far the refreshed tile's morph-in starts from it: an odd vertex's distance from the
    /// midpoint of its two parents (where `NODE_MORPH_IN` starts it).
    morph: Spread,
    /// How far that start sinks an odd vertex towards the centre below the refreshed one (the
    /// dips between the even vertices, which stay), largest, world units.
    dip: f64,
    /// The largest stand-in bound.
    bound_standin: f64,
    /// The largest refreshed bound.
    bound_refreshed: f64,
    /// The largest ratio of the two, tile by tile.
    bound_ratio: f64,
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
            if self.stale.add(angle(wide(tile.stood[k].normal), normal)) {
                self.stale_at = at;
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
                self.morph.add(norm(sub(fresh[k], mid)));
                self.dip = self.dip.max(norm(fresh[k]) - norm(mid));
            }
        }
    }

    /// Prints the figures of level `l`, in radii `radius` and quad edges too.
    fn print(&self, l: u8, radius: f64) {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let edge = self.edges / self.tiles.max(1) as f64;
        let (position, stale, morph) = (self.position, self.stale, self.morph);
        println!(
            "level {l}: {} tiles, {} moved vertices (lift max {:.2} R); edge {edge:.5} = {:.3} R",
            self.tiles,
            position.count,
            self.lift_max / radius,
            edge / radius,
        );
        println!(
            "  position stand-in vs refreshed: max {:.2e} ({:.4} R, {:.4} edge), rms {:.4} R, \
             {:.4} edge; worst at {}",
            position.max,
            position.max / radius,
            position.max / edge,
            position.rms() / radius,
            position.rms() / edge,
            self.position_at,
        );
        println!(
            "  normal, stored (stale) vs refreshed: max {:.1} deg, rms {:.1} deg; worst at {}",
            stale.max,
            stale.rms(),
            self.stale_at,
        );
        println!(
            "  normal, recomputed from the stand-in vs refreshed: max {:.1} deg, rms {:.1} deg \
             (floor, from the refreshed positions: max {:.1}, rms {:.1})",
            self.recomputed.max,
            self.recomputed.rms(),
            self.floor.max,
            self.floor.rms(),
        );
        println!(
            "  refresh morph-in start vs refreshed (odd vertices): max {:.3} R ({:.3} edge), \
             rms {:.3} R, deepest dip {:.3} R",
            morph.max / radius,
            morph.max / edge,
            morph.rms() / radius,
            self.dip / radius,
        );
        println!(
            "  bound: stand-in max {:.2} R, refreshed max {:.3} R, ratio max {:.0}",
            self.bound_standin / radius,
            self.bound_refreshed / radius,
            self.bound_ratio,
        );
    }
}

/// The figures by level of every display tile above `edited` in all three snapshots.
fn compare(
    edited: &BTreeSet<TileKey>,
    before: &Snapshot,
    stood: &Snapshot,
    refreshed: &Snapshot,
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

#[test]
#[ignore = "a diagnosis probe: prints figures, checks nothing"]
fn stand_ins_against_the_refresh_after_a_strong_stroke() {
    GpuTest::run(|gpu| {
        let case = Case::from_env();
        let finest = level(FINEST);
        let dir = TempDir::new("diag");
        let (mut project, mesh) = project(&dir, |config| config);
        let face = FaceId::new(2);
        if case.noise_quads > 0.0 {
            #[expect(clippy::cast_possible_truncation, reason = "a probe's small settings")]
            let amplitude = (case.noise_quads * case.quad) as f32;
            detail(project.store(), face, amplitude);
        }
        let cursor = project.store().subscribe_stale();
        let mut residency = far_residency(gpu, &mesh, project.store());
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
            sculpt::apply_dabs(project.store(), &mut index, &mut stroke_state, step)
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

        let radius = f64::from(case.radius());
        let by_level = compare(&edited, &before, &stood, &refreshed, radius);
        for (l, figures) in &by_level {
            figures.print(*l, radius);
        }
        store.unsubscribe_stale(cursor);
        drop(project.close(false));
    });
}
