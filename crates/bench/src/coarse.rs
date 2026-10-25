//! `coarse-error` (hidden, a design spike): how far the coarse levels a renderer would draw lie
//! from the finished surface, and what three ways of accounting for that cost.
//!
//! A brush edits the finest level only, and nothing carries those edits into the coarser levels,
//! so a tile drawn at level `l` shows none of the detail sculpted finer. This measures, on a
//! project, for each coarse level `l` from `finest − 4` to `finest − 1` and every tile of it on
//! the faces sampled:
//!
//! - `d`: the largest distance from a finest-level vertex to the bilinear interpolation of the
//!   level-`l` quad containing it, matched by grid parameter: the error of drawing level `l` as it
//!   is (`d_n` is the part of it along the coarse quad's normal);
//! - `s`: the same, with the level-`l` grid replaced by the finest level's vertices that coincide
//!   with it (every `2^(finest − l)`-th): the error a display layer that subsamples the finished
//!   surface still leaves;
//! - `e`: the bound built bottom-up by the triangle inequality, tile by tile, from each level's
//!   distance to the next finer one: what an incrementally refitted per-tile error stores, to
//!   compare with the `d` it bounds;
//! - the time each takes per tile, the mean quad edge at each level, and the bytes a display-layer
//!   tile (its offsets from level `l`, in the level's own frame, quantized as corrections are)
//!   takes in a tile record, beside the base layer's own tiles.
//!
//! Then a stroke of the `brush` workload (warm), with, for every step, the finest tiles it edited,
//! the coarse tiles whose values that changes, and the time to bring each option's data up to
//! date: the per-tile bound's finest-to-next distance over the edited tiles and its propagation
//! up, and the display layer's offsets at the coinciding vertices with its tiles re-quantized.
//! Finally the faces the stroke touched are measured again, before and after.
//!
//! Each level's result is a JSON line (`{"bench": "coarse-error", "result": "level", …}`), the
//! stroke's another, the faces the stroke touched one per level; a table goes to stderr. The
//! project is opened as the other subcommands open it and closed without saving: measure a
//! reflinked copy, since closing discards the stroke but a crash leaves recovery files beside it.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, ensure};
use geom::{CorrectionTile, FaceId, Level, TileKey, Vec3};
use sculpt::{DrawParams, Stroke, draw_step};
use store::{Evaluation, Project, TileStore};

use crate::brush::{Plan, Size, corrections};
use crate::cli::Options;
use crate::noise::Rng;
use crate::paths::Paths;
use crate::report::{Output, Record, Spread, ms, to_f64};
use crate::session;

/// How many coarse levels are measured, the finest's parents.
const COARSE_LEVELS: u8 = 4;

/// A level's vertices over a block of one face, row-major.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Grid {
    /// The level.
    level: u8,
    /// Vertices per side.
    side: u32,
    /// The block's first column and row on the face's grid at this level.
    origin: (u32, u32),
    /// The positions.
    points: Vec<Vec3>,
}

impl Grid {
    /// A grid of `points`, `side²` row-major, at `level`, starting at `origin` on the face.
    ///
    /// # Errors
    ///
    /// If `points` doesn't hold `side²` positions, or `side` is below 2.
    pub(crate) fn new(level: u8, side: u32, origin: (u32, u32), points: Vec<Vec3>) -> Result<Self> {
        ensure!(side >= 2, "a grid needs at least one quad");
        ensure!(
            points.len() == usize::try_from(side)?.pow(2),
            "a {side}-vertex grid holds {} positions",
            points.len()
        );
        Ok(Self {
            level,
            side,
            origin,
            points,
        })
    }

    /// The vertex at `(i, j)` of the block.
    fn at(&self, i: u32, j: u32) -> Option<Vec3> {
        if i >= self.side || j >= self.side {
            return None;
        }
        let index = usize::try_from(j * self.side + i).ok()?;
        self.points.get(index).copied()
    }

    /// The vertex at `(x, y)` on the face's grid at this level, if the block holds it.
    fn at_face(&self, x: u32, y: u32) -> Option<Vec3> {
        self.at(x.checked_sub(self.origin.0)?, y.checked_sub(self.origin.1)?)
    }

    /// The vertices of `fine` that coincide with this grid's level (every `2^k`-th, `k` the
    /// difference of levels), over the block of `fine`: what a display layer stores.
    ///
    /// # Errors
    ///
    /// If `fine` is coarser than `level`, or its block doesn't start on the coarse grid.
    pub(crate) fn subsampled(fine: &Self, level: u8) -> Result<Self> {
        let k = fine
            .level
            .checked_sub(level)
            .context("subsampling to a finer level")?;
        let step = 1_u32 << k;
        ensure!(
            fine.origin.0.is_multiple_of(step)
                && fine.origin.1.is_multiple_of(step)
                && (fine.side - 1).is_multiple_of(step),
            "the block isn't aligned to level {level}"
        );
        let side = (fine.side - 1) / step + 1;
        let points = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .map(|(i, j)| fine.at(i * step, j * step).context("inside the block"))
            .collect::<Result<Vec<_>>>()?;
        Self::new(
            level,
            side,
            (fine.origin.0 / step, fine.origin.1 / step),
            points,
        )
    }

    /// The bilinear interpolation of the quad holding the point `(x, y)` of the face's grid at
    /// level `level + k` (finer by `k`), and the quad's unit normal (from its diagonals).
    fn bilinear(&self, k: u8, x: u32, y: u32) -> Option<(Vec3, Vec3)> {
        let step = 1_u32 << k;
        let gx = x.checked_sub(self.origin.0 << k)?;
        let gy = y.checked_sub(self.origin.1 << k)?;
        let qi = (gx >> k).min(self.side - 2);
        let qj = (gy >> k).min(self.side - 2);
        let fraction = |g: u32, q: u32| -> Option<f32> {
            let local = u16::try_from(g.checked_sub(q << k)?).ok()?;
            Some(f32::from(local) / f32::from(u16::try_from(step).ok()?))
        };
        let (along_i, along_j) = (fraction(gx, qi)?, fraction(gy, qj)?);
        let c00 = self.at(qi, qj)?;
        let c10 = self.at(qi + 1, qj)?;
        let c01 = self.at(qi, qj + 1)?;
        let c11 = self.at(qi + 1, qj + 1)?;
        let bottom = c00 + (c10 - c00) * along_i;
        let top = c01 + (c11 - c01) * along_i;
        let normal = (c11 - c00)
            .cross(c01 - c10)
            .normalized()
            .unwrap_or(Vec3::ZERO);
        Some((bottom + (top - bottom) * along_j, normal))
    }

    /// The mean length of the grid's quad edges.
    fn mean_edge(&self) -> f64 {
        let (mut sum, mut count) = (0.0_f64, 0_u64);
        for j in 0..self.side {
            for i in 0..self.side {
                let Some(p) = self.at(i, j) else { continue };
                for q in [self.at(i + 1, j), self.at(i, j + 1)].into_iter().flatten() {
                    sum += f64::from((q - p).length());
                    count += 1;
                }
            }
        }
        sum / to_f64(usize::try_from(count).unwrap_or(usize::MAX).max(1))
    }
}

/// How far one surface lies from another over a region: the largest distance between points at
/// the same grid parameter, and the largest of its parts along the coarser surface's normal.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Deviation {
    /// The largest distance.
    pub(crate) max: f32,
    /// The largest distance along the coarse quad's normal.
    pub(crate) normal: f32,
}

impl Deviation {
    /// The larger of each part of `self` and `other`.
    fn max(self, other: Self) -> Self {
        Self {
            max: self.max.max(other.max),
            normal: self.normal.max(other.normal),
        }
    }
}

/// A rectangle of vertices on a face's grid at one level, both ends included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    /// First column.
    x0: u32,
    /// First row.
    y0: u32,
    /// Last column.
    x1: u32,
    /// Last row.
    y1: u32,
}

impl Rect {
    /// Columns `x0..=x1` of rows `y0..=y1`: the tests' regions.
    #[cfg(test)]
    pub(crate) const fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The vertices of tile `key` on its face's grid at its own level.
    fn of_tile(key: &TileKey) -> Self {
        let q = key.level().tile_quads();
        let (x0, y0) = (u32::from(key.tx()) * q, u32::from(key.ty()) * q);
        Self {
            x0,
            y0,
            x1: x0 + q,
            y1: y0 + q,
        }
    }

    /// The same rectangle on the grid finer by `k` levels.
    const fn finer(self, k: u8) -> Self {
        Self {
            x0: self.x0 << k,
            y0: self.y0 << k,
            x1: self.x1 << k,
            y1: self.y1 << k,
        }
    }
}

/// How far `fine`'s vertices in `region` (on `fine`'s level's face grid) lie from `coarse`
/// interpolated bilinearly at the same parameter.
///
/// Both surfaces are bilinear on `fine`'s quads (the coarse one restricted to a fine quad is
/// bilinear too), so the largest distance over the region's area is found at `fine`'s vertices.
///
/// # Errors
///
/// If `fine` is coarser than `coarse`, or either grid doesn't cover `region`.
pub(crate) fn deviation(coarse: &Grid, fine: &Grid, region: Rect) -> Result<Deviation> {
    let k = fine
        .level
        .checked_sub(coarse.level)
        .context("the fine grid is coarser")?;
    let mut worst = Deviation::default();
    for y in region.y0..=region.y1 {
        for x in region.x0..=region.x1 {
            let point = fine
                .at_face(x, y)
                .context("the fine grid covers the region")?;
            let (drawn, normal) = coarse
                .bilinear(k, x, y)
                .context("the coarse grid covers the region")?;
            let gap = point - drawn;
            worst = worst.max(Deviation {
                max: gap.length(),
                normal: gap.dot(normal).abs(),
            });
        }
    }
    Ok(worst)
}

/// One coarse tile's measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TileMeasure {
    /// The tile.
    pub(crate) key: TileKey,
    /// The finest level's distance from this level as stored.
    pub(crate) d: Deviation,
    /// The finest level's distance from its own vertices subsampled to this level.
    pub(crate) s: Deviation,
    /// The bottom-up bound on `d.max`.
    pub(crate) e: f32,
    /// The display-layer tile's payload bytes.
    pub(crate) display_bytes: usize,
    /// The base layer's tile's payload bytes.
    pub(crate) base_bytes: usize,
}

/// One level's measures over the faces measured.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LevelMeasure {
    /// Each tile's.
    pub(crate) tiles: Vec<TileMeasure>,
    /// Time spent computing `d`.
    pub(crate) d_time: Duration,
    /// Time spent computing `s` (subsampling included).
    pub(crate) s_time: Duration,
    /// Time spent on this level's link of the bound (its distance from the next finer level).
    pub(crate) e_time: Duration,
    /// The sum of each face's mean quad edge, and how many faces.
    pub(crate) edge: (f64, usize),
}

/// Every level's measures, coarse to fine, and the finest level's base tiles' payload bytes.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Measures {
    /// The coarse levels measured, from `finest − 4` (or 1) up.
    pub(crate) levels: BTreeMap<u8, LevelMeasure>,
    /// The payload bytes of each base tile of the finest level on the faces measured.
    pub(crate) finest_bytes: Vec<usize>,
}

/// Every tile of `face` at `level`.
///
/// # Errors
///
/// Never for a level whose tiles per side fit a `u16`.
fn face_tiles(face: FaceId, level: Level) -> Result<Vec<TileKey>> {
    let per_side = u16::try_from(level.tiles_per_side())?;
    (0..per_side)
        .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| Ok(TileKey::displacement(face, level, tx, ty)?))
        .collect()
}

/// `face`'s whole grid at `level`, put together from its tiles.
///
/// # Errors
///
/// If a tile can't be evaluated.
fn face_grid(store: &mut TileStore, face: FaceId, level: Level) -> Result<Grid> {
    let quads = level.face_quads();
    let side = quads + 1;
    let mut points = vec![Vec3::ZERO; usize::try_from(side)?.pow(2)];
    for key in face_tiles(face, level)? {
        let rect = Rect::of_tile(&key);
        let positions = store.evaluate(key)?;
        let tile_side = key.side();
        for (n, &p) in positions.iter().enumerate() {
            let n = u32::try_from(n)?;
            let (x, y) = (rect.x0 + n % tile_side, rect.y0 + n / tile_side);
            let slot = points
                .get_mut(usize::try_from(y * side + x)?)
                .context("a tile vertex lies on its face")?;
            *slot = p;
        }
    }
    Grid::new(level.get(), side, (0, 0), points)
}

/// The payload bytes of the base layer's tile `key` as the store would write it.
///
/// # Errors
///
/// If its corrections can't be read or encoded.
fn base_bytes(store: &mut TileStore, key: TileKey) -> Result<usize> {
    let view = store.correction(key)?;
    let side = view.side();
    let values: Vec<Vec3> = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| view.get(i, j).context("inside the tile"))
        .collect::<Result<_>>()?;
    Ok(store::displacement_payload_len(
        &CorrectionTile::from_values(side, &values)?,
    )?)
}

/// The display-layer tile over `rect` of `level`, quantized as a correction tile is.
///
/// Each vertex holds the offset from `level`'s position to the coinciding finest vertex's, in
/// `level`'s frame there (`t` along the grid's `i`, `n` the normal, `b = n × t`).
///
/// # Errors
///
/// If either grid doesn't cover `rect`, or the offsets are too large to quantize.
fn display_tile(level: &Grid, finest: &Grid, rect: Rect) -> Result<CorrectionTile> {
    let k = finest.level - level.level;
    let side = rect.x1 - rect.x0 + 1;
    let mut values = Vec::with_capacity(usize::try_from(side)?.pow(2));
    for y in rect.y0..=rect.y1 {
        for x in rect.x0..=rect.x1 {
            values.push(display_offset(level, finest, k, x, y)?);
        }
    }
    Ok(CorrectionTile::from_values(side, &values)?)
}

/// The display offset of vertex `(x, y)` of `level` (see [`display_tile`]); `k` is how many
/// levels finer `finest` is.
///
/// # Errors
///
/// If either grid doesn't hold the vertex.
fn display_offset(level: &Grid, finest: &Grid, k: u8, x: u32, y: u32) -> Result<Vec3> {
    let here = level.at_face(x, y).context("the level covers the tile")?;
    let target = finest
        .at_face(x << k, y << k)
        .context("the finest level covers the tile")?;
    let along = |ahead: Option<Vec3>, behind: Option<Vec3>| match (ahead, behind) {
        (Some(ahead), Some(behind)) => ahead - behind,
        (Some(ahead), None) => ahead - here,
        (None, Some(behind)) => here - behind,
        (None, None) => Vec3::ZERO,
    };
    let du = along(
        level.at_face(x + 1, y),
        x.checked_sub(1).and_then(|before| level.at_face(before, y)),
    );
    let dv = along(
        level.at_face(x, y + 1),
        y.checked_sub(1).and_then(|before| level.at_face(x, before)),
    );
    let offset = target - here;
    let (Some(normal), Some(tangent)) = (du.cross(dv).normalized(), du.normalized()) else {
        return Ok(offset);
    };
    let bitangent = normal.cross(tangent);
    Ok(Vec3::new(
        offset.dot(tangent),
        offset.dot(bitangent),
        offset.dot(normal),
    ))
}

/// Measures `face`: every tile of each of `coarse` (consecutive, coarse to fine, each finer
/// than the last up to `finest`), into `measures`.
///
/// # Errors
///
/// If a tile can't be evaluated or its corrections read, or a grid doesn't match its level.
pub(crate) fn measure_face(
    store: &mut TileStore,
    face: FaceId,
    finest: Level,
    coarse: &[Level],
    measures: &mut Measures,
) -> Result<()> {
    let lowest = coarse.first().context("no coarse level")?.get();
    let levels = (lowest..=finest.get())
        .map(Level::new)
        .collect::<Result<Vec<_>, _>>()?;
    let keys = levels
        .iter()
        .map(|&l| face_tiles(face, l))
        .collect::<Result<Vec<_>>>()?
        .concat();
    store.evaluate_many(&keys, Evaluation::Positions)?;
    let grids = levels
        .iter()
        .map(|&l| face_grid(store, face, l))
        .collect::<Result<Vec<_>>>()?;
    let fine = grids.last().context("the finest grid")?;
    for key in face_tiles(face, finest)? {
        let bytes = base_bytes(store, key)?;
        measures.finest_bytes.push(bytes);
    }
    // The bound, finest to coarsest: each level's tiles' link to the next finer level, plus the
    // largest bound of the next level's tiles inside it.
    let mut finer_bounds: BTreeMap<(u16, u16), f32> = BTreeMap::new();
    let mut bounds: BTreeMap<u8, BTreeMap<(u16, u16), f32>> = BTreeMap::new();
    for pair in grids.windows(2).rev() {
        let [grid, next] = pair else { continue };
        let level = Level::new(grid.level)?;
        let ratio = Level::new(next.level)?.tiles_per_side() / level.tiles_per_side();
        let start = Instant::now();
        let mut here = BTreeMap::new();
        for key in face_tiles(face, level)? {
            let link = deviation(grid, next, Rect::of_tile(&key).finer(1))?.max;
            let below = (0..ratio)
                .flat_map(|dy| (0..ratio).map(move |dx| (dx, dy)))
                .filter_map(|(dx, dy)| {
                    let tx = u32::from(key.tx()) * ratio + dx;
                    let ty = u32::from(key.ty()) * ratio + dy;
                    finer_bounds.get(&(u16::try_from(tx).ok()?, u16::try_from(ty).ok()?))
                })
                .fold(0.0_f32, |a, &b| a.max(b));
            here.insert((key.tx(), key.ty()), link + below);
        }
        if let Some(entry) = measures.levels.get_mut(&grid.level) {
            entry.e_time += start.elapsed();
        } else {
            measures.levels.insert(
                grid.level,
                LevelMeasure {
                    e_time: start.elapsed(),
                    ..LevelMeasure::default()
                },
            );
        }
        bounds.insert(grid.level, here.clone());
        finer_bounds = here;
    }
    for (grid, &level) in grids.iter().zip(&levels) {
        if !coarse.contains(&level) {
            continue;
        }
        let k = finest.get() - level.get();
        let entry = measures.levels.entry(level.get()).or_default();
        entry.edge.0 += grid.mean_edge();
        entry.edge.1 += 1;
        let subsampling = Instant::now();
        let sub = Grid::subsampled(fine, level.get())?;
        entry.s_time += subsampling.elapsed();
        for key in face_tiles(face, level)? {
            let rect = Rect::of_tile(&key);
            let measuring = Instant::now();
            let d = deviation(grid, fine, rect.finer(k))?;
            let d_done = Instant::now();
            let s = deviation(&sub, fine, rect.finer(k))?;
            entry.d_time += d_done - measuring;
            entry.s_time += d_done.elapsed();
            let display = display_tile(grid, fine, rect)?;
            let e = bounds
                .get(&level.get())
                .and_then(|b| b.get(&(key.tx(), key.ty())))
                .copied()
                .unwrap_or(f32::NAN);
            let base = base_bytes(store, key)?;
            entry.tiles.push(TileMeasure {
                key,
                d,
                s,
                e,
                display_bytes: store::displacement_payload_len(&display)?,
                base_bytes: base,
            });
        }
    }
    Ok(())
}

/// The coarse levels measured under `finest`: `finest − 4` (at least 1) to `finest − 1`.
///
/// # Errors
///
/// If `finest` is level 0 or 1, which have no coarse level to measure.
pub(crate) fn coarse_levels(finest: Level) -> Result<Vec<Level>> {
    let top = finest.get();
    ensure!(top >= 2, "level {top} has no coarse level to measure");
    (top.saturating_sub(COARSE_LEVELS).max(1)..top)
        .map(|l| Ok(Level::new(l)?))
        .collect()
}

/// `count` faces of `faces`, evenly spaced (all of them if `count` is at least `faces`).
fn sample_faces(faces: u32, count: usize) -> Vec<FaceId> {
    let count = u32::try_from(count).unwrap_or(u32::MAX).min(faces).max(1);
    (0..count)
        .map(|n| {
            FaceId::new(
                u32::try_from(u64::from(n) * u64::from(faces) / u64::from(count)).unwrap_or(0),
            )
        })
        .collect()
}

/// The value at quantile `q` (nearest rank) of `values`, or NaN if there are none.
fn quantile(values: &[f32], q: f64) -> f64 {
    let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(f32::total_cmp);
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a rank within a small vector, from a quantile in [0, 1]"
    )]
    let rank = ((q * n as f64).ceil() as usize).clamp(1, n);
    sorted.get(rank - 1).map_or(f64::NAN, |&v| f64::from(v))
}

/// The mean of `values`, or NaN if there are none.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0_usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { f64::NAN } else { sum / to_f64(n) }
}

/// Microseconds in `d`, per `n` items.
fn us_per(d: Duration, n: usize) -> f64 {
    ms(d) * 1000.0 / to_f64(n.max(1))
}

/// The record of one level's measures.
fn level_record(result: &str, level: u8, m: &LevelMeasure) -> Record {
    let pick = |f: fn(&TileMeasure) -> f32| m.tiles.iter().map(f).collect::<Vec<f32>>();
    let d = pick(|t| t.d.max);
    let ratio: Vec<f32> = m
        .tiles
        .iter()
        .filter(|t| t.d.max > 0.0)
        .map(|t| t.e / t.d.max)
        .collect();
    let n = m.tiles.len();
    Record::new("coarse-error", result)
        .with("level", u64::from(level))
        .with("tiles", n)
        .with("quad_edge_mean", m.edge.0 / to_f64(m.edge.1.max(1)))
        .with("d_max", quantile(&d, 1.0))
        .with("d_p50", quantile(&d, 0.5))
        .with("d_p90", quantile(&d, 0.9))
        .with("d_n_max", quantile(&pick(|t| t.d.normal), 1.0))
        .with("s_max", quantile(&pick(|t| t.s.max), 1.0))
        .with("s_p50", quantile(&pick(|t| t.s.max), 0.5))
        .with("s_n_max", quantile(&pick(|t| t.s.normal), 1.0))
        .with("e_max", quantile(&pick(|t| t.e), 1.0))
        .with("e_over_d_p50", quantile(&ratio, 0.5))
        .with("e_over_d_max", quantile(&ratio, 1.0))
        .with("d_us_per_tile", us_per(m.d_time, n))
        .with("s_us_per_tile", us_per(m.s_time, n))
        .with("e_link_us_per_tile", us_per(m.e_time, n))
        .with(
            "display_bytes_mean",
            mean(m.tiles.iter().map(|t| to_f64(t.display_bytes))),
        )
        .with(
            "base_bytes_mean",
            mean(m.tiles.iter().map(|t| to_f64(t.base_bytes))),
        )
}

/// What one brush step cost each option, beside the step itself.
#[derive(Debug, Clone, Default)]
struct StepCost {
    /// The step's own wall time.
    wall: Duration,
    /// Its edit and evaluate phases: what re-encoding the finest level over the edit costs.
    edit_evaluate: Duration,
    /// Finest tiles it edited.
    edited: usize,
    /// Coarse tiles, per level, holding an edited finest tile.
    coarse: BTreeMap<u8, usize>,
    /// Per-tile bound: evaluating the next-coarser level's tiles over the edit.
    bound_eval: Duration,
    /// Per-tile bound: the finest-to-next distance over the edited tiles, and the propagation.
    bound: Duration,
    /// Display layer: evaluating the coarse tiles over the edit.
    display_eval: Duration,
    /// Display layer: the offsets at the coinciding vertices, and the tiles re-quantized.
    display: Duration,
}

/// Picks one part of a step's cost.
type Part = fn(&StepCost) -> Duration;

/// The finest tiles whose corrections differ between `before` and `after`.
fn edited_tiles(
    before: &BTreeMap<TileKey, Vec<Vec3>>,
    after: &BTreeMap<TileKey, Vec<Vec3>>,
) -> Vec<TileKey> {
    before
        .iter()
        .filter(|&(key, old)| after.get(key).is_some_and(|new| new != old))
        .map(|(&key, _)| key)
        .collect()
}

/// The tile of `level` holding finest tile `key`'s vertices.
///
/// # Errors
///
/// Never for a `level` coarser than `key`'s.
fn ancestor(key: &TileKey, level: Level) -> Result<TileKey> {
    let ratio = key.level().tiles_per_side() / level.tiles_per_side();
    let tx = u16::try_from(u32::from(key.tx()) / ratio)?;
    let ty = u16::try_from(u32::from(key.ty()) / ratio)?;
    Ok(TileKey::displacement(key.face(), level, tx, ty)?)
}

/// A tile's grid, with its level's face origin.
///
/// # Errors
///
/// If the tile can't be evaluated.
fn tile_grid(store: &mut TileStore, key: TileKey) -> Result<Grid> {
    let rect = Rect::of_tile(&key);
    let points = store.evaluate(key)?.to_vec();
    Grid::new(key.level().get(), key.side(), (rect.x0, rect.y0), points)
}

/// Brings each option's data up to date after a step that edited `edited` (finest tiles).
///
/// Each part is timed into `cost`. `display` holds the display-layer tiles' offsets, made in full
/// on first touch, untimed: the layer exists before the stroke.
///
/// # Errors
///
/// If a tile can't be evaluated, or an offset quantized.
fn update_options(
    store: &mut TileStore,
    edited: &[TileKey],
    coarse: &[Level],
    display: &mut BTreeMap<TileKey, Vec<Vec3>>,
    cost: &mut StepCost,
) -> Result<()> {
    let fines = edited
        .iter()
        .map(|&key| tile_grid(store, key))
        .collect::<Result<Vec<_>>>()?;
    update_bound(store, edited, &fines, coarse, cost)?;
    for &level in coarse {
        update_display(store, edited, &fines, level, display, cost)?;
    }
    Ok(())
}

/// The per-tile bound's update: only the finest level moved, so only its link to the next
/// coarser level changes, over the edited tiles.
///
/// The coarser links stand, and the new values propagate up by max (plus each tile's unchanged
/// link, one addition, left out here).
///
/// # Errors
///
/// If a tile can't be evaluated.
fn update_bound(
    store: &mut TileStore,
    edited: &[TileKey],
    fines: &[Grid],
    coarse: &[Level],
    cost: &mut StepCost,
) -> Result<()> {
    let finest = edited
        .first()
        .map(TileKey::level)
        .context("no edited tile")?;
    let parent_level = finest.parent().context("the finest level has a parent")?;
    let evaluating = Instant::now();
    let parents = edited
        .iter()
        .map(|key| ancestor(key, parent_level))
        .collect::<Result<BTreeSet<_>>>()?;
    let parents: Vec<TileKey> = parents.into_iter().collect();
    store.evaluate_many(&parents, Evaluation::Positions)?;
    let parent_grids = parents
        .iter()
        .map(|&key| Ok((key, tile_grid(store, key)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    cost.bound_eval += evaluating.elapsed();
    let computing = Instant::now();
    let mut current: BTreeMap<TileKey, f32> = BTreeMap::new();
    for (key, fine) in edited.iter().zip(fines) {
        let parent = ancestor(key, parent_level)?;
        let grid = parent_grids
            .get(&parent)
            .context("the parent was evaluated")?;
        let link = deviation(grid, fine, Rect::of_tile(key))?.max;
        let slot = current.entry(parent).or_insert(0.0);
        *slot = slot.max(link);
    }
    let mut below = parent_level;
    while let Some(up) = below.parent().filter(|l| coarse.contains(l)) {
        let mut next: BTreeMap<TileKey, f32> = BTreeMap::new();
        for (key, value) in &current {
            let slot = next.entry(ancestor(key, up)?).or_insert(0.0);
            *slot = slot.max(*value);
        }
        current = next;
        below = up;
    }
    cost.bound += computing.elapsed();
    std::hint::black_box(&current);
    Ok(())
}

/// The display layer's update at `level`: the offsets at the vertices coinciding with the edited
/// tiles' are recomputed, and every display tile they lie in is re-quantized.
///
/// # Errors
///
/// If a tile can't be evaluated, or an offset quantized.
fn update_display(
    store: &mut TileStore,
    edited: &[TileKey],
    fines: &[Grid],
    level: Level,
    display: &mut BTreeMap<TileKey, Vec<Vec3>>,
    cost: &mut StepCost,
) -> Result<()> {
    let finest = edited
        .first()
        .map(TileKey::level)
        .context("no edited tile")?;
    let k = finest.get() - level.get();
    let evaluating = Instant::now();
    let owners = edited
        .iter()
        .map(|key| ancestor(key, level))
        .collect::<Result<BTreeSet<_>>>()?;
    let owners: Vec<TileKey> = owners.into_iter().collect();
    store.evaluate_many(&owners, Evaluation::Positions)?;
    let grids = owners
        .iter()
        .map(|&key| Ok((key, tile_grid(store, key)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    cost.display_eval += evaluating.elapsed();
    for (&key, grid) in &grids {
        if let Entry::Vacant(slot) = display.entry(key) {
            let whole = face_grid(store, key.face(), finest)?;
            let tile = display_tile(grid, &whole, Rect::of_tile(&key))?;
            let side = tile.side();
            let values = (0..side)
                .flat_map(|j| (0..side).map(move |i| (i, j)))
                .map(|(i, j)| tile.get(i, j).context("inside the tile"))
                .collect::<Result<Vec<_>>>()?;
            slot.insert(values);
        }
    }
    let computing = Instant::now();
    let step = 1_u32 << k;
    for (key, fine) in edited.iter().zip(fines) {
        let owner = ancestor(key, level)?;
        let grid = grids.get(&owner).context("the owner was evaluated")?;
        let values = display
            .get_mut(&owner)
            .context("the display tile was made")?;
        let rect = Rect::of_tile(key);
        let owner_rect = Rect::of_tile(&owner);
        for row in rect.y0.div_ceil(step)..=rect.y1 / step {
            for column in rect.x0.div_ceil(step)..=rect.x1 / step {
                let offset = display_offset(grid, fine, k, column, row)?;
                let at = (row - owner_rect.y0) * grid.side + (column - owner_rect.x0);
                let slot = values
                    .get_mut(usize::try_from(at)?)
                    .context("inside the display tile")?;
                *slot = offset;
            }
        }
    }
    for owner in &owners {
        let values = display.get(owner).context("the display tile was made")?;
        std::hint::black_box(CorrectionTile::from_values(owner.side(), values)?);
    }
    cost.display += computing.elapsed();
    *cost.coarse.entry(level.get()).or_default() += owners.len();
    Ok(())
}

/// Runs `coarse-error` (see the module notes).
///
/// # Errors
///
/// If the project can't be opened, a tile evaluated, a step run, or the report written.
pub(crate) fn run<J: Write, H: Write>(
    options: &Options,
    paths: &Paths,
    out: &mut Output<J, H>,
) -> Result<()> {
    let finest = session::level(options)?;
    let coarse = coarse_levels(finest)?;
    let mut project = session::open(options, paths)?;
    let faces = project.store().mesh().face_count();
    let sampled = sample_faces(faces, options.samples);
    out.say(&format!(
        "coarse-error: {} of {faces} faces, levels {}..{} against {finest}",
        sampled.len(),
        coarse.first().map_or(0, |l| l.get()),
        coarse.last().map_or(0, |l| l.get()),
    ))?;
    let mut measures = Measures::default();
    let start = Instant::now();
    for (n, &face) in sampled.iter().enumerate() {
        measure_face(project.store(), face, finest, &coarse, &mut measures)?;
        out.progress("faces", n + 1, sampled.len(), start)?;
    }
    report_levels(out, "level", &measures)?;
    let finest_mean = mean(measures.finest_bytes.iter().map(|&b| to_f64(b)));
    out.record(
        &Record::new("coarse-error", "finest")
            .with("level", u64::from(finest.get()))
            .with("tiles", measures.finest_bytes.len())
            .with("base_bytes_mean", finest_mean),
    )?;
    stroke(options, &mut project, finest, &coarse, out)?;
    project.close(false).context("closing the project")?;
    Ok(())
}

/// Writes a record per level of `measures`, as `result`, and the table.
///
/// # Errors
///
/// If writing the report fails.
fn report_levels<J: Write, H: Write>(
    out: &mut Output<J, H>,
    result: &str,
    measures: &Measures,
) -> Result<()> {
    out.say("level  tiles   quad edge     d max      d p50     d_n max     s max      e max   e/d p50  d µs/tile  display B  base B")?;
    for (&level, m) in &measures.levels {
        if m.tiles.is_empty() {
            continue;
        }
        let record = level_record(result, level, m);
        out.record(&record)?;
        let pick = |f: fn(&TileMeasure) -> f32| m.tiles.iter().map(f).collect::<Vec<f32>>();
        out.say(&format!(
            "{level:>5}  {:>5}  {:>10.3e}  {:>9.3e}  {:>9.3e}  {:>9.3e}  {:>9.3e}  {:>9.3e}  {:>7.2}  {:>9.1}  {:>9.0}  {:>6.0}",
            m.tiles.len(),
            m.edge.0 / to_f64(m.edge.1.max(1)),
            quantile(&pick(|t| t.d.max), 1.0),
            quantile(&pick(|t| t.d.max), 0.5),
            quantile(&pick(|t| t.d.normal), 1.0),
            quantile(&pick(|t| t.s.max), 1.0),
            quantile(&pick(|t| t.e), 1.0),
            quantile(
                &m.tiles
                    .iter()
                    .filter(|t| t.d.max > 0.0)
                    .map(|t| t.e / t.d.max)
                    .collect::<Vec<_>>(),
                0.5
            ),
            us_per(m.d_time, m.tiles.len()),
            mean(m.tiles.iter().map(|t| to_f64(t.display_bytes))),
            mean(m.tiles.iter().map(|t| to_f64(t.base_bytes))),
        ))?;
    }
    Ok(())
}

/// The brush part of `coarse-error`: a warm stroke, each step's cost to each option, and the
/// faces it touched measured before and after.
///
/// # Errors
///
/// If the index can't be loaded, a tile evaluated or a step run, or writing the report fails.
fn stroke<J: Write, H: Write>(
    options: &Options,
    project: &mut Project,
    finest: Level,
    coarse: &[Level],
    out: &mut Output<J, H>,
) -> Result<()> {
    let mut index = session::load_index(project, finest, "coarse-error", out)?;
    let direction = Rng::new(options.seed).direction();
    let plan = Plan::new(
        project,
        &index,
        direction,
        Size::Footprint(options.footprint),
        options.steps,
    )?;
    let touched: BTreeSet<FaceId> = plan
        .centres
        .iter()
        .flat_map(|&c| index.query_sphere(c, plan.radius).collect::<Vec<_>>())
        .map(|key| key.face())
        .collect();
    let mut before = Measures::default();
    for &face in &touched {
        measure_face(project.store(), face, finest, coarse, &mut before)?;
    }
    plan.warm(project, &index)?;
    let mut stroke_state = Stroke::new(project.store());
    let mut display = BTreeMap::new();
    let mut costs = Vec::with_capacity(plan.centres.len());
    project.begin_stroke().context("beginning a stroke")?;
    for &centre in &plan.centres {
        let candidates: Vec<TileKey> = index.query_sphere(centre, plan.radius).collect();
        let old = corrections(project, &candidates)?;
        let params = DrawParams::new(centre, plan.radius, options.strength)?;
        let start = Instant::now();
        let step =
            draw_step(project, &mut index, &mut stroke_state, params).context("a draw step")?;
        let wall = start.elapsed();
        let new = corrections(project, &candidates)?;
        let edited = edited_tiles(&old, &new);
        let mut cost = StepCost {
            wall,
            edit_evaluate: step.times.edit + step.times.evaluate,
            edited: edited.len(),
            ..StepCost::default()
        };
        if !edited.is_empty() {
            update_options(project.store(), &edited, coarse, &mut display, &mut cost)?;
        }
        costs.push(cost);
    }
    project.end_stroke().context("ending the stroke")?;
    report_stroke(out, plan.radius, &costs)?;
    let mut after = Measures::default();
    for &face in &touched {
        measure_face(project.store(), face, finest, coarse, &mut after)?;
    }
    out.say(&format!(
        "the {} faces the stroke touched, before it:",
        touched.len()
    ))?;
    report_levels(out, "touched_before", &before)?;
    out.say("and after it:")?;
    report_levels(out, "touched_after", &after)?;
    Ok(())
}

/// Writes the stroke's record and its summary.
///
/// # Errors
///
/// If writing the report fails.
fn report_stroke<J: Write, H: Write>(
    out: &mut Output<J, H>,
    radius: f32,
    costs: &[StepCost],
) -> Result<()> {
    let spread =
        |f: fn(&StepCost) -> Duration| Spread::of(&costs.iter().map(f).collect::<Vec<_>>());
    let mut record = Record::new("coarse-error", "stroke")
        .with("steps", costs.len())
        .with("radius", f64::from(radius))
        .with("edited_mean", mean(costs.iter().map(|c| to_f64(c.edited))));
    let parts: [(&str, Part); 6] = [
        ("wall", |c| c.wall),
        ("edit_evaluate", |c| c.edit_evaluate),
        ("bound_eval", |c| c.bound_eval),
        ("bound", |c| c.bound),
        ("display_eval", |c| c.display_eval),
        ("display", |c| c.display),
    ];
    for (name, f) in parts {
        if let Some(s) = spread(f) {
            record = s.add_to(record, name);
        }
    }
    let levels: BTreeSet<u8> = costs
        .iter()
        .flat_map(|c| c.coarse.keys().copied())
        .collect();
    for level in levels {
        record = record.with(
            format!("coarse_tiles_l{level}_mean"),
            mean(
                costs
                    .iter()
                    .map(|c| to_f64(c.coarse.get(&level).copied().unwrap_or(0))),
            ),
        );
    }
    out.record(&record)?;
    let line = |name: &str, s: Option<Spread>| {
        s.map_or_else(String::new, |s| {
            format!("{name}: p50 {:.3} ms, p99 {:.3} ms; ", s.median, s.p99)
        })
    };
    out.say(&format!(
        "stroke of {} steps, {:.1} finest tiles edited a step: {}{}{}{}{}{}",
        costs.len(),
        mean(costs.iter().map(|c| to_f64(c.edited))),
        line("step", spread(|c| c.wall)),
        line("edit+evaluate", spread(|c| c.edit_evaluate)),
        line("bound eval", spread(|c| c.bound_eval)),
        line("bound", spread(|c| c.bound)),
        line("display eval", spread(|c| c.display_eval)),
        line("display", spread(|c| c.display)),
    ))?;
    Ok(())
}
