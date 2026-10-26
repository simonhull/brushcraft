//! Building display tiles from the finest level: subsampling, the error a level adds, the
//! normals' mip, and a whole face's tiles with their bounds chained.

use super::encode::{self, CHOSEN, CHOSEN_NORMALS};
use super::{DisplayCodec, DisplayError, DisplayTile, NormalEncoding};
use crate::{FaceId, Grid, Level, Region, TileKey, Vec3};

/// Unnormalized mean normals over a face's grid at one level, row-major `side²`.
///
/// Each is the mean of the finest unit normals under its vertex, through [`mip_normals`]'s tent.
/// Their length is at most 1, and less where the normals under a vertex disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanNormals {
    /// The means, as a grid of vectors.
    grid: Grid,
}

impl MeanNormals {
    /// The finest level's normals, each its own mean: unit normals, as the evaluation gives them.
    #[must_use]
    pub const fn from_unit(normals: Grid) -> Self {
        Self { grid: normals }
    }

    /// Vertices per side.
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.grid.side()
    }

    /// The means, row-major.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec3] {
        self.grid.as_slice()
    }
}

/// `(i, j)` of a grid with `side` vertices per side, as an index into its row-major data.
fn at(side: u32, i: u32, j: u32) -> usize {
    usize::try_from(u64::from(j) * u64::from(side) + u64::from(i)).unwrap_or(usize::MAX)
}

/// The vertex coordinates `(i, j)` of a grid with `side` vertices per side, row-major.
fn cells(side: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..side).flat_map(move |j| (0..side).map(move |i| (i, j)))
}

/// The vertices per side a grid of `side` has once halved: `(side − 1) / 2 + 1`, or `None` if
/// `side − 1` isn't even and positive (the grid has no even vertices nested in a coarser one).
fn halved(side: u32) -> Option<u32> {
    let quads = side.checked_sub(1)?;
    (quads >= 2 && quads.is_multiple_of(2)).then_some(quads / 2 + 1)
}

/// The even vertices of `finer` (`(2i, 2j)`), as the grid one level coarser: the finest surface
/// subsampled, nested exactly (every value is copied, bit for bit).
///
/// # Errors
///
/// [`DisplayError::Shape`] if `finer` has no coarser grid nested in it (its quads per side
/// aren't even and at least 2).
pub fn subsample(finer: &Grid) -> Result<Grid, DisplayError> {
    let side = halved(finer.side()).ok_or(DisplayError::Shape)?;
    let data = cells(side)
        .map(|(i, j)| finer.get(2 * i, 2 * j).ok_or(DisplayError::Shape))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Grid::new(side, data)?)
}

/// `δ`: the largest distance from a vertex of `finer` to the surface `coarser` draws at the
/// same grid parameter, `finer` covering the block `coarser` covers at the next level (its side
/// is `2 × (coarser's − 1) + 1`).
///
/// `coarser` draws each quad as two triangles split along its `(i, j)`–`(i + 1, j + 1)` diagonal,
/// so at an even vertex `(2i, 2j)` it draws its own vertex `(i, j)`; at an edge vertex the
/// midpoint of the edge's ends; at a quad centre `(2i + 1, 2j + 1)` the midpoint of the diagonal
/// from `(i, j)` to `(i + 1, j + 1)`. The finer triangles refine the coarser ones, so this is the
/// largest distance between the two surfaces anywhere over the block.
///
/// Computed in `f64` from the `f32` positions and rounded up to an `f32`, so it is a bound
/// itself, not an estimate of one.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the grids' sides don't match.
pub fn step_error(finer: &Grid, coarser: &Grid) -> Result<f32, DisplayError> {
    step_error_at(finer, (0, 0), coarser)
}

/// [`step_error`] over the block of `finer` starting at `origin` (on `finer`'s grid) with
/// `coarser`'s extent: a tile of a face's grid.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the block doesn't lie inside `finer`.
pub(super) fn step_error_at(
    finer: &Grid,
    origin: (u32, u32),
    coarser: &Grid,
) -> Result<f32, DisplayError> {
    let quads = coarser.side().checked_sub(1).ok_or(DisplayError::Shape)?;
    let side = 2 * quads + 1;
    if origin.0 + side > finer.side() || origin.1 + side > finer.side() {
        return Err(DisplayError::Shape);
    }
    let mut worst = 0.0_f64;
    for (i, j) in cells(side) {
        let p = finer
            .get(origin.0 + i, origin.1 + j)
            .ok_or(DisplayError::Shape)?;
        let drawn = drawn_at(coarser, i, j).ok_or(DisplayError::Shape)?;
        worst = worst.max(distance(p, drawn));
    }
    round_up(worst)
}

/// The point `coarser` draws at vertex `(i, j)` of the grid one level finer, in `f64`.
///
/// That is the coarse vertex, or the midpoint of the coarse edge or quad diagonal the vertex
/// lies on (see [`step_error`]). `None` outside the grid.
fn drawn_at(coarser: &Grid, i: u32, j: u32) -> Option<[f64; 3]> {
    let (ci, cj) = (i / 2, j / 2);
    let (a, b) = match (i % 2, j % 2) {
        (0, 0) => return Some(wide(coarser.get(ci, cj)?)),
        (1, 0) => (coarser.get(ci, cj)?, coarser.get(ci + 1, cj)?),
        (0, _) => (coarser.get(ci, cj)?, coarser.get(ci, cj + 1)?),
        _ => (coarser.get(ci, cj)?, coarser.get(ci + 1, cj + 1)?),
    };
    let ([ax, ay, az], [bx, by, bz]) = (wide(a), wide(b));
    Some([
        f64::midpoint(ax, bx),
        f64::midpoint(ay, by),
        f64::midpoint(az, bz),
    ])
}

/// `v` in `f64`, exactly.
fn wide(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

/// The distance from `p` to `q`, in `f64`.
fn distance(p: Vec3, [qx, qy, qz]: [f64; 3]) -> f64 {
    let [px, py, pz] = wide(p);
    let (dx, dy, dz) = (px - qx, py - qy, pz - qz);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// The least `f32` at or above `v`, which is finite and non-negative.
///
/// # Errors
///
/// [`DisplayError::NotFinite`] if `v` is NaN, or rounds up past `f32::MAX`.
fn round_up(v: f64) -> Result<f32, DisplayError> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rounded to the nearest f32, then moved up a step if that fell below"
    )]
    let near = v as f32;
    let up = if f64::from(near) < v {
        near.next_up()
    } else {
        near
    };
    if up.is_finite() && up >= 0.0 {
        Ok(up)
    } else {
        Err(DisplayError::NotFinite)
    }
}

/// The weight of each 3×3 tent neighbour, in sixteenths, by `(di, dj)` from the centre.
const fn tent_weight(di: i64, dj: i64) -> f32 {
    match (di == 0, dj == 0) {
        (true, true) => 4.0,
        (true, false) | (false, true) => 2.0,
        (false, false) => 1.0,
    }
}

/// The mean normals one level coarser, through a 3×3 tent.
///
/// At each coarse vertex `(i, j)`, the tent is over
/// `finer`'s `(2i + di, 2j + dj)`, weights ¼ at the centre, ⅛ at edge neighbours and 1/16 at
/// diagonal ones. Where the face's border cuts the tent off, the weights present are
/// renormalized to sum to 1. The means stay unnormalized, so their length still says how much
/// the normals under them agree.
///
/// # Errors
///
/// [`DisplayError::Shape`] if `finer` has no coarser grid nested in it.
pub fn mip_normals(finer: &MeanNormals) -> Result<MeanNormals, DisplayError> {
    let side = halved(finer.side()).ok_or(DisplayError::Shape)?;
    let fine = &finer.grid;
    let data = cells(side)
        .map(|(i, j)| {
            let (ci, cj) = (i64::from(2 * i), i64::from(2 * j));
            let mut sum = Vec3::ZERO;
            let mut weight = 0.0_f32;
            for (di, dj) in (-1..=1).flat_map(|dj| (-1..=1).map(move |di| (di, dj))) {
                let neighbour = u32::try_from(ci + di)
                    .ok()
                    .zip(u32::try_from(cj + dj).ok())
                    .and_then(|(x, y)| fine.get(x, y));
                if let Some(n) = neighbour {
                    let w = tent_weight(di, dj);
                    sum = sum + n * w;
                    weight += w;
                }
            }
            sum * weight.recip()
        })
        .collect();
    Ok(MeanNormals {
        grid: Grid::new(side, data)?,
    })
}

/// The block of `grid` covering `region` (on the grid's level), copied out.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the region doesn't lie inside the grid.
fn block(grid: &Grid, region: Region) -> Result<Grid, DisplayError> {
    let side = region.quads() + 1;
    let data = cells(side)
        .map(|(i, j)| {
            grid.get(region.x0() + i, region.y0() + j)
                .ok_or(DisplayError::Shape)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Grid::new(side, data)?)
}

/// Every display tile key of `face` at `level`, row by row.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the level has more tiles per side than a key counts (none
/// does).
fn level_keys(face: FaceId, level: Level) -> Result<Vec<TileKey>, DisplayError> {
    let per_side =
        u16::try_from(level.tiles_per_side()).map_err(|_out_of_range| DisplayError::Shape)?;
    (0..per_side)
        .flat_map(|ty| (0..per_side).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            TileKey::display(face, level, tx, ty).map_err(|_out_of_range| DisplayError::Shape)
        })
        .collect()
}

/// Every display tile of `face` at every level below `finest`, with the encoding chosen by
/// measurement: see [`build_face_with`].
///
/// # Errors
///
/// As [`build_face_with`].
pub fn build_face(
    face: FaceId,
    finest_positions: &Grid,
    finest_normals: &Grid,
    finest: Level,
) -> Result<Vec<DisplayTile>, DisplayError> {
    build_face_with(
        &CHOSEN,
        CHOSEN_NORMALS,
        face,
        finest_positions,
        finest_normals,
        finest,
    )
}

/// Every display tile of `face` at every level below `finest`, in key order, positions stored
/// with `codec` and normals with `normals`.
///
/// Each level's positions are the finest ones
/// subsampled ([`subsample`]), its normals the finest ones mipped ([`mip_normals`]), and each
/// tile's bound `e` is chained from the finest level up through the *decoded* positions (see
/// the module notes), so the codec's quantization is inside it.
///
/// `finest_positions` and `finest_normals` are the face's whole grid at `finest`; the normals are
/// unit normals, as evaluated.
///
/// # Errors
///
/// [`DisplayError::Shape`] if either grid isn't the face's whole grid at `finest`;
/// [`DisplayError::NoDisplayLevel`] at level 0 (no level below it);
/// [`DisplayError::NotFinite`] for a position or normal that isn't finite; the codec's errors.
pub fn build_face_with(
    codec: &impl DisplayCodec,
    normals: NormalEncoding,
    face: FaceId,
    finest_positions: &Grid,
    finest_normals: &Grid,
    finest: Level,
) -> Result<Vec<DisplayTile>, DisplayError> {
    let side = finest.face_quads() + 1;
    if finest_positions.side() != side || finest_normals.side() != side {
        return Err(DisplayError::Shape);
    }
    if finest == Level::ZERO {
        return Err(DisplayError::NoDisplayLevel(finest));
    }
    let finite = |g: &Grid| g.as_slice().iter().all(|v| v.is_finite());
    if !finite(finest_positions) || !finite(finest_normals) {
        return Err(DisplayError::NotFinite);
    }
    let mut tiles = Vec::new();
    // Each level's face grid of subsampled positions and mean normals, and of decoded positions
    // with the bounds of its tiles, from `finest − 1` down to 0.
    let mut input = finest_positions.clone();
    let mut means = MeanNormals::from_unit(finest_normals.clone());
    let mut decoded_finer = finest_positions.clone();
    let mut finer_bounds: Vec<f32> = Vec::new();
    let mut finer_level = finest;
    while let Some(level) = finer_level.parent() {
        input = subsample(&input)?;
        means = mip_normals(&means)?;
        let mut decoded = input.clone();
        let keys = level_keys(face, level)?;
        let mut bounds = Vec::with_capacity(keys.len());
        for key in keys {
            let region = Region::of_tile(&key);
            let positions_bytes = codec.encode(&key, &block(&input, region)?)?;
            let positions = codec.decode(&key, &positions_bytes)?;
            let tile_means = block(&means.grid, region)?;
            let (shaded, rough) = encode::shade(tile_means.as_slice());
            let normal_bytes = normals.encode(&shaded, &rough)?;
            let (unit, roughness) = normals.decode(shaded.len(), &normal_bytes)?;
            let delta = step_error_at(
                &decoded_finer,
                (2 * region.x0(), 2 * region.y0()),
                &positions,
            )?;
            let below = children_bound(&key, finer_level, finest, &finer_bounds);
            let error = round_up(f64::from(delta) + f64::from(below))?;
            for (i, j) in cells(positions.side()) {
                let slot =
                    decoded
                        .data_mut()
                        .get_mut(at(input.side(), region.x0() + i, region.y0() + j));
                if let (Some(slot), Some(p)) = (slot, positions.get(i, j)) {
                    *slot = p;
                }
            }
            bounds.push(error);
            tiles.push(DisplayTile {
                key,
                positions: positions.as_slice().to_vec(),
                normals: unit,
                roughness,
                error,
                encoded: encode::join_parts(&positions_bytes, &normal_bytes)?,
            });
        }
        decoded_finer = decoded;
        finer_bounds = bounds;
        finer_level = level;
    }
    tiles.sort_by_key(|tile| tile.key);
    Ok(tiles)
}

/// The largest bound of the level-`finer` display tiles under `key` (one level coarser), from
/// `finer_bounds` (that level's tiles, row by row); 0 if `finer` is the finest level, whose
/// surface is exact.
fn children_bound(key: &TileKey, finer: Level, finest: Level, finer_bounds: &[f32]) -> f32 {
    if finer == finest {
        return 0.0;
    }
    let per_side = finer.tiles_per_side();
    let ratio = per_side / key.level().tiles_per_side();
    let (x0, y0) = (u32::from(key.tx()) * ratio, u32::from(key.ty()) * ratio);
    (0..ratio)
        .flat_map(|dy| (0..ratio).map(move |dx| (dx, dy)))
        .filter_map(|(dx, dy)| finer_bounds.get(at(per_side, x0 + dx, y0 + dy)))
        .fold(0.0, |a, &b| a.max(b))
}
