//! Building display tiles from the finest level.
//!
//! Subsampling, the error a level adds and its chained bound (positions, a face at a time), the
//! normal cones chained the same way (a face at a time), the normals' mip across faces (a level
//! at a time), and the three joined into tiles.

use std::collections::BTreeMap;

use super::encode::{self, CHOSEN, CHOSEN_NORMALS, DisplayCodec, NormalEncoding, mean_of, shade};
use super::{DisplayError, DisplayTile, FinestNormalCone};
use crate::eval::gather;
use crate::subdiv::{CornerRing, Padded};
use crate::{BaseMesh, FaceId, Grid, Level, PositionSource, Region, TileKey, Vec3};

/// Unnormalized mean normals over a face's grid at one level, row-major `side²`.
///
/// Each is the tent of the means one level finer around its vertex ([`mip_normals`]), down from
/// the finest unit normals. Their length is at most 1, and less where the normals under a vertex
/// disagree.
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

    /// The means as a grid, for a [`FaceGrids`] of a level's means.
    #[must_use]
    pub fn into_grid(self) -> Grid {
        self.grid
    }
}

/// Whole-face grids of several faces at one level, read as a [`PositionSource`]: a level's
/// finest positions or normals, or its mean normals, for [`mip_normals`] and [`build_mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct FaceGrids {
    /// The level.
    level: Level,
    /// Each face's grid.
    grids: BTreeMap<FaceId, Grid>,
}

impl FaceGrids {
    /// An empty set at `level`.
    #[must_use]
    pub const fn new(level: Level) -> Self {
        Self {
            level,
            grids: BTreeMap::new(),
        }
    }

    /// The level.
    #[must_use]
    pub const fn level(&self) -> Level {
        self.level
    }

    /// Sets `face`'s grid, returning the one it replaces.
    ///
    /// # Errors
    ///
    /// [`DisplayError::Shape`] unless `grid` is a whole face at the set's level.
    pub fn insert(&mut self, face: FaceId, grid: Grid) -> Result<Option<Grid>, DisplayError> {
        if grid.side() == self.level.face_quads() + 1 {
            Ok(self.grids.insert(face, grid))
        } else {
            Err(DisplayError::Shape)
        }
    }

    /// `face`'s grid, if the set holds it.
    #[must_use]
    pub fn get(&self, face: FaceId) -> Option<&Grid> {
        self.grids.get(&face)
    }
}

impl PositionSource for FaceGrids {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        if level != self.level {
            return None;
        }
        self.grids.get(&face)?.get(i, j)
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
pub(crate) fn subsample(finer: &Grid) -> Result<Grid, DisplayError> {
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
/// Computed in `f64` from the `f32` positions, each distance raised by more than its `f64`
/// rounding can have taken off (2⁻⁴⁸ of the largest coordinate it read), and rounded up
/// to an `f32`: a bound itself, not an estimate of one, whatever the magnitudes.
///
/// # Errors
///
/// [`DisplayError::Shape`] unless `finer`'s side is `2 × (coarser's − 1) + 1`.
#[cfg(test)]
pub(crate) fn step_error(finer: &Grid, coarser: &Grid) -> Result<f32, DisplayError> {
    let quads = coarser.side().checked_sub(1).ok_or(DisplayError::Shape)?;
    if finer.side() != 2 * quads + 1 {
        return Err(DisplayError::Shape);
    }
    step_error_at(finer, (0, 0), coarser)
}

/// What one distance's `f64` arithmetic can lose, relative to the largest coordinate it reads.
///
/// The midpoint, the difference, three squares, two sums and the square root each round by at
/// most half an `f64` step (2⁻⁵³ relative) of values at most twice that coordinate: seven of
/// them stay under 2⁻⁴⁸, the slack added (its biased exponent is `1023 − 48`).
const ROUNDING: f64 = f64::from_bits((1023 - 48) << 52);

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
        let (drawn, magnitude) = drawn_at(coarser, i, j).ok_or(DisplayError::Shape)?;
        let scale = wide(p).into_iter().fold(magnitude, |m, c| m.max(c.abs()));
        worst = worst.max(distance(p, drawn) + scale * ROUNDING);
    }
    round_up(worst)
}

/// The point `coarser` draws at vertex `(i, j)` of the grid one level finer, in `f64`.
///
/// That is the coarse vertex, or the midpoint of the coarse edge or quad diagonal the vertex
/// lies on (see [`step_error`]), with the largest magnitude of the coordinates it was computed
/// from. `None` outside the grid.
fn drawn_at(coarser: &Grid, i: u32, j: u32) -> Option<([f64; 3], f64)> {
    let largest = |v: [f64; 3]| v.into_iter().fold(0.0_f64, |m, c| m.max(c.abs()));
    let (ci, cj) = (i / 2, j / 2);
    let (a, b) = match (i % 2, j % 2) {
        (0, 0) => {
            let v = wide(coarser.get(ci, cj)?);
            return Some((v, largest(v)));
        }
        (1, 0) => (coarser.get(ci, cj)?, coarser.get(ci + 1, cj)?),
        (0, _) => (coarser.get(ci, cj)?, coarser.get(ci, cj + 1)?),
        _ => (coarser.get(ci, cj)?, coarser.get(ci + 1, cj + 1)?),
    };
    let (a, b) = (wide(a), wide(b));
    let ([ax, ay, az], [bx, by, bz]) = (a, b);
    Some((
        [
            f64::midpoint(ax, bx),
            f64::midpoint(ay, by),
            f64::midpoint(az, bz),
        ],
        largest(a).max(largest(b)),
    ))
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

/// `a + b` in `f64`, moved up a step if the sum rounded below the exact one (an `f32` sum can
/// be inexact in `f64` when the magnitudes are far apart), so it is never less than `a + b`.
fn sum_up(a: f32, b: f32) -> f64 {
    let (a, b) = (f64::from(a), f64::from(b));
    let sum = a + b;
    // Knuth's two-sum: the exact sum is `sum + error`.
    let b_part = sum - a;
    let error = (a - (sum - b_part)) + (b - b_part);
    if error > 0.0 { sum.next_up() } else { sum }
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

/// The mean normals of `face` at `level`, from every face's means one level finer in `finer`:
/// at each vertex, a 3×3 tent over the finer level's vertex there and its neighbours, read
/// across face borders.
///
/// The tent weighs the vertex ¼, its edge neighbours ⅛ and its diagonal ones 1/16. At a base
/// vertex (a face corner) of valence `n`, it is the vertex ¼, each of its `n` edge neighbours
/// `1/(2n)` and each of its `n` diagonal ones `1/(4n)`, summed in the ring's canonical order.
/// Every read goes to the vertex's owner copy, and the tent pairs opposite neighbours
/// symmetrically, so a vertex several faces share gets the same bits from every one of them.
/// The means stay unnormalized, so their length still says how much the normals under them
/// agree.
///
/// `finer` holds level `level + 1` of `face`, the one-vertex apron around it on its neighbours,
/// and the rings of its corners' base vertices; [`FaceGrids`] of every face does.
///
/// # Errors
///
/// [`DisplayError::Unavailable`] if `finer` lacks a vertex the tents read;
/// [`DisplayError::NoDisplayLevel`] at the deepest level (nothing finer);
/// [`DisplayError::NotFinite`] if a mean isn't finite.
pub fn mip_normals(
    mesh: &BaseMesh,
    finer: &impl PositionSource,
    face: FaceId,
    level: Level,
) -> Result<MeanNormals, DisplayError> {
    mip_region(mesh, finer, Region::whole_face(face, level))
}

/// The mean normals of display tile `key` alone (its block of its face's grid), from every
/// face's means one level finer in `finer`: [`mip_normals`] over the tile, the same bits.
///
/// `finer` holds level `key.level() + 1` over the tile's area, its one-vertex apron (on
/// neighbouring faces where the tile meets a face border) and the rings of the base vertices at
/// its corners.
///
/// # Errors
///
/// As [`mip_normals`].
pub fn mip_normals_tile(
    mesh: &BaseMesh,
    finer: &impl PositionSource,
    key: &TileKey,
) -> Result<MeanNormals, DisplayError> {
    mip_region(mesh, finer, Region::of_tile(key))
}

/// [`mip_normals`] over `region` (at the coarser level) alone.
///
/// # Errors
///
/// As [`mip_normals`].
fn mip_region(
    mesh: &BaseMesh,
    finer: &impl PositionSource,
    region: Region,
) -> Result<MeanNormals, DisplayError> {
    let level = region.level();
    let finer_level = level.child().ok_or(DisplayError::NoDisplayLevel(level))?;
    let fine = Region::new(
        region.face(),
        finer_level,
        2 * region.x0(),
        2 * region.y0(),
        2 * region.quads(),
    )
    .ok_or(DisplayError::Shape)?;
    let gathered = gather(mesh, finer, fine)?;
    let quads = fine.quads();
    let side = region.quads() + 1;
    let data = cells(side)
        .map(|(i, j)| {
            let (x, y) = (2 * i, 2 * j);
            // The block's corners, where `gather` has the ring of a base vertex (a face corner).
            let corner = match (x, y) {
                (0, 0) => Some(0),
                (_, 0) if x == quads => Some(1),
                _ if x == quads && y == quads => Some(2),
                (0, _) if y == quads => Some(3),
                _ => None,
            };
            let mean = match corner.and_then(|c| gathered.corners.get(c)?.as_ref()) {
                Some(ring) => ring_tent(ring),
                None => grid_tent(&gathered.padded, i64::from(x), i64::from(y)),
            };
            if mean.is_finite() {
                Ok(mean)
            } else {
                Err(DisplayError::NotFinite)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MeanNormals {
        grid: Grid::new(side, data)?,
    })
}

/// The tent at `(x, y)` of a padded block away from base vertices.
///
/// Opposite neighbours are summed in pairs and the pairs together, so the bits don't depend on
/// which way the face's grid runs (any rotation or reflection maps each pair to a pair).
fn grid_tent(padded: &Padded, x: i64, y: i64) -> Vec3 {
    let at = |dx: i64, dy: i64| padded.get(x + dx, y + dy);
    let edges = (at(-1, 0) + at(1, 0)) + (at(0, -1) + at(0, 1));
    let diagonals = (at(-1, -1) + at(1, 1)) + (at(1, -1) + at(-1, 1));
    (at(0, 0) * 4.0 + edges * 2.0 + diagonals) * (1.0 / 16.0)
}

/// The tent at a base vertex: the vertex ¼, each edge neighbour `1/(2n)` and each diagonal one
/// `1/(4n)` at valence `n`, summed in the ring's canonical order (the same from every face).
fn ring_tent(ring: &CornerRing) -> Vec3 {
    let sum = |points: &[Vec3]| points.iter().fold(Vec3::ZERO, |s, &p| s + p);
    let n = f32::from(u8::try_from(ring.edges().len()).unwrap_or(u8::MAX));
    ring.center * 0.25 + sum(ring.edges()) * (0.5 / n) + sum(ring.diagonals()) * (0.25 / n)
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

/// The rows of `grid` covering `region` (on the grid's level), in place, top to bottom: the
/// block [`block`] copies out, read where it is.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the region doesn't lie inside the grid.
fn block_rows(
    grid: &Grid,
    region: Region,
) -> Result<impl Iterator<Item = &[Vec3]> + Clone, DisplayError> {
    let side = usize::try_from(grid.side()).map_err(|_size| DisplayError::Shape)?;
    let at = |v: u32| usize::try_from(v).map_err(|_size| DisplayError::Shape);
    let (x0, y0, width) = (at(region.x0())?, at(region.y0())?, at(region.quads() + 1)?);
    if x0 + width > side || y0 + width > side {
        return Err(DisplayError::Shape);
    }
    Ok(grid
        .as_slice()
        .chunks_exact(side)
        .skip(y0)
        .take(width)
        .filter_map(move |row| row.get(x0..x0 + width)))
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

/// Writes `values`, a tile's at `region`, into the face grid `face`, checking that every vertex
/// already written by another tile gets the same bits again.
///
/// # Errors
///
/// [`DisplayError::Seam`] naming `key` if a shared vertex differs; [`DisplayError::Shape`] if
/// the tile doesn't fit.
fn write_tile(
    face: &mut Grid,
    written: &mut [bool],
    key: TileKey,
    values: &[Vec3],
) -> Result<(), DisplayError> {
    let region = Region::of_tile(&key);
    let (side, tile_side) = (face.side(), region.quads() + 1);
    for ((i, j), &v) in cells(tile_side).zip(values) {
        let index = at(side, region.x0() + i, region.y0() + j);
        let seen = written.get_mut(index).ok_or(DisplayError::Shape)?;
        let slot = face.data_mut().get_mut(index).ok_or(DisplayError::Shape)?;
        if *seen && slot.to_bits() != v.to_bits() {
            return Err(DisplayError::Seam(key));
        }
        *slot = v;
        *seen = true;
    }
    Ok(())
}

/// One display tile's positions: its key, encoded bytes and bound.
#[derive(Debug, Clone, PartialEq)]
struct PositionTile {
    /// The tile.
    key: TileKey,
    /// Its positions as the codec stored them.
    bytes: Vec<u8>,
    /// `e`.
    error: f32,
}

/// One level of a face's display positions.
#[derive(Debug, Clone, PartialEq)]
struct PositionLevel {
    /// The decoded positions of the whole face at the level.
    decoded: Grid,
    /// Its tiles, row by row.
    tiles: Vec<PositionTile>,
}

/// One face's display positions at every level below the finest: each tile's encoded
/// positions and bound `e`, and each level's decoded face grid.
///
/// Positions need nothing from neighbouring faces (border vertices are stored exactly, so they
/// agree across seams by construction), so a face's are built alone, with
/// [`build_positions`]. Normals need their neighbours' ([`mip_normals`]); [`build_level`] joins
/// the two into display tiles.
#[derive(Debug, Clone, PartialEq)]
pub struct FacePositions {
    /// The face.
    face: FaceId,
    /// The finest level, whose surface the tiles' bounds are against.
    finest: Level,
    /// Levels `0..finest`, by level.
    levels: Vec<PositionLevel>,
}

impl FacePositions {
    /// The face.
    #[must_use]
    pub const fn face(&self) -> FaceId {
        self.face
    }

    /// The finest level.
    #[must_use]
    pub const fn finest(&self) -> Level {
        self.finest
    }

    /// The decoded positions of the whole face at `level`, or `None` at the finest level and
    /// above.
    #[must_use]
    pub fn decoded(&self, level: Level) -> Option<&Grid> {
        Some(&self.levels.get(usize::from(level.get()))?.decoded)
    }

    /// The bound `e` of display tile `key` of this face, or `None` if the face holds no such
    /// tile.
    #[must_use]
    pub fn error(&self, key: &TileKey) -> Option<f32> {
        self.tile(key).map(|t| t.error)
    }

    /// Display tile `key`'s positions as its codec stored them, or `None` if the face holds no
    /// such tile: for the encoding study, which measures them apart from the normals.
    #[cfg(any(test, feature = "display-study"))]
    #[must_use]
    pub fn encoded(&self, key: &TileKey) -> Option<&[u8]> {
        self.tile(key).map(|t| t.bytes.as_slice())
    }

    /// Display tile `key` of this face.
    fn tile(&self, key: &TileKey) -> Option<&PositionTile> {
        if key.face() != self.face {
            return None;
        }
        let level = self.levels.get(usize::from(key.level().get()))?;
        let per_side = key.level().tiles_per_side();
        level
            .tiles
            .get(at(per_side, u32::from(key.tx()), u32::from(key.ty())))
    }
}

/// `face`'s display positions at every level below `finest`, with the encoding chosen by
/// measurement.
///
/// Each level's positions are the decoded level below subsampled (the finest one below the
/// finest), and each tile's bound `e` is chained from the finest level up through the *decoded*
/// positions (see the module notes), so the encoding's quantization is inside it.
/// `finest_positions` is the face's whole grid at `finest`.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the grid isn't the face's whole grid at `finest`;
/// [`DisplayError::NoDisplayLevel`] at level 0 (no level below it);
/// [`DisplayError::NotFinite`] for a position that isn't finite;
/// [`DisplayError::Seam`] if the encoding decodes a vertex two tiles share to different bits in
/// each; [`DisplayError::TooLarge`] for a position out of the encoding's range.
pub fn build_positions(
    face: FaceId,
    finest_positions: &Grid,
    finest: Level,
) -> Result<FacePositions, DisplayError> {
    build_positions_with(&CHOSEN, face, finest_positions, finest)
}

/// [`build_positions`] with positions stored with `codec`.
///
/// # Errors
///
/// As [`build_positions`], and the codec's errors.
#[cfg_attr(
    not(any(test, feature = "display-study")),
    expect(
        unreachable_pub,
        reason = "public for the encoding study (the display-study feature) only"
    )
)]
pub fn build_positions_with(
    codec: &impl DisplayCodec,
    face: FaceId,
    finest_positions: &Grid,
    finest: Level,
) -> Result<FacePositions, DisplayError> {
    if finest_positions.side() != finest.face_quads() + 1 {
        return Err(DisplayError::Shape);
    }
    if finest == Level::ZERO {
        return Err(DisplayError::NoDisplayLevel(finest));
    }
    if !finest_positions.as_slice().iter().all(|v| v.is_finite()) {
        return Err(DisplayError::NotFinite);
    }
    let mut levels = Vec::new();
    let mut decoded_finer = finest_positions.clone();
    let mut finer_bounds: Vec<f32> = Vec::new();
    let mut finer_level = finest;
    while let Some(level) = finer_level.parent() {
        let input = subsample(&decoded_finer)?;
        let mut decoded = input.clone();
        let mut written = vec![false; decoded.as_slice().len()];
        let keys = level_keys(face, level)?;
        let mut tiles = Vec::with_capacity(keys.len());
        for key in keys {
            let region = Region::of_tile(&key);
            let fine = Region::new(
                region.face(),
                finer_level,
                2 * region.x0(),
                2 * region.y0(),
                2 * region.quads(),
            )
            .ok_or(DisplayError::Shape)?;
            let below = children_bound(&key, finer_level, finest, &finer_bounds);
            let (bytes, positions, error) = positions_tile(
                codec,
                &key,
                &block(&input, region)?,
                &block(&decoded_finer, fine)?,
                below,
            )?;
            write_tile(&mut decoded, &mut written, key, positions.as_slice())?;
            tiles.push(PositionTile { key, bytes, error });
        }
        finer_bounds = tiles.iter().map(|t| t.error).collect();
        decoded_finer = decoded.clone();
        levels.push(PositionLevel { decoded, tiles });
        finer_level = level;
    }
    levels.reverse();
    Ok(FacePositions {
        face,
        finest,
        levels,
    })
}

/// One display tile built from its inputs alone, with the encoding chosen by measurement: what a
/// store brings up to date after an edit, one tile at a time, the same bits as [`build_mesh`]
/// gives.
///
/// `finer` is the block under it one level finer (side `2 × (side − 1) + 1`): the finest
/// positions there if that is the finest level, else the decoded positions of the display tiles
/// there; the tile stores its even vertices. `children_bound` is the largest bound of those
/// tiles (0 at the finest level); `cone` the tile's normal cone ([`FinestNormalCone::covering`] the
/// finest normals under it one level above the finest, else [`FinestNormalCone::holding`] the cones
/// of those tiles, as [`build_cones`] chains them); `means` the tile's mean normals
/// ([`mip_normals_tile`]).
///
/// # Errors
///
/// [`DisplayError::Shape`] if a block doesn't fit the key; [`DisplayError::NotFinite`] for a
/// position or mean that isn't finite; [`DisplayError::NoDisplayLevel`] for a key of a non-display
/// channel.
pub fn build_tile(
    key: TileKey,
    finer: &Grid,
    children_bound: f32,
    cone: FinestNormalCone,
    means: &MeanNormals,
) -> Result<DisplayTile, DisplayError> {
    if !(finer.as_slice().iter().all(|v| v.is_finite()) && children_bound.is_finite()) {
        return Err(DisplayError::NotFinite);
    }
    let exact = subsample(finer)?;
    let (bytes, positions, error) = positions_tile(&CHOSEN, &key, &exact, finer, children_bound)?;
    let (normal_bytes, normals, roughness) = normals_tile(CHOSEN_NORMALS, &positions, &means.grid)?;
    Ok(DisplayTile {
        key,
        positions: positions.as_slice().to_vec(),
        normals,
        roughness,
        error,
        cone,
        encoded: encode::join_parts(&bytes, &normal_bytes)?,
    })
}

/// One tile's positions: encoded with `codec` from `exact`, decoded, and bounded against `finer`
/// (the block one level finer) and `children_bound`. Returns the bytes, the decoded block and
/// the bound.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the blocks don't fit; the codec's errors.
fn positions_tile(
    codec: &impl DisplayCodec,
    key: &TileKey,
    exact: &Grid,
    finer: &Grid,
    children_bound: f32,
) -> Result<(Vec<u8>, Grid, f32), DisplayError> {
    let bytes = codec.encode(key, exact)?;
    let positions = codec.decode(key, &bytes)?;
    let delta = step_error_at(finer, (0, 0), &positions)?;
    let error = round_up(sum_up(delta, children_bound))?;
    Ok((bytes, positions, error))
}

/// A tile's normals as stored: the bytes, and the unit normals and roughness they decode to.
type StoredNormals = (Vec<u8>, Vec<Vec3>, Vec<u8>);

/// One tile's normals, as stored.
///
/// The unit normals and roughness of `means` (its block), encoded with `normals` against the
/// tile's decoded `positions`, and decoded.
///
/// # Errors
///
/// The encoding's errors.
fn normals_tile(
    normals: NormalEncoding,
    positions: &Grid,
    means: &Grid,
) -> Result<StoredNormals, DisplayError> {
    if means.side() != positions.side() {
        return Err(DisplayError::Shape);
    }
    let (shaded, rough) = shade(means.as_slice());
    let bytes = normals.encode(&shaded, &rough, positions)?;
    let (unit, roughness) = normals.decode(positions, &bytes)?;
    Ok((bytes, unit, roughness))
}

/// The display tiles of one face at `level`: its `positions` and `cones` there, with `means`
/// (the face's mean normals at the level, from [`mip_normals`]) stored with the normal encoding
/// chosen by measurement.
///
/// Also gives back the means as they decode (each decoded unit normal at its roughness's
/// length): what the next coarser level's [`mip_normals`] reads, so a level is computed from
/// what is stored one level finer, and the store can bring a level up to date from the stored
/// tiles below it.
///
/// # Errors
///
/// [`DisplayError::Shape`] if `means` isn't the face's whole grid at `level`, or `cones` aren't
/// the same face's; [`DisplayError::NoDisplayLevel`] if `level` isn't below the finest;
/// [`DisplayError::Seam`] if a vertex two tiles share decodes differently in each;
/// [`DisplayError::NotFinite`] for a mean that isn't finite.
pub fn build_level(
    positions: &FacePositions,
    cones: &FaceCones,
    level: Level,
    means: &MeanNormals,
) -> Result<(Vec<DisplayTile>, MeanNormals), DisplayError> {
    level_tiles(CHOSEN_NORMALS, positions, Some(cones), level, means)
}

/// [`build_level`] with normals stored with `normals`, and every tile's cone the one that
/// covers everything: the encoding study measures positions and normals, never cones.
///
/// # Errors
///
/// As [`build_level`]; [`DisplayError::Shape`] if the encoding's bits are out of range.
#[cfg(any(test, feature = "display-study"))]
pub fn build_level_with(
    normals: NormalEncoding,
    positions: &FacePositions,
    level: Level,
    means: &MeanNormals,
) -> Result<(Vec<DisplayTile>, MeanNormals), DisplayError> {
    level_tiles(normals, positions, None, level, means)
}

/// [`build_level`] with normals stored with `normals`, and each tile's cone from `cones` (the
/// cone that covers everything without them).
///
/// # Errors
///
/// As [`build_level`]; [`DisplayError::Shape`] if the encoding's bits are out of range.
fn level_tiles(
    normals: NormalEncoding,
    positions: &FacePositions,
    cones: Option<&FaceCones>,
    level: Level,
    means: &MeanNormals,
) -> Result<(Vec<DisplayTile>, MeanNormals), DisplayError> {
    let layer = positions
        .levels
        .get(usize::from(level.get()))
        .ok_or(DisplayError::NoDisplayLevel(level))?;
    if means.side() != level.face_quads() + 1 {
        return Err(DisplayError::Shape);
    }
    let mut decoded_means = means.grid.clone();
    let mut written = vec![false; decoded_means.as_slice().len()];
    let mut tiles = Vec::with_capacity(layer.tiles.len());
    for tile in &layer.tiles {
        let region = Region::of_tile(&tile.key);
        let tile_positions = block(&layer.decoded, region)?;
        let (normal_bytes, unit, roughness) =
            normals_tile(normals, &tile_positions, &block(&means.grid, region)?)?;
        let decoded: Vec<Vec3> = unit
            .iter()
            .zip(&roughness)
            .map(|(&n, &r)| mean_of(n, r))
            .collect();
        write_tile(&mut decoded_means, &mut written, tile.key, &decoded)?;
        let cone = match cones {
            Some(cones) => cones.cone(&tile.key).ok_or(DisplayError::Shape)?,
            None => FinestNormalCone::EVERYTHING,
        };
        tiles.push(DisplayTile {
            key: tile.key,
            positions: tile_positions.as_slice().to_vec(),
            normals: unit,
            roughness,
            error: tile.error,
            cone,
            encoded: encode::join_parts(&tile.bytes, &normal_bytes)?,
        });
    }
    Ok((
        tiles,
        MeanNormals {
            grid: decoded_means,
        },
    ))
}

/// Every display tile of every face of `mesh` at every level below the finest, in key order.
///
/// Built with the encoding chosen by measurement, from each face's finest positions and unit
/// normals (at `finest_positions`' level).
///
/// Positions and cones are built face by face ([`build_positions`], [`build_cones`]); normals
/// level by level over all faces, finest first, each level's [`mip_normals`] reading every face's
/// decoded means one level finer ([`build_level`]). This holds the whole mesh in memory: the store
/// builds the same tiles from the same pieces, a region at a time.
///
/// # Errors
///
/// [`DisplayError::Unavailable`] if a face of the mesh is missing from either grid set, or the
/// two sets aren't at the same level; the pieces' errors.
pub fn build_mesh(
    mesh: &BaseMesh,
    finest_positions: &FaceGrids,
    finest_normals: &FaceGrids,
) -> Result<Vec<DisplayTile>, DisplayError> {
    build_mesh_with(
        &CHOSEN,
        CHOSEN_NORMALS,
        mesh,
        finest_positions,
        finest_normals,
    )
}

/// [`build_mesh`] with positions stored with `codec` and normals with `normals`.
///
/// # Errors
///
/// As [`build_mesh`], and the codec's errors.
#[cfg_attr(
    not(any(test, feature = "display-study")),
    expect(
        unreachable_pub,
        reason = "public for the encoding study (the display-study feature) only"
    )
)]
pub fn build_mesh_with(
    codec: &impl DisplayCodec,
    normals: NormalEncoding,
    mesh: &BaseMesh,
    finest_positions: &FaceGrids,
    finest_normals: &FaceGrids,
) -> Result<Vec<DisplayTile>, DisplayError> {
    let finest = finest_positions.level();
    if finest_normals.level() != finest {
        return Err(DisplayError::Unavailable);
    }
    let faces: Vec<FaceId> = mesh.faces().collect();
    let positions = faces
        .iter()
        .map(|&face| {
            let grid = finest_positions
                .get(face)
                .ok_or(DisplayError::Unavailable)?;
            build_positions_with(codec, face, grid, finest)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let cones = faces
        .iter()
        .map(|&face| {
            let grid = finest_normals.get(face).ok_or(DisplayError::Unavailable)?;
            build_cones(face, grid, finest)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut tiles = Vec::new();
    let mut finer: Option<FaceGrids> = None;
    let mut level = finest;
    while let Some(coarser) = level.parent() {
        let source = finer.as_ref().unwrap_or(finest_normals);
        let mut decoded = FaceGrids::new(coarser);
        let by_face = positions.iter().zip(&cones);
        for (face_positions, face_cones) in by_face {
            let face = face_positions.face();
            let means = mip_normals(mesh, source, face, coarser)?;
            let (built, decoded_means) =
                level_tiles(normals, face_positions, Some(face_cones), coarser, &means)?;
            tiles.extend(built);
            decoded.insert(face, decoded_means.grid)?;
        }
        finer = Some(decoded);
        level = coarser;
    }
    tiles.sort_by_key(|tile| tile.key);
    Ok(tiles)
}

/// The largest bound of the level-`finer` display tiles under `key` (one level coarser), from
/// `finer_bounds` (that level's tiles, row by row); 0 if `finer` is the finest level, whose
/// surface is exact.
pub(super) fn children_bound(
    key: &TileKey,
    finer: Level,
    finest: Level,
    finer_bounds: &[f32],
) -> f32 {
    if finer == finest {
        return 0.0;
    }
    children(key, finer, finer_bounds).fold(0.0, |a, &b| a.max(b))
}

/// The values of the level-`finer` display tiles under `key` (one level coarser), row by row,
/// from `finer_values` (that level's tiles, row by row).
fn children<'a, T>(
    key: &TileKey,
    finer: Level,
    finer_values: &'a [T],
) -> impl Iterator<Item = &'a T> {
    let per_side = finer.tiles_per_side();
    let ratio = per_side / key.level().tiles_per_side();
    let (x0, y0) = (u32::from(key.tx()) * ratio, u32::from(key.ty()) * ratio);
    (0..ratio)
        .flat_map(move |dy| (0..ratio).map(move |dx| (dx, dy)))
        .filter_map(move |(dx, dy)| finer_values.get(at(per_side, x0 + dx, y0 + dy)))
}

/// One face's normal cones at every level below the finest: each display tile's, holding every
/// finest normal under it (see [`FinestNormalCone`]).
///
/// Built with [`build_cones`]: like positions, cones need nothing from a neighbouring face (a
/// cone holds the normals under its tile, not its apron's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceCones {
    /// The face.
    face: FaceId,
    /// Levels `0..finest`, by level: each level's tiles' cones, row by row.
    levels: Vec<Vec<FinestNormalCone>>,
}

impl FaceCones {
    /// The face.
    #[must_use]
    pub const fn face(&self) -> FaceId {
        self.face
    }

    /// The cone of display tile `key` of this face, or `None` if the face holds no such tile.
    #[must_use]
    pub fn cone(&self, key: &TileKey) -> Option<FinestNormalCone> {
        if key.face() != self.face {
            return None;
        }
        let level = self.levels.get(usize::from(key.level().get()))?;
        let per_side = key.level().tiles_per_side();
        level
            .get(at(per_side, u32::from(key.tx()), u32::from(key.ty())))
            .copied()
    }
}

/// `face`'s normal cones at every level below `finest`, from `finest_normals`, the face's whole
/// grid of unit normals at `finest`.
///
/// One level below the finest, each tile's cone is [`FinestNormalCone::covering`] the finest
/// normals under it (its block, borders included, row by row); above, [`FinestNormalCone::holding`]
/// the cones of the tiles under it. So a tile's cone holds every finest normal under it, and its
/// children's cones.
///
/// # Errors
///
/// [`DisplayError::Shape`] if the grid isn't the face's whole grid at `finest`;
/// [`DisplayError::NoDisplayLevel`] at level 0 (no level below it);
/// [`DisplayError::NotFinite`] for a normal that isn't finite.
pub fn build_cones(
    face: FaceId,
    finest_normals: &Grid,
    finest: Level,
) -> Result<FaceCones, DisplayError> {
    if finest_normals.side() != finest.face_quads() + 1 {
        return Err(DisplayError::Shape);
    }
    if finest == Level::ZERO {
        return Err(DisplayError::NoDisplayLevel(finest));
    }
    let mut levels: Vec<Vec<FinestNormalCone>> = Vec::new();
    let mut finer_level = finest;
    while let Some(level) = finer_level.parent() {
        let finer_cones = levels.last().map_or(&[][..], Vec::as_slice);
        let cones = level_keys(face, level)?
            .iter()
            .map(|key| {
                if finer_level != finest {
                    let under: Vec<FinestNormalCone> =
                        children(key, finer_level, finer_cones).copied().collect();
                    return FinestNormalCone::holding(&under);
                }
                let region = Region::of_tile(key);
                let fine = Region::new(
                    region.face(),
                    finer_level,
                    2 * region.x0(),
                    2 * region.y0(),
                    2 * region.quads(),
                )
                .ok_or(DisplayError::Shape)?;
                let rows = block_rows(finest_normals, fine)?;
                FinestNormalCone::covering_each(rows.flat_map(|row| row.iter().copied()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        levels.push(cones);
        finer_level = level;
    }
    levels.reverse();
    Ok(FaceCones { face, levels })
}
