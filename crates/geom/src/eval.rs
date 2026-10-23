//! Evaluating a block: gather the parent level, predict, build frames, apply corrections.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::frame::{self, Block, CornerAxes, Frame, Stencil, frames};
use crate::subdiv::{CornerRing, Padded, extraordinary, predict, ring_edge_point};
use crate::topology::{Ring, Topology, corner_at, edge_to_grid};
use crate::{
    BaseMesh, Channel, FaceId, LayerId, Level, MAX_VALENCE, Region, TILE_QUADS, TileKey, Vec3,
};

/// Read access to evaluated positions.
///
/// An implementation must resolve each `(face, level, i, j)` to exactly one stored copy. A vertex
/// on a tile border belongs to several tiles of the same face; answer from
/// [`TileKey::containing`]'s tile. (Across faces, `geom` already reads only the owner face's
/// copy.)
pub trait PositionSource {
    /// The evaluated position of face-grid vertex `(i, j)` of `face` at `level`, or `None` if it
    /// isn't available.
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3>;
}

/// A block of evaluated positions: `side²` vertices, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Vertices per side.
    side: u32,
    /// The `side²` positions, row-major.
    data: Vec<Vec3>,
}

impl Grid {
    /// A grid of `side` vertices per side holding `data`, row-major; the caller makes `data`
    /// `side²` long.
    pub(crate) const fn from_parts(side: u32, data: Vec<Vec3>) -> Self {
        Self { side, data }
    }

    /// Vertices per side.
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.side
    }

    /// The position at `(i, j)`, or `None` outside the block.
    #[must_use]
    pub fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        if i >= self.side || j >= self.side {
            return None;
        }
        self.data
            .get(usize::try_from(j * self.side + i).ok()?)
            .copied()
    }

    /// All positions, row-major.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec3] {
        &self.data
    }
}

/// Why a block couldn't be evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EvalError {
    /// Level 0 is the base mesh; it has nothing to predict from.
    #[error("level 0 has no parent; its positions come from the base mesh")]
    LevelZero,
    /// The region doesn't start and end on even grid lines, so it has no exact parent block.
    /// [`Region::new`] never makes one above level 0.
    #[error("the region doesn't halve exactly into a parent block")]
    NoExactParent,
    /// The parent level's position wasn't available from the source.
    #[error("position of face {face} level {level} vertex ({i}, {j}) is not available")]
    Missing {
        /// Face.
        face: FaceId,
        /// Level.
        level: Level,
        /// Column.
        i: u32,
        /// Row.
        j: u32,
    },
    /// An output buffer doesn't hold exactly one value per block vertex.
    #[error("the output buffer holds {got} values; the block has {expected} vertices")]
    BufferSize {
        /// Vertices in the block: `side²`.
        expected: usize,
        /// Length of the buffer given.
        got: usize,
    },
}

/// Checks that `out` holds exactly `expected` values.
///
/// # Errors
///
/// [`EvalError::BufferSize`] otherwise.
const fn check_buffer(out: &[Vec3], expected: usize) -> Result<(), EvalError> {
    if out.len() == expected {
        Ok(())
    } else {
        Err(EvalError::BufferSize {
            expected,
            got: out.len(),
        })
    }
}

/// The parent level's block, ring and base-vertex rings.
#[derive(Debug)]
pub(crate) struct Parent {
    /// The block with its one-vertex ring, each vertex read from its owner's copy.
    pub(crate) padded: Padded,
    /// Per block corner (`(0, 0)`, `(q, 0)`, `(q, q)`, `(0, q)`), the ring of the base vertex
    /// there; `None` where the corner isn't a base vertex.
    pub(crate) corners: [Option<CornerRing>; 4],
}

/// The error for a position the source can't supply.
fn missing(face: FaceId, level: Level, i: i64, j: i64) -> EvalError {
    EvalError::Missing {
        face,
        level,
        i: u32::try_from(i).unwrap_or(u32::MAX),
        j: u32::try_from(j).unwrap_or(u32::MAX),
    }
}

/// Reads one level's positions from a source the way evaluation does.
///
/// Every read goes to the vertex's owner copy, so all faces compute a shared vertex from the
/// same inputs even if the source's twins disagree.
///
/// Both the full gather ([`gather`]) and the partial normal update ([`refresh_normals_into`])
/// read through it, so a vertex's inputs are the same bits whichever computes it.
struct Reader<'a, S> {
    /// The mesh's topology: owner copies, neighbours across edges, base-vertex rings.
    topology: &'a Topology,
    /// Where positions come from.
    src: &'a S,
    /// The level read.
    level: Level,
    /// Quads per face side at `level`.
    n: u32,
}

impl<'a, S: PositionSource> Reader<'a, S> {
    /// A reader of `level` of `src` over `mesh`.
    fn new(mesh: &'a BaseMesh, src: &'a S, level: Level) -> Self {
        Self {
            topology: mesh.topology(),
            src,
            level,
            n: level.face_quads(),
        }
    }

    /// The owner copy of face-grid vertex `(i, j)` of `f`.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face.
    fn owner(&self, f: FaceId, i: i64, j: i64) -> Result<(FaceId, u32, u32), EvalError> {
        let (Ok(ui), Ok(uj)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing(f, self.level, i, j));
        };
        self.topology
            .owner_copy(f, self.n, ui, uj)
            .ok_or_else(|| missing(f, self.level, i, j))
    }

    /// The position of face-grid vertex `(i, j)` of `f`, read from its owner copy.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face or the source lacks the position.
    fn fetch(&self, f: FaceId, i: i64, j: i64) -> Result<Vec3, EvalError> {
        let (g, gi, gj) = self.owner(f, i, j)?;
        self.src
            .position(g, self.level, gi, gj)
            .ok_or_else(|| missing(g, self.level, i64::from(gi), i64::from(gj)))
    }

    /// The position of face-grid vertex `(i, j)` of `face`, which the caller knows is its own
    /// owner copy (strictly inside a face of the mesh): [`Reader::fetch`] without the lookup.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face or the source lacks the position.
    fn fetch_own(&self, face: FaceId, i: i64, j: i64) -> Result<Vec3, EvalError> {
        let missing = || missing(face, self.level, i, j);
        let (ui, uj) = (u32::try_from(i), u32::try_from(j));
        let (Ok(ui), Ok(uj)) = (ui, uj) else {
            return Err(missing());
        };
        self.src
            .position(face, self.level, ui, uj)
            .ok_or_else(missing)
    }

    /// The tile a read of face-grid vertex `(i, j)` of `f` resolves to: [`TileKey::containing`]
    /// of its owner copy, the tile a source answers it from.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] if `(i, j)` is off the face or no tile holds it.
    fn tile(&self, f: FaceId, i: i64, j: i64) -> Result<TileKey, EvalError> {
        let (g, gi, gj) = self.owner(f, i, j)?;
        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, g, self.level, gi, gj)
            .map(|(key, _, _)| key)
            .ok_or_else(|| missing(g, self.level, i64::from(gi), i64::from(gj)))
    }

    /// The position at point `(i, j)` of `face`'s grid, which may lie one step off the face
    /// across an edge, or NaN where no vertex is there (diagonally off a corner): what a
    /// [`Padded`] block holds at the point.
    ///
    /// # Errors
    ///
    /// As [`Reader::fetch`].
    fn at(&self, face: FaceId, i: i64, j: i64) -> Result<Vec3, EvalError> {
        match self.topology.locate(face, self.n, i, j) {
            Some((g, gi, gj)) => self.fetch(g, i64::from(gi), i64::from(gj)),
            None => Ok(Vec3::NAN),
        }
    }

    /// The tile [`Reader::at`] reads point `(i, j)` of `face`'s grid from, or `None` where it
    /// reads nothing.
    ///
    /// # Errors
    ///
    /// As [`Reader::tile`].
    fn tile_at(&self, face: FaceId, i: i64, j: i64) -> Result<Option<TileKey>, EvalError> {
        self.topology
            .locate(face, self.n, i, j)
            .map(|(g, gi, gj)| self.tile(g, i64::from(gi), i64::from(gj)))
            .transpose()
    }

    /// The ring around face-grid point `(fi, fj)` of `face` if it is a base vertex (a corner of
    /// the face), with `center` as the vertex's own position, else `None`.
    ///
    /// # Errors
    ///
    /// As [`Reader::fetch`], for a ring point.
    fn corner_ring(
        &self,
        face: FaceId,
        (fi, fj): (i64, i64),
        center: Vec3,
    ) -> Result<Option<CornerRing>, EvalError> {
        let Some(ring) = self.ring(face, (fi, fj)) else {
            return Ok(None);
        };
        let Some(&(_, owner_corner)) = ring.entries.first() else {
            return Ok(None);
        };
        let n64 = i64::from(self.n);
        let mut corner = CornerRing::new(center, owner_corner);
        for &(g, c) in &ring.entries {
            let point = |s: i64, d: i64| {
                let (gi, gj) = edge_to_grid(c, n64, s, d);
                self.fetch(g, gi, gj)
            };
            corner.push(point(1, 0)?, point(1, 1)?);
        }
        Ok(Some(corner))
    }

    /// Whether a normal at base vertex `(fi, fj)` of `face` reads a tile `stale` names: the
    /// tiles of its ring's edge points, which with the vertex itself are all a ring normal
    /// reads. `false` if the point isn't a base vertex.
    ///
    /// # Errors
    ///
    /// As [`Reader::tile`], for a ring point.
    fn ring_reads(
        &self,
        face: FaceId,
        (fi, fj): (i64, i64),
        stale: impl Fn(&TileKey) -> bool,
    ) -> Result<bool, EvalError> {
        let Some(ring) = self.ring(face, (fi, fj)) else {
            return Ok(false);
        };
        let n64 = i64::from(self.n);
        for &(g, c) in &ring.entries {
            let (gi, gj) = edge_to_grid(c, n64, 1, 0);
            if stale(&self.tile(g, gi, gj)?) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The topology's ring around face-grid point `(fi, fj)` of `face`, if it is a corner of
    /// the face.
    fn ring(&self, face: FaceId, (fi, fj): (i64, i64)) -> Option<&'a Ring> {
        let n64 = i64::from(self.n);
        let on = |x: i64| x == 0 || x == n64;
        if !(on(fi) && on(fj)) {
            return None;
        }
        let (ui, uj) = (u32::try_from(fi).ok()?, u32::try_from(fj).ok()?);
        self.topology.ring(face, corner_at(ui, uj, self.n))
    }
}

/// Gathers `region` (at its own level) with its one-vertex ring and base-vertex rings.
///
/// Every position is read through one [`Reader`], as the owner's copy: [`parent_tiles`] runs
/// this with a recording source, so what it lists is exactly what is read here.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a needed position.
pub(crate) fn gather(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<Parent, EvalError> {
    let reader = Reader::new(mesh, src, region.level());
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let face = region.face();
    let mut padded = Padded::filled(region.quads(), Vec3::NAN);
    // Strictly inside a face of the mesh, a point is its own owner copy: read it straight away.
    let n = i64::from(reader.n);
    let own = |x: i64| (1..n).contains(&x);
    let on_mesh = reader.topology.contains(face);
    for j in -1..=q + 1 {
        let y = y0 + j;
        for (x, slot) in (x0 - 1..).zip(padded.row_mut(j)) {
            *slot = if on_mesh && own(x) && own(y) {
                reader.fetch_own(face, x, y)?
            } else {
                match reader.topology.locate(face, reader.n, x, y) {
                    Some((g, gi, gj)) => reader.fetch(g, i64::from(gi), i64::from(gj))?,
                    None => Vec3::NAN,
                }
            };
        }
    }
    let mut corners: [Option<CornerRing>; 4] = [None, None, None, None];
    for (slot, (ci, cj)) in corners.iter_mut().zip([(0, 0), (q, 0), (q, q), (0, q)]) {
        // The centre was already read from the owner's copy, like every other input.
        *slot = reader.corner_ring(face, (x0 + ci, y0 + cj), padded.get(ci, cj))?;
    }
    Ok(Parent { padded, corners })
}

/// A block's predicted positions and tangent frames, before corrections: what a brush needs to
/// project its deltas into frames and to re-apply corrections without predicting again.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// The block predicted.
    region: Region,
    /// The predicted positions of the block and its one-vertex ring (the frames' central
    /// differences read the ring at the block's edges); ring slots that would need a missing
    /// diagonal are NaN.
    predicted: Padded,
    /// Each vertex's tangent frame, row-major, `(quads + 1)²`.
    frames: Vec<Frame>,
}

/// The block one level coarser that predicting `region` gathers.
///
/// # Errors
///
/// [`EvalError::LevelZero`] for a level-0 region, or [`EvalError::NoExactParent`] for a region
/// that doesn't halve exactly.
pub(crate) fn parent_region(region: Region) -> Result<Region, EvalError> {
    if region.level() == Level::ZERO {
        return Err(EvalError::LevelZero);
    }
    region.parent().ok_or(EvalError::NoExactParent)
}

/// Predicts `region` (level ≥ 1) from the parent level's positions in `src`, with each vertex's
/// tangent frame.
///
/// # Errors
///
/// [`EvalError::LevelZero`] for a level-0 region, [`EvalError::NoExactParent`] for a region that
/// doesn't halve exactly, or [`EvalError::Missing`] if `src` lacks a parent position.
pub fn predict_region(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<Prediction, EvalError> {
    let parent = gather(mesh, src, parent_region(region)?)?;
    let predicted = predict(&parent.padded, &parent.corners);
    let frames = predicted_frames(mesh, region, &parent, &predicted);
    Ok(Prediction {
        region,
        predicted,
        frames,
    })
}

/// Where `region` sits on its face, for frames and normals.
fn block_of(mesh: &BaseMesh, region: Region) -> Block<'_> {
    Block {
        topology: mesh.topology(),
        face: region.face(),
        face_quads: region.level().face_quads(),
        x0: region.x0(),
        y0: region.y0(),
        quads: region.quads(),
    }
}

/// The tangent frames of `region`, predicted as `predicted` from `parent`.
///
/// Frames at base vertices use the predicted level's ring: the vertex point and every edge point
/// around it, computed from the parent's ring alone so all copies agree.
pub(crate) fn predicted_frames(
    mesh: &BaseMesh,
    region: Region,
    parent: &Parent,
    predicted: &Padded,
) -> Vec<Frame> {
    let edge_points = parent.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| {
            let mut points = [Vec3::NAN; MAX_VALENCE];
            for (k, point) in points.iter_mut().enumerate().take(ring.edges().len()) {
                *point = ring_edge_point(ring, k);
            }
            points
        })
    });
    let axes = std::array::from_fn(|c| {
        let ring = parent.corners.get(c)?.as_ref()?;
        Some(CornerAxes {
            center: extraordinary(ring),
            points: edge_points.get(c)?.as_ref()?.get(..ring.edges().len())?,
            owner_corner: ring.owner_corner,
        })
    });
    frames(&block_of(mesh, region), predicted, &axes)
}

impl Prediction {
    /// The block predicted.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// Vertices per side.
    #[must_use]
    pub const fn side(&self) -> u32 {
        self.region.quads() + 1
    }

    /// The predicted position at block vertex `(i, j)`, or `None` outside the block.
    #[must_use]
    pub fn predicted(&self, i: u32, j: u32) -> Option<Vec3> {
        let side = self.side();
        (i < side && j < side).then(|| self.predicted.get(i64::from(i), i64::from(j)))
    }

    /// The tangent frame at block vertex `(i, j)`, or `None` outside the block.
    #[must_use]
    pub fn frame(&self, i: u32, j: u32) -> Option<Frame> {
        let side = self.side();
        if i >= side || j >= side {
            return None;
        }
        self.frames
            .get(usize::try_from(j * side + i).ok()?)
            .copied()
    }

    /// The final positions: `corrections(i, j)` (block-local coordinates, `0..=quads`) applied
    /// to each predicted position in its frame.
    #[must_use]
    pub fn apply(&self, corrections: impl Fn(u32, u32) -> Vec3) -> Grid {
        let mut data = vec![Vec3::ZERO; self.frames.len()];
        self.write_positions(&corrections, &mut data);
        Grid::from_parts(self.side(), data)
    }

    /// [`Prediction::apply`] into `out`, row-major `side²`, without allocating.
    ///
    /// # Errors
    ///
    /// [`EvalError::BufferSize`] unless `out` holds exactly `side²` values.
    pub fn apply_into(
        &self,
        corrections: impl Fn(u32, u32) -> Vec3,
        out: &mut [Vec3],
    ) -> Result<(), EvalError> {
        check_buffer(out, self.frames.len())?;
        self.write_positions(&corrections, out);
        Ok(())
    }

    /// The final positions into `out` (one per frame, row-major), a row at a time: the
    /// predicted row, the frames' row and the output row as slices.
    fn write_positions(&self, corrections: &impl Fn(u32, u32) -> Vec3, out: &mut [Vec3]) {
        let side = self.side();
        let width = usize::try_from(side).unwrap_or(usize::MAX);
        let rows = self.frames.chunks(width).zip(out.chunks_mut(width));
        for (j, (frames, out_row)) in (0..side).zip(rows) {
            // Columns `0..=quads` of padded row `j`: past the ring's first slot.
            let predicted = self
                .predicted
                .row(i64::from(j))
                .get(1..)
                .unwrap_or_default();
            let vertices = predicted.iter().zip(frames).zip(out_row);
            for (i, ((&p, frame), slot)) in (0..side).zip(vertices) {
                *slot = frame.apply(p, corrections(i, j));
            }
        }
    }
}

/// Evaluates `region` (level ≥ 1): [`predict_region`], then [`Prediction::apply`].
///
/// # Errors
///
/// As [`predict_region`].
pub fn evaluate_region(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    corrections: impl Fn(u32, u32) -> Vec3,
) -> Result<Grid, EvalError> {
    Ok(predict_region(mesh, src, region)?.apply(corrections))
}

/// A source without positions: what [`gathered_tiles`] hands its [`Reader`], whose tile
/// lookups read none.
struct NoPositions;

impl PositionSource for NoPositions {
    fn position(&self, _: FaceId, _: Level, _: u32, _: u32) -> Option<Vec3> {
        None
    }
}

/// The tiles [`gather`] reads for `region`: every tile it resolves a read to, through the
/// owner copy and [`TileKey::containing`], without reading.
///
/// Only the reads that can leave the face are resolved one by one, exactly as [`gather`]
/// resolves them (with the same [`Reader`]): the points of the padded block on or past the
/// face's border, and the rings of base vertices at the block's corners. The rest, the points
/// strictly inside the face, are each their own owner copy, and [`TileKey::containing`] picks
/// a point's tile column from its column alone and its tile row from its row alone; so their
/// tiles are every pairing of the tile columns their columns fall in with the tile rows their
/// rows fall in, found from one point of each. That makes the listing cost the block's
/// perimeter, not its area: a level-9 tile's normal inputs are about 270 lookups instead of
/// 4,500. The tests check it against a recording of [`gather`]'s reads.
///
/// # Errors
///
/// [`EvalError::Missing`] where [`gather`] would miss a read with a source holding every tile
/// (a point that resolves to no tile), which no [`Region`] causes.
fn gathered_tiles(mesh: &BaseMesh, region: Region) -> Result<BTreeSet<TileKey>, EvalError> {
    let reader = Reader::new(mesh, &NoPositions, region.level());
    let face = region.face();
    let n = i64::from(reader.n);
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let mut tiles = BTreeSet::new();
    // The padded block is `-1..=q + 1` on each axis; the face's inside is `1..n`.
    let (xs, ys) = ((x0 - 1, x0 + q + 1), (y0 - 1, y0 + q + 1));
    let inner = |(a, b): (i64, i64)| (a.max(1), b.min(n - 1));
    let ((xa, xb), (ya, yb)) = (inner(xs), inner(ys));
    if xa <= xb && ya <= yb {
        let mut columns: Vec<(i64, TileKey)> = Vec::new();
        for x in xa..=xb {
            let key = reader.tile(face, x, ya)?;
            if columns
                .last()
                .is_none_or(|&(_, last)| last.tx() != key.tx())
            {
                columns.push((x, key));
            }
        }
        let mut rows: Vec<(i64, TileKey)> = Vec::new();
        for y in ya..=yb {
            let key = reader.tile(face, xa, y)?;
            if rows.last().is_none_or(|&(_, last)| last.ty() != key.ty()) {
                rows.push((y, key));
            }
        }
        for &(x, _) in &columns {
            for &(y, _) in &rows {
                tiles.insert(reader.tile(face, x, y)?);
            }
        }
    }
    // Everything else of the padded block: whole rows outside the face's inside, and in the
    // rows inside it, the columns outside.
    for y in ys.0..=ys.1 {
        let row_inside = (1..n).contains(&y);
        for x in xs.0..=xs.1 {
            if row_inside && (1..n).contains(&x) {
                continue;
            }
            if let Some(key) = reader.tile_at(face, x, y)? {
                tiles.insert(key);
            }
        }
    }
    // The rings of base vertices at the block's corners, as `Reader::corner_ring` reads them.
    for (ci, cj) in [(0, 0), (q, 0), (q, q), (0, q)] {
        let Some(ring) = reader.ring(face, (x0 + ci, y0 + cj)) else {
            continue;
        };
        for &(g, c) in &ring.entries {
            for (s, d) in [(1, 0), (1, 1)] {
                let (gi, gj) = edge_to_grid(c, n, s, d);
                tiles.insert(reader.tile(g, gi, gj)?);
            }
        }
    }
    Ok(tiles)
}

/// The tiles of `region`'s parent level that predicting `region` reads.
///
/// They are base-layer displacement keys naming each tile's position (the parent block, its
/// apron, and every face around a base vertex at the block's corners). Reads are resolved to
/// owner copies and then to [`TileKey::containing`], exactly as [`predict_region`] resolves them,
/// so a source holding just these tiles can predict `region`. At level 1 the parent is the base
/// mesh: the keys are level-0 tiles, which `store` serves from the mesh.
///
/// # Errors
///
/// [`EvalError::LevelZero`] or [`EvalError::NoExactParent`].
pub fn parent_tiles(mesh: &BaseMesh, region: Region) -> Result<BTreeSet<TileKey>, EvalError> {
    gathered_tiles(mesh, parent_region(region)?)
}

/// The tiles of `region`'s own level that [`normals`] reads.
///
/// Like [`parent_tiles`], these are base-layer displacement keys naming tile positions (the block,
/// its one-vertex apron, and every face around a base vertex at the block's corners), the tiles
/// of the gather [`normals`] runs, so a source holding just these tiles can compute `region`'s
/// normals. At level 0 the keys are level-0 tiles, which `store` serves from the mesh.
///
/// # Errors
///
/// [`EvalError::Missing`] only if a read falls outside every tile, which no [`Region`] causes.
pub fn normal_tiles(mesh: &BaseMesh, region: Region) -> Result<BTreeSet<TileKey>, EvalError> {
    gathered_tiles(mesh, region)
}

/// The widest level of any tile's dependency cone: the most tiles of one level that evaluating a
/// tile's positions, or its normals, reads, directly or through the tiles it is predicted from.
///
/// The cone of a tile is its own level's tiles to evaluate (the tile, or for its normals the
/// tiles of [`normal_tiles`]), then level by level down to level 1 every tile [`parent_tiles`]
/// lists for a tile of the level above. A cache that evaluates a tile coarse to fine, holding
/// one level's tiles while it builds the next, needs room for two such levels.
///
/// The bound is the larger of two counts, one per kind of level, both derived from the reads:
///
/// - **Levels holding several tiles per face** (finer than `log2(TILE_QUADS)`): at most
///   `max(valence, 4) + 5`. A tile's region starts and ends on even vertices, and the level below
///   reads its half, plus a one-vertex apron and the rings of the base vertices at its corners: an
///   odd range of vertices, which stays inside the tiles next to the region's own. So each level of
///   a cone lies in a 3 × 3 block of tiles around the tile, across face edges where it reaches
///   them. The reads reach at most one face corner, so at most one base vertex's ring: a face is at
///   least two tiles wide at these levels, and the range reaches only one vertex past the tile's
///   own. At a corner of valence `V`, the block's diagonal tile gives way to the corner tiles of
///   all `V` faces around the vertex: those `V`, three more of the tile's own face, and one more
///   along each of the two edges at the corner, so `V + 5` tiles. Away from corners, and at valence
///   3, the block's nine is the most.
/// - **Levels where a tile is a whole face**: [`parent_tiles`] of a face is every face sharing a
///   vertex with it, so each coarser level reaches one ring of faces further. A cone enters these
///   levels at most one ring out from its tile's face, at level `log2(TILE_QUADS)`, and reaches one
///   ring further at each level down to level 1: `log2(TILE_QUADS)` rings in all. Its widest level
///   is so at most the most faces within `log2(TILE_QUADS)` rings of any face (never more than the
///   mesh).
///
/// Both counts are reached: the tests find cones exactly this wide on meshes of valence 3 to
/// [`MAX_VALENCE`], and on a mesh large enough that the rings don't cover it.
///
/// Computed with a breadth-first search from every face: run it once per mesh.
#[must_use]
pub fn cone_width(mesh: &BaseMesh) -> usize {
    let topology = mesh.topology();
    let face_levels = TILE_QUADS.trailing_zeros();
    let tiled = topology.max_valence().max(4) + 5;
    let whole_faces = topology.widest_neighbourhood(face_levels);
    tiled.max(whole_faces)
}

/// Unit normals of `region`'s evaluated positions in `src` (same level).
///
/// Every copy of a shared vertex gets the same bits. Away from base vertices the normal is
/// `du × dv` from central differences in the owner's orientation. At a base vertex `v` (a face
/// corner) with edge neighbours `e_0 … e_{n−1}` at this level, it is the ring normal
/// `Σ_k (e_k − v) × (e_{k+1} − v)`, summed over the whole ring from the owner face, wrapping.
/// [`Prediction`]'s frames use the same rules.
///
/// Non-finite positions give NaN normals. Finite but degenerate ones (a zero cross product or
/// ring sum) give the fallback normal `(0, 0, 1)`, as frames do.
///
/// Like prediction, this reads the region, its one-vertex apron and the rings of the base
/// vertices at its corners, so all must be resident in `src`. At valence 5 or more, a ring
/// includes faces that touch only the vertex, outside the apron.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a position of the region, its apron or a corner's ring.
pub fn normals(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
) -> Result<Grid, EvalError> {
    let side = region.quads() + 1;
    let count = usize::try_from(side * side).unwrap_or(usize::MAX);
    let mut data = vec![Vec3::ZERO; count];
    normals_into(mesh, src, region, &mut data)?;
    Ok(Grid::from_parts(side, data))
}

/// [`normals`] into `out`, row-major `(quads + 1)²`.
///
/// # Errors
///
/// [`EvalError::BufferSize`] unless `out` holds exactly `(quads + 1)²` values, or
/// [`EvalError::Missing`] as [`normals`].
pub fn normals_into(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    out: &mut [Vec3],
) -> Result<(), EvalError> {
    let side = region.quads() + 1;
    check_buffer(out, usize::try_from(side * side).unwrap_or(usize::MAX))?;
    let gathered = gather(mesh, src, region)?;
    // The level's own ring: the base vertex (the owner's copy) and its edge neighbours.
    let axes = gathered.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| CornerAxes {
            center: ring.center,
            points: ring.edges(),
            owner_corner: ring.owner_corner,
        })
    });
    let block = block_of(mesh, region);
    frame::normals(&block, &gathered.padded, &axes, out);
    Ok(())
}

/// Brings `region`'s normals in `out` up to date after positions outside the region changed.
///
/// Recomputes, in place, the normal of every vertex whose inputs include a position read from a
/// tile `stale` names, and returns how many it recomputed.
///
/// `out` must hold the normals [`normals_into`] gave for the region from earlier positions
/// that differ from `src`'s only in tiles `stale` names, and never at a vertex of the region
/// itself. That is the case after an edit that moved vertices of neighbouring tiles but none of
/// the region's own: a brush's neighbours, whose positions didn't change. Every normal is then
/// the same bits a full [`normals_into`] from `src` gives.
///
/// Only the region's border can read a changed position. Inside, a vertex's central
/// differences read its four grid neighbours, all vertices of the region; on the border they
/// reach one vertex past it, into the apron, and at a base vertex (a face corner) the normal
/// reads the vertex's whole ring, which at valence 5 or more reaches faces that touch the
/// region only at the vertex. So the border vertices are visited, and each is recomputed if a
/// read outside the region, or a ring point, resolves (through its owner copy, as every read
/// does) to a tile `stale` names. Reads are resolved to [`TileKey::containing`]'s tile, as
/// [`PositionSource`] answers them.
///
/// Only the positions of the vertices recomputed are read: their four neighbours, or their
/// ring. Those must be resident in `src`.
///
/// # Errors
///
/// [`EvalError::BufferSize`] unless `out` holds exactly `(quads + 1)²` values, or
/// [`EvalError::Missing`] if `src` lacks a position a recomputed normal reads.
pub fn refresh_normals_into(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    stale: impl Fn(&TileKey) -> bool,
    out: &mut [Vec3],
) -> Result<usize, EvalError> {
    let side = region.quads() + 1;
    check_buffer(out, usize::try_from(side * side).unwrap_or(usize::MAX))?;
    let reader = Reader::new(mesh, src, region.level());
    let block = block_of(mesh, region);
    let face = region.face();
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let inside = |(i, j): (i64, i64)| (0..=q).contains(&i) && (0..=q).contains(&j);
    let mut refreshed = 0;
    for (i, j) in border(q) {
        let local = match frame::stencil(&block, i, j) {
            Stencil::Axes([u, v]) => {
                let reads = [
                    (i + u.0, j + u.1),
                    (i - u.0, j - u.1),
                    (i + v.0, j + v.1),
                    (i - v.0, j - v.1),
                ];
                let mut changed = false;
                for (a, b) in reads.into_iter().filter(|&read| !inside(read)) {
                    if reader
                        .tile_at(face, x0 + a, y0 + b)?
                        .is_some_and(|key| stale(&key))
                    {
                        changed = true;
                        break;
                    }
                }
                if !changed {
                    continue;
                }
                let mut values = [Vec3::NAN; 4];
                for (value, (a, b)) in values.iter_mut().zip(reads) {
                    *value = reader.at(face, x0 + a, y0 + b)?;
                }
                frame::central(values)
            }
            Stencil::Corner(_) => {
                let at = (x0 + i, y0 + j);
                if !reader.ring_reads(face, at, &stale)? {
                    continue;
                }
                let center = reader.at(face, at.0, at.1)?;
                match reader.corner_ring(face, at, center)? {
                    Some(ring) => CornerAxes {
                        center: ring.center,
                        points: ring.edges(),
                        owner_corner: ring.owner_corner,
                    }
                    .local(),
                    None => frame::Local::NonFinite,
                }
            }
            Stencil::Outside => continue,
        };
        let index = usize::try_from(j * i64::from(side) + i).unwrap_or(usize::MAX);
        if let Some(slot) = out.get_mut(index) {
            *slot = frame::normal(local);
            refreshed += 1;
        }
    }
    Ok(refreshed)
}

/// The border vertices of a block of `q` quads per side, each once: the bottom and top rows,
/// then the left and right columns between them.
fn border(q: i64) -> impl Iterator<Item = (i64, i64)> {
    let rows = [0, q]
        .into_iter()
        .flat_map(move |j| (0..=q).map(move |i| (i, j)));
    let columns = (1..q).flat_map(move |j| [(0, j), (q, j)]);
    rows.chain(columns)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::reference::pentagonal_prism;
    use crate::test_rng::Rng;
    use crate::topology::edge_to_grid;
    use crate::{CorrectionMap, DenseEvaluation, FaceId, Level, Region, cube_sphere};

    fn level(l: u8) -> Level {
        Level::new(l).unwrap()
    }

    #[test]
    fn level_one_of_a_cube_has_no_nan_and_matches_on_shared_edges() {
        let mesh = cube_sphere(1).unwrap();
        let level = level(1);
        let grids: Vec<Grid> = mesh
            .faces()
            .map(|f| {
                evaluate_region(&mesh, &mesh, Region::whole_face(f, level), |_, _| {
                    Vec3::ZERO
                })
                .unwrap()
            })
            .collect();
        for g in &grids {
            assert!(g.as_slice().iter().all(|p| p.is_finite()), "no NaN");
        }
        for face in mesh.faces() {
            for (i, j) in [(0, 1), (1, 0), (0, 0), (2, 2)] {
                let mine = grids[face.index() as usize].get(i, j).unwrap();
                for copy in mesh.topology().copies(face, level, i, j) {
                    let theirs = grids[copy.face.index() as usize]
                        .get(copy.i, copy.j)
                        .unwrap();
                    assert_eq!(
                        mine.to_bits(),
                        theirs.to_bits(),
                        "{face:?} ({i}, {j}) vs {copy:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn prediction_and_application_split_evaluation() {
        let mesh = cube_sphere(1).unwrap();
        let region = Region::whole_face(FaceId::new(2), level(1));
        let prediction = predict_region(&mesh, &mesh, region).unwrap();
        let c = |i: u32, j: u32| {
            Vec3::new(
                0.01 * f32::from(u16::try_from(i).unwrap()),
                0.02,
                -0.01 * f32::from(u16::try_from(j).unwrap()),
            )
        };
        let whole = evaluate_region(&mesh, &mesh, region, c).unwrap();
        let applied = prediction.apply(c);
        let bits = |g: &Grid| g.as_slice().iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&applied), bits(&whole), "evaluate = predict + apply");
        assert_eq!(prediction.side(), 3, "side");
        for j in 0..3 {
            for i in 0..3 {
                let (p, f) = (
                    prediction.predicted(i, j).unwrap(),
                    prediction.frame(i, j).unwrap(),
                );
                assert_eq!(
                    applied.get(i, j).unwrap().to_bits(),
                    f.apply(p, c(i, j)).to_bits(),
                    "({i}, {j})"
                );
            }
        }
        assert_eq!(prediction.predicted(3, 0), None, "outside: position");
        assert_eq!(prediction.frame(0, 3), None, "outside: frame");
    }

    #[test]
    fn level_zero_has_no_parent() {
        let mesh = cube_sphere(1).unwrap();
        let r = Region::whole_face(FaceId::new(0), Level::ZERO);
        assert_eq!(
            evaluate_region(&mesh, &mesh, r, |_, _| Vec3::ZERO),
            Err(EvalError::LevelZero),
            "level 0"
        );
        assert_eq!(
            parent_tiles(&mesh, r),
            Err(EvalError::LevelZero),
            "level 0: no parent tiles"
        );
    }

    /// Random corrections at levels `1..=top`, half of them on face or tile borders, where seams
    /// can break.
    fn random_corrections(mesh: &BaseMesh, top: u8, edits: u32, seed: u64) -> CorrectionMap {
        let mut rng = Rng::new(seed);
        let mut map = CorrectionMap::new();
        for _ in 0..edits {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let level = level(1 + u8::try_from(rng.below(u32::from(top))).unwrap());
            let (n, tq) = (level.face_quads(), level.tile_quads());
            let pick = |r: &mut Rng| {
                if r.below(2) == 0 {
                    let border = r.below(level.tiles_per_side() + 1) * tq;
                    (border + r.below(3)).saturating_sub(1).min(n)
                } else {
                    r.below(n + 1)
                }
            };
            let (i, j) = (pick(&mut rng), pick(&mut rng));
            let amplitude = 0.5 / f32::from(u16::try_from(n).unwrap());
            map.set(mesh.topology(), face, level, i, j, rng.vec3(amplitude))
                .unwrap();
        }
        map
    }

    /// Asserts `grid` (block `region`) equals the dense evaluation bit for bit.
    fn assert_matches_dense(grid: &Grid, dense: &DenseEvaluation, region: Region, what: &str) {
        let (face, level) = (region.face(), region.level());
        for j in 0..=region.quads() {
            for i in 0..=region.quads() {
                let want = dense
                    .position(face, level, region.x0() + i, region.y0() + j)
                    .unwrap();
                assert_eq!(
                    grid.get(i, j).unwrap().to_bits(),
                    want.to_bits(),
                    "{what} {region:?} ({i}, {j})"
                );
            }
        }
    }

    /// Corrections of `region`'s block from `map`.
    fn block_corrections(map: &CorrectionMap, region: Region) -> impl Fn(u32, u32) -> Vec3 + '_ {
        move |i, j| {
            map.get(
                region.face(),
                region.level(),
                region.x0() + i,
                region.y0() + j,
            )
        }
    }

    /// The dense evaluation, answering only vertices whose containing tile is in `keys`.
    struct Only<'a> {
        dense: &'a DenseEvaluation,
        keys: &'a BTreeSet<TileKey>,
    }

    impl PositionSource for Only<'_> {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let (key, _, _) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
            if !self.keys.contains(&key) {
                return None;
            }
            self.dense.position(face, level, i, j)
        }
    }

    /// The level-`top` tiles of every face whose tile position passes `keep`.
    fn tiles_where(
        mesh: &BaseMesh,
        top: Level,
        keep: impl Fn(u16, u16) -> bool + Copy,
    ) -> Vec<Region> {
        let per_side = u16::try_from(top.tiles_per_side()).unwrap();
        mesh.faces()
            .flat_map(|f| {
                (0..per_side).flat_map(move |ty| {
                    (0..per_side)
                        .filter(move |&tx| keep(tx, ty))
                        .map(move |tx| {
                            Region::of_tile(&TileKey::displacement(f, top, tx, ty).unwrap())
                        })
                })
            })
            .collect()
    }

    /// Each region evaluates bit-identically from only its parent tiles, and each listed tile is
    /// read.
    fn assert_parent_tiles_suffice(
        mesh: &BaseMesh,
        map: &CorrectionMap,
        dense: &DenseEvaluation,
        regions: &[Region],
    ) {
        for &region in regions {
            let keys = parent_tiles(mesh, region).unwrap();
            let parent = region.level().parent().unwrap();
            assert!(
                keys.iter().all(|k| k.level() == parent
                    && k.layer() == LayerId::BASE
                    && k.channel() == Channel::DISPLACEMENT),
                "{region:?}: base-layer displacement keys of the parent level: {keys:?}"
            );
            let src = Only { dense, keys: &keys };
            let grid = evaluate_region(mesh, &src, region, block_corrections(map, region))
                .unwrap_or_else(|e| panic!("{region:?} from {keys:?}: {e}"));
            assert_matches_dense(&grid, dense, region, "from its parent tiles");
            // And no listed tile is spare: without any one of them, prediction misses a read.
            for key in &keys {
                let mut fewer = keys.clone();
                fewer.remove(key);
                let short = Only {
                    dense,
                    keys: &fewer,
                };
                assert!(
                    matches!(
                        predict_region(mesh, &short, region),
                        Err(EvalError::Missing { .. })
                    ),
                    "{region:?}: {key:?} is listed but not read"
                );
            }
        }
    }

    #[test]
    fn parent_tiles_suffice() {
        let top = level(8);
        {
            // Every level-8 tile of the cube, and whole faces at levels 3 and 1 (parent: the
            // mesh).
            let mesh = cube_sphere(1).unwrap();
            let map = random_corrections(&mesh, 8, 800, 3);
            let dense = DenseEvaluation::new(&mesh, top, &map).unwrap();
            let mut regions = tiles_where(&mesh, top, |_, _| true);
            regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(3))));
            regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(1))));
            assert_parent_tiles_suffice(&mesh, &map, &dense, &regions);
        }
        {
            // At valence 3 every face around a base vertex is also across an edge, so the apron
            // already reads it. At valence 5, two faces are reached only through the vertex's
            // ring: check whole faces, and level-8 tiles at face corners, where the ring is read.
            let mesh = pentagonal_prism().subdivide().to_base_mesh();
            let map = random_corrections(&mesh, 8, 800, 4);
            let dense = DenseEvaluation::new(&mesh, top, &map).unwrap();
            let mut regions = tiles_where(&mesh, top, |tx, ty| {
                [0, 3].contains(&tx) && [0, 3].contains(&ty)
            });
            regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(3))));
            regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(1))));
            assert_parent_tiles_suffice(&mesh, &map, &dense, &regions);
        }
    }

    #[test]
    fn valence_five_reads_every_face_around_the_vertex() {
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let t = mesh.topology();
        let (face, ring) = mesh
            .faces()
            .flat_map(|f| (0..4).map(move |c| (f, c)))
            .find_map(|(f, c)| {
                let ring = t.ring(f, c)?;
                (ring.entries.len() == 5).then(|| (f, ring.entries.clone()))
            })
            .expect("the subdivided prism has valence-5 vertices");
        let keys = parent_tiles(&mesh, Region::whole_face(face, level(1))).unwrap();
        for (g, _) in ring {
            let key = TileKey::displacement(g, Level::ZERO, 0, 0).unwrap();
            assert!(
                keys.contains(&key),
                "{face}'s level-1 region reads level 0 of {g}: {keys:?}"
            );
        }
    }

    /// Level-`level` positions held tile by tile, each vertex answered from its containing tile:
    /// how `store`'s cache serves them.
    struct Tiles(HashMap<TileKey, Grid>);

    impl PositionSource for Tiles {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let (key, li, lj) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
            self.0.get(&key)?.get(li, lj)
        }
    }

    #[test]
    fn tile_sources_reproduce_dense() {
        let mesh = cube_sphere(1).unwrap();
        let map = random_corrections(&mesh, 8, 800, 9);
        let dense = DenseEvaluation::new(&mesh, level(8), &map).unwrap();
        let parent = dense.truncated(level(6));
        let level_7 = Tiles(
            tiles_where(&mesh, level(7), |_, _| true)
                .into_iter()
                .map(|r| {
                    let key = TileKey::displacement(
                        r.face(),
                        r.level(),
                        u16::try_from(r.x0() / r.quads()).unwrap(),
                        u16::try_from(r.y0() / r.quads()).unwrap(),
                    )
                    .unwrap();
                    let grid =
                        evaluate_region(&mesh, &parent, r, block_corrections(&map, r)).unwrap();
                    (key, grid)
                })
                .collect(),
        );
        for r in tiles_where(&mesh, level(8), |_, _| true) {
            let grid = evaluate_region(&mesh, &level_7, r, block_corrections(&map, r)).unwrap();
            assert_matches_dense(&grid, &dense, r, "from level-7 tiles");
        }
    }

    #[test]
    fn normals_of_twins_are_bit_identical() {
        assert_normals_of_twins_are_bit_identical(&cube_sphere(2).unwrap(), 21);
        // Valence 3, 4 and 5: the ring sum's canonical order matters at every valence.
        assert_normals_of_twins_are_bit_identical(
            &pentagonal_prism().subdivide().to_base_mesh(),
            22,
        );
    }

    /// Normals at levels 0–3 under random corrections: unit length, bit-identical across every
    /// copy of a shared vertex, and the same from a sub-block as from the whole face.
    fn assert_normals_of_twins_are_bit_identical(mesh: &BaseMesh, seed: u64) {
        let t = mesh.topology();
        let map = random_corrections(mesh, 3, 400, seed);
        let dense = DenseEvaluation::new(mesh, level(3), &map).unwrap();
        for l in 0..=3 {
            let level = level(l);
            let n = level.face_quads();
            let grids: Vec<Grid> = mesh
                .faces()
                .map(|f| normals(mesh, &dense, Region::whole_face(f, level)).unwrap())
                .collect();
            let at = |f: FaceId, i: u32, j: u32| grids[f.index() as usize].get(i, j).unwrap();
            for face in mesh.faces() {
                for j in 0..=n {
                    for i in 0..=n {
                        let mine = at(face, i, j);
                        assert!(
                            (mine.length() - 1.0).abs() < 1e-5,
                            "{face} L{l} ({i}, {j}): unit, got {mine:?}"
                        );
                        for c in t.copies(face, level, i, j) {
                            let r = Region::of_tile(&c.key(level));
                            let theirs = at(c.face, r.x0() + c.i, r.y0() + c.j);
                            assert_eq!(
                                mine.to_bits(),
                                theirs.to_bits(),
                                "{face} L{l} ({i}, {j}) vs {c:?}"
                            );
                        }
                    }
                }
                // A block inside the face gives its vertices the whole face's normals.
                if l >= 2 {
                    let half = n / 2;
                    for (x0, y0) in [(0, 0), (half, 0), (0, half), (half, half)] {
                        let r = Region::new(face, level, x0, y0, half).unwrap();
                        let block = normals(mesh, &dense, r).unwrap();
                        for j in 0..=half {
                            for i in 0..=half {
                                assert_eq!(
                                    block.get(i, j).unwrap().to_bits(),
                                    at(face, x0 + i, y0 + j).to_bits(),
                                    "{r:?} ({i}, {j})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Whether `(i, j)` is a corner of a face with `n` quads per side: a base vertex.
    fn is_corner(i: u32, j: u32, n: u32) -> bool {
        (i == 0 || i == n) && (j == 0 || j == n)
    }

    /// Normals of the unedited cube sphere against radial directions, at every level 0–3.
    ///
    /// Base vertices take the ring normal, which is radial to within `cos > 0.9999` (measured:
    /// above 0.9999999). Elsewhere the central differences reach 0.9987, 0.9981 and 0.9978 at
    /// levels 1–3: the Catmull–Clark surface of a cube sphere isn't a sphere, so radial isn't
    /// its exact normal there.
    #[test]
    fn normals_of_a_sphere_point_outward() {
        let mesh = cube_sphere(2).unwrap();
        let dense = DenseEvaluation::new(&mesh, level(3), &CorrectionMap::new()).unwrap();
        for l in 0..=3 {
            let level = level(l);
            let n = level.face_quads();
            for face in mesh.faces() {
                let grid = normals(&mesh, &dense, Region::whole_face(face, level)).unwrap();
                for j in 0..=n {
                    for i in 0..=n {
                        let p = dense.position(face, level, i, j).unwrap();
                        let normal = grid.get(i, j).unwrap();
                        let cos = normal.dot(p.normalized().unwrap());
                        let bound = if is_corner(i, j, n) { 0.9999 } else { 0.997 };
                        assert!(cos > bound, "{face} L{l} ({i}, {j}): cos {cos}");
                    }
                }
            }
        }
    }

    /// Predicted frames at base vertices take the ring normal of the predicted level: radial on
    /// the unedited sphere.
    #[test]
    fn frames_at_base_vertices_of_a_sphere_point_outward() {
        let mesh = cube_sphere(2).unwrap();
        let parents = DenseEvaluation::new(&mesh, level(2), &CorrectionMap::new()).unwrap();
        for l in 1..=3 {
            let n = level(l).face_quads();
            for face in mesh.faces() {
                let prediction =
                    predict_region(&mesh, &parents, Region::whole_face(face, level(l))).unwrap();
                for (i, j) in [(0, 0), (n, 0), (n, n), (0, n)] {
                    let p = prediction.predicted(i, j).unwrap();
                    let normal = prediction.frame(i, j).unwrap().n();
                    let cos = normal.dot(p.normalized().unwrap());
                    assert!(cos > 0.9999, "{face} L{l} ({i}, {j}): cos {cos}");
                }
            }
        }
    }

    /// At every base vertex, each copy's normal is exactly the ring normal of the dense positions,
    /// summed from the owner: `Σ_k (e_k − v) × (e_{k+1} − v)` over the level's edge neighbours.
    #[test]
    fn normals_at_base_vertices_are_the_ring_normal() {
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let t = mesh.topology();
        let map = random_corrections(&mesh, 2, 200, 31);
        let dense = DenseEvaluation::new(&mesh, level(2), &map).unwrap();
        let mut valences = BTreeSet::new();
        for l in 0..=2 {
            let level = level(l);
            let n = level.face_quads();
            let n64 = i64::from(n);
            for face in mesh.faces() {
                let grid = normals(&mesh, &dense, Region::whole_face(face, level)).unwrap();
                for (i, j) in [(0, 0), (n, 0), (n, n), (0, n)] {
                    let ring = &t.ring(face, corner_at(i, j, n)).unwrap().entries;
                    valences.insert(ring.len());
                    let (owner, owner_corner) = ring[0];
                    let at = |g: FaceId, c: u8, s: i64| {
                        let (gi, gj) = edge_to_grid(c, n64, s, 0);
                        dense
                            .position(
                                g,
                                level,
                                u32::try_from(gi).unwrap(),
                                u32::try_from(gj).unwrap(),
                            )
                            .unwrap()
                    };
                    let v = at(owner, owner_corner, 0);
                    let edges: Vec<Vec3> = ring.iter().map(|&(g, c)| at(g, c, 1)).collect();
                    let sum = (0..edges.len()).fold(Vec3::ZERO, |sum, k| {
                        let next = edges[(k + 1) % edges.len()];
                        sum + (edges[k] - v).cross(next - v)
                    });
                    let want = sum.normalized().unwrap();
                    assert_eq!(
                        grid.get(i, j).unwrap().to_bits(),
                        want.to_bits(),
                        "{face} L{l} ({i}, {j}), ring {ring:?}"
                    );
                }
            }
        }
        assert_eq!(
            valences,
            BTreeSet::from([3, 4, 5]),
            "every valence of the subdivided prism"
        );
    }

    /// Each region's normals come out bit-identical from only its `normal_tiles`, and each listed
    /// tile is read.
    fn assert_normal_tiles_suffice(mesh: &BaseMesh, dense: &DenseEvaluation, regions: &[Region]) {
        for &region in regions {
            let keys = normal_tiles(mesh, region).unwrap();
            assert!(
                keys.iter().all(|k| k.level() == region.level()
                    && k.layer() == LayerId::BASE
                    && k.channel() == Channel::DISPLACEMENT),
                "{region:?}: base-layer displacement keys of the region's level: {keys:?}"
            );
            let want = normals(mesh, dense, region).unwrap();
            let src = Only { dense, keys: &keys };
            let got = normals(mesh, &src, region)
                .unwrap_or_else(|e| panic!("{region:?} from {keys:?}: {e}"));
            let bits = |g: &Grid| g.as_slice().iter().map(|p| p.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{region:?}: from its normal tiles");
            for key in &keys {
                let mut fewer = keys.clone();
                fewer.remove(key);
                let short = Only {
                    dense,
                    keys: &fewer,
                };
                assert!(
                    matches!(
                        normals(mesh, &short, region),
                        Err(EvalError::Missing { .. })
                    ),
                    "{region:?}: {key:?} is listed but not read"
                );
            }
        }
    }

    #[test]
    fn normal_tiles_suffice() {
        let top = level(8);
        let whole_faces = |mesh: &BaseMesh| -> Vec<Region> {
            [0, 1, 3]
                .into_iter()
                .flat_map(|l| mesh.faces().map(move |f| Region::whole_face(f, level(l))))
                .collect()
        };
        {
            // Every level-8 tile of the cube, and whole faces at levels 0, 1 and 3.
            let mesh = cube_sphere(1).unwrap();
            let map = random_corrections(&mesh, 8, 800, 5);
            let dense = DenseEvaluation::new(&mesh, top, &map).unwrap();
            let mut regions = tiles_where(&mesh, top, |_, _| true);
            regions.extend(whole_faces(&mesh));
            assert_normal_tiles_suffice(&mesh, &dense, &regions);
        }
        {
            // Valence 3, 4 and 5: level-8 tiles at face corners, where the rings are read, an
            // interior tile, and whole faces.
            let mesh = pentagonal_prism().subdivide().to_base_mesh();
            let map = random_corrections(&mesh, 8, 800, 6);
            let dense = DenseEvaluation::new(&mesh, top, &map).unwrap();
            let mut regions = tiles_where(&mesh, top, |tx, ty| {
                ([0, 3].contains(&tx) && [0, 3].contains(&ty)) || (tx, ty) == (1, 2)
            });
            regions.extend(whole_faces(&mesh));
            assert_normal_tiles_suffice(&mesh, &dense, &regions);
        }
    }

    /// The widest level of the dependency cones of every tile of levels `from..=to` of `mesh`, and
    /// of their normals' inputs, followed down to level `floor` (1 for whole cones).
    fn widest_cone(mesh: &BaseMesh, from: u8, to: u8, floor: u8) -> usize {
        let mut parents: HashMap<TileKey, BTreeSet<TileKey>> = HashMap::new();
        let mut widest = 0;
        for l in from..=to {
            let per_side = u16::try_from(level(l).tiles_per_side()).unwrap();
            for face in mesh.faces() {
                for (tx, ty) in (0..per_side).flat_map(|ty| (0..per_side).map(move |tx| (tx, ty))) {
                    let key = TileKey::displacement(face, level(l), tx, ty).unwrap();
                    let mut inputs = normal_tiles(mesh, Region::of_tile(&key)).unwrap();
                    inputs.insert(key);
                    for mut cone in [BTreeSet::from([key]), inputs] {
                        widest = widest.max(cone.len());
                        while cone.first().is_some_and(|k| k.level() > level(floor)) {
                            for t in &cone {
                                parents.entry(*t).or_insert_with(|| {
                                    parent_tiles(mesh, Region::of_tile(t)).unwrap()
                                });
                            }
                            cone = cone
                                .iter()
                                .filter_map(|t| parents.get(t))
                                .flatten()
                                .copied()
                                .collect();
                            widest = widest.max(cone.len());
                        }
                    }
                }
            }
        }
        widest
    }

    #[test]
    fn cone_width_bounds_every_cone() {
        use crate::reference::prism;
        let cube = cube_sphere(1).unwrap();
        assert_eq!(
            widest_cone(&cube, 1, 9, 1),
            cone_width(&cube),
            "the cube: a level-8 tile's normals read a 3 × 3 block"
        );
        // Whole cones of every tile, on meshes with valence 3 to 8: small enough that the rings
        // reach every face, so the widest cone is the whole mesh.
        let meshes = [
            cube_sphere(2).unwrap(),
            pentagonal_prism().subdivide().to_base_mesh(),
            prism(8).subdivide().to_base_mesh(),
        ];
        for mesh in &meshes {
            assert_eq!(
                widest_cone(mesh, 1, 8, 1),
                cone_width(mesh),
                "{} faces: the bound is reached",
                mesh.face_count()
            );
        }
        // The multi-tile levels' count, reached at every valence up to the highest.
        for mesh in meshes
            .iter()
            .chain([&prism(MAX_VALENCE).subdivide().to_base_mesh()])
        {
            let valence = mesh.topology().max_valence();
            assert_eq!(
                widest_cone(mesh, 7, 8, 7),
                valence.max(4) + 5,
                "valence {valence}: the multi-tile levels"
            );
        }
        // A mesh the rings don't cover: the whole-face levels' count is reached, in six rings
        // (`log2(TILE_QUADS)`), not seven.
        let large = cube_sphere(7).unwrap();
        let topology = large.topology();
        assert!(
            topology.widest_neighbourhood(7) < usize::try_from(large.face_count()).unwrap(),
            "seven rings don't cover the mesh either"
        );
        assert_eq!(
            widest_cone(&large, 7, 7, 1),
            cone_width(&large),
            "{} faces: the cones into the whole-face levels",
            large.face_count()
        );
        assert_eq!(
            cone_width(&large),
            topology.widest_neighbourhood(6),
            "six rings"
        );
        assert!(
            topology.widest_neighbourhood(7) > cone_width(&large),
            "seven would overcount"
        );
    }

    #[test]
    fn normal_tiles_reach_every_face_around_a_valence_five_vertex() {
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let t = mesh.topology();
        let (face, ring) = mesh
            .faces()
            .flat_map(|f| (0..4).map(move |c| (f, c)))
            .find_map(|(f, c)| {
                let ring = t.ring(f, c)?;
                (ring.entries.len() == 5).then(|| (f, ring.entries.clone()))
            })
            .expect("the subdivided prism has valence-5 vertices");
        let keys = normal_tiles(&mesh, Region::whole_face(face, level(1))).unwrap();
        for (g, _) in ring {
            let key = TileKey::displacement(g, level(1), 0, 0).unwrap();
            assert!(
                keys.contains(&key),
                "{face}'s level-1 normals read {g}: {keys:?}"
            );
        }
    }

    #[test]
    fn apply_into_matches_apply() {
        let mesh = cube_sphere(1).unwrap();
        let map = random_corrections(&mesh, 3, 200, 12);
        let dense = DenseEvaluation::new(&mesh, level(3), &map).unwrap();
        for face in mesh.faces() {
            for l in 1..=3 {
                let region = Region::whole_face(face, level(l));
                let prediction = predict_region(&mesh, &dense, region).unwrap();
                let c = block_corrections(&map, region);
                let grid = prediction.apply(&c);
                let side = usize::try_from(prediction.side()).unwrap();
                let mut out = vec![Vec3::NAN; side * side];
                prediction.apply_into(&c, &mut out).unwrap();
                let bits = |v: &[Vec3]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&out), bits(grid.as_slice()), "{region:?}");
            }
        }
    }

    #[test]
    fn buffers_of_the_wrong_size_are_errors() {
        let mesh = cube_sphere(1).unwrap();
        let region = Region::whole_face(FaceId::new(1), level(2));
        let prediction =
            predict_region(&mesh, &mesh, Region::whole_face(FaceId::new(1), level(1))).unwrap();
        for len in [8, 10, 0] {
            let mut out = vec![Vec3::ZERO; len];
            assert_eq!(
                prediction.apply_into(|_, _| Vec3::ZERO, &mut out),
                Err(EvalError::BufferSize {
                    expected: 9,
                    got: len
                }),
                "apply_into, {len} values"
            );
        }
        let dense = DenseEvaluation::new(&mesh, level(2), &CorrectionMap::new()).unwrap();
        for len in [24, 26] {
            let mut out = vec![Vec3::ZERO; len];
            assert_eq!(
                normals_into(&mesh, &dense, region, &mut out),
                Err(EvalError::BufferSize {
                    expected: 25,
                    got: len
                }),
                "normals_into, {len} values"
            );
        }
    }

    #[test]
    fn normals_into_matches_normals() {
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let map = random_corrections(&mesh, 3, 200, 13);
        let dense = DenseEvaluation::new(&mesh, level(3), &map).unwrap();
        for face in mesh.faces() {
            for l in 0..=3 {
                let region = Region::whole_face(face, level(l));
                let grid = normals(&mesh, &dense, region).unwrap();
                let side = usize::try_from(region.quads() + 1).unwrap();
                let mut out = vec![Vec3::NAN; side * side];
                normals_into(&mesh, &dense, region, &mut out).unwrap();
                let bits = |v: &[Vec3]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&out), bits(grid.as_slice()), "{region:?}");
            }
        }
    }

    /// `dense`, with every vertex read from a `stale` tile moved, except the vertices of
    /// `region` itself: what positions look like after an edit of the region's neighbours.
    struct Moved<'a> {
        /// The positions before the edit.
        dense: &'a DenseEvaluation,
        /// The tiles whose vertices moved.
        stale: &'a BTreeSet<TileKey>,
        /// The region's own vertices, as owner copies: they never move.
        own: BTreeSet<(FaceId, u32, u32)>,
    }

    impl<'a> Moved<'a> {
        /// The positions of `dense` with the vertices of `stale` tiles moved, `region`'s kept.
        fn new(
            mesh: &BaseMesh,
            dense: &'a DenseEvaluation,
            stale: &'a BTreeSet<TileKey>,
            region: Region,
        ) -> Self {
            let n = region.level().face_quads();
            let own = (0..=region.quads())
                .flat_map(|j| (0..=region.quads()).map(move |i| (i, j)))
                .map(|(i, j)| {
                    let (x, y) = (region.x0() + i, region.y0() + j);
                    mesh.topology().owner_copy(region.face(), n, x, y).unwrap()
                })
                .collect();
            Self { dense, stale, own }
        }
    }

    impl PositionSource for Moved<'_> {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let p = self.dense.position(face, level, i, j)?;
            let (key, _, _) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
            if !self.stale.contains(&key) || self.own.contains(&(face, i, j)) {
                return Some(p);
            }
            // A different nudge per vertex, so no two reads cancel.
            let h = (face.index() * 7919 + i * 131 + j * 31) % 97;
            let step = 1e-3 * f32::from(u16::try_from(h + 1).unwrap()) / 97.0;
            Some(p + Vec3::new(step, -0.5 * step, 0.25 * step))
        }
    }

    /// Bit patterns of `values`, for comparing normals exactly.
    fn bits(values: &[Vec3]) -> Vec<[u32; 3]> {
        values.iter().map(|p| p.to_bits()).collect()
    }

    /// Asserts that refreshed normals equal a full recompute, for each region.
    ///
    /// Normals computed before the region's neighbours moved, refreshed after, equal a full
    /// recompute after, bit for bit, whichever neighbours moved. Returns the vertices refreshed
    /// and the vertices in all.
    fn assert_refresh_equals_recompute(
        mesh: &BaseMesh,
        dense: &DenseEvaluation,
        regions: &[Region],
    ) -> (usize, usize) {
        let (mut refreshed, mut total) = (0, 0);
        for &region in regions {
            let side = usize::try_from(region.quads() + 1).unwrap();
            let own = TileKey::containing(
                LayerId::BASE,
                Channel::DISPLACEMENT,
                region.face(),
                region.level(),
                region.x0() + 1,
                region.y0() + 1,
            )
            .unwrap()
            .0;
            let mut neighbours = normal_tiles(mesh, region).unwrap();
            neighbours.remove(&own);
            // Every neighbour at once, then each on its own.
            let sets = std::iter::once(neighbours.clone())
                .chain(neighbours.iter().map(|&key| BTreeSet::from([key])));
            for stale in sets {
                let mut out = vec![Vec3::NAN; side * side];
                normals_into(mesh, dense, region, &mut out).unwrap();
                let moved = Moved::new(mesh, dense, &stale, region);
                let mut want = vec![Vec3::NAN; side * side];
                normals_into(mesh, &moved, region, &mut want).unwrap();
                refreshed +=
                    refresh_normals_into(mesh, &moved, region, |k| stale.contains(k), &mut out)
                        .unwrap();
                total += side * side;
                assert_eq!(bits(&out), bits(&want), "{region:?}, {stale:?} moved");
            }
        }
        (refreshed, total)
    }

    #[test]
    fn refreshed_normals_equal_a_full_recompute() {
        // Level-8 tiles at face corners (valence 3, 4 and 5), on face edges (their corners on
        // the edge between two faces), and inside, and whole faces at levels 1 and 3.
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let map = random_corrections(&mesh, 8, 800, 14);
        let dense = DenseEvaluation::new(&mesh, level(8), &map).unwrap();
        let mut regions = tiles_where(&mesh, level(8), |tx, ty| {
            [0, 1, 3].contains(&tx) && [0, 2, 3].contains(&ty)
        });
        regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(1))));
        regions.extend(mesh.faces().map(|f| Region::whole_face(f, level(3))));
        let (refreshed, total) = assert_refresh_equals_recompute(&mesh, &dense, &regions);
        assert!(
            refreshed > 0 && refreshed * 10 < total,
            "only border vertices are recomputed: {refreshed} of {total}"
        );
    }

    #[test]
    fn refreshing_with_nothing_stale_changes_nothing() {
        let mesh = pentagonal_prism().subdivide().to_base_mesh();
        let dense = DenseEvaluation::new(&mesh, level(2), &CorrectionMap::new()).unwrap();
        let region = Region::whole_face(FaceId::new(5), level(2));
        let mut out = vec![Vec3::new(9.0, 9.0, 9.0); 25];
        assert_eq!(
            refresh_normals_into(&mesh, &dense, region, |_| false, &mut out),
            Ok(0),
            "nothing to recompute"
        );
        assert!(
            out.iter().all(|&p| p == Vec3::new(9.0, 9.0, 9.0)),
            "untouched"
        );
        assert_eq!(
            refresh_normals_into(&mesh, &dense, region, |_| true, &mut out[..24]),
            Err(EvalError::BufferSize {
                expected: 25,
                got: 24
            }),
            "a buffer of the wrong size"
        );
    }

    #[test]
    fn normals_need_the_apron() {
        let mesh = cube_sphere(2).unwrap();
        let dense = DenseEvaluation::new(&mesh, level(2), &CorrectionMap::new()).unwrap();
        let face = FaceId::new(5);
        let keys = BTreeSet::from([TileKey::displacement(face, level(2), 0, 0).unwrap()]);
        let own_face_only = Only {
            dense: &dense,
            keys: &keys,
        };
        assert!(
            matches!(
                normals(&mesh, &own_face_only, Region::whole_face(face, level(2))),
                Err(EvalError::Missing { .. })
            ),
            "the apron lies in neighbouring faces"
        );
    }

    /// A source that answers every read with zero and records the tile each read resolves to:
    /// the oracle [`gathered_tiles`] is checked against.
    #[derive(Debug, Default)]
    struct Recorder(std::cell::RefCell<BTreeSet<TileKey>>);

    impl PositionSource for Recorder {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let (key, _, _) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
            self.0.borrow_mut().insert(key);
            Some(Vec3::ZERO)
        }
    }

    /// The tiles [`gather`] reads for `region`, recorded read by read.
    fn recorded_tiles(mesh: &BaseMesh, region: Region) -> BTreeSet<TileKey> {
        let recorder = Recorder::default();
        gather(mesh, &recorder, region).unwrap();
        recorder.0.into_inner()
    }

    #[test]
    fn gathered_tiles_are_the_tiles_gather_reads() {
        use crate::reference::prism;
        let meshes = [
            cube_sphere(1).unwrap(),
            cube_sphere(2).unwrap(),
            pentagonal_prism().subdivide().to_base_mesh(),
            prism(8).subdivide().to_base_mesh(),
            prism(MAX_VALENCE).subdivide().to_base_mesh(),
        ];
        let mut rng = Rng::new(0x7115_5EAD);
        for mesh in &meshes {
            // Every tile of every level up to 8 (whole faces up to level 6, then 2 × 2 and
            // 4 × 4 tiles a face), and random blocks at levels 1 to 8.
            let mut regions: Vec<Region> = (0..=8)
                .flat_map(|l| tiles_where(mesh, level(l), |_, _| true))
                .collect();
            for _ in 0..400 {
                let l = level(u8::try_from(1 + rng.below(8)).unwrap());
                let face = FaceId::new(rng.below(mesh.face_count()));
                let half = l.face_quads() / 2;
                let quads = 2 * (1 + rng.below(half));
                let x0 = 2 * rng.below(half - quads / 2 + 1);
                let y0 = 2 * rng.below(half - quads / 2 + 1);
                regions.push(Region::new(face, l, x0, y0, quads).unwrap());
            }
            for region in regions {
                assert_eq!(
                    gathered_tiles(mesh, region).unwrap(),
                    recorded_tiles(mesh, region),
                    "{} faces, {region:?}",
                    mesh.face_count()
                );
            }
        }
    }

    /// The hash of every kernel's output over wide blocks of `prism(n)` subdivided once.
    ///
    /// Under fixed-seed random corrections, it hashes predicted positions, positions with
    /// corrections applied (so every frame component), normals, and refreshed border normals.
    ///
    /// The blocks are whole faces and quarter faces at levels 6 and 7, so rows run 31 to 127
    /// vertices inside the face, with base vertices of valence 3, 4 and `n` at their corners.
    fn kernel_hash(n: usize) -> u64 {
        let mesh = crate::reference::prism(n).subdivide().to_base_mesh();
        let (coarse, fine) = (level(6), level(7));
        let mut rng = Rng::new(0x4B45_0000 + u64::try_from(n).unwrap());
        let mut map = CorrectionMap::new();
        for _ in 0..400 {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let l = level(1 + u8::try_from(rng.below(6)).unwrap());
            let side = l.face_quads() + 1;
            let (i, j) = (rng.below(side), rng.below(side));
            map.set(mesh.topology(), face, l, i, j, rng.vec3(0.02))
                .unwrap();
        }
        let dense = DenseEvaluation::new(&mesh, coarse, &map).unwrap();
        let blocks = |l: Level| {
            let q = l.face_quads();
            mesh.faces().flat_map(move |f| {
                [(0, 0, q), (0, 0, q / 2), (q / 2, q / 4, q / 2)]
                    .map(|(x0, y0, quads)| Region::new(f, l, x0, y0, quads).unwrap())
            })
        };
        let mut points = Vec::new();
        for region in blocks(fine) {
            let prediction = predict_region(&mesh, &dense, region).unwrap();
            let side = prediction.side();
            for j in 0..side {
                for i in 0..side {
                    points.push(prediction.predicted(i, j).unwrap());
                }
            }
            let corrections: Vec<Vec3> = (0..side * side).map(|_| rng.vec3(0.01)).collect();
            let c = |i: u32, j: u32| corrections[usize::try_from(j * side + i).unwrap()];
            points.extend_from_slice(prediction.apply(c).as_slice());
        }
        for region in blocks(coarse) {
            let side = usize::try_from(region.quads() + 1).unwrap();
            let mut out = vec![Vec3::ZERO; side * side];
            normals_into(&mesh, &dense, region, &mut out).unwrap();
            points.extend_from_slice(&out);
            out.fill(Vec3::ZERO);
            refresh_normals_into(&mesh, &dense, region, |_| true, &mut out).unwrap();
            points.extend_from_slice(&out);
        }
        crate::test_rng::fnv1a(points)
    }

    /// [`kernel_hash`] for `prism(3)`, `prism(5)` and `prism(8)`, recorded from the kernels as they
    /// stood when these were pinned (before the row and lane rewrites, so those are held to the
    /// per-vertex code's bits).
    ///
    /// To regenerate after a deliberate change to the arithmetic, run
    /// `BRUSHCRAFT_PRINT_GOLDEN=1 cargo nextest run -p geom kernel_bits --no-capture`, check the
    /// change was meant to move the bits, and paste the printed values here.
    const KERNEL_GOLDEN: [(usize, u64); 3] = [
        (3, 0x7BE8_3E3F_9151_C7E5),
        (5, 0x279C_99FF_705C_FE78),
        (8, 0x9955_E72D_6E3F_B2C9),
    ];

    /// Pins the kernels' bits on blocks wide enough that whole rows run through them: the
    /// golden-bits test in `dense` works on level-3 faces, whose rows are too short to.
    #[test]
    fn kernel_bits_are_pinned_on_wide_blocks() {
        let hashes = KERNEL_GOLDEN.map(|(n, _)| (n, kernel_hash(n)));
        if std::env::var_os("BRUSHCRAFT_PRINT_GOLDEN").is_some() {
            for (n, hash) in hashes {
                let groups =
                    [48, 32, 16, 0].map(|shift| format!("{:04X}", (hash >> shift) & 0xFFFF));
                println!("({n}, 0x{}),", groups.join("_"));
            }
            return;
        }
        for ((n, hash), (_, want)) in hashes.into_iter().zip(KERNEL_GOLDEN) {
            assert_eq!(hash, want, "prism({n}): 0x{hash:016X} != 0x{want:016X}");
        }
    }
}
