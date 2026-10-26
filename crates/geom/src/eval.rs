//! Evaluating a block: gather the parent level, predict, build frames, apply corrections.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::frame::{self, Block, CornerAxes, Frame, FrameRows, Scattered, Stencil, frames};
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

    /// The positions of `out.len()` consecutive vertices of row `j` of `face` at `level`, from
    /// column `i`: `out[k]` is exactly what [`PositionSource::position`] gives for `(i + k, j)`.
    /// `None` if any isn't available, leaving `out` partly written; `Some` for an empty run.
    ///
    /// Evaluation reads the inside of a face a row at a time through this. The default reads
    /// the vertices one by one; a source that keeps rows contiguous can copy them instead.
    fn row(&self, face: FaceId, level: Level, i: u32, j: u32, out: &mut [Vec3]) -> Option<()> {
        for (slot, x) in out.iter_mut().zip(i..) {
            *slot = self.position(face, level, x, j)?;
        }
        Some(())
    }
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

    /// The `len` positions of row `j` from column `i`, or `None` if a non-empty run leaves the
    /// block. An empty run is `Some`, wherever it starts, as [`PositionSource::row`]'s default
    /// gives.
    #[cfg(any(test, feature = "test-fixtures"))]
    pub(crate) fn row(&self, i: u32, j: u32, len: usize) -> Option<&[Vec3]> {
        if len == 0 {
            return Some(&[]);
        }
        if i >= self.side || j >= self.side {
            return None;
        }
        let start = usize::try_from(j * self.side + i).ok()?;
        let in_row = usize::try_from(self.side - i).ok()?;
        self.data
            .get(start..start + len.min(in_row))
            .filter(|r| r.len() == len)
    }

    /// All positions, row-major.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec3] {
        &self.data
    }
}

/// Why a block couldn't be evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
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

    /// The positions of a run of row `j` of `face`, from column `i`, into `out`: vertices the
    /// caller knows are their own owner copies (strictly inside a face of the mesh), read as one
    /// [`PositionSource::row`] without the owner lookup.
    ///
    /// # Errors
    ///
    /// [`EvalError::Missing`] for the run's first vertex the source lacks (or that is off the
    /// face), as reading them one by one would report.
    fn own_row(&self, face: FaceId, i: i64, j: i64, out: &mut [Vec3]) -> Result<(), EvalError> {
        let missing = |x: i64| missing(face, self.level, x, j);
        let (Ok(ui), Ok(uj)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing(i));
        };
        if self.src.row(face, self.level, ui, uj, out).is_some() {
            return Ok(());
        }
        // Find which vertex was missing, in order.
        for (x, slot) in (i..).zip(out) {
            *slot = u32::try_from(x)
                .ok()
                .and_then(|ux| self.src.position(face, self.level, ux, uj))
                .ok_or_else(|| missing(x))?;
        }
        Ok(())
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
/// Every position is read through one [`Reader`], as the owner's copy. The part of each row
/// strictly inside the face, its own owner copies, is read as one [`PositionSource::row`]; the
/// points on or past the face's border are located one by one. [`gathered_tiles`] lists the
/// tiles these reads resolve to, and the tests check it against a recording of them.
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
    // A point on or past the face's border is located and read from its owner's copy.
    let located = |slots: &mut [Vec3], first_x: i64, y: i64| -> Result<(), EvalError> {
        for (x, slot) in (first_x..).zip(slots) {
            if let Some((g, gi, gj)) = reader.topology.locate(face, reader.n, x, y) {
                *slot = reader.fetch(g, i64::from(gi), i64::from(gj))?;
            }
        }
        Ok(())
    };
    // Strictly inside a face of the mesh, a point is its own owner copy: those of a row are
    // read as one run.
    let n = i64::from(reader.n);
    let (first, last) = (x0 - 1, x0 + q + 1);
    let (inner_first, inner_last) = (first.max(1), last.min(n - 1));
    let on_mesh = reader.topology.contains(face);
    for j in -1..=q + 1 {
        let y = y0 + j;
        let row = padded.row_mut(j);
        let run = usize::try_from(inner_first - first)
            .ok()
            .zip(usize::try_from(inner_last - inner_first + 1).ok())
            .filter(|_| on_mesh && (1..n).contains(&y));
        let Some((start, len)) = run else {
            located(row, first, y)?;
            continue;
        };
        let (before, rest) = row.split_at_mut(start.min(row.len()));
        let (inside, after) = rest.split_at_mut(len.min(rest.len()));
        located(before, first, y)?;
        reader.own_row(face, inner_first, y, inside)?;
        located(after, inner_last + 1, y)?;
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
/// Positions come from `src` unchecked: a parent position that isn't finite gives non-finite
/// predictions near it and NaN frames, not an error (nor a panic, in any build).
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
    /// The most bytes a tile's prediction holds on the heap: a tile of [`TILE_QUADS`] quads a
    /// side (the largest), its predicted positions with their one-vertex ring and a tangent
    /// frame per vertex, about 206 KB. A coarse tile's, of fewer quads, holds less. What a cache
    /// of predictions is charged per entry ([`Prediction::heap_bytes`] is at most this for any
    /// tile's prediction).
    pub const TILE_BYTES: usize = {
        // The constants are small: the casts are exact.
        let side = TILE_QUADS as usize + 1;
        let padded = side + 2;
        padded * padded * size_of::<Vec3>() + side * side * size_of::<Frame>()
    };

    /// The bytes this prediction holds on the heap: its positions' and frames' allocations.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.predicted
            .heap_bytes()
            .saturating_add(self.frames.capacity().saturating_mul(size_of::<Frame>()))
    }

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
    // The stale vertices, computed together at the end as the full normals compute theirs.
    let mut stale_vertices = Scattered::with_capacity(4 * usize::try_from(side).unwrap_or(0));
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
                frame::Local::Central(values)
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
        if index < out.len() {
            stale_vertices.push(index, local);
        }
    }
    stale_vertices.normals_into(&mut FrameRows::default(), out);
    Ok(stale_vertices.len())
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
mod tests;
