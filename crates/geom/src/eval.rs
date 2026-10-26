//! Evaluating a block: gather the parent level, predict, build frames, apply corrections.

use thiserror::Error;

use crate::frame::{Block, CornerAxes, Frame, frames};
use crate::subdiv::{CornerRing, Padded, extraordinary, predict, ring_edge_point};
use crate::{BaseMesh, FaceId, Level, MAX_VALENCE, Region, TILE_QUADS, Vec3};

mod cone;
mod gather;
mod normals;

pub use cone::{cone_width, normal_tiles, parent_tiles};
pub(crate) use gather::gather;
pub use normals::{normals, normals_into, refresh_normals_into};

/// Read access to evaluated positions.
///
/// An implementation must resolve each `(face, level, i, j)` to exactly one stored copy. A vertex
/// on a tile border belongs to several tiles of the same face; answer from
/// [`TileKey::containing`](crate::TileKey::containing)'s tile. (Across faces, `geom` already reads
/// only the owner face's copy.)
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

/// A source without positions: what [`gathered_tiles`](cone::gathered_tiles) hands its
/// [`Reader`](gather::Reader), whose tile lookups read none.
struct NoPositions;

impl PositionSource for NoPositions {
    fn position(&self, _: FaceId, _: Level, _: u32, _: u32) -> Option<Vec3> {
        None
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
