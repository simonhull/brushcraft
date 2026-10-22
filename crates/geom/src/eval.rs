//! Evaluating a block: gather the parent level, predict, build frames, apply corrections.

use thiserror::Error;

use crate::frame::{Block, CornerAxes, frames};
use crate::subdiv::{CornerRing, Padded, extraordinary, predict, ring_edge_point};
use crate::topology::{corner_at, corner_coords, edge_to_grid};
use crate::{BaseMesh, FaceId, Level, Region, Vec3};

/// Read access to evaluated positions.
pub trait PositionSource {
    /// The evaluated position of face-grid vertex `(i, j)` of `face` at `level`, or `None` if it
    /// isn't available.
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3>;
}

/// A block of evaluated positions: `side²` vertices, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    side: u32,
    data: Vec<Vec3>,
}

impl Grid {
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
    /// The parent level's position wasn't available from the source.
    #[error("position of face {face:?} level {level} vertex ({i}, {j}) is not available")]
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
}

/// The parent level's block, ring and base-vertex rings.
struct Parent {
    padded: Padded,
    corners: [Option<CornerRing>; 4],
}

/// Gathers `region` (at the parent level) with its one-vertex ring and base-vertex rings.
///
/// # Errors
///
/// [`EvalError::Missing`] if `src` lacks a needed position.
fn gather(mesh: &BaseMesh, src: &impl PositionSource, region: Region) -> Result<Parent, EvalError> {
    let topology = mesh.topology();
    let level = region.level();
    let n = level.face_quads();
    let n64 = i64::from(n);
    let q = i64::from(region.quads());
    let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
    let face = region.face();
    let fetch = |f: FaceId, i: i64, j: i64| {
        let missing = || EvalError::Missing {
            face: f,
            level,
            i: u32::try_from(i).unwrap_or(u32::MAX),
            j: u32::try_from(j).unwrap_or(u32::MAX),
        };
        let (Ok(i), Ok(j)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing());
        };
        src.position(f, level, i, j).ok_or_else(missing)
    };
    let mut padded = Padded::filled(region.quads(), Vec3::NAN);
    for j in -1..=q + 1 {
        for i in -1..=q + 1 {
            if let Some((g, gi, gj)) = topology.locate(face, n, x0 + i, y0 + j) {
                padded.set(i, j, fetch(g, i64::from(gi), i64::from(gj))?);
            }
        }
    }
    let mut corners: [Option<CornerRing>; 4] = [None, None, None, None];
    let on = |x: i64| x == 0 || x == n64;
    for (slot, (ci, cj)) in corners.iter_mut().zip([(0, 0), (q, 0), (q, q), (0, q)]) {
        let (fi, fj) = (x0 + ci, y0 + cj);
        if !(on(fi) && on(fj)) {
            continue;
        }
        let (Ok(ui), Ok(uj)) = (u32::try_from(fi), u32::try_from(fj)) else {
            continue;
        };
        let Some(ring) = topology.ring(face, corner_at(ui, uj, n)) else {
            continue;
        };
        let Some(&(owner, owner_corner)) = ring.entries.first() else {
            continue;
        };
        // Fetched from the owner, the same way in every face, so it never relies on the source
        // storing identical copies.
        let (cx, cy) = corner_coords(owner_corner, n64);
        let center = fetch(owner, cx, cy)?;
        let mut edges = Vec::with_capacity(ring.entries.len());
        let mut diagonals = Vec::with_capacity(ring.entries.len());
        for &(g, c) in &ring.entries {
            let point = |s: i64, d: i64| {
                let (gi, gj) = edge_to_grid(c, n64, s, d);
                fetch(g, gi, gj)
            };
            edges.push(point(1, 0)?);
            diagonals.push(point(1, 1)?);
        }
        *slot = Some(CornerRing {
            center,
            edges,
            diagonals,
            owner_corner,
        });
    }
    Ok(Parent { padded, corners })
}

/// Evaluates `region` (level ≥ 1): predicts it from the parent level's positions in `src`, then
/// applies `corrections(i, j)` (block-local coordinates, `0..=quads`) in each vertex's tangent
/// frame.
///
/// # Errors
///
/// [`EvalError::LevelZero`] for a level-0 region, or [`EvalError::Missing`] if `src` lacks a
/// parent position.
pub fn evaluate_region(
    mesh: &BaseMesh,
    src: &impl PositionSource,
    region: Region,
    corrections: impl Fn(u32, u32) -> Vec3,
) -> Result<Grid, EvalError> {
    let parent_region = region.parent().ok_or(EvalError::LevelZero)?;
    let parent = gather(mesh, src, parent_region)?;
    let predicted = predict(&parent.padded, &parent.corners);
    let axes = parent.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| CornerAxes {
            center: extraordinary(ring),
            e0: ring_edge_point(ring, 0),
            e1: ring_edge_point(ring, 1),
            owner_corner: ring.owner_corner,
        })
    });
    let block = Block {
        topology: mesh.topology(),
        face: region.face(),
        face_quads: region.level().face_quads(),
        x0: region.x0(),
        y0: region.y0(),
        quads: region.quads(),
    };
    let frames = frames(&block, &predicted, &axes);
    let side = region.quads() + 1;
    let data = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .zip(&frames)
        .map(|((i, j), frame)| {
            frame.apply(predicted.get(i64::from(i), i64::from(j)), corrections(i, j))
        })
        .collect();
    Ok(Grid::from_parts(side, data))
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::{FaceId, Level, Region, cube_sphere};

    /// Level 0 positions straight from the mesh.
    struct Base<'a>(&'a BaseMesh);
    impl PositionSource for Base<'_> {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            if level != Level::ZERO {
                return None;
            }
            let corner = corner_at(i, j, 1);
            self.0.corner_position(face, corner)
        }
    }

    #[test]
    fn level_one_of_a_cube_has_no_nan_and_matches_on_shared_edges() {
        let mesh = cube_sphere(1).unwrap();
        let level = Level::new(1).unwrap();
        let grids: Vec<Grid> = mesh
            .faces()
            .map(|f| {
                evaluate_region(&mesh, &Base(&mesh), Region::whole_face(f, level), |_, _| {
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
    fn level_zero_has_no_parent() {
        let mesh = cube_sphere(1).unwrap();
        let r = Region::whole_face(FaceId::new(0), Level::ZERO);
        assert_eq!(
            evaluate_region(&mesh, &Base(&mesh), r, |_, _| Vec3::ZERO),
            Err(EvalError::LevelZero),
            "level 0"
        );
    }
}
