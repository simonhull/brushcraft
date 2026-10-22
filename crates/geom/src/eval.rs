//! Evaluating a block: gather the parent level, predict, build frames, apply corrections.

use std::cell::RefCell;
use std::collections::BTreeSet;

use thiserror::Error;

use crate::frame::{self, Block, CornerAxes, Frame, frames};
use crate::subdiv::{CornerRing, Padded, extraordinary, predict, ring_edge_point};
use crate::topology::{corner_at, edge_to_grid};
use crate::{BaseMesh, Channel, FaceId, LayerId, Level, Region, TileKey, Vec3};

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
}

/// The parent level's block, ring and base-vertex rings.
struct Parent {
    padded: Padded,
    corners: [Option<CornerRing>; 4],
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

/// Gathers `region` (at its own level) with its one-vertex ring and base-vertex rings.
///
/// Every position is read through one closure, as the owner's copy: [`parent_tiles`] runs this
/// with a recording source, so what it lists is exactly what is read here.
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
    // Every read goes to the owner's copy, so all faces predict a shared vertex from the same
    // inputs even if the source's twins disagree.
    let fetch = |f: FaceId, i: i64, j: i64| {
        let (Ok(ui), Ok(uj)) = (u32::try_from(i), u32::try_from(j)) else {
            return Err(missing(f, level, i, j));
        };
        let (g, gi, gj) = topology
            .owner_copy(f, n, ui, uj)
            .ok_or_else(|| missing(f, level, i, j))?;
        src.position(g, level, gi, gj)
            .ok_or_else(|| missing(g, level, i64::from(gi), i64::from(gj)))
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
        let Some(&(_, owner_corner)) = ring.entries.first() else {
            continue;
        };
        // Already read from the owner's copy, like every other input.
        let center = padded.get(ci, cj);
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

/// A block's predicted positions and tangent frames, before corrections: what a brush needs to
/// project its deltas into frames and to re-apply corrections without predicting again.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    region: Region,
    predicted: Padded,
    /// Row-major, `(quads + 1)²`.
    frames: Vec<Frame>,
}

/// The block one level coarser that predicting `region` gathers.
///
/// # Errors
///
/// [`EvalError::LevelZero`] for a level-0 region, or [`EvalError::NoExactParent`] for a region
/// that doesn't halve exactly.
fn parent_region(region: Region) -> Result<Region, EvalError> {
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
    // Frames at base vertices use the predicted level's ring: the vertex point and every edge
    // point around it, computed from the ring alone so all copies agree.
    let edge_points = parent.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| {
            (0..ring.edges.len())
                .map(|k| ring_edge_point(ring, k))
                .collect::<Vec<_>>()
        })
    });
    let axes = std::array::from_fn(|c| {
        let ring = parent.corners.get(c)?.as_ref()?;
        Some(CornerAxes {
            center: extraordinary(ring),
            points: edge_points.get(c)?.as_deref()?,
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
    Ok(Prediction {
        region,
        predicted,
        frames,
    })
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
        let side = self.side();
        let data = (0..side)
            .flat_map(|j| (0..side).map(move |i| (i, j)))
            .zip(&self.frames)
            .map(|((i, j), frame)| {
                frame.apply(
                    self.predicted.get(i64::from(i), i64::from(j)),
                    corrections(i, j),
                )
            })
            .collect();
        Grid::from_parts(side, data)
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

/// A source that answers every read with zero and records the tile each read resolves to.
#[derive(Debug, Default)]
struct Recorder(RefCell<BTreeSet<TileKey>>);

impl PositionSource for Recorder {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        let (key, _, _) =
            TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)?;
        self.0.borrow_mut().insert(key);
        Some(Vec3::ZERO)
    }
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
    let recorder = Recorder::default();
    // The same gather `predict_region` runs, so the list can't drift from the reads.
    gather(mesh, &recorder, parent_region(region)?)?;
    Ok(recorder.0.into_inner())
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
    let gathered = gather(mesh, src, region)?;
    // The level's own ring: the base vertex (the owner's copy) and its edge neighbours.
    let axes = gathered.corners.each_ref().map(|c| {
        c.as_ref().map(|ring| CornerAxes {
            center: ring.center,
            points: &ring.edges,
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
    Ok(Grid::from_parts(
        region.quads() + 1,
        frame::normals(&block, &gathered.padded, &axes),
    ))
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
}
