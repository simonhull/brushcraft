//! Dense evaluation of whole meshes: for tests and small meshes. `store` evaluates tile by tile
//! instead.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::topology::Topology;
use crate::{
    BaseMesh, Channel, CorrectionError, CorrectionTile, EvalError, FaceId, Grid, LayerId, Level,
    PositionSource, Region, TileKey, Vec3, evaluate_region,
};

/// An in-memory, sparse set of correction tiles. Absent tiles are all zero.
#[derive(Debug, Clone, Default)]
pub struct CorrectionMap {
    tiles: HashMap<TileKey, CorrectionTile>,
}

impl CorrectionMap {
    /// An empty map: every correction zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A stored tile.
    #[must_use]
    pub fn tile(&self, key: &TileKey) -> Option<&CorrectionTile> {
        self.tiles.get(key)
    }

    /// Stores a tile, returning the one it replaced.
    pub fn insert(&mut self, key: TileKey, tile: CorrectionTile) -> Option<CorrectionTile> {
        self.tiles.insert(key, tile)
    }

    /// Sets the correction at face-grid vertex `(i, j)`, in every copy (see
    /// [`Topology::copies`]).
    ///
    /// # Errors
    ///
    /// [`CorrectionError::NotFinite`] if `c` isn't finite (nothing is written then);
    /// [`CorrectionError::OutOfTile`] if the vertex is outside the face or `face` is not in
    /// `topology`.
    pub fn set(
        &mut self,
        topology: &Topology,
        face: FaceId,
        level: Level,
        i: u32,
        j: u32,
        c: Vec3,
    ) -> Result<(), CorrectionError> {
        if !c.is_finite() {
            return Err(CorrectionError::NotFinite);
        }
        let copies = topology.copies(face, level, i, j);
        if copies.is_empty() {
            return Err(CorrectionError::OutOfTile {
                i,
                j,
                side: level.face_quads() + 1,
            });
        }
        for copy in copies {
            let key = copy.key(level);
            let tile = match self.tiles.entry(key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(CorrectionTile::zero(key.side())?),
            };
            tile.set(copy.i, copy.j, c)?;
        }
        Ok(())
    }

    /// The correction at face-grid vertex `(i, j)`: zero where no tile is stored.
    #[must_use]
    pub fn get(&self, face: FaceId, level: Level, i: u32, j: u32) -> Vec3 {
        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level, i, j)
            .and_then(|(key, li, lj)| self.tiles.get(&key)?.get(li, lj))
            .unwrap_or(Vec3::ZERO)
    }
}

/// Every level of a mesh, evaluated in full.
#[derive(Debug, Clone)]
pub struct DenseEvaluation {
    /// `levels[l][face]`.
    levels: Vec<Vec<Grid>>,
}

impl PositionSource for DenseEvaluation {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        self.grid(face, level)?.get(i, j)
    }
}

impl DenseEvaluation {
    /// Evaluates levels `0..=finest` of `mesh` with `corrections`.
    ///
    /// # Errors
    ///
    /// [`EvalError`] if evaluation fails (it can't for a valid mesh; kept for the shared
    /// signature).
    pub fn new(
        mesh: &BaseMesh,
        finest: Level,
        corrections: &CorrectionMap,
    ) -> Result<Self, EvalError> {
        // Level 0's grid is row-major (0,0), (1,0), (0,1), (1,1): corners 0, 1, 3, 2.
        let base = mesh
            .faces()
            .map(|f| {
                let p = |c| mesh.corner_position(f, c).unwrap_or(Vec3::NAN);
                Grid::from_parts(2, vec![p(0), p(1), p(3), p(2)])
            })
            .collect();
        let mut eval = Self { levels: vec![base] };
        for level in std::iter::successors(Level::ZERO.child(), |l| l.child()) {
            if level > finest {
                break;
            }
            let grids = mesh
                .faces()
                .map(|f| {
                    evaluate_region(mesh, &eval, Region::whole_face(f, level), |i, j| {
                        corrections.get(f, level, i, j)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            eval.levels.push(grids);
        }
        Ok(eval)
    }

    /// One face's grid at one level, or `None` if either isn't held.
    #[must_use]
    pub fn grid(&self, face: FaceId, level: Level) -> Option<&Grid> {
        self.levels
            .get(usize::from(level.get()))?
            .get(usize::try_from(face.index()).ok()?)
    }

    /// The finest level held.
    #[must_use]
    pub fn finest(&self) -> Level {
        u8::try_from(self.levels.len().saturating_sub(1))
            .ok()
            .and_then(|l| Level::new(l).ok())
            .unwrap_or(Level::ZERO)
    }

    /// A copy holding only levels `0..=finest`.
    #[must_use]
    pub fn truncated(&self, finest: Level) -> Self {
        Self {
            levels: self
                .levels
                .iter()
                .take(usize::from(finest.get()) + 1)
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::reference::{Poly, assert_same_points, pentagonal_prism};
    use crate::test_rng::Rng;
    use crate::{Region, TileKey, cube_sphere, evaluate_region};

    fn all_points(eval: &DenseEvaluation, mesh: &BaseMesh, level: Level) -> Vec<Vec3> {
        mesh.faces()
            .flat_map(|f| eval.grid(f, level).unwrap().as_slice().to_vec())
            .collect()
    }

    fn check_against_reference(mesh: &BaseMesh, levels: u8) {
        let eval =
            DenseEvaluation::new(mesh, Level::new(levels).unwrap(), &CorrectionMap::new()).unwrap();
        let mut reference = Poly::from_base_mesh(mesh);
        for l in 1..=levels {
            reference = reference.subdivide();
            assert_same_points(
                &all_points(&eval, mesh, Level::new(l).unwrap()),
                &reference.positions,
                2e-5,
            );
        }
    }

    #[test]
    fn zero_pyramid_matches_reference_with_valence_three() {
        check_against_reference(&cube_sphere(2).unwrap(), 3);
    }

    #[test]
    fn zero_pyramid_matches_reference_with_valence_five() {
        check_against_reference(&pentagonal_prism().subdivide().to_base_mesh(), 3);
    }

    #[test]
    fn absent_tiles_equal_explicit_zero_tiles() {
        let mesh = cube_sphere(2).unwrap();
        let top = Level::new(3).unwrap();
        let mut zeros = CorrectionMap::new();
        for face in mesh.faces() {
            for l in 1..=3 {
                let level = Level::new(l).unwrap();
                let key = TileKey::displacement(face, level, 0, 0).unwrap();
                zeros.insert(key, CorrectionTile::zero(key.side()).unwrap());
            }
        }
        let a = DenseEvaluation::new(&mesh, top, &CorrectionMap::new()).unwrap();
        let b = DenseEvaluation::new(&mesh, top, &zeros).unwrap();
        let bits = |e: &DenseEvaluation| {
            all_points(e, &mesh, top)
                .iter()
                .map(|p| p.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a), bits(&b), "sparse = explicit zero");
    }

    #[test]
    fn twins_are_bit_identical_after_random_edits() {
        let mesh = cube_sphere(2).unwrap();
        let top = Level::new(3).unwrap();
        let mut rng = Rng::new(11);
        let mut map = CorrectionMap::new();
        for _ in 0..400 {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let level = Level::new(1 + u8::try_from(rng.below(3)).unwrap()).unwrap();
            let n = level.face_quads() + 1;
            // Bias towards borders, where seams can break.
            let pick = |random: &mut Rng| {
                if random.below(2) == 0 {
                    [0, n - 1][random.below(2) as usize]
                } else {
                    random.below(n)
                }
            };
            let (i, j) = (pick(&mut rng), pick(&mut rng));
            map.set(mesh.topology(), face, level, i, j, rng.vec3(0.02))
                .unwrap();
        }
        let eval = DenseEvaluation::new(&mesh, top, &map).unwrap();
        for l in 1..=3 {
            let level = Level::new(l).unwrap();
            let n = level.face_quads();
            for face in mesh.faces() {
                for j in 0..=n {
                    for i in 0..=n {
                        let mine = eval.position(face, level, i, j).unwrap();
                        for c in mesh.topology().copies(face, level, i, j) {
                            let (key, ci, cj) = (c.key(level), c.i, c.j);
                            let r = Region::of_tile(&key);
                            let theirs = eval
                                .position(c.face, level, r.x0() + ci, r.y0() + cj)
                                .unwrap();
                            assert_eq!(
                                mine.to_bits(),
                                theirs.to_bits(),
                                "{face:?} L{l} ({i}, {j}) vs {c:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tile_blocks_equal_whole_face_evaluation() {
        // Level 7 splits faces into 2×2 tiles: evaluating each tile alone, reading only its own
        // tile's corrections, must reproduce the whole-face result bit for bit. This is what lets
        // `store` evaluate tile by tile.
        let mesh = cube_sphere(1).unwrap();
        let level = Level::new(7).unwrap();
        let mut rng = Rng::new(5);
        let mut map = CorrectionMap::new();
        for _ in 0..300 {
            let face = FaceId::new(rng.below(6));
            let near = 64 + rng.below(3) - 1; // around the tile border
            let (i, j) = if rng.below(2) == 0 {
                (near, rng.below(129))
            } else {
                (rng.below(129), near)
            };
            map.set(mesh.topology(), face, level, i, j, rng.vec3(0.01))
                .unwrap();
        }
        let eval = DenseEvaluation::new(&mesh, level, &map).unwrap();
        let parent_only = eval.truncated(Level::new(6).unwrap());
        for face in mesh.faces() {
            for (tx, ty) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let key = TileKey::displacement(face, level, tx, ty).unwrap();
                let tile = map.tile(&key);
                let r = Region::of_tile(&key);
                let grid = evaluate_region(&mesh, &parent_only, r, |i, j| {
                    tile.and_then(|t| t.get(i, j)).unwrap_or(Vec3::ZERO)
                })
                .unwrap();
                for j in 0..=64 {
                    for i in 0..=64 {
                        let whole = eval.position(face, level, r.x0() + i, r.y0() + j).unwrap();
                        assert_eq!(
                            grid.get(i, j).unwrap().to_bits(),
                            whole.to_bits(),
                            "{key:?} ({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    /// Frames, checked black-box through what corrections do.
    ///
    /// A unit `t` correction must move every vertex along its owner's grid `+i`, and a unit `n`
    /// correction outward. Each face evaluates its shared vertices in the owner's orientation,
    /// so this covers owner choice, the corner table, the ring's edge points and the `t`/`n`
    /// construction.
    #[test]
    fn corrections_follow_the_owner_orientation() {
        let mesh = cube_sphere(2).unwrap();
        let t = mesh.topology();
        let dense =
            DenseEvaluation::new(&mesh, Level::new(2).unwrap(), &CorrectionMap::new()).unwrap();
        for l in 1..=3 {
            let level = Level::new(l).unwrap();
            let n = level.face_quads();
            let parent = dense.truncated(level.parent().unwrap());
            let eval = |c: Vec3| -> Vec<Grid> {
                mesh.faces()
                    .map(|f| {
                        evaluate_region(&mesh, &parent, Region::whole_face(f, level), |_, _| c)
                            .unwrap()
                    })
                    .collect()
            };
            let (zero, along, normal) = (
                eval(Vec3::ZERO),
                eval(Vec3::new(1.0, 0.0, 0.0)),
                eval(Vec3::new(0.0, 0.0, 1.0)),
            );
            let at = |grids: &[Grid], f: FaceId, i: u32, j: u32| {
                grids[f.index() as usize].get(i, j).unwrap()
            };
            for face in mesh.faces() {
                for j in 0..=n {
                    for i in 0..=n {
                        let p = at(&zero, face, i, j);
                        let tangent = at(&along, face, i, j) - p;
                        let outward = at(&normal, face, i, j) - p;
                        assert!(
                            outward.dot(p) > 0.5 * p.length(),
                            "{face:?} L{l} ({i}, {j}): n points outward"
                        );
                        let owner = t
                            .copies(face, level, i, j)
                            .into_iter()
                            .min_by_key(|c| c.face)
                            .unwrap();
                        let r = Region::of_tile(&owner.key(level));
                        let (oi, oj) = (r.x0() + owner.i, r.y0() + owner.j);
                        let step = if oi < n {
                            at(&zero, owner.face, oi + 1, oj) - at(&zero, owner.face, oi, oj)
                        } else {
                            at(&zero, owner.face, oi, oj) - at(&zero, owner.face, oi - 1, oj)
                        };
                        let cos = tangent.dot(step) / (tangent.length() * step.length());
                        assert!(
                            cos > 0.7,
                            "{face:?} L{l} ({i}, {j}): t is the owner's +i ({owner:?}, cos {cos})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_normal_correction_moves_outward() {
        let mesh = cube_sphere(2).unwrap();
        let level = Level::new(2).unwrap();
        let face = FaceId::new(0);
        let mut map = CorrectionMap::new();
        map.set(mesh.topology(), face, level, 2, 2, Vec3::new(0.0, 0.0, 0.1))
            .unwrap();
        let before = DenseEvaluation::new(&mesh, level, &CorrectionMap::new()).unwrap();
        let after = DenseEvaluation::new(&mesh, level, &map).unwrap();
        let (p0, p1) = (
            before.position(face, level, 2, 2).unwrap(),
            after.position(face, level, 2, 2).unwrap(),
        );
        assert!(
            ((p1 - p0).length() - 0.1).abs() < 1e-5,
            "moved by the correction"
        );
        assert!((p1 - p0).dot(p0) > 0.0, "outward on a sphere");
    }
}
