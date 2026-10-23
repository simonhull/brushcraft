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
    /// The tiles holding a correction, by key: base-layer displacement tiles above level 0.
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

    /// Stores a tile as it is, returning the one it replaced.
    ///
    /// Unlike [`CorrectionMap::set`], this doesn't keep twins in sync: the tile's border vertices
    /// are also held by neighbouring tiles, and the caller must store matching values there
    /// (as loading a consistent set of tiles does), or the copies evaluate differently.
    pub fn insert(&mut self, key: TileKey, tile: CorrectionTile) -> Option<CorrectionTile> {
        self.tiles.insert(key, tile)
    }

    /// Sets the correction at face-grid vertex `(i, j)`, in every copy (see
    /// [`Topology::copies`]).
    ///
    /// # Errors
    ///
    /// [`CorrectionError::NotFinite`] if `c` isn't finite, [`CorrectionError::LevelZero`] at
    /// level 0 (the base mesh), [`CorrectionError::UnknownFace`] if `face` is not in `topology`,
    /// or [`CorrectionError::OutOfTile`] if the vertex is outside the face. Nothing is written
    /// then.
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
        if level == Level::ZERO {
            return Err(CorrectionError::LevelZero);
        }
        if !topology.contains(face) {
            return Err(CorrectionError::UnknownFace(face));
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
        let base = mesh
            .faces()
            .map(|f| {
                let data = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .map(|(i, j)| mesh.position(f, Level::ZERO, i, j).unwrap_or(Vec3::NAN));
                Grid::from_parts(2, data.to_vec())
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
    use crate::reference::{Poly, assert_same_points, pentagonal_prism, prism};
    use crate::test_rng::{Rng, fnv1a};
    use crate::{MAX_VALENCE, Region, TileKey, Topology, cube_sphere, evaluate_region, normals};

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
        assert_twins_are_bit_identical_after_random_edits(&cube_sphere(2).unwrap(), 11);
    }

    #[test]
    fn twins_are_bit_identical_after_random_edits_at_max_valence() {
        assert_twins_are_bit_identical_after_random_edits(
            &prism(MAX_VALENCE).subdivide().to_base_mesh(),
            12,
        );
    }

    /// Random edits at levels 1–3, biased towards borders, then every copy of every vertex
    /// evaluates to the same bits.
    fn assert_twins_are_bit_identical_after_random_edits(mesh: &BaseMesh, seed: u64) {
        let top = Level::new(3).unwrap();
        let mut rng = Rng::new(seed);
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
        let eval = DenseEvaluation::new(mesh, top, &map).unwrap();
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

    /// The hash of level 3 of `prism(n)` subdivided once, under fixed-seed random corrections at
    /// levels 1–3: every face's positions, then every face's normals.
    fn golden_hash(n: usize) -> u64 {
        let mesh = prism(n).subdivide().to_base_mesh();
        let top = Level::new(3).unwrap();
        let mut rng = Rng::new(0x601D_0000 + u64::try_from(n).unwrap());
        let mut map = CorrectionMap::new();
        for _ in 0..200 {
            let face = FaceId::new(rng.below(mesh.face_count()));
            let level = Level::new(1 + u8::try_from(rng.below(3)).unwrap()).unwrap();
            let side = level.face_quads() + 1;
            let (i, j) = (rng.below(side), rng.below(side));
            map.set(mesh.topology(), face, level, i, j, rng.vec3(0.02))
                .unwrap();
        }
        let eval = DenseEvaluation::new(&mesh, top, &map).unwrap();
        let positions = mesh
            .faces()
            .flat_map(|f| eval.grid(f, top).unwrap().as_slice().to_vec());
        let normals = mesh.faces().flat_map(|f| {
            normals(&mesh, &eval, Region::whole_face(f, top))
                .unwrap()
                .as_slice()
                .to_vec()
        });
        fnv1a(positions.chain(normals))
    }

    /// [`golden_hash`] for valences 3 to 32, recorded from the evaluation as it stood when these
    /// were pinned.
    ///
    /// To regenerate after a deliberate change to the arithmetic, run
    /// `BRUSHCRAFT_PRINT_GOLDEN=1 cargo nextest run -p geom golden --no-capture`, check the
    /// change was meant to move the bits, and paste the printed table here.
    const GOLDEN: [u64; MAX_VALENCE - 2] = [
        0x3B41_A5C9_D0F6_7E7D, // valence 3
        0x1B49_B23D_CE92_CE00, // valence 4
        0x4465_118A_B420_4844, // valence 5
        0x2156_D801_BA78_EDF8, // valence 6
        0x7160_A5A8_51D1_B46D, // valence 7
        0x09DD_0649_C7D6_F284, // valence 8
        0xE9F2_E185_744D_FEEC, // valence 9
        0xD7AF_1F99_1FA2_BE93, // valence 10
        0x543D_7A4F_9EE6_7695, // valence 11
        0x2642_703D_1110_98CC, // valence 12
        0x9938_7E0C_29F0_F906, // valence 13
        0x9E20_FF64_C392_5683, // valence 14
        0x6C19_4D25_B30F_36BD, // valence 15
        0xCD0B_3FF1_3BBF_8C08, // valence 16
        0x2732_B66C_30BC_4E7E, // valence 17
        0x0A72_6CE0_F1F1_AD06, // valence 18
        0x8F76_1DDA_91FA_56CE, // valence 19
        0x2466_3B08_DCBD_E713, // valence 20
        0x5F42_CA3B_35A9_4982, // valence 21
        0x1898_1BC2_73A2_E60C, // valence 22
        0xCB9A_EBF0_3F1B_25D4, // valence 23
        0xB881_2109_5221_2331, // valence 24
        0x4B40_B027_557A_ED9D, // valence 25
        0x58D4_39B1_FF4E_C664, // valence 26
        0x82CF_FAEE_77D3_2615, // valence 27
        0xC3D2_48C9_A471_7772, // valence 28
        0x0D7B_3977_3157_76D3, // valence 29
        0x6F40_149E_1D76_4339, // valence 30
        0x4843_2153_7435_F0DB, // valence 31
        0x50D3_9623_FA69_568E, // valence 32
    ];

    /// Pins every summation order by the exact bits it produces at every valence the mesh accepts.
    ///
    /// The extraordinary vertex rule, frames and ring normals all sum in a canonical order. Twin
    /// tests can't see an order change that every copy makes alike; this can.
    #[test]
    fn golden_bits_pin_the_summation_order() {
        let hashes: Vec<(usize, u64)> = (3..=MAX_VALENCE).map(|n| (n, golden_hash(n))).collect();
        if std::env::var_os("BRUSHCRAFT_PRINT_GOLDEN").is_some() {
            println!("const GOLDEN: [u64; MAX_VALENCE - 2] = [");
            for (n, hash) in &hashes {
                let groups =
                    [48, 32, 16, 0].map(|shift| format!("{:04X}", (hash >> shift) & 0xFFFF));
                println!("    0x{}, // valence {n}", groups.join("_"));
            }
            println!("];");
            return;
        }
        for ((n, hash), want) in hashes.into_iter().zip(GOLDEN) {
            assert_eq!(hash, want, "valence {n}: 0x{hash:016X} != 0x{want:016X}");
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

    /// A source whose non-owner copies of shared vertices disagree with the owner's, as stale
    /// copies in a store could.
    struct Perturbed<'a> {
        dense: &'a DenseEvaluation,
        topology: &'a Topology,
    }

    impl PositionSource for Perturbed<'_> {
        fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
            let p = self.dense.position(face, level, i, j)?;
            let owner = self
                .topology
                .copies(face, level, i, j)
                .into_iter()
                .map(|c| {
                    let r = Region::of_tile(&c.key(level));
                    (c.face, r.x0() + c.i, r.y0() + c.j)
                })
                .min()?;
            if owner == (face, i, j) {
                return Some(p);
            }
            let offset = 1e-3 * f32::from(u16::try_from(face.index() + 1).unwrap());
            Some(p + Vec3::new(offset, offset, offset))
        }
    }

    #[test]
    fn prediction_reads_only_owner_copies() {
        let mesh = cube_sphere(2).unwrap();
        let dense =
            DenseEvaluation::new(&mesh, Level::new(3).unwrap(), &CorrectionMap::new()).unwrap();
        let src = Perturbed {
            dense: &dense,
            topology: mesh.topology(),
        };
        let c = Vec3::new(0.01, -0.02, 0.03);
        for l in 1..=3 {
            let level = Level::new(l).unwrap();
            let n = level.face_quads();
            for face in mesh.faces() {
                let region = Region::whole_face(face, level);
                let grid = evaluate_region(&mesh, &src, region, |_, _| c).unwrap();
                let clean = evaluate_region(&mesh, &dense, region, |_, _| c).unwrap();
                for j in 0..=n {
                    for i in 0..=n {
                        assert_eq!(
                            grid.get(i, j).unwrap().to_bits(),
                            clean.get(i, j).unwrap().to_bits(),
                            "{face:?} L{l} ({i}, {j}): a non-owner copy leaked into the prediction"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn set_rejects_bad_vertices_without_writing() {
        let mesh = cube_sphere(1).unwrap();
        let t = mesh.topology();
        let level = Level::new(2).unwrap();
        let mut map = CorrectionMap::new();
        let c = Vec3::new(0.1, 0.0, 0.0);
        assert_eq!(
            map.set(t, FaceId::new(6), level, 1, 1, c),
            Err(CorrectionError::UnknownFace(FaceId::new(6))),
            "unknown face"
        );
        assert_eq!(
            map.set(t, FaceId::new(0), level, 5, 1, c),
            Err(CorrectionError::OutOfTile {
                i: 5,
                j: 1,
                side: 5
            }),
            "outside the face"
        );
        assert_eq!(
            map.set(t, FaceId::new(0), level, 0, 1, Vec3::NAN),
            Err(CorrectionError::NotFinite),
            "not finite"
        );
        assert_eq!(
            map.set(t, FaceId::new(0), Level::ZERO, 0, 1, c),
            Err(CorrectionError::LevelZero),
            "level 0 is the mesh"
        );
        assert!(map.tiles.is_empty(), "nothing written");
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
