//! Tests of the tile evaluator, against the dense evaluator and the reference meshes.

use std::collections::{BTreeSet, HashMap};

use super::cone::gathered_tiles;
use super::*;
use crate::reference::pentagonal_prism;
use crate::test_rng::Rng;
use crate::topology::{corner_at, edge_to_grid};
use crate::{
    Channel, CorrectionMap, DenseEvaluation, FaceId, LayerId, Level, Region, TileKey, cube_sphere,
};

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

/// The base mesh's positions, with every face's vertex `(0, 0)` not a number.
struct Poisoned<'a>(&'a BaseMesh);

impl PositionSource for Poisoned<'_> {
    fn position(&self, face: FaceId, level: Level, i: u32, j: u32) -> Option<Vec3> {
        if (i, j) == (0, 0) {
            return Some(Vec3::new(f32::NAN, 0.0, 0.0));
        }
        self.0.position(face, level, i, j)
    }
}

/// A source with a position that isn't finite gives a prediction with NaN frames, not an
/// error, and not a panic: this runs in the debug test profile, where a debug assertion
/// would fire.
#[test]
fn non_finite_source_positions_give_nan_frames_not_a_panic() {
    let mesh = cube_sphere(1).unwrap();
    let region = Region::whole_face(FaceId::new(2), level(1));
    let prediction = predict_region(&mesh, &Poisoned(&mesh), region).unwrap();
    let side = region.quads() + 1;
    let frames: Vec<Frame> = (0..side)
        .flat_map(|j| (0..side).map(move |i| (i, j)))
        .map(|(i, j)| prediction.frame(i, j).unwrap())
        .collect();
    assert!(
        frames.iter().any(|f| !f.n().is_finite()),
        "a non-finite input shows in a frame"
    );
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

/// A tile's prediction holds what [`Prediction::TILE_BYTES`] charges for it: exactly that
/// for a tile of the most quads, less for a coarser one.
#[test]
fn a_tiles_prediction_holds_at_most_its_charge() {
    let mesh = cube_sphere(1).unwrap();
    let map = CorrectionMap::new();
    let dense = DenseEvaluation::new(&mesh, level(7), &map).unwrap();
    let finest = TileKey::displacement(FaceId::new(3), level(7), 1, 0).unwrap();
    let prediction = predict_region(&mesh, &dense, Region::of_tile(&finest)).unwrap();
    assert_eq!(
        prediction.heap_bytes(),
        Prediction::TILE_BYTES,
        "a full tile: the charge"
    );
    let coarse = TileKey::displacement(FaceId::new(3), level(3), 0, 0).unwrap();
    let smaller = predict_region(&mesh, &dense, Region::of_tile(&coarse)).unwrap();
    assert!(
        smaller.heap_bytes() < Prediction::TILE_BYTES,
        "a coarse tile: less, {}",
        smaller.heap_bytes()
    );
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
fn tiles_where(mesh: &BaseMesh, top: Level, keep: impl Fn(u16, u16) -> bool + Copy) -> Vec<Region> {
    let per_side = u16::try_from(top.tiles_per_side()).unwrap();
    mesh.faces()
        .flat_map(|f| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side)
                    .filter(move |&tx| keep(tx, ty))
                    .map(move |tx| Region::of_tile(&TileKey::displacement(f, top, tx, ty).unwrap()))
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
                let grid = evaluate_region(&mesh, &parent, r, block_corrections(&map, r)).unwrap();
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
    assert_normals_of_twins_are_bit_identical(&pentagonal_prism().subdivide().to_base_mesh(), 22);
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
        let got =
            normals(mesh, &src, region).unwrap_or_else(|e| panic!("{region:?} from {keys:?}: {e}"));
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
                            parents
                                .entry(*t)
                                .or_insert_with(|| parent_tiles(mesh, Region::of_tile(t)).unwrap());
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

/// The tiles [`gather`](fn@super::gather) reads for `region`, recorded read by read.
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

/// The hash of every stage's output over wide blocks of `prism(n)` subdivided once.
///
/// Under fixed-seed random corrections, it hashes predicted positions, positions with
/// corrections applied (so every frame component), normals, and refreshed border normals.
///
/// The blocks are whole faces and quarter faces at levels 6 and 7, so rows run 31 to 127
/// vertices inside the face, with base vertices of valence 3, 4 and `n` at their corners.
fn stage_hash(n: usize) -> u64 {
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

/// [`stage_hash`] for `prism(3)`, `prism(5)` and `prism(8)`, recorded from the stages as they
/// stood when these were pinned (before the row and lane rewrites, so those are held to the
/// per-vertex code's bits).
///
/// To regenerate after a deliberate change to the arithmetic, run
/// `BRUSHCRAFT_PRINT_GOLDEN=1 cargo nextest run -p geom stage_bits --no-capture`, check the
/// change was meant to move the bits, and paste the printed values here.
const STAGE_GOLDEN: [(usize, u64); 3] = [
    (3, 0x7BE8_3E3F_9151_C7E5),
    (5, 0x279C_99FF_705C_FE78),
    (8, 0x9955_E72D_6E3F_B2C9),
];

/// Pins the stages' bits on blocks wide enough that whole rows run through them: the
/// golden-bits test in `dense` works on level-3 faces, whose rows are too short to.
#[test]
fn stage_bits_are_pinned_on_wide_blocks() {
    let hashes = STAGE_GOLDEN.map(|(n, _)| (n, stage_hash(n)));
    if std::env::var_os("BRUSHCRAFT_PRINT_GOLDEN").is_some() {
        for (n, hash) in hashes {
            let groups = [48, 32, 16, 0].map(|shift| format!("{:04X}", (hash >> shift) & 0xFFFF));
            println!("({n}, 0x{}),", groups.join("_"));
        }
        return;
    }
    for ((n, hash), (_, want)) in hashes.into_iter().zip(STAGE_GOLDEN) {
        assert_eq!(hash, want, "prism({n}): 0x{hash:016X} != 0x{want:016X}");
    }
}
