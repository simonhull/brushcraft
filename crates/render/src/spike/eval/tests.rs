//! The GPU's level step against `geom`'s, and against itself at seams, on lavapipe.
//!
//! Two meshes: a cube sphere split twice (base vertices of valence 3 at the cube's corners, three
//! tiles meeting at each, and of valence 4 elsewhere) and a subdivided pentagonal prism (valence
//! 5 too, five tiles meeting), both at level 7, where every
//! face holds 2 × 2 tiles of 64 quads, with random corrections at every level. The differences
//! from `geom` are printed for the spike report (`cargo nextest run -p render --features spike
//! --no-capture spike::eval`).

use geom::fixtures::subdivided_pentagonal_prism;
use geom::{CorrectionMap, DenseEvaluation, FaceId, cube_sphere, evaluate_region, normals};

use super::*;
use crate::gpu::GpuTest;

/// The level evaluated: the first whose faces hold several 64-quad tiles.
const LEVEL: u8 = 7;

/// A seeded random generator (splitmix64), so every run evaluates the same surface.
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float uniform in `[-scale, scale]`.
    fn signed(&mut self, scale: f32) -> f32 {
        let unit =
            f32::from(u16::try_from(self.next_u64() >> 48).expect("16 bits")) / f32::from(u16::MAX);
        scale * (2.0 * unit - 1.0)
    }
}

/// A mesh with random corrections at every level up to [`LEVEL`], evaluated densely by `geom`.
struct Fixture {
    /// The base mesh.
    mesh: BaseMesh,
    /// The corrections.
    corrections: CorrectionMap,
    /// Every level, evaluated by `geom`.
    dense: DenseEvaluation,
    /// The level evaluated.
    level: Level,
}

impl Fixture {
    /// `mesh` with corrections of up to `0.002` (on a surface of radius about 1) on every
    /// vertex of every level.
    fn new(mesh: BaseMesh, seed: u64) -> Self {
        let level = Level::new(LEVEL).expect("a valid level");
        let mut corrections = CorrectionMap::new();
        let mut rng = Rng(seed);
        for l in 1..=LEVEL {
            let l = Level::new(l).expect("a valid level");
            let n = l.face_quads();
            for face in mesh.faces() {
                for j in 0..=n {
                    for i in 0..=n {
                        let c = Vec3::new(rng.signed(0.002), rng.signed(0.002), rng.signed(0.002));
                        corrections
                            .set(mesh.topology(), face, l, i, j, c)
                            .expect("a valid correction");
                    }
                }
            }
        }
        let dense = DenseEvaluation::new(&mesh, level, &corrections).expect("evaluates");
        Self {
            mesh,
            corrections,
            dense,
            level,
        }
    }

    /// Every tile of the level.
    fn tiles(&self) -> Vec<TileKey> {
        let per_side = u16::try_from(self.level.tiles_per_side()).expect("few tiles");
        self.mesh
            .faces()
            .flat_map(|face| {
                (0..per_side).flat_map(move |ty| {
                    (0..per_side).map(move |tx| {
                        TileKey::displacement(face, Level::new(LEVEL).expect("valid"), tx, ty)
                            .expect("a tile of the face")
                    })
                })
            })
            .collect()
    }

    /// The correction at face-grid vertex `(x, y)` of `face`.
    fn correction(&self, face: FaceId, x: u32, y: u32) -> Vec3 {
        self.corrections.get(face, self.level, x, y)
    }

    /// Every tile's inputs.
    fn inputs(&self) -> Inputs {
        Inputs::gather(&self.mesh, &self.dense, &self.tiles(), |tile, i, j| {
            let region = Region::of_tile(tile);
            self.correction(tile.face(), region.x0() + i, region.y0() + j)
        })
        .expect("the inputs gather")
    }

    /// `geom`'s positions and normals of every tile, in the order of `tiles`.
    fn expected(&self, tiles: &[TileKey]) -> Evaluated {
        let mut positions = Vec::new();
        let mut normals_out = Vec::new();
        for tile in tiles {
            let region = Region::of_tile(tile);
            let grid = evaluate_region(&self.mesh, &self.dense, region, |i, j| {
                self.correction(tile.face(), region.x0() + i, region.y0() + j)
            })
            .expect("evaluates");
            positions.extend(grid.as_slice().iter().map(|p| [p.x, p.y, p.z]));
            let normal_grid = normals(&self.mesh, &self.dense, region).expect("normals");
            normals_out.extend(normal_grid.as_slice().iter().map(|n| [n.x, n.y, n.z]));
        }
        Evaluated {
            positions,
            normals: normals_out,
        }
    }
}

/// The fixtures: a cube sphere and a subdivided pentagonal prism.
fn fixtures() -> [(&'static str, Fixture); 2] {
    [
        (
            "cube sphere",
            Fixture::new(cube_sphere(2).expect("a valid mesh"), 0xC0BE),
        ),
        (
            "pentagonal prism",
            Fixture::new(subdivided_pentagonal_prism().expect("a valid mesh"), 0x9E47),
        ),
    ]
}

/// Evaluates every tile of `fixture` on `gpu`.
fn evaluate(gpu: &Gpu, inputs: &Inputs) -> Evaluated {
    let eval = GpuEval::new(gpu).expect("the shaders build");
    let run = eval.prepare(gpu, inputs).expect("the inputs upload");
    run.dispatch(gpu).expect("the dispatch");
    run.read(gpu).expect("the readback")
}

/// The vertex range of tile `slot` in an [`Evaluated`].
fn tile_range(slot: usize) -> std::ops::Range<usize> {
    let vertices = VERTICES as usize;
    slot * vertices..(slot + 1) * vertices
}

#[test]
fn the_gpu_step_matches_geom_within_rounding() {
    GpuTest::run(|gpu| {
        for (name, fixture) in fixtures() {
            let inputs = fixture.inputs();
            let got = evaluate(gpu, &inputs);
            let want = fixture.expected(inputs.tiles());
            // Interior tiles, and tiles with a base vertex of valence other than 4 at a corner.
            let mut differences = [[VecDifference::default(); 2]; 2];
            for slot in 0..inputs.tiles().len() {
                let kind = usize::from(inputs.has_extraordinary_corner(slot));
                let range = tile_range(slot);
                differences[kind][0].add(
                    &got.positions[range.clone()],
                    &want.positions[range.clone()],
                );
                differences[kind][1].add(&got.normals[range.clone()], &want.normals[range]);
            }
            for (kind, [positions, normals]) in
                ["interior", "extraordinary"].into_iter().zip(differences)
            {
                println!(
                    "{name}, {} tiles, {kind}: positions {positions}; normals {normals}",
                    inputs.tiles().len()
                );
                assert!(
                    positions.max_distance() < 1e-5,
                    "{name} {kind}: positions follow geom: {positions}"
                );
                // The corrections are large for level 7's spacing (about 0.01), so the surface
                // is rough and its normals amplify rounding: this is a bound on doing the same
                // computation, not a precision target.
                assert!(
                    normals.max_distance() < 1e-2,
                    "{name} {kind}: normals follow geom: {normals}"
                );
            }
            assert!(
                differences[1][0].compared() > 0 && differences[0][0].compared() > 0,
                "{name}: both kinds of tile are checked"
            );
        }
    });
}

#[test]
fn every_vertex_tiles_share_has_the_same_bits_in_each() {
    GpuTest::run(|gpu| {
        // The base vertices each mesh shares between tiles, by valence: the cube sphere split
        // twice has its 8 cube corners (3) and 18 others (4); the prism its 10 old corners (3),
        // 20 edge and side-face points (4) and 2 cap centres (5).
        let shared = [[0, 0, 0, 8, 18, 0, 0, 0], [0, 0, 0, 10, 20, 2, 0, 0]];
        for (((name, fixture), meeting), base_vertices) in
            fixtures().into_iter().zip([3, 5]).zip(shared)
        {
            let inputs = fixture.inputs();
            let got = evaluate(gpu, &inputs);
            let consistency = seam_consistency(&fixture.mesh, inputs.tiles(), &got);
            println!("{name} on {}: {consistency}", gpu.adapter_info().name);
            assert_eq!(
                (
                    consistency.position_mismatches,
                    consistency.normal_mismatches
                ),
                (0, 0),
                "{name}: seams are self-consistent: {consistency}"
            );
            assert!(
                consistency.most_copies >= meeting,
                "{name}: the {meeting} tiles meeting at a base vertex are compared: {consistency}"
            );
            assert_eq!(
                consistency.base_vertices, base_vertices,
                "{name}: every base vertex is compared, by valence"
            );
        }
    });
}

#[test]
fn the_seam_check_finds_a_vertex_that_differs() {
    GpuTest::run(|gpu| {
        let (_, fixture) = fixtures().into_iter().next().expect("a fixture");
        let inputs = fixture.inputs();
        let mut got = evaluate(gpu, &inputs);
        let flip = |evaluated: &mut Evaluated, k: usize| {
            evaluated.positions[k][0] = f32::from_bits(evaluated.positions[k][0].to_bits() ^ 1);
        };
        // Tile 0's vertex (64, 32) lies on its border with tile 1 of the same face.
        let same_face = (32 * SIDE + 64) as usize;
        flip(&mut got, same_face);
        let consistency = seam_consistency(&fixture.mesh, inputs.tiles(), &got);
        assert_eq!(
            (consistency.position_mismatches, consistency.across_faces),
            (1, 0),
            "one flipped bit is found, within a face: {consistency}"
        );
        flip(&mut got, same_face);
        // Its vertex (32, 0) lies on the face's edge, shared with a tile of the next face.
        flip(&mut got, 32);
        let across = seam_consistency(&fixture.mesh, inputs.tiles(), &got);
        assert_eq!(
            (across.position_mismatches, across.across_faces),
            (1, 1),
            "one flipped bit is found, across faces: {across}"
        );
    });
}

#[test]
fn inputs_reject_tiles_that_are_not_finest_size() {
    let mesh = cube_sphere(1).expect("a valid mesh");
    let dense = DenseEvaluation::new(&mesh, Level::new(2).expect("valid"), &CorrectionMap::new())
        .expect("evaluates");
    let face = mesh.faces().next().expect("a face");
    let small = TileKey::displacement(face, Level::new(2).expect("valid"), 0, 0).expect("a tile");
    assert!(
        matches!(
            Inputs::gather(&mesh, &dense, &[small], |_, _, _| Vec3::ZERO),
            Err(EvalSpikeError::TileSize(_))
        ),
        "a 4-quad tile"
    );
}
