//! Building display tiles, as the store's refresh and rebuild jobs will, at level 9 (513²
//! vertices a face, the benchmark project's finest level): one face's positions at every level
//! below the finest, and a whole six-face mesh's tiles, positions and normals.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use geom::{
    BaseMesh, CorrectionMap, DenseEvaluation, FaceGrids, FaceId, Level, Region, build_mesh,
    build_positions, cube_sphere, normals,
};

fn main() {
    divan::main();
}

/// The cube sphere evaluated to level 9: its mesh and every face's finest positions and unit
/// normals.
///
/// # Panics
///
/// Panics if the fixed benchmark setup is rejected, which would be a bug in the crate.
fn level_9() -> (BaseMesh, FaceGrids, FaceGrids) {
    let mesh = cube_sphere(1).expect("cube");
    let finest = Level::new(9).expect("level");
    let dense = DenseEvaluation::new(&mesh, finest, &CorrectionMap::new()).expect("eval");
    let (mut positions, mut unit) = (FaceGrids::new(finest), FaceGrids::new(finest));
    for face in mesh.faces() {
        let grid = dense.grid(face, finest).expect("the face").clone();
        positions.insert(face, grid).expect("a whole face");
        let n = normals(&mesh, &dense, Region::whole_face(face, finest)).expect("normals");
        unit.insert(face, n).expect("a whole face");
    }
    (mesh, positions, unit)
}

/// One level-9 face's display positions, levels 0 to 8, with the chosen encoding.
///
/// # Panics
///
/// Panics if the fixed benchmark setup is rejected, which would be a bug in the crate.
#[divan::bench(sample_count = 20)]
fn build_positions_level_9(bencher: divan::Bencher<'_, '_>) {
    let (_, positions, _) = level_9();
    let face = FaceId::new(0);
    let grid = positions.get(face).expect("the face");
    let finest = positions.level();
    bencher.bench(|| build_positions(divan::black_box(face), grid, finest).expect("build"));
}

/// Every display tile of a six-face mesh at level 9, positions and normals, levels 0 to 8: six
/// faces' worth of what a rebuild does per face.
///
/// # Panics
///
/// Panics if the fixed benchmark setup is rejected, which would be a bug in the crate.
#[divan::bench(sample_count = 10)]
fn build_mesh_level_9(bencher: divan::Bencher<'_, '_>) {
    let (mesh, positions, unit) = level_9();
    bencher.bench(|| build_mesh(&mesh, divan::black_box(&positions), &unit).expect("build"));
}
