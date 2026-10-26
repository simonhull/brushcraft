//! Building one face's display tiles, every level below the finest, as the store's refresh and
//! rebuild jobs will: a face at level 9 (513² vertices, the benchmark project's finest level)
//! from its finest positions and normals.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use geom::{
    CorrectionMap, DenseEvaluation, FaceId, Level, Region, build_face, cube_sphere, normals,
};

fn main() {
    divan::main();
}

/// Builds every display tile of one level-9 face, levels 0 to 8, with the chosen encoding.
///
/// # Panics
///
/// Panics if the fixed benchmark setup is rejected, which would be a bug in the crate.
#[divan::bench(sample_count = 20)]
fn build_face_level_9(bencher: divan::Bencher<'_, '_>) {
    let mesh = cube_sphere(1).expect("cube");
    let finest = Level::new(9).expect("level");
    let dense = DenseEvaluation::new(&mesh, finest, &CorrectionMap::new()).expect("eval");
    let face = FaceId::new(0);
    let positions = dense.grid(face, finest).expect("the face").clone();
    let unit = normals(&mesh, &dense, Region::whole_face(face, finest)).expect("normals");
    bencher.bench(|| {
        build_face(
            divan::black_box(face),
            divan::black_box(&positions),
            &unit,
            finest,
        )
        .expect("build")
    });
}
