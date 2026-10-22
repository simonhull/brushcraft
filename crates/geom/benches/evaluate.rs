//! One level-9 tile (64×64 quads, 4,225 vertices): predict, frames, corrections. The M0 brush step
//! re-evaluates a handful of these, so this number bounds the "≤ 2 ms p99" target from below.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use geom::{
    CorrectionMap, DenseEvaluation, FaceId, Level, Region, TileKey, Vec3, cube_sphere,
    evaluate_region,
};

fn main() {
    divan::main();
}

/// Evaluates one level-9 tile from its level-8 parent.
///
/// # Panics
///
/// Panics if the fixed benchmark setup is rejected, which would be a bug in the crate.
#[divan::bench]
fn level_9_tile(bencher: divan::Bencher<'_, '_>) {
    let mesh = cube_sphere(1).expect("cube");
    let parent = DenseEvaluation::new(&mesh, Level::new(8).expect("level"), &CorrectionMap::new())
        .expect("eval");
    let key =
        TileKey::displacement(FaceId::new(0), Level::new(9).expect("level"), 3, 4).expect("key");
    let region = Region::of_tile(&key);
    bencher.bench(|| {
        evaluate_region(&mesh, &parent, divan::black_box(region), |i, j| {
            Vec3::new(
                0.0,
                0.0,
                1e-4 * f32::from(u16::try_from(i ^ j).unwrap_or(0)),
            )
        })
        .expect("evaluate")
    });
}
