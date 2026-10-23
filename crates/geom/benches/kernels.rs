//! Each subdivision kernel on one level-9 tile (64×64 quads, 4,225 vertices), alone: the gather,
//! the Catmull–Clark stencils, the frames, the whole `predict_region`, `apply_into`, and the
//! normals in full (`normals_into`) and on the border (`refresh_normals_into`).
//!
//! Run with `cargo bench -p geom --bench kernels --features bench-kernels`: the stages inside
//! `predict_region` are reached through `geom::kernels`, which only that feature compiles.

#![expect(
    clippy::expect_used,
    clippy::missing_panics_doc,
    reason = "benchmark setup; a failure here should abort the run, and every bench would say so"
)]

use std::sync::LazyLock;

use divan::Bencher;
use geom::kernels::Stages;
use geom::{
    BaseMesh, CorrectionMap, CorrectionTile, DenseEvaluation, FaceId, Level, Region, TileKey, Vec3,
    cube_sphere, normals_into, predict_region, refresh_normals_into,
};

fn main() {
    divan::main();
}

/// The benchmark's mesh and every level of it up to the tile's, evaluated once for all benches.
struct Fixture {
    /// A cube projected to a sphere: valence-3 corners, regular elsewhere.
    mesh: BaseMesh,
    /// Levels 0 to 9, without corrections.
    dense: DenseEvaluation,
    /// The tile measured: away from the face's border, so every vertex takes the common path.
    region: Region,
    /// A correction for every vertex of the tile, quantized as the store keeps them.
    corrections: CorrectionTile,
}

/// The shared [`Fixture`].
///
/// # Panics
///
/// If the fixed setup is rejected, which would be a bug in the crate.
static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    let mesh = cube_sphere(1).expect("cube");
    let level = Level::new(9).expect("level");
    let dense = DenseEvaluation::new(&mesh, level, &CorrectionMap::new()).expect("eval");
    let key = TileKey::displacement(FaceId::new(0), level, 3, 4).expect("key");
    let region = Region::of_tile(&key);
    let side = region.quads() + 1;
    let values: Vec<Vec3> = (0..side * side)
        .map(|k| {
            let wave = |m: u32| 1e-4 * f32::from(u16::try_from(k % m).unwrap_or(0));
            Vec3::new(wave(7), wave(11), wave(13))
        })
        .collect();
    let corrections = CorrectionTile::from_values(side, &values).expect("corrections");
    Fixture {
        mesh,
        dense,
        region,
        corrections,
    }
});

/// The tile's vertex count, `(quads + 1)²`.
fn vertices(f: &Fixture) -> usize {
    let side = usize::try_from(f.region.quads() + 1).expect("side");
    side * side
}

/// Reading the parent block, its ring and the corner rings from the source.
#[divan::bench]
fn gather(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let stages = Stages::new(&f.mesh, &f.dense, f.region).expect("stages");
    bencher.bench(|| stages.gather(&f.dense).expect("gather"));
}

/// The Catmull–Clark stencils: the tile and its ring from the gathered parent.
#[divan::bench]
fn subdivide(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let stages = Stages::new(&f.mesh, &f.dense, f.region).expect("stages");
    bencher.bench(|| stages.subdivide());
}

/// Every vertex's tangent frame from the predicted tile.
#[divan::bench]
fn frames(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let stages = Stages::new(&f.mesh, &f.dense, f.region).expect("stages");
    bencher.bench(|| stages.frames());
}

/// `predict_region`: gather, stencils and frames together.
#[divan::bench]
fn predict(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    bencher
        .bench(|| predict_region(&f.mesh, &f.dense, divan::black_box(f.region)).expect("predict"));
}

/// `apply_into`: corrections applied in each vertex's frame, read through a correction view as
/// the store reads them.
#[divan::bench]
fn apply(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let prediction = predict_region(&f.mesh, &f.dense, f.region).expect("predict");
    let view = f.corrections.view();
    let mut out = vec![Vec3::ZERO; vertices(f)];
    bencher.bench_local(|| {
        prediction
            .apply_into(|i, j| view.get(i, j).unwrap_or(Vec3::ZERO), &mut out)
            .expect("apply");
    });
}

/// `normals_into`: every vertex's normal from the tile's evaluated positions.
#[divan::bench]
fn normals(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let mut out = vec![Vec3::ZERO; vertices(f)];
    bencher.bench_local(|| {
        normals_into(&f.mesh, &f.dense, divan::black_box(f.region), &mut out).expect("normals");
    });
}

/// `refresh_normals_into` with every neighbour stale: the whole border recomputed.
#[divan::bench]
fn refresh_normals(bencher: Bencher<'_, '_>) {
    let f = &*FIXTURE;
    let mut out = vec![Vec3::ZERO; vertices(f)];
    bencher.bench_local(|| {
        refresh_normals_into(
            &f.mesh,
            &f.dense,
            divan::black_box(f.region),
            |_| true,
            &mut out,
        )
        .expect("refresh")
    });
}
