//! Each subdivision kernel on a level-9 tile (64×64 quads, 4,225 vertices), alone: the gather,
//! the Catmull–Clark stencils, the frames, the whole `predict_region`, `apply_into`, and the
//! normals in full (`normals_into`) and on the border (`refresh_normals_into`). Each runs on an
//! interior tile, all of whose vertices are strictly inside the face, and on a face-corner tile,
//! whose two face-border sides and base vertex take the one-vertex path.
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
    /// The tiles measured, by [`TILES`] name: one away from the face's border, where every
    /// vertex is strictly inside the face, and one at a face corner (a valence-3 base vertex),
    /// where two of its sides lie on the face's border.
    regions: [Region; 2],
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
    let tile = |tx, ty| {
        Region::of_tile(&TileKey::displacement(FaceId::new(0), level, tx, ty).expect("key"))
    };
    let regions = [tile(3, 4), tile(0, 0)];
    let side = level.tile_quads() + 1;
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
        regions,
        corrections,
    }
});

/// The tiles each bench runs on, as [`Fixture::regions`] holds them.
const TILES: [&str; 2] = ["interior", "face corner"];

/// The tile named `tile` (one of [`TILES`]).
fn region(f: &Fixture, tile: &str) -> Region {
    let k = TILES
        .iter()
        .position(|&t| t == tile)
        .expect("a tile of TILES");
    f.regions.get(k).copied().expect("a region per tile")
}

/// A tile's vertex count, `(quads + 1)²`.
fn vertices(region: Region) -> usize {
    let side = usize::try_from(region.quads() + 1).expect("side");
    side * side
}

/// Reading the parent block, its ring and the corner rings from the source.
#[divan::bench(args = TILES)]
fn gather(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let stages = Stages::new(&f.mesh, &f.dense, region).expect("stages");
    bencher.bench(|| stages.gather(&f.dense).expect("gather"));
}

/// The Catmull–Clark stencils: the tile and its ring from the gathered parent.
#[divan::bench(args = TILES)]
fn subdivide(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let stages = Stages::new(&f.mesh, &f.dense, region).expect("stages");
    bencher.bench(|| stages.subdivide());
}

/// Every vertex's tangent frame from the predicted tile.
#[divan::bench(args = TILES)]
fn frames(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let stages = Stages::new(&f.mesh, &f.dense, region).expect("stages");
    bencher.bench(|| stages.frames());
}

/// `predict_region`: gather, stencils and frames together.
#[divan::bench(args = TILES)]
fn predict(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    bencher.bench(|| predict_region(&f.mesh, &f.dense, divan::black_box(region)).expect("predict"));
}

/// `apply_into`: corrections applied in each vertex's frame, read through a correction view as
/// the store reads them.
#[divan::bench(args = TILES)]
fn apply(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let prediction = predict_region(&f.mesh, &f.dense, region).expect("predict");
    let view = f.corrections.view();
    let mut out = vec![Vec3::ZERO; vertices(region)];
    bencher.bench_local(|| {
        prediction
            .apply_into(|i, j| view.get(i, j).unwrap_or(Vec3::ZERO), &mut out)
            .expect("apply");
    });
}

/// `normals_into`: every vertex's normal from the tile's evaluated positions.
#[divan::bench(args = TILES)]
fn normals(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let mut out = vec![Vec3::ZERO; vertices(region)];
    bencher.bench_local(|| {
        normals_into(&f.mesh, &f.dense, divan::black_box(region), &mut out).expect("normals");
    });
}

/// `refresh_normals_into` with every neighbour stale: the whole border recomputed.
#[divan::bench(args = TILES)]
fn refresh_normals(bencher: Bencher<'_, '_>, tile: &str) {
    let f = &*FIXTURE;
    let region = region(f, tile);
    let mut out = vec![Vec3::ZERO; vertices(region)];
    bencher.bench_local(|| {
        refresh_normals_into(
            &f.mesh,
            &f.dense,
            divan::black_box(region),
            |_| true,
            &mut out,
        )
        .expect("refresh")
    });
}
