//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod correction;
mod dense;
mod eval;
#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixtures;
mod frame;
#[cfg(feature = "bench-kernels")]
#[doc(hidden)]
pub mod kernels;
mod key;
mod math;
mod mesh;
#[cfg(test)]
mod reference;
mod sphere;
mod subdiv;
#[cfg(test)]
mod test_rng;
mod topology;

pub use correction::{CorrectionError, CorrectionTile, CorrectionView, CorrectionViewMut};
pub use dense::{CorrectionMap, DenseEvaluation};
pub use eval::{
    EvalError, Grid, PositionSource, Prediction, cone_width, evaluate_region, normal_tiles,
    normals, normals_into, parent_tiles, predict_region, refresh_normals_into,
};
pub use frame::Frame;
pub use key::{
    Channel, FaceId, LayerId, Level, LevelOutOfRange, MAX_LEVEL, Region, TILE_QUADS, TileKey,
    TileOutOfRange, VertexId,
};
pub use math::Vec3;
pub use mesh::{BaseMesh, MAX_VALENCE, MeshError};
pub use sphere::cube_sphere;
pub use topology::{Copies, MAX_COPIES, Topology, VertexCopy};
