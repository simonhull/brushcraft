//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod correction;
#[cfg(any(test, feature = "test-fixtures"))]
mod dense;
mod eval;
#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixtures;
mod frame;
mod key;
mod math;
mod mesh;
#[cfg(test)]
mod reference;
mod soa;
mod sphere;
#[cfg(feature = "bench-stages")]
#[doc(hidden)]
pub mod stages;
mod subdiv;
#[cfg(test)]
mod test_rng;
mod topology;

pub use correction::{
    CorrectionError, CorrectionTile, CorrectionView, CorrectionViewMut, RowUpdate,
};
#[cfg(any(test, feature = "test-fixtures"))]
pub use dense::{CorrectionMap, DenseEvaluation};
pub use eval::{
    EvalError, Grid, PositionSource, Prediction, cone_width, evaluate_region, normal_tiles,
    normals, normals_into, parent_tiles, predict_region, refresh_normals_into,
};
#[cfg(feature = "eval-inputs")]
pub use eval::{GatheredBlock, GatheredRing, VertexStencil, gather_block, vertex_stencil};
pub use frame::Frame;
pub use key::{
    Channel, FaceId, LayerId, Level, LevelOutOfRange, MAX_LEVEL, Region, TILE_QUADS, TileKey,
    TileOutOfRange, VertexId,
};
pub use math::Vec3;
pub use mesh::{BaseMesh, MAX_VALENCE, MeshError};
pub use sphere::cube_sphere;
pub use topology::{BorderCopies, Copies, MAX_COPIES, TileBorder, Topology, VertexCopy};
