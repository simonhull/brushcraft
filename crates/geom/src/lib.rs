//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod correction;
#[cfg(any(test, feature = "test-fixtures"))]
mod dense;
mod display;
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
#[cfg(any(test, feature = "display-study"))]
pub use display::{BaseOffsets, FixedPoint};
#[cfg(any(test, feature = "display-study"))]
pub use display::{
    DisplayCodec, Interpolating, NormalEncoding, Precision, build_level_with, build_mesh_with,
    build_positions_with, decode_with,
};
pub use display::{
    DisplayError, DisplayTile, FaceCones, FaceGrids, FacePositions, FinestNormalCone, MeanNormals,
    build_cones, build_level, build_mesh, build_positions, build_tile, mip_normals,
    mip_normals_tile,
};
pub use eval::{
    EvalError, GatheredBlock, GatheredRing, Grid, PositionSource, Prediction, VertexStencil,
    cone_width, evaluate_region, gather_block, normal_tiles, normals, normals_into, parent_tiles,
    predict_region, refresh_normals_into, vertex_stencil,
};
pub use frame::Frame;
pub use key::{
    Channel, FaceId, LayerId, Level, LevelOutOfRange, MAX_LEVEL, Region, TILE_QUADS, TileKey,
    TileOutOfRange, VertexId,
};
pub use math::Vec3;
pub use mesh::{BaseMesh, MAX_VALENCE, MeshError};
pub use sphere::cube_sphere;
pub use topology::{BorderCopies, Copies, MAX_COPIES, TileBorder, Topology, VertexCopy};
