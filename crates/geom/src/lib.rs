//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod correction;
mod dense;
mod eval;
mod frame;
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

pub use correction::{CorrectionError, CorrectionTile};
pub use dense::{CorrectionMap, DenseEvaluation};
pub use eval::{
    EvalError, Grid, PositionSource, Prediction, evaluate_region, normals, parent_tiles,
    predict_region,
};
pub use frame::Frame;
pub use key::{
    Channel, FaceId, LayerId, Level, LevelOutOfRange, MAX_LEVEL, Region, TILE_QUADS, TileKey,
    TileOutOfRange, VertexId,
};
pub use math::Vec3;
pub use mesh::{BaseMesh, MeshError};
pub use sphere::cube_sphere;
pub use topology::{Topology, VertexCopy};
