//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod math;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod test_rng;

pub use math::Vec3;
mod key;
pub use key::{
    Channel, FaceId, LayerId, Level, LevelOutOfRange, MAX_LEVEL, Region, TILE_QUADS, TileKey,
    TileOutOfRange, VertexId,
};
mod correction;
mod eval;
mod frame;
pub use correction::{CorrectionError, CorrectionTile};
pub use eval::{EvalError, Grid, PositionSource, evaluate_region};
pub use frame::Frame;
mod mesh;
mod sphere;
mod subdiv;
mod topology;
pub use mesh::{BaseMesh, MeshError};
pub use sphere::cube_sphere;
pub use topology::{Topology, VertexCopy};
