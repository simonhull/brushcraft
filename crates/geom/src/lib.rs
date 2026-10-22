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
#[expect(dead_code, reason = "used by evaluation (M0a Task 8)")]
mod frame;
pub use correction::{CorrectionError, CorrectionTile};
pub use frame::Frame;
mod mesh;
mod sphere;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used by evaluation (M0a Tasks 7–8)")
)]
mod subdiv;
mod topology;
pub use mesh::{BaseMesh, MeshError};
pub use sphere::cube_sphere;
pub use topology::{Topology, VertexCopy};
