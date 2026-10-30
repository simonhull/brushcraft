//! The `wgpu` renderer.
//!
//! For now this crate opens a GPU and runs compute work on it, headless: [`gpu`] chooses an
//! adapter, creates its device, and gives tests a harness ([`gpu::Gpu`] on a software adapter)
//! so every GPU test runs on lavapipe in the gate. [`surface`] is where frames go: a window's
//! swapchain, opened with an adapter that can present to it, or an offscreen texture.
//! [`Renderer::draw`] draws a frame there: its passes in order, the [`overlay`] last.
//! [`residency`] keeps tiles on the GPU within a budget, uploaded through [`upload`]'s staging
//! ring an epoch at a time, and [`tree`] is the tree of resident tiles the GPU walks. [`cull`]
//! and [`lod`] choose the tiles a frame draws, and [`raster`] draws their clusters (the
//! triangles of [`indices`]) into a visibility buffer, watertight at every seam.
//!
//! The crates below this one (`geom`, `store`, `brush`) never learn about the GPU: this crate
//! reads their data and uploads it.
//!
//! With the `spike` feature, the `spike` module holds M1a's GPU experiments, built and tested by
//! the gate but not part of the renderer.

pub mod camera;
pub mod cull;
mod draw;
pub mod gpu;
pub mod indices;
pub mod lod;
pub mod overlay;
pub mod raster;
pub mod residency;
#[cfg(feature = "spike")]
pub mod spike;
pub mod surface;
pub mod tree;
pub mod upload;

pub use draw::{DrawTimes, Drawn, FrameInputs, Outcome, Renderer};
