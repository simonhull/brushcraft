//! The `wgpu` renderer.
//!
//! For now this crate opens a GPU and runs compute work on it, headless: [`gpu`] chooses an
//! adapter, creates its device, and gives tests a harness ([`gpu::Gpu`] on a software adapter)
//! so every GPU test runs on lavapipe in the gate. [`surface`] is where frames go: a window's
//! swapchain, opened with an adapter that can present to it, or an offscreen texture.
//!
//! The crates below this one (`geom`, `store`, `brush`) never learn about the GPU: this crate
//! reads their data and uploads it.
//!
//! With the `spike` feature, the `spike` module holds M1a's GPU experiments, built and tested by
//! the gate but not part of the renderer.

pub mod gpu;
#[cfg(feature = "spike")]
pub mod spike;
pub mod surface;
