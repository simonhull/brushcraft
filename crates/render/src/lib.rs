//! The `wgpu` renderer.
//!
//! For now this crate opens a GPU and runs compute work on it, headless: [`gpu`] chooses an
//! adapter, creates its device, and gives tests a harness ([`gpu::Gpu`] on a software adapter)
//! so every GPU test runs on lavapipe in the gate. Drawing comes with M1a.
//!
//! The crates below this one (`geom`, `store`, `brush`) never learn about the GPU: this crate
//! reads their data and uploads it.

pub mod gpu;
