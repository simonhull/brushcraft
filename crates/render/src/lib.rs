//! The `wgpu` renderer.
//!
//! For now this crate opens a GPU and runs compute work on it, headless: [`gpu`] chooses an
//! adapter, creates its device, and gives tests a harness ([`gpu::Gpu`] on a software adapter)
//! so every GPU test runs on lavapipe in the gate. [`surface`] is where frames go: a window's
//! swapchain, opened with an adapter that can present to it, or an offscreen texture.
//! [`Renderer::draw`] draws a frame there: its passes in order, then the [`overlay`], and last
//! the `ui` module's pass, which draws the `ui` crate's shapes in display colour.
//! [`residency`] keeps tiles on the GPU within a budget, uploaded through [`upload`]'s staging
//! ring an epoch at a time (a finest tile as its positions with the one-vertex ring, whose
//! normals the `normals` module computes on the GPU), and [`tree`] is the tree of resident tiles
//! the GPU walks. [`cull`]
//! and [`lod`] choose the tiles a frame draws, and [`raster`] draws their clusters (the
//! triangles of `indices`) into a visibility buffer, watertight at every seam. [`shade`] rebuilds
//! the surface drawn at each pixel and shades it through a [`material`] (MatCap, ours, generated)
//! into a linear HDR image, which `tone` brings to the frame's format; `visibility_view` shows
//! the visibility buffer as colours instead, a debug view.
//!
//! [`aa`] smooths edges: frames accumulated at jittered sample points while the camera is still,
//! an edge filter in the tone map while it moves.
//!
//! [`timing`] times a frame's phases on the GPU by timestamp queries, read back without
//! waiting, where the device has them ([`Renderer::enable_timing`]).
//!
//! The crates below this one (`geom`, `store`, `brush`) never learn about the GPU: this crate
//! reads their data and uploads it.
//!
//! With the `spike` feature, the `spike` module holds M1a's GPU experiments, built and tested by
//! the gate but not part of the renderer.
//!
//! # Shaders
//!
//! Per-pixel and per-vertex WGSL never indexes a function-local array by a value known only at
//! run time: not a `let` or `var` array, an array inside a local copy of a struct, or a
//! `var<private>` one, even by a loop counter. It picks with `select` or a branch instead, or
//! reads the buffer's array itself. Some drivers move such an array to slow memory, and the
//! whole pass slows with it: the shading pass ran 6 to 20 times slower on an NVIDIA desktop GPU
//! while it picked its triangle's corners that way. Lavapipe, which the tests run on, doesn't
//! show it, so the shaders' tests hold them to the rule through `shader_check`.

pub mod aa;
pub mod camera;
pub mod cull;
mod draw;
pub mod gpu;
pub(crate) mod indices;
pub mod lod;
pub mod material;
pub(crate) mod normals;
pub mod overlay;
pub mod raster;
pub mod residency;
pub mod shade;
#[cfg(test)]
mod shader_check;
#[cfg(feature = "spike")]
pub mod spike;
pub mod standin;
pub mod surface;
pub mod timing;
pub(crate) mod tone;
pub mod tree;
mod ui;
pub mod upload;
mod visibility_view;

pub use draw::{
    DrawError, DrawTimes, Drawn, DroppedReadbacks, FrameInputs, Look, Outcome, Renderer, Scene,
    SceneView,
};

/// The most frames the GPU should have at once: submitted by the CPU and not yet finished.
///
/// A frame's readbacks (its cut, its requests, its GPU times) are read a frame or two after it
/// is submitted, once the GPU has finished it. A CPU that submits frames faster than the GPU
/// finishes them builds a queue: input waits behind it, and each readback ring runs out of
/// buffers, so readbacks are dropped. The app starts no frame while the GPU has this many
/// (`app`'s scheduler), and each readback ring holds more buffers than this, so a submitted frame
/// always finds one: a frame encoded while the GPU still runs the one before it, and a buffer to
/// spare for a readback mapped but not yet read.
///
/// Two lets the CPU encode a frame while the GPU draws the last, and no more.
pub const FRAMES_IN_FLIGHT: usize = 2;
