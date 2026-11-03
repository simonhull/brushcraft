//! Brush math shared by the CPU and the GPU: what a brush does to each vertex.
//!
//! The per-vertex math of a brush lives here, written once: the CPU runs it as plain Rust, and
//! the GPU spike compiles the same source for the GPU. So the crate is `no_std`, allocates
//! nothing, depends on nothing, and works on plain `[f32; 3]` positions (what a GPU buffer holds)
//! rather than on another crate's vector type.
//!
//! Brushes are generic over the pieces a brush is composed from, starting with its
//! [`Falloff`], so each built-in brush is monomorphised into straight-line code.
//!
//! M0 has one brush, [`draw`](fn@draw), with a [`Smoothstep`] falloff: enough to measure edit
//! latency. [`draw_vertex`] is its step for one vertex. A placed dab reaches an action as an
//! [`Imprint`]: [`draw_imprint`] and [`draw_imprint_vertex`] are the draw brush for one, its
//! falloff [`Hardened`] by the dab's hardness, and [`draw_move`] ([`draw_imprint_move`]) turns a
//! vertex's displacement into its move, along the imprint's normal (the dab's area normal), one
//! direction for every vertex.

#![no_std]

mod draw;
mod falloff;
mod imprint;
mod math;

pub use draw::{draw, draw_vertex};
pub use falloff::{Falloff, Smoothstep};
pub use imprint::{
    Hardened, Imprint, draw_imprint, draw_imprint_gradient, draw_imprint_move, draw_imprint_vertex,
    draw_move,
};
