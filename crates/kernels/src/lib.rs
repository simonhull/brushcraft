//! Brush kernels shared by the CPU and the GPU.
//!
//! The per-vertex math of a brush lives here, written once: the CPU runs it as plain Rust, and
//! the GPU spike compiles the same source for the GPU. So the crate is `no_std`, allocates
//! nothing, depends on nothing, and works on plain `[f32; 3]` positions (what a GPU buffer holds)
//! rather than on another crate's vector type.
//!
//! Kernels are generic over the pieces a brush is composed from, starting with its
//! [`Falloff`], so each built-in brush is monomorphised into straight-line code.
//!
//! M0 has one kernel, [`draw`], with a [`Smoothstep`] falloff: enough to measure edit latency.

#![no_std]

mod draw;
mod falloff;
mod math;

pub use draw::draw;
pub use falloff::{Falloff, Smoothstep};
