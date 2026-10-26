//! M1a's spikes: experiments that answer a design question with measurements, built behind the
//! `spike` feature so the gate builds and tests them while the shipped renderer leaves them out.
//!
//! Each spike ends in a report in the docs repository (`research/platform/gpu-spikes.md`) and a
//! decision for the user; its code is then either promoted out of this module or deleted.
//!
//! - [`brush`]: can one brush's Rust source run on the GPU? The draw brush as a compute shader,
//!   compiled from `brush`'s own source by rust-gpu and written by hand in WGSL, each checked
//!   against the CPU and timed.

pub mod brush;
