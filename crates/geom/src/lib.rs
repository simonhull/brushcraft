//! Base mesh, subdivision, correction pyramid, tile math and quantization.
//!
//! No GPU or windowing code: everything here runs headless.

mod math;
#[cfg(test)]
mod test_rng;

pub use math::Vec3;
