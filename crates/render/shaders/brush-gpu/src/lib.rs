//! The draw brush's per-vertex step as a compute shader: `brush::draw_vertex` with the
//! [`Smoothstep`] falloff, compiled from `brush`'s own source by rust-gpu.
//!
//! The only code here is the entry point's plumbing: which vertex an invocation is, where its
//! position and delta live. The brush math is `brush`'s, unchanged. `render`'s spike loads the
//! SPIR-V this builds (`render/src/spike/brush_gpu.spv`) and checks it against the CPU.

#![no_std]

use brush::{Smoothstep, draw_vertex};
use spirv_std::glam::{UVec3, Vec3};
use spirv_std::spirv;

/// The brush, the same for every vertex of a dispatch: the uniform block at binding 0.
///
/// Laid out as `Params` in `draw.wgsl` and on the host: 32 bytes, `center` a vec3 aligned to 16,
/// and `radius` in its fourth lane.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Params {
    /// The brush centre.
    pub center: Vec3,
    /// The brush radius: positive and finite (the host checks it).
    pub radius: f32,
    /// `radius * radius`, computed on the host in `f32`, as `brush::draw` does.
    pub radius_squared: f32,
    /// The displacement at the centre.
    pub strength: f32,
    /// How many vertices the dispatch covers; invocations past it do nothing.
    pub count: u32,
    /// Pads the block to a multiple of 16 bytes.
    pub pad: u32,
}

/// One invocation per vertex: reads its position, writes `brush::draw_vertex`'s delta for it,
/// or 0 outside the brush. Invocations past `count` or the buffers' ends do nothing.
///
/// Bindings: 0, the [`Params`] uniform block; 1, the positions, three floats each, tightly
/// packed; 2, the deltas, one per vertex.
#[spirv(compute(threads(64)))]
pub fn main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &Params,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] positions: &[[f32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] deltas: &mut [f32],
) {
    let k = id.x as usize;
    // The last workgroup runs past `count`, and an empty footprint still binds one element: an
    // invocation without a vertex of its own, or outside the buffers, writes nothing.
    if k >= params.count as usize || k >= positions.len() || k >= deltas.len() {
        return;
    }
    let Vec3 { x, y, z } = params.center;
    #[expect(
        clippy::indexing_slicing,
        reason = "rust-gpu can't compile `slice::get` (an `Option<&T>` needs a pointer-to-integer \
                  cast, which SPIR-V lacks); `k` is checked against both lengths just above"
    )]
    {
        deltas[k] = draw_vertex(
            &Smoothstep,
            [x, y, z],
            params.radius,
            params.radius_squared,
            params.strength,
            positions[k],
        )
        .unwrap_or(0.0);
    }
}
