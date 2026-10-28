// The depth pyramid (render::cull::DepthPyramid): each texel holds the farthest depth of the
// texels it covers, so a test against it is conservative.
//
// Depth is reversed (1 near, 0 far), so "farthest" is the minimum. Mip 0 is half the depth
// image's size rounded up to a power of two, so each mip after it is exactly half the one
// before, down to 1 × 1. A texel of mip m covers depth texels [x·2^(m+1), (x+1)·2^(m+1)) on
// each axis: reads past the depth image's edge are clamped to its last row or column, so
// texels past the edge only repeat texels inside it.

// from_depth: the depth image the frame's raster wrote.
@group(0) @binding(0) var depth: texture_depth_2d;
// from_mip: the mip below the one being written.
@group(0) @binding(1) var below: texture_2d<f32>;
// Both: the mip being written.
@group(0) @binding(2) var written: texture_storage_2d<r32float, write>;

// The farthest of the 2 × 2 depth texels under written texel `id`, each clamped to the image.
@compute @workgroup_size(8, 8)
fn from_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(written);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let last = textureDimensions(depth) - vec2<u32>(1u);
    let p = id.xy * 2u;
    let a = textureLoad(depth, min(p, last), 0);
    let b = textureLoad(depth, min(p + vec2<u32>(1u, 0u), last), 0);
    let c = textureLoad(depth, min(p + vec2<u32>(0u, 1u), last), 0);
    let d = textureLoad(depth, min(p + vec2<u32>(1u, 1u), last), 0);
    textureStore(written, id.xy, vec4<f32>(min(min(a, b), min(c, d)), 0.0, 0.0, 0.0));
}

// The farthest of the 2 × 2 texels under written texel `id` in the mip below, clamped.
@compute @workgroup_size(8, 8)
fn from_mip(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(written);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let last = textureDimensions(below) - vec2<u32>(1u);
    let p = id.xy * 2u;
    let a = textureLoad(below, min(p, last), 0).x;
    let b = textureLoad(below, min(p + vec2<u32>(1u, 0u), last), 0).x;
    let c = textureLoad(below, min(p + vec2<u32>(0u, 1u), last), 0).x;
    let d = textureLoad(below, min(p + vec2<u32>(1u, 1u), last), 0).x;
    textureStore(written, id.xy, vec4<f32>(min(min(a, b), min(c, d)), 0.0, 0.0, 0.0));
}
