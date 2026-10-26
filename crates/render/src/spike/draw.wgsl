// The draw brush's per-vertex step on the GPU: the WGSL twin of `brush::draw_vertex` with the
// `Smoothstep` falloff, one invocation per vertex.
//
// Every operation follows the Rust source in `crates/brush` in the same order, so the two can be
// compared value for value: the squared distance, the bit-level reciprocal square root refined by
// three Newton steps (`brush`'s own `sqrt`, not WGSL's), the quotient by the radius, and the
// smoothstep polynomial in Horner form. What can still differ is the hardware's: fused
// multiply-adds the driver may form, and subnormals it may flush to zero. The spike's tests
// measure both.

// The brush, the same for every vertex of a dispatch. 32 bytes: `center` is a vec3, aligned to
// 16, and `radius` fills its fourth lane.
struct Params {
    // The brush centre.
    center: vec3<f32>,
    // The brush radius: positive and finite (the host checks it, as `brush::draw` does).
    radius: f32,
    // `radius * radius`, computed on the host in `f32`, as `brush::draw` does.
    radius_squared: f32,
    // The displacement at the centre.
    strength: f32,
    // How many vertices the dispatch covers; invocations past it do nothing.
    count: u32,
    // Pads the struct to a multiple of 16 bytes.
    pad: u32,
}

// The brush for this dispatch.
@group(0) @binding(0) var<uniform> params: Params;

// The vertex positions, three floats each, tightly packed (`[[f32; 3]]` on the host): a vec3
// array would be padded to 16 bytes per element.
@group(0) @binding(1) var<storage, read> positions: array<f32>;

// One displacement per vertex: the brush's step along the normal, 0 outside its sphere.
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;

// `2^24`, which lifts a subnormal input into the normal range for `brush_sqrt`.
const SUBNORMAL_LIFT: f32 = 16777216.0;

// `2^-12`, the square root of `1 / SUBNORMAL_LIFT`, which undoes the lift on the result.
const SUBNORMAL_UNDO: f32 = 0.000244140625;

// The smallest positive normal `f32`, `2^-126`.
const MIN_POSITIVE: f32 = 1.17549435e-38;

// `brush`'s `rsqrt_newton`: `sqrt(x)` for a positive, finite, normal `x`, as `x * (1 / sqrt(x))`
// from the bit-level estimate and three Newton steps.
fn rsqrt_newton(x: f32) -> f32 {
    let half = 0.5 * x;
    var y = bitcast<f32>(0x5F375A86u - (bitcast<u32>(x) >> 1u));
    y *= 1.5 - half * y * y;
    y *= 1.5 - half * y * y;
    y *= 1.5 - half * y * y;
    return x * y;
}

// `brush`'s `sqrt`: 0 for zero and negative inputs, the input for infinity and NaN, subnormals
// lifted into the normal range first. Infinity and NaN are recognised by their exponent bits:
// WGSL lets the compiler assume floating-point values are never either, so a test like
// `x != x` may be folded away.
fn brush_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if (bitcast<u32>(x) & 0x7F800000u) == 0x7F800000u {
        return x;
    }
    if x < MIN_POSITIVE {
        return rsqrt_newton(x * SUBNORMAL_LIFT) * SUBNORMAL_UNDO;
    }
    return rsqrt_newton(x);
}

// `Smoothstep::weight`: 1 at and inside the centre, `1 - t^2 (3 - 2t)` inside the rim, 0 at and
// past the rim and for NaN.
fn smoothstep_weight(t: f32) -> f32 {
    if t <= 0.0 {
        return 1.0;
    }
    if t < 1.0 {
        return 1.0 - t * t * (3.0 - 2.0 * t);
    }
    return 0.0;
}

// `brush::draw_vertex` followed by `unwrap_or(0.0)`: the displacement of the vertex at `p`.
fn draw_vertex(p: vec3<f32>) -> f32 {
    let dx = p.x - params.center.x;
    let dy = p.y - params.center.y;
    let dz = p.z - params.center.z;
    let d2 = dx * dx + dy * dy + dz * dz;
    if !(d2 < params.radius_squared) {
        return 0.0;
    }
    return params.strength * smoothstep_weight(brush_sqrt(d2) / params.radius);
}

// One invocation per vertex: reads its position, writes its displacement.
@compute @workgroup_size(64)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    if k >= params.count {
        return;
    }
    let p = vec3<f32>(positions[3u * k], positions[3u * k + 1u], positions[3u * k + 2u]);
    deltas[k] = draw_vertex(p);
}
