// The tone map (render::tone): the shading pass's linear HDR image into the frame's format.
// A fullscreen triangle in the frame's render pass; background pixels (alpha 0) are discarded,
// so the frame's clear colour shows there.

// The shaded image (render::shade::HDR_FORMAT), the frame's size.
@group(0) @binding(0) var shaded: texture_2d<f32>;

// Whether the frame's format stores what it is given (not an sRGB format, which encodes in
// hardware): then the shader encodes sRGB itself.
override ENCODE_SRGB: bool = false;

// Where the curve leaves the identity.
const KNEE: f32 = 0.8;

// A triangle covering the whole frame, from three vertex indices: (-1, -1), (3, -1), (-1, 3).
@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index & 1u) * 4u) - 1.0;
    let y = f32((index >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// The curve, per channel (render::tone): the identity up to KNEE, then a shoulder rising to 1,
// KNEE + (1 - KNEE)(1 - exp(-(c - KNEE) / (1 - KNEE))), whose slope is 1 at the knee.
fn tone(c: vec3<f32>) -> vec3<f32> {
    let over = max(c - vec3<f32>(KNEE), vec3<f32>(0.0));
    let shoulder = vec3<f32>(KNEE) + (1.0 - KNEE) * (vec3<f32>(1.0) - exp(-over / (1.0 - KNEE)));
    return select(max(c, vec3<f32>(0.0)), shoulder, c > vec3<f32>(KNEE));
}

// The sRGB encoding of linear `c` in 0..1.
fn srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

// A pixel of the shaded image, tone mapped; the background is discarded.
@fragment
fn map(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = textureLoad(shaded, vec2<i32>(position.xy), 0);
    if texel.a == 0.0 {
        discard;
    }
    var c = tone(texel.rgb);
    if ENCODE_SRGB {
        c = srgb(c);
    }
    return vec4<f32>(c, 1.0);
}
