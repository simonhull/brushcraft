// The overlay pass: a text image drawn over the frame's top-left corner at a whole-number
// scale, ink near-white and the backing translucent black, premultiplied. The caller lays the
// text out; this only places and shades it (see overlay.rs).

// Where the overlay goes. All in frame pixels, but `scale`, frame pixels per texel.
struct Placement {
    // The overlay's top-left corner.
    origin: vec2<f32>,
    // The overlay's size: the image's times the scale.
    size: vec2<f32>,
    // The frame's size.
    frame: vec2<f32>,
    // Frame pixels per texel of the text image.
    scale: f32,
    // Unused: pads the struct to the 16-byte multiple uniforms need.
    pad: f32,
}

// Where the overlay goes, written each frame.
@group(0) @binding(0) var<uniform> placement: Placement;
// The text image: red 1 where a glyph has ink, 0 on the backing. Larger than the image when
// the image has shrunk; only the part `placement.size` covers is read.
@group(0) @binding(1) var text: texture_2d<f32>;

// The four corners of the overlay's rectangle as a triangle strip, in clip space.
@vertex
fn quad(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    let pixel = placement.origin + corner * placement.size;
    let clip = vec2<f32>(
        pixel.x / placement.frame.x * 2.0 - 1.0,
        1.0 - pixel.y / placement.frame.y * 2.0,
    );
    return vec4<f32>(clip, 0.0, 1.0);
}

// Each pixel's texel, unfiltered, as premultiplied ink over the backing.
@fragment
fn shade(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>((position.xy - placement.origin) / placement.scale);
    let ink = textureLoad(text, texel, 0).r;
    let backing = vec4<f32>(0.0, 0.0, 0.0, 0.6);
    let glyph = vec4<f32>(0.85, 0.85, 0.8, 1.0);
    return mix(backing, glyph, ink);
}
