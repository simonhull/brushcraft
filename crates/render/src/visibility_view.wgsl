// The visibility buffer shown as colours: a debug view until the shading pass (M1a3 Task 5).
// Each cluster (the id's node and cluster bits) gets a colour hashed from them, its quads a
// checker of two shades of it; the background is left to the clear colour (see
// visibility_view.rs).

// The visibility buffer (render::raster::VISIBILITY_FORMAT): one id a pixel, the frame's size.
@group(0) @binding(0) var ids: texture_2d<u32>;

// What the visibility buffer holds where nothing was drawn (render::raster::BACKGROUND).
const BACKGROUND: u32 = 0xffffffffu;

// Bits of an id below its cluster: the quad's (render::raster::VisibilityId).
const QUAD_BITS: u32 = 6u;

// A triangle covering the whole frame, from three vertex indices: (-1, -1), (3, -1), (-1, 3).
@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index & 1u) * 4u) - 1.0;
    let y = f32((index >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// A well-mixed 32-bit hash (a PCG output step), so neighbouring clusters get unrelated colours.
fn hash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Each pixel's cluster colour, linear, its quad's shade of it; the background is discarded.
@fragment
fn shade(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let id = textureLoad(ids, vec2<i32>(position.xy), 0).r;
    if id == BACKGROUND {
        discard;
    }
    let h = hash(id >> QUAD_BITS);
    let bytes = vec3<u32>(h & 0xffu, (h >> 8u) & 0xffu, (h >> 16u) & 0xffu);
    let colour = vec3<f32>(bytes) / 255.0 * 0.75 + 0.25;
    // The quad, 8y + x within its cluster: a checker of x + y.
    let quad = id & ((1u << QUAD_BITS) - 1u);
    let checker = ((quad & 7u) + (quad >> 3u)) & 1u;
    let shade = select(1.0, 0.8, checker == 1u);
    return vec4<f32>(colour * shade, 1.0);
}
