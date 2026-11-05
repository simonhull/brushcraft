// The UI pass: rounded rects shaded by distance to the edge, then glyphs from the atlas. Colours
// are display values, and on the display view blending happens in them (see ui.rs).

struct View {
    // The frame's size, pixels.
    size: vec2<f32>,
    // 1 where the pass draws on an sRGB view, and must hand it linear colours.
    linear: f32,
    // Pads the struct.
    pad: f32,
    // The opacity of each layer.
    fades: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var atlas: texture_2d<f32>;

// A display value in the form the target wants.
fn encoded(colour: vec4<f32>) -> vec4<f32> {
    let low = colour.rgb / 12.92;
    let high = pow((colour.rgb + 0.055) / 1.055, vec3<f32>(2.4));
    let decoded = select(high, low, colour.rgb <= vec3<f32>(0.04045));
    return vec4<f32>(select(colour.rgb, decoded, view.linear > 0.5), colour.a);
}

fn clip(pixel: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(pixel.x / view.size.x * 2.0 - 1.0, 1.0 - pixel.y / view.size.y * 2.0, 0.0, 1.0);
}

struct RectIn {
    // x, y, w, h.
    @location(0) rect: vec4<f32>,
    // Radius, stroke width, layer, unused.
    @location(1) shape: vec4<f32>,
    // Straight alpha.
    @location(2) fill: vec4<f32>,
    @location(3) stroke: vec4<f32>,
}

struct RectOut {
    @builtin(position) position: vec4<f32>,
    // From the rect's centre.
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) half_size: vec2<f32>,
    // Radius, stroke width, this layer's fade.
    @location(2) @interpolate(flat) shape: vec3<f32>,
    @location(3) @interpolate(flat) fill: vec4<f32>,
    @location(4) @interpolate(flat) stroke: vec4<f32>,
}

@vertex
fn rect_vertex(@builtin(vertex_index) index: u32, in: RectIn) -> RectOut {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    // Grown by a pixel so the edge's antialiasing has somewhere to fall.
    let grown = in.rect.xy - 1.0 + corner * (in.rect.zw + 2.0);
    var out: RectOut;
    out.position = clip(grown);
    out.half_size = in.rect.zw * 0.5;
    out.local = grown - in.rect.xy - out.half_size;
    let radius = min(in.shape.x, min(out.half_size.x, out.half_size.y));
    out.shape = vec3<f32>(radius, in.shape.y, view.fades[u32(in.shape.z)]);
    out.fill = encoded(in.fill);
    out.stroke = encoded(in.stroke);
    return out;
}

@fragment
fn rect_fragment(in: RectOut) -> @location(0) vec4<f32> {
    let radius = in.shape.x;
    let stroke_width = in.shape.y;
    let q = abs(in.local) - in.half_size + radius;
    let edge = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    // How much of the pixel lies inside the shape, and inside the hairline's inner edge.
    let inside = clamp(0.5 - edge, 0.0, 1.0);
    let within = clamp(0.5 - (edge + stroke_width), 0.0, 1.0);
    // The hairline lies over the fill, as a border lies over a background: it is composited once,
    // premultiplied, and the shape's own edge then scales the two together, so an edge pixel is
    // the same colour at less coverage.
    let line = select(0.0, (1.0 - within) * in.stroke.a, stroke_width > 0.0);
    let rgb = in.stroke.rgb * line + in.fill.rgb * in.fill.a * (1.0 - line);
    let alpha = line + in.fill.a * (1.0 - line);
    return vec4<f32>(rgb, alpha) * (inside * in.shape.z);
}

struct GlyphIn {
    @location(0) rect: vec4<f32>,
    @location(1) atlas: vec2<f32>,
    @location(2) colour: vec4<f32>,
    @location(3) layer: f32,
}

struct GlyphOut {
    @builtin(position) position: vec4<f32>,
    @location(0) texel: vec2<f32>,
    @location(1) @interpolate(flat) colour: vec4<f32>,
}

@vertex
fn glyph_vertex(@builtin(vertex_index) index: u32, in: GlyphIn) -> GlyphOut {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u));
    var out: GlyphOut;
    out.position = clip(in.rect.xy + corner * in.rect.zw);
    out.texel = in.atlas + corner * in.rect.zw;
    let colour = encoded(in.colour);
    out.colour = vec4<f32>(colour.rgb, colour.a * view.fades[u32(in.layer)]);
    return out;
}

@fragment
fn glyph_fragment(in: GlyphOut) -> @location(0) vec4<f32> {
    let coverage = textureLoad(atlas, vec2<i32>(in.texel), 0).r;
    let alpha = in.colour.a * coverage;
    return vec4<f32>(in.colour.rgb * alpha, alpha);
}
