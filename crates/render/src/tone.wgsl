// The tone map (render::tone): the scene's linear HDR image into the frame's format, composited
// over the frame's clear colour by its coverage, and, while the camera moves, an edge filter.
// A fullscreen triangle in the frame's render pass; pixels with no coverage are discarded, so the
// clear colour the pass began with shows there.
//
// The image is premultiplied: colour times coverage, and the coverage. It is either the shading
// pass's target (coverage 0 or 1) or accumulation's per-pixel means (render::aa).

// The shaded image (render::shade::HDR_FORMAT), the frame's size.
@group(0) @binding(0) var shaded: texture_2d<f32>;
// Accumulation's means, a pixel's four half floats in two words (render::aa).
@group(0) @binding(1) var<storage, read> means: array<vec2<u32>>;
// The pass's settings.
@group(0) @binding(2) var<uniform> settings: ToneSettings;
// Accumulation's records, four words a pixel, the first the frame its history started (bits
// 0..30): how many samples a mean holds (render::aa).
@group(0) @binding(3) var<storage, read> records: array<vec4<u32>>;

// The pass's settings (render::tone::GpuToneSettings): 48 bytes.
struct ToneSettings {
    // The frame's clear colour, linear; `w` unused.
    clear: vec4<f32>,
    // The image's width and height.
    size: vec2<u32>,
    // SOURCE_MEANS to read `means`, else `shaded`.
    source: u32,
    // 1 to filter edges (the shaded image).
    edge_filter: u32,
    // With SOURCE_MEANS: accumulation's frame, against which the records' starts count samples.
    frame: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// The samples over which a mean's filtering fades out: full at one, none from FADE + 1.
const FADE: f32 = 4.0;
// A record's start frame.
const START: u32 = 0x7fffffffu;

// ToneSettings.source: accumulation's means.
const SOURCE_MEANS: u32 = 1u;

// Whether the frame's format stores what it is given (not an sRGB format, which encodes in
// hardware): then the shader encodes sRGB itself.
override ENCODE_SRGB: bool = false;

// Where the curve leaves the identity.
const KNEE: f32 = 0.8;

// The filter (render::tone, "The edge filter"): a pixel is on an edge if the lumas of it and its
// four neighbours span more than EDGE_RELATIVE of their largest, and at least EDGE_ABSOLUTE.
const EDGE_RELATIVE: f32 = 0.125;
const EDGE_ABSOLUTE: f32 = 0.0312;
// How far, in pixels, the walk along an edge looks for each of its ends.
const SEARCH: i32 = 12;
// The most the sub-pixel blend takes.
const SUBPIXEL: f32 = 0.75;

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

// The image's premultiplied pixel at `p`, clamped to the image.
fn fetch(p: vec2<i32>) -> vec4<f32> {
    let last = vec2<i32>(settings.size) - vec2<i32>(1);
    let q = clamp(p, vec2<i32>(0), last);
    if settings.source == SOURCE_MEANS {
        let w = means[u32(q.y) * settings.size.x + u32(q.x)];
        return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
    }
    return textureLoad(shaded, q, 0);
}

// Premultiplied `c` as shown, linear: its colour tone mapped, over the clear colour by its
// coverage. Exactly tone(c) at full coverage.
fn shown(c: vec4<f32>) -> vec3<f32> {
    let a = clamp(c.a, 0.0, 1.0);
    if a <= 0.0 {
        return settings.clear.rgb;
    }
    return tone(c.rgb / a) * a + settings.clear.rgb * (1.0 - a);
}

// The luma the filter compares: of the colour as shown, square-rooted to be about perceptual.
fn luma(c: vec4<f32>) -> f32 {
    return sqrt(max(dot(shown(c), vec3<f32>(0.299, 0.587, 0.114)), 0.0));
}

// The luma of pixel `p`.
fn luma_at(p: vec2<i32>) -> f32 {
    return luma(fetch(p));
}

// Pixel `p` with the edge filter: unchanged off edges; on one, blended with its neighbour across
// the edge, by how near the pixel lies to the edge's nearer end (a stair step's middle blends
// least) or by how far it stands out from its neighbourhood, whichever is more.
fn filtered(p: vec2<i32>) -> vec4<f32> {
    let m = fetch(p);
    let lm = luma(m);
    let ln = luma_at(p + vec2<i32>(0, -1));
    let ls = luma_at(p + vec2<i32>(0, 1));
    let lw = luma_at(p + vec2<i32>(-1, 0));
    let le = luma_at(p + vec2<i32>(1, 0));
    let lo = min(lm, min(min(ln, ls), min(lw, le)));
    let hi = max(lm, max(max(ln, ls), max(lw, le)));
    let range = hi - lo;
    if range < max(EDGE_ABSOLUTE, hi * EDGE_RELATIVE) {
        return m;
    }
    let lnw = luma_at(p + vec2<i32>(-1, -1));
    let lne = luma_at(p + vec2<i32>(1, -1));
    let lsw = luma_at(p + vec2<i32>(-1, 1));
    let lse = luma_at(p + vec2<i32>(1, 1));
    // Luma bending across rows means an edge running along them.
    let across_rows = 2.0 * abs(ln + ls - 2.0 * lm) + abs(lne + lse - 2.0 * le)
        + abs(lnw + lsw - 2.0 * lw);
    let across_columns = 2.0 * abs(lw + le - 2.0 * lm) + abs(lnw + lne - 2.0 * ln)
        + abs(lsw + lse - 2.0 * ls);
    let horizontal = across_rows >= across_columns;
    let before = select(lw, ln, horizontal);
    let after = select(le, ls, horizontal);
    let toward_before = abs(before - lm) >= abs(after - lm);
    let other = select(after, before, toward_before);
    let sign = select(1, -1, toward_before);
    let step = select(vec2<i32>(sign, 0), vec2<i32>(0, sign), horizontal);
    let along = select(vec2<i32>(0, 1), vec2<i32>(1, 0), horizontal);
    let gradient = abs(other - lm);
    let edge = 0.5 * (lm + other);
    // Walk along the edge both ways until the pair of pixels straddling it stops looking like it.
    var ends = vec2<i32>(SEARCH + 1);
    var deltas = vec2<f32>(0.0);
    for (var side = 0; side < 2; side++) {
        let way = select(along, -along, side == 0);
        for (var k = 1; k <= SEARCH; k++) {
            let q = p + way * k;
            let pair = 0.5 * (luma_at(q) + luma_at(q + step));
            if abs(pair - edge) >= 0.25 * gradient {
                ends[side] = k;
                deltas[side] = pair - edge;
                break;
            }
        }
    }
    let nearer = select(1, 0, ends.x <= ends.y);
    let distance = f32(ends[nearer]);
    let span = f32(ends.x + ends.y) - 1.0;
    // Blend only if the pixel lies on the side of the edge the nearer end turns away from.
    let good = (deltas[nearer] < 0.0) != (lm - edge < 0.0);
    let along_edge = select(0.0, max(0.5 - (distance - 0.5) / span, 0.0), good);
    let average = (2.0 * (ln + ls + lw + le) + lnw + lne + lsw + lse) / 12.0;
    let standout = clamp(abs(average - lm) / range, 0.0, 1.0);
    let smooth_standout = (3.0 - 2.0 * standout) * standout * standout;
    let sub_pixel = smooth_standout * smooth_standout * SUBPIXEL;
    let blend = max(along_edge, sub_pixel);
    return mix(m, fetch(p + step), blend);
}

// How much of the edge filter pixel `p` takes: all of it for the shaded image when asked; for a
// mean, by how few samples it holds (render::tone, "Few samples").
fn filter_weight(p: vec2<i32>) -> f32 {
    if settings.source != SOURCE_MEANS {
        return select(0.0, 1.0, settings.edge_filter != 0u);
    }
    let start = records[u32(p.y) * settings.size.x + u32(p.x)].x & START;
    let samples = settings.frame - start + 1u;
    return clamp(1.0 - f32(samples - 1u) / FADE, 0.0, 1.0);
}

// A pixel of the image, filtered as much as `filter_weight` says, tone mapped over the clear
// colour; nothing where it has no coverage.
@fragment
fn map(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    var c = fetch(p);
    let weight = filter_weight(p);
    if weight > 0.0 {
        c = mix(c, filtered(p), weight);
    }
    if c.a <= 0.0 {
        discard;
    }
    var shade = shown(c);
    if ENCODE_SRGB {
        shade = srgb(shade);
    }
    return vec4<f32>(shade, 1.0);
}
