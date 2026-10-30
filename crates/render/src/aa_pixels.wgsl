// Accumulation's pixel pass (render::aa): each pixel's reset rule, then the frame's sample into
// its running mean. After aa::PARAMS_WGSL; one invocation a pixel, in 8 × 8 groups.
//
// A pixel's history (render::aa, "A pixel's history"): the frame it started in, whether it saw
// more tiles than it can name, the tiles it saw (up to three), and the mean of its samples, the
// shading's premultiplied colour and coverage, half floats.

// This frame's visibility buffer, drawn at its jitter.
@group(0) @binding(0) var ids: texture_2d<u32>;
// This frame's shaded image: premultiplied linear colour, alpha the coverage (0 or 1).
@group(0) @binding(1) var shaded: texture_2d<f32>;
// Per node, the latest frame its drawing changed (aa.wgsl's spread).
@group(0) @binding(2) var<storage, read> changed: array<u32>;
// The latest frame any node's drawing changed, in word 0.
@group(0) @binding(3) var<storage, read> latest: array<u32>;
// Per pixel, the mean: red and green, blue and coverage, each pair two half floats.
@group(0) @binding(4) var<storage, read_write> colour: array<vec2<u32>>;
// Per pixel, three words: the start frame (bits 0..31) and OVERFLOW; the first tile (bits
// 0..20) and the third's low 12 bits; the second tile (bits 0..20) and the third's high 8 bits.
@group(0) @binding(5) var<storage, read_write> record: array<u32>;
// The frame's settings.
@group(0) @binding(6) var<uniform> params: AaParams;

// No tile: an empty name in a record, and the node bits of the background's id.
const EMPTY: u32 = 0xfffffu;
// Where a visibility id's node starts (render::raster::VisibilityId).
const NODE_SHIFT: u32 = 12u;
// A record's first word: the pixel saw more tiles than its record names.
const OVERFLOW: u32 = 0x80000000u;

// Whether node `n`'s drawing changed after frame `start`.
fn changed_after(n: u32, start: u32) -> bool {
    return n < arrayLength(&changed) && changed[n] > start;
}

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y {
        return;
    }
    let i = id.y * params.size.x + id.x;
    let node = textureLoad(ids, id.xy, 0).x >> NODE_SHIFT;
    var start = params.frame;
    var overflow = false;
    var seen = array<u32, 3>(EMPTY, EMPTY, EMPTY);
    var mean = vec4<f32>(0.0);
    if (params.flags & RESTART) == 0u {
        let w0 = record[3u * i];
        let w1 = record[3u * i + 1u];
        let w2 = record[3u * i + 2u];
        let since = w0 & ~OVERFLOW;
        let old = array<u32, 3>(w1 & EMPTY, w2 & EMPTY, (w1 >> 20u) | ((w2 >> 20u) << 12u));
        let crowded = (w0 & OVERFLOW) != 0u;
        // The reset rule: a tile it saw changed, the tile this sample sees changed (it may now
        // cover what the history saw), or, past its record, any tile changed.
        var reset = changed_after(node, since) || (crowded && latest[0] > since);
        for (var k = 0u; k < 3u; k++) {
            reset = reset || (old[k] != EMPTY && changed_after(old[k], since));
        }
        if !reset {
            start = since;
            overflow = crowded;
            seen = old;
            let c = colour[i];
            mean = vec4<f32>(unpack2x16float(c.x), unpack2x16float(c.y));
        }
    }
    let n = params.frame - start + 1u;
    if n > params.samples {
        // Converged: the history holds every jitter.
        return;
    }
    mean += (textureLoad(shaded, id.xy, 0) - mean) / f32(n);
    if node != EMPTY && seen[0] != node && seen[1] != node && seen[2] != node {
        if seen[0] == EMPTY {
            seen[0] = node;
        } else if seen[1] == EMPTY {
            seen[1] = node;
        } else if seen[2] == EMPTY {
            seen[2] = node;
        } else {
            overflow = true;
        }
    }
    colour[i] = vec2<u32>(pack2x16float(mean.xy), pack2x16float(mean.zw));
    record[3u * i] = start | select(0u, OVERFLOW, overflow);
    record[3u * i + 1u] = seen[0] | ((seen[2] & 0xfffu) << 20u);
    record[3u * i + 2u] = seen[1] | ((seen[2] >> 12u) << 20u);
}
