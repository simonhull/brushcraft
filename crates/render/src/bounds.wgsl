// The bounds pass (render::cull::BoundsPass): for each listed tile, the sphere holding its
// vertices, its normal cone and its longest quad edge, read from its slot.
//
// Appended to tree::NODE_WGSL, upload::VERTEX_WGSL and cull::CULL_WGSL. One workgroup per
// tile; a dispatch covers the tiles of one slot buffer (one buffer bound per dispatch, no
// binding arrays). Triangles are the quads split along (i, j)-(i + 1, j + 1) (plan decision
// 4), counter-clockwise seen from outside, so a triangle's normal is
// cross(p10 - p00, p11 - p00) and cross(p11 - p00, p01 - p00).
//
// The cone also holds the tile's other triangulations (extra_normal): its parent
// triangulation, where the geomorph starts, and every edge variant's fan (see cull.rs).

// The tile tree: every tile's slot.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// The bounds, by node index: written for the listed tiles.
@group(0) @binding(1) var<storage, read_write> bounds: array<TileBounds>;
// The slot buffer the listed tiles' vertices are in.
@group(0) @binding(2) var<storage, read> vertices: array<TileVertex>;
// The node indices of the tiles to bound, for every dispatch of the frame.
@group(0) @binding(3) var<storage, read> nodes: array<u32>;
// This dispatch's run of `nodes`.
@group(0) @binding(4) var<uniform> run: Run;

// A run of node indices: `nodes[first .. first + count]`.
struct Run {
    // The first.
    first: u32,
    // How many: one workgroup each.
    count: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
}

// Threads per workgroup.
const GROUP: u32 = 64u;
// Bits of TileNode.slot below the size class.
const CLASS_SHIFT: u32 = 29u;
// TileNode.slot's index within its buffer.
const SLOT_INDEX: u32 = 0x1fffffu;
// Relative growth of the radius and quad edge, so rounding never makes them too small.
const GROW: f32 = 1e-5;
// Taken off the cone's cosine for the same reason.
const CONE_MARGIN: f32 = 1e-4;

// Per thread: the box's low corner, then the cone axis's sum.
var<workgroup> low: array<vec3<f32>, GROUP>;
// Per thread: the box's high corner.
var<workgroup> high: array<vec3<f32>, GROUP>;
// Per thread: a scalar being reduced (quad edge, radius, cone cosine).
var<workgroup> scalar: array<f32, GROUP>;

// Vertex (i, j) of the tile whose first vertex is `base`, `side` vertices a row.
fn at(base: u32, side: u32, i: u32, j: u32) -> vec3<f32> {
    return vertices[base + j * side + i].position;
}

// The unit normal of triangle (a, b, c), or zero for a degenerate one.
fn face_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let n = cross(b - a, c - a);
    let l = length(n);
    if l > 1e-30 {
        return n / l;
    }
    return vec3<f32>(0.0);
}

// The position of grid vertex `g` of the tile whose first vertex is `base`.
fn at_grid(base: u32, side: u32, g: vec2<i32>) -> vec3<f32> {
    return at(base, side, u32(g.x), u32(g.y));
}

// The unit normal of the triangle on grid vertices g0, g1, g2, taken counter-clockwise in the
// grid (as the tile's own triangles are), or zero for a degenerate one.
fn oriented(base: u32, side: u32, g0: vec2<i32>, g1: vec2<i32>, g2: vec2<i32>) -> vec3<f32> {
    let u = g1 - g0;
    let v = g2 - g0;
    if u.x * v.y - u.y * v.x < 0 {
        return face_normal(at_grid(base, side, g0), at_grid(base, side, g2), at_grid(base, side, g1));
    }
    return face_normal(at_grid(base, side, g0), at_grid(base, side, g1), at_grid(base, side, g2));
}

// How many extra triangles a tile of `quads` quads a side has: two per 2 × 2 block of its
// parent triangulation, three per odd border vertex of each edge's variant.
fn extra_count(quads: u32) -> u32 {
    let half = quads / 2u;
    return 2u * half * half + 12u * half;
}

// Extra triangle `x`'s unit normal (zero if degenerate). First the parent triangulation: each
// 2 × 2-quad block split along its (0, 0)-(2, 2) diagonal. Then, per edge, per odd border
// vertex o between even a and b, with m, ma and mb the vertices one row in from o, a and b:
// the variant's fan (a, b, m), (a, m, ma), (b, mb, m).
fn extra_normal(base: u32, side: u32, x: u32) -> vec3<f32> {
    let quads = side - 1u;
    let half = quads / 2u;
    let blocks = 2u * half * half;
    if x < blocks {
        let block = x / 2u;
        let p00 = vec2<i32>(2 * i32(block % half), 2 * i32(block / half));
        let p22 = p00 + vec2<i32>(2, 2);
        if x % 2u == 0u {
            return oriented(base, side, p00, p00 + vec2<i32>(2, 0), p22);
        }
        return oriented(base, side, p00, p22, p00 + vec2<i32>(0, 2));
    }
    let y = x - blocks;
    let which = y % 3u;
    let o = i32(2u * ((y / 3u) % half) + 1u);
    let edge = y / (3u * half);
    let q = i32(quads);
    var a = vec2<i32>(o - 1, 0);
    var b = vec2<i32>(o + 1, 0);
    var inward = vec2<i32>(0, 1);
    if edge == 1u {
        a = vec2<i32>(q, o - 1);
        b = vec2<i32>(q, o + 1);
        inward = vec2<i32>(-1, 0);
    } else if edge == 2u {
        a = vec2<i32>(o - 1, q);
        b = vec2<i32>(o + 1, q);
        inward = vec2<i32>(0, -1);
    } else if edge == 3u {
        a = vec2<i32>(0, o - 1);
        b = vec2<i32>(0, o + 1);
        inward = vec2<i32>(1, 0);
    }
    let m = (a + b) / 2 + inward;
    if which == 0u {
        return oriented(base, side, a, b, m);
    }
    if which == 1u {
        return oriented(base, side, a, m, a + inward);
    }
    return oriented(base, side, b, b + inward, m);
}

// Reduces `scalar` to its maximum (`take_max`) or minimum in `scalar[0]`.
fn reduce_scalar(t: u32, take_max: bool) {
    for (var s = GROUP / 2u; s > 0u; s >>= 1u) {
        if t < s {
            let other = scalar[t + s];
            scalar[t] = select(min(scalar[t], other), max(scalar[t], other), take_max);
        }
        workgroupBarrier();
    }
}

// Bounds one listed tile per workgroup.
@compute @workgroup_size(64)
fn bound_tiles(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let index = nodes[run.first + group.x];
    let node = tree[index];
    let size_class = node.slot >> CLASS_SHIFT;
    let side = (1u << size_class) + 1u;
    let count = side * side;
    let base = (node.slot & SLOT_INDEX) * count;
    let quads = side - 1u;

    // The box, and the longest quad edge.
    var lo = vec3<f32>(3.4e38);
    var hi = vec3<f32>(-3.4e38);
    var edge = 0.0;
    for (var v = t; v < count; v += GROUP) {
        let i = v % side;
        let j = v / side;
        let p = at(base, side, i, j);
        lo = min(lo, p);
        hi = max(hi, p);
        if i < quads {
            edge = max(edge, distance(p, at(base, side, i + 1u, j)));
        }
        if j < quads {
            edge = max(edge, distance(p, at(base, side, i, j + 1u)));
        }
    }
    low[t] = lo;
    high[t] = hi;
    scalar[t] = edge;
    workgroupBarrier();
    for (var s = GROUP / 2u; s > 0u; s >>= 1u) {
        if t < s {
            low[t] = min(low[t], low[t + s]);
            high[t] = max(high[t], high[t + s]);
        }
        workgroupBarrier();
    }
    reduce_scalar(t, true);
    let center = (low[0] + high[0]) * 0.5;
    let quad_edge = scalar[0];
    workgroupBarrier();

    // The radius, and the sum of the triangles' unit normals: the cone's axis.
    var radius = 0.0;
    for (var v = t; v < count; v += GROUP) {
        radius = max(radius, distance(at(base, side, v % side, v / side), center));
    }
    var sum = vec3<f32>(0.0);
    for (var q = t; q < quads * quads; q += GROUP) {
        let i = q % quads;
        let j = q / quads;
        let p00 = at(base, side, i, j);
        let p10 = at(base, side, i + 1u, j);
        let p11 = at(base, side, i + 1u, j + 1u);
        let p01 = at(base, side, i, j + 1u);
        sum += face_normal(p00, p10, p11) + face_normal(p00, p11, p01);
    }
    let extras = extra_count(quads);
    for (var x = t; x < extras; x += GROUP) {
        sum += extra_normal(base, side, x);
    }
    scalar[t] = radius;
    low[t] = sum;
    workgroupBarrier();
    for (var s = GROUP / 2u; s > 0u; s >>= 1u) {
        if t < s {
            low[t] = low[t] + low[t + s];
        }
        workgroupBarrier();
    }
    reduce_scalar(t, true);
    let sphere = scalar[0];
    let total = low[0];
    workgroupBarrier();

    // The cone's half-angle: the widest triangle normal from the axis.
    let length_sum = length(total);
    var axis = vec3<f32>(0.0);
    if length_sum > 1e-20 {
        axis = total / length_sum;
    }
    var cosine = 1.0;
    for (var q = t; q < quads * quads; q += GROUP) {
        let i = q % quads;
        let j = q / quads;
        let p00 = at(base, side, i, j);
        let p10 = at(base, side, i + 1u, j);
        let p11 = at(base, side, i + 1u, j + 1u);
        let p01 = at(base, side, i, j + 1u);
        let a = face_normal(p00, p10, p11);
        let b = face_normal(p00, p11, p01);
        if dot(a, a) > 0.0 {
            cosine = min(cosine, dot(a, axis));
        }
        if dot(b, b) > 0.0 {
            cosine = min(cosine, dot(b, axis));
        }
    }
    for (var x = t; x < extras; x += GROUP) {
        let n = extra_normal(base, side, x);
        if dot(n, n) > 0.0 {
            cosine = min(cosine, dot(n, axis));
        }
    }
    scalar[t] = cosine;
    workgroupBarrier();
    reduce_scalar(t, false);

    if t == 0u {
        var cone = -1.0;
        if length_sum > 1e-20 {
            cone = scalar[0] - CONE_MARGIN;
        }
        bounds[index] = TileBounds(
            center,
            sphere * (1.0 + GROW),
            axis,
            cone,
            quad_edge * (1.0 + GROW),
            0u,
            0u,
            0u,
        );
    }
}
