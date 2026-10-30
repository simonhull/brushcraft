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
// triangulation, where the geomorph starts, every edge variant's fan, and each of the tile's
// own and fan triangles with some of its odd vertices where their morph starts and the rest
// where it ends (morph_normal), so every morph's triangles too (see cull.rs).

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
// parent triangulation, three per odd border vertex of each edge's variant, then the morph's
// (morph_count).
fn extra_count(quads: u32) -> u32 {
    let half = quads / 2u;
    return 2u * half * half + 12u * half + morph_count(quads);
}

// How many morph triangles a tile of `quads` quads a side has (none for a single quad, which
// never morphs): each of its own triangles, and each variant fan triangle with moving vertices
// ((a, m, ma) and (b, mb, m) per odd border vertex of each edge), six times: once per mix of
// its three vertices at their start and at their end but all at one end.
fn morph_count(quads: u32) -> u32 {
    if quads < 2u {
        return 0u;
    }
    return 12u * quads * quads + 48u * (quads / 2u);
}

// Grid vertex g where its morph starts, the midpoint of its decision-4 parents, if `start`;
// else where it is.
fn morph_point(base: u32, side: u32, g: vec2<i32>, start: bool) -> vec3<f32> {
    if !start {
        return at_grid(base, side, g);
    }
    let d = vec2<i32>(g.x & 1, g.y & 1);
    return (at_grid(base, side, g - d) + at_grid(base, side, g + d)) * 0.5;
}


// The unit normal of triangle (g0, g1, g2), counter-clockwise in the grid, with vertex k where
// its morph starts for each bit k of `starts` and where it is otherwise (an even vertex is
// where it is either way).
fn morph_oriented(base: u32, side: u32, g0: vec2<i32>, g1: vec2<i32>, g2: vec2<i32>, starts: u32) -> vec3<f32> {
    let p = array<vec3<f32>, 3>(
        morph_point(base, side, g0, (starts & 1u) != 0u),
        morph_point(base, side, g1, (starts & 2u) != 0u),
        morph_point(base, side, g2, (starts & 4u) != 0u),
    );
    let u = g1 - g0;
    let v = g2 - g0;
    if u.x * v.y - u.y * v.x < 0 {
        return face_normal(p[0], p[2], p[1]);
    }
    return face_normal(p[0], p[1], p[2]);
}

// Morph triangle `x`'s unit normal (zero if degenerate). Each of these triangles has two or
// three odd vertices, each with its own morph factor; with all at their start it lies in the
// parent triangulation's plane, with all at their end it is the tile's own (or fan) triangle,
// and here it is taken with some at each end (`starts`, 1 to 6). A triangle whose moving
// vertices stand anywhere on their ways has, its normal being linear in each vertex, a normal
// that is a sum, with weights not negative, of its eight corner normals: in the cone, which is
// convex, so the cone holds every morph state. First each own triangle (quad by quad, its two
// halves), then per edge, per odd border vertex, the fan triangles (a, m, ma) and (b, mb, m).
fn morph_normal(base: u32, side: u32, x: u32) -> vec3<f32> {
    let quads = side - 1u;
    let own = 12u * quads * quads;
    let starts = x % 6u + 1u;
    if x < own {
        let quad = x / 12u;
        let p00 = vec2<i32>(i32(quad % quads), i32(quad / quads));
        let p11 = p00 + vec2<i32>(1, 1);
        if (x / 6u) % 2u == 0u {
            return morph_oriented(base, side, p00, p00 + vec2<i32>(1, 0), p11, starts);
        }
        return morph_oriented(base, side, p00, p11, p00 + vec2<i32>(0, 1), starts);
    }
    let y = x - own;
    let half = quads / 2u;
    let o = i32(2u * ((y / 12u) % half) + 1u);
    let edge = y / (12u * half);
    let fan = fan_of(edge, o, i32(quads));
    let a = fan[0];
    let b = fan[1];
    let inward = fan[2];
    let m = (a + b) / 2 + inward;
    if (y / 6u) % 2u == 0u {
        return morph_oriented(base, side, a, m, a + inward, starts);
    }
    return morph_oriented(base, side, b, b + inward, m, starts);
}

// The odd border vertex o of edge `edge`'s even neighbours a and b, and the step inward from
// the edge, of a tile of `q` quads a side.
fn fan_of(edge: u32, o: i32, q: i32) -> array<vec2<i32>, 3> {
    if edge == 1u {
        return array<vec2<i32>, 3>(vec2<i32>(q, o - 1), vec2<i32>(q, o + 1), vec2<i32>(-1, 0));
    }
    if edge == 2u {
        return array<vec2<i32>, 3>(vec2<i32>(o - 1, q), vec2<i32>(o + 1, q), vec2<i32>(0, -1));
    }
    if edge == 3u {
        return array<vec2<i32>, 3>(vec2<i32>(0, o - 1), vec2<i32>(0, o + 1), vec2<i32>(1, 0));
    }
    return array<vec2<i32>, 3>(vec2<i32>(o - 1, 0), vec2<i32>(o + 1, 0), vec2<i32>(0, 1));
}

// Extra triangle `x`'s unit normal (zero if degenerate). First the parent triangulation: each
// 2 × 2-quad block split along its (0, 0)-(2, 2) diagonal. Then, per edge, per odd border
// vertex o between even a and b, with m, ma and mb the vertices one row in from o, a and b:
// the variant's fan (a, b, m), (a, m, ma), (b, mb, m).
fn extra_normal(base: u32, side: u32, x: u32) -> vec3<f32> {
    let quads = side - 1u;
    let half = quads / 2u;
    let blocks = 2u * half * half;
    if x >= blocks + 12u * half {
        return morph_normal(base, side, x - blocks - 12u * half);
    }
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
