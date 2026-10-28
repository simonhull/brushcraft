// The bounds pass (render::cull::BoundsPass): for each listed tile, the sphere holding its
// vertices, its normal cone and its longest quad edge, read from its slot.
//
// Appended to tree::NODE_WGSL, upload::VERTEX_WGSL and cull::CULL_WGSL. One workgroup per
// tile; a dispatch covers the tiles of one slot buffer (one buffer bound per dispatch, no
// binding arrays). Triangles are the quads split along (i, j)-(i + 1, j + 1) (plan decision
// 4), counter-clockwise seen from outside, so a triangle's normal is
// cross(p10 - p00, p11 - p00) and cross(p11 - p00, p01 - p00).

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
