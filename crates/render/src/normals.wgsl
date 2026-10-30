// The normal pass (render::normals): each finest tile's vertices, positions and unit normals,
// computed from the tile's positions with its one-vertex ring and the rings of the base
// vertices at its corners, and written into the tile's slot.
//
// Appended to upload::VERTEX_WGSL. A dispatch covers the jobs of one record chunk and one slot
// buffer: workgroup (x, y) is the x-th run of GROUP vertices of job y's tile. The record layout
// is normals.rs's (`RingRecord`).
//
// The arithmetic is the CPU's (geom's GatheredBlock::normals, fixtures::ring_normals), written
// out the same way: differences, cross products component by component, a ring summed from zero
// in ring order, the length's squares summed left to right, the reciprocal once and three
// products. Every vertex goes through the same code with its stencil as data, so two tiles
// holding a vertex compute it with the same instructions from the same bits, whatever the
// compiler fuses: a seam's two sides agree bit for bit.

// The records of this dispatch's chunk: words, positions as f32 bits.
@group(0) @binding(0) var<storage, read> records: array<u32>;
// The slot buffer the jobs' tiles are written into.
@group(0) @binding(1) var<storage, read_write> vertices: array<TileVertex>;
// This dispatch's jobs, one per workgroup row.
@group(0) @binding(2) var<storage, read> jobs: array<Job>;

// One tile to compute: where its record is and where its vertices go.
struct Job {
    // The record's first word in `records`.
    record: u32,
    // The tile's first vertex in `vertices`.
    base: u32,
}

// Threads per workgroup.
const GROUP: u32 = 64u;
// Record header: quads per side.
const HEADER_QUADS: u32 = 0u;
// Record header: the stencils, block corners 0 to 3 then tile edges 0 to 3.
const HEADER_STENCILS: u32 = 1u;
// Record header: per block corner, its ring's first edge point, in words from the record's start.
const HEADER_RINGS: u32 = 9u;
// Record header: per block corner, its ring's valence (0 where the corner isn't a base vertex).
const HEADER_VALENCES: u32 = 13u;
// Words before the positions.
const HEADER_WORDS: u32 = 20u;
// A stencil reading the ring of block corner `stencil & 3`.
const STENCIL_CORNER: u32 = 0x80000000u;
// A stencil off the face: NaN.
const STENCIL_OUTSIDE: u32 = 0x40000000u;
// The block's own axes, u = (1, 0) and v = (0, 1), each step stored plus one in two bits.
const STENCIL_OWN: u32 = 150u;
// The smallest normal f32: lengths at or below it have no direction.
const MIN_POSITIVE: f32 = 1.17549435e-38;
// A quiet NaN's bits.
const NAN_BITS: u32 = 0x7fc00000u;
// The exponent bits of an f32: all set for an infinity or a NaN.
const EXPONENT: u32 = 0x7f800000u;

// NaN in every component. `zero` is a runtime zero, so no constant NaN is ever formed (WGSL
// rejects one).
fn nan3(zero: u32) -> vec3<f32> {
    let nan = bitcast<f32>(NAN_BITS | zero);
    return vec3<f32>(nan, nan, nan);
}

// Whether `x` is neither infinite nor NaN, from its bits.
fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & EXPONENT) != EXPONENT;
}

// Whether every component of `v` is finite.
fn finite3(v: vec3<f32>) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z);
}

// The f32 at word `at` of `records`.
fn word(at: u32) -> f32 {
    return bitcast<f32>(records[at]);
}

// The point whose x is at word `at` of `records`.
fn point_at(at: u32) -> vec3<f32> {
    return vec3<f32>(word(at), word(at + 1u), word(at + 2u));
}

// Block vertex `g` (each coordinate -1 ..= quads + 1) of the record at `record`, whose padded
// rows hold `padded` points.
fn position(record: u32, padded: u32, g: vec2<i32>) -> vec3<f32> {
    let index = u32(g.y + 1) * padded + u32(g.x + 1);
    return point_at(record + HEADER_WORDS + 3u * index);
}

// `a - b`, component by component.
fn minus(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.x - b.x, a.y - b.y, a.z - b.z);
}

// `a × b`, component by component, as the CPU writes it.
fn cross_of(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// `n` at unit length where its length is finite and above MIN_POSITIVE; else (0, 0, 1) where
// the inputs were `inputs_finite`, else NaN.
fn unit_or_fallback(n: vec3<f32>, inputs_finite: bool, zero: u32) -> vec3<f32> {
    let length = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if finite(length) && length > MIN_POSITIVE {
        let r = 1.0 / length;
        return vec3<f32>(n.x * r, n.y * r, n.z * r);
    }
    if inputs_finite {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return nan3(zero);
}

// The normal at the base vertex `v` at block corner `c` of the record at `record`, from its
// ring: the sum over k of (e_k - v) × (e_(k+1) - v), from the owner round the ring, wrapping,
// at unit length. NaN without a ring, or for a ring that isn't finite.
fn corner_normal(record: u32, c: u32, v: vec3<f32>, zero: u32) -> vec3<f32> {
    let valence = records[record + HEADER_VALENCES + c];
    if valence == 0u {
        return nan3(zero);
    }
    let first = record + records[record + HEADER_RINGS + c];
    var all_finite = finite3(v);
    for (var k = 0u; k < valence; k++) {
        all_finite = all_finite && finite3(point_at(first + 3u * k));
    }
    if !all_finite {
        return nan3(zero);
    }
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < valence; k++) {
        let a = point_at(first + 3u * k);
        let b = point_at(first + 3u * ((k + 1u) % valence));
        let t = cross_of(minus(a, v), minus(b, v));
        sum = vec3<f32>(sum.x + t.x, sum.y + t.y, sum.z + t.z);
    }
    return unit_or_fallback(sum, true, zero);
}

// The stencil of block vertex (i, j) of a tile of `q` quads a side: a block corner's own, a
// tile edge's (every vertex strictly inside an edge has the same), or the block's own axes
// inside the tile.
fn stencil_at(record: u32, q: u32, i: u32, j: u32) -> u32 {
    let low_i = i == 0u;
    let high_i = i == q;
    let low_j = j == 0u;
    let high_j = j == q;
    if (low_i || high_i) && (low_j || high_j) {
        // Corners (0, 0), (q, 0), (q, q), (0, q): 0, 1, 2, 3.
        var c = 0u;
        if high_i && low_j {
            c = 1u;
        } else if high_i && high_j {
            c = 2u;
        } else if low_i && high_j {
            c = 3u;
        }
        return records[record + HEADER_STENCILS + c];
    }
    // Edges j = 0, i = q, j = q, i = 0: 0, 1, 2, 3.
    if low_j {
        return records[record + HEADER_STENCILS + 4u];
    }
    if high_i {
        return records[record + HEADER_STENCILS + 5u];
    }
    if high_j {
        return records[record + HEADER_STENCILS + 6u];
    }
    if low_i {
        return records[record + HEADER_STENCILS + 7u];
    }
    return STENCIL_OWN;
}

// The grid step stored in two bits of a stencil from bit `shift`, plus one.
fn step(stencil: u32, shift: u32) -> i32 {
    return i32((stencil >> shift) & 3u) - 1;
}

// One vertex of one job's tile: its position from the record, its normal from its stencil.
@compute @workgroup_size(64)
fn finest_normals(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let job = jobs[group.y];
    let record = job.record;
    let q = records[record + HEADER_QUADS];
    let side = q + 1u;
    let index = group.x * GROUP + lane;
    if index >= side * side {
        return;
    }
    let i = index % side;
    let j = index / side;
    let padded = q + 3u;
    // A runtime zero: the tile's quads, masked off.
    let zero = q & 0u;
    let g = vec2<i32>(i32(i), i32(j));
    let here = position(record, padded, g);
    let stencil = stencil_at(record, q, i, j);
    var normal: vec3<f32>;
    if (stencil & STENCIL_CORNER) != 0u {
        normal = corner_normal(record, stencil & 3u, here, zero);
    } else if (stencil & STENCIL_OUTSIDE) != 0u {
        normal = nan3(zero);
    } else {
        let u = vec2<i32>(step(stencil, 0u), step(stencil, 2u));
        let v = vec2<i32>(step(stencil, 4u), step(stencil, 6u));
        let du = minus(position(record, padded, g + u), position(record, padded, g - u));
        let dv = minus(position(record, padded, g + v), position(record, padded, g - v));
        normal = unit_or_fallback(cross_of(du, dv), finite3(du) && finite3(dv), zero);
    }
    vertices[job.base + index] = TileVertex(here, 1.0, normal, 0u);
}
