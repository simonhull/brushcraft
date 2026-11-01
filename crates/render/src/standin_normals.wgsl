// The stand-in pass's `normals` (render::standin, "Normals"): after standin.wgsl and
// standin_replace.wgsl, in the same module, at the replace pass's bindings (it reads no row). One
// workgroup per job, dispatched after every replace of the frame, so every position it reads is
// the frame's.
//
// Each display vertex the job's finest tile holds takes a normal recomputed from positions, as
// the display layer would store it:
// - inside the display tile, from central differences over its own grid (the slot's positions,
//   replaced or stored: what is drawn);
// - on its border, from central differences over the finest grid (the vertex's four finest
//   neighbours), read from the scratch: the job's finest tile, or the one across its edge (the
//   job's links and their maps, across a face's edge too). Every tile holding the vertex, at any
//   level, reads the same four points, so both sides of a seam compute the same normal. A
//   neighbour the scratch doesn't hold (a finest tile no stand-in of the frame reads) is
//   replaced by the vertex itself: a one-sided difference.
// The normal then goes through the display encoding (octahedral, two signed 12-bit fractions)
// and back, so a stand-in's normal is one the display layer could hold. A vertex whose
// differences give no direction keeps its normal.

// The display layer's normal quantization: a signed 12-bit fraction's largest magnitude,
// 2^11 - 1 (geom's levels(12)).
const OCTAHEDRAL_LEVELS: f32 = 2047.0;

// 1 for zero and up, -1 below: the octahedral fold's sign (sign() gives 0 at 0).
fn fold_sign(v: f32) -> f32 {
    return select(-1.0, 1.0, v >= 0.0);
}

// The octahedral fold of `p` (the lower half of the octahedron unfolded onto the square's
// corners): the same map both ways.
fn fold(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((1.0 - abs(p.y)) * fold_sign(p.x), (1.0 - abs(p.x)) * fold_sign(p.y));
}

// Unit vector `n` as the display layer stores it (geom's octahedral, quantize_unit at 12 bits,
// from_octahedral) and decodes it: within about 0.03 degrees of `n`. WGSL's round() takes a half
// to even where the CPU's takes it away from zero: a quantization step's choice at a tie, either
// one a normal the layer can hold.
fn display_encoded(n: vec3<f32>) -> vec3<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    var square = n.xy / l1;
    if n.z < 0.0 {
        square = fold(square);
    }
    let q = round(clamp(square, vec2<f32>(-1.0), vec2<f32>(1.0)) * OCTAHEDRAL_LEVELS)
        / OCTAHEDRAL_LEVELS;
    let height = 1.0 - abs(q.x) - abs(q.y);
    var xy = q;
    if height < 0.0 {
        xy = fold(q);
    }
    return normalize(vec3<f32>(xy, height));
}

// Grid step `d` of a link's map: +i, +j, -i, -j as 0 to 3.
fn grid_step(d: u32) -> vec2<i32> {
    let sign = 1 - 2 * i32(d >> 1u);
    return select(vec2<i32>(0, sign), vec2<i32>(sign, 0), (d & 1u) == 0u);
}

// Finest vertex (x, y) of job `g`'s finest tile, one coordinate in the tile and the other at
// most one step past it (-1 ..= quads + 1), with w = 1; w = 0 where it lies across an edge whose
// finest tile the scratch doesn't hold. Across an edge it is read in the tile there, through the
// link's map (that tile's grid may turn, across a face's edge).
fn finest_point(g: u32, x: i32, y: i32) -> vec4<f32> {
    let fine_side = i32(jobs[g].step >> 16u);
    let quads = fine_side - 1;
    var tile = jobs[g].scratch;
    var fx = x;
    var fy = y;
    var edge = 4u;
    if y < 0 {
        edge = 0u;
    } else if x > quads {
        edge = 1u;
    } else if y > quads {
        edge = 2u;
    } else if x < 0 {
        edge = 3u;
    }
    if edge < 4u {
        // From the job list, not a local copy: indexing a local copy's array at run time is
        // what the crate's notes ("Shaders") rule out.
        tile = jobs[g].links[edge];
        if tile == NONE {
            return vec4<f32>(0.0);
        }
        let map = jobs[g].maps[edge];
        let s = select(y, x, edge == 0u || edge == 2u);
        let origin = vec2<i32>(i32(map & 0xffu), i32((map >> 8u) & 0xffu));
        let p = origin + s * grid_step((map >> 16u) & 3u) + grid_step((map >> 18u) & 3u);
        fx = p.x;
        fy = p.y;
    }
    let w = tile * TILE_WORDS + u32(fy * fine_side + fx) * 3u;
    return vec4<f32>(scratch_point(w), 1.0);
}

// `p`'s point where the scratch holds it, else `centre`: a missing neighbour makes the
// difference one-sided.
fn or_centre(p: vec4<f32>, centre: vec3<f32>) -> vec3<f32> {
    return select(centre, p.xyz, p.w > 0.0);
}

// The position of display vertex (i, j) of a tile `side` vertices a row, from `base`.
fn slot_position(base: u32, side: u32, i: u32, j: u32) -> vec3<f32> {
    return slots[base + j * side + i].position;
}

// Recomputes the normals of the display vertices job `group.x`'s finest tile holds.
//
// Two jobs of one tile can share a vertex (on the border between their finest tiles): both
// compute it from the same points and write the same bits.
@compute @workgroup_size(64)
fn normals(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let g = group.x;
    let job = jobs[g];
    let nx = job.count & 0xffffu;
    let ny = job.count >> 16u;
    let lx = job.lo & 0xffffu;
    let ly = job.lo >> 16u;
    let fi = job.fine & 0xffffu;
    let fj = job.fine >> 16u;
    let step = job.step & 0xffffu;
    let side = job_side(job);
    let quads = side - 1u;
    for (var t = lane; t < nx * ny; t += GROUP) {
        let a = t % nx;
        let b = t / nx;
        let i = lx + a;
        let j = ly + b;
        let at = job.base + j * side + i;
        var du: vec3<f32>;
        var dv: vec3<f32>;
        if i > 0u && i < quads && j > 0u && j < quads {
            du = slot_position(job.base, side, i + 1u, j)
                - slot_position(job.base, side, i - 1u, j);
            dv = slot_position(job.base, side, i, j + 1u)
                - slot_position(job.base, side, i, j - 1u);
        } else {
            let x = i32(fi + a * step);
            let y = i32(fj + b * step);
            let centre = finest_point(g, x, y).xyz;
            du = or_centre(finest_point(g, x + 1, y), centre)
                - or_centre(finest_point(g, x - 1, y), centre);
            dv = or_centre(finest_point(g, x, y + 1), centre)
                - or_centre(finest_point(g, x, y - 1), centre);
        }
        let n = cross(du, dv);
        let length2 = dot(n, n);
        // No direction (or not finite): the stored normal stays.
        if length2 > 0.0 && length2 < 3.0e38 {
            slots[at].normal = display_encoded(n * inverseSqrt(length2));
        }
    }
}
