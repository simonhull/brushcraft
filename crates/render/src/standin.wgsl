// Stand-ins (render::standin): a stroke seen from a distance, shown in the frame of its step.
//
// The passes' shared declarations; each pass's own source follows them (standin_replace.wgsl,
// then standin_normals.wgsl in the same module; standin_settle.wgsl), after upload::VERTEX_WGSL, tree::NODE_WGSL and the
// constants render::standin prefixes: TILE_WORDS (a finest tile's words in the scratch), PENDING,
// EVERYTHING (the cone that covers everything, as its bits), SLACK, MAGNITUDE_SLACK, FLOOR and
// F32_MAX_BITS (see render::standin, "Rounding").
//
// `replace` runs one workgroup per job: a drawn display tile and one edited finest tile under
// it. Every vertex of the display tile that the finest tile holds takes the finest vertex's
// exact bits, read from the scratch the frame staged. A measured job (see render::standin, "The
// bound") also measures how far every finest vertex of its finest tile lies from the stand-in's
// triangles, and keeps in the job's row the largest, and the largest distance a replaced vertex
// beside an unmeasured quad moved; any other job keeps the largest distance any replaced vertex
// moved. `normals` then runs over the same jobs, once every replace is done: each replaced
// vertex's normal recomputed from the positions (see render::standin, "Normals"). `settle` then
// runs one thread per row: a row that a step made pending sets its tile's bound, from what was
// measured if anything was, else from the formula (see render::standin, "The bound"), and every
// live row writes its bound and the cone that covers everything into its node, every frame, over
// whatever the frame's tree upload wrote.

// One display tile and one finest tile under it (render::standin::GpuJob): 64 bytes.
struct Job {
    // The display tile's first vertex in its slot buffer.
    base: u32,
    // The display tile's vertices per side (bits 0..16); which of the finest tiles around the
    // job's, on its face, the stand-in reads (bits 16..25: the tile at offset (ox, oy), each
    // -1 to 1, at bit 16 + (oy + 1) * 3 + ox + 1, its own always set); and MEASURE (bit 31).
    side: u32,
    // Its row.
    row: u32,
    // The finest tile's place in the scratch, in tiles.
    scratch: u32,
    // The first display vertex the finest tile holds, (i, j) in the display tile: i in bits
    // 0..16, j in 16..32.
    lo: u32,
    // How many display vertices it holds along i (bits 0..16) and j (16..32).
    count: u32,
    // The first one's finest vertex, (i, j) in the finest tile, packed as `lo`.
    fine: u32,
    // Finest vertices between display vertices (bits 0..16), and the finest tile's vertices
    // per side (16..32).
    step: u32,
    // The finest tiles across the finest tile's edges, on its face or another, by their place in
    // the scratch, NONE where the scratch doesn't hold one: across j = 0, i = side - 1,
    // j = side - 1 and i = 0, in that order (tree::NODE_WGSL's neighbours).
    links: array<u32, 4>,
    // Per link, where a point one step across the edge lies in the tile across (whose grid may
    // turn): the point at s along the edge (i on edges 0 and 2, j on 1 and 3) is
    // origin + s * along + inward there, origin's i in bits 0..8 and j in 8..16, along in
    // 16..18 and inward in 18..20 as grid steps (+i, +j, -i, -j as 0 to 3).
    maps: array<u32, 4>,
}

// One stand-in's row (render::standin::GpuRow): 32 bytes.
struct Row {
    // Its tile's node, or NONE for a free row.
    node: u32,
    // Its tile's face and place, as the node holds them: a row whose node holds another tile
    // writes nothing.
    face: u32,
    place: u32,
    // The largest bound on a measured finest vertex's distance from the stand-in's triangles this
    // frame, f32 bits (non-negative: ordered as u32s); 0 if no job of the row measured.
    measured: atomic<u32>,
    // PENDING: a step was replaced in this frame, its bound to be raised from `error` (a new
    // row's, the tile's stored bound).
    flags: u32,
    // The step's displacement bound, f32 bits: how far a finest vertex under the tile moved
    // since the data its bound before is of.
    displacement: u32,
    // The tile's bound, f32 bits, once settled.
    error: u32,
    // The largest bound on the distance a replaced vertex moved this frame, f32 bits: in a
    // measured job, of those beside a quad no job measures alone.
    moved: atomic<u32>,
}

// The display tile's vertices per side, from a job's `side`.
fn job_side(job: Job) -> u32 {
    return job.side & 0xffffu;
}

// The largest magnitude of `p`'s coordinates.
fn largest(p: vec3<f32>) -> f32 {
    let a = abs(p);
    return max(a.x, max(a.y, a.z));
}

// A bound, as f32 bits at most F32_MAX_BITS, on the length of a vector whose components were
// computed as `d` from points whose coordinates are at most `magnitude` in size, with at most
// 13 ULPs of `magnitude` of rounding each (see render::standin, "Rounding").
fn bound_bits(d: vec3<f32>, magnitude: f32) -> u32 {
    let raised = max((length(d) + magnitude * MAGNITUDE_SLACK) * SLACK, FLOOR);
    return min(bitcast<u32>(raised), F32_MAX_BITS);
}
