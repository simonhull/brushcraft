// Stand-ins (render::standin): a stroke seen from a distance, shown in the frame of its step.
//
// The two passes' shared declarations; each pass's own source follows them
// (standin_replace.wgsl, standin_settle.wgsl), after upload::VERTEX_WGSL, tree::NODE_WGSL and the
// constants render::standin prefixes: TILE_WORDS (a finest tile's words in the scratch), PENDING,
// EVERYTHING (the cone that covers everything, as its bits) and SLACK.
//
// `replace` runs one workgroup per job: a drawn display tile and one edited finest tile under
// it. Every vertex of the display tile that the finest tile holds takes the finest vertex's
// exact bits, read from the scratch the frame staged; the job's row keeps the largest distance
// a vertex moved. `settle` then runs one thread per row: a row that a step made pending sets
// its tile's bound to the bound before plus the step's displacement plus the larger of the two
// (see render::standin, "The bound"), and every live row writes its bound and the cone that
// covers everything into its node, every frame, over whatever the frame's tree upload wrote.

// One display tile and one finest tile under it (render::standin::GpuJob): 32 bytes.
struct Job {
    // The display tile's first vertex in its slot buffer.
    base: u32,
    // The display tile's vertices per side.
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
}

// One stand-in's row (render::standin::GpuRow): 32 bytes.
struct Row {
    // Its tile's node, or NONE for a free row.
    node: u32,
    // Its tile's face and place, as the node holds them: a row whose node holds another tile
    // writes nothing.
    face: u32,
    place: u32,
    // Zero.
    reserved: u32,
    // PENDING: a step was replaced in this frame, its bound to be raised from `error` (a new
    // row's, the tile's stored bound).
    flags: u32,
    // The step's displacement bound, f32 bits: how far a finest vertex under the tile moved
    // since the data its bound before is of.
    displacement: u32,
    // The tile's bound, f32 bits, once settled.
    error: u32,
    // The largest distance a replaced vertex moved, f32 bits (non-negative: ordered as u32s).
    moved: atomic<u32>,
}
