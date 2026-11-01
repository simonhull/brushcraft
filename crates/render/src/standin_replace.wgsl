// The stand-in pass's `replace` (render::standin): after standin.wgsl. One workgroup per job.

// The slot buffer of the jobs' display tiles.
@group(0) @binding(0) var<storage, read_write> slots: array<TileVertex>;
// The edited finest tiles' positions, TILE_WORDS words a tile, three a vertex, row-major.
@group(0) @binding(1) var<storage, read> scratch: array<u32>;
// The dispatch's jobs, one per workgroup.
@group(0) @binding(2) var<storage, read> jobs: array<Job>;
// The stand-ins' rows.
@group(0) @binding(3) var<storage, read_write> rows: array<Row>;

// Threads per workgroup.
const GROUP: u32 = 64u;

// The largest bound on how far a vertex of this workgroup's job moved, f32 bits.
var<workgroup> moved_most: atomic<u32>;

// The largest bound on a finest vertex's distance from the stand-in's triangles in this
// workgroup's job, f32 bits.
var<workgroup> measured_most: atomic<u32>;

// The finest vertex at word `w` of the scratch.
fn scratch_point(w: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(scratch[w]),
        bitcast<f32>(scratch[w + 1u]),
        bitcast<f32>(scratch[w + 2u]),
    );
}

// Finest vertex (x, y) of `job`'s finest tile, from the scratch.
fn finest_at(job: Job, x: u32, y: u32) -> vec3<f32> {
    let fine_side = job.step >> 16u;
    return scratch_point(job.scratch * TILE_WORDS + (y * fine_side + x) * 3u);
}

// Whether display vertex (i, j) of `job`'s display tile, finest vertex (x, y) of its finest
// tile, is a corner of a display quad that lies in a finest tile the stand-in doesn't read: one
// no job measures, whose finest vertices are only within the bound before plus how far the
// quad's corners moved. A quad beside the vertex lies in the job's finest tile, or (where the
// vertex is on that tile's edge, and the quad on the edge's far side) in the one across, on the
// same face.
fn beside_unmeasured(job: Job, i: u32, j: u32, x: u32, y: u32) -> bool {
    let quads = job_side(job) - 1u;
    let fine_quads = (job.step >> 16u) - 1u;
    var found = false;
    for (var q = 0u; q < 4u; q += 1u) {
        let right = (q & 1u) != 0u;
        let up = (q & 2u) != 0u;
        // The display tile has the quad on that side.
        let exists = select(i > 0u, i < quads, right) && select(j > 0u, j < quads, up);
        // The quad's finest tile, as its offset from the job's plus one.
        let ox = select(select(1u, 0u, x == 0u), select(1u, 2u, x == fine_quads), right);
        let oy = select(select(1u, 0u, y == 0u), select(1u, 2u, y == fine_quads), up);
        let read = (job.side >> (16u + oy * 3u + ox)) & 1u;
        found = found || (exists && read == 0u);
    }
    return found;
}

// Replaces the display vertices job `group.x`'s finest tile holds by the finest vertices they
// subsample, and keeps in the job's row bounds on how far they moved and, in a measured job, on
// how far each finest vertex of the finest tile lies from the stand-in's triangles.
//
// Two jobs of one tile can share a vertex (on the border between their finest tiles): both
// write the same bits there (every copy of a finest vertex is its owner's), and the first to
// read it reads the old position, so its distance is counted. The measurement reads the scratch
// alone: a measured job's quads have every corner in its own finest tile, the bits the replace
// writes.
@compute @workgroup_size(64)
fn replace(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let job = jobs[group.x];
    let side = job_side(job);
    let nx = job.count & 0xffffu;
    let ny = job.count >> 16u;
    let lx = job.lo & 0xffffu;
    let ly = job.lo >> 16u;
    let fi = job.fine & 0xffffu;
    let fj = job.fine >> 16u;
    let step = job.step & 0xffffu;
    let fine_side = job.step >> 16u;
    let measure = (job.side & MEASURE) != 0u;
    var most = 0u;
    for (var t = lane; t < nx * ny; t += GROUP) {
        let a = t % nx;
        let b = t / nx;
        let at = job.base + (ly + b) * side + lx + a;
        let x = fi + a * step;
        let y = fj + b * step;
        let fresh = scratch_point(job.scratch * TILE_WORDS + (y * fine_side + x) * 3u);
        let old = slots[at].position;
        slots[at].position = fresh;
        // A vertex whose bits didn't change moved by nothing; in a measured job only one beside
        // an unmeasured quad counts.
        let changed = any(bitcast<vec3<u32>>(fresh) != bitcast<vec3<u32>>(old));
        if changed && (!measure || beside_unmeasured(job, lx + a, ly + b, x, y)) {
            most = max(most, bound_bits(fresh - old, max(largest(fresh), largest(old))));
        }
    }
    var worst = 0u;
    if measure {
        // Every finest vertex of the finest tile, at its place on the display quad holding it,
        // as the bound is pointwise (render::standin, "The bound"); the last vertex of a row
        // on the last quad. A quad spans `step` finest quads, a power of two: its fractions
        // are exact.
        let shift = countTrailingZeros(step);
        let inverse = bitcast<f32>((127u - shift) << 23u);
        for (var t = lane; t < fine_side * fine_side; t += GROUP) {
            let x = t % fine_side;
            let y = t / fine_side;
            let x0 = min(x >> shift, nx - 2u) << shift;
            let y0 = min(y >> shift, ny - 2u) << shift;
            let fx = f32(x - x0) * inverse;
            let fy = f32(y - y0) * inverse;
            let fine = finest_at(job, x, y);
            let p00 = finest_at(job, x0, y0);
            let p11 = finest_at(job, x0 + step, y0 + step);
            // Split along the (x0, y0)-(x0 + step, y0 + step) diagonal, as the raster draws it;
            // each partial sum a blend of corners, no larger than they are.
            var drawn: vec3<f32>;
            var third: f32;
            if x - x0 >= y - y0 {
                let p10 = finest_at(job, x0 + step, y0);
                drawn = p00 + fx * (p10 - p00) + fy * (p11 - p10);
                third = largest(p10);
            } else {
                let p01 = finest_at(job, x0, y0 + step);
                drawn = p00 + fy * (p01 - p00) + fx * (p11 - p01);
                third = largest(p01);
            }
            let magnitude = max(max(largest(fine), largest(p00)), max(largest(p11), third));
            worst = max(worst, bound_bits(fine - drawn, magnitude));
        }
    }
    atomicMax(&moved_most, most);
    atomicMax(&measured_most, worst);
    workgroupBarrier();
    if lane == 0u {
        atomicMax(&rows[job.row].moved, atomicLoad(&moved_most));
        if measure {
            atomicMax(&rows[job.row].measured, atomicLoad(&measured_most));
        }
    }
}
