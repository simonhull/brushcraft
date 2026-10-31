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

// The largest distance a vertex of this workgroup's job moved, f32 bits.
var<workgroup> moved_most: atomic<u32>;

// The finest vertex at word `w` of the scratch.
fn scratch_point(w: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(scratch[w]),
        bitcast<f32>(scratch[w + 1u]),
        bitcast<f32>(scratch[w + 2u]),
    );
}

// Replaces the display vertices job `group.x`'s finest tile holds by the finest vertices they
// subsample, keeping the largest distance one moved in the job's row.
//
// Two jobs of one tile can share a vertex (on the border between their finest tiles): both
// write the same bits there (every copy of a finest vertex is its owner's), and the first to
// read it reads the old position, so its distance is counted.
@compute @workgroup_size(64)
fn replace(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let job = jobs[group.x];
    let nx = job.count & 0xffffu;
    let ny = job.count >> 16u;
    let lx = job.lo & 0xffffu;
    let ly = job.lo >> 16u;
    let fi = job.fine & 0xffffu;
    let fj = job.fine >> 16u;
    let step = job.step & 0xffffu;
    let fine_side = job.step >> 16u;
    var most = 0u;
    for (var t = lane; t < nx * ny; t += GROUP) {
        let a = t % nx;
        let b = t / nx;
        let at = job.base + (ly + b) * job.side + lx + a;
        let w = job.scratch * TILE_WORDS + ((fj + b * step) * fine_side + fi + a * step) * 3u;
        let fresh = scratch_point(w);
        let old = slots[at].position;
        slots[at].position = fresh;
        most = max(most, bitcast<u32>(length(fresh - old)));
    }
    atomicMax(&moved_most, most);
    workgroupBarrier();
    if lane == 0u {
        atomicMax(&rows[job.row].moved, atomicLoad(&moved_most));
    }
}
