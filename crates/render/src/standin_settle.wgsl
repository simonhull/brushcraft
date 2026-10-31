// The stand-in pass's `settle` (render::standin): after standin.wgsl. One thread per row.

// The tile tree.
@group(0) @binding(0) var<storage, read_write> tree: array<TileNode>;
// The stand-ins' rows.
@group(0) @binding(1) var<storage, read_write> rows: array<Row>;

// The largest finite f32.
const F32_MAX: f32 = 3.40282347e38;

// Settles row `id.x`: a pending row's new bound from the step, then the bound and the cone that
// covers everything into its node, if the node still holds the row's tile.
@compute @workgroup_size(64)
fn settle(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&rows) {
        return;
    }
    let n = rows[i].node;
    if n == NONE || n >= arrayLength(&tree) {
        return;
    }
    let live = (tree[n].flags & NODE_LIVE) != 0u;
    if !live || tree[n].face != rows[i].face || tree[n].place != rows[i].place {
        return;
    }
    var error = bitcast<f32>(rows[i].error);
    if (rows[i].flags & PENDING) != 0u {
        let d = bitcast<f32>(rows[i].displacement);
        let moved = bitcast<f32>(atomicLoad(&rows[i].moved));
        // At most the largest finite f32: a bound that overflows is the store's unknown one.
        error = min((error + d + max(d, moved)) * SLACK, F32_MAX);
        rows[i].error = bitcast<u32>(error);
        rows[i].flags = 0u;
        atomicStore(&rows[i].moved, 0u);
    }
    tree[n].error = error;
    tree[n].finest_cone = EVERYTHING;
}
