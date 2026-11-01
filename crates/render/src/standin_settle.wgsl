// The stand-in pass's `settle` (render::standin): after standin.wgsl. One thread per row.

// The tile tree.
@group(0) @binding(0) var<storage, read_write> tree: array<TileNode>;
// The stand-ins' rows.
@group(0) @binding(1) var<storage, read_write> rows: array<Row>;

// Settles row `id.x`: a pending row's new bound from the step, then the bound and the cone that
// covers everything into its node, if the node still holds the row's tile.
//
// A row its jobs measured takes the larger of what they measured and the bound before plus how
// far a replaced vertex beside an unmeasured quad moved; any other (a coarse tile, a stand-in
// over a preview, a step whose vertices couldn't be replaced) the bound before plus the step's
// displacement plus the larger of that and how far a replaced vertex moved (render::standin,
// "The bound"). Sums are raised by SLACK, more than their rounding can have taken off; a bound
// past the largest finite f32 is the store's unknown one.
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
        let measured = atomicLoad(&rows[i].measured);
        let moved_bits = atomicLoad(&rows[i].moved);
        let moved = bitcast<f32>(moved_bits);
        var raised: f32;
        if measured != 0u {
            var rest = error;
            if moved_bits != 0u {
                rest = (error + moved) * SLACK;
            }
            raised = max(bitcast<f32>(measured), rest);
        } else {
            let d = bitcast<f32>(rows[i].displacement);
            raised = (error + d + max(d, moved)) * SLACK;
        }
        error = bitcast<f32>(min(bitcast<u32>(raised), F32_MAX_BITS));
        rows[i].error = bitcast<u32>(error);
        rows[i].flags = 0u;
        atomicStore(&rows[i].measured, 0u);
        atomicStore(&rows[i].moved, 0u);
    }
    tree[n].error = error;
    tree[n].finest_cone = EVERYTHING;
}
