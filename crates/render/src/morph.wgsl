// The geomorph's stamp pass (render::raster): when each tile of this frame's cut came into it,
// for the morph's time factor, and its drawn edges, for the shading pass. One thread per cut
// entry.
//
// After tree::NODE_WGSL, lod::CLUSTER_WGSL and raster::MORPH_WGSL. A tile keeps its stamp while it
// stays in the cut from frame to frame, whatever data its slot holds: new data for a tile already
// drawn (a refresh replacing a stand-in, at the same level) is swapped in place with no morph
// (spec, "A stroke seen from a distance", amended 2026-11-01). A tile new to the cut is stamped
// now, so it morphs in over the duration, only when it refines what was drawn last frame (an
// ancestor was in the cut) and the camera is still: that is a level change made with the camera
// still (an arrival, a refresh), which would otherwise pop. Any other tile new to the cut is
// stamped a whole duration ago, so it is drawn at its level at once: moving, the distance factor
// alone governs the morph; and a tile replacing its finer descendants (a cut that coarsens, as
// after a stroke), or one with nothing drawn before it, has nothing to morph from.

// The tile tree.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// The cut (render::lod's CutEntry).
@group(0) @binding(1) var<storage, read> cut: array<CutEntry>;
// The word array holding the cut's length, at params.count_word.
@group(0) @binding(2) var<storage, read> counts: array<u32>;
// Per node, its stamp.
@group(0) @binding(3) var<storage, read_write> morph: array<MorphState>;
// The frame's settings.
@group(0) @binding(4) var<uniform> params: RasterParams;
// Per node, its tile's drawn edges as the cut entry gives them (render::lod::DrawnEdges): the
// shading pass rebuilds a visibility id's cluster from them.
@group(0) @binding(5) var<storage, read_write> drawn_edges: array<u32>;

// One tile of the cut (render::lod's GpuCutEntry): 16 bytes.
struct CutEntry {
    // Its face.
    face: u32,
    // Its place.
    place: u32,
    // Its node.
    node: u32,
    // The level choice's flags.
    flags: u32,
}

// The most levels a walk up the tree takes.
const MAX_WALK: u32 = 16u;

// Whether node `n`'s stamp is its tile's and from last frame.
fn stamped_last_frame(n: u32) -> bool {
    let s = morph[n];
    return s.frame == params.frame - 1u && s.face == tree[n].face && s.place == tree[n].place;
}

// Whether an ancestor of node `n` was in last frame's cut. Ancestors of a tile in this frame's
// cut aren't in it, so this pass writes none of their stamps.
fn refines_last_frame(n: u32) -> bool {
    var at = tree[n].parent;
    for (var k = 0u; k < MAX_WALK && at != NONE; k++) {
        if stamped_last_frame(at) {
            return true;
        }
        at = tree[at].parent;
    }
    return false;
}

// Stamps the tile of one cut entry and keeps its drawn edges.
@compute @workgroup_size(64)
fn stamp(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = min(counts[params.count_word], arrayLength(&cut));
    if id.x >= count {
        return;
    }
    let entry = cut[id.x];
    let n = entry.node;
    if n >= arrayLength(&morph) {
        return;
    }
    let s = morph[n];
    let staying = s.face == entry.face && s.place == entry.place
        && (s.frame == params.frame - 1u || s.frame == params.frame);
    var start = s.start;
    if !staying {
        start = params.now - params.duration;
        if (params.flags & MOVING) == 0u && refines_last_frame(n) {
            start = params.now;
        }
    }
    morph[n] = MorphState(entry.face, entry.place, params.frame, start);
    drawn_edges[n] = (entry.flags >> CUT_EDGES_SHIFT) & 0xffu;
}
