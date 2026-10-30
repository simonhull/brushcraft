// Accumulation while the camera is still (render::aa): which tiles' drawing changed this frame,
// then each pixel's history reset where it saw one of them, and the frame's sample added.
//
// After tree::NODE_WGSL and raster::MORPH_WGSL. The passes, in order (aa.rs dispatches them):
//
// 1. mark: the tiles whose data changed this frame (the bounds update's list, from the CPU).
// 2. nodes: per node, what its drawing depends on (its key), compared with last frame's.
// 3. spread: per node, the latest frame its drawing changed: its own key, its data, or a seam
//    input of an edge neighbour's.
// 4. accumulate (aa_pixels.wgsl): per pixel, the reset rule, then the frame's sample into the
//    running mean.
//
// Frame numbers are the accumulation's own (AaParams.frame): one per accumulated frame, from 1.

// The tile tree.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// Per node, the raster's stamp: when its tile came into the cut, and whether it is in this
// frame's.
@group(0) @binding(1) var<storage, read> morph: array<MorphState>;
// Per node, its tile's drawn edges, as the stamp pass kept them.
@group(0) @binding(2) var<storage, read> drawn_edges: array<u32>;
// The raster's frame settings: its frame number, time and the morph's duration.
@group(0) @binding(3) var<uniform> raster: RasterParams;
// Per node, its key and when it changed.
@group(0) @binding(4) var<storage, read_write> nodes: array<AaNode>;
// Per node, the latest frame its drawing changed.
@group(0) @binding(5) var<storage, read_write> changed: array<u32>;
// The latest frame any node's drawing changed, in word 0.
@group(0) @binding(6) var<storage, read_write> latest: array<atomic<u32>>;
// The nodes whose data changed this frame, AaParams.marks of them.
@group(0) @binding(7) var<storage, read> marks: array<u32>;
// The frame's settings.
@group(0) @binding(8) var<uniform> params: AaParams;

// Words of a node's key.
const KEY_WORDS: u32 = 11u;
// The key's words a neighbour's seam vertices read: its e, its parent's e, its morph clock.
const SEAM_FIRST: u32 = 1u;
const SEAM_END: u32 = 4u;

// Per node (render::aa's NODE_WORDS): what its drawing depends on, last frame's, and the
// latest frames its own drawing and its seam inputs changed.
struct AaNode {
    key: array<u32, 11>,
    own: u32,
    seam: u32,
}

// Pass 1: the tiles whose data changed this frame. Their own drawing changed.
@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.marks || id.x >= arrayLength(&marks) {
        return;
    }
    let n = marks[id.x];
    if n < arrayLength(&nodes) {
        nodes[n].own = params.frame;
    }
}

// Node `n`'s morph clock: how far its morph has run, in milliseconds up to the duration (what
// place.wgsl's morph_time is a function of), or NONE when its stamp isn't this frame's or its
// tile's (morph_time is then 1).
fn morph_clock(n: u32) -> u32 {
    if n >= arrayLength(&morph) {
        return NONE;
    }
    let s = morph[n];
    let node = tree[n];
    if s.frame != raster.frame || s.face != node.face || s.place != node.place {
        return NONE;
    }
    return min(raster.now - s.start, raster.duration);
}

// Node `n`'s key: every input place.wgsl's place_vertex and the shading read for its tile that
// can change while the camera is still. Its slot, flags, face and place; its e and its parent's
// (the morph by distance); its morph clock (by time); whether it is in this frame's cut and its
// drawn edges (the cluster's cell: edge variants and finer edges); its edges and edge
// neighbours (whose seam inputs it reads). Its data is marked separately (`mark`): an epoch
// can change it in place.
fn key_of(n: u32) -> array<u32, 11> {
    let node = tree[n];
    var parent_error = NONE;
    if node.parent != NONE {
        parent_error = bitcast<u32>(tree[node.parent].error);
    }
    let clock = morph_clock(n);
    var edges = 0u;
    if n < arrayLength(&drawn_edges) {
        edges = drawn_edges[n] & 0xffu;
    }
    let in_cut = select(0u, 1u, clock != NONE);
    let state = in_cut | (edges << 1u) | ((node.edges & 0xfu) << 9u) | ((node.flags & 0xffu) << 16u);
    return array<u32, 11>(
        node.slot,
        bitcast<u32>(node.error),
        parent_error,
        clock,
        state,
        node.face,
        node.place,
        node.neighbours[0],
        node.neighbours[1],
        node.neighbours[2],
        node.neighbours[3],
    );
}

// Pass 2: per node, its key against last frame's.
@compute @workgroup_size(64)
fn compare(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= arrayLength(&nodes) || n >= arrayLength(&tree) {
        return;
    }
    let key = key_of(n);
    var node = nodes[n];
    var own = false;
    var seam = false;
    for (var k = 0u; k < KEY_WORDS; k++) {
        if key[k] != node.key[k] {
            own = true;
            seam = seam || (k >= SEAM_FIRST && k < SEAM_END);
        }
    }
    if own {
        node.own = params.frame;
    }
    if seam {
        node.seam = params.frame;
    }
    node.key = key;
    nodes[n] = node;
}

// Pass 3: per node, the latest frame its drawing changed: its own, or an edge neighbour's seam
// inputs (a seam vertex's morph takes the larger e and the less advanced clock of the two
// tiles).
@compute @workgroup_size(64)
fn spread(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= arrayLength(&changed) || n >= arrayLength(&nodes) {
        return;
    }
    var at = nodes[n].own;
    let links = nodes[n].key;
    for (var e = 0u; e < 4u; e++) {
        let m = links[7u + e];
        if m != NONE && m < arrayLength(&nodes) {
            at = max(at, nodes[m].seam);
        }
    }
    changed[n] = at;
    // Earlier frames' changes were merged in their own frames: only this frame's contend.
    if at == params.frame {
        atomicMax(&latest[0], at);
    }
}
