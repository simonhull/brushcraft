// The level choice (render::lod): which resident tiles a frame draws, culled, with requests
// for the tiles it wanted and found missing.
//
// Appended to tree::NODE_WGSL, residency::REQUESTS_WGSL, cull::CULL_WGSL and lod::CLUSTER_WGSL.
// One thread per node of the tile tree in every pass but threshold, finish and post. The
// passes, in order (lod.rs dispatches them):
//
// 1. begin: clears the frame's counts and draw arguments.
// 2. choose: per tile, whether it is in view and wants refining (hysteresis against last
//    frame's choice), its screen error, whether its children are all resident.
// 3. close: per tile, its screen error capped by its ancestors' (so the refined set stays a
//    tree under any threshold), and whether every ancestor wants refining and can be; for
//    those, a histogram of the triangles each possible refinement adds, by that error.
// 4. threshold: the error the triangle budget allows refining down to, from the histogram.
// 5. mark: refines every such tile that wants it, can be, and is above the threshold.
// 6. relax, up to a fixed number of passes: a reached tile in the cut is refined if it
//    touches an area two levels finer, or wants refining above the threshold (a tile reached
//    only through an earlier raise). In place and monotone (bits are only set), so passes
//    converge to the same choice in any order; each pass's dispatch is gated on the last one
//    having raised something (gate.wgsl), so passes past the settling cost nothing.
// 7. sweep, one pass per level, coarse to fine: a tile keeps its refinement only if each
//    neighbour across its edges is reached at its level, so neighbours in the cut are at most
//    one level apart by construction; refinements taken back are the shortfall.
// 8. emit: the cut (tiles reached and not refined), read back for residency; requests;
//    frustum, cone and first-phase occlusion culling (the previous frame's pyramid, seen from
//    the previous frame's view); the visible clusters and their draw arguments, and the tiles
//    to test again.
// 9. finish: clamps the counts and sets the second phase's dispatch.
// 10. post (indirect) and post_finish: the second phase, against this frame's pyramid.

// The tile tree.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// The tiles' bounds, by node index.
@group(0) @binding(1) var<storage, read> bounds: array<TileBounds>;
// Per node: last frame's choice (kept across frames) and this frame's working state.
@group(0) @binding(2) var<storage, read_write> work: array<NodeWork>;
// The frame's counts, histogram and draw arguments.
@group(0) @binding(3) var<storage, read_write> frame: LodFrame;
// Residency's request buffer: tiles wanted and missing.
@group(0) @binding(4) var<storage, read_write> requests: TileRequests;
// The cut, read back for residency.
@group(0) @binding(5) var<storage, read_write> cut: array<CutEntry>;
// Tiles that failed the first phase's occlusion test, for the second.
@group(0) @binding(6) var<storage, read_write> retry: array<u32>;
// The frame's settings.
@group(0) @binding(7) var<uniform> params: LodParams;
// This frame's view.
@group(0) @binding(8) var<uniform> view: View;
// The pass index: the level for sweep, the pass for relax.
@group(0) @binding(9) var<uniform> cursor: Step;
// The phase's visible clusters (the first phase's list in main, the second's in post).
@group(1) @binding(0) var<storage, read_write> clusters: array<Cluster>;
// The phase's depth pyramid: the previous frame's in main, this frame's in post.
@group(1) @binding(1) var pyramid: texture_2d<f32>;
// The view the phase's pyramid was made from.
@group(1) @binding(2) var<uniform> occlusion_view: View;

// Per node (render::lod::NodeWork): 32 bytes.
struct NodeWork {
    // The face and place `memory` is about: last frame's tile at this node.
    face: u32,
    // See face.
    place: u32,
    // Last frame's choice: MEM_REFINED.
    memory: u32,
    // This frame's state: W_ bits.
    flags: atomic<u32>,
    // The projected e, pixels (0 out of view).
    priority: f32,
    // The priority capped by every ancestor's.
    effective: f32,
    // Its entry in the cut, or NONE.
    cut_index: u32,
    // Zero.
    reserved: u32,
}

// Draw arguments for one indexed indirect draw (wgpu::util::DrawIndexedIndirectArgs).
struct DrawArgs {
    // Indices per cluster.
    index_count: u32,
    // Clusters: counted as they are appended, clamped by finish.
    instance_count: atomic<u32>,
    // Zero.
    first_index: u32,
    // Zero.
    base_vertex: i32,
    // Zero.
    first_instance: u32,
}

// The frame's results (render::lod::GpuFrame).
struct LodFrame {
    // The first phase's draw.
    main: DrawArgs,
    // The second phase's draw.
    post: DrawArgs,
    // The second phase's dispatch: workgroups along x, y, z.
    post_x: u32,
    // See post_x.
    post_y: u32,
    // See post_x.
    post_z: u32,
    // The lowest histogram bucket refined.
    threshold: u32,
    // Triangles of the level-0 tiles.
    base: atomic<u32>,
    // Triangles of the refined choice, kept within the budget.
    triangles: atomic<u32>,
    // Tiles in the cut.
    cut_count: atomic<u32>,
    // Tiles to test again in the second phase.
    retry_count: atomic<u32>,
    // Triangles of the cut.
    cut_triangles: atomic<u32>,
    // Triangles drawn, both phases.
    drawn_triangles: atomic<u32>,
    // Tiles drawn in the first phase.
    drawn_main: atomic<u32>,
    // Tiles drawn in the second.
    drawn_post: atomic<u32>,
    // Tiles in the cut, in view, with e over 1 px.
    over_pixel: atomic<u32>,
    // ... of which their children aren't resident.
    waiting: atomic<u32>,
    // ... of which the budget held them back.
    budget_limited: atomic<u32>,
    // ... of which the neighbour rule held them back.
    held_by_neighbours: atomic<u32>,
    // Tiles requested.
    requested: atomic<u32>,
    // Tiles in the cut culled by the frustum.
    culled_frustum: atomic<u32>,
    // ... by the normal cone.
    culled_cone: atomic<u32>,
    // ... by the first phase's occlusion test.
    culled_occlusion: atomic<u32>,
    // ... by the second phase's.
    culled_occlusion_post: atomic<u32>,
    // Clusters past the lists' end.
    cluster_overflow: atomic<u32>,
    // Tiles past the retry list's end (drawn in the first phase instead).
    retry_overflow: atomic<u32>,
    // Tiles past the cut list's end.
    cut_overflow: atomic<u32>,
    // Refinements the sweep took back.
    shortfall: atomic<u32>,
    // Tiles a relaxation pass couldn't raise for the budget.
    refused: atomic<u32>,
    // Tiles a relaxation pass couldn't raise for missing children.
    stuck: atomic<u32>,
    // Zero.
    reserved: u32,
    // Tiles raised by each relaxation pass.
    raises: array<atomic<u32>, 16>,
    // Triangles each possible refinement adds, by bucket of its capped screen error.
    histogram: array<atomic<u32>, 256>,
}

// One tile of the cut (render::lod::GpuCutEntry): 16 bytes.
struct CutEntry {
    // Its face.
    face: u32,
    // Its place, as TileNode.place.
    place: u32,
    // Its node.
    node: u32,
    // CUT_DRAWN once drawn by either phase.
    flags: u32,
}

// The frame's settings (render::lod::GpuParams).
struct LodParams {
    // Nodes in the tile tree.
    capacity: u32,
    // The project's finest level.
    finest: u32,
    // The triangle budget.
    budget: u32,
    // Relaxation passes.
    passes: u32,
    // Clusters each phase's list holds.
    cluster_capacity: u32,
    // Entries the cut list holds.
    cut_capacity: u32,
    // Entries the retry list holds.
    retry_capacity: u32,
    // 1 when the main phase has a previous frame's pyramid.
    previous: u32,
    // Refine while e projects above this many pixels.
    refine_error: f32,
    // ... or a quad edge above this many.
    refine_quad: f32,
    // Coarsen only once e projects below this many pixels.
    coarsen_error: f32,
    // ... and a quad edge below this many.
    coarsen_quad: f32,
    // OPTION_ bits.
    options: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// The pass index (render::lod::GpuStep), one per dynamic offset.
struct Step {
    // The level (sweep) or the pass (relax).
    index: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// NodeWork.flags: the node holds a tile.
const W_LIVE: u32 = 1u;
// In view, its bounds grown by e: the frustum and normal-cone tests.
const W_VISIBLE: u32 = 2u;
// Wants refining (by e or the quad edge, with hysteresis).
const W_WANTS: u32 = 4u;
// Every child is resident.
const W_COMPLETE: u32 = 8u;
// Every ancestor wants refining and is complete.
const W_CLOSED: u32 = 16u;
// Refined, by mark or a relaxation pass.
const W_REFINED: u32 = 32u;
// Refined in the end, after the sweep.
const W_FINAL: u32 = 128u;
// A relaxation pass wanted to raise it, and its children aren't resident.
const W_STUCK: u32 = 256u;
// A relaxation pass wanted to raise it, and the budget refused.
const W_REFUSED: u32 = 512u;
// The sweep took its refinement back.
const W_SHORT: u32 = 1024u;
// Wanted refining, below the budget's threshold.
const W_HELD: u32 = 2048u;
// Outside the frustum, by the sphere of what it draws: its seams are off-screen.
const W_OUTSIDE: u32 = 4096u;
// LodParams.options: seams of tiles outside the frustum are exempt from the neighbour rule.
const OPTION_OFFSCREEN_SEAMS: u32 = 1u;
// NodeWork.memory: the tile was refined last frame.
const MEM_REFINED: u32 = 1u;
// CutEntry.flags: drawn.
const CUT_DRAWN: u32 = 1u;
// Tiles at this level and finer have four children; coarser ones one (a whole face).
const SPLIT_FROM: u32 = 6u;
// Histogram buckets: an eighth of an octave each, from 2^-8 px.
const BUCKETS: u32 = 256u;
// Indices per cluster: 8 × 8 quads, two triangles each.
const CLUSTER_INDICES: u32 = 384u;

// The level of a place (TileNode.place).
fn level_of(place: u32) -> u32 {
    return place >> 28u;
}

// Quads along a side of a tile at `level`.
fn quads_of(level: u32) -> u32 {
    return 1u << min(level, SPLIT_FROM);
}

// Triangles a tile at `level` draws.
fn triangles_of(level: u32) -> u32 {
    let q = quads_of(level);
    return 2u * q * q;
}

// Triangles refining a tile at `level` adds: its children draw four times its own.
fn delta_of(level: u32) -> u32 {
    return 3u * triangles_of(level);
}

// This frame's state of node `n`.
fn flags_of(n: u32) -> u32 {
    return atomicLoad(&work[n].flags);
}

// Whether node `n`'s tile has children at all: above the finest level, a display tile.
fn has_children(n: u32) -> bool {
    return level_of(tree[n].place) < params.finest && (tree[n].flags & NODE_FINEST) == 0u;
}

// Whether every child of node `n` is resident.
fn complete(n: u32) -> bool {
    if !has_children(n) {
        return false;
    }
    if level_of(tree[n].place) < SPLIT_FROM {
        return tree[n].children[0] != NONE;
    }
    return tree[n].children[0] != NONE && tree[n].children[1] != NONE
        && tree[n].children[2] != NONE && tree[n].children[3] != NONE;
}

// The histogram bucket of a screen error.
fn bucket(error: f32) -> u32 {
    if !(error > 0.0) {
        return 0u;
    }
    return u32(clamp(floor((log2(error) + 8.0) * 8.0), 0.0, f32(BUCKETS - 1u)));
}

// Whether node `n` is reached when refinements are read from `bit`: a level-0 tile, or one
// whose parent is refined.
fn reached(n: u32, bit: u32) -> bool {
    if level_of(tree[n].place) == 0u {
        return true;
    }
    let parent = tree[n].parent;
    return parent != NONE && (flags_of(parent) & bit) != 0u;
}

// Children of a split tile along its edge `edge` (as TileNode.neighbours orders edges).
fn edge_children(edge: u32) -> vec2<u32> {
    if edge == 1u {
        return vec2<u32>(1u, 3u);
    }
    if edge == 2u {
        return vec2<u32>(2u, 3u);
    }
    if edge == 3u {
        return vec2<u32>(0u, 2u);
    }
    return vec2<u32>(0u, 1u);
}

// Whether a child of node `m` along its edge `edge` is refined, read from `bit`.
fn edge_refined(m: u32, edge: u32, bit: u32) -> bool {
    if level_of(tree[m].place) < SPLIT_FROM {
        let only = tree[m].children[0];
        return only != NONE && (flags_of(only) & bit) != 0u;
    }
    let pair = edge_children(edge);
    let a = tree[m].children[pair.x];
    let b = tree[m].children[pair.y];
    return (a != NONE && (flags_of(a) & bit) != 0u) || (b != NONE && (flags_of(b) & bit) != 0u);
}

// Whether the seam between node `n` and node `m` (NONE: not resident) is exempt from the
// neighbour rule: either side lies outside the frustum, so the edge they share is off-screen.
fn offscreen_seam(n: u32, m: u32) -> bool {
    if (params.options & OPTION_OFFSCREEN_SEAMS) == 0u {
        return false;
    }
    return (flags_of(n) & W_OUTSIDE) != 0u || (m != NONE && (flags_of(m) & W_OUTSIDE) != 0u);
}

// Whether node `n`, in the cut by `bit`, touches an area two levels finer: a neighbour at its
// level refined, with that neighbour's child along the shared edge refined too. Seams off the
// screen don't count.
fn violates(n: u32, bit: u32) -> bool {
    for (var k = 0u; k < 4u; k++) {
        if ((tree[n].edges >> k) & 1u) == 0u {
            continue;
        }
        let m = tree[n].neighbours[k];
        if m == NONE || (flags_of(m) & bit) == 0u || offscreen_seam(n, m) {
            continue;
        }
        for (var back = 0u; back < 4u; back++) {
            if tree[m].neighbours[back] == n && edge_refined(m, back, bit) {
                return true;
            }
        }
    }
    return false;
}

// Clears the frame's counts and sets the fixed draw arguments.
@compute @workgroup_size(256)
fn begin(@builtin(local_invocation_index) t: u32) {
    if t < BUCKETS {
        atomicStore(&frame.histogram[t], 0u);
    }
    if t < 16u {
        atomicStore(&frame.raises[t], 0u);
    }
    if t != 0u {
        return;
    }
    frame.main.index_count = CLUSTER_INDICES;
    atomicStore(&frame.main.instance_count, 0u);
    frame.main.first_index = 0u;
    frame.main.base_vertex = 0;
    frame.main.first_instance = 0u;
    frame.post.index_count = CLUSTER_INDICES;
    atomicStore(&frame.post.instance_count, 0u);
    frame.post.first_index = 0u;
    frame.post.base_vertex = 0;
    frame.post.first_instance = 0u;
    frame.post_x = 0u;
    frame.post_y = 1u;
    frame.post_z = 1u;
    frame.threshold = BUCKETS;
    atomicStore(&frame.base, 0u);
    atomicStore(&frame.triangles, 0u);
    atomicStore(&frame.cut_count, 0u);
    atomicStore(&frame.retry_count, 0u);
    atomicStore(&frame.cut_triangles, 0u);
    atomicStore(&frame.drawn_triangles, 0u);
    atomicStore(&frame.drawn_main, 0u);
    atomicStore(&frame.drawn_post, 0u);
    atomicStore(&frame.over_pixel, 0u);
    atomicStore(&frame.waiting, 0u);
    atomicStore(&frame.budget_limited, 0u);
    atomicStore(&frame.held_by_neighbours, 0u);
    atomicStore(&frame.requested, 0u);
    atomicStore(&frame.culled_frustum, 0u);
    atomicStore(&frame.culled_cone, 0u);
    atomicStore(&frame.culled_occlusion, 0u);
    atomicStore(&frame.culled_occlusion_post, 0u);
    atomicStore(&frame.cluster_overflow, 0u);
    atomicStore(&frame.retry_overflow, 0u);
    atomicStore(&frame.cut_overflow, 0u);
    atomicStore(&frame.shortfall, 0u);
    atomicStore(&frame.refused, 0u);
    atomicStore(&frame.stuck, 0u);
    frame.reserved = 0u;
}

// Whether a tile of bounds `b` and bound `e` faces wholly away for refining: its drawn cone,
// the sphere grown by e. The one place a per-node refine cone (the finest normals' cone the
// display records are to store) replaces the drawn cone when it arrives.
fn refine_facing_away(b: TileBounds, e: f32) -> bool {
    return backfacing(b, view.eye.xyz, e);
}

// Per tile: in view, wants refining (with hysteresis), complete; the screen error.
@compute @workgroup_size(64)
fn choose(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    if (tree[n].flags & NODE_LIVE) == 0u {
        atomicStore(&work[n].flags, 0u);
        return;
    }
    let face = tree[n].face;
    let place = tree[n].place;
    let mine = work[n].face == face && work[n].place == place;
    let was_refined = mine && (work[n].memory & MEM_REFINED) != 0u;
    let b = bounds[n];
    let e = tree[n].error;
    var out = W_LIVE;
    var priority = 0.0;
    if b.radius >= 0.0 && !in_frustum(view, b.center, b.radius) {
        out |= W_OUTSIDE;
    }
    if b.radius >= 0.0
        && in_frustum(view, b.center, b.radius + e)
        && !refine_facing_away(b, e) {
        out |= W_VISIBLE;
        let e_px = projected(view, b, e, e);
        let q_px = projected(view, b, e, b.quad_edge);
        priority = e_px;
        var wants = e_px > params.refine_error || q_px > params.refine_quad;
        if was_refined {
            wants = e_px >= params.coarsen_error || q_px >= params.coarsen_quad;
        }
        if wants {
            out |= W_WANTS;
        }
    }
    if complete(n) {
        out |= W_COMPLETE;
    }
    if level_of(place) == 0u {
        atomicAdd(&frame.base, triangles_of(0u));
    }
    work[n].priority = priority;
    work[n].effective = priority;
    work[n].cut_index = NONE;
    atomicStore(&work[n].flags, out);
}

// Per tile: reachable (every ancestor wants refining and is complete), the capped error, and
// the histogram of what refining it would add.
@compute @workgroup_size(64)
fn close(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    let f = flags_of(n);
    if (f & W_LIVE) == 0u {
        return;
    }
    let own_level = level_of(tree[n].place);
    var level = own_level;
    var ancestor = tree[n].parent;
    var effective = work[n].priority;
    var closed = true;
    // At most one step per level: the walk ends at level 0, or at a missing ancestor (the
    // tile is then unreachable).
    for (var guard = 0u; guard < 16u && level > 0u; guard++) {
        if ancestor == NONE {
            return;
        }
        let fa = flags_of(ancestor);
        if (fa & (W_WANTS | W_COMPLETE)) != (W_WANTS | W_COMPLETE) {
            closed = false;
        }
        effective = min(effective, work[ancestor].priority);
        level = level_of(tree[ancestor].place);
        ancestor = tree[ancestor].parent;
    }
    if level != 0u {
        return;
    }
    work[n].effective = effective;
    if !closed {
        return;
    }
    atomicOr(&work[n].flags, W_CLOSED);
    if (f & (W_WANTS | W_COMPLETE)) == (W_WANTS | W_COMPLETE) {
        atomicAdd(&frame.histogram[bucket(effective)], delta_of(own_level));
    }
}

// The lowest bucket the budget allows refining, from the top bucket down.
@compute @workgroup_size(1)
fn threshold() {
    var total = atomicLoad(&frame.base);
    var chosen = BUCKETS;
    for (var i = BUCKETS; i > 0u; i--) {
        let add = atomicLoad(&frame.histogram[i - 1u]);
        if total > params.budget || add > params.budget - total {
            break;
        }
        total += add;
        chosen = i - 1u;
    }
    frame.threshold = chosen;
    atomicStore(&frame.triangles, total);
}

// Refines every reachable tile that wants it and can be, down to the threshold.
@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    let f = flags_of(n);
    let needed = W_LIVE | W_CLOSED | W_WANTS | W_COMPLETE;
    if (f & needed) != needed {
        return;
    }
    if bucket(work[n].effective) >= frame.threshold {
        atomicOr(&work[n].flags, W_REFINED);
    } else {
        atomicOr(&work[n].flags, W_HELD);
    }
}

// Whether tile `n` (state `f`) wants refining and may be: complete, its capped error at or
// above the budget's threshold.
fn wanted_now(n: u32, f: u32) -> bool {
    return (f & (W_WANTS | W_COMPLETE)) == (W_WANTS | W_COMPLETE)
        && bucket(work[n].effective) >= frame.threshold;
}

// One relaxation pass: a reached tile of the cut touching an area two levels finer, or wanting
// refining above the threshold, is refined if its children are resident and the budget allows.
@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    let f = flags_of(n);
    if (f & W_LIVE) == 0u || (f & W_REFINED) != 0u || !reached(n, W_REFINED) {
        return;
    }
    let violation = violates(n, W_REFINED);
    if !violation && !wanted_now(n, f) {
        return;
    }
    if (f & W_COMPLETE) == 0u {
        atomicOr(&work[n].flags, W_STUCK);
        return;
    }
    let delta = delta_of(level_of(tree[n].place));
    let before = atomicAdd(&frame.triangles, delta);
    if before <= params.budget && delta <= params.budget - before {
        atomicOr(&work[n].flags, W_REFINED);
        atomicAdd(&frame.raises[cursor.index], 1u);
    } else {
        atomicSub(&frame.triangles, delta);
        atomicOr(&work[n].flags, W_REFUSED);
    }
}

// Whether node `n` is reached in the end: level 0, or its parent finally refined.
fn reached_final(n: u32) -> bool {
    return reached(n, W_FINAL);
}

// The sweep at one level: a refined tile stays refined only if every neighbour across its
// edges is reached at its level.
@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    let f = flags_of(n);
    if (f & W_LIVE) == 0u || level_of(tree[n].place) != cursor.index {
        return;
    }
    if (f & W_REFINED) == 0u || !reached_final(n) {
        return;
    }
    var ok = true;
    for (var k = 0u; k < 4u; k++) {
        if ((tree[n].edges >> k) & 1u) == 0u {
            continue;
        }
        let m = tree[n].neighbours[k];
        if (m == NONE || !reached_final(m)) && !offscreen_seam(n, m) {
            ok = false;
        }
    }
    if ok {
        atomicOr(&work[n].flags, W_FINAL);
    } else {
        atomicOr(&work[n].flags, W_SHORT);
        atomicAdd(&frame.shortfall, 1u);
    }
}

// Requests node `n`'s missing children, at screen error `error`.
fn request_children(n: u32, error: f32) {
    let place = tree[n].place;
    let level = level_of(place);
    let child = (level + 1u) << 28u;
    let tx = (place >> 14u) & 0x3fffu;
    let ty = place & 0x3fffu;
    var count = 4u;
    if level < SPLIT_FROM {
        count = 1u;
    }
    for (var k = 0u; k < count; k++) {
        if tree[n].children[k] != NONE {
            continue;
        }
        var cx = 2u * tx + (k & 1u);
        var cy = 2u * ty + (k >> 1u);
        if level < SPLIT_FROM {
            cx = 0u;
            cy = 0u;
        }
        let i = atomicAdd(&requests.count, 1u);
        if i < arrayLength(&requests.entries) {
            requests.entries[i] = TileRequest(tree[n].face, child | (cx << 14u) | cy, error, 0u);
            atomicAdd(&frame.requested, 1u);
        } else {
            atomicAdd(&requests.dropped, 1u);
        }
    }
}

// The edges of node `n` whose neighbour in the cut is a level coarser: bit k for edge k.
fn coarser_edges(n: u32) -> u32 {
    var mask = 0u;
    for (var k = 0u; k < 4u; k++) {
        if ((tree[n].edges >> k) & 1u) == 0u {
            continue;
        }
        let m = tree[n].neighbours[k];
        if m == NONE || !reached_final(m) {
            mask |= 1u << k;
        }
    }
    return mask;
}

// Appends node `n`'s clusters to the phase's list (`phase` 0: main, 1: post) and marks it
// drawn.
fn draw(n: u32, phase: u32) {
    let level = level_of(tree[n].place);
    let quads = quads_of(level);
    let per = max(quads / 8u, 1u);
    let cluster_quads = min(quads, 8u);
    let count = per * per;
    let mask = coarser_edges(n);
    var first = 0u;
    if phase == 0u {
        first = atomicAdd(&frame.main.instance_count, count);
        atomicAdd(&frame.drawn_main, 1u);
    } else {
        first = atomicAdd(&frame.post.instance_count, count);
        atomicAdd(&frame.drawn_post, 1u);
    }
    for (var c = 0u; c < count; c++) {
        let at = first + c;
        if at >= params.cluster_capacity {
            atomicAdd(&frame.cluster_overflow, 1u);
            continue;
        }
        let cx = c % per;
        let cy = c / per;
        var variant = 0u;
        if cy == 0u {
            variant |= mask & 1u;
        }
        if cx == per - 1u {
            variant |= mask & 2u;
        }
        if cy == per - 1u {
            variant |= mask & 4u;
        }
        if cx == 0u {
            variant |= mask & 8u;
        }
        clusters[at] = Cluster(n, cx | (cy << 8u) | (cluster_quads << 16u) | (variant << 24u));
    }
    let entry = work[n].cut_index;
    if entry != NONE {
        cut[entry].flags |= CUT_DRAWN;
    }
    atomicAdd(&frame.drawn_triangles, triangles_of(level));
}

// Whether the sphere (center, radius) is hidden behind the depth in `pyramid`, seen from `v`:
// its nearest depth is farther than the farthest depth over its screen rectangle. A sphere
// reaching behind the near plane, or off the image, is never occluded.
fn occluded(v: View, center: vec3<f32>, radius: f32) -> bool {
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    var nearest = 0.0;
    let q = center - v.eye.xyz;
    for (var k = 0u; k < 8u; k++) {
        let corner = q + vec3<f32>(
            select(-radius, radius, (k & 1u) != 0u),
            select(-radius, radius, (k & 2u) != 0u),
            select(-radius, radius, (k & 4u) != 0u),
        );
        let clip = v.view_proj * vec4<f32>(corner, 1.0);
        if clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        lo = min(lo, ndc.xy);
        hi = max(hi, ndc.xy);
        nearest = max(nearest, ndc.z);
    }
    if nearest >= 1.0 {
        return false;
    }
    // The rectangle, dilated by a pixel against rounding in the projection.
    let size = v.viewport.xy;
    let x0 = (lo.x * 0.5 + 0.5) * size.x - 1.0;
    let x1 = (hi.x * 0.5 + 0.5) * size.x + 1.0;
    let y0 = (0.5 - hi.y * 0.5) * size.y - 1.0;
    let y1 = (0.5 - lo.y * 0.5) * size.y + 1.0;
    if x1 < 0.0 || y1 < 0.0 || x0 >= size.x || y0 >= size.y {
        return false;
    }
    let p0 = vec2<u32>(vec2<f32>(max(x0, 0.0), max(y0, 0.0)));
    let p1 = vec2<u32>(vec2<f32>(min(x1, size.x - 1.0), min(y1, size.y - 1.0)));
    let levels = textureNumLevels(pyramid);
    var m = 0u;
    for (; m + 1u < levels; m++) {
        let s = m + 1u;
        if (p1.x >> s) - (p0.x >> s) <= 1u && (p1.y >> s) - (p0.y >> s) <= 1u {
            break;
        }
    }
    let s = m + 1u;
    let last = textureDimensions(pyramid, m) - vec2<u32>(1u);
    let a = min(p0 >> vec2<u32>(s), last);
    let b = min(p1 >> vec2<u32>(s), last);
    let far = min(
        min(textureLoad(pyramid, a, m).x, textureLoad(pyramid, vec2<u32>(b.x, a.y), m).x),
        min(textureLoad(pyramid, vec2<u32>(a.x, b.y), m).x, textureLoad(pyramid, b, m).x),
    );
    return nearest < far;
}

// Per tile: the cut, requests, culling, the first phase's clusters, last frame's choice for
// the next frame's hysteresis.
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= params.capacity {
        return;
    }
    let f = flags_of(n);
    if (f & W_LIVE) == 0u {
        return;
    }
    let face = tree[n].face;
    let place = tree[n].place;
    let refined = (f & W_FINAL) != 0u;
    work[n].face = face;
    work[n].place = place;
    work[n].memory = select(0u, MEM_REFINED, refined);
    if (f & W_REFUSED) != 0u {
        atomicAdd(&frame.refused, 1u);
    }
    if (f & W_STUCK) != 0u {
        atomicAdd(&frame.stuck, 1u);
    }
    if refined || !reached_final(n) {
        return;
    }
    let index = atomicAdd(&frame.cut_count, 1u);
    if index < params.cut_capacity {
        cut[index] = CutEntry(face, place, n, 0u);
        work[n].cut_index = index;
    } else {
        atomicAdd(&frame.cut_overflow, 1u);
    }
    atomicAdd(&frame.cut_triangles, triangles_of(level_of(place)));
    let priority = work[n].priority;
    if (f & W_VISIBLE) != 0u && priority > params.refine_error {
        atomicAdd(&frame.over_pixel, 1u);
        if has_children(n) && (f & W_COMPLETE) == 0u {
            atomicAdd(&frame.waiting, 1u);
        } else if (f & (W_HELD | W_REFUSED)) != 0u
            || ((f & W_WANTS) != 0u && bucket(work[n].effective) < frame.threshold) {
            atomicAdd(&frame.budget_limited, 1u);
        } else if (f & W_SHORT) != 0u {
            atomicAdd(&frame.held_by_neighbours, 1u);
        }
    }
    let wanted = (f & (W_VISIBLE | W_WANTS)) == (W_VISIBLE | W_WANTS) && (f & W_COMPLETE) == 0u;
    if (wanted || (f & W_STUCK) != 0u) && has_children(n) {
        request_children(n, priority);
    }
    let b = bounds[n];
    if b.radius >= 0.0 {
        if !in_frustum(view, b.center, b.radius) {
            atomicAdd(&frame.culled_frustum, 1u);
            return;
        }
        if backfacing(b, view.eye.xyz, 0.0) {
            atomicAdd(&frame.culled_cone, 1u);
            return;
        }
        if params.previous != 0u && occluded(occlusion_view, b.center, b.radius) {
            atomicAdd(&frame.culled_occlusion, 1u);
            let at = atomicAdd(&frame.retry_count, 1u);
            if at < params.retry_capacity {
                retry[at] = n;
                return;
            }
            atomicAdd(&frame.retry_overflow, 1u);
        }
    }
    draw(n, 0u);
}

// Clamps the first phase's counts and sets the second phase's dispatch.
@compute @workgroup_size(1)
fn finish() {
    let clusters_main = atomicLoad(&frame.main.instance_count);
    atomicStore(&frame.main.instance_count, min(clusters_main, params.cluster_capacity));
    let tested = min(atomicLoad(&frame.retry_count), params.retry_capacity);
    frame.post_x = (tested + 63u) / 64u;
    frame.post_y = 1u;
    frame.post_z = 1u;
}

// The second phase: each tile the first phase's test hid, against this frame's pyramid.
@compute @workgroup_size(64)
fn post(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= min(atomicLoad(&frame.retry_count), params.retry_capacity) {
        return;
    }
    let n = retry[i];
    let b = bounds[n];
    if occluded(occlusion_view, b.center, b.radius) {
        atomicAdd(&frame.culled_occlusion_post, 1u);
        return;
    }
    draw(n, 1u);
}

// Clamps the second phase's cluster count.
@compute @workgroup_size(1)
fn post_finish() {
    let clusters_post = atomicLoad(&frame.post.instance_count);
    atomicStore(&frame.post.instance_count, min(clusters_post, params.cluster_capacity));
}
