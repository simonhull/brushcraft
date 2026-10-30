// The level choice (render::lod): which resident tiles a frame draws, culled, with requests
// for the tiles it wanted and found missing.
//
// Appended to tree::NODE_WGSL, residency::REQUESTS_WGSL, cull::CULL_WGSL, lod::CLUSTER_WGSL and
// raster::MORPH_WGSL.
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
//    frustum, cone (only for tiles whose morph is settled) and first-phase occlusion culling (the previous frame's pyramid, seen from
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
// The raster's stamps as last frame's stamp pass left them (render::raster), or one zeroed
// stamp without a raster: emit's settled test reads them.
@group(0) @binding(10) var<storage, read> stamps: array<MorphState>;
// The raster's settings for this frame (its frame number, time, morph duration, flags and
// distance ramp), or all zero without a raster (nothing morphs).
@group(0) @binding(11) var<uniform> morph_params: RasterParams;
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
    // Budget units (UNIT triangles) the refined choice is charged, kept within the budget.
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
    // Tiles in the cut, in the frustum, that their cone would cull but whose morph isn't
    // settled (drawn instead).
    cone_exempt: atomic<u32>,
    // Tiles raised by each relaxation pass.
    raises: array<atomic<u32>, 16>,
    // Budget units each possible refinement adds, by bucket of its capped screen error.
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
    // CUT_DRAWN once drawn by either phase; bits CUT_EDGES_SHIFT.. the tile's drawn edges.
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

// Triangles a budget unit counts: a full cluster's. The budget counts in units, so its sums
// can't wrap a u32 (at most 192 units a node, against 2^32 / 192 = 22M nodes, past MAX_NODES).
const UNIT: u32 = 128u;

// Budget units refining a tile at `level` is charged: exact from level 3 on (a multiple of a
// cluster), rounded up to one below.
fn delta_units(level: u32) -> u32 {
    return max(delta_of(level) / UNIT, 1u);
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

// Whether the seam across edge `k` of node `n` (reached by `bit`) is exempt from the neighbour
// rule: off the screen, judged at the level of its coarser side. That side is the tile of the
// cut across the edge: n's neighbour at its level if reached, else the neighbour at the first
// coarser level that is (n's edge k lies on its ancestors' edge k: a neighbour that isn't
// reached isn't a sibling, the parent being refined, so complete). The seam is off the screen
// when that tile or n's ancestor at its level lies outside the frustum. Relax (violates, from
// the coarser side) and the sweep (from the finer side) both judge a seam by this, so they
// agree: a seam the relaxation leaves two levels apart because it is off the screen, the sweep
// keeps, and one the sweep would take back, the relaxation raises or requests.
fn offscreen_seam(n: u32, k: u32, bit: u32) -> bool {
    if (params.options & OPTION_OFFSCREEN_SEAMS) == 0u {
        return false;
    }
    var near = n;
    var far = tree[n].neighbours[k];
    // At most one step per level: level 0 is always reached (the pinned floor).
    for (var guard = 0u; guard < 16u; guard++) {
        if far != NONE && reached(far, bit) {
            break;
        }
        let up = tree[near].parent;
        if up == NONE {
            break;
        }
        near = up;
        far = tree[near].neighbours[k];
    }
    return (flags_of(near) & W_OUTSIDE) != 0u
        || (far != NONE && (flags_of(far) & W_OUTSIDE) != 0u);
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
        if m == NONE || (flags_of(m) & bit) == 0u || offscreen_seam(n, k, bit) {
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
    atomicStore(&frame.cone_exempt, 0u);
}

// One step of a finest-normal cone's half-angle (geom::FinestNormalCone): pi / 4095, as the
// CPU computes it in f32.
const CONE_STEP: f32 = 3.1415927 / 4095.0;
// A cone's half-angle step count that means it covers everything.
const CONE_EVERYTHING: u32 = 4095u;
// pi / 2, rounded up in f32: a cone this wide or wider never faces away.
const CONE_QUARTER_TURN: f32 = 1.5707964;
// What finest_facing_away's dot(n, p - eye) bound must clear beyond the sphere's radius, as a
// fraction of the distance to the sphere: the f32 arithmetic's error, under 6e-6 of it (see
// there), with room to spare.
const CONE_SLACK: f32 = 1e-5;

// The octahedral fold's sign: 1 for zero and up, -1 below (as geom's, unlike sign(), which
// gives 0 for 0).
fn fold_sign(v: f32) -> f32 {
    return select(-1.0, 1.0, v >= 0.0);
}

// The axis of the finest-normal cone `bits` (TileNode.finest_cone), decoded as
// geom::FinestNormalCone::axis decodes it: two signed 10-bit fractions of 511 (across in bits
// 0..10, up in 10..20, sign-extended), unfolded from the octahedron, normalized.
fn cone_axis(bits: u32) -> vec3<f32> {
    let across = clamp(f32(bitcast<i32>(bits << 22u) >> 22u) / 511.0, -1.0, 1.0);
    let up = clamp(f32(bitcast<i32>(bits << 12u) >> 22u) / 511.0, -1.0, 1.0);
    let height = 1.0 - abs(across) - abs(up);
    var xy = vec2<f32>(across, up);
    if height < 0.0 {
        xy = vec2<f32>((1.0 - abs(up)) * fold_sign(across), (1.0 - abs(across)) * fold_sign(up));
    }
    // |x| + |y| + |height| is 1 either way, so the vector is never zero.
    return normalize(vec3<f32>(xy, height));
}

// The cosine and sine of `x` in [0, pi / 2], by their Taylor series to x^12 and x^13 in plain
// products and sums. WGSL rounds those correctly, but lets cos and sin be off by 2^-11 (which
// near 0 is an angle of 0.03 rad); these are within 1e-6 of the true values (the series' tails
// are under 7e-9, and the rounding of a few terms of magnitude at most about 1 adds the rest).
fn cone_cos_sin(x: f32) -> vec2<f32> {
    let t = x * x;
    let c = 1.0 + t * (-1.0 / 2.0 + t * (1.0 / 24.0 + t * (-1.0 / 720.0 + t * (1.0 / 40320.0
        + t * (-1.0 / 3628800.0 + t * (1.0 / 479001600.0))))));
    let s = x * (1.0 + t * (-1.0 / 6.0 + t * (1.0 / 120.0 + t * (-1.0 / 5040.0
        + t * (1.0 / 362880.0 + t * (-1.0 / 39916800.0 + t * (1.0 / 6227020800.0)))))));
    return vec2<f32>(c, s);
}

// Whether every finest normal under a tile faces away from `eye`, at every point of its bounds'
// sphere grown by `margin`: the finest surface lies within e of what the tile draws, so with
// margin e none of it can face the eye, and refining the tile shows nothing. (Strictly, the
// cone holds the finest *vertex* normals; a finest triangle's face normal can stray slightly
// outside it. That matters little: such a triangle is all but edge-on, and the raster pass culls
// no triangles by facing, so this test is the only place it is judged.) `cone` is the
// tile's finest-normal cone (TileNode.finest_cone); one that covers everything never faces
// away. Twin: cull::reference::finest_facing_away.
//
// backfacing's test, with the finest-normal cone for the drawn one: with v = center - eye at
// distance d, θ the angle from the axis a to v and α' the half-angle, the least dot(n, p - eye)
// over the cone and the sphere is d·cos(θ + α') - r, and
// dot(v, a)·cos(α') - |v × a|·sin(α') = |a|·d·cos(θ + α') exactly. Taking θ's sine from the
// cross product, not from its cosine, keeps it well conditioned near θ = 0, where
// sqrt(1 - cos²) would turn an f32 cosine's rounding into an angle of 4e-4 rad or more.
//
// Why it is conservative in f32:
// - The angle: every finest normal is within the stored half-angle α of the axis the CPU
//   decodes (FinestNormalCone's guarantee, measured in f64). This decode is a few ULP from
//   that one (within 2e-6 rad), and α is one f32 rounding from the CPU's, so every finest
//   normal is within α' = α + one step (pi / 4095, 7.7e-4 rad) of this axis: the one-step
//   widening FinestNormalCone's notes ask of an f32 consumer, with plenty to spare.
// - The arithmetic: q below is within 6e-6·d of |a|·d·cos(θ + α'): the dot and cross products
//   and the final products and difference round by under 2e-6·d together, cone_cos_sin's
//   1e-6 errors cost at most 2e-6·d (|dot| and |cross| are each at most d), |a| is within 1e-6
//   of 1 (another 1e-6·d), and v = center - eye is correctly rounded (the eye moved by at most
//   6e-8·d). r's rounding is under 6e-8·d where it matters (r < d).
// So q > r + CONE_SLACK·d (1e-5·d) means d·cos(θ + α') > r: every finest normal faces away from
// every point of the grown sphere. A half-angle reaching a quarter turn never passes (the least
// dot product is then at most -r), nor does an eye inside the sphere.
fn finest_facing_away(b: TileBounds, cone: u32, eye: vec3<f32>, margin: f32) -> bool {
    let steps = cone >> 20u;
    if steps == CONE_EVERYTHING || b.radius < 0.0 {
        return false;
    }
    let half = f32(steps + 1u) * CONE_STEP;
    if half >= CONE_QUARTER_TURN {
        return false;
    }
    let v = b.center - eye;
    let d = length(v);
    let r = b.radius + margin;
    if d <= r {
        return false;
    }
    let a = cone_axis(cone);
    let cs = cone_cos_sin(half);
    let q = dot(v, a) * cs.x - length(cross(v, a)) * cs.y;
    return q > r + CONE_SLACK * d;
}

// Whether node `n`, of bounds `b` and bound `e`, faces wholly away for refining, the sphere
// grown by e. A display tile is tested by the cone of the finest normals under it
// (TileNode.finest_cone), not by its own triangles: a coarse tile at a silhouette can face away
// while finest detail under it faces the camera, and the 1 px rule must hold there too
// (spec, "Normal cone"). A finest tile is tested by its drawn cone, as drawing culls it: its
// triangles are the finest surface, and it has nothing to refine into.
//
// Drawing culls every tile by its own triangles (emit's backfacing, margin 0), and a coarse
// quad's normal can lie outside the finest normals' cone: a tile can face away by its finest
// normals while its own triangles face the eye. Such a tile would be drawn, coarse, at its full
// screen error, so it is skipped for refining only when emit culls it too (the same call,
// backfacing(b, eye, 0.0)); otherwise it is refined like any tile in view.
fn refine_facing_away(n: u32, b: TileBounds, e: f32) -> bool {
    if (tree[n].flags & NODE_FINEST) != 0u {
        return backfacing(b, view.eye.xyz, e);
    }
    return finest_facing_away(b, tree[n].finest_cone, view.eye.xyz, e)
        && backfacing(b, view.eye.xyz, 0.0);
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
        && !refine_facing_away(n, b, e) {
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
        atomicAdd(&frame.histogram[bucket(effective)], delta_units(own_level));
    }
}

// The lowest bucket the budget allows refining, from the top bucket down.
@compute @workgroup_size(1)
fn threshold() {
    let budget = params.budget / UNIT;
    var total = (atomicLoad(&frame.base) + UNIT - 1u) / UNIT;
    var chosen = BUCKETS;
    for (var i = BUCKETS; i > 0u; i--) {
        let add = atomicLoad(&frame.histogram[i - 1u]);
        if total > budget || add > budget - total {
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
    let delta = delta_units(level_of(tree[n].place));
    let budget = params.budget / UNIT;
    let before = atomicAdd(&frame.triangles, delta);
    if before <= budget && delta <= budget - before {
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
        if (m == NONE || !reached_final(m)) && !offscreen_seam(n, k, W_FINAL) {
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

// The edges of node `n` whose neighbour at its level is refined in the end, so drawn finer:
// bit k for edge k.
fn finer_edges(n: u32) -> u32 {
    var mask = 0u;
    for (var k = 0u; k < 4u; k++) {
        if ((tree[n].edges >> k) & 1u) == 0u {
            continue;
        }
        let m = tree[n].neighbours[k];
        if m != NONE && reached_final(m) && (flags_of(m) & W_FINAL) != 0u {
            mask |= 1u << k;
        }
    }
    return mask;
}

// Node `n`'s drawn edges (render::lod::DrawnEdges): coarser in bits 0..4, finer in 4..8.
fn drawn_edges(n: u32) -> u32 {
    return coarser_edges(n) | (finer_edges(n) << 4u);
}

// Appends node `n`'s clusters to the phase's list (`phase` 0: main, 1: post), its drawn edges
// `edges` (drawn_edges, computed once by the caller), and marks it drawn.
fn draw(n: u32, phase: u32, edges: u32) {
    let level = level_of(tree[n].place);
    let quads = quads_of(level);
    let per = max(quads / 8u, 1u);
    let count = per * per;
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
        clusters[at] = Cluster(n, cluster_cell(c % per, c / per, quads, edges));
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
    // The rectangle, dilated by a pixel against rounding in the projection and against the half
    // pixel a jittered sample moves while still (render::aa: the raster jitters, this view
    // doesn't): don't tighten it below that.
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

// The settled test's slack for rounding (render::lod, "Cone culling and the morph"): in e's
// order, 1 + 2^-16.
const ORDER: f32 = 1.0000152587890625;
// ... in a projected bound against the ramp: 1 + 2^-8.
const RATIO: f32 = 1.00390625;
// ... in the sphere's view depths: widened by this much of its farthest distance, 2^-10.
const NEAR_SLACK: f32 = 0.0009765625;
// The least a parent e may project to at any vertex for the settled test to trust its order: a
// normal f32 with room to spare.
const SMALLEST: f32 = 1e-30;

// The e of node `n`'s parent, as place.wgsl's parent_error reads it: negative without one.
fn morph_parent_error(n: u32) -> f32 {
    let parent = tree[n].parent;
    if parent == NONE {
        return -1.0;
    }
    return tree[parent].error;
}

// Whether node `n`'s morph clock (place.wgsl's morph_time) is certainly 1 this frame, from its
// stamp as last frame's stamp pass left it and what this frame's stamp pass will make of it
// (morph.wgsl): no duration; a tile staying in the cut whose stamp is at least a duration old; a
// tile new to the cut while the camera moves (stamped a whole duration ago). A tile new to the
// cut with the camera still may be stamped now, so it isn't. A tile this frame's cut lacks reads
// 1 in place.wgsl whatever this says.
fn time_done(n: u32) -> bool {
    let p = morph_params;
    if p.duration == 0u {
        return true;
    }
    if n >= arrayLength(&stamps) {
        return false;
    }
    let s = stamps[n];
    let staying = s.face == tree[n].face && s.place == tree[n].place
        && (s.frame == p.frame - 1u || s.frame == p.frame);
    if staying {
        return p.now - s.start >= p.duration;
    }
    return (p.flags & MOVING) != 0u;
}

// Whether every odd vertex of node `n`'s tile is certainly drawn where it is (t = 1) this
// frame, its finer edges `finer` (render::lod, "Cone culling and the morph": the end case).
fn settled_at_end(n: u32, finer: u32) -> bool {
    if morph_parent_error(n) < 0.0 {
        return true;
    }
    if morph_params.ramp > 0.0 || !time_done(n) {
        return false;
    }
    let node = tree[n];
    for (var edge = 0u; edge < 4u; edge++) {
        let m = node.neighbours[edge];
        let seam = ((node.edges >> edge) & 1u) != 0u && m != NONE;
        if ((finer >> edge) & 1u) == 0u && seam && !time_done(m) {
            return false;
        }
    }
    return true;
}

// Whether odd vertices whose distance factor reads the parent bound `x` and the own bound `y`
// (world units), at view depths between `near` and `far`, certainly get a distance factor of 0
// (render::lod, "Cone culling and the morph": the start case): `x` larger and projecting within
// the ramp at the nearest depth, or not larger and `y` projecting over it at the farthest.
fn starts(x: f32, y: f32, near: f32, far: f32) -> bool {
    let r = morph_params.ramp;
    let f = view.viewport.z;
    if x > y * ORDER {
        return x * f / far >= SMALLEST && x * f / near * RATIO <= r;
    }
    return x <= y && y * f / far > r * RATIO;
}

// Whether every odd vertex of node `n`'s tile is certainly drawn where its morph starts (t = 0)
// this frame, its bounds `b` and finer edges `finer` (render::lod, "Cone culling and the
// morph": the start case).
fn settled_at_start(n: u32, b: TileBounds, finer: u32) -> bool {
    let e = morph_parent_error(n);
    if morph_params.ramp <= 0.0 || finer != 0u || e < 0.0 || b.radius < 0.0 {
        return false;
    }
    let q = b.center - view.eye.xyz;
    let depth = dot(view.planes[4].xyz, q);
    let slack = NEAR_SLACK * (length(q) + b.radius);
    let near = depth - b.radius - slack;
    let far = depth + b.radius + slack;
    if near <= view.viewport.w {
        return false;
    }
    let node = tree[n];
    if !starts(e, node.error, near, far) {
        return false;
    }
    for (var edge = 0u; edge < 4u; edge++) {
        let m = node.neighbours[edge];
        if ((node.edges >> edge) & 1u) == 0u || m == NONE {
            continue;
        }
        let other = morph_parent_error(m);
        if other < 0.0 || !starts(max(e, other), max(node.error, tree[m].error), near, far) {
            return false;
        }
    }
    return true;
}

// Whether node `n`'s tile is drawn this frame with every odd vertex at one end of its morph
// (render::lod, "Cone culling and the morph"), so its drawn triangles are its own (or its
// variant's fans) or lie in its parent triangulation: those its cone holds. Conservative: never
// true while any of them is on its way.
fn settled(n: u32, b: TileBounds, finer: u32) -> bool {
    return settled_at_end(n, finer) || settled_at_start(n, b, finer);
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
    let edges = drawn_edges(n);
    let index = atomicAdd(&frame.cut_count, 1u);
    if index < params.cut_capacity {
        cut[index] = CutEntry(face, place, n, edges << CUT_EDGES_SHIFT);
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
        // The cone holds the tile's triangles only at either end of the morph, not on the way.
        if backfacing(b, view.eye.xyz, 0.0) {
            if settled(n, b, edges >> 4u) {
                atomicAdd(&frame.culled_cone, 1u);
                return;
            }
            atomicAdd(&frame.cone_exempt, 1u);
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
    draw(n, 0u, edges);
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
    draw(n, 1u, drawn_edges(n));
}

// Clamps the second phase's cluster count.
@compute @workgroup_size(1)
fn post_finish() {
    let clusters_post = atomicLoad(&frame.post.instance_count);
    atomicStore(&frame.post.instance_count, min(clusters_post, params.cluster_capacity));
}
