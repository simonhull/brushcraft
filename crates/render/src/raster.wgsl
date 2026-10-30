// The raster (render::raster): every visible cluster into the visibility buffer, one instanced
// indexed indirect draw per phase, and the geomorph that moves each cluster's odd vertices.
//
// After tree::NODE_WGSL, upload::VERTEX_WGSL, cull::CULL_WGSL,
// lod::CLUSTER_WGSL, raster::INDICES_WGSL and the slot bindings raster.rs generates for the
// frame's slot buffers (group 1, read through `slot_vertex`). One instance a cluster of the
// phase's list; the index buffer is the cluster's static triangles (render::indices), each
// index a grid vertex the cluster's edge variant remaps.
//
// The visibility buffer holds, per pixel, the node of the tile tree drawn there (bits 12..32),
// the cluster within its tile (6..12) and the quad within the cluster (0..6, 8y + x): stable
// while the camera is still, since a node stays its tile's while resident. 0xffffffff where
// nothing is drawn. The quad comes flat from each triangle's provoking vertex, its first: both
// triangles of quad (x, y) start at its p00, grid vertex (x, y) of the static index buffer, so
// every vertex hands on the quad its unremapped grid index starts (a vertex an edge variant
// collapses still provokes its quad's surviving triangle). Which of the quad's two triangles a
// pixel is in, the shading pass tells by the diagonal (raster.rs, QUAD_WGSL).

// The tile tree.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// The phase's visible clusters (render::lod::Lod::clusters).
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
// The edge variants' remap tables (render::indices::remap_table).
@group(0) @binding(2) var<storage, read> remap_tables: array<u32>;
// Per node, when its tile last came into the cut (the stamp pass, morph.wgsl).
@group(0) @binding(3) var<storage, read> morph: array<MorphState>;
// The frame's settings.
@group(0) @binding(4) var<uniform> params: RasterParams;
// The frame's view.
@group(0) @binding(5) var<uniform> view: View;

// What the vertex shader hands the fragment shader.
struct ClusterOut {
    // The vertex in clip space.
    @builtin(position) clip: vec4<f32>,
    // The visibility id of the quad this vertex starts (when it is a triangle's first).
    @location(0) @interpolate(flat, first) id: u32,
}

// Where the visibility id's node starts.
const NODE_SHIFT: u32 = 12u;
// Where its cluster within the tile starts.
const CLUSTER_SHIFT: u32 = 6u;
// Bits of TileNode.slot below the size class.
const CLASS_SHIFT: u32 = 29u;
// TileNode.slot's buffer within its class, after shifting.
const SLOT_BUFFER: u32 = 0xffu;
// TileNode.slot's index within its buffer.
const SLOT_INDEX: u32 = 0x1fffffu;
// Quads a full cluster has a side.
const CLUSTER_QUADS: u32 = 8u;

// Vertex (i, j) of the tile in `slot`, `side` vertices a row.
fn tile_position(slot: u32, side: u32, i: u32, j: u32) -> vec3<f32> {
    let size_class = slot >> CLASS_SHIFT;
    let slot_buffer = (slot >> 21u) & SLOT_BUFFER;
    let base = (slot & SLOT_INDEX) * side * side;
    return slot_vertex(size_class, slot_buffer, base + j * side + i).position;
}

// The `e` a tile's level change trades: its parent's, the bound on how far the parent's
// surface (where the morph starts) lies from the finest; negative without a resident parent
// (a level-0 tile), which never morphs.
fn parent_error(n: u32) -> f32 {
    let parent = tree[n].parent;
    if parent == NONE {
        return -1.0;
    }
    return tree[parent].error;
}

// How far node `n`'s morph has run in time, 0 to 1: from its stamp, over the morph's
// duration; 1 when the stamp isn't this frame's (the node isn't in the cut: never so for a
// drawn tile, but kept safe) or isn't its tile's.
fn morph_time(n: u32) -> f32 {
    let s = morph[n];
    if s.frame != params.frame || s.face != tree[n].face || s.place != tree[n].place {
        return 1.0;
    }
    if params.duration == 0u {
        return 1.0;
    }
    let elapsed = params.now - s.start;
    return min(f32(elapsed) / f32(params.duration), 1.0);
}

// How far the morph has run by distance, 0 to 1, at `p`, for the parent's bound `parent` and
// the tile's own `own` (world units): t = (x - r) / (x - y) clamped, with x and y those bounds
// projected at `p` in pixels and r the threshold (params.ramp), so that the drawn error,
// (1 - t) x + t y, stays at r while it can, and the morph ends as the tile's own bound reaches r
// (user decision, 2026-10-29). Where x - y is 0 or less (a parent bound no larger than the
// tile's own), 1 if y is within r, else 0. 1 with no threshold, or at and behind the near
// plane.
fn morph_distance(parent: f32, own: f32, p: vec3<f32>) -> f32 {
    let r = params.ramp;
    if r <= 0.0 {
        return 1.0;
    }
    let depth = dot(view.planes[4].xyz, p - view.eye.xyz);
    if depth <= view.viewport.w {
        return 1.0;
    }
    let x = parent * view.viewport.z / depth;
    let y = own * view.viewport.z / depth;
    if x - y <= 0.0 {
        return select(0.0, 1.0, y <= r);
    }
    return clamp((x - r) / (x - y), 0.0, 1.0);
}

// The edge of a tile of `quads` quads a side that its vertex (i, j) lies on, 4 inside. Only
// for odd vertices, which are never corners, so on one edge at most.
fn edge_of(i: u32, j: u32, quads: u32) -> u32 {
    if j == 0u {
        return 0u;
    }
    if i == quads {
        return 1u;
    }
    if j == quads {
        return 2u;
    }
    if i == 0u {
        return 3u;
    }
    return 4u;
}

// A cluster vertex where it is drawn, and the visibility id (without the triangle) of its
// cluster.
struct Placed {
    // World units.
    position: vec3<f32>,
    // Node and cluster bits.
    id: u32,
}

// Vertex `index` (a grid vertex of the static index buffer) of the phase's cluster `instance`:
// remapped by the cluster's edge variant, read from its tile's slot and geomorphed.
//
// An odd vertex (odd i or j in its tile: not in the parent level's grid) moves from the
// midpoint of its two parents in decision 4's triangulation (the edge's ends, or the
// (i - 1, j - 1)-(i + 1, j + 1) diagonal) towards its position, by the smaller of the morph's
// time and distance factors. On a tile edge the factors are the seam's, from inputs both
// tiles share: the larger parent e, the larger own e, the less advanced stamp and the vertex's
// own bits, so both sides place it the same; where the neighbour is drawn finer the vertex
// stays where it is, as the finer side's even vertex there does; beside a coarser one the
// variant has already replaced it by an even vertex. Even vertices never move.
fn cluster_point(instance: u32, index: u32) -> Placed {
    let c = clusters[instance];
    let n = c.node;
    let node = tree[n];
    let cx = c.cell & 0xffu;
    let cy = (c.cell >> 8u) & 0xffu;
    let size_index = min(countTrailingZeros((c.cell >> 16u) & 0xffu), 3u);
    let variant = (c.cell >> 24u) & 0xfu;
    let finer = c.cell >> 28u;
    let local = remapped(size_index, variant, index);
    let side = (1u << (node.slot >> CLASS_SHIFT)) + 1u;
    let quads = side - 1u;
    let i = cx * CLUSTER_QUADS + local % CLUSTER_SIDE;
    let j = cy * CLUSTER_QUADS + local / CLUSTER_SIDE;
    let p = tile_position(node.slot, side, i, j);
    let per = max(quads / CLUSTER_QUADS, 1u);
    let id = (n << NODE_SHIFT) | ((cy * per + cx) << CLUSTER_SHIFT);
    let odd_i = (i & 1u) == 1u;
    let odd_j = (j & 1u) == 1u;
    if !(odd_i || odd_j) {
        return Placed(p, id);
    }
    var e = parent_error(n);
    var own = tree[n].error;
    var time = morph_time(n);
    let edge = edge_of(i, j, quads);
    if edge < 4u {
        if ((finer >> edge) & 1u) != 0u {
            return Placed(p, id);
        }
        let m = node.neighbours[edge];
        if ((node.edges >> edge) & 1u) != 0u && m != NONE {
            let other = parent_error(m);
            e = select(max(e, other), -1.0, other < 0.0 || e < 0.0);
            own = max(own, tree[m].error);
            time = min(time, morph_time(m));
        }
    }
    if e < 0.0 {
        return Placed(p, id);
    }
    let t = min(time, morph_distance(e, own, p));
    if t >= 1.0 {
        return Placed(p, id);
    }
    var a = vec2<u32>(i, j);
    var b = vec2<u32>(i, j);
    if odd_i {
        a.x -= 1u;
        b.x += 1u;
    }
    if odd_j {
        a.y -= 1u;
        b.y += 1u;
    }
    let mid = (tile_position(node.slot, side, a.x, a.y) + tile_position(node.slot, side, b.x, b.y)) * 0.5;
    return Placed(mix(mid, p, t), id);
}

// Places a cluster vertex in clip space, relative to the eye, with its cluster's id.
@vertex
fn cluster_vertex(
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) instance: u32,
) -> ClusterOut {
    let placed = cluster_point(instance, index);
    let clip = view.view_proj * vec4<f32>(placed.position - view.eye.xyz, 1.0);
    // The quad whose p00 is this grid vertex, unremapped; the last row and column start
    // none, and never provoke.
    let u = min(index % CLUSTER_SIDE, CLUSTER_QUADS - 1u);
    let v = min(index / CLUSTER_SIDE, CLUSTER_QUADS - 1u);
    return ClusterOut(clip, placed.id | (v * CLUSTER_QUADS + u));
}

// The visibility id, as the triangle's first vertex gave it.
@fragment
fn visibility(vertex_out: ClusterOut) -> @location(0) u32 {
    return vertex_out.id;
}

