// Where a cluster vertex is drawn (render::raster, "The geomorph"): shared by the raster's
// vertex shader (raster.wgsl), which draws it there, and the shading pass (shade.wgsl), which
// rebuilds the same triangles from the visibility buffer's ids. One copy of the rule, so what is
// shaded is what was drawn.
//
// After tree::NODE_WGSL, upload::VERTEX_WGSL, raster::INDICES_WGSL, raster::MORPH_WGSL and the
// slot bindings raster.rs generates (`slot_vertex`). The including shader declares the globals
// these functions read, under these names, at bindings of its own: `tree`
// (array<TileNode>), `remap_tables` (array<u32>), `morph` (array<MorphState>), `params`
// (RasterParams) and `view` (View).

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

// Vertices along a side of node `n`'s tile, as its slot holds them.
fn node_side(n: u32) -> u32 {
    return (1u << (tree[n].slot >> CLASS_SHIFT)) + 1u;
}

// Clusters along a side of a tile `side` vertices a side: the cluster ids' row length.
fn clusters_per_side(side: u32) -> u32 {
    return max((side - 1u) / CLUSTER_QUADS, 1u);
}

// Vertex (i, j) of the tile in `slot`, `side` vertices a row.
fn tile_vertex(slot: u32, side: u32, i: u32, j: u32) -> TileVertex {
    let size_class = slot >> CLASS_SHIFT;
    let slot_buffer = (slot >> 21u) & SLOT_BUFFER;
    let base = (slot & SLOT_INDEX) * side * side;
    return slot_vertex(size_class, slot_buffer, base + j * side + i);
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

// A cluster vertex where it is drawn, the visibility id (without the quad) of its cluster, and
// its other attributes carried by the same morph, so the shading pass reads each corner's
// vertex once, here: the drawn point is mix((tile (a) + tile (b)) / 2, tile (at), t) for the
// morph's two parents a and b, and exactly tile (at) where t is 1; the normal and roughness
// term are mixed the same way.
struct Placed {
    // World units.
    position: vec3<f32>,
    // Node and cluster bits.
    id: u32,
    // The tile vertex (i, j) drawn here, after the edge variant's remap.
    at: vec2<u32>,
    // How far the morph has run: 1 for a vertex where it is.
    t: f32,
    // The unit normal (xyz, as mixed: not renormalized) and the roughness term (w), carried
    // as the position is. The raster's vertex shader doesn't read it, so its compiler drops
    // the loads.
    attributes: vec4<f32>,
}

// Grid vertex `index` (of the static index buffer) of the cluster `cell` (Cluster.cell) of node
// `n`'s tile: remapped by the cluster's edge variant, read from the tile's slot and geomorphed.
//
// An odd vertex (odd i or j in its tile: not in the parent level's grid) moves from the
// midpoint of its two parents in decision 4's triangulation (the edge's ends, or the
// (i - 1, j - 1)-(i + 1, j + 1) diagonal) towards its position, by the smaller of the morph's
// time and distance factors. On a tile edge the factors are the seam's, from inputs both
// tiles share: the larger parent e, the larger own e, the less advanced stamp and the vertex's
// own bits, so both sides place it the same; where the neighbour is drawn finer the vertex
// stays where it is, as the finer side's even vertex there does; beside a coarser one the
// variant has already replaced it by an even vertex. Even vertices never move.
fn place_vertex(n: u32, cell: u32, index: u32) -> Placed {
    let node = tree[n];
    let cx = cell & 0xffu;
    let cy = (cell >> 8u) & 0xffu;
    let size_index = min(countTrailingZeros((cell >> 16u) & 0xffu), 3u);
    let variant = (cell >> 24u) & 0xfu;
    let finer = cell >> 28u;
    let local = remapped(size_index, variant, index);
    let side = node_side(n);
    let quads = side - 1u;
    let i = cx * CLUSTER_QUADS + local % CLUSTER_SIDE;
    let j = cy * CLUSTER_QUADS + local / CLUSTER_SIDE;
    let v = tile_vertex(node.slot, side, i, j);
    let p = v.position;
    let attributes = vec4<f32>(v.normal, v.roughness);
    let per = clusters_per_side(side);
    let id = (n << NODE_SHIFT) | ((cy * per + cx) << CLUSTER_SHIFT);
    let at = vec2<u32>(i, j);
    let still = Placed(p, id, at, 1.0, attributes);
    let odd_i = (i & 1u) == 1u;
    let odd_j = (j & 1u) == 1u;
    if !(odd_i || odd_j) {
        return still;
    }
    var e = parent_error(n);
    var own = tree[n].error;
    var time = morph_time(n);
    let edge = edge_of(i, j, quads);
    if edge < 4u {
        if ((finer >> edge) & 1u) != 0u {
            return still;
        }
        // From the tree, not `node`: indexing a local copy's array at run time is what the
        // crate's notes ("Shaders") rule out.
        let m = tree[n].neighbours[edge];
        if ((node.edges >> edge) & 1u) != 0u && m != NONE {
            let other = parent_error(m);
            e = select(max(e, other), -1.0, other < 0.0 || e < 0.0);
            own = max(own, tree[m].error);
            time = min(time, morph_time(m));
        }
    }
    if e < 0.0 {
        return still;
    }
    let t = min(time, morph_distance(e, own, p));
    if t >= 1.0 {
        return still;
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
    let va = tile_vertex(node.slot, side, a.x, a.y);
    let vb = tile_vertex(node.slot, side, b.x, b.y);
    let mid = (va.position + vb.position) * 0.5;
    let mid_attributes = (vec4<f32>(va.normal, va.roughness) + vec4<f32>(vb.normal, vb.roughness)) * 0.5;
    return Placed(mix(mid, p, t), id, at, t, mix(mid_attributes, attributes, t));
}
