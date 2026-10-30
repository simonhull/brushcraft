// The raster (render::raster): every visible cluster into the visibility buffer, one instanced
// indexed indirect draw per phase, and the geomorph that moves each cluster's odd vertices.
//
// After tree::NODE_WGSL, upload::VERTEX_WGSL, cull::CULL_WGSL, lod::CLUSTER_WGSL,
// raster::INDICES_WGSL, raster::MORPH_WGSL, the slot bindings raster.rs generates for the
// frame's slot buffers (group 1, read through `slot_vertex`) and place.wgsl, which places each
// vertex (the geomorph) from the globals declared here. One instance a cluster of the phase's
// list; the index buffer is the cluster's static triangles (render::indices), each index a grid
// vertex the cluster's edge variant remaps.
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

// Vertex `index` (a grid vertex of the static index buffer) of the phase's cluster `instance`,
// where it is drawn (place.wgsl's place_vertex).
fn cluster_point(instance: u32, index: u32) -> Placed {
    let c = clusters[instance];
    return place_vertex(c.node, c.cell, index);
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

