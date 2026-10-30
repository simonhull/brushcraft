// The shading pass (render::shade): each pixel of the visibility buffer rebuilt as the surface
// the raster drew there, handed to the material, written linear into the 16-bit float target.
//
// After tree::NODE_WGSL, upload::VERTEX_WGSL, cull::CULL_WGSL, lod::CLUSTER_WGSL,
// raster::INDICES_WGSL, raster::MORPH_WGSL, the slot bindings (group 1, `slot_vertex`),
// raster::PLACE_WGSL, raster::QUAD_WGSL, material::SURFACE_WGSL and the material's own source
// (group 2, `material`). One invocation a pixel: the compute entry point on desktop, the
// fragment one on tile-based GPUs (render::shade::ShadePath); both run shade_pixel.

// The tile tree.
@group(0) @binding(0) var<storage, read> tree: array<TileNode>;
// The edge variants' remap tables (render::indices::remap_table).
@group(0) @binding(1) var<storage, read> remap_tables: array<u32>;
// Per node, when its tile came into the cut: the raster's stamps, as its last frame left them.
@group(0) @binding(2) var<storage, read> morph: array<MorphState>;
// Per node, its tile's drawn edges (render::lod::DrawnEdges), kept by the raster's stamp pass.
@group(0) @binding(3) var<storage, read> drawn_edges: array<u32>;
// The raster's frame settings, as its last frame left them.
@group(0) @binding(4) var<uniform> params: RasterParams;
// The raster's view, as its last frame left it.
@group(0) @binding(5) var<uniform> view: View;
// The visibility buffer (render::raster::VISIBILITY_FORMAT).
@group(0) @binding(6) var ids: texture_2d<u32>;
// The shading's settings.
@group(0) @binding(7) var<uniform> shading: ShadeParams;

// The shading's settings (render::shade::GpuShadeParams): 64 bytes.
struct ShadeParams {
    // The view's up, unit length (w 0).
    up: vec4<f32>,
    // The view's right, unit length (w 0).
    right: vec4<f32>,
    // The wireframe's colour, linear (rgb), and its lines' width in pixels (w).
    wire: vec4<f32>,
    // WIREFRAME.
    flags: u32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// ShadeParams.flags: draw the wireframe over the shading.
const WIREFRAME: u32 = 1u;
// What the visibility buffer holds where nothing is drawn (render::raster::BACKGROUND).
const BACKGROUND: u32 = 0xffffffffu;
// Bits of a visibility id's cluster, below its node.
const CLUSTER_MASK: u32 = 0x3fu;
// Bits of its quad.
const QUAD_MASK: u32 = 0x3fu;

// A vertex's attributes besides its position: its unit normal (xyz) and roughness term (w).
fn attributes(n: u32, placed: Placed) -> vec4<f32> {
    let slot = tree[n].slot;
    let side = node_side(n);
    let v = tile_vertex(slot, side, placed.at.x, placed.at.y);
    let own = vec4<f32>(v.normal, v.roughness);
    if placed.t >= 1.0 {
        return own;
    }
    // Morphing: carried from its parents' midpoint the same way as its position.
    let a = tile_vertex(slot, side, placed.from_a.x, placed.from_a.y);
    let b = tile_vertex(slot, side, placed.from_b.x, placed.from_b.y);
    let mid = (vec4<f32>(a.normal, a.roughness) + vec4<f32>(b.normal, b.roughness)) * 0.5;
    return mix(mid, own, placed.t);
}

// The pixel position (x right, y down, from the top-left corner) clip point `c` projects to;
// only for a point in front of the eye (c.w > 0).
fn screen(c: vec4<f32>) -> vec2<f32> {
    let ndc = c.xy / c.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * view.viewport.x, (0.5 - ndc.y * 0.5) * view.viewport.y);
}

// The homogeneous coordinates, (x, y, w) of clip space, of pixel position `p`.
fn homogeneous(p: vec2<f32>) -> vec3<f32> {
    let x = p.x / view.viewport.x * 2.0 - 1.0;
    let y = 1.0 - p.y / view.viewport.y * 2.0;
    return vec3<f32>(x, y, 1.0);
}

// The perspective-correct barycentric coordinates, at the pixel position whose homogeneous
// coordinates are `p`, of the triangle whose corners have clip coordinates (x, y, w) `c0`,
// `c1`, `c2`: the weights b with sum 1 such that the corners mixed by b project to `p`, by
// Cramer's rule. They hold for corners behind the eye too, where projecting to the screen
// doesn't; the point the eye sees has them all in 0..1.
fn barycentric(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let b = vec3<f32>(dot(p, cross(c1, c2)), dot(p, cross(c2, c0)), dot(p, cross(c0, c1)));
    let sum = b.x + b.y + b.z;
    if sum == 0.0 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return b / sum;
}

// A pixel of the drawn surface, rebuilt.
struct Rebuilt {
    // Whether the pixel holds the surface.
    drawn: bool,
    // The point, relative to the eye, world units.
    relative: vec3<f32>,
    // The unit normal.
    normal: vec3<f32>,
    // The roughness term.
    roughness: f32,
    // The distance from the pixel's centre to the nearest edge of its triangle, pixels.
    edge: f32,
}

// Rebuilds the triangle the raster drew at `pixel` and the surface at the pixel's centre.
//
// The visibility id names the node, the cluster and the quad; the node's drawn edges give the
// cluster's edge variant and finer edges (cluster_cell), and place_vertex places the quad's four
// corners exactly as the raster's vertex shader did (the same function, the same buffers as its
// frame left them). Which of the quad's triangles holds the pixel is quad_triangle's answer
// (the side of the diagonal), as projected to the screen, confirmed by the barycentric
// coordinates: where they put the centre outside it and the other triangle holds it better
// (the rasterizer's tie rule on the diagonal; or a corner behind the eye, which the screen
// can't hold), it is the other. The point, normal and roughness term are the corners' mixed by
// the perspective-correct barycentric coordinates of the pixel's centre, as the rasterizer
// interpolates.
fn rebuild(pixel: vec2<u32>) -> Rebuilt {
    var out = Rebuilt(false, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 1.0, 1e9);
    let id = textureLoad(ids, pixel, 0).r;
    if id == BACKGROUND {
        return out;
    }
    let n = id >> NODE_SHIFT;
    if n >= arrayLength(&tree) {
        return out;
    }
    let cluster = (id >> CLUSTER_SHIFT) & CLUSTER_MASK;
    let quad = id & QUAD_MASK;
    let side = node_side(n);
    let per = clusters_per_side(side);
    let cell = cluster_cell(cluster % per, cluster / per, side - 1u, drawn_edges[n]);
    let base = (quad / CLUSTER_QUADS) * CLUSTER_SIDE + quad % CLUSTER_QUADS;
    // The quad's corners: p00, p10, p11, p01.
    let placed = array<Placed, 4>(
        place_vertex(n, cell, base),
        place_vertex(n, cell, base + 1u),
        place_vertex(n, cell, base + 1u + CLUSTER_SIDE),
        place_vertex(n, cell, base + CLUSTER_SIDE),
    );
    var relative: array<vec3<f32>, 4>;
    var clip: array<vec4<f32>, 4>;
    var in_front = true;
    for (var k = 0u; k < 4u; k++) {
        relative[k] = placed[k].position - view.eye.xyz;
        clip[k] = view.view_proj * vec4<f32>(relative[k], 1.0);
        in_front = in_front && clip[k].w > 0.0;
    }
    let centre = vec2<f32>(pixel) + vec2<f32>(0.5);
    let p = homogeneous(centre);
    // Triangle 0 is (p00, p10, p11), triangle 1 (p00, p11, p01).
    var second = 0u;
    if in_front {
        second = quad_triangle(
            screen(clip[0]),
            screen(clip[1]),
            screen(clip[2]),
            screen(clip[3]),
            centre,
        );
    }
    var corner = vec3<u32>(0u, 1u + second, 2u + second);
    var c0 = clip[corner.x].xyw;
    var c1 = clip[corner.y].xyw;
    var c2 = clip[corner.z].xyw;
    var b = barycentric(c0, c1, c2, p);
    // A pixel centre within rounding of the diagonal can be the other triangle's by the
    // rasterizer's tie rule (seen on grazing quads, where the other plane extrapolated lies far
    // off); and with a corner behind the eye there is no screen to test on. Where the chosen
    // triangle doesn't hold the centre and the other holds it better, it is the other.
    let outside = min(min(b.x, b.y), b.z);
    if outside < 0.0 {
        let other = vec3<u32>(0u, 2u - second, 3u - second);
        let o0 = clip[other.x].xyw;
        let o1 = clip[other.y].xyw;
        let o2 = clip[other.z].xyw;
        let ob = barycentric(o0, o1, o2, p);
        if min(min(ob.x, ob.y), ob.z) > outside {
            corner = other;
            c0 = o0;
            c1 = o1;
            c2 = o2;
            b = ob;
        }
    }
    out.drawn = true;
    out.relative = relative[corner.x] * b.x + relative[corner.y] * b.y + relative[corner.z] * b.z;
    let a0 = attributes(n, placed[corner.x]);
    let a1 = attributes(n, placed[corner.y]);
    let a2 = attributes(n, placed[corner.z]);
    let mixed = a0 * b.x + a1 * b.y + a2 * b.z;
    let reach = sqrt(dot(mixed.xyz, mixed.xyz));
    if reach > 0.0 {
        out.normal = mixed.xyz / reach;
    }
    out.roughness = clamp(mixed.w, 0.0, 1.0);
    // The distance to each edge, in pixels: a coordinate over how fast it changes across the
    // screen (the edge opposite corner k is where b[k] is 0).
    let bx = barycentric(c0, c1, c2, homogeneous(centre + vec2<f32>(1.0, 0.0))) - b;
    let by = barycentric(c0, c1, c2, homogeneous(centre + vec2<f32>(0.0, 1.0))) - b;
    for (var k = 0u; k < 3u; k++) {
        let rate = length(vec2<f32>(bx[k], by[k]));
        if rate > 0.0 {
            out.edge = min(out.edge, abs(b[k]) / rate);
        }
    }
    return out;
}

// The linear colour of `pixel`, alpha 1 where the surface is drawn and 0 (all zero) where it
// isn't, for the tone map to leave the frame's clear colour there.
fn shade_pixel(pixel: vec2<u32>) -> vec4<f32> {
    let surface = rebuild(pixel);
    if !surface.drawn {
        return vec4<f32>(0.0);
    }
    let position = surface.relative + view.eye.xyz;
    let distance = length(surface.relative);
    var to_eye = vec3<f32>(0.0, 0.0, 1.0);
    if distance > 0.0 {
        to_eye = -surface.relative / distance;
    }
    let point = Surface(position, surface.normal, surface.roughness, vec4<f32>(0.0));
    let eye = Eye(to_eye, shading.up.xyz, shading.right.xyz);
    var colour = material(point, eye);
    if (shading.flags & WIREFRAME) != 0u {
        // A line of the wireframe's width centred on each edge, its sides a pixel's ramp.
        let half = shading.wire.w * 0.5;
        let cover = 1.0 - smoothstep(half - 0.5, half + 0.5, surface.edge);
        colour = mix(colour, shading.wire.rgb, cover);
    }
    return vec4<f32>(colour, 1.0);
}

// The 16-bit float target the compute entry point writes.
@group(3) @binding(0) var target_image: texture_storage_2d<rgba16float, write>;

// One pixel, into the target: the desktop path.
@compute @workgroup_size(8, 8)
fn shade_compute(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(ids);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    textureStore(target_image, id.xy, shade_pixel(id.xy));
}

// A triangle covering the whole frame, from three vertex indices: (-1, -1), (3, -1), (-1, 3).
@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index & 1u) * 4u) - 1.0;
    let y = f32((index >> 1u) * 4u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// One pixel, as a fragment of a fullscreen triangle drawn into the target: the tile-based
// GPUs' path (finished in M1c).
@fragment
fn shade_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return shade_pixel(vec2<u32>(position.xy));
}
