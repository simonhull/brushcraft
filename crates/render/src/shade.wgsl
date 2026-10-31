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
// doesn't; the point the eye sees has them all in 0..1. A degenerate triangle (an edge variant
// collapses one of each border quad's two) has none: its weights' sum is zero, or rounding
// away from it, and it gets NO_TRIANGLE, which no triangle's coordinates are ever below, so it
// never holds a pixel.
fn barycentric(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let b = vec3<f32>(dot(p, cross(c1, c2)), dot(p, cross(c2, c0)), dot(p, cross(c0, c1)));
    let sum = b.x + b.y + b.z;
    let scale = abs(b.x) + abs(b.y) + abs(b.z);
    if !(abs(sum) > DEGENERATE * scale) {
        return NO_TRIANGLE;
    }
    return b / sum;
}

// The coordinates of a degenerate triangle: lower than any real triangle's.
const NO_TRIANGLE: vec3<f32> = vec3<f32>(-3.4e38, -3.4e38, -3.4e38);
// The weights' sum, against the sum of their sizes, below which a triangle is degenerate.
const DEGENERATE: f32 = 1e-6;

// The smallest of `b`: how far inside its triangle a point is (negative outside).
fn inside(b: vec3<f32>) -> f32 {
    return min(min(b.x, b.y), b.z);
}

// Whether the triangle with corners `r0`, `r1`, `r2` (relative to the eye) faces the eye: its
// corners run counter-clockwise seen from the eye, as the raster's front faces do. Unlike a test
// on the screen, this holds for corners behind the eye.
fn facing(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>) -> bool {
    return dot(cross(r1 - r0, r2 - r0), r0) < 0.0;
}

// The reversed-Z depth, at barycentric coordinates `b`, of the triangle whose corners have clip
// coordinates `c0`, `c1`, `c2`; negative where the point lies behind the eye (the ray meets the
// triangle's plane only there).
fn depth_of(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, b: vec3<f32>) -> f32 {
    let c = c0 * b.x + c1 * b.y + c2 * b.z;
    if !(c.w > 0.0) {
        return -1.0;
    }
    return c.z / c.w;
}

// One of a quad's two triangles, at a pixel's centre.
struct Candidate {
    // Its barycentric coordinates there (NO_TRIANGLE for a degenerate one).
    b: vec3<f32>,
    // Whether it faces the eye: the raster culls it otherwise.
    faces: bool,
    // Its depth there: negative behind the eye.
    depth: f32,
}

// Whether the raster could have drawn `t` at the centre at all: facing the eye, the centre in
// front of the eye, and not degenerate.
fn drawable(t: Candidate) -> bool {
    return t.faces && t.depth > 0.0 && inside(t.b) > -1e30;
}

// Whether `other` is the triangle the raster drew rather than `chosen`: it could have been drawn
// and `chosen` couldn't; or `chosen` doesn't hold the centre and `other` holds it better; or both
// hold it and `other` is nearer (the depth test).
fn better(other: Candidate, chosen: Candidate) -> bool {
    if !drawable(other) {
        return false;
    }
    if !drawable(chosen) {
        return true;
    }
    let held = inside(chosen.b) >= 0.0;
    if !held {
        return inside(other.b) > inside(chosen.b);
    }
    return inside(other.b) >= 0.0 && other.depth > chosen.depth;
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
    // Whether a corner of the pixel's quad lies behind the eye (the triangle was chosen without
    // the screen).
    behind: bool,
}

// Rebuilds the triangle the raster drew at `pixel` and the surface at the pixel's centre.
//
// The visibility id names the node, the cluster and the quad; the node's drawn edges give the
// cluster's edge variant and finer edges (cluster_cell), and place_vertex places the quad's four
// corners exactly as the raster's vertex shader did (the same function, the same buffers as its
// frame left them). Which of the quad's triangles holds the pixel is quad_triangle's answer
// (the side of the diagonal), as projected to the screen, confirmed by the barycentric
// coordinates: where they put the centre outside it and the other triangle holds it better
// (the rasterizer's tie rule on the diagonal), it is the other. With a corner behind the eye,
// which the screen can't hold, it is the triangle holding the centre best. A triangle an edge
// variant collapsed holds nothing (barycentric's NO_TRIANGLE), so it is never chosen. The point,
// normal and roughness term are the corners' mixed by the perspective-correct barycentric
// coordinates of the pixel's centre, as the rasterizer interpolates.
fn rebuild(pixel: vec2<u32>) -> Rebuilt {
    var out = Rebuilt(false, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 1.0, 1e9, false);
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
    // The quad's corners: p00, p10, p11, p01. Each is a variable of its own, picked below by
    // `select` or a branch: never an array indexed at run time (the crate's notes, "Shaders").
    let placed0 = place_vertex(n, cell, base);
    let placed1 = place_vertex(n, cell, base + 1u);
    let placed2 = place_vertex(n, cell, base + 1u + CLUSTER_SIDE);
    let placed3 = place_vertex(n, cell, base + CLUSTER_SIDE);
    let relative0 = placed0.position - view.eye.xyz;
    let relative1 = placed1.position - view.eye.xyz;
    let relative2 = placed2.position - view.eye.xyz;
    let relative3 = placed3.position - view.eye.xyz;
    let clip0 = view.view_proj * vec4<f32>(relative0, 1.0);
    let clip1 = view.view_proj * vec4<f32>(relative1, 1.0);
    let clip2 = view.view_proj * vec4<f32>(relative2, 1.0);
    let clip3 = view.view_proj * vec4<f32>(relative3, 1.0);
    let in_front = clip0.w > 0.0 && clip1.w > 0.0 && clip2.w > 0.0 && clip3.w > 0.0;
    let centre = vec2<f32>(pixel) + vec2<f32>(0.5);
    let p = homogeneous(centre);
    // Triangle 0 is (p00, p10, p11), triangle 1 (p00, p11, p01).
    var first = Candidate(
        barycentric(clip0.xyw, clip1.xyw, clip2.xyw, p),
        facing(relative0, relative1, relative2),
        0.0,
    );
    var second_tri = Candidate(
        barycentric(clip0.xyw, clip2.xyw, clip3.xyw, p),
        facing(relative0, relative2, relative3),
        0.0,
    );
    first.depth = depth_of(clip0, clip1, clip2, first.b);
    second_tri.depth = depth_of(clip0, clip2, clip3, second_tri.b);
    var second = select(0u, 1u, inside(second_tri.b) > inside(first.b));
    if in_front {
        second = quad_triangle(screen(clip0), screen(clip1), screen(clip2), screen(clip3), centre);
    }
    // The triangle on the diagonal's side, and the other.
    var chosen = first;
    var other = second_tri;
    if second == 1u {
        chosen = second_tri;
        other = first;
    }
    // The rasterizer drew a triangle facing the eye whose visible part holds the centre, the
    // nearer where two do (a quad folded over itself, seen grazing, puts both on the pixel).
    // The diagonal's side can miss that: a centre within rounding of the diagonal goes by the
    // rasterizer's tie rule, a fold puts the centre on the side of a triangle hidden behind the
    // other or facing away, and with a corner behind the eye there is no screen to test on.
    if better(other, chosen) {
        second = 1u - second;
        chosen = other;
    }
    // The triangle's corners: p00, then p10 and p11 (triangle 0) or p11 and p01 (triangle 1).
    let on_second = second == 1u;
    let c0 = clip0.xyw;
    let c1 = select(clip1, clip2, on_second).xyw;
    let c2 = select(clip2, clip3, on_second).xyw;
    let b = chosen.b;
    out.drawn = true;
    out.behind = !in_front;
    out.relative = relative0 * b.x + select(relative1, relative2, on_second) * b.y
        + select(relative2, relative3, on_second) * b.z;
    var corner1 = placed1;
    var corner2 = placed2;
    if on_second {
        corner1 = placed2;
        corner2 = placed3;
    }
    let a0 = placed0.attributes;
    let a1 = corner1.attributes;
    let a2 = corner2.attributes;
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
