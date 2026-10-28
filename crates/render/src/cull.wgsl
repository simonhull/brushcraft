// Culling: the bounds of a tile, the view, and the tests on them (render::cull, CULL_WGSL).
//
// Declarations and functions only, no bindings: included ahead of the passes that use them
// (the bounds pass, the level choice). Every test here is conservative: it culls only what is
// certainly outside the view or facing away, and its CPU twin in cull.rs decides the same.

// A tile's bounds (render::cull::GpuBounds): 48 bytes, computed by the bounds pass when the
// tile's data arrives or changes.
struct TileBounds {
    // The centre of a sphere holding every vertex the tile draws, world units.
    center: vec3<f32>,
    // Its radius; negative while the bounds aren't computed (nothing is culled then).
    radius: f32,
    // The normal cone's axis, unit length: every triangle's normal is within the cone's
    // half-angle of it. Zero when the tile has no cone.
    axis: vec3<f32>,
    // The cosine of the cone's half-angle. At or below 0 (half a sphere or more, or no cone)
    // the cone culls nothing.
    cone_cos: f32,
    // The longest quad edge, world units: the quad cap's measure.
    quad_edge: f32,
    // Zero.
    reserved0: u32,
    // Zero.
    reserved1: u32,
    // Zero.
    reserved2: u32,
}

// A view (render::camera::GpuView): 192 bytes.
struct View {
    // World to clip coordinates: reversed Z, infinite far plane.
    view_proj: mat4x4<f32>,
    // The frustum's planes, (n, d) with dot(n, p) + d >= 0 inside, |n| = 1: left, right,
    // bottom, top, near, and the far plane at infinity (0, 0, 0, 1).
    planes: array<vec4<f32>, 6>,
    // The eye, w = 1.
    eye: vec4<f32>,
    // Width and height in pixels, pixels a length of 1 covers at a distance of 1, the near
    // plane's distance.
    viewport: vec4<f32>,
}

// Whether the sphere (center, radius) lies wholly on the outside of `plane`.
fn outside(plane: vec4<f32>, center: vec3<f32>, radius: f32) -> bool {
    return dot(plane.xyz, center) + plane.w < -radius;
}

// Whether any of the sphere (center, radius) may be inside the view's frustum.
fn in_frustum(view: View, center: vec3<f32>, radius: f32) -> bool {
    // Unrolled: a value array can't be indexed by a loop counter.
    return !(outside(view.planes[0], center, radius)
        || outside(view.planes[1], center, radius)
        || outside(view.planes[2], center, radius)
        || outside(view.planes[3], center, radius)
        || outside(view.planes[4], center, radius)
        || outside(view.planes[5], center, radius));
}

// Whether every triangle of the tile faces away from `eye`: each normal within the cone and
// each point within the sphere grown by `margin`. With v = center - eye at distance d and θ
// the angle from the axis to v, the smallest dot(n, p - eye) over the cone and sphere is
// d·cos(θ + α) - r, so the tile faces away when cos(θ + α) > r / d.
fn backfacing(bounds: TileBounds, eye: vec3<f32>, margin: f32) -> bool {
    if bounds.cone_cos <= 0.0 || bounds.radius < 0.0 {
        return false;
    }
    let v = bounds.center - eye;
    let d = length(v);
    let r = bounds.radius + margin;
    if d <= r {
        return false;
    }
    let cos_t = dot(v, bounds.axis) / d;
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let sin_a = sqrt(max(0.0, 1.0 - bounds.cone_cos * bounds.cone_cos));
    return cos_t * bounds.cone_cos - sin_t * sin_a > r / d;
}

// Pixels a length of `length` covers at its nearest, for a tile of these bounds grown by
// `margin`: from the sphere's nearest point (never nearer than the near plane).
fn projected(view: View, bounds: TileBounds, margin: f32, length: f32) -> f32 {
    let d = max(distance(bounds.center, view.eye.xyz) - bounds.radius - margin, view.viewport.w);
    return length * view.viewport.z / d;
}
