//! CPU twins of the culling passes, for the tests.
//!
//! The same arithmetic as `cull.wgsl`, `bounds.wgsl`, `pyramid.wgsl` and the level choice's
//! occlusion and finest-normal cone tests, in `f32`, plus helpers that put a synthetic depth
//! image on the GPU and read a pyramid back.

use geom::FinestNormalCone;
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, CompareFunction, DepthStencilState,
    Extent3d, FragmentState, MultisampleState, Origin3d, PipelineCompilationOptions,
    PrimitiveState, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, StencilState,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, VertexState,
};

use super::GpuBounds;
use crate::camera::GpuView;
use crate::gpu::Gpu;
use crate::upload::GpuVertex;

/// `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a · b`.
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `|a - b|`.
pub(crate) fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

/// The unit normal of triangle (a, b, c), or zero for a degenerate one.
pub(crate) fn face_normal(first: [f32; 3], second: [f32; 3], third: [f32; 3]) -> [f32; 3] {
    let normal = cross(sub(second, first), sub(third, first));
    let length = dot(normal, normal).sqrt();
    if length > 1e-30 {
        normal.map(|x| x / length)
    } else {
        [0.0; 3]
    }
}

/// The two triangles of quad `(i, j)` of a `side`-vertex grid of `points`, split along
/// `(i, j)–(i + 1, j + 1)`.
pub(crate) fn quad_triangles(
    points: &[[f32; 3]],
    side: usize,
    i: usize,
    j: usize,
) -> [[[f32; 3]; 3]; 2] {
    let at = |x: usize, y: usize| points[y * side + x];
    let (p00, p10, p11, p01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
    [[p00, p10, p11], [p00, p11, p01]]
}

/// A grid vertex, `(column, row)`.
pub(crate) type Grid = (i32, i32);

/// The tile's extra triangles on a `side`-vertex grid.
///
/// As grid vertices counter-clockwise in the grid: the parent triangulation, then each edge
/// variant's fans (`bounds.wgsl`'s `extra_normal`, in the same order).
///
/// # Panics
///
/// If `side` is 0 or past `i32`.
pub(crate) fn extra_triangles(side: usize) -> Vec<[Grid; 3]> {
    let quads = i32::try_from(side - 1).expect("small");
    let half = quads / 2;
    let ccw = |[g0, g1, g2]: [Grid; 3]| {
        let cross = (g1.0 - g0.0) * (g2.1 - g0.1) - (g1.1 - g0.1) * (g2.0 - g0.0);
        if cross < 0 {
            [g0, g2, g1]
        } else {
            [g0, g1, g2]
        }
    };
    let mut out = Vec::new();
    for by in 0..half {
        for bx in 0..half {
            let p00 = (2 * bx, 2 * by);
            let p22 = (p00.0 + 2, p00.1 + 2);
            out.push(ccw([p00, (p00.0 + 2, p00.1), p22]));
            out.push(ccw([p00, p22, (p00.0, p00.1 + 2)]));
        }
    }
    for edge in 0..4 {
        for s in 0..half {
            let o = 2 * s + 1;
            let (a, b, inward) = match edge {
                0 => ((o - 1, 0), (o + 1, 0), (0, 1)),
                1 => ((quads, o - 1), (quads, o + 1), (-1, 0)),
                2 => ((o - 1, quads), (o + 1, quads), (0, -1)),
                _ => ((0, o - 1), (0, o + 1), (1, 0)),
            };
            let step = |g: Grid| (g.0 + inward.0, g.1 + inward.1);
            let m = step((i32::midpoint(a.0, b.0), i32::midpoint(a.1, b.1)));
            out.push(ccw([a, b, m]));
            out.push(ccw([a, m, step(a)]));
            out.push(ccw([b, step(b), m]));
        }
    }
    out
}

/// A grid vertex of a morph triangle, and whether it stands where its morph starts (the
/// midpoint of its decision-4 parents) rather than where it is.
pub(crate) type MorphCorner = (Grid, bool);

/// The tile's morph triangles on a `side`-vertex grid, in `bounds.wgsl`'s `morph_normal` order.
///
/// Each own triangle and each fan triangle with moving vertices six times, once per mix of its
/// vertices at their start (bit `k` of 1 to 6 for vertex `k`) and at their end, counter-clockwise
/// in the grid. None for a single quad. An even vertex's start is where it is.
///
/// # Panics
///
/// If `side` is 0 or past `i32`.
pub(crate) fn morph_triangles(side: usize) -> Vec<[MorphCorner; 3]> {
    let quads = i32::try_from(side - 1).expect("small");
    if quads < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut push = |[g0, g1, g2]: [Grid; 3]| {
        for starts in 1..=6_u8 {
            out.push([
                (g0, starts & 1 != 0),
                (g1, starts & 2 != 0),
                (g2, starts & 4 != 0),
            ]);
        }
    };
    for j in 0..quads {
        for i in 0..quads {
            let (p00, p10, p11, p01) = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1));
            push([p00, p10, p11]);
            push([p00, p11, p01]);
        }
    }
    let half = quads / 2;
    for edge in 0..4 {
        for s in 0..half {
            let o = 2 * s + 1;
            let (a, b, inward) = match edge {
                0 => ((o - 1, 0), (o + 1, 0), (0, 1)),
                1 => ((quads, o - 1), (quads, o + 1), (-1, 0)),
                2 => ((o - 1, quads), (o + 1, quads), (0, -1)),
                _ => ((0, o - 1), (0, o + 1), (1, 0)),
            };
            let step = |g: Grid| (g.0 + inward.0, g.1 + inward.1);
            let m = step((i32::midpoint(a.0, b.0), i32::midpoint(a.1, b.1)));
            push([a, m, step(a)]);
            push([b, step(b), m]);
        }
    }
    out
}

/// Where a morph corner stands on a `side`-vertex grid of `points`.
///
/// # Panics
///
/// If the corner is off the grid.
pub(crate) fn morph_corner(points: &[[f32; 3]], side: usize, (g, start): MorphCorner) -> [f32; 3] {
    let at = |(x, y): Grid| {
        points[usize::try_from(y).expect("on the grid") * side
            + usize::try_from(x).expect("on the grid")]
    };
    if !start {
        return at(g);
    }
    let d = (g.0 & 1, g.1 & 1);
    let (low, high) = (at((g.0 - d.0, g.1 - d.1)), at((g.0 + d.0, g.1 + d.1)));
    #[expect(
        clippy::manual_midpoint,
        reason = "the bounds pass's arithmetic, (a + b) · 0.5, bit for bit"
    )]
    let mid = [0, 1, 2].map(|k| (low[k] + high[k]) * 0.5);
    mid
}

/// The unit normal of a morph triangle, counter-clockwise in the grid as the bounds pass takes
/// it.
pub(crate) fn morph_normal(points: &[[f32; 3]], side: usize, t: [MorphCorner; 3]) -> [f32; 3] {
    let [p0, p1, p2] = t.map(|c| morph_corner(points, side, c));
    let [(g0, _), (g1, _), (g2, _)] = t;
    let cross = (g1.0 - g0.0) * (g2.1 - g0.1) - (g1.1 - g0.1) * (g2.0 - g0.0);
    if cross < 0 {
        face_normal(p0, p2, p1)
    } else {
        face_normal(p0, p1, p2)
    }
}

/// The bounds of a tile of `side` × `side` vertices, as the bounds pass computes them.
///
/// # Panics
///
/// If `vertices` holds fewer than `side²`.
pub(crate) fn bounds_of(vertices: &[GpuVertex], side: usize) -> GpuBounds {
    bounds_with(vertices, side, true)
}

/// [`bounds_of`], its cone holding the morph triangles only if `morphs` (without, as the first
/// cut of the raster had it, for the tests' bite check).
///
/// # Panics
///
/// If `vertices` holds fewer than `side²`.
pub(crate) fn bounds_with(vertices: &[GpuVertex], side: usize, morphs: bool) -> GpuBounds {
    let points: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut edge = 0.0_f32;
    for j in 0..side {
        for i in 0..side {
            let p = points[j * side + i];
            for axis in 0..3 {
                lo[axis] = lo[axis].min(p[axis]);
                hi[axis] = hi[axis].max(p[axis]);
            }
            if i + 1 < side {
                edge = edge.max(distance(p, points[j * side + i + 1]));
            }
            if j + 1 < side {
                edge = edge.max(distance(p, points[(j + 1) * side + i]));
            }
        }
    }
    let center = [0, 1, 2].map(|a| f32::midpoint(lo[a], hi[a]));
    let radius = points
        .iter()
        .map(|&p| distance(p, center))
        .fold(0.0, f32::max);
    let quads = side - 1;
    let at = |(x, y): Grid| {
        points[usize::try_from(y).expect("on the grid") * side
            + usize::try_from(x).expect("on the grid")]
    };
    let extras = extra_triangles(side)
        .into_iter()
        .map(|[a, b, c]| face_normal(at(a), at(b), at(c)));
    let morph_set = if morphs {
        morph_triangles(side)
    } else {
        Vec::new()
    };
    let morph_normals = morph_set
        .into_iter()
        .map(|t| morph_normal(&points, side, t));
    let normals: Vec<[f32; 3]> = (0..quads * quads)
        .flat_map(|q| quad_triangles(&points, side, q % quads, q / quads))
        .map(|[a, b, c]| face_normal(a, b, c))
        .chain(extras)
        .chain(morph_normals)
        .collect();
    let sum = normals
        .iter()
        .fold([0.0; 3], |s, n| [s[0] + n[0], s[1] + n[1], s[2] + n[2]]);
    let length = dot(sum, sum).sqrt();
    let axis = if length > 1e-20 {
        sum.map(|x| x / length)
    } else {
        [0.0; 3]
    };
    let cosine = normals
        .iter()
        .filter(|n| dot(**n, **n) > 0.0)
        .map(|&n| dot(n, axis))
        .fold(1.0, f32::min);
    GpuBounds {
        center,
        radius: radius * (1.0 + 1e-5),
        axis,
        cone_cos: if length > 1e-20 { cosine - 1e-4 } else { -1.0 },
        quad_edge: edge * (1.0 + 1e-5),
        reserved: [0; 3],
    }
}

/// `cull.wgsl`'s `in_frustum`.
pub(crate) fn in_frustum(view: &GpuView, center: [f32; 3], radius: f32) -> bool {
    let q = relative(view, center);
    view.planes
        .iter()
        .all(|&[a, b, c, d]| dot([a, b, c], q) + d >= -radius)
}

/// `center - eye`, as the GPU forms it.
pub(crate) fn relative(view: &GpuView, center: [f32; 3]) -> [f32; 3] {
    sub(center, [view.eye[0], view.eye[1], view.eye[2]])
}

/// `cull.wgsl`'s `backfacing`.
pub(crate) fn backfacing(bounds: &GpuBounds, eye: [f32; 3], margin: f32) -> bool {
    if bounds.cone_cos <= 0.0 || bounds.radius < 0.0 {
        return false;
    }
    let v = sub(bounds.center, eye);
    let d = dot(v, v).sqrt();
    let r = bounds.radius + margin;
    if d <= r {
        return false;
    }
    let cos_t = dot(v, bounds.axis) / d;
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let sin_a = (1.0 - bounds.cone_cos * bounds.cone_cos).max(0.0).sqrt();
    cos_t * bounds.cone_cos - sin_t * sin_a > r / d
}

/// `lod.wgsl`'s `cone_cos_sin`: the cosine and sine of `x` in `[0, π/2]` by their Taylor
/// series, in `f32` products and sums.
pub(crate) fn cone_cos_sin(x: f32) -> (f32, f32) {
    let t = x * x;
    let c = 1.0
        + t * (-1.0 / 2.0
            + t * (1.0 / 24.0
                + t * (-1.0 / 720.0
                    + t * (1.0 / 40_320.0
                        + t * (-1.0 / 3_628_800.0 + t * (1.0 / 479_001_600.0))))));
    let s = x
        * (1.0
            + t * (-1.0 / 6.0
                + t * (1.0 / 120.0
                    + t * (-1.0 / 5040.0
                        + t * (1.0 / 362_880.0
                            + t * (-1.0 / 39_916_800.0 + t * (1.0 / (479_001_600.0 * 13.0))))))));
    (c, s)
}

/// `lod.wgsl`'s `finest_facing_away`: the same formula in `f32` (the shader may contract it into
/// fused multiply-adds, and takes its lengths through `inverseSqrt`).
///
/// Whether every normal in `cone`, widened by one step, faces away from `eye` at every point of
/// `bounds`' sphere grown by `margin`, with the shader's slack on its rounding. The axis is
/// [`FinestNormalCone::axis`], which the shader's decode mirrors.
pub(crate) fn finest_facing_away(
    bounds: &GpuBounds,
    cone: FinestNormalCone,
    eye: [f32; 3],
    margin: f32,
) -> bool {
    let decoded = cone.axis();
    let errors = ShaderErrors {
        axis: [decoded.x, decoded.y, decoded.z],
        overestimate: 0.0,
    };
    finest_facing_away_with(bounds, cone, eye, margin, errors)
}

/// What [`finest_facing_away_with`] stands in for the shader's own rounding, beyond this
/// arithmetic's: the axis it decodes, and how far its `q` may come out over the true value.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShaderErrors {
    /// The decoded axis the shader uses (within its decode's error of
    /// [`FinestNormalCone::axis`]).
    pub(crate) axis: [f32; 3],
    /// How far over `|a|·d·cos(θ + α')` the shader's `q` comes out, as a fraction of `d`.
    pub(crate) overestimate: f32,
}

/// [`finest_facing_away`] with the shader's errors made explicit (`errors`), so a test can push
/// them to the bounds `lod.wgsl` states and check the test stays conservative.
///
/// # Panics
///
/// Never: a step count is 12 bits, so one more fits a `u16`.
pub(crate) fn finest_facing_away_with(
    bounds: &GpuBounds,
    cone: FinestNormalCone,
    eye: [f32; 3],
    margin: f32,
    errors: ShaderErrors,
) -> bool {
    /// `lod.wgsl`'s `CONE_STEP`.
    const STEP: f32 = core::f32::consts::PI / 4095.0;
    /// `lod.wgsl`'s `CONE_SLACK`.
    const SLACK: f32 = 1e-5;
    if cone.covers_everything() || bounds.radius < 0.0 {
        return false;
    }
    let steps = u16::try_from((cone.to_bits() >> 20) + 1).expect("a 12-bit count");
    let half = f32::from(steps) * STEP;
    if half >= core::f32::consts::FRAC_PI_2 {
        return false;
    }
    let offset = sub(bounds.center, eye);
    let d = dot(offset, offset).sqrt();
    let radius = bounds.radius + margin;
    if d <= radius {
        return false;
    }
    let axis = errors.axis;
    let across = cross(offset, axis);
    let (cos_a, sin_a) = cone_cos_sin(half);
    let q = dot(offset, axis) * cos_a - dot(across, across).sqrt() * sin_a;
    q + errors.overestimate * d > radius + SLACK * d
}

/// `cull.wgsl`'s `projected`.
pub(crate) fn projected(view: &GpuView, bounds: &GpuBounds, margin: f32, extent: f32) -> f32 {
    let q = relative(view, bounds.center);
    let r = bounds.radius + margin;
    let near = view.planes[4];
    let z = (dot([near[0], near[1], near[2]], q) - r).max(view.viewport[3]);
    let d = dot(q, q).sqrt() + r;
    extent * view.viewport[2] * d / (z * z)
}

/// A depth pyramid on the CPU: per mip, its width, height and texels, row by row.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CpuPyramid {
    /// The mips, finest first.
    pub(crate) mips: Vec<(u32, u32, Vec<f32>)>,
}

impl CpuPyramid {
    /// The pyramid of a `width` × `height` depth image, row by row, as `pyramid.wgsl` builds it.
    pub(crate) fn new(width: u32, height: u32, depth: &[f32]) -> Self {
        let mut mips = Vec::new();
        let (mut w, mut h, mut below) = (width, height, depth.to_vec());
        let mut size = super::mip_zero(width, height);
        loop {
            let (nw, nh) = size;
            size = ((nw / 2).max(1), (nh / 2).max(1));
            let at = |x: u32, y: u32| below[(y.min(h - 1) * w + x.min(w - 1)) as usize];
            let mut texels = Vec::with_capacity((nw * nh) as usize);
            for y in 0..nh {
                for x in 0..nw {
                    let (px, py) = (2 * x, 2 * y);
                    texels.push(
                        at(px, py)
                            .min(at(px + 1, py))
                            .min(at(px, py + 1).min(at(px + 1, py + 1))),
                    );
                }
            }
            mips.push((nw, nh, texels.clone()));
            if nw == 1 && nh == 1 {
                break;
            }
            (w, h, below) = (nw, nh, texels);
        }
        Self { mips }
    }
}

/// The level choice's `occluded`, against a CPU pyramid.
///
/// # Panics
///
/// If the pyramid has no mips.
pub(crate) fn occluded(
    view: &GpuView,
    pyramid: &CpuPyramid,
    center: [f32; 3],
    radius: f32,
) -> bool {
    let mut lo = [f32::MAX; 2];
    let mut hi = [f32::MIN; 2];
    let mut nearest = 0.0_f32;
    let columns = view.view_proj;
    let q = relative(view, center);
    for corner in 0..8 {
        let offset = |bit: u32| if corner & bit != 0 { radius } else { -radius };
        let point = [q[0] + offset(1), q[1] + offset(2), q[2] + offset(4)];
        let clip: [f32; 4] = [0, 1, 2, 3].map(|row| {
            columns[0][row] * point[0]
                + columns[1][row] * point[1]
                + columns[2][row] * point[2]
                + columns[3][row]
        });
        if clip[3] <= 0.0 {
            return false;
        }
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        lo = [lo[0].min(ndc[0]), lo[1].min(ndc[1])];
        hi = [hi[0].max(ndc[0]), hi[1].max(ndc[1])];
        nearest = nearest.max(ndc[2]);
    }
    if nearest >= 1.0 {
        return false;
    }
    let (sx, sy) = (view.viewport[0], view.viewport[1]);
    let x0 = (lo[0] * 0.5 + 0.5) * sx - 1.0;
    let x1 = (hi[0] * 0.5 + 0.5) * sx + 1.0;
    let y0 = (0.5 - hi[1] * 0.5) * sy - 1.0;
    let y1 = (0.5 - lo[1] * 0.5) * sy + 1.0;
    if x1 < 0.0 || y1 < 0.0 || x0 >= sx || y0 >= sy {
        return false;
    }
    let p0 = [pixel(x0.max(0.0)), pixel(y0.max(0.0))];
    let p1 = [pixel(x1.min(sx - 1.0)), pixel(y1.min(sy - 1.0))];
    let levels = u32::try_from(pyramid.mips.len()).expect("a few mips");
    let mut level = 0;
    while level + 1 < levels {
        let shift = level + 1;
        if (p1[0] >> shift) - (p0[0] >> shift) <= 1 && (p1[1] >> shift) - (p0[1] >> shift) <= 1 {
            break;
        }
        level += 1;
    }
    let shift = level + 1;
    let (width, height, texels) = &pyramid.mips[usize::try_from(level).expect("small")];
    let low = [
        (p0[0] >> shift).min(width - 1),
        (p0[1] >> shift).min(height - 1),
    ];
    let high = [
        (p1[0] >> shift).min(width - 1),
        (p1[1] >> shift).min(height - 1),
    ];
    let at = |x: u32, y: u32| texels[usize::try_from(y * width + x).expect("small")];
    let far = at(low[0], low[1])
        .min(at(high[0], low[1]))
        .min(at(low[0], high[1]).min(at(high[0], high[1])));
    nearest < far
}

/// A pixel coordinate as WGSL's `u32()` converts it: towards zero.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a non-negative coordinate within the viewport, truncated as the shader truncates it"
)]
const fn pixel(coordinate: f32) -> u32 {
    coordinate as u32
}

/// Writes `depth` (row by row, `width` × `height`) into a new `Depth32Float` texture, by a draw
/// that sets each fragment's depth: depth textures can't be copied into.
pub(crate) fn depth_texture(gpu: &Gpu, width: u32, height: u32, depth: &[f32]) -> Texture {
    use wgpu::util::DeviceExt;
    let device = gpu.device();
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("synthetic depth"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("synthetic depth values"),
        contents: bytemuck::cast_slice(depth),
        usage: BufferUsages::STORAGE,
    });
    let size = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("synthetic depth width"),
        contents: bytemuck::cast_slice(&[width, height, 0, 0]),
        usage: BufferUsages::UNIFORM,
    });
    let source = r"
        @group(0) @binding(0) var<storage, read> values: array<f32>;
        @group(0) @binding(1) var<uniform> size: vec4<u32>;
        @vertex fn full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
            let x = f32(i & 1u) * 4.0 - 1.0;
            let y = f32(i >> 1u) * 4.0 - 1.0;
            return vec4<f32>(x, y, 0.5, 1.0);
        }
        @fragment fn depth(@builtin(position) at: vec4<f32>) -> @builtin(frag_depth) f32 {
            let p = vec2<u32>(at.xy);
            return values[p.y * size.x + p.x];
        }
    ";
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("synthetic depth"),
        source: ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("synthetic depth"),
        layout: None,
        vertex: VertexState {
            module: &module,
            entry_point: Some("full"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("depth"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("synthetic depth"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[super::entry(0, &values), super::entry(1, &size)],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("synthetic depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..RenderPassDescriptor::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    gpu.queue().submit([encoder.finish()]);
    texture
}

/// Mip `level` of `texture` (`R32Float`, `width` × `height` at that mip), row by row.
///
/// # Panics
///
/// If the readback fails.
pub(crate) fn read_mip(
    gpu: &Gpu,
    texture: &Texture,
    level: u32,
    width: u32,
    height: u32,
) -> Vec<f32> {
    let device = gpu.device();
    let row = (width * 4).next_multiple_of(256);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("mip readback"),
        size: u64::from(row * height),
        usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let padded: Vec<f32> = gpu.read_buffer(&buffer).expect("the mip reads back");
    let stride = (row / 4) as usize;
    padded
        .chunks(stride)
        .flat_map(|r| r[..width as usize].to_vec())
        .collect()
}
