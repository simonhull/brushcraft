//! CPU twins of the culling passes, for the tests.
//!
//! The same arithmetic as `cull.wgsl`, `bounds.wgsl`, `pyramid.wgsl` and the level choice's
//! occlusion test, in `f32`, plus helpers that put a synthetic depth image on the GPU and read
//! a pyramid back.

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

/// The bounds of a tile of `side` × `side` vertices, as the bounds pass computes them.
pub(crate) fn bounds_of(vertices: &[GpuVertex], side: usize) -> GpuBounds {
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
    let normals: Vec<[f32; 3]> = (0..quads * quads)
        .flat_map(|q| quad_triangles(&points, side, q % quads, q / quads))
        .map(|[a, b, c]| face_normal(a, b, c))
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
