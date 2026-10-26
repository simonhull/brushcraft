//! Spike 1: the draw brush on the GPU, one invocation per vertex.
//!
//! The question (architecture spec, generational bets 1 and 2): can a brush be written once, in
//! `brush`, and run both on the CPU and as a compute shader? Two answers are built side by side
//! and checked against the CPU (`brush::draw`, the reference):
//!
//! - [`Shader::Wgsl`]: `draw.wgsl`, a hand-written twin of `brush::draw_vertex` with the
//!   [`Smoothstep`] falloff, operation for operation;
//! - [`Shader::RustGpu`]: `brush::draw_vertex` itself, compiled to SPIR-V by rust-gpu from the
//!   unchanged `brush` source (see `shaders/brush-gpu`), to be loaded through `naga`'s SPIR-V front
//!   end, so the same module would run on Vulkan and on Metal.
//!
//! The finding so far: rust-gpu compiles `brush` unchanged (its build has to skip cargo's
//! rust-version check, as its nightly is older than the workspace's), and the SPIR-V validates,
//! but `naga` rejects it: `brush`'s square root tests `f32::is_finite`, which `core` writes as a
//! comparison with an infinite constant, and `naga` refuses infinite literals. A test pins that
//! rejection, so it is noticed when it changes.
//!
//! A [`Footprint`] holds the buffers for one set of positions, so a benchmark can time the
//! upload, the dispatch and the readback separately, as a brush step would pay them.

use std::borrow::Cow;
use std::fmt;
use std::time::Duration;

use brush::{Smoothstep, draw};
use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites, ComputePipeline,
    QUERY_SIZE, QuerySet, QuerySetDescriptor, QueryType, ShaderSource,
};

use crate::gpu::{Gpu, GpuError};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The WGSL twin of `brush::draw_vertex`.
const WGSL: &str = include_str!("draw.wgsl");

/// The SPIR-V rust-gpu compiled from `shaders/brush-gpu`, which calls `brush::draw_vertex`
/// unchanged. Rebuilt by `shaders/build.sh`; checked in so the gate needs no rust-gpu toolchain.
const SPIRV: &[u8] = include_bytes!("brush_gpu.spv");

// SPIR-V is a stream of 32-bit words; `make_spirv_raw` would panic on a partial one.
const _: () = assert!(
    SPIRV.len().is_multiple_of(4),
    "brush_gpu.spv is whole words"
);

/// Invocations per workgroup in both shaders.
const WORKGROUP: u32 = 64;

/// Which source the draw brush's compute shader is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shader {
    /// `draw.wgsl`, the hand-written twin of `brush::draw_vertex`.
    Wgsl,
    /// `brush::draw_vertex` itself, compiled to SPIR-V by rust-gpu. `naga` rejects it for now
    /// (see the module docs), so [`GpuDraw::new`] fails with it.
    RustGpu,
}

impl Shader {
    /// Both shaders, in the order the spike reports them.
    pub const ALL: [Self; 2] = [Self::Wgsl, Self::RustGpu];

    /// The shader's entry point.
    const fn entry_point(self) -> &'static str {
        match self {
            Self::Wgsl => "draw",
            Self::RustGpu => "main",
        }
    }

    /// The shader's source, as `wgpu` takes it.
    fn source(self) -> ShaderSource<'static> {
        match self {
            Self::Wgsl => ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
            Self::RustGpu => ShaderSource::SpirV(wgpu::util::make_spirv_raw(SPIRV)),
        }
    }
}

/// Why a GPU draw couldn't run.
#[derive(Debug, Error)]
pub enum DrawError {
    /// The brush radius isn't positive and finite; `brush::draw` treats such a brush as
    /// covering nothing, so it never reaches the GPU.
    #[error("the brush radius {0} isn't positive and finite")]
    Radius(f32),
    /// More vertices than one dispatch along one dimension covers on this device.
    #[error("{count} vertices is more than one dispatch covers here ({max})")]
    TooManyVertices {
        /// The vertices asked for.
        count: usize,
        /// The most one dispatch covers on this device.
        max: u64,
    },
    /// The GPU failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// How far a GPU's deltas are from the CPU's, over every vertex compared.
///
/// Three measures, because one misleads:
///
/// - units in the last place of the delta itself, over vertices both sides put inside the brush.
///   Near the rim the smoothstep `1 − t²(3 − 2t)` cancels, so a delta there is tiny and its last
///   place is far smaller than the rounding of `t` it inherits: millions of ulps there are a few
///   ulps of the strength;
/// - the absolute difference, over every vertex, and the same in units in the last place of the
///   brush strength (`|Δ| / (|strength| · ε)`): what the surface feels;
/// - vertices inside the brush on one side only, right at the rim, where the squared distance
///   rounded to the other side of `radius²`. Their deltas are next to nothing, but zero against
///   anything is billions of ulps, so they are counted apart.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Difference {
    /// The largest distance in units in the last place, over vertices inside on both sides.
    max_ulps: u64,
    /// The largest absolute difference, over every vertex.
    max_abs: f32,
    /// The largest absolute difference in units in the last place of the brush strength.
    max_strength_ulps: f32,
    /// Vertices whose values differ at all.
    differing: usize,
    /// Vertices inside the brush on one side only: a zero against a non-zero.
    inside_on_one_side: usize,
    /// Vertices compared.
    compared: usize,
}

impl Difference {
    /// Adds the comparison of `gpu`'s deltas against `cpu`'s for `brush`, vertex by vertex; only
    /// the shorter prefix if their lengths differ.
    pub fn add(&mut self, gpu: &[f32], cpu: &[f32], brush: &DrawBrush) {
        let strength_ulp = brush.strength.abs() * f32::EPSILON;
        for (&g, &c) in gpu.iter().zip(cpu) {
            let abs = (g - c).abs();
            self.compared += 1;
            self.max_abs = self.max_abs.max(abs);
            if strength_ulp > 0.0 {
                self.max_strength_ulps = self.max_strength_ulps.max(abs / strength_ulp);
            }
            self.differing += usize::from(ordered(g) != ordered(c));
            if (g == 0.0) == (c == 0.0) {
                self.max_ulps = self.max_ulps.max(ordered(g).abs_diff(ordered(c)));
            } else {
                self.inside_on_one_side += 1;
            }
        }
    }

    /// The largest distance in units in the last place, over vertices inside on both sides.
    #[must_use]
    pub const fn max_ulps(&self) -> u64 {
        self.max_ulps
    }

    /// The largest absolute difference.
    #[must_use]
    pub const fn max_abs(&self) -> f32 {
        self.max_abs
    }

    /// The largest absolute difference in units in the last place of the brush strength.
    #[must_use]
    pub const fn max_strength_ulps(&self) -> f32 {
        self.max_strength_ulps
    }

    /// Vertices compared.
    #[must_use]
    pub const fn compared(&self) -> usize {
        self.compared
    }
}

impl fmt::Display for Difference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} vertices: max {} ulps where both are inside; max |Δ| {:e}, {:.2} ulps of the \
             strength; {} differ, {} inside on one side only",
            self.compared,
            self.max_ulps,
            self.max_abs,
            self.max_strength_ulps,
            self.differing,
            self.inside_on_one_side
        )
    }
}

/// `x`'s bits as an integer that orders like the float, so the difference of two is their
/// distance in units in the last place, across zero too.
fn ordered(x: f32) -> i64 {
    let bits = i64::from(x.to_bits());
    if bits & 0x8000_0000 == 0 {
        bits
    } else {
        0x8000_0000 - bits
    }
}

/// Runs each of `brushes` over `positions` through `draw` on `gpu`, and measures the deltas
/// against the CPU's (`brush::draw`).
///
/// # Errors
///
/// [`DrawError::TooManyVertices`] if one dispatch can't cover `positions`; [`DrawError::Gpu`]
/// if the GPU fails.
pub fn against_cpu(
    gpu: &Gpu,
    draw: &GpuDraw,
    positions: &[[f32; 3]],
    brushes: &[DrawBrush],
) -> Result<Difference, DrawError> {
    let footprint = draw.footprint(gpu, positions)?;
    let mut difference = Difference::default();
    for brush in brushes {
        footprint.dispatch(gpu, draw, brush);
        difference.add(&footprint.read_deltas(gpu)?, &brush.cpu(positions), brush);
    }
    Ok(difference)
}

/// One dab of the draw brush: where it is, how far it reaches, how strongly it pushes.
///
/// Built by [`DrawBrush::new`], which rejects a radius that isn't positive and finite: the GPU
/// path never sees one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawBrush {
    /// The brush centre.
    center: [f32; 3],
    /// The brush radius: positive and finite.
    radius: f32,
    /// The displacement at the centre.
    strength: f32,
}

impl DrawBrush {
    /// A brush at `center` reaching `radius`, pushing `strength` at its centre.
    ///
    /// # Errors
    ///
    /// [`DrawError::Radius`] if `radius` isn't positive and finite.
    pub fn new(center: [f32; 3], radius: f32, strength: f32) -> Result<Self, DrawError> {
        if radius > 0.0 && radius.is_finite() {
            Ok(Self {
                center,
                radius,
                strength,
            })
        } else {
            Err(DrawError::Radius(radius))
        }
    }

    /// The CPU reference: `brush::draw` over `positions`, one delta each.
    #[must_use]
    pub fn cpu(&self, positions: &[[f32; 3]]) -> Vec<f32> {
        let mut out = vec![0.0; positions.len()];
        draw(
            &Smoothstep,
            self.center,
            self.radius,
            self.strength,
            positions,
            &mut out,
        );
        out
    }

    /// The shader's uniform block for this brush over `count` vertices.
    fn params(&self, count: u32) -> Params {
        Params {
            center: self.center,
            radius: self.radius,
            radius_squared: self.radius * self.radius,
            strength: self.strength,
            count,
            pad: 0,
        }
    }
}

/// The shaders' uniform block, laid out as WGSL lays out `Params` in `draw.wgsl` (and as the
/// rust-gpu entry point declares it): 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    /// The brush centre.
    center: [f32; 3],
    /// The brush radius.
    radius: f32,
    /// `radius * radius` in `f32`, as `brush::draw` computes it once for all vertices.
    radius_squared: f32,
    /// The displacement at the centre.
    strength: f32,
    /// The vertices the dispatch covers.
    count: u32,
    /// Pads the block to a multiple of 16 bytes.
    pad: u32,
}

/// The draw brush's compute pipeline, built from one [`Shader`].
#[derive(Debug)]
pub struct GpuDraw {
    /// Which source the pipeline was built from.
    shader: Shader,
    /// The pipeline, with the layout its shader implies (uniform, positions, deltas).
    pipeline: ComputePipeline,
}

impl GpuDraw {
    /// Builds the pipeline from `shader`.
    ///
    /// # Errors
    ///
    /// [`GpuError::Validation`] if `wgpu` rejects the shader.
    pub fn new(gpu: &Gpu, shader: Shader) -> Result<Self, GpuError> {
        let label = match shader {
            Shader::Wgsl => "draw brush (WGSL)",
            Shader::RustGpu => "draw brush (rust-gpu)",
        };
        let pipeline = gpu.compute_pipeline(label, shader.source(), shader.entry_point())?;
        Ok(Self { shader, pipeline })
    }

    /// The shader the pipeline was built from.
    #[must_use]
    pub const fn shader(&self) -> Shader {
        self.shader
    }

    /// The whole step, as a brush would run it once: uploads `positions`, dispatches `brush`
    /// over them, waits, and reads the deltas back.
    ///
    /// # Errors
    ///
    /// [`DrawError::TooManyVertices`] if one dispatch can't cover `positions`; [`DrawError::Gpu`]
    /// if the GPU fails.
    pub fn draw(
        &self,
        gpu: &Gpu,
        brush: &DrawBrush,
        positions: &[[f32; 3]],
    ) -> Result<Vec<f32>, DrawError> {
        let footprint = self.footprint(gpu, positions)?;
        footprint.dispatch(gpu, self, brush);
        Ok(footprint.read_deltas(gpu)?)
    }

    /// Buffers for `positions` on the GPU, with the pipeline bound to them.
    ///
    /// # Errors
    ///
    /// [`DrawError::TooManyVertices`] if one dispatch can't cover `positions`.
    pub fn footprint(&self, gpu: &Gpu, positions: &[[f32; 3]]) -> Result<Footprint, DrawError> {
        let too_many = || DrawError::TooManyVertices {
            count: positions.len(),
            max: u64::from(gpu.device().limits().max_compute_workgroups_per_dimension)
                * u64::from(WORKGROUP),
        };
        let max_workgroups = gpu.device().limits().max_compute_workgroups_per_dimension;
        let count = u32::try_from(positions.len())
            .ok()
            .filter(|count| count.div_ceil(WORKGROUP) <= max_workgroups)
            .ok_or_else(too_many)?;
        let device = gpu.device();
        let params = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("draw params"),
            contents: bytemuck::bytes_of(&Params::zeroed()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        // A zero-sized binding is invalid, so an empty footprint still gets one element.
        let floats = positions.len().max(1);
        let positions_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("draw positions"),
            size: (floats * size_of::<[f32; 3]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue()
            .write_buffer(&positions_buffer, 0, bytemuck::cast_slice(positions));
        let deltas_size = (floats * size_of::<f32>()) as u64;
        let deltas = device.create_buffer(&BufferDescriptor {
            label: Some("draw deltas"),
            size: deltas_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("draw deltas readback"),
            size: deltas_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("draw"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
            ],
        });
        Ok(Footprint {
            count,
            params,
            positions: positions_buffer,
            deltas,
            staging,
            bind_group,
            timer: gpu.timestamp_period().map(|period| Timer::new(gpu, period)),
        })
    }
}

/// One set of positions on the GPU, with the buffers a draw dispatch reads and writes.
///
/// Made by [`GpuDraw::footprint`]; it stays bound to that pipeline's layout.
#[derive(Debug)]
pub struct Footprint {
    /// The vertices it holds.
    count: u32,
    /// The uniform block: the brush.
    params: Buffer,
    /// The positions, three floats each, tightly packed.
    positions: Buffer,
    /// One delta per vertex, written by the shader.
    deltas: Buffer,
    /// Where the deltas are copied to be mapped and read.
    staging: Buffer,
    /// The pipeline's bindings to the three buffers above.
    bind_group: BindGroup,
    /// Times a dispatch on the GPU, if the device has timestamp queries.
    timer: Option<Timer>,
}

impl Footprint {
    /// Writes new `positions` into the footprint's buffer: the per-step upload. Only as many as
    /// the footprint holds are written.
    pub fn upload(&self, gpu: &Gpu, positions: &[[f32; 3]]) {
        let count = positions.len().min(self.count as usize);
        let positions = positions.get(..count).unwrap_or_default();
        gpu.queue()
            .write_buffer(&self.positions, 0, bytemuck::cast_slice(positions));
    }

    /// Dispatches `draw` with `brush` over the footprint and submits it, without waiting.
    pub fn dispatch(&self, gpu: &Gpu, draw: &GpuDraw, brush: &DrawBrush) {
        self.submit(gpu, draw, brush, None);
    }

    /// Dispatches `draw` with `brush`, waits for it, and returns how long the compute pass took
    /// on the GPU, from timestamp queries; `None` if the device has none.
    ///
    /// # Errors
    ///
    /// [`GpuError`] if waiting or reading the timestamps fails.
    pub fn time_dispatch(
        &self,
        gpu: &Gpu,
        draw: &GpuDraw,
        brush: &DrawBrush,
    ) -> Result<Option<Duration>, GpuError> {
        let Some(timer) = &self.timer else {
            self.dispatch(gpu, draw, brush);
            return Ok(None);
        };
        self.submit(gpu, draw, brush, Some(timer));
        let ticks: Vec<u64> = gpu.read_buffer_with(&timer.resolved, &timer.staging)?;
        let elapsed = match ticks.as_slice() {
            [begin, end] => end.saturating_sub(*begin),
            _ => 0,
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a dispatch's ticks are far below 2^52, and the product is nanoseconds"
        )]
        let nanos = (elapsed as f64 * f64::from(timer.period)) as u64;
        Ok(Some(Duration::from_nanos(nanos)))
    }

    /// Reads the deltas back, waiting for every dispatch submitted before.
    ///
    /// # Errors
    ///
    /// [`GpuError`] if waiting or mapping fails.
    pub fn read_deltas(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let mut deltas: Vec<f32> = gpu.read_buffer_with(&self.deltas, &self.staging)?;
        deltas.truncate(self.count as usize);
        Ok(deltas)
    }

    /// Records `brush` into the uniform block and one compute pass of `draw` over the
    /// footprint, timed by `timer` if given, and submits it.
    fn submit(&self, gpu: &Gpu, draw: &GpuDraw, brush: &DrawBrush, timer: Option<&Timer>) {
        gpu.queue().write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&brush.params(self.count)),
        );
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("draw"),
                timestamp_writes: timer.map(|timer| ComputePassTimestampWrites {
                    query_set: &timer.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&draw.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
        }
        if let Some(timer) = timer {
            encoder.resolve_query_set(&timer.queries, 0..2, &timer.resolved, 0);
        }
        gpu.queue().submit([encoder.finish()]);
    }
}

/// Two timestamp queries around a compute pass, and the buffers their values are read through.
#[derive(Debug)]
struct Timer {
    /// The begin and end timestamps of the pass.
    queries: QuerySet,
    /// Where the queries are resolved to, as two `u64` ticks.
    resolved: Buffer,
    /// Where the resolved ticks are copied to be mapped.
    staging: Buffer,
    /// Nanoseconds per tick.
    period: f32,
}

impl Timer {
    /// A timer on `gpu`, whose timestamps tick every `period` nanoseconds.
    fn new(gpu: &Gpu, period: f32) -> Self {
        let device = gpu.device();
        let size = 2 * QUERY_SIZE;
        Self {
            queries: device.create_query_set(&QuerySetDescriptor {
                label: Some("draw timestamps"),
                ty: QueryType::Timestamp,
                count: 2,
            }),
            resolved: device.create_buffer(&BufferDescriptor {
                label: Some("draw timestamps resolved"),
                size: u64::from(size),
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&BufferDescriptor {
                label: Some("draw timestamps readback"),
                size: u64::from(size),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period,
        }
    }
}
