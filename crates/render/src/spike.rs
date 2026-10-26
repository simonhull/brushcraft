//! M1a's spikes: experiments that answer a design question with measurements, built behind the
//! `spike` feature so the gate builds and tests them while the shipped renderer leaves them out.
//!
//! Each spike ends in a report in the docs repository (`research/platform/gpu-spikes.md`) and a
//! decision for the user; its code is then either promoted out of this module or deleted.
//!
//! - [`brush`]: can one brush's Rust source run on the GPU? The draw brush as a compute shader,
//!   compiled from `brush`'s own source by rust-gpu and written by hand in WGSL, each checked
//!   against the CPU and timed.
//! - [`eval`]: what does moving tile evaluation to the GPU buy? One level step of every tile of a
//!   set (prediction, frames, corrections, normals) as compute shaders compiled by rust-gpu,
//!   checked against `geom` and for seam self-consistency, and timed.

pub mod brush;
pub mod eval;

use std::time::Duration;

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassTimestampWrites, QUERY_SIZE,
    QuerySet, QuerySetDescriptor, QueryType,
};

use crate::gpu::{Gpu, GpuError};

/// Two timestamp queries around a compute pass, and the buffers their values are read through.
#[derive(Debug)]
pub(crate) struct Timer {
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
    /// A timer on `gpu`, whose timestamps tick every `period` nanoseconds. Created inside the
    /// caller's error scope.
    pub(crate) fn new(gpu: &Gpu, period: f32) -> Self {
        let device = gpu.device();
        let size = u64::from(2 * QUERY_SIZE);
        let buffer = |label, usage| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            queries: device.create_query_set(&QuerySetDescriptor {
                label: Some("spike timestamps"),
                ty: QueryType::Timestamp,
                count: 2,
            }),
            resolved: buffer(
                "spike timestamps resolved",
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            ),
            staging: buffer(
                "spike timestamps readback",
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
            period,
        }
    }

    /// The pass's timestamp writes: the first query at its beginning, the second at its end.
    pub(crate) const fn writes(&self) -> ComputePassTimestampWrites<'_> {
        ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Records resolving both queries, after the pass.
    pub(crate) fn resolve(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolved, 0);
    }

    /// How long the timed pass took on the GPU, waiting for it.
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer`].
    pub(crate) fn elapsed(&self, gpu: &Gpu) -> Result<Duration, GpuError> {
        let ticks: Vec<u64> = gpu.read_buffer_with(&self.resolved, &self.staging)?;
        let elapsed = match ticks.as_slice() {
            [begin, end] => end.saturating_sub(*begin),
            _ => 0,
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a pass's ticks are far below 2^52, and the product is nanoseconds"
        )]
        let nanos = (elapsed as f64 * f64::from(self.period)) as u64;
        Ok(Duration::from_nanos(nanos))
    }
}
