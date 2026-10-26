//! Opening the software adapter, and a compute shader round trip through it.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferUsages, ComputePassDescriptor, ShaderSource,
};

use super::*;

/// A compute shader that doubles every element of a `u32` storage buffer in place.
const DOUBLE: &str = r"
// Doubles every element of `values` in place, one invocation per element.

// The buffer to double: read and written by the same invocation, so no barriers are needed.
@group(0) @binding(0) var<storage, read_write> values: array<u32>;

// One invocation per element; invocations past the end of the buffer do nothing.
@compute @workgroup_size(64)
fn double(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&values) {
        values[id.x] = values[id.x] * 2u;
    }
}
";

/// Elements in the doubled buffer: a multiple of the workgroup size, so every workgroup is full.
const LEN: u32 = 1024;

/// Workgroup size of [`DOUBLE`].
const WORKGROUP: u32 = 64;

#[test]
fn the_software_adapter_opens_and_is_a_cpu_device() {
    GpuTest::run(|gpu| {
        let info = gpu.adapter_info();
        assert_eq!(
            info.device_type,
            DeviceType::Cpu,
            "a software adapter: {info:?}"
        );
        assert!(
            info.name.to_lowercase().contains("llvmpipe"),
            "lavapipe names its device after llvmpipe: {info:?}"
        );
    });
}

#[test]
fn a_named_request_finds_the_software_adapter_by_name() {
    let gpu = Gpu::open(&AdapterRequest::Named("LLVMPIPE".to_owned())).expect("lavapipe by name");
    assert_eq!(
        gpu.adapter_info().device_type,
        DeviceType::Cpu,
        "the same adapter"
    );
}

#[test]
fn a_request_nobody_matches_lists_the_adapters_seen() {
    let error = Gpu::open(&AdapterRequest::Named("no such adapter, surely".to_owned()))
        .expect_err("nothing matches");
    let message = error.to_string();
    assert!(
        matches!(error, GpuError::NoAdapter { .. }),
        "no adapter: {message}"
    );
    assert!(
        message.to_lowercase().contains("llvmpipe"),
        "lists what it saw: {message}"
    );
}

#[test]
fn a_shader_that_does_not_compile_is_an_error_not_a_panic() {
    GpuTest::run(|gpu| {
        let error = gpu
            .compute_pipeline("broken", ShaderSource::Wgsl("fn nope( {".into()), "nope")
            .expect_err("a syntax error");
        assert!(
            matches!(&error, GpuError::Validation { what, .. } if what == "broken"),
            "a validation error naming the shader: {error}"
        );
    });
}

#[test]
fn a_compute_shader_doubles_a_buffer_through_a_mapped_readback() {
    GpuTest::run(|gpu| {
        let input: Vec<u32> = (0..LEN).map(|k| k * 3 + 1).collect();
        let values = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("values"),
            contents: bytemuck::cast_slice(&input),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let pipeline = gpu
            .compute_pipeline("double", ShaderSource::Wgsl(DOUBLE.into()), "double")
            .expect("the shader compiles");
        let bind_group = gpu.device().create_bind_group(&BindGroupDescriptor {
            label: Some("double"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: values.as_entire_binding(),
            }],
        });
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(LEN / WORKGROUP, 1, 1);
        }
        gpu.queue().submit([encoder.finish()]);
        let output: Vec<u32> = gpu.read_buffer(&values).expect("the readback");
        let expected: Vec<u32> = input.iter().map(|v| v * 2).collect();
        assert_eq!(output, expected, "every element doubled");
    });
}

#[test]
fn a_readback_as_wider_elements_than_the_buffer_holds_is_misaligned() {
    GpuTest::run(|gpu| {
        let buffer = gpu.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("twelve bytes"),
            contents: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            usage: BufferUsages::COPY_SRC,
        });
        let error = gpu
            .read_buffer::<[u8; 8]>(&buffer)
            .expect_err("12 bytes aren't whole 8-byte elements");
        assert!(
            matches!(
                error,
                GpuError::Misaligned {
                    size: 12,
                    element: 8
                }
            ),
            "{error}"
        );
    });
}
