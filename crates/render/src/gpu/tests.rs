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

/// A buffer of `size` bytes with `usage`, created inside an error scope.
fn buffer(gpu: &Gpu, size: u64, usage: BufferUsages) -> Buffer {
    gpu.scoped("test buffer", || {
        gpu.device().create_buffer(&BufferDescriptor {
            label: Some("test"),
            size,
            usage,
            mapped_at_creation: false,
        })
    })
    .expect("a valid buffer")
}

#[test]
fn a_readback_that_is_not_whole_words_is_an_error() {
    GpuTest::run(|gpu| {
        let six = buffer(gpu, 6, BufferUsages::COPY_SRC);
        let error = gpu
            .read_buffer::<u8>(&six)
            .expect_err("6 bytes aren't whole words");
        assert!(
            matches!(error, GpuError::UnalignedCopy { size: 6 }),
            "{error}"
        );
    });
}

#[test]
fn a_readback_of_a_buffer_without_copy_src_is_an_error() {
    GpuTest::run(|gpu| {
        let storage = buffer(gpu, 16, BufferUsages::STORAGE);
        let error = gpu
            .read_buffer::<u32>(&storage)
            .expect_err("not a copy source");
        assert!(matches!(error, GpuError::NotCopySource), "{error}");
    });
}

#[test]
fn a_scoped_validation_error_is_returned_not_raised() {
    GpuTest::run(|gpu| {
        let small = buffer(gpu, 16, BufferUsages::COPY_SRC);
        let other = buffer(gpu, 16, BufferUsages::COPY_DST);
        let error = gpu
            .scoped("an out-of-bounds copy", || {
                let mut encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_buffer(&small, 0, &other, 0, 64);
                gpu.queue().submit([encoder.finish()]);
            })
            .expect_err("64 bytes from a 16-byte buffer");
        assert!(
            matches!(&error, GpuError::Validation { what, .. } if what == "an out-of-bounds copy"),
            "{error}"
        );
        gpu.check()
            .expect("and nothing is left over for the next call");
    });
}

#[test]
fn an_error_outside_any_scope_is_reported_once_by_the_next_call() {
    GpuTest::run(|gpu| {
        // MAP_READ may only be combined with COPY_DST: invalid, and raised outside any scope.
        drop(gpu.device().create_buffer(&BufferDescriptor {
            label: Some("invalid"),
            size: 16,
            usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let error = gpu
            .compute_pipeline("double", ShaderSource::Wgsl(DOUBLE.into()), "double")
            .expect_err("the earlier error");
        assert!(matches!(error, GpuError::Uncaptured { .. }), "{error}");
        gpu.compute_pipeline("double", ShaderSource::Wgsl(DOUBLE.into()), "double")
            .expect("reported once, then the device works");
    });
}

#[test]
fn a_lost_device_fails_every_later_call() {
    GpuTest::run(|gpu| {
        gpu.device().destroy();
        // The loss is reported through the device's callback, which runs as the device is
        // polled.
        drop(gpu.device().poll(PollType::Poll));
        for _ in 0..2 {
            let error = gpu
                .compute_pipeline("double", ShaderSource::Wgsl(DOUBLE.into()), "double")
                .expect_err("the device is gone");
            assert!(matches!(error, GpuError::DeviceLost { .. }), "{error}");
        }
    });
}

#[test]
fn a_staging_buffer_that_cannot_be_mapped_is_an_error() {
    GpuTest::run(|gpu| {
        let source = buffer(gpu, 16, BufferUsages::COPY_SRC);
        let staging = buffer(gpu, 16, BufferUsages::COPY_DST);
        let error = gpu
            .read_buffer_with::<u32>(&source, &staging)
            .expect_err("no MAP_READ");
        assert!(matches!(error, GpuError::StagingUsage), "{error}");
    });
}

#[test]
fn a_staging_buffer_smaller_than_the_source_is_an_error() {
    GpuTest::run(|gpu| {
        let source = buffer(gpu, 32, BufferUsages::COPY_SRC);
        let staging = buffer(gpu, 16, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        let error = gpu
            .read_buffer_with::<u32>(&source, &staging)
            .expect_err("16 bytes can't hold 32");
        assert!(
            matches!(
                error,
                GpuError::StagingTooSmall {
                    size: 32,
                    staging: 16
                }
            ),
            "{error}"
        );
        // A kept staging buffer serves readback after readback: it is unmapped each time.
        let fits = buffer(gpu, 32, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
        for _ in 0..2 {
            let values: Vec<u32> = gpu.read_buffer_with(&source, &fits).expect("the readback");
            assert_eq!(values, vec![0; 8], "a fresh buffer reads zero");
        }
    });
}

#[test]
fn every_unscoped_error_is_counted() {
    GpuTest::run(|gpu| {
        for _ in 0..3 {
            drop(gpu.device().create_buffer(&BufferDescriptor {
                label: Some("invalid"),
                size: 16,
                usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        let error = gpu.check().expect_err("three earlier errors");
        assert!(
            matches!(&error, GpuError::Uncaptured { message } if message.ends_with("(and 2 more)")),
            "{error}"
        );
        gpu.check().expect("reported once");
    });
}

#[test]
fn an_error_another_thread_raises_during_scoped_work_is_that_calls() {
    GpuTest::run(|gpu| {
        let small = buffer(gpu, 16, BufferUsages::COPY_SRC);
        let other = buffer(gpu, 16, BufferUsages::COPY_DST);
        let invalid = || {
            drop(gpu.device().create_buffer(&BufferDescriptor {
                label: Some("invalid"),
                size: 16,
                usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        };
        // Alone: scopes are thread-local, so the other thread's error is unscoped, and the
        // closing check gives it to this call.
        let error = gpu
            .scoped("work beside another thread", || {
                std::thread::scope(|s| {
                    s.spawn(invalid);
                });
            })
            .expect_err("the other thread's error");
        assert!(matches!(error, GpuError::Uncaptured { .. }), "{error}");
        gpu.check().expect("attributed, so not reported again");
        // With a scoped error of its own: that is the error, naming the unscoped one.
        let both = gpu
            .scoped("a bad copy beside another thread", || {
                let mut encoder = gpu
                    .device()
                    .create_command_encoder(&CommandEncoderDescriptor::default());
                encoder.copy_buffer_to_buffer(&small, 0, &other, 0, 64);
                gpu.queue().submit([encoder.finish()]);
                std::thread::scope(|s| {
                    s.spawn(invalid);
                });
            })
            .expect_err("both errors");
        let message = both.to_string();
        assert!(
            matches!(both, GpuError::Validation { .. }),
            "the scoped error first: {message}"
        );
        assert!(
            message.contains("meanwhile, unscoped"),
            "and the unscoped one named: {message}"
        );
        gpu.check().expect("nothing left over");
    });
}

#[test]
fn an_unscoped_error_names_the_thread_that_raised_it() {
    GpuTest::run(|gpu| {
        let invalid = || {
            drop(gpu.device().create_buffer(&BufferDescriptor {
                label: Some("invalid"),
                size: 16,
                usage: BufferUsages::MAP_READ | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        };
        std::thread::scope(|s| {
            std::thread::Builder::new()
                .name("uploads".into())
                .spawn_scoped(s, invalid)
                .expect("a thread");
        });
        // Reported by a call on this thread, but raised on that one: the message says which.
        let error = gpu.check().expect_err("the other thread's error");
        let named = matches!(
            &error,
            GpuError::Uncaptured { message } if message.starts_with("on the \"uploads\" thread: ")
        );
        assert!(named, "{error}");
    });
}
