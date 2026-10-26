//! Opening a GPU: choosing an adapter, creating its device, and the few helpers every piece of
//! GPU work shares (building a compute pipeline with its errors returned, reading a buffer back).
//!
//! Everything here works headless: no window or surface is involved, so tests, benchmarks and
//! tools open a [`Gpu`] the same way the renderer will. Which adapter is used is the caller's
//! choice, as an [`AdapterRequest`]: the system's preferred one, a software rasterizer (the
//! gate's lavapipe), or one picked by name.
//!
//! Only the Vulkan and Metal backends are compiled in: Vulkan on Linux and Android, Metal on
//! macOS and iOS.

use std::fmt;
use std::sync::mpsc;

use bytemuck::Pod;
use thiserror::Error;
use wgpu::{
    Adapter, AdapterInfo, Backends, Buffer, BufferAsyncError, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    DeviceType, ErrorFilter, Features, Instance, InstanceDescriptor, MapMode, MapRangeError,
    PollError, PollType, PowerPreference, Queue, RequestAdapterOptions, RequestDeviceError,
    ShaderModuleDescriptor, ShaderSource,
};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// The optional features GPU work here uses when the adapter offers them.
///
/// Their absence is never fatal: [`Gpu`] reports what it got, and the work that needs one
/// (timing a dispatch) is skipped or reported as unavailable.
const OPTIONAL_FEATURES: Features = Features::TIMESTAMP_QUERY;

/// Which adapter [`Gpu::open`] should use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterRequest {
    /// The system's preferred high-performance adapter: the discrete GPU where there is one.
    Default,
    /// A software rasterizer, an adapter whose device type is [`DeviceType::Cpu`]: lavapipe on
    /// Linux. The gate runs every GPU test on it, so a test never depends on the machine's GPU.
    Software,
    /// The first adapter whose name contains this text, compared case-insensitively: `"4090"`,
    /// `"llvmpipe"`, `"Apple M4"`.
    Named(String),
}

impl fmt::Display for AdapterRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("the default adapter"),
            Self::Software => f.write_str("a software (CPU) adapter"),
            Self::Named(name) => write!(f, "an adapter named like {name:?}"),
        }
    }
}

/// Why opening a GPU, or running work on it, failed.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No adapter on this machine matches the request. The message lists the adapters seen.
    #[error("no adapter matches {request}; adapters seen: {seen}")]
    NoAdapter {
        /// What was asked for.
        request: AdapterRequest,
        /// The adapters the instance enumerated, as `name (backend, device type)`, comma
        /// separated: `none` if there were none.
        seen: String,
    },
    /// The adapter refused to create a device.
    #[error("the adapter refused to create a device")]
    RequestDevice(#[from] RequestDeviceError),
    /// `wgpu` rejected a shader or pipeline: the shader doesn't compile or validate, or doesn't
    /// match the layout it is used with.
    #[error("{what} failed validation: {message}")]
    Validation {
        /// What was being created, from the caller's label.
        what: String,
        /// `wgpu`'s description of the failure, with its sources.
        message: String,
    },
    /// Mapping a buffer for reading failed.
    #[error("mapping a buffer for reading failed")]
    Map(#[from] BufferAsyncError),
    /// A mapped buffer couldn't be viewed.
    #[error("viewing a mapped buffer failed")]
    MapRange(#[from] MapRangeError),
    /// Waiting for the device to finish its work failed (device lost, or a timeout).
    #[error("waiting for the GPU failed")]
    Poll(#[from] PollError),
    /// A buffer's size isn't a whole number of the elements it is read back as.
    #[error("a buffer of {size} bytes isn't a whole number of {element}-byte elements")]
    Misaligned {
        /// The buffer's size in bytes.
        size: u64,
        /// The size of one element in bytes.
        element: usize,
    },
    /// A staging buffer is smaller than the buffer read through it.
    #[error("a {staging}-byte staging buffer can't hold a {size}-byte readback")]
    StagingTooSmall {
        /// The size of the buffer read, in bytes.
        size: u64,
        /// The staging buffer's size, in bytes.
        staging: u64,
    },
}

/// An open GPU: the adapter's description, and the device and queue created on it.
///
/// Opened headless by [`Gpu::open`]. The device has the adapter's full limits (the work here
/// sizes its buffers to what the adapter allows), and timestamp queries when the adapter offers
/// them; [`Gpu::timestamp_period`] says whether dispatches can be timed.
#[derive(Debug)]
pub struct Gpu {
    /// The adapter the device was created on: name, backend, device type, driver.
    adapter_info: AdapterInfo,
    /// The logical device everything is created on.
    device: Device,
    /// The device's queue, where command buffers are submitted and buffers written.
    queue: Queue,
}

impl Gpu {
    /// Opens the adapter `request` names and creates its device and queue, blocking until they
    /// are ready.
    ///
    /// The device takes the adapter's limits, and enables timestamp queries if the adapter offers
    /// them (their absence isn't an error: see [`Gpu::timestamp_period`]).
    ///
    /// # Errors
    ///
    /// [`GpuError::NoAdapter`] if no adapter matches; [`GpuError::RequestDevice`] if the adapter
    /// can't create the device.
    pub fn open(request: &AdapterRequest) -> Result<Self, GpuError> {
        let instance = instance();
        let adapter = pollster::block_on(choose(&instance, request))?;
        let adapter_info = adapter.get_info();
        let descriptor = DeviceDescriptor {
            label: Some("brushcraft"),
            required_features: adapter.features() & OPTIONAL_FEATURES,
            required_limits: adapter.limits(),
            ..DeviceDescriptor::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))?;
        Ok(Self {
            adapter_info,
            device,
            queue,
        })
    }

    /// The adapter the device was created on.
    #[must_use]
    pub const fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    /// The device, to create buffers, pipelines and command encoders on.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// The device's queue, to submit command buffers and write buffers through.
    #[must_use]
    pub const fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Nanoseconds per timestamp tick, if the device can time GPU work with timestamp queries;
    /// `None` if the adapter doesn't offer them.
    #[must_use]
    pub fn timestamp_period(&self) -> Option<f32> {
        self.device
            .features()
            .contains(Features::TIMESTAMP_QUERY)
            .then(|| self.queue.get_timestamp_period())
    }

    /// Builds a compute pipeline from `source`, running its `entry_point`, with the layout the
    /// shader implies.
    ///
    /// `label` names the shader and pipeline in errors and in GPU debuggers.
    ///
    /// # Errors
    ///
    /// [`GpuError::Validation`] if the shader doesn't compile or validate, or has no such entry
    /// point.
    pub fn compute_pipeline(
        &self,
        label: &str,
        source: ShaderSource<'_>,
        entry_point: &str,
    ) -> Result<ComputePipeline, GpuError> {
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source,
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        match pollster::block_on(scope.pop()) {
            None => Ok(pipeline),
            Some(error) => Err(GpuError::Validation {
                what: label.to_owned(),
                message: describe(&error),
            }),
        }
    }

    /// Copies `source` (which must have [`BufferUsages::COPY_SRC`]) into a fresh mapped
    /// buffer, waits for the GPU, and returns its contents as `T`s.
    ///
    /// Everything submitted before the call has finished when it returns. It blocks, so it is
    /// for tests, tools and benchmarks; the renderer won't wait on the GPU like this.
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer_with`].
    pub fn read_buffer<T: Pod>(&self, source: &Buffer) -> Result<Vec<T>, GpuError> {
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: source.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.read_buffer_with(source, &staging)
    }

    /// [`Gpu::read_buffer`] through a staging buffer the caller keeps, so repeated readbacks
    /// (a benchmark's) don't create one each time.
    ///
    /// `staging` needs [`BufferUsages::COPY_DST`] and [`BufferUsages::MAP_READ`], at least
    /// `source`'s size, and must not be mapped.
    ///
    /// # Errors
    ///
    /// [`GpuError::Misaligned`] if `source` isn't a whole number of `T`s;
    /// [`GpuError::StagingTooSmall`] if `staging` can't hold it; [`GpuError::Poll`] if waiting
    /// fails (the device was lost); [`GpuError::Map`] if mapping fails; [`GpuError::MapRange`] if
    /// the mapped bytes can't be viewed.
    pub fn read_buffer_with<T: Pod>(
        &self,
        source: &Buffer,
        staging: &Buffer,
    ) -> Result<Vec<T>, GpuError> {
        let size = source.size();
        let element = size_of::<T>();
        if element == 0 || !size.is_multiple_of(element as u64) {
            return Err(GpuError::Misaligned { size, element });
        }
        if staging.size() < size {
            return Err(GpuError::StagingTooSmall {
                size,
                staging: staging.size(),
            });
        }
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::channel();
        staging.map_async(MapMode::Read, ..size, move |result| {
            // The receiver outlives the poll below, so the send can't fail in practice; if it
            // did, the `recv` below reports the map as never done.
            drop(sender.send(result));
        });
        self.device.poll(PollType::wait_indefinitely())?;
        receiver
            .recv()
            .unwrap_or(Err(BufferAsyncError))
            .map_err(GpuError::Map)?;
        let values = bytemuck::pod_collect_to_vec(&staging.get_mapped_range(..size)?);
        staging.unmap();
        Ok(values)
    }
}

/// Every adapter the instance [`Gpu::open`] uses can see, in enumeration order.
///
/// For tools that report the machine's GPUs (`examples/adapters.rs`); [`Gpu::open`] chooses
/// among the same adapters.
#[must_use]
pub fn adapters() -> Vec<Adapter> {
    pollster::block_on(instance().enumerate_adapters(Backends::all()))
}

/// The `wgpu` instance adapters are found through: every backend compiled in, with `wgpu`'s
/// environment overrides (`WGPU_BACKEND`, `WGPU_DEBUG`, `WGPU_VALIDATION`) applied.
fn instance() -> Instance {
    Instance::new(InstanceDescriptor::new_without_display_handle_from_env())
}

/// The adapter `request` names, among those `instance` sees.
///
/// # Errors
///
/// [`GpuError::NoAdapter`] if none matches.
async fn choose(instance: &Instance, request: &AdapterRequest) -> Result<Adapter, GpuError> {
    let found = match request {
        AdapterRequest::Default => instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..RequestAdapterOptions::default()
            })
            .await
            .ok(),
        AdapterRequest::Software => instance
            .enumerate_adapters(Backends::all())
            .await
            .into_iter()
            .find(|adapter| adapter.get_info().device_type == DeviceType::Cpu),
        AdapterRequest::Named(name) => {
            let name = name.to_lowercase();
            instance
                .enumerate_adapters(Backends::all())
                .await
                .into_iter()
                .find(|adapter| adapter.get_info().name.to_lowercase().contains(&name))
        }
    };
    match found {
        Some(adapter) => Ok(adapter),
        None => Err(GpuError::NoAdapter {
            request: request.clone(),
            seen: seen(instance).await,
        }),
    }
}

/// The adapters `instance` sees, as `name (backend, device type)`, comma separated, or `none`.
async fn seen(instance: &Instance) -> String {
    let names: Vec<String> = instance
        .enumerate_adapters(Backends::all())
        .await
        .iter()
        .map(|adapter| {
            let info = adapter.get_info();
            format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
        })
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// `error`'s message followed by each of its sources', separated by `: `.
fn describe(error: &wgpu::Error) -> String {
    use std::error::Error as _;
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// Runs a closure against a GPU opened on the software adapter: the harness every GPU test
/// uses, so in the gate they all run on lavapipe, never on the machine's own GPU.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct GpuTest;

#[cfg(test)]
impl GpuTest {
    /// Opens [`AdapterRequest::Software`] and runs `test` with it.
    ///
    /// # Panics
    ///
    /// If there is no software adapter: install lavapipe (`vulkan-swrast` on Arch,
    /// `mesa-vulkan-drivers` on Debian and Ubuntu). And whenever `test` panics.
    pub(crate) fn run<R>(test: impl FnOnce(&Gpu) -> R) -> R {
        let gpu = Gpu::open(&AdapterRequest::Software)
            .unwrap_or_else(|error| panic!("GPU tests need lavapipe: {error}"));
        test(&gpu)
    }
}
