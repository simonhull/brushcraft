//! Opening a GPU: choosing an adapter, creating its device, and the few helpers every piece of
//! GPU work shares (building a compute pipeline with its errors returned, reading a buffer back).
//!
//! `wgpu` panics on errors by default; here every error comes back as a [`GpuError`].
//!
//! Everything here works headless: no window or surface is involved, so tests, benchmarks and
//! tools open a [`Gpu`] the same way the renderer will. Which adapter is used is the caller's
//! choice, as an [`AdapterRequest`]: the system's preferred one, a software rasterizer (the
//! gate's lavapipe), or one picked by name.
//!
//! Only the Vulkan and Metal backends are compiled in: Vulkan on Linux and Android, Metal on
//! macOS and iOS.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use bytemuck::Pod;
use thiserror::Error;
use wgpu::{
    Adapter, AdapterInfo, Backends, Buffer, BufferAsyncError, BufferDescriptor, BufferUsages,
    COPY_BUFFER_ALIGNMENT, CommandEncoderDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, DeviceType, ErrorFilter, Features, Instance, InstanceDescriptor,
    MapMode, MapRangeError, PollError, PollType, PowerPreference, Queue, RequestAdapterOptions,
    RequestDeviceError, ShaderModuleDescriptor, ShaderSource,
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
    /// `wgpu` rejected the work: a shader that doesn't compile or validate, a pipeline that
    /// doesn't match its layout, a copy out of bounds, a buffer used without the usage it needs.
    #[error("{what} failed validation: {message}")]
    Validation {
        /// What was being done, from the caller's label.
        what: String,
        /// `wgpu`'s description of the failure, with its sources.
        message: String,
    },
    /// The device ran out of memory doing the work.
    #[error("{what} ran out of GPU memory: {message}")]
    OutOfMemory {
        /// What was being done, from the caller's label.
        what: String,
        /// `wgpu`'s description of the failure, with its sources.
        message: String,
    },
    /// The backend failed where `wgpu`'s validation passed: a shader the driver's compiler
    /// rejects, for one.
    #[error("{what} failed inside the GPU backend: {message}")]
    Internal {
        /// What was being done, from the caller's label.
        what: String,
        /// `wgpu`'s description of the failure, with its sources.
        message: String,
    },
    /// The device is lost (a driver failure or reset, or it was destroyed); every later call on
    /// this [`Gpu`] fails the same way. Open a new one.
    #[error("the GPU device is lost: {message}")]
    DeviceLost {
        /// The reason the driver gave.
        message: String,
    },
    /// Work done directly on [`Gpu::device`], outside this module's error scopes, raised an
    /// error since the last call. Reported once.
    #[error("an earlier GPU call failed: {message}")]
    Uncaptured {
        /// `wgpu`'s description of the failure, with its sources.
        message: String,
    },
    /// Mapping a buffer for reading failed.
    #[error("mapping a buffer for reading failed")]
    Map(#[from] BufferAsyncError),
    /// A mapped buffer couldn't be viewed.
    #[error("viewing a mapped buffer failed")]
    MapRange(#[from] MapRangeError),
    /// Waiting for the GPU didn't finish within [`WAIT_LIMIT`] (or was given another device's
    /// submission).
    #[error("waiting for the GPU failed")]
    Poll(#[from] PollError),
    /// The GPU finished waiting but never reported the buffer mapped.
    #[error("a buffer mapping never completed")]
    MapPending,
    /// A buffer's size isn't a whole number of the elements it is read back as.
    #[error("a buffer of {size} bytes isn't a whole number of {element}-byte elements")]
    Misaligned {
        /// The buffer's size in bytes.
        size: u64,
        /// The size of one element in bytes.
        element: usize,
    },
    /// A buffer to copy isn't a whole number of 4-byte words, as GPU copies require.
    #[error("a {size}-byte buffer can't be copied: copies move whole 4-byte words")]
    UnalignedCopy {
        /// The buffer's size in bytes.
        size: u64,
    },
    /// A buffer to read back lacks [`BufferUsages::COPY_SRC`].
    #[error("a buffer without COPY_SRC can't be read back")]
    NotCopySource,
    /// A staging buffer lacks [`BufferUsages::COPY_DST`] or [`BufferUsages::MAP_READ`].
    #[error("a staging buffer needs COPY_DST and MAP_READ")]
    StagingUsage,
    /// A staging buffer is smaller than the buffer read through it.
    #[error("a {staging}-byte staging buffer can't hold a {size}-byte readback")]
    StagingTooSmall {
        /// The size of the buffer read, in bytes.
        size: u64,
        /// The staging buffer's size, in bytes.
        staging: u64,
    },
}

/// How long a blocking wait for the GPU may take before it is reported as a timeout: far longer
/// than any work here needs, even on a software adapter.
const WAIT_LIMIT: Duration = Duration::from_secs(30);

/// What the device reported outside any call: set by its callbacks, read by [`Gpu::check`].
#[derive(Debug, Default)]
struct Health {
    /// Why the device was lost, once it is: permanent.
    lost: Option<String>,
    /// The first error raised outside an error scope since the last check.
    uncaptured: Option<String>,
}

/// An open GPU: the adapter's description, and the device and queue created on it.
///
/// Opened headless by [`Gpu::open`]. The device has the adapter's full limits (the work here
/// sizes its buffers to what the adapter allows), and timestamp queries when the adapter offers
/// them; [`Gpu::timestamp_period`] says whether dispatches can be timed.
///
/// Errors never panic: work done through its methods runs inside error scopes, and the
/// device's lost and uncaptured-error callbacks only record what happened, which the next call
/// returns as a [`GpuError`].
#[derive(Debug)]
pub struct Gpu {
    /// The adapter the device was created on: name, backend, device type, driver.
    adapter_info: AdapterInfo,
    /// The logical device everything is created on.
    device: Device,
    /// The device's queue, where command buffers are submitted and buffers written.
    queue: Queue,
    /// What the device's callbacks recorded.
    health: Arc<Mutex<Health>>,
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
        let health = Arc::new(Mutex::new(Health::default()));
        let on_lost = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, message| {
            record(&on_lost).lost = Some(format!("{reason:?}: {message}"));
        });
        let on_error = Arc::clone(&health);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            record(&on_error)
                .uncaptured
                .get_or_insert_with(|| describe(&error));
        }));
        Ok(Self {
            adapter_info,
            device,
            queue,
            health,
        })
    }

    /// The adapter the device was created on.
    #[must_use]
    pub const fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
    }

    /// The device, to create buffers, pipelines and command encoders on.
    ///
    /// Errors from work done on it directly aren't scoped: [`Gpu::check`] (and every method of
    /// this type) reports the first of them.
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

    /// Whether the device is still healthy: not lost, and no unscoped error raised since the
    /// last check.
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] once the device is lost, from then on;
    /// [`GpuError::Uncaptured`] once for an error raised outside an error scope.
    pub fn check(&self) -> Result<(), GpuError> {
        let mut health = record(&self.health);
        if let Some(message) = &health.lost {
            return Err(GpuError::DeviceLost {
                message: message.clone(),
            });
        }
        match health.uncaptured.take() {
            Some(message) => Err(GpuError::Uncaptured { message }),
            None => Ok(()),
        }
    }

    /// Runs `work` on the device inside validation, internal and out-of-memory error scopes,
    /// so its errors come back as values; `what` names it in them.
    ///
    /// # Errors
    ///
    /// The scope's error as [`GpuError::Validation`], [`GpuError::Internal`] or
    /// [`GpuError::OutOfMemory`]; and as [`Gpu::check`], before and after.
    pub fn scoped<T>(&self, what: &str, work: impl FnOnce() -> T) -> Result<T, GpuError> {
        self.check()?;
        let out_of_memory = self.device.push_error_scope(ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(ErrorFilter::Internal);
        let validation = self.device.push_error_scope(ErrorFilter::Validation);
        let value = work();
        // Popped innermost first, all of them, before any is awaited.
        let popped = [validation.pop(), internal.pop(), out_of_memory.pop()];
        for scope in popped {
            if let Some(error) = pollster::block_on(scope) {
                return Err(classify(what, &error));
            }
        }
        self.check()?;
        Ok(value)
    }

    /// Builds a compute pipeline from `source`, running its `entry_point`, with the layout the
    /// shader implies.
    ///
    /// `label` names the shader and pipeline in errors and in GPU debuggers.
    ///
    /// # Errors
    ///
    /// [`GpuError::Validation`] if the shader doesn't compile or validate, or has no such entry
    /// point; [`GpuError::Internal`] if the backend's compiler rejects it; and as
    /// [`Gpu::scoped`].
    pub fn compute_pipeline(
        &self,
        label: &str,
        source: ShaderSource<'_>,
        entry_point: &str,
    ) -> Result<ComputePipeline, GpuError> {
        self.scoped(label, || {
            let module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source,
            });
            self.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
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
        let staging = self.scoped("readback staging", || {
            self.device.create_buffer(&BufferDescriptor {
                label: Some("readback"),
                size: source.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })?;
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
    /// Before touching the GPU: [`GpuError::NotCopySource`], [`GpuError::StagingUsage`],
    /// [`GpuError::UnalignedCopy`] (copies move whole 4-byte words), [`GpuError::Misaligned`]
    /// (not a whole number of `T`s), [`GpuError::StagingTooSmall`]. Then: the copy's and map's
    /// errors as [`Gpu::scoped`]; [`GpuError::Poll`] if the GPU doesn't finish within
    /// [`WAIT_LIMIT`]; [`GpuError::Map`] or [`GpuError::MapPending`] if the mapping fails or
    /// never completes; [`GpuError::MapRange`] if the mapped bytes can't be viewed.
    pub(crate) fn read_buffer_with<T: Pod>(
        &self,
        source: &Buffer,
        staging: &Buffer,
    ) -> Result<Vec<T>, GpuError> {
        let size = source.size();
        let element = size_of::<T>();
        if !source.usage().contains(BufferUsages::COPY_SRC) {
            return Err(GpuError::NotCopySource);
        }
        if !staging
            .usage()
            .contains(BufferUsages::COPY_DST | BufferUsages::MAP_READ)
        {
            return Err(GpuError::StagingUsage);
        }
        if !size.is_multiple_of(COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::UnalignedCopy { size });
        }
        if element == 0 || !size.is_multiple_of(element as u64) {
            return Err(GpuError::Misaligned { size, element });
        }
        if staging.size() < size {
            return Err(GpuError::StagingTooSmall {
                size,
                staging: staging.size(),
            });
        }
        let (sender, receiver) = mpsc::channel();
        self.scoped("readback", || {
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
            self.queue.submit([encoder.finish()]);
            staging.map_async(MapMode::Read, ..size, move |result| {
                // If the receiver is gone the caller stopped waiting; nothing to tell it.
                drop(sender.send(result));
            });
        })?;
        self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: Some(WAIT_LIMIT),
        })?;
        self.check()?;
        receiver
            .try_recv()
            .map_err(|_pending| GpuError::MapPending)??;
        let values = bytemuck::pod_collect_to_vec(&staging.get_mapped_range(..size)?);
        staging.unmap();
        Ok(values)
    }
}

/// Locks the device's health record. A callback that panicked while holding it can't have left
/// it half-written (each writes one field), so a poisoned lock is used as it is.
fn record(health: &Mutex<Health>) -> MutexGuard<'_, Health> {
    health.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `error`, raised while doing `what`, as the [`GpuError`] of its kind.
fn classify(what: &str, error: &wgpu::Error) -> GpuError {
    let what = what.to_owned();
    let message = describe(error);
    match error {
        wgpu::Error::Validation { .. } => GpuError::Validation { what, message },
        wgpu::Error::OutOfMemory { .. } => GpuError::OutOfMemory { what, message },
        wgpu::Error::Internal { .. } => GpuError::Internal { what, message },
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
