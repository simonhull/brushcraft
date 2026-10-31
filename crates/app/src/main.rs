//! `brushcraft`: the app. Opens a project and draws it, in a window or offscreen.
//!
//! The main thread owns the project's tile store (it is `!Send`) and runs the frame loop
//! ([`frame`]): input, the brush (M1a4), the stale-tile feed, the uploads that keep the
//! project's tiles on the GPU ([`uploads`]), then the frame handed to the render thread
//! ([`render_thread`]), which encodes the uploads and the passes, submits and presents them,
//! while the main thread runs the store's jobs. Frames are made only when something asks for
//! one ([`frame::schedule`]). A frame draws the project's surface: the level choice picks the
//! resident tiles to draw, the raster draws their clusters into a visibility buffer, and the
//! shading pass shades it through a MatCap into a linear HDR image, tone mapped, under the debug
//! overlay ([`overlay`]); the camera orbits, pans and zooms with the mouse and the pen
//! ([`navigate`], [`camera`]), and the overlay shows the point of the surface under the
//! pointer, picked by the spatial index ([`pick`]). Keys change how the scene is shown (M the
//! MatCap, W the wireframe, C the debug view of a colour per cluster), each a shading pass alone.
//!
//! In a window ([`window`]) the main thread runs winit's event loop. With `--offscreen
//! --frames N` there is no window: frames are drawn into a texture, so the gate opens and
//! closes a project on lavapipe with no display. With `--script` the camera follows the
//! benchmark path ([`script`]) and every frame is recorded, written with `--record` as JSON lines
//! ([`record`]): the renderer's measured run, which `brushcraft-bench perf` makes its render
//! metrics from. Usage: [`cli::USAGE`].

mod camera;
mod cli;
mod frame;
mod navigate;
mod overlay;
mod pace;
mod paths;
mod pick;
mod record;
mod render_thread;
mod script;
#[cfg(test)]
mod test_dir;
mod uploads;
mod window;

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use render::gpu::{Gpu, GpuError};
use render::surface::{Offscreen, Presentation, Target};
use store::{Project, ProjectConfig, StoreConfig, default_recovery_dir};
use winit::event_loop::EventLoop;

use crate::cli::Options;
use crate::frame::schedule::Next;
use crate::frame::{NoBrush, Owner, Setup, Uploads as _};
use crate::pace::{Pacer, SystemTimer};
use crate::pick::Picker;
use crate::render_thread::{Hooks, RenderThread};
use crate::uploads::GpuResidency;
use crate::window::WindowApp;

/// The site the app's op ids carry: one, until sessions on several machines edit one project.
const SITE: u16 = 1;

/// The app's frame loop: no brush yet (M1a4), the project's tiles kept on the GPU.
pub(crate) type AppOwner = Owner<NoBrush, GpuResidency>;

/// The frame loop over `project` as `setup` says, its tiles kept on `gpu`, wanting its first
/// frame now, picking the surface under the pointer by the project's spatial index.
///
/// # Errors
///
/// If GPU residency or the scene's passes (the level choice, the raster) can't be set up on
/// this GPU, or the spatial index can't be loaded: at startup, never mid-session. The project
/// is closed first.
pub(crate) fn start(mut project: Project, gpu: Arc<Gpu>, mut setup: Setup) -> Result<AppOwner> {
    let ready = GpuResidency::new(gpu, project.store())
        .and_then(|mut residency| {
            // A recorded run with the GPU timed times the level choice's stages too.
            if let Some(record) = setup.record.as_mut().filter(|record| record.gpu_timing) {
                record.lod_timing = residency.enable_timing()?;
            }
            Ok(residency)
        })
        .and_then(|residency| Ok((residency, Picker::load(&mut project)?)));
    match ready {
        Ok((residency, picker)) => {
            let owner = Owner::new(project, NoBrush, residency, setup, Instant::now());
            Ok(match picker {
                Some(picker) => owner.with_picker(picker),
                None => owner,
            })
        }
        Err(error) => {
            let keep = project.has_unsaved_edits();
            match project.close(keep) {
                Ok(()) => Err(error),
                Err(close) => {
                    Err(error.context(format!("and closing the project failed: {close}")))
                }
            }
        }
    }
}

/// How a run went, for the lines printed at its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Summary {
    /// Frames the render thread finished.
    pub(crate) frames: u64,
    /// The GPU's description.
    pub(crate) adapter: String,
    /// How frames reached the screen.
    pub(crate) presentation: Presentation,
    /// What the GPU held at the end, as the overlay's tiles line shows it.
    pub(crate) tiles: String,
    /// What the last frame read back drew, as the overlay's scene line shows it.
    pub(crate) scene: String,
    /// A scripted run's phase times ([`frame::PhaseLog::lines`]); empty for an open-ended run.
    pub(crate) phases: Vec<String>,
    /// A recorded run's record lines ([`record`]); empty for any other run.
    pub(crate) records: Vec<String>,
}

impl Summary {
    /// The summary of `owner`'s run: `adapter` presenting as `presentation`. Take it before
    /// the owner closes.
    pub(crate) fn of(owner: &AppOwner, adapter: String, presentation: Presentation) -> Self {
        let residency = owner.uploads();
        Self {
            frames: owner.finished(),
            adapter,
            presentation,
            tiles: uploads::summary(&residency.stats(), residency.floor_tiles()),
            scene: uploads::scene_summary(residency.scene()),
            phases: owner
                .phase_log()
                .map(frame::PhaseLog::lines)
                .unwrap_or_default(),
            records: owner.record_lines(),
        }
    }
}

/// A GPU's name and backend, as the overlay and the summary show it.
pub(crate) fn describe_adapter(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?})", info.name, info.backend)
}

/// A project checked and ready to open, opened only once the GPU (and the window) are.
///
/// Opening a project makes a recovery file, so a launch that fails before the project opens
/// leaves none behind, and one that fails after closes it.
#[derive(Debug, Clone)]
pub(crate) struct ProjectToOpen {
    /// The project file: absolute, and outside the repository.
    path: PathBuf,
    /// How to open it.
    config: ProjectConfig,
}

impl ProjectToOpen {
    /// The project `options` name, with a display layer under `options.level` and its recovery
    /// file in the app's recovery directory.
    ///
    /// # Errors
    ///
    /// If the path is refused (missing, or inside the repository) or there is no recovery
    /// directory.
    fn new(options: &Options) -> Result<Self> {
        let path = paths::project(&options.project)?;
        let recovery_dir = default_recovery_dir()
            .ok_or_else(|| anyhow!("neither XDG_STATE_HOME nor HOME is an absolute path"))?;
        let config = ProjectConfig {
            store: StoreConfig::new(options.budget_bytes, SITE).with_display(options.level),
            recovery_dir,
        };
        Ok(Self { path, config })
    }

    /// Opens it.
    ///
    /// # Errors
    ///
    /// If the project doesn't open (it leaves no recovery file then).
    pub(crate) fn open(self) -> Result<Project> {
        Project::open(&self.path, self.config)
            .with_context(|| format!("opening {}", self.path.display()))
    }
}

/// Runs the app in a window until it closes (or has drawn a scripted run's frames).
///
/// # Errors
///
/// If the event loop or the window fails, or the frame loop does.
fn run_window(options: Options, project: ProjectToOpen) -> Result<Summary> {
    let (done, outcome) = mpsc::channel();
    let ran = EventLoop::new()
        .context("starting the event loop")?
        .run_app(WindowApp::new(options, project, done))
        .context("running the event loop");
    let Ok(outcome) = outcome.try_recv() else {
        ran?;
        bail!("the event loop ended without closing the project");
    };
    let summary = outcome?;
    ran?;
    Ok(summary)
}

/// Runs the app offscreen for `frames` frames, then closes the project. The GPU, the target
/// and the render thread come first, the project last.
///
/// # Errors
///
/// If the GPU or the target can't be opened, the project doesn't open, a frame fails, or the
/// project doesn't close.
fn run_offscreen(options: &Options, project: ProjectToOpen) -> Result<Summary> {
    let gpu = Arc::new(Gpu::open(&options.adapter).context("opening the GPU")?);
    let adapter = describe_adapter(gpu.adapter_info());
    let target = Target::Offscreen(Offscreen::new(&gpu, options.size)?);
    let timing = options.times_gpu();
    let mut render = RenderThread::spawn(Arc::clone(&gpu), target, Hooks::none(), timing)?;
    let opening = Instant::now();
    let project = project.open()?;
    let presentation = render.presentation();
    let setup = Setup {
        size: Some(options.size),
        interval: window::DEFAULT_INTERVAL,
        presentation,
        adapter: adapter.clone(),
        frames: options.frames(),
        script: options.script(),
        record: options.record_setup(render.timing(), opening),
    };
    let mut owner = start(project, gpu, setup)?;
    let ran = drive_offscreen(&mut owner, &mut render);
    let stopped = render.stop().map(drop);
    let summary = Summary::of(&owner, adapter, presentation);
    let closed = owner.close();
    ran.and(stopped).and(closed)?;
    Ok(summary)
}

/// Runs the frame loop with no event loop until the scripted frames are done.
///
/// Where a window would wait for events it waits for the next frame precisely ([`pace`]), and
/// for the upkeep by a plain sleep; it waits on the render thread directly (which only encodes
/// and submits). It never waits on the GPU: while the GPU is too far behind for the next frame
/// it polls the device every [`frame::GPU_POLL`], running the store's jobs meanwhile.
///
/// # Errors
///
/// If a frame, the upkeep, the store's jobs or the render thread fails, or the device is lost.
fn drive_offscreen(owner: &mut AppOwner, render: &mut RenderThread) -> Result<()> {
    let mut pacer = Pacer::default();
    while !owner.done() {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => owner.frame(now, deadline, |frame| render.send(frame))?,
            Next::Render => owner.shown(render.wait()?),
            Next::Gpu => {
                owner.gpu_finished(render.gpu_finished()?);
                let still = matches!(owner.next(now), Next::Gpu);
                if still && !owner.while_gpu_busy(now, now + frame::GPU_POLL)? {
                    std::thread::sleep(frame::GPU_POLL);
                }
            }
            Next::Wait { until } => drop(pacer.wait_until(&mut SystemTimer, until)),
            Next::Idle { upkeep } => {
                std::thread::sleep(upkeep.saturating_duration_since(now));
                owner.upkeep(Instant::now())?;
            }
        }
    }
    Ok(())
}

/// The exit status when no GPU matches `--adapter`: `EX_UNAVAILABLE`, so a caller (the
/// performance suite) can tell a machine with no GPU from a run that failed.
const NO_ADAPTER: u8 = 69;

/// Runs the app ([`run`]); on an error, prints it and exits with a failure, [`NO_ADAPTER`] if
/// no GPU matched.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // Nothing more to do if stderr is gone.
            drop(writeln!(std::io::stderr(), "Error: {error:?}"));
            let no_adapter = error
                .chain()
                .any(|cause| matches!(cause.downcast_ref(), Some(GpuError::NoAdapter { .. })));
            if no_adapter {
                ExitCode::from(NO_ADAPTER)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

/// Parses the command line, checks the project, runs, writes a recorded run's record, and
/// prints a line saying how it went.
///
/// # Errors
///
/// If the arguments don't parse, the project doesn't open, the run fails, or the record can't
/// be written.
fn run() -> Result<()> {
    let options = Options::parse(std::env::args_os().skip(1))?;
    let project = ProjectToOpen::new(&options)?;
    let summary = if options.mode.offscreen {
        run_offscreen(&options, project)?
    } else {
        run_window(options.clone(), project)?
    };
    if let Some(path) = &options.record {
        let mut text = summary.records.join("\n");
        text.push('\n');
        std::fs::write(path, text)
            .with_context(|| format!("writing the record to {}", path.display()))?;
    }
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "brushcraft: {} frames on {} ({}); {} closed",
        summary.frames,
        summary.adapter,
        summary.presentation,
        options.project.display()
    )?;
    writeln!(out, "tiles: {}", summary.tiles)?;
    writeln!(out, "scene: {}", summary.scene)?;
    for line in &summary.phases {
        writeln!(out, "{line}")?;
    }
    Ok(())
}
