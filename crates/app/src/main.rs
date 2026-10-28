//! `brushcraft`: the app. Opens a project and draws it, in a window or offscreen.
//!
//! The main thread owns the project's tile store (it is `!Send`) and runs the frame loop
//! ([`frame`]): input, the brush (M1a4), the stale-tile feed, uploads (M1a3 Task 2), the
//! store's jobs, then the frame handed to the render thread ([`render_thread`]), which encodes,
//! submits and presents it. Frames are made only when something asks for one
//! ([`frame::schedule`]). For now a frame is a clear colour under the debug overlay
//! ([`overlay`]); the camera orbits, pans and zooms with the mouse and the pen ([`navigate`],
//! [`camera`]).
//!
//! In a window ([`window`]) the main thread runs winit's event loop. With `--offscreen
//! --frames N` there is no window: frames are drawn into a texture, so the gate opens and
//! closes a project on lavapipe with no display. Usage: [`cli::USAGE`].

mod camera;
mod cli;
mod frame;
mod navigate;
mod overlay;
mod paths;
mod render_thread;
#[cfg(test)]
mod test_dir;
mod window;

use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, mpsc};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use render::gpu::Gpu;
use render::surface::{Offscreen, Presentation, Target};
use store::{Project, ProjectConfig, StoreConfig, default_recovery_dir};
use winit::event_loop::EventLoop;

use crate::cli::{Mode, Options};
use crate::frame::schedule::Next;
use crate::frame::{NoBrush, NoUploads, Owner, Setup};
use crate::render_thread::{Hooks, RenderThread};
use crate::window::WindowApp;

/// The site the app's op ids carry: one, until sessions on several machines edit one project.
const SITE: u16 = 1;

/// How a run went, for the line printed at its end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Summary {
    /// Frames the render thread finished.
    pub(crate) frames: u64,
    /// The GPU's description.
    pub(crate) adapter: String,
    /// How frames reached the screen.
    pub(crate) presentation: Presentation,
}

/// A GPU's name and backend, as the overlay and the summary show it.
pub(crate) fn describe_adapter(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?})", info.name, info.backend)
}

/// Opens the project at `path` with a display layer under `options.level`, its recovery file
/// in the app's recovery directory.
///
/// # Errors
///
/// If the path is refused (missing, or inside the repository), there is no recovery
/// directory, or the project doesn't open.
fn open_project(path: &Path, options: &Options) -> Result<Project> {
    let path = paths::project(path)?;
    let recovery_dir = default_recovery_dir()
        .ok_or_else(|| anyhow!("neither XDG_STATE_HOME nor HOME is an absolute path"))?;
    let config = ProjectConfig {
        store: StoreConfig::new(options.budget_bytes, SITE).with_display(options.level),
        recovery_dir,
    };
    Project::open(&path, config).with_context(|| format!("opening {}", path.display()))
}

/// Runs the app in a window until it closes (or has drawn a scripted run's frames).
///
/// # Errors
///
/// If the event loop or the window fails, or the frame loop does.
fn run_window(options: Options, project: Project) -> Result<Summary> {
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

/// Runs the app offscreen for `frames` frames, then closes the project.
///
/// # Errors
///
/// If the GPU or the target can't be opened, a frame fails, or the project doesn't close.
fn run_offscreen(options: &Options, project: Project, frames: u64) -> Result<Summary> {
    let gpu = Arc::new(Gpu::open(&options.adapter).context("opening the GPU")?);
    let adapter = describe_adapter(gpu.adapter_info());
    let target = Target::Offscreen(Offscreen::new(&gpu, options.size)?);
    let mut render = RenderThread::spawn(gpu, target, Hooks::none())?;
    let presentation = render.presentation();
    let setup = Setup {
        size: Some(options.size),
        interval: window::DEFAULT_INTERVAL,
        presentation,
        adapter: adapter.clone(),
        frames: Some(frames),
    };
    let mut owner = Owner::new(project, NoBrush, NoUploads, setup, Instant::now());
    let ran = drive_offscreen(&mut owner, &mut render);
    let stopped = render.stop().map(drop);
    let finished = owner.finished();
    let closed = owner.close();
    ran.and(stopped).and(closed)?;
    Ok(Summary {
        frames: finished,
        adapter,
        presentation,
    })
}

/// Runs the frame loop with no event loop until the scripted frames are done: sleeping where
/// a window would wait for events, and waiting on the render thread directly.
///
/// # Errors
///
/// If a frame, the upkeep or the render thread fails.
fn drive_offscreen(owner: &mut Owner<NoBrush, NoUploads>, render: &mut RenderThread) -> Result<()> {
    while !owner.done() {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => render.send(owner.frame(now, deadline)?)?,
            Next::Render => owner.shown(render.wait()?),
            Next::Wait { until } => std::thread::sleep(until.saturating_duration_since(now)),
            Next::Idle { upkeep } => {
                std::thread::sleep(upkeep.saturating_duration_since(now));
                owner.upkeep(Instant::now())?;
            }
        }
    }
    Ok(())
}

/// Parses the command line, opens the project, runs, and prints a line saying how it went.
///
/// # Errors
///
/// If the arguments don't parse, the project doesn't open, or the run fails.
fn main() -> Result<()> {
    let options = Options::parse(std::env::args_os().skip(1))?;
    let project = open_project(&options.project, &options)?;
    let summary = match options.mode {
        Mode::Offscreen { frames } => run_offscreen(&options, project, frames)?,
        Mode::Window { .. } => run_window(options.clone(), project)?,
    };
    writeln!(
        std::io::stdout().lock(),
        "brushcraft: {} frames on {} ({}); {} closed",
        summary.frames,
        summary.adapter,
        summary.presentation,
        options.project.display()
    )?;
    Ok(())
}
