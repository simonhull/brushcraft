//! `brushcraft`: the app. Opens a project and draws it, in a window or offscreen.
//!
//! The main thread owns the project's tile store (it is `!Send`) and runs the frame loop
//! ([`frame`]): input, the brush ([`sculpting`]: the pen sculpts where it touches the surface),
//! the stale-tile feed, the uploads that keep the
//! project's tiles on the GPU ([`uploads`]), then the frame handed to the render thread
//! ([`render_thread`]), which encodes the uploads and the passes, submits and presents them,
//! while the main thread runs the store's jobs. Frames are made only when something asks for
//! one ([`frame::schedule`]). A frame draws the project's surface: the level choice picks the
//! resident tiles to draw, the raster draws their clusters into a visibility buffer, and the
//! shading pass shades it through a MatCap into a linear HDR image, tone mapped, under the debug
//! overlay ([`overlay`]); the camera orbits, pans and zooms with the mouse and the pen
//! ([`navigate`], [`camera`]), and the overlay shows the point of the surface under the
//! pointer, picked by the spatial index ([`pick`]). Keys change how the scene is shown (M the
//! MatCap, W the wireframe, C the debug view of a colour per cluster), each a shading pass alone,
//! and the brush (`[` and `]` its size, `-` and `=` its strength); F frames the model, as opening
//! the project does; Ctrl+S saves ([`saving`]).
//!
//! In a window ([`window`]) the main thread runs winit's event loop. With `--offscreen
//! --frames N` there is no window: frames are drawn into a texture, so the gate opens and
//! closes a project on lavapipe with no display. With `--script` the camera follows the
//! benchmark path ([`script`]) and every frame is recorded, written with `--record` as JSON lines
//! ([`record`]): the renderer's measured run, which `brushcraft-bench perf` makes its render
//! metrics from. An offscreen run with `--screenshot` writes its last frame as a PNG as it ends
//! ([`screenshot`]). Usage: [`cli::USAGE`].

mod camera;
mod cli;
mod dialogs;
mod frame;
mod navigate;
mod overlay;
mod pace;
mod paths;
mod pen;
mod pick;
mod record;
mod render_thread;
mod saving;
mod screenshot;
mod script;
mod sculpting;
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
use crate::frame::{Brush, GpuWait, NoPen, Owner, Setup, Uploads as _};
use crate::pace::{Pacer, SystemTimer};
use crate::pick::Picker;
use crate::render_thread::{Hooks, RenderThread};
use crate::sculpting::Sculpting;
use crate::uploads::GpuResidency;
use crate::window::WindowApp;

/// The site the app's op ids carry: one, until sessions on several machines edit one project.
const SITE: u16 = 1;

/// The app's frame loop: the pen sculpts, the project's tiles kept on the GPU.
pub(crate) type AppOwner = Owner<Sculpting, GpuResidency>;

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
            let brush = Sculpting::new(&mut project);
            let owner = Owner::new(project, brush, residency, setup, Instant::now());
            // Opening a project starts framed (a scripted run's path sets its view instead).
            Ok(match picker {
                Some(picker) => owner.with_picker(picker).framed(),
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
    /// Where the pen stream came from, as the overlay's pen line shows it.
    pub(crate) pen: String,
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
            pen: owner.pen_status().to_string(),
            phases: owner
                .phase_log()
                .map(frame::PhaseLog::lines)
                .unwrap_or_default(),
            records: owner.record_lines(),
        }
    }
}

/// A window's run as it ended: how it went, and its frame loop if it got that far, to close once
/// the window and the event loop are gone, however the run went.
#[derive(Debug)]
pub(crate) struct Ended {
    /// How the run went.
    pub(crate) run: Result<Summary>,
    /// The frame loop, holding the project.
    pub(crate) owner: Option<AppOwner>,
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
    let Ok(ended) = outcome.try_recv() else {
        ran?;
        bail!("the event loop ended without closing the project");
    };
    settled(ended.run, ran, ended.owner, AppOwner::close)
}

/// How a window's run ended, once `owner` has closed through `close`, whatever `run` and the
/// event loop's `ran` say.
///
/// So what it holds lands and the project closes after any failure. The first error is the one
/// returned, with those after it as its context.
///
/// # Errors
///
/// Any of the three.
fn settled<O>(
    run: Result<Summary>,
    ran: Result<()>,
    owner: Option<O>,
    close: impl FnOnce(O) -> Result<()>,
) -> Result<Summary> {
    let closed = owner.map_or(Ok(()), close);
    said_too(said_too(run, ran), closed)
}

/// `first`, or `later`'s error if it was fine; if both failed, the first error with the later as
/// its context, so neither goes unsaid.
///
/// # Errors
///
/// Either's.
pub(crate) fn said_too<T>(first: Result<T>, later: Result<()>) -> Result<T> {
    match (first, later) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) | (Err(error), Ok(())) => Err(error),
        (Err(error), Err(cause)) => Err(error.context(format!("and then: {cause:#}"))),
    }
}

/// Runs the app offscreen for `frames` frames, then closes the project.
///
/// The GPU, the target and the render thread come first, the project last. With
/// `--screenshot`, a run that went well writes the last frame it drew once the render thread has
/// stopped.
///
/// # Errors
///
/// If the GPU or the target can't be opened, the project doesn't open, a frame fails, the
/// screenshot can't be written, or the project doesn't close.
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
        blend: render.blend(),
        adapter: adapter.clone(),
        frames: options.frames(),
        script: options.script(),
        record: options.record_setup(render.timing(), opening),
    };
    let mut owner = start(project, Arc::clone(&gpu), setup)?;
    let ran = drive_offscreen(&mut owner, &mut render);
    let stopped = render.stop().and_then(|stopped| {
        if ran.is_ok()
            && let Some(path) = &options.screenshot
            && let Target::Offscreen(offscreen) = &stopped.target
        {
            screenshot::save(offscreen, &gpu, path)?;
        }
        owner.last_gpu_times(stopped.gpu);
        Ok(())
    });
    let summary = Summary::of(&owner, adapter, presentation);
    let closed = owner.close();
    ran.and(stopped).and(closed)?;
    Ok(summary)
}

/// Runs the frame loop with no event loop until the scripted frames are done.
///
/// Where a window would wait for events it waits for the next frame precisely ([`pace`]), and
/// for the upkeep by a plain sleep; it waits on the render thread directly (which only encodes
/// and submits), but no later than a scripted pen's next sample ([`Owner::pen_due`]), as a
/// window's loop wakes for the pen. It never waits on the GPU: while the GPU is too far behind
/// for the next frame it polls the device every [`frame::GPU_POLL`], running the brush's work
/// between frames or the store's jobs meanwhile. Wherever it would wait, a brush stroke's work
/// between frames comes first ([`Owner::between_frames`]).
///
/// # Errors
///
/// If a frame, the upkeep, the brush, the store's jobs or the render thread fails, or the device
/// is lost.
fn drive_offscreen<B: Brush>(
    owner: &mut Owner<B, GpuResidency>,
    render: &mut RenderThread,
) -> Result<()> {
    let mut pacer = Pacer::default();
    let mut pen = NoPen;
    while !owner.done() {
        let now = Instant::now();
        match owner.next(now) {
            Next::Frame { deadline } => owner.frame(now, deadline, |frame| render.send(frame))?,
            Next::Render if owner.between_frames(now, &mut pen)? => {
                let finished = render.finished()?;
                for presented in finished {
                    owner.shown(presented);
                }
            }
            Next::Render => {
                let presented = match owner.pen_due() {
                    Some(due) => render.wait_until(due)?,
                    None => Some(render.wait()?),
                };
                if let Some(presented) = presented {
                    owner.shown(presented);
                }
            }
            Next::Gpu => match owner.wait_for_gpu(now, &mut pen, || render.gpu_finished())? {
                GpuWait::Ready | GpuWait::Worked => {}
                GpuWait::Idle => std::thread::sleep(frame::GPU_POLL),
            },
            Next::Wait { .. } if owner.between_frames(now, &mut pen)? => {}
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
    writeln!(out, "pen: {}", summary.pen)?;
    for line in &summary.phases {
        writeln!(out, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use anyhow::{Result, anyhow};
    use render::surface::Presentation;

    use super::{Summary, settled};

    /// A run's summary, its details aside.
    fn summary() -> Summary {
        Summary {
            frames: 7,
            adapter: String::new(),
            presentation: Presentation::Offscreen,
            tiles: String::new(),
            scene: String::new(),
            pen: String::new(),
            phases: Vec::new(),
            records: Vec::new(),
        }
    }

    /// However the run or its event loop failed, the frame loop closes: what it holds lands, and
    /// the op ends. An early return on the error would lose them.
    #[test]
    fn a_run_that_failed_still_closes_its_frame_loop_and_says_every_error() {
        let failed = || Err(anyhow!("the device was lost"));
        let cases: [(Result<Summary>, Result<()>); 4] = [
            (Ok(summary()), Ok(())),
            (failed(), Ok(())),
            (Ok(summary()), Err(anyhow!("the loop failed"))),
            (failed(), Err(anyhow!("the loop failed"))),
        ];
        for (run, ran) in cases {
            let failing = run.is_err() || ran.is_err();
            let mut closed = false;
            let outcome = settled(run, ran, Some(()), |()| {
                closed = true;
                Ok(())
            });
            assert!(closed, "closed all the same");
            assert_eq!(outcome.is_err(), failing);
        }
        let both = settled(failed(), Ok(()), Some(()), |()| Err(anyhow!("no room")));
        let said = format!("{:#}", both.expect_err("failed"));
        assert!(
            said.contains("the device was lost") && said.contains("no room"),
            "{said}"
        );
        let closing = settled(Ok(summary()), Ok(()), Some(()), |()| {
            Err(anyhow!("no room"))
        });
        assert!(closing.is_err(), "a close that failed fails the run");
    }
}
