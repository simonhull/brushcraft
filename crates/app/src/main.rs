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
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, mpsc};
use std::time::{Instant, SystemTime};

use anyhow::{Context as _, Result, anyhow, bail};
use render::gpu::{Gpu, GpuError};
use render::surface::{Offscreen, Presentation, Target};
use store::{
    CandidateStatus, Project, ProjectConfig, ProjectError, RecoveryCandidate, StoreConfig,
    default_recovery_dir, find_recoveries,
};
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

/// What becomes of what isn't saved as a window's run ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum End {
    /// It lands, and stays in the recovery file: however a run ends unless the user says otherwise.
    #[default]
    Keep,
    /// It goes: Don't save on the close bar, the one way here.
    Discard,
}

/// A window's run as it ended: how it went, and its frame loop if it got that far, to close once
/// the window and the event loop are gone, however the run went.
#[derive(Debug)]
pub(crate) struct Ended {
    /// How the run went.
    pub(crate) run: Result<Summary>,
    /// The frame loop, holding the project.
    pub(crate) owner: Option<AppOwner>,
    /// What becomes of what isn't saved.
    pub(crate) end: End,
}

/// A GPU's name and backend, as the overlay and the summary show it.
pub(crate) fn describe_adapter(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?})", info.name, info.backend)
}

/// What opening found beyond the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OpenNote {
    /// Its recovery file was restored: it is `path`, written at `when`.
    Restored {
        /// The recovery file, which goes on as the project's.
        path: PathBuf,
        /// When it was last written.
        when: SystemTime,
    },
    /// A recovery file onto an earlier save was found, and not restored.
    EarlierVersion {
        /// What was found.
        candidate: RecoveryCandidate,
    },
}

/// Whether `one` and `other` are one file, however they are spelled: through a symbolic link, with
/// `..` in them, or as two hard links. Paths that can't be followed are no file, so not the same.
fn same_file(one: &Path, other: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let identity = |path| std::fs::metadata(path).map(|file| (file.dev(), file.ino()));
        matches!((identity(one), identity(other)), (Ok(a), Ok(b)) if a == b)
    }
    #[cfg(not(unix))]
    {
        let real = std::fs::canonicalize;
        matches!((real(one), real(other)), (Ok(a), Ok(b)) if a == b)
    }
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

    /// Opens it as the run in `options` should: bringing back what a crash left, or not.
    ///
    /// # Errors
    ///
    /// As [`ProjectToOpen::open`].
    pub(crate) fn open_as(self, options: &Options) -> Result<(Project, Option<OpenNote>)> {
        if options.restores() {
            self.open_restoring()
        } else {
            Ok((self.open()?, None))
        }
    }

    /// Opens it with what the last run left: its restorable recovery file if it has one, and
    /// otherwise as ever, with a note if there's work on an earlier save.
    ///
    /// # Errors
    ///
    /// As [`ProjectToOpen::open`]. Recovery never fails an open.
    pub(crate) fn open_restoring(self) -> Result<(Project, Option<OpenNote>)> {
        let found = find_recoveries(&self.config.recovery_dir);
        self.open_with(found)
    }

    /// [`ProjectToOpen::open_restoring`], with `found` the listing of recovery files. A listing
    /// that failed, or a restore that does, is said on stderr and the project opens as ever: the
    /// user's project must open, and the recovery file is kept for another try.
    ///
    /// # Errors
    ///
    /// As [`ProjectToOpen::open`].
    fn open_with(
        self,
        found: Result<Vec<RecoveryCandidate>, ProjectError>,
    ) -> Result<(Project, Option<OpenNote>)> {
        let found = found.unwrap_or_else(|error| {
            let why = anyhow::Error::new(error);
            saving::log(&format!("brushcraft: looking for recovery files: {why:#}"));
            Vec::new()
        });
        let ours = found.into_iter().filter(|found| self.holds_work_of(found));
        let ours: Vec<RecoveryCandidate> = ours.collect();
        let has = |status: CandidateStatus| {
            let mut candidates = ours.iter();
            candidates.find(|candidate| candidate.status == status)
        };
        if let Some(candidate) = has(CandidateStatus::Restorable) {
            match Project::restore(candidate, self.config.clone()) {
                Ok(project) => {
                    let note = OpenNote::Restored {
                        path: candidate.path.clone(),
                        when: candidate.modified,
                    };
                    return Ok((project, Some(note)));
                }
                Err(error) => {
                    let (path, why) = (candidate.path.display(), anyhow::Error::new(error));
                    saving::log(&format!("brushcraft: restoring {path}: {why:#}"));
                }
            }
        }
        let earlier = has(CandidateStatus::RestorableOnEarlierSave).cloned();
        let note = earlier.map(|candidate| OpenNote::EarlierVersion { candidate });
        Ok((self.open()?, note))
    }

    /// Whether `candidate` has work of this project: it has edits, and the file it extends is the
    /// one being opened, whichever way either path spells it. A copy of the project is another
    /// file, and its project id and save points say nothing of that, so they aren't asked. A
    /// candidate with no edits would bring back nothing to tell the user of.
    fn holds_work_of(&self, candidate: &RecoveryCandidate) -> bool {
        let extends = candidate.base_path.as_deref();
        candidate.last_edit.is_some() && extends.is_some_and(|base| same_file(base, &self.path))
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
    let closing = (ended.owner, ended.end);
    settled(ended.run, ran, closing, AppOwner::close, AppOwner::discard)
}

/// How a window's run ended, once its owner has closed, whatever `run` and the event loop's `ran`
/// say: through `close`, or through `discard` if the run's [`End`] says so.
///
/// So what it holds lands and the project closes after any failure, unless the user said to drop
/// it. The first error is the one returned, with those after it as its context.
///
/// # Errors
///
/// Any of the three.
fn settled<O>(
    run: Result<Summary>,
    ran: Result<()>,
    (owner, end): (Option<O>, End),
    close: impl FnOnce(O) -> Result<()>,
    discard: impl FnOnce(O) -> Result<()>,
) -> Result<Summary> {
    let closed = owner.map_or(Ok(()), |owner| match end {
        End::Keep => close(owner),
        End::Discard => discard(owner),
    });
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
    let (project, note) = project.open_as(options)?;
    if let Some(OpenNote::Restored { path, .. }) = &note {
        writeln!(std::io::stdout(), "restored: {}", path.display())?;
    }
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
    if let Some(note) = &note {
        owner.note(note, Instant::now());
    }
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
    use std::ffi::OsString;

    use anyhow::{Result, anyhow};
    use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
    use render::surface::Presentation;

    use super::*;
    use crate::test_dir::TestDir;

    /// The finest level the test projects keep.
    const FINEST: u8 = 3;

    /// Opening a project at `path`, its recovery files in `dir`'s `recovery`.
    fn to_open(dir: &TestDir, path: &Path) -> ProjectToOpen {
        ProjectToOpen {
            path: path.to_owned(),
            config: ProjectConfig {
                store: StoreConfig::new(512 << 20, SITE),
                recovery_dir: dir.path().join("recovery"),
            },
        }
    }

    /// A saved project called `name` in `dir`, with no edits.
    fn saved(dir: &TestDir, name: &str) -> ProjectToOpen {
        let opening = to_open(dir, &dir.path().join(name));
        let mesh = cube_sphere(1).expect("a cube sphere");
        let mut project = Project::new(mesh, opening.config.clone()).expect("a new project");
        drop(project.save_as(&opening.path).expect("saved"));
        project.close(false).expect("closed");
        opening
    }

    /// Makes one edit of `project`, as a stroke does.
    fn edit(project: &mut Project) {
        let level = Level::new(FINEST).expect("a level");
        let key = TileKey::displacement(FaceId::new(0), level, 0, 0).expect("a tile");
        project.begin_stroke().expect("begun");
        let edited = project
            .store()
            .edit_tile(key, |tile| tile.set(1, 1, Vec3::new(0.0, 0.0, 0.01)));
        edited.expect("the tile").expect("edited");
        project.store().close_epoch().expect("closed");
        project.end_unrecorded_stroke().expect("ended");
    }

    /// Opens `project` plainly, edits it and ends with its recovery file left, as a crash does:
    /// that file's path.
    fn left_work(project: &ProjectToOpen) -> PathBuf {
        let mut opened = project.clone().open().expect("opened");
        edit(&mut opened);
        let path = opened.recovery_path().expect("protected").to_owned();
        opened.close(true).expect("closed, the file kept");
        path
    }

    /// `project`'s recovery files as the app lists them.
    fn listed(project: &ProjectToOpen) -> Vec<RecoveryCandidate> {
        find_recoveries(&project.config.recovery_dir).expect("listed")
    }

    /// The recovery files left by `project`, in order.
    fn left(project: &ProjectToOpen) -> Vec<PathBuf> {
        let files = listed(project).into_iter().map(|found| found.path);
        let mut paths: Vec<PathBuf> = files.collect();
        paths.sort();
        paths
    }

    /// Closes `project`, keeping its recovery file if it has edits, as a run's end does.
    fn close(project: Project) {
        let keep = project.has_unsaved_edits();
        project.close(keep).expect("closed");
    }

    #[test]
    fn a_restorable_recovery_file_is_restored() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        let (project, note) = opening.clone().open_restoring().expect("opened");
        assert!(project.has_unsaved_edits(), "the work is back");
        assert_eq!(project.path(), Some(opening.path.as_path()), "saves here");
        assert_eq!(project.recovery_path(), Some(file.as_path()), "goes on");
        let Some(OpenNote::Restored { path, .. }) = note else {
            panic!("the note says so: {note:?}");
        };
        assert_eq!(path, file);
        close(project);
        assert_eq!(left(&opening), vec![file], "kept while it's unsaved");
    }

    #[test]
    fn the_newest_of_two_is_the_one_restored() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let older = left_work(&opening);
        let newer = left_work(&opening);
        let (project, note) = opening.clone().open_restoring().expect("opened");
        assert_eq!(project.recovery_path(), Some(newer.as_path()));
        assert!(matches!(note, Some(OpenNote::Restored { path, .. }) if path == newer));
        close(project);
        assert!(left(&opening).contains(&older), "the other is left");
    }

    #[test]
    fn a_recovery_file_of_another_project_is_left_alone() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let other = saved(&dir, "second-sculpt.bcrf");
        let theirs = left_work(&other);
        let (project, note) = opening.clone().open_restoring().expect("opened");
        assert_eq!(note, None, "nothing of this project");
        assert!(!project.has_unsaved_edits());
        close(project);
        assert_eq!(left(&opening), vec![theirs], "as it was");
    }

    #[test]
    fn a_copy_of_the_project_is_another_project() {
        // The same project id and save points as the file the work extends: only the path says.
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let copied = to_open(&dir, &dir.path().join("copy.bcrf"));
        std::fs::copy(&opening.path, &copied.path).expect("copied");
        let file = left_work(&opening);
        let (project, note) = copied.clone().open_restoring().expect("opened");
        assert_eq!(note, None, "that work is on the original");
        assert_eq!(project.path(), Some(copied.path.as_path()));
        assert!(!project.has_unsaved_edits());
        close(project);
        assert_eq!(left(&opening), vec![file], "and still there for it");
    }

    #[cfg(unix)]
    #[test]
    fn work_is_found_whichever_way_the_project_is_spelled() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let linked = dir.path().join("linked.bcrf");
        std::os::unix::fs::symlink(&opening.path, &linked).expect("a symbolic link");
        let hard = dir.path().join("hard.bcrf");
        std::fs::hard_link(&opening.path, &hard).expect("a hard link");
        let inner = dir.path().join("inner");
        std::fs::create_dir_all(&inner).expect("a directory");
        let dotted = inner.join("..").join("first-sculpt.bcrf");

        for path in [linked.clone(), hard, dotted] {
            // Made under the real name, opened under this one.
            let file = left_work(&opening);
            let (project, note) = to_open(&dir, &path).open_restoring().expect("opened");
            assert!(project.has_unsaved_edits(), "by {}", path.display());
            assert!(matches!(note, Some(OpenNote::Restored { .. })));
            project.close(false).expect("closed, the file with it");
            assert!(!file.exists(), "that was the one");
        }

        // Made under a link, opened under the real name.
        left_work(&to_open(&dir, &linked));
        let (project, note) = opening.open_restoring().expect("opened");
        assert!(matches!(note, Some(OpenNote::Restored { .. })), "{note:?}");
        project.close(false).expect("closed");
    }

    #[test]
    fn no_recovery_file_opens_as_ever() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let (project, note) = opening.open_restoring().expect("opened");
        assert_eq!(note, None);
        assert!(!project.has_unsaved_edits());
        project.close(false).expect("closed");
    }

    #[test]
    fn a_recovery_file_with_no_edits_brings_nothing_back() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let empty = opening.clone().open().expect("opened");
        let file = empty.recovery_path().expect("protected").to_owned();
        empty.close(true).expect("closed, as a crash leaves it");
        let (project, note) = opening.open_restoring().expect("opened");
        assert_eq!(note, None, "\"Your unsaved work is back\" would be untrue");
        assert_ne!(project.recovery_path(), Some(file.as_path()));
        project.close(false).expect("closed");
    }

    /// Leaves work on `opening`'s current save, then saves the project past it: that file.
    fn left_on_an_earlier_save(opening: &ProjectToOpen) -> PathBuf {
        let file = left_work(opening);
        let mut later = opening.clone().open().expect("opened");
        edit(&mut later);
        drop(later.save().expect("saved"));
        later.close(false).expect("closed");
        file
    }

    #[test]
    fn a_recovery_file_on_an_earlier_save_opens_the_project_with_a_note() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_on_an_earlier_save(&opening);
        let (project, note) = opening.clone().open_restoring().expect("opened");
        assert!(!project.has_unsaved_edits(), "not restored by itself");
        assert_eq!(project.path(), Some(opening.path.as_path()));
        let Some(OpenNote::EarlierVersion { candidate }) = note else {
            panic!("the note asks: {note:?}");
        };
        assert_eq!(candidate.path, file);
        project.close(false).expect("closed");
        assert_eq!(left(&opening), vec![file], "kept for the user's answer");
    }

    #[test]
    fn a_restorable_file_goes_before_one_on_an_earlier_save() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        left_on_an_earlier_save(&opening);
        let latest = left_work(&opening);
        let (project, note) = opening.open_restoring().expect("opened");
        assert!(matches!(note, Some(OpenNote::Restored { path, .. }) if path == latest));
        close(project);
    }

    #[test]
    fn one_that_cannot_be_restored_is_left_alone_with_no_note() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        let real = listed(&opening).pop().expect("listed");
        let statuses = [
            CandidateStatus::ProjectMissing,
            CandidateStatus::SavePointMissing,
            CandidateStatus::ProjectInUse,
            CandidateStatus::Damaged,
            CandidateStatus::NeverSaved,
        ];
        for status in statuses {
            let candidate = RecoveryCandidate {
                status,
                ..real.clone()
            };
            let found = Ok(vec![candidate]);
            let (project, note) = opening.clone().open_with(found).expect("opened");
            assert_eq!(note, None, "{status:?}");
            assert!(!project.has_unsaved_edits(), "{status:?}");
            project.close(false).expect("closed");
            assert_eq!(left(&opening), vec![file.clone()], "{status:?}");
        }
    }

    #[test]
    fn a_project_replaced_at_its_path_keeps_the_file_and_shows_nothing() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        std::fs::remove_file(&opening.path).expect("removed");
        saved(&dir, "first-sculpt.bcrf");
        let (project, note) = opening.clone().open_restoring().expect("opened");
        assert_eq!(note, None);
        assert!(!project.has_unsaved_edits());
        project.close(false).expect("closed");
        assert_eq!(left(&opening), vec![file]);
    }

    #[test]
    fn a_failing_look_for_recovery_files_opens_the_project_all_the_same() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        let failed = Err(ProjectError::Io(std::io::Error::other("no listing")));
        let (project, note) = opening.clone().open_with(failed).expect("opened");
        assert_eq!(note, None);
        project.close(false).expect("closed");
        assert_eq!(left(&opening), vec![file], "kept for the next open");
    }

    #[test]
    fn a_failing_restore_opens_the_project_all_the_same_and_keeps_the_file() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        let real = listed(&opening).pop().expect("listed");
        // A listing gone stale: the save point it names isn't the file's.
        let stale = RecoveryCandidate { base: None, ..real };
        let (project, note) = opening.clone().open_with(Ok(vec![stale])).expect("opened");
        assert_eq!(note, None, "nothing was restored");
        assert!(!project.has_unsaved_edits(), "opened as ever");
        assert_ne!(project.recovery_path(), Some(file.as_path()));
        project.close(false).expect("closed");
        assert_eq!(left(&opening), vec![file], "kept");
    }

    /// The options of `args` on `project`.
    fn options(project: &ProjectToOpen, args: &[&str]) -> Options {
        let args = args.iter().map(OsString::from);
        let path = OsString::from(project.path.as_os_str());
        Options::parse(std::iter::once(path).chain(args)).expect("parsed")
    }

    #[test]
    fn scripted_and_offscreen_runs_never_restore_unless_asked() {
        let dir = TestDir::new();
        let opening = saved(&dir, "first-sculpt.bcrf");
        let file = left_work(&opening);
        let measured: [&[&str]; 4] = [
            &["--frames", "3"],
            &["--script"],
            &["--offscreen", "--frames", "3"],
            &["--offscreen", "--script"],
        ];
        for args in measured {
            let run = options(&opening, args);
            let (project, note) = opening.clone().open_as(&run).expect("opened");
            assert_eq!(note, None, "{args:?}");
            assert!(!project.has_unsaved_edits(), "{args:?}");
            project.close(false).expect("closed");
            assert_eq!(left(&opening), vec![file.clone()], "{args:?}");
        }
        let asked: [&[&str]; 3] = [
            &[],
            &["--offscreen", "--frames", "3", "--restore"],
            &["--offscreen", "--script", "--restore"],
        ];
        for args in asked {
            let run = options(&opening, args);
            let (project, note) = opening.clone().open_as(&run).expect("opened");
            assert!(matches!(note, Some(OpenNote::Restored { .. })), "{args:?}");
            close(project);
        }
    }

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
            let outcome = settled(
                run,
                ran,
                (Some(()), End::Keep),
                |()| {
                    closed = true;
                    Ok(())
                },
                |()| panic!("nobody said so"),
            );
            assert!(closed, "closed all the same");
            assert_eq!(outcome.is_err(), failing);
        }
        let keep = (Some(()), End::Keep);
        let no_room = |()| Err(anyhow!("no room"));
        let both = settled(failed(), Ok(()), keep, no_room, |()| Ok(()));
        let said = format!("{:#}", both.expect_err("failed"));
        assert!(
            said.contains("the device was lost") && said.contains("no room"),
            "{said}"
        );
        let closing = settled(Ok(summary()), Ok(()), keep, no_room, |()| Ok(()));
        assert!(closing.is_err(), "a close that failed fails the run");
    }

    #[test]
    fn settled_discards_when_the_run_says_so() {
        let mut discarded = false;
        let outcome = settled(
            Ok(summary()),
            Ok(()),
            (Some(()), End::Discard),
            |()| panic!("Don't save lands nothing"),
            |()| {
                discarded = true;
                Ok(())
            },
        );
        assert!(discarded, "it went the other way");
        outcome.expect("it went well");

        let failed = settled(
            Ok(summary()),
            Ok(()),
            (Some(()), End::Discard),
            |()| Ok(()),
            |()| Err(anyhow!("no room")),
        );
        assert!(failed.is_err(), "a discard that failed fails the run");
    }
}
