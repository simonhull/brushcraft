//! `--load`: work on the main thread like M1a4's frame loop while the pen is used, so the
//! dispatch delay of tablet events is measured under it.
//!
//! # What the load is
//!
//! A **reflinked copy** of a benchmark project (by default the bench's
//! `$XDG_CACHE_HOME/brushcraft/bench/sphere.bcrf`) is opened, keeping a display layer under the
//! level brushed (9), with a 2 GiB tile budget and the store's default workers, as the app opens
//! one. The copy and its recovery file live in a work directory beside the project
//! (`pen-probe-<pid>-<n>`), removed when the run ends, and by the next run if this one was
//! killed. The source is only ever read, to reflink it; a file system that can't reflink is
//! refused rather than copied in full.
//!
//! If the copy has no display layer (the bench's sphere predates it), the store's rebuild runs
//! to the end before the window opens: about a minute on the sphere. `--guided` does that once,
//! into a prepared copy it saves, so its two runs open with the layer.
//!
//! Then four strokes are planned along the bench's brush path (`brushcraft-bench brush`): each
//! starts at a seeded random direction, its radius covering about 50,000 vertices there, and its
//! 200 dabs step a tenth of a radius along a great circle, each centre put on the surface by a
//! raycast; their tiles are evaluated first (warm, as tiles the view has drawn are). Strength
//! 0.0001, as the bench's.
//!
//! Each frame, paced to the window's refresh interval, runs on the main thread, in the frame
//! loop's order (`crates/app/src/frame.rs`):
//!
//! 1. **brush**: the draw steps due at a pen's rate (173 Hz, the Kamvas's measured rate) since the
//!    last frame, at most 16 (the rest dropped), each a [`sculpt::draw_step`] on the next centre.
//!    After a stroke's 200th dab the stroke ends ([`Project::end_unrecorded_stroke`], which hands
//!    its edits to the recovery writer) and the next begins: the strokes cycle with no gap.
//! 2. **feed**: the stale-tile feed is drained (the display layer's epochs).
//! 3. **upkeep**: [`Project::tick`] and [`Project::poll`]: the recovery file's flushes, at least
//!    once a second during a stroke.
//! 4. **jobs**: [`store::TileStore::run_jobs`] until three quarters of an interval after the frame
//!    began (at least one slice): the display refresh the dabs leave, whose records are written
//!    through to the working file once they pass their share of the budget.
//!
//! So the store's own blocking file writes (flushes and write-through `pwrite`s on the owner
//! thread) land where the app's would. Nothing is drawn on the GPU: the app hands rendering to a
//! thread of its own. The load runs from the window's opening to its closing, pen or no pen, the
//! same in a winit run and in one of ours: like for like. Each frame's phases are recorded
//! ([`FrameRecord`]).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use geom::{Level, TileKey, Vec3};
use sculpt::{DrawParams, Stroke, draw_step};
use store::{
    Evaluation, JobsReport, Project, ProjectConfig, Protection, Ray, SpatialIndex, StaleCursor,
    StoreConfig, TileStore,
};

use crate::frames::FrameRecord;

/// The site the load's op ids carry: any but the reserved one.
const SITE: u16 = 1;
/// The tile store's budget: the bench's brush default.
const BUDGET_BYTES: usize = 2048 << 20;
/// The level brushed, indexed and kept a display layer under: the bench's.
pub(crate) const DEFAULT_LEVEL: u8 = 9;
/// The rate draw steps come at: the Kamvas's measured sample rate.
pub(crate) const PEN_RATE_HZ: f64 = 173.0;
/// The most draw steps one frame runs; the rest due are dropped.
const MAX_DABS_PER_FRAME: u64 = 16;
/// Dabs per stroke: the bench's `--steps` default.
const STROKE_DABS: usize = 200;
/// Strokes planned, then cycled.
const STROKES: usize = 4;
/// Vertices a brush covers where its stroke starts: the bench's `--footprint` default.
const FOOTPRINT: usize = 50_000;
/// How far the brush moves each dab, in radii: the bench's.
const STEP_RADII: f32 = 0.1;
/// How far a dab pushes: the bench's `--strength` default.
const STRENGTH: f32 = 0.0001;
/// The seed of the strokes' directions.
const SEED: u64 = 1;
/// The frame interval when the display doesn't say: 60 Hz.
pub(crate) const DEFAULT_INTERVAL: Duration = Duration::from_micros(16_667);

/// The bench's default project: `$XDG_CACHE_HOME/brushcraft/bench/sphere.bcrf`, else under
/// `$HOME/.cache`.
///
/// # Errors
///
/// If neither variable names an absolute path.
pub(crate) fn default_project() -> Result<PathBuf> {
    Ok(cache_root()?.join("bench").join("sphere.bcrf"))
}

/// BrushCraft's cache directory: `$XDG_CACHE_HOME/brushcraft`, else `$HOME/.cache/brushcraft`.
///
/// # Errors
///
/// If neither variable names an absolute path.
pub(crate) fn cache_root() -> Result<PathBuf> {
    let absolute = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
    };
    let cache = absolute("XDG_CACHE_HOME")
        .or_else(|| absolute("HOME").map(|home| home.join(".cache")))
        .ok_or_else(|| anyhow!("neither XDG_CACHE_HOME nor HOME is an absolute path"))?;
    Ok(cache.join("brushcraft"))
}

/// Fails if `path` lies inside the repository the probe was built from.
///
/// # Errors
///
/// If it does, or it can't be resolved.
pub(crate) fn ensure_outside_repo(path: &Path) -> Result<()> {
    if let Some(root) =
        checkout::containing(path).with_context(|| format!("resolving {}", path.display()))?
    {
        bail!(
            "{} is inside the repository ({}); the probe's files go outside it",
            path.display(),
            root.display()
        );
    }
    Ok(())
}

/// The prefix of a work directory's name.
const WORK_PREFIX: &str = "pen-probe-";

/// The file [`WorkDir::beside`] puts in each work directory it makes: the sweep removes only
/// directories that hold one.
const WORK_MARKER: &str = ".pen-probe-work";

/// A directory beside a project for one run's copy of it, removed with everything in it when
/// dropped: `pen-probe-<pid>-<n>`.
#[derive(Debug)]
pub(crate) struct WorkDir {
    /// The directory.
    dir: PathBuf,
}

impl WorkDir {
    /// A new work directory beside `project`, after removing those that runs no longer alive
    /// left there.
    ///
    /// The directory beside `project` is checked to be outside the repository before anything in
    /// it is touched. Each name is taken with an exclusive `mkdir`, so two runs never share one,
    /// and the new directory gets a [`WORK_MARKER`] file, which the sweep requires before it
    /// removes anything.
    ///
    /// # Errors
    ///
    /// If `project`'s directory is inside the repository, or the work directory or its marker
    /// can't be made.
    pub(crate) fn beside(project: &Path) -> Result<Self> {
        let parent = project
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure_outside_repo(parent)?;
        sweep(parent);
        let pid = std::process::id();
        for n in 0..u32::MAX {
            let dir = parent.join(format!("{WORK_PREFIX}{pid}-{n}"));
            #[expect(
                clippy::create_dir,
                reason = "taking a name must fail if it exists, which `create_dir_all` doesn't"
            )]
            let made = std::fs::create_dir(&dir);
            match made {
                Ok(()) => {
                    // Owned from here: should the marker fail, dropping removes the directory.
                    let work = Self { dir };
                    let marker = work.dir.join(WORK_MARKER);
                    File::create_new(&marker)
                        .with_context(|| format!("creating {}", marker.display()))?;
                    return Ok(work);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => {
                    return Err(e).with_context(|| format!("creating {}", dir.display()));
                }
            }
        }
        bail!("no free work directory name in {}", parent.display())
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        // Best effort: what is left behind, the next run's sweep removes.
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

/// A decimal `u32` written with digits only: no sign, no spaces.
fn digits(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// The pid in a work directory's name, `pen-probe-<pid>-<n>` exactly; `None` for any other name.
fn work_dir_pid(name: &str) -> Option<u32> {
    let (pid, n) = name.strip_prefix(WORK_PREFIX)?.split_once('-')?;
    digits(n)?;
    digits(pid)
}

/// Whether `entry` is a work directory a run no longer alive left: named as one, a directory
/// itself (not a link to one), holding the [`WORK_MARKER`] file, and its pid not running (Linux:
/// no `/proc/<pid>`).
fn is_stale_work_dir(entry: &std::fs::DirEntry) -> bool {
    let Some(pid) = entry.file_name().to_str().and_then(work_dir_pid) else {
        return false;
    };
    // `DirEntry::file_type` doesn't follow a symbolic link.
    let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
    let marked =
        std::fs::symlink_metadata(entry.path().join(WORK_MARKER)).is_ok_and(|m| m.is_file());
    is_dir
        && marked
        && pid != std::process::id()
        && !Path::new("/proc").join(pid.to_string()).exists()
}

/// Removes the work directories in `dir` that runs no longer alive left (see
/// [`is_stale_work_dir`]); does nothing on systems other than Linux.
fn sweep(dir: &Path) {
    if !cfg!(target_os = "linux") {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let stale: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(is_stale_work_dir)
        .map(|entry| entry.path())
        .collect();
    for path in stale {
        drop(std::fs::remove_dir_all(path));
    }
}

/// Reflinks `source` to `target` (a new file): no data copied.
///
/// # Errors
///
/// If `source` can't be read, `target` made, or the file system can't reflink between them (the
/// target is then removed).
fn reflink(source: &Path, target: &Path) -> Result<()> {
    let from = File::open(source).with_context(|| format!("opening {}", source.display()))?;
    let to = File::create_new(target).with_context(|| format!("creating {}", target.display()))?;
    if let Err(e) = rustix::fs::ioctl_ficlone(&to, &from) {
        drop(to);
        drop(std::fs::remove_file(target));
        bail!(
            "{} can't be reflinked to {} ({e}); the load works on reflinked copies only, so \
             the project must be on a file system that can (btrfs, xfs)",
            source.display(),
            target.display()
        );
    }
    Ok(())
}

/// The project configuration the load opens with: recovery files in `dir`.
fn config(level: Level, dir: &Path) -> ProjectConfig {
    ProjectConfig {
        store: StoreConfig::new(BUDGET_BYTES, SITE).with_display(level),
        recovery_dir: dir.join("recovery"),
    }
}

/// Reflinks `source` into `dir` as `name`, and opens it.
///
/// # Errors
///
/// If the reflink or the open fails.
fn open_copy(source: &Path, dir: &Path, name: &str, level: Level) -> Result<(PathBuf, Project)> {
    let copy = dir.join(name);
    reflink(source, &copy)?;
    let project =
        Project::open(&copy, config(level, dir)).with_context(|| format!("opening {name}"))?;
    Ok((copy, project))
}

/// Runs the store's jobs to the end: the display layer's rebuild, if the file had none. Prints
/// progress every few seconds.
///
/// # Errors
///
/// If a job fails.
fn rebuild(store: &mut TileStore) -> Result<()> {
    if !store.has_jobs() {
        return Ok(());
    }
    let start = Instant::now();
    let mut said = start;
    crate::say(
        "load: the project has no display layer; building it first (about a minute on the \
         benchmark sphere)",
    );
    while store.has_jobs() {
        let report = store
            .run_jobs(Instant::now() + Duration::from_millis(100))
            .context("rebuilding the display layer")?;
        if said.elapsed() >= Duration::from_secs(5) {
            said = Instant::now();
            crate::say(format_args!("load: {} display tiles left", report.left));
        }
    }
    crate::say(format_args!(
        "load: display layer built in {:.0} s",
        start.elapsed().as_secs_f64()
    ));
    Ok(())
}

/// Makes a copy of `source` in `dir` with its display layer built and saved; returns its path.
///
/// The runs that open it skip the rebuild. The copy is a reflink, then the save appends the
/// layer to it (about half the project's size again, on the benchmark sphere).
///
/// # Errors
///
/// If the copy can't be made, opened, rebuilt, saved or closed.
pub(crate) fn prepare(source: &Path, dir: &Path, level: Level) -> Result<PathBuf> {
    let (copy, mut project) = open_copy(source, dir, "prepared.bcrf", level)?;
    if project.store().has_jobs() {
        rebuild(project.store())?;
        let start = Instant::now();
        let saved = project.save().context("saving the prepared copy")?;
        // A scratch copy loses nothing if crash protection pauses, but a paused writer means a
        // recovery directory that can't take files, which the runs' copies will need: say so.
        if let Protection::Paused(why) = saved.protection {
            crate::say(format_args!(
                "load: crash protection paused after saving the prepared copy ({why}); the runs \
                 put their recovery files beside their copies too"
            ));
        }
        crate::say(format_args!(
            "load: saved the prepared copy in {:.0} s",
            start.elapsed().as_secs_f64()
        ));
    }
    project.close(false).context("closing the prepared copy")?;
    Ok(copy)
}

/// A stroke ready to run: its radius and the centres of its dabs.
#[derive(Debug, Clone)]
struct Plan {
    /// The brush radius.
    radius: f32,
    /// The dabs' centres, on the surface.
    centres: Vec<Vec3>,
}

/// The load: the project copy, its strokes, and the frames run so far. See the module notes.
#[derive(Debug)]
pub(crate) struct Load {
    /// The open copy. Closed in [`Load::finish`]; should that not run, the work directory
    /// still goes.
    project: Project,
    /// The spatial index over the level brushed.
    index: SpatialIndex,
    /// The session's stroke state.
    stroke: Stroke,
    /// The load's cursor in the stale-tile feed.
    cursor: StaleCursor,
    /// The planned strokes, cycled.
    plans: Vec<Plan>,
    /// The stroke being drawn.
    plan: usize,
    /// The next dab of it.
    dab: usize,
    /// When draw steps started coming due.
    began: Instant,
    /// Draw steps counted as due so far, run or dropped.
    dabs_due: u64,
    /// The frame interval.
    interval: Duration,
    /// When the next frame may start.
    next_frame: Instant,
    /// The frames run so far.
    frames: Vec<FrameRecord>,
    /// The work directory holding the copy: dropped (removed) after the project.
    work: WorkDir,
}

impl Load {
    /// Opens a reflinked copy of `source` beside it, builds its display layer if it has none,
    /// loads the index and plans and warms the strokes; the first stroke begins. Prints what it
    /// does.
    ///
    /// # Errors
    ///
    /// If the copy can't be made or opened, the layer built, the index loaded, a stroke planned
    /// or begun.
    pub(crate) fn open(source: &Path, level: Level) -> Result<Self> {
        let start = Instant::now();
        let work = WorkDir::beside(source)?;
        let (_, project) = open_copy(source, work.path(), "load.bcrf", level)?;
        let load = Self::start(project, work, level)?;
        crate::say(format_args!(
            "load: {} ready in {:.1} s: {STROKES} strokes of {STROKE_DABS} dabs, radius {:.5}, \
             at {PEN_RATE_HZ} Hz",
            source.display(),
            start.elapsed().as_secs_f64(),
            load.plans.first().map_or(0.0, |p| p.radius),
        ));
        Ok(load)
    }

    /// The load on `project`, open in `work`: builds its display layer if it has none, loads
    /// the index, plans and warms the strokes, and begins the first.
    ///
    /// # Errors
    ///
    /// As [`Load::open`], but for the copy.
    fn start(mut project: Project, work: WorkDir, level: Level) -> Result<Self> {
        rebuild(project.store())?;
        let (index, _) = project
            .load_index(level, |_, _| {})
            .context("loading the spatial index")?;
        let mut rng = Rng(SEED);
        let plans = (0..STROKES)
            .map(|_| plan(&mut project, &index, rng.direction()))
            .collect::<Result<Vec<_>>>()?;
        let stroke = Stroke::new(project.store());
        let cursor = project.store().subscribe_stale();
        project.begin_stroke().context("beginning a stroke")?;
        let now = Instant::now();
        Ok(Self {
            project,
            index,
            stroke,
            cursor,
            plans,
            plan: 0,
            dab: 0,
            began: now,
            dabs_due: 0,
            interval: DEFAULT_INTERVAL,
            next_frame: now,
            frames: Vec::new(),
            work,
        })
    }

    /// Paces frames to `interval` from now on.
    pub(crate) const fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// When the next frame may start.
    pub(crate) const fn next_frame(&self) -> Instant {
        self.next_frame
    }

    /// Restarts the draw steps' clock at `now`: none is due for the time before (opening the
    /// window, say).
    pub(crate) fn restart_clock(&mut self, now: Instant) {
        self.began = now;
        self.dabs_due = 0;
        self.next_frame = now;
    }

    /// Runs one frame begun at `now` (see the module notes) and records it; `since` is when the
    /// probe started, which the record's start counts from.
    ///
    /// # Errors
    ///
    /// If a draw step, a stroke's end or beginning, the upkeep or the jobs fail.
    pub(crate) fn frame(&mut self, now: Instant, since: Instant) -> Result<()> {
        let deadline = now + self.interval * 3 / 4;
        let mut record = FrameRecord {
            start: now.saturating_duration_since(since),
            interval: self.interval,
            ..FrameRecord::default()
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative count of dabs, far below 2^53"
        )]
        let due = (now.saturating_duration_since(self.began).as_secs_f64() * PEN_RATE_HZ) as u64;
        let owed = due.saturating_sub(self.dabs_due);
        let run = owed.min(MAX_DABS_PER_FRAME);
        self.dabs_due = due.max(self.dabs_due);
        record.dabs = u32::try_from(run).unwrap_or(u32::MAX);
        record.skipped = u32::try_from(owed - run).unwrap_or(u32::MAX);
        for _ in 0..run {
            self.dab(&mut record)?;
        }

        let feed = Instant::now();
        drop(self.project.store().drain_stale(&self.cursor));
        record.feed = feed.elapsed();

        let upkeep = Instant::now();
        self.project.tick(now).context("ticking the project")?;
        self.project.poll().context("polling the project")?;
        record.upkeep = upkeep.elapsed();

        let jobs_began = Instant::now();
        let jobs = self
            .project
            .store()
            .run_jobs(deadline.max(jobs_began + JobsReport::SLICE_TARGET))
            .context("running the store's jobs")?;
        record.jobs = jobs_began.elapsed();
        record.slices = jobs.slices;
        record.longest_slice = jobs.longest_slice;

        record.busy = now.elapsed();
        self.frames.push(record);
        self.next_frame = now + self.interval;
        Ok(())
    }

    /// One draw step on the next centre, timed into `record`; after a stroke's last, the stroke
    /// ends and the next begins.
    ///
    /// # Errors
    ///
    /// If the step fails, or the stroke can't end or begin.
    fn dab(&mut self, record: &mut FrameRecord) -> Result<()> {
        let lap = Instant::now();
        let plan = self
            .plans
            .get(self.plan)
            .context("the load has no strokes")?;
        let center = plan
            .centres
            .get(self.dab)
            .copied()
            .context("a stroke ran past its dabs")?;
        let params = DrawParams::new(center, plan.radius, STRENGTH)?;
        draw_step(
            self.project.store(),
            &mut self.index,
            &mut self.stroke,
            params,
        )
        .context("a draw step")?;
        record.brush += lap.elapsed();
        self.dab += 1;
        if self.dab >= plan.centres.len() {
            let ending = Instant::now();
            self.project
                .end_unrecorded_stroke()
                .context("ending a stroke")?;
            self.plan = (self.plan + 1) % self.plans.len();
            self.dab = 0;
            self.project.begin_stroke().context("beginning a stroke")?;
            record.stroke_end += ending.elapsed();
        }
        Ok(())
    }

    /// Ends the stroke, closes the copy and removes the work directory; returns the frames run,
    /// and how the close went: an error if the stroke couldn't end or the project close. The
    /// work directory goes either way.
    pub(crate) fn finish(self) -> (Vec<FrameRecord>, Result<()>) {
        let Self {
            mut project,
            cursor,
            frames,
            work,
            ..
        } = self;
        let ended = project
            .end_unrecorded_stroke()
            .context("ending the last stroke");
        project.store().unsubscribe_stale(cursor);
        let closed = project.close(false).context("closing the load's project");
        drop(work);
        (frames, ended.and(closed))
    }
}

/// Plans a warm stroke of [`STROKE_DABS`] dabs from `direction`: the bench's `Plan::new`.
///
/// # Errors
///
/// If a tile can't be evaluated, or a raycast finds no surface.
fn plan(project: &mut Project, index: &SpatialIndex, direction: Vec3) -> Result<Plan> {
    let store = project.store();
    let start = on_surface(index, store, direction)?;
    let radius = calibrated_radius(index, store, start, FOOTPRINT)?;
    let centres = path(direction, STEP_RADII * radius, STROKE_DABS)
        .map(|d| on_surface(index, store, d))
        .collect::<Result<Vec<_>>>()?;
    let mut reached: Vec<TileKey> = centres
        .iter()
        .flat_map(|&c| index.query_sphere(c, radius).collect::<Vec<_>>())
        .collect();
    reached.sort_unstable();
    reached.dedup();
    store
        .evaluate_many(&reached, Evaluation::Normals)
        .context("warming a stroke's tiles")?;
    Ok(Plan { radius, centres })
}

/// Where the surface lies in `direction` from the origin: the bench's `surface::on_surface`.
///
/// # Errors
///
/// If a tile can't be evaluated, or the ray leaves through no tile.
fn on_surface(index: &SpatialIndex, store: &mut TileStore, direction: Vec3) -> Result<Vec3> {
    let ray = Ray::new(Vec3::ZERO, direction).ok_or_else(|| anyhow!("no direction"))?;
    let hit = index
        .raycast(store, &ray)
        .context("raycasting")?
        .ok_or_else(|| anyhow!("the ray along {direction:?} leaves through no tile"))?;
    Ok(ray.at(hit.t))
}

/// The radius covering about `footprint` vertices around `center`, by bisection: the bench's
/// `surface::calibrated_radius`.
///
/// # Errors
///
/// If a tile can't be evaluated.
fn calibrated_radius(
    index: &SpatialIndex,
    store: &mut TileStore,
    center: Vec3,
    footprint: usize,
) -> Result<f32> {
    let (mut low, mut high) = (0.0_f32, 2.0_f32);
    for _ in 0..30 {
        let mid = f32::midpoint(low, high);
        let keys: Vec<TileKey> = index.query_sphere(center, mid).collect();
        let mut count = 0;
        for key in keys {
            count += store
                .evaluate(key)?
                .iter()
                .filter(|&&p| (p - center).length() < mid)
                .count();
        }
        if count < footprint {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

/// The unit directions of a stroke's `steps` dabs along the great circle through `start`
/// heading east, `step` radians apart: the bench's `surface::path`.
fn path(start: Vec3, step: f32, steps: usize) -> impl Iterator<Item = Vec3> {
    let z = Vec3::new(0.0, 0.0, 1.0);
    let from = start.normalized().unwrap_or(z);
    let axis = (z - from * from.dot(z))
        .normalized()
        .unwrap_or_else(|| Vec3::new(-1.0, 0.0, 0.0));
    let heading = axis.cross(from);
    (0_u32..).take(steps).map(move |k| {
        #[expect(clippy::cast_precision_loss, reason = "step counts are far below 2^24")]
        let angle = step * k as f32;
        let (sin, cos) = angle.sin_cos();
        from * cos + heading * sin
    })
}

/// A seeded generator (splitmix64), as the bench's `noise::Rng`: the same seed, the same strokes.
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    /// Uniform in `[-1, 1)`, with 24 bits of resolution.
    fn signed_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        #[expect(clippy::cast_precision_loss, reason = "24 bits are exact in an f32")]
        let bits = (z >> 40) as f32;
        bits / 8_388_608.0 - 1.0
    }

    /// A direction uniform over the unit sphere, by rejection from the cube.
    fn direction(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(self.signed_unit(), self.signed_unit(), self.signed_unit());
            let length = p.length();
            if (1e-3..=1.0).contains(&length)
                && let Some(d) = p.normalized()
            {
                return d;
            }
        }
    }
}

/// `--load-check`: runs the load for `seconds` with no window, paced to 60 Hz, and returns its
/// frames: what the load costs, without a pen.
///
/// # Errors
///
/// As [`Load::open`], [`Load::frame`] and [`Load::finish`].
pub(crate) fn check(source: &Path, level: Level, seconds: u64) -> Result<Vec<FrameRecord>> {
    let mut load = Load::open(source, level)?;
    let since = Instant::now();
    load.restart_clock(since);
    let end = since + Duration::from_secs(seconds);
    let ran = (|| -> Result<()> {
        while Instant::now() < end {
            let now = Instant::now();
            if now < load.next_frame() {
                std::thread::sleep(load.next_frame() - now);
                continue;
            }
            load.frame(now, since)?;
        }
        Ok(())
    })();
    let (frames, closed) = load.finish();
    ran.and(closed).map(|()| frames)
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use geom::cube_sphere;

    use super::*;

    /// A uniquely named directory, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        /// One under the system's temporary directory.
        fn new(name: &str) -> Self {
            Self::under(&std::env::temp_dir(), name)
        }

        /// One under `base`.
        fn under(base: &Path, name: &str) -> Self {
            let dir = base.join(format!(
                "pen-probe-test-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            drop(std::fs::remove_dir_all(&dir));
            std::fs::create_dir_all(&dir).expect("a scratch directory");
            Self(dir)
        }
    }

    /// Makes `dir` a work directory as [`WorkDir::beside`] would leave it: marked.
    fn marked(dir: &Path) {
        std::fs::create_dir_all(dir).expect("made");
        File::create(dir.join(WORK_MARKER)).expect("marked");
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    #[test]
    fn a_work_dir_goes_when_dropped_and_only_stale_ones_are_swept() {
        let scratch = Scratch::new("work");
        let project = scratch.0.join("p.bcrf");
        // No process has pid u32::MAX - 1; pid 1 always runs.
        let dead_pid = u32::MAX - 1;
        let dead = scratch.0.join(format!("{WORK_PREFIX}{dead_pid}-0"));
        marked(&dead);
        let live = scratch.0.join(format!("{WORK_PREFIX}1-0"));
        marked(&live);
        let suffixed = scratch.0.join(format!("{WORK_PREFIX}{dead_pid}-0-extra"));
        marked(&suffixed);
        let unmarked = scratch.0.join(format!("{WORK_PREFIX}{dead_pid}-1"));
        std::fs::create_dir_all(&unmarked).expect("made");
        let file = scratch.0.join(format!("{WORK_PREFIX}{dead_pid}-2"));
        std::fs::write(&file, b"not a directory").expect("written");
        let other = scratch.0.join("keep-me");
        marked(&other);
        let link = scratch.0.join(format!("{WORK_PREFIX}{dead_pid}-3"));
        std::os::unix::fs::symlink(&other, &link).expect("linked");

        let work = WorkDir::beside(&project).expect("a work directory");
        assert!(work.path().join(WORK_MARKER).is_file(), "marked");
        assert!(!dead.exists(), "the dead run's directory is swept");
        for kept in [&live, &suffixed, &unmarked, &file, &other] {
            assert!(kept.exists(), "{} is kept", kept.display());
        }
        assert!(link.is_symlink(), "a link to a marked directory is kept");
        assert!(other.join(WORK_MARKER).is_file(), "and so is its target");
        let path = work.path().to_path_buf();
        drop(work);
        assert!(!path.exists(), "removed when dropped");
    }

    #[test]
    fn a_work_dir_name_is_taken_exclusively() {
        let scratch = Scratch::new("exclusive");
        let first = WorkDir::beside(&scratch.0.join("p.bcrf")).expect("a work directory");
        let second = WorkDir::beside(&scratch.0.join("p.bcrf")).expect("a work directory");
        assert_ne!(first.path(), second.path());
        assert!(second.path().join(WORK_MARKER).is_file());
    }

    #[test]
    fn work_dir_names_match_exactly() {
        assert_eq!(work_dir_pid("pen-probe-12-3"), Some(12));
        for name in [
            "pen-probe-12-3-extra",
            "pen-probe-12-",
            "pen-probe--3",
            "pen-probe-+12-3",
            "pen-probe-12-+3",
            "pen-probe-12",
            "pen-probe-test-load-12-3",
            "pen-probe-99999999999-3",
        ] {
            assert_eq!(work_dir_pid(name), None, "{name}");
        }
    }

    #[test]
    fn a_project_inside_the_repository_is_refused_before_anything_is_swept() {
        let inside = Scratch::under(Path::new(env!("CARGO_MANIFEST_DIR")), "inside");
        let dead = inside.0.join(format!("{WORK_PREFIX}{}-0", u32::MAX - 1));
        marked(&dead);
        let refused = WorkDir::beside(&inside.0.join("p.bcrf")).expect_err("refused");
        assert!(
            refused.to_string().contains("inside the repository"),
            "{refused:#}"
        );
        assert!(
            dead.join(WORK_MARKER).is_file(),
            "nothing beside it is touched"
        );
        let entries = std::fs::read_dir(&inside.0).expect("listed").count();
        assert_eq!(entries, 1, "and no work directory is made");
    }

    #[test]
    fn the_path_heads_east_along_a_great_circle() {
        let points: Vec<Vec3> =
            path(Vec3::new(2.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2, 3).collect();
        assert_eq!(points.len(), 3);
        assert!((points.first().copied().unwrap() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((points.get(1).copied().unwrap() - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn directions_are_unit_and_seeded() {
        let a: Vec<Vec3> = {
            let mut rng = Rng(7);
            (0..8).map(|_| rng.direction()).collect()
        };
        let mut rng = Rng(7);
        for d in &a {
            assert!((d.length() - 1.0).abs() < 1e-5, "{d:?}");
            assert_eq!(*d, rng.direction(), "the same seed, the same directions");
        }
    }

    /// The load on a small project: its frames run brush steps at the pen's rate (dropping
    /// what falls too far behind), end strokes, flush and run jobs, and the work directory goes
    /// at the end.
    #[test]
    fn the_load_runs_frames_and_cleans_up() {
        let scratch = Scratch::new("load");
        let level = Level::new(4).expect("a level");
        let work = WorkDir::beside(&scratch.0.join("source.bcrf")).expect("a work directory");
        let dir = work.path().to_path_buf();
        let project = Project::new(cube_sphere(1).expect("a cube sphere"), config(level, &dir))
            .expect("a new project");
        let mut load = Load::start(project, work, level).expect("the load starts");
        let since = Instant::now();
        load.restart_clock(since);
        // Two strokes' worth of 100 ms frames: each owes 17 dabs, runs 16 and drops one.
        let frames_needed = STROKE_DABS * 2 / 16 + 4;
        for _ in 0..frames_needed {
            std::thread::sleep(Duration::from_millis(100));
            load.frame(Instant::now(), since).expect("a frame");
        }
        let (frames, closed) = load.finish();
        closed.expect("closed");
        assert!(!dir.exists(), "the work directory is gone");
        assert_eq!(frames.len(), frames_needed);
        let dabs: u32 = frames.iter().map(|f| f.dabs).sum();
        assert!(dabs >= u32::try_from(STROKE_DABS).unwrap(), "{dabs} dabs");
        assert!(frames.iter().all(|f| f.dabs <= 16), "at most 16 a frame");
        assert!(
            frames.iter().any(|f| f.skipped > 0),
            "100 ms frames fall behind 173 Hz"
        );
        assert!(
            frames.iter().any(|f| f.stroke_end > Duration::ZERO),
            "a stroke ended"
        );
        assert!(frames.iter().all(|f| f.busy >= f.brush + f.jobs));
    }

    /// A prepared copy is a reflink of the source with its display layer saved, in the work
    /// directory; the source is untouched.
    ///
    /// The source is a tiny generated project in a scratch directory under the cache root
    /// ([`cache_root`]), where the benchmark projects live and the load reflinks: unlike the
    /// temporary directory (often tmpfs), that file system has to reflink for `--load` to work.
    /// If it can't, the test fails saying so, except under CI (`CI` set), whose runners' disks
    /// can't reflink, where it says it didn't run.
    #[test]
    fn prepare_works_on_a_reflinked_copy() {
        let root = cache_root().expect("a cache directory");
        std::fs::create_dir_all(&root).expect("the cache directory");
        let scratch = Scratch::under(&root, "prepare");
        let source = scratch.0.join("source.bcrf");
        let level = Level::new(4).expect("a level");
        let mut project = Project::new(
            cube_sphere(1).expect("a cube sphere"),
            config(level, &scratch.0),
        )
        .expect("a new project");
        let saved = project.save_as(&source).expect("saved");
        assert!(
            matches!(saved.protection, Protection::Active),
            "crash protection runs after the save: {:?}",
            saved.protection
        );
        project.close(false).expect("closed");
        let before = std::fs::read(&source).expect("read");
        let work = WorkDir::beside(&source).expect("a work directory");
        match prepare(&source, work.path(), level) {
            Ok(prepared) => {
                assert!(prepared.starts_with(work.path()));
                assert_eq!(std::fs::read(&source).expect("read"), before, "untouched");
            }
            Err(e) if e.to_string().contains("can't be reflinked") => {
                assert!(
                    std::env::var_os("CI").is_some(),
                    "the cache directory can't reflink, so --load can't run here: {e}"
                );
                eprintln!("NOT RUN: the CI runner's cache directory can't reflink: {e}");
            }
            Err(e) => panic!("{e:#}"),
        }
    }
}
