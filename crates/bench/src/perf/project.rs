//! The projects the suite measures, and where they live: the tiny one made fresh for each run,
//! the medium one cached, the full one the benchmark's own, each measured through a copy.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result};

/// A project size the suite runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scale {
    /// `cube_sphere(2)` to level 6: proves the suite runs end to end in seconds (CI, tests).
    /// Its timings are never recorded.
    Tiny,
    /// `cube_sphere(5)` to level 9: the benchmark sphere's shape (the same levels and tiles,
    /// 150 base faces instead of 2,400), generated in seconds and cached. The suite's everyday
    /// measure.
    Medium,
    /// The benchmark sphere itself (`cube_sphere(20)` to level 9, 1.26B triangles), as
    /// `generate` left it: milestones and requests only.
    Full,
}

/// The generator's settings for a project the suite makes itself.
///
/// Every `generate` option that changes what it writes is here, all passed explicitly
/// ([`Shape::args`]) and all in the cache key ([`cache_key`]), so a changed default of `generate`'s
/// can't make a cached project stale unnoticed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Shape {
    /// The base mesh's `cube_sphere` splits.
    pub(crate) splits: u32,
    /// Levels sculpted, from 1; the last is the one measured.
    pub(crate) levels: u8,
    /// The noise's seed.
    pub(crate) seed: u64,
    /// The noise's amplitude at level 1.
    pub(crate) amplitude: f32,
    /// The store's budget while generating, in MiB: it decides when tiles are evicted and
    /// written, so the file's layout.
    pub(crate) budget_mib: usize,
}

impl Shape {
    /// The suite's shape with `splits` and `levels`: seed 1, amplitude 0.01, a 2 GiB budget, as
    /// the benchmark project was generated.
    const fn new(splits: u32, levels: u8) -> Self {
        Self {
            splits,
            levels,
            seed: 1,
            amplitude: 0.01,
            budget_mib: 2048,
        }
    }

    /// The arguments `generate` takes for this shape: every field, none left to a default.
    pub(crate) fn args(self) -> Vec<OsString> {
        [
            "--splits",
            &self.splits.to_string(),
            "--levels",
            &self.levels.to_string(),
            "--seed",
            &self.seed.to_string(),
            "--amplitude",
            &self.amplitude.to_string(),
            "--budget-mib",
            &self.budget_mib.to_string(),
        ]
        .map(OsString::from)
        .to_vec()
    }
}

/// How much each part does at a scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Workload {
    /// Repetitions of each part: each is one sample of its metrics.
    pub(crate) runs: usize,
    /// `brush`: draw steps in the stroke.
    pub(crate) brush_steps: usize,
    /// The brush footprint, in vertices.
    pub(crate) footprint: usize,
    /// `page-in`: samples of each region per run.
    pub(crate) page_in_samples: usize,
    /// `page-in`: the large region, in quads.
    pub(crate) region_quads: u64,
    /// `save`: the stroke lengths saved, one save each per run, for the save cost fit.
    pub(crate) save_steps: &'static [usize],
    /// `budget`: places the walk pans to.
    pub(crate) moves: usize,
    /// `budget`: draw steps at each place.
    pub(crate) steps_per_move: usize,
    /// `budget`: tiles pinned at each place.
    pub(crate) view_tiles: usize,
}

/// The stroke lengths the save cost model fits over, as in the benchmark's "Save cost model".
const SAVE_STEPS: [usize; 5] = [1, 10, 25, 50, 100];

impl Scale {
    /// The scale's word, the first of its metrics' names.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Medium => "medium",
            Self::Full => "full",
        }
    }

    /// The level measured.
    pub(crate) const fn level(self) -> u8 {
        match self {
            Self::Tiny => 6,
            Self::Medium | Self::Full => 9,
        }
    }

    /// The project's shape, for the scales the suite generates (the full project is
    /// `generate`'s default, made by hand).
    pub(crate) const fn shape(self) -> Option<Shape> {
        match self {
            Self::Tiny => Some(Shape::new(2, 6)),
            Self::Medium => Some(Shape::new(5, 9)),
            Self::Full => None,
        }
    }

    /// What each part does at this scale.
    ///
    /// Medium keeps the full project's per-step work (the same footprint and stroke) and scales
    /// what grows with the model: the region is 10M quads (2,442 level-9 tiles, a quarter of the
    /// project) instead of 42M, and the budget walk makes 10 moves instead of 20.
    pub(crate) const fn workload(self) -> Workload {
        match self {
            Self::Tiny => Workload {
                runs: 1,
                brush_steps: 5,
                footprint: 3000,
                page_in_samples: 2,
                region_quads: 20_000,
                save_steps: &[1, 3],
                moves: 3,
                steps_per_move: 2,
                view_tiles: 6,
            },
            Self::Medium => Workload {
                runs: 5,
                brush_steps: 200,
                footprint: 50_000,
                page_in_samples: 5,
                region_quads: 10_000_000,
                save_steps: &SAVE_STEPS,
                moves: 10,
                steps_per_move: 10,
                view_tiles: 2000,
            },
            Self::Full => Workload {
                runs: 5,
                brush_steps: 200,
                footprint: 50_000,
                page_in_samples: 5,
                region_quads: 42_000_000,
                save_steps: &SAVE_STEPS,
                moves: 20,
                steps_per_move: 10,
                view_tiles: 2000,
            },
        }
    }
}

/// The generator's code fingerprint, as `build.rs` computed it.
const GENERATOR_HASH: u64 = match u64::from_str_radix(env!("BRUSHCRAFT_GENERATOR_HASH"), 16) {
    Ok(hash) => hash,
    // Evaluated at compile time: a malformed hash from build.rs fails the build.
    Err(_) => panic!("build.rs writes the generator hash as 16 hex digits"),
};

/// `hash` continued over `bytes` with FNV-1a (64-bit), as `build.rs` hashes the sources.
fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The cache key of a project of `shape` made by this build, as 16 hex digits: the generator's
/// code fingerprint (see `build.rs`) continued over every field of the shape.
pub(crate) fn cache_key(shape: Shape) -> String {
    let hash = [
        &shape.splits.to_le_bytes()[..],
        &[shape.levels],
        &shape.seed.to_le_bytes(),
        &shape.amplitude.to_bits().to_le_bytes(),
        &u64::try_from(shape.budget_mib)
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    ]
    .into_iter()
    .fold(GENERATOR_HASH, fnv1a);
    format!("{hash:016x}")
}

/// The cached medium project's file in `dir` for key `key`.
pub(crate) fn cached_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("medium-{key}.bcrf"))
}

/// A project file a run measures through a copy, open from before the copy is made until after.
///
/// The copy reads this handle, never the path again, so the file being removed or replaced
/// under its name meanwhile (another run tidying the cache, or generating the same project)
/// can't take the project from under the run.
#[derive(Debug)]
pub(crate) struct Source {
    /// Where it was opened, for messages.
    path: PathBuf,
    /// The open file, locked shared if it is a cached project (see [`Source::cached`]).
    file: File,
}

impl Source {
    /// The project at `path`, open.
    ///
    /// # Errors
    ///
    /// If it can't be opened.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Ok(Self {
            path: path.to_owned(),
            file,
        })
    }

    /// The cached project at `path`, open and locked shared (`flock`) for as long as this lives,
    /// so [`remove_stale`] in another run leaves it alone; `None` if there is none.
    ///
    /// # Errors
    ///
    /// If it exists but can't be opened or locked.
    pub(crate) fn cached(path: &Path) -> Result<Option<Self>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
        };
        file.lock_shared()
            .with_context(|| format!("locking {}", path.display()))?;
        Ok(Some(Self {
            path: path.to_owned(),
            file,
        }))
    }

    /// Where it was opened.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Removes every cached medium project in `dir` but `keep`, except those a run is using.
///
/// Only files named `medium-*.bcrf` are touched, and one only once it is locked exclusively
/// (`flock`, without waiting): a run holds a shared lock on the project it uses
/// ([`Source::cached`]) until its copy is made, so a project in use, say by a run from another
/// worktree whose build has another key, stays. An entry that can't be checked or removed is
/// skipped and reported, as [`WorkDir::sweep`] does. A cached project never has a recovery
/// directory of its own: every run opens a copy of it in a work directory.
///
/// # Errors
///
/// If the directory can't be listed.
pub(crate) fn remove_stale(dir: &Path, keep: &Path) -> Result<Sweep> {
    let mut sweep = Sweep::default();
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(e) => {
                sweep.skipped.push((dir.to_owned(), e.into()));
                continue;
            }
        };
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path == keep
            || !name.starts_with("medium-")
            || path.extension().is_none_or(|e| e != "bcrf")
        {
            continue;
        }
        match remove_unused(&path) {
            Ok(true) => sweep.removed.push(path),
            Ok(false) => {}
            Err(e) => sweep.skipped.push((path, e)),
        }
    }
    Ok(sweep)
}

/// Removes the file at `path` unless another process holds a lock on it; returns whether it did.
///
/// # Errors
///
/// If it can't be opened, its lock can't be tried, or it can't be removed.
fn remove_unused(path: &Path) -> Result<bool> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(false),
        Err(std::fs::TryLockError::Error(e)) => {
            return Err(e).with_context(|| format!("locking {}", path.display()));
        }
    }
    std::fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
    Ok(true)
}

/// How [`WorkDir::copy`] copied a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Copied {
    /// A reflink: the copy shares the source's extents, no data written.
    Reflinked,
    /// A byte-for-byte copy: the file system can't share extents (or the two are on different
    /// ones), so the copy cost the project's size in time and disk.
    InFull,
}

/// The directory under a base directory that holds this user's perf work directories and
/// nothing else, so a sweep never looks at anything another tool or user made.
///
/// It is `<base>/brushcraft-perf-<uid>`: the user id keeps two users' runs apart in a shared
/// directory such as `/tmp`, where one user's directory would otherwise block the other's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkRoot(PathBuf);

impl WorkRoot {
    /// This user's work root under `base`. Nothing is made until a [`WorkDir`] is.
    pub(crate) fn under(base: &Path) -> Self {
        Self(base.join(format!(
            "brushcraft-perf-{}",
            rustix::process::getuid().as_raw()
        )))
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

/// What a clear-up of what other runs left ([`WorkDir::sweep`], [`remove_stale`]) did.
#[derive(Debug, Default)]
pub(crate) struct Sweep {
    /// The entries removed: dead runs' directories, or stale projects.
    pub(crate) removed: Vec<PathBuf>,
    /// The entries that looked removable but couldn't be checked or removed, with why: left in
    /// place for the user, never a reason to fail the run.
    pub(crate) skipped: Vec<(PathBuf, anyhow::Error)>,
}

/// A directory for one run's project copies, removed with everything in it when dropped.
///
/// It is named `work-<pid>-<n>` (the process id only for whoever reads the directory) and is
/// locked (`flock`, exclusive) for as long as it lives. The kernel drops the lock when the
/// process dies however it dies, so a [`WorkDir::sweep`] that can take the lock knows the
/// directory's run is gone, whatever its process id now names, in this PID namespace or another.
#[derive(Debug)]
pub(crate) struct WorkDir {
    /// The directory.
    dir: PathBuf,
    /// The open directory, holding the lock that says its run is alive.
    _lock: File,
}

impl WorkDir {
    /// A fresh, locked directory in `root`, made if missing.
    ///
    /// The directory is made under a `new-` name, locked, then renamed to its `work-` name, so a
    /// sweep never sees it unlocked. A process killed between the first two steps leaves an
    /// empty `new-` directory behind, which nothing sweeps: harmless, and a window of
    /// microseconds.
    ///
    /// # Errors
    ///
    /// If it can't be made, locked or renamed.
    pub(crate) fn new(root: &WorkRoot) -> Result<Self> {
        /// Tells apart the directories one process makes.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = root.path();
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let name = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let staging = root.join(format!("new-{name}"));
        let dir = root.join(format!("work-{name}"));
        #[expect(
            clippy::create_dir,
            reason = "the name is this process's alone: an existing directory is an error"
        )]
        std::fs::create_dir(&staging).with_context(|| format!("creating {}", staging.display()))?;
        let locked = File::open(&staging)
            .with_context(|| format!("opening {}", staging.display()))
            .and_then(|lock| {
                lock.try_lock()
                    .with_context(|| format!("locking {}", staging.display()))?;
                std::fs::rename(&staging, &dir).with_context(|| {
                    format!("renaming {} to {}", staging.display(), dir.display())
                })?;
                Ok(lock)
            });
        match locked {
            Ok(lock) => Ok(Self { dir, _lock: lock }),
            Err(e) => {
                // Best effort: the error says what went wrong; an empty directory is left at worst.
                drop(std::fs::remove_dir(&staging));
                Err(e)
            }
        }
    }

    /// Removes every work directory in `root` whose run has died (a run killed by a signal never
    /// ran its [`Drop`]), which [`WorkDir::new`]'s lock tells: an unlocked `work-` directory is
    /// a dead run's. Only `work-` entries of `root` are looked at, and a live run's directory,
    /// this process's included, stays.
    ///
    /// An entry that can't be opened, locked or removed is skipped and reported in
    /// [`Sweep::skipped`], not an error: clearing up after dead runs never fails a live one.
    ///
    /// # Errors
    ///
    /// If `root` exists but can't be listed.
    pub(crate) fn sweep(root: &WorkRoot) -> Result<Sweep> {
        let root = root.path();
        let mut sweep = Sweep::default();
        let entries = match std::fs::read_dir(root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(sweep),
            Err(e) => return Err(e).with_context(|| format!("listing {}", root.display())),
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    sweep.skipped.push((root.to_owned(), e.into()));
                    continue;
                }
            };
            if !entry.file_name().to_string_lossy().starts_with("work-") {
                continue;
            }
            let path = entry.path();
            match Self::remove_if_dead(&path) {
                Ok(true) => sweep.removed.push(path),
                Ok(false) => {}
                Err(e) => sweep.skipped.push((path, e)),
            }
        }
        Ok(sweep)
    }

    /// Removes the work directory at `path` if its run has died (no lock held on it); returns
    /// whether it did.
    ///
    /// # Errors
    ///
    /// If it can't be opened, its lock can't be tried, or it can't be removed.
    fn remove_if_dead(path: &Path) -> Result<bool> {
        let dir = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        match dir.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Ok(false),
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", path.display()));
            }
        }
        std::fs::remove_dir_all(path).with_context(|| format!("removing {}", path.display()))?;
        Ok(true)
    }

    /// Where a project named `name` goes in it: `<dir>/<name>/project.bcrf`, its own directory
    /// so its recovery files (`recovery/` beside it) stay apart from the others'.
    ///
    /// # Errors
    ///
    /// If the project's directory can't be made.
    pub(crate) fn project(&self, name: &str) -> Result<PathBuf> {
        let dir = self.dir.join(name);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir.join("project.bcrf"))
    }

    /// A copy of the project `source` as project `name` (see [`WorkDir::project`]): a reflink
    /// where the file system shares extents (no data copied, so a 1.7 GB project costs nothing),
    /// else a plain copy. It reads `source`'s open handle from its start, whatever its name now
    /// points at.
    ///
    /// # Errors
    ///
    /// If the source can't be read or the copy written.
    pub(crate) fn copy(&self, source: &Source, name: &str) -> Result<(PathBuf, Copied)> {
        let target = self.project(name)?;
        let mut to =
            File::create(&target).with_context(|| format!("creating {}", target.display()))?;
        if rustix::fs::ioctl_ficlone(&to, &source.file).is_ok() {
            return Ok((target, Copied::Reflinked));
        }
        let mut from = &source.file;
        from.seek(SeekFrom::Start(0))
            .and_then(|_| std::io::copy(&mut from, &mut to))
            .with_context(|| {
                format!("copying {} to {}", source.path.display(), target.display())
            })?;
        Ok((target, Copied::InFull))
    }

    /// The directory.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.dir
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        // Best effort: a directory left behind stays unlocked once this process ends, so the next
        // run's sweep removes it.
        drop(std::fs::remove_dir_all(&self.dir));
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn cache_keys_follow_the_shape() {
        let medium = Scale::Medium.shape().unwrap();
        let key = cache_key(medium);
        assert_eq!(key.len(), 16, "{key}");
        assert_eq!(key, cache_key(medium), "stable");
        for (other, what) in [
            (
                Shape {
                    splits: 6,
                    ..medium
                },
                "splits",
            ),
            (
                Shape {
                    levels: 8,
                    ..medium
                },
                "levels",
            ),
            (Shape { seed: 2, ..medium }, "seed"),
            (
                Shape {
                    amplitude: 0.02,
                    ..medium
                },
                "amplitude",
            ),
            (
                Shape {
                    budget_mib: 1024,
                    ..medium
                },
                "budget",
            ),
        ] {
            assert_ne!(key, cache_key(other), "another {what}, another key");
        }
        assert!(
            cached_file(Path::new("/c"), &key).ends_with(format!("medium-{key}.bcrf")),
            "named for its key"
        );
    }

    /// `generate` gets every field of the shape on its command line, so its own defaults
    /// never decide what the suite's projects hold: parsed back, the options are the shape's.
    #[test]
    fn generate_gets_the_whole_shape() {
        for scale in [Scale::Tiny, Scale::Medium] {
            let shape = scale.shape().unwrap();
            let args = std::iter::once(OsString::from("generate")).chain(shape.args());
            let options = crate::cli::Command::parse(args).unwrap().options;
            assert_eq!(
                (
                    options.splits,
                    options.levels,
                    options.seed,
                    options.amplitude.to_bits(),
                    options.budget_mib
                ),
                (
                    shape.splits,
                    shape.levels,
                    shape.seed,
                    shape.amplitude.to_bits(),
                    shape.budget_mib
                ),
                "{scale:?}"
            );
        }
    }

    /// A scratch directory for one test, empty.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-perf-{name}-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The sweep looks only inside our own work root, and removes only a dead run's directory.
    ///
    /// A live run's directory, an entry of another name, and a foreign `work-` directory beside
    /// the root (another tool's, in a shared directory such as `/tmp`) all stay.
    #[test]
    fn the_sweep_removes_dead_runs_and_touches_nothing_else() {
        let base = scratch("sweep");
        let root = WorkRoot::under(&base);
        assert!(
            root.path().starts_with(&base) && root.path() != base,
            "a directory of its own"
        );
        let live = WorkDir::new(&root).unwrap();
        // A dead run's directory: named like a work directory, nobody holding its lock.
        let dead = root.path().join("work-1-0");
        let other = root.path().join("keep-me");
        let foreign = base.join("work-1-0");
        for dir in [&dead, &other, &foreign] {
            std::fs::create_dir_all(dir.join("inside")).unwrap();
        }
        let sweep = WorkDir::sweep(&root).unwrap();
        assert_eq!(
            sweep.removed,
            std::slice::from_ref(&dead),
            "only the dead run's"
        );
        assert!(sweep.skipped.is_empty(), "{:?}", sweep.skipped);
        assert!(
            live.path().exists() && other.exists() && foreign.exists(),
            "the rest stay"
        );
        drop(live);
        assert!(
            WorkDir::sweep(&WorkRoot::under(&base.join("missing")))
                .unwrap()
                .removed
                .is_empty(),
            "no root, nothing to do"
        );
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// A dead run's directory that can't be removed is skipped with the reason, and the sweep
    /// still succeeds, so the run that swept goes on.
    #[test]
    fn an_unremovable_entry_is_skipped_not_fatal() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = scratch("stuck");
        let root = WorkRoot::under(&base);
        let stuck = root.path().join("work-1-0");
        let sealed = stuck.join("sealed");
        std::fs::create_dir_all(&sealed).unwrap();
        std::fs::write(sealed.join("file"), b"x").unwrap();
        // Read and enter only: the file inside can't be unlinked.
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o500)).unwrap();
        let sweep = WorkDir::sweep(&root);
        std::fs::set_permissions(&sealed, std::fs::Permissions::from_mode(0o700)).unwrap();
        let sweep = sweep.unwrap();
        assert!(sweep.removed.is_empty(), "{:?}", sweep.removed);
        assert_eq!(
            sweep
                .skipped
                .iter()
                .map(|(path, _)| path)
                .collect::<Vec<_>>(),
            [&stuck],
            "skipped, with the reason"
        );
        assert!(WorkDir::new(&root).is_ok(), "the run goes on");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn stale_projects_go_and_nothing_else() {
        let root = scratch("cache");
        let keep = cached_file(&root, "1111111111111111");
        let stale = cached_file(&root, "2222222222222222");
        let other = root.join("sphere.bcrf");
        for path in [&keep, &stale, &other] {
            std::fs::write(path, b"x").unwrap();
        }
        let sweep = remove_stale(&root, &keep).unwrap();
        assert_eq!(
            sweep.removed,
            std::slice::from_ref(&stale),
            "only the stale one"
        );
        assert!(sweep.skipped.is_empty(), "{:?}", sweep.skipped);
        assert!(
            keep.exists() && other.exists() && !stale.exists(),
            "the rest stay"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Two runs from builds with different keys share the cache: neither removes the project
    /// the other is using, and one no longer used goes on the next tidy.
    #[test]
    fn a_project_in_use_is_never_removed() {
        let root = scratch("in-use");
        let mine = cached_file(&root, "1111111111111111");
        let theirs = cached_file(&root, "2222222222222222");
        for path in [&mine, &theirs] {
            std::fs::write(path, b"x").unwrap();
        }
        let using = Source::cached(&theirs).unwrap().unwrap();
        assert!(
            Source::cached(&root.join("medium-none.bcrf"))
                .unwrap()
                .is_none(),
            "none cached"
        );
        let sweep = remove_stale(&root, &mine).unwrap();
        assert!(
            sweep.removed.is_empty() && sweep.skipped.is_empty(),
            "{sweep:?}"
        );
        assert!(theirs.exists(), "in use, so kept");
        drop(using);
        assert_eq!(
            remove_stale(&root, &mine).unwrap().removed,
            std::slice::from_ref(&theirs),
            "unused, so removed"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A copy reads the project the run opened, even if its name was removed or replaced since.
    #[test]
    fn a_copy_reads_the_opened_project_whatever_its_name_holds_now() {
        let root = scratch("replaced");
        let cached = root.join("medium-1.bcrf");
        std::fs::write(&cached, vec![7_u8; 1 << 12]).unwrap();
        let source = Source::cached(&cached).unwrap().unwrap();
        std::fs::remove_file(&cached).unwrap();
        std::fs::write(&cached, b"another project").unwrap();
        let work = WorkDir::new(&WorkRoot::under(&root)).unwrap();
        let (copy, _) = work.copy(&source, "medium").unwrap();
        assert_eq!(
            std::fs::read(copy).unwrap(),
            vec![7_u8; 1 << 12],
            "the opened one"
        );
        drop(work);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn work_dirs_copy_projects_and_clean_up() {
        let root = scratch("work");
        let source = root.join("source.bcrf");
        std::fs::write(&source, vec![5_u8; 1 << 16]).unwrap();
        let work = WorkDir::new(&WorkRoot::under(&root)).unwrap();
        let (copy, _) = work
            .copy(&Source::open(&source).unwrap(), "medium")
            .unwrap();
        assert_eq!(
            copy,
            work.path().join("medium").join("project.bcrf"),
            "its own directory"
        );
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            vec![5_u8; 1 << 16],
            "the same bytes"
        );
        let dir = work.path().to_owned();
        drop(work);
        assert!(!dir.exists(), "removed with its copies");
        assert!(source.exists(), "the source stays");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
