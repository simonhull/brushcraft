//! The projects the suite measures, and where they live: the tiny one made fresh for each run,
//! the medium one cached, the full one the benchmark's own, each measured through a copy.

use std::ffi::OsString;
use std::fs::File;
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

/// Removes every cached medium project in `dir` but `keep`.
///
/// Only files named `medium-*.bcrf` are touched. A cached project never has a recovery
/// directory of its own: every run opens a copy of it in a work directory.
///
/// # Errors
///
/// If the directory can't be listed or a file removed.
pub(crate) fn remove_stale(dir: &Path, keep: &Path) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path != keep
            && name.starts_with("medium-")
            && path.extension().is_some_and(|e| e == "bcrf")
        {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            removed.push(path);
        }
    }
    Ok(removed)
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

/// The process id in a work directory's name, `work-<pid>-<n>`; `None` for any other name.
fn work_pid(name: &str) -> Option<u32> {
    let (pid, n) = name.strip_prefix("work-")?.split_once('-')?;
    n.parse::<u64>().ok()?;
    pid.parse().ok()
}

/// A directory for one run's project copies, removed with everything in it when dropped.
#[derive(Debug)]
pub(crate) struct WorkDir(PathBuf);

impl WorkDir {
    /// A fresh directory under `root`, named for this process.
    ///
    /// # Errors
    ///
    /// If it can't be made.
    pub(crate) fn new(root: &Path) -> Result<Self> {
        /// Tells apart the directories one process makes.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = root.join(format!(
            "work-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self(dir))
    }

    /// Removes every work directory under `root` whose process has died (a run killed by a
    /// signal never ran its [`Drop`]); returns the directories removed. A directory whose
    /// process still runs, or whose name doesn't read `work-<pid>-<n>`, is left alone.
    ///
    /// # Errors
    ///
    /// If `root` can't be listed or a dead run's directory removed.
    pub(crate) fn sweep(root: &Path) -> Result<Vec<PathBuf>> {
        let mut removed = Vec::new();
        let entries = match std::fs::read_dir(root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(removed),
            Err(e) => return Err(e).with_context(|| format!("listing {}", root.display())),
        };
        for entry in entries {
            let path = entry?.path();
            let Some(pid) = path.file_name().and_then(|n| n.to_str()).and_then(work_pid) else {
                continue;
            };
            if pid != std::process::id() && !Path::new("/proc").join(pid.to_string()).exists() {
                std::fs::remove_dir_all(&path)
                    .with_context(|| format!("removing {}", path.display()))?;
                removed.push(path);
            }
        }
        Ok(removed)
    }

    /// Where a project named `name` goes in it: `<dir>/<name>/project.bcrf`, its own directory
    /// so its recovery files (`recovery/` beside it) stay apart from the others'.
    ///
    /// # Errors
    ///
    /// If the project's directory can't be made.
    pub(crate) fn project(&self, name: &str) -> Result<PathBuf> {
        let dir = self.0.join(name);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir.join("project.bcrf"))
    }

    /// A copy of the project `source` as project `name` (see [`WorkDir::project`]): a reflink
    /// where the file system shares extents (no data copied, so a 1.7 GB project costs nothing),
    /// else a plain copy.
    ///
    /// # Errors
    ///
    /// If the source can't be read or the copy written.
    pub(crate) fn copy(&self, source: &Path, name: &str) -> Result<(PathBuf, Copied)> {
        let target = self.project(name)?;
        let from = File::open(source).with_context(|| format!("opening {}", source.display()))?;
        let to = File::create(&target).with_context(|| format!("creating {}", target.display()))?;
        if rustix::fs::ioctl_ficlone(&to, &from).is_ok() {
            return Ok((target, Copied::Reflinked));
        }
        drop(to);
        std::fs::copy(source, &target)
            .with_context(|| format!("copying {} to {}", source.display(), target.display()))?;
        Ok((target, Copied::InFull))
    }

    /// The directory.
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        // Best effort: a copy left behind is in the cache directory, named for its process.
        drop(std::fs::remove_dir_all(&self.0));
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

    /// Work directories of dead runs are swept; live ones, this run's and other names stay.
    #[test]
    fn dead_runs_work_dirs_are_swept() {
        assert_eq!(work_pid("work-123-0"), Some(123), "a work directory");
        for other in ["work-123", "work-x-0", "medium-1.bcrf", "work-1-x"] {
            assert_eq!(work_pid(other), None, "{other}");
        }
        let root =
            std::env::temp_dir().join(format!("brushcraft-perf-sweep-{}", std::process::id()));
        // PIDs past the kernel's maximum (2^22) are never alive.
        let dead = root.join("work-4999999-0");
        let live = root.join(format!("work-{}-7", std::process::id()));
        let init = root.join("work-1-0");
        let other = root.join("keep-me");
        for dir in [&dead, &live, &init, &other] {
            std::fs::create_dir_all(dir).unwrap();
        }
        assert_eq!(
            WorkDir::sweep(&root).unwrap(),
            std::slice::from_ref(&dead),
            "only the dead run's"
        );
        assert!(
            live.exists() && init.exists() && other.exists(),
            "the rest stay"
        );
        assert!(
            WorkDir::sweep(&root.join("missing")).unwrap().is_empty(),
            "no root"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn stale_projects_go_and_nothing_else() {
        let root =
            std::env::temp_dir().join(format!("brushcraft-perf-cache-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let keep = cached_file(&root, "1111111111111111");
        let stale = cached_file(&root, "2222222222222222");
        let other = root.join("sphere.bcrf");
        for path in [&keep, &stale, &other] {
            std::fs::write(path, b"x").unwrap();
        }
        assert_eq!(
            remove_stale(&root, &keep).unwrap(),
            std::slice::from_ref(&stale),
            "only the stale one"
        );
        assert!(
            keep.exists() && other.exists() && !stale.exists(),
            "the rest stay"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn work_dirs_copy_projects_and_clean_up() {
        let root =
            std::env::temp_dir().join(format!("brushcraft-perf-work-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("source.bcrf");
        std::fs::write(&source, vec![5_u8; 1 << 16]).unwrap();
        let work = WorkDir::new(&root).unwrap();
        let (copy, _) = work.copy(&source, "medium").unwrap();
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
