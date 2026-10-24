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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    /// The base mesh's `cube_sphere` splits.
    pub(crate) splits: u32,
    /// Levels sculpted, from 1; the last is the one measured.
    pub(crate) levels: u8,
}

impl Shape {
    /// The arguments `generate` takes for this shape (the seed and amplitude are its
    /// defaults).
    pub(crate) fn args(self) -> Vec<OsString> {
        [
            "--splits",
            &self.splits.to_string(),
            "--levels",
            &self.levels.to_string(),
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
            Self::Tiny => Some(Shape {
                splits: 2,
                levels: 6,
            }),
            Self::Medium => Some(Shape {
                splits: 5,
                levels: 9,
            }),
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

/// FNV-1a over `bytes` (64-bit).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The cache key of a project of `shape` made by this build: the generator's code fingerprint
/// (see `build.rs`) and the shape, hashed together, as 16 hex digits.
pub(crate) fn cache_key(shape: Shape) -> String {
    let text = format!(
        "{} splits {} levels {}",
        env!("BRUSHCRAFT_GENERATOR_HASH"),
        shape.splits,
        shape.levels
    );
    format!("{:016x}", fnv1a(text.as_bytes()))
}

/// The cached medium project's file in `dir` for key `key`.
pub(crate) fn cached_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("medium-{key}.bcrf"))
}

/// Removes every cached medium project in `dir` but `keep`, with its recovery directory if it
/// has one. Only files named `medium-*.bcrf` are touched.
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
    pub(crate) fn copy(&self, source: &Path, name: &str) -> Result<PathBuf> {
        let target = self.project(name)?;
        let from = File::open(source).with_context(|| format!("opening {}", source.display()))?;
        let to = File::create(&target).with_context(|| format!("creating {}", target.display()))?;
        if rustix::fs::ioctl_ficlone(&to, &from).is_err() {
            drop(to);
            std::fs::copy(source, &target)
                .with_context(|| format!("copying {} to {}", source.display(), target.display()))?;
        }
        Ok(target)
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
        assert_ne!(
            key,
            cache_key(Shape {
                splits: 6,
                ..medium
            }),
            "another shape, another key"
        );
        assert!(
            cached_file(Path::new("/c"), &key).ends_with(format!("medium-{key}.bcrf")),
            "named for its key"
        );
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
        let copy = work.copy(&source, "medium").unwrap();
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
