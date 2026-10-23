//! Where the benchmark's files go: outside the repository, under the cache directory unless
//! told otherwise.
//!
//! A benchmark project runs to gigabytes, and its recovery files as much again, so the harness
//! refuses any path inside the repository it was built from, whatever `.gitignore` says.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, ensure};

/// The project file's name in the default directory.
const DEFAULT_NAME: &str = "sphere.bcrf";

/// Where a run's files go: the project file, and the directory its recovery files go in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Paths {
    /// The project file.
    pub(crate) project: PathBuf,
    /// The recovery directory: `recovery/` beside the project file.
    pub(crate) recovery: PathBuf,
}

impl Paths {
    /// The paths for `project`, or for the default project if `None`, checked to be outside
    /// the repository; the project's directory is created if it is missing.
    ///
    /// # Errors
    ///
    /// If no default can be found (neither `XDG_CACHE_HOME` nor `HOME` is an absolute path),
    /// a path lies inside the repository, or the directory can't be made.
    pub(crate) fn resolve(project: Option<&Path>) -> Result<Self> {
        let project = match project {
            Some(path) => std::path::absolute(path)
                .with_context(|| format!("making {} absolute", path.display()))?,
            None => default_dir()?.join(DEFAULT_NAME),
        };
        let dir = project
            .parent()
            .ok_or_else(|| anyhow!("{} has no directory", project.display()))?;
        let recovery = dir.join("recovery");
        for path in [&project, &recovery] {
            ensure!(
                !inside_repo(path)?,
                "{} is inside the repository ({}); benchmark files go outside it",
                path.display(),
                repo_root()
                    .canonicalize()
                    .unwrap_or_else(|_| repo_root())
                    .display()
            );
        }
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { project, recovery })
    }
}

/// The default directory: `$XDG_CACHE_HOME/brushcraft/bench`, else
/// `$HOME/.cache/brushcraft/bench`.
///
/// # Errors
///
/// If neither variable names an absolute path.
fn default_dir() -> Result<PathBuf> {
    /// The variable's value as a path, if it is an absolute one.
    fn absolute(name: &str) -> Option<PathBuf> {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    }
    let cache = absolute("XDG_CACHE_HOME")
        .or_else(|| absolute("HOME").map(|home| home.join(".cache")))
        .ok_or_else(|| anyhow!("neither XDG_CACHE_HOME nor HOME is an absolute path"))?;
    Ok(cache.join("brushcraft").join("bench"))
}

/// The repository the harness was built from: two levels above this crate's manifest.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Whether `path` (absolute) lies inside the repository, symbolic links resolved.
///
/// The path needn't exist: its deepest existing ancestor is resolved and the rest appended,
/// with `..` and `.` taken lexically after that.
///
/// # Errors
///
/// If resolving an existing ancestor fails.
fn inside_repo(path: &Path) -> Result<bool> {
    let Ok(root) = repo_root().canonicalize() else {
        // The repository isn't there any more (the binary was copied elsewhere): nothing to
        // protect.
        return Ok(false);
    };
    Ok(resolve(path)?.starts_with(root))
}

/// `path` (absolute) with its deepest existing ancestor's symbolic links resolved, and the
/// rest appended with `.` and `..` applied lexically.
///
/// # Errors
///
/// If canonicalizing that ancestor fails.
fn resolve(path: &Path) -> Result<PathBuf> {
    let mut existing = path;
    let mut rest = Vec::new();
    while !existing.exists() {
        let (Some(parent), Some(name)) = (existing.parent(), existing.file_name()) else {
            break;
        };
        rest.push(name);
        existing = parent;
    }
    let mut resolved = existing
        .canonicalize()
        .with_context(|| format!("resolving {}", existing.display()))?;
    for name in rest.into_iter().rev() {
        match Path::new(name).components().next() {
            Some(Component::ParentDir) => {
                resolved.pop();
            }
            Some(Component::CurDir) | None => {}
            Some(_) => resolved.push(name),
        }
    }
    Ok(resolved)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn paths_inside_the_repository_are_refused() {
        let inside = repo_root().join("target").join("x.bcrf");
        let err = Paths::resolve(Some(&inside)).unwrap_err();
        assert!(
            err.to_string().contains("inside the repository"),
            "refused: {err}"
        );
        let sneaky = std::env::temp_dir()
            .join("..")
            .join(
                repo_root()
                    .canonicalize()
                    .unwrap()
                    .strip_prefix("/")
                    .unwrap(),
            )
            .join("y.bcrf");
        assert!(
            inside_repo(&std::path::absolute(&sneaky).unwrap()).unwrap(),
            "through `..`: {}",
            sneaky.display()
        );
    }

    #[test]
    fn paths_outside_it_resolve_with_a_recovery_directory_beside() {
        let dir =
            std::env::temp_dir().join(format!("brushcraft-bench-paths-{}", std::process::id()));
        let paths = Paths::resolve(Some(&dir.join("p.bcrf"))).unwrap();
        assert_eq!(paths.recovery, dir.join("recovery"), "beside the project");
        assert!(dir.is_dir(), "the directory is made");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
