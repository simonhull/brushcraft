//! Which project files the app opens: never one inside the repository it was built from.
//!
//! Opening a project writes beside it (a recovery file, locks, and saves into the file
//! itself), and projects run to gigabytes, so a project inside the source tree is refused,
//! whatever `.gitignore` says, as the benchmark harness refuses one. A copy of the binary
//! run away from its repository refuses nothing.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, Result, ensure};

/// `path` made absolute, checked to name a file outside the repository.
///
/// # Errors
///
/// If `path` can't be made absolute or resolved, names nothing, or lies inside the
/// repository.
pub(crate) fn project(path: &Path) -> Result<PathBuf> {
    let project =
        std::path::absolute(path).with_context(|| format!("making {} absolute", path.display()))?;
    ensure!(
        project.is_file(),
        "{} isn't a project file",
        project.display()
    );
    ensure!(
        !inside_repo(&project)?,
        "{} is inside the repository ({}); projects live outside it",
        project.display(),
        repo_root().display()
    );
    Ok(project)
}

/// The repository the app was built from: two levels above this crate's manifest.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Whether `path` (absolute) lies inside the repository, symbolic links resolved.
///
/// # Errors
///
/// If resolving the path fails.
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
    let outward = rest.into_iter().rev();
    for name in outward {
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
    fn a_project_inside_the_repository_is_refused() {
        let inside = repo_root().join("Cargo.toml");
        let error = project(&inside).unwrap_err();
        assert!(
            error.to_string().contains("inside the repository"),
            "refused: {error}"
        );
        let sneaky = std::env::temp_dir().join("..").join(
            repo_root()
                .canonicalize()
                .unwrap()
                .strip_prefix("/")
                .unwrap(),
        );
        assert!(
            inside_repo(&std::path::absolute(sneaky.join("Cargo.toml")).unwrap()).unwrap(),
            "through `..`: {}",
            sneaky.display()
        );
    }

    #[test]
    fn a_file_outside_it_is_taken_and_a_missing_one_refused() {
        let dir = std::env::temp_dir().join(format!("brushcraft-app-paths-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("p.bcrf");
        project(&file).unwrap_err();
        std::fs::write(&file, b"").unwrap();
        assert_eq!(project(&file).unwrap(), file);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
