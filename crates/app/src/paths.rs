//! Which project files the app opens: never one inside the source checkout it was built from.
//!
//! Opening a project writes beside it (a recovery file, locks, and saves into the file
//! itself), and projects run to gigabytes, so a project inside the checkout is refused,
//! whatever `.gitignore` says, as the benchmark harness refuses one. A binary built in a git
//! worktree refuses the main checkout too ([`checkout`]).

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};

/// `path` made absolute, checked to name a file outside the protected checkouts.
///
/// # Errors
///
/// If `path` can't be made absolute or resolved, lies inside a protected checkout, or names
/// no file.
pub(crate) fn project(path: &Path) -> Result<PathBuf> {
    let project =
        std::path::absolute(path).with_context(|| format!("making {} absolute", path.display()))?;
    let inside = checkout::containing(&project)
        .with_context(|| format!("resolving {}", project.display()))?;
    if let Some(root) = inside {
        anyhow::bail!(
            "{} is inside the repository ({}); projects live outside it",
            project.display(),
            root.display()
        );
    }
    ensure!(
        project.is_file(),
        "{} isn't a project file",
        project.display()
    );
    Ok(project)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn a_project_inside_the_repository_is_refused() {
        let inside = checkout::build_root().join("Cargo.toml");
        let error = project(&inside).unwrap_err();
        assert!(
            error.to_string().contains("inside the repository"),
            "refused: {error}"
        );
        let missing = checkout::build_root().join("target").join("none.bcrf");
        let unwritten = project(&missing).unwrap_err();
        assert!(
            unwritten.to_string().contains("inside the repository"),
            "refused whether it exists or not: {unwritten}"
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
