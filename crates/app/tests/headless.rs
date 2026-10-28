//! `brushcraft --offscreen --frames N` end to end, through the built binary, on the software
//! adapter (lavapipe in the gate): the app opens a saved project, draws its frames with no
//! window, closes it, and leaves nothing behind.

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use geom::cube_sphere;
    use store::{Project, ProjectConfig, StoreConfig};

    /// A fresh, empty directory for one test, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        /// A directory named for this process and `name` under the system's temporary one.
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("brushcraft-app-it-{}-{name}", std::process::id()));
            drop(std::fs::remove_dir_all(&dir));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    /// The configuration the tests open projects with: recovery files in `recovery`.
    fn config(recovery: &Path) -> ProjectConfig {
        ProjectConfig {
            store: StoreConfig::new(512 << 20, 1),
            recovery_dir: recovery.to_owned(),
        }
    }

    /// Saves a small project (a 24-face cube sphere, no edits) at `path`, its recovery file
    /// made and deleted in `recovery`.
    fn save_project(path: &Path, recovery: &Path) {
        let mesh = cube_sphere(2).unwrap();
        let mut project = Project::new(mesh, config(recovery)).unwrap();
        let saved = project.save_as(path).unwrap();
        assert!(
            saved.unprotected.is_empty() && saved.kept.is_empty(),
            "{saved:?}"
        );
        project.close(false).unwrap();
    }

    /// Runs `brushcraft` on `project` with `args`, its state (the recovery directory) in
    /// `state`.
    fn brushcraft(project: &Path, state: &Path, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_brushcraft"))
            .arg(project)
            .args(args)
            .env("XDG_STATE_HOME", state)
            .output()
            .unwrap()
    }

    #[test]
    fn the_app_opens_and_closes_a_project_offscreen() {
        let scratch = Scratch::new("offscreen");
        let project = scratch.0.join("small.bcrf");
        let state = scratch.0.join("state");
        save_project(&project, &scratch.0.join("setup"));
        let output = brushcraft(
            &project,
            &state,
            &[
                "--offscreen",
                "--frames",
                "8",
                "--adapter",
                "software",
                "--level",
                "4",
                "--size",
                "320x200",
            ],
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "exit {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        assert!(
            stdout.contains("8 frames on llvmpipe") && stdout.contains("(offscreen)"),
            "the summary: {stdout}"
        );
        let recovery = state.join("brushcraft").join("recovery");
        let left: Vec<_> = std::fs::read_dir(&recovery)
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default();
        assert!(
            left.is_empty(),
            "no edits, so no recovery file kept: {left:?}"
        );
        // Closed and unlocked: it opens again.
        let reopened = Project::open(&project, config(&scratch.0.join("again"))).unwrap();
        reopened.close(false).unwrap();
    }

    #[test]
    fn a_launch_that_fails_before_drawing_leaves_no_recovery_file() {
        let scratch = Scratch::new("bad-adapter");
        let project = scratch.0.join("small.bcrf");
        let state = scratch.0.join("state");
        save_project(&project, &scratch.0.join("setup"));
        let output = brushcraft(
            &project,
            &state,
            &[
                "--offscreen",
                "--frames",
                "1",
                "--adapter",
                "no such adapter, surely",
            ],
        );
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("no adapter matches"), "{stderr}");
        let recovery = state.join("brushcraft").join("recovery");
        let left: Vec<_> = std::fs::read_dir(&recovery)
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default();
        assert!(left.is_empty(), "nothing left behind: {left:?}");
    }

    #[test]
    fn a_missing_project_is_refused_before_any_window_or_gpu() {
        let scratch = Scratch::new("missing");
        let output = brushcraft(
            &scratch.0.join("none.bcrf"),
            &scratch.0,
            &["--offscreen", "--frames", "1", "--adapter", "software"],
        );
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("isn't a project file"), "{stderr}");
    }
}
