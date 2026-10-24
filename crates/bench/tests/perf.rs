//! `brushcraft-bench perf --tiny` end to end, through the built binary: the suite generates a
//! tiny project, runs every part on it in child processes, prints a metric record per line and
//! cleans up after itself. The stage benches are left out (`--no-stages`): they run `cargo
//! bench`, which CI runs separately (see `.github/workflows/ci.yml`).

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    /// A fresh, empty directory for one test, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        /// A directory named for this process and `name` under the system's temporary one.
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("brushcraft-perf-it-{}-{name}", std::process::id()));
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

    /// Runs `brushcraft-bench perf` with `args`, its temporary and cache directories in
    /// `scratch`, so whatever it leaves behind shows there.
    fn perf(scratch: &Path, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_brushcraft-bench"))
            .arg("perf")
            .args(args)
            .env("TMPDIR", scratch)
            .env("XDG_CACHE_HOME", scratch.join("cache"))
            .output()
            .unwrap()
    }

    /// The tiny suite runs every part, prints one unchecked record per metric (none of them a
    /// baseline), and leaves no project behind.
    #[test]
    fn the_tiny_suite_runs_end_to_end() {
        let scratch = Scratch::new("tiny");
        let output = perf(&scratch.0, &["--tiny", "--no-stages"]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{}\n{stderr}", output.status);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let lines: Vec<&str> = stdout.lines().collect();
        for line in &lines {
            assert!(
                line.starts_with(r#"{"bench":"perf","result":"metric","metric":"tiny."#),
                "{line}"
            );
            for field in [
                r#""label":"unchecked""#,
                r#""baseline":false"#,
                r#""project":"tiny""#,
            ] {
                assert!(line.contains(field), "{field} in {line}");
            }
        }
        for metric in [
            "tiny.brush.p50_ms",
            "tiny.brush.p99_ms",
            "tiny.brush.evaluate_ms",
            "tiny.brush.cold_first_dab_ms",
            "tiny.index.ready_ms",
            "tiny.page_in.footprint.cold_pool_ms",
            "tiny.page_in.region.warm_pool_ms",
            "tiny.budget.peak_rss_mib",
            "tiny.save.fixed_ms",
            "tiny.save.per_vertex_us",
            "tiny.save.bytes_per_vertex",
        ] {
            let field = format!(r#""metric":"{metric}""#);
            assert_eq!(
                lines.iter().filter(|l| l.contains(&field)).count(),
                1,
                "one {metric} in {stdout}"
            );
        }
        assert!(stderr.contains("run labelled unchecked"), "{stderr}");
        // The work root (`brushcraft-perf-<uid>`) stays for the next run; what it held goes.
        let left: Vec<_> = std::fs::read_dir(&scratch.0)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|path| !path.ends_with("cache"))
            .flat_map(|root| {
                assert!(
                    root.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("brushcraft-perf-")),
                    "only the work root: {}",
                    root.display()
                );
                std::fs::read_dir(root)
                    .unwrap()
                    .map(|e| e.unwrap().file_name())
            })
            .collect();
        assert!(left.is_empty(), "the work directory is gone: {left:?}");
    }

    /// What the tiny suite refuses: it is never recorded, and `--baseline` needs a history.
    #[test]
    fn the_tiny_suite_is_never_recorded() {
        let scratch = Scratch::new("refused");
        for args in [
            ["--tiny", "--history", "/nonexistent"].as_slice(),
            &["--tiny", "--baseline"],
            &["--baseline"],
        ] {
            let output = perf(&scratch.0, args);
            assert!(!output.status.success(), "{args:?} refused");
            assert!(output.stdout.is_empty(), "{args:?}: nothing measured");
        }
    }
}
