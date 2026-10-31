//! `brushcraft --offscreen --frames N` end to end, through the built binary, on the software
//! adapter (lavapipe in the gate): the app opens a saved project, uploads its floor of display
//! tiles to the GPU, draws its surface in its frames with no window (the finest tiles the level
//! choice asks for arriving), closes it, and leaves nothing behind.

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
                "30",
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
            stdout.contains("30 frames on llvmpipe") && stdout.contains("(offscreen)"),
            "the summary: {stdout}"
        );
        // The floor (24 faces' display tiles at levels 0 to 3, under level 4) went up to the
        // GPU and is resident, and no frame's uploads were lost.
        assert!(
            stdout.contains("(floor 96/96)") && stdout.contains("0 lost frames"),
            "the floor is resident: {stdout}"
        );
        // The level choice drew the surface and asked for finest tiles, which arrived: more than
        // the floor is resident, and the last cut read back wants nothing more.
        let tiles = line(&stdout, "tiles: ");
        let resident = number_before(tiles, " resident");
        assert!(resident > 96, "finest tiles arrived: {tiles}");
        let scene = line(&stdout, "scene: ");
        let cut = number_after(scene, "cut ");
        assert!(cut >= 24, "a cut covering every face came back: {scene}");
        assert!(
            scene.contains(" 0 requested"),
            "every request served: {scene}"
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

    /// The rest of the line of `text` that starts with `prefix`.
    fn line<'a>(text: &'a str, prefix: &str) -> &'a str {
        text.lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("no {prefix:?} line in:\n{text}"))
    }

    /// The number `text` starts with, up to `marker`.
    fn number_before(text: &str, marker: &str) -> usize {
        let (number, _) = text.split_once(marker).expect("the marker");
        number.trim().parse().expect("a number")
    }

    /// The number right after `marker` in `text`.
    fn number_after(text: &str, marker: &str) -> usize {
        let (_, rest) = text.split_once(marker).expect("the marker");
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().expect("a number")
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
        assert_eq!(
            output.status.code(),
            Some(69),
            "no GPU has a status of its own"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("no adapter matches"), "{stderr}");
        let recovery = state.join("brushcraft").join("recovery");
        let left: Vec<_> = std::fs::read_dir(&recovery)
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default();
        assert!(left.is_empty(), "nothing left behind: {left:?}");
    }

    /// The value of `field` in the JSON line `line`, as written: up to the next comma or brace
    /// (arrays and strings with commas aren't asked for).
    fn field<'a>(line: &'a str, field: &str) -> &'a str {
        let key = format!("\"{field}\":");
        let (_, rest) = line
            .split_once(&key)
            .unwrap_or_else(|| panic!("no {field} in {line}"));
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        &rest[..end]
    }

    /// Runs the benchmark script at 8× on a small project on lavapipe, offscreen, timing the GPU
    /// or not, and returns its record's lines.
    fn scripted_record(name: &str, timed: bool) -> Vec<String> {
        let scratch = Scratch::new(name);
        let project = scratch.0.join("small.bcrf");
        let record = scratch.0.join("record.jsonl");
        save_project(&project, &scratch.0.join("setup"));
        let record_arg = record.to_str().unwrap();
        let mut args = vec![
            "--offscreen",
            "--script",
            "--script-speed",
            "8",
            "--record",
            record_arg,
            "--adapter",
            "software",
            "--level",
            "4",
            "--size",
            "160x100",
        ];
        if !timed {
            args.push("--no-gpu-timing");
        }
        let output = brushcraft(&project, &scratch.0.join("state"), &args);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "exit {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        let text = std::fs::read_to_string(&record).unwrap();
        text.lines().map(str::to_owned).collect()
    }

    /// A scripted run's record: its setting, the five segments, then every frame in order, the
    /// camera following the path to its end; the GPU's phases timed where asked and the device
    /// can, `null` where not.
    fn check_record(lines: &[String], timed: bool) {
        let run = &lines[0];
        assert!(
            run.starts_with(r#"{"bench":"render","result":"run""#),
            "{run}"
        );
        let timing = field(run, "gpu_timing");
        if !timed {
            assert_eq!(timing, "false", "not asked: {run}");
        }
        assert_eq!(
            field(run, "readbacks_dropped"),
            "0",
            "every frame found a readback buffer: {run}"
        );
        let segments: Vec<&str> = lines[1..6].iter().map(|l| field(l, "segment")).collect();
        assert_eq!(
            segments,
            [
                r#""open""#,
                r#""orbit""#,
                r#""zoom""#,
                r#""flick""#,
                r#""settle""#
            ]
        );
        let frames = &lines[6..];
        let count: usize = field(run, "frames").parse().unwrap();
        assert_eq!(frames.len(), count, "a line a frame");
        // The run follows the wall clock, so how many frames it makes depends on the machine's
        // load (3 under a load of 15): at least its first and one at its end. What follows
        // depends on frames, not time.
        assert!(count >= 2, "a first frame and a last: {count}");
        for (number, frame) in (1..).zip(frames) {
            assert!(frame.contains(r#""result":"frame""#), "{frame}");
            assert_eq!(field(frame, "frame"), number.to_string(), "in order");
        }
        let last = frames.last().unwrap();
        assert_eq!(field(last, "segment"), r#""settle""#, "run to its end");
        let timed_frames = frames
            .iter()
            .filter(|f| field(f, "gpu_ms") != "null")
            .count();
        // A frame starts only once the GPU has finished all but one before it
        // (`render::FRAMES_IN_FLIGHT`, 2), and its start reads back what they drew: so by the
        // third frame the first's cut (and its level choice's stages) came back, however slowly
        // the frames came.
        let cuts_due = count >= 3;
        if timing == "true" {
            // Every frame's, the last ones' read as the render thread stopped.
            assert_eq!(timed_frames, count, "GPU times came back");
            let stages = frames
                .iter()
                .filter(|f| field(f, "lod_choose_ms") != "null");
            assert!(
                !cuts_due || stages.count() > 0,
                "the level choice's stages too"
            );
        } else {
            assert_eq!(timed_frames, 0, "no GPU times without timing");
        }
        let drawn = frames
            .iter()
            .filter(|f| field(f, "drawn") == "true")
            .count();
        assert!(!cuts_due || drawn > 0, "cuts came back");
        assert!(
            frames.iter().any(|f| field(f, "moving") == "true"),
            "the camera moved"
        );
    }

    #[test]
    fn a_scripted_run_follows_the_path_and_records_every_frame() {
        let timed = scripted_record("script-timed", true);
        check_record(&timed, true);
        let untimed = scripted_record("script-untimed", false);
        check_record(&untimed, false);
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
