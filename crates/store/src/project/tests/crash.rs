//! The crash test, the one this milestone exists for: a real process, killed by
//! `std::process::abort` mid-stroke or mid-flush, and its project restored by the next one.
//!
//! # How it runs
//!
//! Each case re-executes the test binary ([`std::env::current_exe`]) as a child, asking it to
//! run only [`crash_child`] (an ignored test, so a normal run skips it): `--exact` with the
//! test's full path, and `--ignored`. libtest takes these arguments whether the binary was
//! started by `cargo test` or by `cargo nextest run`, which runs each test as
//! `<binary> --exact <name>` too. The [`SCENARIO`] variable tells the child what to do and
//! [`DIR`] where; without them it does nothing.
//!
//! The child creates a project with its recovery files in that directory, saves it once (or
//! never), makes N strokes, waiting after each until its flush is durable, and then crashes:
//! either mid-stroke, after edits with no stroke end, or mid-flush, at a chosen phase of the
//! flush of a stroke that ended. It runs under a shell's `ulimit -c 0`, so the abort leaves no
//! core file.
//!
//! The parent checks that the child died of `SIGABRT`, lists the recovery files, restores the
//! one it finds, and checks the evaluation against the dense evaluation of exactly the edits
//! that should have survived: every completed stroke, none of the aborted one. Then it saves
//! the restored project and checks the project file holds them too.

use std::os::unix::process::ExitStatusExt;
use std::process::Command;
use std::str::FromStr;
use std::time::Duration;

use super::*;
use crate::tiles::tests::{record, tight};

/// The variable holding the child's [`Scenario`]; the test binary is a crash test's child only
/// when it is set.
const SCENARIO: &str = "BRUSHCRAFT_CRASH_SCENARIO";

/// The variable holding the directory the child works in: the parent's [`TestDir`].
const DIR: &str = "BRUSHCRAFT_CRASH_DIR";

/// The child test's full path, as libtest names it.
const CHILD_TEST: &str = "project::tests::crash::crash_child";

/// The signal `std::process::abort` raises.
const SIGABRT: i32 = 6;

/// Edits per stroke.
const EDITS: u32 = 60;

/// Edits of the stroke the crash interrupts: enough for the tight budget to write some of its
/// tiles back.
const ABORTED_EDITS: u32 = 400;

/// The seed of the stroke made before the save.
const SAVED_SEED: u64 = 7;

/// The seed of the stroke the crash interrupts.
const ABORTED_SEED: u64 = 999;

/// The seed of the `k`th stroke after the save, counting from 0.
fn seed(k: u32) -> u64 {
    100 + u64::from(k)
}

/// Where the child aborts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Abort {
    /// Mid-stroke: after the edits of one more stroke, with no stroke end.
    MidStroke,
    /// Mid-flush: one more stroke ends, and the child aborts once its flush is in this phase
    /// (by [`RecoveryWriter`]'s test name of it: `encoding`, `syncing tiles` or
    /// `syncing save point`).
    MidFlush(&'static str),
}

/// What the child does before it aborts.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    /// Whether the project is saved (once, after one stroke) before the strokes.
    saved: bool,
    /// Strokes made after the save (or from the start), each flushed and durable.
    strokes: u32,
    /// Where the child aborts.
    abort: Abort,
    /// Whether the child's store runs on the tight budget, writing dirty tiles back to the
    /// recovery file to make room, mid-stroke too.
    tight: bool,
}

/// `saved:strokes:abort:budget`, for the child's environment: `abort` is `stroke` or the
/// flush phase, `budget` is `tight` or `roomy`.
impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let abort = match self.abort {
            Abort::MidStroke => "stroke",
            Abort::MidFlush(phase) => phase,
        };
        let budget = if self.tight { "tight" } else { "roomy" };
        write!(f, "{}:{}:{abort}:{budget}", self.saved, self.strokes)
    }
}

/// Parses [`Scenario`]'s `Display` form.
impl FromStr for Scenario {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = || format!("not a scenario: {s:?}");
        let mut parts = s.split(':');
        let mut next = || parts.next().ok_or_else(bad);
        let saved = next()?.parse().ok().ok_or_else(bad)?;
        let strokes = next()?.parse().ok().ok_or_else(bad)?;
        let abort = match next()? {
            "stroke" => Abort::MidStroke,
            "encoding" => Abort::MidFlush("encoding"),
            "syncing tiles" => Abort::MidFlush("syncing tiles"),
            "syncing save point" => Abort::MidFlush("syncing save point"),
            _ => return Err(bad()),
        };
        let tight = match next()? {
            "tight" => true,
            "roomy" => false,
            _ => return Err(bad()),
        };
        Ok(Self {
            saved,
            strokes,
            abort,
            tight,
        })
    }
}

impl Scenario {
    /// The child's project configuration in `dir`.
    fn config(self, dir: &Path) -> ProjectConfig {
        ProjectConfig {
            store: if self.tight { tight() } else { roomy() },
            recovery_dir: dir.join("recovery"),
        }
    }

    /// Whether the aborted stroke's flush got as far as its save point: then the restore
    /// holds that stroke too.
    ///
    /// A process crash, unlike a power cut, keeps what was written to the page cache, so
    /// whether the save point counts doesn't hang on its sync: it is in the file once the flush
    /// is syncing it, and not before.
    fn keeps_the_last_stroke(self) -> bool {
        self.abort == Abort::MidFlush("syncing save point")
    }

    /// The edits a restore must hold: the saved stroke, every completed stroke, and the
    /// aborted one only if its save point was written.
    fn survivors(self, mesh: &BaseMesh) -> CorrectionMap {
        let mut map = CorrectionMap::new();
        let mut strokes: Vec<(u64, u32)> = (0..self.strokes).map(|k| (seed(k), EDITS)).collect();
        if self.saved {
            strokes.insert(0, (SAVED_SEED, EDITS));
        }
        if self.keeps_the_last_stroke() {
            strokes.push((ABORTED_SEED, ABORTED_EDITS));
        }
        for (seed, count) in strokes {
            record(mesh, &mut map, &random_edits(mesh, TOP, count, seed));
        }
        map
    }
}

/// Polls `project` until its recovery writer has nothing in flight or queued: the strokes so
/// far are durable in the recovery file.
fn poll_until_idle(project: &mut Project) {
    let began = Instant::now();
    loop {
        project.poll().unwrap();
        if project.writer.as_ref().unwrap().is_idle() {
            return;
        }
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the flush finishes"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Applies the `count` edits of the stroke with `seed` to the project's store.
fn edit(project: &mut Project, count: u32, seed: u64) {
    let mesh = project.store().working().mesh();
    let edits = random_edits(&mesh, TOP, count, seed);
    apply(project.store(), &mut CorrectionMap::new(), &edits);
}

/// The crash test's child: runs the scenario in [`SCENARIO`] in the directory in [`DIR`],
/// then aborts. Does nothing when they aren't set.
#[test]
#[ignore = "the crash test's child: the crash tests run it in a process of its own"]
fn crash_child() {
    let (Ok(scenario), Some(dir)) = (std::env::var(SCENARIO), std::env::var_os(DIR)) else {
        return;
    };
    let scenario: Scenario = scenario.parse().unwrap();
    let dir = PathBuf::from(dir);
    let mut project = Project::new((*mesh()).clone(), scenario.config(&dir)).unwrap();
    if scenario.saved {
        edit(&mut project, EDITS, SAVED_SEED);
        project.end_stroke().unwrap();
        project.save_as(&dir.join("p.bcrf")).unwrap();
    }
    for k in 0..scenario.strokes {
        edit(&mut project, EDITS, seed(k));
        project.end_stroke().unwrap();
        poll_until_idle(&mut project);
    }
    edit(&mut project, ABORTED_EDITS, ABORTED_SEED);
    match scenario.abort {
        Abort::MidStroke => {
            if scenario.tight {
                assert!(
                    project.store().working().has_unsaved_tiles(),
                    "the stroke in progress has tiles in the file past its save point"
                );
            }
        }
        Abort::MidFlush(phase) => {
            project.end_stroke().unwrap();
            let writer = project.writer.as_mut().unwrap();
            while writer.phase() != phase {
                assert_ne!(writer.phase(), "idle", "the flush passes {phase}");
                writer.step_blocking(&mut project.store).unwrap();
            }
        }
    }
    std::process::abort();
}

/// Runs `scenario` in a child process, which aborts, then restores what it left and checks it.
fn crash(scenario: Scenario) {
    let dir = TestDir::new();
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#"ulimit -c 0 && exec "$0" "$@""#)
        .arg(std::env::current_exe().unwrap())
        .args([CHILD_TEST, "--exact", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(SCENARIO, scenario.to_string())
        .env(DIR, dir.path(""))
        .output()
        .unwrap();
    assert_eq!(
        output.status.signal(),
        Some(SIGABRT),
        "{scenario:?}: the child aborts ({}); its output:\n{}{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let mut listed = find_recoveries(&dir.path("recovery")).unwrap();
    assert_eq!(
        listed.len(),
        1,
        "{scenario:?}: one recovery file: {listed:?}"
    );
    let candidate = listed.pop().unwrap();
    let path = dir.path("p.bcrf");
    let expected = if scenario.saved {
        CandidateStatus::Restorable
    } else {
        CandidateStatus::NeverSaved
    };
    assert_eq!(candidate.status, expected, "{scenario:?}: {candidate:?}");
    assert_eq!(
        candidate.base_path,
        scenario.saved.then(|| path.clone()),
        "{scenario:?}: the project file"
    );

    let mut project = Project::restore(&candidate, scenario.config(&dir.path(""))).unwrap();
    let mesh = mesh();
    let survivors = scenario.survivors(&mesh);
    let what = format!("{scenario:?}: restored");
    assert_positions(
        project.store(),
        &dense(&mesh, &survivors),
        &tiles(&mesh, 1, TOP),
        &what,
    );
    assert_eq!(
        project.has_unsaved_edits(),
        scenario.strokes > 0 || scenario.keeps_the_last_stroke(),
        "{scenario:?}: the recovered edits, and only they, are unsaved"
    );

    // What was restored saves: into the project file, or a first one.
    if scenario.saved {
        project.save().unwrap();
    } else {
        project.save_as(&path).unwrap();
    }
    project.close(false).unwrap();
    let saved = format!("{scenario:?}: saved after the restore");
    assert_project_file(&path, &survivors, &saved);
    assert_eq!(
        recovery_files(&dir),
        Vec::<String>::new(),
        "{scenario:?}: the recovery file goes with the close"
    );
}

#[test]
fn a_crash_mid_stroke_restores_exactly_the_completed_strokes() {
    for strokes in [0, 1, 3, 8] {
        crash(Scenario {
            saved: true,
            strokes,
            abort: Abort::MidStroke,
            tight: false,
        });
    }
}

#[test]
fn a_never_saved_project_restores_from_its_recovery_file_alone() {
    for strokes in [0, 2, 5] {
        crash(Scenario {
            saved: false,
            strokes,
            abort: Abort::MidStroke,
            tight: false,
        });
    }
}

#[test]
fn a_crash_discards_the_stroke_in_progress_written_back_to_make_room() {
    for saved in [true, false] {
        crash(Scenario {
            saved,
            strokes: 3,
            abort: Abort::MidStroke,
            tight: true,
        });
    }
}

#[test]
fn a_crash_mid_flush_restores_the_previous_or_the_new_stroke_never_a_mix() {
    for phase in ["encoding", "syncing tiles", "syncing save point"] {
        for saved in [true, false] {
            crash(Scenario {
                saved,
                strokes: 2,
                abort: Abort::MidFlush(phase),
                tight: false,
            });
        }
    }
}
