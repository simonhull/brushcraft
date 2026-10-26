//! The crash test, the one this milestone exists for: a real process, killed mid-stroke or
//! mid-flush, and its project restored by the next one.
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
//! never), makes N strokes, waiting after each until its flush is durable, and then gets to
//! where it is to crash: either mid-stroke, after edits with no stroke end, or mid-flush, at a
//! chosen phase of the flush of a stroke that ended. There it prints [`READY`] and blocks for
//! good, and the parent kills it with `SIGKILL`.
//!
//! `SIGKILL` is as abrupt for the files as an abort: the process runs nothing more, no drop and
//! no flush, and keeps only what it wrote to the page cache. Unlike an abort, it leaves no core
//! dump, so a run of the tests doesn't fill the system's crash log (or set off the desktop's
//! crash notifications).
//!
//! The child keeps a spatial index over the finest level, refitted before every stroke ends (and
//! during the interrupted one), so its bounds stream into the recovery file with the edits. A
//! child crashing mid-flush also edits and refits again while the flush is in flight: bounds of
//! a state past the flush's cut, which its save point must not name.
//!
//! The parent checks that the child died of `SIGKILL`, lists the recovery files, restores the
//! one it finds, and checks the evaluation against the dense evaluation of exactly the edits
//! that should have survived: every completed stroke, none of the interrupted one. The index
//! loaded from the restored files must be the one built from the restored tiles, bit for bit,
//! without evaluating a tile. Then it saves the restored project and checks the project file
//! holds them too.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::mpsc;
use std::time::Duration;

use super::*;
use crate::fixture::{record, tight};
use crate::index::SpatialIndex;
use crate::index::tests::assert_loads_as_built;

/// The level the child indexes: the tests' finest.
fn top() -> Level {
    Level::new(TOP).unwrap()
}

/// The variable holding the child's [`Scenario`]; the test binary is a crash test's child only
/// when it is set.
const SCENARIO: &str = "BRUSHCRAFT_CRASH_SCENARIO";

/// The variable holding the directory the child works in: the parent's [`TestDir`].
const DIR: &str = "BRUSHCRAFT_CRASH_DIR";

/// The child test's full path, as libtest names it.
const CHILD_TEST: &str = "project::tests::crash::crash_child";

/// The line the child prints once it is where it is to crash, before it blocks for the kill.
const READY: &str = "brushcraft crash child: ready to be killed";

/// The signal [`std::process::Child::kill`] sends.
const SIGKILL: i32 = 9;

/// How long the parent waits for the child to get where it is to crash.
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

/// Edits per stroke.
const EDITS: u32 = 60;

/// Edits of the stroke the crash interrupts: enough for the tight budget to write some of its
/// tiles back.
const INTERRUPTED_EDITS: u32 = 400;

/// The seed of the stroke made before the save.
const SAVED_SEED: u64 = 7;

/// The seed of the stroke the crash interrupts.
const INTERRUPTED_SEED: u64 = 999;

/// The seed of the edits a child crashing mid-flush makes while the flush is in flight.
const IN_FLIGHT_SEED: u64 = 555;

/// The seed of the `k`th stroke after the save, counting from 0.
fn seed(k: u32) -> u64 {
    100 + u64::from(k)
}

/// Where the child crashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Crash {
    /// Mid-stroke: after the edits of one more stroke, with no stroke end.
    MidStroke,
    /// Mid-flush: one more stroke ends, and the child crashes once its flush is in this phase
    /// (by [`RecoveryWriter`]'s test name of it: `encoding`, `syncing tiles` or
    /// `syncing save point`).
    MidFlush(&'static str),
}

/// What the child does before it crashes.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    /// Whether the project is saved (once, after one stroke) before the strokes.
    saved: bool,
    /// Strokes made after the save (or from the start), each flushed and durable.
    strokes: u32,
    /// Where the child crashes.
    crash: Crash,
    /// Whether the child's store runs on the tight budget, writing dirty tiles back to the
    /// recovery file to make room, mid-stroke too.
    tight: bool,
}

/// `saved:strokes:crash:budget`, for the child's environment: `crash` is `stroke` or the
/// flush phase, `budget` is `tight` or `roomy`.
impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let crash = match self.crash {
            Crash::MidStroke => "stroke",
            Crash::MidFlush(phase) => phase,
        };
        let budget = if self.tight { "tight" } else { "roomy" };
        write!(f, "{}:{}:{crash}:{budget}", self.saved, self.strokes)
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
        let crash = match next()? {
            "stroke" => Crash::MidStroke,
            "encoding" => Crash::MidFlush("encoding"),
            "syncing tiles" => Crash::MidFlush("syncing tiles"),
            "syncing save point" => Crash::MidFlush("syncing save point"),
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
            crash,
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

    /// Whether the interrupted stroke's flush got as far as its save point: then the restore
    /// holds that stroke too.
    ///
    /// A process crash, unlike a power cut, keeps what was written to the page cache, so
    /// whether the save point counts doesn't hang on its sync: it is in the file once the flush
    /// is syncing it, and not before.
    fn keeps_the_last_stroke(self) -> bool {
        self.crash == Crash::MidFlush("syncing save point")
    }

    /// The edits a restore must hold: the saved stroke, every completed stroke, and the
    /// interrupted one only if its save point was written.
    fn survivors(self, mesh: &BaseMesh) -> CorrectionMap {
        let mut map = CorrectionMap::new();
        let mut strokes: Vec<(u64, u32)> = (0..self.strokes).map(|k| (seed(k), EDITS)).collect();
        if self.saved {
            strokes.insert(0, (SAVED_SEED, EDITS));
        }
        if self.keeps_the_last_stroke() {
            strokes.push((INTERRUPTED_SEED, INTERRUPTED_EDITS));
        }
        for (seed, count) in strokes {
            record(mesh, &mut map, &random_edits(mesh, TOP, count, seed));
        }
        map
    }
}

/// Polls `project` until its recovery writer has nothing in flight or queued: the strokes so
/// far are durable in the recovery file.
pub(super) fn poll_until_idle(project: &mut Project) {
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

/// Begins the stroke with `seed`, applies its `count` edits to the project's store, as dabs of
/// its one op, and refits `index` to every tile of its level.
fn edit(project: &mut Project, index: &mut SpatialIndex, count: u32, seed: u64) {
    project.begin_stroke().unwrap();
    edit_and_refit(project.store(), index, count, seed);
}

/// Applies `count` edits with `seed` to `store` as dabs of its current op, and refits `index`
/// to every tile of its level: the bounds recorded are those of the edits' state.
fn edit_and_refit(store: &mut TileStore, index: &mut SpatialIndex, count: u32, seed: u64) {
    let mesh = store.working().mesh();
    let edits = random_edits(&mesh, TOP, count, seed);
    apply_dabs(store, &mut CorrectionMap::new(), &edits);
    index.update(store, &tiles(&mesh, TOP, TOP)).unwrap();
}

/// Blocks the calling thread for good: the child waits there for its kill, running nothing
/// more, as a crashed process would.
fn wait_for_the_kill() -> ! {
    loop {
        std::thread::park();
    }
}

/// The crash test's child: runs the scenario in [`SCENARIO`] in the directory in [`DIR`] up
/// to where it is to crash, prints [`READY`], and waits for the parent to kill it. Does nothing
/// when they aren't set.
#[test]
#[ignore = "the crash test's child: the crash tests run it in a process of its own"]
fn crash_child() {
    let (Ok(scenario), Some(dir)) = (std::env::var(SCENARIO), std::env::var_os(DIR)) else {
        return;
    };
    let scenario: Scenario = scenario.parse().unwrap();
    let dir = PathBuf::from(dir);
    let mut project = Project::new((*mesh()).clone(), scenario.config(&dir)).unwrap();
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    if scenario.saved {
        edit(&mut project, &mut index, EDITS, SAVED_SEED);
        project.end_stroke().unwrap();
        assert_saved(project.save_as(&dir.join("p.bcrf")));
    }
    for k in 0..scenario.strokes {
        edit(&mut project, &mut index, EDITS, seed(k));
        project.end_stroke().unwrap();
        poll_until_idle(&mut project);
    }
    edit(
        &mut project,
        &mut index,
        INTERRUPTED_EDITS,
        INTERRUPTED_SEED,
    );
    match scenario.crash {
        Crash::MidStroke => {
            if scenario.tight {
                assert!(
                    project.store().working().has_unsaved_records(),
                    "the stroke in progress has tiles in the file past its save point"
                );
            }
        }
        Crash::MidFlush(phase) => {
            project.end_stroke().unwrap();
            // More edits and a refit while the flush is in flight, straight on the store (a
            // project call would advance the flush).
            project.store.begin_op().unwrap();
            edit_and_refit(&mut project.store, &mut index, EDITS, IN_FLIGHT_SEED);
            let writer = project.writer.as_mut().unwrap();
            while writer.phase() != phase {
                assert_ne!(writer.phase(), "idle", "the flush passes {phase}");
                writer.step_blocking(&mut project.store).unwrap();
            }
        }
    }
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{READY}").unwrap();
    stdout.flush().unwrap();
    wait_for_the_kill();
}

/// Asserts that the process `pid`, a killed child, left no test directory of its own: it works
/// in the parent's, as a killed process runs no drops and anything of its own would stay behind.
fn assert_no_directory_of(pid: u32, scenario: Scenario) {
    let own = format!("brushcraft-store-{pid}-");
    let left: Vec<String> = fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name.starts_with(&own))
        .collect();
    assert_eq!(
        left,
        Vec::<String>::new(),
        "{scenario:?}: the child left directories"
    );
}

/// Runs `scenario` in a child process up to where it is to crash, kills it there with
/// `SIGKILL`, then restores what it left and checks it.
fn crash(scenario: Scenario) {
    let dir = TestDir::new();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([CHILD_TEST, "--exact", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(SCENARIO, scenario.to_string())
        .env(DIR, dir.path(""))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Both pipes are read on threads of their own, so neither fills up and stalls the child,
    // and a child that never gets there fails the test instead of hanging it.
    let stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let (ready, ready_seen) = mpsc::channel();
    let out = std::thread::spawn(move || {
        let mut lines = Vec::new();
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            // After libtest's `test <name> ... `, on the same line.
            if line.ends_with(READY) {
                // The parent may have stopped waiting: then nothing listens.
                ready.send(()).unwrap_or_default();
            }
            lines.push(line);
        }
        lines.join("\n")
    });
    let err = std::thread::spawn(move || {
        let mut text = String::new();
        drop(stderr.read_to_string(&mut text));
        text
    });
    let got_there = ready_seen.recv_timeout(CHILD_TIMEOUT);
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert_no_directory_of(child.id(), scenario);
    let output = format!("{}\n{}", out.join().unwrap(), err.join().unwrap());
    assert!(
        got_there.is_ok(),
        "{scenario:?}: the child never got where it was to crash ({status}); its output:\n\
         {output}"
    );
    assert_eq!(
        status.signal(),
        Some(SIGKILL),
        "{scenario:?}: the child is killed ({status}); its output:\n{output}"
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
    let loaded = assert_loads_as_built(project.store(), top(), &what);
    // A project never saved, whose first flush never finished, has no save point to hold any.
    let durable = scenario.saved || scenario.strokes > 0 || scenario.keeps_the_last_stroke();
    assert_eq!(
        loaded.evaluated == 0,
        durable,
        "{scenario:?}: every bound restored with its tiles, once a save point holds them \
         ({loaded:?})"
    );

    // What was restored saves: into the project file, or a first one.
    if scenario.saved {
        assert_saved(project.save());
    } else {
        assert_saved(project.save_as(&path));
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
            crash: Crash::MidStroke,
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
            crash: Crash::MidStroke,
            tight: false,
        });
    }
}

/// Under a tight budget, the stroke in progress has tiles written back to the file past its
/// save point, and a crash discards them, whether or not the project was ever saved.
///
/// The two cases (each a child process and a directory of its own) run side by side: under
/// the tight budget each takes several seconds.
#[test]
fn a_crash_discards_the_stroke_in_progress_written_back_to_make_room() {
    std::thread::scope(|scope| {
        for saved in [true, false] {
            scope.spawn(move || {
                crash(Scenario {
                    saved,
                    strokes: 3,
                    crash: Crash::MidStroke,
                    tight: true,
                });
            });
        }
    });
}

#[test]
fn a_crash_mid_flush_restores_the_previous_or_the_new_stroke_never_a_mix() {
    for phase in ["encoding", "syncing tiles", "syncing save point"] {
        for saved in [true, false] {
            crash(Scenario {
                saved,
                strokes: 2,
                crash: Crash::MidFlush(phase),
                tight: false,
            });
        }
    }
}
