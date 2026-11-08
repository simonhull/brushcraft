//! The crash test, the one this milestone exists for: a real process, killed mid-stroke or
//! mid-flush, and its project restored by the next one.
//!
//! # How it runs
//!
//! Each case re-executes the test binary ([`std::env::current_exe`]) as a child, asking it to
//! run only `crash_child` (an ignored test, so a normal run skips it): `--exact` with the
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
//! Before each stroke the child records the whole display layer. Some children write every
//! display record through to the recovery file as they record it, between flushes, instead of
//! holding it for the next one, and one crashing mid-stroke records the layer again after the
//! stroke's edits: a crash then leaves records past the last save point, built from a state no
//! save point holds, which the restore must discard, and records the save points name, which
//! must be current or unknown.
//!
//! The parent checks that the child died of `SIGKILL`, lists the recovery files, restores the
//! one it finds, and checks the evaluation against the dense evaluation of exactly the edits
//! that should have survived: every completed stroke, none of the interrupted one, each
//! completed stroke's record with its edits (every stroke ends with one). The index
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
use crate::fixture::{dab_at, display_layer, record, tight};
use crate::index::SpatialIndex;
use crate::index::tests::assert_loads_as_built;
use crate::journal::{DAB_BYTES, DabBatch};
use crate::stroke_record::tests::wobbly;
use crate::{FLUSH_INTERVAL, OpId, StrokeRecord};

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
    /// Whether the child's store holds no display record waiting for the recovery file, writing
    /// each through as it records it (between flushes, past the last save point), rather than
    /// keeping them for the next flush.
    through: bool,
}

/// `saved:strokes:crash:budget:display`, for the child's environment: `crash` is `stroke` or
/// the flush phase, `budget` is `tight` or `roomy`, `display` is `through` or `held`.
impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let crash = match self.crash {
            Crash::MidStroke => "stroke",
            Crash::MidFlush(phase) => phase,
        };
        let budget = if self.tight { "tight" } else { "roomy" };
        let display = if self.through { "through" } else { "held" };
        write!(
            f,
            "{}:{}:{crash}:{budget}:{display}",
            self.saved, self.strokes
        )
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
        let through = match next()? {
            "through" => true,
            "held" => false,
            _ => return Err(bad()),
        };
        Ok(Self {
            saved,
            strokes,
            crash,
            tight,
            through,
        })
    }
}

impl Scenario {
    /// The child's project configuration in `dir`.
    ///
    /// It keeps a display layer, its floor's share added to the budget.
    fn config(self, dir: &Path) -> ProjectConfig {
        let store =
            if self.tight { tight() } else { roomy() }.with_display(crate::fixture::level(TOP));
        ProjectConfig {
            store: store.with_budget_bytes(store.budget_bytes + store.floor_bytes(&mesh())),
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

    /// The seeds of the strokes whose records the recovery file must hold: every completed
    /// stroke since the save, and the interrupted one only if its save point was written (its
    /// record is appended with the flush its end starts, before that save point).
    fn recorded_since_save(self) -> Vec<u64> {
        let mut seeds: Vec<u64> = (0..self.strokes).map(seed).collect();
        if self.keeps_the_last_stroke() {
            seeds.push(INTERRUPTED_SEED);
        }
        seeds
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

/// The record a child leaves for the stroke with `seed`, made as `op`.
fn record_of(op: OpId, seed: u64) -> StrokeRecord {
    wobbly(op, 40, seed)
}

/// Ends the project's stroke, the one with `seed`, with its record.
fn end_recorded(project: &mut Project, seed: u64) {
    let op = project.store().current_op().unwrap();
    project.end_stroke(&record_of(op, seed)).unwrap();
}

/// Asserts that `file` holds the records of the strokes with `seeds`, in order, each with its
/// own op, the ops in the order the strokes were made.
fn assert_records(file: &ProjectFile, seeds: &[u64], what: &str) {
    let read: Vec<StrokeRecord> = file.strokes().map(Result::unwrap).collect();
    let expected: Vec<StrokeRecord> = read
        .iter()
        .zip(seeds)
        .map(|(got, &seed)| record_of(got.op(), seed))
        .collect();
    assert_eq!(read.len(), seeds.len(), "{what}: one record a stroke");
    assert_eq!(read, expected, "{what}: the records");
    assert!(
        read.is_sorted_by(|a, b| a.op() < b.op()),
        "{what}: in the order of their ops"
    );
}

/// Begins the stroke with `seed`, applies its `count` edits to the project's store, as dabs of
/// its one op, and refits `index` to every tile of its level.
fn edit(project: &mut Project, index: &mut SpatialIndex, count: u32, seed: u64) {
    project.begin_stroke_and_wait().unwrap();
    edit_and_refit(project.store(), index, count, seed);
}

/// Records the whole display layer, applies `count` edits with `seed` to `store` as dabs of its
/// current op, and refits `index` to every tile of its level.
///
/// The bounds recorded are those of the edits' state; the display tiles, those of the state
/// before them, which the edits must make unknown wherever they change them, or a restore would
/// show them stale.
fn edit_and_refit(store: &mut TileStore, index: &mut SpatialIndex, count: u32, seed: u64) {
    let mesh = store.working().mesh();
    let layer = display_layer(store);
    store.record_display(&layer).unwrap();
    let edits = random_edits(&mesh, TOP, count, seed);
    apply_dabs(store, &mut CorrectionMap::new(), &edits);
    index.update(store, &tiles(&mesh, TOP, TOP)).unwrap();
}

/// Asserts what [`assert_loads_as_built`] does of `store`'s index over [`top`], and that it holds
/// no display tile older (or newer) than its corrections ([`assert_display_current`]); returns
/// the index load's stats.
fn assert_restored_derived(store: &mut TileStore, what: &str) -> LoadStats {
    let loaded = assert_loads_as_built(store, top(), what);
    assert_display_current(store, what);
    loaded
}

/// Asserts that `store` holds no display tile older (or newer) than its corrections: each is
/// unknown, or the one its tiles build, bit for bit.
fn assert_display_current(store: &mut TileStore, what: &str) {
    let built = display_layer(store);
    let keys: Vec<TileKey> = built.iter().map(|t| *t.key()).collect();
    let stored = store.stored_display(&keys).unwrap();
    for (tile, kept) in built.iter().zip(stored) {
        if let Some(kept) = kept {
            assert_eq!(&kept, tile, "{what}: display tile {:?}", tile.key());
        }
    }
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
    if scenario.through {
        project.store.hold_display_waiting_to(0);
    }
    let (mut index, _) = project.load_index(top(), |_, _| {}).unwrap();
    if scenario.saved {
        edit(&mut project, &mut index, EDITS, SAVED_SEED);
        end_recorded(&mut project, SAVED_SEED);
        assert_saved(project.save_as(&dir.join("p.bcrf")));
    }
    for k in 0..scenario.strokes {
        edit(&mut project, &mut index, EDITS, seed(k));
        end_recorded(&mut project, seed(k));
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
            if scenario.through {
                // The layer again, of the interrupted stroke's state, which no save point
                // holds: its records, written through, must go with the tail.
                let layer = display_layer(project.store());
                project.store().record_display(&layer).unwrap();
                assert_eq!(
                    project.store().display_waiting().0,
                    0,
                    "every display record written through before the crash"
                );
                assert!(
                    project.store().working().has_unsaved_records(),
                    "records past the last save point"
                );
            }
        }
        Crash::MidFlush(phase) => {
            end_recorded(&mut project, INTERRUPTED_SEED);
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

/// Asserts that the process `pid`, a killed child, left no test directory of its own.
///
/// It works in the parent's, as a killed process runs no drops and anything of its own would
/// stay behind. `what` names the case.
fn assert_no_directory_of(pid: u32, what: &str) {
    let own = format!("brushcraft-store-{pid}-");
    let left: Vec<String> = fs::read_dir(std::env::temp_dir())
        .unwrap()
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name.starts_with(&own))
        .collect();
    assert_eq!(
        left,
        Vec::<String>::new(),
        "{what}: the child left directories"
    );
}

/// Runs a child test until it prints [`READY`], then kills it `delay` later.
///
/// The child is `child_test`, by its full path, in a process of its own working in `dir`, with
/// `scenario` in `variable`. It is killed with `SIGKILL`, and the call checks that it died of it
/// and left nothing of its own. `what` names the case.
fn run_and_kill(
    child_test: &str,
    (variable, scenario): (&str, &str),
    dir: &TestDir,
    delay: Duration,
    what: &str,
) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([child_test, "--exact", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(variable, scenario)
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
    if got_there.is_ok() {
        std::thread::sleep(delay);
    }
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert_no_directory_of(child.id(), what);
    let output = format!("{}\n{}", out.join().unwrap(), err.join().unwrap());
    assert!(
        got_there.is_ok(),
        "{what}: the child never got where it was to crash ({status}); its output:\n{output}"
    );
    assert_eq!(
        status.signal(),
        Some(SIGKILL),
        "{what}: the child is killed ({status}); its output:\n{output}"
    );
}

/// Runs `scenario` in a child process up to where it is to crash, kills it there with
/// `SIGKILL`, then restores what it left and checks it.
fn crash(scenario: Scenario) {
    let dir = TestDir::new();
    run_and_kill(
        CHILD_TEST,
        (SCENARIO, &scenario.to_string()),
        &dir,
        Duration::ZERO,
        &format!("{scenario:?}"),
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
    let loaded = assert_restored_derived(project.store(), &what);
    assert_records(
        project.store().working(),
        &scenario.recorded_since_save(),
        &what,
    );
    // A project never saved, whose first flush never finished, has no save point to hold any.
    let durable = scenario.saved || scenario.strokes > 0 || scenario.keeps_the_last_stroke();
    assert_eq!(
        loaded.evaluated == 0,
        durable,
        "{scenario:?}: every bound restored with its tiles, once a save point holds them \
         ({loaded:?})"
    );

    save_restored(project, scenario, &dir, &survivors);
}

/// Saves `project`, restored after `scenario`'s crash in `dir`, and closes it.
///
/// It saves into its project file, or a first one. The project file must hold `survivors`, the
/// edits restored, with every stroke's record, and the recovery file go with the close.
fn save_restored(
    mut project: Project,
    scenario: Scenario,
    dir: &TestDir,
    survivors: &CorrectionMap,
) {
    let path = dir.path("p.bcrf");
    if scenario.saved {
        assert_saved(project.save());
    } else {
        assert_saved(project.save_as(&path));
    }
    project.close(false).unwrap();
    let saved = format!("{scenario:?}: saved after the restore");
    assert_project_file(&path, survivors, &saved);
    let mut seeds = scenario.recorded_since_save();
    if scenario.saved {
        seeds.insert(0, SAVED_SEED);
    }
    assert_records(&ProjectFile::open(&path).unwrap(), &seeds, &saved);
    assert_eq!(
        recovery_files(dir),
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
            through: false,
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
            through: false,
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
                    through: false,
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
                through: false,
            });
        }
    }
}

/// A crash after display records were written through restores a layer no older than its tiles.
///
/// Every display record goes to the recovery file as it is recorded, between flushes. A crash
/// mid-stroke (the layer recorded before the stroke's edits and again after them, of a state no
/// save point holds, before the stroke's save point) or mid-flush restores a layer no older or
/// newer than its tiles: the records past the last save point go with the rest of the tail, and
/// those a save point names are current or unknown.
#[test]
fn a_crash_after_display_records_written_through_restores_a_current_layer() {
    for (saved, strokes) in [(true, 0), (true, 3), (false, 2)] {
        crash(Scenario {
            saved,
            strokes,
            crash: Crash::MidStroke,
            tight: false,
            through: true,
        });
    }
    for phase in ["encoding", "syncing save point"] {
        crash(Scenario {
            saved: true,
            strokes: 2,
            crash: Crash::MidFlush(phase),
            tight: false,
            through: true,
        });
    }
}

/// The variable that makes the test binary the journal crash test's child; its value is unread.
const JOURNAL: &str = "BRUSHCRAFT_CRASH_JOURNAL";

/// The journal crash test's child, by its full path.
const JOURNAL_CHILD_TEST: &str = "project::tests::crash::journal_child";

/// The dabs the journal child places a step.
const PLACE_EACH: u32 = 8;

/// The most dabs the journal child places: once there, it waits for the kill.
const JOURNAL_DABS: u32 = 400_000;

/// The dab format the journal child's app sets.
const JOURNAL_FORMAT: u32 = 1;

/// The seed of the journal child's edits, those of its first step; each step's is the next.
const JOURNAL_SEED: u64 = 4_000;

/// The edits standing for the work of the journal child's dabs of step `step`, one a dab.
fn step_edits(mesh: &BaseMesh, step: u32) -> Vec<crate::fixture::Edit> {
    random_edits(mesh, TOP, PLACE_EACH, JOURNAL_SEED + u64::from(step))
}

/// How many dabs the journal child has landed once it has placed `placed`: a backlog of five to
/// eleven dabs, varying with the step, never going back.
fn landed_by(placed: u32, before: u32) -> u32 {
    let backlog = 5 + (placed / PLACE_EACH) % 7;
    placed.saturating_sub(backlog).max(before)
}

/// The journal crash test's child: one long stroke, journalled and landed until it is killed.
///
/// Each step journals [`PLACE_EACH`] dabs and lands dabs some steps behind, as edits, then ticks
/// a clock that runs a flush interval a step, so every tick is due. It prints [`READY`] once a
/// flush is durable and goes on, until it has placed [`JOURNAL_DABS`]; then it waits for the
/// kill. Does nothing unless [`JOURNAL`] is set.
#[test]
#[ignore = "the journal crash test's child: the test runs it in a process of its own"]
fn journal_child() {
    let (Some(_), Some(dir)) = (std::env::var_os(JOURNAL), std::env::var_os(DIR)) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let config = ProjectConfig {
        store: roomy(),
        recovery_dir: dir.join("recovery"),
    };
    let mut project = Project::new((*mesh()).clone(), config).unwrap();
    project.set_dab_format(JOURNAL_FORMAT);
    let mesh = project.store().working().mesh();
    project.begin_stroke_and_wait().unwrap();
    let start = Instant::now();
    let mut ready = false;
    let (mut placed, mut landed) = (0, 0);
    let mut edits = Vec::new();
    for step in 0..JOURNAL_DABS / PLACE_EACH {
        let dabs: Vec<[u8; DAB_BYTES]> = (placed..placed + PLACE_EACH).map(dab_at).collect();
        project.journal(&dabs).unwrap();
        placed += PLACE_EACH;
        edits.extend(step_edits(&mesh, step));
        let target = landed_by(placed, landed);
        apply_dabs(
            project.store(),
            &mut CorrectionMap::new(),
            &edits[landed as usize..target as usize],
        );
        project.journal_landed(target).unwrap();
        landed = target;
        project.tick(start + FLUSH_INTERVAL * (step + 1)).unwrap();
        if !ready && project.writer.as_ref().unwrap().flushes() > 0 {
            ready = true;
            let mut stdout = io::stdout().lock();
            writeln!(stdout, "{READY}").unwrap();
            stdout.flush().unwrap();
        }
    }
    wait_for_the_kill();
}

/// A delay up to `max`, from the clock's nanoseconds: a different moment every run.
fn random_delay(max: Duration) -> Duration {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    max.mul_f64(f64::from(nanos) / 1e9)
}

/// The journal crash test's runs: kills at as many moments.
const JOURNAL_RUNS: u32 = 8;

/// The longest the journal crash test waits after the child's first save point to kill it: well
/// within the time the child takes to place [`JOURNAL_DABS`], even on a loaded machine.
const JOURNAL_KILL_WITHIN: Duration = Duration::from_millis(300);

#[test]
fn a_kill_keeps_every_dab_journalled_by_the_last_save_point() {
    for run in 0..JOURNAL_RUNS {
        let dir = TestDir::new();
        let delay = random_delay(JOURNAL_KILL_WITHIN);
        let what = format!("run {run}, killed {delay:?} after its first save point");
        run_and_kill(JOURNAL_CHILD_TEST, (JOURNAL, "1"), &dir, delay, &what);
        let candidate = find_recoveries(&dir.path("recovery"))
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(candidate.status, CandidateStatus::NeverSaved, "{what}");
        let mut project = Project::restore(&candidate, config(&dir)).unwrap();
        let batches: Vec<DabBatch> = project
            .store()
            .working()
            .batches()
            .map(Result::unwrap)
            .collect();
        let (first, last) = (batches.first().unwrap(), batches.last().unwrap());
        assert!(
            last.placed() < JOURNAL_DABS && last.landed() < last.placed(),
            "{what}: killed mid-stroke, dabs still landing ({last:?})"
        );
        assert_eq!(
            first.first(),
            first.landed(),
            "{what}: from its landed count"
        );
        for pair in batches.windows(2) {
            let (before, after) = (&pair[0], &pair[1]);
            assert_eq!(after.op(), first.op(), "{what}: one stroke");
            assert!(
                after.landed() >= before.landed(),
                "{what}: dabs never unland"
            );
            assert_eq!(
                after.first(),
                before.placed().max(after.landed()),
                "{what}: no gap or overlap among the dabs not landed"
            );
        }
        for batch in &batches {
            let expected: Vec<[u8; DAB_BYTES]> =
                (batch.first()..batch.placed()).map(dab_at).collect();
            assert_eq!(batch.dabs(), expected, "{what}: the dabs as placed");
        }
        // The tiles the last save point names are those of exactly the dabs its last batch
        // counts landed.
        let mesh = mesh();
        let mut map = CorrectionMap::new();
        let steps = last.landed().div_ceil(PLACE_EACH);
        let edits: Vec<crate::fixture::Edit> = (0..steps)
            .flat_map(|step| step_edits(&mesh, step))
            .collect();
        record(&mesh, &mut map, &edits[..last.landed() as usize]);
        assert_positions(
            project.store(),
            &dense(&mesh, &map),
            &tiles(&mesh, 1, TOP),
            &what,
        );
        project.close(false).unwrap();
    }
}

/// The edits of the journal child's first `count` dabs: those of its steps, in order.
fn journal_edits(mesh: &BaseMesh, count: u32) -> Vec<crate::fixture::Edit> {
    let steps = count.div_ceil(PLACE_EACH);
    let mut edits: Vec<crate::fixture::Edit> =
        (0..steps).flat_map(|step| step_edits(mesh, step)).collect();
    edits.truncate(count as usize);
    edits
}

/// The resuming crash test's runs: kills at as many moments.
const RESUME_RUNS: u32 = 20;

/// The longest the resuming crash test waits after the child's first save point to kill it.
const RESUME_KILL_WITHIN: Duration = Duration::from_millis(150);

/// A restore after a kill mid-landing continues the stroke's tiles, rows and journal exactly.
///
/// Killed at random moments of a long stroke's landing, a restore gives them as the last save
/// point left them, and landing the rows not landed gives the tiles of every dab placed by it.
///
/// The fixture's dabs are store edits that each set one vertex's correction outright, so landing
/// them from the restored tiles gives the same tiles as landing them all live: this proves the
/// store's part exact, not the app's stroke, which lands within half a quantization step of the
/// live result (see `crate::project::resume`, "What is exact").
#[test]
fn restoring_and_landing_the_rest_gives_the_tiles_of_every_dab_placed() {
    let mesh = mesh();
    for run in 0..RESUME_RUNS {
        let dir = TestDir::new();
        let delay = random_delay(RESUME_KILL_WITHIN);
        let what = format!("run {run}, killed {delay:?} after its first save point");
        run_and_kill(JOURNAL_CHILD_TEST, (JOURNAL, "1"), &dir, delay, &what);
        let candidate = find_recoveries(&dir.path("recovery"))
            .unwrap()
            .pop()
            .unwrap();
        let mut project = Project::restore(&candidate, config(&dir)).unwrap();
        let open = project.take_open_stroke().unwrap().unwrap();
        assert_eq!(open.dab_format(), JOURNAL_FORMAT, "{what}: its format");
        let (landed, rest) = (open.landed(), u32::try_from(open.dabs().len()).unwrap());
        assert!(rest > 0, "{what}: dabs to land");
        let placed = landed + rest;
        let journalled: Vec<[u8; DAB_BYTES]> = (landed..placed).map(dab_at).collect();
        assert_eq!(open.dabs(), journalled, "{what}: the dabs not landed");
        let edits = journal_edits(&mesh, placed);
        project.resume_stroke(open).unwrap();
        assert!(project.may_land(), "{what}: it lands at once");
        // The rest lands as the live landing would have, a few dabs at a time.
        let mut done = landed;
        while done < placed {
            let next = (done + 3).min(placed);
            apply_dabs(
                project.store(),
                &mut CorrectionMap::new(),
                &edits[done as usize..next as usize],
            );
            project.journal_landed(next).unwrap();
            done = next;
        }
        project.end_unrecorded_stroke().unwrap();
        // Every dab placed by the last save point, landed.
        let mut placed_all = CorrectionMap::new();
        record(&mesh, &mut placed_all, &edits);
        assert_positions(
            project.store(),
            &dense(&mesh, &placed_all),
            &tiles(&mesh, 1, TOP),
            &what,
        );
        project.close(false).unwrap();
    }
}
