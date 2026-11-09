//! A stroke a restore found still landing, taken on by the brush (the dab journal spec,
//! "Restoring a journalled stroke").
//!
//! Its dabs not landed are shown at once and land as a stroke whose pen has lifted, its journal
//! counting on from what had landed; it ends unrecorded; a stroke begun meanwhile waits behind
//! it; and rows of another layout aren't taken on, the restore note saying so.

use store::find_recoveries;

use super::*;
use crate::OpenNote;
use crate::dialogs::UNFINISHED_STROKE;

/// What a crash left in a recovery file: a stroke drawn `Down` to `Up`, every dab shown and some
/// landed by the last cut.
struct Crashed {
    /// The project's directory, its recovery file in `recovery`.
    dir: TestDir,
    /// The stroke's op.
    op: OpId,
    /// Every dab the stroke placed, in order.
    placed: Vec<DabRow>,
    /// How many had landed at the last cut.
    landed: usize,
}

impl Crashed {
    /// The dabs not landed at the last cut: what a restore finds still to land.
    fn waiting(&self) -> &[DabRow] {
        self.placed.get(self.landed..).unwrap_or_default()
    }

    /// [`resumed_in`] the crash's directory.
    fn resumed(
        &self,
        brush: impl FnOnce(&mut Project) -> Sculpting,
    ) -> Owner<Sculpting, NoUploads> {
        resumed_in(&self.dir, brush)
    }
}

/// The project restored from the one recovery file a crash left in `dir`, told the brush's dab
/// format as the app tells every project it opens.
fn restore_in(dir: &TestDir) -> Project {
    let found = find_recoveries(&dir.path().join("recovery")).expect("listed");
    let [candidate] = found.as_slice() else {
        panic!("one recovery file: {found:?}");
    };
    let mut project = Project::restore(candidate, config(dir)).expect("restored");
    project.set_dab_format(stroke::DAB_FORMAT);
    project
}

/// The frame loop over the project restored in `dir` with the brush `brush` makes, the stroke a
/// restore found handed to it as the app does.
fn resumed_in(
    dir: &TestDir,
    brush: impl FnOnce(&mut Project) -> Sculpting,
) -> Owner<Sculpting, NoUploads> {
    let mut owner = owner_of(restore_in(dir), brush, setup());
    owner.resume_stroke();
    owner
}

/// A crash with a stroke landing, in rows of `format`, `landed` of its dabs landed.
///
/// A stroke `Down` to `Up` on a fresh project whose app journals rows of `format`: every dab
/// shown, `landed` of them landed, the journal cut, and the project dropped unclosed, its recovery
/// file holding what was durable.
fn crash_mid_landing(format: u32, landed: usize) -> Crashed {
    let mut run = Journalling::new();
    run.project.set_dab_format(format);
    let shown = run.frame(&stroke(ACROSS, 12));
    assert!(shown.dabs_shown > landed + 2, "{shown:?}");
    assert!(!run.stroke().open(), "the pen lifted");
    let op = run.stroke().op;
    let placed = run.queued_rows();
    run.cut();
    for _ in 0..landed {
        assert_eq!(run.land_one().dabs_applied, 1, "a dab at a time");
    }
    run.cut();
    let Journalling {
        dir,
        project,
        brush,
        picker,
        ..
    } = run;
    drop((brush, picker));
    // A crash: nothing closes the project.
    drop(project);
    Crashed {
        dir,
        op,
        placed,
        landed,
    }
}

/// The rows of the dabs `owner`'s brush still has to land, in order.
fn rows_left(owner: &Owner<Sculpting, NoUploads>) -> Vec<DabRow> {
    let stroke = owner.brush().stroke.as_ref().expect("a stroke under way");
    let rows = stroke.dabs.iter().copied().map(row_of);
    rows.collect()
}

#[test]
fn a_resumed_stroke_is_shown_at_once_and_lands_as_a_lifted_stroke() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT, 3);
    let mut owner = crashed.resumed(Sculpting::new);
    {
        let brush = owner.brush();
        let resumed = brush.stroke.as_ref().expect("the stroke is taken on");
        assert_eq!(resumed.op, crashed.op, "as its own op");
        assert_eq!(resumed.landed, 3, "its landed count goes on");
        assert!(!resumed.open(), "its pen has lifted");
        assert_eq!(
            brush.catching_up(),
            crashed.waiting().len(),
            "every dab not landed is shown at once, before any frame"
        );
    }
    assert_eq!(rows_left(&owner), crashed.waiting(), "the rows as placed");
    // The first frame reports them shown: it draws them.
    assert_eq!(run(&mut owner, 1), 1);
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it landed and ended");
    assert_eq!(owner.brush().error(), None);
    cut(owner.project(), &mut Instant::now());
    let path = owner
        .project()
        .recovery_path()
        .expect("a recovery file")
        .to_owned();
    owner.close().expect("closed");
    let batches = batches_at(&path, crashed.op);
    let total = u32::try_from(crashed.placed.len()).expect("a few");
    let last = batches.last().expect("batches");
    assert_eq!(
        (last.placed(), last.landed()),
        (total, total),
        "every dab landed, its journal counting on from the restored count"
    );
    let counts: Vec<u32> = batches.iter().map(DabBatch::landed).collect();
    assert!(counts.is_sorted(), "never counted back: {counts:?}");
    drop(crashed);
}

#[test]
fn a_resumed_stroke_ends_its_op_unrecorded_once_landed() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT, 2);
    let mut owner = crashed.resumed(Sculpting::new);
    assert!(owner.brush().under_way(), "taken on");
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it landed and ended");
    assert!(owner.project().has_unsaved_edits(), "its edits stand");
    let records = saved_strokes(owner, &crashed.dir, "resumed.bcrf");
    assert!(records.is_empty(), "no record: {records:?}");
}

#[test]
fn a_down_during_a_resumed_landing_waits_as_behind_any_landing() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT, 1);
    // A dab a step: the restored stroke is still landing when the next stroke comes.
    let mut owner =
        crashed.resumed(|project| Sculpting::new(project).with_cost(DabCost::one_at_a_time()));
    queue(&mut owner, below());
    assert_eq!(run(&mut owner, 1), 1);
    let brush = owner.brush();
    assert!(!brush.deferred.is_empty(), "the new stroke waits to begin");
    let landing = brush.stroke.as_ref().expect("still landing");
    assert_eq!(landing.op, crashed.op, "behind the restored stroke");
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "both over");
    assert_eq!(owner.brush().error(), None);
    let records = saved_strokes(owner, &crashed.dir, "both.bcrf");
    let [record] = records.as_slice() else {
        panic!("the new stroke's record alone: {records:?}");
    };
    assert_ne!(
        record.op(),
        crashed.op,
        "an op of its own, after the restored one"
    );
}

#[test]
fn rows_of_another_dab_format_are_not_resumed_and_the_note_says_so() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT + 1, 2);
    let mut owner = crashed.resumed(Sculpting::new);
    assert!(!owner.brush().stroking(), "not taken on");
    let error = owner.brush().error().expect("said").to_owned();
    assert!(error.contains("layout"), "for its layout: {error}");
    let note = OpenNote::Restored {
        path: "recovery.bcrf".into(),
        when: std::time::SystemTime::now(),
    };
    owner.note(&note, Instant::now());
    let bar = owner.bar().expect("the note");
    assert_eq!(bar.title(), "Your unsaved work is back");
    let last = bar.details().last().map(String::as_str);
    assert_eq!(last, Some(UNFINISHED_STROKE), "under the note's time");
    assert_eq!(bar.details().len(), 2, "{:?}", bar.details());
    // The next stroke closes it in the file, and goes on as ever.
    queue(&mut owner, below());
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "the new stroke is over");
    assert_eq!(owner.brush().error(), None);
    let records = saved_strokes(owner, &crashed.dir, "other.bcrf");
    assert_eq!(records.len(), 1, "the new stroke's record");
}

/// The variable holding the directory a killed child works in: the parent's [`TestDir`]. The test
/// binary is a child only when it and [`KILL_LANDING`] are set.
const KILL_DIR: &str = "BRUSHCRAFT_RESUME_KILL_DIR";

/// The variable saying where the child is killed: `cut,more`, the dabs it lands before its last
/// cut, and those it lands after.
const KILL_LANDING: &str = "BRUSHCRAFT_RESUME_KILL_LANDING";

/// The child test's full path, as libtest names it.
const CHILD_TEST: &str = "sculpting::tests::journal::resume::killed_mid_landing_child";

/// The line the child prints once it is where it is to be killed, before it blocks for good.
const READY: &str = "brushcraft resume child: ready to be killed";

/// How long the parent waits for the child to get there.
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

/// The signal [`std::process::Child::kill`] sends.
const SIGKILL: i32 = 9;

/// A pen's sample period, at the scripted pen's rate.
fn pen_period() -> Duration {
    Duration::from_secs_f64(crate::script::PEN_RATE.recip())
}

/// A stroke along [`ACROSS`] in 24 drags, `Down` to `Up`, its samples taken a pen's period apart
/// from a period after `epoch`: the same times from the brush's epoch in any process.
fn stroke_from(epoch: Instant) -> Vec<PenSample> {
    let mut samples = stroke(ACROSS, 24);
    let times = (1..).map(|k| epoch + pen_period() * k);
    let paired = samples.iter_mut().zip(times);
    for (sample, at) in paired {
        sample.timestamp = at;
        sample.received = at;
    }
    samples
}

/// What a landing starts from, the same in every process, over a fresh project in `dir`.
///
/// A small brush's stroke `Down` to `Up`, shown and none of it landed, its journal cut. Returns
/// the run and how many dabs the stroke placed.
fn landing_in(dir: TestDir) -> (Journalling, usize) {
    let epoch = Instant::now()
        .checked_sub(pen_period() * 64)
        .expect("after the clock's start");
    // A small brush, so its dabs come close together: a long landing to kill midway.
    let mut run = Journalling::in_dir(dir, |mut brush| {
        brush.set_radius(12.0);
        brush.with_epoch(epoch)
    });
    run.project.set_dab_format(stroke::DAB_FORMAT);
    run.frame(&stroke_from(epoch));
    assert!(!run.stroke().open(), "the pen lifted");
    // Every dab the stroke journalled: none has landed.
    let placed = run.queued_rows().len();
    run.cut();
    (run, placed)
}

/// The child: lands its stroke dab by dab as [`KILL_LANDING`] says, the journal cut between, then
/// prints [`READY`] and blocks until the parent kills it. Without its variables it does nothing.
#[test]
#[ignore = "a_stroke_killed_mid_landing_comes_back_whole runs it as its child"]
fn killed_mid_landing_child() {
    let (Some(dir), Ok(landing)) = (std::env::var_os(KILL_DIR), std::env::var(KILL_LANDING)) else {
        return;
    };
    let (cut, more) = landing.split_once(',').expect("cut,more");
    let cut: usize = cut.parse().expect("a count");
    let more: usize = more.parse().expect("a count");
    let (mut run, _) = landing_in(TestDir::adopt(dir.into()));
    for _ in 0..cut {
        assert_eq!(run.land_one().dabs_applied, 1, "a dab at a time");
    }
    run.cut();
    // Landed, and not durable: the kill loses them, and the restore lands them again.
    for _ in 0..more {
        assert_eq!(run.land_one().dabs_applied, 1, "a dab at a time");
    }
    println!("{READY}");
    #[expect(clippy::infinite_loop, reason = "it waits for the parent to kill it")]
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

/// Runs the child in `dir` until it is [`READY`], then kills it with `SIGKILL`: it has landed
/// `cut` dabs by its last cut, and `more` after.
fn kill_child(dir: &TestDir, (cut, more): (usize, usize)) {
    use std::io::{BufRead as _, BufReader};
    use std::os::unix::process::ExitStatusExt as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new(std::env::current_exe().expect("the test binary"))
        .args([CHILD_TEST, "--exact", "--ignored", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(KILL_DIR, dir.path())
        .env(KILL_LANDING, format!("{cut},{more}"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the child starts");
    let stdout = child.stdout.take().expect("its output");
    let mut stderr = child.stderr.take().expect("its errors");
    // Both pipes are read on threads of their own, so neither fills up and stalls the child.
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        drop(std::io::Read::read_to_string(&mut stderr, &mut text));
        text
    });
    let (ready, ready_seen) = std::sync::mpsc::channel();
    // Read on a thread of its own, so a child that never gets there fails the test instead of
    // hanging it.
    let reader = std::thread::spawn(move || {
        let lines = BufReader::new(stdout).lines();
        let read = lines.map_while(Result::ok);
        for line in read {
            // After libtest's `test <name> ... `, on the same line.
            if line.ends_with(READY) {
                // The parent may have stopped waiting: then nothing listens.
                ready.send(()).unwrap_or_default();
            }
        }
    });
    let got_there = ready_seen.recv_timeout(CHILD_TIMEOUT);
    child.kill().expect("killed");
    let status = child.wait().expect("reaped");
    reader.join().expect("its output read");
    let errors = errors.join().expect("its errors read");
    assert!(
        got_there.is_ok(),
        "the child got where it is killed; it said:\n{errors}"
    );
    assert_eq!(status.signal(), Some(SIGKILL), "killed: {status}");
}

/// A finest tile's interior quantization steps, and every vertex's correction.
pub(super) type Corrections = ([f32; 3], Vec<geom::Vec3>);

/// Every finest tile's corrections in `project`.
pub(super) fn corrections(project: &mut Project) -> BTreeMap<TileKey, Corrections> {
    let keys = tile_keys(project);
    let store = project.store();
    let tiles = keys.into_iter().map(|key| {
        let view = store.correction(key).expect("its corrections");
        let side = view.side();
        let at = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let values = at.map(|(i, j)| view.get(i, j).expect("in the tile"));
        let values: Vec<geom::Vec3> = values.collect();
        (key, (view.scales(), values))
    });
    tiles.collect()
}

/// How far `got`'s corrections are from `expected`'s at most, in quantization steps of the
/// coarser of each tile's two: over every component of every vertex.
pub(super) fn steps_apart(
    got: &BTreeMap<TileKey, Corrections>,
    expected: &BTreeMap<TileKey, Corrections>,
) -> f32 {
    let (keys, expected_keys): (Vec<&TileKey>, Vec<&TileKey>) =
        (got.keys().collect(), expected.keys().collect());
    assert_eq!(keys, expected_keys, "the same tiles");
    let pairs = got.iter().zip(expected.values());
    let per_tile = pairs.map(|((key, (scales, values)), (other_scales, other))| {
        assert_eq!(values.len(), other.len(), "{key:?}: the same vertices");
        let step = [0, 1, 2].map(|c| scales[c].max(other_scales[c]));
        let vertices = values.iter().zip(other);
        let apart = vertices.flat_map(|(a, b)| {
            let (a, b) = ([a.x, a.y, a.z], [b.x, b.y, b.z]);
            let components = a.into_iter().zip(b).zip(step);
            components.map(|((mine, theirs), unit)| (mine - theirs).abs() / unit)
        });
        apart.fold(0.0_f32, f32::max)
    });
    per_tile.fold(0.0_f32, f32::max)
}

#[test]
fn a_stroke_killed_mid_landing_comes_back_whole() {
    // A run never killed: every dab landed in one process.
    let (mut whole, placed) = landing_in(TestDir::new());
    assert!(
        placed > 16,
        "a stroke long enough to kill midway: {placed} dabs"
    );
    whole.brush.cost = DabCost::any_number();
    whole.land_one();
    assert!(!whole.brush.stroking(), "it landed whole");
    let expected = corrections(&mut whole.project);
    whole.project.close(false).expect("closed");
    // Killed at the start, the middle and near the end of the landing, with dabs landed after
    // the last cut each time but one: those the kill loses, and the restore lands again.
    let kills = [(0, 2), (placed / 2, 0), (placed - 6, 3)];
    for (cut, more) in kills {
        let dir = TestDir::new();
        kill_child(&dir, (cut, more));
        let mut owner = resumed_in(&dir, Sculpting::new);
        let resumed = owner.brush().stroke.as_ref().expect("taken on");
        let op = resumed.op;
        assert_eq!(
            usize::try_from(resumed.landed).expect("a few"),
            cut,
            "the landed count of the last durable cut"
        );
        assert_eq!(
            resumed.dabs.len(),
            placed - cut,
            "every dab not landed by it"
        );
        run(&mut owner, 10_000);
        assert!(!owner.brush().stroking(), "it landed and ended");
        assert_eq!(owner.brush().error(), None);
        let got = corrections(owner.project());
        super::cut(owner.project(), &mut Instant::now());
        let path = owner
            .project()
            .recovery_path()
            .expect("a recovery file")
            .to_owned();
        owner.close().expect("closed");
        let total = u32::try_from(placed).expect("a few");
        let batches = batches_at(&path, op);
        let last = batches.last().expect("batches");
        assert_eq!(
            (last.placed(), last.landed()),
            (total, total),
            "killed after {cut} + {more}: every dab placed, each landed once"
        );
        // The restore starts each vertex it touches from the stored, rounded value, where the
        // run never killed kept it unrounded: one more rounding, of at most half a step (the
        // spec's bound). Each stored value is rounded once more besides, to within half its own
        // step, and the two tiles' steps may differ a little (a tile's scale follows its
        // values): so the stored values are at most a step and a half apart. A step is what
        // shows (measured: 1.0007 steps in the middle of the landing, none at its start).
        let apart = steps_apart(&got, &expected);
        assert!(
            apart <= 1.5,
            "killed after {cut} + {more}: {apart} steps from the run never killed"
        );
        drop(dir);
    }
}

#[test]
fn a_window_with_no_size_yet_takes_the_stroke_on() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT, 2);
    // Minimised as it opened, or not yet configured.
    let unsized_setup = Setup {
        size: None,
        ..setup()
    };
    let mut owner = owner_of(restore_in(&crashed.dir), Sculpting::new, unsized_setup);
    owner.resume_stroke();
    assert!(
        owner.brush().under_way(),
        "taken on: {:?}",
        owner.brush().error()
    );
    assert_eq!(rows_left(&owner), crashed.waiting(), "every dab not landed");
    owner.close().expect("closed, landing it");
}

/// A crash with a stroke landing whose journal holds `rows`, none of them landed, in the
/// brush's dab format: the crash's directory.
fn crash_with_rows(rows: &[DabRow]) -> TestDir {
    let mut run = Journalling::new();
    run.project.set_dab_format(stroke::DAB_FORMAT);
    run.project.begin_stroke_and_wait().expect("begun");
    run.project.journal_placed(rows).expect("journalled");
    run.cut();
    let Journalling { dir, project, .. } = run;
    // A crash: nothing closes the project.
    drop(project);
    dir
}

#[test]
fn a_row_that_cant_be_a_dab_of_the_model_is_not_taken_on() {
    let crashed = crash_mid_landing(stroke::DAB_FORMAT, 0);
    let good: Dab = bytemuck::cast(*crashed.placed.first().expect("a dab"));
    let hostile = [
        Dab {
            radius: f32::NAN,
            ..good
        },
        Dab {
            radius: 1e9,
            ..good
        },
        Dab {
            position: [1e6, 0.0, 0.0],
            ..good
        },
        Dab {
            radius: -1.0,
            ..good
        },
    ];
    for dab in hostile {
        let dir = crash_with_rows(&[row_of(good), row_of(dab)]);
        let mut owner = resumed_in(&dir, Sculpting::new);
        assert!(!owner.brush().stroking(), "not taken on: {dab:?}");
        let error = owner.brush().error().expect("said").to_owned();
        assert!(error.contains("can't be a dab of this model"), "{error}");
        let note = OpenNote::Restored {
            path: "recovery.bcrf".into(),
            when: std::time::SystemTime::now(),
        };
        owner.note(&note, Instant::now());
        let bar = owner.bar().expect("the note");
        let last = bar.details().last().map(String::as_str);
        assert_eq!(last, Some(UNFINISHED_STROKE), "the note says so");
        owner.close().expect("closed");
    }
}
