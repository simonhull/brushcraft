//! Ctrl+S end to end, with no GPU.
//!
//! What a save holds (the dabs still catching up, the stroke the pen is drawing, not a stroke begun
//! after it), what the title says, a save that fails, and a close meanwhile.

use std::path::PathBuf;

use super::*;
use crate::frame::{CloseAnswer, UiKey, WindowRequest};
use crate::saving::{SaveStep, Stage};

/// Where the tests' projects are saved in `dir`.
pub(super) fn path_in(dir: &TestDir) -> PathBuf {
    dir.path().join("p.bcrf")
}

/// A project saved in `dir`, with no edits.
fn saved_project(dir: &TestDir) -> Project {
    let mut project = project(dir);
    drop(project.save_as(&path_in(dir)).expect("saved"));
    project
}

/// The frame loop over a saved project in `dir`, whose brush lands a dab at a time.
fn slow_owner(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    owner_of(
        saved_project(dir),
        |project| Sculpting::new(project).with_cost(DabCost::one_at_a_time()),
        setup(),
    )
}

/// Runs a close to its end as the window's passes do, however long the disk takes: a flush's sync
/// can outlast any count of passes that don't wait for it, so the limit is one of time.
fn close_out(owner: &mut Owner<Sculpting, NoUploads>) {
    let began = Instant::now();
    while !owner.ready_to_close() {
        owner.close_step().expect("a step");
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(began.elapsed() < Duration::from_secs(60), "it ends");
    }
}

/// Runs the save under way as the window's passes do, a step and then a frame, until it ends.
pub(super) fn finish_save(owner: &mut Owner<Sculpting, NoUploads>) {
    let mut passes = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(owner, 1);
        passes += 1;
        assert!(passes < 100_000, "the save ends");
    }
}

/// A copy of the file at `dir`'s project path, as a crash now would leave it: an open project
/// holds the file's lock.
fn copy_on_disk(dir: &TestDir) -> PathBuf {
    let copy = dir.path().join("copy.bcrf");
    std::fs::copy(path_in(dir), &copy).expect("copied");
    copy
}

/// Every tile's positions in the file at `dir`'s project path, as the last save left it.
fn tiles_on_disk(dir: &TestDir) -> BTreeMap<TileKey, Vec<[u32; 3]>> {
    let mut opened = Project::open(&copy_on_disk(dir), config(dir)).expect("it opens");
    let tiles = positions(&mut opened);
    opened.close(false).expect("closed");
    tiles
}

/// The stroke records in the file at `dir`'s project path.
fn records_on_disk(dir: &TestDir) -> usize {
    let file = ProjectFile::open(&copy_on_disk(dir)).expect("it opens");
    file.strokes().count()
}

/// How many save points the file at `dir`'s project path holds.
fn save_points(dir: &TestDir) -> usize {
    let file = ProjectFile::open(&copy_on_disk(dir)).expect("it opens");
    file.save_points().expect("save points").len()
}

/// The files in `dir`'s recovery directory.
pub(super) fn recovery_files(dir: &TestDir) -> usize {
    let found = std::fs::read_dir(dir.path().join("recovery")).expect("the directory");
    found.count()
}

/// A stroke from `line`, in `steps` drags, whose `Down` comes after the Ctrl+S just asked for: a
/// test's strokes are dated back from when they are made, at the pen's rate.
fn stroke_after_ctrl_s(line: ((f64, f64), (f64, f64)), steps: u32) -> Vec<PenSample> {
    let period = Duration::from_secs_f64(crate::script::PEN_RATE.recip());
    std::thread::sleep(period * (steps + 3));
    stroke(line, steps)
}

/// The tiles `samples` leave on a new project in a directory of their own, every dab applied as
/// it is placed: what a save of them must hold.
fn drawn_at_once(samples: Vec<PenSample>, epoch: Instant) -> BTreeMap<TileKey, Vec<[u32; 3]>> {
    let dir = TestDir::new();
    let mut owner = owner_of(
        project(&dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::any_number())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut owner, 10_000);
    queue(&mut owner, samples);
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it landed");
    let tiles = positions(owner.project());
    owner.close().expect("closed");
    tiles
}

#[test]
fn a_saved_stroke_reopens_as_the_same_tiles() {
    let dir = TestDir::new();
    let mut owner = owner_of(saved_project(&dir), Sculpting::new, setup());
    run(&mut owner, 10_000);
    let before = tiles_on_disk_of(&mut owner);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    let drawn = positions(owner.project());
    assert_ne!(drawn, before, "the stroke changed the surface");
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(tiles_on_disk(&dir), drawn);
    assert_eq!(records_on_disk(&dir), 1, "with its record");
}

/// The tiles as they stand in `owner`'s project.
fn tiles_on_disk_of(owner: &mut Owner<Sculpting, NoUploads>) -> BTreeMap<TileKey, Vec<[u32; 3]>> {
    positions(owner.project())
}

#[test]
fn a_save_with_dabs_still_catching_up_holds_them_all() {
    let dir = TestDir::new();
    let epoch = Instant::now();
    let mut owner = owner_of(
        saved_project(&dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut owner, 10_000);
    let samples = stroke(ACROSS, 12);
    queue(&mut owner, samples.clone());
    assert_eq!(run(&mut owner, 1), 1);
    let pending = owner.brush().last().dabs_catching_up;
    assert!(pending > 2, "{pending} dabs still to land");
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert!(!owner.brush().stroking(), "every dab landed");
    assert!(!owner.project().has_unsaved_edits(), "and is saved");
    owner.close().expect("closed");
    assert_eq!(tiles_on_disk(&dir), drawn_at_once(samples, epoch));
    assert_eq!(records_on_disk(&dir), 1);
}

/// A big backlog at Ctrl+S doesn't hold the window up: the request lands nothing, each step a
/// batch, and the title counts down.
#[test]
fn a_save_is_answered_at_once_and_lands_a_batch_a_step() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    let mut pen = to_and_fro(4);
    pen.push(sample(Phase::Up, ACROSS.0, 0.0));
    queue(&mut owner, pen);
    run(&mut owner, 1);
    let backlog = owner.landing();
    assert!(backlog > 8, "{backlog} dabs to land");
    owner.request_save(Instant::now());
    assert_eq!(owner.landing(), backlog, "nothing landed in the call");
    owner.save_step();
    assert_eq!(owner.save_stage(), Some(Stage::Landing));
    let mut said = Vec::new();
    while owner.save_stage() == Some(Stage::Landing) {
        let before = owner.landing();
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        assert!(
            before.saturating_sub(owner.landing()) <= 2,
            "a batch a step: {before} to {}",
            owner.landing()
        );
        said.extend(owner.title_change());
    }
    let counts = said.iter().filter(|title| title.contains("dabs to land"));
    assert!(counts.count() > 1, "counted down: {said:?}");
    finish_save(&mut owner);
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
}

#[test]
fn ctrl_s_during_a_stroke_saves_at_its_end_with_all_of_it() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    let mut samples = stroke(ACROSS, 12);
    let up = samples.pop().expect("its up");
    let later = samples.split_off(8);
    queue(&mut owner, samples);
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    for _ in 0..5 {
        assert_eq!(owner.save_step(), SaveStep::Idle, "the pen is down");
        run(&mut owner, 1);
    }
    assert_eq!(owner.save_stage(), Some(Stage::PenDown));
    assert!(
        owner
            .title_change()
            .is_some_and(|title| title.contains("when the stroke ends")),
        "the title says so"
    );
    assert_eq!(save_points(&dir), 1, "nothing saved yet");
    let placed = owner.landing();
    queue(&mut owner, later);
    run(&mut owner, 1);
    assert!(owner.landing() > placed, "the stroke goes on being drawn");
    owner.pen(PenSample {
        received: Instant::now(),
        ..up
    });
    run(&mut owner, 1);
    finish_save(&mut owner);
    let drawn = positions(owner.project());
    assert!(!owner.brush().stroking(), "the stroke landed");
    assert!(!owner.project().has_unsaved_edits(), "and is saved");
    owner.close().expect("closed");
    assert_eq!(tiles_on_disk(&dir), drawn);
    assert_eq!(records_on_disk(&dir), 1, "one stroke, whole");
    assert_eq!(save_points(&dir), 2, "one save");
}

/// What a save holds is fixed by Ctrl+S: a stroke begun after it waits, its samples kept, and
/// begins once the save is written.
#[test]
fn a_stroke_begun_during_a_save_waits_for_it_and_is_not_in_it() {
    let dir = TestDir::new();
    let epoch = Instant::now();
    let mut owner = owner_of(
        saved_project(&dir),
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut owner, 10_000);
    let first = stroke(ACROSS, 12);
    queue(&mut owner, first.clone());
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    owner.save_step();
    assert_eq!(owner.save_stage(), Some(Stage::Landing));
    let second = stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6);
    queue(&mut owner, second);
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "it waits");
    finish_save(&mut owner);
    assert_eq!(
        tiles_on_disk(&dir),
        drawn_at_once(first, epoch),
        "the first stroke, and nothing of the second"
    );
    assert_eq!(records_on_disk(&dir), 1);
    let title = owner.title_change().expect("the second stroke's to come");
    assert!(title.starts_with('*'), "it isn't saved: {title}");
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it began and landed");
    assert!(owner.project().has_unsaved_edits());
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "the next save holds it");
}

/// With nothing left to land, a stroke begun as the save writes waits all the same: the store
/// mustn't change under it.
#[test]
fn a_stroke_begun_as_a_save_writes_waits_for_it() {
    let dir = TestDir::new();
    let mut owner = owner_of(saved_project(&dir), Sculpting::new, setup());
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    let drawn = positions(owner.project());
    owner.request_save(Instant::now());
    owner.save_step();
    owner.save_step();
    assert_eq!(owner.save_stage(), Some(Stage::Writing));
    queue(
        &mut owner,
        stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6),
    );
    run(&mut owner, 1);
    assert!(!owner.brush().under_way(), "it hasn't begun");
    assert!(!owner.brush().deferred.is_empty(), "it waits, kept");
    finish_save(&mut owner);
    assert_eq!(tiles_on_disk(&dir), drawn, "nothing of it");
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it began and landed");
    assert_ne!(positions(owner.project()), drawn, "it drew");
    owner.close().expect("closed");
}

/// A stroke begun over the one before it, still landing, when Ctrl+S comes was drawn before it:
/// it's in the save.
#[test]
fn a_stroke_begun_over_another_at_ctrl_s_is_in_the_save() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert_eq!(
        owner.brush().strokes.len(),
        2,
        "the second began over the first"
    );
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
}

/// Nor does a stroke let go of by the last one's end and not yet begun go missing.
#[test]
fn a_stroke_about_to_begin_at_ctrl_s_is_in_the_save() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    let waiting = stroke(ACROSS, 12).into_iter().map(Some);
    owner.brush_mut().deferred.extend(waiting);
    owner.request_save(Instant::now());
    let mut steps = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        steps += 1;
        assert!(steps < 100_000, "it ends");
    }
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 1);
}

/// One waiting behind the stroke the pen is drawing, its `Up` lost, doesn't hold the save for ever:
/// the `Down` after Ctrl+S ends it, as it would any other time.
#[test]
fn a_lost_up_before_a_stroke_that_waits_doesnt_hold_the_save_up() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    let mut no_up = stroke(((150.0, 80.0), (170.0, 120.0)), 6);
    no_up.pop();
    queue(&mut owner, no_up);
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    queue(
        &mut owner,
        stroke_after_ctrl_s(((120.0, 90.0), (140.0, 90.0)), 6),
    );
    run(&mut owner, 1);
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "the strokes before it");
}

#[test]
fn a_failed_save_keeps_the_edits_the_marker_and_the_recovery_file() {
    let dir = TestDir::new();
    // Never saved: it has no path to save to, as a failed write has no save point.
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    let drawn = positions(owner.project());
    drop(owner.title_change());
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    let title = owner.title_change().expect("it says so");
    assert!(title.starts_with('*'), "still unsaved: {title}");
    assert!(title.contains("save failed"), "{title}");
    let line = owner.save_line();
    assert!(!line.contains("no unsaved edits"), "{line}");
    assert!(line.contains("save failed"), "{line}");
    assert!(owner.project().has_unsaved_edits());
    assert_eq!(positions(owner.project()), drawn, "nothing lost");

    // Strokes go on, and the failure stays said until the next try.
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "it landed");
    assert!(owner.brush().last().dabs_applied > 0, "it drew");
    assert!(owner.save_line().contains("save failed"));

    // Saving again works, with somewhere to save to.
    drop(owner.project().save_as(&path_in(&dir)).expect("saved"));
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert!(!owner.save_line().contains("failed"));
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
}

#[test]
fn a_failed_save_leaves_the_recovery_file_for_the_close_to_keep() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 1, "its edits are there");
}

/// A landing that drops dabs can't be called saved: the save stops, says why, and strokes go on.
#[test]
fn a_store_error_while_landing_fails_the_save() {
    let (_dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    owner.brush_mut().held.extend(heavy.into_iter().map(Some));
    owner.request_save(Instant::now());
    // No frames between the steps, as a close's test: they would let the pins go.
    let mut steps = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        steps += 1;
        assert!(steps < 10_000, "the save ends");
    }
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("save failed"), "{title}");
    assert!(title.contains("applying dabs"), "{title}");
    assert_eq!(owner.brush().save_asked, None, "strokes aren't held");
    assert_eq!(owner.brush().landing, Landing::InFrames);
    owner.close().expect("closed");
}

#[test]
fn saved_then_closed_leaves_no_recovery_file() {
    let dir = TestDir::new();
    let mut owner = owner_of(saved_project(&dir), Sculpting::new, setup());
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    assert_eq!(recovery_files(&dir), 1);
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 0);
}

#[test]
fn the_unsaved_marker_goes_with_a_save_and_comes_back_with_a_stroke() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    assert_eq!(owner.title_change().as_deref(), Some("p - BrushCraft"));
    assert_eq!(owner.title_change(), None, "said once");
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    assert!(!owner.project().has_unsaved_edits() || owner.landing() > 0);
    assert_eq!(
        owner.title_change().as_deref(),
        Some("*p - BrushCraft"),
        "dabs shown and not landed are unsaved too"
    );
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert_eq!(owner.title_change().as_deref(), Some("p - BrushCraft"));
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert_eq!(owner.title_change().as_deref(), Some("*p - BrushCraft"));
    run(&mut owner, 10_000);
    assert_eq!(owner.landing(), 0, "it landed");
    assert_eq!(owner.title_change(), None, "landed edits are unsaved too");
    owner.close().expect("closed");
}

#[test]
fn ctrl_s_with_nothing_to_save_writes_nothing() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    owner.request_save(Instant::now());
    assert_eq!(owner.save_stage(), None);
    assert!(owner.save_line().contains("nothing to save"));
    owner.close().expect("closed");
    assert_eq!(save_points(&dir), 1, "only the first");
}

#[test]
fn ctrl_s_during_a_save_is_that_save() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    owner.request_save(Instant::now());
    assert_eq!(owner.save_stage(), Some(Stage::Landing), "it goes on");
    finish_save(&mut owner);
    owner.close().expect("closed");
    assert_eq!(save_points(&dir), 2, "one save");
}

/// A close during a save lets it finish, with what waited for it, and no recovery file is left.
#[test]
fn a_close_during_a_save_waits_for_it_and_it_holds_every_stroke() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    queue(
        &mut owner,
        stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6),
    );
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the second waits");
    owner.request_close().expect("a close");
    assert!(!owner.ready_to_close(), "the save goes first");
    close_out(&mut owner);
    assert_eq!(owner.save_stage(), None, "saved");
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
}

/// A close during the writing lets the stroke that waited go: the save waits for all of it, however
/// many batches it lands in, so the file never holds half a stroke.
#[test]
fn a_close_as_a_save_writes_saves_a_waiting_stroke_whole() {
    for steps in [2, 40] {
        let dir = TestDir::new();
        let mut owner = slow_owner(&dir);
        run(&mut owner, 10_000);
        queue(&mut owner, stroke(ACROSS, 12));
        run(&mut owner, 10_000);
        owner.request_save(Instant::now());
        owner.save_step();
        owner.save_step();
        assert_eq!(owner.save_stage(), Some(Stage::Writing));
        queue(
            &mut owner,
            stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), steps),
        );
        run(&mut owner, 1);
        assert!(!owner.brush().deferred.is_empty(), "it waits");
        owner.request_close().expect("a close");
        close_out(&mut owner);
        owner.close().expect("closed");
        assert_eq!(records_on_disk(&dir), 2, "{steps}: both strokes, whole");
        assert_eq!(recovery_files(&dir), 0, "{steps}: nothing unsaved");
    }
}

/// A save that fails as the window closes is the user's to see: the window stays, saying so, and
/// a further close takes it, the failure an error of the run and the recovery file kept.
#[test]
fn a_save_failing_during_a_close_holds_the_window_and_fails_the_run() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    owner.request_save(Instant::now());
    owner.request_close().expect("a close");
    finish_save(&mut owner);
    assert!(!owner.ready_to_close(), "the window stays");
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("save failed"), "{title}");
    let closed = owner.close().expect_err("the run failed");
    assert!(closed.to_string().contains("saving"), "{closed:#}");
    assert_eq!(recovery_files(&dir), 1, "its edits are there");
}

#[test]
fn a_held_close_goes_once_nothing_is_left_to_save() {
    let dir = TestDir::new();
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    owner.request_save(Instant::now());
    owner.request_close().expect("a close");
    finish_save(&mut owner);
    assert!(owner.close_held());
    drop(owner.project().save_as(&path_in(&dir)).expect("saved"));
    owner.request_save(Instant::now());
    assert!(owner.ready_to_close(), "nothing is left unsaved");
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 0);
}

/// Dabs dropped while the save waited for the pen: it can't say it holds them.
#[test]
fn dabs_dropped_with_the_pen_down_fail_the_save() {
    let (_dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    owner.request_save(Instant::now());
    assert_eq!(owner.save_stage(), Some(Stage::PenDown));
    queue(&mut owner, heavy);
    run(&mut owner, 10_000);
    assert!(owner.brush().error().is_some(), "the heavy dabs don't fit");
    finish_save(&mut owner);
    assert!(owner.save_line().contains("save failed"));
    let title = owner.title_change().expect("it says so");
    assert!(title.starts_with('*'), "not saved: {title}");
    assert!(owner.project().has_unsaved_edits());
    owner.close().expect("closed");
}

/// A landing that fails as the project closes leaves the save unmade and the recovery file there.
#[test]
fn a_failed_flush_skips_the_save_and_keeps_the_recovery_file() {
    let (_dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    let recovery = owner.project().recovery_path().expect("a file").to_owned();
    owner.brush_mut().held.extend(heavy.into_iter().map(Some));
    owner.request_save(Instant::now());
    let closed = owner.close().expect_err("the dabs don't fit");
    assert!(
        format!("{closed:#}").contains("applying dabs"),
        "{closed:#}"
    );
    assert!(recovery.exists(), "the edits since the last save");
}

/// The samples of a stroke from before Ctrl+S, not yet routed: it's in the save.
#[test]
fn a_stroke_queued_at_ctrl_s_is_in_the_save() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    owner.request_save(Instant::now());
    assert_eq!(owner.save_step(), SaveStep::Idle, "its samples wait");
    finish_save(&mut owner);
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 1);
}

/// A stroke that placed no dab has no edits: no empty save point for it.
#[test]
fn a_stroke_of_no_dabs_saves_nothing() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    let (from, to) = ACROSS;
    let light = [sample(Phase::Down, from, 0.0), sample(Phase::Drag, to, 0.0)];
    queue(&mut owner, light.to_vec());
    run(&mut owner, 1);
    assert!(owner.brush().under_way(), "the pen is down");
    owner.request_save(Instant::now());
    owner.pen(sample(Phase::Up, to, 0.0));
    finish_save(&mut owner);
    assert!(owner.save_line().contains("nothing to save"));
    assert_eq!(owner.brush().save_asked, None, "strokes aren't held");
    owner.close().expect("closed");
    assert_eq!(save_points(&dir), 1, "only the first");
}

/// The window gone at a second close, the save asked for is still made.
#[test]
fn a_save_cut_off_by_the_window_going_is_finished_as_the_project_closes() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    queue(
        &mut owner,
        stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6),
    );
    run(&mut owner, 1);
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "every stroke landed and saved");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
}

/// Ctrl+S with the window hidden: no frames to land in, and the save goes on all the same.
#[test]
fn a_save_needs_no_frames() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.resize(None);
    owner.request_save(Instant::now());
    let mut steps = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        steps += 1;
        assert!(steps < 100_000, "it ends");
    }
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 1);
}

/// Runs passes of the window's loop until a close bar's save is over, the window's request left
/// for the test to take.
fn finish_closing_save(owner: &mut Owner<Sculpting, NoUploads>) {
    finish_save(owner);
    if owner.save_step() == SaveStep::Waiting {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A failed save is the window's to survive: with `exit_after_save` set it stays, the edits and the
/// recovery file with it, and the title says so.
#[test]
fn a_failed_save_after_asking_keeps_the_window() {
    let dir = TestDir::new();
    // Never saved: no path to save to.
    let mut owner = owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    let drawn = positions(owner.project());
    let now = Instant::now();
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "Save");
    drop(owner.title_change());
    finish_closing_save(&mut owner);
    assert_eq!(owner.take_request(), None, "the window stays");
    assert!(!owner.ready_to_close(), "nothing is on its way out");
    assert!(
        !owner.close_held(),
        "and no close waits: that was a scripted run's"
    );
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("save failed"), "{title}");
    assert!(title.starts_with('*'), "still unsaved: {title}");
    assert_eq!(positions(owner.project()), drawn, "nothing lost");

    // The bar went, and the user can go on: a close asks again.
    assert!(!owner.key(UiKey::Escape, Instant::now()), "no bar left");
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 10_000);
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    owner.close().expect("closed");
    assert_eq!(recovery_files(&dir), 1, "its edits are there");
}

/// Dabs dropped while the bar's save lands fail it as any: the window stays.
#[test]
fn a_save_after_asking_with_dabs_dropped_keeps_the_window() {
    let (_dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    owner.brush_mut().held.extend(heavy.into_iter().map(Some));
    let now = Instant::now();
    owner.request_save(now);
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    // No frames between the steps, as a close's test: they would let the pins go.
    let mut steps = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        steps += 1;
        assert!(steps < 10_000, "the save ends");
    }
    if owner.save_step() == SaveStep::Waiting {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(owner.take_request(), None, "not saved: the window stays");
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("save failed"), "{title}");
    owner.close().expect("closed");
}

/// Everything drawn is in the file before the window goes, the stroke that waited for an earlier
/// save too: the close bar's Save is a promise to keep all of it.
#[test]
fn a_stroke_that_waited_for_the_save_is_saved_before_the_window_goes() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    queue(
        &mut owner,
        stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6),
    );
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the second waits");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let mut request = None;
    let mut passes = 0;
    while request.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        request = owner.take_request();
        passes += 1;
        assert!(passes < 100_000, "it ends");
    }
    assert_eq!(request, Some(WindowRequest::Exit));
    assert!(!owner.project().has_unsaved_edits(), "all of it saved");
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
}

/// A stroke that drew nothing has nothing to save: Save goes all the same.
#[test]
fn saving_nothing_after_asking_exits() {
    let dir = TestDir::new();
    let mut owner = owner_of(saved_project(&dir), Sculpting::new, setup());
    run(&mut owner, 10_000);
    owner.pen(sample(Phase::Down, (1.0, 1.0), 0.5));
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    owner.pen(sample(Phase::Up, (1.0, 1.0), 0.0));
    run(&mut owner, 100);
    assert!(owner.key(UiKey::Enter, Instant::now()), "Save");
    owner.save_step();
    assert_eq!(owner.take_request(), Some(WindowRequest::Exit));
    owner.close().expect("closed");
}

/// A stroke on the model during the bar's save cancels the exit, and waits for the save.
#[test]
fn a_stroke_while_finishing_stays_out_of_the_save_and_the_window_stays() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    let now = Instant::now();
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "Save");
    owner.save_step();
    queue(
        &mut owner,
        stroke_after_ctrl_s(((150.0, 80.0), (170.0, 120.0)), 6),
    );
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "it waits");
    finish_closing_save(&mut owner);
    assert_eq!(owner.take_request(), None, "cancelled");
    assert_eq!(records_on_disk(&dir), 1, "the first stroke alone");
    run(&mut owner, 10_000);
    assert!(owner.project().has_unsaved_edits(), "the second is unsaved");
    owner.close().expect("closed");
}

/// The frame loop over a saved project in `dir`, with a stroke half landed, a second begun over
/// it, the pen still on that, and a sample held over: all that a brush can hold at once.
fn holding_everything(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    let mut owner = slow_owner(dir);
    run(&mut owner, 10_000);
    let mut open = stroke(ACROSS, 12);
    open.pop();
    queue(&mut owner, open);
    run(&mut owner, 1);
    let second = stroke(((150.0, 80.0), (170.0, 120.0)), 6);
    queue(&mut owner, second.clone());
    run(&mut owner, 1);
    assert_eq!(owner.brush().strokes.len(), 2, "the second began over it");
    let held = second.first().copied();
    owner.brush_mut().held.push(held);
    assert!(owner.brush().under_way(), "the first is under way");
    assert!(owner.landing() > 1, "and there's plenty held");
    owner
}

/// Frames go on after it here, though none do in the app, for what is held to land if it can.
#[test]
fn abandoning_forgets_what_is_held_and_lands_nothing() {
    let dir = TestDir::new();
    let mut owner = holding_everything(&dir);
    owner.brush_mut().begin_saving(Instant::now());
    let before = positions(owner.project());
    owner.brush_mut().abandon();
    assert_eq!(owner.landing(), 0, "nothing is held");
    assert_eq!(owner.brush().landing_for_save(), 0, "nor for the save");
    assert!(!owner.brush().pen_down(), "the pen is off it");
    assert!(!owner.brush().stroking(), "no stroke is open");
    assert_eq!(owner.brush().landing, Landing::InFrames);
    assert_eq!(owner.brush().save_asked, None, "no save holds strokes back");
    owner.expose();
    run(&mut owner, 10_000);
    assert_eq!(positions(owner.project()), before, "no tile changed");
    assert_eq!(owner.landing(), 0, "and nothing came back");
    owner.discard().expect("discarded");
}

/// The file's readers see the last save. That nothing landed can't be read here, with the store
/// gone: `frame::tests::discard_tells_the_brush_to_forget_and_never_to_land` says so.
#[test]
fn discard_leaves_the_saved_tiles_and_removes_the_recovery_file() {
    let dir = TestDir::new();
    let owner = holding_everything(&dir);
    let saved = tiles_on_disk(&dir);
    assert_eq!(recovery_files(&dir), 1, "the session's");
    owner.discard().expect("discarded");
    assert_eq!(recovery_files(&dir), 0, "the work is gone, as asked");
    let reopened = tiles_on_disk(&dir);
    assert_eq!(reopened, saved, "the project is as it was saved");
}

#[test]
fn discard_leaves_the_project_file_as_it_was() {
    let dir = TestDir::new();
    let owner = holding_everything(&dir);
    let before = std::fs::read(path_in(&dir)).expect("the file");
    owner.discard().expect("discarded");
    let after = std::fs::read(path_in(&dir)).expect("the file");
    assert!(before == after, "byte for byte");
}

/// Don't save with a save written ahead leaves what a reader sees as it was.
///
/// A save's writing copies records into the file ahead of its save point, so the file grows, but
/// no save point reaches them until the save's last step.
#[test]
fn discard_during_a_save_leaves_what_the_project_file_holds() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    let file = path_in(&dir);
    let length = || std::fs::metadata(&file).expect("the file").len();
    let (tiles, points, before) = (tiles_on_disk(&dir), save_points(&dir), length());
    owner.request_save(Instant::now());
    while owner.save_stage() != Some(Stage::Writing) {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let mut steps = 0;
    loop {
        let slice = Instant::now() + Duration::from_millis(5);
        match owner.project().save_ahead(slice).expect("written ahead") {
            store::SaveAhead::Ready => break,
            store::SaveAhead::Flushing => std::thread::sleep(Duration::from_millis(1)),
            store::SaveAhead::Copying(_) => {}
        }
        steps += 1;
        assert!(steps < 100_000, "it gets there");
    }
    assert_eq!(
        owner.save_stage(),
        Some(Stage::Writing),
        "the save isn't over"
    );
    assert!(length() > before, "the file holds what it copied");
    owner.discard().expect("discarded");
    assert_eq!(tiles_on_disk(&dir), tiles, "what a reader sees");
    assert_eq!(save_points(&dir), points, "no save point more");
    assert_eq!(records_on_disk(&dir), 0, "no stroke of it");
    assert_eq!(recovery_files(&dir), 0, "the work is gone, as asked");
}

/// In the landing, the stroke's op open and dabs shown ahead: all of it just goes.
#[test]
fn discard_in_a_saves_landing_with_an_op_open() {
    let dir = TestDir::new();
    let mut owner = holding_everything(&dir);
    // The store's stroke is open, from the `Down`: its journal takes dabs.
    assert!(owner.project().journal_placed(&[]).is_ok(), "an op is open");
    owner.request_save(Instant::now());
    let saved = tiles_on_disk(&dir);
    owner.discard().expect("discarded");
    assert_eq!(recovery_files(&dir), 0);
    assert_eq!(tiles_on_disk(&dir), saved);
}

#[test]
fn discard_with_a_save_landing_leaves_the_file_as_it_was() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    owner.request_save(Instant::now());
    owner.save_step();
    owner.save_step();
    owner.save_step();
    assert_eq!(owner.save_stage(), Some(Stage::Landing), "under way");
    let saved = tiles_on_disk(&dir);
    owner.discard().expect("discarded");
    assert_eq!(recovery_files(&dir), 0);
    assert_eq!(tiles_on_disk(&dir), saved);
}

/// The recovery files in `dir`.
fn recovery_paths(dir: &TestDir) -> Vec<PathBuf> {
    let found = std::fs::read_dir(dir.path().join("recovery")).expect("the directory");
    found.map(|entry| entry.expect("an entry").path()).collect()
}

#[test]
fn discard_says_when_it_cant_remove_the_recovery_file() {
    let dir = TestDir::new();
    let owner = holding_everything(&dir);
    for file in recovery_paths(&dir) {
        std::fs::remove_file(&file).expect("removed");
        std::fs::create_dir_all(&file).expect("a directory in its place");
        std::fs::write(file.join("in the way"), b"").expect("written");
    }
    let error = owner.discard().expect_err("it can't be removed");
    let said = format!("{error:#}");
    assert!(said.contains("discarding the project"), "{said}");
}

/// A close would fail on this, and a close's failure is no reason to fail what the user asked.
#[test]
fn discard_is_content_with_a_recovery_file_already_gone() {
    let dir = TestDir::new();
    let owner = holding_everything(&dir);
    for file in recovery_paths(&dir) {
        std::fs::remove_file(&file).expect("removed");
    }
    owner.discard().expect("what was wanted");
}

/// A live run's record is on its way to its file as the run goes: Don't save stops the stream as
/// any end does, so what it had reaches the file, and the run's own end can write the rest.
#[test]
fn discard_keeps_a_live_runs_record() {
    let dir = TestDir::new();
    let path = dir.path().join("record.jsonl");
    let record = RecordSetup {
        path: Some(path.clone()),
        project: "p".to_owned(),
        gpu_timing: false,
        lod_timing: false,
        opening: Instant::now(),
    };
    let setup = Setup {
        record: Some(record),
        ..setup()
    };
    let mut owner = owner_of(saved_project(&dir), Sculpting::new, setup);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 10_000);
    assert!(
        !owner.record_lines().is_empty(),
        "the run's end has all of it to write"
    );
    owner.discard().expect("discarded");
    let written = std::fs::read_to_string(&path).expect("the record");
    assert!(
        written.contains("\"result\":\"run\""),
        "the stream reached the file: {written}"
    );
}

/// The real thing: dabs dropped in the frames under the bar, before Save.
#[test]
fn dabs_dropped_under_the_bar_stop_the_exit() {
    let (_dir, mut owner, heavy) = lifecycle::at_the_budgets_edge();
    let drops = owner.brush().drops();
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    queue(&mut owner, heavy);
    run(&mut owner, 10_000);
    assert!(owner.brush().drops() > drops, "the frames dropped dabs");
    owner.request_save(Instant::now());
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let mut steps = 0;
    while owner.save_stage().is_some() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        steps += 1;
        assert!(steps < 100_000, "the save ends");
    }
    if owner.save_step() == SaveStep::Waiting {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        owner.take_request(),
        None,
        "dropped dabs, and the window stays"
    );
    let title = owner.title_change().expect("it says so");
    assert!(title.contains("save failed"), "{title}");
    owner.close().expect("closed");
}

/// A saved project with a backlog: every dab of a stroke shown, none landed.
///
/// The project is in a directory of its own, the stroke to and fro `passes` times, its pen lifted,
/// drawn with a brush that lands a dab at a time from `epoch`.
fn backlogged(passes: u32, epoch: Instant) -> (TestDir, Owner<Sculpting, NoUploads>) {
    backlogged_in(TestDir::new(), passes, epoch)
}

/// [`backlogged`], its project in `dir`.
pub(super) fn backlogged_in(
    dir: TestDir,
    passes: u32,
    epoch: Instant,
) -> (TestDir, Owner<Sculpting, NoUploads>) {
    let mut project = saved_project(&dir);
    // As the app tells every project it opens: a restore reads the journal's rows only so.
    project.set_dab_format(stroke::DAB_FORMAT);
    let mut owner = owner_of(
        project,
        |project| {
            Sculpting::new(project)
                .with_cost(DabCost::one_at_a_time())
                .with_epoch(epoch)
        },
        setup(),
    );
    run(&mut owner, 10_000);
    let mut pen = to_and_fro(passes);
    pen.push(sample(Phase::Up, ACROSS.0, 0.0));
    queue(&mut owner, pen);
    run(&mut owner, 1);
    (dir, owner)
}

/// Save on the close bar with a large backlog: the window is asked to go before any dab lands.
///
/// The journal is durable by then, and the finisher's file is a waited save's, bit for bit: both
/// land the same dabs live, from the same pen-down surface, and only their batches differ, which
/// the landed bits never depend on.
///
/// In process, as the binary can't build a backlog on a small project. The time from Save to the
/// window's request to go ([`WindowRequest::Finish`]), which ends its event loop, is the disk's
/// (a flush's syncs): printed, and bounded here only loosely, as the tracked measurement holds its
/// 100 ms aim.
#[test]
fn save_on_a_large_backlog_lets_the_window_go_before_any_dab_lands() {
    const PASSES: u32 = 512;
    let epoch = Instant::now();
    let (dir, mut owner) = backlogged(PASSES, epoch);
    let backlog = owner.landing();
    assert!(backlog >= 4_000, "{backlog} dabs to land");
    assert_eq!(owner.ask_close(Instant::now()), CloseAnswer::Asked);
    let saved = Instant::now();
    assert!(owner.key(UiKey::Enter, saved), "Save");
    let mut asked = None;
    while asked.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        asked = owner.take_request();
        assert!(saved.elapsed() < Duration::from_secs(60), "it ends");
    }
    let took = saved.elapsed();
    assert_eq!(asked, Some(WindowRequest::Finish));
    assert!(took < Duration::from_secs(5), "{took:?} from Save");
    assert_eq!(owner.landing(), backlog, "nothing waited for the landing");
    assert!(owner.project().flushed(), "every dab placed is durable");
    // The app's own cost model from here: the test's lands a dab a batch.
    owner.brush_mut().cost = DabCost::default();
    let finishing = Instant::now();
    owner
        .finish_unseen(|summary, body| panic!("{summary}: {body}"))
        .expect("finished");
    let finished = finishing.elapsed();
    assert_eq!(recovery_files(&dir), 0, "saved, the recovery file gone");

    let (waited_dir, mut waited) = backlogged(PASSES, epoch);
    waited.brush_mut().cost = DabCost::default();
    waited.request_save(Instant::now());
    let waiting = Instant::now();
    finish_save(&mut waited);
    let in_window = waiting.elapsed();
    waited.close().expect("closed");
    println!(
        "{backlog} dabs: Save to Finish {took:?}; finisher {finished:?}, waited {in_window:?}"
    );
    assert_eq!(
        tiles_on_disk(&dir),
        tiles_on_disk(&waited_dir),
        "bit for bit"
    );
}

/// A stroke begun over the landing is journalled behind it, which no restore takes yet: Save waits
/// in the window until both strokes are saved, and never lets the window go first.
#[test]
fn a_stroke_begun_over_the_landing_keeps_the_window_until_it_is_saved() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert_eq!(owner.brush().strokes.len(), 2, "the second began over it");
    assert!(!owner.brush().pen_down(), "its pen has lifted");
    let now = Instant::now();
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "Save");
    let began = Instant::now();
    let mut asked = None;
    while asked.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        asked = owner.take_request();
        assert!(began.elapsed() < Duration::from_secs(60), "it ends");
    }
    assert_eq!(asked, Some(WindowRequest::Exit), "saved in the window");
    assert_eq!(owner.landing(), 0, "both landed first");
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
}

/// [`slow_owner`] whose strokes never begin over one still landing: each waits for it, as past
/// the open strokes' share ([`Begin::Waits`]).
fn waiting_owner(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    let mut owner = slow_owner(dir);
    owner.brush_mut().knob = Knob::WaitsOverLanding;
    owner
}

/// A stroke waiting for the one before it when Ctrl+S comes was drawn before it: it's in the save.
#[test]
fn a_stroke_waiting_behind_another_at_ctrl_s_is_in_the_save() {
    let dir = TestDir::new();
    let mut owner = waiting_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the second waits");
    assert_eq!(
        owner.brush().deferred_in_save().count(),
        owner.brush().deferred.len(),
        "all of it in the save"
    );
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
}

/// A stroke waiting behind the landing is pen samples, not journalled dabs, so no journal holds it:
/// Save waits in the window until both strokes are saved, and never lets the window go first.
#[test]
fn a_stroke_waiting_behind_the_landing_keeps_the_window_until_it_is_saved() {
    let dir = TestDir::new();
    let mut owner = waiting_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert!(
        !owner.brush().deferred.is_empty(),
        "the second waits, whole"
    );
    assert!(owner.brush().unplaced_for_save() > 0, "unplaced samples");
    let now = Instant::now();
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "Save");
    let began = Instant::now();
    let mut asked = None;
    while asked.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        asked = owner.take_request();
        assert!(began.elapsed() < Duration::from_secs(60), "it ends");
    }
    assert_eq!(asked, Some(WindowRequest::Exit), "saved in the window");
    assert_eq!(owner.landing(), 0, "both landed first");
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
}

/// [`holding_everything`] with the second stroke waiting behind the first ([`Begin::Waits`]).
///
/// A stroke half landed, the pen still on it, a second waiting, unplaced, and a sample held over.
fn holding_everything_waiting(dir: &TestDir) -> Owner<Sculpting, NoUploads> {
    let mut owner = waiting_owner(dir);
    run(&mut owner, 10_000);
    let mut open = stroke(ACROSS, 12);
    open.pop();
    queue(&mut owner, open);
    run(&mut owner, 1);
    let second = stroke(((150.0, 80.0), (170.0, 120.0)), 6);
    queue(&mut owner, second.clone());
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the second waits");
    owner.brush_mut().held.push(second.first().copied());
    assert!(owner.brush().under_way(), "the first is under way");
    assert!(owner.landing() > 1, "and there's plenty held");
    owner
}

#[test]
fn abandoning_forgets_a_waiting_stroke_and_what_is_held() {
    let dir = TestDir::new();
    let mut owner = holding_everything_waiting(&dir);
    let before = positions(owner.project());
    owner.brush_mut().abandon();
    assert_eq!(owner.landing(), 0, "nothing is held");
    assert!(!owner.brush().stroking(), "no stroke is open");
    owner.expose();
    run(&mut owner, 10_000);
    assert_eq!(positions(owner.project()), before, "no tile changed");
    owner.discard().expect("discarded");
}

#[test]
fn discard_with_a_stroke_waiting_leaves_the_saved_tiles() {
    let dir = TestDir::new();
    let owner = holding_everything_waiting(&dir);
    let saved = tiles_on_disk(&dir);
    owner.discard().expect("discarded");
    assert_eq!(recovery_files(&dir), 0, "the work is gone, as asked");
    assert_eq!(tiles_on_disk(&dir), saved, "the project is as it was saved");
}

/// Samples waiting for futures are in the save of a close.
///
/// A stroke over one still landing whose futures can't be had keeps its samples waiting: Save at
/// the close lands the first, places them on the second's own surface, and saves both, the window
/// kept until then.
#[test]
fn samples_waiting_for_futures_are_in_the_save_of_a_close() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    let second = stroke(((150.0, 80.0), (170.0, 120.0)), 6);
    let (early, late) = second.split_at(2);
    queue(&mut owner, early.to_vec());
    run(&mut owner, 1);
    assert_eq!(
        owner.brush().strokes.len(),
        2,
        "the second began over the first"
    );
    owner.brush_mut().knob = Knob::FuturesFail;
    queue(&mut owner, late.to_vec());
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "its samples wait");
    let now = Instant::now();
    assert_eq!(owner.ask_close(now), CloseAnswer::Asked);
    assert!(owner.key(UiKey::Enter, now), "Save");
    let began = Instant::now();
    let mut asked = None;
    while asked.is_none() {
        if owner.save_step() == SaveStep::Waiting {
            std::thread::sleep(Duration::from_millis(1));
        }
        run(&mut owner, 1);
        asked = owner.take_request();
        assert!(began.elapsed() < Duration::from_secs(60), "it ends");
    }
    assert_eq!(asked, Some(WindowRequest::Exit), "saved in the window");
    assert_eq!(owner.landing(), 0, "both landed first");
    assert_eq!(owner.brush().error(), None);
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    let file = ProjectFile::open(&copy_on_disk(&dir)).expect("it opens");
    let read = file.strokes().map(|record| record.expect("a record"));
    let records: Vec<StrokeRecord> = read.collect();
    let counts: Vec<usize> = records.iter().map(StrokeRecord::len).collect();
    assert_eq!(counts, [13, 7], "every sample of both, once");
}
