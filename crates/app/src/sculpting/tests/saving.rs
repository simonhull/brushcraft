//! Ctrl+S end to end, with no GPU.
//!
//! What a save holds (the dabs still catching up, the stroke the pen is drawing, not a stroke begun
//! after it), what the title says, a save that fails, and a close meanwhile.

use std::path::PathBuf;

use super::*;
use crate::saving::{SaveStep, Stage};

/// Where the tests' projects are saved in `dir`.
fn path_in(dir: &TestDir) -> PathBuf {
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

/// Runs the save under way as the window's passes do, a step and then a frame, until it ends.
fn finish_save(owner: &mut Owner<Sculpting, NoUploads>) {
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
fn recovery_files(dir: &TestDir) -> usize {
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
        owner.save_step();
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

/// A stroke waiting for the one before it when Ctrl+S comes was drawn before it: it's in the save.
#[test]
fn a_stroke_waiting_behind_another_at_ctrl_s_is_in_the_save() {
    let dir = TestDir::new();
    let mut owner = slow_owner(&dir);
    run(&mut owner, 10_000);
    queue(&mut owner, stroke(ACROSS, 12));
    run(&mut owner, 1);
    queue(&mut owner, stroke(((150.0, 80.0), (170.0, 120.0)), 6));
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the second waits");
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
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
        owner.save_step();
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
    let mut steps = 0;
    while !owner.ready_to_close() {
        owner.close_step().expect("a step");
        owner.save_step();
        steps += 1;
        assert!(steps < 100_000, "it ends");
    }
    assert_eq!(owner.save_stage(), None, "saved");
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 2, "both strokes");
    assert_eq!(recovery_files(&dir), 0, "nothing unsaved");
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
        owner.save_step();
        steps += 1;
        assert!(steps < 100_000, "it ends");
    }
    assert!(!owner.project().has_unsaved_edits());
    owner.close().expect("closed");
    assert_eq!(records_on_disk(&dir), 1);
}
