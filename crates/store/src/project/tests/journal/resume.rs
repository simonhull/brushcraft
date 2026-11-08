//! Restoring yields the open stroke.
//!
//! A restore finds the stroke still landing in the recovery file's journal, with its dabs not
//! landed and its pen-down surface as the save point its batches name holds it, and resuming it
//! lands the rest as the live stroke would have; a journal it can't use leaves the restored tiles
//! alone, and says why.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::file::frame_batch;
use crate::fixture::Edit;
use crate::tiles::PenDownShape;
use crate::{JournalUnusable, OpenStroke};

/// The dab format the tests' app sets.
const FORMAT: u32 = 7;

/// `count` random edits at the finest level alone, with `seed`.
///
/// A stroke of them never changes the level below the tiles it edits, so its pen-down shapes'
/// boxes are those of the tiles' predictions as they are after it.
fn top_edits(mesh: &BaseMesh, count: u32, seed: u64) -> Vec<Edit> {
    let top = Level::new(TOP).unwrap();
    let on_top = random_edits(mesh, TOP, count * u32::from(TOP) * 4, seed)
        .into_iter()
        .filter(|edit| edit.level == top);
    let edits: Vec<Edit> = on_top.take(count as usize).collect();
    assert_eq!(
        edits.len(),
        count as usize,
        "enough edits on the finest level"
    );
    edits
}

/// The tiles `edits` write, in key order.
fn keys_of(mesh: &BaseMesh, edits: &[Edit]) -> BTreeSet<TileKey> {
    edits.iter().flat_map(|edit| edit.keys(mesh)).collect()
}

/// A new project whose app set [`FORMAT`].
fn new_project(dir: &TestDir) -> Project {
    let mut project = Project::new((*mesh()).clone(), config(dir)).unwrap();
    project.set_dab_format(FORMAT);
    project
}

/// Begins a stroke on `project` whose dabs are `edits`, waiting until it may land.
fn begin_with(project: &mut Project, edits: Vec<Edit>) -> Stroke {
    let op = project.begin_stroke_and_wait().unwrap();
    Stroke {
        op,
        edits,
        placed: 0,
        landed: 0,
    }
}

/// Begins a stroke on `project` of `count` dabs on the finest level, with `seed`.
fn begin_on_top(project: &mut Project, count: u32, seed: u64) -> Stroke {
    let edits = top_edits(&project.store().working().mesh(), count, seed);
    begin_with(project, edits)
}

/// The stroke a restored project goes on with: `stroke`'s dabs, `open`'s counts.
fn going_on(stroke: Stroke, open: &OpenStroke) -> Stroke {
    let placed = open.landed() + u32::try_from(open.dabs().len()).unwrap();
    Stroke {
        op: open.op(),
        edits: stroke.edits,
        placed,
        landed: open.landed(),
    }
}

/// The one recovery file in `dir`'s recovery directory, as listed.
fn candidate(dir: &TestDir) -> RecoveryCandidate {
    let mut listed = find_recoveries(&dir.path("recovery")).unwrap();
    assert_eq!(listed.len(), 1, "one recovery file: {listed:?}");
    listed.pop().unwrap()
}

/// Restores the project of the one recovery file in `dir`.
fn restore(dir: &TestDir) -> Project {
    Project::restore(&candidate(dir), config(dir)).unwrap()
}

/// Asserts that `project` evaluates every tile as the dense evaluation of `map`.
fn assert_tiles(project: &mut Project, map: &CorrectionMap, what: &str) {
    let mesh = mesh();
    assert_positions(
        project.store(),
        &dense(&mesh, map),
        &tiles(&mesh, 1, TOP),
        what,
    );
}

/// Crashes `project` (drops it, keeping its recovery file) with a stroke landing: 20 dabs
/// journalled, 8 landed, flushed. Returns the stroke.
fn crash_landing(mut project: Project, map: &mut CorrectionMap, seed: u64) -> Stroke {
    let mut stroke = begin_on_top(&mut project, 20, seed);
    stroke.place(&mut project, 20);
    stroke.land(&mut project, map, 8);
    flush(&mut project);
    drop(project);
    stroke
}

#[test]
fn a_restore_finds_the_stroke_still_landing() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 41);
    let mut restored = restore(&dir);
    assert_tiles(&mut restored, &map, "the tiles of the 8 dabs landed");
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!(open.op(), stroke.op, "its op");
    assert_eq!(open.landed(), 8, "its landed count");
    assert_eq!(open.dabs(), dabs(8..20), "the dabs not landed");
    assert_eq!(open.dab_format(), FORMAT, "their format");
    assert_eq!(open.pen_down(), PenDown::Base, "begun on the file's start");
    let mesh = mesh();
    let changed: BTreeSet<TileKey> = open.pen_down_keys().copied().collect();
    assert_eq!(
        changed,
        keys_of(&mesh, &stroke.edits[..8]),
        "pen-down shapes for exactly the tiles the 8 dabs changed"
    );
    assert!(restored.take_open_stroke().is_none(), "taken once");
    restored.close(false).unwrap();
}

/// The live pen-down shapes of the tiles `stroke`'s first 12 dabs change, kept on `project`'s
/// store before they land.
fn live_shapes(project: &mut Project, stroke: &Stroke) -> BTreeMap<TileKey, PenDownShape> {
    let mesh = project.store().working().mesh();
    let keys: Vec<TileKey> = keys_of(&mesh, &stroke.edits[..12]).into_iter().collect();
    project.store().keep_pen_down(&keys).unwrap();
    let shapes = keys
        .iter()
        .map(|key| (*key, project.store().pen_down_shape(key).unwrap().clone()));
    shapes.collect()
}

/// Lands `stroke` (begun, with `live` its live shapes) 6 dabs a flush to 12 of 30, crashes,
/// restores, resumes it, and asserts each installed shape is its live one, bit for bit, and its
/// pen-down is `pen_down`.
fn assert_shapes_as_found(
    dir: &TestDir,
    mut project: Project,
    mut stroke: Stroke,
    map: &mut CorrectionMap,
    live: &BTreeMap<TileKey, PenDownShape>,
    pen_down: PenDown,
) {
    stroke.place(&mut project, 30);
    stroke.land(&mut project, map, 6);
    flush(&mut project);
    stroke.land(&mut project, map, 6);
    flush(&mut project);
    drop(project);
    let mut restored = restore(dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!(open.pen_down(), pen_down, "the save point its batches name");
    let keys: Vec<TileKey> = open.pen_down_keys().copied().collect();
    let live_keys: Vec<TileKey> = live.keys().copied().collect();
    assert_eq!(keys, live_keys, "the tiles changed");
    restored.resume_stroke(open).unwrap();
    for (key, shape) in live {
        assert_eq!(
            restored.store().pen_down_shape(key),
            Some(shape),
            "{key:?}: the shape as the stroke found it"
        );
    }
    assert_eq!(restored.store().pen_down_tiles(), live.len(), "no other");
    restored.close(false).unwrap();
}

#[test]
fn the_pen_down_shapes_are_the_tiles_at_the_save_point_its_batches_name() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 300, 42);
    flush(&mut project);
    let pen_down = latest(&project);
    assert!(matches!(pen_down, PenDown::At(_)), "{pen_down:?}");
    let stroke = begin_on_top(&mut project, 30, 43);
    let live = live_shapes(&mut project, &stroke);
    assert_shapes_as_found(&dir, project, stroke, &mut map, &live, pen_down);
}

#[test]
fn a_tile_never_in_the_recovery_file_reads_its_pen_down_shape_from_the_project_file() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 300, 44);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    // A stroke since the save writes a few of the tiles the next one changes, so some of them
    // are in the recovery file at its pen-down, and the rest only in the project file.
    let mesh = mesh();
    let edits = top_edits(&mesh, 30, 45);
    let before: Vec<Edit> = edits[..2]
        .iter()
        .map(|edit| Edit {
            c: edit.c * 0.5,
            ..*edit
        })
        .collect();
    project.begin_stroke_and_wait().unwrap();
    apply_dabs(project.store(), &mut map, &before);
    project.end_unrecorded_stroke().unwrap();
    flush(&mut project);
    let pen_down = latest(&project);
    assert!(matches!(pen_down, PenDown::At(_)), "{pen_down:?}");
    let stroke = begin_with(&mut project, edits);
    let live = live_shapes(&mut project, &stroke);
    let in_recovery = |key: &TileKey| project.store.working().contains(key);
    let in_base = |key: &TileKey| project.store.base().unwrap().contains(key);
    assert!(
        live.keys().any(|key| !in_recovery(key) && in_base(key)),
        "a tile read from the project file"
    );
    assert!(
        live.keys().any(in_recovery),
        "a tile read from the recovery file"
    );
    assert_shapes_as_found(&dir, project, stroke, &mut map, &live, pen_down);
}

#[test]
fn a_base_pen_down_reads_the_base_save() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 300, 46);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    let stroke = begin_on_top(&mut project, 30, 47);
    let live = live_shapes(&mut project, &stroke);
    assert!(
        live.keys()
            .any(|key| project.store.base().unwrap().contains(key)),
        "a tile the save holds"
    );
    assert_shapes_as_found(&dir, project, stroke, &mut map, &live, PenDown::Base);
}

#[test]
fn a_restored_stroke_resumes_its_op() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 48);
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(stroke, &open);
    restored.resume_stroke(open).unwrap();
    let op = resumed.op;
    assert_eq!(restored.store().current_op(), Some(op), "its op again");
    assert!(restored.may_land(), "it may land at once");
    resumed.land(&mut restored, &mut map, 4);
    flush(&mut restored);
    let mesh = mesh();
    for key in keys_of(&mesh, &resumed.edits[8..12]) {
        let record = restored.store().working().read_tile(&key).unwrap().unwrap();
        assert_eq!(record.op, op, "{key:?}: the next edits are the stroke's");
    }
    resumed.land(&mut restored, &mut map, 8);
    restored.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut restored);
    assert_tiles(&mut restored, &map, "every dab landed");
    restored.close(true).unwrap();
    let mut again = restore(&dir);
    assert!(again.take_open_stroke().is_none(), "the stroke ended");
    assert_tiles(&mut again, &map, "restored again");
    again.close(false).unwrap();
}

#[test]
fn a_stroke_resumes_only_where_it_was_found_and_before_another_op() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 49);
    let mut restored = restore(&dir);
    let refused = restored.take_open_stroke().unwrap().unwrap();
    let op = refused.op();
    let begun = restored.begin_stroke().unwrap();
    assert_ne!(begun, op, "a new op");
    assert!(
        matches!(restored.resume_stroke(refused), Err(ProjectError::NotResumable(not)) if not == op),
        "another op began since the restore"
    );
    restored.end_unrecorded_stroke().unwrap();
    assert_eq!(restored.store().pen_down_tiles(), 0, "no shape installed");
    restored.close(true).unwrap();

    // Found in another project: refused there.
    let mut first = restore(&dir);
    let elsewhere = first.take_open_stroke().unwrap().unwrap();
    first.close(true).unwrap();
    let mut second = restore(&dir);
    assert!(
        matches!(
            second.resume_stroke(elsewhere),
            Err(ProjectError::NotResumable(_))
        ),
        "another project's"
    );
    let own = second.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(stroke, &own);
    second.resume_stroke(own).unwrap();
    resumed.land(&mut second, &mut map, 12);
    second.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut second, &map, "its own lands");
    second.close(false).unwrap();
}

/// Crashes a project with stroke `stroke` of 20 dabs begun with `seed`, as `act` leaves it, with
/// `map` the dabs landed, then asserts a restore finds no stroke to land, and its tiles.
fn assert_no_open_stroke(
    seed: u64,
    act: impl FnOnce(&mut Project, &mut Stroke, &mut CorrectionMap),
) {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut stroke = begin_on_top(&mut project, 20, seed);
    act(&mut project, &mut stroke, &mut map);
    drop(project);
    let mut restored = restore(&dir);
    assert!(restored.take_open_stroke().is_none(), "nothing to land");
    assert_tiles(&mut restored, &map, "its tiles");
    restored.close(false).unwrap();
}

#[test]
fn a_stroke_that_had_landed_restores_no_open_stroke() {
    // Ended: its last batch counts every dab landed.
    assert_no_open_stroke(50, |project, ended, map| {
        ended.place(project, 10);
        flush(project);
        ended.land(project, map, 10);
        project.end_unrecorded_stroke().unwrap();
        crash::poll_until_idle(project);
        let last = batches(project).pop().unwrap();
        assert_eq!(last.landed(), last.placed(), "a closing batch");
    });
    // Open, every dab placed landed: nothing to land.
    assert_no_open_stroke(51, |project, landed, map| {
        landed.place(project, 10);
        landed.land(project, map, 10);
        landed.place(project, 2);
        flush(project);
        landed.land(project, map, 2);
        flush(project);
    });
}

/// Asserts that `project`'s restore found a journal and didn't use it, for `why`, and that its
/// tiles are those of `map`, the dabs landed. Closes it.
fn assert_tiles_alone(mut project: Project, map: &CorrectionMap, why: &JournalUnusable) {
    assert_eq!(
        project.take_open_stroke().map(|found| found.map(|_| ())),
        Some(Err(why.clone())),
        "unusable, and why"
    );
    assert_tiles(&mut project, map, "the tiles restored alone");
    project.close(false).unwrap();
}

/// Asserts that batches appended after a crash make the journal unusable for `why`.
///
/// A project crashes with a stroke landing ([`crash_landing`], its dabs from `seed`), and
/// `extra(op)`, the stroke's op given, is appended to its recovery file with a save point of
/// their own. A restore made with the configuration `config` gives must restore the tiles of the
/// dabs landed alone.
fn assert_appended_unusable(
    seed: u64,
    config: fn(&TestDir) -> ProjectConfig,
    extra: impl FnOnce(OpId) -> Vec<DabBatch>,
    why: &JournalUnusable,
) {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, seed);
    let path = candidate(&dir).path;
    let mut file = ProjectFile::open(&path).unwrap();
    let framed: Vec<Vec<u8>> = extra(stroke.op)
        .iter()
        .map(|batch| frame_batch(batch).unwrap())
        .collect();
    file.append_batches(&framed).unwrap();
    file.save().unwrap();
    drop(file);
    let restored = Project::restore(&candidate(&dir), config(&dir)).unwrap();
    assert_tiles_alone(restored, &map, why);
}

/// The batch of `op` holding dabs `range`, `landed` of them landed, its pen-down at the file's
/// start and its rows in [`FORMAT`].
fn batch_of(op: OpId, range: Range<u32>, landed: u32) -> DabBatch {
    batch(op, range, landed).with_dab_format(FORMAT)
}

/// The op after `op`: another stroke's.
fn next_op(op: OpId) -> OpId {
    OpId::new(op.site(), op.seq() + 1).unwrap()
}

#[test]
fn a_gap_or_overlap_in_the_batches_restores_the_tiles_alone_and_says_why() {
    // The crashed stroke's batch holds dabs 8..20.
    let gap = JournalUnusable::Gap {
        expected: 20,
        found: 25,
    };
    assert_appended_unusable(52, config, |op| vec![batch_of(op, 25..27, 8)], &gap);
    let overlap = JournalUnusable::Gap {
        expected: 20,
        found: 15,
    };
    assert_appended_unusable(53, config, |op| vec![batch_of(op, 15..27, 8)], &overlap);
    // The stroke's first batch read must start at or before the landed count.
    let late = JournalUnusable::Gap {
        expected: 2,
        found: 4,
    };
    let other = |op| vec![batch_of(next_op(op), 4..10, 2)];
    assert_appended_unusable(54, config, other, &late);
}

/// Crashes a stroke whose batches are dabs 0..20 then 20..25 (3 landed), damages the `nth`
/// batch, and asserts the restore finds the journal damaged.
fn assert_damaged(nth: usize) {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut stroke = begin_on_top(&mut project, 30, 55);
    stroke.place(&mut project, 20);
    flush(&mut project);
    stroke.place(&mut project, 5);
    stroke.land(&mut project, &mut map, 3);
    flush(&mut project);
    let op = stroke.op;
    assert_eq!(
        batches(&mut project),
        vec![batch_of(op, 0..20, 0), batch_of(op, 20..25, 3)],
        "the second needs the first"
    );
    let loc = project
        .store()
        .working()
        .batch_locations()
        .nth(nth)
        .unwrap();
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    damage(&path, loc);
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::Damaged);
}

#[test]
fn a_damaged_batch_restores_the_tiles_alone_and_says_why() {
    assert_damaged(0);
    assert_damaged(1);
}

/// Crashes a stroke of 20 dabs begun with `seed`, as `act` leaves it, with `map` the dabs landed,
/// and asserts the restore finds its pen-down unknown.
fn assert_pen_down_unknown(
    seed: u64,
    act: impl FnOnce(&TestDir, &mut Project, &mut Stroke, &mut CorrectionMap),
) {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut stroke = begin_on_top(&mut project, 20, seed);
    act(&dir, &mut project, &mut stroke, &mut map);
    let named: Vec<PenDown> = batches(&mut project)
        .iter()
        .map(DabBatch::pen_down)
        .collect();
    assert_eq!(named.last(), Some(&PenDown::Unknown), "named unknown");
    drop(project);
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::PenDownUnknown);
}

#[test]
fn an_unknown_pen_down_restores_the_tiles_alone_and_says_why() {
    // A roll-over in the middle of the stroke: the fresh file holds no save point of the surface
    // it began on.
    assert_pen_down_unknown(56, |_, project, rolled, map| {
        rolled.place(project, 10);
        flush(project);
        rolled.land(project, map, 4);
        project.writer.as_ref().unwrap().fail_syncs(0, 1);
        project.finish_flushes().unwrap();
        flush(project);
    });
    // Begun before the recovery file's base save: a save in the middle of the stroke.
    assert_pen_down_unknown(57, |dir, project, saved, map| {
        saved.place(project, 10);
        saved.land(project, map, 10);
        assert_saved(project.save_as(&dir.path("p.bcrf")));
        saved.place(project, 2);
        flush(project);
    });
}

#[test]
fn a_pen_down_save_point_not_in_the_file_restores_the_tiles_alone_and_says_why() {
    let missing =
        |op| vec![batch_at(next_op(op), PenDown::At(12_345), 0..5, 0).with_dab_format(FORMAT)];
    assert_appended_unusable(58, config, missing, &JournalUnusable::PenDownMissing);
}

#[test]
fn batches_of_one_op_disagreeing_on_pen_down_or_format_restore_the_tiles_alone() {
    let mismatch = JournalUnusable::Mismatch;
    let pen_down = |op| vec![batch_at(op, PenDown::Unknown, 20..22, 8).with_dab_format(FORMAT)];
    assert_appended_unusable(59, config, pen_down, &mismatch);
    let format = |op| vec![batch(op, 20..22, 8).with_dab_format(FORMAT + 1)];
    assert_appended_unusable(60, config, format, &mismatch);
    let landed_back = |op| vec![batch_of(op, 20..22, 5)];
    assert_appended_unusable(61, config, landed_back, &mismatch);
}

#[test]
fn an_unset_dab_format_restores_the_tiles_alone() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let project = Project::new((*mesh()).clone(), config(&dir)).unwrap();
    crash_landing(project, &mut map, 62);
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::UnsetFormat);
}

#[test]
fn a_journal_larger_than_the_budget_restores_the_tiles_alone() {
    /// A restore on the tight budget, whose pools hold a few MB.
    fn tight_config(dir: &TestDir) -> ProjectConfig {
        ProjectConfig {
            store: crate::fixture::tight(),
            recovery_dir: dir.path("recovery"),
        }
    }
    let pools = crate::fixture::pools(&crate::fixture::tight());
    let rows = u32::try_from(pools / DAB_BYTES).unwrap() + 1;
    let past = |op| vec![batch_of(op, 20..20 + rows, 8)];
    assert_appended_unusable(63, tight_config, past, &JournalUnusable::TooLarge);
}

#[test]
fn a_restored_journal_journals_on_with_its_own_dab_format() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 64);
    let mut restored = restore(&dir);
    // The app sets its format after the restore returns: the stroke's rows keep theirs.
    restored.set_dab_format(FORMAT + 2);
    let open = restored.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(stroke, &open);
    restored.resume_stroke(open).unwrap();
    let op = resumed.op;
    resumed.land(&mut restored, &mut map, 4);
    flush(&mut restored);
    let mut written = vec![batch_of(op, 8..20, 8), batch_of(op, 20..20, 12)];
    assert_eq!(
        batches(&mut restored),
        written,
        "the count alone: the rows not landed are in the file already"
    );
    flush(&mut restored);
    assert_eq!(batches(&mut restored), written, "once");
    resumed.land(&mut restored, &mut map, 8);
    restored.end_unrecorded_stroke().unwrap();
    crash::poll_until_idle(&mut restored);
    written.push(batch_of(op, 20..20, 20));
    assert_eq!(batches(&mut restored), written, "its last count");
    // The next stroke takes the format the app set.
    let pen_down = latest(&restored);
    let mut next = begin_on_top(&mut restored, 20, 65);
    next.place(&mut restored, 3);
    flush(&mut restored);
    written.push(batch_at(next.op, pen_down, 0..3, 0).with_dab_format(FORMAT + 2));
    assert_eq!(batches(&mut restored), written, "a new stroke's");
    restored.close(false).unwrap();
}
