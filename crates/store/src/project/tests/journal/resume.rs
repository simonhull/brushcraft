//! Restoring yields the open stroke.
//!
//! A restore finds the stroke still landing in the recovery file's journal, with its dabs not
//! landed and its pen-down surface as the save point its batches name holds it, all exact, and
//! resuming it lands the rest with its own op and its journal going on from its landed count; a
//! journal it can't use leaves the restored tiles alone, and says why.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::file::frame_batch;
use crate::fixture::Edit;
use crate::tiles::PenDownShape;
use crate::{JournalUnusable, OpenStroke};

mod roll_over;

/// The dab format the tests' app sets.
const FORMAT: u32 = 7;

/// `count` random edits at level `level` alone, with `seed`.
///
/// A stroke of them never changes the level below the tiles it edits, so its pen-down shapes'
/// boxes are those of the tiles' predictions as they are after it.
fn level_edits(mesh: &BaseMesh, level: u8, count: u32, seed: u64) -> Vec<Edit> {
    let wanted = Level::new(level).unwrap();
    let on_level = random_edits(mesh, level, count * u32::from(level) * 4, seed)
        .into_iter()
        .filter(|edit| edit.level == wanted);
    let edits: Vec<Edit> = on_level.take(count as usize).collect();
    assert_eq!(edits.len(), count as usize, "enough edits on level {level}");
    edits
}

/// `count` random edits at the finest level alone, with `seed` (see [`level_edits`]).
fn top_edits(mesh: &BaseMesh, count: u32, seed: u64) -> Vec<Edit> {
    level_edits(mesh, TOP, count, seed)
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
fn a_stroke_below_the_finest_level_restores_its_shapes_as_found() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 300, 66);
    flush(&mut project);
    let pen_down = latest(&project);
    let edits = level_edits(&mesh(), TOP - 1, 30, 67);
    let stroke = begin_with(&mut project, edits);
    let live = live_shapes(&mut project, &stroke);
    assert!(
        live.keys().all(|key| key.level().get() == TOP - 1),
        "the level below the finest"
    );
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
    crash_landing(new_project(&dir), &mut map, 49);
    let mut restored = restore(&dir);
    let refused = restored.take_open_stroke().unwrap().unwrap();
    let op = refused.op();
    let begun = restored.begin_stroke().unwrap();
    assert_ne!(begun, op, "a new op");
    let not = restored.resume_stroke(refused).unwrap_err();
    assert!(
        matches!(not.error(), ProjectError::NotResumable(named) if *named == op),
        "another op began since the restore: {not:?}"
    );
    restored.end_unrecorded_stroke().unwrap();
    assert_eq!(restored.store().pen_down_tiles(), 0, "no shape installed");
    restored.close(true).unwrap();
    let mut closed = restore(&dir);
    assert!(closed.take_open_stroke().is_none(), "the next op closed it");
    closed.close(false).unwrap();

    // Found in another project: refused there.
    let other = TestDir::new();
    let stroke = crash_landing(new_project(&other), &mut map, 49);
    let mut first = restore(&other);
    let elsewhere = first.take_open_stroke().unwrap().unwrap();
    first.close(true).unwrap();
    let mut second = restore(&other);
    let not_here = second.resume_stroke(elsewhere).unwrap_err();
    assert!(
        matches!(not_here.error(), ProjectError::NotResumable(_)),
        "another project's: {not_here:?}"
    );
    let own = second.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(stroke, &own);
    second.resume_stroke(own).unwrap();
    resumed.land(&mut second, &mut map, 12);
    second.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut second, &map, "its own lands");
    second.close(false).unwrap();
}

/// An op handed out since the restore refuses the stroke found, though none is current again:
/// one begun and ended, or one reserved and let go.
#[test]
fn a_stroke_resumes_before_any_op_is_handed_out_even_one_ended_or_dropped() {
    let mut map = CorrectionMap::new();
    let ended = TestDir::new();
    crash_landing(new_project(&ended), &mut map, 49);
    let mut restored = restore(&ended);
    let found = restored.take_open_stroke().unwrap().unwrap();
    let op = found.op();
    restored.store().begin_op().unwrap();
    restored.store().end_op();
    assert_eq!(restored.store().current_op(), None, "no op is current");
    let not = restored.resume_stroke(found).unwrap_err();
    assert!(
        matches!(not.error(), ProjectError::NotResumable(named) if *named == op),
        "an op began and ended since the restore: {not:?}"
    );
    restored.close(true).unwrap();

    let dropped = TestDir::new();
    crash_landing(new_project(&dropped), &mut map, 49);
    let mut again = restore(&dropped);
    let lost = again.take_open_stroke().unwrap().unwrap();
    let reserved = again.store().reserve_op().unwrap();
    again.store().drop_reserved_op(reserved);
    let behind = again.resume_stroke(lost).unwrap_err();
    assert!(
        matches!(behind.error(), ProjectError::NotResumable(_)),
        "an op was reserved since the restore: {behind:?}"
    );
    again.close(true).unwrap();
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
    // A roll-over in the middle of a stroke that kept no pen-down shapes of the tiles it changed
    // (the store's own edits keep none): nothing in memory holds the surface it began on, so
    // the fresh file holds no save point of it.
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

#[test]
fn a_stroke_never_resumed_is_closed_by_the_next_op_and_not_offered_again() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 68);
    let mut restored = restore(&dir);
    // The app drops the stroke, and edits on with an op of its own, journalling nothing.
    drop(restored.take_open_stroke().unwrap().unwrap());
    restored.begin_stroke_and_wait().unwrap();
    let edits = random_edits(&mesh(), TOP, 40, 69);
    apply_dabs(restored.store(), &mut map, &edits);
    restored.end_unrecorded_stroke().unwrap();
    flush(&mut restored);
    let op = stroke.op;
    assert_eq!(
        batches(&mut restored),
        vec![batch_of(op, 8..20, 8), batch_of(op, 20..20, 20)],
        "closed in the file, its dabs not landed dropped"
    );
    drop(restored);
    let mut again = restore(&dir);
    assert!(again.take_open_stroke().is_none(), "not offered again");
    assert_tiles(&mut again, &map, "the later op's edits stand");
    again.close(false).unwrap();
}

#[test]
fn an_edit_of_a_later_op_makes_the_journal_its_batches_leave_open_unusable() {
    // Edits of a later op with no stroke begun: no closing batch, but the file's head says the
    // later op's edits stand over the stroke, which can't be finished.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    crash_landing(new_project(&dir), &mut map, 70);
    let mut restored = restore(&dir);
    assert!(restored.take_open_stroke().is_some(), "found");
    restored.store().begin_op().unwrap();
    let edits = random_edits(&mesh(), TOP, 40, 71);
    apply_dabs(restored.store(), &mut map, &edits);
    flush(&mut restored);
    drop(restored);
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::Ended);
}

#[test]
fn a_later_op_of_another_site_leaves_the_stroke_open() {
    // Ops of different sites don't order by time: the head says nothing, and with no closing
    // batch the stroke is still open.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 92);
    let path = candidate(&dir).path;
    let mut file = ProjectFile::open(&path).unwrap();
    let key = tiles(&mesh(), 1, 1)[0];
    let other = OpId::new(stroke.op.site() + 1, 0).unwrap();
    assert!(other > stroke.op, "greater, as another site's");
    let zero = geom::CorrectionTile::zero(key.side()).unwrap();
    file.write_displacement(key, other, &zero).unwrap();
    file.save().unwrap();
    drop(file);
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!(
        (open.op(), open.landed()),
        (stroke.op, 8),
        "the stroke, still open"
    );
    restored.close(false).unwrap();
}

#[test]
fn a_save_in_the_middle_of_a_stroke_leaves_its_journal_unusable() {
    // Its later edits are a new op's, its batches the old one's.
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    let mut saved = begin_on_top(&mut project, 20, 72);
    saved.place(&mut project, 10);
    saved.land(&mut project, &mut map, 10);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    saved.place(&mut project, 4);
    saved.land(&mut project, &mut map, 2);
    flush(&mut project);
    let last = batches(&mut project).pop().unwrap();
    assert_eq!(last.op(), saved.op, "the old op's batch");
    assert!(last.landed() < last.placed(), "dabs not landed");
    drop(project);
    // The new op's edits stand over it.
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::Ended);
}

#[test]
fn a_resume_that_fails_leaves_the_project_as_it_was() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 73);
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    let written = batches(&mut restored);
    restored.fail_next_resume = true;
    let failed = restored.resume_stroke(open).unwrap_err();
    assert!(
        matches!(failed.error(), ProjectError::Cache(_)),
        "{failed:?}"
    );
    assert_eq!(restored.store().current_op(), None, "no op");
    assert_eq!(restored.store().pen_down_tiles(), 0, "no shape");
    assert!(!restored.store().is_landing(), "no journal");
    flush(&mut restored);
    assert_eq!(batches(&mut restored), written, "nothing written of it");
    // Given back, it resumes.
    let given_back = failed.into_stroke();
    let mut resumed = going_on(stroke, &given_back);
    restored.resume_stroke(given_back).unwrap();
    resumed.land(&mut restored, &mut map, 12);
    restored.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut restored, &map, "every dab landed");
    restored.close(false).unwrap();
}

#[test]
fn a_resume_whose_shape_fails_to_install_gives_back_its_charge() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let stroke = crash_landing(new_project(&dir), &mut map, 93);
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    assert!(open.pen_down_keys().count() > 1, "shapes to install");
    let written = batches(&mut restored);
    let key = *open.pen_down_keys().next().unwrap();
    drop(restored.store().pin(&[key]).unwrap());
    let limit = restored.store().pools_limit();
    // The second shape fails, after the first is charged.
    restored.fail_install_at = Some(1);
    let failed = restored.resume_stroke(open).unwrap_err();
    assert!(
        matches!(failed.error(), ProjectError::Cache(_)),
        "{failed:?}"
    );
    assert_eq!(restored.store().pen_down_tiles(), 0, "no shape");
    assert_eq!(restored.store().pen_down_bytes(), 0, "none charged");
    drop(restored.store().pin(&[key]).unwrap());
    assert_eq!(
        restored.store().pools_limit(),
        limit,
        "the pools' room as it was"
    );
    assert_eq!(restored.store().current_op(), None, "no op");
    assert!(!restored.store().is_landing(), "no journal");
    flush(&mut restored);
    assert_eq!(batches(&mut restored), written, "nothing written of it");
    let given_back = failed.into_stroke();
    let mut resumed = going_on(stroke, &given_back);
    restored.resume_stroke(given_back).unwrap();
    resumed.land(&mut restored, &mut map, 12);
    restored.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut restored, &map, "every dab landed");
    restored.close(false).unwrap();
}

#[test]
fn a_journal_with_no_dab_landed_is_unsaved_work_and_restores() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 100, 94);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    assert!(!project.has_unsaved_edits(), "saved");
    let mut landing = begin_on_top(&mut project, 10, 95);
    landing.place(&mut project, 10);
    assert!(project.has_unsaved_edits(), "dabs placed, none landed");
    flush(&mut project);
    drop(project);
    // The file holds no tile edit, only the stroke's journal: it isn't empty.
    let found = candidate(&dir);
    assert_eq!(found.status, CandidateStatus::Restorable, "on the save");
    assert_eq!(found.last_edit, Some(landing.op), "the stroke is its work");
    let mut restored = restore(&dir);
    assert!(restored.has_unsaved_edits(), "the stroke to resume");
    let taken = restored.take_open_stroke().unwrap().unwrap();
    assert_eq!(taken.landed(), 0, "none landed");
    assert_eq!(taken.dabs(), dabs(0..10), "every dab");
    drop(taken);
    assert!(restored.has_unsaved_edits(), "taken, not resumed");
    let keep = restored.has_unsaved_edits();
    restored.close(keep).unwrap();
    let mut again = restore(&dir);
    let open = again.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(landing, &open);
    again.resume_stroke(open).unwrap();
    resumed.land(&mut again, &mut map, 10);
    again.end_unrecorded_stroke().unwrap();
    assert_tiles(&mut again, &map, "every dab landed");
    again.close(false).unwrap();
}

/// Crashes a saved project with a stroke of 10 dabs placed and none landed, flushed: its
/// recovery file holds the stroke's journal and no edit. Returns the stroke.
fn crash_unlanded(dir: &TestDir, map: &mut CorrectionMap, seed: u64) -> Stroke {
    let mut project = new_project(dir);
    stroke(&mut project, map, 100, seed);
    assert_saved(project.save_as(&dir.path("p.bcrf")));
    let mut landing = begin_on_top(&mut project, 10, seed + 1);
    landing.place(&mut project, 10);
    flush(&mut project);
    drop(project);
    landing
}

#[test]
fn a_resumed_stroke_that_landed_nothing_never_has_its_op_handed_out_again() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let landing = crash_unlanded(&dir, &mut map, 96);
    let mut restored = restore(&dir);
    let open = restored.take_open_stroke().unwrap().unwrap();
    let mut resumed = going_on(landing, &open);
    restored.resume_stroke(open).unwrap();
    resumed.land(&mut restored, &mut map, 10);
    restored.end_unrecorded_stroke().unwrap();
    let next = restored.begin_stroke().unwrap();
    assert!(
        next > resumed.op,
        "{next:?} after the resumed {:?}",
        resumed.op
    );
    restored.end_unrecorded_stroke().unwrap();
    restored.close(false).unwrap();
}

#[test]
fn a_restored_stroke_never_resumed_that_landed_nothing_never_has_its_op_handed_out_again() {
    // The next stroke begins straight away, the restored one never resumed.
    let dir = TestDir::new();
    let landing = crash_unlanded(&dir, &mut CorrectionMap::new(), 97);
    let mut restored = restore(&dir);
    let next = restored.begin_stroke().unwrap();
    assert!(
        next > landing.op,
        "{next:?} after the restored {:?}",
        landing.op
    );
    restored.end_unrecorded_stroke().unwrap();
    restored.close(false).unwrap();
}

#[test]
fn a_damaged_pen_down_tile_record_restores_the_tiles_alone() {
    let dir = TestDir::new();
    let mut map = CorrectionMap::new();
    let mut project = new_project(&dir);
    stroke(&mut project, &mut map, 300, 74);
    flush(&mut project);
    let mut landing = begin_on_top(&mut project, 20, 75);
    let mesh = mesh();
    let changed = keys_of(&mesh, &landing.edits[..8]);
    let found = changed
        .iter()
        .find_map(|key| project.store().working().locate(key))
        .expect("a tile the stroke changes is in the recovery file at its pen-down");
    landing.place(&mut project, 20);
    landing.land(&mut project, &mut map, 8);
    flush(&mut project);
    let path = project.recovery_path().unwrap().to_path_buf();
    drop(project);
    // The pen-down record: the restored state names the stroke's own since.
    damage(&path, found);
    assert_tiles_alone(restore(&dir), &map, &JournalUnusable::Damaged);
}

#[test]
fn pen_down_shapes_past_the_budget_restore_the_tiles_alone() {
    /// A restore on the tight budget, whose pools hold a few MB.
    fn tight_config(dir: &TestDir) -> ProjectConfig {
        ProjectConfig {
            store: crate::fixture::tight(),
            recovery_dir: dir.path("recovery"),
        }
    }
    // The rows fit, with less room than the shapes need left beside them.
    let mesh = mesh();
    let edits = top_edits(&mesh, 20, 76);
    let shapes: usize = keys_of(&mesh, &edits[..8])
        .iter()
        .map(|key| PenDownShape::bytes_of(key.side()))
        .sum();
    let pools = crate::fixture::pools(&crate::fixture::tight());
    let rows = u32::try_from((pools - shapes / 2) / DAB_BYTES).unwrap() - 12;
    let past = |op| vec![batch_of(op, 20..20 + rows, 8)];
    assert_appended_unusable(76, tight_config, past, &JournalUnusable::TooLarge);
}
