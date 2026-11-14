//! A chain of strokes a restore found still landing: one landing, and one begun over it, its dabs
//! journalled, none landed (the stroke-while-landing spec, "Showing and landing": restore).
//!
//! The first is taken on and shown at once; the next resumes once it has landed and ended, on the
//! surface it left, and is shown then; a `Down` meanwhile waits until every one has resumed, so
//! none is lost to its fresh op; and strokes the brush can't take make the note say so, in the
//! plural once there are several.

use super::resume::{Corrections, corrections, resumed_in, steps_apart};
use super::*;
use crate::OpenNote;
use crate::dialogs::UNFINISHED_STROKE;
use crate::sculpting::tests::saving::finish_save;

/// What a crash in the middle of a chain left in a recovery file: the first stroke partly landed,
/// the second begun over it, placed and journalled, none of it landed.
struct CrashedChain {
    /// The project's directory, its recovery file in `recovery`.
    dir: TestDir,
    /// The first stroke's op.
    first: OpId,
    /// The second stroke's op.
    second: OpId,
    /// Every finest tile's corrections in the same run never crashed, once both have landed.
    whole: BTreeMap<TileKey, Corrections>,
}

/// The two strokes of the chain, in the formats their rows are journalled in.
#[derive(Debug, Clone, Copy)]
struct Formats {
    /// The first stroke's rows' layout.
    first: u32,
    /// The second's.
    second: u32,
}

/// Both strokes in the brush's layout.
const BRUSH_FORMATS: Formats = Formats {
    first: stroke::DAB_FORMAT,
    second: stroke::DAB_FORMAT,
};

/// A chain drawn in a [`Journalling`] run, and its two ops.
///
/// The first stroke along [`ACROSS`] shown, its cut taken, `landed` of its dabs landed, then the
/// second ([`below`]) begun over it and its journal cut.
fn chain_in(dir: TestDir, formats: Formats, landed: usize) -> (Journalling, OpId, OpId) {
    let mut run = Journalling::in_dir(dir, |brush| brush);
    run.project.set_dab_format(formats.first);
    let shown = run.frame(&stroke(ACROSS, 12));
    assert!(shown.dabs_shown > landed + 2, "{shown:?}");
    run.cut();
    for _ in 0..landed {
        assert_eq!(run.land_one().dabs_applied, 1, "a dab at a time");
    }
    run.project.set_dab_format(formats.second);
    run.frame(&below());
    let ops: Vec<OpId> = run
        .brush
        .strokes
        .iter()
        .map(|stroke| stroke.op.id())
        .collect();
    let [first, second] = ops.as_slice() else {
        panic!("the second began over the first: {ops:?}");
    };
    let (first, second) = (*first, *second);
    run.cut();
    (run, first, second)
}

/// A crash with the chain `drawn` draws open, and the same chain landed whole, never crashed.
///
/// Returns the crash's directory, its recovery file in `recovery`, and every finest tile's
/// corrections of the whole one.
fn crash_with(drawn: impl Fn(TestDir) -> Journalling) -> (TestDir, BTreeMap<TileKey, Corrections>) {
    let mut whole_run = drawn(TestDir::new());
    let flushing = context(
        &mut whole_run.project,
        whole_run.picker.index_mut(),
        &whole_run.camera,
    );
    whole_run.brush.flush(flushing).expect("flushed");
    assert!(!whole_run.brush.stroking(), "every stroke landed");
    let whole = corrections(&mut whole_run.project);
    let Journalling {
        dir: whole_dir,
        project: whole_project,
        ..
    } = whole_run;
    whole_project.close(false).expect("closed");
    drop(whole_dir);

    let Journalling {
        dir,
        project,
        mut brush,
        picker,
        ..
    } = drawn(TestDir::new());
    // A crash: nothing closes the project, and the later strokes' reserved ops go with its store.
    brush.abandon();
    drop((brush, picker));
    drop(project);
    (dir, whole)
}

/// A crash with [`chain_in`]'s chain open, and the same chain landed whole, never crashed.
fn crash_mid_chain(formats: Formats, landed: usize) -> CrashedChain {
    let ops = std::cell::Cell::new(None);
    let (dir, whole) = crash_with(|dir| {
        let (run, first, second) = chain_in(dir, formats, landed);
        ops.set(Some((first, second)));
        run
    });
    let (first, second) = ops.get().expect("drawn");
    CrashedChain {
        dir,
        first,
        second,
        whole,
    }
}

/// A third stroke, over both of [`chain_in`]'s: `Down` to `Up`.
fn over_both() -> Vec<PenSample> {
    stroke(((140.0, 95.0), (180.0, 135.0)), 10)
}

/// The ops of the strokes `owner`'s brush has under way, oldest first.
fn under_way(owner: &Owner<Sculpting, NoUploads>) -> Vec<OpId> {
    let strokes = owner.brush().strokes.iter();
    strokes.map(|stroke| stroke.op.id()).collect()
}

/// Makes frames of `owner` one at a time until its brush has nothing under way or waiting: every
/// op seen under way, in the order first seen.
fn landed_frame_by_frame(owner: &mut Owner<Sculpting, NoUploads>) -> Vec<OpId> {
    let mut seen: Vec<OpId> = under_way(owner);
    let began = Instant::now();
    while owner.brush().stroking() {
        run(owner, 1);
        let now = under_way(owner);
        let fresh = now.into_iter().filter(|op| !seen.contains(op));
        let fresh: Vec<OpId> = fresh.collect();
        seen.extend(fresh);
        assert!(began.elapsed() < Duration::from_secs(120), "it lands");
    }
    seen
}

/// The restore note `owner` shows: its title and detail lines.
fn note_of(owner: &mut Owner<Sculpting, NoUploads>) -> Vec<String> {
    let note = OpenNote::Restored {
        path: "recovery.bcrf".into(),
        when: std::time::SystemTime::now(),
    };
    owner.note(&note, Instant::now());
    let bar = owner.bar().expect("the note");
    assert_eq!(bar.title(), "Your unsaved work is back");
    bar.details().to_vec()
}

#[test]
fn a_restored_chain_shows_the_first_at_once_and_lands_every_stroke_in_order() {
    let crashed = crash_mid_chain(BRUSH_FORMATS, 3);
    // A dab at a time, so no frame lands the second whole as it resumes: each is seen under way.
    let mut owner = resumed_in(&crashed.dir, |project| {
        Sculpting::new(project).with_cost(DabCost::one_at_a_time())
    });
    assert_eq!(under_way(&owner), [crashed.first], "the first taken on");
    assert!(owner.brush().catching_up() > 0, "and shown at once");
    let seen = landed_frame_by_frame(&mut owner);
    assert_eq!(
        seen,
        [crashed.first, crashed.second],
        "the second resumed once the first had landed: {:?}",
        owner.brush().error()
    );
    assert_eq!(owner.brush().error(), None);
    assert_eq!(owner.brush().drops(), 0, "nothing dropped");
    let details = note_of(&mut owner);
    assert_eq!(details.len(), 1, "the note's time alone: {details:?}");
    let got = corrections(owner.project());
    let apart = coarse_steps_apart(&got, &crashed.whole);
    println!("a restored chain: {apart} coarse steps from the chain never crashed");
    assert!(
        apart <= CHAIN_BOUND,
        "{apart} coarse steps from the chain never crashed: every dab of both, once"
    );
    owner.close().expect("closed");
}

/// How far a restored chain of two may land from the same chain never crashed, in each tile's
/// coarsest quantization step ([`coarse_steps_apart`]).
///
/// The first stroke lands within the restore's bound of its live landing (1.5 steps, the kill
/// test's). The second lands on what the first left, so it starts from that surface, and its
/// own landing rounds once more, by at most half a step: the spec's bound holds per resumed
/// stroke, on the surface it lands on. Its dabs, measured on a surface a step or so off, move a
/// vertex a fraction of a step otherwise. So 1.5 + 1, in the coarsest step: a tile's finest
/// component step can be several times finer (a tile flat along one axis), and a difference of
/// one coarse step is several of it.
const CHAIN_BOUND: f32 = 2.5;

/// How far `got`'s corrections are from `expected`'s at most, in the coarsest quantization step
/// of each tile (the largest of its components' steps, in either): over every component of every
/// vertex.
fn coarse_steps_apart(
    got: &BTreeMap<TileKey, Corrections>,
    expected: &BTreeMap<TileKey, Corrections>,
) -> f32 {
    let coarsest = |key: &TileKey| -> Corrections {
        let (scales, values) = got.get(key).cloned().unwrap_or_default();
        let theirs = expected
            .get(key)
            .map_or([0.0; 3], |(their_scales, _)| *their_scales);
        let step = scales.into_iter().chain(theirs).fold(0.0_f32, f32::max);
        ([step; 3], values)
    };
    let keys = got.keys();
    let coarse: BTreeMap<TileKey, Corrections> = keys.map(|key| (*key, coarsest(key))).collect();
    let other = expected.iter();
    let other_coarse = other.map(|(key, (_, values))| (*key, ([0.0; 3], values.clone())));
    let other_coarse: BTreeMap<TileKey, Corrections> = other_coarse.collect();
    steps_apart(&coarse, &other_coarse)
}

#[test]
fn a_down_before_the_chain_has_resumed_waits_and_loses_no_stroke() {
    let crashed = crash_mid_chain(BRUSH_FORMATS, 1);
    // A dab a step: the first is still landing when the pen comes down.
    let mut owner = resumed_in(&crashed.dir, |project| {
        Sculpting::new(project).with_cost(DabCost::one_at_a_time())
    });
    queue(&mut owner, stroke(((140.0, 90.0), (180.0, 90.0)), 8));
    run(&mut owner, 1);
    assert!(
        !owner.brush().deferred.is_empty(),
        "the new stroke waits: a fresh op would end the second"
    );
    assert_eq!(under_way(&owner), [crashed.first]);
    let seen = landed_frame_by_frame(&mut owner);
    let [first, second, new] = seen.as_slice() else {
        panic!("the two restored, then the new one: {seen:?}");
    };
    assert_eq!((*first, *second), (crashed.first, crashed.second));
    assert!(*new > crashed.second, "an op of its own, after theirs");
    assert_eq!(owner.brush().error(), None);
    assert_eq!(owner.brush().drops(), 0, "nothing dropped");
    let records = saved_strokes(owner, &crashed.dir, "chain.bcrf");
    let [record] = records.as_slice() else {
        panic!("the new stroke's record alone: {records:?}");
    };
    assert_eq!(record.op(), *new);
}

#[test]
fn a_later_stroke_the_brush_cant_take_leaves_the_first_and_the_note_says_so() {
    let formats = Formats {
        second: stroke::DAB_FORMAT + 1,
        ..BRUSH_FORMATS
    };
    let crashed = crash_mid_chain(formats, 2);
    let mut owner = resumed_in(&crashed.dir, Sculpting::new);
    assert_eq!(under_way(&owner), [crashed.first], "the first taken on");
    let seen = landed_frame_by_frame(&mut owner);
    assert_eq!(seen, [crashed.first], "the second never resumes");
    let details = note_of(&mut owner);
    assert_eq!(
        details.last().map(String::as_str),
        Some(UNFINISHED_STROKE),
        "one stroke couldn't be finished: {details:?}"
    );
    owner.close().expect("closed");
}

#[test]
fn a_chain_none_of_whose_strokes_can_be_taken_says_so_in_the_plural() {
    let formats = Formats {
        first: stroke::DAB_FORMAT + 1,
        ..BRUSH_FORMATS
    };
    let crashed = crash_mid_chain(formats, 2);
    let mut owner = resumed_in(&crashed.dir, Sculpting::new);
    assert!(!owner.brush().stroking(), "neither taken on");
    let details = note_of(&mut owner);
    assert_eq!(
        details.last().map(String::as_str),
        Some("Your last strokes couldn't be finished; the rest of your work is back."),
        "{details:?}"
    );
    // The next stroke goes on as ever.
    queue(&mut owner, stroke(((140.0, 90.0), (180.0, 90.0)), 8));
    run(&mut owner, 10_000);
    assert!(!owner.brush().stroking(), "the new stroke is over");
    assert_eq!(owner.brush().error(), None);
    owner.close().expect("closed");
}

#[test]
fn a_restored_first_that_drops_dabs_takes_the_rest_of_the_chain_with_it() {
    let crashed = crash_mid_chain(BRUSH_FORMATS, 2);
    let mut owner = resumed_in(&crashed.dir, Sculpting::new);
    owner.brush_mut().knob = Knob::LandingFails;
    let seen = landed_frame_by_frame(&mut owner);
    assert_eq!(seen, [crashed.first], "the second never resumes");
    assert_eq!(owner.brush().drops(), 1, "one drop, said once");
    owner.brush_mut().knob = Knob::None;
    cut(owner.project(), &mut Instant::now());
    let path = owner
        .project()
        .recovery_path()
        .expect("a recovery file")
        .to_owned();
    owner.close().expect("closed");
    let batches = batches_at(&path, crashed.second);
    let last = batches.last().expect("the second was journalled");
    assert!(last.placed() > 0, "{last:?}");
    assert_eq!(
        last.landed(),
        last.placed(),
        "closed in the file, every dab counted landed: a restore lands none"
    );
}

#[test]
fn a_down_waiting_for_the_chain_begins_once_the_first_drops_it() {
    let crashed = crash_mid_chain(BRUSH_FORMATS, 1);
    let mut owner = resumed_in(&crashed.dir, |project| {
        Sculpting::new(project).with_cost(DabCost::one_at_a_time())
    });
    queue(&mut owner, stroke(((140.0, 90.0), (180.0, 90.0)), 8));
    run(&mut owner, 1);
    assert!(!owner.brush().deferred.is_empty(), "the new stroke waits");
    assert_eq!(
        owner.brush().down_awaits,
        DownAwaits::Resumed,
        "for the second to resume, no longer"
    );
    owner.brush_mut().knob = Knob::LandingFails;
    let began = Instant::now();
    while under_way(&owner).contains(&crashed.first) {
        run(&mut owner, 1);
        assert!(began.elapsed() < Duration::from_secs(60), "the first ends");
    }
    owner.brush_mut().knob = Knob::None;
    assert!(
        owner.brush().restored.is_empty(),
        "the second went with the first"
    );
    let seen = landed_frame_by_frame(&mut owner);
    assert!(!seen.contains(&crashed.second), "and never resumes");
    assert!(
        owner.brush().deferred.is_empty(),
        "the new stroke was taken"
    );
    assert_eq!(
        owner.brush().down_awaits,
        DownAwaits::Landed,
        "nothing waits"
    );
    assert!(owner.brush().drops() >= 1, "the first's dabs dropped");
    let records = saved_strokes(owner, &crashed.dir, "after.bcrf");
    assert_eq!(records.len(), 1, "the new stroke began and ended");
}

/// How far a restored chain of three may land from the same chain never crashed.
///
/// In each tile's coarsest quantization step: [`CHAIN_BOUND`]'s 1.5 for the first, and one more
/// for each stroke landing on what the one before left. Measured: 2.0.
const CHAIN_OF_THREE_BOUND: f32 = 3.5;

#[test]
fn a_restored_chain_of_three_lands_each_on_what_the_one_before_left() {
    let (dir, whole) = crash_with(|dir| {
        let (mut run, _, _) = chain_in(dir, BRUSH_FORMATS, 3);
        run.frame(&over_both());
        assert_eq!(run.brush.strokes.len(), 3, "the third began over both");
        run.cut();
        run
    });
    let mut owner = resumed_in(&dir, |project| {
        Sculpting::new(project).with_cost(DabCost::one_at_a_time())
    });
    assert_eq!(under_way(&owner).len(), 1, "the first taken on");
    assert_eq!(owner.brush().restored.len(), 2, "two wait to resume");
    let seen = landed_frame_by_frame(&mut owner);
    assert_eq!(seen.len(), 3, "each resumed in turn: {seen:?}");
    assert!(seen.is_sorted(), "in op order: {seen:?}");
    assert_eq!(owner.brush().error(), None);
    assert_eq!(owner.brush().drops(), 0, "nothing dropped");
    let got = corrections(owner.project());
    let apart = coarse_steps_apart(&got, &whole);
    println!("a restored chain of three: {apart} coarse steps from the chain never crashed");
    assert!(
        apart <= CHAIN_OF_THREE_BOUND,
        "{apart} coarse steps from the chain never crashed"
    );
    owner.close().expect("closed");
}

/// A crash with a stroke landing and a second journalled behind it whose last row is off the
/// model: the crash's directory, and the first stroke's op.
///
/// The row's numbers are a dab's (finite, a radius), so the restore takes the stroke on; only
/// against the model, as it resumes, is it seen to be none of its dabs.
fn crash_with_a_later_row_off_the_model() -> (TestDir, OpId) {
    let mut run = Journalling::new();
    run.project.set_dab_format(stroke::DAB_FORMAT);
    run.frame(&stroke(ACROSS, 12));
    let first = run.stroke().op.id();
    let good = *run.stroke().dabs.front().expect("a dab");
    let far = Dab {
        position: [50.0, 0.0, 0.0],
        ..good
    };
    let later = run.project.begin_stroke_after().expect("reserved");
    run.project
        .journal_placed_after(&later, &[row_of(good), row_of(far)])
        .expect("journalled");
    run.cut();
    let Journalling {
        dir,
        project,
        mut brush,
        picker,
        ..
    } = run;
    // A crash: nothing closes the project, and the reserved op goes with its store.
    later.abandon_with_store_gone();
    brush.abandon();
    drop((brush, picker));
    drop(project);
    (dir, first)
}

#[test]
fn a_later_stroke_found_off_the_model_as_it_resumes_isnt_finished_and_the_note_says_so() {
    let (dir, first) = crash_with_a_later_row_off_the_model();
    let mut owner = resumed_in(&dir, Sculpting::new);
    let details = note_of(&mut owner);
    assert_eq!(
        details.len(),
        1,
        "both taken on, as far as their numbers say: {details:?}"
    );
    let seen = landed_frame_by_frame(&mut owner);
    assert_eq!(seen, [first], "the second isn't resumed");
    assert_eq!(owner.brush().error(), None, "which is no error");
    assert_eq!(
        owner.brush().drops(),
        0,
        "and drops nothing a save must own"
    );
    owner.upkeep(Instant::now()).expect("upkeep");
    let bar = owner.bar().expect("the note, said again");
    assert_eq!(bar.title(), "Your unsaved work is back");
    assert_eq!(
        bar.details().last().map(String::as_str),
        Some(UNFINISHED_STROKE),
        "{:?}",
        bar.details()
    );
    // The rest saves as ever.
    owner.request_save(Instant::now());
    finish_save(&mut owner);
    assert_eq!(owner.saving_succeeded(), Some(true), "saved");
    owner.close().expect("closed");
}
