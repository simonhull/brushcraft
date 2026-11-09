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

/// A crash with [`chain_in`]'s chain open, and the same chain landed whole, never crashed.
fn crash_mid_chain(formats: Formats, landed: usize) -> CrashedChain {
    let (mut whole_run, _, _) = chain_in(TestDir::new(), formats, landed);
    let flushing = context(
        &mut whole_run.project,
        whole_run.picker.index_mut(),
        &whole_run.camera,
    );
    whole_run.brush.flush(flushing).expect("flushed");
    assert!(!whole_run.brush.stroking(), "both landed");
    let whole = corrections(&mut whole_run.project);
    let Journalling {
        dir: whole_dir,
        project: whole_project,
        ..
    } = whole_run;
    whole_project.close(false).expect("closed");
    drop(whole_dir);

    let (run, first, second) = chain_in(TestDir::new(), formats, landed);
    let Journalling {
        dir,
        project,
        mut brush,
        picker,
        ..
    } = run;
    // A crash: nothing closes the project, and the second's reserved op goes with its store.
    brush.abandon();
    drop((brush, picker));
    drop(project);
    CrashedChain {
        dir,
        first,
        second,
        whole,
    }
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
