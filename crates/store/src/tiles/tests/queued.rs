//! The queued evaluation against the dense evaluation, its deadlines and its share.
//!
//! Tiles finish highest priority first and as the dense evaluation has them, withdrawn ones are
//! left alone, deadlines are kept beside the display jobs, no tile is handed over stale after a
//! write, and the job is held to its share of the budget.

use std::time::Duration;

use geom::gather_block;

use super::super::hooks::{TestClock, Turned};
use super::*;
use crate::JobsReport;
use crate::jobs::{ROUND_MAX, ROUND_MAX_ALONE};
use crate::queued::{EvaluatedTile, EvaluationCaps};

/// Every finest tile of the cube, each with a priority from `seed`, all distinct.
fn prioritized(mesh: &BaseMesh, seed: u64) -> Vec<(TileKey, f32)> {
    let mut rng = Rng(seed);
    let keys = tiles(mesh, TOP, TOP);
    let pairs = keys.into_iter().enumerate().map(|(n, key)| {
        let n = f32::from(u16::try_from(n).unwrap());
        (
            key,
            f32::from(u16::try_from(rng.below(1000)).unwrap()) + n / 1000.0,
        )
    });
    pairs.collect()
}

/// A store over a new working file with `config`, with random corrections at every level up to
/// [`TOP`] and its cache emptied; and the corrections.
fn edited(dir: &TestDir, mesh: &Arc<BaseMesh>, config: StoreConfig) -> (TileStore, CorrectionMap) {
    let mut store = new_store(dir, "work.bcp", mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(mesh, TOP, 40, 0x0E7A));
    store.evict_all().unwrap();
    (store, map)
}

/// Runs the store's jobs in calls of `budget` and takes what they finish, until nothing is
/// queued or left to do; returns the tiles in the order they were taken.
fn take_all(store: &mut TileStore, budget: Duration) -> Vec<EvaluatedTile> {
    let began = Instant::now();
    let mut taken = Vec::new();
    while store.has_jobs() || store.queued_evaluations() > 0 {
        store.run_jobs(Instant::now() + budget).unwrap();
        taken.extend(store.take_evaluated());
        assert!(began.elapsed() < Duration::from_secs(60), "the jobs end");
    }
    taken.extend(store.take_evaluated());
    taken
}

/// Asserts that `tile` is the dense evaluation of `map`'s: positions, ring and normals, bit
/// for bit (no normals at all for a tile queued for its positions alone).
fn assert_dense(mesh: &BaseMesh, dense: &DenseEvaluation, tile: &EvaluatedTile, what: &str) {
    let key = tile.key();
    let positions: Vec<Vec3> = tile.rows().flatten().copied().collect();
    let (face, level) = (key.face(), key.level());
    assert_tile(
        &positions,
        |i, j| dense.position(face, level, i, j).unwrap(),
        &key,
        what,
    );
    let region = Region::of_tile(&key);
    let ring = gather_block(mesh, dense, region).unwrap();
    let bits = |values: &[Vec3]| {
        let bits: Vec<[u32; 3]> = values.iter().map(|v| v.to_bits()).collect();
        bits
    };
    assert_eq!(
        bits(tile.ring().padded()),
        bits(ring.padded()),
        "{what} {key:?}: the ring"
    );
    if tile.what() == Evaluation::Positions {
        assert!(tile.normals().is_empty(), "{what} {key:?}: no normals");
        return;
    }
    let want = normals(mesh, dense, region).unwrap();
    assert_tile(
        tile.normals(),
        |i, j| want.get(i - region.x0(), j - region.y0()).unwrap(),
        &key,
        &format!("{what}: normals"),
    );
}

#[test]
fn queued_tiles_finish_highest_priority_first_as_the_dense_evaluation_has_them() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map) = edited(&dir, &mesh, roomy());
    let wanted = prioritized(&mesh, 3);
    assert_eq!(store.queue_evaluation(&wanted).unwrap(), 0, "all fit");
    // Queued again: once each, at the later priority.
    assert_eq!(store.queue_evaluation(&wanted[..4]).unwrap(), 0, "again");
    assert_eq!(store.queued_evaluations(), wanted.len(), "once each");
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), wanted.len(), "every tile, once");
    let priorities: Vec<f32> = taken.iter().map(EvaluatedTile::priority).collect();
    assert!(
        priorities.windows(2).all(|w| w[0] >= w[1]),
        "highest priority first: {priorities:?}"
    );
    let keys: BTreeSet<TileKey> = taken.iter().map(EvaluatedTile::key).collect();
    assert_eq!(keys.len(), wanted.len(), "no tile twice");
    let queued: BTreeMap<TileKey, f32> = wanted.iter().copied().collect();
    for tile in &taken {
        assert_eq!(
            tile.priority().to_bits(),
            queued[&tile.key()].to_bits(),
            "{:?}: the priority it was queued with",
            tile.key()
        );
    }
    let dense = dense(&mesh, &map);
    for tile in &taken {
        assert_dense(&mesh, &dense, tile, "queued");
    }
    assert!(
        store.take_evaluated().is_empty(),
        "each tile is handed over once"
    );
}

/// Queued for their positions alone, tiles come with their positions and ring as the dense
/// evaluation has them, no normals, and none computed for them.
#[test]
fn tiles_queued_for_their_positions_alone_come_right_without_normals() {
    let mesh = mesh();
    for config in [roomy(), tight()] {
        let dir = TestDir::new();
        let (mut store, map) = edited(&dir, &mesh, config);
        let wanted = prioritized(&mesh, 31);
        let wanted = &wanted[..24];
        store
            .queue_evaluation_as(wanted, Evaluation::Positions)
            .unwrap();
        let taken = take_all(&mut store, Duration::from_millis(2));
        assert_eq!(taken.len(), wanted.len(), "every tile");
        let dense = dense(&mesh, &map);
        for tile in &taken {
            assert_eq!(tile.what(), Evaluation::Positions, "positions alone");
            assert_dense(&mesh, &dense, tile, "positions alone");
            assert!(
                !store.is_cached(&tile.key()),
                "{:?}: its normals never computed",
                tile.key()
            );
        }
        drop(store);
    }
}

/// A write under a finished tile queued for its positions alone: it is evaluated again for its
/// positions alone, never handed over stale.
#[test]
fn a_tile_of_positions_alone_outdated_by_a_write_stays_positions_alone() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let wanted = prioritized(&mesh, 37);
    store
        .queue_evaluation_as(&wanted[..12], Evaluation::Positions)
        .unwrap();
    store.hooks.hold_evaluated = true;
    let began = Instant::now();
    while store.hooks.held_evaluated.is_empty() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(1))
            .unwrap();
        assert!(began.elapsed() < Duration::from_secs(30), "one finishes");
    }
    let held = store.hooks.held_evaluated[0].key();
    let region = Region::of_tile(&held);
    let write = Edit {
        face: held.face(),
        level: level(TOP),
        i: region.x0() + 30,
        j: region.y0() + 30,
        c: Vec3::new(0.01, -0.02, 0.015),
    };
    apply(&mut store, &mut map, &[write]);
    store.hooks.hold_evaluated = false;
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), 12, "every tile, the held one again");
    let dense = dense(&mesh, &map);
    for tile in &taken {
        assert_eq!(tile.what(), Evaluation::Positions, "positions alone");
        assert_dense(&mesh, &dense, tile, "after a write");
    }
}

#[test]
fn a_tight_budget_still_finishes_every_tile_right() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, map) = edited(&dir, &mesh, tight());
    let wanted = prioritized(&mesh, 5);
    store.queue_evaluation(&wanted[..24]).unwrap();
    let taken = take_all(&mut store, Duration::from_millis(5));
    assert_eq!(taken.len(), 24, "every tile");
    let dense = dense(&mesh, &map);
    for tile in &taken {
        assert_dense(&mesh, &dense, tile, "tight");
    }
}

#[test]
fn withdrawn_tiles_are_neither_evaluated_nor_handed_over() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy());
    let wanted = prioritized(&mesh, 7);
    store.queue_evaluation(&wanted).unwrap();
    // Run until some are finished (and taken, so the job goes on) and a cone is being stepped
    // through.
    let began = Instant::now();
    let mut early = Vec::new();
    while early.is_empty() || store.evaluation.steps.cone.is_none() {
        store
            .run_jobs(Instant::now() + Duration::from_micros(200))
            .unwrap();
        early.extend(store.take_evaluated());
        assert!(began.elapsed() < Duration::from_secs(30), "some finish");
    }
    let stepping: Vec<TileKey> = store.evaluation.steps.cone.as_ref().unwrap().keys.clone();
    let keep: BTreeSet<TileKey> = wanted
        .iter()
        .map(|&(key, _)| key)
        .filter(|key| (key.tx() + key.ty()) % 2 == 0)
        .collect();
    assert!(
        stepping.iter().any(|key| !keep.contains(key)),
        "a tile being evaluated is withdrawn: {stepping:?}"
    );
    store.retain_evaluation(|key| keep.contains(key));
    assert!(
        store.evaluation.steps.cone.is_none(),
        "the cone of a withdrawn tile is dropped"
    );
    let evaluated_before = store.hooks.log.normals.len();
    let taken = take_all(&mut store, Duration::from_millis(2));
    for tile in &taken {
        assert!(keep.contains(&tile.key()), "{:?} was withdrawn", tile.key());
    }
    let finished_or_left = taken.len();
    assert!(finished_or_left > 0, "the rest finish");
    assert!(finished_or_left < keep.len() + 1, "only those kept");
    let normals_after: BTreeSet<TileKey> = store.hooks.log.normals[evaluated_before..]
        .iter()
        .copied()
        .filter(|key| key.level() == level(TOP))
        .collect();
    let withdrawn: Vec<&TileKey> = normals_after
        .iter()
        .filter(|key| !keep.contains(key))
        .collect();
    assert!(
        withdrawn.is_empty(),
        "no withdrawn tile's normals computed after: {withdrawn:?}"
    );
}

/// On the tests' clock, a call runs no more slices than its deadline holds.
///
/// The clock moves a slice at a time; no slice computes more than a round of tiles' positions or
/// normals, and both jobs move on together.
#[test]
fn run_jobs_keeps_its_deadline_in_slices_and_both_jobs_move_on_together() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    assert!(
        store.has_jobs(),
        "a file without a display layer queues its rebuild"
    );
    store.queue_evaluation(&prioritized(&mesh, 11)).unwrap();
    let slice = Duration::from_millis(1);
    store.hooks.clock = Some(TestClock {
        now: Instant::now(),
        slice,
    });
    let began = Instant::now();
    let (mut built_while_queued, mut finished_while_building) = (0, 0);
    let mut taken = 0;
    while store.has_jobs() || store.queued_evaluations() > 0 {
        let start = store.hooks.clock_now().unwrap();
        let report = store.run_jobs(start + 3 * slice).unwrap();
        assert!(
            report.slices <= 3,
            "a deadline of three slices runs three at most: {}",
            report.slices
        );
        if report.queue_left > 0 {
            built_while_queued += report.built;
        }
        if report.left > 0 {
            finished_while_building += report.queue_finished;
        }
        taken += store.take_evaluated().len();
        assert!(began.elapsed() < Duration::from_secs(60), "the jobs end");
    }
    let turns = &store.hooks.turns;
    // A round beside work of the display jobs' is theirs at most; the queue alone may take
    // more of the time left.
    let heavy: Vec<&Turned> = turns
        .iter()
        .filter(|turn| {
            let most = if turn.contested {
                ROUND_MAX
            } else {
                ROUND_MAX_ALONE
            };
            turn.positions > most || turn.normals > most
        })
        .collect();
    assert!(heavy.is_empty(), "no slice past a round: {heavy:?}");
    assert!(
        turns
            .iter()
            .filter(|turn| turn.contested)
            .all(|turn| turn.positions <= ROUND_MAX && turn.normals <= ROUND_MAX),
        "beside the display jobs, a display job's round"
    );
    assert!(
        turns.iter().any(|turn| turn.positions > 0),
        "slices evaluated"
    );
    assert_eq!(taken, tiles(&mesh, TOP, TOP).len(), "every queued tile");
    assert!(
        built_while_queued > 0,
        "the display jobs built while tiles were queued"
    );
    assert!(
        finished_while_building > 0,
        "the queue finished tiles while the display layer was being built"
    );
}

/// On the real clock, a coarse check that a call ends near its deadline: within its longest
/// slice and a generous allowance for a loaded machine's scheduler.
#[test]
fn run_jobs_ends_near_its_deadline_on_the_real_clock() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    store.queue_evaluation(&prioritized(&mesh, 23)).unwrap();
    let budget = Duration::from_millis(2);
    for call in 0..50 {
        let start = Instant::now();
        let report = store.run_jobs(start + budget).unwrap();
        let spent = start.elapsed();
        assert!(
            spent <= budget + report.longest_slice + Duration::from_millis(50),
            "call {call}: {spent:?} for {budget:?}, longest slice {:?}",
            report.longest_slice
        );
        drop(store.take_evaluated());
    }
}

/// With the queue alone, a call ends near its deadline even when its cone tiles turn ten
/// times dearer than the cost it estimated.
///
/// Within about a millisecond: a call's first group is one tile, measured, and the next are
/// sized at the dearer of that and the estimate, the clock read before each; a step with too
/// little time waits for the next call.
#[test]
fn run_jobs_alone_keeps_its_deadline_when_its_estimate_is_ten_times_too_cheap() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy());
    store
        .queue_evaluation_as(&prioritized(&mesh, 43), Evaluation::Positions)
        .unwrap();
    let budget = Duration::from_millis(8);
    // Calls at the tiles' own cost, until one is measured.
    for _ in 0..3 {
        store.run_jobs(Instant::now() + budget).unwrap();
        drop(store.take_evaluated());
    }
    let estimate = store
        .evaluation
        .steps
        .cone_tile_cost()
        .expect("a cost measured");
    // Every tile ten times that from now on (on top of its own cost).
    store.hooks.slow_cone_tile = estimate * 10;
    let (mut worst, mut finished) = (Duration::ZERO, 0);
    for _ in 0..12 {
        let start = Instant::now();
        let report = store.run_jobs(start + budget).unwrap();
        worst = worst.max(start.elapsed().saturating_sub(budget));
        finished += report.queue_finished;
        drop(store.take_evaluated());
    }
    assert!(finished > 0, "tiles finished");
    // A millisecond, and as much again for a loaded machine's scheduler.
    assert!(
        worst <= Duration::from_millis(2),
        "a call ran {worst:?} past its deadline (estimate {estimate:?})"
    );
}

/// With the queue alone, for positions alone, a call still ends near its deadline: its larger
/// rounds keep to the time left, each slice within a round's time and a step's fixed cost.
#[test]
fn run_jobs_alone_ends_near_its_deadline_on_the_real_clock() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy());
    store
        .queue_evaluation_as(&prioritized(&mesh, 41), Evaluation::Positions)
        .unwrap();
    let budget = Duration::from_millis(3);
    let allowance = Duration::from_millis(50);
    for call in 0..50 {
        let start = Instant::now();
        let report = store.run_jobs(start + budget).unwrap();
        let spent = start.elapsed();
        assert!(
            spent <= budget + report.longest_slice + allowance,
            "call {call}: {spent:?} for {budget:?}, longest slice {:?}",
            report.longest_slice
        );
        assert!(
            report.longest_slice <= crate::jobs::ROUND_TIME_ALONE + allowance,
            "call {call}: a slice within a round's time: {:?}",
            report.longest_slice
        );
        drop(store.take_evaluated());
    }
}

/// During a brush stroke (a write between every call and take), the tiles away from the
/// brush are handed over as they finish, and none under it, nor reading it through its ring,
/// ever is stale.
#[test]
fn during_a_stroke_tiles_away_from_the_brush_are_handed_over() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    // The brush works inside face 0's first tile; the queue wants it, at the top priority, and
    // every tile of faces 3 to 5 whose ring doesn't read it (away from the brush).
    let brushed = TileKey::displacement(FaceId::new(0), level(TOP), 0, 0).unwrap();
    let away = |reading: &TileStore, key: &TileKey| {
        key.face().index() >= 3 && !reading.normal_inputs(key).unwrap().contains(&brushed)
    };
    let mut wanted: Vec<(TileKey, f32)> = prioritized(&mesh, 29)
        .into_iter()
        .filter(|(key, _)| away(&store, key))
        .collect();
    assert!(wanted.len() > 20, "tiles away from the brush");
    wanted.push((brushed, 1e6));
    store.queue_evaluation(&wanted).unwrap();
    let mut taken = Vec::new();
    let mut dab = 0_u32;
    let began = Instant::now();
    while taken.len() < wanted.len() - 1 {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
        // The next frame's brush step, then its uploads.
        store.begin_op().unwrap();
        apply_dabs(
            &mut store,
            &mut map,
            &[Edit {
                face: FaceId::new(0),
                level: level(TOP),
                i: 10 + dab % 40,
                j: 20,
                c: Vec3::new(0.0, 0.0, 0.001 * f32::from(u16::try_from(dab % 7).unwrap())),
            }],
        );
        dab += 1;
        let dense = dense(&mesh, &map);
        for tile in store.take_evaluated() {
            assert_ne!(tile.key(), brushed, "the brushed tile is never handed over");
            assert_dense(&mesh, &dense, &tile, "during the stroke");
            taken.push(tile.key());
        }
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the tiles away from the brush arrive during the stroke"
        );
    }
    assert!(dab > 1, "a stroke of several dabs");
}

/// A lost tile's write undone, with tiles finished: every one is kept back, then handed over as
/// the store reads it after the revert.
#[test]
fn a_lost_write_undone_keeps_finished_tiles_back_until_copied_again() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let lost = TileKey::displacement(FaceId::new(1), level(TOP), 2, 2).unwrap();
    let region = Region::of_tile(&lost);
    apply(
        &mut store,
        &mut map,
        &[Edit {
            face: lost.face(),
            level: level(TOP),
            i: region.x0() + 20,
            j: region.y0() + 20,
            c: Vec3::new(0.02, 0.0, 0.01),
        }],
    );
    let keys: Vec<(TileKey, f32)> = [
        (lost, 1.0),
        (
            TileKey::displacement(FaceId::new(4), level(TOP), 0, 0).unwrap(),
            0.5,
        ),
    ]
    .to_vec();
    store.queue_evaluation(&keys).unwrap();
    while store.evaluation.finished.len() < 2 {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
    }
    store.flush_bounds(&[lost]).unwrap();
    assert!(
        store.take_evaluated().is_empty(),
        "every finished tile kept back after a revert"
    );
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), 2, "both handed over again");
    for tile in &taken {
        let key = tile.key();
        let ring = store.positions_with_ring(key).unwrap();
        let bits = |values: &[Vec3]| {
            let bits: Vec<[u32; 3]> = values.iter().map(|v| v.to_bits()).collect();
            bits
        };
        assert_eq!(
            bits(tile.ring().padded()),
            bits(ring.padded()),
            "{key:?}: the ring as the store reads it now"
        );
        let normals = store.normals(key).unwrap();
        assert_eq!(tile.normals(), normals, "{key:?}: and its normals");
    }
}

/// A tile finished, then queued again before it is taken, is handed over once; one re-queued
/// after a write keeps the newer priority it was queued at meanwhile.
#[test]
fn a_tile_queued_again_is_handed_over_once_at_its_newest_priority() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let key = TileKey::displacement(FaceId::new(2), level(TOP), 1, 2).unwrap();
    store.queue_evaluation(&[(key, 1.0)]).unwrap();
    let run_until_finished = |running: &mut TileStore| {
        while running.evaluation.finished.is_empty() || running.queued_evaluations() > 0 {
            running
                .run_jobs(Instant::now() + Duration::from_millis(2))
                .unwrap();
        }
    };
    run_until_finished(&mut store);
    store.queue_evaluation(&[(key, 2.0)]).unwrap();
    run_until_finished(&mut store);
    assert_eq!(store.evaluation.finished.len(), 1, "one copy");
    // Outdated by a write under it, and queued at a newer priority before the job sees it.
    let region = Region::of_tile(&key);
    apply(
        &mut store,
        &mut map,
        &[Edit {
            face: key.face(),
            level: level(TOP),
            i: region.x0() + 5,
            j: region.y0() + 5,
            c: Vec3::new(0.01, 0.01, 0.0),
        }],
    );
    store.queue_evaluation(&[(key, 7.0)]).unwrap();
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), 1, "handed over once");
    assert_eq!(
        taken[0].priority().to_bits(),
        7.0_f32.to_bits(),
        "at its newest priority"
    );
    assert_dense(&mesh, &dense(&mesh, &map), &taken[0], "after the write");
}

#[test]
fn a_write_under_a_finished_tile_never_hands_it_over_stale() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let wanted = prioritized(&mesh, 13);
    store.queue_evaluation(&wanted[..12]).unwrap();
    // The finished tiles held back, as if still being evaluated, until after a write under
    // the first of them.
    store.hooks.hold_evaluated = true;
    let began = Instant::now();
    while store.hooks.held_evaluated.is_empty() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(1))
            .unwrap();
        assert!(began.elapsed() < Duration::from_secs(30), "one finishes");
    }
    let held = store.hooks.held_evaluated[0].key();
    let region = Region::of_tile(&held);
    let write = |dx: u32, dy: u32| Edit {
        face: held.face(),
        level: level(TOP),
        i: region.x0() + dx,
        j: region.y0() + dy,
        c: Vec3::new(0.01, -0.02, 0.015),
    };
    apply(&mut store, &mut map, &[write(30, 30)]);
    store.hooks.hold_evaluated = false;
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), 12, "every tile, the held one evaluated again");
    let held_dense = dense(&mesh, &map);
    for tile in &taken {
        assert_dense(&mesh, &held_dense, tile, "after a write under a held tile");
    }
}

/// Finished and not taken yet, then a write in a neighbour moves its ring: kept back, and
/// copied again or evaluated again before it is handed over.
#[test]
fn a_write_in_a_finished_tiles_ring_never_hands_it_over_stale() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let key = prioritized(&mesh, 13)[20].0;
    store.queue_evaluation(&[(key, 1.0)]).unwrap();
    while store.evaluation.finished.is_empty() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
    }
    let region = Region::of_tile(&key);
    let neighbour = Edit {
        face: key.face(),
        level: level(TOP),
        i: if region.x0() == 0 {
            65
        } else {
            region.x0() - 1
        },
        j: region.y0() + 10,
        c: Vec3::new(-0.01, 0.01, 0.02),
    };
    apply(&mut store, &mut map, &[neighbour]);
    assert!(
        store.take_evaluated().is_empty(),
        "a tile finished before a write isn't handed over"
    );
    let taken = take_all(&mut store, Duration::from_millis(2));
    assert_eq!(taken.len(), 1, "handed over once brought up to date");
    assert_dense(
        &mesh,
        &dense(&mesh, &map),
        &taken[0],
        "after a neighbour's write",
    );
}

#[test]
fn the_queues_share_follows_the_budget() {
    let config = StoreConfig::new(1 << 30, 1).with_display(level(TOP));
    assert_eq!(
        config.evaluation_bytes(),
        (1 << 30) / 64,
        "a sixty-fourth of the budget"
    );
    let huge = StoreConfig::new(64 << 30, 1).with_display(level(TOP));
    assert_eq!(
        huge.evaluation_bytes(),
        StoreConfig::MAX_EVALUATION_BYTES,
        "at most the cap"
    );
    assert_eq!(
        StoreConfig::new(1 << 30, 1).evaluation_bytes(),
        0,
        "no display layer, no share"
    );
}

#[test]
fn the_queues_share_comes_out_of_the_pools_budget() {
    let dir = TestDir::new();
    let mesh = mesh();
    let config = roomy().with_display(level(TOP));
    let store = new_store(&dir, "work.bcp", &mesh, config);
    let others = config.kept_prediction_bytes().unwrap()
        + config.stroke_state_bytes().unwrap()
        + config.floor_bytes(&mesh)
        + config.display_cache_bytes()
        + config.display_waiting_bytes();
    assert_eq!(
        store.budget.limit(),
        config.budget_bytes - others - config.evaluation_bytes(),
        "the pools' budget less the queue's share"
    );
    assert_eq!(
        store.evaluation.caps,
        EvaluationCaps::of(config.evaluation_bytes()),
        "the caps are the share's"
    );
}

/// Full, the job stops until the caller takes some; past the keys' cap, the lowest go.
#[test]
fn the_queue_is_held_to_its_caps() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy());
    store.evaluation.caps = EvaluationCaps {
        finished: 3,
        queued: 10,
    };
    let wanted = prioritized(&mesh, 17);
    let mut sorted = wanted[..12].to_vec();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    assert_eq!(
        store.queue_evaluation(&wanted[..12]).unwrap(),
        2,
        "two past the cap"
    );
    let began = Instant::now();
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the job fills up"
        );
    }
    assert_eq!(store.evaluation.finished.len(), 3, "no more than the cap");
    assert_eq!(store.queued_evaluations(), 7, "the rest wait");
    let report = store
        .run_jobs(Instant::now() + Duration::from_millis(2))
        .unwrap();
    assert_eq!(report.queue_finished, 0, "nothing more while full");
    let taken = take_all(&mut store, Duration::from_millis(2));
    let keys: Vec<TileKey> = taken.iter().map(EvaluatedTile::key).collect();
    let want: Vec<TileKey> = sorted[..10].iter().map(|&(key, _)| key).collect();
    assert_eq!(keys, want, "the ten highest, in order");
}

#[test]
fn is_cached_never_evaluates() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy());
    let key = TileKey::displacement(FaceId::new(1), level(TOP), 2, 1).unwrap();
    let before = store.hooks.log.positions.len();
    assert!(!store.is_cached(&key), "nothing cached");
    assert_eq!(store.hooks.log.positions.len(), before, "nothing evaluated");
    store.evaluate(key).unwrap();
    assert!(!store.is_cached(&key), "positions alone aren't enough");
    store.normals(key).unwrap();
    assert!(store.is_cached(&key), "positions and normals");
    apply(
        &mut store,
        &mut CorrectionMap::new(),
        &[Edit {
            face: key.face(),
            level: level(TOP),
            i: 140,
            j: 80,
            c: Vec3::new(0.0, 0.0, 0.01),
        }],
    );
    assert!(!store.is_cached(&key), "a write under it outdates it");
}

/// With little budget to spare, the queue's share is what the display layer's shares leave.
#[test]
fn a_short_budget_gives_the_queue_what_is_left() {
    use super::super::config::{Headroom, reserved_shares};

    let mesh = mesh();
    let headroom = Headroom::of(&mesh);
    let base = roomy().with_display(level(TOP));
    let floor = base.floor_bytes(&mesh);
    let fixed = headroom.minimum_budget(base.chunk_bytes)
        + base.kept_prediction_bytes().unwrap()
        + base.stroke_state_bytes().unwrap()
        + floor;
    let mut partial = 0;
    for step in 0..3000 {
        let config = base.with_budget_bytes(fixed + step * 4096);
        let shares = reserved_shares(headroom, &config, floor).unwrap();
        let left = config.budget_bytes - fixed - shares.caches - shares.waiting;
        assert_eq!(
            shares.evaluation,
            config.evaluation_bytes().min(left),
            "step {step}: the share, or what the others leave"
        );
        if shares.evaluation > 0 && shares.evaluation < config.evaluation_bytes() {
            partial += 1;
        }
    }
    assert!(partial > 0, "some budgets give part of the share");
}

/// A write that leaves a finished tile and its ring alone leaves the tile current: handed over
/// at once, neither copied nor evaluated again.
#[test]
fn a_write_elsewhere_leaves_a_finished_tile_current() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, mut map) = edited(&dir, &mesh, roomy());
    let key = TileKey::displacement(FaceId::new(0), level(TOP), 1, 1).unwrap();
    store.queue_evaluation(&[(key, 2.0)]).unwrap();
    while store.evaluation.finished.is_empty() {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
    }
    apply(
        &mut store,
        &mut map,
        &[Edit {
            face: FaceId::new(5),
            level: level(TOP),
            i: 100,
            j: 100,
            c: Vec3::new(0.0, 0.01, 0.0),
        }],
    );
    assert!(!store.has_jobs(), "nothing to bring up to date");
    assert_eq!(store.queued_evaluations(), 0, "nor queued again");
    let taken = store.take_evaluated();
    assert_eq!(taken.len(), 1, "handed over right after the write");
    assert_dense(&mesh, &dense(&mesh, &map), &taken[0], "as it was");
}

/// With both sides able to run, each slice goes to the side that didn't run the one before,
/// within a call and across calls.
#[test]
fn the_display_jobs_and_the_queue_take_turns() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    store.queue_evaluation(&prioritized(&mesh, 19)).unwrap();
    for _ in 0..20 {
        store
            .run_jobs(Instant::now() + Duration::from_millis(1))
            .unwrap();
        drop(store.take_evaluated());
    }
    let turns = &store.hooks.turns;
    let contested: Vec<bool> = turns
        .windows(2)
        .filter(|pair| pair[1].contested)
        .map(|pair| pair[0].queue != pair[1].queue)
        .collect();
    assert!(!contested.is_empty(), "both sides had work at once");
    assert!(
        contested.iter().all(|&alternated| alternated),
        "each contested turn goes to the other side: {turns:?}"
    );
    assert!(
        turns.iter().any(|turn| turn.queue) && turns.iter().any(|turn| !turn.queue),
        "both ran"
    );
}

/// A slice is counted, its length kept if the longest, and counted long past the target.
#[test]
fn slices_are_counted_and_the_long_ones_noted() {
    use super::super::upkeep::note_slice;

    let mut report = JobsReport::default();
    note_slice(&mut report, Duration::from_micros(10));
    note_slice(&mut report, JobsReport::SLICE_TARGET);
    note_slice(
        &mut report,
        JobsReport::SLICE_TARGET + Duration::from_micros(1),
    );
    note_slice(&mut report, Duration::from_micros(20));
    assert_eq!(report.slices, 4, "every slice");
    assert_eq!(report.long_slices, 1, "only one past the target");
    assert_eq!(
        report.longest_slice,
        JobsReport::SLICE_TARGET + Duration::from_micros(1),
        "the longest"
    );
}
