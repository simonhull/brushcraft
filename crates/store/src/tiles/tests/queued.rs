//! The queued evaluation against the dense evaluation, its deadlines and its share.
//!
//! Tiles finish highest priority first and as the dense evaluation has them, withdrawn ones are
//! left alone, deadlines are kept beside the display jobs, no tile is handed over stale after a
//! write, and the job is held to its share of the budget.

use std::time::Duration;

use geom::gather_block;

use super::*;
use crate::JobsReport;
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
/// for bit.
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
    let bits = |values: &[Vec3]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(tile.ring().padded()),
        bits(ring.padded()),
        "{what} {key:?}: the ring"
    );
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
    let dense = dense(&mesh, &map);
    for tile in &taken {
        assert_dense(&mesh, &dense, tile, "queued");
    }
    assert!(
        store.take_evaluated().is_empty(),
        "each tile is handed over once"
    );
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

#[test]
fn run_jobs_keeps_its_deadline_and_both_jobs_move_on_together() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy().with_display(level(TOP)));
    assert!(
        store.has_jobs(),
        "a file without a display layer queues its rebuild"
    );
    store.queue_evaluation(&prioritized(&mesh, 11)).unwrap();
    let budget = Duration::from_micros(300);
    let began = Instant::now();
    let (mut built_while_queued, mut finished_while_building) = (0, 0);
    let mut taken = 0;
    let mut calls = 0;
    let mut overruns = Vec::new();
    while store.has_jobs() || store.queued_evaluations() > 0 {
        let start = Instant::now();
        let report = store.run_jobs(start + budget).unwrap();
        let spent = start.elapsed();
        if spent > budget + report.longest_slice + Duration::from_millis(1) {
            overruns.push((calls, spent, report.longest_slice));
        }
        if report.queue_left > 0 {
            built_while_queued += report.built;
        }
        if report.left > 0 {
            finished_while_building += report.queue_finished;
        }
        taken += store.take_evaluated().len();
        calls += 1;
        assert!(began.elapsed() < Duration::from_secs(60), "the jobs end");
    }
    // Thousands of calls of a fraction of a millisecond each: on a loaded machine (the tests
    // run side by side, every store with its workers) the scheduler takes the thread away for
    // a timeslice now and then, inside a call or out. Those are rare; a call overrunning its
    // deadline by more than its longest slice every time would not be.
    assert!(
        overruns.len() * 100 <= calls,
        "{} of {calls} calls overran by more than a slice: {overruns:?}",
        overruns.len()
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

/// A write that leaves a finished tile alone keeps it: copied again from the cache, not
/// evaluated again.
#[test]
fn a_write_elsewhere_keeps_a_finished_tile_copied_from_the_cache() {
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
    assert!(store.take_evaluated().is_empty(), "kept back after a write");
    let report = store
        .run_jobs(Instant::now() + Duration::from_millis(50))
        .unwrap();
    assert_eq!(report.queue_finished, 0, "nothing evaluated again");
    assert_eq!(store.queued_evaluations(), 0, "nor queued again");
    let taken = store.take_evaluated();
    assert_eq!(taken.len(), 1, "handed over");
    assert_dense(&mesh, &dense(&mesh, &map), &taken[0], "copied again");
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
        .filter(|pair| pair[1].1)
        .map(|pair| pair[0].0 != pair[1].0)
        .collect();
    assert!(!contested.is_empty(), "both sides had work at once");
    assert!(
        contested.iter().all(|&alternated| alternated),
        "each contested turn goes to the other side: {turns:?}"
    );
    assert!(
        turns.iter().any(|&(queue, _)| queue) && turns.iter().any(|&(queue, _)| !queue),
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
