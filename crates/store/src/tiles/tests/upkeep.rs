//! Keeping the display layer up to date: a rebuild and a refresh against the layer built from
//! scratch, bit for bit; deadlines; and refreshes that need unknown neighbours first.

use std::time::Duration;

use geom::DisplayTile;

use super::*;
use crate::jobs::{Job, Upkeep};

/// A store keeping a display layer under [`TOP`], over a new working file.
fn display_store(dir: &TestDir, mesh: &Arc<BaseMesh>) -> TileStore {
    new_store(dir, "work.bcp", mesh, roomy().with_display(level(TOP)))
}

/// Runs the store's jobs until none is left; returns how many calls it took.
fn run_all(store: &mut TileStore) -> usize {
    let mut calls = 0;
    while store.has_jobs() || calls == 0 {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
        calls += 1;
        assert!(calls < 1000, "the jobs end");
    }
    calls
}

/// Asserts that the store's display layer is the one its tiles build from scratch.
fn assert_layer_current(store: &mut TileStore, what: &str) {
    let built = display_layer(store);
    let keys: Vec<TileKey> = built.iter().map(|t| *t.key()).collect();
    let stored = store.stored_display(&keys).unwrap();
    for (tile, kept) in built.iter().zip(stored) {
        let kept: DisplayTile = kept.unwrap_or_else(|| panic!("{what}: {:?} known", tile.key()));
        assert_eq!(&kept, tile, "{what}: {:?} bit for bit", tile.key());
    }
}

#[test]
fn a_rebuild_then_refreshes_after_edits_give_the_layer_built_from_scratch() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 30, 7));
    assert!(
        store.has_jobs(),
        "a file without a display layer queues its rebuild"
    );
    run_all(&mut store);
    assert_layer_current(&mut store, "rebuilt");
    for round in 0..4_u64 {
        apply(
            &mut store,
            &mut map,
            &random_edits(&mesh, TOP, 3, 100 + round),
        );
        run_all(&mut store);
        assert_layer_current(&mut store, &format!("refreshed, round {round}"));
    }
}

/// On the tests' clock, which moves a slice at a time: a call runs no more slices than its
/// deadline holds, and no slice computes more than a round of tiles' positions or normals.
#[test]
fn run_jobs_honours_its_deadline_within_one_slice() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    let slice = Duration::from_millis(1);
    store.hooks.clock = Some(hooks::TestClock {
        now: Instant::now(),
        slice,
    });
    let mut calls = 0;
    let began = Instant::now();
    while store.has_jobs() {
        let start = store.hooks.clock_now().unwrap();
        let report = store.run_jobs(start + 2 * slice).unwrap();
        assert!(
            report.slices <= 2,
            "call {calls}: a deadline of two slices runs two at most: {}",
            report.slices
        );
        calls += 1;
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the rebuild ends"
        );
    }
    let heavy: Vec<_> = store
        .hooks
        .turns
        .iter()
        .filter(|turn| {
            turn.positions > crate::jobs::ROUND_MAX || turn.normals > crate::jobs::ROUND_MAX
        })
        .collect();
    assert!(heavy.is_empty(), "no slice past a round: {heavy:?}");
    assert!(calls > 1, "a rebuild takes many short calls: {calls}");
    assert_layer_current(&mut store, "rebuilt in slices");
}

#[test]
fn a_refresh_builds_the_unknown_neighbours_it_reads_first_across_faces() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    assert_layer_current(&mut store, "rebuilt");
    // A chain across faces: face 0's whole-face tile at level 3 reads level 4 of faces 0 and its
    // neighbours, whose level-4 tiles read level 5, and so on; make one tile unknown at each
    // level from 4 down, each on another face around face 0's corner.
    let corner_faces: Vec<FaceId> = mesh
        .topology()
        .copies(FaceId::new(0), level(1), 0, 0)
        .iter()
        .map(|copy| copy.face)
        .collect();
    assert!(corner_faces.len() >= 3, "a corner of three faces");
    for (n, l) in (4..TOP).enumerate() {
        let face = corner_faces[n % corner_faces.len()];
        let key = TileKey::display(face, level(l), 0, 0).unwrap();
        store.display.set(key, None);
    }
    let coarse = TileKey::display(FaceId::new(0), level(3), 0, 0).unwrap();
    store.display.set(coarse, None);
    store.jobs.push_back(Job::Refresh(Upkeep::new(
        level(TOP),
        [coarse],
        crate::jobs::CacheCaps::MIN_COPIES,
    )));
    let calls = run_all(&mut store);
    assert!(calls >= 1, "ran");
    assert_layer_current(&mut store, "the chain refreshed");
}

/// The surface the display tile `tile` draws at the point `(u, v)` of its own grid (in quads,
/// fractional), in `f64`: its positions on the quad's two triangles, split along its
/// `(i, j)`–`(i + 1, j + 1)` diagonal.
fn drawn(tile: &DisplayTile, (u, v): (f64, f64)) -> [f64; 3] {
    let side = tile.key().side();
    let quads = f64::from(side - 1);
    let (qi, qj) = (u.floor().min(quads - 1.0), v.floor().min(quads - 1.0));
    let (fu, fv) = (u - qi, v - qj);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a quad index inside the tile"
    )]
    let (qi, qj) = (qi as u32, qj as u32);
    let at = |i: u32, j: u32| {
        let p = tile.positions()[usize::try_from(j * side + i).unwrap()];
        [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
    };
    let (p00, p10, p01, p11) = (
        at(qi, qj),
        at(qi + 1, qj),
        at(qi, qj + 1),
        at(qi + 1, qj + 1),
    );
    [0, 1, 2].map(|k| {
        if fu >= fv {
            p00[k] + fu * (p10[k] - p00[k]) + fv * (p11[k] - p10[k])
        } else {
            p00[k] + fu * (p11[k] - p01[k]) + fv * (p01[k] - p00[k])
        }
    })
}

/// Asserts that every finest vertex under display tile `tile` lies within its bound of the
/// surface it draws.
fn assert_bound(store: &mut TileStore, tile: &DisplayTile, what: &str) {
    let key = tile.key();
    let shift = TOP - key.level().get();
    let scale = f64::from(1_u32 << shift);
    let region = Region::of_tile(key);
    let bound = f64::from(tile.error());
    for y in (region.y0() << shift)..=((region.y0() + region.quads()) << shift) {
        for x in (region.x0() << shift)..=((region.x0() + region.quads()) << shift) {
            let (fine, i, j) = TileKey::containing(
                LayerId::BASE,
                Channel::DISPLACEMENT,
                key.face(),
                level(TOP),
                x,
                y,
            )
            .unwrap();
            let p = store.evaluate(fine).unwrap()[usize::try_from(j * fine.side() + i).unwrap()];
            let at = (
                f64::from(x) / scale - f64::from(region.x0()),
                f64::from(y) / scale - f64::from(region.y0()),
            );
            let s = drawn(tile, at);
            let gap = [
                f64::from(p.x) - s[0],
                f64::from(p.y) - s[1],
                f64::from(p.z) - s[2],
            ];
            let dist = gap.iter().map(|g| g * g).sum::<f64>().sqrt();
            assert!(
                dist <= bound + bound * 1e-12,
                "{what}: {key:?} finest ({x}, {y}) is {dist} off, bound {bound}"
            );
        }
    }
}

/// Mid-stroke, after every dab, each display tile drawn is still a bound on every finest vertex
/// under it; a refresh then replaces the raised ones.
///
/// A tile drawn is the stored one, or the one from before the stroke with its bound raised by
/// twice the dab's largest displacement.
#[test]
fn bumped_bounds_stay_bounds_mid_stroke_until_the_refresh() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let keys: Vec<TileKey> = display_layer(&mut store).iter().map(|t| *t.key()).collect();
    let mut rng = Rng(77);
    let face = FaceId::new(2);
    let n = level(TOP).face_quads();
    store.begin_op().unwrap();
    for dab in 0..6 {
        // A dab: a small bump of finest corrections around a random vertex.
        let (ci, cj) = (rng.below(n + 1), rng.below(n + 1));
        let mut writes = Vec::new();
        for j in cj.saturating_sub(6)..=(cj + 6).min(n) {
            for i in ci.saturating_sub(6)..=(ci + 6).min(n) {
                writes.push((i, j, rng.vec3(0.002).z.abs()));
            }
        }
        let touched: BTreeSet<TileKey> = writes
            .iter()
            .flat_map(|&(i, j, _)| {
                mesh.topology()
                    .copies(face, level(TOP), i, j)
                    .iter()
                    .map(|copy| copy.key(level(TOP)))
                    .collect::<Vec<_>>()
            })
            .collect();
        let touched: Vec<TileKey> = touched.into_iter().collect();
        let before: Vec<Vec<Vec3>> = touched
            .iter()
            .map(|k| store.evaluate(*k).unwrap().to_vec())
            .collect();
        for &(i, j, dz) in &writes {
            let (at, li, lj) =
                TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level(TOP), i, j)
                    .unwrap();
            let c = store.correction(at).unwrap().get(li, lj).unwrap();
            store
                .set_correction(face, level(TOP), i, j, c + Vec3::new(0.0, 0.0, dz))
                .unwrap();
        }
        // The dab's largest displacement, measured.
        let moved = touched
            .iter()
            .zip(&before)
            .map(|(k, was)| {
                let now = store.evaluate(*k).unwrap();
                was.iter()
                    .zip(now)
                    .map(|(a, b)| (*a - *b).length())
                    .fold(0.0_f32, f32::max)
            })
            .fold(0.0_f32, f32::max);
        store.bump_display_error(&touched, moved).unwrap();
        let drawn_tiles = store.display_for_draw(&keys).unwrap();
        let mut bumped = 0;
        for tile in drawn_tiles.iter().flatten() {
            assert_bound(&mut store, tile, &format!("dab {dab}"));
            bumped += usize::from(store.bumped.contains_key(tile.key()));
        }
        assert!(bumped > 0, "dab {dab}: some tile is drawn bumped");
    }
    run_all(&mut store);
    assert!(
        store.bumped.is_empty(),
        "the refresh replaces every bumped tile"
    );
    assert_layer_current(&mut store, "refreshed after the stroke");
}

/// Edits between the slices of a rebuild never leave a tile built from stale copies.
///
/// Under the tightest budget, each edit lands on a finest tile whose values a coarse tile's
/// build has copied so far; the layer ends as the one built from scratch.
///
/// The edits stop after a few: a write empties the copies, so edits more often than a coarse
/// tile takes to gather its inputs hold that tile back until they stop.
#[test]
fn edits_between_slices_never_leave_a_tile_built_from_stale_copies() {
    let dir = TestDir::new();
    let mesh = mesh();
    let display = tight().with_display(level(TOP));
    let config = display.with_budget_bytes(display.budget_bytes + display.floor_bytes(&mesh));
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let mut map = CorrectionMap::new();
    let mut edits = 0_u32;
    let began = Instant::now();
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_micros(200))
            .unwrap();
        // A call shorter than a wait for the workers can take returns at once, so the calls
        // are counted in time, not number.
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the rebuild ends"
        );
        // A finest tile the job in progress has copied, if any.
        let copied = store
            .jobs
            .front_mut()
            .and_then(|job| job.upkeep().copies.tiles.keys().next().copied());
        let Some(tile) = copied.filter(|_| edits < 12) else {
            continue;
        };
        let quads = tile.level().tile_quads();
        let edit = Edit {
            face: tile.face(),
            level: tile.level(),
            i: u32::from(tile.tx()) * quads + 1,
            j: u32::from(tile.ty()) * quads + 1,
            c: Vec3::new(0.01, -0.02, 0.015),
        };
        apply(&mut store, &mut map, &[edit]);
        edits += 1;
    }
    assert_eq!(edits, 12, "every edit landed on copies mid-build");
    assert_layer_current(&mut store, "rebuilt among edits");
}

/// A dab's tiles are drawn with the normal cone that covers everything until the refresh.
///
/// A dab turns the finest normals under the tiles it reaches, so those tiles never face away
/// while their bound is raised; the refresh gives them a tight cone again, the one built from
/// scratch.
#[test]
fn a_dab_makes_the_cones_it_reaches_cover_everything_until_the_refresh() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let face = FaceId::new(1);
    let (ci, cj) = (100, 140);
    let (key, li, lj) = TileKey::containing(
        LayerId::BASE,
        Channel::DISPLACEMENT,
        face,
        level(TOP),
        ci,
        cj,
    )
    .unwrap();
    // A sharp spike: it turns the normals around it well past the cones they had.
    store.begin_op().unwrap();
    let c = store.correction(key).unwrap().get(li, lj).unwrap();
    store
        .set_correction(face, level(TOP), ci, cj, c + Vec3::new(0.0, 0.0, 0.05))
        .unwrap();
    store.bump_display_error(&[key], 0.05).unwrap();
    let bumped: Vec<TileKey> = store.bumped.keys().copied().collect();
    let levels: BTreeSet<Level> = bumped.iter().map(TileKey::level).collect();
    assert_eq!(
        levels.len(),
        usize::from(TOP),
        "the dab reaches a tile at every display level"
    );
    let drawn = store.display_for_draw(&bumped).unwrap();
    for (k, tile) in bumped.iter().zip(&drawn) {
        let tile = tile
            .as_ref()
            .unwrap_or_else(|| panic!("{k:?}: drawn bumped"));
        assert!(
            tile.cone().covers_everything(),
            "{k:?}: {:?} while the dab's tiles are bumped",
            tile.cone()
        );
    }
    run_all(&mut store);
    assert!(store.bumped.is_empty(), "the refresh replaces every bump");
    let refreshed = store.display_for_draw(&bumped).unwrap();
    let finest_below: Vec<&DisplayTile> = refreshed
        .iter()
        .flatten()
        .filter(|t| t.key().level() == level(TOP - 1))
        .collect();
    assert!(
        !finest_below.is_empty(),
        "refreshed one level above the finest"
    );
    for tile in finest_below {
        assert!(
            !tile.cone().covers_everything(),
            "{:?}: a tight cone again",
            tile.key()
        );
    }
    assert_layer_current(&mut store, "refreshed after the dab");
}

/// A write no bump covers (outside a dab, or a lost tile's revert) changes a bumped tile's finest
/// values: its bumped bound no longer bounds it, so it is drawn as unknown, never as bumped.
#[test]
fn a_write_no_bump_covers_drops_the_bumps_it_reaches() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let face = FaceId::new(1);
    let (ci, cj) = (100, 140);
    let at = |i, j| {
        TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level(TOP), i, j).unwrap()
    };
    // A dab: one vertex raised a little, bumped by what it moved.
    store.begin_op().unwrap();
    let (key, li, lj) = at(ci, cj);
    let c = store.correction(key).unwrap().get(li, lj).unwrap();
    store
        .set_correction(face, level(TOP), ci, cj, c + Vec3::new(0.0, 0.0, 0.001))
        .unwrap();
    store.bump_display_error(&[key], 0.01).unwrap();
    let bumped: Vec<TileKey> = store
        .bumped
        .keys()
        .copied()
        .filter(|k| k.level() == level(TOP - 1))
        .collect();
    assert!(
        !bumped.is_empty(),
        "the dab bumps tiles one level above the finest"
    );
    assert!(
        store
            .display_for_draw(&bumped)
            .unwrap()
            .iter()
            .all(Option::is_some),
        "drawn bumped"
    );
    // A large write to the same vertex, which no bump covers.
    store
        .set_correction(face, level(TOP), ci, cj, c + Vec3::new(0.0, 0.0, 0.5))
        .unwrap();
    for (k, drawn) in bumped.iter().zip(store.display_for_draw(&bumped).unwrap()) {
        assert!(
            drawn.is_none(),
            "{k:?}: unknown, not drawn with a stale bump"
        );
    }
}

/// A display tile that won't build is left unknown, reported and skipped, never a stall.
///
/// So is every tile reading it (as one from extreme geometry wouldn't build): every other tile
/// is built, the jobs finish, and every display tile is known or unknown. It is tried again
/// after the next change.
#[test]
fn a_tile_that_wont_build_is_skipped_not_a_stall() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let face = FaceId::new(3);
    let bad = TileKey::display(face, level(TOP - 1), 1, 1).unwrap();
    store.hooks.unbuildable_display = Some(bad);
    // An edit under the tile: it and the tiles above it are to be built again.
    store
        .set_correction(face, level(TOP), 200, 200, Vec3::new(0.0, 0.0, 0.01))
        .unwrap();
    let mut failed = 0;
    let mut calls = 0;
    while store.has_jobs() || calls == 0 {
        failed += store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap()
            .failed;
        calls += 1;
        assert!(calls < 1000, "the jobs end");
    }
    let failures = store.failed_display();
    assert!(failures.iter().any(|(k, _)| *k == bad), "the tile failed");
    assert!(failures.len() > 1, "and the tiles reading it");
    assert_eq!(failures.len(), failed, "each failure reported once");
    let keys = display_layer_keys(&mesh);
    let failed_keys: BTreeSet<TileKey> = failures.iter().map(|(k, _)| *k).collect();
    for (key, kept) in keys.iter().zip(store.stored_display(&keys).unwrap()) {
        assert_eq!(
            kept.is_none(),
            failed_keys.contains(key),
            "{key:?}: unknown exactly when it failed"
        );
    }
    // A change: the failed tiles are tried again, and build.
    store.hooks.unbuildable_display = None;
    store
        .set_correction(face, level(TOP), 200, 200, Vec3::ZERO)
        .unwrap();
    assert!(store.failed_display().is_empty(), "retried after a change");
    run_all(&mut store);
    assert_layer_current(&mut store, "built again");
}

/// Every display tile key of `mesh` under [`TOP`].
fn display_layer_keys(mesh: &BaseMesh) -> Vec<TileKey> {
    tiles(mesh, 0, TOP - 1)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect()
}

/// A refresh dropped by `clear_jobs` is queued again by the next `run_jobs`, with no write in
/// between: the unknown tiles it left are still to build.
#[test]
fn a_refresh_cleared_is_queued_again() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 3, 71));
    // One short call queues the refresh and starts it; then it is dropped.
    store.run_jobs(Instant::now()).unwrap();
    assert!(store.has_jobs(), "the refresh is queued");
    store.clear_jobs();
    assert!(!store.has_jobs(), "cleared");
    run_all(&mut store);
    assert_layer_current(&mut store, "refreshed after a clear");
}

/// The decoded display tiles a refresh kept go once a call finds nothing to do, and the caches
/// are sized from the budget's share for them.
#[test]
fn the_display_caches_follow_the_budget_and_empty_when_idle() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 3, 73));
    while store.has_jobs() {
        store
            .run_jobs(Instant::now() + Duration::from_secs(60))
            .unwrap();
    }
    // A call with nothing left to do; the one before may still have found a last tile to
    // look at.
    let mut idle = 0;
    while store
        .run_jobs(Instant::now() + Duration::from_secs(1))
        .unwrap()
        .slices
        > 0
    {
        idle += 1;
        assert!(idle < 10, "the jobs run out");
    }
    assert!(
        store.display_decoded.tiles.is_empty(),
        "nothing to do: the decoded tiles are let go ({} left, jobs {})",
        store.display_decoded.tiles.len(),
        store.has_jobs()
    );
    let caps = crate::jobs::CacheCaps::of(0);
    assert_eq!(
        caps.copies,
        crate::jobs::CacheCaps::MIN_COPIES,
        "a tiny share: the least"
    );
    let big = crate::jobs::CacheCaps::of(StoreConfig::MAX_DISPLAY_CACHE_BYTES);
    assert!(
        big.decoded > caps.decoded && big.copies > caps.copies,
        "a large share: more"
    );
    let config = StoreConfig::new(1 << 30, 1).with_display(level(TOP));
    assert_eq!(
        config.display_cache_bytes(),
        (1 << 30) / 16,
        "a sixteenth of the budget"
    );
    let huge = StoreConfig::new(64 << 30, 1).with_display(level(TOP));
    assert_eq!(
        huge.display_cache_bytes(),
        StoreConfig::MAX_DISPLAY_CACHE_BYTES,
        "at most the cap"
    );
    assert_eq!(
        StoreConfig::new(1 << 30, 1).display_cache_bytes(),
        0,
        "no layer, no caches"
    );
}

/// Runs the store's jobs in short calls until `done` holds of it; panics after a while.
fn run_until(store: &mut TileStore, what: &str, done: impl Fn(&mut TileStore) -> bool) {
    let began = Instant::now();
    while !done(store) {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
        assert!(began.elapsed() < Duration::from_secs(30), "{what}");
    }
}

/// A display tile the workers built from inputs a write has since changed is built again, not
/// recorded: its result is held back until after a write under it, then let go.
#[test]
fn a_build_finished_after_a_write_is_built_again() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    run_all(&mut store);
    let mut map = CorrectionMap::new();
    let edits = random_edits(&mesh, TOP, 2, 81);
    apply(&mut store, &mut map, &edits);
    store.hooks.hold_finished = true;
    run_until(&mut store, "a build is handed out", |store| {
        store
            .jobs
            .front_mut()
            .is_some_and(|job| !job.upkeep().in_flight.is_empty())
            && !store.hooks.held_finished.is_empty()
    });
    // A vertex under the first tile held, moved while it is out: the tile it was built for
    // has changed.
    let held = store
        .hooks
        .held_finished
        .iter()
        .find(|done| !done.decode)
        .map(|done| done.key)
        .unwrap();
    let region = Region::of_tile(&held);
    let shift = TOP - held.level().get();
    let centre = |v: u32| (v + region.quads() / 2) << shift;
    apply(
        &mut store,
        &mut map,
        &[Edit {
            face: held.face(),
            level: level(TOP),
            i: centre(region.x0()),
            j: centre(region.y0()),
            c: Vec3::new(0.03, -0.02, 0.04),
        }],
    );
    store.hooks.hold_finished = false;
    run_all(&mut store);
    assert_layer_current(&mut store, "built again after the write");
}

/// A finer display tile decoded from a record a build has since replaced is dropped.
///
/// The record's generation tells it, so the decode is never kept for the tiles above: it is held
/// back until after the build, then let go.
#[test]
fn a_decode_finished_after_its_record_was_replaced_is_dropped() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut saving = display_store(&dir, &mesh);
    run_all(&mut saving);
    saving.save().unwrap();
    drop(saving);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut store = TileStore::open(file, roomy().with_display(level(TOP))).unwrap();
    run_all(&mut store);
    // A coarse tile unknown: its refresh decodes the finer tiles it reads from the file.
    let coarse = TileKey::display(FaceId::new(0), level(TOP - 2), 0, 0).unwrap();
    store.display.set(coarse, None);
    store.hooks.hold_finished = true;
    run_until(&mut store, "a decode is handed out", |store| {
        store
            .jobs
            .front_mut()
            .is_some_and(|job| !job.upkeep().decoding.is_empty())
            && store.hooks.held_finished.iter().any(|done| done.decode)
    });
    let decoded: Vec<TileKey> = store
        .hooks
        .held_finished
        .iter()
        .filter(|done| done.decode)
        .map(|done| done.key)
        .collect();
    // An edit under them, and their tiles built and recorded anew: other records, a newer
    // generation.
    let finer = decoded[0];
    let region = Region::of_tile(&finer);
    let shift = TOP - finer.level().get();
    let (i, j) = ((region.x0() + 1) << shift, (region.y0() + 1) << shift);
    let mut map = CorrectionMap::new();
    apply(
        &mut store,
        &mut map,
        &[Edit {
            face: finer.face(),
            level: level(TOP),
            i,
            j,
            c: Vec3::new(0.02, -0.01, 0.03),
        }],
    );
    let layer = display_layer(&mut store);
    let rebuilt: Vec<DisplayTile> = layer
        .iter()
        .filter(|tile| decoded.contains(tile.key()))
        .cloned()
        .collect();
    store.record_display(&rebuilt).unwrap();
    store.hooks.hold_finished = false;
    // The held decodes arrive at the next slice: each is of a replaced record, and dropped.
    store
        .run_jobs(Instant::now() + Duration::from_millis(2))
        .unwrap();
    for tile in &rebuilt {
        let kept = store.display_decoded.tiles.get(tile.key());
        assert!(
            kept.is_none_or(|(generation, kept)| {
                *generation == store.display_generation(tile.key()) && kept.tile == *tile
            }),
            "{:?}: never the decode of the record it replaced",
            tile.key()
        );
    }
    run_all(&mut store);
    assert_layer_current(&mut store, "the coarse tile built from the new records");
}

/// The bytes a store's display layer, built whole, encodes to: for sizing a share in tests.
fn layer_bytes(store: &mut TileStore) -> usize {
    let layer = display_layer(store);
    let sizes = layer.iter().map(|tile| tile.encoded().len());
    sizes.sum()
}

/// A rebuild holds the display records waiting for the working file to their share.
///
/// The rest are written through between flushes; the layer saves and reopens bit for bit, and
/// edits after it still make unknown what they change, written-through records included.
#[test]
fn a_rebuild_holds_the_records_waiting_for_the_file_to_their_share() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 30, 11));
    // A quarter of the layer: most of it must go through.
    let share = layer_bytes(&mut store) / 4;
    store.hold_display_waiting_to(share);
    let mut most = 0;
    let mut calls = 0;
    while store.has_jobs() || calls == 0 {
        store
            .run_jobs(Instant::now() + Duration::from_millis(2))
            .unwrap();
        let (waiting, held_to) = store.display_waiting();
        assert_eq!(held_to, share, "the share set");
        assert!(
            waiting <= share,
            "call {calls}: {waiting} bytes waiting, held to {share}"
        );
        most = most.max(waiting);
        calls += 1;
        assert!(calls < 100_000, "the rebuild ends");
    }
    assert!(most > 0, "records waited in memory");
    let written = store.working().display_keys().count();
    assert!(
        written > 0,
        "records went to the working file before any save ({written})"
    );
    assert_layer_current(&mut store, "rebuilt, written through");
    // An edit after, inside one face: the written-through records it changes are unknown.
    // Saved at once, before any refresh, the file holds each display tile current or unknown.
    let edit = Edit {
        face: FaceId::new(0),
        level: level(TOP),
        i: 100,
        j: 90,
        c: Vec3::new(0.01, -0.02, 0.03),
    };
    apply(&mut store, &mut map, &[edit]);
    store.save().unwrap();
    drop(store);
    let reopen = || {
        let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
        TileStore::open(file, roomy().with_display(level(TOP))).unwrap()
    };
    let mut opened = reopen();
    assert_layer_current_or_unknown(&mut opened, "saved straight after the edits");
    assert!(opened.has_jobs(), "the unknown tiles are refreshed");
    opened.hold_display_waiting_to(share);
    run_all(&mut opened);
    assert!(
        opened.display_waiting().0 <= share,
        "the refresh holds to the share too"
    );
    assert_layer_current(&mut opened, "refreshed over written-through records");
    let built = display_layer(&mut opened);
    opened.save().unwrap();
    drop(opened);
    let mut reopened = reopen();
    assert!(!reopened.has_jobs(), "nothing to rebuild");
    let keys: Vec<TileKey> = built.iter().map(|t| *t.key()).collect();
    let stored = reopened.stored_display(&keys).unwrap();
    let saved = built.iter().zip(stored);
    for (tile, back) in saved {
        let back = back.unwrap_or_else(|| panic!("{:?} saved", tile.key()));
        assert_eq!(&back, tile, "{:?} bit for bit", tile.key());
    }
}

/// Asserts that every display tile the store knows is the one its tiles build, bit for bit,
/// and that it knows some.
fn assert_layer_current_or_unknown(store: &mut TileStore, what: &str) {
    let built = display_layer(store);
    let keys: Vec<TileKey> = built.iter().map(|t| *t.key()).collect();
    let stored = store.stored_display(&keys).unwrap();
    let mut known = 0;
    let pairs = built.iter().zip(stored);
    for (tile, kept) in pairs {
        if let Some(kept) = kept {
            assert_eq!(&kept, tile, "{what}: {:?} current or unknown", tile.key());
            known += 1;
        }
    }
    assert!(known > 0, "{what}: some display tiles known");
}

/// The share of the budget the waiting display records are held to: a thirty-second, at most
/// the cap, and nothing without a display layer; a roomy budget gives all of it.
#[test]
fn the_waiting_records_share_follows_the_budget() {
    let config = StoreConfig::new(1 << 30, 1).with_display(level(TOP));
    assert_eq!(
        config.display_waiting_bytes(),
        (1 << 30) / 32,
        "a thirty-second of the budget"
    );
    let huge = StoreConfig::new(64 << 30, 1).with_display(level(TOP));
    assert_eq!(
        huge.display_waiting_bytes(),
        StoreConfig::MAX_DISPLAY_WAITING_BYTES,
        "at most the cap"
    );
    assert_eq!(
        StoreConfig::new(1 << 30, 1).display_waiting_bytes(),
        0,
        "no layer, no share"
    );
    let dir = TestDir::new();
    let mesh = mesh();
    let store = display_store(&dir, &mesh);
    assert_eq!(
        store.display_waiting().1,
        roomy().with_display(level(TOP)).display_waiting_bytes(),
        "a roomy budget gives the whole share"
    );
}
