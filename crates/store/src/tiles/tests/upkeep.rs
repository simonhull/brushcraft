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

#[test]
fn run_jobs_honours_its_deadline_within_one_slice() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = display_store(&dir, &mesh);
    let budget = Duration::from_micros(300);
    let mut calls = 0;
    let began = Instant::now();
    while store.has_jobs() {
        let start = Instant::now();
        let report = store.run_jobs(start + budget).unwrap();
        let spent = start.elapsed();
        assert!(
            spent <= budget + report.longest_slice + Duration::from_millis(1),
            "call {calls}: {spent:?} for {budget:?}, longest slice {:?}",
            report.longest_slice
        );
        calls += 1;
        // A call shorter than a wait for the workers can take returns at once, so the calls
        // are counted in time, not number.
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "the rebuild ends"
        );
    }
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
    store
        .jobs
        .push_back(Job::Refresh(Upkeep::new(level(TOP), [coarse])));
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
