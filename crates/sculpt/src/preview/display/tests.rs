//! Display previews against the finest work.
//!
//! The bound holds against the finest result, and doesn't grow with a stroke's length; border
//! vertices are the finest previews' bits (so seams close at every level); the landed finest work
//! replaces them exactly; the saved file is that of the dabs applied at once; and the normals
//! follow the moved surface.

use std::collections::BTreeMap;

use brush::Imprint;
use geom::{Channel, FaceId, Level, TileKey, Vec3};
use store::{DisplayPreview, Project, SpatialIndex, TileStore};
use stroke::Dab;

use super::{cell_box, cells_touched, finest_place, quad_box};
use crate::apply::apply_dabs;
use crate::preview::tests::{at_once, base, level, open_copy, path, previewed_drawing, run_jobs};
use crate::preview::{DrawsAll, Preview, WAITING_A_CALL};
use crate::state::Stroke;
use crate::testing::{TestDir, dab};

/// A renderer drawing display tiles alone.
fn display_only(key: &TileKey) -> bool {
    key.channel() == Channel::DISPLAY
}

/// Every display preview the store holds, by tile.
fn previews(store: &TileStore) -> BTreeMap<TileKey, DisplayPreview> {
    store
        .display_previewed()
        .filter_map(|key| store.display_preview(key).map(|p| (*key, p.clone())))
        .collect()
}

/// The point display tile `key`'s triangles over `positions` draw at finest grid place `(gx, gy)`.
///
/// On its face, the finest level `finest`, in `f64`: the quad holding it, split along its
/// `(i, j)`–`(i+1, j+1)` diagonal, as every level is drawn.
fn drawn_at(key: &TileKey, positions: &[Vec3], finest: Level, gx: u32, gy: u32) -> [f64; 3] {
    let quads = key.side() - 1;
    let step = f64::from(1_u32 << (finest.get() - key.level().get()));
    let across = f64::from(gx) / step - f64::from(u32::from(key.tx()) * quads);
    let down = f64::from(gy) / step - f64::from(u32::from(key.ty()) * quads);
    let last = f64::from(quads - 1);
    let (col, row) = (
        across.floor().clamp(0.0, last),
        down.floor().clamp(0.0, last),
    );
    let (fx, fy) = (across - col, down - row);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a quad index within the tile, clamped"
    )]
    let (i, j) = (col as usize, row as usize);
    let side = usize::try_from(key.side()).unwrap();
    let p = |a: usize, b: usize| {
        let v = positions[b * side + a];
        [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
    };
    let (p00, p10, p01, p11) = (p(i, j), p(i + 1, j), p(i, j + 1), p(i + 1, j + 1));
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = if fx >= fy {
            p00[c] + fx * (p10[c] - p00[c]) + fy * (p11[c] - p10[c])
        } else {
            p00[c] + fy * (p01[c] - p00[c]) + fx * (p11[c] - p01[c])
        };
    }
    out
}

/// The farthest any finest vertex under display tile `key` lies from what `preview` draws at its
/// place, the finest level as `store` holds it.
fn farthest(store: &mut TileStore, key: &TileKey, preview: &DisplayPreview) -> f64 {
    let finest = store.display_finest().unwrap();
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let span = quads * step / 64;
    let mut worst = 0.0_f64;
    for fy in 0..span {
        for fx in 0..span {
            let (tx, ty) = (
                u32::from(key.tx()) * span + fx,
                u32::from(key.ty()) * span + fy,
            );
            let tile = TileKey::displacement(
                key.face(),
                finest,
                u16::try_from(tx).unwrap(),
                u16::try_from(ty).unwrap(),
            )
            .unwrap();
            let positions = store.evaluate(tile).unwrap().to_vec();
            for b in 0..=64 {
                for a in 0..=64 {
                    let f = positions[usize::try_from(b * 65 + a).unwrap()];
                    let d = drawn_at(key, &preview.positions, finest, tx * 64 + a, ty * 64 + b);
                    let gap = (f64::from(f.x) - d[0])
                        .hypot(f64::from(f.y) - d[1])
                        .hypot(f64::from(f.z) - d[2]);
                    worst = worst.max(gap);
                }
            }
        }
    }
    worst
}

/// Asserts that every preview of `shown` bounds the finest level `store` holds now (the
/// stroke's dabs all landed); returns the largest error over bound.
fn assert_bounds(
    store: &mut TileStore,
    shown: &BTreeMap<TileKey, DisplayPreview>,
    what: &str,
) -> f64 {
    assert!(!shown.is_empty(), "{what}: previews made");
    let mut tightest = 0.0_f64;
    for (key, preview) in shown {
        let error = farthest(store, key, preview);
        let bound = f64::from(preview.bound);
        assert!(
            error <= bound,
            "{what}: {key:?} is {error} from the finest result, past its bound {bound}"
        );
        tightest = tightest.max(error / bound);
    }
    tightest
}

#[test]
fn a_display_preview_bounds_the_finest_result_shown_and_part_landed() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "bound");
    let mut dabs = path(project.store(), 14);
    // And dabs where each coarse level's bound is tightest: a level's finest vertex farthest
    // from what it draws.
    for level in 0..4 {
        let coarse = TileKey::display(FaceId::new(0), Level::new(level).unwrap(), 0, 0).unwrap();
        let at = worst_place(project.store(), &coarse);
        dabs.extend((0..2).map(|n| dab(at, 0.08, 0.02, 0.0, 14 + 2 * u32::from(level) + n)));
    }
    let (first, rest) = dabs.split_at(7);
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, &dabs, &display_only)
        .unwrap();
    let shown = previews(project.store());
    assert!(
        shown.values().all(|p| p.held),
        "every preview is held while its dabs are to land"
    );
    apply_dabs(project.store(), &mut index, &mut stroke, first).unwrap();
    preview
        .land(project.store(), &index, first.len(), &display_only)
        .unwrap();
    let part = previews(project.store());
    apply_dabs(project.store(), &mut index, &mut stroke, rest).unwrap();
    preview
        .land(project.store(), &index, rest.len(), &display_only)
        .unwrap();
    let landed = previews(project.store());
    let store = project.store();
    let shown_ratio = assert_bounds(store, &shown, "shown");
    let part_ratio = assert_bounds(store, &part, "part landed");
    let landed_ratio = assert_bounds(store, &landed, "landed");
    eprintln!(
        "largest error / bound: shown {shown_ratio:.3}, part landed {part_ratio:.3}, landed \
         {landed_ratio:.3}"
    );
    // The bound is near the error at every level, not only at the tile that is worst.
    let mut per_level: BTreeMap<u8, f64> = BTreeMap::new();
    for (key, made) in &shown {
        let ratio = farthest(store, key, made) / f64::from(made.bound);
        let most = per_level.entry(key.level().get()).or_insert(0.0);
        *most = most.max(ratio);
    }
    eprintln!("largest error / bound by level: {per_level:?}");
    for (level, ratio) in &per_level {
        assert!(
            *ratio > 0.2,
            "level {level}: a bound near the error: {ratio}"
        );
    }
    assert!(
        landed.values().all(|p| !p.held),
        "released once every dab landed"
    );
    drop(project.close(false));
}

#[test]
fn border_vertices_are_the_finest_previews_bits_and_the_landed_work_replaces_them_exactly() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "seams");
    let dabs = path(project.store(), 10);
    let (first, rest) = dabs.split_at(4);
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, &dabs, &DrawsAll)
        .unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, first).unwrap();
    preview
        .land(project.store(), &index, first.len(), &DrawsAll)
        .unwrap();
    let finest = level();
    // Mid-stroke and once landed: every border vertex of every display preview, at every level
    // and on every face, holds the bits of the finest vertex as drawn (its preview's, else the
    // store's), as every other tile holding it does.
    for (stage, landing) in [("mid-stroke", None), ("landed", Some(rest))] {
        if let Some(last) = landing {
            apply_dabs(project.store(), &mut index, &mut stroke, last).unwrap();
            preview
                .land(project.store(), &index, last.len(), &DrawsAll)
                .unwrap();
        }
        let store = project.store();
        let shown = previews(store);
        assert!(shown.len() > 4, "{stage}: previews at several levels");
        let mut checked = 0;
        for (key, made) in &shown {
            let side = key.side();
            for j in 0..side {
                for i in 0..side {
                    let border = i == 0 || j == 0 || i + 1 == side || j + 1 == side;
                    let (under, a, b) = finest_place(key, finest, i, j).unwrap();
                    let at = |p: Vec3| p.to_bits();
                    let want = match store.preview(&under) {
                        Some(block) => block.get(i64::from(a), i64::from(b)),
                        None => {
                            store.evaluate(under).unwrap()[usize::try_from(b * 65 + a).unwrap()]
                        }
                    };
                    let got = made.positions[usize::try_from(j * side + i).unwrap()];
                    if border {
                        assert_eq!(
                            at(got),
                            at(want),
                            "{stage}: {key:?} ({i}, {j}) is the finest bits"
                        );
                        checked += 1;
                    } else if landing.is_some() && store.pen_down_shape(&under).is_some() {
                        // Inside a finest tile the stroke wrote: the store's bits.
                        assert_eq!(
                            at(got),
                            at(want),
                            "landed: {key:?} ({i}, {j}) rebased exactly"
                        );
                    }
                }
            }
        }
        assert!(checked > 0);
    }
    preview.end_stroke(project.store()).unwrap();
    project.end_unrecorded_stroke().unwrap();
    run_jobs(project.store());
    assert!(
        project.store().display_previewed().next().is_none(),
        "the refresh replaced every released preview"
    );
    drop(project.close(false));
}

#[test]
fn dabs_shown_on_display_tiles_then_landed_save_the_file_the_dabs_applied_at_once_save() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut finest, mut finest_index, finest_path) = open_copy(&dir, &base, "finest");
    let dabs = path(finest.store(), 14);
    at_once(&mut finest, &mut finest_index, &dabs);
    run_jobs(finest.store());
    drop(finest.save().unwrap());
    finest.close(false).unwrap();
    let (mut shown, mut shown_index, shown_path) = open_copy(&dir, &base, "shown");
    previewed_drawing(&mut shown, &mut shown_index, &dabs, 3, &DrawsAll);
    run_jobs(shown.store());
    drop(shown.save().unwrap());
    shown.close(false).unwrap();
    let (a, b) = (
        std::fs::read(finest_path).unwrap(),
        std::fs::read(shown_path).unwrap(),
    );
    assert!(
        a == b,
        "display previews leave nothing in the file: bit for bit"
    );
}

/// A long, strong stroke along face 0's middle row: `count` dabs of radius 0.12, a tenth of it
/// strong, 0.15 radii apart, on the surface as `store` holds it.
fn ridge(store: &mut TileStore, count: u32) -> Vec<Dab> {
    (0..count)
        .map(|n| {
            let key = TileKey::displacement(FaceId::new(0), level(), u16::from(n >= count / 2), 0)
                .unwrap();
            let i = (n * 6) % 64;
            let centre = crate::testing::vertex(store, key, i, 64);
            dab(centre, 0.12, 0.012, 0.0, n)
        })
        .collect()
}

#[test]
fn a_long_strokes_bound_follows_the_dabs_over_each_place_not_its_length() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, index, _) = open_copy(&dir, &base, "long");
    let dabs = ridge(project.store(), 20);
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    let coarse = TileKey::display(FaceId::new(0), Level::new(4).unwrap(), 0, 0).unwrap();
    let mut bounds = Vec::new();
    for chunk in dabs.chunks(5) {
        preview
            .show(project.store(), &index, chunk, &display_only)
            .unwrap();
        bounds.push(project.store().display_preview(&coarse).unwrap().bound);
    }
    eprintln!("the coarse tile's bound after each five dabs: {bounds:?}");
    let (first, last) = (bounds[0], *bounds.last().unwrap());
    assert!(
        last < 2.0 * first,
        "four times the dabs, not four times the bound: {first} → {last}"
    );
    drop(project.close(false));
}

/// The angle between `a` and `b`, in degrees.
fn degrees(a: Vec3, b: Vec3) -> f64 {
    f64::from(a.dot(b).clamp(-1.0, 1.0)).acos().to_degrees()
}

#[test]
fn a_previews_normals_follow_the_moved_surface_as_the_refresh_makes_them() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "normals");
    let dabs = ridge(project.store(), 20);
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, &dabs, &display_only)
        .unwrap();
    let shown = previews(project.store());
    let keys: Vec<TileKey> = shown.keys().copied().collect();
    let before = project.store().stored_display(&keys).unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, &dabs).unwrap();
    preview
        .land(project.store(), &index, dabs.len(), &display_only)
        .unwrap();
    preview.end_stroke(project.store()).unwrap();
    project.end_unrecorded_stroke().unwrap();
    run_jobs(project.store());
    let store = project.store();
    let refreshed = store.stored_display(&keys).unwrap();
    for (((key, made), tile), old) in shown.iter().zip(refreshed).zip(before) {
        let (tile, old) = (tile.expect("refreshed"), old.expect("recorded"));
        let (mut sum, mut stale, mut count) = (0.0, 0.0, 0.0);
        for ((n, want), was) in made.normals.iter().zip(tile.normals()).zip(old.normals()) {
            let angle = degrees(*n, *want);
            sum += angle * angle;
            stale += degrees(*was, *want).powi(2);
            count += 1.0;
        }
        let rms = (sum / count).sqrt();
        let stale = (stale / count).sqrt();
        eprintln!(
            "{key:?}: preview normals {rms:.2}° rms off the refreshed tile's (the record before, {stale:.2}°)"
        );
        assert!(rms < 10.0, "{key:?}: {rms}° rms (stale {stale})");
    }
}

#[test]
fn the_vertices_in_a_finest_tile_are_those_placed_in_it() {
    let finest = Level::new(9).unwrap();
    for level in [0_u8, 1, 3, 6, 7, 8] {
        let level = Level::new(level).unwrap();
        let per_side = u16::try_from(level.tiles_per_side()).unwrap();
        let key = TileKey::display(FaceId::new(2), level, per_side - 1, 0).unwrap();
        let side = key.side();
        for tile in super::finest_under(&key, finest) {
            let found = super::vertices_in(&key, finest, &tile);
            for (v, col, row) in &found {
                let (i, j) = (
                    u32::try_from(*v).unwrap() % side,
                    u32::try_from(*v).unwrap() / side,
                );
                let (placed, a, b) = finest_place(&key, finest, i, j).unwrap();
                // A vertex on a finest tile's border is in both tiles; placed in one of them.
                if placed == tile {
                    assert_eq!((a, b), (*col, *row), "{level:?}: ({i}, {j})");
                }
                assert!(*col <= 64 && *row <= 64);
            }
            let placed_here = (0..side * side)
                .filter(|v| {
                    let (i, j) = (v % side, v / side);
                    finest_place(&key, finest, i, j).unwrap().0 == tile
                })
                .count();
            assert!(
                found.len() >= placed_here,
                "{level:?}: every vertex placed in it is found"
            );
        }
    }
}

/// The finest vertex of `key`'s face farthest from what the display tile's record draws at its
/// place, as `store` holds it: where a coarse tile's bound is tightest.
fn worst_place(store: &mut TileStore, key: &TileKey) -> Vec3 {
    let finest = store.display_finest().unwrap();
    let record = store
        .display_for_draw(&[*key])
        .unwrap()
        .pop()
        .flatten()
        .unwrap();
    let quads = key.side() - 1;
    let step = 1_u32 << (finest.get() - key.level().get());
    let span = quads * step / 64;
    let mut worst = (0.0_f64, Vec3::ZERO);
    for fy in 0..span {
        for fx in 0..span {
            let (tx, ty) = (
                u32::from(key.tx()) * span + fx,
                u32::from(key.ty()) * span + fy,
            );
            let tile = TileKey::displacement(
                key.face(),
                finest,
                u16::try_from(tx).unwrap(),
                u16::try_from(ty).unwrap(),
            )
            .unwrap();
            let positions = store.evaluate(tile).unwrap().to_vec();
            for b in 0..=64 {
                for a in 0..=64 {
                    let f = positions[usize::try_from(b * 65 + a).unwrap()];
                    let d = drawn_at(key, record.positions(), finest, tx * 64 + a, ty * 64 + b);
                    let gap = (f64::from(f.x) - d[0])
                        .hypot(f64::from(f.y) - d[1])
                        .hypot(f64::from(f.z) - d[2]);
                    if gap > worst.0 {
                        worst = (gap, f);
                    }
                }
            }
        }
    }
    worst.1
}

/// Shows `dabs` on display tiles, applies and lands them, and asserts that every preview made
/// (as shown) bounds the finest result.
fn shown_then_landed(project: &mut Project, index: &mut SpatialIndex, dabs: &[Dab], what: &str) {
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), index, dabs, &display_only)
        .unwrap();
    let shown = previews(project.store());
    apply_dabs(project.store(), index, &mut stroke, dabs).unwrap();
    preview
        .land(project.store(), index, dabs.len(), &display_only)
        .unwrap();
    let ratio = assert_bounds(project.store(), &shown, what);
    eprintln!("{what}: largest error / bound {ratio:.3}");
    preview.end_stroke(project.store()).unwrap();
    project.end_unrecorded_stroke().unwrap();
}

#[test]
fn a_coarse_tile_counts_the_dabs_at_finest_vertices_bulging_past_its_own_box() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "bulge");
    let coarse = TileKey::display(FaceId::new(0), Level::ZERO, 0, 0).unwrap();
    let at = worst_place(project.store(), &coarse);
    let dabs: Vec<Dab> = (0..8).map(|n| dab(at, 0.05, 0.02, 0.0, n)).collect();
    shown_then_landed(&mut project, &mut index, &dabs, "bulging past level 0");
    drop(project.close(false));
}

#[test]
fn a_record_refreshed_mid_stroke_still_gives_its_preview_the_pen_down_surface() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "refreshed");
    // Six strong dabs over one place, then the same six again: what the second six move was
    // moved, and refreshed, already.
    let key = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let at = crate::testing::vertex(project.store(), key, 50, 50);
    let first: Vec<Dab> = (0..6).map(|n| dab(at, 0.15, 0.04, 0.0, n)).collect();
    let rest: Vec<Dab> = first
        .iter()
        .map(|d| Dab {
            seq: d.seq + 6,
            ..*d
        })
        .collect();
    let (first, rest) = (first.as_slice(), rest.as_slice());
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    // Six dabs applied at once, then the refresh rebuilds the records over them.
    apply_dabs(project.store(), &mut index, &mut stroke, first).unwrap();
    preview
        .note_applied(project.store(), &index, first)
        .unwrap();
    run_jobs(project.store());
    // Six more shown, then landed: the previews are made over the rebuilt records.
    preview
        .show(project.store(), &index, rest, &display_only)
        .unwrap();
    let shown = previews(project.store());
    apply_dabs(project.store(), &mut index, &mut stroke, rest).unwrap();
    preview
        .land(project.store(), &index, rest.len(), &display_only)
        .unwrap();
    let ratio = assert_bounds(project.store(), &shown, "over records rebuilt mid-stroke");
    eprintln!("over records rebuilt mid-stroke: largest error / bound {ratio:.3}");
    drop(project.close(false));
}

#[test]
fn a_display_preview_left_undrawn_for_a_call_still_takes_its_dabs() {
    let dir = TestDir::new();
    let base = base(&dir);
    let dabs = path_dabs(&dir, &base);
    // Every call drawing display tiles, against one call that draws none of them.
    let made = |name: &str, middle: &dyn Fn(&TileKey) -> bool| {
        let (mut project, index, _) = open_copy(&dir, &base, name);
        let mut preview = Preview::new();
        project.begin_stroke().unwrap();
        for (n, dab) in dabs.iter().enumerate() {
            let drawn = |key: &TileKey| if n == 1 { middle(key) } else { true };
            preview
                .show(project.store(), &index, std::slice::from_ref(dab), &drawn)
                .unwrap();
        }
        let shown = previews(project.store());
        drop(project.close(false));
        shown
    };
    let always = made("always", &|_: &TileKey| true);
    let skipped = made("skipped", &|key: &TileKey| {
        key.channel() != Channel::DISPLAY
    });
    assert!(!always.is_empty());
    for (key, want) in &always {
        let got = skipped.get(key).expect("previewed in both");
        assert_eq!(
            got.positions, want.positions,
            "{key:?} took the dab it wasn't drawn for"
        );
        assert_eq!(
            got.bound.to_bits(),
            want.bound.to_bits(),
            "{key:?}'s bound too"
        );
    }
}

/// Three dabs over face 0, on `base`'s surface.
fn path_dabs(dir: &TestDir, base: &std::path::Path) -> Vec<Dab> {
    let (mut project, _, _) = open_copy(dir, base, "dabs");
    let dabs = path(project.store(), 3);
    drop(project.close(false));
    dabs
}

#[test]
fn a_preview_that_moved_nothing_over_an_unchanged_tile_is_gone_with_its_stroke() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "unmoved");
    // A tiny dab hovering off the surface: inside display tiles' grown boxes, moving nothing.
    let finest_key = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let on = crate::testing::vertex(project.store(), finest_key, 31, 31);
    let off = on * 1.01;
    let dabs = vec![dab(off, 0.002, 0.01, 0.0, 0)];
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, &dabs, &display_only)
        .unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, &dabs).unwrap();
    preview
        .land(project.store(), &index, 1, &display_only)
        .unwrap();
    preview.end_stroke(project.store()).unwrap();
    project.end_unrecorded_stroke().unwrap();
    let store = project.store();
    let left: Vec<TileKey> = store.display_previewed().copied().collect();
    for key in &left {
        assert!(
            store.display_changes(key).is_some(),
            "{key:?}: a preview outliving its stroke is over a tile the refresh will rebuild"
        );
    }
    drop(project.close(false));
}

#[test]
fn display_previews_are_counted_with_the_pen_down_data_and_go_with_it() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "bytes");
    let dabs = path(project.store(), 6);
    let before = project.store().pen_down_bytes();
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, &dabs, &display_only)
        .unwrap();
    let shown = previews(project.store());
    let vertices: usize = shown.values().map(|p| p.positions.len()).sum();
    let counted = project.store().pen_down_bytes() - before;
    assert!(
        counted > vertices * 25,
        "{counted} bytes counted for {vertices} vertices: the previews and what makes them"
    );
    apply_dabs(project.store(), &mut index, &mut stroke, &dabs).unwrap();
    preview
        .land(project.store(), &index, dabs.len(), &display_only)
        .unwrap();
    preview.end_stroke(project.store()).unwrap();
    project.end_unrecorded_stroke().unwrap();
    run_jobs(project.store());
    assert_eq!(
        project.store().pen_down_bytes(),
        0,
        "refreshed: nothing kept"
    );
    drop(project.close(false));
}

#[test]
fn the_previews_a_renderer_waits_for_are_made_a_few_a_call() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, index, _) = open_copy(&dir, &base, "waiting");
    let dabs = path(project.store(), 6);
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    // A renderer drawing nothing yet: every held display tile waits.
    preview
        .show(project.store(), &index, &dabs, &|_: &TileKey| false)
        .unwrap();
    let store = project.store();
    let waiting: Vec<TileKey> = store
        .held_previews()
        .filter(|key| key.channel() == Channel::DISPLAY)
        .copied()
        .collect();
    assert!(
        waiting.len() > WAITING_A_CALL,
        "{} tiles wait",
        waiting.len()
    );
    preview.make_waiting(store, &waiting).unwrap();
    assert_eq!(previews(store).len(), WAITING_A_CALL, "a few a call");
    let mut calls = 1;
    while waiting.iter().any(|key| store.preview_wanted(key)) {
        preview.make_waiting(store, &waiting).unwrap();
        calls += 1;
        assert!(calls <= waiting.len(), "each call makes some");
    }
    drop(project.close(false));
}

/// A dab that misses a quad's box counts in none of its cells: the box holds them all, rounding
/// included, however finely the quad is cut.
#[test]
fn a_quads_box_holds_every_cell_of_it() {
    let quads = [
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.5),
        ],
        [
            Vec3::new(1000.1, -700.3, 12.7),
            Vec3::new(1000.4, -700.2, 12.9),
            Vec3::new(1000.2, -699.9, 12.1),
            Vec3::new(1000.7, -699.8, 13.3),
        ],
    ];
    for corners in quads {
        for growth in [0.0_f32, 1e-3, 0.25] {
            let whole = quad_box(corners, growth).expect("a box");
            for per_side in [1_usize, 2, 7, 64] {
                let cells = (0..per_side).flat_map(|b| (0..per_side).map(move |a| (a, b)));
                for (a, b) in cells {
                    let cell = cell_box(corners, a, b, per_side, growth).expect("a box");
                    let (low, high) = (cell.min(), cell.max());
                    let (lowest, highest) = (whole.min(), whole.max());
                    assert!(
                        low.x >= lowest.x && low.y >= lowest.y && low.z >= lowest.z,
                        "cell ({a}, {b}) of {per_side} below its quad's box"
                    );
                    assert!(
                        high.x <= highest.x && high.y <= highest.y && high.z <= highest.z,
                        "cell ({a}, {b}) of {per_side} above its quad's box"
                    );
                }
            }
        }
    }
}

/// Looking into a quad by halves finds the cells a dab touches, no more and no fewer, however it
/// lies on the quad.
#[test]
fn the_cells_a_dab_touches_are_found_by_halves_as_one_by_one() {
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.1, 0.0),
        Vec3::new(0.1, 1.0, 0.2),
        Vec3::new(1.2, 1.1, 0.7),
    ];
    let places = [
        ([0.5, 0.5, 0.3], 0.02),
        ([0.13, 0.81, 0.2], 0.1),
        ([0.9, 0.2, 0.1], 0.31),
        ([3.0, 3.0, 3.0], 0.1),
    ];
    for (center, radius) in places {
        let dab = Imprint::new(center, [0.0, 0.0, 1.0], radius, 0.01, 0.0, [0.0; 4]);
        for per_side in [2_usize, 8, 64] {
            for growth in [0.0_f32, 0.01] {
                let one_by_one: Vec<usize> = (0..per_side * per_side)
                    .filter(|n| {
                        let cell = cell_box(corners, n % per_side, n / per_side, per_side, growth);
                        crate::preview::touches(cell.as_ref(), &dab)
                    })
                    .collect();
                let mut by_halves = Vec::new();
                cells_touched(corners, per_side, growth, &dab, &mut by_halves);
                by_halves.sort_unstable();
                assert_eq!(by_halves, one_by_one, "{center:?}, {per_side} a side");
            }
        }
    }
}

/// Dabs applied at once, after dabs shown have landed, move the previews the stroke has: they
/// still bound the finest result, and the next dab shown finds them made.
#[test]
fn dabs_applied_at_once_move_the_previews_the_stroke_has() {
    let dir = TestDir::new();
    let base = base(&dir);
    let dabs = path_dabs(&dir, &base);
    let (mut project, mut index, _) = open_copy(&dir, &base, "applied");
    let (shown_first, then_applied) = dabs.split_at(dabs.len() / 2);
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview
        .show(project.store(), &index, shown_first, &display_only)
        .unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, shown_first).unwrap();
    preview
        .land(project.store(), &index, shown_first.len(), &display_only)
        .unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, then_applied).unwrap();
    preview
        .note_applied(project.store(), &index, then_applied)
        .unwrap();
    let shown = previews(project.store());
    assert_bounds(project.store(), &shown, "after dabs applied at once");
    let last = shown_first.last().expect("a dab");
    let again = preview
        .show(
            project.store(),
            &index,
            std::slice::from_ref(last),
            &display_only,
        )
        .unwrap();
    assert_eq!(again.gathered, 0, "the previews are still at hand");
    drop(project.close(false));
}
