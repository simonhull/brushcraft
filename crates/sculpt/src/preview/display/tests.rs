//! Display previews against the finest work.
//!
//! The bound holds against the finest result, and doesn't grow with a stroke's length; border
//! vertices are the finest previews' bits (so seams close at every level); the landed finest work
//! replaces them exactly; the saved file is that of the dabs applied at once; and the normals
//! follow the moved surface.

use std::collections::BTreeMap;

use geom::{Channel, FaceId, Level, TileKey, Vec3};
use store::{DisplayPreview, TileStore};
use stroke::Dab;

use super::finest_place;
use crate::apply::apply_dabs;
use crate::preview::tests::{at_once, base, level, open_copy, path, previewed_drawing, run_jobs};
use crate::preview::{DrawsAll, Preview};
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
    let dabs = path(project.store(), 14);
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
    assert!(shown_ratio > 0.01, "a bound near the error: {shown_ratio}");
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
                    } else if landing.is_some()
                        && store
                            .display_changes(key)
                            .is_some_and(|(_, f)| f.contains(&under))
                    {
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
