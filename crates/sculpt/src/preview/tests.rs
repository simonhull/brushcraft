//! Previews against the finest work.
//!
//! The dabs shown and landed in pieces give the files the dabs applied at once give, bit for
//! bit; nothing of a preview is saved; previews meet at their seams; the refresh waits for them;
//! and once every dab lands, no preview is left.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use geom::{FaceId, Level, TileKey, Vec3, cube_sphere};
use store::{Drained, EpochKind, Project, ProjectConfig, SpatialIndex, StoreConfig, TileStore};
use stroke::Dab;

use super::{Preview, PreviewStats};
use crate::apply::apply_dabs;
use crate::state::Stroke;
use crate::testing::{TestDir, dab, fixed, vertex};

/// The finest level: 128 quads a face, four tiles of 64.
fn level() -> Level {
    Level::new(7).unwrap()
}

/// The store's configuration: a display layer to the finest level, two workers.
fn config(dir: &TestDir, name: &str) -> ProjectConfig {
    ProjectConfig {
        store: StoreConfig::new(256 << 20, 1)
            .with_display(level())
            .with_workers(fixed(2)),
        recovery_dir: dir.path().join(format!("recovery-{name}")),
    }
}

/// Runs the store's jobs until none is left.
fn run_jobs(store: &mut TileStore) {
    for _ in 0..10_000 {
        store
            .run_jobs(Instant::now() + Duration::from_millis(50))
            .unwrap();
        if !store.has_jobs() {
            return;
        }
    }
    panic!("the jobs never finished");
}

/// A project over the cube sphere with its display layer built, saved as `base.bcrf` in `dir`
/// and closed: every test opens copies of it, so their files start the same.
fn base(dir: &TestDir) -> PathBuf {
    let mut project = Project::new(cube_sphere(1).unwrap(), config(dir, "base")).unwrap();
    project.store().queue_display_rebuild(level());
    run_jobs(project.store());
    let path = dir.path().join("base.bcrf");
    drop(project.save_as(&path).unwrap());
    project.close(false).unwrap();
    path
}

/// A copy of `base` named `name`, opened, with its spatial index at the finest level.
fn open_copy(dir: &TestDir, base: &Path, name: &str) -> (Project, SpatialIndex, PathBuf) {
    let path = dir.path().join(format!("{name}.bcrf"));
    std::fs::copy(base, &path).unwrap();
    let mut project = Project::open(&path, config(dir, name)).unwrap();
    let index = SpatialIndex::build(project.store(), level(), |_, _| {}).unwrap();
    (project, index, path)
}

/// A stroke of `count` dabs across face 0 and over its edges, on the surface as `store` holds
/// it.
///
/// Radius 0.15 (about 16 quads), strengths of both signs, every fifth forty times stronger
/// (tiles' scales grow), varied hardness, each dab's normal the sphere's outward one.
fn path(store: &mut TileStore, count: u32) -> Vec<Dab> {
    (0..count)
        .map(|n| {
            let (tx, ty) = (u16::from(n % 2 == 1), u16::from(n % 4 >= 2));
            let key = TileKey::displacement(FaceId::new(0), level(), tx, ty).unwrap();
            let (i, j) = ((n * 13) % 65, (n * 29) % 65);
            let centre = vertex(store, key, i, j);
            let sign = if n % 3 == 1 { -1.0 } else { 1.0 };
            let strength = if n % 5 == 4 { 0.04 } else { 0.002 } * sign;
            let hardness = [0.0, 0.3, 0.8][usize::try_from(n % 3).unwrap()];
            let length = (centre.x * centre.x + centre.y * centre.y + centre.z * centre.z).sqrt();
            Dab {
                normal: [centre.x / length, centre.y / length, centre.z / length],
                ..dab(centre, 0.15, strength, hardness, n)
            }
        })
        .collect()
}

/// The finest level's tiles of every face.
fn finest_tiles(store: &TileStore) -> Vec<TileKey> {
    crate::testing::tiles(store.mesh(), level())
}

/// Applies `dabs` at once, as finest work.
fn at_once(project: &mut Project, index: &mut SpatialIndex, dabs: &[Dab]) {
    let mut stroke = Stroke::new(project.store());
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), index, &mut stroke, dabs).unwrap();
    project.end_unrecorded_stroke().unwrap();
}

/// Shows every one of `dabs`, then applies them `run` at a time, landing each run: what a brush
/// whose frames fit none of its finest work does. Returns what each landing did.
fn previewed(
    project: &mut Project,
    index: &mut SpatialIndex,
    dabs: &[Dab],
    run: usize,
) -> Vec<PreviewStats> {
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    let shown = preview.show(project.store(), index, dabs).unwrap();
    assert_eq!(shown.dabs, dabs.len());
    assert!(shown.tiles_set > 0, "the dabs moved previews");
    let mut landings = Vec::new();
    for chunk in dabs.chunks(run) {
        apply_dabs(project.store(), index, &mut stroke, chunk).unwrap();
        landings.push(preview.land(project.store(), index, chunk.len()).unwrap());
    }
    assert_eq!(preview.shown(), 0, "every dab landed");
    project.end_unrecorded_stroke().unwrap();
    landings
}

#[test]
fn dabs_shown_then_landed_in_pieces_save_the_file_the_dabs_applied_at_once_save() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut finest, mut finest_index, finest_path) = open_copy(&dir, &base, "finest");
    let dabs = path(finest.store(), 14);
    at_once(&mut finest, &mut finest_index, &dabs);
    run_jobs(finest.store());
    drop(finest.save().unwrap());
    finest.close(false).unwrap();

    let (mut shown, mut shown_index, shown_path) = open_copy(&dir, &base, "shown");
    assert_eq!(
        path(shown.store(), 14),
        dabs,
        "the same dabs, from the same surface"
    );
    let landings = previewed(&mut shown, &mut shown_index, &dabs, 3);
    assert!(
        landings.iter().any(|landing| landing.tiles_set > 0),
        "landings made previews again over what landed"
    );
    assert!(!shown.store().has_previews(), "no preview left");
    run_jobs(shown.store());
    drop(shown.save().unwrap());
    shown.close(false).unwrap();

    let (a, b) = (
        std::fs::read(finest_path).unwrap(),
        std::fs::read(shown_path).unwrap(),
    );
    assert_eq!(a.len(), b.len(), "the same length");
    assert!(a == b, "F = ∞ and F = 0 save the same file, bit for bit");
}

#[test]
fn a_save_while_dabs_are_shown_holds_what_landed_and_nothing_of_the_previews() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut landed, mut landed_index, landed_path) = open_copy(&dir, &base, "landed");
    let dabs = path(landed.store(), 12);
    let (first, _) = dabs.split_at(5);
    let mut landed_stroke = Stroke::new(landed.store());
    landed.begin_stroke().unwrap();
    apply_dabs(landed.store(), &mut landed_index, &mut landed_stroke, first).unwrap();
    drop(landed.save().unwrap());
    landed.end_unrecorded_stroke().unwrap();
    landed.close(false).unwrap();

    let (mut shown, mut shown_index, shown_path) = open_copy(&dir, &base, "shown");
    let mut stroke = Stroke::new(shown.store());
    let mut preview = Preview::new();
    shown.begin_stroke().unwrap();
    preview
        .show(shown.store(), &mut shown_index, &dabs)
        .expect("shown");
    apply_dabs(shown.store(), &mut shown_index, &mut stroke, first).unwrap();
    preview
        .land(shown.store(), &mut shown_index, first.len())
        .expect("landed");
    assert_eq!(preview.shown(), 7, "seven dabs still shown");
    assert!(shown.store().has_previews(), "previews pending");
    drop(shown.save().unwrap());
    shown.end_unrecorded_stroke().unwrap();
    shown.close(false).unwrap();

    let (a, b) = (
        std::fs::read(landed_path).unwrap(),
        std::fs::read(shown_path).unwrap(),
    );
    assert!(
        a == b,
        "a save mid catch-up is the save of what landed: no preview data"
    );
}

#[test]
fn previews_meet_at_their_seams_and_move_points_by_the_draw_action_along_the_dab() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "seams");
    let dabs = path(project.store(), 6);
    let store = project.store();
    let before: BTreeMap<TileKey, Vec<Vec3>> = finest_tiles(store)
        .into_iter()
        .map(|key| (key, store.evaluate(key).unwrap().to_vec()))
        .collect();
    let mut preview = Preview::new();
    // Points by hand: the first dab's centre is face 0's first vertex; every vertex near it moves
    // by the draw action's amount along its own frame normal, as the finest work moves it, not
    // along the dab's normal.
    let first = dabs[0];
    preview.show(store, &mut index, &[first]).unwrap();
    let corner = TileKey::displacement(FaceId::new(0), level(), 0, 0).unwrap();
    let frames = store.frame_normals_many(&[corner]).unwrap().remove(0);
    let imprint = crate::apply::imprint_of(&first).unwrap();
    let side = corner.side();
    let mut off_the_dab = 0;
    for (i, j) in [(0, 0), (5, 3), (2, 9), (11, 1)] {
        let at = usize::try_from(j * side + i).unwrap();
        let stored = before[&corner][at];
        let w = brush::draw_imprint_vertex(
            &brush::Smoothstep,
            &imprint,
            [stored.x, stored.y, stored.z],
        )
        .unwrap();
        let want = stored + frames[at] * w;
        let got = store
            .preview(&corner)
            .expect("previewed")
            .get(i64::from(i), i64::from(j));
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "({i}, {j}) moved as finest work"
        );
        let [nx, ny, nz] = first.normal;
        if frames[at].to_bits() != Vec3::new(nx, ny, nz).to_bits() {
            off_the_dab += 1;
        }
    }
    assert!(off_the_dab > 0, "frame normals that aren't the dab's");
    let gone = preview.land(store, &mut index, 1).unwrap();
    assert!(gone.tiles_cleared > 0, "dropped unapplied: its previews go");
    assert!(!store.has_previews());
    preview.show(store, &mut index, &dabs).unwrap();
    let previewed: Vec<TileKey> = store.previewed().copied().collect();
    assert!(previewed.len() > 1, "{previewed:?}");
    // Every copy of a vertex, from every previewed tile and face, holds the same bits; and a
    // tile not previewed holds the store's.
    let topology = store.mesh().topology();
    let mut seen: BTreeMap<(u32, u16, u16, u32, u32), [u32; 3]> = BTreeMap::new();
    let (mut shared, mut moved) = (0, 0);
    for key in finest_tiles(store) {
        let region = geom::Region::of_tile(&key);
        let q = region.quads();
        for j in 0..=q {
            for i in 0..=q {
                let copies: Vec<geom::VertexCopy> = topology
                    .copies(key.face(), level(), region.x0() + i, region.y0() + j)
                    .into_iter()
                    .collect();
                let owner = geom::Topology::owner(&copies).unwrap();
                let owner = (owner.face.index(), owner.tx, owner.ty, owner.i, owner.j);
                let stored = before[&key][usize::try_from(j * (q + 1) + i).unwrap()];
                let drawn = store
                    .preview(&key)
                    .map_or(stored, |block| block.get(i64::from(i), i64::from(j)));
                if drawn.to_bits() != stored.to_bits() {
                    moved += 1;
                }
                let bits = drawn.to_bits();
                if let Some(&held) = seen.get(&owner) {
                    shared += 1;
                    assert_eq!(held, bits, "{key:?} ({i}, {j}): a seam open");
                } else {
                    seen.insert(owner, bits);
                }
            }
        }
    }
    assert!(shared > 0 && moved > 0, "{shared} shared, {moved} moved");
    drop(project.close(false));
}

#[test]
fn the_refresh_waits_for_previews_and_runs_once_every_dab_has_landed() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "refresh");
    let cursor = project.store().subscribe_stale();
    let dabs = path(project.store(), 8);
    let (first, rest) = dabs.split_at(4);
    let mut stroke = Stroke::new(project.store());
    let mut preview = Preview::new();
    project.begin_stroke().unwrap();
    preview.show(project.store(), &mut index, &dabs).unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, first).unwrap();
    preview
        .land(project.store(), &mut index, first.len())
        .unwrap();
    let store = project.store();
    let held: Vec<TileKey> = store.previewed().copied().collect();
    assert!(!held.is_empty(), "previews left");
    run_jobs(store);
    let above = |key: &TileKey| TileKey::display(key.face(), Level::new(6).unwrap(), 0, 0).unwrap();
    let unrefreshed = |of: &TileStore, key: TileKey| {
        let stale = of.unrefreshed_display();
        stale.iter().any(|tile| tile.tile == key)
    };
    assert!(
        unrefreshed(store, above(&held[0])),
        "a display tile over a preview waits for the dabs to land"
    );
    apply_dabs(store, &mut index, &mut stroke, rest).unwrap();
    preview.land(store, &mut index, rest.len()).unwrap();
    assert!(!store.has_previews());
    run_jobs(store);
    assert!(
        store.unrefreshed_display().is_empty(),
        "every tile refreshed once every dab landed"
    );
    // The feed carried preview epochs of finest tiles alone, and closed none empty.
    let Drained::Epochs(epochs) = store.drain_stale(&cursor) else {
        panic!("no resync");
    };
    let previews = epochs.iter().filter(|e| e.kind == EpochKind::Preview);
    let mut count = 0;
    for epoch in previews {
        count += 1;
        assert!(!epoch.keys.is_empty());
        assert!(epoch.keys.iter().all(|key| key.level() == level()));
    }
    assert!(
        count >= 3,
        "one a show and one a landing that changed previews"
    );
    store.unsubscribe_stale(cursor);
    project.end_unrecorded_stroke().unwrap();
    drop(project.close(false));
}

#[test]
fn a_preview_differs_from_what_lands_only_by_the_stored_encoding() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "encoding");
    let dabs = path(project.store(), 10);
    let store = project.store();
    let before: BTreeMap<TileKey, Vec<Vec3>> = finest_tiles(store)
        .into_iter()
        .map(|key| (key, store.evaluate(key).unwrap().to_vec()))
        .collect();
    let mut preview = Preview::new();
    preview.show(store, &mut index, &dabs).unwrap();
    let shown_now: BTreeMap<TileKey, Vec<Vec3>> = store
        .previewed()
        .copied()
        .collect::<Vec<TileKey>>()
        .into_iter()
        .map(|key| (key, store.preview_positions(&key).unwrap().to_vec()))
        .collect();
    let mut stroke = Stroke::new(store);
    project.begin_stroke().unwrap();
    apply_dabs(project.store(), &mut index, &mut stroke, &dabs).unwrap();
    preview
        .land(project.store(), &mut index, dabs.len())
        .unwrap();
    project.end_unrecorded_stroke().unwrap();
    let landed_store = project.store();
    let (mut moved, mut off) = (0.0_f32, 0.0_f32);
    for (key, drawn) in &shown_now {
        let landed = landed_store.evaluate(*key).unwrap();
        for ((p, l), b) in drawn.iter().zip(landed).zip(&before[key]) {
            moved = moved.max((*l - *b).length());
            off = off.max((*p - *l).length());
        }
    }
    assert!(moved > 0.01, "the dabs moved the surface: {moved}");
    assert!(
        off < moved * 1e-3,
        "shown {off} from what landed, of {moved} moved: the encoding's rounding alone"
    );
    drop(project.close(false));
}

#[test]
fn the_strokes_surface_is_hit_where_its_previews_show_it() {
    let dir = TestDir::new();
    let base = base(&dir);
    let (mut project, mut index, _) = open_copy(&dir, &base, "surface");
    // Strong dabs, every one pushing out, so the preview stands well off the store's surface.
    let dabs: Vec<Dab> = path(project.store(), 8)
        .into_iter()
        .map(|dab| Dab {
            strength: 0.05,
            ..dab
        })
        .collect();
    let store = project.store();
    let mut preview = Preview::new();
    preview.show(store, &mut index, &dabs).unwrap();
    assert!(store.has_previews());
    let mut rays = 0;
    let mut raised = 0;
    for dab in &dabs {
        let [x, y, z] = dab.position;
        let centre = Vec3::new(x, y, z);
        // From outside the sphere, straight down at the dab's centre and a little beside it.
        for offset in [Vec3::ZERO, Vec3::new(0.03, -0.02, 0.01)] {
            let target = centre + offset;
            let origin = target * 3.0;
            let ray = store::Ray::new(origin, target - origin).unwrap();
            let fast = index.raycast_surface(store, &ray).unwrap();
            let every = index.raycast_surface_exhaustive(store, &ray).unwrap();
            assert_eq!(fast, every, "the blocks skip exactly on previews too");
            let truth = index.raycast_exhaustive(store, &ray).unwrap();
            if let (Some(fast), Some(truth)) = (fast, truth) {
                rays += 1;
                if fast.t < truth.t {
                    raised += 1;
                }
            }
        }
    }
    assert!(
        rays > 0 && raised > 0,
        "{raised} of {rays} rays hit the raised preview first"
    );
    // The stroke engine's surface reads it too: a raycast at the first dab hits the preview.
    let [x, y, z] = dabs[0].position;
    let target = stroke::DVec3::new(f64::from(x), f64::from(y), f64::from(z));
    let ray = stroke::Ray::new(target * 3.0, target * -2.0).unwrap();
    let mut surface = crate::StoreSurface::new(store, &index);
    let hit = stroke::Surface::raycast(&mut surface, &ray).expect("a hit");
    surface.finish().unwrap();
    assert!(
        hit.position.length() > target.length() + 0.02,
        "placed on the preview, {} out, not on the store's surface at {}",
        hit.position.length(),
        target.length()
    );
    drop(project.close(false));
}
