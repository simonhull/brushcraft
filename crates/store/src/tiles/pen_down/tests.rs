//! Pen-down shapes: what a shape keeps and how close it decodes, what the store reads at pen-down
//! while a stroke changes its tiles, the budget, and raycasts of the pen-down surface.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use geom::{BaseMesh, Level, TileKey, Vec3};

use super::*;
use crate::fixture::{TOP, level, mesh, meta_for, roomy};
use crate::index::SpatialIndex;
use crate::pool::Budget;
use crate::test_dir::TestDir;
use crate::{ProjectFile, Ray};

/// A store over `mesh` in `dir`, with roomy pools.
fn store_over(dir: &TestDir, mesh: &Arc<BaseMesh>) -> TileStore {
    let file = ProjectFile::create(&dir.path("pen_down.bcp"), meta_for(mesh), Arc::clone(mesh))
        .expect("a file");
    TileStore::open(file, roomy()).expect("a store")
}

/// `points` in bits, to compare exactly.
fn bits(points: &[Vec3]) -> Vec<[u32; 3]> {
    points.iter().map(|p| p.to_bits()).collect()
}

#[test]
fn a_shape_gives_back_the_positions_it_was_kept_with_bit_for_bit() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    let (key, _, _) = tile_at(100, 100);
    store.begin_op().expect("an op");
    push_out(&mut store, (100, 100), 9, 0.02);
    let positions = store.evaluate(key).expect("evaluates").to_vec();
    let shape = PenDownShape::new(
        store.correction(key).expect("corrections").to_tile(),
        &positions,
    );
    let prediction = store.prediction_of_one(key).expect("a prediction");
    assert_eq!(bits(&shape.positions(&prediction)), bits(&positions));
    let finite = positions.iter().copied().filter(|p| p.is_finite());
    assert_eq!(shape.bounds(), Aabb::from_points(finite));
    assert_eq!(
        shape.bytes(),
        size_of::<PenDownShape>() + 63 * 63 * 6 + 256 * 12,
        "three i16 an interior vertex, three f32 a border one"
    );
    assert!(shape.bytes() < 27_100, "{} bytes", shape.bytes());
}

/// The tile of `TOP` holding face 0's grid point `(i, j)`, and that vertex's place in it.
fn tile_at(i: u32, j: u32) -> (TileKey, u32, u32) {
    TileKey::containing(
        LayerId::BASE,
        Channel::DISPLACEMENT,
        FaceId::new(0),
        level(TOP),
        i,
        j,
    )
    .expect("a tile")
}

/// Pushes every vertex of face 0 within `reach` grid steps of `(ci, cj)` at `TOP` out by
/// `height` along its frame's normal, as a stroke's edit would.
fn push_out(store: &mut TileStore, (ci, cj): (u32, u32), reach: u32, height: f32) {
    let face = FaceId::new(0);
    let top = level(TOP);
    for j in cj.saturating_sub(reach)..=cj + reach {
        for i in ci.saturating_sub(reach)..=ci + reach {
            store
                .set_correction(face, top, i, j, Vec3::new(0.0, 0.0, height))
                .expect("a correction");
        }
    }
}

/// The tiles a push of `reach` around `centre` writes, in key order: every tile holding a vertex
/// it sets.
fn tiles_pushed((ci, cj): (u32, u32), reach: u32) -> Vec<TileKey> {
    let mut keys = BTreeSet::new();
    for j in cj.saturating_sub(reach)..=cj + reach {
        for i in ci.saturating_sub(reach)..=ci + reach {
            keys.insert(tile_at(i, j).0);
            // A vertex on a tile seam is held by the tiles on its other sides too.
            keys.insert(tile_at(i.saturating_sub(1), j.saturating_sub(1)).0);
            keys.insert(tile_at(i.saturating_sub(1), j).0);
            keys.insert(tile_at(i, j.saturating_sub(1)).0);
        }
    }
    keys.into_iter().collect()
}

#[test]
fn a_stroke_reads_each_tile_it_changed_as_it_was_at_pen_down() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    store.begin_op().expect("an op");
    // Up to the column before a tile seam: the tile beyond it isn't written, but its normals on
    // the seam read the pushed column.
    let centre = (124, 100);
    let reached = tiles_pushed(centre, 3);
    let (key, _, _) = tile_at(centre.0, centre.1);
    let far = tile_at(10, 200).0;
    let before = store.evaluate(key).expect("evaluates").to_vec();
    let normals_before = store.normals(key).expect("normals").to_vec();
    let far_normals = store.normals(far).expect("normals").to_vec();
    let beside_before = |of: &mut TileStore, around: TileKey| {
        let inputs: Vec<TileKey> = of.normal_inputs(&around).expect("listed").to_vec();
        inputs
            .into_iter()
            .map(|tile| (tile, of.normals(tile).expect("normals").to_vec()))
            .collect::<BTreeMap<TileKey, Vec<Vec3>>>()
    };
    let around = beside_before(&mut store, key);

    store.keep_pen_down(&reached).expect("kept");
    push_out(&mut store, centre, 3, 0.05);
    let now = store.evaluate(key).expect("evaluates").to_vec();
    assert_ne!(now, before, "the push moved the tile");

    let shape = store.pen_down_shape(&key).expect("a shape").clone();
    let pen_down = store.pen_down_positions(key).expect("reads").to_vec();
    assert_eq!(
        bits(&pen_down),
        bits(&before),
        "the positions at pen-down, exactly"
    );
    let normals = store.pen_down_normals(key).expect("normals").to_vec();
    assert_eq!(bits(&normals), bits(&normals_before), "and the normals");
    // A neighbour the push didn't write, whose ring reads a pushed tile: its normals at pen-down
    // read that tile's pen-down shape.
    let beside = tile_at(130, 100).0;
    assert!(!reached.contains(&beside), "the push doesn't write it");
    let unchanged = store.pen_down_normals(far).expect("normals").to_vec();
    assert_eq!(
        bits(&unchanged),
        bits(&far_normals),
        "a tile whose normals read no changed tile: the store's own"
    );
    let beside_now = store.pen_down_normals(beside).expect("normals").to_vec();
    assert_eq!(
        bits(&beside_now),
        bits(&around[&beside]),
        "{beside:?}: its pen-down normals are its normals before the push"
    );
    assert_ne!(
        bits(store.normals(beside).expect("normals")),
        bits(&around[&beside]),
        "{beside:?}: the push turned its normals now"
    );

    // Gathered together, the same blocks as one by one.
    let together = store
        .pen_down_blocks_many(&[key, beside, far])
        .expect("gathered");
    for (tile, block) in [key, beside, far].into_iter().zip(&together) {
        let alone = store.pen_down_block(tile).expect("gathered");
        assert_eq!(
            bits(alone.padded()),
            bits(block.padded()),
            "{tile:?}: gathered together as alone"
        );
    }

    // Coming back to a tile keeps its first shape.
    store.keep_pen_down(&reached).expect("kept");
    push_out(&mut store, centre, 3, 0.1);
    assert_eq!(
        store.pen_down_shape(&key),
        Some(&shape),
        "the first shape stays"
    );
    assert_eq!(store.pen_down_tiles(), reached.len());
}

#[test]
fn the_shapes_are_withheld_from_the_pools_and_given_back_when_the_stroke_ends() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    let opened = store.budget.limit();
    store.begin_op().expect("an op");
    let reached = tiles_pushed((100, 100), 6);
    store.keep_pen_down(&reached).expect("kept");
    assert_eq!(
        store.budget.limit(),
        opened,
        "withheld as a frame pins, not before"
    );
    drop(store.pin(&reached).expect("pinned"));
    let bytes = store.pen_down_bytes();
    assert!(
        bytes >= reached.len() * 26_886,
        "{bytes} bytes for {} tiles",
        reached.len()
    );
    assert_eq!(
        store.budget.limit(),
        opened - bytes,
        "withheld from the pools"
    );
    assert_eq!(
        store.pen_down_beyond_budget(),
        0,
        "the roomy pools hold them"
    );
    store.release_pen_down();
    assert_eq!((store.pen_down_tiles(), store.pen_down_bytes()), (0, 0));
    assert_eq!(store.budget.limit(), opened, "given back");

    store.keep_pen_down(&reached).expect("kept");
    store.begin_op().expect("the next stroke");
    assert_eq!(store.pen_down_tiles(), 0, "a new op lets them go");
    assert_eq!(store.budget.limit(), opened);
}

#[test]
fn a_budget_withholds_down_to_the_floor_and_reports_the_rest() {
    let mut budget = Budget::new(100);
    assert_eq!(budget.withhold(30, 50), 0);
    assert_eq!(budget.limit(), 70);
    assert_eq!(budget.withhold(80, 50), 30, "30 bytes beyond the budget");
    assert_eq!(budget.limit(), 50, "never below the floor");
    assert_eq!(budget.withhold(0, 50), 0);
    assert_eq!(budget.limit(), 100, "all given back");
    assert_eq!(
        budget.withhold(10, 200),
        10,
        "a floor above the limit leaves no room"
    );
    assert_eq!(budget.limit(), 100);
}

#[test]
fn a_raycast_of_the_pen_down_surface_hits_a_changed_tile_where_it_was() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    let top = Level::new(TOP).expect("a level");
    let mut index = SpatialIndex::build(&mut store, top, |_, _| {}).expect("an index");
    let centre = (128, 128);
    let (key, li, lj) = tile_at(centre.0, centre.1);
    let at = usize::try_from(lj * key.side() + li).expect("an index");
    let target = *store
        .evaluate(key)
        .expect("evaluates")
        .get(at)
        .expect("a vertex");
    let origin = target * 3.0;
    let ray = Ray::new(origin, target - origin).expect("a ray");
    let hit_before = index
        .raycast_triangle(&mut store, &ray)
        .expect("casts")
        .expect("hits");

    store.begin_op().expect("an op");
    let reached = tiles_pushed(centre, 20);
    store.keep_pen_down(&reached).expect("kept");
    push_out(&mut store, centre, 20, -0.3);
    let changed = store.stale_after(&reached).expect("listed");
    index.update(&mut store, &changed).expect("refitted");

    let now = index
        .raycast_triangle(&mut store, &ray)
        .expect("casts")
        .expect("hits");
    assert!(now.t > hit_before.t * 1.05, "the push sank the surface");
    let pen_down = index
        .raycast_pen_down(&mut store, &ray)
        .expect("casts")
        .expect("hits the pen-down surface");
    assert_eq!(pen_down.key, hit_before.key);
    assert!(
        (pen_down.t - hit_before.t).abs() <= hit_before.t * 1e-4,
        "{} where it was, {}",
        pen_down.t,
        hit_before.t
    );
    let every = index
        .raycast_pen_down_exhaustive(&mut store, &ray)
        .expect("casts");
    assert_eq!(every, Some(pen_down), "the oracle agrees");

    // Many rays round the push, against the oracle.
    for k in 0..64_u16 {
        let angle = f32::from(k) * 0.4;
        let aim = target + Vec3::new(angle.cos(), angle.sin(), 0.5 * angle.cos()) * 0.05;
        let around = Ray::new(origin, aim - origin).expect("a ray");
        let fast = index.raycast_pen_down(&mut store, &around).expect("casts");
        let slow = index
            .raycast_pen_down_exhaustive(&mut store, &around)
            .expect("casts");
        assert_eq!(fast, slow, "ray {k}");
    }
}

#[test]
fn the_positions_read_last_are_kept_made_counted_and_the_same_bits_afresh() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    let opened = store.budget.limit();
    let top = Level::new(TOP).expect("a level");
    let per_side = u16::try_from(top.tiles_per_side()).expect("a few");
    let keys: Vec<TileKey> = (0..6)
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    TileKey::displacement(FaceId::new(face), top, tx, ty).expect("a tile")
                })
            })
        })
        .collect();
    assert!(keys.len() > MADE_TILES, "more tiles than are kept made");
    store.begin_op().expect("an op");
    store.keep_pen_down(&keys).expect("kept");
    let shapes = store.pen_down_bytes();
    let evaluated: Vec<Vec<[u32; 3]>> = keys
        .iter()
        .map(|key| bits(store.evaluate(*key).expect("evaluates")))
        .collect();
    for (key, was) in keys.iter().zip(&evaluated) {
        let read = bits(store.pen_down_positions(*key).expect("reads"));
        assert_eq!(&read, was, "{key:?}: the evaluation's bits");
    }
    let first = keys.first().expect("a tile");
    let last = keys.last().expect("a tile");
    assert!(
        store.made_pen_down(first).is_none(),
        "the least recently read let go"
    );
    assert!(store.pen_down.made.contains_key(last), "the last read kept");
    assert_eq!(store.pen_down.made.len(), MADE_TILES);
    assert_eq!(
        store.pen_down_bytes(),
        shapes + MADE_TILES * MADE_BYTES,
        "counted with the shapes"
    );
    drop(store.pin(&keys[..1]).expect("pinned"));
    assert_eq!(store.budget.limit(), opened - store.pen_down_bytes());
    let again = bits(store.pen_down_positions(*first).expect("reads"));
    assert_eq!(
        Some(&again),
        evaluated.first(),
        "made afresh, the same bits"
    );
    let (blocks, _) = store
        .pen_down_blocks(*first)
        .expect("reads")
        .map(|(positions, blocks)| (blocks.bytes(), positions.len()))
        .expect("a changed tile");
    assert!(
        blocks + 65 * 65 * 12 + ENTRY_BYTES <= MADE_BYTES,
        "{blocks} bytes of blocks"
    );
    store.release_pen_down();
    assert_eq!(store.budget.limit(), opened, "given back with the shapes");
}

/// Every tile of `TOP`, face by face.
fn all_tiles() -> Vec<TileKey> {
    let top = Level::new(TOP).expect("a level");
    let per_side = u16::try_from(top.tiles_per_side()).expect("a few");
    (0..6)
        .flat_map(|face| {
            (0..per_side).flat_map(move |ty| {
                (0..per_side).map(move |tx| {
                    TileKey::displacement(FaceId::new(face), top, tx, ty).expect("a tile")
                })
            })
        })
        .collect()
}

#[test]
fn pen_down_normals_are_made_once_and_kept_with_the_stroke() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    store.begin_op().expect("an op");
    let centre = (124, 100);
    let reached = tiles_pushed(centre, 3);
    store.keep_pen_down(&reached).expect("kept");
    push_out(&mut store, centre, 3, 0.05);
    let (key, _, _) = tile_at(centre.0, centre.1);
    let (before, made) = (store.pen_down_bytes(), store.pen_down.made.len());
    let first = bits(store.pen_down_normals(key).expect("normals"));
    assert!(store.pen_down.normals.contains_key(&key), "kept");
    let made = store.pen_down.made.len() - made;
    assert_eq!(
        store.pen_down_bytes(),
        before + GRID_BYTES + made * MADE_BYTES,
        "and counted, with the positions its block made"
    );
    let block = store.pen_down_block(key).expect("gathered");
    assert_eq!(
        first,
        bits(block.normals(&mesh).as_slice()),
        "the pen-down block's"
    );
    assert_eq!(bits(store.pen_down_normals(key).expect("normals")), first);
    // A tile whose normals read no changed tile is the store's own, not kept.
    let far = tile_at(10, 200).0;
    store.pen_down_normals(far).expect("normals");
    assert!(!store.pen_down.normals.contains_key(&far));
    store.release_pen_down();
    assert!(store.pen_down.normals.is_empty());
}

#[test]
fn a_brush_keeps_its_tiles_positions_made_apart_from_placements_however_many() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    let keys = all_tiles();
    assert!(keys.len() > MADE_TILES);
    store.begin_op().expect("an op");
    store.keep_pen_down(&keys).expect("kept");
    store.keep_predictions_of(&[]).expect("nothing to keep");
    // Placement reads a few around the pen.
    let near: Vec<TileKey> = keys.iter().take(4).copied().collect();
    for key in &near {
        store.pen_down_positions(*key).expect("reads");
    }
    let predicted: Vec<TileKey> = keys
        .iter()
        .filter(|key| store.kept_prediction(key).is_some())
        .copied()
        .collect();
    store.make_pen_down_many(&keys);
    for key in &predicted {
        let made = store.made_pen_down(key).expect("made for the brush");
        assert_eq!(bits(made), bits(store.evaluate(*key).expect("evaluates")));
    }
    for key in &near {
        assert!(
            store.pen_down.made.contains_key(key),
            "placement's stay made"
        );
    }
    let brush = store.pen_down.brush.len();
    assert!(
        brush + near.len() >= predicted.len(),
        "every predicted tile made"
    );
    // The next dab's tiles: those it doesn't reach go.
    let half: Vec<TileKey> = keys.iter().take(keys.len() / 2).copied().collect();
    store.make_pen_down_many(&half);
    assert!(store.pen_down.brush.keys().all(|key| half.contains(key)));
    assert_eq!(
        store.pen_down_bytes(),
        store.pen_down.bytes
            + store.pen_down.made.len() * MADE_BYTES
            + store.pen_down.brush.len() * GRID_BYTES,
        "counted"
    );
}

#[test]
fn a_previews_pen_down_blocks_are_counted_and_go_with_the_stroke() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = store_over(&dir, &mesh);
    store.begin_op().expect("an op");
    let (key, _, _) = tile_at(100, 100);
    let block = store.positions_with_ring(key).expect("gathered");
    let before = store.pen_down_bytes();
    store.keep_preview_pen_down(key, block.clone());
    let held = store.pen_down_bytes() - before;
    assert!(held >= 67 * 67 * 12, "{held} bytes for a 67² block");
    assert_eq!(store.take_preview_pen_down(&key), Some(block.clone()));
    assert_eq!(store.pen_down_bytes(), before, "taken out, uncounted");
    store.keep_preview_pen_down(key, block);
    store.release_pen_down();
    assert_eq!(
        store.take_preview_pen_down(&key),
        None,
        "gone with the stroke"
    );
}

#[test]
fn a_stroke_s_pen_down_data_never_makes_a_frame_s_pins_fail() {
    // A budget whose pools just hold a frame's pins: the pen-down data is withheld beside them,
    // the rest reported beyond the budget, and the pins taken.
    let keys = all_tiles();
    let probe_dir = TestDir::new();
    let mesh = mesh();
    let probe = store_over(&probe_dir, &mesh);
    let needed = probe.pin_bytes(&keys.iter().copied().collect());
    drop(probe);
    let dir = TestDir::new();
    let config = crate::fixture::with_reserved_shares(
        crate::StoreConfig::new(needed, crate::fixture::SITE)
            .with_chunk_bytes(roomy().chunk_bytes)
            .with_workers(crate::fixture::TEST_THREADS),
    );
    let file = ProjectFile::create(&dir.path("tight.bcp"), meta_for(&mesh), Arc::clone(&mesh))
        .expect("a file");
    let mut store = TileStore::open(file, config).expect("a store");
    drop(store.pin(&keys).expect("the pins fit"));
    store.begin_op().expect("an op");
    store.keep_pen_down(&keys).expect("kept");
    for key in keys.iter().take(MADE_TILES) {
        store.pen_down_positions(*key).expect("reads");
    }
    let pins = store
        .pin(&keys)
        .expect("the pins fit beside the pen-down data");
    assert!(
        store.pen_down_beyond_budget() > 0,
        "the rest held beyond the budget"
    );
    drop(pins);
    store.release_pen_down();
    assert_eq!(store.pen_down_beyond_budget(), 0);
}
