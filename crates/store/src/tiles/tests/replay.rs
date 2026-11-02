//! What a brush replaying edits away from the store reads and writes: the kept predictions it
//! asks for, the corrections as stored, and whole tiles written back.

use super::*;

/// A store of `config` over `mesh` with random edits applied, one op each.
fn edited(dir: &TestDir, mesh: &Arc<BaseMesh>, config: StoreConfig) -> TileStore {
    let mut store = new_store(dir, "replay.bcrf", mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(mesh, TOP, 200, 7));
    store
}

/// Twelve finest tiles of `mesh`.
fn finest(mesh: &BaseMesh) -> Vec<TileKey> {
    tiles(mesh, TOP, TOP).into_iter().take(12).collect()
}

#[test]
fn a_kept_prediction_gives_the_evaluated_bits() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = edited(&dir, &mesh, roomy());
    let keys = finest(&mesh);
    let guard = store.pin(&keys).unwrap();
    assert!(
        store.keep_predictions_of(&keys).unwrap(),
        "the store keeps a dozen"
    );
    for key in &keys {
        let positions = store.evaluate(*key).unwrap().to_vec();
        let corrections = store.cached_correction(key).unwrap();
        let prediction = store.kept_prediction(key).unwrap();
        let applied = prediction.apply(|i, j| corrections.get(i, j).unwrap());
        let bits = |v: &[Vec3]| v.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(applied.as_slice()),
            bits(&positions),
            "{key:?}: the prediction with the stored corrections"
        );
    }
    drop(guard);
}

#[test]
fn listed_frames_are_the_kept_predictions_frames_bit_for_bit() {
    let dir = TestDir::new();
    let mesh = mesh();
    let keys = finest(&mesh);
    // Listed by a store that keeps no prediction: every one predicted afresh.
    let mut listing = edited(&dir, &mesh, roomy().with_kept_predictions(0));
    let listed = listing.frames_many(&keys).unwrap();
    // The same edits, and the predictions kept.
    let other = TestDir::new();
    let mut keeping = edited(&other, &mesh, roomy());
    let guard = keeping.pin(&keys).unwrap();
    assert!(keeping.keep_predictions_of(&keys).unwrap(), "kept");
    let bits = |f: &geom::Frame| [f.t().to_bits(), f.b().to_bits(), f.n().to_bits()];
    for (key, frames) in keys.iter().zip(&listed) {
        let prediction = keeping.kept_prediction(key).unwrap();
        let side = key.side();
        assert_eq!(
            frames.len(),
            usize::try_from(side * side).unwrap(),
            "{key:?}: every vertex"
        );
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        for ((i, j), frame) in grid.zip(frames) {
            let kept = prediction.frame(i, j).unwrap();
            assert_eq!(bits(frame), bits(&kept), "{key:?} ({i}, {j})");
        }
    }
    // And a store reading its own kept predictions lists the same.
    let again = keeping.frames_many(&keys).unwrap();
    assert!(
        again
            .iter()
            .flatten()
            .zip(listed.iter().flatten())
            .all(|(a, b)| bits(a) == bits(b)),
        "the kept predictions' frames, listed"
    );
    drop(guard);
}

#[test]
fn predictions_are_kept_only_for_pinned_current_tiles_within_the_bound() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = edited(&dir, &mesh, roomy().with_kept_predictions(4));
    let keys = finest(&mesh);
    assert_eq!(store.kept_predictions(), 4, "the bound");
    let guard = store.pin(&keys).unwrap();
    assert!(
        !store.keep_predictions_of(&keys).unwrap(),
        "more tiles than the store keeps"
    );
    let few = &keys[..3];
    assert!(store.keep_predictions_of(few).unwrap(), "three of them");
    drop(guard);
    let unpinned = &keys[5..7];
    assert!(
        !store.keep_predictions_of(unpinned).unwrap(),
        "tiles not pinned"
    );
    assert!(
        unpinned
            .iter()
            .all(|key| store.kept_prediction(key).is_none()),
        "none kept for them"
    );
}

/// Every resident correction slot of `keys`, byte for byte, with whether it is unsaved.
fn slots(store: &TileStore, keys: &[TileKey]) -> Vec<(TileKey, Vec<u8>, bool)> {
    keys.iter()
        .map(|key| {
            let bytes = store.corrections.find(key).map_or_else(Vec::new, |id| {
                store.corrections.get(id).unwrap().tile_bytes()
            });
            (*key, bytes, store.unsaved.contains_key(key))
        })
        .collect()
}

#[test]
fn tiles_replaced_whole_are_what_edits_of_them_make() {
    let mesh = mesh();
    let keys = finest(&mesh);
    let key = *keys.first().unwrap();
    // An interior vertex grown past the tile's scale, and a border vertex with its copies.
    let interior = (5, 7, Vec3::new(0.5, -0.25, 3.0));
    let border = (0, 9, Vec3::new(0.125, 0.0, -0.75));

    let dir = TestDir::new();
    let mut edits = edited(&dir, &mesh, roomy());
    edits.begin_op().unwrap();
    edits
        .edit_tile(key, |edit| {
            edit.set(interior.0, interior.1, interior.2).unwrap();
            edit.set(border.0, border.1, border.2).unwrap();
        })
        .unwrap();

    let other = TestDir::new();
    let mut replaced = edited(&other, &mesh, roomy());
    replaced.begin_op().unwrap();
    let region = Region::of_tile(&key);
    let copies = mesh.topology().copies(
        key.face(),
        key.level(),
        region.x0() + border.0,
        region.y0() + border.1,
    );
    let mut tiles: BTreeMap<TileKey, CorrectionTile> = BTreeMap::new();
    for copy in &copies {
        let at = copy.key(key.level());
        let mut tile = replaced.correction(at).unwrap().to_tile();
        tile.set(copy.i, copy.j, border.2).unwrap();
        tiles.insert(at, tile);
    }
    tiles
        .get_mut(&key)
        .unwrap()
        .set(interior.0, interior.1, interior.2)
        .unwrap();
    let whole: Vec<(TileKey, &CorrectionTile)> =
        tiles.iter().map(|(at, tile)| (*at, tile)).collect();
    replaced.replace_corrections(&whole).unwrap();

    let all: Vec<TileKey> = tiles.keys().copied().collect();
    assert!(all.len() > 1, "the border vertex has copies in other tiles");
    assert_eq!(
        slots(&replaced, &all),
        slots(&edits, &all),
        "the same slots, dirty alike"
    );
    for at in &all {
        assert_eq!(
            replaced.evaluate(*at).unwrap(),
            edits.evaluate(*at).unwrap(),
            "{at:?}: the same positions"
        );
    }
}

#[test]
fn a_tile_of_another_side_is_refused_before_anything_is_written() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = edited(&dir, &mesh, roomy());
    let keys = finest(&mesh);
    let before = slots(&store, &keys);
    let small = CorrectionTile::zero(3).unwrap();
    let refused = store.replace_corrections(&[(*keys.first().unwrap(), &small)]);
    assert!(
        matches!(
            refused,
            Err(CacheError::Correction(CorrectionError::BadSide(3)))
        ),
        "{refused:?}"
    );
    assert_eq!(slots(&store, &keys), before, "nothing written");
}

#[test]
fn an_empty_batch_does_nothing_and_begins_no_op() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "empty.bcrf", &mesh, roomy());
    assert_eq!(store.current_op(), None, "no op yet");
    store.replace_corrections(&[]).unwrap();
    assert_eq!(store.current_op(), None, "still none");
}

#[test]
fn a_tile_given_twice_is_refused_before_anything_is_written() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = edited(&dir, &mesh, roomy());
    let keys = finest(&mesh);
    let key = *keys.first().unwrap();
    let before = slots(&store, &keys);
    let mut tile = store.correction(key).unwrap().to_tile();
    tile.set(5, 5, Vec3::new(0.0, 0.0, 1.0)).unwrap();
    let refused = store.replace_corrections(&[(key, &tile), (key, &tile)]);
    assert!(
        matches!(refused, Err(CacheError::NamedTwice(twice)) if twice == key),
        "{refused:?}"
    );
    assert_eq!(slots(&store, &keys), before, "nothing written");
}

#[test]
fn a_border_vertex_changed_without_its_twins_is_refused_before_anything_is_written() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = edited(&dir, &mesh, roomy());
    let key = *finest(&mesh).first().unwrap();
    let (i, j) = (0, 9);
    let region = Region::of_tile(&key);
    let copies = mesh
        .topology()
        .copies(key.face(), key.level(), region.x0() + i, region.y0() + j);
    let all: Vec<TileKey> = copies.iter().map(|copy| copy.key(key.level())).collect();
    assert!(all.len() > 1, "the vertex has copies in other tiles");
    let before = slots(&store, &all);
    let value = Vec3::new(0.0, 0.0, 0.5);
    let mut tiles: BTreeMap<TileKey, CorrectionTile> = BTreeMap::new();
    for copy in &copies {
        let at = copy.key(key.level());
        let mut tile = store.correction(at).unwrap().to_tile();
        tile.set(copy.i, copy.j, value).unwrap();
        tiles.insert(at, tile);
    }
    // The first tile alone: its twins are missing.
    let alone = [(key, &tiles[&key])];
    let refused = store.replace_corrections(&alone);
    assert!(
        matches!(refused, Err(CacheError::TwinsDisagree { tile, .. }) if tile == key),
        "a copy missing: {refused:?}"
    );
    // Every copy, one with other bits.
    let other = *all.iter().find(|&&at| at != key).unwrap();
    let copy = copies
        .iter()
        .find(|copy| copy.key(key.level()) == other)
        .unwrap();
    tiles
        .get_mut(&other)
        .unwrap()
        .set(copy.i, copy.j, value * 2.0)
        .unwrap();
    let differing: Vec<(TileKey, &CorrectionTile)> =
        tiles.iter().map(|(at, tile)| (*at, tile)).collect();
    let refused_too = store.replace_corrections(&differing);
    assert!(
        matches!(refused_too, Err(CacheError::TwinsDisagree { .. })),
        "a copy with other bits: {refused_too:?}"
    );
    assert_eq!(slots(&store, &all), before, "nothing written");
}

/// Each of `keys`' corrections, in bits, row-major.
fn corrections_bits(store: &mut TileStore, keys: &[TileKey]) -> Vec<Vec<[u32; 3]>> {
    keys.iter()
        .map(|key| {
            let tile = store.correction(*key).unwrap();
            let side = key.side();
            (0..side)
                .flat_map(|j| (0..side).map(move |i| (i, j)))
                .map(|(i, j)| tile.get(i, j).unwrap().to_bits())
                .collect()
        })
        .collect()
}

/// A brush-like move of vertex `(i, j)` of tile `n` written in its `frame`: a world move that
/// differs from vertex to vertex, its components along the frame's axes added to `c`.
fn framed_move(n: usize, i: u32, j: u32, c: Vec3, frame: &geom::Frame) -> Option<Vec3> {
    if (i + 2 * j + u32::try_from(n).unwrap()).is_multiple_of(3) {
        return None;
    }
    #[expect(clippy::cast_precision_loss, reason = "small grid places")]
    let world = Vec3::new(1e-3, -2e-3 + 1e-5 * i as f32, 5e-4 * j as f32 * 1e-2);
    Some(
        c + Vec3::new(
            frame.t().dot(world),
            frame.b().dot(world),
            frame.n().dot(world),
        ),
    )
}

#[test]
fn framed_updates_write_what_update_tiles_writes_given_the_frames() {
    let dir = TestDir::new();
    let mesh = mesh();
    let keys = finest(&mesh);
    // Along the kept predictions' frames, as the store reads them.
    let mut along_store = edited(&dir, &mesh, roomy());
    let guard = along_store.pin(&keys).unwrap();
    assert!(along_store.keep_predictions_of(&keys).unwrap(), "kept");
    along_store.begin_op().unwrap();
    assert!(
        along_store.update_tiles_framed(&keys, framed_move).unwrap(),
        "every tile pinned and kept"
    );
    let along = corrections_bits(&mut along_store, &keys);
    drop(guard);
    // With the frames listed and handed to the plain update.
    let other = TestDir::new();
    let mut plain = edited(&other, &mesh, roomy().with_kept_predictions(0));
    let listed_frames = plain.frames_many(&keys).unwrap();
    let plain_guard = plain.pin(&keys).unwrap();
    assert!(
        !plain.update_tiles_framed(&keys, framed_move).unwrap(),
        "no kept predictions: nothing done"
    );
    plain.begin_op().unwrap();
    plain
        .update_tiles(&keys, |n, i, j, c| {
            let side = keys[n].side();
            let frame = listed_frames[n][usize::try_from(j * side + i).unwrap()];
            framed_move(n, i, j, c, &frame)
        })
        .unwrap();
    let listed = corrections_bits(&mut plain, &keys);
    drop(plain_guard);
    assert!(along == listed, "the same bits, tile by tile");
}

#[test]
fn a_worker_predicts_the_bits_the_store_keeps() {
    let dir = TestDir::new();
    let mesh = mesh();
    let keys = finest(&mesh);
    let mut store = edited(&dir, &mesh, roomy());
    store.evaluate_many(&keys, Evaluation::Positions).unwrap();
    let made: Vec<(TileKey, Prediction)> = {
        let predictor = store.predictor();
        keys.iter()
            .filter_map(|key| Some((*key, predictor.predict(key, &store.parent_slots(key)?)?)))
            .collect()
    };
    assert_eq!(made.len(), keys.len(), "every tile's parents current");
    let guard = store.pin(&keys).unwrap();
    assert!(store.keep_predictions_of(&keys).unwrap(), "kept");
    // In bits: a prediction's ring holds NaN slots, which no comparison of values equals.
    let bits = |p: &Prediction, side: u32| {
        let frames: Vec<[[u32; 3]; 3]> = p
            .frames()
            .iter()
            .map(|f| [f.t().to_bits(), f.b().to_bits(), f.n().to_bits()])
            .collect();
        let grid = (0..side).flat_map(|j| (0..side).map(move |i| (i, j)));
        let predicted: Vec<[u32; 3]> = grid
            .map(|(i, j)| p.predicted(i, j).unwrap().to_bits())
            .collect();
        (frames, predicted)
    };
    for (key, prediction) in &made {
        let kept = store.kept_prediction(key).unwrap();
        assert!(
            bits(kept, key.side()) == bits(prediction, key.side()),
            "{key:?}: the kept prediction's bits"
        );
    }
    drop(guard);
    // Kept through `keep_prediction` instead, by a store that kept none.
    let other = TestDir::new();
    let mut fresh = edited(&other, &mesh, roomy());
    fresh.evaluate_many(&keys, Evaluation::Positions).unwrap();
    let fresh_guard = fresh.pin(&keys).unwrap();
    let unpinned = tiles(&mesh, TOP, TOP)
        .into_iter()
        .find(|key| !keys.contains(key))
        .unwrap();
    for (key, prediction) in made {
        if fresh.kept_prediction(&key).is_none() {
            assert!(fresh.keep_prediction(key, prediction), "{key:?} kept");
        }
    }
    assert!(keys.iter().all(|key| fresh.kept_prediction(key).is_some()));
    {
        fresh
            .evaluate_many(&[unpinned], Evaluation::Positions)
            .unwrap();
        let predicted = {
            let predictor = fresh.predictor();
            fresh
                .parent_slots(&unpinned)
                .and_then(|parents| predictor.predict(&unpinned, &parents))
        };
        let prediction = predicted.expect("its parents current");
        assert!(
            !fresh.keep_prediction(unpinned, prediction),
            "a tile not pinned keeps none"
        );
    }
    drop(fresh_guard);
}
