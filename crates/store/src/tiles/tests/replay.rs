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
