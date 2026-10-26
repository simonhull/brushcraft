//! The display layer the store keeps: through save and open, under edits (against a rebuild of
//! every display tile, across faces), and damaged.

use geom::DisplayTile;

use super::*;
use crate::deps::DepGraph;
use crate::display::readers;

/// Every display tile key of the tests' mesh, in key order.
fn display_keys(mesh: &BaseMesh) -> Vec<TileKey> {
    let mut keys: Vec<TileKey> = tiles(mesh, 0, TOP - 1)
        .into_iter()
        .map(|k| TileKey::display(k.face(), k.level(), k.tx(), k.ty()).unwrap())
        .collect();
    keys.sort_unstable();
    keys
}

/// Whether two display tiles hold the same bytes and bound.
fn same(a: &DisplayTile, b: &DisplayTile) -> bool {
    a.encoded() == b.encoded() && a.error().to_bits() == b.error().to_bits()
}

#[test]
fn display_tiles_round_trip_through_save_and_open() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(&mesh, TOP, 40, 3));
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    drop(store);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    let keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    let stored = reopened.stored_display(&keys).unwrap();
    for (tile, back) in layer.iter().zip(stored) {
        let back = back.unwrap_or_else(|| panic!("{:?} kept", tile.key()));
        assert_eq!(&back, tile, "{:?} bit for bit", tile.key());
    }
}

#[test]
fn an_edit_makes_unknown_every_display_tile_it_changes_across_faces() {
    let mesh = mesh();
    let keys = display_keys(&mesh);
    for seed in 0..6_u64 {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
        let before = display_layer(&mut store);
        store.record_display(&before).unwrap();
        let edits = random_edits(&mesh, TOP, 1 + u32::try_from(seed % 3).unwrap(), 50 + seed);
        let mut map = CorrectionMap::new();
        apply(&mut store, &mut map, &edits);
        let stored = store.stored_display(&keys).unwrap();
        let after = display_layer(&mut store);
        let after: BTreeMap<TileKey, DisplayTile> =
            after.into_iter().map(|t| (*t.key(), t)).collect();
        let before: BTreeMap<TileKey, DisplayTile> =
            before.into_iter().map(|t| (*t.key(), t)).collect();
        let mut unknown = BTreeSet::new();
        let mut changed_faces = BTreeSet::new();
        for (key, kept) in keys.iter().zip(&stored) {
            let changed = !same(&before[key], &after[key]);
            if changed {
                changed_faces.insert(key.face());
            }
            match kept {
                None => {
                    unknown.insert(*key);
                }
                Some(kept) => {
                    assert!(
                        !changed,
                        "seed {seed}: {key:?} changed but the store still holds it"
                    );
                    assert!(same(kept, &before[key]), "seed {seed}: {key:?} as recorded");
                }
            }
        }
        // Every change made something unknown, and on more than one face where an edit sits on a
        // seam: the test reaches across faces.
        assert!(!unknown.is_empty(), "seed {seed}: something is unknown");
        // Every copy of each edited vertex is written, on every tile holding it.
        let edited: BTreeSet<TileKey> = edits
            .iter()
            .flat_map(|e| {
                mesh.topology()
                    .copies(e.face, e.level, e.i, e.j)
                    .iter()
                    .map(|copy| copy.key(e.level))
                    .collect::<Vec<_>>()
            })
            .collect();
        let levels: BTreeSet<Level> = (0..TOP).map(level).collect();
        let expected = readers(&mesh, &DepGraph::new(), &edited, &levels).unwrap();
        assert!(
            unknown.is_subset(&expected),
            "seed {seed}: only display readers of the edits are unknown"
        );
        assert!(
            changed_faces.len()
                <= unknown
                    .iter()
                    .map(TileKey::face)
                    .collect::<BTreeSet<_>>()
                    .len(),
            "seed {seed}: every face with a changed tile has an unknown one"
        );
    }
}

/// The finest-level corrections of `an_edit_makes_the_neighbours_display_tiles_unknown`'s
/// case `name`, on face 0: `(i, j, height)`.
///
/// The corner case is one vertex, a base vertex three faces share. The middle case is a broad
/// bump inside face 0's four middle finest tiles: no finest tile of a neighbour moves, and the
/// means it tilts survive quantization up to the coarsest levels, whose tents read across the
/// seams.
fn seam_edit(name: &str) -> Vec<(u32, u32, f32)> {
    if name == "corner" {
        return vec![(0, 0, 0.05)];
    }
    let middle = level(TOP).face_quads() / 2;
    let radius = 50_u16;
    let mut out = Vec::new();
    for j in middle - u32::from(radius)..=middle + u32::from(radius) {
        for i in middle - u32::from(radius)..=middle + u32::from(radius) {
            let d = |a: u32| f32::from(u16::try_from(a.abs_diff(middle)).unwrap());
            let r = (d(i) * d(i) + d(j) * d(j)).sqrt();
            out.push((i, j, 0.1 * (1.0 - r / f32::from(radius)).max(0.0)));
        }
    }
    out
}

/// An edit changes display tiles of neighbouring faces, and the store makes those unknown too.
///
/// Their normals' tent reads across the seam: at a corner (a base vertex three faces share),
/// and from the middle of a face, which reaches its neighbours' coarsest tiles only through the
/// mip (no finest tile of theirs moves).
#[test]
fn an_edit_makes_the_neighbours_display_tiles_unknown() {
    let mesh = mesh();
    let keys = display_keys(&mesh);
    for (name, faces) in [("corner", 3), ("middle", 2)] {
        let dir = TestDir::new();
        let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
        let before = display_layer(&mut store);
        store.record_display(&before).unwrap();
        store.begin_op().unwrap();
        for (i, j, height) in seam_edit(name) {
            store
                .set_correction(
                    FaceId::new(0),
                    level(TOP),
                    i,
                    j,
                    Vec3::new(0.0, 0.0, height),
                )
                .unwrap();
        }
        let stored = store.stored_display(&keys).unwrap();
        let after: BTreeMap<TileKey, DisplayTile> = display_layer(&mut store)
            .into_iter()
            .map(|t| (*t.key(), t))
            .collect();
        let mut changed_faces = BTreeSet::new();
        for ((key, kept), was) in keys.iter().zip(&stored).zip(&before) {
            if !same(was, &after[key]) {
                changed_faces.insert(key.face());
                assert!(kept.is_none(), "{name}: {key:?} changed and is unknown");
            }
        }
        assert!(
            changed_faces.len() >= faces,
            "{name}: tiles of {faces} faces changed: {changed_faces:?}"
        );
    }
}

#[test]
fn a_damaged_display_record_is_unknown_not_trusted() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let layer = display_layer(&mut store);
    store.record_display(&layer).unwrap();
    store.save().unwrap();
    let victim = *layer[3].key();
    let loc = store.working().locate(&victim).unwrap();
    drop(store);
    damage(&dir.path("work.bcp"), loc);
    let file = ProjectFile::open(&dir.path("work.bcp")).unwrap();
    let mut reopened = TileStore::open(file, roomy()).unwrap();
    let keys: Vec<TileKey> = layer.iter().map(|t| *t.key()).collect();
    for (key, back) in keys.iter().zip(reopened.stored_display(&keys).unwrap()) {
        if *key == victim {
            assert!(back.is_none(), "{key:?}: damaged, so unknown");
        } else {
            assert!(back.is_some(), "{key:?}: intact");
        }
    }
}
