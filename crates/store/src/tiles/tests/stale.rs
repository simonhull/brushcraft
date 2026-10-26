//! The stale set: the tiles a re-evaluation after an edit must redo, against a brute-force
//! comparison of every tile's positions and normals before and after the edit.

use super::*;

/// The positions and normals of each of `keys`, evaluated now.
fn snapshot(store: &mut TileStore, keys: &[TileKey]) -> Vec<(Vec<Vec3>, Vec<Vec3>)> {
    keys.iter()
        .map(|key| {
            let positions = store.evaluate(*key).unwrap().to_vec();
            let normals = store.normals(*key).unwrap().to_vec();
            (positions, normals)
        })
        .collect()
}

/// Whether two vertex lists differ in any bit.
fn differs(a: &[Vec3], b: &[Vec3]) -> bool {
    a.iter().zip(b).any(|(a, b)| a.to_bits() != b.to_bits())
}

/// Every tile whose positions or normals an edit changes is in [`TileStore::stale_after`] of
/// the tiles whose positions it changed, and the set comes back sorted, each key once.
///
/// The edits are at a tile's interior, at a border, at a corner (shared by three or four
/// tiles), and next to a border and a corner (read by neighbours' aprons and corner rings).
#[test]
fn stale_after_holds_every_tile_an_edit_changes() {
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, roomy());
    let keys = tiles(&mesh, TOP, TOP);
    let key = keys[5];
    let region = Region::of_tile(&key);
    let spots = [
        (region.x0() + 9, region.y0() + 11),
        (region.x0(), region.y0() + 11),
        (region.x0(), region.y0()),
        (region.x0() + 1, region.y0() + 11),
        (region.x0() + 1, region.y0() + 1),
    ];
    let mut normals_only = 0;
    for (n, (i, j)) in spots.into_iter().enumerate() {
        let before = snapshot(&mut store, &keys);
        store.begin_op().unwrap();
        store
            .set_correction(key.face(), key.level(), i, j, Vec3::new(0.0, 0.0, 0.05))
            .unwrap();
        let after = snapshot(&mut store, &keys);
        let pairs = || keys.iter().zip(before.iter().zip(&after));
        let moved: Vec<TileKey> = pairs()
            .filter(|(_, (b, a))| differs(&b.0, &a.0))
            .map(|(k, _)| *k)
            .collect();
        let changed: BTreeSet<TileKey> = pairs()
            .filter(|(_, (b, a))| differs(&b.0, &a.0) || differs(&b.1, &a.1))
            .map(|(k, _)| *k)
            .collect();
        assert!(!moved.is_empty(), "spot {n}: the edit moved something");
        let stale = store.stale_after(&moved).unwrap();
        assert!(
            stale.windows(2).all(|w| w[0] < w[1]),
            "spot {n}: sorted, each key once"
        );
        let missed: Vec<&TileKey> = changed.iter().filter(|k| !stale.contains(k)).collect();
        assert!(
            missed.is_empty(),
            "spot {n}: changed but not stale: {missed:?}"
        );
        normals_only += changed.len() - moved.len();
    }
    // A vertex next to a border is read by the neighbour's apron: its normals change though its
    // positions don't, so the set is more than the tiles that moved.
    assert!(normals_only > 0, "some neighbour's normals changed alone");
}
