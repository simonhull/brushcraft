//! A tile with its one-vertex ring, read from the store.
//!
//! It is `geom`'s gather of the dense evaluation bit for bit, the same bits on both sides of
//! every seam and round every base vertex, and the normals computed from it are the store's own.

use geom::fixtures::ring_normals;
use geom::{GatheredBlock, gather_block};

use super::*;

/// A store over `mesh` with `config`, edited with random corrections at every level up to
/// `top`; and the dense evaluation of the same corrections.
fn edited(
    dir: &TestDir,
    mesh: &Arc<BaseMesh>,
    config: StoreConfig,
    top: u8,
) -> (TileStore, DenseEvaluation) {
    let mut store = new_store(dir, "work.bcp", mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(mesh, top, 60, 0x816));
    let dense = DenseEvaluation::new(mesh, level(top), &map).unwrap();
    (store, dense)
}

/// The bits of `values`.
fn bits(values: &[Vec3]) -> Vec<[u32; 3]> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A ring's bits: its centre, edge points, diagonals and owner corner.
type RingBits = ([u32; 3], Vec<[u32; 3]>, Vec<[u32; 3]>, u8);

/// The bits of `ring`.
fn ring_bits(ring: &geom::GatheredRing) -> RingBits {
    (
        ring.center().to_bits(),
        bits(ring.edges()),
        bits(ring.diagonals()),
        ring.owner_corner(),
    )
}

/// Asserts that every ring vertex of every block in `blocks` is the vertex its own tile holds,
/// and that every block holding a base vertex at a corner holds the same ring for it.
fn assert_seams_agree(
    mesh: &BaseMesh,
    blocks: &BTreeMap<TileKey, GatheredBlock>,
    top: Level,
    what: &str,
) {
    let topology = mesh.topology();
    let n = top.face_quads();
    let mut across = 0;
    let mut rings: BTreeMap<(FaceId, u16, u16, u32, u32), Vec<&geom::GatheredRing>> =
        BTreeMap::new();
    for (key, block) in blocks {
        let region = block.region();
        let q = i64::from(region.quads());
        let (x0, y0) = (i64::from(region.x0()), i64::from(region.y0()));
        for j in -1..=q + 1 {
            for i in -1..=q + 1 {
                let here = block.get(i, j);
                let Some(copies) = u32::try_from(x0 + i)
                    .ok()
                    .zip(u32::try_from(y0 + j).ok())
                    .filter(|&(fi, fj)| fi <= n && fj <= n)
                else {
                    // Off the face: the vertex across the edge, as some other block holds it.
                    continue;
                };
                // A vertex of the face: every tile holding a copy holds the same bits.
                for copy in topology.copies(region.face(), top, copies.0, copies.1) {
                    let theirs = TileKey::displacement(copy.face, top, copy.tx, copy.ty).unwrap();
                    let (li, lj) = (i64::from(copy.i), i64::from(copy.j));
                    let there = blocks[&theirs].get(li, lj);
                    assert_eq!(
                        here.to_bits(),
                        there.to_bits(),
                        "{what}: {key:?} ({i}, {j}) is {theirs:?}'s ({li}, {lj})"
                    );
                    if theirs != *key {
                        across += 1;
                    }
                }
            }
        }
        for (c, (ci, cj)) in [(0, 0), (q, 0), (q, q), (0, q)].into_iter().enumerate() {
            let Some(ring) = block.corner(c) else {
                continue;
            };
            let (fi, fj) = (
                u32::try_from(x0 + ci).unwrap(),
                u32::try_from(y0 + cj).unwrap(),
            );
            let owner = topology
                .copies(region.face(), top, fi, fj)
                .iter()
                .map(|copy| (copy.face, copy.tx, copy.ty, copy.i, copy.j))
                .min()
                .unwrap();
            rings.entry(owner).or_default().push(ring);
        }
    }
    assert!(across > 0, "{what}: the rings cross seams");
    assert!(!rings.is_empty(), "{what}: base vertices");
    for (vertex, copies) in &rings {
        assert_eq!(
            copies.len(),
            copies[0].valence(),
            "{what} {vertex:?}: a tile corner on every face around it"
        );
        for ring in copies {
            assert_eq!(
                ring_bits(ring),
                ring_bits(copies[0]),
                "{what} {vertex:?}: every face holds the same ring"
            );
        }
    }
}

#[test]
fn a_tile_with_its_ring_is_the_gather_of_the_dense_evaluation_and_agrees_across_seams() {
    let cube = mesh();
    let prism = Arc::new(split_prism(5));
    for (name, mesh, top) in [("cube", &cube, TOP), ("valence-5 prism", &prism, 6)] {
        let dir = TestDir::new();
        let (mut store, dense) = edited(&dir, mesh, roomy(), top);
        let keys = tiles(mesh, top, top);
        let blocks: BTreeMap<TileKey, GatheredBlock> = keys
            .iter()
            .map(|key| (*key, store.positions_with_ring(*key).unwrap()))
            .collect();
        for (key, block) in &blocks {
            let want = gather_block(mesh, &dense, Region::of_tile(key)).unwrap();
            assert_eq!(
                bits(block.padded()),
                bits(want.padded()),
                "{name} {key:?}: the positions and ring are the dense evaluation's"
            );
            for c in 0..4 {
                assert_eq!(
                    block.corner(c).map(ring_bits),
                    want.corner(c).map(ring_bits),
                    "{name} {key:?}: corner {c}'s ring"
                );
            }
            let side = key.side();
            let own = store.cached_positions(key).unwrap();
            for j in 0..side {
                let row = &block.row(i64::from(j))[1..=side as usize];
                let start = (j * side) as usize;
                assert_eq!(
                    bits(row),
                    bits(&own[start..start + side as usize]),
                    "{name} {key:?} row {j}: the tile's own positions"
                );
            }
        }
        assert_seams_agree(mesh, &blocks, level(top), name);
    }
}

#[test]
fn normals_computed_from_the_ring_are_the_stores_normals_bit_for_bit() {
    let cube = mesh();
    let prism = Arc::new(split_prism(5));
    for (name, mesh, top) in [("cube", &cube, TOP), ("valence-5 prism", &prism, 6)] {
        let dir = TestDir::new();
        let (mut store, _) = edited(&dir, mesh, roomy(), top);
        for key in tiles(mesh, 1, top) {
            let block = store.positions_with_ring(key).unwrap();
            let from_ring = ring_normals(mesh, &block);
            let geom_normals = block.normals(mesh);
            let stored = store.normals(key).unwrap();
            assert_eq!(
                bits(&from_ring),
                bits(stored),
                "{name} {key:?}: the plain per-vertex normals of the ring are the store's"
            );
            assert_eq!(
                bits(geom_normals.as_slice()),
                bits(stored),
                "{name} {key:?}: and geom's of the block are too"
            );
        }
    }
}

#[test]
fn a_tile_with_its_ring_reads_the_same_under_a_tight_budget() {
    let mesh = mesh();
    let roomy_dir = TestDir::new();
    let (mut roomy, _) = edited(&roomy_dir, &mesh, roomy(), TOP);
    let tight_dir = TestDir::new();
    let (mut tight, _) = edited(&tight_dir, &mesh, tight(), TOP);
    for key in tiles(&mesh, TOP, TOP) {
        let want = roomy.positions_with_ring(key).unwrap();
        let got = tight.positions_with_ring(key).unwrap();
        assert_eq!(
            bits(got.padded()),
            bits(want.padded()),
            "{key:?}: evicting as it goes, the same bits"
        );
    }
}

#[test]
fn a_cached_read_of_a_tile_with_its_ring_never_evaluates() {
    let dir = TestDir::new();
    let mesh = mesh();
    let (mut store, _) = edited(&dir, &mesh, roomy(), TOP);
    store.evict_all().unwrap();
    let key = TileKey::displacement(FaceId::new(2), level(TOP), 1, 3).unwrap();
    let before = store.hooks.log.positions.len();
    assert!(
        store.cached_positions_with_ring(&key).unwrap().is_none(),
        "nothing cached: nothing to read"
    );
    store.evaluate(key).unwrap();
    assert!(
        store.cached_positions_with_ring(&key).unwrap().is_none(),
        "the tile alone isn't its ring"
    );
    let evaluated = store.hooks.log.positions.len();
    assert!(evaluated > before, "evaluating the tile evaluated");
    let block = store.positions_with_ring(key).unwrap();
    let after = store.hooks.log.positions.len();
    assert!(after > evaluated, "the ring's tiles were evaluated");
    let cached = store.cached_positions_with_ring(&key).unwrap().unwrap();
    assert_eq!(
        store.hooks.log.positions.len(),
        after,
        "a cached read evaluates nothing"
    );
    assert_eq!(bits(cached.padded()), bits(block.padded()), "the same read");
    let past = TileKey::displacement(FaceId::new(9), level(TOP), 0, 0).unwrap();
    assert!(
        matches!(
            store.cached_positions_with_ring(&past),
            Err(CacheError::NotATile(_))
        ),
        "a face the mesh lacks"
    );
}
