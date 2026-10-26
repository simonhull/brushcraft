//! The index loaded from the bounds a project file keeps: the index built from its tiles, bit for
//! bit, after saves, edits, reopens and compaction, evaluating only what isn't stored.

use std::collections::BTreeSet;

use geom::{CorrectionMap, Level};

use super::*;
use crate::bounds::readers;
use crate::file::compact;
use crate::fixture::{TOP, fixed, mesh, random_edits, roomy, tiles};
use crate::tiles::tests::apply;

/// The level indexed: the tests' finest, 96 tiles of the cube.
fn top() -> Level {
    Level::new(TOP).unwrap()
}

/// A store over the project file at `path`, made if `create`, else opened, with `threads`
/// evaluation workers.
fn open(path: &std::path::Path, create: bool, threads: usize) -> TileStore {
    let mesh = mesh();
    let file = if create {
        ProjectFile::create(path, meta_for(&mesh), Arc::clone(&mesh)).unwrap()
    } else {
        ProjectFile::open(path).unwrap()
    };
    TileStore::open(file, roomy().with_workers(fixed(threads))).unwrap()
}

#[test]
fn a_saved_index_loads_bit_for_bit_without_evaluating() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mesh = mesh();
    let built = {
        let mut store = open(&path, true, 2);
        apply(
            &mut store,
            &mut CorrectionMap::new(),
            &random_edits(&mesh, TOP, 120, 3),
        );
        let index = SpatialIndex::build(&mut store, top(), |_, _| {}).unwrap();
        store.save().unwrap();
        index_bits(&index)
    };
    for threads in [1, 2, 5] {
        let mut store = open(&path, false, threads);
        let (loaded, stats) = SpatialIndex::load(&mut store, top(), |_, _| {}).unwrap();
        assert_eq!(
            stats,
            LoadStats {
                stored: built.len(),
                evaluated: 0
            },
            "{threads} threads: every bound stored"
        );
        assert!(index_bits(&loaded) == built, "{threads} threads: as built");
        assert_eq!(
            assert_loads_as_built(&mut store, top(), "reopened").evaluated,
            0,
            "{threads} threads"
        );
    }
}

#[test]
fn edits_saved_and_reopened_still_load_as_built() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mesh = mesh();
    let all = tiles(&mesh, TOP, TOP);
    {
        let mut store = open(&path, true, 2);
        SpatialIndex::build(&mut store, top(), |_, _| {}).unwrap();
        store.save().unwrap();
    }
    // Finest edits, refitted: every bound is recorded again, nothing to evaluate.
    {
        let mut store = open(&path, false, 2);
        let (mut index, _) = SpatialIndex::load(&mut store, top(), |_, _| {}).unwrap();
        apply(
            &mut store,
            &mut CorrectionMap::new(),
            &random_edits(&mesh, TOP, 40, 11)
                .into_iter()
                .filter(|e| e.level == top())
                .collect::<Vec<_>>(),
        );
        index.update(&mut store, &all).unwrap();
        store.save().unwrap();
    }
    {
        let mut store = open(&path, false, 2);
        let stats = assert_loads_as_built(&mut store, top(), "refitted");
        assert_eq!(stats.evaluated, 0, "refitted: nothing evaluated");
    }
    // Coarser edits, saved without a refit: what they change is evaluated, the rest read.
    // Coarse enough to reach past a tile's neighbours, fine enough not to reach every tile.
    let edits: Vec<_> = random_edits(&mesh, TOP - 1, 40, 12)
        .into_iter()
        .filter(|e| e.level == Level::new(TOP - 1).unwrap())
        .take(2)
        .collect();
    let edited: BTreeSet<TileKey> = edits.iter().flat_map(|e| e.keys(&mesh)).collect();
    let stale = readers(
        &mesh,
        &crate::deps::DepGraph::new(),
        &edited,
        &BTreeSet::from([top()]),
    )
    .unwrap();
    {
        let mut store = open(&path, false, 2);
        apply(&mut store, &mut CorrectionMap::new(), &edits);
        store.save().unwrap();
    }
    let mut store = open(&path, false, 2);
    let stats = assert_loads_as_built(&mut store, top(), "edited, not refitted");
    assert_eq!(
        stats.evaluated,
        stale.len(),
        "the bounds the edits changed are evaluated, and only they"
    );
    assert!(
        0 < stats.evaluated && stats.evaluated < all.len(),
        "{stats:?}"
    );
}

#[test]
fn a_file_without_bounds_evaluates_every_tile_once() {
    let dir = TestDir::new();
    let path = dir.path("p.bcrf");
    let mesh = mesh();
    {
        let mut store = open(&path, true, 2);
        apply(
            &mut store,
            &mut CorrectionMap::new(),
            &random_edits(&mesh, TOP, 60, 5),
        );
        store.save().unwrap();
    }
    let all = tiles(&mesh, TOP, TOP).len();
    {
        let mut store = open(&path, false, 2);
        let stats = assert_loads_as_built(&mut store, top(), "never indexed");
        assert_eq!(stats.evaluated, all, "nothing stored: every tile evaluated");
        store.save().unwrap();
    }
    let mut store = open(&path, false, 2);
    let stats = assert_loads_as_built(&mut store, top(), "saved after the load");
    assert_eq!(stats.evaluated, 0, "the load's bounds were saved");
    let coarser = assert_loads_as_built(
        &mut store,
        Level::new(TOP - 1).unwrap(),
        "a level never indexed",
    );
    assert_eq!(coarser.stored, 0, "nothing stored at another level");
}

#[test]
fn compaction_keeps_the_bounds() {
    let dir = TestDir::new();
    let (path, compacted) = (dir.path("p.bcrf"), dir.path("c.bcrf"));
    let mesh = mesh();
    {
        let mut store = open(&path, true, 2);
        let mut index = SpatialIndex::build(&mut store, top(), |_, _| {}).unwrap();
        store.save().unwrap();
        // A second save over the first, so compaction drops records.
        apply(
            &mut store,
            &mut CorrectionMap::new(),
            &random_edits(&mesh, TOP, 30, 8),
        );
        index.update(&mut store, &tiles(&mesh, TOP, TOP)).unwrap();
        store.save().unwrap();
    }
    compact(&path, &compacted).unwrap();
    assert!(
        std::fs::metadata(&compacted).unwrap().len() < std::fs::metadata(&path).unwrap().len(),
        "compaction drops the old records"
    );
    let mut store = open(&compacted, false, 2);
    let stats = assert_loads_as_built(&mut store, top(), "compacted");
    assert_eq!(stats.evaluated, 0, "every bound carried across");
}
