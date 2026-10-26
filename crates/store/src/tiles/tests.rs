//! The tile store against `geom`'s dense evaluation, the oracle: every tile bit for bit, with
//! room to spare and under constant eviction, across edits, saves and pins.

#![expect(clippy::missing_panics_doc, reason = "tests panic to fail")]

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use geom::{
    BaseMesh, Channel, CorrectionError, CorrectionMap, CorrectionTile, DenseEvaluation, FaceId,
    LayerId, Level, PositionSource, Region, TileKey, Vec3, cube_sphere, normal_tiles, normals,
};

use super::config::BATCH_SHARE;
use super::edit::{BorderValues, border_order, update_row};
use super::hooks::EvalLog;
use super::layers::{first_op, saved_base};
use super::round::{
    Held, Positions, Resident, Tiles, held_index, nothing_to_write, recorded_or_listed,
};
use super::*;
use crate::MAX_SEQ;
use crate::deps::list_normal_inputs;
use crate::file::FileError;
use crate::fixture::{
    EVAL, Edit, Rng, SITE, TEST_THREADS, TOP, assert_positions, assert_tile, config, damage, dense,
    fixed, level, mesh, meta_for, pools, random_edits, roomy, split_prism, tight, tiles,
};
use crate::op::RESERVED_SITE;
use crate::pool::{Freshness, SlotData, SlotId};
use crate::test_dir::TestDir;
use crate::workers::Workers;

mod budget;
mod differential;
mod edit;
mod evaluate;
mod feed;
mod invalidate;
mod layered;
mod prefetch;
mod recovery;
mod rounds_kept;
mod workload;

/// Chunks in [`pinning`]'s budget.
const PINNING_CHUNKS: usize = 12;

/// [`tight`] with room for a few pinned tiles, or for the normals' inputs (pinned while a
/// normal tile is computed), besides an evaluation's own pins.
fn pinning() -> StoreConfig {
    config(PINNING_CHUNKS, 4)
}

/// A store over a new working file `name` in `dir`.
fn new_store(dir: &TestDir, name: &str, mesh: &Arc<BaseMesh>, config: StoreConfig) -> TileStore {
    let file = ProjectFile::create(&dir.path(name), meta_for(mesh), Arc::clone(mesh)).unwrap();
    TileStore::open(file, config).unwrap()
}

/// Applies `edits` to the store (each as a new op) and to the oracle's map.
pub(crate) fn apply(store: &mut TileStore, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        store.begin_op().unwrap();
        dab(store, map, e);
    }
}

/// Applies `edits` to the store as dabs of the current op, as one stroke makes them, and to the
/// oracle's map.
pub(crate) fn apply_dabs(store: &mut TileStore, map: &mut CorrectionMap, edits: &[Edit]) {
    for e in edits {
        dab(store, map, e);
    }
}

/// Applies `e` to the store as part of the current op, and to the oracle's map.
fn dab(store: &mut TileStore, map: &mut CorrectionMap, e: &Edit) {
    store
        .set_correction(e.face, e.level, e.i, e.j, e.c)
        .unwrap();
    map.set(store.mesh.topology(), e.face, e.level, e.i, e.j, e.c)
        .unwrap();
}

/// Frees `key`'s clean resident correction slot as an eviction does, op included, without
/// writing anything: its next read loads it from its record again.
pub(crate) fn evict(store: &mut TileStore, key: &TileKey) {
    assert!(
        store.corrections.remove(key).unwrap(),
        "{key:?} was resident"
    );
    store.known_ops.remove(key);
}

/// Breaks `key`'s resident correction slot: it no longer holds a valid tile, so it neither
/// encodes nor saves.
pub(crate) fn break_slot(store: &mut TileStore, key: &TileKey) {
    let id = store.corrections.find(key).unwrap();
    store.corrections.get_mut(id).unwrap().side = 0;
}

/// Asserts that the store's normals of every tile of `keys` are geom's normals of the dense
/// evaluation.
fn assert_normals(
    store: &mut TileStore,
    mesh: &BaseMesh,
    dense: &DenseEvaluation,
    keys: &[TileKey],
    what: &str,
) {
    for key in keys {
        let want = normals(mesh, dense, Region::of_tile(key)).unwrap();
        let got = store.normals(*key).unwrap();
        let region = Region::of_tile(key);
        assert_tile(
            got,
            |i, j| want.get(i - region.x0(), j - region.y0()).unwrap(),
            key,
            what,
        );
    }
}

/// The level-[`TOP`] tile holding face-grid vertex `(i, j)` of `face`.
fn tile_at(face: FaceId, i: u32, j: u32) -> TileKey {
    TileKey::containing(LayerId::BASE, Channel::DISPLACEMENT, face, level(TOP), i, j)
        .unwrap()
        .0
}

/// Of `keys`, how many have a slot in `pool`.
fn resident<T: SlotData>(pool: &Pool<T>, keys: &[TileKey]) -> u64 {
    keys.iter()
        .filter(|key| pool.find(key).is_some())
        .count()
        .try_into()
        .unwrap()
}

/// One tile and edits of it, in its own coordinates.
type TileBatch = (TileKey, Vec<(u32, u32, Vec3)>);

/// `count` tiles of levels 1–8 picked with `seed`, each with `per_tile` edits in its own
/// coordinates (a third of them on its border, where copies in other tiles follow).
fn tile_batches(mesh: &BaseMesh, count: u32, per_tile: u32, seed: u64) -> Vec<TileBatch> {
    let all = tiles(mesh, 1, TOP);
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            let key = all[rng.below(u32::try_from(all.len()).unwrap()) as usize];
            let side = key.side();
            let amplitude = 0.5 / f32::from(u16::try_from(key.level().face_quads()).unwrap());
            let edits = (0..per_tile)
                .map(|_| {
                    let (mut i, mut j) = (rng.below(side), rng.below(side));
                    if rng.below(3) == 0 {
                        if rng.below(2) == 0 {
                            i = [0, side - 1][rng.below(2) as usize];
                        } else {
                            j = [0, side - 1][rng.below(2) as usize];
                        }
                    }
                    (i, j, rng.vec3(amplitude))
                })
                .collect();
            (key, edits)
        })
        .collect()
}

/// Applies `batches` to the store with [`TileStore::edit_tile`], a fresh op each, and to the
/// oracle's map.
fn apply_batches(store: &mut TileStore, map: &mut CorrectionMap, batches: &[TileBatch]) {
    for (key, writes) in batches {
        let region = Region::of_tile(key);
        store.begin_op().unwrap();
        store
            .edit_tile(*key, |edit| {
                for &(i, j, c) in writes {
                    edit.set(i, j, c).unwrap();
                }
            })
            .unwrap();
        for &(i, j, c) in writes {
            let (fi, fj) = (region.x0() + i, region.y0() + j);
            map.set(store.mesh.topology(), key.face(), key.level(), fi, fj, c)
                .unwrap();
        }
    }
}

/// Which vertices of a tile a test changes, so that a tile's first write falls in each of the
/// places [`TileEdit::update`] and [`TileStore::update_tiles`] mark it written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Any vertex: the first write is on border row 0.
    Anywhere,
    /// Interior vertices only: the tile is marked by an interior write alone.
    InteriorOnly,
    /// Nothing before row 5; in row 5 the interior and the last vertex (a border vertex, whose
    /// copies lie in a neighbour); anything after. The first write is interior, and the border
    /// write right after it marks the neighbour: the order of the two marks is checked.
    InteriorThenBorder,
}

impl Pattern {
    /// The pattern for test case `n`.
    const fn of(n: u32) -> Self {
        match n % 3 {
            0 => Self::Anywhere,
            1 => Self::InteriorOnly,
            _ => Self::InteriorThenBorder,
        }
    }

    /// Whether vertex `(i, j)` of a tile with `side` vertices a side may change.
    const fn allows(self, side: u32, i: u32, j: u32) -> bool {
        let last = side - 1;
        let interior = i > 0 && i < last && j > 0 && j < last;
        match self {
            Self::Anywhere => true,
            Self::InteriorOnly => interior,
            Self::InteriorThenBorder => j > 5 || (j == 5 && (interior || i == last)),
        }
    }
}

/// A change for [`updating_tiles_equals_updating_each_in_turn`], a pure function of its
/// arguments.
///
/// The round, the tile's place in the list and the vertex pick it, the tiles all of the finest
/// level. About half the vertices a [`Pattern`] allows move; now and then one by a lot (so the
/// scale grows part-way through a tile). In the `poisoned` round, tile 3 gets a value that
/// isn't finite part-way through row 5.
fn tiles_change(round: u32, poisoned: bool, tile: usize, i: u32, j: u32, c: Vec3) -> Option<Vec3> {
    let tile = u32::try_from(tile).unwrap();
    if poisoned && tile == 3 && (i, j) == (9, 5) {
        return Some(Vec3::new(0.0, f32::INFINITY, 0.0));
    }
    // Poisoned, the tiles up to the failing one write no border, so no copy of theirs reaches
    // a later tile: a later tile is marked only if the failure path marks it.
    let pattern = if poisoned && tile <= 3 {
        Pattern::InteriorOnly
    } else {
        Pattern::of(tile + round)
    };
    if !pattern.allows(level(TOP).tile_quads() + 1, i, j) {
        return None;
    }
    let seed = u64::from(round) << 40 ^ u64::from(tile) << 24 ^ u64::from(j) << 12 ^ u64::from(i);
    let mut rng = Rng(seed);
    let delta = match rng.below(40) {
        0..20 => return None,
        20 => rng.vec3(0.05),
        _ => rng.vec3(0.002),
    };
    Some(Vec3::new(c.x + delta.x, c.y + delta.y, c.z + delta.z))
}

/// Runs round `round` of [`updating_tiles_equals_updating_each_in_turn`] on `store`, as a new op.
///
/// The store named "None" edits one tile at a time (the definition), the others with
/// [`TileStore::update_tiles`]. Returns the outcome, printed.
fn update_round(
    name: &str,
    store: &mut TileStore,
    keys: &[TileKey],
    round: u32,
    poisoned: bool,
) -> String {
    store.begin_op().unwrap();
    let change = |n: usize, i: u32, j: u32, c: Vec3| tiles_change(round, poisoned, n, i, j, c);
    let outcome = if name == "None" {
        (0..).zip(keys).try_for_each(|(n, &key)| {
            store
                .edit_tile(key, |edit| edit.update(|i, j, c| change(n, i, j, c)))
                .and_then(std::convert::identity)
        })
    } else {
        store.update_tiles(keys, change)
    };
    format!("{outcome:?}")
}

/// Polls until every prefetch request is answered; returns how many tiles were installed.
///
/// Fails the test if that takes more than ten seconds: the reads are a few kilobytes each.
fn drain_prefetch(store: &mut TileStore) -> usize {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut installed = store.poll().unwrap();
    while store.prefetching() > 0 {
        assert!(Instant::now() < deadline, "prefetch didn't finish");
        thread::sleep(Duration::from_millis(1));
        installed += store.poll().unwrap();
    }
    installed
}

/// The bytes of the tile `key`'s resident correction slot holds (see
/// [`CorrectionSlot::tile_bytes`]).
fn slot_resident(store: &TileStore, key: &TileKey) -> Vec<u8> {
    let id = store.corrections.find(key).unwrap();
    store.corrections.get(id).unwrap().tile_bytes()
}

/// A store over a new working file `name` holding the correction tiles of `count` random edits
/// (seeded with `seed`), saved, with nothing resident; the oracle's map of the same edits.
fn store_on_disk(
    dir: &TestDir,
    name: &str,
    mesh: &Arc<BaseMesh>,
    config: StoreConfig,
    count: u32,
    seed: u64,
) -> (TileStore, CorrectionMap) {
    let mut store = new_store(dir, name, mesh, config);
    let mut map = CorrectionMap::new();
    apply(&mut store, &mut map, &random_edits(mesh, TOP, count, seed));
    store.save().unwrap();
    store.shrink_to(0).unwrap();
    assert_eq!(store.bytes(), 0, "nothing resident");
    (store, map)
}

/// Dirties every resident correction tile of `store` with a new op, at an interior vertex of
/// each.
fn dirty_every_resident_correction(store: &mut TileStore, all: &[TileKey]) {
    store.begin_op().unwrap();
    let resident: Vec<TileKey> = all
        .iter()
        .filter(|key| store.corrections.find(key).is_some())
        .copied()
        .collect();
    for key in resident {
        store
            .edit_tile(key, |edit| edit.set(1, 1, Vec3::new(0.001, 0.0, 0.0)))
            .unwrap()
            .unwrap();
    }
}
