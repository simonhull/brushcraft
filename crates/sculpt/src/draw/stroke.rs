//! A stroke's state: each touched vertex's start and summed deltas, kept in `f32` so a dab's
//! small delta isn't rounded away by the tile's quantization (see [`Stroke`]).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

use geom::{TILE_QUADS, TileKey};
use store::{OpId, StoreId, TileStore};

/// Vertices of the largest tile, the entries of each [`TileStroke`]: `(TILE_QUADS + 1)²`.
const ENTRIES: usize = (TILE_QUADS as usize + 1) * (TILE_QUADS as usize + 1);

/// The bits of a start not captured yet: a NaN, which a stored correction never is.
const NO_START: u32 = f32::NAN.to_bits();

/// The state one brush stroke keeps across its dabs, so that its small dabs move a tile's
/// interior as far as its border.
///
/// A tile stores its interior corrections quantized to a step of its scale, and its border
/// vertices exactly. Were each dab to add its delta to the stored value, a delta under half a
/// step would vanish on the interior and be kept on the border, building ridges along tile
/// seams. Instead, for every vertex a stroke writes, it keeps the correction's `z` as it first
/// found it (the start) and the sum of its deltas, both in `f32`, and writes `start + sum`: the
/// stored value is rounded when it is written, not added to dab by dab, and interior and border
/// move alike.
///
/// # Error bound
///
/// An interior vertex's stored value differs from its value before the stroke plus the stroke's
/// deltas by at most half a step of its tile's scale for each time its start is captured: once,
/// unless its tile's state is released and the vertex edited again (see "Releasing tiles"),
/// when the start is captured again from the rounded stored value. Should the tile's scale grow
/// during the stroke, the re-quantization rounds the values written before it again, and the
/// bound is one step of the final scale for each capture instead.
///
/// # Memory
///
/// [`Stroke::new`] allocates the state of [`TileStore::stroke_tiles`] tiles up front, two `f32`
/// per vertex of the largest tile ([`store::StoreConfig::STROKE_TILE_BYTES`], 33,800 bytes), and
/// nothing more while it is used over that store: the store has set that much aside from its
/// budget ([`store::StoreConfig::stroke_tiles`]). A store reserves for one stroke state, so keep
/// one `Stroke` per store for the session, and pass it to every [`super::draw_step`]. It starts
/// afresh by itself when the store's op changes (a new stroke, or a save), and when it is used
/// over another store ([`TileStore::id`]), whose reservation it then takes the size of.
///
/// # Releasing tiles
///
/// A tile keeps its state until the stroke ends, unless a tile the current dab edits needs
/// state and none is free. Then the tile with the oldest last dab (then the least key) of those
/// not edited in the last [`Stroke::PROTECTED_DABS`] dabs, the current one included, is
/// released, and its buffer reused. The order is deterministic, so the result doesn't depend on
/// the worker count.
///
/// # The thrash limit
///
/// When the last [`Stroke::PROTECTED_DABS`] dabs together edit more than the store's
/// [`TileStore::stroke_tiles`] tiles (with the default, 512: dabs that each edit over about 128
/// tiles), a tile the current dab edits may find no state free and none to release. Tiles get
/// state in key order; the rest are written per dab, as without a stroke state (the
/// start-of-dab value plus the dab's delta), and counted in
/// [`super::StepStats::tiles_unaccumulated`]. Nothing fails.
#[derive(Debug)]
pub struct Stroke {
    /// The store the state is kept over. A dab over another releases every tile first.
    store: StoreId,
    /// The op of the stroke the state belongs to: the store's op when the last dab ended.
    /// A dab that finds the store at another op releases every tile first.
    op: Option<OpId>,
    /// The dabs counted so far, the current one included: each tile's last dab is one of them.
    dab: u64,
    /// The tiles whose state the stroke keeps, each with its buffers.
    tiles: BTreeMap<TileKey, TileStroke>,
    /// The buffers no tile holds, ready for the next tile that needs state: with `tiles`, the
    /// store's [`TileStore::stroke_tiles`] of them.
    free: Vec<TileStroke>,
}

impl Stroke {
    /// Dabs for which a tile edited stays protected from release: a tile edited in the last 4
    /// dabs, the current one included, is never released (see "Releasing tiles" on
    /// [`Stroke`]).
    pub const PROTECTED_DABS: u64 = 4;

    /// A stroke state over `store`: the buffers of [`TileStore::stroke_tiles`] tiles, allocated
    /// now and held for the `Stroke`'s life (see "Memory" on [`Stroke`]). Zero allocates none,
    /// and every tile is written per dab.
    #[must_use]
    pub fn new(store: &TileStore) -> Self {
        Self {
            store: store.id(),
            op: None,
            dab: 0,
            tiles: BTreeMap::new(),
            free: (0..store.stroke_tiles())
                .map(|_| TileStroke::new())
                .collect(),
        }
    }

    /// Readies the state for a dab over `store` that edits `edited` (in key order): releases
    /// every tile if the store or its op isn't the stroke's (taking the size of a new store's
    /// reservation), counts the dab, and gives each edited tile state, in key order, while some
    /// is free or can be released (see [`Stroke`]). Returns how many edited tiles got none: those
    /// are written per dab.
    pub(super) fn begin_dab(&mut self, store: &TileStore, edited: &[TileKey]) -> usize {
        let op = store.current_op();
        if store.id() != self.store {
            self.release_all();
            self.store = store.id();
            self.free.truncate(store.stroke_tiles());
            let missing = store.stroke_tiles().saturating_sub(self.free.len());
            self.free.extend((0..missing).map(|_| TileStroke::new()));
        } else if op != self.op {
            self.release_all();
        }
        self.op = op;
        self.dab += 1;
        // Tiles already held are marked first, so none of this dab's is released for another.
        for key in edited {
            if let Some(tile) = self.tiles.get_mut(key) {
                tile.last_dab = self.dab;
            }
        }
        let mut overflow = 0;
        for key in edited {
            if self.tiles.contains_key(key) {
                continue;
            }
            if self.free.is_empty() {
                self.release_oldest();
            }
            match self.free.pop() {
                Some(mut tile) => {
                    tile.take(self.dab);
                    self.tiles.insert(*key, tile);
                }
                None => overflow += 1,
            }
        }
        overflow
    }

    /// The state of each tile of `edited`, aligned with it: `None` for a tile written per dab.
    pub(super) fn states(&self, edited: &[TileKey]) -> Vec<Option<&TileStroke>> {
        edited.iter().map(|key| self.tiles.get(key)).collect()
    }

    /// Records `store`'s op as of the end of a dab: the op its edits were made in, which the
    /// edit may have begun itself.
    pub(super) const fn end_dab(&mut self, store: &TileStore) {
        self.op = store.current_op();
    }

    /// How many tiles' state the stroke holds now.
    #[cfg(test)]
    pub(crate) fn tiles_held(&self) -> usize {
        self.tiles.len()
    }

    /// Whether the stroke holds `key`'s state now.
    #[cfg(test)]
    pub(crate) fn holds(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    /// How many tiles' buffers the stroke has, held or free.
    #[cfg(test)]
    pub(crate) fn buffers(&self) -> usize {
        self.tiles.len() + self.free.len()
    }

    /// Releases every tile: a new stroke begins.
    fn release_all(&mut self) {
        while let Some((_, tile)) = self.tiles.pop_first() {
            self.free.push(tile);
        }
    }

    /// Releases the tile with the oldest last dab, the least key among equals, of those not
    /// edited in the last [`Stroke::PROTECTED_DABS`] dabs. Returns whether one was.
    ///
    /// A scan of the tiles held: it runs only when every buffer is taken, which a stroke over the
    /// default reservation reaches only when it has edited 512 tiles.
    fn release_oldest(&mut self) -> bool {
        let dab = self.dab;
        let oldest = self
            .tiles
            .iter()
            .filter(|(_, tile)| dab.saturating_sub(tile.last_dab) >= Self::PROTECTED_DABS)
            .min_by_key(|&(key, tile)| (tile.last_dab, *key))
            .map(|(key, _)| *key);
        let Some(tile) = oldest.and_then(|key| self.tiles.remove(&key)) else {
            return false;
        };
        self.free.push(tile);
        true
    }
}

/// One tile's stroke state: per vertex, row-major in the tile's own side (a coarser tile uses
/// the first `side²` entries), the start and the sum of the stroke's deltas.
///
/// The entries are atomics only because the store's edit calls its closure on several threads
/// (it is `Fn + Sync`): each vertex's entries are touched by one thread per dab, and one dab's
/// edit ends before the next begins, so [`Ordering::Relaxed`] is enough and the result is the
/// same at any worker count (see [`TileStroke::write`]).
pub(super) struct TileStroke {
    /// The `f32` bits of each vertex's correction `z` as the stroke first found it, or
    /// [`NO_START`] until the stroke first writes it.
    start: Box<[AtomicU32]>,
    /// The `f32` bits of the sum of the stroke's deltas for each vertex, from zero.
    acc: Box<[AtomicU32]>,
    /// The dab that last edited the tile.
    last_dab: u64,
    /// Whether a tile has held the buffer since it was cleared: a buffer still clean from
    /// [`TileStroke::new`] needs no clearing before its first tile.
    used: bool,
}

impl std::fmt::Debug for TileStroke {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileStroke")
            .field("entries", &self.start.len())
            .field("last_dab", &self.last_dab)
            .field("used", &self.used)
            .finish_non_exhaustive()
    }
}

impl TileStroke {
    /// A tile's state for the largest tile, every start not captured and every sum zero.
    fn new() -> Self {
        Self {
            start: (0..ENTRIES).map(|_| AtomicU32::new(NO_START)).collect(),
            acc: (0..ENTRIES)
                .map(|_| AtomicU32::new(0.0_f32.to_bits()))
                .collect(),
            last_dab: 0,
            used: false,
        }
    }

    /// Readies the buffer for a tile first edited at dab `dab`: no start captured and every sum
    /// zero, clearing what an earlier tile left.
    fn take(&mut self, dab: u64) {
        if self.used {
            for start in &mut self.start {
                *start.get_mut() = NO_START;
            }
            for acc in &mut self.acc {
                *acc.get_mut() = 0.0_f32.to_bits();
            }
        }
        self.used = true;
        self.last_dab = dab;
    }

    /// The `z` to write for entry `at`, whose correction's `z` is stored as `current`, after
    /// adding `delta` to its sum: `start + sum`, capturing `current` as the start if this is the
    /// stroke's first write of it. `None` if `at` is past the entries (never, for a tile's own
    /// vertex).
    ///
    /// The store's edit calls its closure exactly once per vertex of each tile, a tile's interior
    /// on one thread, and the brush gives a vertex its delta in one tile alone (its owner): so
    /// this entry is read and written by one thread in a dab, and `current` is the value before
    /// this dab writes it, which, the first time, is the value the stroke found.
    pub(super) fn write(&self, at: usize, current: f32, delta: f32) -> Option<f32> {
        let start_bits = self.start.get(at)?;
        let acc_bits = self.acc.get(at)?;
        let mut start = f32::from_bits(start_bits.load(Ordering::Relaxed));
        if start.is_nan() {
            start = current;
            start_bits.store(current.to_bits(), Ordering::Relaxed);
        }
        let acc = f32::from_bits(acc_bits.load(Ordering::Relaxed)) + delta;
        acc_bits.store(acc.to_bits(), Ordering::Relaxed);
        Some(start + acc)
    }
}
