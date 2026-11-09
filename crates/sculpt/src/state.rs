//! A stroke's state: the value each touched vertex's correction is being built to, kept in `f32`
//! so a dab's small delta isn't rounded away by the tile's quantization (see [`Stroke`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};

use geom::{TILE_QUADS, TileKey, Vec3};
use store::{OpId, StoreId, TileStore};

/// Vertices of the largest tile, the entries of each [`TileStroke`]: `(TILE_QUADS + 1)²`.
const ENTRIES: usize = (TILE_QUADS as usize + 1) * (TILE_QUADS as usize + 1);

/// The bits of a value not captured yet: a NaN, which a stored correction never is.
const NO_START: u32 = f32::NAN.to_bits();

/// The state one brush stroke keeps across its dabs, so that its small dabs move a tile's
/// interior as far as its border.
///
/// A tile stores its interior corrections quantized, each component to a step of its plane's
/// scale (the tangential planes' at least 8 times coarser than the normal plane's), and its
/// border vertices exactly. Were each dab to add its delta to the stored value, a delta under half
/// a step would vanish on the interior and be kept on the border, building ridges along tile
/// seams. Instead, for every vertex a stroke writes, it keeps the value it is building for each of
/// the correction's three components in `f32`: the correction as it first found it (the start),
/// plus every dab's delta since, and writes that. The stored value is rounded when it is written,
/// not added to dab by dab, and interior and border move alike. All three components: draw moves
/// a vertex along the dab's area normal, which is no vertex's own frame normal, so its deltas have
/// tangential parts too, and they need the same once-a-stroke rounding as the normal part.
///
/// # Error bound
///
/// Each component of an interior vertex's stored value differs from its value before the stroke
/// plus the stroke's deltas by at most half a step of its plane's scale for each time its start
/// is captured: once, unless its tile's state is released and the vertex edited again (see
/// "Releasing tiles"), when the start is captured again from the rounded stored value. Should a
/// plane's scale grow during the stroke (its own values outgrowing it, or a tangential plane
/// raised with the normal plane's), the re-quantization rounds that plane's values written before
/// it again, and its bound is one step of its final scale for each capture instead. The value
/// accumulates as a running `f32`, so a delta under half an ulp of the value it is added to is
/// lost: far below a quantization step, as the values are corrections, not positions.
///
/// # Memory
///
/// [`Stroke::new`] allocates the state of [`TileStore::stroke_tiles`] tiles up front, three `f32`
/// per vertex of the largest tile ([`store::StoreConfig::STROKE_TILE_BYTES`], 50,700 bytes), and
/// nothing more while it is used over that store: the store has set that much aside from its
/// budget ([`store::StoreConfig::stroke_tiles`]). A store reserves for one stroke state, so keep
/// one `Stroke` per store for the session, and pass it to every [`crate::apply_dabs`]. It starts
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
/// [`crate::ApplyStats::tiles_unaccumulated`]. Nothing fails. Dabs count one by one, in order,
/// however many one [`crate::apply_dabs`] call applies.
#[derive(Debug)]
pub struct Stroke {
    /// The store the state is kept over. A dab over another releases every tile first.
    store: StoreId,
    /// The op of the stroke the state belongs to: the store's op when the last dab ended.
    /// A dab that finds the store at another op releases every tile first.
    op: Option<OpId>,
    /// Which tiles hold state, the dab that last edited each and the dabs counted: the
    /// decisions of "Releasing tiles", kept apart from the buffers so a frame can make them
    /// ahead of its writes (see [`Book`]).
    book: Book,
    /// The tiles whose state the stroke keeps, each with its buffers: those `book` holds.
    tiles: BTreeMap<TileKey, TileStroke>,
    /// The buffers no tile holds, ready for the next tile that needs state: with `tiles`, the
    /// store's [`TileStore::stroke_tiles`] of them, as many as `book` counts free.
    free: Vec<TileStroke>,
}

/// What a dab's edit of a tile finds of the tile's stroke state, as [`Book::begin_dab`] grants
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Grant {
    /// The tile held state before the dab: its values go on.
    Held,
    /// The tile was given a buffer at the dab: every value uncaptured.
    Taken,
    /// No buffer was free or could be released: the tile is written per dab (see "The thrash
    /// limit" on [`Stroke`]).
    Without,
}

/// The bookkeeping of a [`Stroke`]: which tiles hold state, the dab that last edited each, the
/// dabs counted and the buffers free.
///
/// Its decisions are those of "Releasing tiles" on [`Stroke`], made from the tiles each dab
/// edits and nothing else, so a copy can make a frame's decisions after the frame's dabs are
/// worked out and before anything is written ([`crate::apply_dabs`]), and they are the
/// stroke's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Book {
    /// The dabs counted so far, the current one included: each tile's last dab is one of them.
    dab: u64,
    /// The tiles holding state, each with the dab that last edited it.
    held: BTreeMap<TileKey, u64>,
    /// How many buffers no tile holds.
    free: usize,
}

impl Book {
    /// A book of `buffers` free buffers and no dab counted.
    const fn new(buffers: usize) -> Self {
        Self {
            dab: 0,
            held: BTreeMap::new(),
            free: buffers,
        }
    }

    /// Counts a dab that edits `edited` (in key order) and gives each edited tile state, in key
    /// order, while some is free or can be released (see [`Stroke`]). Returns the grants,
    /// aligned with `edited`, after adding the tiles released for them to `released`, in the
    /// order they were released.
    pub(crate) fn begin_dab(
        &mut self,
        edited: &[TileKey],
        released: &mut Vec<TileKey>,
    ) -> Vec<Grant> {
        self.dab += 1;
        // Tiles already held are marked first, so none of this dab's is released for another.
        for key in edited {
            if let Some(last) = self.held.get_mut(key) {
                *last = self.dab;
            }
        }
        let mut grants = Vec::with_capacity(edited.len());
        for key in edited {
            if self.held.contains_key(key) {
                grants.push(Grant::Held);
                continue;
            }
            if self.free == 0
                && let Some(oldest) = self.release_oldest()
            {
                released.push(oldest);
            }
            if self.free > 0 {
                self.free -= 1;
                self.held.insert(*key, self.dab);
                grants.push(Grant::Taken);
            } else {
                grants.push(Grant::Without);
            }
        }
        grants
    }

    /// Whether `key` holds state now.
    pub(crate) fn holds(&self, key: &TileKey) -> bool {
        self.held.contains_key(key)
    }

    /// Whether dabs that edit only tiles of `reach` give every tile they edit state without
    /// releasing any: the tiles of `reach` holding none number at most the buffers free.
    ///
    /// Then every grant they make is [`Grant::Held`] for a tile holding state, and
    /// [`Grant::Taken`] at the first edit of one that doesn't, never [`Grant::Without`]: a buffer
    /// is released only when none is free and an edited tile needs one, and each tile of `reach`
    /// takes at most one. Which tiles the dabs edit, and in what order, doesn't matter.
    fn has_room_for(&self, reach: &BTreeSet<TileKey>) -> bool {
        let unheld = reach.iter().filter(|key| !self.holds(key)).count();
        unheld <= self.free
    }

    /// Releases every tile: a new stroke begins.
    fn release_all(&mut self) {
        self.free += self.held.len();
        self.held.clear();
    }

    /// Releases the tile with the oldest last dab, the least key among equals, of those not
    /// edited in the last [`Stroke::PROTECTED_DABS`] dabs, and returns it; `None` if every
    /// tile held is protected.
    ///
    /// A scan of the tiles held: it runs only when every buffer is taken, which a stroke over the
    /// default reservation reaches only when it has edited 512 tiles.
    fn release_oldest(&mut self) -> Option<TileKey> {
        let dab = self.dab;
        let oldest = self
            .held
            .iter()
            .filter(|&(_, &last)| dab.saturating_sub(last) >= Stroke::PROTECTED_DABS)
            .min_by_key(|&(key, &last)| (last, *key))
            .map(|(key, _)| *key)?;
        self.held.remove(&oldest);
        self.free += 1;
        Some(oldest)
    }
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
            book: Book::new(store.stroke_tiles()),
            tiles: BTreeMap::new(),
            free: (0..store.stroke_tiles())
                .map(|_| TileStroke::new())
                .collect(),
        }
    }

    /// Readies the state for dabs over `store`: releases every tile if the store or its op isn't
    /// the stroke's (taking the size of a new store's reservation), and takes the store's op.
    ///
    /// Every dab does this before it counts. Within one frame, only the first dab can find
    /// anything to do: an op changes only when an edit begins one, which the first dab that
    /// writes does, and the op it begins is the one the next dab finds.
    pub(crate) fn begin_frame(&mut self, store: &TileStore) {
        let op = store.current_op();
        if store.id() != self.store {
            self.release_all();
            self.store = store.id();
            self.free.truncate(store.stroke_tiles());
            let missing = store.stroke_tiles().saturating_sub(self.free.len());
            self.free.extend((0..missing).map(|_| TileStroke::new()));
            self.book = Book::new(self.free.len());
        } else if op != self.op {
            self.release_all();
        }
        self.op = op;
    }

    /// Readies the state for a dab over `store` that edits `edited` (in key order): as
    /// [`Stroke::begin_frame`], then counts the dab and gives each edited tile state, in key
    /// order, while some is free or can be released (see [`Stroke`]). Returns how many edited
    /// tiles got none: those are written per dab.
    pub(crate) fn begin_dab(&mut self, store: &TileStore, edited: &[TileKey]) -> usize {
        self.begin_frame(store);
        let mut released = Vec::new();
        let grants = self.book.begin_dab(edited, &mut released);
        for key in &released {
            if let Some(tile) = self.tiles.remove(key) {
                self.free.push(tile);
            }
        }
        let mut overflow = 0;
        let granted = edited.iter().zip(grants);
        for (key, grant) in granted {
            match grant {
                Grant::Held => {}
                Grant::Taken => {
                    if let Some(mut tile) = self.free.pop() {
                        tile.take();
                        self.tiles.insert(*key, tile);
                    }
                }
                Grant::Without => overflow += 1,
            }
        }
        overflow
    }

    /// The state of each tile of `edited`, aligned with it: `None` for a tile written per dab.
    pub(crate) fn states(&self, edited: &[TileKey]) -> Vec<Option<&TileStroke>> {
        edited.iter().map(|key| self.tiles.get(key)).collect()
    }

    /// The state `key` holds now, if it holds any.
    pub(crate) fn held(&self, key: &TileKey) -> Option<&TileStroke> {
        self.tiles.get(key)
    }

    /// The state the next dab of the stroke of op `op` over `store` goes on from: this one if it
    /// is that stroke's, the store's and the op it last wrote ([`Stroke::end_dab`]); `None` if the
    /// next dab starts afresh ([`Stroke::begin_frame`] releases every tile when it finds another
    /// store or op).
    ///
    /// A stroke whose op hasn't written yet (one not begun, or one waiting on the stroke before
    /// it) never matches: its first dab finds the state at another op, and releases it.
    pub(crate) fn continued(&self, store: &TileStore, op: OpId) -> Option<&Self> {
        (self.store == store.id() && self.op == Some(op)).then_some(self)
    }

    /// Whether the dabs of a stroke that edit only tiles of `reach`, over `store`, from `state`
    /// (as [`Stroke::continued`] gives it; `None`: afresh, every buffer of the store's
    /// reservation free) make no decision but [`Grant::Held`] and a first [`Grant::Taken`]: no
    /// tile's state is released or denied (see [`Book::has_room_for`]).
    ///
    /// What a replay under [`crate::apply::Plan::Continuous`] assumes, so then its tiles are, bit
    /// for bit, what the dabs one at a time leave, whichever tiles they edit.
    pub(crate) fn has_room_for(
        state: Option<&Self>,
        store: &TileStore,
        reach: &BTreeSet<TileKey>,
    ) -> bool {
        match state {
            Some(stroke) => stroke.book.has_room_for(reach),
            None => reach.len() <= store.stroke_tiles(),
        }
    }

    /// The stroke's bookkeeping as it is now: a frame makes its dabs' decisions on a copy (see
    /// [`Book`]).
    pub(crate) const fn book(&self) -> &Book {
        &self.book
    }

    /// Records `store`'s op as of the end of a dab: the op its edits were made in, which the
    /// edit may have begun itself.
    pub(crate) const fn end_dab(&mut self, store: &TileStore) {
        self.op = store.current_op();
    }

    /// Ends dabs written together over `store`: `book` is this stroke's book after them (made
    /// on a copy of [`Stroke::book`] with [`Book::begin_dab`] for each, in order), and `finals`
    /// the state each tile they edited with state ends with.
    ///
    /// Each tile `book` no longer holds gives its buffer back. Each of `finals` that `book`
    /// holds gets its values: in the buffer it held throughout if it was
    /// [`FinalState::continuous`], else in a buffer taken afresh, cleared first as a dab's take
    /// clears it. A tile `book` holds that isn't among `finals` keeps its buffer as it is: the
    /// dabs didn't write it. Then the store's op is the stroke's ([`Stroke::end_dab`]).
    pub(crate) fn end_frame(&mut self, store: &TileStore, book: Book, finals: Vec<FinalState>) {
        let gone: Vec<TileKey> = self
            .tiles
            .keys()
            .filter(|key| !book.holds(key))
            .copied()
            .collect();
        for key in gone {
            if let Some(tile) = self.tiles.remove(&key) {
                self.free.push(tile);
            }
        }
        for state in finals {
            if !book.holds(&state.key) {
                continue;
            }
            let buffer = match self.tiles.remove(&state.key) {
                Some(tile) if state.continuous => Some(tile),
                Some(mut tile) => {
                    tile.take();
                    Some(tile)
                }
                None => self.free.pop().map(|mut tile| {
                    tile.take();
                    tile
                }),
            };
            if let Some(mut tile) = buffer {
                tile.fill(&state.value);
                self.tiles.insert(state.key, tile);
            }
        }
        self.book = book;
        self.end_dab(store);
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

    /// Everything the stroke holds, in bits: its book, and each held tile's values. Two strokes
    /// that would write the same from now on have the same.
    #[cfg(test)]
    pub(crate) fn bits(&self) -> (Book, Vec<TileBits>) {
        let tiles = self.tiles.iter().map(|(key, tile)| {
            let entries = (0..ENTRIES)
                .filter_map(|at| tile.entry(at))
                .map(Vec3::to_bits)
                .collect();
            (*key, entries)
        });
        (self.book.clone(), tiles.collect())
    }

    /// Releases every tile: a new stroke begins.
    fn release_all(&mut self) {
        self.book.release_all();
        while let Some((_, tile)) = self.tiles.pop_first() {
            self.free.push(tile);
        }
    }
}

/// A held tile's stroke state in bits, for the tests: each entry's value.
#[cfg(test)]
pub(crate) type TileBits = (TileKey, Vec<[u32; 3]>);

/// The stroke state a tile ends a frame's dabs with, for [`Stroke::end_frame`]: per vertex of
/// the tile, row-major, the value as [`TileStroke`] holds it.
#[derive(Debug)]
pub(crate) struct FinalState {
    /// The tile.
    pub(crate) key: TileKey,
    /// Whether the tile held state from before the dabs and kept it throughout, so these values
    /// continue the buffer it holds.
    pub(crate) continuous: bool,
    /// Each vertex's value, every component `NaN` where the stroke hasn't captured one.
    pub(crate) value: Vec<Vec3>,
}

/// One tile's stroke state: per vertex, row-major in the tile's own side (a coarser tile uses
/// the first `side²` entries), the value the stroke is building for each of the correction's
/// three components.
///
/// The entries are atomics only because the store's edit calls its closure on several threads
/// (it is `Fn + Sync`): each vertex's entries are touched by one thread per dab, and one dab's
/// edit ends before the next begins, so [`Ordering::Relaxed`] is enough and the result is the
/// same at any worker count (see [`TileStroke::write`]).
pub(crate) struct TileStroke {
    /// The `f32` bits of each vertex's value, its three components in turn: the correction as the
    /// stroke first found it plus every delta since, or [`NO_START`] in every component until the
    /// stroke first writes it.
    value: Box<[AtomicU32]>,
}

impl std::fmt::Debug for TileStroke {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileStroke")
            .field("entries", &(self.value.len() / 3))
            .finish_non_exhaustive()
    }
}

impl TileStroke {
    /// A tile's state for the largest tile, every value not captured.
    fn new() -> Self {
        Self {
            value: (0..3 * ENTRIES).map(|_| AtomicU32::new(NO_START)).collect(),
        }
    }

    /// Readies the buffer for a tile first edited now: no value captured, clearing what an
    /// earlier tile left.
    ///
    /// Every entry is cleared: at the levels a brush edits, every tile is the largest, so
    /// clearing only a tile's own `side²` entries would save nothing; and skipping buffers still
    /// clean from [`TileStroke::new`] measured no faster (the cold first dab of the medium
    /// project's `brush`, interleaved runs).
    fn take(&mut self) {
        for value in &mut self.value {
            *value.get_mut() = NO_START;
        }
    }

    /// Sets the first entries' values to `value`: a tile's state as dabs written together left it
    /// ([`Stroke::end_frame`]).
    fn fill(&mut self, value: &[Vec3]) {
        let entries = self.value.as_chunks_mut::<3>().0.iter_mut().zip(value);
        for (to, built) in entries {
            let components = to.iter_mut().zip([built.x, built.y, built.z]);
            for (component, v) in components {
                *component.get_mut() = v.to_bits();
            }
        }
    }

    /// Entry `at`'s value as the stroke holds it now, every component `NaN` where it hasn't
    /// captured the vertex; `None` if `at` is past the entries.
    pub(crate) fn entry(&self, at: usize) -> Option<Vec3> {
        let [x, y, z] = self.components(at)?;
        let load = |c: &AtomicU32| f32::from_bits(c.load(Ordering::Relaxed));
        Some(Vec3::new(load(x), load(y), load(z)))
    }

    /// The atomics of entry `at`'s three components; `None` if `at` is past the entries.
    fn components(&self, at: usize) -> Option<&[AtomicU32; 3]> {
        let first = at.checked_mul(3)?;
        self.value
            .get(first..first.checked_add(3)?)?
            .try_into()
            .ok()
    }

    /// The correction to write for entry `at`, whose correction is stored as `current`, after
    /// adding `delta`: the value the stroke builds ([`stroke_sum`]), capturing `current` as its
    /// start if this is the stroke's first write of it. `None` if `at` is past the entries
    /// (never, for a tile's own vertex).
    ///
    /// The store's edit calls its closure exactly once per vertex of each tile, a tile's interior
    /// on one thread, and the brush gives a vertex its delta in one tile alone (its owner): so
    /// this entry is read and written by one thread in a dab, and `current` is the value before
    /// this dab writes it, which, the first time, is the value the stroke found.
    pub(crate) fn write(&self, at: usize, current: Vec3, delta: Vec3) -> Option<Vec3> {
        let built = stroke_sum(self.entry(at)?, current, delta);
        let components = self.components(at)?.iter().zip([built.x, built.y, built.z]);
        for (to, v) in components {
            to.store(v.to_bits(), Ordering::Relaxed);
        }
        Some(built)
    }
}

/// A vertex's value after a write of `delta`, which is also the correction written.
///
/// From its value `built` (every component `NaN` if not captured yet) and its correction stored
/// as `current`: the start is `current` if the value wasn't captured, and the value gains `delta`,
/// component by component.
///
/// The one arithmetic of a stroke's write, shared by [`TileStroke::write`] and a frame's
/// replay of its dabs, so both give the same bits.
pub(crate) fn stroke_sum(built: Vec3, current: Vec3, delta: Vec3) -> Vec3 {
    let start = if built.x.is_nan() { current } else { built };
    start + delta
}
