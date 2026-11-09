//! The pen-down shapes of the ops reserved ahead of the current one.
//!
//! Each op's are kept in a set of its own, and one op's set may be read in place of the current
//! stroke's (see "Ops reserved ahead" in the `pen_down` module notes).

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

use geom::{CorrectionTile, TileKey};

use super::{ENTRY_BYTES, PenDown, PenDownShape, check_installed};
use crate::tiles::config::OPEN_STROKES_SHARE;
use crate::tiles::{CacheError, Invariant, ReservedOp, TileStore};
use crate::{Aabb, OpId};

/// The ops reserved and not yet begun ([`TileStore::reserve_op`]), each with its pen-down set,
/// and the readings of them under way.
///
/// The store's `pen_down` holds the set being read: the current stroke's (`reading` is `None`)
/// or a reserved op's (`reading` names it, its set is absent from `sets`, and the current
/// stroke's waits in `current`). So `reading.is_some() == current.is_some()`.
#[derive(Debug, Default)]
pub(crate) struct Reserved {
    /// Each reserved op's set, in op order, but for the op being read.
    sets: BTreeMap<OpId, PenDown>,
    /// The reserved op whose set is being read, if one is.
    reading: Option<OpId>,
    /// The current stroke's set, put aside while a reserved op's is read.
    current: Option<PenDown>,
    /// What each reading under way replaced, the innermost last: each gives it back as it ends
    /// ([`PenDownReading`]'s drop), so readings nest.
    replaced: Vec<Option<OpId>>,
}

impl Reserved {
    /// Reserves `op`, newer than every op reserved, with no shapes yet.
    pub(in crate::tiles) fn insert(&mut self, op: OpId) {
        let newest = self.sets.keys().copied().chain(self.reading).max();
        debug_assert!(
            newest.is_none_or(|newest| newest < op),
            "ops are reserved in order"
        );
        self.sets.insert(op, PenDown::default());
    }

    /// Whether `op` is reserved (its set kept or being read).
    pub(in crate::tiles) fn holds(&self, op: OpId) -> bool {
        self.sets.contains_key(&op) || self.reading == Some(op)
    }

    /// The oldest op reserved, which begins first; `None` with none reserved.
    pub(in crate::tiles) fn oldest(&self) -> Option<OpId> {
        let kept = self.sets.first_key_value().map(|(op, _)| *op);
        kept.into_iter().chain(self.reading).min()
    }

    /// Takes reserved op `op`'s set out, the op no longer reserved; `None` if it isn't, or is
    /// being read (the callers refuse while a reserved op is read).
    pub(in crate::tiles) fn take(&mut self, op: OpId) -> Option<PenDown> {
        self.sets.remove(&op)
    }

    /// Lets `op` and every op reserved after it go, their sets dropped; returns them in op order.
    /// The callers refuse while a reserved op is read.
    pub(in crate::tiles) fn drop_from(&mut self, op: OpId) -> Vec<OpId> {
        let later = self.sets.split_off(&op);
        later.into_keys().collect()
    }

    /// A reserved op whose set some reading under way reads, the innermost's first, if one does:
    /// while one does, nothing may be written to the store, nor an op begun, ended or dropped
    /// (see "Ops reserved ahead" in the module notes).
    pub(in crate::tiles) fn read_now(&self) -> Option<OpId> {
        self.reading
            .or_else(|| self.replaced.iter().rev().flatten().next().copied())
    }

    /// The bytes the sets hold beside the store's `pen_down`: every reserved op's kept here, and
    /// the current stroke's while it is put aside.
    pub(in crate::tiles) fn held(&self) -> usize {
        let current = self.current.as_ref().map_or(0, PenDown::held);
        self.sets.values().map(PenDown::held).sum::<usize>() + current
    }

    /// The union of the boxes of tile `key`'s pen-down shapes in the sets kept here.
    fn bounds(&self, key: &TileKey) -> Option<Aabb> {
        let sets = self.sets.values().chain(self.current.as_ref());
        let boxes = sets.filter_map(|set| set.shapes.get(key).and_then(PenDownShape::bounds));
        boxes.reduce(Aabb::union)
    }
}

/// The store, its pen-down reads going to one open op's shapes: the current stroke's
/// ([`TileStore::read_pen_down`]) or a reserved op's ([`TileStore::read_pen_down_of`]).
///
/// It dereferences to the store, so every pen-down read ([`TileStore::pen_down_positions`],
/// [`TileStore::pen_down_normals`], [`TileStore::pen_down_shape`], the spatial index's
/// [`SpatialIndex::raycast_pen_down`](crate::SpatialIndex::raycast_pen_down) and the rest) reads
/// that op's shapes, a tile it has none of as the store has it now, until it is dropped; then the
/// shapes read before it are read again, whatever happened meanwhile (readings nest, and a panic
/// unwinding through one ends it too). [`TileStore::reading_op`] says whose are read.
///
/// While a reserved op's shapes are read, here or in a reading this one is nested in, the store
/// only reads and takes futures ([`TileStore::install_pen_down_of`]): every edit, every shape kept
/// for the current stroke, and every op begun, ended or dropped is refused (see "Ops reserved
/// ahead" in the module notes).
#[derive(Debug)]
pub struct PenDownReading<'a> {
    /// The store, its reads going to the op's shapes.
    store: &'a mut TileStore,
}

impl Deref for PenDownReading<'_> {
    type Target = TileStore;

    fn deref(&self) -> &TileStore {
        self.store
    }
}

impl DerefMut for PenDownReading<'_> {
    fn deref_mut(&mut self) -> &mut TileStore {
        self.store
    }
}

/// Gives back the shapes read before the reading: as it ends, or as a panic unwinds through it.
impl Drop for PenDownReading<'_> {
    fn drop(&mut self) {
        self.store.end_reading();
    }
}

impl TileStore {
    /// The store, its pen-down reads going to the current stroke's shapes until the reading is
    /// dropped: what a stroke's surface reads. Outside any reading they always do.
    pub fn read_pen_down(&mut self) -> PenDownReading<'_> {
        self.begin_reading(None);
        PenDownReading { store: self }
    }

    /// The store, its pen-down reads going to reserved op `reserved`'s shapes until the reading
    /// is dropped: the futures installed for it ([`TileStore::install_pen_down_of`]), every other
    /// tile read as the store has it now. What the stroke the op was reserved for reads, its
    /// surface the one the strokes before it will leave. Inside a reading of the same op, the
    /// same reading.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotReserved`] if `reserved` isn't reserved on this store.
    pub fn read_pen_down_of(
        &mut self,
        reserved: &ReservedOp,
    ) -> Result<PenDownReading<'_>, CacheError> {
        let op = reserved.op();
        if reserved.store() != self.id() || !self.reserved.holds(op) {
            return Err(CacheError::NotReserved(op));
        }
        self.begin_reading(Some(op));
        Ok(PenDownReading { store: self })
    }

    /// The reserved op whose pen-down shapes the store reads now ([`TileStore::read_pen_down_of`]);
    /// `None` while it reads the current stroke's, as it does outside any reading. What a brush
    /// or a future checks to know which shapes it weighs its dabs on.
    #[must_use]
    pub const fn reading_op(&self) -> Option<OpId> {
        self.reserved.reading
    }

    /// Starts a reading of `op`'s set (a reserved op's, the current stroke's for `None`), held
    /// by the caller to be one: noting what it replaces, for [`TileStore::end_reading`].
    fn begin_reading(&mut self, op: Option<OpId>) {
        self.reserved.replaced.push(self.reserved.reading);
        self.switch_reading(op);
    }

    /// Ends the innermost reading under way: the set it replaced is read again.
    fn end_reading(&mut self) {
        if let Some(replaced) = self.reserved.replaced.pop() {
            self.switch_reading(replaced);
        }
    }

    /// Puts `op`'s set (a reserved op's, the current stroke's for `None`) where the store's reads
    /// find it, the one read now back among the others: nothing if it is there already. The
    /// set must be at hand: a reserved op's is, while it is reserved, and no op is dropped or
    /// begun while one is read.
    fn switch_reading(&mut self, op: Option<OpId>) {
        let reserved = &mut self.reserved;
        if reserved.reading == op {
            return;
        }
        let incoming = match op {
            Some(op) => reserved.sets.remove(&op),
            None => reserved.current.take(),
        };
        // At hand, as the callers make sure; were one not, the reading would read none.
        debug_assert!(incoming.is_some(), "the set read is at hand");
        let outgoing = std::mem::replace(&mut self.pen_down, incoming.unwrap_or_default());
        match reserved.reading {
            Some(read) => {
                reserved.sets.insert(read, outgoing);
            }
            None => reserved.current = Some(outgoing),
        }
        reserved.reading = op;
    }

    /// Refuses a write, or a change of the ops, while a reserved op's shapes are read (see
    /// [`PenDownReading`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] with [`Invariant::ReadingReserved`] while one is.
    pub(in crate::tiles) fn refuse_while_reading(&self) -> Result<(), CacheError> {
        match self.reserved.read_now() {
            Some(op) => Err(Invariant::ReadingReserved(op).into()),
            None => Ok(()),
        }
    }

    /// [`TileStore::refuse_while_reading`] for a call that can't fail: panics instead.
    ///
    /// # Panics
    ///
    /// While a reserved op's shapes are read: a bug of the caller's, which would leave the store
    /// reading the wrong shapes.
    pub(in crate::tiles) fn assert_not_reading(&self) {
        let read = self.reserved.read_now();
        assert!(
            read.is_none(),
            "a reserved op's pen-down shapes are read ({read:?}): no op may change meanwhile"
        );
    }

    /// The set of reserved op `op`: the store's `pen_down` while it is read.
    fn reserved_set_mut(&mut self, op: OpId) -> Option<&mut PenDown> {
        if self.reserved.reading == Some(op) {
            Some(&mut self.pen_down)
        } else {
            self.reserved.sets.get_mut(&op)
        }
    }

    /// Installs `corrections` as tile `key`'s pen-down shape for reserved op `reserved`, if it has
    /// none of the tile: the tile's **future**, the corrections the strokes before the op will
    /// leave it with, which the op's reads find there ([`TileStore::read_pen_down_of`]), at once if
    /// they are under way. Install every future the op reads before it reads them (see "Ops
    /// reserved ahead" in the module notes); the normals and previews' blocks the op keeps made of
    /// the tiles around `key` are made again at their next read. Refit the tile in the spatial
    /// index after ([`SpatialIndex::refit`](crate::SpatialIndex::refit)), so its box there holds
    /// the future and the op's raycasts find it: a reading of the op lists the tiles installed
    /// since it last asked ([`TileStore::take_pen_down_added`]).
    ///
    /// `written_by` labels the shape as [`PenDownShape`]'s op does: the op of the record the
    /// tile will hold once the strokes before land, the last of them whose dabs change it (an
    /// op before `reserved`'s). The shape's box is made from `corrections` over the tile's
    /// prediction, as [`TileStore::keep_pen_down`] makes one, and its bytes are counted with the
    /// open strokes' pen-down data.
    ///
    /// # Errors
    ///
    /// Nothing is installed on an error: [`CacheError::NotReserved`] if `reserved` isn't
    /// reserved on this store; [`CacheError::NotATile`] if `corrections` isn't of `key`'s side,
    /// or `key` isn't a tile a stroke keeps a shape of; [`CacheError::OpenStrokesFull`] if the
    /// shape would take the open strokes' pen-down data past their share
    /// ([`TileStore::open_strokes_share`]); as [`TileStore::evaluate`] for the tile's parents.
    pub fn install_pen_down_of(
        &mut self,
        reserved: &ReservedOp,
        key: TileKey,
        corrections: CorrectionTile,
        written_by: Option<OpId>,
    ) -> Result<(), CacheError> {
        let op = reserved.op();
        if reserved.store() != self.id() || !self.reserved.holds(op) {
            return Err(CacheError::NotReserved(op));
        }
        debug_assert!(
            written_by.is_none_or(|by| by < op),
            "a future is of the strokes before the op's"
        );
        check_installed(key, &corrections)?;
        let has = self
            .reserved_set_mut(op)
            .is_some_and(|set| set.shapes.contains_key(&key));
        if has {
            return Ok(());
        }
        let (held, share) = (self.pen_down_bytes(), self.open_strokes_share());
        let bytes = PenDownShape::bytes_of(key.side()) + ENTRY_BYTES;
        if held.saturating_add(bytes) > share {
            return Err(CacheError::OpenStrokesFull { held, share });
        }
        let future = self.installed_shape(key, corrections, written_by)?;
        let reading = self.reading_around(op, key);
        let set = self
            .reserved_set_mut(op)
            .ok_or(CacheError::NotReserved(op))?;
        set.insert_shape(key, future);
        set.forget_gathered(&reading);
        Ok(())
    }

    /// The tiles reserved op `op` keeps normals or a preview's block made of whose gather reads
    /// tile `key` (its ring and the tile itself, [`TileStore::normal_inputs`]); a tile whose
    /// inputs can't be listed counts as one.
    fn reading_around(&self, op: OpId, key: TileKey) -> Vec<TileKey> {
        let set = if self.reserved.reading == Some(op) {
            Some(&self.pen_down)
        } else {
            self.reserved.sets.get(&op)
        };
        let Some(set) = set else {
            return Vec::new();
        };
        let gathered = set.normals.keys().chain(set.previews.keys());
        gathered
            .filter(|tile| {
                self.normal_inputs(tile)
                    .map_or(true, |inputs| inputs.contains(&key))
            })
            .copied()
            .collect()
    }

    /// The bytes the open strokes' pen-down data ([`TileStore::pen_down_bytes`]) may hold
    /// together before no op is reserved ([`TileStore::reserve_op`]) nor shape installed for one
    /// ([`TileStore::install_pen_down_of`]): the pools' share of the budget, which is what a
    /// restore may hold of one open stroke. Provisional, to be set from measurement (the store's
    /// `OPEN_STROKES_SHARE`). The current stroke's own shapes are never refused: past the share,
    /// it keeps them all the same, held beyond the budget if it must.
    #[must_use]
    pub fn open_strokes_share(&self) -> usize {
        #[cfg(test)]
        if let Some(share) = self.hooks.open_strokes_share {
            return share;
        }
        self.pools_budget().saturating_mul(OPEN_STROKES_SHARE)
    }

    /// The union of the boxes of every pen-down shape of tile `key` the open strokes keep: the
    /// current stroke's and every reserved op's, whichever is being read. What the spatial
    /// index's box of the tile holds besides the tile, so each op's raycasts find it at its
    /// shape.
    pub(crate) fn open_pen_down_bounds(&self, key: &TileKey) -> Option<Aabb> {
        let own = self.pen_down.shapes.get(key).and_then(PenDownShape::bounds);
        own.into_iter()
            .chain(self.reserved.bounds(key))
            .reduce(Aabb::union)
    }
}
