//! The pen-down shapes of the ops reserved ahead of the current one.
//!
//! Each op's are kept in a set of its own, and one op's set may be read in place of the current
//! stroke's (see "Ops reserved ahead" in the `pen_down` module notes).

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

use geom::{CorrectionTile, TileKey};

use super::{ENTRY_BYTES, PenDown, PenDownShape, check_installed};
use crate::tiles::config::OPEN_STROKES_SHARE;
use crate::tiles::{CacheError, ReservedOp, TileStore};
use crate::{Aabb, OpId};

/// The ops reserved and not yet begun ([`TileStore::reserve_op`]), each with its pen-down set.
///
/// An op is reserved while it has a set here, or while its set is being read (then it sits in the
/// store's `pen_down`, and the current stroke's set waits in `reading`).
#[derive(Debug, Default)]
pub(crate) struct Reserved {
    /// Each reserved op's set, in op order, but for the op being read.
    sets: BTreeMap<OpId, PenDown>,
    /// The op whose set is being read, if one is, and the current stroke's set, put aside until
    /// the reading ends.
    reading: Option<Reading>,
}

/// A reserved op's set being read ([`TileStore::read_pen_down_of`]).
#[derive(Debug)]
struct Reading {
    /// The op whose set is the store's `pen_down` while it is read.
    op: OpId,
    /// The current stroke's set, put aside.
    current: PenDown,
}

impl Reserved {
    /// Reserves `op`, newer than every op reserved, with no shapes yet.
    pub(in crate::tiles) fn insert(&mut self, op: OpId) {
        let read = self.reading.as_ref().map(|read| read.op);
        let newest = self.sets.keys().copied().chain(read).max();
        debug_assert!(
            newest.is_none_or(|newest| newest < op),
            "ops are reserved in order"
        );
        self.sets.insert(op, PenDown::default());
    }

    /// Whether `op` is reserved (its set kept or being read).
    pub(in crate::tiles) fn holds(&self, op: OpId) -> bool {
        self.sets.contains_key(&op) || self.reading.as_ref().is_some_and(|read| read.op == op)
    }

    /// The oldest op reserved, which begins first; `None` with none reserved.
    pub(in crate::tiles) fn oldest(&self) -> Option<OpId> {
        let kept = self.sets.first_key_value().map(|(op, _)| *op);
        let read = self.reading.as_ref().map(|read| read.op);
        kept.into_iter().chain(read).min()
    }

    /// Takes reserved op `op`'s set out, the op no longer reserved; `None` if it isn't, or is
    /// being read (the callers stop the reading first).
    pub(in crate::tiles) fn take(&mut self, op: OpId) -> Option<PenDown> {
        self.sets.remove(&op)
    }

    /// Whether a reserved op's set is being read.
    pub(in crate::tiles) const fn is_reading(&self) -> bool {
        self.reading.is_some()
    }

    /// The bytes the sets hold beside the store's `pen_down`: every reserved op's kept here, and
    /// the current stroke's while it is put aside.
    pub(in crate::tiles) fn held(&self) -> usize {
        let current = self.reading.as_ref().map_or(0, |read| read.current.held());
        self.sets.values().map(PenDown::held).sum::<usize>() + current
    }

    /// The union of the boxes of tile `key`'s pen-down shapes in the sets kept here.
    fn bounds(&self, key: &TileKey) -> Option<Aabb> {
        let current = self.reading.as_ref().map(|read| &read.current);
        let sets = self.sets.values().chain(current);
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
/// that op's shapes, a tile it has none of as the store has it now, until it is dropped. It is
/// for reading: keeping a shape through it while it reads a reserved op's is a bug (see the
/// module notes).
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

/// Puts the current stroke's shapes back where its reads find them, if a reserved op's were
/// read: as the reading ends, or as a panic unwinds through it.
impl Drop for PenDownReading<'_> {
    fn drop(&mut self) {
        self.store.stop_reading_reserved();
    }
}

impl TileStore {
    /// The store, its pen-down reads going to the current stroke's shapes, as they always do
    /// outside a [`TileStore::read_pen_down_of`]: what a stroke's surface reads.
    pub fn read_pen_down(&mut self) -> PenDownReading<'_> {
        self.stop_reading_reserved();
        PenDownReading { store: self }
    }

    /// The store, its pen-down reads going to reserved op `reserved`'s shapes until the reading
    /// is dropped: the futures installed for it ([`TileStore::install_pen_down_of`]), every other
    /// tile read as the store has it now. What the stroke the op was reserved for reads, its
    /// surface the one the strokes before it will leave.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotReserved`] if `reserved` isn't reserved on this store.
    pub fn read_pen_down_of(
        &mut self,
        reserved: &ReservedOp,
    ) -> Result<PenDownReading<'_>, CacheError> {
        self.stop_reading_reserved();
        let op = reserved.op();
        let set = if reserved.store() == self.id() {
            self.reserved.take(op)
        } else {
            None
        };
        let set = set.ok_or(CacheError::NotReserved(op))?;
        let current = std::mem::replace(&mut self.pen_down, set);
        self.reserved.reading = Some(Reading { op, current });
        Ok(PenDownReading { store: self })
    }

    /// Ends a reading of a reserved op's shapes, if one is under way: its set goes back among the
    /// reserved ops', and the current stroke's back where its reads find it.
    pub(in crate::tiles) fn stop_reading_reserved(&mut self) {
        if let Some(Reading { op, current }) = self.reserved.reading.take() {
            let set = std::mem::replace(&mut self.pen_down, current);
            self.reserved.sets.insert(op, set);
        }
    }

    /// Installs `corrections` as tile `key`'s pen-down shape for reserved op `reserved`, if it has
    /// none of the tile: the tile's **future**, the corrections the strokes before the op will
    /// leave it with, which the op's reads find there ([`TileStore::read_pen_down_of`]). Install
    /// every future the op reads before it reads them (see "Ops reserved ahead" in the module
    /// notes); the normals and previews' blocks it keeps made of its shapes are made again after
    /// one is installed. Refit the tile in the spatial index after
    /// ([`SpatialIndex::refit`](crate::SpatialIndex::refit)), so its box there holds the future
    /// and the op's raycasts find it.
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
        self.stop_reading_reserved();
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
            .reserved
            .sets
            .get(&op)
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
        let set = self
            .reserved
            .sets
            .get_mut(&op)
            .ok_or(CacheError::NotReserved(op))?;
        set.insert_shape(key, future);
        set.forget_gathered();
        Ok(())
    }

    /// The bytes the open strokes' pen-down data ([`TileStore::pen_down_bytes`]) may hold
    /// together before no op is reserved ([`TileStore::reserve_op`]) nor shape installed for one
    /// ([`TileStore::install_pen_down_of`]): four times the pools' share of the budget, which is
    /// what a restore may hold of one open stroke. Provisional, to be set from measurement (the
    /// store's `OPEN_STROKES_SHARE`). The current stroke's own shapes are never refused: past the
    /// share, it keeps them all the same, held beyond the budget if it must.
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
