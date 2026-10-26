//! Pins held outside the store: [`PinGuard`]s, and the table of pin sets whose guards were dropped,
//! which the store unpins at its next call.

use std::cell::RefCell;
use std::rc::Rc;

use geom::TileKey;

use crate::pool::SlotId;
use crate::slot::{CorrectionSlot, EvaluatedSlot};

/// Pin sets whose guards have been dropped, waiting for the store to unpin them.
///
/// Shared between the store and its guards. It is borrowed only for a push (a guard's drop) or
/// a take (the store's next call), never across anything that could drop a guard, so the
/// borrows never overlap.
#[derive(Debug, Default)]
pub(super) struct PinTable {
    /// Ids of released pin sets.
    pub(super) released: Vec<u64>,
}

/// Keeps a set of tiles resident: their evaluated positions and their correction tiles stay in
/// the pools, in the same slots, until the guard is dropped.
///
/// An edit of a pinned tile (or of anything it was predicted from) doesn't move it: its slot is
/// recomputed in place on its next [`TileStore::evaluate`](super::TileStore::evaluate). The store
/// applies the unpin at its next call after the drop.
#[derive(Debug)]
#[must_use = "dropping the guard unpins the tiles"]
pub struct PinGuard {
    /// The pin set in the store.
    pub(super) id: u64,
    /// The pinned tiles, in key order.
    pub(super) keys: Vec<TileKey>,
    /// Where the drop is reported.
    pub(super) table: Rc<RefCell<PinTable>>,
}

impl PinGuard {
    /// The pinned tiles, in key order, each once.
    #[must_use]
    pub fn keys(&self) -> &[TileKey] {
        &self.keys
    }
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        self.table.borrow_mut().released.push(self.id);
    }
}

/// The slots one pinned tile holds: its evaluated tile and its correction tile.
pub(super) type Pinned = (SlotId<EvaluatedSlot>, SlotId<CorrectionSlot>);
