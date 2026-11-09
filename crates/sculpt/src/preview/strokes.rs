//! The open strokes whose dabs a [`Preview`](super::Preview) shows ([`OpenStrokes`]), and one
//! stroke shown alone ([`Lone`]).
//!
//! A stroke begun while the strokes before it still land reads their **futures**: its pen-down
//! shapes, in a set of its own, are the tiles as those strokes will leave them (the store's "Ops
//! reserved ahead"). Its dabs are shown over those shapes, so the caller lets the preview read
//! each stroke's set ([`OpenStrokes::read`]) and has the futures it is about to read installed
//! first ([`OpenStrokes::ready`]).

use geom::TileKey;
use store::{CacheError, OpId, PenDownReading, SpatialIndex, TileStore};
use stroke::Dab;

/// The strokes under way whose dabs a [`Preview`](super::Preview) shows, oldest first.
///
/// As the store keeps their pen-down shapes: the oldest's op begun, the store's current op, and
/// every later one's reserved behind it (see the module notes).
pub trait OpenStrokes {
    /// The store, its pen-down reads going to the current stroke's shapes
    /// ([`TileStore::read_pen_down`]), and the spatial index.
    fn current(&mut self) -> (PenDownReading<'_>, &SpatialIndex);

    /// The store, its pen-down reads going to stroke `op`'s shapes: the current stroke's for an
    /// op begun, a reserved op's own ([`TileStore::read_pen_down_of`]); and the spatial index.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotReserved`] (or another) if `op` is no stroke under way: its shapes are
    /// gone.
    fn read(&mut self, op: OpId) -> Result<(PenDownReading<'_>, &SpatialIndex), CacheError>;

    /// Whether stroke `op`'s op is reserved: none of its dabs can land yet, and its pen-down
    /// shapes are the futures of the strokes before it.
    fn reserved(&self, op: OpId) -> bool;

    /// Installs, for reserved stroke `op`, the future of every tile the strokes before it still
    /// change that a preview of its dabs reads (none for an op begun): the finest tiles within
    /// the spheres of `dabs` (as the spatial index finds them once refitted over the futures) and
    /// in their rings, and the finest tiles a display preview of each of `displays` reads (every
    /// tile under it, and around those). Returns whether they all are installed: if not, the
    /// preview leaves those dabs or display tiles for later.
    fn ready(&mut self, op: OpId, dabs: &[Dab], displays: &[TileKey]) -> bool;
}

/// One stroke shown alone, the store's current one: what [`Preview::show`](super::Preview::show)
/// and the other calls taking a store show.
#[derive(Debug)]
pub(super) struct Lone<'a> {
    /// The store.
    pub(super) store: &'a mut TileStore,
    /// The spatial index.
    pub(super) index: &'a SpatialIndex,
}

impl OpenStrokes for Lone<'_> {
    fn current(&mut self) -> (PenDownReading<'_>, &SpatialIndex) {
        (self.store.read_pen_down(), self.index)
    }

    fn read(&mut self, _op: OpId) -> Result<(PenDownReading<'_>, &SpatialIndex), CacheError> {
        Ok(self.current())
    }

    fn reserved(&self, _op: OpId) -> bool {
        false
    }

    fn ready(&mut self, _op: OpId, _dabs: &[Dab], _displays: &[TileKey]) -> bool {
        true
    }
}
