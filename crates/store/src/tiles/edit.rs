//! Editing correction tiles: [`TileEdit`], the border and interior passes that carry an edit to
//! every copy of a vertex, and the tiles an edit touched.

use std::collections::BTreeSet;

use geom::{
    CorrectionError, CorrectionView, CorrectionViewMut, FaceId, Frame, Level, MAX_COPIES,
    Prediction, Region, RowUpdate, TileBorder, TileKey, Vec3, VertexCopy,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::TileStore;
use super::error::{CacheError, Invariant};
use super::layers::first_op;
use super::round::keep_first;
use crate::op::OpId;
use crate::pool::SlotId;
use crate::slot::CorrectionSlot;

/// The border vertices of a tile with `side` vertices a side, in row-major order.
///
/// That is the first row, then the first and last vertex of each row inside, then the last row:
/// the order [`TileEdit::update`] visits them in.
pub(super) fn border_order(side: u32) -> impl Iterator<Item = (u32, u32)> {
    let last = side.saturating_sub(1);
    (0..side).flat_map(move |j| {
        let step = if j == 0 || j == last { 1 } else { last.max(1) };
        (0..side)
            .step_by(usize::try_from(step).unwrap_or(1))
            .map(move |i| (i, j))
    })
}

/// A tile's border corrections, read at once in [`border_order`] for the tile's border writes
/// ([`TileEdit::update_border_one`]), instead of a slot lookup per vertex.
#[derive(Debug)]
pub(super) struct BorderValues {
    /// The tile's vertices a side.
    pub(super) side: u32,
    /// The corrections, in [`border_order`].
    pub(super) values: Vec<Vec3>,
}

impl BorderValues {
    /// The correction read for border vertex `(i, j)`; `None` for a vertex inside the tile or
    /// outside it.
    pub(super) fn get(&self, i: u32, j: u32) -> Option<Vec3> {
        let last = self.side.checked_sub(1)?;
        if i > last || j > last {
            return None;
        }
        let at = if j == 0 {
            i
        } else if j == last {
            self.side + 2 * (last - 1) + i
        } else if i == 0 || i == last {
            self.side + 2 * (j - 1) + u32::from(i == last)
        } else {
            return None;
        };
        self.values.get(usize::try_from(at).ok()?).copied()
    }
}

/// What [`update_interior`] did to a tile's interior.
#[derive(Debug)]
struct InteriorPass {
    /// The first row with an interior vertex written: where, in row-major order, the tile's
    /// first interior write falls (after the row's first border vertex, before its last).
    first_row: Option<u32>,
    /// The row where a vertex failed, and why: nothing after it in row-major order was written.
    failed: Option<(u32, CacheError)>,
}

/// Runs `f` over the interior vertices of row `j` of `view`, in order, setting what it returns
/// ([`CorrectionViewMut::update_interior_row`]).
///
/// Returns whether anything was written, and the error of a value that isn't finite, which
/// stops the row (nothing is written for it).
///
/// # Errors
///
/// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
/// [`CorrectionError::TooLarge`] for a value no tile can hold; [`CacheError::Internal`] with
/// [`Invariant::CorrectionSlot`] and the slot's own error for a row outside the tile's interior
/// (the store asks only for interior rows).
pub(super) fn update_row(
    view: &mut CorrectionViewMut<'_>,
    j: u32,
    f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
) -> (bool, Result<(), CacheError>) {
    let RowUpdate { wrote, error } = view.update_interior_row(j, f);
    (wrote, error.map_or(Ok(()), |e| Err(refused_value(e))))
}

/// The store's error for a value a correction slot refused to take.
///
/// The caller's ([`CacheError::Correction`]) for a value no tile can hold,
/// [`CorrectionError::TooLarge`] or [`CorrectionError::NotFinite`]; else the store's own
/// ([`Invariant::CorrectionSlot`]), as the store writes only inside a tile it holds.
pub(super) fn refused_value(e: CorrectionError) -> CacheError {
    if matches!(e, CorrectionError::TooLarge | CorrectionError::NotFinite) {
        CacheError::Correction(e)
    } else {
        Invariant::CorrectionSlot(e).into()
    }
}

/// The half of [`TileStore::update_tiles`] that runs on the workers, for one tile.
///
/// Runs `f` over every interior vertex of the tile in `slot` (`side` vertices a side), row by
/// row in row-major order, setting what it returns, as [`TileEdit::update`] would. Stops at
/// the first vertex that fails.
fn update_interior(
    slot: &mut CorrectionSlot,
    side: u32,
    mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
) -> InteriorPass {
    let mut pass = InteriorPass {
        first_row: None,
        failed: None,
    };
    let mut view = match slot.view_mut() {
        Ok(view) => view,
        Err(e) => {
            pass.failed = Some((0, Invariant::CorrectionSlot(e).into()));
            return pass;
        }
    };
    for j in 1..side.saturating_sub(1) {
        let (wrote, row) = update_row(&mut view, j, &mut f);
        if wrote && pass.first_row.is_none() {
            pass.first_row = Some(j);
        }
        if let Err(e) = row {
            pass.failed = Some((j, e));
            break;
        }
    }
    pass
}

/// A tile [`TileEdit`] has touched: loaded and pinned, perhaps written.
#[derive(Debug, Clone, Copy)]
pub(super) struct Touched {
    /// The tile.
    key: TileKey,
    /// Its correction slot, pinned until the edit ends.
    id: SlotId<CorrectionSlot>,
    /// Whether a copy in it was written (it is dirty and on the feed).
    written: bool,
}

/// One tile's corrections being edited as part of the current op: what
/// [`TileStore::edit_tile`] hands its closure.
///
/// Coordinates are the tile's own, `0..side` on each axis. Setting a vertex on the tile's border
/// sets every copy of it, in the neighbouring tiles of this face and of the faces across the
/// border too, so twins never disagree. The tiles written stay pinned until the edit ends; then
/// each one's evaluations are dropped, once.
pub struct TileEdit<'a> {
    /// The store, borrowed for the edit.
    store: &'a mut TileStore,
    /// The tile edited.
    key: TileKey,
    /// The op every write is recorded with.
    op: OpId,
    /// The tiles holding copies touched so far, the edited tile first; empty once the edit is
    /// finished. A handful (a border vertex's copies lie in the tile's neighbours), so a linear
    /// search finds them.
    pub(super) touched: Vec<Touched>,
}

impl std::fmt::Debug for TileEdit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileEdit")
            .field("key", &self.key)
            .field("op", &self.op)
            .field("touched", &self.touched.len())
            .finish_non_exhaustive()
    }
}

/// Finishes an edit whose closure panicked, so its tiles don't stay pinned and the evaluations
/// of those it wrote don't stay cached. After a normal return the edit is finished already, and
/// this does nothing.
impl Drop for TileEdit<'_> {
    fn drop(&mut self) {
        // While unwinding there is no caller to report to, and the only error is a broken
        // invariant: every touched tile is unpinned either way.
        drop(self.finish());
    }
}

impl TileEdit<'_> {
    /// Ends the edit: unpins every touched tile and drops the evaluations that followed from
    /// the tiles written. Touched tiles are forgotten, so a second call does nothing.
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] if a touched slot's handle went stale (pinned slots are never
    /// freed, so this doesn't happen); every tile is unpinned and invalidated either way.
    fn finish(&mut self) -> Result<(), CacheError> {
        let touched = std::mem::take(&mut self.touched);
        let mut outcome = Ok(());
        for tile in &touched {
            keep_first(&mut outcome, self.store.corrections.unpin(tile.id));
        }
        self.store.invalidate_edited(
            touched
                .iter()
                .filter(|tile| tile.written)
                .map(|tile| tile.key),
        );
        outcome
    }

    /// The tile being edited.
    #[must_use]
    pub const fn key(&self) -> TileKey {
        self.key
    }

    /// The correction at `(i, j)` of the tile, as stored (interior values quantized).
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::OutOfTile`].
    pub fn get(&self, i: u32, j: u32) -> Result<Vec3, CacheError> {
        let side = self.key.side();
        let own = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get(own.id)
            .ok_or(Invariant::StaleHandle)?;
        let value = slot.view().map_err(Invariant::CorrectionSlot)?.get(i, j);
        value.ok_or(CacheError::Correction(CorrectionError::OutOfTile {
            i,
            j,
            side,
        }))
    }

    /// Sets the correction at `(i, j)` of the tile to `c`, in every copy of the vertex.
    ///
    /// An interior vertex has one copy and is written in place. A border vertex's copies are
    /// all loaded (and pinned) before any is written.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
    /// [`CorrectionError::OutOfTile`], before anything is written, or with
    /// [`CorrectionError::TooLarge`] for an interior value no tile's scale can hold (the tile is
    /// unchanged); [`CacheError::Store`],
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if a neighbouring tile holding a copy
    /// can't be loaded, also before anything of this vertex is written.
    pub fn set(&mut self, i: u32, j: u32, c: Vec3) -> Result<(), CacheError> {
        if !c.is_finite() {
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        let side = self.key.side();
        if i >= side || j >= side {
            return Err(CacheError::Correction(CorrectionError::OutOfTile {
                i,
                j,
                side,
            }));
        }
        let inside = 1..side - 1;
        if inside.contains(&i) && inside.contains(&j) {
            return self.write(0, (i, j), c);
        }
        let region = Region::of_tile(&self.key);
        let copies = self.store.mesh.topology().copies(
            self.key.face(),
            self.key.level(),
            region.x0() + i,
            region.y0() + j,
        );
        self.set_copies(&copies, c)
    }

    /// Sets every one of `copies`, the copies of one vertex in [`geom::Topology::copies`]'s order,
    /// to `c` (finite): each one's tile is loaded and pinned first, in that order, then each
    /// copy is written, as [`TileEdit::set`] does for a border vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`] for a neighbouring tile that can't be loaded, before anything of
    /// this vertex is written.
    fn set_copies(&mut self, copies: &[VertexCopy], c: Vec3) -> Result<(), CacheError> {
        let level = self.key.level();
        let mut at = [0; MAX_COPIES];
        for (copy, place) in copies.iter().zip(&mut at) {
            *place = self.touch(copy.key(level))?;
        }
        for (copy, &n) in copies.iter().zip(&at) {
            self.write(n, (copy.i, copy.j), c)?;
        }
        Ok(())
    }

    /// [`TileEdit::set`] for border vertex `(i, j)`, its copies read from `border`, the tile's
    /// [`TileBorder`]: the same writes in the same order, without finding the copies again for
    /// every vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn set_border(
        &mut self,
        border: &TileBorder,
        i: u32,
        j: u32,
        c: Vec3,
    ) -> Result<(), CacheError> {
        let Some(copies) = border.copies(i, j) else {
            return self.set(i, j, c);
        };
        if !c.is_finite() {
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        self.set_copies(&copies, c)
    }

    /// Rewrites the tile's corrections vertex by vertex: `f` sees every vertex in row-major
    /// order, with its coordinates and its correction as stored, and where it returns a new
    /// value, that is set as [`TileEdit::set`] sets it, before `f` sees the next vertex.
    ///
    /// The same as [`TileEdit::get`] then [`TileEdit::set`] for each vertex in that order, and
    /// so the same bits: order matters, as a value that outgrows a plane's scale re-quantizes
    /// that plane of the interior. It is cheaper, as each run of interior vertices of a row (which
    /// have no copies in other tiles) is read and written through one borrowed view instead of
    /// a lookup per call; border vertices go through [`TileEdit::set`], copies and all.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`], for the first vertex that fails; the vertices before it stay
    /// written.
    pub fn update(
        &mut self,
        mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let side = self.key.side();
        let last = side - 1;
        for j in 0..side {
            if j == 0 || j == last {
                for i in 0..side {
                    self.update_one(i, j, &mut f)?;
                }
            } else {
                self.update_one(0, j, &mut f)?;
                self.update_interior_row(j, &mut f)?;
                self.update_one(last, j, &mut f)?;
            }
        }
        Ok(())
    }

    /// [`TileEdit::update`] for vertex `(i, j)` alone, through [`TileEdit::get`] and
    /// [`TileEdit::set`].
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn update_one(
        &mut self,
        i: u32,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        match f(i, j, self.get(i, j)?) {
            Some(c) => self.set(i, j, c),
            None => Ok(()),
        }
    }

    /// [`TileEdit::update`] for the interior vertices of row `j` (`1..side - 1`, for a row
    /// inside the tile), through one view of the tile's slot. The tile is marked written after
    /// the row if anything in it was, before any other tile can be touched: as
    /// [`TileEdit::write`] marks it at its first write.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`] or
    /// [`CorrectionError::TooLarge`] for a value `f` returns that no tile can hold (nothing is
    /// written for it, the row's earlier writes stay);
    /// [`CacheError::Internal`] for a stale handle or a broken slot.
    fn update_interior_row(
        &mut self,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let tile = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get_mut(tile.id)
            .ok_or(Invariant::StaleHandle)?;
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        let (wrote, row) = update_row(&mut view, j, f);
        if wrote {
            self.mark_own_written()?;
        }
        row
    }

    /// The other half of [`TileStore::update_tiles`] for this tile, whose interior `pass`
    /// wrote: runs `f` over the border vertices in row-major order, setting what it returns
    /// with [`TileEdit::set`], and marks the tile written where its first interior write falls
    /// in that order (if no border write marked it first). Stops where the interior pass
    /// failed, as [`TileEdit::update`] would have.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`] for a border vertex; else the interior pass's error. Whatever
    /// fails, and wherever in row-major order, a tile whose interior the pass wrote is marked
    /// written: its slot holds the pass's writes, before the failing vertex or not.
    fn update_border(
        &mut self,
        pass: InteriorPass,
        f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let wrote = pass.first_row.is_some();
        let mut outcome = self.border_writes(pass, f);
        if outcome.is_err() && wrote {
            keep_first(&mut outcome, self.mark_own_written());
        }
        outcome
    }

    /// [`TileEdit::update_border`]'s writes, in row-major order, stopping at the first that
    /// fails; the tile is marked written at its first interior row only if no write before it
    /// fails.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::update_border`].
    fn border_writes(
        &mut self,
        pass: InteriorPass,
        mut f: impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let side = self.key.side();
        let last = side - 1;
        let InteriorPass { first_row, failed } = pass;
        let (failed_row, mut failure) = match failed {
            Some((row, e)) => (Some(row), Some(e)),
            None => (None, None),
        };
        if failed_row == Some(0)
            && let Some(e) = failure.take()
        {
            return Err(e);
        }
        let border = match self.store.tile_border(&self.key) {
            Some(copies) => Some((self.border_values()?, copies)),
            None => None,
        };
        for j in 0..side {
            if j == 0 || j == last {
                for i in 0..side {
                    self.update_border_one(border.as_ref(), i, j, &mut f)?;
                }
                continue;
            }
            self.update_border_one(border.as_ref(), 0, j, &mut f)?;
            if first_row == Some(j) {
                self.mark_own_written()?;
            }
            if failed_row == Some(j)
                && let Some(e) = failure.take()
            {
                return Err(e);
            }
            self.update_border_one(border.as_ref(), last, j, &mut f)?;
        }
        Ok(())
    }

    /// The corrections of the tile's border vertices as they are now, read through one view
    /// (see [`BorderValues`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle or a broken slot.
    fn border_values(&self) -> Result<BorderValues, CacheError> {
        let side = self.key.side();
        let own = self.touched.first().ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get(own.id)
            .ok_or(Invariant::StaleHandle)?;
        let view = slot.view().map_err(Invariant::CorrectionSlot)?;
        let values = border_order(side)
            .map(|(i, j)| view.get(i, j).ok_or(Invariant::StaleHandle))
            .collect::<Result<_, _>>()?;
        Ok(BorderValues { side, values })
    }

    /// [`TileEdit::update_one`] for border vertex `(i, j)`; where the tile has a [`TileBorder`],
    /// with its correction from the [`BorderValues`] read before the tile's border writes and
    /// its copies from the border.
    ///
    /// The values read before are the values [`TileEdit::get`] would read now: the writes since
    /// went to other vertices' copies, and a tile with a [`TileBorder`] holds no vertex twice,
    /// so none of them landed on this vertex.
    ///
    /// # Errors
    ///
    /// As [`TileEdit::set`].
    fn update_border_one(
        &mut self,
        border: Option<&(BorderValues, TileBorder)>,
        i: u32,
        j: u32,
        f: &mut impl FnMut(u32, u32, Vec3) -> Option<Vec3>,
    ) -> Result<(), CacheError> {
        let Some((values, copies)) = border else {
            return self.update_one(i, j, f);
        };
        let current = match values.get(i, j) {
            Some(c) => c,
            None => self.get(i, j)?,
        };
        match f(i, j, current) {
            Some(c) => self.set_border(copies, i, j, c),
            None => Ok(()),
        }
    }

    /// Marks the edited tile written, if no write has yet: dirty, on the change feed with the
    /// edit's op, and its evaluations dropped when the edit ends (see [`TileEdit::write`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::Internal`] for a stale handle.
    fn mark_own_written(&mut self) -> Result<(), CacheError> {
        let tile = self.touched.first_mut().ok_or(Invariant::StaleHandle)?;
        if !tile.written {
            tile.written = true;
            let (key, id) = (tile.key, tile.id);
            self.store.mark_written(key, id, self.op)?;
        }
        Ok(())
    }

    /// The index in `touched` of `key`'s tile, loaded and pinned if it is new.
    ///
    /// # Errors
    ///
    /// As [`TileStore::load_correction`]; [`CacheError::Internal`] for a stale handle.
    fn touch(&mut self, key: TileKey) -> Result<usize, CacheError> {
        if let Some(n) = self.touched.iter().position(|tile| tile.key == key) {
            return Ok(n);
        }
        let id = self.store.load_correction(key)?;
        self.store.corrections.pin(id)?;
        self.touched.push(Touched {
            key,
            id,
            written: false,
        });
        Ok(self.touched.len() - 1)
    }

    /// Writes `c` at `(i, j)` of touched tile `n`, marking the tile written at its first write.
    ///
    /// # Errors
    ///
    /// As [`TileStore::write_correction`].
    fn write(&mut self, n: usize, (i, j): (u32, u32), c: Vec3) -> Result<(), CacheError> {
        let tile = self.touched.get_mut(n).ok_or(Invariant::StaleHandle)?;
        let slot = self
            .store
            .corrections
            .get_mut(tile.id)
            .ok_or(Invariant::StaleHandle)?;
        let mut view = slot.view_mut().map_err(Invariant::CorrectionSlot)?;
        view.set(i, j, c).map_err(refused_value)?;
        if !tile.written {
            tile.written = true;
            self.store.mark_written(tile.key, tile.id, self.op)?;
        }
        Ok(())
    }
}

/// An op reserved for a stroke begun while the one before it still lands
/// ([`TileStore::reserve_op`]): ordered after it, but not current, so no edit is made as it.
///
/// It is the only way to name a reserved op to the store, and it can't be copied: it is given
/// back to begin the op ([`TileStore::begin_reserved_op`]) or let it go
/// ([`TileStore::drop_reserved_op`]), so an op is begun at most once, and an op begun is never
/// read or given shapes as a reserved one. Edits never take an op, reserved or not: they are made
/// as the current op (see [`TileStore::begin_op`]).
///
/// One dropped otherwise would stay reserved and hold every later op back, so a debug build
/// panics as it is dropped (but while a panic unwinds). One whose op went otherwise (with an op
/// before it, or with every reserved op, [`TileStore::drop_every_reserved_op`]) is given back
/// all the same, and lets nothing more go; one whose store is gone (a close, a takeover) is
/// abandoned ([`ReservedOp::abandon_with_store_gone`]).
#[derive(Debug)]
#[must_use = "an op reserved and let go stays reserved, and holds every later op back"]
pub struct ReservedOp {
    /// The op.
    op: OpId,
    /// The store it was reserved on.
    store: super::StoreId,
    /// Whether it was given back to the store, which a debug build checks as it is dropped.
    given_back: bool,
}

impl ReservedOp {
    /// The op reserved: the one the stroke's edits, journal and record carry.
    #[must_use]
    pub const fn op(&self) -> OpId {
        self.op
    }

    /// The store it was reserved on.
    pub(super) const fn store(&self) -> super::StoreId {
        self.store
    }

    /// Lets the op go with its store, which is gone (closed, or taken over): nothing is left to
    /// give it back to, and nothing more is reserved. For a store still open, drop it there
    /// instead ([`TileStore::drop_reserved_op`]), or it stays reserved.
    pub fn abandon_with_store_gone(self) {
        self.give_back();
    }

    /// Gives the op back to the store: it has begun or gone.
    fn give_back(mut self) {
        self.given_back = true;
    }
}

/// Reports, in a debug build, a reserved op dropped without being given back to the store.
impl Drop for ReservedOp {
    fn drop(&mut self) {
        debug_assert!(
            self.given_back || std::thread::panicking(),
            "an op reserved and let go stays reserved, and holds every later op back: {:?}",
            self.op
        );
    }
}

/// A reserved op that couldn't begin ([`TileStore::begin_reserved_op`]): the op, given back, and
/// why.
///
/// It holds the op's token, so it is no [`std::error::Error`]: an error passed up with `?` would
/// take the token with it and drop it far from here, the op reserved for good. It is to be taken
/// apart ([`NotBegun::into_parts`], [`NotBegun::into_reserved`]), and the op begun later or
/// dropped. So `?` doesn't compile:
///
/// ```compile_fail
/// # fn begin(store: &mut store::TileStore) -> Result<(), Box<dyn std::error::Error>> {
/// let reserved = store.reserve_op()?;
/// store.begin_reserved_op(reserved)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[must_use = "it holds the reserved op, which is to be begun later or dropped"]
pub struct NotBegun {
    /// The op, still reserved (or never this store's).
    reserved: ReservedOp,
    /// Why it couldn't begin.
    error: CacheError,
}

impl NotBegun {
    /// Why the op couldn't begin.
    #[must_use]
    pub const fn error(&self) -> &CacheError {
        &self.error
    }

    /// The op, still reserved.
    #[must_use]
    pub const fn reserved(&self) -> &ReservedOp {
        &self.reserved
    }

    /// The op back, to begin later or drop.
    pub fn into_reserved(self) -> ReservedOp {
        self.reserved
    }

    /// The op back, and why it couldn't begin.
    pub fn into_parts(self) -> (ReservedOp, CacheError) {
        (self.reserved, self.error)
    }
}

impl TileStore {
    /// Starts a new op, the site's next sequence number, and returns it: every edit from now
    /// on, [`TileStore::set_correction`] and [`TileStore::edit_tile`] alike, is part of it,
    /// until the next call. A stroke begins one and keeps it across all of its dabs, so the
    /// last stroke's pen-down shapes go ([`TileStore::release_pen_down`]): the new stroke's
    /// surface at pen-down is the store's as it is now.
    ///
    /// Ops are issued here only, never passed in by a caller, so they only grow. An edit with
    /// no op begun (the first after the store opens, or after a rebase) begins one itself.
    ///
    /// There is no way to make an edit as an op of the caller's choosing:
    ///
    /// ```compile_fail
    /// # fn edit(store: &mut store::TileStore, key: geom::TileKey, op: store::OpId) {
    /// store.edit_tile(key, op, |edit| edit.set(1, 1, geom::Vec3::ZERO));
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`CacheError::OpsExhausted`] past the site's last sequence number;
    /// [`CacheError::OpReserved`] while an op is reserved ([`TileStore::reserve_op`]), which begins
    /// next ([`TileStore::begin_reserved_op`]); [`CacheError::Internal`] with
    /// [`Invariant::ReadingReserved`] while a reserved op's pen-down shapes are read
    /// ([`TileStore::read_pen_down_of`]). The current op is kept then.
    pub fn begin_op(&mut self) -> Result<OpId, CacheError> {
        self.refuse_while_reading()?;
        if let Some(reserved) = self.reserved.oldest() {
            return Err(CacheError::OpReserved(reserved));
        }
        let op = self.start_op()?;
        self.release_pen_down();
        Ok(op)
    }

    /// Reserves the site's next op for a stroke begun while the current op's still lands, and
    /// returns it: ordered after the current op and every op reserved before it, but not current.
    /// No edit is made as it until it begins ([`TileStore::begin_reserved_op`]), once the ops
    /// before it have ended; meanwhile it may be given pen-down shapes of its own, the futures of
    /// the tiles the strokes before it still change ([`TileStore::install_pen_down_of`]), and its
    /// reads go to them ([`TileStore::read_pen_down_of`]).
    ///
    /// While an op is reserved, no other op may begin before it: a new one
    /// ([`TileStore::begin_op`]), a resumed one, or a reserved one after it; and once the current
    /// op has ended no edit may be made, which would begin one. So edits keep to op order, one op
    /// at a time.
    ///
    /// Give the op back to the store: begin it, or drop it ([`TileStore::drop_reserved_op`]). One
    /// let go otherwise stays reserved, and holds every later op back (a debug build panics).
    ///
    /// # Errors
    ///
    /// [`CacheError::OpenStrokesFull`] if the open strokes' pen-down data holds its share of the
    /// budget already ([`TileStore::open_strokes_share`]): the stroke waits for those before to
    /// land, and begins its op then ([`TileStore::begin_op`]). [`CacheError::OpsExhausted`] past
    /// the site's last sequence number. Nothing is reserved then.
    pub fn reserve_op(&mut self) -> Result<ReservedOp, CacheError> {
        let (held, share) = (self.pen_down_bytes(), self.open_strokes_share());
        if held >= share {
            return Err(CacheError::OpenStrokesFull { held, share });
        }
        let op = self.next_op.ok_or(CacheError::OpsExhausted)?;
        self.next_op = op.next();
        self.handed_out = Some(op);
        self.reserved.insert(op);
        Ok(ReservedOp {
            op,
            store: self.id(),
            given_back: false,
        })
    }

    /// Makes `reserved`, the oldest op reserved, the current op: every edit from now on is made
    /// as it, as after [`TileStore::begin_op`]. Its pen-down shapes, the futures installed for it,
    /// become the current stroke's (any left of the op before are let go), and its stroke keeps
    /// them as any stroke does from now on.
    ///
    /// Only once the op before has ended, or with no op current: so ops edit one at a time, in
    /// order.
    ///
    /// # Errors
    ///
    /// [`NotBegun`], with `reserved` given back and the store unchanged, for
    /// [`CacheError::NotReserved`] if it isn't reserved on this store,
    /// [`CacheError::OpReserved`] if an older op is reserved (it begins first),
    /// [`CacheError::OpBegun`] if an op is current, or [`CacheError::Internal`] with
    /// [`Invariant::ReadingReserved`] while a reserved op's pen-down shapes are read
    /// ([`TileStore::read_pen_down_of`]).
    pub fn begin_reserved_op(&mut self, reserved: ReservedOp) -> Result<OpId, NotBegun> {
        let op = reserved.op;
        let refused = if let Err(error) = self.refuse_while_reading() {
            Some(error)
        } else if reserved.store != self.id() || !self.reserved.holds(op) {
            Some(CacheError::NotReserved(op))
        } else if let Some(oldest) = self.reserved.oldest().filter(|&oldest| oldest != op) {
            Some(CacheError::OpReserved(oldest))
        } else {
            self.op.map(CacheError::OpBegun)
        };
        if let Some(error) = refused {
            return Err(NotBegun { reserved, error });
        }
        let Some(shapes) = self.reserved.take(op) else {
            let error = CacheError::NotReserved(op);
            return Err(NotBegun { reserved, error });
        };
        reserved.give_back();
        self.pen_down = shapes;
        self.withhold_pen_down(0);
        self.op = Some(op);
        Ok(op)
    }

    /// Lets reserved op `reserved` go, and with it every op reserved after it, whose futures read
    /// its dabs (see [`TileStore::drop_reserved_ops_from`]); returns the ops let go, in op order.
    /// For a stroke that ends before its op begins, as when a stroke before it fails to land its
    /// dabs. Nothing for an op no longer reserved (let go already, with an op before it): the
    /// token is given back all the same. A token of another store is a bug (a debug assertion
    /// says so), as that store's op would stay reserved; one whose store is gone is abandoned
    /// instead ([`ReservedOp::abandon_with_store_gone`]).
    ///
    /// # Panics
    ///
    /// While a reserved op's pen-down shapes are read ([`TileStore::read_pen_down_of`]).
    pub fn drop_reserved_op(&mut self, reserved: ReservedOp) -> Vec<OpId> {
        self.assert_not_reading();
        let (op, store) = (reserved.op, reserved.store);
        reserved.give_back();
        debug_assert!(
            store == self.id(),
            "a reserved op is dropped on the store it was reserved on: {op:?}"
        );
        if store != self.id() {
            return Vec::new();
        }
        self.drop_reserved_ops_from(op)
    }

    /// Lets reserved op `op` go, if it is reserved, and every op reserved after it: their pen-down
    /// shapes go, their bytes given back to the pools, and none of them begins (their ids aren't
    /// handed out again: ops only grow). A later op's futures are the earlier strokes' dabs
    /// applied, `op`'s among them, so they can't stand without it. Their strokes' journals, if
    /// open, close as an ended stroke's do, every dab counted landed: none of them ever lands, and
    /// their last batches say so (see `crate::tiles::journal`). Returns the ops let go, in op
    /// order; their tokens are no longer reserved ([`CacheError::NotReserved`]), and are to be
    /// given back with [`TileStore::drop_reserved_op`], which then lets nothing more go.
    ///
    /// # Panics
    ///
    /// While a reserved op's pen-down shapes are read ([`TileStore::read_pen_down_of`]).
    pub fn drop_reserved_ops_from(&mut self, op: OpId) -> Vec<OpId> {
        self.assert_not_reading();
        if !self.reserved.holds(op) {
            return Vec::new();
        }
        let dropped = self.reserved.drop_from(op);
        self.end_dropped_journals(&dropped);
        self.withhold_pen_down(0);
        dropped
    }

    /// The oldest op reserved ([`TileStore::reserve_op`]), which begins next; `None` with none
    /// reserved.
    pub(crate) fn oldest_reserved_op(&self) -> Option<OpId> {
        self.reserved.oldest()
    }

    /// Whether `reserved` is an op reserved on this store, not yet begun or let go.
    pub(crate) fn holds_reserved(&self, reserved: &ReservedOp) -> bool {
        reserved.store == self.id() && self.reserved.holds(reserved.op)
    }

    /// Lets every reserved op go, as [`TileStore::drop_reserved_ops_from`] the oldest: for an
    /// error that ends the strokes landing, a close or a takeover. Returns the ops let go, in op
    /// order.
    ///
    /// # Panics
    ///
    /// While a reserved op's pen-down shapes are read ([`TileStore::read_pen_down_of`]).
    pub fn drop_every_reserved_op(&mut self) -> Vec<OpId> {
        match self.reserved.oldest() {
            Some(oldest) => self.drop_reserved_ops_from(oldest),
            None => Vec::new(),
        }
    }

    /// Ends the current op: its stroke has ended, its pen-down shapes go
    /// ([`TileStore::release_pen_down`]), and no edit is made as it again. The next edit begins a
    /// new op as after a rebase, unless an op is reserved: then that op begins next
    /// ([`TileStore::begin_reserved_op`]), and edits wait for it. Nothing more with no op
    /// current.
    ///
    /// For the [`crate::Project`] call that ends a stroke, once the stroke's journal is closed: a
    /// journal still open for the op would no longer be carried by a roll-over, which carries the
    /// current op's only (a debug assertion says so).
    ///
    /// # Panics
    ///
    /// While a reserved op's pen-down shapes are read ([`TileStore::read_pen_down_of`]).
    pub(crate) fn end_op(&mut self) {
        self.assert_not_reading();
        debug_assert!(
            self.op.is_none_or(|op| !self.journal_open_for(op)),
            "a stroke's op ends once its journal is closed"
        );
        self.op = None;
        self.release_pen_down();
    }

    /// Makes `op`, a stroke's op the working file holds edits of, the op edits are made as again,
    /// as [`TileStore::begin_op`] would have left it: the edits that land the rest of a stroke
    /// restored after a crash are its own (see [`crate::Project::resume_stroke`]). The op
    /// generator is moved past `op` if it isn't already, so no op is handed out twice: it starts
    /// past every op of the files ([`ProjectFile::last_op`](crate::file::ProjectFile::last_op)),
    /// a stroke that landed nothing included, and this keeps it so whatever `op` the caller
    /// gives. The pen-down shapes stay: the stroke's own, installed first
    /// ([`TileStore::install_pen_down`]).
    ///
    /// Only on a store no op has been handed out on since it opened
    /// ([`TileStore::op_handed_out`]): begun, edited as or reserved, whether it ended or was let
    /// go since. The stroke's edits must be the last before its own.
    ///
    /// # Errors
    ///
    /// [`CacheError::OpReserved`] while an op is reserved ([`TileStore::reserve_op`]);
    /// [`CacheError::OpBegun`] if an op was handed out since the store opened;
    /// [`CacheError::OpsExhausted`] if `op` is its site's last. The store is unchanged then.
    pub(crate) fn resume_op(&mut self, op: OpId) -> Result<(), CacheError> {
        if let Some(reserved) = self.reserved.oldest() {
            return Err(CacheError::OpReserved(reserved));
        }
        if let Some(handed) = self.op_handed_out() {
            return Err(CacheError::OpBegun(handed));
        }
        self.skip_ops_past([Some(op)])?;
        self.op = Some(op);
        self.handed_out = Some(op);
        Ok(())
    }

    /// The op current, if one is, else the last op handed out since the store opened (begun,
    /// edited as, reserved or resumed), whether it has ended or been let go since; `None` while
    /// none has been. A stroke a restore found resumes only while this is `None`.
    pub(crate) const fn op_handed_out(&self) -> Option<OpId> {
        match self.op {
            Some(op) => Some(op),
            None => self.handed_out,
        }
    }

    /// Starts the site's next op, as [`TileStore::begin_op`] does, the pen-down shapes kept: for
    /// an edit that finds no op (after a rebase moved the store's later edits to a new op, in
    /// the middle of a stroke, or after the last op ended). While an op is reserved, such an edit
    /// is refused instead ([`TileStore::edit_op`]): the reserved op's edits come first.
    ///
    /// # Errors
    ///
    /// As [`TileStore::begin_op`].
    fn start_op(&mut self) -> Result<OpId, CacheError> {
        let op = self.next_op.ok_or(CacheError::OpsExhausted)?;
        self.next_op = op.next();
        self.op = Some(op);
        self.handed_out = Some(op);
        Ok(op)
    }

    /// Moves the op generator past every op of `heads` of the store's site, if it isn't already:
    /// for heads of files the store doesn't read, but whose edits its own must be newer than
    /// (other recovery files of the project, see [`crate::Project::open`]).
    ///
    /// # Errors
    ///
    /// [`CacheError::OpsExhausted`] if one of them is the site's last op id; the generator is
    /// unchanged then.
    pub(crate) fn skip_ops_past(
        &mut self,
        heads: impl IntoIterator<Item = Option<OpId>>,
    ) -> Result<(), CacheError> {
        let Some(next) = self.next_op else {
            return Ok(());
        };
        let after = first_op(next.site(), heads)?;
        self.next_op = Some(next.max(after));
        Ok(())
    }

    /// The op edits are made as now: the one [`TileStore::begin_op`] last started; `None` if
    /// none is (the next edit begins one).
    #[must_use]
    pub const fn current_op(&self) -> Option<OpId> {
        self.op
    }

    /// How many tiles' stroke state the budget set aside for a brush stroke
    /// ([`StoreConfig::stroke_tiles`](super::config::StoreConfig::stroke_tiles)): the most tiles
    /// one stroke over this store may keep the value it builds for each vertex's three correction
    /// components for (the correction as it first found it plus every delta since). Its bytes are
    /// already
    /// out of the pools' share, so a stroke that allocates this much, and no more, stays within
    /// the budget.
    #[must_use]
    pub const fn stroke_tiles(&self) -> usize {
        self.stroke_tiles
    }

    /// How many times a correction slot's values have changed since the store opened: every
    /// write ([`TileStore::mark_written`]) and every lost tile's write undone, by
    /// [`TileStore::rebase`], [`TileStore::roll_over`] or [`TileStore::lose_left_out`]
    /// ([`TileStore::drop_lost_slot`], [`TileStore::reload_lost`]). An ordinary page-in reads
    /// what the files hold and changes nothing. While it stays the same, every tile evaluates
    /// to the same bits: what [`WaveBounds`](super::WaveBounds) checks before its bounds are
    /// used.
    pub(crate) const fn edits(&self) -> u64 {
        self.edits
    }

    /// Counts a change of a correction slot's values ([`TileStore::edits`]).
    ///
    /// Called where the values change, not where evaluations are dropped: an invalidation may
    /// skip a tile nothing cached follows from, and a path that drops evaluations some other
    /// way must still be counted.
    pub(super) const fn corrections_changed(&mut self) {
        self.edits = self.edits.wrapping_add(1);
    }

    /// The op for an edit: the current one, or a new one if none is current.
    ///
    /// # Errors
    ///
    /// As [`TileStore::begin_op`]: [`CacheError::Internal`] with [`Invariant::ReadingReserved`]
    /// while a reserved op's pen-down shapes are read; with no op current,
    /// [`CacheError::OpReserved`] while an op is reserved, whose edits come first.
    pub(super) fn edit_op(&mut self) -> Result<OpId, CacheError> {
        self.refuse_while_reading()?;
        match self.op {
            Some(op) => Ok(op),
            None => match self.reserved.oldest() {
                Some(reserved) => Err(CacheError::OpReserved(reserved)),
                None => self.start_op(),
            },
        }
    }

    /// The copies of every border vertex of tile `key`, found once
    /// ([`geom::Topology::tile_border`]), for an edit's border pass.
    ///
    /// `None` for a tile holding one of its border vertices twice, whose edits then find each
    /// vertex's copies one by one. No mesh [`BaseMesh`](geom::BaseMesh) accepts has such a tile:
    /// only a face glued to itself could hold a vertex twice, and a quad that repeats no vertex
    /// (which [`BaseMesh`](geom::BaseMesh) checks) can't share an edge or a corner with itself.
    /// The fallback is kept rather than made an error, as it is the definition the fast path
    /// must match; the tests reach it by forcing it.
    fn tile_border(&self, key: &TileKey) -> Option<TileBorder> {
        #[cfg(test)]
        if self.hooks.without_tile_borders {
            return None;
        }
        self.mesh.topology().tile_border(key)
    }

    /// Bytes the pools hold: whole chunks, never more than the budget.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.budget.used()
    }

    /// `key`'s correction tile: resident, else loaded from the working file, else from the base,
    /// else all zero (an absent tile is zero).
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::Store`] if the record can't be read;
    /// [`CacheError::Full`] or [`CacheError::WriteBack`] if no slot can be had.
    pub fn correction(&mut self, key: TileKey) -> Result<CorrectionView<'_>, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let id = self.load_correction(key)?;
        let slot = self.corrections.get(id).ok_or(Invariant::StaleHandle)?;
        Ok(slot.view().map_err(Invariant::CorrectionSlot)?)
    }

    /// Sets the correction at face-grid vertex `(i, j)` of `face` at `level` to `c`, as part of
    /// the current op (see [`TileStore::begin_op`]): in every copy of the vertex (one per tile
    /// holding it, in every face sharing it).
    ///
    /// Each copy's slot becomes dirty, and its key goes to the change feed with the op (once per
    /// op until a consumer drains it, see [`ChangeFeed::push`](crate::feed::ChangeFeed::push)).
    /// Every cached evaluation that followed from those tiles is dropped (pinned ones are
    /// recomputed in place on their next [`TileStore::evaluate`]); a tile nothing has been
    /// computed from since it was last dropped is skipped, so repeated edits of one tile
    /// between evaluations walk its dependencies once. To edit many vertices of one tile,
    /// [`TileStore::edit_tile`] does the bookkeeping once for all of them.
    ///
    /// The change joins the open epoch of the stale-tile feed ([`TileStore::subscribe_stale`]): the
    /// caller closes it with [`TileStore::close_epoch`] once its operation is done, as a brush
    /// step does once per step.
    ///
    /// # Errors
    ///
    /// [`CacheError::Correction`] with [`CorrectionError::NotFinite`],
    /// [`CorrectionError::LevelZero`], [`CorrectionError::UnknownFace`] or
    /// [`CorrectionError::OutOfTile`], before anything is written, or with
    /// [`CorrectionError::TooLarge`] for an interior value no tile's scale can hold (an interior
    /// vertex has one copy, so nothing is written then either);
    /// [`CacheError::OpsExhausted`] if no op is current and none can be begun;
    /// [`CacheError::Store`], [`CacheError::Full`] or [`CacheError::WriteBack`] if a copy's tile
    /// can't be loaded, also before anything is written. Should writing a copy fail all the same
    /// ([`CacheError::Internal`]: a broken slot), the copies written before it are kept, and
    /// their evaluations dropped, before the error is returned.
    pub fn set_correction(
        &mut self,
        face: FaceId,
        level: Level,
        i: u32,
        j: u32,
        c: Vec3,
    ) -> Result<(), CacheError> {
        self.sync_pins()?;
        self.poll()?;
        if !c.is_finite() {
            return Err(CacheError::Correction(CorrectionError::NotFinite));
        }
        if level == Level::ZERO {
            return Err(CacheError::Correction(CorrectionError::LevelZero));
        }
        let topology = self.mesh.topology();
        if !topology.contains(face) {
            return Err(CacheError::Correction(CorrectionError::UnknownFace(face)));
        }
        let copies = topology.copies(face, level, i, j);
        if copies.is_empty() {
            let side = level.face_quads() + 1;
            return Err(CacheError::Correction(CorrectionError::OutOfTile {
                i,
                j,
                side,
            }));
        }
        let op = self.edit_op()?;
        // Every copy is loaded and pinned before any is written, so a failed load leaves the
        // twins agreeing.
        let mut held = [None; MAX_COPIES];
        let loaded = copies.iter().zip(&mut held).try_for_each(|(copy, place)| {
            let id = self.load_correction(copy.key(level))?;
            self.corrections.pin(id)?;
            *place = Some(id);
            Ok::<_, CacheError>(())
        });
        let mut written = 0;
        let outcome = loaded.and_then(|()| {
            for (copy, &id) in copies.iter().zip(held.iter().flatten()) {
                self.write_correction(copy.key(level), id, (copy.i, copy.j), c, op)?;
                written += 1;
            }
            Ok(())
        });
        let released = held
            .iter()
            .flatten()
            .try_for_each(|&id| self.corrections.unpin(id));
        // The copies written are invalidated even if a later one failed.
        self.invalidate_edited(copies.iter().take(written).map(|copy| copy.key(level)));
        outcome?;
        released?;
        Ok(())
    }

    /// Edits `key`'s corrections as part of the current op (see [`TileStore::begin_op`]): `f`
    /// sets vertices through a [`TileEdit`], and its result is returned.
    ///
    /// This is [`TileStore::set_correction`] for many vertices of one tile, with the bookkeeping
    /// done once per tile instead of once per vertex: each tile written (the edited one, and the
    /// neighbours holding copies of its border vertices) is loaded and pinned when first
    /// touched, marked dirty and put on the change feed at its first write, and its evaluations
    /// are dropped once, when `f` returns. Errors of single writes go to `f`.
    ///
    /// Should `f` panic, the same happens as the panic unwinds through this call: the touched
    /// tiles are unpinned, and the evaluations of those written are dropped. The writes made
    /// before the panic stay, dirty and on the change feed.
    ///
    /// The change joins the open epoch of the stale-tile feed ([`TileStore::subscribe_stale`]): the
    /// caller closes it with [`TileStore::close_epoch`] once its operation is done, as a brush
    /// step does once per step.
    ///
    /// # Errors
    ///
    /// [`CacheError::NotATile`]; [`CacheError::OpsExhausted`] if no op is current and none can
    /// be begun; [`CacheError::Store`], [`CacheError::Full`] or [`CacheError::WriteBack`] if the
    /// tile's corrections can't be loaded (nothing is written then); [`CacheError::Internal`] if
    /// a slot handle went stale.
    pub fn edit_tile<R>(
        &mut self,
        key: TileKey,
        f: impl FnOnce(&mut TileEdit<'_>) -> R,
    ) -> Result<R, CacheError> {
        self.sync_pins()?;
        self.poll()?;
        self.check_key(&key)?;
        let op = self.edit_op()?;
        let id = self.load_correction(key)?;
        self.corrections.pin(id)?;
        let mut edit = TileEdit {
            store: self,
            key,
            op,
            touched: vec![Touched {
                key,
                id,
                written: false,
            }],
        };
        // Should `f` panic, the edit's drop finishes it all the same.
        let result = f(&mut edit);
        edit.finish()?;
        Ok(result)
    }

    /// Rewrites the corrections of each tile of `keys` in turn, as part of the current op: the
    /// same as [`TileStore::edit_tile`] with [`TileEdit::update`] for each tile in order, with
    /// `f(n, i, j, c)` for vertex `(i, j)` of `keys[n]`, and the same bits; faster, as tiles
    /// the caller holds pinned (with [`TileStore::pin`]) have their interiors written on the
    /// store's workers.
    ///
    /// That is sound because a tile's interior vertices have no copies in other tiles, and its
    /// quantization scale covers its interior alone: what an interior write reads and changes
    /// is the tile's own, whichever tile is edited first. So each run of consecutive pinned
    /// tiles (no tile twice) has its interiors written in parallel, each tile's in row-major
    /// order (a value that outgrows a plane's scale re-quantizes that plane, so the order within
    /// a tile matters); then, tile by tile in order, its border vertices are set with their
    /// copies and the tile is marked written where its first write falls in row-major order,
    /// as one tile at a time would. Other tiles are edited one at a time.
    ///
    /// What callers may rely on of the calls to `f`:
    ///
    /// - `f` is called exactly once per vertex of each entry of `keys` (at most once, where an edit
    ///   fails part way), with the vertex's correction as stored just before its write;
    /// - calls run on several threads, one tile's interior all on one thread.
    ///
    /// So `f` may keep state per vertex that only that vertex's call touches (the value a brush
    /// stroke builds for each of the vertex's three components, say), but must give the same result
    /// whatever the thread schedule. A
    /// change here that called `f` twice for a vertex would break such callers.
    ///
    /// The change joins the open epoch of the stale-tile feed ([`TileStore::subscribe_stale`]): the
    /// caller closes it with [`TileStore::close_epoch`] once its operation is done, as a brush
    /// step does once per step.
    ///
    /// # Errors
    ///
    /// As [`TileStore::edit_tile`] and [`TileEdit::update`], for the first tile that fails, in
    /// order: the tiles before it are edited. A tile edited one at a time is left as
    /// [`TileEdit::update`] leaves it; one of a run may have its whole interior pass applied
    /// (every interior vertex up to the one the pass failed at, if any), even where a border
    /// vertex earlier in row-major order failed. Tiles after it in its run may have had their
    /// interiors written too. Every tile of the run whose interior was written is marked
    /// written (dirty, on the change feed with the op) and its evaluations dropped, as for a
    /// whole edit, each of them even if marking another fails; the first error is returned.
    pub fn update_tiles(
        &mut self,
        keys: &[TileKey],
        f: impl Fn(usize, u32, u32, Vec3) -> Option<Vec3> + Sync,
    ) -> Result<(), CacheError> {
        // Every write before this one is settled, so the bump after it (see
        // `TileStore::bump_display_error`) reads display tiles as they are before it alone.
        self.drop_uncovered_bumps()?;
        self.settle_display()?;
        self.sync_pins()?;
        let mut start = 0;
        while let Some(&next) = keys.get(start) {
            let mut run: Vec<TileKey> = Vec::new();
            for key in keys.get(start..).unwrap_or_default() {
                if run.contains(key) || !self.corrections.is_pinned(key) {
                    break;
                }
                run.push(*key);
            }
            if run.len() < 2 {
                self.edit_tile(next, |edit| edit.update(|i, j, c| f(start, i, j, c)))??;
                start += 1;
            } else {
                self.update_run(start, &run, &f)?;
                start += run.len();
            }
        }
        Ok(())
    }

    /// [`TileStore::update_tiles`] for `run`, tiles `first..` of its keys, each pinned by the
    /// caller and none twice: their interiors in parallel, then each tile's border and marking
    /// in order.
    ///
    /// # Errors
    ///
    /// As [`TileStore::update_tiles`].
    fn update_run(
        &mut self,
        first: usize,
        run: &[TileKey],
        f: &(impl Fn(usize, u32, u32, Vec3) -> Option<Vec3> + Sync),
    ) -> Result<(), CacheError> {
        self.check_keys_and_op(run)?;
        let ids = run
            .iter()
            .map(|key| {
                self.corrections
                    .lookup(key)
                    .map(|(id, _)| id)
                    .ok_or(Invariant::StaleHandle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut slots = self
            .corrections
            .get_many_mut(&ids)
            .ok_or(Invariant::StaleHandle)?;
        let interiors: Vec<InteriorPass> = self.workers.install(|| {
            run.par_iter()
                .zip(slots.par_iter_mut())
                .enumerate()
                .map(|(k, (key, slot))| {
                    update_interior(slot, key.side(), |i, j, c| f(first + k, i, j, c))
                })
                .collect()
        });
        let mut outcome = Ok(());
        for (k, (&key, pass)) in run.iter().zip(interiors).enumerate() {
            if outcome.is_err() {
                // Past the failing tile: one whose interior was written is marked and dropped,
                // each of them whatever marking another does, and the first error is kept.
                if pass.first_row.is_some() {
                    #[expect(
                        clippy::redundant_closure_for_method_calls,
                        reason = "the method path names one edit lifetime; edit_tile's closure \
                                  must take any"
                    )]
                    let marked = self
                        .edit_tile(key, |edit| edit.mark_own_written())
                        .and_then(std::convert::identity);
                    keep_first(&mut outcome, marked);
                }
                continue;
            }
            outcome = self
                .edit_tile(key, |edit| {
                    edit.update_border(pass, |i, j, c| f(first + k, i, j, c))
                })
                .and_then(std::convert::identity);
        }
        outcome
    }

    /// [`TileStore::update_tiles`] along each vertex's tangent frame: `f(n, i, j, c, frame)` for
    /// vertex `(i, j)` of `keys[n]`, `frame` its frame (the store's kept prediction's,
    /// [`TileStore::kept_prediction`]: the bits its evaluation uses), with the same calls and the
    /// same bits as [`TileStore::update_tiles`] given each vertex's frame.
    ///
    /// For a brush writing a world move in each vertex's frame: the frames are read where the
    /// write is made, on the workers for the interiors, with no copy of them. Returns `false`,
    /// having done nothing, unless every tile of `keys` is pinned ([`TileStore::pin`]), has its
    /// prediction kept, and comes once.
    ///
    /// # Errors
    ///
    /// As [`TileStore::update_tiles`].
    pub fn update_tiles_framed(
        &mut self,
        keys: &[TileKey],
        f: impl Fn(usize, u32, u32, Vec3, &Frame) -> Option<Vec3> + Sync,
    ) -> Result<bool, CacheError> {
        let distinct: BTreeSet<&TileKey> = keys.iter().collect();
        let ready = distinct.len() == keys.len()
            && keys
                .iter()
                .all(|key| self.corrections.is_pinned(key) && self.kept.get(key).is_some());
        if !ready {
            return Ok(false);
        }
        // Every write before this one is settled, as `update_tiles` settles it.
        self.drop_uncovered_bumps()?;
        self.settle_display()?;
        self.sync_pins()?;
        let still = keys
            .iter()
            .all(|key| self.corrections.is_pinned(key) && self.kept.get(key).is_some());
        if !still {
            return Ok(false);
        }
        self.check_keys_and_op(keys)?;
        let ids = keys
            .iter()
            .map(|key| {
                self.corrections
                    .lookup(key)
                    .map(|(id, _)| id)
                    .ok_or(Invariant::StaleHandle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let kept = &self.kept;
        let mut slots = self
            .corrections
            .get_many_mut(&ids)
            .ok_or(Invariant::StaleHandle)?;
        // Each tile's interior, and a copy of its border vertices' frames for the border writes
        // after: those go through the store, which may drop predictions as it settles.
        let (interiors, borders): (Vec<InteriorPass>, Vec<BorderFrames>) =
            self.workers.install(|| {
                keys.par_iter()
                    .zip(slots.par_iter_mut())
                    .enumerate()
                    .map(|(k, (key, slot))| {
                        let frames = kept.get(key).map_or(&[][..], Prediction::frames);
                        let side = key.side();
                        let pass = update_interior(slot, side, |i, j, c| {
                            let at = usize::try_from(j * side + i).unwrap_or(usize::MAX);
                            let frame = frames.get(at).copied().unwrap_or(Frame::NAN);
                            f(k, i, j, c, &frame)
                        });
                        (pass, BorderFrames::of(side, frames))
                    })
                    .unzip()
            });
        let mut outcome = Ok(());
        let passes = keys.iter().zip(interiors).zip(&borders).enumerate();
        for (k, ((&key, pass), border)) in passes {
            if outcome.is_err() {
                if pass.first_row.is_some() {
                    #[expect(
                        clippy::redundant_closure_for_method_calls,
                        reason = "the method path names one edit lifetime; edit_tile's closure \
                                  must take any"
                    )]
                    let marked = self
                        .edit_tile(key, |edit| edit.mark_own_written())
                        .and_then(std::convert::identity);
                    keep_first(&mut outcome, marked);
                }
                continue;
            }
            outcome = self
                .edit_tile(key, |edit| {
                    edit.update_border(pass, |i, j, c| f(k, i, j, c, &border.frame(i, j)))
                })
                .and_then(std::convert::identity);
        }
        outcome.map(|()| true)
    }
}

/// A tile's border vertices' frames, copied from its kept prediction's for
/// [`TileStore::update_tiles_framed`]'s border writes: rows `0` and `side − 1`, then columns `0`
/// and `side − 1`, each in order.
#[derive(Debug)]
struct BorderFrames {
    /// Vertices per side.
    side: u32,
    /// The frames, `4 · side`; a vertex without one has [`Frame::NAN`].
    frames: Vec<Frame>,
}

impl BorderFrames {
    /// The border frames of a tile of `side` vertices a side from its `frames`, row-major.
    fn of(side: u32, frames: &[Frame]) -> Self {
        let width = usize::try_from(side).unwrap_or(0);
        let last = width.saturating_sub(1);
        let frame = |at: usize| frames.get(at).copied().unwrap_or(Frame::NAN);
        let rows = [0, last]
            .into_iter()
            .flat_map(|j| (0..width).map(move |i| j * width + i));
        let columns = [0, last]
            .into_iter()
            .flat_map(|i| (0..width).map(move |j| j * width + i));
        let border = rows.chain(columns).map(frame).collect();
        Self {
            side,
            frames: border,
        }
    }

    /// Border vertex `(i, j)`'s frame; [`Frame::NAN`] for a vertex not on the border.
    fn frame(&self, i: u32, j: u32) -> Frame {
        let (side, last) = (self.side, self.side.saturating_sub(1));
        let at = if j == 0 {
            Some(i)
        } else if j == last {
            Some(side + i)
        } else if i == 0 {
            Some(2 * side + j)
        } else if i == last {
            Some(3 * side + j)
        } else {
            None
        };
        at.and_then(|at| self.frames.get(usize::try_from(at).ok()?))
            .copied()
            .unwrap_or(Frame::NAN)
    }
}
