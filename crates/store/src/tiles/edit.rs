//! Editing correction tiles: [`TileEdit`], the border and interior passes that carry an edit to
//! every copy of a vertex, and the tiles an edit touched.

use geom::{
    CorrectionError, CorrectionViewMut, MAX_COPIES, Region, RowUpdate, TileBorder, TileKey, Vec3,
    VertexCopy,
};

use super::error::{CacheError, Invariant};
use super::{TileStore, keep_first};
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
pub(super) struct InteriorPass {
    /// The first row with an interior vertex written: where, in row-major order, the tile's
    /// first interior write falls (after the row's first border vertex, before its last).
    pub(super) first_row: Option<u32>,
    /// The row where a vertex failed, and why: nothing after it in row-major order was written.
    pub(super) failed: Option<(u32, CacheError)>,
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
pub(super) fn update_interior(
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
    pub(super) key: TileKey,
    /// Its correction slot, pinned until the edit ends.
    pub(super) id: SlotId<CorrectionSlot>,
    /// Whether a copy in it was written (it is dirty and on the feed).
    pub(super) written: bool,
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
    pub(super) store: &'a mut TileStore,
    /// The tile edited.
    pub(super) key: TileKey,
    /// The op every write is recorded with.
    pub(super) op: OpId,
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
    pub(super) fn finish(&mut self) -> Result<(), CacheError> {
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
    /// so the same bits: order matters, as a value that outgrows the tile's scale re-quantizes
    /// its interior. It is cheaper, as each run of interior vertices of a row (which have no
    /// copies in other tiles) is read and written through one borrowed view instead of a
    /// lookup per call; border vertices go through [`TileEdit::set`], copies and all.
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
    pub(super) fn update_border(
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
    pub(super) fn mark_own_written(&mut self) -> Result<(), CacheError> {
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
