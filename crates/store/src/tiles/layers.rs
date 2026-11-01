//! The layers a store reads through, the working file over the base, and what crosses between them:
//! the dirty tiles a recovery flush snapshots, what a save leaves, and what a roll-over fills.

use std::collections::{BTreeMap, BTreeSet};

use geom::{Level, TileKey};

use super::error::{CacheError, Invariant};
use crate::bounds::Book;
use crate::file::{BaseFile, Loc, ProjectFile};
use crate::hash::TileMap;
use crate::meta::{Meta, Role, SavePoint};
use crate::op::OpId;
use crate::pool::SlotId;
use crate::slot::CorrectionSlot;
use crate::stamp::Stamp;

/// The latest write of a dirty correction slot: what its record is written with, and where that
/// write stands among the store's writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dirty {
    /// The op of the edit that made the write: the op its record carries.
    pub(crate) op: OpId,
    /// The write's stamp: a flush's copy of the slot leaves it clean only if no write came
    /// after (see [`TileStore::append_streamed`](super::TileStore::append_streamed)).
    pub(crate) stamp: Stamp,
}

/// Appends a dirty correction slot's tile to the working file, with the op of its latest write
/// (no save point), and returns that op.
///
/// The write is forgotten: the slot is clean once this returns.
///
/// # Errors
///
/// [`CacheError::WriteBack`] if the file's write fails; [`CacheError::Internal`] if the slot has
/// no recorded write (every slot marked dirty gets one first) or doesn't hold a valid tile.
pub(super) fn write_slot(
    file: &mut ProjectFile,
    unsaved: &mut TileMap<Dirty>,
    key: TileKey,
    slot: &CorrectionSlot,
) -> Result<OpId, CacheError> {
    let Dirty { op, stamp } = unsaved
        .get(&key)
        .copied()
        .ok_or(Invariant::MissingOp(key))?;
    let tile = slot.to_tile().map_err(Invariant::CorrectionSlot)?;
    file.write_displacement_stamped(key, op, stamp, &tile)
        .map_err(CacheError::WriteBack)?;
    unsaved.remove(&key);
    Ok(op)
}

/// The first op the store hands out for `site` after the files' `heads`: the next after the
/// greatest head of `site`, or sequence 0 of `site` if none is of it.
///
/// # Errors
///
/// [`CacheError::OpsExhausted`] if that head is the site's last op id.
pub(super) fn first_op(
    site: u16,
    heads: impl IntoIterator<Item = Option<OpId>>,
) -> Result<OpId, CacheError> {
    let latest = heads
        .into_iter()
        .flatten()
        .filter(|head| head.site() == site)
        .max();
    match latest {
        Some(head) => head.next(),
        None => OpId::new(site, 0),
    }
    .ok_or(CacheError::OpsExhausted)
}

/// A copy of a dirty correction slot, taken for a streaming consumer by
/// [`TileStore::snapshot_dirty`](super::TileStore::snapshot_dirty): what the recovery writer
/// encodes off the owner thread.
#[derive(Debug)]
pub(crate) struct Snapshot {
    /// The tile.
    pub(crate) key: TileKey,
    /// The op of the tile's latest write when the copy was taken: the record's op.
    pub(crate) op: OpId,
    /// The stamp of that write: what places the record against the flush's cut, and what
    /// [`TileStore::append_streamed`](super::TileStore::append_streamed) compares against to tell
    /// whether the slot was written since.
    pub(crate) stamp: Stamp,
    /// The slot's contents, boxed so the hand-off moves a pointer, not 26.9 KB.
    pub(crate) slot: Box<CorrectionSlot>,
}

/// Proof that a save copied every tile edited since the last rebase into a project file, made
/// only by [`TileStore::save_into`](super::TileStore::save_into) and taken by
/// [`TileStore::rebase`](super::TileStore::rebase).
///
/// It holds the project file's new save point, as the base to rebase on, the store's latest
/// write when the save was made (a rebase refuses it if the store has been written since, as
/// those writes are in no save), and the tiles the save found lost: edited, but readable
/// nowhere. There is no other way to make one, so a store can't be rebased past edits that
/// were never saved, except those it has listed as lost.
#[derive(Debug)]
pub(crate) struct Saved {
    /// The project file as of the save point the save wrote.
    pub(super) base: BaseFile,
    /// The stamp of the store's latest write when the save was made.
    pub(super) stamp: Stamp,
    /// The tiles edited since the last rebase whose latest edit couldn't be read (see
    /// [`TileStore::save_into`](super::TileStore::save_into)): the save point holds their last
    /// saved state.
    pub(super) lost: BTreeSet<TileKey>,
    /// The bounds changes the store had counted when the save was made: a rebase refuses it
    /// after a later one, which the save point doesn't hold.
    pub(super) bounds: u64,
    /// How many of the stroke records waiting in the store the save copied: the rebase forgets
    /// them, as the project file holds them.
    pub(super) strokes: usize,
}

impl Saved {
    /// The save point the save wrote: what the fresh working file to rebase on must name.
    pub(crate) const fn save_point(&self) -> SavePoint {
        self.base.save_point()
    }

    /// The tiles whose latest edit the save couldn't read, and left out.
    pub(crate) const fn lost(&self) -> &BTreeSet<TileKey> {
        &self.lost
    }
}

/// A new working file [`TileStore::fill_roll_over`](super::TileStore::fill_roll_over) has filled
/// with every edit and saved, waiting for [`TileStore::roll_over`](super::TileStore::roll_over) to
/// swap it in.
#[derive(Debug)]
pub(crate) struct Filled {
    /// The new working file.
    pub(super) file: ProjectFile,
    /// The tiles that couldn't be carried across.
    pub(super) lost: Vec<TileKey>,
    /// The dirty slots written to it, clean once it is swapped in.
    pub(super) cleaned: Vec<(TileKey, SlotId<CorrectionSlot>)>,
    /// The store's latest write when it was filled: a roll-over refuses it after a later one.
    pub(super) stamp: Stamp,
    /// The bounds changes the store had counted when it was filled: a roll-over refuses it
    /// after a later one, which the new file doesn't hold.
    pub(super) bounds: u64,
    /// How many of the stroke records waiting in the store were copied into it: the roll-over
    /// forgets them, as the new file holds them.
    pub(super) strokes: usize,
}

/// The bounds book of a store over `working` and `base`.
///
/// It tracks every level either file holds bounds of and, for a recovery file, holds its bounds
/// as changed since the last save (no save has copied them into the project file yet). A bound of
/// the recovery file that doesn't read counts as unknown.
///
/// # Errors
///
/// [`CacheError::Store`] if reading the recovery file's bounds fails.
pub(super) fn book_of(working: &ProjectFile, base: Option<&BaseFile>) -> Result<Book, CacheError> {
    let mut levels: BTreeSet<Level> = working
        .bounds_locations()
        .map(|(key, _)| key.level())
        .collect();
    if let Some(base) = base {
        levels.extend(base.bounds_locations().map(|(key, _)| key.level()));
    }
    let changed = match working.meta().role() {
        Role::Recovery { .. } => {
            let entries: Vec<(TileKey, Loc)> = working
                .bounds_locations()
                .map(|(key, loc)| (*key, loc))
                .collect();
            let read = working.read_bounds(&entries)?;
            entries
                .iter()
                .zip(read)
                .map(|((key, _), bounds)| (*key, bounds))
                .collect()
        }
        Role::Project => BTreeMap::new(),
    };
    Ok(Book::new(levels, changed))
}

/// `project` as of the save point a save has just written to it.
///
/// # Errors
///
/// [`CacheError::Internal`] with [`Invariant::NotSaved`] if `project` has none, which a save
/// that went through rules out.
pub(super) fn saved_base(project: &ProjectFile) -> Result<BaseFile, CacheError> {
    project.base().ok_or_else(|| Invariant::NotSaved.into())
}

/// Checks that a working file with `working`'s meta may be layered on `base`.
///
/// Over a base, the working file must be a recovery file of the base's project naming the
/// base's save point exactly: its project, its offset and its head op. A recovery file of an
/// earlier or later save, or of another project, would be read through the wrong saved state.
/// The base itself must be a project file. With no base, the working file must be a project
/// file, or a recovery file of a project never saved; a recovery file that names a save point
/// would be missing the state it extends.
///
/// # Errors
///
/// [`CacheError::WrongBase`].
pub(super) fn check_layers(working: &Meta, base: Option<&BaseFile>) -> Result<(), CacheError> {
    let fits = match base {
        None => matches!(
            working.role(),
            Role::Project | Role::Recovery { base: None }
        ),
        Some(base) => {
            let point = base.save_point();
            base.meta().role() == Role::Project
                && working.project() == point.project
                && working.role() == Role::Recovery { base: Some(point) }
        }
    };
    if fits {
        Ok(())
    } else {
        Err(CacheError::WrongBase)
    }
}
