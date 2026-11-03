//! Why a [`TileStore`](super::TileStore) operation fails: [`CacheError`], and the broken invariants
//! it carries ([`Invariant`]).

use std::convert::Infallible;

use geom::{CorrectionError, EvalError, TileKey};
use thiserror::Error;

use crate::file::FileError;
use crate::op::RESERVED_SITE;
use crate::pool::PoolError;
use crate::workers::StartWorkersError;

/// Why a [`TileStore`](super::TileStore) operation failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CacheError {
    /// Reading the working file, or recording a save point in it, failed.
    #[error("project file operation failed")]
    Store(#[from] FileError),
    /// No slot to be had: every slot of the pool that needs one is pinned, and no chunk without a
    /// pin is left to take back. Only pins taken with [`TileStore::pin`](super::TileStore::pin) can
    /// crowd the store's own work this far (see
    /// [`StoreConfig::minimum_budget`](super::StoreConfig::minimum_budget)).
    #[error("every slot is pinned and no chunk can be taken back")]
    Full,
    /// Writing a dirty correction tile to the working file failed: when its slot was needed, or
    /// on [`TileStore::save`](super::TileStore::save). The tile stays resident and dirty.
    #[error("writing a dirty correction tile back failed")]
    WriteBack(#[source] FileError),
    /// Writing the project file during a save failed: a tile, or the save point. The store is
    /// unchanged (its working file still holds every edit), and the save can be tried again.
    #[error("writing the project file during a save failed")]
    Save(#[source] FileError),
    /// Evaluation failed (not for a valid mesh).
    #[error("evaluation failed")]
    Eval(#[from] EvalError),
    /// A display tile of the store's display layer wouldn't build or decode from what the store
    /// holds.
    #[error("display tile could not be built")]
    Display(#[from] geom::DisplayError),
    /// A correction was rejected: not finite, at level 0, on an unknown face, or outside the face
    /// or tile. Nothing was written.
    #[error("correction rejected")]
    Correction(#[source] CorrectionError),
    /// The key isn't a base-layer displacement tile above level 0 of a face of this mesh (or,
    /// for a preview, of the store's finest level).
    #[error("{0:?} is not a base-layer displacement tile of this mesh above level 0")]
    NotATile(TileKey),
    /// A preview's block isn't gathered over its tile's region
    /// ([`TileStore::set_preview`](super::TileStore::set_preview)).
    #[error("the preview given for {0:?} is not that tile's block")]
    NotItsBlock(TileKey),
    /// A preview was put back for a tile whose preview wasn't taken
    /// ([`TileStore::put_back_preview`](super::TileStore::put_back_preview)).
    #[error("{0:?} has no preview taken to put back")]
    NoPreviewTaken(TileKey),
    /// The configured site is [`RESERVED_SITE`].
    #[error("site {RESERVED_SITE} is reserved")]
    ReservedSite,
    /// The working file and the base under it (or the store and a file given to a rebase or a
    /// save) are over different base meshes: their meta's mesh hashes differ.
    #[error("the working file and its base are over different meshes")]
    MeshMismatch,
    /// The working file isn't one that may be layered on the base given: a recovery file must
    /// name, in its meta, the project and the save point (offset and head op) of its base, and
    /// a file with no base must be a project file or a recovery file of a project never saved.
    /// Or a save was asked to write into a file that isn't a project file.
    #[error("the working file doesn't belong on this base")]
    WrongBase,
    /// [`TileStore::save`](super::TileStore::save) was called while a recovery writer streams the
    /// working file, which writes its save points itself.
    #[error("the working file is streamed by a recovery writer, which writes its save points")]
    Streaming,
    /// [`TileStore::save`](super::TileStore::save) was called while the working file is a recovery
    /// file whose writer failed: a sync of it failed (or the writer's thread stopped), so no
    /// save point may be written there again. The project moves the store to a new file; save
    /// the project ([`crate::Project::save`]) instead.
    #[error("the working file is a recovery file whose sync failed: no save point goes there")]
    FailedWorkingFile,
    /// A rebase was given a save that is out of date: the store was edited after the save was
    /// made, so the save doesn't hold every edit. Save again. Or a roll-over was given a new
    /// working file filled before the latest edit.
    #[error("the store was edited after the save it was to be rebased on")]
    StaleSave,
    /// The configured budget is below
    /// [`StoreConfig::minimum_budget`](super::StoreConfig::minimum_budget) for the mesh, without
    /// counting the reserved shares (see [`CacheError::TooManyKeptPredictions`] and
    /// [`CacheError::TooManyStrokeTiles`]).
    #[error("the budget is below the {minimum} bytes this mesh needs")]
    BudgetTooSmall {
        /// The least budget the mesh works with, at the configured
        /// [`StoreConfig::kept_predictions`](super::StoreConfig::kept_predictions) and
        /// [`StoreConfig::stroke_tiles`](super::StoreConfig::stroke_tiles)
        /// ([`StoreConfig::minimum_budget`](super::StoreConfig::minimum_budget)).
        minimum: usize,
    },
    /// The budget holds the pools' minimum, but not beside the share the configured
    /// [`StoreConfig::kept_predictions`](super::StoreConfig::kept_predictions) need
    /// ([`StoreConfig::kept_prediction_bytes`](super::StoreConfig::kept_prediction_bytes)).
    #[error("keeping {asked} predictions takes more of the budget than the {most} it can spare")]
    TooManyKeptPredictions {
        /// The kept predictions asked for.
        asked: usize,
        /// The most this budget can keep beside the pools' minimum.
        most: usize,
    },
    /// The budget holds the pools' minimum and the kept predictions' share, but not beside the
    /// share the configured [`StoreConfig::stroke_tiles`](super::StoreConfig::stroke_tiles) need
    /// ([`StoreConfig::stroke_state_bytes`](super::StoreConfig::stroke_state_bytes)).
    #[error("stroke state for {asked} tiles takes more of the budget than the {most} it can spare")]
    TooManyStrokeTiles {
        /// The tiles' stroke state asked for.
        asked: usize,
        /// The most tiles' stroke state this budget can set aside beside the pools' minimum
        /// and the kept predictions.
        most: usize,
    },
    /// The display layer's floor (every face's display tiles up to
    /// [`StoreConfig::floor_top`](super::StoreConfig::floor_top), kept resident) doesn't fit
    /// beside the pools' minimum and the other reserved shares.
    #[error(
        "the display floor takes {floor} bytes of the budget, more than the {most} it can spare"
    )]
    FloorTooLarge {
        /// The floor's share, at its largest.
        floor: usize,
        /// The most the budget can set aside beside the pools' minimum and the other shares.
        most: usize,
    },
    /// Every op id of this site has been handed out.
    #[error("the op ids of this site are used up")]
    OpsExhausted,
    /// A pin request can't be held within the budget, even with every unpinned slot freed.
    /// Nothing was evicted or pinned.
    #[error("pinning {tiles} tiles needs {needed} bytes of pool chunks; the budget is {budget}")]
    OverBudget {
        /// Distinct tiles requested.
        tiles: usize,
        /// Bytes of chunks the request and the pins already held would take.
        needed: usize,
        /// The budget.
        budget: usize,
    },
    /// A batch of whole tiles named one tile twice
    /// ([`TileStore::replace_corrections`](super::TileStore::replace_corrections)). Nothing was
    /// written.
    #[error("{0:?} is given twice")]
    NamedTwice(TileKey),
    /// A batch of whole tiles changed a border vertex without giving every copy of it the same
    /// bits: a copy's tile is missing from the batch, or holds other bits there
    /// ([`TileStore::replace_corrections`](super::TileStore::replace_corrections)). Nothing was
    /// written.
    #[error("border vertex ({i}, {j}) of {tile:?} changes without its twins")]
    TwinsDisagree {
        /// The tile changing the vertex.
        tile: TileKey,
        /// The vertex's column in the tile.
        i: u32,
        /// The vertex's row in the tile.
        j: u32,
    },
    /// The store found one of its own invariants broken: a bug in the store, never caused by
    /// its input or the file's contents.
    #[error("internal invariant broken")]
    Internal(#[from] Invariant),
    /// The evaluation worker threads couldn't be started (see
    /// [`StoreConfig::workers`](super::StoreConfig::workers)).
    #[error("starting the evaluation worker threads failed")]
    Workers(#[source] StartWorkersError),
}

/// An invariant of [`TileStore`](super::TileStore) found broken (see [`CacheError::Internal`]).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Invariant {
    /// A dirty correction tile had no op to write it with.
    #[error("dirty correction tile {0:?} has no op")]
    MissingOp(TileKey),
    /// A correction slot didn't hold a valid tile, or refused a write the store had checked.
    #[error("a correction slot is broken")]
    CorrectionSlot(#[source] CorrectionError),
    /// A slot handle was used after its slot was freed.
    #[error("a slot handle outlived its slot")]
    StaleHandle,
    /// Slots borrowed together weren't all live and distinct: a handle outlived its slot, or
    /// two named the same slot.
    #[error("slots borrowed together were stale or named one slot twice")]
    SlotsNotDisjoint,
    /// A tile an evaluation didn't pin was looked for among those it did.
    #[error("{0:?} is not among the tiles the evaluation pinned")]
    NotHeld(TileKey),
    /// A tile read at pen-down as one the stroke changed had no pen-down shape, or no prediction
    /// to make its positions from, where the store had just made sure of both.
    #[error("{0:?} has no pen-down shape to read")]
    NoPenDownShape(TileKey),
    /// A tile was named twice among the tasks of one round, which write one tile each.
    #[error("{0:?} is named twice in one round")]
    NamedTwice(TileKey),
    /// A tile was given a second slot in one pool.
    #[error("a tile was given a second slot")]
    SecondSlot,
    /// A slot was unpinned more often than it was pinned.
    #[error("a slot was unpinned more often than pinned")]
    NotPinned,
    /// A pinned slot was to be freed.
    #[error("a pinned slot was to be freed")]
    FreedPinned,
    /// A dirty slot was to be freed without being written back.
    #[error("a dirty slot was to be freed")]
    FreedDirty,
    /// A rebase found a tile the store could read (in its working file or its base) that the
    /// save it was to be rebased on doesn't hold.
    #[error("{0:?} is not in the save a rebase was given")]
    NotInSave(TileKey),
    /// A save found a tile edited since the last rebase that is neither dirty nor in the
    /// working file: its edit is nowhere to copy from.
    #[error("edited tile {0:?} is neither dirty nor in the working file")]
    NothingToSave(TileKey),
    /// A project file had no save point right after a save wrote one.
    #[error("a project file had no save point after a save")]
    NotSaved,
    /// A correction tile was located in a base the store doesn't have.
    #[error("a tile was located in a missing base")]
    MissingBase,
    /// A slot or buffer held fewer values than a tile has.
    #[error("a buffer of {got} values is shorter than a tile's {expected}")]
    ShortBuffer {
        /// Values a tile has.
        expected: usize,
        /// Values the buffer held.
        got: usize,
    },
    /// A full save point would have named a different number of records than the maps derived
    /// for it hold (see [`FileError::SavePointMismatch`]). Nothing was written.
    #[error("a full save point names {named} records, but its derived maps hold {derived}")]
    SavePointMismatch {
        /// The records the full Index would name.
        named: usize,
        /// The records the derived maps hold.
        derived: usize,
    },
}

impl CacheError {
    /// `error`, from writing a save point, as the store reports it.
    ///
    /// [`FileError::SavePointMismatch`] is a bug in deriving the save point, which trying again
    /// can't fix, so it is [`Invariant::SavePointMismatch`] whatever the operation was. Any other
    /// error is `wrap`ped as the operation reports file errors.
    pub(super) fn from_save_point(error: FileError, wrap: impl FnOnce(FileError) -> Self) -> Self {
        if let FileError::SavePointMismatch { named, derived } = error {
            return Invariant::SavePointMismatch { named, derived }.into();
        }
        wrap(error)
    }
}

/// No operation fails with [`Infallible`]: this lets `?` take pool results whose write-back
/// can't fail.
impl From<Infallible> for CacheError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

/// A pool's error, as the store reports it: [`CacheError::Full`], the write-back's own error,
/// or a broken invariant.
impl<W> From<PoolError<W>> for CacheError
where
    Self: From<W>,
{
    fn from(e: PoolError<W>) -> Self {
        match e {
            PoolError::Full => Self::Full,
            PoolError::WriteBack(w) => w.into(),
            PoolError::Present => Invariant::SecondSlot.into(),
            PoolError::Stale => Invariant::StaleHandle.into(),
            PoolError::NotPinned => Invariant::NotPinned.into(),
            PoolError::Pinned => Invariant::FreedPinned.into(),
            PoolError::Dirty => Invariant::FreedDirty.into(),
        }
    }
}
