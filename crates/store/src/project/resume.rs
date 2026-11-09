//! A stroke still landing when its recovery file was last written, found by a restore and landed
//! on (see [`crate::journal`], and the dab journal spec's "Restoring a journalled stroke").
//!
//! # Finding it
//!
//! A restore opens the recovery file at its last durable save point, which holds every dab batch
//! before it. The **last** of them gives the open stroke: its op, and its landed count, which
//! describes exactly the tiles that save point names. If that batch's dabs have all landed (a
//! stroke that ended writes a last batch so), no stroke is open. If the file holds an edit of a
//! later op of the stroke's site ([`ProjectFile::head`]), the stroke ended with dabs not landed
//! and no closing batch (a save in its middle moved its later edits to a new op, or an op began
//! with no stroke): landing its rows now would land them over that op's edits, so its journal is
//! unusable ([`JournalUnusable::Ended`]). The head orders ops only within a site, so one of
//! another site is never taken for later. Otherwise the batches of its op,
//! read back from the last, give its dabs from the landed count on: each batch's first index is
//! the later of the dabs the one before it placed and its own landed count (a batch's dabs start
//! where the last left off, or past the dabs that landed before they were written), so the read
//! stops at the first batch holding the landed count's dab, and a batch that starts anywhere else
//! is a gap or an overlap.
//!
//! The stroke's **pen-down surface** is the save point its batches name ([`PenDown`]), never one
//! found from the order of ops. A tile whose record at the restored save point is the one it had
//! there is its own pen-down shape: the stroke never changed it. Every other tile's shape is its
//! record there, or, where it had none, the base's (the project file's save point the recovery
//! file extends), or zero, as the store reads a tile.
//!
//! # What is exact
//!
//! What the store restores is exact: the tiles as the save point names them, the stroke's dabs
//! not landed (the very rows it journalled), its pen-down shapes (the tiles as it found them, bit
//! for bit) and its journal, which goes on from its landed count as the live one would. The
//! landing above the store is not: the app's stroke (`sculpt::Stroke`) restarted on the restored
//! tiles takes each vertex's start from the stored, quantized correction, where the live stroke
//! kept it unrounded, so the result matches the live one within half a quantization step per
//! vertex the stroke touches (the dab journal spec, "Restoring a journalled stroke").
//!
//! # When it can't be used
//!
//! Anything that keeps the store from restoring the stroke exactly makes its journal unusable
//! ([`JournalUnusable`]), never an error of the restore: the restored tiles stand, as they would
//! with no journal, and the project opens. That is a batch that doesn't read, a gap or an overlap,
//! batches of the stroke that disagree, rows of no format, a pen-down surface no save point of
//! the file holds, or one that doesn't read, or a later op's edits over the stroke.
//!
//! # Bounds
//!
//! The file may be hostile. Each batch read is bounded by its record ([`crate::journal`]), the
//! batches read are those of the open stroke, from the last back, and the stroke's rows and
//! pen-down shapes together may hold no more than the pools' share of the store's budget
//! ([`TileStore::pools_budget`]): a journal that would is [`JournalUnusable::TooLarge`]. A live
//! stroke has no such cap (its backlog and its shapes are unbounded, the shapes held beyond the
//! budget if they must), so a stroke that big lands live and its journal can't be restored: the
//! price of never letting a hostile file make a restore allocate without bound. A real stroke's
//! rows and shapes are a few MB.

use geom::{CorrectionTile, TileKey};
use thiserror::Error;

use crate::OpId;
use crate::file::{Loc, ProjectFile, ReadScratch};
use crate::journal::{DAB_BYTES, DabBatch, PenDown};
use crate::slot::CorrectionSlot;
use crate::tiles::{PenDownShape, StoreId, TileStore};

/// A stroke whose dabs hadn't all landed when its recovery file was last written.
///
/// A restore finds it ([`crate::Project::take_open_stroke`]): its op, the dabs not landed, and its
/// pen-down shapes, for [`crate::Project::resume_stroke`] to land on. Only a restore makes one, and
/// it names the store it was found in, as that store's working file was then: so it resumes only
/// there, once.
pub struct OpenStroke {
    /// The stroke's op.
    op: OpId,
    /// How many of its dabs had landed: `dabs` starts at this index.
    landed: u32,
    /// The version of the rows' layout, as the app set it when the stroke began (never 0).
    dab_format: u32,
    /// The dabs from `landed` on, in placement order, each a `stroke::Dab`'s bytes.
    dabs: Vec<[u8; DAB_BYTES]>,
    /// The save point its batches name, holding its pen-down surface: never unknown.
    pen_down: PenDown,
    /// Each tile the stroke had changed by the restored save point, as the stroke found it, in
    /// key order.
    shapes: Vec<FoundShape>,
    /// The store it was found in.
    store: StoreId,
    /// That store's working file then ([`TileStore::generation`]).
    generation: u64,
}

impl OpenStroke {
    /// The stroke's op: the one its landing's edits are made as, once resumed.
    #[must_use]
    pub const fn op(&self) -> OpId {
        self.op
    }

    /// How many of the stroke's dabs had landed: the index of [`OpenStroke::dabs`]' first, and
    /// the count [`crate::Project::journal_landed`] goes on from.
    #[must_use]
    pub const fn landed(&self) -> u32 {
        self.landed
    }

    /// The version of the rows' layout, as the app set it when the stroke began
    /// ([`crate::Project::set_dab_format`]): an app that doesn't read rows of it must not land
    /// them. Never 0.
    #[must_use]
    pub const fn dab_format(&self) -> u32 {
        self.dab_format
    }

    /// The dabs not landed, from index [`OpenStroke::landed`] on, in placement order, each a
    /// `stroke::Dab`'s bytes.
    #[must_use]
    pub fn dabs(&self) -> &[[u8; DAB_BYTES]] {
        &self.dabs
    }

    /// The save point holding the stroke's pen-down surface: [`PenDown::Base`] or
    /// [`PenDown::At`], never [`PenDown::Unknown`] or [`PenDown::After`].
    #[must_use]
    pub const fn pen_down(&self) -> PenDown {
        self.pen_down
    }

    /// The tiles the stroke had changed by the restored save point, in key order: those it
    /// keeps pen-down shapes of once resumed. Every other tile is its own. For tests.
    #[cfg(test)]
    pub(crate) fn pen_down_keys(&self) -> impl Iterator<Item = &TileKey> {
        self.shapes.iter().map(|shape| &shape.key)
    }

    /// Each changed tile as the stroke found it, in key order: what
    /// [`crate::Project::resume_stroke`] installs as its pen-down shapes.
    pub(super) fn shapes(&self) -> &[FoundShape] {
        &self.shapes
    }

    /// What the project keeps of the stroke until it is resumed, to close it in the file if it
    /// never is ([`Unresumed`]).
    pub(super) fn unresumed(&self) -> Unresumed {
        let rows = u32::try_from(self.dabs.len()).unwrap_or(u32::MAX);
        Unresumed {
            op: self.op,
            landed: self.landed,
            placed: self.landed.saturating_add(rows),
            pen_down: self.pen_down,
            dab_format: self.dab_format,
            generation: self.generation,
        }
    }

    /// Whether the stroke was found in `store` as it is now: its working file the same.
    pub(super) fn found_in(&self, store: &TileStore) -> bool {
        self.store == store.id() && self.generation == store.generation()
    }
}

/// A tile a restored stroke had changed, as the stroke found it: read from the save point that
/// holds its pen-down surface, for [`TileStore::install_pen_down`].
pub(super) struct FoundShape {
    /// The tile.
    pub(super) key: TileKey,
    /// Its corrections as the stroke found them.
    pub(super) corrections: CorrectionTile,
    /// The op of the record they were read from; `None` where the tile had none, and read as
    /// zero.
    pub(super) op: Option<OpId>,
}

/// Shows the counts, never the rows or the shapes.
impl std::fmt::Debug for OpenStroke {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenStroke")
            .field("op", &self.op)
            .field("landed", &self.landed)
            .field("dabs", &self.dabs.len())
            .field("dab_format", &self.dab_format)
            .field("pen_down", &self.pen_down)
            .field("shapes", &self.shapes.len())
            .finish_non_exhaustive()
    }
}

/// Why [`crate::Project::resume_stroke`] didn't resume a stroke, with the stroke given back: the
/// project is as it was, and a call made again may resume it (after a transient error).
#[derive(Debug, Error)]
#[error("the restored stroke wasn't resumed")]
pub struct NotResumed {
    /// The stroke, as it was given.
    stroke: Box<OpenStroke>,
    /// Why.
    #[source]
    error: crate::ProjectError,
}

impl NotResumed {
    /// `stroke` wasn't resumed, for `error`.
    pub(super) fn new(stroke: OpenStroke, error: crate::ProjectError) -> Self {
        Self {
            stroke: Box::new(stroke),
            error,
        }
    }

    /// Why the stroke wasn't resumed: [`crate::ProjectError::NotResumable`], or the error of
    /// installing its pen-down shapes.
    #[must_use]
    pub const fn error(&self) -> &crate::ProjectError {
        &self.error
    }

    /// The stroke, to try again or let go.
    #[must_use]
    pub fn into_stroke(self) -> OpenStroke {
        *self.stroke
    }
}

/// A stroke a restore found still landing, until it is resumed: if the project begins another
/// op first, the stroke has ended, and the project closes it in the file
/// ([`TileStore::close_restored_journal`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Unresumed {
    /// The stroke's op.
    pub(super) op: OpId,
    /// The landed count of its last batch.
    pub(super) landed: u32,
    /// Its dabs placed: the landed count and the dabs not landed.
    pub(super) placed: u32,
    /// The save point its batches name.
    pub(super) pen_down: PenDown,
    /// Its rows' format.
    pub(super) dab_format: u32,
    /// The store's working file it was found in ([`TileStore::generation`]): one moved onto
    /// since holds none of its batches, so nothing is closed there.
    pub(super) generation: u64,
}

/// Why a journal was found and not used: the restored tiles stand, as they would with no journal,
/// and the stroke's dabs not landed are lost (see the module notes).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JournalUnusable {
    /// A batch the stroke needs, or a tile record of its pen-down surface, doesn't read: it fails
    /// its checksum, isn't one this build reads, or the read fails.
    #[error("the stroke's journal doesn't read")]
    Damaged,
    /// A batch of the stroke starts at another dab than the one after those before it: dabs are
    /// missing (a gap) or held twice (an overlap).
    #[error("the stroke's journal goes from dab {expected} to dab {found}")]
    Gap {
        /// The index the batch should start at: the later of the dabs placed before it and its
        /// landed count (or, for the first batch read, the landed count).
        expected: u32,
        /// The index it starts at.
        found: u32,
    },
    /// The batches name no save point holding the stroke's pen-down surface: the flush that was
    /// to was abandoned, or the stroke began before the recovery file did; or they name only the
    /// stroke before it, which it began behind while that one landed ([`PenDown::After`]), and a
    /// restore doesn't land that one first.
    #[error("no save point holds the surface the stroke began on")]
    PenDownUnknown,
    /// The save point the batches name isn't one of the recovery file's.
    #[error("the save point the stroke began on is missing")]
    PenDownMissing,
    /// Batches of the stroke disagree: on its pen-down save point, on its rows' format, or a
    /// landed count goes back.
    #[error("the stroke's batches disagree")]
    Mismatch,
    /// The rows' format is 0: the app never set it, so no app can say it reads them.
    #[error("the stroke's dabs are of no format")]
    UnsetFormat,
    /// The stroke's dabs and pen-down shapes would hold more than the pools' share of the
    /// store's budget. A live stroke has no such cap, so this can be a real stroke's journal, but
    /// a restore never allocates past it for a file it can't trust (see the module notes).
    #[error("the stroke's journal is larger than the store's budget")]
    TooLarge,
    /// A later op's edits stand over the stroke: its last batch leaves dabs not landed, but the
    /// file holds edits of an op of its site begun after it, so the stroke ended without its
    /// closing batch (a save in its middle moved its later edits to a new op, or an op began
    /// with no stroke). Landing its rows now would land them over those edits.
    #[error("a later op's edits stand over the stroke")]
    Ended,
}

/// The stroke `store`'s working file, just restored, holds still landing (see the module notes).
///
/// `None` if it holds no batch, or its last batch's dabs have all landed (its closing batch, or
/// every dab placed landed). A last batch with dabs not landed under edits of a later op is
/// [`JournalUnusable::Ended`]. A later op is found from the file's head, which is the newest op
/// only among one site's ([`OpId`]'s ordering puts a greater site after every op of a lesser,
/// whenever it was made): so it is compared only when it is of the stroke's site. An op of
/// another site says nothing either way, and the closing batch alone says whether the stroke
/// ended. The head is the greatest op over every site, so a greater site's op hides a later op
/// of the stroke's own: that can't happen while every edit comes from one site, as now. When
/// edits of several sites are merged, the file must keep its head per site for this check; and
/// a batch's [`PenDown::After`], which names an op of its own site before its own (the store
/// reserves every op on its own site), would need an order across sites too.
pub(super) fn find(store: &TileStore) -> Option<Result<OpenStroke, JournalUnusable>> {
    let file = store.working();
    let locations: Vec<Loc> = file.batch_locations().collect();
    let (&last, earlier) = locations.split_last()?;
    let Ok(last) = file.read_batch(last) else {
        return Some(Err(JournalUnusable::Damaged));
    };
    if last.landed() == last.placed() {
        return None;
    }
    let op = last.op();
    let later_op = file
        .head()
        .is_some_and(|head| head.site() == op.site() && head > op);
    if later_op {
        return Some(Err(JournalUnusable::Ended));
    }
    Some(open_stroke(store, last, earlier))
}

/// The open stroke whose last batch is `last`, its batches before it among `earlier`, in file
/// order.
///
/// # Errors
///
/// The [`JournalUnusable`] of the first thing found wrong (see the module notes).
fn open_stroke(
    store: &TileStore,
    last: DabBatch,
    earlier: &[Loc],
) -> Result<OpenStroke, JournalUnusable> {
    let room = store.pools_budget();
    let (op, landed) = (last.op(), last.landed());
    let run = run_of(store.working(), last, earlier, room)?;
    let (pen_down, dab_format) = check_run(&run, landed)?;
    if dab_format == 0 {
        return Err(JournalUnusable::UnsetFormat);
    }
    let earlier_point = match pen_down {
        PenDown::Base => None,
        PenDown::At(offset) => Some(offset),
        // A stroke begun while the one before it landed names that stroke until its own cut is
        // durable. Restoring a chain of strokes, the earlier landed first, isn't done here: one
        // whose batches name only `After` is unusable, and one whose cut became durable
        // meanwhile already fails above, as its batches disagree on their pen-down.
        PenDown::Unknown | PenDown::After(_) => return Err(JournalUnusable::PenDownUnknown),
    };
    let rows = run.iter().flat_map(|batch| {
        let before = usize::try_from(landed.saturating_sub(batch.first())).unwrap_or(usize::MAX);
        batch.dabs().iter().skip(before).copied()
    });
    let dabs: Vec<[u8; DAB_BYTES]> = rows.collect();
    let left = room.saturating_sub(dabs.len().saturating_mul(DAB_BYTES));
    let shapes = pen_down_surface(store, earlier_point, left)?;
    Ok(OpenStroke {
        op,
        landed,
        dab_format,
        dabs,
        pen_down,
        shapes,
        store: store.id(),
        generation: store.generation(),
    })
}

/// The batches of `last`'s op that hold its dabs from its landed count on, in file order.
///
/// `last` is the end of them, and the rest are read from `earlier` back, until one starts at or
/// before the landed count, or one of another op (or the file's first batch) comes first. Their
/// rows from the landed count on may hold at most `room` bytes.
///
/// # Errors
///
/// [`JournalUnusable::Damaged`] if a batch read doesn't; [`JournalUnusable::TooLarge`] past
/// `room`.
fn run_of(
    file: &ProjectFile,
    last: DabBatch,
    earlier: &[Loc],
    room: usize,
) -> Result<Vec<DabBatch>, JournalUnusable> {
    let (op, landed) = (last.op(), last.landed());
    let rows_of = |batch: &DabBatch| -> usize {
        let from = batch.first().max(landed);
        usize::try_from(batch.placed().saturating_sub(from)).unwrap_or(usize::MAX)
    };
    let mut bytes = rows_of(&last).saturating_mul(DAB_BYTES);
    let mut covered = last.first() <= landed;
    let mut run = vec![last];
    let back = earlier.iter().rev();
    for &loc in back {
        if covered {
            break;
        }
        let batch = file.read_batch(loc).or(Err(JournalUnusable::Damaged))?;
        if batch.op() != op {
            break;
        }
        bytes = bytes.saturating_add(rows_of(&batch).saturating_mul(DAB_BYTES));
        if bytes > room {
            return Err(JournalUnusable::TooLarge);
        }
        covered = batch.first() <= landed;
        run.push(batch);
    }
    if bytes > room {
        return Err(JournalUnusable::TooLarge);
    }
    run.reverse();
    Ok(run)
}

/// Checks `run`, the batches of one stroke in file order; returns their pen-down and format.
///
/// The last's landed count is `landed` (see [`run_of`]). They must agree on the pen-down save
/// point and the rows' format, their landed counts never go back, and each starts where the one
/// before leaves off.
///
/// # Errors
///
/// [`JournalUnusable::Mismatch`] if they disagree; [`JournalUnusable::Gap`] at the first batch
/// that starts elsewhere, or for the first if it starts past `landed`.
fn check_run(run: &[DabBatch], landed: u32) -> Result<(PenDown, u32), JournalUnusable> {
    let first = run.first().ok_or(JournalUnusable::Damaged)?;
    let pairs = run.iter().zip(run.iter().skip(1));
    for (before, after) in pairs {
        let agree = after.pen_down() == before.pen_down()
            && after.dab_format() == before.dab_format()
            && after.landed() >= before.landed();
        if !agree {
            return Err(JournalUnusable::Mismatch);
        }
        let expected = before.placed().max(after.landed());
        if after.first() != expected {
            return Err(JournalUnusable::Gap {
                expected,
                found: after.first(),
            });
        }
    }
    if first.first() > landed {
        return Err(JournalUnusable::Gap {
            expected: landed,
            found: first.first(),
        });
    }
    Ok((first.pen_down(), first.dab_format()))
}

/// A stroke's pen-down shapes: the corrections of each tile changed since `earlier`, as they were.
///
/// The tiles are those the working file's latest save point changed since `earlier` (the save
/// point at that offset, or, `None`, the file's start), in key order, each as it was there. A
/// tile with no record there reads the base's, or zero, as the store reads it. The shapes may
/// hold at most `room` bytes.
///
/// # Errors
///
/// [`JournalUnusable::PenDownMissing`] if `earlier` isn't a save point of the file's chain;
/// [`JournalUnusable::Damaged`] if its state doesn't load or a record doesn't read;
/// [`JournalUnusable::TooLarge`] past `room`.
fn pen_down_surface(
    store: &TileStore,
    earlier: Option<u64>,
    room: usize,
) -> Result<Vec<FoundShape>, JournalUnusable> {
    let file = store.working();
    let changed = match file.tiles_changed_since(earlier) {
        Ok(Some(changed)) => changed,
        Ok(None) => return Err(JournalUnusable::PenDownMissing),
        Err(_) => return Err(JournalUnusable::Damaged),
    };
    let needed = changed.iter().fold(0_usize, |bytes, (key, _)| {
        bytes.saturating_add(PenDownShape::bytes_of(key.side()))
    });
    if needed > room {
        return Err(JournalUnusable::TooLarge);
    }
    let mut slot = bytemuck::zeroed_box::<CorrectionSlot>();
    let mut scratch = ReadScratch::new().or(Err(JournalUnusable::Damaged))?;
    let mut shapes = Vec::with_capacity(changed.len());
    for (key, at) in changed {
        let base = store
            .base()
            .and_then(|base| base.locate(&key).map(|loc| (base, loc)));
        let read = match (at, base) {
            (Some(loc), _) => file.read_at(&key, loc, &mut slot, &mut scratch).map(Some),
            (None, Some((base, loc))) => base.read_at(&key, loc, &mut slot, &mut scratch).map(Some),
            (None, None) => Ok(None),
        };
        let op = read.or(Err(JournalUnusable::Damaged))?;
        let corrections = match op {
            Some(_) => slot.to_tile(),
            None => CorrectionTile::zero(key.side()),
        };
        shapes.push(FoundShape {
            key,
            corrections: corrections.or(Err(JournalUnusable::Damaged))?,
            op,
        });
    }
    Ok(shapes)
}
