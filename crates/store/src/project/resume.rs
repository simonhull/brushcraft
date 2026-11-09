//! The strokes still landing when a recovery file was last written, found by a restore.
//!
//! They are landed on in op order (see [`crate::journal`], the dab journal spec's "Restoring a
//! journalled stroke", and the stroke-while-landing spec's "Showing and landing").
//!
//! # Finding them
//!
//! A restore opens the recovery file at its last durable save point, which holds every dab batch
//! before it. Several strokes may be open there: the **writing** one, landing, and strokes begun
//! behind it while it landed, their dabs placed and journalled, none landed (see
//! `crate::tiles::journal`, "Side by side"). Every flush writes each open stroke's batches, so
//! batches of several ops interleave, and the file's last batch may be any open stroke's, or the
//! closing batch of one let go.
//!
//! So the restore looks at every batch, in file order, reading only its fixed fields
//! ([`ProjectFile::read_batch_head`]): an op is **open** if its last batch has dabs not landed
//! (`landed < placed`). A stroke that ended wrote a last batch counting every dab landed, so a
//! closed stroke never hides another, earlier or later, whatever the order of their batches.
//! For each open op the scan keeps its batches from the last one starting at or before its
//! landed count (a batch's dabs start where the one before left off, or past the dabs that
//! landed before it was written); every other op's batches between them are passed over. A
//! batch that starts anywhere else is a gap or an overlap.
//!
//! The strokes are offered in **op order**. If the file holds an edit of a later op of a
//! stroke's site ([`ProjectFile::head`]), that stroke ended with dabs not landed and no closing
//! batch (a save in its middle moved its later edits to a new op, or an op began with no
//! stroke): landing its rows now would land them over that op's edits, so its journal is unusable
//! ([`JournalUnusable::Ended`]). The head orders ops only within a site, so one of another site is
//! never taken for later.
//!
//! # The pen-down surface
//!
//! A stroke's **pen-down surface** is what its batches name ([`PenDown`]), never one found from
//! the order of ops. Its batches may go from naming the stroke before it
//! ([`PenDown::After`], while it was begun behind that one and its own cut wasn't durable) to a
//! save point, never back; its last batch's names it.
//!
//! - **A save point** ([`PenDown::Base`] or [`PenDown::At`]): a tile whose record at the restored
//!   save point is the one it had there is its own pen-down shape (the stroke never changed it).
//!   Every other tile's shape is its record there, or, where it had none, the base's (the project
//!   file's save point the recovery file extends), or zero, as the store reads a tile.
//! - **After the stroke before** ([`PenDown::After`]): the stroke has landed nothing (a batch
//!   counting a dab landed always names its own save point), so it has no shapes; its surface is
//!   the one that stroke leaves. If that stroke is open in the file, it is the open stroke just
//!   before in op order (strokes begin behind the newest open one, and one let go takes every
//!   stroke behind it with it), and the later one resumes only once it has been resumed, landed and
//!   ended ([`OpenStroke::after`]): its surface is then the store's state. If it is closed in the
//!   file, every dab of it is in the restored tiles, which are the later one's surface at once.
//!   That one may have no batch in the file if every dab it placed had landed by the first cut
//!   after its own: then its edits are, and the file's head is at or past its op. If the file holds
//!   neither, its dabs aren't there (it was lost to the crash before its first batch was written,
//!   its landing after the last cut), and the later one can't be landed.
//!
//! # What is exact
//!
//! What the store restores is exact: the tiles as the save point names them, each stroke's dabs
//! not landed (the very rows it journalled), its pen-down shapes (the tiles as it found them, bit
//! for bit) and its journal, which goes on from its landed count as the live one would. The
//! landing above the store is not: the app's stroke (`sculpt::Stroke`) restarted on the restored
//! tiles takes each vertex's start from the stored, quantized correction, where the live stroke
//! kept it unrounded, so the result matches the live one within half a quantization step per
//! vertex the stroke touches (the dab journal spec, "Restoring a journalled stroke"). A stroke
//! resumed after the one before it lands on what that one's landing left, so the bound holds per
//! stroke.
//!
//! # When it can't be used
//!
//! Anything that keeps the store from restoring a stroke exactly makes its journal unusable
//! ([`JournalUnusable`]), never an error of the restore: the restored tiles stand, as they would
//! with no journal, and the project opens. That is a batch that doesn't read where the stroke
//! may need it, a gap or an overlap, batches of the stroke that disagree, rows of no format, a
//! pen-down surface no save point of the file holds, or one that doesn't read, or a later op's
//! edits over the stroke. **A stroke whose earlier stroke is unusable is unusable too**
//! ([`JournalUnusable::EarlierUnusable`]): its surface is the one that stroke would leave.
//!
//! A batch that doesn't read may be any op's: the last of every stroke open when it was written
//! and with no batch after it, so those strokes are [`JournalUnusable::Damaged`]; a stroke one of
//! whose batches comes after it knows its last batch, and finds the damage only if it needs the
//! batch (a gap). With no stroke open, a last batch that doesn't read is offered as a stroke
//! of no op, damaged.
//!
//! # Bounds
//!
//! The file may be hostile. Each batch read is bounded by its record ([`crate::journal`]), the
//! scan keeps only the batches of the strokes open as it goes (a closing batch drops its
//! stroke's), each from the last at or before its landed count, and the strokes' rows and
//! pen-down shapes together may hold no more than the pools' share of the store's budget
//! ([`TileStore::pools_budget`]): a stroke whose journal would go past what the strokes before
//! it left is [`JournalUnusable::TooLarge`], and a file whose open batches would is one of no op,
//! too large. A live stroke has no such cap (its backlog and its shapes are unbounded, the shapes
//! held beyond the budget if they must), so a stroke that big lands live and its journal can't be
//! restored: the price of never letting a hostile file make a restore allocate without bound. A
//! real stroke's rows and shapes are a few MB.
//!
//! The scan reads every batch's record once: the work grows with the file, as opening it does.

use std::collections::BTreeMap;

use geom::{CorrectionTile, TileKey};
use thiserror::Error;

use crate::OpId;
use crate::file::{Loc, ProjectFile, ReadScratch};
use crate::journal::{BatchHead, DAB_BYTES, DabBatch, PenDown};
use crate::slot::CorrectionSlot;
use crate::tiles::{PenDownShape, StoreId, TileStore};

/// A stroke whose dabs hadn't all landed when its recovery file was last written.
///
/// A restore finds it ([`crate::Project::take_open_strokes`]): its op, the dabs not landed, and
/// its pen-down shapes, for [`crate::Project::resume_stroke`] to land on. Only a restore makes
/// one, and it names the store it was found in, as that store's working file was then: so it
/// resumes only there, once.
pub struct OpenStroke {
    /// The stroke's op.
    op: OpId,
    /// How many of its dabs had landed: `dabs` starts at this index.
    landed: u32,
    /// The version of the rows' layout, as the app set it when the stroke began (never 0).
    dab_format: u32,
    /// The dabs from `landed` on, in placement order, each a `stroke::Dab`'s bytes.
    dabs: Vec<[u8; DAB_BYTES]>,
    /// What its last batch names as its pen-down surface: never unknown.
    pen_down: PenDown,
    /// The open stroke found before it, which it must wait for: `Some` only with `pen_down`
    /// [`PenDown::After`] that stroke.
    after: Option<OpId>,
    /// Each tile the stroke had changed by the restored save point, as the stroke found it, in
    /// key order: none for a stroke whose pen-down is [`PenDown::After`].
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
    /// the count [`crate::Project::journal_landed`] goes on from. Always 0 for a stroke whose
    /// pen-down is [`PenDown::After`].
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

    /// Where the stroke's pen-down surface is: the save point holding it, [`PenDown::Base`] or
    /// [`PenDown::At`]; or [`PenDown::After`] the stroke it began behind, its surface the one
    /// that stroke leaves (see the module notes). Never [`PenDown::Unknown`].
    #[must_use]
    pub const fn pen_down(&self) -> PenDown {
        self.pen_down
    }

    /// The open stroke found before this one, whose surface it lands on: it resumes only once
    /// that one has been resumed, has landed and has ended ([`crate::Project::resume_stroke`]).
    /// `None` for a stroke that waits for none: one whose pen-down is a save point, or after a
    /// stroke the file holds closed, every dab of which the restored tiles hold.
    #[must_use]
    pub const fn after(&self) -> Option<OpId> {
        self.after
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
            .field("after", &self.after)
            .field("shapes", &self.shapes.len())
            .finish_non_exhaustive()
    }
}

/// Why [`crate::Project::resume_stroke`] didn't resume a stroke, with the stroke given back.
///
/// The project is as it was, and a call made again may resume it (after a transient error, or
/// once the stroke before it has ended).
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

/// A stroke a restore found still landing, until it is resumed: if the project hands out
/// another op first, the stroke has ended, and the project closes it in the file
/// ([`TileStore::close_restored_journal`]).
#[derive(Debug, Clone, Copy)]
pub(super) struct Unresumed {
    /// The stroke's op.
    pub(super) op: OpId,
    /// The landed count of its last batch.
    pub(super) landed: u32,
    /// Its dabs placed: the landed count and the dabs not landed.
    pub(super) placed: u32,
    /// What its last batch names as its pen-down surface.
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
    /// its checksum, isn't one this build reads, or the read fails. Or a batch that doesn't read
    /// may have been the stroke's last (see the module notes).
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
    /// to was abandoned, or the stroke began before the recovery file did; or they name the
    /// stroke before it ([`PenDown::After`]), and the file holds neither a batch nor an edit of
    /// that one, whose dabs the crash lost.
    #[error("no save point holds the surface the stroke began on")]
    PenDownUnknown,
    /// The save point the batches name isn't one of the recovery file's.
    #[error("the save point the stroke began on is missing")]
    PenDownMissing,
    /// Batches of the stroke disagree: on its pen-down (one going back from a save point to
    /// the stroke before it, or naming two), on its rows' format, or a landed count goes back.
    /// Or it names the stroke before it with dabs landed, or names another than the open stroke
    /// found just before it, or a save point while that one is open.
    #[error("the stroke's batches disagree")]
    Mismatch,
    /// The rows' format is 0: the app never set it, so no app can say it reads them.
    #[error("the stroke's dabs are of no format")]
    UnsetFormat,
    /// The stroke's dabs and pen-down shapes would hold more than what the strokes before it
    /// left of the pools' share of the store's budget. A live stroke has no such cap, so this can
    /// be a real stroke's journal, but a restore never allocates past it for a file it can't
    /// trust (see the module notes).
    #[error("the stroke's journal is larger than the store's budget")]
    TooLarge,
    /// A later op's edits stand over the stroke: its last batch leaves dabs not landed, but the
    /// file holds edits of an op of its site begun after it, so the stroke ended without its
    /// closing batch (a save in its middle moved its later edits to a new op, or an op began
    /// with no stroke). Landing its rows now would land them over those edits.
    #[error("a later op's edits stand over the stroke")]
    Ended,
    /// The stroke began behind an open stroke whose journal is unusable: its surface is the one
    /// that stroke would leave once landed, which can't be had.
    #[error("the stroke before it can't be finished")]
    EarlierUnusable {
        /// The op of the stroke before it.
        earlier: OpId,
        /// Why that one's journal is unusable.
        #[source]
        why: Box<Self>,
    },
}

/// What a restore found of the strokes the working file's journal leaves open.
#[derive(Debug)]
pub(super) struct Found {
    /// Each open stroke, in op order, usable or why not; a last batch of no op found that
    /// doesn't read, or open batches past the budget, last.
    pub(super) strokes: Vec<Result<OpenStroke, JournalUnusable>>,
    /// The newest op any batch that reads names: ops handed out from now on must be past it, as
    /// they must be past the edits' (see [`crate::Project::restore`]).
    pub(super) newest_op: Option<OpId>,
}

/// The strokes `store`'s working file, just restored, holds still landing, in op order (see the
/// module notes).
///
/// None if it holds no batch, or every op's last batch has all its dabs landed (its closing
/// batch, or every dab placed landed). A last batch with dabs not landed under edits of a later
/// op is [`JournalUnusable::Ended`]. A later op is found from the file's head, which is the
/// newest op only among one site's ([`OpId`]'s ordering puts a greater site after every op of a
/// lesser, whenever it was made): so it is compared only when it is of the stroke's site. An op
/// of another site says nothing either way, and the closing batch alone says whether the stroke
/// ended. The head is the greatest op over every site, so a greater site's op hides a later op
/// of the stroke's own: that can't happen while every edit comes from one site, as now. When
/// edits of several sites are merged, the file must keep its head per site for this check; and
/// a batch's [`PenDown::After`], which names an op of its own site before its own (the store
/// reserves every op on its own site), would need an order across sites too.
pub(super) fn find(store: &TileStore) -> Found {
    let file = store.working();
    let scan = Scan::of(file, store.pools_budget());
    if scan.too_large {
        return Found {
            strokes: vec![Err(JournalUnusable::TooLarge)],
            newest_op: scan.newest,
        };
    }
    let mut room = store.pools_budget();
    let mut strokes = Vec::with_capacity(scan.open.len());
    // The open stroke just before, and whether it is usable.
    let mut before: Option<(OpId, Result<(), JournalUnusable>)> = None;
    for (&op, opened) in &scan.open {
        let own = open_stroke(store, op, opened, scan.first_damaged, &mut room);
        let found = chained(file, own, before.as_ref());
        before = Some((op, found.as_ref().map(|_| ()).map_err(Clone::clone)));
        strokes.push(found);
    }
    if scan.last_damaged && scan.open.is_empty() {
        strokes.push(Err(JournalUnusable::Damaged));
    }
    Found {
        strokes,
        newest_op: scan.newest,
    }
}

/// `own`, a stroke found open, as the strokes before it say (see the module notes).
///
/// `before` is the open stroke found just before it and whether that is usable, `None` if it is
/// the oldest.
///
/// # Errors
///
/// `own`'s error; else [`JournalUnusable::EarlierUnusable`] after an unusable one;
/// [`JournalUnusable::Mismatch`] naming another than the open one before it, or a save point
/// with one open before it; [`JournalUnusable::PenDownUnknown`] after a stroke the file holds
/// nothing of ([`holds_work_of`]).
fn chained(
    file: &ProjectFile,
    own: Result<OpenStroke, JournalUnusable>,
    before: Option<&(OpId, Result<(), JournalUnusable>)>,
) -> Result<OpenStroke, JournalUnusable> {
    let mut stroke = own?;
    match (before, stroke.pen_down) {
        (Some((earlier, Err(why))), _) => Err(JournalUnusable::EarlierUnusable {
            earlier: *earlier,
            why: Box::new(why.clone()),
        }),
        (Some((earlier, Ok(()))), PenDown::After(named)) if named == *earlier => {
            stroke.after = Some(named);
            Ok(stroke)
        }
        (Some(_), _) => Err(JournalUnusable::Mismatch),
        (None, PenDown::After(named)) if !holds_work_of(file, named) => {
            Err(JournalUnusable::PenDownUnknown)
        }
        (None, _) => Ok(stroke),
    }
}

/// Whether `file`'s state holds the work of the closed stroke made as `op`, which a stroke after
/// it names.
///
/// That is a batch of it that reads (its closing one, as no batch of it is open), or, with none,
/// edits of it or of a later op of its site (its head). A stroke is journalled from its
/// first flush after its own cut, unless every dab it placed had landed by then: so one with no
/// batch either landed every dab before a cut (its edits are in, and the head is at or past it)
/// or was lost to the crash before its first batch (its landing came after the last cut, and
/// the head is before it). Reads the fixed fields of the batches until it finds one.
fn holds_work_of(file: &ProjectFile, op: OpId) -> bool {
    let edited = file
        .head()
        .is_some_and(|head| head.site() == op.site() && head >= op);
    edited
        || file
            .batch_locations()
            .any(|loc| file.read_batch_head(loc).is_ok_and(|head| head.op == op))
}

/// A batch of an open stroke as the scan keeps it.
#[derive(Debug, Clone, Copy)]
struct Kept {
    /// Its place among the file's batches, from 0 in file order.
    index: usize,
    /// Where it is.
    loc: Loc,
    /// Its fixed fields.
    head: BatchHead,
}

/// One stroke the scan has found open so far.
#[derive(Debug, Default)]
struct Opened {
    /// Its batches from the last at or before its landed count, in file order.
    batches: Vec<Kept>,
    /// Whether a batch that doesn't read came after its last: that one may have been its last.
    damaged: bool,
}

/// What a look over every batch of a file finds (see the module notes, "Finding them").
#[derive(Debug, Default)]
struct Scan {
    /// The ops whose last batch has dabs not landed, each with what the scan keeps of it.
    open: BTreeMap<OpId, Opened>,
    /// The place of the first batch that doesn't read, if one doesn't.
    first_damaged: Option<usize>,
    /// Whether the file's last batch doesn't read.
    last_damaged: bool,
    /// The newest op of a batch that reads.
    newest: Option<OpId>,
    /// Whether the batches kept would hold more than the room given: then none is kept.
    too_large: bool,
}

impl Scan {
    /// Looks over every batch of `file`, in file order, reading their fixed fields alone; the
    /// batches kept may hold `room` bytes at most.
    fn of(file: &ProjectFile, room: usize) -> Self {
        let most = room / size_of::<Kept>();
        let mut scan = Self::default();
        let mut kept = 0_usize;
        let batches = file.batch_locations().enumerate();
        for (index, loc) in batches {
            let Ok(head) = file.read_batch_head(loc) else {
                scan.first_damaged.get_or_insert(index);
                scan.last_damaged = true;
                for opened in scan.open.values_mut() {
                    opened.damaged = true;
                }
                continue;
            };
            scan.last_damaged = false;
            scan.newest = scan.newest.max(Some(head.op));
            if head.landed == head.placed() {
                // Closed: every dab placed has landed, or the stroke ended.
                if let Some(closed) = scan.open.remove(&head.op) {
                    kept = kept.saturating_sub(closed.batches.len());
                }
                continue;
            }
            if scan.too_large {
                continue;
            }
            let opened = scan.open.entry(head.op).or_default();
            opened.damaged = false;
            opened.batches.push(Kept { index, loc, head });
            // The landed count only grows, so a batch before the last that starts at or before
            // it is never needed again.
            let from = opened
                .batches
                .iter()
                .rposition(|batch| batch.head.first <= head.landed)
                .unwrap_or(0);
            opened.batches.drain(..from);
            kept = kept.saturating_add(1).saturating_sub(from);
            if kept > most {
                scan.too_large = true;
                scan.open.clear();
                kept = 0;
            }
        }
        scan
    }
}

/// The open stroke of `op`, as the scan found it (`opened`, with `first_damaged` the place of
/// the first batch that doesn't read), its rows and shapes taken from `room`.
///
/// # Errors
///
/// The [`JournalUnusable`] of the first thing found wrong (see the module notes); `room` is
/// unchanged then.
fn open_stroke(
    store: &TileStore,
    op: OpId,
    opened: &Opened,
    first_damaged: Option<usize>,
    room: &mut usize,
) -> Result<OpenStroke, JournalUnusable> {
    if opened.damaged {
        return Err(JournalUnusable::Damaged);
    }
    let file = store.working();
    let later_op = file
        .head()
        .is_some_and(|head| head.site() == op.site() && head > op);
    if later_op {
        return Err(JournalUnusable::Ended);
    }
    let (Some(first), Some(last)) = (opened.batches.first(), opened.batches.last()) else {
        return Err(JournalUnusable::Damaged);
    };
    let landed = last.head.landed;
    // The batch holding the landed count's dab may be one that doesn't read.
    if first.head.first > landed && first_damaged.is_some_and(|at| at < first.index) {
        return Err(JournalUnusable::Damaged);
    }
    let rows_of = |head: &BatchHead| -> usize {
        let from = head.first.max(landed);
        usize::try_from(head.placed().saturating_sub(from)).unwrap_or(usize::MAX)
    };
    let row_bytes = opened.batches.iter().fold(0_usize, |bytes, kept| {
        bytes.saturating_add(rows_of(&kept.head).saturating_mul(DAB_BYTES))
    });
    if row_bytes > *room {
        return Err(JournalUnusable::TooLarge);
    }
    let read = opened
        .batches
        .iter()
        .map(|kept| file.read_batch(kept.loc).or(Err(JournalUnusable::Damaged)));
    let run: Result<Vec<DabBatch>, JournalUnusable> = read.collect();
    let run = run?;
    let (pen_down, dab_format) = check_run(&run, landed)?;
    if dab_format == 0 {
        return Err(JournalUnusable::UnsetFormat);
    }
    let left = room.saturating_sub(row_bytes);
    let (shapes, shape_bytes) = match pen_down {
        PenDown::Base => pen_down_surface(store, None, left)?,
        PenDown::At(offset) => pen_down_surface(store, Some(offset), left)?,
        PenDown::Unknown => return Err(JournalUnusable::PenDownUnknown),
        // A stroke lands nothing before its own cut, and every batch after its cut names it.
        PenDown::After(_) if landed > 0 => return Err(JournalUnusable::Mismatch),
        PenDown::After(_) => (Vec::new(), 0),
    };
    let rows = run.iter().flat_map(|batch| {
        let before = usize::try_from(landed.saturating_sub(batch.first())).unwrap_or(usize::MAX);
        batch.dabs().iter().skip(before).copied()
    });
    let dabs: Vec<[u8; DAB_BYTES]> = rows.collect();
    *room = left.saturating_sub(shape_bytes);
    Ok(OpenStroke {
        op,
        landed,
        dab_format,
        dabs,
        pen_down,
        after: None,
        shapes,
        store: store.id(),
        generation: store.generation(),
    })
}

/// Checks `run`, the batches of one stroke in file order; returns the pen-down its last names
/// and their format.
///
/// The last's landed count is `landed`. They must agree on the rows' format, their pen-down may
/// go from the stroke before ([`PenDown::After`]) to a save point (or unknown) but never back or
/// elsewhere, their landed counts never go back, and each starts where the one before leaves
/// off.
///
/// # Errors
///
/// [`JournalUnusable::Mismatch`] if they disagree; [`JournalUnusable::Gap`] at the first batch
/// that starts elsewhere, or for the first if it starts past `landed`.
fn check_run(run: &[DabBatch], landed: u32) -> Result<(PenDown, u32), JournalUnusable> {
    let (Some(first), Some(last)) = (run.first(), run.last()) else {
        return Err(JournalUnusable::Damaged);
    };
    let pairs = run.iter().zip(run.iter().skip(1));
    for (before, after) in pairs {
        let pen_down_holds = match (before.pen_down(), after.pen_down()) {
            (was, now) if was == now => true,
            (PenDown::After(_), PenDown::At(_) | PenDown::Base | PenDown::Unknown) => true,
            _ => false,
        };
        let agree = pen_down_holds
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
    Ok((last.pen_down(), first.dab_format()))
}

/// A stroke's pen-down shapes: the corrections of each tile changed since `earlier`, as they were,
/// and the bytes they hold as shapes.
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
) -> Result<(Vec<FoundShape>, usize), JournalUnusable> {
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
    Ok((shapes, needed))
}
