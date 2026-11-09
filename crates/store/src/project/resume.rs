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
//! landed before it was written); every other op's batches between them are passed over. Each
//! batch is checked against the one of its op before it as the scan meets it: one that starts
//! anywhere else (a gap or an overlap), or disagrees with it, breaks the op's run, and none of
//! its batches is kept from then on.
//!
//! The strokes are offered in **op order**. If the file holds an edit of a later op of a
//! stroke's site ([`ProjectFile::head`]), that stroke ended with dabs not landed and no closing
//! batch (a save in its middle moved its later edits to a new op, or an op began with no
//! stroke): landing its rows now would land them over that op's edits, so its journal is unusable
//! ([`JournalUnusable::Ended`]). The head orders ops only within a site, so one of another site is
//! never taken for later.
//!
//! # Chains
//!
//! The open strokes form **chains**: a stroke whose batches name a surface of its own (a save
//! point, or unknown: its own cut was taken) starts one, and every stroke begun behind it names
//! the one just before it ([`PenDown::After`]). Only the last chain can be live: a stroke begins
//! its own cut only once every stroke before it has ended, so an open stroke before the last
//! chain's start is one left open by an earlier session (unusable then, never closed in the
//! file, or a crash before its closing batch) and has ended ([`JournalUnusable::Ended`]); it
//! never makes a later stroke unusable. A stroke naming one that isn't open starts a chain too,
//! on the surface that one left, if the file holds it (see below). A restored project closes
//! every stroke it found open, usable or not, as it hands out a fresh op, so a later restore
//! finds none of them.
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
//!   file (its last batch that reads counts every dab landed, and no batch that doesn't read comes
//!   after it), every dab of it is in the restored tiles, which are the later one's surface at
//!   once. That one may have no batch in the file if every dab it placed had landed by the first
//!   cut after its own: then its edits are, and the file's head is at or past its op; that is
//!   trusted only with every batch read, as one that doesn't read may have been its. If the file
//!   holds neither, its dabs aren't there (it was lost to the crash before its first batch was
//!   written, its landing after the last cut), and the later one can't be landed.
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
//! edits over the stroke. **A stroke whose earlier stroke in its chain is unusable is unusable
//! too** ([`JournalUnusable::EarlierUnusable`]): its surface is the one that stroke would leave,
//! and its own journal isn't read.
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
//! stroke's, a batch that breaks its op's run drops them all), each from the last at or before
//! its landed count, and an entry per op it has met; all of that, and the strokes' rows and
//! pen-down shapes together, may hold no more than the pools' share of the store's budget
//! ([`TileStore::pools_budget`]): a stroke whose journal would go past what the strokes before
//! it left is [`JournalUnusable::TooLarge`], and a file whose batches would make the scan keep
//! more is one of no op, too large. The scan does a constant amount of work a batch, beside
//! reading it, and each stroke's batches are read whole one at a time, each let go once its rows
//! are copied. A live stroke has no such cap (its backlog and its shapes are unbounded, the shapes
//! held beyond the budget if they must), so a stroke that big lands live and its journal can't be
//! restored: the price of never letting a hostile file make a restore allocate without bound. A
//! real stroke's rows and shapes are a few MB.
//!
//! The scan reads every batch's record once: the work grows with the file, as opening it does.

use std::collections::{BTreeMap, VecDeque};

use geom::{CorrectionTile, TileKey};
use thiserror::Error;

use crate::OpId;
use crate::file::{Loc, ProjectFile, ReadScratch};
use crate::journal::{BatchHead, DAB_BYTES, PenDown};
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

/// A stroke a restore found still landing, usable or not, until it is resumed.
///
/// If the project hands out a fresh op first, or lets the restored strokes go, the stroke has
/// ended, and the project closes it in the file ([`TileStore::close_restored_journal`]).
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
    /// Whether the restore offered it usable: it may resume, and is work a close keeps.
    pub(super) usable: bool,
}

impl Unresumed {
    /// The stroke made as `op` whose last batch that reads is `last`, found in the working file
    /// of `generation`; not yet known usable.
    const fn of(op: OpId, last: &BatchHead, generation: u64) -> Self {
        Self {
            op,
            landed: last.landed,
            placed: last.placed(),
            pen_down: last.pen_down,
            dab_format: last.dab_format,
            generation,
            usable: false,
        }
    }
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
    /// Or it names the stroke before it with dabs landed, or names an open stroke other than
    /// the one found just before it.
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
    /// The stroke ended without its closing batch: its last batch leaves dabs not landed, but
    /// the file holds edits of an op of its site begun after it (a save in its middle moved its
    /// later edits to a new op, or an op began with no stroke), or a later stroke began its own
    /// surface (see the module notes, "Chains"). Landing its rows now would land them over what
    /// came after.
    #[error("a later op's work stands over the stroke")]
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
    /// Every op found open, usable or not, in op order, as its last batch that reads leaves it:
    /// what the project closes in the file once a fresh op is handed out, so no stroke left open
    /// is found again by a later restore.
    pub(super) open: Vec<Unresumed>,
    /// The newest op any batch that reads names: ops handed out from now on must be past it, as
    /// they must be past the edits' (see [`crate::Project::restore`]).
    pub(super) newest_op: Option<OpId>,
}

/// The strokes `store`'s working file, just restored, holds still landing, in op order (see the
/// module notes).
///
/// None if it holds no batch, or every op's last batch has all its dabs landed (its closing
/// batch, or every dab placed landed). A last batch with dabs not landed under edits of a later
/// op is [`JournalUnusable::Ended`], and so is one before the last chain's start (see the module
/// notes, "Chains"). A later op is found from the file's head, which is the newest op only among
/// one site's ([`OpId`]'s ordering puts a greater site after every op of a lesser, whenever it
/// was made): so it is compared only when it is of the stroke's site. An op of another site
/// says nothing either way, and the closing batch alone says whether the stroke ended. The head
/// is the greatest op over every site, so a greater site's op hides a later op of the stroke's
/// own: that can't happen while every edit comes from one site, as now. When edits of several
/// sites are merged, the file must keep its head per site for this check; and a batch's
/// [`PenDown::After`], which names an op of its own site before its own (the store reserves
/// every op on its own site), would need an order across sites too.
pub(super) fn find(store: &TileStore) -> Found {
    let file = store.working();
    let scan = Scan::of(file, store.pools_budget());
    if scan.too_large {
        return Found {
            strokes: vec![Err(JournalUnusable::TooLarge)],
            open: Vec::new(),
            newest_op: scan.newest,
        };
    }
    let generation = store.generation();
    let left_open = scan
        .open
        .iter()
        .map(|(&op, opened)| Unresumed::of(op, &opened.last, generation));
    let open: Vec<Unresumed> = left_open.collect();
    let start = scan.chain_start();
    let mut room = store.pools_budget();
    let mut strokes = Vec::with_capacity(scan.open.len());
    // The open stroke just before in the chain, and whether it is usable.
    let mut before: Option<(OpId, Result<(), JournalUnusable>)> = None;
    let found_open = scan.open.iter().enumerate();
    for (at, (&op, opened)) in found_open {
        if at < start {
            // A later stroke began its own surface, so this one ended without its closing batch.
            strokes.push(Err(JournalUnusable::Ended));
            continue;
        }
        let found = match (&before, opened.last.pen_down) {
            // Not read: its surface is the one the unusable stroke would leave.
            (Some((earlier, Err(why))), _) => Err(JournalUnusable::EarlierUnusable {
                earlier: *earlier,
                why: Box::new(why.clone()),
            }),
            (Some((earlier, Ok(()))), PenDown::After(named)) if named == *earlier => {
                open_stroke(store, op, opened, &scan, &mut room).map(|mut stroke| {
                    stroke.after = Some(named);
                    stroke
                })
            }
            (Some(_), _) => Err(JournalUnusable::Mismatch),
            (None, PenDown::After(named)) => scan
                .earlier_closed(file, named)
                .and_then(|()| open_stroke(store, op, opened, &scan, &mut room)),
            (None, _) => open_stroke(store, op, opened, &scan, &mut room),
        };
        before = Some((op, found.as_ref().map(|_| ()).map_err(Clone::clone)));
        strokes.push(found);
    }
    if scan.last_damaged && scan.open.is_empty() {
        strokes.push(Err(JournalUnusable::Damaged));
    }
    Found {
        strokes,
        open,
        newest_op: scan.newest,
    }
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
#[derive(Debug)]
struct Opened {
    /// Its batches from the last at or before its landed count, in file order; none once
    /// `broken`.
    batches: VecDeque<Kept>,
    /// Its last batch that reads.
    last: BatchHead,
    /// Whether a batch that doesn't read came after its last: that one may have been its last.
    damaged: bool,
    /// Why its batches don't make one run, from the first batch that doesn't follow the one
    /// before it (see [`follows`]); none of them is kept from then on.
    broken: Option<JournalUnusable>,
}

impl Opened {
    /// A stroke whose first batch the scan found is `first`.
    fn new(first: Kept) -> Self {
        Self {
            batches: VecDeque::from([first]),
            last: first.head,
            damaged: false,
            broken: None,
        }
    }

    /// `next`, the stroke's next batch: kept if it follows the one before, from which the batches
    /// before the last at or before its landed count go (the landed count only grows, so they
    /// are never needed again); else the stroke is broken.
    fn push(&mut self, next: Kept) {
        self.last = next.head;
        self.damaged = false;
        if self.broken.is_some() {
            return;
        }
        let followed = self
            .batches
            .back()
            .map_or(Ok(()), |before| follows(&before.head, &next.head));
        if let Err(why) = followed {
            self.broken = Some(why);
            self.batches.clear();
            return;
        }
        self.batches.push_back(next);
        // The batches start where the one before left off, so their first indices only grow.
        while self
            .batches
            .get(1)
            .is_some_and(|second| second.head.first <= next.head.landed)
        {
            self.batches.pop_front();
        }
    }
}

/// Whether `after`, a stroke's batch, follows `before`, its batch just before it in the file.
///
/// They must agree on the rows' format (see the module notes), the pen-down stays or goes from the
/// stroke before ([`PenDown::After`]) to a save point (or unknown), never back or elsewhere, the
/// landed count never goes back, and `after` starts at the later of the dabs `before` placed and
/// its own landed count.
///
/// # Errors
///
/// [`JournalUnusable::Mismatch`] if they disagree; [`JournalUnusable::Gap`] if `after` starts
/// elsewhere.
fn follows(before: &BatchHead, after: &BatchHead) -> Result<(), JournalUnusable> {
    let pen_down_holds = match (before.pen_down, after.pen_down) {
        (was, now) if was == now => true,
        (PenDown::After(_), PenDown::At(_) | PenDown::Base | PenDown::Unknown) => true,
        _ => false,
    };
    let agree =
        pen_down_holds && after.dab_format == before.dab_format && after.landed >= before.landed;
    if !agree {
        return Err(JournalUnusable::Mismatch);
    }
    let expected = before.placed().max(after.landed);
    if after.first != expected {
        return Err(JournalUnusable::Gap {
            expected,
            found: after.first,
        });
    }
    Ok(())
}

/// The bytes the scan charges for each op it keeps, open or closed, beside its batches: the
/// entry and a map node's share. An estimate; what matters is that it grows with the ops.
const OP_BYTES: usize = size_of::<Opened>() + size_of::<OpId>() + 4 * size_of::<usize>();

/// What a look over every batch of a file finds (see the module notes, "Finding them").
#[derive(Debug, Default)]
struct Scan {
    /// The ops whose last batch has dabs not landed, each with what the scan keeps of it.
    open: BTreeMap<OpId, Opened>,
    /// The ops whose last batch that reads counts every dab landed, each with that batch's place.
    closed: BTreeMap<OpId, usize>,
    /// The place of the first batch that doesn't read, if one doesn't.
    first_damaged: Option<usize>,
    /// The place of the last batch that doesn't read, if one doesn't.
    last_damaged_at: Option<usize>,
    /// Whether the file's last batch doesn't read.
    last_damaged: bool,
    /// The newest op of a batch that reads.
    newest: Option<OpId>,
    /// Whether what the scan keeps would hold more than the room given: then none is kept.
    too_large: bool,
}

impl Scan {
    /// Looks over every batch of `file`, in file order, reading their fixed fields alone, each
    /// once and in constant work beside it; what it keeps may hold `room` bytes at most.
    ///
    /// A batch whose read fails is taken as damaged, the read error transient or not: it may
    /// hide an op (one whose every batch is unread is never found), which is benign, as it
    /// only ever makes strokes unusable, never lands one on a surface it doesn't hold: the
    /// strokes it may have been the last of are [`JournalUnusable::Damaged`], and so is a later
    /// stroke naming one the scan didn't find (see [`Scan::earlier_closed`]). A restore made
    /// again reads it afresh.
    fn of(file: &ProjectFile, room: usize) -> Self {
        let mut scan = Self::default();
        let mut bytes = 0_usize;
        let batches = file.batch_locations().enumerate();
        for (index, loc) in batches {
            let Ok(head) = file.read_batch_head(loc) else {
                scan.first_damaged.get_or_insert(index);
                scan.last_damaged_at = Some(index);
                scan.last_damaged = true;
                for opened in scan.open.values_mut() {
                    opened.damaged = true;
                }
                continue;
            };
            scan.last_damaged = false;
            scan.newest = scan.newest.max(Some(head.op));
            if scan.too_large {
                continue;
            }
            let kept = Kept { index, loc, head };
            if head.landed == head.placed() {
                // Closed: every dab placed has landed, or the stroke ended.
                if let Some(closed) = scan.open.remove(&head.op) {
                    bytes = bytes.saturating_sub(closed.batches.len() * size_of::<Kept>());
                } else {
                    bytes = bytes.saturating_add(OP_BYTES);
                }
                scan.closed.insert(head.op, index);
                continue;
            }
            if scan.closed.remove(&head.op).is_some() {
                bytes = bytes.saturating_sub(OP_BYTES);
            }
            let (before, after) = if let Some(opened) = scan.open.get_mut(&head.op) {
                let before = opened.batches.len();
                opened.push(kept);
                (before, opened.batches.len())
            } else {
                scan.open.insert(head.op, Opened::new(kept));
                bytes = bytes.saturating_add(OP_BYTES);
                (0, 1)
            };
            bytes = bytes
                .saturating_sub(before * size_of::<Kept>())
                .saturating_add(after * size_of::<Kept>());
            if bytes > room {
                scan.too_large = true;
                scan.open.clear();
                scan.closed.clear();
            }
        }
        scan
    }

    /// Where the last chain of open strokes starts, as a place among them in op order (see the
    /// module notes, "Chains"): the last stroke whose batches name a surface of its own (a save
    /// point, or unknown), or the stroke before it as one not open; 0 with none.
    fn chain_start(&self) -> usize {
        let starts =
            self.open
                .values()
                .enumerate()
                .filter(|(_, opened)| match opened.last.pen_down {
                    PenDown::After(named) => !self.open.contains_key(&named),
                    PenDown::Base | PenDown::At(_) | PenDown::Unknown => true,
                });
        starts.map(|(at, _)| at).next_back().unwrap_or(0)
    }

    /// Whether the stroke made as `named`, not open, which the oldest stroke of the chain names
    /// as the stroke before it, left the restored tiles holding its work (see the module notes).
    ///
    /// Its last batch that reads must count every dab landed, with no batch that doesn't read
    /// after it (that one may have been its last, with dabs not landed). With no batch of it, it
    /// is journalled from its first flush after its own cut unless every dab it placed had landed
    /// by then: so it either landed every dab before a cut (its edits are in `file`, whose head
    /// is at or past it) or was lost to the crash before its first batch (its landing came after
    /// the last cut, and the head is before it); that is told only with every batch read, as an
    /// unread one may have been its.
    ///
    /// # Errors
    ///
    /// [`JournalUnusable::Damaged`] if a batch that doesn't read may have been its last (or its
    /// only); [`JournalUnusable::PenDownUnknown`] if the file holds no work of it.
    fn earlier_closed(&self, file: &ProjectFile, named: OpId) -> Result<(), JournalUnusable> {
        match self.closed.get(&named) {
            Some(&at) if self.last_damaged_at.is_none_or(|damaged| damaged < at) => Ok(()),
            Some(_) => Err(JournalUnusable::Damaged),
            None if self.first_damaged.is_some() => Err(JournalUnusable::Damaged),
            None => {
                let edited = file
                    .head()
                    .is_some_and(|head| head.site() == named.site() && head >= named);
                if edited {
                    Ok(())
                } else {
                    Err(JournalUnusable::PenDownUnknown)
                }
            }
        }
    }
}

/// The open stroke of `op`, as `scan` found it (`opened`), its rows and shapes taken from `room`.
///
/// # Errors
///
/// The [`JournalUnusable`] of the first thing found wrong (see the module notes); `room` is
/// unchanged then.
fn open_stroke(
    store: &TileStore,
    op: OpId,
    opened: &Opened,
    scan: &Scan,
    room: &mut usize,
) -> Result<OpenStroke, JournalUnusable> {
    if let Some(broken) = &opened.broken {
        return Err(broken.clone());
    }
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
    let (Some(first), Some(last)) = (opened.batches.front(), opened.batches.back()) else {
        return Err(JournalUnusable::Damaged);
    };
    let landed = last.head.landed;
    if first.head.first > landed {
        // The batch holding the landed count's dab may be one that doesn't read.
        if scan.first_damaged.is_some_and(|at| at < first.index) {
            return Err(JournalUnusable::Damaged);
        }
        return Err(JournalUnusable::Gap {
            expected: landed,
            found: first.head.first,
        });
    }
    // The batches agree on their format, and their pen-down is the last's (see `follows`).
    let (pen_down, dab_format) = (last.head.pen_down, first.head.dab_format);
    if dab_format == 0 {
        return Err(JournalUnusable::UnsetFormat);
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
    let left = room.saturating_sub(row_bytes);
    let (shapes, shape_bytes) = match pen_down {
        PenDown::Base => pen_down_surface(store, None, left)?,
        PenDown::At(offset) => pen_down_surface(store, Some(offset), left)?,
        PenDown::Unknown => return Err(JournalUnusable::PenDownUnknown),
        // A stroke lands nothing before its own cut, and every batch after its cut names it.
        PenDown::After(_) if landed > 0 => return Err(JournalUnusable::Mismatch),
        PenDown::After(_) => (Vec::new(), 0),
    };
    let mut dabs: Vec<[u8; DAB_BYTES]> = Vec::with_capacity(row_bytes / DAB_BYTES);
    for kept in &opened.batches {
        // Each batch is read whole and goes once its rows are copied.
        let batch = file
            .read_batch(kept.loc)
            .or(Err(JournalUnusable::Damaged))?;
        if batch.op() != op || batch.first() != kept.head.first {
            return Err(JournalUnusable::Damaged);
        }
        let before = usize::try_from(landed.saturating_sub(batch.first())).unwrap_or(usize::MAX);
        dabs.extend(batch.dabs().iter().skip(before));
    }
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
