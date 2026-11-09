//! The open stroke's journal and landing.
//!
//! Whether it may land, its pen-down cut as it begins, the dabs it places and lands, resuming one
//! a restore found, and the flush a close asks for to make it all durable.
//!
//! See [`crate::DabBatch`] for the record, and `crate::recovery` for the cut a stroke's first
//! landing waits for.

use std::time::Instant;

use super::resume::{FoundShape, NotResumed, OpenStroke};
use super::{Project, ProjectError, Retry};
use crate::OpId;
use crate::journal::{DAB_BYTES, PenDown};
use crate::recovery::RecoveryWriter;
use crate::tiles::{CacheError, JournalRefused, PenDownPoint};

impl Project {
    /// Whether the open stroke may land, edit the store: its pen-down flush has taken its cut (see
    /// [`Project::begin_stroke`]). While it may not, the app places, journals and shows its
    /// dabs ahead, but applies none; the frame loop's [`Project::tick`] and [`Project::poll`] drive
    /// the flushes, so it may within the flush in flight's syncs, or at once with
    /// [`Project::wait_to_land`]. True with no stroke open.
    ///
    /// A store edit of the stroke's op before then is a bug (a debug assertion says so): no save
    /// point would hold the surface the stroke found.
    #[must_use]
    pub fn may_land(&self) -> bool {
        self.store.may_land()
    }

    /// Whether a crash now would leave the open stroke restorable from the recovery file, as far as
    /// its pen-down surface goes: true with no stroke open, or with its pen-down surface in a save
    /// point ([`PenDown::Base`] or [`PenDown::At`]) or still to be cut by a flush pending; false
    /// once it is [`PenDown::Unknown`], which no later flush mends (see [`crate::DabBatch`]).
    ///
    /// Not every way there is reported: a cut failing as [`Project::begin_stroke`] begins the
    /// stroke is dropped (the stroke begins all the same), and a roll-over that can't carry the
    /// stroke goes through quietly. A close that would let the window go on the journal asks
    /// this, with [`Project::flushed`]: a durable journal of an unknown pen-down restores
    /// nothing.
    #[must_use]
    pub fn open_stroke_restorable(&self) -> bool {
        self.store.open_stroke_restorable()
    }

    /// Blocks until the open stroke may land ([`Project::may_land`]): the flush in flight is
    /// driven to its end, and the one behind it, holding the stroke's pen-down surface, takes its
    /// cut. For a landing that can't wait for frames, a close's or a save's. At once if it may
    /// already.
    ///
    /// # Errors
    ///
    /// As [`Project::poll`]. The stroke may land then all the same: a flush abandoned leaves its
    /// pen-down unknown.
    pub fn wait_to_land(&mut self) -> Result<(), ProjectError> {
        self.drive_to_land()?;
        self.report()
    }

    /// [`Project::wait_to_land`] without the report.
    ///
    /// # Errors
    ///
    /// As [`Project::with_writer`].
    pub(super) fn drive_to_land(&mut self) -> Result<(), ProjectError> {
        if self.store.may_land() {
            return Ok(());
        }
        let driven = self.with_writer(RecoveryWriter::wait_for_cut, Retry::Now);
        if !self.store.may_land() {
            // No writer is left to take the cut.
            self.store.pen_downs_abandoned();
        }
        driven
    }

    /// What a call that must go on met of crash protection: an old recovery file left undeleted
    /// is kept for the next call that reports ([`Project::report`]); anything else is the
    /// writer's to meet again with its next trigger, and to report then (see
    /// [`Project::begin_stroke`]).
    pub(super) fn keep_report(&mut self, outcome: Result<(), ProjectError>) {
        if let Err(kept @ ProjectError::OldRecoveryKept { .. }) = outcome {
            self.kept = Some(kept);
        }
    }

    /// The cut a stroke's op takes as it begins, at `now` (see [`Project::begin_stroke`]):
    /// where its pen-down surface will be. Unknown with no writer (crash protection paused), or
    /// if the cut fails.
    ///
    /// A writer that failed for good is replaced on the way, as for any call; the fresh file's
    /// save point then holds every edit, and the cut is asked of the new writer, which names it.
    pub(super) fn cut_before_op(&mut self, now: Instant) -> PenDownPoint {
        let generation = self.store.generation();
        let mut pen_down = PenDownPoint::Known(PenDown::Unknown);
        let cut = self.with_writer(
            |writer, store| {
                pen_down = writer.cut_before_op(store, now)?;
                Ok(())
            },
            Retry::Due(now),
        );
        // The stroke begins whatever the cut met.
        self.keep_report(cut);
        if self.store.generation() != generation
            && let Some(writer) = self.writer.as_mut()
        {
            // Rolled over: the new file holds every edit in its save point, and nothing was
            // edited since.
            pen_down = writer
                .cut_before_op(&mut self.store, now)
                .unwrap_or(PenDownPoint::Known(PenDown::Unknown));
        }
        pen_down
    }

    /// Reopens `stroke`'s op and installs its pen-down shapes, so its dabs land on the surface the
    /// stroke began on: the stroke is open again, as [`Project::begin_stroke`] leaves one, and may
    /// land at once (its pen-down surface is in the file, and its cut is the restored state).
    ///
    /// What the store gives back is exact: the tiles, the rows, the pen-down shapes and the
    /// journal. The landing above it isn't quite: the app's stroke restarted on the restored tiles
    /// takes each vertex's start from its stored, quantized correction, so the result matches the
    /// live one within half a quantization step per vertex it touches (see the `resume` module
    /// notes).
    ///
    /// Its edits from now on are made as its op. Its journal goes on from what the recovery file
    /// holds: [`Project::journal_landed`] counts on from [`OpenStroke::landed`], its dabs not
    /// landed are in the file already and never written again, its pen-down stays the save point
    /// its batches name, and its rows keep their format, whatever [`Project::set_dab_format`]
    /// says now. Its pen-down shapes are charged to the budget as a live stroke's are. Once every
    /// dab has landed, end it with [`Project::end_unrecorded_stroke`]: the journal holds no pen
    /// samples to record.
    ///
    /// A stroke never resumed has ended once another op begins: [`Project::begin_stroke`] closes
    /// it in the file, its dabs not landed dropped.
    ///
    /// # Errors
    ///
    /// [`NotResumed`], with `stroke` given back and the project as it was, for
    /// [`ProjectError::NotResumable`] if another op was handed out since the restore (begun,
    /// edited as or reserved, even one ended or let go since), the project has moved onto another
    /// recovery file (a save or a roll-over), or `stroke` isn't this project's; or for
    /// [`ProjectError::Cache`] if a shape can't be installed (evaluating a tile's parents
    /// fails), which a call made again may get past.
    pub fn resume_stroke(&mut self, stroke: OpenStroke) -> Result<(), NotResumed> {
        let op = stroke.op();
        if !stroke.found_in(&self.store) || self.store.op_handed_out().is_some() {
            return Err(NotResumed::new(stroke, ProjectError::NotResumable(op)));
        }
        // The test hook's shape, found by its index before the loop, so the loop carries no index.
        #[cfg(test)]
        let failing = self
            .fail_install_at
            .and_then(|nth| stroke.shapes().get(nth))
            .map(|shape| shape.key);
        let mut shapes = stroke.shapes().iter();
        let installed = shapes.try_for_each(|shape| {
            #[cfg(test)]
            if failing == Some(shape.key) {
                self.fail_install_at = None;
                return Err(CacheError::NotATile(shape.key));
            }
            self.install_shape(shape)
        });
        if let Err(e) = installed {
            self.store.release_pen_down();
            return Err(NotResumed::new(stroke, e.into()));
        }
        let seeded = self.store.resume_journal(
            op,
            stroke.landed(),
            stroke.dabs(),
            stroke.pen_down(),
            stroke.dab_format(),
        );
        if let Err(e) = seeded {
            self.store.release_pen_down();
            return Err(NotResumed::new(stroke, e.into()));
        }
        if let Err(e) = self.resume_op(op) {
            // Dropped, not ended: ending it would write a batch counting every dab landed.
            self.store.discard_journal();
            self.store.release_pen_down();
            return Err(NotResumed::new(stroke, e.into()));
        }
        self.stroke = Some(op);
        self.unresumed = None;
        Ok(())
    }

    /// Installs `shape`, one of a restored stroke's
    /// ([`TileStore::install_pen_down`](crate::tiles::TileStore::install_pen_down)), a step of
    /// [`Project::resume_stroke`].
    ///
    /// # Errors
    ///
    /// As [`TileStore::install_pen_down`](crate::tiles::TileStore::install_pen_down).
    pub(super) fn install_shape(&mut self, shape: &FoundShape) -> Result<(), CacheError> {
        self.store
            .install_pen_down(shape.key, shape.corrections.clone(), shape.op)
    }

    /// Makes `op` the store's op again
    /// ([`TileStore::resume_op`](crate::tiles::TileStore::resume_op)), the last step of
    /// [`Project::resume_stroke`].
    ///
    /// # Errors
    ///
    /// As [`TileStore::resume_op`](crate::tiles::TileStore::resume_op).
    pub(super) fn resume_op(&mut self, op: OpId) -> Result<(), CacheError> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_resume) {
            return Err(CacheError::OpBegun(op));
        }
        self.store.resume_op(op)
    }

    /// The version of the layout of the rows the app journals ([`Project::journal_placed`]): every
    /// stroke begun from now on takes it, every dab batch of the stroke carries it
    /// ([`crate::DabBatch::dab_format`]), and a restore reports it, so the app can refuse rows of
    /// a layout it doesn't read. The store never reads it. It is 0 until set, which means unset: a
    /// restore must refuse rows of format 0. The app sets it right after every open, new or
    /// restore (`stroke::DAB_FORMAT`).
    pub const fn set_dab_format(&mut self, dab_format: u32) {
        self.store.set_dab_format(dab_format);
    }

    /// Dabs the open stroke placed, the next in placement order (symmetry copies included), as
    /// opaque rows, each a `stroke::Dab`'s bytes: kept for the next flush to journal into the
    /// recovery file, with the count landed at its cut (see [`crate::DabBatch`]).
    ///
    /// The cost here is pushing the rows; the flush copies those not yet written on this thread
    /// at its snapshot, and encodes them on the writer thread. The rows stay in memory until
    /// [`Project::journal_landed`] passes them: the landing's backlog.
    ///
    /// # Errors
    ///
    /// [`ProjectError::NoStroke`] if no stroke is open; [`ProjectError::Journal`] with
    /// [`JournalCountError::PastLastIndex`](crate::JournalCountError::PastLastIndex) if the
    /// stroke's dabs would run past the last index a `u32` counts. Nothing is kept then.
    pub fn journal_placed(&mut self, placed: &[[u8; DAB_BYTES]]) -> Result<(), ProjectError> {
        let op = self.stroke.ok_or(ProjectError::NoStroke)?;
        self.store.journal(op, placed).map_err(journal_refused)
    }

    /// How many of the open stroke's dabs have landed: the app passes it with each landing, the
    /// count that landing reaches, on the owner thread, so a flush's cut always pairs the store's
    /// tiles with the count that made them.
    ///
    /// What matters is that no tick (so no cut) falls between a landing's edits and its count,
    /// in either order: count first, a cut between would count dabs landed that its tiles lack,
    /// and a restore would lose them; edits first, it would name tiles holding dabs its count
    /// leaves out, and a restore would land them twice. A dab landed and not yet written is never
    /// written. A stroke that ends with dabs not landed counts them landed:
    /// they were dropped, and a restore never lands them.
    ///
    /// # Errors
    ///
    /// [`ProjectError::NoStroke`] if no stroke is open; [`ProjectError::Journal`] with
    /// [`JournalCountError::LandedBack`](crate::JournalCountError::LandedBack) for a count below
    /// the last, or
    /// [`JournalCountError::LandedPastPlaced`](crate::JournalCountError::LandedPastPlaced) past the
    /// dabs journalled. Nothing changes then.
    pub fn journal_landed(&mut self, landed: u32) -> Result<(), ProjectError> {
        let op = self.stroke.ok_or(ProjectError::NoStroke)?;
        self.store
            .journal_landed(op, landed)
            .map_err(journal_refused)
    }

    /// Asks for a flush now, at `now`, whatever [`FLUSH_INTERVAL`](crate::FLUSH_INTERVAL) says:
    /// every edit and journal row made so far goes to the recovery file in a flush that starts
    /// at once, or is queued behind the flush in flight (see [`RecoveryWriter::request`]).
    /// Never blocks: the app polls ([`Project::poll`]) until [`Project::flushed`] says it is
    /// durable. For a close that lands and saves with no window, which must know the recovery
    /// file holds everything before it lets the window go.
    ///
    /// The 1 s cadence is kept: a flush started here counts as the latest start, as a tick's
    /// does, so the next timed one comes [`FLUSH_INTERVAL`](crate::FLUSH_INTERVAL) after it. With
    /// nothing waiting, it starts nothing.
    ///
    /// # Errors
    ///
    /// As [`Project::tick`]: a writer that fails is replaced by a roll-over (whose fresh file
    /// holds every edit in a durable save point), and a roll-over that can't be made pauses
    /// protection ([`ProjectError::ProtectionPaused`]).
    pub fn request_flush(&mut self, now: Instant) -> Result<(), ProjectError> {
        self.with_writer(|writer, store| writer.request(store, now), Retry::Due(now))?;
        self.report()
    }

    /// Whether every edit and journal row the project holds is in a durable save point of its
    /// recovery file: the writer has no flush in flight or queued, hasn't failed, and owes no
    /// work (no edit or tile not yet flushed, no dab batch due, no stroke record waiting).
    ///
    /// False while crash protection is paused ([`Project::protection`]) or the writer has
    /// failed (the next call rolls over); false again after any edit or row made since, until a
    /// flush holds it. Reads state only: the app calls [`Project::poll`] between checks to move
    /// the flush on.
    ///
    /// Durable isn't usable: rows of an open stroke whose pen-down is unknown are durable, and a
    /// restore can't land them ([`Project::open_stroke_restorable`] says which).
    #[must_use]
    pub fn flushed(&self) -> bool {
        self.paused.is_none()
            && self
                .writer
                .as_ref()
                .is_some_and(|writer| writer.is_flushed(&self.store))
    }
}

/// What the project answers for a call its store's journal refused: [`ProjectError::NoStroke`]
/// for no journal open, [`ProjectError::Journal`] for a count refused.
const fn journal_refused(refused: JournalRefused) -> ProjectError {
    match refused {
        JournalRefused::NotOpen => ProjectError::NoStroke,
        JournalRefused::Count(count) => ProjectError::Journal(count),
    }
}
