//! The open strokes' journals and landing.
//!
//! Whether a stroke may land, its pen-down cut as it begins, the dabs it places and lands,
//! strokes begun behind one still landing, resuming those a restore found, and the flush a close
//! asks for to make it all durable.
//!
//! See [`crate::DabBatch`] for the record, and `crate::recovery` for the cut a stroke's first
//! landing waits for.
//!
//! # A stroke begun while the last one lands
//!
//! A pen comes down while the open stroke's dabs still land: the new stroke begins behind it at
//! once ([`Project::begin_stroke_after`]), its op reserved, ordered after the open one's but not
//! current, and its journal open, so the dabs it places are journalled as it places them
//! ([`Project::journal_placed_after`]) and reach the recovery file within a second. It lands
//! nothing yet: one stroke writes at a time, in op order. Once the stroke before has ended
//! ([`Project::end_stroke`], which then ends its op too), its op begins
//! ([`Project::begin_reserved_stroke`]) and its pen-down cut is taken as any stroke's is, its
//! first landing waiting for it; from then on it is the open stroke. Until that cut is durable its
//! batches name the stroke before ([`PenDown::After`]), then that save point. A stroke reserved
//! that won't land, as when a stroke before it fails, is let go with every stroke behind it
//! ([`Project::drop_stroke_after`]).
//!
//! # Resuming the strokes a restore found
//!
//! A restore finds the strokes still landing in op order ([`Project::take_open_strokes`]), and
//! they resume one at a time, in that order, each once the one before has landed and ended
//! ([`Project::resume_stroke`]): one stroke writes at a time, as live. A stroke whose batches
//! name a save point resumes on it, its pen-down shapes read from there, and may land at once:
//! the restored state is its cut. One whose batches name the stroke before it ([`PenDown::After`])
//! has landed nothing, and its surface is what that stroke left, the store's state as it resumes;
//! so it resumes as a reserved stroke begins its op ([`Project::begin_reserved_stroke`]): its
//! pen-down cut is taken then, it lands nothing until that cut is taken, and its batches go on
//! naming the stroke before until the cut is durable. A crash before then restores the chain as
//! it was, or, once the stroke before has closed in the file, this one on the tiles that hold it.

use std::time::Instant;

use super::resume::{FoundShape, NotResumed, OpenStroke};
use super::{Project, ProjectError, Retry};
use crate::OpId;
use crate::journal::{DAB_BYTES, PenDown};
use crate::recovery::RecoveryWriter;
use crate::tiles::{CacheError, JournalRefused, NotBegun, PenDownPoint, ReservedOp};

impl Project {
    /// A pen came down while the open stroke's dabs still land: a stroke begins behind the newest
    /// open stroke, its op reserved (see the module notes and
    /// [`TileStore::reserve_op`](crate::tiles::TileStore::reserve_op)), and its journal opens,
    /// its batches naming that stroke as where its surface is until its own cut is durable. Its
    /// dabs, placed on the futures installed for its op, are journalled at once
    /// ([`Project::journal_placed_after`]); none lands until its op begins
    /// ([`Project::begin_reserved_stroke`]), once every stroke before it has ended.
    ///
    /// Give the op back: begin it, or let it go ([`Project::drop_stroke_after`]).
    ///
    /// # Errors
    ///
    /// Nothing is reserved on an error: [`ProjectError::NoStroke`] with no stroke open to begin
    /// after (begin one of its own, [`Project::begin_stroke`]); [`ProjectError::Cache`] with
    /// [`CacheError::OpenStrokesFull`] when the open strokes' pen-down data holds its share of the
    /// budget (the stroke waits for those before it to land, and begins its own then), or
    /// [`CacheError::OpsExhausted`] if the site's ops are used up.
    ///
    /// The strokes a restore found and nobody resumed end here, as at [`Project::begin_stroke`]:
    /// none resumes behind a fresh op, and the file says so.
    pub fn begin_stroke_after(&mut self) -> Result<ReservedOp, ProjectError> {
        if self.store.newest_open_stroke().is_none() {
            return Err(ProjectError::NoStroke);
        }
        let reserved = self.store.reserve_op()?;
        self.close_unresumed();
        self.store.begin_journal_after(reserved.op());
        Ok(reserved)
    }

    /// Dabs the stroke reserved as `reserved` placed ([`Project::begin_stroke_after`]), the next
    /// in placement order, as [`Project::journal_placed`] takes the open stroke's: kept for the
    /// next flush to journal, none of them landed. Once its op has begun, its dabs go through
    /// [`Project::journal_placed`].
    ///
    /// # Errors
    ///
    /// [`ProjectError::NoStroke`] if `reserved` isn't reserved on this project: begun, let go
    /// (with a stroke before it, too), or of another project; [`ProjectError::Journal`] as for
    /// [`Project::journal_placed`]. Nothing is kept then.
    pub fn journal_placed_after(
        &mut self,
        reserved: &ReservedOp,
        placed: &[[u8; DAB_BYTES]],
    ) -> Result<(), ProjectError> {
        if !self.store.holds_reserved(reserved) {
            return Err(ProjectError::NoStroke);
        }
        self.store
            .journal(reserved.op(), placed)
            .map_err(journal_refused)
    }

    /// The stroke reserved as `reserved` begins its op, every stroke before it having ended: it
    /// is the open stroke from now on, as one [`Project::begin_stroke`] begins, its pen-down
    /// shapes the futures installed for it. Its pen-down cut is taken now, as for any stroke
    /// (see [`Project::begin_stroke`]): it **lands nothing until that cut is taken**
    /// ([`Project::may_land`]), and its batches go on naming the stroke before until the cut is
    /// durable, then name its save point.
    ///
    /// The stroke begins whatever the cut meets, as [`Project::begin_stroke`]'s does.
    ///
    /// # Errors
    ///
    /// [`NotBegun`], the op handed back with nothing changed, for what
    /// [`TileStore::begin_reserved_op`](crate::tiles::TileStore::begin_reserved_op) refuses:
    /// the stroke before is still open ([`CacheError::OpBegun`]; end it first), an older stroke
    /// is reserved ([`CacheError::OpReserved`]), or `reserved` isn't reserved on this project.
    /// Begin it later, or let it go ([`Project::drop_stroke_after`]).
    pub fn begin_reserved_stroke(&mut self, reserved: ReservedOp) -> Result<OpId, NotBegun> {
        // A refusal is handed back as it is, the token in it.
        let begun = self.store.begin_reserved_op(reserved);
        begun.inspect(|&op| {
            // The op is current and has edited nothing: the cut holds every earlier op's edits
            // and none of its own, as for a stroke begun on its own. A roll-over on the way
            // carries it, which settles its pen-down there, and that stands.
            let pen_down = self.cut_before_op(Instant::now());
            self.store.journal_op_began(op, pen_down);
            self.stroke = Some(op);
        })
    }

    /// Lets the stroke reserved as `reserved` go before its op begins, and every stroke reserved
    /// behind it, whose surfaces were futures over its dabs: none of their dabs ever lands, and
    /// their journals close, the last batch of each counting every dab landed (see
    /// [`TileStore::drop_reserved_ops_from`](crate::tiles::TileStore::drop_reserved_ops_from)).
    /// Returns the ops let go, in op order; the tokens of those behind it are given back here
    /// too, then letting nothing more go.
    ///
    /// # Panics
    ///
    /// While a reserved op's pen-down shapes are read, as
    /// [`TileStore::drop_reserved_op`](crate::tiles::TileStore::drop_reserved_op).
    pub fn drop_stroke_after(&mut self, reserved: ReservedOp) -> Vec<OpId> {
        self.store.drop_reserved_op(reserved)
    }

    /// Whether the stroke made as `op` may land, edit the store: it is the open stroke, its op
    /// begun, and its pen-down flush has taken its cut ([`Project::may_land`]). False for a
    /// stroke reserved behind it, which lands nothing until its op begins, and for a stroke not
    /// open.
    #[must_use]
    pub fn stroke_may_land(&self, op: OpId) -> bool {
        self.store.stroke_may_land(op)
    }

    /// The open stroke's journal closes, and, with a stroke reserved behind it, its op ends, so
    /// that stroke's may begin: a step of [`Project::end_stroke`]. The journal closes first, so
    /// none is open for the op as it ends, as the store asks.
    pub(super) fn end_journal_and_op(&mut self) {
        self.store.end_journal();
        if self.store.oldest_reserved_op().is_some() {
            self.store.end_op();
        }
    }

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
    /// A restore may find several strokes open: they resume in op order, one at a time, each once
    /// the one before it has landed every dab and ended (see the module notes). A stroke begun
    /// behind the one before it ([`OpenStroke::after`]) resumes on what that one left, the store's
    /// state now, and installs no shapes: it has landed nothing. Its pen-down cut is taken now, as
    /// for a stroke reserved whose op begins, and it **lands nothing until that cut is taken**
    /// ([`Project::may_land`], [`Project::wait_to_land`]); its batches go on naming the stroke
    /// before until the cut is durable, then name its save point.
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
    /// [`ProjectError::NotResumable`] if a fresh op was handed out since the restore (begun,
    /// edited as or reserved, even one ended or let go since), a stroke is open (the one before
    /// it, still landing: end it first), the stroke before it ([`OpenStroke::after`]) hasn't been
    /// resumed and ended, a stroke at or after it was resumed already, the project has moved onto
    /// another recovery file (a save or a roll-over), or `stroke` isn't this project's; or for
    /// [`ProjectError::Cache`] if a shape can't be installed (evaluating a tile's parents
    /// fails), which a call made again may get past.
    pub fn resume_stroke(&mut self, stroke: OpenStroke) -> Result<(), NotResumed> {
        let op = stroke.op();
        let earlier_ended = stroke.after().is_none_or(|earlier| {
            self.store.last_resumed_op() == Some(earlier) && !self.store.journal_open_for(earlier)
        });
        let resumable = stroke.found_in(&self.store)
            && self.stroke.is_none()
            && earlier_ended
            && self.store.may_resume(op).is_ok();
        if !resumable {
            return Err(NotResumed::new(stroke, ProjectError::NotResumable(op)));
        }
        if let PenDown::After(earlier) = stroke.pen_down() {
            return self.resume_after(stroke, earlier);
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
            self.store.discard_journal(op);
            self.store.release_pen_down();
            return Err(NotResumed::new(stroke, e.into()));
        }
        self.stroke = Some(op);
        self.unresumed.retain(|unresumed| unresumed.op != op);
        Ok(())
    }

    /// [`Project::resume_stroke`] for `stroke`, whose batches name `earlier`, the stroke it began
    /// behind ([`PenDown::After`]): it has landed nothing, and its surface is what `earlier` left,
    /// the store's state now (see the module notes).
    ///
    /// Its journal opens as a reserved stroke's, its dabs already in the file and its batches
    /// naming `earlier`; its op becomes the store's; then its pen-down cut is taken, as
    /// [`Project::begin_reserved_stroke`] takes one. The journal is open before the cut, so a
    /// roll-over the cut makes carries the stroke, its dabs written again into the fresh file
    /// naming the pen-down cut there, which then stands.
    ///
    /// # Errors
    ///
    /// As [`Project::resume_stroke`], once its checks have passed.
    fn resume_after(&mut self, stroke: OpenStroke, earlier: OpId) -> Result<(), NotResumed> {
        let op = stroke.op();
        let opened =
            self.store
                .resume_journal_after(op, stroke.dabs(), earlier, stroke.dab_format());
        if let Err(e) = opened {
            return Err(NotResumed::new(stroke, e.into()));
        }
        if let Err(e) = self.resume_op(op) {
            self.store.discard_journal(op);
            return Err(NotResumed::new(stroke, e.into()));
        }
        // The op is current and has edited nothing, and every edit of the stroke before is in
        // the store: the cut holds the surface it left and none of this stroke's own.
        let pen_down = self.cut_before_op(Instant::now());
        self.store.journal_op_began(op, pen_down);
        self.stroke = Some(op);
        self.unresumed.retain(|unresumed| unresumed.op != op);
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
