//! The finisher: the close bar's Save while dabs still land lets the window go, and the process
//! lands and saves with no window.
//!
//! The dab journal spec, "Closing" and "Finishing unseen"; the [`super`] module notes, "Save on
//! the close bar while dabs are still landing" and "A finisher can be asked to stop".
//!
//! In the window, the close bar's Save asks whether the journal can hold everything the save does
//! ([`Owner::may_finish_unseen`]), flushes, and asks the window to go once that is durable
//! ([`Owner::journal_step`]). With no window, [`Owner::finish_unseen_unless`] lands, saves and
//! closes, or stops for a takeover ([`Owner::stop_unseen`]).

use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};

use super::{AfterSave, Brush, Owner, SaveStep, Unseen, Uploads, WindowRequest, dabs_dropped};
use crate::said_too;
use crate::saving::{self, SaveFailure};

impl<B: Brush, U: Uploads> Owner<B, U> {
    /// Whether the close bar's Save at `now` can let the window go before the save is written, the
    /// landing and the save finishing unseen: dabs are still landing, the brush journals them, the
    /// pen holds none back, the project has a file to save to, and the journal holds everything
    /// the save does ([`Owner::journal_holds_it`]). Otherwise the window waits for the save, as for
    /// any: a pen still down or a stroke not yet routed needs frames.
    pub(super) fn may_finish_unseen(&self, now: Instant) -> bool {
        let landing = self.brush.journals() && self.brush.landing_for_save() > 0;
        let pen = self.brush.pen_down() || self.pen_queued(now);
        let saved_to = self.project.path().is_some();
        landing && !pen && saved_to && self.saving.stage().is_none() && self.journal_holds_it()
    }

    /// Whether a durable journal would keep everything the save holds, as far as the frame loop
    /// knows: crash protection is running, no flush has failed or fallen short since the last
    /// save, the open stroke's pen-down surface is (or will be) in a save point
    /// ([`Project::open_stroke_restorable`](store::Project::open_stroke_restorable)), and the brush
    /// has placed every pen event the save holds as dabs. A stroke waiting behind the one
    /// landing is raw pen samples ([`Brush::unplaced_for_save`]), which no journal holds until
    /// they are placed; with crash protection paused, or a flush failed or short, the journal
    /// can't be trusted to keep what a failure would leave; and a journal whose pen-down is
    /// unknown restores nothing, which not every way there reports (a cut failing as the stroke
    /// began, a roll-over that couldn't carry it), so
    /// [`Project::flushed`](store::Project::flushed) alone can't say it.
    fn journal_holds_it(&self) -> bool {
        let guarded = self.project.protection().is_active() && self.exposed.is_none();
        guarded && self.project.open_stroke_restorable() && self.brush.unplaced_for_save() == 0
    }

    /// Asks the project at `now` for a flush of every edit and every dab journalled so far, so the
    /// window can go once it is durable ([`Owner::journal_step`]). A project that can't take it
    /// leaves the save to finish in the window, as any.
    pub(super) fn flush_to_finish(&mut self, now: Instant) {
        let asked = self.project.request_flush(now);
        self.after_save = match self.heard(asked, "asking for a flush to close") {
            Ok(()) if self.journal_holds_it() => AfterSave::Unseen(Unseen::Flushing),
            Ok(()) => AfterSave::Exit,
            Err(error) => {
                saving::log(&format!("brushcraft: {error:#}"));
                AfterSave::Exit
            }
        };
    }

    /// A step of the wait for the journal, in place of the save's landing while the window is to
    /// go unseen: the recovery writer polled (the save's landing holds the frames' landing back,
    /// so nothing new lands meanwhile), and once every edit and dab placed is in a durable save
    /// point, the window asked to go ([`WindowRequest::Finish`]). Expected: milliseconds, one
    /// flush's syncs. Until then each pass asks for a flush again
    /// ([`Project::request_flush`](store::Project::request_flush)): that starts nothing with
    /// nothing waiting, and queues at most one behind the flush in flight, so whatever else
    /// edits the project meanwhile (a store job, say) is taken too.
    ///
    /// If the journal stops holding everything the save does ([`Owner::journal_holds_it`]:
    /// protection paused, a flush failed or short, the pen-down unknown, a pen event not placed),
    /// the save goes on in the window instead, landing and then writing as any, and the window
    /// waits for it.
    pub(super) fn journal_step(&mut self, unseen: Unseen) -> SaveStep {
        if unseen == Unseen::Asked {
            return SaveStep::Idle;
        }
        let polled = self.project.poll();
        if let Err(error) = self.heard(polled, "polling the project") {
            saving::log(&format!("brushcraft: {error:#}"));
            self.after_save = AfterSave::Exit;
            return SaveStep::Worked;
        }
        if !self.journal_holds_it() {
            self.after_save = AfterSave::Exit;
            return SaveStep::Worked;
        }
        if self.project.flushed() {
            self.after_save = AfterSave::Unseen(Unseen::Asked);
            self.request = Some(WindowRequest::Finish);
            return SaveStep::Worked;
        }
        self.flush_to_finish(Instant::now());
        SaveStep::Waiting
    }

    /// Finishes unseen what the close bar's Save left when the window went at once
    /// ([`WindowRequest::Finish`]), with no window and no frames (the dab journal spec, "Finishing
    /// unseen"): the brush lands everything it holds in full batches ([`Brush::close_step`], up
    /// to [`crate::sculpting::WAITED_BATCH`] of finest work each), the project ticked and polled
    /// between them, so the journal and the landed tiles keep reaching the recovery file at least
    /// once a second and a kill at any moment loses about a second; then the save under way is
    /// written, and the project closes, its recovery file removed once the save holds everything.
    ///
    /// **A failure** (a landing that fails, dabs dropped, a save that fails, a full disk) keeps
    /// the recovery file, which holds the work: opening the project brings it back. Nobody is at a
    /// window to read it, so it is said through `notify`, as a desktop notification's summary and
    /// body ([`SaveFailure::notice`]), once the project has closed, so it says what the recovery
    /// file kept; and it is the run's error, as a close's is.
    ///
    /// **Asked to stop** (a takeover, [`crate::takeover`]): before each batch it looks whether
    /// `asked` says so. Once it does, the finisher stops where it is ([`Owner::stop_unseen`]): it
    /// flushes, waits for the save point to be durable, and closes the project keeping the
    /// recovery file, which holds every dab placed and the count of those landed, for the opener to
    /// restore. Nothing is saved and nobody is notified: [`Finished::Stopped`].
    ///
    /// # Errors
    ///
    /// As [`Owner::close`], and any error of the landing's steps or the upkeep between them; a
    /// stop has none of its own ([`Owner::stop_unseen`]).
    pub(crate) fn finish_unseen_unless(
        mut self,
        notify: impl FnOnce(&str, &str),
        mut asked: impl FnMut() -> bool,
    ) -> Result<Finished> {
        let landing = self.land_unseen(&mut asked);
        if matches!(landing, Ok(LandedUnseen::Stopped)) {
            return Ok(self.stop_unseen());
        }
        let landed = said_too(landing.map(drop), self.flush_brush());
        let left = self.landing();
        let all = if left == 0 {
            Ok(())
        } else {
            Err(anyhow!("not saved: {left} dabs the brush held didn't land"))
        };
        let landed = said_too(landed, all);
        let whole = landed.is_ok();
        let saved = said_too(landed, self.last_save(whole));
        let (saved, failure) = match saved {
            Err(error) => {
                let failure = self.failure_of(&error);
                (Err(error), Some(failure))
            }
            // Saved, but with dabs dropped that nobody was told of: said as the window's save says
            // it, and the run's error.
            Ok(()) => match self.brush.dropped_since(self.told_drops) {
                Ok(()) => (Ok(()), None),
                Err(dropped) => {
                    let failure = SaveFailure::stroke_lost_but_saved();
                    (Err(dabs_dropped(dropped)), Some(failure))
                }
            },
        };
        let name = self.saving.name().to_owned();
        let closed = self.close_project();
        if let Some(failure) = failure {
            // The close decides what the recovery file kept.
            let uncovered = closed.as_ref().err().is_some_and(saving::says_uncovered);
            let failure = if uncovered {
                failure.exposed()
            } else {
                failure
            };
            let (summary, body) = failure.notice(&name);
            notify(&summary, &body);
        }
        said_too(saved, closed).map(|()| Finished::Done)
    }

    /// [`Owner::finish_unseen_unless`], never asked to stop.
    ///
    /// # Errors
    ///
    /// As [`Owner::finish_unseen_unless`].
    #[cfg(test)]
    pub(crate) fn finish_unseen(self, notify: impl FnOnce(&str, &str)) -> Result<()> {
        self.finish_unseen_unless(notify, || false).map(drop)
    }

    /// The finisher stops for a takeover: flushes every edit and dab journalled so far, waits for
    /// that save point to be durable ([`Owner::flush_to_stop`]), and closes the project keeping
    /// its recovery file, which releases both files' locks. The dabs the brush still holds are
    /// left to the journal: the opener restores them, and lands them.
    ///
    /// Nothing here fails the run: a takeover is no failure. A flush that couldn't be made
    /// durable (crash protection paused, the writer failed), or a close that went wrong, leaves the
    /// recovery file at its last durable save point, as a crash would: said on stderr, and in
    /// [`Finished::Stopped`]'s `flushed`.
    fn stop_unseen(mut self) -> Finished {
        let mut flushed = self.flush_to_stop();
        let Self {
            mut project,
            cursor,
            ..
        } = self;
        project.store().unsubscribe_stale(cursor);
        if let Err(error) = project.close(true) {
            let why = anyhow::Error::new(error);
            saving::log(&format!(
                "brushcraft: closing the project to hand it over: {why:#}"
            ));
            flushed = false;
        }
        Finished::Stopped { flushed }
    }

    /// Flushes every edit and dab journalled so far, and waits until that is in a durable save
    /// point ([`Project::flushed`](store::Project::flushed)), for [`STOP_FLUSH_LIMIT`] at most:
    /// whether it got there. Crash protection paused or failing ends the wait at once: nothing
    /// more can be made durable.
    ///
    /// The limit is this wait's alone: [`Project::close`](store::Project::close) after it waits for
    /// the flushes still in flight, as long as a stalled disk holds them, and the opener shows
    /// its waiting bar meanwhile.
    fn flush_to_stop(&mut self) -> bool {
        let began = Instant::now();
        loop {
            if self.project.flushed() {
                return true;
            }
            let guarded = self.project.protection().is_active() && self.exposed.is_none();
            if !guarded || began.elapsed() >= STOP_FLUSH_LIMIT {
                return false;
            }
            let asked = self.project.request_flush(Instant::now());
            let polled = self
                .heard(asked, "asking for a flush to stop")
                .and_then(|()| {
                    let polled = self.project.poll();
                    self.heard(polled, "polling the project")
                });
            if let Err(error) = polled {
                saving::log(&format!("brushcraft: {error:#}"));
                return false;
            }
            std::thread::sleep(STOP_FLUSH_POLL);
        }
    }

    /// The finisher's landing ([`Owner::finish_unseen_unless`]): the pen's samples no longer taken,
    /// then batch after batch of what the brush holds until it holds nothing, the project's
    /// recovery writer tended before each and once after.
    ///
    /// A landing error doesn't stop it: what is left lands all the same, ticked as ever, so the
    /// recovery file keeps up with it whatever the save then makes of the error. An error tending
    /// the recovery writer does (the store is broken): the recovery file stays at its last durable
    /// save point, as after a crash. It also stops once [`FINISH_STALLS`] batches in a row have
    /// changed nothing, which leaves what is left to [`Owner::flush_brush`].
    ///
    /// # Errors
    ///
    /// The first of [`Owner::request_close`], [`Owner::close_step`] and [`Owner::tend_recovery`]
    /// that failed, those after it as its context.
    ///
    /// **Asked to stop:** `asked` is looked at before each batch, the recovery writer just
    /// tended; once it says so the landing ends there, [`LandedUnseen::Stopped`], and an error met
    /// before it is said on stderr alone: the recovery file holds the work, for the opener.
    fn land_unseen(&mut self, asked: &mut impl FnMut() -> bool) -> Result<LandedUnseen> {
        let mut outcome = self.request_close();
        let mut stalls = 0;
        loop {
            if let Err(error) = self.tend_recovery(Instant::now()) {
                return said_too(outcome, Err(error)).map(|()| LandedUnseen::Landed);
            }
            if asked() {
                if let Err(error) = outcome {
                    saving::log(&format!(
                        "brushcraft: stopping for a takeover after: {error:#}"
                    ));
                }
                return Ok(LandedUnseen::Stopped);
            }
            let before = self.landing();
            if before == 0 || stalls == FINISH_STALLS {
                break;
            }
            let landed = match self.close_step() {
                Ok(landed) => landed,
                Err(error) => {
                    outcome = said_too(outcome, Err(error));
                    false
                }
            };
            let moved = landed || self.landing() != before;
            stalls = if moved { 0 } else { stalls + 1 };
        }
        said_too(outcome, self.tend_recovery(Instant::now())).map(|()| LandedUnseen::Landed)
    }
}

/// How long a finisher stopping for a takeover waits for its flush ([`Owner::flush_to_stop`]).
///
/// A flush's syncs take milliseconds, so this bounds only a disk that stalls, the opener waiting
/// meanwhile. It bounds the wait, not the stop: the close that follows
/// ([`Project::close`](store::Project::close)) finishes the writer's flushes, and blocks as long as
/// the disk stalls; the opener shows its waiting bar ("Finishing your last session") until the
/// locks are free.
pub(super) const STOP_FLUSH_LIMIT: Duration = Duration::from_secs(3);

/// How often a finisher stopping looks at its flush.
const STOP_FLUSH_POLL: Duration = Duration::from_millis(1);

/// How the finisher's landing ended ([`Owner::land_unseen`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LandedUnseen {
    /// It landed what the brush held, or as much as it could.
    Landed,
    /// It was asked to stop, and stopped before its next batch.
    Stopped,
}

/// How a finisher ended ([`Owner::finish_unseen_unless`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finished {
    /// It landed and saved: the recovery file is gone.
    Done,
    /// It was asked to stop and did, its project closed, the recovery file kept for the opener.
    Stopped {
        /// Whether everything placed was in a durable save point as it closed; if not, the
        /// recovery file holds its last durable save point, as after a crash.
        flushed: bool,
    },
}

/// How many of the finisher's batches in a row may change nothing before it stops trying.
///
/// A batch of [`Owner::land_unseen`] that can land nothing shouldn't happen (close steps wait for a
/// stroke's cut): this guards against it, so the finisher never spins.
const FINISH_STALLS: u32 = 3;
