//! An explicit save, from Ctrl+S to the save point: where it has got to, and what the window's
//! title and the overlay say about it.
//!
//! # What a save holds
//!
//! Everything drawn before Ctrl+S, its dabs still catching up included. So a save goes through
//! three [stages](Stage), driven by [`crate::frame::Owner::save_step`], with frames and the
//! window's events between its steps:
//!
//! 1. **The pen** ([`Stage::PenDown`]): a stroke the pen is drawing is the user's to end, and a
//!    save can't cut it, as an op ends with its stroke. So the save waits for its `Up`.
//! 2. **Landing** ([`Stage::Landing`]): what the brush holds lands in batches of up to 100 ms, one
//!    per event-loop pass, as a close's does ([`crate::frame::Brush::save_step`]); the title counts
//!    the dabs down.
//! 3. **Writing** ([`Stage::Writing`]): [`store::Project::save_ahead`] waits for the recovery
//!    writer's flush and copies the edited tiles' records in slices of [`SAVE_SLICE`], and then
//!    [`store::Project::save`] does the rest in one go.
//!
//! **A stroke begun after Ctrl+S waits for the save**, as an overlapping stroke waits for the one
//! before it, and isn't in it: the stroke's samples are kept, it begins as soon as the save has
//! been written or has failed, and the unsaved marker comes back with it. So what a save holds is
//! fixed by Ctrl+S, however long the landing takes and however much the user goes on drawing.
//!
//! # Why the writing is in slices
//!
//! The store has one owner thread, and the event loop runs on it. Measured on the benchmark
//! project after the 25 s spiral stroke (29,083 tiles, 456 MB, a loaded machine),
//! [`store::Project::save`] alone blocks for 3.6 s: 2.5 s on a flush that no upkeep had kept up,
//! 1.05 s copying records, 0.11 s for all the rest. Through the stages above the writing takes
//! 1.4 s in 79 steps, the longest 0.2 s, the save itself: the same order as a landing's longest
//! batches (0.17 to 0.23 s in those runs), and the part of a save that can't be cut, as it ends in
//! the store moving onto the new files.
//!
//! # Failing
//!
//! A failed save loses nothing: the edits stay, unsaved, in the recovery file. The failure bar
//! ([`crate::dialogs`]) says why in the user's words ([`SaveFailure`]) until it is answered or a
//! stroke puts it away, the title and the overlay say so until the next Ctrl+S, the log (stderr)
//! has the whole error, and Ctrl+S or the bar's Try again tries again. Dabs dropped by a store
//! error since Ctrl+S, in any stage, fail the save too, as it can't hold everything drawn.
//!
//! The bar's second line says only what is true: not "safe" with crash protection paused, when
//! the edits may be in memory alone, and "other work" with a stroke's dabs gone.
//!
//! **A save that went through without some tiles' latest edits** ([`SaveOutcome::unprotected`]) is
//! a failed save too: it shows no "Saved", the window stays, and the bar says the edits are gone,
//! as they are ([`TilesLost`]). It offers no Try again: nothing is left to save, and nothing brings
//! them back. So does one that dabs dropped before it left a stroke short, in its own words. Old
//! recovery files it couldn't delete fail nothing.
//!
//! **A recovery file that can't keep up** stops no one sculpting
//! ([`crate::frame::Owner::upkeep`]): the log says why, and a bar tells the user once, offering a
//! save. Word of a pause goes by itself if protection comes back; a flush that failed or left tiles
//! out can't say when it is over, so that lasts until a save. Tiles the store says are gone, by
//! whichever route, are kept until a bar has said so, or the next save does. A scripted run has
//! nobody to tell: its close fails instead.
//!
//! # Closing meanwhile
//!
//! A close asked for during a save lets the save finish: the close lands everything held, a
//! stroke that waited included (the pen isn't taken any more), the save holds all of that, whole,
//! and only then does the window go. A second request takes the window at once, and the save is
//! finished as the project closes ([`crate::frame::Owner::close`]), unless dabs were dropped.
//!
//! If that save fails the window stays, saying so: the user asked for it. Ctrl+S tries again, and
//! a further close takes the window, the recovery file kept, the failure an error of the run.
//!
//! That close is a scripted run's. With a person at the window a close never lands anything: it
//! asks, on a bar ([`crate::dialogs`]), and Save on the bar is an ordinary save that can still be
//! cancelled, the window exiting once it has succeeded ([`crate::frame::Owner::ask_close`]). If it
//! fails the window stays and the failure bar takes the close bar's place. With dabs still landing
//! the window doesn't wait for them: once the recovery file holds every dab placed, it goes, and
//! the save is finished unseen ([`crate::frame::Owner::finish_unseen`]), a failure said in a
//! desktop notification instead, the recovery file kept.
//!
//! **Leaving on a failure:** with any failed save told, Ctrl+S's too, a further close asks nothing
//! and leaves ([`crate::frame::Owner::ask_close`]), as above: the recovery file is kept, opening
//! the project brings the work back, and the failure is an error of the run. A save that works, or
//! a new stroke, overtakes what was told, and a close asks again. So does every close while nothing
//! guards the work, crash protection paused or the recovery writer failed: leaving would lose it
//! unasked, so the way out is the close bar, its Save or its Don't save.

use std::error::Error as StdError;
use std::io::{self, Write as _};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Error;
use store::{FileError, ProjectError, Protection, SaveOutcome};
use thiserror::Error;

/// How long a slice of a save's writing runs, between frames and the window's events: about a
/// 60 Hz frame's, as a catch-up's batch ([`crate::sculpting::CATCH_UP_BATCH`]).
pub(crate) const SAVE_SLICE: Duration = Duration::from_millis(12);

/// How finely the title counts the dabs left to land.
const DABS_STEP: usize = 10;

/// How finely the title counts the tiles left to write.
const TILES_STEP: usize = 100;

/// Where a save has got to (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Waiting for the pen to lift.
    PenDown,
    /// What the brush holds is landing.
    Landing,
    /// Into the project file.
    Writing,
}

/// What a pass of the event loop should do after a save's step
/// ([`crate::frame::Owner::save_step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SaveStep {
    /// Nothing changes: no save needs the loop.
    Idle,
    /// A step was taken: come back at once.
    Worked,
    /// Waiting for the recovery writer's thread, which wakes nothing: look again shortly.
    Waiting,
}

/// How the last save ended.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    /// It went through, or there was nothing to save; what to say about it.
    Done(String),
    /// It failed, and why.
    Failed(String),
}

/// What the window's title says: made again only when this changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Title {
    /// Whether it marks the project as having unsaved edits.
    unsaved: bool,
    /// What it says is going on.
    doing: Doing,
}

/// What is going on, as the title and the overlay say it. Counts are rounded up, so the title isn't
/// told every dab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Doing {
    /// Nothing to say.
    Nothing,
    /// The last save failed.
    Failed,
    /// The last save failed as the window closed, which waits for a further close.
    FailedClosing,
    /// A save waits for the pen.
    PenDown,
    /// A save is landing this many dabs.
    Landing(usize),
    /// A save is writing, this many tiles left if it knows.
    Writing(Option<usize>),
    /// The window is closing, this many dabs left to land.
    Closing(usize),
}

/// Dabs a store error dropped ([`crate::frame::Brush::dropped_since`]): what was drawn didn't all
/// land, so a save can't hold it. Its text is the store's error, for the log.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub(crate) struct DabsDropped(pub(crate) String);

/// A save went through without the latest edits of this many tiles, as its outcome said or the
/// store had: for the user, a failed save.
///
/// Those edits are gone ([`SaveOutcome::unprotected`]): memory no longer holds them, and with the
/// rest saved there is nothing for another save to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("the latest edits of {0} tiles are in no file")]
pub(crate) struct TilesLost(pub(crate) usize);

/// Why a save failed, as the user can act on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SaveTrouble {
    /// There is no room on the disk, or in the user's share of it.
    DiskFull,
    /// The system won't let the file be written.
    NoPermission,
    /// Another open of the file holds its lock.
    InUse,
    /// Dabs were dropped, so the stroke they belong to isn't whole.
    StrokeLost,
    /// The save went through, but edits of some tiles are in no file, nor in memory.
    EditsLost,
    /// Anything else: the log says what.
    Other,
}

impl SaveTrouble {
    /// Reads the cause from `error`'s chain: the first link that says one, outermost first.
    pub(crate) fn of(error: &Error) -> Self {
        let mut causes = error.chain().filter_map(Self::of_cause);
        causes.next().unwrap_or(Self::Other)
    }

    /// Reads the cause from `error` and what lies under it, as [`SaveTrouble::of`] does: for an
    /// error no [`anyhow::Error`] holds, as a pause's cause.
    pub(crate) fn of_source(error: &(dyn StdError + 'static)) -> Self {
        let mut link = Some(error);
        while let Some(cause) = link {
            if let Some(trouble) = Self::of_cause(cause) {
                return trouble;
            }
            link = cause.source();
        }
        Self::Other
    }

    /// What `cause`, a link of an error's chain, says by itself.
    fn of_cause(cause: &(dyn StdError + 'static)) -> Option<Self> {
        if cause.is::<DabsDropped>() {
            return Some(Self::StrokeLost);
        }
        if cause.is::<TilesLost>() {
            return Some(Self::EditsLost);
        }
        match cause.downcast_ref() {
            Some(FileError::Locked) => return Some(Self::InUse),
            Some(FileError::ReadOnly) => return Some(Self::NoPermission),
            Some(_) | None => {}
        }
        let kind = cause.downcast_ref::<io::Error>()?.kind();
        if matches!(
            kind,
            io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
        ) {
            return Some(Self::DiskFull);
        }
        let denied = matches!(
            kind,
            io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
        );
        denied.then_some(Self::NoPermission)
    }

    /// What follows "Couldn't save: ".
    pub(crate) const fn said(self) -> &'static str {
        match self {
            Self::DiskFull => "the disk is full",
            Self::NoPermission => "this file can't be written",
            Self::InUse => "another app has this file open",
            Self::StrokeLost => "a stroke didn't finish",
            Self::EditsLost => "part of a stroke is gone",
            Self::Other => "something went wrong",
        }
    }
}

/// What a failed save leaves of the work it was to hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Work {
    /// All of it, in the project still open.
    Whole,
    /// All but a stroke that didn't all land, in the project still open.
    LessAStroke,
    /// Found work a copy was to hold: it stays in its recovery file, and isn't what is open.
    Found,
    /// All but edits that are gone, and that is saved: a save that went through without them.
    SavedLessEdits,
}

/// Whether the recovery file is guarding the edits, as the frame loop knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Protected {
    /// It is.
    Yes,
    /// It isn't.
    No,
}

/// Whether a crash would spare what a failed save left unsaved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cover {
    /// The recovery file holds it.
    Protected,
    /// It may be in memory alone: crash protection is paused, or the recovery writer failed.
    Exposed,
}

/// A failed save as the user hears of it: why, and what that leaves. Only what is true of the work
/// is said, so the bar's second line depends on all of this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SaveFailure {
    /// Why it failed.
    trouble: SaveTrouble,
    /// What it leaves.
    work: Work,
    /// Whether that would outlive a crash.
    cover: Cover,
}

impl SaveFailure {
    /// The failure `error`, of a save of a project with `protection`, which leaves `work`.
    ///
    /// **Crash protection** decides "safe". A failed [`store::Project::save`] never says it was
    /// paused, as protection never stops a save, so `protection` is what tells;
    /// [`ProjectError::ProtectionPaused`], [`ProjectError::Recovery`] and
    /// [`ProjectError::Unprotected`] in the chain say the same of edits the recovery file doesn't
    /// hold, whatever `protection` came to since.
    pub(crate) fn of(error: &Error, work: Work, protection: &Protection) -> Self {
        let trouble = SaveTrouble::of(error);
        let exposed = !protection.is_active() || error.chain().any(says_edits_arent_covered);
        Self {
            trouble,
            work: match trouble {
                SaveTrouble::StrokeLost => Work::LessAStroke,
                SaveTrouble::EditsLost => Work::SavedLessEdits,
                SaveTrouble::DiskFull
                | SaveTrouble::NoPermission
                | SaveTrouble::InUse
                | SaveTrouble::Other => work,
            },
            cover: if exposed {
                Cover::Exposed
            } else {
                Cover::Protected
            },
        }
    }

    /// The failure `error` of a copy of found work.
    pub(crate) fn of_copy(error: &Error) -> Self {
        Self {
            trouble: SaveTrouble::of(error),
            work: Work::Found,
            cover: Cover::Protected,
        }
    }

    /// This failure, with the frame loop knowing the recovery file lacks edits, as a failure can't
    /// say: the tiles a flush left out are told of once, and may be long past.
    pub(crate) const fn exposed(self) -> Self {
        Self {
            cover: Cover::Exposed,
            ..self
        }
    }

    /// A save that went through with dabs dropped before it: all but that stroke is saved.
    pub(crate) const fn stroke_lost_but_saved() -> Self {
        Self {
            trouble: SaveTrouble::StrokeLost,
            work: Work::SavedLessEdits,
            cover: Cover::Protected,
        }
    }

    /// Edits gone between saves, with `protection` running or not: no save has failed, and the
    /// rest is as it was.
    pub(crate) const fn edits_gone(protection: Protected) -> Self {
        Self {
            trouble: SaveTrouble::EditsLost,
            work: Work::LessAStroke,
            cover: match protection {
                Protected::Yes => Cover::Protected,
                Protected::No => Cover::Exposed,
            },
        }
    }

    /// Whether another try can come to anything. With edits gone and the rest saved there is
    /// nothing left for a save to do: it would find no edits, and say nothing.
    pub(crate) const fn can_be_tried_again(self) -> bool {
        !matches!(self.work, Work::SavedLessEdits)
            && !matches!(self.trouble, SaveTrouble::EditsLost)
    }

    /// Whether the recovery file holds what this left unsaved, as far as the failure says.
    pub(crate) const fn covered(self) -> bool {
        matches!(self.cover, Cover::Protected)
    }

    /// Whether it was a copy of found work that failed, not a save of the project.
    pub(crate) const fn of_found_work(self) -> bool {
        matches!(self.work, Work::Found)
    }

    /// What the bar's title says.
    pub(crate) fn title(self) -> String {
        if matches!(self.work, Work::SavedLessEdits) && self.trouble == SaveTrouble::StrokeLost {
            return "A stroke didn't finish".to_owned();
        }
        if matches!(self.work, Work::LessAStroke) && self.trouble == SaveTrouble::EditsLost {
            return "Part of a stroke is gone".to_owned();
        }
        format!("Couldn't save: {}", self.trouble.said())
    }

    /// What the finisher's desktop notification says of this failure of a save of the project
    /// named `name` (the dab journal spec, "Finishing unseen"): its summary and body, which say
    /// only what is true, as the bar does. The reason comes first in the body, in the user's
    /// words.
    pub(crate) fn notice(self, name: &str) -> (String, String) {
        if matches!(self.work, Work::SavedLessEdits) {
            let summary = format!("BrushCraft saved {name}, less part of a stroke");
            let body = "Part of a stroke is gone. The rest of your work was saved.".to_owned();
            return (summary, body);
        }
        let summary = format!("BrushCraft couldn't save {name}");
        let said = self.trouble.said();
        let mut letters = said.chars();
        let reason: String = letters
            .next()
            .map(|first| first.to_uppercase().chain(letters).collect())
            .unwrap_or_default();
        let rest = match (self.work, self.cover) {
            (_, Cover::Exposed) => {
                "Open it to see what was kept: crash protection was off, so recent work may be lost."
            }
            (Work::LessAStroke, Cover::Protected) => {
                "Your other work is safe and comes back when you open it."
            }
            (Work::Whole | Work::Found | Work::SavedLessEdits, Cover::Protected) => {
                "Your work is safe and comes back when you open it."
            }
        };
        (summary, format!("{reason}. {rest}"))
    }

    /// What the bar says under it, of the work.
    pub(crate) const fn detail(self) -> &'static str {
        match (self.work, self.cover) {
            (Work::Found, _) => "Your unsaved work is safe where it was found.",
            (Work::SavedLessEdits, _) => "The rest of your work was saved.",
            (Work::Whole, Cover::Protected) => "Your work is safe and still open.",
            (Work::LessAStroke, Cover::Protected) => "Your other work is safe and still open.",
            (Work::Whole, Cover::Exposed) => "Your work is still open, but a crash could lose it.",
            (Work::LessAStroke, Cover::Exposed) => {
                "Your other work is still open, but a crash could lose it."
            }
        }
    }
}

/// Whether `error` says the recovery file may not hold the edits, in any link of its chain: a
/// close that failed so leaves a recovery file a finisher's notification can't call safe.
pub(crate) fn says_uncovered(error: &Error) -> bool {
    error.chain().any(says_edits_arent_covered)
}

/// Whether `cause`, a link of an error's chain, says the recovery file may not hold the edits.
fn says_edits_arent_covered(cause: &(dyn StdError + 'static)) -> bool {
    matches!(
        cause.downcast_ref(),
        Some(
            ProjectError::ProtectionPaused(_)
                | ProjectError::Recovery(_)
                | ProjectError::Unprotected(_)
        )
    )
}

/// A project's saves, as the frame loop keeps them.
#[derive(Debug)]
pub(crate) struct Saving {
    /// The project's name, as the title shows it.
    name: String,
    /// The save under way, if one is.
    stage: Option<Stage>,
    /// When the save under way was asked for.
    asked: Instant,
    /// How many times the brush had dropped dabs when it was.
    drops_at: u64,
    /// Why a save failed with the window closing: it stays until a further close.
    failed_closing: Option<String>,
    /// The failed save the user may leave on, until a save or a new stroke overtakes it.
    told: Option<SaveFailure>,
    /// Why a save failed that the window left on, which makes the run fail.
    left_on: Option<String>,
    /// The tiles still to copy, once the copying has begun.
    tiles_left: Option<usize>,
    /// How the last save ended, until the next is asked for.
    outcome: Option<Outcome>,
    /// What the title last said.
    title_says: Option<Title>,
}

impl Saving {
    /// No save yet, of the project at `path`, which names it; "untitled" with none.
    pub(crate) fn new(path: Option<&Path>, now: Instant) -> Self {
        let stem = path.and_then(Path::file_stem);
        Self {
            name: stem.map_or_else(
                || "untitled".to_owned(),
                |s| s.to_string_lossy().into_owned(),
            ),
            stage: None,
            asked: now,
            drops_at: 0,
            failed_closing: None,
            told: None,
            left_on: None,
            tiles_left: None,
            outcome: None,
            title_says: None,
        }
    }

    /// The save under way's stage, if one is.
    pub(crate) const fn stage(&self) -> Option<Stage> {
        self.stage
    }

    /// When the save under way was asked for.
    pub(crate) const fn asked(&self) -> Instant {
        self.asked
    }

    /// How many times the brush had dropped dabs when the save under way was asked for.
    pub(crate) const fn drops_at(&self) -> u64 {
        self.drops_at
    }

    /// Why a save failed with the window closing, if one did and hasn't been tried again.
    pub(crate) fn failed_closing(&self) -> Option<&str> {
        self.failed_closing.as_deref()
    }

    /// The name the project goes by.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Whether the last save went through, or had nothing to do; `None` until one ends.
    pub(crate) const fn last_succeeded(&self) -> Option<bool> {
        match &self.outcome {
            Some(Outcome::Done(_)) => Some(true),
            Some(Outcome::Failed(_)) => Some(false),
            None => None,
        }
    }

    /// A save was asked for at `now`, the brush having dropped dabs `drops` times.
    pub(crate) fn start(&mut self, now: Instant, drops: u64) {
        self.stage = Some(Stage::PenDown);
        self.asked = now;
        self.drops_at = drops;
        self.failed_closing = None;
        self.told = None;
        self.left_on = None;
        self.tiles_left = None;
        self.outcome = None;
    }

    /// The save under way, still waiting for the pen, is given up: nothing has been done, so
    /// there is nothing to say of it.
    pub(crate) const fn let_go(&mut self) {
        self.stage = None;
    }

    /// The save under way has moved on to `stage`.
    pub(crate) const fn reach(&mut self, stage: Stage) {
        self.stage = Some(stage);
    }

    /// `left` tiles are still to copy.
    pub(crate) const fn copying(&mut self, left: usize) {
        self.tiles_left = Some(left);
    }

    /// Ctrl+S found nothing to save, at once or once a stroke of no dabs had ended: the file
    /// doesn't gain an empty save point.
    pub(crate) fn nothing_to_save(&mut self) {
        self.stage = None;
        self.failed_closing = None;
        self.told = None;
        self.outcome = Some(Outcome::Done("nothing to save".to_owned()));
    }

    /// The save under way went through: says so, and how long it took, in the log.
    pub(crate) fn saved(&mut self, outcome: &SaveOutcome, now: Instant) {
        let took = now.saturating_duration_since(self.asked).as_secs_f64();
        let said = said_of(outcome, took);
        log(&format!("brushcraft: {}: {said}", self.name));
        self.stage = None;
        self.told = None;
        self.outcome = Some(Outcome::Done(said));
    }

    /// The save under way failed because of `error`, with the window `closing` or not: says so
    /// in the log.
    pub(crate) fn failed(&mut self, error: &Error, closing: bool) {
        let why = format!("{error:#}");
        log(&format!("brushcraft: {}: save failed: {why}", self.name));
        self.stage = None;
        self.failed_closing = closing.then(|| why.clone());
        self.outcome = Some(Outcome::Failed(why));
    }

    /// The user is being told of `failure`, the save under way's.
    pub(crate) const fn tell(&mut self, failure: SaveFailure) {
        self.told = Some(failure);
    }

    /// The failed save the user may leave on, if none has overtaken it.
    pub(crate) const fn told(&self) -> Option<SaveFailure> {
        self.told
    }

    /// The failure the user was told of can't say "safe" any more.
    pub(crate) fn told_exposed(&mut self) {
        self.told = self.told.map(SaveFailure::exposed);
    }

    /// A new stroke overtook what the user was told: there is new work, and a save may work now.
    pub(crate) const fn told_is_stale(&mut self) {
        self.told = None;
    }

    /// The window is leaving on the failure the user was told of: an error of the run.
    pub(crate) fn leave_on_failure(&mut self) {
        if let Some(Outcome::Failed(why)) = &self.outcome {
            self.left_on = Some(why.clone());
        }
    }

    /// Why a failed save ended the run: the window held for it, or left on it.
    pub(crate) fn run_failed_on(&self) -> Option<&str> {
        self.failed_closing().or(self.left_on.as_deref())
    }

    /// The title, if it no longer says what it should: `unsaved`, `landing` dabs left to land,
    /// the window `closing`.
    pub(crate) fn title_change(
        &mut self,
        unsaved: bool,
        landing: usize,
        closing: bool,
    ) -> Option<String> {
        let title = Title {
            unsaved,
            doing: self.doing(landing, closing),
        };
        if self.title_says == Some(title) {
            return None;
        }
        self.title_says = Some(title);
        let mark = if unsaved { "*" } else { "" };
        let said = self.say(title.doing);
        let said = if said.is_empty() {
            String::new()
        } else {
            format!(": {said}")
        };
        Some(format!("{mark}{} - BrushCraft{said}", self.name))
    }

    /// The overlay's line about the project.
    pub(crate) fn line(&self, unsaved: bool, landing: usize, closing: bool) -> String {
        let edits = if unsaved {
            "unsaved edits"
        } else {
            "no unsaved edits"
        };
        let doing = self.doing(landing, closing);
        let said = match (doing, &self.outcome) {
            (Doing::Nothing, Some(Outcome::Done(said))) => said.clone(),
            _ => self.say(doing),
        };
        format!(
            "project    {}  {edits}  {said}  keys: Ctrl+S save",
            self.name
        )
    }

    /// What is going on, `landing` dabs left to land, the window `closing`.
    fn doing(&self, landing: usize, closing: bool) -> Doing {
        match (self.stage, &self.outcome) {
            (Some(Stage::PenDown), _) => Doing::PenDown,
            (Some(Stage::Landing), _) => Doing::Landing(rounded(landing, DABS_STEP)),
            (Some(Stage::Writing), _) => {
                Doing::Writing(self.tiles_left.map(|left| rounded(left, TILES_STEP)))
            }
            (None, _) if self.failed_closing.is_some() => Doing::FailedClosing,
            (None, _) if closing => Doing::Closing(rounded(landing, DABS_STEP)),
            (None, Some(Outcome::Failed(_))) => Doing::Failed,
            (None, _) => Doing::Nothing,
        }
    }

    /// `doing` in words: empty for nothing.
    fn say(&self, doing: Doing) -> String {
        match doing {
            Doing::Nothing => String::new(),
            Doing::Failed => match &self.outcome {
                Some(Outcome::Failed(why)) => format!("save failed: {why}"),
                Some(Outcome::Done(_)) | None => "save failed".to_owned(),
            },
            Doing::FailedClosing => format!(
                "save failed: {}; Ctrl+S to try again, or close again to leave it unsaved",
                self.failed_closing.as_deref().unwrap_or_default()
            ),
            Doing::PenDown => "saving when the stroke ends".to_owned(),
            Doing::Landing(left) => format!("saving, {left} dabs to land"),
            Doing::Writing(Some(left)) => format!("saving, {left} tiles to write"),
            Doing::Writing(None) => "saving".to_owned(),
            Doing::Closing(left) => format!("closing, {left} dabs to land"),
        }
    }
}

/// `count` rounded up to a multiple of `step`.
const fn rounded(count: usize, step: usize) -> usize {
    count.div_ceil(step) * step
}

/// What to say of a save that went through, in `took` seconds, with what it has to report.
fn said_of(outcome: &SaveOutcome, took: f64) -> String {
    let mut said = vec![format!("saved in {took:.1} s")];
    if let Protection::Paused(why) = &outcome.protection {
        said.push(format!("crash protection paused: {why}"));
    }
    if !outcome.unprotected.is_empty() {
        let lost = outcome.unprotected.len();
        said.push(format!("{lost} tiles without their latest edits"));
    }
    let kept = outcome.kept.iter().map(ToString::to_string);
    said.extend(kept);
    said.join(", ")
}

/// Says `line` in the log, which is stderr.
pub(crate) fn log(line: &str) {
    // Nothing more to do if stderr is gone.
    drop(writeln!(io::stderr(), "{line}"));
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use anyhow::anyhow;
    use store::{CacheError, RecoveryError};

    use super::*;

    /// `kind` as a store's failed write gives it: under a save's context, through the file and the
    /// project.
    fn failed_write(kind: io::ErrorKind) -> Error {
        let file = FileError::Io(io::Error::from(kind));
        Error::new(ProjectError::Store(file)).context("saving the project")
    }

    /// What `error` says as the failure of a save of work whole, under working protection.
    fn failure(error: &Error) -> SaveFailure {
        SaveFailure::of(error, Work::Whole, &Protection::Active)
    }

    /// Protection paused for a recovery file that couldn't be made.
    fn paused() -> Protection {
        let cause = ProjectError::Io(io::Error::from(io::ErrorKind::PermissionDenied));
        Protection::Paused(std::sync::Arc::new(cause))
    }

    #[test]
    fn a_full_disk_says_so() {
        let full = failed_write(io::ErrorKind::StorageFull);
        assert_eq!(SaveTrouble::of(&full), SaveTrouble::DiskFull);
        assert_eq!(failure(&full).title(), "Couldn't save: the disk is full");
        let quota = failed_write(io::ErrorKind::QuotaExceeded);
        assert_eq!(
            SaveTrouble::of(&quota),
            SaveTrouble::DiskFull,
            "the user's share"
        );
    }

    /// A failed write of tiles comes to a save through the store's tile cache.
    #[test]
    fn a_full_disk_says_so_from_as_deep_as_a_save_goes() {
        let write = FileError::Io(io::Error::from(io::ErrorKind::StorageFull));
        let deep = ProjectError::Cache(CacheError::Save(write));
        let error = Error::new(deep).context("saving the project");
        assert_eq!(SaveTrouble::of(&error), SaveTrouble::DiskFull);
    }

    #[test]
    fn no_permission_says_so() {
        let denied = failed_write(io::ErrorKind::PermissionDenied);
        assert_eq!(SaveTrouble::of(&denied), SaveTrouble::NoPermission);
        let read_only = failed_write(io::ErrorKind::ReadOnlyFilesystem);
        assert_eq!(SaveTrouble::of(&read_only), SaveTrouble::NoPermission);
        let opened_so = Error::new(ProjectError::Store(FileError::ReadOnly));
        assert_eq!(SaveTrouble::of(&opened_so), SaveTrouble::NoPermission);
        assert_eq!(
            failure(&denied).title(),
            "Couldn't save: this file can't be written"
        );
    }

    #[test]
    fn a_locked_project_says_so() {
        let locked = Error::new(ProjectError::Store(FileError::Locked)).context("saving");
        assert_eq!(SaveTrouble::of(&locked), SaveTrouble::InUse);
        assert_eq!(
            failure(&locked).title(),
            "Couldn't save: another app has this file open"
        );
    }

    #[test]
    fn dropped_dabs_say_so_in_user_words() {
        let dropped = Error::new(DabsDropped("no room for tile 7".to_owned()));
        let error = dropped.context("dabs were dropped");
        assert_eq!(SaveTrouble::of(&error), SaveTrouble::StrokeLost);
        let failure = failure(&error);
        assert_eq!(failure.title(), "Couldn't save: a stroke didn't finish");
        assert_eq!(failure.detail(), "Your other work is safe and still open.");
    }

    #[test]
    fn anything_else_says_the_project_couldnt_be_saved() {
        let other = failed_write(io::ErrorKind::TimedOut);
        assert_eq!(SaveTrouble::of(&other), SaveTrouble::Other);
        let plain = anyhow!("the disk is full, says a message nobody should read");
        assert_eq!(
            SaveTrouble::of(&plain),
            SaveTrouble::Other,
            "kinds, not text"
        );
        let no_path = Error::new(ProjectError::NoPath);
        assert_eq!(
            failure(&no_path).title(),
            "Couldn't save: something went wrong"
        );
    }

    /// The finisher's notification says what is true, kind by kind, as the bar does.
    #[test]
    fn a_finishers_notice_says_only_what_is_true() {
        let full = failed_write(io::ErrorKind::StorageFull);
        let couldnt = "BrushCraft couldn't save first-sculpt".to_owned();
        let notice = |failure: SaveFailure| failure.notice("first-sculpt");
        assert_eq!(
            notice(failure(&full)),
            (
                couldnt.clone(),
                "The disk is full. Your work is safe and comes back when you open it.".to_owned()
            ),
            "whole, protected"
        );
        let less = SaveFailure::of(&full, Work::LessAStroke, &Protection::Active);
        assert_eq!(
            notice(less),
            (
                couldnt.clone(),
                "The disk is full. Your other work is safe and comes back when you open it."
                    .to_owned()
            ),
            "a stroke's dabs dropped"
        );
        let exposed = "The disk is full. Open it to see what was kept: crash protection was off, \
                       so recent work may be lost."
            .to_owned();
        assert_eq!(
            notice(SaveFailure::of(&full, Work::Whole, &paused())),
            (couldnt.clone(), exposed.clone()),
            "protection paused"
        );
        assert_eq!(
            notice(failure(&full).exposed()),
            (couldnt, exposed),
            "a flush failed meanwhile"
        );
        let gone = Error::new(TilesLost(2));
        assert_eq!(
            notice(failure(&gone)),
            (
                "BrushCraft saved first-sculpt, less part of a stroke".to_owned(),
                "Part of a stroke is gone. The rest of your work was saved.".to_owned()
            ),
            "saved less part of a stroke"
        );
        assert_eq!(
            notice(SaveFailure::stroke_lost_but_saved()).0,
            "BrushCraft saved first-sculpt, less part of a stroke"
        );
    }

    #[test]
    fn work_whole_and_protected_is_safe_and_still_open() {
        let full = failed_write(io::ErrorKind::StorageFull);
        assert_eq!(failure(&full).detail(), "Your work is safe and still open.");
    }

    /// The dabs may be gone for a reason of their own, under another failure's title.
    #[test]
    fn a_stroke_lost_beside_another_failure_is_still_said() {
        let full = failed_write(io::ErrorKind::StorageFull);
        let failure = SaveFailure::of(&full, Work::LessAStroke, &Protection::Active);
        assert_eq!(failure.title(), "Couldn't save: the disk is full");
        assert_eq!(failure.detail(), "Your other work is safe and still open.");
    }

    /// What a failed [`store::Project::save`] looks like with protection paused: the error doesn't
    /// say, and only the project does.
    #[test]
    fn paused_protection_never_says_safe() {
        let full = failed_write(io::ErrorKind::StorageFull);
        let failure = SaveFailure::of(&full, Work::Whole, &paused());
        assert_eq!(failure.title(), "Couldn't save: the disk is full");
        assert_eq!(
            failure.detail(),
            "Your work is still open, but a crash could lose it."
        );
        let lost = SaveFailure::of(&full, Work::LessAStroke, &paused());
        assert_eq!(
            lost.detail(),
            "Your other work is still open, but a crash could lose it."
        );
    }

    /// Links that say the recovery file doesn't hold the edits, whatever the project says now.
    #[test]
    fn a_failed_recovery_writer_never_says_safe() {
        let paused_cause = ProjectError::Store(FileError::Locked);
        let paused = ProjectError::ProtectionPaused(std::sync::Arc::new(paused_cause));
        let writer = ProjectError::Recovery(RecoveryError::Stopped);
        let tiles = ProjectError::Unprotected(Vec::new());
        for error in [paused, writer, tiles] {
            let said = format!("{error}");
            let error = Error::new(error).context("saving the project");
            assert!(
                failure(&error).detail().contains("a crash could lose it"),
                "{said}"
            );
        }
    }

    /// The cause under a pause goes through an `Arc`, which hides the type of the link it is.
    #[test]
    fn a_pauses_cause_still_says_why() {
        let locked = ProjectError::Store(FileError::Locked);
        let paused = ProjectError::ProtectionPaused(std::sync::Arc::new(locked));
        let error = Error::new(paused).context("saving the project");
        assert_eq!(SaveTrouble::of(&error), SaveTrouble::InUse);
        let full = ProjectError::Io(io::Error::from(io::ErrorKind::StorageFull));
        let full = ProjectError::ProtectionPaused(std::sync::Arc::new(full));
        assert_eq!(SaveTrouble::of(&Error::new(full)), SaveTrouble::DiskFull);
    }

    /// Edits gone are said plainly, and no try can bring them back.
    #[test]
    fn tiles_lost_by_a_save_say_so_and_offer_no_try() {
        let lost = Error::new(TilesLost(3));
        let failure = failure(&lost);
        assert_eq!(failure.title(), "Couldn't save: part of a stroke is gone");
        assert_eq!(failure.detail(), "The rest of your work was saved.");
        assert!(!failure.can_be_tried_again());
        let full = failed_write(io::ErrorKind::StorageFull);
        assert!(self::failure(&full).can_be_tried_again());
    }

    /// A pause's cause reaches here with no [`Error`] round it.
    #[test]
    fn a_cause_with_no_context_says_why_too() {
        let full = ProjectError::Io(io::Error::from(io::ErrorKind::StorageFull));
        assert_eq!(SaveTrouble::of_source(&full), SaveTrouble::DiskFull);
        let locked = ProjectError::Store(FileError::Locked);
        assert_eq!(SaveTrouble::of_source(&locked), SaveTrouble::InUse);
        assert_eq!(
            SaveTrouble::of_source(&ProjectError::NoPath),
            SaveTrouble::Other
        );
    }

    #[test]
    fn a_failure_the_frame_loop_knows_is_exposed_never_says_safe() {
        let full = failed_write(io::ErrorKind::StorageFull);
        let exposed = failure(&full).exposed();
        assert_eq!(
            exposed.detail(),
            "Your work is still open, but a crash could lose it."
        );
        assert!(!exposed.covered());
    }

    /// Between saves nothing failed to save, and nothing can be tried.
    #[test]
    fn edits_gone_between_saves_say_so_in_their_own_words() {
        let failure = SaveFailure::edits_gone(Protected::Yes);
        assert_eq!(failure.title(), "Part of a stroke is gone");
        assert_eq!(failure.detail(), "Your other work is safe and still open.");
        assert!(!failure.can_be_tried_again());
        let exposed = SaveFailure::edits_gone(Protected::No);
        assert_eq!(
            exposed.detail(),
            "Your other work is still open, but a crash could lose it."
        );
    }

    /// The save worked: "Couldn't save" would be false, and there is nothing to try.
    #[test]
    fn a_stroke_lost_before_a_save_that_worked_says_so_in_its_own_words() {
        let failure = SaveFailure::stroke_lost_but_saved();
        assert_eq!(failure.title(), "A stroke didn't finish");
        assert_eq!(failure.detail(), "The rest of your work was saved.");
        assert!(!failure.can_be_tried_again());
    }

    /// Found work isn't open, whatever the open project's state: "still open" would be false.
    #[test]
    fn a_failed_copy_says_the_work_is_where_it_was_found() {
        let full = failed_write(io::ErrorKind::StorageFull);
        let failure = SaveFailure::of_copy(&full);
        assert_eq!(failure.title(), "Couldn't save: the disk is full");
        assert_eq!(
            failure.detail(),
            "Your unsaved work is safe where it was found."
        );
        assert!(failure.of_found_work());
        assert!(!self::failure(&full).of_found_work());
    }

    /// A project called `sphere`.
    fn saving() -> Saving {
        Saving::new(Some(Path::new("/some/where/sphere.bcrf")), Instant::now())
    }

    #[test]
    fn a_save_let_go_of_before_it_began_says_nothing() {
        let mut saving = saving();
        saving.start(Instant::now(), 0);
        saving.let_go();
        assert_eq!(saving.stage(), None, "none under way");
        assert_eq!(saving.last_succeeded(), None, "and no outcome");
        assert_eq!(
            saving.title_change(true, 0, false).as_deref(),
            Some("*sphere - BrushCraft")
        );
    }

    #[test]
    fn the_last_outcome_is_known_until_the_next_save_starts() {
        let mut saving = saving();
        assert_eq!(saving.last_succeeded(), None, "no save yet");
        saving.nothing_to_save();
        assert_eq!(saving.last_succeeded(), Some(true), "nothing lost");
        saving.start(Instant::now(), 0);
        assert_eq!(saving.last_succeeded(), None, "a save under way");
        saving.failed(&anyhow!("no room"), false);
        assert_eq!(saving.last_succeeded(), Some(false));
        saving.start(Instant::now(), 0);
        assert_eq!(saving.last_succeeded(), None, "forgotten");
    }

    #[test]
    fn the_title_names_the_project_and_marks_unsaved_edits() {
        let mut saving = saving();
        assert_eq!(
            saving.title_change(false, 0, false).as_deref(),
            Some("sphere - BrushCraft")
        );
        assert_eq!(
            saving.title_change(true, 0, false).as_deref(),
            Some("*sphere - BrushCraft")
        );
    }

    #[test]
    fn a_project_with_no_path_is_untitled() {
        let mut saving = Saving::new(None, Instant::now());
        assert_eq!(
            saving.title_change(false, 0, false).as_deref(),
            Some("untitled - BrushCraft")
        );
    }

    #[test]
    fn the_title_is_said_once_and_again_only_when_it_changes() {
        let mut saving = saving();
        saving.start(Instant::now(), 0);
        saving.reach(Stage::Landing);
        assert!(saving.title_change(true, 95, false).is_some());
        assert_eq!(saving.title_change(true, 91, false), None, "same ten");
        assert_eq!(
            saving.title_change(true, 90, false).as_deref(),
            Some("*sphere - BrushCraft: saving, 90 dabs to land")
        );
    }

    #[test]
    fn the_title_follows_a_save_through_its_stages() {
        let mut saving = saving();
        saving.start(Instant::now(), 0);
        assert_eq!(
            saving.title_change(true, 1, false).as_deref(),
            Some("*sphere - BrushCraft: saving when the stroke ends")
        );
        saving.reach(Stage::Writing);
        assert_eq!(
            saving.title_change(true, 0, false).as_deref(),
            Some("*sphere - BrushCraft: saving")
        );
        saving.copying(1_250);
        assert_eq!(
            saving.title_change(true, 0, false).as_deref(),
            Some("*sphere - BrushCraft: saving, 1300 tiles to write")
        );
    }

    #[test]
    fn a_failure_is_said_until_the_next_save_is_asked_for() {
        let mut saving = saving();
        saving.start(Instant::now(), 0);
        saving.failed(&anyhow!("the disk is full"), false);
        assert_eq!(saving.stage(), None);
        assert_eq!(
            saving.title_change(true, 0, false).as_deref(),
            Some("*sphere - BrushCraft: save failed: the disk is full")
        );
        assert!(saving.line(true, 0, false).contains("the disk is full"));
        saving.start(Instant::now(), 0);
        assert!(!saving.line(true, 0, false).contains("the disk is full"));
    }

    #[test]
    fn a_failure_with_the_window_closing_waits_and_says_what_to_do() {
        let mut saving = saving();
        saving.start(Instant::now(), 0);
        saving.failed(&anyhow!("the disk is full"), true);
        assert_eq!(saving.failed_closing(), Some("the disk is full"));
        let title = saving.title_change(true, 0, true).expect("a title");
        assert!(title.contains("the disk is full"), "{title}");
        assert!(title.contains("close again"), "{title}");
        saving.start(Instant::now(), 0);
        assert_eq!(saving.failed_closing(), None, "tried again");
    }

    #[test]
    fn a_close_counts_down_in_the_title_and_a_save_comes_first() {
        let mut saving = saving();
        assert_eq!(
            saving.title_change(true, 41, true).as_deref(),
            Some("*sphere - BrushCraft: closing, 50 dabs to land")
        );
        saving.start(Instant::now(), 0);
        saving.reach(Stage::Landing);
        assert_eq!(
            saving.title_change(true, 41, true).as_deref(),
            Some("*sphere - BrushCraft: saving, 50 dabs to land")
        );
    }

    #[test]
    fn the_overlay_says_what_the_last_save_came_to() {
        let mut saving = saving();
        assert!(!saving.line(true, 0, false).contains("no unsaved edits"));
        assert!(saving.line(false, 0, false).contains("no unsaved edits"));
        saving.nothing_to_save();
        assert!(saving.line(false, 0, false).contains("nothing to save"));
    }

    #[test]
    fn what_a_save_has_to_report_is_said() {
        let outcome = SaveOutcome {
            protection: Protection::Active,
            unprotected: Vec::new(),
            kept: Vec::new(),
        };
        assert_eq!(said_of(&outcome, 1.25), "saved in 1.2 s");
        let paused = SaveOutcome {
            protection: Protection::Paused(std::sync::Arc::new(ProjectError::NoPath)),
            ..outcome
        };
        assert!(said_of(&paused, 1.0).contains("crash protection paused"));
    }
}
