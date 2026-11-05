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
//! A failed save loses nothing: the edits stay, unsaved, in the recovery file. The title and the
//! overlay say so until the next Ctrl+S, the log (stderr) says why, and Ctrl+S tries again. Dabs
//! dropped by a store error since Ctrl+S, in any stage, fail the save too, as it can't hold
//! everything drawn.
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
//! fails the window stays, the bar goes, and the title says so.

use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Error;
use store::{Protection, SaveOutcome};

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
        self.tiles_left = None;
        self.outcome = None;
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
        self.outcome = Some(Outcome::Done("nothing to save".to_owned()));
    }

    /// The save under way went through: says so, and how long it took, in the log.
    pub(crate) fn saved(&mut self, outcome: &SaveOutcome, now: Instant) {
        let took = now.saturating_duration_since(self.asked).as_secs_f64();
        let said = said_of(outcome, took);
        log(&format!("brushcraft: {}: {said}", self.name));
        self.stage = None;
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
    drop(writeln!(std::io::stderr(), "{line}"));
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use anyhow::anyhow;

    use super::*;

    /// A project called `sphere`.
    fn saving() -> Saving {
        Saving::new(Some(Path::new("/some/where/sphere.bcrf")), Instant::now())
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
            protection: Protection::Paused(std::sync::Arc::new(store::ProjectError::NoPath)),
            ..outcome
        };
        assert!(said_of(&paused, 1.0).contains("crash protection paused"));
    }
}
