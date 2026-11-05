//! What the UI shows and when: the app's whole UI state, on top of the `ui` crate.
//!
//! `ui` knows widgets and nothing of projects; this says which ones the project's state calls for,
//! in the user's words. For now that is the save chip. The owner thread brings it up to date once
//! a frame, after the pen's work ([`Dialogs::update`]), and what it shows goes to the render thread
//! with the frame. The pen touches only the fades ([`Dialogs::pen_down`], [`Dialogs::pen_up`]), and
//! a chip where there was none waits for it to lift, so nothing appears, and no atlas uploads,
//! during a stroke.

use std::time::Instant;

use render::surface::Size;
use ui::{Blend, Chip, Mark, Theme, Ui, UiFrame};

/// What a control on a bar does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    dead_code,
    reason = "the UI's actions: nothing makes them until the bars that offer them"
)]
pub(crate) enum Action {
    /// Save, then go on with what was asked.
    Save,
    /// Close without saving.
    DontSave,
    /// Stay, leaving the question.
    KeepSculpting,
    /// Stop what's under way.
    Cancel,
    /// Try a failed save again.
    TryAgain,
    /// Put the bar away.
    Dismiss,
    /// Open the project from its last save.
    UseLastSave,
    /// Open found work as a copy.
    OpenCopy,
    /// Leave found work where it is.
    Ignore,
}

/// What the owner knows that decides the chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Facts {
    /// Whether anything drawn isn't in a save.
    pub(crate) unsaved: bool,
    /// Whether a save is under way.
    pub(crate) saving: bool,
}

/// What the chip has to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Nothing: no chip.
    Quiet,
    /// There are unsaved edits.
    Unsaved,
    /// A save is under way.
    Saving,
    /// A save just finished.
    Saved,
}

impl Status {
    /// The chip that says it.
    fn chip(self) -> Option<Chip> {
        let chip = |text: &str, mark| {
            Some(Chip {
                text: text.to_owned(),
                mark,
            })
        };
        match self {
            Self::Quiet => None,
            Self::Unsaved => chip("Unsaved", Mark::Dot),
            Self::Saving => chip("Saving", Mark::Ring),
            Self::Saved => chip("Saved", Mark::None),
        }
    }
}

/// What shows when: the app's whole UI state, on top of [`Ui`].
#[derive(Debug)]
pub(crate) struct Dialogs {
    /// The widgets and what they draw.
    ui: Ui<Action>,
    /// The window's size in physical pixels.
    size: (u32, u32),
    /// The window's scale factor.
    scale: f32,
    /// What the chip says now.
    status: Status,
    /// When the last save finished, while "Saved" still shows.
    saved_at: Option<Instant>,
    /// Whether the pen is down.
    pen_is_down: bool,
    /// Whether a chip is waiting for the pen to lift.
    waiting: bool,
}

/// `size` as the UI takes it; a window with none draws nothing, so any will do.
fn pixels(size: Option<Size>) -> (u32, u32) {
    size.map_or((1, 1), |size| (size.width(), size.height()))
}

/// A window's scale factor as the UI takes it.
fn ui_scale(scale: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a scale factor is a small number; a UI's sizes need no more than f32"
    )]
    let scale = scale as f32;
    scale
}

impl Dialogs {
    /// Nothing showing yet, in a window of `size` at `scale`, for a renderer blending as `blend`.
    pub(crate) fn new(size: Option<Size>, scale: f64, blend: Blend, now: Instant) -> Self {
        let (size, scale) = (pixels(size), ui_scale(scale));
        Self {
            ui: Ui::new(Theme::GLASS, blend, size, scale, now),
            size,
            scale,
            status: Status::Quiet,
            saved_at: None,
            pen_is_down: false,
            waiting: false,
        }
    }

    /// The window is now `size`, if it has one.
    pub(crate) fn resize(&mut self, size: Option<Size>) {
        if size.is_some() {
            self.size = pixels(size);
            self.ui.set_view(self.size, self.scale);
        }
    }

    /// The window's scale factor is `scale`: whether that's new.
    pub(crate) fn set_scale(&mut self, scale: f64) -> bool {
        let scale = ui_scale(scale);
        let changed = (scale - self.scale).abs() > f32::EPSILON;
        self.scale = scale;
        self.ui.set_view(self.size, scale);
        changed
    }

    /// Brings the chip up to date; cheap when nothing changed, as nearly always. A chip where there
    /// was none waits for the pen to lift; one already showing changes at once.
    pub(crate) fn update(&mut self, facts: Facts, now: Instant) {
        let status = self.next_status(facts, now);
        let arriving = self.status == Status::Quiet && status != Status::Quiet;
        self.waiting = arriving && self.pen_is_down;
        if status != self.status && !self.waiting {
            self.status = status;
            self.ui.set_chip(status.chip(), now);
        }
    }

    /// A save finished at `now`: "Saved" shows for its while, whether or not a frame saw the save.
    pub(crate) fn saved(&mut self, now: Instant) {
        self.saved_at = Some(now);
    }

    /// What the chip should say now, forgetting a "Saved" that is over.
    fn next_status(&mut self, facts: Facts, now: Instant) -> Status {
        if facts.saving {
            self.saved_at = None;
            return Status::Saving;
        }
        if facts.unsaved {
            self.saved_at = None;
            return Status::Unsaved;
        }
        let over = |at: Instant| now.saturating_duration_since(at) >= Theme::GLASS.saved_hold;
        self.saved_at = self.saved_at.filter(|&at| !over(at));
        self.saved_at.map_or(Status::Quiet, |_| Status::Saved)
    }

    /// What to draw at `now`, if anything.
    pub(crate) fn frame(&mut self, now: Instant) -> Option<UiFrame> {
        self.ui.frame(now)
    }

    /// Whether a frame is wanted at `now`: something is fading, or changed, or a chip that waited
    /// for the pen can come.
    pub(crate) fn needs_frame(&self, now: Instant) -> bool {
        self.ui.needs_frame(now) || (self.waiting && !self.pen_is_down)
    }

    /// When something next changes by itself: the chip's "Saved" leaving.
    pub(crate) fn next_change(&self) -> Option<Instant> {
        self.saved_at.map(|at| at + Theme::GLASS.saved_hold)
    }

    /// The pen touched down: bars fade out and the chip dims. Touches the fades only.
    pub(crate) fn pen_down(&mut self, now: Instant) {
        self.pen_is_down = true;
        self.ui.pen_down(now);
    }

    /// The pen lifted: everything comes back.
    pub(crate) fn pen_up(&mut self, now: Instant) {
        self.pen_is_down = false;
        self.ui.pen_up(now);
    }

    /// Whether the pen is down.
    pub(crate) const fn pen_is_down(&self) -> bool {
        self.pen_is_down
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    /// No edits, no save.
    const CLEAN: Facts = Facts {
        unsaved: false,
        saving: false,
    };

    /// Edits, no save.
    const EDITED: Facts = Facts {
        unsaved: true,
        saving: false,
    };

    /// A save of edits under way.
    const SAVING: Facts = Facts {
        unsaved: true,
        saving: true,
    };

    /// Dialogs in a window of 800 × 600, showing nothing.
    fn dialogs(now: Instant) -> Dialogs {
        Dialogs::new(Size::new(800, 600), 1.0, Blend::Display, now)
    }

    /// How long "Saved" stays.
    const fn hold() -> Duration {
        Theme::GLASS.saved_hold
    }

    #[test]
    fn no_edits_show_no_chip() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(CLEAN, now);
        assert_eq!(dialogs.status, Status::Quiet, "nothing to say");
        assert!(dialogs.frame(now).is_none(), "no chip");
        assert!(!dialogs.needs_frame(now), "no frame");
    }

    #[test]
    fn unsaved_edits_show_unsaved_with_a_dot() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        let chip = dialogs.status.chip().expect("a chip");
        assert_eq!(chip.text, "Unsaved", "in the user's words");
        assert_eq!(chip.mark, Mark::Dot, "an orange dot");
        assert!(dialogs.frame(now).is_some(), "it draws");
        assert!(dialogs.needs_frame(now), "fading in");
    }

    #[test]
    fn a_save_shows_saving_with_a_ring_then_saved_then_nothing_after_two_seconds() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        dialogs.update(SAVING, now);
        let saving = dialogs.status.chip().expect("a chip");
        assert_eq!((saving.text.as_str(), saving.mark), ("Saving", Mark::Ring));
        assert_eq!(dialogs.next_change(), None, "it stays until the save ends");

        let saved_at = now + Duration::from_secs(1);
        dialogs.saved(saved_at);
        dialogs.update(CLEAN, saved_at);
        let saved = dialogs.status.chip().expect("a chip");
        assert_eq!((saved.text.as_str(), saved.mark), ("Saved", Mark::None));
        assert_eq!(dialogs.next_change(), Some(saved_at + hold()), "it will go");

        let almost = saved_at + hold().mul_f32(0.99);
        dialogs.update(CLEAN, almost);
        assert_eq!(dialogs.status, Status::Saved, "still there");

        dialogs.update(CLEAN, saved_at + hold());
        assert_eq!(dialogs.status, Status::Quiet, "gone");
        assert_eq!(dialogs.next_change(), None, "nothing more to come");
        assert!(dialogs.needs_frame(saved_at + hold()), "fading out");
        let gone = saved_at + hold() + Theme::GLASS.fade;
        assert!(dialogs.frame(gone).is_none(), "and then no chip");
    }

    #[test]
    fn a_failed_save_goes_back_to_unsaved() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(SAVING, now);
        dialogs.update(EDITED, now);
        assert_eq!(dialogs.status, Status::Unsaved, "the work is still unsaved");
        assert_eq!(dialogs.next_change(), None, "and that doesn't go by itself");
    }

    #[test]
    fn a_stroke_after_a_save_ends_saved_at_once() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(SAVING, now);
        dialogs.saved(now);
        dialogs.update(CLEAN, now);
        dialogs.update(EDITED, now);
        assert_eq!(dialogs.status, Status::Unsaved, "new edits");
        assert_eq!(dialogs.next_change(), None, "\"Saved\" won't come back");
    }

    #[test]
    fn a_save_no_frame_saw_still_says_saved() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        dialogs.saved(now);
        dialogs.update(CLEAN, now);
        assert_eq!(dialogs.status, Status::Saved, "straight from unsaved");
    }

    #[test]
    fn a_save_with_nothing_to_save_says_nothing() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(SAVING, now);
        dialogs.update(CLEAN, now);
        assert_eq!(dialogs.status, Status::Quiet, "no save was made");
    }

    #[test]
    fn a_first_chip_waits_for_the_pen_and_wants_a_frame_when_it_lifts() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.pen_down(now);
        dialogs.update(EDITED, now);
        assert_eq!(dialogs.status, Status::Quiet, "held back");
        assert!(dialogs.frame(now).is_none(), "nothing to draw");
        assert!(!dialogs.needs_frame(now), "and no frame for it");

        dialogs.pen_up(now);
        assert!(dialogs.needs_frame(now), "its frame");
        dialogs.update(EDITED, now);
        assert_eq!(dialogs.status, Status::Unsaved, "here it comes");
        assert!(dialogs.frame(now).is_some(), "it draws");
    }

    #[test]
    fn a_chip_already_showing_changes_under_the_pen() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        dialogs.pen_down(now);
        dialogs.update(SAVING, now);
        assert_eq!(dialogs.status, Status::Saving, "at once");
    }

    #[test]
    fn a_chip_can_go_under_the_pen() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        dialogs.pen_down(now);
        dialogs.update(CLEAN, now);
        assert_eq!(dialogs.status, Status::Quiet, "gone");
    }

    #[test]
    fn the_same_facts_again_build_nothing() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        let first = dialogs.frame(now).expect("a chip");
        let later = now + Duration::from_secs(1);
        dialogs.update(EDITED, later);
        let second = dialogs.frame(later).expect("a chip");
        assert!(Arc::ptr_eq(&first.list, &second.list), "the same list");
        assert!(!dialogs.needs_frame(later), "and no frame for it");
    }

    #[test]
    fn the_pen_dims_the_chip_and_builds_nothing() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        let before = dialogs.frame(now).expect("a chip");
        dialogs.pen_down(now);
        let during = dialogs.frame(now).expect("a chip");
        assert!(Arc::ptr_eq(&before.list, &during.list), "no rebuild");
        assert!((during.fades[1].to - 0.25).abs() < 1e-6, "to a quarter");
        dialogs.pen_up(now);
        let after = dialogs.frame(now).expect("a chip");
        assert!((after.fades[1].to - 1.0).abs() < 1e-6, "and back");
    }

    #[test]
    fn a_new_scale_is_news_and_the_same_one_isnt() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        assert!(!dialogs.set_scale(1.0), "the same");
        assert!(dialogs.set_scale(1.25), "a fractional scale");
        assert!(!dialogs.set_scale(1.25), "the same again");
    }

    #[test]
    fn a_window_with_no_size_keeps_the_last() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.resize(None);
        assert_eq!(dialogs.size, (800, 600), "minimised");
        dialogs.resize(Size::new(640, 480));
        assert_eq!(dialogs.size, (640, 480), "resized");
    }
}
