//! What the UI shows and when: the app's whole UI state, on top of the `ui` crate.
//!
//! `ui` knows widgets and nothing of projects; this says which ones the project's state calls for,
//! in the user's words: the save chip, and the bar's [`Prompt`]. The owner thread brings it up to
//! date once a frame, after the pen's work ([`Dialogs::update`]), and what it shows goes to the
//! render thread with the frame. The pen touches only the fades ([`Dialogs::pen_down`],
//! [`Dialogs::pen_up`]), and a chip where there was none, or a bar that changed, waits for it to
//! lift, so nothing appears, and no atlas uploads, during a stroke.
//!
//! **Routing:** with a bar up, a `Down` or a press goes here first ([`Dialogs::claim`],
//! [`Dialogs::claim_mouse`]). One on the bar is the UI's, whole: the stroke's other samples stop
//! here too, and its control acts only if the stroke ends with an `Up` inside it
//! ([`Dialogs::release`], [`Dialogs::release_mouse`]). With no bar the owner's one look at
//! [`Dialogs::routes`] is all the pen pays.
//!
//! **Notes** say what opening found ([`Dialogs::note`]). They answer to the same rule as the close
//! bar: a stroke routed to the brush puts one away, and anything else, navigation included, leaves
//! it. The one that only tells also goes by itself after [`Theme::note_hold`].
//!
//! **The failure bar** says a save failed ([`Dialogs::failed`]) and answers to that rule too, but
//! only once it can have been seen: one that arrived under a stroke waits for its `Up`, and that
//! stroke isn't an answer to it. It takes a note's place, and the note asks again after it. So
//! does the bar saying crash protection has stopped ([`Dialogs::protection_stopped`]), which waits
//! for a bar already asking something, and goes by itself if protection comes back.

use std::time::{Instant, SystemTime};

use jiff::Zoned;
use render::surface::Size;
use store::RecoveryCandidate;
use ui::{Bar, Blend, Chip, Control, Emphasis, Mark, Theme, Ui, UiFrame};

use crate::OpenNote;
use crate::camera::Pixel;
use crate::navigate::Button;
use crate::saving::{SaveFailure, SaveTrouble};

/// What a control on a bar does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    /// Save, then go on with what was asked.
    Save,
    /// Close without saving.
    DontSave,
    /// Stay, leaving the question.
    KeepSculpting,
    /// Stop what's under way.
    Cancel,
    /// Try again what failed.
    TryAgain,
    /// Save and stay, with nothing failed yet: crash protection has stopped.
    SaveWork,
    /// Put the bar away.
    Dismiss,
    /// Open the project from its last save.
    UseLastSave,
    /// Open found work as a copy.
    OpenCopy,
    /// Leave found work where it is.
    Ignore,
}

/// Who a `Down` belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Claim {
    /// It landed on a bar: the UI has the whole stroke.
    Ui,
    /// It goes where it would with no bar.
    Elsewhere,
}

/// What a stroke the UI took landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Press {
    /// A bar's glass, or a control the press can't act on: it is swallowed and does nothing.
    Glass,
    /// A control: it acts if the stroke ends inside it.
    Control(Action),
}

impl Press {
    /// What this press does if it ends over `under`: only a control's action, and only there.
    fn ends_over(self, under: Option<Action>) -> Option<Action> {
        match self {
            Self::Control(action) => under.filter(|&ended_on| ended_on == action),
            Self::Glass => None,
        }
    }
}

/// What the owner knows that decides the chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Facts {
    /// Whether anything drawn isn't in a save.
    pub(crate) unsaved: bool,
    /// Whether a save is under way.
    pub(crate) saving: bool,
    /// How many dabs the save under way still waits for.
    pub(crate) landing: usize,
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

/// What the bottom bar is about.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Prompt {
    /// Nothing.
    None,
    /// "Save changes to `name`?"
    AskClose {
        /// The project's name.
        name: String,
    },
    /// A save the window exits after, with a progress line. It says "Finishing your stroke" while
    /// dabs land, and "Saving" once the save is writing.
    Finishing {
        /// The dabs left to land when it began.
        from: usize,
        /// The dabs left to land as the bar last said.
        left: usize,
        /// When the bar last said so: it says so again [`Theme::progress_step`] later at the
        /// soonest.
        said_at: Instant,
    },
    /// "Your unsaved work is back": opening restored it.
    Restored {
        /// When the work dates from, as [`said_when`] says it; `None` if the system clock can't
        /// say.
        said: Option<String>,
        /// When the note goes by itself.
        until: Instant,
    },
    /// "Unsaved work found from an earlier version": found, not restored, and the user's to
    /// decide, however long that takes.
    EarlierVersion {
        /// When the work dates from, as [`said_when`] says it; `None` if the system clock can't
        /// say.
        said: Option<String>,
        /// What was found.
        candidate: RecoveryCandidate,
    },
    /// "Saving": found work on its way to a copy, which holds the thread once it starts.
    Copying,
    /// "Couldn't save: …": what failed and what that leaves, until the user answers or goes back
    /// to work.
    Failed(SaveFailure),
    /// "Crash protection has stopped: …": why, until the user answers, goes back to work, or
    /// protection comes back.
    Stopped(SaveTrouble),
}

impl Prompt {
    /// The bar that says it.
    fn bar(&self) -> Option<Bar<Action>> {
        let quiet = |label: &str, action| Control {
            label: label.to_owned(),
            action,
            emphasis: Emphasis::Quiet,
        };
        match self {
            Self::None => None,
            Self::AskClose { name } => {
                let save = Control {
                    label: "Save".to_owned(),
                    action: Action::Save,
                    emphasis: Emphasis::Filled,
                };
                let controls = vec![
                    quiet("Don't save", Action::DontSave),
                    quiet("Keep sculpting", Action::KeepSculpting),
                    save,
                ];
                Some(Bar::new(format!("Save changes to {name}?"), controls))
            }
            Self::Finishing { from, left, .. } => {
                let title = if *left > 0 {
                    "Finishing your stroke"
                } else {
                    "Saving"
                };
                let cancel = quiet("Cancel", Action::Cancel);
                Some(Bar::new(title, vec![cancel]).with_progress(progress(*left, *from)))
            }
            Self::Restored { said, .. } => {
                let controls = vec![quiet("Use last save instead", Action::UseLastSave)];
                Some(detailed(
                    Bar::new("Your unsaved work is back", controls),
                    said.as_deref(),
                ))
            }
            Self::EarlierVersion { said, .. } => {
                let copy = Control {
                    label: "Open as a copy".to_owned(),
                    action: Action::OpenCopy,
                    emphasis: Emphasis::Filled,
                };
                let controls = vec![quiet("Ignore", Action::Ignore), copy];
                Some(detailed(
                    Bar::new("Unsaved work found from an earlier version", controls),
                    said.as_deref(),
                ))
            }
            Self::Copying => Some(Bar::new("Saving", Vec::new())),
            Self::Failed(failure) => {
                let again = Control {
                    label: "Try again".to_owned(),
                    action: Action::TryAgain,
                    emphasis: Emphasis::Filled,
                };
                let mut controls = vec![quiet("Dismiss", Action::Dismiss)];
                controls.extend(failure.can_be_tried_again().then_some(again));
                let bar = Bar::new(failure.title(), controls).with_detail(failure.detail());
                Some(bar.needing_you())
            }
            Self::Stopped(trouble) => {
                let save = Control {
                    label: "Save".to_owned(),
                    action: Action::SaveWork,
                    emphasis: Emphasis::Filled,
                };
                let controls = vec![quiet("Dismiss", Action::Dismiss), save];
                let title = format!("Crash protection has stopped: {}", trouble.said());
                let bar = Bar::new(title, controls).with_detail("Save your work when you can.");
                Some(bar.needing_you())
            }
        }
    }
}

/// `bar`, with `detail` under its title if there is one.
fn detailed(bar: Bar<Action>, detail: Option<&str>) -> Bar<Action> {
    match detail {
        Some(detail) => bar.with_detail(detail),
        None => bar,
    }
}

/// When something written at `then` dates from, seen from `now`, in `now`'s time zone: "From today
/// at 16:42", "From yesterday at 16:42", or the date, its year only if that isn't this one.
pub(crate) fn said_when(then: &Zoned, now: &Zoned) -> String {
    let then = then.with_time_zone(now.time_zone().clone());
    let (day, today) = (then.date(), now.date());
    let clock = then.strftime("%H:%M");
    if day == today {
        format!("From today at {clock}")
    } else if today.yesterday().is_ok_and(|yesterday| yesterday == day) {
        format!("From yesterday at {clock}")
    } else if day.year() == today.year() {
        format!("From {} at {clock}", then.strftime("%-d %b"))
    } else {
        format!("From {} at {clock}", then.strftime("%-d %b %Y"))
    }
}

/// [`said_when`] of `when`, by the system's clock and time zone; `None` for a time past what they
/// can say.
fn said_now(when: SystemTime) -> Option<String> {
    let then = Zoned::try_from(when).ok()?;
    Some(said_when(&then, &Zoned::now()))
}

/// How far along a wait is with `left` of `from` still to go, 0 to 1.
fn progress(left: usize, from: usize) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a progress line is a few hundred pixels: f32 is far finer"
    )]
    let share = left as f32 / from.max(1) as f32;
    (1.0 - share).max(0.0)
}

/// Who has the pointer's input: the bar, and what it has been pressed with.
#[derive(Debug, Default)]
struct Routing {
    /// Whether a bar is meant to show.
    bar_up: bool,
    /// The pen stroke the UI took, while it lasts.
    pen: Option<Press>,
    /// The mouse press the UI took, while it lasts, and its button.
    mouse: Option<(Button, Press)>,
    /// Where the mouse last was, while input is routed here: a release doesn't say. The mouse's
    /// alone, so a pen in the other hand steers no click.
    mouse_at: Option<Pixel>,
    /// Whether input goes through here first: a bar is up, or a press it took is still down.
    on: bool,
}

impl Routing {
    /// Brings `on` up to date.
    const fn update(&mut self) {
        self.on = self.bar_up || self.pen.is_some() || self.mouse.is_some();
    }
}

/// What the pen does to what shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PenEffect {
    /// Bars fade and wait for it, as ever.
    Fades,
    /// Nothing: a run measuring the UI ([`Dialogs::show_test_bar`]).
    None,
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
    /// What the bar is about.
    prompt: Prompt,
    /// The bar isn't saying what `prompt` does, for the pen is down.
    prompt_waiting: bool,
    /// What the failure bar took the place of, a note, or a bar a close came over before anyone
    /// could see it: it says its piece when what is in front is answered.
    behind: Option<Prompt>,
    /// When a bar put the latest failure in front of the user, if one has.
    failure_shown: Option<Instant>,
    /// What the pen does to what shows.
    pen_effect: PenEffect,
    /// Input on its way to the bar.
    routing: Routing,
    /// How many hit tests were made.
    #[cfg(test)]
    hit_tests: std::cell::Cell<u32>,
}

/// `size` as the UI takes it; a window with none draws nothing, so any will do.
fn pixels(size: Option<Size>) -> (u32, u32) {
    size.map_or((1, 1), |size| (size.width(), size.height()))
}

/// `at` as the UI takes it.
fn ui_point(at: Pixel) -> (f32, f32) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a pixel of a window is far inside what f32 holds exactly enough to hit a control"
    )]
    let point = (at.x as f32, at.y as f32);
    point
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
            prompt: Prompt::None,
            prompt_waiting: false,
            behind: None,
            failure_shown: None,
            pen_effect: PenEffect::Fades,
            routing: Routing::default(),
            #[cfg(test)]
            hit_tests: std::cell::Cell::new(0),
        }
    }

    /// Keeps a close bar up from `now` on, the largest there is, for measuring what the UI costs a
    /// scripted run (`--show-bar`), and nothing else: no person is meant to meet it.
    ///
    /// So the pass and the chip draw in every frame of a stroke, the pen touches nothing here: no
    /// fades, and a chip arrives under it. The bar answers nothing and routes nothing, or a
    /// scripted stroke could land on it, and the run would measure something else.
    pub(crate) fn show_test_bar(&mut self, now: Instant) {
        self.pen_effect = PenEffect::None;
        let prompt = Prompt::AskClose {
            name: "first-sculpt".to_owned(),
        };
        self.ui.set_bar(prompt.bar(), now);
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

    /// Brings the chip and the bar's progress up to date; cheap when nothing changed, as nearly
    /// always. A chip where there was none waits for the pen to lift; one already showing changes
    /// at once.
    pub(crate) fn update(&mut self, facts: Facts, now: Instant) {
        if self.note_until().is_some_and(|until| until <= now) {
            self.clear_prompt(now);
        }
        let status = self.next_status(facts, now);
        let arriving = self.status == Status::Quiet && status != Status::Quiet;
        self.waiting = arriving && self.pen_is_down;
        if status != self.status && !self.waiting {
            self.status = status;
            self.ui.set_chip(status.chip(), now);
        }
        self.update_progress(facts.landing, now);
    }

    /// Moves the progress line to `landing` dabs left, if it is due to move: every
    /// [`Theme::progress_step`], so the list isn't rebuilt each frame, and at once as the title
    /// changes.
    fn update_progress(&mut self, landing: usize, now: Instant) {
        let Prompt::Finishing { left, said_at, .. } = &mut self.prompt else {
            return;
        };
        let due = now.saturating_duration_since(*said_at) >= Theme::GLASS.progress_step;
        let new_title = (landing == 0) != (*left == 0);
        if landing != *left && (due || new_title) {
            *left = landing;
            *said_at = now;
            self.show_prompt(now);
        }
    }

    /// Puts the prompt on the bar, or leaves it for the `Up` with the pen down: nothing is laid out
    /// under a stroke, and no bar can appear there.
    fn show_prompt(&mut self, now: Instant) {
        self.routing.bar_up = self.prompt != Prompt::None;
        self.routing.update();
        self.prompt_waiting = self.pen_is_down;
        if !self.prompt_waiting {
            self.ui.set_bar(self.prompt.bar(), now);
            if matches!(self.prompt, Prompt::Failed(_)) {
                self.failure_shown.get_or_insert(now);
            }
        }
    }

    /// Asks whether to save `name`, the project, before the window closes. What hasn't been seen,
    /// the pen holding it back or its fade just begun, waits behind the question.
    pub(crate) fn ask_close(&mut self, name: &str, now: Instant) {
        let was = std::mem::replace(
            &mut self.prompt,
            Prompt::AskClose {
                name: name.to_owned(),
            },
        );
        let unseen = self.prompt_waiting || !self.failure_shown(now);
        if unseen && matches!(was, Prompt::Failed(_) | Prompt::Stopped(_)) {
            self.behind = Some(was);
        }
        self.show_prompt(now);
    }

    /// Says a save the window exits after is under way, `from` dabs left to land.
    pub(crate) fn finish(&mut self, from: usize, now: Instant) {
        self.prompt = Prompt::Finishing {
            from,
            left: from,
            said_at: now,
        };
        self.show_prompt(now);
    }

    /// Says what opening found; with the bar saying a failure, once that is answered.
    pub(crate) fn note(&mut self, note: &OpenNote, now: Instant) {
        let prompt = match note {
            OpenNote::Restored { when, .. } => Prompt::Restored {
                said: said_now(*when),
                until: now + Theme::GLASS.note_hold,
            },
            OpenNote::EarlierVersion { candidate } => Prompt::EarlierVersion {
                said: said_now(candidate.modified),
                candidate: candidate.clone(),
            },
        };
        if self.failing() || self.says_stopped() {
            self.behind = Some(prompt);
        } else {
            self.prompt = prompt;
            self.show_prompt(now);
        }
    }

    /// When the note goes by itself, if one showing does.
    const fn note_until(&self) -> Option<Instant> {
        match self.prompt {
            Prompt::Restored { until, .. } => Some(until),
            Prompt::None
            | Prompt::AskClose { .. }
            | Prompt::Finishing { .. }
            | Prompt::EarlierVersion { .. }
            | Prompt::Copying
            | Prompt::Failed(_)
            | Prompt::Stopped(_) => None,
        }
    }

    /// Whether the bar says what opening found, or a copy of it is on its way.
    pub(crate) const fn noting(&self) -> bool {
        matches!(
            self.prompt,
            Prompt::Restored { .. } | Prompt::EarlierVersion { .. } | Prompt::Copying
        )
    }

    /// Whether the bar says unsaved work was restored.
    pub(crate) const fn says_restored(&self) -> bool {
        matches!(self.prompt, Prompt::Restored { .. })
    }

    /// Whether the bar says a copy is on its way.
    pub(crate) const fn copying(&self) -> bool {
        matches!(self.prompt, Prompt::Copying)
    }

    /// The bar goes from offering a copy of found work to saying it's saving: that work, if it
    /// offered one.
    pub(crate) fn begin_copy(&mut self, now: Instant) -> Option<RecoveryCandidate> {
        let Prompt::EarlierVersion { candidate, .. } = &self.prompt else {
            return None;
        };
        let candidate = candidate.clone();
        self.prompt = Prompt::Copying;
        self.show_prompt(now);
        Some(candidate)
    }

    /// Takes the bar away, and any note behind it.
    pub(crate) fn clear_prompt(&mut self, now: Instant) {
        self.prompt = Prompt::None;
        self.behind = None;
        self.show_prompt(now);
    }

    /// Says a save failed, in place of whatever the bar said; a note comes back once that is
    /// answered, as what it offers still stands.
    pub(crate) fn failed(&mut self, failure: SaveFailure, now: Instant) {
        self.saved_at = None;
        self.failure_shown = None;
        self.say_over_a_note(Prompt::Failed(failure), now);
    }

    /// Whether the failure last said has been in front of the user by `now`, its fade over.
    pub(crate) fn failure_shown(&self, now: Instant) -> bool {
        let seen = |at: Instant| now.saturating_duration_since(at) >= Theme::GLASS.fade;
        self.failure_shown.is_some_and(seen)
    }

    /// The recovery file has stopped guarding the work: a failure saying it's safe says otherwise
    /// from now on. Whether that changed what draws.
    pub(crate) fn failure_exposed(&mut self, now: Instant) -> bool {
        if let Some(Prompt::Failed(failure)) = &mut self.behind {
            *failure = failure.exposed();
        }
        let Prompt::Failed(failure) = self.prompt else {
            return false;
        };
        let exposed = failure.exposed();
        if exposed == failure {
            return false;
        }
        self.prompt = Prompt::Failed(exposed);
        self.show_prompt(now);
        true
    }

    /// Puts `prompt` on the bar, a note it replaces coming back after it.
    fn say_over_a_note(&mut self, prompt: Prompt, now: Instant) {
        let was = std::mem::replace(&mut self.prompt, prompt);
        if matches!(was, Prompt::Restored { .. } | Prompt::EarlierVersion { .. }) {
            self.behind = Some(was);
        }
        self.show_prompt(now);
    }

    /// Says crash protection has stopped, for `trouble`. Only with
    /// [`Dialogs::can_tell_of_stopping`].
    pub(crate) fn protection_stopped(&mut self, trouble: SaveTrouble, now: Instant) {
        self.say_over_a_note(Prompt::Stopped(trouble), now);
    }

    /// Whether the bar is free to say crash protection has stopped: a question or a failure comes
    /// first, and this can wait for it.
    pub(crate) const fn can_tell_of_stopping(&self) -> bool {
        matches!(
            self.prompt,
            Prompt::None | Prompt::Restored { .. } | Prompt::EarlierVersion { .. }
        )
    }

    /// Whether the bar says crash protection has stopped.
    pub(crate) const fn says_stopped(&self) -> bool {
        matches!(self.prompt, Prompt::Stopped(_))
    }

    /// Whether the bar says crash protection has stopped and the user has had the chance to see it.
    pub(crate) const fn stopping_seen(&self) -> bool {
        self.says_stopped() && !self.prompt_waiting
    }

    /// Crash protection is back: word of its stopping is out of date, showing or waiting.
    /// Whether that changed what draws.
    pub(crate) fn protection_back(&mut self, now: Instant) -> bool {
        if matches!(self.behind, Some(Prompt::Stopped(_))) {
            self.behind = None;
        }
        let showing = self.says_stopped();
        if showing {
            self.dismiss(now);
        }
        showing
    }

    /// A save was asked for: a bar saying what one is the answer to is out of date.
    pub(crate) fn save_asked(&mut self, now: Instant) {
        if self.failing() || self.says_stopped() {
            self.dismiss(now);
        }
    }

    /// Whether the bar says a save failed.
    pub(crate) const fn failing(&self) -> bool {
        matches!(self.prompt, Prompt::Failed(_))
    }

    /// Whether the bar says a save failed and the user has had the chance to see it: it didn't
    /// arrive under the pen, to wait for the `Up`.
    pub(crate) const fn failure_seen(&self) -> bool {
        self.failing() && !self.prompt_waiting
    }

    /// Puts the bar away, a note it took the place of asking again.
    pub(crate) fn dismiss(&mut self, now: Instant) {
        self.prompt = self.behind.take().unwrap_or(Prompt::None);
        self.show_prompt(now);
    }

    /// Answers the failure the bar says, if it does: [`Dialogs::dismiss`], and what failed.
    pub(crate) fn answer_failure(&mut self, now: Instant) -> Option<SaveFailure> {
        let Prompt::Failed(failure) = self.prompt else {
            return None;
        };
        self.dismiss(now);
        Some(failure)
    }

    /// Whether the bar asks about closing.
    pub(crate) const fn asking_close(&self) -> bool {
        matches!(self.prompt, Prompt::AskClose { .. })
    }

    /// Whether a prompt can't be seen, or answered, for the pen is down.
    pub(crate) const fn prompt_held_back(&self) -> bool {
        self.pen_is_down && !matches!(self.prompt, Prompt::None)
    }

    /// Whether the bar says a save the window exits after is under way.
    pub(crate) const fn finishing(&self) -> bool {
        matches!(self.prompt, Prompt::Finishing { .. })
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

    /// When something next changes by itself: the chip's "Saved" leaving, or a note.
    pub(crate) fn next_change(&self) -> Option<Instant> {
        let saved = self.saved_at.map(|at| at + Theme::GLASS.saved_hold);
        match (saved, self.note_until()) {
            (Some(one), Some(other)) => Some(one.min(other)),
            (one, other) => one.or(other),
        }
    }

    /// The pen touched down: bars fade out and the chip dims. Touches the fades only.
    pub(crate) fn pen_down(&mut self, now: Instant) {
        if self.pen_effect == PenEffect::None {
            return;
        }
        self.pen_is_down = true;
        self.ui.pen_down(now);
    }

    /// The pen lifted: everything comes back, and a bar that waited for it arrives.
    pub(crate) fn pen_up(&mut self, now: Instant) {
        if self.pen_effect == PenEffect::None {
            return;
        }
        self.pen_is_down = false;
        if self.prompt_waiting {
            self.show_prompt(now);
        }
        self.ui.pen_up(now);
    }

    /// Whether the pen is down.
    pub(crate) const fn pen_is_down(&self) -> bool {
        self.pen_is_down
    }

    /// Whether input goes through here first. All the pen pays with no bar up.
    pub(crate) const fn routes(&self) -> bool {
        self.routing.on
    }

    /// Whether the bar's glass is under `at`, and which control if one is.
    fn press_at(&self, at: Pixel, now: Instant) -> Option<Press> {
        #[cfg(test)]
        self.hit_tests.set(self.hit_tests.get() + 1);
        let (x, y) = ui_point(at);
        match self.ui.hit(x, y, now) {
            Some(action) => Some(Press::Control(action)),
            None => self.ui.on_bar(x, y, now).then_some(Press::Glass),
        }
    }

    /// Whether `at` is on the bar's glass.
    pub(crate) fn on_bar(&self, at: Pixel, now: Instant) -> bool {
        let (x, y) = ui_point(at);
        self.ui.on_bar(x, y, now)
    }

    /// Who the pen's `Down` at `at` belongs to. If it is the UI, so is the stroke, until
    /// [`Dialogs::release`] or [`Dialogs::drop_pen_press`] ends it.
    pub(crate) fn claim(&mut self, at: Pixel, now: Instant) -> Claim {
        self.routing.pen = self.press_at(at, now);
        self.routing.update();
        if self.routing.pen.is_some() {
            Claim::Ui
        } else {
            Claim::Elsewhere
        }
    }

    /// Whether the UI holds a pen stroke.
    pub(crate) const fn pen_pressed(&self) -> bool {
        self.routing.pen.is_some()
    }

    /// The pen's `Up` at `at` ended the stroke the UI held: what to do, if it ended on the control
    /// it began on.
    pub(crate) fn release(&mut self, at: Pixel, now: Instant) -> Option<Action> {
        let press = self.routing.pen.take();
        self.routing.update();
        let (x, y) = ui_point(at);
        press.and_then(|press| press.ends_over(self.ui.hit(x, y, now)))
    }

    /// The stroke the UI held ended with no `Up`: nothing was pressed after all.
    pub(crate) const fn drop_pen_press(&mut self) {
        self.routing.pen = None;
        self.routing.update();
    }

    /// Who the mouse's press of `button` at `at` belongs to. A press on the bar is the UI's until
    /// [`Dialogs::release_mouse`] or [`Dialogs::drop_mouse_press`], and only the left button acts.
    /// Another button pressed meanwhile is swallowed if it is on the bar, and held by nothing.
    pub(crate) fn claim_mouse(&mut self, button: Button, at: Pixel, now: Instant) -> Claim {
        self.routing.mouse_at = Some(at);
        let Some(press) = self.press_at(at, now) else {
            return Claim::Elsewhere;
        };
        if self.routing.mouse.is_none() {
            let press = if button == Button::MouseLeft {
                press
            } else {
                Press::Glass
            };
            self.routing.mouse = Some((button, press));
            self.routing.update();
        }
        Claim::Ui
    }

    /// Whether the press of `button` is the UI's.
    pub(crate) fn mouse_holds(&self, button: Button) -> bool {
        self.routing.mouse.is_some_and(|(held, _)| held == button)
    }

    /// The press the UI held came up, with the mouse where it last was: what to do, if that's on
    /// the control it began on.
    pub(crate) fn release_mouse(&mut self, now: Instant) -> Option<Action> {
        let press = self.routing.mouse.take();
        self.routing.update();
        let (x, y) = ui_point(self.routing.mouse_at?);
        press.and_then(|(_, press)| press.ends_over(self.ui.hit(x, y, now)))
    }

    /// The press the UI held won't come up where anyone can tell: nothing was pressed after all.
    pub(crate) const fn drop_mouse_press(&mut self) {
        self.routing.mouse = None;
        self.routing.update();
    }

    /// The pointer, the pen's or the mouse's, is at `at` (`None`: gone): whether that changed what
    /// draws.
    pub(crate) fn hover(&mut self, at: Option<Pixel>) -> bool {
        self.ui.hover(at.map(ui_point))
    }

    /// The mouse is at `at` (`None`: out of the window, where it presses nothing).
    pub(crate) const fn mouse_at(&mut self, at: Option<Pixel>) {
        self.routing.mouse_at = at;
    }
}

/// What the frame loop's tests need of a bar: one to show, and where its parts are.
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
impl Dialogs {
    /// Shows `bar`, as seen: its fade began long enough ago to be over.
    pub(crate) fn show_bar(&mut self, bar: Bar<Action>) {
        let seen = Instant::now().checked_sub(Theme::GLASS.fade * 2);
        self.ui
            .set_bar(Some(bar), seen.expect("the clock is past a fade"));
        self.routing.bar_up = true;
        self.routing.update();
    }

    /// Whether a bar is meant to show.
    pub(crate) const fn bar_showing(&self) -> bool {
        self.routing.bar_up
    }

    /// The bar as it should be.
    pub(crate) fn bar(&self) -> Option<Bar<Action>> {
        self.prompt.bar()
    }

    /// Whether a save's "Saved" is on its way or showing.
    pub(crate) const fn says_saved(&self) -> bool {
        self.saved_at.is_some()
    }

    /// Whether the chip says there are unsaved edits.
    pub(crate) fn chip_says_unsaved(&self) -> bool {
        self.status == Status::Unsaved
    }

    /// How many hit tests were made.
    pub(crate) fn hit_tests(&self) -> u32 {
        self.hit_tests.get()
    }

    /// Every pixel of the window, on a coarse grid.
    fn grid(&self) -> impl Iterator<Item = Pixel> {
        let (width, height) = self.size;
        let rows = (0..height).step_by(4);
        let points = rows.flat_map(move |y| (0..width).step_by(4).map(move |x| (x, y)));
        points.map(|(x, y)| Pixel {
            x: f64::from(x),
            y: f64::from(y),
        })
    }

    /// A point on the control that does `action`.
    pub(crate) fn control(&self, action: Action) -> Pixel {
        let now = Instant::now();
        let mut points = self.grid();
        let hits = |at: &Pixel| {
            let (x, y) = ui_point(*at);
            self.ui.hit(x, y, now) == Some(action)
        };
        points.find(hits).expect("a control that does it")
    }

    /// A point on the bar's glass, off any control.
    pub(crate) fn bare_glass(&self) -> Pixel {
        let now = Instant::now();
        let mut points = self.grid();
        let bare = |at: &Pixel| {
            let (x, y) = ui_point(*at);
            self.ui.on_bar(x, y, now) && self.ui.hit(x, y, now).is_none()
        };
        points.find(bare).expect("bare glass")
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::saving::{DabsDropped, TilesLost, Work};

    /// No edits, no save.
    const CLEAN: Facts = Facts {
        unsaved: false,
        saving: false,
        landing: 0,
    };

    /// Edits, no save.
    const EDITED: Facts = Facts {
        unsaved: true,
        saving: false,
        landing: 0,
    };

    /// A save of edits under way.
    const SAVING: Facts = Facts {
        unsaved: true,
        saving: true,
        landing: 0,
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
    fn a_bar_for_measuring_stays_through_the_pen_and_answers_nothing() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.show_test_bar(now);
        let seen = now + Theme::GLASS.fade;
        let bar = dialogs.frame(seen).expect("it draws");
        assert!(!dialogs.routes(), "input goes where it went");

        dialogs.pen_down(seen);
        dialogs.update(EDITED, seen);
        let later = seen + Theme::GLASS.fade * 2;
        let during = dialogs.frame(later).expect("it draws");
        let opaque = |frame: &UiFrame| frame.fades.iter().map(|fade| fade.at(later)).sum::<f32>();
        assert!(
            (opaque(&during) - 2.0).abs() < 1e-6,
            "both at full strength"
        );
        assert!(
            !Arc::ptr_eq(&bar.list, &during.list),
            "the chip came under the pen, to be measured"
        );
        assert_eq!(dialogs.status, Status::Unsaved);
        dialogs.pen_up(later);
        assert!(dialogs.frame(later).is_some(), "and the bar stays");
    }

    /// A disk gone full, the work whole and protected.
    fn full_disk() -> SaveFailure {
        let full = std::io::Error::from(std::io::ErrorKind::StorageFull);
        let error = anyhow::Error::new(full);
        SaveFailure::of(&error, Work::Whole, &store::Protection::Active)
    }

    /// What `dialogs` shows on the bar: its controls' actions.
    fn actions(dialogs: &Dialogs) -> Vec<Action> {
        let bar = dialogs.bar().expect("a bar");
        bar.controls().iter().map(|c| c.action).collect()
    }

    #[test]
    fn the_bar_says_your_work_is_safe_and_offers_try_again() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.failed(full_disk(), now);
        assert!(dialogs.failing());
        let bar = dialogs.bar().expect("a bar");
        let expected = Bar::new("Couldn't save: the disk is full", bar.controls().to_vec())
            .with_detail("Your work is safe and still open.")
            .needing_you();
        assert_eq!(bar, expected, "its words and coral hairline");
        assert_eq!(actions(&dialogs), [Action::Dismiss, Action::TryAgain]);
        let filled = bar
            .controls()
            .iter()
            .filter(|c| c.emphasis == Emphasis::Filled);
        let filled: Vec<&Control<Action>> = filled.collect();
        assert_eq!(
            filled.first().map(|c| c.action),
            Some(Action::TryAgain),
            "the one action"
        );
        assert!(dialogs.routes(), "input comes here first");
        assert!(dialogs.frame(now).is_some(), "it draws");
    }

    /// The longest words at the smallest window lose no letter: every glyph of the bar draws. An
    /// ellipsis takes more letters than it adds, and a space draws none.
    #[test]
    fn every_failure_fits_the_smallest_window() {
        let now = Instant::now();
        let dropped = anyhow::Error::new(DabsDropped(String::new()));
        let locked = anyhow::Error::new(store::FileError::Locked);
        let cause = store::ProjectError::NoPath;
        let paused = store::Protection::Paused(Arc::new(cause));
        let active = store::Protection::Active;
        let failures = [
            SaveFailure::of(&locked, Work::LessAStroke, &paused),
            SaveFailure::of(&dropped, Work::Whole, &paused),
            SaveFailure::of(&locked, Work::Whole, &active),
            SaveFailure::of_copy(&locked),
        ];
        for failure in failures {
            let mut dialogs = Dialogs::new(Size::new(640, 400), 1.0, Blend::Display, now);
            dialogs.failed(failure, now);
            let frame = dialogs.frame(now).expect("it draws");
            let bar = dialogs.bar().expect("a bar");
            let labels = bar.controls().iter().map(|c| c.label.as_str());
            let said = [bar.title(), bar.detail().unwrap_or_default()];
            let letters = said.into_iter().chain(labels).flat_map(str::chars);
            let letters = letters.filter(|letter| !letter.is_whitespace()).count();
            assert_eq!(frame.list.glyphs.len(), letters, "{}", failure.detail());
        }
        assert_stopped_fits(SaveTrouble::DiskFull, (640, 400));
        let longest = [SaveTrouble::NoPermission, SaveTrouble::InUse];
        for trouble in longest {
            assert_stopped_fits(trouble, (1280, 800));
        }
    }

    /// Word of crash protection stopping for `trouble` loses no letter in a window of `size`.
    fn assert_stopped_fits(trouble: SaveTrouble, (width, height): (u32, u32)) {
        let now = Instant::now();
        let mut dialogs = Dialogs::new(Size::new(width, height), 1.0, Blend::Display, now);
        dialogs.protection_stopped(trouble, now);
        let frame = dialogs.frame(now).expect("it draws");
        let bar = dialogs.bar().expect("a bar");
        let labels = bar.controls().iter().map(|c| c.label.as_str());
        let said = [bar.title(), bar.detail().unwrap_or_default()];
        let letters = said.into_iter().chain(labels).flat_map(str::chars);
        let letters = letters.filter(|letter| !letter.is_whitespace()).count();
        assert_eq!(frame.list.glyphs.len(), letters, "{}", bar.title());
    }

    #[test]
    fn crash_protection_stopping_says_so_and_offers_save() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        assert!(dialogs.can_tell_of_stopping());
        dialogs.protection_stopped(SaveTrouble::DiskFull, now);
        let bar = dialogs.bar().expect("a bar");
        let expected = Bar::new(
            "Crash protection has stopped: the disk is full",
            bar.controls().to_vec(),
        )
        .with_detail("Save your work when you can.")
        .needing_you();
        assert_eq!(bar, expected, "its words and coral hairline");
        assert_eq!(actions(&dialogs), [Action::Dismiss, Action::SaveWork]);
        let filled = bar.controls().iter().find(|c| c.action == Action::SaveWork);
        assert_eq!(filled.map(|c| c.emphasis), Some(Emphasis::Filled));
        assert!(!dialogs.can_tell_of_stopping(), "once is enough");
        dialogs.save_asked(now);
        assert!(!dialogs.says_stopped(), "a save is the answer");
    }

    /// A question stays: it is the user's, and this can wait for it.
    #[test]
    fn crash_protection_stopping_waits_for_a_question_or_a_failure() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.ask_close("first-sculpt", now);
        assert!(!dialogs.can_tell_of_stopping());
        dialogs.clear_prompt(now);
        dialogs.failed(full_disk(), now);
        assert!(!dialogs.can_tell_of_stopping());
    }

    #[test]
    fn a_note_comes_back_after_crash_protection_stopping() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&earlier(SystemTime::now()), now);
        assert!(dialogs.can_tell_of_stopping(), "a note only tells");
        dialogs.protection_stopped(SaveTrouble::Other, now);
        assert!(dialogs.says_stopped());
        dialogs.dismiss(now);
        assert!(dialogs.noting());
    }

    /// With nothing to try, the bar offers nothing to try.
    #[test]
    fn edits_gone_are_only_dismissed() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        let lost = anyhow::Error::new(TilesLost(2));
        let failure = SaveFailure::of(&lost, Work::Whole, &store::Protection::Active);
        dialogs.failed(failure, now);
        assert_eq!(actions(&dialogs), [Action::Dismiss]);
    }

    #[test]
    fn it_waits_for_the_up_if_the_pen_is_down() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.pen_down(now);
        dialogs.failed(full_disk(), now);
        assert!(dialogs.failing(), "it is meant to show");
        assert!(!dialogs.failure_seen(), "and hasn't been");
        assert!(dialogs.frame(now).is_none(), "nothing draws under the pen");
        dialogs.pen_up(now);
        assert!(dialogs.failure_seen(), "the Up brings it");
        let frame = dialogs.frame(now).expect("it draws");
        assert!(!frame.list.is_empty());
    }

    #[test]
    fn a_note_behind_a_failure_asks_again_once_it_is_answered() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        let error = anyhow::anyhow!("no room");
        dialogs.failed(SaveFailure::of_copy(&error), now);
        dialogs.note(&earlier(SystemTime::now()), now);
        assert!(dialogs.failing(), "the failure comes first");
        assert!(!dialogs.noting());
        dialogs.dismiss(now);
        assert!(!dialogs.failing());
        assert!(dialogs.noting(), "and the note asks again");
        assert_eq!(actions(&dialogs), [Action::Ignore, Action::OpenCopy]);
    }

    #[test]
    fn a_failure_over_a_note_gives_it_back() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&earlier(SystemTime::now()), now);
        dialogs.failed(full_disk(), now);
        assert!(dialogs.failing());
        assert_eq!(dialogs.answer_failure(now), Some(full_disk()));
        assert!(dialogs.noting());
        assert_eq!(dialogs.answer_failure(now), None, "no failure says so");
    }

    #[test]
    fn clearing_the_bar_takes_a_note_behind_it_too() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&earlier(SystemTime::now()), now);
        dialogs.failed(full_disk(), now);
        dialogs.clear_prompt(now);
        assert!(!dialogs.routes(), "no bar");
        dialogs.dismiss(now);
        assert!(!dialogs.noting(), "and none comes back");
    }

    /// A failure says what it says for as long as it takes: unlike a note, it never goes by itself.
    #[test]
    fn a_failure_stays_until_it_is_answered() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.failed(full_disk(), now);
        assert_eq!(dialogs.next_change(), None, "nothing to wait for");
        let later = now + Theme::GLASS.note_hold * 10;
        dialogs.update(EDITED, later);
        assert!(dialogs.failing());
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

    /// A save under way, with `landing` dabs left.
    const fn landing(landing: usize) -> Facts {
        Facts { landing, ..SAVING }
    }

    /// Whether a bar's fade heads up at `now`.
    fn bar_rising(dialogs: &mut Dialogs, now: Instant) -> bool {
        dialogs
            .frame(now)
            .is_some_and(|frame| frame.fades[0].to > 0.5)
    }

    #[test]
    fn the_close_bar_says_it_in_the_users_words() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.ask_close("first-sculpt", now);
        let quiet = |label: &str, action| Control {
            label: label.to_owned(),
            action,
            emphasis: Emphasis::Quiet,
        };
        let save = Control {
            label: "Save".to_owned(),
            action: Action::Save,
            emphasis: Emphasis::Filled,
        };
        let controls = vec![
            quiet("Don't save", Action::DontSave),
            quiet("Keep sculpting", Action::KeepSculpting),
            save,
        ];
        let bar = Bar::new("Save changes to first-sculpt?", controls);
        assert_eq!(dialogs.bar(), Some(bar));
    }

    #[test]
    fn finishing_says_saving_once_nothing_is_left_to_land() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        let cancel = || Control {
            label: "Cancel".to_owned(),
            action: Action::Cancel,
            emphasis: Emphasis::Quiet,
        };
        dialogs.finish(10, now);
        let finishing = Bar::new("Finishing your stroke", vec![cancel()]).with_progress(0.0);
        assert_eq!(dialogs.bar(), Some(finishing), "no counts");
        dialogs.update(landing(0), now);
        let saving = Bar::new("Saving", vec![cancel()]).with_progress(1.0);
        assert_eq!(dialogs.bar(), Some(saving), "at once, whatever the time");
    }

    #[test]
    fn nothing_to_land_goes_straight_to_saving() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.finish(0, now);
        let cancel = Control {
            label: "Cancel".to_owned(),
            action: Action::Cancel,
            emphasis: Emphasis::Quiet,
        };
        let saving = Bar::new("Saving", vec![cancel]).with_progress(1.0);
        assert_eq!(dialogs.bar(), Some(saving));
    }

    #[test]
    fn progress_moves_at_most_every_100_ms_and_the_list_rebuilds_no_oftener() {
        let now = Instant::now();
        let step = Theme::GLASS.progress_step;
        let mut dialogs = dialogs(now);
        dialogs.update(landing(10), now);
        dialogs.finish(10, now);
        let first = dialogs.frame(now).expect("the bar");

        let early = now + step / 2;
        dialogs.update(landing(5), early);
        let same = dialogs.frame(early).expect("the bar");
        assert!(Arc::ptr_eq(&first.list, &same.list), "too soon");

        let due = now + step;
        dialogs.update(landing(5), due);
        let moved = dialogs.frame(due).expect("the bar");
        assert!(!Arc::ptr_eq(&first.list, &moved.list), "due");
        assert!(dialogs.needs_frame(due), "and it wants the frame");
    }

    #[test]
    fn progress_never_runs_backwards_past_nothing() {
        assert!(progress(20, 10).abs() < 1e-6, "more than it began with");
        assert!((progress(5, 10) - 0.5).abs() < 1e-6, "half");
        assert!((progress(0, 0) - 1.0).abs() < 1e-6, "nothing to wait for");
    }

    #[test]
    fn a_prompt_under_the_pen_waits_for_the_up_and_builds_nothing() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.update(EDITED, now);
        let before = dialogs.frame(now).expect("the chip");
        dialogs.pen_down(now);
        dialogs.ask_close("first-sculpt", now);
        assert!(dialogs.routes(), "input goes through here");
        let during = dialogs.frame(now).expect("the chip");
        assert!(Arc::ptr_eq(&before.list, &during.list), "nothing laid out");
        assert!(!bar_rising(&mut dialogs, now), "and nothing shows");

        let later = now + Theme::GLASS.fade * 3;
        dialogs.update(EDITED, later);
        assert!(!bar_rising(&mut dialogs, later), "however long it takes");

        dialogs.pen_up(later);
        assert!(bar_rising(&mut dialogs, later), "here it comes");
        assert!(dialogs.needs_frame(later), "in frames of its own");
    }

    #[test]
    fn a_prompt_taken_away_under_the_pen_never_comes_back() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.ask_close("first-sculpt", now);
        let seen = now + Theme::GLASS.fade;
        dialogs.pen_down(seen);
        dialogs.clear_prompt(seen);
        assert!(!dialogs.routes(), "the bar's over");
        dialogs.pen_up(seen);
        assert!(!bar_rising(&mut dialogs, seen), "it stays gone");
    }

    /// The moment `at` (`YYYY-MM-DDTHH:MM`), `hours` ahead of UTC.
    fn zoned(at: &str, hours: i8) -> Zoned {
        let moment: jiff::civil::DateTime = at.parse().expect("a date and time");
        let zone = jiff::tz::TimeZone::fixed(jiff::tz::offset(hours));
        moment.to_zoned(zone).expect("a moment")
    }

    #[test]
    fn said_when_says_today_yesterday_or_the_date() {
        let now = zoned("2026-11-05T09:30", 0);
        let said = |then: &str| said_when(&zoned(then, 0), &now);
        assert_eq!(said("2026-11-05T00:00"), "From today at 00:00");
        assert_eq!(said("2026-11-04T16:42"), "From yesterday at 16:42");
        assert_eq!(said("2026-11-03T16:42"), "From 3 Nov at 16:42");
        assert_eq!(said("2025-11-04T16:42"), "From 4 Nov 2025 at 16:42");

        let new_year = zoned("2027-01-01T08:05", 0);
        let across = |then: &str| said_when(&zoned(then, 0), &new_year);
        assert_eq!(across("2026-12-31T23:59"), "From yesterday at 23:59");
        assert_eq!(across("2026-12-30T23:59"), "From 30 Dec 2026 at 23:59");
    }

    #[test]
    fn said_when_goes_by_the_zone_of_now() {
        // Late yesterday where it was written, the next morning here.
        let then = zoned("2026-11-04T23:30", 0);
        let now = zoned("2026-11-05T10:00", 9);
        assert_eq!(said_when(&then, &now), "From today at 08:30");
    }

    /// The note of a restored recovery file written at `when`.
    fn restored(when: SystemTime) -> OpenNote {
        OpenNote::Restored {
            path: "recovery.bcrf".into(),
            when,
        }
    }

    #[test]
    fn the_restored_note_says_so_and_offers_the_last_save() {
        let now = Instant::now();
        let when = SystemTime::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&restored(when), now);
        let last_save = Control {
            label: "Use last save instead".to_owned(),
            action: Action::UseLastSave,
            emphasis: Emphasis::Quiet,
        };
        let said = said_now(when).expect("a time the clock can say");
        let bar = Bar::new("Your unsaved work is back", vec![last_save]).with_detail(said);
        assert_eq!(dialogs.bar(), Some(bar));
    }

    #[test]
    fn the_note_leaves_after_ten_seconds_and_says_when_to_come_back() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&restored(SystemTime::now()), now);
        let almost = now + Duration::from_millis(9_999);
        let until = now + Duration::from_secs(10);
        assert_eq!(dialogs.next_change(), Some(until), "the idle loop wakes");
        dialogs.update(CLEAN, almost);
        assert!(dialogs.noting(), "still there");
        dialogs.update(CLEAN, until);
        assert!(!dialogs.noting(), "gone");
        assert!(!dialogs.routes(), "and nothing goes through it");
        assert_eq!(dialogs.next_change(), None, "nothing more to come");
        assert!(dialogs.needs_frame(until), "fading out");
    }

    #[test]
    fn a_note_and_a_saved_chip_say_the_sooner_of_the_two() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&restored(SystemTime::now()), now);
        dialogs.saved(now);
        assert_eq!(dialogs.next_change(), Some(now + hold()), "the chip first");
    }

    #[test]
    fn the_close_bar_taking_over_stops_the_notes_clock() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&restored(SystemTime::now()), now);
        dialogs.ask_close("first-sculpt", now);
        let later = now + Duration::from_secs(11);
        dialogs.update(EDITED, later);
        assert!(dialogs.asking_close(), "the question stays");
    }

    /// The note of work found on an earlier save, written at `when`.
    fn earlier(when: SystemTime) -> OpenNote {
        let candidate = RecoveryCandidate {
            path: "recovery.bcrf".into(),
            project: store::ProjectId::from_bits(1),
            base: None,
            base_path: None,
            last_edit: None,
            modified: when,
            status: store::CandidateStatus::RestorableOnEarlierSave,
        };
        OpenNote::EarlierVersion { candidate }
    }

    #[test]
    fn the_earlier_version_note_asks_and_waits_for_as_long_as_it_takes() {
        let now = Instant::now();
        let when = SystemTime::now();
        let mut dialogs = dialogs(now);
        dialogs.note(&earlier(when), now);
        let ignore = Control {
            label: "Ignore".to_owned(),
            action: Action::Ignore,
            emphasis: Emphasis::Quiet,
        };
        let copy = Control {
            label: "Open as a copy".to_owned(),
            action: Action::OpenCopy,
            emphasis: Emphasis::Filled,
        };
        let said = said_now(when).expect("a time the clock can say");
        let title = "Unsaved work found from an earlier version";
        let bar = Bar::new(title, vec![ignore, copy]).with_detail(said);
        assert_eq!(dialogs.bar(), Some(bar));
        assert_eq!(dialogs.next_change(), None, "it asks a question");
        dialogs.update(CLEAN, now + Duration::from_secs(3600));
        assert!(dialogs.noting(), "still asking");
    }

    #[test]
    fn beginning_a_copy_says_saving_and_hands_over_what_was_found() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        assert_eq!(dialogs.begin_copy(now), None, "nothing to copy");
        dialogs.note(&restored(SystemTime::now()), now);
        assert_eq!(dialogs.begin_copy(now), None, "nor with this note");
        assert!(dialogs.says_restored(), "which stays");

        dialogs.note(&earlier(SystemTime::now()), now);
        let found = dialogs.begin_copy(now).expect("what was found");
        assert_eq!(found.path, std::path::Path::new("recovery.bcrf"));
        assert_eq!(dialogs.bar(), Some(Bar::new("Saving", Vec::new())));
        assert!(dialogs.copying() && dialogs.noting());
        assert!(dialogs.frame(now).is_some(), "and it draws");
    }

    #[test]
    fn a_note_under_the_pen_waits_for_the_up() {
        let now = Instant::now();
        let mut dialogs = dialogs(now);
        dialogs.pen_down(now);
        dialogs.note(&restored(SystemTime::now()), now);
        assert!(!bar_rising(&mut dialogs, now), "nothing shows");
        dialogs.pen_up(now);
        assert!(bar_rising(&mut dialogs, now), "here it comes");
    }

    #[test]
    fn every_bar_fits_the_smallest_window_at_any_scale() {
        let now = Instant::now();
        let (width, height) = Theme::GLASS.window_min;
        for scale in [1.0, 1.25, 2.0] {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a small whole window size"
            )]
            let size = Size::new(
                (f64::from(width) * scale) as u32,
                (f64::from(height) * scale) as u32,
            );
            let mut dialogs = Dialogs::new(size, scale, Blend::Display, now);
            let seen = now + Theme::GLASS.fade;
            dialogs.ask_close("a project with a name as long as anyone will give it", now);
            let close = [Action::DontSave, Action::KeepSculpting, Action::Save];
            assert_controls_found(&dialogs, &close, seen, scale);

            dialogs.note(&restored(SystemTime::now()), now);
            assert_controls_found(&dialogs, &[Action::UseLastSave], seen, scale);

            dialogs.note(&earlier(SystemTime::now()), now);
            let found = [Action::Ignore, Action::OpenCopy];
            assert_controls_found(&dialogs, &found, seen, scale);
        }
    }

    /// Checks `actions` can all be hit at `seen`.
    fn assert_controls_found(dialogs: &Dialogs, actions: &[Action], seen: Instant, scale: f64) {
        for &action in actions {
            let mut points = dialogs.grid();
            let found = points.find(|&at| {
                let (x, y) = ui_point(at);
                dialogs.ui.hit(x, y, seen) == Some(action)
            });
            assert!(found.is_some(), "{action:?} at {scale}");
        }
    }
}
