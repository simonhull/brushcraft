//! What the UI shows and when: the app's whole UI state, on top of the `ui` crate.
//!
//! `ui` knows widgets and nothing of projects; this says which ones the project's state calls for,
//! in the user's words. For now that is the save chip. The owner thread brings it up to date once
//! a frame, after the pen's work ([`Dialogs::update`]), and what it shows goes to the render thread
//! with the frame. The pen touches only the fades ([`Dialogs::pen_down`], [`Dialogs::pen_up`]), and
//! a chip where there was none waits for it to lift, so nothing appears, and no atlas uploads,
//! during a stroke.
//!
//! **Routing:** with a bar up, a `Down` or a press goes here first ([`Dialogs::claim`],
//! [`Dialogs::claim_mouse`]). One on the bar is the UI's, whole: the stroke's other samples stop
//! here too, and its control acts only if the stroke ends with an `Up` inside it
//! ([`Dialogs::release`], [`Dialogs::release_mouse`]). With no bar the owner's one look at
//! [`Dialogs::routes`] is all the pen pays.

use std::time::Instant;

use render::surface::Size;
use ui::{Blend, Chip, Mark, Theme, Ui, UiFrame};

use crate::camera::Pixel;
use crate::navigate::Button;

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

/// Who has the pointer's input: the bar, and what it has been pressed with.
#[derive(Debug, Default)]
struct Routing {
    /// Whether a bar is meant to show.
    bar_up: bool,
    /// The pen stroke the UI took, while it lasts.
    pen: Option<Press>,
    /// The mouse press the UI took, while it lasts, and its button.
    mouse: Option<(Button, Press)>,
    /// Where the pointer last was, while input is routed here: a release doesn't say.
    pointer: Option<Pixel>,
    /// Whether input goes through here first: a bar is up, or a press it took is still down.
    on: bool,
}

impl Routing {
    /// Brings `on` up to date.
    const fn update(&mut self) {
        self.on = self.bar_up || self.pen.is_some() || self.mouse.is_some();
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
            routing: Routing::default(),
            #[cfg(test)]
            hit_tests: std::cell::Cell::new(0),
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
        self.routing.pointer = Some(at);
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

    /// The press the UI held came up, with the pointer where it last was: what to do, if that's on
    /// the control it began on.
    pub(crate) fn release_mouse(&mut self, now: Instant) -> Option<Action> {
        let press = self.routing.mouse.take();
        self.routing.update();
        let (x, y) = ui_point(self.routing.pointer?);
        press.and_then(|(_, press)| press.ends_over(self.ui.hit(x, y, now)))
    }

    /// The press the UI held won't come up where anyone can tell: nothing was pressed after all.
    pub(crate) const fn drop_mouse_press(&mut self) {
        self.routing.mouse = None;
        self.routing.update();
    }

    /// The pointer is at `at` (`None`: gone): whether that changed what draws.
    pub(crate) fn hover(&mut self, at: Option<Pixel>) -> bool {
        self.routing.pointer = at;
        self.ui.hover(at.map(ui_point))
    }

    /// Takes the bar away.
    pub(crate) fn clear_bar(&mut self, now: Instant) {
        self.routing.bar_up = false;
        self.routing.update();
        self.ui.set_bar(None, now);
    }
}

/// What the frame loop's tests need of a bar: one to show, and where its parts are.
#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
impl Dialogs {
    /// Shows `bar`, as seen: its fade began long enough ago to be over.
    pub(crate) fn show_bar(&mut self, bar: ui::Bar<Action>) {
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
