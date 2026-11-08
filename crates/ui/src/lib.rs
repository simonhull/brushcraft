//! A small headless UI: tokens, text and a few widgets, turned into a list of shapes to draw.
//!
//! The crate knows nothing of BrushCraft and depends on nothing else here, so a painting app could
//! use it unchanged. It opens no window and touches no GPU: a renderer draws what it hands over,
//! and its tests run anywhere. Design: `specs/2026-11-04-ui-foundation-design.md` in the docs repo.

mod draw;
mod raster;
mod text;
mod theme;
mod widgets;

use std::sync::Arc;
use std::time::Instant;

pub use draw::{DrawList, Fade, GlyphQuad, Layer, Rect, RoundedRect, UiFrame};
pub use text::{AtlasImage, Line, PlacedGlyph, Text, Weight};
pub use theme::{Blend, Colour, Theme};
pub use widgets::{Bar, Chip, Control, Emphasis, Mark};
use widgets::{BarLayout, ChipLayout, View};

/// A widget on screen, or on its way off.
#[derive(Debug)]
struct Shown<W, L> {
    /// What was asked for.
    widget: W,
    /// Where it is.
    layout: L,
    /// It has been taken away, and draws only until its fade ends.
    leaving: bool,
}

/// The UI's state: what shows, what the pen is doing, and the list of shapes they make.
///
/// `A` is what a control does, handed back by [`Ui::hit`]. The caller says what should show as
/// often as it likes; the list is rebuilt only when that changes, and the pen touches nothing but
/// the fades. Anything new while the pen is down waits for the up.
#[derive(Debug)]
pub struct Ui<A> {
    /// The look.
    theme: Theme,
    /// The fonts and atlas.
    text: Text,
    /// The window.
    view: View,
    /// The bar.
    bar: Option<Shown<Bar<A>, BarLayout>>,
    /// The chip.
    chip: Option<Shown<Chip, ChipLayout>>,
    /// Whether the pen is down.
    pen_is_down: bool,
    /// Where the pointer last was.
    pointer: Option<(f32, f32)>,
    /// The index of the control showing hover.
    hovered: Option<usize>,
    /// When the bar's controls can be hit: a while after they change, the theme's `swap_lockout`.
    hits_from: Instant,
    /// The bars' opacity.
    bar_fade: Fade,
    /// The chip's opacity.
    chip_fade: Fade,
    /// The shapes as last built.
    list: Arc<DrawList>,
    /// `list` no longer says what should show.
    stale: bool,
}

impl<A: Copy + PartialEq> Ui<A> {
    /// A UI showing nothing, for a renderer that blends as `blend` says, in a window of `size`
    /// physical pixels at `scale` of them a logical one, with printable ASCII ready to draw.
    #[must_use]
    pub fn new(theme: Theme, blend: Blend, size: (u32, u32), scale: f32, now: Instant) -> Self {
        let mut ui = Self {
            theme: theme.for_blend(blend),
            text: Text::new(),
            view: view(size, scale),
            bar: None,
            chip: None,
            pen_is_down: false,
            pointer: None,
            hovered: None,
            hits_from: now,
            bar_fade: Fade::at_rest(0.0, now),
            chip_fade: Fade::at_rest(0.0, now),
            list: Arc::default(),
            stale: false,
        };
        ui.warm();
        ui
    }

    /// Moves to a window of `size` at `scale`. A new scale starts the atlas afresh with printable
    /// ASCII at its sizes, so scales passed through never fill it.
    pub fn set_view(&mut self, size: (u32, u32), scale: f32) {
        let next = view(size, scale);
        if next == self.view {
            return;
        }
        let rescaled = (next.scale - self.view.scale).abs() > f32::EPSILON;
        self.view = next;
        if rescaled {
            self.text.reset();
            self.warm();
        }
        if let Some(shown) = &mut self.bar {
            shown.layout = BarLayout::new(&shown.widget, next, &self.theme, &mut self.text);
        }
        if let Some(shown) = &mut self.chip {
            shown.layout = ChipLayout::new(&shown.widget, next, &self.theme, &mut self.text);
        }
        self.hovered = self.control_under_pointer();
        self.stale = true;
    }

    /// Says which bar should show, if any. The same bar again does nothing, so this may be called
    /// every frame. One bar for another swaps at once; a bar arriving or leaving fades.
    pub fn set_bar(&mut self, bar: Option<Bar<A>>, now: Instant) {
        let held = self.bar.as_ref().filter(|shown| !shown.leaving);
        if held.map(|shown| &shown.widget) == bar.as_ref() {
            return;
        }
        if let Some(widget) = bar {
            let swapped = self.bar.as_ref();
            if swapped.is_some_and(|shown| shown.widget.controls() != widget.controls()) {
                self.hits_from = now + self.theme.swap_lockout;
            }
            let layout = BarLayout::new(&widget, self.view, &self.theme, &mut self.text);
            self.bar = Some(Shown {
                widget,
                layout,
                leaving: false,
            });
            self.hovered = self.control_under_pointer();
            self.stale = true;
        } else if let Some(shown) = &mut self.bar {
            shown.leaving = true;
        }
        self.aim_fades(now);
    }

    /// Says which chip should show, if any. As [`Ui::set_bar`].
    pub fn set_chip(&mut self, chip: Option<Chip>, now: Instant) {
        let held = self.chip.as_ref().filter(|shown| !shown.leaving);
        if held.map(|shown| &shown.widget) == chip.as_ref() {
            return;
        }
        if let Some(widget) = chip {
            let layout = ChipLayout::new(&widget, self.view, &self.theme, &mut self.text);
            self.chip = Some(Shown {
                widget,
                layout,
                leaving: false,
            });
            self.stale = true;
        } else if let Some(shown) = &mut self.chip {
            shown.leaving = true;
        }
        self.aim_fades(now);
    }

    /// Touches the fades only: no rebuild, nothing laid out. On the pen's path.
    pub fn pen_down(&mut self, now: Instant) {
        self.pen_is_down = true;
        self.aim_fades(now);
    }

    /// The pen is up: bars return, and the hover catches up with where the pointer went.
    pub fn pen_up(&mut self, now: Instant) {
        self.pen_is_down = false;
        self.aim_fades(now);
        self.refresh_hover();
    }

    /// The action under the point at `now`; `None` with no bar, the pen down, a bar still too faint
    /// to have been seen, so a quick tap can't land on what isn't there yet, or controls just
    /// swapped for others, so a double click can't land on what its first click brought up.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32, now: Instant) -> Option<A> {
        let shown = self.visible_bar(now).filter(|_| now >= self.hits_from)?;
        let at = shown.layout.control_at(x, y)?;
        shown
            .widget
            .controls()
            .get(at)
            .map(|control| control.action)
    }

    /// Whether the point is on the bar's glass at all at `now`, by the same test as [`Ui::hit`],
    /// but for controls just swapped: a press there still belongs to the bar.
    #[must_use]
    pub fn on_bar(&self, x: f32, y: f32, now: Instant) -> bool {
        self.visible_bar(now)
            .is_some_and(|shown| shown.layout.glass.contains(x, y))
    }

    /// Moves the hover; true if that changed what draws. With the pen down nothing changes.
    pub fn hover(&mut self, at: Option<(f32, f32)>) -> bool {
        self.pointer = at;
        if self.pen_is_down {
            return false;
        }
        self.refresh_hover()
    }

    /// What to draw, rebuilding only if something changed; `None` with nothing to show.
    pub fn frame(&mut self, now: Instant) -> Option<UiFrame> {
        self.drop_what_left(now);
        if !self.showing(now) {
            return None;
        }
        if self.stale {
            self.list = Arc::new(self.build());
            self.stale = false;
        }
        Some(UiFrame {
            list: Arc::clone(&self.list),
            atlas: Arc::clone(self.text.atlas()),
            fades: [self.bar_fade, self.chip_fade],
        })
    }

    /// Whether a fade is moving at `now`.
    #[must_use]
    pub fn animating(&self, now: Instant) -> bool {
        !self.bar_fade.settled(now) || !self.chip_fade.settled(now)
    }

    /// Whether a frame is wanted at `now`: a fade is moving, or what shows has changed since the
    /// last frame, as one bar swapped for another does with no fade. A change nothing would show,
    /// such as a bar waiting for the pen, wants none.
    #[must_use]
    pub fn needs_frame(&self, now: Instant) -> bool {
        self.animating(now) || (self.stale && self.showing(now))
    }

    /// Rasterises printable ASCII at every size the widgets use, at this scale.
    fn warm(&mut self) {
        let Theme { body, status, .. } = self.theme;
        let scale = self.view.scale;
        self.text.warm(&[
            (self.theme.title_weight, body.0 * scale),
            (self.theme.control_weight, body.0 * scale),
            (self.theme.filled_weight, body.0 * scale),
            (self.theme.status_weight, status.0 * scale),
        ]);
    }

    /// Whether anything can be seen at `now`.
    fn showing(&self, now: Instant) -> bool {
        let visible = |shown: bool, fade: Fade| shown && (fade.to > 0.0 || !fade.settled(now));
        visible(self.bar.is_some(), self.bar_fade) || visible(self.chip.is_some(), self.chip_fade)
    }

    /// The bar, if it isn't going or held back by the pen.
    fn live_bar(&self) -> Option<&Shown<Bar<A>, BarLayout>> {
        let shown = self.bar.as_ref()?;
        (!shown.leaving && !self.pen_is_down).then_some(shown)
    }

    /// The bar, if it's live and at least half there at `now`.
    fn visible_bar(&self, now: Instant) -> Option<&Shown<Bar<A>, BarLayout>> {
        self.live_bar().filter(|_| self.bar_fade.at(now) >= 0.5)
    }

    /// Brings the hover up to date with the pointer; whether that changed what draws.
    fn refresh_hover(&mut self) -> bool {
        let hovered = self.control_under_pointer();
        let changed = hovered != self.hovered;
        self.hovered = hovered;
        self.stale |= changed;
        changed
    }

    /// The quiet control under the pointer.
    fn control_under_pointer(&self) -> Option<usize> {
        let (x, y) = self.pointer?;
        self.live_bar()?.layout.quiet_control_at(x, y)
    }

    /// Points the fades at what should show now.
    fn aim_fades(&mut self, now: Instant) {
        let length = self.theme.fade;
        let bar_wanted = self.bar.as_ref().is_some_and(|shown| !shown.leaving);
        let bar = if bar_wanted && !self.pen_is_down {
            1.0
        } else {
            0.0
        };
        self.bar_fade = self.bar_fade.toward(bar, length, now);
        let chip_wanted = self.chip.as_ref().is_some_and(|shown| !shown.leaving);
        let chip = match (chip_wanted, self.pen_is_down) {
            (false, _) => 0.0,
            (true, true) => self.theme.chip_pen_down,
            (true, false) => 1.0,
        };
        self.chip_fade = self.chip_fade.toward(chip, length, now);
    }

    /// Forgets a widget that has finished leaving.
    fn drop_what_left(&mut self, now: Instant) {
        if self.bar_fade.settled(now) && self.bar.as_ref().is_some_and(|shown| shown.leaving) {
            self.bar = None;
            self.hovered = None;
            self.stale = true;
        }
        if self.chip_fade.settled(now) && self.chip.as_ref().is_some_and(|shown| shown.leaving) {
            self.chip = None;
            self.stale = true;
        }
    }

    /// The shapes of what shows.
    fn build(&self) -> DrawList {
        let mut list = DrawList::default();
        if let Some(shown) = &self.bar {
            shown
                .layout
                .draw(&self.theme, self.view, self.hovered, &mut list);
        }
        if let Some(shown) = &self.chip {
            shown.layout.draw(&mut list);
        }
        list
    }
}

/// The view of a window of `size` physical pixels at `scale`.
fn view(size: (u32, u32), scale: f32) -> View {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a window's side is far inside f32's whole numbers"
    )]
    let (width, height) = (size.0 as f32, size.1 as f32);
    View {
        width,
        height,
        scale,
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use super::*;

    /// The window in these tests, physical pixels.
    const WINDOW: (u32, u32) = (1280, 800);

    /// A UI at scale 1 in [`WINDOW`], showing nothing.
    fn new_ui(now: Instant) -> Ui<u8> {
        Ui::new(Theme::GLASS, Blend::Display, WINDOW, 1.0, now)
    }

    /// A control drawn quietly.
    fn quiet(label: &str, action: u8) -> Control<u8> {
        Control {
            label: label.to_owned(),
            action,
            emphasis: Emphasis::Quiet,
        }
    }

    /// The control a bar fills.
    fn filled(label: &str, action: u8) -> Control<u8> {
        Control {
            label: label.to_owned(),
            action,
            emphasis: Emphasis::Filled,
        }
    }

    /// The close bar's controls, under another title: actions 1, 2 and 3.
    fn bar_titled(title: impl Into<String>) -> Bar<u8> {
        let controls = vec![
            quiet("Don't save", 1),
            quiet("Keep sculpting", 2),
            filled("Save", 3),
        ];
        Bar::new(title, controls)
    }

    /// The close bar.
    fn close_bar() -> Bar<u8> {
        bar_titled("Save changes to first-sculpt?")
    }

    /// A chip with a dot.
    fn unsaved() -> Chip {
        Chip {
            text: "Unsaved".to_owned(),
            mark: Mark::Dot,
        }
    }

    /// When a fade begun at `now` is over.
    fn after_a_fade(now: Instant) -> Instant {
        now + Theme::GLASS.fade
    }

    /// The glass of the bar showing.
    fn glass_of(ui: &Ui<u8>) -> Rect {
        ui.bar.as_ref().expect("a bar").layout.glass
    }

    /// Where the bar's controls lie.
    fn control_rects(ui: &Ui<u8>) -> Vec<Rect> {
        let shown = ui.bar.as_ref().expect("a bar");
        shown.layout.controls.iter().map(|c| c.rect).collect()
    }

    /// The centre of a rectangle.
    fn centre(rect: Rect) -> (f32, f32) {
        (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)
    }

    /// The rectangles of a layer.
    fn rects_of(list: &DrawList, layer: Layer) -> Vec<RoundedRect> {
        let matching = list.rects.iter().filter(|rect| rect.layer == layer);
        matching.copied().collect()
    }

    /// The rectangles filled with a colour.
    fn filled_with(list: &DrawList, fill: Colour) -> Vec<Rect> {
        let matching = list.rects.iter().filter(|rect| rect.fill == fill);
        matching.map(|rect| rect.rect).collect()
    }

    /// The glyphs of a colour.
    fn glyphs_in(list: &DrawList, colour: Colour) -> Vec<GlyphQuad> {
        let matching = list.glyphs.iter().filter(|glyph| glyph.colour == colour);
        matching.copied().collect()
    }

    /// Whether two numbers are the same to a thousandth.
    fn near(one: f32, two: f32) -> bool {
        (one - two).abs() < 1e-3
    }

    #[test]
    fn nothing_showing_makes_no_frame() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        assert!(ui.frame(now).is_none(), "nothing to draw");
        assert!(!ui.animating(now), "nothing moving");
    }

    #[test]
    fn a_bar_makes_a_frame_and_the_same_list_until_it_changes() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let first = ui.frame(now).expect("a bar shows");
        let second = ui.frame(after_a_fade(now)).expect("still");
        assert!(Arc::ptr_eq(&first.list, &second.list), "nothing changed");

        ui.set_bar(Some(bar_titled("Something else")), now);
        let third = ui.frame(now).expect("still");
        assert!(!Arc::ptr_eq(&first.list, &third.list), "the title changed");
    }

    #[test]
    fn setting_what_is_already_set_changes_nothing() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.set_chip(Some(unsaved()), now);
        let first = ui.frame(now).expect("a bar shows");
        let later = after_a_fade(now);
        ui.set_bar(Some(close_bar()), later);
        ui.set_chip(Some(unsaved()), later);
        let second = ui.frame(later).expect("still");
        assert!(Arc::ptr_eq(&first.list, &second.list), "no rebuild");
        assert_eq!(first.fades, second.fades, "and no fade");
    }

    #[test]
    fn a_bar_fades_in_and_a_swap_for_another_does_not_fade() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        assert!(ui.animating(now), "coming in");
        let frame = ui.frame(now).expect("a bar shows");
        assert!(near(frame.fades[0].at(now), 0.0), "from nothing");
        let later = after_a_fade(now);
        assert!(!ui.animating(later), "in");
        assert!(near(frame.fades[0].at(later), 1.0), "to full");

        ui.set_bar(Some(bar_titled("Something else")), later);
        assert!(!ui.animating(later), "a swap doesn't fade");
    }

    #[test]
    fn a_bar_hit_gives_its_buttons_action_and_a_miss_gives_none() {
        let now = Instant::now();
        let seen = after_a_fade(now);
        let mut ui = new_ui(now);
        assert_eq!(ui.hit(640.0, 750.0, seen), None, "no bar");
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        let with_actions = rects.into_iter().zip([1, 2, 3]);
        for (rect, action) in with_actions {
            let (x, y) = centre(rect);
            assert_eq!(ui.hit(x, y, seen), Some(action), "{rect:?}");
        }
        assert_eq!(ui.hit(640.0, 100.0, seen), None, "over the model");
        let glass = glass_of(&ui);
        assert_eq!(
            ui.hit(glass.x + 5.0, glass.y + 5.0, seen),
            None,
            "the bare glass"
        );
    }

    #[test]
    fn on_bar_is_the_whole_glass_and_nothing_else() {
        let now = Instant::now();
        let seen = after_a_fade(now);
        let mut ui = new_ui(now);
        assert!(!ui.on_bar(640.0, 750.0, seen), "no bar");
        ui.set_bar(Some(close_bar()), now);
        let glass = glass_of(&ui);
        assert!(ui.on_bar(glass.x + 5.0, glass.y + 5.0, seen), "bare glass");
        assert!(
            !ui.on_bar(glass.x - 1.0, glass.y + 5.0, seen),
            "just outside"
        );
    }

    #[test]
    fn a_bar_is_not_hit_while_the_pen_is_down() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let (x, y) = centre(control_rects(&ui)[2]);
        let seen = after_a_fade(now);
        assert_eq!(ui.hit(x, y, seen), Some(3), "there");
        ui.pen_down(seen);
        assert_eq!(ui.hit(x, y, seen), None, "the pen is sculpting");
        assert!(!ui.on_bar(x, y, seen), "the bar has gone");
        ui.pen_up(seen);
        assert_eq!(ui.hit(x, y, after_a_fade(seen)), Some(3), "back");
    }

    #[test]
    fn a_leaving_bar_is_not_hit() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let (x, y) = centre(control_rects(&ui)[2]);
        let seen = after_a_fade(now);
        ui.set_bar(None, seen);
        assert_eq!(ui.hit(x, y, seen), None, "gone as far as the pen goes");
        assert!(!ui.on_bar(x, y, seen), "and its glass");
    }

    /// A double click's second press mustn't land on what the first brought up.
    #[test]
    fn controls_swapped_under_the_pointer_cant_be_hit_at_once_but_stop_a_press() {
        let now = Instant::now();
        let lockout = Theme::GLASS.swap_lockout;
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let seen = after_a_fade(now);
        let other = Bar::new("Finishing", vec![quiet("Cancel out of all of this now", 7)]);
        ui.set_bar(Some(other), seen);
        let (x, y) = centre(control_rects(&ui)[0]);
        assert_eq!(ui.hit(x, y, seen), None, "the same instant");
        assert!(ui.on_bar(x, y, seen), "it is still the bar's");
        let almost = seen + lockout.mul_f32(0.9);
        assert_eq!(ui.hit(x, y, almost), None, "not quite yet");
        assert_eq!(ui.hit(x, y, seen + lockout), Some(7), "and then it can");
    }

    #[test]
    fn a_progress_step_keeps_the_controls_hittable() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        let with_progress = |progress| {
            Bar::new("Finishing your stroke", vec![quiet("Cancel", 1)]).with_progress(progress)
        };
        ui.set_bar(Some(with_progress(0.25)), now);
        let seen = after_a_fade(now);
        ui.set_bar(Some(with_progress(0.5)), seen);
        let (x, y) = centre(control_rects(&ui)[0]);
        assert_eq!(ui.hit(x, y, seen), Some(1), "the same control");
    }

    #[test]
    fn a_bar_that_arrived_under_the_pen_is_not_hit_until_it_can_be_seen() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.pen_down(now);
        ui.set_bar(Some(close_bar()), now);
        ui.pen_up(now);
        let (x, y) = centre(control_rects(&ui)[0]);
        assert_eq!(ui.hit(x, y, now), None, "at opacity 0");
        assert!(!ui.on_bar(x, y, now), "nor its glass");

        let faint = now + Theme::GLASS.fade / 4;
        assert_eq!(ui.hit(x, y, faint), None, "a quarter there");
        let half = now + Theme::GLASS.fade / 2;
        assert_eq!(ui.hit(x, y, half), Some(1), "half there");
        assert!(ui.on_bar(x, y, half), "and its glass");
    }

    #[test]
    fn a_bar_set_while_the_pen_is_down_fades_in_on_the_up() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_chip(Some(unsaved()), now);
        ui.pen_down(now);
        ui.set_bar(Some(close_bar()), now);
        let during = ui.frame(now).expect("the chip shows");
        assert!(near(during.fades[0].to, 0.0), "the bar waits");
        assert!(near(during.fades[0].at(now), 0.0), "out of sight");

        let up = after_a_fade(now);
        ui.pen_up(up);
        let after = ui.frame(up).expect("the bar and chip");
        assert!(
            near(after.fades[0].at(up), 0.0),
            "the fade starts at the up"
        );
        assert!(near(after.fades[0].to, 1.0), "and rises");
        assert_eq!(after.fades[0].start, up, "from the up");
    }

    #[test]
    fn no_bar_shows_between_down_and_up_whatever_arrives() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_chip(Some(unsaved()), now);
        ui.pen_down(now);
        for step in 0..5 {
            let then = now + Duration::from_millis(step * 40);
            ui.set_bar(Some(bar_titled(format!("Bar {step}"))), then);
            let frame = ui.frame(then).expect("the chip shows");
            assert!(near(frame.fades[0].at(then), 0.0), "step {step}");
        }
    }

    #[test]
    fn the_pen_going_down_fades_bars_out_and_the_chip_to_a_quarter_without_a_rebuild() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.set_chip(Some(unsaved()), now);
        let before = ui.frame(now).expect("both show");

        let down = after_a_fade(now);
        ui.pen_down(down);
        let during = ui.frame(down).expect("both are fading");
        assert!(Arc::ptr_eq(&before.list, &during.list), "no rebuild");
        assert!(near(during.fades[0].at(down), 1.0), "from full");
        assert!(near(during.fades[0].to, 0.0), "bars go");
        assert!(
            near(during.fades[1].to, 0.25),
            "the chip drops to a quarter"
        );
        assert!(ui.animating(down), "a frame is wanted");

        let settled = after_a_fade(down);
        assert!(!ui.animating(settled), "and then none");
        let resting = ui.frame(settled).expect("the chip still shows");
        assert!(Arc::ptr_eq(&before.list, &resting.list), "still no rebuild");

        ui.pen_up(settled);
        let up = ui.frame(settled).expect("both");
        assert!(Arc::ptr_eq(&before.list, &up.list), "the up neither");
        assert!(near(up.fades[0].to, 1.0), "the bar returns");
        assert!(near(up.fades[1].to, 1.0), "the chip returns");
    }

    #[test]
    fn a_bar_alone_makes_no_frames_once_the_pen_has_faded_it() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let down = after_a_fade(now);
        ui.pen_down(down);
        assert!(ui.frame(down).is_some(), "fading");
        assert!(
            ui.frame(after_a_fade(down)).is_none(),
            "no pass under a stroke"
        );
    }

    #[test]
    fn hover_fills_only_the_control_under_the_pointer() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        let hover = Theme::GLASS.hover;
        let hovering = |showing: &mut Ui<u8>| -> Vec<Rect> {
            filled_with(&showing.frame(now).expect("a bar shows").list, hover)
        };
        assert!(hovering(&mut ui).is_empty(), "nothing yet");

        let (x, y) = centre(rects[1]);
        assert!(ui.hover(Some((x, y))), "that changes what draws");
        assert_eq!(hovering(&mut ui), [rects[1]], "that control alone");
        assert!(!ui.hover(Some((x + 1.0, y))), "the same control");

        let (next_x, next_y) = centre(rects[0]);
        assert!(ui.hover(Some((next_x, next_y))), "the next control");
        assert_eq!(hovering(&mut ui), [rects[0]], "and no other");

        assert!(ui.hover(None), "off the bar");
        assert!(hovering(&mut ui).is_empty(), "nothing again");
    }

    #[test]
    fn hover_changes_nothing_on_the_filled_control_or_with_the_pen_down() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        assert!(!ui.hover(Some(centre(rects[2]))), "the filled control");

        assert!(ui.hover(Some(centre(rects[0]))), "a quiet one does");
        let before = ui.frame(now).expect("a bar shows");
        ui.pen_down(now);
        assert!(
            !ui.hover(Some(centre(rects[1]))),
            "nothing moves under a stroke"
        );
        assert!(!ui.hover(None), "not even leaving");
        let during = ui.frame(now).expect("fading");
        assert!(Arc::ptr_eq(&before.list, &during.list), "no rebuild");
    }

    #[test]
    fn the_up_catches_the_hover_up_with_the_pointer() {
        let now = Instant::now();
        let hover = Theme::GLASS.hover;
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        ui.hover(Some(centre(rects[0])));
        ui.pen_down(now);
        ui.hover(Some(centre(rects[1])));
        ui.pen_up(now);
        let frame = ui.frame(now).expect("a bar shows");
        assert_eq!(filled_with(&frame.list, hover), [rects[1]], "where it went");

        ui.pen_down(now);
        ui.pen_up(now);
        let same = ui.frame(now).expect("a bar shows");
        assert!(
            Arc::ptr_eq(&frame.list, &same.list),
            "no change, no rebuild"
        );
    }

    #[test]
    fn a_new_bar_finds_the_pointer_where_it_is() {
        let now = Instant::now();
        let hover = Theme::GLASS.hover;
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        ui.hover(Some(centre(rects[1])));

        let wider = Bar::new("Another", vec![quiet("Cancel out of all of this now", 1)]);
        ui.set_bar(Some(wider), now);
        let over = control_rects(&ui)[0];
        assert!(
            over.contains(centre(rects[1]).0, centre(rects[1]).1),
            "under it"
        );
        let frame = ui.frame(now).expect("a bar shows");
        assert_eq!(filled_with(&frame.list, hover), [over], "hover moved over");

        ui.set_bar(Some(Bar::new("Nothing there", Vec::new())), now);
        let bare = ui.frame(now).expect("a bar shows");
        assert!(
            filled_with(&bare.list, hover).is_empty(),
            "nothing to hover"
        );
    }

    #[test]
    fn a_new_view_finds_the_pointer_where_it_is() {
        let now = Instant::now();
        let hover = Theme::GLASS.hover;
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.hover(Some(centre(control_rects(&ui)[1])));
        ui.set_view((1280, 400), 1.0);
        let frame = ui.frame(now).expect("a bar shows");
        assert!(
            filled_with(&frame.list, hover).is_empty(),
            "the bar moved away"
        );
    }

    #[test]
    fn a_swap_with_no_fade_still_wants_a_frame() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        assert!(!ui.needs_frame(now), "nothing showing");
        ui.set_bar(Some(close_bar()), now);
        let settled = after_a_fade(now);
        ui.frame(settled);
        assert!(!ui.needs_frame(settled), "nothing changed");

        ui.set_bar(Some(bar_titled("Something else")), settled);
        assert!(!ui.animating(settled), "no fade");
        assert!(ui.needs_frame(settled), "but the list changed");
        ui.frame(settled);
        assert!(!ui.needs_frame(settled), "and that frame shows it");
    }

    #[test]
    fn a_bar_waiting_for_the_pen_wants_no_frames() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.pen_down(now);
        ui.set_bar(Some(close_bar()), now);
        assert!(ui.frame(now).is_none(), "out of sight");
        assert!(!ui.needs_frame(now), "so none is wanted under the stroke");
        ui.pen_up(now);
        assert!(ui.needs_frame(now), "until the up");
    }

    #[test]
    fn linear_blending_raises_the_glass_it_draws() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui: Ui<u8> = Ui::new(theme, Blend::Linear, WINDOW, 1.0, now);
        ui.set_chip(Some(unsaved()), now);
        let frame = ui.frame(now).expect("a chip shows");
        let raised = theme.for_blend(Blend::Linear).glass;
        assert_eq!(frame.list.rects[0].fill, raised, "the theme's alpha for it");
    }

    #[test]
    fn glyph_quads_are_whole_pixels_at_any_scale() {
        let now = Instant::now();
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let mut ui = Ui::new(Theme::GLASS, Blend::Display, (1920, 1080), scale, now);
            let bar = close_bar().with_detail("Your work is safe and still open.");
            ui.set_bar(Some(bar.needing_you()), now);
            ui.set_chip(Some(unsaved()), now);
            let frame = ui.frame(now).expect("both show");
            assert!(frame.list.glyphs.len() > 40, "text at {scale}");
            for glyph in &frame.list.glyphs {
                let Rect { x, y, w, h } = glyph.rect;
                let whole = [x, y, w, h].iter().all(|side| side.fract() == 0.0);
                assert!(whole, "{glyph:?} at {scale}");
            }
        }
    }

    #[test]
    fn a_bars_controls_are_44_logical_pixels_tall_at_any_scale() {
        let now = Instant::now();
        for scale in [1.0, 1.25, 1.1, 1.5, 2.0, 3.0] {
            let mut ui = Ui::new(Theme::GLASS, Blend::Display, (3840, 2160), scale, now);
            ui.set_bar(Some(close_bar()), now);
            let rects = control_rects(&ui);
            for rect in &rects {
                assert!(rect.h >= 44.0 * scale, "{rect:?} at {scale}");
                assert!(rect.h < 44.0 * scale + 1.0, "{rect:?} at {scale}");
            }
            let gap = rects[1].x - (rects[0].x + rects[0].w);
            assert!((gap - 8.0 * scale).abs() <= 1.0, "{gap} at {scale}");
        }
    }

    #[test]
    fn the_bar_sits_centred_28_up_and_the_chip_20_in() {
        let now = Instant::now();
        for scale in [1.0, 2.0] {
            let mut ui = Ui::new(Theme::GLASS, Blend::Display, WINDOW, scale, now);
            ui.set_bar(Some(close_bar()), now);
            ui.set_chip(Some(unsaved()), now);
            let frame = ui.frame(now).expect("both show");

            let glass = rects_of(&frame.list, Layer::Bar)[0].rect;
            let right = 1280.0 - (glass.x + glass.w);
            assert!((glass.x - right).abs() <= 1.0, "{glass:?} at {scale}");
            let bottom = 800.0 - (glass.y + glass.h);
            assert!(near(bottom, 28.0 * scale), "{glass:?} at {scale}");

            let chip = rects_of(&frame.list, Layer::Chip)[0];
            assert!(near(chip.rect.y, 20.0 * scale), "{chip:?} at {scale}");
            let inset = 1280.0 - (chip.rect.x + chip.rect.w);
            assert!(near(inset, 20.0 * scale), "{chip:?} at {scale}");
            assert!(near(chip.rect.h, 32.0 * scale), "{chip:?} at {scale}");
            assert!(near(chip.radius, 16.0 * scale), "fully round at {scale}");
        }
    }

    #[test]
    fn a_bar_is_as_wide_as_its_content_within_limits() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(Bar::new("Short", vec![quiet("Cancel", 1)])), now);
        assert!(near(glass_of(&ui).w, 480.0), "at least 480");

        ui.set_bar(Some(close_bar()), now);
        assert!(glass_of(&ui).w > 480.0, "this one needs more");

        let long = Bar::new("Long ".repeat(200), vec![quiet("Cancel", 1)]);
        ui.set_bar(Some(long), now);
        assert!(
            near(glass_of(&ui).w, 1240.0),
            "at most the window less 2 x 20"
        );

        ui.set_view((400, 800), 1.0);
        assert!(
            near(glass_of(&ui).w, 360.0),
            "a small window wins over the least"
        );
    }

    #[test]
    fn a_title_too_long_for_the_bar_stops_short_of_its_controls() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        let long = Bar::new("Long ".repeat(200), vec![quiet("Cancel", 1)]);
        ui.set_bar(Some(long), now);
        let cancel = control_rects(&ui)[0];
        let shown = ui.bar.as_ref().expect("a bar");
        let title = &shown.layout.lines[0];
        let end = title.x + title.line.width;
        let room = cancel.x - Theme::GLASS.gap;
        assert!(end <= room, "{end} against {room}");
        assert!(end > room - 20.0, "the room is used: {end} of {room}");
    }

    #[test]
    fn progress_changes_the_list_but_keeps_the_layout() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        let with_progress = |progress| {
            Bar::new("Finishing your stroke", vec![quiet("Cancel", 1)]).with_progress(progress)
        };
        ui.set_bar(Some(with_progress(0.25)), now);
        let before = ui.frame(now).expect("a bar shows");
        let (glass, controls) = (glass_of(&ui), control_rects(&ui));
        let track = filled_with(&before.list, theme.progress_track)[0];
        let line = filled_with(&before.list, theme.accent)[0];
        assert!(near(line.w, track.w * 0.25), "a quarter of it");

        ui.set_bar(Some(with_progress(0.75)), now);
        let after = ui.frame(now).expect("still");
        assert!(!Arc::ptr_eq(&before.list, &after.list), "the line moved");
        let moved = filled_with(&after.list, theme.accent)[0];
        assert!(near(moved.w, track.w * 0.75), "three quarters");
        assert_eq!(glass_of(&ui), glass, "the glass stays");
        assert_eq!(control_rects(&ui), controls, "the controls stay");
    }

    #[test]
    fn a_faded_out_bar_stops_making_frames_once_settled() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let settled = after_a_fade(now);
        assert!(ui.frame(settled).is_some(), "showing");
        ui.set_bar(None, settled);
        assert!(ui.animating(settled), "leaving");
        let frame = ui.frame(settled).expect("fading out");
        assert!(!frame.list.rects.is_empty(), "it draws until it's gone");

        let gone = after_a_fade(settled);
        assert!(!ui.animating(gone), "gone");
        assert!(ui.frame(gone).is_none(), "and no more frames");
    }

    #[test]
    fn a_left_bars_shapes_leave_the_list_while_the_chip_stays() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.set_chip(Some(unsaved()), now);
        let settled = after_a_fade(now);
        ui.set_bar(None, settled);
        let frame = ui.frame(after_a_fade(settled)).expect("the chip shows");
        assert!(rects_of(&frame.list, Layer::Bar).is_empty(), "the bar went");
        assert!(
            !rects_of(&frame.list, Layer::Chip).is_empty(),
            "the chip didn't"
        );
    }

    #[test]
    fn a_bar_set_again_as_it_leaves_comes_back_from_where_it_had_got_to() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let settled = after_a_fade(now);
        ui.set_bar(None, settled);
        let midway = settled + Theme::GLASS.fade / 2;
        ui.set_bar(Some(close_bar()), midway);
        let frame = ui.frame(midway).expect("back");
        assert!(near(frame.fades[0].at(midway), 0.5), "no jump");
        assert!(near(frame.fades[0].to, 1.0), "rising");
        let (x, y) = centre(control_rects(&ui)[2]);
        assert_eq!(ui.hit(x, y, after_a_fade(midway)), Some(3), "and live");
    }

    #[test]
    fn the_chip_draws_its_mark_as_a_dot_a_ring_or_nothing() {
        let now = Instant::now();
        let accent = Theme::GLASS.accent;
        let chip = |mark| Chip {
            text: "Saving".to_owned(),
            mark,
        };

        let mut ui = new_ui(now);
        ui.set_chip(Some(chip(Mark::Dot)), now);
        let with_dot = ui.frame(now).expect("a chip shows");
        let rects = rects_of(&with_dot.list, Layer::Chip);
        assert_eq!(rects.len(), 2, "glass and dot");
        assert_eq!(rects[1].fill, accent, "a dot is filled");
        assert!(near(rects[1].rect.w, 8.0), "8 px");

        ui.set_chip(Some(chip(Mark::Ring)), now);
        let with_ring = ui.frame(now).expect("a chip shows");
        let ring = rects_of(&with_ring.list, Layer::Chip)[1];
        assert_eq!(ring.fill.a, 0, "a ring is open");
        assert_eq!(ring.stroke, accent, "and stroked");
        assert!(near(ring.stroke_width, 1.5), "1.5 px");

        ui.set_chip(Some(chip(Mark::None)), now);
        let bare = ui.frame(now).expect("a chip shows");
        assert_eq!(rects_of(&bare.list, Layer::Chip).len(), 1, "glass alone");
    }

    /// Every chip the app shows, as it draws: its glass over the brightest clay, its text on that.
    #[test]
    fn a_chip_as_drawn_meets_contrast_over_the_brightest_clay() {
        let now = Instant::now();
        let clay = Colour::rgb(0xD6, 0xD6, 0xD2);
        let chips = [
            ("Unsaved", Mark::Dot),
            ("Saving", Mark::Ring),
            ("Saved", Mark::None),
        ];
        for (text, mark) in chips {
            let mut ui = new_ui(now);
            let chip = Chip {
                text: text.to_owned(),
                mark,
            };
            ui.set_chip(Some(chip), now);
            let frame = ui.frame(now).expect("a chip shows");
            let ground = frame.list.rects[0].fill.over(clay);
            let glyph = frame.list.glyphs[0].colour;
            let drawn = glyph.over(ground);
            assert!(
                drawn.contrast(ground) >= 4.5,
                "{text}: {drawn:?} on {ground:?}"
            );
        }
    }

    #[test]
    fn a_bar_that_needs_you_has_a_coral_hairline_and_mark() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let plain = ui.frame(now).expect("a bar shows");
        assert_eq!(
            plain.list.rects[0].stroke, theme.hairline,
            "a plain hairline"
        );
        let title = |shown: &Ui<u8>| shown.bar.as_ref().expect("a bar").layout.lines[0].x;
        let inset = |shown: &Ui<u8>| title(shown) - glass_of(shown).x;
        let plain_inset = inset(&ui);
        let plain_title = title(&ui);
        let in_the_list = |list: &DrawList, x: f32| {
            let lefts = list.glyphs.iter().map(|glyph| glyph.rect.x);
            lefts.fold(f32::MAX, f32::min) >= x
        };
        assert!(in_the_list(&plain.list, plain_title), "nothing left of it");

        ui.set_bar(Some(close_bar().needing_you()), now);
        let frame = ui.frame(now).expect("a bar shows");
        assert_eq!(
            frame.list.rects[0].stroke, theme.needs_you_hairline,
            "coral"
        );
        assert_eq!(
            filled_with(&frame.list, theme.needs_you).len(),
            1,
            "its mark"
        );
        let moved = inset(&ui) - plain_inset;
        assert!(near(moved, 22.0), "the mark and its gap: {moved}");
        assert!(in_the_list(&frame.list, title(&ui)), "the glyphs moved too");
    }

    #[test]
    fn a_detail_goes_under_the_title_in_quiet_text() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        let bar = Bar::new("Your unsaved work is back", vec![quiet("Use last save", 1)])
            .with_detail("From yesterday at 16:42");
        ui.set_bar(Some(bar), now);
        let frame = ui.frame(now).expect("a bar shows");
        let top = |glyphs: Vec<GlyphQuad>| {
            let tops = glyphs.into_iter().map(|glyph| glyph.rect.y);
            tops.fold(f32::MAX, f32::min)
        };
        let title_top = top(glyphs_in(&frame.list, theme.text));
        let detail_top = top(glyphs_in(&frame.list, theme.quiet_text));
        assert!(detail_top > title_top, "{detail_top} under {title_top}");
    }

    #[test]
    fn a_second_detail_line_goes_under_the_first_and_the_bar_grows_for_it() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let bar = || Bar::new("Your unsaved work is back", vec![quiet("Use last save", 1)]);
        let mut one = new_ui(now);
        one.set_bar(Some(bar().with_detail("From yesterday at 16:42")), now);
        let mut two = new_ui(now);
        let both = bar()
            .with_detail("From yesterday at 16:42")
            .with_detail("Your last stroke couldn't be finished.");
        two.set_bar(Some(both), now);
        let frame = two.frame(now).expect("a bar shows");
        let tops: Vec<f32> = glyphs_in(&frame.list, theme.quiet_text)
            .into_iter()
            .map(|glyph| glyph.rect.y)
            .collect();
        let highest = tops.iter().copied().fold(f32::MAX, f32::min);
        let lowest = tops.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            lowest - highest >= theme.status.1 * 0.5,
            "the second line a status line under the first: {highest} and {lowest}"
        );
        let (short, tall) = (glass_of(&one), glass_of(&two));
        assert!(
            near(tall.h - short.h, theme.status.1),
            "a status line taller: {} and {}",
            short.h,
            tall.h
        );
        assert!(near(tall.y + tall.h, short.y + short.h), "the same bottom");
    }

    #[test]
    fn only_the_first_filled_control_stays_filled() {
        let bar = Bar::new("Two", vec![filled("One", 1), filled("Two", 2)]);
        assert_eq!(bar.controls()[0].emphasis, Emphasis::Filled, "the first");
        assert_eq!(bar.controls()[1].emphasis, Emphasis::Quiet, "the second");
    }

    #[test]
    fn a_new_view_lays_everything_out_again_and_warms_its_scale() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.set_chip(Some(unsaved()), now);
        let before = ui.frame(now).expect("both show");

        ui.set_view((2560, 1600), 2.0);
        let version = ui.text.atlas().version;
        let after = ui.frame(now).expect("both show");
        assert!(!Arc::ptr_eq(&before.list, &after.list), "a new list");
        let glass = rects_of(&after.list, Layer::Bar)[0].rect;
        assert!(near(1600.0 - (glass.y + glass.h), 56.0), "{glass:?}");
        let chip = rects_of(&after.list, Layer::Chip)[0].rect;
        assert!(near(chip.y, 40.0) && near(chip.h, 64.0), "{chip:?}");
        assert!(near(2560.0 - (chip.x + chip.w), 40.0), "{chip:?}");

        ui.set_bar(Some(Bar::new("Fresh words", vec![quiet("Fine", 1)])), now);
        assert_eq!(ui.text.atlas().version, version, "the atlas was ready");
    }

    #[test]
    fn scales_passed_through_leave_nothing_in_the_atlas() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        ui.set_chip(Some(unsaved()), now);
        ui.set_view(WINDOW, 3.0);
        let held = ui.text.held_glyphs();
        let old = ui.frame(now).expect("both show");

        for step in 1..200_u16 {
            ui.set_view(WINDOW, 1.0 + f32::from(step) / 100.0);
        }
        ui.set_view(WINDOW, 3.0);
        assert_eq!(ui.text.held_glyphs(), held, "this scale's glyphs, no more");
        let new = ui.frame(now).expect("both show");
        assert!(
            new.atlas.version > old.atlas.version,
            "a renderer uploads it"
        );
        let side = new.atlas.size as usize;
        for glyph in &new.list.glyphs {
            let (x, y) = (glyph.atlas.0 as usize, glyph.atlas.1 as usize);
            let top_left = y * side + x;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a glyph's size is a small whole number"
            )]
            let (w, h) = (glyph.rect.w as usize, glyph.rect.h as usize);
            let rows = new.atlas.texels[top_left..].chunks(side).take(h);
            let mut inked = rows.flat_map(|row| &row[..w]);
            assert!(
                inked.any(|&texel| texel > 0),
                "{glyph:?} is in the new atlas"
            );
        }
    }

    #[test]
    fn a_move_with_no_new_scale_keeps_the_atlas() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        let version = ui.text.atlas().version;
        ui.set_view((640, 480), 1.0);
        assert_eq!(ui.text.atlas().version, version, "the same glyphs serve");
    }

    #[test]
    fn the_same_view_again_changes_nothing() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let before = ui.frame(now).expect("a bar shows");
        ui.set_view(WINDOW, 1.0);
        let after = ui.frame(now).expect("a bar shows");
        assert!(Arc::ptr_eq(&before.list, &after.list), "no rebuild");
    }

    #[test]
    fn a_frame_carries_the_atlas_its_glyphs_point_into() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_chip(Some(unsaved()), now);
        let frame = ui.frame(now).expect("a chip shows");
        assert!(!frame.list.glyphs.is_empty(), "text");
        let side = frame.atlas.size;
        for glyph in &frame.list.glyphs {
            assert!(glyph.atlas.0 < side && glyph.atlas.1 < side, "{glyph:?}");
        }
        assert!(frame.atlas.texels.iter().any(|&texel| texel > 0), "ink");
    }
}
