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
pub use theme::{Colour, Theme};
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
    /// A UI showing nothing, in a window of `size` physical pixels at `scale` of them a logical
    /// one, with printable ASCII ready to draw.
    #[must_use]
    pub fn new(theme: Theme, size: (u32, u32), scale: f32, now: Instant) -> Self {
        let mut ui = Self {
            theme,
            text: Text::new(),
            view: view(size, scale),
            bar: None,
            chip: None,
            pen_is_down: false,
            pointer: None,
            hovered: None,
            bar_fade: Fade::at_rest(0.0, now),
            chip_fade: Fade::at_rest(0.0, now),
            list: Arc::default(),
            stale: false,
        };
        ui.warm();
        ui
    }

    /// Moves to a window of `size` at `scale`. A new scale rasterises printable ASCII for it.
    pub fn set_view(&mut self, size: (u32, u32), scale: f32) {
        let next = view(size, scale);
        if next == self.view {
            return;
        }
        let rescaled = (next.scale - self.view.scale).abs() > f32::EPSILON;
        self.view = next;
        if rescaled {
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

    /// The pen is up: bars return. Touches the fades only.
    pub fn pen_up(&mut self, now: Instant) {
        self.pen_is_down = false;
        self.aim_fades(now);
    }

    /// The action under the point; `None` with no bar, or the pen down.
    #[must_use]
    pub fn hit(&self, x: f32, y: f32) -> Option<A> {
        let shown = self.live_bar()?;
        let at = shown.layout.control_at(x, y)?;
        shown.widget.controls.get(at).map(|control| control.action)
    }

    /// Whether the point is on the bar's glass at all.
    #[must_use]
    pub fn on_bar(&self, x: f32, y: f32) -> bool {
        self.live_bar()
            .is_some_and(|shown| shown.layout.glass.contains(x, y))
    }

    /// Moves the hover; true if that changed what draws. With the pen down nothing changes.
    pub fn hover(&mut self, at: Option<(f32, f32)>) -> bool {
        self.pointer = at;
        if self.pen_is_down {
            return false;
        }
        let hovered = self.control_under_pointer();
        let changed = hovered != self.hovered;
        self.hovered = hovered;
        self.stale |= changed;
        changed
    }

    /// What to draw, rebuilding only if something changed; `None` with nothing to show.
    pub fn frame(&mut self, now: Instant) -> Option<UiFrame> {
        self.drop_what_left(now);
        let visible = |shown: bool, fade: Fade| shown && (fade.to > 0.0 || !fade.settled(now));
        let bar = visible(self.bar.is_some(), self.bar_fade);
        let chip = visible(self.chip.is_some(), self.chip_fade);
        if !bar && !chip {
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

    /// Whether a fade is moving at `now`: a frame is wanted.
    #[must_use]
    pub fn animating(&self, now: Instant) -> bool {
        !self.bar_fade.settled(now) || !self.chip_fade.settled(now)
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

    /// The bar, if the pen can reach it.
    fn live_bar(&self) -> Option<&Shown<Bar<A>, BarLayout>> {
        let shown = self.bar.as_ref()?;
        (!shown.leaving && !self.pen_is_down).then_some(shown)
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
        Ui::new(Theme::GLASS, WINDOW, 1.0, now)
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

    /// The close bar: actions 1, 2 and 3.
    fn close_bar() -> Bar<u8> {
        let controls = vec![
            quiet("Don't save", 1),
            quiet("Keep sculpting", 2),
            filled("Save", 3),
        ];
        Bar::new("Save changes to first-sculpt?", controls)
    }

    /// A chip with a dot.
    fn unsaved() -> Chip {
        Chip {
            text: "Unsaved".to_owned(),
            mark: Mark::Dot,
            dimmed: false,
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

        let mut other = close_bar();
        other.title = "Something else".to_owned();
        ui.set_bar(Some(other), now);
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

        let mut other = close_bar();
        other.title = "Something else".to_owned();
        ui.set_bar(Some(other), later);
        assert!(!ui.animating(later), "a swap doesn't fade");
    }

    #[test]
    fn a_bar_hit_gives_its_buttons_action_and_a_miss_gives_none() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        assert_eq!(ui.hit(640.0, 750.0), None, "no bar");
        ui.set_bar(Some(close_bar()), now);
        let rects = control_rects(&ui);
        let with_actions = rects.into_iter().zip([1, 2, 3]);
        for (rect, action) in with_actions {
            let (x, y) = centre(rect);
            assert_eq!(ui.hit(x, y), Some(action), "{rect:?}");
        }
        assert_eq!(ui.hit(640.0, 100.0), None, "over the model");
        let glass = glass_of(&ui);
        assert_eq!(ui.hit(glass.x + 5.0, glass.y + 5.0), None, "the bare glass");
    }

    #[test]
    fn on_bar_is_the_whole_glass_and_nothing_else() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        assert!(!ui.on_bar(640.0, 750.0), "no bar");
        ui.set_bar(Some(close_bar()), now);
        let glass = glass_of(&ui);
        assert!(ui.on_bar(glass.x + 5.0, glass.y + 5.0), "bare glass");
        assert!(!ui.on_bar(glass.x - 1.0, glass.y + 5.0), "just outside");
    }

    #[test]
    fn a_bar_is_not_hit_while_the_pen_is_down() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let (x, y) = centre(control_rects(&ui)[2]);
        ui.pen_down(now);
        assert_eq!(ui.hit(x, y), None, "the pen is sculpting");
        assert!(!ui.on_bar(x, y), "the bar has gone");
        ui.pen_up(now);
        assert_eq!(ui.hit(x, y), Some(3), "back");
    }

    #[test]
    fn a_leaving_bar_is_not_hit() {
        let now = Instant::now();
        let mut ui = new_ui(now);
        ui.set_bar(Some(close_bar()), now);
        let (x, y) = centre(control_rects(&ui)[2]);
        ui.set_bar(None, now);
        assert_eq!(ui.hit(x, y), None, "gone as far as the pen goes");
        assert!(!ui.on_bar(x, y), "and its glass");
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
            let mut bar = close_bar();
            bar.title = format!("Bar {step}");
            ui.set_bar(Some(bar), then);
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
        let (x, y) = centre(rects[2]);
        assert!(!ui.hover(Some((x, y))), "the filled control doesn't change");

        ui.pen_down(now);
        let (quiet_x, quiet_y) = centre(rects[0]);
        assert!(
            !ui.hover(Some((quiet_x, quiet_y))),
            "nothing moves under a stroke"
        );
    }

    #[test]
    fn a_bars_controls_are_44_logical_pixels_tall_at_any_scale() {
        let now = Instant::now();
        for scale in [1.0, 1.25, 1.1, 1.5, 2.0, 3.0] {
            let mut ui = Ui::new(Theme::GLASS, (3840, 2160), scale, now);
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
            let mut ui = Ui::new(Theme::GLASS, WINDOW, scale, now);
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
        let frame = ui.frame(now).expect("a bar shows");
        let title = glyphs_in(&frame.list, Theme::GLASS.text);
        let left_of_cancel = title.iter().filter(|glyph| glyph.rect.x < cancel.x);
        let ends = left_of_cancel.map(|glyph| glyph.rect.x + glyph.rect.w);
        let end = ends.fold(0.0, f32::max);
        assert!(end > 900.0, "the room is used: {end}");
        assert!(end <= cancel.x, "{end} against {cancel:?}");
    }

    #[test]
    fn progress_changes_the_list_but_keeps_the_layout() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        let with_progress = |progress| Bar {
            progress: Some(progress),
            ..Bar::new("Finishing your stroke", vec![quiet("Cancel", 1)])
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
        assert_eq!(ui.hit(x, y), Some(3), "and live");
    }

    #[test]
    fn the_chip_draws_its_mark_as_a_dot_a_ring_or_nothing() {
        let now = Instant::now();
        let accent = Theme::GLASS.accent;
        let chip = |mark| Chip {
            text: "Saving".to_owned(),
            mark,
            dimmed: false,
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

    #[test]
    fn a_dimmed_chip_is_all_at_saved_alpha() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        let saved = Chip {
            text: "Saved".to_owned(),
            mark: Mark::None,
            dimmed: true,
        };
        ui.set_chip(Some(saved), now);
        let frame = ui.frame(now).expect("a chip shows");
        let glass = frame.list.rects[0];
        assert_eq!(
            glass.fill,
            theme.glass.faded(theme.saved_alpha),
            "the glass"
        );
        assert_eq!(
            glass.stroke,
            theme.hairline.faded(theme.saved_alpha),
            "its line"
        );
        let text = glyphs_in(&frame.list, theme.text.faded(theme.saved_alpha));
        assert_eq!(text.len(), frame.list.glyphs.len(), "its text");
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
        let title_x = plain.list.glyphs[0].rect.x;

        let failure = Bar {
            needs_you: true,
            ..close_bar()
        };
        ui.set_bar(Some(failure), now);
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
        assert!(frame.list.glyphs[0].rect.x > title_x, "the text moved over");
    }

    #[test]
    fn a_detail_goes_under_the_title_in_quiet_text() {
        let now = Instant::now();
        let theme = Theme::GLASS;
        let mut ui = new_ui(now);
        let bar = Bar {
            detail: Some("From yesterday at 16:42".to_owned()),
            ..Bar::new("Your unsaved work is back", vec![quiet("Use last save", 1)])
        };
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
    fn only_the_first_filled_control_stays_filled() {
        let bar = Bar::new("Two", vec![filled("One", 1), filled("Two", 2)]);
        assert_eq!(bar.controls[0].emphasis, Emphasis::Filled, "the first");
        assert_eq!(bar.controls[1].emphasis, Emphasis::Quiet, "the second");
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

        ui.set_bar(Some(Bar::new("Fresh words", vec![quiet("Fine", 1)])), now);
        assert_eq!(ui.text.atlas().version, version, "the atlas was ready");
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
