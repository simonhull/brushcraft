//! Bars and chips, their layout and hit testing.
//!
//! A widget says what to show; its layout says where, in physical pixels, for one window and
//! scale. Layout is where text is measured and any new glyph rasterised, so it happens when a
//! widget or the window changes, never on the pen's path.

use crate::draw::{DrawList, GlyphQuad, Layer, Rect, RoundedRect};
use crate::text::{Line, Text, Weight};
use crate::theme::{Colour, Theme};

/// How a control draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emphasis {
    /// Text, with a fill on hover.
    Quiet,
    /// The bar's one action: filled with the accent.
    Filled,
}

/// A control on a bar. At most one is `Filled`: [`Bar::new`] keeps the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Control<A> {
    /// What it says.
    pub label: String,
    /// What pressing it means, handed back by `Ui::hit`.
    pub action: A,
    /// How it draws.
    pub emphasis: Emphasis,
}

/// A prompt along the bottom edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Bar<A> {
    /// What it says.
    pub title: String,
    /// More, in quieter text under the title.
    pub detail: Option<String>,
    /// 0 to 1: draws the progress line.
    pub progress: Option<f32>,
    /// A coral hairline: this needs the user.
    pub needs_you: bool,
    /// Its controls, left to right.
    pub controls: Vec<Control<A>>,
}

impl<A> Bar<A> {
    /// A bar with a title and controls, and nothing more. Controls `Filled` after the first become
    /// `Quiet`, since a bar has one action at most.
    #[must_use]
    pub fn new(title: impl Into<String>, mut controls: Vec<Control<A>>) -> Self {
        let filled = controls.iter_mut();
        let mut filled = filled.filter(|control| control.emphasis == Emphasis::Filled);
        filled.next();
        filled.for_each(|control| control.emphasis = Emphasis::Quiet);
        Self {
            title: title.into(),
            detail: None,
            progress: None,
            needs_you: false,
            controls,
        }
    }
}

/// What the chip's mark is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// A filled dot.
    Dot,
    /// An open ring.
    Ring,
    /// No mark.
    None,
}

/// The status in the top right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chip {
    /// What it says.
    pub text: String,
    /// The mark before the text.
    pub mark: Mark,
    /// All of it at the theme's `saved_alpha`: nothing left to do.
    pub dimmed: bool,
}

/// The window being laid out for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct View {
    /// Its width, physical pixels.
    pub(crate) width: f32,
    /// Its height, physical pixels.
    pub(crate) height: f32,
    /// Physical pixels a logical one.
    pub(crate) scale: f32,
}

impl View {
    /// A logical distance in physical pixels, to the nearest whole one so edges stay crisp.
    fn whole(self, logical: f32) -> f32 {
        (logical * self.scale).round()
    }
}

/// A line of text placed in the window.
#[derive(Debug, Clone, PartialEq)]
struct PlacedLine {
    /// The line.
    line: Line,
    /// Its origin's x, a whole pixel.
    x: f32,
    /// Its baseline's y, a whole pixel.
    baseline: f32,
    /// Its colour.
    colour: Colour,
}

impl PlacedLine {
    /// Adds its glyphs to `list`.
    fn draw(&self, layer: Layer, list: &mut DrawList) {
        let glyphs = self.line.glyphs.iter().map(|glyph| GlyphQuad {
            rect: Rect {
                x: self.x + glyph.at.0,
                y: self.baseline + glyph.at.1,
                w: glyph.size.0,
                h: glyph.size.1,
            },
            atlas: glyph.atlas,
            colour: self.colour,
            layer,
        });
        list.glyphs.extend(glyphs);
    }
}

/// `string` as a line no wider than `room`: whole if it fits, otherwise as much as fits with an
/// ellipsis after it.
fn fitted(text: &mut Text, string: &str, weight: Weight, px: f32, room: f32) -> Line {
    let whole = text.line(string, weight, px);
    if whole.width <= room {
        return whole;
    }
    let shortened = |end: usize| {
        let kept = string.get(..end).unwrap_or_default();
        format!("{}\u{2026}", kept.trim_end())
    };
    let ends: Vec<usize> = string.char_indices().map(|(end, _)| end).collect();
    let fitting = ends.partition_point(|&end| text.line(&shortened(end), weight, px).width <= room);
    let last_fitting = fitting.checked_sub(1).and_then(|index| ends.get(index));
    text.line(&shortened(last_fitting.copied().unwrap_or(0)), weight, px)
}

/// A control, placed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ControlLayout {
    /// Its target: at least the theme's control height, whatever the scale.
    pub(crate) rect: Rect,
    /// Its label.
    label: PlacedLine,
    /// How it draws.
    emphasis: Emphasis,
}

/// A progress line, placed.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ProgressLayout {
    /// The groove.
    track: Rect,
    /// How much of it is done, 0 to 1.
    done: f32,
}

/// A bar's measurements at one scale, physical pixels.
#[derive(Debug, Clone, Copy)]
struct BarSizes {
    /// The padding around the controls.
    inset: f32,
    /// Between neighbours.
    gap: f32,
    /// The text in from the left edge.
    text_inset: f32,
    /// The size of body text.
    body_px: f32,
    /// The line height of body text.
    body_height: f32,
    /// The size of status text.
    status_px: f32,
    /// The line height of status text.
    status_height: f32,
    /// The height of the row of controls: never under the theme's control height.
    row_height: f32,
    /// The side of the needs-you mark.
    mark_size: f32,
    /// What the needs-you mark takes from the text, if the bar has one.
    mark_room: f32,
    /// The thickness of the progress line.
    progress_height: f32,
}

impl BarSizes {
    /// The measurements of a bar in `view`.
    fn new(needs_you: bool, view: View, theme: &Theme) -> Self {
        let mark_size = view.whole(theme.mark_size);
        let mark_room = if needs_you {
            mark_size + view.whole(theme.needs_you_gap)
        } else {
            0.0
        };
        Self {
            inset: view.whole(theme.bar_inset),
            gap: view.whole(theme.gap),
            text_inset: view.whole(theme.bar_text_inset),
            body_px: theme.body.0 * view.scale,
            body_height: view.whole(theme.body.1),
            status_px: theme.status.0 * view.scale,
            status_height: view.whole(theme.status.1),
            row_height: (theme.control_height * view.scale).ceil(),
            mark_size,
            mark_room,
            progress_height: view.whole(theme.progress_height),
        }
    }

    /// The bar's surface: as wide as its content within the theme's limits, centred, and up from
    /// the window's bottom edge.
    fn glass<A>(
        &self,
        bar: &Bar<A>,
        labels: &[Label],
        view: View,
        theme: &Theme,
        text: &mut Text,
    ) -> Rect {
        let title = text
            .line(&bar.title, theme.title_weight, self.body_px)
            .width;
        let detail = bar.detail.as_deref().map_or(0.0, |detail| {
            text.line(detail, theme.status_weight, self.status_px).width
        });
        let controls: f32 = labels.iter().map(|label| label.width + self.gap).sum();
        let content =
            self.text_inset + self.mark_room + title.max(detail).ceil() + controls + self.inset;
        let widest = view.width - 2.0 * view.whole(theme.bar_margin);
        let least = view.whole(theme.bar_min_width).min(widest);
        let width = content.max(least).min(widest);
        let extra = bar
            .progress
            .map_or(0.0, |_| self.gap + self.progress_height);
        let height = 2.0 * self.inset + self.row_height + extra;
        Rect {
            x: ((view.width - width) * 0.5).round(),
            y: view.height - view.whole(theme.bar_bottom) - height,
            w: width,
            h: height,
        }
    }

    /// The title and any detail, centred together on the row at `at`, its left and top, and
    /// shortened to `room`.
    fn text_lines<A>(
        &self,
        bar: &Bar<A>,
        at: (f32, f32),
        room: f32,
        theme: &Theme,
        text: &mut Text,
    ) -> Vec<PlacedLine> {
        let (x, row_top) = at;
        let detail = bar.detail.as_deref();
        let block = self.body_height + detail.map_or(0.0, |_| self.status_height);
        let top = row_top + ((self.row_height - block) * 0.5).round();
        let (title_weight, status_weight) = (theme.title_weight, theme.status_weight);
        let title = PlacedLine {
            line: fitted(text, &bar.title, title_weight, self.body_px, room),
            x,
            baseline: top + text.baseline(title_weight, self.body_px, self.body_height),
            colour: theme.text,
        };
        let detail = detail.map(|detail| PlacedLine {
            line: fitted(text, detail, status_weight, self.status_px, room),
            x,
            baseline: top
                + self.body_height
                + text.baseline(status_weight, self.status_px, self.status_height),
            colour: theme.quiet_text,
        });
        std::iter::once(title).chain(detail).collect()
    }
}

/// A control's label, measured but not yet placed.
#[derive(Debug, Clone, PartialEq)]
struct Label {
    /// The text.
    line: Line,
    /// Its weight.
    weight: Weight,
    /// Its colour.
    colour: Colour,
    /// The whole control's width.
    width: f32,
    /// How the control draws.
    emphasis: Emphasis,
}

impl Label {
    /// Measures `control`'s label.
    fn new<A>(
        control: &Control<A>,
        sizes: &BarSizes,
        view: View,
        theme: &Theme,
        text: &mut Text,
    ) -> Self {
        let (weight, padding, colour) = match control.emphasis {
            Emphasis::Quiet => (theme.control_weight, theme.control_padding, theme.text),
            Emphasis::Filled => (theme.filled_weight, theme.filled_padding, theme.accent_text),
        };
        let line = text.line(&control.label, weight, sizes.body_px);
        let width = line.width.ceil() + 2.0 * view.whole(padding);
        Self {
            line,
            weight,
            colour,
            width,
            emphasis: control.emphasis,
        }
    }

    /// The control, its right edge at `right` and its top at `top`.
    fn place(self, right: f32, top: f32, sizes: &BarSizes, text: &Text) -> ControlLayout {
        let rect = Rect {
            x: right - self.width,
            y: top,
            w: self.width,
            h: sizes.row_height,
        };
        let centred = ((sizes.row_height - sizes.body_height) * 0.5).round();
        let label = PlacedLine {
            x: (rect.x + (rect.w - self.line.width) * 0.5).round(),
            baseline: top + centred + text.baseline(self.weight, sizes.body_px, sizes.body_height),
            line: self.line,
            colour: self.colour,
        };
        ControlLayout {
            rect,
            label,
            emphasis: self.emphasis,
        }
    }
}

/// A bar, placed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BarLayout {
    /// Its surface.
    pub(crate) glass: Rect,
    /// Its controls, left to right.
    pub(crate) controls: Vec<ControlLayout>,
    /// Its title, and its detail if it has one.
    lines: Vec<PlacedLine>,
    /// Its needs-you mark, if it has one.
    mark: Option<Rect>,
    /// Its progress line, if it has one.
    progress: Option<ProgressLayout>,
}

impl BarLayout {
    /// Places `bar` in `view`.
    pub(crate) fn new<A>(bar: &Bar<A>, view: View, theme: &Theme, text: &mut Text) -> Self {
        let sizes = BarSizes::new(bar.needs_you, view, theme);
        let labels: Vec<Label> = bar
            .controls
            .iter()
            .map(|control| Label::new(control, &sizes, view, theme, text))
            .collect();
        let glass = sizes.glass(bar, &labels, view, theme, text);
        let row_top = glass.y + sizes.inset;

        let mut right = glass.x + glass.w - sizes.inset;
        let mut controls = Vec::with_capacity(labels.len());
        let right_to_left = labels.into_iter().rev();
        for label in right_to_left {
            let control = label.place(right, row_top, &sizes, text);
            right = control.rect.x - sizes.gap;
            controls.push(control);
        }
        controls.reverse();

        let text_left = glass.x + sizes.text_inset + sizes.mark_room;
        let lines = sizes.text_lines(bar, (text_left, row_top), right - text_left, theme, text);
        let mark = bar.needs_you.then(|| Rect {
            x: glass.x + sizes.text_inset,
            y: row_top + ((sizes.row_height - sizes.mark_size) * 0.5).round(),
            w: sizes.mark_size,
            h: sizes.mark_size,
        });
        let progress = bar.progress.map(|done| ProgressLayout {
            track: Rect {
                x: glass.x + sizes.text_inset,
                y: row_top + sizes.row_height + sizes.gap,
                w: glass.w - sizes.text_inset - sizes.inset - view.whole(theme.progress_end_inset),
                h: sizes.progress_height,
            },
            done: done.clamp(0.0, 1.0),
        });

        Self {
            glass,
            controls,
            lines,
            mark,
            progress,
        }
    }

    /// The index of the control under the point.
    pub(crate) fn control_at(&self, x: f32, y: f32) -> Option<usize> {
        self.controls
            .iter()
            .position(|control| control.rect.contains(x, y))
    }

    /// The index of the quiet control under the point: the one that shows hover.
    pub(crate) fn quiet_control_at(&self, x: f32, y: f32) -> Option<usize> {
        let at = self.control_at(x, y)?;
        let control = self.controls.get(at)?;
        (control.emphasis == Emphasis::Quiet).then_some(at)
    }

    /// Adds its shapes to `list`, with a fill behind the control at `hovered`.
    pub(crate) fn draw(
        &self,
        theme: &Theme,
        view: View,
        hovered: Option<usize>,
        list: &mut DrawList,
    ) {
        let needs_you = self.mark.is_some();
        list.rects.push(RoundedRect {
            rect: self.glass,
            radius: theme.radius * view.scale,
            fill: theme.glass,
            stroke: if needs_you {
                theme.needs_you_hairline
            } else {
                theme.hairline
            },
            stroke_width: view.whole(theme.hairline_width).max(1.0),
            layer: Layer::Bar,
        });
        let numbered = self.controls.iter().enumerate();
        for (index, control) in numbered {
            let fill = match control.emphasis {
                Emphasis::Filled => Some(theme.accent),
                Emphasis::Quiet => (hovered == Some(index)).then_some(theme.hover),
            };
            let shape =
                fill.map(|fill| plain(control.rect, theme.control_radius * view.scale, fill));
            list.rects.extend(shape);
            control.label.draw(Layer::Bar, list);
        }
        let mark = self
            .mark
            .map(|rect| plain(rect, rect.w * 0.5, theme.needs_you));
        list.rects.extend(mark);
        if let Some(ProgressLayout { track, done }) = self.progress {
            let radius = track.h * 0.5;
            list.rects.push(plain(track, radius, theme.progress_track));
            let line = Rect {
                w: track.w * done,
                ..track
            };
            list.rects.push(plain(line, radius, theme.accent));
        }
        for line in &self.lines {
            line.draw(Layer::Bar, list);
        }
    }
}

/// A shape of a bar with a fill and no hairline.
const fn plain(rect: Rect, radius: f32, fill: Colour) -> RoundedRect {
    RoundedRect {
        rect,
        radius,
        fill,
        stroke: Colour::CLEAR,
        stroke_width: 0.0,
        layer: Layer::Bar,
    }
}

/// A chip, placed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChipLayout {
    /// Its surface and hairline.
    glass: RoundedRect,
    /// Its mark, if it has one.
    mark: Option<RoundedRect>,
    /// Its text.
    label: PlacedLine,
}

impl ChipLayout {
    /// Places `chip` in `view`.
    pub(crate) fn new(chip: &Chip, view: View, theme: &Theme, text: &mut Text) -> Self {
        let opacity = if chip.dimmed { theme.saved_alpha } else { 1.0 };
        let px = theme.status.0 * view.scale;
        let line_height = view.whole(theme.status.1);
        let line = text.line(&chip.text, theme.status_weight, px);
        let height = view.whole(theme.chip_height);
        let padding = view.whole(theme.chip_padding);
        let mark_size = view.whole(theme.mark_size);
        let gap = view.whole(theme.gap);
        let mark_room = match chip.mark {
            Mark::Dot | Mark::Ring => mark_size + gap,
            Mark::None => 0.0,
        };
        let width = 2.0 * padding + mark_room + line.width.ceil();
        let inset = view.whole(theme.chip_inset);
        let rect = Rect {
            x: view.width - inset - width,
            y: inset,
            w: width,
            h: height,
        };
        let mark_rect = Rect {
            x: rect.x + padding,
            y: rect.y + ((height - mark_size) * 0.5).round(),
            w: mark_size,
            h: mark_size,
        };
        let accent = theme.accent.faded(opacity);
        let mark = match chip.mark {
            Mark::Dot => Some((accent, Colour::CLEAR, 0.0)),
            Mark::Ring => Some((Colour::CLEAR, accent, theme.ring_width * view.scale)),
            Mark::None => None,
        };
        let mark = mark.map(|(fill, stroke, stroke_width)| RoundedRect {
            rect: mark_rect,
            radius: mark_size * 0.5,
            fill,
            stroke,
            stroke_width,
            layer: Layer::Chip,
        });
        Self {
            glass: RoundedRect {
                rect,
                radius: height * 0.5,
                fill: theme.glass.faded(opacity),
                stroke: theme.hairline.faded(opacity),
                stroke_width: view.whole(theme.hairline_width).max(1.0),
                layer: Layer::Chip,
            },
            mark,
            label: PlacedLine {
                x: rect.x + padding + mark_room,
                baseline: rect.y
                    + ((height - line_height) * 0.5).round()
                    + text.baseline(theme.status_weight, px, line_height),
                line,
                colour: theme.text.faded(opacity),
            },
        }
    }

    /// Adds its shapes to `list`.
    pub(crate) fn draw(&self, list: &mut DrawList) {
        list.rects.push(self.glass);
        list.rects.extend(self.mark);
        self.label.draw(Layer::Chip, list);
    }
}
