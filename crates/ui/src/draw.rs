//! What a renderer draws: rounded rectangles, glyphs and the fades over them.
//!
//! Positions are physical pixels from the window's top left. Colours are display values, straight
//! alpha, and blending in them is the renderer's part of the look.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::text::AtlasImage;
use crate::theme::Colour;

/// Which fade a shape follows: bars and the chip fade apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// A bar, whose fade is `fades[0]`.
    Bar = 0,
    /// The chip, whose fade is `fades[1]`.
    Chip = 1,
}

/// A rectangle, physical pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Rect {
    /// Whether the point is inside.
    #[must_use]
    pub fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.w && y < self.y + self.h
    }
}

/// A rounded rectangle with an optional hairline, which lies inside its edge, over its fill.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundedRect {
    /// Where it goes.
    pub rect: Rect,
    /// Its corners.
    pub radius: f32,
    /// Inside the hairline.
    pub fill: Colour,
    /// The hairline.
    pub stroke: Colour,
    /// The hairline's width: 0 for none.
    pub stroke_width: f32,
    /// The fade it follows.
    pub layer: Layer,
}

/// A glyph: where it goes, where it is in the atlas, its colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphQuad {
    /// Where it goes, a texel a pixel.
    pub rect: Rect,
    /// Its top left texel.
    pub atlas: (u32, u32),
    /// Its colour.
    pub colour: Colour,
    /// The fade it follows.
    pub layer: Layer,
}

/// Everything to draw, back to front: rects, then glyphs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrawList {
    /// The rectangles, back to front.
    pub rects: Vec<RoundedRect>,
    /// The glyphs, over every rectangle.
    pub glyphs: Vec<GlyphQuad>,
}

impl DrawList {
    /// Whether there is nothing to draw.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rects.is_empty() && self.glyphs.is_empty()
    }
}

/// An opacity moving from `from` to `to` over `length` from `start`, linearly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fade {
    /// The opacity at `start`.
    pub from: f32,
    /// The opacity at the end, and after.
    pub to: f32,
    /// When it began.
    pub start: Instant,
    /// How long it takes.
    pub length: Duration,
}

impl Fade {
    /// A fade at rest.
    #[must_use]
    pub fn at_rest(value: f32, now: Instant) -> Self {
        Self {
            from: value,
            to: value,
            start: now,
            length: Duration::ZERO,
        }
    }

    /// The opacity at `now`.
    #[must_use]
    pub fn at(self, now: Instant) -> f32 {
        if self.settled(now) {
            return self.to;
        }
        let elapsed = now.saturating_duration_since(self.start);
        self.from + (self.to - self.from) * (elapsed.as_secs_f32() / self.length.as_secs_f32())
    }

    /// Whether it has finished by `now`.
    #[must_use]
    pub fn settled(self, now: Instant) -> bool {
        now.saturating_duration_since(self.start) >= self.length
    }

    /// This fade, turned at `now` to head for `to` from wherever it has got to. Already heading
    /// there, it carries on as it was.
    pub(crate) fn toward(self, to: f32, length: Duration, now: Instant) -> Self {
        if (self.to - to).abs() < f32::EPSILON {
            return self;
        }
        Self {
            from: self.at(now),
            to,
            start: now,
            length,
        }
    }
}

/// What a frame draws: shared, so handing it on copies nothing.
#[derive(Debug, Clone)]
pub struct UiFrame {
    /// The shapes. The same `Arc` until something changes, so a renderer uploads only then.
    pub list: Arc<DrawList>,
    /// The atlas the glyphs point into.
    pub atlas: Arc<AtlasImage>,
    /// The opacity of each [`Layer`], in its order.
    pub fades: [Fade; 2],
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A fade's length in these tests.
    const LENGTH: Duration = Duration::from_millis(100);

    #[test]
    fn a_rect_holds_its_top_left_and_not_its_bottom_right() {
        let rect = Rect {
            x: 10.0,
            y: 20.0,
            w: 30.0,
            h: 40.0,
        };
        assert!(rect.contains(10.0, 20.0), "the top left corner");
        assert!(rect.contains(39.9, 59.9), "just inside");
        assert!(!rect.contains(40.0, 30.0), "the right edge");
        assert!(!rect.contains(20.0, 60.0), "the bottom edge");
        assert!(!rect.contains(9.9, 30.0), "just left");
    }

    #[test]
    fn a_fade_runs_linearly_and_settles() {
        let start = Instant::now();
        let fade = Fade {
            from: 1.0,
            to: 0.0,
            start,
            length: LENGTH,
        };
        assert!((fade.at(start) - 1.0).abs() < 1e-6, "at its start");
        let halfway = start + LENGTH / 2;
        assert!((fade.at(halfway) - 0.5).abs() < 1e-6, "halfway");
        assert!(!fade.settled(halfway), "still moving");
        assert!(fade.settled(start + LENGTH), "done");
        assert!(fade.at(start + LENGTH * 2).abs() < 1e-6, "and stays done");
    }

    #[test]
    fn a_fade_at_rest_is_settled_at_once() {
        let now = Instant::now();
        let fade = Fade::at_rest(0.25, now);
        assert!(fade.settled(now), "nowhere to go");
        assert!((fade.at(now) - 0.25).abs() < 1e-6, "its value");
    }

    #[test]
    fn a_fade_turned_midway_starts_from_where_it_had_got_to() {
        let start = Instant::now();
        let halfway = start + LENGTH / 2;
        let out = Fade::at_rest(1.0, start).toward(0.0, LENGTH, start);
        let back = out.toward(1.0, LENGTH, halfway);
        assert!((back.at(halfway) - 0.5).abs() < 1e-6, "no jump");
        assert_eq!(back.start, halfway, "it begins again now");
    }

    #[test]
    fn a_fade_already_heading_there_is_left_alone() {
        let start = Instant::now();
        let fade = Fade::at_rest(1.0, start).toward(0.0, LENGTH, start);
        assert_eq!(
            fade.toward(0.0, LENGTH, start + LENGTH / 2),
            fade,
            "no restart"
        );
    }
}
