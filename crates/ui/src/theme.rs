//! The look: every colour, size and duration of the UI, and the contrast arithmetic that checks
//! them.
//!
//! Colours are display (sRGB-encoded) values, and the UI blends in them, as the mockups do.

use std::time::Duration;

use crate::text::Weight;

/// Where a renderer's blending happens, which decides how light glass comes out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    /// In display values, as the mockups do.
    Display,
    /// In linear light, on a surface with no other way: glass comes out lighter.
    Linear,
}

/// A colour in display (sRGB-encoded) values, straight alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Colour {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
    /// Opacity: 0 is none, 255 full.
    pub a: u8,
}

impl Colour {
    /// No colour at all: an unfilled shape.
    pub const CLEAR: Self = Self::rgba(0, 0, 0, 0);

    /// Opaque.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// With alpha.
    #[must_use]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// This with its alpha scaled by `opacity`, 0 to 1.
    #[must_use]
    pub fn faded(self, opacity: f32) -> Self {
        Self {
            a: to_channel(f64::from(self.a) * f64::from(opacity)),
            ..self
        }
    }

    /// This over an opaque `ground`, blended in display values, as the UI pass does. The result is
    /// opaque.
    #[must_use]
    pub fn over(self, ground: Self) -> Self {
        let alpha = f64::from(self.a) / 255.0;
        let blend = |top: u8, under: u8| {
            to_channel(alpha * f64::from(top) + (1.0 - alpha) * f64::from(under))
        };
        Self::rgb(
            blend(self.r, ground.r),
            blend(self.g, ground.g),
            blend(self.b, ground.b),
        )
    }

    /// This over an opaque `ground`, blended as `blend` says. The result is opaque.
    #[must_use]
    pub fn over_in(self, ground: Self, blend: Blend) -> Self {
        match blend {
            Blend::Display => self.over(ground),
            Blend::Linear => {
                let alpha = f64::from(self.a) / 255.0;
                let mixed = |top: u8, under: u8| {
                    encoded(alpha * decoded(top) + (1.0 - alpha) * decoded(under))
                };
                Self::rgb(
                    mixed(self.r, ground.r),
                    mixed(self.g, ground.g),
                    mixed(self.b, ground.b),
                )
            }
        }
    }

    /// WCAG relative luminance, 0 to 1. Alpha plays no part: blend with [`Colour::over`] first.
    #[must_use]
    pub fn luminance(self) -> f64 {
        0.2126 * decoded(self.r) + 0.7152 * decoded(self.g) + 0.0722 * decoded(self.b)
    }

    /// WCAG contrast ratio against `other`, 1 to 21.
    #[must_use]
    pub fn contrast(self, other: Self) -> f64 {
        let (one, two) = (self.luminance(), other.luminance());
        (one.max(two) + 0.05) / (one.min(two) + 0.05)
    }
}

/// A value of 0 to 255 as the nearest channel value.
fn to_channel(value: f64) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded and clamped to 0 to 255 first"
    )]
    let channel = value.round().clamp(0.0, 255.0) as u8;
    channel
}

/// A channel in linear light, 0 to 1: the sRGB decoding WCAG's luminance is defined on.
fn decoded(channel: u8) -> f64 {
    let value = f64::from(channel) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// A value in linear light, 0 to 1, as the nearest channel value.
fn encoded(value: f64) -> u8 {
    let display = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    to_channel(display * 255.0)
}

/// Every colour, size and duration of the UI. Nothing else in the workspace hard-codes one, so a
/// new direction is a new constant. Sizes are logical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Theme {
    /// The one surface: bars and chips.
    pub glass: Colour,
    /// The line around glass.
    pub hairline: Colour,
    /// The fill behind a quiet control under the pointer.
    pub hover: Colour,
    /// The one action on a bar, and work under way.
    pub accent: Colour,
    /// Text on the accent.
    pub accent_text: Colour,
    /// This needs the user.
    pub needs_you: Colour,
    /// The line around glass that needs the user.
    pub needs_you_hairline: Colour,
    /// The groove a progress line runs in.
    pub progress_track: Colour,
    /// Text on glass.
    pub text: Colour,
    /// Secondary text on glass.
    pub quiet_text: Colour,
    /// The width of the line around glass.
    pub hairline_width: f32,
    /// The corners of a bar.
    pub radius: f32,
    /// The corners of a control.
    pub control_radius: f32,
    /// The height of a control: a pen's target.
    pub control_height: f32,
    /// A quiet control's label in from its sides.
    pub control_padding: f32,
    /// The filled control's label in from its sides.
    pub filled_padding: f32,
    /// Between neighbouring controls.
    pub gap: f32,
    /// A bar's padding around its controls.
    pub bar_inset: f32,
    /// A bar's text in from its left edge.
    pub bar_text_inset: f32,
    /// A bar up from the window's bottom edge.
    pub bar_bottom: f32,
    /// A bar's narrowest, in any window that has the room.
    pub bar_min_width: f32,
    /// The least between a bar and the window's sides.
    pub bar_margin: f32,
    /// A progress line's thickness.
    pub progress_height: f32,
    /// A progress line's end in from a bar's padding.
    pub progress_end_inset: f32,
    /// The height of a chip, whose corners are fully round.
    pub chip_height: f32,
    /// A chip in from the window's top and right edges.
    pub chip_inset: f32,
    /// A chip's content in from its sides.
    pub chip_padding: f32,
    /// The side of a dot or ring.
    pub mark_size: f32,
    /// The width of a ring's line.
    pub ring_width: f32,
    /// From a bar's needs-you mark to its text.
    pub needs_you_gap: f32,
    /// Body text: size and line height.
    pub body: (f32, f32),
    /// Status text: size and line height.
    pub status: (f32, f32),
    /// A bar's title.
    pub title_weight: Weight,
    /// A quiet control's label.
    pub control_weight: Weight,
    /// The filled control's label.
    pub filled_weight: Weight,
    /// A bar's detail and a chip's text.
    pub status_weight: Weight,
    /// How long anything takes to appear or leave.
    pub fade: Duration,
    /// A chip's opacity while the pen is down.
    pub chip_pen_down: f32,
    /// How long a bar's new controls can't be hit, so a double click's second press can't land on
    /// what its first brought up.
    pub swap_lockout: Duration,
    /// How long "Saved" stays before it fades.
    pub saved_hold: Duration,
    /// How long a note that asks nothing stays before it fades.
    pub note_hold: Duration,
    /// The least time between steps of a progress line, so its list isn't rebuilt every frame.
    pub progress_step: Duration,
    /// Glass alpha where the pass can only blend in linear light, which lightens glass.
    pub glass_alpha_linear: f32,
    /// The smallest a window may be, logical pixels: a bar's controls always fit.
    pub window_min: (u32, u32),
}

impl Theme {
    /// Glass: smoked, one small accent, the model never dimmed. For now, not final.
    pub const GLASS: Self = Self {
        glass: Colour::rgba(22, 23, 26, 168),
        hairline: Colour::rgba(255, 255, 255, 36),
        hover: Colour::rgba(255, 255, 255, 31),
        accent: Colour::rgb(0xF0, 0x8A, 0x3C),
        accent_text: Colour::rgb(0x1C, 0x12, 0x08),
        needs_you: Colour::rgb(0xFF, 0x6F, 0x61),
        needs_you_hairline: Colour::rgba(0xFF, 0x6F, 0x61, 153),
        progress_track: Colour::rgba(255, 255, 255, 41),
        text: Colour::rgb(0xF4, 0xF5, 0xF6),
        quiet_text: Colour::rgb(0xDC, 0xDE, 0xE1),
        hairline_width: 1.0,
        radius: 14.0,
        control_radius: 10.0,
        control_height: 44.0,
        control_padding: 16.0,
        filled_padding: 24.0,
        gap: 8.0,
        bar_inset: 12.0,
        bar_text_inset: 22.0,
        bar_bottom: 28.0,
        bar_min_width: 480.0,
        bar_margin: 20.0,
        progress_height: 3.0,
        progress_end_inset: 10.0,
        chip_height: 32.0,
        chip_inset: 20.0,
        chip_padding: 12.0,
        mark_size: 8.0,
        ring_width: 1.5,
        needs_you_gap: 14.0,
        body: (15.0, 22.0),
        status: (13.0, 18.0),
        title_weight: Weight::Medium,
        control_weight: Weight::Medium,
        filled_weight: Weight::SemiBold,
        status_weight: Weight::Regular,
        fade: Duration::from_millis(120),
        chip_pen_down: 0.25,
        swap_lockout: Duration::from_millis(120),
        saved_hold: Duration::from_secs(2),
        note_hold: Duration::from_secs(10),
        progress_step: Duration::from_millis(100),
        glass_alpha_linear: 0.9,
        window_min: (640, 400),
    };

    /// This theme for a renderer that blends as `blend` says: as it is in display values, and with
    /// glass raised to `glass_alpha_linear` in linear light.
    #[must_use]
    pub fn for_blend(self, blend: Blend) -> Self {
        match blend {
            Blend::Display => self,
            Blend::Linear => Self {
                glass: Colour {
                    a: to_channel(f64::from(self.glass_alpha_linear) * 255.0),
                    ..self.glass
                },
                ..self
            },
        }
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// The brightest clay the MatCaps show, in display colour.
    const BRIGHTEST_CLAY: Colour = Colour::rgb(0xD6, 0xD6, 0xD2);

    #[test]
    fn text_on_glass_meets_contrast_over_the_brightest_clay() {
        let theme = Theme::GLASS;
        let ground = theme.glass.over(BRIGHTEST_CLAY);
        for text in [theme.text, theme.quiet_text] {
            assert!(text.contrast(ground) >= 4.5, "{text:?} on {ground:?}");
        }
        assert!(
            theme.accent_text.contrast(theme.accent) >= 4.5,
            "text on the accent"
        );
    }

    #[test]
    fn text_on_glass_meets_contrast_in_linear_light_too() {
        let theme = Theme::GLASS.for_blend(Blend::Linear);
        let ground = theme.glass.over_in(BRIGHTEST_CLAY, Blend::Linear);
        for text in [theme.text, theme.quiet_text] {
            assert!(text.contrast(ground) >= 4.5, "{text:?} on {ground:?}");
        }

        // The bite: the same glass as it is, in linear light, falls short.
        let plain = Theme::GLASS.glass.over_in(BRIGHTEST_CLAY, Blend::Linear);
        assert!(theme.text.contrast(plain) < 4.5, "{plain:?}");
    }

    #[test]
    fn only_linear_light_raises_the_glass() {
        let theme = Theme::GLASS;
        assert_eq!(theme.for_blend(Blend::Display), theme, "as it is");
        let raised = theme.for_blend(Blend::Linear);
        assert_eq!(raised.glass.a, 229, "0.9 of 255");
        assert_eq!(raised.hairline, theme.hairline, "nothing else moves");
    }

    #[test]
    fn controls_are_big_enough_for_the_pen() {
        let theme = Theme::GLASS;
        assert!(theme.control_height >= 44.0, "a pen needs a 44 px target");
    }

    #[test]
    fn glass_over_the_brightest_clay_is_the_grey_of_the_mockups() {
        assert_eq!(
            Theme::GLASS.glass.over(BRIGHTEST_CLAY),
            Colour::rgb(88, 88, 89),
            "blended in display values"
        );
    }

    #[test]
    fn fading_scales_alpha_alone() {
        let colour = Colour::rgba(10, 20, 30, 200);
        assert_eq!(colour.faded(0.5), Colour::rgba(10, 20, 30, 100), "half");
        assert_eq!(colour.faded(0.0).a, 0, "none");
        assert_eq!(colour.faded(1.0), colour, "all");
    }

    #[test]
    fn contrast_runs_from_one_to_twenty_one() {
        let (black, white) = (Colour::rgb(0, 0, 0), Colour::rgb(255, 255, 255));
        assert!(
            (black.contrast(white) - 21.0).abs() < 1e-9,
            "black on white"
        );
        assert!((white.contrast(white) - 1.0).abs() < 1e-9, "white on white");
    }
}
