//! A small headless UI: tokens, text and a few widgets, turned into a list of shapes to draw.
//!
//! The crate knows nothing of BrushCraft and depends on nothing else here, so a painting app could
//! use it unchanged. It opens no window and touches no GPU: a renderer draws what it hands over,
//! and its tests run anywhere. Design: `specs/2026-11-04-ui-foundation-design.md` in the docs repo.

mod theme;

pub use theme::{Colour, Theme};
