//! Fonts, the glyph atlas and line layout.
//!
//! Lines run left to right, a glyph a character, with no shaping or kerning: [`Text::line`] is the
//! seam where shaping goes later. Plex has no `tnum` feature and needs none: its default digits all
//! share one advance, in every weight, so status text is tabular as it stands.

use std::collections::HashMap;
use std::sync::Arc;

use ttf_parser::{Face, GlyphId};

use crate::raster::{Raster, coordinate};

/// The regular face.
const REGULAR: &[u8] = include_bytes!("../fonts/IBMPlexSans-Regular.ttf");
/// The medium face.
const MEDIUM: &[u8] = include_bytes!("../fonts/IBMPlexSans-Medium.ttf");
/// The semibold face.
const SEMI_BOLD: &[u8] = include_bytes!("../fonts/IBMPlexSans-SemiBold.ttf");

/// The atlas's side to begin with.
const ATLAS_SIZE: u32 = 1024;
/// The side it grows to when full, enough for a 4K scale's glyphs, and no further.
const LARGEST_ATLAS_SIZE: u32 = 2048;
/// Shelf heights are multiples of this, so glyphs of nearly one height share a shelf.
const SHELF_STEP: u32 = 8;

/// A face of the bundled font.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weight {
    /// 400.
    Regular,
    /// 500.
    Medium,
    /// 600.
    SemiBold,
}

/// The glyph atlas: one byte of coverage per texel, shelf packed. 1024 x 1024; grows to 2048 square
/// if full (a 4K scale's glyphs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtlasImage {
    /// Texels a side.
    pub size: u32,
    /// Coverage, row by row.
    pub texels: Vec<u8>,
    /// Goes up whenever a texel changes, so a renderer uploads only then.
    pub version: u64,
}

impl AtlasImage {
    /// An atlas of the first size with nothing in it.
    fn empty(version: u64) -> Self {
        Self {
            size: ATLAS_SIZE,
            texels: vec![0; (ATLAS_SIZE * ATLAS_SIZE) as usize],
            version,
        }
    }

    /// `coverage`, `width` wide, copied in at `at`, which the caller has reserved inside the image.
    fn put(&mut self, at: (u32, u32), width: usize, coverage: &[u8]) {
        let side = self.size as usize;
        let (x, y) = (at.0 as usize, at.1 as usize);
        let rows = self
            .texels
            .chunks_exact_mut(side)
            .skip(y)
            .zip(coverage.chunks_exact(width));
        for (row, source) in rows {
            if let Some(target) = row.get_mut(x..x + width) {
                target.copy_from_slice(source);
            }
        }
        self.version += 1;
    }

    /// Doubles the side, what it holds staying where it is.
    fn grow(&mut self) {
        let (old, new) = (self.size as usize, self.size as usize * 2);
        let mut texels = vec![0; new * new];
        let rows = texels
            .chunks_exact_mut(new)
            .zip(self.texels.chunks_exact(old));
        for (target, source) in rows {
            if let Some(start) = target.get_mut(..old) {
                start.copy_from_slice(source);
            }
        }
        self.texels = texels;
        self.size *= 2;
        self.version += 1;
    }
}

/// One glyph placed by a layout: where it goes relative to the line's origin, and where it is in
/// the atlas, both in physical pixels. The origin is on the baseline, so y is mostly negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedGlyph {
    /// The top left corner, from the line's origin.
    pub at: (f32, f32),
    /// Width and height.
    pub size: (f32, f32),
    /// The top left texel in the atlas.
    pub atlas: (u32, u32),
}

/// A line laid out. Left to right, unshaped; [`Text::line`] is the seam where shaping goes later.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// What to draw. Blanks have no entry.
    pub glyphs: Vec<PlacedGlyph>,
    /// The sum of the advances, in physical pixels.
    pub width: f32,
}

/// A row of the atlas glyphs of about one height are packed along.
#[derive(Debug, Clone, Copy)]
struct Shelf {
    /// Its top.
    y: u32,
    /// Its height.
    height: u32,
    /// Where the next glyph goes.
    x: u32,
}

/// A rasterised glyph, relative to its origin on the baseline.
#[derive(Debug)]
struct Bitmap {
    /// The left edge.
    left: f32,
    /// The top edge.
    top: f32,
    /// Pixels across.
    width: usize,
    /// Coverage, row by row.
    coverage: Vec<u8>,
}

/// A glyph with ink, in the atlas.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// The top left corner, from the glyph's origin.
    offset: (f32, f32),
    /// Width and height.
    size: (f32, f32),
    /// The top left texel.
    atlas: (u32, u32),
}

/// The glyph cache's key: a glyph of a face at a size, by the bits of its size in pixels.
type GlyphKey = (Weight, GlyphId, u32);

/// The fonts and atlas.
#[derive(Debug)]
pub struct Text {
    /// The regular face.
    regular: Face<'static>,
    /// The medium face.
    medium: Face<'static>,
    /// The semibold face.
    semi_bold: Face<'static>,
    /// Every glyph asked for: `None` is one with nothing to draw, or that didn't fit.
    glyphs: HashMap<GlyphKey, Option<Slot>>,
    /// The atlas, shared with frames in flight: adding a glyph copies it only if one still holds
    /// it.
    atlas: Arc<AtlasImage>,
    /// The atlas's shelves, top to bottom.
    shelves: Vec<Shelf>,
    /// Where the next shelf goes.
    shelves_end: u32,
}

impl Text {
    /// The bundled faces, parsed. Panics are impossible: tests parse the same bytes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            regular: face(REGULAR),
            medium: face(MEDIUM),
            semi_bold: face(SEMI_BOLD),
            glyphs: HashMap::new(),
            atlas: Arc::new(AtlasImage::empty(0)),
            shelves: Vec::new(),
            shelves_end: 0,
        }
    }

    /// Forgets every glyph: a new, empty atlas, with a higher version so a renderer uploads it.
    ///
    /// Lines laid out before this point into the old atlas. Frames in flight keep that one, and
    /// anything still to be drawn must be laid out again.
    pub fn reset(&mut self) {
        let version = self.atlas.version + 1;
        self.atlas = Arc::new(AtlasImage::empty(version));
        self.glyphs.clear();
        self.shelves.clear();
        self.shelves_end = 0;
    }

    /// Rasterises printable ASCII for `sizes`, physical px.
    pub fn warm(&mut self, sizes: &[(Weight, f32)]) {
        for &(weight, px) in sizes {
            for character in ' '..='~' {
                self.place(character, weight, px);
            }
        }
    }

    /// Lays `text` out at `px`, rasterising what is new. A character the font lacks draws as its
    /// missing glyph; a glyph the atlas can't hold draws as nothing, keeping its advance.
    pub fn line(&mut self, text: &str, weight: Weight, px: f32) -> Line {
        let mut pen: f32 = 0.0;
        let mut glyphs = Vec::with_capacity(text.len());
        for character in text.chars() {
            let (advance, slot) = self.place(character, weight, px);
            let origin = pen.round();
            let placed = slot.map(|slot| PlacedGlyph {
                at: (origin + slot.offset.0, slot.offset.1),
                size: slot.size,
                atlas: slot.atlas,
            });
            glyphs.extend(placed);
            pen += advance;
        }
        Line { glyphs, width: pen }
    }

    /// From the top of a line box `line_height` tall to the baseline of a font centred in it, as
    /// CSS centres one, in whole pixels.
    #[must_use]
    pub fn baseline(&self, weight: Weight, px: f32, line_height: f32) -> f32 {
        let face = self.face(weight);
        let scale = px / f32::from(face.units_per_em());
        let ascent = f32::from(face.ascender()) * scale;
        let descent = f32::from(face.descender()) * scale;
        ((line_height - (ascent - descent)) * 0.5 + ascent).round()
    }

    /// The atlas.
    #[must_use]
    pub const fn atlas(&self) -> &Arc<AtlasImage> {
        &self.atlas
    }

    /// How many glyphs it has been asked for.
    #[cfg(test)]
    pub(crate) fn held_glyphs(&self) -> usize {
        self.glyphs.len()
    }

    /// The face of a weight.
    const fn face(&self, weight: Weight) -> &Face<'static> {
        match weight {
            Weight::Regular => &self.regular,
            Weight::Medium => &self.medium,
            Weight::SemiBold => &self.semi_bold,
        }
    }

    /// A character's advance, and its slot, rasterising it if it's new.
    fn place(&mut self, character: char, weight: Weight, px: f32) -> (f32, Option<Slot>) {
        let face = self.face(weight);
        let glyph = face.glyph_index(character).unwrap_or(GlyphId(0));
        let units = face.glyph_hor_advance(glyph).unwrap_or(0);
        let advance = f32::from(units) * px / f32::from(face.units_per_em());
        let key = (weight, glyph, px.to_bits());
        if let Some(&slot) = self.glyphs.get(&key) {
            return (advance, slot);
        }
        let bitmap = rasterise(self.face(weight), glyph, px);
        let slot = bitmap.and_then(|bitmap| self.store(&bitmap));
        self.glyphs.insert(key, slot);
        (advance, slot)
    }

    /// Finds `bitmap` a place in the atlas and copies it there, unless the atlas is full.
    fn store(&mut self, bitmap: &Bitmap) -> Option<Slot> {
        let height = bitmap.coverage.len() / bitmap.width;
        let (width, tall) = (
            u32::try_from(bitmap.width).ok()?,
            u32::try_from(height).ok()?,
        );
        let at = self.reserve(width, tall)?;
        Arc::make_mut(&mut self.atlas).put(at, bitmap.width, &bitmap.coverage);
        Some(Slot {
            offset: (bitmap.left, bitmap.top),
            size: (coordinate(bitmap.width), coordinate(height)),
            atlas: at,
        })
    }

    /// Reserves a `width` by `height` space, growing the atlas if it must.
    fn reserve(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        let shelf_height = height.next_multiple_of(SHELF_STEP);
        loop {
            let size = self.atlas.size;
            let fits = |shelf: &&mut Shelf| shelf.height == shelf_height && shelf.x + width <= size;
            if let Some(shelf) = self.shelves.iter_mut().find(fits) {
                let at = (shelf.x, shelf.y);
                shelf.x += width;
                return Some(at);
            }
            if width <= size && self.shelves_end + shelf_height <= size {
                let y = self.shelves_end;
                self.shelves.push(Shelf {
                    y,
                    height: shelf_height,
                    x: width,
                });
                self.shelves_end += shelf_height;
                return Some((0, y));
            }
            if size >= LARGEST_ATLAS_SIZE {
                return None;
            }
            Arc::make_mut(&mut self.atlas).grow();
        }
    }
}

impl Default for Text {
    fn default() -> Self {
        Self::new()
    }
}

/// A bundled face, parsed.
///
/// # Panics
///
/// Never: the bytes are compiled in, and every test of this module parses them.
fn face(bytes: &'static [u8]) -> Face<'static> {
    #[expect(
        clippy::expect_used,
        reason = "the bytes are compiled in, and every test of this module parses them"
    )]
    Face::parse(bytes, 0).expect("a bundled font parses")
}

/// A glyph's coverage at `px`, or `None` if it has no ink, or could never fit in the atlas, or `px`
/// makes no sense.
fn rasterise(face: &Face<'_>, glyph: GlyphId, px: f32) -> Option<Bitmap> {
    let scale = px / f32::from(face.units_per_em());
    let bounds = face.glyph_bounding_box(glyph)?;
    let left = (f32::from(bounds.x_min) * scale).floor();
    let right = (f32::from(bounds.x_max) * scale).ceil();
    let top = (-f32::from(bounds.y_max) * scale).floor();
    let bottom = (-f32::from(bounds.y_min) * scale).ceil();
    let width = whole_size(right - left)?;
    let height = whole_size(bottom - top)?;
    let mut raster = Raster::placed(width, height, scale, (-left, -top));
    face.outline_glyph(glyph, &mut raster)?;
    let coverage = raster.finish();
    Some(Bitmap {
        left,
        top,
        width,
        coverage,
    })
}

/// A whole number of pixels that could fit in the largest atlas, if it is one.
fn whole_size(size: f32) -> Option<usize> {
    let largest = coordinate(LARGEST_ATLAS_SIZE as usize);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "whole and between 1 and the atlas's side, checked first"
    )]
    (size >= 1.0 && size <= largest).then_some(size as usize)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use ttf_parser::Face;

    use super::*;

    /// Every string the spec has the user read, at the sizes it shows them.
    const SPEC_STRINGS: [&str; 19] = [
        "Save changes to first-sculpt?",
        "Don't save",
        "Keep sculpting",
        "Save",
        "Finishing your stroke",
        "Cancel",
        "Saving",
        "Saved",
        "Unsaved",
        "Your unsaved work is back",
        "From yesterday at 16:42",
        "Use last save instead",
        "Unsaved work found from an earlier version",
        "Open as a copy",
        "Ignore",
        "Couldn't save: the disk is full",
        "Your work is safe and still open.",
        "Dismiss",
        "Try again",
    ];

    /// The sizes warmed in these tests.
    const SIZES: [(Weight, f32); 3] = [
        (Weight::Medium, 15.0),
        (Weight::SemiBold, 15.0),
        (Weight::Regular, 13.0),
    ];

    #[test]
    fn a_line_is_as_wide_as_its_advances_and_its_glyphs_sit_on_the_baseline() {
        let face = Face::parse(REGULAR, 0).unwrap();
        let advance = |c: char| {
            let glyph = face.glyph_index(c).unwrap();
            f32::from(face.glyph_hor_advance(glyph).unwrap())
        };
        let units: f32 = "Save".chars().map(advance).sum();
        let expected = units * 15.0 / f32::from(face.units_per_em());

        let mut text = Text::new();
        let line = text.line("Save", Weight::Regular, 15.0);
        assert!((line.width - expected).abs() < 1e-3, "{}", line.width);
        assert_eq!(line.glyphs.len(), 4, "one quad a letter");
        for glyph in &line.glyphs {
            assert!(glyph.at.1 < 0.0, "{glyph:?} rises from the baseline");
            let bottom = glyph.at.1 + glyph.size.1;
            assert!(
                bottom <= 1.0,
                "{glyph:?} sits on the baseline, overshoot aside"
            );
        }

        let flat = text.line("E", Weight::Regular, 15.0);
        let foot = flat.glyphs[0].at.1 + flat.glyphs[0].size.1;
        assert!(
            foot.abs() < 1e-3,
            "a flat foot ends on the baseline: {foot}"
        );

        let hanging = text.line("gyp", Weight::Regular, 15.0);
        for glyph in &hanging.glyphs {
            let bottom = glyph.at.1 + glyph.size.1;
            assert!(bottom >= 2.0, "{glyph:?} descends");
        }
    }

    #[test]
    fn glyph_origins_are_whole_pixels() {
        let mut text = Text::new();
        let line = text.line("Keep sculpting", Weight::Medium, 15.0);
        for glyph in &line.glyphs {
            assert!((glyph.at.0.fract()).abs() < 1e-6, "{glyph:?}");
            assert!((glyph.at.1.fract()).abs() < 1e-6, "{glyph:?}");
        }
    }

    #[test]
    fn printable_ascii_needs_no_new_glyphs_after_warming() {
        let mut text = Text::new();
        text.warm(&SIZES);
        let version = text.atlas().version;
        for string in SPEC_STRINGS {
            for (weight, px) in SIZES {
                let line = text.line(string, weight, px);
                assert!(line.width > 0.0, "{string} has a width");
            }
        }
        assert_eq!(text.atlas().version, version, "the atlas stayed as it was");
    }

    #[test]
    fn a_new_character_rasterises_on_demand_and_only_once() {
        let mut text = Text::new();
        text.warm(&SIZES);
        let warmed = text.atlas().version;
        let first = text.line("\u{e9}", Weight::Regular, 13.0);
        let taken = text.atlas().version;
        assert!(taken > warmed, "the atlas took the glyph");
        let second = text.line("\u{e9}", Weight::Regular, 13.0);
        assert_eq!(text.atlas().version, taken, "already there");
        assert_eq!(first, second, "the same place");
    }

    #[test]
    fn a_character_the_font_lacks_draws_as_its_missing_glyph_not_nothing() {
        let mut text = Text::new();
        let line = text.line("\u{4E2D}", Weight::Regular, 15.0);
        assert_eq!(line.glyphs.len(), 1, "the missing glyph");
        assert!(line.width > 0.0, "it takes room");
        let again = text.line("\u{6587}", Weight::Regular, 15.0);
        assert_eq!(line, again, "any lacking character looks the same");
    }

    #[test]
    fn a_space_takes_room_and_draws_nothing() {
        let mut text = Text::new();
        let line = text.line(" ", Weight::Regular, 15.0);
        assert!(line.glyphs.is_empty(), "nothing to draw");
        assert!(line.width > 0.0, "it takes room");
    }

    #[test]
    fn digits_are_tabular_in_every_weight() {
        let mut text = Text::new();
        for weight in [Weight::Regular, Weight::Medium, Weight::SemiBold] {
            let widths: Vec<f32> = ('0'..='9')
                .map(|digit| text.line(&digit.to_string().repeat(4), weight, 13.0).width)
                .collect();
            for width in &widths {
                assert!((width - widths[0]).abs() < 1e-3, "{widths:?}");
            }
        }
    }

    #[test]
    fn a_glyph_in_the_atlas_is_the_coverage_it_was_given() {
        let mut text = Text::new();
        let line = text.line("E", Weight::SemiBold, 40.0);
        let glyph = line.glyphs[0];
        let atlas = text.atlas();
        let (x, y) = (glyph.atlas.0 as usize, glyph.atlas.1 as usize);
        let size = atlas.size as usize;
        // The stem of an E at this size is several pixels wide, and starts near its left edge.
        let stem = atlas.texels[(y + whole_size(glyph.size.1).unwrap() / 2) * size + x + 3];
        assert_eq!(stem, 255, "inside the stem");
    }

    #[test]
    fn a_full_atlas_grows_and_leaves_what_it_held_where_it_was() {
        let mut text = Text::new();
        let first = text.line("E", Weight::Regular, 40.0).glyphs[0];
        let width = whole_size(first.size.0).unwrap();
        let before = text.atlas().texels[..width].to_vec();
        assert_eq!(first.atlas, (0, 0), "the first glyph in");
        assert_eq!(text.atlas().size, 1024, "the first size");

        text.warm(&[(Weight::Regular, 200.0)]);
        assert_eq!(text.atlas().size, 2048, "too much for 1024 square");
        assert_eq!(
            text.atlas().texels[..width],
            before[..],
            "its top row is where it was"
        );
        let again = text.line("E", Weight::Regular, 40.0).glyphs[0];
        assert_eq!(again.atlas, first.atlas, "and the cache still points at it");
    }

    #[test]
    fn a_reset_empties_the_atlas_and_leaves_frames_in_flight_theirs() {
        let mut text = Text::new();
        text.warm(&[(Weight::Regular, 200.0)]);
        let in_flight = Arc::clone(text.atlas());
        assert_eq!(in_flight.size, 2048, "grown");

        text.reset();
        assert!(
            text.atlas().version > in_flight.version,
            "a renderer uploads it"
        );
        assert_eq!(text.atlas().size, 1024, "the first size again");
        assert!(text.atlas().texels.iter().all(|&texel| texel == 0), "empty");
        assert!(
            in_flight.texels.iter().any(|&texel| texel > 0),
            "theirs stays"
        );

        let line = text.line("E", Weight::Regular, 40.0);
        assert_eq!(line.glyphs[0].atlas, (0, 0), "packing starts again");
        assert!(
            text.atlas().texels.iter().any(|&texel| texel > 0),
            "and fills"
        );
    }

    #[test]
    fn what_cannot_fit_draws_nothing_and_keeps_its_advance() {
        let mut text = Text::new();
        let line = text.line("M", Weight::Regular, 8000.0);
        assert!(line.glyphs.is_empty(), "bigger than any atlas");
        assert!(line.width > 0.0, "still takes room");
    }

    #[test]
    fn a_senseless_size_draws_nothing_and_never_panics() {
        let mut text = Text::new();
        for px in [0.0, -3.0, f32::NAN, f32::INFINITY] {
            let line = text.line("Save", Weight::Regular, px);
            assert!(line.glyphs.is_empty(), "nothing at {px}");
        }
    }

    #[test]
    fn the_baseline_centres_the_font_in_its_line() {
        let text = Text::new();
        // Plex's ascent and descent are 1.025 and 0.275 em: 19.5 px at 15, 1.25 px of slack above.
        let baseline = text.baseline(Weight::Medium, 15.0, 22.0);
        assert!((baseline - 17.0).abs() < 1e-3, "{baseline}");
    }
}
