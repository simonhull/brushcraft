//! The debug overlay: what the frame loop reports each frame, as lines of text drawn over the
//! frame's top-left corner in our own bitmap font ([`font`]).
//!
//! Two parts here: [`lines`] says what the overlay shows, from a [`Snapshot`] of the frame
//! loop, and [`TextLayout`] lays the lines out into a one-byte-per-texel image on the CPU. The
//! renderer draws that image as the frame's last pass (`render::overlay`), scaled by a whole
//! number, with a translucent backing so it reads over any scene. No UI toolkit: the app starts
//! with this overlay only.

mod font;

use std::time::Duration;

use render::lod::LodStats;
use render::overlay::TextImage;
use render::surface::{Presentation, Size};
use render::{DrawTimes, Look, SceneView};
use store::JobsReport;

use crate::camera::{Camera, Vec3};
use crate::frame::schedule::StrokePacing;
use crate::frame::{FeedStats, OwnerPhases};
use crate::pen::PenStatus;
use crate::pick::Report as PickReport;

/// Everything the overlay reports about one frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Snapshot<'a> {
    /// The frame's number, from 1.
    pub(crate) frame: u64,
    /// The display's refresh rate the frame loop paces itself to, in hertz.
    pub(crate) refresh_hz: f64,
    /// How frames reach the screen.
    pub(crate) presentation: Presentation,
    /// The GPU's name and backend.
    pub(crate) adapter: &'a str,
    /// The frame's size.
    pub(crate) size: Size,
    /// How long this frame's phases took on the owner thread, up to the jobs.
    pub(crate) owner: OwnerPhases,
    /// How long the render thread took over the last frame it finished; `None` before it has
    /// finished one.
    pub(crate) render: Option<DrawTimes>,
    /// What the stale-tile feed has delivered so far.
    pub(crate) feed: FeedStats,
    /// What this frame's store jobs did.
    pub(crate) jobs: JobsReport,
    /// What the GPU holds (tiles per level, over-budget count, residency), from the uploads;
    /// `None` while nothing is uploaded.
    pub(crate) tiles: Option<&'a str>,
    /// What the latest frame read back drew; `None` before the first, or with no scene.
    pub(crate) scene: Option<&'a LodStats>,
    /// The camera.
    pub(crate) camera: Camera,
    /// How the scene is shown.
    pub(crate) look: Look,
    /// The last pick under the pointer; `None` if the frame loop doesn't pick.
    pub(crate) pick: Option<&'a PickReport>,
    /// Where the pen stream comes from: our backend, or winit's events as the fallback.
    pub(crate) pen: &'a PenStatus,
    /// Whether still frames are filtered instead of accumulated (the image is too large for the
    /// device to accumulate).
    pub(crate) filter_only: bool,
    /// The brush's lines ([`crate::frame::Brush::status`]): its settings and keys, its last
    /// stroke, its last error.
    pub(crate) brush: &'a [String],
    /// The line about the project: its name, whether it has unsaved edits, how its save goes.
    pub(crate) save: &'a str,
    /// The pen's latency in the last frame shown that applied dabs, sample to present
    /// submitted; `None` before any.
    pub(crate) latency: Option<Duration>,
    /// How a brush stroke's frames are paced: on input, or to the refresh where presentation is
    /// FIFO (mailbox not offered).
    pub(crate) pacing: StrokePacing,
}

/// `duration` in milliseconds, to two places.
fn ms(duration: Duration) -> String {
    format!("{:.2}", duration.as_secs_f64() * 1e3)
}

/// `v` as `(x, y, z)` to two places.
fn vec3(v: Vec3) -> String {
    format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z)
}

/// The overlay's camera line: the eye, the pivot and the eye's distance to it, and the
/// magnification.
///
/// The magnification shows its factor while the eye is held at the model's boundary and the
/// image magnified instead ([`crate::camera`]), or that it isn't.
fn camera_line(camera: &Camera) -> String {
    let factor = camera.magnification();
    let magnified = if factor > 1.0 {
        format!("magnified {factor:.2}x (eye at the boundary)")
    } else {
        "not magnified".to_owned()
    };
    format!(
        "camera     eye {}  pivot {} at {:.3}  {magnified}",
        vec3(camera.eye()),
        vec3(camera.pivot()),
        (camera.eye() - camera.pivot()).length(),
    )
}

/// The overlay's lines for `snapshot`: plain ASCII, one fact per field.
pub(crate) fn lines(snapshot: &Snapshot<'_>) -> Vec<String> {
    let Snapshot {
        frame,
        refresh_hz,
        presentation,
        adapter,
        size,
        owner,
        render,
        feed,
        jobs,
        tiles,
        scene,
        camera,
        look,
        pick,
        pen,
        filter_only,
        brush,
        save,
        latency,
        pacing,
    } = *snapshot;
    let view = match look.view {
        SceneView::Shaded => format!("matcap {}", look.material.name()),
        SceneView::Clusters => "clusters (debug view)".to_owned(),
    };
    let wireframe = if look.shading.wireframe { "on" } else { "off" };
    let render_line = render.map_or_else(
        || "render ms  -".to_owned(),
        |phases| {
            format!(
                "render ms  acquire {}  encode {}  present {}",
                ms(phases.acquire),
                ms(phases.encode),
                ms(phases.present)
            )
        },
    );
    let latency_line = format!(
        "latency    {}  ({pacing})",
        latency.map_or_else(
            || "- (no dabs applied yet)".to_owned(),
            |latency| format!(
                "{} ms pen sample to present, last frame with dabs",
                ms(latency)
            )
        )
    );
    let mut lines = vec![
        format!(
            "BrushCraft  frame {frame}  {}x{}  {presentation} at {refresh_hz:.1} Hz  {adapter}",
            size.width(),
            size.height()
        ),
        format!(
            "owner ms   input {}  brush {}  feed {}  uploads {}  upkeep {}  prepare {}  jobs {}",
            ms(owner.input),
            ms(owner.brush),
            ms(owner.feed),
            ms(owner.uploads),
            ms(owner.upkeep),
            ms(owner.prepare),
            ms(owner.jobs)
        ),
        render_line,
        format!(
            "feed       {} epochs  {} tiles  {} resyncs",
            feed.epochs, feed.keys, feed.resyncs
        ),
        format!(
            "jobs       {} slices (longest {} ms)  {} built  {} display tiles left",
            jobs.slices,
            ms(jobs.longest_slice),
            jobs.built,
            jobs.left
        ),
        format!("tiles      {}", tiles.unwrap_or("none on the GPU yet")),
        format!("scene      {}", crate::uploads::scene_summary(scene)),
        camera_line(&camera),
        format!(
            "look       {view}  wireframe {wireframe}  (keys: M matcap, W wireframe, C clusters)"
        ),
        format!(
            "aa         {}",
            if filter_only {
                "filtered only: too large to accumulate on this GPU"
            } else {
                "accumulating while still, filtered while moving"
            }
        ),
        format!("pick       {}", crate::pick::summary(pick)),
        format!("pen        {pen}"),
    ];
    lines.push(save.to_owned());
    lines.extend(brush.iter().cloned());
    lines.push(latency_line);
    lines
}

/// Pixels between glyphs on a line.
const GLYPH_GAP: usize = 1;
/// Pixels between lines.
const LINE_GAP: usize = 2;
/// Pixels of backing around the text.
const PADDING: usize = 3;

/// Lines of text laid out as an image: one byte per texel, 255 where a glyph has ink, 0 on the
/// backing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextLayout {
    /// Texels across.
    width: usize,
    /// Texels down.
    height: usize,
    /// The texels, row by row from the top.
    texels: Vec<u8>,
}

impl TextLayout {
    /// `lines` laid out top to bottom, each from the left edge, in [`font`]'s glyphs, with
    /// [`PADDING`] all round.
    pub(crate) fn new(lines: &[String]) -> Self {
        let advance = font::WIDTH + GLYPH_GAP;
        let line_height = font::HEIGHT + LINE_GAP;
        let longest = lines
            .iter()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0);
        let width = 2 * PADDING + (longest * advance).saturating_sub(GLYPH_GAP);
        let height = 2 * PADDING + (lines.len() * line_height).saturating_sub(LINE_GAP);
        let mut image = Self {
            width,
            height,
            texels: vec![0; width * height],
        };
        let numbered_lines = lines.iter().enumerate();
        for (row, line) in numbered_lines {
            let top = PADDING + row * line_height;
            let numbered_chars = line.chars().enumerate();
            for (column, c) in numbered_chars {
                image.stamp(PADDING + column * advance, top, font::glyph(c));
            }
        }
        image
    }

    /// Draws `glyph`'s ink with its top-left corner at texel (`left`, `top`); whatever falls
    /// outside the image is left out.
    fn stamp(&mut self, left: usize, top: usize, glyph: font::Glyph) {
        let glyph_rows = glyph.iter().enumerate();
        for (dy, bits) in glyph_rows {
            let start = (top + dy) * self.width + left;
            let Some(span) = self.texels.get_mut(start..start + font::WIDTH) else {
                continue;
            };
            let pixels = span.iter_mut().enumerate();
            for (dx, texel) in pixels {
                if bits & (1 << (font::WIDTH - 1 - dx)) != 0 {
                    *texel = u8::MAX;
                }
            }
        }
    }

    /// The layout as the image the renderer's overlay pass draws; `None` if it is wider or
    /// taller than a `u32` counts (no overlay is).
    pub(crate) fn image(self) -> Option<TextImage> {
        let width = u32::try_from(self.width).ok()?;
        let height = u32::try_from(self.height).ok()?;
        TextImage::new(width, height, self.texels)
    }

    /// Texels across.
    #[cfg(test)]
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// Texels down.
    #[cfg(test)]
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    /// Whether the texel at column `x`, row `y` is ink; `false` outside the image.
    #[cfg(test)]
    pub(crate) fn ink(&self, x: usize, y: usize) -> bool {
        x < self.width && self.texels.get(y * self.width + x).is_some_and(|&t| t != 0)
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    #[test]
    fn text_is_laid_out_in_cells_inside_the_padding() {
        let image = TextLayout::new(&["AB".to_owned(), "i".to_owned()]);
        // Two glyphs and a gap across, two lines and a gap down, padding all round.
        assert_eq!(image.width(), 2 * PADDING + 2 * font::WIDTH + GLYPH_GAP);
        assert_eq!(image.height(), 2 * PADDING + 2 * font::HEIGHT + LINE_GAP);
        // 'A' starts with a blank pixel then ink on its top row; 'B' with ink.
        assert!(!image.ink(PADDING, PADDING));
        assert!(image.ink(PADDING + 1, PADDING));
        assert!(image.ink(PADDING + font::WIDTH + GLYPH_GAP, PADDING), "B");
        // The gap column between them is blank all the way down.
        let gap = PADDING + font::WIDTH;
        let column = (0..image.height()).map(|y| image.ink(gap, y));
        assert!(!column.into_iter().any(|ink| ink), "the gap between glyphs");
        // 'i's dot sits on the second line's top row, in its middle column.
        let second = PADDING + font::HEIGHT + LINE_GAP;
        assert!(image.ink(PADDING + 2, second));
        // Nothing in the padding.
        let border =
            (0..image.width()).map(|x| image.ink(x, 0) || image.ink(x, image.height() - 1));
        assert!(!border.into_iter().any(|ink| ink), "the padding is backing");
    }

    #[test]
    fn no_lines_make_an_image_of_padding_only() {
        let image = TextLayout::new(&[]);
        assert_eq!((image.width(), image.height()), (2 * PADDING, 2 * PADDING));
    }

    #[test]
    fn the_overlay_reports_every_field() {
        let mut jobs = JobsReport::default();
        jobs.left = 1234;
        let snapshot = Snapshot {
            frame: 7,
            refresh_hz: 59.94,
            presentation: Presentation::Mailbox,
            adapter: "llvmpipe (Vulkan)",
            size: Size::new(640, 480).expect("non-zero"),
            owner: OwnerPhases::default(),
            render: Some(DrawTimes {
                acquire: Duration::from_micros(1500),
                ..DrawTimes::default()
            }),
            feed: FeedStats {
                epochs: 3,
                keys: 45,
                resyncs: 1,
            },
            jobs,
            tiles: None,
            scene: None,
            camera: Camera::default(),
            look: Look {
                view: SceneView::Shaded,
                shading: render::shade::ShadeSettings { wireframe: true },
                material: render::material::MatCapStyle::Glossy,
            },
            pick: None,
            pen: &PenStatus {
                source: crate::pen::PenSource::Fallback {
                    why: "not a Wayland display or window".into(),
                },
                dropped: 0,
                lost: 0,
            },
            filter_only: false,
            brush: &["brush      draw".to_owned()],
            save: "project    sphere",
            latency: Some(Duration::from_micros(6250)),
            pacing: StrokePacing::OnInput,
        };
        let text = lines(&snapshot).join("\n");
        for expected in [
            "frame 7",
            "640x480",
            "mailbox at 59.9 Hz",
            "llvmpipe (Vulkan)",
            "acquire 1.50",
            "3 epochs  45 tiles  1 resyncs",
            "1234 display tiles left",
            "none on the GPU yet",
            "no cut read back yet",
            "eye (0.00, 0.00, 3.00)  pivot (0.00, 0.00, 0.00) at 3.000  not magnified",
            "matcap glossy  wireframe on",
            "pick       off (no spatial index)",
            "pen        winit's tablet events (fallback: not a Wayland display or window)",
            "brush      draw",
            "latency    6.25 ms pen sample to present",
            "(strokes start frames on input)",
        ] {
            assert!(text.contains(expected), "{expected:?} in:\n{text}");
        }
        assert!(text.is_ascii(), "the font draws ASCII");
        let without_render = Snapshot {
            render: None,
            tiles: Some("12 tiles at level 5"),
            ..snapshot
        };
        let rest = lines(&without_render).join("\n");
        assert!(rest.contains("render ms  -"), "{rest}");
        assert!(rest.contains("12 tiles at level 5"), "{rest}");
        let mut close_up = Camera::default();
        close_up.zoom(1e-5);
        let magnified = Snapshot {
            camera: close_up,
            ..snapshot
        };
        let shown = lines(&magnified).join("\n");
        assert!(
            shown.contains("magnified 3.33x (eye at the boundary)"),
            "{shown}"
        );
        let no_latency = Snapshot {
            latency: None,
            pacing: StrokePacing::Refresh,
            ..snapshot
        };
        let none_yet = lines(&no_latency).join("\n");
        assert!(
            none_yet.contains("latency    - (no dabs applied yet)"),
            "{none_yet}"
        );
        assert!(
            none_yet.contains("strokes paced to the refresh (no mailbox: fifo)"),
            "the fallback said: {none_yet}"
        );
    }
}
