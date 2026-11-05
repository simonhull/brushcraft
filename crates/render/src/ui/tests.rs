//! The UI pass on lavapipe: what it draws read back from the frame, and a golden of a bar and chip.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::ui::{
    AtlasImage, Bar, Blend, Chip, Colour, Control, DrawList, Emphasis, Fade, GlyphQuad, Layer,
    Mark, Rect, RoundedRect, Theme, Ui, UiFrame,
};
use wgpu::Color;

use crate::Renderer;
use crate::draw::{FrameInputs, Look, Scene};
use crate::gpu::{Gpu, GpuTest};
use crate::raster::tests::finish;
use crate::shade::tests::flip::flip;
use crate::shade::tests::golden::{GOLDEN_MEAN, GOLDEN_P99, WRITE, golden_dir, linear_of};
use crate::shade::tests::png::{self, Rgb};
use crate::surface::{Offscreen, Size, Target};
use crate::timing::GpuPhase;

/// The brightest clay the MatCaps show, in display colour.
const CLAY: Colour = Colour::rgb(0xD6, 0xD6, 0xD2);

/// A mid grey, in display colour.
const GREY: Colour = Colour::rgb(0x70, 0x70, 0x70);

/// The theme under test.
const THEME: Theme = Theme::GLASS;

/// The golden's size: a bar's least width in a small window, and the chip above it.
const GOLDEN_SIZE: (u32, u32) = (440, 160);

/// A size, which is never zero here.
fn size(width: u32, height: u32) -> Size {
    Size::new(width, height).expect("non-zero")
}

/// `colour` as a frame's clear colour, which is linear.
fn clear_of(colour: Colour) -> Color {
    let decode = |channel: u8| {
        let value = f64::from(channel) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    Color {
        r: decode(colour.r),
        g: decode(colour.g),
        b: decode(colour.b),
        a: 1.0,
    }
}

/// A frame of `list` at full opacity, with `atlas`.
fn frame_of(list: DrawList, atlas: Arc<AtlasImage>, fades: (f32, f32)) -> UiFrame {
    let now = Instant::now();
    UiFrame {
        list: Arc::new(list),
        atlas,
        fades: [Fade::at_rest(fades.0, now), Fade::at_rest(fades.1, now)],
    }
}

/// An atlas with nothing in it.
fn blank_atlas() -> Arc<AtlasImage> {
    Arc::new(AtlasImage {
        size: 4,
        texels: vec![0; 16],
        version: 1,
    })
}

/// A rectangle with no hairline on `layer`.
const fn flat(rect: Rect, fill: Colour, layer: Layer) -> RoundedRect {
    RoundedRect {
        rect,
        radius: 0.0,
        fill,
        stroke: Colour::CLEAR,
        stroke_width: 0.0,
        layer,
    }
}

/// The rectangle at `x`, `y`.
const fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect { x, y, w, h }
}

/// A renderer and a target of `extent`.
fn drawing(gpu: &Gpu, target: Offscreen) -> (Renderer, Target) {
    let target = Target::Offscreen(target);
    let renderer = Renderer::new(gpu, target.format()).expect("renderer");
    (renderer, target)
}

/// Draws one frame into `target` over `ground`; the pixels, as stored.
fn draw(
    gpu: &Gpu,
    renderer: &mut Renderer,
    target: &mut Target,
    ground: Colour,
    ui: Option<&UiFrame>,
) -> Vec<[u8; 4]> {
    let extent = target.size();
    let inputs = FrameInputs {
        size: extent,
        clear: clear_of(ground),
        overlay: None,
        overlay_scale: 1,
        ui,
        scene: Scene::None,
        look: Look::default(),
        frame: 1,
    };
    renderer
        .draw(gpu, target, &inputs, None, || {})
        .expect("drawn");
    finish(gpu);
    let Target::Offscreen(offscreen) = &*target else {
        unreachable!("made offscreen");
    };
    offscreen.read_pixels(gpu).expect("read back")
}

/// Draws `ui` over `ground` on a fresh offscreen target of `extent`; the pixels.
fn drawn(gpu: &Gpu, extent: Size, ground: Colour, ui: &UiFrame) -> Vec<[u8; 4]> {
    let offscreen = Offscreen::new(gpu, extent).expect("target");
    let (mut renderer, mut target) = drawing(gpu, offscreen);
    draw(gpu, &mut renderer, &mut target, ground, Some(ui))
}

/// Whether `pixel` is `colour`, within `tolerance` in each of red, green and blue.
fn close(pixel: [u8; 4], colour: Colour, tolerance: u8) -> bool {
    let wanted = [colour.r, colour.g, colour.b];
    let pairs = pixel.iter().zip(wanted);
    pairs
        .into_iter()
        .all(|(&got, want)| got.abs_diff(want) <= tolerance)
}

/// The pixel at `x`, `y` of an image `width` across.
fn at(pixels: &[[u8; 4]], width: u32, x: u32, y: u32) -> [u8; 4] {
    pixels[(y * width + x) as usize]
}

/// How far `pixel` lies from the straight line between `from` and `to`, in codes. Blending one
/// colour over another by any coverage stays on that line.
fn off_the_line(pixel: [u8; 4], from: Colour, to: Colour) -> f32 {
    let start = [from.r, from.g, from.b].map(f32::from);
    let end = [to.r, to.g, to.b].map(f32::from);
    let got = [pixel[0], pixel[1], pixel[2]].map(f32::from);
    let along: [f32; 3] = std::array::from_fn(|k| end[k] - start[k]);
    let squared: f32 = along.iter().map(|v| v * v).sum();
    let dot: f32 = (0..3).map(|k| (got[k] - start[k]) * along[k]).sum();
    let t = (dot / squared).clamp(0.0, 1.0);
    let off = (0..3).map(|k| (got[k] - (start[k] + t * along[k])).abs());
    off.fold(0.0, f32::max)
}

#[test]
fn an_empty_list_encodes_no_pass() {
    GpuTest::run(|gpu| {
        let extent = size(16, 16);
        let empty = frame_of(DrawList::default(), blank_atlas(), (1.0, 1.0));
        let offscreen = Offscreen::new(gpu, extent).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        let without = draw(gpu, &mut renderer, &mut target, GREY, None);
        let with = draw(gpu, &mut renderer, &mut target, GREY, Some(&empty));
        assert_eq!(without, with, "the same frame");
        assert_eq!(renderer.ui_counts().drawn, 0, "no pass");

        // The bite: a list with something in it does.
        let list = DrawList {
            rects: vec![flat(rect(0.0, 0.0, 4.0, 4.0), CLAY, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let some = frame_of(list, blank_atlas(), (1.0, 1.0));
        draw(gpu, &mut renderer, &mut target, GREY, Some(&some));
        assert_eq!(renderer.ui_counts().drawn, 1, "a pass");
    });
}

/// The pass has its own stamp, and a frame with nothing to show takes none: a stamp claimed would
/// read as a phase run, written or not.
#[test]
fn the_pass_is_timed_by_its_own_stamp_and_only_when_it_draws() {
    GpuTest::run(|gpu| {
        let offscreen = Offscreen::new(gpu, size(16, 16)).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        assert!(
            renderer.enable_timing(gpu).expect("timing"),
            "lavapipe has it"
        );
        let mut timed = |ui: Option<&UiFrame>| {
            draw(gpu, &mut renderer, &mut target, GREY, ui);
            renderer.flush_timings(gpu).expect("flushed");
            finish(gpu);
            let times = renderer.take_timings();
            *times.last().expect("the frame's times")
        };
        let list = DrawList {
            rects: vec![flat(rect(0.0, 0.0, 4.0, 4.0), CLAY, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let some = frame_of(list, blank_atlas(), (1.0, 1.0));
        let empty = frame_of(DrawList::default(), blank_atlas(), (1.0, 1.0));

        assert_eq!(timed(None).phase(GpuPhase::Ui), None, "no UI");
        assert_eq!(
            timed(Some(&empty)).phase(GpuPhase::Ui),
            None,
            "nothing in it"
        );
        let shown = timed(Some(&some));
        let took = shown.phase(GpuPhase::Ui).expect("its stamp");
        assert!(took > Duration::ZERO, "{shown:?}");
    });
}

/// A failure bar with everything a bar can draw, its quiet control under the pointer, and a chip
/// with a ring, over a mid grey with a clay patch. The rects of a frame from a real [`Ui`].
fn golden_frame() -> UiFrame {
    let now = Instant::now();
    let mut ui: Ui<u8> = Ui::new(THEME, Blend::Display, GOLDEN_SIZE, 1.0, now);
    let control = |label: &str, action, emphasis| Control {
        label: label.to_owned(),
        action,
        emphasis,
    };
    let controls = vec![
        control("Dismiss", 1, Emphasis::Quiet),
        control("Try again", 2, Emphasis::Filled),
    ];
    let bar = Bar::new("Couldn't save: disk full", controls)
        .with_detail("Your work is safe.")
        .with_progress(0.6)
        .needing_you();
    ui.set_bar(Some(bar), now);
    let chip = Chip {
        text: "Saving".to_owned(),
        mark: Mark::Ring,
    };
    ui.set_chip(Some(chip), now);
    let seen = now + THEME.fade;
    let points = (0..440_u16).flat_map(|x| (0..160_u16).map(move |y| (x, y)));
    let mut points = points.map(|(x, y)| (f32::from(x), f32::from(y)));
    let pointer = points.find(|&(x, y)| ui.hit(x, y, seen) == Some(1));
    assert!(pointer.is_some(), "Dismiss is on the bar");
    ui.hover(pointer);
    let frame = ui.frame(now).expect("both show");

    let mut list = (*frame.list).clone();
    let patch = rect(220.0, 0.0, 220.0, 160.0);
    list.rects.insert(0, flat(patch, CLAY, Layer::Bar));
    frame_of(list, frame.atlas, (1.0, 1.0))
}

/// `pixels` of the golden's size as an image.
fn image_of(pixels: &[[u8; 4]]) -> Rgb {
    Rgb {
        width: GOLDEN_SIZE.0,
        height: GOLDEN_SIZE.1,
        pixels: pixels.iter().map(|p| [p[0], p[1], p[2]]).collect(),
    }
}

/// Where the golden lives.
fn golden_path() -> PathBuf {
    golden_dir().join("ui-bar-and-chip.png")
}

#[test]
fn a_bar_and_chip_match_their_stored_image() {
    GpuTest::run(|gpu| {
        let extent = size(GOLDEN_SIZE.0, GOLDEN_SIZE.1);
        let frame = golden_frame();
        let image = image_of(&drawn(gpu, extent, GREY, &frame));
        let path = golden_path();
        if std::env::var_os(WRITE).is_some() {
            std::fs::create_dir_all(golden_dir()).expect("the golden directory");
            std::fs::write(&path, png::encode(&image)).expect("written");
        }
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("{}: {error}; write it with {WRITE}=1", path.display()));
        let golden = png::decode(&bytes).expect("a golden PNG");
        let errors = flip(&linear_of(&golden), &linear_of(&image));
        let (mean, p99) = (errors.mean(), errors.percentile(99));
        assert!(
            mean <= GOLDEN_MEAN && p99 <= GOLDEN_P99,
            "FLIP mean {mean}, p99 {p99}"
        );

        // The bite: the same frame with its chip and bar at half opacity is refused.
        let faded = UiFrame {
            fades: [Fade::at_rest(0.5, Instant::now()); 2],
            ..frame
        };
        let dim = image_of(&drawn(gpu, extent, GREY, &faded));
        let refused = flip(&linear_of(&golden), &linear_of(&dim));
        assert!(
            refused.percentile(99) > GOLDEN_P99,
            "{}",
            refused.percentile(99)
        );
    });
}

#[test]
fn glass_blends_in_display_colour() {
    GpuTest::run(|gpu| {
        let list = DrawList {
            rects: vec![flat(rect(0.0, 0.0, 8.0, 8.0), THEME.glass, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let frame = frame_of(list, blank_atlas(), (1.0, 1.0));
        let offscreen = Offscreen::new(gpu, size(8, 8)).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        assert_eq!(Renderer::ui_blend(&target), Blend::Display, "it says so");
        let pixels = draw(gpu, &mut renderer, &mut target, CLAY, Some(&frame));
        let wanted = THEME.glass.over(CLAY);
        assert!(
            close(pixels[27], wanted, 1),
            "{:?}, not {wanted:?}",
            pixels[27]
        );
    });
}

#[test]
fn a_surface_with_no_display_view_draws_on_the_srgb_view_and_text_still_reads() {
    GpuTest::run(|gpu| {
        let extent = size(8, 8);
        let offscreen = Offscreen::without_display_view(gpu, extent).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        let blend = Renderer::ui_blend(&target);
        assert_eq!(blend, Blend::Linear, "it says so");
        let theme = THEME.for_blend(blend);
        let list = DrawList {
            rects: vec![flat(rect(0.0, 0.0, 8.0, 8.0), theme.glass, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let frame = frame_of(list, blank_atlas(), (1.0, 1.0));
        let pixels = draw(gpu, &mut renderer, &mut target, CLAY, Some(&frame));

        let [r, g, b, _] = pixels[27];
        let ground = Colour::rgb(r, g, b);
        assert!(
            THEME.text.contrast(ground) >= 4.5 && THEME.quiet_text.contrast(ground) >= 4.5,
            "text on {ground:?}"
        );
        let wanted = theme.glass.over_in(CLAY, blend);
        assert!(close(pixels[27], wanted, 1), "{ground:?}, not {wanted:?}");
    });
}

#[test]
fn fades_scale_a_layer_and_leave_the_other() {
    GpuTest::run(|gpu| {
        let white = Colour::rgb(255, 255, 255);
        let list = DrawList {
            rects: vec![
                flat(rect(0.0, 0.0, 4.0, 8.0), white, Layer::Bar),
                flat(rect(4.0, 0.0, 4.0, 8.0), white, Layer::Chip),
            ],
            glyphs: Vec::new(),
        };
        let black = Colour::rgb(0, 0, 0);
        let frame = frame_of(list, blank_atlas(), (0.5, 1.0));
        let pixels = drawn(gpu, size(8, 8), black, &frame);
        let half = Colour::rgb(128, 128, 128);
        assert!(
            close(at(&pixels, 8, 1, 4), half, 1),
            "{:?}",
            at(&pixels, 8, 1, 4)
        );
        assert!(
            close(at(&pixels, 8, 6, 4), white, 0),
            "{:?}",
            at(&pixels, 8, 6, 4)
        );
    });
}

/// A glass rectangle with a theme hairline `width` wide, from (8, 8), 32 across.
fn hairlined(width: f32) -> UiFrame {
    let list = DrawList {
        rects: vec![RoundedRect {
            rect: rect(8.0, 8.0, 32.0, 32.0),
            radius: 6.0,
            fill: THEME.glass,
            stroke: THEME.hairline,
            stroke_width: width,
            layer: Layer::Bar,
        }],
        glyphs: Vec::new(),
    };
    frame_of(list, blank_atlas(), (1.0, 1.0))
}

#[test]
fn a_hairline_lies_inside_the_edge_as_wide_as_it_says_over_the_fill_once() {
    GpuTest::run(|gpu| {
        let glass = THEME.glass.over(CLAY);
        let line = THEME.hairline.over(glass);
        for width in [1_u16, 2] {
            let frame = hairlined(f32::from(width));
            let width = u32::from(width);
            let pixels = drawn(gpu, size(48, 48), CLAY, &frame);
            let pixel = |x: u32| at(&pixels, 48, x, 24);
            assert!(close(pixel(7), CLAY, 0), "outside: {:?}", pixel(7));
            for x in 8..8 + width {
                assert!(
                    close(pixel(x), line, 1),
                    "{x} of {width}: {:?}, not {line:?}",
                    pixel(x)
                );
            }
            let inner = 8 + width;
            assert!(close(pixel(inner), glass, 1), "{inner}: {:?}", pixel(inner));
            assert!(close(pixel(24), glass, 1), "the middle: {:?}", pixel(24));
            let far = 39 - width + 1;
            assert!(close(pixel(far), line, 1), "the far side: {:?}", pixel(far));
            assert!(close(pixel(40), CLAY, 0), "past it: {:?}", pixel(40));
        }
    });
}

#[test]
fn a_rounded_edge_fades_to_the_ground_with_no_dark_fringe() {
    GpuTest::run(|gpu| {
        // An open ring, as the chip's: its edge pixels must be accent over ground, partly.
        let ring = RoundedRect {
            rect: rect(4.0, 4.0, 16.0, 16.0),
            radius: 8.0,
            fill: Colour::CLEAR,
            stroke: THEME.accent,
            stroke_width: 1.5,
            layer: Layer::Bar,
        };
        let list = DrawList {
            rects: vec![ring],
            glyphs: Vec::new(),
        };
        let frame = frame_of(list, blank_atlas(), (1.0, 1.0));
        let pixels = drawn(gpu, size(24, 24), CLAY, &frame);
        let worst = pixels
            .iter()
            .map(|&pixel| off_the_line(pixel, CLAY, THEME.accent))
            .fold(0.0, f32::max);
        assert!(worst <= 2.0, "{worst} codes off clay-to-accent");
        let touched = pixels
            .iter()
            .filter(|&&pixel| !close(pixel, CLAY, 2))
            .count();
        assert!(touched > 40, "the ring drew: {touched}");
        assert!(close(at(&pixels, 24, 12, 12), CLAY, 0), "it's open");
    });
}

#[test]
fn a_corner_of_glass_fades_to_the_ground_with_no_dark_fringe() {
    GpuTest::run(|gpu| {
        let frame = hairlined(0.0);
        let pixels = drawn(gpu, size(48, 48), CLAY, &frame);
        let glass = THEME.glass.over(CLAY);
        let worst = pixels
            .iter()
            .map(|&pixel| off_the_line(pixel, CLAY, glass))
            .fold(0.0, f32::max);
        assert!(worst <= 2.0, "{worst} codes off clay-to-glass");
    });
}

#[test]
fn a_glyph_reads_its_texels_one_to_a_pixel() {
    GpuTest::run(|gpu| {
        let mut texels = vec![0; 16];
        texels[5] = 255;
        texels[6] = 128;
        let atlas = Arc::new(AtlasImage {
            size: 4,
            texels,
            version: 1,
        });
        let white = Colour::rgb(255, 255, 255);
        let list = DrawList {
            rects: Vec::new(),
            glyphs: vec![GlyphQuad {
                rect: rect(2.0, 3.0, 2.0, 1.0),
                atlas: (1, 1),
                colour: white,
                layer: Layer::Chip,
            }],
        };
        let frame = frame_of(list, atlas, (1.0, 1.0));
        let black = Colour::rgb(0, 0, 0);
        let pixels = drawn(gpu, size(8, 8), black, &frame);
        assert!(close(at(&pixels, 8, 2, 3), white, 0), "full coverage");
        let half = Colour::rgb(128, 128, 128);
        assert!(close(at(&pixels, 8, 3, 3), half, 1), "half coverage");
        assert!(close(at(&pixels, 8, 4, 3), black, 0), "beside it");
    });
}

#[test]
fn only_what_changed_is_uploaded() {
    GpuTest::run(|gpu| {
        let offscreen = Offscreen::new(gpu, size(8, 8)).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        let black = Colour::rgb(0, 0, 0);
        let list = |colour| DrawList {
            rects: vec![flat(rect(0.0, 0.0, 8.0, 8.0), colour, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let first = frame_of(list(CLAY), blank_atlas(), (1.0, 1.0));
        draw(gpu, &mut renderer, &mut target, black, Some(&first));
        draw(gpu, &mut renderer, &mut target, black, Some(&first));
        let first_counts = renderer.ui_counts();
        assert_eq!(
            (first_counts.lists, first_counts.atlases),
            (1, 1),
            "one of each"
        );

        let second = UiFrame {
            list: Arc::new(list(GREY)),
            ..first.clone()
        };
        let pixels = draw(gpu, &mut renderer, &mut target, black, Some(&second));
        let counts = renderer.ui_counts();
        assert_eq!((counts.lists, counts.atlases), (2, 1), "the list alone");
        assert!(close(pixels[0], GREY, 0), "and it draws: {:?}", pixels[0]);

        let bigger = Arc::new(AtlasImage {
            size: 8,
            texels: vec![0; 64],
            version: 2,
        });
        let third = UiFrame {
            atlas: bigger,
            ..second
        };
        draw(gpu, &mut renderer, &mut target, black, Some(&third));
        assert_eq!(renderer.ui_counts().atlases, 2, "the atlas alone");
    });
}

#[test]
fn a_list_too_big_for_its_buffer_grows_it() {
    GpuTest::run(|gpu| {
        let offscreen = Offscreen::new(gpu, size(64, 64)).expect("target");
        let (mut renderer, mut target) = drawing(gpu, offscreen);
        let black = Colour::rgb(0, 0, 0);
        let small = DrawList {
            rects: vec![flat(rect(0.0, 0.0, 1.0, 1.0), CLAY, Layer::Bar)],
            glyphs: Vec::new(),
        };
        let first = frame_of(small, blank_atlas(), (1.0, 1.0));
        draw(gpu, &mut renderer, &mut target, black, Some(&first));

        let cells = (0..4096_u16).map(|cell| {
            let (x, y) = (f32::from(cell % 64), f32::from(cell / 64));
            flat(rect(x, y, 1.0, 1.0), CLAY, Layer::Bar)
        });
        let big = DrawList {
            rects: cells.collect(),
            glyphs: Vec::new(),
        };
        let second = frame_of(big, blank_atlas(), (1.0, 1.0));
        let pixels = draw(gpu, &mut renderer, &mut target, black, Some(&second));
        assert!(
            pixels.iter().all(|&pixel| close(pixel, CLAY, 0)),
            "every cell drew"
        );
    });
}
