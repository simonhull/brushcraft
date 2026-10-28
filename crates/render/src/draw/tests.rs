//! A frame drawn offscreen on lavapipe and read back: the clear colour, with the overlay over
//! its corner.

use std::convert::Infallible;

use geom::{Level, TileKey, cube_sphere};
use wgpu::PollType;

use super::*;
use crate::gpu::GpuTest;
use crate::overlay::MARGIN;
use crate::residency::{Residency, ResidencyConfig};
use crate::surface::Offscreen;
use crate::upload::{GpuVertex, TileSource};

/// The clear colour the tests draw with, linear.
const CLEAR: Color = Color {
    r: 0.2,
    g: 0.1,
    b: 0.05,
    a: 1.0,
};

/// `value` (linear, 0 to 1) as an 8-bit sRGB code, as the target stores it.
fn srgb(value: f64) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded from 0..=255"
    )]
    let code = (encoded * 255.0).round() as u8;
    code
}

/// Whether `a` and `b` agree to within one code in every channel.
fn near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 1)
}

/// Draws one frame of `size` with `overlay` at `scale` offscreen; returns the drawn frame's
/// report and its pixels.
fn draw(gpu: &Gpu, size: Size, overlay: Option<&TextImage>, scale: u32) -> (Drawn, Vec<[u8; 4]>) {
    let mut target = Target::Offscreen(Offscreen::new(gpu, size).expect("target"));
    let mut renderer = Renderer::new(gpu, target.format()).expect("renderer");
    let mut called = false;
    let inputs = FrameInputs {
        size,
        clear: CLEAR,
        overlay,
        overlay_scale: scale,
    };
    let drawn = renderer
        .draw(gpu, &mut target, &inputs, None, || called = true)
        .expect("drawn");
    assert!(called, "before_present ran");
    let Target::Offscreen(offscreen) = &target else {
        unreachable!("made offscreen");
    };
    (drawn, offscreen.read_pixels(gpu).expect("read back"))
}

#[test]
fn the_overlay_is_drawn_last_over_the_clear_colour() {
    GpuTest::run(|gpu| {
        // A 4 × 2 image: ink in its top-left texel only.
        let mut texels = vec![0; 8];
        texels[0] = u8::MAX;
        let image = TextImage::new(4, 2, texels).expect("4 × 2 texels");
        let size = Size::new(64, 32).expect("non-zero");
        let (drawn, pixels) = draw(gpu, size, Some(&image), 2);
        assert_eq!(drawn.outcome, Outcome::Presented);
        let at = |x: u32, y: u32| pixels[(y * 64 + x) as usize];
        let clear = [srgb(CLEAR.r), srgb(CLEAR.g), srgb(CLEAR.b), 255];
        assert!(near(at(60, 30), clear), "outside: {:?}", at(60, 30));
        // At scale 2 the overlay starts 16 pixels in and each texel covers 2 × 2 pixels.
        let origin = MARGIN * 2;
        let ink = at(origin, origin);
        assert!(ink[0] > 200 && ink[1] > 200, "ink is bright: {ink:?}");
        let backing = at(origin + 2, origin);
        assert!(
            backing[0] < clear[0],
            "backing {backing:?} darkens {clear:?}"
        );
        assert!(
            near(at(origin + 8, origin), clear),
            "the image ends 8 pixels on"
        );
        assert!(near(at(origin, origin + 4), clear), "and 4 down");
    });
}

#[test]
fn with_no_overlay_the_frame_is_the_clear_colour() {
    GpuTest::run(|gpu| {
        let size = Size::new(32, 32).expect("non-zero");
        let (_, pixels) = draw(gpu, size, None, 1);
        let clear = [srgb(CLEAR.r), srgb(CLEAR.g), srgb(CLEAR.b), 255];
        assert!(pixels.iter().all(|&p| near(p, clear)), "every pixel");
    });
}

/// A source whose every tile's vertices are at `(level, 0, 0)`, so a slot read back shows
/// which tile it holds.
struct ByLevel;

impl TileSource for ByLevel {
    type Error = Infallible;

    fn read(&mut self, key: TileKey, out: &mut [GpuVertex]) -> Result<Option<f32>, Infallible> {
        out.fill(Self::vertex(key));
        Ok(Some(0.0))
    }
}

impl ByLevel {
    /// Every vertex it gives `key`.
    fn vertex(key: TileKey) -> GpuVertex {
        GpuVertex {
            position: [f32::from(key.level().get()), 0.0, 0.0],
            roughness: 1.0,
            normal: [0.0, 0.0, 1.0],
            reserved: 0,
        }
    }
}

/// Residency for a one-split cube sphere whose finest level is 7, and the first frame's
/// uploads: its whole floor.
fn floor_uploads(gpu: &Gpu) -> (Residency, FrameUploads) {
    let mesh = cube_sphere(1).expect("a mesh");
    let finest = Level::new(7).expect("a level");
    let mut residency =
        Residency::new(gpu, &mesh, finest, ResidencyConfig::default()).expect("residency");
    residency.begin_frame(gpu).expect("a frame begins");
    let deadline = Instant::now() + Duration::from_secs(60);
    let uploads = residency
        .stage(gpu, &mut ByLevel, deadline)
        .expect("the floor is staged");
    (residency, uploads)
}

/// Whether every floor tile's slot holds what [`ByLevel`] gave it, read back from the GPU, and
/// no frame was lost.
fn floor_arrived(gpu: &Gpu, residency: &mut Residency) -> bool {
    gpu.device()
        .poll(PollType::wait_indefinitely())
        .expect("the GPU finishes");
    residency.begin_frame(gpu).expect("the next frame begins");
    if residency.stats().lost_frames != 0 {
        return false;
    }
    residency.floor().iter().all(|key| {
        let Some(slot) = residency.slot(key) else {
            return false;
        };
        let buffer = &residency.slot_buffers(slot.class())[usize::from(slot.buffer())];
        let vertices: Vec<GpuVertex> = gpu.read_buffer(buffer).expect("read back");
        let first = (slot.index() * slot.class().vertices()) as usize;
        vertices[first] == ByLevel::vertex(*key)
    })
}

#[test]
fn a_frames_uploads_go_in_its_submission_before_the_passes() {
    GpuTest::run(|gpu| {
        let (mut residency, uploads) = floor_uploads(gpu);
        assert!(!uploads.is_empty(), "the floor is staged");
        let size = Size::new(16, 16).expect("non-zero");
        let mut target = Target::Offscreen(Offscreen::new(gpu, size).expect("target"));
        let mut renderer = Renderer::new(gpu, target.format()).expect("renderer");
        let inputs = FrameInputs {
            size,
            clear: CLEAR,
            overlay: None,
            overlay_scale: 1,
        };
        let drawn = renderer
            .draw(gpu, &mut target, &inputs, Some(uploads), || {})
            .expect("drawn");
        assert_eq!(drawn.outcome, Outcome::Presented);
        assert!(
            floor_arrived(gpu, &mut residency),
            "the floor is on the GPU"
        );
    });
}

#[test]
fn a_frame_with_no_image_still_submits_its_uploads() {
    GpuTest::run(|gpu| {
        let (mut residency, uploads) = floor_uploads(gpu);
        let times = DrawTimes::default();
        let drawn = not_drawn(gpu, Some(uploads), Outcome::Retry, times).expect("submitted");
        assert_eq!(drawn.outcome, Outcome::Retry);
        assert!(
            floor_arrived(gpu, &mut residency),
            "the tiles arrive without an image to draw them in"
        );
    });
}

#[test]
fn a_text_image_holds_exactly_its_texels() {
    assert!(TextImage::new(3, 2, vec![0; 5]).is_none());
    let image = TextImage::new(3, 2, vec![0, 1, 2, 3, 4, 5]).expect("six texels");
    assert_eq!((image.width(), image.height()), (3, 2));
    assert_eq!(image.texel(2, 1), Some(5));
    assert_eq!(image.texel(3, 0), None, "outside");
}
