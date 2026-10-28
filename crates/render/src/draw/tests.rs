//! A frame drawn offscreen on lavapipe and read back: the clear colour, with the overlay over
//! its corner.

use super::*;
use crate::gpu::GpuTest;
use crate::overlay::MARGIN;
use crate::surface::Offscreen;

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
        .draw(gpu, &mut target, &inputs, || called = true)
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

#[test]
fn a_text_image_holds_exactly_its_texels() {
    assert!(TextImage::new(3, 2, vec![0; 5]).is_none());
    let image = TextImage::new(3, 2, vec![0, 1, 2, 3, 4, 5]).expect("six texels");
    assert_eq!((image.width(), image.height()), (3, 2));
    assert_eq!(image.texel(2, 1), Some(5));
    assert_eq!(image.texel(3, 0), None, "outside");
}
