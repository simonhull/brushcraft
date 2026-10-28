//! The present mode chosen, frame sizes, and an offscreen target drawn and read back on
//! lavapipe.

use wgpu::{
    Color, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp,
};

use super::*;
use crate::gpu::GpuTest;

#[test]
fn mailbox_is_chosen_where_offered_else_fifo() {
    assert_eq!(
        Presentation::for_modes(&[PresentMode::Fifo, PresentMode::Mailbox]),
        Presentation::Mailbox,
        "offered, so taken"
    );
    assert_eq!(
        Presentation::for_modes(&[PresentMode::Fifo, PresentMode::Immediate]),
        Presentation::Fifo,
        "immediate tears: never taken"
    );
    assert_eq!(
        Presentation::for_modes(&[]),
        Presentation::Fifo,
        "every surface offers FIFO"
    );
    assert_eq!(
        Presentation::Mailbox.present_mode(),
        Some(PresentMode::Mailbox)
    );
    assert_eq!(Presentation::Fifo.present_mode(), Some(PresentMode::Fifo));
    assert_eq!(Presentation::Offscreen.present_mode(), None);
    assert_eq!(Presentation::Mailbox.to_string(), "mailbox");
}

#[test]
fn a_size_is_never_zero() {
    assert_eq!(Size::new(0, 10), None);
    assert_eq!(Size::new(10, 0), None);
    let size = Size::new(640, 480).expect("non-zero");
    assert_eq!((size.width(), size.height()), (640, 480));
    assert_eq!(size.to_string(), "640 × 480");
}

#[test]
fn an_srgb_format_is_preferred() {
    assert_eq!(
        preferred_format(&[TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb]),
        Some(TextureFormat::Bgra8UnormSrgb)
    );
    assert_eq!(
        preferred_format(&[TextureFormat::Rgb10a2Unorm]),
        Some(TextureFormat::Rgb10a2Unorm),
        "else the first"
    );
    assert_eq!(preferred_format(&[]), None);
}

/// Clears `target`'s next frame to `color` and presents it.
fn clear(gpu: &Gpu, target: &mut Target, color: Color) {
    let Acquired::Frame(frame) = target.acquire(gpu).expect("acquired") else {
        panic!("an offscreen target always has a frame");
    };
    let mut encoder = gpu
        .device()
        .create_command_encoder(&CommandEncoderDescriptor::default());
    drop(encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: frame.view(),
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(color),
                store: StoreOp::Store,
            },
        })],
        ..RenderPassDescriptor::default()
    }));
    gpu.queue().submit([encoder.finish()]);
    frame.present(gpu);
}

#[test]
fn an_offscreen_frame_reads_back_what_was_drawn() {
    GpuTest::run(|gpu| {
        // 100 pixels across: 400 bytes a row, padded to 512 for the copy, so the padding is
        // dropped on the way back.
        let size = Size::new(100, 3).expect("non-zero");
        let mut target = Target::Offscreen(Offscreen::new(gpu, size).expect("target"));
        assert_eq!(target.presentation(), Presentation::Offscreen);
        assert_eq!(target.format(), Offscreen::FORMAT);
        clear(gpu, &mut target, Color::RED);
        let Target::Offscreen(offscreen) = &target else {
            unreachable!("made offscreen");
        };
        let pixels = offscreen.read_pixels(gpu).expect("read back");
        assert_eq!(pixels.len(), 300, "width × height, no padding");
        assert!(
            pixels.iter().all(|&p| p == [255, 0, 0, 255]),
            "every pixel red"
        );
    });
}

#[test]
fn resizing_an_offscreen_target_replaces_its_texture() {
    GpuTest::run(|gpu| {
        let small = Size::new(4, 4).expect("non-zero");
        let large = Size::new(8, 2).expect("non-zero");
        let mut target = Target::Offscreen(Offscreen::new(gpu, small).expect("target"));
        target.resize(gpu, large).expect("resized");
        assert_eq!(target.size(), large);
        clear(gpu, &mut target, Color::BLUE);
        let Target::Offscreen(offscreen) = &target else {
            unreachable!("made offscreen");
        };
        assert_eq!(offscreen.read_pixels(gpu).expect("read back").len(), 16);
    });
}
