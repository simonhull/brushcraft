//! Screenshots (`--screenshot FILE`): the last frame an offscreen run drew, written as a PNG.
//!
//! The offscreen target holds 8-bit sRGB-encoded RGBA ([`Offscreen::FORMAT`]), so its texels go
//! into the file as they are, row by row from the top, without their alpha (a frame is opaque),
//! and the file is marked sRGB so viewers show it as the window would. The readback
//! ([`Offscreen::read_pixels`]) maps a copy of the texture and drops the padding the copy gives
//! each row.
//!
//! A window's frames are the compositor's to capture, so `--screenshot` goes with
//! `--offscreen` ([`crate::cli`]).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context as _, Result};
use png::{BitDepth, ColorType, Encoder, EncodingError, SrgbRenderingIntent};
use render::gpu::Gpu;
use render::surface::{Offscreen, Size};
use wgpu::TextureFormat;

/// The offscreen target's texels are what [`encode`] writes as they are: 8-bit RGBA in that
/// order, sRGB-encoded. A change of format needs a conversion here.
const _: () = assert!(
    matches!(Offscreen::FORMAT, TextureFormat::Rgba8UnormSrgb),
    "screenshots write the offscreen texels as they are: 8-bit sRGB RGBA"
);

/// Reads back what `offscreen` holds (the last frame drawn into it, once the GPU has finished
/// everything submitted) and writes it to `path` as a PNG.
///
/// # Errors
///
/// If the readback fails, or the file can't be created or written.
pub(crate) fn save(offscreen: &Offscreen, gpu: &Gpu, path: &Path) -> Result<()> {
    let pixels = offscreen
        .read_pixels(gpu)
        .context("reading the last frame back")?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut out = BufWriter::new(file);
    encode(offscreen.size(), &pixels, &mut out)
        .with_context(|| format!("writing the screenshot to {}", path.display()))?;
    out.flush()
        .with_context(|| format!("writing the screenshot to {}", path.display()))
}

/// Writes `pixels` (`size`'s texels, sRGB-encoded RGBA, row by row from the top) to `out` as an
/// 8-bit sRGB PNG, their alpha dropped.
///
/// # Errors
///
/// If there aren't `size`'s texels, or writing to `out` fails.
fn encode(size: Size, pixels: &[[u8; 4]], out: impl Write) -> Result<(), EncodingError> {
    let mut encoder = Encoder::new(out, size.width(), size.height());
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_source_srgb(SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header()?;
    let colours = pixels.iter().flat_map(|&[r, g, b, _]| [r, g, b]);
    let rgb: Vec<u8> = colours.collect();
    writer.write_image_data(&rgb)?;
    writer.finish()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::io::Cursor;

    use png::Decoder;
    use render::gpu::AdapterRequest;
    use render::surface::{Acquired, Target};
    use wgpu::{
        Color, CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    };

    use super::*;
    use crate::test_dir::TestDir;

    /// A PNG's size and its pixels, 8-bit RGB, row by row from the top.
    fn decode(bytes: &[u8]) -> (u32, u32, Vec<[u8; 3]>) {
        let mut reader = Decoder::new(Cursor::new(bytes)).read_info().expect("a PNG");
        let info = reader.info();
        assert_eq!(
            (info.color_type, info.bit_depth),
            (ColorType::Rgb, BitDepth::Eight)
        );
        assert!(info.srgb.is_some(), "marked sRGB");
        let (width, height) = (info.width, info.height);
        let mut buffer = vec![0; reader.output_buffer_size().expect("a size")];
        let frame = reader.next_frame(&mut buffer).expect("the image");
        let image = buffer
            .get(..frame.buffer_size())
            .expect("within the buffer");
        let pixels = image.as_chunks::<3>().0.iter().copied();
        (width, height, pixels.collect())
    }

    #[test]
    fn texels_go_into_the_png_as_they_are_without_their_alpha() {
        let size = Size::new(3, 2).expect("non-zero");
        let pixels: Vec<[u8; 4]> = (0..6_u8).map(|i| [i, 40 * i, 255 - i, 7]).collect();
        let mut bytes = Vec::new();
        encode(size, &pixels, &mut bytes).expect("encoded");
        let (width, height, decoded) = decode(&bytes);
        assert_eq!((width, height), (3, 2));
        let want: Vec<[u8; 3]> = pixels.iter().map(|&[r, g, b, _]| [r, g, b]).collect();
        assert_eq!(decoded, want, "in order, row by row from the top");
        let short = encode(size, &pixels[..5], &mut Vec::new());
        assert!(short.is_err(), "too few texels are refused");
    }

    #[test]
    fn a_cleared_offscreen_frame_round_trips_into_the_file() {
        let gpu = Gpu::open(&AdapterRequest::Software).expect("GPU tests need lavapipe");
        // 37 texels across: 148 bytes a row, padded to 256 for the copy, so the readback must
        // drop the padding for the rows to line up.
        let size = Size::new(37, 5).expect("non-zero");
        let mut target = Target::Offscreen(Offscreen::new(&gpu, size).expect("a target"));
        let Acquired::Frame(frame) = target.acquire(&gpu).expect("acquired") else {
            unreachable!("an offscreen target always has a frame");
        };
        let mut encoder = gpu
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // Linear 1, 0 and 0.5: sRGB-encoded, 255, 0 and 188.
        let clear = Color {
            r: 1.0,
            g: 0.0,
            b: 0.5,
            a: 1.0,
        };
        drop(encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: frame.view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(clear),
                    store: StoreOp::Store,
                },
            })],
            ..RenderPassDescriptor::default()
        }));
        gpu.queue().submit([encoder.finish()]);
        frame.present(&gpu);
        let Target::Offscreen(offscreen) = &target else {
            unreachable!("made offscreen");
        };
        let dir = TestDir::new();
        let path = dir.path().join("frame.png");
        save(offscreen, &gpu, &path).expect("saved");
        let (width, height, pixels) = decode(&std::fs::read(&path).expect("written"));
        assert_eq!((width, height), (37, 5));
        assert_eq!(pixels.len(), 37 * 5);
        for pixel in pixels {
            let [r, g, b] = pixel;
            assert_eq!((r, g), (255, 0), "{pixel:?}");
            assert!(b.abs_diff(188) <= 1, "{pixel:?}");
        }
    }
}
