//! Framing a stroke record at a stroke's end, on the drawing thread: quantized samples delta-coded,
//! transposed into byte planes and zstd-compressed with the settings, then framed.
//!
//! - [`frame_2000_samples`]: a 2,000-sample stroke (about 8 s of a pen at 240 Hz), every field
//!   moving, with 400 bytes of settings.
//!
//! Run with `cargo bench -p store --features bench --bench stroke_record`.

#![expect(
    clippy::expect_used,
    reason = "benchmark setup; a failure here should abort the run"
)]

use store::{OpId, PenSample, PenTilt, StrokeCamera, StrokeRecord};

fn main() {
    divan::main();
}

/// Samples in the stroke framed.
const SAMPLES: u32 = 2_000;

/// A stroke of [`SAMPLES`] samples along a wobbly line, every field moving as a real pen's do.
///
/// # Panics
///
/// Never: every value is within the record's limits.
#[expect(
    clippy::cast_possible_truncation,
    reason = "pressure and tilt are small, and an f32 holds them"
)]
fn stroke() -> StrokeRecord {
    let camera = StrokeCamera {
        eye: [0.0, 0.5, 4.0],
        target: [0.0; 3],
        up: [0.0, 1.0, 0.0],
        fov_y: 0.785,
        size: [2560, 1440],
    };
    let settings: Vec<u8> = (0..400_u32)
        .map(|k| u8::try_from(k * 37 % 251).expect("below 251"))
        .collect();
    let op = OpId::new(0, 1).expect("a valid op");
    let mut record = StrokeRecord::new(op, 1, settings, camera).expect("settings fit");
    let samples = (0..SAMPLES).map(|k| {
        let t = f64::from(k);
        PenSample {
            x: 300.0 + t * 0.7 + (t * 0.05).sin() * 3.0,
            y: 800.0 - t * 0.3 + (t * 0.03).cos() * 2.0,
            time: 1_000.0 + t / 240.0,
            pressure: ((t * 0.01).sin().abs() * 0.8 + 0.1) as f32,
            tilt: Some(PenTilt {
                amount: ((t * 0.002).sin().abs() * 0.9) as f32,
                direction: ((t * 0.004) % std::f64::consts::TAU) as f32,
            }),
            twist: None,
            buttons: 0,
        }
    });
    for sample in samples {
        record.push(&sample).expect("room for the samples");
    }
    record
}

/// Framing a 2,000-sample stroke record: what a stroke end costs the drawing thread for its
/// record.
///
/// # Panics
///
/// If framing fails: the compressor's error.
#[divan::bench]
fn frame_2000_samples(bencher: divan::Bencher<'_, '_>) {
    let record = stroke();
    bencher.bench(|| divan::black_box(&record).framed_len().expect("frames"));
}
