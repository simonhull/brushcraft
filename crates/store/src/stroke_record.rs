//! The stroke record: what a brush stroke was made from, kept beside its edits.
//!
//! Each stroke's op can carry one [`StrokeRecord`]: its pen samples, quantized, its brush settings
//! as opaque bytes (`stroke::StrokeSettings::to_bytes`; the store never reads them), the camera
//! it was drawn through, and the version of the engine that placed its dabs. The edits stay
//! authoritative: a record is history, for later features and for debugging, and nothing reads
//! it to open a project. Opening a file only steps over its head (see `crate::file`).
//!
//! # Samples
//!
//! A sample is quantized to [`SAMPLE_BYTES`] bytes (see [`PenSample`] for each field's step and
//! error). The samples are then delta-coded, field by field against the sample before (wrapping
//! at the field's width, the difference zigzag-coded so a small step either way has small
//! bytes), the rows transposed into sixteen byte planes, and the planes zstd-compressed with the
//! settings in front of them, as tile bodies are.
//!
//! # Payload
//!
//! A stroke record (kind 6) holds, all numbers little-endian:
//!
//! - the record's version, `2`, as a `u8`, then three reserved zero bytes;
//! - the op, as `OpId::to_bits` packs it, a `u64`;
//! - the engine version, a `u32`;
//! - the camera: eye, target and up, each three `f64`s, the vertical field of view in radians as an
//!   `f64`, the view's width and height in physical pixels as `u32`s, then its magnification and
//!   the image's shift, right and up, three `f64`s (see [`StrokeCamera`]);
//! - the start time (the first sample's, in seconds on the pen's clock), an `f64`;
//! - the settings' length, a `u32`, at most [`MAX_STROKE_SETTINGS`];
//! - the samples' count, a `u32`, at most [`MAX_STROKE_SAMPLES`];
//! - a zstd frame of the settings, then the sample planes: `settings + 16 × count` bytes once
//!   decompressed, exactly.
//!
//! **Version 1**, written before views could be magnified, is the same without the last three
//! `f64`s of the camera: it reads as magnification 1 and no shift, the view it was drawn through.
//!
//! A reader refuses any other version, a reserved byte set, a reserved site's op, a start time
//! that isn't finite, a length or count past its limit, anything but exactly one zstd frame after
//! the fixed fields (a skippable frame, or bytes after the frame), a frame that doesn't decompress
//! to the stated size, and a sample that isn't canonical (a tilt direction with no tilt, see
//! [`PenSample`]; a twist with no tilt is valid).
//!
//! # Where records live
//!
//! No Index names a stroke record: a save point holds every stroke record before it in the
//! file, and a file holds at most one record of an op. Compaction (`crate::file::compact`) keeps
//! the latest state only and drops every stroke record: which history to keep is decided with
//! stroke history in M2.

use std::f64::consts::{FRAC_PI_2, TAU};

use thiserror::Error;

use crate::OpId;
use crate::bytes::{Reader, transpose, untranspose};
use crate::codec::{compress, decompress_frame};

/// Bytes a quantized sample takes, before delta coding and compression.
pub const SAMPLE_BYTES: usize = 16;

/// The most samples a record holds: 2²⁰, over an hour of a pen reporting at 240 Hz.
pub const MAX_STROKE_SAMPLES: usize = 1 << 20;

/// The longest settings a record holds, in bytes. A stroke's settings take a few hundred.
pub const MAX_STROKE_SETTINGS: usize = 1 << 16;

/// The record's version this build writes: 2, whose camera holds the view's magnification and
/// shift.
const VERSION: u8 = 2;

/// The version before magnified views, which this build still reads: its camera holds no
/// magnification or shift, and reads as magnification 1, shift 0.
const UNMAGNIFIED_VERSION: u8 = 1;

/// Payload bytes before the compressed frame, as this build writes it: version and reserved
/// bytes, op, engine, camera, start time, settings length and sample count.
const FIXED: usize = 4 + 8 + 4 + CAMERA_BYTES + 8 + 4 + 4;

/// [`FIXED`] in a version 1 payload, whose camera has no lens.
#[cfg(test)]
const UNMAGNIFIED_FIXED: usize = FIXED - LENS_BYTES;

/// Where the op is in a payload: after the version and reserved bytes. Fixed in every version,
/// so a file can tell which strokes it holds records of without decoding them.
pub(crate) const OP_AT: usize = 4;

/// The longest payload a writer makes: the fixed fields, then a zstd frame of the longest
/// settings and the most samples, which zstd never expands by more than 1/128 plus a small
/// constant.
pub(crate) const MAX_STROKE_PAYLOAD: usize = {
    let raw = MAX_STROKE_SETTINGS + SAMPLE_BYTES * MAX_STROKE_SAMPLES;
    FIXED + raw + raw / 128 + 1024
};

/// Bytes of the camera: nine `f64`s of eye, target and up, the field of view, two `u32`s of size,
/// and the lens.
const CAMERA_BYTES: usize = 10 * 8 + 2 * 4 + LENS_BYTES;

/// Bytes of the camera's lens, since version 2: the magnification and the shift's two `f64`s.
const LENS_BYTES: usize = 3 * 8;

/// Steps per physical pixel of a quantized screen coordinate.
const POSITION_STEPS: f64 = 32.0;

/// Steps per second of a quantized time.
const TIME_STEPS: f64 = 1e6;

/// The largest quantized pressure: pressure 1.
const PRESSURE_STEPS: f64 = 65_535.0;

/// The largest quantized tilt amount: π/2, lying flat.
const TILT_AMOUNT_STEPS: f64 = 254.0;

/// Steps of a full turn of tilt direction.
const TILT_DIRECTION_STEPS: f64 = 256.0;

/// Steps of a full turn of twist.
const TWIST_STEPS: f64 = 255.0;

/// The tilt amount byte of a sample with no tilt; its direction byte is then 0.
const NO_TILT: u8 = u8::MAX;

/// The twist byte of a sample with no twist.
const NO_TWIST: u8 = u8::MAX;

/// The bits of each sample field in the row, in order: x, y, time, pressure, tilt amount, tilt
/// direction, twist, buttons. Their sum is [`SAMPLE_BYTES`] × 8.
const FIELD_BITS: [u32; 8] = [24, 24, 32, 16, 8, 8, 8, 8];

/// The camera a stroke was drawn through: enough to turn its screen samples back into rays.
///
/// The view looks from `eye` towards `target` with `up` as its up and a vertical field of
/// `fov_y`, into a viewport of `size`; then its image is magnified `magnification` times about
/// the viewport's centre and moved by `shift`: a point the plain view shows at `t` on its image
/// plane (right and up, over its depth) shows at `magnification · t + shift`. A stroke drawn with
/// the view unmagnified has magnification 1 and no shift.
///
/// Stored bit for bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeCamera {
    /// Where the eye is, in model units.
    pub eye: [f64; 3],
    /// The point the camera looks at.
    pub target: [f64; 3],
    /// The camera's up direction.
    pub up: [f64; 3],
    /// The vertical field of view, in radians.
    pub fov_y: f64,
    /// The view's width and height, in physical pixels.
    pub size: [u32; 2],
    /// How many times the view's image is magnified: 1 for a plain view.
    pub magnification: f64,
    /// The magnified image's shift, right and up, in units of the plain view's image plane at
    /// unit depth: 0 for a plain view.
    pub shift: [f64; 2],
}

impl Default for StrokeCamera {
    /// Every number zero but the magnification, 1: an unmagnified view of nothing.
    fn default() -> Self {
        Self {
            eye: [0.0; 3],
            target: [0.0; 3],
            up: [0.0; 3],
            fov_y: 0.0,
            size: [0; 2],
            magnification: 1.0,
            shift: [0.0; 2],
        }
    }
}

/// Which way the pen leans and how far, as a [`PenSample`] carries it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PenTilt {
    /// Radians from upright, in `0..=π/2`.
    pub amount: f32,
    /// Radians from `+x` towards `+y`, in `0..2π`.
    pub direction: f32,
}

/// One pen sample, as a stroke record takes and gives it: a plain copy of what the stroke engine
/// read (`stroke::Sample`), so the store needs nothing of the engine.
///
/// A record keeps each field quantized; what [`StrokeRecord::samples`] gives back differs from
/// what [`StrokeRecord::push`] took by at most:
///
/// | field | step | error | range kept |
/// |---|---|---|---|
/// | `x`, `y` | 1/32 px | 1/64 px | ±262,144 px |
/// | `time` | 1 µs | 0.5 µs | 0 to 4,294 s after the first sample |
/// | `pressure` | 1/65,535 | 7.7 × 10⁻⁶ | 0 to 1 |
/// | tilt `amount` | π/508 rad | 0.0031 rad (0.18°) | 0 to π/2 |
/// | tilt `direction` | π/128 rad | 0.0123 rad (0.71°) | a full turn |
/// | `twist` | 2π/255 rad | 0.0123 rad (0.71°) | a full turn |
/// | `buttons` | exact | none | every bit |
///
/// A value outside its range is clamped into it (a direction or twist wraps), and one that isn't
/// a number reads as zero (a time, as the first sample's). The time's error
/// is relative to the first sample's time, kept exactly; a sample before it reads as at it.
/// Both angles come back in `0..2π`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PenSample {
    /// Pixels right of the view's left edge, physical.
    pub x: f64,
    /// Pixels below the view's top edge, physical.
    pub y: f64,
    /// Seconds on the pen's clock.
    pub time: f64,
    /// Tip pressure, in `0..=1`.
    pub pressure: f32,
    /// The pen's tilt, if it reports one.
    pub tilt: Option<PenTilt>,
    /// The pen's barrel rotation in `0..2π` radians, if it reports one.
    pub twist: Option<f32>,
    /// Barrel buttons held, as `stroke::Buttons::bits` gives them.
    pub buttons: u8,
}

/// Why a [`StrokeRecord`] can't take what it was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StrokeRecordError {
    /// The settings are longer than [`MAX_STROKE_SETTINGS`] bytes.
    #[error("stroke settings of {0} bytes are longer than a record holds")]
    SettingsTooLong(usize),
    /// The record holds [`MAX_STROKE_SAMPLES`] samples already.
    #[error("the stroke record holds as many samples as it can")]
    Full,
}

/// One sample, quantized: the integers a record stores (see [`PenSample`] for the steps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Quantized {
    /// The horizontal position in 1/32 px, in the low 24 bits, two's complement.
    x: u32,
    /// The vertical position, as `x`.
    y: u32,
    /// Microseconds after the first sample.
    time: u32,
    /// Pressure in 1/65,535.
    pressure: u16,
    /// Tilt amount in steps of π/508 rad, or [`NO_TILT`].
    tilt_amount: u8,
    /// Tilt direction in steps of π/128 rad; 0 with no tilt.
    tilt_direction: u8,
    /// Twist in steps of 2π/255 rad, or [`NO_TWIST`].
    twist: u8,
    /// Buttons, exactly.
    buttons: u8,
}

impl Quantized {
    /// The fields as unsigned values of their [`FIELD_BITS`].
    fn fields(self) -> [u32; 8] {
        [
            self.x,
            self.y,
            self.time,
            self.pressure.into(),
            self.tilt_amount.into(),
            self.tilt_direction.into(),
            self.twist.into(),
            self.buttons.into(),
        ]
    }

    /// The sample of `fields`, each within its [`FIELD_BITS`]; `None` if it isn't canonical: a
    /// tilt direction with no tilt.
    fn from_fields(
        [x, y, time, pressure, amount, direction, twist, buttons]: [u32; 8],
    ) -> Option<Self> {
        let byte = |v: u32| u8::try_from(v).ok();
        let sample = Self {
            x,
            y,
            time,
            pressure: u16::try_from(pressure).ok()?,
            tilt_amount: byte(amount)?,
            tilt_direction: byte(direction)?,
            twist: byte(twist)?,
            buttons: byte(buttons)?,
        };
        (sample.tilt_amount != NO_TILT || sample.tilt_direction == 0).then_some(sample)
    }

    /// `sample` quantized, its time taken after `start`.
    fn of(sample: &PenSample, start: f64) -> Self {
        let (tilt_amount, tilt_direction) = sample.tilt.map_or((NO_TILT, 0), |tilt| {
            let amount = steps(
                f64::from(tilt.amount) / FRAC_PI_2 * TILT_AMOUNT_STEPS,
                0,
                254,
            );
            (
                low_byte(amount),
                low_byte(turn_steps(tilt.direction, TILT_DIRECTION_STEPS)),
            )
        });
        let twist = sample
            .twist
            .map_or(NO_TWIST, |twist| low_byte(turn_steps(twist, TWIST_STEPS)));
        Self {
            x: coordinate(sample.x),
            y: coordinate(sample.y),
            time: u32::try_from(steps(
                (sample.time - start) * TIME_STEPS,
                0,
                u32::MAX.into(),
            ))
            .unwrap_or(u32::MAX),
            pressure: u16::try_from(steps(
                f64::from(sample.pressure) * PRESSURE_STEPS,
                0,
                u16::MAX.into(),
            ))
            .unwrap_or(u16::MAX),
            tilt_amount,
            tilt_direction,
            twist,
            buttons: sample.buttons,
        }
    }

    /// The sample this stands for, its time after `start`.
    fn sample(self, start: f64) -> PenSample {
        let tilt = (self.tilt_amount != NO_TILT).then(|| PenTilt {
            amount: narrow(f64::from(self.tilt_amount) / TILT_AMOUNT_STEPS * FRAC_PI_2),
            direction: narrow(f64::from(self.tilt_direction) / TILT_DIRECTION_STEPS * TAU),
        });
        let twist =
            (self.twist != NO_TWIST).then(|| narrow(f64::from(self.twist) / TWIST_STEPS * TAU));
        PenSample {
            x: f64::from(sign_extend_24(self.x)) / POSITION_STEPS,
            y: f64::from(sign_extend_24(self.y)) / POSITION_STEPS,
            time: start + f64::from(self.time) / TIME_STEPS,
            pressure: narrow(f64::from(self.pressure) / PRESSURE_STEPS),
            tilt,
            twist,
            buttons: self.buttons,
        }
    }
}

/// `value` rounded to the nearest integer and clamped to `low..=high`; 0, clamped likewise, if
/// it isn't a number.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is rounded and clamped into i64's range before the cast"
)]
fn steps(value: f64, low: i64, high: i64) -> i64 {
    if value.is_nan() {
        return 0.clamp(low, high);
    }
    // Both bounds are well inside f64's exact integers, so the clamp is exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the bounds used are at most 2³², exact in an f64"
    )]
    let clamped = value.round().clamp(low as f64, high as f64);
    clamped as i64
}

/// A screen coordinate in 1/32 px, clamped to 24 bits, as the low 24 bits of a `u32`.
fn coordinate(value: f64) -> u32 {
    let q = steps(value * POSITION_STEPS, -(1 << 23), (1 << 23) - 1);
    let wrapped = q.rem_euclid(1 << 24);
    u32::try_from(wrapped).unwrap_or(0)
}

/// The 24-bit two's complement value in the low bits of `value`, sign-extended.
const fn sign_extend_24(value: u32) -> i32 {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "the shift puts bit 23 in the sign bit, and the arithmetic shift back extends it"
    )]
    let shifted = (value << 8) as i32;
    shifted >> 8
}

/// An angle in steps of `per_turn` steps a turn, wrapped into `0..per_turn`; 0 if it isn't a
/// number.
fn turn_steps(radians: f32, per_turn: f64) -> i64 {
    let turns = f64::from(radians) / TAU;
    if !turns.is_finite() {
        return 0;
    }
    let q = steps(turns.rem_euclid(1.0) * per_turn, 0, i64::from(u32::MAX));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "per_turn is 255 or 256, an exact integer"
    )]
    let modulus = per_turn as i64;
    q.rem_euclid(modulus)
}

/// The low byte of a value its caller has already put in `0..=255`.
fn low_byte(value: i64) -> u8 {
    u8::try_from(value).unwrap_or(0)
}

/// `value` as the nearest `f32`: a dequantized field, which an `f32` holds to well below its
/// step.
#[expect(
    clippy::cast_possible_truncation,
    reason = "every dequantized value is small, and an f32 holds it to well below its step"
)]
const fn narrow(value: f64) -> f32 {
    value as f32
}

/// The mask of a field of `bits` bits.
const fn mask(bits: u32) -> u64 {
    (1 << bits) - 1
}

/// `value - previous` at a field's width, zigzag-coded: small steps either way become small
/// values.
fn delta(value: u32, previous: u32, bits: u32) -> u32 {
    let m = mask(bits);
    let d = u64::from(value).wrapping_sub(u64::from(previous)) & m;
    // The difference as a signed value of `bits` bits, then zigzag: 2d for d >= 0, -2d - 1 for
    // d < 0, both within the field.
    let negative = d >> (bits - 1) & 1;
    let zig = ((d << 1) ^ negative.wrapping_neg()) & m;
    u32::try_from(zig).unwrap_or(0)
}

/// Undoes [`delta`]: the value whose zigzag-coded difference from `previous` is `zig`.
fn undelta(zig: u32, previous: u32, bits: u32) -> u32 {
    let m = mask(bits);
    let z = u64::from(zig) & m;
    let d = ((z >> 1) ^ (z & 1).wrapping_neg()) & m;
    let value = u64::from(previous).wrapping_add(d) & m;
    u32::try_from(value).unwrap_or(0)
}

/// The bytes a field of `bits` bits takes in a row.
fn field_bytes(bits: u32) -> usize {
    usize::try_from(bits / 8).unwrap_or(0)
}

/// One row's fields, as [`FIELD_BITS`] lays them out, little-endian.
fn put_row(out: &mut [u8; SAMPLE_BYTES], fields: [u32; 8]) {
    let bytes = FIELD_BITS
        .iter()
        .zip(fields)
        .flat_map(|(&bits, value)| value.to_le_bytes().into_iter().take(field_bytes(bits)));
    let slots = out.iter_mut().zip(bytes);
    for (slot, byte) in slots {
        *slot = byte;
    }
}

/// The fields of a row [`put_row`] wrote.
fn read_row(row: &[u8; SAMPLE_BYTES]) -> [u32; 8] {
    let mut fields = [0; 8];
    let mut at = row.iter();
    let widths = fields.iter_mut().zip(&FIELD_BITS);
    for (field, &bits) in widths {
        let mut le = [0; 4];
        let bytes = le.iter_mut().zip(at.by_ref().take(field_bytes(bits)));
        for (slot, &byte) in bytes {
            *slot = byte;
        }
        *field = u32::from_le_bytes(le);
    }
    fields
}

/// The record a stroke leaves beside its edits: its op, the engine's version, its settings as
/// opaque bytes, the camera and its samples, quantized (see the module notes).
///
/// Built with [`StrokeRecord::new`] and [`StrokeRecord::push`], one sample at a time, then handed
/// to [`crate::Project::end_stroke`]; read back by
/// [`ProjectFile::strokes`](crate::ProjectFile::strokes), for tools and tests only.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeRecord {
    /// The stroke's op: every edit of the stroke was made as it.
    op: OpId,
    /// The version of the engine that placed the stroke's dabs.
    engine: u32,
    /// The brush settings, as the engine's byte form; at most [`MAX_STROKE_SETTINGS`] bytes.
    settings: Vec<u8>,
    /// The camera the stroke was drawn through.
    camera: StrokeCamera,
    /// The first sample's time, the origin of every sample's quantized time; 0 before any, and
    /// always finite.
    start: f64,
    /// The samples, quantized, at most [`MAX_STROKE_SAMPLES`].
    samples: Vec<Quantized>,
}

impl StrokeRecord {
    /// An empty record of the stroke made as `op`, by engine version `engine`, with `settings`
    /// (the engine's byte form, kept as it is) and seen through `camera`.
    ///
    /// # Errors
    ///
    /// [`StrokeRecordError::SettingsTooLong`] past [`MAX_STROKE_SETTINGS`] bytes.
    pub fn new(
        op: OpId,
        engine: u32,
        settings: Vec<u8>,
        camera: StrokeCamera,
    ) -> Result<Self, StrokeRecordError> {
        if settings.len() > MAX_STROKE_SETTINGS {
            return Err(StrokeRecordError::SettingsTooLong(settings.len()));
        }
        Ok(Self {
            op,
            engine,
            settings,
            camera,
            start: 0.0,
            samples: Vec::new(),
        })
    }

    /// Adds `sample`, quantized (see [`PenSample`] for what is kept). The first sample's time
    /// is the origin of every sample's time, kept exactly (0 if it isn't finite).
    ///
    /// # Errors
    ///
    /// [`StrokeRecordError::Full`] once the record holds [`MAX_STROKE_SAMPLES`]; the sample is
    /// left out.
    pub fn push(&mut self, sample: &PenSample) -> Result<(), StrokeRecordError> {
        if self.samples.len() >= MAX_STROKE_SAMPLES {
            return Err(StrokeRecordError::Full);
        }
        if self.samples.is_empty() {
            self.start = if sample.time.is_finite() {
                sample.time
            } else {
                0.0
            };
        }
        self.samples.push(Quantized::of(sample, self.start));
        Ok(())
    }

    /// The stroke's op.
    #[must_use]
    pub const fn op(&self) -> OpId {
        self.op
    }

    /// The version of the engine that placed the stroke's dabs.
    #[must_use]
    pub const fn engine(&self) -> u32 {
        self.engine
    }

    /// The brush settings, as the engine wrote them.
    #[must_use]
    pub fn settings(&self) -> &[u8] {
        &self.settings
    }

    /// The camera the stroke was drawn through.
    #[must_use]
    pub const fn camera(&self) -> &StrokeCamera {
        &self.camera
    }

    /// The first sample's time, in seconds on the pen's clock; 0 before any sample.
    #[must_use]
    pub const fn start_time(&self) -> f64 {
        self.start
    }

    /// How many samples the record holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the record holds no sample.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Frames the record as a stroke end does, delta coding and compression included, and
    /// returns the framed length: what `benches/stroke_record.rs` times. Built only with the
    /// `bench` feature.
    ///
    /// # Errors
    ///
    /// As framing at a stroke end: the compressor's I/O error.
    #[cfg(feature = "bench")]
    #[doc(hidden)]
    pub fn framed_len(&self) -> Result<usize, crate::FileError> {
        crate::file::frame_stroke(self).map(|framed| framed.len())
    }

    /// The samples, in order, as quantized (see [`PenSample`] for how far each field may be
    /// from what was pushed).
    pub fn samples(&self) -> impl ExactSizeIterator<Item = PenSample> + '_ {
        self.samples
            .iter()
            .map(|quantized| quantized.sample(self.start))
    }

    /// The record's payload (see the module notes).
    ///
    /// # Errors
    ///
    /// The compressor's I/O error.
    pub(crate) fn to_payload(&self) -> std::io::Result<Vec<u8>> {
        let mut raw = Vec::with_capacity(self.settings.len() + self.samples.len() * SAMPLE_BYTES);
        raw.extend_from_slice(&self.settings);
        raw.extend_from_slice(&planes(&self.samples));
        let packed = compress(&raw)?;
        let mut out = Vec::with_capacity(FIXED + packed.len());
        out.extend_from_slice(&[VERSION, 0, 0, 0]);
        out.extend_from_slice(&self.op.to_bits().to_le_bytes());
        out.extend_from_slice(&self.engine.to_le_bytes());
        let camera = &self.camera;
        let floats = camera
            .eye
            .iter()
            .chain(&camera.target)
            .chain(&camera.up)
            .chain(std::iter::once(&camera.fov_y));
        for value in floats {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for side in camera.size {
            out.extend_from_slice(&side.to_le_bytes());
        }
        let lens = std::iter::once(&camera.magnification).chain(&camera.shift);
        for value in lens {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&self.start.to_le_bytes());
        // Both fit: `new` and `push` hold them to their limits, far below u32::MAX.
        let settings_len = u32::try_from(self.settings.len()).unwrap_or(u32::MAX);
        let count = u32::try_from(self.samples.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&settings_len.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&packed);
        Ok(out)
    }

    /// The record a payload holds, or `None` if it isn't one this build reads (see the module
    /// notes for what is refused).
    pub(crate) fn from_payload(payload: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(payload);
        let [version, r0, r1, r2] = <[u8; 4]>::try_from(reader.take(4)?).ok()?;
        let magnified = match version {
            VERSION => true,
            UNMAGNIFIED_VERSION => false,
            _ => return None,
        };
        if (r0, r1, r2) != (0, 0, 0) {
            return None;
        }
        let op = OpId::from_bits(reader.u64()?)?;
        let engine = reader.u32()?;
        let mut f64s = [0.0; 10];
        for value in &mut f64s {
            *value = f64::from_bits(reader.u64()?);
        }
        let [ex, ey, ez, tx, ty, tz, ux, uy, uz, fov_y] = f64s;
        let size = [reader.u32()?, reader.u32()?];
        let [magnification, sx, sy] = if magnified {
            let mut lens = [0.0; 3];
            for value in &mut lens {
                *value = f64::from_bits(reader.u64()?);
            }
            lens
        } else {
            [1.0, 0.0, 0.0]
        };
        let start = f64::from_bits(reader.u64()?);
        if !start.is_finite() {
            return None;
        }
        let settings_len = usize::try_from(reader.u32()?).ok()?;
        let count = usize::try_from(reader.u32()?).ok()?;
        if settings_len > MAX_STROKE_SETTINGS || count > MAX_STROKE_SAMPLES {
            return None;
        }
        let raw = decompress_frame(reader.rest(), settings_len + count * SAMPLE_BYTES).ok()?;
        let (settings, planes) = raw.split_at_checked(settings_len)?;
        Some(Self {
            op,
            engine,
            settings: settings.to_vec(),
            camera: StrokeCamera {
                eye: [ex, ey, ez],
                target: [tx, ty, tz],
                up: [ux, uy, uz],
                fov_y,
                size,
                magnification,
                shift: [sx, sy],
            },
            start,
            samples: unplane(planes, count)?,
        })
    }
}

/// `samples` delta-coded and transposed into [`SAMPLE_BYTES`] byte planes (see
/// [`transpose`]).
fn planes(samples: &[Quantized]) -> Vec<u8> {
    let mut previous = [0; 8];
    let rows = samples.iter().map(|sample| {
        let fields = sample.fields();
        let mut coded = [0; 8];
        let columns = coded
            .iter_mut()
            .zip(&fields)
            .zip(&previous)
            .zip(&FIELD_BITS);
        for (((slot, &value), &before), &bits) in columns {
            *slot = delta(value, before, bits);
        }
        previous = fields;
        let mut row = [0; SAMPLE_BYTES];
        put_row(&mut row, coded);
        row
    });
    transpose(rows)
}

/// Undoes [`planes`] for `count` samples: `None` if the planes aren't `count` rows, or a sample
/// isn't canonical.
fn unplane(planes: &[u8], count: usize) -> Option<Vec<Quantized>> {
    let mut previous = [0; 8];
    let samples = untranspose::<SAMPLE_BYTES>(planes, count)?.map(|row| {
        let coded = read_row(&row);
        let mut fields = [0; 8];
        let columns = fields
            .iter_mut()
            .zip(&coded)
            .zip(&previous)
            .zip(&FIELD_BITS);
        for (((slot, &zig), &before), &bits) in columns {
            *slot = undelta(zig, before, bits);
        }
        previous = fields;
        Quantized::from_fields(fields)
    });
    samples.collect()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
pub(crate) mod tests {
    use std::f32::consts::{FRAC_PI_2 as HALF_PI_32, TAU as TAU_32};

    use super::*;

    /// A record of `n` samples along a wobbly line, every field moving.
    pub(crate) fn wobbly(op: OpId, n: u32, seed: u64) -> StrokeRecord {
        let camera = StrokeCamera {
            eye: [0.0, 0.5, 4.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: 0.785,
            size: [2560, 1440],
            magnification: 6.5,
            shift: [0.125, -0.0703],
        };
        let settings: Vec<u8> = seed
            .to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(90)
            .collect();
        let mut record = StrokeRecord::new(op, 7, settings, camera).unwrap();
        for k in 0..n {
            record.push(&sample_at(k, seed)).unwrap();
        }
        record
    }

    /// The `k`th sample of [`wobbly`]'s line.
    fn sample_at(k: u32, seed: u64) -> PenSample {
        let t = f64::from(k);
        let s = f64::from(u32::try_from(seed).unwrap());
        PenSample {
            x: 300.0 + t * 1.7 + (t * 0.3 + s).sin() * 2.0,
            y: 800.0 - t * 0.9 + (t * 0.2).cos(),
            time: 1234.5 + t / 240.0,
            pressure: narrow((t * 0.05).sin().abs()),
            tilt: (t < 40.0 || t > 60.0).then(|| PenTilt {
                amount: narrow((t * 0.01).sin().abs()) * HALF_PI_32,
                direction: narrow(t * 0.03) % TAU_32,
            }),
            twist: (t > 10.0).then(|| narrow(t * 0.02) % TAU_32),
            buttons: u8::try_from(k / 16 % 4).unwrap(),
        }
    }

    #[test]
    fn records_round_trip_through_their_payload() {
        let record = wobbly(OpId::new(3, 41).unwrap(), 500, 9);
        let payload = record.to_payload().unwrap();
        assert_eq!(
            StrokeRecord::from_payload(&payload),
            Some(record),
            "round trip"
        );
        let empty = StrokeRecord::new(
            OpId::new(0, 0).unwrap(),
            0,
            Vec::new(),
            StrokeCamera::default(),
        )
        .unwrap();
        let back = StrokeRecord::from_payload(&empty.to_payload().unwrap());
        assert_eq!(back, Some(empty), "an empty record");
        // Small: about 16 bytes a sample before compression, a few after.
        assert!(
            payload.len() < 500 * SAMPLE_BYTES / 2,
            "{} bytes for 500 samples",
            payload.len()
        );
    }

    /// `record`'s payload as version 1 wrote it: no lens in its camera.
    fn unmagnified_payload(record: &StrokeRecord) -> Vec<u8> {
        let mut payload = record.to_payload().unwrap();
        payload[0] = UNMAGNIFIED_VERSION;
        // The lens follows the camera's size: after the version, op, engine, ten f64s and two
        // u32s.
        let lens = 4 + 8 + 4 + 10 * 8 + 2 * 4;
        payload.drain(lens..lens + LENS_BYTES);
        payload
    }

    #[test]
    fn a_version_1_record_reads_as_drawn_through_an_unmagnified_view() {
        let record = wobbly(OpId::new(2, 17).unwrap(), 40, 3);
        let old = unmagnified_payload(&record);
        assert_eq!(old.len(), record.to_payload().unwrap().len() - LENS_BYTES);
        let back = StrokeRecord::from_payload(&old).expect("version 1 reads");
        let camera = *back.camera();
        assert_eq!(camera.magnification, 1.0, "magnification 1");
        assert_eq!(camera.shift, [0.0, 0.0], "no shift");
        assert_eq!(
            StrokeCamera {
                magnification: record.camera().magnification,
                shift: record.camera().shift,
                ..camera
            },
            *record.camera(),
            "the rest as written"
        );
        assert!(back.samples().eq(record.samples()), "the same samples");
        assert_eq!(back.settings(), record.settings());
        let mut cut = old.clone();
        cut.truncate(UNMAGNIFIED_FIXED - 1);
        assert_eq!(
            StrokeRecord::from_payload(&cut),
            None,
            "a short version 1 payload"
        );
        let mut reserved = old;
        reserved[3] = 1;
        assert_eq!(
            StrokeRecord::from_payload(&reserved),
            None,
            "a reserved byte set"
        );
    }

    #[test]
    fn every_field_comes_back_within_its_stated_error() {
        let record = wobbly(OpId::new(0, 1).unwrap(), 200, 3);
        let pushed = (0..200).map(|k| sample_at(k, 3));
        let pairs = pushed.zip(record.samples()).enumerate();
        for (k, (given, kept)) in pairs {
            let near = |a: f64, b: f64, error: f64, what: &str| {
                assert!(
                    (a - b).abs() <= error,
                    "sample {k}: {what}: {a} against {b}"
                );
            };
            near(given.x, kept.x, 1.0 / 64.0, "x");
            near(given.y, kept.y, 1.0 / 64.0, "y");
            near(given.time, kept.time, 0.5e-6 + 1e-9, "time");
            near(
                f64::from(given.pressure),
                f64::from(kept.pressure),
                0.5 / 65_535.0 + 1e-7,
                "pressure",
            );
            assert_eq!(
                given.tilt.is_some(),
                kept.tilt.is_some(),
                "sample {k}: tilt"
            );
            if let (Some(a), Some(b)) = (given.tilt, kept.tilt) {
                near(
                    f64::from(a.amount),
                    f64::from(b.amount),
                    FRAC_PI_2 / 508.0 + 1e-6,
                    "tilt amount",
                );
                let turn = (f64::from(a.direction) - f64::from(b.direction)).rem_euclid(TAU);
                near(
                    turn.min(TAU - turn),
                    0.0,
                    TAU / 512.0 + 1e-6,
                    "tilt direction",
                );
            }
            assert_eq!(
                given.twist.is_some(),
                kept.twist.is_some(),
                "sample {k}: twist"
            );
            if let (Some(a), Some(b)) = (given.twist, kept.twist) {
                let turn = (f64::from(a) - f64::from(b)).rem_euclid(TAU);
                near(turn.min(TAU - turn), 0.0, TAU / 510.0 + 1e-6, "twist");
            }
            assert_eq!(given.buttons, kept.buttons, "sample {k}: buttons");
        }
    }

    #[test]
    fn values_out_of_range_are_clamped_or_wrapped() {
        let mut record = StrokeRecord::new(
            OpId::new(0, 1).unwrap(),
            1,
            Vec::new(),
            StrokeCamera::default(),
        )
        .unwrap();
        let wild = PenSample {
            x: 1e9,
            y: f64::NAN,
            time: 10.0,
            pressure: 7.0,
            tilt: Some(PenTilt {
                amount: -1.0,
                direction: -0.5 * TAU_32,
            }),
            twist: Some(f32::INFINITY),
            buttons: 0xFF,
        };
        record.push(&wild).unwrap();
        record.push(&PenSample { time: 5.0, ..wild }).unwrap();
        let kept: Vec<PenSample> = record.samples().collect();
        assert_eq!(kept[0].x, f64::from((1 << 23) - 1) / 32.0, "x clamps");
        assert_eq!(kept[0].y, 0.0, "a NaN reads as 0");
        assert_eq!(kept[0].pressure, 1.0, "pressure clamps");
        let tilt = kept[0].tilt.unwrap();
        assert_eq!(tilt.amount, 0.0, "the amount clamps");
        assert!(
            (tilt.direction - 0.5 * TAU_32).abs() < 1e-6,
            "the direction wraps"
        );
        assert_eq!(kept[0].twist, Some(0.0), "an infinite twist reads as 0");
        assert_eq!(kept[0].buttons, 0xFF, "every button bit");
        assert_eq!(kept[1].time, 10.0, "a sample before the first reads at it");
        assert_eq!(record.start_time(), 10.0, "the first sample's time");
    }

    #[test]
    fn negative_positions_and_extremes_survive_delta_coding() {
        let mut record = StrokeRecord::new(
            OpId::new(0, 1).unwrap(),
            1,
            Vec::new(),
            StrokeCamera::default(),
        )
        .unwrap();
        let xs = [-262_144.0, 262_143.0, -0.5, 0.0, 17.25, -17.25];
        let numbered = (0_u32..).zip(&xs);
        for (k, &x) in numbered {
            record
                .push(&PenSample {
                    x,
                    y: -x,
                    time: f64::from(k) * 500.0,
                    ..PenSample::default()
                })
                .unwrap();
        }
        let back = StrokeRecord::from_payload(&record.to_payload().unwrap()).unwrap();
        let kept: Vec<f64> = back.samples().map(|s| s.x).collect();
        assert_eq!(kept, xs, "every x, exactly");
        let last = back.samples().last().unwrap();
        assert_eq!(last.time, 2500.0, "a long stroke's time");
    }

    #[test]
    fn limits_are_refused() {
        let camera = StrokeCamera::default();
        let op = OpId::new(0, 1).unwrap();
        assert_eq!(
            StrokeRecord::new(op, 0, vec![0; MAX_STROKE_SETTINGS + 1], camera),
            Err(StrokeRecordError::SettingsTooLong(MAX_STROKE_SETTINGS + 1)),
            "settings past the limit"
        );
        let mut full = StrokeRecord::new(op, 0, Vec::new(), camera).unwrap();
        full.samples = vec![Quantized::default(); MAX_STROKE_SAMPLES];
        assert_eq!(
            full.push(&PenSample::default()),
            Err(StrokeRecordError::Full),
            "samples past the limit"
        );
    }

    #[test]
    fn payloads_a_writer_never_makes_are_refused() {
        let record = wobbly(OpId::new(0, 5).unwrap(), 50, 1);
        let payload = record.to_payload().unwrap();
        let refused = |change: &dyn Fn(&mut Vec<u8>), what: &str| {
            let mut bad = payload.clone();
            change(&mut bad);
            assert_eq!(StrokeRecord::from_payload(&bad), None, "{what}");
        };
        refused(&|p| p[0] = 3, "another version");
        refused(&|p| p[0] = 0, "version 0");
        refused(&|p| p[2] = 1, "a reserved byte set");
        refused(
            &|p| p[4..12].copy_from_slice(&u64::MAX.to_le_bytes()),
            "the reserved site",
        );
        let start = FIXED - 16;
        refused(
            &|p| p[start..start + 8].copy_from_slice(&f64::NAN.to_le_bytes()),
            "a start time that isn't finite",
        );
        refused(
            &|p| p[FIXED - 4..FIXED].copy_from_slice(&51_u32.to_le_bytes()),
            "a count the frame doesn't hold",
        );
        refused(
            &|p| p[FIXED - 4..FIXED].copy_from_slice(&u32::MAX.to_le_bytes()),
            "a count past the limit",
        );
        refused(&|p| p.truncate(p.len() - 1), "a short frame");
        refused(
            &|p| p.extend_from_slice(&compress(&[]).unwrap()),
            "an empty frame after the frame",
        );
        refused(
            &|p| p.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0]),
            "a skippable frame after the frame",
        );
        refused(&|p| p.truncate(FIXED - 1), "a short payload");
    }

    #[test]
    fn a_direction_with_no_tilt_is_refused() {
        let mut row = Quantized {
            tilt_amount: NO_TILT,
            tilt_direction: 3,
            ..Quantized::default()
        };
        assert_eq!(Quantized::from_fields(row.fields()), None, "not canonical");
        row.tilt_direction = 0;
        assert_eq!(Quantized::from_fields(row.fields()), Some(row), "canonical");
    }

    #[test]
    fn delta_coding_is_a_bijection_at_every_width() {
        for bits in [8, 16, 24, 32] {
            let top = u32::try_from(mask(bits)).unwrap();
            for (value, previous) in [(0, 0), (top, 0), (0, top), (5, 3), (3, 5), (top / 2, top)] {
                let zig = delta(value, previous, bits);
                assert!(
                    u64::from(zig) <= mask(bits),
                    "{bits} bits: within the field"
                );
                assert_eq!(undelta(zig, previous, bits), value, "{bits} bits: inverse");
            }
            assert_eq!(delta(6, 5, bits), 2, "{bits} bits: +1 is 2");
            assert_eq!(delta(4, 5, bits), 1, "{bits} bits: -1 is 1");
        }
    }
}
