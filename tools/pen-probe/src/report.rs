//! CSV output and the printed summary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::path::Path;

use anyhow::{Context as _, Result};

use crate::Sample;

/// Intervals longer than this are breaks between strokes, not sample spacing.
const STROKE_GAP_S: f64 = 0.1;

/// Below this many pixels per sample, position quantization dominates the speed estimate.
const MIN_STEP_PX: f64 = 2.0;

#[expect(clippy::cast_precision_loss, reason = "counts here are far below 2^52")]
pub(crate) const fn to_f64(v: usize) -> f64 {
    v as f64
}

/// Which clock a sample's time is read from.
#[derive(Debug, Clone, Copy)]
enum Clock {
    /// When the application received the event.
    Receipt,
    /// The platform's own timestamp, if it reports one.
    Device,
}

impl Clock {
    fn seconds(self, sample: &Sample) -> Option<f64> {
        match self {
            Self::Receipt => Some(sample.received.as_secs_f64()),
            Self::Device => sample.data.timestamp.map(|t| t.as_secs_f64()),
        }
    }
}

/// Writes every sample, one row each.
///
/// # Errors
///
/// If the file can't be written.
pub(crate) fn write_csv(samples: &[Sample], path: &Path) -> Result<()> {
    fn opt<T: fmt::Display>(v: Option<T>) -> String {
        v.map(|v| v.to_string()).unwrap_or_default()
    }
    let mut out = String::from(
        "received_s,device_s,wakeup,x,y,contact,force,distance,tilt_x,tilt_y,altitude,azimuth,\
         twist,tangential\n",
    );
    for s in samples {
        let d = &s.data;
        writeln!(
            out,
            "{:.6},{},{},{:.3},{:.3},{},{},{},{},{},{},{},{},{}",
            s.received.as_secs_f64(),
            opt(d.timestamp.map(|t| format!("{:.3}", t.as_secs_f64()))),
            s.wakeup,
            s.x,
            s.y,
            u8::from(s.contact),
            opt(d.force.map(|f| format!("{:.6}", f.normalized(None)))),
            opt(d.distance.map(|v| format!("{v:.6}"))),
            opt(d.tilt.map(|t| t.x)),
            opt(d.tilt.map(|t| t.y)),
            opt(d.angle.map(|a| format!("{:.6}", a.altitude))),
            opt(d.angle.map(|a| format!("{:.6}", a.azimuth))),
            opt(d.twist),
            opt(d.tangential_force),
        )?;
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// In-contact samples grouped into strokes.
fn strokes(samples: &[Sample]) -> Vec<&[Sample]> {
    samples
        .chunk_by(|a, b| a.contact == b.contact)
        .filter(|run| run.iter().all(|s| s.contact))
        .collect()
}

/// Consecutive `(sample, next)` pairs within strokes.
fn steps<'a>(strokes: &[&'a [Sample]]) -> impl Iterator<Item = (&'a Sample, &'a Sample)> {
    strokes
        .iter()
        .flat_map(|stroke| stroke.iter().zip(stroke.iter().skip(1)))
}

fn intervals(strokes: &[&[Sample]], clock: Clock) -> Vec<f64> {
    let mut dts: Vec<f64> = steps(strokes)
        .filter_map(|(a, b)| Some(clock.seconds(b)? - clock.seconds(a)?))
        .filter(|dt| *dt < STROKE_GAP_S)
        .collect();
    dts.sort_by(f64::total_cmp);
    dts
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "p is in 0..=1, so the index is in 0..len"
    )]
    let i = (p * to_f64(sorted.len().saturating_sub(1))).round() as usize;
    sorted.get(i).copied().unwrap_or(f64::NAN)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / to_f64(values.len())
}

fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / to_f64(values.len())).sqrt()
}

/// Median relative change between consecutive speeds (each a 3-step moving average), in percent.
///
/// The pen's real speed changes little from one sample to the next, so this mostly measures
/// timing noise in `clock`. `None` if `clock` has no times.
fn speed_noise(strokes: &[&[Sample]], clock: Clock) -> Option<f64> {
    let mut changes = Vec::new();
    for stroke in strokes {
        let speeds = stroke
            .windows(2)
            .map(|pair| match pair {
                [a, b] => {
                    let dt = clock.seconds(b)? - clock.seconds(a)?;
                    let step = (b.x - a.x).hypot(b.y - a.y);
                    Some((dt > 0.0 && step >= MIN_STEP_PX).then(|| step / dt))
                }
                _ => Some(None),
            })
            .collect::<Option<Vec<_>>>()?;
        let smoothed: Vec<Option<f64>> = speeds
            .windows(3)
            .map(|w| w.iter().copied().sum::<Option<f64>>().map(|sum| sum / 3.0))
            .collect();
        changes.extend(smoothed.windows(2).filter_map(|pair| match pair {
            [Some(a), Some(b)] => Some((b - a).abs() / a),
            _ => None,
        }));
    }
    changes.sort_by(f64::total_cmp);
    (!changes.is_empty()).then(|| percentile(&changes, 0.5) * 100.0)
}

/// Appends the sample rate and interval spread for one clock.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
fn timing(out: &mut String, label: &str, dts: &[f64]) -> fmt::Result {
    if dts.is_empty() {
        return writeln!(out, "{label}: none");
    }
    let ms = |v: f64| v * 1000.0;
    writeln!(
        out,
        "{label}: {:.0} Hz; interval ms p5 {:.2} / median {:.2} / p95 {:.2} / sd {:.2}",
        1.0 / mean(dts),
        ms(percentile(dts, 0.05)),
        ms(percentile(dts, 0.5)),
        ms(percentile(dts, 0.95)),
        ms(std_dev(dts)),
    )
}

/// Appends how many samples carry a value, and how many distinct values there are.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
fn coverage(
    out: &mut String,
    label: &str,
    samples: &[Sample],
    value: impl Fn(&Sample) -> Option<String>,
) -> fmt::Result {
    let values: Vec<String> = samples.iter().filter_map(value).collect();
    let distinct = values.iter().collect::<BTreeSet<_>>().len();
    writeln!(
        out,
        "{label}: {} of {} samples, {distinct} distinct",
        values.len(),
        samples.len()
    )
}

/// A human-readable summary of the recording.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(crate) fn summarise(samples: &[Sample]) -> Result<String, fmt::Error> {
    let mut out = String::new();
    let strokes = strokes(samples);
    let contact: usize = strokes.iter().map(|s| s.len()).sum();
    writeln!(
        out,
        "samples: {} total, {contact} in contact, {} strokes",
        samples.len(),
        strokes.len()
    )?;

    timing(
        &mut out,
        "receipt clock",
        &intervals(&strokes, Clock::Receipt),
    )?;
    timing(
        &mut out,
        "device clock",
        &intervals(&strokes, Clock::Device),
    )?;
    let noise = |clock| {
        speed_noise(&strokes, clock).map_or_else(|| "n/a".to_owned(), |v| format!("{v:.1}%"))
    };
    writeln!(
        out,
        "speed noise (median sample-to-sample change): receipt clock {}, device clock {}",
        noise(Clock::Receipt),
        noise(Clock::Device),
    )?;

    let mut per_wakeup = BTreeMap::<u64, usize>::new();
    for sample in strokes.iter().flat_map(|s| s.iter()) {
        *per_wakeup.entry(sample.wakeup).or_default() += 1;
    }
    let multi = per_wakeup.values().filter(|&&n| n > 1).count();
    let max = per_wakeup.values().max().copied().unwrap_or(0);
    writeln!(
        out,
        "batching: {} wakeups, {multi} with >1 sample, max {max} per wakeup",
        per_wakeup.len()
    )?;

    let in_contact: Vec<Sample> = strokes.iter().flat_map(|s| s.iter()).cloned().collect();
    coverage(&mut out, "pressure", &in_contact, |s| {
        s.data.force.map(|f| format!("{:.6}", f.normalized(None)))
    })?;
    coverage(&mut out, "tilt", samples, |s| {
        s.data.tilt.map(|t| format!("{},{}", t.x, t.y))
    })?;
    coverage(&mut out, "angle", samples, |s| {
        s.data
            .angle
            .map(|a| format!("{:.6},{:.6}", a.altitude, a.azimuth))
    })?;
    coverage(&mut out, "hover distance", samples, |s| {
        s.data.distance.map(|v| format!("{v:.6}"))
    })?;
    coverage(&mut out, "twist", samples, |s| {
        s.data.twist.map(|v| v.to_string())
    })?;
    coverage(&mut out, "tangential", samples, |s| {
        s.data.tangential_force.map(|v| v.to_string())
    })?;
    Ok(out)
}
