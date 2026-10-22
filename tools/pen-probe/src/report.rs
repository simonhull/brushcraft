//! CSV output and the printed summary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::path::Path;

use anyhow::{Context as _, Result};
use input::{Buttons, Phase};

use crate::{Recording, Sample, Selection, Source};

/// Intervals longer than this are breaks between strokes, not sample spacing.
const STROKE_GAP_S: f64 = 0.1;

/// Below this many pixels per sample, position quantization dominates the speed estimate.
const MIN_STEP_PX: f64 = 2.0;

#[expect(clippy::cast_precision_loss, reason = "counts here are far below 2^52")]
pub(crate) const fn to_f64(v: usize) -> f64 {
    v as f64
}

/// Held barrel buttons as a bit mask: 1 primary (`BTN_STYLUS`), 2 secondary (`BTN_STYLUS2`),
/// 4 tertiary (`BTN_STYLUS3`); 0 for none.
fn button_mask(buttons: Buttons) -> u8 {
    [Buttons::PRIMARY, Buttons::SECONDARY, Buttons::TERTIARY]
        .into_iter()
        .zip([1, 2, 4])
        .filter(|&(button, _)| buttons.contains(button))
        .map(|(_, bit)| bit)
        .sum()
}

/// How the time between two consecutive samples is read.
#[derive(Debug, Clone, Copy)]
enum Clock {
    /// When the application received each event.
    Receipt,
    /// Our de-jitter filter's recovered sample times; only our in-stroke samples carry them.
    Dejittered,
    /// A pen sampling at a steady period (seconds): the model a de-jitter filter fits.
    Steady(f64),
}

impl Clock {
    /// Seconds from `a` to `b` on this clock; `None` if either sample lacks it.
    fn step(self, a: &Sample, b: &Sample) -> Option<f64> {
        match self {
            Self::Receipt => Some(b.received.saturating_sub(a.received).as_secs_f64()),
            Self::Dejittered => Some(b.dejittered?.saturating_sub(a.dejittered?).as_secs_f64()),
            Self::Steady(period) => Some(period),
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
        "source,received_s,dejittered_s,wakeup,x,y,contact,pressure,tilt_x,tilt_y,distance,roll,\
         phase,buttons,device_ms\n",
    );
    for s in samples {
        writeln!(
            out,
            "{},{:.6},{},{},{:.3},{:.3},{},{},{},{},{},{},{},{},{}",
            s.source.name(),
            s.received.as_secs_f64(),
            opt(s.dejittered.map(|d| format!("{:.6}", d.as_secs_f64()))),
            s.wakeup,
            s.x,
            s.y,
            u8::from(s.contact),
            opt(s.pressure.map(|p| format!("{p:.6}"))),
            opt(s.tilt.map(|t| format!("{:.3}", t.0))),
            opt(s.tilt.map(|t| format!("{:.3}", t.1))),
            opt(s.distance.map(|d| format!("{d:.6}"))),
            opt(s.roll.map(|r| format!("{r:.3}"))),
            opt(s.phase.map(|p| format!("{p:?}"))),
            button_mask(s.buttons),
            opt(s.device_ms),
        )?;
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// In-contact samples grouped into strokes; `samples` must all come from one source.
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

/// Sorted intervals on `clock` between consecutive samples within strokes.
fn intervals(strokes: &[&[Sample]], clock: Clock) -> Vec<f64> {
    let mut dts: Vec<f64> = steps(strokes)
        .filter_map(|(a, b)| clock.step(a, b))
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
/// timing noise in `clock`. `None` if no stroke is long enough.
fn speed_noise(strokes: &[&[Sample]], clock: Clock) -> Option<f64> {
    let mut changes = Vec::new();
    for stroke in strokes {
        let speeds: Vec<Option<f64>> = stroke
            .windows(2)
            .map(|pair| match pair {
                [a, b] => {
                    let dt = clock.step(a, b)?;
                    let step = (b.x - a.x).hypot(b.y - a.y);
                    (dt > 0.0 && step >= MIN_STEP_PX).then(|| step / dt)
                }
                _ => None,
            })
            .collect();
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

/// Speed noise on `clock`, formatted.
fn noise(strokes: &[&[Sample]], clock: Clock) -> String {
    speed_noise(strokes, clock).map_or_else(|| "n/a".to_owned(), |v| format!("{v:.1}%"))
}

/// Appends the block for one source's samples.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
fn source_block(out: &mut String, source: Source, samples: &[Sample]) -> fmt::Result {
    writeln!(out, "[{}]", source.name())?;
    let strokes = strokes(samples);
    let contact: usize = strokes.iter().map(|s| s.len()).sum();
    writeln!(
        out,
        "samples: {} total, {contact} in contact, {} strokes",
        samples.len(),
        strokes.len()
    )?;

    let receipt = intervals(&strokes, Clock::Receipt);
    timing(out, "receipt clock", &receipt)?;
    let steady = Clock::Steady(percentile(&receipt, 0.5));
    if source == Source::Ours {
        timing(
            out,
            "de-jittered clock",
            &intervals(&strokes, Clock::Dejittered),
        )?;
        writeln!(
            out,
            "speed noise (median sample-to-sample change): receipt {}, de-jittered {}, steady \
             period {}",
            noise(&strokes, Clock::Receipt),
            noise(&strokes, Clock::Dejittered),
            noise(&strokes, steady),
        )?;
    } else {
        writeln!(
            out,
            "speed noise (median sample-to-sample change): receipt {}, steady period {}",
            noise(&strokes, Clock::Receipt),
            noise(&strokes, steady),
        )?;
    }

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
    coverage(out, "pressure", &in_contact, |s| {
        s.pressure.map(|p| format!("{p:.6}"))
    })?;
    coverage(out, "tilt", samples, |s| {
        s.tilt.map(|(x, y)| format!("{x:.3},{y:.3}"))
    })?;
    coverage(out, "distance", samples, |s| {
        s.distance.map(|d| format!("{d:.6}"))
    })?;
    coverage(out, "roll", samples, |s| s.roll.map(|r| format!("{r:.3}")))?;

    let held: Vec<u8> = samples
        .iter()
        .map(|s| button_mask(s.buttons))
        .filter(|&mask| mask != 0)
        .collect();
    let sets: Vec<String> = held
        .iter()
        .collect::<BTreeSet<_>>()
        .iter()
        .map(ToString::to_string)
        .collect();
    writeln!(
        out,
        "buttons: {} of {} samples with any held; sets seen (mask): {}",
        held.len(),
        samples.len(),
        if sets.is_empty() {
            "none".to_owned()
        } else {
            sets.join(", ")
        }
    )?;

    if source == Source::Ours {
        let mut phases = BTreeMap::<String, usize>::new();
        for phase in samples.iter().filter_map(|s| s.phase) {
            *phases.entry(phase_name(phase).to_owned()).or_default() += 1;
        }
        let phases: Vec<String> = phases.iter().map(|(p, n)| format!("{p} {n}")).collect();
        writeln!(out, "phases: {}", phases.join(", "))?;
    }
    Ok(())
}

const fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Enter => "enter",
        Phase::Hover => "hover",
        Phase::Down => "down",
        Phase::Drag => "drag",
        Phase::Up => "up",
        Phase::Leave => "leave",
    }
}

/// A human-readable summary of the recording: a block per source.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(crate) fn summarise(recording: &Recording) -> Result<String, fmt::Error> {
    let mut out = String::new();
    if let Some(note) = &recording.note {
        writeln!(out, "note: {note}")?;
    }
    if let Some(silent) = silent_source(recording) {
        writeln!(
            out,
            "note: {} recorded nothing; only one tablet binding per client may receive events on \
             this compositor",
            silent.name()
        )?;
    }
    for source in Source::ALL {
        let samples: Vec<Sample> = recording
            .samples
            .iter()
            .filter(|s| s.source == source)
            .cloned()
            .collect();
        let ran = source == Source::Ours && (recording.pads > 0 || recording.dropped > 0);
        if samples.is_empty() && !ran {
            continue;
        }
        source_block(&mut out, source, &samples)?;
        if source == Source::Ours {
            writeln!(
                out,
                "pad events: {}; dropped: {}; samples over other surfaces: {}",
                recording.pads, recording.dropped, recording.elsewhere
            )?;
        }
    }
    Ok(out)
}

/// With both sources recorded, the one that got no samples while the other got some.
fn silent_source(recording: &Recording) -> Option<Source> {
    if recording.selection != Selection::Both {
        return None;
    }
    let count = |source| {
        recording
            .samples
            .iter()
            .filter(|s| s.source == source)
            .count()
    };
    match (count(Source::Winit), count(Source::Ours)) {
        (0, 1..) => Some(Source::Winit),
        (1.., 0) => Some(Source::Ours),
        _ => None,
    }
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn sample(source: Source, buttons: Buttons) -> Sample {
        Sample {
            source,
            received: Duration::ZERO,
            dejittered: None,
            wakeup: 0,
            x: 0.0,
            y: 0.0,
            contact: false,
            pressure: None,
            tilt: None,
            distance: None,
            roll: None,
            phase: None,
            buttons,
            device_ms: None,
        }
    }

    #[test]
    fn button_masks_number_the_barrel_buttons() {
        assert_eq!(button_mask(Buttons::default()), 0);
        assert_eq!(button_mask(Buttons::PRIMARY), 1);
        assert_eq!(button_mask(Buttons::SECONDARY), 2);
        let all = Buttons::PRIMARY
            .with(Buttons::SECONDARY, true)
            .with(Buttons::TERTIARY, true);
        assert_eq!(button_mask(all), 7);
    }

    #[test]
    fn notes_a_silent_source_only_when_both_ran() {
        let mut recording = Recording {
            selection: Selection::Both,
            samples: vec![sample(Source::Ours, Buttons::default())],
            ..Recording::default()
        };
        assert_eq!(silent_source(&recording), Some(Source::Winit));
        recording
            .samples
            .push(sample(Source::Winit, Buttons::default()));
        assert_eq!(silent_source(&recording), None);
        recording.selection = Selection::Ours;
        recording.samples.retain(|s| s.source == Source::Ours);
        assert_eq!(silent_source(&recording), None);
    }

    #[test]
    fn summary_counts_held_buttons_and_their_sets() {
        let recording = Recording {
            samples: vec![
                sample(Source::Ours, Buttons::default()),
                sample(Source::Ours, Buttons::SECONDARY),
                sample(
                    Source::Ours,
                    Buttons::SECONDARY.with(Buttons::PRIMARY, true),
                ),
                sample(Source::Ours, Buttons::SECONDARY),
            ],
            ..Recording::default()
        };
        let summary = summarise(&recording).expect("writing to a String can't fail");
        assert!(
            summary.contains("buttons: 3 of 4 samples with any held; sets seen (mask): 2, 3"),
            "{summary}"
        );
    }
}
