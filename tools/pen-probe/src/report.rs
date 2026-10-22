//! The CSV, written and read back, and the printed summary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use input::{Buttons, Phase};

use crate::{Recording, Sample, Selection, Source};

/// The CSV header the probe writes.
const HEADER: &str = "source,received_s,dejittered_s,wakeup,x,y,contact,pressure,tilt_x,tilt_y,\
                      distance,roll,phase,buttons,device_ms";

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
    /// Seconds from `a` to `b` on this clock, as sample spacing: `None` if either sample lacks
    /// the clock, or if the gap is [`STROKE_GAP_S`] or more, a break rather than spacing. Every
    /// metric reads spacing through this.
    fn step(self, a: &Sample, b: &Sample) -> Option<f64> {
        let dt = match self {
            Self::Receipt => b.received.saturating_sub(a.received).as_secs_f64(),
            Self::Dejittered => b.dejittered?.saturating_sub(a.dejittered?).as_secs_f64(),
            Self::Steady(period) => period,
        };
        (dt < STROKE_GAP_S).then_some(dt)
    }
}

/// Writes every sample, one row each.
///
/// # Errors
///
/// If the file can't be written.
pub(crate) fn write_csv(samples: &[Sample], path: &Path) -> Result<()> {
    std::fs::write(path, to_csv(samples)?).with_context(|| format!("writing {}", path.display()))
}

/// The CSV text for `samples`.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
fn to_csv(samples: &[Sample]) -> Result<String, fmt::Error> {
    fn opt<T: fmt::Display>(v: Option<T>) -> String {
        v.map(|v| v.to_string()).unwrap_or_default()
    }
    let mut out = format!("{HEADER}\n");
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
    Ok(out)
}

/// Reads back a CSV the probe wrote, for `--report`.
///
/// Pad events, drops and samples over other surfaces aren't in the CSV, so the recording leaves
/// them unknown; the selection is the sources that have samples.
///
/// # Errors
///
/// If the file can't be read or isn't a probe CSV.
pub(crate) fn read_csv(path: &Path) -> Result<Recording> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let samples = parse_csv(&text).with_context(|| format!("parsing {}", path.display()))?;
    let has = |source| samples.iter().any(|s: &Sample| s.source == source);
    let selection = match (has(Source::Winit), has(Source::Ours)) {
        (true, true) => Selection::Both,
        (true, false) => Selection::Winit,
        (false, _) => Selection::Ours,
    };
    Ok(Recording {
        selection,
        samples,
        ours_counts: None,
        note: None,
    })
}

/// Where each field is in a CSV row; `buttons` is missing from older CSVs.
#[derive(Debug)]
struct Columns {
    source: usize,
    received: usize,
    dejittered: usize,
    wakeup: usize,
    x: usize,
    y: usize,
    contact: usize,
    pressure: usize,
    tilt_x: usize,
    tilt_y: usize,
    distance: usize,
    roll: usize,
    phase: usize,
    buttons: Option<usize>,
    device_ms: usize,
}

impl Columns {
    /// Finds the columns by name in `header`.
    ///
    /// # Errors
    ///
    /// If a column other than `buttons` is missing.
    fn find(header: &str) -> Result<Self> {
        let names: Vec<&str> = header.split(',').map(str::trim).collect();
        let column = |name: &str| names.iter().position(|n| *n == name);
        let required = |name: &str| {
            column(name).with_context(|| format!("no {name} column; expected the header {HEADER}"))
        };
        Ok(Self {
            source: required("source")?,
            received: required("received_s")?,
            dejittered: required("dejittered_s")?,
            wakeup: required("wakeup")?,
            x: required("x")?,
            y: required("y")?,
            contact: required("contact")?,
            pressure: required("pressure")?,
            tilt_x: required("tilt_x")?,
            tilt_y: required("tilt_y")?,
            distance: required("distance")?,
            roll: required("roll")?,
            phase: required("phase")?,
            buttons: column("buttons"),
            device_ms: required("device_ms")?,
        })
    }

    /// Parses one row. Missing or empty fields are `None`, as the writer leaves them.
    ///
    /// # Errors
    ///
    /// If a field doesn't parse, or one every sample has is missing.
    fn sample(&self, row: &str) -> Result<Sample> {
        let fields: Vec<&str> = row.split(',').map(str::trim).collect();
        let field = |i: usize| fields.get(i).copied().filter(|f| !f.is_empty());
        let required = |i: usize, name: &str| field(i).with_context(|| format!("no {name}"));
        let seconds = |i: usize, name: &str| -> Result<Option<Duration>> {
            parsed::<f64>(field(i), name)?
                .map(|s| Duration::try_from_secs_f64(s).with_context(|| format!("bad {name} {s}")))
                .transpose()
        };
        let source = match required(self.source, "source")? {
            "winit" => Source::Winit,
            "ours" => Source::Ours,
            other => bail!("unknown source {other}"),
        };
        let phase = field(self.phase)
            .map(|p| parse_phase(p).with_context(|| format!("unknown phase {p}")))
            .transpose()?;
        let tilt = parsed(field(self.tilt_x), "tilt_x")?.zip(parsed(field(self.tilt_y), "tilt_y")?);
        let mask: Option<u8> = parsed(self.buttons.and_then(field), "buttons")?;
        Ok(Sample {
            source,
            received: seconds(self.received, "received_s")?.context("no received_s")?,
            dejittered: seconds(self.dejittered, "dejittered_s")?,
            wakeup: parsed(field(self.wakeup), "wakeup")?.context("no wakeup")?,
            x: parsed(field(self.x), "x")?.context("no x")?,
            y: parsed(field(self.y), "y")?.context("no y")?,
            contact: parsed::<u8>(field(self.contact), "contact")?.context("no contact")? != 0,
            pressure: parsed(field(self.pressure), "pressure")?,
            tilt,
            distance: parsed(field(self.distance), "distance")?,
            roll: parsed(field(self.roll), "roll")?,
            phase,
            buttons: buttons_from_mask(mask.unwrap_or(0)),
            device_ms: parsed(field(self.device_ms), "device_ms")?,
        })
    }
}

/// `field` parsed as a `T`, if it's there.
///
/// # Errors
///
/// If it's there and doesn't parse.
fn parsed<T>(field: Option<&str>, name: &str) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    field
        .map(|f| f.parse().with_context(|| format!("bad {name} {f:?}")))
        .transpose()
}

/// Parses the samples from CSV text the probe wrote, with or without the `buttons` column.
///
/// # Errors
///
/// If the header lacks a column or a row doesn't parse; the message names the line.
fn parse_csv(text: &str) -> Result<Vec<Sample>> {
    let mut lines = text.lines();
    let columns = Columns::find(lines.next().context("the file is empty")?)?;
    lines
        .zip(2_usize..)
        .filter(|(row, _)| !row.trim().is_empty())
        .map(|(row, line)| columns.sample(row).with_context(|| format!("line {line}")))
        .collect()
}

/// The phase `name` as the CSV writes it (its `Debug` form).
fn parse_phase(name: &str) -> Option<Phase> {
    [
        Phase::Enter,
        Phase::Hover,
        Phase::Down,
        Phase::Drag,
        Phase::Up,
        Phase::Leave,
    ]
    .into_iter()
    .find(|p| format!("{p:?}") == name)
}

/// The inverse of [`button_mask`]; unknown bits are ignored.
fn buttons_from_mask(mask: u8) -> Buttons {
    [Buttons::PRIMARY, Buttons::SECONDARY, Buttons::TERTIARY]
        .into_iter()
        .zip([1, 2, 4])
        .fold(Buttons::default(), |held, (button, bit)| {
            held.with(button, mask & bit != 0)
        })
}

/// In-contact samples grouped into strokes; `samples` must all come from one source.
///
/// A stroke is a run of in-contact samples, and where there are phases, a `Down` always starts a
/// new one, so an `Up` followed directly by a `Down` ends one stroke and starts the next.
fn strokes<'s, 'a>(samples: &'s [&'a Sample]) -> Vec<&'s [&'a Sample]> {
    samples
        .chunk_by(|a, b| a.contact == b.contact && b.phase != Some(Phase::Down))
        .filter(|run| run.iter().all(|s| s.contact))
        .collect()
}

/// Consecutive `(sample, next)` pairs within strokes.
fn steps<'a>(strokes: &[&[&'a Sample]]) -> impl Iterator<Item = (&'a Sample, &'a Sample)> {
    strokes
        .iter()
        .flat_map(|stroke| stroke.iter().copied().zip(stroke.iter().copied().skip(1)))
}

/// Sorted intervals on `clock` between consecutive samples within strokes.
fn intervals(strokes: &[&[&Sample]], clock: Clock) -> Vec<f64> {
    let mut dts: Vec<f64> = steps(strokes)
        .filter_map(|(a, b)| clock.step(a, b))
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
/// timing noise in `clock`. Steps across a break (see [`Clock::step`]) have no speed. `None` if no
/// stroke is long enough.
fn speed_noise(strokes: &[&[&Sample]], clock: Clock) -> Option<f64> {
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
    samples: &[&Sample],
    value: impl Fn(&Sample) -> Option<String>,
) -> fmt::Result {
    let values: Vec<String> = samples.iter().copied().filter_map(value).collect();
    let distinct = values.iter().collect::<BTreeSet<_>>().len();
    writeln!(
        out,
        "{label}: {} of {} samples, {distinct} distinct",
        values.len(),
        samples.len()
    )
}

/// Speed noise on `clock`, formatted.
fn noise(strokes: &[&[&Sample]], clock: Clock) -> String {
    speed_noise(strokes, clock).map_or_else(|| "n/a".to_owned(), |v| format!("{v:.1}%"))
}

/// Appends the block for one source's samples.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
fn source_block(out: &mut String, source: Source, samples: &[&Sample]) -> fmt::Result {
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

    let in_contact: Vec<&Sample> = strokes.iter().flat_map(|s| s.iter().copied()).collect();
    coverage(out, "pressure", &in_contact, |s| {
        s.pressure.map(|p| format!("{p:.6}"))
    })?;
    // winit reports tilt in whole degrees, ours in fractions: compare them at whole degrees.
    if source == Source::Ours {
        coverage(out, "tilt", samples, |s| {
            s.tilt.map(|(x, y)| format!("{x:.3},{y:.3}"))
        })?;
    }
    coverage(out, "tilt (whole degrees)", samples, |s| {
        // Adding 0.0 turns -0.0 into 0.0, so the two print alike.
        s.tilt
            .map(|(x, y)| format!("{:.0},{:.0}", x.round() + 0.0, y.round() + 0.0))
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
        let mut phases = BTreeMap::<&'static str, usize>::new();
        for phase in samples.iter().filter_map(|s| s.phase) {
            *phases.entry(phase_name(phase)).or_default() += 1;
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

/// A human-readable summary of the recording: a block per selected source, even one with no
/// samples.
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
    for source in Source::ALL
        .into_iter()
        .filter(|&s| recording.selection.records(s))
    {
        let samples: Vec<&Sample> = recording
            .samples
            .iter()
            .filter(|s| s.source == source)
            .collect();
        source_block(&mut out, source, &samples)?;
        if source == Source::Ours {
            match recording.ours_counts {
                Some(counts) => writeln!(
                    out,
                    "pad events: {}; dropped: {}; samples over other surfaces: {}",
                    counts.pads, counts.dropped, counts.elsewhere
                )?,
                None => writeln!(
                    out,
                    "pad events, dropped, samples over other surfaces: unknown (not in the CSV)"
                )?,
            }
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

    /// An in-stroke sample of ours at `ms` milliseconds and `x` pixels, on both clocks.
    fn stroke_sample(ms: u64, x: f64, phase: Phase) -> Sample {
        let at = Duration::from_millis(ms);
        Sample {
            received: at,
            dejittered: Some(at),
            x,
            contact: phase.in_stroke(),
            phase: Some(phase),
            ..sample(Source::Ours, Buttons::default())
        }
    }

    /// A stroke of `n` samples `period_ms` apart, moving `step_px` each: `Down`, `Drag`s, `Up`.
    fn stroke(start_ms: u64, n: u64, period_ms: u64, step_px: f64) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let phase = match i {
                    0 => Phase::Down,
                    i if i + 1 == n => Phase::Up,
                    _ => Phase::Drag,
                };
                let x = step_px * f64::from(u32::try_from(i).expect("small"));
                stroke_sample(start_ms + i * period_ms, x, phase)
            })
            .collect()
    }

    fn refs(samples: &[Sample]) -> Vec<&Sample> {
        samples.iter().collect()
    }

    #[test]
    fn percentile_picks_the_nearest_rank() {
        assert!(
            (percentile(&[5.0], 0.5) - 5.0).abs() < f64::EPSILON,
            "one element"
        );
        // An even count: the median rounds half up to the upper middle element.
        let even = [1.0, 2.0, 3.0, 4.0];
        assert!(
            (percentile(&even, 0.5) - 3.0).abs() < f64::EPSILON,
            "median"
        );
        assert!((percentile(&even, 0.0) - 1.0).abs() < f64::EPSILON, "p0");
        assert!((percentile(&even, 1.0) - 4.0).abs() < f64::EPSILON, "p100");
        assert!(percentile(&[], 0.5).is_nan(), "empty");
    }

    #[test]
    fn std_dev_of_constant_values_is_zero() {
        assert!(std_dev(&[0.004; 5]).abs() < 1e-15, "constant");
        assert!((std_dev(&[1.0, 3.0]) - 1.0).abs() < 1e-12, "population sd");
    }

    #[test]
    fn a_constant_period_stroke_has_no_spread() {
        let samples = stroke(0, 20, 4, 3.0);
        let samples = refs(&samples);
        let strokes = strokes(&samples);
        for clock in [Clock::Receipt, Clock::Dejittered] {
            let dts = intervals(&strokes, clock);
            assert_eq!(dts.len(), 19, "every step counts");
            assert!(std_dev(&dts) < 1e-12, "sd {}", std_dev(&dts));
            assert!(
                (1.0 / mean(&dts) - 250.0).abs() < 1e-6,
                "Hz {}",
                1.0 / mean(&dts)
            );
        }
        let mut out = String::new();
        timing(&mut out, "receipt", &intervals(&strokes, Clock::Receipt)).expect("String");
        assert!(out.starts_with("receipt: 250 Hz;"), "{out}");
        assert!(out.ends_with("sd 0.00\n"), "{out}");
    }

    #[test]
    fn constant_speed_has_no_speed_noise() {
        let samples = stroke(0, 20, 4, 3.0);
        let samples = refs(&samples);
        let strokes = strokes(&samples);
        let steady = speed_noise(&strokes, Clock::Steady(0.004)).expect("long enough");
        assert!(steady.abs() < 1e-9, "steady {steady}");
        let receipt = speed_noise(&strokes, Clock::Receipt).expect("long enough");
        assert!(receipt.abs() < 1e-9, "receipt {receipt}");
    }

    #[test]
    fn a_gap_of_a_stroke_break_or_more_is_not_spacing() {
        // Four samples, then a 150 ms pause in contact, then four more.
        let mut samples = stroke(0, 4, 4, 3.0);
        samples.extend(stroke(162, 4, 4, 3.0).into_iter().map(|mut s| {
            s.phase = Some(Phase::Drag);
            s.x += 12.0;
            s
        }));
        let samples = refs(&samples);
        let strokes = strokes(&samples);
        assert_eq!(
            strokes.len(),
            1,
            "a pause in contact doesn't end the stroke"
        );
        let dts = intervals(&strokes, Clock::Receipt);
        assert_eq!(dts.len(), 6, "the gap is left out");
        assert!(dts.iter().all(|&dt| dt < STROKE_GAP_S), "{dts:?}");
        // Each 3-step speed window spans the gap or an end, so none has a speed.
        assert_eq!(
            speed_noise(&strokes, Clock::Receipt),
            None,
            "the gap has no speed"
        );
    }

    #[test]
    fn back_to_back_strokes_split_on_down() {
        let mut samples = stroke(0, 3, 4, 3.0);
        samples.extend(stroke(12, 3, 4, 3.0));
        let phases: Vec<Option<Phase>> = samples.iter().map(|s| s.phase).collect();
        assert_eq!(
            phases,
            [
                Phase::Down,
                Phase::Drag,
                Phase::Up,
                Phase::Down,
                Phase::Drag,
                Phase::Up
            ]
            .map(Some),
            "the fixture"
        );
        let samples = refs(&samples);
        let strokes = strokes(&samples);
        assert_eq!(strokes.len(), 2, "two strokes");
        assert!(strokes.iter().all(|s| s.len() == 3), "three samples each");
    }

    #[test]
    fn strokes_without_phases_split_on_contact() {
        let mut samples = stroke(0, 9, 4, 3.0);
        for (i, s) in samples.iter_mut().enumerate() {
            s.phase = None;
            s.contact = i != 4;
        }
        assert_eq!(strokes(&refs(&samples)).len(), 2, "split at the hover");
    }

    #[test]
    fn every_selected_source_gets_a_block() {
        let recording = Recording {
            selection: Selection::Both,
            ours_counts: Some(crate::OursCounts {
                elsewhere: 7,
                ..crate::OursCounts::default()
            }),
            ..Recording::default()
        };
        let summary = summarise(&recording).expect("writing to a String can't fail");
        assert!(summary.contains("[winit]\nsamples: 0 total"), "{summary}");
        assert!(summary.contains("[ours]\nsamples: 0 total"), "{summary}");
        assert!(
            summary.contains("samples over other surfaces: 7"),
            "{summary}"
        );
    }

    #[test]
    fn a_csv_round_trips_to_the_same_summary() {
        let mut samples = vec![Sample {
            phase: Some(Phase::Hover),
            tilt: Some((-14.074, 0.5)),
            distance: Some(0.25),
            device_ms: Some(1000),
            ..stroke_sample(0, 0.0, Phase::Hover)
        }];
        // Receipt jitter of a millisecond on every other sample, so the noise isn't all zero.
        samples.extend(
            stroke(4, 12, 4, 3.0)
                .into_iter()
                .zip(0_u64..)
                .map(|(s, i)| Sample {
                    received: s.received + Duration::from_millis(i % 2),
                    pressure: Some(0.5),
                    tilt: Some((12.5, -3.25)),
                    roll: Some(90.0),
                    buttons: Buttons::SECONDARY,
                    device_ms: Some(1004),
                    ..s
                }),
        );
        samples.extend(stroke(60, 8, 5, 2.5).into_iter().map(|s| Sample {
            source: Source::Winit,
            dejittered: None,
            phase: None,
            pressure: Some(0.25),
            tilt: Some((20.0, 0.0)),
            ..s
        }));
        let original = Recording {
            selection: Selection::Both,
            samples,
            ..Recording::default()
        };
        let csv = to_csv(&original.samples).expect("writing to a String can't fail");
        let parsed = parse_csv(&csv).expect("the probe's own CSV parses");
        assert_eq!(parsed.len(), original.samples.len(), "every row");
        let read_back = Recording {
            selection: Selection::Both,
            samples: parsed,
            ..Recording::default()
        };
        let summary = |r: &Recording| summarise(r).expect("writing to a String can't fail");
        assert_eq!(summary(&read_back), summary(&original), "same summary");
        let text = summary(&original);
        assert!(text.contains("8 in contact, 1 strokes"), "{text}");
        assert!(text.contains("12 in contact, 1 strokes"), "{text}");
        assert!(!text.contains("receipt 0.0%, de-jittered"), "{text}");
    }

    #[test]
    fn reads_the_older_csv_without_buttons_and_empty_fields_as_none() {
        let csv = "source,received_s,dejittered_s,wakeup,x,y,contact,pressure,tilt_x,tilt_y,\
                   distance,roll,phase,device_ms\n\
                   ours,7346.041406,,163,1126.371,558.885,0,0.000000,-14.074,0.000,,,Enter,39534843\n\
                   winit,1.5,,2,3,4,1,,,,,,\n";
        let samples = parse_csv(csv).expect("the older header parses");
        let [ours, winit] = samples.as_slice() else {
            panic!("two rows, got {samples:?}");
        };
        assert_eq!(ours.phase, Some(Phase::Enter), "phase");
        assert_eq!(ours.tilt, Some((-14.074, 0.0)), "tilt");
        assert_eq!(ours.dejittered, None, "empty is None");
        assert_eq!(ours.buttons, Buttons::default(), "no buttons column");
        assert_eq!(ours.device_ms, Some(39_534_843), "device ms");
        assert_eq!(winit.source, Source::Winit, "source");
        assert!(winit.contact, "contact");
        assert_eq!(
            (winit.pressure, winit.tilt, winit.phase),
            (None, None, None),
            "missing"
        );
        assert_eq!(winit.device_ms, None, "a short row is None");
    }

    #[test]
    fn rejects_a_foreign_csv() {
        parse_csv("received_s,device_s,wakeup,x,y\n1,2,3,4,5\n").unwrap_err();
        parse_csv(&format!("{HEADER}\nmouse,1,,1,1,1,0,,,,,,,0,\n")).unwrap_err();
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
        for mask in 0..8 {
            assert_eq!(button_mask(buttons_from_mask(mask)), mask, "mask {mask}");
        }
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
