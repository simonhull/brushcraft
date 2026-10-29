//! How late the main thread handles tablet events, and what the load did each frame.
//!
//! Three delays, each in milliseconds:
//!
//! - **receipt to handling** (`handled − received`): ours stamps a sample on its own thread and the
//!   main thread drains it later, so this is the main thread's dispatch delay for ours. A winit
//!   event is stamped as the main thread handles it, so for winit it is zero by construction: the
//!   wait happens before winit reads it, where the probe can't see it.
//! - **handled behind the pen's clock** (`handled − de-jittered`, contact samples): the de-jittered
//!   line is the lower envelope of the receipts, so this is how much later than the least-delayed
//!   samples the main thread got each one. It is the dispatch delay the decision reads, for both
//!   sources alike: what the main thread's busy stretches add to the delivery's own jitter.
//! - **receipt behind the pen's clock** (`received − de-jittered`): for ours, the compositor,
//!   socket and input-thread scheduling together, since our thread reads at once: the part of the
//!   handled figure the main thread didn't add. It isn't measured to be the compositor's alone.

use std::fmt::{self, Write as _};
use std::time::Duration;

use super::{percentile, to_f64};
use crate::frames::FrameRecord;
use crate::{Sample, Source};

/// A flat per-frame budget, in milliseconds: the load block counts the frames that keep the main
/// thread away longer, and the comparison prints W's p99 dispatch delay against it.
///
/// Information only: the decision no longer holds W to it. The dispatch wait comes from the frame
/// loop and delays U's samples reaching the brush as much as W's, so W is held to U's figure
/// instead ([`super::compare::DISPATCH_MARGIN_MS`]).
pub(crate) const BUDGET_MS: f64 = 8.0;

/// The spread of a set of delays, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spread {
    /// How many.
    pub(crate) n: usize,
    /// The median.
    pub(crate) p50: f64,
    /// The 99th percentile.
    pub(crate) p99: f64,
    /// The largest.
    pub(crate) max: f64,
}

impl Spread {
    /// The spread of `durations`; `None` if there are none.
    pub(crate) fn of(durations: impl IntoIterator<Item = Duration>) -> Option<Self> {
        let mut ms: Vec<f64> = durations
            .into_iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        ms.sort_by(f64::total_cmp);
        let max = *ms.last()?;
        Some(Self {
            n: ms.len(),
            p50: percentile(&ms, 0.5),
            p99: percentile(&ms, 0.99),
            max,
        })
    }
}

impl fmt::Display for Spread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p50 {:.2} / p99 {:.2} / max {:.2} ms (n {})",
            self.p50, self.p99, self.max, self.n
        )
    }
}

/// `Option<Spread>` for printing.
fn show(spread: Option<Spread>) -> String {
    spread.map_or_else(|| "n/a".to_owned(), |s| s.to_string())
}

/// Receipt to main-thread handling, over every event that records both.
pub(crate) fn handling(samples: &[&Sample]) -> Option<Spread> {
    Spread::of(
        samples
            .iter()
            .filter_map(|s| Some(s.handled?.saturating_sub(s.received))),
    )
}

/// Which time [`behind_clock`] measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moment {
    /// When the sample was received.
    Received,
    /// When the main thread handled it.
    Handled,
}

/// `moment` behind the de-jittered clock, over the contact samples that have both.
pub(crate) fn behind_clock(samples: &[&Sample], moment: Moment) -> Option<Spread> {
    Spread::of(samples.iter().filter_map(|s| {
        let at = match moment {
            Moment::Received => s.received,
            Moment::Handled => s.handled?,
        };
        Some(at.saturating_sub(s.dejittered?))
    }))
}

/// Appends the dispatch lines for one source's samples; nothing for a CSV from before handling
/// was recorded.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(super) fn block(out: &mut String, source: Source, samples: &[&Sample]) -> fmt::Result {
    if samples.iter().all(|s| s.handled.is_none()) {
        return Ok(());
    }
    match source {
        Source::Winit => writeln!(
            out,
            "dispatch: handled behind the pen's clock {} (winit events are stamped as the main \
             thread handles them, so receipt to handling is 0)",
            show(behind_clock(samples, Moment::Handled)),
        ),
        Source::Ours => {
            writeln!(
                out,
                "dispatch: handled behind the pen's clock {}",
                show(behind_clock(samples, Moment::Handled)),
            )?;
            writeln!(
                out,
                "  of which: receipt (our thread) to main-thread handling {}; receipt behind the \
                 pen's clock {} (compositor, socket and input-thread scheduling)",
                show(handling(samples)),
                show(behind_clock(samples, Moment::Received)),
            )
        }
    }
}

/// Appends the load's summary: what each frame did on the main thread.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(super) fn load_block(out: &mut String, frames: &[FrameRecord]) -> fmt::Result {
    writeln!(out, "[load]")?;
    let (Some(first), Some(last)) = (frames.first(), frames.last()) else {
        return writeln!(out, "frames: none");
    };
    let span = (last.start + last.busy).saturating_sub(first.start);
    let dabs: u64 = frames.iter().map(|f| u64::from(f.dabs)).sum();
    let skipped: u64 = frames.iter().map(|f| u64::from(f.skipped)).sum();
    let intervals = Spread::of(frames.iter().map(|f| f.interval));
    #[expect(clippy::cast_precision_loss, reason = "dab counts are far below 2^52")]
    let rate = dabs as f64 / span.as_secs_f64().max(1e-9);
    writeln!(
        out,
        "frames: {} over {:.1} s, paced to {:.2} ms; {dabs} draw steps ({rate:.0} Hz), \
         {skipped} dropped",
        frames.len(),
        span.as_secs_f64(),
        intervals.map_or(0.0, |s| s.p50),
    )?;
    let over = frames
        .iter()
        .filter(|f| f.busy.as_secs_f64() * 1000.0 > BUDGET_MS)
        .count();
    writeln!(
        out,
        "main thread away per frame: {}; {over} frames over {BUDGET_MS:.0} ms ({:.1}%)",
        show(Spread::of(frames.iter().map(|f| f.busy))),
        100.0 * to_f64(over) / to_f64(frames.len()),
    )?;
    let ends: Vec<Duration> = frames
        .iter()
        .map(|f| f.stroke_end)
        .filter(|d| !d.is_zero())
        .collect();
    writeln!(
        out,
        "phases: brush {}; stroke ends {}; upkeep (recovery flushes) {}; jobs (refresh, \
         write-through) {}; longest job slice {:.2} ms",
        show(Spread::of(frames.iter().map(|f| f.brush))),
        show(Spread::of(ends)),
        show(Spread::of(frames.iter().map(|f| f.upkeep))),
        show(Spread::of(frames.iter().map(|f| f.jobs))),
        frames
            .iter()
            .map(|f| f.longest_slice.as_secs_f64() * 1000.0)
            .fold(0.0, f64::max),
    )
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use input::{Buttons, Phase};

    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    fn row(source: Source, received: u64, handled: u64, dejittered: Option<u64>) -> Sample {
        Sample {
            source,
            received: ms(received),
            handled: Some(ms(handled)),
            dejittered: dejittered.map(ms),
            wakeup: 0,
            x: 0.0,
            y: 0.0,
            contact: dejittered.is_some(),
            pressure: None,
            tilt: None,
            distance: None,
            roll: None,
            phase: Some(if dejittered.is_some() {
                Phase::Drag
            } else {
                Phase::Hover
            }),
            buttons: Buttons::default(),
            device_ms: None,
            winit_event: None,
            proximity: Some(true),
            step: None,
        }
    }

    #[test]
    fn spreads_take_nearest_rank_percentiles() {
        let spread = Spread::of((1..=100).map(ms)).expect("some");
        assert_eq!(spread.n, 100);
        assert!((spread.p50 - 51.0).abs() < 1e-9, "{spread:?}");
        assert!((spread.p99 - 99.0).abs() < 1e-9, "{spread:?}");
        assert!((spread.max - 100.0).abs() < 1e-9, "{spread:?}");
        assert_eq!(Spread::of([]), None);
    }

    #[test]
    fn handling_and_the_clock_split_ours_and_winits_is_all_behind_the_clock() {
        // Ours: received 1 ms behind the clock, handled 3 ms after receipt.
        let ours = [
            row(Source::Ours, 11, 14, Some(10)),
            row(Source::Ours, 21, 24, Some(20)),
        ];
        let ours: Vec<&Sample> = ours.iter().collect();
        let ours_handling = handling(&ours).expect("some");
        assert!((ours_handling.p99 - 3.0).abs() < 1e-9, "{ours_handling:?}");
        let behind = behind_clock(&ours, Moment::Received).expect("some");
        assert!((behind.max - 1.0).abs() < 1e-9, "{behind:?}");
        let handled = behind_clock(&ours, Moment::Handled).expect("some");
        assert!((handled.max - 4.0).abs() < 1e-9, "{handled:?}");

        // winit: handled when received, 6 ms behind the clock; the hover row has no clock.
        let winit = [
            row(Source::Winit, 16, 16, Some(10)),
            row(Source::Winit, 30, 30, None),
        ];
        let winit: Vec<&Sample> = winit.iter().collect();
        let winit_delay = behind_clock(&winit, Moment::Handled).expect("some");
        assert_eq!(winit_delay.n, 1, "contact samples only");
        assert!((winit_delay.p50 - 6.0).abs() < 1e-9, "{winit_delay:?}");
        let zero = handling(&winit).expect("some");
        assert!(zero.max.abs() < 1e-9, "zero by construction");
    }

    #[test]
    fn the_load_block_counts_frames_over_budget() {
        let frame = |start: u64, busy_us: u64| FrameRecord {
            start: ms(start),
            interval: Duration::from_micros(16_667),
            dabs: 3,
            busy: Duration::from_micros(busy_us),
            brush: Duration::from_micros(busy_us / 2),
            jobs: Duration::from_micros(busy_us / 2),
            ..FrameRecord::default()
        };
        let frames = [frame(0, 12_000), frame(17, 4_000), frame(33, 7_000)];
        let mut out = String::new();
        load_block(&mut out, &frames).expect("a String");
        assert!(
            out.contains("frames: 3 over 0.0 s, paced to 16.67 ms; 9 draw steps"),
            "{out}"
        );
        assert!(out.contains("1 frames over 8 ms (33.3%)"), "{out}");
        assert!(
            out.contains("away per frame: p50 7.00 / p99 12.00 / max 12.00"),
            "{out}"
        );
        let mut empty = String::new();
        load_block(&mut empty, &[]).expect("a String");
        assert!(empty.ends_with("frames: none\n"), "{empty}");
    }
}
