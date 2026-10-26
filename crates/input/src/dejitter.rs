//! Recovers a pen's steady sample clock from jittery receipt times.
//!
//! The pen samples on a fixed period `P`, and delivery delay is never negative: a sample can
//! arrive late, never early. Number the samples of a stroke `0, 1, 2, …` (their *slots*) and the
//! true clock is the line `t_i = offset + i·P`. Every receipt `r_j` sits on or above that line, so
//! the best estimate of the offset is the *lower envelope*: `offset = min_j (r_j − j·P)`, over the
//! last [`ENVELOPE`] samples. A sample's de-jittered time is the line's value at its slot. With an
//! exact period and an endless window this is the same as `out_i = min(out_{i−1} + P, r_i)`: the
//! output advances one period per sample, and only drops when a receipt proves the clock runs
//! earlier than thought. It never runs later than the receipt and, given the right period, never
//! before the true clock. The window is short so that an error in the period has little lever.
//!
//! **The period.** A stroke starts with the period learned from earlier strokes, from its second
//! sample on (the first has nothing to go on and is passed through). A period is fitted as the
//! slope between the least-delayed samples of the first and last quarter of the stroke (up to
//! [`WINDOW`] samples). Only a fit spanning at least [`SEED_SPAN`] slots seeds later strokes, and
//! it replaces the seed within a stroke only once it spans more slots than the seed's fit did:
//! the slope's error falls with the square of the span. A session of only short strokes (under
//! [`SEED_SPAN`] slots) therefore stays unseeded; fits are not pooled across strokes. A first
//! stroke with no seed passes receipt times through until [`MIN_FIT`] samples are in, then uses
//! the fit of what it has so far, refined with every sample. Such a *young* fit is rough, so its
//! envelope looks back over only half the samples (halving the lever of its period error), and
//! it is not rise-limited.
//!
//! **Checking the seed.** Every stroke checks its seed: at [`SEED_CHECK`] samples against the
//! mean interval (which rules out harmonics), and at 16, 32, 64 and every 64 samples after
//! against its own fit, within the worst case of that fit's error. A seed that fails gives way to
//! the stroke's own fit.
//!
//! **When the envelope moves.** When the least-delayed sample leaves the window, the envelope
//! rises, but the output may rise by at most [`RISE`] of a period per sample beyond its usual
//! step, so a small shift of the envelope is spread out rather than shown as a jump.
//!
//! **Losing the clock.** The rise limit must not hide a wrong period: with too short a period
//! the line falls behind the truth by the error every sample. So the clock counts as *lost*, and
//! the segment restarts from the sample without a seed, when the envelope sits more than the
//! spread plus [`LOST`] of a period above the limited line, or when a seeded stroke gets
//! [`ADRIFT`] samples in a row later behind the line than the seed's spread allows. Speeding up
//! mid-stroke is harmless (the line follows the receipts). Slowing down is caught on a seeded
//! segment, at a cost of up to about 20 ms of early output before it re-locks. On a segment
//! without a seed (a first stroke, or after such a restart) it is **not** caught: the envelope
//! and the limited line fall behind together and the output can run ~100 ms early for the rest
//! of the stroke. Pens don't change rate mid-stroke in practice, but this is a known gap.
//!
//! **Dropped samples.** Slots are sequential unless a drop is unambiguous. The *delay spread* is
//! the largest residual above the line in the envelope window, and while a seed is in use at
//! least the spread carried over with it (a stroke without one waits for [`SPREAD_SAMPLES`]
//! residuals). A sample is *suspect* when it lands closer to a period late than to the spread. A
//! suspect sample is passed through at its receipt time, which is within the spread of the truth if
//! it was dropped-after and within its own delay if it was merely late. After [`DROP_CONFIRM`]
//! suspects in a row the drop is taken as real: those samples move up by the whole periods they
//! lag. While the spread is half a period or more, a late sample and a dropped one look alike, so
//! no drop is ever inferred; a missed one leaves the line a period early (intervals stay right)
//! until the samples before it leave the envelope window, after which the output catches up at
//! [`RISE`] of a period per sample. Measured Wayland pen data had a delay spread close to a full
//! period and no drops at all.
//!
//! **How close to ideal.** With a seed, simulated strokes of up to 150 samples match an oracle
//! that knows the exact period, to within 0.05 ms at the 99th percentile; longer strokes are held
//! back by the short envelope window (p99 about 0.1–0.2 ms against the oracle's 0.03).
//!
//! A gap of more than [`MAX_GAP_PERIODS`] periods (or [`MAX_GAP_UNKNOWN`] with no period yet) is a
//! stall: the fit restarts from the next sample. A receipt at or before the previous one says
//! nothing about the clock and is not fed to the fit; it takes the next slot (a delivery queue can
//! hold a sample back until the next one arrives), and the output never goes backwards. When the
//! spread is under half a period and the next receipt lands more than half a period before the
//! line, the slot is given back: the repeated receipt was spurious.
//!
//! Design: `specs/2026-10-21-input-pen-backend-design.md`.

use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

/// Samples kept for the period fit. At ~170 Hz that's about 3 s.
const WINDOW: usize = 512;
/// The most slots a window spans without a drop; a seed from a span with drops can't otherwise
/// ever be replaced.
#[expect(
    clippy::cast_precision_loss,
    reason = "a window of a few hundred samples converts to f64 exactly"
)]
const MAX_WINDOW_SPAN: f64 = (WINDOW - 1) as f64;
/// Samples the lower envelope looks back over.
const ENVELOPE: usize = 32;
/// Slots a fit must span before its period replaces the seed and seeds later strokes.
const SEED_SPAN: f64 = 128.0;
/// Without a seed, samples needed before the first period estimate; until then output is receipt.
const MIN_FIT: usize = 4;
/// More periods than this between two samples is a stall: the fit restarts.
const MAX_GAP_PERIODS: f64 = 8.0;
/// Plausible pen sample periods: 2 kHz down to 20 Hz.
const PERIODS: RangeInclusive<f64> = 0.0005..=0.05;
/// With no period yet, a gap longer than [`MAX_GAP_PERIODS`] of the slowest plausible period is a
/// stall.
const MAX_GAP_UNKNOWN: f64 = MAX_GAP_PERIODS * *PERIODS.end();
/// Samples after which a seeded stroke checks the seed against its own mean interval.
const SEED_CHECK: usize = 8;
/// The ratio of mean interval to seed within which the seed is kept at [`SEED_CHECK`] samples.
///
/// A delay spread of up to a period moves the mean of [`SEED_CHECK`] samples by at most a seventh,
/// but one long delay on the last sample moves it further, so this only rules out gross
/// mismatches such as a harmonic (a ratio of 2 or ½). Finer ones are caught by the checks against
/// the stroke's own fit and, when the period is too short, by [`LOST`].
const SEED_AGREEMENT: RangeInclusive<f64> = 0.75..=1.33;
/// Relative slack, on top of the spread's worst case, when the seed is checked against the
/// stroke's own fit.
const SEED_SLACK: f64 = 0.002;
/// How far, as a fraction of the period, the output may rise above its usual step per sample.
///
/// Small: with the right period the envelope only rises when its least-delayed sample leaves the
/// window, by about the spread over [`ENVELOPE`], and rising slowly keeps that from showing as a
/// jump followed by a forced fall. It still outruns any plausible error in a learned period
/// (well under a thousandth) by a wide margin.
const RISE: f64 = 1.0 / 512.0;
/// How far, as a fraction of the period, the envelope may sit above the rise-limited line.
///
/// Beyond this it is not a small shift to smooth but a clock the model has lost (a wrong period,
/// or the rate changed), and the segment restarts without a seed.
const LOST: f64 = 0.5;
/// Samples in a row that land more than [`LOST`] of a period beyond the spread carried with the
/// seed before a seeded line counts as lost. A real delay that long would have widened the spread.
const ADRIFT: usize = 4;
/// Suspect samples in a row that confirm a drop.
const DROP_CONFIRM: usize = 8;
/// Without a spread carried over from earlier strokes, residuals needed before the delay spread
/// is trusted (and drops can be inferred).
const SPREAD_SAMPLES: usize = 16;

/// Turns jittery receipt times of one pen's samples into a steady clock.
///
/// Feed every contact sample of a stroke to [`push`](Self::push) in order, and call
/// [`end_stroke`](Self::end_stroke) when the pen lifts.
#[derive(Debug, Default)]
pub struct Dejitter {
    /// The current run of samples on one clock line; `None` between strokes.
    segment: Option<Segment>,
    /// Learned from a long fit and carried across strokes.
    seed: Option<Seed>,
    /// The last time [`push`](Self::push) returned, which the next output never precedes.
    last_out: Option<Instant>,
}

/// One sample on the clock.
#[derive(Clone, Copy, Debug)]
struct Point {
    /// Slot on the pen's clock: a whole number, kept as `f64` for the arithmetic.
    slot: f64,
    /// Seconds since the segment's origin.
    at: f64,
}

/// Where a segment's period comes from.
#[derive(Clone, Copy, Debug)]
enum Period {
    /// Not known yet: output is the receipt time.
    Unknown,
    /// Carried over from an earlier stroke.
    Seeded(Seed),
    /// Fitted to this segment's samples, in seconds.
    Fitted(f64),
}

impl Period {
    /// The period in seconds, seeded or fitted; `None` while unknown.
    const fn secs(self) -> Option<f64> {
        match self {
            Self::Unknown => None,
            Self::Seeded(Seed { period: p, .. }) | Self::Fitted(p) => Some(p),
        }
    }
}

/// A period learned from a long fit, carried across strokes.
#[derive(Clone, Copy, Debug)]
struct Seed {
    /// Seconds.
    period: f64,
    /// Slots the fit spanned: a later fit replaces the seed only when it spans more.
    span: f64,
    /// The largest delay spread (residual above the line in the envelope window) seen over the
    /// fit once it had settled, in seconds. Lets the next stroke infer drops before it has its
    /// own spread, and bounds how late its samples can plausibly be.
    spread: f64,
}

/// The clock line with the delay spread above it, all in seconds.
#[derive(Clone, Copy, Debug)]
struct Clock {
    /// Seconds per slot: the line's slope.
    period: f64,
    /// Seconds since the segment's origin at slot 0: the line's lower envelope.
    offset: f64,
    /// The delay spread: how far samples lie above the line, at most.
    spread: f64,
}

impl Clock {
    /// How far `point` lies above the line.
    fn residual(self, point: Point) -> f64 {
        point.at - (self.offset + point.slot * self.period)
    }
}

/// A run of samples on one clock line.
#[derive(Debug)]
struct Segment {
    /// The receipt time of the segment's first sample, which is slot 0 at time 0.
    origin: Instant,
    /// The latest samples, oldest first: at most [`WINDOW`], for the fit and the envelope.
    points: VecDeque<Point>,
    /// Samples pushed into this segment, for the seed-check schedule; unlike `points.len()` it
    /// keeps counting past the window.
    pushed: usize,
    /// The period the line uses: unknown, seeded or fitted.
    period: Period,
    /// The last output on the line, for the rise limit.
    last_line: Option<Point>,
    /// The slot the next sample takes.
    next_slot: f64,
    /// Suspect samples in a row at the end of `points`.
    suspects: usize,
    /// Samples in a row received more than [`LOST`] of a period beyond the carried spread.
    adrift: usize,
    /// The largest residual above the line in the envelope window, as of the last sample: the
    /// delay spread.
    spread: f64,
    /// The largest spread seen once the fit was no longer young: what a seed learned here carries.
    peak_spread: f64,
    /// The spread carried over with the seed, while the seed is in use.
    prior_spread: Option<f64>,
}

/// What a segment made of a new sample.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The sample doesn't continue the segment: start a new one from it.
    Restart,
    /// The sample stands on its own: output its receipt time.
    Raw,
    /// The sample's time on the recovered clock, seconds since the origin.
    Steady(f64),
    /// The line no longer fits the samples: restart from this one, without the seed.
    Lost,
}

impl Dejitter {
    /// A filter with nothing learned yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            segment: None,
            seed: None,
            last_out: None,
        }
    }

    /// The sample period: the current stroke's estimate if it has one, otherwise the one learned
    /// from earlier strokes.
    #[must_use]
    pub fn period(&self) -> Option<Duration> {
        self.segment
            .as_ref()
            .and_then(|segment| segment.period.secs())
            .or_else(|| self.seed.map(|seed| seed.period))
            .and_then(|p| Duration::try_from_secs_f64(p).ok())
    }

    /// Takes the receipt time of the stroke's next sample and returns its de-jittered time: never
    /// later than `received`, never earlier than the previous output. Receipt times must not go
    /// backwards; if they do, the output stays monotonic and so may be later than `received`.
    pub fn push(&mut self, received: Instant) -> Instant {
        let out = self.estimate(received).min(received);
        let out = self.last_out.map_or(out, |last| out.max(last));
        self.last_out = Some(out);
        out
    }

    /// Ends the stroke. The next [`push`](Self::push) starts a new one, seeded with the period
    /// learned so far.
    pub fn end_stroke(&mut self) {
        self.segment = None;
    }

    /// The de-jittered time of a sample received at `received`, before [`push`](Self::push)
    /// clamps it: the segment's line at the sample's slot, or the receipt time itself when the
    /// segment starts again or has no line yet.
    fn estimate(&mut self, received: Instant) -> Instant {
        let Some(segment) = self.segment.as_mut() else {
            return self.restart(received, self.seed);
        };
        // Before the origin: receipts went backwards, so the sample is not fed to the fit.
        let Some(at) = received
            .checked_duration_since(segment.origin)
            .map(|d| d.as_secs_f64())
        else {
            return received;
        };
        let step = segment.push(at, self.seed.map(|seed| seed.period));
        if let Some(learned) = segment.learned() {
            self.seed = Some(learned);
        }
        match step {
            Step::Restart => self.restart(received, self.seed),
            Step::Lost => self.restart(received, None),
            Step::Raw => received,
            Step::Steady(t) => {
                Duration::try_from_secs_f64(t.max(0.0)).map_or(received, |d| segment.origin + d)
            }
        }
    }

    /// Starts a new segment at the sample received at `received`, with `seed`'s period if
    /// there is one, and returns the receipt time: a segment's first sample has nothing to go
    /// on.
    fn restart(&mut self, received: Instant, seed: Option<Seed>) -> Instant {
        self.segment = Some(Segment {
            origin: received,
            points: VecDeque::from([Point { slot: 0.0, at: 0.0 }]),
            pushed: 1,
            period: seed.map_or(Period::Unknown, Period::Seeded),
            last_line: None,
            next_slot: 1.0,
            suspects: 0,
            adrift: 0,
            spread: 0.0,
            peak_spread: 0.0,
            prior_spread: seed.map(|seed| seed.spread),
        });
        received
    }
}

impl Segment {
    /// Adds a sample received `at` seconds after the origin; `seed` is the period learned before.
    fn push(&mut self, at: f64, seed: Option<f64>) -> Step {
        let Some(&last) = self.points.back() else {
            return Step::Restart;
        };
        // At or before the previous receipt: still a sample, so it takes a slot, but it says
        // nothing about the clock and is not fed to the fit.
        if at <= last.at {
            let slot = self.next_slot;
            self.next_slot += 1.0;
            return self.period.secs().map_or(Step::Raw, |p| self.line(p, slot));
        }
        let max_gap = self
            .period
            .secs()
            .or(seed)
            .map_or(MAX_GAP_UNKNOWN, |p| MAX_GAP_PERIODS * p);
        if at - last.at > max_gap {
            return Step::Restart;
        }

        let clock = self.unambiguous();
        let slot = self.slot_for(at, last.slot, clock);
        self.next_slot = slot + 1.0;
        // Closer to a period late than to the spread: dropped samples are possible.
        let suspect = clock.is_some_and(|clock| {
            clock.residual(Point { slot, at }) > f64::midpoint(clock.spread, clock.period)
        });
        self.suspects = if suspect { self.suspects + 1 } else { 0 };
        if self.points.len() == WINDOW {
            self.points.pop_front();
        }
        self.points.push_back(Point { slot, at });
        self.pushed += 1;
        if self.suspects > 0 {
            if self.suspects < DROP_CONFIRM {
                return Step::Raw;
            }
            self.confirm_drop();
        }
        // A confirmed drop moved the newest sample's slot.
        let slot = self.points.back().map_or(slot, |newest| newest.slot);

        self.check_seed();
        self.refit();
        let Some(p) = self.period.secs() else {
            return Step::Raw;
        };
        let step = self.line(p, slot);
        // With a seed, the carried spread bounds how late a sample can be. Samples persistently
        // later than that behind the line mean the line is falling behind: a wrong seed.
        if let (Step::Steady(line), Some(spread)) = (step, self.prior_spread) {
            self.adrift = if at - line > spread + LOST * p {
                self.adrift + 1
            } else {
                0
            };
            if self.adrift >= ADRIFT {
                return Step::Lost;
            }
        }
        step
    }

    /// The line, as `(period, offset)`, and the delay spread, when the spread is known and
    /// under half a period: only then can a sample's slot be told for certain.
    fn unambiguous(&mut self) -> Option<Clock> {
        let period = self.period.secs()?;
        let skip = self.suspects;
        let offset = self.offset(period, skip)?;
        let (count, window) = self
            .points
            .iter()
            .rev()
            .skip(skip)
            .take(self.envelope())
            .fold((0_usize, 0.0_f64), |(count, spread), p| {
                (count + 1, spread.max(p.at - p.slot * period - offset))
            });
        self.spread = window;
        if !self.young() {
            self.peak_spread = self.peak_spread.max(window);
        }
        let spread = match self.prior_spread {
            Some(prior) => prior.max(self.spread),
            None if count >= SPREAD_SAMPLES => self.spread,
            None => return None,
        };
        (spread < period / 2.0).then_some(Clock {
            period,
            offset,
            spread,
        })
    }

    /// The slot of a sample received `at`: normally the next one. Receipts that didn't move
    /// forward took slots too, but if they were spurious rather than real samples, this one lands
    /// more than half a period before the line, which no real sample can; then their slots are
    /// given back.
    fn slot_for(&self, at: f64, last: f64, clock: Option<Clock>) -> f64 {
        let mut slot = self.next_slot;
        if let Some(clock) = clock {
            while slot > last + 1.0 && clock.residual(Point { slot, at }) < -clock.period / 2.0 {
                slot -= 1.0;
            }
        }
        slot
    }

    /// The last [`DROP_CONFIRM`] samples were all suspect: move them up by the whole periods they
    /// lag the line.
    fn confirm_drop(&mut self) {
        self.suspects = 0;
        let Some(period) = self.period.secs() else {
            return;
        };
        let Some(offset) = self.offset(period, DROP_CONFIRM) else {
            return;
        };
        let lag = self
            .points
            .iter()
            .rev()
            .take(DROP_CONFIRM)
            .map(|p| p.at - (offset + p.slot * period))
            .fold(f64::INFINITY, f64::min);
        let shift = (lag / period).round().max(1.0);
        for point in self.points.iter_mut().rev().take(DROP_CONFIRM) {
            point.slot += shift;
        }
        self.next_slot += shift;
    }

    /// Drops a seed that disagrees with the stroke's own mean interval: a harmonic, or another
    /// device.
    ///
    /// At [`SEED_CHECK`] samples by the mean interval; at 16, 32 and 64 samples and every 64
    /// after by the stroke's own quarter slope, which can be off by at most twice the spread over
    /// half the samples. A rejected seed gives way to the stroke's own fit.
    ///
    /// The schedule counts samples pushed, not the samples in the window: the window stops
    /// growing at [`WINDOW`], so a schedule on its length would check on every sample once it is
    /// full. Today the two agree, because [`MAX_WINDOW_SPAN`] lets a full window always replace
    /// the seed, so no seed is still in use past the window; counting pushes keeps the
    /// documented schedule if the window or that cap changes.
    fn check_seed(&mut self) {
        let Period::Seeded(Seed { period: seed, .. }) = self.period else {
            return;
        };
        let pushed = self.pushed;
        // The samples in the fit, which stop growing at the window.
        let count = self.points.len();
        let agrees = if pushed == SEED_CHECK {
            let (Some(first), Some(last)) = (self.points.front(), self.points.back()) else {
                return;
            };
            let mean = (last.at - first.at) / (last.slot - first.slot);
            SEED_AGREEMENT.contains(&(mean / seed))
        } else if matches!(pushed, 16 | 32) || (pushed >= 64 && pushed.is_multiple_of(64)) {
            let Some(fit) = quarter_slope(&self.points, seed) else {
                return;
            };
            let spread = self
                .prior_spread
                .map_or(self.spread, |p| p.max(self.spread));
            #[expect(
                clippy::cast_precision_loss,
                reason = "a sample count of at most WINDOW"
            )]
            let tolerance = 4.0 * spread / count as f64 + SEED_SLACK * seed;
            (fit - seed).abs() <= tolerance
        } else {
            return;
        };
        if !agrees {
            self.prior_spread = None;
            self.last_line = None;
            self.period = Period::Unknown;
            self.refit();
        }
    }

    /// Refits the period, unless the seed came from a longer span than this segment has. A full
    /// window always counts as long enough ([`MAX_WINDOW_SPAN`]).
    fn refit(&mut self) {
        let span = self.span();
        let guess = match self.period {
            Period::Seeded(seed) if span >= SEED_SPAN && span >= seed.span.min(MAX_WINDOW_SPAN) => {
                seed.period
            }
            Period::Unknown if self.points.len() >= MIN_FIT => {
                match (self.points.front(), self.points.back()) {
                    (Some(first), Some(last)) => (last.at - first.at) / (last.slot - first.slot),
                    _ => return,
                }
            }
            Period::Fitted(p) => p,
            Period::Seeded(_) | Period::Unknown => return,
        };
        if let Some(p) = quarter_slope(&self.points, guess).filter(|p| PERIODS.contains(p)) {
            self.period = Period::Fitted(p);
        }
    }

    /// The period to carry over to later strokes, once the fit spans enough slots.
    fn learned(&self) -> Option<Seed> {
        let span = self.span();
        match self.period {
            Period::Fitted(period) if span >= SEED_SPAN => Some(Seed {
                period,
                span,
                spread: self.peak_spread.max(self.spread),
            }),
            Period::Fitted(_) | Period::Seeded(_) | Period::Unknown => None,
        }
    }

    /// Slots from the first sample kept to the last.
    fn span(&self) -> f64 {
        match (self.points.front(), self.points.back()) {
            (Some(first), Some(last)) => last.slot - first.slot,
            _ => 0.0,
        }
    }

    /// The lower envelope: the least `at − slot·period` over the envelope window, skipping the
    /// newest `skip` samples.
    fn offset(&self, period: f64, skip: usize) -> Option<f64> {
        self.points
            .iter()
            .rev()
            .skip(skip)
            .take(self.envelope())
            .map(|p| p.at - p.slot * period)
            .reduce(f64::min)
    }

    /// The time of `slot` on the line, rising at most [`RISE`] of a period faster than the clock
    /// since the last output.
    ///
    /// A young fit (fitted here, over fewer than [`SEED_SPAN`] slots) is still converging and is
    /// not limited. An envelope more than the spread plus [`LOST`] of a period above the limited
    /// line means the model has lost the clock.
    fn line(&mut self, period: f64, slot: f64) -> Step {
        let Some(offset) = self.offset(period, 0) else {
            return Step::Raw;
        };
        let mut at = offset + slot * period;
        if let Some(last) = self.last_line.filter(|_| !self.young()) {
            let limited = last.at + (slot - last.slot) * period * (1.0 + RISE);
            // The envelope can rise by up to the spread when its least-delayed sample leaves.
            let spread = self
                .prior_spread
                .map_or(self.spread, |prior| prior.max(self.spread));
            if at - limited > spread + LOST * period {
                return Step::Lost;
            }
            at = at.min(limited);
        }
        self.last_line = Some(Point { slot, at });
        Step::Steady(at)
    }

    /// Whether the period was fitted here over fewer than [`SEED_SPAN`] slots.
    fn young(&self) -> bool {
        matches!(self.period, Period::Fitted(_)) && self.span() < SEED_SPAN
    }

    /// Samples the envelope looks back over: [`ENVELOPE`], or half the samples while the fit is
    /// young, so that an early period error has half the lever.
    fn envelope(&self) -> usize {
        if self.young() {
            (self.points.len() / 2).clamp(1, ENVELOPE)
        } else {
            ENVELOPE
        }
    }
}

/// The period through the least-delayed samples of the first and last quarter of `points`, each
/// picked as the lowest against a line of slope `guess`, refined once. `None` with fewer than four
/// points.
fn quarter_slope(points: &VecDeque<Point>, guess: f64) -> Option<f64> {
    fn lowest<'a>(period: f64, part: impl Iterator<Item = &'a Point>) -> Option<Point> {
        part.copied()
            .min_by(|a, b| (a.at - a.slot * period).total_cmp(&(b.at - b.slot * period)))
    }

    let quarter = points.len() / 4;
    if quarter == 0 {
        return None;
    }
    let mut period = guess;
    for _ in 0..2 {
        let first = lowest(period, points.iter().take(quarter))?;
        let last = lowest(period, points.iter().rev().take(quarter))?;
        if last.slot <= first.slot {
            return None;
        }
        period = (last.at - first.at) / (last.slot - first.slot);
    }
    Some(period)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;
