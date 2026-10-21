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
//! the slope's error falls with the square of the span. A first stroke with no seed passes
//! receipt times through until [`MIN_FIT`] samples are in, then uses the fit of what it has so
//! far, refined with every sample; those early fits are rough. A seed far from the stroke's mean
//! interval (a harmonic, or a different device) is dropped for the stroke.
//!
//! **When the envelope moves.** When the least-delayed sample leaves the window, the envelope
//! rises, but the output may rise by at most [`RISE`] of a period per sample beyond its usual
//! step, so a sudden shift of the envelope is spread out rather than shown as a jump.
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
/// Samples the lower envelope looks back over.
const ENVELOPE: usize = 32;
/// Slots a fit must span before its period replaces the seed and seeds later strokes.
const SEED_SPAN: f64 = 128.0;
/// Without a seed, samples needed before the first period estimate; until then output is receipt.
const MIN_FIT: usize = 4;
/// More periods than this between two samples is a stall: the fit restarts.
const MAX_GAP_PERIODS: f64 = 8.0;
/// With no period yet, a gap longer than this (8 × the slowest plausible period) is a stall.
const MAX_GAP_UNKNOWN: f64 = 0.4;
/// Plausible pen sample periods: 2 kHz down to 20 Hz.
const PERIODS: RangeInclusive<f64> = 0.0005..=0.05;
/// Samples after which a seeded stroke checks the seed against its own mean interval.
const SEED_CHECK: usize = 8;
/// The ratio of mean interval to seed within which the seed is kept. A delay spread of up to a
/// period moves the mean of [`SEED_CHECK`] samples by at most a seventh; a harmonic doubles it.
const SEED_AGREEMENT: RangeInclusive<f64> = 0.75..=1.33;
/// How far, as a fraction of the period, the output may rise above its usual step per sample.
const RISE: f64 = 1.0 / 512.0;
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
    segment: Option<Segment>,
    /// Learned from a long fit and carried across strokes.
    seed: Option<Seed>,
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
    /// The delay spread seen over the fit, in seconds.
    spread: f64,
}

/// The clock line with the delay spread above it, all in seconds.
#[derive(Clone, Copy, Debug)]
struct Clock {
    period: f64,
    offset: f64,
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
    origin: Instant,
    points: VecDeque<Point>,
    period: Period,
    /// The last output on the line, for the rise limit.
    last_line: Option<Point>,
    /// The slot the next sample takes.
    next_slot: f64,
    /// Suspect samples in a row at the end of `points`.
    suspects: usize,
    /// The largest residual above the line in the envelope window, as of the last sample: the
    /// delay spread.
    spread: f64,
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

    fn estimate(&mut self, received: Instant) -> Instant {
        let Some(segment) = self.segment.as_mut() else {
            return self.restart(received);
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
            Step::Restart => self.restart(received),
            Step::Raw => received,
            Step::Steady(t) => {
                Duration::try_from_secs_f64(t.max(0.0)).map_or(received, |d| segment.origin + d)
            }
        }
    }

    fn restart(&mut self, received: Instant) -> Instant {
        self.segment = Some(Segment {
            origin: received,
            points: VecDeque::from([Point { slot: 0.0, at: 0.0 }]),
            period: self.seed.map_or(Period::Unknown, Period::Seeded),
            last_line: None,
            next_slot: 1.0,
            suspects: 0,
            spread: 0.0,
            prior_spread: self.seed.map(|seed| seed.spread),
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
            return self
                .period
                .secs()
                .and_then(|p| self.line(p, slot))
                .map_or(Step::Raw, Step::Steady);
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
        self.line(p, slot).map_or(Step::Raw, Step::Steady)
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
            .take(ENVELOPE)
            .fold((0_usize, 0.0_f64), |(count, spread), p| {
                (count + 1, spread.max(p.at - p.slot * period - offset))
            });
        self.spread = window;
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
    fn check_seed(&mut self) {
        let Period::Seeded(Seed { period: seed, .. }) = self.period else {
            return;
        };
        if self.points.len() != SEED_CHECK {
            return;
        }
        let (Some(first), Some(last)) = (self.points.front(), self.points.back()) else {
            return;
        };
        let mean = (last.at - first.at) / (last.slot - first.slot);
        if !SEED_AGREEMENT.contains(&(mean / seed)) {
            self.period = Period::Unknown;
            self.prior_spread = None;
        }
    }

    /// Refits the period, unless the seed came from a longer span than this segment has.
    fn refit(&mut self) {
        let span = self.span();
        let guess = match self.period {
            Period::Seeded(seed) if span >= SEED_SPAN && span > seed.span => seed.period,
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
                spread: self.spread,
            }),
            Period::Fitted(_) | Period::Seeded(_) | Period::Unknown => None,
        }
    }

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
            .take(ENVELOPE)
            .map(|p| p.at - p.slot * period)
            .reduce(f64::min)
    }

    /// The time of `slot` on the line, rising at most [`RISE`] of a period faster than the clock
    /// since the last output.
    fn line(&mut self, period: f64, slot: f64) -> Option<f64> {
        let mut at = self.offset(period, 0)? + slot * period;
        if let Some(last) = self.last_line {
            at = at.min(last.at + (slot - last.slot) * period * (1.0 + RISE));
        }
        self.last_line = Some(Point { slot, at });
        Some(at)
    }
}

/// The period through the least-delayed samples of the first and last quarter of `points`, each
/// picked as the lowest against a line of slope `guess`, refined once. `None` with fewer than four
/// points.
fn quarter_slope(points: &VecDeque<Point>, guess: f64) -> Option<f64> {
    let quarter = points.len() / 4;
    if quarter == 0 {
        return None;
    }
    let lowest = |period: f64, part: &mut dyn Iterator<Item = &Point>| {
        part.copied()
            .min_by(|a, b| (a.at - a.slot * period).total_cmp(&(b.at - b.slot * period)))
    };
    let mut period = guess;
    for _ in 0..2 {
        let first = lowest(period, &mut points.iter().take(quarter))?;
        let last = lowest(period, &mut points.iter().rev().take(quarter))?;
        if last.slot <= first.slot {
            return None;
        }
        period = (last.at - first.at) / (last.slot - first.slot);
    }
    Some(period)
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    const PERIOD: f64 = 0.0058;
    const MS: f64 = 0.001;
    const US: f64 = 0.000_001;
    /// Strokes per case.
    const SEEDS: u64 = 200;
    const LENGTHS: [usize; 5] = [20, 50, 100, 150, 400];

    /// Xorshift64, seeded through splitmix64 so that small seeds don't start near zero.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            Self((z ^ (z >> 31)) | 1)
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "a 53-bit integer converts to f64 exactly"
        )]
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// How late samples arrive.
    #[derive(Clone, Copy, Debug)]
    enum Delay {
        /// Uniform from zero to the given seconds.
        Uniform(f64),
        /// Exponential with the given mean, in seconds.
        Exponential(f64),
    }

    impl Delay {
        const MODELS: [Self; 3] = [
            Self::Uniform(4.0 * MS),
            Self::Uniform(PERIOD),
            Self::Exponential(1.7 * MS),
        ];

        fn draw(self, rng: &mut Rng) -> f64 {
            match self {
                Self::Uniform(max) => rng.unit() * max,
                Self::Exponential(mean) => -mean * (1.0 - rng.unit()).ln(),
            }
        }
    }

    /// One stroke: `(true time, receipt)` per delivered sample, in seconds from `start`.
    struct Stroke {
        start: Instant,
        samples: Vec<(f64, f64)>,
    }

    /// A stroke run through a filter: `(true time, receipt, output)`, seconds from its start.
    type Run = Vec<(f64, f64, f64)>;

    impl Stroke {
        fn new(
            start: Instant,
            period: f64,
            count: usize,
            mut delay: impl FnMut() -> f64,
            dropped: impl Fn(usize) -> bool,
        ) -> Self {
            let samples = (0..count)
                .scan(0.0, |truth, i| {
                    let t = *truth;
                    *truth += period;
                    Some((i, t))
                })
                .filter(|&(i, _)| !dropped(i))
                // Delivery is first in, first out: a sample can't overtake the one before it.
                .scan(f64::NEG_INFINITY, |previous: &mut f64, (_, truth)| {
                    *previous = previous.max(truth + delay());
                    Some((truth, *previous))
                })
                .collect();
            Self { start, samples }
        }

        fn jittered(start: Instant, count: usize, delay: Delay, rng: &mut Rng) -> Self {
            Self::new(start, PERIOD, count, || delay.draw(rng), |_| false)
        }

        fn at(&self, seconds: f64) -> Instant {
            self.start + Duration::from_secs_f64(seconds)
        }

        /// Runs the stroke through `filter` and ends it.
        fn run(&self, filter: &mut Dejitter) -> Run {
            let out = self
                .samples
                .iter()
                .map(|&(truth, receipt)| {
                    let out = filter.push(self.at(receipt));
                    let out = out
                        .checked_duration_since(self.start)
                        .expect("output before the stroke started");
                    (truth, receipt, out.as_secs_f64())
                })
                .collect();
            filter.end_stroke();
            out
        }
    }

    /// Signed error of each output interval against the true one.
    fn interval_errors(run: &Run) -> Vec<f64> {
        run.windows(2)
            .map(|pair| (pair[1].2 - pair[0].2) - (pair[1].0 - pair[0].0))
            .collect()
    }

    /// The same for the raw receipt times.
    fn raw_errors(run: &Run) -> Vec<f64> {
        run.windows(2)
            .map(|pair| (pair[1].1 - pair[0].1) - (pair[1].0 - pair[0].0))
            .collect()
    }

    /// The best any filter can do under the invariants.
    ///
    /// It knows the exact period, outputs never after receipt nor before the truth, and passes
    /// the first sample through. Its lag is the running minimum delay; its interval errors are
    /// the steps of that minimum.
    fn oracle_errors(run: &Run) -> Vec<f64> {
        let lags: Vec<f64> = run
            .iter()
            .scan(f64::INFINITY, |least, &(truth, receipt, _)| {
                *least = least.min(receipt - truth);
                Some(*least)
            })
            .collect();
        lags.windows(2).map(|pair| pair[1] - pair[0]).collect()
    }

    fn rms(errors: &[f64]) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "a short test vector")]
        let count = errors.len().max(1) as f64;
        (errors.iter().map(|e| e * e).sum::<f64>() / count).sqrt()
    }

    /// The `q`th quantile (0–1) of `values`, reordering them. `None` if empty.
    fn quantile(values: &mut [f64], q: f64) -> Option<f64> {
        let last = values.len().checked_sub(1)?;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "a rank into a window of at most WINDOW values, q in 0–1"
        )]
        let rank = ((last as f64) * q.clamp(0.0, 1.0)).round() as usize;
        let (_, value, _) = values.select_nth_unstable_by(rank.min(last), f64::total_cmp);
        Some(*value)
    }

    /// The `q`th quantile (0–1) of the absolute values.
    fn quantile_abs(values: &[f64], q: f64) -> f64 {
        let mut abs: Vec<f64> = values.iter().map(|v| v.abs()).collect();
        quantile(&mut abs, q).unwrap_or(0.0)
    }

    /// Statistics pooled over many strokes.
    #[derive(Debug, Default)]
    struct Pooled {
        filter: Vec<f64>,
        raw: Vec<f64>,
        oracle: Vec<f64>,
        strokes: usize,
        /// Strokes whose RMS interval error beats the raw receipts' (or ties).
        no_worse: usize,
        /// Strokes with an interval error over half a period: a sample in the wrong slot.
        slipped: usize,
        /// Least lag over all samples, and greatest lag after the first sample.
        lag: (f64, f64),
        /// How far the lag after the first sample went beyond a period plus a millisecond, or the
        /// least delay seen so far where that is later (a long delay holds back every filter).
        too_late: f64,
    }

    impl Pooled {
        fn new() -> Self {
            Self {
                lag: (f64::INFINITY, f64::NEG_INFINITY),
                too_late: f64::NEG_INFINITY,
                ..Self::default()
            }
        }

        fn add(&mut self, run: &Run) {
            let (filter, raw) = (interval_errors(run), raw_errors(run));
            self.strokes += 1;
            if rms(&filter) <= rms(&raw) + 1e-12 {
                self.no_worse += 1;
            }
            if filter.iter().any(|e| e.abs() > PERIOD / 2.0) {
                self.slipped += 1;
            }
            // The least delay of the samples so far whose receipt moved forward: a repeated
            // receipt tells the filter nothing.
            let (mut least_delay, mut previous) = (f64::INFINITY, f64::NEG_INFINITY);
            for (i, &(truth, receipt, out)) in run.iter().enumerate() {
                if receipt > previous {
                    least_delay = least_delay.min(receipt - truth);
                }
                previous = receipt;
                self.lag.0 = self.lag.0.min(out - truth);
                if i > 0 {
                    self.lag.1 = self.lag.1.max(out - truth);
                    let bound = (PERIOD + MS).max(least_delay);
                    self.too_late = self.too_late.max(out - truth - bound - US);
                }
            }
            self.filter.extend(filter);
            self.raw.extend(raw);
            self.oracle.extend(oracle_errors(run));
        }

        #[expect(clippy::cast_precision_loss, reason = "counts of a few hundred")]
        fn no_worse_share(&self) -> f64 {
            self.no_worse as f64 / self.strokes.max(1) as f64
        }

        fn summary(&self) -> String {
            format!(
                "filter med {:.3} p99 {:.3} ms | raw med {:.3} p99 {:.3} | oracle p99 {:.3} | \
                 no worse {:.1}% | slipped {} | lag {:.3}..{:.3} ms",
                quantile_abs(&self.filter, 0.5) / MS,
                quantile_abs(&self.filter, 0.99) / MS,
                quantile_abs(&self.raw, 0.5) / MS,
                quantile_abs(&self.raw, 0.99) / MS,
                quantile_abs(&self.oracle, 0.99) / MS,
                self.no_worse_share() * 100.0,
                self.slipped,
                self.lag.0 / MS,
                self.lag.1 / MS,
            )
        }
    }

    /// A filter that learned the period from a 400-sample stroke with `delay`.
    fn seeded(start: Instant, delay: Delay, rng: &mut Rng) -> Dejitter {
        let mut filter = Dejitter::new();
        drop(Stroke::jittered(start, 400, delay, rng).run(&mut filter));
        filter
    }

    /// Seeded strokes of every length for `delay`.
    ///
    /// Each seed warms a filter on a 400-sample stroke, then runs one stroke of each length
    /// through it, longest last so that only the warm-up has seeded the shorter ones.
    fn seeded_cases(delay: Delay) -> Vec<Pooled> {
        let base = Instant::now();
        let mut pooled: Vec<Pooled> = LENGTHS.iter().map(|_| Pooled::new()).collect();
        for seed in 1..=SEEDS {
            let mut rng = Rng::new(seed);
            let mut filter = seeded(base, delay, &mut rng);
            for (n, (case, &length)) in (1..).zip(pooled.iter_mut().zip(&LENGTHS)) {
                let start = base + Duration::from_secs(10 * n);
                case.add(&Stroke::jittered(start, length, delay, &mut rng).run(&mut filter));
            }
        }
        pooled
    }

    /// First strokes, from a fresh filter, of every length for `delay`.
    fn unseeded_cases(delay: Delay) -> Vec<Pooled> {
        let base = Instant::now();
        LENGTHS
            .iter()
            .map(|&length| {
                let mut case = Pooled::new();
                for seed in 1..=SEEDS {
                    let mut rng = Rng::new(seed);
                    let stroke = Stroke::jittered(base, length, delay, &mut rng);
                    case.add(&stroke.run(&mut Dejitter::new()));
                }
                case
            })
            .collect()
    }

    /// The p99 interval error a stroke length may have.
    ///
    /// That's the target or, where no filter can meet it, the exact-period oracle's own p99 plus
    /// a little. (The oracle also knows the delay of a sample queued behind a late one; the
    /// filter, seeing a repeated receipt, does not.)
    fn p99_bound(length: usize, oracle: f64) -> f64 {
        let target = if length >= 50 { 0.5 * MS } else { MS };
        target.max(oracle + 0.25 * MS)
    }

    #[test]
    fn rng_first_draw_is_not_near_zero() {
        for seed in 1..=10 {
            let first = Rng::new(seed).unit();
            assert!(first > 0.01, "seed {seed}: first draw {first}");
        }
    }

    #[test]
    fn removes_jitter_from_seeded_strokes() {
        for delay in Delay::MODELS {
            for (case, length) in seeded_cases(delay).iter().zip(LENGTHS) {
                let what = format!("{delay:?}, {length} samples: {}", case.summary());
                let median = quantile_abs(&case.filter, 0.5);
                assert!(median <= 0.05 * MS, "median: {what}");
                let p99 = quantile_abs(&case.filter, 0.99);
                let oracle = quantile_abs(&case.oracle, 0.99);
                assert!(p99 <= p99_bound(length, oracle), "p99: {what}");
                assert!(case.no_worse_share() >= 0.99, "worse than raw: {what}");
                assert!(case.lag.0 >= -0.05 * MS, "earlier than truth: {what}");
                assert!(case.too_late <= 0.0, "late: {what}");
            }
        }
    }

    #[test]
    fn first_stroke_without_a_seed() {
        for delay in Delay::MODELS {
            for (case, length) in unseeded_cases(delay).iter().zip(LENGTHS) {
                let what = format!("{delay:?}, {length} samples: {}", case.summary());
                if length >= 50 {
                    assert!(case.no_worse_share() >= 0.95, "worse than raw: {what}");
                }
            }
        }
        let mut rng = Rng::new(1);
        let stroke = Stroke::jittered(Instant::now(), 20, Delay::Uniform(4.0 * MS), &mut rng);
        let mut filter = Dejitter::new();
        for &(_, receipt, out) in stroke.run(&mut filter).iter().take(MIN_FIT - 1) {
            assert!(
                (out - receipt).abs() < US,
                "before a period is known, output is receipt"
            );
        }
    }

    #[test]
    fn counts_dropped_samples() {
        let delay = Delay::Uniform(2.0 * MS);
        let base = Instant::now();
        let mut slipped = 0;
        for seed in 1..=SEEDS {
            let mut rng = Rng::new(seed);
            let mut filter = seeded(base, delay, &mut rng);
            let start = base + Duration::from_secs(10);
            let stroke = Stroke::new(start, PERIOD, 400, || delay.draw(&mut rng), |i| i % 10 == 9);
            let run = stroke.run(&mut filter);
            let errors = interval_errors(&run);
            if errors.iter().any(|e| e.abs() > PERIOD / 2.0) {
                slipped += 1;
            }
            // Delivered samples that follow a drop.
            let drops: Vec<usize> = run
                .windows(2)
                .zip(1..)
                .filter(|(pair, _)| pair[1].0 - pair[0].0 > 1.5 * PERIOD)
                .map(|(_, i)| i)
                .collect();
            for (k, &drop) in drops.iter().enumerate() {
                let next = drops.get(k + 1).copied().unwrap_or(run.len());
                for i in (drop + DROP_CONFIRM + 8).min(next)..next {
                    // `errors[i - 1]` is the interval ending at sample `i`.
                    assert!(
                        errors[i - 1].abs() < 0.5 * MS,
                        "seed {seed}: sample {i}, {} after a drop: error {}",
                        i - drop,
                        errors[i - 1]
                    );
                }
            }
        }
        println!("{slipped} of {SEEDS} strokes slipped");
        assert!(
            slipped * 100 <= SEEDS,
            "{slipped} of {SEEDS} strokes slipped"
        );
    }

    #[test]
    fn without_jitter_output_is_receipt() {
        let mut filter = Dejitter::new();
        for n in 0..3 {
            let start = Instant::now() + Duration::from_secs(10 * n);
            let stroke = Stroke::new(start, PERIOD, 400, || 0.0, |_| false);
            for (_, receipt, out) in stroke.run(&mut filter) {
                assert!((out - receipt).abs() < US, "{out} vs {receipt}");
            }
        }
    }

    #[test]
    fn never_after_receipt_and_never_backwards() {
        for seed in 1..=20 {
            let mut rng = Rng::new(seed);
            let period = rng.unit().mul_add(12.0 * MS, 3.0 * MS);
            let jitter = rng.unit() * 0.6 * period;
            let drop_rate = rng.unit() * 0.1;
            let drops: Vec<bool> = (0..2000).map(|_| rng.unit() < drop_rate).collect();
            let stroke = Stroke::new(
                Instant::now(),
                period,
                2000,
                || rng.unit() * jitter,
                |i| drops[i],
            );
            let out = stroke.run(&mut Dejitter::new());
            for &(_, receipt, o) in &out {
                assert!(
                    o <= receipt + 1e-9,
                    "seed {seed}: output {o} after receipt {receipt}"
                );
            }
            for pair in out.windows(2) {
                assert!(pair[1].2 >= pair[0].2, "seed {seed}: went backwards");
            }
        }
    }

    #[test]
    fn a_stall_restarts_the_fit() {
        let mut filter = Dejitter::new();
        let stroke = Stroke::new(Instant::now(), PERIOD, 100, || 0.0, |_| false);
        let mut rng = Rng::new(1);
        let jittered = Stroke::new(
            stroke.start,
            PERIOD,
            100,
            || Delay::Uniform(4.0 * MS).draw(&mut rng),
            |_| false,
        );
        for &(_, receipt) in &jittered.samples {
            filter.push(stroke.at(receipt));
        }
        let late = stroke.at(100.0 * PERIOD + 0.2);
        assert_eq!(
            filter.push(late),
            late,
            "after a 200 ms stall the sample stands on its own"
        );
    }

    #[test]
    fn learned_period_seeds_the_next_stroke() {
        let delay = Delay::Uniform(4.0 * MS);
        let base = Instant::now();
        let (mut learned, mut exact) = (Pooled::new(), Pooled::new());
        for seed in 1..=50 {
            let mut rng = Rng::new(seed);
            let mut filter = seeded(base, delay, &mut rng);
            let period = filter.period().expect("a period after 400 samples");
            assert!(
                (period.as_secs_f64() - PERIOD).abs() < 0.01 * MS,
                "seed {seed}: learned {period:?}"
            );
            let mut reference = Dejitter::new();
            drop(Stroke::new(base, PERIOD, 400, || 0.0, |_| false).run(&mut reference));
            let stroke = Stroke::jittered(base + Duration::from_secs(10), 100, delay, &mut rng);
            learned.add(&stroke.run(&mut filter));
            exact.add(&stroke.run(&mut reference));
        }
        let (a, b) = (rms(&learned.filter), rms(&exact.filter));
        assert!(
            a <= b * 1.02 + US,
            "learned seed RMS {a}, exact seed RMS {b}"
        );
    }

    #[test]
    fn repeated_and_backwards_receipts_are_harmless() {
        let (base, delay) = (Instant::now(), Delay::Uniform(2.0 * MS));
        let mut rng = Rng::new(7);
        let mut filter = seeded(base, delay, &mut rng);
        let start = base + Duration::from_secs(10);
        let stroke = Stroke::jittered(start, 600, delay, &mut rng);
        let mut last = None;
        for (i, &(_, receipt)) in stroke.samples.iter().enumerate() {
            let mut push = |at: f64| {
                let out = filter.push(stroke.at(at));
                assert!(last.is_none_or(|last| out >= last), "went backwards");
                last = Some(out);
            };
            push(receipt);
            if i % 7 == 3 {
                push(receipt);
            }
            if i % 11 == 5 {
                push(receipt - 3.0 * PERIOD);
            }
        }
        filter.end_stroke();
        let period = filter.period().expect("a period").as_secs_f64();
        assert!((period - PERIOD).abs() < 0.01 * MS, "learned {period}");
    }

    #[test]
    fn a_harmonic_seed_is_dropped() {
        let base = Instant::now();
        let mut filter = Dejitter::new();
        let mut rng = Rng::new(3);
        let delay = Delay::Uniform(MS);
        drop(
            Stroke::new(base, PERIOD / 2.0, 400, || delay.draw(&mut rng), |_| false)
                .run(&mut filter),
        );
        let start = base + Duration::from_secs(10);
        drop(Stroke::new(start, PERIOD, 400, || delay.draw(&mut rng), |_| false).run(&mut filter));
        let period = filter.period().expect("a period").as_secs_f64();
        assert!((period - PERIOD).abs() < 0.01 * MS, "learned {period}");
    }

    #[test]
    #[ignore = "prints the results table"]
    fn report() {
        for delay in Delay::MODELS {
            for (case, length) in seeded_cases(delay).iter().zip(LENGTHS) {
                println!("seeded {delay:?} N={length}: {}", case.summary());
            }
            for (case, length) in unseeded_cases(delay).iter().zip(LENGTHS) {
                println!("unseeded {delay:?} N={length}: {}", case.summary());
            }
        }
    }
}
