//! Recovers a pen's steady sample clock from jittery receipt times.
//!
//! Delivery delay is never negative: a sample can arrive late, never early. Plot each sample's
//! receipt time against its slot (its index on the pen's clock): the true clock is a straight line
//! below every point, and the least-delayed samples sit just above it. So the filter fits the
//! lower convex hull of the points and takes the hull edge over a point 30% of the way along the
//! window as the clock line: its slope is the period and, being a hull edge, it lies below every
//! sample. A sample's de-jittered time is the line's value at its slot. The fit covers the whole
//! stroke, up to the last [`WINDOW`] samples.
//!
//! A sample's slot comes from the current line, so dropped samples are counted: a sample that
//! lands a whole period further along than expected took the next slot but one. That only works
//! while the delay spread stays well under a period; when it approaches one, a dropped sample
//! cannot be told apart from a late one. (Measured Wayland pen data showed no drops at all.)
//!
//! The first slots of a stroke have no line to come from. Once [`MIN_FIT`] samples are in, the
//! filter searches for the period that folds their receipt times most tightly onto one phase,
//! charging each sample it would have to call dropped, and slots them by it; it searches again at
//! 32 and 64 samples, where a large delay spread can no longer pass for a different period. After
//! every fit the window is re-slotted against the new line, and the result kept when it fits
//! better.
//!
//! Design: `specs/2026-10-21-input-pen-backend-design.md`.

use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};

/// Samples the fit looks back over. At ~170 Hz that's about 3 s.
const WINDOW: usize = 512;
/// Samples needed before the fit is trusted; until then output is the receipt time.
const MIN_FIT: usize = 16;
/// More periods than this between two samples is a stall: the fit restarts.
const MAX_GAP_PERIODS: f64 = 8.0;
/// Plausible pen sample periods: 2 kHz down to 20 Hz.
const PERIODS: RangeInclusive<f64> = 0.0005..=0.05;
/// Where along the window the clock line is taken from: the hull edge spanning this fraction.
/// Left of centre the edge rests on older, better-established samples.
const FIT_AT: f64 = 0.3;
/// A sample up to this fraction of a period before the line still takes the slot after it: the
/// line can run a little late of the true clock.
const SLOT_MARGIN: f64 = 0.25;
/// Re-slotting rounds after each fit.
/// Window sizes at which the slots are searched for afresh: once to start, and again with more
/// samples to undo a bootstrap that a large delay spread misled.
const REBOOT: [usize; 3] = [MIN_FIT, 32, 64];
const RESLOT_ROUNDS: usize = 4;
/// The bootstrap search covers periods within this fraction of its first guess…
const SEARCH_SPAN: f64 = 0.3;
/// …in this many steps each side.
///
/// Miri gets a coarse search: it checks the arithmetic, and the full one takes minutes there.
const SEARCH_STEPS: i32 = if cfg!(miri) { 40 } else { 400 };
/// What the bootstrap search charges per sample it would call dropped (or doubled up), as a
/// fraction of the period's phase spread.
const DROP_PENALTY: f64 = 1.0;
/// Slack for float rounding when slotting a sample that sits exactly on a phase boundary.
const SLOT_EPSILON: f64 = 1e-6;

/// Turns jittery receipt times of one pen's samples into a steady clock.
///
/// Feed every contact sample of a stroke to [`push`](Self::push) in order, and call
/// [`end_stroke`](Self::end_stroke) when the pen lifts.
#[derive(Debug, Default)]
pub struct Dejitter {
    segment: Option<Segment>,
    /// Seconds; learned from the fit and carried across strokes as a seed.
    period: Option<f64>,
    last_out: Option<Instant>,
}

/// One sample on the clock.
#[derive(Clone, Copy, Debug)]
struct Point {
    /// Slot on the pen's clock: a whole number, kept as `f64` for the arithmetic.
    slot: f64,
    /// Receipt time, seconds since the segment's origin.
    at: f64,
}

/// The recovered clock: slot `n` falls `offset + n·period` seconds after the segment's origin.
#[derive(Clone, Copy, Debug)]
struct Line {
    period: f64,
    offset: f64,
}

impl Line {
    /// The clock time of `slot`, seconds since the origin.
    fn at(self, slot: f64) -> f64 {
        self.offset + slot * self.period
    }

    /// The slot of a sample received at `at`.
    fn slot(self, at: f64) -> f64 {
        ((at - self.offset) / self.period + SLOT_MARGIN).floor()
    }
}

/// A run of samples fitted by one line.
#[derive(Debug)]
struct Segment {
    origin: Instant,
    points: VecDeque<Point>,
    /// The latest fit; `None` until [`MIN_FIT`] samples are in.
    line: Option<Line>,
    /// Scratch space for the fit, kept to avoid allocating per sample.
    scratch: Scratch,
}

/// Reusable buffers.
#[derive(Debug, Default)]
struct Scratch {
    hull: Vec<Point>,
    reslotted: Vec<Point>,
    phases: Vec<f64>,
}

/// What a segment made of a new sample.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// The sample doesn't continue the segment: start a new one from it.
    Restart,
    /// No fit yet: the sample stands on its own.
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
            period: None,
            last_out: None,
        }
    }

    /// The sample period learned so far.
    #[must_use]
    pub fn period(&self) -> Option<Duration> {
        self.period
            .and_then(|p| Duration::try_from_secs_f64(p).ok())
    }

    /// Takes the receipt time of the stroke's next sample and returns its de-jittered time: never
    /// later than `received`, never earlier than the previous output. Receipt times must not go
    /// backwards; if they do, the output stays monotonic.
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
        let Some(at) = received
            .checked_duration_since(segment.origin)
            .map(|d| d.as_secs_f64())
        else {
            return self.restart(received);
        };
        match segment.push(at, self.period) {
            Step::Restart => self.restart(received),
            Step::Raw => received,
            Step::Steady(t) => {
                self.period = segment.line.map(|line| line.period);
                Duration::try_from_secs_f64(t.max(0.0)).map_or(received, |d| segment.origin + d)
            }
        }
    }

    fn restart(&mut self, received: Instant) -> Instant {
        self.segment = Some(Segment {
            origin: received,
            points: VecDeque::from([Point { slot: 0.0, at: 0.0 }]),
            line: None,
            scratch: Scratch::default(),
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
        let period = self
            .line
            .map(|line| line.period)
            .or(seed)
            .unwrap_or(*PERIODS.end());
        if at - last.at > MAX_GAP_PERIODS * period {
            return Step::Restart;
        }
        let slot = self
            .line
            .map_or(last.slot + 1.0, |line| line.slot(at).max(last.slot + 1.0));
        if slot - last.slot > MAX_GAP_PERIODS {
            return Step::Restart;
        }
        if self.points.len() == WINDOW {
            self.points.pop_front();
        }
        self.points.push_back(Point { slot, at });

        let points = self.points.make_contiguous();
        if self.line.is_none() && points.len() < MIN_FIT {
            return Step::Raw;
        }
        if REBOOT.contains(&points.len()) && !bootstrap(points, seed, &mut self.scratch.phases) {
            return Step::Restart;
        }
        self.line = settle(points, &mut self.scratch);
        match (self.line, points.last()) {
            (Some(line), Some(newest)) => Step::Steady(line.at(newest.slot).min(newest.at)),
            _ => Step::Restart,
        }
    }
}

/// Slots `points` from scratch. Returns `false` if no period could be found.
///
/// Searches near `seed` (or, without one, the median interval) for the period that folds the
/// receipt times most tightly onto one phase, charging every sample it would call dropped or
/// doubled up.
fn bootstrap(points: &mut [Point], seed: Option<f64>, phases: &mut Vec<f64>) -> bool {
    let Some(centre) = seed.or_else(|| median_interval(points, phases)) else {
        return false;
    };
    let mut best: Option<(f64, f64, f64)> = None;
    for step in -SEARCH_STEPS..=SEARCH_STEPS {
        let period = centre * (1.0 + SEARCH_SPAN * f64::from(step) / f64::from(SEARCH_STEPS));
        if let Some((score, start)) = fold(points, period, phases)
            && best.is_none_or(|(least, ..)| score < least)
        {
            best = Some((score, period, start));
        }
    }
    let Some((_, period, start)) = best else {
        return false;
    };
    let mut previous = f64::NEG_INFINITY;
    for point in points.iter_mut() {
        point.slot = ((point.at - start) / period + SLOT_EPSILON)
            .floor()
            .max(previous + 1.0);
        previous = point.slot;
    }
    true
}

/// The median interval between consecutive points.
fn median_interval(points: &[Point], buffer: &mut Vec<f64>) -> Option<f64> {
    buffer.clear();
    buffer.extend(points.windows(2).filter_map(|pair| match pair {
        [a, b] => Some(b.at - a.at),
        _ => None,
    }));
    buffer.sort_by(f64::total_cmp);
    buffer.get(buffer.len() / 2).copied()
}

/// Folds the receipt times onto one `period`: its score and the phase where the spread starts.
///
/// The score is the phase spread as a fraction of the period, plus [`DROP_PENALTY`] per sample the
/// folding implies was dropped or doubled up, averaged over the samples. `None` for a degenerate
/// period.
fn fold(points: &[Point], period: f64, phases: &mut Vec<f64>) -> Option<(f64, f64)> {
    if period <= 0.0 {
        return None;
    }
    phases.clear();
    phases.extend(points.iter().map(|point| point.at.rem_euclid(period)));
    phases.sort_by(f64::total_cmp);
    let (&first, &last) = (phases.first()?, phases.last()?);
    // The widest empty stretch of the phase circle; the samples' phases fill the rest.
    let (gap, start) = phases
        .windows(2)
        .filter_map(|pair| match pair {
            [a, b] => Some((b - a, *b)),
            _ => None,
        })
        .fold((first + period - last, first), |widest, next| {
            if next.0 > widest.0 { next } else { widest }
        });

    let mut misfits = 0.0;
    let mut count = 1.0;
    let mut previous = None;
    for point in points {
        let slot = ((point.at - start) / period + SLOT_EPSILON).floor();
        if let Some(previous) = previous {
            let step = slot - previous;
            misfits += if step < 1.0 { 1.0 } else { step - 1.0 };
            count += 1.0;
        }
        previous = Some(slot);
    }
    Some((
        (period - gap) / period + DROP_PENALTY * misfits / count,
        start,
    ))
}

/// Fits the clock line to `points`, then re-slots them against it while that fits better.
/// `None` with too few points or an implausible period.
fn settle(points: &mut [Point], scratch: &mut Scratch) -> Option<Line> {
    let mut line = fit(points, &mut scratch.hull)?;
    let mut cost = residual(points, line);
    for _ in 0..RESLOT_ROUNDS {
        scratch.reslotted.clear();
        let mut previous = f64::NEG_INFINITY;
        scratch.reslotted.extend(points.iter().map(|point| {
            let slot = line.slot(point.at).max(previous + 1.0);
            previous = slot;
            Point { slot, ..*point }
        }));
        if scratch
            .reslotted
            .iter()
            .zip(points.iter())
            .all(|(new, old)| (new.slot - old.slot).abs() < 0.5)
        {
            break;
        }
        let Some(candidate) = fit(&scratch.reslotted, &mut scratch.hull) else {
            break;
        };
        let candidate_cost = residual(&scratch.reslotted, candidate);
        if candidate_cost >= cost {
            break;
        }
        for (point, new) in points.iter_mut().zip(&scratch.reslotted) {
            point.slot = new.slot;
        }
        (line, cost) = (candidate, candidate_cost);
    }
    PERIODS.contains(&line.period).then_some(line)
}

/// How far, in total, the points sit above `line`.
fn residual(points: &[Point], line: Line) -> f64 {
    points
        .iter()
        .map(|point| point.at - line.at(point.slot))
        .sum()
}

/// The lower-hull edge of `points` spanning [`FIT_AT`] of the way along them, as a line. `None`
/// with fewer than two points.
fn fit(points: &[Point], hull: &mut Vec<Point>) -> Option<Line> {
    hull.clear();
    for &point in points {
        while let [.., a, b] = hull.as_slice() {
            // Drop `b` if it sits on or above the segment from `a` to `point`.
            if (b.at - a.at) * (point.slot - a.slot) >= (point.at - a.at) * (b.slot - a.slot) {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(point);
    }
    let (first, last) = (points.first()?, points.last()?);
    let target = first.slot + (last.slot - first.slot) * FIT_AT;
    let mut edges = hull.windows(2).filter_map(|pair| match pair {
        [a, b] => Some((*a, *b)),
        _ => None,
    });
    let (a, b) = edges
        .clone()
        .find(|(a, b)| a.slot <= target && target <= b.slot)
        .or_else(|| edges.next())?;
    let period = (b.at - a.at) / (b.slot - a.slot);
    Some(Line {
        period,
        offset: a.at - a.slot * period,
    })
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use std::ops::RangeInclusive;
    use std::time::{Duration, Instant};

    use super::*;

    /// Xorshift64: deterministic noise without a dependency.
    struct Rng(u64);

    impl Rng {
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

    /// One stroke: `(true time, receipt)` per delivered sample, in seconds from `start`.
    struct Stroke {
        start: Instant,
        samples: Vec<(f64, f64)>,
    }

    impl Stroke {
        fn new(
            period: f64,
            count: u32,
            mut delay: impl FnMut(u32) -> f64,
            dropped: impl Fn(u32) -> bool,
        ) -> Self {
            let samples = (0..count)
                .filter(|&i| !dropped(i))
                .map(|i| {
                    let truth = f64::from(i) * period;
                    (truth, truth + delay(i))
                })
                .collect();
            Self {
                start: Instant::now(),
                samples,
            }
        }

        fn at(&self, seconds: f64) -> Instant {
            self.start + Duration::from_secs_f64(seconds)
        }

        /// Runs the stroke through `filter`: `(true time, receipt, output)` in seconds from
        /// `start`.
        fn run(&self, filter: &mut Dejitter) -> Vec<(f64, f64, f64)> {
            self.samples
                .iter()
                .map(|&(truth, receipt)| {
                    let out = filter.push(self.at(receipt));
                    let out = out
                        .checked_duration_since(self.start)
                        .expect("output before the stroke started");
                    (truth, receipt, out.as_secs_f64())
                })
                .collect()
        }
    }

    /// Error in each output interval against the true one, after the filter has converged.
    fn interval_errors(out: &[(f64, f64, f64)]) -> impl Iterator<Item = f64> + '_ {
        out.windows(2).skip(CONVERGED).map(|pair| {
            let [(t0, _, o0), (t1, _, o1)] = pair else {
                unreachable!()
            };
            ((o1 - o0) - (t1 - t0)).abs()
        })
    }

    /// The `p`th quantile (0–1) of `values`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index into a short test vector"
    )]
    fn percentile(values: &mut [f64], p: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let rank = ((values.len() - 1) as f64 * p).round() as usize;
        values[rank]
    }

    /// Strokes per statistical test; Miri is too slow for 100.
    const SEEDS: RangeInclusive<u64> = if cfg!(miri) { 1..=2 } else { 1..=100 };
    /// Samples the fit gets to converge before it is judged.
    const CONVERGED: usize = 64;
    const PERIOD: f64 = 0.0058;
    const MS: f64 = 0.001;

    #[test]
    fn without_jitter_output_is_receipt() {
        let stroke = Stroke::new(PERIOD, 200, |_| 0.0, |_| false);
        for (_, receipt, out) in stroke.run(&mut Dejitter::new()) {
            assert!((out - receipt).abs() < 1e-6, "{out} vs {receipt}");
        }
    }

    #[test]
    fn removes_delivery_jitter() {
        let (mut errors, mut lags) = (Vec::new(), Vec::new());
        for seed in SEEDS {
            let mut rng = Rng(seed);
            let stroke = Stroke::new(PERIOD, 400, |_| rng.unit() * 4.0 * MS, |_| false);
            let out = stroke.run(&mut Dejitter::new());
            errors.extend(interval_errors(&out));
            lags.extend(out.iter().skip(CONVERGED).map(|&(truth, _, o)| o - truth));
        }
        let (median, p99) = (percentile(&mut errors, 0.5), percentile(&mut errors, 0.99));
        assert!(median <= 0.05 * MS, "median interval error {median}");
        assert!(p99 <= 0.5 * MS, "p99 interval error {p99}");
        let (low, high) = (percentile(&mut lags, 0.0), percentile(&mut lags, 1.0));
        assert!(low >= -0.05 * MS && high <= MS, "lag from {low} to {high}");
    }

    #[test]
    fn counts_dropped_samples() {
        let mut errors = Vec::new();
        for seed in SEEDS {
            let mut rng = Rng(seed);
            let stroke = Stroke::new(PERIOD, 400, |_| rng.unit() * 2.0 * MS, |i| i % 10 == 9);
            let out = stroke.run(&mut Dejitter::new());
            let stroke_errors: Vec<f64> = interval_errors(&out).collect();
            let slips = stroke_errors.iter().filter(|&&e| e > PERIOD / 2.0).count();
            assert_eq!(slips, 0, "seed {seed}: {slips} samples in the wrong slot");
            errors.extend(stroke_errors);
        }
        let median = percentile(&mut errors, 0.5);
        assert!(median <= 0.05 * MS, "median interval error {median}");
    }

    #[test]
    fn a_stall_restarts_the_fit() {
        let mut filter = Dejitter::new();
        let stroke = Stroke::new(PERIOD, 100, |_| 0.0, |_| false);
        drop(stroke.run(&mut filter));
        let late = stroke.at(100.0 * PERIOD + 0.2);
        assert_eq!(
            filter.push(late),
            late,
            "after a 200 ms stall the sample stands on its own"
        );
    }

    #[test]
    fn never_after_receipt_and_never_backwards() {
        // Miri checks for UB and arithmetic faults, not statistics: a short run is enough.
        let (seeds, count) = if cfg!(miri) {
            (1..=2, 300)
        } else {
            (1..=20, 2000)
        };
        for seed in seeds {
            let mut rng = Rng(seed);
            let period = rng.unit().mul_add(12.0 * MS, 3.0 * MS);
            let jitter = rng.unit() * 0.6 * period;
            let stroke = Stroke::new(period, count, |_| rng.unit() * jitter, |i| i % 37 == 0);
            let out = stroke.run(&mut Dejitter::new());
            for &(_, receipt, o) in &out {
                assert!(
                    o <= receipt + 1e-9,
                    "seed {seed}: output {o} after receipt {receipt}"
                );
            }
            for pair in out.windows(2) {
                let [(_, _, o0), (_, _, o1)] = pair else {
                    unreachable!()
                };
                assert!(o1 >= o0, "seed {seed}: went backwards");
            }
        }
    }

    #[test]
    fn learned_period_seeds_the_next_stroke() {
        let mut filter = Dejitter::new();
        let mut rng = Rng(3);
        drop(Stroke::new(PERIOD, 200, |_| rng.unit() * 2.0 * MS, |_| false).run(&mut filter));
        let learned = filter
            .period()
            .expect("a period after 200 samples")
            .as_secs_f64();
        assert!((learned - PERIOD).abs() < 0.01 * MS, "learned {learned}");
        filter.end_stroke();
        assert!(
            filter.period().is_some(),
            "the period survives the end of a stroke"
        );
    }
}
