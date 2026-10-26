//! The filter against simulated pens. The `#[ignore]`d tests are reports, not checks: they print
//! measurements, assert nothing, and are run by hand (`cargo nextest run -p input --run-ignored
//! only -- report`).

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

    /// Delivered in frames: each sample arrives at the first frame boundary after its time
    /// plus a delay of up to 2 ms. The frames' phase is random.
    fn batched(start: Instant, count: usize, frame: f64, rng: &mut Rng) -> Self {
        let phase = rng.unit() * frame;
        let samples = (0..count)
            .scan(0.0, |truth: &mut f64, _| {
                let t = *truth;
                *truth += PERIOD;
                Some(t)
            })
            .map(|truth| {
                let ready = truth + phase + rng.unit() * 2.0 * MS;
                (truth, (ready / frame).ceil() * frame - phase)
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
    /// Strokes with an interval error over half a period where the oracle's step is not: a
    /// sample in the wrong slot, not a forced step.
    slipped: usize,
    /// Samples, and those more than a millisecond earlier than the truth.
    samples: usize,
    early: usize,
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
        let oracle = oracle_errors(run);
        if filter
            .iter()
            .zip(&oracle)
            .any(|(e, o)| e.abs() > PERIOD / 2.0 && o.abs() <= PERIOD / 2.0)
        {
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
            self.samples += 1;
            if out - truth < -MS {
                self.early += 1;
            }
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
             no worse {:.1}% | slipped {} | lag {:.3}..{:.3} ms | >1 ms early {:.2}%",
            quantile_abs(&self.filter, 0.5) / MS,
            quantile_abs(&self.filter, 0.99) / MS,
            quantile_abs(&self.raw, 0.5) / MS,
            quantile_abs(&self.raw, 0.99) / MS,
            quantile_abs(&self.oracle, 0.99) / MS,
            self.no_worse_share() * 100.0,
            self.slipped,
            self.lag.0 / MS,
            self.lag.1 / MS,
            self.early_share() * 100.0,
        )
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "counts of a few hundred thousand"
    )]
    fn early_share(&self) -> f64 {
        self.early as f64 / self.samples.max(1) as f64
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
            // The first fits are rough: a few early samples may run ahead of the truth.
            assert!(case.lag.0 >= -4.0 * MS, "earlier than truth: {what}");
            assert!(case.early_share() <= 0.02, "often early: {what}");
            let (p99, raw) = (
                quantile_abs(&case.filter, 0.99),
                quantile_abs(&case.raw, 0.99),
            );
            assert!(p99 <= raw, "p99 worse than raw: {what}");
        }
    }
}

#[test]
fn outputs_receipt_before_a_period_exists() {
    for delay in Delay::MODELS {
        for seed in 1..=20 {
            let mut rng = Rng::new(seed);
            let stroke = Stroke::jittered(Instant::now(), 20, delay, &mut rng);
            let mut filter = Dejitter::new();
            let run = stroke.run(&mut filter);
            for (i, &(_, receipt, out)) in run.iter().enumerate().take(MIN_FIT - 1) {
                assert!(
                    (out - receipt).abs() < US,
                    "{delay:?}, seed {seed}, sample {i}: output {out}, receipt {receipt}"
                );
            }
        }
    }
}

/// A seeded stroke of 400 slots with every `every`th dropped, delay uniform 0–2 ms, and its
/// interval errors.
fn dropping_stroke(seed: u64, every: usize) -> (Run, Vec<f64>) {
    let delay = Delay::Uniform(2.0 * MS);
    let base = Instant::now();
    let mut rng = Rng::new(seed);
    let mut filter = seeded(base, delay, &mut rng);
    let start = base + Duration::from_secs(10);
    let dropped = |i: usize| i % every == every - 1;
    let stroke = Stroke::new(start, PERIOD, 400, || delay.draw(&mut rng), dropped);
    let run = stroke.run(&mut filter);
    let errors = interval_errors(&run);
    (run, errors)
}

#[test]
fn counts_dropped_samples() {
    let slipped = (1..=SEEDS)
        .filter(|&seed| {
            let (_, errors) = dropping_stroke(seed, 10);
            errors.iter().any(|e| e.abs() > PERIOD / 2.0)
        })
        .count();
    println!("{slipped} of {SEEDS} strokes slipped");
    assert!(
        slipped * 100 <= usize::try_from(SEEDS).unwrap(),
        "{slipped} of {SEEDS} strokes slipped"
    );
}

#[test]
fn recovers_after_a_drop() {
    let mut checked = 0;
    for seed in 1..=SEEDS {
        let (run, errors) = dropping_stroke(seed, 40);
        // Delivered samples that follow a drop.
        let drops: Vec<usize> = run
            .windows(2)
            .zip(1..)
            .filter(|(pair, _)| pair[1].0 - pair[0].0 > 1.5 * PERIOD)
            .map(|(_, i)| i)
            .collect();
        // Slots 39, 79, …, 359 are followed by a sample; slot 399 ends the stroke.
        assert_eq!(drops.len(), 9, "seed {seed}: drops found");
        for (k, &drop) in drops.iter().enumerate() {
            let next = drops.get(k + 1).copied().unwrap_or(run.len());
            for i in (drop + DROP_CONFIRM + 8)..next {
                // `errors[i - 1]` is the interval ending at sample `i`.
                assert!(
                    errors[i - 1].abs() < 0.5 * MS,
                    "seed {seed}: sample {i}, {} after a drop: error {}",
                    i - drop,
                    errors[i - 1]
                );
                checked += 1;
            }
        }
    }
    // 9 drops per stroke, each followed by 38 or more samples of which K + 8 are excused.
    assert!(
        checked >= 9 * 20 * usize::try_from(SEEDS).unwrap(),
        "only {checked} intervals checked"
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
fn a_stall_restarts_without_a_period() {
    let mut filter = Dejitter::new();
    let start = Instant::now();
    let at = |seconds: f64| start + Duration::from_secs_f64(seconds);
    for receipt in [0.0, 0.0071, 0.0112] {
        filter.push(at(receipt));
    }
    assert!(filter.period().is_none(), "no period from three samples");
    // Past the 0.4 s stall limit for an unknown period.
    for receipt in [0.5, 0.5061, 0.5113] {
        assert_eq!(
            filter.push(at(receipt)),
            at(receipt),
            "after a 0.5 s stall with no period the sample stands on its own"
        );
    }
    assert!(filter.period().is_none(), "the stall restarted the fit");
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
    drop(Stroke::new(base, PERIOD / 2.0, 400, || delay.draw(&mut rng), |_| false).run(&mut filter));
    let start = base + Duration::from_secs(10);
    drop(Stroke::new(start, PERIOD, 400, || delay.draw(&mut rng), |_| false).run(&mut filter));
    let period = filter.period().expect("a period").as_secs_f64();
    assert!((period - PERIOD).abs() < 0.01 * MS, "learned {period}");
}

/// A filter whose seed comes from a 1,000-sample stroke at [`PERIOD`] with every 20th
/// sample dropped: its window spans more slots than a window without drops can.
fn seeded_across_drops(start: Instant, rng: &mut Rng) -> Dejitter {
    let mut filter = Dejitter::new();
    let clean = Stroke::new(start, PERIOD, 400, || rng.unit() * 0.1 * MS, |_| false);
    drop(clean.run(&mut filter));
    let stroke = Stroke::new(
        start + Duration::from_secs(10),
        PERIOD,
        1000,
        || rng.unit() * 0.1 * MS,
        |i| i % 20 == 19,
    );
    drop(stroke.run(&mut filter));
    filter
}

#[test]
fn a_seed_learned_across_drops_can_be_replaced() {
    let base = Instant::now();
    let device = PERIOD * 1.001;
    for seed in 1..=20 {
        let mut rng = Rng::new(seed);
        let mut filter = seeded_across_drops(base, &mut rng);
        let span = filter.seed.expect("a seed").span;
        assert!(span > 511.0, "seed {seed}: the seed spans {span} slots");
        let start = base + Duration::from_secs(20);
        let stroke = Stroke::new(start, device, 1000, || rng.unit() * 0.1 * MS, |_| false);
        drop(stroke.run(&mut filter));
        let period = filter.period().expect("a period").as_secs_f64();
        assert!(
            (period - device).abs() < US,
            "seed {seed}: kept {period}, the stroke ran at {device}"
        );
    }
}

/// Runs `(true time, receipt)` pairs through `filter` as one stroke: the worst lag from
/// sample `from` on, and whether every output stayed at or before its receipt.
fn worst_lag(
    filter: &mut Dejitter,
    start: Instant,
    samples: &[(f64, f64)],
    from: usize,
) -> (f64, bool) {
    let mut worst = f64::INFINITY;
    let mut honest = true;
    for (i, &(truth, receipt)) in samples.iter().enumerate() {
        let received = start + Duration::from_secs_f64(receipt);
        let out = filter.push(received);
        honest &= out <= received;
        let lag = out.duration_since(start).as_secs_f64() - truth;
        if i >= from {
            worst = worst.min(lag);
        }
    }
    filter.end_stroke();
    (worst, honest)
}

/// Samples allowed for a lost clock to be noticed and the fit to restart and settle.
///
/// Noticing takes up to [`ADRIFT`] samples after the line has fallen [`LOST`] of a period
/// plus the spread behind (about a dozen samples for a 1 ms period error); then a young fit
/// needs about 20 samples to settle to within a millisecond. Measured worst lag after it:
/// −0.6 ms.
const RELOCK: usize = 48;

#[test]
fn a_wrong_seed_does_not_drift() {
    let base = Instant::now();
    for device in [0.0048, 0.0068] {
        for seed in 1..=50 {
            let mut rng = Rng::new(seed);
            let mut filter = Dejitter::new();
            let warm = Stroke::new(base, device, 400, || rng.unit() * MS, |_| false);
            drop(warm.run(&mut filter));
            let start = base + Duration::from_secs(10);
            let stroke = Stroke::jittered(start, 200, Delay::Uniform(PERIOD), &mut rng);
            let samples = &stroke.samples;
            let mut again = Dejitter::new();
            let rewarm = Stroke::new(base, device, 400, || rng.unit() * MS, |_| false);
            drop(rewarm.run(&mut again));
            let (whole, _) = worst_lag(&mut again, start, samples, 0);
            let (worst, honest) = worst_lag(&mut filter, start, samples, RELOCK);
            assert!(honest, "device {device}, seed {seed}: output after receipt");
            assert!(worst >= -MS, "device {device}, seed {seed}: lag {worst}");
            // Measured worst over the whole stroke: −7.4 ms.
            assert!(
                whole >= -10.0 * MS,
                "device {device}, seed {seed}: lag {whole}"
            );
        }
    }
}

#[test]
fn a_rate_change_restarts_the_fit() {
    let base = Instant::now();
    for (before, after) in [(PERIOD, 0.008), (0.008, PERIOD)] {
        for seed in 1..=50 {
            let mut rng = Rng::new(seed);
            let mut filter = Dejitter::new();
            let warm = Stroke::new(base, before, 400, || rng.unit() * MS, |_| false);
            drop(warm.run(&mut filter));
            let mut previous = f64::NEG_INFINITY;
            let samples: Vec<(f64, f64)> = (0..300_u32)
                .map(|i| {
                    let truth = if i < 100 {
                        f64::from(i) * before
                    } else {
                        100.0 * before + f64::from(i - 100) * after
                    };
                    previous = previous.max(truth + rng.unit() * MS);
                    (truth, previous)
                })
                .collect();
            let start = base + Duration::from_secs(10);
            let (early, honest) = worst_lag(&mut filter, start, &samples[..100], 0);
            assert!(honest, "seed {seed}: output after receipt");
            assert!(
                early >= -0.05 * MS,
                "seed {seed}: lag {early} before the change"
            );
            let mut again = Dejitter::new();
            drop(Stroke::new(base, before, 400, || rng.unit() * MS, |_| false).run(&mut again));
            let (worst, still_honest) = worst_lag(&mut again, start, &samples, 100 + RELOCK);
            assert!(still_honest, "seed {seed}: output after receipt");
            assert!(
                worst >= -MS,
                "{before} to {after}, seed {seed}: lag {worst} after the change"
            );
        }
    }
}

#[test]
#[ignore = "prints adversarial results"]
fn adversarial_report() {
    let base = Instant::now();
    for device in [0.0048, 0.0068] {
        let (mut all, mut settled) = (f64::INFINITY, f64::INFINITY);
        for seed in 1..=50 {
            let mut rng = Rng::new(seed);
            let mut filter = Dejitter::new();
            drop(Stroke::new(base, device, 400, || rng.unit() * MS, |_| false).run(&mut filter));
            let start = base + Duration::from_secs(10);
            let stroke = Stroke::jittered(start, 200, Delay::Uniform(PERIOD), &mut rng);
            let mut copy = Dejitter::new();
            drop(Stroke::new(base, device, 400, || 0.0, |_| false).run(&mut copy));
            all = all.min(worst_lag(&mut filter, start, &stroke.samples, 0).0);
            settled = settled.min(worst_lag(&mut copy, start, &stroke.samples, RELOCK).0);
        }
        println!(
            "seed from {device} s, 5.8 ms stroke: worst lag {:.2} ms, after {RELOCK}: {:.2} ms",
            all / MS,
            settled / MS
        );
    }
    for (before, after) in [(PERIOD, 0.008), (0.008, PERIOD)] {
        let (mut all, mut settled) = (f64::INFINITY, f64::INFINITY);
        for seed in 1..=50 {
            let mut rng = Rng::new(seed);
            let mut previous = f64::NEG_INFINITY;
            let samples: Vec<(f64, f64)> = (0..300_u32)
                .map(|i| {
                    let truth = if i < 100 {
                        f64::from(i) * before
                    } else {
                        100.0 * before + f64::from(i - 100) * after
                    };
                    previous = previous.max(truth + rng.unit() * MS);
                    (truth, previous)
                })
                .collect();
            let start = base + Duration::from_secs(10);
            for (from, worst) in [(0, &mut all), (100 + RELOCK, &mut settled)] {
                let mut filter = Dejitter::new();
                drop(Stroke::new(base, before, 400, || 0.0, |_| false).run(&mut filter));
                *worst = worst.min(worst_lag(&mut filter, start, &samples, from).0);
            }
        }
        println!(
            "rate {before} to {after} s at sample 100: worst lag {:.2} ms, after {RELOCK} more: {:.2} ms",
            all / MS,
            settled / MS
        );
    }
}

/// Samples whose period changes: `(from sample, period)` pairs, delay uniform 0–1 ms,
/// delivered first in, first out.
fn piecewise(count: u32, rates: &[(u32, f64)], rng: &mut Rng) -> Vec<(f64, f64)> {
    let (mut truth, mut previous) = (0.0, f64::NEG_INFINITY);
    (0..count)
        .map(|i| {
            let t = truth;
            let period = rates
                .iter()
                .rev()
                .find(|&&(from, _)| i >= from)
                .map_or(PERIOD, |&(_, period)| period);
            truth += period;
            previous = f64::max(previous, t + rng.unit() * MS);
            (t, previous)
        })
        .collect()
}

/// A rate-change scenario and the worst lags it is held to: over the whole stroke, and from
/// [`RELOCK`] samples after each change.
struct RateChange {
    name: &'static str,
    seeded_first: bool,
    rates: &'static [(u32, f64)],
    worst: f64,
    relocked: f64,
}

/// Rate changes, held to the bounds measured at this commit.
///
/// Slowing down on a segment without a seed is NOT handled: the line drifts early and does
/// not re-lock within the stroke (see the module docs). The bounds are there so that it
/// cannot silently get worse.
const RATE_CHANGES: [RateChange; 4] = [
    RateChange {
        name: "fresh 5.8 to 8 ms",
        seeded_first: false,
        rates: &[(0, PERIOD), (100, 0.008)],
        worst: 115.0 * MS,
        relocked: 115.0 * MS,
    },
    RateChange {
        name: "fresh 5.8 to 6.5 ms",
        seeded_first: false,
        rates: &[(0, PERIOD), (100, 0.0065)],
        worst: 39.0 * MS,
        relocked: 39.0 * MS,
    },
    RateChange {
        name: "fresh 8 to 5.8 ms",
        seeded_first: false,
        rates: &[(0, 0.008), (100, PERIOD)],
        worst: MS,
        relocked: MS,
    },
    RateChange {
        name: "seeded 5.8 to 8 to 10 ms",
        seeded_first: true,
        rates: &[(0, PERIOD), (100, 0.008), (200, 0.010)],
        worst: 100.0 * MS,
        relocked: 100.0 * MS,
    },
];

/// Worst lag over the whole stroke and after each change plus [`RELOCK`], over 50 seeds.
fn rate_change_lags(seeded_first: bool, rates: &[(u32, f64)]) -> (f64, f64, bool) {
    let base = Instant::now();
    let (mut all, mut settled, mut honest) = (f64::INFINITY, f64::INFINITY, true);
    for seed in 1..=50 {
        let mut rng = Rng::new(seed);
        let samples = piecewise(300, rates, &mut rng);
        let start = base + Duration::from_secs(10);
        let mut filter = Dejitter::new();
        if seeded_first {
            drop(Stroke::new(base, PERIOD, 400, || rng.unit() * MS, |_| false).run(&mut filter));
        }
        let mut previous_out = None;
        for (i, &(truth, receipt)) in samples.iter().enumerate() {
            let received = start + Duration::from_secs_f64(receipt);
            let out = filter.push(received);
            honest &= out <= received && previous_out.is_none_or(|p| out >= p);
            previous_out = Some(out);
            let lag = out.duration_since(start).as_secs_f64() - truth;
            all = all.min(lag);
            let since_change = rates
                .iter()
                .filter_map(|&(from, _)| u32::try_from(i).ok()?.checked_sub(from))
                .min()
                .unwrap_or(u32::MAX);
            if since_change as usize >= RELOCK {
                settled = settled.min(lag);
            }
        }
    }
    (all, settled, honest)
}

#[test]
fn a_rate_change_stays_within_measured_bounds() {
    for case in RATE_CHANGES {
        let (all, settled, honest) = rate_change_lags(case.seeded_first, case.rates);
        let name = case.name;
        assert!(honest, "{name}: output after receipt or backwards");
        assert!(all >= -case.worst, "{name}: worst lag {all}");
        assert!(
            settled >= -case.relocked,
            "{name}: lag {settled} after re-locking"
        );
    }
}

#[test]
#[ignore = "prints adversarial results"]
fn adversarial_report_rates() {
    for case in RATE_CHANGES {
        let (all, settled, _) = rate_change_lags(case.seeded_first, case.rates);
        let name = case.name;
        println!(
            "{name}: worst lag {:.2} ms, after {RELOCK}: {:.2} ms",
            all / MS,
            settled / MS
        );
    }
}

#[test]
#[ignore = "prints adversarial results"]
fn adversarial_report_receipts() {
    let base = Instant::now();
    // One spurious duplicate receipt at sample 50 with spread about a period.
    let (mut worst, mut early, mut total) = (f64::INFINITY, 0_u32, 0_u32);
    for seed in 1..=50 {
        let mut rng = Rng::new(seed);
        let delay = Delay::Uniform(PERIOD);
        let mut filter = seeded(base, delay, &mut rng);
        let start = base + Duration::from_secs(10);
        let stroke = Stroke::jittered(start, 400, delay, &mut rng);
        for (i, &(truth, receipt)) in stroke.samples.iter().enumerate() {
            let out = filter.push(stroke.at(receipt));
            if i == 50 {
                filter.push(stroke.at(receipt));
            }
            if i > 50 {
                let lag = out.duration_since(start).as_secs_f64() - truth;
                worst = worst.min(lag);
                total += 1;
                if lag < -PERIOD / 2.0 {
                    early += 1;
                }
            }
        }
        filter.end_stroke();
    }
    println!(
        "spurious duplicate, spread ~P: worst lag {:.2} ms, {:.1}% of later samples > P/2 early",
        worst / MS,
        f64::from(early) * 100.0 / f64::from(total)
    );
    // Latency step: delay 0-1 ms, then 4-5 ms from sample 100.
    let (mut worst_error, mut worst_lag_seen) = (0.0_f64, f64::INFINITY);
    for seed in 1..=50 {
        let mut rng = Rng::new(seed);
        let mut filter = seeded(base, Delay::Uniform(MS), &mut rng);
        let mut k = 0;
        let start = base + Duration::from_secs(10);
        let stroke = Stroke::new(
            start,
            PERIOD,
            300,
            || {
                k += 1;
                rng.unit() * MS + if k > 100 { 4.0 * MS } else { 0.0 }
            },
            |_| false,
        );
        let run = stroke.run(&mut filter);
        worst_error = interval_errors(&run)
            .iter()
            .fold(worst_error, |a, e| a.max(e.abs()));
        worst_lag_seen = run
            .iter()
            .fold(worst_lag_seen, |a, &(t, _, o)| a.min(o - t));
    }
    println!(
        "latency step +4 ms: worst interval error {:.2} ms, worst lag {:.2} ms",
        worst_error / MS,
        worst_lag_seen / MS
    );
}

/// Frame rates for the batched delivery model.
const FRAMES: [f64; 2] = [1.0 / 60.0, 1.0 / 144.0];

/// Seeded strokes of every length delivered in frames of `frame` seconds.
fn batched_cases(frame: f64) -> Vec<Pooled> {
    let base = Instant::now();
    let mut pooled: Vec<Pooled> = LENGTHS.iter().map(|_| Pooled::new()).collect();
    for seed in 1..=SEEDS {
        let mut rng = Rng::new(seed);
        let mut filter = Dejitter::new();
        drop(Stroke::batched(base, 400, frame, &mut rng).run(&mut filter));
        for (n, (case, &length)) in (1..).zip(pooled.iter_mut().zip(&LENGTHS)) {
            let start = base + Duration::from_secs(10 * n);
            case.add(&Stroke::batched(start, length, frame, &mut rng).run(&mut filter));
        }
    }
    pooled
}

#[test]
fn frame_batched_delivery() {
    for frame in FRAMES {
        for (case, length) in batched_cases(frame).iter().zip(LENGTHS) {
            let what = format!("{frame} s frames, {length} samples: {}", case.summary());
            assert!(case.no_worse_share() >= 0.99, "worse than raw: {what}");
            assert!(case.lag.0 >= -0.05 * MS, "earlier than truth: {what}");
            // At 144 Hz the filter must actually improve on the receipts. At 60 Hz, frames
            // of about three samples defeat the seeded line and the output falls back
            // towards the receipt times, so only no-worse and not-early are held.
            if frame < 0.01 {
                let p99 = quantile_abs(&case.filter, 0.99);
                assert!(
                    p99 < quantile_abs(&case.raw, 0.99),
                    "no better than raw: {what}"
                );
            }
        }
    }
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
    for frame in FRAMES {
        for (case, length) in batched_cases(frame).iter().zip(LENGTHS) {
            println!(
                "batched {:.0} Hz N={length}: {}",
                1.0 / frame,
                case.summary()
            );
        }
    }
}

/// Outputs stay between the previous output and the receipt.
///
/// For any monotone receipts (steps of `0..=65_535` µs) with strokes ended anywhere, every
/// output is no later than its receipt and no earlier than the output before it, and the
/// period is unknown or positive.
#[test]
fn outputs_stay_between_the_last_output_and_the_receipt() {
    bolero::check!()
        .with_type::<Vec<(u16, bool)>>()
        .for_each(|steps| {
            let mut filter = Dejitter::new();
            let mut received = Instant::now();
            let mut last: Option<Instant> = None;
            for (n, &(delta_us, end_stroke)) in steps.iter().enumerate() {
                received += Duration::from_micros(delta_us.into());
                let out = filter.push(received);
                assert!(out <= received, "step {n}: after its receipt");
                assert!(last.is_none_or(|last| out >= last), "step {n}: went back");
                last = Some(out);
                if end_stroke {
                    filter.end_stroke();
                }
                let period = filter.period();
                assert!(
                    period.is_none_or(|p| p > Duration::ZERO),
                    "step {n}: {period:?}"
                );
            }
        });
}
