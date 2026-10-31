//! The path against simulated pens.
//!
//! The `#[ignore]`d `report` test is a report, not a check: it prints the filter's measured
//! smoothing and lag and is run by hand (`cargo nextest run -p stroke --run-ignored only --
//! report --no-capture`).

use super::*;
use crate::Unit;

/// A pen at 200 Hz: a sample every 5 ms.
const PERIOD: f64 = 0.005;

/// Xorshift64, seeded through splitmix64 so that small seeds don't start near zero.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    /// A value in `-1..1`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "a 53-bit integer converts to f64 exactly"
    )]
    fn signed(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        2.0 * ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) - 1.0
    }
}

fn sample(x: f64, y: f64, time: f64) -> Sample {
    Sample::new(ScreenPoint::new(x, y).unwrap(), time, Unit::ONE).unwrap()
}

/// `count` samples of a pen moving right along `y = 0` at `speed` px/s, each coordinate off by
/// up to `jitter` px, the jitter drawn from the seed `noise`.
fn line(speed: f64, jitter: f64, count: u32, noise: u64) -> Vec<Sample> {
    let mut rng = Rng::new(noise);
    (0..count)
        .map(|i| {
            let t = f64::from(i) * PERIOD;
            sample(speed * t + jitter * rng.signed(), jitter * rng.signed(), t)
        })
        .collect()
}

/// The root-mean-square distance from the line `y = 0` of `points`, skipping the first `skip`.
fn rms_off_line(points: &[ScreenPoint], skip: usize) -> f64 {
    let tail = points.get(skip..).unwrap_or_default();
    let sum: f64 = tail.iter().map(|p| p.y() * p.y()).sum();
    (sum / f64::from(u32::try_from(tail.len()).unwrap())).sqrt()
}

/// The brush points of `samples` through a fresh path with `settings`.
fn run(settings: PathSettings, samples: &[Sample]) -> Vec<ScreenPoint> {
    let mut path = Path::new(settings);
    samples.iter().map(|&s| path.push(s).position()).collect()
}

/// The jitter left after filtering a slow line, as a fraction of the input's: (input RMS, output
/// RMS) about the true line, after the first 100 ms.
fn slow_smoothing(settings: PathSettings, speed: f64) -> (f64, f64) {
    let samples = line(speed, 1.0, 400, 1);
    let raw: Vec<ScreenPoint> = samples.iter().map(Sample::position).collect();
    let out = run(settings, &samples);
    (rms_off_line(&raw, 20), rms_off_line(&out, 20))
}

/// The lag behind the pen of a clean line at `speed` px/s: (worst after the first `settle`
/// seconds, worst overall), in pixels.
fn lag(settings: PathSettings, speed: f64, settle: f64) -> (f64, f64) {
    let samples = line(speed, 0.0, 200, 0);
    let out = run(settings, &samples);
    let lags = samples
        .iter()
        .zip(&out)
        .map(|(s, o)| (s.time(), s.position().distance(*o)));
    lags.fold((0.0_f64, 0.0_f64), |(settled, all), (t, d)| {
        (
            if t >= settle { settled.max(d) } else { settled },
            all.max(d),
        )
    })
}

#[test]
fn a_slow_jittery_line_comes_out_smoother() {
    let (input, output) = slow_smoothing(PathSettings::default(), 20.0);
    assert!(
        output < 0.5 * input,
        "jitter RMS {output:.3} px out of {input:.3} px in"
    );
}

#[test]
fn a_fast_lines_lag_stays_under_the_bound() {
    let settings = PathSettings::default();
    let bound = settings.lag_bound();
    assert!((bound - 3.183).abs() < 1e-3, "the default bound: {bound}");
    for speed in [300.0, 1000.0, 3000.0, 10_000.0] {
        let (settled, _) = lag(settings, speed, 0.3);
        assert!(
            settled < bound,
            "{speed} px/s: {settled:.2} px behind after 300 ms, bound {bound:.2}"
        );
    }
}

#[test]
fn a_still_pen_is_not_moved() {
    let still: Vec<Sample> = (0..50)
        .map(|i| sample(10.0, 20.0, f64::from(i) * PERIOD))
        .collect();
    let out = run(PathSettings::default(), &still);
    assert!(
        out.iter()
            .all(|&p| p == ScreenPoint::new(10.0, 20.0).unwrap()),
        "a pen at rest stays exactly where it is"
    );
}

#[test]
fn the_lazy_string_keeps_its_length() {
    let settings = PathSettings::default().with_lazy(25.0).unwrap();
    let samples = line(1000.0, 0.0, 100, 0);
    let mut path = Path::new(settings);
    let first = path.push(*samples.first().unwrap()).position();
    assert_eq!(
        first,
        samples.first().unwrap().position(),
        "starts at the pen"
    );
    // The same filter without the string: the point the string is tied to.
    let pen = run(PathSettings::default(), &samples);
    let mut taut = 0;
    for (s, p) in samples.iter().zip(&pen).skip(1) {
        let brush = path.push(*s).position();
        let length = brush.distance(*p);
        assert!(
            length <= 25.0 + 1e-9,
            "never longer than the string: {length}"
        );
        if p.distance(first) > 25.0 {
            assert!((length - 25.0).abs() < 1e-9, "taut once pulled: {length}");
            taut += 1;
        } else {
            assert_eq!(brush, first, "slack: the brush waits");
        }
    }
    assert!(taut > 50, "the string was pulled: {taut} samples");
}

#[test]
fn end_completes_the_path_to_the_last_sample() {
    let settings = PathSettings::default().with_lazy(40.0).unwrap();
    let samples = line(5000.0, 0.5, 30, 3);
    let mut path = Path::new(settings);
    let mut last_brush = None;
    for &s in &samples {
        last_brush = Some(path.push(s));
    }
    let last = *samples.last().unwrap();
    assert_ne!(
        last_brush.map(|b| b.position()),
        Some(last.position()),
        "the filter and the string held the brush back"
    );
    assert_eq!(path.end(), Some(last), "the tail is the last raw sample");
    assert_eq!(path.end(), None, "and the path is ready for a new stroke");
    let restarted = path.push(sample(1.0, 2.0, 99.0));
    assert_eq!(
        restarted.position(),
        ScreenPoint::new(1.0, 2.0).unwrap(),
        "a new stroke starts at its own first sample"
    );
}

#[test]
fn a_repeated_time_holds_the_brush_but_keeps_the_tail() {
    let mut path = Path::new(PathSettings::default());
    path.push(sample(0.0, 0.0, 1.0));
    let moved = path.push(sample(5.0, 0.0, 1.005)).position();
    let held = path.push(sample(9.0, 0.0, 1.005)).position();
    assert_eq!(held, moved, "no time passed: the brush holds");
    assert_eq!(
        path.end().map(|s| s.position()),
        Some(ScreenPoint::new(9.0, 0.0).unwrap()),
        "the tail is still the newest sample"
    );
}

#[test]
fn readings_pass_through_with_the_smoothed_position() {
    let mut path = Path::new(PathSettings::default());
    path.push(sample(0.0, 0.0, 0.0));
    let pressed = sample(10.0, 0.0, PERIOD)
        .with_buttons(crate::Buttons::PRIMARY)
        .with_twist(crate::Twist::new(1.0).unwrap());
    let out = path.push(pressed);
    assert_eq!(out.time(), pressed.time(), "time");
    assert_eq!(out.buttons(), pressed.buttons(), "buttons");
    assert_eq!(out.twist(), pressed.twist(), "twist");
    assert_ne!(
        out.position(),
        pressed.position(),
        "the position is smoothed"
    );
}

#[test]
fn settings_are_validated() {
    assert!(
        PathSettings::new(0.0, 0.05, 5.0, 0.0).is_err(),
        "zero cutoff"
    );
    assert!(
        PathSettings::new(1.0, -0.1, 5.0, 0.0).is_err(),
        "negative slope"
    );
    assert!(PathSettings::new(1.0, 0.05, f64::NAN, 0.0).is_err(), "NaN");
    assert!(
        PathSettings::new(1.0, 0.05, 5.0, -1.0).is_err(),
        "negative lazy"
    );
    assert!(
        PathSettings::new(1.0, 0.0, 5.0, 0.0).is_ok(),
        "no slope: a plain low-pass"
    );
}

#[test]
fn any_samples_give_finite_brush_points() {
    bolero::check!()
        .with_type::<Vec<(i16, i16, u16)>>()
        .for_each(|steps| {
            let mut path = Path::new(PathSettings::default().with_lazy(3.0).unwrap());
            let mut time = 0.0;
            for &(x, y, dt) in steps {
                time += f64::from(dt) * 1e-5;
                let s = sample(f64::from(x) * 50.0, f64::from(y) * 50.0, time);
                let out = path.push(s).position();
                assert!(out.x().is_finite() && out.y().is_finite(), "{out:?}");
            }
        });
}

/// Prints the default filter's smoothing and lag, for the module notes and the report.
#[test]
#[ignore = "a report, run by hand"]
fn report() {
    let settings = PathSettings::default();
    for speed in [5.0, 20.0, 50.0] {
        let (input, output) = slow_smoothing(settings, speed);
        println!(
            "{speed:>6} px/s, ±1 px jitter: RMS {input:.3} -> {output:.3} px ({:.0} %)",
            100.0 * output / input
        );
    }
    println!("lag bound {:.2} px", settings.lag_bound());
    for speed in [100.0, 300.0, 1000.0, 3000.0, 10_000.0] {
        let (settled, worst) = lag(settings, speed, 0.3);
        println!("{speed:>6} px/s: lag {settled:.2} px settled, {worst:.2} px worst at the start");
    }
}
