//! The wait against a model of the system's timer, and (ignored) against the real one.

use super::*;

/// A model timer: made-up time that a sleep moves on by `stretch` times what it asks plus
/// `latency`, and a spin by `spin`.
#[derive(Debug)]
struct Model {
    /// The time now.
    now: Instant,
    /// How much longer than asked a sleep takes.
    stretch: f64,
    /// What every sleep takes beyond that.
    latency: Duration,
    /// What one spin takes.
    spin: Duration,
    /// Sleeps asked for, in order.
    asked: Vec<Duration>,
}

impl Model {
    /// A timer whose sleeps take `stretch` times as long as asked, plus 50 µs.
    fn stretching(stretch: f64) -> Self {
        Self {
            now: Instant::now(),
            stretch,
            latency: Duration::from_micros(50),
            spin: Duration::from_micros(2),
            asked: Vec::new(),
        }
    }
}

impl Timer for Model {
    fn now(&mut self) -> Instant {
        self.now
    }

    fn sleep(&mut self, duration: Duration) {
        self.asked.push(duration);
        self.now += duration.mul_f64(self.stretch) + self.latency;
    }

    fn spin(&mut self) {
        self.now += self.spin;
    }
}

/// The longest a wait may spin: the margin and a sleep too short to take, stretched.
const LONGEST_SPIN: Duration = SPIN_MARGIN.saturating_add(SHORTEST_SLEEP.saturating_mul(2));

/// A 60 Hz frame's interval.
const FRAME: Duration = Duration::from_micros(16_667);

/// The model's waits for `frames` frames in a row, each until a frame after the last ended.
fn frames(model: &mut Model, pacer: &mut Pacer, frames: usize) -> Vec<Waited> {
    (0..frames)
        .map(|_| {
            let deadline = model.now + FRAME;
            pacer.wait_until(model, deadline)
        })
        .collect()
}

#[test]
fn sleeps_that_run_half_again_as_long_still_land_on_the_deadline() {
    let mut model = Model::stretching(1.5);
    let mut pacer = Pacer::default();
    let waits = frames(&mut model, &mut pacer, 60);
    for (index, waited) in waits.iter().enumerate() {
        assert!(
            waited.late <= Duration::from_micros(200),
            "frame {index}: {waited:?}"
        );
        assert!(waited.spun <= LONGEST_SPIN, "frame {index}: {waited:?}");
    }
    assert!(
        (pacer.stretch() - 1.5).abs() < 0.05,
        "it learnt the stretch: {}",
        pacer.stretch()
    );
}

#[test]
fn on_a_machine_whose_sleeps_are_on_time_a_wait_settles_to_one_sleep_and_the_margin_spun() {
    let mut model = Model::stretching(1.0);
    let mut pacer = Pacer::default();
    let waits = frames(&mut model, &mut pacer, 60);
    for waited in &waits {
        assert!(waited.late <= Duration::from_micros(200), "{waited:?}");
    }
    let settled = waits.get(30..).expect("60 frames");
    for waited in settled {
        assert_eq!(waited.sleeps, 1, "{waited:?}");
        // The margin, less the sleep's 50 µs of latency.
        assert!(waited.spun <= LONGEST_SPIN, "{waited:?}");
        assert!(waited.spun >= SPIN_MARGIN / 2, "{waited:?}");
    }
    assert!(pacer.stretch() < 1.01, "{}", pacer.stretch());
}

#[test]
fn a_sleep_longer_than_learnt_raises_the_stretch_at_once_and_a_shorter_lowers_it_slowly() {
    let mut pacer = Pacer::default();
    let ms = Duration::from_millis;
    pacer.slept(ms(10), ms(30));
    assert!((pacer.stretch() - 3.0).abs() < 1e-9);
    pacer.slept(ms(10), ms(10));
    assert!((pacer.stretch() - 2.5).abs() < 1e-9, "a quarter of the way");
    pacer.slept(ms(10), ms(1_000));
    assert!(
        (pacer.stretch() - MOST_STRETCH).abs() < 1e-9,
        "a stall teaches little"
    );
    pacer.slept(ms(1), ms(2));
    assert!(
        (pacer.stretch() - MOST_STRETCH).abs() < 1e-9,
        "too short to tell"
    );
}

#[test]
fn a_past_deadline_returns_at_once_and_a_near_one_only_spins() {
    let mut model = Model::stretching(1.5);
    let mut pacer = Pacer::default();
    let start = model.now;
    let before = start.checked_sub(Duration::from_millis(1)).expect("a time");
    let past = pacer.wait_until(&mut model, before);
    assert_eq!(past.sleeps, 0);
    assert_eq!(past.spun, Duration::ZERO);
    assert_eq!(model.now, start, "no time spent");
    let near = start + SPIN_MARGIN / 2;
    let waited = pacer.wait_until(&mut model, near);
    assert_eq!(waited.sleeps, 0, "within the margin: no sleep");
    let over = model.now.checked_duration_since(near);
    assert!(over.is_some_and(|over| over <= model.spin), "{over:?}");
    assert_eq!(model.asked, []);
}

#[test]
fn a_far_deadline_is_slept_towards_never_spun_and_the_window_s_spin_waits_only_near_it() {
    let pacer = Pacer::default();
    let now = Instant::now();
    let far = now + Duration::from_secs(1);
    let Step::Sleep(asked) = pacer.plan(now, far) else {
        panic!("a sleep first");
    };
    let sleepable = Duration::from_secs(1).saturating_sub(SPIN_MARGIN);
    assert_eq!(asked, sleepable.div_f64(FIRST_STRETCH));
    assert_eq!(pacer.plan(now, now + SPIN_MARGIN), Step::Spin);
    assert_eq!(pacer.plan(now, now), Step::Done);
    let mut model = Model::stretching(1.0);
    let start = model.now;
    pacer.spin_until(&mut model, start + Duration::from_secs(1));
    assert_eq!(model.now, start, "far off: the caller sleeps");
    pacer.spin_until(&mut model, start + SPIN_MARGIN);
    assert!(model.now >= start + SPIN_MARGIN);
}

/// The real timer's waits: how late they land, and how long they spin. Prints its numbers; run
/// it on a quiet machine with `cargo nextest run -p app --run-ignored only measure_real_waits
/// --no-capture`.
#[test]
#[ignore = "a measurement on the real clock, not a check"]
fn measure_real_waits() {
    let mut timer = SystemTimer;
    let mut pacer = Pacer::default();
    let waits: Vec<Waited> = (0..60)
        .map(|_| {
            let deadline = Instant::now() + FRAME;
            pacer.wait_until(&mut timer, deadline)
        })
        .collect();
    let settled = waits.get(10..).expect("60 waits");
    let count = u32::try_from(settled.len()).expect("small");
    let late: Duration = settled.iter().map(|w| w.late).sum();
    let worst = settled.iter().map(|w| w.late).max().unwrap_or_default();
    let spun: Duration = settled.iter().map(|w| w.spun).sum();
    let sleeps: u32 = settled.iter().map(|w| w.sleeps).sum();
    println!(
        "stretch {:.3}; a wait: late {:?} on average, {worst:?} at worst; spun {:?}; {:.2} sleeps",
        pacer.stretch(),
        late / count,
        spun / count,
        f64::from(sleeps) / f64::from(count)
    );
    assert!(worst < Duration::from_millis(2), "{worst:?}");
}
