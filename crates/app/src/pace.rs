//! Waiting for a deadline precisely: the frame loop's waits until the next frame may start.
//!
//! A thread's sleep can run long. On Linux it overshoots by tens of microseconds; on macOS,
//! launched over SSH, sleeps ran about 50 % long (1 ms took 1.5, 16 took 24), so frames paced
//! by sleeping came at 41–47 fps instead of 60. A [`Pacer`] sleeps in steps sized to how long
//! sleeps have run, then spins out the last [`SPIN_MARGIN`]:
//!
//! - **Sleep:** with `remaining` to the deadline, it asks for `(remaining − SPIN_MARGIN) /
//!   stretch`, where the stretch is how much longer than asked sleeps have taken (each sleep's
//!   taken over asked: a longer one raises it at once, a shorter one lowers it a quarter of the
//!   way). It starts at [`FIRST_STRETCH`], the worst seen, so a first wait on a machine with long
//!   sleeps lands too; then it settles at what this machine does: about 1.0 on Linux, where a
//!   frame's wait is one sleep, and 1.5 on that macOS case, where it is two or three.
//! - **Spin:** within [`SPIN_MARGIN`] of the deadline it spins (`std::hint::spin_loop` and a yield,
//!   so other threads on the core still run) until the deadline. It never spins longer than that
//!   margin and a sleep too short to take ([`SHORTEST_SLEEP`], stretched), so a far deadline never
//!   burns a core.
//!
//! **Cost:** about [`SPIN_MARGIN`] of one core spinning each frame: 1 ms in 16.7 at 60 Hz, 6 %
//! of a core (the yield gives it to any thread that wants it). Measured on the Linux desktop
//! (Ryzen 9 5950X, 60 waits of 16.7 ms, the last 50 counted, `measure_real_waits`, three runs):
//! 1.08 sleeps a wait, 1.00 ms spun, landing 0.4–0.5 µs late on average and 1.0 µs at worst; the
//! stretch learnt was 1.004.
//!
//! The loop's waits for the upkeep, idle with no frame wanted, stay plain sleeps: nothing is
//! paced by them, and an idle app should cost nothing.

use std::time::{Duration, Instant};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// How close to the deadline the wait stops sleeping and spins.
pub(crate) const SPIN_MARGIN: Duration = Duration::from_millis(1);

/// The stretch a pacer starts with: sleeps taking 1.5 times as long as asked, the worst seen
/// (macOS, launched over SSH).
pub(crate) const FIRST_STRETCH: f64 = 1.5;

/// The most stretch believed: a sleep far longer than asked (the process stopped, the machine
/// suspended) says nothing about the next.
const MOST_STRETCH: f64 = 4.0;

/// Sleeps shorter than this teach nothing about the stretch: their overshoot is mostly the
/// timer's fixed latency (tens of microseconds), which the spin covers.
const SHORTEST_TELLING: Duration = Duration::from_millis(2);

/// The shortest sleep worth a wake-up: a shorter one is spun, so a spin can run to
/// [`SPIN_MARGIN`] plus this, stretched.
const SHORTEST_SLEEP: Duration = Duration::from_micros(100);

/// What a wait does next, given the time left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// The deadline has come: nothing.
    Done,
    /// Spin until the deadline: it is within [`SPIN_MARGIN`].
    Spin,
    /// Sleep this long, then look again.
    Sleep(Duration),
}

/// What the wait needs from the system: the time, a sleep, a moment's spin. [`SystemTimer`]
/// is the real one; tests give a model.
pub(crate) trait Timer {
    /// The time now.
    fn now(&mut self) -> Instant;

    /// Sleeps for at least `duration`.
    fn sleep(&mut self, duration: Duration);

    /// Spins a moment: a spin-loop hint, and a yield to any other thread wanting the core.
    fn spin(&mut self);
}

/// The system's clock and sleep.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SystemTimer;

impl Timer for SystemTimer {
    fn now(&mut self) -> Instant {
        Instant::now()
    }

    fn sleep(&mut self, duration: Duration) {
        std::thread::sleep(duration);
    }

    fn spin(&mut self) {
        std::hint::spin_loop();
        std::thread::yield_now();
    }
}

/// How a wait went.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Waited {
    /// Sleeps taken.
    pub(crate) sleeps: u32,
    /// Time spent spinning.
    pub(crate) spun: Duration,
    /// How late it ended: from the deadline to its end (zero if on time).
    pub(crate) late: Duration,
}

/// Waits for deadlines precisely, learning how long this machine's sleeps run (see the module
/// docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pacer {
    /// How much longer than asked sleeps take: taken over asked, at least 1.
    stretch: f64,
}

impl Default for Pacer {
    fn default() -> Self {
        Self {
            stretch: FIRST_STRETCH,
        }
    }
}

impl Pacer {
    /// What to do at `now` to reach `deadline`: a sleep shorter than [`SHORTEST_SLEEP`] is
    /// spun instead.
    pub(crate) fn plan(self, now: Instant, deadline: Instant) -> Step {
        let left = deadline.saturating_duration_since(now);
        if left.is_zero() {
            return Step::Done;
        }
        let sleep = left.saturating_sub(SPIN_MARGIN).div_f64(self.stretch);
        if sleep < SHORTEST_SLEEP {
            Step::Spin
        } else {
            Step::Sleep(sleep)
        }
    }

    /// Learns from a sleep that, asked for `asked`, took `took`.
    pub(crate) fn slept(&mut self, asked: Duration, took: Duration) {
        if asked < SHORTEST_TELLING {
            return;
        }
        let seen = (took.as_secs_f64() / asked.as_secs_f64()).clamp(1.0, MOST_STRETCH);
        self.stretch = if seen > self.stretch {
            seen
        } else {
            self.stretch + (seen - self.stretch) / 4.0
        };
    }

    /// How much longer than asked this machine's sleeps take, as learnt.
    #[cfg(test)]
    pub(crate) const fn stretch(self) -> f64 {
        self.stretch
    }

    /// Waits on `timer` until `deadline`: sleeps, then spins the last [`SPIN_MARGIN`]. Returns
    /// at once if the deadline has passed.
    pub(crate) fn wait_until(&mut self, timer: &mut impl Timer, deadline: Instant) -> Waited {
        let mut waited = Waited::default();
        loop {
            let now = timer.now();
            match self.plan(now, deadline) {
                Step::Done => {
                    waited.late = now.saturating_duration_since(deadline);
                    return waited;
                }
                Step::Sleep(asked) => {
                    timer.sleep(asked);
                    waited.sleeps += 1;
                    self.slept(asked, timer.now().saturating_duration_since(now));
                }
                Step::Spin => {
                    let mut at = now;
                    while at < deadline {
                        timer.spin();
                        at = timer.now();
                    }
                    waited.spun += at.saturating_duration_since(now);
                }
            }
        }
    }

    /// Spins on `timer` until `deadline`, if it is within [`SPIN_MARGIN`] (a window's loop,
    /// woken by its event loop's timer just before it); otherwise returns at once, leaving the
    /// sleep to the caller.
    pub(crate) fn spin_until(self, timer: &mut impl Timer, deadline: Instant) {
        let now = timer.now();
        if self.plan(now, deadline) != Step::Spin {
            return;
        }
        while timer.now() < deadline {
            timer.spin();
        }
    }
}
