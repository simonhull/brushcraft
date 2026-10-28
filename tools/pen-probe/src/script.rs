//! The guided script one recording follows (`--script`): timed steps, each with a prompt, that
//! every sample is tagged with, so the report can read each step on its own.
//!
//! The script starts when the pen first comes over the window, after a short countdown, and
//! the window closes when the last step ends. [`Script`] is a pure state machine on the times it
//! is given; the probe prints each prompt to the terminal, puts it in the window's title and
//! clears the canvas at each step.

use std::time::{Duration, Instant};

/// A step of the guided script, or the time before and between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Step {
    /// Before the pen first comes over the window.
    Waiting,
    /// The countdown after it does.
    Countdown,
    /// Hovering, not touching.
    Hover,
    /// Long steady strokes: the speed-noise measure.
    Steady,
    /// Barrel button 1 clicked while hovering.
    Barrel1Hover,
    /// Barrel button 2 clicked while hovering.
    Barrel2Hover,
    /// Barrel button 1 held mid-stroke, then released, the tip still down.
    Barrel1Stroke,
    /// Barrel button 2 held mid-stroke, then released, the tip still down.
    Barrel2Stroke,
    /// After the last step, while the window closes.
    Done,
}

impl Step {
    /// Every step, in order.
    pub(crate) const ALL: [Self; 9] = [
        Self::Waiting,
        Self::Countdown,
        Self::Hover,
        Self::Steady,
        Self::Barrel1Hover,
        Self::Barrel2Hover,
        Self::Barrel1Stroke,
        Self::Barrel2Stroke,
        Self::Done,
    ];
    /// How long the countdown before the first timed step lasts.
    pub(crate) const COUNTDOWN: Duration = Duration::from_secs(3);
    /// The timed steps, in the order the script runs them.
    pub(crate) const TIMED: [Self; 6] = [
        Self::Hover,
        Self::Steady,
        Self::Barrel1Hover,
        Self::Barrel2Hover,
        Self::Barrel1Stroke,
        Self::Barrel2Stroke,
    ];

    /// Its name in the CSV and the report.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Waiting => "waiting",
            Self::Countdown => "countdown",
            Self::Hover => "hover",
            Self::Steady => "steady",
            Self::Barrel1Hover => "barrel1-hover",
            Self::Barrel2Hover => "barrel2-hover",
            Self::Barrel1Stroke => "barrel1-stroke",
            Self::Barrel2Stroke => "barrel2-stroke",
            Self::Done => "done",
        }
    }

    /// The step [`Step::name`] names.
    pub(crate) fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// How long it lasts; zero for the untimed ones.
    pub(crate) const fn duration(self) -> Duration {
        Duration::from_secs(match self {
            Self::Hover => 8,
            Self::Steady => 30,
            Self::Barrel1Hover | Self::Barrel2Hover => 10,
            Self::Barrel1Stroke | Self::Barrel2Stroke => 20,
            Self::Waiting | Self::Countdown | Self::Done => 0,
        })
    }

    /// What the user is asked to do.
    pub(crate) const fn prompt(self) -> &'static str {
        match self {
            Self::Waiting => "Bring the pen over the window to start.",
            Self::Countdown => "Get ready: the first step starts in 3 seconds.",
            Self::Hover => "HOVER: move the pen over the window without touching it.",
            Self::Steady => {
                "STEADY STROKES: draw long strokes at an even speed, lifting between them. Vary \
                 the speed from stroke to stroke, but keep each one even."
            }
            Self::Barrel1Hover => {
                "BARREL 1 WHILE HOVERING: without touching, click barrel button 1 (the lower \
                 one, nearer the tip) about once a second."
            }
            Self::Barrel2Hover => {
                "BARREL 2 WHILE HOVERING: without touching, click barrel button 2 (the upper \
                 one) about once a second."
            }
            Self::Barrel1Stroke => {
                "BARREL 1 MID-STROKE: touch down and draw; while drawing, press and hold barrel \
                 button 1, keep drawing, release it, keep drawing, then lift. Repeat."
            }
            Self::Barrel2Stroke => {
                "BARREL 2 MID-STROKE: touch down and draw; while drawing, press and hold barrel \
                 button 2, keep drawing, release it, keep drawing, then lift. Repeat."
            }
            Self::Done => "Done: the window closes.",
        }
    }

    /// The canvas colour of the progress bar during the step, `0RGB`.
    pub(crate) const fn colour(self) -> u32 {
        match self {
            Self::Waiting | Self::Done => 0x0060_6060,
            Self::Countdown => 0x00C0_C040,
            Self::Hover => 0x0040_80E0,
            Self::Steady => 0x0040_C060,
            Self::Barrel1Hover | Self::Barrel2Hover => 0x00E0_8030,
            Self::Barrel1Stroke | Self::Barrel2Stroke => 0x00D0_40A0,
        }
    }

    /// The timed steps' total, with the countdown: how long a run takes once the pen arrives.
    pub(crate) fn total() -> Duration {
        Self::TIMED
            .iter()
            .map(|s| s.duration())
            .fold(Self::COUNTDOWN, |a, b| a + b)
    }
}

/// Where the script is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// The pen hasn't come yet.
    Waiting,
    /// In the countdown, or a timed step (`step` indexes [`Step::TIMED`], `None` for the
    /// countdown), which began at `began`.
    Running {
        /// The timed step; `None` in the countdown.
        step: Option<usize>,
        /// When it began.
        began: Instant,
    },
    /// Past the last step.
    Done,
}

/// The guided script's progress: see the module notes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Script {
    /// Where it is.
    state: State,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            state: State::Waiting,
        }
    }
}

impl Script {
    /// The step it is in.
    pub(crate) fn step(&self) -> Step {
        match self.state {
            State::Waiting => Step::Waiting,
            State::Running { step: None, .. } => Step::Countdown,
            State::Running { step: Some(i), .. } => {
                Step::TIMED.get(i).copied().unwrap_or(Step::Done)
            }
            State::Done => Step::Done,
        }
    }

    /// When the current step ends; `None` while waiting or done.
    pub(crate) fn ends(&self) -> Option<Instant> {
        match self.state {
            State::Running { began, .. } => Some(began + self.length()),
            State::Waiting | State::Done => None,
        }
    }

    /// How long the current step lasts.
    fn length(&self) -> Duration {
        let step = self.step();
        if step == Step::Countdown {
            Step::COUNTDOWN
        } else {
            step.duration()
        }
    }

    /// How far through the current step `now` is, `0` to `1`.
    pub(crate) fn progress(&self, now: Instant) -> f64 {
        match self.state {
            State::Running { began, .. } => (now.saturating_duration_since(began).as_secs_f64()
                / self.length().as_secs_f64().max(1e-9))
            .clamp(0.0, 1.0),
            State::Waiting => 0.0,
            State::Done => 1.0,
        }
    }

    /// The pen came over the window at `now`: the countdown starts if it hasn't. Returns whether
    /// the step changed.
    pub(crate) fn pen_arrived(&mut self, now: Instant) -> bool {
        if self.state != State::Waiting {
            return false;
        }
        self.state = State::Running {
            step: None,
            began: now,
        };
        true
    }

    /// Moves on to whichever step `now` falls in. Returns whether the step changed.
    pub(crate) fn advance(&mut self, now: Instant) -> bool {
        let mut changed = false;
        while let (Some(ends), State::Running { step, .. }) = (self.ends(), self.state)
            && now >= ends
        {
            let next = step.map_or(0, |i| i + 1);
            self.state = if next < Step::TIMED.len() {
                State::Running {
                    step: Some(next),
                    began: ends,
                }
            } else {
                State::Done
            };
            changed = true;
        }
        changed
    }

    /// Whether the last step has ended.
    pub(crate) fn done(&self) -> bool {
        self.state == State::Done
    }
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for step in Step::ALL {
            assert_eq!(Step::parse(step.name()), Some(step));
        }
        assert_eq!(Step::parse("nap"), None);
    }

    #[test]
    fn the_script_waits_for_the_pen_then_runs_each_step_in_turn() {
        let t0 = Instant::now();
        let mut script = Script::default();
        assert_eq!(script.step(), Step::Waiting);
        assert!(!script.advance(t0 + Duration::from_secs(60)), "waits");
        assert!(script.pen_arrived(t0));
        assert!(!script.pen_arrived(t0), "once");
        assert_eq!(script.step(), Step::Countdown);
        assert_eq!(script.ends(), Some(t0 + Step::COUNTDOWN));
        let mut at = t0 + Step::COUNTDOWN;
        for step in Step::TIMED {
            assert!(script.advance(at), "into {step:?}");
            assert_eq!(script.step(), step);
            assert!((script.progress(at)).abs() < 1e-9, "at its start");
            at += step.duration();
        }
        assert!(script.advance(at));
        assert!(script.done());
        assert_eq!(script.step(), Step::Done);
        assert_eq!(at - t0, Step::total(), "the total");
    }

    #[test]
    fn a_late_advance_skips_to_the_step_now_falls_in() {
        let t0 = Instant::now();
        let mut script = Script::default();
        script.pen_arrived(t0);
        let into_steady = t0 + Step::COUNTDOWN + Step::Hover.duration() + Duration::from_secs(1);
        assert!(script.advance(into_steady));
        assert_eq!(script.step(), Step::Steady);
        let half = script.progress(into_steady);
        assert!((half - 1.0 / 30.0).abs() < 1e-6, "{half}");
    }
}
