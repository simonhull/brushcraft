//! When to render: the frame scheduler, a pure state machine.
//!
//! Frames are made on demand (the M1 spec's "When to render"): only when input arrived, the
//! window needs drawing, a script asks, or something is converging (store jobs left to run,
//! tiles still to upload, later morphs and accumulation). With none of those there is no
//! frame, and about zero GPU work. The scheduler is told why frames are wanted
//! ([`Scheduler::want`]) and when one starts and is shown, and answers what to do next
//! ([`Scheduler::next`]); it never looks at a clock or a window itself, so every rule is
//! tested with made-up times.
//!
//! Its rules:
//!
//! - One frame at a time: none starts while the render thread has the last one.
//! - Paced to the display: a frame starts no sooner than one refresh interval after the last.
//! - A frame's store jobs run until three quarters of an interval after it starts, leaving the rest
//!   for handing it over and handling the window's events.
//! - A frame that couldn't be shown (the surface changed) is wanted again; a hidden window's is
//!   not, until it is shown.
//! - While idle, the loop still wakes now and then for upkeep (the recovery writer's flushes), with
//!   no frame.

use std::fmt;
use std::time::{Duration, Instant};

use render::Outcome;

/// Why a frame is wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reason {
    /// Input moved the view or touched the model.
    Input,
    /// The window needs drawing: first shown, resized, uncovered, or the last frame couldn't
    /// be shown.
    Exposed,
    /// Something is converging: store jobs left, tiles to upload.
    Converging,
    /// A scripted run (`--frames`) wants every frame.
    Scripted,
}

impl Reason {
    /// This reason's bit in [`Reasons`].
    const fn bit(self) -> u8 {
        match self {
            Self::Input => 1,
            Self::Exposed => 1 << 1,
            Self::Converging => 1 << 2,
            Self::Scripted => 1 << 3,
        }
    }
}

/// A set of [`Reason`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Reasons(u8);

impl Reasons {
    /// Whether the set is empty.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether `reason` is in the set.
    pub(crate) const fn contains(self, reason: Reason) -> bool {
        self.0 & reason.bit() != 0
    }

    /// The set with `reason` added.
    #[must_use]
    pub(crate) const fn with(self, reason: Reason) -> Self {
        Self(self.0 | reason.bit())
    }
}

impl fmt::Display for Reasons {
    /// The reasons' names, `+`-separated, or `none`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let all = [
            (Reason::Input, "input"),
            (Reason::Exposed, "exposed"),
            (Reason::Converging, "converging"),
            (Reason::Scripted, "scripted"),
        ];
        let names: Vec<&str> = all
            .into_iter()
            .filter(|&(reason, _)| self.contains(reason))
            .map(|(_, name)| name)
            .collect();
        if names.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&names.join("+"))
        }
    }
}

/// What the frame loop should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Next {
    /// Start a frame now; its store jobs run until `deadline`.
    Frame {
        /// When the frame's jobs stop.
        deadline: Instant,
    },
    /// A frame is wanted, but not before `until` (one interval after the last began).
    Wait {
        /// When the frame may start.
        until: Instant,
    },
    /// The render thread has the last frame: wait until it is shown.
    Render,
    /// No frame is wanted. Sleep until an event, or until `upkeep`, when the store's upkeep is
    /// due.
    Idle {
        /// When upkeep is next due.
        upkeep: Instant,
    },
}

/// How often the loop wakes for upkeep while no frames are made.
pub(crate) const UPKEEP_INTERVAL: Duration = Duration::from_millis(250);

/// The share of an interval a frame's store jobs may take, in quarters.
const JOB_QUARTERS: u32 = 3;

/// Decides when frames are made.
#[derive(Debug, Clone)]
pub(crate) struct Scheduler {
    /// The display's refresh interval: frames start at most this often.
    interval: Duration,
    /// Why a frame is wanted, since the last one started.
    wanted: Reasons,
    /// Whether the render thread has a frame it hasn't finished.
    in_flight: bool,
    /// When the last frame started.
    last_start: Option<Instant>,
    /// When upkeep last ran (a frame runs it too).
    upkept: Instant,
}

impl Scheduler {
    /// A scheduler pacing frames to `interval`, wanting none yet, at `now`.
    pub(crate) const fn new(interval: Duration, now: Instant) -> Self {
        Self {
            interval,
            wanted: Reasons(0),
            in_flight: false,
            last_start: None,
            upkept: now,
        }
    }

    /// The refresh interval frames are paced to.
    pub(crate) const fn interval(&self) -> Duration {
        self.interval
    }

    /// Paces frames to `interval` from now on (the window moved to another display).
    pub(crate) const fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Asks for a frame, for `reason`.
    pub(crate) const fn want(&mut self, reason: Reason) {
        self.wanted = self.wanted.with(reason);
    }

    /// Why a frame is wanted now.
    #[cfg(test)]
    pub(crate) const fn wanted(&self) -> Reasons {
        self.wanted
    }

    /// What to do at `now`.
    pub(crate) fn next(&self, now: Instant) -> Next {
        if self.in_flight {
            return Next::Render;
        }
        if self.wanted.is_empty() {
            return Next::Idle {
                upkeep: self.upkept + UPKEEP_INTERVAL,
            };
        }
        if let Some(last) = self.last_start
            && now < last + self.interval
        {
            return Next::Wait {
                until: last + self.interval,
            };
        }
        Next::Frame {
            deadline: now + self.interval * JOB_QUARTERS / 4,
        }
    }

    /// A frame started at `now`: what was wanted is taken (and returned), and the frame is in
    /// flight until [`Scheduler::shown`].
    pub(crate) fn started(&mut self, now: Instant) -> Reasons {
        self.in_flight = true;
        self.last_start = Some(now);
        self.upkept = now;
        std::mem::take(&mut self.wanted)
    }

    /// The render thread finished the frame in flight.
    pub(crate) const fn shown(&mut self, shown: Outcome) {
        self.in_flight = false;
        if matches!(shown, Outcome::Retry) {
            self.want(Reason::Exposed);
        }
    }

    /// Upkeep ran at `now`, with no frame.
    pub(crate) const fn upkept(&mut self, now: Instant) {
        self.upkept = now;
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A 60 Hz display's interval, rounded to whole microseconds.
    const INTERVAL: Duration = Duration::from_micros(16_667);

    /// A scheduler at 60 Hz, and the time it was made.
    fn scheduler() -> (Scheduler, Instant) {
        let start = Instant::now();
        (Scheduler::new(INTERVAL, start), start)
    }

    #[test]
    fn nothing_wanted_means_no_frame_only_upkeep() {
        let (scheduler, start) = scheduler();
        assert_eq!(
            scheduler.next(start + Duration::from_secs(10)),
            Next::Idle {
                upkeep: start + UPKEEP_INTERVAL
            },
            "however long it has been"
        );
    }

    #[test]
    fn upkeep_moves_the_next_upkeep_on() {
        let (mut scheduler, start) = scheduler();
        let later = start + Duration::from_secs(1);
        scheduler.upkept(later);
        assert_eq!(
            scheduler.next(later),
            Next::Idle {
                upkeep: later + UPKEEP_INTERVAL
            }
        );
    }

    #[test]
    fn input_starts_a_frame_at_once_with_three_quarters_of_an_interval_for_jobs() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Input);
        assert_eq!(
            scheduler.next(start),
            Next::Frame {
                deadline: start + INTERVAL * 3 / 4
            }
        );
    }

    #[test]
    fn one_frame_at_a_time() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Input);
        let taken = scheduler.started(start);
        assert!(taken.contains(Reason::Input), "{taken}");
        assert!(scheduler.wanted().is_empty(), "taken by the frame");
        scheduler.want(Reason::Input);
        let much_later = start + INTERVAL * 10;
        assert_eq!(scheduler.next(much_later), Next::Render, "in flight");
        scheduler.shown(Outcome::Presented);
        assert!(
            matches!(scheduler.next(much_later), Next::Frame { .. }),
            "the input that came meanwhile gets its frame"
        );
    }

    #[test]
    fn frames_are_paced_to_the_refresh_interval() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Input);
        scheduler.started(start);
        scheduler.shown(Outcome::Presented);
        scheduler.want(Reason::Input);
        let soon = start + Duration::from_millis(5);
        assert_eq!(
            scheduler.next(soon),
            Next::Wait {
                until: start + INTERVAL
            }
        );
        assert_eq!(
            scheduler.next(start + INTERVAL),
            Next::Frame {
                deadline: start + INTERVAL + INTERVAL * 3 / 4
            },
            "due at the interval"
        );
    }

    #[test]
    fn converging_keeps_frames_coming_until_it_stops() {
        let (mut scheduler, start) = scheduler();
        let mut now = start;
        for _ in 0..3 {
            scheduler.want(Reason::Converging);
            assert!(matches!(scheduler.next(now), Next::Frame { .. }));
            scheduler.started(now);
            scheduler.shown(Outcome::Presented);
            now += INTERVAL;
        }
        assert!(
            matches!(scheduler.next(now), Next::Idle { .. }),
            "converged: no reason left, no frame"
        );
    }

    #[test]
    fn a_frame_not_shown_is_wanted_again_unless_the_window_is_hidden() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Exposed);
        scheduler.started(start);
        scheduler.shown(Outcome::Retry);
        assert!(scheduler.wanted().contains(Reason::Exposed), "retried");
        scheduler.started(start + INTERVAL);
        scheduler.shown(Outcome::Hidden);
        assert!(
            matches!(scheduler.next(start + INTERVAL * 2), Next::Idle { .. }),
            "hidden: no frames until it is shown"
        );
    }

    #[test]
    fn a_frame_runs_upkeep() {
        let (mut scheduler, start) = scheduler();
        let later = start + Duration::from_secs(3);
        scheduler.want(Reason::Scripted);
        scheduler.started(later);
        scheduler.shown(Outcome::Presented);
        assert_eq!(
            scheduler.next(later),
            Next::Idle {
                upkeep: later + UPKEEP_INTERVAL
            }
        );
    }

    #[test]
    fn the_interval_can_change() {
        let (mut scheduler, start) = scheduler();
        let slower = Duration::from_millis(40);
        scheduler.set_interval(slower);
        assert_eq!(scheduler.interval(), slower);
        scheduler.want(Reason::Input);
        scheduler.started(start);
        scheduler.shown(Outcome::Presented);
        scheduler.want(Reason::Input);
        assert_eq!(
            scheduler.next(start + INTERVAL),
            Next::Wait {
                until: start + slower
            }
        );
    }

    #[test]
    fn reasons_name_themselves() {
        assert_eq!(Reasons::default().to_string(), "none");
        let both = Reasons::default()
            .with(Reason::Scripted)
            .with(Reason::Input);
        assert_eq!(both.to_string(), "input+scripted");
        assert!(!both.contains(Reason::Converging));
    }
}
