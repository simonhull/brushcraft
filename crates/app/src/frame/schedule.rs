//! When to render: the frame scheduler, a pure state machine.
//!
//! Frames are made on demand (the M1 spec's "When to render"): only when input arrived, the
//! window needs drawing, a script asks, how the scene is shown changed, or something is
//! converging: store jobs left to run, uploads in flight, requests pending, a morph under way,
//! an image still accumulating samples. With none of those there is no frame, and about zero GPU
//! work. The scheduler is told why frames are wanted ([`Scheduler::want`]) and when one starts and
//! is shown, and answers what to do next ([`Scheduler::next`]); it never looks at a clock or a
//! window itself, so every rule is tested with made-up times.
//!
//! Its rules:
//!
//! - One frame at a time: none starts while the render thread has the last one.
//! - Never outrunning the GPU: none starts while the GPU has [`render::FRAMES_IN_FLIGHT`] frames it
//!   hasn't finished (the frame the render thread has counts among them). The loop learns of
//!   finished frames by polling the device, never by waiting on it ([`Scheduler::gpu_finished`]),
//!   and does its jobs and upkeep meanwhile ([`Next::Gpu`]). So when the GPU is slower than the
//!   display, frames come at the GPU's rate, input waits behind at most that many frames, and each
//!   frame's readbacks find a buffer.
//! - Paced to the display: a frame starts no sooner than one refresh interval after the last. The
//!   pace is waited for first, precisely, so a GPU keeping up costs no wait of its own.
//! - A frame's store jobs run until three quarters of an interval after it starts, leaving the rest
//!   for handling the window's events.
//! - A frame that couldn't be shown (the surface changed) is wanted again at once.
//! - While the window is hidden (its last frame came back [`Outcome::Hidden`]), frames still wanted
//!   are tried only every [`UPKEEP_INTERVAL`], not every refresh, until one is shown or the window
//!   is exposed: so a hidden window costs next to nothing, yet notices it is visible again even if
//!   no event says so.
//! - Upkeep (the recovery writer's flushes) is due every [`UPKEEP_INTERVAL`] after the last, which
//!   a frame also runs; while idle the loop wakes for it, with no frame.
//! - A frame whose cut changed ([`Scheduler::cut_changed`], heard a frame or two later from its
//!   readback) keeps frames coming for the morph's duration after it: until a frame starts at or
//!   past that frame's time plus the morph, so the last frame drawn shows the morph finished. The
//!   duration is the raster's ([`Scheduler::with_morph`]).
//! - While the camera is still the renderer accumulates samples (`render::aa`): the frame loop asks
//!   for so many frames more after a frame that may restart or reset the image
//!   ([`Scheduler::accumulate`]), and they come as [`Reason::Accumulating`], one fewer each frame
//!   started, until none is left.

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
    /// Something is converging: store jobs left, epochs in the feed to take.
    Converging,
    /// Uploads are in flight: the floor not all up, an epoch still switching.
    Uploads,
    /// Requests are pending: requested tiles still to come, or a request readback still to read.
    Requests,
    /// A level change is morphing: the cut changed within the morph's duration.
    Morphing,
    /// A scripted run (`--frames`) wants every frame.
    Scripted,
    /// How the scene is shown changed (the material, the wireframe, the debug view): a frame
    /// wanted for this alone shades the last scene again, with no raster.
    Look,
    /// The pointer moved, so the overlay's pick line changes ([`crate::pick`]): like
    /// [`Reason::Look`], a frame wanted for this alone shades the last scene again.
    Pick,
    /// The image is still accumulating samples with the camera still (`render::aa`): wanted for
    /// as many frames as [`Scheduler::accumulate`] last asked.
    Accumulating,
}

impl Reason {
    /// This reason's bit in [`Reasons`].
    const fn bit(self) -> u16 {
        match self {
            Self::Input => 1,
            Self::Exposed => 1 << 1,
            Self::Converging => 1 << 2,
            Self::Uploads => 1 << 3,
            Self::Requests => 1 << 4,
            Self::Morphing => 1 << 5,
            Self::Scripted => 1 << 6,
            Self::Look => 1 << 7,
            Self::Pick => 1 << 8,
            Self::Accumulating => 1 << 9,
        }
    }
}

/// A set of [`Reason`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Reasons(u16);

impl Reasons {
    /// Whether the set is empty.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether `reason` is in the set.
    pub(crate) const fn contains(self, reason: Reason) -> bool {
        self.0 & reason.bit() != 0
    }

    /// Whether every reason in the set is also in `other`.
    pub(crate) const fn within(self, other: Self) -> bool {
        self.0 & !other.0 == 0
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
            (Reason::Uploads, "uploads"),
            (Reason::Requests, "requests"),
            (Reason::Morphing, "morphing"),
            (Reason::Scripted, "scripted"),
            (Reason::Look, "look"),
            (Reason::Pick, "pick"),
            (Reason::Accumulating, "accumulating"),
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
    /// A frame is due, but the GPU has [`render::FRAMES_IN_FLIGHT`] frames it hasn't finished:
    /// poll it (never wait on it), and do the store's jobs and upkeep meanwhile, until it
    /// finishes one ([`Scheduler::gpu_finished`]).
    Gpu,
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
    /// Frames started, ever: the last one's number.
    started: u64,
    /// The newest frame the GPU has finished (and every one before it); 0 for none.
    gpu_finished: u64,
    /// The most frames the GPU may have unfinished as a frame starts, the new one not counted:
    /// [`render::FRAMES_IN_FLIGHT`].
    gpu_bound: u64,
    /// When the last frame started.
    last_start: Option<Instant>,
    /// When upkeep last ran (a frame runs it too).
    upkept: Instant,
    /// Whether the window's last frame came back hidden, and none has been shown since.
    hidden: bool,
    /// How long a level change morphs: the raster's duration.
    morph: Duration,
    /// While a morph is under way, when it ends: frames are wanted until one starts then or
    /// later.
    morph_end: Option<Instant>,
    /// Frames still wanted for accumulation.
    accumulating: u32,
}

impl Scheduler {
    /// A scheduler pacing frames to `interval`, wanting none yet, at `now`, with no morph.
    pub(crate) const fn new(interval: Duration, now: Instant) -> Self {
        Self {
            interval,
            wanted: Reasons(0),
            in_flight: false,
            started: 0,
            gpu_finished: 0,
            gpu_bound: render::FRAMES_IN_FLIGHT as u64,
            last_start: None,
            upkept: now,
            hidden: false,
            morph: Duration::ZERO,
            morph_end: None,
            accumulating: 0,
        }
    }

    /// The same scheduler, keeping at most `bound` frames unfinished on the GPU instead of
    /// [`render::FRAMES_IN_FLIGHT`]: for a model of what happens without the bound.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn with_gpu_bound(self, bound: u64) -> Self {
        Self {
            gpu_bound: bound,
            ..self
        }
    }

    /// The same scheduler, a level change morphing over `morph` (the raster's duration,
    /// `render::raster::RasterConfig::morph`).
    #[must_use]
    pub(crate) const fn with_morph(self, morph: Duration) -> Self {
        Self { morph, ..self }
    }

    /// The GPU has finished frame `frame` and every one before it (frames are numbered from 1,
    /// as they start). An older frame than one heard of already changes nothing.
    pub(crate) fn gpu_finished(&mut self, frame: u64) {
        self.gpu_finished = self.gpu_finished.max(frame.min(self.started));
    }

    /// Frames started that the GPU hasn't finished: submitted, or still with the render thread.
    pub(crate) const fn gpu_in_flight(&self) -> u64 {
        self.started.saturating_sub(self.gpu_finished)
    }

    /// Whether the GPU has as many frames unfinished as it may: no frame starts until it
    /// finishes one.
    pub(crate) const fn gpu_busy(&self) -> bool {
        self.gpu_in_flight() >= self.gpu_bound
    }

    /// A frame that started at `at` changed the cut: frames are wanted until one starts at or
    /// after `at` plus the morph.
    pub(crate) fn cut_changed(&mut self, at: Instant) {
        let end = at + self.morph;
        self.morph_end = Some(self.morph_end.map_or(end, |later| later.max(end)));
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
    pub(crate) fn wanted(&self) -> Reasons {
        self.reasons()
    }

    /// Wants the next `frames` frames at least, for the image to accumulate its samples
    /// ([`Reason::Accumulating`]): asking for fewer than are still wanted changes nothing.
    pub(crate) fn accumulate(&mut self, frames: u32) {
        self.accumulating = self.accumulating.max(frames);
    }

    /// Why a frame is wanted: what was asked for, a morph under way, and accumulation.
    fn reasons(&self) -> Reasons {
        let mut reasons = self.wanted;
        if self.morph_end.is_some() {
            reasons = reasons.with(Reason::Morphing);
        }
        if self.accumulating > 0 {
            reasons = reasons.with(Reason::Accumulating);
        }
        reasons
    }

    /// When upkeep is next due: [`UPKEEP_INTERVAL`] after it last ran (it may be past).
    pub(crate) fn upkeep_due(&self) -> Instant {
        self.upkept + UPKEEP_INTERVAL
    }

    /// What to do at `now`.
    pub(crate) fn next(&self, now: Instant) -> Next {
        if self.in_flight {
            return Next::Render;
        }
        if self.reasons().is_empty() {
            return Next::Idle {
                upkeep: self.upkeep_due(),
            };
        }
        let pace = if self.hidden && !self.wanted.contains(Reason::Exposed) {
            UPKEEP_INTERVAL
        } else {
            self.interval
        };
        if let Some(last) = self.last_start
            && now < last + pace
        {
            return Next::Wait { until: last + pace };
        }
        if self.gpu_busy() {
            return Next::Gpu;
        }
        Next::Frame {
            deadline: now + self.interval * JOB_QUARTERS / 4,
        }
    }

    /// A frame started at `now`: what was wanted is taken (and returned), and the frame is in
    /// flight until [`Scheduler::shown`], and on the GPU until [`Scheduler::gpu_finished`]. A morph
    /// ends with the first frame at or past its end; each frame takes one of those accumulation
    /// wants.
    pub(crate) fn started(&mut self, now: Instant) -> Reasons {
        let reasons = self.reasons();
        self.in_flight = true;
        self.started += 1;
        self.last_start = Some(now);
        self.upkept = now;
        self.wanted = Reasons::default();
        if self.morph_end.is_some_and(|end| end <= now) {
            self.morph_end = None;
        }
        self.accumulating = self.accumulating.saturating_sub(1);
        reasons
    }

    /// The render thread finished the frame in flight, with `outcome`.
    pub(crate) const fn shown(&mut self, outcome: Outcome) {
        self.in_flight = false;
        match outcome {
            Outcome::Presented => self.hidden = false,
            Outcome::Retry => self.want(Reason::Exposed),
            Outcome::Hidden => self.hidden = true,
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

    impl Scheduler {
        /// The render thread finished the frame in flight with `outcome`, and the GPU finished
        /// it at once: a GPU that keeps up.
        fn shown_on_gpu(&mut self, outcome: Outcome) {
            self.shown(outcome);
            self.gpu_finished(self.started);
        }
    }

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
        scheduler.shown_on_gpu(Outcome::Presented);
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
        scheduler.shown_on_gpu(Outcome::Presented);
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
            scheduler.shown_on_gpu(Outcome::Presented);
            now += INTERVAL;
        }
        assert!(
            matches!(scheduler.next(now), Next::Idle { .. }),
            "converged: no reason left, no frame"
        );
    }

    #[test]
    fn a_frame_not_shown_is_wanted_again() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Exposed);
        scheduler.started(start);
        scheduler.shown_on_gpu(Outcome::Retry);
        assert!(scheduler.wanted().contains(Reason::Exposed), "retried");
        assert!(matches!(
            scheduler.next(start + INTERVAL),
            Next::Frame { .. }
        ));
    }

    #[test]
    fn a_hidden_window_tries_converging_frames_only_now_and_then() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Converging);
        scheduler.started(start);
        scheduler.shown_on_gpu(Outcome::Hidden);
        scheduler.want(Reason::Converging);
        assert_eq!(
            scheduler.next(start + INTERVAL),
            Next::Wait {
                until: start + UPKEEP_INTERVAL
            },
            "not every refresh"
        );
        assert!(matches!(
            scheduler.next(start + UPKEEP_INTERVAL),
            Next::Frame { .. }
        ));
        scheduler.want(Reason::Exposed);
        assert!(
            matches!(scheduler.next(start + INTERVAL), Next::Frame { .. }),
            "exposed: drawn at once"
        );
        let shown = start + UPKEEP_INTERVAL;
        scheduler.started(shown);
        scheduler.shown_on_gpu(Outcome::Presented);
        scheduler.want(Reason::Converging);
        assert_eq!(
            scheduler.next(shown),
            Next::Wait {
                until: shown + INTERVAL
            },
            "shown again: paced to the display"
        );
    }

    #[test]
    fn upkeep_is_due_an_interval_after_the_last() {
        let (mut scheduler, start) = scheduler();
        assert_eq!(scheduler.upkeep_due(), start + UPKEEP_INTERVAL);
        scheduler.want(Reason::Exposed);
        assert_eq!(
            scheduler.upkeep_due(),
            start + UPKEEP_INTERVAL,
            "wanting a frame doesn't move it"
        );
        scheduler.upkept(start + UPKEEP_INTERVAL * 3);
        assert_eq!(scheduler.upkeep_due(), start + UPKEEP_INTERVAL * 4);
    }

    #[test]
    fn a_frame_runs_upkeep() {
        let (mut scheduler, start) = scheduler();
        let later = start + Duration::from_secs(3);
        scheduler.want(Reason::Scripted);
        scheduler.started(later);
        scheduler.shown_on_gpu(Outcome::Presented);
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
        scheduler.shown_on_gpu(Outcome::Presented);
        scheduler.want(Reason::Input);
        assert_eq!(
            scheduler.next(start + INTERVAL),
            Next::Wait {
                until: start + slower
            }
        );
    }

    /// The raster's morph, as the app's scheduler is given it.
    const MORPH: Duration = render::raster::MORPH;

    /// Makes frames at every interval from `from` while the scheduler wants them, each shown at
    /// once and given `wanted` first; returns when each started.
    fn frames_from(scheduler: &mut Scheduler, from: Instant, wanted: &[Reason]) -> Vec<Instant> {
        let mut starts = Vec::new();
        let mut now = from;
        for _ in 0..1000 {
            for &reason in wanted {
                scheduler.want(reason);
            }
            match scheduler.next(now) {
                Next::Frame { .. } => {
                    scheduler.started(now);
                    scheduler.shown_on_gpu(Outcome::Presented);
                    starts.push(now);
                    now += INTERVAL;
                }
                Next::Wait { until } => now = until,
                Next::Idle { .. } => return starts,
                Next::Render | Next::Gpu => {
                    unreachable!("every frame is shown and finished at once")
                }
            }
        }
        panic!("frames never stopped");
    }

    #[test]
    fn a_cut_change_keeps_frames_coming_for_exactly_the_morph_then_none() {
        let (scheduler, start) = scheduler();
        let mut scheduler = scheduler.with_morph(MORPH);
        scheduler.want(Reason::Input);
        scheduler.started(start);
        scheduler.shown_on_gpu(Outcome::Presented);
        // That frame's readback says its cut changed, heard as a later frame is made.
        scheduler.cut_changed(start);
        assert!(scheduler.wanted().contains(Reason::Morphing));
        let starts = frames_from(&mut scheduler, start + INTERVAL, &[]);
        let after: Vec<Duration> = starts.iter().map(|&t| t - start).collect();
        let last = *starts.last().expect("frames came");
        assert!(
            last >= start + MORPH,
            "the last frame shows the morph finished: {after:?}"
        );
        let before_last = starts.iter().rev().nth(1);
        assert!(
            before_last.is_none_or(|&before| before < start + MORPH),
            "and the frame before it didn't: {after:?}"
        );
        assert_eq!(
            starts.len(),
            9,
            "150 ms at 60 Hz: the 9 frames after the change: {after:?}"
        );
        assert!(matches!(scheduler.next(last + INTERVAL), Next::Idle { .. }));
        assert!(scheduler.wanted().is_empty(), "the morph is over");
    }

    #[test]
    fn a_later_cut_change_moves_the_morphs_end_and_an_earlier_one_does_not() {
        let (scheduler, start) = scheduler();
        let mut scheduler = scheduler.with_morph(MORPH);
        let later = start + Duration::from_millis(100);
        scheduler.cut_changed(later);
        scheduler.cut_changed(start);
        let starts = frames_from(&mut scheduler, start, &[]);
        let last = *starts.last().expect("frames came");
        assert!(last >= later + MORPH && last < later + MORPH + INTERVAL);
    }

    #[test]
    fn with_no_morph_a_cut_change_wants_one_frame() {
        let (mut scheduler, start) = scheduler();
        scheduler.cut_changed(start);
        let starts = frames_from(&mut scheduler, start, &[]);
        assert_eq!(starts, vec![start], "the frame at its end");
    }

    #[test]
    fn pending_requests_and_uploads_keep_frames_coming_past_the_morph() {
        for reason in [Reason::Requests, Reason::Uploads] {
            let (scheduler, start) = scheduler();
            let mut scheduler = scheduler.with_morph(MORPH);
            scheduler.cut_changed(start);
            // Pending for half a second: frames come all that time, past the morph.
            let mut now = start;
            let mut frames = 0;
            while now < start + Duration::from_millis(500) {
                scheduler.want(reason);
                assert!(
                    matches!(scheduler.next(now), Next::Frame { .. }),
                    "{reason:?} wants a frame at {:?}",
                    now - start
                );
                scheduler.started(now);
                scheduler.shown_on_gpu(Outcome::Presented);
                frames += 1;
                now += INTERVAL;
            }
            assert!(frames > 9, "more than the morph's frames: {frames}");
            assert!(
                matches!(scheduler.next(now), Next::Idle { .. }),
                "settled, and the morph long over: none"
            );
        }
    }

    #[test]
    fn reasons_name_themselves() {
        assert_eq!(Reasons::default().to_string(), "none");
        let both = Reasons::default()
            .with(Reason::Scripted)
            .with(Reason::Input);
        assert_eq!(both.to_string(), "input+scripted");
        assert!(!both.contains(Reason::Converging));
        let busy = Reasons::default()
            .with(Reason::Morphing)
            .with(Reason::Requests)
            .with(Reason::Uploads);
        assert_eq!(busy.to_string(), "uploads+requests+morphing");
    }

    #[test]
    fn a_set_is_within_another_holding_all_its_reasons() {
        let shading = Reasons::default().with(Reason::Look).with(Reason::Pick);
        assert_eq!(shading.to_string(), "look+pick");
        let pick = Reasons::default().with(Reason::Pick);
        assert!(pick.within(shading), "a pick alone shades again");
        assert!(Reasons::default().within(shading), "the empty set");
        assert!(
            !pick.with(Reason::Input).within(shading),
            "input draws the scene"
        );
    }

    /// The samples the renderer's accumulation takes.
    const SAMPLES: u32 = render::aa::SAMPLES;

    #[test]
    fn accumulation_keeps_frames_coming_for_its_samples_then_none() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Input);
        scheduler.started(start);
        scheduler.shown_on_gpu(Outcome::Presented);
        // The frame loop: that frame may have restarted the image.
        scheduler.accumulate(SAMPLES);
        assert!(scheduler.wanted().contains(Reason::Accumulating));
        let starts = frames_from(&mut scheduler, start + INTERVAL, &[]);
        assert_eq!(
            starts.len(),
            usize::try_from(SAMPLES).expect("small"),
            "one frame a sample"
        );
        assert!(scheduler.wanted().is_empty(), "converged: idle");
        assert_eq!(
            Reasons::default().with(Reason::Accumulating).to_string(),
            "accumulating"
        );
    }

    #[test]
    fn a_camera_move_holds_accumulation_off_until_it_stops_then_it_starts_again() {
        let (mut scheduler, start) = scheduler();
        let mut now = start;
        // The camera moves for ten frames: each moving frame asks for the next (the frame loop's
        // rule), which, still, restarts the image.
        for _ in 0..10 {
            scheduler.want(Reason::Input);
            assert!(matches!(scheduler.next(now), Next::Frame { .. }));
            let reasons = scheduler.started(now);
            assert!(reasons.contains(Reason::Input));
            scheduler.shown_on_gpu(Outcome::Presented);
            scheduler.accumulate(1);
            now += INTERVAL;
        }
        assert!(
            matches!(scheduler.next(now), Next::Frame { .. }),
            "one still frame"
        );
        let reasons = scheduler.started(now);
        assert_eq!(reasons, Reasons::default().with(Reason::Accumulating));
        scheduler.shown_on_gpu(Outcome::Presented);
        // Its report says the image restarted: the rest of its samples.
        scheduler.accumulate(SAMPLES - 1);
        let rest = frames_from(&mut scheduler, now + INTERVAL, &[]);
        let still_frames = rest.len() + 1;
        assert_eq!(
            still_frames,
            usize::try_from(SAMPLES).expect("small"),
            "SAMPLES still frames after the last move, not twice that"
        );
        // A scene change wants twice as many: a cycle to notice, a cycle to converge.
        let later = *rest.last().expect("frames") + INTERVAL * 10;
        scheduler.want(Reason::Uploads);
        scheduler.started(later);
        scheduler.shown_on_gpu(Outcome::Presented);
        scheduler.accumulate(2 * SAMPLES);
        scheduler.accumulate(SAMPLES);
        let after = frames_from(&mut scheduler, later + INTERVAL, &[]);
        assert_eq!(
            after.len(),
            usize::try_from(2 * SAMPLES).expect("small"),
            "asking for fewer changes nothing"
        );
    }

    /// The most frames the GPU may have unfinished.
    const BOUND: u64 = render::FRAMES_IN_FLIGHT as u64;

    /// A GPU that takes `per_frame` over each frame, one after another, with a readback ring.
    ///
    /// The ring is the renderer's (`render::lod`'s: one buffer more than the frames in flight). A
    /// frame takes a buffer as it is submitted, or drops its readback; the buffer is mapped once
    /// the GPU has finished the frame and a poll hears of it, and goes back to the ring once read,
    /// at the start of the next frame (residency's `begin_frame`, which polls first).
    #[derive(Debug)]
    struct ModelGpu {
        /// The GPU's time per frame.
        per_frame: Duration,
        /// Frames submitted and not finished as last polled: each with when it finishes.
        queue: std::collections::VecDeque<(u64, Instant)>,
        /// When the GPU is free of what it was given.
        free_at: Option<Instant>,
        /// The newest frame finished, as last polled.
        finished: u64,
        /// Readback buffers free.
        buffers: usize,
        /// Frames whose readback is mapped and not read.
        mapped: Vec<u64>,
        /// Frames whose readback was read, in order.
        read: Vec<u64>,
        /// Frames that found no buffer.
        dropped: Vec<u64>,
        /// The most frames it had unfinished as one was submitted, that one included.
        most_in_flight: u64,
    }

    impl ModelGpu {
        /// A GPU taking `per_frame` a frame, idle.
        fn new(per_frame: Duration) -> Self {
            Self {
                per_frame,
                queue: std::collections::VecDeque::new(),
                free_at: None,
                finished: 0,
                buffers: render::FRAMES_IN_FLIGHT + 1,
                mapped: Vec::new(),
                read: Vec::new(),
                dropped: Vec::new(),
                most_in_flight: 0,
            }
        }

        /// Hears at `now` of the frames finished by then (their readbacks mapped); returns the
        /// newest finished.
        fn poll(&mut self, now: Instant) -> u64 {
            while let Some(&(frame, done)) = self.queue.front()
                && done <= now
            {
                self.queue.pop_front();
                self.finished = frame;
                self.mapped.push(frame);
            }
            self.finished
        }

        /// The start of a frame at `now`: polls, then reads the mapped readbacks, freeing their
        /// buffers; returns the newest finished.
        fn begin_frame(&mut self, now: Instant) -> u64 {
            let finished = self.poll(now);
            self.buffers += self.mapped.len();
            self.read.append(&mut self.mapped);
            finished
        }

        /// Frame `frame` is submitted at `now`, its readback with it if a buffer is free.
        fn submit(&mut self, frame: u64, now: Instant) {
            if self.buffers == 0 {
                self.dropped.push(frame);
            } else {
                self.buffers -= 1;
            }
            let start = self.free_at.map_or(now, |free| free.max(now));
            let done = start + self.per_frame;
            self.free_at = Some(done);
            self.queue.push_back((frame, done));
            self.most_in_flight = self.most_in_flight.max(self.queue.len() as u64);
        }
    }

    /// Runs `scheduler` for `run` from `start` on `gpu`; returns when each frame started.
    ///
    /// Every frame is wanted (a scripted run), the render thread is instant, and the owner polls
    /// the GPU every [`crate::frame::GPU_POLL`] while it is busy. At the end the GPU finishes and
    /// what is left is read.
    fn run_on(
        scheduler: &mut Scheduler,
        gpu: &mut ModelGpu,
        start: Instant,
        run: Duration,
    ) -> Vec<Instant> {
        let mut starts = Vec::new();
        let mut now = start;
        while now < start + run {
            scheduler.want(Reason::Scripted);
            match scheduler.next(now) {
                Next::Frame { .. } => {
                    scheduler.gpu_finished(gpu.begin_frame(now));
                    scheduler.started(now);
                    starts.push(now);
                    gpu.submit(starts.len() as u64, now);
                    scheduler.shown(Outcome::Presented);
                }
                Next::Wait { until } => now = until,
                Next::Gpu => {
                    now += crate::frame::GPU_POLL;
                    scheduler.gpu_finished(gpu.poll(now));
                }
                Next::Render | Next::Idle { .. } => unreachable!("frames are wanted, and shown"),
            }
        }
        gpu.begin_frame(now + Duration::from_secs(10));
        starts
    }

    #[test]
    fn a_gpu_slower_than_the_display_gets_frames_at_its_own_rate_and_drops_no_readback() {
        let (mut scheduler, start) = scheduler();
        // The M4 at 4K in the zoom and settle: 36–48 ms a frame.
        let per_frame = Duration::from_millis(40);
        let mut gpu = ModelGpu::new(per_frame);
        let starts = run_on(&mut scheduler, &mut gpu, start, Duration::from_secs(2));
        let made = starts.len() as u64;
        assert!(
            gpu.most_in_flight <= BOUND,
            "never more than {BOUND} frames on the GPU: {}",
            gpu.most_in_flight
        );
        assert_eq!(gpu.dropped, Vec::<u64>::new(), "every frame found a buffer");
        let all: Vec<u64> = (1..=made).collect();
        assert_eq!(gpu.read, all, "every frame's readback was read, in order");
        let intervals: Vec<Duration> = starts.windows(2).map(|w| w[1] - w[0]).collect();
        let late = intervals.iter().skip(render::FRAMES_IN_FLIGHT);
        assert!(
            late.clone()
                .all(|&i| i.abs_diff(per_frame) <= crate::frame::GPU_POLL),
            "at the GPU's rate, each frame within a poll of the GPU finishing: {intervals:?}"
        );
        assert!((49..=51).contains(&made), "2 s at 40 ms: {made}");
    }

    #[test]
    fn without_the_bound_the_same_gpu_falls_behind_and_readbacks_are_dropped() {
        let (scheduler, start) = scheduler();
        let mut scheduler = scheduler.with_gpu_bound(u64::MAX);
        let mut gpu = ModelGpu::new(Duration::from_millis(40));
        let starts = run_on(&mut scheduler, &mut gpu, start, Duration::from_secs(2));
        assert_eq!(starts.len(), 120, "at the display's rate");
        assert!(gpu.most_in_flight > 50, "a queue: {}", gpu.most_in_flight);
        assert!(
            gpu.dropped.len() > 60,
            "most readbacks lost: {}",
            gpu.dropped.len()
        );
    }

    #[test]
    fn a_gpu_keeping_up_never_makes_a_frame_wait_for_it() {
        let (mut scheduler, start) = scheduler();
        let mut gpu = ModelGpu::new(Duration::from_millis(5));
        let mut now = start;
        let mut frames = 0;
        while frames < 60 {
            scheduler.want(Reason::Scripted);
            match scheduler.next(now) {
                Next::Frame { .. } => {
                    scheduler.gpu_finished(gpu.begin_frame(now));
                    scheduler.started(now);
                    frames += 1;
                    gpu.submit(frames, now);
                    scheduler.shown(Outcome::Presented);
                }
                Next::Wait { until } => now = until,
                other @ (Next::Render | Next::Gpu | Next::Idle { .. }) => {
                    panic!("only frames and the pace's waits: {other:?}")
                }
            }
        }
        assert_eq!(gpu.dropped, Vec::<u64>::new());
    }

    #[test]
    fn the_pace_is_waited_for_first_then_the_gpu() {
        let (mut scheduler, start) = scheduler();
        for frame in 0..BOUND {
            scheduler.want(Reason::Input);
            let at = start + INTERVAL * u32::try_from(frame).expect("small");
            assert!(matches!(scheduler.next(at), Next::Frame { .. }));
            scheduler.started(at);
            scheduler.shown(Outcome::Presented);
        }
        assert!(scheduler.gpu_busy(), "{BOUND} frames unfinished");
        scheduler.want(Reason::Input);
        let last = start + INTERVAL * u32::try_from(BOUND - 1).expect("small");
        assert_eq!(
            scheduler.next(last + Duration::from_millis(1)),
            Next::Wait {
                until: last + INTERVAL
            },
            "not due yet: the pace's precise wait"
        );
        assert_eq!(
            scheduler.next(last + INTERVAL),
            Next::Gpu,
            "due, the GPU busy"
        );
        scheduler.gpu_finished(1);
        assert!(
            matches!(scheduler.next(last + INTERVAL), Next::Frame { .. }),
            "one finished: room for the next"
        );
    }

    #[test]
    fn a_busy_gpu_with_nothing_wanted_is_idle_and_polled_for_nothing() {
        let (mut scheduler, start) = scheduler();
        for frame in 0..BOUND {
            scheduler.want(Reason::Input);
            scheduler.started(start + INTERVAL * u32::try_from(frame).expect("small"));
            scheduler.shown(Outcome::Presented);
        }
        assert!(scheduler.gpu_busy());
        let later = start + Duration::from_secs(1);
        assert!(
            matches!(scheduler.next(later), Next::Idle { .. }),
            "render on demand: no frame wanted, nothing to wait for"
        );
        scheduler.want(Reason::Converging);
        assert_eq!(scheduler.next(later), Next::Gpu, "wanted now");
    }

    #[test]
    fn the_render_thread_holding_a_frame_comes_before_the_gpu() {
        let (mut scheduler, start) = scheduler();
        scheduler.want(Reason::Input);
        scheduler.started(start);
        scheduler.want(Reason::Input);
        assert_eq!(
            scheduler.gpu_in_flight(),
            1,
            "the render thread's frame counts"
        );
        assert_eq!(scheduler.next(start + INTERVAL), Next::Render);
    }

    #[test]
    fn finished_frames_only_move_forward_and_never_past_those_started() {
        let (mut scheduler, start) = scheduler();
        for frame in 0..3 {
            scheduler.want(Reason::Input);
            scheduler.started(start + INTERVAL * frame);
            scheduler.shown(Outcome::Presented);
            scheduler.gpu_finished(u64::from(frame) + 1);
        }
        scheduler.gpu_finished(1);
        assert_eq!(
            scheduler.gpu_in_flight(),
            0,
            "an older frame changes nothing"
        );
        scheduler.gpu_finished(99);
        assert_eq!(scheduler.gpu_in_flight(), 0, "not past the frames started");
        scheduler.want(Reason::Input);
        scheduler.started(start + INTERVAL * 3);
        assert_eq!(scheduler.gpu_in_flight(), 1, "the next frame counts");
    }

    #[test]
    fn accumulation_and_a_morph_finish_at_the_gpus_rate() {
        let (scheduler, start) = scheduler();
        let mut scheduler = scheduler.with_morph(MORPH);
        let mut gpu = ModelGpu::new(Duration::from_millis(40));
        scheduler.want(Reason::Input);
        scheduler.started(start);
        gpu.submit(1, start);
        scheduler.shown(Outcome::Presented);
        scheduler.accumulate(SAMPLES);
        scheduler.cut_changed(start);
        let mut now = start + INTERVAL;
        let mut starts = Vec::new();
        loop {
            match scheduler.next(now) {
                Next::Frame { .. } => {
                    scheduler.gpu_finished(gpu.begin_frame(now));
                    scheduler.started(now);
                    starts.push(now);
                    gpu.submit(starts.len() as u64 + 1, now);
                    scheduler.shown(Outcome::Presented);
                }
                Next::Wait { until } => now = until,
                Next::Gpu => {
                    now += crate::frame::GPU_POLL;
                    scheduler.gpu_finished(gpu.poll(now));
                }
                Next::Idle { .. } => break,
                Next::Render => unreachable!("shown at once"),
            }
        }
        let last = *starts.last().expect("frames came");
        assert!(last >= start + MORPH, "the morph is shown finished");
        assert!(
            starts.len() >= usize::try_from(SAMPLES).expect("small"),
            "every sample drawn: {}",
            starts.len()
        );
        assert_eq!(gpu.dropped, Vec::<u64>::new());
    }
}
