//! `--compare`: a winit run (W) beside one of ours (U), and the decision's three checks for W.
//!
//! The checks, from the M1a4 brief: W is chosen if
//!
//! 1. its de-jittered speed noise is within about 0.1–0.3 points of the floor (the steady period's,
//!    in the same run) on the steady-strokes step, or on every stroke without one;
//! 2. its p99 dispatch delay under load is at most [`DISPATCH_MARGIN_MS`] over U's, both handled
//!    behind the pen's clock (see [`super::dispatch`]). The brief's flat [`BUDGET_MS`] is printed
//!    beside it for information only: the wait comes from the frame loop and delays U's samples as
//!    much, so W is held to U's figure (the plan's corrected rule, 2026-10-28);
//! 3. its events tell each barrel button apart from the tip: a click while hovering makes no
//!    pen-down, and a release mid-stroke ends no stroke while the pen still presses.
//!
//! Otherwise U. The numbers are printed beside each check, so a borderline call is the reader's.

use std::fmt::{self, Write as _};

use super::Noise;
use super::buttons::{self, StepFindings};
use super::dispatch::{self, BUDGET_MS, Moment, Spread};
use crate::script::Step;
use crate::{Recording, Sample, Source};

/// The most W's de-jittered speed noise may sit above the floor, in points.
pub(crate) const NOISE_MARGIN: f64 = 0.3;

/// The most W's p99 dispatch delay may exceed U's, in milliseconds.
///
/// The M1a3 plan's corrected rule (2026-10-28): both runs wait on the same frame loop, which
/// delays U's samples reaching the brush as much as W's, so W is judged by what it adds on top of
/// U. The millisecond is the rule's allowance for two runs of one load not being identical.
pub(crate) const DISPATCH_MARGIN_MS: f64 = 1.0;

/// The samples of `source` in `recording`.
fn of(recording: &Recording, source: Source) -> Vec<&Sample> {
    recording
        .samples
        .iter()
        .filter(|s| s.source == source)
        .collect()
}

/// Speed noise on the steady-strokes step, or every stroke if there is no such step.
fn steady_noise(samples: &[&Sample]) -> Noise {
    Noise::of_step(samples, Step::Steady).unwrap_or_else(|| Noise::of(samples))
}

/// `Option<Spread>` for printing.
fn show(spread: Option<Spread>) -> String {
    spread.map_or_else(|| "n/a".to_owned(), |s| s.to_string())
}

/// The main thread's busy stretches in a recording's load, for printing.
fn away(recording: &Recording) -> String {
    recording.frames.as_ref().map_or_else(
        || "no load".to_owned(),
        |frames| show(Spread::of(frames.iter().map(|f| f.busy))),
    )
}

/// A check's outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// It holds.
    Pass,
    /// It doesn't.
    Fail,
    /// The recording can't say.
    Unknown,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pass => "yes",
            Self::Fail => "NO",
            Self::Unknown => "not measured",
        })
    }
}

/// Whether W tells one barrel apart, from its hover and mid-stroke steps.
fn told_apart(hover: StepFindings, stroke: StepFindings) -> Verdict {
    if hover.presses == 0 && stroke.presses == 0 {
        return Verdict::Unknown;
    }
    if hover.downs > 0 || stroke.ends_pressing > 0 {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

/// The three checks' verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Checks {
    /// Speed noise near the floor.
    pub(crate) noise: Verdict,
    /// Dispatch at most [`DISPATCH_MARGIN_MS`] behind U's.
    pub(crate) dispatch: Verdict,
    /// Barrel 1 told apart.
    pub(crate) barrel1: Verdict,
    /// Barrel 2 told apart.
    pub(crate) barrel2: Verdict,
}

impl Checks {
    /// The barrels together: a failure fails, else a pass passes, else unknown.
    pub(crate) fn barrels(self) -> Verdict {
        match (self.barrel1, self.barrel2) {
            (Verdict::Fail, _) | (_, Verdict::Fail) => Verdict::Fail,
            (Verdict::Pass, _) | (_, Verdict::Pass) => Verdict::Pass,
            (Verdict::Unknown, Verdict::Unknown) => Verdict::Unknown,
        }
    }

    /// W only if every check passes.
    pub(crate) fn choice(self) -> &'static str {
        if [self.noise, self.dispatch, self.barrels()]
            .iter()
            .all(|v| *v == Verdict::Pass)
        {
            "W (winit's tablet events, de-jittered)"
        } else {
            "U (our Wayland backend)"
        }
    }
}

/// The dispatch delay the decision reads, for either source: handled behind the pen's clock.
fn dispatch_delay(samples: &[&Sample]) -> Option<Spread> {
    dispatch::behind_clock(samples, Moment::Handled)
}

/// Whether W's dispatch delay `w` is within [`DISPATCH_MARGIN_MS`] of U's `u` at the p99.
fn dispatch_verdict(w: Option<Spread>, u: Option<Spread>) -> Verdict {
    match (w, u) {
        (Some(w), Some(u)) if w.p99 - u.p99 <= DISPATCH_MARGIN_MS => Verdict::Pass,
        (Some(_), Some(_)) => Verdict::Fail,
        _ => Verdict::Unknown,
    }
}

/// The checks for W's samples `winit`, against U's samples `ours` where a check compares them.
pub(crate) fn checks(winit: &[&Sample], ours: &[&Sample]) -> Checks {
    let noise = match steady_noise(winit).above_floor() {
        Some(d) if d <= NOISE_MARGIN => Verdict::Pass,
        Some(_) => Verdict::Fail,
        None => Verdict::Unknown,
    };
    let dispatch = dispatch_verdict(dispatch_delay(winit), dispatch_delay(ours));
    let barrel = |hover, stroke| {
        told_apart(
            buttons::findings(Source::Winit, winit, hover),
            buttons::findings(Source::Winit, winit, stroke),
        )
    };
    Checks {
        noise,
        dispatch,
        barrel1: barrel(Step::Barrel1Hover, Step::Barrel1Stroke),
        barrel2: barrel(Step::Barrel2Hover, Step::Barrel2Stroke),
    }
}

/// The comparison of `winit` (W's run) and `ours` (U's), with the checks and the choice.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(crate) fn compare(winit: &Recording, ours: &Recording) -> Result<String, fmt::Error> {
    let (w, u) = (of(winit, Source::Winit), of(ours, Source::Ours));
    let mut out = String::new();
    writeln!(out, "== W (winit) against U (ours) ==")?;
    writeln!(
        out,
        "speed noise (the steady-strokes step; every stroke without one):"
    )?;
    let (w_noise, u_noise) = (steady_noise(&w), steady_noise(&u));
    writeln!(out, "  W: {w_noise}")?;
    writeln!(out, "  U: {u_noise}")?;
    let (w_delay, u_delay) = (dispatch_delay(&w), dispatch_delay(&u));
    writeln!(out, "dispatch delay:")?;
    writeln!(out, "  W: handled behind the pen's clock {}", show(w_delay))?;
    writeln!(out, "  U: handled behind the pen's clock {}", show(u_delay))?;
    writeln!(
        out,
        "     of which receipt to main-thread handling {}; receipt behind the pen's clock {} \
         (compositor, socket and input-thread scheduling)",
        show(dispatch::handling(&u)),
        show(dispatch::behind_clock(&u, Moment::Received)),
    )?;
    writeln!(
        out,
        "  main thread away per frame: W run {}; U run {}",
        away(winit),
        away(ours)
    )?;
    writeln!(out, "barrel buttons:")?;
    for step in [
        Step::Barrel1Hover,
        Step::Barrel1Stroke,
        Step::Barrel2Hover,
        Step::Barrel2Stroke,
    ] {
        let (fw, fu) = (
            buttons::findings(Source::Winit, &w, step),
            buttons::findings(Source::Ours, &u, step),
        );
        writeln!(
            out,
            "  {}: W {} button events, {} pen-downs, {} strokes ended while pressing; U {} \
             button events, {} pen-downs, {} ends",
            step.name(),
            fw.presses,
            fw.downs,
            fw.ends_pressing,
            fu.presses,
            fu.downs,
            fu.ups,
        )?;
    }
    let checks = checks(&w, &u);
    writeln!(out, "checks for W:")?;
    writeln!(
        out,
        "  1. de-jittered speed noise within ~{NOISE_MARGIN} points of the floor: {} ({})",
        checks.noise,
        w_noise
            .above_floor()
            .map_or_else(|| "n/a".to_owned(), |d| format!("{d:+.1} points")),
    )?;
    let p99 = |spread: Option<Spread>| {
        spread.map_or_else(|| "n/a".to_owned(), |s| format!("{:.2} ms", s.p99))
    };
    let unloaded: Vec<&str> = [("W", winit), ("U", ours)]
        .into_iter()
        .filter(|(_, run)| run.frames.is_none())
        .map(|(name, _)| name)
        .collect();
    writeln!(
        out,
        "  2. p99 dispatch delay under load at most {DISPATCH_MARGIN_MS:.0} ms over U's: {} (W {}, \
         U {}{})",
        checks.dispatch,
        p99(w_delay),
        p99(u_delay),
        match unloaded.as_slice() {
            [] => String::new(),
            [one] => format!("; the {one} run had no --load"),
            _ => "; neither run had --load".to_owned(),
        },
    )?;
    writeln!(
        out,
        "     for information, the brief's flat rule: W's p99 within {BUDGET_MS:.0} ms: {}",
        match w_delay {
            Some(s) if s.p99 <= BUDGET_MS => "yes",
            Some(_) => "no",
            None => "not measured",
        },
    )?;
    writeln!(
        out,
        "  3. barrel buttons told apart from the tip: {} (barrel 1 {}, barrel 2 {})",
        checks.barrels(),
        checks.barrel1,
        checks.barrel2,
    )?;
    writeln!(out, "choice: {}", checks.choice())?;
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use std::time::{Duration, Instant};

    use input::Buttons;

    use super::super::{parse_csv, to_csv};
    use super::*;
    use crate::Selection;
    use crate::winit_tool::{WinitButton, WinitEvent, WinitTool};

    /// A synthetic winit run, through [`WinitTool`] and the CSV and back.
    ///
    /// Three steady strokes at an even speed with delivery jitter, a barrel-1 click while
    /// hovering, and barrel 1 held mid-stroke then released.
    fn winit_run() -> Recording {
        let t0 = Instant::now();
        let period = Duration::from_micros(5_800);
        let delays_us = [0_u64, 3_100, 900, 4_000, 1_700, 2_500, 400, 3_600];
        let mut tool = WinitTool::default();
        let mut samples = Vec::new();
        let mut slot = 0_u32;
        let mut push = |event: WinitEvent, x: f64, force: f64, step: Step| {
            let delay = delays_us
                .get(usize::try_from(slot).unwrap() % delays_us.len())
                .copied()
                .unwrap();
            let at = t0 + period * slot + Duration::from_micros(delay);
            let stamped = tool.apply(event, (x, 0.0), at, u64::from(slot));
            let since = |t: Instant| t.duration_since(t0);
            samples.push(Sample {
                source: Source::Winit,
                received: since(at),
                handled: Some(since(at)),
                dejittered: stamped.timestamp.map(since),
                wakeup: u64::from(slot),
                x,
                y: 0.0,
                contact: stamped.phase.in_stroke(),
                pressure: Some(force),
                tilt: None,
                distance: None,
                roll: None,
                phase: Some(stamped.phase),
                buttons: stamped.buttons,
                device_ms: None,
                winit_event: Some(event),
                proximity: Some(stamped.proximity),
                step: Some(step),
            });
            slot += 1;
        };
        let contact = |pressed| WinitEvent::Button {
            button: WinitButton::Contact,
            pressed,
        };
        for _ in 0..3 {
            for i in 0..150_u32 {
                let event = match i {
                    0 => contact(true),
                    149 => contact(false),
                    _ => WinitEvent::Moved,
                };
                push(event, f64::from(i) * 4.0, 0.5, Step::Steady);
            }
            // A lift: a break of a stroke gap or more.
            for _ in 0..20 {
                push(WinitEvent::Moved, 0.0, 0.0, Step::Steady);
            }
        }
        let hover = Step::Barrel1Hover;
        push(WinitEvent::Moved, 1.0, 0.0, hover);
        push(contact(true), 1.0, 0.0, hover);
        push(WinitEvent::Moved, 2.0, 0.0, hover);
        push(contact(false), 2.0, 0.0, hover);
        push(WinitEvent::Moved, 3.0, 0.0, hover);
        let mid = Step::Barrel1Stroke;
        push(contact(true), 10.0, 0.0, mid);
        push(WinitEvent::Moved, 14.0, 0.4, mid);
        push(contact(true), 18.0, 0.4, mid);
        push(WinitEvent::Moved, 22.0, 0.5, mid);
        push(contact(false), 26.0, 0.5, mid);
        push(WinitEvent::Moved, 30.0, 0.45, mid);
        push(contact(false), 34.0, 0.3, mid);
        let csv = to_csv(&samples).expect("a String");
        Recording {
            selection: Selection::Winit,
            samples: parse_csv(&csv).expect("the probe's own CSV parses"),
            ..Recording::default()
        }
    }

    #[test]
    fn a_winit_run_compares_through_the_csv() {
        let winit = winit_run();
        let w = of(&winit, Source::Winit);
        let noise = steady_noise(&w);
        let (receipt, dejittered) = (noise.receipt.unwrap(), noise.dejittered.unwrap());
        assert!(
            dejittered < receipt / 3.0,
            "the filter steadies winit's clock: {noise}"
        );
        assert!(noise.floor.unwrap() < 1e-6, "an even speed: {noise}");
        assert!(
            dispatch_delay(&w).is_some_and(|d| d.max < 4.1),
            "behind the clock by at most the 4 ms jitter"
        );
        let checks = checks(&w, &[]);
        assert_eq!(checks.barrel1, Verdict::Fail, "{checks:?}");
        assert_eq!(checks.barrel2, Verdict::Unknown, "{checks:?}");
        assert_eq!(
            checks.dispatch,
            Verdict::Unknown,
            "W alone can't be held to U: {checks:?}"
        );

        let ours = Recording {
            samples: vec![Sample {
                source: Source::Ours,
                buttons: Buttons::default(),
                ..w.first().copied().cloned().unwrap()
            }],
            ..Recording::default()
        };
        let text = compare(&winit, &ours).expect("a String");
        assert!(
            text.contains("barrel1-hover: W 2 button events, 1 pen-downs"),
            "{text}"
        );
        assert!(
            text.contains("barrel1-stroke: W 4 button events, 1 pen-downs, 1 strokes ended"),
            "{text}"
        );
        assert!(
            text.contains("3. barrel buttons told apart from the tip: NO"),
            "{text}"
        );
        assert!(text.contains("neither run had --load"), "{text}");
        assert!(
            text.ends_with("choice: U (our Wayland backend)\n"),
            "{text}"
        );
    }

    /// A run of `source` whose one contact sample is handled `behind_us` behind the pen's clock,
    /// under a load of one frame.
    fn handled_behind(source: Source, behind_us: u64) -> Recording {
        let dejittered = Duration::from_millis(10);
        Recording {
            samples: vec![Sample {
                source,
                received: dejittered,
                handled: Some(dejittered + Duration::from_micros(behind_us)),
                dejittered: Some(dejittered),
                wakeup: 0,
                x: 0.0,
                y: 0.0,
                contact: true,
                pressure: Some(0.5),
                tilt: None,
                distance: None,
                roll: None,
                phase: Some(input::Phase::Drag),
                buttons: Buttons::default(),
                device_ms: None,
                winit_event: None,
                proximity: Some(true),
                step: None,
            }],
            frames: Some(vec![crate::frames::FrameRecord::default()]),
            ..Recording::default()
        }
    }

    #[test]
    fn dispatch_holds_w_to_us_handled_behind_the_clock() {
        // The user's measurement: W fails by 18 ms, though both are far over the flat 8 ms.
        let (winit, ours) = (
            handled_behind(Source::Winit, 61_800),
            handled_behind(Source::Ours, 43_700),
        );
        let (w, u) = (of(&winit, Source::Winit), of(&ours, Source::Ours));
        assert_eq!(checks(&w, &u).dispatch, Verdict::Fail);
        let text = compare(&winit, &ours).expect("a String");
        assert!(
            text.contains("  W: handled behind the pen's clock p50 61.80 / p99 61.80"),
            "{text}"
        );
        assert!(
            text.contains("  U: handled behind the pen's clock p50 43.70 / p99 43.70"),
            "{text}"
        );
        assert!(
            text.contains(
                "2. p99 dispatch delay under load at most 1 ms over U's: NO (W 61.80 ms, U 43.70 \
                 ms)\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("the brief's flat rule: W's p99 within 8 ms: no"),
            "{text}"
        );

        // Within a millisecond of U passes, over the flat budget or not.
        let close = handled_behind(Source::Winit, 44_500);
        let close_w = of(&close, Source::Winit);
        assert_eq!(checks(&close_w, &u).dispatch, Verdict::Pass);
        let close_text = compare(&close, &ours).expect("a String");
        assert!(
            close_text.contains("over U's: yes (W 44.50 ms, U 43.70 ms)"),
            "{close_text}"
        );
        assert_eq!(
            dispatch_verdict(dispatch_delay(&close_w), None),
            Verdict::Unknown
        );
    }

    const fn found(presses: usize, downs: usize, ends_pressing: usize) -> StepFindings {
        StepFindings {
            presses,
            downs,
            downs_without_force: downs,
            ups: downs,
            ends_pressing,
        }
    }

    #[test]
    fn a_barrel_is_told_apart_only_with_no_false_downs_or_ends() {
        assert_eq!(told_apart(found(4, 0, 0), found(4, 2, 0)), Verdict::Pass);
        assert_eq!(told_apart(found(4, 2, 0), found(4, 2, 0)), Verdict::Fail);
        assert_eq!(told_apart(found(4, 0, 0), found(4, 2, 1)), Verdict::Fail);
        assert_eq!(told_apart(found(0, 0, 0), found(0, 2, 0)), Verdict::Unknown);
    }

    #[test]
    fn w_is_chosen_only_when_every_check_passes() {
        let pass = Checks {
            noise: Verdict::Pass,
            dispatch: Verdict::Pass,
            barrel1: Verdict::Pass,
            barrel2: Verdict::Unknown,
        };
        assert_eq!(
            pass.barrels(),
            Verdict::Pass,
            "an untested barrel doesn't fail"
        );
        assert!(pass.choice().starts_with('W'));
        for broken in [
            Checks {
                noise: Verdict::Fail,
                ..pass
            },
            Checks {
                dispatch: Verdict::Unknown,
                ..pass
            },
            Checks {
                barrel2: Verdict::Fail,
                ..pass
            },
        ] {
            assert!(broken.choice().starts_with('U'), "{broken:?}");
        }
    }
}
