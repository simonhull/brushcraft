//! What the barrel buttons send: every button event with its force, proximity and contact, and,
//! for a guided recording, what each barrel step shows.
//!
//! A **button event** is, for winit, a `PointerButton` row: winit's button and state. For ours
//! it is a sample whose held buttons differ from the sample before: the decoded bits, before and
//! after (1 is `BTN_STYLUS`, 2 `BTN_STYLUS2`, 4 `BTN_STYLUS3`).
//!
//! The guided script says which button the user pressed, and how:
//!
//! - **(a) a barrel click while hovering** (`barrel1-hover`, `barrel2-hover`): what arrives, with
//!   what force, and how many pen-downs the step shows. winit maps `BTN_STYLUS` to its contact
//!   button, so a winit pen-down in this step is a barrel click read as a touch.
//! - **(b) a barrel held mid-stroke, then released** (`barrel1-stroke`, `barrel2-stroke`): how many
//!   strokes and stroke ends the step shows, and how many of those ends came with the pen still
//!   pressing: a row with force above [`PRESSING`] within [`AFTER`] rows of the `Up` and before the
//!   next `Down`. Those are the barrel's release read as a lift. Ours writes no pressure out of
//!   contact, so for ours this count is zero by construction; its ends are real lifts.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use input::Phase;

use super::button_mask;
use crate::script::Step;
use crate::winit_tool::WinitEvent;
use crate::{Sample, Source};

/// Force above which the pen counts as still pressing.
pub(crate) const PRESSING: f64 = 0.02;
/// Rows after an `Up` (or a pen-down) looked at for force: about 30 ms at the pen's rate.
pub(crate) const AFTER: usize = 5;

/// One button event.
#[derive(Debug, Clone, PartialEq)]
struct ButtonEvent {
    /// What arrived: winit's button and state, or our bits before and after.
    raw: String,
    /// The force (pressure) the event carried.
    force: Option<f64>,
    /// Whether the tool was in proximity.
    proximity: Option<bool>,
    /// Whether the tip was down just before it, as the source saw it.
    touching: bool,
    /// The guided step it came in.
    step: Option<Step>,
}

/// Whether the source saw the tip down at `sample`: in a stroke and not lifting.
fn touching(sample: &Sample) -> bool {
    sample
        .phase
        .map_or(sample.contact, |p| matches!(p, Phase::Down | Phase::Drag))
}

/// Every button event in one source's `samples`.
fn events(source: Source, samples: &[&Sample]) -> Vec<ButtonEvent> {
    let before = std::iter::once(None).chain(samples.iter().copied().map(Some));
    samples
        .iter()
        .copied()
        .zip(before)
        .filter_map(|(s, prev)| {
            let raw = match source {
                Source::Winit => match s.winit_event? {
                    WinitEvent::Button { button, pressed } => format!(
                        "{} {}",
                        button.name(),
                        if pressed { "pressed" } else { "released" }
                    ),
                    WinitEvent::Entered | WinitEvent::Left | WinitEvent::Moved => return None,
                },
                Source::Ours => {
                    let (was, now) = (
                        prev.map_or(0, |p| button_mask(p.buttons)),
                        button_mask(s.buttons),
                    );
                    if was == now {
                        return None;
                    }
                    format!("bits {was}->{now}")
                }
            };
            Some(ButtonEvent {
                raw,
                force: s.pressure,
                proximity: s.proximity,
                touching: prev.is_some_and(touching),
                step: s.step,
            })
        })
        .collect()
}

/// A force range, for printing.
#[derive(Debug, Clone, Copy, Default)]
struct Range(Option<(f64, f64)>);

impl Range {
    /// The range with `v` added.
    fn with(self, v: Option<f64>) -> Self {
        match (self.0, v) {
            (Some((lo, hi)), Some(v)) => Self(Some((lo.min(v), hi.max(v)))),
            (None, Some(v)) => Self(Some((v, v))),
            (range, None) => Self(range),
        }
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some((lo, hi)) if (hi - lo).abs() < 5e-4 => write!(f, "force {lo:.3}"),
            Some((lo, hi)) => write!(f, "force {lo:.3}-{hi:.3}"),
            None => f.write_str("no force"),
        }
    }
}

/// `events` grouped by what arrived and whether the tip was down: `raw, hovering|touching: n at
/// force …`, joined with `; `.
fn grouped<'a>(events: impl IntoIterator<Item = &'a ButtonEvent>) -> String {
    let mut groups = BTreeMap::<(String, &'static str), (usize, Range)>::new();
    for e in events {
        let place = match (e.touching, e.proximity) {
            (true, _) => "touching",
            (false, Some(false)) => "out of range",
            (false, _) => "hovering",
        };
        let entry = groups.entry((e.raw.clone(), place)).or_default();
        entry.0 += 1;
        entry.1 = entry.1.with(e.force);
    }
    if groups.is_empty() {
        return "none".to_owned();
    }
    let parts: Vec<String> = groups
        .iter()
        .map(|((raw, place), (n, range))| format!("{raw}, {place}: {n} at {range}"))
        .collect();
    parts.join("; ")
}

/// The most force among the [`AFTER`] rows after `i` before the next `Down`; `None` if none
/// carries force.
fn force_after(samples: &[&Sample], i: usize) -> Option<f64> {
    samples
        .iter()
        .skip(i + 1)
        .take(AFTER)
        .take_while(|s| s.phase != Some(Phase::Down))
        .filter_map(|s| s.pressure)
        .reduce(f64::max)
}

/// What a guided barrel step shows for one source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StepFindings {
    /// Button events in the step.
    pub(crate) presses: usize,
    /// Pen-downs in the step.
    pub(crate) downs: usize,
    /// Pen-downs in the step after which the pen never presses (force stays at or below
    /// [`PRESSING`] for [`AFTER`] rows).
    pub(crate) downs_without_force: usize,
    /// Stroke ends in the step.
    pub(crate) ups: usize,
    /// Stroke ends with the pen still pressing after them.
    pub(crate) ends_pressing: usize,
}

/// What `step` shows in one source's `samples`.
pub(crate) fn findings(source: Source, samples: &[&Sample], step: Step) -> StepFindings {
    let presses = events(source, samples)
        .iter()
        .filter(|e| e.step == Some(step))
        .count();
    let mut found = StepFindings {
        presses,
        ..StepFindings::default()
    };
    for (i, s) in samples.iter().enumerate() {
        if s.step != Some(step) {
            continue;
        }
        let after = force_after(samples, i);
        match s.phase {
            Some(Phase::Down) => {
                found.downs += 1;
                let most = after.into_iter().chain(s.pressure).fold(0.0, f64::max);
                if most <= PRESSING {
                    found.downs_without_force += 1;
                }
            }
            Some(Phase::Up) => {
                found.ups += 1;
                if after.is_some_and(|f| f > PRESSING) {
                    found.ends_pressing += 1;
                }
            }
            Some(Phase::Enter | Phase::Hover | Phase::Drag | Phase::Leave) | None => {}
        }
    }
    found
}

/// The force at each pen-down of `step` and the most in the [`AFTER`] rows after it: how tip
/// touches start, for setting barrel clicks beside them.
fn down_forces(samples: &[&Sample], step: Step) -> (Range, Range) {
    samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.step == Some(step) && s.phase == Some(Phase::Down))
        .fold(
            (Range::default(), Range::default()),
            |(at, after), (i, s)| (at.with(s.pressure), after.with(force_after(samples, i))),
        )
}

/// Appends the button lines for one source's samples.
///
/// # Errors
///
/// Only if formatting fails, which writing to a `String` never does.
pub(super) fn block(out: &mut String, source: Source, samples: &[&Sample]) -> fmt::Result {
    let all = events(source, samples);
    // Older CSVs record no winit button events; ours' bits are in every CSV.
    if source == Source::Winit && samples.iter().all(|s| s.winit_event.is_none()) {
        return Ok(());
    }
    writeln!(out, "button events: {}: {}", all.len(), grouped(&all))?;
    if samples.iter().all(|s| s.step.is_none()) {
        return Ok(());
    }
    let (at, after) = down_forces(samples, Step::Steady);
    writeln!(
        out,
        "  tip touches (steady step): pen-downs at {at}; next {AFTER} rows at {after}"
    )?;
    for step in [Step::Barrel1Hover, Step::Barrel2Hover] {
        let f = findings(source, samples, step);
        let in_step = all.iter().filter(|e| e.step == Some(step));
        writeln!(
            out,
            "  (a) {}: {}; pen-downs {} ({} never pressing)",
            step.name(),
            grouped(in_step),
            f.downs,
            f.downs_without_force,
        )?;
    }
    for step in [Step::Barrel1Stroke, Step::Barrel2Stroke] {
        let f = findings(source, samples, step);
        let in_step = all.iter().filter(|e| e.step == Some(step));
        writeln!(
            out,
            "  (b) {}: {}; strokes {}, ends {}, ends with the pen still pressing {}",
            step.name(),
            grouped(in_step),
            f.downs,
            f.ups,
            f.ends_pressing,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use std::time::Duration;

    use input::Buttons;

    use super::*;
    use crate::winit_tool::WinitButton;

    /// A row of `source` in `step` with `phase`, `force` and `event`.
    fn row(
        source: Source,
        step: Step,
        phase: Phase,
        force: f64,
        event: Option<WinitEvent>,
        buttons: Buttons,
    ) -> Sample {
        Sample {
            source,
            received: Duration::ZERO,
            handled: Some(Duration::ZERO),
            dejittered: None,
            wakeup: 0,
            x: 0.0,
            y: 0.0,
            contact: phase.in_stroke(),
            pressure: Some(force),
            tilt: None,
            distance: None,
            roll: None,
            phase: Some(phase),
            buttons,
            device_ms: None,
            winit_event: event,
            proximity: Some(true),
            step: Some(step),
        }
    }

    const fn contact(pressed: bool) -> WinitEvent {
        WinitEvent::Button {
            button: WinitButton::Contact,
            pressed,
        }
    }

    const MOVED: Option<WinitEvent> = Some(WinitEvent::Moved);

    /// winit's view of a barrel-1 click while hovering: a pen-down that never presses.
    fn winit_hover_click() -> Vec<Sample> {
        let step = Step::Barrel1Hover;
        let none = Buttons::default();
        vec![
            row(Source::Winit, step, Phase::Hover, 0.0, MOVED, none),
            row(
                Source::Winit,
                step,
                Phase::Down,
                0.0,
                Some(contact(true)),
                none,
            ),
            row(Source::Winit, step, Phase::Drag, 0.0, MOVED, none),
            row(Source::Winit, step, Phase::Drag, 0.0, MOVED, none),
            row(
                Source::Winit,
                step,
                Phase::Up,
                0.0,
                Some(contact(false)),
                none,
            ),
            row(Source::Winit, step, Phase::Hover, 0.0, MOVED, none),
        ]
    }

    /// winit's view of barrel 1 held mid-stroke: the release ends the stroke while the pen
    /// still presses, then the real lift is a release out of contact.
    fn winit_mid_stroke() -> Vec<Sample> {
        let step = Step::Barrel1Stroke;
        let none = Buttons::default();
        vec![
            row(
                Source::Winit,
                step,
                Phase::Down,
                0.0,
                Some(contact(true)),
                none,
            ),
            row(Source::Winit, step, Phase::Drag, 0.4, MOVED, none),
            row(
                Source::Winit,
                step,
                Phase::Drag,
                0.4,
                Some(contact(true)),
                none,
            ),
            row(Source::Winit, step, Phase::Drag, 0.5, MOVED, none),
            row(
                Source::Winit,
                step,
                Phase::Up,
                0.5,
                Some(contact(false)),
                none,
            ),
            row(Source::Winit, step, Phase::Hover, 0.45, MOVED, none),
            row(
                Source::Winit,
                step,
                Phase::Hover,
                0.3,
                Some(contact(false)),
                none,
            ),
            row(Source::Winit, step, Phase::Hover, 0.3, MOVED, none),
        ]
    }

    /// Ours, barrel 1 held mid-stroke: the bits change, one stroke, one end at the lift.
    fn ours_mid_stroke() -> Vec<Sample> {
        let step = Step::Barrel1Stroke;
        let (none, one) = (Buttons::default(), Buttons::PRIMARY);
        vec![
            row(Source::Ours, step, Phase::Down, 0.1, None, none),
            row(Source::Ours, step, Phase::Drag, 0.4, None, one),
            row(Source::Ours, step, Phase::Drag, 0.5, None, one),
            row(Source::Ours, step, Phase::Drag, 0.5, None, none),
            row(Source::Ours, step, Phase::Up, 0.2, None, none),
            row(Source::Ours, step, Phase::Hover, 0.0, None, none),
        ]
    }

    fn refs(samples: &[Sample]) -> Vec<&Sample> {
        samples.iter().collect()
    }

    #[test]
    fn a_winit_hover_click_on_barrel_one_reads_as_a_pen_down() {
        let samples = winit_hover_click();
        let f = findings(Source::Winit, &refs(&samples), Step::Barrel1Hover);
        assert_eq!(
            f,
            StepFindings {
                presses: 2,
                downs: 1,
                downs_without_force: 1,
                ups: 1,
                ends_pressing: 0,
            }
        );
    }

    #[test]
    fn a_winit_barrel_release_mid_stroke_is_an_end_while_pressing() {
        let samples = winit_mid_stroke();
        let f = findings(Source::Winit, &refs(&samples), Step::Barrel1Stroke);
        assert_eq!((f.presses, f.downs, f.ups, f.ends_pressing), (4, 1, 1, 1));
        let mut out = String::new();
        block(&mut out, Source::Winit, &refs(&samples)).expect("a String");
        assert!(
            out.contains("Contact pressed, touching: 1 at force 0.400"),
            "the barrel's press while drawing: {out}"
        );
        assert!(
            out.contains("Contact released, hovering: 1 at force 0.300"),
            "the real lift reads as a release out of contact: {out}"
        );
        assert!(
            out.contains("strokes 1, ends 1, ends with the pen still pressing 1"),
            "{out}"
        );
    }

    #[test]
    fn ours_reports_the_bits_and_ends_only_at_the_lift() {
        let samples = ours_mid_stroke();
        let samples = refs(&samples);
        let f = findings(Source::Ours, &samples, Step::Barrel1Stroke);
        assert_eq!((f.presses, f.downs, f.ups, f.ends_pressing), (2, 1, 1, 0));
        let mut out = String::new();
        block(&mut out, Source::Ours, &samples).expect("a String");
        assert!(
            out.contains("bits 0->1, touching: 1 at force 0.400"),
            "{out}"
        );
        assert!(
            out.contains("bits 1->0, touching: 1 at force 0.500"),
            "{out}"
        );
    }

    #[test]
    fn a_tip_touch_presses_soon_after() {
        let step = Step::Steady;
        let none = Buttons::default();
        let samples = [
            row(
                Source::Winit,
                step,
                Phase::Down,
                0.0,
                Some(contact(true)),
                none,
            ),
            row(Source::Winit, step, Phase::Drag, 0.0, MOVED, none),
            row(Source::Winit, step, Phase::Drag, 0.2, MOVED, none),
        ];
        let f = findings(Source::Winit, &refs(&samples), step);
        assert_eq!((f.downs, f.downs_without_force), (1, 0));
        let (at, after) = down_forces(&refs(&samples), step);
        assert_eq!(at.to_string(), "force 0.000");
        assert_eq!(after.to_string(), "force 0.200");
    }

    #[test]
    fn untagged_recordings_list_the_events_only() {
        let mut samples = winit_hover_click();
        for s in &mut samples {
            s.step = None;
        }
        let mut out = String::new();
        block(&mut out, Source::Winit, &refs(&samples)).expect("a String");
        assert_eq!(
            out,
            "button events: 2: Contact pressed, hovering: 1 at force 0.000; Contact released, \
             touching: 1 at force 0.000\n"
        );
    }
}
