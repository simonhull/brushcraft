//! winit's tablet events as a stroke engine built on them would take them.
//!
//! Phases come from winit's contact button, and contact phases are stamped through
//! [`input::Dejitter`], as `input`'s Wayland backend stamps its own
//! (`crates/input/src/wayland/tool.rs`).
//!
//! winit 0.31.0-beta.3 hands a tablet tool over as `PointerEntered`, `PointerMoved`,
//! `PointerButton` and `PointerLeft`, with no sample time: the probe stamps each with
//! [`Instant::now`] on arrival, on the event-loop thread. Its Wayland backend maps `BTN_STYLUS`
//! (the first barrel button) to [`WinitButton::Contact`], the same as the tip touching, so a
//! stroke engine that follows winit's contact sees a barrel press while hovering as a pen-down
//! and its release mid-stroke as a lift. [`WinitTool`] follows winit's contact exactly that way,
//! one flag: what W would do, which the report measures.
//!
//! One protocol frame can become two winit events at one position: a pen-down's button event
//! and the motion of the same frame (winit's `wp_tablet_input_v2.rs` queues both and sends them
//! together). Our backend makes one sample of such a frame, so here the second event of a pair in
//! one wakeup, at the same position, one of them a button, takes the first one's time instead of
//! a slot of its own on the de-jittered clock. The row is still recorded.

use std::time::Instant;

use input::{Buttons, Dejitter, Phase};

/// A winit tablet tool button, as winit names it, with what the probe needs to write and read
/// it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WinitButton {
    /// `TabletToolButton::Contact`: the tip, and (winit's mapping) `BTN_STYLUS`.
    Contact,
    /// `TabletToolButton::Barrel`: `BTN_STYLUS2`.
    Barrel,
    /// `TabletToolButton::Other(n)`: `Other(1)` is `BTN_STYLUS3`, the rest unmapped codes.
    Other(u16),
}

impl WinitButton {
    /// The button winit reported.
    pub(crate) const fn from_winit(button: winit::event::TabletToolButton) -> Self {
        use winit::event::TabletToolButton as B;
        match button {
            B::Contact => Self::Contact,
            B::Barrel => Self::Barrel,
            B::Other(n) => Self::Other(n),
        }
    }

    /// Its name in the CSV and the report: winit's variant name.
    pub(crate) fn name(self) -> String {
        match self {
            Self::Contact => "Contact".to_owned(),
            Self::Barrel => "Barrel".to_owned(),
            Self::Other(n) => format!("Other({n})"),
        }
    }

    /// The button [`WinitButton::name`] names; `None` for anything else.
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "Contact" => Some(Self::Contact),
            "Barrel" => Some(Self::Barrel),
            other => other
                .strip_prefix("Other(")?
                .strip_suffix(')')?
                .parse()
                .ok()
                .map(Self::Other),
        }
    }
}

/// One winit tablet event, as recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WinitEvent {
    /// `PointerEntered` from a tablet tool: it came into proximity over the window.
    Entered,
    /// `PointerLeft` from a tablet tool: it left proximity, or the window.
    Left,
    /// `PointerMoved` from a tablet tool.
    Moved,
    /// `PointerButton` from a tablet tool.
    Button {
        /// Which button winit says.
        button: WinitButton,
        /// Pressed, or released.
        pressed: bool,
    },
}

impl WinitEvent {
    /// Its name in the CSV's `event` column.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Entered => "entered",
            Self::Left => "left",
            Self::Moved => "moved",
            Self::Button { .. } => "button",
        }
    }

    /// Whether it is a button event.
    const fn is_button(self) -> bool {
        matches!(self, Self::Button { .. })
    }
}

/// What [`WinitTool::apply`] made of one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stamped {
    /// The phase winit's contact implies.
    pub(crate) phase: Phase,
    /// Whether the tool is in proximity after the event.
    pub(crate) proximity: bool,
    /// The barrel buttons winit reports held after the event (`BTN_STYLUS` never: winit calls
    /// it contact).
    pub(crate) buttons: Buttons,
    /// The de-jittered time, for contact phases; `None` otherwise.
    pub(crate) timestamp: Option<Instant>,
}

/// The last event, for telling a frame's second event from the next frame's first.
#[derive(Debug, Clone, Copy)]
struct Last {
    /// The event-loop wakeup it came in.
    wakeup: u64,
    /// Its position.
    pos: (f64, f64),
    /// Whether it was a button event.
    button: bool,
    /// Its de-jittered time, if it had one.
    timestamp: Option<Instant>,
}

/// winit's tablet tool, followed as a stroke engine on winit would: see the module notes.
#[derive(Debug, Default)]
pub(crate) struct WinitTool {
    /// Between `PointerEntered` and `PointerLeft`.
    proximity: bool,
    /// winit's contact: set by a `Contact` press, cleared by a `Contact` release.
    contact: bool,
    /// `Barrel` (as our secondary) and `Other(1)` (as our tertiary).
    buttons: Buttons,
    /// The de-jitter filter contact phases go through.
    dejitter: Dejitter,
    /// The last event applied.
    last: Option<Last>,
}

impl WinitTool {
    /// Applies `event` at `pos`, received at `received` in wakeup `wakeup`, and says what it was.
    pub(crate) fn apply(
        &mut self,
        event: WinitEvent,
        pos: (f64, f64),
        received: Instant,
        wakeup: u64,
    ) -> Stamped {
        let phase = self.phase(event);
        let same_frame = self.last.and_then(|last| {
            (last.wakeup == wakeup && last.pos == pos && (last.button || event.is_button()))
                .then_some(last.timestamp)
                .flatten()
        });
        let timestamp = match phase {
            Phase::Down => {
                self.dejitter.end_stroke();
                Some(self.dejitter.push(received))
            }
            Phase::Drag => Some(same_frame.unwrap_or_else(|| self.dejitter.push(received))),
            Phase::Up => {
                let t = same_frame.unwrap_or_else(|| self.dejitter.push(received));
                self.dejitter.end_stroke();
                Some(t)
            }
            Phase::Leave => {
                self.dejitter.end_stroke();
                None
            }
            Phase::Enter | Phase::Hover => None,
        };
        self.last = Some(Last {
            wakeup,
            pos,
            button: event.is_button(),
            timestamp,
        });
        Stamped {
            phase,
            proximity: self.proximity,
            buttons: self.buttons,
            timestamp,
        }
    }

    /// The phase `event` implies, with the state it changes updated.
    fn phase(&mut self, event: WinitEvent) -> Phase {
        let moving = |contact: bool| if contact { Phase::Drag } else { Phase::Hover };
        match event {
            WinitEvent::Entered => {
                self.proximity = true;
                Phase::Enter
            }
            WinitEvent::Left => {
                self.proximity = false;
                self.contact = false;
                Phase::Leave
            }
            WinitEvent::Moved => moving(self.contact),
            WinitEvent::Button {
                button: WinitButton::Contact,
                pressed,
            } => match (self.contact, pressed) {
                (false, true) => {
                    self.contact = true;
                    Phase::Down
                }
                (true, false) => {
                    self.contact = false;
                    Phase::Up
                }
                // A second press while touching (the tip and `BTN_STYLUS` both read as contact),
                // or a release while not: no change to the one flag.
                (contact, _) => moving(contact),
            },
            WinitEvent::Button { button, pressed } => {
                let bit = match button {
                    WinitButton::Barrel => Some(Buttons::SECONDARY),
                    WinitButton::Other(1) => Some(Buttons::TERTIARY),
                    WinitButton::Contact | WinitButton::Other(_) => None,
                };
                if let Some(bit) = bit {
                    self.buttons = self.buttons.with(bit, pressed);
                }
                moving(self.contact)
            }
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use std::time::Duration;

    use super::*;

    const fn button(button: WinitButton, pressed: bool) -> WinitEvent {
        WinitEvent::Button { button, pressed }
    }

    #[test]
    fn button_names_round_trip() {
        for b in [
            WinitButton::Contact,
            WinitButton::Barrel,
            WinitButton::Other(1),
            WinitButton::Other(331),
        ] {
            assert_eq!(WinitButton::parse(&b.name()), Some(b), "{b:?}");
        }
        assert_eq!(WinitButton::parse("Tip"), None);
        assert_eq!(WinitButton::parse("Other(x)"), None);
    }

    #[test]
    fn contact_press_and_release_are_down_and_up() {
        let t0 = Instant::now();
        let mut tool = WinitTool::default();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let phases: Vec<Phase> = [
            (WinitEvent::Entered, 0),
            (WinitEvent::Moved, 6),
            (button(WinitButton::Contact, true), 12),
            (WinitEvent::Moved, 18),
            (button(WinitButton::Contact, false), 24),
            (WinitEvent::Moved, 30),
            (WinitEvent::Left, 36),
        ]
        .into_iter()
        .zip(0_u32..)
        .map(|((event, ms), i)| tool.apply(event, (f64::from(i), 0.0), at(ms), u64::from(i)))
        .map(|s| s.phase)
        .collect();
        assert_eq!(
            phases,
            [
                Phase::Enter,
                Phase::Hover,
                Phase::Down,
                Phase::Drag,
                Phase::Up,
                Phase::Hover,
                Phase::Leave
            ]
        );
    }

    #[test]
    fn a_barrel_one_release_mid_stroke_ends_the_stroke() {
        // Tip down, BTN_STYLUS pressed (winit: Contact again), released (Contact released):
        // winit's one contact flag reads the release as a lift, though the tip still touches.
        let t0 = Instant::now();
        let mut tool = WinitTool::default();
        let events = [
            button(WinitButton::Contact, true),
            WinitEvent::Moved,
            button(WinitButton::Contact, true),
            WinitEvent::Moved,
            button(WinitButton::Contact, false),
            WinitEvent::Moved,
            button(WinitButton::Contact, false),
        ];
        let phases: Vec<Phase> = events
            .into_iter()
            .zip(0_u32..)
            .map(|(event, i)| {
                let ms = u64::from(i) * 6;
                tool.apply(
                    event,
                    (f64::from(i) * 3.0, 0.0),
                    t0 + Duration::from_millis(ms),
                    u64::from(i),
                )
                .phase
            })
            .collect();
        assert_eq!(
            phases,
            [
                Phase::Down,
                Phase::Drag,
                Phase::Drag,
                Phase::Drag,
                Phase::Up,
                Phase::Hover,
                Phase::Hover
            ],
            "the stroke ends at the barrel's release"
        );
    }

    #[test]
    fn barrel_two_and_three_are_buttons_not_contact() {
        let t0 = Instant::now();
        let mut tool = WinitTool::default();
        let pressed = tool.apply(button(WinitButton::Barrel, true), (0.0, 0.0), t0, 0);
        assert_eq!(pressed.phase, Phase::Hover);
        assert_eq!(pressed.buttons, Buttons::SECONDARY);
        let third = tool.apply(button(WinitButton::Other(1), true), (0.0, 0.0), t0, 1);
        assert_eq!(
            third.buttons,
            Buttons::SECONDARY.with(Buttons::TERTIARY, true)
        );
        let released = tool.apply(button(WinitButton::Barrel, false), (0.0, 0.0), t0, 2);
        assert_eq!(released.buttons, Buttons::TERTIARY);
        let unknown = tool.apply(button(WinitButton::Other(7), true), (0.0, 0.0), t0, 3);
        assert_eq!(
            unknown.buttons,
            Buttons::TERTIARY,
            "unmapped codes are ignored"
        );
    }

    #[test]
    fn only_contact_phases_are_stamped() {
        let t0 = Instant::now();
        let mut tool = WinitTool::default();
        assert_eq!(
            tool.apply(WinitEvent::Moved, (0.0, 0.0), t0, 0).timestamp,
            None
        );
        let down = tool.apply(button(WinitButton::Contact, true), (1.0, 0.0), t0, 1);
        assert_eq!(
            down.timestamp,
            Some(t0),
            "a stroke's first sample is its receipt"
        );
    }

    #[test]
    fn a_frames_second_event_shares_its_time() {
        let t0 = Instant::now();
        let mut tool = WinitTool::default();
        let down = tool.apply(button(WinitButton::Contact, true), (5.0, 5.0), t0, 1);
        // The same frame's motion, a moment later in the same wakeup: no slot of its own.
        let later = t0 + Duration::from_micros(30);
        let moved = tool.apply(WinitEvent::Moved, (5.0, 5.0), later, 1);
        assert_eq!(moved.phase, Phase::Drag);
        assert_eq!(moved.timestamp, down.timestamp, "the pen-down's time");
        // The next frame, in the next wakeup: its own slot.
        let next = t0 + Duration::from_millis(6);
        let drag = tool.apply(WinitEvent::Moved, (5.0, 5.0), next, 2);
        assert_eq!(drag.timestamp, Some(next), "a new slot");
    }

    /// Through the filter a jittered synthetic winit stroke comes out far steadier.
    ///
    /// The trace: a pen sampling every 5.8 ms, its events delivered late by a repeating 0–4 ms,
    /// every frame's motion arriving as one winit event in its own wakeup.
    #[test]
    fn dejitter_steadies_a_jittered_winit_stroke() {
        let t0 = Instant::now();
        let period = Duration::from_micros(5_800);
        let delays_us = [0_u64, 3_100, 900, 4_000, 1_700, 2_500, 400, 3_600];
        let mut tool = WinitTool::default();
        let n = 300_u32;
        let mut received = Vec::new();
        let mut stamped = Vec::new();
        for i in 0..n {
            let event = match i {
                0 => button(WinitButton::Contact, true),
                i if i + 1 == n => button(WinitButton::Contact, false),
                _ => WinitEvent::Moved,
            };
            let delay = delays_us
                .get(usize::try_from(i).expect("small") % delays_us.len())
                .copied()
                .unwrap_or(0);
            let at = t0 + period * i + Duration::from_micros(delay);
            let s = tool.apply(event, (f64::from(i) * 4.0, 0.0), at, u64::from(i));
            received.push(at);
            stamped.push(s.timestamp.expect("contact phases are stamped"));
        }
        let spread = |times: &[Instant]| {
            let dts: Vec<f64> = times
                .windows(2)
                .filter_map(|w| match w {
                    [a, b] => Some(b.duration_since(*a).as_secs_f64()),
                    _ => None,
                })
                .skip(64)
                .collect();
            let mean = dts.iter().sum::<f64>() / f64::from(u32::try_from(dts.len()).unwrap());
            let var = dts.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
                / f64::from(u32::try_from(dts.len()).unwrap());
            var.sqrt()
        };
        let (raw, filtered) = (spread(&received), spread(&stamped));
        assert!(raw > 1e-3, "the receipts jitter: sd {raw}");
        assert!(
            filtered < raw / 5.0,
            "de-jittered sd {filtered} against receipt sd {raw}"
        );
        assert!(
            stamped.iter().zip(&received).all(|(s, r)| s <= r),
            "never after the receipt"
        );
    }
}
