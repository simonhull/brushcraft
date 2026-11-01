//! The fallback: winit's tablet events as pen samples, stamped through `input`'s
//! [`Dejitter`] as our Wayland backend stamps its own (`crates/input/src/wayland/tool.rs`).
//!
//! Adapted from `pen-probe`'s winit source (`tools/pen-probe/src/winit_tool.rs`), which measured
//! it. winit 0.31.0-beta.3 hands a tablet tool over as `PointerEntered`, `PointerMoved`,
//! `PointerButton` and `PointerLeft`, with no sample time: each is stamped on arrival, on the
//! event-loop thread. winit maps `BTN_STYLUS` (the first barrel button) to the tip's contact, so
//! on the fallback that button presses the tip: a barrel press while hovering is a pen-down, and
//! its release mid-stroke a lift. [`WinitPen`] follows winit's one contact flag that way.
//!
//! One protocol frame can become two winit events at one position: a pen-down's button event and
//! the frame's motion. Our backend makes one sample of such a frame, so here the second event of
//! a pair in one wakeup, at the same position, one of them a button, takes the first one's time
//! instead of a slot of its own on the de-jittered clock.

use std::time::Instant;

use input::{Buttons, Dejitter, Phase, Tilt, Tool, Unit};
use winit::event::{
    Force, TabletToolAngle, TabletToolButton, TabletToolData, TabletToolKind, TabletToolTilt,
};

use super::PenSample;
use crate::camera::Pixel;

/// A winit tablet tool button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WinitButton {
    /// `TabletToolButton::Contact`: the tip, and (winit's mapping) `BTN_STYLUS`.
    Contact,
    /// `TabletToolButton::Barrel`: `BTN_STYLUS2`, our second barrel button.
    Barrel,
    /// `TabletToolButton::Other(n)`: `Other(1)` is `BTN_STYLUS3`, the rest unmapped codes.
    Other(u16),
}

impl From<TabletToolButton> for WinitButton {
    fn from(button: TabletToolButton) -> Self {
        match button {
            TabletToolButton::Contact => Self::Contact,
            TabletToolButton::Barrel => Self::Barrel,
            TabletToolButton::Other(n) => Self::Other(n),
        }
    }
}

/// One of winit's tablet tool events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WinitEvent {
    /// `PointerEntered`: the tool came into range over the window.
    Entered,
    /// `PointerLeft`: the tool left range, or the window.
    Left,
    /// `PointerMoved`.
    Moved,
    /// `PointerButton`.
    Button {
        /// Which button.
        button: WinitButton,
        /// Pressed, or released.
        pressed: bool,
    },
}

impl WinitEvent {
    /// Whether it is a button event.
    const fn is_button(self) -> bool {
        matches!(self, Self::Button { .. })
    }
}

/// What winit reports of the tool with an event, in `input`'s terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ToolData {
    /// The kind of tool.
    pub(crate) tool: Tool,
    /// Tip pressure.
    pub(crate) pressure: Unit,
    /// Tilt, if reported.
    pub(crate) tilt: Option<Tilt>,
    /// Barrel rotation in degrees, if reported.
    pub(crate) roll: Option<f64>,
}

impl ToolData {
    /// What winit's `kind` and `data` say of the tool.
    pub(crate) fn from_winit(kind: TabletToolKind, data: &TabletToolData) -> Self {
        let angle = data.angle.or_else(|| data.tilt.map(TabletToolTilt::angle));
        let tilt = data.tilt.or_else(|| data.angle.map(TabletToolAngle::tilt));
        Self {
            tool: tool_kind(kind),
            pressure: data
                .force
                .as_ref()
                .map_or(Unit::ZERO, |force| pressure(force, angle)),
            tilt: tilt.map(|tilt| Tilt::new(f64::from(tilt.x), f64::from(tilt.y))),
            roll: data.twist.map(f64::from),
        }
    }
}

/// `force` as a pressure in `0..=1`; NaN reads as none.
fn pressure(force: &Force, angle: Option<TabletToolAngle>) -> Unit {
    let normalized = force.normalized(angle);
    if normalized.is_nan() {
        return Unit::ZERO;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a pressure in 0..=1 loses nothing that matters in f32, as the protocol's 16 bits"
    )]
    let clamped = normalized.clamp(0.0, 1.0) as f32;
    Unit::try_from(clamped).unwrap_or(Unit::ZERO)
}

/// `input`'s name for winit's tool kind.
const fn tool_kind(kind: TabletToolKind) -> Tool {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "TabletToolKind is non-exhaustive: a kind added later is taken as a pen"
    )]
    match kind {
        TabletToolKind::Eraser => Tool::Eraser,
        TabletToolKind::Brush => Tool::Brush,
        TabletToolKind::Pencil => Tool::Pencil,
        TabletToolKind::Airbrush => Tool::Airbrush,
        TabletToolKind::Finger => Tool::Finger,
        TabletToolKind::Mouse => Tool::Mouse,
        TabletToolKind::Lens => Tool::Lens,
        _ => Tool::Pen,
    }
}

/// The last event, for telling a frame's second event from the next frame's first.
#[derive(Debug, Clone, Copy)]
struct Last {
    /// The event-loop wakeup it came in.
    wakeup: u64,
    /// Its position.
    at: Pixel,
    /// Whether it was a button event.
    button: bool,
    /// Its de-jittered time, if it had one.
    stamped: Option<Instant>,
}

/// winit's tablet tool, followed into pen samples: see the module notes.
#[derive(Debug, Default)]
pub(crate) struct WinitPen {
    /// winit's contact: set by a `Contact` press, cleared by its release or the tool leaving.
    contact: bool,
    /// `Barrel` (our second barrel button) and `Other(1)` (our third) held.
    buttons: Buttons,
    /// What winit last reported of the tool; `PointerLeft` reports nothing.
    data: Option<ToolData>,
    /// The filter a stroke's phases are stamped through.
    dejitter: Dejitter,
    /// The last event applied.
    last: Option<Last>,
}

impl WinitPen {
    /// The sample `event` makes: at `at` (the last position if winit gives none), with `data`
    /// (the last reported if none), received at `received` in event-loop wakeup `wakeup`.
    pub(crate) fn apply(
        &mut self,
        event: WinitEvent,
        at: Option<Pixel>,
        data: Option<ToolData>,
        (received, wakeup): (Instant, u64),
    ) -> PenSample {
        let at = at
            .or_else(|| self.last.map(|last| last.at))
            .unwrap_or_default();
        if data.is_some() {
            self.data = data;
        }
        let phase = self.phase(event);
        let same_frame = self.last.and_then(|last| {
            (last.wakeup == wakeup && last.at == at && (last.button || event.is_button()))
                .then_some(last.stamped)
                .flatten()
        });
        let stamped = match phase {
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
            at,
            button: event.is_button(),
            stamped,
        });
        let tool = self.data.unwrap_or(ToolData {
            tool: Tool::Pen,
            pressure: Unit::ZERO,
            tilt: None,
            roll: None,
        });
        PenSample {
            phase,
            tool: tool.tool,
            at,
            pressure: if phase.in_stroke() && phase != Phase::Up {
                tool.pressure
            } else {
                Unit::ZERO
            },
            distance: None,
            tilt: tool.tilt,
            roll: tool.roll,
            buttons: self.buttons,
            timestamp: stamped.unwrap_or(received),
            received,
        }
    }

    /// The phase `event` implies, with the state it changes updated.
    fn phase(&mut self, event: WinitEvent) -> Phase {
        let moving = |contact: bool| if contact { Phase::Drag } else { Phase::Hover };
        match event {
            WinitEvent::Entered => Phase::Enter,
            WinitEvent::Left => {
                self.contact = false;
                self.buttons = Buttons::default();
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
