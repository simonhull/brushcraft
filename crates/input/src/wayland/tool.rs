//! Tool (pen) events: accumulated per hardware frame into [`StrokeSample`]s.

use std::time::Instant;

use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_tool_v2::{
    self, ButtonState, Type, ZwpTabletToolV2,
};

use super::{State, surface_of};
use crate::{Buttons, Dejitter, Event, Phase, Position, StrokeSample, SurfaceId, Tilt, Tool, Unit};

/// Linux input codes for pen barrel buttons.
const BTN_STYLUS: u32 = 0x14B;
const BTN_STYLUS2: u32 = 0x14C;
const BTN_STYLUS3: u32 = 0x149;

/// One tool's state between frames.
#[derive(Debug, Default)]
pub(super) struct ToolState {
    kind: Tool,
    surface: Option<SurfaceId>,
    pos: Position,
    contact: bool,
    moved: bool,
    pressure: Unit,
    distance: Option<Unit>,
    tilt: Option<Tilt>,
    roll: Option<f64>,
    buttons: Buttons,
    /// Phase changes seen since the last frame, in order.
    pending: Vec<Phase>,
    dejitter: Dejitter,
}

impl ToolState {
    /// Applies one protocol event; a `frame` event returns the samples it completes.
    fn apply(&mut self, event: zwp_tablet_tool_v2::Event, received: Instant) -> Vec<StrokeSample> {
        use zwp_tablet_tool_v2::Event as E;
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            E::Type {
                tool_type: WEnum::Value(kind),
            } => self.kind = tool_kind(kind),
            E::ProximityIn { surface, .. } => {
                self.surface = Some(surface_of(&surface));
                self.pending.push(Phase::Enter);
            }
            E::ProximityOut => self.pending.push(Phase::Leave),
            E::Down { .. } => {
                self.contact = true;
                self.pending.push(Phase::Down);
            }
            E::Up => {
                self.contact = false;
                self.pending.push(Phase::Up);
            }
            E::Motion { x, y } => {
                self.pos = Position { x, y };
                self.moved = true;
            }
            E::Pressure { pressure } => self.pressure = Unit::from_u16_range(pressure),
            E::Distance { distance } => self.distance = Some(Unit::from_u16_range(distance)),
            E::Tilt { tilt_x, tilt_y } => self.tilt = Some(Tilt::new(tilt_x, tilt_y)),
            E::Rotation { degrees } => self.roll = Some(degrees),
            E::Button {
                button,
                state: WEnum::Value(state),
                ..
            } => {
                if let Some(bit) = button_bit(button) {
                    self.buttons = self.buttons.with(bit, state == ButtonState::Pressed);
                }
            }
            E::Frame { time } => return self.frame(time, received),
            _ => {}
        }
        Vec::new()
    }

    /// Turns the frame's accumulated changes into samples.
    fn frame(&mut self, device_ms: u32, received: Instant) -> Vec<StrokeSample> {
        let mut phases = std::mem::take(&mut self.pending);
        if phases.is_empty() && self.moved {
            phases.push(if self.contact {
                Phase::Drag
            } else {
                Phase::Hover
            });
        }
        self.moved = false;
        let Some(surface) = self.surface else {
            return Vec::new();
        };
        phases
            .into_iter()
            .map(|phase| {
                let timestamp = self.timestamp(phase, received);
                StrokeSample {
                    phase,
                    tool: self.kind,
                    surface,
                    pos: self.pos,
                    pressure: if phase.in_stroke() {
                        self.pressure
                    } else {
                        Unit::ZERO
                    },
                    distance: self.distance,
                    tilt: self.tilt,
                    roll: self.roll,
                    buttons: self.buttons,
                    timestamp,
                    received,
                    device_ms,
                    predicted: false,
                }
            })
            .collect()
    }

    /// De-jittered for contact phases; the receipt time otherwise.
    fn timestamp(&mut self, phase: Phase, received: Instant) -> Instant {
        match phase {
            Phase::Down => {
                self.dejitter.end_stroke();
                self.dejitter.push(received)
            }
            Phase::Drag => self.dejitter.push(received),
            Phase::Up => {
                let t = self.dejitter.push(received);
                self.dejitter.end_stroke();
                t
            }
            Phase::Enter | Phase::Hover | Phase::Leave => received,
        }
    }
}

const fn tool_kind(kind: Type) -> Tool {
    match kind {
        Type::Eraser => Tool::Eraser,
        Type::Brush => Tool::Brush,
        Type::Pencil => Tool::Pencil,
        Type::Airbrush => Tool::Airbrush,
        Type::Finger => Tool::Finger,
        Type::Mouse => Tool::Mouse,
        Type::Lens => Tool::Lens,
        // The protocol enum is non-exhaustive: a tool type added later is treated as a pen.
        Type::Pen | _ => Tool::Pen,
    }
}

const fn button_bit(code: u32) -> Option<Buttons> {
    match code {
        BTN_STYLUS => Some(Buttons::PRIMARY),
        BTN_STYLUS2 => Some(Buttons::SECONDARY),
        BTN_STYLUS3 => Some(Buttons::TERTIARY),
        _ => None,
    }
}

impl Dispatch<ZwpTabletToolV2, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwpTabletToolV2,
        event: zwp_tablet_tool_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let received = Instant::now();
        if matches!(event, zwp_tablet_tool_v2::Event::Removed) {
            state.tools.remove(&proxy.id());
            proxy.destroy();
            return;
        }
        let samples = state
            .tools
            .entry(proxy.id())
            .or_default()
            .apply(event, received);
        for sample in samples {
            state.emit(Event::Sample(sample));
        }
    }
}
