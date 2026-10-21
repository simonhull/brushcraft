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
    /// Whether an axis or button changed since the last frame.
    changed: bool,
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
            E::ProximityIn { surface, .. } => self.proximity_in(surface_of(&surface)),
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
                self.changed = true;
            }
            E::Pressure { pressure } => {
                self.pressure = Unit::from_u16_range(pressure);
                self.changed = true;
            }
            E::Distance { distance } => {
                self.distance = Some(Unit::from_u16_range(distance));
                self.changed = true;
            }
            E::Tilt { tilt_x, tilt_y } => {
                self.tilt = Some(Tilt::new(tilt_x, tilt_y));
                self.changed = true;
            }
            E::Rotation { degrees } => {
                self.roll = Some(degrees);
                self.changed = true;
            }
            E::Button {
                button,
                state: WEnum::Value(state),
                ..
            } => {
                if let Some(bit) = button_bit(button) {
                    self.buttons = self.buttons.with(bit, state == ButtonState::Pressed);
                    self.changed = true;
                }
            }
            E::Frame { time } => return self.frame(time, received),
            _ => {}
        }
        Vec::new()
    }

    /// The tool came into proximity over `surface`.
    fn proximity_in(&mut self, surface: SurfaceId) {
        self.surface = Some(surface);
        self.pending.push(Phase::Enter);
    }

    /// Turns the frame's accumulated changes into samples.
    ///
    /// A frame with phase changes yields one sample per change; one with only axis or button
    /// changes yields a `Drag` (in contact) or `Hover` sample.
    fn frame(&mut self, device_ms: u32, received: Instant) -> Vec<StrokeSample> {
        let mut phases = std::mem::take(&mut self.pending);
        if phases.is_empty() && self.changed {
            phases.push(if self.contact {
                Phase::Drag
            } else {
                Phase::Hover
            });
        }
        self.changed = false;
        let Some(surface) = self.surface else {
            return Vec::new();
        };
        let samples = phases
            .iter()
            .map(|&phase| {
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
            .collect();
        if phases.last() == Some(&Phase::Leave) {
            // Out of proximity: the next proximity_in brings fresh values (or none).
            self.surface = None;
            self.distance = None;
            self.tilt = None;
            self.roll = None;
        }
        samples
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
        let (_, tool) = state
            .tools
            .entry(proxy.id())
            .or_insert_with(|| (proxy.clone(), ToolState::default()));
        let samples = tool.apply(event, received);
        for sample in samples {
            state.emit(Event::Sample(sample));
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::missing_panics_doc,
    reason = "test functions are documented by their names; panicking is how they fail"
)]
mod tests {
    use zwp_tablet_tool_v2::Event as E;

    use super::*;

    const SURFACE: SurfaceId = SurfaceId::from_raw(0x1000);

    /// Applies `events`, then a frame, and returns that frame's samples.
    fn frame(tool: &mut ToolState, events: impl IntoIterator<Item = E>) -> Vec<StrokeSample> {
        let now = Instant::now();
        for event in events {
            assert!(tool.apply(event, now).is_empty(), "only a frame emits");
        }
        tool.apply(E::Frame { time: 7 }, now)
    }

    fn phases(samples: &[StrokeSample]) -> Vec<Phase> {
        samples.iter().map(|s| s.phase).collect()
    }

    /// A tool in proximity over [`SURFACE`], its enter frame already consumed.
    fn hovering() -> ToolState {
        let mut tool = ToolState::default();
        tool.proximity_in(SURFACE);
        frame(&mut tool, []);
        tool
    }

    #[test]
    fn enter_down_and_motion_in_one_frame() {
        let mut tool = ToolState::default();
        tool.proximity_in(SURFACE);
        let samples = frame(
            &mut tool,
            [
                E::Down { serial: 1 },
                E::Motion { x: 10.5, y: 20.25 },
                E::Pressure { pressure: 65_535 },
            ],
        );
        assert_eq!(phases(&samples), [Phase::Enter, Phase::Down]);
        for s in &samples {
            assert_eq!(s.surface, SURFACE);
            assert_eq!(s.pos, Position { x: 10.5, y: 20.25 });
            assert_eq!(s.device_ms, 7);
        }
        let pressures: Vec<Unit> = samples.iter().map(|s| s.pressure).collect();
        assert_eq!(
            pressures,
            [Unit::ZERO, Unit::ONE],
            "no pressure while entering"
        );
    }

    #[test]
    fn up_then_proximity_out_ends_and_clears() {
        let mut tool = hovering();
        frame(
            &mut tool,
            [
                E::Down { serial: 1 },
                E::Tilt {
                    tilt_x: 30.0,
                    tilt_y: 0.0,
                },
            ],
        );
        let samples = frame(&mut tool, [E::Up, E::ProximityOut]);
        assert_eq!(phases(&samples), [Phase::Up, Phase::Leave]);
        assert!(
            samples
                .iter()
                .all(|s| s.surface == SURFACE && s.tilt.is_some())
        );
        assert_eq!(tool.surface, None);
        assert_eq!(tool.tilt, None);
        assert!(
            frame(&mut tool, [E::Motion { x: 1.0, y: 1.0 }]).is_empty(),
            "no surface after leaving"
        );
    }

    #[test]
    fn tap_in_one_frame_yields_down_and_up() {
        let mut tool = hovering();
        let samples = frame(&mut tool, [E::Down { serial: 1 }, E::Up]);
        assert_eq!(phases(&samples), [Phase::Down, Phase::Up]);
    }

    #[test]
    fn frame_without_surface_yields_nothing() {
        let mut tool = ToolState::default();
        let samples = frame(
            &mut tool,
            [E::Down { serial: 1 }, E::Motion { x: 1.0, y: 2.0 }],
        );
        assert_eq!(samples, Vec::new());
    }

    #[test]
    fn hover_has_zero_pressure() {
        let mut tool = hovering();
        let samples = frame(
            &mut tool,
            [
                E::Pressure { pressure: 40_000 },
                E::Motion { x: 3.0, y: 4.0 },
            ],
        );
        assert_eq!(phases(&samples), [Phase::Hover]);
        assert_eq!(samples.first().map(|s| s.pressure), Some(Unit::ZERO));
    }

    #[test]
    fn pressure_change_without_motion_is_a_sample() {
        let mut tool = hovering();
        frame(&mut tool, [E::Down { serial: 1 }]);
        let samples = frame(&mut tool, [E::Pressure { pressure: 65_535 }]);
        assert_eq!(phases(&samples), [Phase::Drag]);
        assert_eq!(samples.first().map(|s| s.pressure), Some(Unit::ONE));
    }

    #[test]
    fn empty_frame_yields_nothing() {
        let mut tool = hovering();
        assert_eq!(frame(&mut tool, []), Vec::new());
    }

    #[test]
    fn barrel_buttons_are_tracked() {
        let mut tool = hovering();
        let samples = frame(
            &mut tool,
            [E::Button {
                serial: 1,
                button: BTN_STYLUS2,
                state: WEnum::Value(ButtonState::Pressed),
            }],
        );
        assert_eq!(phases(&samples), [Phase::Hover]);
        assert_eq!(samples.first().map(|s| s.buttons), Some(Buttons::SECONDARY));
    }

    #[test]
    fn tool_types_map() {
        assert_eq!(tool_kind(Type::Pen), Tool::Pen);
        assert_eq!(tool_kind(Type::Eraser), Tool::Eraser);
        assert_eq!(tool_kind(Type::Brush), Tool::Brush);
        assert_eq!(tool_kind(Type::Pencil), Tool::Pencil);
        assert_eq!(tool_kind(Type::Airbrush), Tool::Airbrush);
        assert_eq!(tool_kind(Type::Finger), Tool::Finger);
        assert_eq!(tool_kind(Type::Mouse), Tool::Mouse);
        assert_eq!(tool_kind(Type::Lens), Tool::Lens);
    }

    #[test]
    fn button_codes_map() {
        assert_eq!(button_bit(0x14B), Some(Buttons::PRIMARY));
        assert_eq!(button_bit(0x14C), Some(Buttons::SECONDARY));
        assert_eq!(button_bit(0x149), Some(Buttons::TERTIARY));
        assert_eq!(button_bit(0x110), None, "BTN_LEFT isn't a barrel button");
    }
}
