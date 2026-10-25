//! Tool (pen) events: accumulated per hardware frame into [`StrokeSample`]s.

use std::time::Instant;

use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_tool_v2::{
    self, ButtonState, Type, ZwpTabletToolV2,
};

use super::{State, surface_of};
use crate::{Buttons, Dejitter, Event, Phase, Position, StrokeSample, SurfaceId, Tilt, Tool, Unit};

/// Linux input code of the first pen barrel button.
const BTN_STYLUS: u32 = 0x14B;
/// Linux input code of the second pen barrel button.
const BTN_STYLUS2: u32 = 0x14C;
/// Linux input code of the third pen barrel button.
const BTN_STYLUS3: u32 = 0x149;

/// One tool's state between frames.
#[derive(Debug, Default)]
pub(super) struct ToolState {
    /// What kind of tool it is, as announced.
    kind: Tool,
    /// The surface it is over; `None` out of proximity.
    surface: Option<SurfaceId>,
    /// The latest position, surface-local.
    pos: Position,
    /// Whether the tip touches the tablet.
    contact: bool,
    /// Whether an axis or button changed since the last frame.
    changed: bool,
    /// The latest tip pressure.
    pressure: Unit,
    /// The latest hover distance; `None` until the tool reports one.
    distance: Option<Unit>,
    /// The latest tilt; `None` until the tool reports one.
    tilt: Option<Tilt>,
    /// The latest barrel rotation in degrees; `None` until the tool reports one.
    roll: Option<f64>,
    /// The barrel buttons held.
    buttons: Buttons,
    /// Phase changes seen since the last frame, in order.
    pending: Vec<Phase>,
    /// The device time of the last frame, for samples synthesised between frames.
    device_ms: u32,
    /// Recovers the tool's steady sample clock from the frames' receipt times.
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
            E::ProximityOut => self.proximity_out(),
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

    /// The tool left proximity. The protocol sends `up` first when the tip is down, so a
    /// conforming compositor never leaves a contact open here; should one arrive without it, the
    /// open stroke is closed with an `Up`, as [`removed`](Self::removed) does.
    fn proximity_out(&mut self) {
        if self.contact {
            self.contact = false;
            self.pending.push(Phase::Up);
        }
        self.pending.push(Phase::Leave);
    }

    /// The tool was removed: closes what it left open, an `Up` if it was in contact and a `Leave`
    /// if it was over a surface, so no stroke stays open.
    fn removed(&mut self, received: Instant) -> Vec<StrokeSample> {
        if self.surface.is_none() {
            return Vec::new();
        }
        // A `proximity_out` still waiting for its frame already closes everything.
        if self.pending.last() != Some(&Phase::Leave) {
            if self.contact {
                self.contact = false;
                self.pending.push(Phase::Up);
            }
            self.pending.push(Phase::Leave);
        }
        self.frame(self.device_ms, received)
    }

    /// Turns the frame's accumulated changes into samples.
    ///
    /// A frame with phase changes yields one sample per change; one with only axis or button
    /// changes yields a `Drag` (in contact) or `Hover` sample.
    fn frame(&mut self, device_ms: u32, received: Instant) -> Vec<StrokeSample> {
        self.device_ms = device_ms;
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

/// The tool kind of a protocol tool type.
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

/// The barrel button a Linux input code names; `None` for any other button.
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
            let samples = state
                .tools
                .remove(&proxy.id())
                .map(|(_, mut tool)| tool.removed(received))
                .unwrap_or_default();
            for sample in samples {
                state.emit(Event::Sample(sample));
            }
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
    fn removal_mid_stroke_closes_it() {
        let mut tool = hovering();
        frame(&mut tool, [E::Down { serial: 1 }]);
        frame(&mut tool, [E::Motion { x: 5.0, y: 6.0 }]);
        let samples = tool.removed(Instant::now());
        assert_eq!(phases(&samples), [Phase::Up, Phase::Leave]);
        for s in &samples {
            assert_eq!(s.surface, SURFACE);
            assert_eq!(s.pos, Position { x: 5.0, y: 6.0 });
            assert_eq!(s.device_ms, 7, "the last frame's device time");
        }
        assert!(!tool.contact);
        assert_eq!(tool.surface, None);
    }

    #[test]
    fn proximity_out_while_down_closes_the_stroke() {
        let mut tool = hovering();
        frame(&mut tool, [E::Down { serial: 1 }]);
        let out = frame(&mut tool, [E::ProximityOut]);
        assert_eq!(phases(&out), [Phase::Up, Phase::Leave]);
        assert!(!tool.contact);
        tool.proximity_in(SURFACE);
        assert_eq!(phases(&frame(&mut tool, [])), [Phase::Enter]);
        let back = frame(&mut tool, [E::Motion { x: 1.0, y: 1.0 }]);
        assert_eq!(phases(&back), [Phase::Hover], "back in range, not touching");
    }

    #[test]
    fn removal_while_hovering_leaves() {
        let mut tool = hovering();
        assert_eq!(phases(&tool.removed(Instant::now())), [Phase::Leave]);
    }

    #[test]
    fn removal_after_proximity_out_leaves_once() {
        let mut tool = hovering();
        assert_eq!(tool.apply(E::Up, Instant::now()), Vec::new());
        assert_eq!(tool.apply(E::ProximityOut, Instant::now()), Vec::new());
        assert_eq!(
            phases(&tool.removed(Instant::now())),
            [Phase::Up, Phase::Leave],
            "the pending phases, nothing added"
        );
    }

    #[test]
    fn removal_out_of_proximity_yields_nothing() {
        let mut tool = ToolState::default();
        assert_eq!(tool.removed(Instant::now()), Vec::new());
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

    /// From a hovering tool, any sequence of `Down`, `Up`, `Motion`, `Pressure`, `Frame` and
    /// `ProximityOut` events emits no `Drag` after an `Up` or `Leave` without a `Down` between.
    /// Each step is `(event choice, x, y, pressure)`.
    #[test]
    fn no_drag_without_a_down_after_an_up_or_leave() {
        bolero::check!()
            .with_type::<Vec<(u8, f64, f64, u16)>>()
            .for_each(|steps| {
                let start = Instant::now();
                let mut tool = hovering();
                let mut emitted = Vec::new();
                for (n, &(choice, x, y, pressure)) in steps.iter().enumerate() {
                    let event = match choice % 6 {
                        0 => E::Down {
                            serial: u32::from(choice),
                        },
                        1 => E::Up,
                        2 => E::Motion { x, y },
                        3 => E::Pressure {
                            pressure: pressure.into(),
                        },
                        4 => E::ProximityOut,
                        _ => E::Frame {
                            time: u32::try_from(n).unwrap_or(u32::MAX),
                        },
                    };
                    let received = start + std::time::Duration::from_millis(n.try_into().unwrap());
                    emitted.extend(phases(&tool.apply(event, received)));
                }
                // Hovering: no stroke is open until a `Down`.
                let mut open = false;
                for (n, phase) in emitted.iter().enumerate() {
                    match phase {
                        Phase::Down => open = true,
                        Phase::Up | Phase::Leave => open = false,
                        Phase::Drag => assert!(open, "a drag without a down at {n}: {emitted:?}"),
                        Phase::Enter | Phase::Hover => {}
                    }
                }
            });
    }
}
