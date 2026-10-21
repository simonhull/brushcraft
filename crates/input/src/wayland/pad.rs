//! Pad events: buttons, rings, strips and dials, reported as they happen.

use std::time::Instant;

use wayland_client::{Connection, Dispatch, Proxy, QueueHandle, WEnum};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_dial_v2::{
    self, ZwpTabletPadDialV2,
};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_group_v2::{
    self, ZwpTabletPadGroupV2,
};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_ring_v2::{
    self, ZwpTabletPadRingV2,
};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_strip_v2::{
    self, ZwpTabletPadStripV2,
};
use wayland_protocols::wp::tablet::zv2::client::zwp_tablet_pad_v2::{
    self, ButtonState, ZwpTabletPadV2,
};

use super::State;
use crate::{ControlId, Event, PadEvent, PadEventKind, Unit};

/// A pad and the controls it announced, so they can all be destroyed with it.
#[derive(Debug)]
pub(super) struct Pad {
    proxy: ZwpTabletPadV2,
    groups: Vec<Group>,
}

/// A pad's group and its rings, strips and dials.
#[derive(Debug)]
struct Group {
    proxy: ZwpTabletPadGroupV2,
    rings: Vec<ZwpTabletPadRingV2>,
    strips: Vec<ZwpTabletPadStripV2>,
    dials: Vec<ZwpTabletPadDialV2>,
}

impl Pad {
    pub(super) const fn new(proxy: ZwpTabletPadV2) -> Self {
        Self {
            proxy,
            groups: Vec::new(),
        }
    }

    /// Destroys the pad's controls, its groups, then the pad.
    pub(super) fn destroy(self) {
        for group in self.groups {
            group.rings.iter().for_each(ZwpTabletPadRingV2::destroy);
            group.strips.iter().for_each(ZwpTabletPadStripV2::destroy);
            group.dials.iter().for_each(ZwpTabletPadDialV2::destroy);
            group.proxy.destroy();
        }
        self.proxy.destroy();
    }
}

/// The tracked group behind `proxy`, if its pad is still around.
fn group_mut<'a>(state: &'a mut State, proxy: &ZwpTabletPadGroupV2) -> Option<&'a mut Group> {
    state
        .pads
        .values_mut()
        .flat_map(|pad| pad.groups.iter_mut())
        .find(|group| group.proxy == *proxy)
}

/// Queues one pad event, stamped now.
fn emit(state: &State, kind: PadEventKind) {
    state.emit(Event::Pad(PadEvent {
        received: Instant::now(),
        kind,
    }));
}

fn control(proxy: &impl Proxy) -> ControlId {
    ControlId::from_raw(proxy.id().protocol_id())
}

impl Dispatch<ZwpTabletPadV2, ()> for State {
    wayland_client::event_created_child!(Self, ZwpTabletPadV2, [
        zwp_tablet_pad_v2::EVT_GROUP_OPCODE => (ZwpTabletPadGroupV2, ()),
    ]);

    fn event(
        state: &mut Self,
        proxy: &ZwpTabletPadV2,
        event: zwp_tablet_pad_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            zwp_tablet_pad_v2::Event::Button {
                button,
                state: WEnum::Value(pressed),
                ..
            } => {
                emit(
                    state,
                    PadEventKind::Button {
                        index: button,
                        pressed: pressed == ButtonState::Pressed,
                    },
                );
            }
            zwp_tablet_pad_v2::Event::Group { pad_group } => {
                if let Some(pad) = state.pads.get_mut(&proxy.id()) {
                    pad.groups.push(Group {
                        proxy: pad_group,
                        rings: Vec::new(),
                        strips: Vec::new(),
                        dials: Vec::new(),
                    });
                }
            }
            zwp_tablet_pad_v2::Event::Removed => {
                if let Some(pad) = state.pads.remove(&proxy.id()) {
                    pad.destroy();
                }
            }
            _ => {}
        }
    }
}

/// Groups announce their rings, strips and dials, tracked so they're destroyed with the pad.
impl Dispatch<ZwpTabletPadGroupV2, ()> for State {
    wayland_client::event_created_child!(Self, ZwpTabletPadGroupV2, [
        zwp_tablet_pad_group_v2::EVT_RING_OPCODE => (ZwpTabletPadRingV2, ()),
        zwp_tablet_pad_group_v2::EVT_STRIP_OPCODE => (ZwpTabletPadStripV2, ()),
        zwp_tablet_pad_group_v2::EVT_DIAL_OPCODE => (ZwpTabletPadDialV2, ()),
    ]);

    fn event(
        state: &mut Self,
        proxy: &ZwpTabletPadGroupV2,
        event: zwp_tablet_pad_group_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(group) = group_mut(state, proxy) else {
            return;
        };
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            zwp_tablet_pad_group_v2::Event::Ring { ring } => group.rings.push(ring),
            zwp_tablet_pad_group_v2::Event::Strip { strip } => group.strips.push(strip),
            zwp_tablet_pad_group_v2::Event::Dial { dial } => group.dials.push(dial),
            _ => {}
        }
    }
}

impl Dispatch<ZwpTabletPadRingV2, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwpTabletPadRingV2,
        event: zwp_tablet_pad_ring_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let control = control(proxy);
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            zwp_tablet_pad_ring_v2::Event::Angle { degrees } => {
                emit(
                    state,
                    PadEventKind::Ring {
                        control,
                        angle: Some(degrees),
                    },
                );
            }
            zwp_tablet_pad_ring_v2::Event::Stop => emit(
                state,
                PadEventKind::Ring {
                    control,
                    angle: None,
                },
            ),
            _ => {}
        }
    }
}

impl Dispatch<ZwpTabletPadStripV2, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwpTabletPadStripV2,
        event: zwp_tablet_pad_strip_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let control = control(proxy);
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "the protocol enum is non-exhaustive; the rest don't matter here"
        )]
        match event {
            zwp_tablet_pad_strip_v2::Event::Position { position } => {
                emit(
                    state,
                    PadEventKind::Strip {
                        control,
                        position: Some(Unit::from_u16_range(position)),
                    },
                );
            }
            zwp_tablet_pad_strip_v2::Event::Stop => emit(
                state,
                PadEventKind::Strip {
                    control,
                    position: None,
                },
            ),
            _ => {}
        }
    }
}

impl Dispatch<ZwpTabletPadDialV2, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &ZwpTabletPadDialV2,
        event: zwp_tablet_pad_dial_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_tablet_pad_dial_v2::Event::Delta { value120 } = event {
            emit(
                state,
                PadEventKind::Dial {
                    control: control(proxy),
                    delta: value120,
                },
            );
        }
    }
}
