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
            zwp_tablet_pad_v2::Event::Removed => proxy.destroy(),
            _ => {}
        }
    }
}

/// Groups only announce their rings, strips and dials.
impl Dispatch<ZwpTabletPadGroupV2, ()> for State {
    wayland_client::event_created_child!(Self, ZwpTabletPadGroupV2, [
        zwp_tablet_pad_group_v2::EVT_RING_OPCODE => (ZwpTabletPadRingV2, ()),
        zwp_tablet_pad_group_v2::EVT_STRIP_OPCODE => (ZwpTabletPadStripV2, ()),
        zwp_tablet_pad_group_v2::EVT_DIAL_OPCODE => (ZwpTabletPadDialV2, ()),
    ]);

    fn event(
        _: &mut Self,
        _: &ZwpTabletPadGroupV2,
        _: zwp_tablet_pad_group_v2::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
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
