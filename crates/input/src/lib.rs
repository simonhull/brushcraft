//! Pen input for BrushCraft: samples from native pen backends, de-jittered, for the stroke engine.
//!
//! Backends run on a thread of their own and hand [`Event`]s over a bounded queue (`Events`). On
//! Linux the backend speaks the Wayland tablet protocol directly (`wayland::PenBackend`); there
//! is no X11 backend. Design: `specs/2026-10-21-input-pen-backend-design.md` in the docs repo.

mod dejitter;
mod sample;

pub use dejitter::Dejitter;
pub use sample::{
    Buttons, ControlId, Event, OutOfRange, PadEvent, PadEventKind, Phase, Position, StrokeSample,
    SurfaceId, Tilt, Tool, Unit,
};
