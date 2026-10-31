//! The stroke engine: pen samples in, [`Dab`]s out.
//!
//! A stroke runs as a pipeline whose contract is the dab:
//!
//! ```text
//! pen samples ──► Path ──► placement ──► Dab stream ──► (an action, then a target)
//!                smoothing   spacing, size,
//!                            dynamics, symmetry
//! ```
//!
//! This crate is headless: it never sees the GPU, the window or the store. What it knows of the
//! scene comes through two traits its callers implement: a [`Projector`] from the camera and a
//! [`Surface`] to place dabs on.
//!
//! What is here so far:
//! - **Input:** [`Sample`], one pen report in physical pixels and seconds on the pen's clock. The
//!   app converts `input`'s de-jittered samples: `input` carries the platform's pen backends
//!   (Wayland), and `stroke` depends on no windowing code.
//! - **The path:** [`Path`], a speed-adaptive low-pass with an optional lazy string that never
//!   loses a stroke's tail, configured by [`PathSettings`].
//! - **Size:** [`Radius`] and [`SizeMode`].
//! - **Dynamics:** [`Dynamics`], a table of [`Entry`]s mapping an [`Input`] through a [`Curve`]
//!   onto a [`Param`]; [`StrokeRandom`] for the random input.
//! - **The scene:** [`Projector`] (screen to world [`Ray`]s, and pixel size at a depth) and
//!   [`Surface`] (raycasts, continuity, and the closest point on the stroke's region), in `f64`
//!   world space ([`DVec3`]), with [`Hit`]s on the base mesh.
//! - **Settings:** [`StrokeSettings`], validated plain data with a versioned byte form.
//! - **Output:** [`Dab`], plain data shared by the CPU and the GPU.
//!
//! Placing dabs on the surface (spacing, the size freeze, symmetry copies) comes next.
//!
//! Design: `specs/2026-10-31-brush-engine-design.md` §1 in the docs repo.

mod curve;
mod dab;
mod dynamics;
mod path;
mod sample;
mod settings;
mod size;
mod surface;
mod unit;
mod vector;

pub use curve::{Curve, CurveError, CurvePoint, MAX_CURVE_POINTS, MIN_CURVE_POINTS};
pub use dab::Dab;
pub use dynamics::{
    ACTION_SLOTS, ActionSlot, Dynamics, DynamicsError, Entry, FullScale, Input, Inputs,
    MAX_ENTRIES, Param, StrokeRandom, Values,
};
pub use path::{Path, PathError, PathSettings};
pub use sample::{Buttons, Sample, SampleError, ScreenPoint, Tilt, Twist};
pub use settings::{
    Axis, DecodeError, MirrorPlanes, Radial, SettingsError, Spacing, StrokeSettings, Symmetry,
    SymmetryError,
};
pub use size::{Radius, RadiusError, SizeMode};
pub use surface::{Hit, Projector, Ray, Surface};
pub use unit::{NotUnit, Unit};
pub use vector::DVec3;
