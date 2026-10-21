# Vendored crates

Upstream crates we carry patches for, wired in through `[patch.crates-io]` in the root `Cargo.toml`. Each was copied
unmodified from crates.io in its own commit, so `git log -p vendor/<crate>` shows exactly what we changed.

| Crate | Version | Why |
|---|---|---|
| `winit-core` | 0.31.0-beta.3 | `TabletToolData` gains `timestamp` and `distance`. |
| `winit-wayland` | 0.31.0-beta.3 | Fills them from `zwp_tablet_tool_v2` (`frame` time, `distance`), and reports full-precision `angle` from the protocol's fractional tilt. |

Measurement behind it: `research/platform/pen-input-wayland.md` in the docs repo.

The new fields are struct-literal-breaking for winit's iOS and web backends, which aren't vendored. Linux is the only
target for now; revisit before the iOS or web build (upstream the change, or vendor those too).

Drop a crate from here once upstream ships the change.
