# BrushCraft

A digital sculpting application in Rust, built to hold and sculpt billions of polygons.

Design docs and research live in a separate repo, `~/Code/brushcraft-docs`.

## Layout

| Crate | Responsibility |
|---|---|
| `geom` | Base mesh, Catmull–Clark subdivision, correction pyramid, tile math and quantization |
| `store` | Project file, out-of-core tile paging, tile pool, cache budget and spatial index |
| `brush` | Brush math (falloffs, per-vertex deformation) shared by the CPU and the GPU: `no_std`, generic, allocation-free |
| `sculpt` | Brushes: steps that edit a project's corrections through its store and spatial index |
| `input` | Pen input: native pen backends and a de-jitter filter, feeding the stroke engine |
| `brushcraft-bench` (`crates/bench`) | The M0 benchmark harness and the tracked performance suite |
| `pen-probe` (`tools/pen-probe`) | Records what our pen backend or winit delivers from a pen tablet, and summarises it |

`render` and `app` arrive with their milestones.

## Setup

- Install [`mise`](https://mise.jdx.dev) and `rustup`, then run `mise install` in the repo. It installs the pinned
  stable and nightly toolchains and the cargo tools.
- A C compiler: `zstd-sys` builds libzstd.
- `util-linux`, for `taskset`: the performance suite pins its stage benches to one CPU.
- Profiling with `samply` needs `kernel.perf_event_paranoid` at 1 or lower.

## Checks

`scripts/check.sh` runs everything CI runs, in this order:

1. `cargo fmt --check` (the pinned nightly's rustfmt)
2. clippy with all features, then with default features, `-D warnings`
3. `cargo nextest run`
4. doctests
5. `cargo doc` with `-D warnings`
6. the performance suite on a tiny project: the stage bench build, then `brushcraft-bench perf --tiny`
7. `cargo deny check`
8. `cargo machete`

## Commands

`CLAUDE.md` lists the day-to-day commands: tests, formatting, the performance suite and profiling.
