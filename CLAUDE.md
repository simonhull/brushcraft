# BrushCraft

Rust digital sculpting app. **Specs, roadmap and research are in `~/Code/brushcraft-docs`.** Read
`STATUS.md` there first, then the spec for the current milestone (`specs/`). The code follows the specs;
if they disagree, raise it rather than drifting silently.

## Rules

- **No AI attribution** in commits, PRs or release notes: no `Co-Authored-By`, no "Generated with" lines.
- Workspace lints in the root `Cargo.toml` are the standard, and `clippy.toml` applies them to public and
  private items alike. Don't weaken them; a local
  `#[expect(lint, reason = "…")]` needs a real reason (`#[allow]` is denied).
- `unsafe_code` is denied workspace-wide. Where FFI needs it, allow it at the narrowest scope with
  `#[expect(unsafe_code, reason = "…")]`; every block carries a `// SAFETY:` proof. Pure-Rust `unsafe` must also
  pass `cargo +$BRUSHCRAFT_NIGHTLY miri test` for its crate. FFI can't run under Miri, so it is covered by review
  and the integration tests.
- `geom`, `store`, `brush` and `sculpt` stay headless: no GPU or windowing dependencies (`scripts/headless.sh`
  checks it in the gate).
- Dependencies point downward only (see the crate table in the architecture spec). Add third-party crates
  through `[workspace.dependencies]`.
- Performance claims need a `divan` benchmark before and after. Profile with `samply` before optimizing.
- Never commit benchmark projects (`*.bcrf`); they are gigabytes.

## Commands

- `scripts/check.sh`: everything CI runs.
- `cargo nextest run`: tests. `cargo test --doc`: doctests.
- GPU tests (`render`) run on lavapipe, software Vulkan, through `GpuTest`; `check.sh` requires it (`sudo pacman -S
  vulkan-swrast`) and hides every other GPU from the tests. `cargo run -p render --example adapters` lists the GPUs.
- `cargo fmt` uses the pinned nightly's rustfmt (unstable options in `rustfmt.toml`) through `RUSTFMT` in
  `mise.toml`. Stable rustfmt gives different output: point your editor at `scripts/rustfmt`, and run
  anything that needs nightly (fmt, Miri) via `cargo +$BRUSHCRAFT_NIGHTLY …` under mise.
- `scripts/perf.sh`: the tracked performance suite (stage benches and a cached medium project; `--full` adds the
  benchmark project), appended to the docs repo's `research/platform/perf/<machine>.jsonl` and compared with the
  latest baseline; exit status 3 flags a slowdown beyond noise (or peak RSS over budget + 512 MiB), 4 a comparison
  too noisy to conclude. Run it before merging into `main` and on any change to a hot path; `--full` at milestones;
  `--baseline` only on a quiet machine, from a binary built at a clean commit.
- Profiling: `samply` (pinned in `mise.toml`) on a release build, e.g. `samply record --save-only -o
  /tmp/brush.json.gz target/release/brushcraft-bench brush`, then `samply load /tmp/brush.json.gz`. On Linux it
  needs `kernel.perf_event_paranoid` at 1 or lower (`echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid`).
- Property tests: the parsers' `bolero` checks run in `cargo nextest` (a second each); fuzz one for
  longer with `cargo +$BRUSHCRAFT_NIGHTLY bolero test -p store file::fuzz::<harness> -T 60s`.
- Whole-file fuzzing, in `fuzz/` (a workspace of its own): `cd fuzz && cargo +$BRUSHCRAFT_NIGHTLY fuzz
  run --target x86_64-unknown-linux-gnu project_open -- -max_total_time=300 -timeout=10` (or
  `mesh_from_bytes`). The `--target` is needed: the pinned cargo-fuzz binary defaults to musl. Seeds are
  `fuzz/corpus/<target>/seed-*`, rewritten by `cargo nextest run -p store --run-ignored only
  write_fuzz_seeds`; grown corpora and crash artifacts stay untracked.
- Test strength: `cargo mutants -p store --in-diff <(git diff main...)` (and `-p geom`), `--jobs` at most
  half the cores; settings in `.cargo/mutants.toml`. Survivors in `input/src/dejitter.rs` threshold
  comparisons are tuning slack. The whole workspace runs weekly in CI (`.github/workflows/weekly.yml`).
- Coverage, a map with no threshold: `cargo llvm-cov nextest --workspace --all-features --profile coverage --html` (report in
  `target/llvm-cov/html`); code run only by the killed crash-test children shows as uncovered.
- Proofs of the integer arithmetic: `cargo kani -p store` and `cargo kani -p geom` (the `#[cfg(kani)] mod
  proofs` blocks; `cargo kani setup` once). Kani brings its own nightly toolchain.
- Toolchain and cargo tools are pinned in `mise.toml`; bump `rust` there and `rust-version` in `Cargo.toml`
  together. Run cargo through mise (`mise exec -- cargo …` in a shell without `mise activate`), or you get
  rustup's default toolchain instead.
