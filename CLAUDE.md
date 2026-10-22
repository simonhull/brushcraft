# BrushCraft

Rust digital sculpting app. **Specs, roadmap and research are in `~/Code/brushcraft-docs`.** Read
`STATUS.md` there first, then the spec for the current milestone (`specs/`). The code follows the specs;
if they disagree, raise it rather than drifting silently.

## Rules

- **No AI attribution** in commits, PRs or release notes: no `Co-Authored-By`, no "Generated with" lines.
- Workspace lints in the root `Cargo.toml` are the standard, and `clippy.toml` applies them to public and
  private items alike. Don't weaken them; a local
  `#[expect(lint, reason = "…")]` needs a real reason (`#[allow]` is denied).
- `unsafe_code` is denied workspace-wide. A crate that needs it overrides the lint in its own manifest with a
  justification, every block carries a `// SAFETY:` proof, and it is tested under `cargo +nightly miri test`.
- `geom`, `store` and `sculpt` stay headless: no GPU or windowing dependencies.
- Dependencies point downward only (see the crate table in the architecture spec). Add third-party crates
  through `[workspace.dependencies]`.
- Performance claims need a `divan` benchmark before and after. Profile with `samply` before optimizing.
- Never commit benchmark projects (`*.bcrf`); they are gigabytes.
- No competitor product names in this repo (code, comments, docs, commit messages); describe behaviour or our
  own design instead. Research and specs that cite other products live in `~/Code/brushcraft-docs`.
  `scripts/check.sh` enforces it.

## Commands

- `scripts/check.sh`: everything CI runs.
- `cargo nextest run`: tests. `cargo test --doc`: doctests.
- `cargo fmt` uses the pinned nightly's rustfmt (unstable options in `rustfmt.toml`) through `RUSTFMT` in
  `mise.toml`. Stable rustfmt gives different output: point your editor at `scripts/rustfmt`, and run
  anything that needs nightly (fmt, Miri) via `cargo +$BRUSHCRAFT_NIGHTLY …` under mise.
- Toolchain and cargo tools are pinned in `mise.toml`; bump `rust` there and `rust-version` in `Cargo.toml`
  together. Run cargo through mise (`mise exec -- cargo …` in a shell without `mise activate`), or you get
  rustup's default toolchain instead.
