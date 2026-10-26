#!/usr/bin/env bash
# The gate for the rust-gpu shader crate, which lives outside the main workspace: run by
# scripts/check.sh with the pinned host toolchain. Usage: crates/render/shaders/check.sh
#
# - the checked-in SPIR-V is current: its stamp matches the sources (else rerun build.sh);
# - the crate's lints are the main workspace's, copied verbatim;
# - rustfmt, and clippy on the host target. Clippy for the SPIR-V target itself needs rust-gpu's
#   nightly and codegen backend (`cargo gpu clippy`), so the gate doesn't run it;
# - cargo deny over its lockfile, with the main deny.toml.
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../../.. && pwd)
crate=brush-gpu

if ! diff -u ../src/spike/brush_gpu.spv.sources <(./sources.sh) >&2; then
  echo "brush_gpu.spv is stale: its sources changed since it was built. run crates/render/shaders/build.sh" >&2
  exit 1
fi

lints() { sed -n '/^\[workspace\.lints\.rust\]/,/^\[profile\./p' "$1" | sed '/^\[profile\./d; /^$/d'; }
if ! diff -u <(lints "$root/Cargo.toml") <(lints "$crate/Cargo.toml") >&2; then
  echo "$crate/Cargo.toml's lints differ from the workspace's: copy [workspace.lints.*] from Cargo.toml" >&2
  exit 1
fi

cargo fmt --manifest-path "$crate/Cargo.toml" --check
cargo clippy --manifest-path "$crate/Cargo.toml" --all-targets -- -D warnings
cargo deny --manifest-path "$crate/Cargo.toml" --config "$root/deny.toml" check
