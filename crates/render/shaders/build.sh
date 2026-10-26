#!/usr/bin/env bash
# Rebuilds the rust-gpu shaders and copies their SPIR-V where `render` includes it
# (src/spike/<crate>_<entry point>.spv), with a stamp of their sources (spirv.sources).
# Usage: crates/render/shaders/build.sh
# Needs cargo-gpu (pinned in mise.toml); the first run installs rust-gpu's nightly through rustup
# (about 1.5 GB) and builds its codegen backend (about a minute). The gate never runs this: it
# loads the checked-in SPIR-V and fails if its sources changed since (check.sh): rerun this and
# commit the result.
set -euo pipefail
cd "$(dirname "$0")"

# Re-run under mise so a shell without `mise activate` still gets the pinned cargo-gpu.
if [[ -z ${BRUSHCRAFT_UNDER_MISE:-} ]]; then
  BRUSHCRAFT_UNDER_MISE=1 exec mise exec -- "$PWD/build.sh" "$@"
fi

# rust-gpu 0.10 builds on nightly-2026-07-03, rustc 1.98, older than the workspace's
# rust-version (1.99), and cargo refuses to build `brush` on it. bin/cargo adds
# --ignore-rust-version to the shader build rather than lowering `brush`'s rust-version.
#
# Every entry point gets a module of its own (--multimodule), holding only its own bindings, so
# each pipeline is built from exactly what it runs.
for crate in brush-gpu eval-gpu; do
  # Clippy for the SPIR-V target first: the gate (check.sh) can only run it on the host target.
  PATH="$PWD/bin:$PATH" cargo gpu clippy \
    --shader-crate "$crate" \
    --auto-install-rust-toolchain \
    --deny-warnings
  rm -rf "$crate/target/spirv-out"
  PATH="$PWD/bin:$PATH" cargo gpu build \
    --shader-crate "$crate" \
    --auto-install-rust-toolchain \
    --deny-warnings \
    --multimodule \
    --output-dir "$crate/target/spirv-out"
  for module in "$crate"/target/spirv-out/*.spv; do
    spv=${crate//-/_}_$(basename "$module")
    cp "$module" "../src/spike/$spv"
    echo "updated crates/render/src/spike/$spv"
  done
done
# The stamp check.sh compares, so a stale SPIR-V can't go unnoticed.
./sources.sh > ../src/spike/spirv.sources
echo "updated crates/render/src/spike/spirv.sources"
