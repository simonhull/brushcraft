#!/usr/bin/env bash
# Rebuilds the rust-gpu shaders and copies their SPIR-V where `render` includes it
# (src/spike/brush_gpu.spv). Usage: crates/render/shaders/build.sh
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
# Clippy for the SPIR-V target first: the gate (check.sh) can only run it on the host target.
PATH="$PWD/bin:$PATH" cargo gpu clippy \
  --shader-crate brush-gpu \
  --auto-install-rust-toolchain
PATH="$PWD/bin:$PATH" cargo gpu build \
  --shader-crate brush-gpu \
  --auto-install-rust-toolchain \
  --output-dir brush-gpu/target/spirv-out
cp brush-gpu/target/spirv-out/brush_gpu.spv ../src/spike/brush_gpu.spv
# The stamp check.sh compares, so a stale SPIR-V can't go unnoticed.
./sources.sh > ../src/spike/brush_gpu.spv.sources
echo "updated crates/render/src/spike/brush_gpu.spv and its .sources stamp"
