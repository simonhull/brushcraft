#!/usr/bin/env bash
# Prints the stamp of what the checked-in SPIR-V is built from: the rust-gpu version and a hash
# of every source the shader build reads (brush's sources and manifest, the shader crate's
# sources, manifest and lockfile). build.sh writes it beside the SPIR-V; check.sh recomputes it.
# Usage: crates/render/shaders/sources.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

version=$(sed -n 's/^spirv-std = "=\(.*\)"$/\1/p' crates/render/shaders/brush-gpu/Cargo.toml)
echo "rust-gpu ${version:?spirv-std version not found in crates/render/shaders/brush-gpu/Cargo.toml}"
# Paths relative to the repository and sorted bytewise, so the hash is the same on every machine.
{
  find crates/brush/src crates/render/shaders/brush-gpu/src -type f
  echo crates/brush/Cargo.toml
  echo crates/render/shaders/brush-gpu/Cargo.toml
  echo crates/render/shaders/brush-gpu/Cargo.lock
} | LC_ALL=C sort | xargs sha256sum | sha256sum | sed 's/ .*//; s/^/sources /'
