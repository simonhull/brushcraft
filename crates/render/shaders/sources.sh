#!/usr/bin/env bash
# Prints the stamp of what the checked-in SPIR-V is built from: the rust-gpu version and a hash
# of every source the shader builds read (brush's sources and manifest; each shader crate's
# sources, manifest and lockfile). build.sh writes it beside the SPIR-V
# (crates/render/src/spike/spirv.sources); check.sh recomputes it.
# Usage: crates/render/shaders/sources.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

shaders=(brush-gpu eval-gpu)
versions=$(for crate in "${shaders[@]}"; do
  sed -n 's/^spirv-std = "=\(.*\)"$/\1/p' "crates/render/shaders/$crate/Cargo.toml"
done | sort -u)
if [[ $(wc -l <<<"$versions") -ne 1 || -z $versions ]]; then
  echo "the shader crates must pin one spirv-std version, exactly: found ${versions:-none}" >&2
  exit 1
fi
echo "rust-gpu $versions"
# Paths relative to the repository and sorted bytewise, so the hash is the same on every machine.
{
  find crates/brush/src -type f
  echo crates/brush/Cargo.toml
  for crate in "${shaders[@]}"; do
    find "crates/render/shaders/$crate/src" -type f
    echo "crates/render/shaders/$crate/Cargo.toml"
    echo "crates/render/shaders/$crate/Cargo.lock"
  done
} | LC_ALL=C sort | xargs sha256sum | sha256sum | sed 's/ .*//; s/^/sources /'
