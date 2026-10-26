#!/usr/bin/env bash
# Prints the stamp of what the checked-in SPIR-V is built from: the rust-gpu version and a hash
# of every tracked file the shader builds read. build.sh writes it to
# crates/render/src/spike/spirv.sources, followed by the SPIR-V files' own hashes; check.sh
# recomputes both. Usage: crates/render/shaders/sources.sh
#
# The files: brush's sources and manifest, build.sh and bin/cargo (the build's flags), and each
# shader crate's sources, manifest and lockfile. Only files git tracks count, so a stray local
# file can't make the stamp differ between machines. Of the root manifest, only what brush
# inherits counts: `[workspace.package]` (edition, rust-version) and the `[workspace.lints.*]`
# tables, verbatim (see `inherited`). A workspace dependency bump leaves the stamp alone; brush has
# no dependencies, and its own manifest is hashed whole, so adding one would count.
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
paths=(
  crates/brush/Cargo.toml
  crates/brush/src
  crates/render/shaders/build.sh
  crates/render/shaders/bin/cargo
)
for crate in "${shaders[@]}"; do
  paths+=("crates/render/shaders/$crate/src" "crates/render/shaders/$crate/Cargo.toml" "crates/render/shaders/$crate/Cargo.lock")
done
# Sorted bytewise and relative to the repository, so the hash is the same on every machine.
# The root manifest's tables brush inherits, verbatim and in file order: every line from a
# `[workspace.package]` or `[workspace.lints...]` header up to the next header.
inherited() {
  awk '/^\[/ { keep = ($0 == "[workspace.package]" || $0 ~ /^\[workspace\.lints(\]|\.)/) } keep' Cargo.toml
}
# Sorted bytewise and relative to the repository, so the hash is the same on every machine; the
# inherited tables' hash last, named as a file would be.
{
  git ls-files -z -- "${paths[@]}" | LC_ALL=C sort -z | xargs -0 sha256sum
  inherited | sha256sum | sed 's/ .*/  Cargo.toml [workspace.package] [workspace.lints.*]/'
} | sha256sum | sed 's/ .*//; s/^/sources /'
