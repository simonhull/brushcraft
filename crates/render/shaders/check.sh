#!/usr/bin/env bash
# The gate for the rust-gpu shader crates, which live outside the main workspace: run by
# scripts/check.sh and CI with the pinned host toolchain. Usage: crates/render/shaders/check.sh
#
# - the checked-in SPIR-V is current: the stamp's sources hash matches the sources, and the
#   modules are exactly the ones it lists, with its hashes (else rerun build.sh);
# - each crate takes the workspace lints (`[lints] workspace = true`) and its `[workspace.lints]`
#   table is the main workspace's, copied verbatim;
# - rustfmt, and clippy on the host target. Clippy for the SPIR-V target itself needs rust-gpu's
#   nightly and codegen backend (`cargo gpu clippy`), so only build.sh runs it;
# - cargo deny over each crate's lockfile, with the main deny.toml.
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../../.. && pwd)
stamp=../src/spike/spirv.sources
rebuild="run crates/render/shaders/build.sh"

if ! diff -u <(head -n 2 "$stamp") <(./sources.sh) >&2; then
  echo "the checked-in SPIR-V is stale: its sources changed since it was built. $rebuild" >&2
  exit 1
fi
listed=$(tail -n +3 "$stamp" | sed 's/^[0-9a-f]*  //' | LC_ALL=C sort)
present=$(cd ../src/spike && ls -- *.spv | LC_ALL=C sort)
if [[ $listed != "$present" ]]; then
  echo "the checked-in SPIR-V modules aren't the ones the stamp lists. $rebuild" >&2
  diff -u <(echo "$listed") <(echo "$present") >&2 || true
  exit 1
fi
if ! (cd ../src/spike && tail -n +3 spirv.sources | sha256sum --check --quiet --strict) >&2; then
  echo "a checked-in SPIR-V module isn't the one the stamp records. $rebuild" >&2
  exit 1
fi

lints() { sed -n '/^\[workspace\.lints\.rust\]/,/^\[profile\./p' "$1" | sed '/^\[profile\./d; /^$/d'; }
for crate in brush-gpu eval-gpu; do
  if ! diff -u <(lints "$root/Cargo.toml") <(lints "$crate/Cargo.toml") >&2; then
    echo "$crate/Cargo.toml's lints differ from the workspace's: copy [workspace.lints.*] from Cargo.toml" >&2
    exit 1
  fi
  # The crate must take that table, and have no lints table of its own.
  if [[ $(sed -n '/^\[lints\]$/,/^\[/p' "$crate/Cargo.toml" | sed '/^\[/d; /^$/d') != "workspace = true" ]] ||
    grep -qE '^\[lints\.' "$crate/Cargo.toml"; then
    echo "$crate/Cargo.toml must take the workspace lints: [lints] workspace = true, and nothing else" >&2
    exit 1
  fi
  cargo fmt --manifest-path "$crate/Cargo.toml" --check
  cargo clippy --manifest-path "$crate/Cargo.toml" --all-targets -- -D warnings
  cargo deny --manifest-path "$crate/Cargo.toml" --config "$root/deny.toml" check
done
