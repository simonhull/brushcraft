#!/usr/bin/env bash
# Runs the same gates as CI, with the toolchain pinned in mise.toml. Usage: scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Re-run under mise so a shell without `mise activate` still gets the pinned toolchain.
if [[ -z ${BRUSHCRAFT_UNDER_MISE:-} ]]; then
  BRUSHCRAFT_UNDER_MISE=1 exec mise exec -- "$0" "$@"
fi

cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace --all-features --no-tests=warn
cargo test --workspace --doc --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --all-features
cargo deny check
cargo machete crates tools # vendor/ holds upstream crates we only patch
