#!/usr/bin/env bash
# Runs the same gates as CI, with the toolchain pinned in mise.toml. Usage: scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Re-run under mise so a shell without `mise activate` still gets the pinned toolchain.
if [[ -z ${BRUSHCRAFT_UNDER_MISE:-} ]]; then
  BRUSHCRAFT_UNDER_MISE=1 exec mise exec -- "$0" "$@"
fi

# GPU tests run on lavapipe and nothing else: with only its ICD visible, no test can pick the
# machine's own GPU. It is required, never skipped (lavapipe.sh says how to install it).
VK_ICD_FILENAMES=$(scripts/lavapipe.sh)
export VK_ICD_FILENAMES VK_DRIVER_FILES=$VK_ICD_FILENAMES

cargo fmt --all --check
scripts/headless.sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace --all-features --no-tests=warn
cargo test --workspace --doc --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --all-features
# Private items too, test code included: their links are checked like the public ones'.
RUSTDOCFLAGS='--cfg test -D warnings' cargo doc --workspace --no-deps --document-private-items --all-features
# The performance suite on a tiny project: that it runs, not a measurement. The stage bench is
# built first exactly as perf --tiny builds it (the bench-fast profile), so the run doesn't
# rebuild with other features.
cargo bench --no-run --profile bench-fast -p geom --bench stages --features bench-stages
cargo run -p brushcraft-bench -- perf --tiny
cargo deny check
cargo machete
