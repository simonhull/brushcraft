#!/usr/bin/env bash
# The tracked performance suite (`brushcraft-bench perf`), recorded in the docs repository's
# history. Usage: scripts/perf.sh [--full] [--baseline] [--settle-secs N] [--no-stages]
# History goes to $BRUSHCRAFT_DOCS/research/platform/perf (default ~/Code/docs/brushcraft).
# The tiny project's check, which records nothing (its render part runs the app, built beside the
# bench): cargo build -p app && cargo run -p brushcraft-bench -- perf --tiny
set -euo pipefail
cd "$(dirname "$0")/.."

# Re-run under mise so a shell without `mise activate` still gets the pinned toolchain.
if [[ -z ${BRUSHCRAFT_UNDER_MISE:-} ]]; then
  BRUSHCRAFT_UNDER_MISE=1 exec mise exec -- "$0" "$@"
fi

# The app too: the render part runs it (`brushcraft --script`), found beside the bench.
cargo build --release -p brushcraft-bench -p app
cargo bench --no-run -p geom --bench stages --features bench-stages
exec target/release/brushcraft-bench perf \
  --history "${BRUSHCRAFT_DOCS:-$HOME/Code/docs/brushcraft}/research/platform/perf" "$@"
