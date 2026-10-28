#!/usr/bin/env bash
# Opens a small project in the app and closes it headless (`--offscreen --frames N`), on the
# adapter the app picks by default: in the gate, where only lavapipe's ICD is visible, that can
# only be lavapipe. Its integration test (crates/app/tests/headless.rs) does the same naming the
# software adapter; this runs the binary as a user would. The project, and the recovery
# directory the app writes, go in a temporary directory removed afterwards.
# Usage: scripts/app-headless.sh (run by check.sh and CI, with VK_ICD_FILENAMES set)
set -euo pipefail
cd "$(dirname "$0")/.."

dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cargo run -p brushcraft-bench -- generate --splits 2 --levels 4 --project "$dir/small.bcrf" >/dev/null
XDG_STATE_HOME=$dir cargo run -p app -- "$dir/small.bcrf" --offscreen --frames 30 --level 4
if [[ -n $(ls -A "$dir/brushcraft/recovery" 2>/dev/null) ]]; then
  echo "app-headless: the app left a recovery file behind" >&2
  exit 1
fi
