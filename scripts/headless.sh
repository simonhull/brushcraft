#!/usr/bin/env bash
# Checks that the headless crates (geom, store, brush, sculpt) depend on no GPU or windowing
# crate, as the architecture requires, on every target and with every feature; fails naming the
# crate and what it pulls in, and fails if cargo itself fails.
# Usage: scripts/headless.sh [crate...] (default: the headless crates; run by check.sh and CI)
set -euo pipefail
cd "$(dirname "$0")/.."

# wgpu and its parts, the shader translator, the windowing crates, and raw graphics bindings.
forbidden='^(wgpu|wgpu-[a-z-]+|naga|naga-[a-z-]+|winit|raw-window-handle|ash|metal|objc2-metal|glow|vulkano) '

crates=("$@")
if [[ ${#crates[@]} -eq 0 ]]; then
  crates=(geom store brush sculpt)
fi

status=0
for crate in "${crates[@]}"; do
  # Normal dependencies only: what the crate ships with, not its tests or build scripts. The
  # tree is captured first, so a failing cargo fails the check instead of looking clean.
  if ! tree=$(cargo tree -e normal -p "$crate" --target all --all-features --prefix none --format '{p}'); then
    echo "headless check: cargo tree failed for $crate" >&2
    status=1
    continue
  fi
  found=$(grep -E "$forbidden" <<<"$tree" | sed 's/ (\*)$//' | sort -u || true)
  if [[ -n $found ]]; then
    echo "headless crate $crate depends on GPU or windowing crates:" >&2
    sed 's/^/  /' <<<"$found" >&2
    status=1
  fi
done
exit "$status"
