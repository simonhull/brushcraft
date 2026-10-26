#!/usr/bin/env bash
# Checks that the headless crates (geom, store, brush, sculpt) depend on no GPU or windowing
# crate, as the architecture requires; fails naming the crate and what it pulls in.
# Usage: scripts/headless.sh (run by check.sh and CI, with the pinned toolchain).
set -euo pipefail
cd "$(dirname "$0")/.."

# wgpu and its parts, the windowing crates, and raw Vulkan bindings.
forbidden='^(wgpu|wgpu-[a-z-]+|naga|winit|raw-window-handle|ash) '

status=0
for crate in geom store brush sculpt; do
  # Normal dependencies only: what the crate ships with, not its tests or build scripts.
  found=$(cargo tree -e normal -p "$crate" --prefix none --format '{p}' | grep -E "$forbidden" | sed 's/ (\*)$//' | sort -u || true)
  if [[ -n $found ]]; then
    echo "headless crate $crate depends on GPU or windowing crates:" >&2
    echo "$found" | sed 's/^/  /' >&2
    status=1
  fi
done
exit "$status"
