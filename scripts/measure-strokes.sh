#!/usr/bin/env bash
# The pen's measured runs (M1a4 Task 9): the app's strokes script (near, far, long, heavy and the
# million-vertex brush) recorded at each size, then summarised a row a stroke by
# `brushcraft-bench strokes-summary` (sample to present p50/p99, the brush, the queue, the
# catch-up and time to exact, level jumps, tiles a dab, each target marked). Run it on a quiet
# machine: it measures latency.
#
# Usage: scripts/measure-strokes.sh [--project FILE | --medium] [--sizes "WxH ..."] [--runs N]
#                                   [--window] [--live] [--adapter A] [--out DIR]
#   --project FILE  the project, its display layer saved (default: the benchmark sphere with its
#                   layer, $XDG_CACHE_HOME/brushcraft/bench/sphere-display.bcrf); each run draws
#                   on a fresh reflinked copy of it, removed after, so the file itself is never
#                   edited
#   --medium        instead, generate the performance suite's medium project (a sixteenth of the
#                   benchmark sphere) and save its display layer, in the work directory
#   --sizes "..."   the frame sizes (default "1920x1080 3840x2160")
#   --runs N        runs at each size (default 3)
#   --window        draw in a window (mailbox presentation where offered, as a user sees it)
#                   instead of offscreen; a window opens on the display for each run
#   --live          after the scripted runs, open the project in a window, recorded, for strokes
#                   drawn by hand with the pen (at the first size); close the window to finish
#   --adapter A     the app's --adapter (default hardware)
#   --out DIR       where the records and the summary go (default
#                   $XDG_CACHE_HOME/brushcraft/measure/<date>-<host>)
# Records: DIR/records/<host>-<offscreen|window>-<size>-run<N>.jsonl (and live-<size>.jsonl); the
# summary: DIR/summary.txt (the human table) and DIR/summary.json (a JSON line a stroke).
# Re-summarise a directory of records with:
#   target/release/brushcraft-bench strokes-summary --records DIR/records
set -euo pipefail
cd "$(dirname "$0")/.."

# Re-run under mise so a shell without `mise activate` still gets the pinned toolchain.
if [[ -z ${BRUSHCRAFT_UNDER_MISE:-} ]]; then
  BRUSHCRAFT_UNDER_MISE=1 exec mise exec -- "$0" "$@"
fi

cache=${XDG_CACHE_HOME:-$HOME/.cache}/brushcraft
host=$(hostname -s)
project=$cache/bench/sphere-display.bcrf
medium=false
sizes="1920x1080 3840x2160"
runs=3
mode=offscreen
live=false
adapter=hardware
out=$cache/measure/$(date +%Y-%m-%d)-$host

while [[ $# -gt 0 ]]; do
  case $1 in
    --project) project=$2; shift 2 ;;
    --medium) medium=true; shift ;;
    --sizes) sizes=$2; shift 2 ;;
    --runs) runs=$2; shift 2 ;;
    --window) mode=window; shift ;;
    --live) live=true; shift ;;
    --adapter) adapter=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    *) sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0" >&2; exit 2 ;;
  esac
done

cargo build --release -p app -p brushcraft-bench
app=$PWD/target/release/brushcraft
bench=$PWD/target/release/brushcraft-bench

records=$out/records
mkdir -p "$records" "$cache/measure"
work=$(mktemp -d "$cache/measure/work.XXXXXX")
trap 'rm -rf "$work"' EXIT

# A copy of $1 at $2 sharing its blocks where the file system can (APFS clones, btrfs/XFS
# reflinks), else a plain copy.
clone() {
  if [[ $(uname) == Darwin ]]; then
    cp -c "$1" "$2" 2>/dev/null || cp "$1" "$2"
  else
    cp --reflink=auto "$1" "$2"
  fi
}

if $medium; then
  project=$work/medium.bcrf
  echo "measure-strokes: generating the medium project in $work" >&2
  "$bench" generate --force --splits 5 --levels 9 --seed 1 --amplitude 0.01 --budget-mib 2048 \
    --project "$project" >/dev/null
  "$bench" display-save --project "$project" --level 9 >/dev/null
  rm -rf "$work/recovery"
fi
[[ -f $project ]] || { echo "measure-strokes: no project at $project" >&2; exit 1; }

# Runs the app on a fresh copy of the project, its recovery files in the work directory, with
# the arguments given, then removes the copy.
run_app() {
  # Named as the project is, so the record's run line says which it was.
  local copy=$work/$(basename "$project")
  clone "$project" "$copy"
  XDG_STATE_HOME=$work/state "$app" "$copy" --level 9 --adapter "$adapter" "$@"
  rm -rf "$copy" "$work/state"
}

offscreen=()
[[ $mode == offscreen ]] && offscreen=(--offscreen)
for size in $sizes; do
  for ((run = 1; run <= runs; run++)); do
    record=$records/$host-$mode-$size-run$run.jsonl
    echo "measure-strokes: $mode $size, run $run of $runs -> $record" >&2
    # (The `+` form: macOS's bash 3.2 calls an empty array unbound.)
    run_app ${offscreen[@]+"${offscreen[@]}"} --script --script-path strokes --size "$size" --record "$record"
  done
done

if $live; then
  size=${sizes%% *}
  record=$records/live-$size.jsonl
  echo "measure-strokes: a live session at $size: draw strokes with the pen, near and far, then" \
    "close the window -> $record" >&2
  run_app --size "$size" --record "$record"
fi

"$bench" strokes-summary --records "$records" > "$out/summary.json" 2> >(tee "$out/summary.txt" >&2)
echo "measure-strokes: records and summary in $out" >&2
