#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
RUN_TS="${RUN_TS:-$(rtk /usr/bin/date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/work_dirs/rotation_study_36h_${RUN_TS}}"
START_LOG="$OUT_ROOT/start.log"

rtk /usr/bin/mkdir -p "$OUT_ROOT"
(
  cd "$ROOT_DIR" || exit 1
  rtk setsid tools/rotation_study/run_36h_p0_p3.sh > "$START_LOG" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$OUT_ROOT/pid.txt"
)

printf 'started pid=%s\n' "$(rtk cat "$OUT_ROOT/pid.txt")"
printf 'out_root=%s\n' "$OUT_ROOT"
printf 'result_md=%s\n' "$OUT_ROOT/rotation_study_36h_results.md"
