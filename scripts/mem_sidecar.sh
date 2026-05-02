#!/usr/bin/env bash
#
# Memory sidecar logger — periodic snapshots of system memory + top RSS
# processes, intended to run alongside long sweeps so that any OOM
# (actual or near-miss) is easy to root-cause from a peak-memory timeline.
#
# Usage:
#   ./scripts/mem_sidecar.sh <log_dir> [interval_seconds]
#
# Designed to be backgrounded by the cap-arch sweep harness, which
# kills it via a trap on EXIT.  Safe to send SIGTERM/SIGINT — the
# loop terminates cleanly on the next sleep.
#
# Output format per snapshot:
#   ── 2026-05-01 23:42:17 ──
#   <free -h | head -2>
#   <ps -eo pid,pcpu,pmem,rss,cmd --sort=-rss | head -10>
#

set -u

LOG_DIR="${1:?usage: $0 <log_dir> [interval_seconds]}"
INTERVAL="${2:-30}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mem_$(date +%Y%m%d_%H%M%S).log"

{
  echo "# mem_sidecar.sh"
  echo "# pid=$$"
  echo "# interval=${INTERVAL}s"
  echo "# started=$(date -Iseconds)"
  echo "# host=$(hostname)"
} > "$LOG_FILE"

while true; do
  {
    echo "── $(date '+%Y-%m-%d %H:%M:%S') ──"
    free -h | head -2
    echo
    ps -eo pid,pcpu,pmem,rss,cmd --sort=-rss 2>/dev/null | head -11
    echo
  } >> "$LOG_FILE"
  sleep "$INTERVAL"
done
