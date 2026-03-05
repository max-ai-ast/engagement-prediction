#!/usr/bin/env bash
#
# Training sweep: run every model-type × user-summarization permutation
# against a single shared data-prep run.
#
# MLP experiments run in parallel (lightweight, ~160 MiB GPU each).
# Two-tower experiments run sequentially (heavy full-transformer encoder).
#
# Each invocation of cli.py creates its own ClearML experiment, so all
# runs are tracked automatically.
#
# Usage (survives SSH disconnect):
#   tmux new-session -d -s training-sweep './run_training_sweep.sh'
#   tmux attach -t training-sweep   # to watch live
#   # Ctrl-B D to detach; reconnect later with tmux attach -t training-sweep
#
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
DATA_DIR="outputs/20260213_005409_all_mlp"

EPOCHS=300
BATCH_SIZE=2048
PATIENCE=50
EMA_ALPHA=0.1    # only used when user-summarization=ema
MAX_PARALLEL=4   # max concurrent MLP jobs

# ── Permutations ───────────────────────────────────────────────────────
# Each entry is "user_encoder:user_summarization" (summarization ignored when encoder != summarized)
#
# MLP experiments (parallelisable -- tiny GPU footprint)
MLP_EXPERIMENTS=(
  "summarized:mean"
  "summarized:ema"
  "summarized:linear_recency"
  "full_transformer:"
)

# Two-tower experiments (sequential -- heavy GPU/memory usage)
TT_EXPERIMENTS=(
  "full_transformer:"
  "cross_attention:"
)

# ── Resolve paths ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR_ABS="$(cd "$DATA_DIR" && pwd)"
LOG_DIR="$DATA_DIR_ABS/sweep_logs"
mkdir -p "$LOG_DIR"

SWEEP_LOG="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }

TOTAL=$(( ${#MLP_EXPERIMENTS[@]} + ${#TT_EXPERIMENTS[@]} ))
log "════════════════════════════════════════════════════════════════"
log "Training sweep: $TOTAL experiments"
log "  MLP:       ${#MLP_EXPERIMENTS[@]} (parallel, up to $MAX_PARALLEL at a time)"
log "  Two-tower: ${#TT_EXPERIMENTS[@]} (sequential)"
log "Data dir: $DATA_DIR_ABS"
log "Epochs=$EPOCHS  Batch=$BATCH_SIZE  Patience=$PATIENCE  EMA_alpha=$EMA_ALPHA"
log "════════════════════════════════════════════════════════════════"

# ── Helper: launch one experiment ──────────────────────────────────────
# Writes exit code to a status file so we can collect results later.
# Args: MODEL_TYPE  USER_ENCODER  USER_SUMM  RUN_TAG  RUN_LABEL  RUN_LOG  STATUS_FILE
run_one() {
  local MODEL_TYPE="$1"
  local USER_ENCODER="$2"
  local USER_SUMM="$3"
  local RUN_TAG="$4"
  local RUN_LABEL="$5"
  local RUN_LOG="$6"
  local STATUS_FILE="$7"

  local CMD=(
    python3 cli.py
    --output-dir "$DATA_DIR_ABS"
    --start-from train --stop-after train
    --model-type "$MODEL_TYPE"
    --user-encoder "$USER_ENCODER"
    --run-tag "$RUN_TAG"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --patience "$PATIENCE"
  )

  # Only pass --user-summarization when the encoder is "summarized"
  if [[ "$USER_ENCODER" == "summarized" && -n "$USER_SUMM" ]]; then
    CMD+=(--user-summarization "$USER_SUMM")
    if [[ "$USER_SUMM" == "ema" ]]; then
      CMD+=(--ema-alpha "$EMA_ALPHA")
    fi
  fi

  log "[$RUN_LABEL] Launching: ${CMD[*]}"

  if "${CMD[@]}" > "$RUN_LOG" 2>&1; then
    echo "0" > "$STATUS_FILE"
    log "[$RUN_LABEL] ✓ Completed successfully"
  else
    local EC=$?
    echo "$EC" > "$STATUS_FILE"
    log "[$RUN_LABEL] ✗ Failed (exit $EC). See $RUN_LOG"
  fi
}

# ── Phase 1: MLP experiments (parallel) ───────────────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  Phase 1: MLP experiments (parallel, max $MAX_PARALLEL)           ║"
log "╚══════════════════════════════════════════════════════════════╝"

MLP_PIDS=()
MLP_STATUS_FILES=()
MLP_LABELS=()
RUNNING=0

for i in "${!MLP_EXPERIMENTS[@]}"; do
  SPEC="${MLP_EXPERIMENTS[$i]}"
  USER_ENCODER="${SPEC%%:*}"
  USER_SUMM="${SPEC#*:}"
  RUN_NUM=$((i + 1))
  # Build a human-readable tag for the output directory
  if [[ -n "$USER_SUMM" ]]; then
    RUN_TAG="mlp_${USER_ENCODER}_${USER_SUMM}"
  else
    RUN_TAG="mlp_${USER_ENCODER}"
  fi
  RUN_LABEL="MLP ${RUN_NUM}/${#MLP_EXPERIMENTS[@]}  ${RUN_TAG}"
  RUN_LOG="$LOG_DIR/run_${RUN_TAG}.log"
  STATUS_FILE="$LOG_DIR/.status_${RUN_TAG}"

  # Throttle: wait for a slot if we're at capacity
  while (( RUNNING >= MAX_PARALLEL )); do
    # Wait for any one background job to finish
    wait -n 2>/dev/null || true
    # Recount how many are still running (only count jobs owned by this shell)
    RUNNING=0
    # Capture current running job PIDs for this shell
    mapfile -t CURRENT_JOB_PIDS < <(jobs -pr)
    for pid in "${MLP_PIDS[@]}"; do
      if printf '%s\n' "${CURRENT_JOB_PIDS[@]}" | grep -qw -- "$pid"; then
        (( RUNNING++ )) || true
      fi
    done
  done

  run_one "mlp" "$USER_ENCODER" "$USER_SUMM" "$RUN_TAG" "$RUN_LABEL" "$RUN_LOG" "$STATUS_FILE" &
  MLP_PIDS+=($!)
  MLP_STATUS_FILES+=("$STATUS_FILE")
  MLP_LABELS+=("$RUN_LABEL")
  (( RUNNING++ )) || true
done

# Wait for all MLP jobs to finish
log ""
log "Waiting for all MLP jobs to finish…"
for pid in "${MLP_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

# Collect MLP results
PASSED=0
FAILED=0
for i in "${!MLP_STATUS_FILES[@]}"; do
  if [[ -f "${MLP_STATUS_FILES[$i]}" ]] && [[ "$(cat "${MLP_STATUS_FILES[$i]}")" == "0" ]]; then
    (( PASSED++ )) || true
  else
    (( FAILED++ )) || true
  fi
  rm -f "${MLP_STATUS_FILES[$i]}"
done

log ""
log "MLP phase complete: $PASSED passed, $FAILED failed out of ${#MLP_EXPERIMENTS[@]}"

# ── Phase 2: Two-tower experiments (sequential) ───────────────────────
log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  Phase 2: Two-tower experiments (sequential)                ║"
log "╚══════════════════════════════════════════════════════════════╝"

for i in "${!TT_EXPERIMENTS[@]}"; do
  SPEC="${TT_EXPERIMENTS[$i]}"
  USER_ENCODER="${SPEC%%:*}"
  USER_SUMM="${SPEC#*:}"
  RUN_NUM=$((i + 1))
  RUN_TAG="two-tower_${USER_ENCODER}"
  RUN_LABEL="TT ${RUN_NUM}/${#TT_EXPERIMENTS[@]}  ${RUN_TAG}"
  RUN_LOG="$LOG_DIR/run_${RUN_TAG}.log"
  STATUS_FILE="$LOG_DIR/.status_${RUN_TAG}"

  log ""
  log "────────────────────────────────────────────────────────────────"
  log "[$RUN_LABEL] Starting…"
  log "────────────────────────────────────────────────────────────────"

  run_one "two-tower" "$USER_ENCODER" "$USER_SUMM" "$RUN_TAG" "$RUN_LABEL" "$RUN_LOG" "$STATUS_FILE"

  if [[ -f "$STATUS_FILE" ]] && [[ "$(cat "$STATUS_FILE")" == "0" ]]; then
    (( PASSED++ )) || true
  else
    (( FAILED++ )) || true
  fi
  rm -f "$STATUS_FILE"
done

# ── Summary ────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "Sweep complete: $PASSED passed, $FAILED failed out of $TOTAL"
log "Logs: $LOG_DIR"
log "  MLP logs:       run_mlp_*.log"
log "  Two-tower logs: run_two-tower_*.log"
log "════════════════════════════════════════════════════════════════"

exit $FAILED
