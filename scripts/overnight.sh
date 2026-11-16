#!/usr/bin/env bash
# Unattended chain. Compares three models, then rebuilds the demo with the best one.
#
# The order, and why it is this one:
#   1. waits for the Lovasz run (already started) and evaluates it on the official test set
#   2. waits for the download of the 90 additional scenes
#   3. extracts the patches of the new scenes and retrains the baseline with more data
#   4. evaluates that one too
#   5. picks the best model ON THE OFFICIAL TEST SET, not on validation
#   6. regenerates precompute, downlink measurements, figures, dashboard
#
# Every step that fails is logged and the chain carries on: better to finish
# with four steps out of six than to stop at the second and find everything idle.
#
# Does not touch configs/split.yaml (frozen, never regenerated): val and demo stay
# identical, so the three models are comparable.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=".venv/bin/python"
LOG="artifacts/logs"
mkdir -p "$LOG"
STATUS="$LOG/overnight_status.txt"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say ""; say "=== $* ==="; }

# A job is alive only if a matching PYTHON process exists. A shell that merely
# *names* the pattern -- a wrapper, a tail, an editor open on the file -- is not
# the job.
#
# This distinction cost four hours on the night of 5 September. The Lovasz
# training had been launched from a `bash -c` that, to wait for the GPU, looped
# on `pgrep -f "iceroute.model"`: its own command line contained
# "iceroute.model.train", so it matched itself and never exited. The training
# never started. This chain then waited for `train3_lovasz`, found that stuck
# shell, and sat idle until the 4-hour cap.
#
# `pgrep -af` prints "PID command-line": the second field is the executable.
running() {
  pgrep -af "$1" 2>/dev/null | awk '$2 ~ /python/ {found = 1} END {exit !found}'
}

# Wait for a job to disappear, with a cap: if something hangs, the chain moves
# on instead of staying stuck until morning.
wait_for() {
  local pattern="$1" limit="${2:-21600}" waited=0
  running "$pattern" || { say "No python process for '$pattern': not waiting."; return 0; }
  say "Waiting for: $(pgrep -af "$pattern" | awk '$2 ~ /python/ {print $1}' | tr '\n' ' ')"
  while running "$pattern"; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$limit" ]; then
      say "TIMEOUT after ${limit}s on '$pattern': moving on."
      return 1
    fi
  done
  return 0
}

: > "$STATUS"
say "Starting the unattended chain."

# --- 1. Lovasz --------------------------------------------------------------
step "1/6  Waiting for the Lovasz training"
wait_for "train3_lovasz" 14400
if grep -q "Best mIoU" "$LOG/train3lov.log" 2>/dev/null; then
  say "$(grep 'Best mIoU' "$LOG/train3lov.log" | tail -1)"
  step "1b   Evaluating Lovasz on the official test set"
  $PY -m iceroute.model.evaluate --config configs/train3_lovasz.yaml \
      --split official_test --scenes 20 --skip-baseline \
      --out artifacts/eval_lovasz > "$LOG/eval_lovasz.log" 2>&1 \
    && say "Lovasz: $(grep -E '^\s+mIoU' "$LOG/eval_lovasz.log" | tail -1)" \
    || say "FAILED: Lovasz evaluation."
else
  say "The Lovasz training produced no result."
fi

# --- 2. Download ------------------------------------------------------------
step "2/6  Waiting for the download of the additional scenes"
wait_for "download_subset" 21600
say "Scenes in data/raw/train: $(ls data/raw/train/*.nc 2>/dev/null | wc -l)"

# --- 3. Patches + training with more data ----------------------------------
if [ -f configs/split_extra.yaml ]; then
  step "3/6  Extracting the patches of the new scenes"
  $PY -m iceroute.data.patches --config configs/data.yaml \
      > "$LOG/patches_extra.log" 2>&1 \
    && say "$(grep 'Total patches' "$LOG/patches_extra.log" | tail -1)" \
    || say "FAILED: patch extraction."

  step "4/6  Retraining the baseline with more data"
  $PY -m iceroute.model.train --config configs/train3_more.yaml \
      > "$LOG/train3more.log" 2>&1 \
    && say "$(grep 'Best mIoU' "$LOG/train3more.log" | tail -1)" \
    || say "FAILED: extended training."

  $PY -m iceroute.model.evaluate --config configs/train3_more.yaml \
      --split official_test --scenes 20 --skip-baseline \
      --out artifacts/eval_more > "$LOG/eval_more.log" 2>&1 \
    && say "More data: $(grep -E '^\s+mIoU' "$LOG/eval_more.log" | tail -1)" \
    || say "FAILED: extended evaluation."
else
  say "configs/split_extra.yaml missing: skipping the 'more data' branch."
fi

# --- 5. Model selection -----------------------------------------------------
step "5/6  Picking the model on the official test set"
$PY scripts/pick_model.py > "$LOG/pick_model.log" 2>&1 \
  && say "$(tail -3 "$LOG/pick_model.log")" \
  || say "FAILED: model selection. Keeping configs/train3.yaml."

BEST_CFG="$(cat artifacts/best_config.txt 2>/dev/null || echo configs/train3.yaml)"
say "Picked configuration: $BEST_CFG"

# --- 6. Demo rebuild --------------------------------------------------------
step "6/6  Regenerating demo, measurements and figures"
$PY -m iceroute.model.export --config "$BEST_CFG" > "$LOG/export_final.log" 2>&1 \
  && $PY -m iceroute.model.quantize --config "$BEST_CFG" --scenes 2 \
       >> "$LOG/export_final.log" 2>&1 \
  && say "$(grep 'size: fp32' "$LOG/export_final.log" | tail -1)" \
  || say "FAILED: export/quantization."

$PY -m iceroute.demo.precompute --train-config "$BEST_CFG" \
    > "$LOG/precompute_final.log" 2>&1 \
  && say "$(grep 'CHANGES with the vessel class' "$LOG/precompute_final.log" | tail -1)" \
  || say "FAILED: precompute."

$PY -m iceroute.report.metrics --train-config "$BEST_CFG" \
    > "$LOG/report_final.log" 2>&1 \
  && say "Downlink: $(grep -E 'Ratio|ratio|:1' "$LOG/report_final.log" | tail -1)" \
  || say "FAILED: downlink measurements."

$PY -m iceroute.report.figures >> "$LOG/report_final.log" 2>&1 || say "FAILED: figures."
$PY -m iceroute.report.dashboard >> "$LOG/report_final.log" 2>&1 || say "FAILED: dashboard."
$PY -m pytest tests/ -q > "$LOG/tests_final.log" 2>&1 \
  && say "Test: $(tail -1 "$LOG/tests_final.log")" \
  || say "TESTS FAILED: see $LOG/tests_final.log"

say ""
say "=== DONE ==="
say "Summary in $STATUS · early estimates vs. measurements in artifacts/estimates_vs_measurements.md"
