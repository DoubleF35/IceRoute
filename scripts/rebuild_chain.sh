#!/usr/bin/env bash
# Rebuilds every artefact on the model PICKED by scripts/pick_model.py.
#
# Needed after every model change. Every step that fails is logged and the
# chain carries on: better to finish with six steps out of eight and know which
# two are missing than to stop at the second and find everything half done.
#
# It does NOT pick the model: pick_model.py does that, and this chain only
# reads artifacts/best_config.txt. Keeping the two apart stops a routine rebuild
# from silently changing the model the demo runs on.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=".venv/bin/python"
LOG="artifacts/logs"
mkdir -p "$LOG"
STATUS="$LOG/rebuild_status.txt"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say ""; say "=== $* ==="; }

CFG="$(cat artifacts/best_config.txt 2>/dev/null || echo configs/train3.yaml)"
: > "$STATUS"
say "Rebuilding on the picked model: $CFG"

step "1/7  Evaluating on the official test set (overwrites artifacts/metrics.json)"
$PY -m iceroute.model.evaluate --config "$CFG" --split official_test --scenes 20 \
    --skip-baseline > "$LOG/eval_final.log" 2>&1 \
  && say "$(grep -E '^\s+mIoU' "$LOG/eval_final.log" | tail -1)" \
  || say "FAILED: evaluation."

step "2/7  Export ONNX"
$PY -m iceroute.model.export --config "$CFG" > "$LOG/export_final.log" 2>&1 \
  && say "export ok" || say "FAILED: export."

step "3/7  INT8 quantization"
$PY -m iceroute.model.quantize --config "$CFG" --scenes 2 >> "$LOG/export_final.log" 2>&1 \
  && say "$(grep 'size: fp32' "$LOG/export_final.log" | tail -1)" \
  || say "FAILED: quantization."

step "4/7  Demo precompute"
$PY -m iceroute.demo.precompute --train-config "$CFG" > "$LOG/precompute_final.log" 2>&1 \
  && say "$(grep 'CHANGES with the vessel class' "$LOG/precompute_final.log" | tail -1)" \
  || say "FAILED: precompute."

step "5/7  Downlink measurements and early estimates vs. measurements"
$PY -m iceroute.report.metrics --train-config "$CFG" > "$LOG/report_final.log" 2>&1 \
  && say "$(grep -E 'Ratio|ratio' "$LOG/report_final.log" | tail -1)" \
  || say "FAILED: downlink measurements."

step "6/7  Figures and dashboard"
$PY -m iceroute.report.figures >> "$LOG/report_final.log" 2>&1 \
  && say "figures ok" || say "FAILED: figures."
$PY -m iceroute.report.dashboard >> "$LOG/report_final.log" 2>&1 \
  && say "$(tail -1 "$LOG/report_final.log")" || say "FAILED: dashboard."

step "7/7  Test"
$PY -m pytest tests/ -q > "$LOG/tests_final.log" 2>&1 \
  && say "$(tail -1 "$LOG/tests_final.log")" \
  || say "TESTS FAILED: see $LOG/tests_final.log"

say ""
say "=== DONE ==="
