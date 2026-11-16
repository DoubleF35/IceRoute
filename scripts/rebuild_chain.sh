#!/usr/bin/env bash
# Ricostruisce tutti gli artefatti sul modello SCELTO da scripts/pick_model.py.
#
# Serve dopo ogni cambio di modello. Ogni passo che fallisce viene registrato e
# la catena prosegue: meglio finire con sei passi su otto e sapere quali due
# mancano, che fermarsi al secondo e trovare tutto a metà.
#
# NON sceglie il modello: quello lo fa pick_model.py, e questa catena si limita
# a leggere artifacts/best_config.txt. Tenere separate le due cose evita che una
# ricostruzione di routine cambi di nascosto il modello della demo.

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
say "Ricostruzione sul modello scelto: $CFG"

step "1/7  Valuto sul test ufficiale (sovrascrive artifacts/metrics.json)"
$PY -m iceroute.model.evaluate --config "$CFG" --split test_ufficiale --scenes 20 \
    --skip-baseline > "$LOG/eval_final.log" 2>&1 \
  && say "$(grep -E '^\s+mIoU' "$LOG/eval_final.log" | tail -1)" \
  || say "FALLITA la valutazione."

step "2/7  Export ONNX"
$PY -m iceroute.model.export --config "$CFG" > "$LOG/export_final.log" 2>&1 \
  && say "export ok" || say "FALLITO l'export."

step "3/7  Quantizzazione INT8"
$PY -m iceroute.model.quantize --config "$CFG" --scenes 2 >> "$LOG/export_final.log" 2>&1 \
  && say "$(grep 'dimensioni:' "$LOG/export_final.log" | tail -1)" \
  || say "FALLITA la quantizzazione."

step "4/7  Precompute della demo"
$PY -m iceroute.demo.precompute --train-config "$CFG" > "$LOG/precompute_final.log" 2>&1 \
  && say "$(grep 'CAMBIA con la classe' "$LOG/precompute_final.log" | tail -1)" \
  || say "FALLITO il precompute."

step "5/7  Misure di downlink e verdetto sulle affermazioni"
$PY -m iceroute.report.metrics --train-config "$CFG" > "$LOG/report_final.log" 2>&1 \
  && say "$(grep -E 'compressione' "$LOG/report_final.log" | tail -1)" \
  || say "FALLITE le misure di downlink."

step "6/7  Figure e dashboard"
$PY -m iceroute.report.figures >> "$LOG/report_final.log" 2>&1 \
  && say "figure ok" || say "FALLITE le figure."
$PY -m iceroute.report.dashboard >> "$LOG/report_final.log" 2>&1 \
  && say "$(tail -1 "$LOG/report_final.log")" || say "FALLITA la dashboard."

step "7/7  Test"
$PY -m pytest tests/ -q > "$LOG/tests_final.log" 2>&1 \
  && say "$(tail -1 "$LOG/tests_final.log")" \
  || say "TEST FALLITI: vedi $LOG/tests_final.log"

say ""
say "=== FINE ==="
