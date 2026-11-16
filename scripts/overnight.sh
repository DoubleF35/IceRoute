#!/usr/bin/env bash
# Catena non presidiata. Confronta tre modelli, poi ricostruisce la demo col migliore.
#
# Ordine, e perche' e' questo:
#   1. attende la Lovasz (gia' avviata) e la valuta sul test ufficiale
#   2. attende il download delle 90 scene aggiuntive
#   3. estrae le patch delle nuove e riaddestra il baseline con piu' dati
#   4. valuta anche quello
#   5. sceglie il modello migliore SUL TEST UFFICIALE, non su validazione
#   6. rigenera precompute, misure di downlink, figure, dashboard
#
# Ogni passo che fallisce viene registrato e la catena prosegue: meglio finire
# con quattro passi su sei che fermarsi al secondo e trovare tutto fermo.
#
# Non tocca configs/split.yaml (SPEC §5.3): val e demo restano identiche, quindi
# i tre modelli sono confrontabili.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=".venv/bin/python"
LOG="artifacts/logs"
mkdir -p "$LOG"
STATUS="$LOG/overnight_status.txt"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
step() { say ""; say "=== $* ==="; }

# Un job e' vivo solo se esiste un processo PYTHON che matcha. Un guscio di
# shell che il pattern lo *nomina* soltanto -- un wrapper, un tail, un editor
# aperto sul file -- non e' il job.
#
# Questa distinzione e' costata quattro ore la notte del 5 settembre. Il
# training Lovasz era stato lanciato da un `bash -c` che, per aspettare la GPU,
# girava su `pgrep -f "iceroute.model"`: la propria riga di comando conteneva
# "iceroute.model.train", quindi matchava se stesso e non usciva mai. Il
# training non parti'. Questa catena poi aspetto' `train3_lovasz`, trovo' quel
# guscio bloccato, e resto' ferma fino al tetto delle 4 ore.
#
# `pgrep -af` stampa "PID riga-di-comando": il secondo campo e' l'eseguibile.
running() {
  pgrep -af "$1" 2>/dev/null | awk '$2 ~ /python/ {found = 1} END {exit !found}'
}

# Aspetta che un job sparisca, con un tetto: se qualcosa si impianta, la catena
# prosegue invece di restare appesa fino al mattino.
wait_for() {
  local pattern="$1" limit="${2:-21600}" waited=0
  running "$pattern" || { say "Nessun processo python per '$pattern': non aspetto."; return 0; }
  say "In attesa di: $(pgrep -af "$pattern" | awk '$2 ~ /python/ {print $1}' | tr '\n' ' ')"
  while running "$pattern"; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$limit" ]; then
      say "TIMEOUT dopo ${limit}s su '$pattern': proseguo."
      return 1
    fi
  done
  return 0
}

: > "$STATUS"
say "Avvio catena non presidiata."

# --- 1. Lovasz --------------------------------------------------------------
step "1/6  Attendo il training Lovasz"
wait_for "train3_lovasz" 14400
if grep -q "Miglior mIoU" "$LOG/train3lov.log" 2>/dev/null; then
  say "$(grep 'Miglior mIoU' "$LOG/train3lov.log" | tail -1)"
  step "1b   Valuto Lovasz sul test ufficiale"
  $PY -m iceroute.model.evaluate --config configs/train3_lovasz.yaml \
      --split test_ufficiale --scenes 20 --skip-baseline \
      --out artifacts/eval_lovasz > "$LOG/eval_lovasz.log" 2>&1 \
    && say "Lovasz: $(grep -E '^\s+mIoU' "$LOG/eval_lovasz.log" | tail -1)" \
    || say "FALLITA la valutazione Lovasz."
else
  say "Il training Lovasz non ha prodotto un risultato."
fi

# --- 2. Download ------------------------------------------------------------
step "2/6  Attendo il download delle scene aggiuntive"
wait_for "download_subset" 21600
say "Scene in data/raw/train: $(ls data/raw/train/*.nc 2>/dev/null | wc -l)"

# --- 3. Patch + training con piu' dati -------------------------------------
if [ -f configs/split_extra.yaml ]; then
  step "3/6  Estraggo le patch delle scene nuove"
  $PY -m iceroute.data.patches --config configs/data.yaml \
      > "$LOG/patches_extra.log" 2>&1 \
    && say "$(grep 'Patch totali' "$LOG/patches_extra.log" | tail -1)" \
    || say "FALLITA l'estrazione delle patch."

  step "4/6  Riaddestro il baseline con piu' dati"
  $PY -m iceroute.model.train --config configs/train3_more.yaml \
      > "$LOG/train3more.log" 2>&1 \
    && say "$(grep 'Miglior mIoU' "$LOG/train3more.log" | tail -1)" \
    || say "FALLITO il training esteso."

  $PY -m iceroute.model.evaluate --config configs/train3_more.yaml \
      --split test_ufficiale --scenes 20 --skip-baseline \
      --out artifacts/eval_more > "$LOG/eval_more.log" 2>&1 \
    && say "Piu' dati: $(grep -E '^\s+mIoU' "$LOG/eval_more.log" | tail -1)" \
    || say "FALLITA la valutazione estesa."
else
  say "configs/split_extra.yaml assente: salto il ramo 'piu' dati'."
fi

# --- 5. Scelta del modello --------------------------------------------------
step "5/6  Scelgo il modello sul test ufficiale"
$PY scripts/pick_model.py > "$LOG/pick_model.log" 2>&1 \
  && say "$(tail -3 "$LOG/pick_model.log")" \
  || say "FALLITA la scelta del modello: resta configs/train3.yaml."

BEST_CFG="$(cat artifacts/best_config.txt 2>/dev/null || echo configs/train3.yaml)"
say "Configurazione scelta: $BEST_CFG"

# --- 6. Ricostruzione della demo -------------------------------------------
step "6/6  Rigenero demo, misure e figure"
$PY -m iceroute.model.export --config "$BEST_CFG" > "$LOG/export_final.log" 2>&1 \
  && $PY -m iceroute.model.quantize --config "$BEST_CFG" --scenes 2 \
       >> "$LOG/export_final.log" 2>&1 \
  && say "$(grep 'dimensioni:' "$LOG/export_final.log" | tail -1)" \
  || say "FALLITO export/quantizzazione."

$PY -m iceroute.demo.precompute --train-config "$BEST_CFG" \
    > "$LOG/precompute_final.log" 2>&1 \
  && say "$(grep 'CAMBIA con la classe' "$LOG/precompute_final.log" | tail -1)" \
  || say "FALLITO precompute."

$PY -m iceroute.report.metrics --train-config "$BEST_CFG" \
    > "$LOG/report_final.log" 2>&1 \
  && say "Downlink: $(grep -E 'Rapporto|:1' "$LOG/report_final.log" | tail -1)" \
  || say "FALLITE le misure di downlink."

$PY -m iceroute.report.figures >> "$LOG/report_final.log" 2>&1 || say "FALLITE le figure."
$PY -m iceroute.report.dashboard >> "$LOG/report_final.log" 2>&1 || say "FALLITA la dashboard."
$PY -m pytest tests/ -q > "$LOG/tests_final.log" 2>&1 \
  && say "Test: $(tail -1 "$LOG/tests_final.log")" \
  || say "TEST FALLITI: vedi $LOG/tests_final.log"

say ""
say "=== FINE ==="
say "Riepilogo in $STATUS · verdetto sulle affermazioni in artifacts/claims.md"
