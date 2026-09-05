#!/usr/bin/env bash
# Registra il video di riserva della demo. SPEC §9.5.
#
# Se il portatile muore all'OGR, il pitch continua dal telefono. Per questo la
# SPEC lo vuole pronto entro il 1 ottobre, non il 6: un video registrato la sera
# prima e' un video che nessuno ha guardato.
#
# Registra lo SCHERMO mentre la demo gira davvero, invece di montare le PNG in
# una sequenza. Il motivo e' che il video deve poter sostituire la demo: se
# mostra qualcosa che la demo non fa, la sostituzione si vede.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT_DIR="artifacts/video"
OUT="${OUT_DIR}/iceroute_demo.mp4"
DURATION="${DURATION:-95}"        # 90 s piu' qualche secondo di margine
PORT="${PORT:-8501}"
SIZE="${SIZE:-1920x1080}"
VENV=".venv"

fail() { echo "ERRORE: $*" >&2; exit 1; }

# --- prerequisiti -----------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || fail \
  "ffmpeg non installato. Serve: sudo apt install ffmpeg
   La SPEC §11.1 elenca il video di riserva fra le cose NON tagliabili."

[ -d "$VENV" ] || fail "$VENV assente. Lancia prima: make setup"

[ -f "data/demo/index.json" ] || fail \
  "data/demo/index.json assente. Lancia prima: make precompute
   La demo legge solo file precomputati: senza, non c'e' niente da registrare."

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  fail "Nessun display: la registrazione dello schermo richiede una sessione grafica."
fi

mkdir -p "$OUT_DIR"

# --- avvio della demo -------------------------------------------------------
echo "Avvio della demo su :${PORT}..."
"$VENV/bin/streamlit" run src/iceroute/demo/app.py \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  > "$OUT_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

# Ferma sempre il server, anche se ffmpeg fallisce a meta'.
cleanup() {
  kill "$STREAMLIT_PID" 2>/dev/null || true
  wait "$STREAMLIT_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Attende che risponda davvero invece di dormire un numero fisso di secondi:
# su un portatile scarico parte in un secondo, sotto carico in dieci.
echo -n "Attendo che la demo risponda"
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
    echo " — pronta."
    break
  fi
  echo -n "."
  sleep 0.5
done
curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1 || fail \
  "La demo non ha risposto entro 20 s. Log in $OUT_DIR/streamlit.log"

cat <<EOF

  La demo gira su http://localhost:${PORT}
  Aprila a schermo intero, poi percorri il copione:

    0:00  scena con il ghiaccio che blocca la diretta (20210118T075356)
    0:15  livello SAR      — "questo e' cio' che vede il satellite"
    0:30  livello maschera — "questo lo calcola a bordo, in 3,6 s"
    0:50  livello RIO      — "questo dice POLARIS: qui non si passa"
    1:10  rotta            — "la diretta e' vietata, IceRoute ne trova una"
    1:25  card downlink    — "scende solo questo, non la scena"

  La registrazione parte fra 5 secondi e dura ${DURATION}s.
EOF
sleep 5

# --- registrazione ----------------------------------------------------------
# -draw_mouse 1: il puntatore va visto, e' cio' che rende evidente che qualcuno
# sta usando l'applicazione invece di guardare un rendering.
echo "Registrazione..."
ffmpeg -y -hide_banner -loglevel warning \
  -f x11grab -framerate 25 -video_size "$SIZE" -draw_mouse 1 \
  -i "${DISPLAY:-:0}" \
  -t "$DURATION" \
  -c:v libx264 -preset medium -crf 22 -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

[ -s "$OUT" ] || fail "ffmpeg non ha prodotto un file utilizzabile."

SIZE_MB=$(du -m "$OUT" | cut -f1)
echo
echo "Video in $OUT (${SIZE_MB} MB, ${DURATION}s)"
echo
echo "Da fare adesso, non il 6 ottobre:"
echo "  1. guardalo tutto, con l'audio spento, come lo vedrebbe la sala"
echo "  2. copialo sul telefono"
echo "  3. copialo su una chiavetta"
echo
echo "SPEC §9.5: se il portatile muore all'OGR, il pitch continua dal telefono."
