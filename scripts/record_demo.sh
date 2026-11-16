#!/usr/bin/env bash
# Records the backup video of the demo.
#
# If the laptop dies during a live demo, the walkthrough continues from a phone.
# That is why the video should be ready well in advance, not the night before:
# a video recorded the night before is a video nobody has watched.
#
# It records the SCREEN while the demo is really running, instead of stitching
# the PNGs into a sequence. The reason is that the video must be able to stand in
# for the demo: if it shows something the demo does not do, the substitution shows.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

OUT_DIR="artifacts/video"
OUT="${OUT_DIR}/iceroute_demo.mp4"
DURATION="${DURATION:-95}"        # 90 s plus a few seconds of margin
PORT="${PORT:-8501}"
SIZE="${SIZE:-1920x1080}"
VENV=".venv"

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- prerequisites ----------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || fail \
  "ffmpeg not installed. Install it with: sudo apt install ffmpeg
   The backup video is not optional: it is what a live demo falls back on."

[ -d "$VENV" ] || fail "$VENV missing. Run first: make setup"

[ -f "data/demo/index.json" ] || fail \
  "data/demo/index.json missing. Run first: make precompute
   The demo reads only precomputed files: without them there is nothing to record."

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  fail "No display: screen recording needs a graphical session."
fi

mkdir -p "$OUT_DIR"

# --- start the demo ---------------------------------------------------------
echo "Starting the demo on :${PORT}..."
"$VENV/bin/streamlit" run src/iceroute/demo/app.py \
  --server.port "$PORT" \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  > "$OUT_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!

# Always stop the server, even if ffmpeg fails halfway.
cleanup() {
  kill "$STREAMLIT_PID" 2>/dev/null || true
  wait "$STREAMLIT_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until it really answers instead of sleeping a fixed number of seconds:
# on an idle laptop it starts in one second, under load in ten.
echo -n "Waiting for the demo to answer"
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
    echo " — ready."
    break
  fi
  echo -n "."
  sleep 0.5
done
curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1 || fail \
  "The demo did not answer within 20 s. Log in $OUT_DIR/streamlit.log"

cat <<EOF

  The demo is running at http://localhost:${PORT}
  Open it full screen, then follow the walkthrough:

    0:00  first scene of the carousel (configs/demo.yaml)
    0:15  SAR layer       — "this is what the satellite sees"
    0:30  mask layer      — "this is computed on board, in 3.6 s"
    0:50  RIO layer       — "this is what POLARIS says: no passage here"
    1:10  route           — "the direct route is forbidden; IceRoute finds one"
    1:25  downlink card   — "only this comes down, not the scene"

  Recording starts in 5 seconds and lasts ${DURATION}s.
EOF
sleep 5

# --- recording --------------------------------------------------------------
# -draw_mouse 1: the pointer must be visible; it is what makes it obvious that
# someone is using the application rather than watching a rendering.
echo "Recording..."
ffmpeg -y -hide_banner -loglevel warning \
  -f x11grab -framerate 25 -video_size "$SIZE" -draw_mouse 1 \
  -i "${DISPLAY:-:0}" \
  -t "$DURATION" \
  -c:v libx264 -preset medium -crf 22 -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

[ -s "$OUT" ] || fail "ffmpeg did not produce a usable file."

SIZE_MB=$(du -m "$OUT" | cut -f1)
echo
echo "Video in $OUT (${SIZE_MB} MB, ${DURATION}s)"
echo
echo "To do now, not the night before:"
echo "  1. watch all of it, with the sound off, as a viewer would see it"
echo "  2. copy it to a phone"
echo "  3. copy it to a USB stick"
echo
echo "If the laptop dies during a live demo, the walkthrough continues from the phone."
