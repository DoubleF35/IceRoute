#!/usr/bin/env bash
# Download the demo bundle and the trained weights from a GitHub Release.
#
#   bash scripts/fetch_assets.sh [TAG]        (default: v1.0.0)
#
# After it, `make demo`, `make console` and `make demo-static` work without the
# 40 GB dataset and without a GPU. Uses the GitHub CLI when it is logged in, and
# plain curl otherwise.
set -euo pipefail

TAG="${1:-v1.0.0}"
REPO="DoubleF35/IceRoute"
BUNDLE="iceroute-demo-bundle-${TAG}.tar.gz"
MODELS=(unet3lov_int8.onnx unet3lov_fp32.onnx unet3lov_fp32.pt)

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch() {
  local name="$1"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$name" --dir "$tmp" --clobber
  else
    curl -fL --retry 3 -o "$tmp/$name" \
      "https://github.com/${REPO}/releases/download/${TAG}/${name}"
  fi
}

echo "Fetching ${BUNDLE} (${TAG})"
fetch "$BUNDLE"
fetch SHA256SUMS
for m in "${MODELS[@]}"; do
  echo "Fetching ${m}"
  fetch "$m"
done

( cd "$tmp" && sha256sum --check --ignore-missing SHA256SUMS )

tar -xzf "$tmp/$BUNDLE"          # -> data/demo/, artifacts/metrics.json
mkdir -p artifacts/models
for m in "${MODELS[@]}"; do mv "$tmp/$m" artifacts/models/; done
echo "configs/train3_lovasz.yaml" > artifacts/best_config.txt

echo
echo "  Done. Try:  make demo   or   make console"
echo
