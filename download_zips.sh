#!/usr/bin/env bash
# Downloads ALL dataset zips from ALL URLs (primary + fallbacks) for structure comparison.
# Skips downloads if the file already exists locally.
# Each URL gets its own output file so you can compare structures side-by-side.

set -uo pipefail

mkdir -p datasets/zip_compare

download() {
  local label="$1"
  local url="$2"
  local output="$3"

  if [[ -f "$output" ]]; then
    echo "[skip] $output already exists"
    return 0
  fi

  echo "[download] $label: $url -> $output"
  if curl -L --fail --progress-bar -o "$output" "$url"; then
    echo "[ok] $output ($(du -h "$output" | cut -f1))"
  else
    echo "[FAILED] $label: $url"
  fi
}

echo "=== plate ==="
download "plate-primary" \
  "https://copyparty.guilherme.zip/share/Brazil.Plates.Detector.v2i.yolo26?zip" \
  "datasets/zip_compare/plate-primary.zip"

download "plate-fallback" \
  "https://github.com/drawnator/yolo-license-plate-finetuning/releases/download/plate_dataset/Brazil.Plates.Detector.v2i.yolo26.zip" \
  "datasets/zip_compare/plate-fallback.zip"

echo ""
echo "=== face ==="
download "face-primary" \
  "https://copyparty.guilherme.zip/share/FACE.DETECTION.FYP.v1i.yolov12?zip" \
  "datasets/zip_compare/face-primary.zip"

download "face-fallback" \
  "https://github.com/drawnator/yolo-license-plate-finetuning/releases/download/face_dataset/FACE.DETECTION.FYP.v1i.yolov12.zip" \
  "datasets/zip_compare/face-fallback.zip"

echo ""
echo "=== ALPR ==="
download "ALPR-primary" \
  "https://copyparty.guilherme.zip/share/UFPR-ALPR?zip" \
  "datasets/zip_compare/ALPR-primary.zip"

download "ALPR-fallback" \
  "https://www.inf.ufpr.br/vri/databases/yj4Iu2-UFPR-ALPR.zip" \
  "datasets/zip_compare/ALPR-fallback.zip"

echo ""
echo "=== RODOSOL ==="
download "RODOSOL-primary" \
  "https://copyparty.guilherme.zip/share/RodoSol-ALPR?zip" \
  "datasets/zip_compare/RODOSOL-primary.zip"

download "RODOSOL-fallback" \
  "https://www.inf.ufpr.br/vri/databases/tbFcZE-RodoSol-ALPR.zip" \
  "datasets/zip_compare/RODOSOL-fallback.zip"

echo ""
echo "========================================"
echo "All downloads attempted. Inspect with:"
echo ""
echo "  for f in datasets/zip_compare/*.zip; do"
echo "    echo \"--- \$f ---\""
echo "    unzip -l \"\$f\" | head -30"
echo "    echo"
echo "  done"
echo "========================================"
