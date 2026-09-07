#!/usr/bin/env bash
# ============================================================
# Prepare a fire/smoke YOLO checkpoint into the OpenVINO IR files
# firewatch expects at models/fire/ (best.xml + best.bin + labelmap.txt).
#
# Works on ANY machine with python + ultralytics + openvino installed
# (local, or free Google Colab - see models/fire/README.md).
#
# Usage:
#   ./dev_scripts/prep_fire_model.sh /path/to/best.pt [imgsz] [classes]
# Examples:
#   ./dev_scripts/prep_fire_model.sh ~/Downloads/best.pt 640 "fire,smoke"
#   ./dev_scripts/prep_fire_model.sh ~/Downloads/best.pt 640 "fire"     # fire-only model
#
# The input must be a YOLO checkpoint whose classes are fire/smoke
# (a COCO model will NOT detect fire). Default class order is
# "fire,smoke" - pass a different order to match your checkpoint.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/fire"

PT="${1:?usage: prep_fire_model.sh <best.pt> [imgsz] [classes]}"
IMGSZ="${2:-640}"
CLASSES="${3:-fire,smoke}"

if [ ! -f "$PT" ]; then
  echo "ERROR: checkpoint not found: $PT" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=============================================================="
echo "1) export $PT -> ONNX (ultralytics)"
python3 - "$PT" "$IMGSZ" "$WORK" <<'PY'
import sys
from ultralytics import YOLO
pt, imgsz, work = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model = YOLO(pt)
out = model.export(format="onnx", imgsz=imgsz)  # NMS-free end-to-end export
print("onnx at:", out)
PY

# ultralytics writes best.onnx next to the .pt - move it into our work dir.
PT_DIR="$(dirname "$PT")"
PT_BASE="$(basename "$PT" .pt)"
if [ ! -f "${PT_DIR}/${PT_BASE}.onnx" ]; then
  echo "ERROR: expected ${PT_DIR}/${PT_BASE}.onnx - ultralytics export failed?" >&2
  exit 1
fi
mv "${PT_DIR}/${PT_BASE}.onnx" "$WORK/best.onnx"

echo "=============================================================="
echo "2) convert ONNX -> OpenVINO IR (best.xml + best.bin)"
# `ovc` ships with the `openvino` pip package (>=2023.2). Fall back to `mo`
# (openvino-dev) if ovc is unavailable.
if command -v ovc >/dev/null 2>&1; then
  (cd "$WORK" && ovc best.onnx --output_model best)
elif command -v mo >/dev/null 2>&1; then
  (cd "$WORK" && mo --input_model best.onnx --output_model best --compress_to_fp16)
else
  echo "ERROR: neither 'ovc' nor 'mo' found - pip install openvino" >&2
  exit 1
fi

echo "=============================================================="
echo "3) install into ${DEST_DIR}"
mkdir -p "$DEST_DIR"
cp "$WORK/best.xml" "$DEST_DIR/best.xml"
cp "$WORK/best.bin" "$DEST_DIR/best.bin"
IFS=',' read -r -a LABELS <<< "$CLASSES"
: > "$DEST_DIR/labelmap.txt"
for lab in "${LABELS[@]}"; do
  echo "$lab" >> "$DEST_DIR/labelmap.txt"
done

echo "=============================================================="
echo "DONE. Files in ${DEST_DIR}:"
ls -la "$DEST_DIR"
echo
echo "IMPORTANT:"
echo "  - labelmap.txt was written as: $(tr '\n' ',' < "$DEST_DIR/labelmap.txt")"
echo "    Its line order MUST equal the checkpoint's class index order,"
echo "    otherwise fire/smoke labels/box colors are swapped."
echo "  - Commit the ACTIVE set (it is git-tracked) and deploy with:"
echo "      ./dev_scripts/deploy_all.sh   # full deploy (no subcommands)"
echo "  - Sanity-check model load + one live frame:"
echo "      docker compose exec firewatch python /firewatch/firewatch.py --dry-run"
