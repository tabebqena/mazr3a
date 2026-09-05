#!/usr/bin/env bash
# ============================================================
# Export Ultralytics COCO YOLO candidates to the NMS-free ONNX
# files Frigate's OpenVINO detector (model_type: yolo-generic)
# expects at models/coco/ (yolo11n.onnx + yolov8s.onnx).
#
# No OpenVINO IR conversion is needed: Frigate 0.17.2 loads the
# .onnx directly and post-processes it (verified vs v0.17.2 source).
#
# Requires internet (ultralytics downloads the pretrained weights)
# and a python env with `ultralytics` installed. The workspace venv
# (.venv/) already has ultralytics 8.4.140; `onnx` is added if missing.
#
# Usage:
#   ./scripts/prep_coco_model.sh [imgsz]
# Example:
#   ./scripts/prep_coco_model.sh 640
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/coco"

IMGSZ="${1:-640}"

# Prefer the workspace venv if it has ultralytics, else any python3.
PY="${ROOT_DIR}/.venv/bin/python"
if [ ! -x "$PY" ]; then PY=python3; fi
if ! "$PY" -c "import ultralytics" >/dev/null 2>&1; then
  echo "ERROR: ultralytics not importable by $PY. Install it, e.g.:" >&2
  echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -q ultralytics" >&2
  exit 1
fi
if ! "$PY" -c "import onnx" >/dev/null 2>&1; then
  echo "onnx missing - installing into $($PY -c 'import sys;print(sys.prefix)') ..."
  "$PY" -m pip install -q onnx
fi

mkdir -p "$DEST_DIR"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=============================================================="
echo "1) export yolo11n + yolov8s -> NMS-free ONNX @ ${IMGSZ}"
"$PY" - "$IMGSZ" "$WORK" "$DEST_DIR" <<'PY'
import os, shutil, sys
from ultralytics import YOLO

imgsz, work, dest = int(sys.argv[1]), sys.argv[2], sys.argv[3]
os.chdir(work)
for name in ["yolo11n", "yolov8s"]:
    model = YOLO(f"{name}.pt")          # downloads pretrained weights on first use
    names = model.names
    chk = [names[0], names[2], names[16], names[17], names[18], names[19]]
    out = model.export(format="onnx", imgsz=imgsz, opset=12)  # NMS-free by default
    # work dir may be on a different filesystem than models/coco -> use shutil.move
    shutil.move(out, os.path.join(dest, f"{name}.onnx"))
    print(f"{name}: exported; class check: {' '.join(chk)}")
PY

echo "=============================================================="
echo "2) verify labelmap (must be 80 lines, Ultralytics COCO order)"
LABELMAP="$DEST_DIR/labelmap.txt"
if [ ! -f "$LABELMAP" ]; then
  echo "ERROR: $LABELMAP missing - restore it from git (models/coco/labelmap.txt)." >&2
  exit 1
fi
N=$(wc -l < "$LABELMAP")
echo "labelmap lines: $N"
if [ "$N" -ne 80 ]; then
  echo "ERROR: expected 80 classes, got $N. labelmap order must match the model." >&2
  exit 1
fi
echo "labelmap head: $(head -1 "$LABELMAP") ... tail: $(tail -1 "$LABELMAP")"

echo "=============================================================="
echo "DONE. Files in ${DEST_DIR}:"
ls -la "$DEST_DIR"
echo
echo "IMPORTANT:"
echo "  - Both ONNX files are git-ignored (models/coco/*.onnx)."
echo "  - Benchmark on the host before committing, then set the winner in"
echo "    config/config.yaml top-level model: block (model_type: yolo-generic)."
echo "  - Host benchmark:"
echo "      docker exec frigate /openvino/benchmark_app -m /models/coco/yolo11n.onnx -d CPU -api sync"
echo "      docker exec frigate /openvino/benchmark_app -m /models/coco/yolov8s.onnx -d CPU -api sync"
