#!/usr/bin/env bash
# ============================================================
# Export Ultralytics COCO YOLO candidates to the NMS-free ONNX
# files Frigate's OpenVINO detector (model_type: yolo-generic)
# expects at models/coco/ (yolo11s.onnx ACTIVE + yolo11n.onnx rollback +
# yolov8s.onnx retained alternative).
#
# No OpenVINO IR conversion is needed: Frigate 0.17.2 loads the
# .onnx directly and post-processes it (verified vs v0.17.2 source).
#
# Requires internet (ultralytics downloads the pretrained weights)
# and a python env with `ultralytics` installed. The workspace venv
# (.venv/) already has ultralytics 8.4.140; `onnx` is added if missing.
#
# Usage:
#   ./dev_scripts/prep_coco_model.sh [imgsz] [models]
#   models = comma-separated Ultralytics model names (default: all three)
# Examples:
#   ./dev_scripts/prep_coco_model.sh 640              # export all
#   ./dev_scripts/prep_coco_model.sh 640 yolo11s      # export only yolo11s
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models/coco"

IMGSZ="${1:-640}"
# Which Ultralytics models to export (comma-separated). Defaults to the full set
# kept in models/coco/. Pass a subset to add/refresh one model WITHOUT rewriting
# the other git-tracked .onnx files (avoids noisy binary diffs on re-export).
MODELS="${2:-yolo11n,yolov8s,yolo11s}"

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
echo "1) export ${MODELS} -> NMS-free ONNX @ ${IMGSZ}"
"$PY" - "$IMGSZ" "$WORK" "$DEST_DIR" "$MODELS" <<'PY'
import os, shutil, sys
from ultralytics import YOLO

imgsz, work, dest = int(sys.argv[1]), sys.argv[2], sys.argv[3]
names = [n.strip() for n in sys.argv[4].split(",") if n.strip()]
os.chdir(work)
for name in names:
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
echo "  - Files in ${DEST_DIR} are git-TRACKED (self-contained git deploy, 2026-09-06)."
echo "  - Commit any new export + push, then deploy_all.sh ships it (host git pull)."
echo "  - A models/coco/* change restarts the frigate service in deploy_all.sh."
echo "  - Host benchmark (container path /models/coco/); the live detector device"
echo "    is GPU (iGPU), so benchmark with -d GPU:"
echo "      docker exec frigate /openvino/benchmark_app -m /models/coco/yolo11s.onnx -d GPU -api sync"
echo "      docker exec frigate /openvino/benchmark_app -m /models/coco/yolo11n.onnx -d GPU -api sync"
echo "      docker exec frigate /openvino/benchmark_app -m /models/coco/yolov8s.onnx -d GPU -api sync"
