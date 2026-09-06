# `models/coco/` — moved to `config/coco/`

The COCO ONNX detector artifacts (`yolo11n.onnx`, `yolov8s.onnx`, `labelmap.txt`)
moved to [`config/coco/`](../../config/coco/README.md) on 2026-09-06 and are now
**git-tracked**, so the git-based deploy (`dev_scripts/deploy_all.sh`) is fully
self-contained — Frigate loads them from `/config/coco/` with no manual scp.

This directory is kept only so older doc links to `models/coco/README.md` resolve;
it no longer holds model files. See [`config/coco/README.md`](../../config/coco/README.md).
