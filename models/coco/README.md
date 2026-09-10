# Frigate COCO detector model — `models/coco/`

Active Frigate native-detector model for person/car/animal detection. Frigate
mounts `./models` → `/models` (read-only), so this directory is the in-container
path `/models/coco/` referenced by [`config/config.yaml`](../config/config.yaml)
(top-level `model:` `path`/`labelmap_path` and the detector `model_path`).

> Relocated here from `config/coco/` on 2026-09-06 so that `config/` holds only
> config. This mirrors the tracked `models/fire/` ACTIVE-set pattern.

## Files (all git-tracked — self-contained git deploy)

| File | Purpose |
|---|---|
| `yolo11s.onnx` | **ACTIVE** detector model — NMS-free ONNX @640, Ultralytics COCO (80 classes) |
| `yolo11n.onnx` | Rollback — the previous ACTIVE model (smaller/faster, lower accuracy) |
| `yolov8s.onnx` | Retained alternative — **not referenced** by `config.yaml`; dominated by `yolo11s` (lower mAP *and* more GFLOPs) |
| `labelmap.txt` | 80 lines, one class per line, index order = Ultralytics COCO class order |
| `README.md`   | this file (tracked) |

`config.yaml` loads `yolo11s.onnx` (the iGPU headroom winner — see
[`plans/better-coco-model-on-igpu.md`](../../plans/better-coco-model-on-igpu.md)),
running on the Intel UHD 630 iGPU (`device: GPU`). `yolo11n.onnx` is the one-line
rollback (revert the `path`/`model_path`); `yolov8s.onnx` is kept but is strictly
worse than `yolo11s` for this deployment.

Tracked in git: these files ride `git pull` in
[`dev_scripts/deploy_all.sh`](../../dev_scripts/deploy_all.sh), so a fresh host
clone is fully self-contained — **no scp of model files is needed anymore**. A
change to any file here marks `FRIGATE_CFG_CHANGED` in the deploy changed-set and
restarts the `frigate` service so the new model is reloaded.

## Regenerate / replace

Export new candidates with `dev_scripts/prep_coco_model.sh [imgsz] [models]`
(writes into this dir). Pass a comma-separated model list to export just one
without rewriting the other tracked `.onnx` files, e.g.:

```bash
./dev_scripts/prep_coco_model.sh 640 yolo11s
```

The script verifies `labelmap.txt` still has the 80 classes in Ultralytics COCO
order. Then `git add models/coco && git commit` and deploy. Swap the active model
by editing the `path`/`model_path` in `config/config.yaml`.

## Provenance

| Item | Value |
|---|---|
| Sources | Ultralytics `yolo11n.pt` / `yolo11s.pt` / `yolov8s.pt` (auto-downloaded by `ultralytics`) |
| License | AGPL-3.0 (Ultralytics pretrained weights) — fine for this self-hosted NVR |
| Input | 640×640, NCHW, 0–1 RGB |
| Classes | 80 (COCO), exact Ultralytics index order |
| Format | ONNX, NMS-free, `opset=12` |
| Exported | `yolo11n` + `yolov8s` 2026-09-05, `yolo11s` 2026-09-10 by `dev_scripts/prep_coco_model.sh` |
