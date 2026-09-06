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
| `yolo11n.onnx` | **ACTIVE** detector model — NMS-free ONNX @640, Ultralytics COCO (80 classes) |
| `yolov8s.onnx` | Accuracy-first candidate, kept for future benchmarking — **not referenced** by `config.yaml` |
| `labelmap.txt` | 80 lines, one class per line, index order = Ultralytics COCO class order |
| `README.md`   | this file (tracked) |

`config.yaml` loads `yolo11n.onnx` (the CPU benchmark winner — see
[`plans/replace-coco-with-yolo-on-igpu.md`](../../plans/replace-coco-with-yolo-on-igpu.md));
`yolov8s.onnx` is an unused alternative (INT8 candidate if accuracy is ever wanted).

Tracked in git: these files ride `git pull` in
[`dev_scripts/deploy_all.sh`](../../dev_scripts/deploy_all.sh), so a fresh host
clone is fully self-contained — **no scp of model files is needed anymore**. A
change to any file here marks `FRIGATE_CFG_CHANGED` in the deploy changed-set and
restarts the `frigate` service so the new model is reloaded.

## Regenerate / replace

Export new candidates with `dev_scripts/prep_coco_model.sh` (writes into this
dir), verify `labelmap.txt` still has the 80 classes in Ultralytics COCO order,
then `git add models/coco && git commit` and deploy. Swap the active model by
editing the `path`/`model_path` in `config/config.yaml`.

## Provenance

| Item | Value |
|---|---|
| Sources | Ultralytics `yolo11n.pt` / `yolov8s.pt` (auto-downloaded by `ultralytics`) |
| License | AGPL-3.0 (Ultralytics pretrained weights) — fine for this self-hosted NVR |
| Input | 640×640, NCHW, 0–1 RGB |
| Classes | 80 (COCO), exact Ultralytics index order |
| Format | ONNX, NMS-free, `opset=12` |
| Exported | 2026-09-05 by `dev_scripts/prep_coco_model.sh` |
