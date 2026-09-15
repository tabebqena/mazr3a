# scratch-v3 — training record (configured, not yet trained)

Run directory: `model-training/scratch-model/scratch-v3/`

## Training path

- v3 = **class-expansion fine-tune**: adds `smoke` + `other` on top of v2's fire skill
  ([`plans/fire-model-smoke-class-training.md`](../../../plans/fire-model-smoke-class-training.md),
  [`plans/fire-model-v3-class-expansion.md`](../../../plans/fire-model-v3-class-expansion.md)).

## Original (base) model

- v2 `scratch-v1-dfire.pt` (md5 `3262b25be13b0888d21e7e96d16af20d`), fire-only `nc=1` → head rebuilt to `nc=3`.

## Classes

- `['fire', 'other', 'smoke']` — `nc=3`, production order fire(0)/other(1)/smoke(2)
  (drop-in with `models/fire/labelmap.txt`).

## Datasets (configured `SOURCES`)

| Source | Type | Role |
|---|---|---|
| FireViewer HF corpus (`fireviewer/fire-smoke-detection-corpus-v1`, excl. `alarmod`, train split capped 20,000 clips) | huggingface_fireviewer | train/val/test |
| Abonia `fire-8` (CC BY 4.0) | github_repo | train |
| SalahALHaismawi (`Fire Detection.v1i.yolov8`, bundled) | yolo_dir | train |
| negatives (domain) | yolo_dir | train (background) |
| cctv_test / cctv_emergency | yolo_dir | **test-only (never trained/augmented)** |

## Training parameters (configured, notebook 1.10.0)

| Parameter | Value |
|---|---|
| epochs / patience | 40 / 15 |
| batch / imgsz | 16 / 640 |
| freeze / lr0 | 10 / 0.001 (low-LR fine-tune) |
| dedup | isolation-only (`--no-train-self`), `MAX_BG_SHARE=0.60` |
| static aug | re-render train via `augment_fire_train.py` (rot ±15°, flip, brightness/contrast ±10%, hue/sat, noise) |
| online aug | degrees=15, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.1 |

## Status

- **NOT trained yet.** Notebook generated at `model-training/scratch-model/scratch-v3/fire-scratch-train-colab.ipynb`.

## Open decisions before training (see `plans/fire-model-v3-class-expansion.md`)

1. Audit fire vs smoke vs other vs background box counts; balance smoke if < ~2–3k boxes.
2. Define `other` tightly, or drop it and run a clean 2-class fire/smoke first.
3. Add a per-class validation gate: ship only if fire mAP does not drop and smoke AP improves.
