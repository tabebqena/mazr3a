# scratch-v1 — training record

Run directory: `model-training/scratch-model/scratch-v1/`

## Training path

- v1 = the **first from-scratch fire-only run** of the scratch campaign
  ([`plans/fire-model-scratch-colab.md`](../../plans/fire-model-scratch-colab.md)).
- Produced by the generated Colab notebook `fire-scratch-train-colab.ipynb` driving
  [`dev_scripts/colab/colab_train_scratch.py`](../../dev_scripts/colab/colab_train_scratch.py).

## Original (base) model

- `yolo11s.pt` — COCO-pretrained YOLO11-S (Ultralytics), 80 pretrained classes.
- Detection head **rebuilt from `data.yaml`** to `nc=1`.

## Classes

- `['fire']` — fire-only (`nc=1`). Smoke/other boxes were **dropped** (`None` in the class map), so
  smoke-only images became background negatives (reversible — the source data is untouched).

## Datasets (final clean pool, from `train.log`)

- `train`: **8,035** images (3,214 fire-positive + 4,821 background)
- `val`:   **7,280** images (774 fire-positive + 6,506 background)
- Sources: FireViewer HF corpus (excluding GPL `alarmod`) + Abonia `fire-8` + CCTV Emergency +
  domain negatives — all remapped to fire-only
  ([`plans/fire-model-scratch-colab.md`](../../plans/fire-model-scratch-colab.md)).

## Training parameters (`args.yaml`)

| Parameter | Value |
|---|---|
| epochs / patience | 100 / 20 |
| batch / imgsz | 16 / 640 |
| freeze / lr0 | 0 / 0.01 (from-scratch) |
| optimizer / amp | auto / True |
| device | NVIDIA A100-SXM4-40GB |
| online aug | mosaic=1.0, scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, erasing=0.4, degrees=0.0, mixup=0.0, copy_paste=0.0 |

## Result

- **mAP50 0.5535** / mAP50-95 0.2971 / P 0.585 / R 0.5316
- Checkpoint `scratch-v1.pt` = `weights/best.pt`, md5 `6e3325cca6d07dc58028218b58d3e746`, 19,154,394 B.
- Status: **superseded** by `scratch-v1-dfire` (kept as the v2 base).

## Provenance

- Registry entry `scratch-v1` in [`registry.json`](../registry.json).
