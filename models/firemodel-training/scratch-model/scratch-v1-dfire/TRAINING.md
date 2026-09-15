# scratch-v1-dfire — training record

Run directory: `models/firemodel-training/scratch-model/scratch-v1-dfire/`

## Training path

- v2 = **continued low-LR fine-tune of v1 on clean D-Fire**
  ([`plans/fire-model-scratch-dfire-continued.md`](../../../../plans/fire-model-scratch-dfire-continued.md)).
- Same generated notebook, with `BASE_MODEL = v1/scratch-v1.pt`.

## Original (base) model

- v1 `scratch-v1.pt` (md5 `6e3325cca6d07dc58028218b58d3e746`), fire-only `nc=1`.

## Classes

- `['fire']` — fire-only (`nc=1`, unchanged). D-Fire native `0=smoke,1=fire` was remapped
  `index_map {0: None, 1: 'fire'}` → **smoke boxes dropped**, smoke-only images became background.

## Datasets (clean D-Fire only, from the prep/dedup logs)

- Raw D-Fire (Kaggle `sayedgamal99/smoke-fire-detection-yolo`): **21,527** images —
  train 14,122 / val 3,099 / test 4,306; **fire boxes 14,692**; 15,705 background (smoke dropped).
- After dedup + balance (`MAX_BG_SHARE=0.60`): train **3,410** (1,364 fire-positive / 40%,
  2,046 background / 60%), val **2,399**, test **3,276** (2,675 bg + 601 fire-positive, 1,401 boxes).

## Training parameters (`args.yaml`)

| Parameter | Value |
|---|---|
| epochs / patience | 50 / 15 (early-stopped @ epoch 42) |
| batch / imgsz | 16 / 640 |
| freeze / lr0 | 10 / 0.001 (low-LR fine-tune) |
| optimizer / amp | auto / True |
| device | Tesla T4 |
| online aug | mosaic=1.0, scale=0.5, translate=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, erasing=0.4, degrees=0.0, mixup=0.0, copy_paste=0.0 |

## Result

- **mAP50 0.6643** / mAP50-95 0.329 / P 0.6782 / R 0.6153 (held-out clean D-Fire test).
- Checkpoint `scratch-v1-dfire.pt` = `weights/best.pt`, md5 `3262b25be13b0888d21e7e96d16af20d`, 19,153,242 B.
- Status: **ACTIVE fire-only winner** (base for v3's head expansion).

## Provenance

- Registry entry `scratch-v1-dfire` in [`registry.json`](../registry.json).
