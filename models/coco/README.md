# COCO replacement model — `models/coco/`

Staging dir for the Ultralytics **NMS-free ONNX** COCO detection candidates that will replace
Frigate's bundled `ssdlite_mobilenet_v2` as the native detector (see
[`plans/replace-coco-with-yolo-on-igpu.md`](../../plans/replace-coco-with-yolo-on-igpu.md)).

**Deployment location on the host is `config/coco` → `/config/coco` in-container** (NOT
`/models/coco`): the `ai` deploy user cannot write `/home/dr/frigate/models` (dr-owned), but
the world-writable `config/` dir is deployable via the same `.new`+`mv` trick as `config.yaml`,
and it matches Frigate's own convention (the image already keeps a root-owned `model_cache/`
under `/config`). These `models/coco/` files are the local source that gets scp'd to
`config/coco/`.

## CHOSEN model (active)

> **Pending the host benchmark gate** — see the plan §Results. Either:
> - `YOLO11n` @ 640 (safe default, ~2.6M params), or
> - `YOLOv8s` @ 640 (accuracy-first, ~11.2M params; INT8 variant if CPU budget requires).

Both are pretrained Ultralytics **COCO (80 classes)** checkpoints, exported to NMS-free ONNX —
the exact format Frigate 0.17.2's OpenVINO detector parses with `model_type: yolo-generic`
(`post_process_yolo` decodes raw `cxcywh` + class scores and applies its own NMS).

## Required files (after Step 2)

| File | Purpose |
|---|---|
| `<winner>.onnx` | NMS-free ONNX detection model (Frigate reads it directly; no IR conversion) |
| `labelmap.txt` | 80 lines, one class per line, **index order = Ultralytics COCO class order** |
| `README.md` | this file (tracked) |

`*.onnx` / `*.pt` are large and **git-ignored** (`models/coco/*.onnx`, `models/coco/*.pt`) —
only this README and `labelmap.txt` are tracked.

> `labelmap.txt` order is critical: line *i* must equal the model's class index *i*. The 80
> Ultralytics COCO names are: `person`, `bicycle`, `car`, ... `toothbrush` (see the file).
> A wrong index silently swaps labels/box colors.

> Container paths used in `config/config.yaml` are `/config/coco/<winner>.onnx` and
> `/config/coco/labelmap.txt` (the `config/` dir is already mounted at `/config`).

## License

YOLO11n and YOLOv8n/s Ultralytics **pretrained weights are AGPL-3.0**. This is fine for the
self-hosted NVR use here, but if a permissive license is required, substitute an equivalent
COCO checkpoint (e.g. a YOLO model with an Apache/MIT license) and re-run the export.

## Export the candidates (regenerate)

Use the workspace venv (`.venv/`, Python 3.11, `ultralytics 8.4.140` already present) or any
machine with `ultralytics` + internet:

```bash
# from the workspace root
. .venv/bin/activate
python3 -m pip install -q onnx            # only if missing
bash scripts/prep_coco_model.sh 640       # exports BOTH yolo11n + yolov8s NMS-free ONNX @640
```

The script runs a class-order sanity check (`person car dog horse sheep cow`).

## What Frigate expects (verified against v0.17.2 source)

- Config schema: model config lives in the **top-level `model:` block**; per-detector `model:`
  is discarded; the detector's flat `model_path:` overrides only the path.
- `model_type: yolo-generic` routes to `post_process_yolo` → NMS-free ONNX output
  (`[1, 4+nc, N]` or `[1, N, 4+nc]`, raw `cxcywh` + class scores) is accepted; Frigate runs
  its own NMS.
- Input: `input_tensor: nchw`, `input_dtype: float` (Frigate divides by 255 → 0–1 RGB), and
  default `input_pixel_format: rgb` — exactly Ultralytics preprocessing.

## Provenance

| Item | Value |
|---|---|
| Sources | Ultralytics `yolo11n.pt` and `yolov8s.pt` (auto-downloaded by `ultralytics`) |
| License | AGPL-3.0 (Ultralytics pretrained weights) |
| Input | 640×640, NCHW, 0–1 RGB |
| Classes | 80 (COCO), exact Ultralytics index order |
| Format | ONNX, NMS-free, `opset=12` |
| Acquired by | AI assistant, 2026-09-05 |

## Benchmark result

> To be filled after the host benchmark gate in the plan (Step 3 / §Results).
