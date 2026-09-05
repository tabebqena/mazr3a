# Fire/smoke detection model — `models/fire/`

Holds the model used by the `firewatch` service (`scripts/firewatch.py`, see
[`plans/fire-detection.md`](../../plans/fire-detection.md)). The directory is mounted
read-only at `/models/fire` in the `firewatch` container.

## CHOSEN model (active)

[`SalahALHaismawi/yolov26-fire-detection`](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection)
— **YOLO26-S**, trained on 8,939 images / 100 epochs at 640 (Ultralytics), MIT license.
Classes (index order): `fire`(0), `other`(1), `smoke`(2) — `other` is a catch-all and is
**ignored** by firewatch; only `fire`/`smoke` can alert.

Author-reported metrics: mAP@50 **94.9**, mAP@50-95 **68.0**, precision **89.6**,
recall **88.8**.

The checkpoint is already downloaded to **`models/fire/best.pt`** in this workspace
(git-tracked, 20.3 MB, md5 `2fd972183c2ffec0d327ec534c119086`).

**Deployed + verified (v1, 2026-09-05):** the ACTIVE OpenVINO IR (`best.xml`/`best.bin`/
`labelmap.txt`) was generated from this `.pt` and `firewatch` is running on
`ssh.mazr3a.garden` — `--check` OK (output `[1,300,6]`, classes fire/other/smoke), `--dry-run`
polls all 9 cameras. Model is an **end-to-end YOLO26** (the decoder handles `[1,N,6]`); this
checkpoint is the ACTIVE **v1** — see [`VERSIONS.md`](VERSIONS.md) for v2 (fine-tuned on
Abonia `fire-8`, ~877 imgs) which is **not** yet promoted.

> **Still not proven on your cameras** - those are benchmark numbers on the author's own
> validation. Pilot with `--dry-run` on your day/night/IR, near/distance views and a test
> flame before trusting it as an alarm. If recall is weak on your scenes, fine-tune this
> checkpoint on ~100-200 frames captured from your cameras (transfer learning).

## Required files (after conversion)

| File | Purpose |
|---|---|
| `best.xml` | OpenVINO IR graph (the container looks for `best.xml`) |
| `best.bin`  | OpenVINO IR weights (same basename as the `.xml`) |
| `labelmap.txt` | one class per line, **index order = model class order** |
| `best.pt`   | source checkpoint (used only to (re)convert) |

The **ACTIVE set** (`best.xml`/`best.bin`/`best.pt`/`labelmap.txt`) is **tracked in
git** (2026-09-05, git-deploy decision): deploys ride `git pull`, so firewatch on
the host always matches the repo. Only `models/fire/versions/` (the versioned
archive) stays git-ignored; this README + [`VERSIONS.md`](VERSIONS.md) document it.

## Versioned checkpoints — `versions/`

Every checkpoint version is archived, self-described and immutable under the git-ignored
`models/fire/versions/` directory. Naming convention + full design:
[`plans/model-versioning.md`](../../plans/model-versioning.md); tracked registry with the
current/active version and per-version provenance: [`VERSIONS.md`](VERSIONS.md).

```
models/fire/
├── best.pt / best.xml / best.bin / labelmap.txt   # ACTIVE set (what firewatch loads)
└── versions/
    ├── v1-2026-09-05-hf-yolo26s-8939img/   # HF YOLO26-S baseline (active)
    │   ├── model.pt                        # archived checkpoint
    │   └── VERSION.json                    # provenance: source/dataset/metrics/md5
    └── v2-2026-09-05-hf-abonia877-ft5ep/   # fine-tune on Abonia fire-8 train (877 imgs)
        ├── model.pt
        └── VERSION.json
```

- **`versions/<dir>/model.pt`** — the archived checkpoint (never named `best.pt`, so a
  version folder can't be mistaken for the active model). Name format:
  `v<N>-<YYYY-MM-DD>-<slug>`, e.g. `v1-2026-09-05-hf-yolo26s-8939img`.
- **`versions/<dir>/VERSION.json`** — machine-readable provenance (class order, training
  set, metrics + eval split, parent, md5, status).
- The `models/fire/` root `best.*` files are just the **copy of the active version**;
  `versions/` is the canonical archive. Promoting a version refreshes the root copies.

**Promote a version to ACTIVE:**
```bash
./dev_scripts/promote_fire_model.sh v2-2026-09-05-hf-abonia877-ft5ep   # or unique prefix "v2"
```
It copies `model.pt` → `best.pt` (plus a bundled OpenVINO IR if present), prints the exact
`prep_fire_model.sh` command to regenerate the IR, and reminds you to deploy
(`./dev_scripts/deploy_all.sh` — full deploy, no subcommands) and verify
(`firewatch.py --check` / `--dry-run`) on the host.

## What the watcher expects from the exported model

- Ultralytics YOLO export, single output tensor shaped `[1, 4+nc, N]` or `[1, N, 4+nc]`,
  boxes `cxcywh` in input pixels, per-class scores 0..1. The watcher does its own NMS.
- Input 640x640; `firewatch.py` letterboxes every frame to the model's real input size.

## Convert `best.pt` → OpenVINO IR (one Colab run)

Run this cell in Google Colab (any runtime). It uses the local `best.pt` you upload, or
re-downloads it:

```python
!pip install -q ultralytics openvino
# Option 1: upload models/fire/best.pt from this workspace, OR:
# !wget -q -O best.pt "https://huggingface.co/SalahALHaismawi/yolov26-fire-detection/resolve/main/best.pt"

from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=640)   # YOLO26 needs a current ultralytics
!ovc best.onnx --output_model best                 # -> best.xml + best.bin

# 3 classes in model order: 0=fire 1=other 2=smoke. firewatch ignores 'other'.
open("labelmap.txt", "w").write("fire\nother\nsmoke\n")
import zipfile
with zipfile.ZipFile("fire_model.zip", "w") as z:
    for fn in ["best.xml", "best.bin", "labelmap.txt"]:
        z.write(fn)
print("Download fire_model.zip -> unzip its 3 files into models/fire/")
```

Then unzip `best.xml`, `best.bin`, `labelmap.txt` into `models/fire/` (the ACTIVE set is
git-tracked), commit them, and deploy:
```bash
git add models/fire/best.xml models/fire/best.bin models/fire/labelmap.txt
git commit -m "fire model: update ACTIVE OpenVINO IR"
./dev_scripts/deploy_all.sh            # full deploy (runs every step on the host)
```

**YOLO26 decode check:** after deploying, run
`docker compose exec firewatch python /scripts/firewatch.py --dry-run`. If it logs an
unexpected output shape, share the logged shape and the decoder in `scripts/firewatch.py`
will be adapted.

## Alternative (superseded)

`Abonia1/YOLOv8-Fire-and-Smoke-Detection` (GitHub) - YOLOv8s, unlicensed repo, classes
`Fire`/`default`/`smoke`, mAP@50 85.7 from its own `results.csv`. **Not used** - the HF
YOLO26 model above was chosen for its ~10x larger dataset, higher metrics, cleaner classes,
and MIT license. (Its weights were superseded; `models/fire/best.pt` is now the HF model.)

## Train-your-own fallback (if on-site recall is insufficient)

Fine-tune the chosen checkpoint on ~100-200 frames from your own cameras (best
generalization for your angles/lighting), or train `yolov8n` on a Roboflow fire/smoke
dataset. Use [`dev_scripts/prep_fire_model.sh`](../../dev_scripts/prep_fire_model.sh) to convert
any resulting `.pt` to this directory's OpenVINO IR format.

## Provenance

| Item | Value |
|---|---|
| Source (active) | `huggingface.co/SalahALHaismawi/yolov26-fire-detection` → `best.pt` (file `models/fire/best.pt`) |
| License | MIT (model) / CC BY 4.0 (underlying dataset) |
| Base/input size | YOLO26-S, 640x640 |
| Class order | `fire`(0), `other`(1), `smoke`(2) — `other` ignored by firewatch |
| Reported metrics | mAP@50 94.9 / mAP@50-95 68.0 / P 89.6 / R 88.8 (author-reported) |
| Acquired by | AI assistant, 2026-09-05 (`best.pt` in `models/fire/`) |
