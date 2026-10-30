# Fire/smoke detection model — `models/fire/`

Holds the model used by the `firewatch` service (`firewatch/firewatch.py`, see
[`plans/fire-detection.md`](../../plans/fire-detection.md)). The directory is mounted
read-only at `/models/fire` in the `firewatch` container.

## Active model

**ACTIVE = v7 (2026-10-29, promoted 2026-10-31)** — a fresh two-class
`fire` / `smoke` YOLO26-S trained on the revised `v02_01_01` dataset. It
replaces the three-class v4 model that served from 2026-09-07.

- Classes and labelmap order: **`fire`(0), `smoke`(1)**. There is **no `other` class**.
  `labelmap.txt` must contain exactly two lines.
- Active checkpoint: `models/fire/best.pt`, md5
  `082aa9676cf99c1b75e59e957483a2d5`, 60,233,571 B.
- OpenVINO IR: output `[1, 6, 8400]` raw per-class, input `640x640`.
- Validation (1,529 images): P **0.764**, R **0.665**, mAP@50 **0.753**,
  mAP@50-95 **0.445**; fire mAP@50 **0.735**, smoke **0.771**.
- The test split was not evaluated. The main weakness is tiny/small fire recall.
- On the 4,246-frame Firewatch evidence set, candidate raw best F1 is **43.7% @0.35**;
  with Firewatch motion at bar 0.50 it is **53.1%**, and the moderate retune proxy is
  **66.0%**. These are evidence-set results, not production FP guarantees.
- Active Firewatch retune: `MOTION_MARGIN=0.0`, inside-box `MOTION_NEAR_FRAC=0.02`,
  `MOTION_NEAR_MIN_PX=0` (pixel-count OR disabled), `MOTION_BONUS=0.20`,
  `SCORE_THRESHOLD=0.55`.

The previous active **v4** (`v4-2026-09-07-hf-dfire-cleanft52ep`) is superseded but
retained unchanged in the git-ignored archive for rollback:

```bash
./dev_scripts/deploy/promote_fire_model.sh v4-2026-09-07-hf-dfire-cleanft52ep
```

Full provenance and evidence caveats: [`versions/v7-.../README.md`](versions/v7-2026-10-29-v020101-yolo26s-14168img/README.md)
and [`VERSIONS.md`](VERSIONS.md).

## Required files (after conversion)

| File | Purpose |
|---|---|
| `best.xml` | OpenVINO IR graph (the container looks for `best.xml`) |
| `best.bin`  | OpenVINO IR weights (same basename as the `.xml`) |
| `labelmap.txt` | one class per line, **index order = model class order**; active v7 is exactly `fire`, `smoke` |
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
├── best.pt / best.xml / best.bin / labelmap.txt   # ACTIVE set = v7 (firewatch loads best.xml)
└── versions/
    ├── v7-2026-10-29-v020101-yolo26s-14168img/  # ACTIVE: 2-class v02_01_01 YOLO26s
    │   ├── model.pt                             # archived checkpoint
    │   ├── best.xml / best.bin / labelmap.txt   # bundled IR for deploy-ready promotion
    │   ├── VERSION.json                         # provenance: source/dataset/metrics/md5
    │   └── README.md
    ├── v4-2026-09-07-hf-dfire-cleanft52ep/      # previous ACTIVE, kept for rollback
    └── v6-2026-09-13-fireviewer-hfcorpus-ft15ep/ # unpromoted candidate
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
./dev_scripts/deploy/promote_fire_model.sh <version-dir|unique-prefix>
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
# Upload models/fire/best.pt from this workspace.

from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=640)   # YOLO26 needs a current ultralytics
!ovc best.onnx --output_model best                 # -> best.xml + best.bin

# 2 classes in model order: 0=fire 1=smoke. Must match the checkpoint.
open("labelmap.txt", "w").write("fire\nsmoke\n")
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
`docker compose exec firewatch python /firewatch/firewatch.py --dry-run`. If it logs an
unexpected output shape, share the logged shape and the decoder in `firewatch/firewatch.py`
will be adapted.

## Earlier baseline (superseded)

The original v1 model came from `huggingface.co/SalahALHaismawi/yolov26-fire-detection`
(MIT, 3 classes `fire`/`other`/`smoke`). It is retained in the archive for history but is
no longer active. The active v7 is a fresh 2-class `fire`/`smoke` training run.

## Train-your-own fallback (if on-site recall is insufficient)

Fine-tune the chosen checkpoint on ~100-200 frames from your own cameras (best
generalization for your angles/lighting), or train `yolov8n` on a Roboflow fire/smoke
dataset. Use [`dev_scripts/deploy/prep_fire_model.sh`](../../dev_scripts/deploy/prep_fire_model.sh) to convert
any resulting `.pt` to this directory's OpenVINO IR format.

## Provenance

| Item | Value |
|---|---|
| Source (active) | Fresh YOLO26-S run from COCO `yolo26s.pt` on `sources/training/v02_01_01` |
| License | See `fire_model_training/sources/training/SOURCES.md` |
| Base/input size | YOLO26-S, 640x640, 2 classes |
| Class order | `fire`(0), `smoke`(1) — no `other` class |
| Validation metrics | P 0.764 / R 0.665 / mAP@50 0.753 / mAP@50-95 0.445 |
| Promoted | 2026-10-31, replacing v4 |
