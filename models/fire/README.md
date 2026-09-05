# Fire/smoke detection model — `models/fire/`

This directory holds the OpenVINO IR model used by the `firewatch` service
(`scripts/firewatch.py`, see [`plans/fire-detection.md`](../../plans/fire-detection.md)).
It is mounted read-only at `/models/fire` in the `firewatch` container.

## Required files

| File | Purpose |
|---|---|
| `best.xml` | OpenVINO IR graph (the container looks for `best.xml`) |
| `best.bin`  | OpenVINO IR weights (same basename as the `.xml`) |
| `labelmap.txt` | one class name per line, **index order = model class order**. If absent, `firewatch.py` assumes `fire`, `smoke` in that order |

The `.xml`/`.bin` are large binaries and are **git-ignored** (`models/fire/*.xml`,
`models/fire/*.bin`) — only this README and `labelmap.txt` are tracked. The model is
acquired once and deployed to the host via `scripts/deploy_firewatch.sh`; it is **not**
downloaded by the container at build/start (keeps the image small and the build offline).

## What the watcher expects from the model

- A **fire/smoke** detector (a COCO `person/car/...` model will NOT see fire).
- Any YOLO variant exported from Ultralytics is fine: single output tensor shaped
  `[1, 4+nc, N]` or `[1, N, 4+nc]`, boxes `cxcywh` in input pixels, per-class scores.
  The standalone watcher does its own NMS, so a standard `format=onnx` export works —
  the Frigate-specific NMS-free `[1,N,6]` constraint does **not** apply here.
- Typical input 640x640; `firewatch.py` letterboxes every frame to the model's real size.

---

## Where to download a fire/smoke checkpoint

Pick **either** Option A (download ready weights) **or** Option B (train a small one).
Both end with the same converter step that produces `best.xml` / `best.bin` /
`labelmap.txt` in this directory.

### Option A — download pre-trained weights

1. **Roboflow Universe** — https://universe.roboflow.com → search **"fire and smoke
   detection"** or **"fire detection"**. Prefer a project that shows a **trained model**;
   use its **Model** tab to download the trained checkpoint (usually `best.pt`), or the
   **Dataset** tab → "Download Dataset" → **YOLOv8** format if you choose to train it
   yourself (Option B). Note each project's license and class list (most are
   `fire`, `smoke`; some are fire-only).
2. **GitHub** — search **"fire smoke detection best.pt"** / **"yolo fire detection"**.
   Repos frequently attach a trained `best.pt` under **Releases → Assets**. Verify the
   model's class set matches what you write into `labelmap.txt`, and check its license.
3. Whatever you obtain, the checkpoint (`.pt` or `.onnx`) then goes through the
   converter below.

> I can't pin one canonical download URL for you because such public checkpoints move /
> get taken down; Roboflow Universe and GitHub Releases are the two reliable places to
> find an actively published fire/smoke `best.pt`.

### Option A2 — use a known pre-trained fire+smoke checkpoint (no training)

[`Abonia1/YOLOv8-Fire-and-Smoke-Detection`](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection)
ships a trained **YOLOv8s** checkpoint at `runs/detect/train/weights/best.pt` with classes
`Fire`, `default`, `smoke`. The `default` class is dataset noise - firewatch ignores it and
only `fire`/`smoke` can alert. A copy of the checkpoint is already at `models/fire/best.pt`
in this workspace (git-ignored) if you prefer to convert locally.

Quick **Colab convert** (downloads the checkpoint, exports at 640, converts to OpenVINO IR,
writes `labelmap.txt` with ALL 3 classes in model order - do not drop `default` or the
output columns shift):

```python
!pip install -q ultralytics openvino
!wget -q -O best.pt "https://raw.githubusercontent.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection/main/runs/detect/train/weights/best.pt"

from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=640)   # export for 640 inference
!ovc best.onnx --output_model best                 # -> best.xml + best.bin

# 3 classes in model order, lowercased: 0=fire 1=default 2=smoke.
# firewatch's allowed set {'fire'} / {'fire','smoke'} ignores 'default'.
open("labelmap.txt", "w").write("fire\ndefault\nsmoke\n")
import zipfile
with zipfile.ZipFile("fire_model.zip", "w") as z:
    for fn in ["best.xml", "best.bin", "labelmap.txt"]:
        z.write(fn)
print("Download fire_model.zip -> unzip its 3 files into models/fire/")
```

Then unzip the three files into `models/fire/`, deploy `bash scripts/deploy_firewatch.sh`,
and validate with a `--dry-run` pass on live frames.

Notes on this checkpoint:
- **Measured training quality** (from the repo's `runs/detect/train/results.csv`, validation
  = 48 images): precision **0.83**, recall **0.88**, mAP@50 **0.86**, mAP@50-95 **0.46** at
  epoch 24/25 (YOLOv8s, 878 train images). These are **macro averages over all 3 classes**
  (including the noisy `default`), so per-class fire/smoke accuracy is not reported - treat
  them as "moderate", not proven.
- **Pilot before trusting**: the source data is forest-fire-oriented, validation is only
  48 images, and no per-class metrics exist. Run `--dry-run` over your own cameras at day,
  night/IR, and distance before relying on alerts. If recall is weak on your scenes,
  fine-tune this checkpoint on ~100-200 frames captured from your cameras (transfer
  learning, ~10-20 min in Colab) - that is the recommended path if the pilot under-delivers.
- Trained at imgsz 800; exporting at 640 keeps CPU low and matches the 640x360 detect
  frames. Export at 800 instead for maximum small-flame fidelity (slightly higher CPU).
- The repo has **no license** (proprietary/default); the underlying dataset declares
  **CC BY 4.0**. Verify you are comfortable with this before production use.

### Option B — train your own YOLOv8n (free Google Colab, ~30-60 min)

Most reliable way to get a model whose class order you control. On the Roboflow page:
**Dataset → Download Dataset → YOLOv8 → Continue** (free account) → copy the generated
`roboflow` download snippet (it embeds your dataset key + the version you picked). Then
run this **whole cell** in a Colab notebook - it trains YOLOv8n, converts to OpenVINO IR,
writes `labelmap.txt` from the dataset's own `data.yaml` (class order always correct), and
zips the three files:

```python
!pip install -q ultralytics openvino roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")                  # free key: app.roboflow.com
project = rf.workspace("firedetection-sserj").project("fire_detection-uhbdr")
version = project.version(1)                           # use the version you chose
dataset = version.download("yolov8")                   # writes data.yaml + images

from ultralytics import YOLO
model = YOLO("yolov8n.pt")                             # nano: CPU-friendly
model.train(data=dataset.location + "/data.yaml",
            epochs=60, imgsz=640, batch=16, patience=15)

model.export(format="onnx", imgsz=640)                 # NMS-free ONNX
!ovc best.onnx --output_model best                     # -> best.xml + best.bin

import yaml, zipfile
names = yaml.safe_load(open(dataset.location + "/data.yaml"))["names"]
open("labelmap.txt", "w").write("\n".join(names) + "\n")
print("classes:", names)                               # e.g. ['fire'] or ['fire','smoke']
with zipfile.ZipFile("fire_model.zip", "w") as z:
    for fn in ["best.xml", "best.bin", "labelmap.txt"]:
        z.write(fn)
print("Download fire_model.zip -> unzip its 3 files into models/fire/")
```

After downloading `fire_model.zip`, unzip its three files into this workspace's
`models/fire/`, then `bash scripts/deploy_firewatch.sh`.

> If the dataset is single-class `fire`, the default `TRACK_SMOKE=false` in
> `config/firewatch.conf` is already correct. If it also has `smoke` and you want smoke
> alerts, set `TRACK_SMOKE=true` after deploying.

---

## Converter (one command) — then deploy

Run anywhere python + `ultralytics` + `openvino` are installed (locally or in Colab),
pointing at the checkpoint you downloaded/trained:

```bash
# fire/smoke two-class model (default class order fire,smoke):
bash scripts/prep_fire_model.sh ~/Downloads/best.pt 640 fire,smoke

# fire-only model:
bash scripts/prep_fire_model.sh ~/Downloads/best.pt 640 fire
```

The script exports `.pt` → ONNX → OpenVINO IR and installs the files here. Then:

```bash
bash scripts/deploy_firewatch.sh                     # upload + build + start on host
docker compose exec firewatch python /scripts/firewatch.py --dry-run   # live smoke test
```

If you cannot run the converter locally, either run it in Colab and download the three
files, or place a `best.pt`/`best.onnx` in this workspace and ask to have it converted.

---

## Provenance

| Item | Value |
|---|---|
| Source | `github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection` → `runs/detect/train/weights/best.pt` |
| License | Repo: none listed; underlying dataset: CC BY 4.0 (verify before production) |
| Base/input size | YOLOv8s, trained at 800; exported for inference at 640 |
| Class order | `fire`(0), `default`(1), `smoke`(2) — `default` is ignored by firewatch |
| Acquired by | AI assistant, 2026-09-05 (`best.pt` downloaded to `models/fire/`) |
