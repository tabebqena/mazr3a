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

### Option B — train your own YOLOv8n (free Google Colab, ~30-60 min)

Most reliable way to get a model whose class order you control. Grab a "fire and smoke"
**dataset** from Roboflow Universe (Dataset tab → Download → YOLOv8 → it gives you a
`roboflow` pip snippet with your dataset's key), then in Colab:

```python
!pip install -q ultralytics roboflow openvino

from roboflow import Roboflow
# Paste the dataset's download snippet here (it writes a data.yaml),
# e.g. rf = Roboflow(api_key="..."); project = rf.workspace("...").project("...")
# dataset = project.version(1).download("yolov8")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/datasets/<your-dataset>/data.yaml",
            epochs=60, imgsz=640, batch=16, patience=15)
# best.pt is saved under runs/detect/train/weights/best.pt - download it.
```

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
| Source | _(to be filled in once a checkpoint is chosen)_ |
| License | _(to be filled in)_ |
| Input size | 640x640 (typical YOLOv8n) |
| Class order | `fire`, `smoke` (verify against `labelmap.txt`) |
| Acquired by | _(who / when)_ |
