# `simplae-viewer.py` — YOLO dataset inspector & organizer

A single-file Flask web tool for **manually reviewing** a YOLO-format detection
dataset: it renders each image with its bounding-box labels drawn on, lets you page
through the split, and moves invalid image+label pairs into an `invalid-data`
quarantine folder with one keypress.

Part of the `fsyolo26` training campaign
([`models/firemodel-training/fsyolo26/`](../) — campaign intent in
[`target.txt`](../target.txt): YOLO26-S from scratch on `fire`/`other`/`smoke`).

## Purpose

- Browse the images of a YOLO split in sorted order.
- Visualise the ground-truth boxes (YOLO `class cx cy w h` normalized) drawn over the
  image.
- One-tap quarantine of a mislabelled image **and** its label file into a review
  folder, so the trainer never sees it.

This is the manual-review counterpart to the automated cleaning tools in the same
directory ([`clean.py`](clean.py), [`clean_cv.py`](clean_cv.py)).

## Where it fits

The tool currently targets the merged **Salah Haismawi + Abonia** dataset used by this
campaign, reviewing the YOLOv8-format `train` split:

| Role | Path (hard-coded) | Line |
|---|---|---|
| Input images | `model-training/sources/salah_haismawi_plus_abonia/dataset/Fire Detection.v1i.yolov8/train/images` | [`simplae-viewer.py`](simplae-viewer.py:15) |
| Input labels | `…/train/labels` | [`simplae-viewer.py`](simplae-viewer.py:16) |
| Quarantine target | `…/invalid-data/train/{images,labels}` | [`simplae-viewer.py`](simplae-viewer.py:17) |

## Configuration

Edit the constants at the top of the file before first run:

| Constant | Meaning | Line |
|---|---|---|
| `IMAGE_DIR` | folder scanned for images | [`simplae-viewer.py`](simplae-viewer.py:15) |
| `LABEL_DIR` | folder holding the matching `.txt` labels | [`simplae-viewer.py`](simplae-viewer.py:16) |
| `TARGET_DIR` | where moved image+label pairs land | [`simplae-viewer.py`](simplae-viewer.py:17) |
| `CLASS_NAMES` | class id → display name | [`simplae-viewer.py`](simplae-viewer.py:20) |
| `CLASS_COLORS` | BGR palette, indexed by class id (mod length) | [`simplae-viewer.py`](simplae-viewer.py:26) |

`get_image_paths()` returns images sorted by full path and filters to
`.png/.jpg/.jpeg/.webp` ([`simplae-viewer.py`](simplae-viewer.py:37)).

## Running

Dependencies: `flask` and `opencv-python` (imported as `cv2`).

```bash
pip install flask opencv-python
python3 models/firemodel-training/fsyolo26/utils/simplae-viewer.py
# Viewer running at http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in a browser. The server binds port **5000**
([`simplae-viewer.py`](simplae-viewer.py:224)).

## Web UI + keyboard controls

| Control | Action |
|---|---|
| `←` / `→` arrow keys | previous / next image |
| `M` | move current image + label to `TARGET_DIR` |
| `Previous` / `Next` buttons | same as arrow keys |
| `Move File & Label` button | same as `M` |

The keyboard handling lives in the page script
([`simplae-viewer.py`](simplae-viewer.py:114)); the counter and image swap are driven
client-side against `/info` and `/image/<idx>`.

## HTTP API

| Method | Route | Returns |
|---|---|---|
| `GET` | `/` | the HTML UI ([`simplae-viewer.py`](simplae-viewer.py:126)) |
| `GET` | `/info` | `{"total": N}` — number of images left ([`simplae-viewer.py`](simplae-viewer.py:130)) |
| `GET` | `/image/<idx>` | JPEG with ground-truth boxes drawn ([`simplae-viewer.py`](simplae-viewer.py:135)) |
| `POST` | `/move/<idx>` | `{"success": true, "filename": …}` or `{"success": false, "error": …}` ([`simplae-viewer.py`](simplae-viewer.py:193)) |

## Behaviour details

- **Box drawing** — [`get_image()`](simplae-viewer.py:135) converts YOLO normalized
  `cx cy w h` to pixel `xmin ymin xmax ymax` using the image width/height
  ([`simplae-viewer.py`](simplae-viewer.py:161)), then draws the rectangle and a class
  label with a filled background below the box's bottom-left corner
  ([`simplae-viewer.py`](simplae-viewer.py:171)). If the label would run off the
  bottom edge it is placed inside the box instead
  ([`simplae-viewer.py`](simplae-viewer.py:181)).
- **Colour** is assigned as `CLASS_COLORS[class_id % len(CLASS_COLORS)]`; unknown ids
  render as `ID: <n>` ([`simplae-viewer.py`](simplae-viewer.py:167)).
- **Move semantics** — [`move_file()`](simplae-viewer.py:193) creates
  `TARGET_DIR/images` and `TARGET_DIR/labels`, then `shutil.move`s the image and, if
  present, the same-basename label. A moved image disappears from the browser list on
  the next refresh.
- `TARGET_DIR` is created eagerly at import time
  ([`simplae-viewer.py`](simplae-viewer.py:35)), not lazily on first move.

## Caveats / limitations

- **Local-only, no auth** — the server exposes file moving over unauthenticated HTTP;
  run it only on the dev machine, not on the host.
- **Hard-coded absolute paths** — must be edited for a different dataset/split.
- **Client-side paging** — the "current index" is not persisted server-side; each
  `/image/<idx>` re-scans the directory, so the index shifts as files are moved out.
- Empty label lines are skipped; lines with fewer than 5 fields are ignored
  ([`simplae-viewer.py`](simplae-viewer.py:152)).
- This file is a helper in an otherwise untracked campaign tree, not a deploy
  artifact.

## Related utilities in `utils/`

| File | Purpose |
|---|---|
| [`clean.py`](clean.py) | automated 3-way dedup (train/valid/test overlap) + CleanVision CLI |
| [`clean_cv.py`](clean_cv.py) | CleanVision issue flags + quarantine dashboard |
| [`balance.py`](balance.py) | rebalance splits to train/valid/test percentages |
| [`count.py`](count.py) | count labelled vs unlabelled images per split |
| [`move.py`](move.py) | move images+labels with class remap / discard |
