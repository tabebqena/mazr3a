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

## How to acquire (done once, on any machine with internet + Ultralytics + OpenVINO)

Because Frigate's bundled COCO model has no `fire`/`smoke` class, a dedicated checkpoint
is required. Recommended source: a **YOLOv8n fine-tuned on a fire + smoke dataset** (e.g.
a "fire and smoke detection" model on Roboflow Universe, or a maintained GitHub release —
verify its license and class order before use).

1. Export the checkpoint to ONNX, then convert to OpenVINO IR:

   ```bash
   # pip install ultralytics openvino openvino-dev
   yolo export model=best.pt format=onnx            # NMS-free end-to-end export
   mo --input_model best.onnx --output_dir ./openvino --compress_to_fp16
   ```

   The standalone watcher runs its own NMS, so the Frigate-specific NMS-free
   `[1,N,6]` tensor constraint does **not** apply — a standard YOLO export is fine.
   The decoder accepts a single output tensor shaped `[1, 4+nc, N]` or `[1, N, 4+nc]`
   with `cxcywh` boxes in input pixels (Ultralytics' default layout).

2. Place the artifacts here:

   ```bash
   mkdir -p models/fire
   cp ./openvino/best.xml ./openvino/best.bin models/fire/
   # write labelmap.txt matching the model's class order, e.g.:
   printf 'fire\nsmoke\n' > models/fire/labelmap.txt
   ```

3. **Record provenance below** (source URL, license, input size, class order) so the
   model can be reproduced or swapped later. The typical YOLOv8n input is 640x640; the
   watcher letterboxes every frame to the model's real input size automatically.

4. Deploy to the host: `scripts/deploy_firewatch.sh`, then confirm the service starts
   with `docker compose logs firewatch` showing `input 640x640 ... classes=[...]`.

## Provenance

| Item | Value |
|---|---|
| Source | _(to be filled in once a checkpoint is chosen)_ |
| License | _(to be filled in)_ |
| Input size | 640x640 (typical YOLOv8n) |
| Class order | `fire`, `smoke` (verify against `labelmap.txt`) |
| Acquired by | _(who / when)_ |
