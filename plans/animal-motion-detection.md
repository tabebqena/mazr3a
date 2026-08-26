# Animal Motion Detection — Frigate NVR

Goal: detect and track animals on all 10 cameras, label them in the Frigate UI, and
publish standard MQTT events (no extra alerts/logging in this phase). Two tracks:

1. **Part 1 (immediate, config-only):** enable the built-in COCO animal classes that
   the bundled OpenVINO `ssdlite_mobilenet_v2` model already recognizes.
2. **Part 2 (custom model):** train a YOLOv8 model for animals *not* in COCO
   (`goat`, `camel`, `donkey`, `fox`, `wolf`) and use it to replace the default
   detector so all desired classes come from one model.

---

## Context

| Item | Value |
|---|---|
| Stack | Frigate 0.17 + Mosquitto (Docker Compose), see [`docker-compose.yml`](../docker-compose.yml) |
| Active config | [`config/config.yaml`](../config/config.yaml) (Frigate 0.17 ignores `frigate.yml`) |
| Detector | OpenVINO CPU, bundled model `ssdlite_mobilenet_v2.xml`, labelmap `/labelmap.txt` |
| Currently tracked | `person`, `car` only |
| MQTT | `frigate/events` already published; no changes needed for this task |
| Host | Debian, Intel i7-9700 (8 cores), 7.5 GiB RAM — CPU/RAM constrained |
| Cameras | 10, detect substreams only (640x360 @ 5 fps), record disabled |

### COCO classes available in the bundled model (no retraining needed)
`bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`.

### Missing classes (need custom model)
`goat`, `camel`, `donkey`, `fox`, `wolf` — the farm-relevant animals not in COCO.

---

## Design Decision: one combined model, not two detectors

Frigate assigns **one detector per camera** and cannot merge outputs of two models on
the same camera. Running two detectors would also roughly double CPU load on an
already tight i7-9700. Therefore:

- **Part 2 trains a single YOLOv8 model** whose labelmap is the *union* of the custom
  animals plus `person`, `car`, and the COCO animals we want. This model **replaces**
  the bundled COCO model as the sole detector.
- Until the custom model is ready, Part 1 keeps the bundled COCO model and simply adds
  animal classes to `objects.track`.

---

## Part 1 — Enable built-in COCO animal classes (config only)

### 1.1 Edit [`config/config.yaml`](../config/config.yaml:55)

Add the farm-relevant COCO animal classes to `objects.track` and per-class filters:

```yaml
objects:
  track:
    - person
    - car
    - dog
    - cat
    - bird
    - horse
    - sheep
    - cow
  filters:
    person:
      min_score: 0.5
      threshold: 0.7
    dog:
      min_score: 0.45
      threshold: 0.6
    cat:
      min_score: 0.45
      threshold: 0.6
    bird:
      min_score: 0.4
      threshold: 0.5
    horse:
      min_score: 0.45
      threshold: 0.6
    sheep:
      min_score: 0.45
      threshold: 0.6
    cow:
      min_score: 0.45
      threshold: 0.6
```

Notes:
- Farm animals are often small/far in wide-angle shots, so keep `min_score`/`threshold`
  slightly lower than for `person`. Tune after observing false positives.
- Optionally add `elephant`, `bear`, `zebra`, `giraffe` to `track` — harmless on a farm
  but noisier; leave them out by default.

### 1.2 Optional: per-camera masks to cut false positives

If moving foliage, shadows, or vehicles near camera edges cause spurious detections,
add a `mask` polygon per camera (or a `motion.mask` region) so the detector ignores
those areas:

```yaml
cam01:
  ...
  motion:
    mask:
      - "0,0,320,0,320,180,0,180"   # example: ignore bottom half, etc.
```

Masks are drawn most easily from the Frigate UI Debug view (mask editor), which writes
them back into the config.

### 1.3 Redeploy & verify

```bash
# from this workspace, copy updated config to host
scp config/config.yaml dr@debian:~/frigate/config/config.yaml

# on host
cd ~/frigate
docker compose restart frigate
```

Verification:
- Frigate UI (`http://<host-ip>:5000`) shows animal boxes (allow ~1 min warm-up).
- `mosquitto_sub -h localhost -p 1883 -t 'frigate/#'` shows event JSON whose `label`
  field is e.g. `dog`, `cat`, `sheep`, `cow` when an animal passes a camera.
- No persistent `ERROR` lines in `docker compose logs frigate`.

---

## Part 2 — Custom YOLOv8 model for missing animals

### 2.1 Assemble a labeled dataset

- Use Roboflow (or manual CVAT/labelImg) to collect images of `goat`, `camel`,
  `donkey`, `fox`, `wolf`.
- Include enough scene variety (day/night, distance, partial occlusion) matching the
  actual camera angles. Pulling still frames from the 10 cameras is ideal for
  fine-tuning realism.
- **Important:** because this model will *replace* the COCO model, the dataset must also
  include `person`, `car`, and the COCO animal classes we track (dog, cat, bird, horse,
  sheep, cow), or those detections will be lost. Roboflow can supply pre-labeled COCO
  class images; the custom animals are appended.
- Export as YOLOv8 (Ultralytics) format, train/val split.

### 2.2 Train

- Recommended model: `yolov8n` (CPU-friendly) to start; move to `yolov8s` only if
  accuracy is insufficient and CPU budget allows.
- Train off-box (Google Colab GPU or any GPU machine) — the i7-9700 is not suitable for
  training.
- Produce a best-weight checkpoint, e.g. `best.pt`.

### 2.3 Export to Frigate-compatible OpenVINO IR

Frigate's OpenVINO detector requires:
- OpenVINO IR format (`.xml` + `.bin`).
- **NMS-free** single output tensor of shape `[1, N, 6]` with columns
  `[class_id, confidence, x1, y1, x2, y2]` (Frigate runs its own tracker/aggregation;
  the model must not apply NMS itself).

Export path (verify against current Frigate docs for the exact Ultralytics
`format=` value — it is `frigate` for the end-to-end export that matches Frigate's
expected output tensor):

```bash
yolo export model=best.pt format=onnx            # NMS-free, end-to-end export
# then convert to OpenVINO IR:
mo --input_model best.onnx --output_dir ./openvino --compress_to_fp16
```

Expected artifacts:
```
openvino/best.xml
openvino/best.bin
labelmap.txt   # one class name per line, index order = model class order
```

### 2.4 Add model to workspace & mount into container

Place under `models/animals/` in this workspace:

```
models/
└── animals/
    ├── best.xml
    ├── best.bin
    └── labelmap.txt
```

Add a volume to the `frigate` service in [`docker-compose.yml`](../docker-compose.yml:26):

```yaml
    volumes:
      - ./config:/config
      - ./media:/media/frigate
      - ./models:/models:ro          # NEW: custom models
```

### 2.5 Point the detector at the custom model

In [`config/config.yaml`](../config/config.yaml:44):

```yaml
detectors:
  ov:
    type: openvino
    device: CPU
    model:
      path: /models/animals/best.xml
      labelmap_path: /models/animals/labelmap.txt
```

Update `objects.track` to the union of classes (custom + COCO) and add filters for the
custom classes, e.g.:

```yaml
objects:
  track:
    - person
    - car
    - dog
    - cat
    - bird
    - horse
    - sheep
    - cow
    - goat
    - camel
    - donkey
    - fox
    - wolf
  filters:
    goat:
      min_score: 0.45
      threshold: 0.6
    camel:
      min_score: 0.45
      threshold: 0.6
    donkey:
      min_score: 0.45
      threshold: 0.6
    fox:
      min_score: 0.4
      threshold: 0.5
    wolf:
      min_score: 0.4
      threshold: 0.5
    # ... plus the Part 1 filters
```

### 2.6 Redeploy & verify

```bash
# from workspace
scp -r docker-compose.yml config models dr@debian:~/frigate/

# on host
cd ~/frigate
docker compose up -d --force-recreate frigate
```

Verification:
- UI shows boxes for both COCO and custom animals (goat/camel/etc.) when present.
- `mosquitto_sub ... -t 'frigate/#'` shows `label` values for the custom classes.
- Confirm person/car detections still work (regression check — this model replaced the
  COCO model).
- Check CPU: `top`/`docker stats` — if inference is too slow, drop `detect.fps` from 5
  to 3 or downgrade `yolov8s` → `yolov8n`.

---

## MQTT note

No MQTT changes are needed. Frigate already publishes `frigate/events` with the detected
`label`; animal labels will simply appear in the payload. This keeps the door open for
later phases (SQLite logging, alerts) without rework.

---

## Deferred (not in scope, per user)

- Instant alerts (Telegram/webhook) on animal detection
- SQLite event logging for animals
- Per-camera zones restricting animal detection to specific areas
- Night-vision-specific tuning (IR mode)

---

## Verification checklist

- [ ] Part 1: `objects.track` includes COCO animal classes
- [ ] Part 1: animal boxes appear in UI and MQTT `frigate/events` labels
- [ ] Part 2: `models/animals/` present with `.xml`, `.bin`, `labelmap.txt`
- [ ] Part 2: `docker-compose.yml` mounts `./models`
- [ ] Part 2: `detectors.ov.model.path` points to `/models/animals/best.xml`
- [ ] Part 2: custom animals (goat/camel/donkey/fox/wolf) detected and labeled
- [ ] Part 2: person/car still detected (no regression)
- [ ] CPU within budget at `detect.fps: 5` (reduce to 3 if not)
