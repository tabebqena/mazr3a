#!/usr/bin/env python3
"""firewatch.py - farm fire/smoke watchdog for the Frigate NVR stack.

Runs as the `firewatch` Docker service (docker-compose.yml). It polls each
camera's ALREADY-DECODED detect frame from Frigate's REST API
(GET /api/<camera>/latest.jpg - no extra ffmpeg decode), runs a small
fire/smoke YOLO model via OpenVINO, and sends a Telegram PHOTO alert on a
sustained detection (min consecutive hits above a score threshold, with a
per-camera cooldown to prevent spam).

Design notes
------------
- Frigate's own person/car/animal detection (config/config.yaml) is NOT
  touched; this watcher is fully out-of-band (see plans/fire-detection.md).
- Code/config live on runtime mounts (./scripts, ./config, ./models are
  mounted read-only into the container) so edits need no image rebuild.
- The watcher is stdlib + openvino + numpy + Pillow only. monitor_lib is
  imported from the mounted ./scripts directory.

Model assumption (see plans/fire-detection.md section 5.1)
----------------------------------------------------------
The model under MODEL_DIR is an OpenVINO IR export of a fire/smoke YOLOv8n
(single output tensor [1, 4+nc, N] or [1, N, 4+nc], xywh boxes in input
pixels, class scores 0..1). The decoder also handles an already-sigmoided
output. labelmap.txt lists one class per line (typically: fire, smoke).

Usage
-----
  python firewatch.py            # run the polling loop
  python firewatch.py --once     # single pass over all cameras, then exit
  python firewatch.py --dry-run  # like --once but print instead of sending
  python firewatch.py --check    # validate config + load model, then exit
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import monitor_lib as lib  # noqa: E402

# Pillow is only used to overlay detection boxes on the alert snapshot; it is
# a hard dependency of the firewatch image (see firewatch/requirements.txt).
from PIL import Image, ImageDraw, ImageFont

CONF_PATH = os.environ.get("FIREWATCH_CONF", "/config/firewatch.conf")
LOG = lambda *a: print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------
# tiny KEY=VALUE conf reader with typed defaults (mirrors monitor_lib.load_conf)
# ---------------------------------------------------------------------------
def _raw_conf(path):
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except FileNotFoundError:
        return {}
    return cfg


def _get(raw, key, default=""):
    value = os.environ.get(key) or raw.get(key) or default
    return str(value).strip()


def _getf(raw, key, default):
    try:
        return float(_get(raw, key, str(default)))
    except ValueError:
        return default


def _geti(raw, key, default):
    try:
        return int(_get(raw, key, str(default)))
    except ValueError:
        return default


def _getb(raw, key, default):
    return _get(raw, key, "true" if default else "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ---------------------------------------------------------------------------
# OpenVINO fire/smoke inference
# ---------------------------------------------------------------------------
class FireModel:
    """Loads an OpenVINO IR YOLO model and decodes fire/smoke detections."""

    def __init__(self, model_dir, labelmap_path=None):
        from openvino import Core  # imported lazily so --check works w/o deps

        xml_path = os.path.join(model_dir, "best.xml")
        if not os.path.isfile(xml_path):
            raise RuntimeError(
                f"model not found at {xml_path} - populate {model_dir} "
                "(see models/fire/README.md and plans/fire-detection.md)"
            )
        self._core = Core()
        self._model = self._core.read_model(xml_path)
        self._compiled = self._core.compile_model(self._model, "CPU")
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)

        shape = list(self._input.shape)
        if len(shape) != 4:
            raise RuntimeError(f"unexpected input shape {shape}")
        # OpenVINO reports NCHW for most CV models; handle NHWC defensively.
        if shape[1] in (1, 3) and shape[1] < shape[3]:
            _, self.height, self.width, self.ch = shape
        else:
            _, self.ch, self.height, self.width = shape
        self.input_name = self._input.any_name
        self.output_name = self._output.any_name

        # class name -> index from labelmap (e.g. fire=0, smoke=1)
        self.labels = []
        if labelmap_path and os.path.isfile(labelmap_path):
            with open(labelmap_path, encoding="utf-8") as fh:
                self.labels = [ln.strip() for ln in fh if ln.strip()]
        if not self.labels:
            # guess: first two outputs after xywh are the classes fire/smoke
            self.labels = ["fire", "smoke"]
        self.nc = len(self.labels)
        LOG(f"model {xml_path}: input {self.width}x{self.height} ch={self.ch} "
            f"classes={self.labels}")

    # -- preprocess ------------------------------------------------------
    @staticmethod
    def _letterbox(img, size):
        """Resize keeping aspect ratio, pad with 114 (ultralytics style)."""
        iw, ih = img.size
        scale = min(size / iw, size / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))[None], scale, (size - nw) // 2, (size - nh) // 2

    # -- postprocess -----------------------------------------------------
    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (aa + ba - inter + 1e-9)

    def detect(self, image, score_thresh=0.5, iou_thresh=0.45, allowed=None):
        """Run inference on a PIL image. Returns list of
        dicts {label, score, box(x1,y1,x2,y2 in ORIGINAL image coords)}.
        `allowed` filters to class names (e.g. {'fire'})."""
        blob, scale, pad_x, pad_y = self._letterbox(image, self.width)
        out = self._compiled([blob])[self._output]
        data = np.asarray(out)
        if data.ndim == 3:
            data = data[0]
        # accept [1, 4+nc, N] and [1, N, 4+nc]
        if data.shape[0] == 4 + self.nc and data.shape[1] > data.shape[0]:
            data = data.T
        if data.shape[1] < 4 + self.nc:
            raise RuntimeError(
                f"model output has {data.shape[1]} cols; expected >= {4 + self.nc}"
            )
        boxes = data[:, :4].astype(np.float32)  # cxcywh in input pixels
        scores = data[:, 4 : 4 + self.nc].astype(np.float32)
        # ultralytics exports may leave class logits raw - sigmoid when needed
        if np.nanmax(scores) > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        dets = []
        for ci in range(self.nc):
            if allowed and self.labels[ci] not in allowed:
                continue
            idx = np.where(scores[:, ci] >= score_thresh)[0]
            for i in idx:
                cx, cy, bw, bh = boxes[i]
                x1 = (cx - bw / 2 - pad_x) / scale
                y1 = (cy - bh / 2 - pad_y) / scale
                x2 = (cx + bw / 2 - pad_x) / scale
                y2 = (cy + bh / 2 - pad_y) / scale
                dets.append(
                    {
                        "label": self.labels[ci],
                        "score": float(scores[i, ci]),
                        "box": (max(0.0, x1), max(0.0, y1), x2, y2),
                    }
                )
        # simple NMS across classes
        dets.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) <= iou_thresh for k in keep):
                keep.append(d)
        return keep


# ---------------------------------------------------------------------------
# snapshot + alert helpers
# ---------------------------------------------------------------------------
def fetch_frame(api, camera, timeout=10):
    """Return (jpeg_bytes) for a camera's latest detect frame."""
    url = f"{api}/api/{camera}/latest.jpg"
    req = urllib.request.Request(url, headers={"Accept": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def overlay_boxes(jpeg_bytes, dets, model_hw=None):
    """Return JPEG bytes with red boxes + labels overlaid (no-op without PIL)."""
    if Image is None or not dets:
        return jpeg_bytes
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{d['label']} {d['score']:.2f}"
        draw.text((x1, max(0, y1 - 12)), text, fill="red", font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def build_caption(camera, dets, track_smoke):
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    top = "🔥 FIRE ALERT" if any(d["label"] == "fire" for d in dets) else "⚠️ FIRE WATCH"
    if not track_smoke and all(d["label"] == "smoke" for d in dets):
        top = "💨 SMOKE WATCH"
    lines = [f"<b>{top}</b>", f"<b>Camera:</b> {lib.esc_html(camera)}",
             f"<b>Time:</b> {lib.esc_html(ts)}"]
    for d in sorted(dets, key=lambda x: x["score"], reverse=True)[:5]:
        lines.append(f"• {lib.esc_html(d['label'])} {d['score']:.2f}")
    return "\n".join(lines)


def run_once(cfg, model, dry=False):
    api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    track_smoke = _getb(cfg, "TRACK_SMOKE", False)
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    for cam in cameras:
        try:
            raw = fetch_frame(api, cam)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            dets = model.detect(img, score_thresh=threshold, allowed=allowed)
            line = f"{cam}: {len(dets)} fire" if not track_smoke else \
                f"{cam}: {sum(1 for d in dets if d['label']=='fire')} fire / " \
                f"{sum(1 for d in dets if d['label']=='smoke')} smoke"
            LOG(line)
            if dets:
                if dry:
                    print(build_caption(cam, dets, track_smoke))
                    continue
                jpg = overlay_boxes(raw, dets)
                send_alert(cfg, cam, jpg, dets, track_smoke)
        except urllib.error.HTTPError as exc:
            LOG(f"{cam}: fetch failed HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            LOG(f"{cam}: error: {exc}")
    return True


def send_alert(cfg, cam, jpg, dets, track_smoke):
    text = build_caption(cam, dets, track_smoke)
    try:
        lib.send_telegram_photo(cfg, jpg, text)
        LOG(f"ALERT sent for {cam}")
    except Exception as exc:  # noqa: BLE001
        LOG(f"ALERT FAILED for {cam}: {exc}")


def run_forever(cfg, model):
    api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    interval = _getf(cfg, "POLL_INTERVAL_S", 15)
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    min_hits = _geti(cfg, "MIN_HITS", 3)
    cooldown = _getf(cfg, "COOLDOWN_S", 300)
    track_smoke = _getb(cfg, "TRACK_SMOKE", False)
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    enabled = _getb(cfg, "ENABLED", True)
    if not cameras:
        LOG("ERROR: CAMERAS list empty - exiting")
        sys.exit(2)

    state = {c: {"hits": 0, "last_alert": 0.0, "down": False}
             for c in cameras}
    LOG(f"started: {len(cameras)} cameras, sweep every {interval}s, "
        f"min_hits={min_hits}, cooldown={cooldown}s, smoke={track_smoke}, "
        f"enabled={enabled}")
    while True:
        if not enabled:
            LOG("disabled (ENABLED=false) - sleeping 60s")
            time.sleep(60)
            continue
        for cam in cameras:
            st = state[cam]
            try:
                raw = fetch_frame(api, cam)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                dets = model.detect(img, score_thresh=threshold, allowed=allowed)
                if st["down"]:
                    LOG(f"{cam}: back online")
                    st["down"] = False
                if dets:
                    st["hits"] += 1
                    best = max(d["score"] for d in dets)
                    now = time.monotonic()
                    in_cooldown = now - st["last_alert"] < cooldown
                    if st["hits"] >= min_hits and not in_cooldown:
                        st["last_alert"] = now
                        st["hits"] = 0
                        LOG(f"{cam}: ALERT ({min_hits} hits, conf {best:.2f})")
                        jpg = overlay_boxes(raw, dets)
                        send_alert(cfg, cam, jpg, dets, track_smoke)
                    else:
                        LOG(f"{cam}: fire-hit {st['hits']}/{min_hits} "
                            f"(conf {best:.2f})" + (" [cooldown]" if in_cooldown else ""))
                else:
                    if st["hits"]:
                        LOG(f"{cam}: cleared after {st['hits']} hits")
                    st["hits"] = 0
            except urllib.error.HTTPError as exc:
                st["hits"] = 0
                if not st["down"]:
                    LOG(f"{cam}: HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                st["hits"] = 0
                if not st["down"]:
                    LOG(f"{cam}: error: {exc}")
                    st["down"] = True
        time.sleep(interval)


def main():
    args = set(sys.argv[1:])
    raw = _raw_conf(CONF_PATH)
    if not raw:
        LOG(f"ERROR: no config at {CONF_PATH} - aborting")
        return 2

    # Telegram creds come from the SAME git-ignored telegram.conf the cron
    # scripts use (mounted at /config/telegram.conf in the container).
    telegram_path = os.environ.get("TELEGRAM_CONF", _get(raw, "TELEGRAM_CONF",
                                                         "/config/telegram.conf"))
    cfg = lib.load_conf(telegram_path)
    # merge firewatch tunables into cfg so helpers can read camera config too
    cfg.update(raw)

    try:
        model = FireModel(_get(raw, "MODEL_DIR", "/models/fire"))
    except RuntimeError as exc:
        LOG(f"ERROR: {exc}")
        return 2

    if "--check" in args:
        LOG("config + model OK")
        return 0
    if "--dry-run" in args or "--once" in args:
        try:
            run_once(cfg, model, dry="--dry-run" in args)
        except Exception as exc:  # noqa: BLE001
            LOG(f"ERROR: {exc}")
            return 1
        return 0

    try:
        lib.ensure_creds(cfg)
    except RuntimeError as exc:
        LOG(f"ERROR: {exc}")
        return 2

    run_forever(cfg, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
