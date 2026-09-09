#!/usr/bin/env python3
"""firewatch.py - farm fire/smoke watchdog for the Frigate NVR stack (v3).

Runs as the `firewatch` Docker service (docker-compose.yml). It reads each
camera's ALREADY-DECODED detect frame from Frigate's REST API
(GET /api/<cam>/latest.jpg - no extra ffmpeg decode), runs a fire/smoke YOLO
model via OpenVINO, and sends a Telegram PHOTO alert for a confirmed fire.

v3 (2026-09-09, plans/firewatch-motion-gate-bonus.md - ESTABLISHED SPEC):
firewatch now fuses two signals - the fire model's per-box confidence AND a
motion signal measured on frames pulled ~MOTION_GAP_S apart:
  * Motion is a GATE on running the model: a batched two-frame motion sweep
    decides which cameras get an inference (motion gate passed or a periodic
    BASELINE_EVERY_S run); quiet cameras are skipped (CPU saved).
  * Motion is an EVIDENCE TIER in the score (SPATIAL AGREEMENT): a fire box
    that sits NEAR motion (flicker) is a moving fire - it confirms at the low
    bar SCORE_THRESHOLD with a MOTION_BONUS. A fire box with NO nearby motion
    gets no bonus and must clear the high bar SCORE_HIGH.
  * CONFIRMATION: a camera that gets a hit is promoted to a dense FOLLOW-UP
    (every FOLLOWUP_GAP_S) and alerts on MIN_HITS confirms within HITS_WINDOW.
  * COOLDOWN is per camera after ANY follow-up session ends (alert or not):
    COOLDOWN_S after a confirmed alert, COOLDOWN_NOALERT_S after an
    unconfirmed exit.
  * Evidence store keeps TWO images per fire-marked frame: the ORIGINAL JPEG
    (what the model scored; canonical frames.jpg_path) and an ANNOTATED twin
    (<base>_annotated.jpg, red boxes + label + score). frames.annotated_path
    records the annotated file (nullable, guarded ALTER).
MOTION_ENABLED=false restores the previous score-only polling behaviour
(run_legacy).

Robustness (spec 13): the daemon while-loop is the only unbounded loop; every
pass sleeps a bounded TICK; every external call has a hard timeout; every
per-camera operation is exception-isolated at the camera boundary; timers use
time.monotonic(); SIGTERM/SIGINT set a stop flag checked each tick; the
evidence store is best-effort and never stops the loop.

Evidence store background: SQLite (WAL) at <STORE_DIR>/firewatch.db holds one
`frames` row + per-box `detections` rows per stored frame; JPEGs stay as plain
files under <STORE_DIR>/<cam>/ (see plans/firewatch-sqlite-evidence-store.md).
STORE_ENABLED=false reverts to send-only behaviour.

Usage:
  python firewatch.py            # run the polling loop
  python firewatch.py --once     # single pass over all cameras, then exit
  python firewatch.py --dry-run  # like --once but print instead of sending
  python firewatch.py --check    # validate config + load model, then exit
"""
import collections
import io
import os
import signal
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

# firewatch.py lives in firewatch/, shared helpers (telegram_notify.py) stay in
# the sibling scripts/ dir; make both importable (container mounts mirror this).
_FW_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _FW_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_FW_DIR), "scripts"))
import telegram_notify as tg  # noqa: E402

# Pillow is only used to overlay boxes + blur grayscale frames.
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

CONF_PATH = os.environ.get("FIREWATCH_CONF", "/config/firewatch.conf")
LOG = lambda *a: print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)  # noqa: E731
_STOP = [False]  # set by SIGTERM/SIGINT -> clean, prompt shutdown (spec 13.G)


def _sig_stop(signum, frame):  # noqa: ARG001 - signal handler
    _STOP[0] = True


# ---------------------------------------------------------------------------
# tiny KEY=VALUE conf reader with typed defaults
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
    return str(os.environ.get(key) or raw.get(key) or default).strip()


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
        "1", "true", "yes", "on")


def _clampf(v, lo, hi, default):
    """Clamp a float config value into [lo, hi]; fall back to `default`."""
    v = float(v)
    if not (lo <= v <= hi):
        return default
    return v


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
        if shape[1] in (1, 3) and shape[1] < shape[3]:
            _, self.height, self.width, self.ch = shape
        else:
            _, self.ch, self.height, self.width = shape
        self.input_name = self._input.any_name
        self.output_name = self._output.any_name

        self.labels = []
        labelmap = labelmap_path or os.path.join(model_dir, "labelmap.txt")
        if labelmap and os.path.isfile(labelmap):
            with open(labelmap, encoding="utf-8") as fh:
                self.labels = [ln.strip() for ln in fh if ln.strip()]
        if not self.labels:
            self.labels = ["fire", "smoke"]
        self.nc = len(self.labels)
        LOG(f"model {xml_path}: input {self.width}x{self.height} ch={self.ch} "
            f"output {list(self._output.shape)} classes={self.labels}")

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
        if data.ndim == 2 and data.shape[0] == 4 + self.nc and data.shape[1] > data.shape[0]:
            data = data.T
        cols = data.shape[1]
        raw_cols = 4 + self.nc
        if cols == raw_cols:
            mode = "raw"
        elif cols == 6:
            mode = "e2e"   # YOLO26 end-to-end NMS
        else:
            raise RuntimeError(
                f"model output has {cols} cols; expected {raw_cols} (raw per-class) "
                f"or 6 (end-to-end NMS). See models/fire/README.md"
            )

        dets = []
        if mode == "raw":
            boxes = data[:, :4].astype(np.float32)
            scores = data[:, 4:4 + self.nc].astype(np.float32)
            if np.nanmax(scores) > 1.0:
                scores = 1.0 / (1.0 + np.exp(-scores))
            for ci in range(self.nc):
                if allowed and self.labels[ci].lower() not in allowed:
                    continue
                idx = np.where(scores[:, ci] >= score_thresh)[0]
                for i in idx:
                    cx, cy, bw, bh = boxes[i]
                    x1 = (cx - bw / 2 - pad_x) / scale
                    y1 = (cy - bh / 2 - pad_y) / scale
                    x2 = (cx + bw / 2 - pad_x) / scale
                    y2 = (cy + bh / 2 - pad_y) / scale
                    dets.append({
                        "label": self.labels[ci], "score": float(scores[i, ci]),
                        "box": (max(0.0, x1), max(0.0, y1), x2, y2),
                    })
        else:
            xyxy = data[:, :4].astype(np.float32)
            confs = data[:, 4].astype(np.float32)
            cids = np.clip(data[:, 5].astype(np.int64), 0, self.nc - 1)
            for i in np.where(confs >= score_thresh)[0]:
                label = self.labels[int(cids[i])]
                if allowed and label.lower() not in allowed:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                dets.append({
                    "label": label, "score": float(confs[i]),
                    "box": (max(0.0, (x1 - pad_x) / scale),
                            max(0.0, (y1 - pad_y) / scale),
                            (x2 - pad_x) / scale, (y2 - pad_y) / scale),
                })

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
    """Return (jpeg_bytes) for a camera's latest detect frame (bounded by timeout)."""
    url = f"{api}/api/{camera}/latest.jpg"
    req = urllib.request.Request(url, headers={"Accept": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def overlay_boxes(jpeg_bytes, dets, model_hw=None):  # noqa: ARG001 - API compat
    """Return JPEG bytes with red boxes + labels overlaid."""
    if not dets:
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


def _decode(img_bytes):
    """Decode JPEG bytes to an RGB PIL image. May raise (caller isolates)."""
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _gray(img_rgb):
    """Grayscale uint8 array of a decoded frame."""
    return np.asarray(img_rgb.convert("L"), dtype=np.uint8)


def _blur_gray(gray, radius):
    """Gaussian-blur a grayscale array (PIL-backed; radius >= 0.5)."""
    if radius <= 0.0:
        return gray
    return np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.uint8)


def _morph_open(mask):
    """3x3 morphological open (erode then dilate) - drops 1px speckles.

    Pure numpy (no scipy/cv2 dependency): min/max over a padded 3x3 window.
    """
    h, w = mask.shape
    pad = np.pad(mask, 1)
    eroded = np.ones((h, w), dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            eroded &= pad[dy:dy + h, dx:dx + w]
    pad2 = np.pad(eroded, 1)
    opened = np.zeros((h, w), dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            opened |= pad2[dy:dy + h, dx:dx + w]
    return opened


def _changed_mask(a_gray, b_gray, px_diff, blur_radius):
    """Denoised motion mask: |blur(A) - blur(B)| > px_diff, speckles opened out."""
    a = _blur_gray(a_gray, blur_radius).astype(np.int16)
    b = _blur_gray(b_gray, blur_radius).astype(np.int16)
    m = np.abs(a - b) > px_diff
    return _morph_open(m)


def _adaptive_px(base, night_extra, night_luma, mean_luma):
    """Day/night pixel-diff: raise the threshold when it is dark (ISO/IR noise)."""
    return base + (night_extra if mean_luma < night_luma else 0.0)


def _expanded_region(mask, box, margin):
    """Boolean region of `mask` inside `box` expanded by margin (x box size) per side."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    ax1 = max(0, int(x1 - w * margin))
    ay1 = max(0, int(y1 - h * margin))
    ax2 = min(mask.shape[1], int(x2 + w * margin) + 1)
    ay2 = min(mask.shape[0], int(y2 + h * margin) + 1)
    if ax2 <= ax1 or ay2 <= ay1:
        return None
    return mask[ay1:ay2, ax1:ax2]


def _region_stats(mask, box, margin):
    """(changed_fraction, changed_count) inside the expanded region of `box`."""
    region = _expanded_region(mask, box, margin)
    if region is None:
        return 0.0, 0
    cnt = int(region.sum())
    return (cnt / region.size if region.size else 0.0), cnt


def classify_sample(dets, cur_mask, prev_masks, margin, near_frac, near_min_px,
                    bonus, bar, high):
    """Spatial-agreement classification of one sample's raw detections (spec 6).

    A box is `near` motion when, inside the box expanded by `margin`, the
    changed-pixel fraction >= near_frac OR the changed count >= near_min_px
    (small/distant boxes), OR a previous mask (hotspot younger than the ring)
    shows >= near_min_px changed pixels AND the current frame still shows any
    change (>= 1 px) there - so a purely stale hotspot never grants a bonus.

    eff = min(1, raw + (bonus if near else 0)).
    Returns (marked, hit, n_candidates, n_static):
      marked      boxes with eff >= bar (stored as evidence)
      hit         True if any box confirms: near and eff >= bar,
                  OR (no motion) raw >= high
      n_static    marked-but-not-confirm boxes (static 0.5-0.85, evidence only)
    """
    marked, hit = [], False
    n_static = 0
    prev_masks = prev_masks or []
    for d in dets:
        raw = d["score"]
        near = False
        if cur_mask is not None:
            frac0, cnt0 = _region_stats(cur_mask, d["box"], margin)
            near = frac0 >= near_frac or cnt0 >= near_min_px
            if not near and cnt0 >= 1:            # minimal current change required
                for pm in prev_masks:
                    if pm is None:
                        continue
                    _, pc = _region_stats(pm, d["box"], margin)
                    if pc >= near_min_px:
                        near = True
                        break
        eff = min(1.0, raw + (bonus if near else 0.0))
        box = dict(d)
        box["eff"] = eff
        box["near"] = near
        if eff >= bar:
            marked.append(box)
            if near or raw >= high:
                hit = True
            else:
                n_static += 1
        # raw below the evidence bar (with no near bonus) -> ignored
    return marked, hit, len(dets), n_static


def best_of(boxes):
    """The marked box with the highest effective score (used for log/alert/store)."""
    return max(boxes, key=lambda b: b.get("eff", b["score"]))


# ---------------------------------------------------------------------------
# main loop + alert helpers (single-pass and legacy score-only polling)
# ---------------------------------------------------------------------------
def build_caption(camera, dets, track_smoke):
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    top = "🔥 FIRE ALERT" if any(d["label"].lower() == "fire" for d in dets) else "⚠️ FIRE WATCH"
    if not track_smoke and all(d["label"].lower() == "smoke" for d in dets):
        top = "💨 SMOKE WATCH"
    lines = [f"<b>{top}</b>", f"<b>Camera:</b> {tg.esc_html(camera)}",
             f"<b>Time:</b> {tg.esc_html(ts)}"]
    for d in sorted(dets, key=lambda x: x.get("eff", x["score"]), reverse=True)[:5]:
        lines.append(f"• {tg.esc_html(d['label'])} {d['score']:.2f}"
                     + (" (motion)" if d.get("near") else ""))
    return "\n".join(lines)


def send_alert(cfg, cam, jpg, dets, track_smoke):
    text = build_caption(cam, dets, track_smoke)
    try:
        tg.send_telegram_photo(cfg, jpg, text)
        LOG(f"ALERT sent for {cam}")
    except Exception as exc:  # noqa: BLE001 - Telegram failure must not stop the loop
        LOG(f"ALERT FAILED for {cam}: {exc}")


def run_once(cfg, model, dry=False):
    api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    track_smoke = _getb(cfg, "TRACK_SMOKE", False)
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    for cam in cameras:
        try:
            raw = fetch_frame(api, cam)
            img = _decode(raw)
            dets = model.detect(img, score_thresh=threshold, allowed=allowed)
            line = f"{cam}: {len(dets)} fire" if not track_smoke else \
                f"{cam}: {sum(1 for d in dets if d['label']=='fire')} fire / " \
                f"{sum(1 for d in dets if d['label']=='smoke')} smoke"
            LOG(line)
            if dets:
                if dry:
                    print(build_caption(cam, dets, track_smoke))
                    continue
                store_fire_frame(cfg, cam, raw, dets, alerted=True)
                jpg = overlay_boxes(raw, dets)
                send_alert(cfg, cam, jpg, dets, track_smoke)
        except urllib.error.HTTPError as exc:
            LOG(f"{cam}: fetch failed HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            LOG(f"{cam}: error: {exc}")
    return True


# ---------------------------------------------------------------------------
# SQLite evidence store (WAL). JPEGs stay plain files; DB rows hold paths.
# v3: every stored frame writes TWO files - the ORIGINAL (frames.jpg_path,
# canonical) and an ANNOTATED twin (frames.annotated_path, nullable).
# ---------------------------------------------------------------------------
_STORE_CONN = None
_STORE_LAST_PRUNE = 0.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera          TEXT    NOT NULL,
    captured_at     REAL    NOT NULL,
    ts_utc          TEXT    NOT NULL,
    jpg_path        TEXT    NOT NULL,
    annotated_path  TEXT,
    best_score      REAL    NOT NULL,
    score_threshold REAL    NOT NULL,
    alerted         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS detections (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label    TEXT    NOT NULL,
    score    REAL    NOT NULL,
    x1       REAL    NOT NULL,
    y1       REAL    NOT NULL,
    x2       REAL    NOT NULL,
    y2       REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_camera_ts ON frames(camera, captured_at);
CREATE INDEX IF NOT EXISTS idx_frames_ts         ON frames(captured_at);
CREATE INDEX IF NOT EXISTS idx_detections_frame  ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_label  ON detections(label);
"""


def _store_db_path(cfg):
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return None
    return os.path.join(store_dir, _get(cfg, "STORE_DB", "firewatch.db"))


def _store_conn(cfg):
    """Return the cached WAL-mode evidence connection (schema + guarded migration)."""
    global _STORE_CONN
    if _STORE_CONN is not None:
        try:
            _STORE_CONN.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            _STORE_CONN = None
    if _STORE_CONN is None:
        db_path = _store_db_path(cfg)
        if db_path is None:
            return None
        store_dir = os.path.dirname(db_path) or "."
        os.makedirs(store_dir, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")   # spec 13.E / review 4.1
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # guarded idempotent migrations (existing DBs created before v3)
        for col, ddl in (
            ("alerted", "ALTER TABLE frames ADD COLUMN "
                        "alerted INTEGER NOT NULL DEFAULT 0"),
            ("annotated_path", "ALTER TABLE frames ADD COLUMN "
                               "annotated_path TEXT"),
        ):
            try:
                conn.execute(ddl)
            except sqlite3.Error:
                pass  # column already present
        _STORE_CONN = conn
    return _STORE_CONN


def _prune_store(cfg, conn):
    """Optional retention: STORE_RETENTION_DAYS > 0 prunes old frame rows."""
    global _STORE_LAST_PRUNE
    days = _getf(cfg, "STORE_RETENTION_DAYS", 0)
    if days <= 0:
        return
    now = time.time()
    if now - _STORE_LAST_PRUNE < 60:
        return
    _STORE_LAST_PRUNE = now
    cur = conn.execute(
        "DELETE FROM frames WHERE captured_at < ?", (now - days * 86400,)
    )
    if cur.rowcount:
        LOG(f"store: pruned {cur.rowcount} frame record(s) older than {days:g} days")


def store_fire_frame(cfg, cam, jpeg_bytes, dets, alerted=False):
    """Best-effort persist a fire-marked frame: TWO JPEGs + one DB row.

    Writes <STORE_DIR>/<cam>/<base>.jpg (the ORIGINAL bytes the model scored,
    canonical frames.jpg_path) and <STORE_DIR>/<cam>/<base>_annotated.jpg (red
    boxes + label + score overlay), then one `frames` row (jpg_path,
    annotated_path, best effective score, alerted) + `detections` rows in the
    WAL DB. Never raises (keeps the loop alive); failures are logged/skipped
    and the connection is dropped so the next poll reopens it.
    """
    if not _getb(cfg, "STORE_ENABLED", True):
        return
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return
    try:
        cam_dir = os.path.join(store_dir, cam)
        os.makedirs(cam_dir, exist_ok=True)
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
        best = max((d.get("eff", d["score"]) for d in dets), default=0.0)
        base = f"{stamp}_{int(now % 1 * 1000):03d}_{cam}_conf{best:.2f}"
        raw_path = os.path.join(cam_dir, base + ".jpg")
        ann_path = os.path.join(cam_dir, base + "_annotated.jpg")
        with open(raw_path, "wb") as fh:
            fh.write(jpeg_bytes)
        ann_bytes = overlay_boxes(jpeg_bytes, dets)
        with open(ann_path, "wb") as fh:
            fh.write(ann_bytes)

        conn = _store_conn(cfg)
        if conn is None:
            LOG(f"{cam}: store SKIPPED (no STORE_DIR/STORE_DB path)")
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO frames (camera, captured_at, ts_utc, jpg_path, "
            "annotated_path, best_score, score_threshold, alerted) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (cam, now,
             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
             raw_path, ann_path, round(best, 4),
             _getf(cfg, "SCORE_THRESHOLD", 0.5),
             1 if alerted else 0),
        )
        frame_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO detections (frame_id, label, score, x1, y1, x2, y2) "
            "VALUES (?,?,?,?,?,?,?)",
            [(frame_id, d["label"], round(d["score"], 4),
              round(d["box"][0], 1), round(d["box"][1], 1),
              round(d["box"][2], 1), round(d["box"][3], 1))
             for d in sorted(dets, key=lambda x: x.get("eff", x["score"]),
                             reverse=True)],
        )
        conn.commit()
        _prune_store(cfg, conn)
        LOG(f"{cam}: stored fire frame #{frame_id} "
            f"{os.path.basename(raw_path)} (conf {best:.2f}, "
            f"{len(jpeg_bytes) // 1024} KB, {len(dets)} detections)")
    except Exception as exc:  # noqa: BLE001 - evidence store must never kill loop
        global _STORE_CONN
        _STORE_CONN = None
        LOG(f"{cam}: store FAILED: {exc}")


def _close_store():
    global _STORE_CONN
    if _STORE_CONN is not None:
        try:
            _STORE_CONN.close()
        except sqlite3.Error:
            pass
        _STORE_CONN = None


# ---------------------------------------------------------------------------
# v3 scheduler settings (resolved + clamped once)
# ---------------------------------------------------------------------------
class S:
    """Resolved, clamped tunables + per-camera shared objects (no global drift)."""
    pass


def _resolve_settings(cfg, cameras, threshold):
    s = S()
    s.api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    s.cameras = cameras
    s.allowed = {"fire"} if not _getb(cfg, "TRACK_SMOKE", False) else {"fire", "smoke"}
    s.motion_enabled = _getb(cfg, "MOTION_ENABLED", False)
    s.poll_interval = max(1.0, _getf(cfg, "POLL_INTERVAL_S", 15))
    s.gap_s = _clampf(_getf(cfg, "MOTION_GAP_S", 1.5), 0.2, 60, 1.5)
    s.gap_min = _clampf(_getf(cfg, "MOTION_GAP_MIN_S", 0.5), 0.0, s.gap_s, 0.5)
    s.gap_max = _clampf(_getf(cfg, "MOTION_GAP_MAX_S", 5.0), s.gap_s, 120, 5.0)
    s.fetch_to = _clampf(_getf(cfg, "MOTION_FETCH_TIMEOUT_S", 1.0), 0.2, 10, 1.0)
    k = int(_geti(cfg, "MOTION_DENOISE_KERNEL", 3))
    s.blur_radius = max(0.0, (k - 1) / 2.0)
    s.px_diff = max(1.0, _getf(cfg, "MOTION_PIXEL_DIFF", 18))
    s.night_luma = _clampf(_getf(cfg, "MOTION_NIGHT_LUMA", 25), 0, 255, 25)
    s.night_extra = max(0.0, _getf(cfg, "MOTION_PIXEL_DIFF_NIGHT", 26) - s.px_diff)
    s.frame_frac = _clampf(_getf(cfg, "MOTION_FRAME_FRAC", 0.0015), 0.0, 1.0, 0.0015)
    s.frame_frac_max = _clampf(_getf(cfg, "MOTION_FRAME_FRAC_MAX", 0.20), s.frame_frac,
                               1.0, 0.20)
    s.persist_turns = max(0, int(_geti(cfg, "MOTION_PERSIST_TURNS", 2)))
    s.floor = _clampf(_getf(cfg, "SCORE_FLOOR", 0.35), 0.0, threshold, 0.35)
    s.baseline_every = max(10.0, _getf(cfg, "BASELINE_EVERY_S", 120))
    s.margin = _clampf(_getf(cfg, "MOTION_MARGIN", 0.5), 0.0, 4.0, 0.5)
    s.near_frac = _clampf(_getf(cfg, "MOTION_NEAR_FRAC", 0.02), 0.0, 1.0, 0.02)
    s.near_min_px = max(1, int(_geti(cfg, "MOTION_NEAR_MIN_PX", 15)))
    s.bonus = _clampf(_getf(cfg, "MOTION_BONUS", 0.15), 0.0, 0.9, 0.15)
    s.bar = threshold                                  # SCORE_THRESHOLD
    s.high = _clampf(_getf(cfg, "SCORE_HIGH", 0.85), s.bar, 1.0, 0.85)
    s.followup_gap = max(1.0, _getf(cfg, "FOLLOWUP_GAP_S", 5))
    s.followup_max_s = max(s.followup_gap, _getf(cfg, "FOLLOWUP_MAX_S", 90))
    s.min_hits = max(1, int(_geti(cfg, "MIN_HITS", 3)))
    s.window = max(s.min_hits, int(_geti(cfg, "HITS_WINDOW", 4)))
    s.cooldown = max(0.0, _getf(cfg, "COOLDOWN_S", 300))
    s.cooldown_noalert = max(0.0, _getf(cfg, "COOLDOWN_NOALERT_S", 60))
    s.death_turns = max(1, int(_geti(cfg, "FOLLOW_DEATH_TURNS", 4)))
    s.tick_min = 0.25
    s.heartbeat_s = max(60.0, _getf(cfg, "HEARTBEAT_S", 300))
    return s


def _new_state(s, cameras):
    st = {}
    for c in cameras:
        st[c] = {
            "mode": "IDLE",            # IDLE | FOLLOW | COOL
            "down": False,
            "mask_ring": collections.deque(maxlen=max(1, s.persist_turns)),
            "last_model": 0.0,         # monotonic when the model last ran
            "cool_until": 0.0,
            "next_follow": 0.0,
            "follow_until": 0.0,
            "empty_run": 0,
            "hits": collections.deque(maxlen=s.window),
            "fire_cams": 0,
        }
    return st


def _cam_err(cam, s, st, exc):
    """Log a camera error once per episode; camera goes DOWN; masks reset."""
    st["down"] = True
    st["mask_ring"].clear()
    if not st.get("_err_logged"):
        st["_err_logged"] = True
        LOG(f"{cam}: error: {exc}")
    st["err_count"] = st.get("err_count", 0) + 1


def _cam_ok(cam, s, st):
    if st["down"]:
        LOG(f"{cam}: back online")
    st["down"] = False
    st["_err_logged"] = False


# ---------------------------------------------------------------------------
# v3 scheduler: batched two-frame sweep -> spatial agreement -> follow-up
# ---------------------------------------------------------------------------
def _motion_of(s, a_bytes, b_bytes):
    """(mask, mean_luma) for a valid A/B pair (None-safe per caller)."""
    a = _gray(_decode(a_bytes))
    b = _gray(_decode(b_bytes))
    if a.shape != b.shape:
        return None, float(b.mean())
    px = _adaptive_px(s.px_diff, s.night_extra, s.night_luma, float(b.mean()))
    mask = _changed_mask(a, b, px, s.blur_radius)
    return mask, float(b.mean())


def _classify_dets(s, dets, cur_mask, st):
    return classify_sample(
        dets, cur_mask, list(st["mask_ring"]), s.margin, s.near_frac,
        s.near_min_px, s.bonus, s.bar, s.high)


def _store(cfg, cam, bytes_, boxes, alerted):
    if boxes:
        store_fire_frame(cfg, cam, bytes_, boxes, alerted=alerted)


def _enter_cool(st, now, secs):
    st["mode"] = "COOL"
    st["cool_until"] = now + secs
    st["hits"].clear()
    st["mask_ring"].clear()
    st["empty_run"] = 0


def _end_follow(cam, s, st, cfg, alerted, now):
    if alerted:
        LOG(f"{cam}: follow-up ended - ALERT (cooldown {s.cooldown:g}s)")
        _enter_cool(st, now, s.cooldown)
    else:
        LOG(f"{cam}: follow-up ended, NO alert (cooldown {s.cooldown_noalert:g}s)")
        _enter_cool(st, now, s.cooldown_noalert)


def _send_alert_frame(cfg, cam, s, jpeg_bytes, boxes, track_smoke):
    jpg = overlay_boxes(jpeg_bytes, boxes)
    send_alert(cfg, cam, jpg, boxes, track_smoke)


def run_legacy(cfg, model):
    """Pre-v3 score-only polling (MOTION_ENABLED=false): detect every poll at
    SCORE_THRESHOLD, MIN_HITS/HITS_WINDOW accumulation, cooldown after alert."""
    s0 = S()
    s0.api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    s0.fetch_to = _clampf(_getf(cfg, "MOTION_FETCH_TIMEOUT_S", 10.0), 0.2, 30, 10.0)
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    interval = max(1.0, _getf(cfg, "POLL_INTERVAL_S", 15))
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    min_hits = max(1, _geti(cfg, "MIN_HITS", 3))
    window = max(min_hits, _geti(cfg, "HITS_WINDOW", 0))
    cooldown = max(0.0, _getf(cfg, "COOLDOWN_S", 300))
    track_smoke = _getb(cfg, "TRACK_SMOKE", False)
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    enabled = _getb(cfg, "ENABLED", True)
    heartbeat_s = max(60.0, _getf(cfg, "HEARTBEAT_S", 300))
    state = {c: {"recent": collections.deque(maxlen=window),
                 "last_alert": 0.0, "down": False}
             for c in cameras}
    LOG(f"legacy started: {len(cameras)} cameras every {interval}s, "
        f"min_hits={min_hits}/{window}, cooldown={cooldown:g}s, smoke={track_smoke}")
    last_hb = time.monotonic()
    sweep = 0
    while True:
        if _STOP[0]:
            break
        if not enabled:
            LOG("disabled (ENABLED=false) - sleeping 60s")
            _bounded_sleep(60)
            continue
        sweep += 1
        up = fire_cams = 0
        for cam in cameras:
            st = state[cam]
            try:
                raw = fetch_frame(s0.api, cam, timeout=s0.fetch_to)
                img = _decode(raw)
                dets = model.detect(img, score_thresh=threshold, allowed=allowed)
                up += 1
                if st["down"]:
                    LOG(f"{cam}: back online")
                    st["down"] = False
                prev_hits = sum(st["recent"])
                st["recent"].append(1 if dets else 0)
                hits = sum(st["recent"])
                now = time.monotonic()
                in_cd = now - st["last_alert"] < cooldown
                if dets:
                    fire_cams += 1
                    best = max(d["score"] for d in dets)
                    will_alert = hits >= min_hits and not in_cd
                    store_fire_frame(cfg, cam, raw, dets, alerted=will_alert)
                    if will_alert:
                        st["last_alert"] = now
                        st["recent"].clear()
                        LOG(f"{cam}: ALERT ({hits} fire in last {window} polls, "
                            f"conf {best:.2f})")
                        jpg = overlay_boxes(raw, dets)
                        send_alert(cfg, cam, jpg, dets, track_smoke)
                    else:
                        LOG(f"{cam}: fire-hit {hits}/{min_hits} in last "
                            f"{window} polls (conf {best:.2f})"
                            + (" [cooldown]" if in_cd else ""))
                elif hits == 0 and prev_hits:
                    LOG(f"{cam}: cleared after {prev_hits} recent hits")
            except urllib.error.HTTPError as exc:
                st["recent"].clear()
                if not st["down"]:
                    st["down"] = True
                    LOG(f"{cam}: HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001
                st["recent"].clear()
                if not st["down"]:
                    LOG(f"{cam}: error: {exc}")
                    st["down"] = True
        now = time.monotonic()
        if now - last_hb >= heartbeat_s:
            last_hb = now
            LOG(f"heartbeat: {up}/{len(cameras)} cams OK, {fire_cams} fire-marked "
                f"(sweep #{sweep})")
        _bounded_sleep(interval)
    _close_store()
    LOG("firewatch stopped (legacy)")


def _bounded_sleep(secs):
    """Sleep up to `secs` in small slices, returning early on SIGTERM/SIGINT."""
    end = time.monotonic() + max(0.0, secs)
    while not _STOP[0]:
        left = end - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(0.25, left))


def _do_sweep(s, cfg, model, state):
    """Batched two-frame motion sweep over all cameras (spec 4/5)."""
    track_smoke = "smoke" in s.allowed
    # --- 1) fetch frame A for every camera (short timeout) ---
    a_start = time.monotonic()
    aframes, tA = {}, {}
    for cam in s.cameras:
        st = state[cam]
        if st["mode"] == "FOLLOW":
            continue  # FOLLOW cameras are densely sampled, not double-swept
        try:
            aframes[cam] = fetch_frame(s.api, cam, timeout=s.fetch_to)
            tA[cam] = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - per-camera isolation
            _cam_err(cam, s, st, exc)
    # --- 2) sleep the rest of MOTION_GAP_S once (not once per camera) ---
    elapsed = time.monotonic() - a_start
    if s.gap_s > elapsed:
        _bounded_sleep(s.gap_s - elapsed)
    if _STOP[0]:
        return
    # --- 3) fetch frame B for every camera that gave an A (no in-sweep retry) ---
    bframes, tB = {}, {}
    for cam in aframes:
        st = state[cam]
        try:
            bframes[cam] = fetch_frame(s.api, cam, timeout=s.fetch_to)
            tB[cam] = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            _cam_err(cam, s, st, exc)
    # --- 4) per camera: gap sanity -> motion -> model or skip ---
    now = time.monotonic()
    for cam in bframes:
        st = state[cam]
        _cam_ok(cam, s, st)
        gap = tB[cam] - tA[cam]
        gap_ok = s.gap_min <= gap <= s.gap_max
        b_bytes = bframes[cam]
        try:
            mask, mean = _motion_of(s, aframes[cam], b_bytes)
            if mask is None:  # resolution changed -> unusable this cycle
                gap_ok = False
            frac = float(mask.mean()) if mask is not None else 0.0
            motion = gap_ok and s.frame_frac <= frac <= s.frame_frac_max
            baseline_due = (now - st["last_model"]) >= s.baseline_every
            if not (motion or baseline_due):
                if gap_ok and mask is not None:
                    st["mask_ring"].append(mask)
                continue  # SKIP MODEL (spec 5.3) - log counted below by heartbeat
            st["last_model"] = now
            img = _decode(b_bytes)
            dets = model.detect(img, score_thresh=s.floor, allowed=s.allowed)
            # a global change (> FRAME_FRAC_MAX) is NOT usable motion, even on a
            # baseline run (spec 5.3) - only local change can grant the bonus
            cur_mask = mask if (gap_ok and frac <= s.frame_frac_max) else None
            marked, hit, _n, n_static = _classify_dets(s, dets, cur_mask, st)
            if gap_ok and mask is not None:
                st["mask_ring"].append(mask)
            if marked:
                _store(cfg, cam, b_bytes, marked, alerted=False)
                best = best_of(marked)
                if hit:
                    if st["mode"] == "IDLE":
                        _promote_follow(cam, s, st, now)
                        LOG(f"{cam}: hit -> FOLLOW-UP (conf {best.get('eff', best['score']):.2f})")
                    else:
                        LOG(f"{cam}: hit during "
                            + ("COOL" if st["mode"] == "COOL" else "FOLLOW")
                            + f" (conf {best.get('eff', best['score']):.2f}, stored)")
                else:
                    LOG(f"{cam}: stored {len(marked)} box(es) "
                        f"eff {best.get('eff', best['score']):.2f}, no hit "
                        f"({n_static} static)")
        except Exception as exc:  # noqa: BLE001 - isolate the whole camera turn
            _cam_err(cam, s, st, exc)


def _promote_follow(cam, s, st, now):
    st["mode"] = "FOLLOW"
    st["follow_until"] = now + s.followup_max_s
    st["hits"].clear()
    st["hits"].append(1)          # the promoting hit already counts as sample 1
    st["empty_run"] = 0
    st["next_follow"] = now + s.followup_gap


def _do_follow_sample(cam, s, cfg, model, st, state):  # noqa: ARG001
    """One dense FOLLOW-UP sample: A/B pair -> model always -> count confirms."""
    track_smoke = "smoke" in s.allowed
    now = time.monotonic()
    if st["mode"] != "FOLLOW":
        return
    try:
        a_bytes = fetch_frame(s.api, cam, timeout=s.fetch_to)
        tA = time.monotonic()
        _bounded_sleep(s.gap_s)
        if _STOP[0]:
            return
        b_bytes = fetch_frame(s.api, cam, timeout=s.fetch_to)
        tB = time.monotonic()
        _cam_ok(cam, s, st)
        gap = tB - tA
        gap_ok = s.gap_min <= gap <= s.gap_max
        mask, _mean = _motion_of(s, a_bytes, b_bytes)
        if mask is None:
            gap_ok = False
        frac = float(mask.mean()) if mask is not None else 0.0
        motion = gap_ok and s.frame_frac <= frac <= s.frame_frac_max
        cur_mask = mask if (gap_ok and frac <= s.frame_frac_max) else None
        st["last_model"] = time.monotonic()
        img = _decode(b_bytes)
        dets = model.detect(img, score_thresh=s.floor, allowed=s.allowed)
        marked, hit, _n, _ns = _classify_dets(s, dets, cur_mask, st)
        if gap_ok and mask is not None:
            st["mask_ring"].append(mask)
        now = time.monotonic()
        st["hits"].append(1 if hit else 0)
        st["empty_run"] = 0 if marked else st["empty_run"] + 1
        if marked:
            will_alert = sum(st["hits"]) >= s.min_hits
            _store(cfg, cam, b_bytes, marked, alerted=will_alert)
            best = best_of(marked)
            LOG(f"{cam}: follow conf {best.get('eff', best['score']):.2f} "
                f"hits {sum(st['hits'])}/{s.min_hits}")
            if will_alert:
                _send_alert_frame(cfg, cam, s, b_bytes, marked, track_smoke)
                _end_follow(cam, s, st, cfg, True, now)
                return
        # no alert yet: keep sampling until timeout or the signal dies
        if now >= st["follow_until"] or st["empty_run"] >= s.death_turns:
            _end_follow(cam, s, st, cfg, False, now)
            return
        st["next_follow"] = time.monotonic() + s.followup_gap
    except Exception as exc:  # noqa: BLE001 - per-camera isolation
        _cam_err(cam, s, st, exc)
        st["next_follow"] = time.monotonic() + s.followup_gap


def run_forever(cfg, model):
    api0 = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    if not cameras:
        LOG("ERROR: CAMERAS list empty - exiting")
        sys.exit(2)
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    s = _resolve_settings(cfg, cameras, threshold)
    s.api = api0
    if not s.motion_enabled:
        run_legacy(cfg, model)
        return
    enabled = _getb(cfg, "ENABLED", True)
    state = _new_state(s, cameras)
    signal.signal(signal.SIGTERM, _sig_stop)
    signal.signal(signal.SIGINT, _sig_stop)

    LOG(f"started (v3): {len(cameras)} cameras, sweep {s.poll_interval:g}s, "
        f"gap {s.gap_s:g}s, floor {s.floor:g}, bonus {s.bonus:g} @ {s.bar:g} / "
        f"no-motion >= {s.high:g}, follow every {s.followup_gap:g}s, "
        f"min_hits {s.min_hits}/{s.window}, cooldown {s.cooldown:g}s / "
        f"noalert {s.cooldown_noalert:g}s, baseline {s.baseline_every:g}s, "
        f"heartbeat {s.heartbeat_s:g}s")

    next_sweep = time.monotonic()
    last_hb = time.monotonic()
    sweep = 0
    try:
        while not _STOP[0]:
            now = time.monotonic()
            if not enabled:
                LOG("disabled (ENABLED=false) - sleeping 60s")
                _bounded_sleep(60)
                continue
            # --- state transitions ---
            for cam in cameras:
                st = state[cam]
                if st["mode"] == "COOL" and now >= st["cool_until"]:
                    st["mode"] = "IDLE"
                elif st["mode"] == "FOLLOW" and now >= st["follow_until"]:
                    _end_follow(cam, s, st, cfg, False, now)
            # --- sweep when due ---
            if time.monotonic() >= next_sweep:
                sweep += 1
                _do_sweep(s, cfg, model, state)
                next_sweep = time.monotonic() + s.poll_interval
            # --- dense follow-up samples due between sweeps ---
            now = time.monotonic()
            for cam in cameras:
                st = state[cam]
                if st["mode"] == "FOLLOW" and now >= st["next_follow"]:
                    _do_follow_sample(cam, s, cfg, model, st, state)
                    if _STOP[0]:
                        break
            # --- heartbeat / liveness ---
            now = time.monotonic()
            if now - last_hb >= s.heartbeat_s:
                last_hb = now
                modes = {}
                for c in cameras:
                    m = state[c]["mode"]
                    modes[m] = modes.get(m, 0) + 1
                LOG(f"heartbeat: modes={modes} (sweep #{sweep}, "
                    f"every {s.poll_interval:g}s)")
            # --- bounded sleep until the next due event (never busy-spin) ---
            nxt = next_sweep
            for cam in cameras:
                st = state[cam]
                if st["mode"] == "FOLLOW" and st["next_follow"] < nxt:
                    nxt = st["next_follow"]
            wait = max(s.tick_min, min(1.0, nxt - time.monotonic()))
            _bounded_sleep(wait)
    finally:
        _close_store()
        LOG("firewatch stopped (v3)")


# ---------------------------------------------------------------------------
def main():
    args = set(sys.argv[1:])
    raw = _raw_conf(CONF_PATH)
    if not raw:
        LOG(f"ERROR: no config at {CONF_PATH} - aborting")
        return 2

    # global socket backstop (spec 13.A): no call can ever block forever
    try:
        socket.setdefaulttimeout(30)
    except (OSError, ValueError):
        pass

    telegram_path = os.environ.get("TELEGRAM_CONF", _get(raw, "TELEGRAM_CONF",
                                                         "/config/telegram.conf"))
    cfg = tg.load_conf(telegram_path)
    cfg.update(raw)   # merge firewatch tunables so helpers read camera config too

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
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        LOG(f"ERROR: {exc}")
        return 2

    run_forever(cfg, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
