#!/usr/bin/env python3
"""firewatch.py - farm fire/smoke watchdog for the Frigate NVR stack.

Runs as the `firewatch` Docker service (docker-compose.yml). It polls each
camera's ALREADY-DECODED detect frame from Frigate's REST API
(GET /api/<camera>/latest.jpg - no extra ffmpeg decode), runs a small
fire/smoke YOLO model via OpenVINO, and sends a Telegram PHOTO alert on a
sustained detection (min consecutive hits above a score threshold, with a
per-camera cooldown to prevent spam).

Evidence store (2026-09-08, see plans/firewatch-evidence-store.md; sqlite rework
in plans/firewatch-sqlite-evidence-store.md): EVERY poll that is marked fire (a
detection above SCORE_THRESHOLD) is ALSO persisted - the frame as a JPEG under
<STORE_DIR>/<cam>/ and its metadata (camera, timestamp, best score, per-box
detections, the JPEG path) in a single WAL-mode SQLite DB at
<STORE_DIR>/firewatch.db - INDEPENDENT of the MIN_HITS/HITS_WINDOW alert gate
and cooldown, so a real fire that never accumulates enough hits still leaves a
durable record. This replaces the old per-frame .json sidecar. STORE_DIR
defaults to /media/firewatch (a read-write mount of the git-ignored host
./media tree added in docker-compose.yml); STORE_ENABLED=false reverts to the
old send-only behavior.

Design notes
------------
- Frigate's own person/car/animal detection (config/config.yaml) is NOT
  touched; this watcher is fully out-of-band (see plans/fire-detection.md).
- firewatch.py lives in ./firewatch (the service folder); code/config/model
  live on runtime mounts (./firewatch, ./scripts, ./config, ./models are
  mounted read-only into the container) so edits need no image rebuild.
- The watcher is stdlib + openvino + numpy + Pillow only. telegram_notify is
  imported from the shared ./scripts directory (mounted at /scripts), which
  still ships alongside because firewatch.py and scripts/ are siblings.
- Motion gate + score bonus (2026-09-09, plans/firewatch-motion-gate-bonus.md):
  firewatch now combines the fire model's confidence with MOTION measured inside
  each detection's fire box (frame diff between consecutive polls of the same
  camera). A box must be moving to count toward MIN_HITS (the gate - static
  sun/glint false positives are suppressed) and a moving box gets a score bonus
  (promotes small flickering fires over the alert bar). MOTION_ENABLED=false
  restores the previous score-only behaviour exactly.

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
import collections
import io
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request

import numpy as np

# firewatch.py now lives in firewatch/, while the shared helpers (telegram_notify.py)
# stay in the sibling scripts/ dir. Make both importable: in the container firewatch.py
# is mounted at /firewatch and scripts/ at /scripts (repo-root siblings), so the same
# relative walk works here and locally.
_FW_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _FW_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_FW_DIR), "scripts"))
import telegram_notify as tg  # noqa: E402

# Pillow is only used to overlay detection boxes on the alert snapshot; it is
# a hard dependency of the firewatch image (see firewatch/requirements.txt).
from PIL import Image, ImageDraw, ImageFont

CONF_PATH = os.environ.get("FIREWATCH_CONF", "/config/firewatch.conf")
LOG = lambda *a: print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------
# tiny KEY=VALUE conf reader with typed defaults (mirrors telegram_notify.load_conf)
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

        # class name -> index from labelmap (e.g. fire=0, other=1, smoke=2).
        # labelmap_path is optional: when omitted, labelmap.txt inside model_dir
        # is used (the container passes only MODEL_DIR).
        self.labels = []
        labelmap = labelmap_path or os.path.join(model_dir, "labelmap.txt")
        if labelmap and os.path.isfile(labelmap):
            with open(labelmap, encoding="utf-8") as fh:
                self.labels = [ln.strip() for ln in fh if ln.strip()]
        if not self.labels:
            # fallback guess: first two outputs after xywh are fire/smoke
            self.labels = ["fire", "smoke"]
        self.nc = len(self.labels)
        LOG(f"model {xml_path}: input {self.width}x{self.height} ch={self.ch} "
            f"output {list(self._output.shape)} classes={self.labels}")

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
        # orient so rows are candidate detections: [N, C]
        if data.ndim == 2 and data.shape[0] == 4 + self.nc and data.shape[1] > data.shape[0]:
            data = data.T
        cols = data.shape[1]
        raw_cols = 4 + self.nc
        if cols == raw_cols:
            mode = "raw"   # per-class scores with cxcywh boxes ([1,4+nc,N] or [1,N,4+nc])
        elif cols == 6:
            mode = "e2e"   # end-to-end NMS: x1,y1,x2,y2,score,class_id ([1,N,6]) - YOLO26
        else:
            raise RuntimeError(
                f"model output has {cols} cols; expected {raw_cols} (raw per-class) "
                f"or 6 (end-to-end NMS). See models/fire/README.md"
            )

        dets = []
        if mode == "raw":
            boxes = data[:, :4].astype(np.float32)  # cxcywh in input pixels
            scores = data[:, 4 : 4 + self.nc].astype(np.float32)
            # ultralytics exports may leave class logits raw - sigmoid when needed
            if np.nanmax(scores) > 1.0:
                scores = 1.0 / (1.0 + np.exp(-scores))
            for ci in range(self.nc):
                # allowed is a lowercase name set; compare case-insensitively so
                # models with labels like "Fire" still match the "fire" track.
                if allowed and self.labels[ci].lower() not in allowed:
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
        else:  # e2e: rows are already x1,y1,x2,y2 (input pixels) + score + class id
            xyxy = data[:, :4].astype(np.float32)
            confs = data[:, 4].astype(np.float32)
            cids = np.clip(data[:, 5].astype(np.int64), 0, self.nc - 1)
            for i in np.where(confs >= score_thresh)[0]:
                label = self.labels[int(cids[i])]
                if allowed and label.lower() not in allowed:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                dets.append(
                    {
                        "label": label,
                        "score": float(confs[i]),
                        "box": (max(0.0, (x1 - pad_x) / scale),
                                max(0.0, (y1 - pad_y) / scale),
                                (x2 - pad_x) / scale,
                                (y2 - pad_y) / scale),
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
# motion gate + score bonus (2026-09-09, plans/firewatch-motion-gate-bonus.md)
# ---------------------------------------------------------------------------
def _box_motion_frac(mask, box, margin=0.5):
    """Fraction of 'changed' (True) pixels inside `box`, expanded on each side
    by `margin` (a fraction of the box's own width/height), clipped to the
    frame. Returns 0.0 for a zero/one-pixel box or an out-of-frame region."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return 0.0
    ax1 = max(0, int(x1 - w * margin))
    ay1 = max(0, int(y1 - h * margin))
    ax2 = min(mask.shape[1], int(x2 + w * margin) + 1)
    ay2 = min(mask.shape[0], int(y2 + h * margin) + 1)
    region = mask[ay1:ay2, ax1:ax2]
    return float(region.mean()) if region.size else 0.0


def motion_gate(dets, cur_gray, prev_gray, threshold, gate_frac, margin,
                px_diff, bonus, no_motion_score):
    """Apply the motion gate + score bonus to one poll's raw model detections.

    A box's `motion_frac` = fraction of pixels that changed between the previous
    poll's frame and this one, measured inside the box (expanded by `margin`);
    `motion_present` = motion_frac >= gate_frac. Its effective score
    `eff = min(1, raw + (bonus if motion_present else 0))` - the BONUS.

    Returns (marked, hit, n_candidates, n_static):
      marked       detections (raw 'score' kept for store/overlay, plus 'eff' and
                   'motion' keys) whose effective score >= threshold. These are
                   the fire-marked frames to persist/overlay. Static boxes with
                   raw >= threshold land here too (kept as evidence for review)
                   but do NOT count as hits.
      hit          True when any box crossed the ALERT gate this poll (motion
                   present and eff >= threshold, OR raw >= no_motion_score) ->
                   contributes 1 to the MIN_HITS/HITS_WINDOW accumulator.
      n_candidates number of raw detections the model returned (>= SCORE_FLOOR).
      n_static     number of marked-but-static boxes (evidence kept, gate held).

    First poll after start/error has no previous frame (prev_gray None or a
    different shape): no motion can be judged, so the OLD score-only rule is used
    for that single poll (hit when raw >= threshold). Such a poll cannot alert
    alone (MIN_HITS >= 3), so a static false positive can never accumulate.
    """
    no_baseline = prev_gray is None or prev_gray.shape != cur_gray.shape
    mask = None
    if not no_baseline:
        mask = (np.abs(cur_gray.astype(np.int16)
                       - prev_gray.astype(np.int16)) > px_diff)
    marked, hit = [], False
    n_static = 0
    for d in dets:
        raw = d["score"]
        box = dict(d)
        if no_baseline:
            # old score-only rule on the (no-baseline) first poll: no motion can
            # be judged, so hit exactly when the raw score clears the bar.
            box["eff"], box["motion"] = raw, False
            if raw >= threshold:
                marked.append(box)
                hit = True
            continue
        moving = _box_motion_frac(mask, d["box"], margin) >= gate_frac
        eff = min(1.0, raw + (bonus if moving else 0.0))
        box["eff"] = eff
        box["motion"] = moving
        if eff >= threshold:
            marked.append(box)
            if moving or raw >= no_motion_score:
                hit = True
            else:
                n_static += 1  # static evidence: stored but gated out of alerts
        # else: below the bar and not moving -> ignored entirely
    return marked, hit, len(dets), n_static


def motion_tag(enabled, marked):
    """Short motion/effective-score tag appended to hit/alert log lines."""
    if not enabled or not marked:
        return ""
    moving = sum(1 for d in marked if d.get("motion"))
    eff = max(d.get("eff", d["score"]) for d in marked)
    return f" [motion {moving}/{len(marked)}, eff {eff:.2f}]"


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def build_caption(camera, dets, track_smoke):
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    top = "🔥 FIRE ALERT" if any(d["label"].lower() == "fire" for d in dets) else "⚠️ FIRE WATCH"
    if not track_smoke and all(d["label"].lower() == "smoke" for d in dets):
        top = "💨 SMOKE WATCH"
    lines = [f"<b>{top}</b>", f"<b>Camera:</b> {tg.esc_html(camera)}",
             f"<b>Time:</b> {tg.esc_html(ts)}"]
    for d in sorted(dets, key=lambda x: x["score"], reverse=True)[:5]:
        lines.append(f"• {tg.esc_html(d['label'])} {d['score']:.2f}")
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
                # Evidence store: persist every fire-marked frame (JPEG file +
                # SQLite metadata row) even with no hit accumulation or in a
                # single-pass run. Single-pass mode alerts on every fire-marked
                # frame, so the stored frame is tagged alerted.
                store_fire_frame(cfg, cam, raw, dets, alerted=True)
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
        tg.send_telegram_photo(cfg, jpg, text)
        LOG(f"ALERT sent for {cam}")
    except Exception as exc:  # noqa: BLE001
        LOG(f"ALERT FAILED for {cam}: {exc}")


# ---------------------------------------------------------------------------
# SQLite evidence store (2026-09-08 sqlite rework, see
# plans/firewatch-sqlite-evidence-store.md). The JPEG frame stays a plain file
# on disk; only the metadata that used to live in a per-frame .json sidecar
# (camera, timestamp, best score, threshold, per-box detections, image PATH)
# goes into one WAL-mode SQLite DB - no image BLOBs (user decision).
# ---------------------------------------------------------------------------
_STORE_CONN = None          # cached sqlite3 connection (loop is single-threaded)
_STORE_LAST_PRUNE = 0.0     # throttle for the optional retention sweep

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera          TEXT    NOT NULL,
    captured_at     REAL    NOT NULL,   -- UTC epoch seconds (time.time())
    ts_utc          TEXT    NOT NULL,   -- UTC 'YYYY-MM-DD HH:MM:SS' (readable)
    jpg_path        TEXT    NOT NULL,   -- absolute path of the stored JPEG
    best_score      REAL    NOT NULL,
    score_threshold REAL    NOT NULL,
    alerted         INTEGER NOT NULL DEFAULT 0  -- 1 = this poll also produced a Telegram alert
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
    """Absolute path of the evidence SQLite DB (WAL mode)."""
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return None
    return os.path.join(store_dir, _get(cfg, "STORE_DB", "firewatch.db"))


def _store_conn(cfg):
    """Return the cached WAL-mode evidence connection, creating schema once.

    Re-opens transparently if the DB file is missing/pruned (e.g. the host
    media cleanup removed firewatch.db mid-run) - the next poll recreates the
    file + schema.
    """
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
        conn = sqlite3.connect(db_path)
        # WAL: readers never block this single writer and each commit does far
        # fewer fsyncs than the rollback journal - right for a high-frequency
        # fire-marked-frame append log. synchronous=NORMAL matches (durable to
        # process/OS crash, safe on the journaled host filesystem).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        # Migration (2026-09-08, portal change): `frames` gained `alerted`.
        # CREATE TABLE IF NOT EXISTS does NOT alter an existing table, so add
        # the column for DBs created before this change. Idempotent - safe to
        # run on every open (duplicate-column error is swallowed).
        try:
            conn.execute(
                "ALTER TABLE frames ADD COLUMN "
                "alerted INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.Error:
            pass  # column already present (fresh DB or migrated earlier)
        _STORE_CONN = conn
    return _STORE_CONN


def _prune_store(cfg, conn):
    """Optional retention: STORE_RETENTION_DAYS > 0 deletes old frame records.

    Runs at most once a minute. Frames are cascade-deleted WITH their detection
    rows. Only DB rows are removed - the JPEG files stay (the host media
    cleanup owns file purging) and the DB keeps its high-water size until a
    manual VACUUM (see plans/firewatch-sqlite-evidence-store.md).
    """
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
    """Best-effort persist a fire-marked frame: JPEG file + SQLite metadata.

    Runs on EVERY poll where a fire/smoke detection exists above SCORE_THRESHOLD
    (dets non-empty), INDEPENDENT of the MIN_HITS/HITS_WINDOW alert gate and the
    per-camera cooldown - so a real fire that never accumulates enough hits (or
    keeps re-hitting inside cooldown) still leaves a durable record on disk.
    `alerted` (param) tags whether THIS poll also crossed the alert gate and
    produced a Telegram alert (the caller evaluates the gate BEFORE storing); a
    downstream portal uses it to separate real alerts from raw fire evidence.

    Writes, per fire-marked frame:
      <STORE_DIR>/<cam>/YYYYMMDD_HHMMSS_<ms>_<cam>_conf<best>.jpg   (raw frame;
      the image itself stays on disk - only its PATH goes into SQLite)
      plus one `frames` row (camera, timestamps, jpg_path, best score, score
      threshold, alerted) and one `detections` row per box in the WAL-mode DB at
      <STORE_DIR>/<STORE_DB|firewatch.db> - the old .json sidecar is gone. See
      plans/firewatch-sqlite-evidence-store.md. Never raises (keeps the loop
      alive); failures are logged and skipped.
    """
    if not _getb(cfg, "STORE_ENABLED", True):
        return
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return
    try:
        # 1) image file - same bytes sent to Telegram; kept on disk (no BLOB
        #    columns in the DB, per user decision).
        cam_dir = os.path.join(store_dir, cam)
        os.makedirs(cam_dir, exist_ok=True)
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
        best = max(d["score"] for d in dets)
        base = f"{stamp}_{int(now % 1 * 1000):03d}_{cam}_conf{best:.2f}"
        jpg_path = os.path.join(cam_dir, base + ".jpg")
        with open(jpg_path, "wb") as fh:
            fh.write(jpeg_bytes)

        # 2) metadata (the old .json sidecar) -> SQLite (WAL). JPEG first so a
        #    DB hiccup never leaves a row pointing at a missing file.
        conn = _store_conn(cfg)
        if conn is None:
            LOG(f"{cam}: store SKIPPED (no STORE_DIR/STORE_DB path)")
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO frames (camera, captured_at, ts_utc, jpg_path, "
            "best_score, score_threshold, alerted) VALUES (?,?,?,?,?,?,?)",
            (cam, now,
             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
             jpg_path, round(best, 4),
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
             for d in sorted(dets, key=lambda x: x["score"], reverse=True)],
        )
        conn.commit()
        _prune_store(cfg, conn)
        LOG(f"{cam}: stored fire frame #{frame_id} "
            f"{os.path.basename(jpg_path)} (conf {best:.2f}, "
            f"{len(jpeg_bytes) // 1024} KB, {len(dets)} detections)")
    except Exception as exc:  # noqa: BLE001 - evidence store must never kill loop
        global _STORE_CONN
        _STORE_CONN = None  # drop a possibly-stale conn; next poll reopens it
        LOG(f"{cam}: store FAILED: {exc}")


def run_forever(cfg, model):
    api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    interval = _getf(cfg, "POLL_INTERVAL_S", 15)
    threshold = _getf(cfg, "SCORE_THRESHOLD", 0.5)
    min_hits = max(1, _geti(cfg, "MIN_HITS", 3))
    cooldown = _getf(cfg, "COOLDOWN_S", 300)
    track_smoke = _getb(cfg, "TRACK_SMOKE", False)
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    # Motion gate + score bonus (2026-09-09, plans/firewatch-motion-gate-bonus.md):
    # combine the fire score with motion inside each detection's fire box (frame
    # diff between polls). MOTION_ENABLED=false -> previous score-only behaviour.
    motion_enabled = _getb(cfg, "MOTION_ENABLED", False)
    floor = _getf(cfg, "SCORE_FLOOR", threshold)  # raw detect floor (<= threshold)
    gate_frac = _getf(cfg, "MOTION_GATE_FRAC", 0.05)
    margin = _getf(cfg, "MOTION_MARGIN", 0.5)
    px_diff = _getf(cfg, "MOTION_PIXEL_DIFF", 15)
    bonus = _getf(cfg, "MOTION_BONUS", 0.15)
    no_motion_score = _getf(cfg, "SCORE_NO_MOTION", 0.85)
    enabled = _getb(cfg, "ENABLED", True)
    # Liveness heartbeat (2026-09-09, plans/firewatch-heartbeat-logging.md): a
    # healthy watcher with nothing on fire logs NOTHING between events, so it is
    # indistinguishable from a dead/hung one in `docker logs`. Every HEARTBEAT_S
    # seconds print one summary line (cams OK / fire-marked cams this sweep) to
    # prove the loop is alive. Default 300s, clamped to >=60s to prevent spam.
    heartbeat_s = max(60.0, _getf(cfg, "HEARTBEAT_S", 300))
    # Sliding-window gate (2026-09-06 fix, see plans/fire-detection.md): alert
    # when the last `window` polls contain >= MIN_HITS fire/smoke hits.
    # window == MIN_HITS means strictly-consecutive (old behavior); window >
    # MIN_HITS tolerates the intermittent single-frame misses a close-up fire
    # causes (auto-exposure/white-balance swings dip the conf below threshold on
    # some polls), so a real fire still alerts instead of resetting forever.
    window = max(min_hits, _geti(cfg, "HITS_WINDOW", 0))
    if not cameras:
        LOG("ERROR: CAMERAS list empty - exiting")
        sys.exit(2)

    state = {c: {"recent": collections.deque(maxlen=window),
                 "last_alert": 0.0, "down": False, "prev": None}
             for c in cameras}
    mot = ""
    if motion_enabled:
        mot = (f", motion=gate+bonus (floor {floor:g}, gate {gate_frac:g}, "
               f"margin {margin:g}, px {px_diff:g}, bonus {bonus:g}, "
               f"no_motion>={no_motion_score:g})")
    LOG(f"started: {len(cameras)} cameras, sweep every {interval}s, "
        f"min_hits={min_hits} in window {window}, cooldown={cooldown}s, "
        f"smoke={track_smoke}, enabled={enabled}, heartbeat={heartbeat_s:g}s"
        + mot)
    sweep = 0
    last_hb = time.monotonic()
    while True:
        if not enabled:
            LOG("disabled (ENABLED=false) - sleeping 60s")
            time.sleep(60)
            continue
        sweep += 1
        up = 0
        fire_cams = 0
        for cam in cameras:
            st = state[cam]
            try:
                raw = fetch_frame(api, cam)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                # Motion gate needs the grayscale frame diffed against the last
                # successful poll's frame for THIS camera (see motion_gate).
                cur_gray = np.asarray(img.convert("L"), dtype=np.uint8)
                if motion_enabled:
                    dets = model.detect(img, score_thresh=floor, allowed=allowed)
                    marked, hit, n_cands, n_static = motion_gate(
                        dets, cur_gray, st["prev"], threshold,
                        gate_frac, margin, px_diff, bonus, no_motion_score)
                else:
                    dets = model.detect(img, score_thresh=threshold,
                                        allowed=allowed)
                    marked, hit, n_cands, n_static = dets, bool(dets), 0, 0
                st["prev"] = cur_gray   # baseline for the NEXT poll's diff
                up += 1
                if st["down"]:
                    LOG(f"{cam}: back online")
                    st["down"] = False
                prev_hits = sum(st["recent"])
                st["recent"].append(1 if hit else 0)
                hits = sum(st["recent"])
                now = time.monotonic()
                in_cooldown = now - st["last_alert"] < cooldown
                if hit:
                    fire_cams += 1
                    best = max(d["score"] for d in marked)
                    will_alert = hits >= min_hits and not in_cooldown
                    # Evidence store: persist EVERY fire-marked frame (JPEG +
                    # SQLite metadata) independent of the MIN_HITS alert gate /
                    # cooldown (a real fire below the accumulation bar still
                    # leaves a durable record). The gate is evaluated BEFORE
                    # storing so the frame's `alerted` column reflects whether
                    # THIS poll also crossed the gate and produced a Telegram
                    # alert (the portal uses alerted vs raw evidence frames).
                    store_fire_frame(cfg, cam, raw, marked, alerted=will_alert)
                    if will_alert:
                        st["last_alert"] = now
                        st["recent"].clear()
                        LOG(f"{cam}: ALERT ({hits} fire in last {window} "
                            f"polls, conf {best:.2f})"
                            + motion_tag(motion_enabled, marked))
                        jpg = overlay_boxes(raw, marked)
                        send_alert(cfg, cam, jpg, marked, track_smoke)
                    else:
                        LOG(f"{cam}: fire-hit {hits}/{min_hits} in last "
                            f"{window} polls (conf {best:.2f})"
                            + motion_tag(motion_enabled, marked)
                            + (" [cooldown]" if in_cooldown else ""))
                elif marked:
                    # Fire-marked evidence but NO box crossed the motion gate
                    # this poll (static lookalike in the 0.5-0.85 FP band):
                    # keep the durable evidence record (alerted=0) and say so.
                    fire_cams += 1
                    best = max(d["score"] for d in marked)
                    store_fire_frame(cfg, cam, raw, marked, alerted=False)
                    LOG(f"{cam}: stored {len(marked)} fire box(es) "
                        f"conf {best:.2f}, NO motion - not counted "
                        f"({n_static} static)")
                elif n_cands:
                    # Raw candidates existed but none crossed the alert bar
                    # (static and/or below threshold) - no hit this poll.
                    LOG(f"{cam}: {n_cands} fire candidate(s) below the "
                        f"alert bar (no hit)")
                elif hits == 0 and prev_hits:
                    # window drained to zero: the fire has been gone long enough
                    LOG(f"{cam}: cleared after {prev_hits} recent hits")
            except urllib.error.HTTPError as exc:
                st["recent"].clear()
                st["prev"] = None
                if not st["down"]:
                    # Mark down + log once per episode (not every 15s sweep),
                    # mirroring the generic-error branch below - so a long
                    # Frigate outage can't flood the log, and the next
                    # successful poll logs "back online".
                    st["down"] = True
                    LOG(f"{cam}: HTTP {exc.code}")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                st["recent"].clear()
                st["prev"] = None
                if not st["down"]:
                    LOG(f"{cam}: error: {exc}")
                    st["down"] = True
        now = time.monotonic()
        if now - last_hb >= heartbeat_s:
            last_hb = now
            LOG(f"heartbeat: {up}/{len(cameras)} cams OK, "
                f"{fire_cams} fire-marked camera(s) this sweep "
                f"(sweep #{sweep}, every {interval:g}s)")
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
    cfg = tg.load_conf(telegram_path)
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
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        LOG(f"ERROR: {exc}")
        return 2

    run_forever(cfg, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
