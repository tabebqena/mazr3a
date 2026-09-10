#!/usr/bin/env python3
"""scenewatch.py - farm scene-description watchdog for the Frigate NVR stack.

Runs as the `scenewatch` Docker service (docker-compose.yml). It mirrors the
firewatch motion sweep (firewatch/firewatch.py `_do_sweep`):

  1. fetch frame A for every camera from Frigate's REST API
     (GET /api/<cam>/latest.jpg - the already-decoded detect frame, ~0 extra
     decode),
  2. sleep the remainder of MOTION_GAP_S (must be > 1 s) once,
  3. fetch frame B for the same cameras,
  4. diff A/B into a denoised motion mask per camera,

and when a camera's motion gate passes (or its periodic baseline is due) it
captions frame B with a small vision-language model and stores a one-line
scene description in a WAL-mode SQLite DB.

The model is loaded ONCE at startup and stays resident in RAM for the life of
the process - it is never reloaded per caption (see models/scene/README.md).

Design notes:
  * MODEL CHOICE IS CONSTRAINED BY THE RUNTIME. `openvino_genai.VLMPipeline`
    implements a closed list of VLM architectures - llava, qwen2_vl,
    qwen2_5_vl, gemma3, minicpm, phi3_v, phi4mm - verified by inspecting the
    symbols in libopenvino_genai.so. SmolVLM (model_type "smolvlm"/"idefics3")
    is NOT among them, so **Qwen2-VL-2B-Instruct** is used: the smallest VLM
    this runtime can load (int4, ~1.76 GB on disk). Any model in that list
    works - only MODEL_DIR changes.
  * Runtime = OpenVINO GenAI on **CPU**. The iGPU is left entirely to
    Frigate's OpenVINO detector (config.yaml), which already saturates it.
    INFERENCE_NUM_THREADS caps the cores the captioner may use.
  * CPU is bounded by RATE, not only by model size: the motion gate +
    per-camera CAPTION_COOLDOWN_S + BASELINE_EVERY_S mean a handful of
    captions per hour rather than a continuous stream. Between captions the
    process is idle (no polling of the model).
  * Each caption is scored 0-100 into a coarse `tier` (high/normal/low) so the
    portal can show the few that matter by default and reveal the rest on
    demand - see score_importance().
  * Bounded + exception-isolated (firewatch spec 13): the while-loop is the
    only unbounded loop; every fetch/caption has a hard timeout; every
    per-camera operation is isolated at the camera boundary; timers use
    time.monotonic(); SIGTERM/SIGINT set a stop flag checked each tick; the
    scene store is best-effort and never stops the loop.

Usage:
  python scenewatch.py            # run the sweep loop
  python scenewatch.py --once     # single sweep over all cameras, store, exit
  python scenewatch.py --dry-run  # like --once but describe WITHOUT storing
  python scenewatch.py --check    # validate config + load model + one caption
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
from PIL import Image, ImageFilter

CONF_PATH = os.environ.get("SCENEWATCH_CONF", "/config/scenewatch.conf")
LOG = lambda *a: print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *a, flush=True)  # noqa: E731
_STOP = [False]  # set by SIGTERM/SIGINT -> clean, prompt shutdown

DEFAULT_PROMPT = ("Describe this CCTV scene in one short sentence. Mention "
                  "people, vehicles, animals or notable activity. If nothing "
                  "is happening, say so.")

# ---------------------------------------------------------------------------
# IMPORTANCE annotation (writer-side, see score_importance below).
#
# A camera captioned every few minutes for a day produces thousands of rows,
# most of them dull. Rather than run a second model, each caption is scored
# 0-100 from signals we ALREADY have, then bucketed into a coarse `tier` so the
# portal can show the few that matter by default and reveal the rest on demand:
#
#   * TEXT    - does the caption mention something notable (person, vehicle,
#               animal, fire...) or explicitly say the scene is empty?
#   * MOTION  - how much actually moved, saturating at MOTION_IMPORTANCE_SCALE.
#   * UNUSUAL - motion far above THIS camera's own recent median: a spike on an
#               otherwise quiet camera is more interesting than the same amount
#               of motion on an always-busy one.
#   * NOVELTY - a caption identical to the camera's previous one inside
#               NOVELTY_WINDOW_S is demoted, so a bird that sits in frame gets
#               one interesting row instead of one per sweep.
#
# tier = high (>= TIER_HIGH_MIN) | normal (>= TIER_NORMAL_MIN) | low.
# Terms/thresholds are all config-overridable (config/scenewatch.conf).
# ---------------------------------------------------------------------------
DEFAULT_IMPORTANT_TERMS = (
    "person,people,man,woman,men,women,child,children,boy,girl,human,worker,"
    "crowd,vehicle,car,truck,van,motorcycle,bike,bicycle,tractor,animal,cow,"
    "sheep,goat,dog,cat,horse,camel,bird,chicken,fire,smoke,flame")
DEFAULT_LOW_TERMS = (
    "nothing,no one,no-one,nobody,no people,no vehicles,no activity,"
    "no movement,empty,quiet,still,calm,blank,unchanged,undisturbed,unclear,"
    "not clear,low light")


def _norm_text(text):
    """Lowercase + collapse to alphanumerics - for repeat/novelty detection."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in (text or "").lower())
    return " ".join(cleaned.split())


def _sig_stop(signum, frame):  # noqa: ARG001 - signal handler
    _STOP[0] = True


# ---------------------------------------------------------------------------
# tiny KEY=VALUE conf reader with typed defaults (same parser as firewatch)
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
# frame fetch + numpy-only motion mask (ported from firewatch)
# ---------------------------------------------------------------------------
def fetch_frame(api, camera, timeout=1.0):
    """Return jpeg_bytes for a camera's latest detect frame (bounded by timeout)."""
    url = f"{api}/api/{camera}/latest.jpg"
    req = urllib.request.Request(url, headers={"Accept": "image/jpeg"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


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


def _motion_of(s, a_bytes, b_bytes):
    """(mask, mean_luma) for a valid A/B pair (None mask if the shape changed)."""
    a = _gray(_decode(a_bytes))
    b = _gray(_decode(b_bytes))
    if a.shape != b.shape:
        return None, float(b.mean())
    px = _adaptive_px(s.px_diff, s.night_extra, s.night_luma, float(b.mean()))
    mask = _changed_mask(a, b, px, s.blur_radius)
    return mask, float(b.mean())


# ---------------------------------------------------------------------------
# VLM captioner (OpenVINO GenAI, resident in RAM)
# ---------------------------------------------------------------------------
def _result_text(res):
    """Normalize a VLMPipeline result to a plain string.

    openvino_genai returns a GenerationResult (has `.texts`); some versions
    hand back a bare str. Handle both.
    """
    texts = getattr(res, "texts", None)
    if texts:
        return str(texts[0]).strip()
    return str(res).strip()


class Captioner:
    """Loads an OpenVINO GenAI VLM pipeline once and captions PIL images.

    MODEL_DIR must hold a VLM export whose architecture the runtime implements
    (Qwen2-VL-2B int4 by default - see the module docstring for the supported
    list and why SmolVLM cannot be used).
    """

    def __init__(self, model_dir, device="CPU", threads=4, prompt=None,
                 max_new_tokens=64):
        from openvino import Core  # noqa: F401 - imported for a clear error
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        model_dir = str(model_dir).rstrip("/")
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise RuntimeError(
                f"no OpenVINO VLM at {model_dir} (config.json missing) - run "
                "dev_scripts/prep_scene_model.sh and point MODEL_DIR at it")
        self.model_dir = model_dir
        self.model_name = os.path.basename(model_dir) or model_dir
        self.device = str(device or "CPU").upper()
        self.prompt = prompt or DEFAULT_PROMPT
        self.max_new_tokens = max(1, int(max_new_tokens))
        props = {}
        threads = int(threads)
        if self.device.startswith("CPU") and threads > 0:
            # Bound the CPU the captioner may take so Frigate decode/detect
            # is not starved (the host has 8 cores).
            props["INFERENCE_NUM_THREADS"] = str(threads)
        self._max_threads = threads
        # Declared here so the attribute is statically known (close() sets it
        # back to None on shutdown).
        self.pipe = None
        try:
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device, props)
        except TypeError:
            # Older/newer bindings may not accept a properties mapping.
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device)

    def caption(self, img_rgb):
        """Return (text, latency_ms) for one PIL RGB image."""
        import openvino as ov
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("captioner is closed")
        tensor = ov.Tensor(np.asarray(img_rgb.convert("RGB"), dtype=np.uint8))
        gen = ov_genai.GenerationConfig()
        gen.max_new_tokens = self.max_new_tokens
        started = time.monotonic()
        res = None
        # The image kwarg is `image` in current bindings and `images` in some
        # others; try both before falling back to the positional form.
        for kwargs in ({"image": tensor, "generation_config": gen},
                       {"images": tensor, "generation_config": gen},
                       {"image": tensor, "max_new_tokens": self.max_new_tokens},
                       {"images": tensor, "max_new_tokens": self.max_new_tokens}):
            try:
                res = pipe.generate(self.prompt, **kwargs)
                break
            except TypeError:
                continue
        if res is None:
            res = pipe.generate(self.prompt, tensor, gen)
        latency_ms = int((time.monotonic() - started) * 1000)
        return _result_text(res), latency_ms

    def close(self):
        try:
            self.pipe = None
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# SQLite scene store (WAL). Descriptions are tiny rows; JPEGs are optional.
# ---------------------------------------------------------------------------
_STORE_CONN = None
_STORE_LAST_PRUNE = 0.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera      TEXT    NOT NULL,
    captured_at REAL    NOT NULL,
    ts_utc      TEXT    NOT NULL,
    description TEXT    NOT NULL,
    motion_frac REAL    NOT NULL DEFAULT 0,
    reason      TEXT    NOT NULL DEFAULT 'motion',
    model       TEXT    NOT NULL DEFAULT '',
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    jpg_path    TEXT,
    importance  INTEGER NOT NULL DEFAULT 0,
    tier        TEXT    NOT NULL DEFAULT 'normal'
);
CREATE INDEX IF NOT EXISTS idx_scenes_camera_ts ON scenes(camera, captured_at);
CREATE INDEX IF NOT EXISTS idx_scenes_ts        ON scenes(captured_at);
"""
# NOTE: the (tier, importance) index is created AFTER the guarded migration in
# _store_conn(), never here: on a legacy DB (table exists without those columns)
# an index over them would make executescript() raise "no such column: tier"
# BEFORE the ALTER statements could add them.
_SCHEMA_TIER_INDEX = ("CREATE INDEX IF NOT EXISTS idx_scenes_tier "
                      "ON scenes(tier, importance)")


def _store_db_path(cfg):
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return None
    return os.path.join(store_dir, _get(cfg, "STORE_DB", "scenewatch.db"))


def _store_conn(cfg):
    """Return the cached WAL-mode scene connection (schema created on demand)."""
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
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        # Guarded idempotent migrations for DBs created before importance
        # existed. Defaults keep legacy rows visible ('normal' tier, score 0).
        for _col, ddl in (
            ("importance", "ALTER TABLE scenes ADD COLUMN "
                           "importance INTEGER NOT NULL DEFAULT 0"),
            ("tier", "ALTER TABLE scenes ADD COLUMN "
                     "tier TEXT NOT NULL DEFAULT 'normal'"),
        ):
            try:
                conn.execute(ddl)
            except sqlite3.Error:
                pass  # column already present
        # ...and only now the index that depends on them.
        try:
            conn.execute(_SCHEMA_TIER_INDEX)
        except sqlite3.Error:
            pass
        _STORE_CONN = conn
    return _STORE_CONN


def _prune_store(cfg, conn):
    """Optional retention: STORE_RETENTION_DAYS > 0 prunes old scene rows."""
    global _STORE_LAST_PRUNE
    days = _getf(cfg, "STORE_RETENTION_DAYS", 0)
    if days <= 0:
        return
    now = time.time()
    if now - _STORE_LAST_PRUNE < 60:
        return
    _STORE_LAST_PRUNE = now
    cur = conn.execute("DELETE FROM scenes WHERE captured_at < ?",
                       (now - days * 86400,))
    if cur.rowcount:
        LOG(f"store: pruned {cur.rowcount} scene record(s) older than {days:g} days")


def _list_conf(cfg, key, default):
    """Comma-separated conf value -> lowercase list (empty entries dropped)."""
    return [t.strip().lower() for t in _get(cfg, key, default).split(",") if t.strip()]


def score_importance(s, description, reason, motion_frac, recent_fracs, repeat):
    """Heuristic (0-100 score, tier) for one caption - no extra inference.

    A pure function of signals already in hand, so it adds no CPU cost and is
    easy to reason about and tune. See the IMPORTANCE comment block near the
    top; the tier cut-offs are TIER_HIGH_MIN / TIER_NORMAL_MIN.
    """
    text = (description or "").lower()
    notable = any(t in text for t in s.important_terms)
    dull = any(t in text for t in s.low_terms)
    score = 0.0
    if notable:
        score += s.important_bonus
    elif dull:
        # Only penalise an explicit "nothing here" if nothing notable matched.
        score -= s.low_penalty
    # How much actually moved (saturating: past the scale it stops adding).
    if s.motion_scale > 0:
        score += s.motion_bonus * min(1.0,
                                      max(0.0, float(motion_frac)) / s.motion_scale)
    # A spike relative to this camera's own recent norm.
    if reason == "motion" and recent_fracs:
        ordered = sorted(recent_fracs)
        median = ordered[len(ordered) // 2]
        if median > 0 and float(motion_frac) > s.unusual_factor * median:
            score += s.unusual_bonus
    if repeat:
        score -= s.novelty_penalty
    score = max(0.0, min(100.0, score))
    value = int(round(score))
    tier = ("high" if value >= s.tier_high_min
            else "normal" if value >= s.tier_normal_min else "low")
    return value, tier


def store_scene(cfg, cam, jpeg_bytes, description, motion_frac, reason,
                model_name, latency_ms, importance=0, tier="normal"):
    """Best-effort persist one description (+ optional JPEG). Never raises."""
    global _STORE_CONN
    if not _getb(cfg, "STORE_ENABLED", True):
        return
    store_dir = _get(cfg, "STORE_DIR", "")
    if not store_dir:
        return
    try:
        now = time.time()
        jpg_path = None
        if _getb(cfg, "STORE_IMAGES", False):
            cam_dir = os.path.join(store_dir, cam)
            os.makedirs(cam_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
            base = f"{stamp}_{int(now % 1 * 1000):03d}_{cam}"
            jpg_path = os.path.join(cam_dir, base + ".jpg")
            with open(jpg_path, "wb") as fh:
                fh.write(jpeg_bytes)

        conn = _store_conn(cfg)
        if conn is None:
            LOG(f"{cam}: store SKIPPED (no STORE_DIR/STORE_DB path)")
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO scenes (camera, captured_at, ts_utc, description, "
            "motion_frac, reason, model, latency_ms, jpg_path, importance, "
            "tier) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cam, now,
             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now)),
             description, round(float(motion_frac), 6), reason,
             model_name, int(latency_ms), jpg_path,
             int(importance), str(tier)),
        )
        conn.commit()
        _prune_store(cfg, conn)
        LOG(f"{cam}: stored scene #{cur.lastrowid} "
            f"({reason}, {tier} {importance})")
    except Exception as exc:  # noqa: BLE001 - store must never kill the loop
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
# resolved settings + per-camera state
# ---------------------------------------------------------------------------
class S:
    """Resolved, clamped tunables (no global drift)."""

    def __init__(self):
        # Defaults; every one is overwritten by _resolve_settings(). Declaring
        # them here keeps the attributes statically known and documents them.
        self.api = ""
        self.cameras = []
        self.poll_interval = 15.0
        self.gap_s = 1.5
        self.gap_min = 0.5
        self.gap_max = 5.0
        self.fetch_to = 1.0
        self.blur_radius = 1.0
        self.px_diff = 18.0
        self.night_luma = 25.0
        self.night_extra = 8.0
        self.frame_frac = 0.0015
        self.frame_frac_max = 0.20
        self.baseline_every = 600.0
        self.caption_cooldown = 120.0
        self.store_enabled = True
        self.tick_min = 0.25
        self.heartbeat_s = 300.0
        # Importance scoring (score_importance): text + motion + novelty.
        self.important_terms = []
        self.low_terms = []
        # Calibration: IMPORTANT_BONUS alone reaches TIER_HIGH_MIN, so "the
        # caption mentions something worth looking at" is by itself enough to
        # be high; motion and novelty only adjust from there.
        self.important_bonus = 60.0
        self.low_penalty = 30.0
        self.motion_bonus = 30.0
        self.motion_scale = 0.05
        self.unusual_factor = 3.0
        self.unusual_bonus = 10.0
        self.novelty_penalty = 20.0
        self.novelty_window_s = 1800.0
        self.tier_high_min = 60
        self.tier_normal_min = 30
        self.motion_history = 20


def _resolve_settings(cfg, cameras):
    s = S()
    s.api = _get(cfg, "FRIGATE_API", "http://frigate:5000").rstrip("/")
    s.cameras = cameras
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
    s.frame_frac_max = _clampf(_getf(cfg, "MOTION_FRAME_FRAC_MAX", 0.20),
                               s.frame_frac, 1.0, 0.20)
    s.baseline_every = max(10.0, _getf(cfg, "BASELINE_EVERY_S", 600))
    s.caption_cooldown = max(0.0, _getf(cfg, "CAPTION_COOLDOWN_S", 120))
    s.store_enabled = _getb(cfg, "STORE_ENABLED", True)
    s.tick_min = 0.25
    s.heartbeat_s = max(30.0, _getf(cfg, "HEARTBEAT_S", 300))
    # --- importance scoring (see score_importance) ---
    s.important_terms = _list_conf(cfg, "IMPORTANT_TERMS", DEFAULT_IMPORTANT_TERMS)
    s.low_terms = _list_conf(cfg, "LOW_TERMS", DEFAULT_LOW_TERMS)
    s.important_bonus = _clampf(_getf(cfg, "IMPORTANT_BONUS", 60), 0, 100, 60)
    s.low_penalty = _clampf(_getf(cfg, "LOW_PENALTY", 30), 0, 100, 30)
    s.motion_bonus = _clampf(_getf(cfg, "MOTION_IMPORTANCE_BONUS", 30), 0, 100, 30)
    s.motion_scale = max(1e-6, _getf(cfg, "MOTION_IMPORTANCE_SCALE", 0.05))
    s.unusual_factor = max(1.0, _getf(cfg, "MOTION_UNUSUAL_FACTOR", 3.0))
    s.unusual_bonus = _clampf(_getf(cfg, "MOTION_UNUSUAL_BONUS", 10), 0, 100, 10)
    s.novelty_penalty = _clampf(_getf(cfg, "NOVELTY_PENALTY", 20), 0, 100, 20)
    s.novelty_window_s = max(0.0, _getf(cfg, "NOVELTY_WINDOW_S", 1800))
    s.tier_high_min = int(_clampf(_getf(cfg, "TIER_HIGH_MIN", 60), 1, 100, 60))
    s.tier_normal_min = int(_clampf(_getf(cfg, "TIER_NORMAL_MIN", 30), 0,
                                    s.tier_high_min, 30))
    s.motion_history = max(1, _geti(cfg, "MOTION_HISTORY", 20))
    return s


def _new_state(s, cameras):
    st = {}
    for c in cameras:
        st[c] = {
            "down": False,
            "last_caption": 0.0,   # monotonic of the last caption attempt
            "cool_until": 0.0,     # monotonic cooldown end
            "captions": 0,         # captions taken (this process)
            "fracs": collections.deque(maxlen=max(1, s.motion_history)),
            "last_desc": None,     # normalized previous caption (novelty)
            "last_desc_t": 0.0,
            "err_count": 0,
            "_err_logged": False,
        }
    return st


def _cam_err(cam, s, st, exc):
    """Log a camera error once per episode; camera goes DOWN."""
    st["down"] = True
    if not st.get("_err_logged"):
        st["_err_logged"] = True
        LOG(f"{cam}: error: {exc}")
    st["err_count"] = st.get("err_count", 0) + 1


def _cam_ok(cam, s, st):
    if st["down"]:
        LOG(f"{cam}: back online")
    st["down"] = False
    st["_err_logged"] = False


def _bounded_sleep(secs):
    """Sleep up to `secs` in small slices, returning early on SIGTERM/SIGINT."""
    end = time.monotonic() + max(0.0, secs)
    while not _STOP[0]:
        left = end - time.monotonic()
        if left <= 0:
            break
        time.sleep(min(0.25, left))


def _mem_available_mb():
    """Host MemAvailable in MiB (0 if unreadable). Informational only."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


# ---------------------------------------------------------------------------
# the sweep: two frames >1 s apart -> motion -> caption the moving camera
# ---------------------------------------------------------------------------
def _do_sweep(s, cfg, cap, state, dry=False):
    """Batched two-frame motion sweep over all cameras, then caption movers."""
    # --- 1) fetch frame A for every camera (short timeout) ---
    a_start = time.monotonic()
    aframes, tA = {}, {}
    for cam in s.cameras:
        st = state[cam]
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
        return {"captioned": 0, "skipped": len(s.cameras)}
    # --- 3) fetch frame B for every camera that gave an A ---
    bframes, tB = {}, {}
    for cam in aframes:
        st = state[cam]
        try:
            bframes[cam] = fetch_frame(s.api, cam, timeout=s.fetch_to)
            tB[cam] = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            _cam_err(cam, s, st, exc)
    # --- 4) per camera: gap sanity -> motion -> caption or skip ---
    now = time.monotonic()
    captioned = skipped = 0
    for cam in bframes:
        st = state[cam]
        _cam_ok(cam, s, st)
        gap = tB[cam] - tA[cam]
        gap_ok = s.gap_min <= gap <= s.gap_max
        b_bytes = bframes[cam]
        try:
            mask, _mean = _motion_of(s, aframes[cam], b_bytes)
            if mask is None:  # resolution changed -> unusable this cycle
                gap_ok = False
            frac = float(mask.mean()) if mask is not None else 0.0
            motion = gap_ok and s.frame_frac <= frac <= s.frame_frac_max
            baseline_due = (now - st["last_caption"]) >= s.baseline_every
            if not (motion or baseline_due):
                skipped += 1
                continue
            # Cooldown only silences MOTION captions - a baseline run is
            # already rate-limited by BASELINE_EVERY_S.
            if motion and now < st["cool_until"]:
                skipped += 1
                continue
            reason = "motion" if motion else "baseline"
            text, ms = cap.caption(_decode(b_bytes))
            st["last_caption"] = time.monotonic()
            st["cool_until"] = st["last_caption"] + s.caption_cooldown
            st["captions"] += 1
            captioned += 1
            # Importance (score_importance). A caption identical to this
            # camera's previous one inside NOVELTY_WINDOW_S is demoted, so a
            # bird sitting in frame yields one interesting row, not one per
            # sweep.
            norm = _norm_text(text)
            repeat = (norm == st["last_desc"]
                      and (time.time() - st["last_desc_t"]) < s.novelty_window_s)
            importance, tier = score_importance(
                s, text, reason, frac, list(st["fracs"]), repeat)
            st["last_desc"], st["last_desc_t"] = norm, time.time()
            if reason == "motion":
                st["fracs"].append(float(frac))
            LOG(f"{cam}: [{reason} frac {frac:.4f} {tier} {importance}] "
                f"{text} ({ms} ms)")
            if s.store_enabled and not dry:
                store_scene(cfg, cam, b_bytes, text, frac, reason,
                            cap.model_name, ms, importance, tier)
        except Exception as exc:  # noqa: BLE001 - isolate the whole camera turn
            _cam_err(cam, s, st, exc)
    return {"captioned": captioned, "skipped": skipped}


def run_once(cfg, cap, dry=False):
    """One sweep over all cameras, then exit (used by --once / --dry-run)."""
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    if not cameras:
        LOG("ERROR: CAMERAS list empty")
        return False
    s = _resolve_settings(cfg, cameras)
    if dry:
        s.store_enabled = False
    state = _new_state(s, cameras)
    LOG(f"sweep: {len(cameras)} cameras, gap {s.gap_s:g}s, "
        f"gate {s.frame_frac:g}..{s.frame_frac_max:g}, "
        f"cooldown {s.caption_cooldown:g}s, baseline {s.baseline_every:g}s"
        + (" [dry-run: not storing]" if dry else ""))
    res = _do_sweep(s, cfg, cap, state, dry=dry)
    LOG(f"sweep done: {res['captioned']} captioned, {res['skipped']} skipped")
    return True


def run_forever(cfg, cap):
    cameras = [c.strip() for c in _get(cfg, "CAMERAS", "").split(",") if c.strip()]
    if not cameras:
        LOG("ERROR: CAMERAS list empty - exiting")
        sys.exit(2)
    s = _resolve_settings(cfg, cameras)
    enabled = _getb(cfg, "ENABLED", True)
    state = _new_state(s, cameras)
    signal.signal(signal.SIGTERM, _sig_stop)
    signal.signal(signal.SIGINT, _sig_stop)

    LOG(f"started: {len(cameras)} cameras, sweep {s.poll_interval:g}s, "
        f"gap {s.gap_s:g}s, gate {s.frame_frac:g}..{s.frame_frac_max:g}, "
        f"cooldown {s.caption_cooldown:g}s, baseline {s.baseline_every:g}s, "
        f"device {cap.device}"
        + (f", threads {cap._max_threads}" if cap.device.startswith("CPU") else "")
        + f", model {cap.model_name}, heartbeat {s.heartbeat_s:g}s")

    next_sweep = time.monotonic()
    last_hb = time.monotonic()
    sweep_no = 0
    try:
        while not _STOP[0]:
            if not enabled:
                LOG("disabled (ENABLED=false) - sleeping 60s")
                _bounded_sleep(60)
                continue
            if time.monotonic() >= next_sweep:
                sweep_no += 1
                _do_sweep(s, cfg, cap, state)
                next_sweep = time.monotonic() + s.poll_interval
            now = time.monotonic()
            if now - last_hb >= s.heartbeat_s:
                last_hb = now
                captioned = sum(state[c]["captions"] for c in cameras)
                down = [c for c in cameras if state[c]["down"]]
                LOG(f"heartbeat: sweep #{sweep_no}, captions {captioned}, "
                    f"down {len(down)}{('/' + ','.join(down)) if down else ''}, "
                    f"mem_available {_mem_available_mb()} MiB")
            wait = max(s.tick_min, min(1.0, next_sweep - time.monotonic()))
            _bounded_sleep(wait)
    finally:
        _close_store()
        cap.close()
        LOG("scenewatch stopped")


# ---------------------------------------------------------------------------
def main():
    args = set(sys.argv[1:])
    raw = _raw_conf(CONF_PATH)
    if not raw:
        LOG(f"ERROR: no config at {CONF_PATH} - aborting")
        return 2

    # global socket backstop: no call can ever block forever
    try:
        socket.setdefaulttimeout(30)
    except (OSError, ValueError):
        pass

    model_dir = _get(raw, "MODEL_DIR", "/models/scene")
    try:
        cap = Captioner(
            model_dir,
            device=_get(raw, "MODEL_DEVICE", "CPU"),
            threads=_geti(raw, "INFERENCE_NUM_THREADS", 4),
            prompt=_get(raw, "CAPTION_PROMPT", DEFAULT_PROMPT),
            max_new_tokens=_geti(raw, "CAPTION_MAX_NEW_TOKENS", 48),
        )
    except RuntimeError as exc:
        LOG(f"ERROR: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - import/runtime failures -> exit 2
        LOG(f"ERROR: cannot load the scene model: {exc}")
        return 2

    if "--check" in args:
        # Prove the pipeline actually generates, not just that it loads.
        try:
            probe = Image.new("RGB", (640, 360), (128, 128, 128))
            text, ms = cap.caption(probe)
            LOG(f"config + model OK ({cap.model_name} on {cap.device}, "
                f"probe caption {ms} ms): {text}")
        except Exception as exc:  # noqa: BLE001
            LOG(f"ERROR: model loaded but captioning failed: {exc}")
            return 1
        return 0

    if "--dry-run" in args or "--once" in args:
        try:
            ok = run_once(raw, cap, dry="--dry-run" in args)
        except Exception as exc:  # noqa: BLE001
            LOG(f"ERROR: {exc}")
            return 1
        return 0 if ok else 1

    run_forever(raw, cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
