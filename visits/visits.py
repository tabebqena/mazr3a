#!/usr/bin/env python3
"""visits.py - per-camera person-visit log for the farm Frigate stack.

WHAT IT DOES (see plans/visit-log.md)
--------------------------------------
Turns Frigate's raw person events into ONE line per visit, per camera:

    Estraha (cam01)  Field West  person #2  14:32-14:36  4 min

It does NOT re-detect, does NOT use a VLM, and does NOT read the Frigate SQLite
DB: Frigate is queried through its REST API only (`/api/events`). Frigate frames
are read IN PLACE from the mounted media tree (`<cam>-<event_id>-clean.webp`) to
crop the person for a cheap clothing-colour histogram.

HOW IDENTITY WORKS
------------------
Per camera we keep a set of OPEN PRESENCES. Each event is classified from its
Frigate path + final box as in_place / stopped / lost (or exited when clipped on
the left/right/top edge). A new event joins an open presence by a WEIGHTED SCORE
over time gap, exit->entry spatial continuity, clothing histogram and (only when
both fragments have trustworthy motion) direction. State-specific resume windows
implement A1-A6: continuous/turn, move-stop-move (180 s), stop-and-end, in-place
(one visit for the whole presence), and brief occlusion (60 s). Direction is
never a hard gate. Ambiguous candidates stay SEPARATE and are flagged in
`visit_links` / `visits.needs_review`.

A presence is RETIRED after `INACTIVITY_S` with no person events (4 h by
default -> the overnight gap closes it, so each morning starts a fresh person #).

LOAD REDUCTION (the "drop until count" gate)
--------------------------------------------
"Significant" is decided from metadata only (Frigate's own trajectory span
`data.path_data` -> `motion_disp`, and whether the box touches a frame edge) -
NO image work. Significant events are analysed immediately (crop -> histogram).
INSIGNIFICANT events are buffered and NOT cropped/histogrammed until
`INSIGNIFICANT_COUNT_THRESHOLD` of them have accumulated for a camera, then the
whole batch is analysed once. A visit whose whole presence never really moved
(and never touched an edge) is stored with `significant=0` and left out of the
text log.

Everything derived is REBUILDABLE from the cached `events` table, so changing a
threshold costs no re-capture and no re-crop (raw paths and histograms are cached
per event).

Usage:
  python visits.py                 # run the scheduler loop
  python visits.py --check         # config/API/store probe
  python visits.py --once          # one poll/analyse/rebuild, exit
  python visits.py --backfill      # re-fetch lookback and rebuild path features
  python visits.py --list [--day YYYY-MM-DD] [--all]
  python visits.py --status
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# defaults (overridden by config/visits.conf, then by same-named env vars)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "ENABLED": "true",
    "FRIGATE_API": "http://frigate:5000",
    "FRIGATE_MEDIA_DIR": "/media",
    "CLIPS_SUBDIR": "clips",
    "STORE_DIR": "/media/visits",
    "STORE_DB": "visits.db",
    "PLACES_CONF": "/config/places.conf",
    "POLL_EVERY_S": "60",
    "HEARTBEAT_S": "300",
    "MERGE_ENGINE": "v2",
    "MERGE_GAP_S": "30",
    "INACTIVITY_S": "14400",
    "ANGLE_TOL_DEG": "45",
    "MOTION_MIN_DISP": "0.05",
    "EDGE_MARGIN": "0.05",
    "EXIT_MARGIN": "0.02",
    "INSIGNIFICANT_COUNT_THRESHOLD": "5",
    "MIN_DURATION_S": "8",
    "HIST_THRESHOLD": "0.6",
    "MIN_CROP_PX": "16",
    "VISITS_RETENTION_DAYS": "7",
    "ONLY_PERSON": "true",
    "INITIAL_LOOKBACK_S": "604800",
    "RESCAN_OVERLAP_S": "300",
    "FETCH_LIMIT": "1000",
    "TZ_OFFSET_H": "0",
    # --- v2 state-aware merge -------------------------------------------------
    # State-specific resume windows. A5 stationary presences may resume for the
    # whole visit inactivity window; A3 stops are limited to STOP_GAP_S.
    "CONT_GAP_S": "30",
    "STOP_GAP_S": "180",
    "OCCLUSION_GAP_S": "60",
    "STATIONARY_GAP_S": "14400",
    # Motion classification / direction trust.
    "DETECT_ASPECT": "0.5625",          # detect height / width (640x360)
    "MOVING_DISP": "0.05",
    "DIR_MIN_DISP": "0.05",
    "DIR_MIN_STRAIGHTNESS": "0.5",
    "TAIL_STATIONARY_DIST": "0.02",
    "TAIL_STATIONARY_MIN_GAP_S": "2",
    # Spatial tolerances (aspect-corrected frame-width units).
    "CONT_RADIUS": "0.08",
    "STOP_RADIUS": "0.06",
    "OCCLUSION_RADIUS": "0.15",
    "IN_PLACE_RADIUS": "0.05",
    # Appearance gates.
    "HIST_MIN": "0.60",
    "HIST_MISSING_TIGHT_SPACE": "0.80",
    "HIST_MISSING_MAX_GAP_S": "10",
    # Decision thresholds / score weights.
    "MERGE_SCORE": "0.70",
    "AMBIG_SCORE": "0.50",
    "ASSIGN_MARGIN": "0.05",
    "W_TIME": "0.25",
    "W_SPACE": "0.40",
    "W_HIST": "0.30",
    "W_DIR": "0.05",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    frigate_event_id TEXT PRIMARY KEY,
    camera           TEXT NOT NULL,
    start_time       REAL NOT NULL DEFAULT 0,
    end_time         REAL,
    zones            TEXT NOT NULL DEFAULT '[]',
    box              TEXT,
    motion_disp      REAL NOT NULL DEFAULT 0,
    angle            REAL,
    entry_xy         TEXT,
    exit_xy          TEXT,
    path_json        TEXT,
    end_xy           TEXT,
    net_disp         REAL NOT NULL DEFAULT 0,
    path_len         REAL NOT NULL DEFAULT 0,
    straightness     REAL,
    head_angle       REAL,
    tail_angle       REAL,
    tail_gap_s       REAL,
    tail_stationary  INTEGER NOT NULL DEFAULT 0,
    dir_trusted      INTEGER NOT NULL DEFAULT 0,
    edge_exit        INTEGER NOT NULL DEFAULT 0,
    speed            REAL,
    vel_xy           TEXT,
    state            TEXT,
    significant      INTEGER NOT NULL DEFAULT 0,
    analyzed         INTEGER NOT NULL DEFAULT 0,
    hist_ok          INTEGER NOT NULL DEFAULT 0,
    hist             BLOB,
    created_at       REAL NOT NULL DEFAULT 0,
    updated_at       REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_cam_ts   ON events(camera, start_time);
CREATE INDEX IF NOT EXISTS idx_events_analyzed ON events(analyzed);

CREATE TABLE IF NOT EXISTS visits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera        TEXT NOT NULL,
    day           TEXT NOT NULL,
    display_name  TEXT,
    place         TEXT,
    person_no     INTEGER NOT NULL,
    enter_time    REAL NOT NULL,
    leave_time    REAL NOT NULL,
    duration_s    REAL NOT NULL,
    n_events      INTEGER NOT NULL DEFAULT 0,
    rep_event_id  TEXT,
    moved         INTEGER NOT NULL DEFAULT 0,
    edge          INTEGER NOT NULL DEFAULT 0,
    significant   INTEGER NOT NULL DEFAULT 1,
    state         TEXT,
    confidence    REAL,
    needs_review  INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT,
    first_event_id TEXT,
    last_event_id  TEXT,
    created_at    REAL NOT NULL DEFAULT 0,
    updated_at    REAL NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_visits_key ON visits(camera, enter_time, person_no);
CREATE INDEX IF NOT EXISTS idx_visits_day ON visits(day, camera, enter_time);

CREATE TABLE IF NOT EXISTS visit_events (
    visit_id         INTEGER NOT NULL,
    frigate_event_id TEXT NOT NULL,
    PRIMARY KEY (visit_id, frigate_event_id)
);
CREATE INDEX IF NOT EXISTS idx_visit_events_e ON visit_events(frigate_event_id);

CREATE TABLE IF NOT EXISTS visit_links (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera        TEXT NOT NULL,
    prev_event_id TEXT NOT NULL,
    next_event_id TEXT NOT NULL,
    dt            REAL,
    dist          REAL,
    hist_sim      REAL,
    angle_delta   REAL,
    score         REAL,
    decision      TEXT NOT NULL,
    reason        TEXT,
    created_at    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_visit_links_e ON visit_links(camera, next_event_id);
CREATE INDEX IF NOT EXISTS idx_visit_links_d ON visit_links(decision);
"""

# Columns added after the first store version. `CREATE TABLE IF NOT EXISTS`
# never alters an existing table, so open_store() applies these idempotently.
EVENT_EXTRA_COLUMNS = {
    "path_json": "TEXT",
    "end_xy": "TEXT",
    "net_disp": "REAL NOT NULL DEFAULT 0",
    "path_len": "REAL NOT NULL DEFAULT 0",
    "straightness": "REAL",
    "head_angle": "REAL",
    "tail_angle": "REAL",
    "tail_gap_s": "REAL",
    "tail_stationary": "INTEGER NOT NULL DEFAULT 0",
    "dir_trusted": "INTEGER NOT NULL DEFAULT 0",
    "edge_exit": "INTEGER NOT NULL DEFAULT 0",
    "speed": "REAL",
    "vel_xy": "TEXT",
    "state": "TEXT",
}
VISIT_EXTRA_COLUMNS = {
    "state": "TEXT",
    "confidence": "REAL",
    "needs_review": "INTEGER NOT NULL DEFAULT 0",
    "review_reason": "TEXT",
    "first_event_id": "TEXT",
    "last_event_id": "TEXT",
}


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
def _log(msg):
    print("[{}] {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_conf(path):
    """Parse a KEY=VALUE file (blank lines and # comments skipped)."""
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except OSError:
        return {}
    return cfg


class Settings:
    def __init__(self, cfg):
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS})
        # environment wins (same name), matching the other services' convention
        for key in list(DEFAULTS):
            if os.environ.get(key):
                merged[key] = os.environ[key]
        self.raw = merged

    def s(self, key):
        return str(self.raw.get(key, DEFAULTS.get(key, ""))).strip()

    def i(self, key):
        try:
            return int(float(self.s(key)))
        except ValueError:
            return int(float(DEFAULTS[key]))

    def f(self, key):
        try:
            return float(self.s(key))
        except ValueError:
            return float(DEFAULTS[key])

    def b(self, key):
        return self.s(key).lower() in ("1", "true", "yes", "on")

    @property
    def db_path(self):
        return os.path.join(self.s("STORE_DIR"), self.s("STORE_DB"))


# ---------------------------------------------------------------------------
# places (the human layer; same shape as config/places.conf)
# ---------------------------------------------------------------------------
def _pairs(value):
    out = {}
    for item in (value or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, val = item.partition("=")
        if key.strip():
            out[key.strip().lower()] = val.strip()
    return out


class Places:
    def __init__(self, camera_places, zone_places):
        self.camera_places = camera_places
        self.zone_places = zone_places

    @classmethod
    def load(cls, path):
        cfg = load_conf(path)
        return cls(_pairs(cfg.get("CAMERA_PLACES")), _pairs(cfg.get("ZONE_PLACES")))

    def display_name(self, camera):
        return self.camera_places.get((camera or "").lower(), camera or "")

    def resolve(self, camera, zones):
        cam = (camera or "").strip().lower()
        for zone in zones or ():
            hit = self.zone_places.get(cam + "." + str(zone).strip().lower())
            if hit:
                return hit
        return self.camera_places.get(cam) or (camera or "").strip() or cam


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def _migrate_store(conn):
    """Add columns introduced after the first store version (idempotent)."""
    for table, columns in (("events", EVENT_EXTRA_COLUMNS),
                           ("visits", VISIT_EXTRA_COLUMNS)):
        present = {row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)}
        for name, decl in columns.items():
            if name not in present:
                conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, decl))
    conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_review"
                 " ON visits(needs_review, enter_time)")


def open_store(path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    _migrate_store(conn)
    conn.commit()
    return conn


def utc_day(epoch):
    return datetime.fromtimestamp(
        float(epoch), timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Frigate REST (NO database access)
# ---------------------------------------------------------------------------
def fetch_person_events(api, after, limit=1000, only_person=True):
    """All person events with start_time >= after, via the REST API only.

    Frigate returns events NEWEST-FIRST, so a page that fills `limit` is paged
    BACKWARD with `before`=oldest of the page (paging forward by max would jump
    to the newest and skip the middle - a real bug during the 7-day backfill).
    """
    api = api.rstrip("/")
    out = []
    before = None
    for _ in range(200):  # pagination safety
        params = {
            "after": "{:.3f}".format(float(after)),
            "limit": int(limit),
            "include_thumbnails": 0,
        }
        if before is not None:
            params["before"] = "{:.3f}".format(before)
        if only_person:
            params["label"] = "person"
        url = api + "/api/events?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            batch = json.loads(resp.read().decode("utf-8", "replace"))
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < int(limit):
            break
        oldest = min(float(e.get("start_time") or 0) for e in batch)
        before = oldest - 0.001
    return out


# ---------------------------------------------------------------------------
# vision: clothing-colour histogram (the only expensive step)
# ---------------------------------------------------------------------------
def _hist_half(arr):
    if arr.size == 0:
        return None
    h = (arr[..., 0].astype(np.int32) * 12) // 256
    s = (arr[..., 1].astype(np.int32) * 4) // 256
    idx = (h * 4 + s).ravel()
    return np.bincount(idx, minlength=48).astype(np.float32)


def compute_hist(image_path, box, min_px=16):
    """(vector|None, hist_ok) where hist_ok is 1=ok, 2=unusable."""
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None, 2
    try:
        with Image.open(image_path) as src:
            im = src.convert("RGB")
            width, height = im.size
            x0 = max(0, int(x * width))
            y0 = max(0, int(y * height))
            x1 = min(width, int((x + w) * width))
            y1 = min(height, int((y + h) * height))
            if (x1 - x0) < int(min_px) or (y1 - y0) < int(min_px):
                return None, 2
            arr = np.asarray(im.crop((x0, y0, x1, y1)).convert("HSV"),
                             dtype=np.uint8)
    except Exception:  # noqa: BLE001 - any decode problem = histogram unusable
        return None, 2
    half = arr.shape[0] // 2 or 1
    top, bot = _hist_half(arr[:half]), _hist_half(arr[half:])
    if top is None and bot is None:
        return None, 2
    if top is None:
        top = np.zeros(48, dtype=np.float32)
    if bot is None:
        bot = np.zeros(48, dtype=np.float32)
    vec = np.concatenate([top, bot])
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None, 2
    return (vec / norm).astype(np.float32), 1


def hist_to_blob(vec):
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_hist(blob):
    if not blob:
        return None
    try:
        vec = np.frombuffer(blob, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


def hist_similar(a, b, threshold):
    """True when the two histograms pass the gate (missing => cannot judge)."""
    if a is None or b is None:
        return True
    return float(np.dot(a, b)) >= float(threshold)


def angle_compatible(a, b, tol_deg):
    if a is None or b is None:
        return True
    delta = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return delta <= float(tol_deg)


# ---------------------------------------------------------------------------
# event normalization + analysis
# ---------------------------------------------------------------------------
def _parse_path(path_data):
    """Frigate data.path_data -> [(x, y, timestamp|None), ...].

    Frigate 0.17 stores ``[[x, y], timestamp]``; tolerate a bare ``[x, y]``.
    """
    points = []
    for entry in path_data or ():
        try:
            if (isinstance(entry, (list, tuple)) and entry
                    and isinstance(entry[0], (list, tuple))):
                xy = entry[0]
                ts = entry[1] if len(entry) > 1 else None
            else:
                xy = entry
                ts = None
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError, IndexError):
            continue
        try:
            ts = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts = None
        points.append((x, y, ts))
    return points


def _dist_xy(a, b, aspect):
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * aspect)


def _angle_xy(a, b, aspect):
    if a is None or b is None:
        return None
    dx = b[0] - a[0]
    dy = (b[1] - a[1]) * aspect
    if abs(dx) + abs(dy) <= 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def _box_end_xy(box):
    """Final box bottom-centre, matching Frigate's path point convention."""
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return (x + w / 2.0, y + h)


def _event_features(path_data, box, start_time, end_time, settings):
    """Geometry/state features derived from one event's path and final box."""
    aspect = settings.f("DETECT_ASPECT")
    pts = _parse_path(path_data)
    entry = pts[0][:2] if pts else None
    tail = pts[-1][:2] if pts else None
    end_xy = _box_end_xy(box)
    box_ok = end_xy is not None
    if end_xy is None and tail is not None:
        end_xy = tail

    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        disp = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    else:
        disp = 0.0

    net_disp = _dist_xy(entry, end_xy, aspect) or 0.0
    path_len = 0.0
    for a, b in zip(pts, pts[1:]):
        path_len += _dist_xy(a, b, aspect) or 0.0
    straightness = (net_disp / path_len) if path_len > 0 else 0.0

    head_angle = _angle_xy(pts[0], pts[1], aspect) if len(pts) >= 2 else None
    tail_angle = _angle_xy(pts[-2], pts[-1], aspect) if len(pts) >= 2 else None
    legacy_angle = _angle_xy(entry, tail, aspect) if len(pts) >= 2 else None

    tail_gap = None
    if end_time is not None and pts and pts[-1][2] is not None:
        tail_gap = max(0.0, float(end_time) - pts[-1][2])
    tail_stationary = False
    if box_ok and tail is not None:
        close = ((_dist_xy(tail, end_xy, aspect) or 0.0)
                 <= settings.f("TAIL_STATIONARY_DIST"))
        # A moving event usually has its last path point right at end_time.
        # Only call it stationary when no point was sampled for a while.
        if tail_gap is None:
            tail_stationary = close
        else:
            tail_stationary = close and tail_gap >= settings.f("TAIL_STATIONARY_MIN_GAP_S")

    dir_trusted = (len(pts) >= 2
                   and net_disp >= settings.f("DIR_MIN_DISP")
                   and straightness >= settings.f("DIR_MIN_STRAIGHTNESS"))

    speed = None
    vel_xy = None
    if len(pts) >= 2 and pts[-1][2] is not None and pts[-2][2] is not None:
        dt = pts[-1][2] - pts[-2][2]
        if dt > 0:
            speed = (_dist_xy(pts[-2], pts[-1], aspect) or 0.0) / dt
            vel_xy = ((pts[-1][0] - pts[-2][0]) / dt,
                      (pts[-1][1] - pts[-2][1]) / dt)
    if speed is None and end_time is not None and start_time is not None:
        duration = max(0.0, float(end_time) - float(start_time))
        if duration > 0:
            speed = net_disp / duration

    edge = box_near_edge(box, settings.f("EDGE_MARGIN"))
    exit_edge = box_exit_edge(box, settings.f("EXIT_MARGIN"))
    moving_disp = settings.f("MOVING_DISP")
    if exit_edge:
        state = "exited"
    elif tail_stationary and net_disp >= moving_disp:
        state = "stopped"
    elif net_disp < moving_disp:
        state = "in_place"
    else:
        state = "lost"

    return {
        "disp": disp,
        "entry": entry,
        "tail": tail,
        "end_xy": end_xy,
        "net_disp": net_disp,
        "path_len": path_len,
        "straightness": straightness,
        "head_angle": head_angle,
        "tail_angle": tail_angle,
        "legacy_angle": legacy_angle,
        "tail_gap_s": tail_gap,
        "tail_stationary": tail_stationary,
        "dir_trusted": dir_trusted,
        "edge": edge,
        "edge_exit": exit_edge,
        "speed": speed,
        "vel_xy": vel_xy,
        "state": state,
    }


def box_near_edge(box, margin):
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return (x <= margin or y <= margin
            or (x + w) >= (1.0 - margin) or (y + h) >= (1.0 - margin))


def box_exit_edge(box, margin):
    """Exit candidate: clipped on left/right/top, NOT the bottom edge.

    Frigate's path y is the box bottom; people standing/walking on the ground
    routinely sit at y+h ~= 1.0, so the bottom edge is not a usable exit signal.
    """
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return (x <= margin or y <= margin or (x + w) >= (1.0 - margin))


def normalize_event(event, settings):
    data = event.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    zones = event.get("zones")
    if not isinstance(zones, list):
        zones = []
    box = data.get("box")
    start = float(event.get("start_time") or 0)
    end = float(event["end_time"]) if event.get("end_time") else None
    path_data = data.get("path_data")
    feat = _event_features(path_data, box, start, end, settings)
    edge = feat["edge"]
    significant = (feat["disp"] >= settings.f("MOTION_MIN_DISP")) or edge
    return {
        "frigate_event_id": event.get("id"),
        "camera": event.get("camera") or "",
        "start_time": start,
        "end_time": end,
        "zones": json.dumps(zones),
        "box": json.dumps(box) if box else None,
        "motion_disp": feat["disp"],
        "angle": feat["legacy_angle"],
        "entry_xy": json.dumps(feat["entry"]) if feat["entry"] else None,
        "exit_xy": json.dumps(feat["tail"]) if feat["tail"] else None,
        "path_json": json.dumps(path_data) if path_data else None,
        "end_xy": json.dumps(feat["end_xy"]) if feat["end_xy"] else None,
        "net_disp": feat["net_disp"],
        "path_len": feat["path_len"],
        "straightness": feat["straightness"],
        "head_angle": feat["head_angle"],
        "tail_angle": feat["tail_angle"],
        "tail_gap_s": feat["tail_gap_s"],
        "tail_stationary": 1 if feat["tail_stationary"] else 0,
        "dir_trusted": 1 if feat["dir_trusted"] else 0,
        "edge_exit": 1 if feat["edge_exit"] else 0,
        "speed": feat["speed"],
        "vel_xy": json.dumps(feat["vel_xy"]) if feat["vel_xy"] else None,
        "state": feat["state"],
        "significant": 1 if significant else 0,
    }


def _clean_frame(settings, camera, event_id):
    clips = os.path.join(settings.s("FRIGATE_MEDIA_DIR"), settings.s("CLIPS_SUBDIR"))
    for suffix in ("-clean.webp", ".jpg"):
        path = os.path.join(clips, "{}-{}{}".format(camera, event_id, suffix))
        if os.path.isfile(path):
            return path
    return None


def analyze_event(conn, row, settings):
    """Run the expensive pipeline for ONE event: crop -> histogram."""
    box = json.loads(row["box"]) if row["box"] else None
    hist, ok = None, 2
    if box:
        frame = _clean_frame(settings, row["camera"], row["frigate_event_id"])
        if frame:
            hist, ok = compute_hist(frame, box, settings.i("MIN_CROP_PX"))
    now = time.time()
    conn.execute(
        "UPDATE events SET analyzed=1, hist_ok=?, hist=?, updated_at=? WHERE frigate_event_id=?",
        (ok, hist_to_blob(hist), now, row["frigate_event_id"]))
    return ok


def analyze_pending(conn, settings):
    """Analyse significant events now; batch insignificant ones at the count."""
    batch = settings.i("INSIGNIFICANT_COUNT_THRESHOLD")
    done_sig = done_insig = 0
    rows = conn.execute(
        "SELECT * FROM events WHERE analyzed=0 AND significant=1"
        " ORDER BY start_time ASC").fetchall()
    for row in rows:
        analyze_event(conn, row, settings)
        done_sig += 1
    conn.commit()
    for camera in [r["camera"] for r in conn.execute(
            "SELECT DISTINCT camera FROM events WHERE analyzed=0 AND significant=0")]:
        pending = conn.execute(
            "SELECT * FROM events WHERE analyzed=0 AND significant=0 AND camera=?"
            " ORDER BY start_time ASC", (camera,)).fetchall()
        if len(pending) >= batch:
            for row in pending:
                analyze_event(conn, row, settings)
                done_insig += 1
            conn.commit()
    if done_sig or done_insig:
        _log("analyzed: significant={} insignificant-batched={}".format(
            done_sig, done_insig))
    return done_sig, done_insig


# ---------------------------------------------------------------------------
# merge engine -> visits
# ---------------------------------------------------------------------------
def _new_presence(row, camera, hist):
    start = float(row["start_time"])
    end = float(row["end_time"]) if row["end_time"] else start
    return {
        "camera": camera,
        "enter": start,
        "leave": max(start, end),
        "events": [row["frigate_event_id"]],
        "zones": set(json.loads(row["zones"]) if row["zones"] else []),
        "disp": float(row["motion_disp"] or 0.0),
        "edge": bool(box_near_edge(json.loads(row["box"]) if row["box"] else None, 0.0)),
        "angle": row["angle"],
        "hist": hist,
        "rep": row["frigate_event_id"],
        "rep_dur": max(0.0, end - start),
        "n": 1,
    }


def _json_point(value):
    try:
        point = json.loads(value)
        return (float(point[0]), float(point[1]))
    except (TypeError, ValueError, IndexError):
        return None


def _row_features(row, settings):
    """Reconstruct one event's features, preferring the stored raw path."""
    path = None
    if row["path_json"]:
        try:
            path = json.loads(row["path_json"])
        except (TypeError, ValueError):
            path = None
    box = json.loads(row["box"]) if row["box"] else None
    start = float(row["start_time"] or 0)
    end = float(row["end_time"]) if row["end_time"] else start
    end = max(start, end)

    if path:
        feat = _event_features(path, box, start, end, settings)
    else:
        # Rows written before path_json existed: use the cached summary fields.
        aspect = settings.f("DETECT_ASPECT")
        entry = _json_point(row["entry_xy"])
        tail = _json_point(row["exit_xy"])
        end_xy = _box_end_xy(box) or tail
        net_disp = _dist_xy(entry, end_xy, aspect) or 0.0
        tail_stationary = False
        if end_xy is not None and tail is not None:
            tail_stationary = ((_dist_xy(tail, end_xy, aspect) or 0.0)
                               <= settings.f("TAIL_STATIONARY_DIST"))
        edge = box_near_edge(box, settings.f("EDGE_MARGIN"))
        exit_edge = box_exit_edge(box, settings.f("EXIT_MARGIN"))
        moving_disp = settings.f("MOVING_DISP")
        if exit_edge:
            state = "exited"
        elif tail_stationary and net_disp >= moving_disp:
            state = "stopped"
        elif net_disp < moving_disp:
            state = "in_place"
        else:
            state = "lost"
        duration = max(0.0, end - start)
        feat = {
            "disp": float(row["motion_disp"] or 0.0),
            "entry": entry,
            "tail": tail,
            "end_xy": end_xy,
            "net_disp": net_disp,
            "path_len": net_disp,
            "straightness": 1.0 if net_disp > 0 else 0.0,
            "head_angle": None,
            "tail_angle": row["angle"],
            "legacy_angle": row["angle"],
            "tail_gap_s": None,
            "tail_stationary": tail_stationary,
            "dir_trusted": (row["angle"] is not None
                            and net_disp >= settings.f("DIR_MIN_DISP")),
            "edge": edge,
            "edge_exit": exit_edge,
            "speed": (net_disp / duration) if duration > 0 else None,
            "vel_xy": None,
            "state": state,
        }

    hist = blob_to_hist(row["hist"]) if row["hist_ok"] == 1 else None
    zones = json.loads(row["zones"]) if row["zones"] else []
    return {
        "id": row["frigate_event_id"],
        "camera": row["camera"],
        "start": start,
        "end": end,
        "duration": max(0.0, end - start),
        "zones": zones,
        "hist": hist,
        "box": box,
        **feat,
    }


def _review_mark(p, reason):
    p["needs_review"] = 1
    if not reason:
        return
    old = p.get("review_reason") or ""
    parts = [x for x in old.split(",") if x]
    if reason not in parts:
        parts.append(reason)
    p["review_reason"] = ",".join(parts[:6])


def _new_presence_v2(feat, camera):
    return {
        "camera": camera,
        "enter": feat["start"],
        "leave": feat["end"],
        "events": [feat["id"]],
        "zones": set(feat["zones"] or []),
        "disp": feat["disp"],
        "edge": bool(feat["edge"]),
        "angle": feat["tail_angle"],
        "hist": feat["hist"],
        "rep": feat["id"],
        "rep_dur": feat["duration"],
        "n": 1,
        "state": feat["state"],
        "confidence": 1.0,
        "needs_review": 0,
        "review_reason": None,
        "first_event_id": feat["id"],
        "last_event_id": feat["id"],
        "last_pos": feat["end_xy"],
        "last_tail_angle": feat["tail_angle"],
        "last_speed": feat["speed"],
        "last_vel_xy": feat["vel_xy"],
        "last_dir_trusted": bool(feat["dir_trusted"]),
    }


def _merge_into(p, feat, score):
    p["leave"] = max(p["leave"], feat["end"])
    p["n"] += 1
    p["events"].append(feat["id"])
    p["disp"] = max(p["disp"], feat["disp"])
    p["edge"] = p["edge"] or bool(feat["edge"])
    if feat["hist"] is not None:
        p["hist"] = feat["hist"]
    if feat["tail_angle"] is not None:
        p["angle"] = feat["tail_angle"]
    if feat["duration"] > p["rep_dur"]:
        p["rep"], p["rep_dur"] = feat["id"], feat["duration"]
    if feat["zones"]:
        p["zones"].update(feat["zones"])
    p["state"] = feat["state"]
    p["confidence"] = min(float(p.get("confidence") or 1.0), float(score))
    p["last_event_id"] = feat["id"]
    p["last_pos"] = feat["end_xy"] or p["last_pos"]
    p["last_tail_angle"] = feat["tail_angle"]
    p["last_speed"] = feat["speed"]
    p["last_vel_xy"] = feat["vel_xy"]
    p["last_dir_trusted"] = bool(feat["dir_trusted"])


def _allowed_gap(p, feat, settings):
    prev = p["state"]
    if prev == "exited":
        return 0.0
    if prev == "stopped":
        if feat["state"] in ("stopped", "in_place"):
            return settings.f("STATIONARY_GAP_S")
        return settings.f("STOP_GAP_S")
    if prev == "in_place":
        return settings.f("STATIONARY_GAP_S")
    if prev == "lost":
        return settings.f("OCCLUSION_GAP_S")
    return settings.f("CONT_GAP_S")


def _radius_for(p, settings):
    prev = p["state"]
    if prev == "stopped":
        return settings.f("STOP_RADIUS")
    if prev == "in_place":
        return settings.f("IN_PLACE_RADIUS")
    if prev == "lost":
        return settings.f("OCCLUSION_RADIUS")
    return settings.f("CONT_RADIUS")


def _candidate_distance(p, feat, settings):
    aspect = settings.f("DETECT_ASPECT")
    target = feat["entry"] or feat["end_xy"]
    if target is None or p.get("last_pos") is None:
        return None
    d_last = _dist_xy(p["last_pos"], target, aspect)
    d = d_last
    if p["state"] == "lost" and p.get("last_vel_xy") is not None:
        dt = max(0.0, feat["start"] - p["leave"])
        vx, vy = p["last_vel_xy"]
        predicted = (p["last_pos"][0] + vx * dt,
                     p["last_pos"][1] + vy * dt)
        d_pred = _dist_xy(predicted, target, aspect)
        if d_pred is not None and (d is None or d_pred < d):
            d = d_pred
    return d


def _candidate_score(p, feat, settings, dt, allowed):
    radius = _radius_for(p, settings)
    d = _candidate_distance(p, feat, settings)
    if d is None or radius <= 0:
        return None
    if d > radius * 1.5:
        return None
    if dt > allowed:
        return None

    time_score = 1.0 - 0.5 * (dt / allowed) if allowed > 0 else 0.0
    space_score = max(0.0, 1.0 - (d / radius))
    h = None
    if p.get("hist") is not None and feat["hist"] is not None:
        h = float(np.dot(p["hist"], feat["hist"]))
    hist_score = 0.5 if h is None else h

    angle_delta = None
    dir_score = 0.5
    if (p.get("last_dir_trusted") and feat["dir_trusted"]
            and p.get("last_tail_angle") is not None
            and feat["head_angle"] is not None):
        angle_delta = abs((float(p["last_tail_angle"])
                           - float(feat["head_angle"]) + 180.0) % 360.0 - 180.0)
        dir_score = max(0.0, math.cos(math.radians(angle_delta)))

    score = (settings.f("W_TIME") * time_score
             + settings.f("W_SPACE") * space_score
             + settings.f("W_HIST") * hist_score
             + settings.f("W_DIR") * dir_score)

    reasons = []
    hard_ok = True
    if d > radius:
        hard_ok = False
        reasons.append("distance")
    if h is not None and h < settings.f("HIST_MIN"):
        hard_ok = False
        reasons.append("hist")
    if h is None and not (space_score >= settings.f("HIST_MISSING_TIGHT_SPACE")
                          and dt <= settings.f("HIST_MISSING_MAX_GAP_S")):
        hard_ok = False
        reasons.append("hist_missing")

    # A6 shares the "lost" state with ordinary A1/A2 fragmentation. Gaps past
    # the normal continuation window must therefore clear a higher bar.
    if (p["state"] == "lost" and dt > settings.f("CONT_GAP_S")
            and score < settings.f("MERGE_SCORE") + settings.f("ASSIGN_MARGIN")):
        hard_ok = False
        reasons.append("occlusion_low_confidence")

    return {
        "prev": p,
        "dt": dt,
        "allowed": allowed,
        "dist": d,
        "hist": h,
        "angle_delta": angle_delta,
        "score": score,
        "hard_ok": hard_ok,
        "reasons": reasons,
    }


def _add_link(links, camera, prev_id, next_id, cand, decision, reason):
    links.append({
        "camera": camera,
        "prev_event_id": prev_id,
        "next_event_id": next_id,
        "dt": cand["dt"] if cand else None,
        "dist": cand["dist"] if cand else None,
        "hist_sim": cand["hist"] if cand else None,
        "angle_delta": cand["angle_delta"] if cand else None,
        "score": cand["score"] if cand else None,
        "decision": decision,
        "reason": reason,
    })


def _finalize_presences(presences, settings, places):
    presences.sort(key=lambda p: (p["camera"], p["enter"]))
    min_duration = settings.f("MIN_DURATION_S")
    motion_min = settings.f("MOTION_MIN_DISP")
    kept = []
    counters = {}
    for p in presences:
        p["duration"] = max(0.0, p["leave"] - p["enter"])
        if p["duration"] < min_duration:
            continue
        day = utc_day(p["enter"])
        key = (p["camera"], day)
        counters[key] = counters.get(key, 0) + 1
        p["day"] = day
        p["person_no"] = counters[key]
        p["moved"] = p["disp"] >= motion_min
        p["significant"] = p["moved"] or bool(p["edge"])
        p["place"] = places.resolve(p["camera"], sorted(p["zones"]))
        p["display_name"] = places.display_name(p["camera"])
        p.setdefault("state", "in_place")
        p.setdefault("confidence", None)
        p.setdefault("needs_review", 0)
        p.setdefault("review_reason", None)
        p.setdefault("first_event_id", p["events"][0] if p["events"] else None)
        p.setdefault("last_event_id", p["events"][-1] if p["events"] else None)
        kept.append(p)
    return kept


def _build_visits_legacy(rows, settings):
    """Pre-v2 merge, retained as MERGE_ENGINE=legacy for rollback."""
    gap_max = settings.f("MERGE_GAP_S")
    inactivity = settings.f("INACTIVITY_S")
    angle_tol = settings.f("ANGLE_TOL_DEG")
    hist_thr = settings.f("HIST_THRESHOLD")
    edge_margin = settings.f("EDGE_MARGIN")

    by_camera = {}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row)

    presences = []
    for camera, events in by_camera.items():
        open_p = []
        for row in events:
            start = float(row["start_time"])
            end = float(row["end_time"]) if row["end_time"] else start
            end = max(start, end)

            still = []
            for p in open_p:
                if start - p["leave"] >= inactivity:
                    presences.append(p)
                else:
                    still.append(p)
            open_p = still

            e_hist = blob_to_hist(row["hist"]) if row["hist_ok"] == 1 else None
            e_angle = row["angle"]
            best, best_gap = None, None
            for p in open_p:
                gap = start - p["leave"]
                if gap < 0 or gap > gap_max:
                    continue
                if not angle_compatible(p["angle"], e_angle, angle_tol):
                    continue
                if not hist_similar(p["hist"], e_hist, hist_thr):
                    continue
                if best is None or gap < best_gap:
                    best, best_gap = p, gap

            if best is not None:
                p = best
                p["leave"] = max(p["leave"], end)
                p["n"] += 1
                p["events"].append(row["frigate_event_id"])
                p["disp"] = max(p["disp"], float(row["motion_disp"] or 0.0))
                p["edge"] = p["edge"] or box_near_edge(
                    json.loads(row["box"]) if row["box"] else None, edge_margin)
                if e_angle is not None:
                    p["angle"] = e_angle
                if e_hist is not None:
                    p["hist"] = e_hist
                dur = end - start
                if dur > p["rep_dur"]:
                    p["rep"], p["rep_dur"] = row["frigate_event_id"], dur
                if row["zones"]:
                    p["zones"].update(json.loads(row["zones"]))
            else:
                p = _new_presence(row, camera, e_hist)
                p["edge"] = box_near_edge(
                    json.loads(row["box"]) if row["box"] else None, edge_margin)
                open_p.append(p)
        presences.extend(open_p)
    return presences


def build_visits_v2(rows, settings, places):
    """State-aware per-camera merge; returns (visits, candidate links)."""
    inactivity = settings.f("INACTIVITY_S")
    merge_score = settings.f("MERGE_SCORE")
    ambig_score = settings.f("AMBIG_SCORE")
    assign_margin = settings.f("ASSIGN_MARGIN")

    by_camera = {}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row)

    presences = []
    links = []
    for camera, events in by_camera.items():
        open_p = []
        for row in events:
            feat = _row_features(row, settings)
            start = feat["start"]

            still = []
            for p in open_p:
                if start - p["leave"] >= inactivity:
                    presences.append(p)
                else:
                    still.append(p)
            open_p = still

            candidates = []
            overlap_p = None
            for p in open_p:
                dt = start - p["leave"]
                if dt < 0:
                    if overlap_p is None or p["leave"] > overlap_p["leave"]:
                        overlap_p = p
                    continue
                allowed = _allowed_gap(p, feat, settings)
                if allowed <= 0 or dt > allowed:
                    continue
                cand = _candidate_score(p, feat, settings, dt, allowed)
                if cand is not None:
                    candidates.append(cand)

            chosen = None
            if candidates:
                candidates.sort(key=lambda c: -c["score"])
                mergeable = [c for c in candidates
                             if c["hard_ok"] and c["score"] >= merge_score]
                ambiguous = [c for c in candidates
                             if c["score"] >= ambig_score
                             and c["dist"] <= _radius_for(c["prev"], settings) * 1.2]
                if mergeable:
                    best = mergeable[0]
                    conflict = (len(mergeable) > 1
                                and best["score"] - mergeable[1]["score"] < assign_margin)
                    if conflict:
                        newp = _new_presence_v2(feat, camera)
                        _review_mark(newp, "ambiguous_assignment:%s" % feat["id"])
                        for c in mergeable[:2]:
                            _review_mark(c["prev"], "ambiguous_assignment:%s" % feat["id"])
                            _add_link(links, camera, c["prev"]["last_event_id"],
                                      feat["id"], c, "ambiguous",
                                      "assignment_conflict")
                        for c in candidates:
                            if c not in mergeable[:2]:
                                _add_link(links, camera, c["prev"]["last_event_id"],
                                          feat["id"], c, "separate", "not_best")
                        open_p.append(newp)
                    else:
                        chosen = best
                        prev_event_id = chosen["prev"]["last_event_id"]
                        prev_state = chosen["prev"]["state"]
                        hist_missing = feat["hist"] is None
                        _merge_into(chosen["prev"], feat, chosen["score"])
                        if hist_missing:
                            _review_mark(chosen["prev"],
                                         "appearance_unconfirmed:%s" % feat["id"])
                        if prev_state == "lost" and chosen["dt"] > settings.f("CONT_GAP_S"):
                            _review_mark(chosen["prev"],
                                         "occlusion_resume:%s" % feat["id"])
                        if chosen["dt"] > settings.f("STOP_GAP_S"):
                            _review_mark(chosen["prev"],
                                         "long_resume:%s" % feat["id"])
                        _add_link(links, camera, prev_event_id,
                                  feat["id"], chosen, "merged",
                                  ",".join(chosen["reasons"]) or "score")
                        for c in candidates:
                            if c is not chosen:
                                _add_link(links, camera, c["prev"]["last_event_id"],
                                          feat["id"], c, "separate", "not_best")
                elif ambiguous:
                    best = ambiguous[0]
                    newp = _new_presence_v2(feat, camera)
                    _review_mark(newp, "ambiguous:%s" % feat["id"])
                    _review_mark(best["prev"], "ambiguous:%s" % feat["id"])
                    _add_link(links, camera, best["prev"]["last_event_id"],
                              feat["id"], best, "ambiguous",
                              ",".join(best["reasons"]) or "low_score")
                    for c in candidates:
                        if c is not best:
                            _add_link(links, camera, c["prev"]["last_event_id"],
                                      feat["id"], c, "separate", "not_best")
                    open_p.append(newp)
                else:
                    for c in candidates:
                        _add_link(links, camera, c["prev"]["last_event_id"],
                                  feat["id"], c, "separate",
                                  ",".join(c["reasons"]) or "low_score")
                    open_p.append(_new_presence_v2(feat, camera))
            else:
                if overlap_p is not None:
                    _add_link(links, camera, overlap_p["last_event_id"],
                              feat["id"], None, "overlap", "overlap_unsupported")
                open_p.append(_new_presence_v2(feat, camera))
        presences.extend(open_p)

    return _finalize_presences(presences, settings, places), links


def build_visits(rows, settings, places):
    """Dispatch to the configured merge engine; always returns (visits, links)."""
    if settings.s("MERGE_ENGINE").lower() == "legacy":
        return _finalize_presences(_build_visits_legacy(rows, settings),
                                   settings, places), []
    return build_visits_v2(rows, settings, places)


def write_visits(conn, presences, links=None):
    """Replace the derived visits; return the list of NEW significant visits."""
    old = {(r["camera"], float(r["enter_time"]), int(r["person_no"]))
           for r in conn.execute(
               "SELECT camera, enter_time, person_no FROM visits")}
    conn.execute("DELETE FROM visit_events")
    conn.execute("DELETE FROM visit_links")
    conn.execute("DELETE FROM visits")
    now = time.time()
    new_visits = []
    for p in presences:
        cur = conn.execute(
            "INSERT INTO visits (camera, day, display_name, place, person_no,"
            " enter_time, leave_time, duration_s, n_events, rep_event_id, moved,"
            " edge, significant, state, confidence, needs_review, review_reason,"
            " first_event_id, last_event_id, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["camera"], p["day"], p["display_name"], p["place"], p["person_no"],
             p["enter"], p["leave"], p["duration"], p["n"], p["rep"],
             1 if p["moved"] else 0, 1 if p["edge"] else 0,
             1 if p["significant"] else 0, p.get("state"),
             p.get("confidence"), 1 if p.get("needs_review") else 0,
             p.get("review_reason"), p.get("first_event_id"),
             p.get("last_event_id"), now, now))
        visit_id = cur.lastrowid
        for ev_id in p["events"]:
            conn.execute(
                "INSERT OR IGNORE INTO visit_events (visit_id, frigate_event_id)"
                " VALUES (?,?)", (visit_id, ev_id))
        key = (p["camera"], p["enter"], p["person_no"])
        if key not in old and p["significant"]:
            new_visits.append({
                "display_name": p["display_name"],
                "camera": p["camera"],
                "place": p["place"],
                "person_no": p["person_no"],
                "enter_time": p["enter"],
                "leave_time": p["leave"],
                "duration_s": p["duration"],
            })

    for link in links or ():
        conn.execute(
            "INSERT INTO visit_links (camera, prev_event_id, next_event_id, dt,"
            " dist, hist_sim, angle_delta, score, decision, reason, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (link["camera"], link["prev_event_id"], link["next_event_id"],
             link.get("dt"), link.get("dist"), link.get("hist_sim"),
             link.get("angle_delta"), link.get("score"), link["decision"],
             link.get("reason"), now))
    return new_visits


def prune(conn, days):
    if days <= 0:
        return
    cutoff = time.time() - days * 86400.0
    conn.execute("DELETE FROM visit_events WHERE visit_id IN"
                 " (SELECT id FROM visits WHERE enter_time < ?)", (cutoff,))
    conn.execute("DELETE FROM visits WHERE enter_time < ?", (cutoff,))
    conn.execute("DELETE FROM events WHERE start_time < ?", (cutoff,))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def human_duration(seconds):
    secs = max(0.0, float(seconds))
    if secs < 60:
        return "under a minute"
    if secs < 3600:
        return "{} min".format(int(round(secs / 60.0)))
    hours = int(secs // 3600)
    mins = int(round((secs % 3600) / 60.0))
    if mins:
        return "{}h {}min".format(hours, mins)
    return "{}h".format(hours)


def format_visit(visit, tz_offset_h=0.0):
    def clock(epoch):
        dt = datetime.fromtimestamp(float(epoch), timezone.utc)
        if tz_offset_h:
            from datetime import timedelta
            dt = dt + timedelta(hours=float(tz_offset_h))
        return dt.strftime("%H:%M")
    return "{} ({})  {}  person #{}  {}-{}  {}".format(
        visit["display_name"], visit["camera"], visit["place"],
        visit["person_no"], clock(visit["enter_time"]),
        clock(visit["leave_time"]), human_duration(visit["duration_s"]))


# ---------------------------------------------------------------------------
# cycle
# ---------------------------------------------------------------------------
def _cursor_path(settings):
    return os.path.join(settings.s("STORE_DIR"), ".cursor")


def _load_cursor(settings):
    try:
        with open(_cursor_path(settings), encoding="utf-8") as fh:
            return float(json.load(fh).get("cursor") or 0)
    except (OSError, ValueError, TypeError):
        return 0.0


def _save_cursor(settings, cursor):
    try:
        with open(_cursor_path(settings), "w", encoding="utf-8") as fh:
            json.dump({"cursor": float(cursor)}, fh)
    except OSError:
        pass


def _upsert_events(conn, events, settings):
    """Normalise + upsert API events. Returns (changed, max_start)."""
    changed = 0
    max_start = 0.0
    for event in events:
        norm = normalize_event(event, settings)
        if not norm["frigate_event_id"]:
            continue
        exists = conn.execute(
            "SELECT 1 FROM events WHERE frigate_event_id=?",
            (norm["frigate_event_id"],)).fetchone()
        now = time.time()
        values = (
            norm["camera"], norm["start_time"], norm["end_time"], norm["zones"],
            norm["box"], norm["motion_disp"], norm["angle"], norm["entry_xy"],
            norm["exit_xy"], norm["path_json"], norm["end_xy"], norm["net_disp"],
            norm["path_len"], norm["straightness"], norm["head_angle"],
            norm["tail_angle"], norm["tail_gap_s"], norm["tail_stationary"],
            norm["dir_trusted"], norm["edge_exit"], norm["speed"],
            norm["vel_xy"], norm["state"], norm["significant"],
        )
        if exists is None:
            conn.execute(
                "INSERT INTO events (camera, start_time, end_time, zones, box,"
                " motion_disp, angle, entry_xy, exit_xy, path_json, end_xy,"
                " net_disp, path_len, straightness, head_angle, tail_angle,"
                " tail_gap_s, tail_stationary, dir_trusted, edge_exit, speed,"
                " vel_xy, state, significant, frigate_event_id, created_at,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                values + (norm["frigate_event_id"], now, now))
            changed += 1
        else:
            conn.execute(
                "UPDATE events SET camera=?, start_time=?, end_time=?, zones=?,"
                " box=?, motion_disp=?, angle=?, entry_xy=?, exit_xy=?,"
                " path_json=?, end_xy=?, net_disp=?, path_len=?, straightness=?,"
                " head_angle=?, tail_angle=?, tail_gap_s=?, tail_stationary=?,"
                " dir_trusted=?, edge_exit=?, speed=?, vel_xy=?, state=?,"
                " significant=?, updated_at=? WHERE frigate_event_id=?",
                values + (now, norm["frigate_event_id"]))
        max_start = max(max_start, norm["start_time"])
    conn.commit()
    return changed, max_start


def _rebuild_visits(conn, settings, places):
    rows = conn.execute(
        "SELECT * FROM events WHERE start_time > 0"
        " ORDER BY camera ASC, start_time ASC").fetchall()
    presences, links = build_visits(rows, settings, places)
    new = write_visits(conn, presences, links)
    prune(conn, settings.i("VISITS_RETENTION_DAYS"))
    conn.commit()
    return new, len(rows)


def cycle(settings, conn, places):
    now = time.time()
    cursor = _load_cursor(settings)
    after = (cursor - settings.f("RESCAN_OVERLAP_S")) if cursor \
        else (now - settings.f("INITIAL_LOOKBACK_S"))
    if after < 0:
        after = 0.0
    try:
        events = fetch_person_events(settings.s("FRIGATE_API"), after,
                                     settings.i("FETCH_LIMIT"), settings.b("ONLY_PERSON"))
    except Exception as exc:  # noqa: BLE001 - Frigate down must not kill the loop
        _log("WARNING: Frigate API fetch failed: {}".format(exc))
        return

    changed, max_start = _upsert_events(conn, events, settings)
    if max_start > cursor:
        _save_cursor(settings, max_start)

    analyze_pending(conn, settings)
    new, total = _rebuild_visits(conn, settings, places)
    tz = settings.f("TZ_OFFSET_H")
    for visit in new:
        _log("visit: " + format_visit(visit, tz))
    _log("cycle: fetched={} new_events={} visits={} new={}".format(
        len(events), changed, total, len(new)))


def backfill(settings, conn, places):
    """One-off 7-day path/feature backfill (safe to run repeatedly)."""
    after = time.time() - settings.f("INITIAL_LOOKBACK_S")
    events = fetch_person_events(settings.s("FRIGATE_API"), after,
                                 settings.i("FETCH_LIMIT"), settings.b("ONLY_PERSON"))
    changed, _ = _upsert_events(conn, events, settings)
    analyze_pending(conn, settings)
    new, total = _rebuild_visits(conn, settings, places)
    _log("backfill: fetched={} new_events={} visits={} new={}".format(
        len(events), changed, total, len(new)))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_check(settings, places):
    print("visits --check")
    print("  FRIGATE_API   :", settings.s("FRIGATE_API"))
    print("  MEDIA_DIR     :", settings.s("FRIGATE_MEDIA_DIR"))
    print("  STORE         :", settings.db_path)
    print("  MERGE_ENGINE  :", settings.s("MERGE_ENGINE"))
    print("  INACTIVITY_S  :", settings.i("INACTIVITY_S"))
    print("  cameras/places:", len(places.camera_places), "cameras,",
          len(places.zone_places), "zones")
    problems = []
    try:
        events = fetch_person_events(settings.s("FRIGATE_API"), 0, 1)
        print("  Frigate API   : OK ({} event(s) sampled)".format(len(events)))
    except Exception as exc:  # noqa: BLE001
        problems.append("Frigate API: {}".format(exc))
    try:
        conn = open_store(settings.db_path)
        conn.close()
        print("  store         : OK (writable)")
    except Exception as exc:  # noqa: BLE001
        problems.append("store: {}".format(exc))
    for p in problems:
        print("  PROBLEM       :", p)
    return 1 if problems else 0


def cmd_list(settings, day=None, show_all=False):
    conn = open_store(settings.db_path)
    sql = ("SELECT * FROM visits WHERE 1=1")
    params = []
    if day:
        sql += " AND day = ?"
        params.append(day)
    if not show_all:
        sql += " AND significant = 1"
    sql += " ORDER BY day ASC, camera ASC, enter_time ASC"
    for visit in conn.execute(sql, params):
        print(format_visit(visit, settings.f("TZ_OFFSET_H")))
    conn.close()
    return 0


def status_counts(conn):
    e = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    v = conn.execute("SELECT count(*) FROM visits").fetchone()[0]
    sig = conn.execute(
        "SELECT count(*) FROM visits WHERE significant=1").fetchone()[0]
    review = conn.execute(
        "SELECT count(*) FROM visits WHERE needs_review=1").fetchone()[0]
    pend = conn.execute(
        "SELECT count(*) FROM events WHERE analyzed=0").fetchone()[0]
    return ("events={} visits={} significant={} review={} unanalyzed={}".format(
        e, v, sig, review, pend))


def cmd_status(settings):
    conn = open_store(settings.db_path)
    print(status_counts(conn))
    conn.close()
    return 0


def run_forever(settings, places):
    conn = open_store(settings.db_path)
    poll = max(5, settings.i("POLL_EVERY_S"))
    heartbeat = max(poll, settings.i("HEARTBEAT_S"))
    _log("started: api={} store={} poll={}s engine={} inactivity={}s"
         .format(settings.s("FRIGATE_API"), settings.db_path, poll,
                 settings.s("MERGE_ENGINE"), settings.i("INACTIVITY_S")))
    last_hb = 0.0
    while True:
        try:
            if settings.b("ENABLED"):
                cycle(settings, conn, places)
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            _log("ERROR in cycle: {}".format(exc))
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        now = time.time()
        if now - last_hb >= heartbeat:
            _log("status: " + status_counts(conn))
            last_hb = now
        time.sleep(poll)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frigate per-camera person-visit log")
    parser.add_argument("--conf", default=os.environ.get(
        "VISITS_CONF", "/config/visits.conf"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--backfill", action="store_true",
                        help="re-fetch INITIAL_LOOKBACK_S and rebuild (path features)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--all", action="store_true", help="list includes insignificant")
    parser.add_argument("--day", default=None, help="YYYY-MM-DD for --list")
    args = parser.parse_args(argv)

    cfg = load_conf(args.conf)
    settings = Settings(cfg)
    places = Places.load(settings.s("PLACES_CONF"))

    if args.check:
        return cmd_check(settings, places)
    if args.list:
        return cmd_list(settings, args.day, args.all)
    if args.status:
        return cmd_status(settings)
    if args.backfill:
        conn = open_store(settings.db_path)
        backfill(settings, conn, places)
        conn.close()
        return 0
    if args.once:
        conn = open_store(settings.db_path)
        cycle(settings, conn, places)
        conn.close()
        return 0
    run_forever(settings, places)
    return 0


if __name__ == "__main__":
    sys.exit(main())
