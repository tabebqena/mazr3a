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
Per camera we keep a set of OPEN PRESENCES. A new event joins an open presence
when the time gap is small (`MERGE_GAP_S`), the motion direction is compatible
(`ANGLE_TOL_DEG`, derived from Frigate's `data.path_data`), and the clothing
histogram is similar enough (`HIST_THRESHOLD`). Otherwise it starts a NEW
presence with the next `person #`. A presence is RETIRED after `INACTIVITY_S`
with no person events (4 h by default -> the overnight gap closes it, so each
morning starts a fresh person #).

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
threshold costs no re-capture and no re-crop (histograms are cached per event).

Usage:
  python visits.py                 # run the scheduler loop
  python visits.py --check         # READ-ONLY config/API/store probe
  python visits.py --once          # one poll/analyse/rebuild, exit
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
    "MERGE_GAP_S": "30",
    "INACTIVITY_S": "14400",
    "ANGLE_TOL_DEG": "45",
    "MOTION_MIN_DISP": "0.05",
    "EDGE_MARGIN": "0.05",
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
"""


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
def open_store(path):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
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
def _path_features(path_data):
    """(disp, angle_deg, entry_xy, exit_xy) from Frigate data.path_data."""
    pts = []
    for entry in path_data or ():
        try:
            xy = entry[0]
            pts.append((float(xy[0]), float(xy[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if not pts:
        return 0.0, None, None, None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    disp = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    angle = None
    if len(pts) >= 2:
        dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
        if abs(dx) + abs(dy) > 1e-6:
            angle = math.degrees(math.atan2(dy, dx))
    return disp, angle, pts[0], pts[-1]


def box_near_edge(box, margin):
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return (x <= margin or y <= margin
            or (x + w) >= (1.0 - margin) or (y + h) >= (1.0 - margin))


def normalize_event(event, settings):
    data = event.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    zones = event.get("zones")
    if not isinstance(zones, list):
        zones = []
    box = data.get("box")
    disp, angle, entry, exit_ = _path_features(data.get("path_data"))
    edge = box_near_edge(box, settings.f("EDGE_MARGIN"))
    significant = (disp >= settings.f("MOTION_MIN_DISP")) or edge
    return {
        "frigate_event_id": event.get("id"),
        "camera": event.get("camera") or "",
        "start_time": float(event.get("start_time") or 0),
        "end_time": float(event["end_time"]) if event.get("end_time") else None,
        "zones": json.dumps(zones),
        "box": json.dumps(box) if box else None,
        "motion_disp": disp,
        "angle": angle,
        "entry_xy": json.dumps(entry) if entry else None,
        "exit_xy": json.dumps(exit_) if exit_ else None,
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


def build_visits(rows, settings, places):
    """Deterministic per-camera presence merge. Pure function of `rows`."""
    gap_max = settings.f("MERGE_GAP_S")
    inactivity = settings.f("INACTIVITY_S")
    angle_tol = settings.f("ANGLE_TOL_DEG")
    hist_thr = settings.f("HIST_THRESHOLD")
    motion_min = settings.f("MOTION_MIN_DISP")
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

    presences.sort(key=lambda p: (p["camera"], p["enter"]))
    counters = {}
    for p in presences:
        day = utc_day(p["enter"])
        key = (p["camera"], day)
        counters[key] = counters.get(key, 0) + 1
        p["day"] = day
        p["person_no"] = counters[key]
        p["duration"] = max(0.0, p["leave"] - p["enter"])
        p["moved"] = p["disp"] >= motion_min
        p["significant"] = p["moved"] or p["edge"]
        p["place"] = places.resolve(p["camera"], sorted(p["zones"]))
        p["display_name"] = places.display_name(p["camera"])
    return presences


def write_visits(conn, presences):
    """Replace the derived visits; return the list of NEW significant visits."""
    old = {(r["camera"], float(r["enter_time"]), int(r["person_no"]))
           for r in conn.execute(
               "SELECT camera, enter_time, person_no FROM visits")}
    conn.execute("DELETE FROM visit_events")
    conn.execute("DELETE FROM visits")
    now = time.time()
    new_visits = []
    for p in presences:
        cur = conn.execute(
            "INSERT INTO visits (camera, day, display_name, place, person_no,"
            " enter_time, leave_time, duration_s, n_events, rep_event_id, moved,"
            " edge, significant, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["camera"], p["day"], p["display_name"], p["place"], p["person_no"],
             p["enter"], p["leave"], p["duration"], p["n"], p["rep"],
             1 if p["moved"] else 0, 1 if p["edge"] else 0,
             1 if p["significant"] else 0, now, now))
        visit_id = cur.lastrowid
        for ev_id in p["events"]:
            conn.execute(
                "INSERT OR IGNORE INTO visit_events (visit_id, frigate_event_id)"
                " VALUES (?,?)", (visit_id, ev_id))
        key = (p["camera"], p["enter"], p["person_no"])
        if key not in old and p["significant"]:
            new_visits.append(p)
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
    changed = 0
    max_start = cursor
    for event in events:
        norm = normalize_event(event, settings)
        if not norm["frigate_event_id"]:
            continue
        exists = conn.execute(
            "SELECT 1 FROM events WHERE frigate_event_id=?",
            (norm["frigate_event_id"],)).fetchone()
        now2 = time.time()
        if exists is None:
            conn.execute(
                "INSERT INTO events (frigate_event_id, camera, start_time, end_time,"
                " zones, box, motion_disp, angle, entry_xy, exit_xy, significant,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (norm["frigate_event_id"], norm["camera"], norm["start_time"],
                 norm["end_time"], norm["zones"], norm["box"], norm["motion_disp"],
                 norm["angle"], norm["entry_xy"], norm["exit_xy"],
                 norm["significant"], now2, now2))
            changed += 1
        else:
            conn.execute(
                "UPDATE events SET camera=?, start_time=?, end_time=?, zones=?, box=?,"
                " motion_disp=?, angle=?, entry_xy=?, exit_xy=?, significant=?,"
                " updated_at=? WHERE frigate_event_id=?",
                (norm["camera"], norm["start_time"], norm["end_time"], norm["zones"],
                 norm["box"], norm["motion_disp"], norm["angle"], norm["entry_xy"],
                 norm["exit_xy"], norm["significant"], now2,
                 norm["frigate_event_id"]))
        max_start = max(max_start, norm["start_time"])
    conn.commit()
    if max_start > cursor:
        _save_cursor(settings, max_start)

    analyze_pending(conn, settings)
    rows = conn.execute(
        "SELECT * FROM events WHERE start_time > 0"
        " ORDER BY camera ASC, start_time ASC").fetchall()
    new = write_visits(conn, build_visits(rows, settings, places))
    prune(conn, settings.i("VISITS_RETENTION_DAYS"))
    conn.commit()
    tz = settings.f("TZ_OFFSET_H")
    for visit in new:
        _log("visit: " + format_visit(visit, tz))
    _log("cycle: fetched={} new_events={} visits={} new={}".format(
        len(events), changed, len(rows), len(new)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_check(settings, places):
    print("visits --check")
    print("  FRIGATE_API   :", settings.s("FRIGATE_API"))
    print("  MEDIA_DIR     :", settings.s("FRIGATE_MEDIA_DIR"))
    print("  STORE         :", settings.db_path)
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
    pend = conn.execute(
        "SELECT count(*) FROM events WHERE analyzed=0").fetchone()[0]
    return ("events={} visits={} significant={} unanalyzed={}".format(
        e, v, sig, pend))


def cmd_status(settings):
    conn = open_store(settings.db_path)
    print(status_counts(conn))
    conn.close()
    return 0


def run_forever(settings, places):
    conn = open_store(settings.db_path)
    poll = max(5, settings.i("POLL_EVERY_S"))
    heartbeat = max(poll, settings.i("HEARTBEAT_S"))
    _log("started: api={} store={} poll={}s inactivity={}s merge_gap={}s"
         .format(settings.s("FRIGATE_API"), settings.db_path, poll,
                 settings.i("INACTIVITY_S"), settings.i("MERGE_GAP_S")))
    last_hb = 0.0
    while True:
        try:
            if settings.b("ENABLED"):
                cycle(settings, conn, places)
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            _log("ERROR in cycle: {}".format(exc))
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
    if args.once:
        conn = open_store(settings.db_path)
        cycle(settings, conn, places)
        conn.close()
        return 0
    run_forever(settings, places)
    return 0


if __name__ == "__main__":
    sys.exit(main())
