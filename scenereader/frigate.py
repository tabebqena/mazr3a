"""frigate.py - read-only access to Frigate's captures (disk) and metadata (DB).

Verified on the production host (2026-09-10, see plans/event-scene-reader.md §6):

  * Frigate writes event snapshots into `media/clips/` (there is NO
    `media/snapshots/`), as a PAIR per event:
        <camera>-<event_id>.jpg          <- annotated (bounding box / timestamp)
        <camera>-<event_id>-clean.webp   <- un-annotated  (used for captioning)
    with `clips/{previews,thumbs,export,review,cache}` subdirs that must be
    skipped (we scan the TOP LEVEL only).
  * **The filename carries the Frigate event id**, e.g.
    `cam01-1789069083.752571-8xvwzu.jpg` <-> `event.id = '1789069083.752571-8xvwzu'`,
    so the disk scan joins the DB by EXACT id - no fuzzy time matching.
  * Frigate's `config/frigate.db` has a single denormalised `event` table holding
    camera/label/sub_label/zones/start_time/end_time/has_snapshot, and it opens
    READ-ONLY (`mode=ro`) while Frigate is running. NOTE: the `score`, `top_score`
    and `box` COLUMNS ARE NULL - the live values live inside the `data` JSON
    (keys: score, top_score, box, region, attributes, path_data, max_severity).
  * The REST API `/api/events/<id>` returns the same fields, version-sanitised,
    and is the fallback when the DB cannot be opened.

Nothing here writes anything, anywhere.
"""
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request

# <camera>-<epoch>.<micro>-<hash>.jpg  (camera may itself contain a '-')
_CLIP_RE = re.compile(r"^(?P<cam>.+?)-(?P<ts>\d{9,}(?:\.\d+)?)-(?P<hash>[0-9a-z]+)\.jpg$")

# Frigate's `data` JSON carries the values that the flat columns do not.
_DATA_KEYS = ("score", "top_score", "box", "region", "attributes", "path_data",
              "average_estimated_speed", "velocity_angle", "max_severity")


# ---------------------------------------------------------------------------
# disk: the event snapshots
# ---------------------------------------------------------------------------
def parse_clip_name(name):
    """`cam01-1789069083.752571-8xvwzu.jpg` -> (camera, event_id, start_time).

    Returns None for anything that is not an event snapshot (the `-clean.webp`
    siblings, `clip.mp4`, previews/thumbs, temp files...).
    """
    match = _CLIP_RE.match(name or "")
    if not match:
        return None
    camera = match.group("cam")
    ts, digest = match.group("ts"), match.group("hash")
    try:
        start_time = float(ts)
    except ValueError:
        return None
    return camera, ts + "-" + digest, start_time


def clean_frame_path(jpg_path):
    """The un-annotated sibling of an annotated snapshot, if it exists."""
    base = jpg_path[:-4] if jpg_path.lower().endswith(".jpg") else jpg_path
    candidate = base + "-clean.webp"
    return candidate if os.path.isfile(candidate) else None


def scan_clips(clips_dir, since=0.0, limit=None, want_clean=True):
    """Event snapshots in `clips_dir` (TOP LEVEL only), newest first.

    `since` filters by the event start time parsed from the name (use the newest
    known start_time as a cursor to avoid re-reading the whole directory). Each
    item is a dict ready for `store.upsert_event`:
        {frigate_event_id, camera, start_time, frame_path, frame_clean_path}
    """
    if not clips_dir or not os.path.isdir(clips_dir):
        return []
    items = []
    try:
        with os.scandir(clips_dir) as it:
            for entry in it:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                parsed = parse_clip_name(entry.name)
                if parsed is None:
                    continue
                camera, event_id, start_time = parsed
                if since and start_time <= float(since):
                    continue
                path = entry.path
                items.append({
                    "frigate_event_id": event_id,
                    "camera": camera,
                    "start_time": start_time,
                    "frame_path": path,
                    "frame_clean_path": clean_frame_path(path) if want_clean else None,
                })
    except OSError:
        return []
    items.sort(key=lambda d: d["start_time"], reverse=True)
    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


# ---------------------------------------------------------------------------
# DB: the `event` table (primary metadata source)
# ---------------------------------------------------------------------------
def open_frigate_db(path):
    """Open Frigate's DB READ-ONLY. Returns a connection or None.

    `mode=ro` is preferred (it replays the WAL, so the newest events are
    visible). `immutable=1` is the fallback for a filesystem where the WAL index
    cannot be touched - it may lag by whatever is still in the WAL.
    """
    if not path or not os.path.isfile(path):
        return None
    for uri in ("file:" + path + "?mode=ro", "file:" + path + "?immutable=1"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM event LIMIT 1").fetchone()
            return conn
        except sqlite3.Error:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - closing a bad handle is best-effort
                pass
            continue
    return None


def _data_fields(raw):
    """Pull the live values out of Frigate's `data` JSON (columns are NULL)."""
    out = {}
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    for key in _DATA_KEYS:
        if key in data:
            out[key] = data[key]
    return out


def _normalize(row):
    """A raw `event` row -> the flat dict the rest of the code expects."""
    zones = row["zones"] if "zones" in row.keys() else None
    try:
        zones_list = json.loads(zones) if zones else []
    except (TypeError, ValueError):
        zones_list = []
    if not isinstance(zones_list, list):
        zones_list = []
    data = _data_fields(row["data"] if "data" in row.keys() else None)
    meta = {
        "frigate_event_id": row["id"],
        "camera": row["camera"],
        "label": row["label"] or "",
        "sub_label": row["sub_label"],
        "zones": json.dumps(zones_list),
        "start_time": float(row["start_time"] or 0),
        "end_time": float(row["end_time"]) if row["end_time"] else None,
        "has_snapshot": int(row["has_snapshot"] or 0),
        "false_positive": int(row["false_positive"] or 0),
        "score": data.get("score"),
        "top_score": data.get("top_score"),
        "box": json.dumps(data["box"]) if data.get("box") else None,
        "meta_json": row["data"] if "data" in row.keys() else None,
    }
    start = meta["start_time"]
    meta["duration"] = (meta["end_time"] - start) if meta["end_time"] else None
    return meta


def lookup_event(conn, event_id):
    """One event by primary key, normalised; None when absent."""
    if conn is None or not event_id:
        return None
    try:
        row = conn.execute("SELECT * FROM event WHERE id = ?", (event_id,)).fetchone()
    except sqlite3.Error:
        return None
    return _normalize(row) if row is not None else None


# ---------------------------------------------------------------------------
# REST: fallback metadata source
# ---------------------------------------------------------------------------
def event_from_api(api_base, event_id, timeout=8.0):
    """`GET {api}/api/events/<id>` normalised like `lookup_event`; None on error."""
    if not api_base or not event_id:
        return None
    url = api_base.rstrip("/") + "/api/events/" + str(event_id)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(obj, dict) or not obj.get("id"):
        return None
    zones = obj.get("zones") or []
    try:
        start = float(obj.get("start_time") or 0)
    except (TypeError, ValueError):
        start = 0.0
    end = obj.get("end_time")
    try:
        end = float(end) if end else None
    except (TypeError, ValueError):
        end = None
    meta = {
        "frigate_event_id": obj["id"],
        "camera": obj.get("camera") or "",
        "label": obj.get("label") or "",
        "sub_label": obj.get("sub_label"),
        "zones": json.dumps(zones if isinstance(zones, list) else []),
        "start_time": start,
        "end_time": end,
        "duration": (end - start) if end else None,
        "has_snapshot": int(bool(obj.get("has_snapshot"))),
        "false_positive": int(bool(obj.get("false_positive"))),
        "score": obj.get("score"),
        "top_score": obj.get("top_score"),
        "box": json.dumps(obj["box"]) if obj.get("box") else None,
        "meta_json": json.dumps(obj),
    }
    return meta


def event_metadata(event_id, db_conn=None, api_base="", source="db", timeout=8.0):
    """Metadata for one event, honouring FRIGATE_METADATA_SOURCE.

    `source`: "db" (default), "api", or "auto" (try db, then api). Returns None
    when nothing answers - the caller then stores a PROVISIONAL row and retries
    on a later scan (never blocks, never raises).
    """
    source = (source or "db").lower()
    lookups = []
    if source in ("db", "auto"):
        lookups.append(lambda: lookup_event(db_conn, event_id))
    if source in ("api", "auto"):
        lookups.append(lambda: event_from_api(api_base, event_id, timeout=timeout))
    for lookup in lookups:
        try:
            meta = lookup()
        except Exception:  # noqa: BLE001 - a metadata lookup must never break a scan
            meta = None
        if meta:
            return meta
    return None
