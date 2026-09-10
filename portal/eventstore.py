"""Read-only access to the scenereader store (`media/events/events.db`).

`scenereader` is the single writer; the portal only ever SELECTs (PRAGMA
query_only=ON). Two things are exposed:

  * **episodes** - the cross-camera person stories, each with BOTH narratives
    (`narrative` English, `narrative_ar` Arabic) and its ordered visits. This is
    the required output ("Person A entered Field 1 ... then went to Store").
  * **events** - the per-capture scene log (deterministic description + optional
    VLM caption + importance tier).

IMAGE HANDLING: nothing here touches image paths. Every visit/event carries the
Frigate `frigate_event_id`, and the portal already proxies the frame through
Frigate itself at `/api/events/<id>/snapshot.jpg` (`portal/app.py`). That means
scenereader never has to copy a frame AND the portal never has to read a
filesystem path from a DB row - the arbitrary-file-read class of bug cannot
exist here.

No schema migration is performed (the portal never writes). If the DB or a table
is missing the endpoints degrade to an empty result plus a `note`.
"""
import json
import os
import sqlite3

# store filenames inside the scenereader store dir
DB_NAME = "events.db"
STATUS_NAME = "reader_status.json"
TRIGGER_NAME = ".drain_request"


def store_dir_from_db(db_path):
    return os.path.dirname(db_path) or "."


def default_paths(store_dir):
    """(db, status, trigger) for a store dir - keeps the names in one place."""
    store_dir = store_dir.rstrip("/") or "."
    return (os.path.join(store_dir, DB_NAME),
            os.path.join(store_dir, STATUS_NAME),
            os.path.join(store_dir, TRIGGER_NAME))


def open_db(path):
    """Open the scenereader DB read-only. Returns a connection or raises."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA query_only=ON")   # this module must never write
    return conn


def _num(value):
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _col(row, name, default=None):
    """sqlite3.Row has no .get(): tolerate a column a newer writer may add."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _tables(conn):
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.Error:
        return set()


def _image_url(row):
    """Frigate snapshot proxy for this capture (the portal serves it)."""
    event_id = _col(row, "frigate_event_id")
    if not event_id:
        return None
    return "/api/events/{}/snapshot.jpg".format(event_id)


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
def _episode_dict(row):
    return {
        "id": row["id"],
        "day": _col(row, "day", ""),
        "anon_name": _col(row, "anon_name", ""),
        "label": _col(row, "label", "person"),
        # The narratives are built deterministically by scenereader from the
        # structured visits - both languages from the same facts.
        "narrative": _col(row, "narrative", "") or "",
        "narrative_ar": _col(row, "narrative_ar", "") or "",
        "person_name": _col(row, "person_name"),
        "start_time": _num(_col(row, "start_time")),
        "end_time": _num(_col(row, "end_time")),
        "link_confidence": _col(row, "link_confidence"),
    }


def _visits(conn, episode_id):
    try:
        rows = conn.execute(
            "SELECT ee.seq, ee.place, ee.enter_time, ee.leave_time, ee.duration_s,"
            " e.camera, e.label, e.frigate_event_id, e.description_meta,"
            " e.description_vlm, e.vlm_status, e.tier, e.importance"
            " FROM episode_events ee JOIN events e ON e.id = ee.event_id"
            " WHERE ee.episode_id = ? ORDER BY ee.seq ASC", (episode_id,)).fetchall()
    except sqlite3.Error:
        return []
    return [{
        "seq": r["seq"],
        "place": r["place"],
        "camera": r["camera"],
        "label": r["label"],
        "enter_time": _num(r["enter_time"]),
        "leave_time": _num(r["leave_time"]),
        "duration_s": _num(r["duration_s"]),
        "description_meta": r["description_meta"],
        "description_vlm": r["description_vlm"],
        "vlm_status": r["vlm_status"],
        "tier": r["tier"],
        "importance": int(_num(r["importance"])),
        "image_url": _image_url(r),
    } for r in rows]


def list_episodes(db_path, *, day=None, camera=None, after=None, before=None,
                  limit=50, offset=0, with_visits=False):
    """Episodes, newest first. `{items, total}` or `{items: [], note: ...}`."""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if not db_path or not os.path.exists(db_path):
        return {"items": [], "total": 0, "note": "scenereader store not found"}
    conn = open_db(db_path)
    try:
        if "episodes" not in _tables(conn):
            return {"items": [], "total": 0, "note": "no episodes table yet"}
        clauses, params = [], []
        if day:
            clauses.append("day = ?")
            params.append(str(day))
        if after is not None:
            clauses.append("start_time >= ?")
            params.append(float(after))
        if before is not None:
            clauses.append("start_time < ?")
            params.append(float(before))
        if camera:
            # a camera filter on an EPISODE means "touched this camera at all"
            clauses.append("id IN (SELECT ee.episode_id FROM episode_events ee"
                           " JOIN events e ON e.id = ee.event_id WHERE e.camera = ?)")
            params.append(camera)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        total = conn.execute("SELECT COUNT(*) FROM episodes " + where,
                             params).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM episodes " + where +
            " ORDER BY start_time DESC, id DESC LIMIT ? OFFSET ?",
            params + [limit, offset]).fetchall()
        items = []
        for row in rows:
            item = _episode_dict(row)
            item["visit_count"] = conn.execute(
                "SELECT COUNT(*) FROM episode_events WHERE episode_id = ?",
                (row["id"],)).fetchone()[0]
            if with_visits:
                item["visits"] = _visits(conn, row["id"])
            items.append(item)
        return {"items": items, "total": total}
    except sqlite3.OperationalError as exc:
        return {"items": [], "total": 0, "note": str(exc)}
    finally:
        conn.close()


def get_episode(db_path, episode_id):
    """One episode WITH its ordered visits, or None."""
    if not db_path or not os.path.exists(db_path):
        return None
    conn = open_db(db_path)
    try:
        if "episodes" not in _tables(conn):
            return None
        try:
            row = conn.execute("SELECT * FROM episodes WHERE id = ?",
                               (int(episode_id),)).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        item = _episode_dict(row)
        item["visits"] = _visits(conn, row["id"])
        item["visit_count"] = len(item["visits"])
        return item
    finally:
        conn.close()


def days(db_path):
    """Distinct days that have episodes (newest first), for a date filter."""
    if not db_path or not os.path.exists(db_path):
        return []
    conn = open_db(db_path)
    try:
        if "episodes" not in _tables(conn):
            return []
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT day FROM episodes ORDER BY day DESC LIMIT 60")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# events (the per-capture scene log)
# ---------------------------------------------------------------------------
def _event_dict(row):
    return {
        "id": row["id"],
        "frigate_event_id": _col(row, "frigate_event_id", ""),
        "camera": _col(row, "camera", ""),
        "label": _col(row, "label", ""),
        "sub_label": _col(row, "sub_label"),
        "place": _col(row, "place"),
        "start_time": _num(_col(row, "start_time")),
        "end_time": _num(_col(row, "end_time")),
        "duration": _num(_col(row, "duration")),
        "score": _col(row, "score"),
        "description_meta": _col(row, "description_meta"),
        "description_vlm": _col(row, "description_vlm"),
        "vlm_status": _col(row, "vlm_status"),
        "tier": _col(row, "tier", "normal") or "normal",
        "importance": int(_num(_col(row, "importance"))),
        "episode_id": _col(row, "episode_id"),
        "image_url": _image_url(row),
    }


def list_events(db_path, *, camera=None, label=None, min_tier=None,
                has_caption=None, after=None, before=None,
                sort="time", limit=50, offset=0):
    """Captures, newest (or most important) first. Degrades with a `note`."""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if not db_path or not os.path.exists(db_path):
        return {"items": [], "total": 0, "note": "scenereader store not found"}
    conn = open_db(db_path)
    try:
        if "events" not in _tables(conn):
            return {"items": [], "total": 0, "note": "no events table yet"}
        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        clauses, params = [], []
        if camera:
            clauses.append("camera = ?")
            params.append(camera)
        if label:
            clauses.append("label = ?")
            params.append(label)
        # "at least this important": high -> high; normal -> high+normal
        tiers = {"high": ("high",), "normal": ("high", "normal")}
        if min_tier in tiers and "tier" in cols:
            clauses.append("tier IN (" + ",".join("?" * len(tiers[min_tier])) + ")")
            params.extend(tiers[min_tier])
        if has_caption is True:
            clauses.append("description_vlm IS NOT NULL AND description_vlm != ''")
        elif has_caption is False:
            clauses.append("(description_vlm IS NULL OR description_vlm = '')")
        if after is not None:
            clauses.append("start_time >= ?")
            params.append(float(after))
        if before is not None:
            clauses.append("start_time < ?")
            params.append(float(before))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        if sort == "importance" and "importance" in cols:
            order = "ORDER BY importance DESC, start_time DESC, id DESC"
        else:
            order = "ORDER BY start_time DESC, id DESC"
        total = conn.execute("SELECT COUNT(*) FROM events " + where,
                             params).fetchone()[0]
        rows = conn.execute("SELECT * FROM events " + where + " " + order +
                            " LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        return {"items": [_event_dict(r) for r in rows], "total": total}
    except sqlite3.OperationalError as exc:
        return {"items": [], "total": 0, "note": str(exc)}
    finally:
        conn.close()


def get_event(db_path, event_id):
    """One capture by local row id (used by the legacy Scenes-tab shim)."""
    if not db_path or not os.path.exists(db_path):
        return None
    conn = open_db(db_path)
    try:
        if "events" not in _tables(conn):
            return None
        try:
            row = conn.execute("SELECT * FROM events WHERE id = ?",
                               (int(event_id),)).fetchone()
        except sqlite3.OperationalError:
            return None
        return _event_dict(row) if row is not None else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# service status / trigger
# ---------------------------------------------------------------------------
def read_status(status_path):
    """The service heartbeat file (empty dict when absent/unreadable)."""
    try:
        with open(status_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def request_drain(trigger_path):
    """Ask scenereader for a caption batch now (its 'Process now' button).

    A flag file on the shared ./media mount, consumed and cleared by the service
    on its next tick. Chosen over a network port: no new attack surface, and the
    portal already mounts ./media read-write.
    """
    try:
        os.makedirs(os.path.dirname(trigger_path) or ".", exist_ok=True)
        with open(trigger_path, "w", encoding="utf-8") as fh:
            fh.write(str(int(__import__("time").time())))
        return True
    except OSError:
        return False
