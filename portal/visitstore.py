"""Read-only access to the visits store (`media/visits/visits.db`).

`visits` (visits/visits.py) is the single writer; the portal only ever SELECTs
(PRAGMA query_only=ON). It turns Frigate's raw person events into ONE row per
person visit, per camera, plus the cached per-event analysis behind it:

    Estraha (cam01)  Field West  person #2  14:32-14:36  4 min

IMAGE HANDLING: nothing here touches image paths. Each visit carries its
representative Frigate `frigate_event_id`, and the portal already proxies the
frame through Frigate at `/api/events/<id>/snapshot.jpg` (`portal/app.py`). The
portal never reads a filesystem path out of a DB row.

No schema migration is performed (the portal never writes). If the DB or a table
is missing the endpoints degrade to an empty result plus a `note`.
"""
import os
import sqlite3

# store filename inside the visits store dir (visits/visits.py STORE_DB)
DB_NAME = "visits.db"


def open_db(path):
    """Open the visits DB read-only. Returns a connection or raises."""
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


def _image_url(event_id):
    """Frigate snapshot proxy for this capture (the portal serves it)."""
    if not event_id:
        return None
    return "/api/events/{}/snapshot.jpg".format(event_id)


# ---------------------------------------------------------------------------
# visits
# ---------------------------------------------------------------------------
def _visit_dict(row):
    rep = _col(row, "rep_event_id")
    return {
        "id": row["id"],
        "camera": _col(row, "camera", ""),
        "display_name": _col(row, "display_name", "") or _col(row, "camera", ""),
        "place": _col(row, "place", ""),
        "day": _col(row, "day", ""),
        "person_no": int(_num(_col(row, "person_no"))),
        "enter_time": _num(_col(row, "enter_time")),
        "leave_time": _num(_col(row, "leave_time")),
        "duration_s": _num(_col(row, "duration_s")),
        "n_events": int(_num(_col(row, "n_events"))),
        "moved": bool(_col(row, "moved")),
        "edge": bool(_col(row, "edge")),
        "significant": bool(_col(row, "significant")),
        "image_url": _image_url(rep),
    }


def _visit_events(conn, visit_id):
    """The ordered Frigate captures behind one visit (snapshot proxy URLs)."""
    try:
        rows = conn.execute(
            "SELECT frigate_event_id FROM visit_events WHERE visit_id = ?",
            (visit_id,)).fetchall()
    except sqlite3.Error:
        return []
    return [{"frigate_event_id": r["frigate_event_id"],
             "image_url": _image_url(r["frigate_event_id"])} for r in rows]


def list_visits(db_path, *, day=None, camera=None, after=None, before=None,
                significant=None, limit=50, offset=0):
    """Visits, newest first. `{items, total}` or `{items: [], note: ...}`.

    `significant` (None = every visit, True/False = filter) mirrors the
    visits service's own text-log rule: a visit whose whole presence never
    really moved and never touched a frame edge is stored with significant=0.
    """
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if not db_path or not os.path.exists(db_path):
        return {"items": [], "total": 0, "note": "visits store not found"}
    conn = open_db(db_path)
    try:
        if "visits" not in _tables(conn):
            return {"items": [], "total": 0, "note": "no visits table yet"}
        clauses, params = [], []
        if day:
            clauses.append("day = ?")
            params.append(str(day))
        if camera:
            clauses.append("camera = ?")
            params.append(camera)
        if after is not None:
            clauses.append("leave_time >= ?")
            params.append(float(after))
        if before is not None:
            clauses.append("enter_time < ?")
            params.append(float(before))
        if significant is not None:
            clauses.append("significant = ?")
            params.append(1 if significant else 0)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        total = conn.execute("SELECT COUNT(*) FROM visits " + where,
                             params).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM visits " + where +
            " ORDER BY enter_time DESC, id DESC LIMIT ? OFFSET ?",
            params + [limit, offset]).fetchall()
        return {"items": [_visit_dict(r) for r in rows], "total": total}
    except sqlite3.OperationalError as exc:
        return {"items": [], "total": 0, "note": str(exc)}
    finally:
        conn.close()


def get_visit(db_path, visit_id):
    """One visit WITH its ordered Frigate captures, or None."""
    if not db_path or not os.path.exists(db_path):
        return None
    conn = open_db(db_path)
    try:
        if "visits" not in _tables(conn):
            return None
        try:
            row = conn.execute("SELECT * FROM visits WHERE id = ?",
                               (int(visit_id),)).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        item = _visit_dict(row)
        item["events"] = _visit_events(conn, row["id"])
        return item
    finally:
        conn.close()


def days(db_path):
    """Distinct days that have visits (newest first), for a date filter."""
    if not db_path or not os.path.exists(db_path):
        return []
    conn = open_db(db_path)
    try:
        if "visits" not in _tables(conn):
            return []
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT day FROM visits ORDER BY day DESC LIMIT 60")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def cameras(db_path):
    """Distinct cameras that have visits (alphabetical), for a filter."""
    if not db_path or not os.path.exists(db_path):
        return []
    conn = open_db(db_path)
    try:
        if "visits" not in _tables(conn):
            return []
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT camera FROM visits WHERE camera IS NOT NULL"
            " ORDER BY camera LIMIT 100")]
    except sqlite3.Error:
        return []
    finally:
        conn.close()
