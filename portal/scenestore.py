"""Read-only access to the scenewatch scene-description SQLite DB (WAL).

scenewatch writes <STORE_DIR>/scenewatch.db with a `scenes` table
(id, camera, captured_at, ts_utc, description, motion_frac, reason, model,
latency_ms, jpg_path) - schema in `scenewatch/scenewatch.py` `_SCHEMA`. The
portal is a read-only consumer: only SELECTs run (PRAGMA query_only=ON) while
scenewatch stays the single writer.

Mount topology: the host `./media` tree is mounted at `/media` in this
container, and scenewatch writes `STORE_DIR=/media/scenewatch`, so a stored
absolute `jpg_path` such as `/media/scenewatch/cam01/x.jpg` is directly
reachable here - no prefix remap is needed (unlike the firewatch evidence DB,
which stores `/media/firewatch/...` because of its mount). `scene_image_path()`
still validates that the file exists AND lives under the scene-store root
before the portal will serve it.

No schema migration is performed (the portal never writes); if the DB exists but
scenewatch has not created the `scenes` table yet, the list endpoints degrade to
an empty result with a `note` instead of raising.
"""
import os
import sqlite3


def _num(value):
    """Normalize a stored numeric to float (REAL, but tolerate BLOB/None)."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) >= 4:
            import struct
            return struct.unpack("<f", bytes(value[:4]))[0]
        return 0.0
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def open_db(path):
    """Open the scene DB read-only. Returns a connection or raises."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    # Safety: this module must never write to the scenewatch DB.
    conn.execute("PRAGMA query_only=ON")
    return conn


# Importance tiers the writer assigns (scenewatch score_importance). Ordering
# matters: a "at least this tier" filter is expressed as a set of tiers.
_TIER_ORDER = ("high", "normal", "low")


def _col(row, name, default=None):
    """sqlite3.Row has no .get(): tolerate a column added by a later writer."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _table_columns(conn, table):
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
    except sqlite3.Error:
        return set()


def _conditions(camera=None, reason=None, min_tier=None, after=None,
                before=None, with_image=None, cols=None):
    """Build (sql, params) for the scenes WHERE clause.

    `min_tier` is "at least this important": 'high' -> only high; 'normal' ->
    high+normal; 'low'/None -> no tier restriction.
    """
    cols = cols or set()
    clauses, params = [], []
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if reason:
        clauses.append("reason = ?")
        params.append(reason)
    if min_tier in _TIER_ORDER and "tier" in cols:
        keep = _TIER_ORDER[: _TIER_ORDER.index(min_tier) + 1]
        clauses.append("tier IN (" + ",".join("?" * len(keep)) + ")")
        params.extend(keep)
    if after is not None:
        clauses.append("captured_at >= ?")
        params.append(float(after))
    if before is not None:
        clauses.append("captured_at < ?")
        params.append(float(before))
    if with_image is True:
        clauses.append("jpg_path IS NOT NULL AND jpg_path != ''")
    elif with_image is False:
        clauses.append("(jpg_path IS NULL OR jpg_path = '')")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _row_to_dict(row):
    jpg = row["jpg_path"]
    return {
        "id": row["id"],
        "camera": row["camera"],
        "captured_at": _num(row["captured_at"]),
        "ts_utc": row["ts_utc"],
        "description": row["description"],
        "motion_frac": _num(row["motion_frac"]),
        "reason": row["reason"],
        "model": row["model"],
        "latency_ms": int(_num(row["latency_ms"])),
        # Writer-side importance (0-100 + coarse tier). Defaults keep rows from
        # a pre-importance DB visible rather than hiding everything.
        "importance": int(_num(_col(row, "importance", 0))),
        "tier": _col(row, "tier", "normal") or "normal",
        "has_image": bool(jpg),
        "image_url": "/api/scenes/{}/image.jpg".format(row["id"]) if jpg else None,
    }


def list_scenes(db_path, *, camera=None, reason=None, min_tier=None,
                after=None, before=None, with_image=None, sort="importance",
                limit=50, offset=0):
    """Return {items: [...], total: n}.

    `sort` is "importance" (most important first, then newest - the Scenes tab
    default, so the rows that matter appear on first opening) or "time"
    (newest first). Degrades to an empty list (with a `note`) when the DB or the
    `scenes` table does not exist yet - e.g. scenewatch never captioned.
    """
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if not db_path or not os.path.exists(db_path):
        return {"items": [], "total": 0, "note": "scene DB not found"}
    conn = open_db(db_path)
    try:
        cols = _table_columns(conn, "scenes")
        if not cols:
            return {"items": [], "total": 0, "note": "scenes table not found"}
        where, params = _conditions(camera, reason, min_tier, after, before,
                                    with_image, cols)
        # Importance ordering needs the column; fall back to time otherwise.
        if sort == "time" or "importance" not in cols:
            order = "ORDER BY captured_at DESC, id DESC"
        else:
            order = "ORDER BY importance DESC, captured_at DESC, id DESC"
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM scenes " + where, params
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM scenes " + where + " " + order +
                " LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # No `scenes` table yet (scenewatch has not written anything).
            return {"items": [], "total": 0, "note": str(exc)}
        return {"items": [_row_to_dict(r) for r in rows], "total": total}
    finally:
        conn.close()


def scene_image_path(db_path, scene_id, root=None):
    """Resolve a stored scene JPEG to a path this container can serve (or None).

    Accepts the stored path only when it exists AND sits under the scene-store
    root (default: the directory holding the DB). That keeps a malformed/foreign
    `jpg_path` from turning the image endpoint into an arbitrary file read.
    """
    if not db_path or not os.path.exists(db_path):
        return None
    conn = open_db(db_path)
    try:
        try:
            row = conn.execute(
                "SELECT jpg_path FROM scenes WHERE id = ?", (int(scene_id),)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    finally:
        conn.close()
    if row is None:
        return None
    stored = row[0]
    if not stored:
        return None
    root = os.path.realpath(root) if root else os.path.realpath(
        os.path.dirname(db_path))
    path = os.path.realpath(stored)
    if not path.startswith(root + os.sep):
        return None
    return path if os.path.isfile(path) else None
