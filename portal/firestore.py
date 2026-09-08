"""Read-only access to the firewatch evidence SQLite DB (WAL).

firewatch writes <STORE_DIR>/firewatch.db with `frames` (id, camera,
captured_at, ts_utc, jpg_path, best_score, score_threshold, alerted) and
`detections` (frame_id FK, label, score, x1,y1,x2,y2) rows - schema in
firewatch/firewatch.py `_SCHEMA`. The portal is a read-only consumer: only
SELECTs run (PRAGMA query_only=ON) while firewatch stays the single writer.

Mount topology: the host ./media tree is mounted at /media in this container;
firewatch sees the same tree at /media/firewatch (STORE_DIR=/media/firewatch).
So a stored jpg_path `/media/firewatch/cam01/x.jpg` is reachable here as
`/media/cam01/x.jpg` - remapped via FIREWATCH_JPG_PREFIX -> FIREWATCH_JPG_REPLACE.

SQLite WAL reads need access to the -shm/-wal side files, so ./media is mounted
read-write into the portal container (portal never writes; query_only guards).
"""
import os
import sqlite3
import struct


def _num(value):
    """Normalize a stored numeric to float.

    firewatch writes REAL coords, but legacy rows (pre-2026-09-08 store) may
    hold raw little-endian float32 BLOBs; coerce both so the API always returns
    numbers (FastAPI cannot JSON-encode bytes).
    """
    if isinstance(value, (bytes, bytearray)):
        if len(value) >= 4:
            return struct.unpack("<f", bytes(value[:4]))[0]
        return 0.0
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def open_db(path):
    """Open the evidence DB read-only. Returns a connection or raises."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    # Safety: this module must never write to the firewatch DB.
    conn.execute("PRAGMA query_only=ON")
    return conn


def _conditions(camera=None, alerted=None, label=None,
                after=None, before=None):
    """Build (sql, params) for the frames WHERE clause + a total-count clause."""
    clauses, params = [], []
    if camera:
        clauses.append("f.camera = ?")
        params.append(camera)
    if alerted is not None:
        clauses.append("f.alerted = ?")
        params.append(1 if alerted else 0)
    if after is not None:
        clauses.append("f.captured_at >= ?")
        params.append(float(after))
    if before is not None:
        clauses.append("f.captured_at < ?")
        params.append(float(before))
    if label:
        clauses.append("EXISTS (SELECT 1 FROM detections d "
                       "WHERE d.frame_id = f.id AND d.label = ?)")
        params.append(label)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _row_to_dict(row):
    return {
        "id": row["id"],
        "camera": row["camera"],
        "captured_at": _num(row["captured_at"]),
        "ts_utc": row["ts_utc"],
        "best_score": _num(row["best_score"]),
        "score_threshold": _num(row["score_threshold"]),
        "alerted": bool(row["alerted"]),
        "jpg_path": row["jpg_path"],
    }


def list_frames(db_path, *, camera=None, alerted=None, label=None,
                after=None, before=None, limit=50, offset=0):
    """Return {items: [...], total: n} newest-first, detections embedded."""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if not db_path or not os.path.exists(db_path):
        return {"items": [], "total": 0, "note": "evidence DB not found"}
    where, params = _conditions(camera, alerted, label, after, before)
    conn = open_db(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM frames f " + where, params
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM frames f " + where +
            " ORDER BY f.captured_at DESC, f.id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        items = [_row_to_dict(r) for r in rows]
        if items:
            ids = [i["id"] for i in items]
            marks = ",".join("?" * len(ids))
            det_rows = conn.execute(
                "SELECT frame_id, label, score, x1, y1, x2, y2 "
                "FROM detections WHERE frame_id IN (" + marks + ")",
                ids,
            ).fetchall()
            by_frame = {}
            for d in det_rows:
                by_frame.setdefault(d["frame_id"], []).append({
                    "label": d["label"],
                    "score": _num(d["score"]),
                    "x1": _num(d["x1"]), "y1": _num(d["y1"]),
                    "x2": _num(d["x2"]), "y2": _num(d["y2"]),
                })
            for it in items:
                it["detections"] = by_frame.get(it["id"], [])
                it["image_url"] = "/api/fire/{}/image.jpg".format(it["id"])
        return {"items": items, "total": total}
    finally:
        conn.close()


def frame_image_path(cfg, db_path, frame_id):
    """Resolve a stored evidence JPEG to this container's path (or None)."""
    if not db_path or not os.path.exists(db_path):
        return None
    conn = open_db(db_path)
    try:
        row = conn.execute(
            "SELECT jpg_path FROM frames WHERE id = ?", (int(frame_id),)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    stored = row[0]
    prefix = cfg.get("FIREWATCH_JPG_PREFIX", "/media/firewatch/")
    replace = cfg.get("FIREWATCH_JPG_REPLACE", "/media/")
    path = stored
    if prefix and stored.startswith(prefix):
        path = replace + stored[len(prefix):]
    # Guard: the remapped path must live under the replace root and exist.
    if not (replace and path.startswith(replace) and os.path.isfile(path)):
        return None
    return path
