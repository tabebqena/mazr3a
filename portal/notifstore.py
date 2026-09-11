"""Portal notification feed (SQLite) - the portal's OWN notification store.

The portal keeps ONE global notification timeline that every signed-in user sees;
what is per-user is only the READ position. This module is the single writer of
`media/portal/notifications.db` (the same pattern as portal/usage.py and
portal/userstore.py - a small DB under the rw ./media mount the portal owns).

Tables:
  * notifications      - the timeline (kind, title, body, url, camera,
                         dedup_key, created_at). A PARTIAL UNIQUE index on a
                         non-empty `dedup_key` makes ingestion idempotent, so a
                         poller can re-scan the same source safely.
  * notification_state - per-user read marker `read_upto_id`. Ids are monotonic,
                         so "read" is simply `id <= read_upto_id`; opening the
                         tab / marking read advances the marker. This keeps read
                         state O(users) instead of O(users x notifications).
  * notification_meta  - tiny key/value store (the fire-alert watcher's
                         high-water mark and similar small facts).

Sources feeding the timeline (see portal/app.py):
  * the portal's background fire-alert watcher records one item per new
    Firewatch `alerted=1` frame (dedup_key `fire:<id>`);
  * an admin can publish a system message via POST /api/admin/notifications.

RETENTION is in-app (`prune()`), NOT the host disk heartbeat: scripts/
heartbeat_cleanup.py hard-protects *.db and cannot delete rows.
"""
import os
import sqlite3
import time

# Notification kinds the feed recognises (free-form, but these drive UI colour).
KINDS = ("fire", "system", "info")

# Generated avatar-style guard for title/body length (defensive; the SPA also
# limits the admin publish form).
MAX_TITLE = 200
MAX_BODY = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL DEFAULT 'info',
    title       TEXT    NOT NULL DEFAULT '',
    body        TEXT    NOT NULL DEFAULT '',
    url         TEXT    NOT NULL DEFAULT '',
    camera      TEXT    NOT NULL DEFAULT '',
    dedup_key   TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL
);
-- Idempotent ingestion: a non-empty dedup_key may exist only once. The partial
-- WHERE keeps the many rows with an empty dedup_key (admin messages) allowed.
CREATE UNIQUE INDEX IF NOT EXISTS notifications_dedup
    ON notifications (dedup_key) WHERE dedup_key <> '';
CREATE INDEX IF NOT EXISTS notifications_created
    ON notifications (created_at);
CREATE TABLE IF NOT EXISTS notification_state (
    username     TEXT PRIMARY KEY,
    read_upto_id INTEGER NOT NULL DEFAULT 0,
    updated_at   REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS notification_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""


class NotificationStore:
    """SQLite-backed notification feed, safe to read on every request.

    Single-process by design (the portal runs one uvicorn worker); each call
    opens a short-lived connection, so no cross-thread state is shared.
    """

    def __init__(self, path, retention_days=90):
        self.path = path
        self.retention_days = max(1, int(retention_days))

    # ---- setup -----------------------------------------------------------
    def configure(self):
        """Create the DB dir/table. Raises only on a genuinely unusable path."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---- write -----------------------------------------------------------
    def add(self, kind="info", title="", body="", url="", camera="",
            dedup_key="", created_at=None):
        """Insert one notification. Returns the item dict, or None.

        None means either invalid input (empty title) or a duplicate
        `dedup_key` (already recorded) - callers treat both as "not added".
        """
        title = str(title or "").strip()
        if not title:
            return None
        try:
            conn = self._connect()
        except sqlite3.Error:
            return None
        try:
            cur = conn.execute(
                "INSERT INTO notifications (kind, title, body, url, camera, "
                "dedup_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(kind or "info").strip()[:24] or "info",
                 title[:MAX_TITLE],
                 str(body or "")[:MAX_BODY],
                 str(url or "")[:300],
                 str(camera or "")[:64],
                 str(dedup_key or "").strip()[:120],
                 float(created_at) if created_at else time.time()))
            conn.commit()
            row = conn.execute("SELECT * FROM notifications WHERE id = ?",
                               (cur.lastrowid,)).fetchone()
            return self._item(row, 0) if row else None
        except sqlite3.IntegrityError:
            return None      # duplicate dedup_key - already recorded
        except sqlite3.Error:
            return None
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ---- meta (small key/value facts) ------------------------------------
    def get_meta(self, key, default=None):
        try:
            conn = self._connect()
        except sqlite3.Error:
            return default
        try:
            row = conn.execute("SELECT value FROM notification_meta WHERE key = ?",
                               (str(key),)).fetchone()
            return row["value"] if row else default
        except sqlite3.Error:
            return default
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def set_meta(self, key, value):
        try:
            conn = self._connect()
        except sqlite3.Error:
            return False
        try:
            conn.execute(
                "INSERT INTO notification_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(key), str(value)))
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ---- read ------------------------------------------------------------
    @staticmethod
    def _item(row, read_upto):
        return {
            "id": row["id"],
            "kind": row["kind"] or "info",
            "title": row["title"] or "",
            "body": row["body"] or "",
            "url": row["url"] or "",
            "camera": row["camera"] or "",
            "created_at": float(row["created_at"] or 0),
            "read": int(row["id"]) <= int(read_upto),
        }

    @staticmethod
    def _read_upto(conn, username):
        try:
            row = conn.execute(
                "SELECT read_upto_id FROM notification_state WHERE username = ?",
                (str(username or ""),)).fetchone()
            return int(row["read_upto_id"]) if row else 0
        except sqlite3.Error:
            return 0

    @staticmethod
    def _set_read_upto(conn, username, value):
        conn.execute(
            "INSERT INTO notification_state (username, read_upto_id, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(username) DO UPDATE SET "
            "read_upto_id = excluded.read_upto_id, updated_at = excluded.updated_at",
            (str(username or ""), int(value), time.time()))

    def list_for(self, username, limit=50, offset=0, after_id=None):
        """The feed for one user.

        Without `after_id`: a newest-first PAGE (limit/offset).
        With `after_id`: the NEWER items, ASCENDING (max `limit`) - the shape a
        client poller wants to raise one browser alert per new item.

        Always carries `unread` (count of id > read_upto_id), `total` and
        `latest_id` so a client can baseline without a second call.
        """
        try:
            limit = max(1, min(int(limit), 200))
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            limit, offset = 50, 0
        username = str(username or "")
        out = {"items": [], "unread": 0, "total": 0, "latest_id": 0}
        try:
            conn = self._connect()
        except sqlite3.Error:
            return out
        try:
            read_upto = self._read_upto(conn, username)
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS mx "
                "FROM notifications").fetchone()
            out["total"] = int(row["n"]) if row else 0
            out["latest_id"] = int(row["mx"]) if row else 0
            out["unread"] = max(0, int(conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE id > ?",
                (read_upto,)).fetchone()[0]))
            if after_id is not None:
                try:
                    after_id = int(after_id)
                except (TypeError, ValueError):
                    after_id = 0
                rows = conn.execute(
                    "SELECT * FROM notifications WHERE id > ? "
                    "ORDER BY id ASC LIMIT ?", (after_id, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM notifications ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset)).fetchall()
            out["items"] = [self._item(r, read_upto) for r in rows]
            return out
        except sqlite3.Error:
            return out
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def unread(self, username):
        """How many notifications this user has not seen yet."""
        try:
            conn = self._connect()
        except sqlite3.Error:
            return 0
        try:
            read_upto = self._read_upto(conn, username)
            row = conn.execute("SELECT COUNT(*) FROM notifications WHERE id > ?",
                               (read_upto,)).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ---- read state ------------------------------------------------------
    def mark_read(self, username, ids):
        """Advance a user's read marker past the given ids. Returns new unread."""
        try:
            ids = [int(i) for i in (ids or [])
                   if str(i).strip().lstrip("-").isdigit()]
        except (TypeError, ValueError):
            ids = []
        if ids:
            try:
                conn = self._connect()
            except sqlite3.Error:
                return self.unread(username)
            try:
                current = self._read_upto(conn, username)
                target = max(current, max(ids))
                if target > current:
                    self._set_read_upto(conn, username, target)
                    conn.commit()
            except sqlite3.Error:
                pass
            finally:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        return self.unread(username)

    def mark_all_read(self, username):
        """Mark every existing notification read. Returns new unread (0)."""
        try:
            conn = self._connect()
        except sqlite3.Error:
            return self.unread(username)
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS mx FROM notifications").fetchone()
            self._set_read_upto(conn, username, int(row["mx"]) if row else 0)
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        return self.unread(username)

    # ---- retention -------------------------------------------------------
    def prune(self, retention_days=None):
        """Delete notifications older than the retention window. Returns count."""
        days = self.retention_days if retention_days is None else int(retention_days)
        cutoff = time.time() - days * 86400
        try:
            conn = self._connect()
        except sqlite3.Error:
            return 0
        try:
            cur = conn.execute("DELETE FROM notifications WHERE created_at < ?",
                               (cutoff,))
            conn.commit()
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        except sqlite3.Error:
            return 0
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
