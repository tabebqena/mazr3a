"""Per-user bandwidth accounting for the portal (Live + Events video).

The portal is the single EGRESS CHOKEPOINT for every byte a logged-in user
receives: Live (MSE over WebSocket, HLS fallback), event clips and snapshots are
all proxied server-side through the portal, behind the session cookie. That makes
the portal itself the accurate place to measure real egress per username.

UNLIKE eventstore.py / firestore.py (read-only consumers of DBs another service
writes), the portal IS THE WRITER of this one DB. It records DAILY COUNTERS
keyed by (username, day, kind) - no per-camera breakdown - and prunes them to a
retention window.

   kind = "live"    live video transports (MSE WS + HLS playlist/segments)
   kind = "events"  event video clips (/api/events/<id>/clip.mp4)
   kind = "other"   snapshots (live/event) + the events JSON listing

COUNTING MUST NEVER AFFECT PLAYBACK. `add()` only bumps an in-process
accumulator: no I/O, never raises. A periodic task (see `run()`) persists the
accumulator with atomic UPSERTs; any DB error is swallowed and the batch is kept
for the next flush. Bytes still buffered are lost on a container restart - the
flush interval bounds that.

RETENTION is in-app (`prune()`), NOT the host disk heartbeat: scripts/
heartbeat_cleanup.py hard-protects *.db and cannot delete rows.
"""
import asyncio
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

# Byte kinds the portal accounts for. Keep in sync with the callers in app.py.
KINDS = ("live", "events", "other")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_daily (
    username   TEXT    NOT NULL,
    day        TEXT    NOT NULL,   -- UTC YYYY-MM-DD
    kind       TEXT    NOT NULL,   -- live | events | other
    bytes      INTEGER NOT NULL DEFAULT 0,
    requests   INTEGER NOT NULL DEFAULT 0,
    updated_at REAL    NOT NULL,
    PRIMARY KEY (username, day, kind)
);
CREATE INDEX IF NOT EXISTS usage_daily_day ON usage_daily (day);
"""

_UPSERT = """
INSERT INTO usage_daily (username, day, kind, bytes, requests, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(username, day, kind) DO UPDATE SET
    bytes      = bytes + excluded.bytes,
    requests   = requests + excluded.requests,
    updated_at = excluded.updated_at
"""


def utc_day(ts=None):
    """UTC day key (YYYY-MM-DD) for a unix timestamp (default: now)."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None \
        else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _empty_bucket() -> dict:
    return {"live": 0, "events": 0, "other": 0, "bytes": 0}


def _add_to(bucket, kind, nbytes):
    bucket[kind] = bucket.get(kind, 0) + nbytes
    bucket["bytes"] += nbytes


class UsageStore:
    """Daily per-user byte counters backed by a small SQLite DB.

    Single-process by design: the portal runs one uvicorn worker, so the
    in-memory accumulator needs no cross-process coordination.
    """

    def __init__(self, path, retention_days=180, flush_interval=30.0,
                 prune_interval=86400.0):
        self.path = path
        self.retention_days = max(1, int(retention_days))
        self.flush_interval = max(1.0, float(flush_interval))
        self.prune_interval = max(60.0, float(prune_interval))
        self._pending = {}          # (username, day, kind) -> [bytes, requests]
        self._log = None            # optional print callback

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
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---- write path ------------------------------------------------------
    def add(self, username, kind, nbytes, requests=1):
        """Accumulate `nbytes` for a user/kind. Never raises, never blocks."""
        try:
            n = int(nbytes)
        except (TypeError, ValueError):
            return
        if n <= 0 or not username or kind not in KINDS:
            return
        try:
            key = (str(username), utc_day(), kind)
            rec = self._pending.get(key)
            if rec is None:
                self._pending[key] = [n, max(0, int(requests))]
            else:
                rec[0] += n
                rec[1] += max(0, int(requests))
        except Exception:
            pass

    def flush(self):
        """Persist the accumulator. On failure the batch is kept and retried."""
        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        try:
            conn = self._connect()
        except sqlite3.Error:
            self._requeue(pending)
            return
        try:
            now = time.time()
            for (username, day, kind), (nbytes, requests) in pending.items():
                try:
                    conn.execute(_UPSERT,
                                 (username, day, kind, nbytes, requests, now))
                except sqlite3.Error:
                    pass
            conn.commit()
        except sqlite3.Error:
            pass
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _requeue(self, pending):
        for key, (nbytes, requests) in pending.items():
            rec = self._pending.setdefault(key, [0, 0])
            rec[0] += nbytes
            rec[1] += requests

    # ---- retention -------------------------------------------------------
    def prune(self, retention_days=None):
        """Delete day rows older than the retention window. Returns count."""
        days = self.retention_days if retention_days is None else int(retention_days)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d")
        try:
            conn = self._connect()
        except sqlite3.Error:
            return 0
        try:
            cur = conn.execute("DELETE FROM usage_daily WHERE day < ?", (cutoff,))
            conn.commit()
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        except sqlite3.Error:
            return 0
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ---- read path -------------------------------------------------------
    def _rows(self, where="", params=()):
        try:
            conn = self._connect()
        except sqlite3.Error:
            return []
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT username, day, kind, bytes, requests FROM usage_daily "
                + where, params)
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            return []
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _bucket_rows(self, rows):
        """Fold rows into today / 7d / 30d / total byte windows."""
        today = utc_day()
        d7 = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")
        d30 = (datetime.now(timezone.utc) - timedelta(days=29)).strftime("%Y-%m-%d")
        out: dict = {"today": _empty_bucket(), "d7": _empty_bucket(),
                     "d30": _empty_bucket(), "total": _empty_bucket()}
        for row in rows:
            n = int(row.get("bytes") or 0)
            kind = row.get("kind")
            if kind not in KINDS or n <= 0:
                continue
            day = row.get("day") or ""
            if day == today:
                _add_to(out["today"], kind, n)
            if day >= d7:
                _add_to(out["d7"], kind, n)
            if day >= d30:
                _add_to(out["d30"], kind, n)
            _add_to(out["total"], kind, n)
        return out

    def _pending_bucket(self, username):
        """Not-yet-flushed bytes, so the UI is live without waiting a flush."""
        today = utc_day()
        bucket = _empty_bucket()
        if not username:
            return bucket
        for (user, day, kind), (nbytes, _requests) in list(self._pending.items()):
            if user == str(username) and day == today:
                _add_to(bucket, kind, nbytes)
        return bucket

    def totals(self, username):
        """One user's today / 7d / 30d / total breakdown (incl. pending bytes)."""
        rows = self._rows("WHERE username = ? AND day >= ?",
                          (str(username), self._oldest_day()))
        out = self._bucket_rows(rows)
        pend = self._pending_bucket(username)
        if pend["bytes"]:
            for kind in KINDS:
                out["today"][kind] += pend[kind]
            out["today"]["bytes"] += pend["bytes"]
        out["username"] = str(username)
        out["retention_days"] = self.retention_days
        return out

    def _oldest_day(self):
        return (datetime.now(timezone.utc) - timedelta(days=self.retention_days)
                ).strftime("%Y-%m-%d")

    def all_users(self):
        """Every user's breakdown (admin view), biggest consumer first."""
        rows = self._rows("WHERE day >= ?", (self._oldest_day(),))
        by_user = {}
        for row in rows:
            by_user.setdefault(row.get("username") or "?", []).append(row)
        for key, pend in list(self._pending.items()):   # include unflushed
            user, day, kind = key
            rows2 = by_user.setdefault(user, [])
            if day == utc_day():
                rows2.append({"username": user, "day": day, "kind": kind,
                              "bytes": pend[0], "requests": pend[1]})
        users = []
        for name, user_rows in by_user.items():
            item = self._bucket_rows(user_rows)
            item["username"] = name
            users.append(item)
        users.sort(key=lambda u: u["total"]["bytes"], reverse=True)
        return {"users": users, "retention_days": self.retention_days,
                "today": utc_day()}

    # ---- background loop -------------------------------------------------
    async def run(self):
        """Flush every `flush_interval`s; prune once per `prune_interval`s."""
        last_prune = 0.0
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                self.flush()
                now = time.time()
                if now - last_prune >= self.prune_interval:
                    last_prune = now
                    self.prune()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
