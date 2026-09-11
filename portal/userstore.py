
"""Persistent portal user store (SQLite) - accounts managed at RUNTIME.

Users used to be `user <name> <pbkdf2_hash>` lines in config/portal.conf, parsed
ONCE at startup (see `pconf.parse_file`) and mounted read-only into the
container. ANY account change therefore needed an operator edit plus
`docker compose restart portal`.

This module moves users into a small SQLite DB the portal OWNS (the same pattern
as portal/usage.py), stored under the already-rw `./media` mount, so accounts can
be created / edited / deleted from the Account tab with NO container restart.

BOOTSTRAP / MIGRATION: on the very first run (empty table) the store is SEEDED
from the `user` lines still present in portal.conf, so an existing deployment
keeps working unchanged. After that the DB is the single source of truth;
portal.conf only keeps `SECRET_KEY` and the non-user tunables.

Columns: username | display_name | photo | is_admin | is_active | permissions
         | quota_bytes | password_hash | default_camera | created_at | updated_at

`is_active` is an account enable flag: an INACTIVE account cannot sign in and an
existing session for it is rejected on the next request (effective immediately,
no restart). The stored `password_hash` (pbkdf2$...) is exposed to ADMINS only,
in the users list, so it can be inspected/migrated.

`permissions` is a JSON array of the standard SPA tabs a user may open
(ALLOWED_PERMISSIONS). An EMPTY array means "all standard tabs" (the default);
admin-only tabs (debug / user management) are implied by `is_admin` and are
never stored here.

`quota_bytes` is the per-user egress quota (bytes), shown in the SPA as a
progress bar (used / quota). The `users` table DEFAULT is the initial 5 GiB;
a stored 0 (or missing value) means UNLIMITED, so an admin may run without a
cap. It is display-only for now (not enforced at the egress chokepoints).
"""
import json
import os
import re
import sqlite3
import time

# The app tabs a NON-ADMIN can be granted, stored as `tab_<name>` permission
# keys (e.g. `tab_live`). An ADMIN implicitly has EVERY tab - current AND future
# - so an admin's access never depends on these keys. A non-admin gets exactly
# the stored keys (an EMPTY list = no tabs). Keep AVAILABLE_TABS in sync with
# TABS in portal/static/app.js.
AVAILABLE_TABS = ("live", "events", "fire", "episodes", "adaptive", "scenes",
                  "notifications")
ALLOWED_PERMISSIONS = tuple("tab_" + t for t in AVAILABLE_TABS)

# Usernames are path-safe (used in /api/users/<name>) and modest in length.
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Photo is stored inline as a data URL. The SPA downsizes to <=256 px first,
# so this cap only guards against a hand-made oversized request.
MAX_PHOTO_BYTES = 512 * 1024
_DATA_URL_RE = re.compile(r"^data:image/(png|jpe?g|webp|gif|svg\+xml);base64,", re.I)

# The quota column DEFAULT is the initial per-user quota in bytes (5 GiB). The
# SPA edits it in decimal GB (1 GB = 1024**3 bytes = 5368709120); a stored 0 (or
# a missing value) means UNLIMITED, so an admin may run without a cap.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username       TEXT PRIMARY KEY COLLATE NOCASE,
    display_name   TEXT NOT NULL DEFAULT '',
    photo          TEXT NOT NULL DEFAULT '',
    is_admin       INTEGER NOT NULL DEFAULT 0,
    is_active      INTEGER NOT NULL DEFAULT 1,
    permissions    TEXT NOT NULL DEFAULT '[]',
    quota_bytes    INTEGER NOT NULL DEFAULT 5368709120,
    password_hash  TEXT NOT NULL,
    default_camera TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
"""

# Columns a caller may update through `update()`.
_UPDATABLE = ("display_name", "photo", "is_admin", "is_active", "permissions",
              "quota_bytes", "password_hash", "default_camera")


def normalize_username(name):
    """Trimmed username, validated. Raises ValueError on an invalid one."""
    name = ("" if name is None else str(name)).strip()
    if not USERNAME_RE.match(name):
        raise ValueError("invalid username")
    return name


def normalize_permissions(value):
    """Coerce permissions input into a canonical JSON list of known tabs.

    Accepts a list/tuple, a JSON array string, or a comma-separated string.
    Unknown tabs are dropped; order follows ALLOWED_PERMISSIONS.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [p.strip() for p in text.split(",")]
        value = parsed
    if isinstance(value, dict):
        value = [k for k, v in value.items() if v]
    if not isinstance(value, (list, tuple, set)):
        return []
    # Accept a bare tab name ('live') or the prefixed key ('tab_live').
    wanted = set()
    for item in value:
        key = str(item).strip().lower()
        if not key:
            continue
        if not key.startswith("tab_"):
            key = "tab_" + key
        wanted.add(key)
    return [p for p in ALLOWED_PERMISSIONS if p in wanted]


def valid_photo(value):
    """True when `value` is an acceptable photo (empty string or a data URL)."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    if value == "":
        return True
    if len(value) > MAX_PHOTO_BYTES:
        return False
    return bool(_DATA_URL_RE.match(value))


def _now():
    return time.time()


def _public(row):
    """Row -> the API shape (no password hash)."""
    return {
        "username": row["username"],
        "display_name": row["display_name"] or "",
        "photo": row["photo"] or "",
        "is_admin": bool(row["is_admin"]),
        "is_active": bool(row["is_active"]),
        "permissions": json.loads(row["permissions"] or "[]"),
        "quota_bytes": int(row["quota_bytes"] or 0),
        "default_camera": row["default_camera"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _full(row):
    """Row -> the internal shape (INCLUDES password_hash, for auth only)."""
    out = _public(row)
    out["password_hash"] = row["password_hash"]
    return out


class UserStore:
    """SQLite-backed user accounts, safe to read on every request.

    Single-process by design (the portal runs one uvicorn worker); each call
    opens a short-lived connection, so no cross-thread state is shared.
    """

    def __init__(self, path):
        self.path = path

    # ---- setup -----------------------------------------------------------
    def configure(self):
        """Create the DB dir/table. Raises only on a genuinely unusable path."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Migration: a DB created before `is_active` existed gains the column
            # (default 1 = enabled) instead of having to be recreated.
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
            if "is_active" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN is_active "
                             "INTEGER NOT NULL DEFAULT 1")
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---- bootstrap -------------------------------------------------------
    def count(self):
        """Number of stored users (0 before the first seed)."""
        try:
            conn = self._connect()
        except sqlite3.Error:
            return 0
        try:
            cur = conn.execute("SELECT COUNT(*) AS n FROM users")
            row = cur.fetchone()
            return int(row["n"]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def seed(self, seed_users):
        """Seed from parsed portal.conf `user` lines, ONLY when empty.

        `seed_users` items: {username, password_hash, default_camera}. The user
        literally named `admin` becomes the first administrator. Returns the
        number of rows inserted.
        """
        if self.count() > 0:
            return 0
        inserted = 0
        for entry in seed_users or []:
            name = (entry.get("username") or "").strip()
            phash = entry.get("password_hash") or ""
            if not name or not phash:
                continue
            try:
                self.create(
                    name, phash,
                    display_name=name,
                    is_admin=(name.lower() == "admin"),
                    permissions=[],
                    default_camera=entry.get("default_camera") or "",
                )
                inserted += 1
            except (ValueError, sqlite3.Error):
                continue
        return inserted

    # ---- read ------------------------------------------------------------
    def get(self, username):
        """Internal record (WITH password_hash) or None."""
        if not username:
            return None
        try:
            conn = self._connect()
        except sqlite3.Error:
            return None
        try:
            cur = conn.execute("SELECT * FROM users WHERE username = ?",
                               (str(username).strip(),))
            row = cur.fetchone()
            return _full(row) if row else None
        except sqlite3.Error:
            return None
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def get_public(self, username):
        """API record (no password hash) or None."""
        record = self.get(username)
        if record is None:
            return None
        record.pop("password_hash", None)
        return record

    def list(self, include_secrets=False):
        """Every user (API shape), ordered by username.

        `include_secrets=True` (ADMIN callers only) also carries the stored
        password_hash, so the admin Users table can show it.
        """
        try:
            conn = self._connect()
        except sqlite3.Error:
            return []
        try:
            cur = conn.execute("SELECT * FROM users ORDER BY username")
            out = []
            for row in cur.fetchall():
                item = _public(row)
                if include_secrets:
                    item["password_hash"] = row["password_hash"]
                out.append(item)
            return out
        except sqlite3.Error:
            return []
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def count_admins(self):
        """How many ACTIVE administrators exist (protects the last usable one)."""
        try:
            conn = self._connect()
        except sqlite3.Error:
            return 0
        try:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM users WHERE is_admin = 1 "
                "AND is_active = 1")
            row = cur.fetchone()
            return int(row["n"]) if row else 0
        except sqlite3.Error:
            return 0
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ---- write -----------------------------------------------------------
    def create(self, username, password_hash, display_name="", is_admin=False,
               is_active=True, permissions=None, quota_bytes=None,
               default_camera=""):
        """Insert a new user. Raises ValueError on bad input/SQLite on conflict.

        `quota_bytes` omitted (None) leaves the column at its table DEFAULT
        (5 GiB); an explicit value is clamped to >= 0 (0 = unlimited).
        """
        name = normalize_username(username)
        if not password_hash:
            raise ValueError("password hash required")
        perms = json.dumps(normalize_permissions(permissions))
        now = _now()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO users (username, display_name, photo, is_admin, "
                "is_active, permissions, password_hash, default_camera, "
                "created_at, updated_at) "
                "VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?)",
                (name, str(display_name or "")[:64], 1 if is_admin else 0,
                 1 if is_active else 0, perms, password_hash,
                 str(default_camera or "")[:64], now, now))
            if quota_bytes is not None:
                conn.execute(
                    "UPDATE users SET quota_bytes = ? WHERE username = ?",
                    (max(0, int(quota_bytes or 0)), name))
            conn.commit()
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        return self.get_public(name)

    def update(self, username, **fields):
        """Update the given columns of an existing user. Returns the record."""
        sets = []
        vals = []
        for key, value in fields.items():
            if key not in _UPDATABLE:
                continue
            if key == "permissions":
                value = json.dumps(normalize_permissions(value))
            elif key in ("is_admin", "is_active"):
                value = 1 if value else 0
            elif key == "quota_bytes":
                value = max(0, int(value or 0))
            elif key in ("display_name", "default_camera"):
                value = str(value or "")[:64]
            elif key == "photo":
                value = str(value or "")
            sets.append(key + " = ?")
            vals.append(value)
        if not sets:
            return self.get_public(username)
        sets.append("updated_at = ?")
        vals.append(_now())
        vals.append(str(username).strip())
        conn = self._connect()
        try:
            conn.execute("UPDATE users SET " + ", ".join(sets)
                         + " WHERE username = ?", vals)
            conn.commit()
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        return self.get_public(username)

    def set_password(self, username, password_hash):
        """Replace a user's password hash."""
        return self.update(username, password_hash=password_hash)

    def delete(self, username):
        """Delete a user. Returns True when a row was removed."""
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM users WHERE username = ?",
                               (str(username).strip(),))
            conn.commit()
            return bool(cur.rowcount)
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
