"""store.py - persistence for the scenereader cross-camera episode narrator.

Single writer = the `scenereader` service (WAL SQLite). The portal reads the same
DB read-only through `portal/eventstore.py`; nothing else writes.

Two layers of data live here (see plans/event-scene-reader.md):

  L0  `events`        one row per Frigate capture event. Frames are NEVER copied:
                      `frame_path` points at Frigate's own media tree (read-only),
                      and metadata comes from Frigate's `event` table or REST API.
  L3  `episodes`      one row per linked person-track chain, plus the
                      `episode_events` join that carries the derived visits.
      `person_aliases` day-scoped anonymous -> real name (assisted labeling).

Design notes
------------
* Every function is explicit about its connection: callers own commit/rollback
  where it matters (`upsert_event`, `replace_episodes`), so a scan pass is one
  atomic unit rather than many tiny writes.
* "database is locked" is treated as WHAT IT IS - contention - never as a schema
  or data problem. The store has exactly one long-lived writer (the service, which
  commits once per pass) plus transient readers (the portal) and one-shot commands
  (`--drain`), so the answer to a lock error is to WAIT AND RETRY (see
  `retry_locked`), not to fail a pass. Diagnostics use `open_reader()` so a probe
  can never hold the write lock.
* The schema is created lazily and migrated idempotently: columns added after the
  first release are ALTERed in with defaults, and indexes that depend on a
  migrated column are created ONLY after the migration (the same bug class that
  once broke scenewatch - see plans/scene-description.md 12.3).
* `frigate_event_id` is UNIQUE: Frigate emits new/update/end for the same track,
  so an upsert collapses them to one row that is enriched, never duplicated.
"""
import json
import os
import sqlite3
import time

# ---------------------------------------------------------------------------
# concurrency - "database is locked" is CONTENTION, not corruption
# ---------------------------------------------------------------------------
def is_locked_error(exc):
    """True for SQLite's lock/busy errors (its message is all we get)."""
    text = str(exc).lower()
    return isinstance(exc, sqlite3.Error) and ("locked" in text or "busy" in text)


def retry_locked(call, op="sqlite", attempts=6, delay=0.5, log=None):
    """Run `call`, retrying SQLite lock/busy errors with a linear backoff.

    SQLite never says WHO holds the lock or for how long, so a lock error tells us
    only one thing: someone else is mid-write and will finish. Since every writer
    here commits per pass (never holds a transaction across a model run), waiting
    is always the correct answer - the retry succeeds as soon as they commit. Any
    NON-lock error is raised unchanged, so real bugs never hide behind this.
    """
    for attempt in range(max(1, int(attempts))):
        try:
            return call()
        except sqlite3.Error as exc:
            if not is_locked_error(exc) or attempt + 1 >= attempts:
                raise
            if log:
                log("store busy during {} (attempt {}/{}): {}".format(
                    op, attempt + 1, attempts, exc))
            time.sleep(delay * (attempt + 1))


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    frigate_event_id  TEXT    NOT NULL UNIQUE,
    camera            TEXT    NOT NULL,
    label             TEXT    NOT NULL DEFAULT '',
    sub_label         TEXT,
    zones             TEXT    NOT NULL DEFAULT '[]',
    place             TEXT,
    start_time        REAL    NOT NULL DEFAULT 0,
    end_time          REAL,
    duration          REAL,
    score             REAL,
    top_score         REAL,
    box               TEXT,
    frame_path        TEXT,
    frame_clean_path  TEXT,
    has_snapshot      INTEGER NOT NULL DEFAULT 0,
    false_positive    INTEGER NOT NULL DEFAULT 0,
    meta_json         TEXT,
    description_meta  TEXT,
    description_attr  TEXT,
    description_vlm   TEXT,
    vlm_model         TEXT,
    vlm_latency_ms    INTEGER,
    vlm_at            REAL,
    vlm_status        TEXT,
    episode_id        INTEGER,
    tier              TEXT    NOT NULL DEFAULT 'normal',
    importance        INTEGER NOT NULL DEFAULT 0,
    created_at        REAL    NOT NULL DEFAULT 0,
    updated_at        REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera, start_time);
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(start_time);
CREATE INDEX IF NOT EXISTS idx_events_vlm       ON events(vlm_at);
CREATE INDEX IF NOT EXISTS idx_events_episode   ON events(episode_id);

CREATE TABLE IF NOT EXISTS episodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    day              TEXT    NOT NULL,
    anon_name        TEXT    NOT NULL,
    label            TEXT    NOT NULL DEFAULT 'person',
    start_time       REAL    NOT NULL DEFAULT 0,
    end_time         REAL,
    narrative        TEXT,
    narrative_ar     TEXT,
    person_name      TEXT,
    link_confidence  REAL,
    created_at       REAL    NOT NULL DEFAULT 0,
    updated_at       REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_day_ts ON episodes(day, start_time);

CREATE TABLE IF NOT EXISTS episode_events (
    episode_id  INTEGER NOT NULL,
    event_id    INTEGER NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    place       TEXT,
    enter_time  REAL,
    leave_time  REAL,
    duration_s  REAL,
    PRIMARY KEY (episode_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_episode_events_event ON episode_events(event_id);

CREATE TABLE IF NOT EXISTS person_aliases (
    day          TEXT NOT NULL,
    anon_name    TEXT NOT NULL,
    person_name  TEXT NOT NULL,
    updated_at   REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, anon_name)
);
"""

# Columns added after the initial release: (name, DDL). Applied with a guarded
# ALTER so an existing DB keeps working. Indexes over these MUST come after.
_MIGRATIONS = (
    ("frame_clean_path", "ALTER TABLE events ADD COLUMN frame_clean_path TEXT"),
    ("description_attr", "ALTER TABLE events ADD COLUMN description_attr TEXT"),
    ("link_confidence", "ALTER TABLE episodes ADD COLUMN link_confidence REAL"),
    ("narrative_ar", "ALTER TABLE episodes ADD COLUMN narrative_ar TEXT"),
)

# Bumped whenever `_SCHEMA`/`_MIGRATIONS` change; stored in the DB's own
# `user_version`. WHY: a writer used to run `executescript(_SCHEMA)` on EVERY
# open - so a diagnostic or a one-shot command took the store's write lock just to
# ask a question. DDL now runs only when this number says it must.
_SCHEMA_VERSION = 1

# Event columns the rest of the code may set on an existing row (enrichment).
_ENRICH_COLS = (
    "label", "sub_label", "zones", "place", "start_time", "end_time", "duration",
    "score", "top_score", "box", "frame_path", "frame_clean_path",
    "has_snapshot", "false_positive", "meta_json", "description_meta",
    "description_attr", "description_vlm", "vlm_model", "vlm_latency_ms",
    "vlm_at", "vlm_status", "episode_id", "tier", "importance",
)


def _schema_outdated(conn):
    """True when the DB's `user_version` is older than this module's schema."""
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row is not None else 0
    except (sqlite3.Error, TypeError, ValueError, IndexError):
        return True
    return version < _SCHEMA_VERSION


def open_writer(path, timeout=30.0):
    """Open (creating if needed) the scenereader WAL DB for writing.

    `busy_timeout` + `retry_locked` make a pass WAIT for the other writer (the
    portal's reader, a cron helper, a concurrent one-shot) instead of dying with
    SQLite's terse "database is locked". The 30 s default is deliberately generous:
    the service commits once per pass, so the longest realistic wait is one commit.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=float(timeout))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout={}".format(int(float(timeout) * 1000)))
    # WAL is persistent, but the PRAGMA still has to take a lock to confirm it -
    # which raises "database is locked" while another connection is mid-write.
    retry_locked(lambda: conn.execute("PRAGMA journal_mode=WAL"),
                 op="journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if _schema_outdated(conn):
        retry_locked(lambda: conn.executescript(_SCHEMA), op="schema")
        # Guarded, idempotent migrations. "duplicate column name" means the column
        # is already there (ignore it) - but a LOCK must never be swallowed here:
        # that would stamp the new user_version onto a HALF-migrated DB.
        for _name, ddl in _MIGRATIONS:
            try:
                conn.execute(ddl)
            except sqlite3.Error as exc:
                if is_locked_error(exc):
                    raise
        conn.execute("PRAGMA user_version = {}".format(_SCHEMA_VERSION))
    conn.commit()
    return conn


def open_reader(path, timeout=10.0):
    """Open the store READ-ONLY - the mode for every diagnostic.

    No DDL, no `journal_mode` switch, no write lock, so `--check`/status probes can
    run against a LIVE store. `mode=ro` replays the WAL (newest rows visible);
    `immutable=1` is the fallback for a filesystem where the WAL index cannot be
    touched - it may lag by whatever is still in the WAL.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("store not found: {!r}".format(path))
    last = None
    for uri in ("file:" + path + "?mode=ro", "file:" + path + "?immutable=1"):
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=float(timeout))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout={}".format(int(float(timeout) * 1000)))
            conn.execute("PRAGMA query_only=ON")   # a diagnostic must never write
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            return conn
        except sqlite3.Error as exc:
            last = exc
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            continue
    raise last if last is not None else sqlite3.OperationalError(
        "cannot open the store READ-ONLY")


# ---------------------------------------------------------------------------
# events (L0)
# ---------------------------------------------------------------------------
def _row_value(payload, key, default=None):
    """Accept either a mapping or an object for the event payload."""
    if payload is None:
        return default
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def upsert_event(conn, payload):
    """Insert-or-update one event keyed by `frigate_event_id`.

    `payload` is a mapping (or object) with any of `_ENRICH_COLS` plus the key.
    Only the keys actually present are written, so a later scan can enrich a row
    without clobbering fields it does not know. Returns (row_id, inserted).
    """
    event_id = _row_value(payload, "frigate_event_id")
    if not event_id:
        raise ValueError("upsert_event needs frigate_event_id")
    now = time.time()

    row = conn.execute("SELECT id FROM events WHERE frigate_event_id = ?",
                       (event_id,)).fetchone()
    if row is None:
        cols = ["frigate_event_id", "camera", "created_at", "updated_at"]
        vals = [event_id, _row_value(payload, "camera", ""), now, now]
        for col in _ENRICH_COLS:
            val = _row_value(payload, col, None)
            if val is not None:
                cols.append(col)
                vals.append(val)
        placeholders = ",".join("?" * len(cols))
        cur = conn.execute(
            "INSERT INTO events (" + ",".join(cols) + ") VALUES (" + placeholders + ")",
            vals,
        )
        return cur.lastrowid, True

    row_id = row["id"]
    sets, vals = ["updated_at = ?"], [now]
    for col in _ENRICH_COLS:
        if isinstance(payload, dict):
            if col not in payload:
                continue
            val = payload[col]
        else:
            val = getattr(payload, col, None)
            if val is None:
                continue
        sets.append(col + " = ?")
        vals.append(val)
    if len(sets) > 1:
        vals.append(row_id)
        conn.execute("UPDATE events SET " + ", ".join(sets) + " WHERE id = ?", vals)
    return row_id, False


def events_by_camera_window(conn, camera, after, before):
    """Events for one camera with `after <= start_time <= before`, oldest first.

    Used by the enricher/describer to look at a camera's recent history (novelty,
    motion baseline) without a full-table scan.
    """
    return conn.execute(
        "SELECT * FROM events WHERE camera = ? AND start_time >= ? AND start_time <= ?"
        " ORDER BY start_time ASC",
        (camera, float(after), float(before)),
    ).fetchall()


def events_for_episode(conn, episode_id):
    """The ordered `episode_events` rows (visits) of one episode."""
    return conn.execute(
        "SELECT ee.*, e.camera, e.label, e.start_time, e.end_time, e.description_meta"
        " FROM episode_events ee JOIN events e ON e.id = ee.event_id"
        " WHERE ee.episode_id = ? ORDER BY ee.seq ASC",
        (episode_id,),
    ).fetchall()


def pending_events(conn, limit=50, only_person=False):
    """Events still needing a VLM caption, oldest first.

    Only rows that HAVE a frame and are not already captioned (or marked
    `missing`/`error`) are returned; `error` rows are retried by the caller
    deliberately via `mark_vlm(..., status='')`.
    """
    sql = ("SELECT * FROM events WHERE (vlm_at IS NULL OR vlm_at = 0)"
           " AND (vlm_status IS NULL OR vlm_status = '')"
           " AND frame_path IS NOT NULL AND frame_path != ''")
    params = []
    if only_person:
        sql += " AND label = 'person'"
    sql += " ORDER BY start_time ASC LIMIT ?"
    params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def mark_vlm(conn, event_id, status, text=None, model=None, latency_ms=None):
    """Record the outcome of a caption attempt. `status=''` re-queues the row."""
    now = time.time()
    conn.execute(
        "UPDATE events SET vlm_status = ?, description_vlm = COALESCE(?, description_vlm),"
        " vlm_model = COALESCE(?, vlm_model), vlm_latency_ms = COALESCE(?, vlm_latency_ms),"
        " vlm_at = ?, updated_at = ? WHERE id = ?",
        (status, text, model, latency_ms, now, now, event_id),
    )


def prune_events(conn, days):
    """Delete events older than `days` (0 = keep everything). Returns row count."""
    if not days or float(days) <= 0:
        return 0
    cutoff = time.time() - float(days) * 86400.0
    # Detach any episode links first so the join table never points at a ghost.
    conn.execute(
        "DELETE FROM episode_events WHERE event_id IN"
        " (SELECT id FROM events WHERE start_time < ?)", (cutoff,))
    cur = conn.execute("DELETE FROM events WHERE start_time < ?", (cutoff,))
    return cur.rowcount


# ---------------------------------------------------------------------------
# episodes (L2/L3/L4)
# ---------------------------------------------------------------------------
def clear_episodes(conn):
    """Drop all derived episode state so L2/L3 can be rebuilt from `events`.

    Descriptions (`description_*`) live on `events` and are untouched, so a
    rebuild after a places.conf or gap change costs no re-capture and no model run.
    """
    conn.execute("DELETE FROM episode_events")
    conn.execute("DELETE FROM episodes")
    conn.execute("UPDATE events SET episode_id = NULL")


def insert_episode(conn, day, anon_name, label, start_time, end_time,
                   narrative=None, person_name=None, link_confidence=None):
    """Insert one episode (without its visits) and return its new id."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO episodes (day, anon_name, label, start_time, end_time, narrative,"
        " person_name, link_confidence, created_at, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (day, anon_name, label, float(start_time), end_time, narrative,
         person_name, link_confidence, now, now),
    )
    return cur.lastrowid


def set_narratives(conn, day, anon_name, narrative, narrative_ar):
    """Write BOTH language variants of one episode's narrative.

    The Arabic text is built by the same deterministic composer as the English
    one - it is a translation of structured facts, never a model's guess.
    """
    conn.execute(
        "UPDATE episodes SET narrative = ?, narrative_ar = ?, updated_at = ?"
        " WHERE day = ? AND anon_name = ?",
        (narrative, narrative_ar, time.time(), day, anon_name),
    )


def add_visit(conn, episode_id, event_id, seq, place, enter_time, leave_time,
              duration_s, event_ids=None):
    """Attach one STAY to an episode at position `seq`.

    `event_id` is the REPRESENTATIVE capture of the stay (the thumbnail) and
    `event_ids` may list every capture the stay merges (Frigate re-fires detection,
    so several events can be one continuous stay). ONE row is written - the visit
    list the portal renders stays meaningful - while EVERY merged capture is linked
    to the episode, so `events.episode_id` remains complete.
    """
    conn.execute(
        "INSERT OR REPLACE INTO episode_events"
        " (episode_id, event_id, seq, place, enter_time, leave_time, duration_s)"
        " VALUES (?,?,?,?,?,?,?)",
        (episode_id, event_id, int(seq), place, enter_time, leave_time, duration_s),
    )
    for link_id in (event_ids or [event_id]):
        conn.execute("UPDATE events SET episode_id = ? WHERE id = ?",
                     (episode_id, link_id))


def days_with_events(conn):
    """Distinct UTC days that have events, oldest first (for episode rebuild)."""
    rows = conn.execute(
        "SELECT DISTINCT date(start_time, 'unixepoch') AS d FROM events"
        " WHERE start_time > 0 ORDER BY d ASC").fetchall()
    return [r["d"] for r in rows if r["d"]]


# ---------------------------------------------------------------------------
# person aliases (assisted labeling)
# ---------------------------------------------------------------------------
def get_aliases(conn, day):
    """{anon_name: person_name} for one UTC day."""
    rows = conn.execute(
        "SELECT anon_name, person_name FROM person_aliases WHERE day = ?",
        (day,)).fetchall()
    return {r["anon_name"]: r["person_name"] for r in rows}


def set_alias(conn, day, anon_name, person_name):
    """Record (or clear, when `person_name` is empty) one day-scoped name."""
    if not person_name:
        conn.execute("DELETE FROM person_aliases WHERE day = ? AND anon_name = ?",
                     (day, anon_name))
        return
    conn.execute(
        "INSERT INTO person_aliases (day, anon_name, person_name, updated_at)"
        " VALUES (?,?,?,?) ON CONFLICT(day, anon_name)"
        " DO UPDATE SET person_name = excluded.person_name,"
        " updated_at = excluded.updated_at",
        (day, anon_name, person_name, time.time()))


def merge_names_file(conn, path):
    """Import `names.json` written by the portal (assisted labeling).

    Expected shape: {"YYYY-MM-DD": {"Person A": "Ali", ...}}. Missing/!json files
    are ignored; unknown anon names are still recorded (the operator knows best).
    """
    if not path or not os.path.isfile(path):
        return 0
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    count = 0
    for day, names in data.items():
        if not isinstance(names, dict):
            continue
        for anon, name in names.items():
            set_alias(conn, str(day), str(anon), str(name or ""))
            count += 1
    return count


def json_col(value, default):
    """Parse a JSON text column, tolerating NULL/garbage."""
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
