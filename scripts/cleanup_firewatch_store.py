#!/usr/bin/env python3
"""cleanup_firewatch_store.py - cap firewatch evidence storage (image cap + row expiry).

Runs INSIDE the firewatch container from the host crontab (scripts/crontab.sample):

    */30 * * * * /usr/bin/docker exec firewatch python /scripts/cleanup_firewatch_store.py

The WAL-mode SQLite evidence DB (frames/detections) is the source of truth: every
stored firewatch frame references its JPEGs via frames.jpg_path (the original) and
frames.annotated_path (the annotated twin). This script therefore never scans/glob-
deletes the store tree - it only removes the files that belong to a row it is
deleting - so it can never touch Frigate recordings/clips/exports that share the
host ./media mount.

It does two things:
  1. ROW EXPIRY: delete frames (and their detections) whose captured_at is older than
     ROW_EXPIRE_DAYS (default 90 = 3 months), then remove their JPEG files.
  2. IMAGE CAP: sum the on-disk bytes of the REMAINING stored JPEGs; while that
     exceeds MAX_IMAGES_GB (default 2 GiB), delete the OLDEST frames (rows + JPEGs)
     until under the cap.

Tunables (config/cleanup_firewatch.conf, KEY=VALUE; FW_* env vars or --flags override):
  MAX_IMAGES_GB    image-storage cap, GiB (float)  [2]
  ROW_EXPIRE_DAYS  rows older than N days expire    [90]
  LOG_FILE         append-only log file             [<store_dir>/cleanup.log]

Store dir/DB come from the container (STORE_DIR/STORE_DB env or config/firewatch.conf),
never from the tunable file, so cleanup can't drift from where firewatch writes.

--dry-run reports what would be deleted without changing anything.
Never raises; failures are logged and skipped (keeps cron quiet).
"""
import argparse
import os
import sqlite3
import sys
import time


def raw_conf(path):
    """Tiny KEY=VALUE reader (mirrors firewatch.conf parsing). Missing = {}."""
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                cfg[key.strip()] = value.strip()
    except (FileNotFoundError, OSError):
        pass
    return cfg


def env_or(raw, key, default):
    return os.environ.get(key) or raw.get(key) or default


def log(msg, log_file):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    try:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # logging is best-effort


def file_bytes(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0  # missing file: count 0 bytes, skip unlink later


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def main():
    ap = argparse.ArgumentParser(
        description="Cap firewatch evidence JPEG storage and expire old DB rows.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, change nothing")
    ap.add_argument("--db", default=None, help="override evidence DB path")
    ap.add_argument("--max-images-gb", type=float, default=None,
                    help="override image cap in GiB")
    ap.add_argument("--row-expire-days", type=int, default=None,
                    help="override row expiry in days")
    ap.add_argument("--log-file", default=None, help="override log file")
    args = ap.parse_args()

    # -- resolve store location from the container, not from the tunable file --
    fw_conf = raw_conf(os.environ.get("FIREWATCH_CONF", "/config/firewatch.conf"))
    store_dir = os.environ.get("STORE_DIR") or fw_conf.get("STORE_DIR") or "/media/firewatch"
    db_name = os.environ.get("STORE_DB") or fw_conf.get("STORE_DB") or "firewatch.db"
    db_path = args.db or os.path.join(store_dir, db_name)

    # -- tunables: config file < FW_* env < --flags --
    clean = raw_conf(os.environ.get("FW_CLEANUP_CONF", "/config/cleanup_firewatch.conf"))
    try:
        max_gb = (args.max_images_gb if args.max_images_gb is not None
                  else float(env_or(os.environ, "FW_MAX_IMAGES_GB",
                                    clean.get("MAX_IMAGES_GB", "2")) or 2))
    except ValueError:
        max_gb = 2.0
    try:
        expire_days = (args.row_expire_days if args.row_expire_days is not None
                       else int(env_or(os.environ, "FW_ROW_EXPIRE_DAYS",
                                       clean.get("ROW_EXPIRE_DAYS", "90")) or 90))
    except ValueError:
        expire_days = 90
    log_file = (args.log_file or env_or(os.environ, "FW_LOG_FILE",
                                        clean.get("LOG_FILE", ""))
                or os.path.join(store_dir, "cleanup.log"))

    cap_bytes = max_gb * (1024 ** 3)
    cutoff = time.time() - expire_days * 86400.0

    if not os.path.isfile(db_path):
        log(f"cleanup: no evidence DB at {db_path} - nothing to do", log_file)
        return 0

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")  # firewatch may be mid-write (WAL)
    conn.execute("PRAGMA foreign_keys=ON")
    rows = conn.execute(
        "SELECT id, captured_at, jpg_path, annotated_path FROM frames "
        "ORDER BY captured_at ASC, id ASC"
    ).fetchall()  # oldest first

    def refs(r):
        """All on-disk files a frame row references (raw + annotated twin)."""
        paths = [r[2]]
        if r[3]:
            paths.append(r[3])
        return paths

    expired = [r for r in rows if r[1] < cutoff]
    kept = [r for r in rows if r[1] >= cutoff]

    # image cap against the REMAINING (post-expiry) images, oldest first
    total = sum(file_bytes(p) for r in kept for p in refs(r))
    cap_del = []
    for r in kept:                      # already oldest-first
        if total <= cap_bytes:
            break
        total -= sum(file_bytes(p) for p in refs(r))
        cap_del.append(r)

    doomed = {r[0]: r for r in expired}
    for r in cap_del:
        doomed.setdefault(r[0], r)
    delete_ids = sorted(doomed)

    stored_bytes = sum(file_bytes(p) for r in rows for p in refs(r))
    log(f"cleanup: db={db_path} max_images={max_gb:g}GiB({fmt_bytes(cap_bytes)}) "
        f"expire={expire_days}d stored_frames={len(rows)} "
        f"image_bytes={fmt_bytes(stored_bytes)} "
        f"expired={len(expired)} over_cap={len(cap_del)} "
        f"total_to_delete={len(delete_ids)}", log_file)

    if args.dry_run:
        for i in delete_ids[:25]:
            log(f"  dry: would delete frame {i} {doomed[i][2]}", log_file)
        if len(delete_ids) > 25:
            log(f"  ... and {len(delete_ids) - 25} more", log_file)
        log("cleanup: DRY-RUN - nothing changed", log_file)
        conn.close()
        return 0

    # delete detections + frames explicitly (no reliance on FK cascade)
    if delete_ids:
        ph = ",".join("?" * len(delete_ids))
        conn.execute(f"DELETE FROM detections WHERE frame_id IN ({ph})", delete_ids)
        conn.execute(f"DELETE FROM frames WHERE id IN ({ph})", delete_ids)
        conn.commit()

    # remove BOTH JPEGs (raw + annotated twin) that belonged to the deleted rows
    unlinked = 0
    for i in delete_ids:
        for p in refs(doomed[i]):
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    unlinked += 1
            except OSError:
                pass

    # best-effort: rmdir now-empty per-camera dirs (only succeeds when empty, so
    # it can never remove a directory that still holds Frigate/non-evidence files)
    removed_dirs = 0
    parents = {os.path.dirname(p) for i in delete_ids for p in refs(doomed[i])}
    for d in sorted(parents):
        try:
            os.rmdir(d)
            removed_dirs += 1
        except OSError:
            pass

    conn.close()
    log(f"cleanup: deleted {len(delete_ids)} frame row(s), "
        f"{unlinked} image(s), removed {removed_dirs} empty dir(s)", log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
