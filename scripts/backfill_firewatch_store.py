#!/usr/bin/env python3
"""backfill_firewatch_store.py - register evidence JPEGs missing from firewatch.db.

WHY THIS EXISTS
---------------
firewatch writes TWO JPEGs per fire-marked frame and THEN one `frames` row. If
the evidence DB is unwritable - e.g. its `-wal`/`-shm` sidecars were created by
another user, so every INSERT fails with "attempt to write a readonly database"
(see plans/firewatch-store-readonly-recovery.md) - the JPEGs still land on disk
but the row is lost: the frame never appears in the portal's Fire tab while its
Telegram alert DID go out. This worker re-registers those orphan frames so the
missed alerts become visible again.

WHAT IT DERIVES
---------------
  * `camera`      - the camera DIRECTORY the JPEG lives in.
  * `captured_at` - from the filename firewatch itself wrote, which is UTC:
                    <YYYYMMDD>_<HHMMSS>_<mmm>_<cam>_conf<score>.jpg
  * `best_score`  - the `conf<score>` in the same filename (the effective score
                    firewatch recorded), NOT a fresh model score.
  * `detections`  - from a fresh model pass over the stored JPEG (same model,
                    SCORE_FLOOR and class filter as firewatch) so the portal can
                    draw the boxes. `--no-model` skips this (frames then register
                    with the score but no boxes).
  * `alerted`     - 0 unless matched to a logged alert (see --alert-times).

ALERT STATUS
------------
`frames.alerted` drives the portal's "alerts only" default. Pass `--alert-times`
the raw `docker logs firewatch` output (or plain `cam <epoch|ISO>` lines) and for
each logged "ALERT sent for <cam>" the NEWEST backfilled frame at/just before
that time is marked `alerted=1` - exactly the single frame firewatch would have
flagged when the confirm count was reached.

USAGE (inside the firewatch container, as the store owner uid 1000)
-------------------------------------------------------------------
    docker exec -i firewatch python /scripts/backfill_firewatch_store.py            # dry run
    docker exec -i firewatch python /scripts/backfill_firewatch_store.py --commit
    docker logs firewatch 2>&1 | grep "ALERT sent" > /tmp/alerts.txt
    docker exec -i firewatch python /scripts/backfill_firewatch_store.py \
        --alert-times /tmp/alerts.txt --commit

The write is APPEND-ONLY: no existing row is modified or deleted, apart from
flipping `alerted` on the frames matched to a logged alert. It never touches the
JPEGs themselves. Dry run is the default - nothing is written without --commit.
"""
import argparse
import calendar
import os
import re
import sqlite3
import sys
import time

# <YYYYMMDD>_<HHMMSS>_<mmm>_<camera>_conf<score>.jpg  (the base firewatch writes)
FRAME_RE = re.compile(
    r"^(?P<ymd>\d{8})_(?P<hms>\d{6})_(?P<ms>\d{3})_(?P<cam>.+)_conf(?P<conf>\d+\.\d+)$")
# a raw firewatch log line: "[2026-09-12 15:41:59] ALERT sent for cam01"
LOG_ALERT_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+ALERT sent for\s+(?P<cam>\S+)")

INSERT_FRAME = ("INSERT INTO frames (camera, captured_at, ts_utc, jpg_path, "
                "annotated_path, best_score, score_threshold, alerted) "
                "VALUES (?,?,?,?,?,?,?,?)")
INSERT_DET = ("INSERT INTO detections (frame_id, label, score, x1, y1, x2, y2) "
              "VALUES (?,?,?,?,?,?,?)")


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def raw_conf(path):
    """Tiny KEY=VALUE reader (mirrors the firewatch/cleanup parsers)."""
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


def as_bool(value, default=False):
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_base(base, cam):
    """(captured_at_epoch_utc, conf) from a firewatch evidence base name, or None.

    The name is written by firewatch from `time.gmtime()`, so it is UTC; the
    camera segment must match the directory the file sits in (a mismatch means
    the file is not what we think it is - skip it rather than mis-attribute).
    """
    m = FRAME_RE.match(base)
    if not m or m.group("cam") != cam:
        return None
    try:
        epoch = calendar.timegm(
            time.strptime(m.group("ymd") + m.group("hms"), "%Y%m%d%H%M%S"))
    except ValueError:
        return None
    return epoch + int(m.group("ms")) / 1000.0, float(m.group("conf"))


def parse_alert_times(path):
    """{cam: [epoch, ...]} from raw `docker logs` output or `cam <epoch|ISO>` lines.

    Recognized per line (blank/# lines ignored):
      * [2026-09-12 15:41:59] ALERT sent for cam01   (container logs are UTC)
      * cam01 2026-09-12T15:41:59Z                   (ISO, UTC)
      * cam01 1757684519.5                           (epoch seconds)
    """
    alerts = {}
    bad = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        log(f"ERROR: cannot read --alert-times {path}: {exc}")
        return alerts
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LOG_ALERT_RE.match(line)
        if m:
            cam, stamp = m.group("cam"), m.group("ts")
            try:
                epoch = calendar.timegm(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                bad += 1
                continue
        else:
            parts = line.split()
            if len(parts) != 2:
                bad += 1
                continue
            cam, stamp = parts
            try:
                epoch = float(stamp)
            except ValueError:
                try:
                    epoch = calendar.timegm(time.strptime(stamp.rstrip("Z"),
                                                          "%Y-%m-%dT%H:%M:%S"))
                except ValueError:
                    bad += 1
                    continue
        alerts.setdefault(cam, []).append(epoch)
    for cam in alerts:
        alerts[cam] = sorted(set(alerts[cam]))
    if bad:
        log(f"alert-times: ignored {bad} unrecognized line(s)")
    return alerts


def collect(store_dir, known):
    """[(cam, base, raw_path, ann_path, captured_at, conf), ...] oldest first.

    `known` is the set of `frames.jpg_path` values already in the DB; a JPEG
    whose ORIGINAL path is known is skipped (the annotated twin alone never
    identifies a frame).
    """
    out = []
    try:
        cams = sorted(e.name for e in os.scandir(store_dir) if e.is_dir())
    except OSError as exc:
        log(f"ERROR: cannot scan store dir {store_dir}: {exc}")
        return out
    for cam in cams:
        cam_dir = os.path.join(store_dir, cam)
        try:
            names = sorted(os.listdir(cam_dir))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".jpg") or name.endswith("_annotated.jpg"):
                continue
            parsed = parse_base(name[:-4], cam)
            if parsed is None:
                continue
            captured_at, conf = parsed
            raw_path = os.path.join(cam_dir, name)
            if raw_path in known:
                continue
            ann_path = os.path.join(cam_dir, name[:-4] + "_annotated.jpg")
            out.append((cam, name[:-4], raw_path,
                        ann_path if os.path.isfile(ann_path) else None,
                        captured_at, conf))
    out.sort(key=lambda r: (r[4], r[0]))
    return out


def load_model(model_dir, track_smoke):
    """firewatch's own FireModel (imported from the mounted daemon source)."""
    fw_dir = os.environ.get("FIREWATCH_DIR", "/firewatch")
    if fw_dir not in sys.path:
        sys.path.insert(0, fw_dir)
    try:
        import firewatch  # noqa: PLC0415 - in-container source, imported lazily
    except Exception as exc:  # noqa: BLE001 - report and fall back to --no-model
        log(f"ERROR: cannot import firewatch from {fw_dir}: {exc}")
        return None, None
    try:
        model = firewatch.FireModel(model_dir)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: cannot load the fire model from {model_dir}: {exc}")
        return None, None
    allowed = {"fire"} if not track_smoke else {"fire", "smoke"}
    return model, allowed


def detect(model, allowed, floor, raw_path):
    """Dets {label,score,box} for one stored JPEG ([] on any failure)."""
    try:
        from PIL import Image  # noqa: PLC0415 - present in the firewatch image
        with Image.open(raw_path) as img:
            return model.detect(img.convert("RGB"), score_thresh=floor,
                                allowed=allowed)
    except Exception as exc:  # noqa: BLE001 - one bad JPEG must not stop the run
        log(f"  ! {os.path.basename(raw_path)}: model pass failed: {exc}")
        return []


def main():
    ap = argparse.ArgumentParser(
        description="Register evidence JPEGs that have no firewatch.db frames row.")
    ap.add_argument("--commit", action="store_true",
                    help="actually write the rows (default: dry run)")
    ap.add_argument("--db", default=None, help="override the evidence DB path")
    ap.add_argument("--store-dir", default=None, help="override STORE_DIR")
    ap.add_argument("--model-dir", default=None, help="override MODEL_DIR")
    ap.add_argument("--floor", type=float, default=None,
                    help="detection floor for the model pass (SCORE_FLOOR)")
    ap.add_argument("--no-model", action="store_true",
                    help="skip the model pass (register frames with no boxes)")
    ap.add_argument("--alert-times", default=None, metavar="FILE",
                    help="raw `docker logs firewatch` output (or `cam <time>` "
                         "lines) used to set frames.alerted=1")
    ap.add_argument("--limit", type=int, default=0,
                    help="register at most N frames (0 = all)")
    ap.add_argument("--cams", default="",
                    help="comma-separated camera filter (default: all)")
    args = ap.parse_args()

    fw_conf = raw_conf(os.environ.get("FIREWATCH_CONF", "/config/firewatch.conf"))
    store_dir = (args.store_dir or os.environ.get("STORE_DIR")
                 or fw_conf.get("STORE_DIR") or "/media/firewatch")
    db_name = os.environ.get("STORE_DB") or fw_conf.get("STORE_DB") or "firewatch.db"
    db_path = args.db or os.path.join(store_dir, db_name)
    model_dir = (args.model_dir or os.environ.get("MODEL_DIR")
                 or fw_conf.get("MODEL_DIR") or "/models/fire")
    floor = (args.floor if args.floor is not None
             else float(env_or(fw_conf, "SCORE_FLOOR", "0.35")))
    threshold = float(env_or(fw_conf, "SCORE_THRESHOLD", "0.50"))
    track_smoke = as_bool(env_or(fw_conf, "TRACK_SMOKE", "false"))

    if not os.path.isfile(db_path):
        log(f"ERROR: no evidence DB at {db_path} - nothing to backfill")
        return 2

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        known = {r[0] for r in conn.execute("SELECT jpg_path FROM frames")}
    except sqlite3.Error as exc:
        log(f"ERROR: cannot open {db_path}: {exc}")
        log("ERROR: the store DB (and its -wal/-shm sidecars) must be writable by "
            f"uid {os.getuid()}. Fix the ownership on the host first "
            "(see plans/firewatch-store-readonly-recovery.md).")
        return 2

    todo = collect(store_dir, known)
    cams = {c.strip() for c in args.cams.split(",") if c.strip()}
    if cams:
        todo = [r for r in todo if r[0] in cams]
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]

    log(f"store {store_dir} | db {db_path} | {len(known)} registered frame(s) | "
        f"{len(todo)} orphan JPEG(s) to register")
    if not todo:
        conn.close()
        return 0

    alerts = parse_alert_times(args.alert_times) if args.alert_times else {}
    if args.alert_times and not alerts:
        log("WARNING: no usable alert times parsed - every frame stays alerted=0")

    for cam, base, raw_path, ann_path, captured_at, conf in todo:
        log(f"  + {cam} {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(captured_at))}Z "
            f"conf {conf:.2f} {base}.jpg"
            + ("" if ann_path else " (no annotated twin)"))
    if not args.commit:
        log("DRY RUN - nothing written. Re-run with --commit to register these frames.")
        conn.close()
        return 0

    model, allowed = (None, set())
    if not args.no_model:
        model, allowed = load_model(model_dir, track_smoke)
        if model is None:
            log("WARNING: falling back to --no-model (frames registered without boxes)")

    inserted = []
    for cam, base, raw_path, ann_path, captured_at, conf in todo:
        dets = detect(model, allowed, floor, raw_path) if model else []
        # `best_score` is the effective score firewatch itself recorded - it is
        # read straight from the filename, never re-derived from the fresh pass.
        try:
            cur = conn.execute(INSERT_FRAME, (
                cam, captured_at,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(captured_at)),
                raw_path, ann_path, round(float(conf), 4), threshold, 0))
            frame_id = cur.lastrowid
            if dets:
                conn.executemany(INSERT_DET, [
                    (frame_id, d["label"], round(d["score"], 4),
                     round(d["box"][0], 1), round(d["box"][1], 1),
                     round(d["box"][2], 1), round(d["box"][3], 1))
                    for d in sorted(dets, key=lambda x: x["score"], reverse=True)])
        except sqlite3.Error as exc:
            log(f"ERROR: insert failed for {base}: {exc}")
            conn.rollback()
            conn.close()
            return 1
        inserted.append((frame_id, cam, captured_at))
        log(f"  registered #{frame_id} {cam} {base} "
            f"({len(dets)} detection(s), alerted=0)")
    conn.commit()

    # --- alert status: the newest backfilled frame at/just before each alert ---
    marked = 0
    for cam, times in alerts.items():
        frames = sorted((f for f in inserted if f[1] == cam), key=lambda f: f[2])
        for when in times:
            hit = None
            for fid, _cam, ts in frames:
                if ts <= when + 1.0:
                    hit = fid
                else:
                    break
            if hit is None:
                log(f"alert-times: {cam} {when:.0f} has no backfilled frame "
                    "before it - skipped")
                continue
            conn.execute("UPDATE frames SET alerted=1 WHERE id=?", (hit,))
            marked += 1
            log(f"  alerted=1 -> #{hit} ({cam}, matched alert at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(when))}Z)")
    conn.commit()
    conn.close()

    log(f"done: {len(inserted)} frame(s) registered, {marked} marked alerted "
        f"for {len(alerts)} camera(s) with logged alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
