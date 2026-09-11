#!/usr/bin/env python3
"""export_camera_clips.py - copy Frigate recording VIDEO segments for chosen cameras
AND carry the event/evidence context for each clip into local sidecar files.

General-purpose sibling of `export_firewatch_clips.py`. That tool is driven by the
firewatch alert log (fire moments only); THIS one selects raw Frigate recording
segments per CAMERA (optionally filtered by time / motion / object count) so the
video can be pulled to a dev machine for:

  * firewatch fire/smoke model testing - extract frames, re-score, hunt false
    positives (e.g. the cam01 dogs scored as FIRE), and
  * scenereader fine-tuning - the same event-only footage Frigate recorded.

It also writes LOCAL context so a clip can be interpreted offline:

  * Frigate events (label, score, zones, false_positive) overlapping each clip.
  * firewatch evidence frames (best_score, alerted, fire/smoke labels) inside
    each clip - this is the true/false alert information (alerted=1 -> Telegram).

Why the `recordings` table
--------------------------
Recording is EVENT-ONLY (detect substream, 640x360, ~10 s segments). Frigate
stores each segment at `/media/frigate/recordings/<date>/<hour>/<cam>/<ss.ff>.mp4`
and indexes it in the `recordings` table (camera, path, start_time, end_time,
duration, motion, objects). Copying those files is the cheapest, lossless way to
get the video; the event clips shown in the UI are synthesised from them.

Run ON the Frigate host (as a user that can read Frigate's media tree). Read-only
against the DBs; the ONLY writes are under --out.

Usage
-----
  python3 export_camera_clips.py --list                       # size/range per camera
  python3 export_camera_clips.py --cams cam01,cam02,cam03,cam08 --dry-run
  python3 export_camera_clips.py --cams cam01,cam02,cam03,cam08

Tunables (env or flag; host defaults in []):

  FRIGATE_DB   / --db         frigate sqlite db [/home/dr/frigate/config/frigate.db]
  MEDIA_ROOT   / --media-root host media dir, prefix for /media/frigate/... [/home/dr/frigate/media]
  CAM_CLIP_OUT / --out        output dir [/home/dr/frigate/cam-clips]
  --firewatch-db              firewatch evidence sqlite [/home/dr/frigate/media/firewatch.db]
  --cams cam01,cam02          cameras to export (default: every camera in the DB)
  --since / --until           only segments overlapping the window (ISO 8601 or epoch)
  --min-motion N              keep segments whose motion >= N (recordings.motion)
  --min-objects N             keep segments whose object count >= N
  --limit N                   at most N segments per camera (newest kept)
  --with-firewatch-images     also copy the firewatch evidence JPEGs (small)
  --force                     re-copy files that already exist in --out

Output layout
-------------
  <out>/<cam>/<date>_<hour>_<ss.ff>.mp4   the video segments (collision-proof names)
  <out>/manifest.csv      one row per clip: time/camera/size + linked event ids/labels
  <out>/events.csv        one row per Frigate event + the clip file(s) covering it
  <out>/firewatch_frames.csv  one row per firewatch evidence frame (true/false alerts)
  <out>/clips.json        the same data nested per clip (easy local consumption)
  <out>/README.txt        column glossary
"""
import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sqlite3
import struct
import sys

UTC = datetime.timezone.utc

# /media/frigate/recordings/<date>/<hour>/<cam>/<name>.mp4
_SEG_RE = re.compile(
    r"/recordings/(?P<date>\d{4}-\d{2}-\d{2})/(?P<hour>\d{2})/"
    r"(?P<cam>[^/]+)/(?P<name>[^/]+)$")


def parse_when(value):
    """None | epoch string | ISO-8601 string -> epoch seconds (UTC)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        raise SystemExit("ERROR: --since/--until must be epoch seconds or "
                         "ISO-8601 (got %r)" % value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def fmt_utc(epoch):
    return datetime.datetime.fromtimestamp(float(epoch), UTC).strftime(
        "%Y-%m-%d %H:%M:%S")


def human_mb(nbytes):
    return "%.1f MB" % (nbytes / 1048576.0)


def _num(value):
    """Coerce a stored numeric to float (legacy firewatch rows may be float32 BLOBs)."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) >= 4:
            return struct.unpack("<f", bytes(value[:4]))[0]
        return 0.0
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def host_path(media_root, container_path):
    """Map a Frigate container path (/media/frigate/...) to the host path."""
    if container_path.startswith("/media/frigate/"):
        return os.path.join(media_root, container_path[len("/media/frigate/"):])
    return container_path  # already absolute/host


def fw_host_path(media_root, stored_path):
    """Map a firewatch stored path (/media/firewatch/...) to the host path.

    firewatch's STORE_DIR is /media/firewatch in its container, which is the SAME
    host ./media tree, so /media/firewatch/<rest> -> <media_root>/<rest>.
    """
    if stored_path.startswith("/media/firewatch/"):
        return os.path.join(media_root, stored_path[len("/media/firewatch/"):])
    return stored_path


def dest_name(container_path, start):
    """Collision-proof output filename: <date>_<hour>_<orig>, else a timestamp.

    The recording basename (`ss.ff.mp4`) repeats across hours, so the date+hour
    from the container path are PREPENDED to keep the per-camera tree unique.
    """
    match = _SEG_RE.search(container_path or "")
    if match:
        return "%s_%s_%s" % (match.group("date"), match.group("hour"),
                             match.group("name"))
    stamp = datetime.datetime.fromtimestamp(float(start or 0), UTC).strftime(
        "%Y%m%d_%H%M%S")
    return "%s_%s" % (stamp, os.path.basename(container_path or "segment.mp4"))


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def camera_list(cur, wanted):
    """The cameras to export: --cams split on commas, else every DB camera."""
    if wanted:
        return [c.strip() for c in wanted.split(",") if c.strip()]
    return [row[0] for row in cur.execute(
        "SELECT DISTINCT camera FROM recordings ORDER BY camera")]


def load_segments(cur, cams, since, until, min_motion, min_objects, limit):
    """Recording segments per camera, oldest-first, with optional filters."""
    out = []
    for cam in cams:
        sql = ("SELECT path, start_time, end_time, duration, motion, objects "
               "FROM recordings WHERE camera = ?")
        params = [cam]
        if since is not None:
            sql += " AND end_time >= ?"
            params.append(since)
        if until is not None:
            sql += " AND start_time <= ?"
            params.append(until)
        if min_motion is not None:
            sql += " AND motion >= ?"
            params.append(min_motion)
        if min_objects is not None:
            sql += " AND objects >= ?"
            params.append(min_objects)
        if limit:
            # keep the NEWEST `limit` segments, then present them oldest-first
            rows = list(cur.execute(
                sql + " ORDER BY start_time DESC LIMIT ?", params + [int(limit)]))
            rows.reverse()
        else:
            rows = list(cur.execute(sql + " ORDER BY start_time", params))
        for row in rows:
            out.append({
                "camera": cam, "path": row[0], "start": row[1], "end": row[2],
                "duration": row[3], "motion": row[4], "objects": row[5],
            })
    return out


def _event_score(raw):
    """score / top_score live in the `data` JSON (the columns are NULL)."""
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data.get("score"), data.get("top_score")


def load_events(cur, cams, since, until):
    """Frigate events for the cameras/window - context for what the clips show."""
    out = []
    for cam in cams:
        sql = ("SELECT id, camera, label, sub_label, start_time, end_time, "
               "zones, false_positive, has_clip, data FROM event "
               "WHERE camera = ?")
        params = [cam]
        if since is not None:
            sql += " AND end_time >= ?"
            params.append(since)
        if until is not None:
            sql += " AND start_time <= ?"
            params.append(until)
        sql += " ORDER BY start_time"
        for row in cur.execute(sql, params):
            score, top_score = _event_score(row[9])
            out.append({
                "id": row[0], "camera": row[1], "label": row[2] or "",
                "sub_label": row[3] or "", "start": row[4], "end": row[5],
                "zones": row[6] or "[]", "false_positive": int(row[7] or 0),
                "has_clip": row[8], "score": score, "top_score": top_score,
            })
    return out


def load_firewatch(fw_db, cams, since, until):
    """firewatch evidence frames (+fire/smoke labels) for the cameras/window.

    firewatch stores EVERY frame its fire/smoke model scored (not only alerts):
    `alerted`=1 is a frame that produced a Telegram alert, 0 is stored evidence
    only. `detections` carries the per-box label/score. This is the true/false
    alert record for the clips.
    """
    if not fw_db or not os.path.isfile(fw_db):
        return []
    con = sqlite3.connect("file:%s?mode=ro" % fw_db, uri=True)
    con.row_factory = sqlite3.Row
    frames = []
    try:
        for cam in cams:
            sql = ("SELECT id, camera, captured_at, ts_utc, jpg_path, "
                   "best_score, alerted, annotated_path FROM frames "
                   "WHERE camera = ?")
            params = [cam]
            if since is not None:
                sql += " AND captured_at >= ?"
                params.append(since)
            if until is not None:
                sql += " AND captured_at <= ?"
                params.append(until)
            sql += " ORDER BY captured_at"
            rows = list(con.execute(sql, params))
            labels = {}
            ids = [r["id"] for r in rows]
            if ids:
                marks = ",".join("?" * len(ids))
                try:
                    det_rows = con.execute(
                        "SELECT frame_id, label FROM detections "
                        "WHERE frame_id IN (" + marks + ")", ids)
                    for det in det_rows:
                        labels.setdefault(det["frame_id"], set()).add(det["label"])
                except sqlite3.Error:
                    pass  # older store may lack `detections`; labels stay empty
            for row in rows:
                frames.append({
                    "id": row["id"], "camera": row["camera"],
                    "captured_at": _num(row["captured_at"]),
                    "ts_utc": row["ts_utc"],
                    "best_score": _num(row["best_score"]),
                    "alerted": int(row["alerted"] or 0),
                    "jpg_path": row["jpg_path"],
                    "annotated_path": row["annotated_path"],
                    "labels": sorted(labels.get(row["id"], [])),
                })
    except sqlite3.Error as exc:
        print("WARNING: firewatch DB unreadable (%s) - evidence sidecars empty" % exc)
    finally:
        con.close()
    return frames


# ---------------------------------------------------------------------------
# linking: clip <-> Frigate event <-> firewatch evidence
# ---------------------------------------------------------------------------
def link_segment(seg, events_by_cam, fw_by_cam):
    """Frigate events + firewatch frames overlapping a segment's window."""
    cam = seg["camera"]
    start, end = seg["start"], seg["end"]
    ev = [e for e in events_by_cam.get(cam, [])
          if e["start"] <= end and (e["end"] or e["start"]) >= start]
    fw = [f for f in fw_by_cam.get(cam, [])
          if start <= f["captured_at"] <= end]
    return ev, fw


def group_by_camera(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item["camera"], []).append(item)
    return grouped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get(
        "FRIGATE_DB", "/home/dr/frigate/config/frigate.db"))
    ap.add_argument("--firewatch-db", dest="firewatch_db", default=os.environ.get(
        "FIREWATCH_DB", "/home/dr/frigate/media/firewatch.db"))
    ap.add_argument("--media-root", default=os.environ.get(
        "MEDIA_ROOT", "/home/dr/frigate/media"))
    ap.add_argument("--out", default=os.environ.get(
        "CAM_CLIP_OUT", "/home/dr/frigate/cam-clips"))
    ap.add_argument("--cams", help="comma-separated camera names (default: all)")
    ap.add_argument("--since", help="segments ending on/after this (ISO 8601 or epoch)")
    ap.add_argument("--until", help="segments starting on/before this (ISO 8601 or epoch)")
    ap.add_argument("--min-motion", type=float, dest="min_motion",
                    help="keep segments with recordings.motion >= N")
    ap.add_argument("--min-objects", type=int, dest="min_objects",
                    help="keep segments with object count >= N")
    ap.add_argument("--limit", type=int,
                    help="max segments per camera (newest kept)")
    ap.add_argument("--with-firewatch-images", action="store_true",
                    help="also copy the (small) firewatch evidence JPEGs")
    ap.add_argument("--force", action="store_true",
                    help="re-copy files that already exist in --out")
    ap.add_argument("--dry-run", action="store_true", help="plan only, copy nothing")
    ap.add_argument("--list", action="store_true",
                    help="size/range summary only, copy nothing")
    args = ap.parse_args()

    since = parse_when(args.since)
    until = parse_when(args.until)

    if not os.path.isfile(args.db):
        raise SystemExit("ERROR: Frigate DB not found: %s" % args.db)

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    cur = con.cursor()
    cams = camera_list(cur, args.cams)
    if not cams:
        raise SystemExit("ERROR: no cameras selected/found")
    segments = load_segments(cur, cams, since, until, args.min_motion,
                             args.min_objects, args.limit)
    events = load_events(cur, cams, since, until)
    con.close()
    fw_frames = load_firewatch(args.firewatch_db, cams, since, until)

    # resolve the host file for each segment and measure it
    for seg in segments:
        seg["src"] = host_path(args.media_root, seg["path"])
        seg["exists"] = os.path.isfile(seg["src"])
        seg["size"] = os.path.getsize(seg["src"]) if seg["exists"] else 0
        seg["dest_rel"] = os.path.join(seg["camera"],
                                       dest_name(seg["path"], seg["start"]))

    present = [s for s in segments if s["exists"]]
    missing = len(segments) - len(present)

    # clip <-> event/evidence linkage (only clips we actually have)
    events_by_cam = group_by_camera(events)
    fw_by_cam = group_by_camera(fw_frames)
    for seg in present:
        ev, fw = link_segment(seg, events_by_cam, fw_by_cam)
        seg["events"] = ev
        seg["firewatch"] = fw

    # reverse map: the clip file(s) covering each event / evidence frame
    for ev in events:
        ev["clips"] = [s["dest_rel"] for s in present
                       if s["camera"] == ev["camera"]
                       and ev["start"] <= s["end"]
                       and (ev["end"] or ev["start"]) >= s["start"]]
    for frame in fw_frames:
        frame["clips"] = [s["dest_rel"] for s in present
                          if s["camera"] == frame["camera"]
                          and s["start"] <= frame["captured_at"] <= s["end"]]

    print("cameras: %s" % ", ".join(cams))
    if since or until:
        print("window : %s -> %s UTC" % (
            fmt_utc(since) if since else "-", fmt_utc(until) if until else "-"))
    print("segments in DB: %d (%d on disk, %d cleaned up)  "
          "frigate events: %d  firewatch frames: %d" % (
              len(segments), len(present), missing, len(events), len(fw_frames)))
    print()
    for cam in cams:
        rows = [s for s in present if s["camera"] == cam]
        total = sum(s["size"] for s in rows)
        span = ("%s -> %s" % (fmt_utc(rows[0]["start"]), fmt_utc(rows[-1]["end"]))
                if rows else "-")
        print("  %-8s %5d seg  %10s  %s" % (cam, len(rows), human_mb(total), span))

    if args.list:
        return 0

    print("\ntotal on disk: %d file(s), %s  ->  %s" % (
        len(present), human_mb(sum(s["size"] for s in present)), args.out))

    if args.dry_run:
        linked = sum(1 for s in present if s["events"] or s["firewatch"])
        print("dry-run: nothing copied. %d/%d clip(s) have linked event/evidence."
              % (linked, len(present)))
        return 0

    # ---- copy the video segments ----
    copied = 0
    for seg in present:
        dst = os.path.join(args.out, seg["dest_rel"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not args.force:
            continue
        shutil.copy2(seg["src"], dst)
        copied += 1

    # ---- copy the firewatch evidence images (optional) ----
    fw_images = 0
    if args.with_firewatch_images:
        for frame in fw_frames:
            for key in ("jpg_path", "annotated_path"):
                stored = frame.get(key)
                if not stored:
                    continue
                src = fw_host_path(args.media_root, stored)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(args.out, "firewatch", frame["camera"],
                                   os.path.basename(src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst) or args.force:
                    shutil.copy2(src, dst)
                    fw_images += 1

    # ---- manifest.csv: one row per clip (with linkage) ----
    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "camera", "start_utc", "end_utc", "duration_s", "motion", "objects",
            "size_bytes", "dest", "event_ids", "event_labels",
            "event_false_positive", "firewatch_ids", "firewatch_labels",
            "firewatch_alerted"])
        for seg in present:
            writer.writerow([
                seg["camera"], fmt_utc(seg["start"]), fmt_utc(seg["end"]),
                ("%.3f" % seg["duration"]) if seg["duration"] is not None else "",
                seg["motion"], seg["objects"], seg["size"], seg["dest_rel"],
                ";".join(e["id"] for e in seg["events"]),
                ";".join(sorted({e["label"] for e in seg["events"] if e["label"]})),
                int(any(e["false_positive"] for e in seg["events"])),
                ";".join(str(f["id"]) for f in seg["firewatch"]),
                ";".join(sorted({lab for f in seg["firewatch"]
                                 for lab in f["labels"]})),
                int(any(f["alerted"] for f in seg["firewatch"]))])

    # ---- events.csv: one row per Frigate event + the clips covering it ----
    events_csv = os.path.join(args.out, "events.csv")
    with open(events_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "event_id", "camera", "label", "sub_label", "start_utc", "end_utc",
            "duration_s", "score", "top_score", "false_positive", "has_clip",
            "zones", "clips"])
        for ev in events:
            duration = (ev["end"] - ev["start"]) if ev["end"] else ""
            writer.writerow([
                ev["id"], ev["camera"], ev["label"], ev["sub_label"],
                fmt_utc(ev["start"]), fmt_utc(ev["end"]) if ev["end"] else "",
                ("%.2f" % duration) if duration != "" else "",
                ev["score"], ev["top_score"], ev["false_positive"],
                ev["has_clip"], ev["zones"], ";".join(ev["clips"])])

    # ---- firewatch_frames.csv: the true/false alert evidence per clip ----
    fw_csv = os.path.join(args.out, "firewatch_frames.csv")
    with open(fw_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame_id", "camera", "ts_utc", "captured_utc", "best_score",
            "alerted", "labels", "jpg_path", "clips"])
        for frame in fw_frames:
            writer.writerow([
                frame["id"], frame["camera"], frame["ts_utc"],
                fmt_utc(frame["captured_at"]), frame["best_score"],
                frame["alerted"], ";".join(frame["labels"]),
                frame["jpg_path"], ";".join(frame["clips"])])

    # ---- clips.json: the same data nested per clip ----
    clips_json = os.path.join(args.out, "clips.json")
    with open(clips_json, "w", encoding="utf-8") as fh:
        json.dump([
            {
                "camera": seg["camera"],
                "start_utc": fmt_utc(seg["start"]),
                "end_utc": fmt_utc(seg["end"]),
                "duration_s": seg["duration"],
                "motion": seg["motion"],
                "objects": seg["objects"],
                "size_bytes": seg["size"],
                "clip": seg["dest_rel"],
                "frigate_events": [
                    {"id": e["id"], "label": e["label"],
                     "sub_label": e["sub_label"], "score": e["score"],
                     "top_score": e["top_score"],
                     "false_positive": e["false_positive"],
                     "start_utc": fmt_utc(e["start"]),
                     "end_utc": fmt_utc(e["end"]) if e["end"] else None,
                     "zones": e["zones"]}
                    for e in seg["events"]],
                "firewatch_frames": [
                    {"id": f["id"], "ts_utc": f["ts_utc"],
                     "captured_utc": fmt_utc(f["captured_at"]),
                     "best_score": f["best_score"], "alerted": f["alerted"],
                     "labels": f["labels"], "jpg_path": f["jpg_path"]}
                    for f in seg["firewatch"]],
            }
            for seg in present], fh, indent=1)

    # ---- README.txt: column glossary for offline use ----
    with open(os.path.join(args.out, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "Camera clip export\n"
            "==================\n\n"
            "Video: <cam>/<date>_<hour>_<ss.ff>.mp4 - Frigate recording segments\n"
            "(event-only, 640x360 detect substream, ~10 s each).\n\n"
            "manifest.csv - one row per clip:\n"
            "  start_utc/end_utc    segment window (UTC)\n"
            "  motion/objects       Frigate recordings.motion / object count\n"
            "  event_ids/labels     Frigate events overlapping the clip\n"
            "  event_false_positive 1 if any overlapping event is user-flagged FP\n"
            "  firewatch_ids        firewatch evidence frames inside the clip\n"
            "  firewatch_labels     fire/smoke labels on those frames\n"
            "  firewatch_alerted    1 if a frame in the clip sent a Telegram alert\n\n"
            "events.csv - one row per Frigate event (label/score/zones/\n"
            "false_positive) + the 'clips' column listing covering clip files.\n\n"
            "firewatch_frames.csv - one row per firewatch evidence frame\n"
            "(best_score, alerted, fire/smoke labels) + covering clips.\n"
            "  alerted=1 -> the frame produced a Telegram fire alert (the\n"
            "  'true/false' signal; a dog scored as fire shows up here).\n\n"
            "clips.json - the same data nested per clip for scripting.\n")

    print("\ncopied %d segment(s) into %s" % (copied, args.out))
    if args.with_firewatch_images:
        print("copied %d firewatch evidence image(s)" % fw_images)
    print("manifest: %s" % manifest)
    print("events  : %s" % events_csv)
    print("firewatch: %s" % fw_csv)
    print("json    : %s" % clips_json)
    if missing:
        print("skipped %d DB row(s) whose file was already cleaned up" % missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
