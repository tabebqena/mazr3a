#!/usr/bin/env python3
"""separate_fire_event_images.py - explore fire model events above a score and
SEPARATE the images related to each one.

The camera-clip export ([`dev_scripts/export_camera_clips.py`](export_camera_clips.py))
writes, next to the recorded video segments, a sidecar
``firewatch_frames.csv`` - one row per evidence frame the fire model scored
(``best_score``, ``alerted``, fire/smoke ``labels``, the stored ``jpg_path`` and
the covering local ``clips``). This helper:

  1. EXPLORES it: keeps the events whose ``labels`` contains ``fire`` (or the
     configured label) and whose ``best_score`` is ABOVE the threshold (default
     ``0.5``), then prints counts per camera, score buckets and alert totals.
  2. SEPARATES the related images into a per-camera tree, resolving each frame's
     image in priority order:
       a. an already-downloaded evidence JPEG (``--images-dir``), else
       b. a frame extracted from the covering LOCAL clip at the exact
          ``captured_utc - clip_start`` offset (ffmpeg, cv2 fallback), else
       c. recorded as MISSING (no local video - the CLI prints why).

Output (default ``camera-clips/fire_events_over_0.5/``):
    fire_events_over_0.5.csv   the filtered fire events + image provenance
    images/<cam>/<rank>_score<score>_<ts>_f<id>.jpg
    missing_no_local_video.csv events with no local image (with a reason)
    SUMMARY.txt                the human-readable exploration report

Usage:
    .venv/bin/python dev_scripts/separate_fire_event_images.py
    .venv/bin/python dev_scripts/separate_fire_event_images.py --min-score 0.7
    .venv/bin/python dev_scripts/separate_fire_event_images.py --label any
    .venv/bin/python dev_scripts/separate_fire_event_images.py \
        --images-dir camera-clips/firewatch

Read-only against the input; the ONLY writes are under --out.
"""
import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys

UTC = datetime.timezone.utc
DEFAULT_CLIPS_DIR = "camera-clips"
DEFAULT_MIN_SCORE = 0.5


def parse_ts(value):
    """'2026-09-08 13:44:19' (or ISO-8601 with Z/offset) -> aware UTC datetime."""
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def human_ts(dt):
    """Aware datetime -> 'YYYYMMDD_HHMMSS' (filesystem-safe)."""
    return dt.strftime("%Y%m%d_%H%M%S")


def load_clip_windows(manifest_path):
    """manifest.csv dest -> (start_dt, end_dt). Empty dict when absent."""
    windows = {}
    if not os.path.isfile(manifest_path):
        return windows
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            start = parse_ts(row.get("start_utc"))
            end = parse_ts(row.get("end_utc"))
            dest = row.get("dest")
            if dest and start:
                windows[dest] = (start, end or start)
    return windows


def load_fire_frames(frames_csv):
    """Every row of firewatch_frames.csv as a dict (scores -> float)."""
    rows = []
    with open(frames_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                row["best_score"] = float(row.get("best_score") or 0.0)
            except (TypeError, ValueError):
                row["best_score"] = 0.0
            row["_captured"] = parse_ts(row.get("captured_utc"))
            row["_labels"] = [s.strip().lower()
                              for s in (row.get("labels") or "").split(";") if s.strip()]
            row["_clips"] = [s.strip()
                             for s in (row.get("clips") or "").split(";") if s.strip()]
            rows.append(row)
    return rows


def select_fire_events(rows, label, min_score):
    """Keep label-matching rows whose best_score is STRICTLY above min_score."""
    want = (label or "").strip().lower()
    out = []
    for row in rows:
        if want not in ("", "any", "all") and want not in row["_labels"]:
            continue
        if row["best_score"] <= min_score:
            continue
        out.append(row)
    return out


def _find_evidence_image(images_dir, frame):
    """An already-downloaded evidence JPEG for this frame, if present."""
    if not images_dir:
        return None
    stored = frame.get("jpg_path") or ""
    candidates = [os.path.join(images_dir, frame.get("camera", ""),
                               os.path.basename(stored))]
    if stored:
        candidates.append(os.path.join(images_dir, stored.lstrip("/")))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def _extract_with_ffmpeg(clip_path, offset, dest):
    """Grab the frame at `offset` seconds; True on success."""
    offset = max(0.0, float(offset))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", "%.3f" % offset,
           "-i", clip_path, "-frames:v", "1", "-q:v", "2", dest]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and os.path.isfile(dest)


def _extract_with_cv2(clip_path, offset, dest):
    """cv2 fallback (only when ffmpeg is unavailable)."""
    try:
        import cv2  # noqa: PLC0415 - optional dependency, lazy import
    except Exception:
        return False
    cap = cv2.VideoCapture(clip_path)
    ok = False
    try:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(offset)) * 1000.0)
            got, frame = cap.read()
            ok = bool(got) and cv2.imwrite(dest, frame)
    finally:
        cap.release()
    return ok


def _dest_name(frame):
    """Output JPEG filename: <rank>_score<s>_<ts>_f<id>.jpg (rank per camera)."""
    captured = frame["_captured"]
    return "%03d_score%.3f_%s_f%s.jpg" % (
        frame["_rank"], frame["best_score"],
        human_ts(captured) if captured else "unknown",
        frame.get("frame_id") or "0")


def resolve_image(frame, clips_dir, windows, images_dir, out_images):
    """Separate this event's image into `out_images/<cam>/`.

    Returns (status, source, dest) with status in copy/extract/missing:
      copy    - an already-downloaded evidence JPEG was copied verbatim
      extract - a frame was extracted from a covering local clip
      missing - neither source was available
    """
    cam = frame.get("camera", "unknown")
    captured = frame["_captured"]
    dest = os.path.join(out_images, cam, _dest_name(frame))
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # 1) the exact evidence JPEG, if it has been downloaded
    evidence = _find_evidence_image(images_dir, frame)
    if evidence:
        shutil.copy2(evidence, dest)
        return "copy", evidence, dest

    # 2) a frame extracted from a covering local clip
    for clip_rel in frame["_clips"]:
        clip_path = os.path.join(clips_dir, clip_rel)
        if not os.path.isfile(clip_path):
            continue
        start, end = windows.get(clip_rel, (None, None))
        if start is None:
            offset = 0.0
        else:
            if captured is not None and not (start <= captured <= end):
                continue  # not actually covered by this segment
            offset = ((captured - start).total_seconds()
                      if captured is not None else 0.0)
        if _extract_with_ffmpeg(clip_path, offset, dest) or \
                _extract_with_cv2(clip_path, offset, dest):
            return "extract", clip_rel, dest

    return "missing", (frame["_clips"][0] if frame["_clips"] else ""), ""


# bucket label -> exclusive upper bound (lower bound is the previous bucket)
SCORE_BUCKETS = [(">0.5-0.6", 0.6), (">0.6-0.7", 0.7),
                 (">0.7-0.8", 0.8), (">0.8-1.0", 1.0001)]


def score_buckets(events):
    """Count events per score bucket (>0.5-0.6, >0.6-0.7, >0.7-0.8, >0.8-1.0)."""
    counts = {label: 0 for label, _upper in SCORE_BUCKETS}
    for ev in events:
        score = ev["best_score"]
        for label, upper in SCORE_BUCKETS:
            if score <= upper:
                counts[label] += 1
                break
    return counts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", default=DEFAULT_CLIPS_DIR,
                    help="camera-clip export root [%s]" % DEFAULT_CLIPS_DIR)
    ap.add_argument("--frames-csv", dest="frames_csv",
                    help="firewatch evidence CSV [<clips-dir>/firewatch_frames.csv]")
    ap.add_argument("--manifest", help="clip manifest [<clips-dir>/manifest.csv]")
    ap.add_argument("--min-score", dest="min_score", type=float,
                    default=DEFAULT_MIN_SCORE,
                    help="keep events with best_score STRICTLY above this [%s]"
                         % DEFAULT_MIN_SCORE)
    ap.add_argument("--label", default="fire",
                    help="required label substring; 'any' keeps all [fire]")
    ap.add_argument("--images-dir",
                    help="dir of already-downloaded evidence JPEGs "
                         "(<dir>/<cam>/<basename>)")
    ap.add_argument("--out", help="output dir [<clips-dir>/fire_events_over_<score>]")
    ap.add_argument("--keep-going", action="store_true",
                    help="exit 0 even when some events have no local image")
    args = ap.parse_args()

    clips_dir = args.clips_dir
    frames_csv = args.frames_csv or os.path.join(clips_dir, "firewatch_frames.csv")
    manifest = args.manifest or os.path.join(clips_dir, "manifest.csv")
    score_tag = ("%g" % args.min_score).replace(".", "_")
    out = args.out or os.path.join(clips_dir, "fire_events_over_%s" % score_tag)
    out_images = os.path.join(out, "images")

    if not os.path.isfile(frames_csv):
        sys.exit("ERROR: firewatch frames CSV not found: %s" % frames_csv)

    rows = load_fire_frames(frames_csv)
    windows = load_clip_windows(manifest)
    events = select_fire_events(rows, args.label, args.min_score)

    if not events:
        sys.exit("no events matched label=%r score>%s in %s" % (
            args.label, args.min_score, frames_csv))

    # rank per camera, highest score first (drives the output filenames)
    events.sort(key=lambda r: (r.get("camera", ""), -r["best_score"]))
    rank = {}
    for ev in events:
        cam = ev.get("camera", "unknown")
        rank[cam] = rank.get(cam, 0) + 1
        ev["_rank"] = rank[cam]

    os.makedirs(out_images, exist_ok=True)
    results = []
    counts = {"copy": 0, "extract": 0, "missing": 0}
    per_cam = {}
    for ev in events:
        status, source, image = resolve_image(
            ev, clips_dir, windows, args.images_dir, out_images)
        counts[status] += 1
        cam = ev.get("camera", "unknown")
        per_cam[cam] = per_cam.get(cam, 0) + 1
        results.append({
            "frame_id": ev.get("frame_id", ""),
            "camera": cam,
            "captured_utc": ev.get("captured_utc", ""),
            "best_score": ev["best_score"],
            "alerted": ev.get("alerted", ""),
            "labels": ";".join(ev["_labels"]),
            "image_status": status,
            "image": os.path.relpath(image, out) if image else "",
            "source": source,
        })

    # ---- filtered event list -------------------------------------------------
    csv_path = os.path.join(out, "fire_events_over_%s.csv" % score_tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ---- missing (no local image) -------------------------------------------
    miss_path = os.path.join(out, "missing_no_local_video.csv")
    with open(miss_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_id", "camera", "captured_utc", "best_score",
                         "alerted", "labels", "jpg_path"])
        for ev, res in zip(events, results):
            if res["image_status"] != "missing":
                continue
            writer.writerow([ev.get("frame_id", ""), ev.get("camera", ""),
                             ev.get("captured_utc", ""), ev["best_score"],
                             ev.get("alerted", ""), ";".join(ev["_labels"]),
                             ev.get("jpg_path", "")])

    # ---- exploration summary -------------------------------------------------
    buckets = score_buckets(events)
    alerted = sum(1 for e in events if str(e.get("alerted", "0")) == "1")
    found = counts["copy"] + counts["extract"]
    lines = []
    lines.append("Fire events above %s in %s" % (args.min_score, frames_csv))
    lines.append("=" * 60)
    lines.append("label filter        : %s" % args.label)
    lines.append("score filter        : best_score > %s" % args.min_score)
    lines.append("events matching     : %d (of %d evidence frames)" % (len(events), len(rows)))
    lines.append("alerted (Telegram)  : %d" % alerted)
    lines.append("")
    lines.append("Per camera:")
    for cam in sorted(per_cam):
        cam_events = [e for e in events if e.get("camera") == cam]
        best = max(e["best_score"] for e in cam_events)
        lines.append("  %-7s %4d events   max_score=%.4f" % (cam, per_cam[cam], best))
    lines.append("")
    lines.append("Score buckets:")
    for key in (">0.5-0.6", ">0.6-0.7", ">0.7-0.8", ">0.8-1.0"):
        lines.append("  %-10s %4d" % (key, buckets[key]))
    lines.append("")
    lines.append("Images separated:")
    lines.append("  copied evidence JPEGs : %d" % counts["copy"])
    lines.append("  extracted from clips  : %d" % counts["extract"])
    lines.append("  missing (no local clip): %d" % counts["missing"])
    lines.append("  -> %d/%d events have an image" % (found, len(events)))
    lines.append("")
    lines.append("Output:")
    lines.append("  events : %s" % csv_path)
    lines.append("  images : %s/<cam>/" % out_images)
    lines.append("  missing: %s" % miss_path)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out, "SUMMARY.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    if counts["missing"] and not args.keep_going:
        print("\nNOTE: %d event(s) have no local clip; their exact evidence JPEGs "
              "live on the host (/home/dr/frigate/media/firewatch/<cam>/)."
              % counts["missing"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
