#!/usr/bin/env python3
"""export_firewatch_clips.py - pull Frigate recordings that cover firewatch alerts.

Run ON the Frigate host (ai or dr, docker group for `docker logs`). It treats the
firewatch log as the AUTHORITATIVE registry of fire alerts: for every registered
`camNN: ALERT` line it finds the Frigate event recordings whose time window covers
the alert instant and copies those .mp4 segments out.

Everything that has NO firewatch log/DB record is ignored by construction (we only
select recordings that overlap a registered fire alert).

Why this exists (user request 2026-09-08): firewatch never stores images - it only
sends a Telegram photo per alert. Frigate does not track fire (fire is out-of-band),
but event-only recording still captured 10 s segments while persons/objects were
near the fires, so the alert timestamps let us recover the video clips.

Sources
-------
- Fire alert registry : `docker logs firewatch` (default) or a captured log file.
  Only the CURRENT container's log is recoverable (created 2026-09-07 17:02 UTC);
  the 2026-09-06 alerts predate it and their logs were never saved - they cannot be
  recovered, so they are intentionally absent here (see plans/firewatch-clip-export.md).
- Recording index    : Frigate sqlite DB (`config/frigate.db` -> `recordings` table),
  which stores container paths `/media/frigate/recordings/<date>/<hour>/<cam>/<t>.mp4`
  with per-segment start/end epoch. Host files live under `media/` on the host.

Usage
-----
  python3 export_firewatch_clips.py --list-alerts            # show parsed registry
  python3 export_firewatch_clips.py --dry-run                # show alert->clip map
  python3 export_firewatch_clips.py                          # copy clips to --out

Tunables (env or flag; host defaults in []):
  FIREWATCH_LOG / --log       log source: 'docker:firewatch' or a FILE path [docker:firewatch]
  FRIGATE_DB    / --db        frigate sqlite db [/home/dr/frigate/config/frigate.db]
  MEDIA_ROOT    / --media-root host media dir, prefix for /media/frigate/... [/home/dr/frigate/media]
  FW_CLIP_OUT   / --out       output dir [/home/dr/frigate/firewatch_clips]
  FW_PRE_S      / --pre       seconds of recordings to include BEFORE an alert [120]
  FW_POST_S     / --post      seconds of recordings to include AFTER  an alert [120]
  --cam cam01,cam03           restrict to these cameras
  --utc-offset                hours added to the firewatch log clock if the container
                              is not UTC (default 0; this deploy logs UTC)
"""
import argparse
import csv
import datetime
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys

UTC = datetime.timezone.utc

# ---------------------------------------------------------------------------
# 1) Fire alert registry from the firewatch log
# ---------------------------------------------------------------------------
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
ALERT_CAM_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+"
                          r"([A-Za-z0-9_]+):\s*ALERT\s*\(")
SENT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*"
                     r"ALERT sent for\s+([A-Za-z0-9_]+)")


def read_log(log_source):
    """Return the full firewatch log text from 'docker:<name>' or a file."""
    if log_source.startswith("docker:"):
        name = log_source.split(":", 1)[1]
        out = subprocess.run(["docker", "logs", name], capture_output=True,
                             text=True, check=False)
        if out.returncode != 0:
            raise SystemExit(f"ERROR: `docker logs {name}` failed: {out.stderr}")
        return out.stdout
    with open(log_source, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_alerts(log_text, utc_offset=0.0):
    """Return sorted [{cam, ts_epoch, ts_str, conf?}] from `cam: ALERT (...)` lines.

    `ALERT sent for cam` lines are used only when no `cam: ALERT (` line exists
    (they are otherwise 1-2 s echoes of the same registration).
    """
    regs = []
    for line in log_text.splitlines():
        m = ALERT_CAM_RE.match(line)
        if m:
            ts_str, cam = m.groups()
            regs.append({"cam": cam, "ts_str": ts_str})
        else:
            m2 = SENT_RE.match(line)
            if m2:
                ts_str, cam = m2.groups()
                regs.append({"cam": cam, "ts_str": ts_str})
    if not regs:
        return []
    for r in regs:
        naive = datetime.datetime.strptime(r["ts_str"], "%Y-%m-%d %H:%M:%S")
        naive = naive.replace(tzinfo=UTC)  # assume UTC unless --utc-offset given
        r["ts_epoch"] = naive.timestamp() - utc_offset * 3600.0
    regs.sort(key=lambda r: (r["ts_epoch"], r["cam"]))
    # drop the 1-2 s "ALERT sent for" echoes that duplicate an "ALERT (" moment
    out = []
    for r in regs:
        if out and out[-1]["cam"] == r["cam"] and r["ts_epoch"] - out[-1]["ts_epoch"] < 10:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 2) Recording selection from Frigate DB
# ---------------------------------------------------------------------------
def host_path(media_root, container_path):
    """Map a Frigate container path (/media/frigate/...) to the host path."""
    if container_path.startswith("/media/frigate/"):
        return os.path.join(media_root, container_path[len("/media/frigate/"):])
    return container_path  # already absolute/host


def select_recordings(db_path, media_root, alerts, pre_s, post_s, only_cams=None):
    """Return {alert_idx: [ {path, start, end, exists} ]} in time order + gaps.

    Query is per (cam, t) window [t-pre_s, t+post_s]; segments must OVERLAP it.
    Only files present on disk are returned (DB rows may outlive cleanup).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cams = sorted({a["cam"] for a in alerts})
    if only_cams:
        cams = [c for c in cams if c in only_cams]
    by_cam = {}
    for c in cams:
        cur.execute("SELECT path, start_time, end_time FROM recordings "
                    "WHERE camera=? ORDER BY start_time", (c,))
        by_cam[c] = cur.fetchall()
    con.close()

    result = {}
    for i, a in enumerate(alerts):
        if only_cams and a["cam"] not in only_cams:
            continue
        lo, hi = a["ts_epoch"] - pre_s, a["ts_epoch"] + post_s
        picks = []
        for path, start, end in by_cam.get(a["cam"], []):
            if start <= hi and end >= lo:  # overlap
                hp = host_path(media_root, path)
                picks.append({"path": hp, "start": start, "end": end,
                              "exists": os.path.isfile(hp)})
        picks.sort(key=lambda p: p["start"])
        result[i] = picks
    return result


# ---------------------------------------------------------------------------
# 3) main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=os.environ.get("FIREWATCH_LOG", "docker:firewatch"),
                    help="log source: 'docker:<name>' or a file")
    ap.add_argument("--db", default=os.environ.get("FRIGATE_DB",
                    "/home/dr/frigate/config/frigate.db"))
    ap.add_argument("--media-root", default=os.environ.get("MEDIA_ROOT",
                    "/home/dr/frigate/media"))
    ap.add_argument("--out", default=os.environ.get("FW_CLIP_OUT",
                    "/home/dr/frigate/firewatch_clips"))
    ap.add_argument("--pre", type=float, default=float(os.environ.get("FW_PRE_S", "120")))
    ap.add_argument("--post", type=float, default=float(os.environ.get("FW_POST_S", "120")))
    ap.add_argument("--cam", help="comma-separated camera filter")
    ap.add_argument("--utc-offset", type=float, default=0.0,
                    help="hours to subtract from firewatch log clock if not UTC")
    ap.add_argument("--list-alerts", action="store_true", help="print the alert registry")
    ap.add_argument("--dry-run", action="store_true", help="plan only, copy nothing")
    args = ap.parse_args()

    only_cams = {c.strip() for c in args.cam.split(",")} if args.cam else None

    log_text = read_log(args.log)
    alerts = parse_alerts(log_text, args.utc_offset)
    if not alerts:
        print("No firewatch ALERT lines found in log source:", args.log)
        return 1

    print(f"fire alert registry: {len(alerts)} alert(s) from {args.log}\n")
    if args.list_alerts:
        for a in alerts:
            ts = datetime.datetime.fromtimestamp(a["ts_epoch"], UTC) \
                     .strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {ts} UTC  {a['cam']}")
        return 0

    picks = select_recordings(args.db, args.media_root, alerts,
                              args.pre, args.post, only_cams)

    n_picks = sum(len(v) for v in picks.values())
    n_exist = sum(1 for v in picks.values() for p in v if p["exists"])
    n_missing = sum(1 for v in picks.values() for p in v if not p["exists"])
    n_nocover = sum(
        1 for i, _a in enumerate(alerts)
        if not any(p["exists"] for p in picks.get(i, []))
    )
    total_bytes = sum(p.get("_size", os.path.getsize(p["path"])) if p["exists"] else 0
                      for v in picks.values() for p in v)

    if args.dry_run:
        print(f"matched {n_picks} DB segment(s) -> {n_exist} on disk, "
              f"{n_missing} cleaned up, {n_nocover} alert(s) with NO covering clip\n")
        for i, a in enumerate(alerts):
            if only_cams and a["cam"] not in only_cams:
                continue
            ts = datetime.datetime.fromtimestamp(a["ts_epoch"], UTC) \
                     .strftime("%Y-%m-%d %H:%M:%S")
            files = [p for p in picks.get(i, []) if p["exists"]]
            if not files:
                print(f"  NO COVER  {ts} UTC {a['cam']}")
                continue
            print(f"  cover  {ts} UTC {a['cam']}  ({len(files)} seg, "
                  f"{sum(os.path.getsize(p['path']) for p in files)//1024} KiB):")
            for p in files:
                st = datetime.datetime.fromtimestamp(p["start"], UTC) \
                        .strftime("%H:%M:%S")
                en = datetime.datetime.fromtimestamp(p["end"], UTC) \
                        .strftime("%H:%M:%S")
                print(f"      {os.path.basename(p['path'])}  [{st}->{en}]")
        return 0

    # ---- real extraction ----
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    copied = 0
    for i, a in enumerate(alerts):
        if only_cams and a["cam"] not in only_cams:
            continue
        files = [p for p in picks.get(i, []) if p["exists"]]
        if not files:
            continue
        folder = os.path.join(
            args.out,
            datetime.datetime.fromtimestamp(a["ts_epoch"], UTC)
                .strftime("%Y%m%d_%H%M%S") + "_" + a["cam"])
        os.makedirs(folder, exist_ok=True)
        for p in files:
            base = os.path.basename(p["path"])
            dst = os.path.join(folder, base)
            if not os.path.exists(dst):
                shutil.copy2(p["path"], dst)
                copied += 1
            st = datetime.datetime.fromtimestamp(p["start"], UTC) \
                    .strftime("%Y-%m-%d %H:%M:%S")
            en = datetime.datetime.fromtimestamp(p["end"], UTC) \
                    .strftime("%Y-%m-%d %H:%M:%S")
            manifest.append({"alert_utc": datetime.datetime
                             .fromtimestamp(a["ts_epoch"], UTC)
                             .strftime("%Y-%m-%d %H:%M:%S"),
                             "camera": a["cam"], "seg_start_utc": st,
                             "seg_end_utc": en,
                             "source": p["path"],
                             "dest": os.path.relpath(dst, args.out)})

    # summary file for the run
    summary = os.path.join(args.out, "manifest.csv")
    with open(summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys())
                           if manifest else ["alert_utc"])
        w.writeheader()
        w.writerows(manifest)

    print(f"copied {copied} segment(s) into {args.out}")
    print(f"alerts with no covering clip (not downloaded): {n_nocover}")
    print(f"manifest: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
