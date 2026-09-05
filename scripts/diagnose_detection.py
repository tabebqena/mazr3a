#!/usr/bin/env python3
"""Run ON the remote Frigate host (Phase 1 diagnosis - READ ONLY, no changes).

Collects:
  - /api/stats   per-camera camera_fps / process_fps / detection_fps / detection_enabled
                 + detector inference_speed (CPU headroom verdict)
  - /api/events  recent event label histogram, per-camera counts, day vs night split
  - frigate.log  scan for dropped-frame / error / watchdog / crash patterns
  - /api/config  effective detect fps, model width/height, person filters

Usage:  python3 scripts/diagnose_detection.py   (on the host, /home/dr/frigate)
"""
import json
import time
import urllib.request
from collections import Counter
from datetime import datetime

BASE = "http://localhost:5000"


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.load(r)


def main():
    print("=" * 70)
    print("PHASE 1 DIAGNOSIS - gathered", datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("=" * 70)

    # ---------------------------------------------------------- stats
    try:
        stats = get("/api/stats")
    except Exception as e:
        print("FATAL: /api/stats failed:", e)
        return

    dets = stats.get("detectors", {})
    for name, d in dets.items():
        print(f"\n[detector:{name}] inference_speed={d.get('inference_speed')}ms "
              f"detection_start={d.get('detection_start')}")
        print(f"  CPU/other stats: {json.dumps(d, default=str)[:300]}")

    print("\n[per-camera stats]")
    print(f"{'cam':7} {'camera_fps':>10} {'process_fps':>11} {'detect_fps':>10} {'det_en':>6} "
          f"{'motion':>8} {'fps_ratio':>9}")
    cams = stats.get("cameras", {})
    for c in sorted(cams):
        v = cams[c]
        cf = v.get("camera_fps", 0) or 0
        pf = v.get("process_fps", 0) or 0
        df = v.get("detection_fps", 0) or 0
        de = v.get("detection_enabled")
        ratio = (df / pf) if pf else 0.0
        mot = v.get("motion")
        print(f"{c:7} {cf:10.2f} {pf:11.2f} {df:10.2f} {str(de):>6} "
              f"{(mot if mot is not None else 0):8.0f} {ratio:9.2f}")

    # CPU headroom verdict
    df_list = [cams[c].get("detection_fps", 0) or 0 for c in cams]
    pf_list = [cams[c].get("process_fps", 0) or 0 for c in cams]
    tot_df = sum(df_list)
    tot_pf = sum(pf_list)
    print(f"\n[verdict] sum(detection_fps)={tot_df:.1f}  sum(process_fps)={tot_pf:.1f}")
    if tot_pf > 0 and tot_df < tot_pf * 0.85:
        print("  -> CPU LIMITED: detection_fps trails process_fps (frames skipped). "
              "Raising detect.fps would likely make this WORSE. Model upgrade/detect.fps "
              "changes must account for this.")
    elif tot_pf > 0:
        print("  -> HEADROOM: detection keeps up with processing; raising detect.fps on "
              "specific cameras is viable if fast movers are the issue.")

    # --------------------------------------------------------- events
    try:
        evs = get("/api/events?limit=1000")
    except Exception as e:
        print("FATAL: /api/events failed:", e)
        evs = []
    print("\n[events] total fetched:", len(evs))
    if evs:
        labels = Counter(e.get("label") for e in evs)
        print("  by label:", dict(labels))
        percam = Counter(e.get("camera") for e in evs)
        print("  by camera:", dict(percam))

        # day vs night (host local time = Asia/Riyadh)
        def is_day(ts):
            if not ts:
                return None
            try:
                return 7 <= time.localtime(ts).tm_hour < 19
            except Exception:
                return None

        day = Counter(e.get("label") for e in evs if is_day(e.get("start_time")) is True)
        night = Counter(e.get("label") for e in evs if is_day(e.get("start_time")) is False)
        print("  DAY labels  :", dict(day))
        print("  NIGHT labels:", dict(night))
        # recent sample
        for e in evs[:15]:
            st = e.get("start_time")
            tstr = datetime.fromtimestamp(st).strftime("%m-%d %H:%M") if st else "?"
            print(f"    {tstr}  cam={e.get('camera'):7} label={e.get('label'):10} "
                  f"score={e.get('data',{}).get('top_score')}")

    # -------------------------------------------------------- config
    try:
        cfg = get("/api/config")
        m = cfg.get("model", {})
        print("\n[config] model:", m.get("path"), "w/h:", m.get("width"), m.get("height"))
        print("  global track:", cfg.get("objects", {}).get("track"))
        pf = cfg.get("objects", {}).get("filters", {}).get("person", {})
        print("  person filter:", pf)
        print("  motion:", cfg.get("motion"))
        for c in sorted(cams):
            cam = cfg.get("cameras", {}).get(c, {})
            det = cam.get("detect", {})
            print(f"  {c}: detect={det.get('enabled')} fps={det.get('fps')} "
                  f"w/h={det.get('width')}x{det.get('height')} "
                  f"track={cam.get('objects',{}).get('track','<global>')}")
    except Exception as e:
        print("[config] failed:", e)

    # ----------------------------------------------------------- logs
    print("\n[log scan: /home/dr/frigate/config/frigate.log]")
    try:
        with open("/home/dr/frigate/config/frigate.log", "r", errors="replace") as f:
            lines = f.readlines()[-4000:]
        patterns = {
            "dropped frame": "dropped frame|frame drop",
            "ERROR": "ERROR",
            "watchdog/restart": "watchdog|restarting|crashed unexpectedly",
            "VPS/invalid": "VPS 0 does not exist|Invalid data|Invalid or missing video",
            "detection stopped": "Detection appears to have stopped|detection toggled",
        }
        for name, pat in patterns.items():
            hits = [ln.strip()[:220] for ln in lines if __import__("re").search(pat, ln, __import__("re").I)]
            print(f"  -- {name}: {len(hits)} hit(s)")
            for h in hits[-5:]:
                print("     ", h)
    except Exception as e:
        print("  log read failed:", e)


if __name__ == "__main__":
    main()
