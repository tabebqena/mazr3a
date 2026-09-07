#!/usr/bin/env python3
"""watchdog_baseline.py - long-run machine + Frigate state sampler for
before/after comparisons (e.g. a camera substream-quality change).

Runs on the Frigate HOST for a fixed duration (default 24 h) and appends one
CSV row every INTERVAL seconds (default 30). Captures exactly what is needed to
compare machine cost before/after a change:

  host      - load average 1/5/15 (/proc/loadavg)
            - host CPU % over the sample interval (/proc/stat delta)
            - memory used/available MB + % (/proc/meminfo)
            - hottest CPU temp (reuses scripts/collect_sensors.py)
  frigate   - container CPU % (docker stats --no-stream, best-effort)
            - global detection fps + detector inference ms (/api/stats)
            - cameras total / cameras online (camera_fps >= ONLINE_FPS_MIN)
            - events started in the last interval + running cumulative total
              (/api/events?after=<prev>&before=<now>)

Output (git-ignored host files - never written into the tracked tree):
  CSV          <out>/watchdog_baseline_<tag>_<start>.csv     (one row/sample)
  Summary JSON <out>/watchdog_baseline_<tag>_<start>.summary.json
               (mean/max/p95 aggregates, written at end and on SIGINT/SIGTERM)

Tunables come from CLI args or WATCHDOG_* env vars, each with a baked-in
default, so the script is safe to run with no arguments:

  --hours H / WATCHDOG_HOURS      run duration in hours            (default 24)
  --minutes M / WATCHDOG_MINUTES  run duration in minutes (overrides hours)
  --interval S / WATCHDOG_INTERVAL  seconds between samples        (default 30)
  --tag NAME / WATCHDOG_TAG       run tag in file names (default host-start)
  --out DIR / WATCHDOG_OUT        output dir (default <deploy>/media/watchdog)
  --once                           print a single sample and exit (sanity)
  --frames N                       stop after N samples (sanity / short runs)

Usage examples:
  python3 watchdog_baseline.py --once
  python3 watchdog_baseline.py --minutes 2 --interval 10     # smoke test
  nohup python3 watchdog_baseline.py --hours 24 > media/watchdog/run.log 2>&1 &

Every probe is best-effort: if Frigate/sensors/docker are unavailable the row
still records (blank numeric fields) and sampling continues.
"""
import argparse
import datetime
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import collect_sensors as sensors  # noqa: E402

# ---------------------------------------------------------------------------
# defaults (env-overridable)
# ---------------------------------------------------------------------------
DEFAULT_HOURS = 24
DEFAULT_INTERVAL = 30
DEFAULT_FRIGATE_API = os.environ.get("FRIGATE_API", "http://127.0.0.1:5000")
ONLINE_FPS_MIN = 0.5  # a camera counts as online when camera_fps >= this


def _env_int(name, default):
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


def _env_str(name, default):
    return str(os.environ.get(name, "")).strip() or default


# ---------------------------------------------------------------------------
# host probes (pure python, best-effort)
# ---------------------------------------------------------------------------
def _read_lines(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def load_avg():
    """Return (load1, load5, load15) floats, or Nones on failure."""
    line = _read_lines("/proc/loadavg")
    if not line:
        return None, None, None
    parts = line[0].split()
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return None, None, None


def _proc_stat_cpu():
    """Return a dict of the first 'cpu ' line of /proc/stat (jiffies)."""
    for line in _read_lines("/proc/stat"):
        if line.startswith("cpu "):
            parts = line.split()
            try:
                values = [int(v) for v in parts[1:]]
            except ValueError:
                return None
            if len(values) >= 8:
                return {
                    "user": values[0],
                    "nice": values[1],
                    "system": values[2],
                    "idle": values[3],
                    "iowait": values[4],
                    "irq": values[5],
                    "softirq": values[6],
                    "steal": values[7],
                }
            return None
    return None


def host_cpu_pct(prev_stat, cur_stat):
    """Host CPU% between two /proc/stat snapshots (None for first sample)."""
    if prev_stat is None or cur_stat is None:
        return None
    prev_total = sum(prev_stat.values())
    cur_total = sum(cur_stat.values())
    prev_idle = prev_stat["idle"] + prev_stat["iowait"]
    cur_idle = cur_stat["idle"] + cur_stat["iowait"]
    delta_total = cur_total - prev_total
    delta_idle = cur_idle - prev_idle
    if delta_total <= 0:
        return None
    return round(100.0 * (delta_total - delta_idle) / delta_total, 2)


def mem_info():
    """Return (used_mb, avail_mb, pct) or Nones on failure."""
    total_kb = avail_kb = None
    for line in _read_lines("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
    if not total_kb:
        return None, None, None
    avail_kb = avail_kb or 0
    used_kb = total_kb - avail_kb
    return (
        round(used_kb / 1024.0, 1),
        round(avail_kb / 1024.0, 1),
        round(used_kb * 100.0 / total_kb, 1),
    )


def frigate_docker_cpu():
    """Frigate container CPU % via `docker stats` (best-effort)."""
    try:
        result = subprocess.run(
            [
                "docker", "stats", "--no-stream",
                "--format", "{{.CPUPerc}}", "frigate",
            ],
            capture_output=True, text=True, timeout=20,
        )
        text = result.stdout.strip().rstrip("%")
        return round(float(text), 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# frigate probes (REST, best-effort)
# ---------------------------------------------------------------------------
def _api(path):
    try:
        with urllib.request.urlopen(
            f"{DEFAULT_FRIGATE_API}{path}", timeout=6
        ) as resp:
            return json.load(resp)
    except Exception:
        return None


def frigate_stats():
    """Return dict with frigate-level detection stats + per-camera fps.

    Returns:
      {"detection_fps": float|None, "inference_ms": float|None,
       "cameras_total": int|None, "cameras_online": int|None}
    """
    data = _api("/api/stats")
    if not isinstance(data, dict):
        return {
            "detection_fps": None, "inference_ms": None,
            "cameras_total": None, "cameras_online": None,
        }
    detection_raw = data.get("detection_fps")
    if isinstance(detection_raw, (int, float)) and not isinstance(
        detection_raw, bool
    ):
        detection = float(detection_raw)
    else:
        detection = None

    inference = []
    detectors = data.get("detectors") or {}
    for value in detectors.values():
        if isinstance(value, dict) and value.get("inference_speed"):
            try:
                inference.append(float(value["inference_speed"]))
            except (TypeError, ValueError):
                pass

    cams = data.get("cameras") or {}
    online = 0
    for value in cams.values():
        if isinstance(value, dict) and (value.get("camera_fps") or 0) >= ONLINE_FPS_MIN:
            online += 1
    return {
        "detection_fps": detection,
        "inference_ms": round(statistics.mean(inference), 1) if inference else None,
        "cameras_total": len(cams) if isinstance(cams, dict) else None,
        "cameras_online": online,
    }


def frigate_event_count(after_ts, before_ts):
    """Number of Frigate events that started in (after_ts, before_ts).

    Returns an int, or None when the API is unreachable. Best-effort: the
    per-window count is limited to 1000 events (generous for a 30 s window).
    """
    path = (
        f"/api/events?after={int(after_ts)}&before={int(before_ts)}"
        f"&limit=1000"
    )
    data = _api(path)
    if isinstance(data, list):
        return len(data)
    return None


# ---------------------------------------------------------------------------
# CSV + summary
# ---------------------------------------------------------------------------
CSV_HEADER = (
    "run,epoch,iso,load1,load5,load15,host_cpu_pct,"
    "mem_used_mb,mem_avail_mb,mem_pct,temp_c,"
    "frigate_cpu_pct,detection_fps,inference_ms,"
    "cameras_total,cameras_online,events_interval,events_cum"
)


def _f(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def row_values(run, sample):
    iso = datetime.datetime.fromtimestamp(
        sample["epoch"], datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        run, _f(sample["epoch"]), iso,
        _f(sample["load1"]), _f(sample["load5"]), _f(sample["load15"]),
        _f(sample["host_cpu_pct"]),
        _f(sample["mem_used_mb"]), _f(sample["mem_avail_mb"]), _f(sample["mem_pct"]),
        _f(sample["temp_c"]),
        _f(sample["frigate_cpu_pct"]),
        _f(sample["detection_fps"]), _f(sample["inference_ms"]),
        _f(sample["cameras_total"]), _f(sample["cameras_online"]),
        _f(sample["events_interval"]), _f(sample["events_cum"]),
    ]


def write_csv_header(csv_path):
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER + "\n")


def append_csv(csv_path, run, sample):
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write(",".join(str(v) for v in row_values(run, sample)) + "\n")
        fh.flush()


def _pct(values):
    clean = [v for v in values if v is not None]
    return clean


def _agg(clean_values):
    if not clean_values:
        return None
    return {
        "mean": round(statistics.mean(clean_values), 2),
        "max": round(max(clean_values), 2),
        "p95": round(sorted(clean_values)[int(len(clean_values) * 0.95) - 1], 2),
    }


def build_summary(samples, run_start, run_end, tag):
    """Aggregate the collected samples into a comparable summary dict."""
    summary = {
        "tag": tag,
        "run_start_iso": datetime.datetime.fromtimestamp(
            run_start, datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_end_iso": datetime.datetime.fromtimestamp(
            run_end, datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(run_end - run_start, 1),
        "samples": len(samples),
    }
    numeric = {
        "load1": lambda s: s["load1"],
        "host_cpu_pct": lambda s: s["host_cpu_pct"],
        "mem_pct": lambda s: s["mem_pct"],
        "temp_c": lambda s: s["temp_c"],
        "frigate_cpu_pct": lambda s: s["frigate_cpu_pct"],
        "detection_fps": lambda s: s["detection_fps"],
        "inference_ms": lambda s: s["inference_ms"],
        "cameras_online": lambda s: s["cameras_online"],
        "events_interval": lambda s: s["events_interval"],
    }
    for key, getter in numeric.items():
        values = _pct([getter(s) for s in samples])
        summary[key] = _agg(values)
    last = samples[-1] if samples else {}
    summary["events_cum_total"] = last.get("events_cum")
    summary["cameras_total"] = last.get("cameras_total")
    return summary


def save_summary(summary, summary_path):
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return True
    except OSError as exc:
        print(f"[watchdog] WARNING: cannot write summary {summary_path}: {exc}",
              file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def collect_sample(prev_stat, prev_epoch, events_cum):
    """Gather one full row of metrics (best-effort, never raises)."""
    cur_stat = _proc_stat_cpu()
    sample = {
        "epoch": time.time(),
        "load1": None, "load5": None, "load15": None,
        "host_cpu_pct": None,
        "mem_used_mb": None, "mem_avail_mb": None, "mem_pct": None,
        "temp_c": None,
        "frigate_cpu_pct": None,
        "detection_fps": None, "inference_ms": None,
        "cameras_total": None, "cameras_online": None,
        "events_interval": None, "events_cum": events_cum,
    }

    try:
        sample["load1"], sample["load5"], sample["load15"] = load_avg()
        sample["host_cpu_pct"] = host_cpu_pct(prev_stat, cur_stat)
        sample["mem_used_mb"], sample["mem_avail_mb"], sample["mem_pct"] = mem_info()
        sample["temp_c"] = sensors.get_cpu_temp_max()
    except Exception as exc:  # noqa: BLE001 - a probe must never kill the run
        print(f"[watchdog] host probe error: {exc}", file=sys.stderr)

    try:
        sample["frigate_cpu_pct"] = frigate_docker_cpu()
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] docker probe error: {exc}", file=sys.stderr)

    stats = frigate_stats()
    sample.update(stats)

    if prev_epoch is not None:
        count = frigate_event_count(prev_epoch, sample["epoch"])
        if count is not None:
            sample["events_interval"] = count
            sample["events_cum"] = (events_cum or 0) + count
    return sample, cur_stat


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Machine + Frigate state sampler (baseline recorder)."
    )
    parser.add_argument("--hours", type=int,
                        default=_env_int("WATCHDOG_HOURS", DEFAULT_HOURS),
                        help="run duration in hours (default %(default)s)")
    parser.add_argument("--minutes", type=int,
                        default=_env_int("WATCHDOG_MINUTES", 0),
                        help="run duration in minutes (overrides --hours)")
    parser.add_argument("--interval", type=int,
                        default=_env_int("WATCHDOG_INTERVAL", DEFAULT_INTERVAL),
                        help="seconds between samples (default %(default)s)")
    parser.add_argument("--tag", default=_env_str("WATCHDOG_TAG", ""),
                        help="run tag used in output file names")
    parser.add_argument("--out", default=_env_str("WATCHDOG_OUT", ""),
                        help="output directory (default <deploy>/media/watchdog)")
    parser.add_argument("--once", action="store_true",
                        help="print a single sample and exit (sanity check)")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N samples (short runs; 0 = until duration)")
    return parser.parse_args(argv)


def default_out_dir():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    deploy_root = os.path.abspath(os.path.join(script_dir, ".."))
    return os.path.join(deploy_root, "media", "watchdog")


def make_run_tag():
    host = os.uname().nodename if hasattr(os, "uname") else "host"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{host}_{stamp}"


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.minutes:
        run_seconds = max(1, args.minutes * 60)
    else:
        run_seconds = max(1, args.hours * 3600)
    interval = max(2, args.interval)
    tag = args.tag or make_run_tag()

    # --once: emit one row to stdout (no files) and exit.
    if args.once:
        sample, _ = collect_sample(None, None, 0)
        fields = zip(CSV_HEADER.split(","), row_values(tag, sample))
        print(" ".join(f"{k}={v}" for k, v in fields if v != ""))
        print(f"[watchdog] run={tag} epoch={_f(sample['epoch'])} "
              f"cameras_online={_f(sample['cameras_online'])} "
              f"events={_f(sample['events_cum'])} temp={_f(sample['temp_c'])}")
        return 0

    out_dir = args.out or default_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    start_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    base = f"watchdog_baseline_{tag}_{start_iso}"
    csv_path = os.path.join(out_dir, base + ".csv")
    summary_path = os.path.join(out_dir, base + ".summary.json")

    write_csv_header(csv_path)
    print(f"[watchdog] run tag={tag}")
    print(f"[watchdog] duration={run_seconds}s interval={interval}s")
    print(f"[watchdog] csv={csv_path}")

    run_start = time.time()
    deadline = run_start + run_seconds
    samples = []
    events_cum = 0
    prev_stat = None
    prev_epoch = None
    frigate_down = False
    stop = {"flag": False}

    def _handler(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    try:
        while not stop["flag"]:
            sample, prev_stat = collect_sample(
                prev_stat, prev_epoch, events_cum
            )
            # Frigate reachability for the event counter: only advance the
            # counting baseline when /api/stats answered (frigate_stats fills
            # cameras_total). Keeps the window free of huge catch-up gaps.
            if sample["cameras_total"] is not None:
                frigate_down = False
                events_cum = sample["events_cum"]
                prev_epoch = sample["epoch"]
            else:
                frigate_down = True

            samples.append(sample)
            append_csv(csv_path, tag, sample)

            elapsed = time.time() - run_start
            # one short progress line per sample (visible in a nohup log)
            print(
                f"[watchdog] {datetime.datetime.fromtimestamp(sample['epoch'], datetime.timezone.utc).strftime('%H:%M:%SZ')} "
                f"load={_f(sample['load1'])} cpu%={_f(sample['host_cpu_pct'])} "
                f"mem%={_f(sample['mem_pct'])} temp={_f(sample['temp_c'])} "
                f"frigate_cpu%={_f(sample['frigate_cpu_pct'])} "
                f"online={_f(sample['cameras_online'])}/{_f(sample['cameras_total'])} "
                f"events={_f(sample['events_cum'])}",
                flush=True,
            )

            if args.frames and len(samples) >= args.frames:
                break
            if time.time() >= deadline:
                break
            time.sleep(max(0.0, min(interval, deadline - time.time())))
    except KeyboardInterrupt:
        pass

    run_end = time.time()
    summary = build_summary(samples, run_start, run_end, tag)
    save_summary(summary, summary_path)
    print(f"[watchdog] done: {len(samples)} samples in "
          f"{run_end - run_start:.0f}s -> {csv_path}")
    print(f"[watchdog] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
