#!/usr/bin/env python3
"""watchdog_baseline.py - machine + Frigate state sampler for before/after
comparisons (e.g. a camera substream-quality change) AND as a cron-driven
rolling recorder.

TWO run modes:

1) CRON mode (--cron) - the standard way to keep it running unattended.
   Each cron invocation samples ONCE, appends one row to a PERSISTENT CSV
   (one file per tag, e.g. watchdog_baseline_pre.csv) and exits. Host CPU% is
   computed against the previous invocation via a small sidecar file, so the
   metric stays meaningful across separate processes. The CSV is bounded by
   --max-rows (oldest rows pruned), i.e. the stored log has a size limit.

   crontab example (every minute, quiet - data lives in the CSV):
       * * * * * /usr/bin/python3 /home/dr/frigate/scripts/watchdog_baseline.py \
                   --cron --tag baseline_pre >> /dev/null 2>&1

2) LONG-RUN mode (--hours, default 24) - one process that samples every
   --interval seconds for the duration, then writes a summary JSON (this is
   the original 24 h one-shot experiment). SIGINT/SIGTERM still writes the
   summary.

Metrics per row:
  host      - load average 1/5/15 (/proc/loadavg)
            - host CPU % (cron: delta vs previous invocation via sidecar;
              long-run: delta between samples via /proc/stat)
            - memory used/available MB + % (/proc/meminfo)
            - hottest CPU temp (scripts/collect_sensors.py)
            - GPU busy % + GPU temp (scripts/collect_sensors.py; left blank when
              the host exposes none - the Intel iGPU has no separate temp
              sensor, so on this host the temp column is normally empty)
  frigate   - container CPU % (docker stats --no-stream, best-effort)
            - global detection fps + detector inference ms (/api/stats)
            - cameras total / cameras online (camera_fps >= ONLINE_FPS_MIN)
            - events started in the recent window + running cumulative total
              (/api/events?after=<prev>&before=<now>)

Storage (git-ignored host files - never written into the tracked tree):
  cron mode   <out>/watchdog_baseline_<tag>.csv  (persistent, capped rows)
  long-run    <out>/watchdog_baseline_<tag>_<start>.csv + .summary.json
  out dir     default <deploy>/media/watchdog, falling back to ~/watchdog when
              the deploy dir is not writable (e.g. cron running as a non-owner).

  NOTE (2026-09-10): the columns gpu_usage_pct,gpu_temp_c were APPENDED to the
  CSV. Rows written by an older version therefore have two fewer fields than
  the header; the in-repo reader is index-based and unaffected, but start a new
  --tag (e.g. "yolo11s_after") for a clean, uniform file.

Tunables: CLI args or WATCHDOG_* env vars, each with a baked-in default.
  --cron / WATCHDOG_CRON=1       one-shot cron sampling mode
  --tag NAME / WATCHDOG_TAG      run/file tag (default: hostname in cron mode)
  --max-rows N / WATCHDOG_MAX_ROWS  keep at most N CSV rows (cron retention)
  --hours H / WATCHDOG_HOURS     long-run duration in hours     (default 24)
  --minutes M / WATCHDOG_MINUTES long-run duration in minutes   (overrides h)
  --interval S / WATCHDOG_INTERVAL seconds between long-run samples (30)
  --out DIR / WATCHDOG_OUT       output directory
  --once                          print a single sample and exit (sanity)
  --frames N                      stop after N long-run samples

Examples:
  python3 watchdog_baseline.py --once
  python3 watchdog_baseline.py --cron --tag baseline_pre     # cron tick
  python3 watchdog_baseline.py --hours 24                    # 24 h one-shot

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
DEFAULT_MAX_ROWS = 20000        # cron CSV retention cap (~14 days @ 1/min)
CRON_EVENT_WINDOW_S = 70        # events window covering a 1-minute cron tick
DEFAULT_FRIGATE_API = os.environ.get("FRIGATE_API", "http://127.0.0.1:5000")
ONLINE_FPS_MIN = 0.5            # a camera counts as online when camera_fps >= this


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
    """Host CPU% between two /proc/stat snapshots (None if any missing)."""
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


def _load_cpu_stat_file(path):
    """Load the persisted /proc/stat snapshot for cross-run CPU% (best-effort)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            key: int(value)
            for key, value in data.items()
            if key in ("user", "nice", "system", "idle", "iowait", "irq",
                       "softirq", "steal")
        }
    except (OSError, ValueError, TypeError):
        return None


def _save_cpu_stat_file(path, stat):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stat, fh)
    except OSError:
        pass  # best-effort: losing the sidecar only blanks one CPU% sample


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
    """Return dict with frigate-level detection stats + per-camera fps."""
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
# CSV helpers (append + bounded retention)
# ---------------------------------------------------------------------------
CSV_HEADER = (
    "run,epoch,iso,load1,load5,load15,host_cpu_pct,"
    "mem_used_mb,mem_avail_mb,mem_pct,temp_c,"
    "frigate_cpu_pct,detection_fps,inference_ms,"
    "cameras_total,cameras_online,events_interval,events_cum,"
    "gpu_usage_pct,gpu_temp_c"
)
# Column index of events_cum (for cross-run cumulative reads). Resolved BY NAME
# so appending new columns at the end can never silently shift it.
_EVENTS_CUM_COL = CSV_HEADER.split(",").index("events_cum")


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
        _f(sample["gpu_usage_pct"]), _f(sample["gpu_temp_c"]),
    ]


def ensure_csv(csv_path, run, sample):
    """Create the CSV with a header when it does not exist yet."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        try:
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write(CSV_HEADER + "\n")
                fh.flush()
        except OSError:
            return False
    return True


def append_csv(csv_path, run, sample):
    """Append one row and flush (caller ensures the header exists)."""
    try:
        with open(csv_path, "a", encoding="utf-8") as fh:
            fh.write(",".join(str(v) for v in row_values(run, sample)) + "\n")
            fh.flush()
        return True
    except OSError:
        return False


def read_last_events_cum(csv_path):
    """Read the cumulative event count from the last CSV row (or 0)."""
    try:
        with open(csv_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        for line in reversed(lines):
            if not line or line.startswith("run,"):
                continue
            fields = line.split(",")
            if len(fields) > _EVENTS_CUM_COL and fields[_EVENTS_CUM_COL]:
                return int(float(fields[_EVENTS_CUM_COL]))
            return 0
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def prune_csv(csv_path, max_rows):
    """Trim a persistent CSV to keep the header + the newest max_rows rows.

    Prunes only once the file exceeds max_rows (not on every append) and keeps
    a headroom of 10% (min 500) so rewrites stay infrequent. This is the
    "log size limit" for the cron CSV.
    """
    try:
        with open(csv_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return
    if len(lines) - 1 <= max_rows:
        return
    headroom = max(1, int(max_rows * 0.10))
    keep = max(1, max_rows - headroom)
    kept = [lines[0]] + lines[-(keep):] if lines else []
    tmp = csv_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(kept) + "\n")
            fh.flush()
        os.replace(tmp, csv_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# summary (long-run mode)
# ---------------------------------------------------------------------------
def _pct(values):
    return [v for v in values if v is not None]


def _agg(clean_values):
    if not clean_values:
        return None
    return {
        "mean": round(statistics.mean(clean_values), 2),
        "max": round(max(clean_values), 2),
        "p95": round(sorted(clean_values)[int(len(clean_values) * 0.95) - 1], 2),
    }


def build_summary(samples, run_start, run_end, tag):
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
        "gpu_usage_pct": lambda s: s["gpu_usage_pct"],
        "gpu_temp_c": lambda s: s["gpu_temp_c"],
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
        "gpu_usage_pct": None, "gpu_temp_c": None,
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

    try:
        # GPU busy% (i915 RC6 counter) + GPU temp when the host exposes one.
        sample["gpu_usage_pct"] = sensors.get_gpu_usage()
        sample["gpu_temp_c"] = sensors.get_gpu_temp()
    except Exception as exc:  # noqa: BLE001 - a probe must never kill the run
        print(f"[watchdog] gpu probe error: {exc}", file=sys.stderr)

    stats = frigate_stats()
    sample.update(stats)

    if prev_epoch is not None:
        count = frigate_event_count(prev_epoch, sample["epoch"])
        if count is not None:
            sample["events_interval"] = count
            sample["events_cum"] = (events_cum or 0) + count
    return sample, cur_stat


def _acquire_lock(lock_path):
    """Simple exclusive lock so two cron ticks never append at once."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return lock_path
    except OSError:
        return None


def _release_lock(lock_path):
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def run_cron(args):
    """One-shot cron sampling: append one row to a persistent capped CSV."""
    tag = args.tag or (os.uname().nodename if hasattr(os, "uname") else "host")
    max_rows = max(2, args.max_rows)
    out_dir = args.out or default_out_dir()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"[watchdog] cannot create out dir {out_dir}: {exc}",
              file=sys.stderr)
        return 1

    csv_path = os.path.join(out_dir, f"watchdog_baseline_{tag}.csv")
    stat_path = os.path.join(out_dir, f"watchdog_baseline_{tag}.stat")
    lock_path = os.path.join(out_dir, f"watchdog_baseline_{tag}.lock")

    lock = _acquire_lock(lock_path)
    if lock is None:
        return 0  # previous tick still running - skip silently
    try:
        if not ensure_csv(csv_path, tag, {}):
            print(f"[watchdog] cannot create {csv_path}", file=sys.stderr)
            return 1

        # cross-run CPU%: delta vs the previous cron invocation's stat snapshot
        prev_stat = _load_cpu_stat_file(stat_path)
        prev_cum = read_last_events_cum(csv_path)
        now = time.time()
        # events counted in the last CRON_EVENT_WINDOW_S (covers a 1-min tick)
        prev_epoch = now - CRON_EVENT_WINDOW_S
        sample, cur_stat = collect_sample(prev_stat, prev_epoch, prev_cum)
        _save_cpu_stat_file(stat_path, cur_stat)

        if not append_csv(csv_path, tag, sample):
            print(f"[watchdog] cannot append to {csv_path}", file=sys.stderr)
            return 1
        prune_csv(csv_path, max_rows)
        return 0
    finally:
        _release_lock(lock_path)


# ---------------------------------------------------------------------------
# CLI + long-run mode
# ---------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Machine + Frigate state sampler (baseline recorder)."
    )
    parser.add_argument("--cron", action="store_true",
                        default=os.environ.get("WATCHDOG_CRON") == "1",
                        help="one-shot cron sampling mode (append one row)")
    parser.add_argument("--tag",
                        default=_env_str("WATCHDOG_TAG", ""),
                        help="run/file tag (default: hostname in cron mode)")
    parser.add_argument("--max-rows", type=int,
                        default=_env_int("WATCHDOG_MAX_ROWS", DEFAULT_MAX_ROWS),
                        help="keep at most N CSV rows in cron mode "
                             "(default %(default)s)")
    parser.add_argument("--hours", type=int,
                        default=_env_int("WATCHDOG_HOURS", DEFAULT_HOURS),
                        help="long-run duration in hours (default %(default)s)")
    parser.add_argument("--minutes", type=int,
                        default=_env_int("WATCHDOG_MINUTES", 0),
                        help="long-run duration in minutes (overrides --hours)")
    parser.add_argument("--interval", type=int,
                        default=_env_int("WATCHDOG_INTERVAL", DEFAULT_INTERVAL),
                        help="seconds between long-run samples "
                             "(default %(default)s)")
    parser.add_argument("--out", default=_env_str("WATCHDOG_OUT", ""),
                        help="output directory (default deploy media/watchdog "
                             "or ~/watchdog fallback)")
    parser.add_argument("--once", action="store_true",
                        help="print a single sample and exit (sanity check)")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N long-run samples (0 = until duration)")
    return parser.parse_args(argv)


def default_out_dir():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    deploy_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.path.join(deploy_root, "media", "watchdog"),
        os.path.join(os.path.expanduser("~"), "watchdog"),
    ]
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            return candidate
    return candidates[0]


def make_run_tag():
    host = os.uname().nodename if hasattr(os, "uname") else "host"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{host}_{stamp}"


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # cron mode: one sample per invocation, persistent capped CSV, exit.
    if args.cron:
        return run_cron(args)

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
            # Only advance the event baseline when /api/stats answered
            # (cameras_total is filled) - keeps the window free of huge gaps.
            if sample["cameras_total"] is not None:
                events_cum = sample["events_cum"]
                prev_epoch = sample["epoch"]

            samples.append(sample)
            append_csv(csv_path, tag, sample)

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


def write_csv_header(csv_path):
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER + "\n")


if __name__ == "__main__":
    sys.exit(main())
