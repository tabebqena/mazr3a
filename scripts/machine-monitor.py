#!/usr/bin/env python3
"""machine-monitor.py - CPU temperature watchdog for the Frigate host.

Reads the hottest live CPU temperature (lm-sensors) and posts a Telegram alert
only when the temperature stays at/above MAX_TEMP across MIN_HITS separate
samples taken at least MIN_INTERVAL minutes apart. A single transient spike
resets the counter and does NOT alert - the temp has to be sustained over
roughly (MIN_HITS - 1) * MIN_INTERVAL minutes (defaults: 75°C, 3 hits,
30 min -> ~1 hour).

Because the host cron runs this script every few minutes
(scripts/crontab.sample, every 10), the debounce state - how many spaced
above-threshold samples have been seen - must survive between runs, so it is
kept in a small JSON state file. Default: <scripts>/../machine-monitor.state
(override the path with the MONITOR_STATE env var). The state file is
git-ignored host runtime state and is never deployed.

Credentials and tunables (MAX_TEMP, MIN_HITS, MIN_INTERVAL) come from
config/telegram.conf (git-ignored; template config/telegram.conf.example). Each
key falls back to a baked-in default here, so the script is safe to run before
the config exists. Telegram helpers live in scripts/telegram_notify.py (reused
by machine-status.py and firewatch.py); lm-sensors parsing lives in
scripts/collect_sensors.py.

Usage: python3 machine-monitor.py [--dry-run]
  --dry-run  print what this run would do / the alert text, without persisting
             state or sending anything.
"""
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import collect_sensors as sensors  # noqa: E402
import telegram_notify as tg       # noqa: E402

DRY = "--dry-run" in sys.argv

# Fallback defaults, used only when the key is missing/blank in telegram.conf.
DEFAULT_MAX_TEMP = 75
DEFAULT_MIN_HITS = 3
DEFAULT_MIN_INTERVAL_MIN = 30  # minutes between qualifying samples

STATE_FILE = os.environ.get(
    "MONITOR_STATE",
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "machine-monitor.state"
    ),
)


def _cfg_int(cfg, key, default):
    """Return the integer value of cfg[key], or `default` if missing/blank/bad."""
    try:
        return int(str(cfg.get(key, "")).strip() or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# cross-run debounce state (persisted between cron invocations)
# ---------------------------------------------------------------------------
def load_state(path=STATE_FILE):
    """Read the persisted {"hits": n, "last_hit_ts": epoch} dict.

    Best-effort: an unreadable/corrupt/missing file starts a fresh (empty) state
    so a state hiccup never blocks alerting.
    """
    default = {"hits": 0, "last_hit_ts": 0.0}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {
            "hits": max(0, int(data.get("hits", 0))),
            "last_hit_ts": float(data.get("last_hit_ts", 0.0)),
        }
    except (OSError, ValueError, TypeError):
        return dict(default)


def save_state(path, state):
    """Persist {"hits": n, "last_hit_ts": epoch}. Best-effort (never crashes)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        return True
    except OSError as exc:
        print(f"[machine-monitor] WARNING: cannot write state {path}: {exc}",
              file=sys.stderr)
        return False


def evaluate(current, max_temp, min_hits, min_interval_min, state, now=None):
    """Apply one temperature sample to the debounce state.

    Pure decision helper (no I/O - unit-testable). Returns a dict:

      {
        "alert":  bool,   # true -> sustained high temp, send now
        "hits":   int,    # spaced high-temp samples counted so far (this run)
        "waited": bool,   # true -> hot but too soon since last counted sample
        "next_ts": float | None,  # when the next sample may be counted
        "state":  {"hits": int, "last_hit_ts": float},  # new persisted state
      }

    Rules (see module docstring):
      * below MAX_TEMP            -> counter reset, no alert.
      * at/above MAX_TEMP and >= MIN_INTERVAL minutes since the last counted
        sample -> count it; on the MIN_HITS-th counted sample -> ALERT + reset.
      * at/above MAX_TEMP but < MIN_INTERVAL minutes since the last counted
        sample -> wait: no count and no state change, so a cron that fires more
        often than MIN_INTERVAL still yields samples exactly MIN_INTERVAL apart.
    """
    now = time.time() if now is None else now
    hits = int(state.get("hits", 0))
    last_hit_ts = float(state.get("last_hit_ts", 0.0))
    interval_s = max(0, int(min_interval_min)) * 60
    min_hits = max(1, int(min_hits))

    if current is None or current < max_temp:
        return {
            "alert": False,
            "hits": 0,
            "waited": False,
            "next_ts": None,
            "state": {"hits": 0, "last_hit_ts": last_hit_ts},
        }

    if hits > 0 and (now - last_hit_ts) < interval_s:
        return {
            "alert": False,
            "hits": hits,
            "waited": True,
            "next_ts": last_hit_ts + interval_s,
            "state": {"hits": hits, "last_hit_ts": last_hit_ts},
        }

    hits += 1
    alert = hits >= min_hits
    return {
        "alert": alert,
        "hits": hits,
        "waited": False,
        "next_ts": None,
        "state": {"hits": 0 if alert else hits, "last_hit_ts": now},
    }


# ---------------------------------------------------------------------------
# alert rendering + main flow
# ---------------------------------------------------------------------------
def alert_text(host, current, max_temp, min_hits, min_interval_min):
    return "\n".join(
        [
            "<b>⚠️ HIGH TEMPERATURE ALERT!</b>",
            f"<b>Server:</b> {tg.esc_html(host)}",
            f"<b>Current Temp:</b> {current}°C",
            f"<b>Max Temp Limit:</b> {max_temp}°C",
            f"<b>Confirmed:</b> {min_hits} samples ≥{min_interval_min} min apart",
        ]
    )


def main():
    cfg = tg.load_conf()
    try:
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        print(f"[machine-monitor] ERROR: {exc}", file=sys.stderr)
        return 1

    max_temp = _cfg_int(cfg, "MAX_TEMP", DEFAULT_MAX_TEMP)
    min_hits = _cfg_int(cfg, "MIN_HITS", DEFAULT_MIN_HITS)
    min_interval_min = _cfg_int(cfg, "MIN_INTERVAL", DEFAULT_MIN_INTERVAL_MIN)

    current = sensors.get_cpu_temp_max()
    if current is None:
        print(
            "[machine-monitor] ERROR: no temperature reading "
            "(is lm-sensors installed?)",
            file=sys.stderr,
        )
        return 1

    result = evaluate(current, max_temp, min_hits, min_interval_min, load_state())
    if not DRY:
        save_state(STATE_FILE, result["state"])

    if result["waited"]:
        mins_left = max(
            0, int((result["next_ts"] - time.time()) // 60) + 1
        )
        print(
            f"[machine-monitor] {current}°C ≥ {max_temp}°C - "
            f"confirmation sample {result['hits']}/{min_hits} recorded; "
            f"next sample in ~{mins_left} min; no alert yet"
        )
        return 0

    if not result["alert"]:
        if current < max_temp:
            print(f"[machine-monitor] {current}°C < {max_temp}°C - normal")
        else:
            print(
                f"[machine-monitor] {current}°C ≥ {max_temp}°C - "
                f"high-temp sample {result['hits']}/{min_hits} "
                f"(spacing {min_interval_min} min); no alert yet"
            )
        return 0

    host = socket.gethostname() or "unknown"
    text = alert_text(host, current, max_temp, min_hits, min_interval_min)
    if DRY:
        print(text)
        return 0
    try:
        tg.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly under cron
        print(f"[machine-monitor] failed to send alert: {exc}", file=sys.stderr)
        return 1
    print(
        f"[machine-monitor] ALERT sent: {current}°C ≥ {max_temp}°C "
        f"after {min_hits} spaced samples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
