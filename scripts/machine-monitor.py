#!/usr/bin/env python3
"""machine-monitor.py - CPU temperature watchdog for the Frigate host.

Reads the hottest live CPU temperature (lm-sensors) and posts a Telegram alert
using TWO tiers:

1) CRITICAL tier - any SINGLE sample at/above CRITICAL_TEMP (default 80 degC)
   alerts immediately. This catches the brief single-minute spikes this host
   shows (up to 83 degC, caused by periodic heavy Frigate bursts) that a slower
   debounce would never see.

2) WARM tier - a sample at/above MAX_TEMP (default 72 degC) is remembered; when
   MIN_HITS (default 3) such samples have occurred within the last
   MIN_WINDOW_MIN (default 240) minutes, an alert is sent. Samples are NOT
   "reset" by a cool reading - they simply age out of the window, so a day of
   repeated 72-79 degC excursions (like the hourly Frigate bursts recorded on
   2026-09-08) finally warns you instead of being silently ignored.

Why not the old "spaced samples" design: the previous logic only alerted when
the temp stayed at/above MAX_TEMP on MIN_HITS samples spaced >= MIN_INTERVAL
minutes apart AND reset the counter to zero on any cooler sample. The recorder
data (scripts/watchdog_baseline.py, cron every minute) shows this host NEVER
sustains a high plateau - it spikes to 74-83 degC for ~1 minute per heavy burst
then returns to 50-60 degC - so the old debounce could never reach MIN_HITS and
the host ran at 74-83 degC for hours with zero Telegram alerts (peak 83 degC on
2026-09-08, which moreover fell between the 10-minute cron ticks and was never
even sampled).

Alert spam is capped by COOLDOWN_MIN (default 60): after any alert the monitor
stays quiet for that long before it may alert again.

Because the host cron runs this script every minute (scripts/crontab.sample),
the recent-sample history needed by the WARM tier must survive between runs, so
it is kept in a small JSON state file. Default: <scripts>/../machine-monitor.state
(override the path with the MONITOR_STATE env var). The state file is
git-ignored host runtime state and is never deployed.

If the state file cannot be used (e.g. it does not exist and cannot be created,
or its directory is read-only), the script does not silently lose the history:
it falls back to keeping the state in memory and sampling every minute within
the SAME run until it can decide, so a real over-temp still alerts even without
a usable state file.

Credentials and tunables (MAX_TEMP, MIN_HITS, MIN_WINDOW_MIN, CRITICAL_TEMP,
COOLDOWN_MIN) come from config/telegram.conf (git-ignored; template
config/telegram.conf.example). Each key falls back to a baked-in default here,
so the script is safe to run before the config exists. Telegram helpers live in
scripts/telegram_notify.py (reused by machine-status.py and firewatch.py);
lm-sensors parsing lives in scripts/collect_sensors.py.

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
DEFAULT_MAX_TEMP = 72           # degC - WARM tier threshold
DEFAULT_MIN_HITS = 3            # WARM fires after this many >= MAX_TEMP samples...
DEFAULT_MIN_WINDOW_MIN = 240    # ...within this rolling window (minutes)
DEFAULT_CRITICAL_TEMP = 80      # degC - any single sample >= this alerts at once
DEFAULT_COOLDOWN_MIN = 60       # minutes of quiet between two alerts
IN_MEMORY_SAMPLE_S = 60         # fallback in-memory sampling period (seconds)
IN_MEMORY_MAX_S = 7200          # cap for the in-memory fallback run (seconds)

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
# cross-run history state (persisted between cron invocations)
# ---------------------------------------------------------------------------
def load_state(path=STATE_FILE):
    """Read the persisted {"hits": [epoch,...], "last_alert_ts": epoch} dict.

    Best-effort: an unreadable/corrupt/missing file (or a legacy {"hits": int,
    "last_hit_ts": ...} state from the old spaced-sample design) starts a fresh
    (empty) state so a state hiccup never blocks alerting.
    """
    default = {"hits": [], "last_alert_ts": 0.0}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        hits = data.get("hits", [])
        if not isinstance(hits, list):
            hits = []
        return {
            "hits": [float(t) for t in hits if isinstance(t, (int, float))],
            "last_alert_ts": float(data.get("last_alert_ts", 0.0)),
        }
    except (OSError, ValueError, TypeError):
        return dict(default)


def save_state(path, state):
    """Persist {"hits": [...], "last_alert_ts": epoch}. Best-effort (never crashes)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        return True
    except OSError as exc:
        print(f"[machine-monitor] WARNING: cannot write state {path}: {exc}",
              file=sys.stderr)
        return False


def state_file_usable(path=STATE_FILE):
    """True when the state file can be read (if present) and written/created.

    Best-effort probe (never raises). False means cross-run persistence is
    impossible (missing/read-only parent dir, unwritable file), so ``main()``
    runs the decision in memory instead of silently losing the history every
    cron run.
    """
    try:
        parent = os.path.dirname(path) or "."
        if os.path.exists(path):
            # open("r+") requires both read and write permission on the file.
            with open(path, "r+", encoding="utf-8"):
                pass
        else:
            if not os.path.isdir(parent):
                return False
            if not os.access(parent, os.W_OK):
                return False
        return True
    except OSError:
        return False


def evaluate(current, max_temp, min_hits, window_min, critical_temp,
             cooldown_min, state, now=None):
    """Decide alert action from one temperature sample.

    Pure decision helper (no I/O - unit-testable). Returns a dict:

      {
        "alert":         bool,  # true -> send now (CRITICAL or WARM fired)
        "critical":      bool,  # true -> fired via the single-sample CRITICAL tier
        "suppressed":    bool,  # true -> alert due but inside COOLDOWN (quiet)
        "hits_in_window": int,  # samples >= MAX_TEMP currently within the window
        "state":         {...}, # new persisted state
      }

    Rules (see module docstring):
      * sample >= CRITICAL_TEMP        -> alert immediately (single sample).
      * sample >= MAX_TEMP             -> remember it; alert once MIN_HITS such
        samples are within the last MIN_WINDOW_MIN minutes (no reset on cool
        samples - they just age out of the window).
      * a cool sample (< MAX_TEMP and < CRITICAL_TEMP) does nothing except let
        old hits age out naturally.
      * once an alert is due it is sent only if COOLDOWN_MIN has passed since
        the previous alert; otherwise it is suppressed (and the pending hits
        are cleared so they do not pile up during the quiet period).
    """
    now = time.time() if now is None else now
    window_s = max(0, int(window_min)) * 60
    cooldown_s = max(0, int(cooldown_min)) * 60
    min_hits = max(1, int(min_hits))

    hits = [float(t) for t in state.get("hits", [])
            if isinstance(t, (int, float))]
    # age out samples older than the window
    hits = [t for t in hits if now - t <= window_s]
    last_alert = float(state.get("last_alert_ts", 0.0))

    critical = current is not None and current >= critical_temp
    alert_due = False
    if current is None:
        pass  # no reading: keep history untouched (no alert, no reset)
    elif critical:
        alert_due = True
    elif current >= max_temp:
        hits.append(now)
        hits = [t for t in hits if now - t <= window_s]  # keep bounded
        alert_due = len(hits) >= min_hits

    if not alert_due:
        return {
            "alert": False,
            "critical": critical,
            "suppressed": False,
            "hits_in_window": len(hits),
            "state": {"hits": hits, "last_alert_ts": last_alert},
        }

    # An alert is due - respect the cooldown; never pile up hits while quiet.
    suppressed = (now - last_alert) < cooldown_s
    if suppressed:
        return {
            "alert": False,
            "critical": critical,
            "suppressed": True,
            "hits_in_window": len(hits),
            "state": {"hits": [], "last_alert_ts": last_alert},
        }
    return {
        "alert": True,
        "critical": critical,
        "suppressed": False,
        "hits_in_window": len(hits),
        "state": {"hits": [], "last_alert_ts": now},
    }


# ---------------------------------------------------------------------------
# alert rendering + main flow
# ---------------------------------------------------------------------------
def alert_text(host, current, critical, max_temp, min_hits, window_min,
               critical_temp):
    if critical:
        return "\n".join(
            [
                "<b>🔴 CRITICAL TEMPERATURE!</b>",
                f"<b>Server:</b> {tg.esc_html(host)}",
                f"<b>Current Temp:</b> {current}°C",
                f"<b>Critical Limit:</b> {critical_temp}°C "
                "(single-sample - immediate)",
            ]
        )
    return "\n".join(
        [
            "<b>⚠️ HIGH TEMPERATURE</b> (repeated/sustained)",
            f"<b>Server:</b> {tg.esc_html(host)}",
            f"<b>Current Temp:</b> {current}°C",
            f"<b>Warn Level:</b> ≥{max_temp}°C on {min_hits} samples "
            f"within {window_min} min",
        ]
    )


def send_alert(cfg, host, current, critical, max_temp, min_hits, window_min,
               critical_temp):
    """Send the alert text; prints and returns the exit code."""
    text = alert_text(host, current, critical, max_temp, min_hits, window_min,
                      critical_temp)
    try:
        tg.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly under cron
        print(f"[machine-monitor] failed to send alert: {exc}", file=sys.stderr)
        return 1
    kind = "CRITICAL" if critical else "WARM"
    print(
        f"[machine-monitor] {kind} alert sent: {current}°C "
        f"({'single sample ≥' + str(critical_temp) + '°C' if critical else str(min_hits) + ' samples ≥' + str(max_temp) + '°C within ' + str(window_min) + ' min'})"
    )
    return 0


def run_in_memory(current, max_temp, min_hits, window_min, critical_temp,
                  cooldown_min, cfg):
    """Self-contained decision when the state file cannot be used.

    Runs when STATE_FILE is unavailable (missing read-only dir, unwritable file):
    the recent-sample history would otherwise be lost between cron runs. Instead,
    keep it in memory and sample every IN_MEMORY_SAMPLE_S within THIS single
    process until an alert is sent, the temp drops below both thresholds with no
    pending history, or IN_MEMORY_MAX_S elapses. Returns the process exit code.
    """
    host = socket.gethostname() or "unknown"
    state = {"hits": [], "last_alert_ts": 0.0}
    deadline = time.time() + IN_MEMORY_MAX_S
    print(
        "[machine-monitor] WARNING: state file unusable - "
        f"running in-memory (sampling every {IN_MEMORY_SAMPLE_S}s, "
        f"cap {IN_MEMORY_MAX_S // 60} min)",
        file=sys.stderr,
    )
    while True:
        if current is None:
            print(
                "[machine-monitor] ERROR: no temperature reading "
                "(is lm-sensors installed?)",
                file=sys.stderr,
            )
            return 1
        result = evaluate(current, max_temp, min_hits, window_min,
                          critical_temp, cooldown_min, state)
        state = result["state"]
        if result["alert"]:
            return send_alert(cfg, host, current, result["critical"],
                              max_temp, min_hits, window_min, critical_temp)
        if result["suppressed"]:
            print(
                f"[machine-monitor] alert due ({current}°C) but inside "
                f"cooldown ({cooldown_min} min) - quiet",
                file=sys.stderr,
            )
            return 0
        if current < max_temp and current < critical_temp:
            print(f"[machine-monitor] {current}°C < {max_temp}°C - normal")
            return 0
        if time.time() > deadline:
            print(
                "[machine-monitor] in-memory cap reached with no alert - "
                "giving up this run",
                file=sys.stderr,
            )
            return 1
        print(
            f"[machine-monitor] {current}°C ≥ {max_temp}°C - "
            f"warm sample {result['hits_in_window']}/{min_hits} "
            f"in window (in-memory); no alert yet"
        )
        time.sleep(IN_MEMORY_SAMPLE_S)
        current = sensors.get_cpu_temp_max()


def main():
    cfg = tg.load_conf()
    try:
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        print(f"[machine-monitor] ERROR: {exc}", file=sys.stderr)
        return 1

    max_temp = _cfg_int(cfg, "MAX_TEMP", DEFAULT_MAX_TEMP)
    min_hits = _cfg_int(cfg, "MIN_HITS", DEFAULT_MIN_HITS)
    window_min = _cfg_int(cfg, "MIN_WINDOW_MIN", DEFAULT_MIN_WINDOW_MIN)
    critical_temp = _cfg_int(cfg, "CRITICAL_TEMP", DEFAULT_CRITICAL_TEMP)
    cooldown_min = _cfg_int(cfg, "COOLDOWN_MIN", DEFAULT_COOLDOWN_MIN)

    current = sensors.get_cpu_temp_max()
    if current is None:
        print(
            "[machine-monitor] ERROR: no temperature reading "
            "(is lm-sensors installed?)",
            file=sys.stderr,
        )
        return 1

    # STATE_FILE cannot be read/written (e.g. missing dir, read-only): the
    # normal per-run path would lose the history between cron ticks, so fall
    # back to an in-memory decision done in this one process run.
    if not DRY and not state_file_usable(STATE_FILE):
        return run_in_memory(current, max_temp, min_hits, window_min,
                             critical_temp, cooldown_min, cfg)

    state = load_state()
    result = evaluate(current, max_temp, min_hits, window_min, critical_temp,
                      cooldown_min, state)
    if not DRY:
        save_state(STATE_FILE, result["state"])

    if result["alert"]:
        host = socket.gethostname() or "unknown"
        if DRY:
            print(alert_text(host, current, result["critical"], max_temp,
                             min_hits, window_min, critical_temp))
            return 0
        return send_alert(cfg, host, current, result["critical"], max_temp,
                          min_hits, window_min, critical_temp)

    if result["suppressed"]:
        print(
            f"[machine-monitor] alert due ({current}°C) but inside cooldown "
            f"({cooldown_min} min) - quiet"
        )
        return 0

    if current >= critical_temp:
        # Should not be reached: a critical sample is an immediate alert above.
        print(f"[machine-monitor] {current}°C ≥ {critical_temp}°C (critical)")
        return 0
    if current < max_temp:
        print(f"[machine-monitor] {current}°C < {max_temp}°C - normal")
    else:
        print(
            f"[machine-monitor] {current}°C ≥ {max_temp}°C - "
            f"warm sample {result['hits_in_window']}/{min_hits} "
            f"within {window_min} min; no alert yet"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
