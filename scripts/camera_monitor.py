#!/usr/bin/env python3
"""camera_monitor.py - per-camera online/offline watchdog for the Frigate host.

Replaces guessing at a dead camera from ffmpeg log spam with a debounced
Telegram alert. It reads Frigate's own `/api/stats` and marks a camera offline
when `cameras.<name>.camera_fps == 0` (no decoded frames) - the exact signal the
daily `machine-status.py` report uses. That makes it catch a dead camera **or** a
broken go2rtc restream path (Frigate's source since 2026-10-02), i.e. whatever
Frigate actually sees, unlike a raw `ping`.

Debounce: a camera must be offline for OFFLINE_HITS consecutive runs (default 3)
before the first alert, and online for ONLINE_HITS consecutive runs (default 2)
before a recovery notice, so a single dropped frame or the daily Frigate restart
(<=~30 s) never spams. Alerts fire on state CHANGES only, so a camera that stays
offline alerts once (the daily report carries the ongoing state).

Two outage classes are reported distinctly:
  * one/few cameras offline -> a camera problem;
  * ALL cameras offline     -> a stack problem (Frigate/go2rtc/network), so the
    operator is not misled into chasing a single camera.
A consecutive `/api/stats` failure (Frigate itself unreachable) is also alerted.

Runs every couple of minutes from the host crontab (scripts/crontab.sample).
State persists in a small JSON file next to this script (git-ignored host
runtime state; override with CAMERA_STATE). Reuses scripts/telegram_notify.py.

Usage: python3 camera_monitor.py [--dry-run]
  --dry-run  print the decision/alert text without sending or saving state.
"""
import json
import os
import socket
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import telegram_notify as tg  # noqa: E402

DRY = "--dry-run" in sys.argv

FRIGATE_API = os.environ.get("FRIGATE_API", "http://127.0.0.1:5000")

# Fallback defaults, used only when the key is missing/blank in telegram.conf.
DEFAULT_OFFLINE_HITS = 3   # consecutive zero-fps runs before an offline alert
DEFAULT_ONLINE_HITS = 2    # consecutive good runs before a recovery notice

STATE_FILE = os.environ.get(
    "CAMERA_STATE",
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "camera-monitor.state"
    ),
)


def _cfg_int(cfg, key, default):
    """Return the integer value of cfg[key], or `default` if missing/blank/bad."""
    try:
        return int(str(cfg.get(key, "")).strip() or default)
    except ValueError:
        return default


def fetch_camera_fps(timeout=5):
    """Return {camera_name: camera_fps float} from Frigate /api/stats.

    Raises on any transport/decode failure so the caller can count Frigate API
    outages (a whole-stack condition) instead of misreading them as all cameras
    being offline.
    """
    with urllib.request.urlopen(f"{FRIGATE_API}/api/stats", timeout=timeout) as resp:
        data = json.load(resp)
    cams = {}
    for name, cam in (data.get("cameras") or {}).items():
        try:
            cams[name] = float(cam.get("camera_fps") or 0)
        except (TypeError, ValueError):
            cams[name] = 0.0
    return cams


def load_state(path=STATE_FILE):
    """Read the persisted watchdog state; a bad/missing file starts fresh."""
    default = {"cams": {}, "api_fail": 0, "api_alerted": False}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        cams = data.get("cams", {})
        if not isinstance(cams, dict):
            cams = {}
        return {
            "cams": cams,
            "api_fail": int(data.get("api_fail", 0) or 0),
            "api_alerted": bool(data.get("api_alerted", False)),
        }
    except (OSError, ValueError, TypeError):
        return dict(default)


def save_state(path, state):
    """Persist the state; best-effort (never crashes the watchdog)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        return True
    except OSError as exc:
        print(f"[camera-monitor] WARNING: cannot write state {path}: {exc}",
              file=sys.stderr)
        return False


def evaluate(cameras, state, offline_hits, online_hits):
    """Update per-camera counters from one sample and report transitions.

    Pure decision helper (no I/O - unit-testable). `cameras` is
    {name: fps}; `state` is the persisted dict. Returns:

      {
        "offline":   [names newly crossing into offline],
        "recovered": [names newly crossing back online],
        "all_offline": bool,   # every camera at fps 0 in THIS sample
        "cams_known": int,
        "state": {...},        # new persisted state (cams only)
      }
    """
    cams = {}
    for name, counters in (state.get("cams") or {}).items():
        if isinstance(counters, dict):
            cams[name] = {
                "oc": int(counters.get("oc", 0) or 0),
                "nc": int(counters.get("nc", 0) or 0),
                "alerted": bool(counters.get("alerted", False)),
            }

    newly_offline = []
    recovered = []
    for name, fps in cameras.items():
        c = cams.setdefault(name, {"oc": 0, "nc": 0, "alerted": False})
        if fps > 0:
            c["nc"] += 1
            c["oc"] = 0
            if c["alerted"] and c["nc"] >= online_hits:
                recovered.append(name)
                c["alerted"] = False
                c["nc"] = 0
        else:
            c["oc"] += 1
            c["nc"] = 0
            if not c["alerted"] and c["oc"] >= offline_hits:
                newly_offline.append(name)
                c["alerted"] = True

    return {
        "offline": sorted(newly_offline),
        "recovered": sorted(recovered),
        "all_offline": bool(cameras) and all(fps <= 0 for fps in cameras.values()),
        "cams_known": len(cameras),
        "state": {"cams": cams},
    }


def evaluate_api_failure(state, offline_hits):
    """Count a Frigate-API failure; alert once when it crosses offline_hits.

    Returns (alert_now, new_api_state) where new_api_state is
    {"api_fail": int, "api_alerted": bool}.
    """
    fail = int(state.get("api_fail", 0) or 0) + 1
    alerted = bool(state.get("api_alerted", False))
    alert_now = False
    if fail >= offline_hits and not alerted:
        alert_now = True
        alerted = True
    return alert_now, {"api_fail": fail, "api_alerted": alerted}


def offline_text(host, names, all_offline):
    if all_offline:
        head = "<b>🔴 ALL CAMERAS OFFLINE</b>"
        detail = ("No camera is producing frames - this is a STACK issue "
                  "(Frigate / go2rtc / network), not a single camera.")
    else:
        head = "<b>📵 Camera offline</b>"
        detail = "No frames for the debounce window (Frigate /api/stats)."
    lines = [head, f"<b>Server:</b> {tg.esc_html(host)}",
             f"<b>Detail:</b> {detail}",
             "<b>Cameras:</b> " + ", ".join(tg.esc_html(n) for n in names)]
    return "\n".join(lines)


def recovery_text(host, names):
    lines = [
        "<b>✅ Camera recovered</b>",
        f"<b>Server:</b> {tg.esc_html(host)}",
        "<b>Cameras:</b> " + ", ".join(tg.esc_html(n) for n in names),
    ]
    return "\n".join(lines)


def api_down_text(host, fail_count):
    lines = [
        "<b>🔴 Frigate API unreachable</b>",
        f"<b>Server:</b> {tg.esc_html(host)}",
        f"<b>Detail:</b> /api/stats failed {fail_count} consecutive runs - "
        "Frigate itself is down or not answering.",
    ]
    return "\n".join(lines)


def main():
    cfg = tg.load_conf()
    try:
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        print(f"[camera-monitor] ERROR: {exc}", file=sys.stderr)
        return 1

    offline_hits = _cfg_int(cfg, "CAMERA_OFFLINE_HITS", DEFAULT_OFFLINE_HITS)
    online_hits = _cfg_int(cfg, "CAMERA_ONLINE_HITS", DEFAULT_ONLINE_HITS)
    host = socket.gethostname() or "unknown"

    state = load_state()
    try:
        cameras = fetch_camera_fps()
    except Exception as exc:  # noqa: BLE001 - classify as an API outage
        alert_now, api_state = evaluate_api_failure(state, offline_hits)
        state.update(api_state)
        if not DRY:
            save_state(STATE_FILE, state)
        text = api_down_text(host, api_state["api_fail"])
        if alert_now:
            if DRY:
                print(text)
                return 0
            try:
                tg.send_telegram(cfg, text)
            except Exception as send_exc:  # noqa: BLE001
                print(f"[camera-monitor] failed to send alert: {send_exc}",
                      file=sys.stderr)
                return 1
            print(f"[camera-monitor] API-down alert sent (fail x{api_state['api_fail']})")
            return 0
        print(f"[camera-monitor] /api/stats unreachable ({exc}); "
              f"fail {api_state['api_fail']}/{offline_hits} - no alert yet")
        return 0

    # API is back: clear the outage (and note recovery if we had alerted).
    was_api_alerted = bool(state.get("api_alerted", False))
    state["api_fail"] = 0
    state["api_alerted"] = False

    result = evaluate(cameras, state, offline_hits, online_hits)
    state["cams"] = result["state"]["cams"]
    if not DRY:
        save_state(STATE_FILE, state)

    texts = []
    if was_api_alerted:
        texts.append("<b>✅ Frigate API recovered</b>\n"
                     f"<b>Server:</b> {tg.esc_html(host)}")
    if result["offline"]:
        texts.append(offline_text(host, result["offline"], result["all_offline"]))
    if result["recovered"]:
        texts.append(recovery_text(host, result["recovered"]))

    if not texts:
        off = [n for n, fps in cameras.items() if fps <= 0]
        print(f"[camera-monitor] {len(cameras)} cameras; "
              f"offline now: {', '.join(off) if off else 'none'} - no change")
        return 0

    body = "\n\n".join(texts)
    if DRY:
        print(body)
        return 0
    try:
        tg.send_telegram(cfg, body)
    except Exception as exc:  # noqa: BLE001 - fail loudly under cron
        print(f"[camera-monitor] failed to send alert: {exc}", file=sys.stderr)
        return 1
    print(f"[camera-monitor] alert sent: offline={result['offline']} "
          f"recovered={result['recovered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
