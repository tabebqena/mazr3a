#!/usr/bin/env python3
"""machine-status.py - daily machine health report for the Frigate host,
posted to the Telegram group configured in config/telegram.conf
(git-ignored; template config/telegram.conf.example).

Collects (best-effort; a failing probe degrades to n/a):
  - host, date/time, uptime
  - CPU temperature (max) + load average (1/5/15 min)
  - memory usage (from /proc/meminfo)
  - root disk usage
  - Frigate media dir size / filesystem usage
  - Docker/Frigate stack state (docker compose ps)
  - Frigate detection summary + per-camera online/offline from /api/stats

Runs once a day from the host crontab (see scripts/crontab.sample).
Usage: python3 machine-status.py [--dry-run]
"""
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import collect_sensors as sensors  # noqa: E402
import telegram_notify as tg       # noqa: E402

DRY = "--dry-run" in sys.argv

FRIGATE_DIR = os.environ.get("FRIGATE_DIR", "/home/dr/frigate")
MEDIA_DIR = os.environ.get("MEDIA_DIR", os.path.join(FRIGATE_DIR, "media"))
FRIGATE_API = os.environ.get("FRIGATE_API", "http://127.0.0.1:5000")


def _run(cmd, cwd=None, timeout=15):
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    except Exception:
        return ""


def format_uptime():
    try:
        with open("/proc/uptime", encoding="utf-8") as fh:
            seconds = float(fh.read().split()[0])
    except Exception:
        return "n/a"
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts = []
    if days:
        parts.append(f"{days} days" if days != 1 else "1 day")
    if hours:
        parts.append(f"{hours} hours" if hours != 1 else "1 hour")
    parts.append(f"{minutes} minutes" if minutes != 1 else "1 minute")
    return ", ".join(parts)


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit != "B" else f"{value:.0f}B"
        value /= 1024.0
    return f"{value:.0f}P"


def load_summary():
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            vals = fh.read().split()
        return f"{vals[0]} / {vals[1]} / {vals[2]}"
    except Exception:
        return "n/a"


def memory_summary():
    total_kb = available_kb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
    except Exception:
        return "n/a"
    if not total_kb:
        return "n/a"
    used_kb = total_kb - (available_kb or 0)
    return (
        f"{used_kb / 1048576:.1f} / {total_kb / 1048576:.1f} GiB "
        f"({used_kb * 100 / total_kb:.0f}%)"
    )


def disk_summary(path):
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return "n/a"
    pct = usage.used * 100 / usage.total if usage.total else 0
    return (
        f"{human_size(usage.used)} used of {human_size(usage.total)}, "
        f"{human_size(usage.free)} free ({pct:.0f}%)"
    )


def media_dir_summary():
    if not os.path.isdir(MEDIA_DIR):
        return f"missing ({MEDIA_DIR})"
    du = _run(["du", "-sh", MEDIA_DIR], timeout=60).split(maxsplit=1)
    size = du[0] if du else "n/a"
    try:
        usage = shutil.disk_usage(MEDIA_DIR)
        pct = f"{usage.used * 100 / usage.total:.0f}%"
    except Exception:
        pct = "n/a"
    return f"{size} ({pct} of filesystem used)"


def docker_stack_lines():
    if not os.path.isfile(os.path.join(FRIGATE_DIR, "docker-compose.yml")):
        return ["🐳 Stack: docker or compose file not available"]
    services = [
        line
        for line in _run(
            [
                "docker",
                "compose",
                "ps",
                "--format",
                "{{.Service}}: {{.State}}",
            ],
            cwd=FRIGATE_DIR,
            timeout=30,
        ).splitlines()
        if line.strip()
    ]
    if not services:
        return ["🐳 Stack: no services running"]
    lines = ["🐳 Stack:"]
    lines += [f"   • {tg.esc_html(s)}" for s in services]
    return lines


def frigate_lines():
    try:
        with urllib.request.urlopen(f"{FRIGATE_API}/api/stats", timeout=5) as resp:
            data = json.load(resp)
    except Exception:
        return []
    cams = data.get("cameras", {})
    detection = float(data.get("detection_fps") or 0)
    inference = []
    for value in data.get("detectors", {}).values():
        if isinstance(value, dict) and value.get("inference_speed"):
            inference.append(int(value["inference_speed"]))
    det_txt = f"detection {detection:.1f} fps"
    if inference:
        det_txt += "; inference {} ms".format(
            " / ".join(str(i) for i in inference)
        )
    lines = [f"🎞️ Frigate: {det_txt}"]
    if cams:
        lines.append("📷 Cameras:")
        for name in sorted(cams):
            fps = cams[name].get("camera_fps") or 0
            label = f"✅ online ({fps:.1f} fps)" if fps > 0 else "❌ offline"
            lines.append(f"   • {tg.esc_html(name)} {label}")
    return lines


def main():
    cfg = tg.load_conf()
    try:
        tg.ensure_creds(cfg)
    except RuntimeError as exc:
        print(f"[machine-status] ERROR: {exc}", file=sys.stderr)
        return 1

    report = []
    add = report.append

    host = socket.gethostname() or "unknown"
    add(f"<b>🖥️ {tg.esc_html(host)} - daily report</b>")
    now = datetime.datetime.now().astimezone()
    add(f"📅 {now.strftime('%Y-%m-%d %H:%M %Z')}")

    add(f"⏱ Uptime: {tg.esc_html(format_uptime())}")

    temp = sensors.get_cpu_temp_max()
    if temp is not None:
        add(f"🌡 CPU temp (max): {temp}°C")
    else:
        add("🌡 CPU temp (max): n/a")

    add(f"⚙️ Load 1/5/15m: {tg.esc_html(load_summary())}")
    add(f"🧠 Memory used: {tg.esc_html(memory_summary())}")
    add(f"💾 Disk /: {tg.esc_html(disk_summary('/'))}")
    add(f"🎥 Media dir: {tg.esc_html(media_dir_summary())}")

    report.extend(docker_stack_lines())
    report.extend(frigate_lines())

    text = "\n".join(report)
    if DRY:
        print(text)
        return 0
    try:
        tg.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly under cron
        print(f"[machine-status] failed to send report: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
