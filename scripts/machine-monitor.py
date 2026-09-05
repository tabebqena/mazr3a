#!/usr/bin/env python3
"""machine-monitor.py - CPU temperature watchdog for the Frigate host.

Reads the hottest live CPU temperature (lm-sensors) and posts a Telegram
alert when it reaches/exceeds MAX_TEMP.

Credentials and MAX_TEMP come from config/telegram.conf (git-ignored;
template config/telegram.conf.example) - not hard-coded in this file.

Runs from the host crontab every few minutes (see scripts/crontab.sample).
Usage: python3 machine-monitor.py [--dry-run]
"""
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import collect_sensors as lib

DRY = "--dry-run" in sys.argv


def main():
    cfg = lib.load_conf()
    try:
        lib.ensure_creds(cfg)
    except RuntimeError as exc:
        print(f"[machine-monitor] ERROR: {exc}", file=sys.stderr)
        return 1

    max_temp = int(cfg.get("MAX_TEMP", 70) or 70)
    current = lib.get_cpu_temp_max()
    if current is None:
        print(
            "[machine-monitor] ERROR: no temperature reading "
            "(is lm-sensors installed?)",
            file=sys.stderr,
        )
        return 1

    if current < max_temp:
        return 0  # normal - no alert

    host = socket.gethostname() or "unknown"
    text = "\n".join(
        [
            "<b>⚠️ HIGH TEMPERATURE ALERT!</b>",
            f"<b>Server:</b> {lib.esc_html(host)}",
            f"<b>Current Temp:</b> {current}°C",
            f"<b>Max Temp Limit:</b> {max_temp}°C",
        ]
    )
    if DRY:
        print(text)
        return 0
    try:
        lib.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly under cron
        print(f"[machine-monitor] failed to send alert: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
