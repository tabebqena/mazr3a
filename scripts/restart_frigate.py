#!/usr/bin/env python3
"""restart_frigate.py - daily Frigate restart to reset the memory sawtooth.

Why: on this ~7.5 GiB host, Frigate's RSS ratchets up over days from CPython
arena fragmentation + Transparent Huge Pages (`[always]`), until its cgroup sits
at the 3 GiB `mem_limit` (`memory.events max` was 3122). A restart resets it -
measured 2.90 GiB -> 1.56 GiB after a manual restart (2026-10-01). This is the
pragmatic reset until THP is switched to `madvise` (needs root).

Runs once a day from the host crontab (scripts/crontab.sample, 04:00). It is
QUIET on success (the daily machine-status report already carries memory/health)
and sends a Telegram alert on failure, so cron can direct output to /dev/null.

Optional idle gate: when `MAX_LOADAVG` is set (env or a `MAX_LOADAVG` key in
config/telegram.conf), the run is skipped if loadavg(1m) >= that value, so a
restart never fires while the host is busy. Unset = always restart.

Usage: python3 restart_frigate.py [--dry-run]
  --dry-run  print the decision without restarting or alerting.
"""
import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import telegram_notify as tg  # noqa: E402

DRY = "--dry-run" in sys.argv

FRIGATE_DIR = os.environ.get("FRIGATE_DIR", "/home/dr/frigate")
RESTART_TIMEOUT_S = 300


def loadavg1():
    """Return the 1-minute load average, or None if unavailable."""
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def max_loadavg(cfg):
    """Return the MAX_LOADAVG gate (float) or None when unset/blank/bad."""
    raw = (os.environ.get("MAX_LOADAVG") or cfg.get("MAX_LOADAVG", "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def alert_failure(cfg, text):
    """Best-effort Telegram failure alert; never raises."""
    try:
        tg.send_telegram(cfg, text)
    except Exception as exc:  # noqa: BLE001 - alerting must not mask the failure
        print(f"[restart-frigate] failed to send alert: {exc}", file=sys.stderr)


def main():
    cfg = tg.load_conf()
    host = socket.gethostname() or "unknown"

    gate = max_loadavg(cfg)
    la = loadavg1()
    if gate is not None and la is not None and la >= gate:
        print(f"[restart-frigate] loadavg1 {la} >= {gate} - skipped (busy)")
        return 0

    cmd = ["docker", "compose", "restart", "frigate"]
    if DRY:
        print(f"[restart-frigate] would run: {' '.join(cmd)} (cwd {FRIGATE_DIR}); "
              f"loadavg1={la} gate={gate}")
        return 0

    try:
        proc = subprocess.run(
            cmd, cwd=FRIGATE_DIR, capture_output=True, text=True,
            timeout=RESTART_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - timeout/missing docker/etc.
        alert_failure(
            cfg,
            "<b>🔴 Frigate daily restart FAILED</b>\n"
            f"<b>Server:</b> {tg.esc_html(host)}\n"
            f"<b>Error:</b> {tg.esc_html(exc)}",
        )
        print(f"[restart-frigate] restart failed: {exc}", file=sys.stderr)
        return 1

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = " | ".join(tail[-3:]) if tail else f"exit {proc.returncode}"
        alert_failure(
            cfg,
            "<b>🔴 Frigate daily restart FAILED</b>\n"
            f"<b>Server:</b> {tg.esc_html(host)}\n"
            f"<b>Exit:</b> {proc.returncode}\n"
            f"<b>Detail:</b> {tg.esc_html(detail)}",
        )
        print(f"[restart-frigate] restart failed (exit {proc.returncode}): {detail}",
              file=sys.stderr)
        return 1

    print(f"[restart-frigate] frigate restarted (loadavg1={la} gate={gate})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
