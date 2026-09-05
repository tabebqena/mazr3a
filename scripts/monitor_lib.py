#!/usr/bin/env python3
"""Shared helpers for the host monitoring scripts.

Provides:
  - load_conf():      read the KEY=VALUE config (config/telegram.conf) holding
                      the Telegram credentials + tunables (git-ignored;
                      template config/telegram.conf.example).
  - ensure_creds():   validate Telegram credentials are present.
  - esc_html():       HTML-escape a dynamic value for parse_mode=html.
  - send_telegram():  post a message to the configured chat/group.
  - run_sensors():    run `sensors` and return its stdout lines (live).
  - find_value():     extract one whitespace-separated field from sensor lines.
  - collect():        parse `sensors` output into ordered (key, value) pairs.
  - get_cpu_temp_max(): hottest live CPU temperature in °C (read on demand).

The sensor helpers are cache-free: they read live data from lm-sensors on
every call and never read/write any cache file. scripts/collect_sensors.py (a
separate daemon) imports run_sensors()/collect() from this module.

Default conf path: <scripts>/../config/telegram.conf (override with the
TELEGRAM_CONF environment variable).
"""
import html
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

LIB_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONF = os.path.join(LIB_DIR, "..", "config", "telegram.conf")


def load_conf(path=None):
    """Parse a KEY=VALUE config file (skipping blank lines and # comments)."""
    path = path or os.environ.get("TELEGRAM_CONF") or DEFAULT_CONF
    cfg = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    cfg[key.strip()] = value.strip()
    except FileNotFoundError:
        return cfg
    return cfg


def ensure_creds(cfg):
    """Return (token, chat_id) or raise RuntimeError with a clear message."""
    token = cfg.get("BOT_TOKEN", "").strip()
    chat = cfg.get("CHAT_ID", "").strip()
    if not token or not chat or token == "CHANGE_ME":
        raise RuntimeError(
            f"Telegram not configured - check {DEFAULT_CONF} "
            "(create it from telegram.conf.example)"
        )
    return token, chat


def esc_html(text):
    """HTML-escape &, < and > so dynamic values are safe in html parse mode."""
    return html.escape(str(text), quote=False)


def send_telegram(cfg, text, parse_mode="html"):
    """Post `text` to the configured chat. Raises on transport/API errors."""
    token, chat = ensure_creds(cfg)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat, "text": text, "parse_mode": parse_mode}
    ).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.load(response)
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body.get('description')}")


def send_telegram_photo(cfg, photo_bytes, caption="", parse_mode="html"):
    """Post a photo (JPEG bytes) with an optional caption to the configured
    chat via the Bot API sendPhoto method (multipart/form-data, stdlib only).

    Used by the fire-watch watcher (scripts/firewatch.py) to send the alert
    snapshot together with the detection caption.
    """
    token, chat = ensure_creds(cfg)
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    boundary = f"----firewatch{int(__import__('time').time() * 1000)}"
    fields = [
        ("chat_id", str(chat)),
        ("caption", caption),
        ("parse_mode", parse_mode),
    ]
    body = bytearray()
    for name, value in fields:
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")
    body += (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="photo"; '
        'filename="snapshot.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    body += bytes(photo_bytes)
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result.get('description')}")


def run_sensors():
    """Run `sensors` and return its stdout as a list of lines.

    Live, on-demand read - no cache file is read or written. On failure it
    reports to stderr and returns [] so callers can fall back gracefully.
    """
    try:
        result = subprocess.run(
            ["sensors"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.stderr.write("sensors: command not found\n")
        return []
    if result.returncode != 0:
        sys.stderr.write(f"sensors failed (rc={result.returncode}): "
                         f"{result.stderr.strip()}\n")
        return []
    return result.stdout.splitlines()


def find_value(lines, label, field_index, section=None, context_lines=0):
    """Return the field at ``field_index`` (0-based) of the first line that
    contains ``label``.

    Mirrors the shell pipeline::

        grep 'label' | awk '{print $N}'           # field_index = N - 1

    When ``section`` is given, only lines within ``context_lines`` lines after
    a line containing ``section`` are considered, mirroring::

        grep -A N 'section' | grep 'label' | awk '{print $N}'
    """
    if section is not None:
        # Collect the section line plus the N lines that follow it.
        region = []
        for i, line in enumerate(lines):
            if section in line:
                region.extend(lines[i:i + context_lines + 1])
    else:
        region = lines

    for line in region:
        if label in line:
            fields = line.split()
            if len(fields) > field_index:
                return fields[field_index]
    return None


def collect(lines):
    """Parse ``sensors`` output into an ordered list of (key, value).

    Cache-free: parses whatever lines it is given (usually run_sensors()).
    Reused by collect_sensors.py (a separate daemon) and get_cpu_temp_max().
    """
    return [
        ("CPU_PACK", find_value(lines, "Package id 0", 3)),
        ("CORE_0",   find_value(lines, "Core 0", 2)),
        ("CORE_1",   find_value(lines, "Core 1", 2)),
        ("CORE_2",   find_value(lines, "Core 2", 2)),
        ("CORE_3",   find_value(lines, "Core 3", 2)),
        ("CORE_4",   find_value(lines, "Core 4", 2)),
        ("CORE_5",   find_value(lines, "Core 5", 2)),
        # Values sourced from the alienware_wmi block.
        ("GPU_TEMP", find_value(lines, "GPU:", 1,
                                section="alienware_wmi", context_lines=5)),
        ("CPU_FAN",  find_value(lines, "CPU Fan:", 2,
                                section="alienware_wmi", context_lines=5)),
        ("GPU_FAN",  find_value(lines, "GPU Fan:", 2,
                                section="alienware_wmi", context_lines=5)),
        # NVMe SSD temperature (Composite) from the nvme block.
        ("NVME_SSD", find_value(lines, "Composite", 1,
                                section="nvme-pci", context_lines=2)),
        # RAM/SODIMM temperature from the dell_ddv block.
        ("RAM_TEMP", find_value(lines, "SODIMM:", 1,
                                section="dell_ddv", context_lines=10)),
    ]


def get_cpu_temp_max():
    """Hottest live CPU temperature in °C (int) or None if unavailable.

    Reads live sensors on demand via run_sensors()/collect() - no cache file
    involved. Only the CPU fields (CPU_PACK, CORE_*) are considered; the first
    °C value on each sensor line is the live reading (high/crit are not).
    """
    try:
        values = collect(run_sensors())
    except Exception:
        return None
    temps = []
    for key, value in values:
        if not (key == "CPU_PACK" or key.startswith("CORE_")):
            continue
        if value is None:
            continue
        try:
            temps.append(float(value.replace("°C", "").lstrip("+")))
        except ValueError:
            continue
    return int(round(max(temps))) if temps else None
