#!/usr/bin/env python3
"""Shared helpers for the host monitoring scripts.

Provides:
  - load_conf():      read the KEY=VALUE config (config/telegram.conf) holding
                      the Telegram credentials + tunables (git-ignored;
                      template config/telegram.conf.example).
  - ensure_creds():   validate Telegram credentials are present.
  - esc_html():       HTML-escape a dynamic value for parse_mode=html.
  - send_telegram():  post a message to the configured chat/group.
  - get_cpu_temp_max(): hottest live CPU temperature from lm-sensors.

Default conf path: <scripts>/../config/telegram.conf (override with the
TELEGRAM_CONF environment variable).
"""
import html
import json
import os
import re
import subprocess
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


_SENSOR_KEYS = ("Core", "Package", "Tctl", "Tdie", "Tccd")
_TEMP_RE = re.compile(r"[+-]?\d+(?:\.\d+)?°C")


def get_cpu_temp_max():
    """Hottest live CPU temperature in °C (int) or None if unavailable.

    lm-sensors prints the live reading and the high/crit thresholds on the
    same line, so we keep only the FIRST °C value on each sensor line (the
    live reading) and take the maximum across cores/packages.
    """
    try:
        result = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=15
        )
        output = result.stdout
    except Exception:
        return None
    best = None
    for line in output.splitlines():
        if not any(key in line for key in _SENSOR_KEYS):
            continue
        match = _TEMP_RE.search(line)
        if not match:
            continue
        try:
            value = float(match.group(0).rstrip("°C").lstrip("+"))
        except ValueError:
            continue
        if best is None or value > best:
            best = value
    return int(round(best)) if best is not None else None
