#!/usr/bin/env python3
"""Shared helpers for the host monitoring scripts.

Provides:
  - load_conf():      read the KEY=VALUE config (config/telegram.conf) holding
                      the Telegram credentials + tunables (git-ignored;
                      template config/telegram.conf.example).
  - ensure_creds():   validate Telegram credentials are present.
  - esc_html():       HTML-escape a dynamic value for parse_mode=html.
  - send_telegram():  post a message to the configured chat/group.
  - get_cpu_temp_max(): hottest live CPU temperature from lm-sensors
                      (parsing delegated to collect_sensors.py).

Default conf path: <scripts>/../config/telegram.conf (override with the
TELEGRAM_CONF environment variable).
"""
import html
import json
import os
import sys
import urllib.parse
import urllib.request

LIB_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONF = os.path.join(LIB_DIR, "..", "config", "telegram.conf")

# Make the sibling collect_sensors.py importable however monitor_lib itself is
# loaded, and reuse its canonical lm-sensors parser (single source of truth
# with the conky cache writer scripts/collect_sensors.py).
sys.path.insert(0, LIB_DIR)
import collect_sensors  # noqa: E402


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


def get_cpu_temp_max():
    """Hottest live CPU temperature in °C (int) or None if unavailable.

    Delegates to collect_sensors.run_sensors()/collect() - the same parser
    that feeds the conky cache - so monitoring alerts and the dashboard agree.
    Only the CPU fields (CPU_PACK, CORE_*) are considered; the first °C value
    on each sensor line is the live reading (high/crit are not).
    """
    try:
        values = collect_sensors.collect(collect_sensors.run_sensors())
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
