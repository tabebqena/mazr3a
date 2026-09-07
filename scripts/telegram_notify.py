#!/usr/bin/env python3
"""telegram_notify.py - reusable Telegram alerting helpers for the host scripts.

Consolidated home for everything that talks to the Telegram Bot API or reads the
shared git-ignored credentials config, so the other host scripts do not each
re-implement it:

  - load_conf():         read a KEY=VALUE config file (default config/telegram.conf)
                         holding the Telegram credentials + tunables (git-ignored;
                         template config/telegram.conf.example). Override the path
                         with the TELEGRAM_CONF environment variable.
  - ensure_creds():      validate Telegram credentials are present; return
                         (token, chat_id) or raise RuntimeError.
  - esc_html():          HTML-escape a dynamic value for parse_mode=html.
  - send_telegram():     post a message to the configured chat/group.
  - send_telegram_photo(): post a photo (JPEG) with an optional caption.

Consumers (all stdlib-only, no third-party deps):
  - scripts/machine-monitor.py   debounced high-CPU-temperature alert
  - scripts/machine-status.py    daily machine health report
  - scripts/firewatch.py         fire/smoke alert photos (firewatch container
                                 mounts ./scripts read-only at /scripts:ro, so
                                 this module ships with the script)
"""
import html
import json
import os
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
