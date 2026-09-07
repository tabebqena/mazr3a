#!/usr/bin/env python3
"""telegram_notify.py - reusable Telegram alerting helpers for the host scripts.

Consolidated home for everything that talks to the Telegram Bot API or reads the
shared git-ignored credentials config, so the other host scripts do not each
re-implement it:

  - load_conf():          read a KEY=VALUE config file (default config/telegram.conf)
                          holding the Telegram credentials + tunables (git-ignored;
                          template config/telegram.conf.example). Override the path
                          with the TELEGRAM_CONF environment variable.
  - ensure_creds():       validate Telegram credentials are present; return
                          (token, chat_ids) or raise RuntimeError. chat_ids is the
                          list of recipient chats parsed from CHAT_ID.
  - esc_html():           HTML-escape a dynamic value for parse_mode=html.
  - send_telegram():      post a message to EVERY configured chat/group.
  - send_telegram_photo(): post a photo (JPEG) with an optional caption to EVERY
                          configured chat/group.

Every notification is delivered to ALL recipients in CHAT_ID (comma/whitespace-
separated - a single id is also accepted). One bot can post to many chats; each
destination just needs its chat_id listed. A private user must have started the
bot first, the bot must be a member of a group, and an admin of a channel. All
recipients are always attempted: if one cannot be reached the others still get
the message, and a summary error is raised afterwards so cron runs fail loudly.

Consumers (all stdlib-only, no third-party deps):
  - scripts/machine-monitor.py   debounced high-CPU-temperature alert
  - scripts/machine-status.py    daily machine health report
  - firewatch/firewatch.py       fire/smoke alert photos (the firewatch container
                                 mounts ./scripts read-only at /scripts:ro, so
                                 this module ships with the script)
"""
import html
import json
import os
import re
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


def parse_recipients(raw):
    """Split a raw CHAT_ID value into an ordered list of unique chat ids.

    Accepts a single id (legacy) or several separated by commas and/or
    whitespace, e.g. '7844679766,-5402647496' - a mix of private user ids,
    group ids, channel ids and @usernames. Duplicates are dropped.
    """
    return list(
        dict.fromkeys(
            part.strip()
            for part in re.split(r"[,\s]+", raw.strip())
            if part.strip()
        )
    )


def ensure_creds(cfg):
    """Return (token, chat_ids) or raise RuntimeError with a clear message.

    chat_ids is the ordered list of recipients parsed from the (comma/whitespace-
    separated) CHAT_ID value. At least one recipient is required.
    """
    token = cfg.get("BOT_TOKEN", "").strip()
    chat_ids = parse_recipients(cfg.get("CHAT_ID", ""))
    if not token or token == "CHANGE_ME":
        raise RuntimeError(
            f"Telegram not configured - check {DEFAULT_CONF} "
            "(create it from telegram.conf.example)"
        )
    if not chat_ids:
        raise RuntimeError(
            f"Telegram not configured - no chat id in {DEFAULT_CONF} "
            "(CHAT_ID may hold several comma-separated ids)"
        )
    return token, chat_ids


def esc_html(text):
    """HTML-escape &, < and > so dynamic values are safe in html parse mode."""
    return html.escape(str(text), quote=False)


def _post_message(token, chat_id, text, parse_mode):
    """Post `text` to ONE chat via sendMessage. Raises on transport/API errors."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    ).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.load(response)
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body.get('description')}")


def send_telegram(cfg, text, parse_mode="html"):
    """Post `text` to every configured chat/group.

    All recipients are always attempted; if any fail, a summary RuntimeError
    naming the failed recipient(s) is raised after the loop.
    """
    token, chat_ids = ensure_creds(cfg)
    failures = []
    for chat_id in chat_ids:
        try:
            _post_message(token, chat_id, text, parse_mode)
        except Exception as exc:  # noqa: BLE001 - collect per-recipient errors
            failures.append(f"{chat_id}: {exc}")
    if failures:
        raise RuntimeError(
            "Telegram send failed for recipient(s): " + "; ".join(failures)
        )


def _post_photo(token, chat_id, photo_bytes, caption, parse_mode):
    """Post a photo (JPEG bytes) with an optional caption to ONE chat via the
    Bot API sendPhoto method (multipart/form-data, stdlib only).

    Per-recipient worker used by send_telegram_photo().
    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    boundary = f"----firewatch{int(__import__('time').time() * 1000)}"
    fields = [
        ("chat_id", str(chat_id)),
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


def send_telegram_photo(cfg, photo_bytes, caption="", parse_mode="html"):
    """Post a photo (JPEG bytes) with an optional caption to EVERY configured
    chat via the Bot API sendPhoto method.

    Used by the fire-watch watcher (firewatch/firewatch.py) to send the alert
    snapshot together with the detection caption. All recipients are always
    attempted; if any fail, a summary RuntimeError naming the failed
    recipient(s) is raised after the loop.
    """
    token, chat_ids = ensure_creds(cfg)
    failures = []
    for chat_id in chat_ids:
        try:
            _post_photo(token, chat_id, photo_bytes, caption, parse_mode)
        except Exception as exc:  # noqa: BLE001 - collect per-recipient errors
            failures.append(f"{chat_id}: {exc}")
    if failures:
        raise RuntimeError(
            "Telegram photo send failed for recipient(s): "
            + "; ".join(failures)
        )
