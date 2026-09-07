#!/usr/bin/env python3
"""telegram_bot.py - on-demand command responder for the farm Telegram bot.

A small, dependency-free (stdlib-only) **long-poll getUpdates loop** that answers
commands (/status, /start, /help) IN THE CHAT THAT ASKED, so you can query the
host live from Telegram instead of waiting for the daily cron report or an alert.

It runs as the `telegram-bot` docker compose service (see docker-compose.yml):
a dependency-free python:3.11-slim image with ./scripts + ./config mounted
read-only (code/config edits need only `docker compose restart telegram-bot`),
plus read-only HOST bind mounts so /status reports real host health:
  - /proc            -> $HOST_PROC    (uptime, load, meminfo, hostname)
  - /sys/class/hwmon -> $HOST_HWMON   (coretemp -> CPU temp)
  - ./media          -> $MEDIA_DIR    (media filesystem usage)
  - ./config         -> $ROOT_STAT    (config mount sits on the host root fs)
and the Frigate REST API ($FRIGATE_API, e.g. http://frigate:5000). The bot
container NEVER writes to the host - every mount is read-only.

Commands (use /cmd@mazr3a_garden_bot in a group with several bots):
  /status   live host + Frigate health report (CPU temp, load, memory, disk,
            cameras) - mirrors scripts/machine-status.py
  /help     this help text
  /start    greeting

Auth: a message is answered only when it is authorized. If ALLOWED_USER_IDS is
set in config/telegram.conf, only those user ids may trigger commands (in any
chat). Otherwise the message's chat must be one of the configured CHAT_ID
recipients (the owner's private chat and the farm group by default).

Telegram creds come from TELEGRAM_CONF (default config/telegram.conf,
git-ignored). Long-polling getUpdates must be the ONLY consumer of the bot's
updates (no webhook, no manual curl loops) - the cron scripts and firewatch
only SEND, so this is safe. A second consumer would return HTTP 409 conflicts.
"""
import datetime
import json
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import telegram_notify as tg  # noqa: E402

FRIGATE_API = os.environ.get("FRIGATE_API", "http://frigate:5000").rstrip("/")
HOST_PROC = os.environ.get("HOST_PROC", "/host/proc")
HOST_HWMON = os.environ.get("HOST_HWMON", "/host/hwmon")
MEDIA_DIR = os.environ.get("MEDIA_DIR", "/media")
ROOT_STAT = os.environ.get("ROOT_STAT", "/config")

# getUpdates long-poll timing. The Bot API documents `timeout` as
# "0..50" seconds - 50 is therefore the MAXIMUM the server will hold a
# poll open. There is no higher value that reduces reconnects further;
# raising it above 50 is ignored by Telegram. The local socket timeout
# must EXCEED the server timeout so urllib never cuts a poll short -
# keep ~25 s of headroom for the TCP/TLS handshake + server reply.
LONG_POLL_TIMEOUT_S = 50              # max Telegram accepts for getUpdates
POLL_SOCKET_TIMEOUT_S = LONG_POLL_TIMEOUT_S + 25


# ---------------------------------------------------------------------------
# host health probes (best-effort; every failure degrades to n/a)
# ---------------------------------------------------------------------------
def _read_first(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return None


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}"
        value /= 1024.0
    return f"{value:.0f}P"


def host_name():
    hn = _read_first(os.path.join(HOST_PROC, "sys", "kernel", "hostname"))
    return (hn or socket.gethostname() or "host").strip()


def format_uptime():
    raw = _read_first(os.path.join(HOST_PROC, "uptime"))
    if not raw:
        return "n/a"
    try:
        seconds = float(raw.split()[0])
    except (ValueError, IndexError):
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


def load_summary():
    raw = _read_first(os.path.join(HOST_PROC, "loadavg"))
    if not raw:
        return "n/a"
    vals = raw.split()
    return f"{vals[0]} / {vals[1]} / {vals[2]}" if len(vals) >= 3 else "n/a"


def memory_summary():
    total_kb = available_kb = None
    try:
        with open(os.path.join(HOST_PROC, "meminfo"), encoding="utf-8") as fh:
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


def cpu_temp_max(hwmon_root=HOST_HWMON):
    """Hottest live coretemp reading in °C (int) or None when unavailable."""
    try:
        dirs = os.listdir(hwmon_root)
    except Exception:
        return None
    best = None
    for entry in dirs:
        d = os.path.join(hwmon_root, entry)
        try:
            with open(os.path.join(d, "name"), encoding="utf-8") as fh:
                name = fh.read().strip()
        except Exception:
            continue
        if "coretemp" not in name:
            continue
        try:
            inputs = os.listdir(d)
        except Exception:
            continue
        for fname in inputs:
            if not fname.startswith("temp") or not fname.endswith("_input"):
                continue
            try:
                with open(os.path.join(d, fname), encoding="utf-8") as fh:
                    celsius = int(fh.read().strip()) / 1000.0
            except Exception:
                continue
            if best is None or celsius > best:
                best = celsius
    return int(round(best)) if best is not None else None


def frigate_lines():
    """Frigate detection + per-camera online/offline (same as machine-status)."""
    try:
        with urllib.request.urlopen(f"{FRIGATE_API}/api/stats", timeout=6) as resp:
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
        det_txt += "; inference {} ms".format(" / ".join(str(i) for i in inference))
    lines = [f"🎞️ Frigate: {tg.esc_html(det_txt)}"]
    if cams:
        lines.append("📷 Cameras:")
        for name in sorted(cams):
            fps = cams[name].get("camera_fps") or 0
            label = f"✅ online ({fps:.1f} fps)" if fps > 0 else "❌ offline"
            lines.append(f"   • {tg.esc_html(name)} {label}")
    return lines


def build_status():
    now = datetime.datetime.now().astimezone()
    lines = []
    add = lines.append
    add(f"<b>🖥️ {tg.esc_html(host_name())} - live status</b>")
    add(f"📅 {now.strftime('%Y-%m-%d %H:%M %Z')}")
    add(f"⏱ Uptime: {tg.esc_html(format_uptime())}")
    temp = cpu_temp_max()
    add(f"🌡 CPU temp (max): {temp}°C" if temp is not None else "🌡 CPU temp (max): n/a")
    add(f"⚙️ Load 1/5/15m: {tg.esc_html(load_summary())}")
    add(f"🧠 Memory used: {tg.esc_html(memory_summary())}")
    add(f"💾 Disk /: {tg.esc_html(disk_summary(ROOT_STAT))}")
    add(f"🎥 Media dir: {tg.esc_html(disk_summary(MEDIA_DIR))}")
    lines.extend(frigate_lines())
    return "\n".join(lines)


def help_text():
    return "\n".join(
        [
            "<b>Mazr3a farm bot</b> 🤖🌾",
            "",
            "Commands:",
            "  /status - live host + Frigate report",
            "  /help   - this help",
            "  /start  - greeting",
            "",
            "In a group with several bots use",
            "/status@mazr3a_garden_bot",
        ]
    )


def start_text():
    return "\n".join(
        [
            "Hello! I post farm alerts (fire/smoke photos, CPU-temperature",
            "watchdog, daily machine report) and can answer on demand.",
            "",
            "Try /status for a live host report, or /help.",
        ]
    )


# ---------------------------------------------------------------------------
# command dispatch
# ---------------------------------------------------------------------------
def authorized(cfg, msg):
    """Only answer senders/chats we trust (see module docstring)."""
    from_id = str(msg.get("from", {}).get("id", ""))
    allowed_users = tg.parse_recipients(cfg.get("ALLOWED_USER_IDS", ""))
    if allowed_users:
        return from_id in allowed_users
    chat_id = str(msg.get("chat", {}).get("id", ""))
    recipients = tg.parse_recipients(cfg.get("CHAT_ID", ""))
    return chat_id in recipients


def handle(cfg, token, msg):
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        return
    cmd = text.split()[0].lower()
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]  # /status@botname -> /status
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if not chat_id or not authorized(cfg, msg):
        return
    if cmd == "/status":
        reply = build_status()
    elif cmd == "/help":
        reply = help_text()
    elif cmd == "/start":
        reply = start_text()
    else:
        return  # unknown command: stay quiet
    tg.send_message_to(cfg, chat_id, reply)
    print(f"[telegram-bot] answered '{cmd}' to chat {chat_id}", flush=True)


# ---------------------------------------------------------------------------
# long-poll getUpdates loop
# ---------------------------------------------------------------------------
def get_updates(token, offset):
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "timeout": str(LONG_POLL_TIMEOUT_S),  # 50 s = Telegram's documented max
        "allowed_updates": json.dumps(["message"]),
    }
    if offset:
        params["offset"] = str(offset)
    data = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(request, timeout=POLL_SOCKET_TIMEOUT_S) as response:
        body = json.load(response)
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body.get('description')}")
    return body.get("result", [])


def run(cfg, token):
    print("[telegram-bot] listening for commands ...", flush=True)
    offset = None
    while True:
        try:
            updates = get_updates(token, offset)
            for update in updates:
                # Confirm each update (else Telegram re-delivers after restart).
                offset = int(update.get("update_id", 0)) + 1
                msg = update.get("message") or {}
                if not msg.get("text"):
                    continue
                try:
                    handle(cfg, token, msg)
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    print(f"[telegram-bot] handler error: {exc}", flush=True)
        except urllib.error.HTTPError as exc:
            # 409 = another getUpdates consumer (manual curl? second instance?)
            print(f"[telegram-bot] HTTP {exc.code} - backing off", flush=True)
            time.sleep(10)
        except Exception as exc:  # noqa: BLE001 - network hiccups; retry
            print(f"[telegram-bot] poll error: {exc}", flush=True)
            time.sleep(5)


def main():
    cfg = tg.load_conf()  # honors TELEGRAM_CONF (default config/telegram.conf)
    token = (cfg.get("BOT_TOKEN") or "").strip()
    if not token or token == "CHANGE_ME":
        print(
            "[telegram-bot] ERROR: BOT_TOKEN missing/CHANGE_ME - check "
            f"{os.environ.get('TELEGRAM_CONF', 'config/telegram.conf')}",
            file=sys.stderr,
        )
        return 2
    run(cfg, token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
