"""portal.conf loader + typed getters.

portal.conf is an INI-style KEY=VALUE file (like firewatch.conf/telegram.conf)
with # comments, plus repeated `user` lines:

    user <name> <pbkdf2_hash> [default_camera=<cam>]

Generate a hash line with:  python portal/genpass.py --username NAME

Every option can be overridden by a same-named environment variable prefixed
PORTAL_ (e.g. PORTAL_FRIGATE_API), which wins. The real file lives at
/config/portal.conf (git-ignored); a committed example documents every option
in config/portal.conf.example.
"""
import os

# Keys are uppercase (config file keys are upper-cased on load).
DEFAULTS = {
    "SECRET_KEY": "",                 # signs session cookies (must be set)
    # Runtime user accounts live in the portal's OWN SQLite DB (portal/
    # userstore.py) so add/edit/delete applies with NO container restart. The
    # `user` lines in portal.conf are only a first-run SEED (see parse_file).
    "PORTAL_USERS_DB": "/media/portal/users.db",
    # Notification feed (portal/notifstore.py) - the portal's OWN DB under the
    # same rw ./media mount, holding the notification timeline + per-user read
    # markers. The background watcher records one notification per NEW Firewatch
    # alert (NOTIFY_POLL_S cadence) and rows are pruned in-app to
    # NOTIFY_RETENTION_DAYS (the host heartbeat hard-protects *.db, so it can
    # never prune rows for us).
    "PORTAL_NOTIF_DB": "/media/portal/notifications.db",
    "NOTIFY_POLL_S": "30",            # fire-alert -> notification watcher cadence
    "NOTIFY_RETENTION_DAYS": "90",    # notification history kept (in-app prune)
    "SESSION_DAYS": "7",              # cookie lifetime in days
    "COOKIE_SECURE": "true",          # Secure flag on the session cookie
    "STREAM_IDLE_TIMEOUT_S": "300",   # live-view idle time watch default
    "DEFAULT_CAMERA": "",             # optional landing camera override
    "FRIGATE_API": "http://frigate:5000",
    # Live view is served SAME-ORIGIN as HLS through the portal:
    # /api/live/<cam>/hls/stream.m3u8 -> Frigate go2rtc HLS (master + .ts
    # segments) via Frigate's /api/go2rtc/* reverse proxy. No public stream
    # URL template is needed: the SPA (hls.js) fetches its own origin, which
    # stays behind the session cookie and works through the Cloudflare Tunnel
    # without extra path mappings.
    # firewatch evidence: DB path + stored-jpg prefix remap. The host ./media
    # tree is mounted at /media here; firewatch sees it at /media/firewatch
    # (STORE_DIR), so stored paths /media/firewatch/<cam>/x.jpg -> /media/<cam>/x.jpg.
    "FIREWATCH_DB": "/media/firewatch.db",
    "FIREWATCH_JPG_PREFIX": "/media/firewatch/",
    "FIREWATCH_JPG_REPLACE": "/media/",
    # Admin Debug tab: read-only Docker-logs sidecar (compose service `logs`).
    # Reached only server-side on the internal compose network; overridden by
    # the PORTAL_LOGS_API env set in docker-compose.yml.
    "LOGS_API": "http://logs:8090",
}


def _env_override(key, value):
    """Allow PORTAL_<KEY> env to win (matches how firewatch/telegram confs work)."""
    return os.environ.get("PORTAL_" + key.upper(), value)


def parse_file(path):
    """Return (opts dict, users list) parsed from a portal.conf file."""
    opts = {}
    users = []
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("user "):
                    parts = line.split(None, 2)
                    if len(parts) == 3:
                        name, rest = parts[1], parts[2]
                        fields = rest.split()
                        phash = fields[0] if fields else ""
                        default_camera = ""
                        for tok in fields[1:]:
                            if tok.lower().startswith("default_camera="):
                                default_camera = tok.split("=", 1)[1]
                        users.append({"username": name,
                                      "password_hash": phash,
                                      "default_camera": default_camera})
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    opts[key.strip().upper()] = val.strip()
    # Merge defaults + env overrides for every known option.
    merged = {}
    for key, default in DEFAULTS.items():
        merged[key] = _env_override(key, opts.get(key.upper(), default))
    for key in opts:  # any extra keys not in DEFAULTS
        if key not in merged:
            merged[key] = _env_override(key, opts[key])
    return merged, users


def get(cfg, key, default=None):
    """Case-insensitive value from cfg, falling back to defaults."""
    val = cfg.get(key.upper())
    if val is None:
        val = DEFAULTS.get(key.upper(), default)
    return val


def geti(cfg, key, default=0):
    try:
        return int(get(cfg, key, default))
    except (TypeError, ValueError):
        return default


def getb(cfg, key, default=False):
    val = str(get(cfg, key, "true" if default else "false")).strip().lower()
    return val in ("1", "true", "yes", "on")
