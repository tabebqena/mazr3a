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
    "SESSION_DAYS": "7",              # cookie lifetime in days
    "COOKIE_SECURE": "true",          # Secure flag on the session cookie
    "STREAM_IDLE_TIMEOUT_S": "300",   # live-view idle time watch default
    "DEFAULT_CAMERA": "",             # optional landing camera override
    "FRIGATE_API": "http://frigate:5000",
    # Live MSE is proxied SAME-ORIGIN through the portal:
    # ws(s)://<portal>/api/live/<cam>/mse  ->  Frigate go2rtc MSE
    # (.../live/mse/api/ws?src=<cam>). No public stream URL template is needed:
    # the SPA connects to its own origin, which stays behind the session cookie
    # and works through the Cloudflare Tunnel without extra path mappings.
    # firewatch evidence: DB path + stored-jpg prefix remap. The host ./media
    # tree is mounted at /media here; firewatch sees it at /media/firewatch
    # (STORE_DIR), so stored paths /media/firewatch/<cam>/x.jpg -> /media/<cam>/x.jpg.
    "FIREWATCH_DB": "/media/firewatch.db",
    "FIREWATCH_JPG_PREFIX": "/media/firewatch/",
    "FIREWATCH_JPG_REPLACE": "/media/",
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
