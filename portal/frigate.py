"""Small typed wrappers over Frigate's REST API (httpx, async).

Frigate 0.17 endpoints used: /api/config, /api/stats, /api/events and the event
media endpoints (/api/events/<id>/snapshot.jpg, /api/events/<id>/clip.mp4).
Camera RTSP credentials never leave Frigate - only these public REST routes are
reached, always server-side from the portal container over the compose network.
"""
import re

_CAMERA_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def valid_camera_name(name):
    return bool(name) and bool(_CAMERA_RE.match(name))


async def fetch_json(client, base, path, params=None, timeout=8.0):
    """GET a Frigate JSON endpoint, raising HTTPStatusError on failure."""
    resp = await client.get(base.rstrip("/") + path, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


async def camera_list(client, base):
    """Return [{name, enabled, online}] from /api/config + /api/stats.

    Best-effort: a stats failure only drops the online flag, never the list.
    """
    cfg = await fetch_json(client, base, "/api/config", timeout=8.0)
    cameras = cfg.get("cameras") or {}
    names = sorted(cameras.keys())

    online = {}
    try:
        stats = await fetch_json(client, base, "/api/stats", timeout=5.0)
        for name, data in (stats.get("cameras") or {}).items():
            online[name] = bool(data.get("camera_fps"))
    except Exception:
        pass  # online flags stay unknown; not fatal

    items = []
    for name in names:
        cam = cameras.get(name) or {}
        detect = cam.get("detect") or {}
        # detect.enabled is the field that gates live detection/stream; treat a
        # missing key as enabled (the detect block may serialize differently).
        enabled = detect.get("enabled", True)
        items.append({
            "name": name,
            "enabled": bool(enabled),
            "online": bool(online.get(name)),
        })
    return items
