"""ONVIF PTZ autodetection for the Frigate camera list.

The portal probes every camera host found in Frigate's go2rtc streams. A camera
is considered PTZ when its ONVIF profiles contain a PTZConfiguration and the
PTZ service exposes a node. The result is cached in a portal-owned JSON file so
startup does not repeat the probe when the file already exists.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from portal import frigate
from portal import onvif

DEFAULT_DETECTED_FILE = "/media/portal/ptz_detected.json"
DEFAULT_TIMEOUT_S = 2.0


class PTZDetectError(RuntimeError):
    """PTZ autodetection could not complete."""


def _camera_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _onvif_credentials_for(camera: str) -> tuple[str, str]:
    key = _camera_key(camera)
    return (
        _env_first(f"PTZ_ONVIF_USER_{key}", "PTZ_ONVIF_USER", "ONVIF_USER"),
        _env_first(f"PTZ_ONVIF_PASS_{key}", "PTZ_ONVIF_PASS", "ONVIF_PASS"),
    )


def _host_from_rtsp(url) -> str | None:
    if isinstance(url, list):
        url = next((item for item in url if isinstance(item, str)), None)
    if not isinstance(url, str) or "://" not in url or url.startswith("ffmpeg:"):
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host or host in ("127.0.0.1", "localhost"):
        return None
    return host


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def load_detected(path: str) -> dict:
    empty = {"version": 1, "detected_at": None, "cameras": {}}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return empty
    except Exception as exc:
        print(f"WARNING: cannot read PTZ detection file {path}: {exc}",
              flush=True)
        return empty
    cameras = raw.get("cameras") if isinstance(raw, dict) else None
    if not isinstance(cameras, dict):
        return empty
    clean = {}
    for name, info in cameras.items():
        if not isinstance(info, dict):
            continue
        host = str(info.get("host") or "").strip()
        if not host:
            continue
        clean[str(name)] = {
            "name": str(info.get("name") or name),
            "host": host,
            "profile": str(info.get("profile") or "Profile_1"),
            "profile_name": str(info.get("profile_name") or ""),
            "node": str(info.get("node") or ""),
        }
    return {
        "version": 1,
        "detected_at": raw.get("detected_at"),
        "cameras": clean,
    }


def save_detected(path: str, cameras: dict, detected_at: str | None = None) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "detected_at": detected_at or _now_iso(),
        "cameras": cameras,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=".ptz_detected.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return payload


async def detect_ptz_cameras(
    cfg: dict,
    http_client,
    frigate_base: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    timeout_s = max(0.5, float(timeout_s))
    try:
        config = await frigate.fetch_json(
            http_client, frigate_base, "/api/config", timeout=timeout_s)
    except Exception as exc:
        raise PTZDetectError(f"cannot read Frigate config: {exc}") from exc

    cameras = config.get("cameras") or {}
    streams = (config.get("go2rtc") or {}).get("streams") or {}
    hosts = {}
    for camera in cameras:
        host = _host_from_rtsp(streams.get(camera))
        if host:
            hosts[str(camera)] = host
    if not hosts:
        return {
            "cameras": {},
            "stats": {"probed": 0, "responded": 0, "errors": []},
        }

    async def probe(camera: str, host: str):
        user, password = _onvif_credentials_for(camera)
        if not user or not password:
            return camera, host, None, "ONVIF credentials missing"
        client = onvif.ONVIFProbe(
            host, user, password, timeout=timeout_s)
        try:
            return camera, host, await client.detect(), None
        except Exception as exc:
            return camera, host, None, str(exc)
        finally:
            await client.aclose()

    results = await asyncio.gather(
        *(probe(camera, host) for camera, host in hosts.items()))

    detected = {}
    errors = []
    responded = 0
    for camera, host, result, error in results:
        if error:
            errors.append(f"{camera} ({host}): {error}")
            continue
        responded += 1
        if result:
            detected[camera] = {
                "name": camera,
                "host": host,
                "profile": result.get("profile") or "Profile_1",
                "profile_name": result.get("profile_name") or "",
                "node": result.get("node") or "",
            }

    if responded == 0:
        detail = "; ".join(errors[:4]) or "no camera responded"
        raise PTZDetectError(f"no camera responded to ONVIF probe: {detail}")

    return {
        "cameras": detected,
        "stats": {
            "probed": len(hosts),
            "responded": responded,
            "errors": errors,
        },
    }


async def detect_and_store(
    cfg: dict,
    http_client,
    frigate_base: str,
    path: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    result = await detect_ptz_cameras(
        cfg, http_client, frigate_base, timeout_s=timeout_s)
    payload = save_detected(path, result["cameras"])
    payload["stats"] = result["stats"]
    return payload
